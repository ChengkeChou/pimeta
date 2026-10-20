"""Tests for the command-line interface (``pimeta`` / ``python -m pimeta``)."""

from __future__ import annotations

import csv
import json

import pytest

from pimeta.cli import build_parser, main, read_table

STUDIES = {
    "study": ["S1", "S2", "S3", "S4", "S5", "S6"],
    "yi": [0.20, 0.40, 0.10, 0.50, 0.30, 0.15],
    "vi": [0.04, 0.05, 0.03, 0.06, 0.02, 0.05],
    "sei": [0.20, 0.2236, 0.1732, 0.2449, 0.1414, 0.2236],
    "dose": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
}


def _write_csv(path, columns):
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        keys = list(columns)
        writer.writerow(keys)
        for i in range(len(columns[keys[0]])):
            writer.writerow([columns[k][i] for k in keys])
    return str(path)


@pytest.fixture()
def studies(tmp_path):
    return _write_csv(tmp_path / "studies.csv", STUDIES)


# --------------------------------------------------------------------------- #
# parser and reading
# --------------------------------------------------------------------------- #
def test_parser_has_the_three_commands():
    parser = build_parser()
    actions = [a for a in parser._actions if a.dest == "command"]
    assert actions[0].choices.keys() == {"fit", "escalc", "network"}


def test_read_table_csv_and_json(tmp_path):
    path = _write_csv(tmp_path / "a.csv", {"x": [1, 2], "y": [3, 4]})
    cols = read_table(path)
    assert cols["x"] == ["1", "2"]

    js = tmp_path / "b.json"
    js.write_text(json.dumps([{"x": 1, "y": 2}, {"x": 3, "y": 4}]), encoding="utf-8")
    assert read_table(str(js))["x"] == [1, 3]

    jl = tmp_path / "c.jsonl"
    jl.write_text('{"x": 1}\n{"x": 2}\n', encoding="utf-8")
    assert read_table(str(jl))["x"] == [1, 2]


def test_read_table_missing_file():
    with pytest.raises(FileNotFoundError):
        read_table("does-not-exist.csv")


# --------------------------------------------------------------------------- #
# fit
# --------------------------------------------------------------------------- #
def test_fit_prints_a_coefficient_table(studies, capsys):
    assert main(["fit", studies, "--yi", "yi", "--vi", "vi"]) == 0
    out = capsys.readouterr().out
    assert "term" in out and "intrcpt" in out
    assert "estimate" in out and "std.error" in out


def test_fit_with_formula_moderator(studies, capsys):
    code = main(["fit", studies, "--yi", "yi", "--vi", "vi", "--mods", "~ dose"])
    assert code == 0
    out = capsys.readouterr().out
    assert "intrcpt" in out and "dose" in out


def test_fit_accepts_sei(studies, capsys):
    assert main(["fit", studies, "--yi", "yi", "--sei", "sei"]) == 0
    assert "intrcpt" in capsys.readouterr().out


def test_fit_requires_vi_or_sei(studies, capsys):
    assert main(["fit", studies, "--yi", "yi"]) == 2
    assert "--vi" in capsys.readouterr().err


def test_fit_json_output_is_valid(studies, capsys):
    assert main(["fit", studies, "--yi", "yi", "--vi", "vi", "--format", "json"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert "estimate" in payload


def test_fit_writes_csv(tmp_path, studies, capsys):
    out = tmp_path / "coef.csv"
    assert main(["fit", studies, "--yi", "yi", "--vi", "vi", "-o", str(out)]) == 0
    assert out.exists()
    header = out.read_text(encoding="utf-8").splitlines()[0]
    assert header.startswith("term,")
    # The model-level row is written alongside.
    assert (tmp_path / "coef_glance.csv").exists()


def test_fit_forest_plot(tmp_path, studies, capsys):
    png = tmp_path / "forest.png"
    assert main(["fit", studies, "--yi", "yi", "--vi", "vi",
                 "--forest", str(png)]) == 0
    assert png.exists() and png.stat().st_size > 0


def test_fit_unknown_column(studies, capsys):
    assert main(["fit", studies, "--yi", "nope", "--vi", "vi"]) == 2
    err = capsys.readouterr().err
    assert "not found" in err and "available columns" in err


def test_fit_non_numeric_column(tmp_path, capsys):
    path = _write_csv(tmp_path / "bad.csv", {"yi": ["a"], "vi": ["0.1"]})
    assert main(["fit", path, "--yi", "yi", "--vi", "vi"]) == 2
    assert "non-numeric" in capsys.readouterr().err


def test_fit_warns_about_missing_values(tmp_path, capsys, recwarn):
    data = dict(STUDIES, yi=[0.20, 0.40, "", 0.50, 0.30, 0.15])
    path = _write_csv(tmp_path / "gaps.csv", data)
    assert main(["fit", path, "--yi", "yi", "--vi", "vi"]) == 0
    messages = [str(w.message) for w in recwarn.list]
    assert any("dropped 1 of 6 studies" in m for m in messages)
    assert "intrcpt" in capsys.readouterr().out


def test_fit_no_warn_missing_silences(tmp_path, capsys, recwarn):
    data = dict(STUDIES, yi=[0.20, 0.40, "", 0.50, 0.30, 0.15])
    path = _write_csv(tmp_path / "gaps2.csv", data)
    assert main(["fit", path, "--yi", "yi", "--vi", "vi",
                 "--no-warn-missing"]) == 0
    assert not [w for w in recwarn.list if "dropped" in str(w.message)]


# --------------------------------------------------------------------------- #
# escalc
# --------------------------------------------------------------------------- #
def test_escalc_binary(tmp_path, capsys):
    path = _write_csv(tmp_path / "raw.csv", {
        "ai": [3, 6, 5], "bi": [52, 48, 55], "ci": [8, 10, 9], "di": [47, 42, 51],
    })
    code = main(["escalc", path, "--measure", "RR",
                 "--ai", "ai", "--bi", "bi", "--ci", "ci", "--di", "di"])
    assert code == 0
    lines = capsys.readouterr().out.strip().splitlines()
    assert lines[0] == "yi\tvi"
    assert len(lines) == 4


def test_escalc_writes_file(tmp_path, capsys):
    path = _write_csv(tmp_path / "raw2.csv", {
        "ai": [3, 6], "bi": [52, 48], "ci": [8, 10], "di": [47, 42],
    })
    out = tmp_path / "effects.csv"
    code = main(["escalc", path, "--measure", "OR",
                 "--ai", "ai", "--bi", "bi", "--ci", "ci", "--di", "di",
                 "-o", str(out)])
    assert code == 0
    assert out.exists()
    assert out.read_text(encoding="utf-8").startswith("yi,vi")


# --------------------------------------------------------------------------- #
# network
# --------------------------------------------------------------------------- #
def test_network_league(tmp_path, capsys):
    te = [0.10, -0.20, 0.30, 0.15, -0.05, 0.25]
    se = [0.10, 0.12, 0.11, 0.09, 0.13, 0.10]
    path = _write_csv(tmp_path / "nma.csv", {
        "TE": te, "seTE": se,
        "t1": ["A", "A", "B", "B", "C", "A"],
        "t2": ["B", "C", "C", "D", "D", "D"],
        "studlab": [f"S{i + 1}" for i in range(6)],
    })
    code = main(["network", path, "--te", "TE", "--se", "seTE",
                 "--treat1", "t1", "--treat2", "t2",
                 "--studlab", "studlab", "--sm", "MD"])
    assert code == 0
    lines = capsys.readouterr().out.strip().splitlines()
    assert lines[0].startswith("treatment\t")
    assert len(lines) == 5           # header + four treatments


def test_network_missing_treatment_column(tmp_path, capsys):
    path = _write_csv(tmp_path / "nma2.csv", {
        "TE": [0.1, 0.2], "seTE": [0.1, 0.1], "t1": ["A", "B"], "t2": ["B", "C"],
    })
    code = main(["network", path, "--te", "TE", "--se", "seTE",
                 "--treat1", "t1", "--treat2", "nope"])
    assert code == 2
    assert "not found" in capsys.readouterr().err


# --------------------------------------------------------------------------- #
# module entry point
# --------------------------------------------------------------------------- #
def test_python_dash_m_runs(tmp_path):
    import subprocess
    import sys

    path = _write_csv(tmp_path / "s.csv", STUDIES)
    proc = subprocess.run(
        [sys.executable, "-m", "pimeta", "fit", path, "--yi", "yi", "--vi", "vi"],
        capture_output=True, text=True, timeout=120,
    )
    assert proc.returncode == 0, proc.stderr
    assert "intrcpt" in proc.stdout


def test_version_flag():
    import subprocess
    import sys

    proc = subprocess.run([sys.executable, "-m", "pimeta", "--version"],
                          capture_output=True, text=True, timeout=120)
    assert proc.returncode == 0
    assert "pimeta" in proc.stdout
