"""The specification report: one table for the whole grid, failures included.

The defect these tests guard against is the one the module exists for: `curve()`
is already the subset that worked, so a report assembled from it by hand silently
drops the specifications that broke. Every row of the grid therefore has to be
present, with its status, and the caveats have to travel with the table rather
than living in a guide nobody pastes.
"""

from __future__ import annotations

import json

import matplotlib

matplotlib.use("Agg")

import numpy as np  # noqa: E402
import pytest  # noqa: E402

import pimeta as ma  # noqa: E402


def _spec():
    return ma.AnalysisSpec(decisions=(
        ma.Decision(name="method", values=("REML", "DL"), default="REML",
                    reasons={"REML": "a", "DL": "b"}),
        ma.Decision(name="test", values=("z", "knha"), default="z",
                    reasons={"z": "a", "knha": "b"}),
    ))


YI = np.array([0.15, 0.42, 0.28, 0.61, 0.19, 0.35])
SEI = np.array([0.12, 0.20, 0.15, 0.24, 0.18, 0.16])


def _clean(choices):
    return ma.rma(YI, sei=SEI, **choices)


def test_every_specification_appears_exactly_once():
    mv = ma.multiverse(_spec(), _clean)
    report = ma.specification_report(mv)
    assert len(report) == 4
    assert [row["id"] for row in report.rows] == [o.id for o in mv.outcomes]
    assert len({row["id"] for row in report.rows}) == 4


def test_failed_specifications_are_rows_not_omissions():
    def run(choices):
        if choices["method"] == "DL":
            raise RuntimeError("no convergence")
        return _clean(choices)

    mv = ma.multiverse(_spec(), run)
    report = ma.specification_report(mv)
    assert len(report) == 4
    failed = [row for row in report.rows if row["status"] == "failed"]
    assert len(failed) == 2
    text = report.markdown
    assert "no convergence" in text
    assert failed[0]["id"][:12] in text


class Plain:
    """A fitted result with no diagnostics attribute at all."""

    def __init__(self, est):
        self.b = [est]
        self.ci_lb = [est - 0.1]
        self.ci_ub = [est + 0.1]


def test_excluded_specifications_show_the_codes_that_excluded_them():
    class Diag:
        worst = "error"

        def codes(self):
            return ["not_converged"]

    class Fitted(Plain):
        diagnostics = Diag()

    def run(choices):
        if choices["method"] == "DL":
            return Fitted(-0.4)
        return Plain(0.3)          # no `diagnostics` attribute: not flagged

    mv = ma.multiverse(_spec(), run)
    report = ma.specification_report(mv)
    excluded = [row for row in report.rows if row["status"] == "excluded"]
    assert len(excluded) == 2
    assert excluded[0]["codes"] == ["not_converged"]
    assert "excluded: not_converged" in report.markdown
    assert report.excluded == 2
    # the excluded numbers must not appear anywhere in the table
    assert "-0.4000" not in report.markdown


def test_the_caveats_are_part_of_the_report_not_the_guide():
    report = ma.specification_report(_mv_clean(), title="Pooled effect")
    text = report.markdown
    assert text.startswith("# Pooled effect")
    assert "**What this table does not say**" in text
    assert "not a probability that the effect is real" in text
    assert "one-way decomposition" in text
    assert all(note in text for note in report.notes)


def test_the_counts_line_reconciles_with_the_rows():
    # The exact numbers depend on the fixture and on which checks fire, so the
    # assertion is that the sentence agrees with the rows rather than a literal.
    report = ma.specification_report(_mv_clean())
    line = [ln for ln in report.markdown.splitlines() if "specifications ran" in ln]
    assert len(line) == 1
    counts = report.counts
    assert (f"{counts['run']} of {counts['considered']} specifications ran"
            in line[0])
    assert f"{counts['usable']} usable" in line[0]
    assert f"{counts['failed']} failed" in line[0]
    assert f"{counts['flagged']} flagged" in line[0]
    assert f"({report.excluded} excluded" in line[0]
    assert counts["usable"] + counts["failed"] + report.excluded == counts["run"]


def test_a_flagged_but_kept_row_says_so_beside_its_number():
    class Diag:
        worst = "warning"

        def codes(self):
            return ["few_studies"]

    class Fitted:
        def __init__(self, est):
            self.b = [est]
            self.ci_lb = [est - 0.1]
            self.ci_ub = [est + 0.1]
            self.diagnostics = Diag()

    mv = ma.multiverse(_spec(), lambda c: Fitted(0.3))
    text = ma.specification_report(mv).markdown
    assert "ok (few_studies)" in text
    assert "+0.3000" in text


def test_the_export_round_trips_and_names_the_provenance():
    report = ma.specification_report(_mv_clean(), title="T",
                                     provenance="bcg data, 2026-10-26")
    payload = json.loads(json.dumps(report.to_dict()))
    assert payload["title"] == "T"
    assert "2026-10-26" in payload["provenance"]
    assert len(payload["rows"]) == 4
    assert payload["notes"] == list(report.notes)
    assert payload["counts"] == report.counts


def test_saving_writes_the_markdown(tmp_path):
    report = ma.specification_report(_mv_clean())
    path = report.save(tmp_path / "report.md")
    assert path.read_text(encoding="utf-8") == report.markdown
    assert str(report) == report.markdown


def test_a_grid_with_nothing_usable_still_renders():
    def run(choices):
        raise RuntimeError("everything broke")

    report = ma.specification_report(ma.multiverse(_spec(), run))
    text = report.markdown
    assert "0 usable, 4 failed" in text
    assert text.count("failed: RuntimeError: everything broke") == 4


def _mv_clean():
    return ma.multiverse(_spec(), _clean)


def test_the_report_is_not_sorted_by_estimate():
    # Grid order is the audit; `curve()` is the sorted view. If the report sorted,
    # a reader could not tell which cell of the grid a row came from.
    def run(choices):
        return ma.rma(YI, sei=SEI, **choices)

    mv = ma.multiverse(_spec(), run)
    rows = ma.specification_report(mv).rows
    order = [(r["choices"]["method"], r["choices"]["test"]) for r in rows]
    assert order == [("REML", "z"), ("REML", "knha"), ("DL", "z"), ("DL", "knha")]


@pytest.mark.parametrize("title", ["", "A title"])
def test_the_title_is_optional(title):
    text = ma.specification_report(_mv_clean(), title=title).markdown
    assert text.startswith("# ") is bool(title)
