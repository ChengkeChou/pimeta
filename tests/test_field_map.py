"""Every claim in ``docs/reference/field-map.md`` is checked against a real fit.

The page exists because porting between R and Python fails in three ways, and the
worst of them is a field that keeps its name and changes meaning. A table of
those claims is worthless if it drifts, so this module reads the page and
verifies what can be verified:

* every pimeta field the page names exists on a fitted ``rma``;
* every "different name" pair really is the dotted name with ``.`` replaced by
  ``_`` — except the two rows the page flags as *not* mechanical;
* every field the page calls absent is genuinely absent.

What cannot be checked here is the R side: whether metafor really returns a field
of that name. That claim rests on the parity tests and on the source, which is
the honest limit of a Python repository.
"""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pytest

import pimeta as ma

ROOT = Path(__file__).resolve().parents[1]
PAGE = ROOT / "docs" / "reference" / "field-map.md"

#: The two rows of section 2 that are not a mechanical rename, and why.
NOT_MECHANICAL = {"weight": "weights_pct", "R2": "R2"}


@pytest.fixture(scope="module")
def fit():
    yi = np.array([0.10, 0.45, 0.30, 0.62, 0.20, 0.51, 0.33, 0.44])
    sei = np.array([0.15, 0.20, 0.12, 0.25, 0.18, 0.16, 0.22, 0.19])
    return ma.rma(yi, sei ** 2, mods=np.arange(yi.size, dtype=float))


def _text() -> str:
    return PAGE.read_text(encoding="utf-8")


def _section(title: str) -> str:
    text = _text()
    start = text.index(title)
    rest = text[start + len(title):]
    nxt = rest.find("\n## ")
    return rest if nxt < 0 else rest[:nxt]


def _backticked(body: str) -> list[str]:
    return re.findall(r"`([^`]+)`", body)


def test_the_page_exists_and_names_its_three_categories():
    text = _text()
    for needle in ("Same name, same statistic", "Same statistic, different name",
                   "No counterpart in metafor", "No counterpart in pimeta"):
        assert needle in text, needle


def test_every_field_named_as_present_exists_on_a_fit(fit):
    # Section 1 lists one row per statistic, so a cell can hold "b, se".
    names = []
    for cell in _backticked(_section("## 1. Same name, same statistic")):
        names.extend(part.strip() for part in cell.split(","))
    assert len(names) >= 25, names
    missing = [n for n in names if not hasattr(fit, n)]
    assert not missing, f"the page names fields a fit does not have: {missing}"


def test_every_renamed_pair_is_the_mechanical_rename(fit):
    body = _section("## 2. Same statistic, different name")
    rows: list[tuple[str, str]] = []
    for line in body.splitlines():
        if not line.startswith("|") or set(line) <= set("|- \t"):
            continue
        cells = [c for c in line.strip().strip("|").split("|")]
        if len(cells) < 2:
            continue
        r_names = [c.strip().strip("*").strip("`") for c in
                   re.findall(r"`([^`]+)`", cells[0])]
        py_names = [c.strip().strip("*").strip("`") for c in
                    re.findall(r"`([^`]+)`", cells[1])]
        if len(r_names) == len(py_names) and r_names and r_names[0] != "metafor":
            rows.extend(zip(r_names, py_names))
    assert len(rows) >= 8, rows
    for r_name, py_name in rows:
        if r_name in NOT_MECHANICAL:
            assert py_name == NOT_MECHANICAL[r_name]
        else:
            assert py_name == r_name.replace(".", "_"), (r_name, py_name)
        assert hasattr(fit, py_name), f"{py_name} is named but not present"


def test_the_two_non_mechanical_rows_are_still_true(fit):
    # `weights` is the raw weight and `weights_pct` the percentage; if those two
    # ever swap, the row that warns about it becomes the bug.
    assert np.allclose(np.sum(fit.weights_pct), 100.0, atol=1e-8)
    assert not np.allclose(np.sum(fit.weights), 100.0)
    # R2 needs a null model to compare against; without moderators it is None
    plain = ma.rma(fit.yi, fit.vi)
    assert plain.R2 is None


#: Fields that live on a saved record rather than on a fitted object.
RECORD_ONLY = {"analysis", "settings"}


def test_every_pimeta_only_field_is_present(fit):
    body = _section("## 3. No counterpart in metafor")
    names: list[str] = []
    for line in body.splitlines():
        if not line.startswith("|") or set(line) <= set("|- \t"):
            continue
        first_cell = line.strip().strip("|").split("|")[0]
        names.extend(re.findall(r"`([^`]+)`", first_cell))
    assert names, body
    for name in names:
        if name.endswith("()"):
            assert callable(getattr(fit, name[:-2], None)), name
        elif name in RECORD_ONLY:
            continue                      # documented as record fields
        else:
            assert hasattr(fit, name), name


def test_every_metafor_only_field_is_genuinely_absent(fit):
    body = _section("## 4. No counterpart in pimeta")
    for name in _backticked(body):
        # the section lists matrices in one cell and R's printing options in
        # another; only single identifiers are checkable
        if not re.fullmatch(r"[A-Z]", name):
            continue
        assert not hasattr(fit, name), (
            f"the page says {name} has no counterpart, but the fit has it")
