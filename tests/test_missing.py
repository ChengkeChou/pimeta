"""Tests for missing-value handling: drop the rows, but say so."""

from __future__ import annotations

import warnings

import numpy as np
import pytest

import pimeta as ma
from pimeta.missing import MissingDataWarning, complete_cases, is_missing

YI = [0.20, 0.40, float("nan"), 0.50, 0.30, 0.15]
VI = [0.04, 0.05, 0.03, 0.06, 0.02, 0.05]
STUDY = ["A", "A", "B", "B", "C", "C"]


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def test_is_missing_numeric():
    mask = is_missing([1.0, np.nan, np.inf, -np.inf, 2.0])
    assert mask.tolist() == [False, True, True, True, False]


def test_is_missing_integer_and_object():
    assert is_missing([1, 2, 3]).tolist() == [False, False, False]
    assert is_missing(["a", None, float("nan")]).tolist() == [False, True, True]


def test_complete_cases_reports_rows():
    keep, dropped = complete_cases({"a": [1.0, np.nan, 3.0]}, warn=False)
    assert keep.tolist() == [True, False, True]
    assert dropped.tolist() == [1]


def test_complete_cases_is_rowwise_across_columns():
    keep, dropped = complete_cases(
        {"a": [1.0, 2.0, 3.0], "b": [9.0, 8.0, np.nan]}, warn=False)
    assert keep.tolist() == [True, True, False]
    assert dropped.tolist() == [2]


def test_complete_cases_warns_with_row_numbers():
    with pytest.warns(MissingDataWarning, match="dropped 2 of 4 rows"):
        complete_cases({"a": [1.0, np.nan, 3.0, np.nan]}, context="rows")


def test_complete_cases_silent_when_asked():
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        complete_cases({"a": [np.nan]}, warn=False)


def test_complete_cases_rejects_ragged():
    with pytest.raises(ValueError, match="different lengths"):
        complete_cases({"a": [1.0, 2.0], "b": [1.0]}, warn=False)


def test_complete_cases_no_warning_when_clean():
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        keep, dropped = complete_cases({"a": [1.0, 2.0]})
    assert keep.all() and dropped.size == 0


# --------------------------------------------------------------------------- #
# rma
# --------------------------------------------------------------------------- #
def test_rma_drops_and_warns():
    with pytest.warns(MissingDataWarning, match="dropped 1 of 6 studies"):
        r = ma.rma(YI, VI)
    assert r.k == 5
    assert np.isfinite(r.b[0])


def test_rma_warn_missing_false_is_silent():
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        r = ma.rma(YI, VI, warn_missing=False)
    assert r.k == 5


def test_rma_slab_and_mods_stay_aligned():
    dose = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
    with pytest.warns(MissingDataWarning):
        r = ma.rma(YI, VI, mods=dose, slab=list("abcdef"))
    assert r.k == 5
    assert list(r.slab) == ["a", "b", "d", "e", "f"]
    assert [float(v) for v in np.asarray(r.mods).ravel()] == [1.0, 2.0, 4.0, 5.0, 6.0]


def test_rma_missing_moderator_is_dropped():
    dose = [1.0, 2.0, float("nan"), 4.0, 5.0, 6.0]
    with pytest.warns(MissingDataWarning, match="mods\\[0\\]"):
        r = ma.rma([0.2, 0.4, 0.1, 0.5, 0.3, 0.15], VI, mods=dose)
    assert r.k == 5


def test_rma_missing_vi_is_dropped():
    vi = [0.04, 0.05, np.nan, 0.06, 0.02, 0.05]
    with pytest.warns(MissingDataWarning):
        r = ma.rma([0.2, 0.4, 0.1, 0.5, 0.3, 0.15], vi)
    assert r.k == 5


# --------------------------------------------------------------------------- #
# rma_ls
# --------------------------------------------------------------------------- #
def test_rma_ls_drops_scale_missing():
    scale = [1.0, 2.0, float("nan"), 4.0, 5.0, 6.0]
    with pytest.warns(MissingDataWarning, match="scale\\[0\\]"):
        r = ma.rma([0.2, 0.4, 0.1, 0.5, 0.3, 0.15], VI,
                   mods=[1, 2, 3, 4, 5, 6], scale=scale)
    assert len(np.atleast_1d(r.tau2)) == 5


# --------------------------------------------------------------------------- #
# metagen
# --------------------------------------------------------------------------- #
def test_metagen_warns_and_keeps_k_all():
    with pytest.warns(MissingDataWarning, match="dropped 1 of 3 studies"):
        r = ma.metagen([0.2, float("nan"), 0.4], [0.1, 0.1, 0.12])
    assert r.k == 2 and r.k_all == 3


def test_metagen_silent_switch():
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        r = ma.metagen([0.2, float("nan"), 0.4], [0.1, 0.1, 0.12],
                       warn_missing=False)
    assert r.k == 2


# --------------------------------------------------------------------------- #
# rma_mv
# --------------------------------------------------------------------------- #
def test_rma_mv_drops_and_warns():
    yi = [0.20, 0.40, float("nan"), 0.50, 0.30, 0.15]
    with pytest.warns(MissingDataWarning, match="dropped 1 of 6 studies"):
        fit = ma.rma_mv(yi, VI, random="~ 1 | study", data={"study": STUDY})
    assert fit.k == 5


def test_rma_mv_silent_switch():
    yi = [0.20, 0.40, float("nan"), 0.50, 0.30, 0.15]
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        fit = ma.rma_mv(yi, VI, random="~ 1 | study", data={"study": STUDY},
                        warn_missing=False)
    assert fit.k == 5


# --------------------------------------------------------------------------- #
# netmeta
# --------------------------------------------------------------------------- #
def _nma(te, warn_missing: bool = True):
    return ma.netmeta(
        te, [0.10, 0.12, 0.11, 0.09, 0.13, 0.10],
        ["A", "A", "B", "B", "C", "A"],
        ["B", "C", "C", "D", "D", "D"],
        [f"S{i + 1}" for i in range(6)], sm="MD",
        warn_missing=warn_missing,
    )


def test_netmeta_drops_comparisons():
    with pytest.warns(MissingDataWarning, match="dropped 1 of 6 comparisons"):
        res = _nma([0.10, float("nan"), 0.30, 0.15, -0.05, 0.25])
    assert res.m == 5


def test_netmeta_silent_switch():
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        res = _nma([0.10, float("nan"), 0.30, 0.15, -0.05, 0.25],
                   warn_missing=False)
    assert res.m == 5


def test_netmeta_rejects_all_missing():
    with pytest.warns(MissingDataWarning):
        with pytest.raises(ValueError, match="no comparisons left"):
            _nma([float("nan")] * 6)
