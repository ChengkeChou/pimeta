"""Tests for the sensitivity table.

The assertions that matter here are the ones that compare a row against an
independent computation rather than against a hand-written constant. A
leave-one-out row has to agree with ``ma.leave1out`` exactly, because the table
is a re-presentation of that computation and a disagreement means the table is
lying about which study it dropped.

The first version of this module normalised ``RmaResult.level`` (a proportion)
as if it were an alpha, so every leave-one-out interval came out 0.006 wide
instead of 0.18. Nothing in a range check would have caught it; comparing
against ``leave1out`` does.
"""

from __future__ import annotations

import json

import numpy as np
import pytest

import pimeta as ma
from pimeta.sensitivity_table import FAMILIES, METHODS, SensitivityResult, sensitivity_table


def _data(k: int = 12, seed: int = 3, outlier: float | None = None):
    rng = np.random.default_rng(seed)
    y = rng.normal(0.3, 0.25, k)
    v = rng.uniform(0.02, 0.09, k)
    if outlier is not None:
        y[-1] = outlier
    return y, v


@pytest.fixture(scope="module")
def fit():
    y, v = _data()
    return ma.rma(y, vi=v, slab=[f"Study {i + 1}" for i in range(y.size)])


@pytest.fixture(scope="module")
def table(fit):
    return sensitivity_table(fit)


# ------------------------------------------------------------------ baseline

def test_baseline_is_the_fit_itself(fit, table):
    b = table.baseline
    assert b.family == "baseline"
    assert b.estimate == pytest.approx(float(fit.b[0]))
    assert b.se == pytest.approx(float(fit.se[0]))
    assert b.ci_lb == pytest.approx(float(fit.ci_lb[0]))
    assert b.ci_ub == pytest.approx(float(fit.ci_ub[0]))
    assert b.tau2 == pytest.approx(float(fit.tau2))


def test_result_records_the_shape_of_the_fit(fit, table):
    assert table.k == fit.k
    assert table.p == fit.p
    assert table.level == pytest.approx(fit.level)
    assert table.families == list(FAMILIES)


# ---------------------------------------------------------------------- tau2

def test_tau2_rows_cover_every_other_estimator(fit, table):
    variants = [r.variant for r in table.family("tau2")]
    for method in METHODS:
        if method == fit.method:
            continue
        assert f"tau2 = {method}" in variants
    assert f"tau2 = {fit.method}" not in variants, (
        "the estimator the fit already used is the baseline, not a variant")


def test_each_tau2_row_is_a_refit_with_its_own_tau2(table):
    for row in table.family("tau2"):
        assert row.reason, row.variant
        assert row.tau2 is not None
    fe = next(r for r in table.family("tau2") if r.variant == "tau2 = FE")
    assert fe.tau2 == pytest.approx(0.0)
    assert fe.se <= table.baseline.se + 0.02, (
        "a fixed-effect row must have the narrowest interval of the family")


def test_tau2_rows_agree_with_a_direct_refit(fit, table):
    y, v = _data()
    dl = ma.rma(y, vi=v, method="DL")
    row = next(r for r in table.family("tau2") if r.variant == "tau2 = DL")
    assert row.estimate == pytest.approx(float(dl.b[0]), rel=1e-12)
    assert row.se == pytest.approx(float(dl.se[0]), rel=1e-12)
    assert row.tau2 == pytest.approx(float(dl.tau2), rel=1e-10)


def test_the_eb_row_says_it_is_a_convergence_check_not_an_alternative(table):
    row = next(r for r in table.family("tau2") if r.variant == "tau2 = EB")
    assert "converges to PM" in row.reason


# ---------------------------------------------------------------- leave1out

def test_leave_one_out_rows_match_the_standalone_function(fit, table):
    y, v = _data()
    ref = ma.leave1out(y, v, method="REML")
    rows = table.family("leave1out")
    assert len(rows) == y.size
    for i, row in enumerate(rows):
        assert row.estimate == pytest.approx(float(ref.estimate[i]), rel=1e-10)
        assert row.se == pytest.approx(float(ref.se[i]), rel=1e-10)
        # The interval width is the guard against the level-units bug: a
        # proportion read as an alpha gives +/-0.06 instead of +/-1.96.
        assert row.ci_ub - row.ci_lb == pytest.approx(
            2.0 * 1.959963984540054 * row.se, rel=1e-9)


def test_leave_one_out_names_the_study_it_dropped(table):
    variants = [r.variant for r in table.family("leave1out")]
    assert variants[0] == "without Study 1"
    assert variants[-1] == "without Study 12"


def test_the_most_influential_study_is_the_one_that_moves_the_estimate_most():
    # A single displaced study: removing it must move the pooled estimate more
    # than removing an ordinary one, and the row must carry the Cook's distance
    # that explains why.
    y, v = _data(k=12, seed=5, outlier=2.0)
    fit = ma.rma(y, vi=v)
    table = sensitivity_table(fit, over=("leave1out",))
    rows = table.family("leave1out")
    moved = [abs(r.estimate - table.baseline.estimate) for r in rows]
    assert int(np.argmax(moved)) == y.size - 1
    hottest = rows[int(np.argmax(moved))]
    assert hottest.detail["cook_d"] == max(r.detail["cook_d"] for r in rows)
    assert hottest.detail["influential"] is True
    # Removing the outlier must pull the estimate *towards* the others.
    assert hottest.estimate < table.baseline.estimate


def test_leave_one_out_shrinks_tau2_when_the_heterogeneous_study_is_dropped():
    y, v = _data(k=12, seed=5, outlier=2.0)
    fit = ma.rma(y, vi=v)
    table = sensitivity_table(fit, over=("leave1out",))
    rows = table.family("leave1out")
    assert rows[y.size - 1].tau2 < table.baseline.tau2


# ---------------------------------------------------------------------- bias

def test_bias_rows_keep_the_same_estimand(table):
    rows = table.family("bias")
    assert rows, "the default fit should produce at least one bias adjustment"
    for row in rows:
        assert row.reason, row.variant
        assert np.isfinite(row.estimate)
        assert row.ci_lb <= row.estimate <= row.ci_ub, row.variant


def test_bias_is_refused_for_a_meta_regression():
    y, v = _data()
    rng = np.random.default_rng(11)
    fit = ma.rma(y, vi=v, mods=rng.normal(size=y.size))
    table = sensitivity_table(fit, over=("bias",))
    assert table.rows == []
    assert table.refused
    assert "coefficients" in table.refused[0]["reason"]
    assert table.refused[0]["family"] == "bias"


def test_bias_is_refused_below_eight_studies():
    y, v = _data(k=5)
    fit = ma.rma(y, vi=v)
    table = sensitivity_table(fit, over=("bias",))
    assert table.refused
    assert "identifiable" in table.refused[0]["reason"]


def test_a_refused_family_says_why_and_does_not_raise():
    y, v = _data(k=5)
    table = sensitivity_table(ma.rma(y, vi=v), over=("tau2", "bias"))
    assert table.family("tau2")
    assert table.refused
    assert all(item["reason"] for item in table.refused)


# ------------------------------------------------------------------ refusals

def test_rho_is_refused_for_a_univariate_fit_with_a_reason(table):
    univariate = sensitivity_table(ma.rma(*_swap(_data())), over=("tau2", "rho"))
    reasons = [r for r in univariate.refused if r["family"] == "rho"]
    assert reasons
    assert "multivariate" in reasons[0]["reason"]


def _swap(pair):
    y, v = pair
    return y, v


def test_unknown_family_raises_with_the_valid_names(fit):
    with pytest.raises(ValueError, match="unknown family"):
        sensitivity_table(fit, over=("tau2", "nonsense"))


def test_only_refusals_returns_the_refusals_and_no_rows(fit):
    table = sensitivity_table(fit, over=("rho",))
    assert table.rows == []
    assert table.families == []
    assert table.refused and table.refused[0]["family"] == "rho"
    assert "multivariate" in str(table)


def test_a_non_rma_result_is_rejected():
    with pytest.raises(TypeError, match="RmaResult"):
        sensitivity_table(object())


# ------------------------------------------------------------------ plumbing

def test_level_is_read_as_a_proportion(fit, table):
    # 0.95 is a proportion; reading it as an alpha would widen the intervals by
    # a factor of about 32 and the header would say 0.95%.
    assert "95% intervals" in str(table)
    wide = sensitivity_table(fit, over=("leave1out",), level=0.99)
    narrow = sensitivity_table(fit, over=("leave1out",), level=0.90)
    assert "99% intervals" in str(wide)
    assert wide.range("leave1out")[0] == pytest.approx(
        narrow.range("leave1out")[0])  # the estimates do not depend on level
    w = wide.family("leave1out")[0]
    n = narrow.family("leave1out")[0]
    assert (w.ci_ub - w.ci_lb) > (n.ci_ub - n.ci_lb)


def test_rows_round_trip_through_json(table):
    payload = json.loads(json.dumps(table.to_dict()))
    assert payload["baseline"]["family"] == "baseline"
    assert len(payload["rows"]) == len(table.rows)
    assert payload["families"] == table.families
    assert payload["k"] == table.k


def test_family_and_range_helpers_agree_with_the_rows(table):
    rows = table.rows
    assert sum(len(table.family(f)) for f in table.families) == len(rows)
    lo, hi = table.range("tau2")
    assert lo == min(r.estimate for r in table.family("tau2"))
    assert hi == max(r.estimate for r in table.family("tau2"))


def test_a_restricted_request_computes_only_what_was_asked(fit):
    table = sensitivity_table(fit, over=("tau2",))
    assert table.families == ["tau2"]
    assert table.family("leave1out") == []
    assert table.family("bias") == []


def test_methods_argument_restricts_the_tau2_family(fit):
    table = sensitivity_table(fit, over=("tau2",), methods=("DL", "FE"))
    assert [r.variant for r in table.family("tau2")] == ["tau2 = DL",
                                                         "tau2 = FE"]


def test_the_result_is_printable_without_the_internals(fit, table):
    text = str(table)
    assert "baseline" in text and "tau2:" in text and "leave1out:" in text
    assert isinstance(table, SensitivityResult)
