"""The last public functions that had no test: ``gosh`` subsets, ``rma_glmm``
argument checking, ``peters_test``, ``lrp_common``, ``metaadd`` and
``points_regplot``."""

from __future__ import annotations

import importlib

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pytest

import pimeta as ma

_metabin = importlib.import_module("pimeta.metabin")
_publication_bias = importlib.import_module("pimeta.publication_bias")

YI = [0.10, 0.25, -0.05, 0.40, 0.15, 0.30, 0.05, 0.20]
VI = [0.010, 0.020, 0.015, 0.030, 0.012, 0.025, 0.018, 0.022]
NI = [50, 80, 40, 120, 60, 90, 45, 100]


@pytest.fixture(autouse=True)
def _close_figures():
    yield
    plt.close("all")


# ---------------------------------------------------------------------------
# gosh
# ---------------------------------------------------------------------------
def test_gosh_random_subsets_keeps_the_method():
    """``subsets`` is a count; it used to shadow the ``method`` variable."""
    res = ma.gosh(YI, VI, subsets=6, seed=1)
    assert res.method == "REML"
    assert len(res.tau2) == 6


def test_gosh_random_subsets_is_reproducible():
    a = ma.gosh(YI, VI, subsets=6, seed=42)
    b = ma.gosh(YI, VI, subsets=6, seed=42)
    assert list(a.tau2) == pytest.approx(list(b.tau2))


def test_gosh_exact_enumeration():
    res = ma.gosh(YI[:4], VI[:4])
    # every subset with at least one study -> 2^4 - 1
    assert len(res.tau2) == 15
    assert res.k_total == 4


def test_gosh_with_a_fixed_tau2():
    res = ma.gosh(YI[:4], VI[:4], tau2=0.02)
    # every subset with more than one study keeps the supplied tau2
    kept = [v for v in res.tau2 if v == v and v > 0]
    assert kept and all(np.isclose(v, 0.02) for v in kept)


def test_gosh_with_a_moderator():
    x = np.arange(1.0, len(YI) + 1)
    res = ma.gosh(YI, VI, X=x, subsets=8, seed=2)
    assert res.p == 2
    assert np.asarray(res.beta).shape[1] == 2


def test_gosh_rejects_a_too_small_subset_count():
    with pytest.raises(ValueError, match=">= 2"):
        ma.gosh(YI, VI, subsets=1)


def test_gosh_rejects_an_unknown_method():
    with pytest.raises(ValueError, match="method"):
        ma.gosh(YI, VI, method="PM")


# ---------------------------------------------------------------------------
# rma_glmm argument checking
# ---------------------------------------------------------------------------
def test_rma_glmm_two_arm_measures():
    kwargs = dict(ai=[10, 12, 8], bi=[40, 38, 44], ci=[15, 14, 12],
                  di=[35, 42, 40])
    or_ = ma.rma_glmm("OR", **kwargs)
    rr = ma.rma_glmm("RR", **kwargs)
    assert np.isfinite(np.asarray(or_.b)[0])
    assert np.isfinite(np.asarray(rr.b)[0])
    assert or_.measure == "OR" and rr.measure == "RR"


def test_rma_glmm_single_arm_measure():
    res = ma.rma_glmm("PLO", xi=[10, 12, 8], ni=[50, 52, 45])
    assert np.isfinite(np.asarray(res.b)[0])
    assert res.measure == "PLO"


def test_rma_glmm_rejects_a_mismatched_measure():
    """A two-arm measure without ai/bi/ci/di used to raise a bare TypeError."""
    with pytest.raises(ValueError, match="two-arm measure"):
        ma.rma_glmm("OR", xi=[10, 12], ni=[50, 52])
    with pytest.raises(ValueError, match="single-arm measure"):
        ma.rma_glmm("PLO", ai=[10], bi=[40], ci=[15], di=[35])


def test_rma_glmm_rejects_an_unknown_measure():
    with pytest.raises(ValueError, match="unsupported measure"):
        ma.rma_glmm("XX", ai=[1], bi=[1], ci=[1], di=[1])


def test_rma_glmm_rejects_an_unknown_method():
    with pytest.raises(ValueError, match="method"):
        ma.rma_glmm("OR", ai=[10], bi=[40], ci=[15], di=[35], method="PM")


def test_rma_glmm_fixed_effects():
    res = ma.rma_glmm("OR", ai=[10, 12, 8], bi=[40, 38, 44],
                      ci=[15, 14, 12], di=[35, 42, 40], method="FE")
    assert np.isfinite(np.asarray(res.b)[0])
    assert res.method == "FE"


# ---------------------------------------------------------------------------
# peters_test / lrp_common
# ---------------------------------------------------------------------------
def test_peters_test_reports_a_regression_on_ninv():
    res = _publication_bias.peters_test(YI, VI, NI)
    assert res.predictor == "ninv"
    # a negative slope on 1/n means small studies report larger effects
    assert res.stat < 0
    assert 0.0 <= res.p <= 1.0
    assert res.k == len(YI)


def test_peters_test_is_antisymmetric_in_the_effect_direction():
    """Reversing every effect flips the sign of the regression statistic."""
    a = _publication_bias.peters_test(YI, VI, NI)
    b = _publication_bias.peters_test([-y for y in YI], VI, NI)
    assert a.stat == pytest.approx(-b.stat, abs=1e-8)
    assert a.p == pytest.approx(b.p, abs=1e-10)


def test_peters_test_accepts_the_other_tests():
    res = _publication_bias.peters_test(YI, VI, NI, test="z")
    assert 0.0 <= res.p <= 1.0


def test_lrp_common_pools_two_arms():
    est, se, phi = _metabin.lrp_common([10, 12], [100, 110],
                                       [15, 14], [105, 108])
    assert float(est) < 0.0            # fewer events on the experimental arm
    assert float(se) > 0.0
    assert float(phi) >= 1.0
    # the Firth fit is not the naive Wald estimate
    naive = np.log((10 / 90) / (15 / 90))
    assert float(est) != pytest.approx(naive, abs=1e-6)


def test_lrp_common_is_antisy_mmetric_in_the_arms():
    a = _metabin.lrp_common([10, 12], [100, 110], [15, 14], [105, 108])
    b = _metabin.lrp_common([15, 14], [105, 108], [10, 12], [100, 110])
    assert float(a[0]) == pytest.approx(-float(b[0]), abs=1e-8)


# ---------------------------------------------------------------------------
# metaadd / points_regplot
# ---------------------------------------------------------------------------
def _rma():
    return ma.rma([0.1, 0.2, 0.3], vi=[0.01, 0.02, 0.03])


def test_metabind_requires_study_level_estimates():
    """metabind() needs the per-study TE/seTE, which rma() does not retain."""
    with pytest.raises(ValueError, match="study-level"):
        ma.metabind(_rma(), _rma())


def _meta():
    return ma.metagen([0.1, 0.2, 0.3], [0.1, 0.14, 0.17])


def test_metaadd_returns_the_object_when_nothing_is_supplied():
    """R's metaadd() with no 'type' returns the input untouched."""
    m = _meta()
    assert ma.metaadd(m) is m


def test_metaadd_appends_a_common_effect_result():
    res = ma.metaadd(_meta(), type="common", TE=0.5, lower=0.4, upper=0.6,
                     se=0.05)
    assert len(res.added) == 1
    row = res.added[0]
    assert row["type"] == "common"
    assert row["TE"] == pytest.approx(0.5)
    assert row["se"] == pytest.approx(0.05)
    assert res.is_metaadd is True


def test_metaadd_appends_a_tau2_result():
    res = ma.metaadd(_meta(), type="tau2", TE=0.02)
    assert res.added[0]["type"] == "tau2"
    assert res.added[0]["TE"] == pytest.approx(0.02)


def test_metaadd_accepts_several_rows():
    res = ma.metaadd(_meta(), type=["common", "tau2"], TE=[0.5, 0.02],
                     lower=[0.4, None], upper=[0.6, None], se=[0.05, None])
    assert [r["type"] for r in res.added] == ["common", "tau2"]


def test_metaadd_from_another_meta_analysis():
    other = _meta()
    res = ma.metaadd(_meta(), data=other)
    assert res.added
    assert {r["type"] for r in res.added} <= {"common", "random", "tau2",
                                              "prediction"}


def test_metaadd_result_exposes_r_style_vectors():
    m = _meta()
    res = ma.metaadd(m, type="common", TE=0.5, lower=0.4, upper=0.6, se=0.05)
    values = list(np.atleast_1d(res.seTE_common))
    assert len(values) == 2
    assert values[-1] == pytest.approx(0.05)


def test_metaadd_requires_the_arguments_its_type_needs():
    with pytest.raises(ValueError, match="'TE' must be provided"):
        ma.metaadd(_meta(), type="common", lower=0.4, upper=0.6)
    with pytest.raises(ValueError, match="'lower' and 'upper'"):
        ma.metaadd(_meta(), type="common", TE=0.5)
    with pytest.raises(ValueError, match="type must be one of"):
        ma.metaadd(_meta(), type="nope", TE=0.5)


def test_metaadd_rejects_a_single_study_meta_analysis():
    one = ma.metagen([0.1], [0.1])
    with pytest.raises(ValueError, match="at least two studies"):
        ma.metaadd(one, type="common", TE=0.1, lower=0.0, upper=0.2)


def test_metaadd_rejects_non_meta_data():
    with pytest.raises(TypeError, match="meta-analysis result"):
        ma.metaadd(_meta(), data=object())


def test_points_regplot_draws_on_an_axes():
    x = np.arange(1.0, len(YI) + 1)
    reg = ma.rma(YI, vi=VI, mods=x)
    fig, ax = plt.subplots()
    out = ma.points_regplot(reg, ax=ax)
    assert out is not None


def test_points_regplot_needs_a_fitted_model_with_data():
    class _Bare:
        pass

    with pytest.raises(TypeError, match="fitted model"):
        ma.points_regplot(_Bare(), ax=plt.subplots()[1])
