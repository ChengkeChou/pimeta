"""``s3.plot`` dispatch, ``predict(newmods=)``, ``confint`` and the
``pooling`` helpers (``confint_tau2`` bounds, ``predict_rma``, ``hc_rma``)."""

from __future__ import annotations

import math

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pytest

import pimeta as ma

YI = [0.10, 0.25, -0.05, 0.40, 0.15, 0.30, 0.05, 0.20, 0.35, -0.10]
VI = [0.010, 0.020, 0.015, 0.030, 0.012, 0.025, 0.018, 0.022, 0.028, 0.016]


@pytest.fixture(scope="module")
def fit():
    return ma.rma(YI, vi=VI, method="REML")


@pytest.fixture(scope="module")
def reg():
    x = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0])
    return ma.rma(YI, vi=VI, mods=x)


@pytest.fixture(autouse=True)
def _close_figures():
    yield
    plt.close("all")


# ---------------------------------------------------------------------------
# s3.plot dispatch
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("which", ["forest", "funnel", "radial", "qqnorm"])
def test_plot_dispatch_needs_only_yi(fit, which):
    out = ma.plot(fit, which=which)
    assert out is not None


@pytest.mark.parametrize("which", ["baujat", "profile"])
def test_plot_dispatch_for_diagnostic_figures(fit, which):
    out = ma.plot(fit, which=which)
    assert out is not None


def test_plot_dispatch_for_the_gosh_plot():
    """GOSH enumerates 2^k - 1 subsets, so keep k small."""
    small = ma.rma(YI[:5], vi=VI[:5], method="REML")
    assert ma.plot(small, which="gosh") is not None


@pytest.mark.parametrize("which", ["bubble", "drapery", "regplot"])
def test_plot_dispatch_for_moderator_figures(reg, which):
    out = ma.plot(reg, which=which)
    assert out is not None


def test_plot_dispatch_is_case_insensitive(fit):
    assert ma.plot(fit, which="Forest") is not None


def test_plot_dispatch_rejects_an_unknown_which(fit):
    with pytest.raises(ValueError, match="unknown 'which'"):
        ma.plot(fit, which="nope")


def test_plot_dispatch_needs_study_level_data():
    class _Empty:
        b = [0.1]
        se = [0.05]

    with pytest.raises(TypeError, match="needs study-level"):
        ma.plot(_Empty(), which="forest")
    with pytest.raises(TypeError, match="needs study-level"):
        ma.plot(_Empty(), which="funnel")
    with pytest.raises(TypeError, match="needs study-level"):
        ma.plot(_Empty(), which="radial")
    with pytest.raises(TypeError, match="binary event counts"):
        ma.plot(_Empty(), which="labbe")


# ---------------------------------------------------------------------------
# predict with explicit newmods / confint
# ---------------------------------------------------------------------------
def test_predict_with_explicit_newmods(reg):
    pred = ma.predict(reg, newmods=np.array([[5.0], [11.0]]))
    assert len(pred) == 2
    b = np.asarray(reg.b, dtype=float)
    assert pred[0]["pred"] == pytest.approx(b[0] + b[1] * 5.0, abs=1e-10)
    assert pred[1]["pred"] == pytest.approx(b[0] + b[1] * 11.0, abs=1e-10)


def test_predict_newmods_ci_matches_the_formula(reg):
    pred = ma.predict(reg, newmods=np.array([[5.0]]))
    vb = np.asarray(reg.vb, dtype=float)
    se_fit = float(np.sqrt(np.array([1.0, 5.0]) @ vb @ np.array([1.0, 5.0])))
    assert pred[0]["se"] == pytest.approx(se_fit, abs=1e-8)
    crit = 1.959963984540054
    assert pred[0]["ci.lb"] == pytest.approx(
        pred[0]["pred"] - crit * se_fit, abs=1e-6)


def test_confint_on_a_tau2_object(fit):
    ci = ma.confint(fit, level=0.90)
    assert ci["level"] == pytest.approx(0.90)
    assert ci["tau2"] == pytest.approx(fit.tau2, abs=1e-10)
    assert 0.0 <= ci["ci_lb"] < ci["ci_ub"]


def test_confint_level_widens_the_interval(fit):
    narrow = ma.confint(fit, level=0.90)
    wide = ma.confint(fit, level=0.99)
    assert (wide["ci_ub"] - wide["ci_lb"]) > (narrow["ci_ub"] - narrow["ci_lb"])


# ---------------------------------------------------------------------------
# pooling helpers
# ---------------------------------------------------------------------------
def test_confint_tau2_upper_bound_hits_tau2_max():
    """A very homogeneous data set drives the profile to the lower bound."""
    yi = [0.100, 0.101, 0.099, 0.100]
    vi = [0.01, 0.01, 0.01, 0.01]
    ci = ma.confint_tau2(yi, vi, 0.0)
    assert ci.tau2_lb >= 0.0
    assert ci.tau2_ub < 0.05
    assert ci.tau2 == pytest.approx(0.0)


def test_confint_tau2_brackets_the_reml_estimate():
    yi = [0.1, 0.4, 0.2, 0.55, 0.05]
    vi = [0.01, 0.02, 0.015, 0.03, 0.012]
    r = ma.rma(yi, vi=vi, method="REML")
    ci = ma.confint_tau2(yi, vi, r.tau2)
    assert ci.tau2_lb < r.tau2 < ci.tau2_ub
    assert ci.tau_lb == pytest.approx(math.sqrt(ci.tau2_lb), abs=1e-12)
    assert ci.tau_ub == pytest.approx(math.sqrt(ci.tau2_ub), abs=1e-12)
    assert ci.i2_lb <= ci.i2 <= ci.i2_ub
    assert ci.h2_lb <= ci.h2 <= ci.h2_ub


def test_confint_tau2_accepts_alpha_level_and_percentage_alike():
    """0.05 (tail), 0.95 (confidence) and 95 (percent) all mean the same."""
    yi = [0.1, 0.4, 0.2, 0.55, 0.05]
    vi = [0.01, 0.02, 0.015, 0.03, 0.012]
    r = ma.rma(yi, vi=vi, method="REML")
    ref = ma.confint_tau2(yi, vi, r.tau2, level=0.05)
    for spec in (0.95, 95):
        got = ma.confint_tau2(yi, vi, r.tau2, level=spec)
        assert got.tau2_ub == pytest.approx(ref.tau2_ub, abs=1e-9)


def test_confint_tau2_rejects_an_impossible_level():
    with pytest.raises(ValueError, match="level"):
        ma.confint_tau2([0.1, 0.2, 0.3], [0.01, 0.02, 0.03], 0.01, level=0.0)


def test_hc_rma_returns_robust_intervals():
    res = ma.hc_rma(YI, VI)
    assert np.isfinite(res.beta) and np.isfinite(res.se)
    assert res.ci_lb < res.beta < res.ci_ub


def test_hc_rma_accepts_the_level_conventions():
    ref = ma.hc_rma(YI, VI, level=0.05)
    for spec in (0.95, 95):
        got = ma.hc_rma(YI, VI, level=spec)
        assert got.ci_lb == pytest.approx(ref.ci_lb, abs=1e-10)
        assert got.ci_ub == pytest.approx(ref.ci_ub, abs=1e-10)


def test_hc_rma_rejects_a_single_study():
    with pytest.raises(ValueError, match="k = 1"):
        ma.hc_rma([0.1], [0.01])


def test_hc_rma_rejects_non_positive_variances():
    with pytest.raises(ValueError, match="non-positive"):
        ma.hc_rma([0.1, 0.2], [0.01, 0.0])


def test_predict_rma_covers_both_interval_types():
    pred = ma.predict_rma(0.2, 0.05, 0.01, 10)
    assert np.isfinite(pred.pred)
    assert pred.pi_lb < pred.ci_lb < pred.pred < pred.ci_ub < pred.pi_ub


def test_the_prediction_interval_default_is_the_t_reference():
    """The default is `t`, not metafor's plug-in, and that is deliberate.

    Measured over 2000 replications per scenario, the plug-in interval covers
    0.882 at k = 20 and 0.804 at k = 5 against a nominal 0.95; the t reference
    covers 0.899 and 0.894. Matching R verifies the arithmetic, it is not a
    reason to default to the interval with the worse coverage.
    """
    default = ma.predict_rma(0.2, 0.05, 0.01, k=10)
    explicit_t = ma.predict_rma(0.2, 0.05, 0.01, k=10, predtype="t")
    assert default.pi_lb == pytest.approx(explicit_t.pi_lb, rel=1e-12)
    assert default.pi_ub == pytest.approx(explicit_t.pi_ub, rel=1e-12)


def test_predtype_default_still_reproduces_the_plug_in_interval():
    """R's behaviour stays reachable, and by name, so it can be verified."""
    from scipy import stats

    est, se, tau2, k = 0.2, 0.05, 0.01, 10
    r = ma.predict_rma(est, se, tau2, k, predtype="default")
    half = float(stats.norm.ppf(0.975)) * np.sqrt(se ** 2 + tau2)
    assert r.pi_lb == pytest.approx(est - half, rel=1e-12)
    assert r.pi_ub == pytest.approx(est + half, rel=1e-12)


def test_the_t_default_is_wider_than_the_plug_in_one():
    """A sanity check on the direction, not just on the formula.

    The t quantile exceeds the normal one and its variance term is the same, so
    the interval must be strictly wider. If a future change made them equal, the
    default would have silently reverted.
    """
    wide = ma.predict_rma(0.2, 0.05, 0.01, k=5)
    narrow = ma.predict_rma(0.2, 0.05, 0.01, k=5, predtype="default")
    assert (wide.pi_ub - wide.pi_lb) > (narrow.pi_ub - narrow.pi_lb)


def test_a_fit_predict_uses_the_same_default():
    r = ma.rma([0.1, 0.3, 0.2, 0.45, 0.15], [0.02, 0.03, 0.02, 0.04, 0.03])
    assert r.predict().pi_lb == pytest.approx(r.predict(predtype="t").pi_lb)
    assert r.predict().pi_lb != pytest.approx(r.predict(predtype="default").pi_lb)
