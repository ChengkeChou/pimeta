"""Joint inference on a C-statistic and a calibration slope.

Two checks, for two different reasons.  Against ``metafor::rma.mv`` because the
pooling is a bivariate meta-analysis and the arithmetic has to be the same
arithmetic.  Against *separate* univariate pooling with ``struct="DIAG"`` and
``rho = 0``, because if the two metrics are independent within and between
studies then the joint fit must collapse to the two marginal ones — and if it
does not, the difference is a bug rather than a feature.

The within-study correlation is the assumption the whole thing rests on, and it
is not recoverable from published summaries: ``sensitivity()`` exists so that
the assumption is reported rather than hidden.
"""

import pathlib

import numpy as np
import pytest

import pimeta as ma

DATA = np.genfromtxt(
    pathlib.Path(__file__).resolve().parent / "data" / "prediction_joint.csv",
    delimiter=",", skip_header=1)
AUC = DATA[:, 1]
N_EVENT = DATA[:, 2]
N_NONEVENT = DATA[:, 3]
SLOPE = DATA[:, 4]
SLOPE_SE = DATA[:, 5]

# --- R 4.6.1, metafor 5.2.1 ---------------------------------------------
R_STUDY1_LOGIT = 1.0102003979
R_STUDY1_VAR = 0.0331588721

R_RHO40_BETA = [1.0077995375, 0.7225667563]
R_RHO40_SE = [0.0814428491, 0.0687812880]
R_RHO40_TAU2 = [0.0840113357, 0.0682756859]
R_RHO40_SIGMA_RHO = -0.1180944829
R_RHO40_AUC = 0.7325892944
R_RHO40_SLOPE_Z = -4.0335569732
R_RHO40_JOINT = 170.1121860422

R_RHO0_BETA = [1.0075465827, 0.7255313919]
R_RHO0_TAU2 = [0.0833176059, 0.0678565182]

R_SEPARATE = {
    "auc": (1.0075577286, 0.0812253120, 0.0833574662),
    "slope": (0.7255313919, 0.0686199354, 0.0678565197),
}


@pytest.fixture(scope="module")
def fit():
    return ma.prediction_joint(auc=AUC, n_event=N_EVENT, n_nonevent=N_NONEVENT,
                               slope=SLOPE, slope_se=SLOPE_SE, rho=0.4)


def test_the_auc_variance_is_hanley_mcneil(fit):
    yi, vi = ma.auc_logit(AUC[0], n_event=N_EVENT[0], n_nonevent=N_NONEVENT[0])
    assert float(np.ravel(yi)[0]) == pytest.approx(R_STUDY1_LOGIT, rel=1e-9)
    assert float(np.ravel(vi)[0]) == pytest.approx(R_STUDY1_VAR, rel=1e-8)


def test_the_joint_fit_matches_metafor(fit):
    assert fit.beta == pytest.approx(R_RHO40_BETA, abs=1e-6)
    assert fit.se == pytest.approx(R_RHO40_SE, abs=1e-6)
    assert np.diag(fit.Sigma) == pytest.approx(R_RHO40_TAU2, abs=1e-5)
    sd = np.sqrt(np.diag(fit.Sigma))
    assert (fit.Sigma / np.outer(sd, sd))[0, 1] == pytest.approx(
        R_RHO40_SIGMA_RHO, abs=1e-5)
    assert fit.auc == pytest.approx(R_RHO40_AUC, abs=1e-7)
    assert fit.test_calibration()["stat"] ** 0.5 == pytest.approx(
        abs(R_RHO40_SLOPE_Z), rel=1e-5)
    assert fit.test_both()["stat"] == pytest.approx(R_RHO40_JOINT, rel=1e-5)


def test_an_independent_analysis_collapses_to_two_marginal_pools():
    """With rho = 0 and a diagonal Sigma the two metrics cannot interact."""
    fit = ma.prediction_joint(auc=AUC, n_event=N_EVENT, n_nonevent=N_NONEVENT,
                              slope=SLOPE, slope_se=SLOPE_SE, rho=0.0,
                              struct="DIAG")
    yi, vi = ma.auc_logit(AUC, n_event=N_EVENT, n_nonevent=N_NONEVENT)
    auc_pool = ma.rma(yi, vi=vi, method="REML")
    slope_pool = ma.rma(SLOPE, vi=SLOPE_SE ** 2, method="REML")
    assert fit.beta == pytest.approx(
        [R_SEPARATE["auc"][0], R_SEPARATE["slope"][0]], abs=1e-5)
    assert fit.se == pytest.approx(
        [R_SEPARATE["auc"][1], R_SEPARATE["slope"][1]], abs=1e-5)
    assert np.diag(fit.Sigma) == pytest.approx(
        [R_SEPARATE["auc"][2], R_SEPARATE["slope"][2]], abs=1e-5)
    # ... and pimeta's own univariate engine agrees with the same numbers
    assert float(auc_pool.b[0]) == pytest.approx(R_SEPARATE["auc"][0],
                                                 abs=1e-5)
    assert float(slope_pool.b[0]) == pytest.approx(R_SEPARATE["slope"][0],
                                                   abs=1e-5)


def test_estimating_the_between_metric_correlation_is_not_the_same_as_zeroing_it():
    """The point of the joint model: UN estimates a correlation of its own.

    A reader could reasonably expect the joint fit with ``rho = 0`` to equal two
    separate pools.  It does not, and metafor does not either: ``struct="UN"``
    estimates the *between*-study correlation too, which on this fixture is
    -0.12 even though the within-study one was set to zero.
    """
    un = ma.prediction_joint(auc=AUC, n_event=N_EVENT, n_nonevent=N_NONEVENT,
                             slope=SLOPE, slope_se=SLOPE_SE, rho=0.0)
    assert un.beta == pytest.approx(R_RHO0_BETA, abs=1e-6)
    assert np.diag(un.Sigma) == pytest.approx(R_RHO0_TAU2, abs=1e-5)
    assert un.beta[0] != pytest.approx(R_SEPARATE["auc"][0], abs=1e-5)


def test_the_joint_test_is_the_closed_form(fit):
    b = np.asarray(fit.beta)
    V = np.asarray(fit.vcov)
    diff = b - np.array([0.0, 1.0])
    stat = float(diff @ np.linalg.solve(V, diff))
    assert fit.test_both()["stat"] == pytest.approx(stat, rel=1e-12)
    assert fit.test_both()["df"] == 2
    # the two marginal tests are the diagonal pieces of the same quadratic form
    assert fit.test_discrimination()["stat"] == pytest.approx(
        b[0] ** 2 / V[0, 0], rel=1e-12)
    assert fit.test_calibration()["stat"] == pytest.approx(
        (b[1] - 1.0) ** 2 / V[1, 1], rel=1e-12)


def test_the_sensitivity_shows_what_the_assumed_rho_is_worth(fit):
    rows = fit.sensitivity()
    assert [row["rho"] for row in rows] == [0.0, 0.25, 0.5, 0.75]
    # the AUC barely moves; the slope does, monotonically
    assert max(row["auc"] for row in rows) - min(
        row["auc"] for row in rows) < 0.01
    slopes = [row["slope"] for row in rows]
    assert all(a > b for a, b in zip(slopes, slopes[1:]))
    # every scenario rejects the null, so the conclusion is not the assumption
    assert all(row["p_joint"] < 1e-6 for row in rows)


def test_each_metric_keeps_its_own_scale(fit):
    """The AUC is reported back-transformed, the slope as it was pooled."""
    assert 0.0 < fit.auc < 1.0
    assert fit.auc_lb < fit.auc < fit.auc_ub
    assert fit.slope_lb < fit.slope < fit.slope_ub
    # the slope is on the identity scale, so its interval is beta +- z se
    z = 1.959963984540054
    assert fit.slope_lb == pytest.approx(fit.slope - z * fit.se[1], rel=1e-9)


def test_a_study_reporting_only_one_metric_still_contributes():
    auc = AUC.copy()
    auc[0] = np.nan                      # study 1 reports no C-statistic
    fit = ma.prediction_joint(auc=auc, n_event=N_EVENT,
                              n_nonevent=N_NONEVENT, slope=SLOPE,
                              slope_se=SLOPE_SE, rho=0.4)
    assert fit.k == AUC.size             # every study is still used
    assert fit.slope == pytest.approx(0.7225667563, abs=1e-3)


def test_a_reported_auc_se_is_accepted_instead_of_group_sizes():
    yi, vi = ma.auc_logit(AUC, n_event=N_EVENT, n_nonevent=N_NONEVENT)
    se = np.sqrt(vi) * (AUC * (1 - AUC))          # back to the AUC scale
    from_se = ma.prediction_joint(auc=AUC, auc_se=se, slope=SLOPE,
                                  slope_se=SLOPE_SE, rho=0.4)
    from_sizes = ma.prediction_joint(auc=AUC, n_event=N_EVENT,
                                     n_nonevent=N_NONEVENT, slope=SLOPE,
                                     slope_se=SLOPE_SE, rho=0.4)
    assert from_se.beta == pytest.approx(from_sizes.beta, abs=1e-9)


def test_the_two_regions_answer_different_questions(fit):
    conf = fit.confidence_region()
    pred = fit.prediction_region()
    assert conf["df"] if "df" in conf else True
    # the prediction region adds Sigma, so it is always wider
    assert np.all(np.diag(np.asarray(pred.cov)) > np.diag(
        np.asarray(conf["cov"])))
    assert pred.contains(pred.center)
    assert conf["crit"] == pytest.approx(5.991464547107979, rel=1e-12)


def test_refusals(fit):
    with pytest.raises(ValueError, match="rho"):
        ma.prediction_joint(auc=AUC, n_event=N_EVENT, n_nonevent=N_NONEVENT,
                            slope=SLOPE, slope_se=SLOPE_SE, rho=1.5)
    with pytest.raises(ValueError, match="n_event"):
        ma.prediction_joint(auc=AUC, slope=SLOPE, slope_se=SLOPE_SE)
    with pytest.raises(ValueError, match="same length"):
        ma.prediction_joint(auc=AUC, n_event=N_EVENT, n_nonevent=N_NONEVENT,
                            slope=SLOPE[:-1], slope_se=SLOPE_SE)
    with pytest.raises(ValueError, match="one of the two metrics"):
        ma.prediction_joint(auc=AUC, n_event=N_EVENT, n_nonevent=N_NONEVENT,
                            slope=np.full(AUC.size, np.nan),
                            slope_se=SLOPE_SE, rho=0.4)


def test_the_record_and_the_text(fit):
    record = fit.to_dict()
    assert record["k"] == AUC.size
    assert record["rho"] == 0.4
    assert record["auc"] == pytest.approx(fit.auc, rel=1e-12)
    assert len(record["vcov"]) == 2
    text = str(fit)
    assert "within-study correlation assumed" in text
    assert "calibration slope" in text
    assert "both at once" in text
    table = fit.coef_table()
    assert [row["metric"] for row in table] == ["auc", "calibration_slope"]
