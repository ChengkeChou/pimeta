"""Time-course meta-analysis.

The fixture is 16 studies reporting a treatment effect at 3, 6 and 12 months,
with a genuinely linear random slope across studies, so the model the package
fits is the model that generated the data.  ``phi = 0.6`` gives the within-study
AR(1) correlation.

``metafor::rma.mv`` is the reference, with the same block-diagonal ``V`` built
from the same ``phi`` and ``random = ~ 1 + tc | study, struct = "GEN"``.

One bug is pinned here because nothing else would have caught it: ``g_matrix()``
returned ``None`` for every ``struct="GEN"`` fit, because the GEN component
stores its dimension as ``q`` and the accessor looked only for ``J``. Every
random-slope fit therefore reported a between-study covariance of zero, and
``predict_region`` — which adds that covariance — reported a prediction region
no wider than the confidence one.
"""

import pathlib

import numpy as np
import pytest

import pimeta as ma

DATA = np.genfromtxt(
    pathlib.Path(__file__).resolve().parent / "data" / "timecourse.csv",
    delimiter=",", skip_header=1)
STUDY = DATA[:, 0]
TIME = DATA[:, 1]
YI = DATA[:, 2]
SE = DATA[:, 3]

# --- R 4.6.1, metafor 5.2.1, phi = 0.6, struct = "GEN" ------------------
R_BETA = [0.4062128757, -0.0343886083]
R_SE = [0.0253548977, 0.0049060637]
R_TAU2 = [0.0067965525, 0.0001892062]
R_RHO = 0.1587569915
R_TREND = 49.1318175566
R_CURVE_T = [3.0, 6.0, 12.0]
R_CURVE_EST = [0.5093787005, 0.4062128757, 0.1998812262]
R_CURVE_SE = [0.0297633178, 0.0253548977, 0.0381661271]


@pytest.fixture(scope="module")
def fit():
    return ma.timecourse(YI, study=STUDY, time=TIME, se=SE, phi=0.6)


def test_the_linear_time_course_matches_metafor(fit):
    assert fit.names == ["(Intercept)", "time"]
    assert fit.coef == pytest.approx(R_BETA, abs=1e-9)
    assert fit.se == pytest.approx(R_SE, abs=1e-6)
    assert fit.tau2 == pytest.approx(R_TAU2, abs=1e-6)
    sd = np.sqrt(np.diag(fit.Sigma))
    assert (fit.Sigma / np.outer(sd, sd))[0, 1] == pytest.approx(R_RHO,
                                                                 abs=1e-5)


def test_the_trend_test_matches_metafor(fit):
    test = fit.test_trend()
    assert test["df"] == 1
    # `rel=1e-5` rather than 1e-6: the statistic is (b/se)^2, so it doubles the
    # relative error of the standard error, and the two optimisers stop at
    # tolerances that differ in the sixth digit. The coefficient and its SE are
    # pinned separately in `test_the_linear_time_course_matches_metafor`, which
    # is where the tight comparison belongs.
    assert test["stat"] == pytest.approx(R_TREND, rel=1e-5)
    assert test["pval"] < 1e-6


def test_the_curve_matches_metafor(fit):
    curve = fit.curve(R_CURVE_T)
    assert list(curve.x) == R_CURVE_T
    assert curve.est == pytest.approx(R_CURVE_EST, abs=1e-9)
    assert curve.se == pytest.approx(R_CURVE_SE, abs=1e-7)


def test_the_between_study_covariance_is_reported_for_a_random_slope(fit):
    """The regression test for the ``g_matrix`` bug.

    A ``struct="GEN"`` fit has a ``q x q`` covariance; before the fix the
    accessor skipped it and every caller saw zeros.
    """
    assert np.asarray(fit.Sigma).shape == (2, 2)
    assert np.diag(fit.Sigma).min() > 0.0
    G, names = fit.fit.g_matrix()
    assert G is not None
    assert G.shape == (2, 2)
    assert names == ["intrcpt", "b1"]
    # and the prediction region built on it is wider than the confidence one
    region = ma.predict_region(fit.fit)
    assert np.all(np.diag(region.cov) > np.diag(np.asarray(fit.vcov)))


def test_the_prediction_band_is_wider_than_the_confidence_band(fit):
    curve = fit.curve()
    assert np.all(curve.pred_ub - curve.pred_lb
                  > curve.ci_ub - curve.ci_lb)
    assert np.all(curve.est - curve.ci_lb < curve.est - curve.pred_lb)


def test_points_outside_the_observed_range_are_flagged(fit):
    curve = fit.curve([1.0, 6.0, 20.0])
    assert list(curve.outside) == [True, False, True]


def test_the_within_study_correlation_is_the_one_that_was_asked_for(fit):
    """A study's block is ``se_i se_j phi^|t_i - t_j|``, not a free matrix."""
    V = np.asarray(fit.fit.V)
    for i in range(STUDY.size):
        for j in range(STUDY.size):
            if STUDY[i] != STUDY[j]:
                continue
            expect = (SE[i] ** 2 if i == j else
                      SE[i] * SE[j] * 0.6 ** abs(TIME[i] - TIME[j]))
            assert V[i, j] == pytest.approx(expect, rel=1e-12)


def test_ignoring_the_correlation_changes_the_answer():
    """The reason the argument is required rather than defaulted."""
    ar1 = ma.timecourse(YI, study=STUDY, time=TIME, se=SE, phi=0.6)
    indep = ma.timecourse(YI, study=STUDY, time=TIME, se=SE, rho=0.0)
    assert ar1.se[1] != pytest.approx(indep.se[1], rel=1e-6)
    # a positive within-study correlation carries less information than
    # independence claims, so the trend is less precise
    assert ar1.se[1] > indep.se[1]


def test_a_spline_needs_asking_for_and_reports_a_non_linearity_test():
    linear = ma.timecourse(YI, study=STUDY, time=TIME, se=SE, phi=0.6)
    with pytest.raises(ValueError, match="non-linear"):
        linear.test_nonlinear()
    spline = ma.timecourse(YI, study=STUDY, time=TIME, se=SE, phi=0.6,
                           n_knots=3)
    assert len(spline.coef) > 2
    test = spline.test_nonlinear()
    assert test["df"] >= 1
    # the fixture really is linear, so the curvature is not needed
    assert test["pval"] > 0.05


def test_the_refusals():
    with pytest.raises(ValueError, match="not independent"):
        ma.timecourse(YI, study=STUDY, time=TIME, se=SE)
    with pytest.raises(ValueError, match="'phi' must lie"):
        ma.timecourse(YI, study=STUDY, time=TIME, se=SE, phi=1.5)
    with pytest.raises(ValueError, match="give 'se' or 'vi'"):
        ma.timecourse(YI, study=STUDY, time=TIME, phi=0.6)
    with pytest.raises(ValueError, match="at least two studies"):
        ma.timecourse(YI[:3], study=STUDY[:3], time=TIME[:3], se=SE[:3],
                      phi=0.6)
    with pytest.raises(ValueError, match="same length"):
        ma.timecourse(YI, study=STUDY, time=TIME[:-1], se=SE, phi=0.6)
    with pytest.raises(ValueError, match="unknown test"):
        ma.timecourse(YI, study=STUDY, time=TIME, se=SE, phi=0.6, test="knha")


def test_vi_is_accepted_instead_of_se():
    by_se = ma.timecourse(YI, study=STUDY, time=TIME, se=SE, phi=0.6)
    by_vi = ma.timecourse(YI, study=STUDY, time=TIME, vi=SE ** 2, phi=0.6)
    assert by_vi.coef == pytest.approx(by_se.coef, rel=1e-12)


def test_the_record_and_the_text(fit):
    record = fit.to_dict()
    assert record["k"] == 16
    assert record["phi"] == 0.6
    assert record["support"] == [3.0, 12.0]
    assert record["basis"] is None
    assert len(record["coef"]) == 2
    text = str(fit)
    assert "within-study correlation" in text
    assert "effect changes with time" in text
    table = fit.coef_table()
    assert [row["term"] for row in table] == ["(Intercept)", "time"]
