"""One-stage IPD models with random slopes, pinned against ``lme4``.

The random-intercept one-stage model in :mod:`pimeta.onestage` has a closed-form
likelihood, because ``V_i = sigma2 I + tau2 J`` inverts by a rank-one update.
A random *slope* removes that: ``V_i = sigma2 (I + Z_i Lambda Z_i')`` does not
factor, and the likelihood has to be evaluated one study at a time.

``lme4::lmer`` is the reference, and it is the right one here —the one-stage
model *is* a linear mixed model, so ``y ~ treat + agec + treat:agec +
(1 + treat | study)`` is the same likelihood rather than an analogy of it.

The fixture (``tests/data/onestage_slopes.csv``) is 24 studies of 165–275
participants with a correlated random intercept and treatment slope, drawn from
``D = [[0.09, 0.03], [0.03, 0.04]]`` and residual SD 0.8.  Studies are large on
purpose: with the 25–60 participant studies a first attempt used, the sampling
noise in each study's own intercept and slope is comparable to the variance
being estimated, both implementations land on the correlation boundary
``rho = 0.99998``, and there is nothing left to discriminate.  The estimates
here are interior (``rho = 0.712`` estimated against 0.5 true).
"""

import pathlib

import numpy as np
import pytest

import pimeta as ma

DATA = np.genfromtxt(
    pathlib.Path(__file__).resolve().parent / "data" / "onestage_slopes.csv",
    delimiter=",", skip_header=1)
STUDY = DATA[:, 0]
TREAT = DATA[:, 1]
AGE = DATA[:, 2]
Y = DATA[:, 3]
COV = {"agec": AGE - 55.0, "treat:agec": TREAT * (AGE - 55.0)}

# --- lme4 2.1.0, R 4.6.1, from tests/data/onestage_slopes.csv ---------------
# Regenerate the fixture with tools/make_onestage_slopes_fixture.py and these
# with tools/r_reference/onestage_slopes.R.
R_INT_B = [0.574846269376, -0.158773991757, 0.022382670718, 0.008308243906]
R_INT_SE = [0.080088206405, 0.022218186785, 0.001747292394, 0.002455889992]
R_INT_TAU2 = 0.147992622635
R_INT_SIGMA2 = 0.659599378963
R_INT_LOGLIK = -6532.613438871319

R_UN_B = [0.573360420176, -0.156558653359, 0.022090035412, 0.008651410203]
R_UN_SE = [0.064596613890, 0.046637636794, 0.001735416031, 0.002440666935]
R_UN_D = [[0.094274026644, 0.043976216246],
          [0.043976216246, 0.040444152689]]
R_UN_SIGMA2 = 0.650000594750
R_UN_LOGLIK = -6502.327209688169

R_DG_D = [0.103062576665, 0.046201311974]
R_DG_SIGMA2 = 0.649698105290
R_DG_LOGLIK = -6506.939969039937

R_ML_D = [[0.090097842575, 0.042419225898],
          [0.042419225898, 0.038176412984]]
R_ML_SIGMA2 = 0.649765192239
R_ML_LOGLIK = -6487.288469115336


@pytest.fixture(scope="module")
def intercept_only():
    return ma.ipd_one_stage(Y, study=STUDY, treat=TREAT, covariates=COV)


@pytest.fixture(scope="module")
def slope():
    return ma.ipd_one_stage(Y, study=STUDY, treat=TREAT, covariates=COV,
                            re_slopes=["treat"])


@pytest.fixture(scope="module")
def slope_diag():
    return ma.ipd_one_stage(Y, study=STUDY, treat=TREAT, covariates=COV,
                            re_slopes=["treat"], struct="DIAG")


@pytest.fixture(scope="module")
def slope_ml():
    return ma.ipd_one_stage(Y, study=STUDY, treat=TREAT, covariates=COV,
                            re_slopes=["treat"], method="ML")


def test_the_random_intercept_model_still_matches_lme4(intercept_only):
    """The path the random-slope code did not touch, on the same data."""
    fit = intercept_only
    assert fit.coefnames == ["(Intercept)", "treat", "agec", "treat:agec"]
    assert fit.b == pytest.approx(R_INT_B, rel=1e-6)
    assert fit.se == pytest.approx(R_INT_SE, rel=1e-5)
    assert fit.tau2 == pytest.approx(R_INT_TAU2, abs=1e-6)
    assert float(fit.sigma2) == pytest.approx(R_INT_SIGMA2, abs=1e-6)
    assert fit.loglik == pytest.approx(R_INT_LOGLIK, abs=1e-6)
    assert fit.D is None


def test_a_correlated_random_slope_matches_lme4(slope):
    """``(1 + treat | study)``: the covariance is 2x2 and unstructured."""
    fit = slope
    assert fit.re_names == ["(Intercept)", "treat"]
    assert fit.b == pytest.approx(R_UN_B, rel=1e-6)
    assert fit.se == pytest.approx(R_UN_SE, rel=1e-5)
    assert np.asarray(fit.D) == pytest.approx(np.array(R_UN_D), abs=1e-5)
    assert float(fit.sigma2) == pytest.approx(R_UN_SIGMA2, abs=1e-6)
    assert fit.loglik == pytest.approx(R_UN_LOGLIK, abs=1e-6)
    # tau2 is the intercept variance, and the covariance is on the off-diagonal
    assert fit.tau2 == pytest.approx(R_UN_D[0][0], abs=1e-5)


def test_a_diagonal_structure_matches_lme4(slope_diag):
    """``(1 | study) + (0 + treat | study)``: the same Z, independent effects."""
    fit = slope_diag
    assert np.asarray(fit.D)[0, 1] == 0.0
    assert np.diag(np.asarray(fit.D)) == pytest.approx(np.array(R_DG_D),
                                                       abs=1e-5)
    assert float(fit.sigma2) == pytest.approx(R_DG_SIGMA2, abs=1e-6)
    assert fit.loglik == pytest.approx(R_DG_LOGLIK, abs=1e-6)


def test_maximum_likelihood_matches_lme4(slope_ml):
    fit = slope_ml
    assert fit.method == "ML"
    assert np.asarray(fit.D) == pytest.approx(np.array(R_ML_D), abs=1e-5)
    assert float(fit.sigma2) == pytest.approx(R_ML_SIGMA2, abs=1e-6)
    assert fit.loglik == pytest.approx(R_ML_LOGLIK, abs=1e-6)


def test_the_correlated_model_fits_better_than_the_diagonal_one(slope,
                                                               slope_diag):
    """Two extra parameters, and the likelihood has to say they were worth it."""
    assert slope.loglik > slope_diag.loglik
    lr = 2.0 * (slope.loglik - slope_diag.loglik)
    # one extra parameter, on a boundary-free interior estimate
    assert 0.5 < lr < 20.0


def test_the_slope_variance_is_what_makes_the_standard_errors_differ(
        intercept_only, slope):
    """Ignoring a real random slope understates the treatment effect's se."""
    assert slope.se[1] > intercept_only.se[1]


def test_a_contrast_reports_a_prediction_band_wider_than_its_interval(slope):
    """A new study's treatment effect at a given age, not the average's."""
    rows = slope.contrast([[0.0, 1.0, 0.0, 0.0]])
    assert rows[0]["est"] == pytest.approx(slope.b[1], rel=1e-12)
    assert rows[0]["se"] == pytest.approx(slope.se[1], rel=1e-12)
    width = rows[0]["ci_ub"] - rows[0]["ci_lb"]
    pred = rows[0]["pred_ub"] - rows[0]["pred_lb"]
    assert pred > width
    # the gap is the study-level treatment-effect SD
    extra = np.sqrt(rows[0]["se"] ** 2 + np.asarray(slope.D)[1, 1])
    assert (pred - width) / 2 == pytest.approx(
        1.959963985 * (extra - rows[0]["se"]), rel=1e-6)


def test_a_non_linear_interaction_is_a_contrast(slope):
    """The one-stage answer to a curve: a row of weights per covariate value."""
    x = np.array([45.0, 55.0, 65.0])
    z = x - 55.0
    # treat * (1, z, z^2): the slope coefficient, its trend, its curvature
    L = np.column_stack([np.zeros(x.size), np.ones(x.size), np.zeros(x.size),
                         np.zeros(x.size)])
    rows = slope.contrast(L)
    assert [row["est"] for row in rows] == pytest.approx(
        [slope.b[1]] * 3, rel=1e-12)
    assert all("pred_lb" in row for row in rows)

    # a real quadratic contrast: treat + treat:agec * z
    L2 = np.column_stack([np.zeros(x.size), np.ones(x.size), np.zeros(x.size),
                          z])
    rows2 = slope.contrast(L2)
    est = np.array([row["est"] for row in rows2])
    assert np.all(np.diff(est) > 0) or np.all(np.diff(est) < 0)


def test_a_slope_passed_as_an_array_needs_no_coefficient_name():
    fit = ma.ipd_one_stage(Y, study=STUDY, treat=TREAT, covariates=COV,
                            re_slopes=[(AGE - 55.0) / 10.0])
    assert fit.re_names == ["(Intercept)", "re1"]
    assert fit.D.shape == (2, 2)
    # no coefficient to add, so there is no new-study band
    rows = fit.contrast([[0.0, 1.0, 0.0, 0.0]])
    assert "pred_lb" not in rows[0]


def test_a_collinear_random_design_is_rejected():
    with pytest.raises(ValueError, match="collinear"):
        ma.ipd_one_stage(Y, study=STUDY, treat=TREAT, covariates=COV,
                         re_slopes=["(Intercept)"])


def test_an_unknown_column_is_rejected():
    with pytest.raises(ValueError, match="unknown column"):
        ma.ipd_one_stage(Y, study=STUDY, treat=TREAT, covariates=COV,
                         re_slopes=["no_such_column"])


def test_an_unknown_struct_is_rejected():
    with pytest.raises(ValueError, match="unknown struct"):
        ma.ipd_one_stage(Y, study=STUDY, treat=TREAT, covariates=COV,
                         re_slopes=["treat"], struct="HCS")


def test_random_slopes_are_refused_for_a_binary_outcome():
    with pytest.raises(ValueError, match="gaussian"):
        ma.ipd_one_stage((Y > 0).astype(float), study=STUDY, treat=TREAT,
                         family="binomial", re_slopes=["treat"])


def test_more_random_effects_than_studies_is_refused():
    small = np.isin(STUDY, [1.0, 2.0])
    with pytest.raises(ValueError, match="cannot identify"):
        ma.ipd_one_stage(Y[small], study=STUDY[small],
                         treat=np.linspace(0.0, 1.0, int(small.sum())),
                         re_slopes=["treat"])


def test_the_record_carries_the_covariance(slope):
    record = slope.to_dict()
    assert record["struct"] == "UN"
    assert record["re_names"] == ["(Intercept)", "treat"]
    assert np.asarray(record["D"]) == pytest.approx(np.asarray(slope.D),
                                                    rel=1e-12)
    assert slope.to_dict()["D"] is not None

