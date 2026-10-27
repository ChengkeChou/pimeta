"""The Gaussian-process scale: the two limits, and the fit that lands on the spline.

``tau="gp"`` puts the spline's bending basis under a squared-exponential GP
prior.  That is not a second smoother bolted on beside the P-spline — it is a
one-parameter family that *contains* it, and the whole justification for the
extra complexity is the pair of limits:

* a short length scale makes the kernel the identity, so the prior on the
  bending coefficients is a ridge.  The GP then *is* the P-spline, and the two
  fits must agree;
* a long length scale makes the kernel nearly constant.  The constant direction
  is orthogonal to every bending column (that is what the projection in
  ``_SplineBasis`` enforces), so the prior precision blows up, the bending
  function goes to zero, and the scale collapses to the linear model.

Both of those are measured below rather than asserted, because both were got
wrong on the first attempt: the initial version used ``B'KB`` as the precision
instead of its inverse, which agrees with the ridge only at a short length scale
and *unpenalises* the fit at a long one, making the length scale almost
invisible in the answer.  A single fit at one hyperparameter would not have
shown that; the limits do.

The pinned numbers are the fixture the spline fixtures already use, so the two
scales are compared on the same data rather than each on data of its own.
"""

from __future__ import annotations

import warnings

import numpy as np
import pytest

import pimeta as ma

FIXTURE = "dynhet_spline.csv"

#: measured on ``tests/data/dynhet_spline.csv`` with the length scale estimated;
#: the spline's numbers are in ``tests/test_dynhet.py`` and the two agree to
#: about 1e-5, which is the point
GP_ALPHA = [-3.0631794530, -0.0052691866, -4.7176779363, -1.4526472579]
GP_SIGMA = 4.1209864376
GP_EDF = 1.5328748965
GP_LOGLIK = -8.2906563128
GP_CURVE = [0.0249946740, 0.0470658616, 0.1077402577]
GP_CURVE_AT = [1980.0, 1990.0, 2000.0]


@pytest.fixture(scope="module")
def data():
    loaded = np.genfromtxt(
        __file__.replace("tests\\test_dynamichet_gp.py", "tests/data/" + FIXTURE),
        delimiter=",", names=True)
    return loaded["yi"], loaded["vi"], loaded["year"]


@pytest.fixture(scope="module", autouse=True)
def _quiet():
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        yield


@pytest.fixture(scope="module")
def fitted(data):
    return ma.tau_trend(*data, tau="gp")


def test_the_gp_fit_reproduces_the_pinned_numbers(fitted):
    assert fitted.tau_kind == "gp"
    assert np.allclose(fitted.alpha, GP_ALPHA, rtol=1e-6, atol=1e-8)
    assert fitted.sigma_gp == pytest.approx(GP_SIGMA, rel=1e-6)
    assert fitted.edf == pytest.approx(GP_EDF, rel=1e-6)
    assert fitted.loglik == pytest.approx(GP_LOGLIK, rel=1e-6)
    curve = fitted.predict(GP_CURVE_AT)["tau2"]
    assert np.allclose(curve, GP_CURVE, rtol=1e-6)


def test_an_estimated_length_scale_here_lands_on_the_ridge(data):
    """On this fixture the marginal likelihood prefers a GP shorter than the spacing.

    That is not a failure to find a length scale: it is the data saying the
    between-knot smoothing adds nothing beyond the ridge, which is why the
    answer coincides with the P-spline's.  Reported rather than hidden, because
    a reader who fits both should know they will agree on this dataset and that
    this is the reason.
    """
    fit = ma.tau_trend(*data, tau="gp")
    gaps = np.diff(np.unique(data[2]))
    assert fit.length_scale < float(np.min(gaps))
    spline = ma.tau_trend(*data, tau="spline")
    assert fit.edf == pytest.approx(spline.edf, abs=0.01)
    # atol as well as rtol: one of these coefficients is the per-unit-of-time
    # scale slope and is near zero, where a relative tolerance is meaningless
    assert np.allclose(fit.alpha, spline.alpha, rtol=1e-4, atol=1e-4)


def test_a_short_length_scale_is_the_penalised_spline(data):
    y, v, t = data
    gp = ma.tau_trend(y, v, t, tau="gp", length_scale=1e-3)
    spline = ma.tau_trend(y, v, t, tau="spline")
    grid = np.linspace(t.min(), t.max(), 25)
    a = gp.predict(grid)["tau2"]
    b = spline.predict(grid)["tau2"]
    # the residual difference is the kernel's off-diagonal, which is not zero at
    # a finite length scale; 1 per cent is the scale of that, not a tolerance
    assert float(np.max(np.abs(a - b) / b)) < 0.01
    assert gp.edf == pytest.approx(spline.edf, abs=0.05)


def test_a_long_length_scale_collapses_the_bending_block(data):
    y, v, t = data
    long_gp = ma.tau_trend(y, v, t, tau="gp", length_scale=10.0)
    linear = ma.tau_trend(y, v, t, tau="linear")
    grid = np.linspace(t.min(), t.max(), 25)
    # edf is what the bending block spends: at an infinite length scale the
    # prior pins it at zero and the scale is the linear model
    assert long_gp.edf is not None and long_gp.edf < 0.1
    a = long_gp.predict(grid)["tau2"]
    b = linear.predict(grid)["tau2"]
    assert float(np.max(np.abs(a - b) / b)) < 0.05


def test_edf_falls_monotonically_as_the_length_scale_grows(data):
    """The direction the first implementation got backwards."""
    y, v, t = data
    scales = [1e-3, 0.1, 0.6, 2.0, 10.0]
    edfs = [ma.tau_trend(y, v, t, tau="gp", length_scale=s).edf
            for s in scales]
    assert all(later <= earlier + 0.05
               for earlier, later in zip(edfs, edfs[1:])), edfs
    assert edfs[0] > edfs[-1]


def test_the_gp_reports_the_hyperparameters_and_the_curve(fitted):
    record = fitted.to_dict()
    assert record["tau"] == "gp"
    assert record["length_scale"] > 0.0 and record["sigma_gp"] > 0.0
    assert record["edf"] == pytest.approx(GP_EDF, rel=1e-6)
    assert "sigma_s" not in record, (
        "a GP fit has no spline wiggliness; reporting 0.0 would say it does")
    assert set(fitted.names_alpha) == {"alpha.intrcpt", "alpha.time",
                                      "alpha.bend1", "alpha.bend2"}


def test_the_gp_refuses_a_wald_test_and_reports_the_edf_lrt(fitted):
    with pytest.raises(ValueError, match="penalised"):
        fitted.test_trend("wald")
    result = fitted.test_trend("lrt")
    assert result["df1"] == pytest.approx(1.0 + GP_EDF, rel=1e-6)
    assert 0.0 <= result["p_value"] <= 1.0


def test_the_gp_needs_enough_studies(data):
    y, v, t = data
    with pytest.raises(ValueError, match="needs at least"):
        ma.tau_trend(y[:6], v[:6], t[:6], tau="gp")


def test_the_gp_is_refused_by_the_gaussian_sampler(data):
    with pytest.raises(ValueError, match="Stan reference"):
        ma.bayes_tau_trend(*data, tau="gp")
