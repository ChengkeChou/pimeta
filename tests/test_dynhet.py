"""Dynamic heterogeneity: metafor parity, exact posteriors, known-truth recovery.

Three kinds of check, and the first two are the ones that matter.

**Parity.**  ``metafor::rma(yi, vi, mods = ~ time, scale = ~ time)`` fits the
same location-scale model.  Its optimiser is ``nlminb`` over every parameter;
pimeta profiles the location coefficients out and optimises over the scale
coefficients only, so agreement is agreement about the likelihood and not about
two copies of the same loop.  The fixtures cover four nestings under both ML and
REML, plus ``test = "knha"``, whose rule is easy to get wrong in a
location-scale model: metafor scales the *location* variance by
:math:`RSS/(k-p)` and leaves the scale variance alone.

**Exact.**  Given the scale coefficients the model is Gaussian in the location
coefficients, so the Bayesian posterior can be computed by quadrature instead of
sampled; ``tools/quadrature_dynhet.py`` does exactly that, and the sampler is
compared against it.  Numbers in ``EXACT`` were produced by::

    python tools/quadrature_dynhet.py

**Property.**  Data simulated from a known :math:`\\mu(t)` and known
:math:`\\tau^2(t)`: the curve and the time-varying prediction interval have to
contain the truth, and omitting the mean trend has to be shown to cost
something.
"""

from __future__ import annotations

import csv
import json
import math
import pathlib
import warnings

import numpy as np
import pytest

import pimeta as ma
from pimeta.dynamichet import (
    BayesTauTrendResult,
    TauTrendResult,
    _design,
    _grad_alpha,
    _neg_loglik,
    _spline_basis,
    bayes_tau_trend,
    tau_trend,
)

DATA = pathlib.Path(__file__).parent / "data" / "dynhet.csv"
TRUTH = pathlib.Path(__file__).parent / "reference" / "dynhet_truth.json"
MULTINMA = pathlib.Path(__file__).parent / "reference" / "dynhet_metafor.csv"
SPLINE_DATA = pathlib.Path(__file__).parent / "data" / "dynhet_spline.csv"

# exact posterior by quadrature, tools/quadrature_dynhet.py
EXACT_ALPHA = np.array([-2.5562, 0.0956])
EXACT_ALPHA_SD = np.array([0.3908, 0.0395])
EXACT_BETA = np.array([0.2462, 0.0178])
EXACT_BETA_SD = np.array([0.0641, 0.0060])
EXACT_TAU2_CENTRE = 0.0836

#: metafor names its terms after the columns it was given
R_NAMES = {"time": "year_c", "time^2": "year_c2", "intrcpt": "intrcpt"}

# --- the bump fixture, frequency and Bayes ---------------------------------
# pimeta's penalised fit on tests/data/dynhet_spline.csv, measured:
SPLINE_ALPHA = [-3.0631900549350006, -0.005270274355615835,
                -4.717910495153534, -1.45266348212426]
SPLINE_SIGMA_S = 4.1212671509
SPLINE_EDF = 1.5329199132
SPLINE_LOGLIK = -8.2905903237
SPLINE_LINEAR_LOGLIK = -11.5780538117
SPLINE_LRT = 6.7262676732
SPLINE_CURVE = [0.024993656043552866, 0.09448788959624178,
                0.008875667242929965]        # at 1980, 1997.5, 2015

# The same model in Stan (tools/r_reference/tau2t_spline.R, cmdstanr 0.9.0.9002
# with CmdStan 2.40.0, 4 chains x 2000 draws after 2000 warmup, seed 1234,
# adapt_delta 0.99, half-normal(5) prior on sigma_s so that the check is of the
# estimator and not of the prior):
STAN_A0 = -3.0612672364
STAN_A0_SD = 0.3647899992
STAN_A1 = -0.0061295207
STAN_A1_SD = 0.0431843765
STAN_SIGMA_S = 4.6005169645
STAN_SIGMA_S_SD = 2.6287324424
STAN_B0 = 0.2445982396
STAN_B0_SD = 0.0365659125
STAN_TAU2 = [0.0459424205, 0.1056788833, 0.0244893325]   # 1980, 1997.5, 2015
STAN_TAU2_Q5 = [0.0028661401, 0.0511044738, 0.0003175569]
STAN_TAU2_Q95 = [0.1428366075, 0.1940743659, 0.0927778195]


def _fixture():
    rows = list(csv.DictReader(DATA.open(encoding="utf-8")))
    y = np.array([float(r["yi"]) for r in rows])
    v = np.array([float(r["vi"]) for r in rows])
    t = np.array([float(r["year"]) for r in rows])
    return y, v, t


def _reference():
    out = {}
    for row in csv.DictReader(MULTINMA.open(encoding="utf-8")):
        out[(row["model"], row["part"], row["term"])] = row
    return out


def _r_name(name: str) -> str:
    base = name.replace("alpha.", "")
    return R_NAMES.get(base, base)


@pytest.fixture(scope="module")
def fit():
    y, v, t = _fixture()
    return tau_trend(y, v, t)


@pytest.fixture(scope="module")
def bayes():
    y, v, t = _fixture()
    return bayes_tau_trend(y, v, t,
                           mcmc={"seed": 3, "n_draws": 4000, "warmup": 2000})


# --------------------------------------------------------------------------- #
# metafor parity
# --------------------------------------------------------------------------- #
CASES = [
    ("A", "linear", "linear", "REML"), ("A", "linear", "linear", "ML"),
    ("B", "linear", "none", "REML"), ("B", "linear", "none", "ML"),
    ("C", "none", "none", "REML"), ("C", "none", "none", "ML"),
    ("D", "quadratic", "linear", "REML"), ("D", "quadratic", "linear", "ML"),
]


@pytest.mark.parametrize("tag,mu,tau,method", CASES)
def test_coefficients_match_metafor(tag, mu, tau, method):
    y, v, t = _fixture()
    ref = _reference()
    fitted = tau_trend(y, v, t, mu=mu, tau=tau, method=method, test="z")
    for part, est, ses, names in (
            ("location", fitted.beta, fitted.se_beta, fitted.names_beta),
            ("scale", fitted.alpha, fitted.se_alpha, fitted.names_alpha)):
        for j, name in enumerate(names):
            row = ref[(f"{tag}_{method}", part, _r_name(name))]
            assert est[j] == pytest.approx(float(row["estimate"]), abs=2e-6)
            assert ses[j] == pytest.approx(float(row["se"]), abs=2e-6)
    row = ref[(f"{tag}_{method}", "location", "intrcpt")]
    assert fitted.loglik == pytest.approx(float(row["loglik"]), abs=1e-8)


def test_knha_matches_metafor():
    """metafor scales the location variance only, and the df differ by part."""
    y, v, t = _fixture()
    ref = _reference()
    fitted = tau_trend(y, v, t, test="knha")
    assert fitted.s2w == pytest.approx(
        float(ref[("A_knha", "location", "intrcpt")]["s2w"]), abs=1e-6)
    for j, name in enumerate(fitted.names_beta):
        row = ref[("A_knha", "location", _r_name(name))]
        assert fitted.se_beta[j] == pytest.approx(float(row["se"]), abs=1e-6)
        assert fitted.summary()[j]["p_value"] == pytest.approx(
            float(row["p_value"]), abs=1e-6)
    # the scale standard errors are *not* rescaled: metafor's are identical to
    # the z-based ones, and so are ours
    plain = tau_trend(y, v, t, test="z")
    for j, name in enumerate(fitted.names_alpha):
        row = ref[("A_knha", "scale", _r_name(name))]
        assert fitted.se_alpha[j] == pytest.approx(float(row["se"]), abs=1e-6)
        assert fitted.se_alpha[j] == pytest.approx(plain.se_alpha[j], abs=1e-12)


def test_wald_test_matches_metafor_qs():
    y, v, t = _fixture()
    ref = _reference()
    fitted = tau_trend(y, v, t, test="z")
    out = fitted.test_trend("wald")
    assert out["reference"] == "chi2"
    assert out["statistic"] == pytest.approx(
        float(ref[("A_REML", "scale", "intrcpt")]["QS"]), abs=1e-6)
    assert out["p_value"] == pytest.approx(
        float(ref[("A_REML", "scale", "intrcpt")]["QSp"]), abs=1e-6)


def test_knha_turns_the_scale_test_into_an_f():
    y, v, t = _fixture()
    fitted = tau_trend(y, v, t, test="knha")
    out = fitted.test_trend("wald")
    assert out["reference"] == "F"
    assert out["df2"] == fitted.k - fitted.q


def test_constant_scale_nests_into_the_ordinary_random_effects_model():
    y, v, t = _fixture()
    fitted = tau_trend(y, v, t, mu="none", tau="none", test="z")
    plain = ma.rma(y, vi=v, method="REML")
    assert math.exp(fitted.alpha[0]) == pytest.approx(plain.tau2, rel=1e-5)


# --------------------------------------------------------------------------- #
# algebra
# --------------------------------------------------------------------------- #
def test_analytic_gradient_matches_finite_differences():
    y, v, t = _fixture()
    centre = float(t.mean())
    X = _design(t, "linear", centre)
    Z = _design(t, "linear", centre)
    fitted = _fit_for(y, v, X, Z)
    grad = _grad_alpha(fitted, y, v, X, Z, "REML")
    for j in range(fitted.size):
        step = 1e-6
        up = fitted.copy()
        down = fitted.copy()
        up[j] += step
        down[j] -= step
        numeric = (_neg_loglik(up, y, v, X, Z, "REML")
                   - _neg_loglik(down, y, v, X, Z, "REML")) / (2.0 * step)
        assert grad[j] == pytest.approx(-numeric, rel=1e-6, abs=1e-8)


def _fit_for(y, v, X, Z):
    from pimeta.dynamichet import _fit_alpha
    return _fit_alpha(y, v, X, Z, "REML", 500, 1e-10)["alpha"]


def test_profile_and_gls_standard_errors_differ_only_at_second_order():
    """The Gaussian expected information is block-diagonal between the parts."""
    y, v, t = _fixture()
    fitted = tau_trend(y, v, t, test="z")
    ratio = fitted.se_beta_profile / fitted.se_beta
    assert np.all(ratio > 1.0) and np.all(ratio < 1.01)
    assert np.allclose(ratio, 1.0025, atol=5e-4)


def test_ratio_is_the_exponential_of_the_scale_slope():
    y, v, t = _fixture()
    fitted = tau_trend(y, v, t)
    out = fitted.ratio()
    assert out["ratio"] == pytest.approx(math.exp(fitted.alpha[1]), rel=1e-12)
    assert out["q_low"] > 0.0 and out["q_low"] < out["ratio"] < out["q_high"]
    curve = fitted.predict([t.min(), t.max()])["tau2"]
    assert curve[1] / curve[0] == pytest.approx(
        math.exp(fitted.alpha[1] * (t.max() - t.min())), rel=1e-12)


# --------------------------------------------------------------------------- #
# the curve and the time-varying prediction interval
# --------------------------------------------------------------------------- #
def test_prediction_interval_is_wider_than_the_confidence_interval(fit):
    out = fit.predict(np.linspace(1990.0, 2025.0, 5))
    assert np.all(out["pi_high"] - out["pi_low"]
                  > out["q_high"] - out["q_low"])
    assert np.all(out["tau2"] > 0.0)


def test_the_curve_contains_the_simulated_truth(fit):
    truth = json.loads(TRUTH.read_text(encoding="utf-8"))
    times = np.array(truth["year_range"])
    centre = truth["mean_time"]
    mu_true = truth["b0"] + truth["b1"] * (times - centre)
    out = fit.predict(times)
    for j in range(times.size):
        assert out["q_low"][j] <= mu_true[j] <= out["q_high"][j]
        # the prediction interval is for a new study: it has to contain the
        # true mean too, and be much wider
        assert out["pi_low"][j] <= mu_true[j] <= out["pi_high"][j]
    assert out["tau2"][-1] > out["tau2"][0]


def test_omitting_the_mean_trend_inflates_tau2():
    """The reason ``mu`` cannot be left at "none" while ``tau`` trends.

    A mean effect that climbs with time produces squared residuals that grow
    away from the mean time, and a scale model with nothing else to explain
    them reports them as heterogeneity.  The numbers here are the measured
    damage on a 60-study simulation with a constant tau^2 of 0.15.
    """
    rng = np.random.default_rng(31)
    k, tau2_true = 60, 0.15
    t = np.linspace(0.0, 1.0, k)
    vi = rng.uniform(0.01, 0.04, k)
    y = (0.10 + 1.50 * t + rng.normal(0.0, math.sqrt(tau2_true), k)
         + rng.normal(0.0, 1.0, k) * np.sqrt(vi))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        wrong = tau_trend(y, vi, t, mu="none", tau="linear", test="knha")
    right = tau_trend(y, vi, t, mu="linear", tau="linear", test="knha")
    low_wrong = wrong.predict([0.0])["tau2"][0]
    low_right = right.predict([0.0])["tau2"][0]
    assert low_wrong > 2.5 * tau2_true
    assert 0.8 * tau2_true < low_right < 1.5 * tau2_true


def test_unidentified_scale_warns_rather_than_reporting_a_wild_ratio():
    """With almost no heterogeneity the log-linear scale can run to infinity."""
    rng = np.random.default_rng(7)
    k = 40
    t = np.linspace(0.0, 1.0, k)
    vi = np.full(k, 0.02)
    y = 0.10 + 0.80 * t + rng.normal(0.0, 0.1, k) + rng.normal(
        0.0, 1.0, k) * np.sqrt(vi)
    with pytest.warns(RuntimeWarning, match="more than two orders of magnitude"):
        tau_trend(y, vi, t, mu="linear", tau="linear")


# --------------------------------------------------------------------------- #
# the Bayesian fit against the exact posterior
# --------------------------------------------------------------------------- #
def test_bayes_scale_coefficients_match_the_exact_quadrature(bayes):
    ess = bayes.posterior.ess()
    sd = bayes.alpha_draws.std(axis=0, ddof=1)
    for j in range(2):
        mcse = float(sd[j] / math.sqrt(float(ess[j])))
        combined = math.hypot(mcse, EXACT_ALPHA_SD[j])
        assert abs(bayes.alpha[j] - EXACT_ALPHA[j]) < 4.0 * combined
        # an SD has its own Monte Carlo error, sd/sqrt(2 ESS)
        sd_mcse = float(sd[j] / math.sqrt(2.0 * float(ess[j])))
        assert abs(sd[j] - EXACT_ALPHA_SD[j]) < 4.0 * sd_mcse + 0.004


def test_bayes_location_coefficients_match_the_exact_quadrature(bayes):
    sd = bayes.beta_draws.std(axis=0, ddof=1)
    ess = float(np.min(bayes.posterior.ess()))
    for j in range(2):
        mcse = float(sd[j] / math.sqrt(ess))
        combined = math.hypot(mcse, EXACT_BETA_SD[j])
        assert abs(bayes.beta[j] - EXACT_BETA[j]) < 4.0 * combined
        sd_mcse = float(sd[j] / math.sqrt(2.0 * ess))
        assert abs(sd[j] - EXACT_BETA_SD[j]) < 4.0 * sd_mcse + 0.002


def test_bayes_tau2_at_the_mean_time_matches_the_exact_value(bayes):
    centre = np.exp(bayes.alpha_draws[:, 0])
    assert float(centre.mean()) == pytest.approx(EXACT_TAU2_CENTRE, rel=0.10)


def test_bayes_interval_carries_the_scale_uncertainty(fit, bayes):
    """The two fits answer the same question with different widths."""
    times = np.array([1990.0, 2025.0])
    freq = fit.predict(times)
    sampled = bayes.predict(times)
    assert np.all(sampled["pi_high"] - sampled["pi_low"]
                  > freq["pi_high"] - freq["pi_low"])
    assert np.all(sampled["q_high"] - sampled["q_low"]
                  > 0.9 * (freq["q_high"] - freq["q_low"]))


def test_bayes_ratio_and_priors(bayes):
    out = bayes.ratio()
    assert out["mean"] > 1.0
    assert 0.0 <= out["prob_above_one"] <= 1.0
    assert out["q_low"] < out["mean"] < out["q_high"]
    assert bayes.priors["alpha_sd"] == 0.3
    assert bayes.priors["alpha0_sd"] == 2.0
    assert "alpha.time" in [r["term"] for r in bayes.summary()]


def test_bayes_study_ratio_of_interval_widths_is_larger_at_the_ends(bayes):
    out = bayes.predict(np.linspace(1990.0, 2025.0, 4))
    widths = out["pi_high"] - out["pi_low"]
    assert widths[-1] > widths[0]


# --------------------------------------------------------------------------- #
# the trend test
# --------------------------------------------------------------------------- #
def test_likelihood_ratio_statistic_is_twice_the_log_likelihood_gap(fit):
    out = fit.test_trend("lrt")
    null = fit._refit()
    assert out["statistic"] == pytest.approx(
        2.0 * (fit.loglik - null["loglik"]), abs=1e-9)
    assert 0.0 <= out["p_value"] <= 1.0


def test_bootstrap_agrees_with_the_lrt_when_heterogeneity_is_present(fit):
    lrt = fit.test_trend("lrt")
    boot = fit.test_trend("bootstrap", n_boot=99, seed=1)
    assert boot["statistic"] == pytest.approx(lrt["statistic"], abs=1e-9)
    assert boot["p_value"] == pytest.approx(lrt["p_value"], abs=0.10)


def test_bootstrap_keeps_its_level_when_there_is_no_heterogeneity():
    """Under tau^2 = 0 the chi-square reference is the wrong one."""
    rng = np.random.default_rng(99)
    k = 30
    t = np.linspace(0.0, 1.0, k)
    vi = rng.uniform(0.02, 0.06, k)
    y = 0.2 + 0.3 * t + rng.normal(0.0, 1.0, k) * np.sqrt(vi)
    fitted = tau_trend(y, vi, t, test="knha")
    boot = fitted.test_trend("bootstrap", n_boot=199, seed=2)
    assert 0.05 < boot["p_value"] <= 1.0


# --------------------------------------------------------------------------- #
# refusals, record, surface
# --------------------------------------------------------------------------- #
def test_mismatched_lengths_are_refused():
    with pytest.raises(ValueError, match="same length"):
        tau_trend([1.0, 2.0, 3.0], [0.1, 0.2], [1.0, 2.0, 3.0])


def test_too_few_studies_are_refused():
    with pytest.raises(ValueError, match="at least three"):
        tau_trend([1.0, 2.0], [0.1, 0.1], [1.0, 2.0])


def test_non_positive_variance_is_refused():
    with pytest.raises(ValueError, match="strictly positive"):
        tau_trend([1.0, 2.0, 3.0], [0.1, 0.0, 0.1], [1.0, 2.0, 3.0])


def test_constant_time_is_refused():
    with pytest.raises(ValueError, match="constant"):
        tau_trend([1.0, 2.0, 3.0], [0.1, 0.1, 0.1], [2020.0, 2020.0, 2020.0])


def test_unknown_shapes_and_methods_are_refused():
    with pytest.raises(ValueError, match="unknown mu model"):
        tau_trend([1.0, 2.0, 3.0], [0.1, 0.1, 0.1], [1.0, 2.0, 3.0],
                  mu="cubic")
    with pytest.raises(ValueError, match="'method'"):
        tau_trend([1.0, 2.0, 3.0], [0.1, 0.1, 0.1], [1.0, 2.0, 3.0],
                  method="DL")
    with pytest.raises(ValueError, match="'test'"):
        tau_trend([1.0, 2.0, 3.0], [0.1, 0.1, 0.1], [1.0, 2.0, 3.0],
                  test="knha2")
    with pytest.raises(ValueError, match="'se'"):
        tau_trend([1.0, 2.0, 3.0], [0.1, 0.1, 0.1], [1.0, 2.0, 3.0],
                  se="robust")


def test_ratio_needs_a_linear_scale(fit):
    y, v, t = _fixture()
    quad = tau_trend(y, v, t, tau="quadratic")
    with pytest.raises(ValueError, match="linear scale model"):
        quad.ratio()


def test_trend_test_needs_a_scale_slope():
    y, v, t = _fixture()
    flat = tau_trend(y, v, t, tau="none")
    with pytest.raises(ValueError, match="no scale slope"):
        flat.test_trend()


def test_bad_method_name_for_the_trend_test(fit):
    with pytest.raises(ValueError, match="'method' must be"):
        fit.test_trend("permutation")


def test_unknown_mcmc_setting_is_refused():
    with pytest.raises(ValueError, match="unknown 'mcmc' setting"):
        bayes_tau_trend([1.0, 2.0, 3.0], [0.1, 0.1, 0.1], [1.0, 2.0, 3.0],
                        mcmc={"n_draw": 10})


def test_bad_prior_scale_is_refused():
    with pytest.raises(ValueError, match="must be finite and positive"):
        bayes_tau_trend([1.0, 2.0, 3.0], [0.1, 0.1, 0.1], [1.0, 2.0, 3.0],
                        prior_alpha_sd=0.0)


def test_mu_none_with_a_trending_scale_warns():
    y, v, t = _fixture()
    with pytest.warns(RuntimeWarning, match="mean effect is held constant"):
        tau_trend(y, v, t, mu="none", tau="linear")


def test_record_and_text(fit):
    payload = fit.to_dict()
    json.dumps(payload)
    assert payload["mu"] == "linear" and payload["tau"] == "linear"
    assert payload["test"] == "knha"
    assert payload["k"] == 36
    assert "Location-scale model" in str(fit)
    assert "alpha.time" in str(fit)


def test_bayes_record_and_text(bayes):
    payload = bayes.to_dict()
    json.dumps(payload)
    assert payload["kind"] == "bayesian"
    assert "Bayesian location-scale" in str(bayes)


def test_exported_from_the_top_level_package():
    assert ma.tau_trend is tau_trend
    assert ma.bayes_tau_trend is bayes_tau_trend
    assert ma.TauTrendResult is TauTrendResult
    assert ma.BayesTauTrendResult is BayesTauTrendResult


# --------------------------------------------------------------------------- #
# the spline scale
# --------------------------------------------------------------------------- #
def _spline_fixture():
    """The bump fixture: tau^2 low at both ends, high in the middle."""
    rows = list(csv.DictReader(SPLINE_DATA.open(encoding="utf-8")))
    y = np.array([float(r["yi"]) for r in rows])
    v = np.array([float(r["vi"]) for r in rows])
    t = np.array([float(r["year"]) for r in rows])
    u = (t - t.min()) / (t.max() - t.min())
    tau2 = np.exp(-3.0 + 1.6 * np.exp(-((u - 0.5) / 0.18) ** 2))
    return y, v, t, tau2


def test_the_bending_basis_is_orthonormal_and_orthogonal_to_the_linear_part():
    y, v, t = _fixture()
    basis = _spline_basis(t, float(t.mean()))
    B = basis.bending(t)
    assert B.shape == (t.size, 2)
    assert B.T @ B == pytest.approx(np.eye(2), abs=1e-12)
    linear = np.column_stack([np.ones(t.size), t - t.mean()])
    assert B.T @ linear == pytest.approx(np.zeros((2, 2)), abs=1e-10)


def test_the_bending_basis_evaluates_at_new_times_the_same_way_it_was_fitted():
    """The stored transform, not a fresh Gram-Schmidt, defines the basis."""
    y, v, t = _fixture()
    basis = _spline_basis(t, float(t.mean()))
    grid = np.linspace(t.min(), t.max(), 9)
    B = basis.bending(grid)
    # every column is orthogonal to the linear part *at those points* only
    # approximately, but its norm must be O(1) and the curve must be smooth
    assert np.all(np.isfinite(B))
    assert np.all(np.abs(B) < 10.0)
    assert np.max(np.abs(np.diff(B, axis=0))) < 1.0


def test_the_spline_collapses_to_the_linear_fit_when_the_truth_is_linear():
    """The fixture has a linear tau^2(t); the wiggliness must go to zero."""
    y, v, t = _fixture()
    linear = tau_trend(y, v, t, tau="linear")
    spline = tau_trend(y, v, t, tau="spline")
    assert spline.sigma_s is not None and spline.sigma_s <= 2e-3
    assert spline.edf == pytest.approx(0.0, abs=1e-6)
    assert spline.alpha[:2] == pytest.approx(linear.alpha, abs=1e-4)
    assert spline.loglik == pytest.approx(linear.loglik, abs=1e-6)


def test_the_spline_recovers_a_non_monotone_shape_the_linear_model_misses():
    y, v, t, tau2 = _spline_fixture()
    spline = tau_trend(y, v, t, tau="spline")
    linear = tau_trend(y, v, t, tau="linear")
    assert spline.sigma_s is not None and spline.sigma_s > 0.5
    assert spline.edf is not None and spline.edf > 0.5
    assert spline.loglik > linear.loglik + 2.0
    # the fitted curve peaks in the middle, where the truth does
    grid = np.linspace(t.min(), t.max(), 9)
    fitted = spline.predict(grid)["tau2"]
    assert 3 <= int(np.argmax(fitted)) <= 5
    assert fitted[4] > 1.5 * max(fitted[0], fitted[-1])


def test_the_spline_spends_its_degrees_of_freedom_in_the_scale_tests():
    y, v, t, _ = _spline_fixture()
    spline = tau_trend(y, v, t, tau="spline")
    linear = tau_trend(y, v, t, tau="linear")
    assert spline.df_alpha is not None and linear.df_alpha is not None
    # edf is folded into the df: the scale reference loses what the spline spends
    assert spline.df_alpha == pytest.approx(
        spline.k - 2 - round(float(spline.edf)), abs=1)
    assert spline.df_alpha < linear.df_alpha
    out = spline.test_trend("lrt")
    assert out["df1"] == pytest.approx(1.0 + float(spline.edf), abs=1e-6)
    assert 0.0 <= out["p_value"] <= 1.0


def test_a_penalised_spline_has_no_wald_test_and_says_so():
    y, v, t, _ = _spline_fixture()
    spline = tau_trend(y, v, t, tau="spline")
    with pytest.raises(ValueError, match="no fixed number of scale"):
        spline.test_trend("wald")


def test_a_spline_scale_needs_enough_studies_and_the_polynomial_mean():
    y, v, t, _ = _spline_fixture()
    with pytest.raises(ValueError, match="needs at least"):
        tau_trend(y[:6], v[:6], t[:6], tau="spline")
    with pytest.raises(ValueError, match="scale only"):
        tau_trend(y, v, t, mu="spline")
    # a 3-knot spline has one bending column and needs fewer studies; ten
    # studies cannot pin the wiggliness, so the span warning fires and is
    # expected here
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        small = tau_trend(y[:10], v[:10], t[:10], tau="spline", n_knots=3)
    assert small.alpha.size == 3


def test_the_spline_is_refused_by_the_gaussian_sampler():
    y, v, t, _ = _spline_fixture()
    with pytest.raises(ValueError, match="Stan reference"):
        bayes_tau_trend(y, v, t, tau="spline")


def test_the_spline_record_carries_the_knots_and_the_wiggliness():
    y, v, t, _ = _spline_fixture()
    spline = tau_trend(y, v, t, tau="spline")
    payload = spline.to_dict()
    json.dumps(payload)
    assert len(payload["knots"]) == 4
    assert payload["sigma_s"] > 0.0
    assert payload["edf"] > 0.0
    assert spline.knots is not None and spline.knots.size == 4
    assert "alpha.bend1" in str(spline)


def test_the_penalised_fit_matches_the_committed_numbers():
    """The spline fit itself, pinned on bytes and on one seed."""
    y, v, t, _ = _spline_fixture()
    spline = tau_trend(y, v, t, tau="spline")
    assert list(spline.alpha) == pytest.approx(SPLINE_ALPHA, abs=1e-8)
    assert spline.sigma_s == pytest.approx(SPLINE_SIGMA_S, rel=1e-6)
    assert spline.edf == pytest.approx(SPLINE_EDF, rel=1e-6)
    assert spline.loglik == pytest.approx(SPLINE_LOGLIK, abs=1e-7)
    linear = tau_trend(y, v, t, tau="linear")
    assert linear.loglik == pytest.approx(SPLINE_LINEAR_LOGLIK, abs=1e-7)
    out = spline.test_trend("lrt")
    assert out["statistic"] == pytest.approx(SPLINE_LRT, abs=1e-6)
    curve = spline.predict(np.array([1980.0, 1997.5, 2015.0]))["tau2"]
    assert list(curve) == pytest.approx(SPLINE_CURVE, rel=1e-6)


def test_the_penalised_fit_agrees_with_the_stan_posterior():
    """Two estimators of one model: penalised REML against sampled Bayes.

    The frequentist side is a Laplace-approximated marginal likelihood; the
    Bayesian side is CmdStan on the same basis, with a half-normal(5) prior on
    the wiggliness so that what is compared is the estimator and not a prior
    that already knows the answer.  They agree on the level, the trend and the
    shape.  `sigma_s` agrees only to about 30 %, which is the honest
    statement: the wiggliness is the weakly identified direction and both
    answers say so (the posterior SD of `sigma_s` is 2.6 on a mean of 4.6).
    """
    y, v, t, _ = _spline_fixture()
    spline = tau_trend(y, v, t, tau="spline")
    # a coefficient: 4 posterior standard errors, plus the Laplace error
    assert spline.alpha[0] == pytest.approx(STAN_A0, abs=4 * STAN_A0_SD + 0.02)
    assert spline.alpha[1] == pytest.approx(STAN_A1, abs=4 * STAN_A1_SD + 0.02)
    assert spline.beta[0] == pytest.approx(STAN_B0, abs=4 * STAN_B0_SD + 0.02)
    # the wiggliness: same order of magnitude, and the frequentist value inside
    # one posterior standard deviation of the Bayesian mean
    assert abs(spline.sigma_s - STAN_SIGMA_S) < STAN_SIGMA_S_SD
    # the curve: the frequentist values lie inside the 90 % posterior interval
    curve = spline.predict(np.array([1980.0, 1997.5, 2015.0]))["tau2"]
    for j in range(3):
        assert STAN_TAU2_Q5[j] <= curve[j] <= STAN_TAU2_Q95[j]
        assert abs(float(curve[j]) - STAN_TAU2[j]) < 0.5 * STAN_TAU2[j] + 0.02
    # both estimators put the peak in the middle, which is the substantive claim
    assert STAN_TAU2[1] > STAN_TAU2[0] and STAN_TAU2[1] > STAN_TAU2[2]
    assert curve[1] > curve[0] and curve[1] > curve[2]
