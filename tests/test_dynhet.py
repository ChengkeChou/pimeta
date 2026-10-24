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
    bayes_tau_trend,
    tau_trend,
)

DATA = pathlib.Path(__file__).parent / "data" / "dynhet.csv"
TRUTH = pathlib.Path(__file__).parent / "reference" / "dynhet_truth.json"
MULTINMA = pathlib.Path(__file__).parent / "reference" / "dynhet_metafor.csv"

# exact posterior by quadrature, tools/quadrature_dynhet.py
EXACT_ALPHA = np.array([-2.5562, 0.0956])
EXACT_ALPHA_SD = np.array([0.3908, 0.0395])
EXACT_BETA = np.array([0.2462, 0.0178])
EXACT_BETA_SD = np.array([0.0641, 0.0060])
EXACT_TAU2_CENTRE = 0.0836

#: metafor names its terms after the columns it was given
R_NAMES = {"time": "year_c", "time^2": "year_c2", "intrcpt": "intrcpt"}


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
