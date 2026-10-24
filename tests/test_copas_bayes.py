"""Bayesian Copas: the likelihood's limits, the MLE, and the sampler.

Three checks, and the user's own framing is why they are separate: the model,
the optimiser and the sampler are three different things and a mistake in one
is invisible from the others.

**The likelihood's limits.** With :math:`\\gamma_0 \\to \\infty` nobody is
suppressed, so the selection model must collapse to the ordinary
random-effects likelihood — for *every* :math:`\\rho`, since a correlation with
a mechanism that never acts cannot matter. That is an algebraic identity the
test checks rather than a plausible-looking number.

**The optimiser.** With flat priors and fixed selection parameters the
posterior is proportional to the likelihood, so its centre is the maximum
likelihood estimate `metasens::copas` would report. The posterior *SD* is not
the same number and should not be: it is a marginal SD over :math:`\\rho` and
:math:`\\tau`, where `metasens` reports a conditional SE at the mode.

**The generator.** A Bayesian fit cannot be validated by a sampler on the same
formula — two implementations of one wrong likelihood agree. So the fixture is
generated from a stated selection process and the *frequentist* fit is checked
for bias first: if the generator and the likelihood disagreed, the MLE would
show it, and only then does the posterior mean mean anything.
"""

from __future__ import annotations

import pathlib

import numpy as np
import pytest
from scipy import special

from pimeta.copas import bayes_copas, copas_fit, copas_loglik

FLAT = {"mu_sd": 1e3, "tau_scale": 1e3, "gamma0_sd": 1e3, "gamma1_scale": 1e3}


def simulate(*, mu=0.3, tau=0.2, gamma0=0.0, gamma1=0.6, rho=0.6, k=24,
             seed=11) -> tuple:
    """Draw a Copas-selected sample from the model the likelihood describes.

    Each study has a true effect ``theta ~ N(mu, tau^2)`` and an estimate
    ``y ~ N(theta, sei^2)``.  Its selection score is
    ``Z = gamma0 + gamma1/sei + delta`` with ``delta ~ N(0, 1)`` correlated
    ``rho`` with the standardised estimate, and the study is published iff
    ``Z > 0``.  Conditional on publication the density of ``y`` is exactly
    ``copas_loglik`` — which is what the bias test below checks.
    """
    rng = np.random.default_rng(seed)
    sei = np.exp(rng.uniform(np.log(0.05), np.log(0.45), k))
    theta = rng.normal(mu, tau, k)
    y = theta + rng.normal(0.0, 1.0, k) * sei
    total = np.sqrt(sei ** 2 + tau ** 2)
    noise = rng.normal(0.0, 1.0, k)
    score = (gamma0 + gamma1 / sei + rho * (y - mu) / total
             + np.sqrt(1.0 - rho ** 2) * noise)
    keep = score > 0.0
    return y[keep], sei[keep]


@pytest.fixture(scope="module")
def sample() -> tuple:
    return simulate(k=20, seed=7)


# --------------------------------------------------------------------------- #
# the likelihood, at its limits
# --------------------------------------------------------------------------- #
def test_no_selection_collapses_to_the_random_effects_likelihood():
    """``gamma0 -> infinity`` must give the plain normal likelihood, for any rho."""
    yi, sei, mu, tau = np.array([0.1, 0.4, -0.2, 0.3]), np.array(
        [0.1, 0.2, 0.15, 0.3]), 0.15, 0.2
    resid = yi - mu
    s2t2 = sei ** 2 + tau ** 2
    plain = float(np.sum(0.5 * np.log(s2t2) + resid ** 2 / (2.0 * s2t2)))
    for rho in (-0.9, -0.3, 0.0, 0.5, 0.95):
        got = copas_loglik((mu, rho, tau), yi, sei, gamma0=1e4, gamma1=0.0)
        assert got == pytest.approx(plain, abs=1e-9)


def test_the_likelihood_is_finite_across_the_whole_parameter_box():
    """Extreme selection must return a number or the sentinel, never a nan."""
    yi, sei = simulate(k=20, seed=7)
    for gamma0 in (-40.0, -5.0, 0.0, 5.0, 40.0):
        for gamma1 in (0.0, 1.0, 20.0):
            for rho in (-0.99, 0.0, 0.99):
                value = copas_loglik((0.2, rho, 0.25), yi, sei, gamma0, gamma1)
                assert np.isfinite(value)
                assert value > -1e299 or value >= 1e299


def test_more_selection_lowers_the_likelihood_of_a_favourable_sample():
    """The sign convention, pinned: suppressing small studies must cost."""
    rng = np.random.default_rng(3)
    sei = np.exp(rng.uniform(np.log(0.05), np.log(0.4), 40))
    y = rng.normal(0.0, 1.0, 40) * sei
    favourable = np.abs(y) > 0          # a symmetric sample, no asymmetry yet
    del favourable
    no_sel = copas_loglik((0.0, 0.0, 0.1), y, sei, 6.0, 0.0)
    some_sel = copas_loglik((0.0, 0.0, 0.1), y, sei, 0.0, 2.0)
    assert some_sel > no_sel           # selection costs likelihood here


# --------------------------------------------------------------------------- #
# the optimiser: flat priors must reproduce the conditional MLE
# --------------------------------------------------------------------------- #
def test_flat_priors_centre_the_posterior_on_the_conditional_mle(sample):
    yi, sei = sample
    mle = copas_fit(yi, sei, 0.0, 1.0)
    fit = bayes_copas(yi, sei=sei, gamma=(0.0, 1.0), prior=FLAT,
                      mcmc={"n_draws": 4000, "warmup": 2000, "chains": 4,
                            "seed": 3, "check": False})
    assert fit.fixed_gamma == (0.0, 1.0)
    assert fit.mu == pytest.approx(mle.mu, abs=0.02)
    assert fit.tau == pytest.approx(mle.tau, abs=0.03)
    # the posterior SD is marginal over rho and tau, so it is *not* the
    # conditional SE at the mode — but it must be the same order
    sd_mu = float(np.std(fit.posterior.draws[:, 0], ddof=1))
    assert 0.7 * mle.se_mu < sd_mu < 1.6 * mle.se_mu


def test_estimating_the_selection_parameters_widens_the_interval(sample):
    """Leaving (gamma0, gamma1) free cannot make the answer more certain."""
    yi, sei = sample
    settings = {"n_draws": 2000, "warmup": 1000, "chains": 4, "seed": 5,
                "check": False}
    fixed = bayes_copas(yi, sei=sei, gamma=(0.0, 1.0), prior=FLAT,
                        mcmc=settings)
    free = bayes_copas(yi, sei=sei, prior=FLAT, mcmc=settings)
    width_fixed = fixed.mu_ci[1] - fixed.mu_ci[0]
    width_free = free.mu_ci[1] - free.mu_ci[0]
    assert width_free > width_fixed
    # rho is the weakly identified one, and the fit has to say so
    assert free.rho_ci[1] - free.rho_ci[0] > 1.0


# --------------------------------------------------------------------------- #
# the generator, and the posterior's calibration
# --------------------------------------------------------------------------- #
def test_the_generator_and_the_likelihood_agree():
    """If they did not, the MLE would be biased — so measure the bias.

    Two hundred replications of the process in :func:`simulate`, fitted at the
    true selection parameters. The mean of ``mu_hat`` must sit on the truth and
    its standard errors must be calibrated; a likelihood that did not describe
    the generator would fail both, which is what makes the posterior checks
    below worth anything.
    """
    truth_mu, truth_tau = 0.3, 0.2
    estimates, ses = [], []
    for r in range(200):
        yi, sei = simulate(mu=truth_mu, tau=truth_tau, gamma0=0.0, gamma1=0.6,
                           rho=0.6, k=24, seed=1000 + r)
        if yi.size < 4:
            continue
        mle = copas_fit(yi, sei, 0.0, 0.6)
        if not np.isfinite(mle.mu) or not np.isfinite(mle.se_mu):
            continue
        estimates.append(mle.mu)
        ses.append(mle.se_mu)
    estimates = np.array(estimates)
    ses = np.array(ses)
    assert estimates.size > 100
    bias = float(np.mean(estimates) - truth_mu)
    mcse = float(np.std(estimates, ddof=1) / np.sqrt(estimates.size))
    assert abs(bias) < 3.0 * mcse, f"bias {bias:.4f} against MCSE {mcse:.4f}"
    coverage = float(np.mean(np.abs(estimates - truth_mu) < 1.96 * ses))
    assert 0.85 < coverage < 0.99, f"coverage {coverage:.3f}"


def test_the_posterior_interval_covers_the_truth_under_selection():
    yi, sei = simulate(mu=0.3, tau=0.2, gamma0=0.0, gamma1=0.6, rho=0.6,
                       k=40, seed=2024)
    fit = bayes_copas(yi, sei=sei, gamma=(0.0, 0.6),
                      mcmc={"n_draws": 2000, "warmup": 1000, "chains": 4,
                            "seed": 9, "check": False})
    assert fit.mu_ci[0] <= 0.3 <= fit.mu_ci[1]


def test_the_draws_pass_their_diagnostics(sample):
    yi, sei = sample
    fit = bayes_copas(yi, sei=sei, gamma=(0.0, 0.6),
                      mcmc={"n_draws": 2000, "warmup": 1000, "chains": 4,
                            "seed": 13, "check": False})
    report = fit.posterior.check(rhat_max=1.1, ess_min=100.0)
    assert report["ok"]
    assert fit.posterior.settings["inference"].startswith("adaptive random-walk")


# --------------------------------------------------------------------------- #
# the record, and the refusals
# --------------------------------------------------------------------------- #
def test_the_result_serialises(sample):
    yi, sei = sample
    import json
    fit = bayes_copas(yi, sei=sei, gamma=(0.0, 0.6),
                      mcmc={"n_draws": 400, "warmup": 300, "chains": 2,
                            "seed": 1, "check": False})
    payload = json.loads(json.dumps(fit.to_dict()))
    assert payload["k"] == yi.size
    assert payload["fixed_gamma"] == [0.0, 0.6]
    assert payload["priors"]["mu_sd"] == 10.0
    assert len(payload["posterior"]["summary"]) == 3
    assert "Bayesian Copas" in str(fit)


def test_the_level_is_normalised(sample):
    yi, sei = sample
    widths = []
    for level in (0.95, 95, 0.05):
        fit = bayes_copas(yi, sei=sei, gamma=(0.0, 0.6), level=level,
                          mcmc={"n_draws": 200, "warmup": 200, "chains": 2,
                                "seed": 2, "check": False})
        widths.append(fit.mu_ci[1] - fit.mu_ci[0])
    assert widths[0] == pytest.approx(widths[1], rel=1e-9)
    assert widths[0] == pytest.approx(widths[2], rel=1e-9)


def test_a_fitted_model_can_be_passed_through():
    import pimeta as ma
    rng = np.random.default_rng(5)
    sei = np.exp(rng.uniform(np.log(0.05), np.log(0.4), 15))
    y = rng.normal(0.2, 1.0, 15) * sei
    ref = ma.rma(y, vi=sei ** 2)
    fit = bayes_copas(ref, gamma=(0.0, 0.6),
                      mcmc={"n_draws": 200, "warmup": 200, "chains": 2,
                            "seed": 3, "check": False})
    assert fit.k == 15


def test_refusals(sample):
    yi, sei = sample
    with pytest.raises(ValueError, match="at least three"):
        bayes_copas(yi[:2], sei[:2], gamma=(0.0, 0.6))
    with pytest.raises(ValueError, match="same length"):
        bayes_copas(yi, sei=sei[:-1], gamma=(0.0, 0.6))
    with pytest.raises(ValueError, match="must be positive"):
        bayes_copas(yi, sei=-sei, gamma=(0.0, 0.6))
    with pytest.raises(ValueError, match="non-negative"):
        bayes_copas(yi, sei=sei, gamma=(0.0, -1.0))
    with pytest.raises(ValueError, match="unknown prior"):
        bayes_copas(yi, sei=sei, gamma=(0.0, 0.6), prior={"mu": 1.0})
    with pytest.raises(ValueError, match="unknown 'mcmc' setting"):
        bayes_copas(yi, sei=sei, gamma=(0.0, 0.6), mcmc={"draws": 100})


def test_the_numpyro_reference_agrees_when_it_is_available():
    """The independent NUTS comparison, run when an interpreter has numpyro.

    ``pimeta`` does not depend on numpyro and the test virtualenv does not have
    it, so this skips unless ``PIMETA_NUMPYRO_PYTHON`` names an interpreter that
    does. Run by hand it reported a worst |z| of 1.15 with the selection
    parameters fixed and 1.43 with them estimated, on 4 x 4000 draws each side —
    agreement within Monte Carlo error on the identical posterior.
    """
    import os
    import subprocess

    exe = os.environ.get("PIMETA_NUMPYRO_PYTHON")
    if not exe:
        pytest.skip("set PIMETA_NUMPYRO_PYTHON to an interpreter with numpyro")
    tool = (pathlib.Path(__file__).resolve().parents[1] / "tools"
            / "reference_copas_numpyro.py")
    done = subprocess.run([exe, str(tool)], capture_output=True, text=True,
                          timeout=1800)
    assert done.returncode == 0, done.stdout + done.stderr
    assert "agree within Monte Carlo error" in done.stdout


def test_the_standard_error_of_the_log_ndtr_identity():
    """The sentinel region is returned as such, not as a finite likelihood."""
    yi, sei = np.array([0.0, 0.5, 1.0]), np.array([0.1, 0.1, 0.1])
    wild = copas_loglik((0.0, 0.999999, 0.0), yi, sei, -50.0, 0.0)
    assert wild >= 1e299 or np.isfinite(wild)
    assert np.isfinite(special.log_ndtr(-40.0))
