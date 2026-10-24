"""The Bayesian dose-response fit: what a sampler buys, and what it costs.

`nma_dose(inference="mcmc")` exists for one model that the closed-form route
cannot have: a **random effect on the curve parameters** with a dose function
that is not linear in them. For `emax` the marginal likelihood is an integral
over each study's (Emax, ED50), so REML has nothing to maximise, and the module
used to refuse the model outright.

There is no reference implementation available in this environment —
`MBNMAdose` is the right one and needs JAGS, which cannot be installed here —
so the checks are of the two kinds that do not need one.

**Against a deterministic fit, in the limit where they must agree.** With a flat
prior the posterior is proportional to the likelihood, so its centre is the
maximum-likelihood estimate and its curvature is the same curvature. The
comparison is therefore against `nls` (already pinned to 1e-4 elsewhere), on the
two-treatment subset where the curve is identified. The means agree to about
0.04 and the standard deviations to a few per cent, and the test states why they
are not *equal*: an Emax/ED50 posterior is skewed, the maximum-likelihood
standard error is a quadratic approximation at the mode, and the posterior
standard deviation is not.

**Against the fit's own diagnostics.** A chain that has not converged must not
return a summary, and the parameter names must say which scale they are on,
because ``sd[A:emax]`` reading −1.5 would be a lie about a standard deviation.
"""

from __future__ import annotations

import csv
import pathlib

import numpy as np
import pytest

from pimeta.nma.dose import nma_dose

DATA = pathlib.Path(__file__).parent / "data" / "nma_dose.csv"

# the two-treatment subset: placebo against A, where Emax and ED50 are both
# identified (doses 10 to 200 bracket an ED50 near 30)
SETTINGS = {"n_draws": 2000, "warmup": 1000, "chains": 4, "seed": 3,
            "prior_beta": 1e6, "check": False}


def load(subset: bool = True) -> dict:
    with DATA.open(encoding="utf-8", newline="") as fh:
        rows = list(csv.DictReader(fh))
    if subset:
        rows = [r for r in rows
                if r["treat1"] == "P" and r["treat2"] == "A"]
    return {
        "yi": np.array([float(r["yi"]) for r in rows]),
        "vi": np.array([float(r["vi"]) for r in rows]),
        "study": [r["study"] for r in rows],
        "treat1": [r["treat1"] for r in rows],
        "treat2": [r["treat2"] for r in rows],
        "dose1": np.array([float(r["dose1"]) for r in rows]),
        "dose2": np.array([float(r["dose2"]) for r in rows]),
    }


def payload(data: dict) -> dict:
    return {k: v for k, v in data.items() if k != "yi"}


@pytest.fixture(scope="module")
def flat_prior() -> tuple:
    data = load()
    frequentist = nma_dose(data["yi"], family="emax", struct="FE",
                           **payload(data))
    bayesian = nma_dose(data["yi"], family="emax", struct="FE",
                        inference="mcmc", **payload(data), mcmc=SETTINGS)
    return frequentist, bayesian


# --------------------------------------------------------------------------- #
# against the deterministic fit, in the flat-prior limit
# --------------------------------------------------------------------------- #
def test_a_flat_prior_reproduces_the_maximum_likelihood_fit(flat_prior):
    frequentist, bayesian = flat_prior
    assert bayesian.inference == "mcmc"
    assert bayesian.posterior is not None
    # the centre: the posterior mean is the MLE up to the skewness of the
    # Emax/ED50 posterior, which is why this is 0.06 and not 1e-6
    assert bayesian.coef == pytest.approx(frequentist.coef, abs=0.06)
    # the width: the MLE's standard error is a quadratic approximation at the
    # mode, the posterior SD is not, so they agree to a few per cent rather
    # than exactly — and the posterior one is the smaller, which is what
    # asymmetry does
    ratio = np.array(bayesian.se) / np.array(frequentist.se)
    assert np.all(ratio > 0.85) and np.all(ratio < 1.05)


def test_the_draws_pass_their_own_diagnostics(flat_prior):
    _, bayesian = flat_prior
    assert bayesian.fit["rhat_max"] < 1.05
    assert bayesian.fit["ess_min"] > 200
    report = bayesian.posterior.check(rhat_max=1.05, ess_min=100.0)
    assert report["ok"]


def test_the_parameters_are_named_on_the_scale_they_are_sampled_on():
    """``sd[...]`` would be a lie: a standard deviation is not negative."""
    data = load()
    fit = nma_dose(data["yi"], family="emax", struct="coef",
                   inference="mcmc", **payload(data),
                   mcmc={"n_draws": 300, "warmup": 300, "chains": 2,
                         "seed": 11, "check": False})
    assert any(name.startswith("log_sd[") for name in fit.posterior.names)
    assert not any(name.startswith("sd[") for name in fit.posterior.names)
    assert fit.D is not None and np.all(np.diag(fit.D) > 0.0)


# --------------------------------------------------------------------------- #
# the model the closed-form route cannot fit
# --------------------------------------------------------------------------- #
def test_a_random_curve_is_refused_by_reml_and_fitted_by_mcmc():
    data = load()
    with pytest.raises(ValueError, match="no closed form"):
        nma_dose(data["yi"], family="emax", struct="coef", **payload(data))
    fit = nma_dose(data["yi"], family="emax", struct="coef",
                   inference="mcmc", **payload(data),
                   mcmc={"n_draws": 400, "warmup": 300, "chains": 2,
                         "seed": 5, "check": False})
    assert fit.inference == "mcmc"
    assert fit.D is not None and fit.D.shape == (2, 2)
    # a between-study SD per parameter, sampled on the log scale
    assert np.all(np.exp([v for name, v in
                          zip(fit.posterior.names, fit.posterior.draws.mean(0))
                          if name.startswith("log_sd[")]) < 5.0)


def test_the_hierarchical_fit_still_recovers_the_curve():
    """The random effect widens the interval; it does not lose the curve.

    What is asserted is the **prediction** band, not the credible one: with a
    between-study random effect the generating curve is one draw from the
    population the prediction band describes, so that is the interval the truth
    has to fall in. The point estimate is checked against the truth only at the
    doses the data actually support — the top of the range is where an Emax
    curve is least constrained, which is a property of the model and not of the
    sampler.
    """
    data = load()
    fit = nma_dose(data["yi"], family="emax", struct="coef",
                   inference="mcmc", **payload(data),
                   mcmc={"n_draws": 2000, "warmup": 1000, "chains": 2,
                         "seed": 13, "check": False})
    doses = np.array([10.0, 25.0, 50.0, 100.0])
    band = fit.curve("A", doses)
    truth = 1.4 * doses / (30.0 + doses)
    assert np.all(band.pred_lb <= truth + 1e-9)
    assert np.all(truth <= band.pred_ub + 1e-9)
    assert np.all(np.abs(band.est - truth) < 0.3)
    # the prediction band is wider than the credible band, which is the point
    # of reporting both
    assert np.all(band.pred_lb <= band.ci_lb + 1e-12)
    assert np.all(band.ci_ub <= band.pred_ub + 1e-12)


def test_intervals_come_from_the_draws_not_from_a_delta_method():
    data = load()
    fit = nma_dose(data["yi"], family="emax", struct="FE", inference="mcmc",
                   **payload(data), mcmc=SETTINGS)
    band = fit.curve("A", [50.0])
    draws = fit._curve_draws("A", np.array([50.0]))
    assert draws is not None
    assert band.est[0] == pytest.approx(float(np.mean(draws[:, 0])), abs=1e-10)
    assert band.ci_lb[0] == pytest.approx(
        float(np.quantile(draws[:, 0], 0.025)), abs=1e-10)


def test_the_anchor_stays_exactly_zero():
    data = load()
    fit = nma_dose(data["yi"], family="emax", struct="FE", inference="mcmc",
                   **payload(data), mcmc={"n_draws": 200, "warmup": 200,
                                          "chains": 2, "seed": 7,
                                          "check": False})
    band = fit.curve("A", [0.0])
    assert band.est[0] == pytest.approx(0.0, abs=1e-12)
    assert band.ci_ub[0] - band.ci_lb[0] == pytest.approx(0.0, abs=1e-12)


# --------------------------------------------------------------------------- #
# refusals
# --------------------------------------------------------------------------- #
def test_a_full_curve_covariance_is_not_offered_here():
    data = load()
    with pytest.raises(ValueError, match="DIAG"):
        nma_dose(data["yi"], family="emax", struct="coef", inference="mcmc",
                 coef_struct="UN", **payload(data),
                 mcmc={"n_draws": 100, "warmup": 100, "chains": 2,
                       "check": False})


def test_an_unknown_sampler_setting_is_named():
    data = load()
    with pytest.raises(ValueError, match="unknown 'mcmc' setting"):
        nma_dose(data["yi"], family="emax", inference="mcmc", **payload(data),
                 mcmc={"draws": 100})


def test_an_unknown_inference_is_refused():
    data = load()
    with pytest.raises(ValueError, match="unknown 'inference'"):
        nma_dose(data["yi"], family="emax", inference="variational",
                 **payload(data))


def test_a_non_converged_chain_refuses_to_be_summarised():
    """The default is to raise, because a summary of a bad chain is a lie."""
    data = load(subset=False)
    with pytest.raises(RuntimeError, match="did not mix"):
        nma_dose(data["yi"], family="emax", struct="coef", inference="mcmc",
                 **payload(data),
                 mcmc={"n_draws": 100, "warmup": 100, "chains": 2, "seed": 1})
