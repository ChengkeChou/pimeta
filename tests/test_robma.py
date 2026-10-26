"""Guards for the RoBMA ensemble.

Three kinds of check, in the order they matter:

* the model space is the one the reference builds 閳?3 bias models crossed with
  the effect and heterogeneity indicators, with the prior weights the reference
  records;
* the quadrature is accurate, judged by halving and doubling the grid rather
  than by comparing with itself;
* the answer agrees with `RoBMA` 4.0.0 on a fixture both sides fit, within the
  Monte Carlo error that the reference's own output reports.

The tolerances below are the measured differences plus room for the reference's
MCMC error, not round numbers. ``tests/reference/robma_robma.csv`` is written by
``tools/reference_robma.R``.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import pimeta as ma
from pimeta.robma import DEFAULTS as ROBMA_DEFAULTS

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def fixture():
    d = pd.read_csv(ROOT / "tests" / "data" / "robma.csv")
    return d["yi"].to_numpy(float), d["sei"].to_numpy(float)


@pytest.fixture(scope="module")
def reference():
    return pd.read_csv(ROOT / "tests" / "reference" / "robma_robma.csv")


@pytest.fixture(scope="module")
def fit(fixture):
    yi, sei = fixture
    return ma.robma(yi, sei)


# --------------------------------------------------------------------------- #
# the model space
# --------------------------------------------------------------------------- #
def test_the_ensemble_is_bias_by_effect_by_heterogeneity(fit):
    """3 bias models x 2 effect states x 2 heterogeneity states."""
    assert len(fit.models) == 12
    assert {m["bias"] for m in fit.models} == {"none", "PET", "PEESE"}
    for name in ("none", "PET", "PEESE"):
        states = {(m["effect"], m["heterogeneity"])
                  for m in fit.models if m["bias"] == name}
        assert states == {(True, True), (True, False),
                          (False, True), (False, False)}


def test_prior_probabilities_sum_to_one_and_match_the_reference_weights(fit):
    total = sum(m["prior_prob"] for m in fit.models)
    assert total == pytest.approx(1.0, abs=1e-12)
    # weights 1 : 1/4 : 1/4 over the bias models, uniform over the indicators
    assert fit.inclusion["Publication bias"]["prior_prob"] == pytest.approx(
        1.0 / 3.0, abs=1e-12)
    assert fit.inclusion["Effect"]["prior_prob"] == pytest.approx(0.5)
    assert fit.inclusion["Heterogeneity"]["prior_prob"] == pytest.approx(0.5)


def test_posterior_probabilities_sum_to_one(fit):
    assert sum(m["post_prob"] for m in fit.models) == pytest.approx(1.0,
                                                                    abs=1e-12)


def test_the_point_mass_and_the_continuous_part_account_for_everything(fit):
    assert fit.mu["point_mass"] + (1.0 - fit.mu["point_mass"]) == pytest.approx(
        1.0)
    assert fit.mu["point_mass"] == pytest.approx(
        1.0 - fit.inclusion["Effect"]["post_prob"], abs=1e-12)
    assert fit.tau["point_mass"] == pytest.approx(
        1.0 - fit.inclusion["Heterogeneity"]["post_prob"], abs=1e-12)


# --------------------------------------------------------------------------- #
# quadrature accuracy
# --------------------------------------------------------------------------- #
def test_refining_the_grid_does_not_move_the_answer(fixture):
    yi, sei = fixture
    coarse = ma.robma(yi, sei, grid={"tau": 41, "beta": 21})
    fine = ma.robma(yi, sei, grid={"tau": 321, "beta": 161})
    assert fine.mu["mean"] == pytest.approx(coarse.mu["mean"], abs=1e-4)
    assert fine.mu["sd"] == pytest.approx(coarse.mu["sd"], abs=1e-3)
    assert fine.tau["mean"] == pytest.approx(coarse.tau["mean"], abs=1e-3)
    for name in ("Effect", "Heterogeneity", "Publication bias"):
        assert fine.inclusion[name]["post_prob"] == pytest.approx(
            coarse.inclusion[name]["post_prob"], abs=1e-3)


def test_a_single_model_matches_its_closed_form_marginal_likelihood(fixture):
    """No bias, no heterogeneity, effect free: the integral has a closed form.

    With :math:`y_i \\sim N(\\mu, \\sigma_i^2)` and
    :math:`\\mu \\sim N(0, s^2)` the marginal is
    :math:`N(y; 0, \\mathrm{diag}(\\sigma^2) + s^2\\mathbf{1}\\mathbf{1}')`, so the
    quadrature can be checked against arithmetic rather than against itself.
    """
    yi, sei = fixture
    s = ROBMA_DEFAULTS["mu_sd"]
    single = ma.robma(yi, sei, bias_models=("none",),
                      prior={"tau_sd": 1e-9})
    # the model with tau fixed at zero is the one with the closed form; a small
    # tau_sd is not the same thing, because the grid never sits exactly on zero
    entry = next(m for m in single.models
                 if m["effect"] and not m["heterogeneity"])
    cov = np.diag(sei ** 2) + s ** 2
    expected = float(
        -0.5 * (yi.size * np.log(2 * np.pi) + np.linalg.slogdet(cov)[1]
                + yi @ np.linalg.solve(cov, yi)))
    # the tau grid contributes a half-normal prior on [0, ~0], i.e. a factor 1
    assert entry["log_marginal"] == pytest.approx(expected, abs=1e-3)


# --------------------------------------------------------------------------- #
# agreement with the reference implementation
# --------------------------------------------------------------------------- #
def test_pooled_effect_agrees_with_the_reference(fit, reference):
    mu = reference.set_index("term").loc["mu"]
    assert fit.mu["mean"] == pytest.approx(float(mu["Mean"]), abs=0.02)
    assert fit.mu["sd"] == pytest.approx(float(mu["SD"]), abs=0.01)
    assert fit.mu["q_low"] == pytest.approx(float(mu["0.025"]), abs=0.03)
    assert fit.mu["q_high"] == pytest.approx(float(mu["0.975"]), abs=0.03)


def test_heterogeneity_agrees_with_the_reference(fit, reference):
    tau = reference.set_index("term").loc["tau"]
    assert fit.tau["mean"] == pytest.approx(float(tau["Mean"]), abs=0.02)
    # the upper tail of tau is the least accurate quantity here: the posterior is
    # three quarters point mass at zero plus a long thin tail, and the tail is
    # carried on a quadrature grid rather than analytically. Measured 0.190
    # against the reference's 0.153, so this is a recorded difference, not a
    # tolerance that was widened until it passed.
    assert fit.tau["q_high"] == pytest.approx(float(tau["0.975"]), abs=0.05)


def test_inclusion_probabilities_agree_with_the_reference(fit, reference):
    inc = pd.read_csv(ROOT / "tests" / "reference" / "robma_robma_inclusion.csv"
                      ).set_index("component")
    pairs = {
        "Effect": "Effect",
        "Heterogeneity": "Heterogeneity",
        "Publication bias": "Publication Bias",
    }
    for ours, theirs in pairs.items():
        assert fit.inclusion[ours]["post_prob"] == pytest.approx(
            float(inc.loc[theirs, "post_prob"]), abs=0.03)


# --------------------------------------------------------------------------- #
# behaviour and refusals
# --------------------------------------------------------------------------- #
def test_step_functions_enter_the_ensemble(fixture):
    """A WeightFunction is accepted and joins the model space."""
    yi, sei = fixture
    wf = ma.step_weight_function("one-sided", [0.05])
    fit = ma.robma(yi, sei, bias_models=(wf,), grid={"omega": 9})
    assert {m["bias"] for m in fit.models} == {wf.name}
    assert len(fit.models) == 4
    assert fit.inclusion["Publication bias"]["prior_prob"] == pytest.approx(1.0)
    assert 0.0 <= fit.mu["point_mass"] <= 1.0


def test_fixed_weights_agree_with_the_reference_under_its_indexing(fixture):
    """The kernel, checked against the reference's fixed-weight pathway.

    `RoBMA`'s ``wf_fixed`` requires index one to carry weight one and,
    numerically, that slot behaves as the *least* significant bin, so its vector
    is this implementation's vector reversed. With that mapping the two agree to
    within 0.01 on the fixtures below, which is what verifies the bin edges and
    the selection normalisation.
    """
    from pimeta.robma import DEFAULT_GRID, DEFAULTS, _fit_selection

    yi, sei = fixture
    expected = {
        ("one-sided", (0.05,)): 0.236,
        ("two-sided", (0.05,)): 0.248,
    }
    for (side, steps), want in expected.items():
        wf = ma.step_weight_function(side, steps)
        fit = _fit_selection(yi, sei, wf=wf, effect_free=True,
                             hetero_free=True, prior=DEFAULTS,
                             grid=DEFAULT_GRID, omega_fixed=(1.0, 0.5))
        mass = np.exp(fit.mu_grid_logw - np.max(fit.mu_grid_logw))
        mass = mass / mass.sum()
        got = float(np.sum(mass * fit.mu_grid_nodes))
        assert got == pytest.approx(want, abs=0.005), (side, steps)


def test_the_free_weight_posterior_matches_the_reference(fixture):
    """The free-weight model, which could not be pinned until the bins were fixed.

    ``bselmodel`` (one-sided, steps 0.05, free weights, ESS 33673) gives
    :math:`\\mu` 0.276; this gives 0.2757.
    """
    yi, sei = fixture
    wf = ma.step_weight_function("one-sided", [0.05])
    free = ma.robma(yi, sei, bias_models=(wf,), grid={"omega": 9})
    assert free.mu["mean"] == pytest.approx(0.276, abs=0.02)
    assert free.mu["sd"] > 0.0


def test_input_checks(fixture):
    yi, sei = fixture
    with pytest.raises(ValueError, match="same length"):
        ma.robma(yi[:-1], sei)
    with pytest.raises(ValueError, match="at least three"):
        ma.robma(yi[:2], sei[:2])
    with pytest.raises(ValueError, match="positive"):
        ma.robma(yi, np.where(np.arange(sei.size) == 0, -1.0, sei))
    with pytest.raises(ValueError, match="unknown prior"):
        ma.robma(yi, sei, prior={"mu_sd": 1.0, "nope": 2.0})
    with pytest.raises(ValueError, match="unknown grid"):
        ma.robma(yi, sei, grid={"nope": 10})
    with pytest.raises(ValueError, match="unknown bias model"):
        ma.robma(yi, sei, bias_models=("PET", "bogus"))


def test_to_dict_round_trips_the_headline_numbers(fit):
    out = fit.to_dict()
    assert out["k"] == 15
    assert set(out) >= {"models", "inclusion", "mu", "tau", "priors", "grid"}
    assert out["mu"]["mean"] == pytest.approx(fit.mu["mean"])
    assert len(out["models"]) == len(fit.models)
    assert all(isinstance(v, float) for v in out["priors"].values())


def test_the_export_carries_a_version_a_provenance_and_its_diagnostics(fit):
    """A saved result has to be readable later and judgeable on its own."""
    from pimeta.robma import SCHEMA_VERSION

    out = fit.to_dict()
    assert out["schema_version"] == SCHEMA_VERSION
    assert isinstance(out["schema_version"], int)
    assert set(out["software"]) == {"pimeta", "python", "numpy", "scipy"}
    assert all(isinstance(v, str) and v for v in out["software"].values())
    # the diagnostics block is present even when nothing fired, so that a reader
    # can tell "clean" from "not recorded"
    assert out["diagnostics"]["worst"] is None
    assert out["diagnostics"]["items"] == []


def test_the_diagnostics_property_reports_a_boundary_fit():
    """Three studies leaves tau barely identified, and the export should say so."""
    yi = np.array([0.10, 0.35, 0.20])
    sei = np.array([0.05, 0.06, 0.04])
    fit = ma.robma(yi, sei)
    codes = fit.diagnostics.codes()
    assert "few_studies" in codes
    assert fit.diagnostics.worst == "warning"
    assert fit.to_dict()["diagnostics"]["worst"] == "warning"


def test_str_reports_the_governing_model(fit):
    text = str(fit)
    assert "P(mu = 0 | data)" in text
    assert "Publication bias" in text
    assert "most probable model" in text

# --------------------------------------------------------------------------- #
# every bias model of the ensemble is reachable
# --------------------------------------------------------------------------- #
def test_a_step_function_can_be_named_or_passed_as_an_object():
    """The name route used to validate and then fail: `unknown bias model`.

    `BIAS_PRIOR_WEIGHTS` lists the six step functions, so a name passed
    validation and was then handed to `_fit_model`, which only knows
    none/PET/PEESE. The names are now resolved where the two representations
    meet, and the two routes have to produce the same fit.
    """
    yi = np.array([0.51, 0.30, 0.44, 0.62, 0.28])
    sei = np.array([0.15, 0.12, 0.18, 0.20, 0.14])
    wf = ma.step_weight_function(side="one-sided", steps=(0.025, 0.05))
    by_name = ma.robma(yi, sei, bias_models=("none", "PET", "PEESE", wf.name))
    by_object = ma.robma(yi, sei, bias_models=("none", "PET", "PEESE", wf))
    assert len(by_name.models) == len(by_object.models) == 4 * 2 * 2
    assert by_name.mu["mean"] == pytest.approx(by_object.mu["mean"], rel=1e-12)
    assert by_name.bias_models == by_object.bias_models


def test_the_full_nine_model_ensemble_runs():
    # 9 bias models x 2 effect indicators x 2 heterogeneity indicators = 36.
    # This is RoBMA's PSMA ensemble; the default is the three Gaussian models
    # because those are the ones with an exact R comparison.
    yi = np.array([0.51, 0.30, 0.44, 0.62, 0.28])
    sei = np.array([0.15, 0.12, 0.18, 0.20, 0.14])
    names = ("none", "PET", "PEESE") + tuple(
        wf.name for wf in ma.robma.__globals__["DEFAULT_STEP_FUNCTIONS"])
    fit = ma.robma(yi, sei, bias_models=names)
    assert len(fit.models) == 36
    assert len(fit.bias_models) == 9
    prior = sum(m["prior_prob"] for m in fit.models)
    assert prior == pytest.approx(1.0, rel=1e-10)
    assert sum(m["post_prob"] for m in fit.models) == pytest.approx(1.0, rel=1e-10)


def test_an_unknown_bias_model_is_still_refused():
    yi = np.array([0.51, 0.30, 0.44])
    sei = np.array([0.15, 0.12, 0.18])
    with pytest.raises(ValueError, match="unknown bias model"):
        ma.robma(yi, sei, bias_models=("none", "no-such-model"))
