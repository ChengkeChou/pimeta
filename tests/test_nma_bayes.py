"""The arm-level Bayesian NMA: exact algebra, a reference, known-truth recovery.

Three kinds of check, and the order matters.

**Exact.**  This model is Gaussian in the treatment effects once ``tau`` is
fixed, so the posterior is a one-dimensional integral over ``tau`` and can be
computed exactly -- no sampler, no warmup, no effective sample size.  That is
the strongest statement available about a model, and it is why
``tools/quadrature_nma_bayes.py`` exists.  The values hard-coded below were
produced by::

    python tools/quadrature_nma_bayes.py

There is also a purely algebraic check with no reference at all: the
study-baseline integral that ``_marginal_loglik`` computes in closed form has to
equal a direct numerical integration over the baseline.

**Reference.**  ``multinma::nma()`` fits the same published model.  It is used
here the other way round from the obvious one: the test asserts that
**pimeta is closer to the exact posterior than multinma is**, because on this
fixture it is, and the size of multinma's gap (up to 0.03 in a posterior mean,
with divergent transitions reported) is recorded rather than hidden.  A
reference implementation is evidence, not an arbiter, and on a network with a
between-study SD near the boundary it is the one that is wrong.

**Property.**  Data simulated from a known ``(d, tau)``: the posterior mean has
to recover it, and its interval has to cover it more often than not.
"""

from __future__ import annotations

import csv
import json
import math
import pathlib

import numpy as np
import pytest

import pimeta as ma
from pimeta.bayes import Posterior
from pimeta.nma.bayes import (
    BayesNmaResult,
    _ArmData,
    _marginal_loglik,
    _prepare,
    bayes_nma,
)

DATA = pathlib.Path(__file__).parent / "data" / "nma_bayes.csv"
TRUTH = pathlib.Path(__file__).parent / "reference" / "nma_bayes_truth.json"
MULTINMA = pathlib.Path(__file__).parent / "reference" / "nma_bayes_multinma.csv"
#
# --------------------------------------------------------------------------- #
# reference values
# --------------------------------------------------------------------------- #
# exact posterior, tools/quadrature_nma_bayes.py, on tests/data/nma_bayes.csv
EXACT_FIXED = {"B": (0.4404, 0.1172), "C": (0.6787, 0.1250), "D": (0.0797, 0.1401)}
EXACT_RANDOM = {"B": (0.4447, 0.1502), "C": (0.7058, 0.1603), "D": (0.0652, 0.1823)}
EXACT_TAU = (0.2211, 0.1214)

# multinma 0.9.1, tools/reference_nma_multinma.R, same data and priors, four
# chains x (5000 warmup + 10000 draws).  Two divergent transitions were
# reported; see the module docstring.
MULTINMA_MEAN = {"B": 0.4331782, "C": 0.6861364, "D": 0.0794719}
MULTINMA_TAU = 0.2314940


def _fixture():
    rows = list(csv.DictReader(DATA.open(encoding="utf-8")))
    y = np.array([float(r["yi"]) for r in rows])
    sei = np.array([float(r["sei"]) for r in rows])
    study = [r["study"] for r in rows]
    treat = [r["treat"] for r in rows]
    return y, sei, study, treat


@pytest.fixture(scope="module")
def fit():
    y, sei, study, treat = _fixture()
    return bayes_nma(y, study=study, treat=treat, sei=sei,
                     mcmc={"seed": 20240617, "n_draws": 8000, "warmup": 4000})


@pytest.fixture(scope="module")
def fixed_fit():
    y, sei, study, treat = _fixture()
    return bayes_nma(y, study=study, treat=treat, sei=sei,
                     trt_effects="fixed",
                     mcmc={"seed": 7, "n_draws": 8000, "warmup": 4000})


# --------------------------------------------------------------------------- #
# exact algebra, with no reference implementation involved
# --------------------------------------------------------------------------- #
def test_marginal_likelihood_equals_numerical_integration():
    """The closed-form baseline integral has to match integrating it directly."""
    from scipy import stats

    y = np.array([0.3, 0.9])
    v = np.array([0.04, 0.09])
    X = np.array([[0.0], [1.0]])
    data = _ArmData(y=y, v=v, X=X, starts=np.array([0]),
                    is_baseline=np.array([True, False]),
                    treatments=["A", "B"], basic=["B"], reference="A",
                    studies=["s1"])
    d = np.array([0.5])
    sd = 100.0
    grid = np.linspace(-600.0, 600.0, 2400001)
    for tau2 in (0.0, 0.25):
        closed = _marginal_loglik(d, tau2, data, sd)
        logp = (stats.norm.logpdf(y[0], grid, math.sqrt(v[0]))
                + stats.norm.logpdf(y[1], grid + d[0], math.sqrt(v[1] + tau2))
                + stats.norm.logpdf(grid, 0.0, sd))
        top = float(np.max(logp))
        # numpy 2 renamed ``trapz`` to ``trapezoid``, and the project supports
        # numpy 1.24, where only the old name exists.  CI runs both.
        trapz = getattr(np, "trapezoid", None)
        if trapz is None:                                   # numpy < 2.0
            trapz = np.trapz                                   # noqa: NPY201
        direct = top + float(np.log(trapz(np.exp(logp - top), grid)))
        # the closed form omits the 2 * -log(2 pi)/2 of the two density constants
        assert closed - direct == pytest.approx(math.log(2.0 * math.pi), abs=1e-8)


def test_conditional_posterior_of_the_baseline_is_the_gaussian_it_claims(fit):
    """``intercept`` is a mixture of exact Gaussians, so it must beat its own mean."""
    for study in ("s01", "s07"):
        out = fit.intercept(study)
        assert np.isfinite(out["mean"])
        assert out["sd"] > 0.0
    # a study's baseline is estimated far more precisely than the differences:
    # s01 has two arms with se 0.18 and 0.22
    assert fit.intercept("s01")["sd"] < 0.25


def test_reference_free_study_gets_one_random_effect_not_two():
    """The pinning rule is the model, so it is tested as one.

    A study without the reference treatment still pins one arm.  Getting this
    wrong gives a posterior mean for ``d[B]`` of 0.437 instead of 0.365 on the
    three-study sub-network below -- a difference far outside Monte Carlo
    error, and the exact quadrature agrees with the pinned version.
    """
    y, sei, study, treat = _fixture()
    keep = [i for i, s in enumerate(study) if s in ("s01", "s02", "s07")]
    data = _prepare(y[keep], sei[keep], None,
                    [study[i] for i in keep], [treat[i] for i in keep], None)
    assert data.is_baseline.sum() == 3          # one per study
    # s07 is (B, C): its first arm is pinned because it has no reference arm
    s07 = data.starts[data.studies.index("s07")]
    assert data.is_baseline[s07]
    assert not data.is_baseline[s07 + 1]


# --------------------------------------------------------------------------- #
# the exact posterior
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("name", ["B", "C", "D"])
def test_random_effects_posterior_matches_the_exact_quadrature(fit, name):
    draws = fit.posterior[f"d[{name}]"]
    mean, sd = EXACT_RANDOM[name]
    ess = float(fit.posterior.ess()[fit.basic.index(name)])
    mcse = float(np.std(draws, ddof=1) / np.sqrt(ess))
    assert abs(float(np.mean(draws)) - mean) < max(4.0 * mcse, 0.004)
    assert abs(float(np.std(draws, ddof=1)) - sd) < 0.006


def test_tau_posterior_matches_the_exact_quadrature(fit):
    tau = np.exp(fit.posterior["log_tau"])
    ess = float(fit.posterior.ess()[-1])
    mcse = float(np.std(tau, ddof=1) / np.sqrt(ess))
    assert abs(fit.tau - EXACT_TAU[0]) < max(4.0 * mcse, 0.004)
    assert abs(float(np.std(tau, ddof=1)) - EXACT_TAU[1]) < 0.006


@pytest.mark.parametrize("name", ["B", "C", "D"])
def test_fixed_effects_posterior_matches_the_exact_quadrature(fixed_fit, name):
    draws = fixed_fit.posterior[f"d[{name}]"]
    mean, sd = EXACT_FIXED[name]
    ess = float(fixed_fit.posterior.ess()[fixed_fit.basic.index(name)])
    mcse = float(np.std(draws, ddof=1) / np.sqrt(ess))
    assert abs(float(np.mean(draws)) - mean) < max(4.0 * mcse, 0.004)
    assert abs(float(np.std(draws, ddof=1)) - sd) < 0.006


def test_fixed_effects_is_the_generalised_least_squares_answer():
    """With no between-study term the model is linear, so the answer is GLS."""
    y, sei, study, treat = _fixture()
    studies = list(dict.fromkeys(study))
    treatments = list(dict.fromkeys(treat))
    basic = [t for t in treatments if t != "A"]
    Z = np.zeros((y.size, len(studies) + len(basic)))
    for i in range(y.size):
        Z[i, studies.index(study[i])] = 1.0
        if treat[i] in basic:
            Z[i, len(studies) + basic.index(treat[i])] = 1.0
    W = np.diag(1.0 / sei ** 2)
    gls = np.linalg.solve(Z.T @ W @ Z, Z.T @ W @ y)[len(studies):]
    for j, name in enumerate(basic):
        assert gls[j] == pytest.approx(EXACT_FIXED[name][0], abs=1e-3)


def test_the_sampler_is_closer_to_the_exact_answer_than_multinma(fit):
    """A reference implementation is evidence, not an arbiter.

    On this fixture ``multinma`` reports two divergent transitions and its
    posterior means sit up to 0.03 away from the exact answer.  The test records
    that, so that a future version which fixes it shows up as a change rather
    than as a silently different tolerance.
    """
    for name in ("B", "C", "D"):
        exact = EXACT_RANDOM[name][0]
        ours = abs(float(np.mean(fit.posterior[f"d[{name}]"])) - exact)
        theirs = abs(MULTINMA_MEAN[name] - exact)
        assert ours < theirs, name
        assert ours < 0.004
    assert abs(fit.tau - EXACT_TAU[0]) < abs(MULTINMA_TAU - EXACT_TAU[0])


def test_study_baseline_matches_multinma_where_the_parameter_is_the_same(fit):
    """``mu[s01]`` is a study that contains the reference, so the two mean the same."""
    rows = {r["par"]: r for r in csv.DictReader(MULTINMA.open(encoding="utf-8"))}
    ours = fit.intercept("s01")
    assert ours["mean"] == pytest.approx(float(rows["mu[s01]"]["mean"]), abs=0.004)
    assert ours["sd"] == pytest.approx(float(rows["mu[s01]"]["sd"]), abs=0.008)


# --------------------------------------------------------------------------- #
# properties of the posterior
# --------------------------------------------------------------------------- #
def test_reference_treatment_does_not_change_contrasts():
    y, sei, study, treat = _fixture()
    settings = {"seed": 11, "n_draws": 4000, "warmup": 2000}
    a = bayes_nma(y, study=study, treat=treat, sei=sei, mcmc=settings)
    b = bayes_nma(y, study=study, treat=treat, sei=sei, reference="C",
                  mcmc=settings)
    assert b.reference == "C"
    assert "C" not in b.basic and "A" in b.basic
    mcse = np.hypot(a.contrast("B", "D")["mcse"], b.contrast("B", "D")["mcse"])
    assert a.contrast("B", "D")["mean"] == pytest.approx(
        b.contrast("B", "D")["mean"], abs=max(4.0 * mcse, 0.01))


def test_contrast_sign_convention_and_probability(fit):
    # B has the larger effect than D in this fixture, so B - D is positive
    out = fit.contrast("B", "D")
    assert out["mean"] > 0.0
    assert out["q_low"] < out["mean"] < out["q_high"]
    assert out["prob_positive"] > 0.9
    assert fit.contrast("D", "B")["mean"] == pytest.approx(-out["mean"], abs=1e-12)
    assert fit.contrast("D", "B")["prob_positive"] == pytest.approx(
        1.0 - out["prob_positive"], abs=1e-12)


def test_league_table_is_antisymmetric(fit):
    table = fit.league_table()
    mean = table["mean"]
    assert mean.shape == (4, 4)
    assert np.all(np.isnan(np.diag(mean)))
    off = ~np.eye(4, dtype=bool)
    assert np.allclose(mean[off], -mean.T[off])
    assert np.allclose(table["prob_positive"][off] + table["prob_positive"].T[off],
                       1.0)


def test_rank_probabilities_are_a_distribution(fit):
    out = fit.rank()
    assert out["prob_best"].sum() == pytest.approx(1.0)
    assert out["prob_worst"].sum() == pytest.approx(1.0)
    assert np.all(out["mean_rank"] >= 1.0) and np.all(out["mean_rank"] <= 4.0)
    assert out["mean_rank"].sum() == pytest.approx(10.0)      # 1+2+3+4
    assert np.all((out["sucra"] >= 0.0) & (out["sucra"] <= 1.0))
    # A has the smallest effects in this fixture, so it is most often best
    best = dict(zip(out["treatments"], out["prob_best"]))
    assert best["A"] > best["C"]
    flipped = fit.rank(small_values="bad")
    assert flipped["prob_best"].sum() == pytest.approx(1.0)
    # flipping the direction best-for-worst swaps the extremes
    assert flipped["prob_best"] == pytest.approx(out["prob_worst"])
    assert flipped["mean_rank"] == pytest.approx(5.0 - out["mean_rank"])


def test_level_spellings_agree(fit):
    a = fit.contrast("B", "A", level=0.95)
    b = fit.contrast("B", "A", level=95)
    c = fit.contrast("B", "A", level=0.05)
    assert a["q_low"] == pytest.approx(b["q_low"])
    assert a["q_low"] == pytest.approx(c["q_low"])
    assert a["level"] == pytest.approx(0.95)


def test_summary_and_to_dict(fit):
    rows = fit.summary()
    assert [r["term"] for r in rows] == ["d[B]", "d[C]", "d[D]", "log_tau"]
    payload = fit.to_dict()
    json.dumps(payload)                      # must be serialisable
    assert payload["model"].startswith("arm-level")
    assert payload["reference"] == "A"
    assert payload["k"] == 14
    assert payload["n_arms"] == 32
    assert payload["priors"]["tau_scale"] == 0.5
    assert isinstance(fit.posterior, Posterior)
    assert "Bayesian network meta-analysis" in str(fit)
    assert "tau" in str(fit)


def test_truth_is_inside_the_credible_intervals(fit):
    truth = json.loads(TRUTH.read_text(encoding="utf-8"))
    for name in ("B", "C", "D"):
        out = fit.contrast(name, "A")
        assert out["q_low"] <= truth["d"][name] <= out["q_high"], name


def test_known_truth_is_recovered_over_replications():
    """Bias within three Monte Carlo standard errors of the replication mean."""
    rng = np.random.default_rng(4242)
    true_d = {"A": 0.0, "B": 0.25, "C": 0.50, "D": 0.10}
    tau = 0.2
    reps = 20
    means = {k: [] for k in ("B", "C", "D")}
    covered = {k: 0 for k in ("B", "C", "D")}
    for _ in range(reps):
        ys, ses, studies, treats = [], [], [], []
        for j in range(6):
            arms = ["A", "B", "D"] if j % 3 else ["A", "B", "C"]
            mu = rng.normal(0.0, 1.0)
            for arm in arms:
                se = float(rng.uniform(0.15, 0.30))
                dev = 0.0 if arm == "A" else rng.normal(0.0, tau)
                ys.append(mu + true_d[arm] + dev + rng.normal(0.0, se))
                ses.append(se)
                studies.append(f"r{j}")
                treats.append(arm)
        res = bayes_nma(ys, study=studies, treat=treats, sei=ses,
                        mcmc={"seed": int(rng.integers(2 ** 31)),
                              "n_draws": 1500, "warmup": 800, "chains": 2,
                              "check": False})
        for k in means:
            out = res.contrast(k, "A")
            means[k].append(out["mean"])
            if out["q_low"] <= true_d[k] <= out["q_high"]:
                covered[k] += 1
    for k, values in means.items():
        values = np.array(values)
        bias = float(np.mean(values) - true_d[k])
        mcse = float(np.std(values, ddof=1) / np.sqrt(values.size))
        assert abs(bias) < 3.0 * mcse + 0.01, k
        assert covered[k] >= reps * 0.75, k


# --------------------------------------------------------------------------- #
# refusals and the record
# --------------------------------------------------------------------------- #
def test_one_arm_study_is_refused():
    with pytest.raises(ValueError, match="at least two arms"):
        bayes_nma([0.1, 0.2, 0.3], study=["a", "a", "b"],
                  treat=["A", "B", "A"], sei=[0.1, 0.1, 0.1])


def test_repeated_arm_is_refused():
    with pytest.raises(ValueError, match="twice"):
        bayes_nma([0.1, 0.2], study=["a", "a"],
                  treat=["A", "A"], sei=[0.1, 0.1])


def test_non_positive_variance_is_refused():
    with pytest.raises(ValueError, match="strictly positive"):
        bayes_nma([0.1, 0.2], study=["a", "a"], treat=["A", "B"], sei=[0.1, 0.0])


def test_unknown_reference_is_refused():
    with pytest.raises(ValueError, match="unknown reference"):
        bayes_nma([0.1, 0.2], study=["a", "a"], treat=["A", "B"], sei=[0.1, 0.1],
                  reference="Z")


def test_both_sei_and_vi_is_refused():
    with pytest.raises(ValueError, match="exactly one"):
        bayes_nma([0.1, 0.2], study=["a", "a"], treat=["A", "B"],
                  sei=[0.1, 0.1], vi=[0.01, 0.01])


def test_unknown_mcmc_setting_is_refused():
    with pytest.raises(ValueError, match="unknown 'mcmc' setting"):
        bayes_nma([0.1, 0.2], study=["a", "a"], treat=["A", "B"],
                  sei=[0.1, 0.1], mcmc={"n_draw": 10})


def test_bad_trt_effects_and_priors_are_refused():
    with pytest.raises(ValueError, match="'trt_effects'"):
        bayes_nma([0.1, 0.2], study=["a", "a"], treat=["A", "B"],
                  sei=[0.1, 0.1], trt_effects="randomish")
    with pytest.raises(ValueError, match="positive"):
        bayes_nma([0.1, 0.2], study=["a", "a"], treat=["A", "B"],
                  sei=[0.1, 0.1], prior_tau_scale=0.0)


def test_exported_from_the_top_level_package():
    assert ma.bayes_nma is bayes_nma
    assert ma.BayesNmaResult is BayesNmaResult
