"""The MCMC engine: analytic posteriors, diagnostics that must not lie.

There is no reference implementation to compare a sampler against, so the
checks are of three different kinds.

**Against an exact posterior.**  A Gaussian target and a beta-binomial have
closed-form answers, so the draws can be judged by whether they reproduce them
within Monte Carlo error rather than by whether they look plausible.

**Against the diagnostics' own failure modes.**  A diagnostic is only worth
having if it fires when it should *and* stays quiet when it should not, so the
tests construct both: chains started far apart, which ``R-hat`` must catch, and
a skewed posterior, where the raw variance-ratio version over-reports and the
rank-normalised one does not.

**Against the refusals.**  One chain cannot report convergence, four draws
cannot be split, and a log density that returns ``nan`` is a bug in the caller
rather than something to average over.
"""

from __future__ import annotations

import numpy as np
import pytest

from pimeta.bayes import (
    Posterior,
    ess_bulk,
    rhat,
    sample_log_posterior,
)

# a 3-parameter target with a known covariance, used by several tests
COV = np.array([[1.0, 0.6, 0.0],
                [0.6, 1.0, 0.2],
                [0.0, 0.2, 1.0]])
PREC = np.linalg.inv(COV)


def gaussian_logp(theta):
    return -0.5 * float(np.asarray(theta) @ PREC @ np.asarray(theta))


@pytest.fixture(scope="module")
def gaussian() -> Posterior:
    return sample_log_posterior(gaussian_logp, np.zeros(3),
                                names=["a", "b", "c"], init_sd=0.5,
                                n_draws=4000, warmup=2000, chains=4, seed=11)


# --------------------------------------------------------------------------- #
# against an exact posterior
# --------------------------------------------------------------------------- #
def test_a_gaussian_target_is_reproduced(gaussian):
    draws = gaussian.draws
    # the mean of 16000 correlated draws: compare through the Monte Carlo error
    rows = {r["term"]: r for r in gaussian.summary()}
    for name in ("a", "b", "c"):
        assert rows[name]["mean"] == pytest.approx(0.0, abs=4 * rows[name]["mcse"])
        assert rows[name]["sd"] == pytest.approx(1.0, rel=0.05)
    assert np.cov(draws, rowvar=False) == pytest.approx(COV, abs=0.06)


def test_a_beta_binomial_matches_the_conjugate_posterior():
    prior_a, prior_b, successes, trials = 2.0, 3.0, 7.0, 20.0
    post_a, post_b = prior_a + successes, prior_b + trials - successes

    def logpost(theta):
        p = float(theta[0])
        if not 0.0 < p < 1.0:
            return -np.inf
        return (post_a - 1.0) * np.log(p) + (post_b - 1.0) * np.log1p(-p)

    draws = sample_log_posterior(logpost, [0.3], names=["p"], n_draws=4000,
                                 warmup=1000, init_sd=0.2, chains=4, seed=5)
    from scipy import stats as st

    exact_mean = post_a / (post_a + post_b)
    exact_sd = np.sqrt(post_a * post_b / ((post_a + post_b) ** 2
                                          * (post_a + post_b + 1.0)))
    row = draws.summary()[0]
    assert row["mean"] == pytest.approx(exact_mean, abs=4 * row["mcse"])
    assert row["sd"] == pytest.approx(exact_sd, rel=0.05)
    lo, hi = draws.ci("p")
    assert (lo, hi) == pytest.approx(
        (st.beta.ppf(0.025, post_a, post_b), st.beta.ppf(0.975, post_a, post_b)),
        abs=0.02)


def test_a_flat_prior_leaves_the_likelihood_in_charge():
    """With a flat prior the posterior mode is the MLE of a normal mean."""
    data = np.array([0.4, 1.1, 0.7, 0.9, 1.3])
    known_sd = 0.5

    def logpost(theta):
        mu = float(theta[0])
        return float(np.sum(-0.5 * ((data - mu) / known_sd) ** 2))

    draws = sample_log_posterior(logpost, [0.0], names=["mu"], n_draws=4000,
                                 warmup=1000, init_sd=0.3, chains=4, seed=7)
    exact_mean = float(np.mean(data))
    exact_sd = known_sd / np.sqrt(data.size)
    row = draws.summary()[0]
    assert row["mean"] == pytest.approx(exact_mean, abs=4 * row["mcse"])
    assert row["sd"] == pytest.approx(exact_sd, rel=0.05)


# --------------------------------------------------------------------------- #
# the diagnostics, tested against their own failure modes
# --------------------------------------------------------------------------- #
def test_rhat_catches_chains_that_start_far_apart():
    """Chains that never meet must not look converged."""
    def logpost(theta):
        return gaussian_logp(np.asarray(theta) * 0.05)

    stuck = sample_log_posterior(
        logpost, [[-12.0, 0.0, 0.0], [12.0, 0.0, 0.0]],
        names=["a", "b", "c"], n_draws=400, warmup=50, chains=2,
        init_sd=0.01, adapt=False, seed=3)
    assert np.max(rhat(stuck.chains)) > 1.2
    with pytest.raises(RuntimeError, match="did not mix"):
        stuck.check()


def test_rhat_is_rank_based_and_so_ignores_a_monotone_rescaling():
    """A rank statistic cannot change when the draws are transformed monotonically.

    That is the whole point of ranking first: a heavy right tail, which is what
    makes the raw variance-ratio version over-report, is exactly a monotone
    rescaling.  The raw version is computed here to show that it *does* move.
    """
    rng = np.random.default_rng(21)
    chains = np.exp(rng.standard_normal((4, 1000, 1)))   # heavy right tail

    def raw_rhat(x):
        n_chains, n_draws, _ = x.shape
        between = n_draws * float(np.var(np.mean(x[:, :, 0], axis=1), ddof=1))
        within = float(np.mean(np.var(x[:, :, 0], axis=1, ddof=1)))
        var_plus = (n_draws - 1.0) / n_draws * within + between / n_draws
        return float(np.sqrt(var_plus / within))

    cubed = chains ** 3
    assert rhat(chains) == pytest.approx(rhat(cubed), abs=1e-12)
    assert raw_rhat(chains) != pytest.approx(raw_rhat(cubed), abs=1e-9)
    assert np.max(rhat(chains)) < 1.05


def test_ess_of_independent_draws_is_about_the_sample_size():
    rng = np.random.default_rng(2)
    iid = rng.standard_normal((4, 2000, 2))
    ess = ess_bulk(iid)
    assert np.all(ess > 0.7 * 8000)
    assert np.all(ess < 1.3 * 8000)


def test_ess_detects_autocorrelation():
    rng = np.random.default_rng(3)
    n, phi = 4000, 0.9
    x = np.empty((1, n, 1))
    x[0, 0, 0] = 0.0
    for i in range(1, n):
        x[0, i, 0] = phi * x[0, i - 1, 0] + rng.standard_normal()
    ess = ess_bulk(x)[0]
    assert ess < n / 5.0
    assert ess == pytest.approx(n * (1 - phi) / (1 + phi), rel=0.6)


def test_a_well_mixed_fit_passes_its_own_check(gaussian):
    report = gaussian.check()
    assert report["ok"]
    assert report["rhat_max"] < 1.01
    assert report["ess_min"] > 100


def test_the_check_can_be_asked_to_report_instead_of_raise(gaussian):
    report = gaussian.check(rhat_max=0.5, warn_only=True)
    assert not report["ok"]
    assert any("R-hat" in p for p in report["problems"])


# --------------------------------------------------------------------------- #
# the sampler's contract
# --------------------------------------------------------------------------- #
def test_the_same_seed_gives_the_same_draws():
    first = sample_log_posterior(gaussian_logp, np.zeros(3), n_draws=200,
                                 warmup=100, chains=2, seed=42)
    again = sample_log_posterior(gaussian_logp, np.zeros(3), n_draws=200,
                                 warmup=100, chains=2, seed=42)
    other = sample_log_posterior(gaussian_logp, np.zeros(3), n_draws=200,
                                 warmup=100, chains=2, seed=43)
    assert np.array_equal(first.chains, again.chains)
    assert not np.array_equal(first.chains, other.chains)


def test_adaptation_is_what_finds_the_scale():
    """A proposal that is far too small cannot mix; adaptation fixes that."""
    frozen = sample_log_posterior(gaussian_logp, np.zeros(3), n_draws=500,
                                  warmup=500, chains=2, init_sd=0.002,
                                  adapt=False, seed=8)
    adapted = sample_log_posterior(gaussian_logp, np.zeros(3), n_draws=500,
                                   warmup=500, chains=2, init_sd=0.002,
                                   adapt=True, seed=8)
    assert np.mean(frozen.accept) > 0.95        # it never leaves the start
    assert 0.05 < np.mean(adapted.accept) < 0.6
    assert np.std(adapted.draws[:, 0]) > np.std(frozen.draws[:, 0])


def test_a_target_that_raises_is_treated_as_a_rejected_move():
    """A covariance that is not positive definite is a rejection, not a crash."""
    def logpost(theta):
        x = np.asarray(theta)
        if x[0] < 0.0:
            raise np.linalg.LinAlgError("outside the support")
        return -0.5 * float(x @ x)

    draws = sample_log_posterior(logpost, [1.0], names=["x"], n_draws=4000,
                                 warmup=2000, chains=3, init_sd=0.4, seed=17)
    assert np.all(draws["x"] >= 0.0)
    report = draws.check(rhat_max=1.05)
    assert report["ok"]


def test_a_bad_start_is_reported():
    def logpost(theta):
        return float("-inf")

    with pytest.raises(ValueError, match="inside the support"):
        sample_log_posterior(logpost, [0.0], n_draws=100, warmup=10,
                             chains=2, seed=1)


def test_refusals():
    with pytest.raises(ValueError, match="at least 2"):
        sample_log_posterior(gaussian_logp, np.zeros(3), chains=1)
    with pytest.raises(ValueError, match="at least 4"):
        sample_log_posterior(gaussian_logp, np.zeros(3), n_draws=3, chains=2)
    with pytest.raises(ValueError, match="'names' has"):
        sample_log_posterior(gaussian_logp, np.zeros(3), names=["a"], chains=2)
    with pytest.raises(ValueError, match="positive"):
        sample_log_posterior(gaussian_logp, np.zeros(3), init_sd=0.0, chains=2)
    with pytest.raises(ValueError, match="vector, a matrix"):
        sample_log_posterior(gaussian_logp, np.zeros((4, 2, 2)), chains=2)
    with pytest.raises(TypeError, match="single number"):
        sample_log_posterior(lambda t: [1.0, 2.0], np.zeros(3), chains=2)


def test_an_integer_valued_density_is_accepted():
    calls = {"n": 0}

    def logpost(theta):
        calls["n"] += 1
        return -1        # an int, not a float

    draws = sample_log_posterior(logpost, [0.0], n_draws=20, warmup=10,
                                 chains=2, seed=1)
    assert draws.n_draws == 20
    assert calls["n"] > 0


# --------------------------------------------------------------------------- #
# block updates
# --------------------------------------------------------------------------- #
def hierarchical():
    """A small hierarchical model: group means with a common parent."""
    rng = np.random.default_rng(4)
    n_groups, per_group = 6, 5
    y = rng.normal(rng.normal(0.0, 1.0, n_groups)[:, None], 1.0,
                   (n_groups, per_group))

    def logpost(theta):
        parent, log_tau = float(theta[0]), float(theta[1])
        tau = np.exp(log_tau)
        means = np.asarray(theta[2:])
        value = -0.5 * (parent / 5.0) ** 2 - 0.5 * (log_tau / 2.0) ** 2
        value += float(np.sum(-0.5 * ((means - parent) / tau) ** 2 - log_tau))
        value += float(np.sum(-0.5 * (y - means[:, None]) ** 2))
        return value

    start = np.concatenate([[0.0, 0.0], np.zeros(n_groups)])
    schedule = ([np.array([0, 1])]
                + [np.array([2 + j]) for j in range(n_groups)])
    return logpost, start, schedule


def test_blocks_give_the_same_posterior_on_a_gaussian():
    """Blocking changes how the chain moves, not what it converges to."""
    whole = sample_log_posterior(gaussian_logp, np.zeros(3), n_draws=3000,
                                 warmup=1000, chains=2, init_sd=0.5, seed=31)
    split = sample_log_posterior(gaussian_logp, np.zeros(3), n_draws=3000,
                                 warmup=1000, chains=2, init_sd=0.5, seed=31,
                                 blocks=[np.array([0]), np.array([1, 2])])
    for j in range(3):
        lo, hi = split.ci(str(whole.names[j]))
        assert lo <= float(np.mean(whole.draws[:, j])) <= hi
        assert split.draws[:, j].std() == pytest.approx(
            whole.draws[:, j].std(), rel=0.15)


def test_blocks_mix_a_hierarchical_model_better():
    """A global random walk struggles where a per-group walk does not."""
    logpost, start, schedule = hierarchical()
    whole = sample_log_posterior(logpost, start, n_draws=800, warmup=400,
                                 chains=2, init_sd=0.3, seed=9)
    split = sample_log_posterior(logpost, start, n_draws=800, warmup=400,
                                 chains=2, init_sd=0.3, seed=9,
                                 blocks=schedule)
    assert np.min(split.ess()) > np.min(whole.ess())
    assert np.max(split.rhat()) < np.max(whole.rhat())


def test_the_block_schedule_is_validated():
    with pytest.raises(ValueError, match="cannot be empty"):
        sample_log_posterior(gaussian_logp, np.zeros(3), chains=2,
                             blocks=[np.array([], dtype=int)])
    with pytest.raises(ValueError, match="disjoint"):
        sample_log_posterior(gaussian_logp, np.zeros(3), chains=2,
                             blocks=[np.array([0, 1]), np.array([1, 2])])
    with pytest.raises(ValueError, match="outside"):
        sample_log_posterior(gaussian_logp, np.zeros(3), chains=2,
                             blocks=[np.array([0, 1, 2, 5])])
    with pytest.raises(ValueError, match="never updated"):
        sample_log_posterior(gaussian_logp, np.zeros(3), chains=2,
                             blocks=[np.array([0, 1])])


# --------------------------------------------------------------------------- #
# the container
# --------------------------------------------------------------------------- #
def test_the_posterior_reports_everything_a_record_needs(gaussian):
    payload = gaussian.to_dict()
    assert payload["names"] == ["a", "b", "c"]
    assert payload["n_chains"] == 4 and payload["n_draws"] == 4000
    assert len(payload["summary"]) == 3
    assert payload["settings"]["inference"].startswith("adaptive random-walk")
    for row in payload["summary"]:
        assert set(row) == {"term", "mean", "sd", "mcse", "ess", "rhat",
                            "q_low", "median", "q_high"}
        assert row["q_low"] < row["median"] < row["q_high"]
        assert row["mcse"] > 0.0


def test_the_posterior_can_be_indexed_and_printed(gaussian):
    assert gaussian["b"].size == gaussian.n_chains * gaussian.n_draws
    text = str(gaussian)
    assert "Rhat" in text and "acceptance" in text
    lo, hi = gaussian.ci("a", level=0.5)
    assert lo < 0.0 < hi
    assert len(gaussian.ci()) == 3


def test_the_draws_can_be_serialised(gaussian):
    import json
    payload = json.loads(gaussian.draws_json())
    assert payload["names"] == ["a", "b", "c"]
    assert len(payload["chains"]) == 4
    assert len(payload["chains"][0]) == 4000
