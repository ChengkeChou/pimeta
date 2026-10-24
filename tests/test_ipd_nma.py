"""One-stage IPD network meta-analysis.

Three references, for three different reasons.

``lme4::lmer``
    The Gaussian one-stage model *is* a linear mixed model, so this is not an
    analogy: ``lmer(y ~ 0 + study + A + x + (0 + A | study))`` with the
    treatment-dummy matrix ``A`` is the same likelihood, fitted by a different
    (Laplace/pseudo-inverse) route.  That pins the network parameterisation
    after contact with a likelihood, which is the thing the algebraic tests in
    ``test_nma_design.py`` cannot reach.

``pimeta.rma`` on the study-level differences
    With two treatments and a common residual variance the one-stage model
    collapses to an ordinary random-effects meta-analysis of the mean
    differences.  A model that fails its own one-parameter limit is wrong no
    matter what a second implementation says.

OLS inside one study
    With one study there is nothing to pool, so ``d`` must be the within-study
    regression coefficient.  This is the smallest possible check and it is the
    one that catches a mis-indexed design column.

``multinma`` (Stan, MCMC) is fitted on the same fixture where it is available;
its posterior means agree with these REML estimates to Monte-Carlo accuracy
rather than to machine precision, which is why the pinned numbers here come
from ``lme4`` and the Stan comparison is described rather than asserted.
"""

import pathlib

import numpy as np
import pytest

import pimeta as ma
from pimeta.nma.ipd import ipd_nma

DATA = np.genfromtxt(
    pathlib.Path(__file__).resolve().parent / "data" / "ipd_nma.csv",
    delimiter=",", skip_header=1)
STUDY = DATA[:, 0]
DB = DATA[:, 1]
DC = DATA[:, 2]
XVAR = DATA[:, 3]
Y = DATA[:, 4]
TREAT = np.array(["B" if b > 0.5 else "C" if c > 0.5 else "A"
                  for b, c in zip(DB, DC)])

# --- R 4.6.1, lme4 2.0.6, lmer(y ~ 0 + study + A + x + (0 + A | study)) ---
R_D = [0.0702242209, 0.1491445777]
R_SE = [0.1118358797, 0.1396767898]
R_GAMMA = 0.5871860904
R_SIGMA2 = 0.4955479661
R_SIGMA = [[0.0945520125, -0.0384954842],
           [-0.0384954842, 0.1147433433]]
R_LOGLIK = -4269.2622289694


@pytest.fixture(scope="module")
def fit():
    return ipd_nma(Y, study=STUDY, treat=TREAT, covariates={"x": XVAR})


def test_the_network_model_matches_lme4(fit):
    assert fit.reference == "A"
    assert fit.estimable == ["B", "C"]
    assert fit.d == pytest.approx(R_D, abs=1e-5)
    assert fit.se == pytest.approx(R_SE, abs=1e-5)
    assert fit.gamma == pytest.approx([R_GAMMA], abs=1e-6)
    assert fit.sigma2 == pytest.approx(R_SIGMA2, abs=1e-6)
    # the between-study covariance is the flattest direction of the
    # likelihood, so it is pinned where the profile actually agrees
    assert np.asarray(fit.Sigma) == pytest.approx(np.array(R_SIGMA), abs=5e-5)
    assert fit.loglik == pytest.approx(R_LOGLIK, abs=1e-4)


def test_the_network_counts_the_way_the_graph_says_it_should(fit):
    """Fourteen independent within-study contrasts, three treatments."""
    assert fit.k == 8
    assert fit.participants == Y.size
    assert fit.network.n_comparisons == 14
    assert fit.network.df_inconsistency == 12
    assert fit.network.is_connected()


def test_two_treatments_collapse_to_a_random_effects_meta_analysis():
    """The one-parameter limit, checked against ``rma`` exactly."""
    rng = np.random.default_rng(77)
    k, per = 10, 60
    study, treat, y = [], [], []
    for i in range(1, k + 1):
        mu = float(rng.normal(0.0, 0.3))
        delta = float(rng.normal(0.2, 0.25))
        study += [i] * (2 * per)
        treat += ["A"] * per + ["B"] * per
        y += list(mu + rng.normal(0, 0.8, per))
        y += list(mu + delta + rng.normal(0, 0.8, per))
    study = np.array(study)
    treat = np.array(treat)
    y = np.array(y)
    fit = ipd_nma(y, study=study, treat=treat)

    md, vi = [], []
    for i in range(1, k + 1):
        yc = y[(study == i) & (treat == "A")]
        yt = y[(study == i) & (treat == "B")]
        md.append(float(yt.mean() - yc.mean()))
        vi.append(fit.sigma2 * (1.0 / yc.size + 1.0 / yt.size))
    pool = ma.rma(np.array(md), vi=np.array(vi), method="REML")
    assert fit.d[0] == pytest.approx(float(pool.b[0]), rel=1e-10)
    assert fit.tau2[0] == pytest.approx(float(pool.tau2), rel=1e-6)


def test_one_study_gives_the_within_study_regression():
    rng = np.random.default_rng(5)
    n = 120
    treat = np.array(["A", "B"] * (n // 2))
    x = rng.normal(0.0, 1.0, n)
    y = 1.0 + 0.7 * (treat == "B") + 0.4 * x + rng.normal(0.0, 0.5, n)
    study = np.ones(n)
    fit = ipd_nma(y, study=study, treat=treat, covariates={"x": x})
    X = np.column_stack([np.ones(n), (treat == "B").astype(float), x])
    beta = np.linalg.lstsq(X, y, rcond=None)[0]
    assert fit.d[0] == pytest.approx(float(beta[1]), rel=1e-8)
    assert fit.gamma[0] == pytest.approx(float(beta[2]), rel=1e-8)


def test_the_reference_choice_changes_the_parameterisation_not_the_network():
    """Every contrast is invariant; only the reported basic parameters move."""
    base = ipd_nma(Y, study=STUDY, treat=TREAT, covariates={"x": XVAR})
    other = ipd_nma(Y, study=STUDY, treat=TREAT, covariates={"x": XVAR},
                    reference="B")
    assert other.reference == "B"
    for a, b in (("A", "C"), ("B", "C"), ("A", "B")):
        assert other.contrast(a, b)["estimate"] == pytest.approx(
            base.contrast(a, b)["estimate"], abs=1e-6)
        assert other.contrast(a, b)["se"] == pytest.approx(
            base.contrast(a, b)["se"], abs=1e-6)


def test_a_contrast_is_the_difference_of_basic_parameters(fit):
    """``C - B`` is exact arithmetic on the fitted network, not a second fit."""
    direct = fit.contrast("C", "B")
    via = (fit.contrast("C", "A")["estimate"]
           - fit.contrast("B", "A")["estimate"])
    assert direct["estimate"] == pytest.approx(via, rel=1e-12)
    # and the variance follows the joint covariance, which needs the
    # off-diagonal term: ignoring it would change the standard error
    vb = np.asarray(fit.vcov)
    assert direct["se"] == pytest.approx(
        np.sqrt(vb[0, 0] + vb[1, 1] - 2 * vb[0, 1]), rel=1e-12)


def test_the_league_table_contains_every_pair_once(fit):
    rows = fit.league_table()
    pairs = {(r["treat1"], r["treat2"]) for r in rows}
    assert pairs == {("A", "B"), ("A", "C"), ("B", "C")}
    assert len(rows) == 3
    for row in rows:
        assert row["ci_lb"] < row["estimate"] < row["ci_ub"]


def test_centring_a_covariate_within_study_does_not_change_the_network():
    """A within-study shift is absorbed by the study intercept, by design."""
    shifted = XVAR + np.array([7.0 if s == 3 else 0.0 for s in STUDY])
    base = ipd_nma(Y, study=STUDY, treat=TREAT, covariates={"x": XVAR})
    moved = ipd_nma(Y, study=STUDY, treat=TREAT, covariates={"x": shifted})
    assert moved.d == pytest.approx(base.d, abs=1e-8)
    assert moved.gamma == pytest.approx(base.gamma, abs=1e-8)


def test_a_treatment_by_covariate_interaction_is_estimable(fit):
    with_int = ipd_nma(Y, study=STUDY, treat=TREAT, covariates={"x": XVAR},
                       interaction=["x"])
    assert with_int.interaction == ["x"]
    assert with_int.delta is not None
    # two non-reference treatments × one interacted covariate
    assert len(with_int.delta) == 2
    # the interaction changes the treatment effects at the covariate mean,
    # which here is zero because the covariate was centred
    assert with_int.d == pytest.approx(fit.d, abs=0.06)


def test_the_refusals():
    with pytest.raises(ValueError, match="gaussian"):
        ipd_nma(Y, study=STUDY, treat=TREAT, family="binomial")
    with pytest.raises(ValueError, match="not in the data"):
        ipd_nma(Y, study=STUDY, treat=TREAT, reference="Z")
    with pytest.raises(ValueError, match="is not among the covariates"):
        ipd_nma(Y, study=STUDY, treat=TREAT, covariates={"x": XVAR},
                interaction=["age"])
    with pytest.raises(ValueError, match="disconnected"):
        # two studies, two treatments, and no study compares them
        ipd_nma(np.arange(8.0), study=np.array([1, 1, 1, 1, 2, 2, 2, 2]),
                treat=np.array(["A", "A", "A", "A",
                                "B", "B", "B", "B"]))
    with pytest.raises(ValueError, match="unknown method"):
        ipd_nma(Y, study=STUDY, treat=TREAT, method="PM")


def test_a_disconnected_network_is_refused_before_it_is_fitted():
    """``_network_from`` alone must see the split, not the optimiser later."""
    from pimeta.nma.ipd import _network_from

    net = _network_from(
        study=np.array([1, 1, 2, 2]),
        treat=np.array(["A", "A", "B", "B"]), reference=None)
    assert net.n_comparisons == 0
    with pytest.raises(ValueError, match="disconnected"):
        ipd_nma(np.arange(4.0), study=np.array([1, 1, 2, 2]),
                treat=np.array(["A", "A", "B", "B"]))


def test_the_record_and_the_text(fit):
    record = fit.to_dict()
    assert record["reference"] == "A"
    assert record["treatments"] == ["A", "B", "C"]
    assert record["df_inconsistency"] == 12
    assert len(record["coef"]) == 2
    text = str(fit)
    assert "one-stage IPD network meta-analysis" in text
    assert "degree(s) of freedom for inconsistency" in text
