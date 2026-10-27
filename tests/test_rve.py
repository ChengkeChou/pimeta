"""Cluster-robust variance estimation for ``rma`` and ``rma.mv``.

The univariate reference numbers come from ``metafor`` 5.2.1 on R 4.6.1::

    res <- rma(yi, vi, data = dat, method = "REML")   # tau2 = 0 here
    robust(res, cluster = dat$grp)                    # CR1
    robust(res, cluster = dat$grp, adjust = FALSE)    # CR0
    robust(res, cluster = dat$grp, adjust = "Stata1") # CR1.S1
    robust(res, cluster = dat$grp, adjust = "Stata2") # CR1.S2
    robust(res, cluster = dat$grp, clubSandwich = TRUE,
           vcov = "CR2", coef_test = "Satterthwaite")

The fixture has nine estimates in five clusters, so the small-sample factor is
some way from 1 and a mistake in it would show up.
"""

from __future__ import annotations

import numpy as np
import pytest

import pimeta as ma

YI = [0.20, 0.40, 0.10, 0.50, 0.30, 0.25, 0.45, 0.15, 0.35]
VI = [0.040, 0.050, 0.030, 0.060, 0.020, 0.050, 0.045, 0.035, 0.055]
GRP = ["A", "A", "B", "B", "C", "C", "D", "D", "E"]
DOSE = list(range(1, 10))


@pytest.fixture()
def fit():
    return ma.rma(YI, vi=VI, method="REML")


# --------------------------------------------------------------------------- #
# the univariate sandwich, against metafor::robust.rma.uni
# --------------------------------------------------------------------------- #
def test_robust_cr1_matches_metafor(fit):
    out = ma.robust(fit, cluster=GRP)
    assert out["vbest"] == "CR1"
    assert out["n"] == 5
    assert out["dfs"] == pytest.approx(4.0)
    assert out["se"] == pytest.approx([0.01293040], rel=1e-6)
    assert out["zval"] == pytest.approx([21.58924], rel=1e-6)
    assert out["pval"] == pytest.approx([2.722799e-05], rel=1e-5)
    assert out["ci_lb"] == pytest.approx([0.2432570], rel=1e-6)
    assert out["ci_ub"] == pytest.approx([0.3150581], rel=1e-6)
    assert out["vb"][0, 0] == pytest.approx(0.0001671953, rel=1e-6)


def test_robust_cr0_matches_metafor(fit):
    out = ma.robust(fit, cluster=GRP, adjust=False)
    assert out["vbest"] == "CR0"
    assert out["se"] == pytest.approx([0.0115653], rel=1e-6)
    assert out["vb"][0, 0] == pytest.approx(0.0001337563, rel=1e-6)


def test_robust_stata_variants_match_metafor(fit):
    # With p = 1 all three small-sample factors coincide, so this checks the
    # label as much as the number; the moderator test below separates them.
    for adjust, label in (("Stata1", "CR1.S1"), ("Stata2", "CR1.S2")):
        out = ma.robust(fit, cluster=GRP, adjust=adjust)
        assert out["vbest"] == label
        assert out["se"] == pytest.approx([0.0129304], rel=1e-6)


def test_robust_omnibus_test_matches_metafor(fit):
    out = ma.robust(fit, cluster=GRP)
    assert out["QM"] == pytest.approx(466.0952, rel=1e-6)
    assert out["QMdf"][0] == 1
    assert out["QMdf"][1] == pytest.approx(4.0)
    assert out["QMp"] == pytest.approx(2.722799e-05, rel=1e-5)


def test_robust_with_a_moderator_matches_metafor():
    fitm = ma.rma(YI, vi=VI, mods=DOSE, method="REML")
    out = ma.robust(fitm, cluster=GRP)
    assert out["beta"] == pytest.approx([0.2436215, 0.007233647], rel=1e-5)
    assert out["se"] == pytest.approx([0.05462394, 0.009711815], rel=1e-5)
    assert out["pval"] == pytest.approx([0.02098795, 0.5104158], rel=1e-5)
    assert out["dfs"] == pytest.approx(3.0)
    # The default omnibus test covers the slopes, not the intercept.
    assert out["QM"] == pytest.approx(0.5547711, rel=1e-5)
    assert out["QMdf"][0] == 1
    assert out["QMdf"][1] == pytest.approx(3.0)
    assert out["QMp"] == pytest.approx(0.5104158, rel=1e-5)


def test_robust_cr2_satterthwaite_matches_clubsandwich(fit):
    out = ma.robust(fit, cluster=GRP, clubSandwich=True, vcov="CR2",
                    coef_test="Satterthwaite")
    assert out["vbest"] == "CR2"
    assert out["se"] == pytest.approx([0.01218723], rel=1e-6)
    assert out["zval"] == pytest.approx([22.90574], rel=1e-6)
    assert out["pval"] == pytest.approx([6.643951e-05], rel=1e-5)
    assert out["ci_lb"] == pytest.approx([0.2431438], rel=1e-6)
    assert out["ci_ub"] == pytest.approx([0.3151713], rel=1e-6)
    assert out["dfs"][0] == pytest.approx(3.46232, rel=1e-5)
    # The omnibus test is not reported on this path; clubSandwich's reference
    # distribution would be needed to state it honestly.
    assert out["QM"] is None


# --------------------------------------------------------------------------- #
# validation and shared behaviour
# --------------------------------------------------------------------------- #
def test_robust_needs_a_cluster_for_a_univariate_fit(fit):
    # metafor::robust.rma.uni has no default either.
    with pytest.raises(ValueError, match="'cluster' must be specified"):
        ma.robust(fit)


def test_robust_rejects_a_cluster_of_the_wrong_length(fit):
    with pytest.raises(ValueError, match="one entry per observation"):
        ma.robust(fit, cluster=GRP[:4])


def test_robust_needs_more_clusters_than_coefficients(fit):
    with pytest.raises(ValueError, match="must exceed"):
        ma.robust(fit, cluster=["A"] * len(YI))


def test_robust_accepts_a_percentage_or_a_fraction_for_level(fit):
    as_fraction = ma.robust(fit, cluster=GRP, level=0.90)
    as_percent = ma.robust(fit, cluster=GRP, level=90)
    assert as_fraction["ci_lb"] == pytest.approx(as_percent["ci_lb"])


def test_a_narrower_level_gives_a_narrower_interval(fit):
    wide = ma.robust(fit, cluster=GRP, level=0.99)
    narrow = ma.robust(fit, cluster=GRP, level=0.90)
    assert narrow["ci_ub"][0] < wide["ci_ub"][0]


def test_robust_works_on_a_multilevel_fit_as_before():
    yi = np.array([0.2, 0.4, 0.1, 0.5, 0.3, 0.25])
    vi = np.array([0.04, 0.05, 0.03, 0.06, 0.02, 0.05])
    mv = ma.rma_mv(yi, vi=vi, data={"study": np.arange(6)}, random="~ 1 | study")
    out = ma.robust(mv, cluster=[1, 1, 2, 2, 3, 3])
    assert out["vbest"] == "CR1"
    assert out["n"] == 3
    assert len(out["se"]) == 1


def test_the_old_import_path_still_works():
    from pimeta.rma_mv import robust as via_rma_mv

    assert via_rma_mv is ma.robust


# --------------------------------------------------------------------------- #
# the CR2 factorisation when a cluster block is not positive definite
# --------------------------------------------------------------------------- #
def test_the_symmetric_root_reproduces_a_positive_definite_matrix():
    from pimeta.rve import _symmetric_root

    rng = np.random.default_rng(4)
    A = rng.normal(size=(5, 5))
    A = A @ A.T + np.eye(5)
    R = _symmetric_root(A)
    assert R == pytest.approx(R.T, abs=1e-12)
    assert R @ R.T == pytest.approx(A, abs=1e-10)


def test_the_symmetric_root_is_defined_when_cholesky_is_not():
    """A cluster block with a negative eigenvalue is not a Cholesky input.

    That is the case a two-stage IPD fit produces when a variance component is
    estimated on its boundary: the block is nearly singular, and rounding or a
    boundary estimate makes it indefinite.  Cholesky raises; the symmetric root
    with a floored spectrum does not, which is what keeps the CR2 sandwich
    available in exactly those fits.
    """
    from pimeta.rve import _symmetric_root

    A = np.array([[1.0, 0.2], [0.2, -1e-9]])
    with pytest.raises(np.linalg.LinAlgError):
        np.linalg.cholesky(A)
    lam, V = np.linalg.eigh(A)
    assert lam[0] < 0.0            # this matrix really is indefinite, not just
    R = _symmetric_root(A)         # numerically awkward
    assert np.all(np.isfinite(R))
    # the only change to A is that the negative eigenvalue is raised to the
    # floor: R R' = A + (floor - lambda_min) v v'
    expected = (1e-10 - lam[0]) * np.outer(V[:, 0], V[:, 0])
    assert R @ R.T - A == pytest.approx(expected, abs=1e-9)


def test_cr2_runs_on_a_cluster_whose_block_is_indefinite():
    from pimeta.rve import _cr_vcov, _View

    # three clusters of two rows, one block indefinite
    X = np.column_stack([np.ones(6), [0.0, 0.1, 1.0, 1.1, 2.0, 2.1]])
    M = np.zeros((6, 6))
    blocks = [np.array([[0.2, 0.05], [0.05, 0.2]]),
              np.array([[0.3, 0.1], [0.1, 0.3]]),
              np.array([[0.4, 0.2], [0.2, -1e-10]])]
    for j, blk in enumerate(blocks):
        M[2 * j:2 * j + 2, 2 * j:2 * j + 2] = blk
    yi = np.array([0.1, 0.2, 0.4, 0.3, 0.5, 0.6])
    beta = np.array([[0.15], [0.1]])
    vb = np.eye(2) * 0.01
    view = _View(X=X, yi=yi, beta=beta, M=M, vb=vb, p=2, k=6, level=0.95,
                 btt=[1], multilevel=True, stored_cluster=None)
    cluster = np.repeat([1, 2, 3], 2)
    with pytest.raises(np.linalg.LinAlgError):
        np.linalg.cholesky(blocks[2])
    vb_cr, E_list, _X, _Omega, _B = _cr_vcov(view, cluster, "CR2")
    assert np.all(np.isfinite(vb_cr))
    assert np.all(np.isfinite(np.column_stack(E_list)))
