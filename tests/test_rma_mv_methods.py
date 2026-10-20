"""metafor's ``rma.mv`` diagnostics, pinned to R 4.6.1 / metafor 5.2-1.

``hatvalues.rma.mv``, ``weights.rma.mv`` and ``rstandard.rma.mv`` are closed
form in ``M`` and ``X``; ``cooks.distance.rma.mv``, ``dfbetas.rma.mv`` and
``rstudent.rma.mv`` drop each study/cluster and refit (re-estimating the
variance components, which is metafor's default).
"""

import numpy as np
import pytest

import pimeta as ma

YI = [0.10, 0.05, -0.20, -0.15, 0.05, 0.02]
VI = [0.01, 0.012, 0.014, 0.013, 0.011, 0.015]
STUDY = [1, 1, 2, 2, 3, 3]
SLAB = list("abcdef")

# R 4.6.1 / metafor 5.2-1, rma.mv(yi, vi, random = ~ 1 | study)
R_HAT = [0.189344741613, 0.157787284677, 0.155470799261,
         0.167430091512, 0.190365624771, 0.139601458166]
R_W_DIAG = [19.2160627706, 17.5934354681, 15.1795024635,
            15.7944276194, 17.4133866098, 14.8031850688]
R_W_ROWSUM = [18.9344741613, 15.7787284677, 15.5470799261,
              16.7430091512, 19.0365624771, 13.9601458166]
R_W_MATRIX_DIAG = [62.8131472354, 57.5091300246, 49.6185058606,
                   51.6285630100, 56.9205893031, 48.3884058030]
R_RS_Z = [0.933304986486, 0.503659671764, -1.302120829409,
          -0.971128726140, 0.518489096120, 0.257465736891]
R_RS_SE = [0.125440800679, 0.133174301105, 0.140482719488,
           0.136877297149, 0.129365352683, 0.143997897467]
R_COOK = [5.65991321976e-02, 2.07057003004e-05, 1.90505364453e-01,
          2.86190446786e-02, 1.96230083838e-02, 1.63995315573e-05]
R_DFBETAS = [0.25799870266164, 0.00438901827957, -0.67374829241052,
             -0.18950613777088, 0.13664910125456, 0.00390635894318]
R_RSTUDENT_Z = [0.861805890039, 0.377601274211, -1.603928615371,
                -0.899021087534, 0.445448494745, 0.200275035232]
R_COOK_CLUSTER = [0.426519088722, 0.967597849376, 0.139094011098]
R_RS_CLUSTER_X2 = [0.909119094235, 2.068273206175, 0.279404004171]


def _mv():
    return ma.rma_mv(YI, V=np.diag(VI), data={"study": STUDY},
                     method="REML", slab=SLAB)


# --------------------------------------------------------------------------- #
# hatvalues / weights / rstandard (closed form)
# --------------------------------------------------------------------------- #

def test_mv_hatvalues_match_r():
    assert ma.hatvalues(_mv()) == pytest.approx(R_HAT, abs=1e-9)


def test_mv_hatvalues_matrix():
    h = ma.hatvalues(_mv(), type="matrix")
    assert h.shape == (6, 6)
    assert np.diag(h) == pytest.approx(R_HAT, abs=1e-9)


def test_mv_weights_diagonal_matches_r():
    assert ma.weights(_mv(), pct=True) == pytest.approx(R_W_DIAG, rel=1e-6)


def test_mv_weights_rowsum_matches_r():
    assert ma.weights(_mv(), type="rowsum") == pytest.approx(R_W_ROWSUM,
                                                             rel=1e-6)


def test_mv_weights_matrix_matches_r():
    w = ma.weights(_mv())
    assert w.shape == (6, 6)
    assert np.diag(w) == pytest.approx(R_W_MATRIX_DIAG, rel=1e-6)


def test_mv_weights_reject_an_unknown_type():
    with pytest.raises(ValueError, match="'type' must be"):
        ma.weights(_mv(), type="nope")


def test_mv_rstandard_matches_r():
    out = ma.rstandard(_mv())
    assert out["z"] == pytest.approx(R_RS_Z, rel=1e-6)
    assert out["se"] == pytest.approx(R_RS_SE, rel=1e-6)
    assert out["slab"] == SLAB


def test_mv_rstandard_cluster_quadratic_forms():
    out = ma.rstandard(_mv(), cluster=STUDY)
    assert out["cluster"]["X2"] == pytest.approx(R_RS_CLUSTER_X2, rel=1e-6)
    assert out["cluster"]["k"] == [2, 2, 2]
    assert out["cluster"]["slab"] == ["1", "2", "3"]


def test_mv_rstandard_rejects_a_bad_cluster():
    with pytest.raises(ValueError, match="does not match the length"):
        ma.rstandard(_mv(), cluster=[1, 2])


# --------------------------------------------------------------------------- #
# refit-based diagnostics
# --------------------------------------------------------------------------- #

def test_mv_cooks_distance_matches_r():
    assert ma.cooks_distance(_mv()) == pytest.approx(R_COOK, rel=1e-4,
                                                     abs=1e-10)


def test_mv_cooks_distance_by_cluster():
    assert ma.cooks_distance(_mv(), cluster=STUDY) == pytest.approx(
        R_COOK_CLUSTER, rel=1e-6)


def test_mv_dfbetas_matches_r():
    got = np.asarray(ma.dfbetas(_mv()), dtype=float).ravel()
    assert got == pytest.approx(R_DFBETAS, rel=1e-4, abs=1e-9)


def test_mv_rstudent_matches_r():
    out = ma.rstudent(_mv())
    assert out["z"] == pytest.approx(R_RSTUDENT_Z, rel=1e-6)
    assert out["slab"] == [f"-{s}" for s in SLAB]


# --------------------------------------------------------------------------- #
# robust() now defaults the cluster to the one stored on the fit
# --------------------------------------------------------------------------- #

def test_robust_uses_the_stored_cluster():
    mv = ma.rma_mv(YI, V=np.diag(VI), data={"study": STUDY}, cluster=STUDY)
    out = ma.robust(mv)
    assert out is not None


def test_robust_requires_a_cluster():
    with pytest.raises(ValueError, match="'cluster' must be specified"):
        ma.robust(_mv())


# --------------------------------------------------------------------------- #
# qqnorm
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("type_", ["resid", "rstandard", "rstudent"])
def test_mv_qqnorm(type_):
    out = ma.qqnorm_rma(_mv(), type=type_)
    assert out.figure is not None


def test_plot_which_qqnorm_accepts_an_mv_fit():
    assert ma.plot(_mv(), which="qqnorm").figure is not None


# --------------------------------------------------------------------------- #
# rma_mv(sigma2=) and the retained inputs
# --------------------------------------------------------------------------- #

def test_sigma2_reproduces_the_fit():
    mv = _mv()
    fixed = ma.rma_mv(YI, V=np.diag(VI), data={"study": STUDY},
                      method="REML", sigma2=mv.sigma2)
    assert fixed.beta == pytest.approx(mv.beta, abs=1e-12)
    assert fixed.se == pytest.approx(mv.se, abs=1e-12)


def test_sigma2_length_is_validated():
    with pytest.raises(ValueError, match="variance component"):
        ma.rma_mv(YI, V=np.diag(VI), data={"study": STUDY},
                  sigma2=[0.01, 0.02])


def test_rma_mv_keeps_its_inputs():
    mv = _mv()
    assert np.asarray(mv.V).shape == (6, 6)
    assert mv.slab == SLAB
    assert mv._data == {"study": STUDY}
    assert mv.test == "z"
    assert mv.intercept_incl is True
