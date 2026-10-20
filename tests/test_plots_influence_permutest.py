"""Tests for the last two metafor plotting methods and the coefficient labels.

Covers ``plot_influence`` (``metafor::plot.infl.rma.uni``), ``plot_permutest``
(``metafor::plot.permutest.rma.uni``) and the metafor-style short names
``dfbetas`` / ``rstandard``.
"""

import numpy as np
import pytest

import pimeta as ma
from pimeta.utils import _coef_names

YI = [0.10, -0.20, 0.05, -0.15, 0.30, 0.25, 0.12, -0.05]
VI = [0.01, 0.0144, 0.0121, 0.0169, 0.0196, 0.0225, 0.01, 0.0121]
MOD = [1.0, 2.0, 1.5, 3.0, 2.5, 0.5, 1.2, 2.2]


def _fit():
    return ma.rma(YI, vi=VI, mods=MOD, method="REML")


# --------------------------------------------------------------------------- #
# plot_influence
# --------------------------------------------------------------------------- #

def test_plot_influence_draws_eight_panels():
    res = ma.influence(YI, VI)
    assert res.k == 8
    out = ma.plot_influence(res)
    # rstudent, dffits, cook.d, cov.r, tau2.del, QE.del, hat, weight
    axes = [a for a in out.figure.axes if a.get_visible()]
    assert len(axes) == 8
    assert [a.get_title() for a in axes] == [
        "rstudent", "dffits", "cook.d", "cov.r",
        "tau2.del", "QE.del", "hat", "weight",
    ]


def test_plot_influence_accepts_panel_indices():
    res = ma.influence(YI, VI)
    out = ma.plot_influence(res, plotinf=[1])
    assert [a.get_title() for a in out.figure.axes
            if a.get_visible()] == ["rstudent"]
    out = ma.plot_influence(res, plotinf=[4, 3, 3])
    # duplicates collapse, order follows the caller
    assert [a.get_title() for a in out.figure.axes
            if a.get_visible()] == ["cov.r", "cook.d"]


def test_plot_influence_rejects_bad_plotinf():
    res = ma.influence(YI, VI)
    with pytest.raises(ValueError, match="index the 8 panels"):
        ma.plot_influence(res, plotinf=[9])
    with pytest.raises(ValueError, match="must be true"):
        ma.plot_influence(res, plotinf=False)


def test_plot_influence_plotdfbs_adds_one_panel_per_coefficient():
    res = ma.influence(YI, VI, X=MOD)
    out = ma.plot_influence(res, plotinf=[1], plotdfbs=True)
    titles = [a.get_title() for a in out.figure.axes if a.get_visible()]
    assert titles == ["rstudent", "dfbs: intrcpt", "dfbs: mod1"]


def test_plot_influence_covariance_ratio_is_log_by_default():
    res = ma.influence(YI, VI)
    on = ma.plot_influence(res, plotinf=[4])
    assert on.ax.get_yscale() == "log"
    off = ma.plot_influence(res, plotinf=[4], logcov=False)
    assert off.ax.get_yscale() == "linear"


def test_plot_influence_honours_slab_and_filename(tmp_path):
    res = ma.influence(YI, VI)
    path = tmp_path / "infl.png"
    slab = [f"S{i}" for i in range(1, 9)]
    out = ma.plot_influence(res, slab=slab, filename=str(path))
    assert out.filename == str(path)
    assert path.exists() and path.stat().st_size > 0
    assert [t.get_text() for t in out.ax.get_xticklabels()] == slab


def test_plot_influence_wrong_slab_falls_back_to_indices():
    res = ma.influence(YI, VI)
    out = ma.plot_influence(res, slab=["a", "b"])
    assert [t.get_text() for t in out.ax.get_xticklabels()] == [
        str(i) for i in range(1, 9)
    ]


def test_plot_which_influence_from_a_fitted_model():
    out = ma.plot(_fit(), which="influence")
    assert len([a for a in out.figure.axes if a.get_visible()]) == 8


def test_plot_which_influence_accepts_an_influence_result():
    res = ma.influence(YI, VI)
    out = ma.plot(res, which="infl")
    assert len([a for a in out.figure.axes if a.get_visible()]) == 8


def test_plot_which_influence_needs_study_data():
    class Bare:
        pass

    with pytest.raises(TypeError, match="needs study-level yi/vi"):
        ma.plot(Bare(), which="influence")


# --------------------------------------------------------------------------- #
# plot_permutest
# --------------------------------------------------------------------------- #

def test_permutest_keeps_the_permutation_distributions():
    perm = ma.permutest(_fit(), iter=100, seed=3)
    assert np.asarray(perm.zval_perm).shape == (100, 2)
    assert len(perm.QM_perm) == 100
    assert perm.coefnames == ["intrcpt", "mod1"]
    assert np.allclose(perm.zval, _fit().zval, equal_nan=True)


def test_plot_permutest_one_panel_per_coefficient():
    perm = ma.permutest(_fit(), iter=150, seed=4)
    out = ma.plot_permutest(perm)
    axes = [a for a in out.figure.axes if a.get_visible()]
    assert len(axes) == 2
    assert "mod1" in axes[1].get_title()


def test_plot_permutest_btt_and_nbin():
    perm = ma.permutest(_fit(), iter=120, seed=5)
    out = ma.plot_permutest(perm, btt=[2], nbin=7)
    axes = [a for a in out.figure.axes if a.get_visible()]
    assert len(axes) == 1
    # a histogram with 7 bins lives on the axes as a single BarContainer
    assert len(axes[0].patches) == 7


def test_plot_permutest_rejects_a_plain_model():
    with pytest.raises(TypeError, match="permutation distributions"):
        ma.plot_permutest(_fit())


def test_plot_permutest_without_a_distribution_is_rejected_by_the_plotter():
    from pimeta.diagnostics import PermutestResult

    bare = PermutestResult(pval=[0.5], QM=1.0, QMp=0.5, QMp_perm=0.5,
                           dfs=[7], k=8, p=1, nperm=10)
    with pytest.raises(TypeError, match="permutation distributions"):
        ma.plot_permutest(bare)


def test_plot_which_permutest_computes_the_test():
    out = ma.plot(_fit(), which="permutest", iter=80, seed=6)
    assert len([a for a in out.figure.axes if a.get_visible()]) == 2


def test_plot_which_permutest_from_a_permutest_result(tmp_path):
    perm = ma.permutest(_fit(), iter=90, seed=7)
    path = tmp_path / "perm.png"
    out = ma.plot(perm, which="permutation", filename=str(path))
    assert path.exists() and path.stat().st_size > 0


# --------------------------------------------------------------------------- #
# coefficient labels / aliases
# --------------------------------------------------------------------------- #

def test_coef_names_follow_metafor():
    assert _coef_names(ma.rma(YI, vi=VI)) == ["intrcpt"]
    assert _coef_names(_fit()) == ["intrcpt", "mod1"]
    two = ma.rma(YI, vi=VI, mods=np.column_stack([MOD, list(range(8))]))
    assert _coef_names(two) == ["intrcpt", "mod1", "mod2"]


def test_dfbetas_and_rstandard_aliases():
    assert ma.dfbetas is ma.dfbetas_rma
    assert ma.rstandard is ma.rstandard_rma
    assert len(ma.dfbetas(_fit())) == 8


def test_plot_rejects_an_unknown_which():
    with pytest.raises(ValueError, match="unknown 'which' value"):
        ma.plot(_fit(), which="nonsense")
