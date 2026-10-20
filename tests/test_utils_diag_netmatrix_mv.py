"""Remaining branches of ``utils``, ``diagnostics``, ``netmatrix`` and ``rma_mv``.

These target the code paths the main suite never reaches: the argument-checking
branches, the alternative-selection branches of ``tes``/``permutest``, the
``netmatrix`` summarisers/recoding and the ``rma_mv`` structure options.
"""

from __future__ import annotations

import numpy as np
import pytest

import pimeta as ma

YI = [0.10, 0.25, -0.05, 0.40, 0.15, 0.30, 0.05, 0.20]
VI = [0.010, 0.020, 0.015, 0.030, 0.012, 0.025, 0.018, 0.022]
NI = [50, 80, 40, 120, 60, 90, 45, 100]


# ---------------------------------------------------------------------------
# utils: blsplit / column / as_data_frame_*
# ---------------------------------------------------------------------------
def test_blsplit_without_args():
    a = np.arange(16.0).reshape(4, 4)
    out = ma.blsplit(a, ["a", "a", "b", "b"])
    assert set(out) == {"a", "b"}
    assert out["a"].shape == (2, 2)


def test_blsplit_with_args():
    a = np.arange(16.0).reshape(4, 4)
    out = ma.blsplit(a, [1, 1, 2, 2], fun=np.mean, args=())
    assert out[1] == pytest.approx(np.mean(a[np.ix_([0, 1], [0, 1])]))


def test_blsplit_sorts_the_keys():
    a = np.arange(16.0).reshape(4, 4)
    assert list(ma.blsplit(a, ["b", "b", "a", "a"], sort=True)) == ["a", "b"]
    assert list(ma.blsplit(a, ["b", "b", "a", "a"], sort=False)) == ["b", "a"]


def test_blsplit_validates_its_input():
    with pytest.raises(ValueError, match="matrix"):
        ma.blsplit([1, 2, 3], [1, 1, 1])
    with pytest.raises(ValueError, match="square"):
        ma.blsplit(np.zeros((2, 3)), [1, 2])
    with pytest.raises(ValueError, match="cluster"):
        ma.blsplit(np.eye(3))
    with pytest.raises(ValueError, match="does not correspond"):
        ma.blsplit(np.eye(3), [1, 2])


def test_as_data_frame_confint_lists_the_interval():
    r = ma.rma(YI, vi=VI, method="REML")
    rng = ma.confint_tau2(YI, VI, r.tau2)
    rows = ma.as_data_frame_confint(rng, include_na=True)
    assert isinstance(rows, list)


def test_as_data_frame_list_anova_on_a_heterogeneity_test():
    r = ma.rma(YI, vi=VI, method="REML")
    rows = ma.as_data_frame_list_anova(r)
    assert isinstance(rows, list)


# ---------------------------------------------------------------------------
# diagnostics: tes / permutest alternative branches
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("alternative", ["two.sided", "greater", "less"])
def test_tes_with_tau2_uses_the_integral(alternative):
    """With tau2 > 0 the power comes from a numeric integral per study."""
    res = ma.tes(YI, VI, tau2=0.02, alternative=alternative)
    assert 0.0 <= res.pval <= 1.0
    power = np.atleast_1d(np.asarray(res.power, dtype=float))
    assert np.all(power >= 0.0) and np.all(power <= 1.0)


@pytest.mark.parametrize("tes_alternative", ["greater", "less"])
def test_tes_alternative_of_the_test_itself(tes_alternative):
    res = ma.tes(YI, VI, tes_alternative=tes_alternative)
    assert 0.0 <= res.pval <= 1.0


def test_tes_with_a_fixed_common_effect_and_tau2():
    res = ma.tes(YI, VI, theta=0.2, tau2=0.01)
    assert 0.0 <= res.pval <= 1.0


@pytest.mark.parametrize("alternative", ["greater", "less", "two.sided"])
def test_permutest_one_sided_alternatives(alternative):
    r = ma.rma(YI, vi=VI)
    res = ma.permutest(r, iter=99, alternative=alternative, seed=3)
    assert all(0.0 < p <= 1.0 for p in res.pval)


def test_permutest_exact_intercept_only():
    """exact = TRUE enumerates the 2^k sign flips."""
    r = ma.rma(YI[:5], vi=VI[:5])
    res = ma.permutest(r, exact=True)
    assert res.exact is True
    assert 0.0 < res.pval[0] <= 1.0


# ---------------------------------------------------------------------------
# netmatrix: summarisers, recoding and the tie-breaking policies
# ---------------------------------------------------------------------------
NMA_TE = [0.10, -0.30, -0.40, 0.55, -0.25, 0.20, 0.40, 0.20, 0.50, -0.10]
NMA_SE = [0.05, 0.06, 0.07, 0.10, 0.12, 0.09, 0.11, 0.13, 0.15, 0.14]
NMA_T1 = ["A", "A", "B", "A", "A", "D", "D", "E", "B", "B"]
NMA_T2 = ["B", "C", "C", "D", "D", "E", "C", "C", "E", "E"]
NMA_ST = ["k1", "k1", "k1", "k2", "k3", "k4", "k4", "k4", "k5", "k6"]


@pytest.fixture(scope="module")
def nma():
    return ma.netmeta(NMA_TE, NMA_SE, NMA_T1, NMA_T2, NMA_ST, sm="MD")


@pytest.mark.parametrize("func", ["mode", "min", "max", "mean", "median", "sum"])
def test_netmatrix_summarisers(nma, func):
    var = np.asarray(nma.TE_raw, dtype=float)
    mat = np.asarray(ma.netmatrix(nma, var, func=func).matrix, dtype=float)
    assert mat.shape == (nma.n, nma.n)
    assert np.all(np.isnan(np.diag(mat)))


def test_netmatrix_recodes_levels_into_labels(nma):
    var = np.asarray(nma.TE_raw, dtype=float)
    levels = sorted(set(var.tolist()))
    labels = [f"L{i}" for i in range(len(levels))]
    res = ma.netmatrix(nma, var, levels=levels, labels=labels)
    flat = [v for row in np.asarray(res.matrix, dtype=object) for v in row
            if v == v]
    assert flat
    assert all(v in labels for v in flat)


def test_netmatrix_rejects_mismatched_levels_and_labels(nma):
    var = np.asarray(nma.TE_raw, dtype=float)
    with pytest.raises(ValueError, match="Different lengths"):
        ma.netmatrix(nma, var, levels=[0.1, 0.2], labels=["only-one"])


@pytest.mark.parametrize("ties", ["first", "last", "random"])
def test_netmatrix_tie_breaking(nma, ties):
    var = np.asarray(nma.TE_raw, dtype=float)
    res = ma.netmatrix(nma, var, ties_method=ties, seed=1)
    assert res.matrix.shape == (nma.n, nma.n)


def test_netmatrix_rejects_bad_arguments(nma):
    var = np.asarray(nma.TE_raw, dtype=float)
    with pytest.raises(ValueError, match="func must be"):
        ma.netmatrix(nma, var, func="median2")
    with pytest.raises(ValueError, match="ties_method"):
        ma.netmatrix(nma, var, ties_method="middle")


# ---------------------------------------------------------------------------
# rma_mv: structure options
# ---------------------------------------------------------------------------
def _mv_data():
    yi = [0.10, 0.20, 0.30, 0.05, 0.15, 0.25]
    V = np.zeros((6, 6))
    np.fill_diagonal(V, 0.02)
    mods = np.array([[1.0], [2.0], [3.0], [1.5], [2.5], [3.5]])
    data = {"study": [1, 1, 2, 2, 3, 3]}
    return yi, V, mods, data


@pytest.mark.parametrize("struct", ["CS", "HCS", "UN", "ID", "DIAG"])
def test_rma_mv_structures(struct):
    yi, V, mods, data = _mv_data()
    res = ma.rma_mv(yi, V=V, mods=mods, data=data, struct=struct,
                    method="REML")
    assert np.isfinite(np.asarray(res.beta).ravel()[0])


def test_rma_mv_without_intercept():
    yi, V, mods, data = _mv_data()
    res = ma.rma_mv(yi, V=V, mods=mods, data=data, intercept=False,
                    method="REML")
    assert np.asarray(res.beta).ravel().size == 1


def test_rma_mv_rejects_an_unknown_structure():
    yi, V, mods, data = _mv_data()
    with pytest.raises(ValueError, match="unknown 'struct'"):
        ma.rma_mv(yi, V=V, mods=mods, data=data, struct="NOPE")


def test_rma_mv_explains_a_missing_grouping_variable():
    yi, V, mods, _ = _mv_data()
    with pytest.raises(ValueError, match="no 'data' was supplied"):
        ma.rma_mv(yi, V=V, mods=mods, struct="CS")


def test_rma_mv_explains_a_formula_without_a_grouping_factor():
    yi, V, mods, data = _mv_data()
    with pytest.raises(ValueError, match="no grouping factor"):
        ma.rma_mv(yi, V=V, mods=mods, data=data, random="~ 1")


def test_rma_mv_explains_an_unknown_grouping_column():
    yi, V, mods, _ = _mv_data()
    with pytest.raises(ValueError, match="not a column of 'data'"):
        ma.rma_mv(yi, V=V, mods=mods, data={"id": [1, 1, 2, 2, 3, 3]})
