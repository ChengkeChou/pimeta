"""Order-2/3 delta method, the ``tes`` variants, ``vif`` specs and the
moderator-based permutation test.

These cover the branches of ``diagnostics.py`` that the main suite never
reaches; the identities asserted here are exact (delta method for ``a/b``,
VIF algebra) or structural (bounds, monotonicity).
"""

from __future__ import annotations

import math

import numpy as np
import pytest

import pimeta as ma

YI = [0.10, 0.25, -0.05, 0.40, 0.15, 0.30, 0.05, 0.20]
VI = [0.010, 0.020, 0.015, 0.030, 0.012, 0.025, 0.018, 0.022]


# ---------------------------------------------------------------------------
# deltamethod
# ---------------------------------------------------------------------------
COEF = np.array([1.0, 2.0])
VCOV = np.array([[0.01, 0.002], [0.002, 0.04]])


def test_deltamethod_order1_matches_the_analytic_ratio():
    res = ma.deltamethod(COEF, VCOV, lambda a, b: a / b, order=1)
    # g = a/b, g' = (1/b, -a/b^2) = (0.5, -0.25)
    want = 0.25 * 0.01 + 2 * 0.5 * (-0.25) * 0.002 + 0.0625 * 0.04
    assert res.coef[0] == pytest.approx(0.5, abs=1e-12)
    assert res.se[0] == pytest.approx(math.sqrt(want), abs=1e-6)


def test_deltamethod_accepts_a_vector_valued_function():
    res = ma.deltamethod(COEF, VCOV, lambda v: np.array([v[0] / v[1],
                                                         v[0] * v[1]]))
    assert len(res.coef) == 2
    assert res.coef == pytest.approx([0.5, 2.0], abs=1e-12)
    assert np.asarray(res.vcov).shape == (2, 2)


def test_deltamethod_orders_agree_for_a_linear_function():
    """A linear transformation has no curvature, so every order coincides."""
    f = lambda a, b: 2.0 * a + 3.0 * b
    r1 = ma.deltamethod(COEF, VCOV, f, order=1)
    r2 = ma.deltamethod(COEF, VCOV, f, order=2)
    r3 = ma.deltamethod(COEF, VCOV, f, order=3)
    assert r1.coef == r2.coef == r3.coef
    for r in (r2, r3):
        assert r.se[0] == pytest.approx(r1.se[0], abs=1e-6)


def test_deltamethod_order2_differs_for_a_bilinear_function():
    """a*b has a non-zero mixed second derivative."""
    r1 = ma.deltamethod(COEF, VCOV, lambda a, b: a * b, order=1)
    r2 = ma.deltamethod(COEF, VCOV, lambda a, b: a * b, order=2)
    assert r2.se[0] != pytest.approx(r1.se[0], abs=1e-6)


def test_deltamethod_order2_adds_a_variance_term():
    """/b is non-linear, so order 2 must differ from order 1."""
    r1 = ma.deltamethod(COEF, np.array([[0.05, 0.0], [0.0, 0.05]]),
                        lambda a, b: a / b, order=1)
    r2 = ma.deltamethod(COEF, np.array([[0.05, 0.0], [0.0, 0.05]]),
                        lambda a, b: a / b, order=2)
    assert np.isfinite(r2.se[0])
    assert r2.se[0] != pytest.approx(r1.se[0], abs=1e-6)


def test_deltamethod_order3_is_finite():
    r3 = ma.deltamethod(COEF, np.array([[0.05, 0.0], [0.0, 0.05]]),
                        lambda a, b: a / b, order=3)
    assert np.all(np.isfinite(r3.se))
    assert np.all(np.isfinite(np.asarray(r3.vcov)))


def test_deltamethod_ci_and_h0():
    res = ma.deltamethod(COEF, VCOV, lambda a, b: a / b, level=0.90, H0=0.4)
    assert res.level == 0.90
    crit = 1.6448536269514722
    assert res.ci_lb[0] == pytest.approx(res.coef[0] - crit * res.se[0],
                                         abs=1e-9)
    assert res.zval[0] == pytest.approx((res.coef[0] - 0.4) / res.se[0],
                                        abs=1e-6)


def test_deltamethod_h0_per_transformed_coefficient():
    res = ma.deltamethod(COEF, VCOV,
                         lambda v: np.array([v[0] / v[1], v[0] * v[1]]),
                         H0=[0.5, 2.0])
    assert res.zval[0] == pytest.approx(0.0, abs=1e-6)


def test_deltamethod_accepts_a_diagonal_vcov():
    res = ma.deltamethod(COEF, [0.01, 0.04], lambda a, b: a + b)
    assert res.se[0] == pytest.approx(math.sqrt(0.05), abs=1e-7)


@pytest.mark.parametrize("kwargs, match", [
    ({"fun": "not callable"}, "callable"),
    ({"fun": lambda v: v[0], "order": 4}, "order"),
    ({"fun": lambda v: v[0], "vcov": np.zeros((2, 3))}, "square"),
    ({"fun": lambda v: v[0], "vcov": np.array([[0.1, 0.2], [0.0, 0.1]])},
     "symmetric"),
    ({"fun": lambda v: v[0], "vcov": np.eye(3)}, "does not match"),
    ({"fun": lambda a, b, c: a}, "number of function arguments"),
    ({"fun": lambda a, b: a / b, "H0": [0.1, 0.2, 0.3]}, "H0"),
])
def test_deltamethod_rejects_bad_arguments(kwargs, match):
    kw = {"coef": COEF, "vcov": VCOV, "fun": lambda v: v[0]}
    kw.update(kwargs)
    with pytest.raises((TypeError, ValueError), match=match):
        ma.deltamethod(**kw)


# ---------------------------------------------------------------------------
# tes
# ---------------------------------------------------------------------------
def test_tes_two_sided_runs():
    res = ma.tes(YI, VI)
    assert 0.0 <= res.pval <= 1.0
    assert res.k == len(YI)


def test_tes_one_sided_alternatives():
    two = ma.tes(YI, VI, alternative="two.sided")
    greater = ma.tes(YI, VI, alternative="greater")
    less = ma.tes(YI, VI, alternative="less")
    assert 0.0 <= greater.pval <= 1.0
    assert 0.0 <= less.pval <= 1.0
    # the one-sided critical values differ, so the tests cannot all coincide
    assert (greater.pval, less.pval) != (two.pval, two.pval)


def test_tes_accepts_standard_errors():
    from_vi = ma.tes(YI, VI)
    from_sei = ma.tes(YI, sei=np.sqrt(VI))
    assert from_sei.pval == pytest.approx(from_vi.pval, abs=1e-10)


def test_tes_accepts_a_fixed_theta():
    res = ma.tes(YI, VI, theta=0.5)
    assert 0.0 <= res.pval <= 1.0


def test_tes_accepts_a_vector_theta():
    res = ma.tes(YI, VI, theta=[0.5] * len(YI))
    assert 0.0 <= res.pval <= 1.0


def test_tes_accepts_tau2():
    res = ma.tes(YI, VI, tau2=0.02)
    assert 0.0 <= res.pval <= 1.0


@pytest.mark.parametrize("test", ["exact", "chi2", "binom"])
def test_tes_test_variants(test):
    res = ma.tes(YI, VI, test=test)
    assert 0.0 <= res.pval <= 1.0


def test_tes_chi2_continuity_correction():
    plain = ma.tes(YI, VI, test="chi2", correct=False)
    corrected = ma.tes(YI, VI, test="chi2", correct=True)
    assert 0.0 <= corrected.pval <= 1.0
    assert corrected.pval != pytest.approx(plain.pval, abs=1e-12)


def test_tes_theta_lim_can_be_skipped():
    res = ma.tes(YI, VI, find_lim=False)
    assert 0.0 <= res.pval <= 1.0


def test_tes_requires_a_variance():
    with pytest.raises(ValueError, match="vi.*sei"):
        ma.tes(YI)


def test_tes_rejects_a_length_mismatch():
    with pytest.raises(ValueError, match="not the same"):
        ma.tes(YI, VI[:-1])


def test_tes_rejects_an_empty_sample():
    with pytest.raises(ValueError, match="k = 0"):
        ma.tes([], [])


def test_tes_rejects_an_unknown_alternative():
    with pytest.raises(ValueError, match="alternative"):
        ma.tes(YI, VI, alternative="both")


# ---------------------------------------------------------------------------
# vif
# ---------------------------------------------------------------------------
MODS = np.array([[1.0, 2.0, 1.0], [2.0, 1.0, 3.0], [3.0, 4.0, 2.0],
                 [4.0, 3.0, 5.0], [5.0, 6.0, 4.0], [6.0, 5.0, 6.0],
                 [7.0, 8.0, 3.0], [8.0, 7.0, 7.0]])
YI8 = [0.10, 0.25, -0.05, 0.40, 0.15, 0.30, 0.05, 0.20]
VI8 = [0.010, 0.020, 0.015, 0.030, 0.012, 0.025, 0.018, 0.022]


@pytest.fixture(scope="module")
def reg():
    return ma.rma(YI8, vi=VI8, mods=MODS)


def test_vif_default_is_one_per_slope(reg):
    res = ma.vif(reg)
    assert len(res.vifs) == MODS.shape[1]
    assert res.bttspec is False
    assert all(v >= 1.0 - 1e-9 for v in res.vifs)


def test_vif_joint_specification(reg):
    res = ma.vif(reg, btt=[[2, 3]])
    assert len(res.vifs) == 1
    assert res.m == [2]
    assert res.bttspec is True


def test_vif_accepts_a_flat_list(reg):
    res = ma.vif(reg, btt=[2, 3])
    assert len(res.vifs) == 1 and res.m == [2]


def test_vif_accepts_coefficient_names(reg):
    res = ma.vif(reg, btt=["mods1"])
    assert len(res.vifs) == 1
    assert res.coefnames[0] == "mods1"


def test_vif_intercept_flag(reg):
    res = ma.vif(reg, intercept=True)
    assert len(res.vifs) == MODS.shape[1] + 1


def test_vif_gvif_bound():
    """One moderator cannot be inflated by the others."""
    one = ma.rma(YI8, vi=VI8, mods=MODS[:, 0])
    res = ma.vif(one)
    assert res.vifs[0] == pytest.approx(1.0, abs=1e-9)


def test_vif_rejects_an_intercept_only_model():
    r = ma.rma(YI8, vi=VI8)
    with pytest.raises(ValueError, match="intercept-only"):
        ma.vif(r)


def test_vif_rejects_a_non_rma_argument():
    with pytest.raises(TypeError, match="RmaResult"):
        ma.vif(object())


def test_vif_rejects_an_empty_specification(reg):
    with pytest.raises(ValueError, match="cannot compute"):
        ma.vif(reg, btt=[99])


# ---------------------------------------------------------------------------
# permutest with moderators (row permutations instead of sign flips)
# ---------------------------------------------------------------------------
def test_permutest_with_moderators_runs(reg):
    res = ma.permutest(reg, iter=199, seed=42)
    # pval holds one entry per model coefficient (intercept + 3 moderators)
    assert res.p == MODS.shape[1] + 1
    assert len(res.pval) == res.p
    assert all(0.0 < v <= 1.0 for v in res.pval)
    assert res.dfs and all(d >= 1 for d in res.dfs)


def test_permutest_exact_with_moderators():
    r = ma.rma(YI8, vi=VI8, mods=MODS[:, 0])
    res = ma.permutest(r, exact=True)
    assert res.exact is True
    assert 0.0 < res.pval[1] <= 1.0


def test_permutest_permci_with_moderators():
    r = ma.rma(YI8, vi=VI8, mods=MODS[:, 0])
    res = ma.permutest(r, iter=199, permci=True, seed=7)
    assert res.ci_lb is not None and res.ci_ub is not None
    assert res.ci_lb[1] < res.ci_ub[1]


def test_permutest_reports_the_omnibus_p_value(reg):
    res = ma.permutest(reg, iter=199, seed=11)
    assert 0.0 <= res.QMp_perm <= 1.0
    assert res.QM == pytest.approx(reg.QM, abs=1e-8)
