"""Coverage of the remaining ``utils`` / ``diagnostics`` branches.

Everything here exercises real logic rather than defensive ``raise`` lines:
the ``rcalc`` row reordering, the model-column accessors, the unit
conversions, the Tukey-style ``tes`` alternatives and the ``permutest``
two-sided conventions.
"""

import math

import numpy as np
import pytest

import pimeta as ma
from pimeta.utils import FitstatsResult, _coef_names, _fitstats_core

YI = [0.10, -0.20, 0.05, -0.15, 0.30, 0.25]
VI = [0.01, 0.0144, 0.0121, 0.0169, 0.0196, 0.0225]
MOD = [1.0, 2.0, 1.5, 3.0, 2.5, 0.5]

R = np.array([[1.0, 0.5, 0.3], [0.5, 1.0, 0.2], [0.3, 0.2, 1.0]])


# --------------------------------------------------------------------------- #
# rcalc: rowid reordering and dnames
# --------------------------------------------------------------------------- #

def test_rcalc_rowid_reorders_the_block_diagonal_matrix():
    out = ma.rcalc([R, R], [100, 120])
    keys = [f"{i}.{a}.{i}.{b}" for i, (a, b) in
            zip([0] * 3 + [1] * 3,
                [("x1", "x2"), ("x1", "x3"), ("x2", "x3")] * 2)]
    flipped = list(reversed(keys))
    reordered = ma.rcalc([R, R], [100, 120], rowid=flipped)
    assert reordered.V.shape == out.V.shape
    # reversing the rows reverses the matching variances on the diagonal
    diag_out = np.diag(np.asarray(out.V, dtype=float))
    diag_new = np.diag(np.asarray(reordered.V, dtype=float))
    assert diag_new == pytest.approx(diag_out[::-1], abs=1e-15)


def test_rcalc_rejects_a_rowid_that_is_not_present():
    with pytest.raises(ValueError):
        ma.rcalc([R, R], [100, 120], rowid=["nope"])


def test_rcalc_dnames_are_used_for_the_labels():
    out = ma.rcalc(R, 100, dnames=["a", "b", "c"])
    assert set(out.dat["var1"]) <= {"a", "b", "c"}
    assert "x1" not in out.dat["var1"]
    plain = ma.rcalc(R, 100)
    assert plain.dat["var1"] == ["x1", "x1", "x2"]


def test_rcalc_rejects_a_dnames_length_mismatch():
    with pytest.raises(ValueError, match="does not match"):
        ma.rcalc(R, 100, dnames=["a", "b"])


# --------------------------------------------------------------------------- #
# coefficient names
# --------------------------------------------------------------------------- #

def test_coef_names_from_a_list_of_strings():
    class Fake:
        p = 3
        mods = ["age", "sex"]
        int_incl = True

    assert _coef_names(Fake()) == ["intrcpt", "age", "sex"]


def test_coef_names_from_a_formula_string():
    class Fake:
        p = 3
        mods = "age + sex"
        int_incl = True

    assert _coef_names(Fake()) == ["intrcpt", "age", "sex"]


def test_coef_names_without_an_intercept():
    class Fake:
        p = 2
        mods = np.zeros((5, 2))
        int_incl = False

    assert _coef_names(Fake()) == ["mod1", "mod2"]


# --------------------------------------------------------------------------- #
# fitstats / column selection
# --------------------------------------------------------------------------- #

def test_fitstats_column_defaults_to_the_models_estimator():
    fs = ma.fitstats(ma.rma(YI, vi=VI, method="REML"))
    assert fs.column() == fs.REML
    fs_ml = ma.fitstats(ma.rma(YI, vi=VI, method="ML"))
    assert fs_ml.column() == fs_ml.ML
    assert fs.column(REML=True) == fs.REML


def test_fitstats_result_round_trips():
    fs = ma.fitstats(ma.rma(YI, vi=VI, method="ML"))
    d = fs.as_dict()
    again = FitstatsResult(d["ML"], d["REML"], fs.method, fs.parms, fs.k, fs.p)
    assert again.column(REML=False) == fs.column(REML=False)
    assert "FitstatsResult" in repr(fs)


def test_fitstats_core_saturated_model():
    class Fake:
        yi = np.array([0.1, 0.2])
        vi = np.array([0.01, 0.02])
        X = np.ones((2, 2))
        b = np.zeros(2)
        tau2 = 0.0
        tau2_fix = True
        method = "FE"

    fs = _fitstats_core(Fake())
    # k == p: no residual degrees of freedom, so the deviance is 0 and R's
    # log(n - p) = log(0) = -Inf makes the REML BIC -Inf.
    assert fs.ML["dev"] == 0.0
    assert fs.REML["BIC"] == -math.inf
    assert math.isfinite(fs.ML["BIC"])


def test_fitstats_on_a_matreg_object():
    m = ma.matreg(1, [2, 3], R, n=100)
    fs = ma.fitstats(m)
    assert fs.k == 100 and fs.p == 3 and fs.method == "ML"
    assert fs.column(REML=False)["AIC"] == pytest.approx(ma.aic_rma(m))


# --------------------------------------------------------------------------- #
# unit conversions / misc utils
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("units", ["in", "cm", "mm", "px"])
def test_inches2units_and_back(units):
    for value in (0.5, 1.0, 3.25):
        back = ma.units2inches(ma.inches2units(value, units), units)
        assert back == pytest.approx(value, abs=1e-12)


def test_unit_conversions_reject_unknown_units():
    with pytest.raises(ValueError, match="units must be"):
        ma.inches2units(1.0, "furlong")
    with pytest.raises(ValueError, match="units must be"):
        ma.units2inches(1.0, "furlong")


def test_as_data_frame_confint_renames_the_point_estimate():
    m = ma.matreg(1, [2, 3], R, n=100)
    rows = ma.as_data_frame_confint(m)
    assert len(rows) == 2
    assert rows[0]["estimate"] == pytest.approx(m.beta[0])
    assert rows[0]["ci.lb"] == pytest.approx(m.ci_lb[0])
    assert rows[0]["ci.ub"] == pytest.approx(m.ci_ub[0])


def test_as_data_frame_confint_from_a_dict():
    rows = ma.as_data_frame_confint({"beta": [1.0, 2.0],
                                     "ci_lb": [0.5, 1.5],
                                     "ci_ub": [1.5, 2.5]})
    assert [r["estimate"] for r in rows] == [1.0, 2.0]
    assert [r["ci.lb"] for r in rows] == [0.5, 1.5]


def test_as_data_frame_confint_keeps_na_rows_on_request():
    rows = ma.as_data_frame_confint({"beta": [1.0, float("nan")],
                                     "ci_lb": [0.5, float("nan")],
                                     "ci_ub": [1.5, float("nan")]},
                                    include_na=True)
    assert len(rows) == 2


# --------------------------------------------------------------------------- #
# tes: every alternative and test
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("alternative", ["two.sided", "greater", "less"])
@pytest.mark.parametrize("test", ["chi2", "binom", "exact"])
def test_tes_alternatives_and_tests(alternative, test):
    out = ma.tes(YI, VI, H0=0.0, alternative=alternative, test=test)
    assert 0.0 <= out.pval <= 1.0
    assert math.isfinite(out.E)


def test_tes_less_and_greater_are_not_symmetric_here():
    # O > E on this data set, so the "less" tail probability is the small one
    greater = ma.tes(YI, VI, H0=0.0, alternative="greater", test="chi2")
    less = ma.tes(YI, VI, H0=0.0, alternative="less", test="chi2")
    assert less.pval < greater.pval


def test_tes_exact_two_sided_uses_the_observed_bin():
    exact = ma.tes(YI, VI, H0=0.0, alternative="two.sided", test="exact")
    one = ma.tes(YI, VI, H0=0.0, alternative="greater", test="exact")
    assert exact.pval >= one.pval * 0.5


# --------------------------------------------------------------------------- #
# permutest: two-sided conventions and the coefficient statistic
# --------------------------------------------------------------------------- #

def test_permutest_px2_convention():
    fit = ma.rma(YI, vi=VI, mods=MOD)
    r = ma.permutest(fit, iter=200, seed=11, p2defn="px2")
    a = ma.permutest(fit, iter=200, seed=11, p2defn="abs")
    assert all(0.0 <= v <= 1.0 for v in r.pval)
    assert r.pval != a.pval          # the conventions differ by construction


def test_permutest_coefficient_statistic():
    fit = ma.rma(YI, vi=VI, mods=MOD)
    r = ma.permutest(fit, iter=200, seed=12, stat="coef")
    assert len(r.pval) == 2
    assert np.asarray(r.zval_perm).shape == (200, 2)


def test_permutest_one_sided():
    fit = ma.rma(YI, vi=VI, mods=MOD)
    g = ma.permutest(fit, iter=200, seed=13, alternative="greater")
    l = ma.permutest(fit, iter=200, seed=13, alternative="less")
    assert all(0.0 <= v <= 1.0 for v in g.pval + l.pval)


def test_permutest_rejects_unknown_options():
    fit = ma.rma(YI, vi=VI, mods=MOD)
    with pytest.raises(ValueError, match="p2defn"):
        ma.permutest(fit, iter=50, p2defn="nope")
    with pytest.raises(ValueError, match="stat"):
        ma.permutest(fit, iter=50, stat="nope")
    with pytest.raises(ValueError, match="alternative"):
        ma.permutest(fit, iter=50, alternative="nope")


def test_permutest_exact_enumerates_every_sign_pattern():
    fit = ma.rma(YI[:4], vi=VI[:4])
    r = ma.permutest(fit, exact=True)
    assert r.exact is True and r.nperm == 2 ** 4


def test_permutest_with_khna_falls_back_to_z():
    fit = ma.rma(YI, vi=VI, test="knha")
    r = ma.permutest(fit, iter=150, seed=14)
    assert len(r.pval) == 1
