"""R-pinned tests for the formatting helpers (``fmt.py``) and the
effect-size conversion utilities (``conv.py``) plus ``extras.py``.
"""
import math

import numpy as np
import pytest

import pimeta as ma

# --------------------------------------------------------------------------- #
# metafor::fmtx / fmtt / fmtp / fmtp2 / dfround
# --------------------------------------------------------------------------- #

def test_fmtx_matches_r():
    assert ma.fmtx([1.23456, 123456, 0.0001234], digits=3) == [
        "1.235", "123456.000", "0.000"]


def test_fmtx_sign_flag_matches_r():
    """R: fmtx(c(1.23456, -2.5), digits = 4, flag = "+")."""
    assert ma.fmtx([1.23456, -2.5], digits=4, flag="+") == [
        "+1.2346", "-2.5000"]


def test_fmtx_space_flag_pads_positives():
    assert ma.fmtx([1.5, -2.5], digits=2, flag=" ") == [" 1.50", "-2.50"]


def test_fmtx_matrix_and_nan():
    got = ma.fmtx(np.array([[1.23456, 2.0], [np.nan, 3.0]]), digits=2)
    assert got[0][0] == "1.23"
    assert got[1][0] == "NA"


def test_fmtx_drop0ifint_and_add0():
    assert ma.fmtx(2.0, digits=2, drop0ifint=True) == "2"
    assert ma.fmtx(0.5, digits=2, add0=False) == ".50"


def test_fmtx_thresh():
    assert ma.fmtx(0.0004, digits=4, thresh=0.001) == "0"
    with pytest.raises(ValueError, match="thresh"):
        ma.fmtx([0.1, 0.2], thresh=0.001)


def test_fmtp_matches_r():
    assert ma.fmtp([0.0001, 0.049, 0.5], digits=3) == ["<.001", "0.049", "0.500"]


def test_fmtp_name_sep_add0_matches_r():
    assert ma.fmtp([0.0001, 0.049], digits=3, pname="p", sep=True,
                   add0=True) == ["p < 0.001", "p  0.049"]


def test_fmtp2_matches_r():
    assert ma.fmtp2([0.0001, 0.03, 0.5]) == ["p < .001", "p = .030", "p = .50"]
    assert ma.fmtp2([0.0001, 0.5], pname="p", sep=True, add0=True,
                    quote=True) == ["p < 0.001", "p = 0.50"]


def test_fmtt_matches_r():
    assert ma.fmtt(2.34567, tname="t", df=12, digits=3,
                   pval=0.0345) == "t(df = 12) = 2.346, p-val = 0.035"
    assert ma.fmtt(3.1, tname="F", df1=2, df2=15, digits=3,
                   pval=0.04) == "F(df1 = 2, df2 = 15) = 3.100, p-val = 0.040"


def test_dfround():
    """dfround(): drop0=True rounds to numbers, drop0=False formats to strings."""
    got = ma.dfround({"a": [1.23456, 2.5, 3.0]}, digits=2)
    assert list(got["a"]) == pytest.approx([1.23, 2.5, 3.0])
    got = ma.dfround({"a": [1.23456, 2.5]}, digits=1, drop0=False)
    assert list(got["a"]) == ["1.2", "2.5"]


def test_dfround_passes_through_non_numeric_columns():
    got = ma.dfround({"a": [1.23456], "b": ["x"]}, digits=1)
    assert list(got["b"]) == ["x"]


# --------------------------------------------------------------------------- #
# meta::conv.*
# --------------------------------------------------------------------------- #

def test_conv_delta_matches_the_delta_method():
    """conv_delta() takes a variance and returns a variance (package convention)."""
    r = ma.conv_delta(0.5, 0.04, math.exp)
    # exp(0.5) with var = (exp(0.5))^2 * 0.04; math.exp forces the central
    # difference fallback, so allow for that
    assert r.yi[0] == pytest.approx(math.exp(0.5), abs=1e-12)
    assert r.vi[0] == pytest.approx(math.exp(0.5) ** 2 * 0.04, abs=1e-9)
    # a vectorised transform uses the complex-step derivative: exact
    r = ma.conv_delta(0.5, 0.04, np.exp)
    assert r.vi[0] == pytest.approx(math.exp(0.5) ** 2 * 0.04, abs=1e-12)


def test_conv_wald_from_a_ci():
    """The SE is recovered from the CI width and squared back to a variance."""
    r = ma.conv_wald(0.5, 0.1, 0.9)
    assert r.yi[0] == pytest.approx(0.5, abs=1e-12)
    # ((0.9 - 0.1) / (2 * qnorm(0.975)))^2 = 0.0416508...
    assert r.vi[0] == pytest.approx(0.04165084, abs=1e-8)


def test_conv_fivenum_test_flag_controls_the_skewness_screen():
    """metafor also returns NA for this summary when test = TRUE."""
    r = ma.conv_fivenum(min=1, q1=2, median=3, q3=4.5, max=7, n=50)
    assert np.isnan(r.mean) and np.isnan(r.sd)
    r = ma.conv_fivenum(min=1, q1=2, median=3, q3=4.5, max=7, n=50,
                        test=False)
    assert 1.0 <= r.mean <= 7.0
    assert np.isfinite(r.sd) and r.sd > 0


def test_conv_fivenum_rejects_bad_input():
    with pytest.raises(ValueError, match="at least some"):
        ma.conv_fivenum()
    with pytest.raises(ValueError, match="method"):
        ma.conv_fivenum(min=1, median=3, max=7, n=50, method="nope")


# --------------------------------------------------------------------------- #
# extras.py
# --------------------------------------------------------------------------- #

BCGY = [-0.88931133, -1.58538866, -1.34807315, -1.44155119, -0.21754732,
        -0.78611559, -1.59664331, -0.32966938, -0.55783318, -0.49888521,
        -0.96960854, -0.48266968, -0.41683796]
BCGV = [0.13320328, 0.26373142, 0.21275645, 0.14693293, 0.14293235,
        0.12151837, 0.44437921, 0.15956733, 0.13432539, 0.33222552,
        0.20969889, 0.17053331, 0.13000306]


def test_hettest_runs():
    fit = ma.rma(BCGY, BCGV, method="REML")
    out = ma.hettest(fit)
    assert out is not None


def test_vcalc_runs():
    # BCGV holds sampling *variances*, so it is the 'vi' argument; BCGY holds
    # the effects, which belong in rma() rather than here.
    out = ma.vcalc(BCGV, cluster=[1] * 7 + [2] * 6,
                   obs=[0, 1, 2, 3, 4, 5, 6, 0, 1, 2, 3, 4, 5], rho=0.5)
    assert out.shape == (13, 13)
    assert np.allclose(np.diag(out), BCGV)
    assert np.allclose(out, out.T)


def test_matreg_runs():
    R = np.array([[1.0, 0.4, 0.3],
                  [0.4, 1.0, 0.2],
                  [0.3, 0.2, 1.0]])
    out = ma.matreg(1, [2, 3], R, n=100)
    assert out is not None


def test_profile_rma_runs():
    fit = ma.rma(BCGY, BCGV, method="REML")
    out = ma.profile_rma(fit)
    assert out is not None


def test_simulate_rma_runs():
    fit = ma.rma(BCGY, BCGV, method="REML")
    out = np.asarray(ma.simulate_rma(fit, nsim=3, seed=1))
    assert out.shape == (13, 3)
    assert np.all(np.isfinite(out))
