"""``matreg`` pinned to R 4.6.1 / metafor 5.2-1.

The old implementation pooled the y-x correlations in an intercept-only
fixed-effect model, which is not what ``metafor::matreg`` does.  These tests
pin the standardised path coefficients ``b = R[x,x]^-1 R[x,y]``, the residual
mean square standard errors, the covariance-matrix branch (delta method over
the unique elements of ``R``) and the fit statistics.
"""

import math

import numpy as np
import pytest

import pimeta as ma

R4 = np.array([[1.0, 0.5, 0.3, 0.2],
               [0.5, 1.0, 0.2, 0.4],
               [0.3, 0.2, 1.0, 0.6],
               [0.2, 0.4, 0.6, 1.0]])
NAMES = ["y", "x1", "x2", "x3"]


def _fit_234():
    return ma.matreg(1, [2, 3, 4], R4, n=100, names=NAMES)


def _fit_23():
    return ma.matreg(1, [2, 3], R4, n=100, names=NAMES)


# --------------------------------------------------------------------------- #
# n-branch (test = "t")
# --------------------------------------------------------------------------- #

def test_matreg_t_branch_coefficients():
    m = _fit_234()
    assert m.test == "t" and m.intercept is False
    assert m.beta == pytest.approx(
        [0.514925373134, 0.313432835821, -0.194029850746], abs=1e-11)
    assert m.se == pytest.approx(
        [0.0919812147801, 0.1053777198109, 0.1126535210663], abs=1e-11)
    assert m.tval == pytest.approx(
        [5.59815799743, 2.97437481456, -1.72235939818], abs=1e-10)
    assert m.pval == pytest.approx(
        [2.01313949360e-07, 3.70459317137e-03, 8.81914945022e-02], abs=1e-13)
    assert m.ci_lb == pytest.approx(
        [0.332368123483, 0.104287232023, -0.417615905993], abs=1e-11)
    assert m.ci_ub == pytest.approx(
        [0.6974826227861, 0.5225784396190, 0.0295562045001], abs=1e-11)


def test_matreg_t_branch_vcov():
    vb = np.asarray(_fit_234().vb, dtype=float)
    assert vb == pytest.approx(np.array([
        [0.008460543872421, 0.000528783992026, -0.00370148794418],
        [0.000528783992026, 0.011104463832553, -0.00687419189634],
        [-0.003701487944184, -0.006874191896342, 0.01269081580863],
    ]), abs=1e-12)


def test_matreg_t_branch_goodness_of_fit():
    m = _fit_234()
    assert m.R2 == pytest.approx(0.31268656716418, abs=1e-13)
    assert m.R2adj == pytest.approx(0.29142945068472, abs=1e-13)
    assert m.F == pytest.approx([14.709735794426, 3.0, 97.0], abs=1e-11)
    assert m.Fp == pytest.approx(5.7222740190291e-08, rel=1e-11)
    assert m.df == 97 and m.nobs == 100
    assert m.sigma2_ml == pytest.approx(0.68044029850746, abs=1e-13)
    assert m.sigma2_reml == pytest.approx(0.70148484382213, abs=1e-13)
    assert m.deviance == pytest.approx(68.044029850746, abs=1e-12)


def test_matreg_fit_stats_match_r():
    fs = _fit_23().fit_stats
    assert fs["ML"]["ll"] == pytest.approx(-124.14931221321, abs=1e-11)
    assert fs["ML"]["dev"] == pytest.approx(70.125, abs=1e-12)
    assert fs["ML"]["AIC"] == pytest.approx(254.29862442641, abs=1e-11)
    assert fs["ML"]["BIC"] == pytest.approx(262.11413498438, abs=1e-11)
    assert fs["ML"]["AICc"] == pytest.approx(254.54862442641, abs=1e-11)
    assert fs["REML"]["ll"] == pytest.approx(-127.23096748037, abs=1e-11)
    assert fs["REML"]["dev"] == pytest.approx(254.46193496075, abs=1e-11)
    assert fs["REML"]["AIC"] == pytest.approx(260.46193496075, abs=1e-11)
    assert fs["REML"]["BIC"] == pytest.approx(268.21683739676, abs=1e-11)
    assert fs["REML"]["AICc"] == pytest.approx(260.71725410969, abs=1e-11)


def test_matreg_two_predictors():
    m = _fit_23()
    assert m.beta == pytest.approx([0.458333333333, 0.208333333333], abs=1e-12)
    assert m.se == pytest.approx([0.0867701174828, 0.0867701174828], abs=1e-13)
    assert m.R2 == pytest.approx(0.29166666666667, abs=1e-13)
    assert m.R2adj == pytest.approx(0.27721088435374, abs=1e-13)
    assert m.F == pytest.approx([20.176470588235, 2.0, 98.0], abs=1e-11)
    assert m.Fp == pytest.approx(4.5882418691747e-08, rel=1e-11)


def test_matreg_covariance_matrix_branch():
    m = ma.matreg(1, [2, 3], R4, n=100, cov=True, names=NAMES)
    assert m.intercept is True
    assert m.names == ["intrcpt", "x1", "x2"]
    assert math.isnan(m.beta[0])          # no means supplied -> NA intercept
    assert m.beta[1:] == pytest.approx([0.458333333333, 0.208333333333],
                                       abs=1e-12)
    assert m.se[1:] == pytest.approx([0.0872162392794, 0.0872162392794],
                                     abs=1e-12)
    assert m.R2adj == pytest.approx(0.2770618556701, abs=1e-12)
    assert m.F == pytest.approx([19.970588235294, 2.0, 97.0], abs=1e-11)
    assert m.Fp == pytest.approx(5.4516436960153e-08, rel=1e-11)
    vb = np.asarray(m.vb, dtype=float)
    assert vb[1:, 1:] == pytest.approx(np.array([
        [0.00760667239404, -0.00152133447881],
        [-0.00152133447881, 0.00760667239404]]), abs=1e-12)
    assert np.all(np.isnan(vb[0, :])) and np.all(np.isnan(vb[:, 0]))
    assert m.sigma2_ml == pytest.approx(0.70125, abs=1e-13)
    assert m.sigma2_reml == pytest.approx(0.7229381443299, abs=1e-13)
    assert m.fit_stats["REML"]["ll"] == pytest.approx(-128.77939835569, abs=1e-11)


def test_matreg_covariance_matrix_with_means():
    means = [0.0, 1.0, 2.0, 3.0]
    m = ma.matreg(1, [2, 3], R4, n=100, cov=True, means=means, names=NAMES)
    assert np.all(np.isfinite(m.beta))
    assert np.all(np.isfinite(np.asarray(m.vb, dtype=float)))
    # b0 = mean(y) - means(x)' b
    expected = means[0] - (np.array([1.0, 2.0]) @ np.array(m.beta[1:]))
    assert m.beta[0] == pytest.approx(expected, abs=1e-12)


def test_matreg_level_conventions():
    assert _fit_23().level == pytest.approx(0.05)
    assert ma.matreg(1, [2, 3], R4, n=100, level=90).level == pytest.approx(0.1)
    lo = ma.matreg(1, [2, 3], R4, n=100, level=90).ci_lb
    # 1e-10 rather than 1e-12: the reference is R's own output, and the
    # matrix solve behind it accumulates in a different order on different
    # BLAS builds (Python 3.10 on the CI runner lands 1.9e-12 away).
    assert lo == pytest.approx([0.3142471091423, 0.0642471091423], abs=1e-10)
    hi = ma.matreg(1, [2, 3], R4, n=100, level=0.95).ci_lb
    assert hi == pytest.approx(_fit_23().ci_lb, abs=1e-13)


def test_matreg_variable_names_and_default_predictors():
    by_name = ma.matreg("y", ["x1", "x2"], R4, n=100, names=NAMES)
    assert by_name.beta == pytest.approx(_fit_23().beta, abs=1e-14)
    default = ma.matreg(1, None, R4, n=100, names=NAMES)
    assert default.p == 3
    assert default.beta == pytest.approx(_fit_234().beta, abs=1e-14)


# --------------------------------------------------------------------------- #
# V-branch (test = "z")
# --------------------------------------------------------------------------- #

def test_matreg_z_branch_delta_method():
    V = np.diag(np.repeat(0.01, 6))
    m = ma.matreg(1, [2, 3, 4], R4, V=V)
    assert m.test == "z" and m.intercept is False
    assert m.beta == pytest.approx(
        [0.514925373134, 0.313432835821, -0.194029850746], abs=1e-11)
    assert m.se == pytest.approx(
        [0.146863729853, 0.217401629750, 0.254392822225], abs=1e-11)
    assert m.zval == pytest.approx(
        [3.506143917550, 1.441722567496, -0.762717473902], abs=1e-11)
    assert m.pval == pytest.approx(
        [0.000454649317642, 0.149380654567199, 0.445631907166613], abs=1e-14)
    assert m.ci_lb == pytest.approx(
        [0.227077751986, -0.112666528669, -0.692630620233], abs=1e-11)
    assert m.QM == pytest.approx(30.810856956715, abs=1e-11)
    assert m.QMp == pytest.approx(9.3167631653919e-07, rel=1e-11)
    assert m.R2 == pytest.approx(0.31268656716418, abs=1e-13)


def test_matreg_z_branch_vcov():
    V = np.diag(np.repeat(0.01, 6))
    vb = np.asarray(ma.matreg(1, [2, 3, 4], R4, V=V).vb, dtype=float)
    assert vb == pytest.approx(np.array([
        [0.0215689551465, 0.0142126584422, -0.0255157051809],
        [0.0142126584422, 0.0472634686179, -0.0482976821736],
        [-0.0255157051809, -0.0482976821736, 0.0647157079996],
    ]), abs=1e-12)


def test_matreg_z_branch_has_no_fit_statistics():
    V = np.diag(np.repeat(0.01, 6))
    m = ma.matreg(1, [2, 3], R4, V=V)
    assert m.fit_stats is None and m.nobs is None
    with pytest.raises(ValueError, match="fit statistics"):
        ma.fitstats(m)
    with pytest.raises(ValueError, match="sigma"):
        ma.sigma_matreg(m)


# --------------------------------------------------------------------------- #
# validation
# --------------------------------------------------------------------------- #

def test_matreg_requires_n_or_v():
    with pytest.raises(ValueError, match="Either 'V' or 'n'"):
        ma.matreg(1, [2, 3], R4)
    with pytest.raises(ValueError, match="Either 'V' or 'n'"):
        ma.matreg(1, [2, 3], R4, n=100, V=np.eye(6))


def test_matreg_rejects_bad_input():
    with pytest.raises(ValueError, match="'R' argument"):
        ma.matreg(1, [2, 3], None, n=100)
    with pytest.raises(ValueError, match="square matrix"):
        ma.matreg(1, [2, 3], np.ones((3, 2)), n=100)
    with pytest.raises(ValueError, match="at least of size 2x2"):
        ma.matreg(1, [1], np.array([[1.0]]), n=100)
    with pytest.raises(ValueError, match="duplicated"):
        ma.matreg(1, [2, 2], R4, n=100)
    with pytest.raises(ValueError, match="should not be an element"):
        ma.matreg(1, [1, 2], R4, n=100)
    with pytest.raises(ValueError, match="Indices in 'x'"):
        ma.matreg(1, [9], R4, n=100)
    with pytest.raises(ValueError, match="outside \\[-1,1\\]"):
        bad = R4.copy()
        bad[0, 1] = bad[1, 0] = 2.0
        ma.matreg(1, [2, 3], bad, n=100)
    with pytest.raises(ValueError, match="'n' should be a single number"):
        ma.matreg(1, [2, 3], R4, n=[100, 200])
    with pytest.raises(ValueError, match="'cov' must be either TRUE or FALSE"):
        ma.matreg(1, [2, 3], R4, n=100, cov="yes")
    with pytest.raises(ValueError, match="'ztor' must be either TRUE or FALSE"):
        ma.matreg(1, [2, 3], R4, n=100, ztor=1)
    with pytest.raises(ValueError, match="ztor=True"):
        ma.matreg(1, [2, 3], R4, n=100, cov=True, ztor=True)
    with pytest.raises(ValueError, match="do not match"):
        ma.matreg(1, [2, 3], R4, V=np.eye(4))


def test_matreg_rejects_too_small_n():
    with pytest.raises(ValueError, match="Cannot fit model"):
        ma.matreg(1, [2, 3], R4, n=2)


def test_matreg_names_are_required_for_strings():
    with pytest.raises(ValueError, match="dimension names"):
        ma.matreg("y", ["x1"], R4, n=100)
    with pytest.raises(ValueError, match="Could not find variable"):
        ma.matreg("nope", ["x1"], R4, n=100, names=NAMES)


def test_matreg_nearpd_repairs_a_deficient_matrix():
    # x1/x2 perfectly correlated -> R[x,x] is singular
    R = np.array([[1.0, 0.5, 0.5, 0.1],
                  [0.5, 1.0, 1.0, 0.2],
                  [0.5, 1.0, 1.0, 0.2],
                  [0.1, 0.2, 0.2, 1.0]])
    with pytest.raises(ValueError, match="Cannot invert"):
        ma.matreg(1, [2, 3], R, n=100)
    m = ma.matreg(1, [2, 3], R, n=100, nearpd=True)
    assert np.all(np.isfinite(m.beta))


def test_matreg_ztor_round_trip():
    # feed Fisher-z R values and ztor=TRUE: tanh() recovers the correlations
    Rz = np.arctanh(R4 * 0.999)
    np.fill_diagonal(Rz, 0.0)
    ztor = ma.matreg(1, [2, 3], Rz, n=100, ztor=True)
    plain = ma.matreg(1, [2, 3], np.tanh(Rz), n=100)
    assert ztor.beta == pytest.approx(plain.beta, abs=1e-12)


def test_matreg_helpers_on_the_fitted_model():
    m = _fit_23()
    fs = ma.fitstats(m)
    assert fs.method == "ML" and fs.parms == 3
    assert ma.fitstats(m, REML=True)["AIC"] == pytest.approx(
        260.46193496075, abs=1e-11)
    assert ma.fitstats(m, REML=False)["AIC"] == pytest.approx(
        254.29862442641, abs=1e-11)
    assert ma.aic_rma(m) == pytest.approx(254.29862442641, abs=1e-11)
    assert ma.aic_rma(m, correct=True) == pytest.approx(254.54862442641,
                                                        abs=1e-11)
    assert ma.bic_rma(m) == pytest.approx(262.11413498438, abs=1e-11)
    assert ma.sigma_matreg(m) == pytest.approx(math.sqrt(0.7155612244898),
                                               abs=1e-13)
    assert ma.sigma_matreg(m, REML=False) == pytest.approx(
        math.sqrt(0.70125), abs=1e-13)
