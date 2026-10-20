"""Location-scale models (``rma(yi, vi, mods=~x, scale=~z)``).

Reference values were printed by metafor 5.2-1 in R 4.6.1 from the same
24-study dataset.

Convergence note: metafor stops its ``optim``/``constrOptim`` runs on a
*relative function* tolerance, and the location-scale log-likelihood is very
flat in ``alpha``.  On this dataset metafor's own gradient at the reported
estimate is ~1e-3 (identity link) and ~1e-4 (log link), whereas
:func:`pimeta.rma_ls` polishes with Newton steps down to ~1e-9.  The
coefficients therefore agree to ~1e-7 (log link) but ``alpha`` only to ~1e-5
abs; :func:`test_rma_ls_reaches_at_least_r_loglikelihood` pins the fact that
ours is the *better* optimum rather than a wrong one.
"""

import math

import numpy as np
import pytest

import pimeta as ma
from pimeta.rma_ls import (
    RmaLsProfileResult,
    RmaLsResult,
    confint_rma_ls,
    profile_rma_ls,
    rma_ls,
)
from pimeta.s3 import plot

YI = [-0.815857207674611, 1.81115386419507, 0.439857268929623,
      -0.337448955378181, -0.466179706278849, 1.09792416890174,
      0.0590590886071745, 0.682017895138013, 0.31311249092908,
      0.770241366506847, 0.200693202790505, -0.768889181695764,
      0.748184747903724, 2.0544254373089, -0.882623340895745,
      1.10804298887967, 0.544497149756942, 0.0883454202089342,
      0.352249332733821, 0.494147117415249, 2.0886665183714,
      1.57087116547472, 2.13292640551877, 2.21117102552949]
VI = [0.0142210614220239, 0.0150190716185607, 0.0225172992586158,
      0.0112963107670657, 0.0269012653152458, 0.024116185861174,
      0.028924165034201, 0.0251609062999487, 0.01360724150436,
      0.0214548269025981, 0.0265910310312174, 0.0275050001717173,
      0.0184563769772649, 0.0272048096684739, 0.0158918990474194,
      0.0122174268169329, 0.00948380306037143, 0.0269211491486058,
      0.0145599598395638, 0.0286909840856679, 0.017184922627639,
      0.00914465109538287, 0.0147222665799782, 0.0198286695890129]
X = [0.5] * 4 + [1.0] * 4 + [1.5] * 4 + [2.0] * 4 + [2.5] * 4 + [3.0] * 4
Z = [0.3, 0.9, 1.5, 2.1] * 6

# R: rma(yi, vi, mods=~x, scale=~z, method=, link=)
R_FITS = {
    ("REML", "log"): dict(
        beta=[-0.287958168235403, 0.534414670938974],
        se=[0.399101527639523, 0.204762576669817],
        pval=[0.47059205324696401, 0.00905624590991413],
        ci_lb=[-1.070182788583785, 0.133087395284511],
        ci_ub=[0.494266452112980, 0.935741946593436],
        vb=[0.1592820293642010, -0.0734366209199413,
            -0.0734366209199413, 0.0419277128044626],
        tau2=[0.689028170312065, 0.707256771546714,
              0.725967619977180, 0.745173473705683],
        alpha=[-0.3855289373292587, 0.0435193810632569],
        se_alpha=[0.693712768802071, 0.516917298771534],
        va=[0.481237405599035, -0.320851689259189,
            -0.320851689259189, 0.267203493769260],
        QS=0.00708799312992685, QM=6.81170093505347918,
        QE=888.371183034517, QEp=1.04326930273140e-173,
        ll=-27.8490992805628, ll_ml=-29.3787586633905,
        dev_ml=110.396746649772,
        AIC=63.6981985611257, BIC=68.062368374559,
        AICc=66.0511397375963,
        I2_1=97.5052372286725, H2_1=40.0839715700852),
    ("ML", "log"): dict(
        beta=[-0.287930660238911, 0.534496921684003],
        se=[0.382040690863204, 0.195991564677686],
        pval=[0.45105046147150185, 0.00638854468569216],
        ci_lb=[-1.036716654959592, 0.150360513642085],
        ci_ub=[0.46085533448177, 0.91863332972592],
        vb=[0.1459550894752344, -0.0672856044040589,
            -0.0672856044040589, 0.0384126934248077],
        tau2=[0.626614362997187, 0.645464572830919,
              0.664881846606621, 0.684883243100672],
        alpha=[-0.4822434911678399, 0.0493983751865696],
        se_alpha=[0.698572044431693, 0.527210016385376],
        va=[0.488002901261476, -0.333268255633737,
            -0.333268255633737, 0.277950401377069],
        QS=0.00877926226759682, QM=7.43730610426740402,
        QE=888.371183034517, QEp=1.04326930273140e-173,
        ll=-29.3344120131202, ll_ml=-29.3344120131202,
        dev_ml=110.308053349231,
        AIC=63.7843730950313, BIC=68.1485429084645,
        AICc=66.1373142715019,
        I2_1=97.2635466692838, H2_1=36.5436526461157),
    ("REML", "identity"): dict(
        beta=[-0.288553249090435, 0.535014187072486],
        se=[0.398890182649649, 0.204651572205278],
        pval=[0.46944019021797467, 0.00894186985449373],
        ci_lb=[-1.070363640870351, 0.133904476170644],
        ci_ub=[0.493257142689481, 0.936123897974329],
        vb=[0.1591133778142703, -0.0733582406876067,
            -0.0733582406876067, 0.0418822660060920],
        tau2=[0.671038078509301, 0.701756412222555,
              0.732474745935810, 0.763193079649064],
        alpha=[0.6556789116526738, 0.0511972228554241],
        se_alpha=[0.598381240228785, 0.481190966127604],
        va=[0.358060108657739, -0.267476037830178,
            -0.267476037830178, 0.231544745882817],
        QS=0.0113202984507993, QM=6.83440051517743896,
        QE=888.371183034517, QEp=1.04326930273140e-173,
        ll=-27.8468644072883, ll_ml=-29.3754347859608,
        dev_ml=110.390098894912,
        AIC=63.6937288145767, BIC=68.05789862801,
        AICc=66.0466699910473,
        I2_1=97.4400664304233, H2_1=39.063513674121),
    ("ML", "identity"): dict(
        beta=[-0.289030044465511, 0.535465379875206],
        se=[0.381736002032008, 0.195831436164681],
        pval=[0.44896215062606570, 0.00625096815487235],
        ci_lb=[-1.037218860050556, 0.151642817951676],
        ci_ub=[0.459158771119535, 0.919287941798736],
        vb=[0.1457223752473815, -0.0671774365422400,
            -0.0671774365422400, 0.0383499513903216],
        tau2=[0.603205348611256, 0.638422094844350,
              0.673638841077444, 0.708855587310538],
        alpha=[0.5855969754947089, 0.0586945770551567],
        se_alpha=[0.577348675069555, 0.475059193637806],
        va=[0.333331492604571, -0.258293784143488,
            -0.258293784143488, 0.225681237459802],
        QS=0.0152651297664803, QM=7.47649378030917511,
        QE=888.371183034517, QEp=1.04326930273140e-173,
        ll=-29.3309447604439, ll_ml=-29.3309447604439,
        dev_ml=110.301118843878,
        AIC=63.7807219569209, BIC=68.1448917703541,
        AICc=66.1336631333915,
        I2_1=97.1603667612778, H2_1=35.2158154216418),
}


def _fit(method="REML", link="log"):
    return rma_ls(YI, VI, mods=X, scale=Z, method=method, link=link)


# --------------------------------------------------------------------------- #
# coefficients
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("method,link", [("REML", "log"), ("ML", "log")])
def test_rma_ls_log_coefficients_match_r(method, link):
    ref = R_FITS[(method, link)]
    r = _fit(method, link)
    np.testing.assert_allclose(r.b, ref["beta"], rtol=1e-6, atol=1e-9)
    np.testing.assert_allclose(r.se, ref["se"], rtol=1e-6, atol=1e-9)
    np.testing.assert_allclose(r.pval, ref["pval"], rtol=1e-5, atol=1e-9)
    np.testing.assert_allclose(r.ci_lb, ref["ci_lb"], rtol=1e-6, atol=1e-9)
    np.testing.assert_allclose(r.ci_ub, ref["ci_ub"], rtol=1e-6, atol=1e-9)
    np.testing.assert_allclose(np.asarray(r.vb).ravel(), ref["vb"],
                               rtol=1e-6, atol=1e-9)


@pytest.mark.parametrize("method,link", [("REML", "log"), ("ML", "log")])
def test_rma_ls_log_scale_part_matches_r(method, link):
    ref = R_FITS[(method, link)]
    r = _fit(method, link)
    np.testing.assert_allclose(r.alpha, ref["alpha"], rtol=0, atol=1e-5)
    np.testing.assert_allclose(r.se_alpha, ref["se_alpha"], rtol=5e-4)
    np.testing.assert_allclose(np.asarray(r.va).ravel(), ref["va"],
                               rtol=5e-4)
    assert r.QS == pytest.approx(ref["QS"], rel=5e-4)
    assert r.QM == pytest.approx(ref["QM"], rel=1e-6)


@pytest.mark.parametrize("method,link", [("REML", "identity"),
                                         ("ML", "identity")])
def test_rma_ls_identity_matches_r(method, link):
    """The identity link is flat; metafor's own gradient there is ~1e-3."""
    ref = R_FITS[(method, link)]
    r = _fit(method, link)
    np.testing.assert_allclose(r.b, ref["beta"], rtol=5e-3)
    np.testing.assert_allclose(r.se, ref["se"], rtol=5e-3)
    np.testing.assert_allclose(r.alpha, ref["alpha"], rtol=5e-3, atol=2e-4)
    np.testing.assert_allclose(r.se_alpha, ref["se_alpha"], rtol=5e-3)
    np.testing.assert_allclose(np.asarray(r.va).ravel(), ref["va"],
                               rtol=5e-3)
    assert r.ll == pytest.approx(ref["ll"], rel=0, abs=2e-6)


@pytest.mark.parametrize("method,link", sorted(R_FITS))
def test_rma_ls_reaches_at_least_r_loglikelihood(method, link):
    """metafor stops on a relative *function* tolerance on a very flat surface.

    Our Newton polish goes further, so the log-likelihood must be at least as
    good as metafor's (never worse) and agree to ~1e-6.
    """
    ref = R_FITS[(method, link)]
    r = _fit(method, link)
    assert r.ll >= ref["ll"] - 1e-9
    assert abs(r.ll - ref["ll"]) < 1e-6


# --------------------------------------------------------------------------- #
# heterogeneity and fit statistics
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("method,link", sorted(R_FITS))
def test_rma_ls_heterogeneity_matches_r(method, link):
    ref = R_FITS[(method, link)]
    r = _fit(method, link)
    assert r.k == 24 and r.p == 2 and r.q == 2
    assert r.parms == 4                       # p + q, as in metafor
    assert r.QE == pytest.approx(ref["QE"], rel=1e-12)
    assert r.QEp == pytest.approx(ref["QEp"], rel=1e-6)
    np.testing.assert_allclose(r.tau2[:4], ref["tau2"], rtol=1e-3)
    # tau^2 repeats over the six distinct z values
    assert len(r.tau2) == 24
    assert r.tau2[0] == r.tau2[4] == r.tau2[8]
    assert r.I2[0] == pytest.approx(ref["I2_1"], rel=1e-3)
    assert r.H2[0] == pytest.approx(ref["H2_1"], rel=1e-3)
    assert r.R2 is None                       # metafor returns NULL here

# R: r$fit.stats for the *fitted* estimator (ll / AIC / BIC / AICc).
R_FITSTATS = {
    ("REML", "log"): (-27.8490992805628, 63.6981985611257,
                      68.062368374559, 66.0511397375963),
    ("ML", "log"): (-29.3344120131202, 66.6688240262404,
                    71.3810393476322, 68.7740871841352),
    ("REML", "identity"): (-27.8468644072883, 63.6937288145767,
                           68.05789862801, 66.0466699910473),
    ("ML", "identity"): (-29.3309447604439, 66.6618895208877,
                         71.3741048422795, 68.7671526787825),
}


@pytest.mark.parametrize("method,link", sorted(R_FITS))
def test_rma_ls_fit_statistics_match_r(method, link):
    ll_, aic, bic, aicc = R_FITSTATS[(method, link)]
    col = _fit(method, link).fit_stats[method]
    assert col["ll"] == pytest.approx(ll_, rel=0, abs=2e-6)
    assert col["AIC"] == pytest.approx(aic, rel=0, abs=2e-6)
    assert col["BIC"] == pytest.approx(bic, rel=0, abs=2e-6)
    assert col["AICc"] == pytest.approx(aicc, rel=0, abs=2e-6)


@pytest.mark.parametrize("method,link", sorted(R_FITS))
def test_rma_ls_ml_deviance_matches_r(method, link):
    """The ML deviance is evaluated at the optimum, so it tracks alpha to 1e-3."""
    ref = R_FITS[(method, link)]
    col = _fit(method, link).fit_stats["ML"]
    assert col["dev"] == pytest.approx(ref["dev_ml"], rel=0, abs=1e-2)


# --------------------------------------------------------------------------- #
# integration with rma()
# --------------------------------------------------------------------------- #

def test_rma_scale_delegates_to_rma_ls():
    r = ma.rma(YI, VI, mods=X, scale=Z, method="REML")
    assert isinstance(r, RmaLsResult)
    assert r.model == "rma.ls"
    assert r.alpha[1] == pytest.approx(0.0435193810632569, rel=0, abs=5e-6)


def test_rma_ls_accepts_column_names_with_data():
    d = {"yi": YI, "vi": VI, "x": X, "z": Z}
    r = rma_ls("yi", "vi", mods="x", scale="z", data=d, method="REML")
    assert r.alpha[0] == pytest.approx(-0.3855289373292587, rel=0, abs=5e-6)


def test_rma_ls_no_intercepts():
    r = rma_ls(YI, VI, mods=X, scale=Z, intercept=False,
               scale_intercept=False, method="REML")
    assert r.p == 1 and r.q == 1
    assert r.intercept is False and r.scale_intercept is False
    assert r.Z_int_incl is False


def test_rma_ls_fixed_alpha():
    r = rma_ls(YI, VI, mods=X, scale=Z, method="REML",
               alpha=[math.nan, 1.0])
    assert r.alpha[1] == pytest.approx(1.0)
    assert r.alpha_fix == [False, True]
    assert r.parms == 3                       # p + one free alpha


def test_rma_ls_result_protocol():
    r = _fit()
    assert r.beta is r.b and r.coef is r.b
    d = r.to_dict()
    assert d["link"] == "log" and d["q"] == 2
    assert len(d["tau2"]) == 24
    assert "alpha" in repr(r)


# --------------------------------------------------------------------------- #
# validation
# --------------------------------------------------------------------------- #

def test_rma_ls_rejects_bad_link():
    with pytest.raises(ValueError, match="link must be"):
        rma_ls(YI, VI, mods=X, scale=Z, link="probit")


def test_rma_ls_rejects_non_ml_method():
    with pytest.raises(ValueError, match="ML or REML"):
        rma_ls(YI, VI, mods=X, scale=Z, method="DL")


def test_rma_ls_requires_scale():
    with pytest.raises(ValueError, match="'scale' is required"):
        rma_ls(YI, VI, mods=X)


def test_rma_ls_rejects_bad_alpha_length():
    with pytest.raises(ValueError, match="does not match"):
        rma_ls(YI, VI, mods=X, scale=Z, alpha=[1.0, 2.0, 3.0])


def test_rma_ls_rejects_all_fixed_alpha():
    with pytest.raises(ValueError, match="does not contain any estimated"):
        rma_ls(YI, VI, mods=X, scale=Z, alpha=[0.1, 0.2])


def test_rma_ls_rejects_intercept_false_without_mods():
    with pytest.raises(ValueError, match="intercept=False"):
        rma_ls(YI, VI, scale=Z, intercept=False)


def test_rma_ls_requires_vi_or_sei():
    with pytest.raises(ValueError, match="vi' or 'sei'"):
        rma_ls(YI, scale=Z)


def test_rma_ls_rejects_missing_data_column():
    with pytest.raises(ValueError, match="column 'nope'"):
        rma_ls("yi", "vi", mods="nope", scale="z",
               data={"yi": YI, "vi": VI, "z": Z})


# --------------------------------------------------------------------------- #
# profile likelihood
# --------------------------------------------------------------------------- #

# R: profile(r, alpha = 2, steps = 6, plot = FALSE) on the REML log-link fit
R_PROFILE = dict(
    grid=[-1.287971344623803, -0.755375054348979, -0.222778764074155,
          0.309817526200669, 0.842413816475492, 1.375010106750316],
    ll=[-30.9998161245351, -29.0237704091868, -27.9819664889111,
        -27.9805370420149, -28.9808846865694, -30.7760038089331],
    beta1=[-0.474384463533048, -0.371917563515823, -0.300296426082449,
           -0.295292679896446, -0.367477151935095, -0.494939964320949],
)


def test_profile_rma_ls_explicit_grid_matches_r():
    prof = profile_rma_ls(_fit(), alpha=2, steps=R_PROFILE["grid"])
    assert isinstance(prof, RmaLsProfileResult)
    np.testing.assert_allclose(prof.ll, R_PROFILE["ll"], rtol=0, atol=1e-11)
    np.testing.assert_allclose([b[0] for b in prof.beta],
                               R_PROFILE["beta1"], rtol=0, atol=1e-8)


def test_profile_rma_ls_default_window_is_metafor():
    r = _fit()
    prof = profile_rma_ls(r, alpha=2, steps=6)
    assert len(prof.alpha) == 6
    half = 2.5758293035489004 * r.se_alpha[1]      # qnorm(0.995)
    assert prof.alpha[0] == pytest.approx(r.alpha[1] - half, rel=1e-12)
    assert prof.alpha[-1] == pytest.approx(r.alpha[1] + half, rel=1e-12)
    # the grid straddles the fitted value and the profile peaks near it
    assert prof.alpha[0] < r.alpha[1] < prof.alpha[-1]
    assert max(prof.ll) <= prof.maxll + 1e-9


def test_profile_rma_ls_other_coefficient_stays_free():
    """Only the profiled coefficient is pinned; alpha_1 is re-estimated."""
    prof = profile_rma_ls(_fit(), alpha=2, steps=3)
    assert all(b[0] != pytest.approx(-0.287958168235403, rel=1e-9)
               for b in prof.beta)


def test_profile_rma_ls_defaults_to_first_free_coefficient():
    prof = profile_rma_ls(_fit(), steps=4)
    assert prof.index == 1


def test_profile_rma_ls_rejects_out_of_range():
    with pytest.raises(ValueError, match="No such 'alpha'"):
        profile_rma_ls(_fit(), alpha=3)
    with pytest.raises(ValueError, match="only specify one"):
        profile_rma_ls(_fit(), alpha=[1, 2])


def test_profile_rma_ls_rejects_fixed_component():
    r = rma_ls(YI, VI, mods=X, scale=Z, alpha=[math.nan, 1.0])
    with pytest.raises(ValueError, match="was fixed"):
        profile_rma_ls(r, alpha=2)


def test_profile_rma_ls_rejects_optbeta():
    r = _fit()
    r.optbeta = True
    with pytest.raises(ValueError, match="optbeta"):
        profile_rma_ls(r)


def test_profile_rma_ls_rejects_bad_steps_and_xlim():
    with pytest.raises(ValueError, match="steps"):
        profile_rma_ls(_fit(), steps=1)
    with pytest.raises(ValueError, match="length 2"):
        profile_rma_ls(_fit(), xlim=[1.0, 2.0, 3.0])


def test_profile_rma_ls_plot_routing(tmp_path):
    r = _fit()
    out = tmp_path / "ls.png"
    res = plot(r, which="profile", filename=str(out), steps=6)
    assert out.exists() and out.stat().st_size > 0
    assert res.figure is not None

    prof = profile_rma_ls(r, steps=6)
    out2 = tmp_path / "ls2.png"
    plot(prof, filename=str(out2))
    assert out2.exists()


# --------------------------------------------------------------------------- #
# Wald intervals for the scale coefficients
# --------------------------------------------------------------------------- #

def test_confint_rma_ls_matches_the_wald_ci():
    r = _fit()
    ci = confint_rma_ls(r)
    assert ci["level"] == pytest.approx(0.95)
    assert ci["parm"] == ["z1", "z2"]
    crit = 1.959963984540054
    assert ci["ci_lb"][1] == pytest.approx(r.alpha[1] - crit * r.se_alpha[1],
                                          rel=1e-12)
    assert ci["ci_ub"][1] == pytest.approx(r.alpha[1] + crit * r.se_alpha[1],
                                          rel=1e-12)
    # metafor's own Wald limits
    assert ci["ci_lb"][0] == pytest.approx(-1.745180979796878, rel=0, abs=5e-4)
    assert ci["ci_ub"][0] == pytest.approx(0.974123105138361, rel=0, abs=5e-4)


def test_confint_rma_ls_selection():
    r = _fit()
    ci = confint_rma_ls(r, parm=[2], level=0.9)
    assert ci["parm"] == ["z2"]
    assert ci["level"] == pytest.approx(0.90)
    with pytest.raises(ValueError, match="does not exist"):
        confint_rma_ls(r, parm=[5])
    with pytest.raises(ValueError, match="unknown scale coefficient"):
        confint_rma_ls(r, parm=["nope"])
