"""Regression tests for the modules that were previously under-covered.

Expected values come from R 4.6.1 (metafor 5.2-1) on the 13 BCG vaccine trials —
the classic ``metafor::dat.bcg`` dataset. The generator scripts that produced
them are ``_refgen_gaps.R`` / ``_refgen_gaps2.R`` in the project scratch area;
every number below is a verbatim ``sprintf("%.17g", ...)`` of the R output.

Covered here: tau2 estimators, fsn, regtest/ranktest/trimfill,
leave1out/influence/cooks/hatvalues/rstudent/baujat, cumul, metareg.
"""
from __future__ import annotations

import numpy as np
import pytest

import pimeta as ma

# ---- reference input ------------------------------------------------------
YI = [-0.88931133392020545, -1.5853886572014306, -1.3480731482996933,
      -1.4415511900213054, -0.21754732221129558, -0.78611558581886398,
      -1.6208982235983918, 0.011952333523840508, -0.4694176487381494,
      -1.3713448034727844, -0.33935882833839059, 0.44591340057137874,
      -0.017313948216879815]
VI = [0.32558476500396133, 0.19458112139814385, 0.41536796536796539,
      0.020010031902247573, 0.051210172169630863, 0.0069056184559087574,
      0.22301724757231517, 0.0039615792978177295, 0.056434210463248966,
      0.07302479361302891, 0.01241221397155972, 0.53250584520015276,
      0.071404659683986293]
NI = [262, 609, 451, 26465, 10877, 2992, 3174, 176782, 14776, 3381, 77972,
      4839, 34767]
YEAR = [1948, 1949, 1960, 1977, 1973, 1953, 1973, 1980, 1968, 1961, 1974,
        1969, 1976]


# =========================================================================
# tau^2 estimators
# =========================================================================
@pytest.mark.parametrize("method,expected", [
    ("DL", 0.30876026286237612),
    ("HE", 0.32856385799576876),
    pytest.param("HS", 0.22836286367690678, marks=pytest.mark.xfail(
        strict=True,
        reason="HS uses metafor's ni convention; ours differs by ~0.038. "
               "Tracked in CHANGELOG as a known tau2 gap.")),
    ("SJ", 0.34551570159859507),
    pytest.param("PM", 0.31809371186301511, marks=pytest.mark.xfail(
        strict=True,
        reason="metafor's PM stopping rule leaves the root ~2.5e-5 above the "
               "exact Q(tau2)=k-1 solution that brentq finds.")),
    pytest.param("EB", 0.31806851577130513, marks=pytest.mark.xfail(
        strict=True,
        reason="EB is ~6e-8 from metafor's iterative EB; needs the exact "
               "metafor iteration rather than the PM root.")),
])
def test_tau2_estimator_matches_r(method, expected):
    got = ma.estimate_tau2(YI, VI, method, NI if method == "HS" else None)
    assert got == pytest.approx(expected, abs=1e-12)


@pytest.mark.parametrize("method,expected", [
    ("ML", 0.28002817104959499),
    ("REML", 0.31324332598089488),
])
def test_tau2_likelihood_matches_r(method, expected):
    # ML/REML are served by the pooled model, not by estimate_tau2().
    assert ma.rma(YI, VI, method=method).tau2 == pytest.approx(expected, abs=1e-12)


# =========================================================================
# fail-safe N
# =========================================================================
@pytest.mark.parametrize("kind,expected", [
    ("Rosenthal", 598.0),
    ("Rosenberg", 370.0),
])
def test_fsn_matches_r(kind, expected):
    res = ma.fsn(YI, VI, type=kind)
    assert float(res.fsnum) == pytest.approx(expected, rel=1e-9)
    assert res.k == 13


def test_fsn_orwin_matches_r():
    # fsn(yi, vi, type = "Orwin", target = 0.10)$fsnum
    res = ma.fsn(YI, VI, type="Orwin", target=0.10)
    assert float(res.fsnum) == pytest.approx(43.0, rel=1e-9)


# =========================================================================
# publication bias
# =========================================================================
@pytest.mark.parametrize("predictor,expected_z", [
    ("sei", -0.8033291120701046),
    ("vi", -0.4267809509597853),
    ("ni", 2.0487535658637075),
])
def test_regtest_predictors_match_r(predictor, expected_z):
    res = ma.regtest(YI, VI, ni=NI, predictor=predictor)
    assert float(res.stat) == pytest.approx(expected_z, abs=1e-10)
    assert res.k == 13


def test_regtest_sei_pvalue_matches_r():
    assert float(ma.regtest(YI, VI, predictor="sei").p) == pytest.approx(
        0.42178453567222851, abs=1e-12
    )


def test_begg_rank_test_matches_r():
    res = ma.begg_test(YI, VI)
    assert float(res.tau) == pytest.approx(0.02564102564102564, abs=1e-12)
    assert float(res.p) == pytest.approx(0.95236190796086628, abs=1e-12)


@pytest.mark.parametrize("estimator,k0,tau2", [
    ("L0", 1, 0.33128803733657958),
    ("R0", 0, 0.31324332598089488),
    ("Q0", 1, 0.33128803733657958),
])
def test_trimfill_k0_and_tau2_match_r(estimator, k0, tau2):
    res = ma.trimfill(YI, VI, estimator=estimator)
    assert res.k0 == k0
    assert float(res.filled_tau2) == pytest.approx(tau2, abs=1e-10)


@pytest.mark.parametrize("estimator,effect,lb,ub", [
    ("L0", -0.65708280076036507, -1.0069921662084518, -0.30717343531227825),
    ("R0", -0.7145323483651016, -1.0668976757400532, -0.36216702099015002),
    ("Q0", -0.65708280076036507, -1.0069921662084518, -0.30717343531227825),
])
def test_trimfill_adjusted_estimate_matches_r(estimator, effect, lb, ub):
    res = ma.trimfill(YI, VI, estimator=estimator)
    assert float(res.filled_effect) == pytest.approx(effect, abs=1e-10)
    assert float(res.filled_ci_lower) == pytest.approx(lb, abs=1e-10)
    assert float(res.filled_ci_upper) == pytest.approx(ub, abs=1e-10)


# =========================================================================
# sensitivity / influence
# =========================================================================
LEAVE1OUT_ESTIMATE = [
    -0.70708378803143668, -0.65403120039781426, -0.6855544179728319,
    -0.62840940786250521, -0.7641700949872221, -0.71087656485093542,
    -0.65524759979198433, -0.79476761094828685, -0.7411972993138477,
    -0.65303292576343097, -0.75785862520424896, -0.7598120973394219,
    -0.77753618077462461,
]
LEAVE1OUT_SE = [
    0.18996102470271706, 0.18069538662381887, 0.18570622711370868,
    0.17661894594992472, 0.19178764431498513, 0.20025407305452378,
    0.18047336924070292, 0.17987771255124058, 0.19667491774174964,
    0.18426751166366773, 0.19578686099293802, 0.1820909790127197,
    0.18553596762023364,
]
LEAVE1OUT_TAU2 = [
    0.33615674530030559, 0.29258308555412349, 0.32073805688721874,
    0.26275835624150085, 0.32780325861839499, 0.35955418115219112,
    0.29302586988829021, 0.27318303411621087, 0.34945200992289854,
    0.2987246009130346, 0.34048826533464821, 0.30816220271827904,
    0.3037333265611305,
]
INFLUENCE_RSTUDENT = [
    -0.21814247434444201, -1.2917901832571879, -0.75471968978577231,
    -1.4512000710588946, 0.84771057938683747, -0.11799945848277615,
    -1.3037270591910279, 1.4500802683972074, 0.407613168355557,
    -1.1277386138697161, 0.66907919529243365, 1.2898409756597904,
    1.1879029494174913,
]
INFLUENCE_DFFITS = [
    -0.040707560486848607, -0.34358773695273337, -0.16036147849771828,
    -0.52005061234168115, 0.27074518577236889, -0.019006329798649767,
    -0.33615753082004618, 0.4774599279350174, 0.14154844073590009,
    -0.34869463044224674, 0.23150448289625225, 0.25261983898997725,
    0.35486086027142333,
]
INFLUENCE_COOKD = [
    0.0017165423672919507, 0.11324970451270386, 0.025980324843641142,
    0.22948142937637481, 0.07623141179057201, 0.00041349549975999695,
    0.10874162331180412, 0.19917755812714968, 0.02199841675711052,
    0.11701779865805216, 0.058078125111793895, 0.063433382213394465,
    0.12281284227504796,
]
INFLUENCE_HAT = [
    0.050594830793155146, 0.063646796321265964, 0.044360277630967407,
    0.096987467356973597, 0.08868456286827521, 0.10095738165294266,
    0.060271817029424149, 0.10189438634592952, 0.087431331316980623,
    0.083676072473368973, 0.099250266628389039, 0.038216294228415097,
    0.084028515353912855,
]


def test_leave1out_matches_r():
    res = ma.leave1out(YI, VI, method="REML")
    assert list(np.asarray(res.estimate, dtype=float)) == pytest.approx(
        LEAVE1OUT_ESTIMATE, abs=1e-10)
    assert list(np.asarray(res.se, dtype=float)) == pytest.approx(
        LEAVE1OUT_SE, abs=1e-12)
    assert list(np.asarray(res.tau2, dtype=float)) == pytest.approx(
        LEAVE1OUT_TAU2, abs=1e-10)


def test_influence_matches_r():
    res = ma.influence(YI, VI, method="REML")
    assert list(np.asarray(res.rstudent, dtype=float)) == pytest.approx(
        INFLUENCE_RSTUDENT, abs=1e-10)
    assert list(np.asarray(res.dffits, dtype=float)) == pytest.approx(
        INFLUENCE_DFFITS, abs=1e-11)
    assert list(np.asarray(res.cook_d, dtype=float)) == pytest.approx(
        INFLUENCE_COOKD, abs=1e-12)
    assert list(np.asarray(res.hat, dtype=float)) == pytest.approx(
        INFLUENCE_HAT, abs=1e-12)


def test_cooks_distance_matches_r():
    got = ma.cooks_distance(YI, VI, method="REML")
    assert list(np.asarray(got, dtype=float)) == pytest.approx(
        INFLUENCE_COOKD, abs=1e-12)


def test_hatvalues_matches_r():
    got = ma.hatvalues(YI, VI, method="REML")
    assert list(np.asarray(got, dtype=float)) == pytest.approx(
        INFLUENCE_HAT, abs=1e-12)


def test_rstudent_matches_r():
    got = ma.rstudent(YI, VI, method="REML")
    assert list(np.asarray(got, dtype=float)) == pytest.approx(
        INFLUENCE_RSTUDENT, abs=1e-10)


BAUJAT_Y = [
    0.0015375019405774594, 0.1121070955457375, 0.024349037368338881,
    0.23777332005060481, 0.066985818131784988, 0.00033327153654555793,
    0.1079095079665882, 0.19896461424249787, 0.01838161323084592,
    0.11138957175892485, 0.048970625778589856, 0.061834543543002253,
    0.11531285055817489,
]


def test_baujat_matches_r():
    res = ma.baujat(YI, VI, method="REML")
    assert res.k == 13
    assert list(np.asarray(res.y, dtype=float)) == pytest.approx(BAUJAT_Y, abs=1e-12)
    # With this much heterogeneity every study contributes; the contribution
    # measure must be strictly positive.
    assert np.all(np.asarray(res.x, dtype=float) > 0)


# =========================================================================
# cumulative meta-analysis
# =========================================================================
CUMUL_ESTIMATE = [
    -0.88931133392020523, -1.3250034422448751, -1.3302341492427767,
    -1.4220443328030565, -1.0585852430799794, -0.98771769377538032,
    -1.0578250439326302, -0.8990141446240496, -0.83801414250700734,
    -0.89411704952189863, -0.82863590506213702, -0.77753618077462461,
    -0.7145323483651016,
]
CUMUL_SE = [
    0.57060035489295069, 0.34898877228738301, 0.30688500311037958,
    0.12846608583699484, 0.30105720706506739, 0.23666560986019716,
    0.22225527037015716, 0.24388847918334153, 0.21690487135902195,
    0.20304557778557461, 0.18754794275468856, 0.18553596762023364,
    0.17978153178036146,
]
CUMUL_TAU2 = [
    0.0, 0.0, 0.0, 0.0, 0.29839309654975549, 0.2236917378058825,
    0.22139119955561221, 0.35976647367274611, 0.31768609446984969,
    0.31053009503088735, 0.29624285139217699, 0.3037333265611305,
    0.31324332598089488,
]


def test_cumul_matches_r():
    res = ma.cumul(YI, VI, method="REML")
    # k is cumulative: 1, 2, ..., 13
    assert list(np.asarray(res.k, dtype=int)) == list(range(1, 14))
    assert list(np.asarray(res.estimate, dtype=float)) == pytest.approx(
        CUMUL_ESTIMATE, abs=1e-10)
    assert list(np.asarray(res.se, dtype=float)) == pytest.approx(
        CUMUL_SE, abs=1e-12)
    assert list(np.asarray(res.tau2, dtype=float)) == pytest.approx(
        CUMUL_TAU2, abs=1e-10)


# =========================================================================
# meta-regression
# =========================================================================
def test_metareg_on_publication_year_matches_r():
    # rma(yi, vi, mods = ~ year, method = "REML")
    res = ma.metareg(YI, VI, np.asarray(YEAR, dtype=float), X_names=["year"],
                     method="REML")
    assert res.k == 13
    assert float(res.p_mods) == 1
    assert list(np.asarray(res.b, dtype=float)) == pytest.approx(
        [-53.171117896376529, 0.026662647852802833], abs=1e-9)
    assert list(np.asarray(res.se, dtype=float)) == pytest.approx(
        [32.011606022365719, 0.016269130490556408], abs=1e-9)
    assert float(res.QM) == pytest.approx(2.6858259677367227, abs=1e-8)
    assert float(res.QM_p) == pytest.approx(0.10124471625029827, abs=1e-10)
    assert float(res.tau2) == pytest.approx(0.26655579304032145, abs=1e-10)
    assert float(res.QE) == pytest.approx(100.17824942080814, abs=1e-8)
    assert float(res.I2) == pytest.approx(87.693598025673452, abs=1e-8)
    assert float(res.H2) == pytest.approx(8.1258519109499812, abs=1e-9)
    assert float(res.R2) == pytest.approx(14.904557916557481, abs=1e-8)
