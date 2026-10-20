"""Regression tests for the modules that were previously under-covered.

Expected values come from R 4.6.1 (metafor 5.2-1) on the 13 BCG vaccine trials —
the classic ``metafor::dat.bcg`` dataset. The generator scripts that produced
them are ``_refgen_gaps.R`` / ``_refgen_gaps2.R`` in the project scratch area;
every number below is a verbatim ``sprintf("%.17g", ...)`` of the R output.

Covered here: tau2 estimators, fsn, regtest/ranktest/trimfill,
leave1out/influence/cooks/hatvalues/rstudent/baujat, cumul, metareg.
"""
from __future__ import annotations

import warnings

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
    ("HS", 0.22836286367690678),
    ("SJ", 0.34551570159859507),
    ("PM", 0.31809371186301511),
    ("EB", 0.31806851577130513),
])
def test_tau2_estimator_matches_r(method, expected):
    got = ma.estimate_tau2(YI, VI, method, NI if method == "HS" else None)
    assert got == pytest.approx(expected, abs=1e-14)


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


# =========================================================================
# netcontrib(method="randomwalk") — Davies' algorithm
# (netmeta::contribution.matrix.davies)
# =========================================================================
NMA_TE = [0.10, -0.20, 0.05, -0.15, 0.30, 0.25]
NMA_SE = [0.10, 0.12, 0.11, 0.13, 0.14, 0.15]
NMA_T1 = ["Placebo", "Placebo", "DrugA", "DrugA", "DrugB", "DrugC"]
NMA_T2 = ["DrugA", "DrugB", "DrugB", "DrugC", "DrugC", "Placebo"]

# netcontrib(nm, method = "randomwalk")$random  (R 4.6.1 / netmeta 3.7-0)
R_RANDOMWALK = [
    [0.517030283, 0.111817479, 0.129210530, 0.112731178, 0.128296830, 0.000913699],
    [0.124797401, 0.495253669, 0.126407214, 0.127134502, 0.002337102, 0.124070113],
    [0.123082146, 0.107941540, 0.536832246, 0.001120382, 0.124202528, 0.106821158],
    [0.133717741, 0.135112150, 0.001394409, 0.482982162, 0.124093973, 0.122699564],
    [0.135432995, 0.002210499, 0.137643494, 0.110402206, 0.506119099, 0.108191707],
    [0.001143503, 0.139125103, 0.140268606, 0.129411874, 0.128268371, 0.461782542],
]


def _nma():
    return ma.netmeta(
        TE=NMA_TE, seTE=NMA_SE, treat1=NMA_T1, treat2=NMA_T2,
        studlab=[f"s{i}" for i in range(1, 7)], sm="MD",
        reference_group="Placebo",
    )


def test_netcontrib_randomwalk_matches_r():
    nc = ma.netcontrib(_nma(), method="randomwalk")
    assert nc.method == "randomwalk"
    assert nc.directs == [
        "DrugA:DrugB", "DrugA:DrugC", "DrugA:Placebo",
        "DrugB:DrugC", "DrugB:Placebo", "DrugC:Placebo",
    ]
    got = np.asarray(nc.random, dtype=float)
    assert got.shape == (6, 6)
    assert got == pytest.approx(np.asarray(R_RANDOMWALK), abs=1e-8)
    # Every comparison's contributions form a full decomposition.
    assert np.allclose(got.sum(axis=1), 1.0)


def test_netcontrib_randomwalk_agrees_with_shortestpath_on_complete_network():
    nm = _nma()
    rw = np.asarray(ma.netcontrib(nm, method="randomwalk").random, dtype=float)
    sp = np.asarray(ma.netcontrib(nm, method="shortestpath").random, dtype=float)
    assert rw == pytest.approx(sp, abs=1e-8)


# An incomplete network: the DrugC:Placebo comparison has no direct evidence.
INC_TE = [0.10, -0.20, 0.05, -0.15, 0.30]
INC_SE = [0.10, 0.12, 0.11, 0.13, 0.14]
INC_T1 = ["Placebo", "Placebo", "DrugA", "DrugA", "DrugB"]
INC_T2 = ["DrugA", "DrugB", "DrugB", "DrugC", "DrugC"]

# netcontrib(nm, method = "randomwalk")$common / $random  (R 4.6.1 / netmeta 3.7-0)
R_RW_INC_COMMON = [
    [0.54722302441182, 0.0907040903477, 0.1356843974464, 0.0907040903477,
     0.1356843974464],
    [0.12668587825425, 0.6209807740754, 0.0418824898054, 0.1685683680596,
     0.0418824898054],
    [0.11213586565816, 0.0247825383464, 0.7013806536446, 0.0247825383464,
     0.1369184040045],
    [0.14692563395167, 0.1954994091106, 0.0485737751589, 0.5604274066200,
     0.0485737751589],
    [0.16147564654775, 0.0356868552188, 0.1971625017665, 0.0356868552188,
     0.5699881412482],
    [0.00970000839739, 0.2733165795182, 0.2830165879155, 0.2218334162831,
     0.2121334078858],
]
R_RW_INC_RANDOM = [
    [0.51581015179598, 0.1134287149704, 0.1286662091316, 0.1134287149704,
     0.1286662091316],
    [0.12554486949029, 0.6236441538908, 0.0417553690429, 0.1673002385332,
     0.0417553690429],
    [0.12265330396571, 0.0359626026676, 0.6468055840658, 0.0359626026676,
     0.1586159066333],
    [0.13236020640770, 0.1763823101192, 0.0440221037115, 0.6032132760501,
     0.0440221037115],
    [0.13525177193228, 0.0396565406460, 0.1749083125783, 0.0396565406460,
     0.6105268341973],
    [0.00192771034972, 0.2578781729440, 0.2598058832937, 0.2411579718811,
     0.2392302615314],
]


def _nma_incomplete():
    return ma.netmeta(
        TE=INC_TE, seTE=INC_SE, treat1=INC_T1, treat2=INC_T2,
        studlab=[f"s{i}" for i in range(1, 6)], sm="MD",
        reference_group="Placebo",
    )


def test_netcontrib_randomwalk_incomplete_network_matches_r():
    nc = ma.netcontrib(_nma_incomplete(), method="randomwalk")
    assert nc.directs == [
        "DrugA:DrugB", "DrugA:DrugC", "DrugA:Placebo",
        "DrugB:DrugC", "DrugB:Placebo",
    ]
    assert np.asarray(nc.common, dtype=float) == pytest.approx(
        np.asarray(R_RW_INC_COMMON), abs=1e-10
    )
    got = np.asarray(nc.random, dtype=float)
    assert got.shape == (6, 5)
    assert got == pytest.approx(np.asarray(R_RW_INC_RANDOM), abs=1e-10)
    assert np.allclose(got.sum(axis=1), 1.0)


# =========================================================================
# netmeta(method_tau = "ML" / "REML") — metafor::rma.mv on the star design
# =========================================================================
NMA1_TE = [0.10, -0.20, 0.05, -0.15, 0.30, 0.25]
NMA1_SE = [0.10, 0.12, 0.11, 0.13, 0.14, 0.15]
NMA1_T1 = ["Placebo", "Placebo", "DrugA", "DrugA", "DrugB", "DrugC"]
NMA1_T2 = ["DrugA", "DrugB", "DrugB", "DrugC", "DrugC", "Placebo"]
NMA1_ST = [f"s{i}" for i in range(1, 7)]
# netmeta(..., reference.group = "Placebo")$tau (R 4.6.1 / netmeta 3.7-0)
NMA1_TAU = {
    "DL": 0.17102725023638912,
    "ML": 0.082442793611206794,
    "REML": 0.17024539107325981,
}

NMA4_TE = [0.10, -0.30, -0.40, 0.55, -0.25, 0.20, 0.40, 0.20, 0.50, -0.10]
NMA4_SE = [0.05, 0.06, 0.07, 0.10, 0.12, 0.09, 0.11, 0.13, 0.15, 0.14]
NMA4_T1 = ["A", "A", "B", "A", "A", "D", "D", "E", "B", "B"]
NMA4_T2 = ["B", "C", "C", "D", "D", "E", "C", "C", "E", "E"]
NMA4_ST = ["k1", "k1", "k1", "k2", "k3", "k4", "k4", "k4", "k5", "k6"]
# $tau with reference.group = "" (per-study reference) and = "A"
NMA4_TAU = {
    "": {"DL": 0.46030534745646717, "ML": 0.30909902431853664,
         "REML": 0.45591212187397667},
    "A": {"DL": 0.46030534745646717, "ML": 0.30665168954488048,
          "REML": 0.45454522862451824},
}

#: DL is a closed-form expression, so it reproduces R to machine precision.
#: ML/REML go through a numerical optimiser and agree to ~1e-8 relative.
NMA_TAU_TOL = {"DL": 1e-12, "ML": 1e-7, "REML": 1e-7}


def _nma4(ref):
    return ma.netmeta(TE=NMA4_TE, seTE=NMA4_SE, treat1=NMA4_T1, treat2=NMA4_T2,
                      studlab=NMA4_ST, sm="MD", reference_group=ref)


@pytest.mark.parametrize("method", ["DL", "ML", "REML"])
def test_netmeta_method_tau_matches_r(method):
    m = ma.netmeta(TE=NMA1_TE, seTE=NMA1_SE, treat1=NMA1_T1, treat2=NMA1_T2,
                   studlab=NMA1_ST, sm="MD", reference_group="Placebo",
                   method_tau=method)
    assert m.method_tau == method
    assert m.tau == pytest.approx(NMA1_TAU[method], abs=NMA_TAU_TOL[method])
    # tau^2 is what the random-effects pooling actually consumes
    assert m.tau ** 2 == pytest.approx(NMA1_TAU[method] ** 2,
                                       abs=2 * NMA1_TAU[method] * NMA_TAU_TOL[method])


@pytest.mark.parametrize("ref", ["", "A"])
@pytest.mark.parametrize("method", ["DL", "ML", "REML"])
def test_netmeta_method_tau_multistudy_three_arm(ref, method):
    m = ma.netmeta(TE=NMA4_TE, seTE=NMA4_SE, treat1=NMA4_T1, treat2=NMA4_T2,
                   studlab=NMA4_ST, sm="MD", reference_group=ref,
                   method_tau=method)
    want = NMA4_TAU[ref][method]
    assert m.tau == pytest.approx(want, abs=NMA_TAU_TOL[method])


def test_netmeta_dl_tau_is_invariant_to_reference_group():
    """Jackson's method of moments never depends on the per-study reference."""
    a = _nma4("") .tau
    b = _nma4("A").tau
    assert a == pytest.approx(b, abs=1e-12)


def test_netmeta_ml_reml_tau_can_differ_by_reference_group():
    """ML/REML fit the star-shaped design, which *does* depend on it."""
    ml_default = ma.netmeta(TE=NMA4_TE, seTE=NMA4_SE, treat1=NMA4_T1,
                            treat2=NMA4_T2, studlab=NMA4_ST, sm="MD",
                            reference_group="", method_tau="ML").tau
    ml_ref_a = ma.netmeta(TE=NMA4_TE, seTE=NMA4_SE, treat1=NMA4_T1,
                          treat2=NMA4_T2, studlab=NMA4_ST, sm="MD",
                          reference_group="A", method_tau="ML").tau
    assert abs(ml_default - ml_ref_a) > 1e-4


def test_netmeta_rejects_unknown_tau_estimator():
    with pytest.raises(ValueError, match="method_tau"):
        ma.netmeta(TE=NMA1_TE, seTE=NMA1_SE, treat1=NMA1_T1, treat2=NMA1_T2,
                   studlab=NMA1_ST, method_tau="PM")


def test_netmeta_ml_with_events_uses_the_blocked_calc_v():
    """With n/events the reference arm supplies the within-study covariance."""
    te = [-0.31015492830383962, 0.2607526605731495, -0.064538521137571178,
          0.16472036131694595, -0.34858230543772495, 0.2336148511815051]
    se = [0.31109532788247246, 0.31081886686117899, 0.36438685181792407,
          0.27155060999096053, 0.31739150709904934, 0.41345251535834238]
    t1 = ["P", "P", "A", "A", "B", "C"]
    t2 = ["A", "B", "B", "C", "C", "P"]
    st = [f"t{i}" for i in range(1, 7)]
    kwargs = dict(TE=te, seTE=se, treat1=t1, treat2=t2, studlab=st, sm="OR",
                  reference_group="P", n1=[120, 100, 90, 150, 130, 110],
                  n2=[118, 105, 95, 140, 125, 108],
                  event1=[24, 31, 18, 40, 22, 15],
                  event2=[30, 27, 20, 33, 28, 12])
    # R: tau = 2.4672384063227577e-06 (REML) — i.e. the boundary tau^2 = 0.
    m = ma.netmeta(**kwargs, method_tau="REML")
    assert 0.0 <= m.tau < 1e-5
    assert m.tau == pytest.approx(0.0, abs=1e-5)


# =========================================================================
# netmetabin(method = "NCH") — conditional non-central hypergeometric fit
# =========================================================================
NCH_E1 = [9, 12, 7, 15, 8, 14]
NCH_N1 = [60, 65, 55, 70, 58, 62]
NCH_E2 = [5, 6, 4, 9, 5, 7]
NCH_N2 = [58, 62, 50, 68, 60, 59]
NCH_T1 = ["A", "A", "B", "A", "B", "C"]
NCH_T2 = ["B", "C", "C", "D", "D", "D"]
NCH_ST = [f"s{i}" for i in range(1, 7)]

# netmetabin(..., method = "NCH")$TE.common (R 4.6.1 / netmeta 3.7-0)
NCH_R_TE = [
    [0.0, 0.2971123, 0.4593067, 0.7828089],
    [-0.2971123, 0.0, 0.1621943, 0.4856965],
    [-0.4593067, -0.1621943, 0.0, 0.3235022],
    [-0.7828089, -0.4856965, -0.3235022, 0.0],
]
NCH_R_SE = [
    [0.0, 0.3784877, 0.3474910, 0.3325432],
    [0.3784877, 0.0, 0.3916154, 0.3806076],
    [0.3474910, 0.3916154, 0.0, 0.3365944],
    [0.3325432, 0.3806076, 0.3365944, 0.0],
]


def _nch():
    return ma.netmetabin(NCH_E1, NCH_N1, NCH_E2, NCH_N2, NCH_T1, NCH_T2,
                         studlab=NCH_ST, method="NCH")


def test_netmetabin_nch_matches_r():
    res = _nch()
    assert res.method == "NCH"
    assert res.trts == ["A", "B", "C", "D"]
    # R's optim() stops at reltol 1e-8 on a very flat conditional likelihood,
    # so it lands ~1e-5 away from the maximum; see the next test.
    assert np.asarray(res.TE_common).ravel() == pytest.approx(
        np.asarray(NCH_R_TE).ravel(), abs=1e-4)
    assert np.asarray(res.seTE_common).ravel() == pytest.approx(
        np.asarray(NCH_R_SE).ravel(), abs=1e-6)
    assert np.all(np.isfinite(np.asarray(res.TE_common)))


def test_netmetabin_nch_reaches_a_higher_likelihood_than_r():
    """R's L-BFGS-B stops short; our Newton polish must not do worse."""
    import math

    trts = ["A", "B", "C", "D"]
    tindex = {t: i for i, t in enumerate(trts)}
    arms = {}
    for s, t, e, nn in zip(NCH_ST, NCH_T1, NCH_E1, NCH_N1):
        arms.setdefault(s, {})[t] = (float(e), float(nn))
    for s, t, e, nn in zip(NCH_ST, NCH_T2, NCH_E2, NCH_N2):
        arms.setdefault(s, {})[t] = (float(e), float(nn))
    studies = []
    for s in sorted(arms):
        ordered = sorted(arms[s], key=lambda t: tindex[t])
        studies.append([(tindex[t], arms[s][t][0], arms[s][t][1]) for t in ordered])

    def loglik(x):
        total_ll = 0.0
        for arm_list in studies:
            t0, r0, n0 = arm_list[0]
            cond = 1.0 if t0 != 0 else 0.0
            n_events = r0
            s_tot = n0
            for tk, rk, nk in arm_list[1:]:
                eta = x[tk] - cond * x[t0]
                total_ll += rk * eta
                n_events += rk
                s_tot += nk * math.exp(eta)
            total_ll -= n_events * math.log(s_tot)
        return total_ll

    # TE.common[1, k] == theta_1 - theta_k == -x_k (x_1 is left at 0).
    x_r = np.array([0.0] + [-NCH_R_TE[0][k] for k in range(1, 4)])
    res = _nch()
    x_o = np.array([0.0] + [-res.TE_common[0][k] for k in range(1, 4)])
    assert loglik(x_o) >= loglik(x_r)


def test_netmetabin_nch_has_no_mh_inconsistency_statistics():
    res = _nch()
    assert np.isnan(res.Q_inconsistency)
    # Direct per-comparison estimates are still the Mantel-Haenszel ones.
    mh = ma.netmetabin(NCH_E1, NCH_N1, NCH_E2, NCH_N2, NCH_T1, NCH_T2,
                       studlab=NCH_ST, method="MH")
    assert np.asarray(res.TE_direct_common) == pytest.approx(
        np.asarray(mh.TE_direct_common), abs=1e-12, nan_ok=True)


def test_netmetabin_nch_and_mh_differ():
    nch = np.asarray(_nch().TE_common)
    mh = np.asarray(ma.netmetabin(NCH_E1, NCH_N1, NCH_E2, NCH_N2, NCH_T1,
                                  NCH_T2, studlab=NCH_ST,
                                  method="MH").TE_common)
    assert np.max(np.abs(nch - mh)) > 1e-3


# =========================================================================
# netmetareg(consistency = False) — the inconsistency model
# =========================================================================
IA_TE = [0.30, 0.42, 0.10, 0.25, 0.55, 0.40, -0.20, 0.05, 0.60, 0.35]
IA_SE = [0.20, 0.18, 0.22, 0.19, 0.15, 0.17, 0.21, 0.20, 0.16, 0.18]
IA_T1 = ["A", "A", "B", "B", "A", "C", "B", "C", "B", "C"]
IA_T2 = ["B", "B", "A", "A", "C", "A", "C", "B", "D", "D"]
IA_ST = [f"S{i}" for i in range(1, 11)]
IA_YR = [1, 4, 7, 10, 3, 9, 5, 11, 8, 12]


def _ia_net():
    return ma.netmeta(IA_TE, IA_SE, IA_T1, IA_T2, studlab=IA_ST, sm="MD",
                      reference_group="A", common=False, random=True,
                      method_tau="DL")


def _ia(**kw):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return ma.netmetareg(_ia_net(), IA_YR, covar_name="yr", **kw)


def test_netmetareg_inconsistency_independent_matches_r():
    """netmetareg(net, covar = yr, consistency = FALSE) with REML."""
    res = _ia(consistency=False, assumption="independent", method_tau="REML")
    assert res.names == ["B", "C", "D", ".comp_A_vs_B:Ival:yr",
                         ".comp_A_vs_C:Ival:yr", ".comp_B_vs_C:Ival:yr"]
    assert (res.k, res.p, res.QEdf) == (10, 6, 4)
    assert res.beta == pytest.approx(
        [-0.61275158672643248, -0.84588613071909324, -1.2051925708104192,
         0.087025104938048284, 0.13360094901262604, 0.035415076612411452],
        abs=1e-6)
    assert res.se == pytest.approx(
        [0.16994858726021705, 0.19501964638165148, 0.20235205292662722,
         0.027621500193503148, 0.032863789165912365, 0.026387755105463186],
        abs=1e-6)
    assert res.tau2 == pytest.approx(0.0038154727598880079, abs=1e-7)
    assert res.QE == pytest.approx(4.5695348829235733, abs=1e-6)
    assert res.QM == pytest.approx(37.437438357545993, abs=1e-4)


def test_netmetareg_inconsistency_common_matches_r():
    res = _ia(consistency=False, assumption="common", method_tau="REML")
    assert res.names == ["B", "C", "D", "Ival:yr"]
    assert (res.k, res.p, res.QEdf) == (10, 4, 6)
    assert res.beta == pytest.approx(
        [-0.32037610699638241, -0.46530334097080261, -1.3557592873880298,
         0.048988048870297771], abs=1e-6)
    assert res.tau2 == pytest.approx(0.049786800389355838, abs=1e-7)
    assert res.QM == pytest.approx(11.199479648310387, abs=1e-4)


def test_netmetareg_inconsistency_max_ia_keeps_single_study_comparisons():
    """max.ia = TRUE keeps B:D and C:D, which only one study contributes."""
    res = _ia(consistency=False, assumption="independent", method_tau="ML",
              max_ia=True)
    assert res.p == 7
    assert res.names[-1] == ".comp_B_vs_D:Ival:yr"
    assert res.beta == pytest.approx(
        [-0.62149412202980281, -0.84478377402432581, -1.1947837740243257,
         0.088036639078333603, 0.13330762866054971, 0.034284613655444884,
         -0.0033387935006845976], abs=1e-6)


def test_netmetareg_consistency_true_names_unchanged():
    res = _ia(consistency=True, assumption="independent", method_tau="REML")
    assert res.names == ["B", "C", "D", "B:yr", "C:yr", "D:yr"]
    assert res.tau2 == pytest.approx(0.0073541661837948176, abs=1e-7)
    assert res.beta == pytest.approx(
        [-0.62309697500151895, -0.81251913911986284, -2.1596399823918682,
         0.091247053084778942, 0.12522152540257267, 0.20831492900857329],
        abs=1e-6)


def test_netmetareg_inconsistency_default_directions_warn():
    """Without direction1/direction2 netmeta falls back to treatment order."""
    with pytest.warns(UserWarning, match="treatment order"):
        ma.netmetareg(_ia_net(), IA_YR, consistency=False,
                      assumption="independent", method_tau="REML",
                      covar_name="yr")


def test_netmetareg_direction_columns_reproduce_the_default():
    """direction1/direction2 = treatment order must equal the mkIval fallback."""
    default = _ia(consistency=False, assumption="independent")
    # mkIval(): 0 for the alphabetically first arm of each study, 1 otherwise.
    d1 = [0.0 if a < b else 1.0 for a, b in zip(IA_T1, IA_T2)]
    d2 = [0.0 if b < a else 1.0 for a, b in zip(IA_T1, IA_T2)]
    explicit = _ia(consistency=False, assumption="independent",
                   direction1=d1, direction2=d2)
    assert explicit.beta == pytest.approx(default.beta, abs=1e-10)


def test_netmetareg_inconsistency_rejects_out_of_range_directions():
    with pytest.raises(ValueError, match="-1, 0, or 1"):
        ma.netmetareg(_ia_net(), IA_YR, consistency=False,
                      direction1=[0] * 10, direction2=[2] * 10)


# =========================================================================
# confint() and predict(newmods=) on a model WITH moderators
# =========================================================================
CI_YI = [0.10, 0.35, -0.20, 0.55, 0.05, 0.40, -0.10, 0.25]
CI_VI = [0.04, 0.06, 0.05, 0.09, 0.03, 0.07, 0.08, 0.05]
CI_MODS = [1, 3, 5, 7, 2, 4, 6, 8]


def _ci_fit():
    return ma.rma(CI_YI, CI_VI, mods=CI_MODS, method="REML")


def test_rma_with_moderators_matches_r():
    fit = _ci_fit()
    assert list(fit.b) == pytest.approx(
        [0.071455317166275401, 0.017528984831107555], abs=1e-12)
    assert list(fit.se) == pytest.approx(
        [0.16042397648149287, 0.034399302919051515], abs=1e-12)
    assert fit.tau2 == pytest.approx(0.0010805786744490066, abs=1e-12)
    assert fit.QE == pytest.approx(6.8863753331529729, abs=1e-10)
    assert fit.I2 == pytest.approx(1.9338902721202715, abs=1e-10)
    assert fit.H2 == pytest.approx(1.019720271126147, abs=1e-12)


def test_confint_qp_with_moderators_matches_r():
    ci = _ci_fit().confint(type="QP")
    assert ci.tau2_lb == pytest.approx(0.0, abs=1e-12)
    assert ci.tau2_ub == pytest.approx(0.29534380876337607, abs=1e-8)
    assert ci.i2_ub == pytest.approx(84.3504138, abs=1e-6)
    assert ci.h2_ub == pytest.approx(6.3899453, abs=1e-6)


def test_confint_pl_with_moderators_matches_r():
    ci = _ci_fit().confint(type="PL")
    assert ci.tau2_ub == pytest.approx(0.1999417, abs=1e-6)
    assert ci.i2_ub == pytest.approx(78.4894537, abs=1e-5)
    assert ci.h2_ub == pytest.approx(4.6488824, abs=1e-5)


def test_confint_ht_with_moderators_is_rejected():
    """metafor: "Method only applicable to models without moderators."."""
    with pytest.raises(ValueError, match="without moderators"):
        _ci_fit().confint(type="HT")


def test_predict_newmods_matches_r():
    pr = _ci_fit().predict(newmods=[2, 6])
    assert isinstance(pr, list) and len(pr) == 2
    assert pr[0].pred == pytest.approx(0.10651328682849051, abs=1e-12)
    assert pr[0].se == pytest.approx(0.10714918265841189, abs=1e-12)
    assert pr[0].ci_lb == pytest.approx(-0.10349525215490049, abs=1e-12)
    assert pr[0].pi_lb == pytest.approx(-0.11315597576041736, abs=1e-12)
    assert pr[0].pi_ub == pytest.approx(0.32618254941739838, abs=1e-12)
    assert pr[1].pred == pytest.approx(0.17662922615292073, abs=1e-12)
    assert pr[1].ci_ub == pytest.approx(0.38562634027830955, abs=1e-12)
    assert pr[1].pi_ub == pytest.approx(0.39533174601301951, abs=1e-12)


def test_predict_single_newmods_returns_one_result():
    pr = _ci_fit().predict(newmods=[2])
    assert not isinstance(pr, list)
    assert pr.pred == pytest.approx(0.10651328682849051, abs=1e-12)


def test_predict_without_newmods_on_a_moderator_model_errors():
    with pytest.raises(ValueError, match="newmods"):
        _ci_fit().predict()


def test_predict_with_wrong_newmods_width_errors():
    with pytest.raises(ValueError, match="column"):
        _ci_fit().predict(newmods=[[1, 2], [3, 4]])


# =========================================================================
# permutest — sign-flip randomisation test, per-coefficient p-values, permci
# =========================================================================
PERM_YI = [-0.88931133, -1.58538866, -1.34807315, -1.44155119, -0.21754732,
           -0.78611559, -1.59664331, -0.32966938, -0.55783318, -0.49888521,
           -0.96960854, -0.48266968, -0.41683796]
PERM_VI = [0.13320328, 0.26373142, 0.21275645, 0.14693293, 0.14293235,
           0.12151837, 0.44437921, 0.15956733, 0.13432539, 0.33222552,
           0.20969889, 0.17053331, 0.13000306]
PERM_MODS = list(range(1, 14))


def test_permutest_intercept_only_matches_r():
    """R: permutest(rma(yi, vi), iter = 999)$pval = 0.001001."""
    fit = ma.rma(PERM_YI, PERM_VI, method="REML")
    pt = ma.permutest(fit, iter=999, seed=42)
    assert pt.nperm == 999
    assert pt.exact is False
    assert pt.pval[0] == pytest.approx(0.001001, abs=1e-6)
    assert pt.QMp_perm == pytest.approx(0.001001, abs=1e-6)


def test_permutest_exact_sign_flips_match_r():
    """Exact enumeration of the 2^k sign patterns: R also gives 0.031250."""
    fit = ma.rma(PERM_YI[:6], PERM_VI[:6], method="REML")
    pt = ma.permutest(fit, exact=True, seed=1)
    assert pt.exact is True
    assert pt.nperm == 2 ** 6
    assert pt.pval[0] == pytest.approx(0.031250, abs=1e-9)


def test_permutest_gives_one_p_value_per_coefficient():
    """R: 0.028028 / 0.069069; ours differs only by the RNG draw."""
    fit = ma.rma(PERM_YI, PERM_VI, mods=PERM_MODS, method="REML")
    pt = ma.permutest(fit, iter=999, seed=7)
    assert len(pt.pval) == 2
    assert pt.pval[0] != pytest.approx(pt.pval[1])
    assert pt.pval[0] == pytest.approx(0.028, abs=0.03)
    assert pt.pval[1] == pytest.approx(0.069, abs=0.03)


def test_permutest_uses_sign_flips_for_intercept_only():
    """Row permutations leave a constant design unchanged, so the old
    implementation returned p = 1 for every data set."""
    strong = ma.rma(PERM_YI, PERM_VI, method="REML")
    p_strong = ma.permutest(strong, iter=499, seed=3).pval[0]
    # a null data set: 13 effects centred on zero with the same variances
    null_yi = [-0.05, 0.10, -0.02, 0.07, -0.11, 0.03, 0.08, -0.06,
               0.01, -0.09, 0.04, 0.02, -0.03]
    null = ma.rma(null_yi, PERM_VI, method="REML")
    p_null = ma.permutest(null, iter=499, seed=3).pval[0]
    assert p_strong < 0.05
    assert p_null > 0.10
    assert p_strong < p_null


def test_permutest_permci_contains_the_estimate():
    fit = ma.rma(PERM_YI, PERM_VI, method="REML")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        pt = ma.permutest(fit, iter=199, permci=True, seed=11)
    assert pt.ci_lb is not None and pt.ci_ub is not None
    assert pt.ci_lb[0] < pt.beta[0] < pt.ci_ub[0]
    # A permutation CI is Monte-Carlo (only 199 permutations, so the reference
    # distribution itself is estimated); metafor's is noisier still because it
    # redraws at every root-search step.  Require agreement in magnitude with
    # the model CI rather than a strict ordering.
    assert abs(pt.ci_lb[0] - fit.ci_lb[0]) < 0.2
    assert abs(pt.ci_ub[0] - fit.ci_ub[0]) < 0.2
    assert (pt.ci_ub[0] - pt.ci_lb[0]) == pytest.approx(
        fit.ci_ub[0] - fit.ci_lb[0], rel=0.5)


def test_permutest_permci_on_a_selected_coefficient():
    fit = ma.rma(PERM_YI, PERM_VI, mods=PERM_MODS, method="REML")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        pt = ma.permutest(fit, iter=199, permci=[2], seed=11)
    assert np.isnan(pt.ci_lb[0]) and np.isnan(pt.ci_ub[0])
    assert pt.ci_lb[1] < fit.b[1] < pt.ci_ub[1]


def test_permutest_permci_warns_when_there_are_too_few_permutations():
    fit = ma.rma(PERM_YI[:6], PERM_VI[:6], method="REML")
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        pt = ma.permutest(fit, iter=10, permci=True, seed=1)
    assert any("too few" in str(w.message) for w in caught)
    assert pt.ci_lb is None


def test_permutest_rejects_bad_options():
    fit = ma.rma(PERM_YI, PERM_VI, method="REML")
    with pytest.raises(ValueError, match="alternative"):
        ma.permutest(fit, alternative="both")
    with pytest.raises(ValueError, match="stat"):
        ma.permutest(fit, stat="pval")
    with pytest.raises(ValueError, match="btt"):
        ma.permutest(fit, btt=[5])


# =========================================================================
# Firth / Jeffreys-penalised logistic regression + metabin / netmetabin "LRP"
# =========================================================================
LRP_E1 = [9, 12, 7, 15]
LRP_N1 = [60, 65, 55, 70]
LRP_E2 = [5, 6, 4, 9]
LRP_N2 = [58, 62, 50, 68]

LRP_E1H = [20, 6, 18, 4, 15, 7]
LRP_N1H = [50, 55, 48, 52, 45, 60]
LRP_E2H = [5, 22, 6, 20, 4, 18]
LRP_N2H = [52, 48, 55, 50, 58, 54]


def _lrp_design(e1, n1, e2, n2):
    """meta:::runLRP's design: treat arm labelled B, control arm labelled A."""
    from pimeta._firth import treatment_design

    k = len(e1)
    trt_idx, study_idx, succ, tot = [], [], [], []
    for i in range(k):
        trt_idx += [1, 0]
        study_idx += [i, i]
        succ += [float(e1[i]), float(e2[i])]
        tot += [float(n1[i]), float(n2[i])]
    return (treatment_design(trt_idx, study_idx, 2, k),
            np.array(succ), np.array(tot))


# ---------------------------------------------------------------- Firth ---
def test_firth_logit_matches_brglm2_when_brglm2_is_converged():
    """brglm2 with epsilon=1e-12; its *default* epsilon=1e-8 stops ~4e-7 short."""
    from pimeta._firth import firth_logit

    X, succ, tot = _lrp_design(LRP_E1, LRP_N1, LRP_E2, LRP_N2)
    beta, vcov, _mu = firth_logit(X, succ, tot)
    # same column order as R: (Intercept), treat, study2, study3, study4
    want = [-2.3149447523982016, 0.61132859910140214, 0.19663950165447863,
            -0.14157232005528059, 0.43813210454295293]
    assert list(beta) == pytest.approx(want, abs=1e-11)
    want_se = [0.32843779198771483, 0.27018088635989368, 0.37913434126216877,
               0.42246539947503026, 0.36058889982279713]
    got_se = [float(np.sqrt(vcov[j, j])) for j in range(len(want))]
    assert got_se == pytest.approx(want_se, abs=1e-11)


def test_firth_logit_score_is_zero_at_the_solution():
    from pimeta._firth import _expit, _fisher, firth_logit

    X, succ, tot = _lrp_design(LRP_E1, LRP_N1, LRP_E2, LRP_N2)
    beta, _vcov, _mu = firth_logit(X, succ, tot)
    mu = _expit(X @ beta)
    w = tot * mu * (1.0 - mu)
    info_inv = np.linalg.inv(_fisher(X, w))
    hhat = np.einsum("ij,jk,ik->i", X, info_inv, X)
    grad = X.T @ ((succ - tot * mu) + w * hhat * (0.5 - mu))
    assert np.max(np.abs(grad)) < 1e-10


def test_firth_logit_is_finite_with_zero_cells():
    """Complete separation would send the unpenalised MLE to +/-infinity."""
    from pimeta._firth import firth_logit

    X, succ, tot = _lrp_design([0, 12, 7], [60, 65, 55], [5, 0, 4], [58, 62, 50])
    beta, vcov, mu = firth_logit(X, succ, tot)
    assert np.all(np.isfinite(beta))
    assert np.all(np.isfinite(vcov))
    assert np.all((mu > 0) & (mu < 1))


def test_dispersion_phi_matches_meta():
    from pimeta._firth import dispersion_phi, firth_logit

    X, succ, tot = _lrp_design(LRP_E1H, LRP_N1H, LRP_E2H, LRP_N2H)
    _beta, _vcov, mu = firth_logit(X, succ, tot)
    assert dispersion_phi(succ, tot, mu, X.shape[1]) == pytest.approx(
        13.668012298174339, abs=1e-5)
    # no residual df -> not estimable, metafor/meta report 1
    assert dispersion_phi(succ[:2], tot[:2], mu[:2], 2) == 1.0


# --------------------------------------------------------------- metabin ---
def test_metabin_lrp_matches_r():
    m = ma.metabin(LRP_E1, LRP_N1, LRP_E2, LRP_N2, sm="OR", method="LRP")
    assert m.method == "Lrp"
    assert m.phi == pytest.approx(1.0)
    assert m.fixed.pooled_est == pytest.approx(0.61132822173082302, abs=1e-6)
    assert m.fixed.pooled_se == pytest.approx(0.27018088357202591, abs=1e-6)
    assert m.fixed.q == pytest.approx(0.08875379, abs=1e-6)
    assert m.random.pooled_est == pytest.approx(m.fixed.pooled_est, abs=1e-12)
    assert m.random.pooled_se == pytest.approx(m.fixed.pooled_se, abs=1e-12)


def test_metabin_lrp_overdispersion_matches_r():
    m = ma.metabin(LRP_E1H, LRP_N1H, LRP_E2H, LRP_N2H, sm="OR", method="LRP")
    assert m.fixed.pooled_est == pytest.approx(-0.074340412573228945, abs=1e-6)
    assert m.fixed.pooled_se == pytest.approx(0.18947928093484381, abs=1e-6)
    assert m.phi == pytest.approx(13.668012298174339, abs=1e-4)
    # "random" = common estimate with the SE multiplied by phi
    assert m.random.pooled_est == pytest.approx(m.fixed.pooled_est, abs=1e-12)
    assert m.random.pooled_se == pytest.approx(2.5898051420666759, abs=1e-4)
    # LRP has no tau^2, but Q / I^2 are still reported
    assert np.isnan(m.fixed.tau2)
    assert m.fixed.q == pytest.approx(63.22606, abs=1e-4)
    assert m.fixed.i2 == pytest.approx(92.09187, abs=1e-3)


def test_metabin_lrp_single_study():
    m = ma.metabin([9], [60], [5], [58], sm="OR", method="LRP")
    assert m.fixed.pooled_est == pytest.approx(0.58464355260034029, abs=1e-10)
    assert m.fixed.pooled_se == pytest.approx(0.57508736709903696, abs=1e-10)
    assert m.phi == 1.0


def test_metabin_lrp_zero_cells():
    m = ma.metabin([0, 12, 7], [60, 65, 55], [5, 0, 4], [58, 62, 50],
                   sm="OR", method="LRP")
    assert m.fixed.pooled_est == pytest.approx(0.71522938222902366, abs=1e-6)
    assert m.fixed.pooled_se == pytest.approx(0.40848284782939448, abs=1e-6)
    assert m.phi == pytest.approx(6.4835208278433907, abs=1e-4)


def test_metabin_lrp_rejects_non_or():
    with pytest.raises(ValueError, match="sm='OR'"):
        ma.metabin([9], [60], [5], [58], sm="RR", method="LRP")


# ------------------------------------------------------------ netmetabin ---
LRP_R_TE = [
    [0.0, 0.3468862, 0.5178376, 0.8980439],
    [-0.3468862, 0.0, 0.1709515, 0.5511578],
    [-0.5178376, -0.1709515, 0.0, 0.3802063],
    [-0.8980439, -0.5511578, -0.3802063, 0.0],
]
LRP_R_SE = [
    [0.0, 0.3961894, 0.3688837, 0.3544509],
    [0.3961894, 0.0, 0.4094774, 0.3985177],
    [0.3688837, 0.4094774, 0.0, 0.3606714],
    [0.3544509, 0.3985177, 0.3606714, 0.0],
]
LRP_R_DIRECT = [
    [np.nan, 0.5846436, 0.7084855, 0.5591415],
    [-0.5846436, np.nan, 0.4687141, 0.5297278],
    [-0.7084855, -0.4687141, np.nan, 0.7384950],
    [-0.5591415, -0.5297278, -0.7384950, np.nan],
]


def _nma_lrp():
    return ma.netmetabin(NCH_E1, NCH_N1, NCH_E2, NCH_N2, NCH_T1, NCH_T2,
                         studlab=NCH_ST, method="LRP")


def test_netmetabin_lrp_matches_r():
    res = _nma_lrp()
    assert res.method == "LRP"
    assert res.phi == pytest.approx(1.0)
    assert np.asarray(res.TE_common) == pytest.approx(
        np.array(LRP_R_TE), abs=1e-6)
    assert np.asarray(res.seTE_common) == pytest.approx(
        np.array(LRP_R_SE), abs=1e-6)
    # the direct (per-comparison) estimates also come from LRP, not MH
    got = np.asarray(res.TE_direct_common, dtype=float)
    assert np.nanmax(np.abs(got - np.array(LRP_R_DIRECT))) < 1e-6


def test_netmetabin_lrp_random_effects_scale_with_phi():
    res = _nma_lrp()
    assert np.asarray(res.TE_random) == pytest.approx(
        np.asarray(res.TE_common), abs=1e-12)
    assert np.asarray(res.seTE_random) == pytest.approx(
        np.asarray(res.seTE_common) * res.phi, abs=1e-12)


def test_netmetabin_lrp_large_network_matches_r():
    e1 = [48, 33, 14, 48, 11, 9, 46, 46, 48, 13, 13, 6, 27, 27, 27, 13, 13, 17,
          68, 68, 68, 51, 51, 57, 25, 25, 25, 30, 30, 25]
    n1 = [145, 139, 111, 177, 95, 93, 123, 123, 129, 164, 164, 80, 184, 184,
          189, 127, 127, 167, 163, 163, 163, 122, 122, 137, 134, 134, 134, 166,
          166, 139]
    e2 = [38, 22, 23, 29, 13, 11, 48, 39, 39, 6, 12, 12, 27, 23, 23, 17, 16, 16,
          51, 57, 81, 57, 81, 81, 30, 25, 34, 25, 34, 34]
    n2 = [116, 96, 184, 108, 108, 113, 129, 105, 105, 80, 154, 154, 189, 160,
          160, 167, 163, 163, 122, 137, 195, 137, 195, 195, 166, 139, 185, 139,
          185, 185]
    t1 = ["A", "A", "B", "C", "D", "A", "A", "A", "B", "B", "B", "C", "A", "A",
          "D", "C", "C", "D", "A", "A", "A", "B", "B", "C", "B", "B", "B", "C",
          "C", "D"]
    t2 = ["B", "C", "D", "E", "E", "E", "B", "C", "C", "C", "D", "D", "D", "E",
          "E", "D", "E", "E", "B", "C", "D", "C", "D", "D", "C", "D", "E", "D",
          "E", "E"]
    st = (["S1", "S2", "S3", "S4", "S5", "S6"] + ["S7"] * 3 + ["S8"] * 3
          + ["S9"] * 3 + ["S10"] * 3 + ["S11"] * 6 + ["S12"] * 6)
    res = ma.netmetabin(e1, n1, e2, n2, t1, t2, studlab=st, method="LRP")
    want_te = [
        [0.0, 0.004457298, 0.0192353728, 0.022202070, 0.0188699474],
        [-0.004457298, 0.0, 0.0147780753, 0.017744772, 0.0144126499],
        [-0.019235373, -0.014778075, 0.0, 0.002966697, -0.0003654255],
        [-0.022202070, -0.017744772, -0.0029666967, 0.0, -0.0033321222],
        [-0.018869947, -0.014412650, 0.0003654255, 0.003332122, 0.0],
    ]
    want_se = [
        [0.0, 0.1279832, 0.1287471, 0.1336333, 0.1527798],
        [0.1279832, 0.0, 0.1327887, 0.1325331, 0.1580782],
        [0.1287471, 0.1327887, 0.0, 0.1303105, 0.1407403],
        [0.1336333, 0.1325331, 0.1303105, 0.0, 0.1422686],
        [0.1527798, 0.1580782, 0.1407403, 0.1422686, 0.0],
    ]
    assert np.asarray(res.TE_common) == pytest.approx(np.array(want_te),
                                                      abs=1e-6)
    assert np.asarray(res.seTE_common) == pytest.approx(np.array(want_se),
                                                        abs=1e-6)


def test_netmetabin_lrp_has_no_tau2_and_no_inconsistency_q():
    res = _nma_lrp()
    assert np.isnan(res.tau2)
    assert np.isnan(res.Q_inconsistency)
    assert np.isnan(np.asarray(res.tau2_direct, dtype=float)).all()


def test_netmetabin_lrp_rejects_non_or():
    with pytest.raises(ValueError, match="sm='OR'"):
        ma.netmetabin(NCH_E1, NCH_N1, NCH_E2, NCH_N2, NCH_T1, NCH_T2,
                      studlab=NCH_ST, method="LRP", sm="RR")


def test_netmetabin_lrp_differs_from_mh_and_nch():
    lrp = np.asarray(_nma_lrp().TE_common)
    mh = np.asarray(ma.netmetabin(NCH_E1, NCH_N1, NCH_E2, NCH_N2, NCH_T1,
                                  NCH_T2, studlab=NCH_ST, method="MH").TE_common)
    nch = np.asarray(_nch().TE_common)
    assert np.max(np.abs(lrp - mh)) > 1e-3
    assert np.max(np.abs(lrp - nch)) > 1e-3
