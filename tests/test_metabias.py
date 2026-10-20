"""Tests for :func:`pimeta.metabias` (``meta::metabias``).

Every expected number below was printed by ``meta`` 8.5-0 in R from the same
input, so the assertions pin full-precision parity rather than approximations.
"""

import math

import numpy as np
import pytest

import pimeta as ma
from pimeta import MetabiasResult, metabias, metabin

EV_E = [12, 20, 9, 15, 30, 8, 22, 11, 18, 25]
N_E = [100, 120, 90, 110, 150, 80, 130, 95, 140, 160]
EV_C = [20, 24, 14, 22, 26, 15, 30, 16, 20, 28]
N_C = [100, 125, 95, 105, 145, 85, 128, 100, 135, 155]

TE = [0.1, -0.2, 0.05, -0.15, 0.3, 0.25, 0.12, -0.08, 0.2, 0.02]
SE = [0.10, 0.12, 0.11, 0.13, 0.14, 0.15, 0.09, 0.10, 0.12, 0.11]


@pytest.fixture(scope="module")
def or_result():
    return metabin(EV_E, N_E, EV_C, N_C, sm="OR", method="Inverse",
                   random=True, method_tau="DL")


# R: metabin(...) with the above counts, sm="OR", method="Inverse".
# (statistic, df, pval, bias, se.bias, intercept, se.intercept, tau)
R_OR = {
    "Begg": (-2.2360679774998, None, 0.025347318677468,
             -25.0, 11.180339887499, None, None, None),
    "Egger": (-2.8665077685457, 8, 0.020940963566831,
              -3.0729895200376, 1.0720325106939,
              0.79933586384049, 0.38287989609425, 0.51857409959776),
    "Thompson": (-1.4864966850636, 8, 0.17545752880622,
                 -3.0729895200376, 2.0672696756846,
                 0.79933586384049, 0.73833208482883, 0.0),
    "Schwarzer": (-2.2360679774998, None, 0.025347318677468,
                  -25.0, 11.180339887499, None, None, None),
    "Harbord": (-2.6978842937229, 8, 0.027162885581392,
                -3.1614983267202, 1.1718435568479,
                0.81829258691363, 0.41317378195074, 0.53081704855079),
    "Peters": (-3.648298678352, 8, 0.0065103568042751,
               -219.415703383536, 60.141924422331,
               0.63171516174279, 0.25771841250422, 0.91736879171815),
    "Deeks": (-3.8286510894657, 8, 0.0050275376265214,
              -28.2199965132009, 7.3707412490123,
              1.5328401133719, 0.48111147013787, 2.3717897904642),
    "Pustejovsky": (-3.6692282038817, 8, 0.0063165121894641,
                    -14.5843412437768, 3.9747708328275,
                    1.5920764727208, 0.51417422378586, 0.45076138024327),
    "Macaskill": (3.7075421111605, 8, 0.0059774264414523,
                  0.0040220961118253, 0.0010848416528346,
                  -1.2843771119645, 0.27418349900574, 0.44782250790984),
}


@pytest.mark.parametrize("method", sorted(R_OR))
def test_metabias_matches_r(or_result, method):
    stat, df, pval, bias, se_bias, itc, se_itc, tau = R_OR[method]
    res = metabias(or_result, method_bias=method)

    assert res.method_bias == method
    assert res.k == 10 and res.k_min == 10
    assert res.statistic == pytest.approx(stat, rel=0, abs=1e-9)
    assert res.pval == pytest.approx(pval, rel=0, abs=1e-12)
    assert res.estimate[0] == pytest.approx(bias, rel=0, abs=1e-9)
    assert res.estimate[1] == pytest.approx(se_bias, rel=0, abs=1e-9)

    if df is None:
        assert res.df is None
        assert res.intercept is None and res.se_intercept is None
        assert res.tau is None
        assert res.var_model == ""
    else:
        assert res.df == df
        assert res.intercept == pytest.approx(itc, rel=0, abs=1e-9)
        assert res.se_intercept == pytest.approx(se_itc, rel=0, abs=1e-9)
        assert res.tau == pytest.approx(tau, rel=0, abs=1e-9)


def test_metabias_method_labels(or_result):
    rank = metabias(or_result, method_bias="Begg")
    assert rank.method == "Rank correlation test of funnel plot asymmetry"
    corr = metabias(or_result, method_bias="Begg", correct=True)
    assert corr.method.endswith("(with continuity correction)")
    assert metabias(or_result, method_bias="Deeks").method == (
        "Funnel plot test for diagnostic odds ratios")
    assert metabias(or_result, method_bias="Egger").method == (
        "Linear regression test of funnel plot asymmetry")


def test_metabias_var_model(or_result):
    assert metabias(or_result, method_bias="Begg").var_model == ""
    assert metabias(or_result, method_bias="Schwarzer").var_model == ""
    assert metabias(or_result, method_bias="Thompson").var_model == "additive"
    assert metabias(or_result, method_bias="Egger").var_model == (
        "multiplicative")


def test_metabias_default_method_is_harbord_for_or():
    res = metabias(metabin(EV_E, N_E, EV_C, N_C, sm="OR", method="Inverse"))
    assert res.method_bias == "Harbord"
    assert res.statistic == pytest.approx(-2.6978842937229, rel=0, abs=1e-9)


def test_metabias_default_method_is_egger_for_rr():
    res = metabias(metabin(EV_E, N_E, EV_C, N_C, sm="RR", method="Inverse"))
    assert res.method_bias == "Egger"


def test_metabias_rr_harbord_matches_r():
    # R: metabin(sm="RR", method="Inverse"); metabias(m, method.bias="Harbord")
    m = metabin(EV_E, N_E, EV_C, N_C, sm="RR", method="Inverse")
    res = metabias(m, method_bias="Harbord")
    assert res.statistic == pytest.approx(-2.5830056292647, rel=0, abs=1e-9)
    assert res.df == 8
    assert res.pval == pytest.approx(0.032464233615187, rel=0, abs=1e-12)
    assert res.estimate[0] == pytest.approx(-2.6971492559581, rel=0, abs=1e-9)
    assert res.estimate[1] == pytest.approx(1.0441902353600, rel=0, abs=1e-9)
    assert res.intercept == pytest.approx(0.54580916329527, rel=0, abs=1e-9)
    assert res.se_intercept == pytest.approx(0.30697873824768, rel=0, abs=1e-9)
    assert res.tau == pytest.approx(0.54986800407968, rel=0, abs=1e-9)


# R: metabias(TE, seTE) on the generic 10-study vector above.
R_GENERIC = {
    "Egger": (0.51777846236912, 0.61862277663171, 1.5237087200236,
              2.9427811907273, -0.11904347867467, 0.33272026191577,
              1.3847422416824),
    "Thompson": (0.73655416591749, 0.48243108059692, 2.1301704039775,
                 2.8920756986338, -0.18917899718643, 0.33394224648628,
                 0.10830061990985),
    "Begg": (0.54321447625511, 0.58698214057129, 6.0, 11.045361017187),
}


@pytest.mark.parametrize("method", sorted(R_GENERIC))
def test_metabias_generic_matches_r(method):
    want = R_GENERIC[method]
    res = metabias(TE, SE, method_bias=method)
    assert res.statistic == pytest.approx(want[0], rel=0, abs=1e-9)
    assert res.pval == pytest.approx(want[1], rel=0, abs=1e-12)
    assert res.estimate[0] == pytest.approx(want[2], rel=0, abs=1e-9)
    assert res.estimate[1] == pytest.approx(want[3], rel=0, abs=1e-9)
    if len(want) == 7:
        assert res.df == 8
        assert res.intercept == pytest.approx(want[4], rel=0, abs=1e-9)
        assert res.se_intercept == pytest.approx(want[5], rel=0, abs=1e-9)
        assert res.tau == pytest.approx(want[6], rel=0, abs=1e-9)
    else:
        assert res.intercept is None


def test_metabias_continuity_correction_matches_r(or_result):
    # R: ks = -24 instead of -25, z = -2.1466252583998, p = 0.0318231273864902
    for method in ("Begg", "Schwarzer"):
        res = metabias(or_result, method_bias=method, correct=True)
        assert res.estimate[0] == pytest.approx(-24.0, rel=0, abs=1e-12)
        assert res.estimate[1] == pytest.approx(11.1803398874989, rel=0,
                                                abs=1e-9)
        assert res.statistic == pytest.approx(-2.1466252583998, rel=0,
                                              abs=1e-9)
        assert res.pval == pytest.approx(0.0318231273864902, rel=0, abs=1e-12)


# R: metformin subset (first six studies) with k.min=3.
R_SUBSET = {
    "Begg": (-1.6908055859299, -9.0, 5.32290647422377),
    "Schwarzer": (-1.6908055859299, -9.0, 5.32290647422377),
    "Harbord": (-3.10511582976737, -4.54329361256875, 1.46316397250441),
    "Peters": (-4.566085234348, -316.699828859779, 69.3591583611778),
    "Egger": (-3.25551636986184, -4.2831961537917, 1.31567335782541),
    "Deeks": (-4.78838631467935, -41.6501780886819, 8.69816580191085),
    "Pustejovsky": (-5.07363664956607, -21.6947595065724, 4.27597816024683),
    "Macaskill": (6.64275585536138, 0.0064502913362863,
                  0.000971026404813638),
    "Thompson": (-1.68379221295534, -4.28319615379169, 2.54377952388434),
}


@pytest.mark.parametrize("method", sorted(R_SUBSET))
def test_metabias_six_study_subset_matches_r(method):
    m = metabin(EV_E[:6], N_E[:6], EV_C[:6], N_C[:6], sm="OR",
                method="Inverse", method_tau="DL")
    res = metabias(m, method_bias=method, k_min=3)
    want = R_SUBSET[method]
    assert res.k == 6
    assert res.statistic == pytest.approx(want[0], rel=0, abs=1e-9)
    assert res.estimate[0] == pytest.approx(want[1], rel=0, abs=1e-9)
    assert res.estimate[1] == pytest.approx(want[2], rel=0, abs=1e-9)


def test_metabias_k_below_k_min_returns_only_k():
    res = metabias(TE[:4], SE[:4])
    assert isinstance(res, MetabiasResult)
    assert res.k == 4
    assert res.k_min == 10
    assert res.statistic is None and res.estimate is None
    assert res.intercept is None


def test_metabias_all_same_precision_raises():
    with pytest.raises(ValueError, match="same precision"):
        metabias(TE, [0.1] * 10, method_bias="Egger")


def test_metabias_requires_binary_data_for_harbord():
    with pytest.raises(ValueError, match="metabin"):
        metabias(TE, SE, method_bias="Harbord")
    with pytest.raises(ValueError, match="metabin"):
        metabias(TE, SE, method_bias="Deeks")
    with pytest.raises(ValueError, match="metabin"):
        metabias(TE, SE, method_bias="Schwarzer")
    with pytest.raises(ValueError, match="metabin"):
        metabias(TE, SE, method_bias="Peters")


def test_metabias_macaskill_needs_sample_sizes():
    with pytest.raises(ValueError, match="sample size"):
        metabias(TE, SE, method_bias="Macaskill")


def test_metabias_pustejovsky_needs_both_groups():
    with pytest.raises(ValueError, match="experimental group"):
        metabias(TE, SE, method_bias="Pustejovsky", n_c=list(range(10)))
    with pytest.raises(ValueError, match="control group"):
        metabias(TE, SE, method_bias="Pustejovsky", n_e=list(range(10)))


def test_metabias_length_mismatch():
    with pytest.raises(ValueError, match="must be equal"):
        metabias(TE, SE[:-1])


def test_metabias_unknown_method():
    with pytest.raises(ValueError, match="unknown bias method"):
        metabias(TE, SE, method_bias="Nope")


def test_metabias_abbreviations_and_legacy_aliases(or_result):
    assert metabias(or_result, method_bias="Beg").method_bias == "Begg"
    assert metabias(or_result, method_bias="Th").method_bias == "Thompson"
    # meta < 8 names still work (setmethodbias)
    assert metabias(or_result, method_bias="rank").method_bias == "Begg"
    assert metabias(or_result, method_bias="linreg").method_bias == "Egger"
    assert metabias(or_result, method_bias="mm").method_bias == "Thompson"
    assert metabias(or_result, method_bias="count").method_bias == "Schwarzer"
    assert metabias(or_result, method_bias="score").method_bias == "Harbord"


def test_metabias_drops_missing_studies(or_result):
    te = list(TE)
    se = list(SE)
    te[3] = math.nan
    res = metabias(te, se)
    assert res.k == 9


def test_metabias_result_dict_and_repr(or_result):
    res = metabias(or_result, method_bias="Harbord")
    d = res.to_dict()
    assert d["method_bias"] == "Harbord"
    assert d["k"] == 10
    assert d["estimate"] == res.estimate
    assert "Harbord" in repr(res)

    short = metabias(TE[:3], SE[:3], method_bias="Begg")
    assert "k=3" in repr(short)
    assert short.to_dict()["estimate"] is None


def test_metabias_metabin_keeps_raw_counts():
    m = metabin(EV_E, N_E, EV_C, N_C, sm="OR")
    assert m.event_e == pytest.approx(EV_E)
    assert m.n_e == pytest.approx(N_E)
    assert m.event_c == pytest.approx(EV_C)
    assert m.n_c == pytest.approx(N_C)
    assert m.method_tau == "REML"           # meta's own default
    assert metabin(EV_E, N_E, EV_C, N_C, method_tau="DL").method_tau == "DL"


# R: metagen(TE, seTE, method.tau = mt); metabias(m, method.bias = "Thompson")
R_THOMPSON_TAU = {
    "DL": (0.736140742629569, 0.482669113724613, 2.13262004576109,
           2.89702759575996, -0.189462320964861, 0.334542307862233,
           0.108713141045801),
    "REML": (0.73655416591749, 0.482431080596918, 2.13017040397746,
             2.8920756986338, -0.189178997186427, 0.333942246486276,
             0.108300619909852),
    "ML": (0.754725797689656, 0.47204416855599, 1.93673861394914,
           2.56614868589072, -0.16680753628511, 0.294334826338507,
           0.0788431308997834),
}


@pytest.mark.parametrize("method_tau", sorted(R_THOMPSON_TAU))
def test_metabias_thompson_uses_the_requested_tau2_estimator(method_tau):
    """The Thompson variant must honour ``x$method.tau`` exactly like meta."""
    z, p, bias, se_bias, itc, se_itc, tau = R_THOMPSON_TAU[method_tau]
    res = metabias(TE, SE, method_bias="Thompson", method_tau=method_tau)
    assert res.statistic == pytest.approx(z, rel=0, abs=1e-9)
    assert res.pval == pytest.approx(p, rel=0, abs=1e-12)
    assert res.estimate[0] == pytest.approx(bias, rel=0, abs=1e-9)
    assert res.estimate[1] == pytest.approx(se_bias, rel=0, abs=1e-9)
    assert res.intercept == pytest.approx(itc, rel=0, abs=1e-9)
    assert res.se_intercept == pytest.approx(se_itc, rel=0, abs=1e-9)
    assert res.tau == pytest.approx(tau, rel=0, abs=1e-9)


def test_metabias_needs_result_object_or_arrays():
    with pytest.raises(TypeError, match="pimeta result"):
        metabias({"yi": [1, 2, 3]})


# --------------------------------------------------------------------------- #
# continuous data: metacont() keeps n.e / n.c so Macaskill/Pustejovsky work
# --------------------------------------------------------------------------- #

MC_N_E = [50, 60, 45, 55, 70, 40, 65, 48, 68, 75]
MC_MEAN_E = [1.0, 1.9, 0.5, 1.6, 2.2, 0.7, 1.5, 0.9, 1.8, 1.1]
MC_SD_E = [0.50, 0.62, 0.44, 0.58, 0.71, 0.39, 0.66, 0.47, 0.69, 0.55]
MC_N_C = [52, 62, 47, 57, 72, 42, 67, 50, 70, 77]
MC_MEAN_C = [1.2, 1.5, 0.9, 1.2, 1.7, 1.0, 1.2, 1.1, 1.3, 1.0]
MC_SD_C = [0.60, 0.66, 0.52, 0.63, 0.75, 0.46, 0.70, 0.55, 0.72, 0.61]


def _mc():
    return ma.metacont(MC_N_E, MC_MEAN_E, MC_SD_E, MC_N_C, MC_MEAN_C, MC_SD_C,
                       sm="MD", method_tau="REML")


# R: metabias(metacont(...), method.bias=mb)
R_METACONT = {
    "Macaskill": (3.28262295858295, 0.011144076792296, 0.0106841377362253,
                  0.00325475629428895, -1.17309759976526, 0.383483275525377,
                  2.25867517178005),
    "Pustejovsky": (-3.70276443103624, 0.00601862886390642,
                    -13.5038820833968, 3.64697304808497, 2.61497072419453,
                    0.693685372496256, 2.10046906187831),
    "Egger": (3.4672728686131, 0.00847665098267389, 25.142886593742,
              7.25148770993587, -2.63658767157558, 0.781323896975299,
              2.18724444444582),
    "Thompson": (3.71203870975194, 0.00593892300685881, 26.4436510745322,
                 7.12375412601753, -2.77799973281629, 0.778319702897791,
                 0.2012720380105),
}


@pytest.mark.parametrize("method", sorted(R_METACONT))
def test_metabias_metacont_matches_r(method):
    z, p, bias, se_bias, itc, se_itc, tau = R_METACONT[method]
    res = metabias(_mc(), method_bias=method)
    assert res.statistic == pytest.approx(z, rel=0, abs=1e-9)
    assert res.pval == pytest.approx(p, rel=0, abs=1e-12)
    assert res.estimate[0] == pytest.approx(bias, rel=0, abs=1e-9)
    assert res.estimate[1] == pytest.approx(se_bias, rel=0, abs=1e-9)
    assert res.intercept == pytest.approx(itc, rel=0, abs=1e-9)
    assert res.se_intercept == pytest.approx(se_itc, rel=0, abs=1e-9)
    assert res.tau == pytest.approx(tau, rel=0, abs=1e-9)


def test_metabias_metacont_begg_matches_r():
    # R: z = 2.59383885389976, ks = 29, se.ks = 11.1803398874989
    res = metabias(_mc(), method_bias="Begg")
    assert res.statistic == pytest.approx(2.59383885389976, rel=0, abs=1e-9)
    assert res.pval == pytest.approx(0.0094910960534409, rel=0, abs=1e-12)
    assert res.estimate[0] == pytest.approx(29.0, rel=0, abs=1e-12)
    assert res.estimate[1] == pytest.approx(11.1803398874989, rel=0, abs=1e-9)


def test_metabias_metacont_keeps_arm_sizes():
    m = _mc()
    assert m.n_e == pytest.approx(MC_N_E)
    assert m.n_c == pytest.approx(MC_N_C)
    assert m.method_tau == "REML"
    assert m.to_dict()["n_e"] == pytest.approx(MC_N_E)


def test_metabias_accepts_a_metagen_result():
    """MetagenResult names its columns TE/seTE (like R), not yi/vi."""
    res = metabias(ma.metagen(TE, SE))
    assert res.method_bias == "Egger"
    assert res.statistic == pytest.approx(0.51777846236912, rel=0, abs=1e-9)
    assert res.pval == pytest.approx(0.61862277663171, rel=0, abs=1e-12)
    assert res.intercept == pytest.approx(-0.11904347867467, rel=0, abs=1e-9)


# --------------------------------------------------------------------------- #
# single-arm data: metaprop keeps event / n so Peters uses R's one-arm variance
# --------------------------------------------------------------------------- #

MP_EVENT = [5, 8, 3, 10, 6, 7, 4, 9, 5, 6]
MP_N = [100, 120, 90, 110, 150, 80, 130, 95, 140, 160]


def _mp():
    return ma.metaprop(MP_EVENT, MP_N, sm="PFT", method_tau="REML")


# R: metabias(metaprop(event, n, sm="PFT", method.tau="REML"), method.bias=mb)
R_METAPROP = {
    "Peters": (2.49252558163254, 0.0373748339110065, 17.8514570099874,
               7.16199550429293, 0.0969820604604067, 0.0656849769675382,
               0.10538869007608),
    "Macaskill": (-2.27921192324588, 0.0521354155025589,
                  -0.00120083870752983, 0.000526865753588939,
                  0.384908953315165, 0.0662218910289178, 0.925622638671691),
    "Egger": (2.22371847233794, 0.0568513762879757, 6.05186177973358,
              2.72150537714914, -0.0397803787053327, 0.125267438633865,
              0.934513843027846),
    "Thompson": (2.07809569539651, 0.0713260015068885, 6.05186177973349,
                 2.91221515599106, -0.0397803787053274, 0.134045567723209,
                 0.0),
}


@pytest.mark.parametrize("method", sorted(R_METAPROP))
def test_metabias_metaprop_matches_r(method):
    z, p, bias, se_bias, itc, se_itc, tau = R_METAPROP[method]
    res = metabias(_mp(), method_bias=method)
    assert res.statistic == pytest.approx(z, rel=0, abs=1e-9)
    assert res.pval == pytest.approx(p, rel=0, abs=1e-12)
    assert res.estimate[0] == pytest.approx(bias, rel=0, abs=1e-9)
    assert res.estimate[1] == pytest.approx(se_bias, rel=0, abs=1e-9)
    assert res.intercept == pytest.approx(itc, rel=0, abs=1e-9)
    assert res.se_intercept == pytest.approx(se_itc, rel=0, abs=1e-9)
    assert res.tau == pytest.approx(tau, rel=0, abs=1e-9)


def test_metabias_metaprop_begg_matches_r():
    # R: z = 1.52052622469986, ks = 17, se.ks = 11.1803398874989
    res = metabias(_mp(), method_bias="Begg")
    assert res.method_bias == "Begg"          # single-arm data -> not Harbord
    assert res.statistic == pytest.approx(1.52052622469986, rel=0, abs=1e-9)
    assert res.estimate[0] == pytest.approx(17.0, rel=0, abs=1e-12)


def test_metabias_metaprop_keeps_event_counts():
    m = _mp()
    assert m.event == pytest.approx(MP_EVENT)
    assert m.method_tau == "REML"
    assert m.to_dict()["event"] == pytest.approx(MP_EVENT)


def test_metabias_metaprop_rejects_two_arm_methods():
    for method in ("Harbord", "Deeks", "Schwarzer"):
        with pytest.raises(ValueError, match="metabin"):
            metabias(_mp(), method_bias=method)
    # Pustejovsky needs both arms
    with pytest.raises(ValueError, match="control group"):
        metabias(_mp(), method_bias="Pustejovsky")
