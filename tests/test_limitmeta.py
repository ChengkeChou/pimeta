"""Limit meta-analysis, checked against ``metasens::limitmeta``.

Two datasets: Moore 1998 (37 trials on smoking cessation, the example in
metasens' own documentation, with pronounced small-study effects) and a small
hand-written set whose effects fall almost exactly on a line in the
generalised radial plot.

Reference values: metasens 1.5.3 / meta 8.5.0, R 4.6.1.
"""

import numpy as np
import pytest

import pimeta as ma
from pimeta.limitmeta import limitmeta, radial_regression

EXACT = 1e-12

# --- Moore 1998, log odds ratios and standard errors -----------------------
MOORE_TE = [
    1.8827312474337816, 0.38327279651380214, 1.470342414602102,
    1.7186372043990923, 2.5649493574615367, 0.88562590510750827,
    1.0216512475319812, -0.5389965007326869, 1.1284652518177909,
    1.2527629684953681, 1.6835458845878224, 0.11592946700706044,
    0.1667748330556548, 3.4692021107762128, 2.4163195401244288,
    1.8777018990287797, 0.74721440183022114, 1.3221949179257073,
    0.26021025178758261, 2.2335922215070942, 2.0149030205422647,
    3.2188758248682006, 1.6094379124341003, 0.38769002226859095,
    1.7339046765298807, 0.86977071623926216, 1.8294997972109019,
    1.550597412411167, 1.4240346891027378, 2.6190165787318125,
    3.5828064339728143, 4.7449321283632502, 0.15848908142585647,
    1.6657187483326901, 1.4945079457601234, 3.044522437723423,
    1.7382707842770688,
]
MOORE_SE = [
    0.57994431088399179, 0.26011309639902785, 0.30968140174865444,
    0.45978890324122285, 0.93644171037127399, 0.42949435602427444,
    0.55777335102271708, 0.46929531772445293, 0.32168665190168177,
    0.47245559126153402, 0.74691060902591144, 0.29827633048217583,
    0.40864587322428053, 0.88561488554009526, 0.52896466255853325,
    0.6129817340955942, 0.37297506923157348, 0.28059265805522104,
    0.44770119063582292, 0.7420210496308296, 0.61553951042064625,
    1.5452159420910423, 0.59052345314809362, 0.31234104964959797,
    0.46758094782601417, 0.52461076067542711, 0.8623164985025763,
    0.67098170632026188, 0.75955452531274992, 0.49638821456512672,
    1.4734400083705761, 1.6021724382035569, 0.29979087115414549,
    0.35295112401201656, 0.62451446341803407, 1.2598815766974241,
    0.67666495245095848,
]
R_MOORE_TAU = 0.69856178932381652
R_MOORE_ALPHA = 3.7634282584062486
R_MOORE_BETA = -2.0134874331864072
R_MOORE_ADJUST = {
    "beta0": (0.61549974499767623, 0.2211004153219942,
              0.18215089399971973, 1.0488485959956328),
    "betalim": (0.41845564549683734, 0.19233403874699398,
                0.041487856551597979, 0.79542343444207675),
    "mulim": (0.56966827511367235, 0.075870852934626573,
              0.42096413588546922, 0.71837241434187549),
}
R_MOORE_Q = 113.34890225370093
R_MOORE_Q_SMALL = 44.02666677345141
R_MOORE_Q_RESID = 69.322235480249518
R_MOORE_G2 = 0.91536792872817019
R_MOORE_TE_LIMIT_1 = 0.98428860934238749

# --- a synthetic set that lies almost on a line ---------------------------
SYN_TE = [0.10, 0.25, 0.40, 0.55, 0.70, 0.20, 0.35, 0.50, 0.65, 0.80]
SYN_SE = [0.80, 0.55, 0.35, 0.22, 0.10, 0.70, 0.45, 0.28, 0.18, 0.08]
R_SYN_TAU = 0.03836369681827239
R_SYN_ALPHA = -1.1607371175670971
R_SYN_BETA = 0.8646287914171974
R_SYN_ADJUST = {
    "beta0": (0.82009862455313787, 0.081766518994962734,
              0.65983919218180076, 0.98035805692447497),
    "betalim": (0.82638401479718016, 0.087287805114723629,
                0.65530306048277076, 0.99746496911158955),
    "mulim": (0.82265420805400191, 0.054189511491892661,
              0.71644471719007297, 0.92886369891793086),
}
R_SYN_Q_SMALL = 5.1378288472508586
R_SYN_Q_RESID = 0.55674897157087277
R_SYN_G2 = 0.00068416181027286527
R_SYN_TE_LIMIT_1 = 0.82800339603443862


def _fit(te, se):
    return ma.rma(te, vi=[s * s for s in se], method="REML")


def _moore():
    return limitmeta(_fit(MOORE_TE, MOORE_SE))


def _syn():
    return limitmeta(_fit(SYN_TE, SYN_SE), method_adjust="beta0")


# --------------------------------------------------------------------------- #
# the radial plot underneath
# --------------------------------------------------------------------------- #

def test_radial_regression_is_ordinary_least_squares_on_the_inverse_scale():
    te = np.array([0.2, -0.1, 0.5, 0.9, 0.3])
    se = np.array([0.4, 0.3, 0.2, 0.1, 0.25])
    reg = radial_regression(te, se)
    x, y = 1.0 / se, te / se
    slope, intercept = np.polyfit(x, y, 1)
    assert reg.slope == pytest.approx(slope, rel=EXACT)
    assert reg.intercept == pytest.approx(intercept, rel=EXACT)


def test_radial_regression_standard_error_uses_a_unit_residual_variance():
    """The standardised residuals have variance 1 by construction."""
    te = np.array([0.2, -0.1, 0.5, 0.9, 0.3])
    se = np.array([0.4, 0.3, 0.2, 0.1, 0.25])
    reg = radial_regression(te, se)
    x = 1.0 / se
    ssx = float(np.sum((x - x.mean()) ** 2))
    assert reg.se_slope == pytest.approx(1.0 / np.sqrt(ssx), rel=EXACT)
    assert reg.se_intercept == pytest.approx(reg.se_slope * np.mean(x * x),
                                             rel=EXACT)


def test_radial_regression_rejects_too_few_studies():
    with pytest.raises(ValueError, match="three"):
        radial_regression([0.1, 0.2], [0.1, 0.2])


# --------------------------------------------------------------------------- #
# the estimator, on two datasets
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("method", ["beta0", "betalim", "mulim"])
def test_moore1998_matches_metasens(method):
    r = limitmeta(_fit(MOORE_TE, MOORE_SE), method_adjust=method)
    te, se, lo, hi = R_MOORE_ADJUST[method]
    assert r.TE_adjust == pytest.approx(te, abs=EXACT)
    assert r.seTE_adjust == pytest.approx(se, rel=1e-12)
    assert r.ci_lb == pytest.approx(lo, abs=1e-10)
    assert r.ci_ub == pytest.approx(hi, abs=1e-10)


@pytest.mark.parametrize("method", ["beta0", "betalim", "mulim"])
def test_synthetic_matches_metasens(method):
    r = limitmeta(_fit(SYN_TE, SYN_SE), method_adjust=method)
    te, se, lo, hi = R_SYN_ADJUST[method]
    assert r.TE_adjust == pytest.approx(te, abs=EXACT)
    assert r.seTE_adjust == pytest.approx(se, rel=1e-12)
    assert r.ci_lb == pytest.approx(lo, abs=1e-10)
    assert r.ci_ub == pytest.approx(hi, abs=1e-10)


def test_moore1998_bias_parameter_and_limit_effect():
    r = _moore()
    assert r.tau == pytest.approx(R_MOORE_TAU, rel=1e-12)
    assert r.alpha == pytest.approx(R_MOORE_ALPHA, rel=1e-11)
    assert r.beta == pytest.approx(R_MOORE_BETA, rel=1e-11)
    assert r.TE_limit[0] == pytest.approx(R_MOORE_TE_LIMIT_1, rel=1e-12)


def test_moore1998_heterogeneity_decomposition():
    r = _moore()
    assert r.Q == pytest.approx(R_MOORE_Q, rel=1e-12)
    assert r.Q_small == pytest.approx(R_MOORE_Q_SMALL, rel=1e-11)
    assert r.Q_resid == pytest.approx(R_MOORE_Q_RESID, rel=1e-11)
    assert r.G_squared == pytest.approx(R_MOORE_G2, rel=1e-11)
    # the decomposition has to close
    assert r.Q_small + r.Q_resid == pytest.approx(r.Q, rel=1e-12)


def test_synthetic_bias_parameter_and_decomposition():
    r = _syn()
    assert r.tau == pytest.approx(R_SYN_TAU, rel=1e-12)
    assert r.alpha == pytest.approx(R_SYN_ALPHA, rel=1e-11)
    assert r.beta == pytest.approx(R_SYN_BETA, rel=1e-11)
    assert r.TE_limit[0] == pytest.approx(R_SYN_TE_LIMIT_1, rel=1e-12)
    assert r.Q_small == pytest.approx(R_SYN_Q_SMALL, rel=1e-11)
    assert r.Q_resid == pytest.approx(R_SYN_Q_RESID, rel=1e-11)
    assert r.G_squared == pytest.approx(R_SYN_G2, rel=1e-10)


def test_beta0_is_the_limit_effect_shifted_by_the_bias():
    r = _moore()
    assert r.TE_adjust == pytest.approx(r.beta + r.tau * r.alpha, rel=EXACT)


def test_shrinkage_pulls_studies_towards_the_limit_effect():
    r = _moore()
    te = np.asarray(r.TE)
    move = np.abs(np.asarray(r.TE_limit) - r.beta)
    before = np.abs(te - r.beta)
    assert np.all(move <= before + 1e-12)
    # the shrinkage factor is tau / s_i, so it shrinks as the study gets
    # less precise.  (How far a study *moves* also depends on where it
    # started, so the factor is the statement that holds in general.)
    factor = (np.asarray(r.TE_limit) - r.beta) / (te - r.beta)
    assert np.allclose(factor, r.tau / np.sqrt(np.asarray(r.seTE) ** 2
                                               + r.tau ** 2))
    assert factor[4] < factor[22]       # study 5 is the least precise


# --------------------------------------------------------------------------- #
# interface
# --------------------------------------------------------------------------- #

def test_accepts_a_meta_style_result_holding_the_fit():
    class Wrapper:
        def __init__(self, random):
            self.random = random

    fit = _fit(SYN_TE, SYN_SE)
    a = limitmeta(fit)
    b = limitmeta(Wrapper(fit))
    assert b.TE_adjust == pytest.approx(a.TE_adjust, rel=EXACT)


def test_level_defaults_to_the_model_and_accepts_both_conventions():
    fit = _fit(SYN_TE, SYN_SE)          # RmaResult stores 0.95, not 95
    default = limitmeta(fit)
    assert default.level == pytest.approx(95.0, rel=EXACT)
    same = limitmeta(fit, level=95)
    assert same.ci_lb == pytest.approx(default.ci_lb, rel=EXACT)
    as_fraction = limitmeta(fit, level=0.95)
    assert as_fraction.ci_lb == pytest.approx(default.ci_lb, rel=EXACT)
    narrow = limitmeta(fit, level=90)
    assert narrow.ci_ub - narrow.ci_lb < default.ci_ub - default.ci_lb


def test_rejects_an_unknown_adjustment():
    with pytest.raises(ValueError, match="method_adjust"):
        limitmeta(_fit(SYN_TE, SYN_SE), method_adjust="shrink")


def test_rejects_a_model_with_moderators():
    fit = ma.rma(MOORE_TE, vi=[s * s for s in MOORE_SE],
                 mods=np.arange(len(MOORE_TE), dtype=float))
    with pytest.raises(ValueError, match="intercept-only"):
        limitmeta(fit)


def test_rejects_something_that_is_not_a_fit():
    with pytest.raises(TypeError, match="RmaResult"):
        limitmeta({"TE": [0.1, 0.2, 0.3]})


def test_coef_table_and_text():
    r = _moore()
    row = r.coef_table()[0]
    assert row["method"] == "beta0"
    assert row["estimate"] == pytest.approx(r.TE_adjust, rel=EXACT)
    text = str(r)
    assert "limit meta-analysis" in text
    assert "Q.small" in text


# --------------------------------------------------------------------------- #
# the record round-trip the package requires of every new model
# --------------------------------------------------------------------------- #

def test_a_limit_meta_analysis_round_trips_through_a_record(tmp_path):
    from pimeta.record import load_record, save_record

    r = _moore()
    path = tmp_path / "limitmeta.json"
    save_record(path, r, mode="digest", function="limitmeta")
    loaded = load_record(path)
    assert loaded.kind == "limitmeta"
    assert loaded.results["values"]["TE_adjust"] == pytest.approx(
        r.TE_adjust, rel=EXACT)
    assert loaded.results["values"]["alpha"] == pytest.approx(r.alpha,
                                                              rel=EXACT)
