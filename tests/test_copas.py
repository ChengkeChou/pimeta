"""Copas selection model, checked against ``metasens::copas``.

Two datasets. **Fleiss 1993** is the example in metasens' own documentation:
funnel asymmetry is present but the residual-selection test is never
significant, so the analysis reports the "no selection" point. The
**synthetic** set has a deliberate small-study effect, so the test *is*
significant at first and the reported estimate is a later point on the path —
that is the branch worth testing.

References: R 4.6.1, metasens 1.5.3, meta 8.5.0.

What is and is not bit-comparable
---------------------------------
The likelihood, the fits at a given ``(gamma0, gamma1)``, the sensitivity
surface, the publication probabilities and the implied number of unpublished
studies all reproduce metasens. The *path* the adjusted estimate is reported
from is metasens' graphical heuristic: it is built from iso-effect contours,
and metasens' contour tracer splits a level into more pieces than the
connected-component grouping used here (16 pieces against 12 on the synthetic
set). That changes the path slightly, so the adjusted estimate agrees to about
1e-5 rather than to machine precision, and the number of points on the path
can differ. The tolerances below say so explicitly instead of hiding it.
"""

import numpy as np
import pytest

import pimeta as ma
from pimeta.copas import copas, copas_fit, copas_loglik

EXACT = 1e-12
FITTED = 1e-5

# --- Fleiss 1993, log odds ratios from metabin(method = "Inverse") ---------
FLEISS_TE = [
    -0.32890115955623839, -0.38454574588442231, -0.21956223542392733,
    -0.22222061631560888, -0.22546720381201127, 0.12463630249491103,
    -0.11096579579004434,
]
FLEISS_SE = [
    0.1972198088133556, 0.20289716633892102, 0.14314855217027966,
    0.2544888962290453, 0.18761568222036212, 0.098064939620212996,
    0.038809765598515331,
]
R_FLEISS_TAU = 0.12142187950906422
R_FLEISS_MU = -0.14124304491207845
# copas.loglik.without.beta / .with.beta at (mu, rho=0.3, tau), gamma=(0.5,0.04)
R_FLEISS_LL = -6.8665726835019356
R_FLEISS_LL_BETA = -6.5493135704206802
R_FLEISS_ADJUST = (-0.12736417271470959, 0.070979115157140879)
# TE.contour at four corners and four interior grid points (gamma0, gamma1)
R_FLEISS_CONTOUR = [
    (0, 0, -0.00043633872705700296),
    (0, 19, -0.030855765976738541),
    (19, 0, -0.12457705508754401),
    (19, 19, -0.12587759952615946),
    (4, 9, -0.05988246394988006),
    (9, 3, -0.095526776866085544),
]
R_FLEISS_NOBS = [22, 23, 23, 23, 23, 23, 23, 23, 22, 21, 12, 6]

# --- synthetic data with a clear small-study effect -----------------------
SYN_SE = [0.050, 0.070, 0.090, 0.110, 0.130, 0.150,
          0.180, 0.210, 0.250, 0.300, 0.350, 0.400]
SYN_WIGGLE = [0.010, -0.020, 0.015, -0.012, 0.018, -0.015,
              0.012, -0.010, 0.014, -0.011, 0.013, -0.009]
R_SYN_ADJUST = (0.50996805418832281, 0.030819202168196842)
R_SYN_PVAL_RSB = [0.020043567507689986, 0.021214876812955863,
                  0.045190989452606911, 0.1059273733219116,
                  0.14711738621221382, 0.26217757967781515]


def _fleiss():
    return ma.rma(FLEISS_TE, vi=[s * s for s in FLEISS_SE], method="REML")


# the surface costs 400 fits; build it once per session
_FLEISS_CACHE = None


def _fleiss_cached():
    global _FLEISS_CACHE
    if _FLEISS_CACHE is None:
        _FLEISS_CACHE = copas(_fleiss())
    return _FLEISS_CACHE


def _synthetic():
    se = np.asarray(SYN_SE)
    yi = 0.60 - 1.20 * se + np.asarray(SYN_WIGGLE)
    return ma.rma(yi, vi=se * se, method="REML"), yi, se


# --------------------------------------------------------------------------- #
# the likelihood
# --------------------------------------------------------------------------- #

def test_the_likelihood_matches_metasens():
    p = [R_FLEISS_MU, 0.3, R_FLEISS_TAU]
    assert copas_loglik(p, FLEISS_TE, FLEISS_SE, 0.5, 0.04) == pytest.approx(
        R_FLEISS_LL, rel=EXACT)


def test_the_likelihood_with_the_residual_slope_matches_metasens():
    p = [R_FLEISS_MU, 0.3, R_FLEISS_TAU, 0.15]
    assert copas_loglik(p, FLEISS_TE, FLEISS_SE, 0.5, 0.04) == pytest.approx(
        R_FLEISS_LL_BETA, rel=EXACT)


def test_the_likelihood_rejects_a_wrong_number_of_parameters():
    with pytest.raises(ValueError, match="3 or 4"):
        copas_loglik([1.0, 0.2], FLEISS_TE, FLEISS_SE, 0.5, 0.04)


# --------------------------------------------------------------------------- #
# fitting at one point of the grid
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("g0,g1,expected", R_FLEISS_CONTOUR)
def test_a_grid_fit_matches_metasens(g0, g1, expected):
    g0_values = np.linspace(-0.48212536825293562, 2.0, 20)
    g1_values = np.linspace(0.0, 0.059073328753450262, 20)
    fit = copas_fit(FLEISS_TE, FLEISS_SE, g0_values[g0], g1_values[g1],
                    mu0=R_FLEISS_MU, rho0=-0.49995, tau0=R_FLEISS_TAU)
    assert fit.mu == pytest.approx(expected, abs=FITTED)
    assert fit.converged


# --------------------------------------------------------------------------- #
# the analysis
# --------------------------------------------------------------------------- #

def test_fleiss_ranges_and_grid_follow_metasens():
    r = _fleiss_cached()
    assert r.left is False
    assert r.gamma1_range[0] == 0.0
    assert r.gamma1_range[1] == pytest.approx(0.059073328753450262,
                                              rel=1e-12)
    assert r.gamma0_range[0] == pytest.approx(-0.48212536825293562,
                                              rel=1e-12)
    assert r.gamma0_range[1] == pytest.approx(2.0, rel=EXACT)
    assert r.gamma0.size == 20 and r.gamma1.size == 20


def test_fleiss_surface_matches_metasens():
    r = _fleiss_cached()
    for i, j, expected in R_FLEISS_CONTOUR:
        assert r.te_contour[i, j] == pytest.approx(expected, abs=FITTED)


def test_fleiss_contour_levels_and_piece_sizes_match_metasens():
    r = _fleiss_cached()
    assert r.contour["nobs"] == R_FLEISS_NOBS
    # metasens' automatic levels for this surface are -0.12 .. -0.01
    assert r.levels[0] == pytest.approx(-0.12, abs=1e-12)
    assert r.levels[-1] == pytest.approx(-0.01, abs=1e-12)
    assert len(r.levels) == 12


def test_fleiss_reports_the_no_selection_point():
    """Residual selection is never significant, so nothing is adjusted."""
    r = _fleiss_cached()
    assert np.all(np.asarray(r.pval_rsb) > 0.1)
    assert r.te_adjust == pytest.approx(R_FLEISS_ADJUST[0], abs=1e-6)
    assert r.se_adjust == pytest.approx(R_FLEISS_ADJUST[1], rel=1e-3)


def test_fleiss_publication_probabilities_match_metasens():
    r = _fleiss_cached()
    expected = [1.0, 0.93621246723340579, 0.85054558380747969,
                0.77077087649556986, 0.7005699300364665,
                0.63984787844561297, 0.58757382087928389,
                0.54192372093131769, 0.50106927379317512,
                0.45970467858647429, 0.39537091417560682]
    assert np.asarray(r.publprob) == pytest.approx(expected, abs=1e-5)


def test_the_synthetic_data_detects_selection_and_adjusts():
    """The residual-selection p-value crosses 0.1, so a later point is used."""
    fit, _yi, _se = _synthetic()
    r = copas(fit)
    pval = np.asarray(r.pval_rsb)
    assert pval[0] <= 0.1                       # selection is significant
    assert np.any(pval > 0.1)                   # and stops being so
    assert r.te_adjust == pytest.approx(R_SYN_ADJUST[0], rel=1e-4)
    assert r.se_adjust == pytest.approx(R_SYN_ADJUST[1], rel=5e-3)
    # the reported value is a point on the path, not the first one
    assert r.te_adjust == pytest.approx(r.te_line[3], rel=1e-12)


def test_the_synthetic_p_values_track_metasens():
    fit, _yi, _se = _synthetic()
    r = copas(fit)
    got = np.asarray(r.pval_rsb)[:5]
    assert got == pytest.approx(R_SYN_PVAL_RSB[:5], abs=5e-3)


def test_the_number_unpublished_counts_implied_studies():
    fit, yi, se = _synthetic()
    r = copas(fit)
    # no selection implies none missing
    assert r.n_unpubl[0] == pytest.approx(0.0, abs=1e-9)
    # and it grows as the analysis assumes more selection
    assert np.all(np.diff(np.asarray(r.n_unpubl)) >= -1e-9)
    # recomputed independently at the first real point
    from scipy import stats as _stats

    p = _stats.norm.cdf(r.gamma0_line[1] + r.gamma1_line[1] / np.array(se))
    expected = float(np.sum((1.0 - p) / p))
    assert r.n_unpubl[1] == pytest.approx(expected, rel=1e-10)


def test_a_standard_error_is_withheld_when_rho_sits_on_its_bound():
    """A constrained fit has no meaningful curvature in that direction."""
    fit, _yi, _se = _synthetic()
    r = copas(fit)
    at_bound = np.abs(np.abs(np.asarray(r.rho_line)) - r.rho_bound) < 1e-6
    if np.any(at_bound):
        assert np.all(np.isnan(np.asarray(r.se_line)[at_bound]))


# --------------------------------------------------------------------------- #
# interface
# --------------------------------------------------------------------------- #

def test_accepts_plain_arrays():
    r = copas(FLEISS_TE, sei=FLEISS_SE)
    assert r.te_adjust == pytest.approx(R_FLEISS_ADJUST[0], abs=1e-5)


def test_rejects_too_few_studies():
    with pytest.raises(ValueError, match="three"):
        copas([0.1, 0.2], sei=[0.1, 0.2])


def test_rejects_non_positive_standard_errors():
    with pytest.raises(ValueError, match="positive"):
        copas([0.1, 0.2, 0.3], sei=[0.1, 0.0, 0.3])


def test_requires_standard_errors_for_arrays():
    with pytest.raises(ValueError, match="sei"):
        copas([0.1, 0.2, 0.3])


def test_the_slope_can_be_fixed():
    fit = _fleiss()
    a = copas(fit)
    b = copas(fit, slope=0.5)
    assert b.slope == 0.5
    assert a.slope != b.slope


def test_coef_and_sensitivity_tables():
    r = _fleiss_cached()
    row = r.coef_table()[0]
    assert row["estimate"] == pytest.approx(r.te_adjust, rel=EXACT)
    table = r.sensitivity_table()
    assert len(table) == len(r.publprob)
    assert table[0]["publprob"] == pytest.approx(1.0, rel=EXACT)
    assert "Copas selection model" in str(r)


# --------------------------------------------------------------------------- #
# the record round-trip the package requires of every new model
# --------------------------------------------------------------------------- #

def test_a_copas_analysis_round_trips_through_a_record(tmp_path):
    from pimeta.record import load_record, save_record

    r = _fleiss_cached()
    path = tmp_path / "copas.json"
    save_record(path, r, mode="digest", function="copas")
    loaded = load_record(path)
    assert loaded.kind == "copas"
    assert loaded.results["values"]["te_adjust"] == pytest.approx(
        r.te_adjust, rel=EXACT)
    assert np.asarray(loaded.results["values"]["te_contour"]) == pytest.approx(
        np.asarray(r.te_contour), rel=EXACT)
