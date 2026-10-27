"""Multivariate reporting layer, checked against metafor.

The reference numbers come from metafor 5.2.1 (R 4.6.1) on the Berkey 1998
dataset, the example metafor's own `vcalc` documentation uses.  ``V`` there is
built with the documented idiom

    vcalc(dat$vi, cluster = dat$trial, obs = dat$outcome, rho = 0.5)

so the same call in pimeta has to produce the same matrix, and the fit on top
of it the same variances.

Tolerances follow the package convention: anything that comes out of the
``rma.mv`` optimiser is compared at 1e-6 (a different optimiser, the same
objective), everything determined in closed form at 1e-12.
"""

import numpy as np
import pytest

import pimeta as ma

EXACT = 1e-12
FITTED = 1e-6

# --- Berkey 1998, 5 trials x 2 outcomes (PD, AL) ---------------------------
YI = [0.47, -0.32, 0.20, -0.60, 0.40, -0.12, 0.26, -0.31, 0.56, -0.39]
VI = [0.0075, 0.0077, 0.0057, 0.0008, 0.0021,
      0.0014, 0.0029, 0.0015, 0.0148, 0.0304]
STUDY = [1, 1, 2, 2, 3, 3, 4, 4, 5, 5]
OUTCOME = ["PD", "AL"] * 5

# metafor: rma.mv(yi, V, mods = ~ outcome - 1, random = ~ outcome | trial,
#                 struct = "UN", method = "REML")
R_BETA = [-0.34526550634451253, 0.35777925561202201]   # AL, PD (sorted)
R_SE = [0.088114938509632437, 0.060799946925128849]
R_TAU2 = [0.032770893240807383, 0.012822674623970366]
R_RHO = 0.54377810074998145
R_G = [[0.032770893240807383, 0.01114692718471918],
       [0.01114692718471918, 0.012822674623970368]]
R_VB = [[0.0077642423885563062, 0.0027727214226177621],
        [0.0027727214226177621, 0.003696633546098485]]
R_QE = 135.74928438400249
R_V12 = 0.003799671038392665     # V[1, 2] of the vcalc-built matrix

# metafor, refitting on a grid of within-study correlations.
# The columns of `~ outcome - 1` are AL, PD (R sorts the factor levels).
R_GRID_RHO = [0.0, 0.2, 0.5, 0.8]
R_GRID_AL = [-0.32535451315816316, -0.33358474215180672,
             -0.34526550612958284, -0.35638856538062169]
R_GRID_PD = [0.34455541844705023, 0.34961568151466382,
             0.35777925600944505, 0.36653587218493616]
R_GRID_SE_AL = [0.087854758594548832, 0.087884124383308684,
                0.088114938281950342, 0.088631548900879362]
R_GRID_SE_PD = [0.054600248113116902, 0.057068008028820741,
                0.060799947582569028, 0.064561331041831316]


def _data():
    return {"yi": np.array(YI), "study": np.array(STUDY),
            "outcome": np.array(OUTCOME)}


def _fit():
    from pimeta.data_compat import design_matrix

    data = _data()
    X, _ = design_matrix("~ outcome - 1", data)
    V = ma.vcalc(np.array(VI), cluster=data["study"], obs=data["outcome"],
                 rho=0.5)
    return ma.rma_mv(np.array(YI), V=V, mods=X, data=data,
                     random="~ outcome | study", struct="UN",
                     intercept=False)


# --------------------------------------------------------------------------- #
# vcalc — the four dependency structures
# --------------------------------------------------------------------------- #

def test_vcalc_rebuilds_metafor_v_matrix():
    V = ma.vcalc(np.array(VI), cluster=np.array(STUDY), obs=np.array(OUTCOME),
                 rho=0.5)
    assert V[0, 1] == pytest.approx(R_V12, rel=EXACT)
    assert V[1, 0] == pytest.approx(R_V12, rel=EXACT)
    assert V[0, 0] == pytest.approx(VI[0], rel=EXACT)
    # different trials are independent
    assert V[0, 2] == 0.0
    assert np.allclose(V, V.T)


def test_vcalc_same_construct_scalar_rho():
    """R: obs = c(1,2,3,1,2,3), rho = 0.4 -> 0.4 on every within-cluster pair."""
    vi = [0.04, 0.05, 0.06, 0.07, 0.08, 0.09]
    V = ma.vcalc(vi, cluster=[1, 1, 1, 2, 2, 2], obs=[1, 2, 3, 1, 2, 3],
                 rho=0.4)
    for i in range(3):
        for j in range(i + 1, 3):
            assert V[i, j] / np.sqrt(vi[i] * vi[j]) == pytest.approx(0.4,
                                                                     rel=EXACT)


def test_vcalc_same_construct_correlation_matrix():
    """R returns exactly the requested per-pair correlations."""
    vi = [0.04, 0.05, 0.06, 0.07, 0.08, 0.09]
    R = np.array([[1.0, 0.2, 0.3], [0.2, 1.0, 0.6], [0.3, 0.6, 1.0]])
    V = ma.vcalc(vi, cluster=[1, 1, 1, 2, 2, 2], obs=[1, 2, 3, 1, 2, 3],
                 rho=R)
    got = {(0, 1): 0.2, (0, 2): 0.3, (1, 2): 0.6}
    for (i, j), want in got.items():
        assert V[i, j] / np.sqrt(vi[i] * vi[j]) == pytest.approx(want,
                                                                 rel=EXACT)


def test_vcalc_matrix_is_read_in_the_factor_order():
    """`rho_names` reorders a matrix given in a different level order."""
    vi = [0.04, 0.05, 0.06, 0.07]
    R = np.array([[1.0, 0.9], [0.9, 1.0]])     # indexed as [b, a]
    V = ma.vcalc(vi, cluster=[1, 1, 2, 2], obs=["a", "b", "a", "b"], rho=R,
                 rho_names=["b", "a"])
    assert V[0, 1] / np.sqrt(vi[0] * vi[1]) == pytest.approx(0.9, rel=EXACT)


def test_vcalc_shared_control_equal_weights_give_exactly_half():
    """Equal arm sizes make the derived correlation 0.5, not an assumption."""
    vi = [0.04, 0.05, 0.06]
    V = ma.vcalc(vi, cluster=[1, 1, 1],
                 grp1=["os", "pfs", "orr"], grp2=["t1", "t1", "t2"])
    assert V[0, 1] / np.sqrt(vi[0] * vi[1]) == pytest.approx(0.5, rel=EXACT)
    # different reference conditions do not share an arm
    assert V[0, 2] == 0.0
    assert V[1, 2] == 0.0


def test_vcalc_shared_control_unequal_weights_match_metafor():
    """R gives 0.4780914437 for these weights; so must we."""
    vi = [0.04, 0.05, 0.06]
    V = ma.vcalc(vi, cluster=[1, 1, 1],
                 grp1=["os", "pfs", "orr"], grp2=["t1", "t1", "t2"],
                 w1=[200, 400, 100], w2=[300, 300, 300])
    assert V[0, 1] / np.sqrt(vi[0] * vi[1]) == pytest.approx(
        0.4780914437, rel=1e-10)


def test_shared_control_corr_is_half_for_equal_arms():
    assert ma.shared_control_corr(10, 10, 10) == pytest.approx(0.5, rel=EXACT)
    # a *small* shared arm carries almost all the information, so both
    # contrasts are nearly the same estimate and the correlation approaches 1
    assert ma.shared_control_corr(1, 1000, 1000) > 0.99
    # a huge shared arm is measured without error, so the contrasts are
    # nearly independent
    assert ma.shared_control_corr(1000, 10, 10) < 0.02


def test_vcalc_subgroups_are_independent():
    vi = [0.04, 0.05, 0.06, 0.07]
    V = ma.vcalc(vi, cluster=[1, 1, 1, 1], subgroup=["a", "a", "b", "b"],
                 obs=[1, 2, 1, 2], rho=0.5)
    assert V[0, 1] != 0.0
    assert V[2, 3] != 0.0
    assert V[0, 2] == 0.0
    assert V[1, 3] == 0.0


def test_vcalc_different_constructs_use_the_between_correlation():
    vi = [0.04, 0.05, 0.06, 0.07]
    V = ma.vcalc(vi, cluster=[1, 1, 2, 2], type=["a", "b", "a", "b"],
                 rho=0.25)
    assert V[0, 1] / np.sqrt(vi[0] * vi[1]) == pytest.approx(0.25, rel=EXACT)


def test_vcalc_with_both_type_and_obs_takes_two_correlations():
    vi = [0.04, 0.05, 0.06, 0.07]
    V = ma.vcalc(vi, cluster=[1, 1, 1, 1], type=["a", "a", "b", "b"],
                 obs=[1, 2, 1, 2], rho=(0.6, 0.4))
    # same construct, two measures -> the within-construct correlation
    assert V[0, 1] / np.sqrt(vi[0] * vi[1]) == pytest.approx(0.6, rel=EXACT)
    assert V[2, 3] / np.sqrt(vi[2] * vi[3]) == pytest.approx(0.6, rel=EXACT)
    # different constructs -> the between-construct correlation
    assert V[0, 2] / np.sqrt(vi[0] * vi[2]) == pytest.approx(0.4, rel=EXACT)


# --------------------------------------------------------------------------- #
# vcalc — the one behaviour not copied from metafor
# --------------------------------------------------------------------------- #

def test_vcalc_honours_rho_when_obs_is_missing():
    """metafor discards it and returns 1; pimeta applies what was asked."""
    vi = [0.04, 0.05]
    V = ma.vcalc(vi, cluster=[1, 1], rho=0.3)
    assert V[0, 1] / np.sqrt(vi[0] * vi[1]) == pytest.approx(0.3, rel=EXACT)


def test_vcalc_refuses_to_invent_a_correlation():
    with pytest.raises(ValueError, match="rho"):
        ma.vcalc([0.04, 0.05], cluster=[1, 1])


def test_vcalc_r_compat_reproduces_metafor_silence():
    """The opt-out keeps metafor's behaviour reachable: rho dropped, corr 1."""
    vi = [0.04, 0.05]
    V = ma.vcalc(vi, cluster=[1, 1], rho=0.3, r_compat=True)
    assert V[0, 1] / np.sqrt(vi[0] * vi[1]) == pytest.approx(1.0, rel=EXACT)
    # ... and the same when no rho was given at all
    V = ma.vcalc(vi, cluster=[1, 1], r_compat=True)
    assert V[0, 1] / np.sqrt(vi[0] * vi[1]) == pytest.approx(1.0, rel=EXACT)


def test_vcalc_rejects_a_correlation_outside_minus_one_to_one():
    with pytest.raises(ValueError, match=r"\[-1, 1\]"):
        ma.vcalc([0.04, 0.05], cluster=[1, 1], obs=[1, 2], rho=1.5)


# --------------------------------------------------------------------------- #
# vcalc — within-subject designs (time1 / time2 with phi)
# --------------------------------------------------------------------------- #

# Ishak et al. (2007) is the dataset metafor's own `vcalc` example uses, with
# the AR(1) autocorrelation of 0.97 that its help page estimates from the data.
# These five rows are Barichella (measured at times 1 and 3) and Burchiel
# (times 1, 2 and 3); the four entries are what R returns for them.
R_ISHAK_PHI = 0.97
R_ISHAK_VI = [7.3, 5.7, 8.0, 8.0, 5.0]
R_ISHAK_TIME = [1.0, 3.0, 1.0, 2.0, 3.0]
R_ISHAK_STUDY = ["Barichella (2003)", "Barichella (2003)",
                 "Burchiel (1999)", "Burchiel (1999)", "Burchiel (1999)"]
R_ISHAK_V = {(0, 1): 6.0693520102314036,
             (2, 3): 7.7599999999999989,
             (2, 4): 5.9507741009048551,
             (3, 4): 6.1348186607266548}


def test_vcalc_within_subject_recovers_metafor_ar1_matrix():
    """R: ``vcalc(vi, cluster=study, time1=time, phi=0.97)``."""
    V = ma.vcalc(np.array(R_ISHAK_VI), cluster=R_ISHAK_STUDY,
                 time1=R_ISHAK_TIME, phi=R_ISHAK_PHI)
    for (i, j), want in R_ISHAK_V.items():
        assert V[i, j] == pytest.approx(want, rel=EXACT)
        assert V[j, i] == pytest.approx(want, rel=EXACT)
    # different trials are independent, and the diagonal is still vi
    assert V[0, 2] == 0.0
    assert np.allclose(np.diag(V), R_ISHAK_VI, rtol=EXACT)


def test_vcalc_within_subject_correlation_is_phi_to_the_time_gap():
    vi = [0.1, 0.1, 0.1]
    V = ma.vcalc(vi, cluster=[1, 1, 1], time1=[1.0, 2.0, 4.0], phi=0.5)
    corr = V / np.sqrt(np.outer(vi, vi))
    assert corr[0, 1] == pytest.approx(0.5, rel=EXACT)       # one unit
    assert corr[1, 2] == pytest.approx(0.25, rel=EXACT)      # two units
    assert corr[0, 2] == pytest.approx(0.125, rel=EXACT)     # three units


def test_vcalc_multiplies_the_outcome_and_arm_factors():
    """metafor's entry is the outcome correlation *times* the arm correlation."""
    vi = [0.04, 0.05]
    V = ma.vcalc(vi, cluster=[1, 1], obs=[1, 2], rho=0.6,
                 time1=[1.0, 2.0], phi=0.7)
    # 'obs' says 0.6 and the one-unit time gap says 0.7
    assert V[0, 1] / np.sqrt(vi[0] * vi[1]) == pytest.approx(0.42, rel=EXACT)


# Every argument at once, and the correlations R returns for that design.
R_ALL_ARGS_CORR = np.array([
    [1.0, 0.42000000000000000, 0.14599927901768631, 0.09961174629530393],
    [0.42000000000000004, 1.0, 0.10219949531238040, 0.14230249470757705],
    [0.14599927901768631, 0.10219949531238039, 1.0, 0.41988149502066152],
    [0.09961174629530393, 0.14230249470757705, 0.41988149502066158, 1.0],
])


def test_vcalc_every_argument_at_once_matches_metafor():
    """Two constructs, two measures, three arms, two time points, four weights."""
    vi = [0.10, 0.12, 0.15, 0.16]
    V = ma.vcalc(vi, cluster=["A"] * 4, type=[1, 1, 2, 2], obs=[1, 2, 1, 2],
                 grp1=[1, 1, 1, 1], grp2=[2, 2, 3, 3],
                 time1=[1.0, 2.0, 1.0, 2.0], time2=[1.0, 2.0, 1.0, 2.0],
                 w1=[20, 20, 20, 22], w2=[20, 20, 18, 18],
                 rho=(0.6, 0.3), phi=0.7)
    corr = V / np.sqrt(np.outer(vi, vi))
    assert corr == pytest.approx(R_ALL_ARGS_CORR, abs=EXACT)


def test_vcalc_requires_phi_with_a_time_index():
    with pytest.raises(ValueError, match="phi"):
        ma.vcalc([0.04, 0.05], cluster=[1, 1], time1=[1.0, 2.0])


def test_vcalc_requires_phi_to_lie_in_the_unit_interval():
    with pytest.raises(ValueError, match=r"\[-1, 1\]"):
        ma.vcalc([0.04, 0.05], cluster=[1, 1], time1=[1.0, 2.0], phi=1.5)


def test_vcalc_needs_the_second_condition_for_a_second_time():
    with pytest.raises(ValueError, match="grp2"):
        ma.vcalc([0.04, 0.05], cluster=[1, 1], time1=[1.0, 2.0],
                 time2=[1.0, 1.0], phi=0.5)


def test_vcalc_warns_when_phi_makes_a_block_singular():
    """R's ``checkpd`` warning; phi = 1 correlates every estimate with every
    other one inside a cluster."""
    with pytest.warns(RuntimeWarning, match="not positive definite"):
        ma.vcalc([0.04, 0.05, 0.06], cluster=[1, 1, 1],
                 time1=[1.0, 2.0, 3.0], phi=1.0)


def test_vcalc_can_skip_the_positive_definiteness_check():
    V = ma.vcalc([0.04, 0.05, 0.06], cluster=[1, 1, 1],
                 time1=[1.0, 2.0, 3.0], phi=1.0, check_pd=False)
    # phi = 1 makes the correlation exactly 1, which is what the check warns
    # about and does not prevent
    assert V[0, 1] == pytest.approx(np.sqrt(0.04 * 0.05), rel=EXACT)


# --------------------------------------------------------------------------- #
# tau_cor
# --------------------------------------------------------------------------- #

def test_tau_cor_matches_metafor_G_and_cov2cor():
    tc = ma.tau_cor(_fit())
    assert tc.names == ["AL", "PD"]
    assert tc.tau2 == pytest.approx(R_TAU2, abs=FITTED)
    assert np.asarray(tc.G) == pytest.approx(np.array(R_G), abs=FITTED)
    assert tc.R[0, 1] == pytest.approx(R_RHO, abs=2e-5)
    assert tc.R[0, 0] == 1.0 and tc.R[1, 1] == 1.0
    assert np.allclose(tc.R, tc.R.T)


def test_tau_cor_reports_rho_in_metafor_order():
    tc = ma.tau_cor(_fit())
    assert list(tc.rho) == pytest.approx([tc.R[1, 0]], abs=EXACT)


def test_tau_cor_table_and_str():
    tc = ma.tau_cor(_fit())
    rows = tc.coef_table()
    assert [r.get("outcome") for r in rows[:2]] == ["AL", "PD"]
    assert rows[2]["rho"] == pytest.approx(tc.R[0, 1], abs=EXACT)
    text = str(tc)
    assert "between-outcome correlations" in text
    assert "AL" in text and "PD" in text


def test_tau_cor_needs_a_multivariate_random_term():
    yi = [0.1, 0.2, 0.3, 0.4]
    vi = [0.01, 0.02, 0.03, 0.04]
    fit = ma.rma_mv(yi, vi=vi, random="~ 1 | study",
                    data={"study": [1, 1, 2, 2]})
    with pytest.raises(ValueError, match="multivariate"):
        ma.tau_cor(fit)


# --------------------------------------------------------------------------- #
# rho_sensitivity
# --------------------------------------------------------------------------- #

def test_rho_sensitivity_reproduces_the_metafor_refits():
    res = ma.rho_sensitivity(np.array(YI), np.array(VI),
                             study=np.array(STUDY), outcome=np.array(OUTCOME),
                             mods="~ outcome - 1", rho_grid=R_GRID_RHO)
    assert list(res.rho) == R_GRID_RHO
    assert res.names == ["outcome[AL]", "outcome[PD]"]
    assert np.asarray(res.estimate[:, 0]) == pytest.approx(R_GRID_AL,
                                                           abs=FITTED)
    assert np.asarray(res.estimate[:, 1]) == pytest.approx(R_GRID_PD,
                                                           abs=FITTED)
    assert np.asarray(res.se[:, 0]) == pytest.approx(R_GRID_SE_AL, abs=FITTED)
    assert np.asarray(res.se[:, 1]) == pytest.approx(R_GRID_SE_PD, abs=FITTED)


def test_rho_sensitivity_intervals_widen_with_rho():
    """A positive within-study correlation downweights the larger trials."""
    res = ma.rho_sensitivity(np.array(YI), np.array(VI),
                             study=np.array(STUDY), outcome=np.array(OUTCOME),
                             mods="~ outcome - 1")
    width = np.asarray(res.ci_ub) - np.asarray(res.ci_lb)
    assert np.all(np.diff(width[:, 0]) > 0)
    assert np.all(np.diff(width[:, 1]) > 0)
def test_rho_sensitivity_reports_the_spread_of_a_coefficient():
    res = ma.rho_sensitivity(np.array(YI), np.array(VI),
                             study=np.array(STUDY), outcome=np.array(OUTCOME),
                             mods="~ outcome - 1", rho_grid=(0.0, 0.8))
    summary = res.range_of("outcome[PD]")
    assert summary["coefficient"] == "outcome[PD]"
    assert summary["estimate_spread"] == pytest.approx(
        abs(R_GRID_PD[-1] - R_GRID_PD[0]), abs=FITTED)
    assert summary["sign_stable"] is True


def test_rho_sensitivity_accepts_a_design_matrix_and_a_plain_model():
    X = np.column_stack([np.ones(10), [1.0, 0.0] * 5])
    res = ma.rho_sensitivity(np.array(YI), np.array(VI),
                             study=np.array(STUDY), outcome=np.array(OUTCOME),
                             mods=X, rho_grid=(0.4,))
    assert res.estimate.shape == (1, 2)
    base = ma.rho_sensitivity(np.array(YI), np.array(VI),
                              study=np.array(STUDY), outcome=np.array(OUTCOME),
                              rho_grid=(0.4,))
    assert base.names == ["intercept"]


def test_rho_sensitivity_rejects_an_invalid_grid():
    with pytest.raises(ValueError, match="grid"):
        ma.rho_sensitivity(np.array(YI), np.array(VI), study=np.array(STUDY),
                           outcome=np.array(OUTCOME), rho_grid=())


# --------------------------------------------------------------------------- #
# the record round-trip the package requires of every new model
# --------------------------------------------------------------------------- #

def test_tau_cor_round_trips_through_a_record(tmp_path):
    from pimeta.record import load_record, save_record

    tc = ma.tau_cor(_fit())
    path = tmp_path / "taucor.json"
    save_record(path, tc, mode="digest", function="tau_cor")
    loaded = load_record(path)
    assert loaded.kind == "tau_cor"
    assert np.asarray(loaded.results["values"]["R"]) == pytest.approx(
        np.asarray(tc.R), rel=EXACT)
    assert np.asarray(loaded.results["values"]["tau2"]) == pytest.approx(
        np.asarray(tc.tau2), rel=EXACT)


def test_rho_sensitivity_round_trips_through_a_record(tmp_path):
    from pimeta.record import load_record, save_record

    res = ma.rho_sensitivity(np.array(YI), np.array(VI),
                             study=np.array(STUDY), outcome=np.array(OUTCOME),
                             mods="~ outcome - 1", rho_grid=(0.0, 0.5))
    path = tmp_path / "rhosens.json"
    save_record(path, res, mode="digest", function="rho_sensitivity")
    loaded = load_record(path)
    assert loaded.kind == "rho_sensitivity"
    assert np.asarray(loaded.results["values"]["rho"]) == pytest.approx(
        [0.0, 0.5], rel=EXACT)
    assert np.asarray(loaded.results["values"]["estimate"]) == pytest.approx(
        np.asarray(res.estimate), rel=EXACT)


# -- the correlation cone: box bounds do not imply positive definiteness ----- #

def test_the_un_cone_check_distinguishes_valid_from_indefinite():
    """Two correlations at 1 and a third negative is not a correlation matrix."""
    from pimeta.rma_mv import _UNComp

    comp = _UNComp(np.array([0, 1, 2]), np.array([0, 0, 0]), 3)
    # parameters: tau2 (3), then rho in column-major lower-triangular order
    good = np.array([0.1, 0.1, 0.1, 0.5, 0.5, 0.0])
    bad = np.array([0.1, 0.1, 0.1, 1.0 - 1e-9, 1.0 - 1e-9, -0.2])
    assert comp.is_valid(good) is True
    assert comp.is_valid(bad) is False


def test_the_cholesky_vector_round_trips_through_tau2_and_rho():
    """The parametrisation the free path optimises over is a bijection."""
    from pimeta.rma_mv import _UNComp

    comp = _UNComp(np.array([0, 1, 2]), np.array([0, 0, 0]), 3)
    p = np.array([0.04, 0.09, 0.16, 0.6, -0.3, 0.2])
    l = comp.to_chol_vec(p)
    assert l.shape == (6,)
    back = comp.from_chol_vec(l)
    assert back == pytest.approx(p, abs=1e-9)


def test_an_invalid_point_is_snapped_into_the_cone_for_the_conversion():
    """``to_chol_vec`` has to work for any initial value, valid or not."""
    from pimeta.rma_mv import _UNComp

    comp = _UNComp(np.array([0, 1, 2]), np.array([0, 0, 0]), 3)
    bad = np.array([0.1, 0.1, 0.1, 0.99, 0.99, -0.5])
    l = comp.to_chol_vec(bad)
    back = comp.from_chol_vec(l)
    assert comp.is_valid(back) is True
