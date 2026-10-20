"""metafor's ``rma.mh`` / ``rma.peto`` methods, pinned to R 4.6.1 / metafor 5.2-1.

``cumul.rma.mh``, ``leave1out.rma.mh``, ``rstandard.rma.mh``,
``rstudent.rma.mh``, ``weights.rma.mh``, ``confint.rma.mh`` and
``qqnorm.rma.mh`` (and the seven ``rma.peto`` counterparts) all just refit the
closed-form estimator on subsets of the 2x2 tables, so they are transcribed
here and reachable through the normal generics.
"""

import numpy as np
import pytest

import pimeta as ma

AI = [10, 12, 8, 15, 9, 11]
BI = [90, 88, 92, 85, 91, 89]
CI = [20, 18, 22, 16, 21, 19]
DI = [80, 82, 78, 84, 79, 81]
SLAB = [f"study{i}" for i in range(1, 7)]


def _mh(measure="OR"):
    return ma.rma_mh(ai=AI, bi=BI, ci=CI, di=DI, measure=measure, slab=SLAB)


def _peto():
    return ma.rma_peto(ai=AI, bi=BI, ci=CI, di=DI, slab=SLAB)


# --------------------------------------------------------------------------- #
# the tables are retained on the result
# --------------------------------------------------------------------------- #

def test_results_keep_the_tables_and_labels():
    mh = _mh()
    assert mh.ai == pytest.approx(AI)
    assert mh.di == pytest.approx(DI)
    assert mh.slab == SLAB
    assert mh.correct is True
    assert _peto().slab == SLAB


def test_default_labels_are_generated():
    mh = ma.rma_mh(ai=AI, bi=BI, ci=CI, di=DI)
    assert mh.slab == [f"Study {i}" for i in range(1, 7)]


def test_slab_length_is_validated():
    with pytest.raises(ValueError, match="length of 'slab'"):
        ma.rma_mh(ai=AI, bi=BI, ci=CI, di=DI, slab=["a", "b"])


# --------------------------------------------------------------------------- #
# rstandard / rstudent
# --------------------------------------------------------------------------- #

def test_mh_rstandard_matches_r():
    z = ma.rstandard(_mh())["z"]
    assert z == pytest.approx([
        -0.321704535668, 0.498213639578, -1.134301107678,
        1.536066025094, -0.730167395134, 0.088442299703,
    ], abs=1e-11)


def test_mh_rstudent_matches_r():
    z = ma.rstudent(_mh())["z"]
    assert z == pytest.approx([
        -0.351950817177, 0.545829941532, -1.240345336933,
        1.693172703086, -0.798536421146, 0.096814287331,
    ], abs=1e-11)


def test_peto_rstandard_matches_r():
    out = ma.rstandard(_peto())
    # R's Peto residuals standardise by 1/sqrt(V), not by sqrt(V)
    assert out["se"] == pytest.approx(
        [0.395067628859] * 3 + [0.389791473268] + [0.395067628859] * 2,
        abs=1e-11)
    assert out["z"] == pytest.approx([
        -0.303893232844, 0.486242024875, -1.094028490562,
        1.499173588180, -0.698960861703, 0.091174396016,
    ], abs=1e-11)


def test_peto_rstudent_matches_r():
    z = ma.rstudent(_peto())["z"]
    assert z == pytest.approx([
        -0.332747924699, 0.532410752174, -1.197906601573,
        1.645988132502, -0.765327263136, 0.099831413737,
    ], abs=1e-11)


def test_residual_dicts_carry_resid_se_z_and_slab():
    out = ma.rstandard(_mh())
    assert set(out) == {"resid", "se", "z", "slab"}
    assert out["slab"] == SLAB
    assert out["z"] == pytest.approx(
        [r / s for r, s in zip(out["resid"], out["se"])], abs=1e-14)


# --------------------------------------------------------------------------- #
# weights
# --------------------------------------------------------------------------- #

def test_mh_weights_percentages_match_r():
    assert ma.weights(_mh(), pct=True) == pytest.approx([
        17.3577627772, 15.2748312440, 19.5178399229,
        13.1147540984, 18.4281581485, 16.3066538091,
    ], abs=1e-10)


def test_mh_weights_matrix_is_the_unnormalised_diagonal():
    w = ma.weights(_mh())
    assert np.diag(w) == pytest.approx([9.0, 7.92, 10.12, 6.8, 9.555, 8.455],
                                       abs=1e-12)
    assert np.allclose(w, np.diag(np.diag(w)))


def test_peto_weights_match_r():
    assert ma.weights(_peto(), pct=True) == pytest.approx(
        [16.5913009532] * 3 + [17.0434952341] + [16.5913009532] * 2,
        abs=1e-10)


def test_rr_and_rd_weights():
    # RR: (ci/Ni)(ai+bi); RD: ((ai+bi)/Ni)(ci+di)
    assert ma.weights(_mh("RR"), pct=True) == pytest.approx(
        [17.241379] * 1 + [15.517241, 18.965517, 13.793103, 18.103448,
                           16.37931], abs=1e-6)
    assert ma.weights(_mh("RD"), pct=True) == pytest.approx(
        [16.666667] * 6, abs=1e-6)


def test_weights_type_is_validated():
    with pytest.raises(ValueError, match="'type' must be"):
        ma.__dict__["weights"].dispatch(object)(_mh(), pct=False) \
            if False else __import__(
                "pimeta.rma_peto_mh", fromlist=["mh_weights"]
            ).mh_weights(_mh(), type="nope")


# --------------------------------------------------------------------------- #
# confint
# --------------------------------------------------------------------------- #

def test_mh_confint_matches_r():
    ci = ma.confint(_mh())
    assert ci["estimate"] == pytest.approx(-0.676886659688, abs=1e-11)
    assert ci["ci.lb"] == pytest.approx(-1.003953138882, abs=1e-11)
    assert ci["ci.ub"] == pytest.approx(-0.349820180495, abs=1e-11)


def test_peto_confint_matches_r():
    ci = ma.confint(_peto())
    assert ci["estimate"] == pytest.approx(-0.660333777937, abs=1e-11)
    assert ci["ci.lb"] == pytest.approx(-0.975732375162, abs=1e-11)
    assert ci["ci.ub"] == pytest.approx(-0.344935180712, abs=1e-11)


def test_confint_accepts_every_level_convention():
    mh = _mh()
    a = ma.confint(mh)
    b = ma.confint(mh, level=95)
    c = ma.confint(mh, level=0.05)
    for other in (b, c):
        assert other == pytest.approx(a, abs=1e-15)


def test_confint_back_transforms_ratio_measures():
    ci = ma.confint(_mh(), transf=True)
    assert ci["estimate"] == pytest.approx(np.exp(-0.676886659688), abs=1e-11)


# --------------------------------------------------------------------------- #
# leave1out / cumul
# --------------------------------------------------------------------------- #

def test_mh_leave1out_matches_r():
    out = ma.leave1out(_mh())
    assert out.estimate == pytest.approx([
        -0.650879323984, -0.717805648502, -0.585775747821,
        -0.809543830302, -0.618127662091, -0.684085926843,
    ], abs=1e-11)
    assert out.se == pytest.approx([
        0.182180412220, 0.183499369269, 0.181151078331,
        0.186189988278, 0.181629962435, 0.182803274744,
    ], abs=1e-12)
    assert out.QE == pytest.approx([
        4.39806477090, 4.26449016942, 2.97219907796,
        1.72532135326, 3.86867642871, 4.53407095958,
    ], abs=1e-10)
    assert out.I2 == pytest.approx([
        9.05090742486, 6.20215216625, 0.0, 0.0, 0.0, 11.77906045898,
    ], abs=1e-10)
    assert out.slab == [f"-study{i}" for i in range(1, 7)]


def test_peto_leave1out_matches_r():
    out = ma.leave1out(_peto())
    assert out.estimate == pytest.approx([
        -0.636452279730, -0.698545185070, -0.574359374391,
        -0.780392156863, -0.605405827060, -0.667498732400,
    ], abs=1e-11)
    assert out.se == pytest.approx(
        [0.176200035953] * 3 + [0.176679614768] + [0.176200035953] * 2,
        abs=1e-12)


def test_leave1out_with_clusters():
    out = ma.leave1out(_mh(), cluster=[1, 1, 2, 2, 3, 3])
    assert len(out.estimate) == 3
    assert out.slab == ["-1", "-2", "-3"]


def test_leave1out_rejects_a_bad_cluster():
    with pytest.raises(ValueError, match="does not match the length"):
        ma.leave1out(_mh(), cluster=[1, 2])


def test_leave1out_needs_more_than_one_study():
    one = ma.rma_mh(ai=[10], bi=[90], ci=[20], di=[80])
    with pytest.raises(ValueError, match="k = 1"):
        ma.leave1out(one)


def test_mh_cumul_matches_r():
    out = ma.cumul(_mh())
    assert out.k == [1, 2, 3, 4, 5, 6]
    assert out.estimate == pytest.approx([
        -0.810930216216, -0.640200407586, -0.809082811294,
        -0.612559067910, -0.684085926843, -0.676886659688,
    ], abs=1e-11)
    assert out.se == pytest.approx([
        0.416666666667, 0.288954699670, 0.240306153226,
        0.202740501316, 0.182803274744, 0.166873718994,
    ], abs=1e-12)
    assert out.h2 == pytest.approx([
        1.0, 0.333701312182, 0.689326060522,
        1.287181755960, 1.133517739896, 0.907762988110,
    ], abs=1e-11)
    assert out.q_p[0] == 1.0
    assert out.slab == SLAB


def test_peto_cumul_matches_r():
    out = ma.cumul(_peto())
    assert out.estimate == pytest.approx([
        -0.780392156863, -0.624313725490, -0.780392156863,
        -0.600710842787, -0.667498732400, -0.660333777937,
    ], abs=1e-11)


def test_cumul_honours_the_order_argument():
    out = ma.cumul(_mh(), order=[6, 5, 4, 3, 2, 1])
    assert out.estimate == pytest.approx([
        -0.640730921428, -0.810236398108, -0.550288204135,
        -0.695153199287, -0.650879323984, -0.676886659688,
    ], abs=1e-11)
    assert out.slab == ["study6", "study5", "study4", "study3", "study2",
                        "study1"]


def test_cumul_collapse_groups_equal_order_values():
    out = ma.cumul(_mh(), order=[1, 1, 2, 2, 3, 3], collapse=True)
    assert out.k == [2, 4, 6]
    assert out.slab == ["1", "2", "3"]


def test_cumul_rejects_a_bad_order():
    with pytest.raises(ValueError, match="Length of the 'order'"):
        ma.cumul(_mh(), order=[1, 2, 3])


# --------------------------------------------------------------------------- #
# qqnorm
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("type_", ["resid", "rstandard", "rstudent"])
@pytest.mark.parametrize("which", ["mh", "peto"])
def test_qqnorm_works_for_both_models(type_, which):
    obj = _mh() if which == "mh" else _peto()
    out = ma.qqnorm_rma(obj, type=type_)
    assert out.figure is not None


def test_qqnorm_rejects_a_bad_type():
    with pytest.raises(ValueError, match="'type' must be"):
        ma.qqnorm_rma(_mh(), type="nope")


# --------------------------------------------------------------------------- #
# the old call signatures still work
# --------------------------------------------------------------------------- #

def test_leave1out_and_rstudent_still_take_vectors():
    yi = [0.1, -0.2, 0.05, -0.15, 0.3, 0.25]
    vi = [0.01, 0.0144, 0.0121, 0.0169, 0.0196, 0.0225]
    out = ma.leave1out(yi, vi)
    assert len(out.estimate) == 6
    z = ma.rstudent(yi, vi)
    assert len(z) == 6


def test_cumul_still_takes_vectors():
    out = ma.cumul([0.1, 0.2, 0.3], [0.01, 0.02, 0.03])
    assert out.estimate[0] == pytest.approx(0.1)
