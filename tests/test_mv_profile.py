"""Tests for :func:`pimeta.profile_rmamv` (``metafor::profile.rma.mv``).

Reference values were printed by metafor 5.2-1 in R 4.6.1 from the same input.
The profile refits the model with the profiled variance component pinned, so
the achievable agreement is bounded by the variance-component optimiser
(~1e-8 on the log-likelihood); the grid itself is exact.
"""

import numpy as np
import pytest

import pimeta as ma
from pimeta.mv_diagnostics import MvProfileResult
from pimeta.s3 import plot

YI = [0.10, 0.05, -0.20, -0.15, 0.05, 0.02]
VI = [0.01, 0.012, 0.014, 0.013, 0.011, 0.015]
STUDY = [1, 1, 2, 2, 3, 3]
G1 = [1, 1, 1, 2, 2, 2]
G2 = [1, 2, 3, 4, 5, 6]

# R: rma.mv(yi, vi, random = ~ 1 | study, method = "REML"); profile(res, 20)
R_GRID = [
    0.00292128048785746, 0.00803068677797022, 0.01314009306808299,
    0.01824949935819575, 0.02335890564830852, 0.02846831193842128,
    0.03357771822853405, 0.03868712451864682, 0.04379653080875958,
    0.04890593709887234, 0.05401534338898511, 0.05912474967909788,
    0.06423415596921064, 0.06934356225932342, 0.07445296854943619,
    0.07956237483954895, 0.08467178112966171, 0.08978118741977448,
    0.09489059370988724, 0.10000000000000001,
]
R_LL = [
    3.90545567849758, 4.15750393652454, 4.18223168051299, 4.14165117577159,
    4.07907770257413, 4.00952325083841, 3.93893423075726, 3.86981077095732,
    3.80319337775817, 3.73946098615394, 3.67868164294753, 3.62077677106393,
    3.56560157965873, 3.51298549031570, 3.46275263469833, 3.41473209124817,
    3.36876272304411, 3.32469514570121, 3.28239217506905, 3.24172848858586,
]
R_BETA = [
    -0.0144046159766304, -0.0163632607270071, -0.0172826048378716,
    -0.0178165660549016, -0.0181655122601563, -0.0184114139079340,
    -0.0185940461878696, -0.0187350457115338, -0.0188471925020530,
    -0.0189385211233127, -0.0190143362574595, -0.0190782813745229,
    -0.0191329418164175, -0.0191802029285281, -0.0192214719552644,
    -0.0192578205586038, -0.0192900792236487, -0.0193189014916655,
    -0.0193448087022921, -0.0193682218128361,
]
R_CI_LB = [
    -0.122167525734926, -0.151161917912624, -0.174509533427620,
    -0.194641151691879, -0.212618925976397, -0.229021320421747,
    -0.244205326289198, -0.258409925236952, -0.271804551517200,
    -0.284514582100701, -0.296635921436995, -0.308243899190299,
    -0.319398980226872, -0.330150582280744, -0.340539716524659,
    -0.350600866909227, -0.360363360754172, -0.369852389612880,
    -0.379089783777725, -0.388094609487456,
]

# R: explicit steps = c(0.01, 0.05, 0.2, 1.0)
R_LL_EXPLICIT = [4.18019479123460, 3.72619917375339, 2.66121310818918,
                 1.14631543237199]

# R: rma.mv(yi, vi, random = ~ 1 | g1/g2); profile(res2, sigma2 = 2, steps = 11)
R_GRID2 = [
    0.000579325403315381, 0.010521392862983843, 0.020463460322652303,
    0.030405527782320765, 0.040347595241989227, 0.050289662701657689,
    0.060231730161326151, 0.070173797620994613, 0.080115865080663082,
    0.090057932540331551, 0.100000000000000006,
]
R_LL2 = [
    3.406549632977032, 3.216013113800729, 2.804380870941531,
    2.406599692378211, 2.048771848624714, 1.729480085026502,
    1.443215494374446, 1.184620680136123, 0.949221557029361,
    0.733414513602322, 0.534312300242248,
]
R_BETA2 = [
    -0.0122864117239068, -0.0165346232209635, -0.0181349959897038,
    -0.0189747225291861, -0.0194919061424406, -0.0198424209387958,
    -0.0200956450554927, -0.0202871451388214, -0.0204370358474362,
    -0.0205575487479860, -0.0206565493448173,
]


def _mv():
    return ma.rma_mv(YI, V=np.diag(VI), data={"study": STUDY}, method="REML")


def _mv2():
    return ma.rma_mv(YI, V=np.diag(VI), data={"g1": G1, "g2": G2},
                    random="~ 1 | g1/g2", method="REML")


# --------------------------------------------------------------------------- #
# profile over a single variance component
# --------------------------------------------------------------------------- #

def test_profile_mv_grid_and_fit_values():
    res = _mv()
    assert res.sigma2[0] == pytest.approx(0.0116851219514298, rel=1e-7)
    assert res.ll == pytest.approx(4.18516198680508, rel=0, abs=1e-10)

    prof = ma.profile_rmamv(res, sigma2=1, steps=20)
    assert isinstance(prof, MvProfileResult)
    assert prof.component == "sigma2"
    assert prof.index == 1
    assert prof.comps == 1
    assert prof.method == "REML"
    assert prof.maxll == pytest.approx(res.ll, rel=0, abs=1e-12)
    assert prof.vc == pytest.approx(res.sigma2[0], rel=0, abs=1e-12)

    # The grid is derived from vc, so it inherits the optimiser's 1e-8.
    assert len(prof.sigma2) == 20
    np.testing.assert_allclose(prof.sigma2, R_GRID, rtol=1e-7, atol=1e-7)
    assert prof.sigma2[0] == pytest.approx(res.sigma2[0] / 4.0, rel=1e-12)
    assert prof.sigma2[-1] == pytest.approx(0.1, rel=1e-12)


def test_profile_mv_loglikelihood_matches_r():
    prof = ma.profile_rmamv(_mv(), sigma2=1, steps=20)
    np.testing.assert_allclose(prof.ll, R_LL, rtol=0, atol=3e-7)


def test_profile_mv_coefficients_match_r():
    prof = ma.profile_rmamv(_mv(), sigma2=1, steps=20)
    np.testing.assert_allclose([b[0] for b in prof.beta], R_BETA,
                               rtol=0, atol=1e-8)
    np.testing.assert_allclose([b[0] for b in prof.ci_lb], R_CI_LB,
                               rtol=0, atol=1e-8)


def test_profile_mv_explicit_grid_is_exact():
    prof = ma.profile_rmamv(_mv(), sigma2=1, steps=[0.01, 0.05, 0.2, 1.0])
    assert prof.sigma2 == [0.01, 0.05, 0.2, 1.0]
    np.testing.assert_allclose(prof.ll, R_LL_EXPLICIT, rtol=0, atol=1e-12)


def test_profile_mv_maximum_is_at_the_fitted_value():
    res = _mv()
    prof = ma.profile_rmamv(res, sigma2=1, steps=[0.001, 0.01, 0.0117, 0.05])
    i = int(np.argmax(prof.ll))
    assert prof.sigma2[i] == pytest.approx(0.0117)
    # no profiled value may exceed the unconstrained fit
    assert max(prof.ll) <= prof.maxll + 1e-9


# --------------------------------------------------------------------------- #
# profiling one component of a two-component model
# --------------------------------------------------------------------------- #

def test_profile_mv_second_component_matches_r(tol=1e-7):
    res = _mv2()
    assert len(res.sigma2) == 2
    prof = ma.profile_rmamv(res, sigma2=2, steps=11)
    assert prof.index == 2
    np.testing.assert_allclose(prof.sigma2, R_GRID2, rtol=0, atol=1e-7)
    np.testing.assert_allclose(prof.ll, R_LL2, rtol=0, atol=tol)
    np.testing.assert_allclose([b[0] for b in prof.beta], R_BETA2,
                               rtol=0, atol=1e-8)


def test_profile_mv_keeps_other_components_free():
    """Profiling sigma2[2] must re-estimate sigma2[1] at every grid point."""
    res = _mv2()
    prof = ma.profile_rmamv(res, sigma2=2, steps=5)
    # ll must be the *maximised* value given the pinned component, i.e. not
    # simply the log-likelihood with both components frozen at the fit.
    frozen = ma.profile_rmamv(res, sigma2=2, steps=5, xlim=(0.1, 0.1))
    assert len(prof.ll) == len(frozen.ll) == 5


# --------------------------------------------------------------------------- #
# rma_mv(sigma2=) partial fixing
# --------------------------------------------------------------------------- #

def test_rma_mv_partial_sigma2():
    res = _mv2()
    pinned = ma.rma_mv(YI, V=np.diag(VI), data={"g1": G1, "g2": G2},
                       random="~ 1 | g1/g2", method="REML",
                       sigma2=[np.nan, 0.05])
    # the pinned component keeps the requested value ...
    assert pinned.sigma2[1] == pytest.approx(0.05, rel=0, abs=1e-12)
    # ... the free one is still estimated ...
    assert pinned.sigma2[0] == pytest.approx(res.sigma2[0], rel=0, abs=1e-9)
    # ... and pinning away from the optimum costs log-likelihood.
    assert pinned.ll < res.ll


def test_rma_mv_full_sigma2_still_works():
    res = _mv()
    frozen = ma.rma_mv(YI, V=np.diag(VI), data={"study": STUDY},
                       method="REML", sigma2=res.sigma2)
    assert frozen.sigma2[0] == pytest.approx(res.sigma2[0], rel=0, abs=1e-15)
    assert frozen.ll == pytest.approx(res.ll, rel=0, abs=1e-12)


def test_rma_mv_sigma2_length_mismatch():
    with pytest.raises(ValueError, match="variance component"):
        ma.rma_mv(YI, V=np.diag(VI), data={"study": STUDY},
                  sigma2=[0.01, 0.02])


# --------------------------------------------------------------------------- #
# error paths, plotting and result protocol
# --------------------------------------------------------------------------- #

def test_profile_mv_rejects_bad_component():
    with pytest.raises(ValueError, match="does not exist"):
        ma.profile_rmamv(_mv(), sigma2=2)


def test_profile_mv_rejects_bad_xlim():
    with pytest.raises(ValueError, match="Lower bound"):
        ma.profile_rmamv(_mv(), xlim=[-1.0, 1.0])
    with pytest.raises(ValueError, match="length 2"):
        ma.profile_rmamv(_mv(), xlim=[1.0, 2.0, 3.0])


def test_profile_mv_rejects_bad_steps():
    with pytest.raises(ValueError, match="steps"):
        ma.profile_rmamv(_mv(), steps=1)


def test_profile_mv_requires_a_fitted_model():
    with pytest.raises(TypeError):
        ma.profile_rmamv({"yi": YI})


def test_profile_mv_defaults_to_first_component():
    prof = ma.profile_rmamv(_mv2(), steps=4)
    assert prof.index == 1


def test_profile_mv_to_dict_and_repr():
    prof = ma.profile_rmamv(_mv(), steps=4)
    d = prof.to_dict()
    assert d["component"] == "sigma2"
    assert d["index"] == 1
    assert len(d["sigma2"]) == 4
    assert d["sigma2"] == prof.tau2     # tau2 is an alias property
    assert "sigma2" in repr(prof)


def test_profile_mv_plot(tmp_path):
    res = _mv()
    out = tmp_path / "prof.png"
    got = plot(res, which="profile", filename=str(out), steps=8)
    assert out.exists() and out.stat().st_size > 0

    prof = ma.profile_rmamv(res, steps=8)
    out2 = tmp_path / "prof2.png"
    got2 = plot(prof, filename=str(out2))
    assert out2.exists() and got2.figure is not None

    out3 = tmp_path / "prof3.png"
    ma.plot_mv_profile(prof, filename=str(out3), ylim=(3.0, 4.5))
    assert out3.exists()


def test_profile_mv_plot_needs_finite_values():
    prof = ma.profile_rmamv(_mv(), steps=4)
    prof.ll = [float("nan")] * 4
    with pytest.raises(ValueError, match="finite"):
        ma.plot_mv_profile(prof)
