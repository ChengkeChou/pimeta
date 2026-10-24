"""Profile-likelihood intervals for an ``rma_mv`` variance component.

Checked against ``metafor::confint`` on the Berkey 1998 fit that
``tests/test_mv.py`` already uses, so the model itself is pinned there and this
file only has to pin the inference on top of it.  R 4.6.1, metafor 5.2.1.

What is compared, and at what tolerance
---------------------------------------
The *profile* is the strong claim: fix ``rho`` at a value, re-estimate the two
``tau2`` components, and read the likelihood.  At the seven values where
metafor can compute it, the two agree to 1e-10 or better -- so the objective and
the refit are the same function, not a similar one.

The *endpoints* inherit the optimiser's tolerance, and metafor finds them with
``uniroot`` on its own profile, so they agree to about 1e-5 rather than to 1e-12.
The tolerance below says so.

Two endpoints are deliberately different, and the tests say which and why:

- ``rho`` at 99%: metafor reports -0.8354, this module reports the bound.  A
  fit in which one between-study variance is zero has likelihood 1.3813 at
  *every* correlation -- the correlation is then not a parameter of the model --
  and 1.3813 is above the 99% threshold of 0.2755.  An interval that excluded
  it would exclude points with a *higher* profile likelihood.  metafor's
  optimiser never leaves the interior branch, which is also why its profile
  returns ``NA`` just beyond.  ``r_compat=True`` reproduces it.
- ``tau2[2]`` lower: metafor reports 1.14e-4, where its own profile is 1.7551,
  above the 1.6722 threshold -- so that number is the last point its search
  could fit, not the crossing.  Near a zero variance both implementations are
  numerically fragile; the test asserts the agreement of the profile there and
  only a loose bound on the endpoint.
"""

import numpy as np
import pytest

import pimeta as ma
from pimeta.data_compat import design_matrix
from pimeta.mv.confint import _parameter_slots, _profile_loglik

EXACT = 1e-12
FITTED = 1e-4        # metafor's own uniroot tolerance is 1.2e-4

# --- Berkey 1998, the same fit as tests/test_mv.py -------------------------
YI = [0.47, -0.32, 0.20, -0.60, 0.40, -0.12, 0.26, -0.31, 0.56, -0.39]
VI = [0.0075, 0.0077, 0.0057, 0.0008, 0.0021,
      0.0014, 0.0029, 0.0015, 0.0148, 0.0304]
STUDY = [1, 1, 2, 2, 3, 3, 4, 4, 5, 5]
OUTCOME = ["PD", "AL"] * 5

# metafor: confint(rma.mv(...), rho = 1) / confint(..., tau2 = i)
R_RHO_CI95_LB = -0.62738012345164995
R_RHO_CI90_LB = -0.47087851425494598
R_RHO_CI99_LB = -0.83541493325287697        # see the module docstring
R_TAU2_1_LB = 0.0094965463366053493
R_TAU2_1_UB = 0.20826909353118001
R_TAU2_2_LB = 0.000114155555519552           # see the module docstring
R_TAU2_2_UB = 0.118651077618005

# metafor: profile(res, rho = 1, xlim = c(v, v + 1e-7), steps = 2)$ll
R_PROFILE_RHO = [
    (-0.7, 1.3104115672929302),
    (-0.5, 2.1501039792277314),
    (-0.3, 2.6741059321273104),
    (0.0, 3.1865565399910567),
    (0.2, 3.4125264533493858),
    (0.8, 3.3554218771098725),
    (0.9999, 2.4019481400443361),
]


def _fit():
    data = {"study": STUDY, "outcome": OUTCOME,
            "yi": np.array(YI), "vi": np.array(VI)}
    V = ma.vcalc(np.array(VI), cluster=data["study"], obs=data["outcome"],
                 rho=0.5)
    X, _ = design_matrix("~ outcome - 1", data)
    return ma.rma_mv(np.array(YI), V=V, mods=X, data=data,
                     random="~ outcome | study", struct="UN",
                     intercept=False)


@pytest.fixture(scope="module")
def fit():
    return _fit()


def test_the_profile_is_metafor_profile(fit):
    """The refit reproduces metafor's profile likelihood, point by point."""
    slots = _parameter_slots(fit)
    bounds = [b for _, b in slots]
    index = [i for i, (name, _) in enumerate(slots) if name == "rho"][0]
    for value, expected in R_PROFILE_RHO:
        got = _profile_loglik(fit, index, value, bounds, {}, r_compat=True)
        assert got == pytest.approx(expected, abs=1e-10), value


def test_rho_interval_matches_metafor(fit):
    out = ma.confint_rma_mv(fit, rho=1)
    assert out.estimate == pytest.approx(fit.rho[0], rel=EXACT)
    assert out.ci_lb == pytest.approx(R_RHO_CI95_LB, abs=FITTED)
    # the profile is still above the threshold at rho = 1, so 1 *is* the bound
    assert out.ci_ub == pytest.approx(1.0, abs=1e-8)
    assert out.at_bound_ub and not out.at_bound_lb


def test_rho_interval_narrows_as_the_level_falls(fit):
    ninety = ma.confint_rma_mv(fit, rho=1, level=90)
    assert ninety.ci_lb == pytest.approx(R_RHO_CI90_LB, abs=FITTED)
    assert ninety.ci_lb > ma.confint_rma_mv(fit, rho=1).ci_lb


def test_tau2_intervals_match_metafor(fit):
    first = ma.confint_rma_mv(fit, tau2=1)
    assert first.ci_lb == pytest.approx(R_TAU2_1_LB, abs=FITTED)
    assert first.ci_ub == pytest.approx(R_TAU2_1_UB, abs=FITTED)

    second = ma.confint_rma_mv(fit, tau2=2)
    assert second.ci_ub == pytest.approx(R_TAU2_2_UB, abs=FITTED)
    # near a zero variance the crossing is unstable in both implementations
    assert 0.0 <= second.ci_lb < 3e-4


def test_the_interval_is_asymmetric(fit):
    """A profile interval is not the estimate +/- 1.96 se, and must not be."""
    out = ma.confint_rma_mv(fit, tau2=2)
    below = abs(out.estimate - out.ci_lb)
    above = abs(out.ci_ub - out.estimate)
    # a variance interval is right-skewed: it reaches much further up than down
    assert above > 5 * below


def test_the_supremum_includes_the_zero_variance_fit(fit):
    """metafor stops on the interior branch and reports -0.8354; we do not.

    One between-study variance can collapse to zero, and then the correlation
    is not a parameter of the model: the likelihood there is 1.3813 whatever
    ``rho`` is, which is above the 99% threshold of 0.2755.  An interval that
    excluded it would exclude points with a *higher* profile likelihood, so
    the honest answer at this level is the whole range.
    """
    out = ma.confint_rma_mv(fit, rho=1, level=99)
    assert out.at_bound_lb
    assert out.ci_lb == pytest.approx(-1.0, abs=1e-6)
    slots = _parameter_slots(fit)
    bounds = [b for _, b in slots]
    index = [i for i, (name, _) in enumerate(slots) if name == "rho"][0]
    ll = _profile_loglik(fit, index, -0.9, bounds, {}, r_compat=True)
    assert ll > out.ll_max - out.crit          # -0.9 is inside the interval
    # and the interior branch is what metafor reports for the levels below
    assert ma.confint_rma_mv(fit, rho=1).ci_lb == pytest.approx(
        R_RHO_CI95_LB, abs=FITTED)


def test_a_level_can_be_given_three_ways(fit):
    a = ma.confint_rma_mv(fit, rho=1, level=95)
    b = ma.confint_rma_mv(fit, rho=1, level=0.95)
    c = ma.confint_rma_mv(fit, rho=1, level=0.05)
    assert a.ci_lb == b.ci_lb == c.ci_lb
    assert a.level == b.level == c.level == 95.0


def test_the_model_level_is_the_default(fit):
    assert ma.confint_rma_mv(fit, rho=1).level == fit.level


def test_only_one_component_at_a_time(fit):
    with pytest.raises(ValueError, match="exactly one"):
        ma.confint_rma_mv(fit, rho=1, tau2=1)
    with pytest.raises(ValueError, match="exactly one"):
        ma.confint_rma_mv(fit)


def test_an_index_out_of_range_is_rejected(fit):
    with pytest.raises(ValueError, match="1 component"):
        ma.confint_rma_mv(fit, rho=2)
    with pytest.raises(ValueError, match="2 component"):
        ma.confint_rma_mv(fit, tau2=3)


def test_the_curve_holds_every_evaluated_point(fit):
    out = ma.confint_rma_mv(fit, tau2=1)
    values = [v for v, _ in out.curve]
    assert values == sorted(values)
    assert values[0] <= out.ci_lb and values[-1] >= out.ci_ub
    # the estimate is the maximum of the curve
    assert max(ll for _, ll in out.curve) == pytest.approx(out.ll_max, abs=1e-12)
    # the endpoint is where the curve crosses the threshold
    assert out.ll_max - out.crit == pytest.approx(
        out.ll_max - 0.5 * 3.841458820694124, abs=1e-12)


def test_the_table_and_the_text(fit):
    out = ma.confint_rma_mv(fit, rho=1)
    row = out.to_table()[0]
    assert row["name"] == "rho"
    assert row["ci.lb"] == out.ci_lb
    text = str(out)
    assert "rho" in text and "0.5438" in text and "-0.6274" in text


def test_the_wrong_kind_of_object_is_rejected():
    with pytest.raises(TypeError, match="RmaMvResult"):
        ma.confint_rma_mv({"rho": 0.5}, rho=1)
