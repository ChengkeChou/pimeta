"""Dose-response network meta-analysis: equations, references, properties.

Three kinds of check, and they are not interchangeable.

**Equation level.**  A dose-response network's consistency equation must hold at
*every* dose, including doses nobody studied, and it must hold because of the
parameterisation rather than because the fit was lucky.  That is what the
circuit test below asserts: the same curve is used wherever a treatment appears,
so summing effects around a cycle cancels identically, and the delta-method
variance of a sum of contrasts must equal the variance of the direct contrast.

**Model level.**  Two independent implementations on the same data:

- ``metafor::rma.mv(yi, V, mods = ~ 0 + x, random = ~ 1 | study)``, for every
  dose function that is linear in its parameters.  ``rma.uni`` is *not* a
  reference here and the numbers show why: it treats a study's several contrasts
  as independent, so on this fixture it reports ``tau2 = 0.2216`` where the
  model with a study-level random effect gives ``0.3013``.  ``rma.mv`` gives
  ``0.3013134977`` against pimeta's ``0.3013134873``.
- ``nls(..., weights = 1/vi)`` with the PORT algorithm, for ``emax`` and
  ``exponential``, whose mean is not linear in the parameters.  ``nls``
  estimates a dispersion from the residuals; the comparison is against
  ``sqrt(diag(vcov())) / summary()$sigma``, because here the variances are
  known and no scale should be estimated.

**Property level.**  The fixture was simulated with known curves, and the
fitted curve's interval has to contain the truth at the doses studied.  That is
a weak statement on one dataset, which is why it is the third check and not the
first.
"""

from __future__ import annotations

import csv
import pathlib
import warnings

import numpy as np
import pytest

import pimeta as ma
from pimeta.basis import _rcs_matrix
from pimeta.nma.dose import dose_functions

DATA = pathlib.Path(__file__).parent / "data" / "nma_dose.csv"

# --------------------------------------------------------------------------- #
# reference values
# --------------------------------------------------------------------------- #
# metafor 5.2-1, rma.mv(..., random = ~ 1 | study, method = "REML")
R_TAU2 = {
    ("A", "linear"): 0.3013134977,
    ("A", "quadratic"): 0.0776049377,
    ("A", "loglinear"): 0.1185762741,
    ("A", "rcs"): 0.0744737323,
    ("B", "linear"): 0.0,
    ("B", "quadratic"): 0.0,
    ("B", "loglinear"): 0.0053122487,
    ("B", "rcs"): 0.0,
}
R_COEF = {
    ("A", "linear"): [0.0070301656],
    ("A", "quadratic"): [0.0216828713, -0.0000731446],
    ("A", "loglinear"): [0.2652932448],
    ("A", "rcs"): [0.0230416192, -0.0860155714, 0.1253910529],
    ("B", "linear"): [0.0145201945],
    ("B", "quadratic"): [0.0251142823, -0.0001710347],
    ("B", "loglinear"): [0.1628288599],
    ("B", "rcs"): [0.0131317822, 0.0873123169, -0.2282324038],
}
R_SE = {
    ("A", "linear"): [0.0011882175],
    ("A", "quadratic"): [0.0031986629, 0.0000164717],
    ("A", "loglinear"): [0.0301013745],
    ("A", "rcs"): [0.0045555484, 0.0515428958, 0.0915625965],
    ("B", "linear"): [0.0019402850],
    ("B", "quadratic"): [0.0055588138, 0.0000840991],
    ("B", "loglinear"): [0.0245997921],
    ("B", "rcs"): [0.0078515392, 0.0796044945, 0.1750182300],
}
KNOTS = {"A": [10.0, 40.0, 100.0, 200.0], "B": [5.0, 20.0, 40.0, 80.0]}

# nls(yi ~ f, weights = 1/vi, algorithm = "port"), same fixture
NLS_COEF = {
    "emax": [1.8700747346, 3.6970259849, 1.2826845249, 3.6222841059],
    "exponential": [1.5218168711, 3.7154209684, 0.9352075522, 3.4116001349],
}
NLS_SE_KNOWN_SCALE = {
    "emax": [0.2008950604, 0.3153700171, 0.5028628505, 0.8051227258],
    "exponential": [0.1111276220, 0.2072138087, 0.2519758118, 0.5632591237],
}
RSS = {"emax": 36.404448916970, "exponential": 33.941959705668}

# rms::rcs basis anchored at dose zero: knots 10/40/100/200, evaluated at
# 0, 25, 50, 200 (columns 2 and 3; column 1 is the dose itself)
RMS_BASIS = np.array([
    [0.0, 0.0, 0.0],
    [25.0, 0.0934903047, 0.0],
    [50.0, 1.7728531856, 0.0277008310],
    [200.0, 137.3684210526, 69.1412742382],
])

# the fixture's generating curves
TRUE_EMAX = {"A": (1.4, 30.0), "B": (0.9, 25.0)}


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def load() -> dict:
    with DATA.open(encoding="utf-8", newline="") as fh:
        rows = list(csv.DictReader(fh))
    return {
        "yi": np.array([float(r["yi"]) for r in rows]),
        "vi": np.array([float(r["vi"]) for r in rows]),
        "study": [r["study"] for r in rows],
        "treat1": [r["treat1"] for r in rows],
        "treat2": [r["treat2"] for r in rows],
        "dose1": np.array([float(r["dose1"]) for r in rows]),
        "dose2": np.array([float(r["dose2"]) for r in rows]),
    }


def subset(data: dict, treat: str) -> dict:
    keep = [i for i, (a, b) in enumerate(zip(data["treat1"], data["treat2"]))
            if a == "P" and b == treat]
    return {key: (np.asarray(value)[keep] if isinstance(value, np.ndarray)
                  else [value[i] for i in keep])
            for key, value in data.items()}


def fit(data: dict, family: str, **kwargs):
    payload = {key: value for key, value in data.items() if key != "yi"}
    return ma.nma_dose(data["yi"], family=family, **payload, **kwargs)


@pytest.fixture(scope="module")
def full() -> dict:
    return load()


@pytest.fixture(scope="module")
def two_a(full) -> dict:
    return subset(full, "A")


@pytest.fixture(scope="module")
def two_b(full) -> dict:
    return subset(full, "B")


# --------------------------------------------------------------------------- #
# model level: the linear dose functions against metafor::rma.mv
# --------------------------------------------------------------------------- #
@pytest.mark.filterwarnings(
    "ignore:the between-study variance is estimated at the boundary:RuntimeWarning")
@pytest.mark.parametrize("treat", ["A", "B"])
@pytest.mark.parametrize("family", ["linear", "quadratic", "loglinear", "rcs"])
def test_each_dose_function_matches_rma_mv(treat, family, two_a, two_b):
    data = two_a if treat == "A" else two_b
    kwargs = {"knots": KNOTS[treat]} if family == "rcs" else {}
    res = fit(data, family, **kwargs)
    assert res.coef == pytest.approx(R_COEF[(treat, family)], abs=1e-7)
    assert res.se == pytest.approx(R_SE[(treat, family)], rel=1e-6, abs=1e-12)
    assert res.tau2 == pytest.approx(R_TAU2[(treat, family)], abs=5e-8)


def test_the_rcs_basis_matches_rms():
    basis = _rcs_matrix(np.array([0.0, 25.0, 50.0, 200.0]),
                        np.array(KNOTS["A"]))
    assert basis == pytest.approx(RMS_BASIS, abs=1e-9)


def test_rma_uni_is_not_the_reference(two_a):
    """A study with several contrasts is why the reference is rma.mv.

    ``rma.uni`` assumes independent rows, so it reports 0.2216 here; the model
    with a study-level random effect gives 0.3013, which both ``rma.mv`` and
    pimeta find.  The test pins the two apart so nobody "fixes" pimeta back to
    the wrong one.
    """
    res = fit(two_a, "linear")
    assert res.tau2 == pytest.approx(0.3013134977, abs=5e-8)
    assert abs(res.tau2 - 0.2216198081) > 0.05


# --------------------------------------------------------------------------- #
# model level: the non-linear dose functions against nls
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("family", ["emax", "exponential"])
def test_non_linear_dose_functions_match_nls(family, full):
    res = fit(full, family, struct="FE")
    assert res.coef == pytest.approx(NLS_COEF[family], abs=1e-4)
    assert res.se == pytest.approx(NLS_SE_KNOWN_SCALE[family], rel=1e-4,
                                   abs=1e-6)


@pytest.mark.parametrize("family", ["emax", "exponential"])
def test_the_fitted_means_reproduce_the_nls_residual_sum_of_squares(
        family, full):
    """The residual sum of squares ``nls`` reports, recomputed through the API."""
    res = fit(full, family, struct="FE")
    mu = np.array([
        res.contrast(a, d1, b, d2).est[0]
        for a, b, d1, d2 in zip(full["treat1"], full["treat2"],
                                full["dose1"], full["dose2"])
    ])
    rss = float(np.sum((full["yi"] - mu) ** 2 / full["vi"]))
    assert rss == pytest.approx(RSS[family], abs=1e-8)


# --------------------------------------------------------------------------- #
# equation level: consistency, at every dose
# --------------------------------------------------------------------------- #
def test_consistency_holds_at_doses_nobody_studied(full):
    """``f_B(15) - f_P(0)`` equals the sum through the intermediate curve."""
    res = fit(full, "emax")
    direct = res.contrast("P", 0.0, "B", 15.0)
    step1 = res.contrast("P", 0.0, "A", 37.0)
    step2 = res.contrast("A", 37.0, "B", 15.0)
    assert direct.est[0] == pytest.approx(step1.est[0] + step2.est[0],
                                          abs=1e-12)
    # the same functional has to have the same Jacobian whichever way it is
    # assembled, so the delta-method variance cannot depend on the route
    vcov = np.asarray(res.vcov)
    one = _functional(res, "B", 15.0, "P", 0.0)
    two = (_functional(res, "A", 37.0, "P", 0.0)
           + _functional(res, "B", 15.0, "A", 37.0))
    assert one == pytest.approx(two, abs=1e-10)
    assert float(one @ vcov @ one) == pytest.approx(float(two @ vcov @ two),
                                                    rel=1e-12)


def _functional(res, plus_treat: str, plus_dose: float,
                minus_treat: str, minus_dose: float) -> np.ndarray:
    """The contrast's row of the Jacobian: ``d f_b(x_b) - d f_a(x_a)``."""
    return (res._curve_jacobian(plus_treat, np.array([plus_dose]))[0]
            - res._curve_jacobian(minus_treat, np.array([minus_dose]))[0])


def test_every_curve_is_zero_at_the_anchor(full):
    for family in ("linear", "quadratic", "loglinear", "emax", "exponential"):
        res = fit(full, family)
        for treat in ("A", "B"):
            point = res.curve(treat, [0.0])
            assert point.est[0] == pytest.approx(0.0, abs=1e-12)
            assert point.se[0] == pytest.approx(0.0, abs=1e-12)


def test_a_placebo_arm_never_gets_curve_parameters(full):
    """Treatment P appears only at dose 0, so it has no curve to estimate."""
    res = fit(full, "linear")
    assert "P" not in res.curves
    assert all(name.startswith(("A:", "B:")) for name in res.names)


# --------------------------------------------------------------------------- #
# property level: the fitted curve covers the truth
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("treat", ["A", "B"])
def test_the_curve_interval_covers_the_generating_curve(treat, full):
    res = fit(full, "emax")
    emax, ed50 = TRUE_EMAX[treat]
    doses = sorted({float(d) for t, d in zip(full["treat2"], full["dose2"])
                    if t == treat})
    band = res.curve(treat, doses)
    truth = np.array([emax * x / (ed50 + x) for x in doses])
    assert np.all(band.ci_lb <= truth + 1e-12)
    assert np.all(truth <= band.ci_ub + 1e-12)


def test_a_steeper_curve_is_estimated_steeper(full):
    """The fixture's A curve is above B's at every shared dose."""
    res = fit(full, "emax")
    for dose in (10.0, 20.0, 40.0):
        a = res.curve("A", [dose])
        b = res.curve("B", [dose])
        assert a.est[0] > b.est[0]


# --------------------------------------------------------------------------- #
# the heterogeneity structures
# --------------------------------------------------------------------------- #
def test_a_random_curve_differs_from_a_random_level(full):
    level = fit(full, "linear", struct="study")
    curve = fit(full, "linear", struct="coef")
    assert level.D is None and not np.isnan(level.tau2)
    assert curve.D is not None and np.isnan(curve.tau2)
    assert np.allclose(curve.D, np.diag(np.diag(curve.D)))
    assert not np.allclose(curve.coef, level.coef)


def test_a_full_covariance_for_the_curve(full):
    res = fit(full, "quadratic", struct="coef+study", coef_struct="UN")
    assert res.tau2 > 0.0
    assert res.D.shape == (4, 4)
    first = res.D[:2, :2]
    assert first[0, 1] == pytest.approx(first[1, 0], abs=1e-12)


def test_a_shared_curve_for_a_class_of_treatments(full):
    res = fit(full, "linear", treatment_specific=False)
    assert res.names == ["dose"]
    assert res.curves == ["A", "B"]


def test_the_boundary_tau_is_reported_not_hidden(two_b):
    with pytest.warns(RuntimeWarning, match="boundary"):
        res = fit(two_b, "linear")
    assert res.tau2 < 1e-6


def test_rho_is_inert_when_every_study_has_one_contrast(two_a):
    """The correlation can only matter where a study contributes twice."""
    seen, keep = set(), []
    for i, study in enumerate(two_a["study"]):
        if study not in seen:
            seen.add(study)
            keep.append(i)
    single = {key: (np.asarray(value)[keep] if isinstance(value, np.ndarray)
                    else [value[i] for i in keep])
              for key, value in two_a.items()}
    plain = fit(single, "linear")
    correlated = fit(single, "linear", rho=0.6)
    assert correlated.coef == pytest.approx(plain.coef, rel=1e-12)
    assert correlated.se == pytest.approx(plain.se, rel=1e-12)
    assert correlated.tau2 == pytest.approx(plain.tau2, abs=1e-10)


def test_rho_matters_when_a_study_reports_several_doses(two_a):
    """The direction is a property of the data, and that is the point.

    Here a positive within-study correlation makes the slope slightly *more*
    precise (0.000897 against 0.001188), because the contrasts it links are at
    different doses within the same study, so the study's own spread carries
    more of the information.  On another dataset the sign could go the other
    way.  What is not optional is the assumption: ignoring the correlation
    changes the estimate, the variance and the heterogeneity together, which is
    why ``rho`` has to be supplied rather than defaulted to zero.
    """
    plain = fit(two_a, "linear")
    correlated = fit(two_a, "linear", rho=0.6)
    assert correlated.coef[0] == pytest.approx(0.006264875990828662, abs=1e-9)
    assert correlated.tau2 == pytest.approx(0.3246430749057714, abs=1e-7)
    assert not np.allclose(plain.se, correlated.se, rtol=1e-6)
    assert plain.loglik != pytest.approx(correlated.loglik, abs=1e-6)


# --------------------------------------------------------------------------- #
# refusals, because a wrong answer is worse than none
# --------------------------------------------------------------------------- #
def test_a_random_curve_needs_a_linear_dose_function(full):
    with pytest.raises(ValueError, match="no closed form"):
        fit(full, "emax", struct="coef")


def test_a_curve_needs_more_than_one_dose(full):
    one_dose = subset(full, "A")
    keep = [i for i, d in enumerate(one_dose["dose2"]) if d == 10.0]
    trimmed = {key: (np.asarray(value)[keep] if isinstance(value, np.ndarray)
                     else [value[i] for i in keep])
               for key, value in one_dose.items()}
    with pytest.raises(ValueError, match="cannot be identified"):
        fit(trimmed, "quadratic")


def test_unknown_dose_function_is_refused(full):
    with pytest.raises(ValueError, match="unknown dose function"):
        fit(full, "spline")


def test_a_non_positive_definite_covariance_is_refused(full):
    bad = dict(full)
    bad["vi"] = np.full(full["vi"].size, -1.0)
    with pytest.raises(ValueError, match="must be positive"):
        fit(bad, "linear")


def test_a_bad_level_is_normalised():
    data = load()
    payload = {key: value for key, value in data.items() if key != "yi"}
    widths = []
    for value in (0.95, 95, 0.05):
        res = ma.nma_dose(data["yi"], family="linear", level=value, **payload)
        band = res.curve("A", [50.0])
        widths.append(float(band.ci_ub[0] - band.ci_lb[0]))
    assert widths[0] == pytest.approx(widths[1], rel=1e-12)
    assert widths[0] == pytest.approx(widths[2], rel=1e-12)


# --------------------------------------------------------------------------- #
# reporting
# --------------------------------------------------------------------------- #
def test_the_dose_test_rejects_no_dose_effect(full):
    res = fit(full, "linear")
    out = res.test_dose()
    assert out["df"] == 2
    assert out["pval"] < 1e-8


def test_the_result_serialises(full):
    res = fit(full, "emax")
    payload = res.to_dict()
    assert payload["family"] == "emax"
    assert payload["k"] == full["yi"].size
    assert payload["curves"] == ["A", "B"]
    assert set(payload["dose_range"]) == {"A", "B", "P"}
    import json
    assert json.loads(json.dumps(payload))["struct"] == "study"


def test_the_coefficient_table_is_complete(full):
    res = fit(full, "linear")
    rows = res.coef_table()
    assert len(rows) == 2
    for row in rows:
        assert set(row) == {"term", "estimate", "se", "zval", "pval",
                            "ci_lb", "ci_ub"}
        assert row["ci_lb"] < row["estimate"] < row["ci_ub"]


def test_the_summary_names_the_structure(full):
    text = str(fit(full, "linear", struct="coef"))
    assert "Dose-response network meta-analysis" in text
    assert "coef" in text


def test_a_curve_outside_the_observed_doses_is_flagged(full):
    res = fit(full, "linear")
    band = res.curve("A", [50.0, 500.0])
    assert not band.outside[0]
    assert band.outside[1]


def test_the_dose_functions_are_listed():
    assert dose_functions() == [
        "linear", "quadratic", "loglinear", "emax", "exponential", "rcs"]


def test_a_single_study_falls_back_to_least_squares(full):
    one = {key: (np.asarray(value)[:3] if isinstance(value, np.ndarray)
                 else value[:3]) for key, value in full.items()}
    res = fit(one, "linear", struct="FE")
    assert res.k == 3
    assert np.isfinite(res.loglik)


def test_a_between_study_variance_needs_two_studies(full):
    one = {key: (np.asarray(value)[:1] if isinstance(value, np.ndarray)
                 else value[:1]) for key, value in full.items()}
    with pytest.raises(ValueError, match="at least two studies"):
        fit(one, "linear")


def test_the_optimiser_does_not_wander_for_emax(full):
    """A weaker fit must not produce a better likelihood than a stronger one."""
    fe = fit(full, "emax", struct="FE")
    re = fit(full, "emax", struct="study")
    assert re.loglik > fe.loglik


def test_no_warning_is_raised_for_a_well_behaved_fit(full):
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        fit(full, "linear")
