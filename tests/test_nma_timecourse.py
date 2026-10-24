"""Time-course network meta-analysis: three levels of check.

**Equation level.**  The consistency equation must cancel at *every* time, not
only at the times that were measured, and it must cancel because the same
parameter vector is used wherever a treatment appears.  The composition test
asserts ``d_db(t) = d_da(t) + d_ab(t)`` at a time no study measured.

**Model level.**  ``metafor::rma.mv`` with exactly the same within-study
covariance — ``phi**|t_i - t_j|`` inside a study — for all three bases, and
``netmeta`` for the time-invariant limit: with ``basis="constant"`` the model
*is* standard network meta-analysis, so a treatment effect, a standard error and
a `tau2` can be compared directly.

**Property level.**  The fixture was simulated from known trajectories, so the
fitted curve has to cover the truth and the trend test has to reject.
"""

from __future__ import annotations

import csv
import json
import pathlib

import numpy as np
import pytest

import pimeta as ma
from pimeta.nma.timecourse import nma_timecourse, time_bases

DATA = pathlib.Path(__file__).parent / "data" / "nma_timecourse.csv"

# metafor 5.2-1, rma.mv(..., V = within-study AR(1) with phi = 0.6,
#                       random = ~ 1 | study, method = "REML")
R_COEF = {
    "linear": [0.3862875393, 0.0759636953, 1.0925306887, -0.0284844404],
    "quadratic": [0.3625886567, 0.1040298784, -0.0023429736,
                  1.1216695553, -0.0524305255, 0.0015433631],
    "constant": [0.8017759983, 0.9100456224],
}
R_SE = {
    "linear": [0.1105962768, 0.0094939862, 0.1267130871, 0.0089622306],
    "quadratic": [0.1139418668, 0.0366244249, 0.0029529176,
                  0.1313311840, 0.0317572763, 0.0019640325],
    "constant": [0.1012978602, 0.1170772388],
}
R_TAU2 = {"linear": 0.0573241462, "quadratic": 0.0561652046,
          "constant": 0.0631550784}

# the fixture's generating trajectories (months)
TRUE = {"A": (0.45, 0.075), "B": (0.95, -0.035)}


def load(only_time: float | None = None) -> dict:
    with DATA.open(encoding="utf-8", newline="") as fh:
        rows = list(csv.DictReader(fh))
    if only_time is not None:
        rows = [r for r in rows if float(r["time"]) == only_time]
    return {
        "yi": np.array([float(r["yi"]) for r in rows]),
        "vi": np.array([float(r["vi"]) for r in rows]),
        "study": [r["study"] for r in rows],
        "treat1": [r["treat1"] for r in rows],
        "treat2": [r["treat2"] for r in rows],
        "time": np.array([float(r["time"]) for r in rows]),
    }


def fit(data: dict, basis: str = "linear", **kwargs):
    payload = {key: value for key, value in data.items() if key != "yi"}
    return nma_timecourse(data["yi"], basis=basis, **payload, **kwargs)


@pytest.fixture(scope="module")
def full() -> dict:
    return load()


# --------------------------------------------------------------------------- #
# model level: the three bases against metafor::rma.mv
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("basis", ["linear", "quadratic", "constant"])
def test_each_basis_matches_rma_mv(basis, full):
    res = fit(full, basis, phi=0.6)
    assert res.coef == pytest.approx(R_COEF[basis], abs=1e-7)
    assert res.se == pytest.approx(R_SE[basis], rel=1e-6, abs=1e-12)
    assert res.tau2 == pytest.approx(R_TAU2[basis], abs=2e-7)


def test_the_time_invariant_limit_is_standard_nma(full):
    """``basis="constant"`` on one row per study is ordinary NMA, so `netmeta`
    has to agree with it — on the effect, the standard error and ``tau2``."""
    data = load(only_time=0.0)
    res = fit(data, "constant")
    net = ma.netmeta(
        data["yi"], np.sqrt(data["vi"]), data["treat1"], data["treat2"],
        studlab=data["study"], method_tau="REML", reference_group="P")
    row = net.trts.index("P")
    for column, name in ((net.trts.index("A"), "A"), (net.trts.index("B"), "B")):
        i = res.names.index(f"{name}:intercept")
        assert res.coef[i] == pytest.approx(
            float(net.TE_random[row][column]), abs=1e-8)
        assert res.se[i] == pytest.approx(
            float(net.seTE_random[row][column]), rel=1e-8)
    assert res.tau2 == pytest.approx(float(net.tau2), abs=1e-8)


# --------------------------------------------------------------------------- #
# equation level
# --------------------------------------------------------------------------- #
def test_the_contrast_composes_at_a_time_nobody_measured(full):
    res = fit(full, "linear", phi=0.6)
    direct = res.contrast("P", 0.0, "B", 7.0)
    step = (res.contrast("P", 0.0, "A", 2.5).est[0]
            + res.contrast("A", 2.5, "B", 7.0).est[0])
    assert direct.est[0] == pytest.approx(step, abs=1e-12)


def test_the_reference_curve_is_zero(full):
    res = fit(full, "linear")
    assert res.reference == "P"
    band = res.curve("P", [0.0, 6.0, 12.0])
    assert np.allclose(band.est, 0.0, atol=1e-14)
    assert np.allclose(band.se, 0.0, atol=1e-14)


def test_the_reference_can_be_chosen(full):
    res = fit(full, "linear", reference="A")
    assert res.reference == "A"
    assert res.curves == ["B", "P"]
    assert res.names[0].startswith("B:")


# --------------------------------------------------------------------------- #
# property level
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("treat", ["A", "B"])
def test_the_trajectory_covers_the_truth(treat, full):
    res = fit(full, "linear", phi=0.6)
    intercept, slope = TRUE[treat]
    times = np.array([0.0, 4.0, 8.0, 12.0, 16.0])
    band = res.curve(treat, times)
    truth = intercept + slope * times
    assert np.all(band.ci_lb <= truth + 1e-12)
    assert np.all(truth <= band.ci_ub + 1e-12)


def test_the_trend_test_rejects_a_flat_curve(full):
    res = fit(full, "linear", phi=0.6)
    out = res.test_trend()
    assert out["df"] == 2
    assert out["pval"] < 1e-6


def test_the_effect_test_rejects_no_difference(full):
    res = fit(full, "linear", phi=0.6)
    out = res.test_effect()
    assert out["df"] == 2
    assert out["pval"] < 1e-6


def test_the_quadratic_term_is_not_needed(full):
    """This fixture is linear in time, so the curvature test must not reject."""
    res = fit(full, "quadratic", phi=0.6)
    assert res.test_nonlinear()["pval"] > 0.05


def test_a_constant_basis_has_no_trend_to_test(full):
    res = fit(full, "constant")
    with pytest.raises(ValueError, match="standard network meta-analysis"):
        res.test_trend()


def test_a_linear_basis_has_no_curvature(full):
    res = fit(full, "linear")
    with pytest.raises(ValueError, match="no non-linearity"):
        res.test_nonlinear()


# --------------------------------------------------------------------------- #
# the within-study correlation
# --------------------------------------------------------------------------- #
def test_the_correlation_is_inert_with_one_row_per_study():
    data = load(only_time=0.0)
    plain = fit(data, "constant")
    correlated = fit(data, "constant", phi=0.6)
    assert correlated.coef == pytest.approx(plain.coef, rel=1e-12)
    assert correlated.tau2 == pytest.approx(plain.tau2, abs=1e-12)


def test_the_correlation_changes_the_fit_when_a_study_repeats(full):
    plain = fit(full, "linear")
    correlated = fit(full, "linear", phi=0.6)
    assert not np.allclose(plain.se, correlated.se, rtol=1e-6)
    assert plain.loglik != pytest.approx(correlated.loglik, abs=1e-6)


def test_rho_and_phi_are_two_ways_of_saying_the_same_thing(full):
    a = fit(full, "linear", phi=0.6)
    b = fit(full, "linear", rho=0.6)
    c = fit(full, "linear", phi=0.0)
    d = fit(full, "linear", rho=0.0)
    assert a.loglik != pytest.approx(b.loglik, abs=1e-6)
    assert c.coef == pytest.approx(d.coef, rel=1e-12)
    assert c.tau2 == pytest.approx(d.tau2, abs=1e-10)


def test_only_one_correlation_assumption_is_allowed(full):
    with pytest.raises(ValueError, match="at most one"):
        fit(full, "linear", phi=0.6, rho=0.6)
    with pytest.raises(ValueError, match="at most one"):
        fit(full, "linear", phi=0.6, vcov=np.eye(full["yi"].size) * 0.05)


def test_a_study_with_two_treatment_pairs_needs_an_explicit_covariance(full):
    """Two arms at the same time share the control, and ``phi`` cannot say so."""
    data = load()
    extra = dict(data)
    extra["treat2"] = list(data["treat2"])
    extra["treat2"][0] = "B" if extra["treat2"][0] == "A" else "A"
    with pytest.raises(ValueError, match="vcov"):
        fit(extra, "linear", phi=0.6)


def test_an_impossible_correlation_is_refused(full):
    with pytest.raises(ValueError, match="must be in"):
        fit(full, "linear", phi=1.0)


# --------------------------------------------------------------------------- #
# uneven schedules and identifiability
# --------------------------------------------------------------------------- #
def test_uneven_schedules_are_fitted_as_reported(full):
    """Studies measuring different times are not padded or dropped."""
    res = fit(full, "linear", phi=0.6)
    observed_a = res.times["A"]
    assert observed_a == sorted({float(t) for t, trt in
                                 zip(full["time"], full["treat2"])
                                 if trt == "A"})
    assert 3.0 in observed_a and 6.0 in observed_a and 12.0 in observed_a
    assert 8.0 in res.times["B"] and 16.0 in res.times["B"]


def test_a_quadratic_needs_three_distinct_times():
    data = load()
    keep = [i for i, (t, trt) in enumerate(zip(data["time"], data["treat2"]))
            if trt != "B" or t in (0.0, 4.0)]
    trimmed = {key: (np.asarray(value)[keep] if isinstance(value, np.ndarray)
                     else [value[i] for i in keep])
               for key, value in data.items()}
    with pytest.raises(ValueError, match="fewer distinct times"):
        fit(trimmed, "quadratic", phi=0.6)


def test_a_duplicated_cell_is_refused(full):
    data = load()
    duplicated = {key: (np.concatenate([value, value[:1]])
                        if isinstance(value, np.ndarray)
                        else list(value) + value[:1])
                  for key, value in data.items()}
    with pytest.raises(ValueError, match="duplicate"):
        fit(duplicated, "linear")


def test_an_unknown_basis_is_refused(full):
    with pytest.raises(ValueError, match="unknown time basis"):
        fit(full, "exponential")


def test_a_single_study_has_no_between_study_variance(full):
    data = load()
    one = {key: (np.asarray(value)[:2] if isinstance(value, np.ndarray)
                 else value[:2]) for key, value in data.items()}
    with pytest.raises(ValueError, match="at least two studies"):
        fit(one, "linear")


# --------------------------------------------------------------------------- #
# reporting
# --------------------------------------------------------------------------- #
def test_the_result_serialises(full):
    res = fit(full, "linear", phi=0.6)
    payload = res.to_dict()
    assert json.loads(json.dumps(payload))["reference"] == "P"
    assert payload["phi"] == 0.6
    assert payload["basis"] == "linear"
    assert payload["k"] == full["yi"].size


def test_the_coefficient_table_is_complete(full):
    for row in fit(full, "linear").coef_table():
        assert set(row) == {"term", "estimate", "se", "zval", "pval",
                            "ci_lb", "ci_ub"}
        assert row["ci_lb"] < row["estimate"] < row["ci_ub"]


def test_the_summary_names_the_structure(full):
    text = str(fit(full, "linear", phi=0.6, struct="coef"))
    assert "Time-course network meta-analysis" in text
    assert "reference: P" in text
    assert "phi = 0.6" in text


def test_a_curve_outside_the_observed_times_is_flagged(full):
    band = fit(full, "linear").curve("B", [8.0, 60.0])
    assert not band.outside[0]
    assert band.outside[1]


def test_levels_are_normalised(full):
    widths = []
    for level in (0.95, 95, 0.05):
        band = fit(full, "linear", level=level).curve("A", [12.0])
        widths.append(float(band.ci_ub[0] - band.ci_lb[0]))
    assert widths[0] == pytest.approx(widths[1], rel=1e-12)
    assert widths[0] == pytest.approx(widths[2], rel=1e-12)


def test_the_bases_are_listed():
    assert time_bases() == ["constant", "linear", "quadratic", "rcs"]


def test_an_rcs_basis_runs(full):
    knotted = fit(full, "rcs", phi=0.6, n_knots=2)
    assert knotted.basis == "rcs"
    assert len(knotted.names) == 2 * 3          # two treatments, 1 + 2 columns
    # a restricted cubic spline contains the straight line, so under ML its
    # likelihood cannot be worse.  Under REML it could: restricted likelihoods
    # of models with different fixed effects are not comparable, which is a
    # reason to prefer ML for this comparison rather than a detail.
    straight = fit(full, "linear", phi=0.6, method="ML")
    spline = fit(full, "rcs", phi=0.6, n_knots=2, method="ML")
    assert spline.loglik >= straight.loglik - 1e-9
