"""Dose-response meta-analysis: the covariance, the model, and R parity.

All reference numbers come from R 4.6.1 on a small hand-made dataset (three
studies, four dose levels, 500 participants each) so that nothing has to be
redistributed:

* the adjusted counts are ``dosresmeta:::grl`` (dosresmeta 2.2.0);
* the covariance matrix is the one those counts imply;
* the fit is ``metafor::rma.mv`` (5.2.1) called with that block-diagonal
  covariance, ``mods = ~ rcs(dose, parms = c(5, 15, 25))`` and
  ``random = ~ 1 | id``.

``dosresmeta`` fits the same mixed model but also puts random effects on the
spline coefficients, so its default is not this estimator; the ``rma.mv`` call
is the exact counterpart, which is why it is the reference.
"""

from __future__ import annotations

import numpy as np
import pytest

import pimeta as ma

# --------------------------------------------------------------------------- #
# the dataset
# --------------------------------------------------------------------------- #
STUDY = ["A"] * 4 + ["B"] * 4 + ["C"] * 4
DOSE = [0.0, 10.0, 20.0, 30.0] * 3
CASES = [10, 18, 25, 31, 14, 20, 27, 30, 8, 15, 24, 33]
N = [500] * 12


def _effects():
    """log RR against the study's dose 0, and the delta-method variance."""
    cases = np.array(CASES, dtype=float)
    n = np.array(N, dtype=float)
    ref = np.repeat(cases[[0, 4, 8]], 4)
    ref_n = np.repeat(n[[0, 4, 8]], 4)
    logrr = np.log((cases / n) / (ref / ref_n))
    var = 1.0 / cases - 1.0 / n + 1.0 / ref - 1.0 / ref_n
    var[[0, 4, 8]] = 0.0                     # the reference rows
    logrr[[0, 4, 8]] = 0.0
    return logrr, var


LOGRR, VAR = _effects()

# dosresmeta:::grl(logrr, se^2, cases, n, type = "cc")$A, per study
R_ADJUSTED = {
    "A": [10.281542333125259, 18.212328196638556, 24.941598837951698,
          30.56169901750539],
    "B": [14.29481010266835, 20.176901931073886, 26.859459112237214,
          29.6667694095818],
    "C": [8.2690810722394588, 15.283527821040188, 24.013236158189017,
          32.434064578294056],
}

# The 3x3 covariance of study A's three contrasts.
R_COV_A = [
    [0.156287126742537613, 0.099303661793967352, 0.099303661793967352],
    [0.099303661793967352, 0.141502326796887956, 0.099303661793967352],
    [0.099303661793967352, 0.099303661793967352, 0.134154561131338002],
]

# metafor::rma.mv(yi, V = <block diag>, mods = ~ rcs(dose, parms = c(5,15,25))
#                 [, -1], random = ~ 1 | id)
R_COEF = [-0.0085607969416592322, 0.050733627715477236, -0.019805511441233797]
R_SE = [0.58459656653494008, 0.047504783464277757, 0.040899103930977281]
R_SIGMA2 = 1.7981253820429208e-11
R_QE = 1.980616534986072
R_QM = 10.701515522202243
R_QM_P = 0.0047445543981362399


def _fit(**kw):
    return ma.dosres(LOGRR, study=STUDY, dose=DOSE, vi=VAR, cases=CASES, n=N,
                     type="cc", knots=[5.0, 15.0, 25.0], **kw)


# --------------------------------------------------------------------------- #
# the reference group
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("study", ["A", "B", "C"])
def test_greenland_matches_dosresmeta(study):
    rows = [i for i, s in enumerate(STUDY) if s == study]
    adjusted = ma.greenland_counts(
        [CASES[i] for i in rows], [N[i] for i in rows],
        [LOGRR[i] for i in rows], [VAR[i] for i in rows], type="cc",
    )
    # The contrasts are the part that matters: dosresmeta stops at tol = 1e-5
    # on the squared change.  Its returned *reference* count is one iteration
    # behind its own contrasts (its totals are not conserved), so it is only
    # compared loosely -- pimeta returns the count consistent with what it
    # reports, which is the point of the adjustment.
    assert adjusted[1:] == pytest.approx(R_ADJUSTED[study][1:], abs=1e-4)
    assert adjusted[0] == pytest.approx(R_ADJUSTED[study][0], rel=1e-3)


def test_greenland_leaves_a_consistent_table_almost_alone():
    """The correction moves the counts; that is the point of it."""
    adjusted = ma.greenland_counts(
        [CASES[i] for i in range(4)], [N[i] for i in range(4)],
        [LOGRR[i] for i in range(4)], [VAR[i] for i in range(4)], type="cc",
    )
    # The reported log RRs were built from the raw counts, so the move is small
    # but not zero: the reference group's variance is shared.
    assert adjusted == pytest.approx(np.array(CASES[:4], dtype=float), abs=1.0)


def test_greenland_rejects_a_missing_reference():
    with pytest.raises(ValueError, match="exactly one reference"):
        ma.greenland_counts([10, 20, 30], [100, 100, 100], [0, 0.1, 0.2],
                            [0.0, 0.0, 0.01])
    with pytest.raises(ValueError, match="at least two dose levels"):
        ma.greenland_counts([10], [100], [0.0])


def test_greenland_rejects_impossible_counts():
    with pytest.raises(ValueError, match="must be positive"):
        ma.greenland_counts([0, 20], [100, 100], [0.0, 0.1], [0.0, 0.01])


# --------------------------------------------------------------------------- #
# the covariance
# --------------------------------------------------------------------------- #
def test_dose_cov_reproduces_the_greenland_matrix():
    rows = [i for i, s in enumerate(STUDY) if s == "A"]
    cov = ma.dose_cov(
        [DOSE[i] for i in rows], [CASES[i] for i in rows], [N[i] for i in rows],
        effects=[LOGRR[i] for i in rows], variances=[VAR[i] for i in rows],
        type="cc",
    )
    # abs 1e-4: the reference count differs in its fifth decimal, and the
    # shared term is 1 / A0, so the matrix moves by about that much.
    assert cov == pytest.approx(np.array(R_COV_A), abs=1e-4)


def test_the_covariance_is_shared_control_plus_own_variance():
    cov = ma.dose_cov(
        DOSE[:4], CASES[:4], N[:4],
        effects=LOGRR[:4], variances=VAR[:4], type="cc",
    )
    off = cov[0, 1]
    assert np.allclose(cov[np.triu_indices(3, 1)], off)      # constant
    assert np.all(np.diag(cov) > off)                         # own + shared


def test_a_constant_correlation_covariance_is_what_it_says():
    vi = np.array([0.04, 0.09, 0.16])
    cov = ma.dose_cov([1, 2, 3], variances=[0.0, 0.04, 0.09, 0.16], rho=0.5)
    expected = 0.5 * np.outer(np.sqrt(vi), np.sqrt(vi))
    np.fill_diagonal(expected, vi)
    assert cov == pytest.approx(expected)


def test_dose_cov_needs_one_of_the_two_routes():
    with pytest.raises(ValueError, match="give either"):
        ma.dose_cov([1, 2, 3], variances=[0.0, 0.01, 0.02])
    with pytest.raises(ValueError, match="needs 'variances'"):
        ma.dose_cov([1, 2, 3], rho=0.5)


def test_dose_cov_rejects_an_unknown_type():
    with pytest.raises(ValueError, match="unknown data type"):
        ma.dose_cov([1, 2], events=[1, 2], totals=[10, 10], effects=[0, 1],
                    variances=[0.0, 0.1], type="odds")


# --------------------------------------------------------------------------- #
# the model
# --------------------------------------------------------------------------- #
@pytest.fixture(scope="module")
def fit():
    return _fit()


def test_the_fit_matches_metafors_rma_mv(fit):
    # Absolutes: the intercept sits near zero, and the reference covariance is
    # the one dosresmeta stopped at rather than the converged one.
    assert fit.b == pytest.approx(R_COEF, abs=2e-5)
    assert fit.se == pytest.approx(R_SE, rel=1e-4)
    assert fit.tau2 == pytest.approx(R_SIGMA2, abs=1e-9)
    assert fit.QE == pytest.approx(R_QE, rel=1e-4)


def test_the_reference_rows_are_excluded(fit):
    assert fit.studies == 3
    assert fit.k == 9                 # 12 rows, 3 reference levels
    assert fit.coefnames == ["intrcpt", "dose[1]", "dose[2]"]


def test_the_spline_term_is_testable(fit):
    test = ma.nonlinear_test(fit, "dose")
    assert test.overall.df == 2
    assert test.nonlinear.df == 1
    # metafor's anova(mv, btt = 2:4) covers the two spline coefficients.
    both = ma.wald_test(fit, btt=[2, 3])
    assert both.QM == pytest.approx(R_QM, rel=1e-4)
    assert both.pval == pytest.approx(R_QM_P, abs=1e-5)


def test_a_curve_can_be_predicted(fit):
    curve = ma.predict_curve(fit, fit.basis, grid=[0.0, 10.0, 20.0])
    assert curve.fit[0] == pytest.approx(fit.b[0], rel=1e-12)
    assert curve.x == pytest.approx([0.0, 10.0, 20.0])
    # A new study adds tau^2, so the prediction band is the wider one.
    assert np.all(curve.pi_lb <= curve.ci_lb)


def test_the_independent_covariance_is_available():
    """Ignoring the shared reference is possible -- it is just usually wrong."""
    fit = ma.dosres(LOGRR, study=STUDY, dose=DOSE, vi=VAR,
                    knots=[5.0, 15.0, 25.0])
    assert fit.covariance == "independent"
    assert np.all(np.isfinite(fit.b))


def test_the_constant_correlation_route_runs():
    fit = ma.dosres(LOGRR, study=STUDY, dose=DOSE, vi=VAR, rho=0.5,
                    knots=[5.0, 15.0, 25.0])
    assert fit.covariance == "constant"
    assert np.all(np.isfinite(fit.b))


def test_a_single_row_study_is_rejected():
    with pytest.raises(ValueError, match="at least two contrasts"):
        ma.dosres([0.1, 0.2], study=["A", "B"], dose=[1.0, 2.0],
                  vi=[0.01, 0.02], rho=0.5)


def test_interleaved_rows_are_accepted():
    """Blocks and rows must line up even when a study's rows are not contiguous."""
    order = [0, 4, 8, 1, 5, 9, 2, 6, 10, 3, 7, 11]
    fit = ma.dosres(
        [LOGRR[i] for i in order], study=[STUDY[i] for i in order],
        dose=[DOSE[i] for i in order], vi=[VAR[i] for i in order],
        cases=[CASES[i] for i in order], n=[N[i] for i in order],
        type="cc", knots=[5.0, 15.0, 25.0],
    )
    assert fit.b == pytest.approx(R_COEF, abs=2e-5)


def test_a_dose_response_fit_round_trips_through_a_record(tmp_path):
    from pimeta.record import load_record, save_record

    fit = _fit()
    path = tmp_path / "dosres.json"
    save_record(path, fit, mode="digest", function="dosres")
    loaded = load_record(path)
    assert loaded.kind == "dosres"
    assert loaded.results["values"]["tau2"] == pytest.approx(fit.tau2, rel=1e-12)
    assert loaded.results["values"]["b"] == pytest.approx(fit.b, rel=1e-12)
