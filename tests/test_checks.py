"""Guards for the structured diagnostics.

Every check needs two tests, and the second is the one that is usually missing:
one that builds the condition and asserts the code fires, and one that asserts it
does **not** fire on a clean fit. A diagnostic that only ever fires is as
useless as one that never does.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

import pimeta as ma


def _stub(**kw):
    """A fitted-result stand-in carrying only the fields a check reads."""
    base = {"tau2": None, "sigma2": None, "vb": None, "rho": None,
            "b": None, "se": None, "weights": None,
            "k": 10, "k_all": 10, "priors": {}, "tau": {}}
    base.update(kw)
    return SimpleNamespace(**base)


def _fixture():
    d = pd.read_csv(Path(__file__).resolve().parents[1]
                    / "tests" / "data" / "robma.csv")
    return d["yi"].to_numpy(float), d["sei"].to_numpy(float)


# --------------------------------------------------------------------------- #
# the record itself
# --------------------------------------------------------------------------- #
def test_a_diagnostic_rejects_a_bad_severity():
    with pytest.raises(ValueError, match="severity"):
        ma.Diagnostic(code="x", category="data", severity="fatal",
                      message="m", action="a")


def test_a_diagnostic_rejects_a_code_that_is_not_an_identifier():
    with pytest.raises(ValueError, match="snake_case"):
        ma.Diagnostic(code="not a code", category="data", severity="info",
                      message="m", action="a")


def test_to_dict_round_trips():
    d = ma.Diagnostic(code="tau2_zero", category="boundary", severity="warning",
                      message="m", action="a")
    out = d.to_dict()
    assert out["code"] == "tau2_zero" and out["severity"] == "warning"


# --------------------------------------------------------------------------- #
# the checks, each with its silent counterpart
# --------------------------------------------------------------------------- #
def test_tau2_zero_fires_when_the_random_effects_model_collapses():
    # identical effects with identical variances: the between-study variance is
    # exactly zero, not merely small
    yi = np.full(6, 0.4)
    fit = ma.rma(yi, np.full(6, 0.05))
    assert float(fit.tau2) == pytest.approx(0.0, abs=1e-10)
    assert "tau2_zero" in ma.diagnose(fit).codes()


def test_tau2_zero_stays_silent_on_heterogeneous_data():
    # a spread of effects far wider than their standard errors
    yi = np.array([0.10, 0.50, 0.90, 1.30, -0.20, 0.70])
    sei = np.full(yi.size, 0.05)
    fit = ma.rma(yi, sei ** 2)
    assert float(fit.tau2) > 1e-6
    assert "tau2_zero" not in ma.diagnose(fit).codes()


def test_few_studies_fires_below_five():
    fit = ma.rma(np.array([0.1, 0.3, 0.2]), np.array([0.01, 0.02, 0.015]))
    assert "few_studies" in ma.diagnose(fit).codes()


def test_few_studies_stays_silent_with_enough_studies():
    yi, sei = _fixture()
    fit = ma.rma(yi, sei ** 2)
    assert "few_studies" not in ma.diagnose(fit).codes()


def test_posterior_equals_prior_fires_when_the_data_cannot_update_tau():
    """Huge standard errors leave the heterogeneity prior untouched."""
    rng = np.random.default_rng(7)
    sei = np.full(8, 50.0)
    yi = rng.normal(0.3, 1.0, sei.size)
    fit = ma.robma(yi, sei, bias_models=("none",))
    codes = ma.diagnose(fit).codes()
    assert "posterior_equals_prior" in codes


def test_posterior_equals_prior_stays_silent_on_informative_data():
    yi, sei = _fixture()
    fit = ma.robma(yi, sei)
    assert "posterior_equals_prior" not in ma.diagnose(fit).codes()


# --------------------------------------------------------------------------- #
# the container
# --------------------------------------------------------------------------- #
def test_a_clean_fit_reports_nothing():
    yi, sei = _fixture()
    fit = ma.robma(yi, sei)
    diagnosis = ma.diagnose(fit)
    assert diagnosis.worst is None
    assert len(diagnosis) == 0
    assert str(diagnosis) == "no diagnostics"


def test_the_worst_severity_comes_first():
    yi, sei = _fixture()
    fit = ma.robma(yi[:3], sei[:3])
    codes = ma.diagnose(fit).codes()
    if "posterior_equals_prior" in codes and "few_studies" in codes:
        order = [d.code for d in ma.diagnose(fit)]
        assert order.index("posterior_equals_prior") < order.index("few_studies")


# --------------------------------------------------------------------------- #
# conditions read off the fitted arrays
# --------------------------------------------------------------------------- #
def test_non_finite_estimate_fires_and_is_an_error():
    """Tested on a stub: ``rma`` drops non-finite rows before it fits.

    The guard exists for paths that do not pre-clean, and pretending a
    ``rma`` fit can reach it would be testing nothing.
    """
    diagnosis = ma.diagnose(_stub(b=[0.2, float("nan")], se=[0.1, 0.1]))
    assert "non_finite_estimate" in diagnosis.codes()
    assert diagnosis.worst == "error"


def test_non_finite_estimate_stays_silent_on_a_finite_fit():
    assert "non_finite_estimate" not in ma.diagnose(
        _stub(b=[0.2, 0.3], se=[0.1, 0.1])).codes()


def test_zero_weight_study_fires_for_a_study_with_no_weight():
    assert "zero_weight_study" in ma.diagnose(
        _stub(weights=[0.4, 0.0, 0.6])).codes()


def test_zero_weight_study_stays_silent_when_every_study_counts():
    assert "zero_weight_study" not in ma.diagnose(
        _stub(weights=[0.4, 0.2, 0.6])).codes()


def test_studies_dropped_fires_when_the_fit_covers_less_than_the_review():
    diagnosis = ma.diagnose(_stub(k=5, k_all=6))
    assert "studies_dropped" in diagnosis.codes()
    assert diagnosis.items[0].severity == "warning"


def test_studies_dropped_fires_on_a_real_fit_that_dropped_a_row():
    """It could not, until ``rma`` started recording what it removed.

    ``k_all`` used to be set to ``k`` and ``not_na`` to ``[True] * k`` after the
    rows were filtered, so the only evidence of the drop was erased before the
    result existed and this check could never fire outside a stub. The fit now
    keeps the pre-filter count and the keep mask.
    """
    with pytest.warns(ma.MissingDataWarning):
        fit = ma.rma([0.1, float("nan"), 0.3, 0.4],
                     vi=[0.04, 0.05, 0.06, 0.07])
    assert fit.k == 3 and fit.k_all == 4
    assert "studies_dropped" in ma.diagnose(fit).codes()
    policy = fit.missing
    assert policy is not None and policy.dropped == (1,)


def test_studies_dropped_is_silent_when_nothing_was_dropped():
    assert "studies_dropped" not in ma.diagnose(_stub(k=6, k_all=6)).codes()


def test_not_converged_fires_and_is_an_error():
    diagnosis = ma.diagnose(_stub(converged=False))
    assert "not_converged" in diagnosis.codes()
    assert diagnosis.worst == "error"


def test_not_converged_is_silent_when_the_search_converged():
    assert "not_converged" not in ma.diagnose(_stub(converged=True)).codes()


def test_not_converged_is_skipped_when_the_result_has_no_such_flag():
    assert "not_converged" not in ma.diagnose(_stub()).codes()


# --------------------------------------------------------------------------- #
# sampling variances that cannot be variances
# --------------------------------------------------------------------------- #
def test_negative_variance_fires_on_a_real_fit_that_returned_plausible_numbers():
    """The point of the check: the fit does not fail, it produces a number.

    Measured on this exact input before writing the check: pimeta returns
    ``b = 0.2091``, ``se = 0.0786`` from one negative variance among six. No
    warning, no exception, no NaN. A reader has no way to tell.
    """
    yi = np.array([0.1, 0.3, 0.2, 0.5, 0.4, 0.2])
    vi = np.array([-0.01, 0.04, 0.05, 0.03, 0.06, 0.05])
    fit = ma.rma(yi, vi)
    assert np.isfinite(fit.b[0]) and np.isfinite(fit.se[0])
    diagnosis = ma.diagnose(fit)
    assert "negative_variance" in diagnosis.codes()
    assert diagnosis.worst == "error"


def test_negative_variance_is_silent_on_ordinary_data():
    yi = np.array([0.1, 0.3, 0.2, 0.5, 0.4, 0.2])
    assert "negative_variance" not in ma.diagnose(ma.rma(yi, np.full(6, 0.04))).codes()


def test_zero_variance_is_named_as_well_as_the_nan_it_causes():
    """A zero variance has two consequences, and they are different facts.

    The weights become infinite, which is the cause; the estimate becomes NaN,
    which is the symptom. Reporting only the symptom leaves the user hunting for
    a numerical problem in the optimiser.
    """
    yi = np.array([0.1, 0.3, 0.2, 0.5, 0.4, 0.2])
    vi = np.array([0.0, 0.04, 0.05, 0.03, 0.06, 0.05])
    with pytest.warns(RuntimeWarning):      # divide by zero in the weights
        fit = ma.rma(yi, vi)
    diagnosis = ma.diagnose(fit)
    assert {"zero_variance", "non_finite_estimate"} <= set(diagnosis.codes())
    assert diagnosis.worst == "error"


def test_zero_variance_is_silent_when_every_variance_is_positive():
    yi = np.array([0.1, 0.3, 0.2, 0.5, 0.4, 0.2])
    assert "zero_variance" not in ma.diagnose(ma.rma(yi, np.full(6, 0.04))).codes()


def test_variance_checks_are_skipped_when_the_result_carries_no_variances():
    diagnosis = ma.diagnose(_stub())
    assert "negative_variance" not in diagnosis.codes()
    assert "zero_variance" not in diagnosis.codes()


# --------------------------------------------------------------------------- #
# designs and likelihood surfaces
# --------------------------------------------------------------------------- #
def test_near_rank_deficient_fires_on_a_collinear_design():
    """The exact case cannot be diagnosed -- `rma` refuses it -- but this can.

    Two moderators that differ by 1e-7 give a full-rank design with a condition
    number around 1e8: the fit exists, and its coefficients are not separately
    interpretable.
    """
    rng = np.random.default_rng(3)
    y = np.array([0.1, 0.3, 0.2, 0.5, 0.4, 0.2, 0.35, 0.45])
    x1 = np.arange(8.0)
    x2 = x1 + 1e-7 * rng.normal(size=8)
    fit = ma.rma(y, vi=np.full(8, 0.04), mods=np.column_stack([x1, x2]))
    assert np.linalg.cond(fit.X) > 1e6
    assert "near_rank_deficient" in ma.diagnose(fit).codes()


def test_near_rank_deficient_is_silent_on_an_ordinary_design():
    y = np.array([0.1, 0.3, 0.2, 0.5, 0.4, 0.2, 0.35, 0.45])
    x = np.arange(8.0)
    fit = ma.rma(y, vi=np.full(8, 0.04), mods=np.column_stack([x, x ** 2]))
    assert "near_rank_deficient" not in ma.diagnose(fit).codes()


def test_near_rank_deficient_is_skipped_without_a_design():
    assert "near_rank_deficient" not in ma.diagnose(_stub()).codes()


def test_flat_likelihood_is_silent_on_a_heterogeneous_fit():
    yi = np.array([0.10, 0.50, 0.90, 1.30, -0.20, 0.70])
    fit = ma.rma(yi, vi=np.full(6, 0.0025))
    assert float(fit.tau2) > 1e-6
    assert "flat_likelihood" not in ma.diagnose(fit).codes()


def test_flat_likelihood_fires_when_the_surface_is_concave_up(monkeypatch):
    """A curvature that is not negative means the estimate is not a maximum.

    The check is a scalar here because the location coefficients are profiled
    out, so the whole Hessian is one number. It is exercised with a patched
    likelihood because a real fit cannot easily be made to stop on a rising
    surface.
    """
    import pimeta.checks as checks
    import pimeta.pooling as pooling

    calls = {"n": 0}

    def fake_loglik(y, v, X, tau2, method):
        calls["n"] += 1
        # concave up in tau2: the log-likelihood rises as tau2 moves away
        return (float(tau2) - 0.05) ** 2

    monkeypatch.setattr(pooling, "_reml_ml_loglik", fake_loglik)
    stub = _stub(tau2=0.2, yi=np.array([0.1, 0.2, 0.3, 0.4]),
                 vi=np.full(4, 0.05), X=np.ones((4, 1)), method="REML")
    diagnosis = checks.diagnose(stub)
    assert "flat_likelihood" in diagnosis.codes()
    assert calls["n"] == 3          # exactly the three evaluations it needs


def test_flat_likelihood_is_skipped_at_the_boundary():
    # tau2 = 0 is the boundary case, covered by tau2_zero, not by curvature
    stub = _stub(tau2=0.0, yi=np.array([0.1, 0.2, 0.3, 0.4]),
                 vi=np.full(4, 0.05), X=np.ones((4, 1)), method="REML")
    assert "flat_likelihood" not in ma.diagnose(stub).codes()
