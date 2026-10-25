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


def test_studies_dropped_is_silent_when_nothing_was_dropped():
    assert "studies_dropped" not in ma.diagnose(_stub(k=6, k_all=6)).codes()
