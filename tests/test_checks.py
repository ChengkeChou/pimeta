"""Guards for the structured diagnostics.

Every check needs two tests, and the second is the one that is usually missing:
one that builds the condition and asserts the code fires, and one that asserts it
does **not** fire on a clean fit. A diagnostic that only ever fires is as
useless as one that never does.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import pimeta as ma


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
