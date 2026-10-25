"""The versioned export contract, checked on the two result types that have it.

The point of these three fields is that a saved result can be read and judged
later without the code that produced it. A test that only checks the keys exist
would not notice if the diagnostics block were silently dropped for fits that
have nothing to report, so the clean case is checked explicitly.
"""

from __future__ import annotations

import numpy as np
import pytest

import pimeta as ma
from pimeta.checks import SCHEMA_VERSION


@pytest.fixture(scope="module")
def univariate():
    # a genuinely heterogeneous spread with small standard errors, so tau2 is
    # clearly positive and nothing fires; a tight cluster of similar effects
    # estimates tau2 at zero and tau2_zero would (correctly) report it
    yi = np.array([0.10, 0.50, 0.90, 1.30, -0.20, 0.70])
    vi = np.full(yi.size, 0.0025)
    return ma.rma(yi, vi)


def test_the_univariate_export_carries_version_provenance_and_diagnostics(
        univariate):
    out = univariate.to_dict()
    assert out["schema_version"] == SCHEMA_VERSION
    assert set(out["software"]) == {"pimeta", "python", "numpy", "scipy"}
    assert "diagnostics" in out
    assert set(out["diagnostics"]) == {"worst", "items"}


def test_a_clean_univariate_fit_reports_an_empty_block_not_a_missing_key(
        univariate):
    out = univariate.to_dict()
    assert out["diagnostics"]["worst"] is None
    assert out["diagnostics"]["items"] == []


def test_the_univariate_export_still_has_its_own_fields(univariate):
    """The three new keys must not have displaced anything."""
    out = univariate.to_dict()
    assert {"b", "se", "tau2", "k", "method"} <= set(out)


def test_a_boundary_univariate_fit_says_so_in_the_export():
    yi = np.full(6, 0.4)
    fit = ma.rma(yi, np.full(6, 0.05))
    assert "tau2_zero" in fit.diagnostics.codes()
    assert fit.to_dict()["diagnostics"]["worst"] == "warning"


def test_the_schema_version_is_shared_by_both_result_types(univariate):
    yi = np.array([0.2, 0.4, 0.3, 0.5])
    sei = np.array([0.1, 0.1, 0.1, 0.1])
    robma_fit = ma.robma(yi, sei)
    assert (univariate.to_dict()["schema_version"]
            == robma_fit.to_dict()["schema_version"] == SCHEMA_VERSION)
