"""The coverage baseline has to be reproducible by the shipped code, not just quoted.

``tests/test_coverage_baselines.py`` holds the docstrings to the committed
numbers. That leaves one gap: nothing checks that the numbers can still be
*produced*. A refactor that quietly breaks the bootstrap region or the LKJ
penalty would leave the baseline untouched and the docstrings honest about a
past that no longer exists.

So this module re-runs both harnesses at a size the default suite can afford --
12 replications instead of 60 to 100 -- and asserts the same orderings. At 12
replications the Monte Carlo standard error on a rate is about 0.13, which is far
too coarse to check a level and exactly right to check a *direction*: the two
things measured here are that the bootstrap cutoff exceeds the chi-square one on
average, and that the LKJ penalty moves a boundary-prone fit off the boundary.

The full measurements stay in ``tools/coverage_*.py`` with the replication
counts the baseline records.
"""

from __future__ import annotations

import warnings

import numpy as np
import pytest

import pimeta as ma
from tools import coverage_ipd_curve as band_harness
from tools import coverage_predict_region as region_harness

REPS = 12


@pytest.fixture(scope="module")
def quiet():
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        yield


def test_the_ipd_harness_reproduces_the_lkj_direction(quiet):
    """LKJ(2) must reduce the boundary rate and not make the edge worse.

    At 12 replications the boundary *rate* is noise; the count of fitting
    studies is not, and neither is the five-covariate coverage vector's shape.
    """
    plain = _band_reps(0.0)
    penalised = _band_reps(2.0)
    assert plain["fitted"] >= 10 and penalised["fitted"] >= 10
    assert penalised["boundary"] <= plain["boundary"]
    # every covariate keeps a coverage in the plausible range for 12 reps; this
    # is a smoke test of the machinery, not a second measurement
    for value in penalised["coverage"]:
        assert 0.5 <= value <= 1.0


def _band_reps(eta: float) -> dict:
    hits = np.zeros(len(band_harness.GRID))
    truth = band_harness.truth(band_harness.GRID)
    fitted = 0
    boundary = 0
    for rep in range(REPS):
        study, age, treat, y = band_harness.simulate(rep)
        try:
            fit = ma.ipd_two_stage_curve(
                y, study=study, treat=treat, covariate=age,
                knots=band_harness.KNOTS, test="z", lkj_eta=eta)
        except ValueError:                       # pragma: no cover - unstable fit
            continue
        fitted += 1
        boundary += bool(fit.identifiability()["boundary"])
        curve = fit.curve(band_harness.GRID, robust=True, df="point")
        hits += (curve.ci_lb <= truth) & (truth <= curve.ci_ub)
    return {"fitted": fitted,
            "boundary": boundary,
            "coverage": hits / max(fitted, 1)}


def test_the_region_harness_reproduces_the_bootstrap_direction(quiet):
    """The calibrated cutoff must be larger than the chi-square one.

    ``test="bootstrap"`` exists precisely because the closed-form references
    treat ``Sigma`` as known; if a change ever made the bootstrap cutoff equal to
    or smaller than the chi-square one, the reference would have stopped doing
    the one thing it is for.
    """
    rng = np.random.default_rng(0)
    wider = 0
    compared = 0
    for _ in range(REPS):
        fit, _fresh, V, X, data = region_harness.one_fit(rng, 10)
        crit = region_harness.bootstrap_critical(
            fit, V, X, data, level=0.95, n_boot=25, rng=rng, k=10)
        if not np.isfinite(crit):
            continue
        compared += 1
        wider += crit > ma.predict_region(fit, test="z").crit
    assert compared >= 10
    assert wider >= compared - 1, (
        f"the bootstrap cutoff exceeded chi-square in only {wider} of "
        f"{compared} fits; the calibration has stopped doing its job")
