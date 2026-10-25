"""The selection kernel's bin probabilities, pinned by arithmetic.

These are the tests that were missing. The twelve-model Gaussian ensemble
matched the reference exactly, and that agreement said nothing about the
selection models: nothing asserted that a bin probability was non-negative,
that the bins summed to one, or that the vector agreed with a hand computation.

The bug they now catch: differencing ``(1, q, 0)`` backwards instead of
``(0, q, 1)`` forwards, which reversed two bins and, with three steps, produced
a *negative* probability. A negative bin probability does not make a likelihood
infinite, only wrong, so no fit complained.
"""

from __future__ import annotations

import numpy as np
import pytest
from scipy import stats

import pimeta as ma
from pimeta.robma import _bin_probabilities


def _bins(side, steps, mu=0.0, tau2=0.0, sei=1.0):
    wf = ma.step_weight_function(side, steps)
    out = _bin_probabilities(np.array(mu), np.array(sei),
                             np.array(np.sqrt(sei ** 2 + tau2)), wf)
    return np.asarray(out, dtype=float).ravel()


def test_one_sided_two_bins_matches_the_hand_computation():
    """mu = 0, s = 1, steps = (0.05): 5% below 1.645, 95% above."""
    assert _bins("one-sided", (0.05,)) == pytest.approx([0.05, 0.95])


def test_one_sided_three_bins_matches_the_hand_computation():
    """The case whose middle bin used to be negative."""
    assert _bins("one-sided", (0.025, 0.05)) == pytest.approx(
        [0.025, 0.025, 0.95])


def test_two_sided_two_bins_matches_the_hand_computation():
    """A two-sided 0.05 cut is 1.96 standard deviations, not 1.645."""
    assert _bins("two-sided", (0.05,)) == pytest.approx([0.05, 0.95])


def test_two_sided_three_bins_matches_the_hand_computation():
    assert _bins("two-sided", (0.025, 0.05)) == pytest.approx(
        [0.025, 0.025, 0.95])


@pytest.mark.parametrize("side", ["one-sided", "two-sided"])
@pytest.mark.parametrize("steps", [(0.05,), (0.025, 0.05),
                                   (0.025, 0.05, 0.5)])
@pytest.mark.parametrize("mu", [-1.0, -0.2, 0.0, 0.3, 1.5])
@pytest.mark.parametrize("tau2", [0.0, 0.01, 0.2])
def test_bin_probabilities_are_a_distribution(side, steps, mu, tau2):
    """Non-negative and summing to one, for any mean, spread and cut points."""
    bins = _bins(side, steps, mu=mu, tau2=tau2)
    assert np.all(bins >= -1e-12), bins
    assert float(np.sum(bins)) == pytest.approx(1.0, abs=1e-12)


def test_the_bins_are_ordered_from_the_most_significant_end():
    """The first bin is the extreme tail, so it is the smallest of the three."""
    bins = _bins("one-sided", (0.025, 0.05))
    assert bins[0] <= bins[-1]


def test_a_shifted_mean_empties_the_most_significant_bin():
    """A mean far below zero puts almost nothing in the significant tail."""
    bins = _bins("one-sided", (0.05,), mu=-10.0)
    assert bins[0] == pytest.approx(0.0, abs=1e-12)
    assert bins[1] == pytest.approx(1.0, abs=1e-12)


def test_the_observed_bin_agrees_with_the_probability_scale():
    """A study whose p-value equals a cut lands in the bin the cuts define."""
    wf = ma.step_weight_function("one-sided", (0.05,))
    z = stats.norm.isf(0.05)
    yi = np.array([z + 0.01, z - 0.01])
    sei = np.ones(2)
    from pimeta.robma import _observed_bin

    assert list(_observed_bin(yi, sei, wf)) == [0, 1]
