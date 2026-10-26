"""Declared missing-data handling: the assumption is part of the answer.

The tests here check three things: that the strategy names and their stated
assumptions stay in step, that the policy refuses to describe an impossible
situation, and that a real fit's recorded handling matches what the fit actually
did. The last one matters most -- a policy derived from the wrong counts would be
worse than no policy, because it would be printed next to the estimate as if it
had been observed.
"""

from __future__ import annotations

import numpy as np
import pytest

import pimeta as ma
from pimeta.missing_policy import (
    STRATEGIES,
    MissingPolicy,
    complete_case_policy,
    policy_from_result,
)

# --------------------------------------------------------------- the strategies

def test_every_strategy_states_its_assumption():
    assert set(STRATEGIES) == {"complete_case", "available_case", "likelihood"}
    for name, text in STRATEGIES.items():
        assert len(text) > 30, name
        assert text[0].islower()


def test_the_assumptions_are_not_interchangeable():
    # If two strategies ever get the same sentence, one of them has stopped
    # meaning anything.
    assert len(set(STRATEGIES.values())) == len(STRATEGIES)


def test_multiple_imputation_is_not_offered_as_an_equivalent():
    assert "multiple_imputation" not in STRATEGIES
    assert "mi" not in STRATEGIES


# ------------------------------------------------------------------ the policy

def test_a_complete_policy_reports_nothing_dropped():
    policy = MissingPolicy(strategy="complete_case", n_total=5, n_used=5)
    assert policy.is_complete
    assert policy.n_dropped == 0
    assert "nothing dropped" in str(policy)


def test_a_policy_with_drops_names_the_rows_one_based():
    policy = MissingPolicy(strategy="complete_case", n_total=5, n_used=3,
                           dropped=(0, 3), mechanism="MAR")
    text = str(policy)
    assert "2 of 5" in text or "3 of 5" in text
    assert "rows 1, 4" in text
    assert "MAR" in text


def test_an_unknown_strategy_is_refused():
    with pytest.raises(ValueError, match="unknown strategy"):
        MissingPolicy(strategy="dropna", n_total=5, n_used=5)


def test_an_unknown_mechanism_is_refused():
    with pytest.raises(ValueError, match="unknown mechanism"):
        MissingPolicy(strategy="complete_case", n_total=5, n_used=5,
                      mechanism="probably fine")


def test_more_used_than_offered_is_refused():
    with pytest.raises(ValueError, match="n_used"):
        MissingPolicy(strategy="complete_case", n_total=4, n_used=5)


def test_the_default_mechanism_is_unknown_rather_than_optimistic():
    policy = MissingPolicy(strategy="complete_case", n_total=3, n_used=3)
    assert policy.mechanism == "unknown"


def test_the_policy_round_trips_through_a_dict():
    policy = MissingPolicy(strategy="likelihood", n_total=10, n_used=10,
                           mechanism="MAR", note="FIML over the missing arms")
    payload = policy.to_dict()
    assert payload["strategy"] == "likelihood"
    assert payload["n_dropped"] == 0
    assert "integrates" in payload["assumption"]
    assert payload["note"].startswith("FIML")


# ------------------------------------------------- deriving it from the columns

def test_complete_case_policy_counts_what_it_would_drop():
    columns = {"yi": [0.1, np.nan, 0.3, 0.4],
               "vi": [0.04, 0.05, np.nan, 0.08]}
    policy = complete_case_policy(columns)
    assert policy.n_total == 4
    assert policy.n_used == 2
    assert policy.dropped == (1, 2)
    assert "yi, vi" in policy.note


def test_a_clean_frame_declares_nothing_dropped():
    policy = complete_case_policy({"yi": [0.1, 0.2], "vi": [0.04, 0.05]})
    assert policy.is_complete
    assert policy.note == ""


def test_infinities_count_as_missing():
    policy = complete_case_policy({"yi": [0.1, np.inf, 0.3]})
    assert policy.dropped == (1,)


# --------------------------------------------------------- from a real result

def test_a_clean_fit_declares_a_complete_policy():
    fit = ma.rma([0.1, 0.2, 0.3], vi=[0.04, 0.05, 0.06])
    policy = fit.missing
    assert policy is not None
    assert policy.strategy == "complete_case"
    assert policy.is_complete
    assert policy.n_total == policy.n_used == 3


def test_a_fit_that_dropped_rows_declares_exactly_those_rows():
    yi = [0.1, np.nan, 0.3, 0.4]
    vi = [0.04, 0.05, 0.06, 0.07]
    with pytest.warns(ma.MissingDataWarning):
        fit = ma.rma(yi, vi=vi)
    policy = fit.missing
    assert policy.n_total == 4
    assert policy.n_used == 3
    assert policy.dropped == (1,)
    assert policy.mechanism == "unknown"


def test_the_declared_policy_agrees_with_the_fit_it_describes():
    yi = [0.1, 0.2, np.nan, 0.4]
    vi = [0.04, 0.05, 0.06, 0.07]
    with pytest.warns(ma.MissingDataWarning):
        fit = ma.rma(yi, vi=vi)
    policy = fit.missing
    assert policy.n_used == fit.k
    assert policy.n_total == fit.k_all


def test_a_result_without_counts_gets_no_policy_rather_than_a_guess():
    class Bare:
        pass

    assert policy_from_result(Bare()) is None


def test_impossible_counts_get_no_policy():
    class Odd:
        k_all = 3
        k = 5

    assert policy_from_result(Odd()) is None
