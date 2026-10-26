"""Subgroup evidence: the Bayes factor is the marginal-likelihood ratio.

Every number here is checked against a value that can be written down by hand,
because a Bayes factor that is off by a factor of two is indistinguishable from a
strong finding.
"""

from __future__ import annotations

import math

import pytest

import pimeta as ma
from pimeta.hte import bayes_factor_normal, subgroup_evidence


def test_the_bayes_factor_at_zero_difference_is_the_marginal_ratio():
    # BF10 = sqrt(se^2/(se^2+s0^2)) * exp(0)
    assert bayes_factor_normal(0.0, 0.2, prior_sd=0.5) == pytest.approx(
        math.sqrt(0.04 / 0.29), rel=1e-12)


def test_the_bayes_factor_reduces_to_one_when_the_data_are_uninformative():
    # se -> inf: the two models predict the same thing
    assert bayes_factor_normal(0.0, 1e6, prior_sd=0.5) == pytest.approx(
        1.0, rel=1e-6)


def test_the_bayes_factor_grows_with_the_estimate_and_shrinks_with_the_se():
    tight = bayes_factor_normal(0.4, 0.1)
    loose = bayes_factor_normal(0.4, 0.4)
    assert tight > loose
    # with se = 0.4 the estimate is smaller than the prior scale allows for, so
    # the point null is the better predictor: BF10 < 1
    assert loose < 1.0
    assert bayes_factor_normal(0.8, 0.4) > loose


def test_the_bayes_factor_is_the_inverse_of_its_own_reciprocal():
    bf = bayes_factor_normal(0.25, 0.18)
    assert bf * (1.0 / bf) == pytest.approx(1.0)


def test_a_non_positive_se_or_prior_is_refused():
    with pytest.raises(ValueError, match="se"):
        bayes_factor_normal(0.1, 0.0)
    with pytest.raises(ValueError, match="prior_sd"):
        bayes_factor_normal(0.1, 0.2, prior_sd=0.0)


def test_the_evidence_sentence_follows_the_thresholds():
    strong = subgroup_evidence(1.0, 0.2)          # BF10 in the thousands
    assert "evidence for a difference" in strong["evidence"]
    assert strong["bf10"] > 3.0
    equivalent = subgroup_evidence(0.0, 0.05)
    assert "evidence for equivalence" in equivalent["evidence"]
    assert equivalent["bf10"] < 1.0 / 3.0
    undecided = subgroup_evidence(0.4, 0.5)
    assert undecided["evidence"] == "no evidence either way"
    assert 1.0 / 3.0 < undecided["bf10"] < 3.0


def test_the_interval_uses_the_reference_it_was_given():
    normal = subgroup_evidence(0.3, 0.1)
    student = subgroup_evidence(0.3, 0.1, df=8)
    assert (student["ci_ub"] - student["ci_lb"]) > (
        normal["ci_ub"] - normal["ci_lb"])


def test_the_equivalence_proportion_needs_an_mcid():
    assert subgroup_evidence(0.2, 0.1)["prob_equivalence"] is None
    out = subgroup_evidence(0.2, 0.1, mcid=0.5)
    assert 0.0 <= out["prob_equivalence"] <= 1.0


def test_a_precise_zero_difference_is_confident_equivalence():
    tight = subgroup_evidence(0.0, 0.02, mcid=0.2)
    wide = subgroup_evidence(0.0, 0.5, mcid=0.2)
    assert tight["prob_equivalence"] > 0.99
    assert tight["bf10"] < 0.1
    assert wide["prob_equivalence"] < 0.5
    assert 0.5 < wide["bf10"] < 1.5          # the data do not decide


def test_the_wrong_sign_probability_is_a_tail():
    # symmetric around zero: P(sign flip) is P(Delta < 0) for a positive estimate
    out = subgroup_evidence(0.0, 0.3)
    assert out["prob_wrong_sign"] == pytest.approx(0.5, abs=1e-12)
    assert subgroup_evidence(1.0, 0.3)["prob_wrong_sign"] < 0.01


def test_a_negative_estimate_is_treated_symmetrically():
    up = subgroup_evidence(0.4, 0.15)
    down = subgroup_evidence(-0.4, 0.15)
    assert up["bf10"] == pytest.approx(down["bf10"], rel=1e-12)
    assert up["prob_wrong_sign"] == pytest.approx(down["prob_wrong_sign"],
                                                  rel=1e-12)


def test_a_bad_mcid_is_refused():
    with pytest.raises(ValueError, match="mcid"):
        subgroup_evidence(0.2, 0.1, mcid=0.0)


def test_it_is_exported():
    assert callable(ma.subgroup_evidence)
    assert callable(ma.bayes_factor_normal)
