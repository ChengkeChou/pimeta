"""The Copas likelihood region: what the data cannot rule out.

``metasens`` reports one adjusted number, chosen by walking a path drawn by
hand down a contour plot.  ``ma.copas_region`` keeps the same selection model
but reports the piece of it that carries inference: the set of
``(gamma0, gamma1)`` whose profile likelihood clears the chi-square-2
threshold, and the range of the adjusted effect over that set.

What is checked here is that the region *is* that likelihood set — the
membership rule, the threshold, the best point, monotonicity in the level —
and that it behaves the way the model says it should: a dataset with no funnel
asymmetry leaves almost no uncertainty about the selection scenario, and one
with strong asymmetry leaves a lot.

What is **not** checked here: the coverage of the projected interval.  A
projection of a joint confidence set is conservative by construction — it is
at least as wide as the profile for a single coordinate — but proving the
coverage would take a simulation study of its own, and claiming it without one
would be the sort of thing this package is supposed to avoid.
"""

import numpy as np
import pytest
from scipy import stats

import pimeta as ma
from pimeta.copas import copas_region

SE = np.array([0.050, 0.070, 0.090, 0.110, 0.130, 0.150,
               0.180, 0.210, 0.250, 0.300, 0.350, 0.400])
WIGGLE = np.array([0.010, -0.020, 0.015, -0.012, 0.018, -0.015,
                   0.012, -0.010, 0.014, -0.011, 0.013, -0.009])

_CACHE: dict = {}


def _fit(kind):
    if kind not in _CACHE:
        if kind == "asymmetric":
            yi = 0.60 - 1.20 * SE + WIGGLE
        elif kind == "symmetric":
            yi = 0.45 + 0.3 * WIGGLE
        else:                                # pragma: no cover
            raise ValueError(kind)
        _CACHE[kind] = ma.copas(ma.rma(yi, vi=SE * SE, method="REML"))
    return _CACHE[kind]


# --------------------------------------------------------------------------- #
# the region is the likelihood set, by construction
# --------------------------------------------------------------------------- #

def test_membership_follows_the_likelihood_ratio_rule():
    fit = _fit("asymmetric")
    region = copas_region(fit, level=95)
    ll = np.asarray(fit.loglik_contour, dtype=float)
    ll_max = float(np.max(ll))
    inside_by_rule = 2.0 * (ll_max - ll) <= region.crit
    # every returned point satisfies the rule, and the count matches
    g0 = np.broadcast_to(np.asarray(fit.gamma0)[:, None], ll.shape)
    g1 = np.broadcast_to(np.asarray(fit.gamma1)[None, :], ll.shape)
    for a, b in zip(region.gamma0_in, region.gamma1_in, strict=True):
        i = int(np.argmin(np.abs(np.asarray(fit.gamma0) - a)))
        j = int(np.argmin(np.abs(np.asarray(fit.gamma1) - b)))
        assert 2.0 * (ll_max - ll[i, j]) <= region.crit
    assert region.mu_in.size == int(np.sum(inside_by_rule & np.isfinite(ll)))
    assert g0.shape == g1.shape


def test_the_threshold_is_the_chi_square_two_quantile():
    region = copas_region(_fit("asymmetric"), level=95)
    assert region.crit == pytest.approx(stats.chi2.ppf(0.95, 2), rel=1e-12)


def test_a_higher_level_gives_a_fatter_region():
    fit = _fit("asymmetric")
    narrow = copas_region(fit, level=68)
    wide = copas_region(fit, level=99)
    assert narrow.crit < wide.crit
    assert narrow.mu_in.size < wide.mu_in.size
    assert (narrow.mu_ub - narrow.mu_lb) <= (wide.mu_ub - wide.mu_lb) + 1e-12


def test_the_best_fit_is_inside_and_anchors_the_interval():
    region = copas_region(_fit("asymmetric"))
    assert region.mu_lb <= region.mu_best <= region.mu_ub
    assert region.mu_lb <= region.mu_ub


def test_the_projected_interval_brackets_every_admissible_effect():
    region = copas_region(_fit("asymmetric"))
    assert np.all(np.asarray(region.mu_in) >= region.mu_lb - 1e-12)
    assert np.all(np.asarray(region.mu_in) <= region.mu_ub + 1e-12)


# --------------------------------------------------------------------------- #
# the model's own predictions
# --------------------------------------------------------------------------- #

def test_without_funnel_asymmetry_the_selection_scenario_barely_matters():
    """No asymmetry means nothing to explain, so the effect cannot move."""
    region = copas_region(_fit("symmetric"))
    assert (region.mu_ub - region.mu_lb) < 0.01
    assert region.no_selection_inside


def test_with_asymmetry_the_interval_is_wider():
    symmetric = copas_region(_fit("symmetric"))
    asymmetric = copas_region(_fit("asymmetric"))
    assert (asymmetric.mu_ub - asymmetric.mu_lb) > \
        (symmetric.mu_ub - symmetric.mu_lb)


def test_no_precision_dependent_selection_is_admissible_for_both():
    """Neither dataset is skewed enough to rule selection out entirely."""
    for kind in ("asymmetric", "symmetric"):
        region = copas_region(_fit(kind))
        assert region.no_selection_inside is True
        assert region.gamma1_min == pytest.approx(0.0, abs=1e-12)


def test_a_clear_effect_stays_on_one_side_of_zero():
    region = copas_region(_fit("asymmetric"))
    assert region.sign_stable is True


def test_publication_probability_is_reported_on_a_probability_scale():
    region = copas_region(_fit("asymmetric"))
    assert 0.0 < region.publprob_lb <= region.publprob_ub <= 1.0


# --------------------------------------------------------------------------- #
# interface
# --------------------------------------------------------------------------- #

def test_level_accepts_percent_and_proportion():
    fit = _fit("asymmetric")
    a = copas_region(fit, level=95)
    b = copas_region(fit, level=0.95)
    assert a.mu_lb == pytest.approx(b.mu_lb, rel=1e-12)
    assert a.crit == pytest.approx(b.crit, rel=1e-12)


def test_an_impossible_level_is_rejected():
    with pytest.raises(ValueError, match="level"):
        copas_region(_fit("asymmetric"), level=0)


def test_the_region_carries_nothing_it_did_not_compute():
    region = copas_region(_fit("asymmetric"))
    payload = region.to_dict()
    assert payload["mu_lb"] == region.mu_lb
    assert len(payload["gamma0_in"]) == len(payload["mu_in"])
    assert "Copas selection region" in str(region)


def test_the_region_round_trips_through_a_record(tmp_path):
    from pimeta.record import load_record, save_record

    region = copas_region(_fit("asymmetric"))
    path = tmp_path / "copas_region.json"
    save_record(path, region, mode="digest", function="copas_region")
    loaded = load_record(path)
    assert loaded.kind == "copas_region"
    assert loaded.results["values"]["mu_lb"] == pytest.approx(
        region.mu_lb, rel=1e-12)
    assert np.asarray(loaded.results["values"]["mu_in"]) == pytest.approx(
        np.asarray(region.mu_in), rel=1e-12)
