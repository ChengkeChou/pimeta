"""The confidence-distribution prediction interval.

The measured behaviour, at 2000 replications per scenario against the true effect
of a new study drawn from ``N(mu, tau^2)`` (``tools/coverage_pi.py``):

===========  ====  =============  =========  =========
scenario     k     plug-in        t          cd
===========  ====  =============  =========  =========
null         20    0.882 ± 0.007  0.899      0.949 ± 0.005
tiny         5     0.804 ± 0.009  0.893      0.986 ± 0.003
unequal      20    0.883 ± 0.007  0.903      0.947 ± 0.005
===========  ====  =============  =========  =========

So the CD interval is calibrated at ``k = 20`` and **conservative** at ``k = 5``.
That is the reason it is offered rather than imposed: it never under-covers,
which is the property that matters most, but at five studies it is roughly 25%
wider than the coverage requires.
"""

from __future__ import annotations

import numpy as np
import pytest

import pimeta as ma
from pimeta.predict_cd import cd_prediction_interval, heterogeneity_confidence_distribution

YI = np.array([0.10, 0.45, 0.30, 0.62, 0.20])
VI = np.array([0.04, 0.05, 0.03, 0.06, 0.05])


def test_the_interval_brackets_the_pooled_estimate():
    out = cd_prediction_interval(YI, VI)
    assert out["pi_lb"] < out["pred"] < out["pi_ub"]


def test_the_interval_is_wider_than_the_t_reference_at_small_k():
    # the point of the method: the t interval still treats tau2 as known
    fit = ma.rma(YI, vi=VI)
    cd = fit.predict(predtype="cd")
    t = fit.predict(predtype="t")
    assert (cd.pi_ub - cd.pi_lb) > (t.pi_ub - t.pi_lb)


def test_the_q_statistic_is_strictly_decreasing_in_tau2():
    # the inversion in the module assumes it; if this ever fails the bisection
    # silently returns a wrong root rather than raising
    from pimeta.predict_cd import _q_stat

    grid = np.array([0.0, 0.01, 0.1, 1.0, 10.0, 100.0])
    q = _q_stat(YI, VI, grid)
    assert np.all(np.diff(q) < 0.0)


def test_the_confidence_distribution_keeps_mass_above_zero_when_tau2_is_zero():
    """The failure the plug-in interval cannot escape.

    Mildly inconsistent effects with large sampling variances give a point
    estimate of exactly zero heterogeneity, while the pivot still puts
    probability on tau2 > 0 -- which is precisely the tail the parametric
    bootstrap loses when it resamples from tau2_hat = 0.
    """
    yi = np.array([0.30, 0.42, 0.36, 0.48, 0.34])
    vi = np.full(5, 0.05)
    fit = ma.rma(yi, vi=vi)
    assert float(fit.tau2) == pytest.approx(0.0, abs=1e-12)
    dist = heterogeneity_confidence_distribution(yi, vi)
    assert 0.0 < dist["point_mass"] < 1.0
    assert float(np.max(dist["tau2"])) > 0.0
    out = cd_prediction_interval(yi, vi)
    assert out["pi_ub"] > out["pred"]


def test_perfectly_identical_studies_put_all_the_mass_at_zero():
    # Q(0) = 0 exactly: no evidence of heterogeneity at all, so the confidence
    # distribution is a point mass and the interval is the t interval on tau2=0
    yi = np.full(6, 0.4)
    vi = np.full(6, 0.05)
    dist = heterogeneity_confidence_distribution(yi, vi)
    assert dist["point_mass"] == pytest.approx(1.0)
    assert float(np.max(dist["tau2"])) == 0.0


def test_the_interval_widens_when_the_data_disagree():
    tight = cd_prediction_interval(np.array([0.40, 0.42, 0.38, 0.41, 0.39]),
                                   np.full(5, 0.04))
    wide = cd_prediction_interval(np.array([0.10, 0.55, 0.20, 0.70, 0.30]),
                                  np.full(5, 0.04))
    assert (wide["pi_ub"] - wide["pi_lb"]) > (tight["pi_ub"] - tight["pi_lb"])


def test_fewer_than_three_studies_is_refused():
    with pytest.raises(ValueError, match="three studies"):
        cd_prediction_interval([0.1, 0.2], [0.04, 0.05])


def test_a_non_positive_variance_is_refused():
    with pytest.raises(ValueError, match="positive"):
        cd_prediction_interval([0.1, 0.2, 0.3], [0.04, 0.0, 0.05])


def test_mismatched_lengths_are_refused():
    with pytest.raises(ValueError, match="same length"):
        cd_prediction_interval([0.1, 0.2, 0.3], [0.04, 0.05])


def test_the_level_can_be_given_as_a_proportion_or_a_percentage():
    a = cd_prediction_interval(YI, VI, level=0.95)
    b = cd_prediction_interval(YI, VI, level=95)
    assert a["pi_lb"] == pytest.approx(b["pi_lb"], rel=1e-12)


def test_predict_returns_the_same_numbers_as_the_function():
    fit = ma.rma(YI, vi=VI)
    out = fit.predict(predtype="cd")
    direct = cd_prediction_interval(YI, VI)
    assert out.pi_lb == pytest.approx(direct["pi_lb"], rel=1e-12)
    assert out.pi_ub == pytest.approx(direct["pi_ub"], rel=1e-12)
    assert out.pi_dist == "cd"


def test_a_prediction_at_new_moderators_refuses_the_cd_interval():
    # There is no confidence distribution for tau2 conditional on a new
    # moderator value, so the interval would be an approximation wearing the
    # same label.
    fit = ma.rma([0.1, 0.3, 0.2, 0.5, 0.4], vi=np.full(5, 0.04),
                 mods=np.arange(5.0))
    with pytest.raises(ValueError, match="intercept-only"):
        fit.predict(newmods=[2.0], predtype="cd")


def test_the_point_mass_is_reported():
    out = cd_prediction_interval(np.full(6, 0.4), np.full(6, 0.05))
    assert out["point_mass"] > 0.9
    assert np.isfinite(out["tau2_median"])
