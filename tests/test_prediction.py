"""Prediction-model performance: the right scale and variance per metric.

The pooling itself is `rma`, which is verified against metafor elsewhere. What
is checked here is the part that is specific to prediction models:

* the logit / log back-transformations, exactly;
* the O:E variance, which is `1/O + 1/E` by construction;
* the Hanley–McNeil variance of an AUC, against a **simulation** — the only
  check that can catch a wrong constant, since re-writing the same formula in
  the test would prove nothing.
"""

from __future__ import annotations

import numpy as np
import pytest

import pimeta as ma

AUC = [0.75, 0.80, 0.72]
N_EVENT = [100, 60, 140]
N_NONEVENT = [400, 300, 500]


# --------------------------------------------------------------------------- #
# the transforms
# --------------------------------------------------------------------------- #
def test_the_auc_is_pooled_on_the_logit_scale():
    yi, vi = ma.auc_logit(AUC, n_event=N_EVENT, n_nonevent=N_NONEVENT)
    assert yi == pytest.approx(np.log(np.array(AUC) / (1 - np.array(AUC))),
                               rel=1e-14)
    assert np.all(vi > 0)
    # The smallest study (60 events, 300 non-events) carries the largest
    # variance; the largest (140 / 500) the smallest.
    assert vi[1] > vi[0] > vi[2]


def test_the_oe_ratio_uses_the_poisson_variance():
    yi, vi = ma.oe_ratio([120.0, 80.0], [100.0, 90.0])
    assert yi == pytest.approx([np.log(1.2), np.log(80 / 90)], rel=1e-14)
    assert vi == pytest.approx([1 / 120 + 1 / 100, 1 / 80 + 1 / 90], rel=1e-14)


def test_auc_logit_rejects_an_impossible_input():
    with pytest.raises(ValueError, match="lie in"):
        ma.auc_logit([1.0], n_event=[10], n_nonevent=[10])
    with pytest.raises(ValueError, match="positive"):
        ma.auc_logit([0.7], n_event=[0], n_nonevent=[10])
    with pytest.raises(ValueError, match="same length"):
        ma.auc_logit([0.7, 0.8], n_event=[10], n_nonevent=[10, 10])


def test_oe_ratio_rejects_a_zero_count():
    with pytest.raises(ValueError, match="positive"):
        ma.oe_ratio([0.0], [10.0])


# --------------------------------------------------------------------------- #
# the variance of an AUC, by simulation
# --------------------------------------------------------------------------- #
def test_the_hanley_mcneil_variance_is_the_sampling_variance():
    """Simulate logit-AUC replicates and compare their spread with our SE.

    The formula is analytic; this is the only way to know it is the right one
    rather than merely the same one written twice.
    """
    rng = np.random.default_rng(20261023)
    n0, n1 = 400, 100
    reps = 400
    estimates = np.empty(reps)
    for r in range(reps):
        # a model that separates the groups imperfectly: non-events N(0, 1),
        # events N(1.0, 1), which is AUC = 0.760 and independent draws.
        scores0 = rng.normal(0.0, 1.0, n0)
        scores1 = rng.normal(1.0, 1.0, n1)
        # Mann-Whitney AUC
        grid = scores1[:, None] - scores0[None, :]
        auc = float(((grid > 0).sum() + 0.5 * (grid == 0).sum()) / (n1 * n0))
        estimates[r] = np.log(auc / (1.0 - auc))
    simulated_sd = float(np.std(estimates, ddof=1))

    mean_auc = float(np.mean(1.0 / (1.0 + np.exp(-estimates))))
    _, vi = ma.auc_logit(mean_auc, n_event=n1, n_nonevent=n0)
    analytic_sd = float(np.sqrt(vi[0]))

    # 15%: 400 replicates put the sampling error of the SD itself at about 3.5%,
    # and the two are compared at the average AUC rather than per replicate.
    assert analytic_sd == pytest.approx(simulated_sd, rel=0.15)


# --------------------------------------------------------------------------- #
# the pooling
# --------------------------------------------------------------------------- #
def test_the_pooled_auc_is_the_back_transformed_pooled_logit():
    yi, vi = ma.auc_logit(AUC, n_event=N_EVENT, n_nonevent=N_NONEVENT)
    fit = ma.prediction_meta(yi, vi, metric="auc")
    assert fit.scale == "logit"
    assert fit.pooled == pytest.approx(1.0 / (1.0 + np.exp(-fit.b)), rel=1e-14)
    assert fit.pooled_ci_lb == pytest.approx(
        1.0 / (1.0 + np.exp(-fit.ci_lb)), rel=1e-14)
    # a CI on a proportion must stay inside [0, 1]
    assert 0.0 < fit.pooled_ci_lb < fit.pooled < fit.pooled_ci_ub < 1.0


def test_the_pooling_is_exactly_rma():
    yi, vi = ma.auc_logit(AUC, n_event=N_EVENT, n_nonevent=N_NONEVENT)
    fit = ma.prediction_meta(yi, vi, metric="auc")
    reference = ma.rma(yi, vi=vi, method="REML")
    assert fit.b == pytest.approx(reference.b[0], rel=1e-14)
    assert fit.se == pytest.approx(reference.se[0], rel=1e-14)
    assert fit.tau2 == pytest.approx(
        float(np.asarray(reference.tau2).ravel()[0]), rel=1e-14)
    assert fit.QE == pytest.approx(reference.QE, rel=1e-14)


def test_the_oe_ratio_is_back_transformed_on_the_log_scale():
    yi, vi = ma.oe_ratio([120.0, 80.0], [100.0, 90.0])
    fit = ma.prediction_meta(yi, vi, metric="oe_ratio")
    assert fit.scale == "log"
    assert fit.pooled == pytest.approx(np.exp(fit.b), rel=1e-14)
    assert fit.pooled_ci_lb < fit.pooled < fit.pooled_ci_ub


def test_a_calibration_slope_is_left_alone():
    fit = ma.prediction_meta([0.9, 1.1, 1.0], [0.04, 0.05, 0.03],
                             metric="calibration_slope")
    assert fit.scale == "identity"
    assert fit.pooled == pytest.approx(fit.b, rel=1e-14)


def test_the_metrics_are_not_interchangeable():
    """An AUC and a calibration slope are on different scales by construction."""
    yi, vi = ma.auc_logit(AUC, n_event=N_EVENT, n_nonevent=N_NONEVENT)
    auc = ma.prediction_meta(yi, vi, metric="auc")
    slope = ma.prediction_meta(yi, vi, metric="calibration_slope")
    assert auc.pooled != slope.pooled        # logit back-transformed vs raw
    assert slope.pooled == pytest.approx(auc.b, rel=1e-14)


def test_an_unknown_metric_is_rejected():
    with pytest.raises(ValueError, match="unknown metric"):
        ma.prediction_meta([1.0], [1.0], metric="c_index")


def test_a_prediction_meta_analysis_round_trips_through_a_record(tmp_path):
    from pimeta.record import load_record, save_record

    yi, vi = ma.auc_logit(AUC, n_event=N_EVENT, n_nonevent=N_NONEVENT)
    fit = ma.prediction_meta(yi, vi, metric="auc")
    path = tmp_path / "discrimination.json"
    save_record(path, fit, mode="digest", function="prediction_meta")
    loaded = load_record(path)
    assert loaded.kind == "prediction_meta"
    assert loaded.results["values"]["pooled"] == pytest.approx(fit.pooled,
                                                               rel=1e-12)
    assert loaded.results["values"]["metric"] == "auc"
