"""Survival outcomes: Kaplan-Meier, RMST, and reconstructing individual data.

The RMST reference is R 4.6.1 integrating the same step function two ways —
a hand-written step sum and ``integrate(stepfun(...), 0, tau)`` — which agree to
15 digits:

    times <- c(1, 2, 4, 6, 8, 12, 18, 24); S <- exp(-0.05 * times); tau <- 24
    step sum : 15.293570167242342
    integrate: 15.29357016724234

The reconstruction is checked by rebuilding the curve from the individual data
and comparing with the risk sets and survival values it was built from.
"""

from __future__ import annotations

import numpy as np
import pytest

import pimeta as ma

# --------------------------------------------------------------------------- #
# reference: an exponential curve, integrated in R
# --------------------------------------------------------------------------- #
EXP_TIME = [1.0, 2.0, 4.0, 6.0, 8.0, 12.0, 18.0, 24.0]
EXP_SURV = [0.95122942450071402, 0.90483741803595952, 0.81873075307798182,
            0.74081822068171788, 0.67032004603563933, 0.54881163609402639,
            0.40656965974059911, 0.30119421191220203]
R_RMST_24 = 15.293570167242342

# a curve with censoring: dosresmeta-style risk-set columns
KM_TIME = [2.0, 5.0, 9.0, 14.0, 20.0]
KM_RISK = [36, 30, 21, 12, 4]
KM_EVENTS = [4, 5, 6, 5, 2]
KM_TOTAL = 40
KM_SURV = [0.88888888888888884, 0.7407407407407407, 0.52910052910052907,
           0.30864197530864196, 0.15432098765432098]


# --------------------------------------------------------------------------- #
# Kaplan-Meier
# --------------------------------------------------------------------------- #
def test_km_curve_matches_the_closed_form():
    km = ma.km_curve([1, 2, 2, 5], [1, 1, 0, 1])
    # at t=1: 1 - 1/4; at t=2: (3/4)(1 - 1/3); the censoring at t=2 changes
    # nothing; at t=5: (1/2)(1 - 1/1) = 0
    assert km.surv == pytest.approx([0.75, 0.5, 0.0])
    assert list(km.n_risk) == [4, 3, 1]
    assert list(km.n_event) == [1, 1, 1]
    assert list(km.n_censored) == [0, 1, 0]
    assert km.n == 4


def test_km_curve_rejects_bad_input():
    with pytest.raises(ValueError, match="same length"):
        ma.km_curve([1, 2], [1])
    with pytest.raises(ValueError, match="negative"):
        ma.km_curve([-1, 2], [1, 1])
    with pytest.raises(ValueError, match="0 \\(censored\\) or 1"):
        ma.km_curve([1, 2], [1, 2])


# --------------------------------------------------------------------------- #
# RMST
# --------------------------------------------------------------------------- #
def test_rmst_matches_rs_step_sum():
    result = ma.rmst(EXP_TIME, EXP_SURV, tau=24.0)
    assert result.rmst == pytest.approx(R_RMST_24, rel=1e-12)
    assert result.tau == pytest.approx(24.0)
    assert result.truncated is False
    assert result.se is None            # no risk sets, no standard error


def test_rmst_truncates_at_the_horizon():
    full = ma.rmst(EXP_TIME, EXP_SURV, tau=24.0)
    part = ma.rmst(EXP_TIME, EXP_SURV, tau=10.0)
    # 0-1 at S=1, 1-2 at S(1), 2-4 at S(2), 4-6 at S(4), 6-8 at S(6), 8-10 at S(8)
    expected = (1.0 + EXP_SURV[0] + 2 * EXP_SURV[1] + 2 * EXP_SURV[2]
                + 2 * EXP_SURV[3] + 2 * EXP_SURV[4])
    assert part.rmst == pytest.approx(expected, rel=1e-12)
    assert part.truncated is True
    assert part.rmst < full.rmst


def test_rmst_standard_error_matches_the_hand_calculation():
    # one event at t = 1 out of 100, tau = 2: area 1 + 0.99 = 1.99, and the
    # only event time contributes (0.99 * 1)^2 * 1 / (100 * 99).
    result = ma.rmst([1.0], [0.99], tau=2.0, n_risk=[100], n_event=[1])
    assert result.rmst == pytest.approx(1.99, rel=1e-14)
    assert result.se == pytest.approx(np.sqrt(0.99 ** 2 / 9900.0), rel=1e-14)
    assert result.ci_lb < result.rmst < result.ci_ub


def test_rmst_rejects_an_impossible_curve():
    with pytest.raises(ValueError, match="same length"):
        ma.rmst([1, 2], [0.9])
    with pytest.raises(ValueError, match="strictly increasing"):
        ma.rmst([2, 1], [0.9, 0.8])
    with pytest.raises(ValueError, match=r"lie in \[0, 1\]"):
        ma.rmst([1, 2], [0.9, 1.2])
    with pytest.raises(ValueError, match="positive"):
        ma.rmst([1, 2], [0.9, 0.8], tau=0.0)


def test_rmst_reads_like_a_sentence():
    text = str(ma.rmst(EXP_TIME, EXP_SURV, tau=24.0))
    assert text.startswith("RMST(24) = 15.2936")


# --------------------------------------------------------------------------- #
# reconstructing individual data
# --------------------------------------------------------------------------- #
def test_the_reconstruction_reproduces_the_published_curve():
    ipd = ma.km_reconstruct(KM_TIME, KM_RISK, KM_EVENTS, total=KM_TOTAL)
    assert ipd.n == KM_TOTAL
    assert ipd.n_event == sum(KM_EVENTS)

    reported = np.isin(ipd.km.time, KM_TIME)
    assert ipd.km.surv[reported] == pytest.approx(KM_SURV, rel=1e-12)
    assert list(ipd.km.n_risk[reported]) == KM_RISK
    assert list(ipd.km.n_event[reported]) == KM_EVENTS


def test_the_reconstruction_keeps_the_risk_set_shrinking():
    ipd = ma.km_reconstruct(KM_TIME, KM_RISK, KM_EVENTS, total=KM_TOTAL)
    assert np.all(np.diff(ipd.km.n_risk) <= 0)
    assert ipd.time.min() >= 0.0


def test_reconstructing_gives_back_the_rmst():
    ipd = ma.km_reconstruct(KM_TIME, KM_RISK, KM_EVENTS, total=KM_TOTAL)
    km = ipd.km
    from_ipd = ma.rmst(km.time, km.surv, tau=20.0, n_risk=km.n_risk,
                       n_event=km.n_event)
    from_table = ma.rmst(KM_TIME, KM_SURV, tau=20.0)
    assert from_ipd.rmst == pytest.approx(from_table.rmst, rel=1e-12)


def test_the_reconstruction_rejects_inconsistent_risk_sets():
    with pytest.raises(ValueError, match="cannot increase"):
        ma.km_reconstruct([2.0, 5.0], [10, 12], [1, 1])
    with pytest.raises(ValueError, match="cannot exceed"):
        ma.km_reconstruct([2.0], [10], [11])
    with pytest.raises(ValueError, match="inconsistent risk sets"):
        ma.km_reconstruct([2.0, 5.0], [10, 9], [2, 1])
    with pytest.raises(ValueError, match="smaller than the first risk set"):
        ma.km_reconstruct([2.0], [10], [1], total=5)


def test_an_rmst_round_trips_through_a_record(tmp_path):
    from pimeta.record import load_record, save_record

    result = ma.rmst(EXP_TIME, EXP_SURV, tau=24.0)
    path = tmp_path / "rmst.json"
    save_record(path, result, mode="digest", function="rmst")
    loaded = load_record(path)
    assert loaded.kind == "rmst"
    assert loaded.results["values"]["rmst"] == pytest.approx(R_RMST_24, rel=1e-12)
    assert loaded.results["values"]["tau"] == pytest.approx(24.0)
