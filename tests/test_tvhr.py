"""Time-varying hazard ratios, checked against R's Poisson GLM.

The fixture is generated from a formula rather than stored, so both languages
build exactly the same participants: study ``s`` has baseline hazard
``0.05 + 0.01 (s - 1)``, a treated participant's hazard is multiplied by
``hr[k] * exp(0.2 (s - 3.5))`` in interval ``k`` with ``hr = (0.6, 0.8, 1.0)``,
and follow-up is administratively censored at 18.  The true log hazard ratio is
therefore ``log(hr[k]) + 0.2 (s - 3.5)``: the same shape in every study with a
study-specific offset.

References: R 4.6.1, ``glm(family = poisson)`` with
``glm.control(epsilon = 1e-14, maxit = 200)``, then ``metafor::rma``
(method = "REML") per interval.  **The tolerance matters**: with R's default
``epsilon = 1e-8`` the GLM stops after six iterations and its standard errors
are off by 1.4e-5, which looks like an implementation difference and is not
one.
"""

import numpy as np
import pytest

import pimeta as ma
from pimeta.tvhr import piecewise_exp, poisson_irls, split_followup

EXACT = 1e-9

BREAKS = [0.0, 6.0, 12.0, 30.0]

# checksums of the generated data (R: sum(time), number of events)
R_SUM_TIME = 12948.384213333245
R_EVENTS = 816

# per-study log hazard ratios, study-major (6 studies x 3 intervals)
R_LOGHR = [
    -1.008885840405944, -0.46624921172447181, -0.65979665945957922,
    -0.94797108468391411, -0.59675811841307314, -0.28677645323367096,
    -0.49371878012475467, -0.21615240575213618, -0.19943481545703989,
    -0.55953385567627012, -0.19421533764839138, 0.011730056691197727,
    -0.13563744223886645, 0.20470882124935627, 0.14669992172295038,
    -0.11928351707791261, 0.19009669343614979, 0.24511017150409387,
]
R_SE = [
    0.39317854970634669, 0.35634832171979619, 0.39695812102975891,
    0.3320488069684886, 0.31180478177204102, 0.37893237325257856,
    0.28083633556764331, 0.29821603924634532, 0.35939764304781469,
    0.25070257691675274, 0.2775554665766895, 0.36397614271916678,
    0.22388683110066873, 0.28141170298837076, 0.37161166760972036,
    0.21846572435891554, 0.25949964797254954, 0.37161167647505305,
]

# pooled per interval
R_POOLED = [
    (-0.46421523726943603, 0.14566685709802468, -0.74971703092270703,
     -0.17871344361616498, 0.052109785119138476, 8.6575365390616152),
    (-0.1425815118054044, 0.13410708069996741, -0.40542656004914707,
     0.12026353643833826, 0.021397774559009902, 6.2033991547743579),
    (-0.1114715610356532, 0.15235068218610848, -0.41007341114053375,
     0.18713028906922735, 0.0, 3.6999415301582399),
]


def _ipd():
    """The deterministic piecewise-exponential cohort (see the docstring)."""
    n, per = 1200, 200
    study = np.arange(n) // per + 1
    treat = (np.arange(n) // 100) % 2
    u = ((np.arange(1, n + 1) * 7919) % 997) / 997.0
    target = -np.log(1.0 - u)
    base = 0.05 + 0.01 * (study - 1)
    shift = 0.2 * (study - 3.5)
    hr = np.array([0.6, 0.8, 1.0])
    time = np.zeros(n)
    event = np.zeros(n, dtype=int)
    cens = 18.0
    for i in range(n):
        mult = hr * np.exp(shift[i]) if treat[i] == 1 else np.ones(3)
        hz = base[i] * mult
        cum1 = hz[0] * 6.0
        cum2 = cum1 + hz[1] * 6.0
        if target[i] <= cum1:
            t = target[i] / hz[0]
        elif target[i] <= cum2:
            t = 6.0 + (target[i] - cum1) / hz[1]
        else:
            t = 12.0 + (target[i] - cum2) / hz[2]
        time[i] = min(t, cens)
        event[i] = 1 if t <= cens else 0
    return time, event, study, treat


def _fit():
    time, event, study, treat = _ipd()
    return piecewise_exp(time, event, study=study, treat=treat, breaks=BREAKS)


# --------------------------------------------------------------------------- #
# the data generator itself (a wrong fixture would hide a wrong estimator)
# --------------------------------------------------------------------------- #

def test_the_fixture_matches_the_data_r_generated():
    time, event, _study, _treat = _ipd()
    assert float(np.sum(time)) == pytest.approx(R_SUM_TIME, abs=1e-9)
    assert int(np.sum(event)) == R_EVENTS


# --------------------------------------------------------------------------- #
# splitting follow-up
# --------------------------------------------------------------------------- #

def test_split_followup_splits_at_the_boundaries():
    rows, ints, exposure, events = split_followup(
        [5.0, 8.0, 20.0], [1, 0, 0], [0.0, 6.0, 12.0, 30.0])
    assert rows.tolist() == [0, 1, 1, 2, 2, 2]
    assert ints.tolist() == [0, 0, 1, 0, 1, 2]
    assert exposure.tolist() == [5.0, 6.0, 2.0, 6.0, 6.0, 8.0]
    # the event belongs to the interval it fell in, and only to that one
    assert events.tolist() == [1, 0, 0, 0, 0, 0]


def test_split_followup_carries_the_third_interval_to_the_horizon():
    rows, ints, exposure, _events = split_followup(
        [25.0], [0], [0.0, 6.0, 12.0, 30.0])
    assert rows.tolist() == [0, 0, 0]
    assert ints.tolist() == [0, 1, 2]
    assert exposure.tolist() == [6.0, 6.0, 13.0]


def test_split_followup_rejects_breaks_that_do_not_start_at_zero():
    with pytest.raises(ValueError, match="start at 0"):
        split_followup([5.0], [1], [1.0, 6.0])


# --------------------------------------------------------------------------- #
# the Poisson solver
# --------------------------------------------------------------------------- #

def test_poisson_irls_recovers_a_known_ratio_exactly():
    """One binary covariate and an offset: the MLE has a closed form."""
    treat = np.array([0.0, 0.0, 0.0, 1.0, 1.0, 1.0])
    exposure = np.array([10.0, 12.0, 9.0, 10.0, 11.0, 8.0])
    events = np.array([4.0, 6.0, 3.0, 7.0, 8.0, 6.0])
    X = np.column_stack([np.ones(6), treat])
    beta, vcov, _iters, converged = poisson_irls(
        events, X, offset=np.log(exposure))
    assert converged
    rate0 = 13.0 / 31.0
    rate1 = 21.0 / 29.0
    assert beta[0] == pytest.approx(np.log(rate0), abs=1e-10)
    assert beta[1] == pytest.approx(np.log(rate1 / rate0), abs=1e-10)
    # the variance of a log rate is 1/events
    assert np.sqrt(vcov[0, 0]) == pytest.approx(1.0 / np.sqrt(13.0), abs=1e-10)


def test_poisson_irls_rejects_a_rank_deficient_design():
    X = np.column_stack([np.ones(4), np.ones(4)])
    with pytest.raises(ValueError, match="rank deficient"):
        poisson_irls([1.0, 2, 3, 4], X)


# --------------------------------------------------------------------------- #
# the two-stage estimator
# --------------------------------------------------------------------------- #

def test_per_study_log_hazard_ratios_match_glm():
    fit = _fit()
    assert np.asarray(fit.log_hr).ravel() == pytest.approx(R_LOGHR,
                                                           abs=1e-8)
    assert np.asarray(fit.se_hr).ravel() == pytest.approx(R_SE, abs=1e-8)


def test_pooled_hazard_ratios_match_metafor():
    fit = _fit()
    for j, (est, se, lo, hi, tau2, qe) in enumerate(R_POOLED):
        assert fit.beta[j] == pytest.approx(est, abs=1e-8)
        assert fit.se[j] == pytest.approx(se, abs=1e-8)
        assert fit.ci_lb[j] == pytest.approx(lo, abs=1e-8)
        assert fit.ci_ub[j] == pytest.approx(hi, abs=1e-8)
        assert fit.tau2[j] == pytest.approx(tau2, abs=1e-8)
        assert fit.QE[j] == pytest.approx(qe, abs=2e-6)


def test_every_study_contributes_to_every_interval():
    fit = _fit()
    assert fit.k == [6, 6, 6]
    assert fit.dropped == []


def test_hazard_ratios_are_reported_on_the_ratio_scale():
    rows = _fit().coef_table()
    assert [r["interval"] for r in rows] == ["0-6", "6-12", "12-30"]
    assert rows[0]["hr"] == pytest.approx(np.exp(R_POOLED[0][0]), rel=1e-8)
    assert rows[0]["hr_lb"] < rows[0]["hr"] < rows[0]["hr_ub"]


def test_the_fixture_shows_the_hazard_ratio_rising_over_time():
    """The generated truth is 0.6, 0.8, 1.0, and the fit must recover it."""
    rows = _fit().coef_table()
    assert rows[0]["hr"] < 0.8
    assert rows[1]["hr"] < rows[2]["hr"] + 0.1
    assert rows[0]["hr"] < rows[2]["hr"]


def test_level_accepts_percent_and_proportion():
    time, event, study, treat = _ipd()
    a = piecewise_exp(time, event, study=study, treat=treat, breaks=BREAKS,
                      level=95)
    b = piecewise_exp(time, event, study=study, treat=treat, breaks=BREAKS,
                      level=0.95)
    assert a.ci_lb == pytest.approx(b.ci_lb, rel=1e-12)


def test_an_interval_with_no_events_is_removed_rather_than_fitted():
    """A zero-event cell has no finite estimate; it must not be invented.

    Studies 2 and 3 only enter the second interval, so their first-interval
    hazard ratio does not exist -- and the fit says so with NaN instead of
    returning a coefficient pushed to minus infinity.
    """
    time = np.array([5.0, 5.0, 20.0, 20.0, 20.0, 20.0])
    event = np.array([1, 1, 1, 1, 1, 1])
    study = np.array([1, 1, 2, 2, 3, 3])
    treat = np.array([0, 1, 0, 1, 0, 1])
    fit = piecewise_exp(time, event, study=study, treat=treat,
                        breaks=[0.0, 6.0, 30.0])
    # study 1 has events in the first interval only, so it contributes there
    assert np.isfinite(fit.log_hr[0, 0])
    assert not np.isfinite(fit.log_hr[0, 1])
    # studies 2 and 3 have events in the second interval only
    assert not np.isfinite(fit.log_hr[1, 0])
    assert np.isfinite(fit.log_hr[1, 1])


def test_rejects_a_non_binary_treatment_indicator():
    time, event, study, _treat = _ipd()
    with pytest.raises(ValueError, match="0/1"):
        piecewise_exp(time, event, study=study, treat=np.full(time.size, 2),
                      breaks=BREAKS)


def test_rejects_a_single_study():
    time, _event, _study, treat = _ipd()
    with pytest.raises(ValueError, match="two studies"):
        piecewise_exp(time, np.ones(time.size, dtype=int),
                      study=np.zeros(time.size), treat=treat, breaks=BREAKS)


# --------------------------------------------------------------------------- #
# the record round-trip the package requires of every new model
# --------------------------------------------------------------------------- #

def test_a_piecewise_fit_round_trips_through_a_record(tmp_path):
    from pimeta.record import load_record, save_record

    fit = _fit()
    path = tmp_path / "tvhr.json"
    save_record(path, fit, mode="digest", function="piecewise_exp")
    loaded = load_record(path)
    assert loaded.kind == "piecewise_exp"
    assert np.asarray(loaded.results["values"]["beta"]) == pytest.approx(
        np.asarray(fit.beta), rel=1e-12)
    assert loaded.results["values"]["labels"] == fit.labels


def test_piecewise_exp_is_reachable_from_the_package_root():
    assert ma.piecewise_exp is piecewise_exp
