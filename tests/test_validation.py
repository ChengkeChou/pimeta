"""Tests for the simulation engine.

Two kinds of test live here and they answer different questions.

The first kind checks the machinery: that a draw is reproducible from its seed,
that the four precision layouts do what their names say, that a contamination
lands on exactly one study, and that ``summarise_fits`` counts what it is given.

The second kind checks that the summaries are *computed* correctly on records
whose answer can be worked out by hand. It deliberately does not assert that
coverage equals 0.95: that is a property of the estimator, not of this code, and
it is measured against stored baselines by ``tools/validate.py``. A unit test
that asserted 0.95 would either be flaky or be testing nothing.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pytest

from pimeta.validation import (
    SCENARIOS,
    Dataset,
    Scenario,
    run_scenario,
    simulate_study,
    summarise_fits,
)

BASELINE_DIR = (Path(__file__).resolve().parents[1] / "tests" / "validation"
                / "baseline")


# ---------------------------------------------------------------- generation

def test_simulate_study_is_reproducible_from_its_seed():
    s = SCENARIOS["null"]
    a = simulate_study(s, 11)
    b = simulate_study(s, 11)
    assert np.array_equal(a.yi, b.yi)
    assert np.array_equal(a.sei, b.sei)
    assert a.mu_new == b.mu_new


def test_simulate_study_differs_between_seeds():
    s = SCENARIOS["null"]
    assert not np.array_equal(simulate_study(s, 1).yi, simulate_study(s, 2).yi)


def test_simulate_study_records_the_truth_it_used():
    s = Scenario(name="t", k=8, mu=0.4, tau2=0.09, se=0.3)
    d = simulate_study(s, 0)
    assert len(d) == 8
    assert d.mu == pytest.approx(0.4)
    assert d.tau2 == pytest.approx(0.09)
    assert d.truth["tau"] == pytest.approx(0.3)
    assert np.allclose(d.vi, d.sei ** 2)


def test_new_study_effect_is_drawn_from_the_population_not_taken_as_the_mean():
    # The target of a prediction interval is a fresh draw from N(mu, tau^2); a
    # simulator that returned mu would make every prediction interval look
    # calibrated no matter how wrong it was.
    s = Scenario(name="t", k=4, mu=0.0, tau2=0.25, se=0.2)
    draws = np.array([simulate_study(s, seed).mu_new for seed in range(300)])
    assert draws.std() == pytest.approx(0.5, rel=0.2)
    assert abs(draws.mean()) < 0.1


def test_equal_precisions_are_all_the_typical_se():
    d = simulate_study(Scenario(name="t", k=6, se=0.25, se_kind="equal"), 3)
    assert np.allclose(d.sei, 0.25)


def test_wide_precisions_span_a_factor_of_four():
    d = simulate_study(Scenario(name="t", k=5, se=0.2, se_kind="wide"), 3)
    assert d.sei.max() / d.sei.min() == pytest.approx(4.0)


def test_one_precise_study_has_one_small_standard_error():
    d = simulate_study(Scenario(name="t", k=6, se=0.2, se_kind="one_precise"), 3)
    assert np.sum(np.isclose(d.sei, 0.02)) == 1
    assert np.sum(np.isclose(d.sei, 0.2)) == 5


def test_lognormal_precisions_are_positive_and_spread():
    d = simulate_study(Scenario(name="t", k=50, se=0.2, se_kind="lognormal"), 3)
    assert np.all(d.sei > 0)
    assert d.sei.max() / d.sei.min() > 2.0


def test_contamination_moves_exactly_one_study_by_the_requested_amount():
    plain = Scenario(name="t", k=10, mu=0.0, tau2=0.04, se=0.2)
    dirty = Scenario(name="t", k=10, mu=0.0, tau2=0.04, se=0.2, outlier=5.0)
    clean = simulate_study(plain, 42)
    polluted = simulate_study(dirty, 42)
    moved = np.flatnonzero(~np.isclose(clean.yi, polluted.yi))
    assert moved.size == 1
    j = moved[0]
    assert polluted.yi[j] - clean.yi[j] == pytest.approx(5.0 * clean.sei[j])


@pytest.mark.parametrize("bad,kwargs", [
    ("k", {"k": 1}),
    ("tau2", {"tau2": -0.1}),
    ("se", {"se": 0.0}),
    ("se_kind", {"se_kind": "nope"}),
])
def test_scenario_rejects_impossible_settings(bad, kwargs):
    with pytest.raises(ValueError):
        Scenario(name="t", **kwargs)


# ---------------------------------------------------------------- summarising

def _record(b, se=0.1, ci=None, pval=0.5, tau2=0.0, i2=0.0, mu_new=0.0,
            pi=None):
    lb, ub = ci if ci is not None else (b - 1.96 * se, b + 1.96 * se)
    pi_lb, pi_ub = pi if pi is not None else (b - 2.0, b + 2.0)
    return {"b": b, "se": se, "ci_lb": lb, "ci_ub": ub, "pval": pval,
            "tau2": tau2, "I2": i2, "pi_lb": pi_lb, "pi_ub": pi_ub,
            "pi_lb_t": pi_lb, "pi_ub_t": pi_ub, "mu_new": mu_new}


def test_bias_and_coverage_are_the_frequencies_they_claim_to_be():
    # Estimates 0.0, 0.1, 0.2, 0.3 against a truth of 0.2: mean 0.15, so the
    # bias is -0.05. The intervals are given explicitly rather than as a
    # half-width so that which two contain 0.2 is a matter of arithmetic, not of
    # whether the comparison is closed at the endpoint.
    recs = [_record(0.0, se=0.1, ci=(-0.2, 0.1)),
            _record(0.1, se=0.1, ci=(0.0, 0.2)),
            _record(0.2, se=0.1, ci=(0.25, 0.5)),
            _record(0.3, se=0.1, ci=(0.1, 0.4))]
    out = summarise_fits(recs, mu=0.2, tau2=0.0)
    assert out["bias"]["mean_estimate"] == pytest.approx(0.15)
    assert out["bias"]["estimate"] == pytest.approx(-0.05)
    assert out["bias"]["relative"] == pytest.approx(-0.25)
    assert out["coverage"]["ci"]["estimate"] == pytest.approx(0.5)
    assert out["coverage"]["ci"]["hits"] == 2
    assert out["counts"]["reps_used"] == 4


def test_rmse_is_computed_about_the_truth_not_about_the_mean():
    recs = [_record(b, se=0.5) for b in (0.0, 0.1, 0.2, 0.3)]
    out = summarise_fits(recs, mu=0.2, tau2=0.0)
    expect = math.sqrt(np.mean([(b - 0.2) ** 2 for b in (0.0, 0.1, 0.2, 0.3)]))
    assert out["dispersion"]["rmse"] == pytest.approx(expect)
    assert out["dispersion"]["rmse"] > abs(out["bias"]["estimate"])


def test_se_ratio_is_reported_against_the_empirical_spread():
    recs = [_record(b, se=0.25) for b in (0.0, 0.5, 1.0)]
    out = summarise_fits(recs, mu=0.5, tau2=0.0)
    assert out["dispersion"]["mean_reported_se"] == pytest.approx(0.25)
    # SD of (0, 0.5, 1.0) with ddof=1 is 0.5.
    assert out["dispersion"]["se_ratio"] == pytest.approx(0.5)


def test_prediction_interval_coverage_uses_the_new_studys_effect():
    # The interval contains mu (0.0) but not the new study's effect (0.9); a
    # check written against mu would call this a hit and hide the miss.
    recs = [_record(0.0, se=0.1, pi=(-0.2, 0.2), mu_new=0.9) for _ in range(5)]
    out = summarise_fits(recs, mu=0.0, tau2=0.0)
    assert out["coverage"]["ci"]["estimate"] == pytest.approx(1.0)
    assert out["coverage"]["pi"]["estimate"] == pytest.approx(0.0)


def test_rejection_rate_is_labelled_as_size_or_power_by_the_truth():
    recs = [_record(0.0, pval=0.01) for _ in range(4)]
    assert summarise_fits(recs, mu=0.0, tau2=0.0)["tests"]["role"] == "size"
    assert summarise_fits(recs, mu=0.3, tau2=0.0)["tests"]["role"] == "power"


def test_non_finite_replications_are_counted_not_averaged_away():
    recs = [_record(0.1), _record(float("nan")), _record(0.3)]
    out = summarise_fits(recs, mu=0.2, tau2=0.0)
    assert out["counts"] == {"reps": 3, "reps_used": 2, "reps_non_finite": 1}
    assert out["reliability"]["fraction_finite"] == pytest.approx(2 / 3)
    assert out["bias"]["mean_estimate"] == pytest.approx(0.2)


def test_all_non_finite_replications_raise_instead_of_returning_nan():
    recs = [_record(float("nan")) for _ in range(3)]
    with pytest.raises(ValueError, match="non-finite"):
        summarise_fits(recs, mu=0.0, tau2=0.0)


def test_tau2_boundary_fraction_is_reported():
    recs = [_record(0.0, tau2=0.0), _record(0.0, tau2=0.1),
            _record(0.0, tau2=0.0), _record(0.0, tau2=0.05)]
    out = summarise_fits(recs, mu=0.0, tau2=0.04)
    assert out["heterogeneity"]["tau2_at_boundary"] == pytest.approx(0.5)
    assert out["reliability"]["fraction_tau2_at_boundary"] == pytest.approx(0.5)


def test_summarise_without_a_baseline_key_still_works_for_hand_written_records():
    # predict() may be unavailable on some fitted objects; the PI block then has
    # no defined replications rather than silently scoring them as misses.
    rec = _record(0.0)
    rec["pi_lb"] = None
    rec["pi_ub"] = None
    out = summarise_fits([rec], mu=0.0, tau2=0.0)
    assert out["coverage"]["pi"]["reps"] == 0
    assert out["coverage"]["pi_defined_reps"] == 0


# ------------------------------------------------------------ run_scenario

def test_run_scenario_is_reproducible_and_carries_its_settings():
    s = Scenario(name="quick", k=12, mu=0.1, tau2=0.04, se=0.2)
    a = run_scenario(s, reps=20, seed=99)
    b = run_scenario(s, reps=20, seed=99)
    assert json.dumps(a, sort_keys=True) == json.dumps(b, sort_keys=True)
    assert a["settings"] == {"reps": 20, "seed": 99, "method": "REML",
                             "level": 0.95}
    assert a["scenario"]["name"] == "quick"


def test_run_scenario_reports_the_six_blocks():
    s = Scenario(name="quick", k=12, mu=0.1, tau2=0.04, se=0.2)
    out = run_scenario(s, reps=20, seed=5)
    for block in ("bias", "dispersion", "coverage", "tests",
                  "heterogeneity", "reliability"):
        assert block in out, block
    assert out["counts"]["reps"] == 20


def test_run_scenario_rejects_a_single_replication():
    with pytest.raises(ValueError, match="at least 2"):
        run_scenario(SCENARIOS["null"], reps=1)


def test_run_scenario_accepts_an_injected_estimator():
    seen = []

    class Fake:
        b = [0.25]
        se = [0.1]
        ci_lb = [0.05]
        ci_ub = [0.45]
        pval = [0.01]
        tau2 = 0.04
        I2 = 50.0

        def predict(self, level=None, **kw):
            raise RuntimeError("no prediction available")

    def fit(data: Dataset):
        seen.append(len(data))
        return Fake()

    s = Scenario(name="inj", k=7, mu=0.25, tau2=0.04, se=0.2)
    out = run_scenario(s, reps=6, seed=1, fit=fit)
    assert seen == [7] * 6
    assert out["bias"]["mean_estimate"] == pytest.approx(0.25)
    # A model that cannot predict must not be scored as having missed.
    assert out["coverage"]["pi"]["reps"] == 0


# ------------------------------------------------------------- baselines

BASELINES = sorted(BASELINE_DIR.glob("*.json"))


@pytest.mark.parametrize("path", BASELINES, ids=lambda p: p.stem)
def test_stored_baseline_has_every_block(path: Path):
    report = json.loads(path.read_text(encoding="utf-8"))
    assert report["schema_version"] == 1
    assert report["scenario"]["name"] == path.stem
    for block in ("counts", "bias", "dispersion", "coverage", "tests",
                  "heterogeneity", "reliability", "settings"):
        assert block in report, block
    assert report["settings"]["reps"] >= 200
    assert report["counts"]["reps"] == report["settings"]["reps"]


@pytest.mark.parametrize("path", BASELINES, ids=lambda p: p.stem)
def test_stored_baseline_records_its_targets_next_to_its_numbers(path: Path):
    # A coverage number without the level it was checked against cannot be read
    # as good or bad, so the baselines carry the target, the MCSE and the role.
    report = json.loads(path.read_text(encoding="utf-8"))
    cov = report["coverage"]
    assert cov["ci_target"] == pytest.approx(0.95)
    assert cov["pi_target"] == pytest.approx(0.95)
    for key in ("ci", "pi", "pi_t"):
        assert cov[key]["mcse"] > 0.0
    assert report["tests"]["alpha"] == pytest.approx(0.05)
    assert report["tests"]["role"] in ("size", "power")


def test_every_registered_scenario_is_a_real_scenario():
    assert set(SCENARIOS) == {p.stem for p in BASELINES}, (
        "every scenario in SCENARIOS needs a baseline, and every baseline "
        "needs a scenario; regenerate with tools/validate.py --all --write"
    )
