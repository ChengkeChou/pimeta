"""Coverage numbers are data, not prose: the baselines and the docstrings must agree.

The measurements in this project are quoted inside docstrings, and a docstring
is not checked by anything. That is how two claims about `struct="HCS"` survived
unmeasured for weeks, including one in a runtime note a user actually sees.

So the numbers live in ``tests/reference/coverage_*.json`` and this module does
two things with them:

* **The findings are asserted.** Ordering claims that the prose makes — the
  bootstrap beats the closed-form references at small ``k``, the LKJ penalty
  halves the boundary rate, CR2 is worse at the edge, ``"HCS"`` raises the
  boundary rate — are checked as inequalities on the baseline. If a future
  measurement changes a sign, this fails and the prose has to change with it.
* **The quoted numbers are checked against the baseline.** Every rate the
  docstrings quote has to appear, formatted the same way, in the baseline. That
  is what stops a docstring drifting from its measurement.

What this module cannot do is re-run the simulations: at 60 to 100 replications
per configuration they take minutes, and a default test run must not. That is
what ``tools/coverage_*.py`` are for, and that is why the replications and the
seed are recorded here.
"""

from __future__ import annotations

import inspect
import json
from pathlib import Path

import pytest

import pimeta as ma

REFERENCE = Path(__file__).resolve().parent / "reference"
BAND = json.loads((REFERENCE / "coverage_ipd_band.json").read_text(encoding="utf-8"))
REGION = json.loads(
    (REFERENCE / "coverage_predict_region.json").read_text(encoding="utf-8"))


# --------------------------------------------------------------------------- #
# the IPD band
# --------------------------------------------------------------------------- #
def _band(run: str) -> dict:
    return BAND["comparisons"]["runs"][run]


def test_the_ipd_baseline_has_the_five_covariates_and_a_nominal_level():
    assert BAND["nominal"] == 0.95
    for column in BAND["table"].values():
        assert len(column) == 5
        assert all(0.0 <= value <= 1.0 for value in column)
    for run in BAND["comparisons"]["runs"].values():
        assert len(run["coverage"]) == 5
        assert all(0.0 <= value <= 1.0 for value in run["coverage"])


def test_per_point_degrees_of_freedom_are_what_the_baseline_was_measured_with():
    """The 250-replication table: per-point df fix the left edge and the middle."""
    model = BAND["table"]["model_t"]
    robust = BAND["table"]["robust_point_df"]
    # robust is better at the left edge and in the middle
    assert robust[0] > model[0]
    assert robust[1] > model[1]
    assert robust[2] > model[2]
    # and no better at the right edge, which is the known weakness
    assert robust[4] == pytest.approx(model[4], abs=0.02)


def test_the_lkj_penalty_halves_the_boundary_rate_and_lifts_the_edge():
    plain = _band("cr1_un")
    weak = _band("cr1_lkj2_un")
    assert weak["boundary_share"] < 0.75 * plain["boundary_share"] + 0.05
    assert weak["coverage"][4] > plain["coverage"][4]
    # the other four covariates are unchanged by the penalty
    assert weak["coverage"][:4] == pytest.approx(plain["coverage"][:4], abs=0.05)


def test_eta_four_does_not_buy_more_coverage_than_eta_two():
    """The boundary rate keeps falling but the band does not improve."""
    weak = _band("cr1_lkj2_un")
    strong = _band("cr1_lkj4_un")
    assert strong["boundary_share"] < weak["boundary_share"]
    assert strong["coverage"][4] == pytest.approx(weak["coverage"][4], abs=0.02)


def test_cr2_and_hcs_are_recorded_as_negative_results():
    """Both were plausible and both were measured to fail; keep them failing."""
    plain = _band("cr1_un")
    cr2 = _band("cr2_un")
    hcs = _band("cr1_hcs")
    assert cr2["coverage"][4] < plain["coverage"][4]
    assert hcs["coverage"][4] == pytest.approx(plain["coverage"][4], abs=0.02)
    assert hcs["boundary_share"] > plain["boundary_share"]


def test_the_adaptive_reference_fixes_the_right_edge():
    """The one change that worked, and the constraint it had to satisfy.

    Neither the variance estimator nor the covariance structure moved this
    number: CR2 made it worse, ``HCS`` left it alone, the LKJ penalty added two
    points.  What was wrong was the *shape* of the reference -- a point-wise t
    where the contrast needs several basis directions at once -- and the fix
    shows up as a single column of the campaign, so it is pinned here rather
    than described.
    """
    run = _band("adaptive_band")
    point = run["point_coverage"]
    adaptive = run["coverage"]
    assert adaptive[4] > point[4] + 0.10, (adaptive[4], point[4])
    assert adaptive[4] >= 0.85
    # and it does not buy that by widening everything: no covariate gets worse
    for j in range(4):
        assert adaptive[j] >= point[j] - 0.03, j


# --------------------------------------------------------------------------- #
# the joint prediction region
# --------------------------------------------------------------------------- #
def test_the_region_baseline_records_every_reference_at_both_sample_sizes():
    main = REGION["runs"]["main"]["k"]
    for k in ("10", "20"):
        assert {"chi2", "f", "hk", "kr", "bootstrap"} <= set(main[k])


def test_the_hk_reference_beats_the_closed_forms_at_k_ten():
    """The study-based divisor is what makes it work, and it is pinned here.

    Counting rows instead of studies gives the same coverage as Hotelling, so a
    later refactor that "simplifies" the divisor would silently undo the whole
    correction.  The gap to nominal is recorded too: HK is the best closed form,
    not a calibrated one.
    """
    k10 = REGION["runs"]["main"]["k"]["10"]
    assert k10["hk"] > k10["f"]
    assert k10["hk"] > k10["chi2"]
    assert k10["hk"] < REGION["nominal"]


def test_the_bootstrap_is_the_only_reference_that_improves_at_small_k():
    k10 = REGION["runs"]["main"]["k"]["10"]
    assert k10["bootstrap"] > k10["chi2"]
    assert k10["bootstrap"] > k10["f"]
    # and it does not reach nominal, which is what the docstring says
    assert k10["bootstrap"] < REGION["nominal"]
    k20 = REGION["runs"]["main"]["k"]["20"]
    assert k20["bootstrap"] >= k20["chi2"]


def test_the_two_runs_of_the_bootstrap_agree_within_monte_carlo_error():
    """A 5-point spread at 100 replications is the MC error, not the method."""
    main = REGION["runs"]["main"]["k"]["10"]["bootstrap"]
    second = REGION["runs"]["second"]["k"]["10"]["bootstrap"]
    assert abs(main - second) <= 0.10


def test_the_per_outcome_intervals_never_reach_nominal():
    """The naive answer the joint region replaces; it ignores the correlation."""
    for run in REGION["runs"].values():
        for row in run["k"].values():
            if "marginal" in row:
                assert row["marginal"] < REGION["nominal"]


# --------------------------------------------------------------------------- #
# the docstrings quote the baseline, not the other way round
# --------------------------------------------------------------------------- #
def _quoted(text: str, value: float) -> bool:
    return f"{value:.3f}" in text


def test_the_band_docstring_quotes_the_baseline_numbers():
    text = inspect.getdoc(ma.IpdCurveResult.curve) or ""
    # the table the docstring prints
    for column in BAND["table"].values():
        for value in column:
            assert _quoted(text, value), (
                "the curve docstring does not quote the table value "
                f"{value:.3f}; either the measurement or the prose is stale")
    # and the two negative results it quotes in prose: CR2 is quoted in full,
    # HCS only by the edge it fails to fix, which is what the sentence claims
    for value in _band("cr2_un")["coverage"]:
        assert _quoted(text, value), (
            f"the curve docstring does not quote {value:.3f} for CR2")
    edge = _band("cr1_hcs")["coverage"][4]
    assert _quoted(text, edge), (
        f"the curve docstring does not quote the HCS edge {edge:.3f}")
    penalty_edge = _band("cr1_lkj2_un")["coverage"][4]
    assert f"{penalty_edge:.3f}" in text.replace("0.750", "0.750"), (
        "the curve docstring does not quote the LKJ edge " f"{penalty_edge:.3f}")
    assert "50%" in text and "98%" in text, (
        "the boundary rates the prose quotes (73% -> 50% with LKJ, 98% with "
        "HCS) are no longer in the docstring")


def test_the_region_docstring_quotes_the_baseline_numbers():
    text = inspect.getdoc(ma.predict_region) or ""
    for value in (REGION["runs"]["main"]["k"]["10"]["chi2"],
                  REGION["runs"]["main"]["k"]["10"]["bootstrap"],
                  REGION["runs"]["main"]["k"]["20"]["chi2"],
                  REGION["runs"]["main"]["k"]["20"]["bootstrap"]):
        assert _quoted(text, value), (
            f"predict_region's docstring does not quote {value:.3f} from the "
            "baseline")
