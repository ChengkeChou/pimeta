"""Tests for the multiverse summary.

The numbers that need testing are the ones a reader will quote: the failure
count, the sign agreement, and the share of intervals excluding zero. Each is
checked against a small set of contrived results whose answers can be written
down by hand, because a summary that agrees with itself is not evidence.

The decomposition is tested for the property it claims and no more: with a
balanced grid of uncorrelated decisions the shares add up, and a decision that
does not move the estimate gets a share of zero.
"""

from __future__ import annotations

import json
import math

import pytest

import pimeta as ma
from pimeta.multiverse import DEFAULT_EXTRACT, MultiverseResult, multiverse
from pimeta.spec import AnalysisSpec, Decision


class Fake:
    """A result with the three attributes an intercept-only fit exposes."""

    def __init__(self, est: float, lb: float, ub: float):
        self.b = [est]
        self.ci_lb = [lb]
        self.ci_ub = [ub]


def _spec() -> AnalysisSpec:
    return AnalysisSpec(decisions=(
        Decision(name="shift", values=(0.0, 0.1), default=0.0,
                 reasons={0.0: "no shift", 0.1: "a shift the data allows"}),
        Decision(name="sign", values=(1, -1), default=1,
                 reasons={1: "as estimated", -1: "reversed for the test"}),
    ))


def _run(choices):
    base = 0.2 + float(choices["shift"])
    sign = int(choices["sign"])
    est = sign * base
    return Fake(est, est - 0.1, est + 0.1)


# ---------------------------------------------------------------- extracting

def test_default_extract_reads_the_first_coefficient():
    assert DEFAULT_EXTRACT(Fake(0.3, 0.1, 0.5)) == (0.3, 0.1, 0.5)
    est, lb, ub = DEFAULT_EXTRACT(ma.rma([0.1, 0.3, 0.2, 0.5],
                                        vi=[0.04, 0.05, 0.03, 0.06]))
    assert lb < est < ub


# ------------------------------------------------------------------- counts

def test_counts_name_all_four_numbers():
    result = multiverse(_spec(), _run)
    assert result.counts() == {"considered": 4, "run": 4, "usable": 4,
                               "failed": 0, "flagged": 0}
    assert result.failure_rate == 0.0


def test_a_failed_specification_is_counted_and_named():
    def run(choices):
        if choices["shift"] == 0.1:
            raise RuntimeError("does not converge")
        return _run(choices)

    result = multiverse(_spec(), run)
    assert result.counts()["usable"] == 2
    assert result.counts()["failed"] == 2
    assert result.failure_rate == pytest.approx(0.5)
    assert {f["id"] for f in result.failed} <= {s.id for s in _spec().expand()}
    assert all("does not converge" in f["error"] for f in result.failed)
    assert all("choices" in f for f in result.failed)


def test_a_result_that_cannot_be_summarised_is_a_failure_not_a_skip():
    def run(choices):
        return object()          # no .b, so extraction raises

    result = multiverse(_spec(), run)
    assert result.usable == 0
    assert len(result.failed) == 4
    assert all(f["error"].startswith("extract:") for f in result.failed)


def test_a_non_finite_estimate_is_a_failure():
    def run(choices):
        return Fake(float("nan"), 0.0, 1.0)

    result = multiverse(_spec(), run)
    assert result.usable == 0
    assert all("non-finite" in f["error"] for f in result.failed)


# ------------------------------------------------------------------ summary

def test_sign_agreement_and_zero_exclusion_are_frequencies():
    result = multiverse(_spec(), _run)
    block = result.summary()
    # Estimates: +0.2, +0.3, -0.2, -0.3. Two positive, two negative, and every
    # interval is +/-0.1 wide so all four exclude zero.
    assert block["sign"] == {"positive": 2, "negative": 2, "zero": 0,
                             "agreement": 0.5, "agreement_note":
                             block["sign"]["agreement_note"]}
    assert block["excludes_zero"]["count"] == 4
    assert block["excludes_zero"]["share"] == 1.0
    assert "not a probability" in block["excludes_zero"]["note"]


def test_an_interval_that_covers_zero_does_not_count():
    def run(choices):
        est = 0.05
        return Fake(est, -0.1, 0.2)

    block = multiverse(_spec(), run).summary()
    assert block["excludes_zero"]["count"] == 0
    assert block["sign"]["agreement"] == 1.0
    assert block["sign"]["zero"] == 0


def test_a_zero_estimate_is_reported_as_zero_rather_than_as_a_sign():
    def run(choices):
        return Fake(0.0, -0.1, 0.1)

    block = multiverse(_spec(), run).summary()
    assert block["sign"]["zero"] == 4
    assert block["sign"]["agreement"] == 0.0


def test_the_estimate_block_describes_the_distribution():
    block = multiverse(_spec(), _run).summary()["estimates"]
    assert block["min"] == pytest.approx(-0.3)
    assert block["max"] == pytest.approx(0.3)
    assert block["range"] == pytest.approx(0.6)
    assert block["median"] == pytest.approx(0.0)
    assert block["sd"] > 0.0


def test_an_empty_multiverse_says_so_instead_of_dividing_by_zero():
    def run(choices):
        raise RuntimeError("no")

    result = multiverse(_spec(), run)
    block = result.summary()
    assert block["estimates"] is None
    assert "note" in block
    assert "no specification produced an estimate" in str(result)


# -------------------------------------------------------------------- curve

def test_the_curve_is_ordered_by_estimate():
    curve = multiverse(_spec(), _run).curve()
    assert [row["estimate"] for row in curve] == \
        sorted(row["estimate"] for row in curve)
    assert len(curve) == 4
    assert {"shift", "sign", "id", "ci_lb", "ci_ub"} <= set(curve[0])


def test_the_curve_keeps_the_choices_that_produced_each_estimate():
    curve = multiverse(_spec(), _run).curve()
    for row in curve:
        assert row["sign"] * (0.2 + row["shift"]) == pytest.approx(row["estimate"])


# ------------------------------------------------------------ decomposition

def test_a_decision_that_does_not_move_the_estimate_gets_no_share():
    spec = AnalysisSpec(decisions=(
        Decision(name="shift", values=(0.0, 0.1), default=0.0,
                 reasons={0.0: "a", 0.1: "b"}),
        Decision(name="inert", values=(1, 2), default=1,
                 reasons={1: "a", 2: "b"}),
    ))

    def run(choices):
        est = 0.2 + float(choices["shift"])
        return Fake(est, est - 0.1, est + 0.1)

    rows = {r["decision"]: r for r in multiverse(spec, run).variance_axes()}
    assert rows["shift"]["share"] == pytest.approx(1.0)
    assert rows["inert"]["share"] == pytest.approx(0.0)
    assert rows["inert"]["levels"] == 2


def test_the_shares_of_a_balanced_additive_grid_add_up():
    # Two decisions that each shift the estimate by a constant, on a balanced
    # 2x2 grid: a one-way decomposition is exhaustive here, so the shares do
    # add up. Estimates 0.2, 0.3, 0.4, 0.5; the second decision accounts for
    # 0.8 of the variance and the first for 0.2.
    spec = AnalysisSpec(decisions=(
        Decision(name="a", values=(0.0, 0.1), default=0.0,
                 reasons={0.0: "small", 0.1: "a bit larger"}),
        Decision(name="b", values=(0.0, 0.2), default=0.0,
                 reasons={0.0: "none", 0.2: "a lot larger"}),
    ))

    def run(choices):
        est = 0.2 + float(choices["a"]) + float(choices["b"])
        return Fake(est, est - 0.1, est + 0.1)

    rows = {r["decision"]: r for r in multiverse(spec, run).variance_axes()}
    assert rows["b"]["share"] == pytest.approx(0.8)
    assert rows["a"]["share"] == pytest.approx(0.2)
    assert sum(r["share"] for r in rows.values()) == pytest.approx(1.0)


def test_an_interaction_leaves_part_of_the_spread_unattributed():
    # sign * (0.2 + shift) has an interaction: the shift matters for one sign
    # and cancels for the other. A one-way decomposition cannot attribute an
    # interaction, so the shares sum to less than one -- 0.9615 here. That gap
    # is the reason the decomposition is called descriptive rather than
    # causal, and it is asserted instead of being smoothed over.
    rows = multiverse(_spec(), _run).variance_axes()
    total = sum(r["share"] for r in rows)
    assert total == pytest.approx(0.9615384615384613)
    assert total < 1.0


def test_the_decomposition_is_labelled_descriptive():
    rows = multiverse(_spec(), _run).variance_axes()
    assert all("descriptive" in r["note"] for r in rows)
    assert all("level_means" in r for r in rows)


def test_axes_are_ordered_by_how_much_they_move_the_estimate():
    rows = multiverse(_spec(), _run).variance_axes()
    assert rows == sorted(rows, key=lambda r: -r["between_variance"])


# ------------------------------------------------------------------ plumbing

def test_a_subset_can_be_run_and_is_counted_as_considered():
    subset = _spec().expand()[:2]
    result = multiverse(_spec(), _run, specs=subset)
    assert result.counts() == {"considered": 2, "run": 2, "usable": 2,
                               "failed": 0, "flagged": 0}


def test_the_result_round_trips_through_json():
    payload = json.loads(json.dumps(multiverse(_spec(), _run).to_dict()))
    assert payload["counts"]["usable"] == 4
    assert len(payload["curve"]) == 4
    assert payload["decisions"] == ["shift", "sign"]


def test_a_custom_extract_is_used():
    calls = []

    def extract(result):
        calls.append(result)
        return 1.0, 0.5, 1.5

    result = multiverse(_spec(), _run, extract=extract)
    assert len(calls) == 4
    assert result.summary()["estimates"]["min"] == pytest.approx(1.0)


def test_the_result_is_printable_and_names_the_failures():
    def run(choices):
        if choices["sign"] == -1:
            raise ValueError("broken")
        return _run(choices)

    text = str(multiverse(_spec(), run))
    assert "2 usable, 2 failed" in text
    assert "failed specifications:" in text
    assert "broken" in text


def test_multiverse_on_a_real_fit_grid():
    y = [0.1, 0.3, 0.2, 0.5, 0.4, 0.2]
    v = [0.04, 0.05, 0.03, 0.06, 0.05, 0.04]
    spec = AnalysisSpec(decisions=(
        Decision(name="method", values=("REML", "DL"), default="REML",
                 reasons={"REML": "a", "DL": "b"}),
        Decision(name="test", values=("z", "knha"), default="z",
                 reasons={"z": "a", "knha": "b"}),
    ))
    result = multiverse(spec, lambda c: ma.rma(y, vi=v, **c))
    assert isinstance(result, MultiverseResult)
    assert result.counts()["usable"] == 4
    summary = result.summary()
    assert math.isfinite(summary["estimates"]["range"])
    assert summary["sign"]["agreement"] == 1.0


# --------------------------------------------------------------- diagnostics

class _Diag:
    """The two attributes of a Diagnosis this module reads."""

    def __init__(self, worst: str | None, codes: tuple[str, ...]):
        self.worst = worst
        self._codes = codes

    def codes(self) -> list[str]:
        return list(self._codes)


class _Fitted(Fake):
    def __init__(self, est: float, worst: str | None, codes: tuple[str, ...]):
        super().__init__(est, est - 0.1, est + 0.1)
        self.diagnostics = _Diag(worst, codes)


def test_an_error_severity_fit_is_flagged_and_kept_out_of_the_curve():
    # A fit that says its own numbers are not reportable must not contribute to
    # a stability statistic. Before this, "usable" only meant "produced a finite
    # number", so a non-converged specification counted the same as a good one.
    def run(choices):
        if choices["sign"] == -1:
            return _Fitted(-0.2, "error", ("not_converged",))
        return _Fitted(0.2, None, ())

    result = multiverse(_spec(), run)
    assert result.counts()["usable"] == 2
    assert result.counts()["flagged"] == 2
    assert result.counts()["failed"] == 0
    excluded = [f for f in result.flagged if f["excluded"]]
    assert len(excluded) == 2
    assert excluded[0]["codes"] == ["not_converged"]
    assert all(row["estimate"] > 0 for row in result.curve())


def test_require_clean_false_keeps_the_flagged_fits_but_still_names_them():
    def run(choices):
        return _Fitted(0.2, "error", ("not_converged",))

    result = multiverse(_spec(), run, require_clean=False)
    assert result.counts()["usable"] == 4
    assert result.counts()["flagged"] == 4
    assert all(not f["excluded"] for f in result.flagged)


def test_a_warning_severity_fit_stays_usable():
    def run(choices):
        return _Fitted(0.2, "warning", ("few_studies",))

    result = multiverse(_spec(), run)
    assert result.counts()["usable"] == 4
    assert result.counts()["flagged"] == 4
    assert all(not f["excluded"] for f in result.flagged)
    assert "few_studies" in str(result)


def test_a_result_without_diagnostics_is_neither_flagged_nor_excluded():
    result = multiverse(_spec(), _run)
    assert result.counts()["flagged"] == 0
    assert result.flagged == []


def test_a_flagged_specification_is_never_also_a_failure():
    # `flagged` and `failed` must not overlap, or `counts()` cannot be
    # reconciled: run = usable + failed + excluded.
    def run(choices):
        if choices["sign"] == -1:
            return _Fitted(0.2, "error", ("not_converged",))
        return Fake(float("nan"), 0.0, 1.0)

    result = multiverse(_spec(), run)
    counts = result.counts()
    excluded = sum(1 for f in result.flagged if f["excluded"])
    assert counts["run"] == counts["usable"] + counts["failed"] + excluded
    assert counts["flagged"] == 2 and counts["failed"] == 2


def test_the_flagged_entries_reach_the_export():
    def run(choices):
        return _Fitted(0.2, "error", ("not_converged",))

    payload = json.loads(json.dumps(multiverse(_spec(), run).to_dict()))
    assert payload["counts"]["flagged"] == 4
    assert payload["flagged"][0]["codes"] == ["not_converged"]
    assert payload["summary"]["flagged"]


# ---------------------------------------------------------- diagnostics block

def test_a_clean_grid_has_no_diagnostics():
    result = multiverse(_spec(), _run)
    assert result.diagnostics.codes() == []
    assert result.diagnostics.worst is None


def test_partial_failure_is_a_warning_naming_the_count():
    def run(choices):
        if choices["sign"] == 1:
            raise RuntimeError("boom")
        return Fake(0.2, 0.1, 0.3)

    result = multiverse(_spec(), run)
    diag = result.diagnostics
    assert diag.codes() == ["specification_failed"]
    assert diag.worst == "warning"
    assert "2 of 4" in str(diag)


def test_excluded_specifications_are_reported_as_such():
    def run(choices):
        if choices["sign"] == -1:
            return _Fitted(0.2, "error", ("not_converged",))
        return Fake(0.2, 0.1, 0.3)

    diag = multiverse(_spec(), run).diagnostics
    assert diag.codes() == ["specification_excluded"]
    assert diag.worst == "warning"


def test_a_grid_with_nothing_usable_is_an_error_not_an_empty_curve():
    # An empty curve with no diagnostic reads as "no spread", which is the
    # opposite of the truth: nothing was measured.
    def run(choices):
        raise RuntimeError("boom")

    result = multiverse(_spec(), run)
    assert result.counts()["usable"] == 0
    assert result.diagnostics.worst == "error"
    assert result.diagnostics.codes() == ["no_usable_specification"]
    assert result.summary()["estimates"] is None
