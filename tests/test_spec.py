"""Tests for the analysis specification.

The ids are the part worth testing hardest. A specification id is what lets two
runs be diffed and a recorded result be recognised, so it has to depend on the
choices and on nothing else: not on the order the decisions were declared, not on
the prose in the justifications, not on the order the values happen to hash in.

The refusals matter as much. A specification that accepts a one-valued
"decision", or a value with no justification, is a specification that records
less than the reader thinks it does.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

import pimeta as ma
from pimeta.spec import SPEC_VERSION, AnalysisSpec, Decision, RunOutcome, Specification, execute


def _method() -> Decision:
    return Decision(
        name="method",
        values=("REML", "DL", "FE"),
        reasons={
            "REML": "the default in this package; less biased than DL at small k",
            "DL": "what most published meta-analyses used, so it aids comparison",
            "FE": "the assumption that there is no heterogeneity at all",
        },
        default="REML",
    )


def _test() -> Decision:
    return Decision(
        name="test",
        values=("z", "knha"),
        reasons={
            "z": "the nominal reference; slightly liberal at small k",
            "knha": "wider when the studies disagree; better coverage at small k",
        },
        default="z",
    )


@pytest.fixture()
def spec() -> AnalysisSpec:
    return AnalysisSpec(decisions=(_method(), _test()))


# ------------------------------------------------------------------ decisions

def test_a_decision_needs_two_values():
    with pytest.raises(ValueError, match="not a decision"):
        Decision(name="method", values=("REML",), reasons={"REML": "the default"})


def test_every_value_needs_a_justification():
    with pytest.raises(ValueError, match="no justification"):
        Decision(name="method", values=("REML", "DL"), reasons={"REML": "fine"})


def test_a_blank_justification_is_not_a_justification():
    with pytest.raises(ValueError, match="no justification"):
        Decision(name="method", values=("REML", "DL"),
                 reasons={"REML": "fine", "DL": "   "})


def test_a_repeated_value_is_refused():
    with pytest.raises(ValueError, match="repeats a value"):
        Decision(name="m", values=("REML", "REML"),
                 reasons={"REML": "fine"})


def test_a_default_must_be_one_of_the_values():
    with pytest.raises(ValueError, match="not one of its values"):
        Decision(name="m", values=("REML", "DL"),
                 reasons={"REML": "a", "DL": "b"}, default="FE")


def test_decision_round_trips_through_a_dict():
    d = _method()
    again = Decision.from_dict(d.to_dict())
    assert again == d


# ---------------------------------------------------------------- the grid

def test_size_is_the_product_of_the_decisions(spec):
    assert spec.size() == 6
    assert len(spec.expand()) == 6


def test_every_combination_appears_exactly_once(spec):
    seen = {tuple(sorted(s.as_dict().items())) for s in spec.expand()}
    assert len(seen) == 6
    assert {s.get("method") for s in spec.expand()} == {"REML", "DL", "FE"}
    assert {s.get("test") for s in spec.expand()} == {"z", "knha"}


def test_specification_ids_are_stable_across_runs(spec):
    assert [s.id for s in spec.expand()] == [s.id for s in spec.expand()]


def test_ids_do_not_depend_on_the_order_the_decisions_were_declared():
    a = AnalysisSpec(decisions=(_method(), _test()))
    b = AnalysisSpec(decisions=(_test(), _method()))
    ids_a = {tuple(sorted(s.as_dict().items())): s.id for s in a.expand()}
    ids_b = {tuple(sorted(s.as_dict().items())): s.id for s in b.expand()}
    assert ids_a == ids_b


def test_ids_do_not_depend_on_the_prose():
    # Rewriting a justification must not invalidate a recorded run: the id is a
    # hash of the decisions, not of the documentation.
    edited = Decision(name=_method().name, values=_method().values,
                      reasons={v: "rewritten" for v in _method().values},
                      default="REML")
    a = AnalysisSpec(decisions=(_method(), _test()))
    b = AnalysisSpec(decisions=(edited, _test()))
    assert [s.id for s in a.expand()] == [s.id for s in b.expand()]


def test_ids_differ_when_a_choice_differs(spec):
    ids = [s.id for s in spec.expand()]
    assert len(set(ids)) == len(ids)


def test_ids_are_sha256_digests(spec):
    for s in spec.expand():
        assert s.id.startswith("sha256:")
        assert len(s.id) == len("sha256:") + 32


def test_the_default_needs_a_default_for_every_decision():
    partial = Decision(name="m", values=("a", "b"), reasons={"a": "x", "b": "y"})
    spec = AnalysisSpec(decisions=(partial,))
    with pytest.raises(ValueError, match="no default declared"):
        spec.default()


def test_the_default_is_the_declared_values(spec):
    one = spec.default()
    assert one.as_dict() == {"method": "REML", "test": "z"}
    assert one.id in {s.id for s in spec.expand()}


def test_lookup_by_id_and_by_choices_agree(spec):
    one = spec.expand()[4]
    assert spec.get(one.id) == one
    assert spec.get(one.as_dict()) == one


def test_lookup_rejects_an_unknown_id(spec):
    with pytest.raises(KeyError, match="no specification with id"):
        spec.get("sha256:" + "0" * 32)


def test_lookup_rejects_an_inadmissible_value(spec):
    with pytest.raises(ValueError, match="not an admissible value"):
        spec.get({"method": "PM", "test": "z"})


def test_lookup_rejects_missing_and_unknown_decisions(spec):
    with pytest.raises(KeyError, match="missing decision"):
        spec.get({"method": "REML"})
    with pytest.raises(KeyError, match="unknown decision"):
        spec.get({"method": "REML", "test": "z", "extra": 1})


def test_two_decisions_may_not_share_a_name():
    with pytest.raises(ValueError, match="share a name"):
        AnalysisSpec(decisions=(_method(), _method()))


def test_an_empty_specification_is_refused():
    with pytest.raises(ValueError, match="at least one decision"):
        AnalysisSpec(decisions=())


# ---------------------------------------------------------------- file i/o

def test_round_trip_through_a_file(spec, tmp_path: Path):
    path = tmp_path / "nested" / "spec.json"
    spec.save(path)
    again = AnalysisSpec.load(path)
    assert again.to_dict() == spec.to_dict()
    assert [s.id for s in again.expand()] == [s.id for s in spec.expand()]


def test_the_file_records_its_own_size(spec, tmp_path: Path):
    path = spec.save(tmp_path / "spec.json")
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["kind"] == "pimeta.analysis_spec"
    assert payload["version"] == SPEC_VERSION
    assert payload["size"] == spec.size()


def test_a_hand_edited_size_is_refused(spec, tmp_path: Path):
    path = spec.save(tmp_path / "spec.json")
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["size"] = 99
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="edited by hand"):
        AnalysisSpec.load(path)


def test_a_future_version_is_refused_rather_than_guessed(tmp_path: Path):
    payload = {"kind": "pimeta.analysis_spec", "version": SPEC_VERSION + 1,
               "decisions": [_method().to_dict()]}
    with pytest.raises(ValueError, match="cannot be read"):
        AnalysisSpec.from_dict(payload)


def test_a_foreign_file_is_refused(tmp_path: Path):
    with pytest.raises(ValueError, match="not an analysis specification"):
        AnalysisSpec.from_dict({"kind": "something else"})


def test_the_specification_prints_its_reasons(spec):
    text = str(spec)
    assert "6 specification(s)" in text
    assert "REML" in text and "less biased than DL" in text


# ------------------------------------------------------------------ running

def _fit_run(choices):
    y = [0.1, 0.3, 0.2, 0.5]
    v = [0.04, 0.05, 0.03, 0.06]
    return ma.rma(y, vi=v, method=choices["method"], test=choices["test"])


def test_execute_returns_one_outcome_per_specification(spec):
    outcomes = execute(spec, _fit_run)
    assert len(outcomes) == spec.size()
    assert all(o.ok for o in outcomes)
    assert {o.id for o in outcomes} == {s.id for s in spec.expand()}


def test_execute_passes_the_choices_it_was_given(spec):
    seen = []

    def run(choices):
        seen.append(dict(choices))
        return None

    execute(spec, run)
    assert len(seen) == 6
    assert all(set(c) == {"method", "test"} for c in seen)
    assert len({tuple(sorted(c.items())) for c in seen}) == 6


def test_execute_records_a_failure_instead_of_dropping_it():
    spec = AnalysisSpec(decisions=(Decision(
        name="k", values=(4, 1),
        reasons={4: "enough studies", 1: "too few, which is the point"},
        default=4),))

    def run(choices):
        if choices["k"] < 2:
            raise ValueError("a meta-analysis needs at least two studies")
        return ma.rma([0.1, 0.3, 0.2, 0.5], vi=[0.04, 0.05, 0.03, 0.06])

    outcomes = execute(spec, run)
    assert [o.ok for o in outcomes] == [True, False]
    failed = outcomes[1]
    assert "at least two studies" in failed.error
    assert failed.error.startswith("ValueError:")
    assert failed.result is None


def test_execute_can_raise_instead_when_the_caller_cannot_continue(spec):
    def run(choices):
        raise RuntimeError("boom")

    with pytest.raises(RuntimeError, match="boom"):
        execute(spec, run, on_error="raise")


def test_execute_rejects_an_unknown_error_policy(spec):
    with pytest.raises(ValueError, match="on_error"):
        execute(spec, _fit_run, on_error="ignore")


def test_outcomes_carry_the_diagnostic_codes_of_the_fit(spec):
    outcomes = execute(spec, _fit_run)
    for o in outcomes:
        assert isinstance(o.diagnostics, tuple)
        assert all(isinstance(c, str) for c in o.diagnostics)


def test_an_outcome_round_trips_and_keeps_the_choice_that_failed():
    spec = AnalysisSpec(decisions=(Decision(
        name="k", values=(4, 1), reasons={4: "ok", 1: "fails"}, default=4),))
    outcomes = execute(spec, lambda c: (_ for _ in ()).throw(ValueError("no")))
    payload = json.loads(json.dumps(outcomes[0].to_dict()))
    assert payload["ok"] is False
    assert payload["choices"] == {"k": 4}
    assert payload["error"].startswith("ValueError")


def test_execute_can_run_a_subset(spec):
    subset = spec.expand()[:2]
    outcomes = execute(spec, _fit_run, specs=subset)
    assert [o.id for o in outcomes] == [s.id for s in subset]


def test_specifications_are_the_objects_the_docs_promise(spec):
    one = spec.expand()[0]
    assert isinstance(one, Specification)
    assert isinstance(spec, AnalysisSpec)
    assert isinstance(RunOutcome(id="x", choices={}, ok=True), RunOutcome)
