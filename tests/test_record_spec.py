"""A saved fit must be able to name the specification that produced it.

Stage 6 asks that "each result names the specification that produced it". That
is the link that makes a multiverse auditable: given one of ``M`` fits, which of
the declared analyses is it?

The record layer deliberately does not import :mod:`pimeta.spec`. It accepts the
object (and reads ``id``/``as_dict``), the id string, or a mapping, so the record
format does not depend on the specification module existing -- and so a script
that only has the id can still write a correct record.

Attaching a specification must not change the analysis fingerprint: the choices
are already in ``settings`` whenever they can move a number, and a record that
verified before must still verify after. That is asserted here rather than
assumed, because silently invalidating every stored record would be an
expensive way to learn it.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

import pimeta as ma
from pimeta.record import build_record, load_record, save_record
from pimeta.spec import AnalysisSpec, Decision


@pytest.fixture()
def data():
    return {"yi": np.array([0.10, 0.25, 0.40, 0.55, 0.70]),
            "vi": np.array([0.04, 0.05, 0.06, 0.03, 0.08]),
            "study": [f"S{i}" for i in range(5)]}


@pytest.fixture()
def fit(data):
    return ma.rma(data["yi"], vi=data["vi"], method="DL")


@pytest.fixture()
def spec() -> AnalysisSpec:
    return AnalysisSpec(decisions=(
        Decision(name="method", values=("REML", "DL"), default="REML",
                 reasons={"REML": "a", "DL": "b"}),
    ))


def test_a_record_without_a_specification_says_so(fit, data):
    record = build_record(fit, data=data, function="rma")
    assert record.specification is None


def test_attaching_the_object_records_its_id_and_choices(fit, data, spec):
    one = spec.get({"method": "DL"})
    record = build_record(fit, data=data, function="rma", specification=one)
    block = record.specification
    assert block is not None
    assert block["id"] == one.id
    assert block["choices"] == {"method": "DL"}


def test_a_bare_id_is_accepted(fit, data, spec):
    one = spec.get({"method": "DL"})
    record = build_record(fit, data=data, specification=one.id)
    assert record.specification == {"id": one.id}


def test_a_mapping_is_accepted(fit, data):
    record = build_record(fit, data=data,
                          specification={"id": "sha256:" + "a" * 32,
                                         "choices": {"method": "DL"}})
    assert record.specification["choices"] == {"method": "DL"}


def test_something_that_cannot_name_a_specification_is_refused(fit, data):
    with pytest.raises(TypeError, match="specification must be"):
        build_record(fit, data=data, specification=42)


def test_attaching_a_specification_does_not_move_the_fingerprints(fit, data, spec):
    # The record has to verify either way: the specification is provenance, not
    # a setting, and adding one must not invalidate records already on disk.
    plain = build_record(fit, data=data, function="rma")
    tagged = build_record(fit, data=data, function="rma",
                          specification=spec.get({"method": "DL"}))
    assert plain.inputs["fingerprint"] == tagged.inputs["fingerprint"]
    assert plain.analysis["settings"] == tagged.analysis["settings"]


def test_the_specification_survives_the_file_and_the_replay(fit, data, spec,
                                                           tmp_path: Path):
    one = spec.get({"method": "DL"})
    path = tmp_path / "m1.json"
    save_record(path, fit, data=data, mode="bundle", function="rma",
                call={"yi": "yi", "vi": "vi", "method": "DL"},
                specification=one)
    reloaded = load_record(path)
    assert reloaded.specification["id"] == one.id
    assert reloaded.specification["choices"] == {"method": "DL"}
    # The provenance block must not disturb replay.
    replay = reloaded.replay()
    assert replay is not None


def test_the_record_is_still_plain_json(fit, data, spec):
    record = build_record(fit, data=data, specification=spec.get({"method": "DL"}))
    payload = json.loads(json.dumps(record.to_dict()))
    assert payload["analysis"]["specification"]["choices"] == {"method": "DL"}


def test_a_multiverse_outcome_can_be_traced_back_to_its_record(data, spec):
    # The end-to-end point of the feature: run the grid, save one fit's record
    # with the id the outcome carries, and recover the choices from the record.
    def run(choices):
        return ma.rma(data["yi"], vi=data["vi"], method=choices["method"])

    outcomes = ma.execute(spec, run)
    outcome = outcomes[-1]
    assert outcome.ok
    record = build_record(outcome.result, data=data, function="rma",
                          specification=outcome.id)
    assert record.specification["id"] == outcome.id
    assert record.specification["id"] == spec.get({"method": "DL"}).id
