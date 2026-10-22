"""Tests for the standardised, traceable I/O layer (``pimeta.record``).

Three properties are being defended here:

1. **Portable output** -- every file the package writes is strict JSON (no bare
   ``NaN``/``Infinity``), so a reader that is not Python can open it.
2. **Honest fingerprints** -- relabelling a study must not change what
   "the same analysis" means, while changing a number must.
3. **Round trips** -- what was written can be read back, verified and replayed.
"""

from __future__ import annotations

import json
import math

import numpy as np
import pytest

import pimeta as ma
from pimeta.record import (
    RECORD_SCHEMA,
    AnalysisRecord,
    EstimatesStore,
    RecordError,
    RecordMismatch,
    analysis_fingerprint,
    build_record,
    compare_results,
    read_table,
    source_fingerprint,
    write_table,
)
from pimeta.record.jsonio import (
    JSONError,
    canonical,
    dump,
    dumps,
    from_jsonable,
    load,
    loads,
    to_jsonable,
)

DATA = {
    "study": ["S1", "S2", "S3", "S4", "S5"],
    "TE": [0.20, 0.40, 0.10, 0.50, 0.30],
    "seTE2": [0.04, 0.05, 0.03, 0.06, 0.02],
    "dose": [1.0, 2.0, 3.0, 4.0, 5.0],
}
CALL = {"yi": "TE", "vi": "seTE2", "method": "REML"}


@pytest.fixture()
def fit():
    return ma.rma(DATA["TE"], vi=DATA["seTE2"], method="REML", slab=DATA["study"])


# --------------------------------------------------------------------------- #
# jsonio: strict and lossless
# --------------------------------------------------------------------------- #
def test_non_finite_floats_never_reach_the_output_as_bare_tokens():
    text = dumps({"a": float("nan"), "b": float("inf"), "c": float("-inf")})
    assert "NaN" not in text
    assert "Infinity" not in text
    # The text is valid JSON under a strict parser too.
    json.loads(text, parse_constant=lambda token: pytest.fail(f"invalid {token}"))


def test_non_finite_floats_survive_a_round_trip():
    back = loads(dumps({"a": float("nan"), "b": float("inf"), "c": float("-inf")}))
    assert math.isnan(back["a"])
    assert back["b"] == math.inf
    assert back["c"] == -math.inf


def test_nan_is_not_confused_with_missing():
    back = loads(dumps({"present": None, "undefined": float("nan")}))
    assert back["present"] is None
    assert math.isnan(back["undefined"])


def test_negative_zero_keeps_its_sign():
    back = loads(dumps({"z": -0.0}))
    assert back["z"] == 0.0
    assert math.copysign(1.0, back["z"]) == -1.0


def test_numpy_scalars_are_written_as_primitives():
    payload = to_jsonable({"i": np.int64(3), "f": np.float32(0.5), "b": np.bool_(True)})
    assert payload == {"i": 3, "f": 0.5, "b": True}
    assert [type(v) for v in payload.values()] == [int, float, bool]


def test_arrays_are_lists_by_default_and_keep_dtype_when_tagged():
    arr = np.arange(6, dtype="int16").reshape(2, 3)
    plain = loads(dumps({"a": arr}))
    assert plain["a"] == [[0, 1, 2], [3, 4, 5]]

    tagged = loads(dumps({"a": arr}, arrays="tagged"))["a"]
    assert tagged.dtype == np.dtype("int16")
    assert tagged.shape == (2, 3)
    np.testing.assert_array_equal(tagged, arr)


def test_unrepresentable_types_fail_loudly():
    class Opaque:
        pass

    with pytest.raises(JSONError, match="Opaque"):
        dumps({"x": Opaque()})


def test_dumps_writes_utf8_without_escaping_it():
    assert "统计分析" in dumps({"k": "统计分析"})


def test_dump_is_atomic_and_leaves_no_temporary_files(tmp_path):
    target = tmp_path / "out" / "record.json"
    dump({"a": 1}, target)
    assert target.exists()
    assert [p.name for p in target.parent.iterdir()] == ["record.json"]
    assert load(target) == {"a": 1}


def test_dump_normalises_line_endings_so_records_hash_the_same_everywhere(tmp_path):
    target = tmp_path / "r.json"
    dump({"a": [1, 2, 3]}, target)
    assert b"\r\n" not in target.read_bytes()
    assert target.read_text(encoding="utf-8").endswith("\n")


def test_canonical_is_key_order_independent_and_stable():
    assert canonical({"b": 1, "a": 2}) == canonical({"a": 2, "b": 1})
    assert canonical({"x": 0.1}) == canonical({"x": 0.1})


def test_from_jsonable_inverts_to_jsonable():
    value = {"n": float("nan"), "a": [1, 2], "s": "x"}
    back = from_jsonable(to_jsonable(value))
    assert math.isnan(back["n"])
    assert back["a"] == [1, 2]
    assert back["s"] == "x"


# --------------------------------------------------------------------------- #
# fingerprints
# --------------------------------------------------------------------------- #
def test_fingerprint_is_deterministic_across_calls():
    assert source_fingerprint(DATA) == source_fingerprint(DATA)
    assert source_fingerprint(DATA).startswith("sha256:")


def test_relabelling_a_study_changes_the_source_fingerprint_only():
    relabelled = {**DATA, "study": ["A", "B", "C", "D", "E"]}
    assert source_fingerprint(relabelled) != source_fingerprint(DATA)
    assert (
        analysis_fingerprint(relabelled, columns=["TE", "seTE2"], settings={"method": "REML"})
        == analysis_fingerprint(DATA, columns=["TE", "seTE2"], settings={"method": "REML"})
    )


def test_column_order_does_not_affect_the_analysis_fingerprint():
    settings = {"method": "REML"}
    forward = analysis_fingerprint(DATA, columns=["TE", "seTE2"], settings=settings)
    backward = analysis_fingerprint(DATA, columns=["seTE2", "TE"], settings=settings)
    assert forward == backward


def test_changing_a_number_or_a_setting_changes_the_analysis_fingerprint():
    settings = {"method": "REML"}
    base = analysis_fingerprint(DATA, columns=["TE", "seTE2"], settings=settings)
    moved = {**DATA, "TE": [0.20, 0.40, 0.10, 0.50, 0.31]}
    assert analysis_fingerprint(moved, columns=["TE", "seTE2"], settings=settings) != base
    assert analysis_fingerprint(DATA, columns=["TE", "seTE2"], settings={"method": "DL"}) != base


def test_fingerprint_rejects_unknown_columns():
    with pytest.raises(KeyError, match="nope"):
        source_fingerprint(DATA, columns=["nope"])


# --------------------------------------------------------------------------- #
# table I/O
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("suffix", [".csv", ".tsv", ".json", ".jsonl", ".npz"])
def test_every_writer_has_a_reader_that_returns_the_same_columns(tmp_path, suffix):
    path = write_table(tmp_path / f"t{suffix}", DATA)
    back = read_table(path)
    assert list(back) == list(DATA)
    for name in DATA:
        left = [float(v) for v in back[name]] if name != "study" else list(back[name])
        right = DATA[name]
        assert left == right, (suffix, name)


def test_text_output_keeps_full_float_precision(tmp_path):
    values = {"x": [1 / 3, 0.1 + 0.2, 1e-17]}
    path = write_table(tmp_path / "t.csv", values)
    back = [float(v) for v in read_table(path)["x"]]
    assert back == values["x"]          # repr round-trips exactly


def test_text_output_can_be_rounded_on_request(tmp_path):
    path = write_table(tmp_path / "t.csv", {"x": [1 / 3]}, float_format="%.3f")
    assert read_table(path)["x"] == ["0.333"]


def test_read_table_converts_numbers_only_when_asked(tmp_path):
    path = write_table(tmp_path / "t.csv", {"x": [1, 2], "y": ["a", "b"]})
    raw = read_table(path)
    assert raw["x"] == ["1", "2"]
    inferred = read_table(path, infer=True)
    assert inferred["x"] == [1, 2]
    assert inferred["y"] == ["a", "b"]


def test_missing_cells_are_none_not_the_string_na(tmp_path):
    path = tmp_path / "t.csv"
    path.write_text("x,y\n1,\n2,NA\n", encoding="utf-8")
    cols = read_table(str(path))
    assert cols["y"] == [None, None]


def test_npz_refuses_object_columns_instead_of_pickling(tmp_path):
    with pytest.raises(ValueError, match="pickle"):
        write_table(tmp_path / "t.npz", {"x": [{"a": 1}, None]})


# --------------------------------------------------------------------------- #
# the record
# --------------------------------------------------------------------------- #
def test_record_has_the_five_blocks_and_a_schema_tag(fit):
    record = build_record(fit, data=DATA, mode="bundle", function="rma", call=CALL)
    payload = record.to_dict()
    assert payload["schema"] == RECORD_SCHEMA
    for block in ("inputs", "analysis", "process", "environment", "results"):
        assert block in payload
    assert payload["environment"]["pimeta"] == ma.__version__
    assert payload["results"]["class"] == "RmaResult"


def test_record_round_trips_through_a_file(fit, tmp_path):
    record = build_record(fit, data=DATA, mode="bundle", function="rma", call=CALL)
    path = record.save(tmp_path / "m1.json")

    assert "NaN" not in path.read_text(encoding="utf-8")
    loaded = AnalysisRecord.load(path)
    assert loaded.fingerprint == record.fingerprint
    assert loaded.kind == record.kind
    assert loaded.inputs["n_rows"] == len(DATA["TE"])


def test_bundle_record_verifies_itself(fit):
    record = build_record(fit, data=DATA, mode="bundle", function="rma", call=CALL)
    fresh = record.verify()
    assert fresh["source"] == record.fingerprint["source"]
    assert fresh["analysis"] == record.fingerprint["analysis"]


def test_tampered_bundle_is_caught(fit):
    record = build_record(fit, data=DATA, mode="bundle", function="rma", call=CALL)
    record.inputs["data"]["TE"][0] = 0.999
    with pytest.raises(RecordMismatch, match="does not match"):
        record.verify()


def test_digest_record_refuses_to_verify_without_data(fit):
    record = build_record(fit, data=DATA, mode="digest", function="rma", call=CALL)
    with pytest.raises(RecordError, match="digest"):
        record.verify()
    # ... but can be checked against the original table.
    assert record.verify(DATA)["source"] == record.fingerprint["source"]


def test_digest_mode_does_not_copy_the_data(fit):
    record = build_record(fit, data=DATA, mode="digest", function="rma", call=CALL)
    assert "data" not in record.inputs
    assert record.inputs["columns"] == list(DATA)


def test_record_without_a_source_table_verifies_against_its_vectors(fit):
    record = build_record(fit, function="rma", call=CALL)
    fresh = record.verify()
    assert fresh["basis"] == "arrays"
    assert fresh["analysis"] == record.fingerprint["analysis"]

    record.inputs["arrays"]["yi"][0] += 0.5
    with pytest.raises(RecordMismatch, match="no longer hash"):
        record.verify()


def test_bundle_record_replays_the_same_numbers(fit):
    record = build_record(fit, data=DATA, mode="bundle", function="rma", call=CALL)
    replayed = record.replay()
    assert replayed.b[0] == pytest.approx(fit.b[0], abs=1e-12)
    assert replayed.tau2 == pytest.approx(fit.tau2, abs=1e-12)


def test_replay_without_a_call_says_so(fit):
    record = build_record(fit, data=DATA, mode="bundle", function="rma")
    with pytest.raises(RecordError, match="nothing to replay"):
        record.replay()


def test_replay_without_a_function_says_so(fit):
    record = build_record(fit, data=DATA, mode="bundle", call=CALL)
    with pytest.raises(RecordError, match="analysis.function"):
        record.replay()


def test_replay_refuses_a_name_outside_pimeta(fit):
    record = build_record(
        fit, data=DATA, mode="bundle", function="os.system", call={"command": "echo hi"}
    )
    with pytest.raises(RecordError, match="not part of pimeta"):
        record.replay()


def test_replay_flags_a_changed_estimate(fit):
    record = build_record(fit, data=DATA, mode="bundle", function="rma", call=CALL)
    record.results["values"]["b"] = [999.0]
    with pytest.raises(RecordMismatch, match="replayed"):
        record.replay()
    # Non-strict replay still returns the fitted model.
    assert record.replay(strict=False).b[0] == pytest.approx(fit.b[0], abs=1e-12)


def test_record_rejects_an_unknown_schema():
    with pytest.raises(RecordError, match="unsupported record schema"):
        AnalysisRecord.from_dict(
            {"schema": "something-else/9", "kind": "rma", "inputs": {}, "analysis": {}}
        )


def test_record_rejects_a_payload_that_is_not_a_record():
    with pytest.raises(RecordError, match="missing"):
        AnalysisRecord.from_dict({"schema": RECORD_SCHEMA})


def test_summary_mentions_the_fingerprints(fit):
    record = build_record(fit, data=DATA, mode="bundle", function="rma", call=CALL)
    text = record.summary()
    assert record.fingerprint["source"] in text
    assert record.fingerprint["analysis"] in text


def test_compare_results_ignores_non_numeric_differences():
    left = {"b": [0.5], "slab": ["S1"], "class": "RmaResult"}
    right = {"b": [0.5], "slab": ["Renamed"], "class": "RmaResult"}
    assert compare_results(left, right) == []


def test_compare_results_treats_nan_as_a_match():
    assert compare_results({"I2": float("nan")}, {"I2": float("nan")}) == []


def test_compare_results_reports_the_offending_path():
    differences = compare_results({"a": {"b": [1.0]}}, {"a": {"b": [1.5]}})
    assert differences and differences[0][0] == "a.b[0]"


# --------------------------------------------------------------------------- #
# estimates store
# --------------------------------------------------------------------------- #
def test_store_keeps_models_under_names(fit):
    store = EstimatesStore("primary")
    store.store("fe", ma.rma(DATA["TE"], vi=DATA["seTE2"], method="FE"))
    store.store("re", fit)
    assert store.keys() == ["fe", "re"]
    assert len(store) == 2
    assert "re" in store
    assert store.restore("re") is fit


def test_store_replaces_a_duplicate_name(fit):
    store = EstimatesStore()
    store.store("m", fit)
    other = ma.rma(DATA["TE"], vi=DATA["seTE2"], method="FE")
    store.store("m", other)
    assert len(store) == 1
    assert store.restore("m") is other


def test_store_needs_a_name(fit):
    with pytest.raises(ValueError, match="non-empty"):
        EstimatesStore().store("  ", fit)


def test_compare_lines_up_the_coefficient_tables(fit):
    store = EstimatesStore()
    store.store("fe", ma.rma(DATA["TE"], vi=DATA["seTE2"], method="FE"))
    store.store("re", fit)
    table = store.compare()
    assert table["model"] == ["fe", "re"]
    assert len(table["estimate"]) == 2
    assert table["ci_lb"][0] < table["estimate"][0] < table["ci_ub"][0]


def test_compare_rejects_an_unknown_name(fit):
    store = EstimatesStore()
    store.store("re", fit)
    with pytest.raises(KeyError, match="nope"):
        store.compare("re", "nope")


def test_store_survives_a_save_and_load_but_not_its_live_objects(fit, tmp_path):
    store = EstimatesStore("primary")
    store.store("re", fit, data=DATA, mode="bundle", function="rma", call=CALL)
    path = store.save(tmp_path / "estimates.json")

    reloaded = EstimatesStore.load(path)
    assert reloaded.keys() == ["re"]
    assert reloaded.name == "primary"
    assert reloaded["re"].coef_table()["estimate"] == store["re"].coef_table()["estimate"]
    # The numbers survived; the fitted object did not, and the store says so.
    with pytest.raises(RecordError, match="only its record survives"):
        reloaded.restore("re")
    # A loaded entry can still be replayed from its record.
    assert reloaded["re"].record.replay().b[0] == pytest.approx(fit.b[0], abs=1e-12)


def test_store_summary_lists_each_model(fit):
    store = EstimatesStore("primary")
    store.store("re", fit)
    text = store.summary()
    assert "primary" in text
    assert "re" in text
    assert "tau2=" in text


def test_store_rejects_an_unknown_schema():
    with pytest.raises(RecordError, match="unsupported store schema"):
        EstimatesStore.from_dict({"schema": "other/2", "models": {}})


def test_save_without_a_path_complains(fit):
    store = EstimatesStore()
    store.store("re", fit)
    with pytest.raises(ValueError, match="pass save"):
        store.save()


# --------------------------------------------------------------------------- #
# network meta-analysis
# --------------------------------------------------------------------------- #
NETWORK = {
    "TE": [0.10, 0.20, 0.05, 0.15, 0.30, 0.25, 0.12, 0.18],
    "seTE": [0.04, 0.05, 0.03, 0.06, 0.05, 0.07, 0.04, 0.06],
    "treat1": ["A", "B", "A", "C", "A", "B", "A", "C"],
    "treat2": ["B", "C", "C", "D", "D", "D", "B", "D"],
}
NETWORK_CALL = {
    "TE": "TE",
    "seTE": "seTE",
    "treat1": "treat1",
    "treat2": "treat2",
    "sm": "MD",
}


def _network(**kwargs):
    return ma.netmeta(**NETWORK_CALL, data=NETWORK, **kwargs)


def test_a_network_fit_round_trips_through_a_bundle_record(tmp_path):
    fit = _network()
    record = build_record(fit, data=NETWORK, mode="bundle",
                          function="netmeta", call=NETWORK_CALL)
    # netmeta() takes column names, so the existing inference already sees
    # that the bundled table has to be injected again.
    assert record.analysis["pass_data"] is True

    path = record.save(tmp_path / "nma.json")
    assert "NaN" not in path.read_text(encoding="utf-8")

    loaded = AnalysisRecord.load(path)
    assert loaded.kind == "netmeta"
    assert loaded.inputs["n_rows"] == 8
    loaded.verify()

    again = loaded.replay()
    assert np.allclose(
        np.asarray(again.TE_random, dtype=float),
        np.asarray(fit.TE_random, dtype=float),
        rtol=1e-9,
        equal_nan=True,
    )
    assert again.Q == pytest.approx(fit.Q, rel=1e-12)


def test_the_tau_estimator_is_part_of_the_fingerprint():
    digests = {}
    for method in ("DL", "REML"):
        record = build_record(_network(method_tau=method), data=NETWORK,
                              mode="digest", function="netmeta",
                              call=NETWORK_CALL)
        assert record.analysis["settings"]["method_tau"] == method
        digests[method] = record.fingerprint["analysis"]
    # Different estimator, different analysis -- even though the source table
    # is byte-identical.
    assert digests["DL"] != digests["REML"]


def test_the_tau_estimator_does_not_change_the_source_fingerprint():
    digests = {
        method: build_record(_network(method_tau=method), data=NETWORK,
                             mode="digest").fingerprint["source"]
        for method in ("DL", "REML")
    }
    assert digests["DL"] == digests["REML"]


def test_network_replay_catches_an_edited_estimate(tmp_path):
    record = build_record(_network(), data=NETWORK, mode="bundle",
                          function="netmeta", call=NETWORK_CALL)
    record.save(tmp_path / "nma.json")
    loaded = AnalysisRecord.load(tmp_path / "nma.json")
    loaded.results["values"]["Q"] = 999.0
    with pytest.raises(RecordMismatch, match="replayed"):
        loaded.replay()


def test_a_network_fit_has_no_coefficient_table():
    store = EstimatesStore("networks")
    store.store("dl", _network())
    # A network has no coefficient vector, so compare() contributes no rows
    # rather than inventing a table.
    assert store["dl"].coef_table() == {}
    assert store.compare() == {}
