"""The CI sharding tool: parsing pytest's duration report and balancing it.

The tool exists so the test suite can be split into equally *timed* shards
without adding a dependency, so its two pure functions are worth pinning: a
mis-parsed report would silently produce shards of wildly unequal length, which
looks like a slow CI rather than like a bug.
"""

from __future__ import annotations

import importlib.util
import json
import pathlib


def _load_tool():
    """Import ``tools/shard_tests.py`` without putting ``tools`` on sys.path."""
    path = pathlib.Path(__file__).resolve().parents[1] / "tools" / "shard_tests.py"
    spec = importlib.util.spec_from_file_location("shard_tests", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


shard_tests = _load_tool()


# --- a real `pytest --durations=0` report, abridged ------------------------
REPORT = """\
============================= slowest durations ==============================
9.50s call     tests/test_copas.py::test_fleiss_surface_matches_metasens
2.00s teardown tests/test_copas.py::test_fleiss_surface_matches_metasens
0.50s setup    tests/test_copas.py::test_fleiss_surface_matches_metasens
3.25s call     tests/test_mv.py::test_a[1-2]
0.00s call     tests/test_mv.py::test_b
========================= 3 passed in 15.25s ==============================
"""


def test_durations_are_summed_over_the_three_phases():
    totals = shard_tests.parse_durations(REPORT)
    assert totals["tests/test_copas.py::test_fleiss_surface_matches_metasens"] == 12.0
    assert totals["tests/test_mv.py::test_a[1-2]"] == 3.25
    assert totals["tests/test_mv.py::test_b"] == 0.0


def test_the_summary_line_is_not_a_test():
    totals = shard_tests.parse_durations(REPORT)
    assert len(totals) == 3
    assert not any("passed" in key for key in totals)


def test_an_empty_report_gives_no_durations():
    assert shard_tests.parse_durations("") == {}


def test_assignment_balances_time_rather_than_count():
    weights = {"slow": 100.0, "a": 30.0, "b": 30.0, "c": 30.0, "d": 10.0}
    shards = shard_tests.assign(weights, 2)
    loads = [sum(weights[t] for t in shard) for shard in shards]
    # the greedy rule puts 'slow' alone and everything else opposite it
    assert sorted(loads) == [100.0, 100.0]


def test_assignment_is_stable_and_covers_every_test():
    weights = {f"t{i}": float(i) for i in range(10)}
    first = shard_tests.assign(weights, 3)
    second = shard_tests.assign(dict(reversed(list(weights.items()))), 3)
    assert first == second                       # no dependence on dict order
    flat = [t for shard in first for t in shard]
    assert sorted(flat) == sorted(weights)
    assert len(first) == 3


def test_equal_weights_split_by_count():
    weights = {f"t{i}": 1.0 for i in range(9)}
    shards = shard_tests.assign(weights, 3)
    assert [len(shard) for shard in shards] == [3, 3, 3]


def test_assignment_rejects_a_silly_shard_count():
    try:
        shard_tests.assign({"a": 1.0}, 0)
    except ValueError as exc:
        assert "at least 1" in str(exc)
    else:                                        # pragma: no cover
        raise AssertionError("expected a ValueError")


def test_record_writes_the_map_it_parsed(tmp_path):
    log = tmp_path / "run.log"
    log.write_text(REPORT, encoding="utf-8")
    out = tmp_path / "durations.json"
    assert shard_tests.main(["--record", str(log), "--out", str(out)]) == 0
    written = json.loads(out.read_text(encoding="utf-8"))
    assert written["tests/test_mv.py::test_a[1-2]"] == 3.25
