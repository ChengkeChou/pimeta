#!/usr/bin/env python
"""Split the test suite into equally *timed* shards.

The suite takes hours of wall clock, which is too long for a check that runs on
every push.  Splitting it by file is easy and wrong: the slowest file is orders
of magnitude slower than the fastest, so an equal number of files is an unequal
amount of work.  This tool therefore shards by recorded time.

It reads ``.test_durations.json`` -- a map of test id to seconds -- and assigns
tests to shards with the longest-processing-time heuristic: sort by cost,
largest first, and give each test to the shard that currently has the least
work.  That is within a few percent of optimal for this kind of list and takes
no time to compute.  Without the map every test costs 1 second, so the fallback
is an equal split by count, which is at least deterministic.

Usage::

    python tools/shard_tests.py --shard 2 --of 4 > shard.txt
    mapfile -t ids < shard.txt
    python -m pytest "${ids[@]}"

Refresh the map from a full run::

    pytest tests -q --durations=0 > run.log
    python tools/shard_tests.py --record run.log

The tool deliberately does not depend on ``pytest-split`` or on the network:
sharding a test suite should not be able to fail on its own.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import re
import subprocess
import sys

DEFAULT_DURATIONS = ".test_durations.json"

# `pytest --durations=0` prints one line per phase:
#     32.10s call     tests/test_copas.py::test_fleiss_surface_matches_metasens
_DURATION_LINE = re.compile(
    r"^\s*(?P<seconds>\d+(?:\.\d+)?)s\s+(?:call|setup|teardown)\s+(?P<nodeid>\S.*?)\s*$")


def parse_durations(text):
    """Sum the phases of a ``pytest --durations=0`` report into one cost each.

    A test's cost is its call time plus its setup and teardown, which is what
    the shard has to wait for.
    """
    totals: dict[str, float] = {}
    for line in text.splitlines():
        match = _DURATION_LINE.match(line)
        if match is None:
            continue
        nodeid = match.group("nodeid")
        totals[nodeid] = totals.get(nodeid, 0.0) + float(match.group("seconds"))
    return totals


def assign(weights, n_shards):
    """Longest-processing-time assignment.

    Returns ``n_shards`` lists of test ids.  Ties are broken by id so the
    result does not depend on dictionary order.
    """
    if n_shards < 1:
        raise ValueError("n_shards must be at least 1")
    shards: list[list[str]] = [[] for _ in range(n_shards)]
    load = [0.0] * n_shards
    for nodeid in sorted(weights, key=lambda k: (-weights[k], k)):
        target = min(range(n_shards), key=lambda i: (load[i], i))
        shards[target].append(nodeid)
        load[target] += weights[nodeid]
    for shard in shards:
        shard.sort()
    return shards


def collect(paths):
    """Ask pytest for the node ids under ``paths``."""
    result = subprocess.run(
        [sys.executable, "-m", "pytest", "--collect-only", "-q", *paths],
        capture_output=True, text=True, check=False)
    if result.returncode != 0:
        raise SystemExit(
            "could not collect tests:\n" + (result.stdout + result.stderr).strip())
    out = []
    for line in result.stdout.splitlines():
        line = line.strip()
        # the summary line ends in "... collected in 1.23s"
        if "::" in line and not line.startswith(("=", "no tests")):
            out.append(line)
    return out


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--shard", type=int,
                        help="1-based shard to print")
    parser.add_argument("--of", type=int, default=4,
                        help="number of shards (default 4)")
    parser.add_argument("--tests", nargs="+", default=["tests"],
                        help="paths to collect (default: tests)")
    parser.add_argument("--durations", default=DEFAULT_DURATIONS,
                        help=f"duration map (default {DEFAULT_DURATIONS})")
    parser.add_argument("--record", metavar="LOG",
                        help="parse a pytest log and refresh the duration map")
    parser.add_argument("--out", default=DEFAULT_DURATIONS,
                        help="where --record writes the map")
    args = parser.parse_args(argv)

    if args.record:
        text = pathlib.Path(args.record).read_text(encoding="utf-8",
                                                   errors="replace")
        totals = parse_durations(text)
        pathlib.Path(args.out).write_text(
            json.dumps(totals, indent=0, sort_keys=True) + "\n",
            encoding="utf-8")
        print(f"recorded {len(totals)} test durations in {args.out}")
        return 0

    if args.shard is None:
        parser.error("either --record or --shard is required")
    if not 1 <= args.shard <= args.of:
        parser.error(f"--shard must be in 1..{args.of}")

    node_ids = collect(args.tests)
    if not node_ids:
        return 0
    path = pathlib.Path(args.durations)
    known = {}
    if path.is_file():
        known = json.loads(path.read_text(encoding="utf-8"))
    weights = {node: float(known.get(node, 1.0)) for node in node_ids}
    for node in assign(weights, args.of)[args.shard - 1]:
        print(node)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
