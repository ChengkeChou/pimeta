#!/usr/bin/env python
"""Split the test suite into equally *timed* shards.

The suite takes about eleven minutes, and CI runs it once per interpreter, so
the feedback loop is three times longer than it needs to be.  Splitting it by
file is easy and wrong: the slowest file is orders of magnitude slower than the
fastest, so an equal number of files is an unequal amount of work.  This tool
therefore shards by recorded time.

It reads ``.test_durations.json`` -- a map of test id to seconds -- and assigns
tests to shards with the longest-processing-time heuristic: sort by cost,
largest first, and give each test to the shard that currently has the least
work.  That is within a few percent of optimal for this kind of list and takes
no time to compute.  Without the map every test costs the same epsilon, so the
fallback is an equal split by count, which is at least deterministic.

Only tests above pytest's reporting threshold appear in the map -- a little
over a quarter of them here -- because the rest cost nothing measurable.  Those
reported are exactly the ones that decide how long a shard takes, so they carry
the ordering; the unmeasured ones are spread by count on top of it.

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

    Weights of zero mean "no measurable cost" -- pytest does not report a
    duration for most tests.  Those must **not** go through the same rule: a
    zero-cost assignment never raises the shard's load, so the same shard stays
    the least loaded and takes every one of them (1315 of 1656 tests, the first
    time this was tried).  They are round-robined over the shard with the
    fewest tests instead, which keeps the test counts even without disturbing
    the time balance.
    """
    if n_shards < 1:
        raise ValueError("n_shards must be at least 1")
    shards: list[list[str]] = [[] for _ in range(n_shards)]
    load = [0.0] * n_shards
    timed = {k: v for k, v in weights.items() if v > 0.0}
    untimed = sorted(k for k, v in weights.items() if v <= 0.0)
    for nodeid in sorted(timed, key=lambda k: (-timed[k], k)):
        target = min(range(n_shards), key=lambda i: (load[i], i))
        shards[target].append(nodeid)
        load[target] += timed[nodeid]
    for nodeid in untimed:
        target = min(range(n_shards),
                     key=lambda i: (len(shards[i]), load[i], i))
        shards[target].append(nodeid)
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


def read_log(path):
    """Read a pytest log, which is UTF-16 when PowerShell wrote it.

    ``pytest ... > run.log`` in PowerShell 5.1 writes UTF-16LE, while
    ``pytest ... | tee run.log`` in a POSIX shell writes UTF-8; the tool is run
    in both, so it sniffs the byte-order mark instead of guessing.
    """
    raw = pathlib.Path(path).read_bytes()
    if raw[:2] in (b"\xff\xfe", b"\xfe\xff"):
        return raw.decode("utf-16", errors="replace")
    return raw.decode("utf-8-sig", errors="replace")


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
        totals = parse_durations(read_log(args.record))
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
    # Zero rather than a second for the unmeasured tests: pytest omits anything
    # below its reporting threshold, and calling those a whole second each would
    # swamp the real costs.  `assign` spreads them by count on top of the timed
    # ones.
    weights = {node: float(known.get(node, 0.0)) for node in node_ids}
    for node in assign(weights, args.of)[args.shard - 1]:
        print(node)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
