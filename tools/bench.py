"""Measure, before optimising, and keep the measurement.

An optimisation without a baseline is an anecdote. This tool records wall time
and peak traced memory for one fit per model family, at a size each family can
run in a fraction of a second, and compares a fresh run against the committed
baseline in ``tests/perf/baseline.json``.

Usage::

    python tools/bench.py                      # compare with the baseline
    python tools/bench.py --write              # record/replace the baseline
    python tools/bench.py --case rma_reml_200  # one case
    python tools/bench.py --repeat 7           # more work per case

What it does **not** record, and why: optimiser iteration counts and the number
of likelihood evaluations are not exposed uniformly across the model families,
and inventing a per-family translation would produce a column that cannot be
compared across rows. Wall time and memory are comparable everywhere, so those
are the two recorded. Phase splits (preparation, objective, linear algebra) are
also omitted: they need per-family instrumentation, which is an optimisation
project of its own rather than a measurement.

Timing is inherently noisy, so a case is run once to warm up and then
``--repeat`` times, and the reported number is the **minimum**, which is the
statistic least corrupted by unrelated load. The comparison tolerance is
deliberately loose (25% slower counts as a change): this is a regression alarm
for algorithmic damage, not a benchmark competition.
"""

from __future__ import annotations

import argparse
import gc
import json
import platform
import statistics
import sys
import time
import tracemalloc
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import pimeta as ma  # noqa: E402

BASELINE = ROOT / "tests" / "perf" / "baseline.json"
SCHEMA_VERSION = 1

#: A case is a name, a description of what it stresses, and a zero-argument
#: callable that performs exactly one fit.
CASES: dict[str, dict[str, Any]] = {}


def _case(name: str, stresses: str, k: int = 0, p: int = 0):
    def register(fn: Callable[[], Any]) -> Callable[[], Any]:
        CASES[name] = {"stresses": stresses, "k": k, "p": p, "run": fn}
        return fn
    return register


def _yi_vi(k: int, seed: int = 20240101) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    mu = rng.normal(0.3, 0.2, k)
    sei = 0.2 * np.exp(rng.normal(0.0, 0.4, k))
    return rng.normal(mu, sei), sei ** 2


@_case("rma_reml_20", "iterative tau^2 at a small k", k=20, p=1)
def _rma_reml_20() -> Any:
    y, v = _yi_vi(20)
    return ma.rma(y, vi=v)


@_case("rma_reml_200", "iterative tau^2 at a large k", k=200, p=1)
def _rma_reml_200() -> Any:
    y, v = _yi_vi(200)
    return ma.rma(y, vi=v)


@_case("rma_dl_200", "the closed-form path, as a floor on the iterative one",
       k=200, p=1)
def _rma_dl_200() -> Any:
    y, v = _yi_vi(200)
    return ma.rma(y, vi=v, method="DL")


@_case("metareg_2mods_200", "a design matrix on the iterative path", k=200, p=3)
def _metareg_2mods_200() -> Any:
    y, v = _yi_vi(200)
    rng = np.random.default_rng(7)
    mods = rng.normal(size=(200, 2))
    return ma.rma(y, vi=v, mods=mods)


@_case("rma_mv_2outcomes_40", "a block-diagonal covariance with 80 rows",
       k=40, p=2)
def _rma_mv_2outcomes_40() -> Any:
    rng = np.random.default_rng(11)
    k, j = 40, 2
    study = np.repeat(np.arange(k), j)
    outcome = np.tile(np.arange(j), k)
    rows = k * j
    vi = 0.05 + 0.02 * rng.random(rows)
    V = ma.vcalc(vi, cluster=study, rho=0.3)
    y = rng.normal(0.3, 0.2, rows)
    return ma.rma_mv(y, V=V, data={"study": study, "outcome": outcome},
                     random="~ outcome | study", struct="UN")


@_case("robma_20", "exact quadrature over the model set", k=20, p=1)
def _robma_20() -> Any:
    y, v = _yi_vi(20)
    return ma.robma(y, sei=np.sqrt(v))


@_case("tau_trend_40", "dynamic heterogeneity, analytic gradient", k=40, p=2)
def _tau_trend_40() -> Any:
    # The between-study variance has to actually trend, or the scale
    # coefficients are unidentified and the fit measures its own singularity
    # handling instead of the normal path.
    rng = np.random.default_rng(13)
    k = 40
    t = np.linspace(0.0, 10.0, k)
    tau_t = 0.10 + 0.05 * t
    y = 0.2 + 0.02 * t + rng.normal(0.0, tau_t) + rng.normal(0.0, 0.2, k)
    v = np.full(k, 0.04)
    return ma.tau_trend(y, vi=v, time=t)


def measure(name: str, *, repeat: int) -> dict[str, Any]:
    """Time one case, then measure its memory in a separate pass.

    The two are deliberately not measured together. ``tracemalloc`` adds an
    overhead to every allocation, so timing a run with tracing enabled does not
    measure the code a user runs -- and it penalises the allocation-heavy cases
    differently from the others, which makes the comparison between rows
    meaningless as well as the absolute number wrong. The first version of this
    tool did exactly that and reported ``rma(method="DL")`` at 22.9 ms where the
    untraced time is 2.5 ms.
    """
    case = CASES[name]
    case["run"]()                                     # warm up: imports, caches

    times: list[float] = []
    for _ in range(repeat):
        gc.collect()
        start = time.perf_counter()
        case["run"]()
        times.append(time.perf_counter() - start)

    gc.collect()
    tracemalloc.start()
    case["run"]()
    peak = tracemalloc.get_traced_memory()[1]
    tracemalloc.stop()

    return {
        "stresses": case["stresses"],
        "k": case["k"],
        "p": case["p"],
        "repeat": repeat,
        "min_seconds": min(times),
        "median_seconds": statistics.median(times),
        "peak_kb": peak / 1024.0,
    }


def run_all(*, repeat: int, only: list[str] | None = None
            ) -> dict[str, Any]:
    names = only or list(CASES)
    unknown = [n for n in names if n not in CASES]
    if unknown:
        raise SystemExit(f"unknown case(s): {', '.join(unknown)}\n"
                         f"available: {', '.join(CASES)}")
    cases = {}
    for name in names:
        print(f"  {name} ...", flush=True)
        cases[name] = measure(name, repeat=repeat)
    return {
        "schema_version": SCHEMA_VERSION,
        "environment": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "platform": platform.platform(),
            "processor": platform.processor(),
            "pimeta": getattr(ma, "__version__", "unknown"),
        },
        "settings": {"repeat": repeat},
        "cases": cases,
    }


#: A case is called "changed" when it is slower than the baseline by more than
#: this factor, or uses more peak memory by more than the second.
SLOWDOWN_LIMIT = 1.25
MEMORY_LIMIT = 1.40


def compare(observed: dict, baseline: dict) -> list[dict[str, Any]]:
    """Compare two runs case by case.

    A case that appears in only one of the two is reported as missing rather
    than silently skipped: a renamed case and a deleted case look identical
    unless the difference is named.
    """
    rows: list[dict[str, Any]] = []
    names = sorted(set(observed["cases"]) | set(baseline["cases"]))
    for name in names:
        new = observed["cases"].get(name)
        old = baseline["cases"].get(name)
        if new is None or old is None:
            rows.append({"case": name, "ok": False, "reason":
                         "present in only one of the two runs"})
            continue
        ratio = new["min_seconds"] / old["min_seconds"] if old["min_seconds"] else float("inf")
        mem = (new["peak_kb"] / old["peak_kb"]) if old["peak_kb"] else float("inf")
        rows.append({
            "case": name,
            "ok": ratio <= SLOWDOWN_LIMIT and mem <= MEMORY_LIMIT,
            "time_ratio": ratio,
            "memory_ratio": mem,
            "baseline_seconds": old["min_seconds"],
            "observed_seconds": new["min_seconds"],
            "baseline_kb": old["peak_kb"],
            "observed_kb": new["peak_kb"],
        })
    return rows


def _print(rows: list[dict[str, Any]]) -> bool:
    ok = True
    print(f"\n{'case':24s} {'time':>12s} {'x':>6s} {'memory':>14s} {'x':>6s}")
    for row in rows:
        if "reason" in row:
            print(f"{row['case']:24s} {'':>12s} {'':>6s} {'':>14s}  {row['reason']}")
            ok = False
            continue
        flag = "  " if row["ok"] else "->"
        print(f"{flag}{row['case']:22s} "
              f"{row['baseline_seconds'] * 1e3:9.1f} ms "
              f"{row['time_ratio']:6.2f} "
              f"{row['baseline_kb']:9.0f} KB {row['memory_ratio']:6.2f}")
        ok &= row["ok"]
    return ok


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--case", action="append", default=None)
    ap.add_argument("--repeat", type=int, default=3,
                    help="timed runs per case; the minimum is reported")
    ap.add_argument("--write", action="store_true",
                    help="write the baseline instead of comparing with it")
    args = ap.parse_args(argv)

    print(f"benchmarking {len(args.case or CASES)} case(s) ...")
    observed = run_all(repeat=args.repeat, only=args.case)

    if args.write:
        BASELINE.parent.mkdir(parents=True, exist_ok=True)
        BASELINE.write_text(json.dumps(observed, indent=2) + "\n",
                            encoding="utf-8", newline="\n")
        print(f"wrote {BASELINE.relative_to(ROOT)}")
        return 0

    if not BASELINE.exists():
        print(f"no baseline at {BASELINE.relative_to(ROOT)}; "
              "run with --write to record one")
        return 1
    baseline = json.loads(BASELINE.read_text(encoding="utf-8"))
    if baseline.get("schema_version") != SCHEMA_VERSION:
        print("the baseline was written by a different schema version; "
              "regenerate it with --write after checking the change")
        return 1
    return 0 if _print(compare(observed, baseline)) else 1


if __name__ == "__main__":
    raise SystemExit(main())
