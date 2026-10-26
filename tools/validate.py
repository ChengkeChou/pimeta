"""Run a validation scenario and compare it with a stored baseline.

Usage::

    python tools/validate.py --scenario null --reps 2000 --seed 20240101
    python tools/validate.py --all --reps 500          # compare everywhere
    python tools/validate.py --scenario null --write   # refresh the baseline

The baseline files live in ``tests/validation/baseline``. They are not a
specification of what the estimator *should* do; they record what it *does*,
measured with a fixed seed, so that a change in behaviour has to be noticed
instead of absorbed. Whether the recorded numbers are acceptable is a separate
question, answered by the ``coverage_target`` and ``alpha`` fields the report
carries next to them.

Tolerances are derived from the Monte Carlo error rather than chosen by hand:
two runs of the same estimator differ by at most a few MCSEs, so a tolerance
that ignores the MCSE of the *reference* run would flag noise as a regression
when the reference was produced with few replications.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from pimeta.validation import SCENARIOS, run_scenario  # noqa: E402

BASELINE_DIR = ROOT / "tests" / "validation" / "baseline"


#: Which numbers are compared, and how far apart two runs may be before the
#: difference is called a change. ``mcse`` names the Monte Carlo errors that
#: widen the tolerance; ``floor`` is a minimum tolerance for cases where the
#: MCSE is zero or meaningless (a rate of exactly 0 or 1, for instance).
COMPARISONS: tuple[dict[str, Any], ...] = (
    {"path": "bias.estimate", "floor": 0.01, "mcse": ("bias.mcse",)},
    {"path": "dispersion.se_ratio", "floor": 0.02,
     "mcse": ("dispersion.se_ratio_mcse",)},
    {"path": "coverage.ci.estimate", "floor": 0.02,
     "mcse": ("coverage.ci.mcse",)},
    {"path": "coverage.pi.estimate", "floor": 0.02,
     "mcse": ("coverage.pi.mcse",)},
    {"path": "coverage.pi_t.estimate", "floor": 0.02,
     "mcse": ("coverage.pi_t.mcse",)},
    {"path": "tests.rejection_rate.estimate", "floor": 0.02,
     "mcse": ("tests.rejection_rate.mcse",)},
    {"path": "heterogeneity.tau2_bias", "floor": 0.01,
     "mcse": ("heterogeneity.tau2_bias_mcse",)},
    {"path": "reliability.fraction_finite", "floor": 0.02, "mcse": ()},
)


def _get(report: dict, path: str) -> float | None:
    node: Any = report
    for part in path.split("."):
        if not isinstance(node, dict) or part not in node:
            return None
        node = node[part]
    if isinstance(node, bool) or not isinstance(node, (int, float)):
        return None
    value = float(node)
    return None if math.isnan(value) else value


def compare(report: dict, baseline: dict, *, k_mcse: float = 3.0) -> list[dict]:
    """Return one row per comparison, with ``ok`` set."""
    rows = []
    for spec in COMPARISONS:
        path = spec["path"]
        new = _get(report, path)
        old = _get(baseline, path)
        if new is None or old is None:
            rows.append({"path": path, "baseline": old, "observed": new,
                         "tolerance": None, "delta": None, "ok": False,
                         "reason": "missing from one of the reports"})
            continue
        var = 0.0
        for key in spec["mcse"]:
            for rep in (report, baseline):
                m = _get(rep, key)
                if m is not None and math.isfinite(m):
                    var += m * m
        tol = max(float(spec["floor"]), k_mcse * math.sqrt(var))
        delta = new - old
        rows.append({"path": path, "baseline": old, "observed": new,
                     "tolerance": tol, "delta": delta,
                     "ok": abs(delta) <= tol, "reason": ""})
    return rows


def _print_rows(name: str, rows: list[dict]) -> bool:
    bad = [r for r in rows if not r["ok"]]
    status = "ok" if not bad else f"{len(bad)} changed"
    print(f"\n{name}: {status}")
    for r in rows:
        if r["delta"] is None:
            print(f"  {'??':>6}  {r['path']:<36} {r['reason']}")
            continue
        flag = "  " if r["ok"] else "->"
        print(f"  {flag}{r['delta']:+.4f}  {r['path']:<36} "
              f"baseline {r['baseline']:.4f}  observed {r['observed']:.4f}  "
              f"tol +/-{r['tolerance']:.4f}")
    return not bad


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--scenario", action="append", default=None,
                    help="scenario name; repeatable")
    ap.add_argument("--all", action="store_true", help="run every scenario")
    ap.add_argument("--reps", type=int, default=None,
                    help="replications (default: the baseline's own count)")
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--method", default=None, help="tau^2 estimator")
    ap.add_argument("--jobs", type=int, default=None,
                    help="worker processes for the fits; the report is "
                         "identical to the serial one")
    ap.add_argument("--write", action="store_true",
                    help="write the baseline instead of comparing with it")
    ap.add_argument("--out", default=None,
                    help="where to write the report when --write is used")
    args = ap.parse_args(argv)

    names = list(SCENARIOS) if args.all or not args.scenario else args.scenario
    unknown = [n for n in names if n not in SCENARIOS]
    if unknown:
        print(f"unknown scenario(s): {', '.join(unknown)}\n"
              f"available: {', '.join(SCENARIOS)}", file=sys.stderr)
        return 2

    all_ok = True
    for name in names:
        baseline_path = BASELINE_DIR / f"{name}.json"
        baseline = None
        if baseline_path.exists():
            baseline = json.loads(baseline_path.read_text(encoding="utf-8"))

        reps = args.reps
        seed = args.seed
        method = args.method
        if baseline is not None:
            reps = reps or baseline["settings"]["reps"]
            seed = seed if seed is not None else baseline["settings"]["seed"]
            method = method or baseline["settings"]["method"]
        reps = reps or 2000
        seed = seed if seed is not None else 20240101
        method = method or "REML"

        print(f"[{name}] {reps} reps, seed {seed}, method {method} ...",
              flush=True)
        report = run_scenario(SCENARIOS[name], reps=reps, seed=seed,
                              method=method, n_jobs=args.jobs)

        if args.write:
            BASELINE_DIR.mkdir(parents=True, exist_ok=True)
            target = Path(args.out) if args.out else baseline_path
            target.write_text(json.dumps(report, indent=2) + "\n",
                              encoding="utf-8")
            print(f"  wrote {target}")
            continue

        if baseline is None:
            print(f"  no baseline at {baseline_path}; "
                  f"run with --write to record one")
            all_ok = False
            continue
        all_ok &= _print_rows(name, compare(report, baseline))

    return 0 if all_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
