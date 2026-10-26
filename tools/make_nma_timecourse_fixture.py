"""Regenerate ``tests/data/nma_timecourse.csv`` from a documented simulation.

Each study compares placebo with **one** treatment on a repeated-measures
schedule, which is the design the AR(1) within-study correlation describes
exactly: every pair of a study's contrasts is measured in the same two arms at
two times.

* agent A's benefit grows, ``0.45 + 0.075 * t``; agent B's wears off,
  ``0.95 - 0.035 * t``.  The two therefore cross inside the follow-up, so a
  model that only estimates a single slope per agent cannot fit both;
* studies differ in their schedule — two to four measurements at 0, 3, 4, 6, 8,
  12 or 16 — so the design matrix is genuinely unbalanced and the fit has to
  carry a per-study covariance rather than a common one;
* between-study offsets have SD ``tau = 0.18``, within-study errors are AR(1)
  with ``phi = 0.6`` and SD ``0.22``, and ``vi`` records the marginal variance
  ``0.22 ** 2`` because that is what an arm-level contrast reports.

Usage::

    python tools/make_nma_timecourse_fixture.py            # rewrite
    python tools/make_nma_timecourse_fixture.py --check    # exit 1 if stale
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "tests" / "data" / "nma_timecourse.csv"

SEED = 20261102
TAU = 0.18
SE = 0.22
PHI = 0.6

TRUTH = {
    "A": lambda t: 0.45 + 0.075 * t,          # growing benefit
    "B": lambda t: 0.95 - 0.035 * t,          # wearing off
}

SCHEDULES = (
    (("A",), (0.0, 3.0, 6.0, 12.0)),
    (("A",), (0.0, 6.0, 12.0)),
    (("A",), (0.0, 3.0, 12.0)),
    (("A",), (0.0, 12.0)),
    (("A",), (0.0, 3.0, 6.0)),
    (("A",), (0.0, 6.0, 12.0)),
    (("A",), (0.0, 3.0, 6.0, 12.0)),
    (("A",), (0.0, 12.0)),
    (("B",), (0.0, 4.0, 8.0)),
    (("B",), (0.0, 8.0, 16.0)),
    (("B",), (0.0, 4.0, 16.0)),
    (("B",), (0.0, 16.0)),
    (("B",), (0.0, 8.0)),
    (("B",), (0.0, 4.0, 8.0, 16.0)),
)


def build() -> str:
    """The exact bytes of the fixture."""
    rng = np.random.default_rng(SEED)
    rows: list[list] = []
    for i, (arms, sched) in enumerate(SCHEDULES, start=1):
        u = rng.normal(0.0, TAU)
        for arm in arms:
            dev = rng.normal(0.0, TAU)
            errors: list[float] = []
            prev = 0.0
            for _ in sched:
                prev = PHI * prev + np.sqrt(1.0 - PHI ** 2) * rng.normal(0.0, SE)
                errors.append(prev)
            for t, err in zip(sched, errors):
                rows.append([f"study{i:02d}", "P", arm, t,
                             TRUTH[arm](t) + u + dev + err, SE ** 2])
    out = ["study,treat1,treat2,time,yi,vi"]
    for r in rows:
        out.append(f"{r[0]},{r[1]},{r[2]},{r[3]:g},{r[4]:.10f},{r[5]:.10f}")
    return "\n".join(out) + "\n"


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--check", action="store_true",
                    help="do not write; exit 1 if the file differs")
    args = ap.parse_args(argv)
    text = build()
    if args.check:
        current = OUT.read_text(encoding="utf-8") if OUT.exists() else ""
        if current != text:
            print(f"{OUT.name} is not reproducible from this script",
                  file=sys.stderr)
            return 1
        print(f"{OUT.name}: reproducible")
        return 0
    OUT.write_text(text, encoding="utf-8", newline="\n")
    print(f"wrote {OUT} ({len(text.splitlines()) - 1} contrasts, "
          f"{len(SCHEDULES)} studies)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
