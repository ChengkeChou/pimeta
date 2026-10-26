"""Regenerate ``tests/data/nma_dose.csv`` from a documented simulation.

A dose-response network with two agents and a placebo.  Each agent follows its
own Emax curve, ``emax * dose / (ed50 + dose)``, so the *shape* is known and a
fitted curve that misses the true one is a bug rather than a modelling
disagreement:

* agent A reaches ``emax = 1.4`` with ``ed50 = 30``, agent B ``0.9`` with
  ``ed50 = 25`` — different asymptotes *and* different half-maximal doses, which
  is what makes a shared-shape model distinguishable from agent-specific ones;
* every study draws its own study-level offset and its own per-arm deviation,
  both normal with SD ``tau = 0.22``, so the between-study heterogeneity is in
  the data rather than assumed by the model;
* within-study standard errors are ``0.20`` (``vi = 0.04``), and the three
  head-to-head contrasts carry ``0.20 * sqrt(2)`` because they are differences
  of two arms;
* doses bracket both ED50s, and three studies compare unequal doses
  head-to-head, so the curve is identified from contrasts and not only from
  placebo comparisons.

The reference numbers in ``tests/test_nma_dose.py`` come from the closed-form
likelihood in this package; ``tests/test_nma_dose_mcmc.py`` checks the MCMC
route against it on the same bytes.

Usage::

    python tools/make_nma_dose_fixture.py            # rewrite
    python tools/make_nma_dose_fixture.py --check    # exit 1 if stale
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "tests" / "data" / "nma_dose.csv"

SEED = 20261101
EMAX_A, ED50_A = 1.4, 30.0
EMAX_B, ED50_B = 0.9, 25.0
TAU = 0.22
SE = 0.20

#: One entry per study: the arms it randomises beyond placebo.  A ``list`` entry
#: is a head-to-head contrast ``[arm1, dose1, arm2, dose2]``.
PLAN = (
    (("A", 10.0),),
    (("A", 25.0),),
    (("A", 50.0),),
    (("A", 100.0),),
    (("A", 200.0),),
    (("A", 10.0), ("A", 50.0)),
    (("A", 25.0), ("A", 100.0)),
    (("A", 50.0), ("A", 200.0)),
    (("B", 5.0),),
    (("B", 10.0),),
    (("B", 20.0),),
    (("B", 40.0), ("B", 80.0)),
    (("B", 10.0), ("B", 40.0)),
    (("A", 25.0, "B", 40.0),),
    (("A", 50.0, "B", 10.0), ("A", 200.0)),
    (("A", 100.0, "B", 80.0), ("B", 20.0)),
)


def f_a(x: float) -> float:
    return EMAX_A * x / (ED50_A + x)


def f_b(x: float) -> float:
    return EMAX_B * x / (ED50_B + x)


def build() -> str:
    """The exact bytes of the fixture."""
    rng = np.random.default_rng(SEED)
    curve = {"A": f_a, "B": f_b}
    rows: list[list] = []
    for i, arms in enumerate(PLAN, start=1):
        u = rng.normal(0.0, TAU)
        dev = {"A": rng.normal(0.0, TAU), "B": rng.normal(0.0, TAU)}
        for arm in arms:
            if len(arm) == 4:
                a_name, a_dose, b_name, b_dose = arm
                se = SE * 1.414
                rows.append([f"study{i:02d}", a_name, b_name, a_dose, b_dose,
                             curve[b_name](b_dose) - curve[a_name](a_dose)
                             + dev[b_name] - dev[a_name]
                             + rng.normal(0.0, se), se ** 2])
            else:
                name, dose = arm
                rows.append([f"study{i:02d}", "P", name, 0.0, dose,
                             curve[name](dose) + u + dev[name]
                             + rng.normal(0.0, SE), SE ** 2])
    out = ["study,treat1,treat2,dose1,dose2,yi,vi"]
    for r in rows:
        out.append(f"{r[0]},{r[1]},{r[2]},{r[3]:g},{r[4]:g},"
                   f"{r[5]:.10f},{r[6]:.10f}")
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
          f"{len(PLAN)} studies)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
