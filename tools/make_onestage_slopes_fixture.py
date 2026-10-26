"""Regenerate ``tests/data/onestage_slopes.csv`` from a documented simulation.

The fixture exists to discriminate between one-stage IPD models with and
without a random treatment slope, so it has to satisfy three requirements and
they are the reason for the numbers below:

* **Studies are large.** 165–275 participants.  With the 25–60 participant
  studies a first attempt used, the sampling noise in each study's own intercept
  and slope is comparable to the variance being estimated, both implementations
  land on the correlation boundary and there is nothing left to discriminate.
* **The random effects are interior.**  ``D = [[0.09, 0.03], [0.03, 0.04]]``
  gives a true correlation of 0.5 and a slope SD of 0.2, so a model that
  estimates a boundary correlation can be told apart from one that estimates
  the truth.  The residual SD is 0.8, which makes the slope identifiable from
  200 participants but not from 40.
* **The fixed effects are the ones a reader expects**: an intercept near 0.44,
  a treatment effect near −0.11, a small age slope and a small treatment-by-age
  interaction, on the age-centred scale (``agec = age - 55``) the tests use.

The reference numbers in ``tests/test_onestage_slopes.py`` come from
``tools/r_reference/onestage_slopes.R``, which fits the same likelihood with
``lme4::lmer``.

Usage::

    python tools/make_onestage_slopes_fixture.py            # rewrite
    python tools/make_onestage_slopes_fixture.py --check    # exit 1 if stale
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "tests" / "data" / "onestage_slopes.csv"

SEED = 20261103
N_STUDIES = 24
#: Study sizes are multiples of five between 165 and 275, so the design is
#: unbalanced without any study being small.
SIZE_LOW, SIZE_HIGH, SIZE_STEP = 165, 275, 5
AGE_MEAN, AGE_SD = 55.0, 9.0
D = np.array([[0.09, 0.03], [0.03, 0.04]])
SIGMA = 0.8
BETA = (0.44, -0.11, 0.021, 0.0097)      # intercept, treat, agec, treat:agec


def build() -> str:
    rng = np.random.default_rng(SEED)
    steps = np.arange(SIZE_LOW, SIZE_HIGH + 1, SIZE_STEP)
    lines = ["study,treat,age,y"]
    for i in range(1, N_STUDIES + 1):
        n = int(rng.choice(steps))
        treat = (np.arange(n) % 2).astype(float)
        age = np.round(rng.normal(AGE_MEAN, AGE_SD, n), 3)
        agec = age - AGE_MEAN
        u = rng.multivariate_normal(np.zeros(2), D)
        mean = (BETA[0] + BETA[1] * treat + BETA[2] * agec
                + BETA[3] * treat * agec + u[0] + u[1] * treat)
        y = mean + rng.normal(0.0, SIGMA, n)
        for j in range(n):
            lines.append(f"{i},{treat[j]:g},{age[j]:.3f},{y[j]:.10f}")
    return "\n".join(lines) + "\n"


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
    print(f"wrote {OUT} ({len(text.splitlines()) - 1} rows, "
          f"{N_STUDIES} studies)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
