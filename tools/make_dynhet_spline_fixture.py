"""Regenerate ``tests/data/dynhet_spline.csv`` from a documented simulation.

Sixty studies at evenly spaced publication years between 1980 and 2015, with a
between-study variance that is **low at both ends and high in the middle**:

.. math::

    \\tau^2(t) = \\exp\\{-3 + 1.6\\exp[-(u - 0.5)^2 / 0.18^2]\\},
    \\qquad u = \\frac{t - t_{\\min}}{t_{\\max} - t_{\\min}}

so :math:`\\tau^2` runs from 0.050 at the ends to 0.246 at the middle — a shape
no linear or quadratic scale model can follow, and one a four-knot restricted
cubic spline can (the bump is inside the knot range, which is where a restricted
spline is allowed to bend).

This fixture exists because the *linear* dynamic-heterogeneity fixture
(``dynhet.csv``) cannot test the spline: its heterogeneity really does trend
linearly, so the spline correctly collapses to the linear fit and no test of the
wiggliness path is possible on it.

The reference numbers in ``tests/test_dynhet.py`` come from
``tools/r_reference/tau2t_spline.R``, which fits the same model in Stan
(CmdStan through cmdstanr): the same spline basis, a weakly informative prior on
the trend and a half-normal prior on the wiggliness.  Two estimators, one model.

Usage::

    python tools/make_dynhet_spline_fixture.py            # rewrite
    python tools/make_dynhet_spline_fixture.py --check    # exit 1 if stale
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
TARGET = ROOT / "tests" / "data" / "dynhet_spline.csv"

SEED = 7
K = 60
YEAR_MIN, YEAR_MAX = 1980.0, 2015.0
MU_B0, MU_B1, MU_CENTRE = 0.3, 0.01, 2000.0


def build() -> str:
    """The fixture as CSV text."""
    rng = np.random.default_rng(SEED)
    year = np.linspace(YEAR_MIN, YEAR_MAX, K)
    u = (year - YEAR_MIN) / (YEAR_MAX - YEAR_MIN)
    tau2 = np.exp(-3.0 + 1.6 * np.exp(-((u - 0.5) / 0.18) ** 2))
    vi = rng.uniform(0.01, 0.05, K)
    yi = (MU_B0 + MU_B1 * (year - MU_CENTRE)
          + rng.normal(0.0, 1.0, K) * np.sqrt(vi + tau2))
    lines = ["yi,vi,year"]
    for a, b, c in zip(yi, vi, year, strict=True):
        lines.append(f"{a:.10f},{b:.10f},{c:.4f}")
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true",
                        help="exit 1 if the committed file differs")
    args = parser.parse_args(argv)
    text = build()
    if args.check:
        current = TARGET.read_text(encoding="utf-8") if TARGET.exists() else ""
        if current != text:
            print(f"{TARGET.name}: STALE", file=sys.stderr)
            return 1
        print(f"{TARGET.name}: reproducible")
        return 0
    TARGET.write_text(text, encoding="utf-8", newline="\n")
    print(f"wrote {TARGET} ({K} studies)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
