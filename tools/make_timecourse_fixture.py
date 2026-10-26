"""Regenerate ``tests/data/timecourse.csv`` from a documented simulation.

The fixture is 16 studies reporting a treatment effect at 3, 6 and 12 months.
The model the package fits is the model that generated the data, so a fitted
slope that misses the true one is a bug rather than a modelling disagreement:

* each study has its own intercept (at month 6) and its own linear slope, drawn
  from a bivariate normal with a small negative correlation — the correlation is
  what `struct="GEN"` estimates and what `tests/test_timecourse.py` checks
  against `metafor::rma.mv`;
* the three measurements inside a study are correlated through an AR(1) process
  with ``phi = 0.6`` and a standard error of 0.09, which is the block structure
  the fit is given;
* times are 3, 6 and 12 months and the model's covariate is centred at 6, so the
  intercept is the month-6 effect.

The reference numbers in the test come from
``tools/r_reference/timecourse.R``, which fits the same likelihood with
``metafor::rma.mv``.

Usage::

    python tools/make_timecourse_fixture.py            # rewrite
    python tools/make_timecourse_fixture.py --check    # exit 1 if stale
"""

from __future__ import annotations

import argparse
import hashlib
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "tests" / "data" / "timecourse.csv"

SEED = 20261026
N_STUDIES = 16
TIMES = (3.0, 6.0, 12.0)
SE = 0.09
PHI = 0.6
MU_INTERCEPT = 0.4445          # effect at month 6
SD_INTERCEPT = 0.076
MU_SLOPE = -0.0397             # per month
SD_SLOPE = 0.0168
RHO_RANDOM = -0.114


def ar1_cov(times: np.ndarray, se: float, phi: float) -> np.ndarray:
    """Within-study covariance: ``se^2 * phi^|t_i - t_j|``."""
    d = np.abs(times[:, None] - times[None, :])
    return se ** 2 * phi ** d


def chol(cov: np.ndarray) -> np.ndarray:
    return np.linalg.cholesky(cov)


def build() -> str:
    rng = np.random.default_rng(SEED)
    times = np.array(TIMES)
    centred = times - 6.0
    within = chol(ar1_cov(times, SE, PHI))
    between = np.array([
        [SD_INTERCEPT ** 2, RHO_RANDOM * SD_INTERCEPT * SD_SLOPE],
        [RHO_RANDOM * SD_INTERCEPT * SD_SLOPE, SD_SLOPE ** 2],
    ])
    between_chol = chol(between)

    lines = ["study,time,yi,se"]
    for i in range(1, N_STUDIES + 1):
        intercept, slope = np.array([MU_INTERCEPT, MU_SLOPE]) + between_chol @ rng.normal(size=2)
        noise = within @ rng.normal(size=times.size)
        yi = intercept + slope * centred + noise
        for t, y in zip(times, yi):
            lines.append(f"{i},{t:g},{y:.10f},{SE:.10f}")
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--check", action="store_true",
                        help="do not write; exit 1 if the fixture is stale")
    args = parser.parse_args(argv)

    text = build()
    if args.check:
        current = OUT.read_text(encoding="utf-8") if OUT.exists() else ""
        if current != text:
            print("tests/data/timecourse.csv does not match the generator; run "
                  "`python tools/make_timecourse_fixture.py` and recompute the "
                  "reference numbers with `Rscript tools/r_reference/timecourse.R`")
            return 1
        return 0
    OUT.write_text(text, encoding="utf-8")
    print(f"wrote {OUT.relative_to(ROOT)}: {N_STUDIES} studies x "
          f"{len(TIMES)} times, {len(text.encode())} bytes, "
          f"sha256 {hashlib.sha256(text.encode()).hexdigest()[:16]}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
