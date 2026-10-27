"""Regenerate ``tests/data/tvhr_piecewise.csv`` from a documented simulation.

Sixteen studies of 400 participants, piecewise-exponential survival with a
hazard ratio that **falls** over follow-up.  The points of the design:

* the true log hazard ratio per interval is ``(0.60, 0.30, 0.00)`` over
  ``[0, 6, 12, 24)`` months, so the hazard ratio goes from 1.82 to 1.00 and a
  trend test has something real to find;
* each study draws its *own* three interval-specific deviations from a
  multivariate normal with SD 0.12 and correlation 0.5, so the between-study
  heterogeneity is in the data rather than assumed, and the interval-to-interval
  correlation is what ``struct="UN"`` tries to estimate;
* each study has its own baseline hazard (``base = 0.08`` with a lognormal
  study effect of SD 0.3) and censoring is exponential with mean 40 months, so
  the intervals do not all carry the same amount of information;
* the seed is chosen so that the unstructured between-study covariance lands on
  the correlation boundary.  That is deliberate: the boundary is the case the
  warning and ``structure_sensitivity()`` exist for, and a fixture without it
  would never exercise that path.  It is a property of the fixture, not a claim
  about the method — the fitted statistic is whatever the ridge makes it, which
  is exactly what the module says out loud.

The reference numbers in ``tests/test_tvhr_joint.py`` come from
``tools/r_reference/tvhr_piecewise.R``, which rebuilds the split-follow-up
design, fits each study with ``glm.fit`` and pools with ``metafor::rma.mv``.

History: the seed below used to expose a defect.  ``rma_mv(struct="UN")``
once bounded each correlation to (-1, 1) *per pair*, which does not keep the
implied matrix positive definite, and on these bytes the fit reported
``rho = (1.000, 1.000, -0.203)`` — not a correlation matrix — where
``metafor::rma.mv`` reports ``(0.864, 0.637, 0.163)``.  Since the Cholesky
parametrisation landed (the fit now agrees with metafor to six decimals on
these exact bytes), this file is a generator again: the seed is kept as the
regression case for that defect.

Usage::

    python tools/make_tvhr_piecewise_fixture.py            # rewrite
    python tools/make_tvhr_piecewise_fixture.py --check    # exit 1 if stale
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "tests" / "data" / "tvhr_piecewise.csv"

SEED = 20261200
N_STUDIES = 16
PER_STUDY = 400
BREAKS = np.array([0.0, 6.0, 12.0, 24.0])
EFFECT = np.array([0.60, 0.30, 0.00])
SD_STUDY = 0.12
RHO_STUDY = 0.5
BASE = 0.08
SD_BASE = 0.3
MEAN_CENSOR = 40.0
HORIZON = 24.0


def build() -> str:
    rng = np.random.default_rng(SEED)
    k_int = EFFECT.size
    cov = np.full((k_int, k_int), RHO_STUDY * SD_STUDY ** 2)
    np.fill_diagonal(cov, SD_STUDY ** 2)

    out_t: list[float] = []
    out_e: list[int] = []
    out_s: list[int] = []
    out_tr: list[int] = []
    for i in range(1, N_STUDIES + 1):
        u = rng.multivariate_normal(np.zeros(k_int), cov)
        rate0 = BASE * np.exp(rng.normal(0.0, SD_BASE))
        treat = (np.arange(PER_STUDY) % 2).astype(float)
        cens = rng.exponential(MEAN_CENSOR, PER_STUDY)
        for b in range(PER_STUDY):
            t = 0.0
            event = 0.0
            for j in range(k_int):
                rate = rate0 * np.exp((EFFECT[j] + u[j]) * treat[b])
                wait = rng.exponential(1.0 / rate)
                if t + wait <= min(cens[b], BREAKS[j + 1]):
                    t += wait
                    event = 1.0
                    break
                if cens[b] <= BREAKS[j + 1]:
                    t = cens[b]
                    break
                t = BREAKS[j + 1]
            t = min(t, HORIZON)
            out_t.append(t)
            out_e.append(int(event) if t <= HORIZON else 0)
            out_s.append(i)
            out_tr.append(int(treat[b]))

    lines = ["study,treat,time,event"]
    for s, tr, t, e in zip(out_s, out_tr, out_t, out_e, strict=True):
        lines.append(f"{s},{tr},{t:.6f},{e}")
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
