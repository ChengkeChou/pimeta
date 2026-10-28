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
* the seed is chosen because it once exposed a defect in the unstructured
  fit: on these bytes the old per-pair box-bounds parameterisation reported
  ``rho = (1.000, 1.000, -0.203)`` — not a correlation matrix — where
  ``metafor::rma.mv`` reports ``(0.864, 0.637, 0.163)``.  Under the
  Cholesky parameterisation the fit agrees with metafor to six decimals and
  sits in the interior of the correlation cone, which is why the boundary
  warning is exercised by a separate synthetic case in the tests instead of
  by this fixture.

The reference numbers in ``tests/test_tvhr_joint.py`` come from
``tools/r_reference/tvhr_piecewise.R``, which rebuilds the split-follow-up
design, fits each study with ``glm.fit`` and pools with ``metafor::rma.mv``.

Usage::

    python tools/make_tvhr_piecewise_fixture.py            # rewrite
    python tools/make_tvhr_piecewise_fixture.py --check    # exit 1 if stale

The check is structural on purpose: the times are not portable.  See
``_problems`` for why.
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


def _problems(current: str, text: str) -> list[str]:
    """Differences between the table on disk and the regenerated one.

    The times cannot be compared, and not because of the last digit.  They
    come from ``Generator.multivariate_normal``, whose decomposition is a
    LAPACK call, and every draw decides how many exponentials the record
    consumes; the stream therefore diverges between BLAS builds and the same
    seed produces a different table, time by time, on a different machine.
    Byte equality would make this script fail on every machine but the one
    that wrote the file.

    What is portable is the design: the header, one row per participant, the
    study and treat columns, times inside the follow-up horizon, and every
    study contributing both events and censoring.  That is what ``--check``
    asserts.  The reference numbers in ``tests/test_tvhr_joint.py`` were
    computed from the committed file, so the platform dependence of the draws
    does not reach the fits.
    """
    want = text.splitlines()
    got = current.splitlines()
    if not got or got[0] != want[0]:
        return [f"header is {got[0] if got else 'missing'!r}, "
                f"expected {want[0]!r}"]
    if len(got) != len(want):
        return [f"{len(got) - 1} data rows on disk, {len(want) - 1} "
                "regenerated"]
    problems: list[str] = []
    events: dict[str, int] = {}
    rows: dict[str, int] = {}
    for i, (new, old) in enumerate(zip(want[1:], got[1:]), start=2):
        s, tr, _t, e = new.split(",")
        s2, tr2, t2, e2 = old.split(",")
        if (s, tr) != (s2, tr2):
            problems.append(f"row {i}: study/treat differ, "
                            f"{old!r} vs {new!r}")
            continue
        if not 0.0 < float(t2) <= HORIZON:
            problems.append(f"row {i}: time {t2} is outside the horizon")
        if e2 not in ("0", "1"):
            problems.append(f"row {i}: event is {e2!r}")
        rows[s2] = rows.get(s2, 0) + 1
        events[s2] = events.get(s2, 0) + int(e2)
        if len(problems) >= 5:
            break
    else:
        for study, n in sorted(rows.items()):
            if n != PER_STUDY:
                problems.append(f"study {study} has {n} rows, "
                                f"expected {PER_STUDY}")
            elif not 0 < events[study] < n:
                problems.append(f"study {study} has {events[study]} events "
                                f"in {n} rows")
    return problems


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--check", action="store_true",
                    help="do not write; exit 1 if the file differs")
    args = ap.parse_args(argv)
    text = build()
    if args.check:
        current = OUT.read_text(encoding="utf-8") if OUT.exists() else ""
        problems = _problems(current, text)
        if problems:
            print(f"{OUT.name} is not reproducible from this script:",
                  file=sys.stderr)
            for line in problems:
                print(f"  {line}", file=sys.stderr)
            return 1
        print(f"{OUT.name}: design reproduces (times are BLAS-dependent)")
        return 0
    OUT.write_text(text, encoding="utf-8", newline="\n")
    print(f"wrote {OUT} ({len(text.splitlines()) - 1} rows, "
          f"{N_STUDIES} studies)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
