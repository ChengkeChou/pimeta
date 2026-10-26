"""Regenerate ``tests/data/ipd_nma.csv`` from a documented simulation.

Eight studies, three arms, participant-level data, one covariate.  The design is
deliberately not complete: studies 3 and 6 randomise only A and C, so the
network needs the indirect route through B for the A-versus-C comparison.  That
is the point of the fixture — an IPD network that happens to be fully connected
through every study would not exercise the consistency equations.

The generating model is the model the package fits::

    y_ij = b_study(i) + mu_arm(j) + gamma * x_ij + u_arm(j),study(i) + e_ij

with

* ``mu_B = 0.07``, ``mu_C = 0.15`` against A as the reference, so the two
  treatment contrasts differ by more than their standard errors and the network
  estimate of B-versus-C is a genuine combination of direct and indirect
  evidence;
* ``gamma = 0.587`` on a covariate drawn as ``N(0, 0.2)`` — the covariate has to
  be present for the one-stage fit's ``x`` coefficient to be identified;
* a 3 x 3 between-study covariance ``S`` over (A, B, C) random treatment
  effects.  The values a reader will see in the fit are the *contrast*
  parameterisation ``L S L'`` with ``L = [B - A, C - A]``, which is what
  ``tests/test_ipd_nma.py`` pins, so ``--check`` prints it;
* residual SD ``0.704``, so a study of 360 has enough information for the
  within-study regression coefficient to be well determined.

Every study contributes 180 participants per arm, which makes the size pattern
(540, 540, 360, ...) a direct read-out of how many arms the study has.

The reference numbers in ``tests/test_ipd_nma.py`` come from
``tools/r_reference/ipd_nma.R``, which fits the same likelihood with
``lme4::lmer``.

Usage::

    python tools/make_ipd_nma_fixture.py            # rewrite
    python tools/make_ipd_nma_fixture.py --check    # exit 1 if stale
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "tests" / "data" / "ipd_nma.csv"

SEED = 20261104
ARMS = ("A", "B", "C")
N_STUDIES = 8
#: Studies that randomise only A and C, so the network is not a stack of
#: complete three-arm trials.
TWO_ARM = (3, 6)
PER_ARM = 180
MU = {"A": 0.0, "B": 0.07, "C": 0.15}
GAMMA = 0.587
SIGMA = 0.704
SD_STUDY = 0.30
SD_X = 0.20
#: Between-study covariance of the (A, B, C) random treatment effects.
S = np.array([[0.020, 0.010, 0.005],
              [0.010, 0.100, -0.030],
              [0.005, -0.030, 0.120]])


def contrast_covariance() -> np.ndarray:
    """``L S L'`` with ``L = [B - A, C - A]`` — what the model reports."""
    L = np.array([[-1.0, 1.0, 0.0], [-1.0, 0.0, 1.0]])
    return L @ S @ L.T


def build() -> str:
    rng = np.random.default_rng(SEED)
    chol = np.linalg.cholesky(S)
    lines = ["study,dB,dC,x,y"]
    for i in range(1, N_STUDIES + 1):
        arms = ("A", "C") if i in TWO_ARM else ARMS
        b_study = rng.normal(0.0, SD_STUDY)
        u = chol @ rng.normal(size=3)
        random_effect = dict(zip(ARMS, u, strict=True))
        for arm in arms:
            x = rng.normal(0.0, SD_X, PER_ARM)
            mean = b_study + MU[arm] + GAMMA * x + random_effect[arm]
            y = mean + rng.normal(0.0, SIGMA, PER_ARM)
            db = 1.0 if arm == "B" else 0.0
            dc = 1.0 if arm == "C" else 0.0
            for j in range(PER_ARM):
                lines.append(
                    f"{i:.10f},{db:.10f},{dc:.10f},{x[j]:.10f},{y[j]:.10f}")
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
        print("implied contrast covariance (B-A, C-A):")
        print(np.array2string(contrast_covariance(), precision=4))
        return 0
    OUT.write_text(text, encoding="utf-8", newline="\n")
    print(f"wrote {OUT} ({len(text.splitlines()) - 1} rows)")
    print("implied contrast covariance (B-A, C-A):")
    print(np.array2string(contrast_covariance(), precision=4))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
