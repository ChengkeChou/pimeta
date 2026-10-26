"""Regenerate ``tests/data/prediction_joint.csv`` from a documented simulation.

Why this script exists
----------------------
A synthetic fixture with no generator is a file whose origin nobody can state,
which is what ``tests/data/MANIFEST.json`` recorded as ``unverified``. The data
here stand in for 18 diagnostic-accuracy studies reporting a C-statistic and a
calibration slope; the file is used to check that pimeta's bivariate pooling
reproduces ``metafor::rma.mv`` (the numbers in the test come from
``tools/r_reference/prediction_joint.R``) and that a zero within-study
correlation collapses the joint fit onto two marginal pools.

The data-generating process
---------------------------
Per study ``i``:

* ``logit(A_i^true)`` and ``slope_i^true`` are drawn from a bivariate normal with
  means ``logit(0.75)`` and ``0.85``, between-study SDs ``0.25`` and ``0.30``,
  and correlation ``-0.30``. The negative correlation is deliberate: it is what
  makes "zeroing the within-study correlation" different from "the two metrics
  are unrelated".
* Group sizes are drawn uniformly from the ranges the fixture covers
  (``n_event`` 45-98, ``n_nonevent`` 145-360), so the study sizes vary the way
  real accuracy studies do.
* ``A_i`` is the true value plus sampling error whose SD is the Hanley-McNeil
  standard error of the C-statistic, mapped to the logit scale by the delta
  method. The *analysis* recomputes that variance from the observed ``A_i``,
  which is the usual plug-in and not a self-fulfilling construction.
* ``slope_i = slope_i^true + N(0, 0.13)``, with ``slope_se = 0.13`` known. A
  constant SE keeps the fixture simple; it is not a claim about calibration
  studies.

Every number in the file is a function of ``SEED``; re-running the script
reproduces the bytes, which ``--check`` asserts.

Usage::

    python tools/make_prediction_joint_fixture.py            # rewrite
    python tools/make_prediction_joint_fixture.py --check    # exit 1 if stale
"""

from __future__ import annotations

import argparse
import hashlib
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "tests" / "data" / "prediction_joint.csv"

SEED = 20261026
N_STUDIES = 18
MU_AUC = 0.75                      # on the AUC scale; logit-transformed below
MU_SLOPE = 0.85
SD_LOGIT_AUC = 0.25
SD_SLOPE = 0.30
RHO_BETWEEN = -0.30
SLOPE_SE = 0.13
N_EVENT_RANGE = (45, 98)
N_NONEVENT_RANGE = (145, 360)


def hanley_mcneil_var(auc: np.ndarray, n1: np.ndarray,
                      n0: np.ndarray) -> np.ndarray:
    """Variance of the C-statistic on the AUC scale (Hanley & McNeil 1982)."""
    q1 = auc / (2.0 - auc)
    q2 = 2.0 * auc ** 2 / (1.0 + auc)
    return (auc * (1.0 - auc) + (n1 - 1.0) * (q1 - auc ** 2)
            + (n0 - 1.0) * (q2 - auc ** 2)) / (n1 * n0)


def build() -> str:
    rng = np.random.default_rng(SEED)
    mu_logit = float(np.log(MU_AUC / (1.0 - MU_AUC)))
    cov = np.array([[SD_LOGIT_AUC ** 2, RHO_BETWEEN * SD_LOGIT_AUC * SD_SLOPE],
                    [RHO_BETWEEN * SD_LOGIT_AUC * SD_SLOPE, SD_SLOPE ** 2]])
    truth = rng.multivariate_normal([mu_logit, MU_SLOPE], cov, size=N_STUDIES)

    n_event = rng.integers(N_EVENT_RANGE[0], N_EVENT_RANGE[1] + 1, N_STUDIES)
    n_nonevent = rng.integers(N_NONEVENT_RANGE[0], N_NONEVENT_RANGE[1] + 1,
                              N_STUDIES)

    auc_true = 1.0 / (1.0 + np.exp(-truth[:, 0]))
    # sampling error on the logit scale, at the true value
    var_auc = hanley_mcneil_var(auc_true, n_event.astype(float),
                                n_nonevent.astype(float))
    var_logit = var_auc / (auc_true * (1.0 - auc_true)) ** 2
    logit_obs = truth[:, 0] + rng.normal(0.0, np.sqrt(var_logit))
    auc_obs = 1.0 / (1.0 + np.exp(-logit_obs))

    slope_obs = truth[:, 1] + rng.normal(0.0, SLOPE_SE, N_STUDIES)
    slope_se = np.full(N_STUDIES, SLOPE_SE)

    lines = ["study,auc,n_event,n_nonevent,slope,slope_se"]
    for i in range(N_STUDIES):
        lines.append(
            f"{i + 1},{auc_obs[i]:.10f},{int(n_event[i])},"
            f"{int(n_nonevent[i])},{slope_obs[i]:.10f},{slope_se[i]:.2f}")
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--check", action="store_true",
                        help="do not write; exit 1 if the fixture is stale")
    args = parser.parse_args(argv)

    text = build()
    digest = hashlib.sha256(text.encode()).hexdigest()
    if args.check:
        current = OUT.read_text(encoding="utf-8") if OUT.exists() else ""
        if current != text:
            print("tests/data/prediction_joint.csv does not match the generator; "
                  "run `python tools/make_prediction_joint_fixture.py` and "
                  "recompute the reference numbers with "
                  "`Rscript tools/r_reference/prediction_joint.R`")
            return 1
        return 0
    OUT.write_text(text, encoding="utf-8")
    print(f"wrote {OUT.relative_to(ROOT)}: {N_STUDIES} studies, "
          f"{len(text.encode())} bytes, sha256 {digest[:16]}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
