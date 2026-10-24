"""Monte-Carlo coverage of the two-stage IPD curve.

The differential tests establish that `ipd_two_stage_curve` computes what
metafor computes.  They cannot establish that the answer is *right*: two
packages can agree and both be wrong.  This script asks the only question that
separates them -- when the truth is known, does the 95% band contain it 95% of
the time?

Design
------
Twelve studies, 200 participants each.  The true treatment effect is quadratic
in the covariate,

    Delta(z) = a0 + a1*z + a2*z^2,        z = age - 55

and each study's own curve is that plus an independent draw from a normal
perturbation, so there is real between-study heterogeneity.  A restricted cubic
spline with four knots spans quadratics exactly, so the model is correctly
specified and any miscalibration is the interval's, not the mean function's.

Reported per covariate value: coverage of the confidence band for the average
curve, and of the prediction band for a new study's curve, with the Monte-Carlo
standard error.  Also the fraction of fits whose between-study covariance sat
on a boundary -- coverage is only meaningful for the fits that were estimable.

Usage::

    python tools/coverage_ipd_curve.py --reps 300
"""

from __future__ import annotations

import argparse
import sys

import numpy as np

import pimeta as ma

KNOTS = [45.0, 52.0, 60.0, 68.0]
A0, A1, A2 = 0.30, -0.012, 0.0009      # the true population curve
SD = (0.35, 0.006, 0.0004)             # between-study SD of each coefficient
GRID = [45.0, 50.0, 55.0, 60.0, 65.0]


def truth(x):
    z = np.asarray(x, dtype=float) - 55.0
    return A0 + A1 * z + A2 * z ** 2


def simulate(seed, n_study=12, per_study=200):
    rng = np.random.default_rng(seed)
    pid = np.arange(1, n_study * per_study + 1)
    study = (pid - 1) // per_study + 1
    age = 40.0 + 30.0 * ((pid * 37) % 101) / 101.0
    treat = (pid % 2).astype(float)
    z = age - 55.0

    a0 = A0 + rng.normal(0.0, SD[0], n_study)
    a1 = A1 + rng.normal(0.0, SD[1], n_study)
    a2 = A2 + rng.normal(0.0, SD[2], n_study)
    curve = a0[study - 1] + a1[study - 1] * z + a2[study - 1] * z ** 2
    shift = rng.normal(0.0, 0.3, n_study)[study - 1]
    eta = -0.4 + 0.10 * treat + 0.03 * z + shift + treat * curve
    y = (rng.random(eta.size) < 1.0 / (1.0 + np.exp(-eta))).astype(int)
    return study, age, treat, y


def run(reps, seed0=0, test="z"):
    cover_ci = np.zeros((reps, len(GRID)))
    cover_pred = np.zeros((reps, len(GRID)))
    fitted = np.zeros(reps, dtype=bool)
    at_boundary = 0
    tau2_max = np.zeros(reps)
    true = truth(GRID)

    for rep in range(reps):
        study, age, treat, y = simulate(seed0 + rep)
        try:
            fit = ma.ipd_two_stage_curve(y, study=study, treat=treat,
                                         covariate=age, knots=KNOTS,
                                         test=test)
        except ValueError:
            continue
        fitted[rep] = True
        curve = fit.curve(GRID)
        cover_ci[rep] = (curve.ci_lb <= true) & (true <= curve.ci_ub)
        cover_pred[rep] = (curve.pred_lb <= true) & (true <= curve.pred_ub)
        tau2_max[rep] = float(np.max(fit.tau2))
        if fit.diagnostics()["boundary"]:
            at_boundary += 1

    n = int(fitted.sum())
    print(f"test = {test!r}")
    print(f"replications: {reps}, fitted: {n}, "
          f"boundary fits: {at_boundary} ({at_boundary / max(n, 1):.1%})")
    print(f"median max tau2: {np.median(tau2_max[fitted]):.4g}")
    print()
    print(f"{'x':>6}{'coverage(CI)':>15}{'MC se':>9}"
          f"{'coverage(pred)':>17}{'MC se':>9}")
    for j, x in enumerate(GRID):
        p_ci = cover_ci[fitted, j].mean()
        p_pr = cover_pred[fitted, j].mean()
        se_ci = np.sqrt(p_ci * (1 - p_ci) / n)
        se_pr = np.sqrt(p_pr * (1 - p_pr) / n)
        print(f"{x:6.1f}{p_ci:15.3f}{se_ci:9.3f}{p_pr:17.3f}{se_pr:9.3f}")
    worst = min(
        abs(cover_ci[fitted, j].mean() - 0.95) for j in range(len(GRID)))
    print(f"\nworst |coverage(CI) - 0.95| = {worst:.3f}")
    return 0 if worst <= 0.06 else 1


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--reps", type=int, default=300)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--test", default="z",
                        help="reference distribution: z, t, knha or adhoc")
    args = parser.parse_args(argv)
    return run(args.reps, args.seed, args.test)


if __name__ == "__main__":
    sys.exit(main())
