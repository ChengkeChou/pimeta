"""Monte-Carlo coverage of the joint confidence region, three references.

The differential tests show that ``predict_region`` computes what metafor
computes.  They cannot show that the region *covers*: with ten studies the
covariance of the pooled vector is itself estimated from ten clusters, and every
reference distribution is a claim about that.

Three references are compared on the same replications:

* ``chi2`` -- the Wald reference, which ignores the estimation of the
  covariance entirely;
* ``t`` -- the current default, ``J * F(J, k - p)``; charges the variance
  components as if they were free parameters;
* ``kr`` -- Kenward-Roger degrees of freedom from the observed information for
  the variance parameters (`pimeta.mv.kr`).

Design: ``k`` studies, two correlated outcomes, true pooled effect ``(0.4, 0.7)``
and a true between-study covariance whose correlation is 0.6.  The statistic is
``(beta_hat - mu)' V^-1 (beta_hat - mu)`` with ``V`` the model-based covariance,
and coverage is the fraction of replications where it falls below the critical
value.  A replication whose fit lands on the correlation boundary is counted
and reported separately, because KR is not defined there.

Usage::

    python tools/coverage_region_kr.py --reps 300 --k 10
"""

from __future__ import annotations

import argparse

import numpy as np
from scipy import stats

import pimeta as ma

MU = np.array([0.4, 0.7])
TAU2 = np.array([0.05, 0.05])
RHO = 0.6
COV = np.array([[TAU2[0], RHO * np.sqrt(TAU2[0] * TAU2[1])],
                [RHO * np.sqrt(TAU2[0] * TAU2[1]), TAU2[1]]])


def one(k, seed):
    rng = np.random.default_rng(seed)
    study_effect = rng.multivariate_normal(MU, COV, k)
    vi = rng.uniform(0.02, 0.05, k)
    y = np.column_stack([
        rng.normal(study_effect[:, 0], np.sqrt(vi)),
        rng.normal(study_effect[:, 1], np.sqrt(vi)),
    ])
    V = np.zeros((2 * k, 2 * k))
    for i in range(k):
        V[2 * i:2 * i + 2, 2 * i:2 * i + 2] = np.diag([vi[i], vi[i]])
    X = np.zeros((2 * k, 2))
    X[0::2, 0] = 1.0
    X[1::2, 1] = 1.0
    data = {"metric": ["a", "b"] * k,
            "study": [str(i) for i in range(k) for _ in range(2)]}
    fit = ma.rma_mv(y.T.ravel(), V=V, mods=X, data=data,
                    random="~ metric | study", struct="UN", intercept=False)
    beta = np.asarray(fit.beta, dtype=float).ravel()
    diff = beta - MU
    stat = float(diff @ np.linalg.solve(np.asarray(fit.vb), diff))
    return fit, stat


def run(k, reps, seed0=0):
    crit = {"chi2": stats.chi2.ppf(0.95, 2),
            "t": 2.0 * stats.f.ppf(0.95, 2, k - 2)}
    hits = {name: 0 for name in ("chi2", "t", "kr")}
    kr_fail = 0
    m_values = []
    for rep in range(reps):
        fit, stat = one(k, seed0 + rep)
        hits["chi2"] += stat <= crit["chi2"]
        hits["t"] += stat <= crit["t"]
        try:
            out = ma.mv.kr.kenward_roger_region(fit, level=95)
        except ValueError:
            kr_fail += 1
            continue
        m_values.append(out["m"])
        hits["kr"] += stat <= out["crit"]

    n = reps
    print(f"k = {k}, replications = {n}, KR refusals = {kr_fail}")
    if m_values:
        finite = [m for m in m_values if np.isfinite(m)]
        print(f"  KR m: median {np.median(finite):.2f}, min {min(finite):.2f}, "
              f"max {max(finite):.2f}, infinite {len(m_values) - len(finite)}")
    print(f"  {'reference':<10}{'coverage':>10}{'MC se':>9}")
    for name in ("chi2", "t", "kr"):
        p = hits[name] / n
        print(f"  {name:<10}{p:>10.4f}{np.sqrt(p * (1 - p) / n):>9.4f}")
    return hits


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--reps", type=int, default=300)
    ap.add_argument("--k", type=int, nargs="+", default=[10, 20, 40])
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    for k in args.k:
        run(k, args.reps, args.seed)
        print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
