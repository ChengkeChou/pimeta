"""Monte-Carlo coverage of the joint prediction region.

`predict_region` makes a claim a reader can check: draw a new study from the
model's own law, and it lands inside the ellipsoid :math:`1-\\alpha` of the
time.  Nothing in metafor implements a joint prediction region for `rma.mv`, so
there is no second implementation to agree with -- simulation is the only
evidence available.

Design
------
``J`` outcomes, ``k`` studies.  Study :math:`i` has true effects

    theta_i ~ N(mu, Sigma)

and reports them with sampling covariance :math:`V_i`,

    y_i = theta_i + e_i,   e_i ~ N(0, V_i).

Each replication fits the model once, then draws a **fresh** study's true
effects from the *true* :math:`(\\mu, \\Sigma)` and asks whether the region
contains them.  The fit is never told the truth, so what is measured is the
whole procedure: estimation error in :math:`\\hat\\mu`, in :math:`\\hat V`, and
in :math:`\\hat\\Sigma`.

Reported per ``k``:

- ``chi2`` and ``F`` -- coverage of the same ellipsoid under the two reference
  distributions ``test="z"`` and ``test="t"``.  The covariance is identical, so
  only the critical value changes and the comparison is exact.
- ``marginal`` -- how often a fresh study falls inside *both* per-outcome
  :math:`z` intervals.  That is the naive answer the joint region replaces, and
  it is anti-conservative exactly because it ignores the correlation.
- ``boundary`` -- how often a between-study variance is estimated at zero.
  Coverage is an average over estimable and non-estimable fits alike, so it is
  reported split as well.

Usage::

    python tools/coverage_predict_region.py --reps 300
    python tools/coverage_predict_region.py --reps 100 --k 10 20 --boot 199

``--boot`` adds a column for ``predict_region(test="bootstrap")``: the cutoff
is calibrated by re-simulating *and refitting* in every replicate, which is the
only one of the three references that carries the uncertainty of the estimated
between-study covariance.  It costs ``n_boot`` refits per fit.
"""

from __future__ import annotations

import argparse
import sys

import numpy as np
from scipy import stats

import pimeta as ma

MU = np.array([0.30, -0.20])
SIGMA = np.array([[0.06, 0.03], [0.03, 0.09]])
LEVEL = 0.95


def one_fit(rng, k, mu=MU, sigma=SIGMA):
    """Simulate one meta-analysis; return the fit and a fresh study's truth."""
    theta = rng.multivariate_normal(mu, sigma, k)
    vi = rng.uniform(0.01, 0.03, (k, 2))
    V = np.zeros((k, 2, 2))
    V[:, 0, 0] = vi[:, 0]
    V[:, 1, 1] = vi[:, 1]
    V[:, 0, 1] = V[:, 1, 0] = 0.5 * np.sqrt(vi[:, 0] * vi[:, 1])
    yi = theta + rng.normal(0.0, 1.0, (k, 2)) * np.sqrt(vi)

    big = np.zeros((2 * k, 2 * k))
    for i in range(k):
        big[2 * i:2 * i + 2, 2 * i:2 * i + 2] = V[i]
    X = np.zeros((2 * k, 2))
    X[np.arange(2 * k), np.tile([0, 1], k)] = 1.0
    data = {"coef": list(np.tile([0, 1], k)),
            "study": [f"s{i}" for i in range(k) for _ in range(2)]}
    fit = ma.rma_mv(yi.ravel(), V=big, mods=X, data=data,
                    random="~ coef | study", struct="UN", intercept=False)
    return fit, rng.multivariate_normal(mu, sigma), big, X, data


def _mean(x):
    return float(np.mean(x)) if np.size(x) else float("nan")


def _se(p, n):
    return float(np.sqrt(p * (1.0 - p) / n))


def bootstrap_critical(fit, V, mods, data, *, level, n_boot, rng, k):
    """Critical value from the shipped code path, not a copy of it.

    The simulator closes over the design, so the library only has to do the
    calibration arithmetic: draw a new study's true effect from the fitted
    model, refit the replicate, and measure the distance in the replicate's
    metric.  Keeping one implementation is what makes the numbers in the
    docstring a measurement of the function a user calls.
    """
    reg = ma.predict_region(fit, level=level)
    centre = np.asarray(reg.center, dtype=float).ravel()
    sigma = np.asarray(reg.cov, dtype=float) - np.asarray(fit.vb, dtype=float)
    sigma = 0.5 * (sigma + sigma.T)
    sd = np.sqrt(np.diag(V).reshape(k, 2))

    def simulate(gen):
        theta = gen.multivariate_normal(centre, sigma, k)
        yi = theta + gen.normal(0.0, 1.0, (k, 2)) * sd
        return ma.rma_mv(yi.ravel(), V=V, mods=mods, data=data,
                         random="~ coef | study", struct="UN", intercept=False)

    try:
        region = ma.predict_region(fit, level=level, test="bootstrap",
                                   simulate=simulate, n_boot=n_boot,
                                   seed=int(rng.integers(2 ** 32)))
    except ValueError:
        return float("nan")
    return float(region.crit)


def run(reps, k_list=(10, 20, 40, 80), seed0=0, level=LEVEL, n_boot=0):
    print(f"J = {MU.size} outcomes, nominal coverage {level:.2f}, "
          f"{reps} replications per k")
    print(f"true mu = {MU.tolist()}, true Sigma = {SIGMA.tolist()}")
    print("the F reference is the Hotelling one, m = k - p")
    print()
    print(f"{ 'k':>5}{'chi2':>8}{'MC se':>8}{'F':>8}{'MC se':>8}"
          f"{'HK':>8}{'MC se':>8}"
          f"{'KR':>8}{'MC se':>8}{'KR fail':>9}"
          f"{'marginal':>10}{'boundary':>10}{'chi2|ok':>9}{'F|ok':>8}"
          + (f"{'boot':>8}" if n_boot else ""))
    z = stats.norm.ppf(0.5 + 0.5 * level)
    rows = []
    for k in k_list:
        rng = np.random.default_rng([seed0, k])
        hit_z = np.zeros(reps, dtype=bool)
        hit_f = np.zeros(reps, dtype=bool)
        hit_hk = np.zeros(reps, dtype=bool)
        hit_kr = np.zeros(reps, dtype=bool)
        kr_bad = np.zeros(reps, dtype=bool)
        hit_boot = np.zeros(reps, dtype=bool)
        marg = np.zeros(reps, dtype=bool)
        bad = np.zeros(reps, dtype=bool)
        for r in range(reps):
            fit, fresh, V_big, X_big, data_big = one_fit(rng, k)
            reg = ma.predict_region(fit, level=level)
            d = reg.distance(fresh)
            tst = ma.predict_region(fit, level=level, test="t")
            hk = ma.predict_region(fit, level=level, test="hk")
            hit_z[r] = d <= reg.crit
            hit_f[r] = d <= tst.crit
            hit_hk[r] = d <= hk.crit
            try:
                kr = ma.mv.kenward_roger_region(fit, level=level)
                hit_kr[r] = d <= kr["crit"]
            except ValueError:
                kr_bad[r] = True
            marg[r] = bool(np.all(
                np.abs(fresh - reg.center) <= z * np.sqrt(np.diag(reg.cov))))
            bad[r] = bool(np.any(np.asarray(fit.sigma2, dtype=float) <= 1e-10))
            if n_boot:
                crit = bootstrap_critical(fit, V_big, X_big, data_big,
                                          level=level, n_boot=n_boot,
                                          rng=rng, k=k)
                hit_boot[r] = bool(np.isfinite(crit) and d <= crit)
        pz, pf, pm = hit_z.mean(), hit_f.mean(), marg.mean()
        phk = hit_hk.mean()
        okz, okf = hit_z[~bad], hit_f[~bad]
        usable = ~kr_bad
        pkr = hit_kr[usable].mean() if usable.any() else float("nan")
        print(f"{k:5d}{pz:8.3f}{_se(pz, reps):8.3f}{pf:8.3f}{_se(pf, reps):8.3f}"
              f"{phk:8.3f}{_se(phk, reps):8.3f}"
              f"{pkr:8.3f}{_se(pkr, int(usable.sum())):8.3f}"
              f"{kr_bad.mean():9.3f}"
              f"{pm:10.3f}{bad.mean():10.3f}"
              f"{_mean(okz):9.3f}{_mean(okf):8.3f}"
              + (f"{hit_boot.mean():8.3f}" if n_boot else ""))
        rows.append((k, pz, pf, pm, pkr, phk))
    print()
    print("marginal is the coverage of two per-outcome z intervals: it "
          "under-covers because it ignores the correlation")
    for j, name in ((1, "chi2"), (2, "F")):
        worst = max(abs(row[j] - level) for row in rows)
        print(f"worst |coverage({name}) - {level:.2f}| = {worst:.3f}")
    if any(np.isfinite(row[4]) for row in rows):
        worst = max(abs(row[4] - level) for row in rows if np.isfinite(row[4]))
        print(f"worst |coverage(KR) - {level:.2f}| = {worst:.3f}")
    worst_hk = max(abs(row[5] - level) for row in rows)
    print(f"worst |coverage(HK) - {level:.2f}| = {worst_hk:.3f}")
    return rows


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--reps", type=int, default=150)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--k", type=int, nargs="+", default=[10, 20, 40, 80])
    parser.add_argument("--boot", type=int, default=0,
                        help="parametric-bootstrap replicates per fit (0 = off); "
                             "carries the uncertainty of the estimated Sigma")
    args = parser.parse_args(argv)
    run(args.reps, tuple(args.k), args.seed, n_boot=args.boot)
    return 0


if __name__ == "__main__":
    sys.exit(main())
