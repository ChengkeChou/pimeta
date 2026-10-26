"""What the prediction interval actually covers, and which reference helps.

The default interval is ``est +/- z_{1-alpha/2} * sqrt(se^2 + tau2_hat)``. It
treats ``tau2_hat`` as if it were the true between-study variance, so it does
not charge for having estimated it, and at small ``k`` it covers less than it
claims. This script measures three references against the same replications so
the sizes can be compared instead of argued about:

``plug-in``
    The default: a normal quantile on ``se^2 + tau2_hat``.
``t``
    ``predtype="t"``: the same variance with a t quantile on ``k - p`` df.
``bootstrap``
    The whole experiment is simulated from the fitted parameters, refitted, and
    a new study's true effect drawn from each replication; the interval is the
    percentile range of the difference. This is the only one of the three that
    reflects the sampling distribution of ``tau2_hat`` directly.

The target is the true effect of a **new** study, drawn from ``N(mu, tau^2)``
-- not ``mu``. Checking against ``mu`` asks a different question and makes every
one of these intervals look conservative.

Usage::

    python tools/coverage_pi.py                 # 500 replications, three scenarios
    python tools/coverage_pi.py --reps 2000     # tighter Monte Carlo error

The numbers this prints for the fields ``plug-in`` and ``t`` at 2000
replications are the ones quoted in ``docs/guide/pooling.md``; they are also
recomputed for every ``validate.py`` run, where the ``pi`` and ``pi_t`` blocks
of the baseline carry their own Monte Carlo standard errors.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from pimeta.rma_uni import rma  # noqa: E402
from pimeta.validation import SCENARIOS, simulate_study  # noqa: E402


def bootstrap_pi(yi: np.ndarray, vi: np.ndarray, *, method: str = "REML",
                 level: float = 0.95, nboot: int = 199,
                 rng: np.random.Generator) -> tuple[float, float]:
    """Parametric-bootstrap interval for a new study's true effect."""
    fit = rma(yi, vi=vi, method=method)
    mu_hat = float(fit.b[0])
    tau2_hat = max(float(fit.tau2), 0.0)
    k = yi.size
    draws = np.full(nboot, np.nan)
    for b in range(nboot):
        mu_i = rng.normal(mu_hat, np.sqrt(tau2_hat), size=k)
        y_star = rng.normal(mu_i, np.sqrt(vi))
        try:
            star = rma(y_star, vi=vi, method=method)
        except Exception:  # noqa: BLE001 - a replication that fails is data
            continue
        mu_star = float(star.b[0])
        tau2_star = max(float(star.tau2), 0.0)
        draws[b] = rng.normal(mu_star, np.sqrt(tau2_star)) - mu_star
    alpha = 1.0 - level
    lo, hi = np.nanpercentile(draws, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return mu_hat + float(lo), mu_hat + float(hi)


def _mcse(p: float, n: int) -> float:
    return float(np.sqrt(max(p * (1.0 - p), 0.0) / n))


def run(name: str, *, reps: int, seed: int, nboot: int) -> dict:
    scen = SCENARIOS[name]
    rng = np.random.default_rng(seed)
    hits = {"plug-in": 0, "t": 0, "bootstrap": 0}
    for _ in range(reps):
        data = simulate_study(scen, rng)
        fit = rma(data.yi, vi=data.vi)
        target = data.mu_new
        if fit.ci_lb[0] <= target <= fit.ci_ub[0]:  # keeps the rng in step
            pass
        p0 = fit.predict(0.95)
        if p0.pi_lb <= target <= p0.pi_ub:
            hits["plug-in"] += 1
        p1 = fit.predict(0.95, predtype="t")
        if p1.pi_lb <= target <= p1.pi_ub:
            hits["t"] += 1
        lo, hi = bootstrap_pi(data.yi, data.vi, nboot=nboot, rng=rng)
        if lo <= target <= hi:
            hits["bootstrap"] += 1
    return {key: {"estimate": value / reps, "mcse": _mcse(value / reps, reps)}
            for key, value in hits.items()}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--reps", type=int, default=500)
    ap.add_argument("--seed", type=int, default=20240101)
    ap.add_argument("--nboot", type=int, default=199)
    ap.add_argument("--scenario", action="append", default=None)
    args = ap.parse_args(argv)

    names = args.scenario or ["null", "tiny", "unequal"]
    unknown = [n for n in names if n not in SCENARIOS]
    if unknown:
        print(f"unknown scenario(s): {', '.join(unknown)}", file=sys.stderr)
        return 2

    print(f"{'scenario':10s} {'k':>3s}  {'plug-in':>16s} {'t':>16s} "
          f"{'bootstrap':>16s}   (reps {args.reps}, nboot {args.nboot}, "
          f"nominal 0.95)")
    for name in names:
        out = run(name, reps=args.reps, seed=args.seed, nboot=args.nboot)
        cells = "  ".join(
            f"{out[key]['estimate']:.3f} +/- {out[key]['mcse']:.3f}"
            for key in ("plug-in", "t", "bootstrap"))
        print(f"{name:10s} {SCENARIOS[name].k:3d}  {cells}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
