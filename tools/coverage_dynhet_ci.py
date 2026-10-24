"""Coverage of the intervals this module reports, measured rather than assumed.

Run from the repository root::

    python tools/coverage_dynhet_ci.py            # the full study, a few minutes
    python tools/coverage_dynhet_ci.py --quick    # smaller, for a smoke check

Three questions, all of which the module makes a claim about:

1. **Is the location interval calibrated?**  ``tau_trend`` offers the metafor
   formula (GLS variance, z reference) and the Knapp–Hartung correction
   (residual-scaled variance, t reference on ``k - p``).  The second is the
   reason the first is not the default in the literature, and this measures it.
2. **Is the scale interval calibrated?**  The Wald interval for
   :math:`\\alpha_1` is referred to a z or to a t on ``k - q``.
3. **Does the trend test keep its level?**  Two nulls, and they are different:
   heterogeneity present but constant (where :math:`\\alpha_1 = 0` is an
   interior point) and no heterogeneity at all (where :math:`\\alpha_0` sits on
   its boundary).  Wald, the chi-square LRT and the parametric bootstrap are
   all measured under both, because that difference is exactly what the module
   documentation claims.

The data-generating process has a *known* :math:`\\mu(t)` and
:math:`\\tau^2(t)`, so a nominal 95 % interval either contains the truth or it
does not, with no reference implementation in the loop.
"""

from __future__ import annotations

import argparse
import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))

from pimeta.dynamichet import tau_trend  # noqa: E402

LEVEL = 0.95


def _simulate(rng, k, mean_slope, log_tau2_intercept, log_tau2_slope):
    t = np.sort(rng.uniform(0.0, 1.0, k))
    tc = t - t.mean()
    vi = rng.uniform(0.01, 0.05, k)
    mu = 0.20 + mean_slope * tc
    tau2 = np.exp(log_tau2_intercept + log_tau2_slope * tc)
    y = mu + rng.normal(0.0, 1.0, k) * np.sqrt(tau2) + \
        rng.normal(0.0, 1.0, k) * np.sqrt(vi)
    return y, vi, t


def _inside(interval, truth):
    return bool(interval[0] <= truth <= interval[1])


def coverage(reps: int, seed: int = 20261025) -> dict:
    rng = np.random.default_rng(seed)
    k = 40
    mean_slope = 0.8
    a0, a1 = -1.5, 0.9            # log tau^2 per unit of the centred time
    hits = {"beta_z": 0, "beta_knha": 0, "alpha_z": 0, "alpha_knha": 0}
    for _ in range(reps):
        y, vi, t = _simulate(rng, k, mean_slope, a0, a1)
        z = tau_trend(y, vi, t, test="z")
        knha = tau_trend(y, vi, t, test="knha")
        hits["beta_z"] += _inside(
            (z.summary()[1]["q_low"], z.summary()[1]["q_high"]), mean_slope)
        hits["beta_knha"] += _inside(
            (knha.summary()[1]["q_low"], knha.summary()[1]["q_high"]),
            mean_slope)
        hits["alpha_z"] += _inside(
            (z.summary()[-1]["q_low"], z.summary()[-1]["q_high"]), a1)
        hits["alpha_knha"] += _inside(
            (knha.summary()[-1]["q_low"], knha.summary()[-1]["q_high"]), a1)
    return {key: value / reps for key, value in hits.items()}


def rejection_rates(reps: int, seed: int = 424242, n_boot: int = 99) -> dict:
    rng = np.random.default_rng(seed)
    k = 30
    out = {}
    for label, a0 in (("heterogeneity present", -2.3), ("no heterogeneity", None)):
        counts = {"wald_z": 0, "wald_f": 0, "lrt": 0, "bootstrap": 0}
        for _ in range(reps):
            if a0 is None:
                t = np.sort(rng.uniform(0.0, 1.0, k))
                tc = t - t.mean()
                vi = rng.uniform(0.02, 0.06, k)
                y = 0.2 + 0.3 * tc + rng.normal(0.0, 1.0, k) * np.sqrt(vi)
            else:
                y, vi, t = _simulate(rng, k, 0.3, a0, 0.0)
            z = tau_trend(y, vi, t, test="z")
            knha = tau_trend(y, vi, t, test="knha")
            for key, fitted in (("wald_z", z), ("wald_f", knha)):
                try:
                    if fitted.test_trend("wald")["p_value"] < 0.05:
                        counts[key] += 1
                except ValueError:
                    # the scale coefficients were not identified: no evidence
                    # of a trend, and no Wald statistic either
                    pass
            if z.test_trend("lrt")["p_value"] < 0.05:
                counts["lrt"] += 1
            if (z.test_trend("bootstrap", n_boot=n_boot, seed=1)["p_value"]
                    < 0.05):
                counts["bootstrap"] += 1
        out[label] = {key: value / reps for key, value in counts.items()}
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true")
    args = parser.parse_args()
    reps = 120 if args.quick else 400
    test_reps = 30 if args.quick else 80

    print(f"interval coverage, {reps} replications, k = 40, nominal {LEVEL}")
    for key, value in coverage(reps).items():
        print(f"  {key:<12} {value:.3f}")

    print(f"\nrejection rate of the trend test at 5 %, {test_reps} replications, "
          "k = 30")
    for label, values in rejection_rates(test_reps).items():
        print(f"  {label}:")
        for key, value in values.items():
            print(f"    {key:<10} {value:.3f}")


if __name__ == "__main__":
    main()
