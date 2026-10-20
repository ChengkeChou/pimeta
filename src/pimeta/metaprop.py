"""Single-arm proportion meta-analysis (meta::metaprop).

Replicates ``meta::metaprop``: transforms per-study proportions to a normal
scale, pools with inverse-variance weighting (fixed or random effects), then
back-transforms the pooled result and CI to the proportion scale.

Supported ``sm`` (summary measure) and their transformations:

- ``PFT``    Freeman-Tukey double arcsine (default)
- ``PAS``    arcsine
- ``PLN``    log
- ``PLOGIT`` logit
- ``PRAW``   raw proportion

For PFT the back-transformation needs a sample size: ``meta`` uses the
harmonic mean ``1 / mean(1/n)`` of the sample sizes for both the fixed- and
random-effects estimates. The ``incr`` continuity correction is applied to
PLN / PLOGIT / PRAW study-level values (default 0.5).
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from scipy import stats

from .pooling import PooledResult, pool_yv

Z_975 = stats.norm.ppf(0.975)

_SM = {"PFT", "PAS", "PLN", "PLOGIT", "PRAW"}


@dataclass
class MetapropResult:
    """Proportion meta-analysis result (fixed + random, proportion scale)."""

    sm: str
    k: int
    n: list[float]                 # per-study sample sizes
    # proportion-scale pooled estimates + CI
    fixed_prop: float
    fixed_lower: float
    fixed_upper: float
    random_prop: float | None
    random_lower: float | None
    random_upper: float | None
    # analysis-scale per-study values
    yi: list[float]
    vi: list[float]
    # underlying PooledResult objects (analysis scale, with heterogeneity)
    fixed: PooledResult
    random: PooledResult | None
    # event counts and the tau^2 estimator, kept because R exposes x$event /
    # x$method.tau (metabias' Peters variant reads them for single-arm data)
    event: list[float] | None = None
    method_tau: str = "REML"

    def to_dict(self) -> dict:
        return {
            "sm": self.sm,
            "k": self.k,
            "n": self.n,
            "event": self.event,
            "method_tau": self.method_tau,
            "fixed_prop": self.fixed_prop,
            "fixed_lower": self.fixed_lower,
            "fixed_upper": self.fixed_upper,
            "random_prop": self.random_prop,
            "random_lower": self.random_lower,
            "random_upper": self.random_upper,
            "yi": self.yi,
            "vi": self.vi,
            "fixed": _pool_dict(self.fixed),
            "random": _pool_dict(self.random) if self.random else None,
        }


def _pool_dict(r: PooledResult | None) -> dict | None:
    if r is None:
        return None
    return {
        "pooled_effect": r.pooled_effect,
        "ci_lower": r.pooled_ci_lower,
        "ci_upper": r.pooled_ci_upper,
        "pooled_est": r.pooled_est,
        "pooled_se": r.pooled_se,
        "z": r.z,
        "p": r.p_value,
        "q": r.q,
        "q_df": r.q_df,
        "q_p": r.q_p,
        "tau2": r.tau2,
        # meta package stores $I2 as a proportion (0-1); PooledResult.i2 is
        # the percentage (0-100, metafor convention), so divide by 100.
        "i2": r.i2 / 100.0,
        "h2": r.h2,
        "weights": r.weights,
        "pi_lower": r.pi_lower,
        "pi_upper": r.pi_upper,
    }


def _study_effect(event, n, sm: str, incr: float):
    """Per-study transformed proportion and sampling variance (meta formulas)."""
    yi = []
    vi = []
    for x, ni in zip(event, n):
        if sm == "PFT":
            y = 0.5 * (
                math.asin(math.sqrt(x / (ni + 1.0)))
                + math.asin(math.sqrt((x + 1.0) / (ni + 1.0)))
            )
            v = 1.0 / (4.0 * ni + 2.0)
        elif sm == "PAS":
            y = math.asin(math.sqrt(x / ni))
            v = 1.0 / (4.0 * ni)
        elif sm == "PLN":
            y = math.log((x + incr) / (ni + incr))
            if x == ni:
                v = 1.0 / x - 1.0 / (ni + incr)
            else:
                v = 1.0 / (x + incr) - 1.0 / (ni + incr)
        elif sm == "PLOGIT":
            y = math.log((x + incr) / (ni - x + incr))
            v = 1.0 / (x + incr) + 1.0 / (ni - x + incr)
        else:  # PRAW
            y = x / ni
            v = (x + incr) * (ni - x + incr) / (ni + 2.0 * incr) ** 3
        yi.append(y)
        vi.append(v)
    return yi, vi


def _asin2p(x: float, n: float | None = None) -> float:
    """Back-transform a (Freeman-Tukey) arcsine value to a proportion.

    With ``n=None`` this is the plain arcsine inverse ``sin(x)^2``; with a
    sample size it is the Miller (1978) Freeman-Tukey inverse, matching
    ``meta::asin2p``.
    """
    if n is None:
        return math.sin(x) ** 2

    minimum = 0.5 * (
        math.asin(math.sqrt(0.0 / (n + 1.0))) + math.asin(math.sqrt(1.0 / (n + 1.0)))
    )
    maximum = 0.5 * (
        math.asin(math.sqrt(n / (n + 1.0))) + math.asin(math.sqrt((n + 1.0) / (n + 1.0)))
    )
    if x < minimum:
        return 0.0
    if x > maximum:
        return 1.0
    t = 2.0 * x
    return 0.5 * (
        1.0
        - math.copysign(1.0, math.cos(t))
        * math.sqrt(1.0 - (math.sin(t) + (math.sin(t) - 1.0 / math.sin(t)) / n) ** 2)
    )


def _backtransf(x: float, sm: str, n: float | None = None) -> float:
    """Back-transform an analysis-scale pooled value to a proportion."""
    if sm == "PLN":
        return min(1.0, max(0.0, math.exp(x)))
    if sm == "PLOGIT":
        return 1.0 / (1.0 + math.exp(-x))
    if sm == "PAS":
        return math.sin(x) ** 2
    if sm == "PFT":
        return _asin2p(x, n)
    # PRAW
    return min(1.0, max(0.0, x))


def metaprop(
    event,
    n,
    sm: str = "PLOGIT",
    random: bool = True,
    method_tau: str = "REML",
    incr: float = 0.5,
    alpha: float = 0.05,
) -> MetapropResult:
    """Pool single-arm proportions (mirrors ``meta::metaprop``).

    Args:
        event: number of events per study.
        n: sample size per study.
        sm: "PFT" (Freeman-Tukey double arcsine, default), "PAS", "PLN",
            "PLOGIT" or "PRAW".
        random: also fit the random-effects model.
        method_tau: tau-squared estimator (default DL).
        incr: continuity correction for PLN/PLOGIT/PRAW (default 0.5).
    """
    sm = sm.upper()
    if sm not in _SM:
        raise ValueError(f"sm must be one of {sorted(_SM)}")

    event = np.asarray(event, dtype=float)
    n = np.asarray(n, dtype=float)
    if len(event) != len(n) or len(event) == 0:
        raise ValueError("event and n must be non-empty equal-length arrays")

    yi, vi = _study_effect(event, n, sm, incr)

    # effect_type for pool_yv: choose one that leaves analysis scale untouched
    # except PLN (log -> exp is correct). We always back-transform ourselves,
    # so use "PAS"/"PFT" (raw) to keep pooled_est/se on the analysis scale.
    fixed = pool_yv(yi, vi, list(n), "PFT", model="fixed", alpha=alpha)
    random_res = (
        pool_yv(yi, vi, list(n), "PFT", model="random", method=method_tau, alpha=alpha)
        if random
        else None
    )

    # meta back-transforms PFT (double arcsine) using the harmonic mean of
    # the sample sizes for BOTH the fixed- and random-effects estimates.
    n_bt = 1.0 / float(np.mean(1.0 / n))
    n_fixed = n_random = n_bt

    def _ci(res: PooledResult, nn: float):
        lo = res.pooled_est - Z_975 * res.pooled_se
        hi = res.pooled_est + Z_975 * res.pooled_se
        return _backtransf(lo, sm, nn), _backtransf(hi, sm, nn)

    fixed_prop = _backtransf(fixed.pooled_est, sm, n_fixed)
    fl, fu = _ci(fixed, n_fixed)

    random_prop: float | None
    rl: float | None
    ru: float | None
    if random_res is not None:
        random_prop = _backtransf(random_res.pooled_est, sm, n_random)
        rl, ru = _ci(random_res, n_random)
    else:
        random_prop = rl = ru = None

    return MetapropResult(
        sm=sm,
        k=len(yi),
        n=list(n),
        fixed_prop=fixed_prop,
        fixed_lower=fl,
        fixed_upper=fu,
        random_prop=random_prop,
        random_lower=rl,
        random_upper=ru,
        yi=yi,
        vi=vi,
        fixed=fixed,
        random=random_res,
        event=[float(v) for v in event],
        method_tau=str(method_tau).upper(),
    )
