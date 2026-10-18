"""Cumulative meta-analysis (meta::cumul / metafor::cumul.rma.uni).

Repeatedly pools the first 1, 2, ..., k studies (in the supplied order),
returning the pooled estimate, SE, z, p, CI, Cochran Q, tau^2, I^2 and H^2 at
each step. Mirrors the random-effects (or fixed-effect) cumulative analysis.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
from scipy import stats

from .pooling import PooledResult, pool_yv

Z_975 = stats.norm.ppf(0.975)


@dataclass
class CumulResult:
    """Cumulative meta-analysis: one row per prefix of studies."""

    estimate: list[float]
    se: list[float]
    z: list[float]
    p: list[float]
    ci_lower: list[float]
    ci_upper: list[float]
    q: list[float]
    q_p: list[float]
    tau2: list[float]
    i2: list[float]
    h2: list[float]
    k: list[int] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "estimate": self.estimate,
            "se": self.se,
            "z": self.z,
            "p": self.p,
            "ci_lower": self.ci_lower,
            "ci_upper": self.ci_upper,
            "q": self.q,
            "q_p": self.q_p,
            "tau2": self.tau2,
            "i2": self.i2,
            "h2": self.h2,
            "k": self.k,
        }


def _single_study(y: float, v: float, alpha: float) -> PooledResult:
    """PooledResult for a single study (no heterogeneity possible)."""
    se = math.sqrt(v)
    z = y / se
    p = 2.0 * float(stats.norm.cdf(-abs(z)))
    return PooledResult(
        model="random",
        method="",
        k=1,
        effect_type="",
        pooled_effect=y,
        pooled_ci_lower=y - Z_975 * se,
        pooled_ci_upper=y + Z_975 * se,
        pooled_est=y,
        pooled_se=se,
        z=z,
        p_value=p,
        q=0.0,
        q_df=0,
        q_p=1.0,   # metafor::rma.uni returns QEp=1 for a single study
        tau2=0.0,
        i2=0.0,
        h2=1.0,    # metafor::rma.uni returns H2=1 for a single study
        weights=[1.0 / v],
    )


def cumul(
    yi,
    vi,
    n=None,
    effect_type: str = "MD",
    method: str = "DL",
    alpha: float = 0.05,
) -> CumulResult:
    """Cumulative (sequential) meta-analysis (mirrors ``meta::cumul``).

    Args:
        yi, vi: per-study effect and sampling variance (analysis scale).
        n: per-study sample size (for the HS tau2 estimator).
        effect_type: passed through to pooling (for back-transformation).
        method: tau-squared estimator (default DL).
    """
    yi = np.asarray(yi, dtype=float)
    vi = np.asarray(vi, dtype=float)
    k = len(yi)
    if k == 0:
        raise ValueError("no studies supplied")

    est, se, z, p, ci_l, ci_u = [], [], [], [], [], []
    q, qp, tau2, i2, h2, kk = [], [], [], [], [], []

    for i in range(1, k + 1):
        y, v = yi[:i], vi[:i]
        ni = (list(n[:i]) if n is not None else None)
        if i == 1:
            res = _single_study(float(y[0]), float(v[0]), alpha)
        else:
            res = pool_yv(list(y), list(v), ni, effect_type, model="random", method=method, alpha=alpha)
        est.append(res.pooled_effect)
        se.append(res.pooled_se)
        z.append(res.z)
        p.append(res.p_value)
        ci_l.append(res.pooled_ci_lower)
        ci_u.append(res.pooled_ci_upper)
        q.append(res.q)
        qp.append(res.q_p)
        tau2.append(res.tau2)
        i2.append(res.i2)
        h2.append(res.h2)
        kk.append(i)

    return CumulResult(est, se, z, p, ci_l, ci_u, q, qp, tau2, i2, h2, kk)
