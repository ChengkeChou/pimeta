"""Correlation meta-analysis (meta::metacor).

Replicates ``meta::metacor``: transforms correlations to Fisher's z
(``sm="ZCOR"``, the default) and pools with inverse-variance weighting, then
back-transforms to the correlation scale with ``tanh``. ``sm="COR"`` pools
raw correlations directly.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from scipy import stats

from .checks import attach_diagnostics
from .metaprop import _pool_dict
from .pooling import PooledResult, attach_tau2, pool_yv

Z_975 = stats.norm.ppf(0.975)

_SM = {"ZCOR", "COR"}


@dataclass
@attach_tau2
@attach_diagnostics
class MetacorResult:
    """Correlation meta-analysis result (fixed + random, correlation scale)."""

    sm: str
    k: int
    fixed_cor: float
    fixed_lower: float
    fixed_upper: float
    random_cor: float | None
    random_lower: float | None
    random_upper: float | None
    yi: list[float]
    vi: list[float]
    fixed: PooledResult
    random: PooledResult | None

    def to_dict(self) -> dict:
        return {
            "sm": self.sm,
            "k": self.k,
            "fixed_cor": self.fixed_cor,
            "fixed_lower": self.fixed_lower,
            "fixed_upper": self.fixed_upper,
            "random_cor": self.random_cor,
            "random_lower": self.random_lower,
            "random_upper": self.random_upper,
            "yi": self.yi,
            "vi": self.vi,
            "fixed": _pool_dict(self.fixed),
            "random": _pool_dict(self.random) if self.random else None,
        }


def _z2cor(x: float) -> float:
    """Inverse Fisher z-transform (== tanh)."""
    return math.tanh(x)


def metacor(
    cor,
    n,
    sm: str = "ZCOR",
    random: bool = True,
    method_tau: str = "REML",
    alpha: float = 0.05,
) -> MetacorResult:
    """Pool correlations (mirrors ``meta::metacor``).

    Args:
        cor: correlation coefficient per study.
        n: sample size per study.
        sm: "ZCOR" (Fisher's z, default) or "COR" (raw).
        random: also fit the random-effects model.
        method_tau: tau-squared estimator (default DL).
    """
    sm = sm.upper()
    if sm not in _SM:
        raise ValueError(f"sm must be one of {sorted(_SM)}")

    cor = np.asarray(cor, dtype=float)
    n = np.asarray(n, dtype=float)
    if len(cor) != len(n) or len(cor) == 0:
        raise ValueError("cor and n must be non-empty equal-length arrays")

    yi, vi = [], []
    for r, ni in zip(cor, n):
        if sm == "ZCOR":
            yi.append(0.5 * math.log((1.0 + r) / (1.0 - r)))
            vi.append(1.0 / (ni - 3.0))
        else:  # COR
            yi.append(float(r))
            vi.append((1.0 - r ** 2) ** 2 / (ni - 1.0))

    fixed = pool_yv(yi, vi, list(n), "PFT", model="fixed", alpha=alpha)
    random_res = (
        pool_yv(yi, vi, list(n), "PFT", model="random", method=method_tau, alpha=alpha)
        if random
        else None
    )

    bt = _z2cor if sm == "ZCOR" else (lambda x: x)

    def _ci(res: PooledResult):
        lo = res.pooled_est - Z_975 * res.pooled_se
        hi = res.pooled_est + Z_975 * res.pooled_se
        return bt(lo), bt(hi)

    fl, fu = _ci(fixed)
    if random_res is not None:
        rl, ru = _ci(random_res)
    else:
        rl = ru = None

    return MetacorResult(
        sm=sm,
        k=len(yi),
        fixed_cor=bt(fixed.pooled_est),
        fixed_lower=fl,
        fixed_upper=fu,
        random_cor=bt(random_res.pooled_est) if random_res else None,
        random_lower=rl,
        random_upper=ru,
        yi=yi,
        vi=vi,
        fixed=fixed,
        random=random_res,
    )
