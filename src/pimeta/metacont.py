"""Continuous-outcome meta-analysis (meta::metacont).

Thin wrapper over :mod:`.escalc` + :mod:`.pooling`: computes MD or SMD
(Hedges' g) per study and pools with inverse-variance weighting (fixed or
random effects, any tau^2 estimator).
"""

from __future__ import annotations

from dataclasses import dataclass

from .escalc import escalc
from .pooling import PooledResult, pool_yv
from .metabin import _result_dict


@dataclass
class MetacontResult:
    """Continuous meta-analysis result: fixed + random pooled effects."""

    sm: str                       # "MD" | "SMD" | "SMDH"
    k: int
    fixed: PooledResult
    random: PooledResult | None
    yi: list[float]
    vi: list[float]
    # arm sizes and the tau^2 estimator, kept because R exposes x$n.e / x$n.c /
    # x$method.tau (metabias' Macaskill and Pustejovsky variants read them).
    n_e: list[float] | None = None
    n_c: list[float] | None = None
    method_tau: str = "REML"

    def to_dict(self) -> dict:
        return {
            "sm": self.sm,
            "k": self.k,
            "fixed": _result_dict(self.fixed),
            "random": _result_dict(self.random) if self.random else None,
            "yi": self.yi,
            "vi": self.vi,
            "n_e": self.n_e,
            "n_c": self.n_c,
            "method_tau": self.method_tau,
        }


def metacont(
    n_e,
    mean_e,
    sd_e,
    n_c,
    mean_c,
    sd_c,
    sm: str = "MD",
    random: bool = True,
    method_tau: str = "REML",
    alpha: float = 0.05,
) -> MetacontResult:
    """Pool continuous (mean/SD) study data (mirrors ``meta::metacont``).

    Args:
        n_e, mean_e, sd_e: treatment-arm size, mean, SD (per study).
        n_c, mean_c, sd_c: control-arm size, mean, SD (per study).
        sm: "MD" (mean difference) or "SMD"/"SMDH" (standardised mean
            difference, Hedges' g by default like meta::metacont).
        random: also fit the random-effects model.
        method_tau: tau-squared estimator (default DL).
    """
    sm = sm.upper()
    if sm not in ("MD", "SMD", "SMDH"):
        raise ValueError("sm must be MD, SMD or SMDH")

    esc = escalc(
        sm, n1i=n_e, m1i=mean_e, sd1i=sd_e, n2i=n_c, m2i=mean_c, sd2i=sd_c
    )
    yi, vi, ni = esc.yi, esc.vi, esc.ni

    fixed = pool_yv(yi, vi, ni, sm, model="fixed", alpha=alpha)
    random_result = (
        pool_yv(yi, vi, ni, sm, model="random", method=method_tau, alpha=alpha)
        if random
        else None
    )
    return MetacontResult(
        sm, len(yi), fixed, random_result, yi, vi,
        n_e=[float(v) for v in n_e], n_c=[float(v) for v in n_c],
        method_tau=str(method_tau).upper(),
    )
