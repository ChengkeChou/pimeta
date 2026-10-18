"""Incidence-rate meta-analysis (meta::metainc).

Replicates ``meta::metainc`` for event / person-time data:

- ``IRR``  incidence rate ratio (log scale, default)
- ``IRD``  incidence rate difference
- ``IRSD`` incidence rate square-root difference

The ``incr`` continuity correction follows ``meta``'s ``method.incr="if0all"``
default: if any study has zero events in either arm, ``incr`` (0.5) is added
to the event counts of *all* studies.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, replace

import numpy as np
from scipy import stats

from .pooling import PooledResult, pool_yv
from .metaprop import _pool_dict

Z_975 = stats.norm.ppf(0.975)

_SM = {"IRR", "IRD", "IRSD"}
_METHODS = {"MH", "Cochran", "Inverse"}


@dataclass
class MetaincResult:
    """Incidence-rate meta-analysis result (fixed + random).

    ``fixed_effect`` / ``random_effect`` are on the *analysis* scale (log for
    IRR), matching ``meta``'s ``$TE.common`` / ``$TE.random``. The
    back-transformed rate-scale values are exposed as ``fixed_rate`` /
    ``random_rate`` (what ``meta`` shows in printouts when ``backtransf=TRUE``).
    """

    sm: str
    method: str
    k: int
    fixed_effect: float
    fixed_lower: float
    fixed_upper: float
    random_effect: float | None
    random_lower: float | None
    random_upper: float | None
    fixed_rate: float
    fixed_rate_lower: float
    fixed_rate_upper: float
    random_rate: float | None
    random_rate_lower: float | None
    random_rate_upper: float | None
    yi: list[float]
    vi: list[float]
    fixed: PooledResult
    random: PooledResult | None

    def to_dict(self) -> dict:
        return {
            "sm": self.sm,
            "method": self.method,
            "k": self.k,
            "fixed_effect": self.fixed_effect,
            "fixed_lower": self.fixed_lower,
            "fixed_upper": self.fixed_upper,
            "random_effect": self.random_effect,
            "random_lower": self.random_lower,
            "random_upper": self.random_upper,
            "fixed_rate": self.fixed_rate,
            "fixed_rate_lower": self.fixed_rate_lower,
            "fixed_rate_upper": self.fixed_rate_upper,
            "random_rate": self.random_rate,
            "random_rate_lower": self.random_rate_lower,
            "random_rate_upper": self.random_rate_upper,
            "yi": self.yi,
            "vi": self.vi,
            "fixed": _pool_dict(self.fixed),
            "random": _pool_dict(self.random) if self.random else None,
        }


def _study_effect(event_e, time_e, event_c, time_c, sm: str, incr: float):
    yi, vi = [], []
    for x1, t1, x2, t2 in zip(event_e, time_e, event_c, time_c):
        if sm == "IRR":
            y = math.log(((x1 + incr) / t1) / ((x2 + incr) / t2))
            v = 1.0 / (x1 + incr) + 1.0 / (x2 + incr)
        elif sm == "IRD":
            y = x1 / t1 - x2 / t2
            v = (x1 + incr) / t1 ** 2 + (x2 + incr) / t2 ** 2
        else:  # IRSD
            y = math.sqrt(x1 / t1) - math.sqrt(x2 / t2)
            v = 0.25 / t1 + 0.25 / t2
        yi.append(y)
        vi.append(v)
    return yi, vi


def _mh_common(event_e, time_e, event_c, time_c, sm, yi):
    """Mantel-Haenszel common-effect estimate (Greenland & Robins 1985)."""
    x = np.asarray(event_e, dtype=float)
    y = np.asarray(event_c, dtype=float)
    n = np.asarray(time_e, dtype=float)
    m = np.asarray(time_c, dtype=float)
    N = n + m
    t = x + y

    if sm in ("IRR", "VE"):
        D = n * m * t / N ** 2
        R = x * m / N
        S = y * n / N
        sum_R = float(np.sum(R))
        sum_S = float(np.sum(S))
        te = math.log(sum_R / sum_S)
        se = math.sqrt(float(np.sum(D)) / (sum_R * sum_S))
    else:  # IRD
        L = (x * m ** 2 + y * n ** 2) / N ** 2
        S = n * m / N
        sum_S = float(np.sum(S))
        te = float(np.average(yi, weights=S))
        se = math.sqrt(float(np.sum(L)) / sum_S ** 2)
    return te, se


def _cochran_common(event_e, time_e, event_c, time_c, yi):
    """Cochran's common-effect estimate (IRR / VE only)."""
    w = np.asarray(event_c, dtype=float) * np.asarray(time_e, dtype=float) / np.asarray(
        time_c, dtype=float
    )
    te = float(np.average(yi, weights=w))
    se = math.sqrt(1.0 / float(np.sum(event_e)) + 1.0 / float(np.sum(event_c)))
    return te, se


def metainc(
    event_e,
    time_e,
    event_c,
    time_c,
    sm: str = "IRR",
    random: bool = True,
    method: str = "MH",
    method_tau: str = "REML",
    incr: float = 0.5,
    alpha: float = 0.05,
) -> MetaincResult:
    """Pool incidence rates (mirrors ``meta::metainc``).

    Args:
        event_e, time_e: treatment-arm event count and person-time per study.
        event_c, time_c: control-arm event count and person-time per study.
        sm: "IRR" (rate ratio, default), "IRD" (rate difference) or "IRSD".
        random: also fit the random-effects model.
        method: common-effect method "MH" (default), "Cochran" or "Inverse".
        method_tau: tau-squared estimator (default DL).
        incr: continuity correction added when any study has zero events.
    """
    sm = sm.upper()
    if sm not in _SM:
        raise ValueError(f"sm must be one of {sorted(_SM)}")
    method = method if sm != "IRSD" else "Inverse"
    if method not in _METHODS:
        raise ValueError(f"method must be one of {sorted(_METHODS)}")

    event_e = np.asarray(event_e, dtype=float)
    time_e = np.asarray(time_e, dtype=float)
    event_c = np.asarray(event_c, dtype=float)
    time_c = np.asarray(time_c, dtype=float)
    if not (len(event_e) == len(time_e) == len(event_c) == len(time_c) > 0):
        raise ValueError("event/time arrays must be non-empty equal-length")

    # method.incr = "if0all": add incr to all studies iff any study has a zero
    sparse = bool(np.any((event_e == 0) | (event_c == 0)))
    eff_incr = incr if sparse else 0.0

    yi, vi = _study_effect(event_e, time_e, event_c, time_c, sm, eff_incr)

    # ni (total person-time) for HS tau2 estimator
    ni = [t1 + t2 for t1, t2 in zip(time_e, time_c)]

    # inverse-variance results carry Q / I^2 / tau^2 (meta computes these on
    # the IV scale regardless of the common-effect method).
    fixed_iv = pool_yv(yi, vi, ni, "PFT", model="fixed", alpha=alpha)
    random_res = (
        pool_yv(yi, vi, ni, "PFT", model="random", method=method_tau, alpha=alpha)
        if random
        else None
    )

    if method == "MH":
        te_common, se_common = _mh_common(event_e, time_e, event_c, time_c, sm, yi)
    elif method == "Cochran":
        te_common, se_common = _cochran_common(event_e, time_e, event_c, time_c, yi)
    else:  # Inverse
        te_common, se_common = fixed_iv.pooled_est, fixed_iv.pooled_se

    z = te_common / se_common
    p = 2.0 * float(stats.norm.cdf(-abs(z)))
    lo = te_common - Z_975 * se_common
    hi = te_common + Z_975 * se_common

    fixed = replace(
        fixed_iv,
        pooled_est=te_common,
        pooled_se=se_common,
        pooled_effect=te_common,
        pooled_ci_lower=lo,
        pooled_ci_upper=hi,
        z=z,
        p_value=p,
    )

    bt = math.exp if sm == "IRR" else (lambda x: x)

    fl = fixed.pooled_est - Z_975 * fixed.pooled_se
    fu = fixed.pooled_est + Z_975 * fixed.pooled_se
    if random_res is not None:
        rl = random_res.pooled_est - Z_975 * random_res.pooled_se
        ru = random_res.pooled_est + Z_975 * random_res.pooled_se
    else:
        rl = ru = None

    return MetaincResult(
        sm=sm,
        method=method,
        k=len(yi),
        fixed_effect=fixed.pooled_est,
        fixed_lower=fl,
        fixed_upper=fu,
        random_effect=random_res.pooled_est if random_res else None,
        random_lower=rl,
        random_upper=ru,
        fixed_rate=bt(fixed.pooled_est),
        fixed_rate_lower=bt(fl),
        fixed_rate_upper=bt(fu),
        random_rate=bt(random_res.pooled_est) if random_res else None,
        random_rate_lower=bt(rl) if rl is not None else None,
        random_rate_upper=bt(ru) if ru is not None else None,
        yi=yi,
        vi=vi,
        fixed=fixed,
        random=random_res,
    )
