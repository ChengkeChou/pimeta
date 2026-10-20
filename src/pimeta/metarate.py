"""Single-arm incidence-rate meta-analysis (meta::metarate).

Replicates ``meta::metarate``: transforms per-study incidence rates to a
normal scale, pools with inverse-variance weighting (fixed or random effects),
then back-transforms the pooled result and CI to the rate scale.

Supported ``sm`` (summary measure) and their transformations:

- ``IRLN``   log rate  (default): ``yi = log((event + incr)/time)``,
  ``vi = 1/(event + incr)``
- ``IR``     raw rate: ``yi = (event + incr)/time``, ``vi = yi/time``
- ``IRS``    square-root: ``yi = sqrt(event/time)``, ``vi = 1/(4*time)``
- ``IRFT``   Freeman-Tukey double arcsine:
  ``yi = 0.5*(sqrt(event/time) + sqrt((event+1)/time))``, ``vi = 1/(4*time)``

The ``incr`` continuity correction is applied to zero-event studies for
``IR``/``IRLN`` (``method.incr="only0"`` default), matching ``meta``. For
``IRFT`` the back-transformation uses the harmonic mean of ``time``, exactly
as ``meta::metarate`` does with ``meta::asin2ir``.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from scipy import stats

from .pooling import PooledResult, pool_yv

Z_975 = stats.norm.ppf(0.975)

_SM = {"IR", "IRLN", "IRS", "IRFT"}
_METHOD_INCR = {"only0", "all", "if0all", "user"}


@dataclass
class MetarateResult:
    """Incidence-rate meta-analysis result (fixed + random, rate scale)."""

    sm: str
    k: int
    event: list[float]                 # per-study number of events
    time: list[float]                  # per-study person-time at risk
    incr_event: list[float]            # continuity correction actually used
    # analysis-scale per-study values
    yi: list[float]
    vi: list[float]
    # rate-scale pooled estimates + CI
    fixed_rate: float
    fixed_lower: float
    fixed_upper: float
    random_rate: float | None
    random_lower: float | None
    random_upper: float | None
    # overall heterogeneity (meta convention: I2 = (Q - df)/Q as a proportion)
    i2: float
    q: float
    q_df: int
    q_p: float
    tau2: float | None
    # underlying PooledResult objects (analysis scale)
    fixed: PooledResult
    random: PooledResult | None

    def to_dict(self) -> dict:
        return {
            "sm": self.sm,
            "k": self.k,
            "event": self.event,
            "time": self.time,
            "incr_event": self.incr_event,
            "yi": self.yi,
            "vi": self.vi,
            "fixed_rate": self.fixed_rate,
            "fixed_lower": self.fixed_lower,
            "fixed_upper": self.fixed_upper,
            "random_rate": self.random_rate,
            "random_lower": self.random_lower,
            "random_upper": self.random_upper,
            "i2": self.i2,
            "q": self.q,
            "q_df": self.q_df,
            "q_p": self.q_p,
            "tau2": self.tau2,
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
        "i2": r.i2 / 100.0,
        "h2": r.h2,
        "weights": r.weights,
        "pi_lower": r.pi_lower,
        "pi_upper": r.pi_upper,
    }


def _incr_event(event: np.ndarray, sm: str, incr: float, method_incr: str) -> np.ndarray:
    """Per-study continuity correction (meta::metarate 'incr.event')."""
    k = len(event)
    # Only IR / IRLN use a continuity correction for zero events.
    sel = (event == 0) if sm in ("IR", "IRLN") else np.zeros(k, dtype=bool)
    sparse = bool(sel.any())
    if method_incr == "all":
        return np.full(k, incr)
    if method_incr == "if0all":
        return np.full(k, incr) if sparse else np.zeros(k)
    # "only0" (default) and "user" both handled by caller for user-supplied incr
    return incr * sel.astype(float)


def _study_effect(event, time, sm: str, incr_event):
    """Per-study transformed rate and sampling variance (meta formulas)."""
    yi = []
    vi = []
    for x, t, inc in zip(event, time, incr_event):
        if sm == "IR":
            y = (x + inc) / t
            v = y / t
        elif sm == "IRLN":
            y = math.log((x + inc) / t)
            v = 1.0 / (x + inc)
        elif sm == "IRS":
            y = math.sqrt(x / t)
            v = 1.0 / (4.0 * t)
        else:  # IRFT
            y = 0.5 * (math.sqrt(x / t) + math.sqrt((x + 1.0) / t))
            v = 1.0 / (4.0 * t)
        yi.append(y)
        vi.append(v)
    return yi, vi


def _asin2ir(x: float, time: float) -> float:
    """Back-transform a Freeman-Tukey arcsine rate value (meta::asin2ir)."""
    minimum = 0.5 * (math.sqrt(0.0 / time) + math.sqrt(1.0 / time))
    if x < minimum:
        return 0.0
    num = 1.0 / time - 8.0 * x ** 2 + 16.0 * time * x ** 4
    den = 16.0 * x ** 2 * time
    res = num / den
    return max(0.0, res)


def _backtransf(x: float, sm: str, time: float | None = None) -> float:
    """Back-transform an analysis-scale pooled value to a rate."""
    if sm == "IRLN":
        return math.exp(x)
    if sm == "IRS":
        return x ** 2
    if sm == "IRFT":
        return _asin2ir(x, time) if time is not None else float("nan")
    # IR
    return x


def metarate(
    event,
    time,
    sm: str = "IRLN",
    random: bool = True,
    method_tau: str = "REML",
    incr: float = 0.5,
    method_incr: str = "only0",
    alpha: float = 0.05,
) -> MetarateResult:
    """Pool single-arm incidence rates (mirrors ``meta::metarate``).

    Args:
        event: number of events per study.
        time: person-time at risk per study.
        sm: "IRLN" (log rate, default), "IR", "IRS" or "IRFT".
        random: also fit the random-effects model.
        method_tau: tau-squared estimator (default REML).
        incr: continuity correction added to zero-event studies (default 0.5).
        method_incr: "only0" (default), "all", "if0all" or "user" (in which
            case ``incr`` may be a per-study vector).
    """
    sm = sm.upper()
    if sm not in _SM:
        raise ValueError(f"sm must be one of {sorted(_SM)}")
    if method_incr not in _METHOD_INCR:
        raise ValueError(f"method_incr must be one of {sorted(_METHOD_INCR)}")

    event = np.asarray(event, dtype=float)
    time = np.asarray(time, dtype=float)
    if len(event) != len(time) or len(event) == 0:
        raise ValueError("event and time must be non-empty equal-length arrays")
    if (time <= 0).any():
        raise ValueError("'time' must be positive")
    if (event < 0).any():
        raise ValueError("'event' must be non-negative")

    if method_incr == "user":
        incr_event = np.broadcast_to(np.asarray(incr, dtype=float), event.shape)
    else:
        incr_event = _incr_event(event, sm, float(incr), method_incr)

    yi, vi = _study_effect(event, time, sm, incr_event)

    # pool_yv with a "raw" effect type keeps pooled_est/se on the analysis
    # scale (no internal back-transform); we back-transform ourselves.
    fixed = pool_yv(yi, vi, [float("nan")] * len(yi), "PAS", model="fixed", alpha=alpha)
    random_res = (
        pool_yv(yi, vi, [float("nan")] * len(yi), "PAS", model="random", method=method_tau, alpha=alpha)
        if random
        else None
    )

    q = fixed.q
    q_df = fixed.q_df
    # meta's $I2 is the Higgins-Thompson Q-based proportion (model-independent).
    i2 = max(0.0, (q - q_df) / q) if q > 0 else 0.0

    # meta back-transforms IRFT using the harmonic mean of the person-times.
    time_bt = 1.0 / float(np.mean(1.0 / time))

    def _ci(res: PooledResult):
        lo = res.pooled_est - Z_975 * res.pooled_se
        up = res.pooled_est + Z_975 * res.pooled_se
        return _backtransf(lo, sm, time_bt), _backtransf(up, sm, time_bt)

    fixed_rate = _backtransf(fixed.pooled_est, sm, time_bt)
    fixed_lo, fixed_up = _ci(fixed)
    random_rate = random_lo = random_up = None
    if random_res is not None:
        random_rate = _backtransf(random_res.pooled_est, sm, time_bt)
        random_lo, random_up = _ci(random_res)

    return MetarateResult(
        sm=sm,
        k=len(event),
        event=event.tolist(),
        time=time.tolist(),
        incr_event=incr_event.tolist(),
        yi=list(yi),
        vi=list(vi),
        fixed_rate=fixed_rate,
        fixed_lower=fixed_lo,
        fixed_upper=fixed_up,
        random_rate=random_rate,
        random_lower=random_lo,
        random_upper=random_up,
        i2=i2,
        q=q,
        q_df=q_df,
        q_p=fixed.q_p,
        tau2=random_res.tau2 if random_res else None,
        fixed=fixed,
        random=random_res,
    )
