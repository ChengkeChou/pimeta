"""Survival outcomes: Kaplan–Meier input, RMST, and curve reconstruction.

A survival meta-analysis rarely has hazard ratios as its starting point — it has
published Kaplan–Meier curves. Two things are needed to use those:

``rmst``
    The restricted mean survival time, the area under the curve up to a
    horizon: the number of months (or years) a patient can expect to live in
    the next :math:`\\tau`. It is the one summary that stays meaningful when
    proportional hazards does not hold, which is exactly when a pooled hazard
    ratio is worst.

``km_reconstruct``
    Individual times and event indicators rebuilt from the (time, number at
    risk, number of events) columns of a published curve — the reconstruction
    of Guyot et al. (2012). This is how IPD is obtained for most published
    trials, and it is what feeds :func:`~pimeta.ipd.ipd_two_stage`.

    >>> km = ma.km_curve(time, event)              # study-level vectors
    >>> ma.rmst(km.time, km.surv, tau=24)
    >>> ipd = ma.km_reconstruct(t, n_risk, n_event, total=40)

**What the reconstruction does and does not do.** Events are placed at the
reported time points and censoring is spread uniformly across the interval that
produced it. That reproduces the published curve exactly at every reported time
point — which is what any downstream analysis actually uses — but it does not
invent interior event times the paper never published. Guyot's refinement
(placing events inside an interval using an interpolated curve) needs digitised
survival values between the time points; when you have those, they change the
reconstruction, not the curve.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy import stats

__all__ = [
    "KmResult",
    "ReconstructedIpD",
    "RmstResult",
    "km_curve",
    "km_reconstruct",
    "rmst",
]


# --------------------------------------------------------------------------- #
# Kaplan-Meier
# --------------------------------------------------------------------------- #
@dataclass
class KmResult:
    """A Kaplan–Meier curve, with the risk sets it came from.

    Attributes:
        time: the event times, excluding time 0.
        n_risk: number at risk just before each of those times.
        n_event: events at each of those times.
        n_censored: censored just after each of those times.
        surv: the survival estimate at each time (``S(0) = 1`` implied).
        se: Greenwood standard error of ``surv``.
        n: number of participants.
    """

    time: np.ndarray
    n_risk: np.ndarray
    n_event: np.ndarray
    n_censored: np.ndarray
    surv: np.ndarray
    se: np.ndarray
    n: int

    def to_dict(self) -> dict:
        return {
            "time": np.asarray(self.time, dtype=float).tolist(),
            "n_risk": np.asarray(self.n_risk).tolist(),
            "n_event": np.asarray(self.n_event).tolist(),
            "n_censored": np.asarray(self.n_censored).tolist(),
            "surv": np.asarray(self.surv, dtype=float).tolist(),
            "se": np.asarray(self.se, dtype=float).tolist(),
            "n": self.n,
        }


def km_curve(time, event) -> KmResult:
    """Kaplan–Meier survivor function from individual times.

    Args:
        time: follow-up time per participant.
        event: 1 for an event, 0 for a censoring.

    Returns:
        :class:`KmResult`.

    Raises:
        ValueError: for mismatched lengths, negative times, or an event
            indicator that is not 0/1.

    Example:
        >>> km = ma.km_curve([1, 2, 2, 5], [1, 1, 0, 1])
        >>> [round(v, 4) for v in km.surv]
        [0.75, 0.5, 0.5]
    """
    time = np.asarray(time, dtype=float).ravel()
    event = np.asarray(event, dtype=float).ravel()
    if time.size != event.size:
        raise ValueError("'time' and 'event' must be the same length")
    if np.any(time < 0.0):
        raise ValueError("'time' cannot be negative")
    if not np.all((event == 0.0) | (event == 1.0)):
        raise ValueError("'event' must be 0 (censored) or 1 (event)")

    times = np.unique(time)
    n = time.size
    n_risk = np.empty(times.size, dtype=int)
    n_event = np.empty(times.size, dtype=int)
    n_censored = np.empty(times.size, dtype=int)
    surv = np.empty(times.size, dtype=float)
    se = np.empty(times.size, dtype=float)

    at_risk = n
    running = 1.0
    greenwood = 0.0
    for i, t in enumerate(times):
        deaths = int(np.sum((time == t) & (event == 1.0)))
        censored = int(np.sum((time == t) & (event == 0.0)))
        n_risk[i] = at_risk
        n_event[i] = deaths
        n_censored[i] = censored
        if deaths:
            running *= 1.0 - deaths / at_risk
            if at_risk > deaths:
                greenwood += deaths / (at_risk * (at_risk - deaths))
            else:
                greenwood = np.inf
        surv[i] = running
        se[i] = running * np.sqrt(greenwood) if np.isfinite(greenwood) else np.nan
        at_risk -= deaths + censored
    return KmResult(time=times, n_risk=n_risk, n_event=n_event,
                    n_censored=n_censored, surv=surv, se=se, n=n)


# --------------------------------------------------------------------------- #
# restricted mean survival time
# --------------------------------------------------------------------------- #
@dataclass
class RmstResult:
    """Restricted mean survival time up to a horizon.

    Attributes:
        rmst: the estimate.
        se: standard error (``None`` when the risk sets were not supplied).
        ci_lb, ci_ub: confidence interval (``None`` alongside ``se``).
        tau: the horizon.
        level: confidence level.
        surv: the survival values used.
        time: the times used.
        truncated: whether ``tau`` fell before the last reported time.
    """

    rmst: float
    se: float | None
    ci_lb: float | None
    ci_ub: float | None
    tau: float
    level: float
    surv: np.ndarray = field(repr=False, default_factory=lambda: np.empty(0))
    time: np.ndarray = field(repr=False, default_factory=lambda: np.empty(0))
    truncated: bool = False

    def to_dict(self) -> dict:
        return {
            "rmst": self.rmst,
            "se": self.se,
            "ci_lb": self.ci_lb,
            "ci_ub": self.ci_ub,
            "tau": self.tau,
            "level": self.level,
            "truncated": self.truncated,
        }

    def __str__(self) -> str:
        if self.se is None:
            return f"RMST({self.tau:g}) = {self.rmst:.4f}"
        return (f"RMST({self.tau:g}) = {self.rmst:.4f} "
                f"(se {self.se:.4f}; {self.level:.0%} CI "
                f"{self.ci_lb:.4f} to {self.ci_ub:.4f})")


def rmst(time, surv, *, tau: float | None = None, n_risk=None, n_event=None,
         level: float = 0.95) -> RmstResult:
    """Restricted mean survival time: the area under the survival curve.

    Args:
        time: the times at which the curve is reported (event times).
        surv: the survival estimate at each of those times. ``S(0) = 1`` is
            assumed and must not be included.
        tau: the horizon. Defaults to the last reported time.
        n_risk: number at risk just before each time; supplying it (with
            ``n_event``) adds the standard error.
        n_event: events at each time.
        level: confidence level.

    Returns:
        :class:`RmstResult`.

    Raises:
        ValueError: for mismatched lengths, times that are not increasing, or
            ``tau`` beyond the data with no curve to integrate.

    Example:
        >>> ma.rmst([12, 24], [0.8, 0.6], tau=24).rmst      # doctest: +SKIP
    """
    time = np.asarray(time, dtype=float).ravel()
    surv = np.asarray(surv, dtype=float).ravel()
    if time.size != surv.size:
        raise ValueError("'time' and 'surv' must be the same length")
    if time.size == 0:
        raise ValueError("the curve is empty")
    if np.any(np.diff(time) <= 0.0):
        raise ValueError("'time' must be strictly increasing")
    if np.any((surv < 0.0) | (surv > 1.0)):
        raise ValueError("'surv' must lie in [0, 1]")

    last = float(time[-1])
    limit = last if tau is None else float(tau)
    if limit <= 0.0:
        raise ValueError("'tau' must be positive")
    truncated = limit < last

    # The step function: S is constant on [t_i, t_{i+1}), starting from S = 1
    # on [0, t_1).  Beyond the last reported time the curve is held flat, which
    # is the same convention survival::survfit uses.
    knots = np.concatenate([[0.0], time])
    heights = np.concatenate([[1.0], surv])
    keep = knots < limit
    knots = np.concatenate([knots[keep], [limit]])
    heights = np.concatenate([heights[keep], heights[keep][-1:]])
    area = float(np.sum(heights[:-1] * np.diff(knots)))

    if n_risk is None or n_event is None:
        return RmstResult(rmst=area, se=None, ci_lb=None, ci_ub=None, tau=limit,
                          level=level, surv=surv, time=time,
                          truncated=truncated)

    n_risk = np.asarray(n_risk, dtype=float).ravel()
    n_event = np.asarray(n_event, dtype=float).ravel()
    if n_risk.size != time.size or n_event.size != time.size:
        raise ValueError("'n_risk' and 'n_event' must match 'time'")
    # Uno et al. (2014): Var = sum over event times of
    #   (area remaining after t_i)^2 * d_i / (n_i (n_i - d_i)).
    variance = 0.0
    for i, t in enumerate(time):
        if t > limit or n_event[i] <= 0.0 or n_risk[i] <= n_event[i]:
            continue
        mask = knots >= t
        tail_knots = knots[mask]
        tail_heights = heights[mask]
        remaining = float(np.sum(tail_heights[:-1] * np.diff(tail_knots)))
        variance += remaining ** 2 * n_event[i] / (n_risk[i] * (n_risk[i] - n_event[i]))
    se = float(np.sqrt(variance))
    crit = float(stats.norm.ppf(0.5 + level / 2.0))
    return RmstResult(rmst=area, se=se, ci_lb=area - crit * se,
                      ci_ub=area + crit * se, tau=limit, level=level,
                      surv=surv, time=time, truncated=truncated)


# --------------------------------------------------------------------------- #
# reconstructing individual data from a published curve
# --------------------------------------------------------------------------- #
@dataclass
class ReconstructedIpD:
    """Individual times and event indicators rebuilt from a KM curve.

    Attributes:
        time: reconstructed follow-up time per participant.
        event: 1 for an event, 0 for a censoring.
        n: number of participants.
        n_event: number of events.
        km: the :class:`KmResult` the reconstruction reproduces.
    """

    time: np.ndarray
    event: np.ndarray
    n: int
    n_event: int
    km: KmResult = field(repr=False)

    def to_dict(self) -> dict:
        return {
            "time": np.asarray(self.time, dtype=float).tolist(),
            "event": np.asarray(self.event).tolist(),
            "n": self.n,
            "n_event": self.n_event,
        }


def km_reconstruct(time, n_risk, n_event, *, total: int | None = None) -> ReconstructedIpD:
    """Rebuild individual data from a published Kaplan–Meier summary.

    Args:
        time: the reported time points (strictly increasing, all positive).
        n_risk: number at risk just before each time point.
        n_event: events at each time point.
        total: participants at risk at time 0. Defaults to ``n_risk[0]``.

    Censoring is spread evenly across the interval that produced it. It has to
    fall strictly inside the interval: placing it at the interval's end would
    put those participants back into the next risk set and change the curve.

    Returns:
        :class:`ReconstructedIpD`. Feeding ``time``/``event`` to
        :func:`km_curve` reproduces ``n_risk``, ``n_event`` and the survival
        values implied by the input.

    Raises:
        ValueError: for inconsistent risk sets — a value that increases, or
            fewer at risk than the events plus the next risk set.

    Example:
        >>> ipd = ma.km_reconstruct([2, 5, 9], [36, 30, 21], [4, 5, 6], total=40)
        >>> ipd.n, ipd.n_event
        (40, 15)
    """
    time = np.asarray(time, dtype=float).ravel()
    n_risk = np.asarray(n_risk, dtype=int).ravel()
    n_event = np.asarray(n_event, dtype=int).ravel()
    if not (time.size == n_risk.size == n_event.size):
        raise ValueError("'time', 'n_risk' and 'n_event' must be the same length")
    if time.size == 0:
        raise ValueError("the curve is empty")
    if np.any(np.diff(time) <= 0.0):
        raise ValueError("'time' must be strictly increasing")
    if np.any(n_event < 0) or np.any(n_event > n_risk):
        raise ValueError("'n_event' cannot exceed 'n_risk'")
    if np.any(np.diff(n_risk) > 0.0):
        raise ValueError(
            "the number at risk cannot increase; check that the curve is "
            "ordered and that competing risks were not mixed in")

    n0 = int(n_risk[0]) if total is None else int(total)
    if n0 < int(n_risk[0]):
        raise ValueError(
            f"'total' ({n0}) cannot be smaller than the first risk set "
            f"({int(n_risk[0])})")

    times: list[float] = []
    events: list[int] = []

    # Between time 0 and the first reported time the paper publishes only the
    # risk set, so those participants are censored at time 0.  Putting them at
    # the first event time instead would inflate that time's risk set.
    gap = n0 - int(n_risk[0])
    if gap > 0:
        times.extend([0.0] * gap)
        events.extend([0] * gap)

    for i in range(time.size):
        deaths = int(n_event[i])
        times.extend([time[i]] * deaths)
        events.extend([1] * deaths)
        remaining = int(n_risk[i]) - deaths
        next_risk = int(n_risk[i + 1]) if i + 1 < time.size else 0
        censored = remaining - next_risk
        if censored < 0:
            raise ValueError(
                f"inconsistent risk sets at time {time[i]:g}: {int(n_risk[i])} "
                f"at risk, {deaths} events, but {next_risk} at risk at the "
                "next time point")
        if censored == 0:
            continue
        lo = float(time[i])
        hi = float(time[i + 1]) if i + 1 < time.size else float(time[i])
        positions = np.linspace(lo, hi, censored + 2)[1:-1]
        times.extend(float(v) for v in positions)
        events.extend([0] * censored)

    time_arr = np.asarray(times, dtype=float)
    event_arr = np.asarray(events, dtype=int)
    km = km_curve(time_arr, event_arr)
    return ReconstructedIpD(time=time_arr, event=event_arr, n=time_arr.size,
                            n_event=int(event_arr.sum()), km=km)
