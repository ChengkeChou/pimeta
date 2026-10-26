"""Prediction intervals from the confidence distribution of tau^2.

The plug-in interval ``est +/- z sqrt(se^2 + tau2_hat)`` treats the estimated
between-study variance as known. At ``k = 5`` that costs about fifteen points of
coverage, and neither a ``t`` quantile nor a parametric bootstrap recovers them:
the bootstrap draws from ``tau2_hat``, so when the estimate is zero it generates
homogeneous data and removes exactly the tail that was missing.

This module does what Nagashima, Noma & Furukawa (2019, *Statistics in Medicine*
38:1260) do instead --- integrate the uncertainty of ``tau2`` out exactly rather
than summarising it:

1. ``Q(tau2) = sum_i p_i (y_i - mu(tau2))^2`` with ``p_i = 1 / (v_i + tau2)`` is a
   pivot: at the true ``tau2`` it is ``chi^2_{k-1}`` for **any** ``k``, with no
   asymptotics. It is strictly decreasing in ``tau2``.
2. Inverting it gives a confidence distribution for ``tau2``: for a ``chi^2_{k-1}``
   quantile ``u``, the corresponding ``tau2`` is the root of ``Q(tau2) = u``, or
   zero when ``u >= Q(0)``. The point mass at zero is handled by the same formula
   rather than by a separate branch --- and, importantly, the positive part of the
   distribution keeps its mass even when the point estimate is zero.
3. Given ``tau2``, a new study's true effect is ``t_{k-2}`` around
   ``mu(tau2)`` with scale ``sqrt(V(mu) + tau2)``. Averaging that over the
   confidence distribution gives the predictive distribution, and its
   ``alpha/2`` and ``1 - alpha/2`` quantiles are the interval.

Why it works at small ``k`` and the plug-in does not, in one sentence: the
interval is no longer a point estimate of ``tau2`` scaled by a quantile, it is an
average over every ``tau2`` the data are compatible with, weighted by how
compatible they are.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from scipy import stats

__all__ = ["cd_prediction_interval", "heterogeneity_confidence_distribution"]

#: Grid size over the chi-square quantiles. 201 points resolves the predictive
#: quantiles to about 1e-4, which is well below the Monte Carlo error of any
#: coverage study that would check them.
DEFAULT_GRID = 201


def _q_stat(yi: np.ndarray, vi: np.ndarray, tau2: np.ndarray) -> np.ndarray:
    """Generalised Cochran ``Q`` at each ``tau2``, vectorised over ``tau2``.

    ``yi`` is ``(k,)`` and ``tau2`` is ``(m,)``; the result is ``(m,)``.
    """
    w = 1.0 / (vi[:, None] + tau2[None, :])
    sw = w.sum(axis=0)
    mu = (w * yi[:, None]).sum(axis=0) / sw
    return (w * (yi[:, None] - mu[None, :]) ** 2).sum(axis=0)


def heterogeneity_confidence_distribution(yi, vi, *, n_grid: int = DEFAULT_GRID
                                          ) -> dict[str, Any]:
    """The confidence distribution of ``tau2`` implied by the exact pivot.

    Args:
        yi: study effects.
        vi: their sampling variances.
        n_grid: how many chi-square quantiles to invert.

    Returns:
        Dictionary with ``tau2`` (the grid of values, ascending), ``weights``
        (equal, the grid is in probability space), ``mu`` and ``var_mu`` (the
        pooled effect and its variance at each ``tau2``), ``q0`` and
        ``point_mass`` (the probability the pivot puts at ``tau2 = 0``).

    Raises:
        ValueError: fewer than three studies. With ``k < 3`` the pivot has no
            degrees of freedom left and ``t_{k-2}`` does not exist.
    """
    yi = np.asarray(yi, dtype=float).ravel()
    vi = np.asarray(vi, dtype=float).ravel()
    if yi.size != vi.size:
        raise ValueError("'yi' and 'vi' must be the same length")
    if yi.size < 3:
        raise ValueError(
            "the confidence-distribution interval needs at least three studies")
    if np.any(vi <= 0.0) or not np.all(np.isfinite(vi)):
        raise ValueError("'vi' must be finite and positive")

    df = yi.size - 1
    probs = (np.arange(n_grid) + 0.5) / n_grid
    u = stats.chi2.ppf(probs, df=df)

    # Q is strictly decreasing, so each u has at most one root. Solve all of them
    # by vectorised bisection: monotone, deterministic, and about 60 iterations of
    # a k x n_grid array operation for the whole grid.
    q0 = float(_q_stat(yi, vi, np.zeros(1))[0])
    tau2 = np.zeros(n_grid)
    need = u < q0
    if np.any(need):
        lo = np.zeros(int(need.sum()))
        hi = np.full(int(need.sum()), max(1.0, float(np.var(yi)) * 10.0))
        target = u[need]
        for _ in range(200):
            mid = 0.5 * (lo + hi)
            above = _q_stat(yi, vi, mid) > target
            lo = np.where(above, mid, lo)
            hi = np.where(above, hi, mid)
            if np.max(hi - lo) <= 1e-12 * (1.0 + np.max(hi)):
                break
        tau2[need] = 0.5 * (lo + hi)

    w = 1.0 / (vi[:, None] + tau2[None, :])
    sw = w.sum(axis=0)
    mu = (w * yi[:, None]).sum(axis=0) / sw
    return {"tau2": tau2, "mu": mu, "var_mu": 1.0 / sw,
            "q0": q0, "point_mass": float(np.mean(u >= q0)),
            "df": np.array([df])}


def cd_prediction_interval(yi, vi, *, level: float = 0.95,
                           n_grid: int = DEFAULT_GRID,
                           hk: bool = False) -> dict[str, float]:
    """Prediction interval for a new study's true effect, by confidence distribution.

    Args:
        yi: study effects on the analysis scale.
        vi: their sampling variances.
        level: confidence level, as a proportion (0.95) or a percentage (95).
        n_grid: chi-square quantiles used to represent the confidence
            distribution.
        hk: scale the pooled-effect variance by the Hartung-Knapp factor
            ``max(1, Q/(k-1))`` at each grid point instead of using ``1/sum p``.
            Off by default because the published method does not use it; it is
            available because it is the one knob that could matter at very
            small ``k``, and a measurement rather than an argument should decide.

    Returns:
        ``{"pred", "pi_lb", "pi_ub", "level", "tau2_median", "point_mass"}``.

    Example:
        >>> out = cd_prediction_interval([0.1, 0.3, 0.2, 0.5], [0.04]*4)
        >>> out["pi_lb"] < out["pred"] < out["pi_ub"]
        True
    """
    from .levels import level_to_alpha

    alpha = level_to_alpha(level)
    dist = heterogeneity_confidence_distribution(yi, vi, n_grid=n_grid)
    tau2 = dist["tau2"]
    mu = dist["mu"]
    var_mu = dist["var_mu"]
    k = np.asarray(yi, dtype=float).size
    if hk:
        q = _q_stat(np.asarray(yi, dtype=float).ravel(),
                    np.asarray(vi, dtype=float).ravel(), tau2)
        var_mu = np.maximum(1.0, q / (k - 1.0)) * var_mu
    scale = np.sqrt(var_mu + tau2)
    df_t = max(1.0, k - 2.0)

    def cdf(x: float) -> float:
        return float(np.mean(stats.t.cdf((x - mu) / scale, df=df_t)))

    centre = float(np.median(mu))
    span = float(np.max(scale)) * 12.0 + 1.0
    lo, hi = centre - span, centre + span
    # expand until the tails bracket the target probabilities
    for _ in range(60):
        if cdf(lo) < alpha / 2.0:
            break
        lo -= span
    for _ in range(60):
        if cdf(hi) > 1.0 - alpha / 2.0:
            break
        hi += span
    pi_lb = _bisect(cdf, alpha / 2.0, lo, centre)
    pi_ub = _bisect(cdf, 1.0 - alpha / 2.0, centre, hi)
    return {"pred": float(np.median(mu)), "pi_lb": pi_lb, "pi_ub": pi_ub,
            "level": float(1.0 - alpha),
            "tau2_median": float(np.median(tau2)),
            "point_mass": dist["point_mass"], "df": float(k - 2.0)}


def _bisect(f, target: float, lo: float, hi: float, *, tol: float = 1e-10
            ) -> float:
    """Root of ``f(x) = target`` on ``[lo, hi]``, where ``f`` is increasing."""
    flo = f(lo) - target
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        fmid = f(mid) - target
        if fmid == 0.0 or (hi - lo) <= tol * (1.0 + abs(mid)):
            return mid
        if (fmid > 0.0) == (flo > 0.0):
            lo, flo = mid, fmid
        else:
            hi = mid
    return 0.5 * (lo + hi)
