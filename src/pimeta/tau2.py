"""Tau-squared moment/iterative estimators, transcribed from metafor's R source.

metafor's ``rma()`` and meta's ``metagen()`` support DL, HE, HS, SJ, PM and EB
(plus GENQ/PMM/GENQM). Every estimator here is a direct transcription of
metafor's ``rma.fit`` so the numeric output matches R exactly — no third-party
meta-analysis package is used.

Every estimator here operates on the **analysis scale** arrays ``y`` (effect)
and ``v`` (sampling variance), i.e. the log scale for RR/OR/HR/PETO/IRR.

Estimators implemented (matching ``meta::metagen(method.tau=)``):
- ``DL``  DerSimonian-Laird
- ``HE``  Hedges                     == DL when there are no moderators
- ``HS``  Hunter-Schmidt             (requires per-study total sample size n)
- ``SJ``  Sidik-Jonkman              (one-step moment estimator)
- ``PM``  Paule-Mandel               (solve Q(tau^2) = k-1)
- ``EB``  Empirical Bayes            (iterative PM; converges to PM)
"""

from __future__ import annotations

import numpy as np
from scipy import optimize

# Maximum tau^2 for the root search / iteration. tau^2 > 100 is biologically
# meaningless for log-ratio scales, and bounds the bracketing loop.
_TAU2_MAX = 1000.0


def _weighted_q(y: list[float] | np.ndarray, w: list[float] | np.ndarray) -> float:
    """Cochran-style Q for a *fixed* weight vector (weights supplied directly)."""
    sw = sum(w)
    mu = sum(wi * yi for wi, yi in zip(w, y)) / sw
    return sum(wi * (yi - mu) ** 2 for wi, yi in zip(w, y))


def _q_at(y: list[float] | np.ndarray, v: list[float] | np.ndarray, tau2: float) -> float:
    """Q(tau^2) = sum w_i * (y_i - mu_hat)^2 with w_i = 1/(v_i + tau^2)."""
    w = [1.0 / (vi + tau2) for vi in v]
    return _weighted_q(y, w)


def _dl(y: list[float] | np.ndarray, v: list[float] | np.ndarray) -> float:
    """DerSimonian-Laird moment estimator."""
    k = len(y)
    w = [1.0 / vi for vi in v]
    sw = sum(w)
    q = _weighted_q(y, w)
    c = sw - sum(wi * wi for wi in w) / sw
    return max(0.0, (q - (k - 1)) / c)


def _hs(y: list[float] | np.ndarray, v: list[float] | np.ndarray, n: list[float]) -> float:
    """Hunter-Schmidt estimator (sample-size weights)."""
    k = len(y)
    if not n or any(ni is None or ni <= 0 for ni in n):
        raise ValueError("HS requires per-study total sample size n (all > 0)")
    w = [float(ni) for ni in n]
    q = _weighted_q(y, w)
    return max(0.0, (q - (k - 1)) / sum(w))


def _sj(y: list[float] | np.ndarray, v: list[float] | np.ndarray) -> float:
    """Sidik-Jonkman (2005) one-step estimator."""
    k = len(y)
    ybar = sum(y) / k
    tau2 = sum((yi - ybar) ** 2 for yi in y) / k
    if tau2 == 0.0:
        return 0.0
    r = [vi / tau2 for vi in v]
    w = [1.0 / (1.0 + ri) for ri in r]
    sw = sum(w)
    mu = sum(wi * yi for wi, yi in zip(w, y)) / sw
    return max(0.0, sum(wi * (yi - mu) ** 2 for wi, yi in zip(w, y)) / (k - 1))


def _pm(y: list[float] | np.ndarray, v: list[float] | np.ndarray) -> float:
    """Paule-Mandel estimator: solve Q(tau^2) = k-1 by bisection."""
    k = len(y)

    def f(tau2: float) -> float:
        return _q_at(y, v, tau2) - (k - 1)

    if f(0.0) <= 0.0:
        return 0.0
    # Q is strictly decreasing in tau^2, so bracket the root.
    hi = 1.0
    while f(hi) > 0.0 and hi < _TAU2_MAX:
        hi *= 10.0
    if f(hi) > 0.0:
        return hi
    return float(optimize.brentq(f, 0.0, hi))


def _eb(y: list[float] | np.ndarray, v: list[float] | np.ndarray) -> float:
    """Empirical Bayes estimator: iterative Paule-Mandel.

    metafor's EB converges to the same tau^2 as PM, so we return the PM root.
    """
    return _pm(y, v)


# Map method name (upper) -> callable(y, v, n).
_METHODS = {
    "DL": lambda y, v, n: _dl(y, v),
    "HE": lambda y, v, n: _dl(y, v),  # HE == DL without moderators
    "HS": _hs,
    "SJ": lambda y, v, n: _sj(y, v),
    "PM": lambda y, v, n: _pm(y, v),
    "EB": lambda y, v, n: _eb(y, v),
}

# Likelihood-based methods (handled by metafor's Fisher-scoring loop in
# ``pooling._metafor_fisher_scoring_tau2``, not as moment estimators here).
LIKELIHOOD_METHODS = {"ML", "REML"}

SUPPORTED_TAU2_METHODS = frozenset(_METHODS) | LIKELIHOOD_METHODS | {"FE"}


def estimate_tau2(
    y: list[float] | np.ndarray,
    v: list[float] | np.ndarray,
    method: str,
    n: list[float] | np.ndarray | None = None,
) -> float:
    """Return tau-squared for a moment/iterative estimator.

    ``method`` is case-insensitive and must be one of DL/HE/HS/SJ/PM/EB (the
    likelihood-based ML/REML are handled by metafor's Fisher-scoring loop in
    ``pooling``). ``n`` is the per-study total sample size, required only for
    ``HS``.
    """
    key = method.upper()
    if key not in _METHODS:
        raise ValueError(
            f"unsupported tau2 method: {method!r} "
            f"(expected one of {sorted(_METHODS)})"
        )
    return _METHODS[key](y, v, n)
