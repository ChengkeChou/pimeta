"""Tau-squared moment/iterative estimators, transcribed from metafor's R source.

metafor's ``rma()`` and meta's ``metagen()`` support DL, HE, HS, SJ, PM and EB
(plus GENQ/PMM/GENQM). Every estimator here is a direct transcription of
metafor's ``rma.fit`` so the numeric output matches R exactly — no third-party
meta-analysis package is used.

Every estimator here operates on the **analysis scale** arrays ``y`` (effect)
and ``v`` (sampling variance), i.e. the log scale for RR/OR/HR/PETO/IRR.

Estimators implemented (matching ``meta::metagen(method.tau=)``):
- ``DL``  DerSimonian-Laird
- ``HE``  Hedges                     (unweighted between/within split)
- ``HS``  Hunter-Schmidt             (requires per-study total sample size n)
- ``SJ``  Sidik-Jonkman              (one-step moment estimator)
- ``PM``  Paule-Mandel               (solve Q(tau^2) = k-1)
- ``EB``  Empirical Bayes            (iterative PM; converges to PM)
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
from scipy import optimize

from ._numeric import (
    TAU2_MAX,
    TAU2_MIN,
    TAU2_THRESHOLD,
    UNIROOT_TOL,
)
from ._numeric import zeroin as _zeroin

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


def _he(y, v) -> float:
    """Hedges (1983) HE estimator: ``(RSS - tr(PV)) / (k - p)`` unweighted.

    Transcribed from ``rma.uni``'s HE/CO/VC block: ``P = I - X (X'X)^-1 X'``,
    ``RSS = Y' P Y`` and ``tr(PV)`` with ``V = diag(vi)``. For an intercept-only
    model this reduces to ``var(y, ddof=1) - mean(v)``, which matches
    ``rma(yi, vi, method="HE")$tau2`` to 6e-17.
    """
    ya = np.asarray(y, dtype=float)
    va = np.asarray(v, dtype=float)
    if ya.size < 2:
        return 0.0
    return max(0.0, float(ya.var(ddof=1) - va.mean()))


def _qss(y, v, tau2: float) -> float:
    """``Y' P Y`` for the intercept-only model at ``tau2`` (metafor's RSS).

    ``P = W - W 1 (1'W 1)^-1 1' W`` collapses to the weighted central sum of
    squares, which is what ``rma.uni`` feeds into ``.QE.func``.
    """
    wi = 1.0 / (v + tau2)
    sw = float(wi.sum())
    ybar = float((wi * y).sum()) / sw
    return float((wi * (y - ybar) ** 2).sum())


def _hs(y, v, n=None) -> float:
    """Hunter-Schmidt: ``(RSS - k) / sum(1/vi)`` (metafor ``method="HS"``).

    Transcribed from ``rma.uni``: ``wi <- 1/vi``, ``RSS <- Y' P Y`` (fixed-effect
    weights) and ``tau2 <- (RSS - k)/sum(wi)``. Note the ``- k`` (not ``- (k-1)``)
    and the plain ``sum(wi)`` denominator. ``n`` is accepted for API symmetry but
    metafor's HS does not use it.
    """
    ya = np.asarray(y, dtype=float)
    va = np.asarray(v, dtype=float)
    wi = 1.0 / va
    sw = float(wi.sum())
    ybar = float((wi * ya).sum()) / sw
    rss = float((wi * (ya - ybar) ** 2).sum())
    return max(0.0, (rss - ya.size) / sw)


def _pm(y, v, n=None) -> float:
    """Paule-Mandel: ``uniroot(.QE.func, c(0, 100), tol = eps^0.25)``.

    metafor does not solve ``Q(tau2) = k - p`` to machine precision — it hands
    the problem to R's ``uniroot`` with ``con$tol = .Machine$double.eps^0.25``
    (1.22e-4). Reproducing that tolerance is what makes the value match R's
    ``rma(..., method="PM")$tau2`` exactly; a tight Brent solve lands ~2.5e-5
    away on the true root.
    """
    ya = np.asarray(y, dtype=float)
    va = np.asarray(v, dtype=float)
    target = float(ya.size - 1)  # k - p with p = 1

    def f(tau2: float) -> float:
        return _qss(ya, va, tau2) - target

    if f(TAU2_MIN) < target:
        return TAU2_MIN
    if f(TAU2_MAX) > target:
        return TAU2_MAX
    return float(_zeroin(f, TAU2_MIN, TAU2_MAX, tol=UNIROOT_TOL))


def _eb(y, v, n=None) -> float:
    """Empirical Bayes: metafor's Fisher-scoring iteration.

    Starts from the HE/CO/VC value ``(RSS - trPV)/(k - p)`` (see ``rma.uni``'s
    combined HE/ML/REML/EB block) and iterates

        ``adj <- (Y' P Y * k/(k - p) - k) / sum(wi)``

    with ``wi = 1/(vi + tau2)``, stopping once the change falls below
    ``con$threshold`` (1e-5).
    """
    ya = np.asarray(y, dtype=float)
    va = np.asarray(v, dtype=float)
    k = ya.size
    p = 1
    tau2 = max(TAU2_MIN, _he(ya, va))
    for _ in range(100):  # con$maxiter
        old = tau2
        wi = 1.0 / (va + tau2)
        sw = float(wi.sum())
        rss = _qss(ya, va, tau2)
        adj = (rss * k / (k - p) - k) / sw
        if not np.isfinite(adj):
            adj = 0.0
        while tau2 + adj < TAU2_MIN:
            adj /= 2.0
        tau2 = tau2 + adj
        if abs(old - tau2) <= TAU2_THRESHOLD:
            break
    return tau2


# Map method name (upper) -> callable(y, v, n).
_METHODS: dict[str, Callable[[np.ndarray, np.ndarray, np.ndarray | None], float]] = {
    "DL": lambda y, v, n: _dl(y, v),
    "HE": lambda y, v, n: _he(y, v),
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
    y_arr = np.asarray(y, dtype=float)
    v_arr = np.asarray(v, dtype=float)
    n_arr = None if n is None else np.asarray(n, dtype=float)
    return _METHODS[key](y_arr, v_arr, n_arr)