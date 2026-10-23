"""Prediction-model performance: pooling discrimination and calibration.

A prognostic-model review does not pool one effect size. It pools at least
four, and they are not interchangeable:

===================  =========================  =====================
metric               scale pooled               where the variance is
===================  =========================  =====================
discrimination (AUC)  logit                      Hanley–McNeil, then the
                                                 delta method
calibration slope     as reported                the study's own
calibration intercept as reported                the study's own
O:E ratio             log                        Poisson: 1/O + 1/E
Brier score           as reported                the study's own
===================  =========================  =====================

Two consequences that the module exists to enforce. First, an AUC cannot go
into the same pool as a calibration slope — different scale, different
interpretation. Second, an AUC's variance is *not* a function of the AUC alone:
it depends on how many events and non-events the study had, which is why
:func:`auc_logit` asks for them.

    >>> yi, vi = ma.auc_logit(auc, n_event=n1, n_nonevent=n0)
    >>> fit = ma.prediction_meta(yi, vi, metric="auc")
    >>> fit.pooled, fit.ci_lb, fit.ci_ub      # back on the AUC scale

The pooling itself is :func:`~pimeta.rma_uni.rma`: a prediction-model
performance meta-analysis is an inverse-variance pool of a transformed metric,
and pretending otherwise would add a model without adding an estimator.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

__all__ = [
    "PredictionResult",
    "auc_logit",
    "oe_ratio",
    "prediction_meta",
]

#: Which metrics are pooled on a log scale and which on their own.
_METRICS = {
    "auc": ("logit", True),
    "oe_ratio": ("log", True),
    "calibration_slope": ("identity", False),
    "calibration_intercept": ("identity", False),
    "brier": ("identity", False),
}


def _logit(p):
    p = np.asarray(p, dtype=float)
    if np.any((p <= 0.0) | (p >= 1.0)):
        raise ValueError("a proportion to be logit-transformed must lie in (0, 1)")
    return np.log(p / (1.0 - p))


def _expit(x):
    x = np.asarray(x, dtype=float)
    out = np.empty_like(x)
    pos = x >= 0
    out[pos] = 1.0 / (1.0 + np.exp(-x[pos]))
    ex = np.exp(x[~pos])
    out[~pos] = ex / (1.0 + ex)
    return out


def _metric(metric: str) -> tuple[str, bool]:
    key = str(metric).lower()
    if key not in _METRICS:
        known = ", ".join(sorted(_METRICS))
        raise ValueError(f"unknown metric {metric!r}; use one of {known}")
    return _METRICS[key]


def auc_logit(auc, *, n_event, n_nonevent):
    """Logit-transform an AUC and give it the Hanley–McNeil variance.

    An AUC is a proportion, so it is pooled on the logit scale — that keeps the
    back-transformed interval inside [0, 1] instead of letting it run past 1.
    Its standard error comes from Hanley & McNeil (1982),

    .. math::

        \\mathrm{Var}(A) = \\frac{A(1-A) + (n_1-1)(Q_1 - A^2)
                                + (n_0-1)(Q_2 - A^2)}{n_1 n_0},

    with :math:`Q_1 = A/(2-A)` and :math:`Q_2 = 2A^2/(1+A)`, mapped to the logit
    scale by the delta method. Note what this means: two studies reporting the
    same AUC have different variances if their event counts differ.

    Args:
        auc: the reported C-statistic(s), strictly inside (0, 1).
        n_event: number of participants with the outcome.
        n_nonevent: number without it.

    Returns:
        ``(yi, vi)`` on the logit scale.

    Raises:
        ValueError: for an AUC outside (0, 1), or non-positive group sizes.

    Example:
        >>> yi, vi = ma.auc_logit(0.75, n_event=100, n_nonevent=400)
    """
    auc = np.asarray(auc, dtype=float).ravel()
    n1 = np.asarray(n_event, dtype=float).ravel()
    n0 = np.asarray(n_nonevent, dtype=float).ravel()
    if not (auc.size == n1.size == n0.size):
        raise ValueError("'auc', 'n_event' and 'n_nonevent' must be the same length")
    if np.any(n1 <= 0.0) or np.any(n0 <= 0.0):
        raise ValueError("group sizes must be positive")
    q1 = auc / (2.0 - auc)
    q2 = 2.0 * auc ** 2 / (1.0 + auc)
    var_auc = (auc * (1.0 - auc)
               + (n1 - 1.0) * (q1 - auc ** 2)
               + (n0 - 1.0) * (q2 - auc ** 2)) / (n1 * n0)
    yi = _logit(auc)
    # delta method: d logit(A) / dA = 1 / (A (1 - A))
    vi = var_auc / (auc * (1.0 - auc)) ** 2
    return yi, vi


def oe_ratio(observed, expected):
    """Log observed-to-expected ratio and its Poisson variance.

    Args:
        observed: observed events.
        expected: expected events under the model.

    Returns:
        ``(yi, vi)`` with ``yi = log(O / E)`` and ``vi = 1/O + 1/E``.

    Raises:
        ValueError: for a non-positive count.

    Example:
        >>> yi, vi = ma.oe_ratio(120, 100)      # log(1.2), 1/120 + 1/100
    """
    o = np.asarray(observed, dtype=float).ravel()
    e = np.asarray(expected, dtype=float).ravel()
    if o.size != e.size:
        raise ValueError("'observed' and 'expected' must be the same length")
    if np.any(o <= 0.0) or np.any(e <= 0.0):
        raise ValueError("counts must be positive; add a correction upstream")
    return np.log(o / e), 1.0 / o + 1.0 / e


@dataclass
class PredictionResult:
    """A pooled performance metric.

    Attributes:
        metric: as passed in.
        yi, vi: the per-study values on the analysis scale.
        b, se, ci_lb, ci_ub: the pooled estimate on the analysis scale.
        pooled, pooled_ci_lb, pooled_ci_ub: the same, back-transformed to the
            scale the studies reported.
        tau2, QE, QE_df, QE_p, I2: heterogeneity.
        pi_lb, pi_ub: prediction interval, back-transformed.
        k: studies pooled.
        method: τ² estimator.
        level: confidence level.
        scale: ``"logit"``, ``"log"`` or ``"identity"``.
        fit: the underlying :class:`~pimeta.rma_uni.RmaResult`.
    """

    metric: str
    yi: list[float]
    vi: list[float]
    b: float
    se: float
    ci_lb: float
    ci_ub: float
    pooled: float
    pooled_ci_lb: float
    pooled_ci_ub: float
    tau2: float
    QE: float
    QE_df: int
    QE_p: float
    I2: float
    pi_lb: float | None
    pi_ub: float | None
    k: int
    method: str
    level: float
    scale: str
    fit: Any = field(repr=False)

    def to_dict(self) -> dict:
        return {
            "metric": self.metric,
            "yi": list(self.yi),
            "vi": list(self.vi),
            "b": self.b,
            "se": self.se,
            "ci_lb": self.ci_lb,
            "ci_ub": self.ci_ub,
            "pooled": self.pooled,
            "pooled_ci_lb": self.pooled_ci_lb,
            "pooled_ci_ub": self.pooled_ci_ub,
            "tau2": self.tau2,
            "QE": self.QE,
            "QE_df": self.QE_df,
            "QE_p": self.QE_p,
            "I2": self.I2,
            "pi_lb": self.pi_lb,
            "pi_ub": self.pi_ub,
            "k": self.k,
            "method": self.method,
            "level": self.level,
            "scale": self.scale,
        }

    def __str__(self) -> str:
        return (f"{self.metric}: {self.pooled:.4f} "
                f"({self.level:.0%} CI {self.pooled_ci_lb:.4f} to "
                f"{self.pooled_ci_ub:.4f}), k = {self.k}, "
                f"I2 = {100 * self.I2:.1f}%")


def prediction_meta(yi, vi, *, metric: str, method: str = "REML",
                    level: float = 0.95, test: str = "z") -> PredictionResult:
    """Pool a prediction-model performance metric on its own scale.

    Args:
        yi: the transformed values — from :func:`auc_logit` or
            :func:`oe_ratio`, or the reported values themselves for a
            calibration slope/intercept or a Brier score.
        vi: their variances.
        metric: ``"auc"``, ``"oe_ratio"``, ``"calibration_slope"``,
            ``"calibration_intercept"`` or ``"brier"``. It decides the
            back-transformation, nothing else.
        method: τ² estimator.
        level: confidence level.
        test: reference distribution.

    Returns:
        :class:`PredictionResult`.

    Raises:
        ValueError: for an unknown metric or mismatched inputs.

    Example:
        >>> yi, vi = ma.auc_logit([0.75, 0.80, 0.72], n_event=[100, 60, 140],
        ...                       n_nonevent=[400, 300, 500])
        >>> ma.prediction_meta(yi, vi, metric="auc").pooled     # doctest: +SKIP
    """
    from .rma_uni import rma

    scale, inverse = _metric(metric)
    yi = np.asarray(yi, dtype=float).ravel()
    vi = np.asarray(vi, dtype=float).ravel()
    if yi.size != vi.size:
        raise ValueError("'yi' and 'vi' must be the same length")
    if yi.size == 0:
        raise ValueError("nothing to pool")

    model = rma(yi, vi=vi, method=method, level=level, test=test)
    b = float(model.b[0])
    lo = float(model.ci_lb[0])
    hi = float(model.ci_ub[0])
    pi_lb = getattr(model, "pi_lb", None)
    pi_ub = getattr(model, "pi_ub", None)

    def back(value):
        if value is None:
            return None
        if scale == "logit":
            return float(_expit(np.asarray([value]))[0])
        if scale == "log":
            return float(np.exp(value))
        return float(value)

    return PredictionResult(
        metric=str(metric).lower(),
        yi=[float(v) for v in yi],
        vi=[float(v) for v in vi],
        b=b,
        se=float(model.se[0]),
        ci_lb=lo,
        ci_ub=hi,
        pooled=back(b),
        pooled_ci_lb=back(lo),
        pooled_ci_ub=back(hi),
        tau2=float(np.asarray(model.tau2, dtype=float).ravel()[0]),
        QE=float(model.QE),
        QE_df=int(model.k - model.p),
        QE_p=float(getattr(model, "QEp", float("nan"))),
        I2=float(getattr(model, "I2", float("nan"))),
        pi_lb=back(pi_lb),
        pi_ub=back(pi_ub),
        k=int(model.k),
        method=str(model.method),
        level=float(level),
        scale=scale,
        fit=model,
    )
