"""Deterministic meta-analysis pooling, replicating R's ``metafor`` + ``meta``.

Self-contained NumPy implementation — no third-party meta-analysis package
(pymare/statsmodels) is used. Every estimator is transcribed directly from the
R source so the numbers match R to full double precision:

- ``metafor::rma(yi, vi, method="DL"|"REML"|"ML"|...)``
- ``meta::metagen(TE, seTE, method.tau="DL"|"REML"|...)``

The DerSimonian-Laird (DL) moment estimator lives in :mod:`.tau2`; the ML/REML
tau-squared comes from metafor's exact Fisher-scoring loop
(``_metafor_fisher_scoring_tau2``); the pooled estimate is plain
inverse-variance weighting. No LLM, no hallucination — pure linear algebra.

Effect scale:
- Ratio types (RR, OR, HR) are pooled on the **log** scale and back-transformed
  with ``exp``, exactly like R's ``measure="RR"/"OR"/"HR"``.
- Difference types (MD, SMD, WMD, RD) are pooled on the raw scale.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass, field

import numpy as np
from numpy.typing import ArrayLike
from scipy import integrate, stats

from ._numeric import zeroin as _zeroin
from ._projection import (
    quad_form,
    quad_form_squared,
    trace_projection,
    trace_projection_squared,
)
from .tau2 import estimate_tau2

# 95% CI z-quantile, equivalent to R's qnorm(0.975).
Z_975 = stats.norm.ppf(0.975)  # 1.95996398454...

RATIO_TYPES = {"RR", "OR", "HR", "PETO", "IRR", "PLN", "PLO"}  # log scale
DIFF_TYPES = {"MD", "SMD", "WMD", "RD", "AS", "PAS", "PFT"}    # raw scale
SUPPORTED_TYPES = RATIO_TYPES | DIFF_TYPES

# metafor::rma and meta::metagen both use a z-test (normal reference) for the
# CI and p-value by default; prediction intervals use a t reference.


@dataclass
class StudyEffect:
    """One study's effect estimate + dispersion (SE/variance/CI)."""

    effect_size: float
    ci_lower: float | None = None
    ci_upper: float | None = None
    standard_error: float | None = None
    variance: float | None = None
    n: float | None = None      # total sample size (for HS tau2 estimator)
    study_id: str = ""


@dataclass
class PooledResult:
    """Pooled estimate + heterogeneity diagnostics (R-identical numbers)."""

    model: str                     # "fixed" | "random"
    method: str                    # "DL" | "REML" | "" (fixed has no tau2 method)
    k: int
    effect_type: str
    # raw-scale pooled estimate + CI (back-transformed for ratio types)
    pooled_effect: float
    pooled_ci_lower: float
    pooled_ci_upper: float
    # analysis-scale (log scale for ratio types) — metafor reports `se` here
    pooled_est: float
    pooled_se: float
    z: float
    p_value: float
    # heterogeneity
    q: float
    q_df: int
    q_p: float
    tau2: float
    i2: float
    h2: float
    # per-study analysis-scale weights
    weights: list[float] = field(default_factory=list)
    # prediction interval (random-effects only)
    pi_lower: float | None = None
    pi_upper: float | None = None


def _is_ratio(effect_type: str) -> bool:
    return effect_type in RATIO_TYPES


def _prepare(
    effects: list[StudyEffect], effect_type: str
) -> tuple[list[float], list[float], list[float]]:
    """Convert raw effects to analysis-scale ``(y, v, n)`` arrays."""
    if not effects:
        raise ValueError("no studies supplied")
    if effect_type not in SUPPORTED_TYPES:
        raise ValueError(f"unsupported effect_type: {effect_type}")
    ratio = _is_ratio(effect_type)

    y_list: list[float] = []
    v_list: list[float] = []
    n_list: list[float] = []
    for e in effects:
        y = math.log(e.effect_size) if ratio else e.effect_size
        n_list.append(e.n if e.n is not None else math.nan)

        if e.variance is not None:
            v = e.variance  # assumed already on the analysis scale
        elif e.standard_error is not None:
            v = e.standard_error ** 2
        elif e.ci_lower is not None and e.ci_upper is not None:
            if ratio:
                se = (math.log(e.ci_upper) - math.log(e.ci_lower)) / (2 * Z_975)
            else:
                se = (e.ci_upper - e.ci_lower) / (2 * Z_975)
            v = se ** 2
        else:
            raise ValueError(
                "each study needs variance, standard_error, or a CI "
                f"(study '{e.study_id}' has none)"
            )
        y_list.append(y)
        v_list.append(v)
    return y_list, v_list, n_list


def _reml_ml_loglik(
    y: np.ndarray, v: np.ndarray, X: np.ndarray, tau2: float, method: str
) -> float:
    """(Restricted) log-likelihood at ``tau2``, using metafor's convention.

    ``y`` is centered by its global mean first (metafor computes the RSS on
    ``scale(yi, center=TRUE, scale=FALSE)``), so the result reproduces
    metafor's ``ll0``/``ll`` values used in its ``ll0check`` step exactly.
    """
    k = y.shape[0]
    p = X.shape[1]
    ymc = y - y.mean()
    wi = 1.0 / (v + tau2)
    XWX = (X.T * wi) @ X
    beta = np.linalg.solve(XWX, (X.T * wi) @ ymc)
    rss = float(np.sum(wi * (ymc - X @ beta) ** 2))
    ll = -0.5 * float(np.sum(np.log(v + tau2))) - 0.5 * rss
    if method == "ML":
        ll += -0.5 * k * math.log(2 * math.pi)
    else:  # REML
        _, logdet = np.linalg.slogdet(XWX)
        ll += -0.5 * (k - p) * math.log(2 * math.pi) - 0.5 * float(logdet)
    return float(ll)


def _metafor_fisher_scoring_tau2(
    y: ArrayLike, v: ArrayLike, X: np.ndarray, method: str
) -> float:
    """ML/REML tau^2 via metafor's exact Fisher-scoring iteration.

    A generic profile-likelihood optimizer (e.g. ``scipy.optimize.minimize_scalar``)
    converges to the true optimum, but
    ``metafor::rma`` *stops* its Fisher scoring once ``change < 1e-5``, which
    leaves the estimate a few 1e-8 away from the exact optimum. To reproduce R
    to full double precision we re-implement that exact loop: Hedges start
    value, global-mean centering, the ``P = W - W X (X'W X)^-1 X'W`` projection
    updated each step, and step-halving at the tau^2 >= 0 boundary.
    """
    y = np.asarray(y, dtype=float).ravel()
    v = np.asarray(v, dtype=float).ravel()
    X = np.asarray(X, dtype=float)
    k = y.shape[0]
    p = X.shape[1]

    # metafor centers yi by its GLOBAL mean: scale(yi, center=TRUE, scale=FALSE)
    ymc = y - y.mean()

    # Hedges (HE) start value (also metafor's ML/REML/EB initial tau^2):
    #   P = I - X (X'X)^-1 X';  tau2 = (ymc' P ymc - tr(P V)) / (k - p)
    # The projection is never formed: from ._projection gives tr(P V) as the
    # p x p expression, which is what keeps a k of 20000 from asking for 3 GiB.
    stXX = np.linalg.inv(X.T @ X)
    rss = quad_form(X, np.ones(k), ymc)
    tr_pv = float(np.sum(v) - np.trace(stXX @ (X.T @ (v[:, None] * X))))
    tau2 = max(0.0, (rss - tr_pv) / (k - p))

    threshold = 1e-5  # metafor's con$threshold
    for _ in range(100):  # metafor's con$maxiter
        wi = 1.0 / (v + tau2)
        yppy = quad_form_squared(X, wi, ymc)
        if method == "ML":
            adj = (yppy - float(wi.sum())) / float(np.sum(wi ** 2))
        else:  # REML
            adj = (yppy - trace_projection(X, wi)) / trace_projection_squared(X, wi)

        while tau2 + adj < 0.0:  # step-halving at the tau^2 >= 0 boundary
            adj /= 2.0

        tau2_new = tau2 + adj
        change = abs(tau2_new - tau2)
        tau2 = tau2_new
        if change < threshold:
            break

    # metafor's ll0check: if ll(tau^2 = 0) is larger, set tau^2 = 0.
    if bool(np.all(v > 0)) and tau2 > threshold:
        ll0 = _reml_ml_loglik(y, v, X, 0.0, method)
        ll = _reml_ml_loglik(y, v, X, tau2, method)
        tol = float(np.finfo(np.float64).eps) ** 0.25
        if ll0 - ll > tol:
            tau2 = 0.0

    return float(tau2)


def _resolve_tau2(
    y: ArrayLike,
    v: ArrayLike,
    model: str,
    method: str,
    n: list[float],
) -> tuple[float, str]:
    """Resolve tau-squared for the requested model/method.

    Returns ``(tau2, used_method)``. ML/REML come from metafor's Fisher-scoring
    loop, DL/HE/HS/SJ/PM/EB from :mod:`.tau2` (all transcribed from R source).
    """
    if model == "fixed":
        return 0.0, ""

    method = method.upper()
    ya = np.asarray(y, dtype=float).ravel()
    va = np.asarray(v, dtype=float).ravel()

    if method in ("ML", "REML"):
        X = np.ones((ya.size, 1))
        return _metafor_fisher_scoring_tau2(ya, va, X, method), method

    if method == "DL":
        return estimate_tau2(ya, va, "DL", None), "DL"

    if method in ("HE", "HS", "SJ", "PM", "EB"):
        n_list = [ni for ni in n if ni == ni and ni > 0] or None
        return estimate_tau2(ya, va, method, n_list), method

    raise ValueError(
        f"unsupported tau2 method: {method!r} "
        "(expected DL/HE/HS/SJ/ML/REML/PM/EB)"
    )


def _fit_wls(y: ArrayLike, v: ArrayLike, tau2: float, alpha: float) -> dict:
    """Inverse-variance pooling at a fixed tau^2 (pure NumPy).

    ``est``, ``se``, ``ci``, ``z`` and ``p`` are on the analysis scale. Q and
    its p-value come from Cochran's fixed-effect Q (weights 1/vi, independent
    of tau^2), matching metafor::rma exactly. The CI and p-value use the Wald
    (z) reference, which is both metafor's and meta's default.
    """
    y = np.asarray(y, dtype=float).ravel()
    v = np.asarray(v, dtype=float).ravel()
    k = y.shape[0]

    # Random/fixed-effect weights at the supplied tau^2.
    wi = 1.0 / (v + tau2)
    sw = float(wi.sum())
    est = float(np.sum(wi * y) / sw)
    se = float(math.sqrt(1.0 / sw))
    z = est / se
    z_crit = stats.norm.ppf(1.0 - alpha / 2.0)
    p = 2.0 * stats.norm.sf(abs(z))

    # Cochran's Q (fixed-effect weights 1/vi, independent of tau^2).
    wi_fe = 1.0 / v
    sw_fe = float(wi_fe.sum())
    mu_fe = float(np.sum(wi_fe * y) / sw_fe)
    q = float(np.sum(wi_fe * (y - mu_fe) ** 2))
    q_p = float(stats.chi2.sf(q, k - 1))

    return {
        "est": est,
        "se": se,
        "ci_lower": est - z_crit * se,
        "ci_upper": est + z_crit * se,
        "z": z,
        "p": p,
        "q": q,
        "q_p": q_p,
    }


def _i2_h2(v: ArrayLike, tau2: float, q: float, model: str) -> tuple[float, float]:
    """metafor's I^2 and H^2 for an intercept-only model (p = 1).

    metafor computes the 'typical' sampling variance from the FIXED-EFFECT
    weights (wi = 1/vi, *not* 1/(vi + tau2))::

        trP = sum(1/vi) - sum(1/vi^2) / sum(1/vi)
        vt  = (k - 1) / trP

    and then, for random-effects models::

        H2 = tau2/vt + 1,   I2 = 100 * tau2/(vt + tau2)

    while fixed-effect models use the Q-based forms::

        H2 = Q/(k - 1),     I2 = max(0, 100*(Q - (k - 1))/Q)
    """
    va = np.asarray(v, dtype=float)
    k = va.size
    if k <= 1:
        return 0.0, 1.0
    wi = 1.0 / va
    tr_p = float(wi.sum() - (wi ** 2).sum() / wi.sum())
    vt = (k - 1) / tr_p
    if model == "fixed":
        h2 = q / (k - 1)
        i2 = max(0.0, 100.0 * (q - (k - 1)) / q) if q > 0 else 0.0
    else:
        h2 = tau2 / vt + 1.0
        i2 = 100.0 * tau2 / (vt + tau2)
    return i2, h2


def _back_transform(value: float, effect_type: str) -> float:
    return math.exp(value) if _is_ratio(effect_type) else value


# ---------------------------------------------------------------------------
# confint() — CI for tau^2 (metafor::confint.rma.uni)
# ---------------------------------------------------------------------------
#
# metafor reports the CI for the heterogeneity components (tau^2, tau, I^2,
# H^2) using one of several methods. The default is the Q-profile method
# ("QP"): the generalized Q statistic QE(tau^2) = y' P y (P projects out the
# intercept under inverse-variance weights 1/(vi + tau^2)) is monotone
# decreasing in tau^2, so the CI bounds are the roots of
#
#     QE(tau^2) = qchisq(level/2, k - p)     (upper critical -> lower bound)
#     QE(tau^2) = qchisq(1 - level/2, k - p) (lower critical -> upper bound)
#
# "PL" is the profile-(restricted)-likelihood interval: roots of
# -2*(ll(tau^2) - ll(tau^2_hat)) = qchisq(1 - level, 1).
# "HT" is Higgins & Thompson's analytic interval on the log-H scale.


def _qe_func(tau2: float, y: np.ndarray, v: np.ndarray, objective: float = 0.0,
             X: np.ndarray | None = None) -> float:
    """metafor's ``.QE.func``.

    Returns ``RSS - objective`` where ``RSS = y' P y`` with
    ``P = W - W X (X'WX)^-1 X' W`` and ``W = diag(1/(vi + tau2))``.  Without
    ``X`` this is the intercept-only form
    ``sum(wi yi^2) - (sum(wi yi))^2 / sum(wi)``.
    """
    wi = 1.0 / (v + tau2)
    if X is None:
        sw = float(wi.sum())
        sy = float(np.sum(wi * y))
        rss = float(np.sum(wi * y * y)) - sy * sy / sw
        return rss - objective
    WX = wi[:, None] * X
    beta = np.linalg.solve(X.T @ WX, WX.T @ y)
    rss = float(y @ (wi * y - WX @ beta))
    return rss - objective


def _profile_tau2_func(
    tau2: float, y: np.ndarray, v: np.ndarray, X: np.ndarray,
    method: str, ll_opt: float, objective: float,
) -> float:
    """metafor's ``.profile.rma.uni`` with ``confint=TRUE``.

    Returns ``-2 * (logLik(tau2) - logLik(tau2_hat)) - objective``.
    """
    ll = _reml_ml_loglik(y, v, X, tau2, method)
    return -2.0 * (ll - ll_opt) - objective


def _resolve_alpha(level: float) -> float:
    """Normalise a confidence specification to a tail probability.

    This module historically took ``level`` to be the *alpha* (``0.05`` for a
    95% interval) while the rest of the package (and metafor) use the
    confidence level (``95``).  Passing ``0.95`` therefore silently produced a
    95%-alpha interval instead of a 95%-confidence one.  Both spellings are
    accepted now, disambiguated the only sensible way: anything ``>= 0.5`` is a
    confidence level (``95`` or ``0.95``), anything below is an alpha.
    """
    level = float(level)
    if np.isnan(level) or not (0.0 < level < 100.0):
        raise ValueError(
            "argument 'level' must be a confidence level (95 or 0.95) or a "
            "tail probability such as 0.05")
    if level >= 1.0:
        return 1.0 - level / 100.0
    if level > 0.5:
        return 1.0 - level
    return level


@dataclass
class Tau2Confint:
    """CI for tau^2 / tau / I^2 / H^2 (mirrors ``metafor::confint``)."""

    type: str                      # "QP" | "PL" | "HT"
    tau2: float                    # estimate
    tau2_lb: float
    tau2_ub: float
    tau_lb: float | None           # sqrt of tau2 (NA if tau2 bound < 0)
    tau_ub: float | None
    i2: float
    i2_lb: float
    i2_ub: float
    h2: float
    h2_lb: float
    h2_ub: float
    lb_sign: str = ""              # "<" / ">" when a bound hits the search edge
    ub_sign: str = ""
    lb_conv: bool = False
    ub_conv: bool = False


def confint_tau2(
    y: ArrayLike,
    v: ArrayLike,
    tau2: float,
    method: str = "REML",
    level: float = 0.05,
    type: str = "QP",
    X: np.ndarray | None = None,
) -> Tau2Confint:
    """CI for tau^2 (transcribed from ``metafor::confint.rma.uni``).

    Args:
        y: analysis-scale effects.
        v: analysis-scale sampling variances.
        tau2: the estimated tau^2 (from the fitted model).
        method: "ML" or "REML" (used only by the "PL" profile-likelihood
            interval, which refits at fixed tau^2 with the same method).
        level: **alpha** level (0.05 -> 95% CI), matching metafor's internal
            ``x$level`` convention.
        type: "QP" (Q-profile, default), "PL" (profile-likelihood), or "HT"
            (Higgins-Thompson).
        X: the fitted design matrix (``k x p``, intercept included).  Omitted
            for an intercept-only model.  metafor refuses ``type="HT"`` when
            the model has moderators, and so do we.
    """
    y = np.asarray(y, dtype=float).ravel()
    v = np.asarray(v, dtype=float).ravel()
    k = y.shape[0]
    X = None if X is None else np.asarray(X, dtype=float)
    p = 1 if X is None else X.shape[1]
    type = type.upper()
    if type not in ("QP", "PL", "HT"):
        raise ValueError(f"unsupported confint type: {type!r}")
    level = _resolve_alpha(level)

    # Typical sampling variance (fixed-effect weights 1/vi) + I^2/H^2.
    # metafor: vt <- (k - p) / .tr(P)  with  P = W.FE - W.FE X (X'W.FE X)^-1 X' W.FE
    wi_fe = 1.0 / v
    if X is None:
        tr_p = float(wi_fe.sum() - (wi_fe ** 2).sum() / wi_fe.sum())
    else:
        WX = wi_fe[:, None] * X
        tr_p = float(wi_fe.sum() - np.trace(
            np.linalg.solve(X.T @ WX, X.T @ ((wi_fe ** 2)[:, None] * X))))
    vt = (k - p) / tr_p
    i2 = 100.0 * tau2 / (vt + tau2)
    h2 = tau2 / vt + 1.0
    # Cochran Q at tau^2 = 0 (QE.func with objective=0).
    q = _qe_func(0.0, y, v, 0.0, X=X)

    tau2_min = min(0.0, tau2)
    tau2_max = max(100.0, tau2 * 10.0, tau2_min * 10.0)

    tau2_lb = tau2_ub = math.nan
    lb_sign = ub_sign = ""
    lb_conv = ub_conv = False

    if type == "QP":
        crit_u = float(stats.chi2.ppf(1.0 - level / 2.0, k - p))
        crit_l = float(stats.chi2.ppf(level / 2.0, k - p))
        qe_max = _qe_func(tau2_max, y, v, 0.0, X=X)
        try:
            qe_min = _qe_func(tau2_min, y, v, 0.0, X=X)
        except Exception:
            qe_min = math.nan

        if qe_min < crit_l:
            tau2_lb = tau2_ub = tau2_min
            lb_sign = ub_sign = "<"
            lb_conv = ub_conv = True
        else:
            if qe_max > crit_l:
                tau2_ub = tau2_max
                ub_sign = ">"
                ub_conv = True
            else:
                tau2_ub = float(_zeroin(
                    lambda t: _qe_func(t, y, v, crit_l, X=X),
                    tau2_min, tau2_max))
                ub_conv = True

        if qe_max > crit_u:
            tau2_lb = tau2_ub = tau2_max
            lb_sign = ub_sign = ">"
            lb_conv = ub_conv = True
        else:
            if qe_min < crit_u:
                tau2_lb = tau2_min
                lb_conv = True
                if tau2_min > 0:
                    lb_sign = "<"
            else:
                tau2_lb = float(_zeroin(
                    lambda t: _qe_func(t, y, v, crit_u, X=X),
                    tau2_min, tau2_max))
                lb_conv = True

    elif type == "PL":
        objective = float(stats.chi2.ppf(1.0 - level, 1.0))
        X = np.ones((k, 1)) if X is None else X
        ll_opt = _reml_ml_loglik(y, v, X, tau2, method)

        try:
            f_min = _profile_tau2_func(tau2_min, y, v, X, method, ll_opt, objective)
        except Exception:
            f_min = math.nan
        try:
            f_max = _profile_tau2_func(tau2_max, y, v, X, method, ll_opt, objective)
        except Exception:
            f_max = math.nan

        if not math.isnan(f_min):
            if f_min < 0:
                tau2_lb = tau2_min
                lb_conv = True
                if tau2_min > 0:
                    lb_sign = "<"
            else:
                tau2_lb = float(_zeroin(
                    lambda t: _profile_tau2_func(t, y, v, X, method, ll_opt, objective),
                    tau2_min, tau2))
                lb_conv = True

        if not math.isnan(f_max):
            if f_max < 0:
                tau2_ub = tau2_max
                ub_conv = True
                ub_sign = ">"
            else:
                tau2_ub = float(_zeroin(
                    lambda t: _profile_tau2_func(t, y, v, X, method, ll_opt, objective),
                    tau2, tau2_max))
                ub_conv = True

    elif type == "HT":
        if p > 1:
            # metafor: "Method only applicable to models without moderators."
            raise ValueError(
                "HT method is only applicable to models without moderators"
            )
        if k <= 2:
            raise ValueError("HT method requires k > 2")
        if q > k:
            se_lnH = 0.5 * (math.log(q) - math.log(k - 1)) / (
                math.sqrt(2.0 * q) - math.sqrt(2.0 * k - 3.0))
        else:
            se_lnH = math.sqrt(1.0 / (2.0 * (k - 2)) * (1.0 - 1.0 / (3.0 * (k - 2) ** 2)))
        crit = float(stats.norm.ppf(1.0 - level / 2.0))
        h2_lb = math.exp(math.log(h2) - crit * 2.0 * se_lnH)
        h2_ub = math.exp(math.log(h2) + crit * 2.0 * se_lnH)
        i2_lb = (h2_lb - 1.0) / h2_lb
        i2_ub = (h2_ub - 1.0) / h2_ub
        tau2_lb = max(0.0, i2_lb * vt / (1.0 - i2_lb))
        tau2_ub = i2_ub * vt / (1.0 - i2_ub)
        lb_conv = ub_conv = True

    # Convert tau^2 bounds back to I^2 / H^2 bounds (all methods).
    i2_lb = 100.0 * tau2_lb / (vt + tau2_lb)
    i2_ub = 100.0 * tau2_ub / (vt + tau2_ub)
    h2_lb = tau2_lb / vt + 1.0
    h2_ub = tau2_ub / vt + 1.0

    return Tau2Confint(
        type=type,
        tau2=tau2,
        tau2_lb=tau2_lb,
        tau2_ub=tau2_ub,
        tau_lb=math.sqrt(tau2_lb) if tau2_lb >= 0 else None,
        tau_ub=math.sqrt(tau2_ub) if tau2_ub >= 0 else None,
        i2=i2,
        i2_lb=i2_lb,
        i2_ub=i2_ub,
        h2=h2,
        h2_lb=h2_lb,
        h2_ub=h2_ub,
        lb_sign=lb_sign,
        ub_sign=ub_sign,
        lb_conv=lb_conv,
        ub_conv=ub_conv,
    )


@dataclass
class PredictResult:
    """Predicted value + CI/PI (mirrors ``metafor::predict.rma``)."""

    pred: float                    # predicted (analysis scale)
    se: float                      # SE of the prediction
    ci_lb: float
    ci_ub: float
    pi_lb: float | None            # prediction interval (None if fixed/tau2=0)
    pi_ub: float | None
    pi_dist: str                   # "norm" | "t"
    ddf: float                     # df for the CI
    pi_ddf: float | None           # df for the PI
    level: float                   # alpha level


def predict_rma(
    est: float,
    se: float,
    tau2: float,
    k: int,
    p: int = 1,
    parms: int | None = None,
    level: float = 0.05,
    predtype: str = "default",
    test: str = "z",
    newvi: float = 0.0,
    transf: Callable[..., float] | None = None,
    targs=None,
) -> PredictResult:
    """Prediction (+ CI/PI) for an intercept-only model (``metafor::predict.rma``).

    Replicates ``predict.rma.uni`` for the ``k.new = 1`` (intercept-only) case.
    ``est``/``se``/``tau2`` are on the analysis scale (log scale for ratio
    types); ``level`` is the alpha level (0.05 -> 95%).

    ``predtype`` controls the prediction interval reference distribution, exactly
    as in metafor:

    - ``"default"`` — uses the CI reference (normal for ``test="z"``).
    - ``"simple"``  — ``crit = qnorm``, ``pi.se = sqrt(tau2 + newvi)`` (drops ``se``).
    - ``"riley"``   — ``crit = qt(level/2, df = k - parms)``  (Riley et al. 2011).
    - ``"t"``       — ``crit = qt(level/2, df = k - p)``.

    ``parms`` defaults to ``p + 1`` (rma.uni counts one extra parameter for
    tau^2). ``test`` mirrors metafor's ``test`` (``"z"`` default, or
    ``"knha"``/``"t"`` for a t-based CI).

    **Measured coverage.** The prediction interval contains ``sqrt(se^2 +
    tau2)``, which treats ``tau2`` as known; over 2000 replications per
    scenario against the true effect of a new study, the default covered
    0.882 +/- 0.007 at ``k = 20`` and 0.804 +/- 0.014 at ``k = 5`` against a
    nominal 0.95. ``predtype="t"`` covered 0.899 and 0.894 at the same
    settings: never worse, and materially better when ``k`` is small. Neither
    reaches nominal, because the residual is the uncertainty in ``tau2``, which
    no plug-in variance includes. ``tools/coverage_pi.py`` reproduces the
    table, including a parametric bootstrap that does not beat ``t``.
    """
    predtype = predtype.lower()
    if predtype not in ("default", "simple", "riley", "t"):
        raise ValueError(f"unsupported predtype: {predtype!r}")
    level = _resolve_alpha(level)
    if parms is None:
        parms = p + 1  # rma.uni: p coefficients + 1 variance component (tau2)

    # metafor sets x$ddf = NA for rma.uni, then `ddf[is.na(ddf)] <- k - p`.
    ddf = float(k - p)

    # --- CI (uses the model's test reference) ---
    if test in ("knha", "adhoc", "t"):
        crit_ci = float(stats.t.ppf(1.0 - level / 2.0, ddf))
        pi_dist = "t"
    else:
        crit_ci = float(stats.norm.ppf(1.0 - level / 2.0))
        pi_dist = "norm"

    pred = float(est)
    vpred = se * se
    se_pred = float(se)
    ci_lb = pred - crit_ci * se_pred
    ci_ub = pred + crit_ci * se_pred

    # --- PI (overrides crit / vpred / pi_dist depending on predtype) ---
    pi_ddf = ddf
    vpred_pi = vpred
    crit_pi = crit_ci
    if predtype == "simple":
        crit_pi = float(stats.norm.ppf(1.0 - level / 2.0))
        vpred_pi = 0.0
        pi_dist = "norm"
    elif predtype in ("riley", "t"):
        if predtype == "riley":
            pi_ddf = ddf - parms + p
        else:
            pi_ddf = ddf
        pi_ddf = max(1.0, pi_ddf)
        crit_pi = float(stats.t.ppf(1.0 - level / 2.0, pi_ddf))
        pi_dist = "t"

    pi_se = math.sqrt(vpred_pi + tau2 + newvi)
    pi_lb = pred - crit_pi * pi_se
    pi_ub = pred + crit_pi * pi_se

    if transf is not None:
        pred = float(transf(pred) if targs is None else transf(pred, targs))
        ci_lb = float(transf(ci_lb) if targs is None else transf(ci_lb, targs))
        ci_ub = float(transf(ci_ub) if targs is None else transf(ci_ub, targs))
        pi_lb = float(transf(pi_lb) if targs is None else transf(pi_lb, targs))
        pi_ub = float(transf(pi_ub) if targs is None else transf(pi_ub, targs))

    return PredictResult(
        pred=pred,
        se=se_pred,
        ci_lb=ci_lb,
        ci_ub=ci_ub,
        pi_lb=pi_lb,
        pi_ub=pi_ub,
        pi_dist=pi_dist,
        ddf=ddf,
        pi_ddf=pi_ddf,
        level=level,
    )


@dataclass
class HcResult:
    """Hartung-Knapp (Sidik-Jonkman exact) interval (``metafor::hc.rma.uni``)."""

    beta: float          # DL pooled estimate (analysis scale)
    se: float            # HK-adjusted standard error
    ci_lb: float
    ci_ub: float
    tau2: float          # DerSimonian-Laird tau^2
    beta_rma: float      # pooled estimate from the original rma() model
    se_rma: float
    ci_lb_rma: float
    ci_ub_rma: float
    tau2_rma: float
    level: float         # alpha level


def hc_rma(
    yi,
    vi,
    level: float = 0.05,
    *,
    beta_rma: float | None = None,
    se_rma: float | None = None,
    ci_lb_rma: float | None = None,
    ci_ub_rma: float | None = None,
    tau2_rma: float | None = None,
) -> HcResult:
    """Hartung-Knapp interval for an intercept-only model (``metafor::hc.rma.uni``).

    Replicates the *exact* Sidik & Jonkman (2002) / Hartung (1999) method: it
    refits tau^2 with the DerSimonian-Laird estimator, then finds the inflated
    critical value ``u0 = SDR * t0`` where ``t0`` solves

        int_t0^inf P(Q <= finv(r/t0)) phi(r) dr = level/2

    with ``Q ~ Gamma(shape(r), scale(r))``. The same ``_zeroin`` (R
    ``uniroot``) root finder and a Gauss-Kronrod integration with R's
    ``integrate`` tolerance are used, so ``t0`` matches R's ``uniroot`` root.
    """
    yi = np.asarray(yi, dtype=float)
    vi = np.asarray(vi, dtype=float)
    k = yi.shape[0]
    if k == 1:
        raise ValueError("hc: k = 1")
    if np.any(vi <= 0.0):
        raise ValueError("hc: one or more sampling variances are non-positive")
    level = _resolve_alpha(level)

    wi = 1.0 / vi
    W1 = float(np.sum(wi))
    W2 = float(np.sum(wi**2) / W1)
    W3 = float(np.sum(wi**3) / W1)
    W4 = float(np.sum(wi**4) / W1)

    beta = float(np.sum(wi * yi) / W1)
    Q = float(np.sum(wi * (yi - beta) ** 2))
    tau2 = max(0.0, (Q - (k - 1)) / (W1 - W2))
    vb = (tau2 * W2 + 1.0) / W1
    se = math.sqrt(vb)

    VR = 1.0 + tau2 * W2
    SDR = math.sqrt(VR)

    def eq_func(r: float) -> float:
        return (k - 1) + tau2 * (W1 - W2) + (tau2**2) * ((1.0 / VR**2) * (r**2) - 1.0 / VR) * (W3 - W2**2)

    def vq_func(r: float) -> float:
        rsq = r * r
        recipvr2 = 1.0 / VR**2
        return (
            2.0 * (k - 1)
            + 4.0 * tau2 * (W1 - W2)
            + 2.0 * tau2**2 * (W1 * W2 - 2.0 * W3 + W2**2)
            + 4.0 * tau2**2 * (recipvr2 * rsq - 1.0 / VR) * (W3 - W2**2)
            + 4.0 * tau2**3 * (recipvr2 * rsq - 1.0 / VR) * (W4 - 2.0 * W2 * W3 + W2**3)
            + 2.0 * tau2**4 * (recipvr2 - 2.0 * (1.0 / VR**3) * rsq) * (W3 - W2**2) ** 2
        )

    def scale(r: float) -> float:
        return vq_func(r) / eq_func(r)

    def shape(r: float) -> float:
        return eq_func(r) ** 2 / vq_func(r)

    def finv(f: float) -> float:
        return (W1 / W2 - 1.0) * (f * f - 1.0) + (k - 1)

    # integration tolerance matching R's integrate() default
    # (rel.tol = abs.tol = .Machine$double.eps^0.25)
    _itol = 1.220703125e-4

    def eqn(x: float) -> float:
        if x <= 0.0:
            # as x -> 0+, finv(r/x) -> +inf for r > 0, so the gamma CDF -> 1
            # and the integral of dnorm over (0, inf) is 1/2.
            return 0.5 - level / 2.0

        def integrand(r: float) -> float:
            cdf = stats.gamma.cdf(finv(r / x), a=shape(SDR * r), scale=scale(SDR * r))
            return cdf * stats.norm.pdf(r)

        integral, _ = integrate.quad(integrand, x, np.inf, epsabs=_itol, epsrel=_itol)
        return integral - level / 2.0

    t0 = _zeroin(eqn, 0.0, 2.0, tol=_itol)
    u0 = SDR * t0
    ci_lb = beta - u0 * se
    ci_ub = beta + u0 * se

    if beta_rma is None:
        beta_rma = beta
    if se_rma is None:
        se_rma = se
    if ci_lb_rma is None:
        ci_lb_rma = ci_lb
    if ci_ub_rma is None:
        ci_ub_rma = ci_ub
    if tau2_rma is None:
        tau2_rma = tau2

    return HcResult(
        beta=beta,
        se=se,
        ci_lb=ci_lb,
        ci_ub=ci_ub,
        tau2=tau2,
        beta_rma=beta_rma,
        se_rma=se_rma,
        ci_lb_rma=ci_lb_rma,
        ci_ub_rma=ci_ub_rma,
        tau2_rma=tau2_rma,
        level=level,
    )


def pool_yv(
    y: list[float],
    v: list[float],
    n: list[float] | None,
    effect_type: str,
    model: str = "random",
    method: str = "DL",
    alpha: float = 0.05,
) -> PooledResult:
    """Pool analysis-scale ``(y, v)`` arrays directly.

    This is the shared numeric core behind :func:`run_meta` and the
    binary-data wrappers (metabin). ``y`` must already be on the analysis
    scale (log scale for ratio types).
    """
    k = len(y)
    n = n or [math.nan] * k

    tau2, used_method = _resolve_tau2(y, v, model, method, n)
    fit = _fit_wls(y, v, tau2, alpha)

    q_df = k - 1
    i2, h2 = _i2_h2(v, tau2, fit["q"], model)
    weights = [1.0 / (vi + tau2) for vi in v]

    pi_lower = pi_upper = None
    if model != "fixed":
        z_crit = stats.norm.ppf(1 - alpha / 2)
        t_crit = stats.t.ppf(1 - alpha / 2, df=k - 2) if k > 2 else z_crit
        pi_se = math.sqrt(tau2 + fit["se"] ** 2)
        pi_lower = fit["est"] - t_crit * pi_se
        pi_upper = fit["est"] + t_crit * pi_se

    return PooledResult(
        model=model,
        method=used_method,
        k=k,
        effect_type=effect_type,
        pooled_effect=_back_transform(fit["est"], effect_type),
        pooled_ci_lower=_back_transform(fit["ci_lower"], effect_type),
        pooled_ci_upper=_back_transform(fit["ci_upper"], effect_type),
        pooled_est=fit["est"],
        pooled_se=fit["se"],
        z=fit["z"],
        p_value=fit["p"],
        q=fit["q"],
        q_df=q_df,
        q_p=fit["q_p"],
        tau2=tau2,
        i2=i2,
        h2=h2,
        weights=weights,
        pi_lower=_back_transform(pi_lower, effect_type) if pi_lower is not None else None,
        pi_upper=_back_transform(pi_upper, effect_type) if pi_upper is not None else None,
    )


def run_meta(
    effects: list[StudyEffect],
    effect_type: str,
    model: str = "random",
    method: str = "DL",
    alpha: float = 0.05,
) -> PooledResult:
    """Pool a set of study effects (mirrors ``rma`` / ``metagen``).

    Args:
        effects: per-study effect estimates with SE/variance/CI.
        effect_type: one of RR/OR/HR/PETO/IRR/PLN/PLO (log scale) or
            MD/SMD/WMD/RD/AS/PAS/PFT (raw scale).
        model: "fixed" or "random".
        method: tau-squared estimator, one of DL/HE/HS/SJ/ML/REML/PM/EB
            (meta/metafor default is DL). Ignored for fixed-effect.
        alpha: significance level for CI/PI (0.05 -> 95%).
    """
    y, v, n = _prepare(effects, effect_type)
    return pool_yv(y, v, n, effect_type, model=model, method=method, alpha=alpha)


def result_to_dict(result: PooledResult) -> dict:
    """Serialize a PooledResult to a plain ``dict`` of Python scalars."""
    return {
        "model": result.model,
        "method": result.method,
        "k": result.k,
        "effect_type": result.effect_type,
        "pooled_effect": result.pooled_effect,
        "pooled_ci_lower": result.pooled_ci_lower,
        "pooled_ci_upper": result.pooled_ci_upper,
        "pooled_est": result.pooled_est,
        "pooled_se": result.pooled_se,
        "z": result.z,
        "p_value": result.p_value,
        "q": result.q,
        "q_df": result.q_df,
        "q_p": result.q_p,
        "tau2": result.tau2,
        "i2": result.i2,
        "h2": result.h2,
        "weights": result.weights,
        "pi_lower": result.pi_lower,
        "pi_upper": result.pi_upper,
    }
