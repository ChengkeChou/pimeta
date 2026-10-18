"""Publication-bias tests (metafor::regtest/ranktest/trimfill).

Implements:

- Egger's regression test (weighted regression of ``y`` on ``sqrt(v)`` with
  weights ``1/v``, the standard formulation of Egger et al. 1997).
- Begg & Mazumdar's rank correlation test (1994).
- Duval & Tweedie's trim-and-fill (2000) with the L0/R0 missing-study
  estimators and a random-effects re-pooling of the filled dataset.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from scipy import stats

from .pooling import pool_yv

Z_975 = stats.norm.ppf(0.975)


@dataclass
class EggerResult:
    """Egger regression test output."""

    intercept: float
    intercept_se: float
    stat: float          # t (model="lm") or z (model="rma")
    p: float
    slope: float
    slope_se: float
    k: int
    model: str           # "lm" | "rma"

    def to_dict(self) -> dict:
        return {
            "intercept": self.intercept,
            "intercept_se": self.intercept_se,
            "stat": self.stat,
            "p": self.p,
            "slope": self.slope,
            "slope_se": self.slope_se,
            "k": self.k,
            "model": self.model,
        }


@dataclass
class BeggResult:
    """Begg & Mazumdar rank correlation test output."""

    tau: float
    z: float
    p: float
    k: int

    def to_dict(self) -> dict:
        return {"tau": self.tau, "z": self.z, "p": self.p, "k": self.k}


@dataclass
class TrimfillResult:
    """Trim-and-fill output."""

    k0: int                       # number of filled studies
    estimator: str                # "L0" | "R0" | "Q0"
    side: str                     # "left" | "right"
    se_k0: float                  # SE of k0 (metafor::trimfill)
    p_k0: float | None            # p-value for H0: k0 = 0 (R0 only)
    filled_effect: float          # pooled effect after filling (analysis scale)
    filled_se: float
    filled_ci_lower: float
    filled_ci_upper: float
    filled_tau2: float
    original_effect: float
    k: int

    def to_dict(self) -> dict:
        return {
            "k0": self.k0,
            "estimator": self.estimator,
            "side": self.side,
            "se_k0": self.se_k0,
            "p_k0": self.p_k0,
            "filled_effect": self.filled_effect,
            "filled_se": self.filled_se,
            "filled_ci_lower": self.filled_ci_lower,
            "filled_ci_upper": self.filled_ci_upper,
            "filled_tau2": self.filled_tau2,
            "original_effect": self.original_effect,
            "k": self.k,
        }


def _wls_beta(X, w, y):
    """Return (beta, cov) for weighted least squares."""
    X = np.asarray(X, dtype=float)
    w = np.asarray(w, dtype=float)
    y = np.asarray(y, dtype=float)
    XtW = X.T * w
    XWX = XtW @ X
    beta = np.linalg.solve(XWX, XtW @ y)
    cov = np.linalg.inv(XWX)
    return beta, cov


def egger_test(y, v, model: str = "rma", method: str = "REML") -> EggerResult:
    """Egger's regression test (metafor::regtest).

    ``model="lm"``  : classical Egger test — weighted regression of ``y`` on
                      ``sqrt(v)`` with weights ``1/v``, multiplicative
                      dispersion, and a t-test on the *slope*.
    ``model="rma"`` : mixed-effects meta-regression version (metafor default)
                      — ``metareg(y, v, X=sqrt(v))`` with a z-test on the
                      *slope*.

    The test statistic is the coefficient of ``sqrt(v)`` (the slope), exactly
    as in metafor::regtest. A significant slope indicates funnel-plot
    asymmetry; the intercept is the "limit estimate" (pooled effect under
    infinite precision).
    """
    y = np.asarray(y, dtype=float)
    v = np.asarray(v, dtype=float)
    k = len(y)
    se = np.sqrt(v)

    if model == "lm":
        # metafor: fit <- lm(yi ~ 0 + X, weights=1/vi); X = [1, sei]
        X = np.column_stack([np.ones(k), se])
        w = 1.0 / v
        beta, cov_no_disp = _wls_beta(X, w, y)
        b0, b1 = beta
        # multiplicative dispersion term: sigma^2 = sum(w*r^2) / (k - p)
        resid = y - X @ beta
        sigma2 = float(np.sum(w * resid ** 2) / (k - 2))
        cov = sigma2 * cov_no_disp
        se0 = math.sqrt(cov[0, 0])
        se1 = math.sqrt(cov[1, 1])
        stat = b1 / se1
        p = 2.0 * float(stats.t.sf(abs(stat), df=k - 2))
    elif model == "rma":
        from .metareg import metareg

        res = metareg(list(y), list(v), list(se), X_names=["sei"], method=method)
        b0, b1 = res.b[0], res.b[1]
        se0, se1 = res.se[0], res.se[1]
        stat = b1 / se1
        p = 2.0 * float(stats.norm.sf(abs(stat)))
    else:
        raise ValueError(f"unknown model: {model!r} (expected 'lm' or 'rma')")

    return EggerResult(
        intercept=float(b0),
        intercept_se=float(se0),
        stat=float(stat),
        p=p,
        slope=float(b1),
        slope_se=float(se1),
        k=k,
        model=model,
    )


@dataclass
class RegtestResult:
    """metafor::regtest output (Egger / Peters / other predictors)."""

    predictor: str
    stat: float          # z (model="rma", test="z") or t
    p: float
    model: str
    test: str
    method: str
    intercept: float | None
    intercept_ci_lower: float | None
    intercept_ci_upper: float | None
    k: int

    def to_dict(self) -> dict:
        return {
            "predictor": self.predictor,
            "stat": self.stat,
            "p": self.p,
            "model": self.model,
            "test": self.test,
            "method": self.method,
            "intercept": self.intercept,
            "intercept_ci_lower": self.intercept_ci_lower,
            "intercept_ci_upper": self.intercept_ci_upper,
            "k": self.k,
        }


_PREDICTORS = {"sei", "vi", "ni", "ninv", "sqrtni", "sqrtninv"}


def _predictor_values(predictor: str, v: np.ndarray, ni):
    """Column vector for the chosen predictor (metafor::regtest)."""
    if predictor == "sei":
        return np.sqrt(v)
    if predictor == "vi":
        return np.asarray(v, dtype=float)
    if ni is None:
        raise ValueError("sample sizes (ni) required for this predictor")
    ni = np.asarray(ni, dtype=float)
    if predictor == "ni":
        return ni
    if predictor == "ninv":
        return 1.0 / ni
    if predictor == "sqrtni":
        return np.sqrt(ni)
    if predictor == "sqrtninv":
        return 1.0 / np.sqrt(ni)
    raise ValueError(f"unknown predictor: {predictor!r}")


def regtest(
    y,
    v,
    ni=None,
    predictor: str = "sei",
    model: str = "rma",
    method: str = "REML",
    test: str = "z",
    alpha: float = 0.05,
) -> RegtestResult:
    """Regression test for funnel-plot asymmetry (metafor::regtest).

    Fits ``y ~ 1 + predictor`` (a meta-regression, or a weighted linear model
    when ``model="lm"``) and tests the slope. ``predictor="sei"`` is Egger's
    test; ``predictor="ninv"`` (1/n) is Peters' test; ``predictor="vi"`` is
    the additive variant. When ``test="knha"`` the coefficient test uses the
    Knapp-Hartung adjustment (recommended for Peters' test).
    """
    y = np.asarray(y, dtype=float)
    v = np.asarray(v, dtype=float)
    k = len(y)

    if predictor not in _PREDICTORS:
        raise ValueError(f"unknown predictor: {predictor!r}")

    xp = _predictor_values(predictor, v, ni)

    if model == "rma":
        from .metareg import metareg

        res = metareg(list(y), list(v), list(xp), X_names=[predictor],
                      method=method, alpha=alpha, test=test)
        stat = res.z[1]
        p = res.p[1]
        est = res.b[0]
        ci_l = res.ci_lower[0]
        ci_u = res.ci_upper[0]
        used_method = res.method
    elif model == "lm":
        X = np.column_stack([np.ones(k), xp])
        w = 1.0 / v
        beta, cov_no_disp = _wls_beta(X, w, y)
        resid = y - X @ beta
        sigma2 = float(np.sum(w * resid ** 2) / (k - 2))
        cov = sigma2 * cov_no_disp
        se0 = math.sqrt(cov[0, 0])
        se1 = math.sqrt(cov[1, 1])
        stat = beta[1] / se1
        p = 2.0 * float(stats.t.sf(abs(stat), df=k - 2))
        est = float(beta[0])
        crit = stats.t.ppf(1.0 - alpha / 2.0, df=k - 2)
        ci_l = est - crit * se0
        ci_u = est + crit * se0
        used_method = "lm"
    else:
        raise ValueError(f"unknown model: {model!r} (expected 'lm' or 'rma')")

    # The "limit estimate" (intercept) is only meaningful for these predictors
    # and an intercept-only base model (metafor::regtest condition).
    has_limit = predictor in ("sei", "vi", "ninv", "sqrtninv")

    return RegtestResult(
        predictor=predictor,
        stat=float(stat),
        p=float(p),
        model=model,
        test=test,
        method=used_method,
        intercept=float(est) if has_limit else None,
        intercept_ci_lower=float(ci_l) if has_limit else None,
        intercept_ci_upper=float(ci_u) if has_limit else None,
        k=k,
    )


def peters_test(y, v, ni, test: str = "knha", method: str = "REML") -> RegtestResult:
    """Peters' regression test: ``y ~ 1 + 1/n`` (metafor::regtest, ninv)."""
    return regtest(y, v, ni=ni, predictor="ninv", model="rma", method=method, test=test)


def begg_test(y, v) -> BeggResult:
    """Begg & Mazumdar rank correlation test (metafor::ranktest)."""
    y = np.asarray(y, dtype=float)
    v = np.asarray(v, dtype=float)
    k = len(y)

    # Fixed-effect centre and its variance.
    w = 1.0 / v
    sw = float(np.sum(w))
    mu = float(np.sum(w * y) / sw)
    var_mu = 1.0 / sw

    # Standardised effect sizes (Begg & Mazumdar 1994).
    y_std = (y - mu) / np.sqrt(v - var_mu)

    # Kendall's tau between y_std and v (exact p-value matches R's cor.test).
    tau, p = stats.kendalltau(y_std, v)

    # z for the large-sample approximation (informational only).
    var_tau = 2.0 * (2.0 * k + 5.0) / (9.0 * k * (k - 1.0))
    z = tau / math.sqrt(var_tau)
    return BeggResult(tau=float(tau), z=float(z), p=float(p), k=k)


_FIXED_METHODS = {"EE", "FE", "CE"}


def _pool_est(y: np.ndarray, v: np.ndarray, method: str) -> float:
    """Pooled intercept estimate (analysis scale) for trimfill iterations.

    Mirrors ``rma.uni(yi, vi, method=x$method)``: fixed-effect for
    EE/FE/CE, random-effects otherwise (tau2 via ``method``).
    """
    if method.upper() in _FIXED_METHODS:
        w = 1.0 / v
        return float(np.sum(w * y) / np.sum(w))
    res = pool_yv(list(y), list(v), None, "MD", model="random", method=method)
    return res.pooled_est


def trimfill(
    y,
    v,
    effect_type: str = "MD",
    estimator: str = "L0",
    method_tau: str = "REML",
    alpha: float = 0.05,
) -> TrimfillResult:
    """Duval & Tweedie trim-and-fill (metafor::trimfill.rma.uni).

    Replicates the exact iterative algorithm:

    1. Determine the side of asymmetry from Egger's regression test
       (slope of ``y ~ sqrt(v)`` < 0 -> "right", else "left").
    2. If "right", flip the signs so the problem is a left-side fill.
    3. Sort by ascending effect; iteratively truncate the ``k0`` most
       extreme studies, re-estimating the centre until ``k0`` converges.
    4. Impute ``k0`` mirrored studies (``2*centre - yi``) and re-pool.

    ``method_tau`` mirrors metafor's ``x$method`` (EE/FE/CE for fixed,
    DL/REML/ML/... for random effects).
    """
    y = np.asarray(y, dtype=float)
    v = np.asarray(v, dtype=float)
    k = len(y)

    estimator = estimator.upper()
    if estimator not in ("L0", "R0", "Q0"):
        raise ValueError(f"unknown estimator: {estimator!r} (expected L0/R0/Q0)")

    fixed = method_tau.upper() in _FIXED_METHODS

    # --- side of asymmetry via regression test -------------------------
    reg = egger_test(y, v, model=("lm" if fixed else "rma"), method=method_tau)
    side = "right" if reg.slope < 0 else "left"

    # --- flip data if examining the right side -------------------------
    yw = -y if side == "right" else y.copy()
    vw = v.copy()

    # --- sort by ascending effect --------------------------------------
    ix = np.argsort(yw)
    yw = yw[ix]
    vw = vw[ix]

    # --- iterative trim & fill -----------------------------------------
    k0 = 0
    k0_sav = -1
    beta = 0.0
    se_k0 = 0.0
    for _ in range(100):  # maxiter = 100 (metafor default)
        k0_sav = k0

        beta = _pool_est(yw[: k - k0], vw[: k - k0], method_tau)

        yic = yw - beta
        ranks = stats.rankdata(np.abs(yic), method="ordinal")
        signed = np.sign(yic) * ranks

        if estimator == "L0":
            Sr = float(np.sum(signed[signed > 0]))
            k0 = max(0, round((4.0 * Sr - k * (k + 1)) / (2.0 * k - 1)))
            var_sr = (1.0 / 24.0) * (
                k * (k + 1) * (2 * k + 1)
                + 10 * k0 ** 3 + 27 * k0 ** 2 + 17 * k0
                - 18 * k * k0 ** 2 - 18 * k * k0 + 6 * k ** 2 * k0
            )
            se_k0 = 4.0 * math.sqrt(max(0.0, var_sr)) / (2.0 * k - 1)
        elif estimator == "R0":
            neg = signed[signed < 0]
            k0 = max(0, round((k - int(np.max(-1.0 * neg))) - 1))
            se_k0 = math.sqrt(2.0 * max(0, k0) + 2.0)
        else:  # Q0
            Sr = float(np.sum(signed[signed > 0]))
            k0 = max(0, round(k - 0.5 - math.sqrt(2.0 * k * k - 4.0 * Sr + 0.25)))
            var_sr = (1.0 / 24.0) * (
                k * (k + 1) * (2 * k + 1)
                + 10 * k0 ** 3 + 27 * k0 ** 2 + 17 * k0
                - 18 * k * k0 ** 2 - 18 * k * k0 + 6 * k ** 2 * k0
            )
            denom = (k - 0.5) ** 2 - k0 * (2 * k - k0 - 1)
            se_k0 = 2.0 * math.sqrt(max(0.0, var_sr)) / math.sqrt(max(denom, 1e-12))

        if k0 == k0_sav:
            break

    # --- R0 p-value for H0: no missing studies -------------------------
    if estimator == "R0":
        # 1 - sum_{j=1}^{k0+1} (1/2)^j  (metafor::trimfill)
        p_k0 = 1.0 - float(np.sum(0.5 ** np.arange(1, k0 + 2)))
    else:
        p_k0 = None

    # --- original pooled estimate --------------------------------------
    orig = pool_yv(list(y), list(v), None, effect_type,
                   model=("fixed" if fixed else "random"),
                   method=method_tau, alpha=alpha)

    # --- final centre (analysis scale, original space) -----------------
    centre = _pool_est(yw[: k - k0], vw[: k - k0], method_tau)
    if side == "right":
        centre = -centre

    if k0 == 0:
        return TrimfillResult(
            k0=0, estimator=estimator, side=side, se_k0=se_k0, p_k0=p_k0,
            filled_effect=orig.pooled_est, filled_se=orig.pooled_se,
            filled_ci_lower=orig.pooled_ci_lower,
            filled_ci_upper=orig.pooled_ci_upper,
            filled_tau2=orig.tau2, original_effect=orig.pooled_est, k=k,
        )

    # --- impute mirrored studies ---------------------------------------
    trimmed_y = yw[k - k0:]          # k0 most extreme (flip space)
    trimmed_v = vw[k - k0:]
    trimmed_orig = -trimmed_y if side == "right" else trimmed_y
    filled_y = 2.0 * centre - trimmed_orig
    filled_v = trimmed_v

    fy = np.concatenate([y, filled_y])
    fv = np.concatenate([v, filled_v])

    pooled = pool_yv(list(fy), list(fv), None, effect_type,
                     model=("fixed" if fixed else "random"),
                     method=method_tau, alpha=alpha)

    return TrimfillResult(
        k0=k0, estimator=estimator, side=side, se_k0=se_k0, p_k0=p_k0,
        filled_effect=pooled.pooled_est, filled_se=pooled.pooled_se,
        filled_ci_lower=pooled.pooled_ci_lower,
        filled_ci_upper=pooled.pooled_ci_upper,
        filled_tau2=pooled.tau2, original_effect=orig.pooled_est, k=k,
    )
