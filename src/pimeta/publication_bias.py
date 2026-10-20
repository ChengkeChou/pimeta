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
from scipy import special, stats

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
    # slope and residual dispersion, needed by metabias() (meta's linregcore)
    slope: float | None = None
    slope_se: float | None = None
    intercept_se: float | None = None
    tau: float | None = None
    df: int | None = None

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
            "slope": self.slope,
            "slope_se": self.slope_se,
            "intercept_se": self.intercept_se,
            "tau": self.tau,
            "df": self.df,
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
        slope = float(res.b[1])
        slope_se = float(res.se[1])
        intercept_se = float(res.se[0])
        tau = math.sqrt(float(res.tau2)) if res.tau2 is not None else None
        df = int(k - 2)
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
        slope = float(beta[1])
        slope_se = float(se1)
        intercept_se = float(se0)
        tau = math.sqrt(sigma2)
        df = int(k - 2)
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
        slope=slope,
        slope_se=slope_se,
        intercept_se=intercept_se,
        tau=tau,
        df=df,
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


# --------------------------------------------------------------------------- #
# meta::metabias — tests for funnel-plot asymmetry (all nine methods)
# --------------------------------------------------------------------------- #

#: meta's ``method.bias`` choices plus the pre-8.x aliases.
_BIAS_METHODS = ("Begg", "Egger", "Thompson", "Schwarzer", "Harbord",
                 "Peters", "Deeks", "Pustejovsky", "Macaskill")
_BIAS_ALIASES = {"rank": "Begg", "linreg": "Egger", "mm": "Thompson",
                 "count": "Schwarzer", "score": "Harbord"}


def _bias_method(method):
    """Prefix-match ``method`` against meta's ``meth4bias`` (``setmethodbias``)."""
    if method is None:
        return None
    name = str(method).strip()
    low = name.lower()
    for alias, full in _BIAS_ALIASES.items():
        if low and alias.startswith(low):
            return full
    hits = [m for m in _BIAS_METHODS if m.lower().startswith(low)]
    if len(hits) == 1:
        return hits[0]
    if len(hits) > 1:
        raise ValueError(f"'{name}' is ambiguous: {hits}")
    raise ValueError(f"unknown bias method {name!r}; expected one of "
                     f"{list(_BIAS_METHODS)}")


def _kentau(x, y, correct=False):
    """``meta::kentau`` — Kendall's tau with meta's tie corrections.

    Returns ``(ks, se_ks, statistic, p_value)``; ``ks`` is Kendall's S (with
    the optional continuity correction ``ks - sign(ks)`` applied) and
    ``se_ks`` its tie-corrected standard error.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if x.shape != y.shape:
        raise ValueError("length of argument x and y must be equal")
    ok = np.isfinite(x) & np.isfinite(y)
    x, y = x[ok], y[ok]
    n = x.size
    if n < 3:
        raise ValueError("at least three observations are needed")
    t = np.unique(x, return_counts=True)[1].astype(float)
    u = np.unique(y, return_counts=True)[1].astype(float)
    N = 0.5 * n * (n - 1.0)
    N1 = N - float(np.sum(t * (t - 1.0) / 2.0))
    N2 = N - float(np.sum(u * (u - 1.0) / 2.0))

    # R's cor(..., method="kendall") is tau-b.
    ks = math.sqrt(N1) * math.sqrt(N2) * float(stats.kendalltau(x, y).statistic)

    se2 = (1.0 / 18.0 * (n * (n - 1.0) * (2.0 * n + 5.0)
                         - float(np.sum(t * (t - 1.0) * (2.0 * t + 5.0)))
                         - float(np.sum(u * (u - 1.0) * (2.0 * u + 5.0))))
           + 1.0 / (9.0 * n * (n - 1.0) * (n - 2.0))
           * float(np.sum(t * (t - 1.0) * (t - 2.0)))
           * float(np.sum(u * (u - 1.0) * (u - 2.0)))
           + 1.0 / (2.0 * n * (n - 1.0))
           * float(np.sum(t * (t - 1.0))) * float(np.sum(u * (u - 1.0))))
    se_ks = math.sqrt(se2)

    sign = 0.0 if ks == 0 else (1.0 if ks > 0 else -1.0)
    ks = ks - sign * (1.0 if correct else 0.0)
    statistic = ks / se_ks
    p_value = 2.0 * float(stats.norm.sf(abs(statistic)))
    return ks, se_ks, statistic, p_value


@dataclass
class MetabiasResult:
    """Result of :func:`metabias` (``meta::metabias``).

    ``statistic`` is the z of the rank tests and the t of the regression
    tests; ``estimate`` holds ``[bias, se.bias]`` — the slope of the
    regression of the effect on the chosen predictor — and ``intercept`` /
    ``se_intercept`` the "limit estimate".  When ``k < k_min`` (or ``k < 3``)
    meta reports only ``k`` and ``k_min``; the remaining fields are ``None``.
    """

    method_bias: str
    method: str = ""
    var_model: str = ""            # "" | "additive" | "multiplicative"
    statistic: float | None = None
    pval: float | None = None
    p_value: float | None = None
    estimate: list[float] | None = None       # [bias, se.bias] / [ks, se.ks]
    intercept: float | None = None
    se_intercept: float | None = None
    tau: float | None = None
    df: int | None = None
    k: int | None = None
    k_min: float = 10

    def __repr__(self):
        if self.estimate is None:
            return (f"MetabiasResult(method_bias={self.method_bias!r}, "
                    f"k={self.k!r}, k_min={self.k_min!r})")
        return (f"MetabiasResult(method_bias={self.method_bias!r}, "
                f"statistic={self.statistic:.6g}, pval={self.pval:.6g}, "
                f"estimate={self.estimate})")

    def to_dict(self) -> dict:
        return {
            "method_bias": self.method_bias, "method": self.method,
            "var_model": self.var_model, "statistic": self.statistic,
            "pval": self.pval, "p_value": self.p_value,
            "estimate": self.estimate, "intercept": self.intercept,
            "se_intercept": self.se_intercept, "tau": self.tau,
            "df": self.df, "k": self.k, "k_min": self.k_min,
        }


def _linregcore(te, se, covar=None, model="lm", method_tau="DL", alpha=0.05):
    """``meta:::linregcore`` — the regression core shared by the bias tests.

    ``model="lm"``  weighted least squares (``w = 1/se^2``) with a
                    multiplicative dispersion factor and a t test on ``k - 2``
                    degrees of freedom.
    ``model="rma"`` random-effects meta-regression (metafor ``rma``) with a
                    t test, as meta's Thompson variant does.
    """
    te = np.asarray(te, dtype=float)
    se = np.asarray(se, dtype=float)
    k = te.size
    xp = se if covar is None else np.asarray(covar, dtype=float)
    df = k - 2

    if model == "lm":
        X = np.column_stack([np.ones(k), xp])
        w = 1.0 / se ** 2
        beta, cov_no_disp = _wls_beta(X, w, te)
        resid = te - X @ beta
        sigma2 = float(np.sum(w * resid ** 2) / df)
        cov = sigma2 * cov_no_disp
        b0, b1 = float(beta[0]), float(beta[1])
        se0, se1 = math.sqrt(cov[0, 0]), math.sqrt(cov[1, 1])
        tau = math.sqrt(sigma2)
    elif model == "rma":
        from .metareg import metareg

        res = metareg(list(te), list(se ** 2), list(xp), method=method_tau,
                      test="t", alpha=alpha)
        b0, b1 = float(res.b[0]), float(res.b[1])
        se0, se1 = float(res.se[0]), float(res.se[1])
        tau = math.sqrt(float(res.tau2))
    else:
        raise ValueError("model must be 'lm' or 'rma'")

    statistic = b1 / se1
    return {
        "intercept": b0, "se.intercept": se0,
        "slope": b1, "se.slope": se1,
        "statistic": statistic, "df": df,
        "pval": 2.0 * float(stats.t.sf(abs(statistic), df=df)),
        "tau": tau,
    }


def _hypergeom_mean_var(n1, m1, N, psi):
    """Mean and variance of ``meta:::hypergeometric``'s *noncentral* law.

    ``meta`` builds the distribution of the upper-left cell of a 2x2 table
    with margins ``n1``/``m1``/``N`` and **odds ratio** ``psi`` — i.e.
    ``P(x) ∝ C(n1, x) C(N - n1, m1 - x) psi^x`` — and the Schwarzer test uses
    its mean and variance as the null moments.
    """
    n2 = N - n1
    lo = int(max(0, m1 - n2))
    hi = int(min(n1, m1))
    xs = np.arange(lo, hi + 1, dtype=float)
    logw = (special.gammaln(n1 + 1.0) - special.gammaln(xs + 1.0)
            - special.gammaln(n1 - xs + 1.0)
            + special.gammaln(n2 + 1.0) - special.gammaln(m1 - xs + 1.0)
            - special.gammaln(n2 - m1 + xs + 1.0)
            + xs * math.log(psi))
    w = np.exp(logw - logw.max())
    w /= w.sum()
    mean = float(np.sum(w * xs))
    var = float(np.sum(w * xs ** 2)) - mean ** 2
    return mean, var


def _metabias_fields(x):
    """Pull the columns ``metabias`` needs out of a pimeta result object."""
    yi = getattr(x, "yi", None)
    vi = getattr(x, "vi", None)
    if yi is not None and vi is not None:
        with np.errstate(invalid="ignore"):
            se = np.sqrt(np.asarray(vi, dtype=float))
        te = np.asarray(yi, dtype=float)
    else:
        # meta-style results (metagen / metacum / ...) name them TE / seTE.
        te = getattr(x, "TE", None)
        se = getattr(x, "seTE", None)
        if te is None or se is None:
            raise TypeError(
                "argument 'x' must be a pimeta result with yi/vi (metabin, "
                "metacont, ...) or TE/seTE (metagen, ...), or an array of "
                "effect sizes supplied together with 'seTE'")
        te = np.asarray(te, dtype=float)
        se = np.asarray(se, dtype=float)
    return {
        "TE": te, "seTE": se,
        "n_e": getattr(x, "n_e", None),
        "n_c": getattr(x, "n_c", None),
        "event_e": getattr(x, "event_e", None),
        "event_c": getattr(x, "event_c", None),
        "sm": getattr(x, "sm", None),
        "method_tau": getattr(x, "method_tau", None) or "REML",
        # metaprop / metarate keep a single arm (x$event, x$n in R)
        "single": (getattr(x, "event_e", None) is None
                   and getattr(x, "event", None) is not None),
    }


def metabias(x, seTE=None, method_bias=None, plotit=False, correct=False,
             k_min=10, n_e=None, n_c=None, event_e=None, event_c=None,
             sm=None, method_tau=None, alpha=0.05):
    """Test for funnel-plot asymmetry (``meta::metabias``).

    All nine methods meta supports are available:

    ===============  ========================================================
    ``Begg``         rank correlation of the standardised effect sizes with
                     their variances (``kentau``, with meta's tie corrections)
    ``Egger``        regression of the effect on its standard error (``lm``)
    ``Thompson``     the same regression fitted as a random-effects model
                     (additive; uses the analysis' tau^2 estimator)
    ``Schwarzer``    hypergeometric variant of the rank test (binary data)
    ``Harbord``      score test built from the 2x2 tables (odds-ratio scale)
    ``Peters``       regression on ``1/n`` with the Peters variances
    ``Deeks``        regression on ``1/sqrt(ESS)`` for diagnostic odds ratios
    ``Pustejovsky``  regression on ``sqrt(1/n_e + 1/n_c)``
    ``Macaskill``    regression weighted by the total sample size
    ===============  ========================================================

    Args:
        x: a result object (``metabin`` / ``metagen`` / ...) or, together with
            ``seTE``, the effect sizes themselves.
        seTE: standard errors when ``x`` holds the effect sizes.
        method_bias: one of the names above (prefix-matched; the pre-8.x
            aliases ``rank``/``linreg``/``mm``/``count``/``score`` also work).
            Defaults to ``"Harbord"`` for a binary odds-ratio analysis and
            ``"Egger"`` otherwise, matching meta.
        plotit: accepted for interface parity — draw the radial plot yourself
            with :func:`pimeta.plotting.radial`.
        correct: continuity correction for the rank tests.
        k_min: meta requires ``k >= k_min`` and ``k >= 3``; otherwise only
            ``k`` and ``k_min`` are reported, exactly like R.
        n_e, n_c, event_e, event_c, sm: override the values taken from ``x``.
        method_tau: tau^2 estimator used by the Thompson variant (defaults to
            the analysis' own estimator, i.e. R's ``x$method.tau``).
        alpha: significance level used for the regression CIs.

    Returns:
        :class:`MetabiasResult`.
    """
    if seTE is not None:
        fields = {"TE": np.asarray(x, dtype=float),
                  "seTE": np.asarray(seTE, dtype=float),
                  "n_e": n_e, "n_c": n_c, "event_e": event_e,
                  "event_c": event_c, "sm": sm,
                  "method_tau": method_tau or "REML", "single": False}
    else:
        fields = _metabias_fields(x)
        for key, val in (("n_e", n_e), ("n_c", n_c), ("event_e", event_e),
                         ("event_c", event_c), ("sm", sm)):
            if val is not None:
                fields[key] = val
        if method_tau is not None:
            fields["method_tau"] = method_tau
        # a single-arm analysis only has x$event / x$n
        if fields.get("event_e") is None and getattr(x, "event", None) \
                is not None and n_e is None:
            fields["event_e"] = np.asarray(x.event, dtype=float)
            fields["n_e"] = np.asarray(getattr(x, "n", None), dtype=float)

    TE = np.asarray(fields["TE"], dtype=float)
    se = np.asarray(fields["seTE"], dtype=float)
    if TE.shape != se.shape:
        raise ValueError("Length of argument TE and seTE must be equal.")
    tauf = str(fields["method_tau"] or "REML")

    if method_bias is None:
        binary = fields.get("event_e") is not None
        method_bias = ("Harbord" if binary
                       and str(fields.get("sm") or "").upper() == "OR"
                       else "Egger")
    method_bias = _bias_method(method_bias)

    # Drop studies with a missing effect or standard error.
    ok = np.isfinite(TE) & np.isfinite(se)
    if not ok.all():
        TE, se = TE[ok], se[ok]
        for key in ("n_e", "n_c", "event_e", "event_c"):
            if fields.get(key) is not None:
                fields[key] = np.asarray(fields[key], dtype=float)[ok]

    k = int(TE.size)
    if k < k_min or k < 3:
        return MetabiasResult(method_bias=method_bias, k=k, k_min=k_min)

    if method_bias in ("Begg", "Egger", "Thompson", "Macaskill",
                       "Pustejovsky") and np.unique(se).size == 1:
        raise ValueError(
            "Test for small-study effects not feasible as all studies have "
            "the same precision.")

    if method_bias in ("Begg", "Schwarzer"):
        if method_bias == "Begg":
            common = pool_yv(list(TE), list(se ** 2), None, "MD",
                             model="fixed")
            te_c = float(common.pooled_effect)
            se_c = float(common.pooled_se)
            var_s = se ** 2 - se_c ** 2
            with np.errstate(invalid="ignore", divide="ignore"):
                te_s = (TE - te_c) / np.sqrt(var_s)
            ks, se_ks, stat, pval = _kentau(te_s, se ** 2, correct=correct)
        else:
            if fields.get("single") or fields.get("event_c") is None:
                raise ValueError("method.bias='Schwarzer' requires "
                                 "metabin() data")
            from .metabin import metabin as _metabin_of

            ev_e = np.asarray(fields["event_e"], dtype=float)
            nn_e = np.asarray(fields["n_e"], dtype=float)
            ev_c = np.asarray(fields["event_c"], dtype=float)
            nn_c = np.asarray(fields["n_c"], dtype=float)
            te_mh = float(_metabin_of(ev_e, nn_e, ev_c, nn_c, sm="OR",
                                      method="MH", random=False
                                      ).fixed.pooled_est)
            n11_e = np.empty(k)
            n11_v = np.empty(k)
            for i in range(k):
                n11_e[i], n11_v[i] = _hypergeom_mean_var(
                    nn_e[i], ev_e[i] + ev_c[i], nn_e[i] + nn_c[i],
                    math.exp(te_mh))
            with np.errstate(invalid="ignore", divide="ignore"):
                te_s = (ev_e - n11_e) / np.sqrt(n11_v)
            ks, se_ks, stat, pval = _kentau(te_s, 1.0 / n11_v,
                                            correct=correct)
        label = "Rank correlation test of funnel plot asymmetry"
        if correct:
            label += " (with continuity correction)"
        return MetabiasResult(
            method_bias=method_bias, method=label, var_model="",
            statistic=float(stat), pval=float(pval), p_value=float(pval),
            estimate=[float(ks), float(se_ks)], k=k, k_min=k_min,
        )

    # --- regression-based methods ----------------------------------------
    if method_bias in ("Egger", "Thompson"):
        lreg = _linregcore(TE, se, None,
                           "lm" if method_bias == "Egger" else "rma",
                           tauf, alpha)
    elif method_bias == "Harbord":
        if fields.get("single") or fields.get("event_c") is None:
            raise ValueError("method.bias='Harbord' requires metabin() data")
        ev_e = np.asarray(fields["event_e"], dtype=float)
        nn_e = np.asarray(fields["n_e"], dtype=float)
        ev_c = np.asarray(fields["event_c"], dtype=float)
        nn_c = np.asarray(fields["n_c"], dtype=float)
        if str(fields.get("sm") or "OR").upper() == "RR":
            Z = ((ev_e * (nn_e + nn_c) - (ev_e + ev_c) * nn_e)
                 / (nn_e - ev_e + nn_c - ev_c))
            V = ((nn_e + nn_c) / 4.0 * (ev_e + ev_c)
                 / (nn_e - ev_e + nn_c - ev_c))
        else:
            Z = ev_e - (ev_e + ev_c) * nn_e / (nn_e + nn_c)
            V = (nn_e * nn_c * (ev_e + ev_c)
                 * (nn_e - ev_e + nn_c - ev_c)
                 / ((nn_e + nn_c) ** 2 * ((nn_e + nn_c) - 1.0)))
        with np.errstate(invalid="ignore", divide="ignore"):
            te_score = Z / V
            se_score = 1.0 / np.sqrt(V)
        lreg = _linregcore(te_score, se_score, None, "lm", tauf, alpha)
    elif method_bias == "Macaskill":
        if fields.get("n_e") is not None and fields.get("n_c") is not None:
            covar = (np.asarray(fields["n_e"], dtype=float)
                     + np.asarray(fields["n_c"], dtype=float))
        elif fields.get("n_e") is not None:
            covar = np.asarray(fields["n_e"], dtype=float)
        elif fields.get("n_c") is not None:
            covar = np.asarray(fields["n_c"], dtype=float)
        else:
            raise ValueError("No information on sample size available.")
        lreg = _linregcore(TE, se, covar, "lm", tauf, alpha)
    elif method_bias == "Peters":
        if fields.get("event_e") is None:
            raise ValueError("method.bias='Peters' requires metabin() or "
                             "metaprop() data")
        ev_e = np.asarray(fields["event_e"], dtype=float)
        nn_e = np.asarray(fields["n_e"], dtype=float)
        with np.errstate(invalid="ignore", divide="ignore"):
            if fields.get("single"):
                # metaprop: sqrt(1/event + 1/(n - event)), covariate 1/n
                se_peters = np.sqrt(1.0 / ev_e + 1.0 / (nn_e - ev_e))
                covar = 1.0 / nn_e
            else:
                ev_c = np.asarray(fields["event_c"], dtype=float)
                nn_c = np.asarray(fields["n_c"], dtype=float)
                se_peters = np.sqrt(1.0 / (ev_e + ev_c)
                                    + 1.0 / (nn_e - ev_e + nn_c - ev_c))
                covar = 1.0 / (nn_e + nn_c)
        lreg = _linregcore(TE, se_peters, covar, "lm", tauf, alpha)
    elif method_bias == "Deeks":
        if fields.get("single") or fields.get("n_c") is None:
            raise ValueError("method.bias='Deeks' requires metabin() data")
        nn_e = np.asarray(fields["n_e"], dtype=float)
        nn_c = np.asarray(fields["n_c"], dtype=float)
        ess = 4.0 * nn_e * nn_c / (nn_e + nn_c)
        lreg = _linregcore(TE, np.sqrt(1.0 / ess), 1.0 / np.sqrt(ess), "lm",
                           tauf, alpha)
    else:  # Pustejovsky
        if fields.get("n_e") is None:
            raise ValueError("Sample size in experimental group (n.e) "
                             "missing.")
        if fields.get("n_c") is None:
            raise ValueError("Sample size in control group (n.c) missing.")
        nn_e = np.asarray(fields["n_e"], dtype=float)
        nn_c = np.asarray(fields["n_c"], dtype=float)
        lreg = _linregcore(TE, se, np.sqrt(1.0 / nn_e + 1.0 / nn_c), "lm",
                           tauf, alpha)

    label = ("Funnel plot test for diagnostic odds ratios"
             if method_bias == "Deeks"
             else "Linear regression test of funnel plot asymmetry")
    var_model = "additive" if method_bias == "Thompson" else "multiplicative"
    return MetabiasResult(
        method_bias=method_bias, method=label, var_model=var_model,
        statistic=float(lreg["statistic"]), pval=float(lreg["pval"]),
        p_value=float(lreg["pval"]),
        estimate=[float(lreg["slope"]), float(lreg["se.slope"])],
        intercept=float(lreg["intercept"]),
        se_intercept=float(lreg["se.intercept"]),
        tau=float(lreg["tau"]), df=int(lreg["df"]), k=k, k_min=k_min,
    )
