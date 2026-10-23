"""Limit meta-analysis: an effect estimate adjusted for small-study effects.

Port of ``metasens::limitmeta`` (Rücker, Schwarzer, Carpenter, Binder and
Schumacher 2011).  The model behind it allows the effect to depend on the
standard error:

.. math::

    \\theta_i = \\beta + \\sqrt{se_i^2 + \\tau^2}\\;(\\varepsilon_i + \\alpha),
    \\qquad \\varepsilon_i \\sim N(0, 1)

so :math:`\\alpha` is the bias induced by small-study effects and
:math:`\\beta + \\tau\\alpha` is the effect expected at infinite precision —
the *limit* effect.  Dividing by :math:`s_i = \\sqrt{se_i^2 + \\tau^2}` turns
the likelihood into a straight line,

.. math::

    \\frac{\\theta_i}{s_i} = \\alpha + \\beta\\,\\frac{1}{s_i} + \\varepsilon_i,

which is the generalised radial plot: regress the standardised effect on the
inverse of its total standard error.  The intercept is the bias parameter and
the slope is the limit effect.

Everything here is closed form given :math:`\\tau`, which is why the routine is
short: no optimiser is needed, and the standard errors are the ordinary
least-squares formulas — valid as they stand because the residuals of the
standardised regression have variance exactly 1 by construction.

Which adjusted estimate is reported depends on ``method_adjust``:

===========================  ==========================================
``"beta0"`` (default)        :math:`\\hat\\beta + \\hat\\tau\\hat\\alpha`,
                             the expectation of the extended model
``"betalim"``                slope of the radial plot of the
                             shrunken estimates
``"mulim"``                  common-effect estimate of the shrunken
                             estimates
===========================  ==========================================
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
from scipy import stats

__all__ = ["LimitmetaResult", "limitmeta", "radial_regression"]

_ADJUST = ("beta0", "betalim", "mulim")


@dataclass
class RadialRegression:
    """Ordinary least squares on the generalised radial plot.

    Attributes:
        intercept: :math:`\\hat\\alpha`, the small-study bias parameter.
        slope: :math:`\\hat\\beta`, the effect at infinite precision.
        se_intercept, se_slope: OLS standard errors.  The residual variance is
            fixed at 1 rather than estimated, because the standardised effects
            have unit variance by construction.
        sigma: residual standard deviation of the fitted line.
        r_squared: coefficient of determination.
        k: number of studies.
    """

    intercept: float
    slope: float
    se_intercept: float
    se_slope: float
    sigma: float
    r_squared: float
    k: int


def radial_regression(yi, se):
    """Fit ``yi/se ~ 1/se``, the generalised radial plot.

    Args:
        yi: effect estimates.
        se: standard errors (or total standard errors including
            heterogeneity).

    Returns:
        :class:`RadialRegression`.

    Raises:
        ValueError: fewer than three studies, a non-positive standard error,
            or no variation in precision (the line is then not identified).
    """
    yi = np.asarray(yi, dtype=float).ravel()
    se = np.asarray(se, dtype=float).ravel()
    k = yi.size
    if se.size != k:
        raise ValueError("'yi' and 'se' must be the same length")
    if k < 3:
        raise ValueError("at least three studies are needed")
    if np.any(se <= 0.0):
        raise ValueError("standard errors must be positive")
    x = 1.0 / se
    y = yi / se
    if np.ptp(x) <= 0.0:
        raise ValueError(
            "all studies have the same precision, so the radial plot has no "
            "slope to estimate")
    xbar = float(np.mean(x))
    ssx = float(np.sum((x - xbar) ** 2))
    ybar = float(np.mean(y))
    ssxy = float(np.sum((x - xbar) * (y - ybar)))
    slope = ssxy / ssx
    intercept = ybar - slope * xbar
    fitted = intercept + slope * x
    resid = y - fitted
    sse = float(resid @ resid)
    # the residual variance is 1 by construction; the OLS formulas follow
    se_slope = 1.0 / math.sqrt(ssx)
    se_intercept = se_slope * float(np.mean(x * x))
    sst = float(np.sum((y - ybar) ** 2))
    r_squared = 0.0 if sst <= 0.0 else 1.0 - sse / sst
    dof = k - 2
    sigma = math.sqrt(sse / dof) if dof > 0 else float("nan")
    return RadialRegression(intercept=intercept, slope=slope,
                            se_intercept=se_intercept, se_slope=se_slope,
                            sigma=sigma, r_squared=r_squared, k=k)


@dataclass
class LimitmetaResult:
    """Adjusted effect estimate from a limit meta-analysis.

    Attributes:
        method_adjust: which adjusted estimate was reported.
        TE, seTE: the studies' effects and standard errors.
        TE_limit, seTE_limit: shrunken per-study estimates.
        TE_adjust, seTE_adjust: the adjusted overall effect and its SE.
        ci_lb, ci_ub, zval, pval: inference on the adjusted effect.
        alpha, beta: intercept and slope of the radial plot — the bias
            parameter and the effect at infinite precision.
        tau: between-study standard deviation used for the shrinkage.
        TE_random, seTE_random: the unadjusted random-effects estimate.
        Q, df_Q: Cochran's Q of the unadjusted model.
        Q_small: heterogeneity attributed to small-study effects.
        Q_resid: heterogeneity remaining beyond them.
        G_squared: proportion of the radial-plot variation left unexplained
            by the shrunken model.
        k, level: number of studies and the confidence level in percent.
        slab: study labels, when the input model had them.
    """

    method_adjust: str
    TE: np.ndarray = field(repr=False)
    seTE: np.ndarray = field(repr=False)
    TE_limit: np.ndarray = field(repr=False)
    seTE_limit: np.ndarray = field(repr=False)
    TE_adjust: float
    seTE_adjust: float
    ci_lb: float = field(repr=False)
    ci_ub: float = field(repr=False)
    zval: float = field(repr=False)
    pval: float = field(repr=False)
    alpha: float
    beta: float
    tau: float
    TE_random: float
    seTE_random: float
    Q: float
    df_Q: int
    Q_small: float
    Q_resid: float
    G_squared: float
    k: int
    level: float = 95.0
    slab: list[str] | None = field(repr=False, default=None)

    def coef_table(self) -> list[dict]:
        """The adjusted estimate as a one-row table."""
        return [{
            "estimate": self.TE_adjust,
            "se": self.seTE_adjust,
            "zval": self.zval,
            "pval": self.pval,
            "ci_lb": self.ci_lb,
            "ci_ub": self.ci_ub,
            "tau": self.tau,
            "alpha": self.alpha,
            "method": self.method_adjust,
        }]

    def to_dict(self) -> dict:
        return {
            "method_adjust": self.method_adjust,
            "TE": self.TE.tolist(),
            "seTE": self.seTE.tolist(),
            "TE_limit": self.TE_limit.tolist(),
            "seTE_limit": self.seTE_limit.tolist(),
            "TE_adjust": self.TE_adjust,
            "seTE_adjust": self.seTE_adjust,
            "ci_lb": self.ci_lb,
            "ci_ub": self.ci_ub,
            "zval": self.zval,
            "pval": self.pval,
            "alpha": self.alpha,
            "beta": self.beta,
            "tau": self.tau,
            "TE_random": self.TE_random,
            "seTE_random": self.seTE_random,
            "Q": self.Q,
            "df_Q": self.df_Q,
            "Q_small": self.Q_small,
            "Q_resid": self.Q_resid,
            "G_squared": self.G_squared,
            "k": self.k,
            "level": self.level,
            "slab": self.slab,
        }

    def __str__(self) -> str:
        crit = stats.norm.ppf(0.5 + self.level / 200.0)
        return (
            f"limit meta-analysis (adjust={self.method_adjust})\n"
            f"  studies        {self.k}\n"
            f"  random effects {self.TE_random:+.4f} "
            f"[{self.TE_random - crit * self.seTE_random:+.4f}, "
            f"{self.TE_random + crit * self.seTE_random:+.4f}]\n"
            f"  adjusted       {self.TE_adjust:+.4f} "
            f"[{self.ci_lb:+.4f}, {self.ci_ub:+.4f}]\n"
            f"  bias alpha     {self.alpha:+.4f}   limit beta {self.beta:+.4f}\n"
            f"  tau            {self.tau:.4f}\n"
            f"  Q {self.Q:.4f} = Q.small {self.Q_small:.4f} + "
            f"Q.resid {self.Q_resid:.4f}"
        )


def _as_rma(result):
    """Accept an ``RmaResult`` or a ``meta``-style result carrying one."""
    from .rma_uni import RmaResult

    if isinstance(result, RmaResult):
        return result
    inner = getattr(result, "random", None)
    if isinstance(inner, RmaResult):
        return inner
    raise TypeError(
        "argument 'result' must be a fitted RmaResult (or a meta-analysis "
        "result whose '.random' is one)")


def limitmeta(result, method_adjust="beta0", level=None):
    """Adjust a pooled effect for small-study effects (Rücker et al. 2011).

    Args:
        result: fitted :class:`~pimeta.rma_uni.RmaResult`, or a meta-analysis
            result holding one in ``.random``.  Its :math:`\\tau^2` — from
            whichever estimator was used — drives the shrinkage, so it must be
            an intercept-only random-effects fit.
        method_adjust: ``"beta0"`` (default), ``"betalim"`` or ``"mulim"``.
        level: confidence level, either as a percentage (``95``) or a
            proportion (``0.95``).  Defaults to the level of the input model.
            (The package is not consistent here — ``RmaResult.level`` is a
            proportion while ``rma_mv(level=)`` is a percentage — so both are
            accepted rather than one of them silently misread.)

    Returns:
        :class:`LimitmetaResult`.

    Raises:
        TypeError: the argument is not a fitted model.
        ValueError: the model is not intercept-only, fewer than three studies,
            or an unknown ``method_adjust``.
    """
    fit = _as_rma(result)
    adj = str(method_adjust).lower()
    if adj not in _ADJUST:
        raise ValueError(
            f"method_adjust must be one of {_ADJUST}, got {method_adjust!r}")
    if not getattr(fit, "int_only", True):
        raise ValueError(
            "limit meta-analysis needs an intercept-only random-effects "
            "model; the extended model has no covariates")

    yi = np.asarray(fit.yi, dtype=float).ravel()
    vi = np.asarray(fit.vi, dtype=float).ravel()
    se = np.sqrt(vi)
    k = int(fit.k)
    if k < 3:
        raise ValueError("at least three studies are needed")
    tau = math.sqrt(max(float(fit.tau2), 0.0))
    if level is None:
        level = float(getattr(fit, "level", 95.0))
    level = float(level)
    if level <= 1.0:
        level *= 100.0

    # total standard error and the random-effects weights it implies
    se_tau = np.sqrt(vi + tau * tau)
    w_random = 1.0 / se_tau ** 2

    # radial plot of the random-effects model: intercept = bias, slope = limit
    reg_r = radial_regression(yi, se_tau)
    alpha_r = reg_r.intercept
    beta_r = reg_r.slope

    # shrunken per-study estimates: pull each study towards beta by tau/s_i
    TE_limit = beta_r + (tau / se_tau) * (yi - beta_r)
    seTE_limit = se.copy()

    w_fe = 1.0 / vi
    te_common = float(np.sum(w_fe * TE_limit) / np.sum(w_fe))
    se_common = float(1.0 / math.sqrt(np.sum(w_fe)))
    reg_l = radial_regression(TE_limit, se)

    if adj == "beta0":
        TE_adjust = beta_r + tau * alpha_r
        x = np.sqrt(w_random)
        var_x = float(np.var(x, ddof=1))
        if var_x <= 0.0:
            raise ValueError("all studies have the same precision")
        var_beta = 1.0 / var_x / (k - 1)
        var_alpha = float(np.mean(x * x)) / var_x / (k - 1)
        cov_ab = -float(np.mean(x)) / var_x / (k - 1)
        var_adj = var_beta + tau * tau * var_alpha + 2.0 * tau * cov_ab
        se_adjust = math.sqrt(var_adj) if var_adj > 0.0 else float("nan")
    elif adj == "betalim":
        TE_adjust = reg_l.slope
        se_adjust = reg_l.se_slope
    else:                                    # mulim
        TE_adjust = te_common
        se_adjust = se_common

    crit = stats.norm.ppf(0.5 + level / 200.0)
    zval = TE_adjust / se_adjust if se_adjust > 0.0 else float("nan")
    pval = float(2.0 * stats.norm.sf(abs(zval))) if np.isfinite(zval) \
        else float("nan")
    ci_lb = TE_adjust - crit * se_adjust
    ci_ub = TE_adjust + crit * se_adjust

    # heterogeneity decomposition (Rücker et al. 2011, section 7)
    reg_f = radial_regression(yi, se)
    Q_resid = reg_f.sigma ** 2 * (k - 2)
    Q = float(fit.QE)
    Q_small = Q - Q_resid
    G_squared = 1.0 - reg_l.r_squared

    slab = list(fit.slab) if getattr(fit, "slab", None) else None
    return LimitmetaResult(
        method_adjust=adj,
        TE=yi, seTE=se, TE_limit=TE_limit, seTE_limit=seTE_limit,
        TE_adjust=float(TE_adjust), seTE_adjust=float(se_adjust),
        ci_lb=float(ci_lb), ci_ub=float(ci_ub),
        zval=float(zval), pval=pval,
        alpha=float(alpha_r), beta=float(beta_r), tau=float(tau),
        TE_random=float(fit.b[0]), seTE_random=float(fit.se[0]),
        Q=Q, df_Q=k - 1, Q_small=float(Q_small), Q_resid=float(Q_resid),
        G_squared=float(G_squared), k=k, level=float(level), slab=slab,
    )
