"""selmodel — a faithful transcription of ``metafor::selmodel`` (selection models).

Implements the full family of publication-bias selection models on top of an
``rma`` fit: ``beta``, ``halfnorm``/``halfnorm1``/``halfnorm2``, ``negexp``/
``negexp1``/``negexp2``, ``logistic``/``logistic1``/``logistic2``, ``power``/
``power1``/``power2``, ``negexppow``, ``stepfun`` (incl. decreasing) and
``trunc``/``truncest``.

The likelihood is maximised numerically (BFGS) exactly as metafor does; the
Hessian is obtained by Richardson-extrapolated central differences (mirroring
``numDeriv::hessian`` with ``r=6``). Only NumPy/SciPy are used.
"""

from __future__ import annotations

import math
import warnings
from dataclasses import dataclass, field
from typing import cast

import numpy as np
from scipy import integrate, optimize, special, stats

from .rma_uni import RmaResult, rma

_EPS = np.finfo(float).eps

TYPE_OPTIONS = [
    "beta",
    "halfnorm", "negexp", "logistic", "power", "negexppow",
    "halfnorm1", "negexp1", "logistic1", "power1",
    "halfnorm2", "negexp2", "logistic2", "power2",
    "stepfun", "stepcon", "trunc", "truncest",
]


@dataclass
class SelModelResult:
    """Fitted ``rma.uni.selmodel`` model, mirroring the R ``$`` field set."""

    beta: list[float]
    b: list[float]
    se: list[float]
    zval: list[float]
    pval: list[float]
    ci_lb: list[float]
    ci_ub: list[float]
    vb: np.ndarray
    tau2: float
    se_tau2: float
    ci_lb_tau2: float
    ci_ub_tau2: float
    delta: list[float]
    vd: np.ndarray
    se_delta: list[float]
    zval_delta: list[float]
    pval_delta: list[float]
    ci_lb_delta: list[float]
    ci_ub_delta: list[float]
    QM: float
    QMp: float
    ll: float
    ll0: float
    LRT: float
    LRTdf: int
    LRTp: float
    LRT_tau2: float
    LRTp_tau2: float
    parms: int
    fit_stats: dict
    pvals: list[float]
    type: str
    alternative: str
    steps: list[float]
    pgrp: list
    k0: int
    k1: int
    model: str
    k: int
    p: int
    tau2_fix: bool
    method: str
    yi: list[float]
    vi: list[float]
    level: float

    def to_dict(self) -> dict:
        return {
            "beta": self.beta,
            "b": self.b,
            "se": self.se,
            "zval": self.zval,
            "pval": self.pval,
            "ci.lb": self.ci_lb,
            "ci.ub": self.ci_ub,
            "tau2": self.tau2,
            "se.tau2": self.se_tau2,
            "delta": self.delta,
            "se.delta": self.se_delta,
            "zval.delta": self.zval_delta,
            "pval.delta": self.pval_delta,
            "ci.lb.delta": self.ci_lb_delta,
            "ci.ub.delta": self.ci_ub_delta,
            "QM": self.QM,
            "QMp": self.QMp,
            "ll": self.ll,
            "ll0": self.ll0,
            "LRT": self.LRT,
            "LRTdf": self.LRTdf,
            "LRTp": self.LRTp,
            "LRT.tau2": self.LRT_tau2,
            "LRTp.tau2": self.LRTp_tau2,
            "parms": self.parms,
            "type": self.type,
            "alternative": self.alternative,
            "steps": self.steps,
        }


# --------------------------------------------------------------------------- #
# Small numeric helpers transcribed from metafor internals
# --------------------------------------------------------------------------- #

def _plogis(x):
    return special.expit(x)


def _qlogis(x):
    return special.logit(x)


def _is_dummy(x, eps=1e-8):
    return np.all((np.abs(x) < eps) | (np.abs(x - 1) < eps))


def _mapfun(x, lb, ub, fun=None):
    """metafor's ``.mapfun`` — forward map from optimisation scale."""
    if fun is None:
        if lb == 0.0 and ub == 1.0:
            y = _plogis(x)
        else:
            y = lb + (ub - lb) / (1.0 + np.exp(-x))
    else:
        if fun == "exp":
            y = np.exp(x)
        elif fun == "I":
            y = x
        elif fun == "plogis":
            y = _plogis(x)
        else:
            y = x
        y = np.minimum(np.maximum(y, lb), ub)
    return y


def _mapinvfun(x, lb, ub, fun=None):
    """metafor's ``.mapinvfun`` — inverse map onto the optimisation scale."""
    if fun is None:
        if lb == 0.0 and ub == 1.0:
            return _qlogis(x)
        return np.log((x - lb) / (ub - x))
    if fun == "log":
        return np.log(x)
    if fun == "I":
        return x
    if fun == "qlogis":
        return _qlogis(x)
    return x


def _selmodel_pval(yi, vi, alternative):
    """metafor's ``.selmodel.pval``."""
    zi = yi / np.sqrt(vi)
    if alternative == "two.sided":
        return 2.0 * stats.norm.sf(np.abs(zi))
    if alternative == "less":
        return stats.norm.cdf(zi)
    return stats.norm.sf(zi)


def _wi_fun(type_, x, delta, yi, vi, preci, alternative, steps):
    """Selection weight function ``wi.fun`` for every supported type.

    ``x`` is a p-value (scalar or array); ``delta`` the selection parameter
    vector on the natural scale; ``steps`` a 1-D array (length 1 or 2 for the
    smooth models, or the full vector for ``stepfun``).
    """
    x = np.asarray(x, dtype=float)
    if type_ == "beta":
        z = np.clip(x, steps[0], steps[1])
        return z ** (delta[0] - 1.0) * (1.0 - z) ** (delta[1] - 1.0)

    if type_ in ("halfnorm", "negexp", "logistic", "power"):
        s0 = steps[0]
        if type_ == "halfnorm":
            return np.where(x <= s0, 1.0, np.exp(-delta[0] * preci * x ** 2) / np.exp(-delta[0] * preci * s0 ** 2))
        if type_ == "negexp":
            return np.where(x <= s0, 1.0, np.exp(-delta[0] * preci * x) / np.exp(-delta[0] * preci * s0))
        if type_ == "logistic":
            num = 2.0 * np.exp(-delta[0] * preci * x) / (1.0 + np.exp(-delta[0] * preci * x))
            den = 2.0 * np.exp(-delta[0] * preci * s0) / (1.0 + np.exp(-delta[0] * preci * s0))
            return np.where(x <= s0, 1.0, num / den)
        # power
        return np.where(x <= s0, 1.0, (1.0 - x) ** (preci * delta[0]) / (1.0 - s0) ** (preci * delta[0]))

    if type_ in ("halfnorm1", "negexp1", "logistic1", "power1"):
        s0 = steps[0]
        t = (x - s0) / (1.0 - s0)
        if type_ == "halfnorm1":
            return np.where(x <= s0, 1.0, np.exp(-delta[0] * preci * t ** 2))
        if type_ == "negexp1":
            return np.where(x <= s0, 1.0, np.exp(-delta[0] * preci * t))
        if type_ == "logistic1":
            return np.where(x <= s0, 1.0, 2.0 * np.exp(-delta[0] * preci * t) / (1.0 + np.exp(-delta[0] * preci * t)))
        # power1
        return np.where(x <= s0, 1.0, (1.0 - t) ** (preci * delta[0]))

    if type_ == "negexppow":
        s0 = steps[0]
        return np.where(
            x <= s0, 1.0,
            np.exp(-delta[0] * preci * x ** (1.0 / delta[1]))
            / np.exp(-delta[0] * preci * s0 ** (1.0 / delta[1])),
        )

    if type_ in ("halfnorm2", "negexp2", "logistic2", "power2"):
        s0 = steps[0]
        if type_ == "halfnorm2":
            inner = np.exp(-delta[1] * preci * x ** 2) / np.exp(-delta[1] * preci * s0 ** 2)
        elif type_ == "negexp2":
            inner = np.exp(-delta[1] * preci * x) / np.exp(-delta[1] * preci * s0)
        elif type_ == "logistic2":
            num = 2.0 * np.exp(-delta[1] * preci * x) / (1.0 + np.exp(-delta[1] * preci * x))
            den = 2.0 * np.exp(-delta[1] * preci * s0) / (1.0 + np.exp(-delta[1] * preci * s0))
            inner = num / den
        else:  # power2
            inner = (1.0 - x) ** (preci * delta[1]) / (1.0 - s0) ** (preci * delta[1])
        return np.where(x <= s0, 1.0, (delta[0] + inner) / (1.0 + delta[0]))

    if type_ in ("stepfun", "stepcon"):
        idx = np.searchsorted(steps, x, side="left")
        return delta[idx] / preci

    if type_ in ("trunc", "truncest"):
        sd = np.sqrt(vi)
        cutoff = steps[0] if type_ == "trunc" else delta[1]
        if alternative == "less":
            yival = stats.norm.ppf(x, scale=sd)
            return np.where(yival < cutoff, 1.0, delta[0])
        yival = stats.norm.isf(x, scale=sd)
        return np.where(yival > cutoff, 1.0, delta[0])

    raise ValueError(f"unsupported selection model type: {type_!r}")


# --------------------------------------------------------------------------- #
# Log-likelihood builders
# --------------------------------------------------------------------------- #

def _clamp_tau2(tau2, tau2_max):
    if tau2 < _EPS * 10.0:
        tau2 = 0.0
    if tau2 > tau2_max:
        tau2 = tau2_max
    return tau2


def _transform_delta(delta, deltas, delta_min, delta_max, mapfun, decreasing):
    """Apply the forward delta map (``delta.transf=TRUE`` path)."""
    if decreasing:
        d = np.exp(delta)
        d = np.cumsum(np.concatenate([[0.0], -d[1:]]))
        return np.exp(d)
    out = np.array([
        _mapfun(delta[j], delta_min[j], delta_max[j], mapfun[j])
        for j in range(deltas)
    ])
    return out


def _gh_rule(n=200):
    """Gauss-Hermite rule for ``E[f(U)]`` with ``U ~ N(0, 1)``."""
    nodes, weights = np.polynomial.hermite_e.hermegauss(int(n))
    return nodes, weights / math.sqrt(2.0 * math.pi)


_GH_RULE_CACHE = {}


def _gh():
    if 200 not in _GH_RULE_CACHE:
        _GH_RULE_CACHE[200] = _gh_rule(200)
    return _GH_RULE_CACHE[200]


def _wi_array(type_, x, delta, preci, steps):
    """Vectorised :func:`_wi_scalar` (``x`` and ``preci`` may broadcast)."""
    x = np.asarray(x, dtype=float)
    preci = np.asarray(preci, dtype=float)
    s0 = float(steps[0])

    with np.errstate(over="ignore", divide="ignore", invalid="ignore",
                     under="ignore"):
        if type_ == "beta":
            z = np.clip(x, steps[0], steps[1])
            return z ** (delta[0] - 1.0) * (1.0 - z) ** (delta[1] - 1.0)

        tiny = np.finfo(float).tiny
        if type_ == "halfnorm":
            num = np.exp(-delta[0] * preci * x * x)
            den = np.exp(-delta[0] * preci * s0 * s0)
        elif type_ == "negexp":
            num = np.exp(-delta[0] * preci * x)
            den = np.exp(-delta[0] * preci * s0)
        elif type_ == "logistic":
            num = (2.0 * np.exp(-delta[0] * preci * x)
                   / (1.0 + np.exp(-delta[0] * preci * x)))
            den = (2.0 * np.exp(-delta[0] * preci * s0)
                   / (1.0 + np.exp(-delta[0] * preci * s0)))
        elif type_ == "power":
            num = (1.0 - x) ** (preci * delta[0])
            den = (1.0 - s0) ** (preci * delta[0])
        elif type_ in ("halfnorm1", "negexp1", "logistic1", "power1"):
            t = (x - s0) / (1.0 - s0)
            if type_ == "halfnorm1":
                num, den = np.exp(-delta[0] * preci * t * t), 1.0
            elif type_ == "negexp1":
                num, den = np.exp(-delta[0] * preci * t), 1.0
            elif type_ == "logistic1":
                num = 2.0 * np.exp(-delta[0] * preci * t) / (
                    1.0 + np.exp(-delta[0] * preci * t))
                den = 1.0
            else:
                num, den = (1.0 - t) ** (preci * delta[0]), 1.0
        elif type_ == "negexppow":
            num = np.exp(-delta[0] * preci * x ** (1.0 / delta[1]))
            den = np.exp(-delta[0] * preci * s0 ** (1.0 / delta[1]))
        elif type_ in ("halfnorm2", "negexp2", "logistic2", "power2"):
            if type_ == "halfnorm2":
                inner = (np.exp(-delta[1] * preci * x * x)
                         / np.exp(-delta[1] * preci * s0 * s0))
            elif type_ == "negexp2":
                inner = (np.exp(-delta[1] * preci * x)
                         / np.exp(-delta[1] * preci * s0))
            elif type_ == "logistic2":
                num = (2.0 * np.exp(-delta[1] * preci * x)
                       / (1.0 + np.exp(-delta[1] * preci * x)))
                den_ = (2.0 * np.exp(-delta[1] * preci * s0)
                        / (1.0 + np.exp(-delta[1] * preci * s0)))
                inner = num / den_
            else:
                inner = ((1.0 - x) ** (preci * delta[1])
                         / (1.0 - s0) ** (preci * delta[1]))
            return (delta[0] + inner) / (1.0 + delta[0])
        else:
            raise ValueError(f"unsupported selection model type: {type_!r}")

        with np.errstate(divide="ignore", invalid="ignore"):
            out = num / den
        out = np.where(den == 0.0, np.where(num == 0.0, 1.0, np.inf), out)
    out = np.where(x <= s0, 1.0, out) if type_ != "beta" else out
    return np.maximum(out, tiny)


def _make_ll_cont(yi, vi, X, preci, subset, p, pvals, deltas, delta_arg,
                  mapfun, delta_min, delta_max, decreasing, delta_transf,
                  tau2_arg, tau2_transf, tau2_max, beta_arg, type_, steps,
                  alternative, pval_min, dofit=False):
    """``.selmodel.ll.cont`` — numeric integration over the density.

    The per-study integrals use adaptive quadrature with exactly the tolerances
    metafor passes to R's ``integrate()`` (``rel.tol = .Machine$double.eps^0.25``,
    ``subdivisions = 100``), so the numerical accuracy — and hence the point at
    which the optimiser stops — matches the reference implementation.

    A fixed Gauss-Hermite rule would be two orders of magnitude faster, but the
    selection weights ``exp(-delta * preci * p^2)`` are step functions once
    ``delta`` grows (the exponent reaches -600 on the realistic fits), and a
    200-node rule loses 5e-4 of relative accuracy there — enough to move the
    optimum more than R's own quadrature does.  Accuracy wins over speed.
    """
    k = len(yi)
    sub_idx = np.where(subset)[0]
    nsub_idx = np.where(~subset)[0]
    # R: integrate(..., rel.tol = .Machine$double.eps^0.25, subdivisions = 100)
    _quad_tol = float(_EPS ** 0.25)

    def ll(par):
        beta = np.array(par[0:p], dtype=float)
        tau2 = float(par[p])
        delta = np.array(par[p + 1:p + 1 + deltas], dtype=float)

        beta = np.where(np.isnan(beta_arg), beta, beta_arg)
        if tau2_transf:
            tau2 = np.exp(tau2)
        tau2 = np.where(~np.isnan(tau2_arg), tau2_arg, tau2)
        tau2 = _clamp_tau2(float(tau2), tau2_max)

        if delta.shape == ():
            delta = np.array([float(delta)])
        if delta_transf:
            delta = _transform_delta(delta, deltas, delta_min, delta_max, mapfun, decreasing)
        delta = np.where(np.isnan(delta_arg), delta, delta_arg)

        yhat = X @ beta

        Ai = np.zeros(k)
        for i in sub_idx:
            Ai[i], _ = integrate.quad(
                lambda yv: _int_fn(
                    yv, vi[i], preci[i], yhat[i], type_, delta, tau2,
                    alternative, pval_min, steps,
                ),
                -np.inf, np.inf,
                epsabs=_quad_tol, epsrel=_quad_tol, limit=100,
            )

        llval0 = float(np.sum(
            stats.norm.logpdf(yi[nsub_idx], yhat[nsub_idx], np.sqrt(vi[nsub_idx] + tau2))
        ))
        wi = _wi_fun(type_, pvals[sub_idx], delta, yi[sub_idx], vi[sub_idx],
                     preci[sub_idx], alternative, steps)
        # log(0) -> -Inf is exactly what metafor produces here (and what the
        # optimiser has to cope with); silence NumPy's divide-by-zero warning.
        with np.errstate(divide="ignore", invalid="ignore"):
            llval1 = float(np.sum(
                np.log(wi)
                + stats.norm.logpdf(yi[sub_idx], yhat[sub_idx], np.sqrt(vi[sub_idx] + tau2))
                - np.log(Ai[sub_idx])
            ))
        llval = llval0 + llval1

        if dofit:
            return {"ll": llval, "beta": beta, "tau2": tau2, "delta": delta}
        return -llval

    return ll


def _stepcon_ineq(par, p, deltas, delta_arg, delta_min, delta_max, mapfun):
    """metafor's ``.rma.selmodel.ineqfun.pos`` for ``type="stepcon"``.

    Returns ``-diff(delta)``, constrained to ``[0, 1]``. Since every ``delta``
    already lives in ``[0, 1]`` the upper bound is implied, so SciPy only needs
    the ``>= 0`` half (monotone non-increasing step weights).
    """
    d = np.asarray(par[p + 1:p + 1 + deltas], dtype=float)
    d = _transform_delta(d, deltas, delta_min, delta_max, mapfun, False)
    d = np.where(np.isnan(delta_arg), d, delta_arg)
    return -np.diff(d)


def _wi_scalar(type_, x, delta, preci, steps):
    """Scalar (fast) counterpart of :func:`_wi_fun` for the continuous models.

    Mirrors the exact floating-point expressions metafor evaluates in
    ``wi.fun`` (i.e. the same ratios of exponentials, not the algebraically
    simplified form), so that the optimiser sees the same surface.
    """
    s0 = steps[0]
    if type_ == "beta":
        z = np.clip(x, steps[0], steps[1])
        return float(z) ** (delta[0] - 1.0) * (1.0 - float(z)) ** (delta[1] - 1.0)

    if x <= s0:
        return 1.0

    if type_ == "halfnorm":
        return math.exp(-delta[0] * preci * x * x) / math.exp(-delta[0] * preci * s0 * s0)
    if type_ == "negexp":
        return math.exp(-delta[0] * preci * x) / math.exp(-delta[0] * preci * s0)
    if type_ == "logistic":
        num = 2.0 * math.exp(-delta[0] * preci * x) / (1.0 + math.exp(-delta[0] * preci * x))
        den = 2.0 * math.exp(-delta[0] * preci * s0) / (1.0 + math.exp(-delta[0] * preci * s0))
        return num / den
    if type_ == "power":
        return (1.0 - x) ** (preci * delta[0]) / (1.0 - s0) ** (preci * delta[0])

    if type_ in ("halfnorm1", "negexp1", "logistic1", "power1"):
        t = (x - s0) / (1.0 - s0)
        if type_ == "halfnorm1":
            return math.exp(-delta[0] * preci * t * t)
        if type_ == "negexp1":
            return math.exp(-delta[0] * preci * t)
        if type_ == "logistic1":
            return 2.0 * math.exp(-delta[0] * preci * t) / (
                1.0 + math.exp(-delta[0] * preci * t)
            )
        return (1.0 - t) ** (preci * delta[0])

    if type_ == "negexppow":
        return math.exp(-delta[0] * preci * x ** (1.0 / delta[1])) / math.exp(
            -delta[0] * preci * s0 ** (1.0 / delta[1])
        )

    if type_ in ("halfnorm2", "negexp2", "logistic2", "power2"):
        if type_ == "halfnorm2":
            inner = math.exp(-delta[1] * preci * x * x) / math.exp(
                -delta[1] * preci * s0 * s0
            )
        elif type_ == "negexp2":
            inner = math.exp(-delta[1] * preci * x) / math.exp(
                -delta[1] * preci * s0
            )
        elif type_ == "logistic2":
            num = 2.0 * math.exp(-delta[1] * preci * x) / (
                1.0 + math.exp(-delta[1] * preci * x)
            )
            den = 2.0 * math.exp(-delta[1] * preci * s0) / (
                1.0 + math.exp(-delta[1] * preci * s0)
            )
            inner = num / den
        else:  # power2
            inner = (1.0 - x) ** (preci * delta[1]) / (1.0 - s0) ** (preci * delta[1])
        return (delta[0] + inner) / (1.0 + delta[0])

    raise ValueError(f"unsupported selection model type: {type_!r}")


def _int_fn(yval, vi, preci, yhat, type_, delta, tau2, alternative, pval_min, steps):
    """``.selmodel.int`` integrand.

    Implemented with plain floats + ``scipy.special.ndtr``; the adaptive
    quadrature calls this function tens of thousands of times per fit, and the
    array-based path dominated the runtime.
    """
    if alternative == "two.sided":
        pv = 2.0 * special.ndtr(-abs(yval / math.sqrt(vi)))
    elif alternative == "less":
        pv = special.ndtr(yval / math.sqrt(vi))
    else:
        pv = special.ndtr(-yval / math.sqrt(vi))

    if pval_min > 0.0:
        if pv < pval_min:
            pv = pval_min
        elif pv > 1.0 - pval_min:
            pv = 1.0 - pval_min

    w = _wi_scalar(type_, pv, delta, preci, steps)
    v = vi + tau2
    dens = math.exp(-0.5 * (yval - yhat) ** 2 / v) / math.sqrt(2.0 * math.pi * v)
    return w * dens


def _make_ll_stepfun(yi, vi, X, preci, subset, p, pvals, deltas, delta_arg,
                     mapfun, delta_min, delta_max, decreasing, delta_transf,
                     tau2_arg, tau2_transf, tau2_max, beta_arg, type_, steps,
                     alternative, pval_min, dofit=False):
    """``.selmodel.ll.stepfun`` — closed-form (analytic) integrals."""
    k = len(yi)
    sub_idx = np.where(subset)[0]
    nsub_idx = np.where(~subset)[0]
    N = len(steps)
    pgrp = np.searchsorted(steps, pvals, side="left")

    def ll(par):
        beta = np.array(par[0:p], dtype=float)
        tau2 = float(par[p])
        delta = np.array(par[p + 1:p + 1 + deltas], dtype=float)

        beta = np.where(np.isnan(beta_arg), beta, beta_arg)
        if tau2_transf:
            tau2 = np.exp(tau2)
        tau2 = np.where(~np.isnan(tau2_arg), tau2_arg, tau2)
        tau2 = _clamp_tau2(float(tau2), tau2_max)

        if delta.shape == ():
            delta = np.array([float(delta)])
        if decreasing:
            if delta_transf:
                d = np.exp(delta)
                d = np.cumsum(np.concatenate([[0.0], -d[1:]]))
                delta = np.exp(d)
            if np.any(~np.isnan(delta_arg)):
                delta = np.maximum.accumulate(delta[::-1])[::-1]
        else:
            if delta_transf:
                delta = _transform_delta(delta, deltas, delta_min, delta_max, mapfun, decreasing)
        delta = np.where(np.isnan(delta_arg), delta, delta_arg)

        yhat = X @ beta
        sei = np.sqrt(vi + tau2)

        Ai = np.zeros(k)
        for i in sub_idx:
            sd = np.sqrt(vi[i])
            if alternative == "greater":
                ys = sd * stats.norm.isf(steps)
                Ai[i] = stats.norm.sf(ys[0], yhat[i], sei[i])
                for j in range(1, N):
                    if j < N - 1:
                        Ai[i] += delta[j] / preci[i] * (
                            stats.norm.sf(ys[j], yhat[i], sei[i])
                            - stats.norm.sf(ys[j - 1], yhat[i], sei[i])
                        )
                    else:
                        Ai[i] += delta[j] / preci[i] * stats.norm.cdf(ys[j - 1], yhat[i], sei[i])
            elif alternative == "less":
                ys = sd * stats.norm.ppf(steps)
                Ai[i] = stats.norm.cdf(ys[0], yhat[i], sei[i])
                for j in range(1, N):
                    if j < N - 1:
                        Ai[i] += delta[j] / preci[i] * (
                            stats.norm.cdf(ys[j], yhat[i], sei[i])
                            - stats.norm.cdf(ys[j - 1], yhat[i], sei[i])
                        )
                    else:
                        Ai[i] += delta[j] / preci[i] * stats.norm.sf(ys[j - 1], yhat[i], sei[i])
            else:  # two.sided
                half = steps / 2.0
                yu = sd * stats.norm.isf(half)
                yl = sd * stats.norm.ppf(half)
                Ai[i] = (stats.norm.sf(yu[0], yhat[i], sei[i])
                         + stats.norm.cdf(yl[0], yhat[i], sei[i]))
                for j in range(1, N):
                    if j < N - 1:
                        Ai[i] += delta[j] / preci[i] * (
                            (stats.norm.sf(yu[j], yhat[i], sei[i])
                             - stats.norm.sf(yu[j - 1], yhat[i], sei[i]))
                            + (stats.norm.cdf(yl[j], yhat[i], sei[i])
                               - stats.norm.cdf(yl[j - 1], yhat[i], sei[i]))
                        )
                    else:
                        Ai[i] += delta[j] / preci[i] * (
                            stats.norm.cdf(yu[j - 1], yhat[i], sei[i])
                            - stats.norm.cdf(yl[j - 1], yhat[i], sei[i])
                        )

        llval0 = float(np.sum(
            stats.norm.logpdf(yi[nsub_idx], yhat[nsub_idx], sei[nsub_idx])
        ))
        llval1 = float(np.sum(
            np.log(delta[pgrp[sub_idx]] / preci[sub_idx])
            + stats.norm.logpdf(yi[sub_idx], yhat[sub_idx], sei[sub_idx])
            - np.log(Ai[sub_idx])
        ))
        llval = llval0 + llval1

        if dofit:
            return {"ll": llval, "beta": beta, "tau2": tau2, "delta": delta}
        return -llval

    return ll


def _make_ll_trunc(yi, vi, X, preci, subset, p, pvals, deltas, delta_arg,
                   mapfun, delta_min, delta_max, decreasing, delta_transf,
                   tau2_arg, tau2_transf, tau2_max, beta_arg, type_, steps,
                   alternative, pval_min, dofit=False):
    """``.selmodel.ll.trunc`` — closed-form truncated likelihood."""
    k = len(yi)
    sub_idx = np.where(subset)[0]
    nsub_idx = np.where(~subset)[0]

    def ll(par):
        beta = np.array(par[0:p], dtype=float)
        tau2 = float(par[p])
        delta = np.array(par[p + 1:p + 1 + deltas], dtype=float)

        beta = np.where(np.isnan(beta_arg), beta, beta_arg)
        if tau2_transf:
            tau2 = np.exp(tau2)
        tau2 = np.where(~np.isnan(tau2_arg), tau2_arg, tau2)
        tau2 = _clamp_tau2(float(tau2), tau2_max)

        if delta.shape == ():
            delta = np.array([float(delta)])
        if delta_transf:
            delta = _transform_delta(delta, deltas, delta_min, delta_max, mapfun, decreasing)
        delta = np.where(np.isnan(delta_arg), delta, delta_arg)

        yhat = X @ beta
        sei = np.sqrt(vi + tau2)

        cutoff = steps[0] if not np.isnan(steps[0]) else delta[1]

        ll0i = stats.norm.logpdf(yi[nsub_idx], yhat[nsub_idx], sei[nsub_idx])

        ys = yi[sub_idx]
        yh = yhat[sub_idx]
        se = sei[sub_idx]
        if alternative == "less":
            ll1i = (np.where(ys < cutoff, 0.0, np.log(delta[0]))
                    + stats.norm.logpdf(ys, yh, se)
                    - np.log(1.0 - (1.0 - delta[0]) * stats.norm.sf(cutoff, yh, se)))
        else:
            ll1i = (np.where(ys > cutoff, 0.0, np.log(delta[0]))
                    + stats.norm.logpdf(ys, yh, se)
                    - np.log(1.0 - (1.0 - delta[0]) * stats.norm.cdf(cutoff, yh, se)))

        llval = float(np.sum(ll0i) + np.sum(ll1i))

        if dofit:
            return {"ll": llval, "beta": beta, "tau2": tau2, "delta": delta}
        return -llval

    return ll


# --------------------------------------------------------------------------- #
# Numerical Hessian (numDeriv::hessian Richardson, r = 6)
# --------------------------------------------------------------------------- #

def _neville(xs, ys, target=0.0):
    """Neville extrapolation to ``target`` (step sizes ``xs``, estimates ``ys``)."""
    xs = np.asarray(xs, dtype=float)
    a = np.asarray(ys, dtype=float).copy()
    n = len(xs)
    for j in range(1, n):
        for i in range(n - 1, j - 1, -1):
            a[i] = ((target - xs[i - j]) * a[i] - (target - xs[i]) * a[i - 1]) / (xs[i] - xs[i - j])
    return a[n - 1]


def _num_hessian(fun, x, d=1e-4, r=6, v=2.0):
    """Richardson-extrapolated central-difference Hessian (numDeriv default)."""
    n = len(x)
    x = np.asarray(x, dtype=float)
    H = np.zeros((n, n))
    for i in range(n):
        for j in range(i, n):
            ei = np.zeros(n)
            ei[i] = 1.0
            ej = np.zeros(n)
            ej[j] = 1.0
            vals = []
            for kk in range(r):
                h = d * v ** kk
                if i == j:
                    val = (fun(x + 2.0 * h * ei) - 2.0 * fun(x) + fun(x - 2.0 * h * ei)) / (4.0 * h * h)
                else:
                    val = (fun(x + h * (ei + ej)) - fun(x + h * (ei - ej))
                           - fun(x + h * (-ei + ej)) + fun(x + h * (-ei - ej))) / (4.0 * h * h)
                vals.append(val)
            h2 = np.array([(d * v ** kk) ** 2 for kk in range(r)])
            H[i, j] = H[j, i] = _neville(h2, np.array(vals))
    return H


def _ml_loglik(yi, vi, X, beta, tau2):
    """ML log-likelihood of an ML-fitted model (metafor's ``logLik(x, REML=FALSE)``)."""
    k = len(yi)
    yhat = X @ np.asarray(beta, dtype=float)
    v = np.asarray(vi, dtype=float) + tau2
    return -k / 2.0 * math.log(2.0 * math.pi) - 0.5 * np.sum(np.log(v)) - 0.5 * np.sum((yi - yhat) ** 2 / v)


# --------------------------------------------------------------------------- #
# Main entry point
# --------------------------------------------------------------------------- #

def selmodel(
    x: RmaResult,
    type: str,
    alternative: str = "greater",
    prec=None,
    subset=None,
    delta=None,
    steps=None,
    decreasing: bool = False,
    verbose: bool = False,
    digits: int = 4,
    control: dict | None = None,
    **kwargs,
) -> SelModelResult:
    """Fit a publication-bias selection model on an existing ``rma`` fit.

    Args:
        x: fitted ``RmaResult`` (from :func:`rma`).
        type: selection model type (one of ``TYPE_OPTIONS``, prefix-matched).
        alternative: "greater" (default), "less" or "two.sided".
        prec: optional precision weights ("sei"/"vi"/"ninv"/"sqrtninv").
        subset: optional boolean vector selecting studies to model.
        delta: optional fixed values for the selection parameters.
        steps: optional p-value cut-points (required for ``stepfun``).
        decreasing: monotone decreasing step weights (``stepfun`` only).
        verbose: print progress.
        digits: printing precision (recorded only).
        control: dict of optimisation/Hessian control options.

    Returns:
        :class:`SelModelResult` with the full metafor ``$`` field set.
    """
    if alternative not in ("two.sided", "greater", "less"):
        raise ValueError("alternative must be 'two.sided', 'greater' or 'less'")

    # prefix-match the type
    matched = [t for t in TYPE_OPTIONS if t.startswith(type)]
    if not matched:
        raise ValueError(f"unknown 'type' {type!r} (see selmodel options)")
    type_ = matched[0]

    if type_ in ("trunc", "truncest") and alternative == "two.sided":
        raise ValueError("cannot use alternative='two.sided' with this selection model")

    if type_ != "stepfun" and decreasing:
        decreasing = False

    control = dict(control or {})
    con = {
        "verbose": False, "delta.init": None, "beta.init": None, "tau2.init": None,
        "delta.min": None, "delta.max": None, "tau2.max": np.inf,
        "tau2tol": min(min(x.vi) / 10.0, 1e-4), "deltatol": 1e-4, "pval.min": None,
        "optimizer": "optim", "optmethod": "BFGS",
        "hes.beta.fix": False, "hes.tau2.fix": False, "hes.delta.fix": False,
        "htransf": False, "hessianCtrl": None, "hesspack": "numDeriv",
        "scaleX": True,
    }
    for kk in control:
        if kk in con:
            con[kk] = control[kk]
    verbose = bool(verbose or con["verbose"])

    # --- refit by ML if the input was not an ML/FE model -------------------- #
    if x.method not in ("FE", "EE", "CE", "ML"):
        x = cast(RmaResult, rma(
            np.asarray(x.yi), np.asarray(x.vi), mods=x.mods,
            method="ML", test="z", level=x.level,
        ))

    yi = np.asarray(x.yi, dtype=float)
    vi = np.asarray(x.vi, dtype=float)
    X = np.asarray(x.X, dtype=float)
    p = x.p
    k = x.k

    if np.any(vi <= 0):
        raise ValueError("cannot fit selection model when one or more sampling variances are non-positive")

    # --- tau2 / beta arguments --------------------------------------------- #
    tau2_arg = np.nan
    if x.method in ("FE", "EE", "CE"):
        tau2_arg = 0.0
    elif x.tau2_fix:
        tau2_arg = x.tau2

    beta_arg = np.full(p, np.nan)

    # --- precision weights -------------------------------------------------- #
    if prec is not None:
        precspec = True
        if prec == "sei":
            preci = np.sqrt(vi)
        elif prec == "vi":
            preci = vi
        elif prec in ("ninv", "sqrtninv"):
            raise ValueError("no sample size information stored in the model object")
        else:
            raise ValueError("prec must be 'sei', 'vi', 'ninv' or 'sqrtninv'")
        preci = preci / np.max(preci)
    else:
        precspec = False
        preci = np.ones(k)

    # --- p-values ----------------------------------------------------------- #
    pvals = _selmodel_pval(yi, vi, alternative)

    # --- steps -------------------------------------------------------------- #
    if steps is None:
        stepsspec = False
        steps_arr = np.array([np.nan])
    else:
        stepsspec = True
        steps_arr = np.sort(np.unique(np.asarray(steps, dtype=float)))
        if type_ != "trunc" and np.any((steps_arr < 0) | (steps_arr > 1)):
            raise ValueError("'steps' values must be between 0 and 1")
        if type_ not in ("trunc", "beta"):
            if steps_arr[0] == 0:
                raise ValueError("lowest 'steps' value must be > 0")
            if steps_arr[-1] != 1:
                steps_arr = np.append(steps_arr, 1.0)

    if type_ == "trunc" and not stepsspec:
        stepsspec = True
        steps_arr = np.array([0.0])

    # --- subset ------------------------------------------------------------- #
    if subset is None:
        subset = np.ones(k, dtype=bool)
    else:
        subset = np.asarray(subset, dtype=bool)

    # --- type configuration -------------------------------------------------- #
    mapfun_arr: list[str | None] | None = None
    mapinvfun_arr: list[str | None] | None = None
    if type_ == "beta":
        if not stepsspec:
            steps_arr = np.array([0.0, 1.0])
        deltas = 2
        delta_lb = np.array([0.0, 0.0]); delta_ub = np.array([np.inf, np.inf])
        delta_lb_excl = [True, True]; delta_ub_excl = [False, False]
        delta_init = np.array([1.0, 1.0])
        delta_min = np.array([1e-5, 1e-5]); delta_max = np.array([100.0, 100.0])
        H0_delta = np.array([1.0, 1.0]); delta_LRT = [True, True]
        pval_min = 1e-5
        ll_kind = "cont"
    elif type_ in ("halfnorm", "negexp", "logistic", "power"):
        if not stepsspec:
            steps_arr = np.array([0.0])
        deltas = 1
        delta_lb = np.array([0.0]); delta_ub = np.array([np.inf])
        delta_lb_excl = [False]; delta_ub_excl = [False]
        delta_init = np.array([1.0])
        delta_min = np.array([0.0]); delta_max = np.array([100.0])
        H0_delta = np.array([0.0]); delta_LRT = [True]
        pval_min = 1e-5 if type_ == "power" else 0.0
        ll_kind = "cont"
    elif type_ in ("halfnorm1", "negexp1", "logistic1", "power1"):
        if not stepsspec:
            steps_arr = np.array([0.0])
        deltas = 1
        delta_lb = np.array([0.0]); delta_ub = np.array([np.inf])
        delta_lb_excl = [False]; delta_ub_excl = [False]
        delta_init = np.array([1.0])
        delta_min = np.array([0.0]); delta_max = np.array([100.0])
        H0_delta = np.array([0.0]); delta_LRT = [True]
        pval_min = 1e-5 if type_.endswith("1") and type_.startswith("power") else 0.0
        ll_kind = "cont"
    elif type_ in ("halfnorm2", "negexp2", "logistic2", "power2"):
        if not stepsspec:
            steps_arr = np.array([0.0])
        deltas = 2
        delta_lb = np.array([0.0, 0.0]); delta_ub = np.array([np.inf, np.inf])
        delta_lb_excl = [False, False]; delta_ub_excl = [False, False]
        delta_init = np.array([1.0, 0.25])
        delta_min = np.array([0.0, 0.0]); delta_max = np.array([100.0, 100.0])
        H0_delta = np.array([0.0, 0.0]); delta_LRT = [True, True]
        pval_min = 0.0
        ll_kind = "cont"
    elif type_ == "negexppow":
        if not stepsspec:
            steps_arr = np.array([0.0])
        deltas = 2
        delta_lb = np.array([0.0, 0.0]); delta_ub = np.array([np.inf, np.inf])
        delta_lb_excl = [False, False]; delta_ub_excl = [False, False]
        delta_init = np.array([1.0, 1.0])
        delta_min = np.array([0.0, 0.0]); delta_max = np.array([100.0, 100.0])
        H0_delta = np.array([0.0, 0.0]); delta_LRT = [True, True]
        pval_min = 0.0
        ll_kind = "cont"
    elif type_ == "stepfun":
        if not stepsspec:
            raise ValueError("must specify the 'steps' argument for stepfun")
        deltas = len(steps_arr)
        if decreasing:
            mapfun_arr = ["I"] * deltas
            mapinvfun_arr = ["I"] * deltas
            delta_lb = np.concatenate([[0.0], np.full(deltas - 1, -np.inf)])
            delta_ub = np.concatenate([[1.0], np.full(deltas - 1, np.inf)])
            delta_lb_excl = [False] * deltas; delta_ub_excl = [False] * deltas
            delta_init = np.concatenate([[1.0], np.full(deltas - 1, -2.0)])
            delta_min = np.zeros(deltas); delta_max = np.ones(deltas)
        else:
            mapfun_arr = [None] * deltas
            mapinvfun_arr = [None] * deltas
            delta_lb = np.zeros(deltas); delta_ub = np.full(deltas, np.inf)
            delta_lb_excl = [False] * deltas; delta_ub_excl = [False] * deltas
            delta_init = np.linspace(1.0, 0.8, deltas)
            delta_min = np.zeros(deltas); delta_max = np.full(deltas, 100.0)
        H0_delta = np.ones(deltas); delta_LRT = [True] * deltas
        pval_min = 0.0
        ll_kind = "stepfun"
    elif type_ == "stepcon":
        if not stepsspec:
            raise ValueError("must specify the 'steps' argument for stepcon")
        if prec is not None:
            warnings.warn(
                "argument 'prec' ignored (not applicable to this type of "
                "selection model)", stacklevel=2,
            )
            precspec = False
            preci = np.ones(k)
        deltas = len(steps_arr)
        # metafor sets delta.transf.fun = rep("plogis", deltas) but leaves
        # mapfun = NA; for lb=0, ub=1 the NA branch of .mapfun is plogis(), so
        # both coincide. monotone non-increasing delta is enforced as a
        # constraint rather than through the parameterisation.
        mapfun_arr = [None] * deltas
        mapinvfun_arr = [None] * deltas
        delta_lb = np.zeros(deltas); delta_ub = np.ones(deltas)
        delta_lb_excl = [False] * deltas; delta_ub_excl = [False] * deltas
        delta_init = np.linspace(1.0, 0.5, deltas)
        delta_min = np.zeros(deltas); delta_max = np.ones(deltas)
        H0_delta = np.ones(deltas); delta_LRT = [True] * deltas
        pval_min = 0.0
        ll_kind = "stepfun"
    elif type_ == "trunc":
        if len(steps_arr) != 1:
            raise ValueError("can only specify a single value for 'steps'")
        deltas = 1
        delta_lb = np.array([0.0]); delta_ub = np.array([np.inf])
        delta_lb_excl = [False]; delta_ub_excl = [False]
        delta_init = np.array([1.0])
        delta_min = np.array([0.0]); delta_max = np.array([100.0])
        H0_delta = np.array([1.0]); delta_LRT = [True]
        pval_min = 0.0
        ll_kind = "trunc"
    elif type_ == "truncest":
        if not stepsspec:
            steps_arr = np.array([np.nan])
        deltas = 2
        # metafor leaves mapfun = c(NA, "I") here (scaled-logistic for delta[1],
        # then explicitly forces mapfun[2] <- "I"); see the type=="truncest"
        # override further below in selmodel.rma.uni().
        mapfun_arr = [None, "I"]; mapinvfun_arr = [None, "I"]
        delta_lb = np.array([0.0, -np.inf]); delta_ub = np.array([np.inf, np.inf])
        delta_lb_excl = [False, False]; delta_ub_excl = [False, False]
        delta_init = np.array([1.0, float(np.mean(yi))])
        if alternative == "greater":
            delta_min = np.array([0.0, float(np.min(yi) - np.std(yi))])
            delta_max = np.array([100.0, float(np.max(yi))])
        else:
            delta_min = np.array([0.0, float(np.min(yi))])
            delta_max = np.array([100.0, float(np.max(yi) + np.std(yi))])
        H0_delta = np.array([1.0, 0.0]); delta_LRT = [True, False]
        pval_min = 0.0
        ll_kind = "trunc"
    else:
        raise ValueError(f"unsupported selection model type: {type_!r}")

    # --- delta argument (user-specified fixed values) ----------------------- #
    if delta is None:
        delta_arg = np.full(deltas, np.nan)
    else:
        delta_arg = np.atleast_1d(np.asarray(delta, dtype=float)).ravel()
        if delta_arg.size == 1 and deltas > 1:
            delta_arg = np.full(deltas, float(delta_arg[0]))
        if len(delta_arg) != deltas:
            raise ValueError(f"'delta' should be of length {deltas}")

    if type_ == "stepfun":
        if decreasing or np.isnan(delta_arg[0]):
            delta_arg[0] = 1.0
    elif type_ == "stepcon":
        delta_arg[0] = 1.0

    # --- con overrides ------------------------------------------------------- #
    if con["delta.min"] is not None:
        delta_min = np.atleast_1d(np.asarray(con["delta.min"], dtype=float)).ravel()
        if delta_min.size == 1 and deltas > 1:
            delta_min = np.full(deltas, float(delta_min[0]))
        if delta_min.size != deltas:
            raise ValueError(f"'delta.min' should be of length {deltas}")
    if con["delta.max"] is not None:
        delta_max = np.atleast_1d(np.asarray(con["delta.max"], dtype=float)).ravel()
        if delta_max.size == 1 and deltas > 1:
            delta_max = np.full(deltas, float(delta_max[0]))
        if delta_max.size != deltas:
            raise ValueError(f"'delta.max' should be of length {deltas}")

    # mapfun / mapinvfun (default: scaled-logistic onto [delta.min, delta.max]).
    # ``stepfun`` (decreasing) and ``truncest`` set their own maps above; an
    # infinite delta.max forces the exponential transform (metafor's
    # ``ddd$defmap || any(is.infinite(delta.max))`` branch).
    if mapfun_arr is None:
        mapfun_arr = [None] * deltas
        mapinvfun_arr = [None] * deltas
    if np.any(np.isinf(delta_max)):
        mapfun_arr = ["exp"] * deltas
        mapinvfun_arr = ["log"] * deltas

    if con["pval.min"] is not None:
        pval_min = float(con["pval.min"])

    # clamp p-values
    pvals = np.where(pvals < pval_min, pval_min, pvals)
    pvals = np.where(pvals > 1.0 - pval_min, 1.0 - pval_min, pvals)

    # --- p-value grouping (stepfun) ----------------------------------------- #
    if type_ != "trunc" and stepsspec:
        pgrp = np.searchsorted(steps_arr, pvals, side="left")
    else:
        pgrp = np.full(k, -1, dtype=int)

    # --- scaling of moderators ---------------------------------------------- #
    scaleX = con["scaleX"] and (not x.int_only) and x.int_incl
    Xsave = X
    mX = None
    if scaleX:
        Xsub = X[:, 1:p]
        meanX = np.mean(Xsub, axis=0)
        sdX = np.std(Xsub, axis=0, ddof=1)
        is_d = [_is_dummy(X[:, j]) for j in range(p)]
        mX = np.zeros((p, p))
        mX[0, 0] = 1.0
        for j in range(1, p):
            if is_d[j]:
                mX[0, j] = 0.0
                mX[j, j] = 1.0
            else:
                mX[0, j] = -meanX[j - 1] / sdX[j - 1]
                mX[j, j] = 1.0 / sdX[j - 1]
        X = X.copy()
        for j in range(1, p):
            if not is_d[j]:
                X[:, j] = (X[:, j] - meanX[j - 1]) / sdX[j - 1]

    # --- starting values ----------------------------------------------------- #
    beta_init = np.asarray(x.b, dtype=float)
    if scaleX:
        assert mX is not None
        imX = np.linalg.inv(mX)
        beta_init = imX @ beta_init

    tau2_init = math.log(x.tau2 + 0.001)
    if con["tau2.init"] is not None:
        tau2_init = math.log(float(con["tau2.init"]))
    if con["beta.init"] is not None:
        beta_init = np.asarray(con["beta.init"], dtype=float)

    delta_init = np.array([
        _mapinvfun(delta_init[j], delta_min[j], delta_max[j],
                   (mapinvfun_arr or [None] * deltas)[j])
        for j in range(deltas)
    ])
    delta_init = np.where(np.isnan(delta_arg), delta_init, delta_arg)
    if con["delta.init"] is not None:
        cdi = np.asarray(con["delta.init"], dtype=float)
        if cdi.size == 1 and deltas > 1:
            cdi = np.full(deltas, float(cdi))
        delta_init = cdi

    par0 = np.concatenate([beta_init, [tau2_init], delta_init])

    # --- build log-likelihood ------------------------------------------------ #
    if ll_kind == "cont":
        make = _make_ll_cont
    elif ll_kind == "stepfun":
        make = _make_ll_stepfun
    else:
        make = _make_ll_trunc

    ll_fun = make(
        yi, vi, X, preci, subset, p, pvals, deltas, delta_arg,
        mapfun_arr, delta_min, delta_max, decreasing, True, tau2_arg,
        True, con["tau2.max"], beta_arg, type_, steps_arr, alternative, pval_min,
        dofit=False,
    )

    # --- optimise ------------------------------------------------------------ #
    def _run_opt(fun, x0):
        """BFGS for the unconstrained types; SLSQP for ``stepcon``.

        metafor delegates ``stepcon`` to ``Rsolnp::solnp`` (an augmented
        Lagrangian SQP solver) with the monotonicity constraint
        ``0 <= delta[j] - delta[j+1] <= 1``; SciPy's SLSQP is the same class of
        solver and converges to the same KKT point.
        """
        if type_ == "stepcon":
            return optimize.minimize(
                fun, x0, method="SLSQP",
                constraints=[{
                    "type": "ineq",
                    "fun": lambda par: _stepcon_ineq(
                        par, p, deltas, delta_arg, delta_min, delta_max,
                        mapfun_arr,
                    ),
                }],
                options={"maxiter": 1000, "ftol": 1e-10},
            )
        return optimize.minimize(
            fun, x0, method="BFGS",
            options={"gtol": 1e-9, "maxiter": 1000},
        )

    opt_res = _run_opt(ll_fun, par0)

    # The truncated / step-function likelihoods are piecewise constant in the
    # cut-point, so the score is exactly zero across whole intervals and a
    # gradient method happily stops on such a plateau; two-parameter weights
    # are flat enough to stall as well.  metafor's optim() has the same
    # weakness (its ``truncest`` and ``negexppow`` fits are not the maxima).
    # One derivative-free restart costs little and never makes the fit worse.
    _restart = type_ in ("trunc", "truncest", "stepfun", "stepcon",
                         "negexppow")
    if _restart and np.any(np.isnan(delta_arg)):
        try:
            nm = optimize.minimize(
                ll_fun, np.asarray(opt_res.x, dtype=float),
                method="Nelder-Mead",
                options={"maxiter": 250, "xatol": 1e-8,
                         "fatol": 1e-10, "adaptive": True},
            )
            if (np.isfinite(nm.fun)
                    and float(np.asarray(nm.fun)) < float(np.asarray(opt_res.fun))):
                polished = _run_opt(ll_fun, np.asarray(nm.x, dtype=float))
                opt_res = (polished
                           if float(np.asarray(polished.fun))
                           <= float(np.asarray(nm.fun)) else nm)
        except Exception:
            pass
    par_opt = opt_res.x

    fit = make(
        yi, vi, X, preci, subset, p, pvals, deltas, delta_arg,
        mapfun_arr, delta_min, delta_max, decreasing, True, tau2_arg,
        True, con["tau2.max"], beta_arg, type_, steps_arr, alternative, pval_min,
        dofit=True,
    )
    fit = fit(par_opt)
    ll = fit["ll"]
    beta = np.asarray(fit["beta"], dtype=float).ravel()
    tau2 = float(fit["tau2"])
    delta = np.asarray(fit["delta"], dtype=float).ravel()

    # --- Hessian (original scale, htransf=FALSE) ---------------------------- #
    # metafor: hes.tau2.fix | tau2 < tau2tol -> tau2 treated as fixed;
    # stepfun additionally freezes deltas that collapsed below deltatol.
    if (bool(con["hes.tau2.fix"]) or
            tau2 < float(con["tau2tol"] if con["tau2tol"] is not None else 0.0)):
        tau2_hes = tau2
    else:
        tau2_hes = tau2_arg
    if con["hes.delta.fix"]:
        delta_hes = delta
    elif type_ == "stepfun":
        delta_hes = np.where(delta < con["deltatol"], delta, delta_arg)
    else:
        delta_hes = delta_arg
    if con["hes.beta.fix"]:
        beta_hes = beta
    else:
        beta_hes = beta_arg

    hest = np.concatenate([
        np.isnan(beta_hes),
        [np.isnan(tau2_hes)],
        np.isnan(delta_hes),
    ])
    H = np.full((p + 1 + deltas, p + 1 + deltas), np.nan)
    vb = np.full((p, p), np.nan)
    se_tau2 = np.nan
    vd = np.full((deltas, deltas), np.nan)

    # Hessian of the log-likelihood evaluated on the natural scale
    ll_fun_nat = make(
        yi, vi, X, preci, subset, p, pvals, deltas, delta_hes,
        mapfun_arr, delta_min, delta_max, decreasing, False, tau2_hes,
        False, con["tau2.max"], beta_hes, type_, steps_arr, alternative, pval_min,
        dofit=False,
    )
    x_nat = np.concatenate([beta, [tau2], delta])
    try:
        H = _num_hessian(ll_fun_nat, x_nat, d=1e-4, r=6, v=2.0)
        hest_idx = np.nonzero(hest)[0]
        H_hest = H[np.ix_(hest_idx, hest_idx)]
        try:
            iH_hest = np.linalg.inv(H_hest)
        except np.linalg.LinAlgError:
            iH_hest = None
        if iH_hest is not None and np.all(np.isfinite(iH_hest)):
            npar = p + 1 + deltas
            iH = np.zeros((npar, npar))
            iH[np.ix_(hest_idx, hest_idx)] = iH_hest
            b_sel = np.arange(p)[np.isnan(beta_hes)]
            if len(b_sel) > 0:
                vb[np.ix_(b_sel, b_sel)] = iH[np.ix_(b_sel, b_sel)]
            if np.isnan(tau2_hes):
                se_tau2 = math.sqrt(iH[p, p])
            d_sel = np.arange(deltas)[np.isnan(delta_hes)]
            if len(d_sel) > 0:
                iH_d = p + 1 + d_sel
                vd[np.ix_(d_sel, d_sel)] = iH[np.ix_(iH_d, iH_d)]
    except Exception:
        pass

    # --- un-scale betas ------------------------------------------------------ #
    if scaleX:
        assert mX is not None
        beta = mX @ beta
        vb = mX @ vb @ mX.T
        X = Xsave

    # --- fixed-effects tests ------------------------------------------------- #
    btt_idx = [0] if x.int_only else list(range(1, p))
    m = len(btt_idx)
    b_tt = beta[btt_idx]
    vb_tt = vb[np.ix_(btt_idx, btt_idx)]
    try:
        QM = float(b_tt @ np.linalg.solve(vb_tt, b_tt))
    except np.linalg.LinAlgError:
        QM = np.nan
    QMp = float(stats.chi2.sf(QM, m)) if m > 0 else np.nan

    se = np.sqrt(np.diag(vb))
    zval = beta / se
    pval = 2.0 * stats.norm.sf(np.abs(zval))
    crit = stats.norm.isf((1.0 - x.level) / 2.0)
    ci_lb = beta - crit * se
    ci_ub = beta + crit * se

    se_delta = np.sqrt(np.diag(vd))
    zval_delta = (delta - H0_delta) / se_delta
    pval_delta = 2.0 * stats.norm.sf(np.abs(zval_delta))
    ci_lb_delta = delta - crit * se_delta
    ci_ub_delta = delta + crit * se_delta

    ci_lb_delta = np.maximum(ci_lb_delta, delta_lb)
    ci_ub_delta = np.minimum(ci_ub_delta, delta_ub)
    ci_lb_delta = np.maximum(ci_lb_delta, delta_min)
    ci_ub_delta = np.minimum(ci_ub_delta, delta_max)

    ci_lb_tau2 = tau2 - crit * se_tau2
    ci_ub_tau2 = tau2 + crit * se_tau2
    ci_lb_tau2 = max(ci_lb_tau2, 0.0)

    # --- heterogeneity test (LRT.tau2) --------------------------------------- #
    LRT_tau2 = np.nan
    LRTp_tau2 = np.nan
    if not x.tau2_fix and x.method not in ("FE", "EE", "CE"):
        ll_fun_t0 = make(
            yi, vi, X, preci, subset, p, pvals, deltas, delta_arg,
            mapfun_arr, delta_min, delta_max, decreasing, True, 0.0,
            False, con["tau2.max"], beta_arg, type_, steps_arr, alternative, pval_min,
            dofit=False,
        )
        par0_t0 = np.concatenate([beta_init, [tau2_init], delta_init])
        opt_t0 = _run_opt(ll_fun_t0, par0_t0)
        fit_t0 = make(
            yi, vi, X, preci, subset, p, pvals, deltas, delta_arg,
            mapfun_arr, delta_min, delta_max, decreasing, True, 0.0,
            False, con["tau2.max"], beta_arg, type_, steps_arr, alternative, pval_min,
            dofit=True,
        )(opt_t0.x)
        ll0_t0 = fit_t0["ll"]
        LRT_tau2 = max(0.0, -2.0 * (ll0_t0 - ll))
        LRTp_tau2 = float(stats.chi2.sf(LRT_tau2, 1))

    # --- LRT for the selection model parameters ----------------------------- #
    ll0 = _ml_loglik(yi, vi, X, x.b, x.tau2)
    LRT = max(0.0, -2.0 * (ll0 - ll))
    LRTdf = int(np.sum(np.isnan(delta_arg) & np.array(delta_LRT)))
    LRTp = float(stats.chi2.sf(LRT, LRTdf)) if LRTdf > 0 else np.nan

    # --- fit statistics ------------------------------------------------------- #
    parms = p + (0 if (x.method in ("FE", "EE", "CE") or x.tau2_fix) else 1) + int(np.sum(np.isnan(delta_arg)))
    ll_ML = ll
    dev_ML = -2.0 * ll_ML
    AIC_ML = -2.0 * ll_ML + 2.0 * parms
    BIC_ML = -2.0 * ll_ML + parms * math.log(k)
    AICc_ML = -2.0 * ll_ML + 2.0 * parms * max(k, parms + 2) / (max(k, parms + 2) - parms - 1)
    fit_stats = {
        "ML": {"ll": ll_ML, "dev": dev_ML, "AIC": AIC_ML, "BIC": BIC_ML, "AICc": AICc_ML},
        "REML": {"ll": np.nan, "dev": np.nan, "AIC": np.nan, "BIC": np.nan, "AICc": np.nan},
    }

    out = SelModelResult(
        beta=[float(v) for v in beta],
        b=[float(v) for v in beta],
        se=[float(v) for v in se],
        zval=[float(v) for v in zval],
        pval=[float(v) for v in pval],
        ci_lb=[float(v) for v in ci_lb],
        ci_ub=[float(v) for v in ci_ub],
        vb=vb,
        tau2=tau2,
        se_tau2=float(se_tau2),
        ci_lb_tau2=float(ci_lb_tau2),
        ci_ub_tau2=float(ci_ub_tau2),
        delta=[float(v) for v in delta],
        vd=vd,
        se_delta=[float(v) for v in se_delta],
        zval_delta=[float(v) for v in zval_delta],
        pval_delta=[float(v) for v in pval_delta],
        ci_lb_delta=[float(v) for v in ci_lb_delta],
        ci_ub_delta=[float(v) for v in ci_ub_delta],
        QM=float(QM),
        QMp=QMp,
        ll=float(ll),
        ll0=float(ll0),
        LRT=float(LRT),
        LRTdf=LRTdf,
        LRTp=float(LRTp),
        LRT_tau2=float(LRT_tau2),
        LRTp_tau2=float(LRTp_tau2),
        parms=parms,
        fit_stats=fit_stats,
        pvals=[float(v) for v in pvals],
        type=type_,
        alternative=alternative,
        steps=[float(v) for v in steps_arr] if not np.all(np.isnan(steps_arr)) else [float(v) for v in steps_arr],
        pgrp=[int(v) for v in pgrp],
        k0=int(np.sum(~subset)),
        k1=int(np.sum(subset)),
        model="rma.uni.selmodel",
        k=k,
        p=p,
        tau2_fix=x.tau2_fix,
        method=x.method,
        yi=[float(v) for v in yi],
        vi=[float(v) for v in vi],
        level=x.level,
    )
    # the source fit is needed by profile_selmodel() to refit on a delta grid
    setattr(out, "_model", x)
    return out



# --------------------------------------------------------------------------- #
# profile.rma.uni.selmodel
# --------------------------------------------------------------------------- #

class SelProfileResult:
    """Result of :func:`profile_selmodel` (``profile.rma.uni.selmodel``).

    ``delta`` holds the profiled selection parameters, ``ll`` the corresponding
    maximised log-likelihoods and ``ci_lb``/``ci_ub`` the profile-likelihood
    limits (the ``delta`` values where ``ll`` drops by ``chi2(1, 1-alpha)/2``).
    """

    __slots__ = ("delta", "ll", "maxll", "delta_hat", "ci_lb", "ci_ub",
                 "level", "type", "k")

    def __init__(self, delta, ll, maxll, delta_hat, ci_lb, ci_ub, level,
                 type_, k):
        self.delta = delta
        self.ll = ll
        self.maxll = maxll
        self.delta_hat = delta_hat
        self.ci_lb = ci_lb
        self.ci_ub = ci_ub
        self.level = level
        self.type = type_
        self.k = k

    def __repr__(self):
        return (f"SelProfileResult(type={self.type!r}, delta_hat="
                f"{self.delta_hat!r}, ci=[{self.ci_lb!r}, {self.ci_ub!r}])")

    def to_dict(self):
        return {"delta": self.delta, "ll": self.ll, "maxll": self.maxll,
                "delta_hat": self.delta_hat, "ci_lb": self.ci_lb,
                "ci_ub": self.ci_ub, "level": self.level, "type": self.type,
                "k": self.k}


def profile_selmodel(result, delta=None, n_points: int = 20, span=None,
                     level: float = 0.95):
    """Profile the log-likelihood of a selection model over ``delta``.

    ``profile.rma.uni.selmodel`` refits the model on a grid of fixed ``delta``
    values and records the maximised log-likelihood, then inverts the
    likelihood-ratio test to obtain a profile CI for the selection parameter.
    The refits here re-estimate tau^2 exactly as the rest of this module does;
    R additionally fixes tau^2 at the full-model estimate, so the curve can
    differ in the third or fourth digit.

    Args:
        result: :class:`SelModelResult` from :func:`selmodel`.
        delta: explicit grid (default: ``n_points`` values spanning
            ``delta_hat +/- span * |delta_hat|``, or +/-2 when
            ``delta_hat == 0``).
        n_points: grid size when ``delta`` is not given.
        span: half-width of the default grid, as a multiple of ``|delta_hat|``.
        level: confidence level for the profile limits.

    Returns:
        :class:`SelProfileResult`.
    """
    if not isinstance(result, SelModelResult):
        raise TypeError("argument 'result' must be a SelModelResult")
    hat = float(np.asarray(result.delta, dtype=float).ravel()[0])
    if delta is None:
        half = (abs(hat) * (1.5 if span is None else float(span))) or 2.0
        delta = list(np.linspace(hat - half, hat + half, int(n_points)))
    delta = [float(v) for v in delta]

    ll = []
    for d in delta:
        try:
            fit = selmodel(getattr(result, "_model"), result.type,
                           alternative=result.alternative,
                           delta=[d], control=None)
            ll.append(float(fit.ll))
        except Exception:  # pragma: no cover - degenerate grid point
            ll.append(float("nan"))

    arr = np.asarray(ll, dtype=float)
    maxll = float(np.nanmax(arr)) if np.isfinite(arr).any() else float("nan")
    cut = maxll - 0.5 * float(stats.chi2.ppf(level, 1))
    ok = np.asarray(delta, dtype=float)[arr >= cut]
    ci_lb = float(np.min(ok)) if ok.size else float("nan")
    ci_ub = float(np.max(ok)) if ok.size else float("nan")
    return SelProfileResult(delta, ll, maxll, hat, ci_lb, ci_ub, level,
                            str(result.type), int(result.k))
