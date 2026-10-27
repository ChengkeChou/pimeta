"""Dynamic heterogeneity: a pooled effect and a tau^2 that both move with time.

The question this module answers is not "how much heterogeneity is there" but
"is there more of it now than there used to be", and those are different
problems.  A single :math:`\\tau^2` is a variance component with a boundary at
zero; :math:`\\tau^2(t)` is a *function* whose shape has to be estimated, and
the estimator that behaves at :math:`k = 12` is not the one that behaves at
:math:`k = 60`.

The model
---------
A location-scale model (Viechtbauer & López-López, 2022) with the mean and the
heterogeneity variance both functions of a continuous index:

.. math::

    y_i \\sim N\\big(x_i'\\beta,\\; v_i + \\tau_i^2\\big), \\qquad
    \\log \\tau_i^2 = z_i'\\alpha .

A log link keeps :math:`\\tau_i^2 > 0` without a constrained optimiser.
:math:`x_i` and :math:`z_i` are built from the same index but their
coefficients are separate: :math:`\\beta` is the pooled effect and its trend,
:math:`\\alpha` is the level of the heterogeneity and its trend.  This is
exactly the model ``metafor::rma(yi, vi, mods = ~ time, scale = ~ time)``
fits, and that is the reference the tests use.

Why the mean model is not optional
----------------------------------
Fitting :math:`\\tau^2(t)` while holding :math:`\\mu` constant is the most
common way to manufacture a spurious trend.  If the pooled effect rises with
time and the heterogeneity does not, the early and the late studies sit far
from one common mean, their squared residuals grow, and the scale model has
nowhere to put that but a time trend in :math:`\\tau^2`.  The module therefore
requires the location model to be named as well, and warns when ``mu="none"``
is combined with a non-constant ``tau``.  ``tests/test_dynhet.py`` measures how
large the resulting bias is instead of asserting that it exists.

Two location standard errors, and neither is the interesting one
---------------------------------------------------------------
``metafor`` reports the location SE from :math:`(X'WX)^{-1}` with
:math:`W = \\operatorname{diag}(1/(v_i + \\hat\\tau_i^2))` — it treats the
estimated scale as known.  ``se="profile"`` reports the block of the joint
inverse information instead, which lets the scale uncertainty through.

The two are closer than the argument about them suggests, and the reason is
worth stating: in a Gaussian model the *expected* information is
block-diagonal between location and scale, so the difference is
:math:`O(1/k)`.  Measured on the 36-study fixture, the observed
cross-information is 4 % of the diagonal and the profile SE is **0.25 %**
larger than the GLS one.  Both are stored on the result, so the difference is
reported rather than asserted.

What does matter in small samples is the reference distribution, and the
measurement is less tidy than the advice.  ``tools/coverage_dynhet_ci.py``
simulates from a known :math:`\\mu(t)` and :math:`\\tau^2(t)`, 120 replications
at :math:`k = 40`, nominal 95 %:

======================  ======================
:math:`\\beta_1`, z    0.917
:math:`\\beta_1`, KH   0.925
:math:`\\alpha_1`, z   0.950
:math:`\\alpha_1`, t   0.967
======================  ======================

The Monte Carlo standard error of a coverage rate at 120 replications is 0.025,
so the two location figures are not distinguishable from each other: what the
study establishes is that the location interval is a little short of nominal and
that Knapp–Hartung does not repair it, while the *scale* interval is calibrated.
The recommendation to use ``test="knha"`` stands on the smaller-sample
behaviour in the literature, not on this measurement.

Small ``k``
-----------
Above roughly 45 studies the profile likelihood is well behaved and the fit is
instantaneous.  Below roughly 15 no continuous :math:`\\tau^2(t)` is identified
with useful power, and the honest analysis is a two-period comparison — which
this module supports by passing a 0/1 period indicator as ``time`` with
``tau="linear"``, giving
:math:`\\tau^2_{\\text{late}}/\\tau^2_{\\text{early}} = e^{\\alpha_1}`.  Between
those, :func:`bayes_tau_trend` is the better answer, because a shrinkage prior
on :math:`\\alpha_1` is what stops three influential early studies from
deciding the shape.

Where the boundary actually is, stated precisely
------------------------------------------------
The test of :math:`\\alpha_1 = 0` is **not** a boundary test in general:
:math:`\\alpha_1` may be negative (heterogeneity shrinking over time), so zero
is an interior point and the likelihood-ratio statistic is asymptotically
:math:`\\chi^2_1`.  The boundary is on :math:`\\alpha_0`: when the truth is no
heterogeneity at all, :math:`\\alpha_0 \\to -\\infty` and :math:`\\alpha_1`
stops being identified, and there the :math:`\\chi^2_1` reference is wrong in
the usual way.  The module offers three references — Wald, :math:`\\chi^2_1`
and a parametric bootstrap — and says which null each one assumes.
"""

from __future__ import annotations

import math
import warnings
from collections.abc import Callable
from dataclasses import dataclass, field
from functools import partial
from typing import Any

import numpy as np
from scipy import optimize, stats

from .bayes import Posterior, sample_log_posterior
from .levels import level_to_alpha
from .parallel import map_ordered

__all__ = ["BayesTauTrendResult", "TauTrendResult", "bayes_tau_trend", "tau_trend"]

_KINDS = ("none", "linear", "quadratic", "spline")
_TERMS = ("intrcpt", "time", "time^2")

#: smallest number of studies a spline scale is allowed to be fitted to.  The
#: spline adds ``n_knots - 2`` bending columns on top of the two scale
#: coefficients, and a fitted wiggliness needs more studies than that.
_SPLINE_MIN_EXTRA = 3

_MCMC_DEFAULTS: dict[str, Any] = {
    "n_draws": 4000,
    "warmup": 2000,
    "chains": 4,
    "seed": None,
    "check": True,
    "rhat_max": 1.05,
    "ess_min": 100.0,
}

#: prior scales of the Bayesian fit; see :func:`bayes_tau_trend`
_BAYES_PRIORS = {"beta_sd": 100.0, "alpha0_sd": 2.0, "alpha_sd": 0.3}


# --------------------------------------------------------------------------- #
# design and likelihood
# --------------------------------------------------------------------------- #
def _design(time: np.ndarray, kind: str, centre: float) -> np.ndarray:
    """Design matrix for one part of the model, on the centred time scale.

    ``kind="spline"`` is handled by :class:`_SplineBasis`, which carries the
    fitted knot vector and the orthogonalisation; this function is for the
    polynomial parts only.
    """
    tc = np.asarray(time, dtype=float) - float(centre)
    if kind == "none":
        return np.ones((tc.size, 1))
    if kind == "linear":
        return np.column_stack([np.ones(tc.size), tc])
    if kind == "quadratic":
        return np.column_stack([np.ones(tc.size), tc, tc * tc])
    if kind == "spline":
        raise ValueError(
            "a spline scale is evaluated through its fitted basis, not through "
            "_design; call tau_trend(tau='spline') and use its predict()")
    raise ValueError(f"unknown model {kind!r}; choose from {', '.join(_KINDS)}")


@dataclass(frozen=True)
class _SplineBasis:
    """Orthonormal bending basis for :math:`\\log\\tau^2(t)`.

    The restricted cubic spline on centred time contributes ``nk - 2``
    non-linear columns.  Two changes are made before they are used as scale
    regressors, and both have a reason:

    * **Centred on the linear part.**  The non-linear columns are projected on
      :math:`[1, t]` and what is left is kept, so the spline adds *shape* only:
      the level and the linear trend keep their own coefficients and the two
      parametrisations cannot steal each other's signal.
    * **Orthonormalised.**  A Gram–Schmidt pass makes ``B'B = I``, which turns
      the single wiggliness hyperparameter into one ridge penalty
      :math:`\\lambda = 1/\\sigma_s^2`: every bending direction is shrunk by the
      same amount instead of by an amount that depends on the units of a cubic
      column.

    The transform is stored (``C`` for the projection, ``R`` for the triangular
    factor) so that the basis can be evaluated at times that are not in the
    data — a prediction curve needs exactly that, and rebuilding a fresh
    Gram–Schmidt at new points would silently define a different basis.
    """

    knots: np.ndarray
    centre: float
    project: np.ndarray
    factor: np.ndarray

    @property
    def n_bend(self) -> int:
        return int(self.project.shape[1])

    @property
    def names(self) -> list[str]:
        return ["alpha.intrcpt", "alpha.time"] + [
            f"alpha.bend{i + 1}" for i in range(self.n_bend)]

    def bending(self, time) -> np.ndarray:
        """The orthonormal bending columns evaluated at ``time``."""
        from .basis import _rcs_matrix

        x = np.asarray(time, dtype=float).ravel() - float(self.centre)
        columns = _rcs_matrix(x, self.knots)[:, 1:]
        linear = np.column_stack([np.ones(x.size), x])
        residual = columns - linear @ self.project
        # B = residual @ inv(factor);  inv(factor) is triangular, use solve
        return np.linalg.solve(self.factor.T, residual.T).T

    def matrix(self, time) -> np.ndarray:
        """``[1, t_c, B]`` — the design for :math:`\\log\\tau^2(t)`."""
        x = np.asarray(time, dtype=float).ravel() - float(self.centre)
        return np.column_stack([np.ones(x.size), x, self.bending(time)])


def _spline_basis(time: np.ndarray, centre: float,
                  knots=None, n_knots: int = 4) -> _SplineBasis:
    """Build the orthonormal bending basis on the observed time points."""
    from .basis import _rcs_matrix, default_knots

    tc = np.asarray(time, dtype=float) - float(centre)
    kn = (default_knots(tc, int(n_knots)) if knots is None
          else np.sort(np.asarray(knots, dtype=float).ravel()))
    if kn.size < 3:
        raise ValueError("a spline scale needs at least three knots")
    if np.any(np.diff(kn) <= 0):
        raise ValueError("'knots' must be strictly increasing")
    columns = _rcs_matrix(tc, kn)[:, 1:]
    linear = np.column_stack([np.ones(tc.size), tc])
    project = np.linalg.lstsq(linear, columns, rcond=None)[0]
    q, r = np.linalg.qr(columns - linear @ project)
    signs = np.where(np.diag(r) < 0.0, -1.0, 1.0)
    return _SplineBasis(knots=np.asarray(kn, dtype=float), centre=float(centre),
                        project=project, factor=np.diag(signs) @ r)


def _names(kind: str, scale: bool) -> list[str]:
    if kind == "spline":
        # the bending names need the column count, which the fitted basis
        # supplies; this static fallback is only used before a fit exists
        terms = ["alpha.intrcpt", "alpha.time"] if scale else list(_TERMS[:2])
        bend = ["alpha.bend1", "alpha.bend2"] if scale else []
        return terms + bend
    n = {"none": 1, "linear": 2, "quadratic": 3}[kind]
    poly = _TERMS[:n]
    return [f"alpha.{t}" for t in poly] if scale else list(poly)


def _check_inputs(yi, vi, time):
    y = np.asarray(yi, dtype=float).ravel()
    v = np.asarray(vi, dtype=float).ravel()
    t = np.asarray(time, dtype=float).ravel()
    if not (y.size == v.size == t.size):
        raise ValueError(
            "'yi', 'vi' and 'time' must have the same length "
            f"({y.size}, {v.size}, {t.size})")
    if y.size < 3:
        raise ValueError("at least three studies are needed")
    if not np.all(np.isfinite(y)) or not np.all(np.isfinite(t)):
        raise ValueError("'yi' and 'time' must be finite")
    if not np.all(np.isfinite(v)) or np.any(v <= 0.0):
        raise ValueError("every study needs a finite, strictly positive variance")
    if np.allclose(t, t[0]):
        raise ValueError(
            "'time' is constant: there is no index for a dynamic heterogeneity "
            "model, and a constant scale is tau='none'")
    return y, v, t


def _profile(y, v, X, Z, alpha, method: str):
    """GLS location estimate, weights, residuals and the profile log-likelihood."""
    with np.errstate(over="ignore", invalid="ignore"):
        tau2 = np.exp(Z @ alpha)
    d = v + tau2
    if not np.all(np.isfinite(d)) or np.any(d <= 0.0):
        return None
    w = 1.0 / d
    xtwx = X.T @ (w[:, None] * X)
    sign, logdet = np.linalg.slogdet(xtwx)
    if sign <= 0:
        return None
    vcov = np.linalg.inv(xtwx)
    beta = vcov @ (X.T @ (w * y))
    resid = y - X @ beta
    rss = float(np.sum(w * resid * resid))
    k, p = X.shape
    loglik = -0.5 * float(np.sum(np.log(d))) - 0.5 * rss
    if method == "REML":
        # metafor's REML adds 0.5 log|X'X| so the value is comparable with ML;
        # it does not depend on the parameters and so changes no estimate
        sign_xx, logdet_xx = np.linalg.slogdet(X.T @ X)
        assert sign_xx > 0
        loglik += (-0.5 * (k - p) * math.log(2.0 * math.pi)
                   + 0.5 * float(logdet_xx) - 0.5 * float(logdet))
    else:
        loglik -= 0.5 * k * math.log(2.0 * math.pi)
    return tau2, w, beta, resid, rss, vcov, float(loglik)


def _neg_loglik(alpha, y, v, X, Z, method):
    out = _profile(y, v, X, Z, alpha, method)
    return np.inf if out is None else -out[-1]


def _neg_grad_alpha(alpha, y, v, X, Z, method) -> np.ndarray:
    """Gradient of :func:`_neg_loglik`.

    :func:`_grad_alpha` returns :math:`d\\ell/d\\alpha`; a minimiser of the
    negated objective needs the negation, and getting this backwards makes a
    line search reject every step and report convergence at its start point.
    """
    return -_grad_alpha(alpha, y, v, X, Z, method)


def _grad_alpha(alpha, y, v, X, Z, method) -> np.ndarray:
    """``d loglik / d alpha`` with the location coefficients profiled out.

    The derivative of :math:`\\beta` with respect to :math:`\\alpha` drops out
    of the residual sum of squares because :math:`\\beta` solves the normal
    equations.  What is left is one weighted sum per scale coefficient.

    Checked against central differences in ``tests/test_dynhet.py`` rather than
    trusted, because a wrong gradient still converges — to the wrong place.
    """
    out = _profile(y, v, X, Z, alpha, method)
    if out is None:
        return np.full(np.asarray(alpha).size, np.nan)
    tau2, w, _beta, resid, _rss, vcov, _ll = out
    leverage = np.sum((X @ vcov) * X, axis=1)
    inner = -w + (w * resid) ** 2
    if method == "REML":
        inner = inner + w * w * leverage
    return 0.5 * (Z.T @ (tau2 * inner))


def _fit_alpha(y, v, X, Z, method: str, max_iter: int, tol: float) -> dict:
    """Minimise the profile likelihood over the scale coefficients."""
    base = math.log(max(float(np.var(y)), 1e-3))
    first = np.zeros(Z.shape[1])
    first[0] = base
    starts = [first]
    if Z.shape[1] > 1:
        for slope in (0.1, -0.1):
            s = np.zeros(Z.shape[1])
            s[0] = base
            s[1] = slope
            starts.append(s)

    opts = {"maxiter": int(max_iter), "gtol": float(tol)}
    best = None
    for start in starts:
        try:
            res = optimize.minimize(_neg_loglik, start, args=(y, v, X, Z, method),
                                    jac=_neg_grad_alpha, method="BFGS",
                                    options=opts)
        except (ValueError, FloatingPointError):        # pragma: no cover
            continue
        if not np.all(np.isfinite(res.x)) or not np.isfinite(res.fun):
            continue
        if best is None or float(res.fun) < best[0]:
            best = (float(res.fun), res.x.copy(), int(res.nit))
    if best is None:                                    # pragma: no cover
        raise RuntimeError(
            "the location-scale likelihood could not be evaluated at any "
            "starting point")

    # a derivative-free pass, because BFGS stops short on a flat ridge and the
    # scale surface is flat when the trend is small
    polish = optimize.minimize(
        _neg_loglik, best[1], args=(y, v, X, Z, method), method="Nelder-Mead",
        options={"maxiter": 2000 * Z.shape[1], "xatol": 1e-10, "fatol": 1e-13})
    if np.all(np.isfinite(polish.x)) and float(polish.fun) <= best[0]:
        best = (float(polish.fun), polish.x.copy(), int(polish.nit))
    final = optimize.minimize(_neg_loglik, best[1], args=(y, v, X, Z, method),
                              jac=_neg_grad_alpha, method="BFGS", options=opts)
    if np.all(np.isfinite(final.x)) and float(final.fun) <= best[0]:
        best = (float(final.fun), final.x.copy(), int(final.nit))

    alpha = best[1]
    out = _profile(y, v, X, Z, alpha, method)
    assert out is not None
    tau2, w, beta, resid, rss, vcov, loglik = out
    grad = _grad_alpha(alpha, y, v, X, Z, method)
    return {"alpha": alpha, "tau2": tau2, "w": w, "beta": beta,
            "resid": resid, "rss": rss, "vcov": vcov, "loglik": loglik,
            "converged": bool(np.max(np.abs(grad)) < 1e-4), "n_iter": best[2]}


def _penalty_matrix(n_bend: int) -> np.ndarray:
    """``diag(0, 0, 1, ..., 1)``: the penalty touches the bending columns only."""
    q = 2 + int(n_bend)
    pen = np.zeros(q)
    pen[2:] = 1.0
    return np.diag(pen)


def _neg_penalised(alpha, y, v, X, Z, method, lam: float, pen: np.ndarray):
    """Profile negative log-likelihood plus ``lam/2 * alpha' P alpha``."""
    value = _neg_loglik(alpha, y, v, X, Z, method)
    if not np.isfinite(value):
        return np.inf
    return value + 0.5 * float(lam) * float(alpha @ pen @ alpha)


def _grad_penalised(alpha, y, v, X, Z, method, lam: float,
                    pen: np.ndarray) -> np.ndarray:
    """Gradient of :func:`_neg_penalised`."""
    grad = _neg_grad_alpha(alpha, y, v, X, Z, method)
    if not np.all(np.isfinite(grad)):
        return grad
    return grad + float(lam) * (pen @ alpha)


def _gamma_information(y, v, X, Z, alpha, method: str,
                       m: int) -> np.ndarray:
    """Observed information of the profile likelihood in the bending block.

    Central differences on the profiled log-likelihood, moving only the last
    ``m`` coefficients.  Needed for the Laplace correction that estimates
    :math:`\\sigma_s`, which the penalised objective alone cannot do: the
    penalty is smallest at :math:`\\gamma = 0`, so minimising it directly drives
    :math:`\\sigma_s \\to 0` regardless of the data.
    """
    q = alpha.size
    step = 1e-4

    def at(di: float, dj: float, i: int, j: int) -> float:
        point = alpha.copy()
        point[q - m + i] += di
        point[q - m + j] += dj
        out = _profile(y, v, X, Z, point, method)
        return -np.inf if out is None else float(out[-1])

    info = np.zeros((m, m))
    for i in range(m):
        for j in range(i, m):
            value = (at(step, step, i, j) - at(step, -step, i, j)
                     - at(-step, step, i, j) + at(-step, -step, i, j))
            info[i, j] = info[j, i] = -value / (4.0 * step * step)
    return info


def _fit_spline(y, v, X, time, basis: _SplineBasis, method: str,
                max_iter: int, tol: float) -> dict:
    """Fit the scale with a spline and one wiggliness hyperparameter.

    The bending coefficients get a Gaussian prior
    :math:`\\gamma \\sim N(0, \\sigma_s^2 I)`, so for a *given* :math:`\\sigma_s`
    the conditional mode solves a ridge problem.  :math:`\\sigma_s` itself is
    then estimated by the **Laplace approximation to the marginal likelihood**
    of that random effect (the mixed-model view of a penalised spline: Eilers &
    Marx 1996, Ruppert, Wand & Carroll 2003).

    Profiling the penalised objective over :math:`\\sigma_s` would not work and
    is worth saying out loud: the penalty is minimised at :math:`\\gamma = 0`,
    so the objective improves monotonically as :math:`\\sigma_s \\to 0` and the
    fit would always report the linear model.  The marginal likelihood adds the
    two terms that make the comparison honest — the prior normalisation
    :math:`-\\tfrac{m}{2}\\log\\sigma_s^2` and the Laplace volume
    :math:`\\tfrac12\\log\\det(J + \\lambda I)` — and those are what can prefer a
    positive wiggliness when the data show one.

    Returns the same dictionary as :func:`_fit_alpha` plus ``sigma_s`` (the
    fitted wiggliness), ``edf`` (effective degrees of freedom spent by the
    bending block) and ``laplace_deviance``.
    """
    Z = basis.matrix(np.asarray(time, dtype=float))
    q = Z.shape[1]
    m = basis.n_bend
    pen = _penalty_matrix(m)
    # start from the *linear* scale fit: the penalised surface has a flat trough
    # along the linear fit and a zero start sits at its worst point
    linear_fit = _fit_alpha(y, v, X, _design(time, "linear", basis.centre),
                            method, max_iter, tol)
    start = np.zeros(q)
    start[:2] = linear_fit["alpha"][:2]
    if not np.isfinite(start[0]):                        # pragma: no cover
        start[0] = math.log(max(float(np.var(y)), 1e-3))

    def solve_for(lam: float, from_alpha: np.ndarray) -> np.ndarray:
        # bounds, not a plain BFGS: the profile likelihood is ``inf`` outside
        # the region where every fitted variance is positive, and an unbounded
        # line search that steps into it collapses back to its start point and
        # reports convergence at a coefficient vector it never improved.  The
        # bounds are wide enough to contain any sane fit (tau^2 between 1e-8
        # and 1e3) and they make every trial point feasible.
        bounds = [(-18.0, 7.0), (-50.0, 50.0)] + [(-50.0, 50.0)] * m
        res = optimize.minimize(
            _neg_penalised, from_alpha, args=(y, v, X, Z, method, lam, pen),
            jac=_grad_penalised, method="L-BFGS-B", bounds=bounds,
            options={"maxiter": int(max_iter), "gtol": float(tol)})
        if not np.all(np.isfinite(res.x)):                # pragma: no cover
            return from_alpha
        return res.x

    def deviance(s: float, from_alpha: np.ndarray) -> tuple[float, np.ndarray]:
        """-2 times the Laplace-approximated marginal log-likelihood."""
        lam = math.exp(-2.0 * float(s))
        point = solve_for(lam, from_alpha)
        out = _profile(y, v, X, Z, point, method)
        if out is None:                                  # pragma: no cover
            return np.inf, point
        loglik = float(out[-1])
        ridge = 0.5 * lam * float(point @ pen @ point)
        info = _gamma_information(y, v, X, Z, point, method, m)
        if not np.all(np.isfinite(info)):                # pragma: no cover
            return np.inf, point
        sign, logdet = np.linalg.slogdet(info + lam * np.eye(m))
        if sign <= 0:                                    # pragma: no cover
            return np.inf, point
        value = 2.0 * (-loglik + ridge - 0.5 * m * math.log(lam)
                       + 0.5 * float(logdet))
        return float(value), point

    # sigma_s on a log grid from 1e-3 (indistinguishable from linear) to 30:
    # the top is far beyond any wiggliness a scale function needs, so reaching
    # it is a statement that the data want an effectively unpenalised spline
    grid = np.linspace(math.log(1e-3), math.log(30.0), 13)
    values = []
    point = start.copy()
    for s in grid:
        value, point = deviance(float(s), point)
        values.append(value)
    best_index = int(np.argmin(values))
    lo = grid[max(best_index - 1, 0)]
    hi = grid[min(best_index + 1, grid.size - 1)]
    res = optimize.minimize_scalar(
        lambda s: deviance(float(s), point)[0], bounds=(lo, hi),
        method="bounded", options={"xatol": 1e-3})
    candidates = [(values[best_index], float(grid[best_index]))]
    if np.isfinite(res.fun):
        candidates.append((float(res.fun), float(res.x)))
    dev, log_sigma = min(candidates)
    _, alpha = deviance(log_sigma, point)
    sigma_s = float(math.exp(log_sigma))
    lam = math.exp(-2.0 * log_sigma)

    out = _profile(y, v, X, Z, alpha, method)
    assert out is not None
    tau2, w, beta, resid, rss, vcov, loglik = out

    # effective degrees of freedom: how many of the m bending columns the
    # penalised fit actually spends, trace((J + lam P)^-1 J) over that block
    info_alpha = _gamma_information(y, v, X, Z, alpha, method, m)
    edf = 0.0
    try:
        solved = np.linalg.solve(info_alpha + lam * np.eye(m), info_alpha)
        edf = float(np.clip(np.trace(solved), 0.0, float(m)))
    except np.linalg.LinAlgError:                        # pragma: no cover
        edf = float(m)
    grad = _grad_penalised(alpha, y, v, X, Z, method, lam, pen)
    return {"alpha": alpha, "tau2": tau2, "w": w, "beta": beta,
            "resid": resid, "rss": rss, "vcov": vcov, "loglik": loglik,
            "sigma_s": sigma_s, "edf": edf, "lam": lam, "Z": Z,
            "laplace_deviance": float(dev),
            "converged": bool(np.max(np.abs(grad)) < 1e-3),
            "n_iter": int(getattr(res, "nfev", 0))}


def _observed_information(y, v, X, Z, beta, alpha, method: str) -> np.ndarray:
    """Observed information of the *joint* (beta, alpha) log-likelihood.

    Numerical, by central differences.  The ``alpha`` block of the inverse of
    this matrix equals the inverse of the profile information for ``alpha``
    (the Schur complement identity), which is what metafor's ``va`` is; the
    ``beta`` block is the quantity metafor replaces with the GLS variance.
    """
    p, q = X.shape[1], Z.shape[1]
    n = p + q
    theta = np.concatenate([beta, alpha])
    sign_xx, logdet_xx = np.linalg.slogdet(X.T @ X)
    assert sign_xx > 0
    k = y.size

    def loglik(point: np.ndarray) -> float:
        b, a = point[:p], point[p:]
        d = v + np.exp(Z @ a)
        if not np.all(np.isfinite(d)) or np.any(d <= 0.0):
            return float("nan")
        w = 1.0 / d
        resid = y - X @ b
        value = (-0.5 * float(np.sum(np.log(d)))
                 - 0.5 * float(np.sum(w * resid * resid)))
        sign, logdet = np.linalg.slogdet(X.T @ (w[:, None] * X))
        if sign <= 0:
            return float("nan")
        if method == "REML":
            value += (-0.5 * (k - p) * math.log(2.0 * math.pi)
                      + 0.5 * float(logdet_xx) - 0.5 * float(logdet))
        else:
            value -= 0.5 * k * math.log(2.0 * math.pi)
        return value

    hess = np.full((n, n), np.nan)
    steps = np.maximum(1e-4, 1e-4 * np.abs(theta))
    for i in range(n):
        for j in range(i, n):
            ei = np.zeros(n)
            ej = np.zeros(n)
            ei[i] = steps[i]
            ej[j] = steps[j]
            value = (loglik(theta + ei + ej) - loglik(theta + ei - ej)
                     - loglik(theta - ei + ej) + loglik(theta - ei - ej))
            hess[i, j] = hess[j, i] = -value / (4.0 * steps[i] * steps[j])
    return hess


def _lrt_once(y, v, X, Z, method: str, max_iter: int = 300,
              tol: float = 1e-8) -> float | None:
    """LRT statistic for one sample, or ``None`` when a fit fails."""
    try:
        full = _fit_alpha(y, v, X, Z, method, max_iter, tol)
        null = _fit_alpha(y, v, X, Z[:, :1], method, max_iter, tol)
    except (RuntimeError, np.linalg.LinAlgError):        # pragma: no cover
        return None
    return 2.0 * (full["loglik"] - null["loglik"])


# --------------------------------------------------------------------------- #
# the frequentist result
# --------------------------------------------------------------------------- #
@dataclass
class TauTrendResult:
    """A fitted location-scale model with time in the mean and the variance.

    Attributes:
        beta: location coefficients, on the centred time scale.
        se_beta: location standard errors as the ``se`` argument asked for.
        se_beta_profile: the same with the scale uncertainty propagated; always
            computed, so the two can be compared.
        alpha: scale coefficients, :math:`\\log\\tau^2(t) = z(t)'\\alpha`.
        se_alpha: scale standard errors, from the joint observed information.
        tau2: the fitted :math:`\\tau^2_i`, one per study.
        time_centre: the constant subtracted before the design was built.
        s2w: the Knapp–Hartung scale factor applied to the location variance.
        loglik: the maximised likelihood, including metafor's
            :math:`\\tfrac12\\log|X'X|` term under REML so that the value is
            directly comparable with ``logLik`` of the R fit.
    """

    beta: np.ndarray
    se_beta: np.ndarray
    se_beta_profile: np.ndarray
    vcov_beta: np.ndarray
    vcov_beta_profile: np.ndarray
    alpha: np.ndarray
    se_alpha: np.ndarray
    vcov_alpha: np.ndarray
    vcov: np.ndarray
    tau2: np.ndarray
    time_centre: float
    mu_kind: str
    tau_kind: str
    method: str
    test: str
    level: float
    s2w: float
    loglik: float
    k: int
    p: int
    q: int
    rss: float
    converged: bool
    kind: str = "frequentist"
    #: knot vector of a spline scale, ``None`` for the polynomial scales
    knots: np.ndarray | None = None
    #: fitted wiggliness hyperparameter of a spline scale
    sigma_s: float | None = None
    #: effective degrees of freedom the spline's bending block spends
    edf: float | None = None
    _basis: _SplineBasis | None = field(default=None, repr=False)
    _refit: Callable[[], dict] | None = field(default=None, repr=False)

    # -- names and degrees of freedom -------------------------------------- #
    @property
    def names_beta(self) -> list[str]:
        return _names(self.mu_kind, scale=False)

    @property
    def names_alpha(self) -> list[str]:
        if self.tau_kind == "spline" and self._basis is not None:
            return self._basis.names
        return _names(self.tau_kind, scale=True)

    @property
    def df(self) -> int | None:
        """Degrees of freedom for the location tests, or ``None`` for z."""
        return self.k - self.p if self.test in ("knha", "t") else None

    @property
    def df_alpha(self) -> int | None:
        if self.test not in ("knha", "t"):
            return None
        if self.tau_kind == "spline" and self.edf is not None:
            # the spline spends edf of its bending degrees of freedom, so the
            # scale df is k minus two coefficients minus that, not k - q
            return max(int(round(self.k - 2.0 - float(self.edf))), 1)
        return self.k - self.q

    def _crit(self, level: float, df: int | None) -> float:
        if df is None or df <= 0:
            return float(stats.norm.ppf(0.5 + level / 2.0))
        return float(stats.t.ppf(0.5 + level / 2.0, df))

    # -- tables ------------------------------------------------------------ #
    def summary(self, level=None) -> list[dict]:
        """Coefficient table: estimate, SE, statistic, p-value and interval."""
        lvl = self.level if level is None else 1.0 - level_to_alpha(level)
        out = []
        for part, names, est, ses, df in (
                ("location", self.names_beta, self.beta, self.se_beta, self.df),
                ("scale", self.names_alpha, self.alpha, self.se_alpha,
                 self.df_alpha)):
            crit = self._crit(lvl, df)
            for j, name in enumerate(names):
                se = float(ses[j])
                stat = float(est[j] / se) if se > 0 else float("nan")
                p = (float(2.0 * stats.t.sf(abs(stat), df)) if df
                     else float(2.0 * stats.norm.sf(abs(stat))))
                out.append({"part": part, "term": name,
                            "estimate": float(est[j]), "se": se,
                            "statistic": stat, "p_value": p,
                            "q_low": float(est[j] - crit * se),
                            "q_high": float(est[j] + crit * se)})
        return out

    def test_trend(self, method: str = "wald", *, n_boot: int = 999,
                   seed=None, n_jobs: int | None = None) -> dict:
        """Test that the scale does not change with time.

        ``"wald"`` is the Wald statistic on the scale slopes, referred to an F
        with ``(q-1, k-q)`` degrees of freedom when ``test`` is knha and to
        :math:`\\chi^2_{q-1}` otherwise.  ``"lrt"`` is the likelihood-ratio
        statistic against the same location model with ``tau="none"``, referred
        to :math:`\\chi^2_{q-1}` — valid when heterogeneity is present and only
        its *shape* is in question.

        ``"bootstrap"`` is a parametric bootstrap of that LRT under the fitted
        null.  It is the only one of the three that keeps its level when the
        null is **no heterogeneity at all**, because there :math:`\\alpha_0`
        sits on its boundary and the chi-square reference is wrong.  It costs
        ``n_boot`` pairs of refits.

        ``n_jobs`` spreads those refits over processes.  Every bootstrap sample
        is drawn first, serially, from the seeded generator, so the samples do
        not depend on how many workers run them and the p-value is identical to
        the serial one to the last bit (``tests/test_parallel.py``).
        """
        if self.q == 1:
            raise ValueError(
                "this fit has a constant scale: there is no scale slope to test")
        if self.tau_kind == "spline":
            r = 1.0 + float(self.edf or 0.0)
        else:
            r = float(self.q - 1)
        if method == "wald":
            if self.tau_kind == "spline":
                raise ValueError(
                    "a penalised spline has no fixed number of scale "
                    "coefficients to form a Wald statistic from: the shape it "
                    "spends is estimated.  Use method='lrt' (degrees of "
                    "freedom = 1 + edf) or 'bootstrap'")
            block = self.vcov_alpha[1:, 1:]
            if (not np.all(np.isfinite(block))
                    or np.linalg.matrix_rank(block) < int(r)):
                raise ValueError(
                    "the scale coefficients are not identified in this fit, so "
                    "there is no Wald statistic to compute; this happens when "
                    "the heterogeneity is driven to zero.  Use method='lrt' or "
                    "'bootstrap', which compare likelihoods")
            stat = float(self.alpha[1:] @ np.linalg.solve(block, self.alpha[1:]))
            if self.df_alpha:
                f = stat / r
                return {"method": "wald", "statistic": f, "df1": int(r),
                        "df2": self.df_alpha,
                        "p_value": float(stats.f.sf(f, int(r), self.df_alpha)),
                        "reference": "F"}
            return {"method": "wald", "statistic": stat, "df1": int(r),
                    "df2": None,
                    "p_value": float(stats.chi2.sf(stat, int(r))),
                    "reference": "chi2"}
        if self._refit is None:                          # pragma: no cover
            raise RuntimeError("this result carries no refit closure")
        null = self._refit()
        stat = max(2.0 * (self.loglik - null["loglik"]), 0.0)
        if method == "lrt":
            return {"method": "lrt", "statistic": float(stat), "df1": r,
                    "df2": None, "p_value": float(stats.chi2.sf(stat, r)),
                    "reference": "chi2", "null_loglik": float(null["loglik"])}
        if method == "bootstrap":
            rng = np.random.default_rng(seed)
            centre = null["X"] @ null["beta"]
            sd = np.sqrt(null["v"] + null["tau2"])
            # Draw every replicate before any of them is fitted: the stream a
            # replicate consumes must not depend on the worker that fits it.
            draws = [centre + rng.normal(0.0, 1.0, centre.size) * sd
                     for _ in range(int(n_boot))]
            values = map_ordered(
                partial(_lrt_once, v=null["v"], X=null["X"], Z=null["Z"],
                        method=null["method"]),
                draws, n_jobs=n_jobs)
            count = sum(1 for value in values if value is not None and value >= stat)
            return {"method": "bootstrap", "statistic": float(stat), "df1": r,
                    "df2": None,
                    "p_value": float((count + 1) / (int(n_boot) + 1)),
                    "reference": "parametric bootstrap", "n_boot": int(n_boot)}
        raise ValueError(
            f"'method' must be 'wald', 'lrt' or 'bootstrap', not {method!r}")

    # -- the curve --------------------------------------------------------- #
    def predict(self, time, level=None) -> dict:
        """Pooled effect, heterogeneity and a time-varying prediction interval.

        ``pi_low``/``pi_high`` are for a **new study's true effect** at that
        time,

        .. math::

            \\hat\\mu(t) \\pm t_{df}
            \\sqrt{\\widehat{\\mathrm{Var}}\\{\\hat\\mu(t)\\} + \\hat\\tau^2(t)}.

        That interval does not carry the uncertainty of :math:`\\hat\\tau^2(t)`
        — no frequentist interval of this form can.  :func:`bayes_tau_trend`
        does, and the difference between the two is worth looking at before
        quoting either.
        """
        lvl = self.level if level is None else 1.0 - level_to_alpha(level)
        t = np.atleast_1d(np.asarray(time, dtype=float))
        X = _design(t, self.mu_kind, self.time_centre)
        if self.tau_kind == "spline":
            if self._basis is None:                     # pragma: no cover
                raise RuntimeError("this spline fit carries no basis")
            Z = self._basis.matrix(t)
        else:
            Z = _design(t, self.tau_kind, self.time_centre)
        mu = X @ self.beta
        var = np.maximum(np.sum((X @ self.vcov_beta) * X, axis=1), 0.0)
        tau2 = np.exp(Z @ self.alpha)
        crit = self._crit(lvl, self.df)
        se = np.sqrt(var)
        return {"time": t, "mu": mu, "se": se, "tau2": tau2,
                "q_low": mu - crit * se, "q_high": mu + crit * se,
                "pi_low": mu - crit * np.sqrt(var + tau2),
                "pi_high": mu + crit * np.sqrt(var + tau2),
                "level": lvl, "df": self.df}

    def ratio(self, level=None) -> dict:
        """The change in :math:`\\tau^2` per unit of time, as a ratio.

        For a linear scale model this one number is the whole statement about
        the trend: :math:`\\exp(\\alpha_1)` is how much the heterogeneity
        variance multiplies for a one-unit increase in ``time``.  The interval
        is built on the log scale and exponentiated, so it cannot go negative.
        """
        if self.tau_kind != "linear":
            raise ValueError(
                "the ratio is defined for a linear scale model; this fit has "
                f"tau={self.tau_kind!r}")
        lvl = self.level if level is None else 1.0 - level_to_alpha(level)
        crit = self._crit(lvl, self.df_alpha)
        return {"log_ratio": float(self.alpha[1]),
                "se": float(self.se_alpha[1]),
                "ratio": float(np.exp(self.alpha[1])),
                "q_low": float(np.exp(self.alpha[1] - crit * self.se_alpha[1])),
                "q_high": float(np.exp(self.alpha[1] + crit * self.se_alpha[1])),
                "level": lvl}

    def to_dict(self) -> dict:
        out = {
            "model": "location-scale with time in the mean and the variance",
            "kind": "frequentist", "method": self.method, "test": self.test,
            "mu": self.mu_kind, "tau": self.tau_kind,
            "beta": self.beta.tolist(), "se_beta": self.se_beta.tolist(),
            "se_beta_profile": self.se_beta_profile.tolist(),
            "alpha": self.alpha.tolist(), "se_alpha": self.se_alpha.tolist(),
            "s2w": float(self.s2w), "loglik": float(self.loglik),
            "k": int(self.k), "p": int(self.p), "q": int(self.q),
            "level": float(self.level), "time_centre": float(self.time_centre),
            "converged": bool(self.converged),
        }
        if self.knots is not None:
            out["knots"] = [float(v) for v in self.knots]
            out["sigma_s"] = float(self.sigma_s or 0.0)
            out["edf"] = float(self.edf or 0.0)
        return out

    def __str__(self) -> str:
        rows = self.summary()
        width = max(len(r["term"]) for r in rows) + 2
        lines = [f"Location-scale model ({self.method}, test={self.test}), "
                 f"k = {self.k}",
                 f"{'':<{width}}{'estimate':>10}{'se':>9}{'stat':>9}"
                 f"{'p':>9}{'lo':>9}{'hi':>9}"]
        for r in rows:
            lines.append(f"{r['term']:<{width}}{r['estimate']:>10.4f}"
                         f"{r['se']:>9.4f}{r['statistic']:>9.3f}"
                         f"{r['p_value']:>9.4f}{r['q_low']:>9.4f}"
                         f"{r['q_high']:>9.4f}")
        return "\n".join(lines)


# --------------------------------------------------------------------------- #
# the frequentist fit
# --------------------------------------------------------------------------- #
def tau_trend(yi, vi, time, *, mu: str = "linear", tau: str = "linear",
              method: str = "REML", test: str = "knha", se: str = "gls",
              level=0.95, max_iter: int = 500, tol: float = 1e-10,
              max_ratio: float = 100.0, knots=None,
              n_knots: int = 4) -> TauTrendResult:
    """Fit :math:`y_i \\sim N(x_i'\\beta,\\ v_i + \\exp(z_i'\\alpha))`.

    Args:
        yi: effect estimates.
        vi: their sampling variances.
        time: the continuous index — year, follow-up time, dose, any ordering.
            Centred internally, so the intercepts are the values at the mean
            time and ``alpha[1]`` is per unit of the original scale.
        mu: shape of the mean, ``"none"``, ``"linear"`` or ``"quadratic"``.
        tau: shape of :math:`\\log\\tau^2`, ``"none"``, ``"linear"``,
            ``"quadratic"`` or ``"spline"``.  ``"none"`` gives a
            constant :math:`\\tau^2`, which makes the fit an ordinary
            random-effects model with moderators.  ``"spline"`` puts a
            restricted cubic spline on the log scale, orthogonalised against
            ``[1, t]`` and shrunk by one wiggliness hyperparameter
            :math:`\\sigma_s` — see :class:`TauTrendResult` for what is
            reported and why a spline is not allowed to spend ``nk - 2`` free
            coefficients.
        method: ``"REML"`` (default) or ``"ML"``.
        test: ``"knha"`` (default), ``"t"`` or ``"z"``.  Knapp–Hartung scales
            the location variance by :math:`RSS/(k-p)` and refers the location
            tests to a t with ``k-p`` degrees of freedom; the scale tests use
            ``k-q``.
        se: ``"gls"`` for :math:`(X'WX)^{-1}`, what metafor reports and what the
            parity tests check; ``"profile"`` for the joint-information block,
            which propagates the scale uncertainty.  Both are always stored.
        level: confidence level, as ``0.95``, ``95`` or ``0.05``.
        max_iter: optimiser iterations.
        tol: optimiser gradient tolerance.
        max_ratio: warn when the fitted :math:`\\tau^2(t)` spans more than this
            factor across the observed range.  A log-linear scale fitted to data
            with almost no heterogeneity can run to :math:`\\pm\\infty` — on a
            simulation with a constant :math:`\\tau^2 = 0.02` the estimate was
            :math:`\\alpha_1 = 10.2`, a 27 000-fold change, which ``metafor``
            reports just as happily.  Set to ``inf`` to silence it.
        knots: explicit knot positions for ``tau="spline"``; the default is
            :func:`pimeta.default_knots` on the centred time.
        n_knots: number of knots for ``tau="spline"``, 4 by default (two
            bending columns).

    Returns:
        :class:`TauTrendResult`.

    Raises:
        ValueError: for mismatched lengths, fewer than three studies, a
            non-positive variance, a constant ``time``, an unknown shape, or a
            number of location coefficients that leaves nothing for the scale.
    """
    y, v, t = _check_inputs(yi, vi, time)
    for kind, name in ((mu, "mu"), (tau, "tau")):
        if kind not in _KINDS:
            raise ValueError(
                f"unknown {name} model {kind!r}; choose from {', '.join(_KINDS)}")
    if mu == "spline":
        raise ValueError(
            "a spline is available for the scale only: the mean model is "
            "'none', 'linear' or 'quadratic'.  A spline mean belongs in "
            "tau_trend's caller (fit the mean with ma.rma(mods=...) first)")
    if method not in ("ML", "REML"):
        raise ValueError("'method' must be 'ML' or 'REML'")
    if test not in ("z", "knha", "t"):
        raise ValueError("'test' must be 'z', 'knha' or 't'")
    if se not in ("gls", "profile"):
        raise ValueError("'se' must be 'gls' or 'profile'")
    if mu == "none" and tau != "none":
        warnings.warn(
            "the mean effect is held constant while the heterogeneity may "
            "change with time: a time trend in the mean will be read as a "
            "trend in tau^2.  Pass mu='linear' unless that is the model you "
            "mean to fit", RuntimeWarning, stacklevel=2)

    centre = float(np.mean(t))
    X = _design(t, mu, centre)
    k, p = y.size, X.shape[1]
    basis = None
    if tau == "spline":
        basis = _spline_basis(t, centre, knots=knots, n_knots=n_knots)
        m = basis.n_bend
        if k < p + m + _SPLINE_MIN_EXTRA:
            raise ValueError(
                f"a {n_knots}-knot spline scale needs at least "
                f"{p + m + _SPLINE_MIN_EXTRA} studies with {p} location "
                f"coefficients; this fit has {k}.  Use tau='linear', or fewer "
                "knots")
        fit = _fit_spline(y, v, X, t, basis, method, max_iter, tol)
        Z = fit["Z"]
    else:
        Z = _design(t, tau, centre)
        if p >= k:
            raise ValueError(
                f"{p} location coefficients for {k} studies leaves nothing to "
                "estimate the scale from")
        fit = _fit_alpha(y, v, X, Z, method, max_iter, tol)
    q = Z.shape[1]

    info = _observed_information(y, v, X, Z, fit["beta"], fit["alpha"], method)
    if basis is not None:
        # the penalised information, not the plain one: with a ridge penalty
        # the covariance of the scale coefficients is the Bayesian posterior
        # covariance (J + lambda P)^-1, which is what a P-spline reports
        penalised = info.copy()
        penalised[p:, p:] += float(fit["lam"]) * _penalty_matrix(basis.n_bend)
        info = penalised
    try:
        vcov = np.linalg.inv(info)
    except np.linalg.LinAlgError:
        # at a boundary fit -- tau^2 driven to zero -- the information matrix
        # becomes numerically singular; a pseudo-inverse at least reports the
        # directions that are identified
        warnings.warn(
            "the observed information is singular, which happens when tau^2 "
            "is driven to zero; the standard errors are computed from a "
            "pseudo-inverse and the scale coefficients are not identified",
            RuntimeWarning, stacklevel=2)
        vcov = np.linalg.pinv(info)
    vcov_beta_profile = vcov[:p, :p]
    vcov_alpha = vcov[p:, p:]

    # Knapp-Hartung: metafor scales the location variance and leaves the scale
    # variance alone, and the tests follow it
    s2w = 1.0
    if test in ("knha", "t") and k > p:
        s2w = float(fit["rss"] / (k - p))
    if test in ("knha", "t"):
        vcov_beta_gls = s2w * fit["vcov"]
        vcov_beta_profile = s2w * vcov_beta_profile
    else:
        vcov_beta_gls = fit["vcov"]
    vcov_beta = vcov_beta_gls if se == "gls" else vcov_beta_profile

    def _refit() -> dict:
        null = _fit_alpha(y, v, X, Z[:, :1], method, max_iter, tol)
        return {"loglik": float(null["loglik"]), "beta": null["beta"],
                "tau2": null["tau2"], "X": X, "Z": Z, "v": v,
                "method": method}

    if tau != "none" and np.isfinite(max_ratio):
        if basis is not None:
            fitted = np.exp(basis.matrix(t) @ fit["alpha"])
            ratio = float(np.max(fitted) / max(np.min(fitted), 1e-300))
        else:
            span = float(np.ptp(t))
            ratio = math.exp(abs(float(fit["alpha"][1]) * span)) if q > 1 else 1.0
        if ratio > max_ratio:
            warnings.warn(
                f"the fitted tau^2 changes by a factor of {ratio:.3g} across the "
                "observed range of 'time', which is more than two orders of "
                "magnitude: a log-linear scale fitted to data with little "
                "heterogeneity can run to infinity, and the shape it reports is "
                "then an artefact.  Check the scale coefficients' standard "
                "errors before reading the trend",
                RuntimeWarning, stacklevel=2)

    return TauTrendResult(
        beta=fit["beta"],
        se_beta=np.sqrt(np.maximum(np.diag(vcov_beta), 0.0)),
        se_beta_profile=np.sqrt(np.maximum(np.diag(vcov_beta_profile), 0.0)),
        vcov_beta=vcov_beta, vcov_beta_profile=vcov_beta_profile,
        alpha=fit["alpha"],
        se_alpha=np.sqrt(np.maximum(np.diag(vcov_alpha), 0.0)),
        vcov_alpha=vcov_alpha, vcov=vcov, tau2=fit["tau2"],
        time_centre=centre, mu_kind=mu, tau_kind=tau, method=method,
        test=test, level=1.0 - level_to_alpha(level), s2w=s2w,
        loglik=float(fit["loglik"]), k=k, p=p, q=q, rss=float(fit["rss"]),
        converged=bool(fit["converged"]),
        knots=None if basis is None else basis.knots.copy(),
        sigma_s=None if basis is None else float(fit["sigma_s"]),
        edf=None if basis is None else float(fit["edf"]),
        _basis=basis, _refit=_refit)


# --------------------------------------------------------------------------- #
# the Bayesian fit
# --------------------------------------------------------------------------- #
@dataclass
class BayesTauTrendResult:
    """A sampled location-scale model with time in the mean and the variance.

    Only ``alpha`` is sampled.  Given ``alpha`` the model is Gaussian in
    ``beta``, so each draw of ``beta`` comes from its exact conditional
    posterior — the chain stays two- or three-dimensional, and the priors on
    the *shape* of the heterogeneity carry the regularisation instead of the
    sampler's inefficiency doing it by accident.
    """

    posterior: Posterior
    alpha: np.ndarray
    alpha_ci: np.ndarray
    beta: np.ndarray
    beta_ci: np.ndarray
    alpha_draws: np.ndarray | None = field(repr=False, default=None)
    beta_draws: np.ndarray | None = field(repr=False, default=None)
    time_centre: float = 0.0
    mu_kind: str = "linear"
    tau_kind: str = "linear"
    level: float = 0.95
    priors: dict = field(default_factory=dict)
    k: int = 0
    kind: str = "bayesian"

    def predict(self, time, level=None, seed: int = 0) -> dict:
        """Posterior of the pooled effect and of :math:`\\tau^2(t)`, and a prediction interval.

        Every number comes from the draws.  ``pi_low``/``pi_high`` add one
        sampled between-study deviation per draw, so they carry the uncertainty
        in :math:`\\hat\\tau^2(t)` as well — that is the substantive difference
        from :meth:`TauTrendResult.predict`, not a different formula for the
        same thing.
        """
        lvl = self.level if level is None else 1.0 - level_to_alpha(level)
        t = np.atleast_1d(np.asarray(time, dtype=float))
        X = _design(t, self.mu_kind, self.time_centre)
        Z = _design(t, self.tau_kind, self.time_centre)
        mu_draws = self.beta_draws @ X.T
        tau2_draws = np.exp(self.alpha_draws @ Z.T)
        noise = np.random.default_rng(seed).normal(0.0, 1.0, mu_draws.shape)
        pred = mu_draws + np.sqrt(tau2_draws) * noise
        tail = (1.0 - lvl) / 2.0
        lo, hi = np.quantile(mu_draws, [tail, 1.0 - tail], axis=0)
        plo, phi = np.quantile(pred, [tail, 1.0 - tail], axis=0)
        return {"time": t, "mu": mu_draws.mean(axis=0),
                "sd": mu_draws.std(axis=0, ddof=1),
                "q_low": lo, "q_high": hi,
                "tau2": tau2_draws.mean(axis=0),
                "pi_low": plo, "pi_high": phi, "level": lvl}

    def ratio(self, level=None) -> dict:
        """Posterior of :math:`\\tau^2(t+1)/\\tau^2(t)` for a linear scale model."""
        if self.tau_kind != "linear":
            raise ValueError(
                "the ratio is defined for a linear scale model; this fit has "
                f"tau={self.tau_kind!r}")
        lvl = self.level if level is None else 1.0 - level_to_alpha(level)
        draws = self.posterior.draws
        values = np.exp(draws[:, 1])
        tail = (1.0 - lvl) / 2.0
        lo, hi = np.quantile(values, [tail, 1.0 - tail])
        return {"mean": float(values.mean()), "sd": float(values.std(ddof=1)),
                "q_low": float(lo), "q_high": float(hi),
                "prob_above_one": float(np.mean(values > 1.0)), "level": lvl}

    def summary(self, level=None) -> list[dict]:
        lvl = self.level if level is None else 1.0 - level_to_alpha(level)
        return self.posterior.summary(level=lvl)

    def fit(self) -> dict:
        return {"n_chains": self.posterior.n_chains,
                "n_draws": self.posterior.n_draws,
                "accept": [float(v) for v in self.posterior.accept],
                "rhat_max": float(np.max(self.posterior.rhat())),
                "ess_min": float(np.min(self.posterior.ess()))}

    def to_dict(self) -> dict:
        return {"model": "Bayesian location-scale with time in mean and variance",
                "kind": "bayesian", "mu": self.mu_kind, "tau": self.tau_kind,
                "alpha": self.alpha.tolist(), "beta": self.beta.tolist(),
                "k": int(self.k), "level": float(self.level),
                "priors": dict(self.priors),
                "posterior": self.posterior.to_dict()}

    def __str__(self) -> str:
        rows = self.summary()
        width = max(len(r["term"]) for r in rows) + 2
        lines = [f"Bayesian location-scale model, k = {self.k}",
                 f"{'':<{width}}{'mean':>10}{'sd':>9}{'lo':>9}{'hi':>9}"]
        for r in rows:
            lines.append(f"{r['term']:<{width}}{r['mean']:>10.4f}"
                         f"{r['sd']:>9.4f}{r['q_low']:>9.4f}"
                         f"{r['q_high']:>9.4f}")
        return "\n".join(lines)


def bayes_tau_trend(yi, vi, time, *, mu: str = "linear", tau: str = "linear",
                    prior_beta_sd: float | None = None,
                    prior_alpha0_sd: float | None = None,
                    prior_alpha_sd: float | None = None,
                    level=0.95, mcmc: dict | None = None) -> BayesTauTrendResult:
    """Sample the posterior of the location-scale model with time-varying tau^2.

    Priors, stated rather than buried:

    - :math:`\\beta \\sim N(0, 100^2)` — flat enough that the location answers
      agree with the frequentist fit, proper enough that the marginal likelihood
      exists.
    - :math:`\\alpha_0 \\sim N(0, 2^2)` — a 95 % prior range of
      :math:`\\tau^2 \\in [0.02, 55]`, which excludes nothing anyone would
      plausibly report.  A first version used :math:`N(0, 1)`; on the fixture
      that pulled :math:`\\alpha_0` from the frequentist :math:`-2.61` to
      :math:`-2.31`, a 35 % change in :math:`\\tau^2` at the mean time, and a
      prior nobody asked for should not do that.
    - :math:`\\alpha_1, \\alpha_2 \\sim N(0, 0.3^2)` — the shrinkage that makes
      this the better answer at small ``k``.  It says "unless the data are
      clear, the heterogeneity does not change with time", which is what stops
      three influential early studies from deciding the shape.

    Args:
        yi, vi, time: as for :func:`tau_trend`.
        mu, tau: as for :func:`tau_trend`.
        prior_beta_sd, prior_alpha0_sd, prior_alpha_sd: override the scales
            above; all three are recorded on the result.
        level: credible level.
        mcmc: sampler overrides — ``n_draws``, ``warmup``, ``chains``,
            ``seed``, ``check``, ``rhat_max``, ``ess_min``.

    Returns:
        :class:`BayesTauTrendResult`.

    Raises:
        ValueError: for the input problems :func:`tau_trend` refuses, for an
            unknown ``mcmc`` key, and for a non-positive prior scale.
        RuntimeError: from :meth:`Posterior.check` when the draws did not mix.
    """
    y, v, t = _check_inputs(yi, vi, time)
    for kind, name in ((mu, "mu"), (tau, "tau")):
        if kind not in _KINDS:
            raise ValueError(
                f"unknown {name} model {kind!r}; choose from {', '.join(_KINDS)}")
    if tau == "spline" or mu == "spline":
        raise ValueError(
            "the sampler covers the polynomial scale models only: a spline "
            "scale adds a wiggliness hyperparameter that this sampler is not "
            "written for.  Use tau_trend(tau='spline') (penalised REML), or "
            "the Stan reference in tools/r_reference/tau2t_spline.stan, which "
            "fits exactly this model with a weakly informative prior")
    cfg = dict(_MCMC_DEFAULTS)
    if mcmc:
        unknown = set(mcmc) - set(cfg)
        if unknown:
            raise ValueError(
                f"unknown 'mcmc' setting(s): {', '.join(sorted(map(str, unknown)))}; "
                f"known keys are {', '.join(sorted(cfg))}")
        cfg.update(mcmc)
    beta_sd = (_BAYES_PRIORS["beta_sd"] if prior_beta_sd is None
               else float(prior_beta_sd))
    a0_sd = (_BAYES_PRIORS["alpha0_sd"] if prior_alpha0_sd is None
             else float(prior_alpha0_sd))
    a_sd = (_BAYES_PRIORS["alpha_sd"] if prior_alpha_sd is None
            else float(prior_alpha_sd))
    for name, value in (("prior_beta_sd", beta_sd),
                        ("prior_alpha0_sd", a0_sd),
                        ("prior_alpha_sd", a_sd)):
        if not np.isfinite(value) or value <= 0.0:
            raise ValueError(f"'{name}' must be finite and positive")

    centre = float(np.mean(t))
    X = _design(t, mu, centre)
    Z = _design(t, tau, centre)
    k, p, q = y.size, X.shape[1], Z.shape[1]
    if p >= k:
        raise ValueError(
            f"{p} location coefficients for {k} studies leaves nothing to "
            "estimate the scale from")
    eye_b = np.eye(p)

    def conditional(alpha: np.ndarray):
        """``beta | alpha, y``: exact mean and covariance."""
        d = v + np.exp(Z @ alpha)
        prec = X.T @ (X / d[:, None]) + eye_b / beta_sd ** 2
        cov = np.linalg.inv(prec)
        return cov @ (X.T @ (y / d)), cov

    def log_post(alpha: np.ndarray) -> float:
        if not np.all(np.isfinite(alpha)):
            return -np.inf
        d = v + np.exp(Z @ alpha)
        if not np.all(np.isfinite(d)) or np.any(d <= 0.0):
            return -np.inf
        # p(y | alpha) with beta integrated out under its Gaussian prior:
        # y ~ N(0, D + s_beta^2 X X')
        cov = np.diag(d) + beta_sd ** 2 * (X @ X.T)
        chol = np.linalg.cholesky(cov)
        z = np.linalg.solve(chol, y)
        value = -0.5 * float(z @ z) - float(np.sum(np.log(np.diag(chol))))
        value -= 0.5 * (alpha[0] / a0_sd) ** 2
        if q > 1:
            value -= 0.5 * float(alpha[1:] @ alpha[1:]) / a_sd ** 2
        return float(value)

    start = np.zeros(q)
    start[0] = math.log(max(float(np.var(y)), 1e-3))
    if not np.isfinite(log_post(start)):                # pragma: no cover
        start = np.zeros(q)

    post = sample_log_posterior(
        log_post, start, names=_names(tau, scale=True),
        n_draws=int(cfg["n_draws"]), warmup=int(cfg["warmup"]),
        chains=int(cfg["chains"]), seed=cfg["seed"], init_sd=0.1)
    if cfg["check"]:
        post.check(rhat_max=float(cfg["rhat_max"]), ess_min=float(cfg["ess_min"]))

    alpha_draws = np.asarray(post.draws, dtype=float)
    rng = np.random.default_rng(0)
    beta_draws = np.empty((alpha_draws.shape[0], p))
    for i, alpha in enumerate(alpha_draws):
        mean, cov = conditional(alpha)
        beta_draws[i] = rng.multivariate_normal(mean, cov, method="cholesky")
    tail = (1.0 - level_to_alpha(level)) / 2.0
    return BayesTauTrendResult(
        posterior=post,
        alpha=alpha_draws.mean(axis=0),
        alpha_ci=np.quantile(alpha_draws, [tail, 1.0 - tail], axis=0),
        beta=beta_draws.mean(axis=0),
        beta_ci=np.quantile(beta_draws, [tail, 1.0 - tail], axis=0),
        alpha_draws=alpha_draws, beta_draws=beta_draws,
        time_centre=centre, mu_kind=mu, tau_kind=tau,
        level=1.0 - level_to_alpha(level),
        priors={"beta_sd": beta_sd, "alpha0_sd": a0_sd, "alpha_sd": a_sd},
        k=k)
