"""Copas selection model: adjusting a pooled effect for selection bias.

Port of ``metasens::copas`` (Copas and Shi 2000, 2001).  The model says a
study is published when an unobserved score clears a threshold,

.. math::

    Z_i = \\gamma_0 + \\gamma_1 / se_i + \\delta_i, \\qquad
    \\delta_i \\sim N(0, 1),

so the probability that study :math:`i` is published is
:math:`\\Phi(\\gamma_0 + \\gamma_1/se_i)`: large studies almost surely, small
ones only if their result happens to be favourable.  :math:`\\gamma_1` measures
how strongly publication depends on precision, and :math:`\\gamma_0` how
generous publication is overall.  Only their combination is identified from
the observed studies, which is why the answer is a *surface*, not a point.

The likelihood conditions on publication.  For fixed :math:`(\\gamma_0,
\\gamma_1)` the free parameters are the pooled effect :math:`\\mu`, the
correlation :math:`\\rho` between the effect and the selection score, and the
between-study SD :math:`\\tau`; a second fit adds :math:`\\beta`, a residual
small-study slope, and the likelihood-ratio test of :math:`\\beta = 0` is the
test for *residual* selection bias.

How the reported adjusted estimate is chosen is worth stating plainly, because
it is not a maximum likelihood estimate of anything:

1. Evaluate :math:`\\hat\\mu` on a grid of :math:`(\\gamma_0, \\gamma_1)`;
   that is the sensitivity surface.
2. Trace the iso-:math:`\\hat\\mu` contours, fit a straight line to each, and
   combine their slopes.  Those lines are roughly parallel across contours, and
   the direction orthogonal to them is the direction in which the estimated
   effect moves, so it is the natural path to walk.
3. Walk that path from "everyone published" inwards, refitting at each step,
   and record the effect, its SE, the publicity probability,
   :math:`\\Phi(\\gamma_0+\\gamma_1/se_{\\max})`, the number of unpublished
   studies implied, and the residual-selection p-value.
4. Report the first point where residual selection is no longer significant —
   the least selection that would explain the funnel asymmetry.

Steps 2–4 are a graphical heuristic, and they are metasens' heuristic; they are
kept so that a published Copas analysis can be reproduced.

**For inference, use** :func:`copas_region` instead.  It keeps the whole set of
selection scenarios consistent with the data — the likelihood-ratio region for
:math:`(\\gamma_0, \\gamma_1)` — and reports the range of the adjusted effect
over that set, whether a scenario with no precision-dependent selection is
admissible, and whether every admissible scenario puts the effect on the same
side of zero.  Those are statements with a definable error rate; a single point
picked off a contour plot is not.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
from scipy import optimize, special, stats

__all__ = [
    "CopasRegion",
    "CopasResult",
    "copas",
    "copas_fit",
    "copas_loglik",
    "copas_region",
]


# --------------------------------------------------------------------------- #
# the likelihood
# --------------------------------------------------------------------------- #

def _inv_mills(u):
    """``phi(u) / Phi(u)``, the inverse Mills ratio, without overflowing.

    For very negative ``u`` both the density and the tail probability
    underflow, so the ratio is formed on the log scale.
    """
    u = np.asarray(u, dtype=float)
    out = np.exp(stats.norm.logpdf(u) - special.log_ndtr(u))
    return np.where(np.isfinite(out), out, 1e300)


def _sigma(sei, rho, u):
    """Sampling SD after conditioning on publication (Copas and Shi)."""
    lam = _inv_mills(u)
    denom = 1.0 - rho * rho * lam * (u + lam)
    return np.sqrt(sei * sei / denom), denom


def copas_loglik(params, yi, sei, gamma0, gamma1):
    """Negative log-likelihood of the Copas selection model.

    Args:
        params: ``(mu, rho, tau)``, or ``(mu, rho, tau, beta)`` to include the
            residual small-study slope.
        yi: observed effects.
        sei: their standard errors.
        gamma0, gamma1: the selection parameters.

    Returns:
        The negative log-likelihood, the quantity metasens'
        ``copas.loglik.without.beta`` returns (and ``.with.beta`` for the
        four-parameter form).

    Raises:
        ValueError: the parameter vector is not of length three or four.
    """
    yi = np.asarray(yi, dtype=float).ravel()
    sei = np.asarray(sei, dtype=float).ravel()
    p = np.asarray(params, dtype=float).ravel()
    if p.size not in (3, 4):
        raise ValueError(f"expected 3 or 4 parameters, got {p.size}")
    mu, rho, tau = float(p[0]), float(p[1]), float(p[2])
    shift = 0.0 if p.size == 3 else float(p[3]) * sei
    u = float(gamma0) + float(gamma1) / sei
    sigma, denom = _sigma(sei, rho, u)
    if np.any(denom <= 0.0) or not np.all(np.isfinite(sigma)):
        return 1e300
    s2t2 = sigma * sigma + tau * tau
    rho_t = rho * sigma / np.sqrt(s2t2)
    with np.errstate(invalid="ignore", divide="ignore"):
        v = (u + rho_t * (yi - mu - shift) / np.sqrt(s2t2)) \
            / np.sqrt(1.0 - rho_t * rho_t)
    v = np.maximum(v, -37.0)
    if not np.all(np.isfinite(v)):
        return 1e300
    resid = yi - mu - shift
    ll = 0.5 * np.log(s2t2) + resid * resid / (2.0 * s2t2) \
        - special.log_ndtr(v)
    total = float(np.sum(ll))
    return total if np.isfinite(total) else 1e300


@dataclass
class CopasFit:
    """One fit of the selection model at a fixed ``(gamma0, gamma1)``."""

    gamma0: float
    gamma1: float
    mu: float
    se_mu: float
    rho: float
    tau: float
    beta: float | None = None
    loglik: float = float("nan")
    converged: bool = False

    def to_dict(self) -> dict:
        return {
            "gamma0": self.gamma0, "gamma1": self.gamma1,
            "mu": self.mu, "se_mu": self.se_mu, "rho": self.rho,
            "tau": self.tau, "beta": self.beta, "loglik": self.loglik,
            "converged": self.converged,
        }


def copas_fit(yi, sei, gamma0, gamma1, *, rho_bound=0.9999, mu0=None,
              rho0=None, tau0=None, with_beta=False, se_mu=True,
              rho_lim=None):
    """Fit the selection model at one ``(gamma0, gamma1)``.

    Args:
        yi: observed effects.
        sei: their standard errors.
        gamma0, gamma1: the selection parameters to condition on.
        rho_bound: the correlation is constrained to ``[-rho_bound, rho_bound]``.
        mu0, rho0, tau0: starting values.  metasens starts from the ordinary
            random-effects estimate, ``-rho_bound/2`` and the random-effects
            :math:`\\hat\\tau`; the same defaults are used here.
        with_beta: also estimate the residual small-study slope.
        se_mu: compute the standard error from the numerical Hessian.
        rho_lim: explicit ``(lower, upper)`` bounds for the correlation.  The
            residual-selection fit needs this: metasens restricts :math:`\\rho`
            to the side of the funnel the missing studies were assumed to be
            on, which is narrower than the bound used for the surface and can
            matter for which optimum is reached.

    Returns:
        :class:`CopasFit`.
    """
    yi = np.asarray(yi, dtype=float).ravel()
    sei = np.asarray(sei, dtype=float).ravel()
    if mu0 is None:
        mu0 = float(np.sum(yi / sei ** 2) / np.sum(1.0 / sei ** 2))
    if rho0 is None:
        rho0 = -rho_bound / 2.0
    if tau0 is None:
        tau0 = max(float(np.std(yi, ddof=1)) if yi.size > 1 else 0.01, 1e-4)

    start = [mu0, rho0, tau0] if not with_beta else [mu0, rho0, tau0, 0.0]
    lo_rho, hi_rho = ((-rho_bound, rho_bound) if rho_lim is None
                      else (float(rho_lim[0]), float(rho_lim[1])))
    bounds = [(None, None), (lo_rho, hi_rho), (0.0, None)]
    if with_beta:
        bounds.append((None, None))

    res = optimize.minimize(
        copas_loglik, np.array(start, dtype=float),
        args=(yi, sei, gamma0, gamma1),
        method="L-BFGS-B", bounds=bounds,
        options={"maxiter": 2000, "maxfun": 20000, "ftol": 1e-14,
                 "gtol": 1e-10})

    p = np.asarray(res.x, dtype=float)
    at_bound = bool(abs(abs(p[1]) - hi_rho) < 1e-6
                    or abs(p[1] - lo_rho) < 1e-6)
    se_val = float("nan")
    if se_mu and not at_bound:
        try:
            hess = _num_hessian(
                lambda q: copas_loglik(q, yi, sei, gamma0, gamma1), p)
            var = float(np.linalg.inv(hess + 1e-8)[0, 0])
            if var > 0.0:
                se_val = math.sqrt(var)
            elif hess[0, 0] > 0.0:
                se_val = math.sqrt(1.0 / hess[0, 0])
        except (np.linalg.LinAlgError, ValueError):
            if hess[0, 0] > 0.0:
                se_val = math.sqrt(1.0 / hess[0, 0])
    return CopasFit(
        gamma0=float(gamma0), gamma1=float(gamma1),
        mu=float(p[0]), se_mu=se_val, rho=float(p[1]), tau=float(p[2]),
        beta=float(p[3]) if with_beta else None,
        loglik=float(res.fun), converged=bool(res.success),
    )


def _num_hessian(fun, x, step=1e-4):
    """Central-difference Hessian; the parameter scales here are all O(1)."""
    x = np.asarray(x, dtype=float)
    n = x.size
    h = np.maximum(np.abs(x) * step, step)
    H = np.zeros((n, n))

    def shift(idx, delta):
        point = x.copy()
        for k, d in zip(idx, delta, strict=True):
            point[k] += d
        return point

    for i in range(n):
        for j in range(n):
            if i == j:
                H[i, i] = (fun(shift([i], [h[i]])) - 2.0 * fun(x)
                           + fun(shift([i], [-h[i]]))) / (h[i] ** 2)
            elif j > i:
                val = (
                    fun(shift([i, j], [h[i], h[j]]))
                    - fun(shift([i, j], [h[i], -h[j]]))
                    - fun(shift([i, j], [-h[i], h[j]]))
                    + fun(shift([i, j], [-h[i], -h[j]]))
                ) / (4.0 * h[i] * h[j])
                H[i, j] = H[j, i] = val
    return H


# --------------------------------------------------------------------------- #
# contours
# --------------------------------------------------------------------------- #

def _pretty_step(span, n):
    """The step size R's ``pretty`` lands on for ``n`` intervals over ``span``."""
    if span <= 0.0:
        return 1.0
    raw = span / max(int(n), 1)
    unit = 10.0 ** math.floor(math.log10(raw))
    mult = min((1.0, 2.0, 5.0, 10.0), key=lambda m: abs(m * unit - raw))
    return mult * unit


def _auto_levels(z, nlevels):
    """Levels spanning the inside of ``z``'s range, as ``contourLines`` picks."""
    lo = float(np.nanmin(z))
    hi = float(np.nanmax(z))
    step = _pretty_step(hi - lo, nlevels)
    if step <= 0.0:
        return []
    start = math.ceil(lo / step) * step
    stop = math.floor(hi / step) * step
    out = []
    value = start
    while value <= stop + step * 1e-9:
        if lo < value < hi:
            out.append(float(value))
        value += step
    return out


def _contour_lines(x, y, z, level):
    """Connected pieces of one contour level, as lists of crossing points.

    A contour of a gridded surface is a set of polylines.  metasens fits a
    separate least-squares line to *each* polyline, so the number of pieces at
    a level matters: a level that closes into several loops must not have its
    crossings pooled into one regression.  The crossings themselves are fixed
    by linear interpolation along grid edges; only the grouping needs
    deciding, and grouping by "two crossings share a cell" is unambiguous --
    including at a saddle, where either connection rule puts the same four
    points in the same piece.
    """
    xs = np.asarray(x, dtype=float)
    ys = np.asarray(y, dtype=float)
    Z = np.asarray(z, dtype=float)
    nx, ny = Z.shape

    # every crossing, keyed by the grid edge it sits on
    h_ids: dict[tuple[int, int], int] = {}
    v_ids: dict[tuple[int, int], int] = {}
    px: list[float] = []
    py: list[float] = []

    def add(px_val, py_val):
        px.append(px_val)
        py.append(py_val)
        return len(px) - 1

    for i in range(nx - 1):
        for j in range(ny):
            a, b = Z[i, j], Z[i + 1, j]
            if (a - level) * (b - level) < 0.0:
                t = (level - a) / (b - a)
                h_ids[(i, j)] = add(xs[i] + t * (xs[i + 1] - xs[i]), ys[j])
    for i in range(nx):
        for j in range(ny - 1):
            a, b = Z[i, j], Z[i, j + 1]
            if (a - level) * (b - level) < 0.0:
                t = (level - a) / (b - a)
                v_ids[(i, j)] = add(xs[i], ys[j] + t * (ys[j + 1] - ys[j]))
    if not px:
        return []

    parent = list(range(len(px)))

    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    # crossings that share a cell belong to the same piece
    for i in range(nx - 1):
        for j in range(ny - 1):
            here = [
                h_ids.get((i, j)),        # bottom
                h_ids.get((i, j + 1)),    # top
                v_ids.get((i, j)),        # left
                v_ids.get((i + 1, j)),    # right
            ]
            here = [k for k in here if k is not None]
            for k in here[1:]:
                union(here[0], k)

    pieces: dict[int, list[int]] = {}
    for k in range(len(px)):
        pieces.setdefault(find(k), []).append(k)
    return [(np.array([px[k] for k in idx]), np.array([py[k] for k in idx]))
            for idx in pieces.values()]


# --------------------------------------------------------------------------- #
# the analysis
# --------------------------------------------------------------------------- #

@dataclass
class CopasResult:
    """A Copas selection-model analysis.

    Attributes:
        yi, seTE: the studies.
        tau_random, TE_random, seTE_random: the ordinary random-effects fit
            the analysis starts from.
        left: which side the missing studies were assumed to be on.
        rho_bound: the bound placed on the effect-selection correlation.
        gamma0_range, gamma1_range: the grid ranges.
        gamma0, gamma1: the grid itself.
        te_contour: ``(ngrid, ngrid)`` pooled effect at each grid point, with
            ``gamma0`` varying down the rows and ``gamma1`` across columns.
        loglik_contour: ``(ngrid, ngrid)`` maximised log-likelihood at each
            grid point — the surface :func:`copas_region` turns into a
            confidence region.
        levels: the contour levels that were traced.
        contour: per level, the number of crossing points, the fitted slope,
            its standard error, the intercept and the adjusted R-squared.
        slope: slope of the line orthogonal to the contours, in rescaled
            coordinates.
        gamma0_line, gamma1_line: the path walked, in natural units.
        publprob: publication probability of the largest study at each point.
        te_line, se_line, rho_line, tau_line: the fit along the path.
        pval_rsb: p-value of the residual-selection test (beta = 0).
        n_unpubl: number of studies implied to be unpublished.
        te_adjust, se_adjust, ci_lb, ci_ub, zval, pval: the reported answer.
        sign_rsb, level: settings.
    """

    yi: np.ndarray = field(repr=False)
    seTE: np.ndarray = field(repr=False)
    tau_random: float
    TE_random: float
    seTE_random: float
    left: bool
    rho_bound: float
    gamma0_range: tuple
    gamma1_range: tuple
    gamma0: np.ndarray = field(repr=False)
    gamma1: np.ndarray = field(repr=False)
    te_contour: np.ndarray = field(repr=False)
    loglik_contour: np.ndarray = field(repr=False)
    levels: list
    contour: dict = field(repr=False)
    slope: float
    gamma0_line: np.ndarray = field(repr=False)
    gamma1_line: np.ndarray = field(repr=False)
    publprob: np.ndarray = field(repr=False)
    te_line: np.ndarray = field(repr=False)
    se_line: np.ndarray = field(repr=False)
    rho_line: np.ndarray = field(repr=False)
    tau_line: np.ndarray = field(repr=False)
    pval_rsb: np.ndarray = field(repr=False)
    n_unpubl: np.ndarray = field(repr=False)
    te_adjust: float
    se_adjust: float
    ci_lb: float = field(repr=False)
    ci_ub: float = field(repr=False)
    zval: float = field(repr=False)
    pval: float = field(repr=False)
    sign_rsb: float = 0.1
    level: float = 95.0

    def coef_table(self) -> list[dict]:
        """The reported adjusted estimate as a one-row table."""
        return [{
            "estimate": self.te_adjust, "se": self.se_adjust,
            "zval": self.zval, "pval": self.pval,
            "ci_lb": self.ci_lb, "ci_ub": self.ci_ub,
            "tau": self.tau_random, "left": self.left,
        }]

    def sensitivity_table(self) -> list[dict]:
        """One row per point along the path."""
        rows = []
        for i in range(len(self.publprob)):
            rows.append({
                "gamma0": float(self.gamma0_line[i]),
                "gamma1": float(self.gamma1_line[i]),
                "publprob": float(self.publprob[i]),
                "estimate": float(self.te_line[i]),
                "se": float(self.se_line[i]),
                "tau": float(self.tau_line[i]),
                "rho": float(self.rho_line[i]),
                "pval_rsb": float(self.pval_rsb[i]),
                "n_unpubl": float(self.n_unpubl[i]),
            })
        return rows

    def to_dict(self) -> dict:
        return {
            "yi": self.yi.tolist(),
            "seTE": self.seTE.tolist(),
            "tau_random": self.tau_random,
            "TE_random": self.TE_random,
            "seTE_random": self.seTE_random,
            "left": self.left,
            "rho_bound": self.rho_bound,
            "gamma0_range": list(self.gamma0_range),
            "gamma1_range": list(self.gamma1_range),
            "gamma0": self.gamma0.tolist(),
            "gamma1": self.gamma1.tolist(),
            "te_contour": self.te_contour.tolist(),
            "loglik_contour": self.loglik_contour.tolist(),
            "levels": [float(x) for x in self.levels],
            "contour": {k: np.asarray(v).tolist()
                        for k, v in self.contour.items()},
            "slope": self.slope,
            "gamma0_line": self.gamma0_line.tolist(),
            "gamma1_line": self.gamma1_line.tolist(),
            "publprob": self.publprob.tolist(),
            "te_line": self.te_line.tolist(),
            "se_line": self.se_line.tolist(),
            "rho_line": self.rho_line.tolist(),
            "tau_line": self.tau_line.tolist(),
            "pval_rsb": self.pval_rsb.tolist(),
            "n_unpubl": self.n_unpubl.tolist(),
            "te_adjust": self.te_adjust,
            "se_adjust": self.se_adjust,
            "ci_lb": self.ci_lb,
            "ci_ub": self.ci_ub,
            "zval": self.zval,
            "pval": self.pval,
            "sign_rsb": self.sign_rsb,
            "level": self.level,
        }

    def __str__(self) -> str:
        rows = ["Copas selection model"]
        rows.append(f"  studies            {self.yi.size}")
        rows.append(f"  random effects     {self.TE_random:+.4f} "
                    f"(se {self.seTE_random:.4f}, tau {self.tau_random:.4f})")
        if np.isfinite(self.te_adjust):
            rows.append(f"  adjusted           {self.te_adjust:+.4f} "
                        f"[{self.ci_lb:+.4f}, {self.ci_ub:+.4f}]")
        else:
            rows.append("  adjusted           not identified "
                        "(residual selection significant everywhere)")
        rows.append(f"  missing studies on the "
                    f"{'left' if self.left else 'right'} of the funnel")
        return "\n".join(rows)


def _egger_intercept(yi, sei):
    """Intercept of the radial regression ``yi/sei ~ 1/sei``."""
    x = 1.0 / np.asarray(sei, dtype=float)
    y = np.asarray(yi, dtype=float) / np.asarray(sei, dtype=float)
    xc = x - x.mean()
    return float(y.mean() - (np.sum(xc * (y - y.mean())) / np.sum(xc * xc))
                 * x.mean())

def copas(yi, sei=None, *, tau=None, mu0=None, se0=None, ngrid=20,
          nlevels=10, levels=None, rho_bound=0.9999, left=None, slope=None,
          sign_rsb=0.1, level=95.0, gamma0_range=None, gamma1_range=None):
    """Copas selection-model analysis of a set of effect sizes.

    Args:
        yi: observed effects, or a fitted :class:`~pimeta.rma_uni.RmaResult`
            whose ``.yi`` / ``.vi`` / ``.tau2`` are used.
        sei: standard errors, when ``yi`` is an array.
        tau: between-study SD to start from; defaults to the random-effects
            estimate.
        mu0, se0: the random-effects estimate and SE.  When ``yi`` is a fitted
            model these come from the fit.
        ngrid: grid size for the sensitivity surface.
        nlevels: number of contour levels when ``levels`` is not given.
        levels: explicit contour levels of the pooled effect.
        rho_bound: bound on the effect-selection correlation.
        left: whether the unpublished studies are assumed to lie on the left
            of the funnel (negative effects).  Defaults to the sign of Egger's
            intercept, as metasens does.
        slope: slope of the line orthogonal to the contours, in rescaled
            coordinates.  Defaults to the value derived from the contours.
        sign_rsb: significance level of the residual-selection test used to
            decide where along the path to stop.
        level: confidence level in percent.
        gamma0_range, gamma1_range: grid ranges; defaults follow metasens.

    Returns:
        :class:`CopasResult`.

    Raises:
        ValueError: too few studies, mismatched lengths, or non-positive
            standard errors.
    """
    from .rma_uni import RmaResult

    is_fit = isinstance(yi, RmaResult)
    if is_fit:
        sei_arr = np.sqrt(np.asarray(yi.vi, dtype=float))
        yi_arr = np.asarray(yi.yi, dtype=float)
    else:
        if sei is None:
            raise ValueError("'sei' is required when 'yi' is an array")
        yi_arr = np.asarray(yi, dtype=float).ravel()
        sei_arr = np.asarray(sei, dtype=float).ravel()
    if yi_arr.size != sei_arr.size:
        raise ValueError("'yi' and 'sei' must be the same length")
    if yi_arr.size < 3:
        raise ValueError("at least three studies are needed")
    if np.any(sei_arr <= 0.0):
        raise ValueError("standard errors must be positive")
    if is_fit:
        tau_default = math.sqrt(max(float(yi.tau2), 0.0))
        mu_default = float(yi.b[0])
        se_default = float(yi.se[0])
    else:
        tau_default = float(np.std(yi_arr, ddof=1))
        mu_default = float(np.sum(yi_arr / sei_arr ** 2)
                           / np.sum(1.0 / sei_arr ** 2))
        se_default = float(1.0 / math.sqrt(np.sum(1.0 / sei_arr ** 2)))
    tau_use = float(tau) if tau is not None else float(tau_default)
    mu_start = float(mu0) if mu0 is not None else mu_default
    se_random = float(se0) if se0 is not None else se_default

    level = float(level)
    if level <= 1.0:
        level *= 100.0

    se_min = float(np.min(sei_arr))
    se_max = float(np.max(sei_arr))
    if gamma1_range is None:
        spread = 1.0 / se_min - 1.0 / se_max
        gamma1_range = (0.0, 1.29 / spread if spread > 0 else 1.0)
    if gamma0_range is None:
        gamma0_range = (-0.25 - gamma1_range[1] / se_max, 2.0)
    g0 = np.linspace(gamma0_range[0], gamma0_range[1], int(ngrid))
    g1 = np.linspace(gamma1_range[0], gamma1_range[1], int(ngrid))

    if left is None:
        left = _egger_intercept(yi_arr, sei_arr) > 0.0
    rho0 = rho_bound / 2.0 if left else -rho_bound / 2.0

    contour_grid = np.full((g0.size, g1.size), np.nan)
    loglik_grid = np.full((g0.size, g1.size), np.nan)
    for i in range(g0.size):
        for j in range(g1.size):
            fit = copas_fit(yi_arr, sei_arr, g0[i], g1[j],
                            rho_bound=rho_bound, mu0=mu_start, rho0=rho0,
                            tau0=tau_use, se_mu=False)
            contour_grid[i, j] = fit.mu
            loglik_grid[i, j] = -fit.loglik

    if levels is None:
        level_list = _auto_levels(contour_grid, nlevels)
    else:
        level_list = [float(x) for x in np.asarray(levels).ravel()]

    # contour lines are traced on the rescaled unit square, as metasens does
    g0_min, g0_max = float(g0.min()), float(g0.max())
    g1_min, g1_max = float(g1.min()), float(g1.max())
    span0 = g0_max - g0_min
    span1 = g1_max - g1_min
    g0_resc = (g0 - g0_min) / span0 if span0 > 0 else np.zeros_like(g0)
    g1_resc = (g1 - g1_min) / span1 if span1 > 0 else np.zeros_like(g1)

    stats_rows = {"levels": [], "nobs": [], "slopes": [], "se_slopes": [],
                  "intercepts": [], "adj_r_squared": []}
    for lv in level_list:
        for px, py in _contour_lines(g0_resc, g1_resc, contour_grid, lv):
            n = px.size
            if n < 3:
                continue
            xc = px - px.mean()
            sxx = float(np.sum(xc * xc))
            if sxx <= 0.0:
                continue
            sl = float(np.sum(xc * (py - py.mean())) / sxx)
            ic = float(py.mean() - sl * px.mean())
            resid = py - (ic + sl * px)
            sse = float(resid @ resid)
            sst = float(np.sum((py - py.mean()) ** 2))
            r2 = 1.0 - sse / sst if sst > 0 else 0.0
            adj = (1.0 - (1.0 - r2) * (n - 1) / (n - 2)) if n > 2 \
                else float("nan")
            se_sl = math.sqrt((sse / (n - 2)) / sxx) if n > 2 else float("nan")
            stats_rows["levels"].append(float(lv))
            stats_rows["nobs"].append(int(n))
            stats_rows["slopes"].append(sl)
            stats_rows["se_slopes"].append(se_sl)
            stats_rows["intercepts"].append(ic)
            stats_rows["adj_r_squared"].append(
                -100.0 if not np.isfinite(adj) else float(adj))

    if slope is None:
        a_r2 = np.asarray(stats_rows["adj_r_squared"], dtype=float)
        keep = a_r2 > 0.0
        if not np.any(keep):
            slope_value = float("nan")
        else:
            w = np.asarray(stats_rows["nobs"], dtype=float)[keep]
            s = np.asarray(stats_rows["slopes"], dtype=float)[keep]
            slope_value = -1.0 / float(np.sum(w * s) / np.sum(w))
    else:
        slope_value = float(slope)

    intercepts = np.asarray(stats_rows["intercepts"], dtype=float)
    slopes = np.asarray(stats_rows["slopes"], dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        x_slope = (1.0 - slope_value - intercepts) / (slopes - slope_value)
    y_slope = 1.0 + slope_value * (x_slope - 1.0)
    gamma0_line = x_slope * span0 + g0_min
    gamma1_line = y_slope * span1 + g1_min

    inside = (
        (gamma0_line >= min(gamma0_range)) & (gamma0_line <= max(gamma0_range))
        & (gamma1_line >= min(gamma1_range))
        & (gamma1_line <= max(gamma1_range))
        & np.isfinite(gamma0_line) & np.isfinite(gamma1_line)
    )
    x_slope = x_slope[inside]
    y_slope = y_slope[inside]
    gamma0_line = gamma0_line[inside]
    gamma1_line = gamma1_line[inside]
    order = np.argsort(-stats.norm.cdf(gamma0_line + gamma1_line / se_max),
                       kind="stable")
    x_slope = x_slope[order]
    y_slope = y_slope[order]
    gamma0_line = gamma0_line[order]
    gamma1_line = gamma1_line[order]
    # "everyone published" anchors the path at its generous end
    gamma0_line = np.concatenate([[10.0], gamma0_line])
    gamma1_line = np.concatenate([[0.0], gamma1_line])

    n_pts = gamma0_line.size
    rho_limits = (0.0, rho_bound) if left else (-rho_bound, 0.0)
    te_line = np.full(n_pts, np.nan)
    se_line = np.full(n_pts, np.nan)
    rho_line = np.full(n_pts, np.nan)
    tau_line = np.full(n_pts, np.nan)
    pval_rsb = np.full(n_pts, np.nan)
    for i in range(n_pts):
        f1 = copas_fit(yi_arr, sei_arr, gamma0_line[i], gamma1_line[i],
                       rho_bound=rho_bound, mu0=mu_start, rho0=rho0,
                       tau0=tau_use, with_beta=False)
        te_line[i] = f1.mu
        se_line[i] = f1.se_mu
        rho_line[i] = f1.rho
        tau_line[i] = f1.tau
        f2 = copas_fit(yi_arr, sei_arr, gamma0_line[i], gamma1_line[i],
                       rho_bound=rho_bound, mu0=mu_start, rho0=rho0,
                       tau0=tau_use, with_beta=True, se_mu=False,
                       rho_lim=rho_limits)
        if np.isfinite(f1.loglik) and np.isfinite(f2.loglik):
            stat = 2.0 * (f1.loglik - f2.loglik)
            pval_rsb[i] = float(stats.chi2.sf(max(stat, 0.0), 1))

    p_si = stats.norm.cdf(gamma0_line[:, None] + gamma1_line[:, None] / sei_arr)
    with np.errstate(divide="ignore", invalid="ignore"):
        n_unpubl = np.sum((1.0 - p_si) / p_si, axis=1)
    publprob = stats.norm.cdf(gamma0_line + gamma1_line / se_max)

    order = np.argsort(-publprob, kind="stable")
    publprob = publprob[order]
    te_line = te_line[order]
    se_line = se_line[order]
    rho_line = rho_line[order]
    tau_line = tau_line[order]
    pval_rsb = pval_rsb[order]
    n_unpubl = n_unpubl[order]
    gamma0_line = gamma0_line[order]
    gamma1_line = gamma1_line[order]

    significant = np.nan_to_num(pval_rsb, nan=1.0) <= sign_rsb
    cum = np.cumsum(significant)
    diff = np.arange(1, n_pts + 1) - cum
    if np.all(diff == 0):
        te_adj = float("nan")
        se_adj = float("nan")
    elif np.any(diff == 0):
        idx = int(np.argmax(diff > 0))
        te_adj = float(te_line[idx])
        se_adj = float(se_line[idx])
    else:
        te_adj = float(te_line[0])
        se_adj = float(se_line[0])
    crit = stats.norm.ppf(0.5 + level / 200.0)
    ci_lb = te_adj - crit * se_adj
    ci_ub = te_adj + crit * se_adj
    zval = te_adj / se_adj if se_adj and np.isfinite(se_adj) else float("nan")
    pval = float(2.0 * stats.norm.sf(abs(zval))) if np.isfinite(zval) \
        else float("nan")

    return CopasResult(
        yi=yi_arr, seTE=sei_arr, tau_random=tau_use, TE_random=mu_start,
        seTE_random=se_random, left=bool(left), rho_bound=float(rho_bound),
        gamma0_range=tuple(gamma0_range), gamma1_range=tuple(gamma1_range),
        gamma0=g0, gamma1=g1, te_contour=contour_grid,
        loglik_contour=loglik_grid,
        levels=[float(x) for x in stats_rows["levels"]],
        contour=stats_rows, slope=slope_value,
        gamma0_line=gamma0_line, gamma1_line=gamma1_line,
        publprob=publprob, te_line=te_line, se_line=se_line,
        rho_line=rho_line, tau_line=tau_line, pval_rsb=pval_rsb,
        n_unpubl=n_unpubl, te_adjust=te_adj, se_adjust=se_adj,
        ci_lb=float(ci_lb), ci_ub=float(ci_ub), zval=float(zval),
        pval=pval, sign_rsb=float(sign_rsb), level=level,
    )


# --------------------------------------------------------------------------- #
# the inferential object: a likelihood region
# --------------------------------------------------------------------------- #

@dataclass
class CopasRegion:
    """Which selection scenarios the data cannot rule out.

    ``metasens`` reduces the Copas analysis to one number by walking a path
    drawn by hand.  This is the same model with an actual inference attached:
    the set of ``(gamma0, gamma1)`` whose profile likelihood is within the
    usual chi-square-2 threshold of the best value.  Everything the model can
    say about the adjusted effect then follows from that set.

    Attributes:
        level: confidence level in percent.
        crit: the threshold ``2*(max loglik - loglik)`` must not exceed.
        gamma0_in, gamma1_in: the grid points inside the region.
        mu_in: the pooled effect at those points.
        publprob_in: publication probability of the largest study at those
            points — the interpretable coordinate.
        mu_best: the pooled effect at the single best point.
        mu_lb, mu_ub: **projection interval** for the adjusted effect: the
            range of the effect over the whole region.  Conservative by
            construction, in the same sense that a projection of a joint
            confidence set always is.
        publprob_lb, publprob_ub: the same for the publication probability.
        no_selection_inside: whether the region contains a scenario in which
            publication does not depend on precision (``gamma1 = 0``).  If it
            does, the data do not require any selection at all.
        gamma1_min: the smallest ``|gamma1|`` among admissible scenarios; zero
            means precision-dependent selection is not needed to explain the
            data.
        sign_stable: whether every admissible scenario has the effect on the
            same side of zero.  This is the question a reader actually asks,
            and it is answerable without any of the path machinery.
        k: number of studies.
    """

    level: float
    crit: float
    gamma0_in: np.ndarray = field(repr=False)
    gamma1_in: np.ndarray = field(repr=False)
    mu_in: np.ndarray = field(repr=False)
    publprob_in: np.ndarray = field(repr=False)
    mu_best: float
    mu_lb: float
    mu_ub: float
    publprob_lb: float
    publprob_ub: float
    no_selection_inside: bool
    gamma1_min: float
    sign_stable: bool
    k: int

    def to_dict(self) -> dict:
        return {
            "level": self.level,
            "crit": self.crit,
            "gamma0_in": self.gamma0_in.tolist(),
            "gamma1_in": self.gamma1_in.tolist(),
            "mu_in": self.mu_in.tolist(),
            "publprob_in": self.publprob_in.tolist(),
            "mu_best": self.mu_best,
            "mu_lb": self.mu_lb,
            "mu_ub": self.mu_ub,
            "publprob_lb": self.publprob_lb,
            "publprob_ub": self.publprob_ub,
            "no_selection_inside": self.no_selection_inside,
            "gamma1_min": self.gamma1_min,
            "sign_stable": self.sign_stable,
            "k": self.k,
        }

    def __str__(self) -> str:
        rows = [f"Copas selection region ({self.level:.0f}% likelihood)"]
        rows.append(f"  studies                 {self.k}")
        rows.append(f"  admissible scenarios    {self.mu_in.size} grid points")
        rows.append(f"  effect at the best fit  {self.mu_best:+.4f}")
        rows.append(f"  effect over the region  "
                    f"[{self.mu_lb:+.4f}, {self.mu_ub:+.4f}]")
        rows.append(f"  publication probability "
                    f"[{self.publprob_lb:.3f}, {self.publprob_ub:.3f}]")
        rows.append(f"  no selection admissible "
                    f"{'yes' if self.no_selection_inside else 'no'}")
        rows.append(f"  sign stable             "
                    f"{'yes' if self.sign_stable else 'no'}")
        return "\n".join(rows)


def copas_region(result, level=95.0):
    """Likelihood region for the Copas selection parameters.

    The model identifies only the combination ``gamma0 + gamma1/se``, so a
    point estimate of the adjusted effect is not the honest output.  This
    keeps the whole set of selection scenarios consistent with the data:

    .. math::

        2\\{\\ell_{\\max} - \\ell(\\gamma_0, \\gamma_1)\\}
        \\le \\chi^2_2(1-\\alpha)

    and reports the range of the adjusted effect over that set.  Projecting a
    joint confidence set onto one coordinate gives a conservative interval —
    wider than a profile that fixes the others — which is the right direction
    to err in when the "others" are unidentified.

    Args:
        result: a :class:`CopasResult` from :func:`copas`; the surface is
            reused, so nothing is refitted.
        level: confidence level in percent (``95``) or as a proportion
            (``0.95``).

    Returns:
        :class:`CopasRegion`.

    Raises:
        ValueError: the level is not in ``(0, 100)``, or the surface has no
            finite point to anchor the maximum.
    """
    level = float(level)
    if level <= 1.0:
        level *= 100.0
    if not 0.0 < level < 100.0:
        raise ValueError("'level' must be between 0 and 100")

    ll = np.asarray(result.loglik_contour, dtype=float)
    mu = np.asarray(result.te_contour, dtype=float)
    finite = np.isfinite(ll) & np.isfinite(mu)
    if not np.any(finite):
        raise ValueError("the likelihood surface has no finite point")

    ll_max = float(np.max(ll[finite]))
    alpha = 1.0 - level / 100.0
    crit = float(stats.chi2.ppf(1.0 - alpha, 2))
    inside = finite & (2.0 * (ll_max - ll) <= crit)

    g0 = np.broadcast_to(np.asarray(result.gamma0)[:, None], ll.shape)
    g1 = np.broadcast_to(np.asarray(result.gamma1)[None, :], ll.shape)
    # the largest study's publication probability identifies the direction of
    # selection that matters, while gamma0 alone does not
    se_max = float(np.max(result.seTE))
    publprob = stats.norm.cdf(g0 + g1 / se_max)

    mu_in = mu[inside]
    pp_in = publprob[inside]
    best = np.unravel_index(int(np.argmax(np.where(finite, ll, -np.inf))),
                            ll.shape)

    # "no selection" means publication does not depend on precision: the
    # likelihood cannot separate gamma0 and gamma1 beyond the combination
    # gamma0 + gamma1/se, but gamma1 = 0 is a definite statement about
    # precision, so membership is checked on gamma1 alone.
    gamma1_floor = float(np.min(np.abs(result.gamma1)))
    no_sel = bool(np.any(inside & (np.abs(g1) <= gamma1_floor + 1e-12)))

    return CopasRegion(
        level=level, crit=crit,
        gamma0_in=g0[inside], gamma1_in=g1[inside],
        mu_in=mu_in, publprob_in=pp_in,
        mu_best=float(mu[best]),
        mu_lb=float(np.min(mu_in)), mu_ub=float(np.max(mu_in)),
        publprob_lb=float(np.min(pp_in)),
        publprob_ub=float(np.max(pp_in)),
        no_selection_inside=no_sel,
        gamma1_min=float(np.min(np.abs(g1[inside]))),
        sign_stable=bool(np.all(mu_in > 0.0) or np.all(mu_in < 0.0)),
        k=int(result.seTE.size),
    )
