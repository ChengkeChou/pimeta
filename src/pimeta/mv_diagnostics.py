"""metafor's ``rma.mv`` diagnostic methods.

metafor exports ``hatvalues.rma.mv``, ``weights.rma.mv``,
``rstandard.rma.mv``, ``cooks.distance.rma.mv``, ``dfbetas.rma.mv``,
``rstudent.rma.mv`` and ``qqnorm.rma.mv``; only the last three need refits, the
first four are closed form in the marginal var-covariance matrix ``M`` and the
design matrix ``X``:

``H = X (X'WX)^-1 X'W``,  ``resid = yi - X beta``,
``ve = (I-H) M (I-H)'``,  ``rstandard = resid / sqrt(diag(ve))``.

The refit-based ones drop each study (or cluster), refit ``rma_mv`` and compare
the fixed effects, exactly as metafor's ``.cooks.distance.rma.mv`` /
``.dfbetas.rma.mv`` / ``.rstudent.rma.mv`` helpers do.
"""

from __future__ import annotations

import numpy as np
from scipy import stats

__all__ = [
    "mv_hatvalues", "mv_weights", "mv_rstandard", "mv_cooks_distance",
    "mv_dfbetas", "mv_rstudent", "profile_rmamv",
]


def _check(x):
    from .rma_mv import RmaMvResult

    if not isinstance(x, RmaMvResult):
        raise TypeError(f"not an rma.mv fit: {type(x).__name__}")
    if x.M.size == 0:
        raise ValueError(
            "Information needed to compute the diagnostics is not available "
            "in the model object.")
    return x


def _W(x):
    """metafor uses ``W = M^-1`` unless a weight matrix was supplied."""
    return np.linalg.inv(np.asarray(x.M, dtype=float))


def _H(x):
    """Hat matrix ``X (X'WX)^-1 X'W``."""
    X = np.asarray(x.X, dtype=float)
    W = _W(x)
    xtwx = X.T @ W @ X
    return X @ np.linalg.pinv(xtwx) @ X.T @ W


def mv_hatvalues(x, type="diagonal"):
    """``hatvalues.rma.mv`` — leverages (diagonal) or the hat matrix."""
    x = _check(x)
    h = _H(x)
    kind = str(type).lower()
    if kind.startswith("diag"):
        d = np.diag(h).copy()
        d[d > 1.0 - 10 * 2.220446049250313e-16] = 1.0
        return [float(v) for v in d]
    if kind == "matrix":
        return h
    raise ValueError("'type' must be 'diagonal' or 'matrix'")


def mv_weights(x, type="diagonal"):
    """``weights.rma.mv`` — ``diag(M^-1)``, the full matrix, or row sums."""
    x = _check(x)
    W = _W(x)
    kind = str(type).lower()
    if kind.startswith("diag"):
        w = np.diag(W)
        return [float(v) for v in w / np.sum(w) * 100.0]
    if kind == "matrix":
        return W
    if kind == "rowsum":
        if not x.int_only:
            raise ValueError(
                "Row-sum weights are only meaningful for intercept-only "
                "models.")
        w = np.sum(W, axis=1)
        return [float(v) for v in w / np.sum(w) * 100.0]
    raise ValueError("'type' must be 'diagonal', 'matrix' or 'rowsum'")


def mv_rstandard(x, cluster=None):
    """``rstandard.rma.mv`` — standardised residuals.

    ``resid = yi - X beta``, ``ve = (I-H) M (I-H)'``, ``z = resid/sqrt(diag(ve))``.
    With ``cluster`` the per-cluster quadratic form ``X2`` is returned as well,
    as a second dict under the key ``"cluster"``.
    """
    x = _check(x)
    X = np.asarray(x.X, dtype=float)
    M = np.asarray(x.M, dtype=float)
    yi = np.asarray(x.yi, dtype=float).ravel()
    beta = np.asarray(x.beta, dtype=float).ravel()
    h = _H(x)
    imh = np.eye(h.shape[0]) - h
    resid = yi - X @ beta
    resid[np.abs(resid) < 100 * 2.220446049250313e-16] = 0.0
    ve = imh @ M @ imh.T
    se = np.sqrt(np.maximum(np.diag(ve), 0.0))
    with np.errstate(invalid="ignore", divide="ignore"):
        z = resid / se
    out = {"resid": [float(v) for v in resid],
           "se": [float(v) for v in se],
           "z": [float(v) for v in z],
           "slab": list(x.slab)}
    if cluster is None:
        return out
    cl = list(cluster)
    k = resid.size
    if len(cl) != k:
        raise ValueError(
            f"Length of the variable specified via 'cluster' ({len(cl)}) does "
            f"not match the length of the data ({k}).")
    ids = list(dict.fromkeys(cl))
    x2 = np.full(len(ids), np.nan)
    for i, g in enumerate(ids):
        incl = np.array([c == g for c in cl])
        try:
            sve = np.linalg.inv(ve[np.ix_(incl, incl)])
        except np.linalg.LinAlgError:
            continue
        e = resid[incl]
        x2[i] = float(e @ sve @ e)
    out["cluster"] = {
        "X2": [float(v) for v in x2],
        "k": [int(np.sum([c == g for c in cl])) for g in ids],
        "slab": [str(v) for v in ids],
    }
    return out


def _mv_refit(x, keep, sigma2=None):
    """Refit ``rma_mv`` on the rows selected by ``keep``."""
    from .rma_mv import rma_mv

    V = np.asarray(x.V, dtype=float)[np.ix_(keep, keep)]
    mods = None
    if x.mods is not None:
        mods = np.asarray(x.mods, dtype=float)
        mods = mods[keep] if mods.ndim == 1 else mods[keep, :]
    data = getattr(x, "_data", None)
    if data is not None:
        data = {k_: (np.asarray(v_)[keep].tolist() if hasattr(v_, "__len__")
                     and len(v_) == x.k else v_)
                for k_, v_ in data.items()}
    return rma_mv(
        np.asarray(x.yi, dtype=float)[keep], V=V, mods=mods, data=data,
        random=x.random, struct=x.struct, method=x.method, level=x.level,
        test=x.test, dfs=(x.dfs if x.dfs is not None else "residual"),
        intercept=x.intercept_incl, sigma2=sigma2,
    )


def _groups(x, cluster):
    k = int(x.k)
    if cluster is None:
        return [[i] for i in range(k)]
    cl = list(cluster)
    if len(cl) != k:
        raise ValueError(
            f"Length of the variable specified via 'cluster' ({len(cl)}) does "
            f"not match the length of the data ({k}).")
    order = list(dict.fromkeys(cl))
    return [[i for i in range(k) if cl[i] == g] for g in order]


def mv_cooks_distance(x, cluster=None, reestimate=True, btt=None):
    """``cooks.distance.rma.mv`` — Cook's distance per study/cluster.

    ``dfb' vb^-1 dfb`` with ``vb`` the full-model coefficient covariance and
    ``dfb = beta - beta(-i)``.  Like ``metafor`` the refits re-estimate the
    variance components by default (``reestimate=True``); pass
    ``reestimate=False`` to keep them fixed at the full-model values.
    """
    x = _check(x)
    beta = np.asarray(x.beta, dtype=float).ravel()
    p = beta.size
    idx = list(range(p)) if btt is None else [int(b) - 1 for b in np.atleast_1d(btt)]
    sub = np.asarray(x.vb, dtype=float)[np.ix_(idx, idx)]
    try:
        svb = np.linalg.inv(sub)
    except np.linalg.LinAlgError:
        svb = np.linalg.pinv(sub)
    out = []
    for grp in _groups(x, cluster):
        keep = np.ones(int(x.k), dtype=bool)
        keep[grp] = False
        try:
            res = _mv_refit(x, keep, sigma2=None if reestimate else x.sigma2)
        except (ValueError, np.linalg.LinAlgError):
            out.append(float("nan"))
            continue
        dfb = beta[idx] - np.asarray(res.beta, dtype=float).ravel()[idx]
        out.append(float(dfb @ svb @ dfb))
    return out


def mv_dfbetas(x, cluster=None):
    """``dfbetas.rma.mv`` — ``(beta - beta(-i)) / sqrt(diag(vb(-i)))``.

    The refits re-estimate the variance components and use the leave-one-out
    standard errors, which is what metafor's ``.dfbetas.rma.mv`` helper does.
    """
    x = _check(x)
    beta = np.asarray(x.beta, dtype=float).ravel()
    rows = []
    for grp in _groups(x, cluster):
        keep = np.ones(int(x.k), dtype=bool)
        keep[grp] = False
        try:
            res = _mv_refit(x, keep)
            # vb at the leave-one-out variance components but on the full data
            tmp = _mv_refit(x, np.ones(int(x.k), dtype=bool),
                            sigma2=res.sigma2)
        except (ValueError, np.linalg.LinAlgError):
            rows.append([float("nan")] * beta.size)
            continue
        dfb = beta - np.asarray(res.beta, dtype=float).ravel()
        sd = np.sqrt(np.diag(np.asarray(tmp.vb, dtype=float)))
        with np.errstate(invalid="ignore", divide="ignore"):
            rows.append([float(v) for v in dfb / sd])
    return rows


def mv_rstudent(x, cluster=None):
    """``rstudent.rma.mv`` — deleted residuals for each study/cluster.

    For every dropped group the model is refitted, and the deleted residual is
    divided by ``sqrt(M_ii + x_i' vb(-i) x_i)``.  Returns a dict with
    ``delresid``, ``sedelresid``, ``z``, ``X2`` and ``slab``.
    """
    x = _check(x)
    yi = np.asarray(x.yi, dtype=float).ravel()
    X = np.asarray(x.X, dtype=float)
    M = np.asarray(x.M, dtype=float)
    delresid = np.full(int(x.k), np.nan)
    sedel = np.full(int(x.k), np.nan)
    groups = _groups(x, cluster)
    labels = ([f"-{x.slab[i]}" for i in range(int(x.k))] if cluster is None
              else [f"-{g}" for g in list(dict.fromkeys(list(cluster)))])
    x2 = []
    for grp, lab in zip(groups, labels):
        keep = np.ones(int(x.k), dtype=bool)
        keep[grp] = False
        try:
            # metafor's .rstudent.rma.mv re-estimates the variance components
            # for the deleted fit, then reads M at *those* values.
            res = _mv_refit(x, keep)
            tmp = _mv_refit(x, np.ones(int(x.k), dtype=bool),
                            sigma2=res.sigma2)
        except (ValueError, np.linalg.LinAlgError):
            x2.append(float("nan"))
            continue
        Xi = X[grp, :]
        beta = np.asarray(res.beta, dtype=float).ravel()
        depth = np.asarray(res.vb, dtype=float)
        vdel = Xi @ depth @ Xi.T
        delresid[grp] = yi[grp] - Xi @ beta
        block = np.asarray(tmp.M, dtype=float)[np.ix_(grp, grp)] + vdel
        sedel[grp] = np.sqrt(np.maximum(np.diag(block), 0.0))
        try:
            sve = np.linalg.inv(block)
        except np.linalg.LinAlgError:
            x2.append(float("nan"))
            continue
        e = delresid[grp]
        x2.append(float(e @ sve @ e))
    with np.errstate(invalid="ignore", divide="ignore"):
        z = delresid / sedel
    return {"delresid": [float(v) for v in delresid],
            "sedelresid": [float(v) for v in sedel],
            "z": [float(v) for v in z],
            "X2": [float(v) for v in x2],
            "slab": labels}



def profile_rmamv(x, sigma2=None, steps=20, xlim=None, plot=False, **kw):
    """``profile.rma.mv`` — profile the log-likelihood over one variance
    component.

    For every grid value ``v`` of the chosen component the model is refitted
    with **that** component pinned to ``v`` while every other component is
    re-estimated, which is exactly what metafor's ``.profile.rma.mv`` does
    (``rma.mv(..., sigma2 = c(NA, ..., v, ..., NA))``).  The returned object
    holds the maximised log-likelihood, the fixed effects and their confidence
    intervals at each grid point.

    Args:
        x: an :class:`~pimeta.RmaMvResult`.
        sigma2: 1-based index of the variance component to profile (metafor's
            ``sigma2=`` argument).  Defaults to the first component.
        steps: number of grid points, or an explicit sequence of values
            (metafor's ``steps`` argument — a vector of length ``>= 2`` is
            used as the grid itself).
        xlim: profiling range; by default ``[max(0, vc/4), max(0.1, vc*4)]``
            around the fitted value ``vc``, like metafor.
        plot: draw the profile plot immediately.
        **kw: forwarded to :func:`pimeta.plotting.plot_mv_profile`.

    Returns:
        :class:`MvProfileResult`.
    """
    x = _check(x)
    comps = list(getattr(x, "sigma2", None) or [])
    if not comps:
        raise ValueError("Model does not contain any (estimated) 'sigma2' "
                         "components.")

    # `sigma2` first positional argument doubles as metafor's component index.
    if sigma2 is None:
        pos = 0
    else:
        pos = int(np.atleast_1d(sigma2)[0]) - 1
    if not 0 <= pos < len(comps):
        raise ValueError(f"Component {pos + 1} does not exist; the model has "
                         f"{len(comps)} variance component(s).")
    vc = float(comps[pos])

    if xlim is None:
        lo, hi = max(0.0, vc / 4.0), max(0.1, vc * 4.0)
    else:
        lim = np.sort(np.asarray(xlim, dtype=float).ravel())
        if lim.size != 2:
            raise ValueError("Argument 'xlim' should be a vector of length 2.")
        if lim[0] < 0:
            raise ValueError("Lower bound for profiling must be >= 0.")
        lo, hi = float(lim[0]), float(lim[1])

    step_arr = np.atleast_1d(np.asarray(steps, dtype=float)).ravel()
    if step_arr.size >= 2:
        grid = step_arr
    else:
        n = int(step_arr[0]) if step_arr.size else 20
        if n < 2:
            raise ValueError("Argument 'steps' must be >= 2.")
        grid = np.linspace(lo, hi, n)

    # Everything except the profiled component stays free (NaN in R).
    free = np.full(len(comps), np.nan, dtype=float)
    ll = []
    beta = []
    ci_lb = []
    ci_ub = []
    for val in grid:
        pinned = free.copy()
        pinned[pos] = float(val)
        try:
            fit = _mv_refit(x, np.ones(int(x.k), dtype=bool), sigma2=pinned)
        except Exception:                       # pragma: no cover - bad fit
            ll.append(np.nan)
            beta.append([np.nan] * int(x.p))
            ci_lb.append([np.nan] * int(x.p))
            ci_ub.append([np.nan] * int(x.p))
            continue
        ll.append(float(fit.ll))
        beta.append([float(b) for b in np.asarray(fit.beta).ravel()])
        ci_lb.append([float(b) for b in np.asarray(fit.ci_lb).ravel()])
        ci_ub.append([float(b) for b in np.asarray(fit.ci_ub).ravel()])

    out = MvProfileResult(
        component="sigma2",
        index=pos + 1,
        sigma2=[float(v) for v in grid],
        ll=ll,
        maxll=float(x.ll),
        beta=beta,
        ci_lb=ci_lb,
        ci_ub=ci_ub,
        vc=vc,
        method=str(x.method),
        xlab=(r"$\sigma^2$ Value" if len(comps) == 1
              else rf"$\sigma_{{{pos + 1}}}^2$ Value"),
        title=(r"Profile Plot for $\sigma^2$" if len(comps) == 1
               else rf"Profile Plot for $\sigma_{{{pos + 1}}}^2$"),
    )
    if plot:                                    # pragma: no cover - plotting
        from .plotting import plot_mv_profile

        plot_mv_profile(out, **kw)
    return out


class MvProfileResult:
    """Result of :func:`profile_rmamv` (mirrors ``profile.rma`` objects).

    ``sigma2`` holds the profiling grid, ``ll`` the maximised log-likelihood
    at each grid point and ``beta`` / ``ci_lb`` / ``ci_ub`` the fixed effects
    (one row per grid point).  ``maxll`` is the log-likelihood of the
    unconstrained fit.
    """

    __slots__ = ("component", "index", "sigma2", "ll", "maxll", "beta",
                 "ci_lb", "ci_ub", "vc", "method", "xlab", "title", "comps")

    def __init__(self, component, index, sigma2, ll, maxll, beta, ci_lb,
                 ci_ub, vc, method, xlab, title, comps=1):
        self.component = component
        self.index = index
        self.sigma2 = sigma2
        self.ll = ll
        self.maxll = maxll
        self.beta = beta
        self.ci_lb = ci_lb
        self.ci_ub = ci_ub
        self.vc = vc
        self.method = method
        self.xlab = xlab
        self.title = title
        self.comps = comps

    @property
    def tau2(self):
        """Alias for :attr:`sigma2` (metafor uses ``tau2`` for ``rma.uni``)."""
        return self.sigma2

    def __repr__(self):
        return (f"MvProfileResult(component='sigma2', index={self.index}, "
                f"points={len(self.sigma2)}, vc={self.vc:.6g}, "
                f"maxll={self.maxll:.6g})")

    def to_dict(self) -> dict:
        return {
            "component": self.component, "index": self.index,
            "sigma2": self.sigma2, "ll": self.ll, "maxll": self.maxll,
            "beta": self.beta, "ci_lb": self.ci_lb, "ci_ub": self.ci_ub,
            "vc": self.vc, "method": self.method,
        }
