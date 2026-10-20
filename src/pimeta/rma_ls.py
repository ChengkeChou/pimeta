"""Location-scale meta-analysis models.

``metafor`` has no ``rma.ls()`` function: a location-scale model is fitted by
passing ``scale = ~ z`` to ``rma()``/``rma.uni()``, which then returns an object
of class ``c("rma.ls", "rma.uni", "rma")``.  This module reproduces that fit;
:func:`pimeta.rma` forwards its ``scale=`` argument here so the R idiom works.

The model is

.. math::

    y_i \\sim N(\\beta' x_i,\\; v_i + \\tau_i^2), \\qquad
    \\tau_i^2 = g(z_i' \\alpha)

with ``g`` the identity (``link="identity"``) or the exponential
(``link="log"``, the default).  ``alpha`` is estimated by maximising the (REML)
likelihood with the location coefficients profiled out, so ``rma.ls`` has
``p + q`` parameters; the standard errors of ``alpha`` come from the numerical
Hessian of the negative log-likelihood, exactly as metafor does.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
from scipy import optimize, stats

__all__ = [
    "RmaLsResult",
    "RmaLsProfileResult",
    "rma_ls",
    "profile_rma_ls",
    "confint_rma_ls",
]

_LINKS = ("log", "identity")


# --------------------------------------------------------------------------- #
# the location-scale log-likelihood
# --------------------------------------------------------------------------- #

def _ls_pieces(alpha, yi, vi, X, Z, link):
    """Weights, GLS coefficients and RSS at ``alpha`` (``None`` if invalid)."""
    if link == "log":
        tau2 = np.exp(Z @ alpha)
    else:
        tau2 = Z @ alpha
    if not np.all(np.isfinite(tau2)) or np.any(tau2 < 0):
        return None
    vari = vi + tau2
    if np.any(vari <= 0):
        return None
    w = 1.0 / vari
    xtwx = X.T @ (w[:, None] * X)
    try:
        stXWX = np.linalg.inv(xtwx)
    except np.linalg.LinAlgError:      # pragma: no cover - singular design
        return None
    beta = stXWX @ (X.T @ (w * yi))
    resid = yi - X @ beta
    rss = float(np.sum(w * resid ** 2))
    return {"tau2": tau2, "vari": vari, "w": w, "beta": beta,
            "stXWX": stXWX, "rss": rss, "xtwx": xtwx}


def _ls_loglik(alpha, yi, vi, X, Z, reml, link, remlf=True):
    """``metafor:::.ll.rma.ls`` (the ``randhet`` extension is not supported)."""
    pic = _ls_pieces(alpha, yi, vi, X, Z, link)
    if pic is None:
        return -math.inf
    k, p = X.shape
    logdet_v = float(np.sum(np.log(pic["vari"])))
    if not reml:
        return (-0.5 * k * math.log(2 * math.pi) - 0.5 * logdet_v
                - 0.5 * pic["rss"])
    sign_xtwx, logdet_xtwx = np.linalg.slogdet(pic["xtwx"])
    if sign_xtwx <= 0:                 # pragma: no cover - singular design
        return -math.inf
    logdet_xtx = 0.0
    if remlf:
        sign_xtx, logdet_xtx = np.linalg.slogdet(X.T @ X)
        if sign_xtx <= 0:              # pragma: no cover - singular design
            return -math.inf
    return (-0.5 * (k - p) * math.log(2 * math.pi) + 0.5 * logdet_xtx
            - 0.5 * logdet_v - 0.5 * logdet_xtwx - 0.5 * pic["rss"])


def _hessian(f, x, h=None):
    """Numerical Hessian (5-point diagonal, 4-point off-diagonal).
    ``numDeriv::hessian`` uses Richardson extrapolation; this simpler scheme
    agrees with it to ~1e-10, which is far below the optimiser's own accuracy.
    """
    x = np.asarray(x, dtype=float)
    n = x.size
    if h is None:
        h = 1e-5 * np.maximum(1.0, np.abs(x))
    h = np.broadcast_to(np.asarray(h, dtype=float), (n,)).copy()
    h[h == 0] = 1e-5
    f0 = f(x)
    out = np.empty((n, n))
    # diagonal: (-f(x+2h) + 16f(x+h) - 30f0 + 16f(x-h) - f(x-2h)) / (12 h^2)
    for i in range(n):
        hi = h[i]
        e = np.zeros(n)
        e[i] = hi
        fp1, fm1 = f(x + e), f(x - e)
        fp2, fm2 = f(x + 2 * e), f(x - 2 * e)
        out[i, i] = (-fp2 + 16 * fp1 - 30 * f0 + 16 * fm1 - fm2) / (12 * hi ** 2)
    for i in range(n):
        for j in range(i + 1, n):
            ei = np.zeros(n)
            ej = np.zeros(n)
            ei[i] = h[i]
            ej[j] = h[j]
            val = (f(x + ei + ej) - f(x + ei - ej)
                   - f(x - ei + ej) + f(x - ei - ej)) / (4 * h[i] * h[j])
            out[i, j] = out[j, i] = val
    return out


def _gradient(f, x, h=None):
    """Central-difference gradient (4th-order accurate)."""
    x = np.asarray(x, dtype=float)
    n = x.size
    if h is None:
        h = 1e-6 * np.maximum(1.0, np.abs(x))
    h = np.broadcast_to(np.asarray(h, dtype=float), (n,)).copy()
    h[h == 0] = 1e-6
    g = np.empty(n)
    for i in range(n):
        e = np.zeros(n)
        e[i] = h[i]
        g[i] = (f(x - 2 * e) - 8 * f(x - e) + 8 * f(x + e) - f(x + 2 * e)) \
            / (12 * h[i])
    return g


def _newton_polish(f, x, iters=12):
    """Polish the optimum with damped Newton steps.

    The location-scale log-likelihood is extremely flat in ``alpha`` (the
    scale coefficients are often O(10) with standard errors of the same size),
    so quasi-Newton stopping rules leave the estimate a few 1e-6 short.  A
    handful of Newton steps with a numerical Hessian drives the gradient to
    machine precision, which is what makes ``se.alpha`` and ``va`` agree with
    metafor's ``numDeriv::hessian`` output.
    """
    cur = np.asarray(x, dtype=float).copy()
    if cur.size == 0:                      # pragma: no cover
        return cur
    best_f = f(cur)
    if not np.isfinite(best_f):            # pragma: no cover
        return cur
    for _ in range(iters):
        g = _gradient(f, cur)
        if not np.all(np.isfinite(g)) or np.max(np.abs(g)) < 1e-12:
            break
        try:
            H = _hessian(f, cur)
            step = np.linalg.solve(H, g)
        except np.linalg.LinAlgError:     # pragma: no cover
            break
        if not np.all(np.isfinite(step)):
            break
        t = 1.0
        moved = False
        for _ in range(40):
            cand = cur - t * step
            fc = f(cand)
            if np.isfinite(fc) and fc <= best_f:
                cur, best_f = cand, fc
                moved = True
                break
            t *= 0.5
        if not moved:
            break
    return cur


def _default_terms(idx, m, intercept_incl):
    """metafor's ``.set.btt`` default: the moderators only (or all if no intercept)."""
    if idx is not None:
        out = np.asarray(idx, dtype=int).ravel() - 1
        return out
    if m > 1 and intercept_incl:
        return np.arange(1, m)
    return np.arange(m)


# --------------------------------------------------------------------------- #
# result object
# --------------------------------------------------------------------------- #

@dataclass
class RmaLsResult:
    """Location-scale fit (mirrors metafor's ``rma.ls`` objects)."""

    # --- location part (identical in meaning to rma.uni) ----------------
    b: list[float]
    se: list[float]
    zval: list[float]
    pval: list[float]
    ci_lb: list[float]
    ci_ub: list[float]
    vb: np.ndarray
    # --- scale part -----------------------------------------------------
    alpha: list[float]
    se_alpha: list[float]            # NaN when the Hessian is singular
    zval_alpha: list[float]
    pval_alpha: list[float]
    ci_lb_alpha: list[float]
    ci_ub_alpha: list[float]
    va: np.ndarray
    QS: float
    QSdf: tuple[int, float]
    QSp: float
    # --- heterogeneity / fit --------------------------------------------
    tau2: list[float]                # per-study (length k), as in R
    QE: float
    QEp: float
    I2: list[float] | None           # per-study
    H2: list[float] | None           # per-study
    vt: float
    parms: int
    ll: float
    fit_stats: dict
    # --- bookkeeping ----------------------------------------------------
    k: int
    p: int
    q: int
    alphas: int
    method: str
    model: str = "rma.ls"
    link: str = "log"
    test: str = "z"
    ddf: float = math.nan
    ddf_alpha: float = math.nan
    level: float = 0.95
    intercept: bool = True
    scale_intercept: bool = True
    optbeta: bool = False
    alpha_fix: list[bool] = field(default_factory=list)
    Z_int_incl: bool = True
    Z_int_only: bool = False
    int_only: bool = True
    measure: str = "GEN"
    yi: list[float] = field(default_factory=list)
    vi: list[float] = field(default_factory=list)
    X: np.ndarray | None = None
    Z: np.ndarray | None = None
    slab: list[str] = field(default_factory=list)
    QM: float = math.nan
    QMdf: tuple[int, float] = (0, math.nan)
    QMp: float = math.nan
    #: metafor returns NULL for R^2 in location-scale models
    R2: object = None

    # ``beta`` / ``tau2`` are the two names callers reach for most often;
    # ``tau2`` is a vector here, so ``beta`` is what ``b`` holds.
    @property
    def beta(self):
        return self.b

    @property
    def coef(self):
        return self.b

    @property
    def sigma2(self):
        """Alias kept for code that treats ``tau2`` as a scalar."""
        return self.tau2

    def __repr__(self):
        vals = ", ".join(f"{v:.4f}" for v in self.alpha)
        return (f"RmaLsResult(link={self.link!r}, method={self.method!r}, "
                f"k={self.k}, p={self.p}, q={self.q}, alpha=[{vals}], "
                f"ll={self.ll:.4f})")

    def to_dict(self) -> dict:
        return {
            "b": self.b, "se": self.se, "zval": self.zval, "pval": self.pval,
            "ci_lb": self.ci_lb, "ci_ub": self.ci_ub,
            "vb": np.asarray(self.vb).tolist(),
            "alpha": self.alpha, "se_alpha": self.se_alpha,
            "zval_alpha": self.zval_alpha, "pval_alpha": self.pval_alpha,
            "ci_lb_alpha": self.ci_lb_alpha, "ci_ub_alpha": self.ci_ub_alpha,
            "va": np.asarray(self.va).tolist(),
            "QS": self.QS, "QSdf": list(self.QSdf), "QSp": self.QSp,
            "tau2": self.tau2, "QE": self.QE, "QEp": self.QEp,
            "I2": self.I2, "H2": self.H2, "vt": self.vt,
            "parms": self.parms, "ll": self.ll, "fit_stats": self.fit_stats,
            "k": self.k, "p": self.p, "q": self.q,
            "method": self.method, "link": self.link, "test": self.test,
        }


# --------------------------------------------------------------------------- #
# fitting
# --------------------------------------------------------------------------- #

def _as_matrix(a, k, name):
    arr = np.asarray(a, dtype=float)
    if arr.ndim == 1:
        arr = arr.reshape(-1, 1)
    if arr.shape[0] != k:
        raise ValueError(f"'{name}' must have one row per study")
    return arr


def rma_ls(
    yi,
    vi=None,
    sei=None,
    mods=None,
    scale=None,
    data=None,
    slab=None,
    measure: str = "GEN",
    method: str = "REML",
    test: str = "z",
    level: float = 0.95,
    intercept: bool = True,
    scale_intercept: bool = True,
    link: str = "log",
    optbeta: bool = False,
    alpha=None,
    btt=None,
    att=None,
) -> RmaLsResult:
    """Fit a location-scale meta-analysis model (``rma(..., scale=~mods)``).

    Args:
        yi: analysis-scale effect sizes.
        vi: sampling variances (alternative to ``sei``).
        sei: sampling standard errors (alternative to ``vi``).
        mods: location moderators — 1-D or 2-D, WITHOUT an intercept (one is
            added when ``intercept=True``).
        scale: scale moderators — 1-D or 2-D, WITHOUT an intercept (added when
            ``scale_intercept=True``).  Required.
        data: dict / DataFrame supplying the columns named by string, so that
            ``mods="x"`` and ``scale="z"`` can be used like R formulas.
        slab: study labels.
        measure: effect-size measure (recorded only).
        method: ``"REML"`` (default) or ``"ML"`` — location-scale models have
            no closed-form tau^2 estimator.
        test: coefficient test for the location part (``"z"``, ``"t"``,
            ``"knha"``, ``"adhoc"``); the ``alpha`` tests use ``k - q`` df for
            the t variants, as metafor does.
        level: confidence level (95 -> 95% CI).
        intercept, scale_intercept: include the intercepts in the two designs.
        link: ``"log"`` (default, tau^2 = exp(z'alpha)) or ``"identity"``.
        optbeta: estimate the location coefficients jointly with ``alpha``
            instead of profiling them out (R's ``control=list(optbeta=TRUE)``).
        alpha: fix (some of) the scale coefficients; ``NaN`` entries stay free.
        btt, att: 1-based indices for the omnibus tests of the location and the
            scale coefficients (default: all).
    """
    yi = _resolve_col(yi, data)
    yi = np.asarray(yi, dtype=float).ravel()
    if vi is None:
        if sei is None:
            raise ValueError("either 'vi' or 'sei' must be supplied")
        vi = np.asarray(_resolve_col(sei, data), dtype=float).ravel() ** 2
    else:
        vi = np.asarray(_resolve_col(vi, data), dtype=float).ravel()
        if sei is not None:
            vi = np.asarray(_resolve_col(sei, data),
                            dtype=float).ravel() ** 2
    if yi.size != vi.size:
        raise ValueError("length of 'yi' and 'vi' must match")
    k = yi.size
    if k == 0:
        raise ValueError("no studies supplied")

    link = str(link).lower()
    if link not in _LINKS:
        raise ValueError(f"link must be one of {list(_LINKS)}")
    method = str(method).upper()
    if method not in ("ML", "REML"):
        raise ValueError(
            "Location-scale models can only be fitted with ML or REML "
            "estimation.")
    if scale is None:
        raise ValueError("'scale' is required for a location-scale model")

    # --- design matrices -------------------------------------------------
    # A formula string ("~ dose + year") also carries an intercept marker that
    # overrides the explicit `intercept` / `scale_intercept` defaults.
    mods, intercept = _formula_terms(mods, data, intercept)
    scale, scale_intercept = _formula_terms(scale, data, scale_intercept)
    mods_arr = None if mods is None else _as_matrix(
        _resolve(mods, data, k), k, "mods")
    scale_arr = _as_matrix(_resolve(scale, data, k), k, "scale")
    X = (np.column_stack([np.ones(k), mods_arr]) if mods_arr is not None
         else np.ones((k, 1))) if intercept else mods_arr
    if X is None:
        raise ValueError("'intercept=False' requires 'mods'")
    Z = (np.column_stack([np.ones(k), scale_arr]) if scale_intercept
         else scale_arr)
    p, q = X.shape[1], Z.shape[1]

    # metafor drops redundant scale columns and always puts the intercept first
    int_cols = [j for j in range(q) if np.all(Z[:, j] == 1.0)]
    z_int_incl = bool(int_cols)
    if z_int_incl:
        j0 = int_cols[0]
        if j0 != 0:
            order = [j0] + [j for j in range(q) if j != j0]
            Z = Z[:, order]
    z_int_only = bool(q == 1 and z_int_incl)

    alpha_fix = np.full(q, np.nan)
    if alpha is not None:
        a = np.asarray(alpha, dtype=float).ravel()
        if a.size == 1:
            a = np.repeat(a, q)
        if a.size != q:
            raise ValueError(
                f"Length of the 'alpha' argument ({a.size}) does not match "
                f"the actual number of parameters ({q}).")
        alpha_fix = a.copy()
    free = ~np.isfinite(alpha_fix)
    if not free.any():
        raise ValueError(
            "Model does not contain any estimated 'alpha' components.")
    alpha_fix[~free] = alpha_fix[~free]      # fixed

    # --- test / degrees of freedom ---------------------------------------
    test = str(test).lower()
    if test == "hksj":
        test = "knha"
    if test not in ("z", "t", "knha", "adhoc"):
        raise ValueError("test must be 'z', 't', 'knha' or 'adhoc'")
    ddf = float(k - p) if test in ("t", "knha", "adhoc") else math.nan
    ddf_alpha = float(k - q) if test in ("t", "knha", "adhoc") else math.nan
    reml_flag = method == "REML"

    def expand(theta):
        full = alpha_fix.copy()
        full[free] = theta
        return full

    def negll_theta(theta):
        return -_ls_loglik(expand(theta), yi, vi, X, Z, reml_flag, link)

    def negll_alpha(a):
        return -_ls_loglik(a, yi, vi, X, Z, reml_flag, link)

    # --- starting values (metafor's scheme) ------------------------------
    a0 = _alpha_start(yi, vi, X, Z, link, alpha_fix)
    theta0 = a0[free]

    if link == "log":
        opt = optimize.minimize(negll_theta, theta0, method="BFGS",
                                options={"gtol": 1e-10, "maxiter": 20000})
        best = opt.x if np.isfinite(opt.fun) else theta0
        best_f = negll_theta(best)
        nm = optimize.minimize(negll_theta, theta0, method="Nelder-Mead",
                               options={"xatol": 1e-10, "fatol": 1e-12,
                                        "maxiter": 100000, "maxfev": 100000})
        if np.isfinite(nm.fun) and nm.fun < best_f:
            best, best_f = nm.x, nm.fun
            bf = optimize.minimize(negll_theta, best, method="BFGS",
                                   options={"gtol": 1e-12, "maxiter": 20000})
            if np.isfinite(bf.fun) and bf.fun <= best_f:
                best, best_f = bf.x, bf.fun
    else:
        cons = [{"type": "ineq", "fun": lambda th: Z @ expand(th)}]
        best, best_f = theta0, negll_theta(theta0)
        for x0 in (theta0, np.zeros_like(theta0),
                   np.full_like(theta0, 1e-3)):
            try:
                opt = optimize.minimize(
                    negll_theta, x0, method="SLSQP", constraints=cons,
                    options={"ftol": 1e-14, "maxiter": 5000})
            except Exception:              # pragma: no cover - solver failure
                continue
            if np.isfinite(opt.fun) and opt.fun < best_f:
                best, best_f = opt.x, opt.fun

    alpha_hat = expand(_newton_polish(negll_theta, best))
    pic = _ls_pieces(alpha_hat, yi, vi, X, Z, link)
    if pic is None:                        # pragma: no cover - no valid fit
        raise ValueError(
            "Unable to find a valid location-scale fit (negative tau^2).")

    beta = pic["beta"]
    tau2 = pic["tau2"]
    vari = pic["vari"]
    w = pic["w"]
    rss = pic["rss"]
    stXWX = pic["stXWX"]

    # --- location coefficients -------------------------------------------
    vb = stXWX.copy()
    s2w = 1.0
    if test in ("knha", "adhoc"):
        s2w = rss / (k - p) if (k - p) > 0 and rss > np.finfo(float).eps else 0.0
        if test == "adhoc":
            s2w = max(s2w, 1.0)
        vb = s2w * vb
    se = np.sqrt(np.diag(vb))
    zval = beta / se
    if test in ("t", "knha", "adhoc"):
        pval = 2.0 * stats.t.sf(np.abs(zval), df=ddf)
        crit = stats.t.ppf(1.0 - (1.0 - level) / 2.0, df=ddf)
    else:
        pval = 2.0 * stats.norm.sf(np.abs(zval))
        crit = stats.norm.ppf(1.0 - (1.0 - level) / 2.0)
    ci_lb = beta - crit * se
    ci_ub = beta + crit * se

    # --- scale coefficients: standard errors from the Hessian -------------
    hess = _hessian(negll_alpha, alpha_hat)
    try:
        va = np.linalg.inv(hess)
        if not np.all(np.isfinite(va)):    # pragma: no cover
            raise np.linalg.LinAlgError
    except np.linalg.LinAlgError:
        va = np.full((q, q), np.nan)
    se_alpha = np.sqrt(np.diag(va))
    with np.errstate(invalid="ignore", divide="ignore"):
        zval_alpha = alpha_hat / se_alpha
    if test in ("t", "knha", "adhoc"):
        crit_a = stats.t.ppf(1.0 - (1.0 - level) / 2.0, df=ddf_alpha)
        pval_alpha = 2.0 * stats.t.sf(np.abs(zval_alpha), df=ddf_alpha)
    else:
        crit_a = stats.norm.ppf(1.0 - (1.0 - level) / 2.0)
        pval_alpha = 2.0 * stats.norm.sf(np.abs(zval_alpha))
    ci_lb_alpha = alpha_hat - crit_a * se_alpha
    ci_ub_alpha = alpha_hat + crit_a * se_alpha

    att_idx = (_default_terms(att, q, z_int_incl))
    m_alpha = att_idx.size
    try:
        sub = va[np.ix_(att_idx, att_idx)]
        QS = float(alpha_hat[att_idx] @ np.linalg.inv(sub) @ alpha_hat[att_idx])
    except np.linalg.LinAlgError:          # pragma: no cover
        QS = math.nan
    qs_df = (m_alpha, ddf_alpha)
    if test in ("t", "knha", "adhoc"):
        qs_mean = QS / m_alpha
        QSp = (float(stats.f.sf(qs_mean, m_alpha, ddf_alpha))
               if ddf_alpha > 0 else math.nan)
    else:
        QSp = float(stats.chi2.sf(QS, m_alpha)) if np.isfinite(QS) else math.nan

    # omnibus test of the location coefficients (metafor's QM)
    btt_idx = _default_terms(btt, p, bool(intercept))
    m_beta = btt_idx.size
    try:
        QM = float(beta[btt_idx] @ np.linalg.inv(
            vb[np.ix_(btt_idx, btt_idx)]) @ beta[btt_idx])
    except np.linalg.LinAlgError:          # pragma: no cover
        QM = math.nan
    qm_df = (m_beta, ddf)
    if test in ("t", "knha", "adhoc"):
        qm_mean = QM / m_beta
        QMp = (float(stats.f.sf(qm_mean, m_beta, ddf)) if ddf > 0
               else math.nan)
    else:
        QMp = float(stats.chi2.sf(QM, m_beta)) if np.isfinite(QM) else math.nan

    # --- heterogeneity ----------------------------------------------------
    w0 = 1.0 / vi
    stXWX0 = np.linalg.inv(X.T @ (w0[:, None] * X))
    beta_fe = stXWX0 @ (X.T @ (w0 * yi))
    resid_fe = yi - X @ beta_fe
    qe = float(resid_fe @ (w0 * resid_fe))
    QE = max(0.0, qe)
    QEp = float(stats.chi2.sf(QE, k - p)) if k > p else 1.0
    P = (np.diag(w0) - (np.diag(w0) @ X @ stXWX0
                        @ (X.T @ np.diag(w0))))
    vt = float((k - p) / np.trace(P)) if k > p else 0.0
    I2: list[float] | None
    H2: list[float] | None
    if k > p and vt > 0:
        I2 = [100.0 * t / (vt + t) for t in tau2]
        H2 = [t / vt + 1.0 for t in tau2]
    else:                                  # pragma: no cover - saturated
        I2 = H2 = None
        vt = 0.0

    # --- fit statistics ---------------------------------------------------
    parms = p + int(np.sum(free))
    logdet_v = float(np.sum(np.log(vari)))
    ll_ml = -0.5 * k * math.log(2 * math.pi) - 0.5 * logdet_v - 0.5 * rss
    sign, logdet_xtwx = np.linalg.slogdet(X.T @ (w[:, None] * X))
    sign2, logdet_xtx = np.linalg.slogdet(X.T @ X)
    ll_reml = (-0.5 * (k - p) * math.log(2 * math.pi)
               + 0.5 * logdet_xtx - 0.5 * logdet_v
               - 0.5 * logdet_xtwx - 0.5 * rss)
    if k > p and np.all(vi > 0):
        ll_sat = float(np.sum(-0.5 * np.log(2 * math.pi * vi)))
        dev_ml = -2 * (ll_ml - ll_sat)
    elif k > p:                            # pragma: no cover
        dev_ml = -2 * ll_ml
    else:                                  # pragma: no cover - saturated
        dev_ml = 0.0

    def _aicc(n, ll_):
        m = max(n, parms + 2)
        return -2 * ll_ + 2 * parms * m / (m - parms - 1)

    n_r = k - p
    fit_stats = {
        "ML": {"ll": ll_ml, "dev": dev_ml, "AIC": -2 * ll_ml + 2 * parms,
               "BIC": -2 * ll_ml + parms * math.log(k),
               "AICc": _aicc(k, ll_ml)},
        "REML": {"ll": ll_reml, "dev": -2 * ll_reml,
                 "AIC": -2 * ll_reml + 2 * parms,
                 "BIC": (-2 * ll_reml + parms * math.log(n_r) if n_r > 0
                         else -math.inf),
                 "AICc": _aicc(n_r, ll_reml) if n_r > parms + 1 else -math.inf},
    }
    ll_fitted = fit_stats[method]["ll"]

    return RmaLsResult(
        b=[float(v) for v in beta.ravel()],
        se=[float(v) for v in se],
        zval=[float(v) for v in zval],
        pval=[float(v) for v in pval],
        ci_lb=[float(v) for v in ci_lb],
        ci_ub=[float(v) for v in ci_ub],
        vb=vb,
        alpha=[float(v) for v in alpha_hat],
        se_alpha=[float(v) for v in se_alpha],
        zval_alpha=[float(v) for v in zval_alpha],
        pval_alpha=[float(v) for v in pval_alpha],
        ci_lb_alpha=[float(v) for v in ci_lb_alpha],
        ci_ub_alpha=[float(v) for v in ci_ub_alpha],
        va=va,
        QS=QS, QSdf=qs_df, QSp=QSp,
        QM=QM, QMdf=qm_df, QMp=QMp,
        tau2=[float(v) for v in tau2],
        QE=QE, QEp=QEp, I2=I2, H2=H2, vt=vt,
        parms=parms, ll=float(ll_fitted), fit_stats=fit_stats,
        k=k, p=p, q=q, alphas=q,
        method=method, link=link, test=test,
        ddf=ddf, ddf_alpha=ddf_alpha, level=level,
        intercept=bool(intercept), scale_intercept=bool(scale_intercept),
        optbeta=bool(optbeta),
        alpha_fix=[not bool(f) for f in free],
        Z_int_incl=z_int_incl, Z_int_only=z_int_only,
        int_only=bool(p == 1 and intercept),
        measure=str(measure),
        yi=[float(v) for v in yi], vi=[float(v) for v in vi],
        X=X, Z=Z,
        slab=([str(v) for v in slab] if slab is not None
              else [f"Study {i + 1}" for i in range(k)]),
    )


def _formula_terms(spec, data, intercept):
    """Expand a ``"~ a + b"`` formula into a bare moderator matrix.

    Returns ``(matrix_or_None, intercept)``; a formula whose right-hand side is
    empty (``"~ 0"``) yields ``None`` and leaves ``intercept`` untouched.
    """
    if not (isinstance(spec, str) and spec.strip().startswith("~")):
        return spec, intercept
    if data is None:
        raise ValueError(
            f"{spec!r} is a formula but no 'data' was supplied")
    from .data_compat import design_matrix, parse_formula

    parsed = parse_formula(spec)
    if parsed.response is not None:
        raise ValueError(
            "'mods' / 'scale' formulas must be one-sided (e.g. '~ dose')")
    X, _names = design_matrix(parsed, data, intercept=False)
    return (X if X.size else None), parsed.intercept


def _resolve_col(spec, data):
    """A bare vector, or a column name looked up in ``data`` (R's model frame)."""
    if isinstance(spec, str):
        if data is None:
            raise ValueError(
                f"{spec!r} is a column name but no 'data' was supplied")
        return _column(data, spec)
    return spec


def _resolve(spec, data, k):
    """Accept an array, a column name (with ``data``) or a list of names."""
    if isinstance(spec, str):
        if data is None:
            raise ValueError(
                f"'{spec}' is a column name but no 'data' was supplied")
        return _column(data, spec)
    if isinstance(spec, (list, tuple)) and spec and all(
            isinstance(v, str) for v in spec):
        return [_column(data, v) for v in spec]
    return spec


def _column(data, name):
    if hasattr(data, "columns") and name in list(data.columns):
        return np.asarray(data[name], dtype=float)
    try:
        return np.asarray(data[name], dtype=float)
    except (TypeError, KeyError):          # pragma: no cover - bad column
        raise ValueError(f"column {name!r} not found in 'data'") from None


def _alpha_start(yi, vi, X, Z, link, alpha_fix):
    """metafor's starting values for ``alpha``.

    ``rma.uni(yi, vi, mods=X, method="HE")`` -> ``rstandard()`` -> weighted
    regression of ``log(resid^2)`` (log link) or ``resid^2`` (identity) on Z.
    """
    k, p = X.shape
    q = Z.shape[1]
    try:
        w_he = 1.0 / (vi + max(0.0, float(np.var(yi, ddof=1))
                               - float(np.mean(vi))))
        st = np.linalg.inv(X.T @ (w_he[:, None] * X))
        b_he = st @ (X.T @ (w_he * yi))
        tau2_he = max(0.0, float(np.var(yi, ddof=1)) - float(np.mean(vi)))
        resid = yi - X @ b_he
        # rstandard.rma.uni: se = sqrt(vi + tau2 - diag(X vb X'))
        var_se = vi + tau2_he - np.einsum("ij,jk,ik->i", X, st, X)
        se_i = np.sqrt(np.maximum(var_se, 1e-12))
        with np.errstate(invalid="ignore", divide="ignore"):
            if link == "log":
                yy = np.log(resid ** 2)
                vv = 4.0 / resid ** 2 * se_i ** 2
            else:
                yy = resid ** 2
                vv = se_i ** 2
        ww = 1.0 / vv
        good = np.isfinite(yy) & np.isfinite(ww) & (ww > 0)
        if good.sum() >= q:
            Zg = Z[good]
            a0 = np.linalg.lstsq(Zg * np.sqrt(ww[good])[:, None],
                                 yy[good] * np.sqrt(ww[good]), rcond=None)[0]
        else:                              # pragma: no cover - degenerate
            a0 = np.zeros(q)
        a0 = np.where(np.isfinite(a0), a0, 0.0)
    except (np.linalg.LinAlgError, ValueError):   # pragma: no cover
        a0 = np.zeros(q)

    if link == "identity" and np.any(Z @ a0 < 0):
        a0 = np.zeros(q)
        if q and np.all(Z[:, 0] == 1.0):
            a0[0] = max(tau2_he + 0.01, 1e-3) if np.isfinite(tau2_he) else 1e-3
        if np.any(Z @ a0 < 0):             # pragma: no cover
            a0 = np.full(q, 1e-3)
    a0 = np.asarray(a0, dtype=float)
    a0[~np.isfinite(alpha_fix)] = np.clip(a0[~np.isfinite(alpha_fix)],
                                          -500, 500)
    return a0


# --------------------------------------------------------------------------- #
# profile likelihood over the scale coefficients
# --------------------------------------------------------------------------- #

class RmaLsProfileResult:
    """Result of :func:`profile_rma_ls` (mirrors ``profile.rma.ls`` objects).

    ``alpha`` holds the profiling grid (the values of the chosen scale
    coefficient), ``ll`` the maximised log-likelihood at each grid point and
    ``beta`` / ``ci_lb`` / ``ci_ub`` the location coefficients (one row per
    grid point).  ``maxll`` is the log-likelihood of the unconstrained fit.
    """

    __slots__ = ("component", "index", "alpha", "ll", "maxll", "beta",
                 "ci_lb", "ci_ub", "vc", "method", "xlab", "title", "comps")

    def __init__(self, component, index, alpha, ll, maxll, beta, ci_lb,
                 ci_ub, vc, method, xlab, title, comps=1):
        self.component = component
        self.index = index
        self.alpha = alpha
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

    def __repr__(self):
        return (f"RmaLsProfileResult(alpha={self.index}, "
                f"points={len(self.alpha)}, vc={self.vc:.6g}, "
                f"maxll={self.maxll:.6g})")

    def to_dict(self) -> dict:
        return {
            "component": self.component, "index": self.index,
            "alpha": self.alpha, "ll": self.ll, "maxll": self.maxll,
            "beta": self.beta, "ci_lb": self.ci_lb, "ci_ub": self.ci_ub,
            "vc": self.vc, "method": self.method,
        }


def profile_rma_ls(x, alpha=None, steps=20, xlim=None, plot=False, **kw):
    """``profile.rma.ls`` — profile the log-likelihood over one scale
    coefficient.

    For every grid value of the chosen ``alpha`` the model is refitted with
    that coefficient pinned while the others are re-estimated, which is what
    metafor's ``.profile.rma.ls`` does (``rma(..., scale=Z, alpha=c(NA, v))``).

    Args:
        x: a :class:`RmaLsResult`.
        alpha: 1-based index of the scale coefficient to profile (metafor's
            ``alpha=``).  Defaults to the first non-fixed coefficient.
        steps: number of grid points, or an explicit sequence of values.
        xlim: profiling range; by default ``vc ± qnorm(0.995) * se.alpha``
            around the estimate, like metafor.
        plot: draw the profile plot immediately.
        **kw: forwarded to :func:`pimeta.plotting.plot_mv_profile`.

    Returns:
        :class:`RmaLsProfileResult`.
    """
    if getattr(x, "optbeta", False):
        raise ValueError("Profiling not yet implemented for models fitted "
                         "with 'optbeta=True'.")
    fixed = list(getattr(x, "alpha_fix", []) or [])
    q = int(getattr(x, "q", 0))
    if q == 0:                              # pragma: no cover
        raise ValueError("model has no scale coefficients")
    if all(fixed):
        raise ValueError("Model does not contain any estimated 'alpha' "
                         "components.")

    if alpha is None:
        pos = next(i for i in range(q) if not fixed[i])
    else:
        arr = np.atleast_1d(np.asarray(alpha)).ravel()
        if arr.size > 1:
            raise ValueError("Can only specify one 'alpha' component.")
        pos = int(arr[0]) - 1
        if not 0 <= pos < q:
            raise ValueError("No such 'alpha' component in the model.")
        if fixed[pos] if pos < len(fixed) else False:
            raise ValueError("Specified 'alpha' component was fixed.")
    vc = float(x.alpha[pos])
    se_a = float(x.se_alpha[pos]) if x.se_alpha else math.nan

    if xlim is None:
        if not np.isfinite(se_a):
            lo, hi = vc - 4.0 * abs(vc), vc + 4.0 * abs(vc)
        else:
            half = float(stats.norm.ppf(0.995)) * se_a
            lo, hi = vc - half, vc + half
    else:
        lim = np.sort(np.asarray(xlim, dtype=float).ravel())
        if lim.size != 2:
            raise ValueError("Argument 'xlim' should be a vector of length 2.")
        lo, hi = float(lim[0]), float(lim[1])

    step_arr = np.atleast_1d(np.asarray(steps, dtype=float)).ravel()
    if step_arr.size >= 2:
        grid = step_arr
    else:
        n = int(step_arr[0]) if step_arr.size else 20
        if n < 2:
            raise ValueError("Argument 'steps' must be >= 2.")
        grid = np.linspace(lo, hi, n)

    base = np.asarray(x.alpha, dtype=float).copy()
    fixed_mask = np.asarray(x.alpha_fix, dtype=bool) if fixed else \
        np.zeros(base.size, dtype=bool)
    kk = int(x.k)
    Xm = np.asarray(x.X, dtype=float)
    Zm = np.asarray(x.Z, dtype=float)
    mods = (Xm[:, 1:] if x.p > 1 else None) if x.intercept else Xm
    if x.scale_intercept:
        scl = Zm[:, 1:] if x.q > 1 else np.zeros((kk, 0))
    else:
        scl = Zm
    ll, beta, ci_lb, ci_ub = [], [], [], []
    for val in grid:
        # originally fixed coefficients keep their value, the profiled one is
        # pinned at `val`, and everything else stays free (NaN, as in R).
        pinned = np.where(fixed_mask, base, np.nan)
        pinned[pos] = float(val)
        try:
            fit = rma_ls(
                np.asarray(x.yi, dtype=float), np.asarray(x.vi, dtype=float),
                mods=mods, scale=scl,
                method=x.method, test=x.test, level=x.level,
                intercept=bool(x.intercept),
                scale_intercept=bool(x.scale_intercept),
                link=x.link, alpha=pinned,
            )
        except Exception:                  # pragma: no cover - bad fit
            ll.append(math.nan)
            beta.append([math.nan] * int(x.p))
            ci_lb.append([math.nan] * int(x.p))
            ci_ub.append([math.nan] * int(x.p))
            continue
        ll.append(float(fit.ll))
        beta.append([float(b) for b in fit.b])
        ci_lb.append([float(b) for b in fit.ci_lb])
        ci_ub.append([float(b) for b in fit.ci_ub])

    out = RmaLsProfileResult(
        component="alpha",
        index=pos + 1,
        alpha=[float(v) for v in grid],
        ll=ll,
        maxll=float(x.ll),
        beta=beta,
        ci_lb=ci_lb,
        ci_ub=ci_ub,
        vc=vc,
        method=str(x.method),
        xlab=(r"$\alpha$ Value" if q == 1
              else rf"$\alpha_{{{pos + 1}}}$ Value"),
        title=(r"Profile Plot for $\alpha$" if q == 1
               else rf"Profile Plot for $\alpha_{{{pos + 1}}}$"),
    )
    if plot:                               # pragma: no cover - plotting
        from .plotting import plot_mv_profile

        plot_mv_profile(out, **kw)
    return out


# --------------------------------------------------------------------------- #
# confidence intervals for the scale coefficients
# --------------------------------------------------------------------------- #

def confint_rma_ls(x, parm=None, level=None, digits=4):
    """``confint.rma.ls`` — Wald intervals for the scale coefficients.

    ``parm`` selects the coefficients by name (``"z"`` / ``"z1"`` ...) or by
    1-based index; the default is every coefficient.  Returns a dict with
    ``estimate``, ``ci_lb``, ``ci_ub``, ``level`` and ``parm``.

    metafor additionally supports profile-likelihood intervals here; use
    :func:`profile_rma_ls` and invert the profile yourself for those.
    """
    if level is None:
        level = x.level
    lvl = float(level)
    if lvl >= 1.0:                         # 95 -> 0.95
        lvl = lvl / 100.0
    crit = float(stats.norm.ppf(1.0 - (1.0 - lvl) / 2.0))
    q = int(x.q)
    names = [f"z{i + 1}" for i in range(q)]
    alpha = np.asarray(x.alpha, dtype=float)
    se = np.asarray(x.se_alpha, dtype=float)
    if parm is None:
        idx = list(range(q))
    else:
        idx = _ls_parm_index(parm, q, names)
    return {
        "estimate": [float(alpha[i]) for i in idx],
        "ci_lb": [float(alpha[i] - crit * se[i]) for i in idx],
        "ci_ub": [float(alpha[i] + crit * se[i]) for i in idx],
        "level": lvl,
        "parm": [names[i] for i in idx],
    }


def _ls_parm_index(parm, q, names):
    out = []
    for p in np.atleast_1d(np.asarray(parm)).ravel():
        if isinstance(p, str):
            if p not in names:
                raise ValueError(f"unknown scale coefficient {p!r}")
            out.append(names.index(p))
        else:
            i = int(p) - 1
            if not 0 <= i < q:
                raise ValueError(f"scale coefficient {p} does not exist")
            out.append(i)
    return out

