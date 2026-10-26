"""Additional metafor utilities: simulation, hettest, matreg, profile.

Implements ``metafor::simulate.rma``, ``metafor::hettest``, ``metafor::matreg``
and ``profile.rma.uni``.  ``metafor::vcalc`` lives in :mod:`pimeta.mv.vcov` and
is re-exported here so that the long-standing ``ma.vcalc`` name keeps working.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from scipy import stats

from .checks import attach_diagnostics
from .mv.vcov import vcalc

__all__ = [
    "simulate_rma", "vcalc", "hettest", "HettestResult",
    "matreg", "MatregResult", "profile_rma", "ProfileResult",
]


# --------------------------------------------------------------------------- #
# simulate.rma
# --------------------------------------------------------------------------- #

def simulate_rma(result, nsim=1, seed=None):
    """``metafor::simulate.rma`` — simulate effect sizes from a fitted model.

    Draws ``theta_i ~ N(X_i beta, tau2)`` and ``y_i ~ N(theta_i, v_i)``, i.e.
    new effect sizes with the same sampling variances and the estimated
    heterogeneity.

    Args:
        result: fitted :class:`RmaResult`.
        nsim: number of simulated datasets.
        seed: optional RNG seed.

    Returns:
        ``(k, nsim)`` array of simulated effect sizes.
    """
    from .rma_uni import RmaResult

    if not isinstance(result, RmaResult):
        raise TypeError("argument 'result' must be an RmaResult")
    rng = np.random.default_rng(seed)
    yi = np.asarray(result.yi, dtype=float)
    vi = np.asarray(result.vi, dtype=float)
    X = np.asarray(result.X, dtype=float)
    beta = np.asarray(result.b, dtype=float)
    tau2 = float(result.tau2)

    mu = X @ beta
    out = np.empty((yi.size, int(nsim)))
    for s in range(int(nsim)):
        theta = rng.normal(mu, math.sqrt(tau2) if tau2 > 0 else 0.0)
        out[:, s] = rng.normal(theta, np.sqrt(vi))
    return out


# --------------------------------------------------------------------------- #
# hettest
# --------------------------------------------------------------------------- #

@dataclass
class HettestResult:
    """Result of the (permutation/score) heterogeneity test."""

    method: str
    test: str
    QE: float
    pval: float
    k: int
    p: int


def hettest(result, method=None, test="score", seed=None, iter=1000):
    """``metafor::hettest`` — test for residual heterogeneity.

    Implements metafor's ``.hettest.score`` (Rao's score test evaluated with the
    *estimated* ``tau^2`` and the random-effects weights)::

        vari = tau2_hat + vi;  wi = 1/vari;  S = sum(wi)
        Ui   = 0.5*wi^2*(yi-mu_hat)^2 - 0.5*wi + 0.5*wi^2/S
        I    = outer(-1/(sqrt(2)*S*vari^2)) with
               diag(I) = 0.5*(wi - 1/(S*vari^2))^2
        X2   = Ui' inv(I) Ui          ~ chi^2(k-1)

    Args:
        result: fitted :class:`RmaResult` (intercept-only models only).
        method: ``"REML"`` (default) or ``"ML"`` — selects the score variant.
        test: ``"score"`` (metafor default) or ``"QE"`` (classical Cochran test,
            provided as a convenience).
        seed, iter: accepted for API compatibility (no bootstrap is run).

    Returns:
        :class:`HettestResult`.
    """
    from .rma_uni import RmaResult

    if not isinstance(result, RmaResult):
        raise TypeError("argument 'result' must be an RmaResult")
    yi = np.asarray(result.yi, dtype=float)
    vi = np.asarray(result.vi, dtype=float)
    X = np.asarray(result.X, dtype=float)
    k, p = X.shape
    tau2 = float(result.tau2)
    method = (method or result.method or "REML").upper()

    if test == "QE":
        w = 1.0 / vi
        WX = X * w[:, None]
        XtWX_inv = np.linalg.pinv(X.T @ WX)
        beta_fe = XtWX_inv @ (X.T @ (w * yi))
        resid = yi - X @ beta_fe
        QE = float(resid @ (w * resid))
        pval = float(stats.chi2.sf(QE, k - p)) if k > p else np.nan
        return HettestResult(method=method, test="QE", QE=QE, pval=pval,
                             k=k, p=p)

    # metafor's .hettest.score (intercept-only case, mu_hat = RE pooled mean)
    vari = tau2 + vi
    wi = 1.0 / vari
    S = float(np.sum(wi))
    mu_hat = float(np.sum(wi * yi) / S)

    if method == "ML":
        x2 = float(0.5 * np.sum((wi * (yi - mu_hat) ** 2 - 1.0) ** 2))
    else:
        Ui = 0.5 * wi ** 2 * (yi - mu_hat) ** 2 - 0.5 * wi + 0.5 * wi ** 2 / S
        I = np.outer(1.0 / (math.sqrt(2.0) * S * vari ** 2),
                     1.0 / (math.sqrt(2.0) * S * vari ** 2))
        np.fill_diagonal(I, 0.5 * (wi - 1.0 / (S * vari ** 2)) ** 2)
        try:
            x2 = float(Ui @ np.linalg.solve(I, Ui))
        except np.linalg.LinAlgError:
            return HettestResult(method=method, test="score", QE=np.nan,
                                 pval=np.nan, k=k, p=p)

    df = k - 1
    pval = float(stats.chi2.sf(x2, df)) if df > 0 else np.nan
    return HettestResult(method=method, test="score", QE=float(x2),
                         pval=pval, k=k, p=p)


# --------------------------------------------------------------------------- #
# matreg
# --------------------------------------------------------------------------- #

@attach_diagnostics
@dataclass
class MatregResult:
    """Result of :func:`matreg` (``metafor::matreg``).

    ``test`` is ``"t"`` when the model was fitted from ``n`` (the standard
    error of the standardised path coefficients comes from the residual mean
    square and the coefficient is referred to a t distribution) and ``"z"``
    when a sampling covariance matrix ``V`` was supplied (delta-method
    standard errors, normal reference distribution).
    """

    beta: list[float]
    se: list[float]
    pval: list[float]
    ci_lb: list[float]
    ci_ub: list[float]
    vb: np.ndarray
    names: list[str]
    test: str
    level: float                     # tail probability, as R stores it
    intercept: bool
    p: int                           # number of regression coefficients
    parms: int
    R2: float
    # --- fitted from `n` (test = "t") -----------------------------------
    tval: list[float] | None = None
    df: int | None = None
    df_residual: int | None = None
    R2adj: float | None = None
    F: list[float] | None = None     # (value, df1, df2)
    Fdf: list[int] | None = None
    Fp: float | None = None
    sigma2_ml: float | None = None
    sigma2_reml: float | None = None
    nobs: int | None = None
    deviance: float | None = None
    fit_stats: dict | None = None
    # --- fitted from `V` (test = "z") -----------------------------------
    zval: list[float] | None = None
    QM: float | None = None
    QMdf: list[int] | None = None
    QMp: float | None = None
    # --- kept for backwards compatibility -------------------------------
    k: int = 0                       # number of predictors (R: m)
    yi: list[float] | None = None
    vi: list[float] | None = None
    QE: float | None = None
    QEp: float | None = None

    def to_dict(self) -> dict:
        return {
            "beta": self.beta, "se": self.se, "pval": self.pval,
            "ci_lb": self.ci_lb, "ci_ub": self.ci_ub,
            "vb": np.asarray(self.vb).tolist(),
            "names": self.names, "test": self.test, "level": self.level,
            "intercept": self.intercept, "p": self.p, "parms": self.parms,
            "R2": self.R2, "tval": self.tval, "df": self.df,
            "R2adj": self.R2adj, "F": self.F, "Fp": self.Fp,
            "sigma2_ml": self.sigma2_ml, "sigma2_reml": self.sigma2_reml,
            "nobs": self.nobs, "deviance": self.deviance,
            "zval": self.zval, "QM": self.QM, "QMp": self.QMp,
        }


def _matreg_level(level):
    """R's ``.level()``: accept 95, 0.95 or 0.05 and return the tail area."""
    lv = float(level)
    if lv > 1.0:
        lv = lv / 100.0
    if lv >= 0.5:
        lv = 1.0 - lv
    if not 0.0 < lv < 0.5:
        raise ValueError("argument 'level' must be a confidence level "
                         "(e.g. 95 or 0.95)")
    return lv


def _matreg_chol_inv(a):
    """``chol2inv(chol(a))`` with a clear error when `a` is not PD."""
    try:
        c = np.linalg.cholesky(a)
    except np.linalg.LinAlgError as exc:  # pragma: no cover - exotic input
        raise ValueError(
            "Cannot invert R[x,x] matrix.") from exc
    ci = np.linalg.inv(c)
    # a = c c', so a^-1 = (c')^-1 c^-1 = ci' ci
    return ci.T @ ci


def _matreg_index(R, which, p, names):
    """Resolve a 1-based index / variable name to 0-based positions."""
    if which is None:
        return None
    out = []
    for v in np.atleast_1d(which):
        if isinstance(v, str):
            if names is None:
                raise ValueError(
                    "'R' must have dimension names when specifying a variable "
                    "name for 'y'/'x'.")
            if len(set(names)) != len(names):
                raise ValueError("Dimension names of 'R' must be unique.")
            hits = [i for i, nm in enumerate(names) if nm == v]
            if not hits:
                raise ValueError(f"Could not find variable {v!r} in the 'R' "
                                 f"matrix.")
            out.append(hits[0])
        else:
            j = int(round(float(v))) - 1
            if j < 0 or j >= p:
                raise ValueError(f"Indices in 'x' must be >= 1 or <= {p}.")
            out.append(j)
    return out


def matreg(y, x=None, R=None, n=None, V=None, cov=False, means=None,
           ztor=False, nearpd=False, level=95, digits=4, names=None,
           measure=None):
    """``metafor::matreg`` — regression from a correlation / covariance matrix.

    Transcribes ``metafor::matreg``: the standardised path coefficients are
    ``b = R[x,x]^-1 R[x,y]`` and their standard errors come either from the
    residual mean square (``n`` supplied) or from a delta-method propagation of
    a sampling covariance matrix ``V`` over the unique elements of ``R``.

    Args:
        y: 1-based index of the outcome variable in ``R`` (or its name, see
            ``names``).
        x: 1-based indices of the predictors.  Defaults to every variable
            except ``y``.
        R: symmetric correlation matrix.
        n: sample size — exclusive with ``V``.
        V: sampling covariance matrix of the ``p(p-1)/2`` unique
            off-diagonal elements of ``R`` (column-major lower triangle, i.e.
            R's ``R[lower.tri(R)]``), or ``p(p+1)/2`` when ``cov=True``.
        cov: treat ``R`` as a covariance matrix and add an intercept.
        means: variable means, used to centre the intercept when ``cov=True``.
        ztor: ``R`` holds Fisher z values; transform back with ``tanh``.
        nearpd: repair a non-positive-definite ``R[x,x]`` by clipping its
            eigenvalues (R uses ``Matrix::nearPD``, so the repaired matrix and
            therefore the estimates need not agree to the last digit).
        level: confidence level (``95``, ``0.95`` or ``0.05``).  The result
            stores the tail area, exactly like R.
        names: variable names for ``R``, so that ``y``/``x`` can be strings.

    Returns:
        :class:`MatregResult`.
    """
    if R is None:
        raise ValueError("Must specify the 'R' argument.")
    R = np.asarray(R, dtype=float)
    if R.ndim != 2 or R.shape[0] != R.shape[1]:
        raise ValueError("Argument 'R' must be a square matrix.")
    R = R.copy()
    p_all = R.shape[0]
    if p_all <= 1:
        raise ValueError("The 'R' matrix must be at least of size 2x2.")
    if measure is not None:
        raise TypeError(
            "argument 'measure' is not a metafor argument; use 'ztor', "
            "'cov' and 'V' instead")
    alpha = _matreg_level(level)
    if not isinstance(cov, bool):
        raise ValueError("Argument 'cov' must be either TRUE or FALSE.")
    if not isinstance(ztor, bool):
        raise ValueError("Argument 'ztor' must be either TRUE or FALSE.")
    if cov and ztor:
        raise ValueError(
            "Cannot use a covariance matrix as input when 'ztor=True'.")
    if (V is None) == (n is None):
        raise ValueError("Either 'V' or 'n' must be specified.")

    yj = _matreg_index(R, [y], p_all, names)[0]
    xs = _matreg_index(R, x, p_all, names)
    if xs is None:
        xs = [i for i in range(p_all) if i != yj]
    if len(set(xs)) != len(xs):
        raise ValueError("Argument 'x' should not contain duplicated elements.")
    if yj in xs:
        raise ValueError("Variable 'y' should not be an element of 'x'.")
    m = len(xs)
    var_names = ([str(names[i]) for i in xs] if names is not None
                 else [f"x{i + 1}" for i in xs])

    R[np.triu_indices(p_all, 1)] = R.T[np.triu_indices(p_all, 1)]

    if V is not None:
        V = np.asarray(V, dtype=float)
        if V.ndim != 2 or V.shape[0] != V.shape[1]:
            raise ValueError("Argument 'V' must be a square matrix.")
        V = V.copy()
        V[np.triu_indices(V.shape[0], 1)] = V.T[np.triu_indices(V.shape[0], 1)]
        s = p_all * (p_all + 1) // 2 if cov else p_all * (p_all - 1) // 2
        if V.shape[0] != s:
            raise ValueError(
                f"Dimensions of 'V' ({V.shape[0]}x{V.shape[1]}) do not match "
                f"the number of elements in 'R' ({s}).")

    if ztor:
        if V is not None:
            idx = np.tril_indices(p_all, -1)
            zij = R[idx]
            d = 2.0 / (np.cosh(2.0 * zij) + 1.0)
            V = np.diag(d) @ V @ np.diag(d)
        R = np.tanh(R)
        np.fill_diagonal(R, 1.0)

    if cov:
        S = R.copy()
        d = np.sqrt(np.diag(S))
        R = S / np.outer(d, d)
        sdy = float(d[yj])
        sdx = d[xs]
    else:
        if np.any(np.abs(R) > 1):
            raise ValueError("Argument 'R' must be a correlation matrix, but "
                             "contains values outside [-1,1].")
        np.fill_diagonal(R, 1.0)
        sdy = 1.0
        sdx = np.ones(m)

    Rxy = R[np.ix_(xs, [yj])]          # (m, 1)
    Rxx = R[np.ix_(xs, xs)]
    invRxx = _matreg_chol_inv(Rxx) if not nearpd else None
    if invRxx is None:                 # nearpd=True
        w, Q = np.linalg.eigh(Rxx)
        Rxx = Q @ np.diag(np.clip(w, 1e-8, None)) @ Q.T
        invRxx = _matreg_chol_inv(Rxx)
    b = invRxx @ Rxy                   # (m, 1)

    if n is not None:
        if np.asarray(n).size != 1:
            raise ValueError("Argument 'n' should be a single number.")
        n_val = float(np.asarray(n).ravel()[0])
        df = n_val - m - (1 if cov else 0)
        if df <= 0:
            raise ValueError("Cannot fit model when 'n' is equal to or less "
                             "than the number of regression coefficients.")

        sse = 1.0 - float((b.T @ Rxy).ravel()[0])
        mse_vb = sse / df              # R uses this one for the coefficient vcov
        vb = mse_vb * invRxx
        R2 = 1.0 - sse
        R2adj = 1.0 - (1.0 - R2) * ((n_val - (1 if cov else 0)) / df)
        Fval = (R2 / m) / mse_vb
        Fp = float(stats.f.sf(Fval, m, df))
        mse = sdy ** 2 * (n_val - 1.0) * (1.0 - R2) / df   # R's `mse`
        sigma2_ml = mse * df / n_val
        sigma2_reml = mse

        coef = np.asarray(b, dtype=float).ravel()
        if cov:
            b_resc = coef * sdy / sdx
            intercept = 0.0 if means is None else float(
                np.asarray(means, dtype=float)[yj]
                - np.asarray(means, dtype=float)[xs] @ b_resc)
            coef_full = np.concatenate([[intercept], b_resc])
            blk = np.zeros((m + 1, m + 1), dtype=float)
            blk[1:, 1:] = S[np.ix_(xs, xs)]
            # R: tcrossprod(c(1, means[x])) with means defaulting to zeros, so
            # the intercept row/column is always 1/(0...) and XtX stays PD.
            mu = (np.zeros(p_all) if means is None
                  else np.asarray(means, dtype=float))
            mvec = np.concatenate([[1.0], mu[xs]])
            xtx = (n_val - 1.0) * blk + n_val * np.outer(mvec, mvec)
            try:
                inv_xtx = _matreg_chol_inv(xtx)
            except ValueError:                        # pragma: no cover
                inv_xtx = np.full((m + 1, m + 1), np.nan)
            # R reassigns `mse` before this vcov, so the covariance-matrix
            # branch uses the residual mean square mse = sdy^2 (n-1)(1-R2)/df
            # while the plain correlation branch keeps sse/df above.
            vb = mse * inv_xtx
            if means is None:
                coef_full[0] = np.nan
                vb[0, :] = np.nan
                vb[:, 0] = np.nan
            coef_out = coef_full
            names_out = ["intrcpt"] + var_names
        else:
            xtx = Rxx * (n_val - 1.0)
            coef_out = coef
            names_out = var_names

        se = np.sqrt(np.diag(vb))
        with np.errstate(invalid="ignore", divide="ignore"):
            tval = coef_out / se
        pv = 2.0 * stats.t.sf(np.abs(tval), df)
        crit = float(stats.t.isf(alpha / 2.0, df))
        ll_ml = (-n_val / 2.0 * math.log(2.0 * math.pi * sigma2_ml)
                 - 0.5 * (mse * df) / sigma2_ml)
        sign, logdet_xtx = np.linalg.slogdet(xtx)
        ll_reml = (-df / 2.0 * math.log(2.0 * math.pi * sigma2_reml)
                   - 0.5 * (mse * df) / sigma2_reml
                   - 0.5 * (logdet_xtx if sign > 0 else np.nan))
        parms = m + 1
        deviance = mse * df
        fit_stats = {
            "ML": {
                "ll": ll_ml, "dev": deviance,
                "AIC": -2 * ll_ml + 2 * parms,
                "BIC": -2 * ll_ml + parms * math.log(n_val),
                "AICc": -2 * ll_ml + (2 * parms * max(n_val, parms + 2)
                                      / (max(n_val, parms + 2) - parms - 1)),
            },
            "REML": {
                "ll": ll_reml, "dev": -2 * ll_reml,
                "AIC": -2 * ll_reml + 2 * parms,
                "BIC": -2 * ll_reml + parms * math.log(n_val - m),
                "AICc": -2 * ll_reml + (2 * parms * max(n_val - m, parms + 2)
                                        / (max(n_val - m, parms + 2)
                                           - parms - 1)),
            },
        }
        return MatregResult(
            beta=[float(v) for v in coef_out],
            se=[float(v) for v in se],
            pval=[float(v) for v in pv],
            ci_lb=[float(v) for v in coef_out - crit * se],
            ci_ub=[float(v) for v in coef_out + crit * se],
            vb=np.asarray(vb, dtype=float),
            names=names_out,
            test="t",
            level=alpha,
            intercept=bool(cov),
            p=m,
            parms=parms,
            R2=float(R2),
            tval=[float(v) for v in tval],
            df=int(df),
            df_residual=int(df),
            R2adj=float(R2adj),
            F=[float(Fval), float(m), float(df)],
            Fdf=[int(m), int(df)],
            Fp=Fp,
            sigma2_ml=float(sigma2_ml),
            sigma2_reml=float(sigma2_reml),
            nobs=int(n_val),
            deviance=float(deviance),
            fit_stats=fit_stats,
            k=int(m),
            yi=[float(v) for v in np.asarray(Rxy).ravel()],
            vi=[1.0] * m,
        )

    # ---- V supplied: delta-method standard errors, z reference ------------
    R2 = float((b.T @ Rxy).ravel()[0])
    if cov:
        b = b * sdy / sdx
        Rxy = S[np.ix_(xs, [yj])]
        invRxx = (np.diag(1.0 / sdx) @ invRxx @ np.diag(1.0 / sdx))
        u_diag = True
    else:
        u_diag = False

    s = p_all * (p_all + 1) // 2 if u_diag else p_all * (p_all - 1) // 2
    U = np.zeros((p_all, p_all), dtype=int)
    tri = np.tril_indices(p_all, 0 if u_diag else -1)
    for k_, (i, j) in enumerate(zip(*tri), start=1):
        U[i, j] = k_
        U[j, i] = k_
    Uxx = U[np.ix_(xs, xs)]
    Uxy = U[np.ix_(xs, [yj])]
    set_xx = {int(v) for v in Uxx.ravel() if v}
    set_xy = {int(v) for v in Uxy.ravel() if v}

    A = np.zeros((m, s), dtype=float)
    for a in range(1, s + 1):
        if a in set_xx:
            J = np.zeros((m, m), dtype=float)
            ii, jj = np.where(Uxx == a)
            J[ii[0], jj[0]] = 1.0
            J[jj[0], ii[0]] = 1.0
            A[:, a - 1] = (-invRxx @ J @ invRxx @ Rxy).ravel()
        if a in set_xy:
            ii = np.where(Uxy == a)[0]
            A[:, a - 1] = invRxx[:, ii[0]]
    vb = A @ V @ A.T

    coef = np.asarray(b, dtype=float).ravel()
    se = np.sqrt(np.diag(vb))
    with np.errstate(invalid="ignore", divide="ignore"):
        zval = coef / se
        pv = 2.0 * stats.norm.sf(np.abs(zval))
    crit = float(stats.norm.isf(alpha / 2.0))
    if cov:
        b_resc = coef * sdy / sdx
        intercept = 0.0 if means is None else float(
            np.asarray(means, dtype=float)[yj]
            - np.asarray(means, dtype=float)[xs] @ b_resc)
        coef_out = np.concatenate([[intercept], b_resc])
        vb_full = np.zeros((m + 1, m + 1), dtype=float)
        vb_full[1:, 1:] = vb
        if means is None:
            coef_out[0] = np.nan
            vb_full[0, :] = np.nan
            vb_full[:, 0] = np.nan
        sub = vb_full[1:, 1:]
        zval_out = np.concatenate([[np.nan], zval])
        pv_out = np.concatenate([[np.nan], pv])
        se_out = np.concatenate([[np.nan], se])
        names_out = ["intrcpt"] + var_names
        vb = vb_full
    else:
        coef_out, se_out, zval_out, pv_out = coef, se, zval, pv
        names_out = var_names
        sub = vb

    try:
        QM = float(coef.T @ np.linalg.solve(sub, coef))
    except np.linalg.LinAlgError:            # pragma: no cover
        QM = float("nan")
    return MatregResult(
        beta=[float(v) for v in coef_out],
        se=[float(v) for v in se_out],
        pval=[float(v) for v in pv_out],
        ci_lb=[float(v) for v in coef_out - crit * se_out],
        ci_ub=[float(v) for v in coef_out + crit * se_out],
        vb=np.asarray(vb, dtype=float),
        names=names_out,
        test="z",
        level=alpha,
        intercept=bool(cov),
        p=m,
        parms=m,
        R2=float(R2),
        zval=[float(v) for v in zval_out],
        QM=QM,
        QMdf=[int(m), None],
        QMp=float(stats.chi2.sf(QM, m)),
        k=int(m),
        yi=[float(v) for v in np.asarray(Rxy).ravel()],
        vi=[1.0] * m,
    )


# --------------------------------------------------------------------------- #
# profile.rma.uni
# --------------------------------------------------------------------------- #

@dataclass
class ProfileResult:
    """Result of :func:`profile_rma` (``metafor::profile.rma.uni``)."""

    tau2: list[float]
    ll: list[float]
    beta: list[float]
    ci_lb: float
    ci_ub: float
    maxll: float
    method: str
    k: int


def profile_rma(result, n_points=20, level=0.05):
    """``metafor::profile.rma.uni`` — profile log-likelihood of ``tau^2``.

    A :class:`~.selmodel.SelModelResult` is handled too (R's
    ``profile.rma.uni.selmodel``), in which case the profile is over the
    selection parameter ``delta``.

    Otherwise the (restricted) log-likelihood is evaluated on an evenly spaced
    grid that runs from the QP lower to the QP upper confidence bound of
    ``tau^2`` — exactly the grid metafor uses (``n_points = 20`` by default).
    At each grid point ``beta`` is re-estimated, so the curve is the profile
    (not the conditional) likelihood.

    Args:
        result: fitted :class:`RmaResult` or :class:`SelModelResult`.
        n_points: number of grid points.
        level: alpha for the QP bounds (0.05 → 95% bounds).

    Returns:
        :class:`ProfileResult` (or ``SelProfileResult`` for a selection model).
    """
    from .selmodel import SelModelResult, profile_selmodel

    if isinstance(result, SelModelResult):
        return profile_selmodel(result, n_points=n_points,
                                level=1.0 - level if level < 0.5 else level)

    from .pooling import confint_tau2
    from .rma_uni import RmaResult

    if not isinstance(result, RmaResult):
        raise TypeError("argument 'result' must be an RmaResult")
    yi = np.asarray(result.yi, dtype=float)
    vi = np.asarray(result.vi, dtype=float)
    X = np.asarray(result.X, dtype=float)
    k, p = X.shape
    method = result.method

    ci = confint_tau2(yi, vi, float(result.tau2), method=method, level=level, type="QP")
    lb = float(ci.tau2_lb)
    ub = float(ci.tau2_ub)
    grid = np.linspace(lb, ub, int(n_points))

    is_fe = method in ("FE", "EE", "CE")

    def _ll(t2):
        w = 1.0 / (vi + t2)
        WX = X * w[:, None]
        XtWX = X.T @ WX
        try:
            inv = np.linalg.inv(XtWX)
            logdet_xtwx = math.log(abs(np.linalg.det(XtWX)))
        except np.linalg.LinAlgError:
            inv = np.linalg.pinv(XtWX)
            logdet_xtwx = math.log(max(abs(np.linalg.det(inv)), 1e-300)) * -1.0
        beta = inv @ (X.T @ (w * yi))
        r = yi - X @ beta
        rss = float(r @ (w * r))
        logdet_m = float(np.sum(np.log(vi + t2)))
        if is_fe:
            return -0.5 * rss, beta
        if method == "ML":
            ll = -0.5 * k * math.log(2.0 * math.pi) - 0.5 * logdet_m - 0.5 * rss
        else:  # REML (metafor includes the +0.5*log|X'X| term)
            logdet_xtx = math.log(max(abs(np.linalg.det(X.T @ X)), 1e-300))
            ll = (-0.5 * (k - p) * math.log(2.0 * math.pi)
                  + 0.5 * logdet_xtx
                  - 0.5 * logdet_m - 0.5 * logdet_xtwx - 0.5 * rss)
        return ll, beta

    lls, betas = [], []
    for t2 in grid:
        ll, b = _ll(float(t2))
        lls.append(ll)
        betas.append(float(np.asarray(b).ravel()[0]))

    return ProfileResult(
        tau2=[float(v) for v in grid],
        ll=[float(v) for v in lls],
        beta=betas,
        ci_lb=lb, ci_ub=ub,
        maxll=float(np.max(lls)),
        method=method, k=k,
    )
