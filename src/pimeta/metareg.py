"""Meta-regression (metafor::rma with mods / meta::metareg).

Fits ``y_i = X_i b + u_i + e_i`` by weighted least squares, estimating
tau^2 with DerSimonian-Laird, ML or REML (matching metafor). Reports the
omnibus moderator test (QM), residual heterogeneity (QE), I^2, H^2 and the
pseudo-R^2 (proportional reduction in tau^2 relative to an intercept-only
model).

All formulas follow Viechtbauer (2005, J. Stat. Softw.), the reference for
metafor::rma.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
from scipy import stats

from .pooling import _metafor_fisher_scoring_tau2

Z_975 = stats.norm.ppf(0.975)


@dataclass
class MetaregResult:
    """Meta-regression coefficients + heterogeneity / model diagnostics."""

    b: list[float]
    se: list[float]
    z: list[float]
    p: list[float]
    ci_lower: list[float]
    ci_upper: list[float]
    names: list[str] = field(default_factory=list)
    tau2: float = 0.0
    method: str = "REML"
    k: int = 0
    p_mods: int = 0                 # number of moderators (excl. intercept)
    QM: float = 0.0                 # omnibus test of moderators
    QM_df: int = 0
    QM_p: float = math.nan
    QE: float = 0.0                 # residual heterogeneity
    QE_df: int = 0
    QE_p: float = math.nan
    I2: float = 0.0
    H2: float = 0.0
    R2: float = 0.0                 # pseudo-R^2 vs intercept-only tau^2

    def to_dict(self) -> dict:
        return {
            "b": self.b,
            "se": self.se,
            "z": self.z,
            "p": self.p,
            "ci_lower": self.ci_lower,
            "ci_upper": self.ci_upper,
            "names": self.names,
            "tau2": self.tau2,
            "method": self.method,
            "k": self.k,
            "QM": self.QM,
            "QM_df": self.QM_df,
            "QM_p": self.QM_p,
            "QE": self.QE,
            "QE_df": self.QE_df,
            "QE_p": self.QE_p,
            "I2": self.I2,
            "H2": self.H2,
            "R2": self.R2,
        }


def _wls(X, w, y):
    """Weighted least squares: b, cov, fitted, weighted residual SS."""
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float)
    w = np.asarray(w, dtype=float)
    XtW = X.T * w
    XWX = XtW @ X
    XWy = XtW @ y
    b = np.linalg.solve(XWX, XWy)
    cov = np.linalg.inv(XWX)
    fitted = X @ b
    resid = y - fitted
    rss = float(np.sum(w * resid ** 2))
    return b, cov, rss


def _i2_h2(tau2, v, X=None):
    """Higgins-Thompson I^2 and H^2 (metafor's i2def=1 definition).

    The typical within-study variance is ``vt = (k-p) / tr(P)`` with
    ``P = W - W X (X'WX)^-1 X'W`` and ``W = diag(1/v)``. For an
    intercept-only model this reduces to
    ``(k-1) sum(w) / (sum(w)^2 - sum(w)^2)``.
    """
    v = np.asarray(v, dtype=float)
    k = len(v)
    if k <= 1:
        return 0.0, 1.0
    if X is None:
        X = np.ones((k, 1))
    X = np.asarray(X, dtype=float)
    p = X.shape[1]
    w = 1.0 / v
    W = np.diag(w)
    XtW = X.T @ W
    stXWX = np.linalg.inv(XtW @ X)
    P = W - W @ X @ stXWX @ XtW
    trP = float(np.trace(P))
    vt = (k - p) / trP if trP > 0 else 0.0
    if vt <= 0:
        return 0.0, 1.0
    i2 = 100.0 * tau2 / (tau2 + vt)
    h2 = tau2 / vt + 1.0
    return i2, h2


def _dl_tau2(y, v, X):
    """DerSimonian-Laird tau^2 for meta-regression (Viechtbauer 2005, eq. 6)."""
    k = len(y)
    p = X.shape[1]
    w = 1.0 / np.asarray(v, dtype=float)
    _, _, qe = _wls(X, w, y)
    XtW2X = (X.T * w) @ (X * w[:, None])
    XWX = (X.T * w) @ X
    denom = float(np.sum(w) - np.trace(np.linalg.solve(XWX, XtW2X)))
    tau2 = max(0.0, (qe - (k - p)) / denom)
    return tau2


def _likelihood_tau2(y, v, X, method):
    """ML/REML tau^2 via metafor's exact Fisher-scoring iteration.

    A generic optimizer converges to the *exact* profile-likelihood optimum,
    but ``metafor::rma`` stops its Fisher scoring once ``change < 1e-5``, so
    the estimate can differ from that optimum by ~1e-6 in tau^2. To reproduce
    R exactly we run the same loop (Hedges start, ``P = W - WX(X'WX)^-1 X'W``
    updated each step, step-halving at the tau^2 >= 0 boundary, and the
    ``ll0`` check). See :func:`.pooling._metafor_fisher_scoring_tau2`.
    """
    X = np.asarray(X, dtype=float)
    return _metafor_fisher_scoring_tau2(y, v, X, method.upper())


def metareg(
    y,
    v,
    X,
    X_names=None,
    method: str = "REML",
    alpha: float = 0.05,
    test: str = "z",
) -> MetaregResult:
    """Fit a meta-regression (mirrors ``metafor::rma(yi, vi, mods=X)``).

    Args:
        y, v: analysis-scale effect and sampling variance (log scale for ratio
            measures).
        X: design matrix (n, p) WITHOUT an intercept column (one is added
            automatically), or (n, 1) single-moderator array.
        X_names: moderator names (len p).
        method: "FE" (fixed), "DL", "ML" or "REML" (default REML, metafor
            default).
        test: "z" (default), "t", "knha" (Knapp-Hartung / HKSJ) or "adhoc"
            (Sidik-Jonkman ad-hoc). Non-z tests use a t distribution with
            ``k-p`` df; "knha"/"adhoc" additionally scale the variance by
            ``s2w = RSS.knha/(k-p)`` (metafor's scalar implementation).
    """
    y = np.asarray(y, dtype=float).ravel()
    v = np.asarray(v, dtype=float).ravel()
    X = np.asarray(X, dtype=float)
    if X.ndim == 1:
        X = X.reshape(-1, 1)
    k = len(y)
    p_mods = X.shape[1]

    # Design matrix with intercept.
    Xd = np.column_stack([np.ones(k), X])
    p = Xd.shape[1]

    names = list(X_names) if X_names else [f"mod{i + 1}" for i in range(p_mods)]
    all_names = ["intrcpt"] + names

    if method.upper() == "FE":
        tau2, used = 0.0, "FE"
    elif method.upper() == "DL":
        tau2, used = _dl_tau2(y, v, Xd), "DL"
    elif method.upper() in ("ML", "REML"):
        tau2, used = _likelihood_tau2(y, v, Xd, method.upper()), method.upper()
    else:
        raise ValueError("method must be FE/DL/ML/REML")

    w = 1.0 / (v + tau2)
    b, vb, _ = _wls(Xd, w, y)

    test = test.lower()
    if test == "hksj":
        test = "knha"
    if test not in ("z", "t", "knha", "adhoc"):
        raise ValueError("test must be 'z', 't', 'knha' or 'adhoc'")

    # Knapp-Hartung / Sidik-Jonkman scalar scaling of the variance matrix.
    if test in ("knha", "adhoc"):
        resid = y - Xd @ b
        RSS_knha = float(np.sum(w * resid ** 2))
        s2w = RSS_knha / (k - p) if (k - p) > 0 and RSS_knha > np.finfo(float).eps else 0.0
        if test == "adhoc":
            s2w = max(s2w, 1.0)
        vb = s2w * vb

    se = np.sqrt(np.diag(vb))
    z = b / se
    ddf = k - p

    if test in ("knha", "adhoc", "t"):
        pvals = 2.0 * stats.t.cdf(-np.abs(z), df=ddf)
        crit = stats.t.ppf(1.0 - alpha / 2.0, df=ddf)
    else:
        pvals = 2.0 * stats.norm.cdf(-np.abs(z))
        crit = Z_975

    ci_l = b - crit * se
    ci_u = b + crit * se

    # Omnibus moderator test QM (Wald test on the moderator coefficients).
    if p_mods > 0:
        b_m = b[1:]
        cov_m = vb[1:, 1:]
        QM = float(b_m @ np.linalg.solve(cov_m, b_m))
    else:
        QM = 0.0
    QM_df = p_mods
    if test in ("knha", "adhoc", "t"):
        QM_stat = QM / QM_df if QM_df > 0 else QM
        QM_p = float(stats.f.sf(QM_stat, QM_df, ddf)) if QM_df > 0 else math.nan
    else:
        QM_p = float(stats.chi2.sf(QM, QM_df)) if QM_df > 0 else math.nan

    # Cochran's QE: fixed-effect weights (1/v), independent of tau^2.
    _, _, qe = _wls(Xd, 1.0 / v, y)
    QE = qe
    QE_df = k - p
    QE_p = float(stats.chi2.sf(QE, QE_df)) if QE_df > 0 else math.nan

    if used == "FE":
        I2 = max(0.0, 100.0 * (QE - QE_df) / QE) if QE > 0 else 0.0
        H2 = QE / QE_df if QE_df > 0 else 1.0
    else:
        I2, H2 = _i2_h2(tau2, v, Xd)

    # Pseudo-R^2: reduction in tau^2 vs intercept-only model (same method).
    R2 = 0.0
    if tau2 > 0 and p_mods > 0:
        X0 = np.ones((k, 1))
        if used == "DL":
            tau2_0 = _dl_tau2(y, v, X0)
        elif used in ("ML", "REML"):
            tau2_0 = _likelihood_tau2(y, v, X0, used)
        else:
            tau2_0 = 0.0
        if tau2_0 > 0:
            R2 = max(0.0, 1.0 - tau2 / tau2_0)

    return MetaregResult(
        b=[float(x) for x in b],
        se=[float(x) for x in se],
        z=[float(x) for x in z],
        p=[float(x) for x in pvals],
        ci_lower=[float(x) for x in ci_l],
        ci_upper=[float(x) for x in ci_u],
        names=all_names,
        tau2=tau2,
        method=used,
        k=k,
        p_mods=p_mods,
        QM=QM,
        QM_df=QM_df,
        QM_p=QM_p,
        QE=QE,
        QE_df=QE_df,
        QE_p=QE_p,
        I2=I2,
        H2=H2,
        R2=R2,
    )
