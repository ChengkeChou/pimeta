"""Sensitivity analysis & influence diagnostics.

Replicates ``metafor::influence.rma.uni`` and ``metafor::leave1out.rma.uni``:
leave-one-out analysis, externally studentized residuals, DFFITS, Cook's
distance, covariance ratio, leave-one-out tau^2 and QE, hat values, weights
and DFBETAS.

Reference: Viechtbauer & Cheung (2010), Res. Synth. Methods 1(2), 112-125.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
from scipy import stats

from .metareg import _dl_tau2, _i2_h2, _likelihood_tau2, _wls
from .rma_list import RowSlicable

Z_975 = stats.norm.ppf(0.975)


@dataclass
class InfluenceResult(RowSlicable):
    """Leave-one-out / influence diagnostics (metafor::influence)."""

    rstudent: list[float]        # externally standardized (deleted) residuals
    dffits: list[float]
    cook_d: list[float]
    cov_r: list[float]
    tau2_del: list[float]        # leave-one-out tau^2
    qe_del: list[float]          # leave-one-out residual heterogeneity QE
    hat: list[float]             # diagonal of the hat matrix
    weight: list[float]          # fitting weights (percent)
    dfbs: list[list[float]]      # (k, p) DFBETAS
    beta_del: list[list[float]]  # (k, p) leave-one-out coefficients
    se_del: list[list[float]]    # (k, p) leave-one-out coefficient SEs
    is_infl: list[bool]
    tau2: float                  # full-model tau^2
    QE: float                    # full-model QE
    k: int
    p: int
    names: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "rstudent": self.rstudent,
            "dffits": self.dffits,
            "cook_d": self.cook_d,
            "cov_r": self.cov_r,
            "tau2_del": self.tau2_del,
            "qe_del": self.qe_del,
            "hat": self.hat,
            "weight": self.weight,
            "dfbs": self.dfbs,
            "beta_del": self.beta_del,
            "se_del": self.se_del,
            "is_infl": self.is_infl,
            "tau2": self.tau2,
            "QE": self.QE,
            "k": self.k,
            "p": self.p,
            "names": self.names,
        }


@dataclass
class LeaveOneOutResult(RowSlicable):
    """Leave-one-out pooled estimates (metafor::leave1out)."""

    estimate: list[float]
    se: list[float]
    z: list[float]
    p: list[float]
    ci_lower: list[float]
    ci_upper: list[float]
    QE: list[float]
    QE_p: list[float]
    tau2: list[float]
    I2: list[float]
    H2: list[float]

    def to_dict(self) -> dict:
        return {
            "estimate": self.estimate,
            "se": self.se,
            "z": self.z,
            "p": self.p,
            "ci_lower": self.ci_lower,
            "ci_upper": self.ci_upper,
            "QE": self.QE,
            "QE_p": self.QE_p,
            "tau2": self.tau2,
            "I2": self.I2,
            "H2": self.H2,
        }


def _design_matrix(X, k):
    """Return (design_with_intercept, p, names)."""
    if X is None:
        return np.ones((k, 1)), 1, ["intrcpt"]
    X = np.asarray(X, dtype=float)
    if X.ndim == 1:
        X = X.reshape(-1, 1)
    Xd = np.column_stack([np.ones(k), X])
    return Xd, Xd.shape[1], None


def _fit(y, v, X, method, tau2_fixed):
    """Fit WLS meta-regression; return (tau2, beta, cov, qe)."""
    y = np.asarray(y, dtype=float).ravel()
    v = np.asarray(v, dtype=float).ravel()
    X = np.asarray(X, dtype=float)
    if tau2_fixed is not None:
        tau2 = float(tau2_fixed)
    else:
        mu = method.upper()
        if mu == "FE":
            tau2 = 0.0
        elif mu == "DL":
            tau2 = _dl_tau2(y, v, X)
        elif mu in ("ML", "REML"):
            tau2 = _likelihood_tau2(y, v, X, mu)
        else:
            raise ValueError("method must be FE/DL/ML/REML")
    w = 1.0 / (v + tau2)
    beta, cov, _ = _wls(X, w, y)
    # Cochran's QE uses fixed-effect weights (1/v), independent of tau^2.
    _, _, qe = _wls(X, 1.0 / v, y)
    return tau2, beta, cov, qe


def _s2w(X, w, y, beta, test):
    """Residual variance factor: 1 for z-test, RSS/(k-p) for knha."""
    if test == "z":
        return 1.0
    rss = float(np.sum(w * (y - X @ beta) ** 2))
    k, p = X.shape
    return rss / (k - p) if (k - p) > 0 else 0.0


def influence(
    y,
    v,
    X=None,
    X_names=None,
    method: str = "REML",
    tau2: float | None = None,
    test: str = "z",
    alpha: float = 0.05,
) -> InfluenceResult:
    """Leave-one-out influence diagnostics (metafor::influence).

    Args:
        y, v: analysis-scale effect and sampling variance.
        X: moderator design matrix (n, m) WITHOUT intercept, or None for an
            intercept-only model.
        method: tau^2 estimator ("FE", "DL", "ML" or "REML"; default REML).
        tau2: optional fixed tau^2 (when given, it is not re-estimated in the
            leave-one-out refits).
        test: "z" (default) or "knha".
    """
    y = np.asarray(y, dtype=float).ravel()
    v = np.asarray(v, dtype=float).ravel()
    k = len(y)
    Xd, p, _ = _design_matrix(X, k)

    names = ["intrcpt"] + (
        list(X_names) if X_names else [f"mod{i + 1}" for i in range(p - 1)]
    )

    # Full model.
    tau2_full, beta, vb, QE = _fit(y, v, Xd, method, tau2)
    w = 1.0 / (v + tau2_full)
    s2w_full = _s2w(Xd, w, y, beta, test)

    # Hat values and weights.
    XtW = Xd.T * w
    stXWX = np.linalg.inv(XtW @ Xd)
    H = Xd @ stXWX @ (Xd.T * w)
    hat = np.diag(H).copy()
    weight = w / w.sum() * 100.0
    pred_full = Xd @ beta
    svb = np.linalg.inv(vb)  # inverse var-cov of full model (for Cook's D)

    rstudent = np.full(k, np.nan)
    dffits = np.full(k, np.nan)
    cook_d = np.full(k, np.nan)
    cov_r = np.full(k, np.nan)
    tau2_del = np.full(k, np.nan)
    qe_del = np.full(k, np.nan)
    dfbs = np.full((k, p), np.nan)
    beta_del = np.full((k, p), np.nan)
    se_del = np.full((k, p), np.nan)
    delpred = np.full(k, np.nan)
    vdelpred = np.full(k, np.nan)

    for i in range(k):
        idx = [j for j in range(k) if j != i]
        t2_i, b_i, vb_i, qe_i = _fit(y[idx], v[idx], Xd[idx], method, tau2)
        tau2_del[i] = t2_i
        qe_del[i] = qe_i
        beta_del[i] = b_i
        se_del[i] = np.sqrt(np.diag(vb_i))

        Xi = Xd[i]
        delpred[i] = float(Xi @ b_i)
        vdelpred[i] = float(Xi @ vb_i @ Xi)

        # DFFITS (standard weights).
        dffits[i] = (pred_full[i] - delpred[i]) / math.sqrt(
            s2w_full * hat[i] * (t2_i + v[i])
        )

        # DFBETAS: full design matrix with tau2.del[i] plugged in.
        w_del = 1.0 / (v + t2_i)
        vb_del = np.linalg.inv((Xd.T * w_del) @ Xd)
        dfbs[i] = (beta - b_i) / np.sqrt(s2w_full * np.diag(vb_del))

        # Cook's distance (Mahalanobis of full vs deleted coefficients).
        dfb = beta - b_i
        cook_d[i] = float(dfb @ svb @ dfb)

        # Covariance ratio.
        cov_r[i] = float(np.linalg.det(vb_i) / np.linalg.det(vb))

    # Externally standardized (deleted) residuals.
    resid = y - delpred
    seresid = np.sqrt(v * s2w_full + vdelpred + tau2_del * s2w_full)
    rstudent = resid / seresid

    # Influential-case flags (metafor cut-offs).
    is_infl = (
        (np.abs(dffits) > 3.0 * math.sqrt(p / (k - p)))
        | (stats.chi2.cdf(cook_d, df=p) > 0.50)
        | (hat > 3.0 * p / k)
        | (np.abs(dfbs) > 1.0).any(axis=1)
    )

    return InfluenceResult(
        rstudent=[float(x) for x in rstudent],
        dffits=[float(x) for x in dffits],
        cook_d=[float(x) for x in cook_d],
        cov_r=[float(x) for x in cov_r],
        tau2_del=[float(x) for x in tau2_del],
        qe_del=[float(x) for x in qe_del],
        hat=[float(x) for x in hat],
        weight=[float(x) for x in weight],
        dfbs=[[float(x) for x in row] for row in dfbs],
        beta_del=[[float(x) for x in row] for row in beta_del],
        se_del=[[float(x) for x in row] for row in se_del],
        is_infl=[bool(x) for x in is_infl],
        tau2=float(tau2_full),
        QE=float(QE),
        k=k,
        p=p,
        names=names,
    )


def leave1out(
    y,
    v=None,
    method: str = "REML",
    tau2: float | None = None,
    alpha: float = 0.05,
    cluster=None,
) -> LeaveOneOutResult:
    """Leave-one-out analysis for an intercept-only model (metafor::leave1out).

    For each study, refits the model without it and records the pooled
    estimate, SE, z, p, CI, QE and tau^2 / I^2 / H^2.  An ``rma.mh`` /
    ``rma.peto`` result can be passed directly, in which case metafor's
    ``leave1out.rma.mh`` / ``.peto`` is used (``cluster`` collapses studies).
    """
    from .rma_peto_mh import is_mh_result, mh_leave1out

    if is_mh_result(y):
        return mh_leave1out(y, cluster=cluster)
    y = np.asarray(y, dtype=float).ravel()
    v = np.asarray(v, dtype=float).ravel()
    k = len(y)

    estimate = np.full(k, np.nan)
    se = np.full(k, np.nan)
    z = np.full(k, np.nan)
    pval = np.full(k, np.nan)
    ci_l = np.full(k, np.nan)
    ci_u = np.full(k, np.nan)
    qe = np.full(k, np.nan)
    qe_p = np.full(k, np.nan)
    tau2v = np.full(k, np.nan)
    i2 = np.full(k, np.nan)
    h2 = np.full(k, np.nan)

    X0 = np.ones((k, 1))
    for i in range(k):
        idx = [j for j in range(k) if j != i]
        t2_i, b_i, vb_i, qe_i = _fit(y[idx], v[idx], X0[idx], method, tau2)
        est = float(b_i[0])
        se_i = float(np.sqrt(vb_i[0, 0]))
        df = (k - 1) - 1  # k-1 studies, one intercept
        estimate[i] = est
        se[i] = se_i
        z[i] = est / se_i
        pval[i] = 2.0 * float(stats.norm.cdf(-abs(est / se_i)))
        ci_l[i] = est - Z_975 * se_i
        ci_u[i] = est + Z_975 * se_i
        qe[i] = qe_i
        qe_p[i] = float(stats.chi2.sf(qe_i, df))
        tau2v[i] = t2_i
        i2[i], h2[i] = _i2_h2(t2_i, v[idx])

    return LeaveOneOutResult(
        estimate=[float(x) for x in estimate],
        se=[float(x) for x in se],
        z=[float(x) for x in z],
        p=[float(x) for x in pval],
        ci_lower=[float(x) for x in ci_l],
        ci_upper=[float(x) for x in ci_u],
        QE=[float(x) for x in qe],
        QE_p=[float(x) for x in qe_p],
        tau2=[float(x) for x in tau2v],
        I2=[float(x) for x in i2],
        H2=[float(x) for x in h2],
    )


def hatvalues(y, v=None, X=None, method: str = "REML",
              tau2: float | None = None, type="diagonal"):
    """Diagonal of the hat matrix (metafor::hatvalues).

    An ``rma.mv`` fit can be passed instead of ``(y, v)``, in which case
    metafor's ``hatvalues.rma.mv`` is used and ``type="matrix"`` returns the
    full hat matrix.
    """
    from .rma_mv import RmaMvResult

    if isinstance(y, RmaMvResult):
        from .mv_diagnostics import mv_hatvalues

        return mv_hatvalues(y, type=type)
    y = np.asarray(y, dtype=float).ravel()
    v = np.asarray(v, dtype=float).ravel()
    k = len(y)
    Xd, p, _ = _design_matrix(X, k)
    tau2_full, _, _, _ = _fit(y, v, Xd, method, tau2)
    w = 1.0 / (v + tau2_full)
    stXWX = np.linalg.inv((Xd.T * w) @ Xd)
    H = Xd @ stXWX @ (Xd.T * w)
    return [float(x) for x in np.diag(H)]


def cooks_distance(
    y, v=None, X=None, method: str = "REML", tau2: float | None = None,
    cluster=None,
):
    """Cook's distances (metafor::cooks.distance).

    An ``rma.mv`` fit can be passed instead of ``(y, v)``, in which case
    metafor's ``cooks.distance.rma.mv`` is used (``cluster`` collapses the
    studies into groups).
    """
    from .rma_mv import RmaMvResult

    if isinstance(y, RmaMvResult):
        from .mv_diagnostics import mv_cooks_distance

        return mv_cooks_distance(y, cluster=cluster)
    inf = influence(y, v, X=X, method=method, tau2=tau2)
    return inf.cook_d


def rstudent(
    y, v=None, X=None, method: str = "REML", tau2: float | None = None,
    cluster=None,
):
    """Externally standardized (deleted) residuals (metafor::rstudent).

    Accepts ``(y, v, ...)``, an ``rma.mh`` / ``rma.peto`` result or an
    ``rma.mv`` fit; the latter two use metafor's own methods and return a dict
    (``resid``/``se``/``z``/``slab`` for mh/peto, ``delresid``/``sedelresid``/
    ``z``/``X2``/``slab`` for rma.mv).
    """
    from .rma_mv import RmaMvResult
    from .rma_peto_mh import is_mh_result, mh_rstudent

    if is_mh_result(y):
        return mh_rstudent(y)
    if isinstance(y, RmaMvResult):
        from .mv_diagnostics import mv_rstudent

        return mv_rstudent(y, cluster=cluster)
    inf = influence(y, v, X=X, method=method, tau2=tau2)
    return inf.rstudent


@dataclass
class BaujatResult:
    """Baujat plot coordinates (metafor::baujat.rma)."""

    x: list[float]   # squared Pearson residual: contribution to heterogeneity
    y: list[float]   # standardized influence on the pooled result
    k: int

    def to_dict(self) -> dict:
        return {"x": self.x, "y": self.y, "k": self.k}


def baujat(
    y, v, method: str = "REML", tau2: float | None = None
) -> BaujatResult:
    """Baujat diagnostic coordinates for an intercept-only model.

    ``x`` = ``(yi - mu)^2 / (tau^2 + vi)`` (squared Pearson residual);
    ``y`` = ``(mu_{-i} - mu)^2 / var(mu_{-i})`` (standardized influence of the
    leave-one-out refit on the pooled estimate), exactly as metafor::baujat.
    """
    y = np.asarray(y, dtype=float).ravel()
    v = np.asarray(v, dtype=float).ravel()
    k = len(y)
    X0 = np.ones((k, 1))

    tau2_full, beta_full, _, _ = _fit(y, v, X0, method, tau2)
    mu_full = float(beta_full[0])

    x = (y - mu_full) ** 2 / (tau2_full + v)

    yv = np.full(k, np.nan)
    for i in range(k):
        idx = [j for j in range(k) if j != i]
        _, b_i, vb_i, _ = _fit(y[idx], v[idx], X0[idx], method, tau2)
        delpred = float(b_i[0])
        vdelpred = float(vb_i[0, 0])
        yv[i] = (delpred - mu_full) ** 2 / vdelpred

    return BaujatResult(
        x=[float(xi) for xi in x],
        y=[float(xi) for xi in yv],
        k=k,
    )
