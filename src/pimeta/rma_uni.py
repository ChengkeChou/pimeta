"""rma.uni — a faithful transcription of ``metafor::rma(yi, vi, mods=...)``.

This module reproduces metafor's flagship univariate (random/fixed-effects)
meta-analysis model in pure NumPy/SciPy. It mirrors the full ``$`` field set of
an ``rma.uni`` object so callers can read ``b``, ``se``, ``zval``, ``pval``,
``ci.lb``/``ci.ub``, ``vb``, ``tau2``, ``I2``, ``H2``, ``R2``, ``vt``, ``QE``,
``QM`` etc. exactly as they would in R.

Supported tau-squared estimators (both intercept-only and, where metafor
defines one, the generalised moderator version):

- ``FE``     fixed effects (tau2 = 0)
- ``DL``     DerSimonian-Laird (generalised, Viechtbauer 2005 eq. 6)
- ``HE``     Hedges
- ``HS``     Hunter-Schmidt (metafor's 1/vi weighting)
- ``SJ``     Sidik-Jonkman
- ``PM``/``MP``  Paule-Mandel
- ``ML``/``REML`` maximum (restricted) likelihood via metafor's Fisher scoring
- ``EB``     empirical Bayes

Supported tests of the coefficients: ``z`` (Wald), ``t``, ``knha``
(Knapp-Hartung) and ``adhoc`` (Sidik-Jonkman ad-hoc).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
from scipy import optimize, stats

from .data_compat import FormulaSpec, design_matrix, resolve, resolve_mods
from .missing import complete_cases
from .pooling import (
    _metafor_fisher_scoring_tau2,
    confint_tau2,
    predict_rma,
)

if TYPE_CHECKING:                       # pragma: no cover - import cycle guard
    from .rma_ls import RmaLsResult

Z_975 = stats.norm.ppf(0.975)

_TAU2_MAX = 1e6


# --------------------------------------------------------------------------- #
# tau-squared estimation (generalised — arbitrary design matrix X)
# --------------------------------------------------------------------------- #
def _proj(X: np.ndarray, w: np.ndarray) -> np.ndarray:
    """``P = W - W X (X'W X)^-1 X' W`` with ``W = diag(w)``."""
    W = np.diag(w)
    stXWX = np.linalg.inv(X.T @ W @ X)
    return W - W @ X @ stXWX @ X.T @ W


def _tau2_he(y, v, X, ymc, k, p):
    """Hedges estimator (unweighted projection)."""
    P = np.eye(k) - X @ np.linalg.inv(X.T @ X) @ X.T
    rss = float(ymc @ P @ ymc)
    tr_pv = float(np.sum(np.diag(P) * v))
    return max(0.0, (rss - tr_pv) / (k - p))


def _tau2_eb(y, v, X, ymc, k, p):
    """Empirical Bayes via metafor's Fisher-scoring style loop."""
    tau2 = _tau2_he(y, v, X, ymc, k, p)
    threshold = 1e-5
    for _ in range(100):
        wi = 1.0 / (v + tau2)
        W = np.diag(wi)
        stXWX = np.linalg.inv(X.T @ W @ X)
        P = W - W @ X @ stXWX @ X.T @ W
        adj = (float(ymc @ P @ ymc) * k / (k - p) - k) / float(wi.sum())
        while tau2 + adj < 0.0:
            adj /= 2.0
        tau2_new = tau2 + adj
        change = abs(tau2_new - tau2)
        tau2 = tau2_new
        if change < threshold:
            break
    return max(0.0, float(tau2))


def _tau2_estimate(y, v, X, method):
    """Return tau^2 for any supported estimator (generalised in X)."""
    y = np.asarray(y, dtype=float).ravel()
    v = np.asarray(v, dtype=float).ravel()
    X = np.asarray(X, dtype=float)
    k = y.shape[0]
    p = X.shape[1]
    method = method.upper()
    ymc = y - y.mean()

    if method == "FE":
        return 0.0

    if method == "HE":
        return _tau2_he(y, v, X, ymc, k, p)

    if method == "HS":
        w = 1.0 / v
        P = _proj(X, w)
        rss = float(ymc @ P @ ymc)
        return max(0.0, (rss - k) / float(w.sum()))

    if method == "DL":
        w = 1.0 / v
        P = _proj(X, w)
        rss = float(ymc @ P @ ymc)
        tr_p = float(np.trace(P))
        return max(0.0, (rss - (k - p)) / tr_p)

    if method == "SJ":
        tau2_0 = float(np.var(ymc, ddof=0))  # sum((y-mean)^2)/k
        w = 1.0 / (v + tau2_0)
        P = _proj(X, w)
        rss = float(ymc @ P @ ymc)
        return max(0.0, tau2_0 * rss / (k - p))

    if method in ("PM", "MP"):
        target = float(k - p)

        def f(t2: float) -> float:
            w = 1.0 / (v + t2)
            P = _proj(X, w)
            return float(ymc @ P @ ymc) - target

        if f(0.0) <= 0.0:
            return 0.0
        hi = 1.0
        while f(hi) > 0.0 and hi < _TAU2_MAX:
            hi *= 10.0
        if f(hi) > 0.0:
            return hi
        return float(optimize.brentq(f, 0.0, hi))

    if method in ("ML", "REML"):
        return float(_metafor_fisher_scoring_tau2(y, v, X, method))

    if method == "EB":
        return _tau2_eb(y, v, X, ymc, k, p)

    raise ValueError(
        f"unsupported method: {method!r} "
        "(expected FE/DL/HE/HS/SJ/PM/MP/ML/REML/EB)"
    )


def _se_tau2(v, X, tau2, method, k, p):
    """Standard error of tau^2 (metafor's closed-form expressions)."""
    method = method.upper()
    if method in ("FE",):
        return None
    wi = 1.0 / (v + tau2)
    W = np.diag(wi)
    stXWX = np.linalg.inv(X.T @ W @ X)
    P = W - W @ X @ stXWX @ X.T @ W
    if method == "ML":
        return float(math.sqrt(2.0 / float(np.sum(wi ** 2))))
    if method == "REML":
        return float(math.sqrt(2.0 / float(np.sum(P * P))))
    if method == "DL":
        tr_p = float(np.trace(P))
        return float(math.sqrt(
            (1.0 / tr_p ** 2)
            * (2.0 * (k - p) + 4.0 * max(tau2, 0.0) * tr_p
               + 2.0 * max(tau2, 0.0) ** 2 * float(np.sum(P * P)))
        ))
    return None


# --------------------------------------------------------------------------- #
# result object
# --------------------------------------------------------------------------- #
@dataclass
class RmaResult:
    """Fitted ``rma.uni`` model, mirroring the R ``$`` field set."""

    b: list[float]
    se: list[float]
    zval: list[float]
    pval: list[float]
    ci_lb: list[float]
    ci_ub: list[float]
    vb: np.ndarray
    tau2: float
    se_tau2: float | None
    tau2_fix: bool
    I2: float
    H2: float
    R2: float | None
    vt: float
    QE: float
    QEp: float
    QM: float
    QMdf: list
    QMp: float
    k: int
    k_f: int
    k_eff: int
    k_all: int
    p: int
    p_eff: int
    parms: int
    int_only: bool
    int_incl: bool
    yi: list[float]
    vi: list[float]
    X: np.ndarray
    mods: np.ndarray | None
    weights: list[float]
    weights_pct: list[float]
    fitted: list[float]
    residuals: list[float]
    method: str
    test: str
    dfs: float | None
    ddf: float | None
    s2w: float
    measure: str
    slab: list[str]
    not_na: list[bool]
    level: float
    coefnames: list[str] | None = None
    #: 1-based indices of the coefficients the omnibus ``QM`` test covers
    #: (metafor's ``btt``).  Stored so ``robust()`` can reproduce the same
    #: test on the cluster-robust covariance instead of guessing the default.
    btt: list[int] | None = None

    def to_dict(self) -> dict:
        return {
            "b": self.b,
            "se": self.se,
            "zval": self.zval,
            "pval": self.pval,
            "ci_lb": self.ci_lb,
            "ci_ub": self.ci_ub,
            "vb": self.vb.tolist(),
            "tau2": self.tau2,
            "se_tau2": self.se_tau2,
            "tau2_fix": self.tau2_fix,
            "I2": self.I2,
            "H2": self.H2,
            "R2": self.R2,
            "vt": self.vt,
            "QE": self.QE,
            "QEp": self.QEp,
            "QM": self.QM,
            "QMdf": self.QMdf,
            "QMp": self.QMp,
            "k": self.k,
            "k_f": self.k_f,
            "k_eff": self.k_eff,
            "k_all": self.k_all,
            "p": self.p,
            "p_eff": self.p_eff,
            "parms": self.parms,
            "int_only": self.int_only,
            "int_incl": self.int_incl,
            "yi": self.yi,
            "vi": self.vi,
            "weights": self.weights,
            "weights_pct": self.weights_pct,
            "fitted": self.fitted,
            "residuals": self.residuals,
            "method": self.method,
            "test": self.test,
            "dfs": self.dfs,
            "ddf": self.ddf,
            "s2w": self.s2w,
            "measure": self.measure,
            "slab": self.slab,
        }

    def predict(self, level: float | None = None, transf=None, targs=None,
                newmods=None, **kw):
        """Prediction (+ CI/PI) — ``metafor::predict.rma``.

        Without ``newmods`` this returns the single prediction for the model's
        reference point (intercept-only models only).  With ``newmods`` it
        returns one prediction per row of the moderator matrix, exactly as
        ``predict(fit, newmods = ...)`` does: ``newmods`` must *not* include an
        intercept column (one is prepended).  ``level`` is the confidence
        level (0.95 -> 95%).
        """
        conf = self.level if level is None else level
        alpha = 1.0 - conf
        if newmods is None:
            if not self.int_only:
                raise ValueError(
                    "the model has moderators; pass 'newmods' to predict()"
                )
            return predict_rma(
                est=self.b[0],
                se=self.se[0],
                tau2=self.tau2,
                k=self.k,
                p=self.p,
                parms=self.parms,
                level=alpha,
                test=self.test,
                transf=transf,
                targs=targs,
                **kw,
            )

        new = np.asarray(newmods, dtype=float)
        if new.ndim == 1:
            # metafor: a plot vector of length p-1 is one new study, otherwise
            # a plain vector of values is one new study per element when the
            # model has a single moderator.
            if self.p - 1 == 1:
                new = new.reshape(-1, 1)
            elif new.size == self.p - 1:
                new = new.reshape(1, -1)
            else:
                raise ValueError(
                    f"'newmods' has {new.size} value(s); the model has "
                    f"{self.p - 1} moderator(s)"
                )
        if new.shape[1] != self.p - 1:
            raise ValueError(
                f"'newmods' has {new.shape[1]} column(s); the model has "
                f"{self.p - 1}"
            )
        Xnew = np.hstack([np.ones((new.shape[0], 1)), new])
        est = Xnew @ self.b
        se = np.sqrt(np.maximum(np.einsum("ij,jk,ik->i", Xnew, self.vb, Xnew), 0.0))
        out = [
            predict_rma(
                est=float(est[r]),
                se=float(se[r]),
                tau2=self.tau2,
                k=self.k,
                p=self.p,
                parms=self.parms,
                level=alpha,
                test=self.test,
                transf=transf,
                targs=targs,
                **kw,
            )
            for r in range(Xnew.shape[0])
        ]
        return out[0] if len(out) == 1 else out

    def confint(self, level: float | None = None, type: str = "QP"):
        """CI for tau^2 / tau / I^2 / H^2 (``metafor::confint.rma.uni``).

        Works with and without moderators.  ``type="HT"`` needs an
        intercept-only model, as in metafor.
        """
        conf = self.level if level is None else level
        alpha = 1.0 - conf
        return confint_tau2(
            self.yi,
            self.vi,
            self.tau2,
            method=self.method,
            level=alpha,
            type=type,
            X=None if self.int_only else self.X,
        )


# --------------------------------------------------------------------------- #
# main entry point
# --------------------------------------------------------------------------- #
def rma(
    yi,
    vi=None,
    sei=None,
    mods=None,
    data=None,
    slab=None,
    measure: str = "GEN",
    method: str = "REML",
    test: str = "z",
    level: float = 0.95,
    btt=None,
    digits: int = 4,
    scale=None,
    scale_intercept: bool = True,
    link: str = "log",
    optbeta: bool = False,
    alpha=None,
    warn_missing: bool = True,
) -> RmaResult | RmaLsResult:
    """Fit a univariate meta-analysis model (``metafor::rma``).

    Args:
        yi: analysis-scale effect sizes (e.g. log risk ratios).
        vi: sampling variances (alternative to ``sei``).
        sei: sampling standard errors (alternative to ``vi``).
        mods: moderator values — a 1-D vector for a single moderator or a
            2-D array for several, WITHOUT an intercept column (one is added).
        slab: study labels.
        measure: effect-size measure (recorded, does not change the scale).
        method: tau^2 estimator: FE/DL/HE/HS/SJ/PM/MP/ML/REML/EB (default REML).
        test: "z" (default), "t", "knha" or "adhoc".
        level: confidence level (0.95 -> 95% CI).  alpha = 1 - level.
        btt: 1-based coefficient indices to include in the QM omnibus test
            (default: the intercept for intercept-only models, otherwise all
            moderator coefficients).
        digits: printing precision (recorded only).
        scale, scale_intercept, link, optbeta, alpha: when ``scale`` is given
            a location-scale model is fitted instead (``rma(yi, vi, mods=~x,
            scale=~z)`` in R, class ``rma.ls``); the fit is delegated to
            :func:`pimeta.rma_ls` and an :class:`~pimeta.RmaLsResult` is
            returned.
        warn_missing: emit a :class:`~pimeta.missing.MissingDataWarning` naming
            the studies dropped because ``yi``, ``vi`` or a moderator is
            missing.  R drops them silently through ``na.action``; pass
            ``False`` to drop them silently here too.
    """
    if scale is not None:
        from .rma_ls import rma_ls

        return rma_ls(
            yi, vi=vi, sei=sei, mods=mods, scale=scale, data=data,
            slab=slab, measure=measure, method=method, test=test,
            level=level, scale_intercept=scale_intercept, link=link,
            optbeta=optbeta, alpha=alpha, btt=btt, warn_missing=warn_missing,
        )
    yi = resolve(yi, data, name="yi")
    if vi is None:
        if sei is None:
            raise ValueError("either 'vi' or 'sei' must be supplied")
        sei = resolve(sei, data, name="sei")
        vi = sei ** 2
    else:
        vi = resolve(vi, data, name="vi")
    if len(yi) != len(vi):
        raise ValueError("length of 'yi' and 'vi' must match")

    if slab is not None:
        slab = resolve(slab, data, name="slab", dtype=None)

    n_rows = len(yi)

    # Build the design matrix (intercept + optional moderators).
    mods_arr, add_intercept = resolve_mods(mods, data, n_rows)

    # A formula knows what its columns are called; carry those names through so
    # coefficients read ``dose``/``arm[B]`` instead of ``V1``/``V2``.
    coefnames: list[str] | None = None
    if data is not None and isinstance(mods, (str, FormulaSpec)) and mods_arr is not None:
        _, _term_names = design_matrix(mods, data, intercept=False)
        coefnames = ((["intrcpt"] if add_intercept else []) + list(_term_names))

    # Drop incomplete cases the way R's na.action does, but say so.
    checked: dict = {"yi": yi, "vi": vi}
    if mods_arr is not None:
        for j in range(mods_arr.shape[1]):
            checked[f"mods[{j}]"] = mods_arr[:, j]
    keep, _dropped = complete_cases(checked, context="studies", warn=warn_missing)
    if not keep.all():
        yi = yi[keep]
        vi = vi[keep]
        if mods_arr is not None:
            mods_arr = mods_arr[keep]
        if slab is not None and len(slab) == keep.size:
            slab = [s for s, ok in zip(slab, keep) if ok]

    k = len(yi)
    not_na = [True] * k

    if mods_arr is None:
        if not add_intercept:
            raise ValueError(
                "a no-intercept model needs at least one moderator")
        p_mods = 0
        X = np.ones((k, 1))
    else:
        p_mods = mods_arr.shape[1]
        X = (np.column_stack([np.ones(k), mods_arr]) if add_intercept
             else mods_arr)

    p = X.shape[1]
    int_only = p_mods == 0

    # A collinear design makes (X'WX) singular; say so plainly instead of
    # letting numpy raise "Singular matrix" from deep inside the estimator.
    if p > 1:
        rank = int(np.linalg.matrix_rank(X))
        if rank < p:
            if coefnames is not None:
                names = list(coefnames)
            elif add_intercept:
                names = ["intrcpt"] + [f"mods[{i}]" for i in range(1, p)]
            else:
                names = [f"mods[{i}]" for i in range(p)]
            raise ValueError(
                "the design matrix is not of full rank: "
                f"{p} columns but rank {rank} (moderators {names}).\n"
                "At least two moderators are collinear, or a dummy-coded factor "
                "is coded with all of its levels."
            )

    # --- tau^2 ------------------------------------------------------------ #
    method_key = method.upper()
    if method_key not in ("FE", "DL", "HE", "HS", "SJ", "PM", "MP", "ML", "REML", "EB"):
        raise ValueError(
            f"unsupported method: {method!r} "
            "(expected FE/DL/HE/HS/SJ/PM/MP/ML/REML/EB)"
        )
    tau2_fix = method_key == "FE"
    tau2 = _tau2_estimate(yi, vi, X, method_key)

    # --- weighted least squares ------------------------------------------ #
    wi = 1.0 / (vi + tau2) if not tau2_fix else 1.0 / vi
    W = np.diag(wi)
    stXWX = np.linalg.inv(X.T @ W @ X)
    b = stXWX @ X.T @ W @ yi
    vb = stXWX  # (X'W X)^-1

    # --- test / s2w scaling ---------------------------------------------- #
    test = test.lower()
    if test == "hksj":
        test = "knha"
    if test not in ("z", "t", "knha", "adhoc"):
        raise ValueError("test must be 'z', 't', 'knha' or 'adhoc'")

    s2w = 1.0
    if test in ("knha", "adhoc"):
        resid = yi - X @ b
        rss_knha = float(np.sum(wi * resid ** 2))
        if rss_knha <= np.finfo(float).eps:
            s2w = 0.0
        else:
            s2w = rss_knha / (k - p) if (k - p) > 0 else 0.0
        if test == "adhoc":
            s2w = max(s2w, 1.0)
    vb = s2w * vb

    se = np.sqrt(np.diag(vb))
    zval = b / se
    ddf = (k - p) if test in ("knha", "adhoc", "t") else None
    dfs = ddf

    if test in ("knha", "adhoc", "t"):
        pvals = 2.0 * stats.t.cdf(-np.abs(zval), df=ddf)
        crit = stats.t.ppf(1.0 - (1.0 - level) / 2.0, df=ddf)
    else:
        pvals = 2.0 * stats.norm.cdf(-np.abs(zval))
        crit = stats.norm.ppf(1.0 - (1.0 - level) / 2.0)

    ci_lb = b - crit * se
    ci_ub = b + crit * se

    # --- QM omnibus test of (a subset of) coefficients -------------------- #
    if btt is None:
        btt_idx = [0] if int_only else list(range(1, p))
    else:
        btt_idx = [int(i) - 1 for i in np.atleast_1d(btt)]  # 1-based -> 0-based
        btt_idx = [i for i in btt_idx if 0 <= i < p]
    m = len(btt_idx)
    b_tt = b[btt_idx]
    vb_tt = vb[np.ix_(btt_idx, btt_idx)]
    qm_wald = float(b_tt @ np.linalg.solve(vb_tt, b_tt)) if m > 0 else 0.0
    if test in ("knha", "adhoc", "t"):
        QM = qm_wald / m if m > 0 else 0.0
        QMdf = [m, ddf]
        QMp = float(stats.f.sf(QM, m, ddf)) if m > 0 and ddf and ddf > 0 else math.nan
    else:
        QM = qm_wald
        QMdf = [m, None]
        QMp = float(stats.chi2.sf(QM, m)) if m > 0 else math.nan

    # --- QE residual heterogeneity (fixed-effect weights) ----------------- #
    wi_fe = 1.0 / vi
    P_fe = _proj(X, wi_fe)
    ymc = yi - yi.mean()
    QE = max(0.0, float(ymc @ P_fe @ ymc))
    QE_df = k - p
    QEp = float(stats.chi2.sf(QE, QE_df)) if QE_df > 0 else math.nan

    # --- I^2 / H^2 / vt --------------------------------------------------- #
    tr_p_fe = float(np.trace(P_fe))
    if k > p and tr_p_fe > 0:
        vt = (k - p) / tr_p_fe
    else:
        vt = 0.0
    if tau2_fix:
        I2 = max(0.0, 100.0 * (QE - QE_df) / QE) if QE > 0 else 0.0
        H2 = QE / QE_df if QE_df > 0 else 1.0
    else:
        if vt > 0:
            I2 = 100.0 * tau2 / (vt + tau2)
            H2 = tau2 / vt + 1.0
        else:
            I2, H2 = 0.0, 1.0

    # --- pseudo R^2 ------------------------------------------------------- #
    R2 = None
    if not int_only and p_mods > 0:
        if tau2_fix:
            # metafor: 100 * adjusted R^2 of the weighted least-squares fit.
            b_w, _, _ = _wls(X, wi, yi)
            fitted = X @ b_w
            rss = float(np.sum(wi * (yi - fitted) ** 2))
            ybar_w = float(np.sum(wi * yi) / np.sum(wi))
            tss = float(np.sum(wi * (yi - ybar_w) ** 2))
            r2 = 1.0 - rss / tss if tss > 0 else 0.0
            adj_r2 = 1.0 - (1.0 - r2) * (k - 1) / (k - p) if k > p else 0.0
            R2 = max(0.0, 100.0 * adj_r2)
        elif tau2 > 0:
            X0 = np.ones((k, 1))
            tau2_0 = _tau2_estimate(yi, vi, X0, method_key)
            if tau2_0 > 0:
                R2 = max(0.0, 100.0 * (1.0 - tau2 / tau2_0))

    # --- derived quantities ----------------------------------------------- #
    fitted = X @ b
    residuals = yi - fitted
    sw = float(wi.sum())
    weights_pct = [100.0 * w / sw for w in wi]

    parms = p + (0 if tau2_fix else 1)

    if slab is None:
        slab_list = [f"Study {i + 1}" for i in range(k)]
    else:
        slab_list = list(slab)

    return RmaResult(
        b=[float(x) for x in b],
        se=[float(x) for x in se],
        zval=[float(x) for x in zval],
        pval=[float(x) for x in pvals],
        ci_lb=[float(x) for x in ci_lb],
        ci_ub=[float(x) for x in ci_ub],
        vb=vb,
        tau2=float(tau2),
        se_tau2=_se_tau2(vi, X, tau2, method_key, k, p),
        tau2_fix=tau2_fix,
        I2=I2,
        H2=H2,
        R2=R2,
        vt=vt,
        QE=QE,
        QEp=QEp,
        QM=QM,
        QMdf=QMdf,
        QMp=QMp,
        k=k,
        k_f=k,
        k_eff=k,
        k_all=k,
        p=p,
        p_eff=p,
        parms=parms,
        int_only=int_only,
        int_incl=True,
        yi=[float(x) for x in yi],
        vi=[float(x) for x in vi],
        X=X,
        mods=mods_arr,
        weights=[float(w) for w in wi],
        weights_pct=weights_pct,
        fitted=[float(x) for x in fitted],
        residuals=[float(x) for x in residuals],
        method=method_key,
        test=test,
        dfs=dfs,
        ddf=ddf,
        s2w=s2w,
        measure=measure,
        slab=slab_list,
        not_na=not_na,
        level=level,
        coefnames=coefnames,
        btt=[i + 1 for i in btt_idx],
    )


def _wls(X, w, y):
    """Weighted least squares: b, cov, rss (kept local for self-containment)."""
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float)
    w = np.asarray(w, dtype=float)
    XtW = X.T * w
    XWX = XtW @ X
    b = np.linalg.solve(XWX, XtW @ y)
    cov = np.linalg.inv(XWX)
    fitted = X @ b
    resid = y - fitted
    rss = float(np.sum(w * resid ** 2))
    return b, cov, rss
