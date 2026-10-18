"""Additional metafor utilities: simulation, V-matrix construction, hettest, matreg.

Implements ``metafor::simulate.rma``, ``metafor::vcalc``, ``metafor::hettest``
and ``metafor::matreg``.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from scipy import stats

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
# vcalc
# --------------------------------------------------------------------------- #

def vcalc(vi, cluster=None, obs=None, rho=0.5, phi=0.0):
    """``metafor::vcalc`` — build a block-diagonal sampling-variance matrix.

    Within a cluster, the covariance between two effect sizes is
    ``rho * sqrt(vi_i * vi_j)``; between clusters it is zero.

    Args:
        vi: sampling variances (length ``k``).
        cluster: cluster identifiers (length ``k``); ``None`` puts everything
            in one cluster.
        obs: observation identifiers used to order the within-cluster blocks
            (optional; defaults to the input order).
        rho: within-cluster correlation.
        phi: optional additive heterogeneity factor (``phi * sqrt(vi_i*vi_j)``
            used by some ``vcalc`` variants); default 0.

    Returns:
        ``(k, k)`` float array.
    """
    vi = np.asarray(vi, dtype=float).ravel()
    k = vi.size
    if cluster is None:
        cl = np.zeros(k, dtype=int)
    else:
        cl = np.asarray(cluster).ravel()
        if cl.size != k:
            raise ValueError("length of 'cluster' does not match 'vi'")
    sd = np.sqrt(vi)
    V = np.diag(vi).astype(float)
    for c in np.unique(cl):
        idx = np.where(cl == c)[0]
        if idx.size < 2:
            continue
        if obs is not None:
            o = np.asarray(obs).ravel()[idx]
            idx = idx[np.argsort(o, kind="stable")]
        for a in range(idx.size):
            for b in range(a + 1, idx.size):
                cov = (rho + phi) * sd[idx[a]] * sd[idx[b]]
                V[idx[a], idx[b]] = cov
                V[idx[b], idx[a]] = cov
    return V


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

@dataclass
class MatregResult:
    """Result of :func:`matreg` (``metafor::matreg``)."""

    beta: list[float]
    se: list[float]
    zval: list[float]
    pval: list[float]
    ci_lb: list[float]
    ci_ub: list[float]
    vb: np.ndarray
    QE: float
    QEp: float
    QM: float
    QMp: float
    k: int
    yi: list[float]
    vi: list[float]


def matreg(y, x, R, n, measure="ZCOR", level=0.95):
    """``metafor::matreg`` — meta-regression from a correlation matrix.

    Pools the correlations between the outcome variable ``y`` and each of the
    predictor variables ``x`` (after Fisher's z transform when
    ``measure="ZCOR"``).

    Args:
        y: 1-based index of the outcome variable in ``R``.
        x: 1-based index (or sequence) of the predictor variables in ``R``.
        R: symmetric correlation matrix.
        n: sample size(s); a scalar or one value per correlation.
        measure: ``"ZCOR"`` (Fisher's z) or ``"COR"`` (raw correlations).
        level: confidence level.

    Returns:
        :class:`MatregResult`.
    """
    from .rma_uni import rma

    R = np.asarray(R, dtype=float)
    if R.ndim != 2 or R.shape[0] != R.shape[1]:
        raise ValueError("argument 'R' must be a square matrix")
    xs = list(np.atleast_1d(x))
    y0 = int(y) - 1
    yi = np.array([R[y0, int(j) - 1] for j in xs], dtype=float)
    nn = np.broadcast_to(np.asarray(n, dtype=float), yi.shape).copy()
    if measure == "ZCOR":
        yi = np.arctanh(np.clip(yi, -1 + 1e-12, 1 - 1e-12))
        vi = 1.0 / (nn - 3.0)
    elif measure == "COR":
        vi = (1.0 - yi ** 2) ** 2 / (nn - 1.0)
    else:
        raise ValueError("argument 'measure' must be 'ZCOR' or 'COR'")

    r = rma(yi, vi=vi, method="REML" if False else "FE")
    beta = np.asarray(r.b, dtype=float)
    se = np.asarray(r.se, dtype=float)
    crit = stats.norm.isf((1.0 - level) / 2.0)
    zval = beta / se
    return MatregResult(
        beta=[float(v) for v in beta],
        se=[float(v) for v in se],
        zval=[float(v) for v in zval],
        pval=[float(v) for v in 2.0 * stats.norm.sf(np.abs(zval))],
        ci_lb=[float(v) for v in beta - crit * se],
        ci_ub=[float(v) for v in beta + crit * se],
        vb=np.asarray(r.vb, dtype=float),
        QE=float(r.QE), QEp=float(r.QEp),
        QM=float(r.QM), QMp=float(r.QMp),
        k=int(r.k), yi=[float(v) for v in yi], vi=[float(v) for v in vi],
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

    Evaluates the (restricted) log-likelihood on an evenly spaced grid that runs
    from the QP lower to the QP upper confidence bound of ``tau^2`` — exactly the
    grid metafor uses (``n_points = 20`` by default). At each grid point ``beta``
    is re-estimated, so the curve is the profile (not the conditional)
    likelihood.

    Args:
        result: fitted :class:`RmaResult`.
        n_points: number of grid points.
        level: alpha for the QP bounds (0.05 → 95% bounds).

    Returns:
        :class:`ProfileResult`.
    """
    from .rma_uni import RmaResult
    from .pooling import confint_tau2

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
