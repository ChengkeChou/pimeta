"""Jeffreys-penalised (Firth) logistic regression and meta's dispersion factor.

``meta::metabin(method = "LRP")`` and ``netmeta::netmetabin(method = "LRP")``
both fit ``glm(..., method = brglm2::brglmFit, type = "MPL_Jeffreys")``, i.e. they
maximise the log-likelihood penalised by half the log-determinant of the Fisher
information::

    l*(b) = l(b) + 0.5 * log|I(b)|,     I(b) = X' diag(n_i p_i (1 - p_i)) X

For a canonical link this coincides with Firth's (1993) bias-reduced estimator,
so the fit can be obtained by Newton's method on the penalised score.  With
``w_i = n_i p_i (1 - p_i)``, ``hhat_i = x_i' I^-1 x_i`` and the leverage
``h_i = w_i hhat_i``::

    u_i  = (y_i - n_i p_i) + h_i * (1/2 - p_i)        # Firth's modified score
    grad = X' u

    a_i  =  (1 - 2 p_i) w_i                           # scale of dI/db_j
    b_i  =  (1 - 6 p_i + 6 p_i^2) w_i                 # scale of d2I/db_j db_k

    G_j  = dI/db_j      = X' diag(a o x_j) X
    E_jk = d2I/db_j db_k = X' diag(b o x_j o x_k) X

    H_jk = -I_jk + 0.5 tr(I^-1 E_jk) - 0.5 tr(I^-1 G_j I^-1 G_k)

``brglm2`` solves the same estimating equation numerically; its ``vcov`` is
``I^-1`` evaluated at the estimate, which is what :func:`firth_logit` returns.
"""

from __future__ import annotations

import math

import numpy as np


def _expit(eta: np.ndarray) -> np.ndarray:
    """Overflow-free logistic function."""
    out = np.empty_like(eta, dtype=float)
    pos = eta >= 0
    out[pos] = 1.0 / (1.0 + np.exp(-eta[pos]))
    e = np.exp(eta[~pos])
    out[~pos] = e / (1.0 + e)
    return out


def _fisher(X: np.ndarray, w: np.ndarray) -> np.ndarray:
    return X.T @ (w[:, None] * X)


def firth_logit(X, successes, totals, maxit: int = 300,
                tol: float = 1e-12) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Jeffreys-penalised logistic regression (``brglmFit``, ``MPL_Jeffreys``).

    Args:
        X: design matrix (``n x p``), intercept included.
        successes: number of events per row.
        totals: number of trials per row (``prior.weights`` in R).

    Returns:
        ``(beta, vcov, mu)`` — the penalised MLE, the inverse Fisher information
        at that estimate (R's ``vcov(res.lrp)``), and the fitted probabilities.
    """
    X = np.asarray(X, dtype=float)
    y = np.asarray(successes, dtype=float).reshape(-1)
    n = np.asarray(totals, dtype=float).reshape(-1)
    p = X.shape[1]

    # Warm start from the ordinary logistic MLE, like brglm2 (which starts from
    # glm.fit).  Keep it small: with complete separation the unpenalised MLE
    # runs off to infinity and the penalty is what makes the problem well posed.
    beta = np.zeros(p)
    for _ in range(25):
        mu0 = _expit(X @ beta)
        w0 = n * mu0 * (1.0 - mu0)
        try:
            step = np.linalg.solve(_fisher(X, w0), X.T @ (y - n * mu0))
        except np.linalg.LinAlgError:
            break
        if not np.all(np.isfinite(step)):
            break
        beta = beta + step
        if np.max(np.abs(step)) < 1e-8:
            break
    if not np.all(np.isfinite(beta)) or np.max(np.abs(beta), initial=0.0) > 30.0:
        beta = np.zeros(p)

    mu = _expit(X @ beta)
    for _ in range(maxit):
        mu = _expit(X @ beta)
        w = n * mu * (1.0 - mu)
        info = _fisher(X, w)
        try:
            info_inv = np.linalg.inv(info)
        except np.linalg.LinAlgError:
            info_inv = np.linalg.pinv(info)
        hhat = np.einsum("ij,jk,ik->i", X, info_inv, X)
        u = (y - n * mu) + w * hhat * (0.5 - mu)
        grad = X.T @ u
        if np.max(np.abs(grad)) < tol:
            break

        a = (1.0 - 2.0 * mu) * w
        b = (1.0 - 6.0 * mu + 6.0 * mu * mu) * w
        hess = -info + 0.5 * _fisher(X, b * hhat)
        g_mats = [info_inv @ _fisher(X, a * X[:, j]) for j in range(p)]
        for j in range(p):
            for k in range(j, p):
                corr = 0.5 * float(np.trace(g_mats[j] @ g_mats[k]))
                hess[j, k] -= corr
                hess[k, j] = hess[j, k]

        try:
            step = np.linalg.solve(hess, -grad)
        except np.linalg.LinAlgError:
            step = np.linalg.lstsq(hess, -grad, rcond=None)[0]
        if not np.all(np.isfinite(step)):
            raise RuntimeError("Firth logistic regression diverged")
        # A full Newton step can overshoot where the penalty is flat; halve it
        # until the penalised log-likelihood does not decrease.
        scale = 1.0
        ll0 = _firth_loglik(X, y, n, beta)
        for _ in range(60):
            cand = beta + scale * step
            if _firth_loglik(X, y, n, cand) >= ll0 - 1e-12:
                break
            scale *= 0.5
        beta = beta + scale * step
        if np.max(np.abs(scale * step)) < tol:
            break

    mu = _expit(X @ beta)
    w = n * mu * (1.0 - mu)
    vcov = np.linalg.inv(_fisher(X, w))
    return beta, vcov, mu


def _firth_loglik(X, y, n, beta) -> float:
    """Penalised log-likelihood ``l(b) + 0.5 log|I(b)|``."""
    mu = _expit(X @ beta)
    with np.errstate(divide="ignore", invalid="ignore"):
        ll = float(np.sum(y * np.log(mu) + (n - y) * np.log1p(-mu)))
    info = _fisher(X, n * mu * (1.0 - mu))
    sign, logdet = np.linalg.slogdet(info)
    if sign <= 0 or not np.isfinite(logdet):
        return -math.inf
    return ll + 0.5 * float(logdet)


def dispersion_phi(successes, totals, mu, n_params: int) -> float:
    """``meta:::phi`` — the c-hat dispersion factor used for LRP standard errors.

    ``(sum of squared Pearson residuals / df) / (1 + mean((1-2p)/(n p (1-p))))``
    floored at 1, exactly as ``meta`` computes it.  Returns 1 when the
    dispersion is not estimable (no residual degrees of freedom).
    """
    y = np.asarray(successes, dtype=float).reshape(-1)
    n = np.asarray(totals, dtype=float).reshape(-1)
    mu = np.asarray(mu, dtype=float).reshape(-1)
    var = n * mu * (1.0 - mu)
    df = y.size - int(n_params)
    if df <= 0 or np.any(var <= 0):
        return 1.0
    chisq = float(np.sum((y - n * mu) ** 2 / var))
    s_bar = float(np.mean((1.0 - 2.0 * mu) / var))
    denom = 1.0 + s_bar
    if not np.isfinite(denom) or denom <= 0:
        return 1.0
    return max(chisq / df / denom, 1.0)


def treatment_design(trt_idx, study_idx, n_trts: int, n_studies: int) -> np.ndarray:
    """``~ 1 + as.factor(treat) + as.factor(studlab)`` with treatment 0 as ref.

    ``trt_idx`` / ``study_idx`` are 0-based integer codes (the factors are coded
    in the order R's ``factor()`` would sort them).
    """
    n = len(trt_idx)
    cols = [np.ones((n, 1))]
    if n_trts > 1:
        d = np.zeros((n, n_trts - 1))
        for i, t in enumerate(trt_idx):
            if t > 0:
                d[i, t - 1] = 1.0
        cols.append(d)
    if n_studies > 1:
        s = np.zeros((n, n_studies - 1))
        for i, g in enumerate(study_idx):
            if g > 0:
                s[i, g - 1] = 1.0
        cols.append(s)
    return np.hstack(cols)


def n_treatment_cols(n_trts: int) -> int:
    """Number of ``as.factor(treat)`` columns (1..``n_trts`` - 1)."""
    return max(n_trts - 1, 0)
