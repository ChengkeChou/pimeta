"""rma.glmm — a faithful transcription of ``metafor::rma.glmm``.

Exact mixed-effects logistic/Poisson models for 2x2 binary data (OR/RR/RD)
and single-arm proportions (PLO/PR/PLN), matching lme4's ``glmer`` backend
(adaptive Gauss-Hermite quadrature, ``nAGQ=7``).

The AGQ objective replicates lme4's ``glmerAGQ`` machinery exactly:

* the deviance residual is the binomial deviance (link independent),
* the per-study random effect ``u`` is **profiled** by solving the penalised
  score ``g2 = 0`` (lme4's ``uOnly`` PIRLS), while ``(theta, beta, gamma)``
  remain free outer-loop variables,
* the correction term uses the Gauss-Hermite nodes/weights and the Laplace
  factor ``ldL2 = sum(log(1 + theta^2 ZWZ_j))`` with the GLM working weights
  ``W = n * (dmu/deta)^2 / (mu (1-mu))``,
* ``vcov = 2 * solve(H_dev)`` where ``H_dev`` is the Hessian of the AGQ
  deviance w.r.t. the full parameter vector.

Numeric engine: NumPy/SciPy only (no statsmodels / pymare).
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from scipy import optimize, stats
from scipy.special import expit, gammaln, xlogy

from .checks import attach_diagnostics
from .escalc import escalc

# --------------------------------------------------------------------------- #
# measure -> link mapping (metafor::rma.glmm)
# --------------------------------------------------------------------------- #
_MEASURE_LINK = {
    "OR": "logit",
    "PLO": "logit",
    "RR": "log",
    "PLN": "log",
    "RD": "identity",
    "PR": "identity",
}

_BINARY_GLMM = {"OR", "RR", "RD"}
_SINGLE_GLMM = {"PLO", "PR", "PLN"}

# Gauss-Hermite quadrature (7 nodes, scaled to standard normal).
_ZG, _WG = np.polynomial.hermite.hermgauss(7)
_GH_Z = _ZG * math.sqrt(2.0)
_GH_W = _WG / math.sqrt(math.pi)


# --------------------------------------------------------------------------- #
# link / score helpers
# --------------------------------------------------------------------------- #
def _mu_s(eta, link):
    """Scalar inverse link (clipped for numerical stability)."""
    eta = float(np.asarray(eta, dtype=float).ravel()[0])
    if link == "logit":
        return float(expit(min(max(eta, -40.0), 40.0)))
    if link == "log":
        return float(math.exp(min(max(eta, -700.0), 700.0)))
    return float(min(max(eta, 1e-10), 1.0 - 1e-10))  # identity


def _mu_vec(eta, link):
    """Vectorised inverse link."""
    eta = np.asarray(eta, dtype=float)
    if link == "logit":
        return expit(np.clip(eta, -40.0, 40.0))
    if link == "log":
        return np.exp(np.clip(eta, -700.0, 700.0))
    return np.clip(eta, 1e-10, 1.0 - 1e-10)


def _dmu_deta(mu, link):
    """d(mu)/d(eta)."""
    mu = np.asarray(mu, dtype=float)
    if link == "logit":
        return mu * (1.0 - mu)
    if link == "log":
        return mu
    return np.ones_like(mu)  # identity


def _score_s(s, n, mu, link):
    """d(logpmf)/d(eta) for a single binomial observation."""
    if link == "logit":
        return s - n * mu
    if link == "log":
        return (s - n * mu) / (1.0 - mu)
    return (s - n * mu) / (mu * (1.0 - mu))


def _score2_s(s, n, mu, link):
    """d^2(logpmf)/d(eta)^2 for a single binomial observation."""
    if link == "logit":
        return -n * mu * (1.0 - mu)
    if link == "log":
        return mu * (s - n) / ((1.0 - mu) ** 2)
    # identity
    return -(s * (1.0 - mu) ** 2 + (n - s) * mu ** 2) / (mu ** 2 * (1.0 - mu) ** 2)


def _binomial_loglik(s, n, mu):
    """Full binomial log-likelihood (with the combinatorial constant)."""
    s = np.asarray(s, dtype=float)
    n = np.asarray(n, dtype=float)
    mu = np.asarray(mu, dtype=float)
    return float(
        np.sum(
            gammaln(n + 1.0)
            - gammaln(s + 1.0)
            - gammaln(n - s + 1.0)
            + xlogy(s, mu)
            + xlogy(n - s, 1.0 - mu)
        )
    )


def _glm_binomial(s, n, Z, link, maxiter=200, tol=1e-13):
    """Binomial GLM via IRLS.

    Returns ``(beta, vcov, mu, eta, loglik)``.
    """
    s = np.asarray(s, dtype=float)
    n = np.asarray(n, dtype=float)
    Z = np.asarray(Z, dtype=float)
    y = s / n

    # sensible starting value
    y_init = (s + 0.5) / (n + 1.0)
    if link == "logit":
        eta_init = np.log(y_init / (1.0 - y_init))
    elif link == "log":
        eta_init = np.log(y_init)
    else:
        eta_init = y_init
    beta = np.linalg.lstsq(Z, eta_init, rcond=None)[0]

    for _ in range(maxiter):
        eta = Z @ beta
        mu = _mu_vec(eta, link)
        dmu = _dmu_deta(mu, link)
        var = np.maximum(mu * (1.0 - mu), 1e-15)
        w = n * dmu * dmu / var
        z = eta + (y - mu) / np.maximum(dmu, 1e-15)
        XtW = Z.T * w
        XtWX = XtW @ Z
        try:
            beta_new = np.linalg.solve(XtWX, XtW @ z)
        except np.linalg.LinAlgError:
            beta_new = np.linalg.lstsq(XtWX, XtW @ z, rcond=None)[0]
        if np.max(np.abs(beta_new - beta)) < tol:
            beta = beta_new
            break
        beta = beta_new

    eta = Z @ beta
    mu = _mu_vec(eta, link)
    dmu = _dmu_deta(mu, link)
    var = np.maximum(mu * (1.0 - mu), 1e-15)
    w = n * dmu * dmu / var
    XtWX = (Z * w[:, None]).T @ Z
    vcov = np.linalg.inv(XtWX)
    loglik = _binomial_loglik(s, n, mu)
    return beta, vcov, mu, eta, loglik


# --------------------------------------------------------------------------- #
# AGQ objective (two-arm)
# --------------------------------------------------------------------------- #
def _u_profile_twoarm(theta, beta, gamma_j, st, nt, sc, nc, link):
    """Solve g2=0 for u_j (per-study, gamma_j held fixed)."""
    th = abs(theta)
    if link != "identity":
        u = 0.0
        for _ in range(100):
            etat = beta + gamma_j + th * u * 0.5
            etac = gamma_j - th * u * 0.5
            mut = _mu_s(etat, link)
            muc = _mu_s(etac, link)
            fpt = _score_s(st, nt, mut, link)
            fpc = _score_s(sc, nc, muc, link)
            g2 = th * 0.5 * (fpt - fpc) - u
            J = -1.0 + th * th * 0.25 * (
                _score2_s(st, nt, mut, link) + _score2_s(sc, nc, muc, link)
            )
            step = -g2 / J
            u += step
            if abs(step) < 1e-14:
                break
        return u

    # identity link: g2(u) is monotone decreasing over the region where both
    # mu are in (0,1); bracket the unique root and solve with brentq (Newton
    # from u=0 starts outside the valid region when beta+gamma_j < 0).
    a = beta + gamma_j      # eta_t at u = 0
    c = gamma_j             # eta_c at u = 0
    lo = -2.0 * a / th                          # mut > 0
    hi = 2.0 * (1.0 - a) / th                   # mut < 1
    lo = max(lo, 2.0 * (c - 1.0) / th)          # muc < 1
    hi = min(hi, 2.0 * c / th)                  # muc > 0

    def g2(u):
        mut = _mu_s(a + th * u * 0.5, link)
        muc = _mu_s(c - th * u * 0.5, link)
        return th * 0.5 * (_score_s(st, nt, mut, link) - _score_s(sc, nc, muc, link)) - u

    if lo < hi:
        eps = 1e-9 * (hi - lo)
        lo_i = lo + eps
        hi_i = hi - eps
        glo = g2(lo_i)
        ghi = g2(hi_i)
        if glo > 0.0 and ghi < 0.0:
            try:
                return float(optimize.brentq(g2, lo_i, hi_i, xtol=1e-13, maxiter=200))
            except ValueError:
                pass

    # fallback: minimize the per-study penalized deviance directly.
    def h(u):
        mut = _mu_s(a + th * u * 0.5, link)
        muc = _mu_s(c - th * u * 0.5, link)
        dr = 2.0 * (xlogy(st, st / mut) + xlogy(nt - st, (nt - st) / (1.0 - mut)))
        dr += 2.0 * (xlogy(sc, sc / muc) + xlogy(nc - sc, (nc - sc) / (1.0 - muc)))
        return dr + u * u

    if lo < hi:
        res = optimize.minimize_scalar(
            h, bounds=(lo, hi), method="bounded",
            options={"xatol": 1e-13, "maxiter": 200},
        )
        return float(res.x)
    return 0.0


def _pirls_twoarm(th, lp0, Xf, Z, s, n, y, link, tolPwrss=1e-7, maxit=2000):
    """PIRLS joint solve of ``(beta, gamma, u)`` at fixed theta (lme4 PQL).

    Faithful to lme4's ``pwrssUpdate`` for the identity-link (RD) two-arm
    model: iterated penalised weighted least squares on the linear predictor
    with ``mu = linkinv(eta)`` and working weights
    ``W = n * muEta^2 / variance``, objective ``drsum + sqrL`` (the binomial
    deviance residual sum plus the squared random-effect penalty, *not* the
    working residual sum of squares).

    Returns ``(bg, u, drsum, sqrL, ldL2, mu)`` where ``bg`` stacks ``beta``
    then ``gamma`` and ``ldL2 = sum(log(1 + theta^2 ZWZ_j))`` is the Laplace
    log-determinant term ``2*log|L|`` for the diagonal ``Z'WZ``.
    """
    th = float(th)
    eta = np.asarray(lp0, dtype=float).copy()
    mu = _mu_vec(eta, link)
    k = Z.shape[1]
    ncol = Xf.shape[1]
    pwrss = np.inf
    bg = np.zeros(ncol)
    u = np.zeros(k)
    for _ in range(maxit):
        oldpwrss = pwrss
        dmu = _dmu_deta(mu, link)
        W = n * dmu * dmu / np.maximum(mu * (1.0 - mu), 1e-15)
        z = eta + (y - mu) / np.maximum(dmu, 1e-15)
        XtWX = Xf.T @ (Xf * W[:, None])
        XtWZ = Xf.T @ (Z * W[:, None])
        ZtWZ = Z.T @ (Z * W[:, None])
        A = np.block([[XtWX, th * XtWZ],
                      [th * XtWZ.T, th * th * ZtWZ + np.eye(k)]])
        rhs = np.concatenate([Xf.T @ (W * z), th * (Z.T @ (W * z))])
        sol = np.linalg.solve(A, rhs)
        bg = sol[:ncol]
        u = sol[ncol:]
        eta = Xf @ bg + th * (Z @ u)
        mu = _mu_vec(eta, link)
        devres = 2.0 * n * (xlogy(y, y / mu) + xlogy(1.0 - y, (1.0 - y) / (1.0 - mu)))
        pwrss = np.sum(devres) + np.sum(u * u)
        if abs((oldpwrss - pwrss) / max(1.0, abs(pwrss))) < tolPwrss:
            break
    dmu = _dmu_deta(mu, link)
    W = n * dmu * dmu / np.maximum(mu * (1.0 - mu), 1e-15)
    drsum = np.sum(2.0 * n * (xlogy(y, y / mu) + xlogy(1.0 - y, (1.0 - y) / (1.0 - mu))))
    sqrL = np.sum(u * u)
    ZWZ = 0.25 * (W[0::2] + W[1::2])
    ldL2 = np.sum(np.log(1.0 + th * th * ZWZ))
    return bg, u, drsum, sqrL, ldL2, mu


def _agq_twoarm(theta, beta, gamma, Xfit, s, n, group12, study, k, link,
                ret_drsum=False):
    """AGQ deviance for a two-arm model (u profiled via g2=0)."""
    th = abs(theta)
    u = np.empty(k)
    for j in range(k):
        u[j] = _u_profile_twoarm(
            th, beta, gamma[j], s[2 * j], n[2 * j], s[2 * j + 1], n[2 * j + 1], link
        )

    eta = Xfit @ beta + gamma[study] + th * u[study] * group12
    if link == "identity":
        lo = float(np.min(eta))
        hi = float(np.max(eta))
        if lo <= 0.0 or hi >= 1.0:
            viol = max(0.0, -lo) + max(0.0, hi - 1.0)
            pen = 1e6 * (1.0 + viol)
            return (pen, pen) if ret_drsum else pen
    mu = _mu_vec(eta, link)
    y = s / n
    devres = 2.0 * n * (xlogy(y, y / mu) + xlogy(1.0 - y, (1.0 - y) / (1.0 - mu)))
    dmu = _dmu_deta(mu, link)
    W = n * dmu * dmu / np.maximum(mu * (1.0 - mu), 1e-15)
    ZWZ = 0.25 * (W[0::2] + W[1::2])
    ldL2 = np.sum(np.log(1.0 + th * th * ZWZ))
    sqrL = np.sum(u * u)
    drsum = np.sum(devres)
    sd = 1.0 / np.sqrt(1.0 + th * th * ZWZ)

    def devc(uu):
        eta_u = Xfit @ beta + gamma[study] + th * uu[study] * group12
        mu_u = _mu_vec(eta_u, link)
        dr_u = 2.0 * n * (
            xlogy(y, y / mu_u) + xlogy(1.0 - y, (1.0 - y) / (1.0 - mu_u))
        )
        drsum_u = np.bincount(study, weights=dr_u, minlength=k)
        return drsum_u + uu * uu

    devc0 = devc(u)
    mult = np.zeros(k)
    for a in range(len(_GH_Z)):
        if _GH_Z[a] == 0.0:
            mult += _GH_W[a]
        else:
            uu = u + _GH_Z[a] * sd
            d = devc(uu) - devc0
            mult += np.exp(-0.5 * d + 0.5 * _GH_Z[a] * _GH_Z[a]) * _GH_W[a]

    obj = ldL2 + sqrL + drsum - 2.0 * np.sum(np.log(mult))
    if ret_drsum:
        return obj, drsum
    return obj


# --------------------------------------------------------------------------- #
# AGQ objective (single-arm)
# --------------------------------------------------------------------------- #
def _u_profile_single(theta, beta, xj, sj, nj, link):
    """Solve g2=0 for u_j (single-arm random intercept)."""
    u = 0.0
    for _ in range(100):
        eta = float(xj @ beta) + theta * u
        mu = _mu_s(eta, link)
        f1 = _score_s(sj, nj, mu, link)
        g2 = theta * f1 - u
        f2 = _score2_s(sj, nj, mu, link)
        J = theta * theta * f2 - 1.0
        step = -g2 / J
        u += step
        if abs(step) < 1e-14:
            break
    return u


def _agq_single(theta, beta, Xfit, s, n, k, link, ret_drsum=False):
    """AGQ deviance for a single-arm model (random intercept)."""
    th = abs(theta)
    u = np.empty(k)
    for j in range(k):
        u[j] = _u_profile_single(th, beta, Xfit[j], s[j], n[j], link)

    eta = Xfit @ beta + th * u
    if link == "identity":
        lo = float(np.min(eta))
        hi = float(np.max(eta))
        if lo <= 0.0 or hi >= 1.0:
            viol = max(0.0, -lo) + max(0.0, hi - 1.0)
            pen = 1e6 * (1.0 + viol)
            return (pen, pen) if ret_drsum else pen
    mu = _mu_vec(eta, link)
    y = s / n
    devres = 2.0 * n * (xlogy(y, y / mu) + xlogy(1.0 - y, (1.0 - y) / (1.0 - mu)))
    dmu = _dmu_deta(mu, link)
    W = n * dmu * dmu / np.maximum(mu * (1.0 - mu), 1e-15)
    ldL2 = np.sum(np.log(1.0 + th * th * W))
    sqrL = np.sum(u * u)
    drsum = np.sum(devres)
    sd = 1.0 / np.sqrt(1.0 + th * th * W)

    def devc(uu):
        eta_u = Xfit @ beta + th * uu
        mu_u = _mu_vec(eta_u, link)
        dr_u = 2.0 * n * (
            xlogy(y, y / mu_u) + xlogy(1.0 - y, (1.0 - y) / (1.0 - mu_u))
        )
        return dr_u + uu * uu

    devc0 = devc(u)
    mult = np.zeros(k)
    for a in range(len(_GH_Z)):
        if _GH_Z[a] == 0.0:
            mult += _GH_W[a]
        else:
            uu = u + _GH_Z[a] * sd
            d = devc(uu) - devc0
            mult += np.exp(-0.5 * d + 0.5 * _GH_Z[a] * _GH_Z[a]) * _GH_W[a]

    obj = ldL2 + sqrL + drsum - 2.0 * np.sum(np.log(mult))
    if ret_drsum:
        return obj, drsum
    return obj


# --------------------------------------------------------------------------- #
# heterogeneity statistics (escalc yi/vi based)
# --------------------------------------------------------------------------- #
def _proj_mat(X, w):
    W = np.diag(w)
    stXWX = np.linalg.inv(X.T @ W @ X)
    return W - W @ X @ stXWX @ X.T @ W


def _het_stats(yi, vi, X, tau2):
    yi = np.asarray(yi, dtype=float)
    vi = np.asarray(vi, dtype=float)
    X = np.asarray(X, dtype=float)
    k = yi.shape[0]
    p = X.shape[1]
    w = 1.0 / vi
    P = _proj_mat(X, w)
    tr_p = float(np.trace(P))
    vt = (k - p) / tr_p if (k > p and tr_p > 0) else 0.0
    if vt > 0 and tau2 > 0:
        I2 = 100.0 * tau2 / (vt + tau2)
        H2 = tau2 / vt + 1.0
    else:
        I2 = 0.0
        H2 = 1.0
    return vt, I2, H2


def _dl_tau2(yi, vi, X):
    yi = np.asarray(yi, dtype=float)
    vi = np.asarray(vi, dtype=float)
    X = np.asarray(X, dtype=float)
    k = yi.shape[0]
    p = X.shape[1]
    w = 1.0 / vi
    P = _proj_mat(X, w)
    ymc = yi - yi.mean()
    rss = float(ymc @ P @ ymc)
    tr_p = float(np.trace(P))
    return max(0.0, (rss - (k - p)) / tr_p)


def _hessian(fun, x, h=1e-4):
    """Finite-difference Hessian (4-point central)."""
    x = np.asarray(x, dtype=float)
    d = x.shape[0]
    H = np.zeros((d, d))
    for i in range(d):
        for j in range(i, d):
            xp = x.copy()
            xm = x.copy()
            xpp = x.copy()
            xmm = x.copy()
            xp[i] += h
            xp[j] += h
            xm[i] -= h
            xm[j] += h
            xpp[i] += h
            xpp[j] -= h
            xmm[i] -= h
            xmm[j] -= h
            Hij = (fun(xp) - fun(xpp) - fun(xm) + fun(xmm)) / (4.0 * h * h)
            H[i, j] = H[j, i] = Hij
    return H


def _vcov_rx(theta, beta, gamma, u, Xfit, S, group12, s, n, study, k, link):
    """RX-based vcov (lme4 fallback when the deviance Hessian is non-PD).

    Replicates ``crossprod(lme4::getME(res, "RX")^-1)`` at the conditional
    mode: the top-left ``p x p`` block of the inverse of the augmented
    penalised-weighted cross-product ``[[X'WX, X'WZ], [Z'WX, Z'WZ + Lambda]]``.
    """
    th = abs(theta)
    Xf = np.hstack([Xfit, S])
    Z = np.zeros((2 * k, k))
    for j in range(k):
        rows = np.where(study == j)[0]
        Z[rows, j] = group12[rows]
    eta = Xfit @ beta + gamma[study] + th * u[study] * group12
    mu = _mu_vec(eta, link)
    dmu = _dmu_deta(mu, link)
    w = n * dmu * dmu / np.maximum(mu * (1.0 - mu), 1e-15)
    XtWX = Xf.T @ (Xf * w[:, None])
    XtWZ = Xf.T @ (Z * w[:, None])
    ZtWZ = Z.T @ (Z * w[:, None])
    lam = np.eye(k) / (th * th)
    A = np.block([[XtWX, XtWZ], [XtWZ.T, ZtWZ + lam]])
    try:
        Ai = np.linalg.inv(A)
    except np.linalg.LinAlgError:
        Ai = np.linalg.pinv(A)
    p = Xfit.shape[1]
    return Ai[:p, :p]


def _is_pd(A):
    """True if ``A`` is finite and positive definite."""
    A = np.asarray(A, dtype=float)
    if A.size == 0 or np.any(~np.isfinite(A)):
        return False
    try:
        return bool(np.all(np.linalg.eigvalsh((A + A.T) / 2.0) > 1e-12))
    except np.linalg.LinAlgError:
        return False


def _vcov_from_hessian_or_rx(H, fun, x, p, theta, beta, gamma, Xfit, S, group12,
                             s, n, study, k, link, two_arm):
    """Fixed-effect vcov: ``2*solve(H)``, falling back to the RX vcov when
    the deviance Hessian is singular/non-PD (lme4's behaviour for, e.g., the
    RD identity-link model)."""
    if _is_pd(H):
        try:
            vcov_all = 2.0 * np.linalg.inv(H)
            if _is_pd(vcov_all):
                return vcov_all[1 : 1 + p, 1 : 1 + p].copy()
        except np.linalg.LinAlgError:
            pass
    if two_arm and gamma is not None:
        u = np.empty(k)
        for j in range(k):
            u[j] = _u_profile_twoarm(
                theta, beta, gamma[j], s[2 * j], n[2 * j], s[2 * j + 1], n[2 * j + 1], link
            )
        return _vcov_rx(theta, beta, gamma, u, Xfit, S, group12, s, n, study, k, link)
    # single-arm fallback: pseudoinverse of the (scaled) Hessian.
    vcov_all = 2.0 * np.linalg.pinv(H)
    return vcov_all[1 : 1 + p, 1 : 1 + p].copy()


def _fit_stats(ll, dev, parms, k_eff):
    n_eff = max(k_eff, parms + 2)
    aic = -2.0 * ll + 2.0 * parms
    bic = -2.0 * ll + parms * math.log(k_eff)
    aicc = -2.0 * ll + 2.0 * parms * n_eff / (n_eff - parms - 1.0)
    return {
        "ll.ML": ll,
        "dev.ML": dev,
        "AIC.ML": aic,
        "BIC.ML": bic,
        "AICc.ML": aicc,
    }


# --------------------------------------------------------------------------- #
# result container
# --------------------------------------------------------------------------- #
@dataclass
@attach_diagnostics
class RmaGlmmResult:
    """Fitted ``rma.glmm`` model, mirroring the R ``$`` field set."""

    b: list[float]
    beta: list[float]
    se: list[float]
    zval: list[float]
    pval: list[float]
    ci_lb: list[float]
    ci_ub: list[float]
    vb: np.ndarray
    tau2: float
    se_tau2: float | None
    sigma2: float
    rho: float
    I2: float
    H2: float
    vt: float
    QE_Wld: float
    QEp_Wld: float
    QE_LRT: float
    QEp_LRT: float
    QE_df: int
    QM: float
    QMdf: list
    QMp: float
    k: int
    k_f: int
    k_yi: int
    k_eff: int
    k_all: int
    p: int
    p_eff: int
    parms: int
    int_only: bool
    yi: list[float]
    vi: list[float]
    X: np.ndarray
    fit_stats: dict
    measure: str
    method: str
    model: str
    test: str
    digits: int
    level: float
    slab: list[str]

    def to_dict(self) -> dict:
        return {
            "b": self.b,
            "beta": self.beta,
            "se": self.se,
            "zval": self.zval,
            "pval": self.pval,
            "ci_lb": self.ci_lb,
            "ci_ub": self.ci_ub,
            "vb": self.vb.tolist(),
            "tau2": self.tau2,
            "se_tau2": self.se_tau2,
            "sigma2": self.sigma2,
            "rho": self.rho,
            "I2": self.I2,
            "H2": self.H2,
            "vt": self.vt,
            "QE.Wld": self.QE_Wld,
            "QEp.Wld": self.QEp_Wld,
            "QE.LRT": self.QE_LRT,
            "QEp.LRT": self.QEp_LRT,
            "QE.df": self.QE_df,
            "QM": self.QM,
            "QMdf": self.QMdf,
            "QMp": self.QMp,
            "k": self.k,
            "k.f": self.k_f,
            "k.yi": self.k_yi,
            "k.eff": self.k_eff,
            "k.all": self.k_all,
            "p": self.p,
            "p.eff": self.p_eff,
            "parms": self.parms,
            "int.only": self.int_only,
            "yi": self.yi,
            "vi": self.vi,
            "X": self.X.tolist(),
            "fit.stats": self.fit_stats,
            "measure": self.measure,
            "method": self.method,
            "model": self.model,
            "test": self.test,
            "digits": self.digits,
            "level": self.level,
            "slab": self.slab,
        }


# --------------------------------------------------------------------------- #
# main entry point
# --------------------------------------------------------------------------- #
def rma_glmm(
    measure: str,
    ai=None,
    bi=None,
    ci=None,
    di=None,
    xi=None,
    ni=None,
    mods=None,
    data=None,
    slab=None,
    method: str = "ML",
    test: str = "z",
    level: float = 0.95,
    btt=None,
    digits: int = 4,
    nAGQ: int = 7,
    add: float = 0.5,
    to: str = "only0",
    drop00: bool = True,
    verbose: bool = False,
    control=None,
) -> RmaGlmmResult:
    """Fit a generalised linear mixed-effects meta-analysis model.

    ``measure`` is one of ``OR``/``RR``/``RD`` (two-arm; pass ``ai/bi/ci/di``)
    or ``PLO``/``PR``/``PLN`` (single-arm; pass ``xi/ni``).  ``method`` is
    ``"ML"`` (random effects via AGQ) or ``"FE"`` (fixed effects).
    """
    measure = measure.upper()
    if measure not in _MEASURE_LINK:
        raise ValueError(f"unsupported measure: {measure!r}")
    link = _MEASURE_LINK[measure]
    method = method.upper()
    if method not in ("ML", "FE"):
        raise ValueError("method must be 'ML' or 'FE'")

    two_arm = measure in _BINARY_GLMM

    # metafor's rma.glmm() dispatches on the measure, so a mismatched pair of
    # arguments produced a bare ``TypeError: 'NoneType' object is not
    # iterable``.  Name the arguments the chosen measure actually needs.
    if two_arm:
        if any(v is None for v in (ai, bi, ci, di)):
            raise ValueError(
                f"measure {measure!r} is a two-arm measure: 'ai', 'bi', 'ci' "
                f"and 'di' are all required (single-arm measures are "
                f"PLO, PR and PLN, which take 'xi'/'ni')")
    else:
        if xi is None or ni is None:
            raise ValueError(
                f"measure {measure!r} is a single-arm measure: 'xi' and 'ni' "
                f"are both required (two-arm measures are OR, RR and RD, "
                f"which take 'ai'/'bi'/'ci'/'di')")

    # ---- raw data -------------------------------------------------------- #
    if two_arm:
        ai = np.asarray([float(x) for x in ai])
        bi = np.asarray([float(x) for x in bi])
        ci = np.asarray([float(x) for x in ci])
        di = np.asarray([float(x) for x in di])
        k = len(ai)
        s = np.empty(2 * k)
        n = np.empty(2 * k)
        s[0::2] = ai
        s[1::2] = ci
        n[0::2] = ai + bi
        n[1::2] = ci + di
        group1 = np.tile([1.0, 0.0], k)
        group12 = np.tile([0.5, -0.5], k)
        study = np.repeat(np.arange(k), 2)
    else:
        xi = np.asarray([float(x) for x in xi])
        ni = np.asarray([float(x) for x in ni])
        k = len(xi)
        s = xi.copy()
        n = ni.copy()
        group1 = np.ones(k)
        group12 = np.ones(k)
        study = np.arange(k)

    # ---- design matrix --------------------------------------------------- #
    if mods is None:
        X = np.ones((k, 1))
        p = 1
    else:
        X = np.asarray(mods, dtype=float)
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        X = np.column_stack([np.ones(k), X])
        p = X.shape[1]
    int_only = p == 1

    if two_arm:
        Xfit = X[study] * group1[:, None]  # (2k, p)
    else:
        Xfit = X  # (k, p)

    # ---- escalc yi/vi for heterogeneity stats ---------------------------- #
    if two_arm:
        esc = escalc(measure, ai=ai, bi=bi, ci=ci, di=di, add=add, drop00=drop00)
    else:
        esc = escalc(measure, xi=xi, ni=ni)
    yi = np.asarray(esc.yi)
    vi = np.asarray(esc.vi)

    # ---- fixed-effects model --------------------------------------------- #
    if two_arm:
        S = np.eye(k)[study]  # (2k, k) study dummies
        Z_FE = np.hstack([Xfit, S])
    else:
        Z_FE = Xfit
    beta_fe, vcov_fe, _, _, ll_fe = _glm_binomial(s, n, Z_FE, link)

    # ---- saturated model (for QE.Wld / QE.LRT) --------------------------- #
    if two_arm:
        inter = np.zeros((2 * k, k - 1))
        for j in range(k - 1):
            inter[:, j] = ((group1 == 1.0) & (study == j)).astype(float)
        Z_QE = np.hstack([Xfit, S, inter])
    else:
        S = np.eye(k)
        Z_QE = np.hstack([Xfit, S[:, :-1]])
    beta_qe, vcov_qe, _, _, _ = _glm_binomial(s, n, Z_QE, link)
    ll_qe = _binomial_loglik(s, n, s / n)

    # b2.QE / vb2.QE are the study-heterogeneity contrast parameters.
    idx_start = p + k if two_arm else p
    b2_qe = beta_qe[idx_start:]
    vb2_qe = vcov_qe[idx_start:, idx_start:]

    qe_wld = float(b2_qe @ np.linalg.solve(vb2_qe, b2_qe))
    qe_lrt = -2.0 * (ll_fe - ll_qe)
    qe_df = k - p

    # ---- ML model -------------------------------------------------------- #
    tau2 = 0.0
    drsum_ml = qe_lrt  # deviance of the fixed-effects model
    vb = vcov_fe[:p, :p].copy()
    beta = beta_fe[:p].copy()

    if method == "ML":
        if two_arm:
            gamma0 = beta_fe[p:].copy()
            if link == "identity":
                # RD (identity link): lme4's PQL backend — profile out
                # (beta, gamma, u) by PIRLS at each theta, optimise theta in
                # one dimension.  The Laplace deviance objective (lme4's
                # ``glmResp::Laplace``) is ``-2*logLik + sqrL + ldL2``, where
                # the -2*logLik part is the binomial AIC (``-2*logLik``), *not*
                # the deviance residual sum (that is only reported later as
                # ``deviance(res.ML)``).
                Xf = np.hstack([Xfit, S])
                Z = np.zeros((2 * k, k))
                for j in range(k):
                    rows = np.where(study == j)[0]
                    Z[rows, j] = group12[rows]
                y = s / n

                def pirls(th, lp0):
                    return _pirls_twoarm(th, lp0, Xf, Z, s, n, y, link, 1e-7)

                # cold start: theta=1, eta=0 -> converged linear predictor lp0
                bg1, u1, _, _, _, _ = pirls(1.0, np.zeros(2 * k))
                lp0 = Xf @ bg1 + 1.0 * (Z @ u1)

                def devfun(th):
                    _, _, _, sqrL, ldL2, mu = pirls(th, lp0)
                    return -2.0 * _binomial_loglik(s, n, mu) + sqrL + ldL2

                res = optimize.minimize_scalar(
                    devfun, bounds=(0.0, 5.0), method="bounded",
                    options={"xatol": 1e-12, "maxiter": 2000},
                )
                theta_star = float(res.x)
                bg_star, u_star, drsum_ml, _, _, _ = pirls(theta_star, lp0)
                beta_star = bg_star[:p]
                gamma_star = bg_star[p:]
                tau2 = float(theta_star ** 2)
                vb = _vcov_rx(
                    theta_star, beta_star, gamma_star, u_star,
                    Xfit, S, group12, s, n, study, k, link,
                )
                beta = beta_star.copy()
            else:
                dl = _dl_tau2(yi, vi, X)
                theta0 = float(np.clip(math.sqrt(max(dl, 1e-6)), 1e-3, 10.0))

                def fun(x):
                    return _agq_twoarm(
                        x[0], x[1 : 1 + p], x[1 + p :], Xfit, s, n, group12, study, k, link
                    )

                x0 = np.concatenate([[theta0], beta.copy(), gamma0])
                bounds = [(1e-6, None)] + [(None, None)] * (p + k)
                res = optimize.minimize(
                    fun, x0, method="L-BFGS-B", bounds=bounds,
                    options={"ftol": 1e-13, "gtol": 1e-8, "maxiter": 5000, "maxfun": 200000},
                )
                theta_star = abs(res.x[0])
                beta_star = res.x[1 : 1 + p]
                tau2 = float(theta_star ** 2)
                obj_star, drsum_ml = _agq_twoarm(
                    theta_star, beta_star, res.x[1 + p :], Xfit, s, n, group12, study, k,
                    link, ret_drsum=True,
                )
                H = _hessian(fun, res.x)
                vb = _vcov_from_hessian_or_rx(
                    H, fun, res.x, p, theta_star, beta_star, res.x[1 + p :],
                    Xfit, S, group12, s, n, study, k, link, two_arm=True,
                )
                beta = beta_star.copy()
        else:
            dl = _dl_tau2(yi, vi, X)
            theta0 = float(np.clip(math.sqrt(max(dl, 1e-6)), 1e-3, 10.0))

            def fun(x):
                return _agq_single(x[0], x[1 : 1 + p], Xfit, s, n, k, link)

            x0 = np.concatenate([[theta0], beta.copy()])
            if link == "identity":
                res = optimize.minimize(
                    fun, x0, method="Powell",
                    options={"xtol": 1e-10, "ftol": 1e-13, "maxiter": 100000, "maxfev": 1000000},
                )
            else:
                bounds = [(1e-6, None)] + [(None, None)] * p
                res = optimize.minimize(
                    fun, x0, method="L-BFGS-B", bounds=bounds,
                    options={"ftol": 1e-13, "gtol": 1e-8, "maxiter": 5000, "maxfun": 200000},
                )
            theta_star = abs(res.x[0])
            beta_star = res.x[1 : 1 + p]
            tau2 = float(theta_star ** 2)
            obj_star, drsum_ml = _agq_single(
                theta_star, beta_star, Xfit, s, n, k, link, ret_drsum=True,
            )
            H = _hessian(fun, res.x)
            vb = _vcov_from_hessian_or_rx(
                H, fun, res.x, p, theta_star, beta_star, None,
                Xfit, S, None, s, n, None, k, link, two_arm=False,
            )
            beta = beta_star.copy()

    # ---- post-fit stats -------------------------------------------------- #
    se = np.sqrt(np.diag(vb))
    zval = beta / se
    pvals = 2.0 * stats.norm.cdf(-np.abs(zval))
    crit = stats.norm.ppf(1.0 - (1.0 - level) / 2.0)
    ci_lb = beta - crit * se
    ci_ub = beta + crit * se

    # QM omnibus test
    if btt is None:
        btt_idx = list(range(p))
    else:
        btt_idx = [int(i) - 1 for i in np.atleast_1d(btt)]
        btt_idx = [i for i in btt_idx if 0 <= i < p]
    m = len(btt_idx)
    b_tt = beta[btt_idx]
    vb_tt = vb[np.ix_(btt_idx, btt_idx)]
    qm = float(b_tt @ np.linalg.solve(vb_tt, b_tt)) if m > 0 else 0.0
    qm_df = [m, None]
    qm_p = float(stats.chi2.sf(qm, m)) if m > 0 else math.nan

    vt, I2, H2 = _het_stats(yi, vi, X, tau2)

    # fit statistics
    ll_ml = ll_qe - drsum_ml / 2.0
    if method == "ML":
        fit_parms = p + k + 1 if two_arm else p + 1
    else:
        fit_parms = p + k if two_arm else p
    k_eff = 2 * k if two_arm else k
    fit_stats = _fit_stats(ll_ml, drsum_ml, fit_parms, k_eff)

    # parms / p.eff / k.eff bookkeeping
    p_eff = p + k if two_arm else p
    parms = fit_parms

    if slab is None:
        slab_list = [f"Study {i + 1}" for i in range(k)]
    else:
        slab_list = list(slab)

    qep_wld = float(stats.chi2.sf(qe_wld, qe_df)) if qe_df > 0 else math.nan
    qep_lrt = float(stats.chi2.sf(qe_lrt, qe_df)) if qe_df > 0 else math.nan

    b_list = [float(x) for x in beta]
    return RmaGlmmResult(
        b=b_list,
        beta=b_list,
        se=[float(x) for x in se],
        zval=[float(x) for x in zval],
        pval=[float(x) for x in pvals],
        ci_lb=[float(x) for x in ci_lb],
        ci_ub=[float(x) for x in ci_ub],
        vb=vb,
        tau2=float(tau2),
        se_tau2=None,
        sigma2=float(tau2),
        rho=0.0,
        I2=I2,
        H2=H2,
        vt=vt,
        QE_Wld=qe_wld,
        QEp_Wld=qep_wld,
        QE_LRT=qe_lrt,
        QEp_LRT=qep_lrt,
        QE_df=qe_df,
        QM=qm,
        QMdf=qm_df,
        QMp=qm_p,
        k=k,
        k_f=k,
        k_yi=k,
        k_eff=k_eff,
        k_all=k_eff,
        p=p,
        p_eff=p_eff,
        parms=parms,
        int_only=int_only,
        yi=[float(x) for x in yi],
        vi=[float(x) for x in vi],
        X=X,
        fit_stats=fit_stats,
        measure=measure,
        method=method,
        model="UM.FS",
        test=test,
        digits=digits,
        level=level,
        slab=slab_list,
    )
