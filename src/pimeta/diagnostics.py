"""Meta-analysis diagnostics — metafor parity.

Implements ``metafor::deltamethod``, ``metafor::tes`` (test of excess
significance), ``metafor::vif`` (generalized variance-inflation factors) and
``metafor::permutest`` (permutation test for the fixed effects).

All routines are pure NumPy/SciPy transcriptions; numerical derivatives replace
the ``calculus`` package that metafor uses for :func:`deltamethod`.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from itertools import combinations

import numpy as np
from scipy import integrate, stats

__all__ = [
    "DeltaMethodResult", "deltamethod",
    "TesResult", "tes",
    "VifResult", "vif",
    "PermutestResult", "permutest",
]

_EPS = np.finfo(float).eps
_QUAD_TOL = float(_EPS ** 0.25)


# --------------------------------------------------------------------------- #
# small numeric helpers
# --------------------------------------------------------------------------- #

def _cov2cor(v):
    """R's ``cov2cor``."""
    v = np.asarray(v, dtype=float)
    d = np.sqrt(np.diag(v))
    return v / np.outer(d, d)


def _num_grad(fun, x, h=None, order=1):
    """Central-difference gradient of a vector-valued ``fun`` at ``x``.

    Returns an array of shape ``(q, p)`` where ``q = len(fun(x))``.
    """
    x = np.asarray(x, dtype=float)
    p = x.size
    f0 = np.atleast_1d(fun(x))
    q = f0.size
    if h is None:
        h = _EPS ** (1.0 / 3.0) * np.maximum(np.abs(x), 1e-8)
    g = np.empty((q, p))
    for j in range(p):
        xp = x.copy(); xp[j] += h[j]
        xm = x.copy(); xm[j] -= h[j]
        g[:, j] = (np.atleast_1d(fun(xp)) - np.atleast_1d(fun(xm))) / (2.0 * h[j])
    return g


def _num_hess(fun, x, h=None):
    """Central-difference Hessian tensor of a vector-valued ``fun``.

    Returns an array of shape ``(q, p, p)``.
    """
    x = np.asarray(x, dtype=float)
    p = x.size
    f0 = np.atleast_1d(fun(x))
    q = f0.size
    if h is None:
        h = _EPS ** 0.25 * np.maximum(np.abs(x), 1e-8)
    H = np.empty((q, p, p))
    for j in range(p):
        for k in range(j, p):
            if j == k:
                xp = x.copy(); xp[j] += h[j]
                xm = x.copy(); xm[j] -= h[j]
                val = (np.atleast_1d(fun(xp)) - 2.0 * f0 + np.atleast_1d(fun(xm))) / h[j] ** 2
            else:
                xpp = x.copy(); xpp[j] += h[j]; xpp[k] += h[k]
                xpm = x.copy(); xpm[j] += h[j]; xpm[k] -= h[k]
                xmp = x.copy(); xmp[j] -= h[j]; xmp[k] += h[k]
                xmm = x.copy(); xmm[j] -= h[j]; xmm[k] -= h[k]
                val = (np.atleast_1d(fun(xpp)) - np.atleast_1d(fun(xpm))
                       - np.atleast_1d(fun(xmp)) + np.atleast_1d(fun(xmm))) / (4.0 * h[j] * h[k])
            H[:, j, k] = val
            H[:, k, j] = val
    return H


def _num_third(fun, x, i, j, k, h=None):
    """Third derivative ``d^3 f / dx_i dx_j dx_k`` (central differences)."""
    x = np.asarray(x, dtype=float)
    if h is None:
        h = _EPS ** 0.25 * np.maximum(np.abs(x), 1e-8)
    hi, hj, hk = h[i], h[j], h[k]

    def f(step):
        xx = x.copy()
        for idx, s in zip((i, j, k), step):
            xx[idx] += s
        return np.atleast_1d(fun(xx))

    total = np.zeros_like(np.atleast_1d(fun(x)))
    for si in (-1, 1):
        for sj in (-1, 1):
            for sk in (-1, 1):
                total = total + si * sj * sk * f((si * hi, sj * hj, sk * hk))
    return total / (8.0 * hi * hj * hk)


# --------------------------------------------------------------------------- #
# deltamethod
# --------------------------------------------------------------------------- #

@dataclass
class DeltaMethodResult:
    """Result of :func:`deltamethod` (``metafor::deltamethod``)."""

    coef: list[float]
    se: list[float]
    zval: list[float]
    pval: list[float]
    ci_lb: list[float]
    ci_ub: list[float]
    vcov: np.ndarray
    level: float
    order: int
    test: str = "z"


def deltamethod(coef, vcov, fun, order=1, level=0.95, H0=0.0):
    """Delta-method standard errors for a transformation of the coefficients.

    Args:
        coef: coefficient vector.
        vcov: corresponding variance-covariance matrix.
        fun: callable mapping the coefficient vector (or, when its signature has
            one parameter per coefficient, one scalar argument per coefficient)
            to a transformed value (scalar or vector).
        order: ``1``, ``2`` or ``3`` — order of the delta-method approximation.
        level: confidence level (0.95 → 95% CIs).
        H0: null value(s) used for the z-test.

    Returns:
        :class:`DeltaMethodResult`.

    Note:
        metafor computes the required derivatives with the ``calculus`` package;
        here they are obtained with high-accuracy central differences.
    """
    coef = np.asarray(coef, dtype=float).ravel()
    if not callable(fun):
        raise TypeError("argument 'fun' must be a callable")
    if order not in (1, 2, 3):
        raise ValueError("argument 'order' must be equal to 1, 2, or 3")

    vcov = np.asarray(vcov, dtype=float)
    if vcov.ndim == 1:
        vcov = np.diag(vcov)
    if vcov.shape[0] != vcov.shape[1]:
        raise ValueError("argument 'vcov' must be a square matrix")
    if not np.allclose(vcov, vcov.T):
        raise ValueError("argument 'vcov' must be a symmetric matrix")
    p = coef.size
    if vcov.shape[0] != p:
        raise ValueError(
            f"length of the 'coef' vector ({p}) does not match the dimensions "
            f"of 'vcov' ({vcov.shape[0]}x{vcov.shape[1]})"
        )

    # metafor: a one-argument function receives the whole coefficient vector,
    # a p-argument function receives one scalar per coefficient.
    import inspect

    try:
        nargs = len(inspect.signature(fun).parameters)
    except (TypeError, ValueError):
        nargs = 1
    if nargs == 1:
        fun_v = fun
    elif nargs == p:
        fun_v = lambda v: fun(*[float(z) for z in np.atleast_1d(v)])
    else:
        raise ValueError(
            f"number of function arguments ({nargs}) does not match the number "
            f"of coefficients ({p})"
        )

    coef_t = np.atleast_1d(fun_v(coef)).astype(float)
    q = coef_t.size
    grad = _num_grad(fun_v, coef)
    if order == 1:
        vcov_t = grad @ vcov @ grad.T
    elif order == 2:
        Hess = _num_hess(fun_v, coef)
        vcov_t = grad @ vcov @ grad.T
        for a in range(q):
            for b in range(a, q):
                val = vcov_t[a, b] + 0.5 * np.trace(
                    Hess[a] @ vcov @ Hess[b] @ vcov
                )
                vcov_t[a, b] = vcov_t[b, a] = val
    else:
        Hess = _num_hess(fun_v, coef)
        Third = np.empty((q, p, p, p))
        for i in range(p):
            for j in range(i, p):
                for k in range(j, p):
                    val = _num_third(fun_v, coef, i, j, k)
                    for perm in {(i, j, k), (i, k, j), (j, i, k),
                                 (j, k, i), (k, i, j), (k, j, i)}:
                        Third[:, perm[0], perm[1], perm[2]] = val
        u = np.array([
            [float(np.sum(Third[a] * vcov)) for _k in range(p)]
            for a in range(q)
        ])
        K3 = np.kron(vcov, np.kron(vcov, vcov))
        vcov_t = np.empty((q, q))
        for a in range(q):
            for b in range(a, q):
                ga = grad[a]
                gb = grad[b]
                term11 = float(ga @ vcov @ gb)
                term22 = 0.5 * float(np.trace(Hess[a] @ vcov @ Hess[b] @ vcov))
                term13 = 0.5 * float(ga @ vcov @ u[b] + gb @ vcov @ u[a])
                Ta = Third[a].ravel(order="F")
                Tb = Third[b].ravel(order="F")
                term33 = float(Ta @ (K3 @ Tb)) / 6.0 + float(u[a] @ vcov @ u[b]) / 4.0
                vcov_t[a, b] = vcov_t[b, a] = term11 + term22 + term13 + term33

    crit = stats.norm.isf((1.0 - level) / 2.0)
    se = np.sqrt(np.diag(vcov_t))
    H0v = np.full(q, float(H0)) if np.size(H0) == 1 else np.asarray(H0, dtype=float)
    if H0v.size != q:
        raise ValueError(
            f"length of the 'H0' argument ({H0v.size}) does not match the "
            f"number of transformed coefficients ({q})"
        )
    zval = (coef_t - H0v) / se
    pval = 2.0 * stats.norm.sf(np.abs(zval))

    return DeltaMethodResult(
        coef=[float(v) for v in coef_t],
        se=[float(v) for v in se],
        zval=[float(v) for v in zval],
        pval=[float(v) for v in pval],
        ci_lb=[float(v) for v in coef_t - crit * se],
        ci_ub=[float(v) for v in coef_t + crit * se],
        vcov=vcov_t,
        level=level,
        order=order,
    )


# --------------------------------------------------------------------------- #
# tes
# --------------------------------------------------------------------------- #

def _tes_intfun(x, theta, tau, sei, H0, alternative, crit):
    """metafor's ``.tes.intfun``."""
    if alternative == "two.sided":
        pow_ = (stats.norm.sf(crit, (x - H0) / sei)
                + stats.norm.cdf(-crit, (x - H0) / sei))
    elif alternative == "greater":
        pow_ = stats.norm.sf(crit, (x - H0) / sei)
    else:
        pow_ = stats.norm.cdf(crit, (x - H0) / sei)
    return float(pow_ * stats.norm.pdf(x, theta, tau))


@dataclass
class TesResult:
    """Result of :func:`tes` (``metafor::tes``)."""

    k: int
    O: int
    E: float
    OEratio: float
    test: str
    X2: float
    pval: float
    power: list[float]
    sig: list[bool]
    theta: object
    theta_lim: object
    tes_alternative: str
    tes_alpha: float


def tes(yi, vi=None, sei=None, *, H0=0.0, alternative="two.sided", alpha=0.05,
        theta=None, tau2=None, test=None, tes_alternative="greater",
        tes_alpha=0.1, correct=False, find_lim=True, rel_tol=_QUAD_TOL,
        subdivisions=100, tau2_lb=0.0):
    """Test of excess significance (``metafor::tes``).

    Args:
        yi: study estimates.
        vi: sampling variances (or supply ``sei``).
        sei: standard errors (used when ``vi`` is not given).
        H0: null value of the effect size (default 0).
        alternative: ``"two.sided"``, ``"greater"`` or ``"less"``.
        alpha: significance level used to declare a study significant.
        theta: assumed (common) true effect size(s); estimated when ``None``.
        tau2: between-study variance used for the power computation.
        test: ``"exact"``, ``"chi2"`` or ``"binom"`` (chosen automatically
            when ``None``).
        tes_alternative: alternative for the test of excess significance.
        tes_alpha: level used to derive ``theta_lim``.
        correct: apply Yates' continuity correction (``chi2`` test).
        find_lim: compute the ``theta`` value at which the test would no longer
            be significant.
        rel_tol: relative tolerance for the power integrals.
        subdivisions: maximum number of subdivisions for the integrals.
        tau2_lb: values of ``tau2`` at or below this are treated as zero.

    Returns:
        :class:`TesResult`.
    """
    yi = np.asarray(yi, dtype=float).ravel()
    if vi is None:
        if sei is None:
            raise ValueError("must specify the 'vi' or 'sei' argument")
        vi = np.asarray(sei, dtype=float).ravel() ** 2
    else:
        vi = np.asarray(vi, dtype=float).ravel()
    if yi.size != vi.size:
        raise ValueError("length of 'yi' and 'vi' (or 'sei') are not the same")
    k = yi.size
    if k == 0:
        raise ValueError("stopped because k = 0")
    sei = np.sqrt(vi)

    if alternative == "two.sided":
        crit = stats.norm.isf(alpha / 2.0)
    elif alternative == "greater":
        crit = stats.norm.isf(alpha)
    elif alternative == "less":
        crit = stats.norm.ppf(alpha)
    else:
        raise ValueError("alternative must be 'two.sided', 'greater' or 'less'")

    if theta is None:
        single_theta, est_theta = True, True
        theta_arr = np.zeros(k)
    else:
        th = np.atleast_1d(np.asarray(theta, dtype=float))
        if th.size == 1:
            single_theta, est_theta = True, False
            theta_1 = float(th[0])
            theta_arr = np.full(k, theta_1)
        else:
            single_theta, est_theta = False, False
            if th.size != k:
                raise ValueError("length of 'theta' and 'yi' are not the same")
            theta_arr = th

    use_tau2 = tau2 is not None and tau2 > tau2_lb
    wi = 1.0 / (vi + tau2) if use_tau2 else 1.0 / vi
    if est_theta:
        theta_1 = float(np.sum(wi * yi) / np.sum(wi))
        theta_arr = np.full(k, theta_1)

    zi = (yi - H0) / sei
    if not use_tau2:
        if alternative == "two.sided":
            pow_ = (stats.norm.sf(crit, (theta_arr - H0) / sei)
                    + stats.norm.cdf(-crit, (theta_arr - H0) / sei))
        elif alternative == "greater":
            pow_ = stats.norm.sf(crit, (theta_arr - H0) / sei)
        else:
            pow_ = stats.norm.cdf(crit, (theta_arr - H0) / sei)
    else:
        tau = math.sqrt(tau2)
        pow_ = np.empty(k)
        for i in range(k):
            pow_[i] = integrate.quad(
                _tes_intfun,
                theta_arr[i] - 5 * tau, theta_arr[i] + 5 * tau,
                args=(theta_arr[i], tau, sei[i], H0, alternative, crit),
                epsabs=rel_tol, epsrel=rel_tol, limit=subdivisions,
            )[0]

    if alternative == "two.sided":
        sig = np.abs(zi) >= crit
    elif alternative == "greater":
        sig = zi >= crit
    else:
        sig = zi <= crit

    E = float(np.sum(pow_))
    O = int(np.sum(sig))

    if tes_alternative == "two.sided":
        js = list(range(0, k + 1))
    elif tes_alternative == "greater":
        js = list(range(O, k + 1))
    elif tes_alternative == "less":
        js = list(range(0, O + 1))
    else:
        raise ValueError("tes.alternative must be 'two.sided', 'greater' or 'less'")

    if test is None:
        test = "exact" if sum(math.comb(k, j) for j in js) <= 10 ** 6 else "chi2"
    if test not in ("chi2", "binom", "exact"):
        raise ValueError("test must be 'chi2', 'binom' or 'exact'")

    X2 = np.nan
    if test == "chi2":
        # R's prop.test(O, k, p = E/k, alternative = ..., correct = correct)
        p0 = E / k
        if correct:
            stat = (abs(O - k * p0) - 0.5) ** 2 / (k * p0 * (1 - p0)) if k * p0 * (1 - p0) > 0 else np.nan
        else:
            stat = (O - k * p0) ** 2 / (k * p0 * (1 - p0)) if k * p0 * (1 - p0) > 0 else np.nan
        X2 = float(stat)
        if tes_alternative == "greater":
            pval = float(stats.chi2.sf(X2, 1) / 2.0) if O > k * p0 else float(1 - stats.chi2.cdf(X2, 1) / 2.0)
        elif tes_alternative == "less":
            pval = float(stats.chi2.sf(X2, 1) / 2.0) if O < k * p0 else float(1 - stats.chi2.cdf(X2, 1) / 2.0)
        else:
            pval = float(stats.chi2.sf(X2, 1))
    elif test == "binom":
        pval = float(stats.binomtest(O, k, E / k, alternative=tes_alternative).pvalue)
    else:
        prj = np.empty(len(js))
        ids = list(range(k))
        for jj, j in enumerate(js):
            if j == 0:
                prj[jj] = float(np.prod(1.0 - pow_))
            elif j == k:
                prj[jj] = float(np.prod(pow_))
            else:
                total = 0.0
                for sel in combinations(ids, j):
                    sel_set = set(sel)
                    notsel = [i for i in ids if i not in sel_set]
                    total += float(np.prod(pow_[list(sel)]) * np.prod(1.0 - pow_[notsel]))
                prj[jj] = total
        if tes_alternative == "two.sided":
            ref = prj[O]
            pval = float(np.sum(prj[prj <= ref + math.sqrt(_EPS)]))
        else:
            pval = float(np.sum(prj))
        pval = min(pval, 1.0)

    theta_lim = None
    if find_lim and single_theta:
        def _lim(th):
            return _tes_pval(yi, vi, H0, alternative, alpha, th, tau2, test,
                             tes_alternative, correct, rel_tol, subdivisions,
                             tau2_lb) - tes_alpha

        if tes_alternative == "two.sided":
            lo = _tes_theta_lim(yi, vi, H0, alternative, alpha, theta_1, tau2,
                                test, "greater", correct, rel_tol, subdivisions,
                                tau2_lb, tes_alpha / 2.0)
            hi = _tes_theta_lim(yi, vi, H0, alternative, alpha, theta_1, tau2,
                                test, "less", correct, rel_tol, subdivisions,
                                tau2_lb, tes_alpha / 2.0)
            theta_lim = [lo, hi]
        else:
            theta_lim = _tes_theta_lim(yi, vi, H0, alternative, alpha, theta_1,
                                       tau2, test, tes_alternative, correct,
                                       rel_tol, subdivisions, tau2_lb, tes_alpha)

    return TesResult(
        k=k, O=O, E=E, OEratio=(O / E if E != 0 else np.inf), test=test,
        X2=X2, pval=pval,
        power=[float(v) for v in pow_],
        sig=[bool(v) for v in sig],
        theta=(float(theta_1) if single_theta else [float(v) for v in theta_arr]),
        theta_lim=theta_lim,
        tes_alternative=tes_alternative, tes_alpha=tes_alpha,
    )


def _tes_pval(yi, vi, H0, alternative, alpha, theta, tau2, test,
              tes_alt, correct, rel_tol, subdivisions, tau2_lb):
    return tes(yi, vi=vi, H0=H0, alternative=alternative, alpha=alpha,
               theta=theta, tau2=tau2, test=test, tes_alternative=tes_alt,
               correct=correct, find_lim=False, rel_tol=rel_tol,
               subdivisions=subdivisions, tau2_lb=tau2_lb).pval


def _tes_theta_lim(yi, vi, H0, alternative, alpha, theta_1, tau2, test,
                   tes_alt, correct, rel_tol, subdivisions, tau2_lb, tes_alpha):
    """metafor's ``.tes.lim`` root finding for ``theta``."""
    def f(th):
        return _tes_pval(yi, vi, H0, alternative, alpha, th, tau2, test,
                         tes_alt, correct, rel_tol, subdivisions, tau2_lb) - tes_alpha

    f_h0 = f(H0)
    if tes_alt == "greater" and f_h0 >= 0:
        return None
    if tes_alt == "less" and f_h0 <= 0:
        return None
    lo, hi = (H0, theta_1) if theta_1 >= H0 else (theta_1, H0)
    try:
        from scipy.optimize import brentq
        # R's uniroot() default tolerance (.Machine$double.eps^0.25)
        return float(brentq(f, lo, hi, xtol=_QUAD_TOL))
    except Exception:
        return None


# --------------------------------------------------------------------------- #
# vif
# --------------------------------------------------------------------------- #

@dataclass
class VifResult:
    """Result of :func:`vif` (``metafor::vif``)."""

    vifs: list[float]
    sifs: list[float]
    m: list[int]
    specs: list[str]
    coefs: list[str]
    coefnames: list[str]
    bttspec: bool


def vif(result, btt=None, intercept=False):
    """Generalized variance-inflation factors (Fox & Monette).

    Args:
        result: a fitted :class:`RmaResult`.
        btt: indices (1-based, like metafor) or names of the coefficients to
            assess. Defaults to all non-intercept coefficients, or to all
            coefficients when the model has no intercept.
        intercept: keep the intercept when computing the VIFs.

    Returns:
        :class:`VifResult`.

    Raises:
        ValueError: for intercept-only models, where VIFs are undefined.
    """
    from .rma_uni import RmaResult

    if not isinstance(result, RmaResult):
        raise TypeError("argument 'result' must be an RmaResult")
    if result.int_only:
        raise ValueError("VIFs not applicable to intercept-only models")

    vb = np.asarray(result.vb, dtype=float)
    p = vb.shape[0]
    has_icpt = bool(result.int_incl)
    if has_icpt:
        names = ["intrcpt"] + [f"mods{i}" for i in range(1, p)]
    else:
        names = [f"mods{i+1}" for i in range(p)]

    bttspec = btt is not None
    if btt is None:
        # metafor: btt <- as.list(2:x$p)  -> one VIF per coefficient
        if has_icpt and not intercept:
            btt = [[i] for i in range(2, p + 1)]
        else:
            btt = [[i] for i in range(1, p + 1)]
    else:
        # a flat specification is a single (joint) set, like metafor's
        # `if (!is.list(btt)) btt <- list(btt)`
        if isinstance(btt, str):
            btt = [[btt]]
        elif isinstance(btt, (list, tuple)) and btt and not isinstance(btt[0], (list, tuple)):
            btt = [list(btt)]
        elif not isinstance(btt, (list, tuple)):
            btt = [[btt]]
        btt = [list(b) for b in btt]

    def _idx(one):
        if all(isinstance(b, str) for b in one):
            return [names.index(b) + 1 for b in one]
        return [int(b) for b in np.atleast_1d(one)]

    if has_icpt and not intercept:
        vb_use = vb[1:, 1:]
    else:
        vb_use = vb
    rb = _cov2cor(vb_use)

    vifs, sifs, ms, specs, coefs, cnames = [], [], [], [], [], []
    for one in btt:
        idx = [i for i in _idx(one) if 1 <= i <= p]
        if not idx:
            raise ValueError("cannot compute VIF(s) for the specified 'btt' argument")
        if has_icpt and not intercept:
            if idx == [1]:
                raise ValueError("cannot compute VIF(s) for the specified 'btt' argument")
            # 1-based index i (i >= 2) -> 0-based column i-2 once the intercept
            # is dropped (metafor's `.compvif` does `btt <- btt - 1` on the
            # reduced 1-based index, which is the same thing).
            btt0 = [i - 2 for i in idx if i > 1]
        else:
            btt0 = [i - 1 for i in idx]
        keep = [j for j in range(rb.shape[0]) if j not in btt0]
        try:
            gvif = float(
                np.linalg.det(rb[np.ix_(btt0, btt0)])
                * (np.linalg.det(rb[np.ix_(keep, keep)]) if keep else 1.0)
                / np.linalg.det(rb)
            )
        except np.linalg.LinAlgError:
            gvif = np.nan
        m = len(btt0)
        vifs.append(gvif)
        sifs.append(gvif ** (1.0 / (2.0 * m)) if gvif > 0 else np.nan)
        ms.append(m)
        specs.append(",".join(str(i) for i in idx))
        coefs.append(",".join(str(i) for i in idx))
        cnames.append(names[idx[0] - 1] if len(idx) == 1 else "")

    return VifResult(
        vifs=vifs, sifs=sifs, m=ms, specs=specs, coefs=coefs,
        coefnames=cnames, bttspec=bttspec,
    )


# --------------------------------------------------------------------------- #
# permutest
# --------------------------------------------------------------------------- #

@dataclass
class PermutestResult:
    """Result of :func:`permutest` (``metafor::permutest``)."""

    pval: list[float]
    QM: float
    QMp: float
    QMp_perm: list[float]
    dfs: list[int]
    k: int
    p: int
    nperm: int


def permutest(result, exact=False, iter=1000, permci=False, seed=None):
    """Permutation test for the model coefficients (``metafor::permutest``).

    The observed data are repeatedly permuted (rows of ``X`` kept aligned with
    the outcome only under H0) and the Wald-type statistic is recomputed.

    Args:
        result: fitted :class:`RmaResult`.
        exact: enumerate all permutations (only feasible for small ``k``).
        iter: number of random permutations when ``exact=False``.
        permci: also compute permutation confidence intervals (not implemented;
            accepted for API compatibility).
        seed: optional RNG seed.

    Returns:
        :class:`PermutestResult`.
    """
    from .rma_uni import rma

    rng = np.random.default_rng(seed)
    yi = np.asarray(result.yi, dtype=float)
    vi = np.asarray(result.vi, dtype=float)
    X = np.asarray(result.X, dtype=float)
    p = X.shape[1]
    k = yi.size
    btt = [0] if result.int_only else list(range(1, p))
    m = len(btt)

    def _stat(beta, vb):
        idx = np.ix_(btt, btt)
        try:
            return float(beta[btt] @ np.linalg.solve(vb[idx], beta[btt]))
        except np.linalg.LinAlgError:
            return np.nan

    beta_obs = np.asarray(result.b, dtype=float)
    QM = _stat(beta_obs, np.asarray(result.vb, dtype=float))
    QMp = float(stats.chi2.sf(QM, m)) if m > 0 else np.nan

    def _fit(Y):
        # permute the design instead of the outcome (equivalent under H0 and
        # keeps the var-cov structure intact)
        r = rma(Y, vi=vi, mods=(X[:, 1:] if p > 1 else None),
                method=result.method, test="z", level=result.level)
        return _stat(np.asarray(r.b, dtype=float), np.asarray(r.vb, dtype=float))

    stats_perm = []
    if exact:
        from itertools import permutations as _perms
        for perm in _perms(range(k)):
            stats_perm.append(_fit(yi[list(perm)]))
    else:
        for _ in range(int(iter)):
            stats_perm.append(_fit(yi[rng.permutation(k)]))

    stats_perm = np.asarray(stats_perm, dtype=float)
    pvals = [float(np.mean(stats_perm >= QM)) for _ in range(m)]
    return PermutestResult(
        pval=pvals, QM=QM, QMp=QMp, QMp_perm=pvals,
        dfs=[k - p] * m, k=k, p=p, nperm=stats_perm.size,
    )
