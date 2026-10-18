"""``transf.*`` — a faithful transcription of metafor's effect-size transforms.

Every function mirrors the corresponding ``metafor::transf.<name>`` (metafor
5.2-1) including its argument checks, its handling of the parameter-space
bounds and its ``NaN``/``Inf`` behaviour (e.g. ``transf.ahw(3)`` is ``NaN``,
matching R's ``(-2)^(1/3)``).

The ``.int`` variants integrate the transform over the random-effects
distribution ``N(xi, tau2)`` (truncated to the transform's support) and the
``.mode`` variants return the mode of the back-transformed distribution, exactly
like metafor.

Only NumPy/SciPy are used.
"""

from __future__ import annotations

import math

import numpy as np
from scipy import integrate, optimize, special, stats

__all__ = [
    "transf_abt", "transf_ahw", "transf_arcsin", "transf_clestod",
    "transf_dtobesd", "transf_dtocles", "transf_dtocles_int", "transf_dtocliffd",
    "transf_dtolnor_logis", "transf_dtolnor_norm", "transf_dtomd", "transf_dtoovl",
    "transf_dtorbis", "transf_dtorpb", "transf_dtou1", "transf_dtou2",
    "transf_dtou3", "transf_exp_int", "transf_exp_mode",
    "transf_iabt", "transf_iabt_int", "transf_iabt_mode", "transf_iahw",
    "transf_iahw_int", "transf_iahw_mode", "transf_iarcsin", "transf_iarcsin_int",
    "transf_iarcsin_mode", "transf_iirft", "transf_ilogit", "transf_ilogit_int",
    "transf_ilogit_mode", "transf_ipft", "transf_ipft_hm", "transf_iprobit",
    "transf_iprobit_int", "transf_iprobit_mode", "transf_irft", "transf_isqrt",
    "transf_lnortod_logis", "transf_lnortod_norm", "transf_lnortord",
    "transf_lnortorr", "transf_lnortortet_digby", "transf_lnortortet_pearson",
    "transf_logit", "transf_pft", "transf_probit", "transf_r2toz",
    "transf_rpbtod", "transf_rpbtorbis", "transf_rtod", "transf_rtorpb",
    "transf_rtoz", "transf_ztor", "transf_ztor_int", "transf_ztor_mode",
    "transf_ztor2",
]

_EPS = np.finfo(float).eps
# R's integrate() default: rel.tol = .Machine$double.eps^0.25
_QUAD_TOL = float(_EPS ** 0.25)


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #

def _asarray(x):
    """Return ``(arr, scalar_input)`` for input ``x``."""
    if np.isscalar(x):
        return np.asarray([x], dtype=float), True
    return np.asarray(x, dtype=float), False


def _ret(y, scalar):
    y = np.asarray(y, dtype=float)
    return float(y[0]) if scalar else y


def _r_pow(base, exp):
    """R's ``base^exp``: ``NaN`` when ``base < 0`` and ``exp`` is fractional."""
    base = np.asarray(base, dtype=float)
    out = np.full(base.shape, np.nan)
    ok = (base >= 0) | np.isclose(exp, np.round(exp))
    with np.errstate(invalid="ignore", divide="ignore"):
        out[ok] = np.power(base[ok], exp)
    return out


def _expand1(x, n):
    """R's ``.expand1``: recycle a length-1 vector to length ``n``."""
    if x is None:
        return None
    a = np.asarray(x, dtype=float)
    if a.ndim == 0 or a.size == 1:
        return np.full(n, float(a.ravel()[0]))
    return a


def _hi(n1i, n2i, k):
    """``hi = m/n1 + m/n2`` with ``m = n1 + n2 - 2`` (metafor's default 4)."""
    if n1i is None or n2i is None:
        return np.full(k, 4.0)
    n1i = _expand1(n1i, k)
    n2i = _expand1(n2i, k)
    if n1i.size != n2i.size:
        raise ValueError("length of 'n1i' does not match the length of 'n2i'")
    if n1i.size != k:
        raise ValueError("lengths of 'n1i'/'n2i' do not match the length of 'x'")
    mi = n1i + n2i - 2.0
    return mi / n1i + mi / n2i


def _chktargsint(targs):
    """metafor's ``.chktargsint`` — ``targs`` must carry ``tau2``."""
    if targs is None:
        return {"tau2": 0.0, "lower": None, "upper": None}
    if isinstance(targs, dict):
        if "tau2" not in targs:
            raise ValueError("must specify a 'tau2' value via the 'targs' argument")
        return dict(targs)
    return {"tau2": float(targs), "lower": None, "upper": None}


def _int_over_dist(fun, xi, targs, lower=None, upper=None, density=None):
    """Integrate ``fun(z) * dnorm(z; xi, tau) [/ density]`` over ``[lower, upper]``."""
    targs = _chktargsint(targs)
    tau2 = float(targs["tau2"])
    if tau2 < _EPS:
        tau2 = 0.0
    tau = math.sqrt(tau2)
    lo = targs.get("lower")
    up = targs.get("upper")
    if lo is None:
        lo = lower(xi, tau) if callable(lower) else -np.inf
    if up is None:
        up = upper(xi, tau) if callable(upper) else np.inf

    xi_arr, scalar = _asarray(xi)
    out = np.empty(xi_arr.size)
    for i, mu in enumerate(xi_arr):
        if tau2 == 0.0:
            out[i] = fun(mu)
            continue
        lo_i = float(np.atleast_1d(lo)[0]) if np.ndim(lo) else float(lo)
        up_i = float(np.atleast_1d(up)[0]) if np.ndim(up) else float(up)
        norm = density(mu, tau) if density is not None else 1.0

        def _integrand(z, mu=mu, tau=tau, norm=norm):
            return fun(z) * float(stats.norm.pdf(z, mu, tau)) / norm

        try:
            out[i] = integrate.quad(
                _integrand, lo_i, up_i,
                epsabs=_QUAD_TOL, epsrel=_QUAD_TOL, limit=100,
            )[0]
        except Exception:
            out[i] = np.nan
    return _ret(out, scalar)


def _mode_over(fun, xi, tau2, lower, upper):
    """metafor's ``.mode`` helper: maximise ``fun`` on ``[lower, upper]``."""
    tau2 = float(tau2)
    if tau2 < 0:
        raise ValueError("value of 'tau2' must be >= 0")
    if tau2 < _EPS:
        tau2 = 0.0
    tau = math.sqrt(tau2)
    xi_arr, scalar = _asarray(xi)
    out = np.empty(xi_arr.size)
    for i, mu in enumerate(xi_arr):
        res = optimize.minimize_scalar(
            lambda z, mu=mu, tau=tau: -fun(z, mu, tau),
            bounds=(lower, upper), method="bounded",
        )
        out[i] = res.x
    return _ret(out, scalar)


def _tau2_scalar(targs):
    if targs is None:
        raise ValueError("must specify a 'tau2' value via the 'targs' argument")
    tau2 = float(targs["tau2"] if isinstance(targs, dict) else targs)
    if tau2 < 0:
        raise ValueError("value of 'tau2' must be >= 0")
    return 0.0 if tau2 < _EPS else tau2


# --------------------------------------------------------------------------- #
# log / logit / probit / arcsine family
# --------------------------------------------------------------------------- #

def transf_logit(xi):
    """``transf.logit`` — logit transform of a proportion."""
    a, s = _asarray(xi)
    return _ret(special.logit(a), s)


def transf_probit(xi):
    """``transf.probit`` — probit transform of a proportion."""
    a, s = _asarray(xi)
    return _ret(special.ndtri(a), s)


def transf_arcsin(xi):
    """``transf.arcsin`` — arcsine (angular) transform of a proportion."""
    a, s = _asarray(xi)
    return _ret(np.arcsin(np.sqrt(a)), s)


def transf_ilogit(xi):
    """``transf.ilogit`` — inverse logit."""
    a, s = _asarray(xi)
    return _ret(special.expit(a), s)


def transf_iprobit(xi):
    """``transf.iprobit`` — inverse probit."""
    a, s = _asarray(xi)
    return _ret(stats.norm.cdf(a), s)


def transf_iarcsin(xi):
    """``transf.iarcsin`` — inverse arcsine (truncated to [0, 1])."""
    a, s = _asarray(xi)
    z = np.sin(a) ** 2
    z[a < 0] = 0.0
    z[a > np.arcsin(1.0)] = 1.0
    return _ret(z, s)


def transf_iabt(xi):
    """``transf.iabt`` — inverse of the log(1-x) transform."""
    a, s = _asarray(xi)
    with np.errstate(invalid="ignore", over="ignore"):
        z = 1.0 - np.exp(-a)
    z[a < 0] = 0.0
    z = np.where(np.isnan(z), np.nan, z)
    return _ret(z, s)


def transf_iahw(xi):
    """``transf.iahw`` — inverse of the ``1-(1-x)^(1/3)`` transform."""
    a, s = _asarray(xi)
    z = 1.0 - (1.0 - a) ** 3
    z = np.where(np.isnan(z), np.nan, z)
    z[a > 1] = 1.0
    z[a < 0] = 0.0
    return _ret(z, s)


def transf_isqrt(xi):
    """``transf.isqrt`` — inverse square-root transform (``x^2``)."""
    a, s = _asarray(xi)
    z = a * a
    z[a < 0] = 0.0
    return _ret(z, s)


# --------------------------------------------------------------------------- #
# Fisher r <-> z and friends
# --------------------------------------------------------------------------- #

def transf_rtoz(xi):
    """``transf.rtoz`` — Fisher's r-to-z (clamped to [-1, 1])."""
    a, s = _asarray(xi)
    a = a.copy()
    a[a > 1] = 1.0
    a[a < -1] = -1.0
    with np.errstate(divide="ignore", invalid="ignore"):
        z = np.arctanh(a)
    return _ret(z, s)


def transf_ztor(xi):
    """``transf.ztor`` — Fisher's z-to-r."""
    a, s = _asarray(xi)
    return _ret(np.tanh(a), s)


def transf_ztor2(xi):
    """``transf.ztor2`` — z-to-r, squared (coefficient of determination)."""
    a, s = _asarray(xi)
    return _ret(np.tanh(a) ** 2, s)


def transf_r2toz(xi):
    """``transf.r2toz`` — R^2 to Fisher's z (clamped to [0, 1])."""
    a, s = _asarray(xi)
    a = a.copy()
    a[a > 1] = 1.0
    a[a < 0] = 0.0
    return _ret(np.arctanh(np.sqrt(a)), s)


def transf_ztor_int(xi, targs=None):
    """``transf.ztor.int`` — E[tanh(theta)] under ``theta ~ N(xi, tau2)``."""
    return _int_over_dist(
        np.tanh, xi, targs,
        lower=lambda mu, tau: mu - 10.0 * tau,
        upper=lambda mu, tau: mu + 10.0 * tau,
    )


def transf_ztor_mode(xi, targs=None):
    """``transf.ztor.mode`` — mode of tanh(theta)."""
    tau2 = _tau2_scalar(targs)
    return _mode_over(
        lambda z, mu, tau: float(stats.norm.pdf(np.arctanh(z), mu, tau) / (1.0 - z ** 2)),
        xi, tau2, -0.9999, 0.9999,
    )


def transf_ilogit_int(xi, targs=None):
    """``transf.ilogit.int`` — E[logistic(theta)] under ``theta ~ N(xi, tau2)``."""
    return _int_over_dist(
        lambda z: float(special.expit(z)), xi, targs,
        lower=lambda mu, tau: mu - 10.0 * tau,
        upper=lambda mu, tau: mu + 10.0 * tau,
    )


def transf_ilogit_mode(xi, targs=None):
    """``transf.ilogit.mode`` — mode of the back-transformed logit.

    Mirrors metafor's implementation: scan ``seq(0, 1, length=1e5)`` and only
    return a root when the density is unimodal.
    """
    tau2 = _tau2_scalar(targs)
    xi_arr, scalar = _asarray(xi)
    out = np.empty(xi_arr.size)
    xs = np.linspace(0.0, 1.0, 100000)

    def modefun(x, mu, tau):
        return tau ** 2 * (2.0 * x - 1.0) + mu - float(special.logit(x))

    for i, mu in enumerate(xi_arr):
        if tau2 == 0.0:
            out[i] = float(special.expit(mu))
            continue
        tau = math.sqrt(tau2)
        ys = modefun(xs, mu, tau)
        if len(np.unique(np.sign(np.diff(ys)))) == 1:
            try:
                out[i] = optimize.brentq(modefun, 1e-12, 1.0 - 1e-12, args=(mu, tau))
            except Exception:
                out[i] = np.nan
        else:
            out[i] = np.nan
    return _ret(out, scalar)


def transf_iprobit_int(xi, targs=None):
    """``transf.iprobit.int`` — E[Phi(theta)] under ``theta ~ N(xi, tau2)``."""
    return _int_over_dist(
        lambda z: float(stats.norm.cdf(z)), xi, targs,
        lower=lambda mu, tau: mu - 10.0 * tau,
        upper=lambda mu, tau: mu + 10.0 * tau,
    )


def transf_iprobit_mode(xi, targs=None):
    """``transf.iprobit.mode`` — mode of the back-transformed probit."""
    tau2 = _tau2_scalar(targs)
    return _mode_over(
        lambda x, mu, tau: float(
            stats.norm.pdf(special.ndtri(x), mu, tau) / stats.norm.pdf(special.ndtri(x))
        ),
        xi, tau2, 0.0, 1.0,
    )


def transf_iarcsin_int(xi, targs=None):
    """``transf.iarcsin.int`` — E[sin(theta)^2] under ``theta ~ N(xi, tau2)``."""
    return _int_over_dist(
        lambda z: float(np.sin(z) ** 2), xi, targs,
        lower=lambda mu, tau: max(0.0, mu - 10 * tau),
        upper=lambda mu, tau: min(math.pi / 2, mu + 10 * tau),
        density=lambda mu, tau: float(
            stats.norm.cdf((math.pi / 2 - mu) / tau) - stats.norm.cdf(-mu / tau)
        ),
    )


def transf_iarcsin_mode(xi, targs=None):
    """``transf.iarcsin.mode`` — mode of the back-transformed arcsine."""
    tau2 = _tau2_scalar(targs)
    return _mode_over(
        lambda x, mu, tau: float(
            stats.norm.pdf(math.asin(math.sqrt(x)), mu, tau)
            / (2.0 * math.sqrt(x * (1.0 - x)))
        ),
        xi, tau2, 0.0, 1.0,
    )


def transf_iabt_int(xi, targs=None):
    """``transf.iabt.int`` — E[1-exp(-theta)] under a truncated normal."""
    return _int_over_dist(
        lambda z: float(1.0 - np.exp(-z)) if z >= 0 else 0.0, xi, targs,
        lower=lambda mu, tau: max(0.0, mu - 10 * tau),
        upper=lambda mu, tau: mu + 10 * tau,
        density=lambda mu, tau: float(stats.norm.cdf(mu / tau)),
    )


def transf_iabt_mode(xi, targs=None):
    """``transf.iabt.mode`` — mode of the inverse log(1-x) transform."""
    tau2 = _tau2_scalar(targs)
    return _mode_over(
        lambda x, mu, tau: float(
            stats.norm.pdf(-math.log(1.0 - x), mu, tau)
            / ((1.0 - x) * stats.norm.cdf(mu / tau))
        ),
        xi, tau2, 0.0, 1.0,
    )


def transf_iahw_int(xi, targs=None):
    """``transf.iahw.int`` — E[1-(1-theta)^3] under a truncated normal."""
    return _int_over_dist(
        lambda z: float(1.0 - (1.0 - z) ** 3), xi, targs,
        lower=lambda mu, tau: max(0.0, mu - 10 * tau),
        upper=lambda mu, tau: min(1.0, mu + 10 * tau),
        density=lambda mu, tau: float(
            stats.norm.cdf((1.0 - mu) / tau) - stats.norm.cdf(-mu / tau)
        ),
    )


def transf_iahw_mode(xi, targs=None):
    """``transf.iahw.mode`` — mode of the inverse ``1-(1-x)^(1/3)`` transform."""
    tau2 = _tau2_scalar(targs)
    return _mode_over(
        lambda x, mu, tau: float(
            stats.norm.pdf(1.0 - np.cbrt(1.0 - x), mu, tau)
            / (3.0 * (1.0 - x) ** (2.0 / 3.0)
               * (stats.norm.cdf((1.0 - mu) / tau) - stats.norm.cdf(-mu / tau)))
        ),
        xi, tau2, 0.0, 1.0,
    )


def transf_dtocles_int(xi, targs=None):
    """``transf.dtocles.int`` — E[Phi(theta/sqrt(2))] under ``N(xi, tau2)``."""
    return _int_over_dist(
        lambda z: float(stats.norm.cdf(z / math.sqrt(2.0))), xi, targs,
        lower=lambda mu, tau: mu - 10.0 * tau,
        upper=lambda mu, tau: mu + 10.0 * tau,
    )


def transf_exp_int(xi, targs=None):
    """``transf.exp.int`` — ``exp(xi + tau2/2)`` (mean of the log-normal)."""
    targs = _chktargsint(targs)
    tau2 = float(targs["tau2"])
    a, s = _asarray(xi)
    return _ret(np.exp(a + tau2 / 2.0), s)


def transf_exp_mode(xi, targs=None):
    """``transf.exp.mode`` — ``exp(xi - tau2)`` (mode of the log-normal)."""
    tau2 = _tau2_scalar(targs)
    a, s = _asarray(xi)
    return _ret(np.exp(a - tau2), s)


# --------------------------------------------------------------------------- #
# standardized mean difference family
# --------------------------------------------------------------------------- #

def transf_dtobesd(xi):
    """``transf.dtobesd`` — Cohen's d to Bonett's unbiased d."""
    a, s = _asarray(xi)
    rpbi = a / np.sqrt(a ** 2 + 4.0)
    return _ret(0.5 + rpbi / 2.0, s)


def transf_dtocles(xi):
    """``transf.dtocles`` — Cohen's d to common-language effect size."""
    a, s = _asarray(xi)
    return _ret(stats.norm.cdf(a / math.sqrt(2.0)), s)


def transf_dtocliffd(xi):
    """``transf.dtocliffd`` — Cohen's d to Cliff's delta."""
    a, s = _asarray(xi)
    return _ret(2.0 * stats.norm.cdf(a / math.sqrt(2.0)) - 1.0, s)


def transf_dtolnor_logis(xi):
    """``transf.dtolnor.logis`` — d to log odds ratio (logistic distribution)."""
    a, s = _asarray(xi)
    return _ret(a / math.sqrt(3.0) * math.pi, s)


def transf_dtolnor_norm(xi):
    """``transf.dtolnor.norm`` — d to log odds ratio (normal distribution, 1.65)."""
    a, s = _asarray(xi)
    return _ret(a * 1.65, s)


def transf_dtomd(xi, targs=None):
    """``transf.dtomd`` — Cohen's d to a raw mean difference (needs ``sd``)."""
    if targs is None:
        raise ValueError("must specify a standard deviation value via the 'targs' argument")
    sd = targs["sd"] if isinstance(targs, dict) else targs
    sd = np.asarray(sd, dtype=float)
    if sd.size != 1:
        raise ValueError("specify a single standard deviation value via the 'targs' argument")
    a, s = _asarray(xi)
    return _ret(a * float(sd), s)


def transf_dtoovl(xi):
    """``transf.dtoovl`` — d to the overlap coefficient (U1)."""
    a, s = _asarray(xi)
    return _ret(2.0 * stats.norm.cdf(-np.abs(a) / 2.0), s)


def transf_dtou1(xi):
    """``transf.dtou1`` — d to Cohen's U1."""
    a, s = _asarray(xi)
    u2 = stats.norm.cdf(np.abs(a) / 2.0)
    return _ret((2.0 * u2 - 1.0) / u2, s)


def transf_dtou2(xi):
    """``transf.dtou2`` — d to Cohen's U2."""
    a, s = _asarray(xi)
    return _ret(stats.norm.cdf(a / 2.0), s)


def transf_dtou3(xi):
    """``transf.dtou3`` — d to Cohen's U3."""
    a, s = _asarray(xi)
    return _ret(stats.norm.cdf(a), s)


def transf_dtorpb(xi, n1i=None, n2i=None):
    """``transf.dtorpb`` — Cohen's d to the point-biserial correlation."""
    a, s = _asarray(xi)
    hi = _hi(n1i, n2i, a.size)
    return _ret(a / np.sqrt(a ** 2 + hi), s)


def transf_rtorpb(xi, pi=0.5):
    """``transf.rtorpb`` — point-biserial r to Cohen's d."""
    a, s = _asarray(xi)
    pi_ = _expand1(pi, a.size)
    if np.any((pi_ < 0) | (pi_ > 1)):
        raise ValueError("one or more 'pi' values are < 0 or > 1")
    return _ret(a * stats.norm.pdf(special.ndtri(pi_)) / np.sqrt(pi_ * (1.0 - pi_)), s)


def transf_rpbtod(xi, n1i=None, n2i=None):
    """``transf.rpbtod`` — point-biserial r to Cohen's d."""
    a, s = _asarray(xi)
    hi = _hi(n1i, n2i, a.size)
    with np.errstate(divide="ignore", invalid="ignore"):
        z = np.sqrt(hi) * a / np.sqrt(1.0 - a ** 2)
    return _ret(z, s)


def transf_dtorbis(xi, n1i=None, n2i=None):
    """``transf.dtorbis`` — Cohen's d to the biserial correlation."""
    a, s = _asarray(xi)
    if n1i is None or n2i is None:
        hi = 4.0
        n1_ = np.ones(a.size)
        n2_ = np.ones(a.size)
    else:
        hi = _hi(n1i, n2i, a.size)
        n1_ = _expand1(n1i, a.size)
        n2_ = _expand1(n2i, a.size)
    rpbi = a / np.sqrt(a ** 2 + hi)
    pi_ = n1_ / (n1_ + n2_)
    return _ret(
        np.sqrt(pi_ * (1.0 - pi_)) / stats.norm.pdf(special.ndtri(pi_)) * rpbi, s
    )


def transf_rpbtorbis(xi, pi=0.5):
    """``transf.rpbtorbis`` — point-biserial r to the biserial correlation."""
    a, s = _asarray(xi)
    pi_ = _expand1(pi, a.size)
    if np.any((pi_ < 0) | (pi_ > 1)):
        raise ValueError("one or more 'pi' values are < 0 or > 1")
    return _ret(
        np.sqrt(pi_ * (1.0 - pi_)) / stats.norm.pdf(special.ndtri(pi_)) * a, s
    )


def transf_rtod(xi, n1i=None, n2i=None):
    """``transf.rtod`` — biserial correlation to Cohen's d."""
    a, s = _asarray(xi)
    if n1i is None or n2i is None:
        hi = 4.0
        n1_ = np.ones(a.size)
        n2_ = np.ones(a.size)
    else:
        hi = _hi(n1i, n2i, a.size)
        n1_ = _expand1(n1i, a.size)
        n2_ = _expand1(n2i, a.size)
    if np.any((n1_ < 0) | (n2_ < 0)):
        raise ValueError("one or more values specified via 'n1i'/'n2i' are negative")
    pi_ = n1_ / (n1_ + n2_)
    rpbi = a * stats.norm.pdf(special.ndtri(pi_)) / np.sqrt(pi_ * (1.0 - pi_))
    with np.errstate(divide="ignore", invalid="ignore"):
        z = np.sqrt(hi) * rpbi / np.sqrt(1.0 - rpbi ** 2)
    return _ret(z, s)


# --------------------------------------------------------------------------- #
# transformations of proportions / counts (Freeman-Tukey, log, ...)
# --------------------------------------------------------------------------- #

def transf_pft(xi, ni):
    """``transf.pft`` — Freeman-Tukey double arcsine of a proportion."""
    a, s = _asarray(xi)
    ni_ = _expand1(ni, a.size)
    x = a * ni_
    z = 0.5 * (np.arcsin(np.sqrt(x / (ni_ + 1.0)))
               + np.arcsin(np.sqrt((x + 1.0) / (ni_ + 1.0))))
    return _ret(z, s)


def transf_ipft(xi, ni):
    """``transf.ipft`` — inverse Freeman-Tukey double arcsine."""
    a, s = _asarray(xi)
    ni_ = _expand1(ni, a.size)
    with np.errstate(divide="ignore", invalid="ignore"):
        st = np.sin(2.0 * a)
        val = 1.0 - (st + (st - 1.0 / st) / ni_) ** 2
        z = 0.5 * (1.0 - np.sign(np.cos(2.0 * a)) * np.sqrt(val))
    z = np.where(np.isnan(z), np.nan, z)
    z[a > transf_pft(1.0, ni_)] = 1.0
    z[a < transf_pft(0.0, ni_)] = 0.0
    return _ret(z, s)


def transf_ipft_hm(xi, targs):
    """``transf.ipft.hm`` — inverse FT using the harmonic mean sample size."""
    if targs is None:
        raise ValueError("must specify the sample sizes via the 'targs' argument")
    ni = targs["ni"] if isinstance(targs, dict) else targs
    ni = np.asarray(ni, dtype=float)
    nhm = 1.0 / float(np.mean(1.0 / ni))
    a, s = _asarray(xi)
    with np.errstate(divide="ignore", invalid="ignore"):
        st = np.sin(2.0 * a)
        val = 1.0 - (st + (st - 1.0 / st) / nhm) ** 2
        z = 0.5 * (1.0 - np.sign(np.cos(2.0 * a)) * np.sqrt(val))
    z = np.where(np.isnan(z), np.nan, z)
    z[a > transf_pft(1.0, nhm)] = 1.0
    z[a < transf_pft(0.0, nhm)] = 0.0
    return _ret(z, s)


def transf_irft(xi, ti):
    """``transf.irft`` — inverse of the square-root rate transform."""
    a, s = _asarray(xi)
    ti_ = _expand1(ti, a.size)
    return _ret(0.5 * (np.sqrt(a) + np.sqrt(a + 1.0 / ti_)), s)


def transf_iirft(xi, ti):
    """``transf.iirft`` — inverse of ``transf.irft`` (back to a rate)."""
    a, s = _asarray(xi)
    ti_ = _expand1(ti, a.size)
    with np.errstate(divide="ignore", invalid="ignore"):
        z = (1.0 / ti_ - 8.0 * a ** 2 + 16.0 * ti_ * a ** 4) / (16.0 * a ** 2 * ti_)
    z = np.where(np.isnan(z), np.nan, z)
    z[a < transf_irft(0.0, ti_)] = 0.0
    z[z <= _EPS] = 0.0
    return _ret(z, s)


# --------------------------------------------------------------------------- #
# risk / odds-ratio based transforms
# --------------------------------------------------------------------------- #

def transf_abt(xi):
    """``transf.abt`` — ``-log(1-x)`` (back-transformed proportion)."""
    a, s = _asarray(xi)
    with np.errstate(divide="ignore", invalid="ignore"):
        z = -np.log(1.0 - a)
    return _ret(z, s)


def transf_ahw(xi):
    """``transf.ahw`` — ``1-(1-x)^(1/3)`` (Arcsine-Hellinger / Whittemore)."""
    a, s = _asarray(xi)
    z = 1.0 - _r_pow(1.0 - a, 1.0 / 3.0)
    return _ret(z, s)


def transf_clestod(xi):
    """``transf.clestod`` — CLES (common language effect size) to Cohen's d."""
    a, s = _asarray(xi)
    with np.errstate(invalid="ignore"):
        z = special.ndtri(a) * math.sqrt(2.0)
    return _ret(z, s)


def transf_lnortod_logis(xi):
    """``transf.lnortod.logis`` — log OR to Cohen's d (logistic)."""
    a, s = _asarray(xi)
    return _ret(math.sqrt(3.0) / math.pi * a, s)


def transf_lnortod_norm(xi):
    """``transf.lnortod.norm`` — log OR to Cohen's d (normal)."""
    a, s = _asarray(xi)
    return _ret(a / 1.65, s)


def transf_lnortord(xi, pc):
    """``transf.lnortord`` — log OR to a risk difference (needs ``pc``)."""
    a, s = _asarray(xi)
    pc_ = _expand1(pc, a.size)
    if np.any((pc_ < 0) | (pc_ > 1)):
        raise ValueError("the control group risk 'pc' must be between 0 and 1")
    e = np.exp(a)
    return _ret(e * pc_ / (1.0 - pc_ + pc_ * e) - pc_, s)


def transf_lnortorr(xi, pc):
    """``transf.lnortorr`` — log OR to a risk ratio (needs ``pc``)."""
    a, s = _asarray(xi)
    pc_ = _expand1(pc, a.size)
    if np.any((pc_ < 0) | (pc_ > 1)):
        raise ValueError("the control group risk 'pc' must be between 0 and 1")
    e = np.exp(a)
    return _ret(e / (pc_ * (e - 1.0) + 1.0), s)


def transf_lnortortet_digby(xi):
    """``transf.lnortortet.digby`` — log OR to tetrachoric r (Digby)."""
    a, s = _asarray(xi)
    e = np.exp(a) ** 0.75
    return _ret((e - 1.0) / (e + 1.0), s)


def transf_lnortortet_pearson(xi):
    """``transf.lnortortet.pearson`` — log OR to tetrachoric r (Pearson)."""
    a, s = _asarray(xi)
    return _ret(np.cos(math.pi / (1.0 + np.sqrt(np.exp(a)))), s)
