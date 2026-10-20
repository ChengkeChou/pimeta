"""Effect-measure conversions �?``meta`` package parity.

Transcriptions of ``meta::asin2ir``, ``asin2p``, ``p2asin``, ``cor2z``,
``z2cor``, ``logit2p``, ``p2logit``, ``smd2or``, ``or2smd``, ``logVR2VE``,
``VE2logVR``, ``rd`` and ``nnt``.

``smd2or``/``or2smd`` return a :class:`~.metagen.MetagenResult` (they call
``metagen()`` in R); ``rd``/``nnt`` accept a :class:`~.metagen.MetagenResult`
and a control-group risk ``p_c``.
"""

from __future__ import annotations

import math

import numpy as np
from scipy import special

__all__ = [
    "asin2ir", "asin2p", "p2asin", "cor2z", "z2cor", "logit2p", "p2logit",
    "smd2or", "or2smd", "logVR2VE", "VE2logVR",
    "rd", "nnt", "RdResult", "NntResult",
]


def _arr(x):
    scalar = np.isscalar(x)
    return np.atleast_1d(np.asarray(x, dtype=float)), scalar


def _ret(y, scalar):
    y = np.asarray(y, dtype=float)
    return float(y[0]) if scalar else y


def asin2ir(x, time=None):
    """``meta::asin2ir`` �?inverse Freeman-Tukey double arcsine to an incidence rate."""
    a, s = _arr(x)
    if np.all(np.isnan(a)):
        return _ret(a, s)
    if time is None:
        raise ValueError("argument 'time' is required")
    t = np.broadcast_to(np.asarray(time, dtype=float), a.shape).copy()
    minimum = 0.5 * (np.sqrt(0.0 / t) + np.sqrt(1.0 / t))
    sel0 = a < minimum
    with np.errstate(divide="ignore", invalid="ignore"):
        res = (1.0 / t - 8.0 * a ** 2 + 16.0 * t * a ** 4) / (16.0 * a ** 2 * t)
    res = np.where(np.isnan(a), np.nan, res)
    res = np.where(sel0, 0.0, res)
    res = np.where(res < 0, 0.0, res)
    return _ret(res, s)


def asin2p(x, n=None):
    """``meta::asin2p`` �?inverse Freeman-Tukey double arcsine to a proportion."""
    a, s = _arr(x)
    if np.all(np.isnan(a)):
        return _ret(a, s)
    if n is None:
        minimum = math.asin(math.sqrt(0.0))
        maximum = math.asin(math.sqrt(1.0))
        nn = None
    else:
        nn = np.broadcast_to(np.asarray(n, dtype=float), a.shape).astype(float).copy()
        minimum = 0.5 * (np.arcsin(np.sqrt(0.0 / (nn + 1)))
                         + np.arcsin(np.sqrt(1.0 / (nn + 1))))
        maximum = 0.5 * (np.arcsin(np.sqrt(nn / (nn + 1)))
                         + np.arcsin(np.sqrt((nn + 1) / (nn + 1))))
    sel0 = a < minimum
    sel1 = a > maximum
    with np.errstate(divide="ignore", invalid="ignore"):
        if nn is None:
            res = np.sin(a) ** 2
        else:
            st = np.sin(2.0 * a)
            res = 0.5 * (1.0 - np.sign(np.cos(2.0 * a))
                         * np.sqrt(1.0 - (st + (st - 1.0 / st) / nn) ** 2))
    res = np.where(np.isnan(a), np.nan, res)
    res = np.where(sel0, 0.0, res)
    res = np.where(sel1, 1.0, res)
    return _ret(res, s)


def p2asin(x):
    """``meta::p2asin`` �?proportion to arcsine."""
    a, s = _arr(x)
    return _ret(np.arcsin(np.sqrt(a)), s)


def cor2z(x):
    """``meta::cor2z`` �?correlation to Fisher's z."""
    a, s = _arr(x)
    return _ret(0.5 * np.log((1.0 + a) / (1.0 - a)), s)


def z2cor(x):
    """``meta::z2cor`` �?Fisher's z to a correlation."""
    a, s = _arr(x)
    return _ret(np.tanh(a), s)


def logit2p(x):
    """``meta::logit2p`` �?logit to a proportion."""
    a, s = _arr(x)
    return _ret(special.expit(a), s)


def p2logit(x):
    """``meta::p2logit`` �?proportion to logit."""
    a, s = _arr(x)
    return _ret(special.logit(a), s)


def logVR2VE(x):
    """``meta::logVR2VE`` �?log variance ratio to a percentage reduction."""
    a, s = _arr(x)
    return _ret(100.0 * (1.0 - np.exp(a)), s)


def VE2logVR(x):
    """``meta::VE2logVR`` �?percentage reduction to a log variance ratio."""
    a, s = _arr(x)
    return _ret(np.log(1.0 - a / 100.0), s)


def smd2or(smd, se_smd, method="HH"):
    """``meta::smd2or`` �?SMD to log odds ratio, then pool with ``metagen``.

    Args:
        smd: standardized mean differences.
        se_smd: their standard errors.
        method: ``"HH"`` (Hasselblad-Hedges, ``pi/sqrt(3)``) or ``"CS"``
            (Chinn, ``1.65``).

    Returns:
        :class:`~.metagen.MetagenResult` on the log-OR scale (``sm="OR"``).
    """
    from .metagen import metagen

    if method not in ("HH", "CS"):
        raise ValueError("argument 'method' must be 'HH' or 'CS'")
    smd = np.asarray(smd, dtype=float)
    se_smd = np.asarray(se_smd, dtype=float)
    fac = math.pi / math.sqrt(3.0) if method == "HH" else 1.65
    return metagen(smd * fac, se_smd * fac, sm="OR")


def or2smd(lnOR, selnOR, method="HH"):
    """``meta::or2smd`` �?log odds ratio to SMD, then pool with ``metagen``.

    Args:
        lnOR: log odds ratios.
        selnOR: their standard errors.
        method: ``"HH"`` (``sqrt(3)/pi``) or ``"CS"`` (``1/1.65``).

    Returns:
        :class:`~.metagen.MetagenResult` on the SMD scale (``sm="SMD"``).
    """
    from .metagen import metagen

    if method not in ("HH", "CS"):
        raise ValueError("argument 'method' must be 'HH' or 'CS'")
    lnOR = np.asarray(lnOR, dtype=float)
    selnOR = np.asarray(selnOR, dtype=float)
    fac = math.sqrt(3.0) / math.pi if method == "HH" else 1.0 / 1.65
    return metagen(lnOR * fac, selnOR * fac, sm="SMD")


# --------------------------------------------------------------------------- #
# risk difference / number needed to treat
# --------------------------------------------------------------------------- #

def _logRR2rd(te, pc):
    """meta:::logRR2rd — RD = p.e - p.c with p.e = exp(logRR) * p.c."""
    return np.exp(te) * pc - pc


def _logOR2rd(te, pc):
    """meta:::logOR2rd — RD = p.e - p.c with the OR-derived p.e."""
    e = np.exp(te)
    return e * pc / (1.0 - pc + e * pc) - pc


def _logHR2rd(te, pc, pscale=1.0):
    """meta:::logHR2rd — RD = p.e - p.c assuming exponentially distributed events."""
    e = np.exp(te)
    return pc ** e - pc


def rd(result, p_c=None, common=True, random=True, small_values="desirable"):
    """``meta::rd`` �?risk difference / absolute risk reduction from a pooled result.

    Args:
        result: :class:`~.metagen.MetagenResult` with ``sm`` in
            ``{"RD", "RR", "OR", "HR"}``.
        p_c: control-group risk(s). Defaults to ``0.1 .. 0.9`` by 0.1.
        common: include the common-effect estimate.
        random: include the random-effects estimate.
        small_values: ``"desirable"`` or ``"undesirable"`` (sign convention).

    Returns:
        :class:`RdResult` with ``common``/``random`` dicts of arrays.
    """
    from .metagen import MetagenResult

    if not isinstance(result, MetagenResult):
        raise TypeError("argument 'result' must be a MetagenResult")
    sm = getattr(result, "sm", "")
    if sm not in ("RD", "RR", "OR", "HR"):
        raise ValueError(
            "calculation of ARRs/ABIs only possible for RD, RR, OR or HR "
            "as summary measure"
        )
    if p_c is None:
        p_c = np.arange(0.1, 1.0, 0.1)
        if sm == "HR":
            p_c = p_c[::-1]
    pc = np.atleast_1d(np.asarray(p_c, dtype=float))

    def _conv(te):
        te = float(np.asarray(te).ravel()[0])
        if sm == "RD":
            return np.full(pc.size, te)
        if sm == "RR":
            return _logRR2rd(te, pc)
        if sm == "OR":
            return _logOR2rd(te, pc)
        return _logHR2rd(te, pc)

    out = {}
    if common:
        out["common"] = {
            "TE": float(np.asarray(result.TE_common).ravel()[0]),
            "lower": float(np.asarray(result.lower_common).ravel()[0]),
            "upper": float(np.asarray(result.upper_common).ravel()[0]),
            "p_c": pc,
            "RD": _conv(result.TE_common),
            "lower.RD": _conv(result.lower_common),
            "upper.RD": _conv(result.upper_common),
        }
    if random:
        out["random"] = {
            "TE": float(np.asarray(result.TE_random).ravel()[0]),
            "lower": float(np.asarray(result.lower_random).ravel()[0]),
            "upper": float(np.asarray(result.upper_random).ravel()[0]),
            "p_c": pc,
            "RD": _conv(result.TE_random),
            "lower.RD": _conv(result.lower_random),
            "upper.RD": _conv(result.upper_random),
        }
    return RdResult(common=out.get("common"), random=out.get("random"),
                    sm=sm, small_values=small_values)


def nnt(result, p_c=None, common=True, random=True, small_values="desirable"):
    """``meta::nnt`` �?number needed to treat from a pooled result.

    See :func:`rd` for the arguments; ``NNT = 1 / RD`` with the sign convention
    from ``small_values``.
    """
    r = rd(result, p_c=p_c, common=common, random=random,
           small_values=small_values)

    def _inv(d):
        # meta's logOR2nnt / logRR2nnt are 1/(p.c - p.e) = -1/RD
        with np.errstate(divide="ignore", invalid="ignore"):
            n = -1.0 / d["RD"]
            lo = -1.0 / d["lower.RD"]
            up = -1.0 / d["upper.RD"]
        return {"p_c": d["p_c"], "NNT": n, "lower.NNT": lo, "upper.NNT": up}

    out = {}
    if r.common is not None:
        out["nnt.common"] = _inv(r.common)
    if r.random is not None:
        out["nnt.random"] = _inv(r.random)
    if small_values == "undesirable":
        for k in ("nnt.common", "nnt.random"):
            if k in out:
                d = out[k]
                tmp_lo = d["lower.NNT"]
                d["NNT"] = -d["NNT"]
                d["lower.NNT"] = -d["upper.NNT"]
                d["upper.NNT"] = -tmp_lo
    return NntResult(sm=r.sm, common=r.common, random=r.random,
                     nnt_common=out.get("nnt.common"),
                     nnt_random=out.get("nnt.random"))


class RdResult:
    """Container returned by :func:`rd`."""

    def __init__(self, common, random, sm, small_values):
        self.common = common
        self.random = random
        self.sm = sm
        self.small_values = small_values

    def __repr__(self) -> str:
        return f"RdResult(sm={self.sm!r})"


class NntResult:
    """Container returned by :func:`nnt`."""

    def __init__(self, sm, common, random, nnt_common, nnt_random):
        self.sm = sm
        self.common = common
        self.random = random
        self.nnt_common = nnt_common
        self.nnt_random = nnt_random

    def __repr__(self) -> str:
        return f"NntResult(sm={self.sm!r})"
