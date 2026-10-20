"""``conv.*`` / ``to.*`` — metafor's data-conversion helpers.

Implements ``metafor::conv.delta``, ``conv.wald``, ``to.long``, ``to.wide`` and
``to.table``. The heavier ``conv.fivenum`` (five-number-summary reconstruction)
and ``conv.2x2`` (arbitrary reported statistics -> 2x2 tables) are not yet
transcribed.
"""

from __future__ import annotations

import builtins
import math
import sys

import numpy as np
from scipy import stats

__all__ = [
    "conv_delta", "conv_wald", "conv_fivenum", "to_long", "to_wide", "to_table",
    "ConvDeltaResult", "ConvWaldResult", "ConvFivenumResult",
    "conv_2x2", "Conv2x2Result",
]


class ConvDeltaResult:
    """Result of :func:`conv_delta` (``metafor::conv.delta``)."""

    def __init__(self, yi, vi):
        self.yi = yi
        self.vi = vi

    def __repr__(self) -> str:
        return f"ConvDeltaResult(yi={self.yi}, vi={self.vi})"


class ConvWaldResult:
    """Result of :func:`conv_wald` (``metafor::conv.wald``)."""

    def __init__(self, yi, vi):
        self.yi = yi
        self.vi = vi

    def __repr__(self) -> str:
        return f"ConvWaldResult(yi={self.yi}, vi={self.vi})"


def _apply(fun, arr):
    """Evaluate ``fun`` on an array, falling back to a per-element loop.

    ``transf`` is conventionally vectorised (``numpy.exp``), but users often
    pass a scalar function from :mod:`math`.  Calling it with an array raised a
    bare ``TypeError: only 0-dimensional arrays...`` from the middle of the
    derivative code, so both forms are accepted.
    """
    try:
        return np.asarray(fun(arr), dtype=float)
    except (TypeError, ValueError):
        a = np.asarray(arr)
        flat = [fun(v.item() if hasattr(v, "item") else v) for v in a.ravel()]
        return np.asarray(flat, dtype=float).reshape(a.shape)


def _num_deriv(fun, x, h=None):
    """Derivative of a scalar-vector function.

    Uses the complex-step method (machine precision, no subtractive
    cancellation) when the function accepts complex input, and falls back to a
    central difference otherwise.
    """
    x = np.asarray(x, dtype=float)
    try:
        d = np.imag(np.asarray(fun(x + 1e-20j), dtype=complex)) / 1e-20
        d = np.asarray(d, dtype=float).ravel()
        if d.size == x.size and np.all(np.isfinite(d)):
            return d
    except Exception:
        pass
    if h is None:
        h = np.finfo(float).eps ** (1.0 / 3.0) * np.maximum(np.abs(x), 1e-8)
    return (_apply(fun, x + h) - _apply(fun, x - h)) / (2.0 * h)


def conv_delta(yi, vi, transf):
    """``metafor::conv.delta`` — transform an effect size and its variance.

    Applies ``transf`` to ``yi`` and propagates ``vi`` with the first-order
    delta method, i.e. ``vi_new = transf'(yi)^2 * vi``.

    Args:
        yi: effect sizes.
        vi: sampling variances.
        transf: callable applied elementwise to ``yi``.

    Returns:
        :class:`ConvDeltaResult` with the transformed ``yi`` and ``vi``.
    """
    yi = np.asarray(yi, dtype=float).ravel()
    vi = np.asarray(vi, dtype=float).ravel()
    if yi.size != vi.size:
        raise ValueError("length of 'yi' and 'vi' are not the same")
    yi_new = _apply(transf, yi).ravel()
    d = _num_deriv(transf, yi)
    return ConvDeltaResult(yi_new, (d ** 2) * vi)


def _level_pct(level):
    """Normalise a confidence level to a percentage.

    metafor's ``chklevel()`` accepts both 95 and 0.95 and rescales the latter.
    Without this, ``conv_wald(..., level=0.95)`` silently interpreted 0.95 as
    "0.95 percent" and returned a variance three orders of magnitude too
    large.
    """
    level = float(level)
    if not (0.0 < level < 100.0):
        raise ValueError(
            "argument 'level' must be a confidence level, e.g. 95 or 0.95")
    return level * 100.0 if level < 1.0 else level


def conv_wald(out, ci_lb, ci_ub, level=95.0, transf=None):
    """``metafor::conv.wald`` — recover ``yi``/``vi`` from an estimate and its CI.

    ``vi = ((ci.ub - ci.lb) / (2 * qnorm(1 - (1 - level/100)/2)))^2``.

    Args:
        out: reported estimates (on the analysis scale unless ``transf`` is
            given, in which case ``transf`` maps them onto that scale).
        ci_lb, ci_ub: the corresponding confidence limits.
        level: confidence level in **percent** (metafor convention, 95 = 95%).
        transf: optional transform applied to ``out``/``ci_lb``/``ci_ub``.

    Returns:
        :class:`ConvWaldResult`.
    """
    out = np.asarray(out, dtype=float).ravel()
    lo = np.asarray(ci_lb, dtype=float).ravel()
    hi = np.asarray(ci_ub, dtype=float).ravel()
    if not (out.size == lo.size == hi.size):
        raise ValueError("'out', 'ci.lb' and 'ci.ub' must have the same length")
    level = _level_pct(level)
    crit = stats.norm.isf((1.0 - level / 100.0) / 2.0)
    vi = ((hi - lo) / (2.0 * crit)) ** 2
    yi = out if transf is None else np.asarray(transf(out), dtype=float).ravel()
    return ConvWaldResult(yi, vi)


# --------------------------------------------------------------------------- #
# conv.fivenum
# --------------------------------------------------------------------------- #

class ConvFivenumResult:
    """Result of :func:`conv_fivenum` (``metafor::conv.fivenum``)."""

    def __init__(self, mean, sd, case, sig):
        self.mean = mean
        self.sd = sd
        self.case = case
        self.sig = sig

    def __repr__(self) -> str:
        return f"ConvFivenumResult(mean={self.mean}, sd={self.sd})"


def _isna(v):
    return v is None or (isinstance(v, float) and math.isnan(v)) or np.isnan(v)


def conv_fivenum(min=None, q1=None, median=None, q3=None, max=None, n=None,
                 method="luo/wan/shi", test=True):
    """``metafor::conv.fivenum`` — reconstruct mean/SD from a five-number summary.

    Transcribes metafor's three cases (``case1``: min/median/max only,
    ``case2``: q1/median/q3 only, ``case3``: all five) for the
    ``"luo/wan/shi"`` (default), ``"wan2014"``, ``"hozo2005"`` and
    ``"bland2015"`` estimators.

    The default (Luo-Wan-Shi) uses the weighted combinations of the range- and
    IQR-based approximations that metafor implements; ``"wan2014"`` uses the
    unweighted Wan et al. (2014) formulas (for ``case3`` the SD is the equally
    weighted mean of the range- and IQR-based estimates).

    Args:
        min, q1, median, q3, max: the five-number summary (arrays or scalars).
        n: sample size(s).
        method: estimator name.
        test: when ``True`` (metafor default), skip observations whose summary
            is significantly skewed (the estimated value is then ``NaN``).

    Returns:
        :class:`ConvFivenumResult` with ``mean``, ``sd``, ``case`` and ``sig``.

    Note:
        Only ``dist="norm"`` is supported; metafor's log-normal back-transform
        (``dist="lnorm"``) and the ``estmeansd``/``metaBLUE`` based methods
        (``qe``/``bc``/``mln``/``blue``) are not transcribed.
    """
    if all(v is None for v in (min, q1, median, q3, max)):
        raise ValueError(
            "must specify at least some of these arguments: "
            "'min', 'q1', 'median', 'q3', 'max'"
        )
    if method not in ("luo/wan/shi", "wan2014", "hozo2005", "bland2015"):
        raise ValueError(
            "method must be 'luo/wan/shi', 'wan2014', 'hozo2005' or 'bland2015'"
        )

    def _vec(v, name):
        if v is None:
            return None
        return np.atleast_1d(np.asarray(v, dtype=float))

    arrs = {k: _vec(v, k) for k, v in
            (("min", min), ("q1", q1), ("median", median), ("q3", q3), ("max", max))}
    sizes = [a.size for a in arrs.values() if a is not None]
    if len(set(sizes)) > 1:
        raise ValueError("supplied data vectors are not all of the same length")
    k = sizes[0]
    n_arr = np.broadcast_to(np.asarray(n, dtype=float), (k,)).copy()

    def col(name):
        a = arrs[name]
        if a is None:
            return np.full(k, np.nan)
        return np.broadcast_to(a, (k,)).astype(float).copy()

    mn, q1v, md, q3v, mx = (col("min"), col("q1"), col("median"),
                            col("q3"), col("max"))

    means = np.full(k, np.nan)
    sds = np.full(k, np.nan)
    case = np.zeros(k, dtype=int)
    sig = np.zeros(k, dtype=bool)

    for i in range(k):
        has5 = not (np.isnan(mn[i]) or np.isnan(md[i]) or np.isnan(mx[i]))
        has3 = not (np.isnan(q1v[i]) or np.isnan(md[i]) or np.isnan(q3v[i]))
        case[i] = 3 if (has5 and has3) else (1 if has5 else (2 if has3 else 0))
        if case[i] == 0:
            continue
        nn = n_arr[i]
        if np.isnan(nn):
            continue

        vals = [mn[i], q1v[i], md[i], q3v[i], mx[i]]
        vals = [v for v in vals if not np.isnan(v)]
        if any(vals[j] > vals[j + 1] for j in range(len(vals) - 1)):
            raise ValueError(
                f"found 'min <= q1 <= median <= q3 <= max' not true in row {i}"
            )

        xi = 2.0 * stats.norm.ppf((nn - 0.375) / (nn + 0.25))
        eta = 2.0 * stats.norm.ppf((0.75 * nn - 0.125) / (nn + 0.25))

        if case[i] == 1:
            tv = abs((mn[i] + mx[i] - 2 * md[i]) / (mx[i] - mn[i]))
            crit = 1.0 / math.log(nn + 9.0) + 2.5 / (nn + 1.0)
            sig[i] = tv >= crit
            if test and sig[i]:
                continue
            if method == "luo/wan/shi":
                w = 4.0 / (4.0 + nn ** 0.75)
                means[i] = w * (mn[i] + mx[i]) / 2.0 + (1.0 - w) * md[i]
                sds[i] = (mx[i] - mn[i]) / xi
            elif method == "wan2014":
                means[i] = (mn[i] + 2 * md[i] + mx[i]) / 4.0
                sds[i] = (mx[i] - mn[i]) / xi
            elif method == "hozo2005":
                if nn <= 25:
                    means[i] = (mn[i] + 2 * md[i] + mx[i]) / 4.0
                else:
                    means[i] = md[i]
                if nn <= 15:
                    sds[i] = (1.0 / math.sqrt(12.0)) * math.sqrt(
                        (mn[i] - 2 * md[i] + mx[i]) ** 2 / 4.0
                        + (mx[i] - mn[i]) ** 2
                    )
                elif nn <= 70:
                    sds[i] = (mx[i] - mn[i]) / 4.0
                else:
                    sds[i] = (mx[i] - mn[i]) / 6.0
            elif method == "bland2015":
                means[i] = (mn[i] + 2 * md[i] + mx[i]) / 4.0
                sds[i] = (mx[i] - mn[i]) / xi

        elif case[i] == 2:
            tv = abs((q1v[i] + q3v[i] - 2 * md[i]) / (q3v[i] - q1v[i]))
            crit = 2.65 / math.sqrt(nn) - 6.0 / nn ** 2
            sig[i] = tv >= crit
            if test and sig[i]:
                continue
            if method == "luo/wan/shi":
                w = 0.7 + 0.39 / nn
                means[i] = w * (q1v[i] + q3v[i]) / 2.0 + (1.0 - w) * md[i]
                sds[i] = (q3v[i] - q1v[i]) / eta
            elif method == "wan2014":
                means[i] = (q1v[i] + md[i] + q3v[i]) / 3.0
                sds[i] = (q3v[i] - q1v[i]) / eta
            # hozo2005 does not cover the q1/median/q3-only case -> NaN

        else:  # case 3
            tv = builtins.max(
                2.65 * math.log(0.6 * nn) / math.sqrt(nn)
                * abs((mn[i] + mx[i] - 2 * md[i]) / (mx[i] - mn[i])),
                abs((q1v[i] + q3v[i] - 2 * md[i]) / (q3v[i] - q1v[i])),
            )
            crit = 3.0 / math.sqrt(nn) - 40.0 / nn ** 3
            sig[i] = tv >= crit
            if test and sig[i]:
                continue
            if method == "luo/wan/shi":
                w1 = 2.2 / (2.2 + nn ** 0.75)
                w2 = 0.7 - 0.72 / nn ** 0.55
                means[i] = (w1 * (mn[i] + mx[i]) / 2.0
                            + w2 * (q1v[i] + q3v[i]) / 2.0
                            + (1.0 - w1 - w2) * md[i])
                w = 1.0 / (1.0 + 0.07 * nn ** 0.6)
                sds[i] = (w * (mx[i] - mn[i]) / xi
                          + (1.0 - w) * (q3v[i] - q1v[i]) / eta)
            elif method == "wan2014":
                means[i] = (mn[i] + 2 * q1v[i] + 2 * md[i] + 2 * q3v[i]
                            + mx[i]) / 8.0
                sds[i] = 0.5 * ((mx[i] - mn[i]) / xi
                                + (q3v[i] - q1v[i]) / eta)
            elif method == "hozo2005":
                # hozo2005 does not cover the five-number case -> NaN
                pass
            elif method == "bland2015":
                means[i] = (mn[i] + 2 * q1v[i] + 2 * md[i] + 2 * q3v[i]
                            + mx[i]) / 8.0
                sds[i] = math.sqrt(
                    (mn[i] ** 2 + 2 * q1v[i] ** 2 + 2 * md[i] ** 2
                     + 2 * q3v[i] ** 2 + mx[i] ** 2) / 16.0
                    + (mn[i] * q1v[i] + q1v[i] * md[i] + md[i] * q3v[i]
                       + q3v[i] * mx[i]) / 8.0
                    - (mn[i] + 2 * q1v[i] + 2 * md[i] + 2 * q3v[i]
                       + mx[i]) ** 2 / 64.0
                )

    scalar = all(v is None or np.ndim(v) == 0 for v in
                 (min, q1, median, q3, max))
    if scalar:
        return ConvFivenumResult(float(means[0]), float(sds[0]), int(case[0]),
                                 bool(sig[0]))
    return ConvFivenumResult(means, sds, case, sig)


# --------------------------------------------------------------------------- #
# to.long / to.wide / to.table
# --------------------------------------------------------------------------- #

def to_long(measure="RR", ai=None, bi=None, ci=None, di=None,
            n1i=None, n2i=None, event=None, n=None, grp=None, study=None):
    """``metafor::to.long`` — reshape arm-level data from wide to long format.

    Accepts either a 2x2 layout (``ai``/``bi``/``ci``/``di`` or
    ``n1i``/``n2i``/``event``) and returns a long table with one row per arm.

    Args:
        measure: effect-size measure (kept for parity with metafor).
        ai, bi, ci, di: 2x2 cell counts (experimental/control events and
            non-events).
        n1i, n2i: arm sample sizes (used when ``event``/``n`` are given).
        event, n: generic per-arm events and totals.
        grp: optional arm labels (defaults to ``1, 2``).
        study: optional study labels.

    Returns:
        ``dict`` of columns with 2 rows per study.
    """
    if ai is not None and bi is not None and ci is not None and di is not None:
        ai = np.asarray(ai, dtype=float).ravel()
        bi = np.asarray(bi, dtype=float).ravel()
        ci = np.asarray(ci, dtype=float).ravel()
        di = np.asarray(di, dtype=float).ravel()
        k = ai.size
        ev = np.empty(2 * k)
        nn = np.empty(2 * k)
        ev[0::2] = ai
        nn[0::2] = ai + bi
        ev[1::2] = ci
        nn[1::2] = ci + di
    elif event is not None and n is not None:
        ev = np.asarray(event, dtype=float).ravel()
        nn = np.asarray(n, dtype=float).ravel()
        k = ev.size // 2
    else:
        raise ValueError(
            "must specify 'ai'/'bi'/'ci'/'di' or 'event'/'n'"
        )

    if study is None:
        study_out = [str(i + 1) for i in range(k) for _ in range(2)]
    else:
        study_out = [str(s) for s in study for _ in range(2)]
    if grp is None:
        grp_out = [g for _ in range(k) for g in (1, 2)]
    else:
        grp_out = [g for _ in range(k) for g in list(grp)[:2]]

    return {
        "study": study_out,
        "group": grp_out,
        "event": ev,
        "n": nn,
        "measure": measure,
    }


def to_wide(data, study="study", group="group", event="event", n="n"):
    """``metafor::to.wide`` — reshape a long arm-level table back to wide format.

    Args:
        data: mapping of column name -> values (the output of :func:`to_long`).
        study, group, event, n: column names to use.

    Returns:
        ``dict`` with ``ai``/``bi``/``ci``/``di`` and ``n1i``/``n2i``.
    """
    studies = list(data[study])
    groups = list(data[group])
    ev = np.asarray(data[event], dtype=float)
    nn = np.asarray(data[n], dtype=float)

    order: list[object] = []
    seen = set()
    for s in studies:
        if s not in seen:
            seen.add(s)
            order.append(s)
    grp_levels = sorted(set(groups))

    ai, bi, ci, di, n1i, n2i = [], [], [], [], [], []
    for s in order:
        idx = [i for i, v in enumerate(studies) if v == s]
        row = {groups[i]: i for i in idx}
        i1 = row.get(grp_levels[0])
        i2 = row.get(grp_levels[1]) if len(grp_levels) > 1 else None
        if i1 is None or i2 is None:
            ai.append(np.nan); ci.append(np.nan)
            bi.append(np.nan); di.append(np.nan)
            n1i.append(np.nan); n2i.append(np.nan)
            continue
        e1, n_1 = ev[i1], nn[i1]
        e2, n_2 = ev[i2], nn[i2]
        ai.append(e1); n1i.append(n_1)
        ci.append(e2); n2i.append(n_2)
        bi.append(n_1 - e1); di.append(n_2 - e2)

    return {
        "study": order,
        "ai": np.array(ai), "bi": np.array(bi),
        "ci": np.array(ci), "di": np.array(di),
        "n1i": np.array(n1i), "n2i": np.array(n2i),
    }


def to_table(ai, bi, ci, di, study=None):
    """``metafor::to.table`` — build per-study 2x2 contingency tables.

    Args:
        ai, bi, ci, di: 2x2 cell counts.
        study: optional study labels.

    Returns:
        List of ``2x2`` nested lists, one per study.
    """
    ai = np.asarray(ai, dtype=float).ravel()
    bi = np.asarray(bi, dtype=float).ravel()
    ci = np.asarray(ci, dtype=float).ravel()
    di = np.asarray(di, dtype=float).ravel()
    return [
        [[float(a), float(b)], [float(c), float(d)]]
        for a, b, c, d in zip(ai, bi, ci, di)
    ]


# --------------------------------------------------------------------------- #
# metafor::conv.2x2 — reconstruct cell frequencies from summary statistics
# --------------------------------------------------------------------------- #

class Conv2x2Result:
    """Result of :func:`conv_2x2` (``metafor::conv.2x2``)."""

    __slots__ = ("ai", "bi", "ci", "di", "var_names")

    def __init__(self, ai, bi, ci, di, var_names):
        self.ai = ai
        self.bi = bi
        self.ci = ci
        self.di = di
        self.var_names = var_names

    def __repr__(self):
        return f"Conv2x2Result(k={len(self.ai)})"

    def as_dict(self):
        return {n: v for n, v in zip(self.var_names,
                                     (self.ai, self.bi, self.ci, self.di))}

    def to_frame(self):
        """Return a pandas ``DataFrame`` (or a list of dicts if pandas is absent)."""
        try:
            import pandas as pd
        except ImportError:  # pragma: no cover
            return [{n: v[i] for n, v in zip(
                self.var_names, (self.ai, self.bi, self.ci, self.di))}
                for i in range(len(self.ai))]
        return pd.DataFrame(self.as_dict())


_NAN = float("nan")


def _is_na(v):
    return v is None or (isinstance(v, float) and math.isnan(v))


def _rec2x2diag(sens, spec, ppv, ni, round_=True):
    """``metafor:::.rec2x2diag`` — cell counts implied by sens/spec/PPV."""
    den = sens * (1.0 - ppv) + ppv * (1.0 - spec)
    ai = ni * sens * ppv * (1.0 - spec) / den
    bi = ni * sens * (1.0 - ppv) * (1.0 - spec) / den
    ci = ni * (1.0 - sens) * ppv * (1.0 - spec) / den
    di = ni * sens * spec * (1.0 - ppv) / den
    out = [ai, bi, ci, di]
    if round_:
        out = [float(round(x)) for x in out]
    return out


def _funconv2x2diag(par, obs, ni):
    """``metafor:::.funconv2x2diag`` — squared-error loss of the reconstruction."""
    if par[0] * (1.0 - par[2]) + par[2] * (1.0 - par[1]) <= sys.float_info.epsilon:
        return 10.0
    rec = _rec2x2diag(par[0], par[1], par[2], ni, round_=False)
    ni_rec = sum(rec)
    sens_rec = rec[0] / (rec[0] + rec[2])
    spec_rec = rec[3] / (rec[1] + rec[3])
    ppv_rec = rec[0] / (rec[0] + rec[1])
    npv_rec = rec[3] / (rec[2] + rec[3])
    return ((obs[0] - sens_rec) ** 2 + (obs[1] - spec_rec) ** 2
            + (obs[2] - ppv_rec) ** 2 + (obs[3] - npv_rec) ** 2
            + (ni - ni_rec) ** 2)


def _largestremaindermethod(x, n):
    """``metafor:::.largestremaindermethod`` — round while preserving the total."""
    xfloor = [math.floor(v) for v in x]
    remainder = int(round(n - sum(xfloor)))
    x_int = list(xfloor)
    if remainder > 0:
        frac = [x[k] - xfloor[k] for k in range(len(x))]
        idx = sorted(range(len(x)), key=lambda k: (-frac[k], k))
        for k in idx[:remainder]:
            x_int[k] += 1
    return [float(v) for v in x_int]


def conv_2x2(ori=None, ri=None, x2i=None, ni=None, n1i=None, n2i=None,
             sens=None, spec=None, ppv=None, npv=None, correct=True,
             drop01=True, method="optim",
             var_names=("ai", "bi", "ci", "di")):
    """``metafor::conv.2x2`` — reconstruct 2x2 cell frequencies.

    Cell counts can be recovered from either an odds ratio (``ori``), a phi
    coefficient (``ri``) or a chi-square statistic (``x2i``) together with the
    marginal counts ``n1i``/``n2i``, or from at least three of
    sensitivity/specificity/PPV/NPV.  This is a faithful transcription of
    ``metafor::conv.2x2`` including the ``"simple"``, ``"optim"`` and
    ``"search"`` reconstruction methods.

    Args:
        ori, ri, x2i: odds ratio, phi coefficient, chi-square statistic.
        ni: total sample size.
        n1i, n2i: marginal counts (e.g. number of cases and controls).
        sens, spec, ppv, npv: diagnostic accuracy measures.
        correct: continuity correction for the ``x2i`` -> ``ri`` conversion.
        drop01: skip tables whose sens/spec/PPV/NPV equal 0 or 1.
        method: ``"simple"``, ``"optim"`` (default) or ``"search"``.
        var_names: names of the four output cells.

    Returns:
        :class:`Conv2x2Result` with ``ai``, ``bi``, ``ci``, ``di``.
    """
    from scipy import optimize as _opt

    if len(var_names) != 4:
        raise ValueError("argument 'var_names' must be of length 4")
    method = str(method)
    if method not in ("simple", "optim", "search"):
        raise ValueError("method must be 'simple', 'optim' or 'search'")

    inputs = [ori, ri, x2i, sens, spec, ppv, npv, ni, n1i, n2i]
    lengths = {len(v) for v in inputs if v is not None
               and not isinstance(v, (int, float))}
    if len(lengths) > 1:
        raise ValueError("Supplied data vectors are not all of the same length")
    k = max(lengths) if lengths else 1

    def _vec(v):
        if v is None:
            return [_NAN] * k
        if isinstance(v, (int, float)):
            return [float(v)] * k
        return [float(x) if x is not None else _NAN for x in v]

    ori, ri, x2i, sens, spec = (_vec(v) for v in (ori, ri, x2i, sens, spec))
    ppv, npv, ni, n1i, n2i = (_vec(v) for v in (ppv, npv, ni, n1i, n2i))

    if isinstance(correct, bool):
        correct = [correct] * k
    else:
        correct = [True if c is None else bool(c) for c in correct]
        if len(correct) != k:
            raise ValueError(
                f"Length of the 'correct' argument ({len(correct)}) does not "
                f"match the length of the data ({k})"
            )

    ni = [float(round(v)) if not _is_na(v) else _NAN for v in ni]
    n1i = [float(round(v)) if not _is_na(v) else _NAN for v in n1i]
    n2i = [float(round(v)) if not _is_na(v) else _NAN for v in n2i]

    for arr, msg in (
        ([v for v in ni + n1i + n2i if not _is_na(v) and v < 0],
         "One or more sample sizes or marginal counts are negative."),
        ([v for v in [a - b for a, b in zip(n1i, ni)] if not _is_na(v) and v > 0]
         + [v for v in [a - b for a, b in zip(n2i, ni)] if not _is_na(v) and v > 0],
         "One or more marginal counts are larger than the sample sizes."),
        ([v for v in ri if not _is_na(v) and abs(v) > 1],
         "One or more phi coefficients are > 1 or < -1."),
        ([v for v in sens if not _is_na(v) and (v < 0 or v > 1)],
         "One or more sensitivity values are < 0 or > 1."),
        ([v for v in spec if not _is_na(v) and (v < 0 or v > 1)],
         "One or more specificity values are < 0 or > 1."),
        ([v for v in ppv if not _is_na(v) and (v < 0 or v > 1)],
         "One or more positive predictive values are < 0 or > 1."),
        ([v for v in npv if not _is_na(v) and (v < 0 or v > 1)],
         "One or more negative predictive values are < 0 or > 1."),
    ):
        if arr:
            raise ValueError(msg)

    p1i = [n1 / n if not _is_na(n) else _NAN for n1, n in zip(n1i, ni)]
    p2i = [n2 / n if not _is_na(n) else _NAN for n2, n in zip(n2i, ni)]
    p11i = [_NAN] * k
    ai = [_NAN] * k
    bi = [_NAN] * k
    ci = [_NAN] * k
    di = [_NAN] * k

    for i in range(k):
        n_i = ni[i]
        if _is_na(n_i):
            continue
        n_diag = sum(1 for v in (sens[i], spec[i], ppv[i], npv[i])
                     if not _is_na(v))
        if n_diag >= 3:
            if drop01 and any(not _is_na(v) and v in (0.0, 1.0)
                              for v in (sens[i], spec[i], ppv[i], npv[i])):
                continue
            if _is_na(sens[i]):
                sens[i] = ((1 - spec[i]) * ppv[i] * npv[i]
                           / ((1 - spec[i]) * ppv[i] * npv[i]
                              + spec[i] * (1 - ppv[i]) * (1 - npv[i])))
            if _is_na(spec[i]):
                spec[i] = ((1 - sens[i]) * ppv[i] * npv[i]
                           / ((1 - sens[i]) * ppv[i] * npv[i]
                              + sens[i] * (1 - ppv[i]) * (1 - npv[i])))
            if _is_na(ppv[i]):
                ppv[i] = (sens[i] * spec[i] * (1 - npv[i])
                          / (sens[i] * spec[i] * (1 - npv[i])
                             + (1 - sens[i]) * (1 - spec[i]) * npv[i]))
            if _is_na(npv[i]):
                npv[i] = (sens[i] * spec[i] * (1 - ppv[i])
                          / (sens[i] * spec[i] * (1 - ppv[i])
                             + (1 - sens[i]) * (1 - spec[i]) * ppv[i]))
            if sens[i] * (1 - ppv[i]) + ppv[i] * (1 - spec[i]) == 0:
                continue
            if method == "simple":
                tmp = _rec2x2diag(sens[i], spec[i], ppv[i], n_i, round_=False)
            elif method == "optim":
                obs = [sens[i], spec[i], ppv[i], npv[i]]
                start = [sens[i], spec[i], ppv[i]]
                try:
                    res = _opt.minimize(
                        _funconv2x2diag, start, args=(obs, n_i),
                        method="BFGS", options={"gtol": 1e-8, "maxiter": 1000},
                    )
                    if not np.all(np.isfinite(res.x)):
                        tmp = [_NAN] * 4
                    else:
                        tmp = _rec2x2diag(res.x[0], res.x[1], res.x[2], n_i,
                                          round_=False)
                except Exception:
                    tmp = [_NAN] * 4
            else:  # search
                obs = [sens[i], spec[i], ppv[i], npv[i]]

                def _grid(v):
                    return np.linspace(max(0.0, v - 0.005),
                                       min(1.0, v + 0.005), 21)

                gs, gp, gq = _grid(sens[i]), _grid(spec[i]), _grid(ppv[i])
                best = None
                best_loss = None
                for s_val in gs:
                    for p_val in gp:
                        for q_val in gq:
                            loss = _funconv2x2diag([s_val, p_val, q_val],
                                                   obs, n_i)
                            if best_loss is None or loss < best_loss:
                                best_loss = loss
                                best = (s_val, p_val, q_val)
                tmp = _rec2x2diag(best[0], best[1], best[2], n_i, round_=False)
            tmp = _largestremaindermethod(tmp, n_i)
            ai[i], bi[i], ci[i], di[i] = tmp
        else:
            if _is_na(n1i[i]) or _is_na(n2i[i]):
                continue
            if not _is_na(ori[i]):
                if ori[i] == 1:
                    p11i[i] = n1i[i] * n2i[i] / n_i ** 2
                else:
                    p1_ = p1i[i]
                    p2_ = 1.0 - p1i[i]
                    q_1 = p2i[i]
                    x = ori[i] * (p1_ + q_1) + p2_ - q_1
                    y = math.sqrt(x ** 2
                                  - 4.0 * p1_ * q_1 * ori[i] * (ori[i] - 1.0))
                    p11i[i] = (x - y) / (2.0 * (ori[i] - 1.0))
            if _is_na(ri[i]) and not _is_na(x2i[i]):
                if correct[i]:
                    ri[i] = math.copysign(
                        math.sqrt(abs(x2i[i]) / n_i)
                        + n_i / (2.0 * math.sqrt(n1i[i] * (n_i - n1i[i])
                                                 * n2i[i] * (n_i - n2i[i]))),
                        x2i[i],
                    )
                else:
                    ri[i] = math.copysign(math.sqrt(abs(x2i[i]) / n_i), x2i[i])
            if _is_na(p11i[i]) and not _is_na(ri[i]):
                p11i[i] = (p1i[i] * p2i[i]
                           + ri[i] * math.sqrt(p1i[i] * (1 - p1i[i])
                                               * p2i[i] * (1 - p2i[i])))
            ai[i] = float(round(n_i * p11i[i]))
            bi[i] = n1i[i] - ai[i]
            ci[i] = n2i[i] - ai[i]
            di[i] = n_i - ai[i] - bi[i] - ci[i]

    hasneg = [not _is_na(a) and (a < 0 or b < 0 or c < 0 or d < 0)
              for a, b, c, d in zip(ai, bi, ci, di)]
    if any(hasneg):
        for arr in (ai, bi, ci, di):
            for i, flag in enumerate(hasneg):
                if flag:
                    arr[i] = _NAN

    return Conv2x2Result(ai, bi, ci, di, tuple(var_names))
