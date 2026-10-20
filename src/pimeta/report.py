"""Reporting / layout helpers — the last metafor & meta exports.

Transcribes (functionally, not necessarily character-for-character in the
rendered output) the remaining exports:

``metafor``
    ``aggregate.escalc``, ``cbind.escalc``, ``rbind.escalc``, ``emmprep``,
    ``addpoly``, ``points.regplot``, ``reporter``

``meta``
    ``cidprop``, ``cilayout``, ``forest_dims``, ``longarm``, ``rob``
"""

from __future__ import annotations

import math

import numpy as np
from scipy import stats

from .escalc import EscalcResult

__all__ = [
    "aggregate_escalc", "cbind_escalc", "rbind_escalc", "emmprep",
    "addpoly", "points_regplot", "reporter",
    "cidprop", "cilayout", "forest_dims", "longarm", "rob",
    "CidpropResult", "LongarmResult", "ReportResult",
]

_NAN = float("nan")


# --------------------------------------------------------------------------- #
# cbind.escalc / rbind.escalc / aggregate.escalc
# --------------------------------------------------------------------------- #

def _escalc_cols(x):
    return {"measure": x.measure, "yi": list(x.yi), "vi": list(x.vi),
            "ni": list(x.ni) if x.ni is not None else None}


def cbind_escalc(*results, names=None):
    """``metafor::cbind.escalc`` — combine escalc tables column-wise.

    Each input contributes its ``yi``/``vi`` (and ``ni``) columns under a
    unique name (``yi``, ``yi.1``, ...).  The pooled result is an
    :class:`~.escalc.EscalcResult` carrying all columns.

    Returns:
        dict with ``yi``/``vi``/``ni`` as ``{name: column}`` and ``measure``.
    """
    ks = [len(r.yi) for r in results]
    if not ks:
        raise ValueError("no escalc objects supplied")
    if len(set(ks)) > 1:
        raise ValueError(
            f"arguments imply differing number of rows: {', '.join(map(str, ks))}"
        )
    if names is None:
        names = [f"escalc{i + 1}" for i in range(len(results))]
    if len(names) != len(results):
        raise ValueError("length of 'names' must match the number of objects")

    yi, vi, ni, measure = {}, {}, {}, []
    for nm, r in zip(names, results):
        yi[nm] = [float(v) for v in r.yi]
        vi[nm] = [float(v) for v in r.vi]
        if r.ni is not None:
            ni[nm] = [float(v) for v in r.ni]
        measure.append(r.measure)
    return {"yi": yi, "vi": vi, "ni": ni or None,
            "measure": measure[0] if len(set(measure)) == 1 else measure,
            "k": ks[0]}


def rbind_escalc(*results, slab=None):
    """``metafor::rbind.escalc`` — stack escalc tables row-wise."""
    if not results:
        raise ValueError("no escalc objects supplied")
    yi, vi, ni = [], [], []
    have_ni = all(r.ni is not None for r in results)
    for r in results:
        yi.extend(float(v) for v in r.yi)
        vi.extend(float(v) for v in r.vi)
        if have_ni:
            ni.extend(float(v) for v in r.ni)
    if slab is None:
        slab = [f"Study {i + 1}" for i in range(len(yi))]
    return EscalcResult(measure=results[0].measure, yi=yi, vi=vi,
                        ni=ni or None, k=len(yi)), list(slab)[:len(yi)]


def aggregate_escalc(result, cluster, struct="CS", rho=0.5,
                     weighted=True, fun=None, na_rm=True):
    """``metafor::aggregate.escalc`` — aggregate an escalc table by cluster.

    For each cluster the effects are combined by inverse-variance weighting
    (``weighted=True``) or by an unweighted mean, and the variance of the
    aggregate is ``1 / sum(w)``.  ``struct="CS"`` accounts for the within-cluster
    correlation ``rho`` when forming the weights.

    Args:
        result: an :class:`~.escalc.EscalcResult`.
        cluster: one cluster label per row.
        struct: ``"ID"`` (independent) or ``"CS"`` (compound symmetry).
        rho: within-cluster correlation.
        weighted: inverse-variance weights.
        fun: optional aggregation function ``f(values) -> scalar`` applied to
            ``yi`` instead of the weighted mean.
        na_rm: ignore missing values.

    Returns:
        :class:`~.escalc.EscalcResult` with one row per cluster.
    """
    if struct not in ("ID", "CS", "CAR", "CS+CAR", "CS*CAR"):
        raise ValueError(f"unknown 'struct' specified: {struct!r}")
    yi = np.asarray(result.yi, dtype=float)
    vi = np.asarray(result.vi, dtype=float)
    cl = list(cluster)
    if len(cl) != yi.size:
        raise ValueError(
            f"Length of the variable specified via 'cluster' ({len(cl)}) does "
            f"not match the length of the data ({yi.size})."
        )
    keys = list(dict.fromkeys(cl))

    out_yi, out_vi = [], []
    for key in keys:
        idx = [i for i, v in enumerate(cl) if v == key]
        if na_rm:
            idx = [i for i in idx if not (math.isnan(yi[i])
                                          or math.isnan(vi[i]))]
        if not idx:
            out_yi.append(_NAN)
            out_vi.append(_NAN)
            continue
        y = yi[idx]
        v = vi[idx]
        if fun is not None:
            out_yi.append(float(fun(y)))
            out_vi.append(float(np.sum(v)))
            continue
        if not weighted:
            w = np.ones_like(y)
        elif struct == "ID" or len(idx) == 1:
            w = 1.0 / v
        else:
            m = len(idx)
            s = np.full((m, m), rho)
            np.fill_diagonal(s, 1.0)
            V = np.sqrt(np.outer(v, v)) * s
            try:
                w = np.linalg.solve(V, np.ones(m))
            except np.linalg.LinAlgError:
                w = 1.0 / v
        sw = float(np.sum(w))
        out_yi.append(float(np.sum(w * y) / sw))
        out_vi.append(1.0 / sw)
    return EscalcResult(measure=result.measure, yi=out_yi, vi=out_vi,
                        ni=None, k=len(out_yi))


# --------------------------------------------------------------------------- #
# emmprep
# --------------------------------------------------------------------------- #

def emmprep(result, data=None, df=None, sigma=None, tran=None):
    """``metafor::emmprep`` — prepare a fit for ``emmeans``-style contrasts.

    The R function builds an ``emmGrid`` through ``emmeans::qdrg``.  Python has
    no ``emmeans``, so this returns the *ingredients* ``qdrg`` would consume —
    exactly what is needed to construct a reference grid by hand.

    Args:
        result: a :class:`~.rma_uni.RmaResult`.
        data: the original data (defaults to ``result``'s ``yi``/``vi``).
        df: denominator degrees of freedom (defaults to the model's).
        sigma: residual SD (defaults to ``sqrt(tau2)``).
        tran: back-transformation name (inferred from the effect measure).

    Returns:
        dict with ``formula``, ``data``, ``coef``, ``vcov``, ``df``,
        ``sigma`` and ``tran``.
    """
    b = np.asarray(result.b, dtype=float).ravel()
    vb = np.asarray(result.vb, dtype=float)
    names = ["(Intercept)"] if result.int_only else None
    if names is None:
        names = ["(Intercept)"] + [f"mod{i + 1}"
                                   for i in range(b.size - 1)]

    if df is None:
        ddf = getattr(result, "ddf", None)
        df = float("inf") if (ddf is None or (
            isinstance(ddf, float) and math.isnan(ddf))) else float(ddf)
    if sigma is None:
        sigma = math.sqrt(float(result.tau2))

    if data is None:
        data = {"yi": [float(v) for v in np.asarray(result.yi).ravel()],
                "vi": [float(v) for v in np.asarray(result.vi).ravel()]}

    if tran is None:
        meas = str(getattr(result, "measure", "") or "").upper()
        if meas in {"RR", "OR", "PETO", "IRR", "ROM", "PLN", "IRLN", "HR",
                    "VR", "CVR", "SDLN", "MNLN", "CVLN", "REH"}:
            tran = "log"
        elif meas == "PLO":
            tran = "logit"
        elif meas == "PRZ":
            tran = "probit"
        elif meas == "IRS":
            tran = "sqrt"
        elif meas in {"ZCOR", "ZTET", "ZPHI", "ZPB", "ZBIS"}:
            tran = "r-to-z"

    return {"formula": "~1" if result.int_only else "~mods",
            "data": data, "coef": dict(zip(names, b.tolist())),
            "vcov": vb, "df": df, "sigma": float(sigma), "tran": tran}


# --------------------------------------------------------------------------- #
# addpoly / points.regplot
# --------------------------------------------------------------------------- #

def addpoly(result=None, yi=None, vi=None, ci_lb=None, ci_ub=None,
            rows=-1, level=0.95, annotate=True, mlab=None, ax=None,
            col="gray", border="black", digits=4, transf=None):
    """``metafor::addpoly`` — add summary polygons to a forest plot.

    Draws one diamond per estimate at the given row position(s) on an existing
    (or new) axes, mirroring ``addpoly.rma`` / ``addpoly.default``.

    Args:
        result: optional fitted model; its pooled estimate is drawn.
        yi, vi: effect and variance (when ``result`` is not given).
        ci_lb, ci_ub: explicit confidence bounds (override ``vi``).
        rows: row position(s); negative counts from the bottom.
        level: confidence level (used when ``ci_lb``/``ci_ub`` are absent).
        annotate: label the diamond with the numeric estimate.
        mlab: row label.
        ax: matplotlib axes (created when omitted).
        col, border: diamond fill / edge colour.
        digits: rounding of the annotation.
        transf: optional back-transformation callable.

    Returns:
        :class:`~.plotting.PlotResult`.
    """
    import matplotlib.pyplot as plt
    from matplotlib.patches import Polygon


    if result is not None:
        est = float(result.TE_random if getattr(result, "random", False)
                    else result.TE_common)
        if ci_lb is None:
            ci_lb = float(result.lower_random if getattr(result, "random", False)
                          else result.lower_common)
        if ci_ub is None:
            ci_ub = float(result.upper_random if getattr(result, "random", False)
                          else result.upper_common)
        yi = est
    if yi is None:
        raise ValueError("either 'result' or 'yi' must be supplied")
    est = float(np.asarray(yi, dtype=float).ravel()[0])
    if ci_lb is None or ci_ub is None:
        if vi is None:
            raise ValueError("supply 'ci_lb'/'ci_ub' or 'vi'")
        crit = float(stats.norm.isf((1.0 - level) / 2.0))
        se = math.sqrt(float(np.asarray(vi, dtype=float).ravel()[0]))
        ci_lb, ci_ub = est - crit * se, est + crit * se

    if ax is None:
        _, ax = plt.subplots(figsize=(6, 1.4))
    row = float(rows) if rows >= 0 else float(rows)
    height = 0.18
    poly = Polygon([[ci_lb, row], [est, row + height],
                    [ci_ub, row], [est, row - height]],
                   closed=True, facecolor=col, edgecolor=border, zorder=6)
    ax.add_patch(poly)
    if annotate:
        lo = min(ci_lb, est)
        hi = max(ci_ub, est)
        if transf is not None:
            lo, hi = transf(lo), transf(hi)
        txt = f"{lo:.{digits}g} to {hi:.{digits}g}"
        ax.text(hi, row, " " + txt, va="center", ha="left", fontsize=7)
    if mlab:
        ax.text(0, row, mlab, transform=ax.get_yaxis_transform(),
                va="center", ha="right", fontsize=8)
    ax.axvline(0.0, color="0.6", linewidth=0.8, zorder=1)
    ax.set_yticks([])
    from .plotting import PlotResult as _PR
    return _PR(ax=ax, figure=ax.figure, filename=None)


def points_regplot(result, ax=None):
    """``metafor::points.regplot`` — overlay study points on a regplot.

    Args:
        result: the dict returned by :func:`~.plotting.regplot` **or** a
            ``(xi, yi, ...)`` collection; when given a
            :class:`~.rma_uni.RmaResult` the moderator is taken from ``X``.
        ax: axes to draw on (defaults to the current axes).

    Returns:
        ``ax``.
    """
    if ax is None:
        import matplotlib.pyplot as plt

        ax = plt.gca()
    mod = getattr(result, "X", None)
    yi = getattr(result, "yi", None)
    if mod is None or yi is None:
        raise TypeError("points_regplot needs a fitted model with X and yi")
    mod = np.asarray(mod, dtype=float)
    xi = mod[:, 1] if mod.ndim == 2 and mod.shape[1] > 1 else mod.ravel()
    ax.scatter(xi, np.asarray(yi, dtype=float), s=30, color="black", zorder=4)
    return ax


# --------------------------------------------------------------------------- #
# reporter
# --------------------------------------------------------------------------- #

class ReportResult:
    """Result of :func:`reporter` — a structured, printable meta-analysis report.

    Attributes:
        title: report title.
        sections: list of ``(heading, lines)`` tuples.
        template: the template used (``"JAMA"``, ``"RevMan"`` or ``"none"``).
        text: the rendered plain-text report.
    """

    __slots__ = ("title", "sections", "template", "text", "slab", "meta")

    def __init__(self, title, sections, template, slab, meta):
        self.title = title
        self.sections = sections
        self.template = template
        self.slab = slab
        self.meta = meta
        self.text = self._render()

    def _render(self):
        out = [self.title, "=" * len(self.title), ""]
        for head, lines in self.sections:
            out.append(head)
            out.append("-" * len(head))
            out.extend(lines)
            out.append("")
        return "\n".join(out)

    def __repr__(self):
        return f"ReportResult(sections={len(self.sections)})"

    def save(self, path):
        """Write the plain-text report to ``path``."""
        with open(path, "w", encoding="utf-8") as fh:
            fh.write(self.text)
        return path


def reporter(result, title="Meta-analysis", template="JAMA", digits=4,
             study_results=True, append=False):
    """``metafor::reporter`` — assemble a human-readable summary report.

    R writes a ``.docx``; here the report is returned as structured sections
    plus rendered plain text (``ReportResult.text`` / ``.save()``).

    Args:
        result: a fitted meta-analysis result with ``TE_common``/``TE_random``
            (``MetagenResult``) or ``b``/``se`` (``RmaResult``).
        title: report title.
        template: ``"JAMA"``, ``"RevMan"`` or ``"none"``.
        digits: rounding.
        study_results: include the per-study table.
        append: reserved (kept for API parity).

    Returns:
        :class:`ReportResult`.
    """
    has_meta = hasattr(result, "TE_common") and hasattr(result, "TE_random")
    fmt = lambda v: (f"{float(v):.{digits}g}"  # noqa: E731
                     if v is not None and v == v else "NA")
    sections = []

    if has_meta:
        sections.append(("Studies", [
            f"k = {result.k}" + (" (all)" if True else ""),
        ]))
        sections.append(("Pooled estimates", [
            f"Common effect : {fmt(result.TE_common)} "
            f"[{fmt(result.lower_common)}; {fmt(result.upper_common)}]",
            f"Random effects: {fmt(result.TE_random)} "
            f"[{fmt(result.lower_random)}; {fmt(result.upper_random)}]",
            f"tau^2 = {fmt(getattr(result, 'tau2', None))}, "
            f"I^2 = {fmt(getattr(result, 'I2', None))}",
            f"Q = {fmt(getattr(result, 'Q', None))} "
            f"(df = {getattr(result, 'df_Q', 'NA')}, "
            f"p = {fmt(getattr(result, 'pval_Q', None))})",
        ]))
        if study_results:
            lines = ["study                 TE        seTE"]
            slab = getattr(result, "slab", None) or \
                [f"Study {i + 1}" for i in range(len(result.TE))]
            for s, y, e in zip(slab, result.TE, result.seTE):
                lines.append(f"{str(s)[:20]:<20} {fmt(y):>9} {fmt(e):>10}")
            sections.append(("Individual studies", lines))
        sections.append(("Method", [
            f"tau^2 estimator : {getattr(result, 'method_tau', 'NA')}",
            f"Effect measure  : {getattr(result, 'sm', '') or 'NA'}",
            f"Template        : {template}",
        ]))
    else:
        b = np.asarray(getattr(result, "b", []), dtype=float)
        se = np.asarray(getattr(result, "se", []), dtype=float)
        sections.append(("Coefficients", [
            f"b{i + 1} = {fmt(v)} (se {fmt(se[i])})"
            for i, v in enumerate(b)
        ]))
        sections.append(("Method", [
            f"Estimator : {getattr(result, 'method', 'NA')}",
            f"tau^2     : {fmt(getattr(result, 'tau2', None))}",
            f"Template  : {template}",
        ]))
    return ReportResult(title, sections, template,
                        getattr(result, "slab", None), result)


# --------------------------------------------------------------------------- #
# meta::cilayout / forest_dims / longarm / rob
# --------------------------------------------------------------------------- #

_CI_LAYOUT = {"CIbracket": "[", "CIseparator": ", ",
              "CIlower.blank": False, "CIupper.blank": False}


def cilayout(bracket="[", separator=", ", lower_blank=False, upper_blank=False):
    """``meta::cilayout`` — set how confidence intervals are printed."""
    if bracket not in ("[", "(", "{", ""):
        raise ValueError("bracket must be one of '[', '(', '{', ''")
    _CI_LAYOUT.update({"CIbracket": bracket, "CIseparator": separator,
                       "CIlower.blank": bool(lower_blank),
                       "CIupper.blank": bool(upper_blank)})
    return dict(_CI_LAYOUT)


def format_ci(lower, upper, digits=3):
    """Render a CI using the current :func:`cilayout` settings."""
    lo = "" if _CI_LAYOUT["CIlower.blank"] else f"{lower:.{digits}f}"
    up = "" if _CI_LAYOUT["CIupper.blank"] else f"{upper:.{digits}f}"
    b = _CI_LAYOUT["CIbracket"]
    close = {"[": "]", "(": ")", "{": "}"}.get(b, b)
    return f"{b}{lo}{_CI_LAYOUT['CIseparator']}{up}{close}"


def forest_dims(k=10, label_chars=20, units="in", row_height=0.22,
                header_height=1.2, extra_width=0.3, extra_height=0.8):
    """``meta::forest_dims`` — recommended figure size for a forest plot.

    R measures the rendered grid; this computes the equivalent from the number
    of rows and the label width.

    Args:
        k: number of study rows.
        label_chars: characters used by the longest study label.
        units: ``"in"``, ``"cm"`` or ``"mm"``.
        row_height: height of one row (inches).
        header_height: extra height for the column headers (inches).
        extra_width, extra_height: margins (inches).

    Returns:
        dict with ``width``, ``height`` and ``units``.
    """
    from .utils import inches2units

    width = 6.0 + label_chars * 0.06 + extra_width
    height = k * row_height + header_height + extra_height
    return {"width": float(inches2units(width, units)),
            "height": float(inches2units(height, units)),
            "units": units}


class LongarmResult:
    """Result of :func:`longarm` — one row per study arm."""

    __slots__ = ("rows", "type", "columns")

    def __init__(self, rows, type, columns):
        self.rows = rows
        self.type = type
        self.columns = columns

    def __repr__(self):
        return f"LongarmResult(k={len(self.rows)}, type={self.type!r})"

    def as_dict(self):
        return {"data": self.rows, "type": self.type}

    def to_frame(self):
        try:
            import pandas as pd
        except ImportError:  # pragma: no cover
            return self.rows
        return pd.DataFrame(self.rows, columns=self.columns)


def longarm(treat1, treat2=None, event1=None, n1=None, event2=None, n2=None,
            mean1=None, sd1=None, mean2=None, sd2=None, time1=None, time2=None,
            studlab=None, id1=None, id2=None, keep_duplicated=False):
    """``meta::longarm`` — reshape comparison-level data into arm-level rows.

    Each comparison contributes two rows (arm 1 and arm 2).  The type is
    inferred from the supplied columns: ``event``/``n`` → ``"binary"``,
    ``mean``/``sd``/``n`` → ``"continuous"``, ``event``/``time`` → ``"count"``.

    Args:
        treat1, treat2: treatment labels of the two arms.
        event1, n1, event2, n2: binary data.
        mean1, sd1, mean2, sd2: continuous data.
        time1, time2: person-time (count data).
        studlab: study labels (defaults to ``1..k``).
        id1, id2: optional row identifiers.
        keep_duplicated: keep repeated study/treatment rows.

    Returns:
        :class:`LongarmResult`.
    """
    t1 = [str(v) for v in np.asarray(treat1).ravel()]
    t2 = ([str(v) for v in np.asarray(treat2).ravel()]
          if treat2 is not None else t1)
    k = len(t1)

    def _v(x):
        if x is None:
            return [None] * k
        arr = np.asarray(x, dtype=float).ravel()
        return list(arr) if arr.size == k else [float(arr[0])] * k

    e1, c1, e2, c2 = _v(event1), _v(n1), _v(event2), _v(n2)
    m1, s1, m2, s2 = _v(mean1), _v(sd1), _v(mean2), _v(sd2)
    ti1, ti2 = _v(time1), _v(time2)
    slab = ([str(v) for v in np.asarray(studlab).ravel()] if studlab is not None
            else [str(i + 1) for i in range(k)])

    if event1 is not None and n1 is not None:
        type_ = "binary"
        cols = ["studlab", "treat", "n", "events", "nonevents"]
    elif mean1 is not None:
        type_ = "continuous"
        cols = ["studlab", "treat", "n", "mean", "sd"]
    elif time1 is not None:
        type_ = "count"
        cols = ["studlab", "treat", "n", "events", "time"]
    else:
        raise ValueError(
            "supply either event1/n1 (binary), mean1/sd1/n1 (continuous) or "
            "time1/event1 (count)"
        )

    rows = []
    for i in range(k):
        for arm in (1, 2):
            rec = {"studlab": slab[i]}
            if type_ == "binary":
                ev, nn = (e1[i], c1[i]) if arm == 1 else (e2[i], c2[i])
                rec["treat"] = t1[i] if arm == 1 else t2[i]
                rec["events"] = ev
                rec["nonevents"] = (None if ev is None or nn is None
                                    else nn - ev)
                rec["n"] = nn
            elif type_ == "continuous":
                mn, sd, nn = ((m1[i], s1[i], c1[i]) if arm == 1
                              else (m2[i], s2[i], c2[i]))
                rec["treat"] = t1[i] if arm == 1 else t2[i]
                rec["n"] = nn
                rec["mean"] = mn
                rec["sd"] = sd
            else:
                ev, tt = (e1[i], ti1[i]) if arm == 1 else (e2[i], ti2[i])
                rec["treat"] = t1[i] if arm == 1 else t2[i]
                rec["events"] = ev
                rec["time"] = tt
                rec["n"] = c1[i] if arm == 1 else c2[i]
            rows.append(rec)

    if not keep_duplicated:
        seen = set()
        uniq = []
        for r in rows:
            key = tuple(r.get(c) for c in cols)
            if key in seen:
                continue
            seen.add(key)
            uniq.append(r)
        rows = uniq
    return LongarmResult(rows, type_, cols)


def rob(labels, judgements, domains=None, weights=None, ax=None,
        figsize=(7, 4), filename=None):
    """``meta::rob`` — risk-of-bias summary bar chart.

    Draws a stacked horizontal bar per domain showing the proportion of
    studies judged low / unclear / high risk of bias.

    Args:
        labels: study labels.
        judgements: ``{domain: [judgement per study]}`` or a list of
            per-study lists.  Judgements are ``"low"``/``"unclear"``/``"high"``
            or ``0``/``1``/``2``.
        domains: domain names.
        weights: optional per-study weights (default: all equal).
        ax, figsize, filename: plotting options.

    Returns:
        :class:`~.plotting.PlotResult`.
    """
    import matplotlib.pyplot as plt

    from .plotting import PlotResult

    if isinstance(judgements, dict):
        dom_names = list(judgements)
        mat = [list(judgements[d]) for d in dom_names]
    else:
        mat = [list(row) for row in judgements]
        dom_names = domains or [f"Domain {i + 1}" for i in range(len(mat))]

    # NB: no `True`/`False` keys here — they hash to 1 and 0, so they would
    # silently overwrite the integer mappings below.
    norm = {"low": 0, "unclear": 1, "high": 2,
            0: 0, 1: 1, 2: 2, "0": 0, "1": 1, "2": 2}
    w = (np.ones(len(labels)) if weights is None
         else np.asarray(weights, dtype=float))
    w = w / w.sum()

    colours = ["#2ca02c", "#ff7f0e", "#d62728"]
    names = ["Low risk", "Some concerns", "High risk"]

    if ax is None:
        _, ax = plt.subplots(figsize=figsize)
    left = np.zeros(len(mat))
    y = np.arange(len(mat))
    for level in (0, 1, 2):
        props = np.array([
            float(sum(w[i] for i, j in enumerate(row)
                      if norm.get(j, 1) == level))
            for row in mat
        ])
        ax.barh(y, props, left=left, color=colours[level],
                label=names[level], edgecolor="white")
        left = left + props
    ax.set_yticks(y)
    ax.set_yticklabels(dom_names)
    ax.set_xlabel("Proportion of studies")
    ax.set_xlim(0, 1)
    ax.legend(loc="lower right", fontsize=8)
    ax.invert_yaxis()
    ax.figure.tight_layout()
    if filename:
        ax.figure.savefig(filename)
    return PlotResult(ax=ax, figure=ax.figure, filename=filename)


# --------------------------------------------------------------------------- #
# meta::cidprop
# --------------------------------------------------------------------------- #

class CidpropResult:
    """Result of :func:`cidprop` (``meta::cidprop``)."""

    __slots__ = ("prop_cid_below_null", "prop_cid_above_null", "prop_within_cid",
                 "cid", "cid_below_null", "cid_above_null", "label_cid",
                 "label_cid_below_null", "label_cid_above_null",
                 "small_values", "ref")

    def __init__(self, **kw):
        for k in self.__slots__:
            setattr(self, k, kw.get(k))

    def __repr__(self):
        return (f"CidpropResult(within={self.prop_within_cid!r})")

    def as_dict(self):
        return {k.replace("_", "."): getattr(self, k) for k in self.__slots__}


_RELATIVE = {"RR", "OR", "HR", "IRR", "ROM", "PETO", "PLN", "IRLN", "VR",
             "CVR", "SDLN", "MNLN", "CVLN", "REH"}
_SINGLE = {"PLO", "PLN", "PR", "IR", "IRLN", "IRS", "IRFT", "PFT",
           "MRAW", "MLN", "MNLN", "PRAW", "PLOGIT", "PAS"}


def cidprop(result, cid=None, cid_below_null=None, cid_above_null=None,
            label_cid="", label_cid_below_null=None,
            label_cid_above_null=None, small_values="desirable"):
    """``meta::cidprop`` — proportions outside clinically important differences.

    Uses the prediction interval to compute the probability that a future
    study's effect falls below / above the decision thresholds, exactly as in
    the R function::

        prop.below = pt((cid.below - TE.random) / seTE.predict, df.predict)
        prop.above = pt((cid.above - TE.random) / seTE.predict,
                        df.predict, lower.tail = FALSE)
        prop.within = 1 - max(prop.below) - max(prop.above)

    Args:
        result: a :class:`~.metagen.MetagenResult` with a prediction interval.
        cid: decision thresholds (increasing).
        cid_below_null, cid_above_null: thresholds used for one side only.
        label_cid, label_cid_below_null, label_cid_above_null: labels.
        small_values: ``"desirable"`` or ``"undesirable"``.

    Returns:
        :class:`CidpropResult`.
    """
    sm = str(getattr(result, "sm", "") or "").upper()
    is_relative = sm in _RELATIVE
    is_single = sm in _SINGLE

    ref = float(getattr(result, "null_effect", 0.0) or 0.0)
    if is_relative:
        ref = 1.0

    if small_values not in ("desirable", "undesirable"):
        raise ValueError("small_values must be 'desirable' or 'undesirable'")

    avail = cid is not None and not np.all(np.isnan(np.asarray(cid, float)))
    avail_b = (cid_below_null is not None
               and not np.all(np.isnan(np.asarray(cid_below_null, float))))
    avail_a = (cid_above_null is not None
               and not np.all(np.isnan(np.asarray(cid_above_null, float))))

    if not avail and not avail_b and not avail_a:
        if getattr(result, "null_effect", None) is None:
            raise ValueError(
                "At least one decision threshold (argument 'cid', "
                "'cid.below.null', or 'cid.above.null') must be specified."
            )
        cid = [float(result.null_effect)]
        avail = True

    if avail:
        cid = [float(v) for v in np.asarray(cid, dtype=float).ravel()]
        if any(v != v for v in cid):
            raise ValueError("Missing values not allowed in argument 'cid'.")
        if any(np.diff(cid) <= 0):
            raise ValueError("Values for argument 'cid' must be increasing.")
        if any(v < ref for v in cid) and any(v > ref for v in cid):
            raise ValueError(
                f"All values provided for argument 'cid' must be either smaller "
                f"or larger than reference value of {ref}."
            )
        if is_single:
            cid_below_null = list(cid)
            cid_above_null = list(cid)
        else:
            if all(v == ref for v in cid):
                cid_below_null = list(cid)
                cid_above_null = list(cid)
            elif all(v < ref for v in cid):
                cid_below_null = list(cid)
                cid_above_null = ([1.0 / v for v in reversed(cid)]
                                  if is_relative else [-v for v in reversed(cid)])
            else:
                cid_above_null = list(cid)
                cid_below_null = ([1.0 / v for v in reversed(cid)]
                                  if is_relative else [-v for v in reversed(cid)])
        avail_b = avail_a = True

    if avail_b:
        cid_below_null = [float(v) for v in np.asarray(cid_below_null).ravel()]
        below_t = ([math.log(v) for v in cid_below_null] if is_relative
                   else list(cid_below_null))
    else:
        cid_below_null = [_NAN]
        below_t = [_NAN]
    if avail_a:
        cid_above_null = [float(v) for v in np.asarray(cid_above_null).ravel()]
        above_t = ([math.log(v) for v in cid_above_null] if is_relative
                   else list(cid_above_null))
    else:
        cid_above_null = [_NAN]
        above_t = [_NAN]

    if label_cid is None or label_cid == "":
        label_cid = ""
    if label_cid_below_null is None:
        label_cid_below_null = ("" if len(cid_below_null) == 1
                                else list(range(1, len(cid_below_null) + 1)))
    if label_cid_above_null is None:
        label_cid_above_null = ("" if len(cid_above_null) == 1
                                else list(range(1, len(cid_above_null) + 1)))

    te = float(result.TE_random)
    se = float(result.seTE_predict)
    df = float(getattr(result, "df_predict", result.k - 1))

    prop_below = [float(stats.t.cdf((v - te) / se, df)) for v in below_t]
    prop_above = [float(stats.t.sf((v - te) / se, df)) for v in above_t]

    def _max0(vals):
        vals = [v for v in vals if v == v]
        return max(vals) if vals else 0.0

    within = 1.0 - _max0(prop_below) - _max0(prop_above)
    if abs(within) < np.finfo(float).eps:
        within = 0.0

    return CidpropResult(
        prop_cid_below_null=prop_below,
        prop_cid_above_null=prop_above,
        prop_within_cid=float(within),
        cid=cid, cid_below_null=cid_below_null,
        cid_above_null=cid_above_null, label_cid=label_cid,
        label_cid_below_null=label_cid_below_null,
        label_cid_above_null=label_cid_above_null,
        small_values=small_values, ref=ref,
    )
