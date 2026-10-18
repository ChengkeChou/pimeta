"""Composite meta-analysis helpers — ``meta`` package parity.

Implements ``meta::metabind`` (combine several meta-analyses into one) and
``meta::metamerge`` (merge two meta-analyses of the same set of studies).
Both operate on the result objects produced by this package and re-pool the
study-level ``TE``/``seTE`` pairs with :func:`~.metagen.metagen`.
"""

from __future__ import annotations

import math
import warnings

import numpy as np

__all__ = ["metabind", "metamerge", "metaadd", "metacr", "MetabindResult",
           "MetaaddResult", "MetacrResult"]


class MetabindResult:
    """Result of :func:`metabind` — a combined meta-analysis.

    Attributes:
        meta: the pooled :class:`~.metagen.MetagenResult`.
        TE: concatenated study effects.
        seTE: concatenated standard errors.
        studlab: concatenated study labels.
        sm: effect measure (taken from the first input).
        parts: number of input meta-analyses that were combined.
    """

    def __init__(self, meta, TE, seTE, studlab, sm, parts):
        self.meta = meta
        self.TE = TE
        self.seTE = seTE
        self.studlab = studlab
        self.sm = sm
        self.parts = parts

    def __repr__(self) -> str:
        return f"MetabindResult(k={len(self.TE)}, parts={self.parts})"


def _te_se(res, name):
    """Extract ``(TE, seTE)`` from a result object."""
    te = getattr(res, "TE", None)
    se = getattr(res, "seTE", None)
    if te is None or se is None:
        te = getattr(res, "te", None)
        se = getattr(res, "seTE", None)
    if te is None or se is None:
        raise ValueError(
            f"{name} does not expose study-level 'TE'/'seTE'; "
            "metabind() needs the per-study estimates"
        )
    return np.asarray(te, dtype=float).ravel(), np.asarray(se, dtype=float).ravel()


def metabind(*results, method_tau="REML", sm=None, studlab=None, level=0.95):
    """``meta::metabind`` — combine several meta-analyses into a single one.

    The study-level estimates of all inputs are concatenated and re-pooled with
    :func:`~.metagen.metagen`, which is what ``meta::metabind`` does (it keeps
    the individual study rows rather than the summary estimates).

    Args:
        *results: meta-analysis result objects exposing ``TE``/``seTE``.
        method_tau: ``tau^2`` estimator passed to :func:`metagen`.
        sm: effect measure; defaults to the ``sm`` of the first input.
        studlab: optional study labels (one per input study).
        level: confidence level (0.95).

    Returns:
        :class:`MetabindResult`.

    Raises:
        ValueError: when no inputs are given or an input lacks study-level data.
    """
    from .metagen import metagen

    if not results:
        raise ValueError("metabind() requires at least one meta-analysis")

    TE, seTE, labels = [], [], []
    for i, res in enumerate(results):
        te_i, se_i = _te_se(res, f"results[{i}]")
        TE.append(te_i)
        seTE.append(se_i)
        lab = getattr(res, "studlab", None)
        if studlab is not None:
            labels.extend(
                [str(v) for v in np.asarray(studlab).ravel()[len(labels):len(labels) + te_i.size]]
            )
        elif lab is not None and len(lab) == te_i.size:
            labels.extend([str(v) for v in lab])
        else:
            labels.extend([f"{i + 1}.{j + 1}" for j in range(te_i.size)])

    TE = np.concatenate(TE)
    seTE = np.concatenate(seTE)
    if sm is None:
        sm = getattr(results[0], "sm", "") or ""
    if studlab is not None and len(labels) != TE.size:
        labels = [str(v) for v in np.asarray(studlab).ravel()]

    meta = metagen(TE, seTE, studlab=labels, sm=sm, method_tau=method_tau,
                   level=level)
    return MetabindResult(meta=meta, TE=TE, seTE=seTE, studlab=labels,
                          sm=sm, parts=len(results))


def metamerge(x, y, method_tau="REML", level=0.95):
    """``meta::metamerge`` — merge two meta-analyses of the same studies.

    ``meta::metamerge`` is used when the *same* studies were analysed twice
    (e.g. two effect measures, or a subgroup split); the two study lists are
    aligned by position and pooled jointly, and the separate analyses are kept
    in the returned object.

    Args:
        x, y: meta-analysis result objects exposing ``TE``/``seTE``.
        method_tau: ``tau^2`` estimator.
        level: confidence level.

    Returns:
        :class:`MetabindResult` whose ``meta`` is the pooled combination and
        which additionally carries ``x``/``y`` for the two inputs.
    """
    tx, sx = _te_se(x, "x")
    ty, sy = _te_se(y, "y")
    if tx.size != ty.size:
        raise ValueError(
            "metamerge() requires the same number of studies in both analyses "
            f"(got {tx.size} and {ty.size})"
        )
    out = metabind(x, y, method_tau=method_tau, level=level)
    out.x = x
    out.y = y
    return out


# --------------------------------------------------------------------------- #
# meta::metaadd — append pooled results to an existing meta-analysis
# --------------------------------------------------------------------------- #

class MetaaddResult:
    """Result of :func:`metaadd` (``meta::metaadd``).

    ``meta::metaadd`` appends additional *pooled* results (from another
    meta-analysis or supplied directly) to a ``meta`` object.  In R those
    fields are plain vectors, so the accessors here return lists whose first
    element is the original value and whose remaining elements are the added
    ones.

    Attributes:
        base: the original :class:`~.metagen.MetagenResult`.
        added: list of ``dict`` rows, one per appended result, with the keys
            ``type``, ``TE``, ``se``, ``lower``, ``upper``, ``statistic``,
            ``pval``, ``df``, ``method``, ``method_ci``, ``text``.
        common, random, prediction: whether the corresponding model is present
            after the additions.
        is_metaadd: always ``True`` (mirrors the ``"metaadd"`` class tag).
    """

    def __init__(self, base, added, common, random, prediction):
        self.base = base
        self.added = added
        self.common = common
        self.random = random
        self.prediction = prediction
        self.is_metaadd = True

    # -- R-style vector accessors ------------------------------------------- #
    def _collect(self, key, attr):
        vals = []
        if self.common and key in ("common",):
            vals.append(getattr(self.base, attr + "_common", None))
        elif self.random and key in ("random",):
            vals.append(getattr(self.base, attr + "_random", None))
        elif key == "tau2":
            vals.append(getattr(self.base, "tau2", None))
        elif key == "prediction":
            vals.append(getattr(self.base, attr + "_predict", None))
        vals.extend(r[key.removesuffix("_predict")]
                    for r in self.added if r["type"] == key)
        return vals

    @property
    def TE_common(self):
        return self._vec("common", "TE")

    @property
    def seTE_common(self):
        return self._vec("common", "se")

    @property
    def lower_common(self):
        return self._vec("common", "lower")

    @property
    def upper_common(self):
        return self._vec("common", "upper")

    @property
    def TE_random(self):
        return self._vec("random", "TE")

    @property
    def seTE_random(self):
        return self._vec("random", "se")

    @property
    def lower_random(self):
        return self._vec("random", "lower")

    @property
    def upper_random(self):
        return self._vec("random", "upper")

    @property
    def tau2(self):
        return self._vec("tau2", "TE")

    def _vec(self, kind, field):
        base_attr = {
            ("common", "TE"): "TE_common",
            ("common", "se"): "seTE_common",
            ("common", "lower"): "lower_common",
            ("common", "upper"): "upper_common",
            ("random", "TE"): "TE_random",
            ("random", "se"): "seTE_random",
            ("random", "lower"): "lower_random",
            ("random", "upper"): "upper_random",
            ("tau2", "TE"): "tau2",
            ("prediction", "se"): "seTE_predict",
            ("prediction", "lower"): "lower_predict",
            ("prediction", "upper"): "upper_predict",
        }.get((kind, field))
        out = []
        if base_attr is not None:
            v = getattr(self.base, base_attr, None)
            if v is not None:
                out.append(v)
        out.extend(r[field] for r in self.added if r["type"] == kind)
        return out

    def __repr__(self):
        return (f"MetaaddResult(k={getattr(self.base, 'k', '?')}, "
                f"added={len(self.added)})")

    def as_dict(self):
        return {
            "TE.common": self.TE_common, "seTE.common": self.seTE_common,
            "lower.common": self.lower_common, "upper.common": self.upper_common,
            "TE.random": self.TE_random, "seTE.random": self.seTE_random,
            "lower.random": self.lower_random, "upper.random": self.upper_random,
            "tau2": self.tau2, "common": self.common, "random": self.random,
            "prediction": self.prediction,
        }


_TYPES = ("common", "random", "prediction", "tau2")


def metaadd(result, type=None, TE=None, lower=None, upper=None,
            statistic=None, pval=None, df=None, se=None, method="",
            method_ci="", text="Added result", data=None):
    """``meta::metaadd`` — append pooled results to a meta-analysis.

    Either pass another meta-analysis as ``data`` (its common-effect,
    random-effects, tau^2 and prediction results are appended) or supply the
    values directly through ``type``/``TE``/``lower``/``upper``/``se``.

    Args:
        result: a :class:`~.metagen.MetagenResult`.
        type: ``"common"``, ``"random"``, ``"prediction"`` or ``"tau2"``.
        TE, lower, upper, statistic, pval, df, se: the result to append.
        method, method_ci: labels for the estimation method.
        text: label used when printing.
        data: alternatively a :class:`~.metagen.MetagenResult` whose pooled
            results are appended.

    Returns:
        :class:`MetaaddResult`.
    """
    k = getattr(result, "k", None)
    if k is not None and k < 2:
        raise ValueError(
            "Meta-analysis object provided in argument 'result' must combine "
            "at least two studies."
        )

    n_rows = 1
    if isinstance(type, (list, tuple)):
        n_rows = len(type)

    def _expand(v):
        if v is None:
            return [None] * n_rows
        if isinstance(v, (list, tuple, np.ndarray)):
            return list(v)
        return [v] * n_rows

    if isinstance(method, (list, tuple)):
        method_v = list(method)
    else:
        method_v = [method] * n_rows
    if isinstance(method_ci, (list, tuple)):
        method_ci_v = list(method_ci)
    else:
        method_ci_v = [method_ci] * n_rows
    if isinstance(text, (list, tuple)):
        text_v = list(text)
    else:
        text_v = [text] * n_rows

    added = []
    is_meta = isinstance(data, (MetaaddResult,)) or (
        data is not None and hasattr(data, "TE_common")
    )

    if data is not None and not is_meta:
        raise TypeError("argument 'data' must be a meta-analysis result")

    if is_meta:
        src = data.base if isinstance(data, MetaaddResult) else data
        has_random = getattr(src, "TE_random", None) is not None
        has_common = getattr(src, "TE_common", None) is not None
        if has_common or not has_random:
            added.append(dict(
                type="common", TE=getattr(src, "TE_common", None),
                se=getattr(src, "seTE_common", None),
                lower=getattr(src, "lower_common", None),
                upper=getattr(src, "upper_common", None),
                statistic=getattr(src, "statistic_common", None),
                pval=getattr(src, "pval_common", None), df=None,
                method=getattr(src, "method_tau", ""), method_ci="",
                text="", phi=None,
            ))
        if has_random:
            added.append(dict(
                type="random", TE=getattr(src, "TE_random", None),
                se=getattr(src, "seTE_random", None),
                lower=getattr(src, "lower_random", None),
                upper=getattr(src, "upper_random", None),
                statistic=getattr(src, "statistic_random", None),
                pval=getattr(src, "pval_random", None),
                df=getattr(src, "df_random", None),
                method=getattr(src, "method_tau", ""), method_ci="",
                text="", phi=None,
            ))
            added.append(dict(
                type="tau2", TE=getattr(src, "tau2", None),
                se=getattr(src, "se_tau2", None),
                lower=getattr(src, "lower_tau2", None),
                upper=getattr(src, "upper_tau2", None),
                statistic=None, pval=None, df=None,
                method=getattr(src, "method_tau", ""), method_ci="",
                text="", phi=getattr(src, "phi", None),
            ))
        if getattr(src, "prediction", False):
            added.append(dict(
                type="prediction", TE=None,
                se=getattr(src, "seTE_predict", None),
                lower=getattr(src, "lower_predict", None),
                upper=getattr(src, "upper_predict", None),
                statistic=None, pval=None,
                df=getattr(src, "df_predict", None),
                method=getattr(src, "method_predict", ""), method_ci="",
                text="", phi=None,
            ))
        if not added:
            raise ValueError("No pooled results in meta-analysis object.")
    else:
        types = _expand(type)
        if all(t is None or t == "" for t in types):
            # nothing supplied -> R returns the object unchanged
            return result
        TE_v = _expand(TE)
        se_v = _expand(se)
        lower_v = _expand(lower)
        upper_v = _expand(upper)
        stat_v = _expand(statistic)
        pval_v = _expand(pval)
        df_v = _expand(df)
        for i, t in enumerate(types):
            if t not in _TYPES:
                raise ValueError(
                    f"type must be one of {_TYPES}, got {t!r}"
                )
            if t in ("common", "random", "tau2") and TE_v[i] is None:
                raise ValueError(
                    "Argument 'TE' must be provided for common effect, random "
                    "effects model, or between-study variance."
                )
            if t in ("common", "random", "prediction") and (
                    lower_v[i] is None or upper_v[i] is None):
                raise ValueError(
                    "Arguments 'lower' and 'upper' must be provided."
                )
            added.append(dict(
                type=t, TE=TE_v[i], se=se_v[i], lower=lower_v[i],
                upper=upper_v[i], statistic=stat_v[i], pval=pval_v[i],
                df=df_v[i], method=method_v[i], method_ci=method_ci_v[i],
                text=text_v[i], phi=None,
            ))

    common = any(r["type"] == "common" for r in added) or bool(
        getattr(result, "TE_common", None) is not None)
    random = any(r["type"] in ("random", "tau2") for r in added) or bool(
        getattr(result, "TE_random", None) is not None)
    prediction = any(r["type"] == "prediction" for r in added)

    return MetaaddResult(result, added, common, random, prediction)


# --------------------------------------------------------------------------- #
# meta::metacr — re-run one comparison/outcome of a Cochrane (RevMan) review
# --------------------------------------------------------------------------- #

class MetacrResult:
    """Result of :func:`metacr` (``meta::metacr``).

    ``meta::metacr`` operates on a RevMan 5 (``.rm5``) or Cochrane data
    (``.cdir``) object, selects one comparison and outcome, and re-runs the
    appropriate ``meta`` analysis.  This port takes the *data frame* that such
    an object carries (the columns ``comp.no``, ``outcome.no``, ``type``,
    ``sm``, ``method``, ``event.e``, ``n.e``, ``event.c``, ``n.c``,
    ``mean.e``, ``sd.e``, ``mean.c``, ``sd.c``, ``O.E``, ``V``, ``TE``,
    ``seTE``) instead of parsing the RevMan file.

    Attributes:
        meta: the fitted analysis object.
        comp_no, outcome_no: the selected comparison and outcome.
        type: ``"D"`` (binary), ``"C"`` (continuous), ``"P"`` (generic
            Peto / O-E), or ``"I"`` (inverse variance).
        sm: effect measure used.
        method: ``"Inverse"``, ``"MH"`` or ``"Peto"``.
        swap_events: whether the 2x2 tables were swapped.
        n_studies: number of studies in the selected comparison.
    """

    def __init__(self, meta, comp_no, outcome_no, type, sm, method,
                 swap_events, n_studies):
        self.meta = meta
        self.comp_no = comp_no
        self.outcome_no = outcome_no
        self.type = type
        self.sm = sm
        self.method = method
        self.swap_events = swap_events
        self.n_studies = n_studies

    def __repr__(self):
        return (f"MetacrResult(comp={self.comp_no}, outcome={self.outcome_no}, "
                f"type={self.type!r}, k={self.n_studies})")


def _as_records(x):
    if x is None:
        raise ValueError("argument 'x' is required")
    if hasattr(x, "to_dict") and hasattr(x, "columns"):     # pandas DataFrame
        return x.to_dict("records")
    if isinstance(x, list):
        return list(x)
    raise TypeError("argument 'x' must be a pandas DataFrame or a list of dicts")


def _uniq(rows, key):
    vals = []
    for r in rows:
        v = r.get(key)
        if v not in vals:
            vals.append(v)
    return vals


def _col(rows, key):
    out = []
    for r in rows:
        v = r.get(key)
        out.append(float(v) if v is not None else float("nan"))
    return out


def metacr(x, comp_no=1, outcome_no=1, method=None, sm=None, common=True,
           random=True, method_tau="DL", level=0.05, level_ma=0.95,
           swap_events=None, prediction=False, level_predict=0.05):
    """``meta::metacr`` — refit one comparison/outcome of a review.

    Args:
        x: the review's data — a ``pandas.DataFrame`` or a list of dicts with
            the RevMan column names.
        comp_no, outcome_no: which comparison and outcome to analyse.
        method: ``"Inverse"``, ``"MH"`` or ``"Peto"`` (default: from the data).
        sm: effect measure (default: from the data).  ``"PETO_OR"`` is mapped
            to ``"OR"`` exactly like R.
        common, random, prediction: which models to fit.
        method_tau: heterogeneity estimator (R's ``metacr`` default is ``"DL"``).
        level, level_ma: confidence levels for the study-level and pooled
            intervals.
        swap_events: swap the event counts of the two arms.
        level_predict: level of the prediction interval.

    Returns:
        :class:`MetacrResult`, or ``None`` when the selection is empty.
    """
    from .metabin import metabin
    from .metacont import metacont
    from .metagen import metagen

    rows = _as_records(x)
    sel = [r for r in rows
           if r.get("comp.no") == comp_no and r.get("outcome.no") == outcome_no]
    if not sel:
        warnings.warn(
            f"No data available for comp.no = {comp_no} and "
            f"outcome.no = {outcome_no}.",
            stacklevel=2,
        )
        return None

    types = _uniq(sel, "type")
    type_ = types[0] if len(types) == 1 else None
    if type_ not in ("D", "C", "P", "I"):
        raise ValueError(
            f"the selected rows must share one 'type' in D/C/P/I, got {types}"
        )

    if method is None:
        methods = _uniq(sel, "method")
        method = methods[0] if len(methods) == 1 else "Inverse"
    method = str(method)
    if method not in ("Inverse", "MH", "Peto"):
        raise ValueError("method must be 'Inverse', 'MH' or 'Peto'")

    if sm is None:
        sms = _uniq(sel, "sm")
        sm = sms[0] if len(sms) == 1 else ""
    if sm == "PETO_OR":
        sm = "OR"
    if sm == "OTHER":
        warnings.warn('Meta-analysis not possible for sm = "OTHER".',
                      stacklevel=2)
        return None

    studlab = [r.get("studlab") for r in sel]

    if type_ == "D":
        event_e = _col(sel, "event.e")
        n_e = _col(sel, "n.e")
        event_c = _col(sel, "event.c")
        n_c = _col(sel, "n.c")
        if swap_events is None:
            swap_events = bool(_uniq(sel, "swap.events")[0]) \
                if _uniq(sel, "swap.events")[0] is not None else False
        if swap_events:
            event_e = [n - e for e, n in zip(event_e, n_e)]
            event_c = [n - e for e, n in zip(event_c, n_c)]
        meta = metabin(event_e, n_e, event_c, n_c, sm=sm or "RR",
                       method=method, random=random, method_tau=method_tau,
                       alpha=level_ma)
    elif type_ == "C":
        meta = metacont(_col(sel, "n.e"), _col(sel, "mean.e"),
                        _col(sel, "sd.e"), _col(sel, "n.c"),
                        _col(sel, "mean.c"), _col(sel, "sd.c"),
                        sm=sm or "MD", random=random, method_tau=method_tau,
                        alpha=level_ma)
    elif type_ == "P":
        o_e = _col(sel, "O.E")
        v = _col(sel, "V")
        meta = metagen([a / b for a, b in zip(o_e, v)],
                       [math.sqrt(1.0 / b) for b in v], studlab=studlab,
                       sm=sm or "", method_tau=method_tau, common=common,
                       random=random, prediction=prediction, level=level_ma,
                       level_predict=level_predict)
    else:  # type_ == "I"
        if method == "Peto":
            o_e = _col(sel, "O.E")
            v = _col(sel, "V")
            meta = metagen([a / b for a, b in zip(o_e, v)],
                           [math.sqrt(1.0 / b) for b in v], studlab=studlab,
                           sm=sm or "", method_tau=method_tau, common=common,
                           random=random, prediction=prediction,
                           level=level_ma, level_predict=level_predict)
        else:
            meta = metagen(_col(sel, "TE"), _col(sel, "seTE"),
                           studlab=studlab, sm=sm or "",
                           method_tau=method_tau, common=common, random=random,
                           prediction=prediction, level=level_ma,
                           level_predict=level_predict)

    return MetacrResult(meta=meta, comp_no=comp_no, outcome_no=outcome_no,
                        type=type_, sm=sm, method=method,
                        swap_events=swap_events, n_studies=len(sel))
