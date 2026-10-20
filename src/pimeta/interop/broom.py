"""``tidy`` / ``glance`` / ``augment`` — the broom protocol.

R users know these three functions from `broom <https://broom.tidymodels.org/>`_:
one table of coefficients (``tidy``), one row of model-level statistics
(``glance``), and one row per observation with diagnostics (``augment``).

The protocol matters more than the library.  Anything that consumes tidy tables
— ``pandas``, ``polars``, a ``gt``/``great_tables`` renderer, a CSV writer, a
plotting layer that maps ``term`` to an axis — works with pimeta results once
these three functions exist, no adapter required.

All three return whatever :func:`pimeta.interop.tables.to_frame` produces for
``backend`` (a ``dict`` by default when pandas is absent, so nothing here
hard-depends on a dataframe library).
"""

from __future__ import annotations

import math

import numpy as np

from .base import coefficient_names
from .tables import to_frame

__all__ = ["tidy", "glance", "augment"]


# --------------------------------------------------------------------------- #
# field discovery
# --------------------------------------------------------------------------- #
_ESTIMATE = ("b", "estimate", "TE", "pooled_est", "pooled", "Effect")
_SE = ("se", "seTE", "se_est", "seTE_random", "SE")
_LOWER = ("ci_lb", "ci_lower", "lower", "ci_lb_random")
_UPPER = ("ci_ub", "ci_upper", "upper", "ci_ub_random")
_STAT = ("zval", "z", "statistic", "tval", "t")
_PVAL = ("pval", "p", "p_value", "pval_random")


def _vector(obj, names) -> np.ndarray | None:
    """Return the first present attribute in ``names`` as a 1-D float array."""
    for name in names:
        value = getattr(obj, name, None)
        if value is None:
            continue
        arr = np.atleast_1d(np.asarray(value, dtype=float))
        if arr.ndim == 1 and arr.size:
            return arr
    return None


def _labels(obj, n: int, coefficient: bool) -> list[str]:
    """Build row labels: coefficient names, study labels, or 1-based indices."""
    if coefficient:
        try:
            names = coefficient_names(obj, n)
            if len(names) == n:
                return [str(v) for v in names]
        except Exception:                             # pragma: no cover - defensive
            pass
    slab = getattr(obj, "slab", None)
    if isinstance(slab, (list, tuple)) and len(slab) == n:
        return [str(v) for v in slab]
    treat1 = getattr(obj, "treat1", None)
    treat2 = getattr(obj, "treat2", None)
    if (isinstance(treat1, (list, tuple)) and isinstance(treat2, (list, tuple))
            and len(treat1) == n and len(treat2) == n):
        return [f"{a} vs {b}" for a, b in zip(treat1, treat2)]
    return [str(i + 1) for i in range(n)]


def _ci(obj, est: np.ndarray, se: np.ndarray | None, level: float,
        lb_names=_LOWER, ub_names=_UPPER):
    """Return (lower, upper) at ``level``, reusing stored limits when possible."""
    stored_lb = _vector(obj, lb_names)
    stored_ub = _vector(obj, ub_names)
    stored_level = float(getattr(obj, "level", 0.95) or 0.95)
    if (stored_lb is not None and stored_ub is not None
            and stored_lb.size == est.size and stored_ub.size == est.size
            and abs(stored_level - level) < 1e-12):
        return stored_lb, stored_ub
    if se is None or se.size != est.size:
        z = np.full(est.size, np.nan)
        return est + z, est + z
    from scipy import stats

    test = str(getattr(obj, "test", "z") or "z")
    ddf = getattr(obj, "ddf", None)
    if test in ("knha", "adhoc", "t") and ddf:
        crit = float(stats.t.isf((1.0 - level) / 2.0, float(ddf)))
    else:
        crit = float(stats.norm.isf((1.0 - level) / 2.0))
    return est - crit * se, est + crit * se


# --------------------------------------------------------------------------- #
# tidy
# --------------------------------------------------------------------------- #
def tidy(result, *, conf_int: bool = True, conf_level: float | None = None,
         exponentiate: bool = False, component: str = "beta",
         backend: str = "auto"):
    """One row per coefficient (or per comparison / per study).

    Mirrors ``broom::tidy()``: the table carries ``term``, ``estimate``,
    ``std.error``, ``statistic``, ``p.value``, ``conf.low`` and ``conf.high``.

    Args:
        result: any fitted pimeta result object.
        conf_int: include ``conf.low`` / ``conf.high``.
        conf_level: override the model's confidence level (0-1).
        exponentiate: back-transform ``estimate``, ``conf.low`` and ``conf.high``
            with ``exp()`` — useful for log-scale ratio measures.
        component: for scale models (``rma_ls``), ``"beta"`` (location),
            ``"alpha"`` (scale) or ``"both"``.
        backend: dataframe backend, see :mod:`pimeta.interop.tables`.

    Returns:
        A table with one row per coefficient (or per study for list-like
        results), in the requested backend.

    Raises:
        TypeError: when the object has no recognisable estimate vector.
    """
    if component not in ("beta", "alpha", "both"):
        raise ValueError("'component' must be 'beta', 'alpha' or 'both'")

    blocks: list[tuple[str, dict]] = []
    if component in ("beta", "both"):
        block = _tidy_block(result, "beta", conf_int, conf_level, exponentiate)
        if block is not None:
            blocks.append(("location", block))
    if component in ("alpha", "both"):
        block = _tidy_block(result, "alpha", conf_int, conf_level, exponentiate)
        if block is not None:
            blocks.append(("scale", block))

    if not blocks:
        raise TypeError(
            f"{type(result).__name__} has no estimate vector that 'tidy' "
            "recognises; try to_dataframe() for a field dump")

    if len(blocks) == 1:
        return to_frame(blocks[0][1], backend=backend)

    merged: dict[str, list] = {}
    for label, block in blocks:
        block["component"] = [label] * len(block["term"])
    order: list[str] = []
    for _, block in blocks:
        for key in block:
            if key not in order:
                order.append(key)
    for _, block in blocks:
        n = len(block["term"])
        for key in order:
            merged.setdefault(key, []).extend(block.get(key, [None] * n))
    return to_frame(merged, backend=backend)


def _tidy_block(result, component, conf_int, conf_level, exponentiate):
    if component == "alpha":
        est = _vector(result, ("alpha",))
        se = _vector(result, ("se_alpha", "sealpha"))
        stat = _vector(result, ("zval_alpha", "zalpha"))
        pval = _vector(result, ("pval_alpha", "palpha"))
        if est is None:
            return None
        names = getattr(result, "scale_names", None) or getattr(result, "alpha_names", None)
        labels = ([str(v) for v in names] if names is not None and len(names) == est.size
                  else [f"z{i + 1}" for i in range(est.size)])
        lb_names = ("ci_lb_alpha", "ci_lower_alpha")
        ub_names = ("ci_ub_alpha", "ci_upper_alpha")
    else:
        est = _vector(result, _ESTIMATE)
        if est is None:
            return None
        se = _vector(result, _SE)
        stat = _vector(result, _STAT)
        pval = _vector(result, _PVAL)
        # A coefficient model exposes ``b``; a row-wise result exposes ``estimate``
        # (leave1out, cumul, influence) or ``TE`` (netmeta comparisons).
        coefficient = hasattr(result, "b") and getattr(result, "b", None) is not None
        labels = _labels(result, est.size, coefficient)
        lb_names, ub_names = _LOWER, _UPPER

    if stat is None and se is not None and se.size == est.size:
        with np.errstate(divide="ignore", invalid="ignore"):
            stat = est / se
    if pval is None and stat is not None:
        from scipy import stats

        test = str(getattr(result, "test", "z") or "z")
        ddf = getattr(result, "ddf", None)
        if test in ("knha", "adhoc", "t") and ddf:
            pval = 2.0 * stats.t.sf(np.abs(stat), float(ddf))
        else:
            pval = 2.0 * stats.norm.sf(np.abs(stat))

    level = float(conf_level if conf_level is not None
                  else getattr(result, "level", 0.95) or 0.95)
    columns: dict[str, list] = {
        "term": labels,
        "estimate": est.tolist(),
    }
    if se is not None and se.size == est.size:
        columns["std.error"] = se.tolist()
    if stat is not None and stat.size == est.size:
        columns["statistic"] = stat.tolist()
    if pval is not None and pval.size == est.size:
        columns["p.value"] = pval.tolist()
    if conf_int:
        lb, ub = _ci(result, est, se, level, lb_names, ub_names)
        columns["conf.low"] = np.asarray(lb, dtype=float)[: est.size].tolist()
        columns["conf.high"] = np.asarray(ub, dtype=float)[: est.size].tolist()

    if exponentiate:
        for key in ("estimate", "conf.low", "conf.high"):
            if key in columns:
                columns[key] = np.exp(np.asarray(columns[key], dtype=float)).tolist()
    return columns


# --------------------------------------------------------------------------- #
# glance
# --------------------------------------------------------------------------- #
#: Order in which model-level statistics are presented.
_GLANCE_ORDER = (
    "k", "nobs", "p", "df", "tau2", "tau", "I2", "H2", "R2",
    "QE", "QEp", "QM", "QMp", "logLik", "AIC", "BIC",
    "method", "test", "measure", "level", "dfs", "ddf",
)


def glance(result, *, backend: str = "auto"):
    """A single row of model-level statistics.

    Mirrors ``broom::glance()``: one row describing the whole fit (heterogeneity,
    information criteria, estimator settings).  Derived quantities that are not
    stored on the object (``tau``, ``logLik``, ``AIC``, ``BIC``) are computed via
    the corresponding :mod:`pimeta.s3` generics when possible.

    Args:
        result: any fitted pimeta result object.
        backend: dataframe backend.

    Returns:
        A one-row table.
    """
    values: dict = {}
    for name in _GLANCE_ORDER:
        if hasattr(result, name):
            value = getattr(result, name)
            if isinstance(value, list) and len(value) == 1:
                value = value[0]
            if not isinstance(value, (list, tuple, np.ndarray)):
                values[name] = value

    if "tau" not in values and "tau2" in values:
        try:
            values["tau"] = math.sqrt(max(float(values["tau2"]), 0.0))
        except (TypeError, ValueError):
            pass

    for key, func in (("logLik", "logLik"), ("AIC", "AIC"), ("BIC", "BIC")):
        if key in values:
            continue
        try:
            from .. import s3
            fn = getattr(s3, func, None)
            if fn is not None:
                values[key] = float(fn(result))
        except Exception:                             # pragma: no cover - model-dependent
            pass

    return to_frame(values, backend=backend)


# --------------------------------------------------------------------------- #
# augment
# --------------------------------------------------------------------------- #
def augment(result, *, diagnostics: bool = True, backend: str = "auto"):
    """One row per study with fitted values and influence diagnostics.

    Mirrors ``broom::augment()``.  The leading dot in ``.fitted``, ``.resid``,
    ``.hat``, ``.std.resid`` and ``.cooksd`` follows broom's convention so the
    columns do not collide with moderator names.

    Args:
        result: a fitted ``rma``/``rma_mv`` style result with per-study vectors.
        diagnostics: also compute hat values, standardised residuals and Cook's
            distance (skipped silently when the model type does not support them).
        backend: dataframe backend.

    Returns:
        A table with one row per study.

    Raises:
        TypeError: when the result carries no per-study vectors.
    """
    yi = getattr(result, "yi", None)
    if yi is None:
        raise TypeError(f"{type(result).__name__} carries no per-study 'yi' vector")
    yi = np.asarray(yi, dtype=float)
    n = yi.size
    slab = getattr(result, "slab", None)
    columns: dict[str, list] = {
        "study": [str(v) for v in slab] if isinstance(slab, (list, tuple)) and len(slab) == n
                 else [str(i + 1) for i in range(n)],
        "yi": yi.tolist(),
    }
    for name, column in (("vi", "vi"), ("weights", "weight"),
                         ("fitted", ".fitted"), ("residuals", ".resid")):
        value = getattr(result, name, None)
        if value is None:
            continue
        arr = np.asarray(value, dtype=float)
        if arr.size == n:
            columns[column] = arr.tolist()

    if diagnostics:
        for label, func in ((".hat", _hat_values),
                            (".std.resid", _std_resid),
                            (".cooksd", _cooks_d)):
            try:
                value = func(result)
                arr = np.asarray(value, dtype=float)
                if arr.size == n:
                    columns[label] = arr.tolist()
            except Exception:                         # pragma: no cover - model-dependent
                continue

    return to_frame(columns, backend=backend)


def _hat_values(result):
    """Leverage values (``metafor::hatvalues.rma.uni``)."""
    from ..s3 import hatvalues_rma

    return hatvalues_rma(result)


def _std_resid(result):
    """Standardised residuals (``metafor::rstandard.rma.uni``)."""
    from ..s3 import rstandard_rma

    return rstandard_rma(result)


def _cooks_d(result):
    """Cook's distance (``metafor::cooks.distance.rma.uni``)."""
    from ..utils import cooks_distance_rma

    return cooks_distance_rma(result)
