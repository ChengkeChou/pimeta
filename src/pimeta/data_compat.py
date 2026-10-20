"""Data-frame interoperability: ``data=`` frames, formulas and table output.

R's meta-analysis packages evaluate their arguments inside a ``data`` frame, so

.. code-block:: r

    rma(yi, vi, mods = ~ dose + year, data = dat)

works with bare column names.  Python has no non-standard evaluation, so pimeta
offers an explicit equivalent: pass a column **name** wherever a vector is
expected, and hand the frame in through ``data=``.

``data`` may be

* a ``dict`` mapping column name -> sequence,
* a pandas ``DataFrame``,
* a polars ``DataFrame``,
* a NumPy structured array / ``recarray``,
* a sequence of record ``dict``\\ s (``[{"yi": 0.2, "vi": 0.04}, ...]``).

Moderators may additionally be written as a **formula string**, which is
expanded into a design matrix:

    ``"~ dose + year"``     main effects
    ``"~ dose*year"``       ``dose``, ``year`` and ``dose:year``
    ``"~ dose:year"``       the interaction only
    ``"~ 0 + dose"``        no intercept (``"- 1"`` is accepted too)
    ``"~ ."``               every other column in ``data``

String columns are dummy-coded the way R does it — first level dropped, columns
named ``arm[B]`` — so the resulting coefficients line up with a ``metafor`` fit.
"""

from __future__ import annotations

import html as _html
import re
from collections.abc import Mapping

import numpy as np

__all__ = [
    "as_columns",
    "column_names",
    "lookup",
    "resolve",
    "FormulaSpec",
    "parse_formula",
    "design_matrix",
    "model_frame",
    "resolve_mods",
    "to_dataframe",
    "to_html",
    "install_repr_html",
]


# --------------------------------------------------------------------------- #
# Frames
# --------------------------------------------------------------------------- #
def as_columns(data) -> dict:
    """Normalise any supported frame into ``{column name: ndarray}``.

    Args:
        data: ``dict``, pandas ``DataFrame``, polars ``DataFrame``, NumPy
            structured array, sequence of record dicts, or an object whose
            ``__dict__`` holds the columns.

    Returns:
        A plain ``dict`` mapping column names to one-dimensional arrays.

    Raises:
        TypeError: when ``data`` is none of the supported types.
    """
    if data is None:
        return {}
    if isinstance(data, Mapping):
        return {str(k): np.asarray(v) for k, v in data.items()}

    # pandas / polars both expose `.columns` and `.to_dict()`.
    if hasattr(data, "columns") and hasattr(data, "to_dict"):
        try:
            cols = data.to_dict(orient="list")          # pandas
        except TypeError:
            cols = data.to_dict(as_series=False)        # polars
        if isinstance(cols, Mapping):
            return {str(k): np.asarray(v) for k, v in cols.items()}

    # NumPy structured array / recarray.
    if isinstance(data, np.ndarray) and data.dtype.names:
        return {str(n): np.asarray(data[n]) for n in data.dtype.names}

    # Sequence of records.
    if isinstance(data, (list, tuple)) and data and isinstance(data[0], Mapping):
        names: list[str] = []
        for rec in data:
            for key in rec:
                if str(key) not in names:
                    names.append(str(key))
        return {n: np.asarray([rec.get(n, np.nan) for rec in data])
                for n in names}

    if hasattr(data, "__dict__") and not isinstance(data, type):
        return {k: np.asarray(v) for k, v in vars(data).items()
                if not k.startswith("_")}

    raise TypeError(
        "argument 'data' must be a dict, a pandas/polars DataFrame, a NumPy "
        f"structured array or a list of records, not {type(data).__name__}")


def column_names(data) -> list:
    """The column names of any frame accepted by :func:`as_columns`."""
    return list(as_columns(data))


def lookup(data, name: str) -> np.ndarray:
    """Fetch column ``name`` from ``data``, with a helpful error when missing."""
    cols = as_columns(data)
    if name not in cols:
        have = ", ".join(sorted(cols)) or "(none)"
        raise ValueError(
            f"column {name!r} not found in 'data'; available columns: {have}")
    return cols[name]


def resolve(spec, data=None, k=None, name: str = "value", dtype=float,
            strings_as_names=None):
    """Resolve one argument that may be a value, a column name or a name list.

    Args:
        spec: the value itself, a column name (requires ``data``), or a list of
            column names (returns a list of arrays).
        data: the ``data=`` frame.
        k: expected length; checked when given.
        name: argument name, used in error messages.
        dtype: target NumPy dtype.  ``None`` keeps the input dtype (needed for
            string arguments such as study labels or treatment names).
        strings_as_names: whether a list of strings should be read as column
            names.  ``None`` (default) decides automatically: only when ``data``
            was supplied *and* every string is an existing column.  That keeps
            legitimate string vectors such as ``slab=["Study 1", "Study 2"]``
            working, which a blanket "list of strings means column names" rule
            would break.

    Returns:
        ``None``, an ``ndarray``, or a ``list`` of ``ndarray``.

    Raises:
        ValueError: when a column name is used without ``data``, a column is
            missing, or the resolved length does not match ``k``.
    """
    if spec is None:
        return None

    if isinstance(spec, str):
        if data is None:
            raise ValueError(
                f"{name}={spec!r} looks like a column name but no 'data' was "
                "supplied")
        spec = lookup(data, spec)
    elif (isinstance(spec, (list, tuple)) and spec
          and all(isinstance(v, str) for v in spec)):
        if strings_as_names is None:
            available = set(column_names(data)) if data is not None else set()
            strings_as_names = bool(available) and all(v in available for v in spec)
        if strings_as_names:
            if data is None:
                raise ValueError(
                    f"{name}={list(spec)!r} looks like column names but no "
                    "'data' was supplied")
            parts = [lookup(data, v) for v in spec]
            if dtype is not None:
                parts = [np.asarray(a, dtype=dtype) for a in parts]
            if k is not None:
                for a in parts:
                    if a.shape[0] != k:
                        raise ValueError(
                            f"{name} has {a.shape[0]} values but {k} were expected")
            return parts

    out = np.asarray(spec, dtype=dtype) if dtype is not None else np.asarray(spec)
    if k is not None and out.ndim >= 1 and out.shape[0] != k:
        raise ValueError(
            f"{name} has {out.shape[0]} values but {k} were expected")
    return out


# --------------------------------------------------------------------------- #
# Formulas
# --------------------------------------------------------------------------- #
class FormulaSpec:
    """A parsed formula: ``response ~ term1 + term2``.

    Attributes:
        response: name of the left-hand-side column, or ``None``.
        terms: list of term strings; an interaction is written ``"a:b"``.
        intercept: whether the model includes an intercept column.
    """

    __slots__ = ("response", "terms", "intercept")

    def __init__(self, response=None, terms=None, intercept=True):
        self.response = response
        self.terms = list(terms or [])
        self.intercept = bool(intercept)

    def __repr__(self):
        lhs = f"{self.response} ~ " if self.response else "~ "
        rhs = " + ".join(self.terms) if self.terms else "1"
        if not self.intercept:
            rhs = "0 + " + rhs
        return f"FormulaSpec({lhs}{rhs!r})"

    def __eq__(self, other):
        return (isinstance(other, FormulaSpec)
                and self.response == other.response
                and self.terms == other.terms
                and self.intercept == other.intercept)


_IDENT = r"(?:`[^`]+`|[A-Za-z_][A-Za-z0-9_.]*|\.[A-Za-z0-9_.]*|\d+)"


def _strip_backticks(name: str) -> str:
    return name[1:-1] if name.startswith("`") and name.endswith("`") else name


def _expand_operand(op: str) -> list:
    """``a*b`` -> ``[a, b, a:b]``; ``a:b`` -> ``[a:b]``; ``a`` -> ``[a]``."""
    factors = [_strip_backticks(f.strip()) for f in op.split(":") if f.strip()]
    if "*" in op:
        parts = [_strip_backticks(p.strip()) for p in op.split("*") if p.strip()]
        out = list(parts)
        for i, left in enumerate(parts):
            for right in parts[i + 1:]:
                out.append(f"{left}:{right}")
        return out
    if not factors:
        return []
    return [":".join(factors)]


def parse_formula(formula) -> FormulaSpec:
    """Parse a formula string into a :class:`FormulaSpec`.

    Accepts ``"y ~ a + b"``, ``"~ a + b"``, ``"a + b"``, ``"~ a*b"``,
    ``"~ a:b"``, ``"~ 0 + a"``, ``"~ a - 1"`` and ``"~ ."``.

    Args:
        formula: the formula string (a :class:`FormulaSpec` is returned as-is).

    Returns:
        The parsed :class:`FormulaSpec`.

    Raises:
        ValueError: for an empty or malformed formula.
    """
    if isinstance(formula, FormulaSpec):
        return formula
    text = str(formula).strip()
    if not text:
        raise ValueError("empty formula")

    if "~" in text:
        lhs, rhs = text.split("~", 1)
    else:
        lhs, rhs = "", text
    response = lhs.strip() or None
    if response is not None:
        response = _strip_backticks(response)

    terms: list[str] = []
    intercept = True
    sign = "+"
    for chunk in re.split(r"\s*([+-])\s*", rhs.strip()):
        chunk = chunk.strip()
        if chunk in ("+", "-"):
            sign = chunk
            continue
        if not chunk:
            continue
        if chunk in ("0", "1"):
            intercept = False if (chunk == "0" or sign == "-") else True
            continue
        expanded = _expand_operand(chunk)
        if sign == "-":
            for term in expanded:
                if term in terms:
                    terms.remove(term)
        else:
            for term in expanded:
                if term not in terms:
                    terms.append(term)

    return FormulaSpec(response=response, terms=terms, intercept=intercept)


def _design_block(name: str, values) -> list:
    """One moderator column, expanded to dummies when it is not numeric."""
    arr = np.asarray(values)
    if arr.dtype.kind in "biufc":
        return [(name, arr.astype(float).ravel())]
    labels = [str(v) for v in arr.ravel()]
    levels = sorted(set(labels))
    return [(f"{name}[{lv}]",
             np.array([1.0 if v == lv else 0.0 for v in labels]))
            for lv in levels[1:]]           # R drops the first level


def design_matrix(formula, data, intercept=None):
    """Build a design matrix from a formula and a frame.

    Args:
        formula: formula string or :class:`FormulaSpec`.
        data: any frame accepted by :func:`as_columns`.
        intercept: force the intercept on/off; ``None`` uses the formula's own
            ``0`` / ``1`` marker.

    Returns:
        ``(X, names)`` — the ``k x p`` design matrix (``p == 0`` when the model
        has neither an intercept nor terms) and its column names.

    Raises:
        ValueError: when a term names a missing column, or a ``"."`` expansion
            leaves nothing.
    """
    spec = parse_formula(formula)
    cols = as_columns(data)
    if intercept is None:
        intercept = spec.intercept

    terms = list(spec.terms)
    if "." in terms:
        terms = [t for t in terms if t != "."]
        terms += [c for c in cols if c != spec.response and c not in terms]

    blocks: list[tuple] = []
    for term in terms:
        factors = term.split(":")
        parts: list[list] = []
        for factor in factors:
            if factor not in cols:
                have = ", ".join(sorted(cols)) or "(none)"
                raise ValueError(
                    f"term {factor!r} in the formula does not name a column; "
                    f"available columns: {have}")
            parts.append(_design_block(factor, cols[factor]))

        # Cartesian product of the factor blocks, row-wise.
        acc = list(parts[0])
        for nxt in parts[1:]:
            joined = []
            for left_name, left in acc:
                for right_name, right in nxt:
                    joined.append((f"{left_name}:{right_name}", left * right))
            acc = joined
        blocks.extend(acc)

    names = [n for n, _ in blocks]
    columns = [c for _, c in blocks]
    if intercept:
        k = len(next(iter(cols.values()))) if cols else 0
        names = ["(Intercept)"] + names
        columns = [np.ones(k)] + columns
    if not columns:
        return np.zeros((0, 0)), []
    return np.column_stack(columns), names


def model_frame(formula, data):
    """Split a two-sided formula into ``(y, X, names)``.

    Args:
        formula: formula string such as ``"yi ~ dose + year"``.
        data: any frame accepted by :func:`as_columns`.

    Returns:
        ``(y, X, names)`` where ``y`` is ``None`` for a one-sided formula.

    Raises:
        ValueError: when the formula has no terms, or a column is missing.
    """
    spec = parse_formula(formula)
    X, names = design_matrix(spec, data)
    y = None
    if spec.response is not None:
        y = np.asarray(lookup(data, spec.response), dtype=float)
    return y, X, names


def resolve_mods(mods, data=None, k=None):
    """Resolve the ``mods`` argument, accepting formulas and column names.

    ``rma()`` prepends its own intercept column, so the intercept encoded in a
    formula is reported separately through the second return value instead of
    being materialised here.

    Args:
        mods: ``None``; a 1-D/2-D array-like; a list of column names; or a
            formula string such as ``"~ dose + year"``.
        data: the ``data=`` frame.
        k: number of studies, used for length checking.

    Returns:
        ``(array_or_None, add_intercept)``.

    Raises:
        ValueError: when a formula has a left-hand side, or a column is missing.
    """
    if mods is None:
        return None, True

    if isinstance(mods, str):
        if data is None:
            raise ValueError(
                f"mods={mods!r} looks like a formula but no 'data' was "
                "supplied")
        spec = parse_formula(mods)
        if spec.response is not None:
            raise ValueError(
                "'mods' must be one-sided (e.g. '~ dose + year'); the response "
                "is passed separately as 'yi'")
        X, _names = design_matrix(spec, data, intercept=False)
        return (X if X.size else None), spec.intercept

    if isinstance(mods, FormulaSpec):
        if data is None:
            raise ValueError(
                "a FormulaSpec for 'mods' needs a 'data' frame")
        X, _names = design_matrix(mods, data, intercept=False)
        return (X if X.size else None), mods.intercept

    if (isinstance(mods, (list, tuple)) and mods
            and all(isinstance(v, str) for v in mods)):
        if data is None:
            raise ValueError(
                f"mods={list(mods)!r} looks like column names but no 'data' "
                "was supplied")
        arr = np.column_stack([np.asarray(lookup(data, v), dtype=float)
                               for v in mods])
        return arr, True

    arr = np.asarray(mods, dtype=float)
    if arr.ndim == 1:
        arr = arr.reshape(-1, 1)
    if k is not None and arr.shape[0] != k:
        raise ValueError("'mods' must have one row per study")
    return arr, True


# --------------------------------------------------------------------------- #
# Table output
# --------------------------------------------------------------------------- #
def _is_scalar(value) -> bool:
    if isinstance(value, (str, bytes)):
        return True
    if isinstance(value, np.ndarray):
        return value.ndim == 0
    return bool(np.isscalar(value))


def _flatten(obj, prefix=""):
    """Yield ``(name, value)`` pairs; arrays are summarised, not expanded."""
    data = obj.to_dict() if hasattr(obj, "to_dict") else dict(vars(obj))
    for key, value in data.items():
        name = f"{prefix}{key}"
        if isinstance(value, Mapping):
            yield from _flatten(value, prefix=f"{name}.")
        else:
            yield name, value


def _short(value, max_items: int = 6):
    if value is None:
        return "—"
    if isinstance(value, (list, tuple, np.ndarray)):
        arr = np.asarray(value)
        if arr.ndim == 0:
            return repr(arr.item())
        flat = arr.ravel()
        head = ", ".join(_num(v) for v in flat[:max_items])
        more = f", … (+{flat.size - max_items})" if flat.size > max_items else ""
        return f"[{head}{more}]"
    return _num(value)


def _num(value):
    if isinstance(value, (bool, np.bool_)):
        return str(bool(value))
    if isinstance(value, (int, np.integer)):
        return str(int(value))
    if isinstance(value, (float, np.floating)):
        return f"{float(value):.6g}"
    return str(value)


def to_dataframe(obj, rows=None):
    """Turn a result object into a pandas ``DataFrame`` (or a dict of columns).

    Scalar fields become one row; when the object carries per-row vectors (a
    ``leave1out`` / ``influence`` / ``cumul`` style result) every row-level
    field becomes a column instead.

    Args:
        obj: any pimeta result object.
        rows: optional row selection for list-like results.

    Returns:
        A ``pandas.DataFrame`` when pandas is importable, otherwise a plain
        ``dict`` of columns (so the function never hard-requires pandas).
    """
    from .rma_list import as_data_frame_rma_list, _row_lengths

    k = getattr(obj, "k", None)
    names = _row_lengths(obj, int(k) if isinstance(k, int) else None)
    if names and not isinstance(k, int):
        cols = as_data_frame_rma_list(obj, rows=rows)
    elif names and len(names) > 1 and not _is_scalar(getattr(obj, names[0], None)):
        cols = as_data_frame_rma_list(obj, rows=rows)
    else:
        cols = {name: [value] if _is_scalar(value) else [value]
                for name, value in _flatten(obj)}

    try:
        import pandas as pd
    except ImportError:                              # pragma: no cover - optional
        return cols
    return pd.DataFrame(cols)


def to_html(obj, max_rows: int = 60) -> str:
    """Render a result object as a compact HTML table (Jupyter ``_repr_html_``).

    Args:
        obj: any pimeta result object.
        max_rows: maximum number of fields shown before truncating.

    Returns:
        An HTML ``<table>`` string.
    """
    items = list(_flatten(obj))
    title = type(obj).__name__
    body = []
    for name, value in items[:max_rows]:
        body.append(
            "<tr><th style='text-align:left;padding:2px 10px 2px 0;"
            "font-weight:500;white-space:nowrap'>"
            f"{_html.escape(str(name))}</th>"
            "<td style='text-align:left;padding:2px 0;font-family:monospace'>"
            f"{_html.escape(_short(value))}</td></tr>")
    if len(items) > max_rows:
        body.append(
            "<tr><td colspan='2' style='color:#888;padding-top:4px'>"
            f"… {len(items) - max_rows} more fields</td></tr>")
    return (
        "<div style='display:inline-block'>"
        f"<div style='font-weight:600;margin-bottom:4px'>{_html.escape(title)}"
        "</div><table style='border-collapse:collapse;font-size:12px'>"
        + "".join(body)
        + "</table></div>")


def _repr_html_(self):                               # pragma: no cover - notebook
    return to_html(self)


def install_repr_html(namespace) -> int:
    """Attach ``_repr_html_`` to every public class in ``namespace`` with ``to_dict``.

    Called once from :mod:`pimeta.__init__` so that ``repr()`` in Jupyter shows
    a formatted table for every result type, without repeating the mixin in 40
    class definitions.

    Args:
        namespace: a module ``globals()`` mapping.

    Returns:
        The number of classes that received the method.
    """
    count = 0
    for obj in list(namespace.values()):
        if not isinstance(obj, type):
            continue
        if not getattr(obj, "__module__", "").startswith("pimeta"):
            continue
        if not hasattr(obj, "to_dict"):
            continue
        if "_repr_html_" in vars(obj):
            continue
        setattr(obj, "_repr_html_", _repr_html_)
        count += 1
    return count
