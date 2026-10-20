"""Frame backends — turn a column mapping into whatever the caller uses.

pimeta never hard-depends on a dataframe library.  Every table-producing helper
builds a plain ``{column: values}`` mapping first and only then hands it to a
backend.  That keeps the analytic code dependency-free while still letting a
pandas, polars or Arrow user get a native object.

Supported ``backend`` values:

===========  =============================================================
``"auto"``   pandas if installed, else ``dict``
``"dict"``   ``{column: list}``
``"records"`` ``[{"col": value, ...}, ...]`` (one dict per row)
``"pandas"`` ``pandas.DataFrame``
``"polars"`` ``polars.DataFrame``
``"arrow"``  ``pyarrow.Table`` (also implements ``__dataframe__``)
===========  =============================================================
"""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np

__all__ = ["BACKENDS", "to_frame", "normalize_columns"]

BACKENDS = ("auto", "dict", "records", "pandas", "polars", "arrow")


def normalize_columns(columns: Mapping) -> dict:
    """Normalise a column mapping so every column has the same length.

    Scalars are broadcast to the row count; a mapping of only scalars yields a
    single row.  NumPy scalars and 0-d arrays are treated as scalars.

    Args:
        columns: ``{name: scalar | sequence | ndarray}``.

    Returns:
        ``{name: list}`` with equal lengths.

    Raises:
        ValueError: when the columns disagree on length.
    """
    lengths = []
    for value in columns.values():
        n = _length(value)
        if n is not None:
            lengths.append(n)
    n = max(lengths) if lengths else 1
    bad = [i for i in lengths if i not in (1, n)]
    if bad:
        raise ValueError(
            f"columns have inconsistent lengths {sorted(set(lengths))}; "
            "every column must be length 1 or the same length")
    out: dict = {}
    for key, value in columns.items():
        ln = _length(value)
        if ln is None:
            out[key] = [value] * n
        elif ln == 1 and n > 1:
            out[key] = list(np.asarray(value, dtype=object)) * n
        else:
            out[key] = list(_as_list(value))
    return out


def _as_list(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (list, tuple)):
        return list(value)
    return [value]


def _length(value) -> int | None:
    if isinstance(value, (str, bytes)):
        return None
    if isinstance(value, np.ndarray):
        return None if value.ndim == 0 else int(value.size if value.ndim == 1
                                                else len(value))
    if isinstance(value, Mapping):
        return None
    if isinstance(value, (list, tuple)):
        return len(value)
    if isinstance(value, (int, float, bool, np.generic)):
        return None
    try:
        return len(value)  # pandas Series, polars Series, ...
    except TypeError:
        return None


def to_frame(columns: Mapping, backend: str = "auto"):
    """Materialise a column mapping in the requested backend.

    Args:
        columns: ``{name: scalar | sequence}``.
        backend: one of :data:`BACKENDS`.

    Returns:
        The backend-native table, or a ``dict`` when the requested library is
        missing and ``backend="auto"``.

    Raises:
        ValueError: for an unknown backend.
        ImportError: when an explicit backend is requested but not installed.
    """
    backend = str(backend).lower()
    if backend not in BACKENDS:
        raise ValueError(f"unknown backend {backend!r}; choices: {', '.join(BACKENDS)}")
    cols = normalize_columns(columns)

    if backend == "dict":
        return cols
    if backend == "records":
        keys = list(cols)
        return [{k: cols[k][i] for k in keys} for i in range(len(cols[keys[0]]))] if keys else []

    if backend == "auto":
        try:
            import pandas as pd
        except ImportError:                           # pragma: no cover - optional
            return cols
        return pd.DataFrame(cols)

    if backend == "pandas":
        try:
            import pandas as pd
        except ImportError as exc:                    # pragma: no cover - optional
            raise ImportError("backend='pandas' requires pandas "
                              "(pip install pimeta[pandas])") from exc
        return pd.DataFrame(cols)

    if backend == "polars":
        try:
            import polars as pl
        except ImportError as exc:                    # pragma: no cover - optional
            raise ImportError("backend='polars' requires polars "
                              "(pip install polars)") from exc
        return pl.DataFrame(cols)

    # backend == "arrow"
    try:
        import pyarrow as pa
    except ImportError as exc:                        # pragma: no cover - optional
        raise ImportError("backend='arrow' requires pyarrow "
                          "(pip install pyarrow)") from exc
    return pa.table({k: pa.array(v) for k, v in cols.items()})
