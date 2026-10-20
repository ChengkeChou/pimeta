"""Missing-value handling shared by the model-fitting entry points.

R drops incomplete cases through ``na.action`` (``na.omit`` by default) and says
nothing about it.  pimeta keeps the dropping but makes it *visible*: a
``UserWarning`` names how many rows went away and which ones, so a silent typo
in a CSV cannot quietly halve the analysis.

``escalc`` deliberately does **not** drop rows — its output is aligned with the
input frame — so the warning surfaces when the data reaches a model.
"""

from __future__ import annotations

import math
import warnings

import numpy as np

__all__ = ["is_missing", "complete_cases", "MissingDataWarning"]


class MissingDataWarning(UserWarning):
    """Raised when rows are dropped because of missing values."""


def is_missing(values) -> np.ndarray:
    """Boolean mask of missing entries in ``values``.

    Numeric arrays use ``~np.isfinite`` (so ``nan``, ``inf`` and ``-inf`` all
    count as missing, as R's ``NA`` does after arithmetic).  Object arrays treat
    ``None`` and ``float('nan')`` as missing.  Integer and boolean arrays have no
    missing values and yield an all-``False`` mask.

    Args:
        values: a 1-D array-like.

    Returns:
        A boolean ``ndarray`` with one entry per element.
    """
    arr = np.asarray(values)
    if arr.ndim == 0:
        arr = arr.reshape(1)
    kind = arr.dtype.kind
    if kind in "fc":
        return ~np.isfinite(arr.astype(float, copy=False))
    if kind in "iub":
        return np.zeros(arr.shape[0], dtype=bool)
    return np.array([v is None or (isinstance(v, float) and math.isnan(v))
                     for v in arr.ravel()], dtype=bool)


def complete_cases(columns, *, context: str = "rows", warn: bool = True):
    """Row mask that keeps only complete cases, plus the dropped indices.

    Args:
        columns: ``{name: sequence}`` — every column must have the same length.
        context: noun used in the warning message ("studies", "rows", ...).
        warn: emit :class:`MissingDataWarning` when anything is dropped.

    Returns:
        ``(keep, dropped)`` where ``keep`` is a boolean mask over the original
        rows and ``dropped`` holds the 0-based indices that were removed.

    Raises:
        ValueError: when the columns do not all have the same length.
    """
    arrays = {name: np.asarray(values) for name, values in columns.items()
              if values is not None}
    if not arrays:
        return np.zeros(0, dtype=bool), np.zeros(0, dtype=int)

    lengths = {name: arr.shape[0] for name, arr in arrays.items()}
    n = max(lengths.values())
    ragged = {name: ln for name, ln in lengths.items() if ln != n}
    if ragged:
        raise ValueError(
            f"columns have different lengths: {ragged} (expected {n})")

    bad = np.zeros(n, dtype=bool)
    for arr in arrays.values():
        bad |= is_missing(arr)

    keep = ~bad
    dropped = np.flatnonzero(bad)
    if warn and dropped.size:
        shown = ", ".join(str(int(i) + 1) for i in dropped[:10])
        more = f", ... (+{dropped.size - 10} more)" if dropped.size > 10 else ""
        names = ", ".join(f"'{name}'" for name in arrays)
        warnings.warn(
            f"dropped {dropped.size} of {n} {context} with missing values in "
            f"{names} (row{'s' if dropped.size > 1 else ''} {shown}{more}); "
            "pass warn_missing=False to silence this",
            MissingDataWarning,
            stacklevel=3,
        )
    return keep, dropped
