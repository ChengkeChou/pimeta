"""Reproducibility fingerprints.

A record is only worth keeping if you can answer two different questions
about it:

1. **"Is this the same source file?"**  -- for audit trails, so a reviewer can
   point at a CSV on disk and prove the record was produced from it.  The
   answer must change when *anything* changes, including a study label.
2. **"Would the same numbers come out?"** -- for reproducibility.  The answer
   must *not* change when a display-only argument changes: relabelling
   ``slab`` from ``"S3"`` to ``"Smith 2019"`` cannot move a coefficient, and
   neither can bumping ``digits``.

One digest cannot serve both, so this module keeps them apart:

``source_fingerprint``
    Every column, in the order given, every value verbatim.  Renaming a
    label, reordering columns, adding an unused column -- all change it.

``analysis_fingerprint``
    Only the columns that entered the model (sorted by name, so column order
    is irrelevant) plus the *settings* that can move a number.  Text columns
    such as ``slab`` are not included unless the caller asks.

Both are SHA-256 over :func:`pimeta.record.jsonio.canonical`, which sorts keys
and renders floats with Python's shortest-round-trip ``repr``.  A digest
computed on Windows therefore equals one computed on Linux for the same data.
"""

from __future__ import annotations

import hashlib
import hmac
from collections.abc import Iterable, Mapping, Sequence
from typing import Any

import numpy as np

from ..data_compat import as_columns
from .jsonio import canonical

__all__ = [
    "analysis_fingerprint",
    "fingerprint",
    "matches",
    "source_fingerprint",
]

#: Version of the hashing *scheme*.  Bump it when the payload layout changes,
#: so an old digest is never silently compared against a new one.
FINGERPRINT_VERSION = 1

#: Digest length in hex characters.  32 hex chars = 128 bits: far beyond any
#: collision risk for a project's worth of analyses, while staying short
#: enough to read out loud.
DIGEST_CHARS = 32

PREFIX = "sha256:"


def fingerprint(payload: Any) -> str:
    """Digest any JSON-able structure.

    Args:
        payload: anything :func:`~pimeta.record.jsonio.canonical` accepts.

    Returns:
        ``"sha256:"`` followed by :data:`DIGEST_CHARS` hex characters.
    """
    digest = hashlib.sha256(canonical(payload).encode("utf-8"))
    return PREFIX + digest.hexdigest()[:DIGEST_CHARS]


def _column_values(column: Any) -> list:
    """Normalise one column to a plain list of JSON-friendly values.

    NumPy scalars are converted so the canonical form is the same whether the
    column arrived as ``ndarray``, ``list`` or a DataFrame series.
    """
    arr = np.asarray(column)
    if arr.dtype.kind in "biufc":
        return arr.tolist()
    # Object/str dtypes: go value by value, keeping None distinct from "None"
    # and from a missing value the package writes as NaN.
    return [_plain(v) for v in arr.tolist()]


def _plain(value: Any) -> Any:
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, float) and value != value:
        return "nan"
    return value


def _frame_payload(columns: Mapping[str, Any], names: Sequence[str]) -> dict:
    data = {name: _column_values(columns[name]) for name in names}
    lengths = {len(values) for values in data.values()}
    if len(lengths) > 1:
        raise ValueError(f"columns have different lengths: {sorted(lengths)}")
    return {
        "v": FINGERPRINT_VERSION,
        "n": lengths.pop() if lengths else 0,
        "columns": list(names),
        "data": data,
    }


def source_fingerprint(
    data: Any,
    *,
    columns: Iterable[str] | None = None,
) -> str:
    """Digest a table exactly as supplied.

    Includes column names *in the given order* and every value, so this is the
    "same file?" check.  Use it in an audit trail next to the file name.

    Args:
        data: anything :func:`pimeta.as_columns` accepts -- dict, pandas or
            polars frame, structured array, list of records.
        columns: restrict to these columns; ``None`` uses all of them in
            frame order.

    Returns:
        ``"sha256:..."``.
    """
    cols = as_columns(data)
    names = list(cols) if columns is None else [str(c) for c in columns]
    missing = [n for n in names if n not in cols]
    if missing:
        raise KeyError(f"columns not in the data: {missing}")
    return fingerprint(_frame_payload(cols, names))


def analysis_fingerprint(
    data: Any = None,
    *,
    columns: Iterable[str] | None = None,
    settings: Mapping[str, Any] | None = None,
    arrays: Mapping[str, Any] | None = None,
) -> str:
    """Digest what actually determines the numbers.

    Column *order* is ignored (names are sorted) and text columns are only
    included when named explicitly, so a relabelled ``slab`` does not change
    the result.  Anything in ``settings`` -- estimator, test, confidence
    level, transformation -- does.

    Args:
        data: table to take the columns from; may be ``None`` when the
            analysis is fully described by ``arrays``/``settings``.
        columns: which columns enter the model.  ``None`` uses all of them.
        settings: scalars that affect the computation (``{"method": "REML"}``).
        arrays: extra named vectors not present in ``data`` (for example the
            ``yi``/``vi`` already filtered for missing values).

    Returns:
        ``"sha256:..."``.
    """
    payload: dict[str, Any] = {
        "v": FINGERPRINT_VERSION,
        "settings": dict(settings or {}),
    }

    if data is not None:
        cols = as_columns(data)
        names = sorted(cols) if columns is None else sorted(str(c) for c in columns)
        missing = [n for n in names if n not in cols]
        if missing:
            raise KeyError(f"columns not in the data: {missing}")
        payload.update(_frame_payload(cols, names))
        payload["column_order"] = "sorted"

    if arrays:
        payload["arrays"] = {
            str(k): _column_values(v) for k, v in sorted(arrays.items())
        }

    return fingerprint(payload)


def matches(expected: str, actual: str) -> bool:
    """Compare two digests without leaking where they differ (constant time).

    Args:
        expected: digest recorded alongside the analysis.
        actual: digest recomputed now.

    Returns:
        ``True`` when they are identical.
    """
    return hmac.compare_digest(
        str(expected).encode("utf-8"), str(actual).encode("utf-8")
    )
