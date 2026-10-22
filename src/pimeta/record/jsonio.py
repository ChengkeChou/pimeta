"""Strict, lossless JSON for numerical results.

Python's :mod:`json` module writes ``NaN``, ``Infinity`` and ``-Infinity`` by
default.  Those tokens are *not* JSON: ``JSON.parse`` in every browser
rejects them, so does PostgreSQL's ``jsonb``, so does ``jq`` when built
strictly, and so does most strongly typed readers.  A file that only Python
can read is not a portable record.

The usual workaround -- writing ``null`` -- silently destroys information that
matters in a meta-analysis.  ``tau2 = 0`` is a real estimate; ``I2 = nan``
with ``k = 1`` means "undefined", not "zero"; ``+inf`` shows up in ``H2``
when the pooled variance collapses.  ``null`` conflates all three.

So non-finite floats become explicit tagged objects::

    {"$pimeta": "nan"}
    {"$pimeta": "inf"}
    {"$pimeta": "-inf"}

``$pimeta`` is a reserved key.  Every special value this module emits carries
it, so a decoded dict that happens to contain ``$pimeta`` is unambiguously one
of ours.  ``loads(dumps(x))`` returns ``x`` unchanged for every value the
package produces -- including NaN (compared by :func:`math.isnan`, since
``nan != nan``).

Numeric arrays are written as plain nested lists by default: readable, and
consumable by anything that understands JSON.  Pass ``arrays="tagged"`` when
the dtype and shape have to survive the round trip -- that form is::

    {"$pimeta": "array", "dtype": "<f8", "shape": [2, 3], "data": [[...]]}

Writes are atomic: the text goes to a temporary file in the destination
directory and is then renamed over the target, so an interrupted save can
never leave a half-written record behind.
"""

from __future__ import annotations

import base64
import dataclasses
import datetime as _dt
import json
import os
import tempfile
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np

__all__ = [
    "JSONError",
    "canonical",
    "dump",
    "dumps",
    "from_jsonable",
    "load",
    "loads",
    "to_jsonable",
]

#: Reserved discriminator key.  Nothing else in this module writes it.
TAG = "$pimeta"

#: Default indentation for :func:`dumps`; ``None`` writes one dense line.
DEFAULT_INDENT = 2

ARRAY_MODES = ("list", "tagged")


class JSONError(ValueError):
    """Raised when a value has no lossless JSON representation.

    Deliberately a ``ValueError`` so callers that already handle bad data can
    catch it without importing this module.  The message always names the
    offending Python type, because the fix is usually "call ``to_dict()``
    first" or "convert this column to numbers".
    """


# --------------------------------------------------------------------------- #
# encoding
# --------------------------------------------------------------------------- #
def _encode(obj: Any, arrays: str) -> Any:
    """Recursively convert ``obj`` into JSON-safe primitives."""
    # Order matters: bool is a subclass of int, numpy scalars are not float.
    if obj is None or isinstance(obj, (str, bool)):
        return obj
    if isinstance(obj, int):
        return obj
    if isinstance(obj, float):
        if obj != obj:                      # NaN
            return {TAG: "nan"}
        if obj == float("inf"):
            return {TAG: "inf"}
        if obj == float("-inf"):
            return {TAG: "-inf"}
        return obj
    if isinstance(obj, np.bool_):
        return bool(obj)
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return _encode(float(obj), arrays)
    if isinstance(obj, np.str_):
        return str(obj)
    if isinstance(obj, complex):
        return {TAG: "complex", "re": obj.real, "im": obj.imag}
    if isinstance(obj, np.complexfloating):
        return _encode(complex(obj), arrays)
    if isinstance(obj, np.ndarray):
        return _encode_array(obj, arrays)
    if isinstance(obj, np.datetime64):
        return {TAG: "datetime64", "value": np.datetime_as_string(obj)}
    if isinstance(obj, (np.timedelta64,)):
        return {TAG: "timedelta64", "value": str(obj)}
    if isinstance(obj, (bytes, bytearray)):
        return {TAG: "bytes", "b64": base64.b64encode(bytes(obj)).decode("ascii")}
    if isinstance(obj, (Path, os.PathLike)):
        return os.fspath(obj)
    if isinstance(obj, (_dt.datetime, _dt.date, _dt.time)):
        return {TAG: "datetime", "iso": obj.isoformat()}
    if isinstance(obj, (set, frozenset)):
        # Sorted so the bytes are stable across runs: set iteration order
        # depends on the hash seed.
        items = [_encode(v, arrays) for v in obj]
        return {TAG: "set", "items": sorted(items, key=_sort_key)}
    if dataclasses.is_dataclass(obj) and not isinstance(obj, type):
        return {f.name: _encode(getattr(obj, f.name), arrays)
                for f in dataclasses.fields(obj)}
    if isinstance(obj, Mapping):
        return {str(k): _encode(v, arrays) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_encode(v, arrays) for v in obj]
    raise JSONError(
        f"{type(obj).__name__} objects have no lossless JSON form; call "
        ".to_dict() first or pass plain numbers/strings/lists"
    )


def _encode_array(arr: np.ndarray, arrays: str) -> Any:
    if arrays not in ARRAY_MODES:
        raise ValueError(f"arrays must be one of {ARRAY_MODES}, not {arrays!r}")
    data = _encode(arr.tolist(), arrays)
    if arrays == "list":
        return data
    return {
        TAG: "array",
        "dtype": arr.dtype.str,
        "shape": list(arr.shape),
        "data": data,
    }


def _sort_key(value: Any) -> str:
    """Deterministic sort key for members of a set."""
    return json.dumps(value, sort_keys=True, ensure_ascii=False, allow_nan=False)


def to_jsonable(obj: Any, *, arrays: str = "list") -> Any:
    """Return a JSON-safe copy of ``obj``.

    Args:
        obj: any value built from numbers, strings, mappings, sequences,
            NumPy scalars/arrays, dataclasses, paths, datetimes, sets, bytes.
        arrays: ``"list"`` (default) writes arrays as nested lists and loses
            dtype/shape; ``"tagged"`` keeps dtype and shape.

    Returns:
        A structure of ``dict``/``list``/``str``/``int``/``float``/``bool``/
        ``None`` only.

    Raises:
        JSONError: for a type with no lossless representation.
    """
    return _encode(obj, arrays)


# --------------------------------------------------------------------------- #
# decoding
# --------------------------------------------------------------------------- #
def _decode(obj: Any) -> Any:
    """Inverse of :func:`to_jsonable`."""
    if isinstance(obj, list):
        return [_decode(v) for v in obj]
    if not isinstance(obj, dict):
        return obj

    tag = obj.get(TAG)
    if tag is None:
        return {k: _decode(v) for k, v in obj.items()}

    if tag == "nan":
        return float("nan")
    if tag == "inf":
        return float("inf")
    if tag == "-inf":
        return float("-inf")
    if tag == "complex":
        return complex(obj["re"], obj["im"])
    if tag == "array":
        values = _decode(obj["data"])
        arr = np.array(values, dtype=np.dtype(obj["dtype"]))
        return arr.reshape(obj["shape"])
    if tag == "datetime":
        return _dt.datetime.fromisoformat(obj["iso"])
    if tag == "datetime64":
        return np.datetime64(obj["value"])
    if tag == "timedelta64":
        return np.timedelta64(obj["value"])
    if tag == "bytes":
        return base64.b64decode(obj["b64"])
    if tag == "set":
        return {_freeze(_decode(v)) for v in obj["items"]}
    # Unknown tag: leave it alone rather than guessing.  This keeps records
    # written by a newer schema readable by an older reader.
    return {k: _decode(v) for k, v in obj.items()}


def _freeze(value: Any) -> Any:
    """Make a decoded value hashable so it can go back into a set."""
    if isinstance(value, list):
        return tuple(_freeze(v) for v in value)
    if isinstance(value, dict):
        return tuple(sorted((k, _freeze(v)) for k, v in value.items()))
    return value


def from_jsonable(obj: Any) -> Any:
    """Inverse of :func:`to_jsonable`; undoes the tagged special values."""
    return _decode(obj)


# --------------------------------------------------------------------------- #
# text
# --------------------------------------------------------------------------- #
def dumps(
    obj: Any,
    *,
    indent: int | None = DEFAULT_INDENT,
    arrays: str = "list",
    sort_keys: bool = False,
) -> str:
    """Serialise ``obj`` to strict JSON text.

    Args:
        obj: value to serialise.
        indent: indentation width; ``None`` emits a single dense line.
        arrays: ``"list"`` or ``"tagged"`` (see the module docstring).
        sort_keys: sort object keys.  Keeps output byte-identical between
            runs, which is what makes the fingerprints stable.

    Returns:
        JSON text.  Always valid JSON -- ``allow_nan`` is off and non-finite
        floats have already been replaced by tagged objects.
    """
    payload = _encode(obj, arrays)
    return json.dumps(
        payload,
        ensure_ascii=False,
        allow_nan=False,
        indent=indent,
        sort_keys=sort_keys,
        separators=(",", ": ") if indent is not None else (",", ":"),
    )


def loads(text: str) -> Any:
    """Parse JSON text produced by :func:`dumps` (or any ordinary JSON)."""
    try:
        raw = json.loads(text)
    except json.JSONDecodeError as exc:
        raise JSONError(f"not valid JSON: {exc}") from exc
    return _decode(raw)


def canonical(obj: Any) -> str:
    """A byte-stable representation of ``obj``, for hashing.

    Sorted keys, no whitespace, no indentation.  Floats are rendered by
    Python's shortest-round-trip ``repr``, which is identical on every
    platform for the same IEEE-754 double, so a digest computed on Windows
    matches one computed on Linux.
    """
    return json.dumps(
        _encode(obj, arrays="list"),
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    )


# --------------------------------------------------------------------------- #
# files
# --------------------------------------------------------------------------- #
def dump(
    obj: Any,
    path: str | os.PathLike,
    *,
    indent: int | None = DEFAULT_INDENT,
    arrays: str = "list",
    sort_keys: bool = False,
) -> Path:
    """Write ``obj`` to ``path`` atomically and return the path.

    The text is written to a temporary file in the same directory and then
    renamed over the target, so a crash mid-write cannot truncate an existing
    record.  ``newline="\\n"`` keeps a record written on Windows byte-identical
    to one written on Linux -- otherwise the same analysis would hash
    differently depending on the platform that saved it.

    Args:
        obj: value to serialise.
        path: destination; parent directories are created.
        indent: indentation width, ``None`` for one dense line.
        arrays: ``"list"`` or ``"tagged"``.
        sort_keys: sort object keys.

    Returns:
        The destination path.
    """
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    text = dumps(obj, indent=indent, arrays=arrays, sort_keys=sort_keys) + "\n"

    handle = tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        newline="\n",
        dir=target.parent,
        prefix=f".{target.name}.",
        suffix=".tmp",
        delete=False,
    )
    try:
        with handle:
            handle.write(text)
        os.replace(handle.name, target)
    except BaseException:
        Path(handle.name).unlink(missing_ok=True)
        raise
    return target


def load(path: str | os.PathLike) -> Any:
    """Read a JSON file written by :func:`dump` (or any ordinary JSON)."""
    return loads(Path(path).read_text(encoding="utf-8"))
