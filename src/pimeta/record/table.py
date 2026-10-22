"""Symmetric table I/O: CSV, TSV, JSON, JSONL and NPZ.

Every reader here has a writer that accepts its output, which is what makes
``fit -> save -> reload -> fit again`` a meaningful test rather than a
one-way pipe.  No optional dependency is used: CSV/TSV go through :mod:`csv`,
the JSON forms through :mod:`pimeta.record.jsonio` (so NaN and infinity
survive), and NPZ through :mod:`numpy`.

Text formats are lossy by construction -- a CSV cell cannot distinguish
"missing" from "NaN" from the string ``"NaN"``.  Use JSON when the record has
to be exact; use CSV when it has to be opened in Excel.

Precision is the other trap.  Data written with ``"%.3f"`` cannot reproduce
the analysis that produced it, so text output defaults to Python's shortest
round-trip ``repr`` -- the shortest string that parses back to the identical
double.  Pass ``float_format=`` when a human-readable table is wanted instead,
and accept that it is no longer a record.
"""

from __future__ import annotations

import csv
import io
import os
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from ..data_compat import as_columns
from .jsonio import loads

__all__ = ["read_table", "write_table"]

#: Extensions that select a tab-separated dialect.
_TSV_SUFFIXES = {".tsv", ".tab"}
#: Cell text that means "no value" on input.
_MISSING_TOKENS = {"", "NA", "NaN", "nan", "N/A", ".", "null", "NULL"}


def _dialect(path: Path, delimiter: str | None) -> str | None:
    """Delimiter implied by the extension, or ``None`` to sniff it."""
    if delimiter is not None:
        return delimiter
    if path.suffix.lower() == ".csv":
        return ","
    if path.suffix.lower() in _TSV_SUFFIXES:
        return "\t"
    return None


def _coerce(value: str | None) -> Any:
    """Turn one CSV/TSV cell into a number when it clearly is one."""
    if value is None or value in _MISSING_TOKENS:
        return None
    try:
        return int(value)
    except ValueError:
        pass
    try:
        return float(value)
    except ValueError:
        return value


def read_table(
    path: str | os.PathLike,
    delimiter: str | None = None,
    *,
    infer: bool = False,
) -> dict:
    """Read a table into ``{column: list}``.

    Args:
        path: file to read.  The extension selects the parser: ``.csv``
            (comma), ``.tsv``/``.tab``/``.txt`` (tab), ``.json`` (a list of
            records or a column dict), ``.jsonl`` (one record per line) or
            ``.npz``.
        delimiter: CSV/TSV delimiter; overrides the extension.
        infer: convert numeric cells to ``int``/``float``.  Off by default so
            the result matches the file byte for byte; the CLI does its own
            conversion, while a record wants numbers.

    Returns:
        A dict of columns.  Blank cells and ``NA``/``NaN`` become ``None``.

    Raises:
        FileNotFoundError: when ``path`` does not exist.
        ValueError: when the file is not a recognised table.
    """
    file = Path(path)
    if not file.exists():
        raise FileNotFoundError(f"no such file: {path}")
    suffix = file.suffix.lower()

    if suffix == ".npz":
        with np.load(file, allow_pickle=False) as archive:
            return {name: archive[name].tolist() for name in archive.files}

    if suffix in (".json", ".jsonl"):
        raw = file.read_text(encoding="utf-8").strip()
        if not raw:
            return {}
        if suffix == ".jsonl":
            records = [loads(line) for line in raw.splitlines() if line.strip()]
        else:
            data = loads(raw)
            if isinstance(data, Mapping):
                return {
                    str(k): list(v) if isinstance(v, (list, tuple)) else [v]
                    for k, v in data.items()
                }
            if not isinstance(data, list):
                raise ValueError(f"{path}: expected a list of records or a column dict")
            records = data
        if not records:
            return {}
        names: list[str] = []
        for rec in records:
            for key in rec:
                if str(key) not in names:
                    names.append(str(key))
        return {n: [rec.get(n) for rec in records] for n in names}

    delim = _dialect(file, delimiter)
    with file.open(newline="", encoding="utf-8-sig") as handle:
        if delim is None:
            sample = handle.read(8192)
            handle.seek(0)
            try:
                delim = csv.Sniffer().sniff(sample, delimiters=",\t;|").delimiter
            except csv.Error:
                delim = ","
        reader = csv.DictReader(handle, delimiter=delim)
        if reader.fieldnames is None:
            raise ValueError(f"{path}: empty file")
        columns: dict[str, list] = {name: [] for name in reader.fieldnames}
        for row in reader:
            for name in reader.fieldnames:
                cell = row.get(name)
                columns[name].append(_coerce(cell) if infer else
                                     (None if cell in _MISSING_TOKENS else cell))
    return columns


def _cell(value: Any, float_format: str | None) -> str:
    if value is None or (isinstance(value, float) and value != value):
        return ""
    if isinstance(value, float):
        if float_format is None:
            return repr(value)
        # Accept both spellings people reach for: "%.3f" and ".3f".
        return float_format % value if "%" in float_format else format(value, float_format)
    if isinstance(value, np.floating):
        return _cell(float(value), float_format)
    if isinstance(value, (np.integer, np.bool_)):
        return str(value.item())
    return str(value)


def _npz_columns(columns: Mapping[str, Sequence]) -> dict:
    """Pick the columns :mod:`numpy` can store without pickling.

    Numbers and fixed-width strings both live happily in an ``.npz``; an
    object column (mixed types, ``None``) would need ``allow_pickle``, which
    turns loading a data file into arbitrary code execution.  Refuse that
    instead, and point at JSON.
    """
    out: dict[str, np.ndarray] = {}
    for name, values in columns.items():
        try:
            arr = np.asarray(list(values))
        except (ValueError, TypeError) as exc:
            raise ValueError(
                f"column {name!r} cannot be stored in an .npz file: {exc}. "
                "Use .json for mixed-type columns."
            ) from None
        if arr.dtype.kind in "biufUS":
            out[str(name)] = arr
        else:
            raise ValueError(
                f"column {name!r} has dtype {arr.dtype} and cannot be stored in "
                "an .npz file without pickle. Use .json for mixed-type columns."
            )
    return out


def write_table(
    path: str | os.PathLike,
    columns: Mapping[str, Sequence] | Any,
    *,
    delimiter: str | None = None,
    float_format: str | None = None,
) -> Path:
    """Write ``columns`` to ``path``, choosing the format from the extension.

    Args:
        path: destination; the extension selects the format (``.csv``,
            ``.tsv``/``.tab``/``.txt``, ``.json``, ``.jsonl``, ``.npz``);
            parent directories are created.
        columns: a mapping of ``{name: values}``, or any frame accepted by
            :func:`pimeta.as_columns`.
        delimiter: text delimiter; overrides the extension.
        float_format: a ``format`` spec such as ``"%.3f"``.  ``None`` (the
            default) writes the shortest string that round-trips exactly.

    Returns:
        The destination path.

    Raises:
        ValueError: for an extension with no writer, or when the columns have
            different lengths.
    """
    from .jsonio import dump as dump_json
    from .jsonio import dumps as dumps_json

    file = Path(path)
    data = dict(columns) if isinstance(columns, Mapping) else as_columns(columns)
    data = {str(k): list(v) for k, v in data.items()}
    lengths = {len(v) for v in data.values()}
    if len(lengths) > 1:
        raise ValueError(f"columns have different lengths: {sorted(lengths)}")
    n_rows = lengths.pop() if lengths else 0
    file.parent.mkdir(parents=True, exist_ok=True)
    suffix = file.suffix.lower()

    if suffix == ".npz":
        stored = _npz_columns(data)
        if not stored:
            raise ValueError("an .npz file needs at least one column")
        np.savez_compressed(file, **stored)
        return file

    if suffix in (".json", ".jsonl"):
        if suffix == ".jsonl":
            body = [dumps_json(row, indent=None) for row in _records(data, n_rows)]
            file.write_text(
                "\n".join(body) + ("\n" if body else ""),
                encoding="utf-8",
                newline="\n",
            )
            return file
        dump_json(data, file)
        return file

    delim = _dialect(file, delimiter) or ","
    buffer = io.StringIO()
    writer = csv.writer(buffer, delimiter=delim, lineterminator="\n")
    writer.writerow(list(data))
    for i in range(n_rows):
        writer.writerow([_cell(data[k][i], float_format) for k in data])
    file.write_text(buffer.getvalue(), encoding="utf-8", newline="\n")
    return file


def _records(columns: Mapping[str, Sequence], n_rows: int):
    keys = list(columns)
    for i in range(n_rows):
        yield {k: columns[k][i] for k in keys}
