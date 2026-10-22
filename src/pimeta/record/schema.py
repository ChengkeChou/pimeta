"""The analysis record: one file that explains how a number was produced.

Layout (schema ``pimeta.record/1``)::

    {
      "schema": "pimeta.record/1",
      "kind": "rma",
      "created_utc": "2026-10-21T15:40:02+00:00",
      "inputs":      { ... },   # 输入数据
      "analysis":    { ... },   # 分析配置
      "process":     { ... },   # 运算过程
      "environment": { ... },   # 运行环境
      "results":     { ... }    # 结果与诊断
    }

The five blocks answer the five questions a reader of a meta-analysis
eventually asks: *what data went in*, *what was asked of it*, *what happened
while it ran*, *with what software*, and *what came out*.

Two storage modes
-----------------
``mode="digest"``
    Store the fingerprints, the column names and the row count -- not the
    data.  Records stay small and no participant-level data is copied, but
    the record can only be *checked* against a file you still have.

``mode="bundle"``
    Store the data too.  The record stands alone and :meth:`replay` can
    re-run the analysis without the original file.  Use this for anything
    that has to be reproducible years later.

Both modes record the same fingerprints, so a bundle can always be verified
against the file it came from.
"""

from __future__ import annotations

import dataclasses
import datetime as _dt
import importlib
import os
import platform
import sys
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from ..data_compat import as_columns
from .fingerprint import (
    FINGERPRINT_VERSION,
    analysis_fingerprint,
    matches,
    source_fingerprint,
)
from .jsonio import dump as dump_json
from .jsonio import load as load_json

__all__ = [
    "AnalysisRecord",
    "RecordError",
    "RecordMismatch",
    "RECORD_SCHEMA",
    "build_record",
    "compare_results",
    "environment",
    "load_record",
    "save_record",
]

#: Bumped whenever the block layout changes incompatibly.
RECORD_SCHEMA = "pimeta.record/1"

#: Input vectors copied out of a fitted result when the caller supplies no
#: explicit ``data``.  These are exactly the arrays the estimator consumed.
_INPUT_ATTRS = ("yi", "vi", "mods")

#: Attributes of a result that can move a number.  Deliberately excludes
#: ``slab``, ``measure`` and ``digits``: those change the *presentation*.
_SETTING_ATTRS = (
    "method",
    "test",
    "level",
    "tau2_fix",
    "dfs",
    "ddf",
    "int_only",
    "int_incl",
    "parms",
    "link",
    "scale_intercept",
    "optbeta",
    "alpha",
    "model",
    "struct",
)

#: Attributes of a result worth keeping under "process" when present.
_PROCESS_ATTRS = ("iterations", "convergence", "n_iter", "optimizer", "elapsed_ms")

_THREAD_VARS = (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "NUMEXPR_NUM_THREADS",
)


class RecordError(RuntimeError):
    """A record cannot be verified or replayed.

    Raised for a digest-only record asked to replay without data, a record
    with no ``call`` block, or a ``function`` that is not a pimeta function.
    The message always says what to do about it.
    """


class RecordMismatch(RecordError):
    """Recomputed numbers disagree with the values stored in the record."""

    def __init__(self, message: str, differences: list[tuple[str, Any, Any]]):
        super().__init__(message)
        self.differences = differences


# --------------------------------------------------------------------------- #
# environment
# --------------------------------------------------------------------------- #
def _pkg_version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except Exception:                                    # pragma: no cover
        return None


def environment(*, seed: int | None = None) -> dict:
    """Describe the software and machine that produced a result.

    Versions of everything that can change a floating-point answer are
    recorded, because "reproducible" only means something when the reader can
    see whether their NumPy differs from yours.  Thread counts matter for the
    same reason: BLAS reductions are not always order-stable across thread
    counts, so a 16-thread run can differ from a 1-thread run in the last bits.

    Args:
        seed: the random seed in force, when the analysis used one.

    Returns:
        A JSON-able dict.  Missing optional packages are reported as ``None``
        rather than omitted, so the shape of the block is stable.
    """
    from .. import __version__ as pimeta_version

    return {
        "pimeta": pimeta_version,
        "python": sys.version.split()[0],
        "implementation": platform.python_implementation(),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor() or None,
        "cpu_count": os.cpu_count(),
        "numpy": np.__version__,
        "scipy": _pkg_version("scipy"),
        "matplotlib": _pkg_version("matplotlib"),
        "fp_error": dict(np.geterr()),
        "threads": {name: os.environ.get(name) for name in _THREAD_VARS},
        "seed": seed,
        "timezone": _utc_offset(),
    }


def _utc_offset() -> str:
    offset = _dt.datetime.now().astimezone().utcoffset()
    if offset is None:
        return "UTC"
    total = int(offset.total_seconds())
    sign = "+" if total >= 0 else "-"
    hours, minutes = divmod(abs(total) // 60, 60)
    return f"UTC{sign}{hours:02d}:{minutes:02d}"


def _now() -> str:
    return _dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds")


# --------------------------------------------------------------------------- #
# pulling values out of a fitted result
# --------------------------------------------------------------------------- #
def _result_payload(result: Any) -> dict:
    """A JSON-able snapshot of a fitted model."""
    payload: dict[str, Any] | None = None
    for name in ("to_dict", "as_dict"):
        method = getattr(result, name, None)
        if callable(method):
            try:
                payload = dict(method())
                break
            except Exception:                            # pragma: no cover
                continue
    if payload is None:
        if dataclasses.is_dataclass(result) and not isinstance(result, type):
            payload = {f.name: getattr(result, f.name)
                       for f in dataclasses.fields(result)}
        elif hasattr(result, "__dict__"):
            payload = {k: v for k, v in vars(result).items()
                       if not k.startswith("_")}
        else:                                            # pragma: no cover
            payload = {"repr": repr(result)}
    return {"class": type(result).__name__, "values": payload}


def _settings_from(result: Any) -> dict:
    out: dict[str, Any] = {}
    for name in _SETTING_ATTRS:
        if hasattr(result, name):
            out[name] = getattr(result, name)
    return out


def _arrays_from(result: Any) -> dict:
    out: dict[str, Any] = {}
    for name in _INPUT_ATTRS:
        value = getattr(result, name, None)
        if value is not None:
            out[name] = np.asarray(value).ravel().tolist()
    return out


def _process_from(result: Any) -> dict:
    out: dict[str, Any] = {}
    for name in _PROCESS_ATTRS:
        if hasattr(result, name):
            out[name] = getattr(result, name)
    return out


def _kind_of(result: Any) -> str:
    """``RmaMvResult`` -> ``rma_mv``, ``NetmetaResult`` -> ``netmeta``."""
    name = type(result).__name__
    if name.endswith("Result") and len(name) > len("Result"):
        name = name[: -len("Result")]
    snake = "".join(
        "_" + ch.lower() if ch.isupper() and i else ch.lower()
        for i, ch in enumerate(name)
    )
    return snake.lstrip("_")


# --------------------------------------------------------------------------- #
# the record
# --------------------------------------------------------------------------- #
@dataclass
class AnalysisRecord:
    """One analysis, fully described.

    Attributes:
        kind: short label (``"rma"``, ``"netmeta"``, ...).
        created_utc: ISO-8601 timestamp of when the record was built.
        inputs: the ``输入数据`` block.
        analysis: the ``分析配置`` block.
        process: the ``运算过程`` block.
        environment: the ``运行环境`` block.
        results: the ``结果与诊断`` block.
        schema: layout identifier; see :data:`RECORD_SCHEMA`.
    """

    kind: str
    inputs: dict = field(default_factory=dict)
    analysis: dict = field(default_factory=dict)
    process: dict = field(default_factory=dict)
    environment: dict = field(default_factory=dict)
    results: dict = field(default_factory=dict)
    schema: str = RECORD_SCHEMA
    created_utc: str = field(default_factory=_now)

    # -- serialisation ----------------------------------------------------- #
    def to_dict(self) -> dict:
        """The record as a plain dict, ready to serialise."""
        return {
            "schema": self.schema,
            "kind": self.kind,
            "created_utc": self.created_utc,
            "inputs": self.inputs,
            "analysis": self.analysis,
            "process": self.process,
            "environment": self.environment,
            "results": self.results,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> AnalysisRecord:
        """Rebuild a record from :meth:`to_dict` output."""
        schema = str(payload.get("schema", RECORD_SCHEMA))
        if schema.split("/")[0] != RECORD_SCHEMA.split("/")[0]:
            raise RecordError(
                f"unsupported record schema {schema!r}; this build understands "
                f"{RECORD_SCHEMA!r}"
            )
        missing = [k for k in ("kind", "inputs", "analysis", "results")
                   if k not in payload]
        if missing:
            raise RecordError(
                f"not an analysis record: missing {missing}. "
                f"Expected schema {RECORD_SCHEMA}."
            )
        return cls(
            kind=str(payload["kind"]),
            inputs=dict(payload["inputs"]),
            analysis=dict(payload["analysis"]),
            process=dict(payload.get("process", {})),
            environment=dict(payload.get("environment", {})),
            results=dict(payload["results"]),
            schema=schema,
            created_utc=str(payload.get("created_utc", "")),
        )

    def save(self, path: str | os.PathLike, *, arrays: str = "list") -> Path:
        """Write the record to ``path`` as strict JSON.

        Args:
            path: destination file (``.json``); parents are created.
            arrays: ``"list"`` for readable output, ``"tagged"`` to keep
                dtype and shape of the numeric arrays.

        Returns:
            The destination path.
        """
        return dump_json(self.to_dict(), path, arrays=arrays, sort_keys=False)

    @classmethod
    def load(cls, path: str | os.PathLike) -> AnalysisRecord:
        """Read a record written by :meth:`save`."""
        return cls.from_dict(load_json(path))

    # -- inspection -------------------------------------------------------- #
    @property
    def fingerprint(self) -> dict:
        """The ``inputs.fingerprint`` sub-block."""
        return dict(self.inputs.get("fingerprint", {}))

    @property
    def mode(self) -> str:
        """``"bundle"`` when the data is stored, else ``"digest"``."""
        return str(self.inputs.get("mode", "digest"))

    def summary(self) -> str:
        """A short human-readable description, one line per block."""
        lines = [
            f"{self.kind}  ({self.schema})",
            f"  created      {self.created_utc}",
            f"  pimeta       {self.environment.get('pimeta', '?')}",
            f"  rows         {self.inputs.get('n_rows', '?')}  (mode={self.mode})",
            f"  source fp    {self.fingerprint.get('source', '-')}",
            f"  analysis fp  {self.fingerprint.get('analysis', '-')}",
        ]
        settings = self.analysis.get("settings") or {}
        if settings:
            shown = ", ".join(f"{k}={v}" for k, v in sorted(settings.items())
                              if not isinstance(v, (list, dict)))
            lines.append(f"  settings     {shown}")
        return "\n".join(lines)

    # -- verification ------------------------------------------------------ #
    def verify(self, data: Any = None) -> dict:
        """Recompute the fingerprints and compare them with the record.

        Args:
            data: the table to check.  Defaults to the bundled data, so a
                ``"bundle"`` record verifies itself with no arguments.  A
                record built without a source table still verifies against the
                analysed vectors it stored.

        Returns:
            The recomputed fingerprint block.

        Raises:
            RecordError: when there is nothing to check against.
            RecordMismatch: when a digest disagrees.
        """
        stored = self.fingerprint
        # Mirror the basis the record was built on: a record hashed from the
        # source table must be re-hashed from the table, not from the frozen
        # copies of the analysed vectors.
        basis = stored.get("basis", "data" if "source" in stored else "arrays")

        if data is None and basis == "arrays" and self.inputs.get("arrays"):
            fresh = {
                "algorithm": "sha256",
                "version": FINGERPRINT_VERSION,
                "basis": "arrays",
                "analysis": analysis_fingerprint(
                    settings=self.analysis.get("settings"),
                    arrays=self.inputs["arrays"],
                ),
            }
            if "analysis" in stored and not matches(stored["analysis"],
                                                    fresh["analysis"]):
                raise RecordMismatch(
                    "the stored analysed vectors no longer hash to the recorded "
                    "digest; the record has been edited", []
                )
            return fresh

        if data is None:
            data = self.inputs.get("data")
        if data is None:
            raise RecordError(
                "this record was written in 'digest' mode and carries no data, "
                "so it can only be verified against the original table: "
                "record.verify(data=frame). Re-save with mode='bundle' to make "
                "the record self-contained."
            )

        cols = as_columns(data)
        names = self.inputs.get("columns") or list(cols)
        used = self.inputs.get("analysis_columns") or names
        fresh = {
            "algorithm": "sha256",
            "version": FINGERPRINT_VERSION,
            "basis": basis,
            "source": source_fingerprint(cols, columns=names),
            "analysis": analysis_fingerprint(
                cols if basis == "data" else None,
                columns=used if basis == "data" else None,
                settings=self.analysis.get("settings"),
                arrays=None if basis == "data" else self.inputs.get("arrays"),
            ),
        }
        problems = [k for k in ("source", "analysis")
                    if k in stored and not matches(stored[k], fresh[k])]
        if problems:
            detail = ", ".join(
                f"{k}: recorded {stored.get(k, '-')[:19]}... but recomputed "
                f"{fresh[k][:19]}..."
                for k in problems
            )
            raise RecordMismatch(
                f"the data does not match this record ({detail})", []
            )
        return fresh

    # -- replay ------------------------------------------------------------ #
    def replay(
        self,
        *,
        strict: bool = True,
        rtol: float = 1e-9,
        atol: float = 1e-12,
    ) -> Any:
        """Re-run the analysis and check that the numbers come back.

        Calls ``analysis["function"]`` with ``analysis["call"]``, injecting
        the bundled data when the original call used column names.

        Args:
            strict: raise :class:`RecordMismatch` when any number differs.
            rtol: relative tolerance for the comparison.
            atol: absolute tolerance.

        Returns:
            The freshly fitted result object.

        Raises:
            RecordError: when the record lacks a ``function``/``call`` block
                or the data needed to rebuild the call.
            RecordMismatch: when ``strict`` and the numbers differ.
        """
        name = self.analysis.get("function")
        if not name:
            raise RecordError(
                "this record has no analysis.function, so it cannot be "
                "replayed. Pass function='rma' (etc.) to build_record()."
            )
        call = dict(self.analysis.get("call") or {})
        if not call:
            raise RecordError(
                "this record has no analysis.call, so there is nothing to "
                "replay. Pass call={...} to build_record()."
            )

        data = self.inputs.get("data")
        if self.analysis.get("pass_data"):
            if data is None:
                raise RecordError(
                    "the original call referenced columns by name, but this "
                    "record is in 'digest' mode and has no data. Re-save with "
                    "mode='bundle'."
                )
            call["data"] = data

        function = _resolve_callable(str(name))
        fresh = function(**call)

        differences = compare_results(
            self.results.get("values", {}), _result_payload(fresh)["values"],
            rtol=rtol, atol=atol,
        )
        if differences and strict:
            shown = "; ".join(
                f"{key}: recorded {old!r}, replayed {new!r}"
                for key, old, new in differences[:5]
            )
            raise RecordMismatch(
                f"replay produced different numbers for {self.analysis.get('function')}"
                f" ({len(differences)} field(s)): {shown}",
                differences,
            )
        return fresh


def _resolve_callable(name: str):
    """Look up a pimeta function by name, refusing anything else."""
    import pimeta

    target = getattr(pimeta, name, None)
    if target is None and "." in name:
        module_name, _, attr = name.rpartition(".")
        try:
            module = importlib.import_module(f"pimeta.{module_name}")
        except ImportError:
            module = None
        if module is not None:
            target = getattr(module, attr, None)
    if target is None:
        raise RecordError(
            f"analysis.function {name!r} is not part of pimeta; a record can "
            "only replay functions the installed package provides"
        )
    origin = getattr(target, "__module__", "") or ""
    if not callable(target) or not origin.startswith("pimeta"):
        raise RecordError(
            f"analysis.function {name!r} resolves to a non-pimeta object "
            f"({type(target).__name__}); refusing to call it"
        )
    return target


# --------------------------------------------------------------------------- #
# comparing two result dicts
# --------------------------------------------------------------------------- #
def _numeric_leaves(value: Any, prefix: str = "") -> dict[str, float]:
    """Flatten a result payload into ``{dotted path: number}``."""
    out: dict[str, float] = {}
    if isinstance(value, Mapping):
        for k, v in value.items():
            out.update(_numeric_leaves(v, f"{prefix}.{k}" if prefix else str(k)))
    elif isinstance(value, (list, tuple)):
        for i, v in enumerate(value):
            out.update(_numeric_leaves(v, f"{prefix}[{i}]"))
    elif isinstance(value, np.ndarray):
        out.update(_numeric_leaves(value.tolist(), prefix))
    elif isinstance(value, (bool,)):
        out[prefix] = float(value)
    elif isinstance(value, (int, float, np.integer, np.floating)):
        out[prefix] = float(value)
    return out


def compare_results(
    recorded: Mapping[str, Any],
    fresh: Mapping[str, Any],
    *,
    rtol: float = 1e-9,
    atol: float = 1e-12,
) -> list[tuple[str, Any, Any]]:
    """Compare two result payloads field by field.

    Only numbers are compared -- a changed ``class`` name or a longer
    ``slab`` is not a reproducibility failure.  NaN is considered equal to
    NaN, since "undefined here, undefined there" is a match.

    Args:
        recorded: the ``results.values`` block from the record.
        fresh: the same block recomputed now.
        rtol: relative tolerance.
        atol: absolute tolerance.

    Returns:
        A list of ``(path, recorded value, fresh value)`` for mismatches, in
        insertion order.  Empty means the analyses agree.
    """
    left = _numeric_leaves(recorded)
    right = _numeric_leaves(fresh)
    differences: list[tuple[str, Any, Any]] = []
    for key in left:
        if key not in right:
            continue
        a, b = left[key], right[key]
        if a != a and b != b:                       # NaN matches NaN
            continue
        if not np.isclose(a, b, rtol=rtol, atol=atol, equal_nan=True):
            differences.append((key, a, b))
    return differences


# --------------------------------------------------------------------------- #
# builder
# --------------------------------------------------------------------------- #
def build_record(
    result: Any,
    *,
    kind: str | None = None,
    data: Any = None,
    mode: str = "digest",
    columns: Sequence[str] | None = None,
    analysis_columns: Sequence[str] | None = None,
    settings: Mapping[str, Any] | None = None,
    function: str | None = None,
    call: Mapping[str, Any] | None = None,
    process: Mapping[str, Any] | None = None,
    diagnostics: Mapping[str, Any] | None = None,
    seed: int | None = None,
) -> AnalysisRecord:
    """Wrap a fitted result in an :class:`AnalysisRecord`.

    Args:
        result: anything the package fitted -- an ``RmaResult``,
            ``NetmetaResult``, ``MetagenResult``, ...
        kind: short label; defaults to ``function`` or the class name.
        data: the source table.  Supplying it is what makes the record
            verifiable; without it only the arrays pulled from ``result``
            are fingerprinted.
        mode: ``"digest"`` (default) or ``"bundle"``.
        columns: source columns to fingerprint; defaults to every column of
            ``data``.
        analysis_columns: columns that entered the model; defaults to
            ``columns``.
        settings: values that can move a number.  Defaults to the result's
            own ``method``/``test``/``level``/... attributes.
        function: name of the pimeta function that was called, for
            :meth:`AnalysisRecord.replay`.
        call: the keyword arguments that were passed.  Column names are
            looked up in ``data`` again on replay.
        process: extra ``运算过程`` entries (iterations, convergence, timings).
        diagnostics: extra ``结果与诊断`` entries.
        seed: random seed in force, recorded in the environment block.

    Returns:
        The populated record.

    Raises:
        ValueError: for an unknown ``mode``.
    """
    if mode not in ("digest", "bundle"):
        raise ValueError(f"mode must be 'digest' or 'bundle', not {mode!r}")

    resolved_settings = dict(settings) if settings is not None else _settings_from(result)
    arrays = _arrays_from(result)

    inputs: dict[str, Any] = {"mode": mode}
    cols: dict[str, Any] | None = as_columns(data) if data is not None else None

    if cols is not None:
        names = list(cols) if columns is None else [str(c) for c in columns]
        used = list(names) if analysis_columns is None else [str(c) for c in analysis_columns]
        for label, wanted in (("columns", names), ("analysis_columns", used)):
            missing = [n for n in wanted if n not in cols]
            if missing:
                raise KeyError(f"{label} not in the data: {missing}")
        inputs.update(
            {
                "n_rows": len(cols[names[0]]) if names else 0,
                "columns": names,
                "analysis_columns": used,
                "arrays": arrays,
                "fingerprint": {
                    "algorithm": "sha256",
                    "version": FINGERPRINT_VERSION,
                    "basis": "data",
                    # When the source table travels with the record the analysed
                    # vectors add nothing -- they are the same numbers, already
                    # after the NA filter -- so leave them out. Otherwise the
                    # digest would be impossible to recompute from `data` alone
                    # and `verify()` could never pass on a reloaded record.
                    "source": source_fingerprint(cols, columns=names),
                    "analysis": analysis_fingerprint(
                        cols, columns=used, settings=resolved_settings
                    ),
                },
            }
        )
        if mode == "bundle":
            inputs["data"] = {n: cols[n].tolist() for n in names}
    else:
        inputs["n_rows"] = len(next(iter(arrays.values()), [])) or None
        inputs["arrays"] = arrays
        inputs["fingerprint"] = {
            "algorithm": "sha256",
            "version": FINGERPRINT_VERSION,
            "basis": "arrays",
            "analysis": analysis_fingerprint(
                columns=None, settings=resolved_settings, arrays=arrays
            ),
        }

    call_payload = dict(call or {})
    # The original call used column names; replay has to resolve them again,
    # which only works when the data travels with the record.
    known = cols or {}
    pass_data = bool(known) and any(
        isinstance(v, str) and v in known for v in call_payload.values()
    )

    analysis_block: dict[str, Any] = {"settings": resolved_settings}
    if function:
        analysis_block["function"] = function
    if call_payload:
        analysis_block["call"] = call_payload
        analysis_block["pass_data"] = pass_data

    process_block = _process_from(result)
    process_block.update(process or {})

    results_block = _result_payload(result)
    if diagnostics:
        results_block["diagnostics"] = dict(diagnostics)

    return AnalysisRecord(
        kind=str(kind or function or _kind_of(result)),
        inputs=inputs,
        analysis=analysis_block,
        process=process_block,
        environment=environment(seed=seed),
        results=results_block,
    )


def save_record(
    path: str | os.PathLike,
    result: Any,
    **kwargs: Any,
) -> AnalysisRecord:
    """Build a record and write it in one call.

    ``save_record(path, result, data=frame, mode="bundle", function="rma",
    call={"yi": "TE", "vi": "seTE2", "method": "REML"})`` saves and returns
    the record; ``load_record(path).replay()`` re-runs it.

    Args:
        path: destination JSON file.
        result: the fitted result.
        **kwargs: forwarded to :func:`build_record`.

    Returns:
        The record that was written.
    """
    arrays = kwargs.pop("arrays", "list")
    record = build_record(result, **kwargs)
    record.save(path, arrays=arrays)
    return record


def load_record(path: str | os.PathLike) -> AnalysisRecord:
    """Read a record from disk."""
    return AnalysisRecord.load(path)
