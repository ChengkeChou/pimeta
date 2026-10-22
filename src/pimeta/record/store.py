"""A named registry of fitted models -- the equivalent of ``estimates store``.

Fitting one model is rarely the point.  A paper compares a common-effects and
a random-effects fit, three tau^2 estimators, or a sensitivity analysis with
one study dropped.  Those fits need names, they need to survive a kernel
restart, and they need to be compared without re-running anything.

``EstimatesStore`` gives them names::

    from pimeta.record import EstimatesStore

    models = EstimatesStore("primary")
    models.store("fe", pimeta.rma(yi, vi=vi, method="FE"))
    models.store("re", pimeta.rma(yi, vi=vi, method="REML"))
    models.compare("fe", "re")

Each entry keeps the full :class:`~pimeta.record.schema.AnalysisRecord`, so a
stored model is traceable on its own, and the live result object, so
``restore()`` hands back the real thing while the process is running.  After
``save()``/``load()`` the live object is gone and ``restore()`` raises: a
record can prove what the numbers were, but it cannot pretend to be the
fitted model.
"""

from __future__ import annotations

import os
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

from .jsonio import dump as dump_json
from .jsonio import load as load_json
from .schema import AnalysisRecord, RecordError, build_record

__all__ = ["EstimatesStore", "STORE_SCHEMA", "StoredModel", "estimates_store"]

#: Layout identifier for a saved store.
STORE_SCHEMA = "pimeta.store/1"

#: Alternate spellings used across the package for the coefficient table.
_ESTIMATE_KEYS = ("b", "estimate", "coef", "TE", "mu")
_SE_KEYS = ("se", "SE")
_STAT_KEYS = ("zval", "z", "statistic", "tval")
_PVAL_KEYS = ("pval", "p", "pval_z")
_LB_KEYS = ("ci_lb", "ci.lb", "lower")
_UB_KEYS = ("ci_ub", "ci.ub", "upper")


def _first(values: Mapping[str, Any], names: Sequence[str]) -> list | None:
    for name in names:
        if name in values:
            value = values[name]
            return list(value) if isinstance(value, (list, tuple)) else [value]
    return None


@dataclass
class StoredModel:
    """One stored fit: its record plus a few read-only conveniences."""

    key: str
    record: AnalysisRecord
    note: str | None = None
    live: Any = field(default=None, repr=False, compare=False)

    @property
    def kind(self) -> str:
        """``"rma"``, ``"netmeta"``, ... -- taken from the record."""
        return self.record.kind

    @property
    def values(self) -> dict:
        """The ``results.values`` block."""
        return dict(self.record.results.get("values", {}))

    def coef_table(self) -> dict:
        """The coefficient table as ``{column: list}``.

        Returns an empty dict when the model has no named coefficients (a
        network meta-analysis, for example).  Column names follow metafor's
        ``print`` layout so the table can be compared with R output directly.
        """
        values = self.values
        estimates = _first(values, _ESTIMATE_KEYS)
        if estimates is None:
            return {}
        n = len(estimates)
        terms = values.get("coefnames") or values.get("term") or []
        if len(terms) != n:
            terms = ["(Intercept)"] if n == 1 else [f"x{i + 1}" for i in range(n)]
        table = {"model": [self.key] * n, "term": list(terms), "estimate": estimates}
        for column, keys in (
            ("se", _SE_KEYS),
            ("statistic", _STAT_KEYS),
            ("pval", _PVAL_KEYS),
            ("ci_lb", _LB_KEYS),
            ("ci_ub", _UB_KEYS),
        ):
            found = _first(values, keys)
            if found is not None and len(found) == n:
                table[column] = found
        return table

    def summary(self) -> str:
        """A one-line description used by :meth:`EstimatesStore.summary`."""
        values = self.values
        bits = [f"{self.key}: {self.kind}"]
        if "k" in values:
            bits.append(f"k={values['k']}")
        point = _first(values, _ESTIMATE_KEYS)
        if point:
            bits.append(f"est={point[0]:.4g}")
        if "tau2" in values:
            bits.append(f"tau2={values['tau2']:.4g}")
        if "I2" in values:
            bits.append(f"I2={values['I2']:.1f}%")
        return "  ".join(bits)


class EstimatesStore:
    """An ordered, named collection of fitted models.

    Args:
        name: label written into the saved file; purely descriptive.
        path: default file used by :meth:`save` and :meth:`load`.
    """

    def __init__(self, name: str = "pimeta", *, path: str | os.PathLike | None = None):
        self.name = str(name)
        self.path = None if path is None else str(path)
        self._models: dict[str, StoredModel] = {}

    # -- container protocol ------------------------------------------------ #
    def __len__(self) -> int:
        return len(self._models)

    def __iter__(self) -> Iterator[str]:
        return iter(self._models)

    def __contains__(self, key: object) -> bool:
        return key in self._models

    def __getitem__(self, key: str) -> StoredModel:
        return self._models[key]

    def keys(self) -> list[str]:
        """Stored names, in insertion order."""
        return list(self._models)

    def items(self):
        """``(name, StoredModel)`` pairs, in insertion order."""
        return self._models.items()

    # -- storing ----------------------------------------------------------- #
    def store(
        self,
        key: str,
        result: Any,
        *,
        note: str | None = None,
        **record_kwargs: Any,
    ) -> StoredModel:
        """Save ``result`` under ``key``, replacing any entry of that name.

        Args:
            key: the name to look the model up by later.
            result: a fitted result object.
            note: free text kept with the entry, e.g. "sensitivity, no Smith 2019".
            **record_kwargs: forwarded to
                :func:`~pimeta.record.schema.build_record` -- pass ``data=``,
                ``function=``, ``call=``, ``mode="bundle"`` to make the entry
                replayable.

        Returns:
            The stored entry.
        """
        if not str(key).strip():
            raise ValueError("a stored model needs a non-empty name")
        record = build_record(result, **record_kwargs)
        entry = StoredModel(key=str(key), record=record, note=note, live=result)
        self._models[str(key)] = entry
        return entry

    def restore(self, key: str) -> Any:
        """Return the live result object stored under ``key``.

        Raises:
            KeyError: when ``key`` is not stored.
            RecordError: when the entry came from :meth:`load`, where only the
                record exists.
        """
        entry = self._models[key]
        if entry.live is None:
            raise RecordError(
                f"{key!r} was loaded from a file, so only its record survives. "
                "Use store[key].coef_table() / store[key].record.replay()."
            )
        return entry.live

    def drop(self, key: str) -> None:
        """Remove ``key``; a missing key is not an error."""
        self._models.pop(key, None)

    def clear(self) -> None:
        """Forget every stored model."""
        self._models.clear()

    # -- comparison -------------------------------------------------------- #
    def compare(self, *keys: str) -> dict:
        """One long-format table across models -- the ``estimates table`` view.

        Args:
            *keys: models to include; all of them when omitted.

        Returns:
            ``{"model": [...], "term": [...], "estimate": [...], ...}`` with
            one block of rows per model.  Models without a coefficient table
            contribute no rows.
        """
        chosen = list(keys) if keys else self.keys()
        merged: dict[str, list] = {}
        for key in chosen:
            if key not in self._models:
                raise KeyError(f"no stored model named {key!r}; have {self.keys()}")
            table = self._models[key].coef_table()
            for column, values in table.items():
                merged.setdefault(column, []).extend(values)
        return merged

    def summary(self) -> str:
        """One line per stored model."""
        if not self._models:
            return f"{self.name}: (empty)"
        lines = [f"{self.name}  ({len(self._models)} model(s))"]
        lines.extend("  " + entry.summary() for entry in self._models.values())
        return "\n".join(lines)

    # -- persistence ------------------------------------------------------- #
    def to_dict(self) -> dict:
        """The store as a plain dict."""
        return {
            "schema": STORE_SCHEMA,
            "name": self.name,
            "models": {
                key: {"note": entry.note, "record": entry.record.to_dict()}
                for key, entry in self._models.items()
            },
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> EstimatesStore:
        """Rebuild a store from :meth:`to_dict` output."""
        schema = str(payload.get("schema", STORE_SCHEMA))
        if schema.split("/")[0] != STORE_SCHEMA.split("/")[0]:
            raise RecordError(
                f"unsupported store schema {schema!r}; this build understands "
                f"{STORE_SCHEMA!r}"
            )
        store = cls(str(payload.get("name", "pimeta")))
        for key, entry in dict(payload.get("models", {})).items():
            store._models[str(key)] = StoredModel(
                key=str(key),
                record=AnalysisRecord.from_dict(entry["record"]),
                note=entry.get("note"),
                live=None,
            )
        return store

    def save(self, path: str | os.PathLike | None = None, *, arrays: str = "list") -> str:
        """Write the store to JSON.

        Args:
            path: destination; defaults to :attr:`path`.
            arrays: ``"list"`` or ``"tagged"``.

        Returns:
            The destination path.

        Raises:
            ValueError: when no path was given and none was set on the store.
        """
        target = path or self.path
        if target is None:
            raise ValueError("no path given and the store has none; pass save(path)")
        dump_json(self.to_dict(), target, arrays=arrays)
        self.path = str(target)
        return self.path

    @classmethod
    def load(cls, path: str | os.PathLike) -> EstimatesStore:
        """Read a store written by :meth:`save`."""
        store = cls.from_dict(load_json(path))
        store.path = str(path)
        return store


#: Process-local default store, mirroring R's session-level ``estimates``.
#: Convenient for interactive work; create your own instance for anything
#: that has to be reproducible.
estimates_store = EstimatesStore("session")
