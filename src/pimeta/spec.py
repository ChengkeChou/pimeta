"""Declared analysis decisions, enumerated and identified by their fingerprint.

An analysis specification is a written record of the choices that were not forced
by the data, each with the values that are statistically defensible and a
one-line reason for each. Two rules keep it from turning into a toy:

**A decision enters only if more than one choice is defensible.** A setting with
a single sensible value is not a decision, it is the method; :class:`Decision`
refuses it rather than letting a specification pad itself with non-choices.

**The number of specifications is a consequence, never a target.** ``expand()``
returns the Cartesian product of what was declared. It is not a search, it does
not sample, and nothing here suggests that a larger grid is more thorough.

Every expanded specification carries an id derived from its own choices by
:func:`pimeta.record.fingerprint`. The id is computed from the choices alone, not
from the justifications, so editing prose does not invalidate a recorded run and
two runs that made the same decisions collide on purpose -- that is how they are
recognised as the same analysis.
"""

from __future__ import annotations

import itertools
import json
import math
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .record.fingerprint import fingerprint

__all__ = [
    "SPEC_VERSION",
    "Decision",
    "Specification",
    "AnalysisSpec",
    "RunOutcome",
    "execute",
]

#: Version of the specification layout written into the file and into every id.
SPEC_VERSION = 1


@dataclass(frozen=True)
class Decision:
    """One choice that the data did not force, with its admissible values.

    Attributes:
        name: the key that appears in a run record and in a table of results.
        values: the admissible values, in the order the reader should meet them.
        reasons: one line per value, saying why that value is defensible. Every
            value needs one: a value nobody can justify is not an option, and an
            empty string is not a justification.
        default: the value used when a single specification is wanted; must be
            one of ``values``.

    Raises:
        ValueError: for fewer than two values, a missing or blank reason, a
            duplicate value, or a default that is not among the values.
    """

    name: str
    values: tuple[Any, ...]
    reasons: Mapping[Any, str]
    default: Any = None

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("a decision needs a name")
        values = tuple(self.values)
        if len(values) < 2:
            raise ValueError(
                f"decision {self.name!r} has {len(values)} value(s); a setting "
                "with one defensible value is the method, not a decision")
        if len({_key(v) for v in values}) != len(values):
            raise ValueError(f"decision {self.name!r} repeats a value")
        missing = [v for v in values if not str(self.reasons.get(v, "")).strip()]
        if missing:
            raise ValueError(
                f"decision {self.name!r} has no justification for {missing}; a "
                "value that cannot be argued for is not an option")
        if self.default is not None and self.default not in values:
            raise ValueError(
                f"decision {self.name!r} has default {self.default!r}, which is "
                f"not one of its values")
        object.__setattr__(self, "values", values)

    @property
    def n_values(self) -> int:
        return len(self.values)

    def reason(self, value: Any) -> str:
        return str(self.reasons[value])

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "values": [_plain(v) for v in self.values],
            "reasons": {str(_plain(v)): str(self.reasons[v]) for v in self.values},
            "default": None if self.default is None else _plain(self.default),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> Decision:
        values = tuple(payload["values"])
        reasons = {v: str(payload["reasons"][str(v)]) for v in values}
        default = payload.get("default")
        return cls(name=str(payload["name"]), values=values, reasons=reasons,
                   default=default)


@dataclass(frozen=True)
class Specification:
    """One point of the grid: the choices, and the id they hash to."""

    id: str
    choices: tuple[tuple[str, Any], ...]

    def get(self, name: str) -> Any:
        for key, value in self.choices:
            if key == name:
                return value
        raise KeyError(name)

    def as_dict(self) -> dict[str, Any]:
        return {key: value for key, value in self.choices}

    def to_dict(self) -> dict:
        return {"id": self.id,
                "choices": {k: _plain(v) for k, v in self.choices}}

    def __str__(self) -> str:
        shown = ", ".join(f"{k}={v}" for k, v in self.choices)
        return f"{_short_id(self.id)}: {shown}"


@dataclass
class AnalysisSpec:
    """A set of declared decisions and the grid it defines.

    Attributes:
        decisions: the declared decisions, in the order they should be read.
        version: layout version, written to disk and into every id.
    """

    decisions: tuple[Decision, ...]
    version: int = SPEC_VERSION

    def __post_init__(self) -> None:
        decisions = tuple(self.decisions)
        names = [d.name for d in decisions]
        if len(set(names)) != len(names):
            raise ValueError(f"two decisions share a name: {names}")
        if not decisions:
            raise ValueError("a specification needs at least one decision")
        object.__setattr__(self, "decisions", decisions)

    # ---------------------------------------------------------------- shape

    def size(self) -> int:
        """How many specifications this grid defines."""
        return math.prod(d.n_values for d in self.decisions)

    @property
    def names(self) -> tuple[str, ...]:
        return tuple(d.name for d in self.decisions)

    def decision(self, name: str) -> Decision:
        for d in self.decisions:
            if d.name == name:
                return d
        raise KeyError(name)

    # ------------------------------------------------------------- expanding

    def _spec(self, choices: Mapping[str, Any]) -> Specification:
        payload = {"v": self.version,
                   "choices": {k: _plain(v) for k, v in choices.items()}}
        return Specification(id=fingerprint(payload),
                             choices=tuple((k, v) for k, v in choices.items()))

    def expand(self) -> list[Specification]:
        """The whole grid, in a deterministic order.

        The order is the product of the decisions in the order they were
        declared, with each decision's values in their declared order, so two
        runs produce the same list in the same order. The ids do not depend on
        the order at all -- they are hashes of the choices.
        """
        combos = itertools.product(*(d.values for d in self.decisions))
        return [self._spec(dict(zip(self.names, combo))) for combo in combos]

    def default(self) -> Specification:
        """The single specification made of each decision's default.

        Requires every decision to declare a default: without one there is no
        answer to "what would you have reported".
        """
        missing = [d.name for d in self.decisions if d.default is None]
        if missing:
            raise ValueError(
                f"no default declared for {missing}; a specification cannot be "
                "collapsed to one analysis without knowing which value was "
                "the primary one")
        return self._spec({d.name: d.default for d in self.decisions})

    def get(self, id_or_choices: Any) -> Specification:
        """Look a specification up by id or by a complete set of choices."""
        if isinstance(id_or_choices, str):
            for spec in self.expand():
                if spec.id == id_or_choices:
                    return spec
            raise KeyError(f"no specification with id {id_or_choices!r}")
        choices = dict(id_or_choices)
        unknown = set(choices) - set(self.names)
        if unknown:
            raise KeyError(f"unknown decision(s): {sorted(unknown)}")
        missing = [n for n in self.names if n not in choices]
        if missing:
            raise KeyError(f"missing decision(s): {missing}")
        for name, value in choices.items():
            if value not in self.decision(name).values:
                raise ValueError(
                    f"{name}={value!r} is not an admissible value; "
                    f"expected one of {self.decision(name).values}")
        return self._spec({n: choices[n] for n in self.names})

    # -------------------------------------------------------------- file i/o

    def to_dict(self) -> dict:
        return {
            "kind": "pimeta.analysis_spec",
            "version": self.version,
            "size": self.size(),
            "decisions": [d.to_dict() for d in self.decisions],
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> AnalysisSpec:
        kind = payload.get("kind")
        if kind is not None and kind != "pimeta.analysis_spec":
            raise ValueError(f"not an analysis specification: {kind!r}")
        version = int(payload.get("version", SPEC_VERSION))
        if version != SPEC_VERSION:
            raise ValueError(
                f"specification version {version} cannot be read by this "
                f"version of pimeta, which writes {SPEC_VERSION}; the layout "
                "changed and the two must not be silently mixed")
        decisions = tuple(Decision.from_dict(d) for d in payload["decisions"])
        spec = cls(decisions=decisions, version=version)
        declared = payload.get("size")
        if declared is not None and int(declared) != spec.size():
            raise ValueError(
                f"the file declares {declared} specifications but its decisions "
                f"define {spec.size()}; one of the two was edited by hand")
        return spec

    def save(self, path: str | Path) -> Path:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(self.to_dict(), indent=2) + "\n",
                          encoding="utf-8", newline="\n")
        return target

    @classmethod
    def load(cls, path: str | Path) -> AnalysisSpec:
        return cls.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))

    def __str__(self) -> str:
        lines = [f"AnalysisSpec: {len(self.decisions)} decision(s), "
                 f"{self.size()} specification(s)"]
        for d in self.decisions:
            default = "" if d.default is None else f" (default {d.default!r})"
            lines.append(f"  {d.name}: {d.n_values} value(s){default}")
            for value in d.values:
                lines.append(f"    {value!r}: {d.reason(value)}")
        return "\n".join(lines)


@dataclass(frozen=True)
class RunOutcome:
    """What happened when one specification was run.

    A failed specification is an outcome, not an exception: ``ok`` is ``False``,
    ``error`` says why, and the entry stays in the list. Dropping failures is
    the largest silent bias in a published multiverse, and it starts here.
    """

    id: str
    choices: dict[str, Any]
    ok: bool
    result: Any = field(default=None, repr=False)
    error: str = ""
    diagnostics: tuple[str, ...] = ()

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "choices": {k: _plain(v) for k, v in self.choices.items()},
            "ok": self.ok,
            "error": self.error,
            "diagnostics": list(self.diagnostics),
        }


def execute(
    spec: AnalysisSpec,
    run: Callable[[dict[str, Any]], Any],
    *,
    on_error: str = "record",
    specs: list[Specification] | None = None,
) -> list[RunOutcome]:
    """Run every specification and record what happened.

    Args:
        spec: the grid.
        run: called with one ``{decision: value}`` mapping per specification; it
            receives the choices, so it is the caller's job to turn them into
            keyword arguments for a fit.
        on_error: ``"record"`` (default) keeps a failed specification in the
            output with its error message; ``"raise"`` re-raises instead, which
            is what a script that cannot continue should ask for.
        specs: an explicit subset of specifications to run; defaults to the whole
            grid.

    Returns:
        One :class:`RunOutcome` per specification, in the order given.

    Raises:
        ValueError: for an unknown ``on_error``.
    """
    if on_error not in ("record", "raise"):
        raise ValueError(f"on_error must be 'record' or 'raise', "
                         f"not {on_error!r}")
    outcomes: list[RunOutcome] = []
    for one in (spec.expand() if specs is None else specs):
        choices = one.as_dict()
        try:
            result = run(dict(choices))
        except Exception as exc:  # noqa: BLE001 - recording is the point
            if on_error == "raise":
                raise
            outcomes.append(RunOutcome(id=one.id, choices=choices, ok=False,
                                       error=f"{type(exc).__name__}: {exc}"))
            continue
        codes: tuple[str, ...] = ()
        diag = getattr(result, "diagnostics", None)
        if diag is not None and hasattr(diag, "codes"):
            codes = tuple(str(c) for c in diag.codes())
        outcomes.append(RunOutcome(id=one.id, choices=choices, ok=True,
                                   result=result, diagnostics=codes))
    return outcomes


# ------------------------------------------------------------------ helpers

def _key(value: Any) -> str:
    return json.dumps(_plain(value), sort_keys=True, default=str)


def _plain(value: Any) -> Any:
    """A JSON-able stand-in for a decision value, keeping the value itself."""
    if isinstance(value, (str, bool, int, float)) or value is None:
        return value
    return str(value)


def _short_id(id: str) -> str:
    return id.replace("sha256:", "")[:12]
