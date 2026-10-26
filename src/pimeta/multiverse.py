"""The statistics of stability across a declared set of analyses.

A multiverse is not a forest plot of everything that was tried. What it has to
report is how far the answer moves across analyses that were *all defensible in
advance*, and what the movement is attributable to. This module computes five
things and nothing else:

1. the counts — specifications considered, run, usable and failed;
2. the sign agreement rate, and the share of intervals that exclude zero;
3. the range and distribution of the point estimate;
4. the specification curve, ordered by estimate;
5. a variance decomposition across decision axes, **labelled exploratory**.

Three statements about what these numbers are not, which the guide repeats and
which the code cannot enforce:

* ``M`` analyses are not ``M`` independent experiments. The share of
  specifications whose interval excludes zero is **not** a probability that the
  effect is real, and averaging it with `1 / M` weights would be a statement
  about the grid rather than about the evidence.
* Which decisions "matter" is confounded by an unbalanced grid and by decisions
  that are correlated with one another, so the decomposition is descriptive.
  It says where the variation sits, not what caused it.
* A specification that failed is kept and counted. Dropping the runs that did not
  converge is the largest silent bias in a published multiverse: the analyses
  that break are not a random subset of the analyses that work.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from .checks import Diagnosis, Diagnostic
from .spec import AnalysisSpec, RunOutcome, Specification, execute

__all__ = ["MultiverseResult", "multiverse", "DEFAULT_EXTRACT",
           "SpecificationReport", "specification_report"]


def DEFAULT_EXTRACT(result: Any) -> tuple[float, float, float]:
    """``(estimate, ci_lb, ci_ub)`` for an intercept-only fit.

    Reads ``b[0]``, ``ci_lb[0]`` and ``ci_ub[0]``, which is what every
    single-parameter result in this package exposes. A model with moderators
    gives its intercept, which is rarely the quantity of interest; pass an
    explicit ``extract`` for those rather than being surprised by this one.
    """
    est = float(np.asarray(result.b, dtype=float).ravel()[0])
    lb = float(np.asarray(result.ci_lb, dtype=float).ravel()[0])
    ub = float(np.asarray(result.ci_ub, dtype=float).ravel()[0])
    return est, lb, ub


@dataclass
class MultiverseResult:
    """What the declared analyses did, including the ones that failed."""

    outcomes: list[RunOutcome]
    choices: list[dict[str, Any]]
    estimates: list[float]
    ci_lb: list[float]
    ci_ub: list[float]
    ids: list[str]
    failed: list[dict[str, Any]]
    considered: int
    decisions: tuple[str, ...]
    #: Specifications that ran and produced an estimate but carried a
    #: diagnostic. ``excluded`` marks the ones dropped from ``usable`` because
    #: the fit reported an ``error``: a number a fit says is not reportable
    #: does not belong in a stability statistic.
    flagged: list[dict[str, Any]] = field(default_factory=list)

    # ---------------------------------------------------------------- counts

    @property
    def run(self) -> int:
        return len(self.outcomes)

    @property
    def usable(self) -> int:
        return len(self.estimates)

    @property
    def failure_rate(self) -> float:
        return 0.0 if not self.outcomes else len(self.failed) / len(self.outcomes)

    # -------------------------------------------------------------- the curve

    def counts(self) -> dict[str, int]:
        """Considered, run, usable, failed, flagged.

        ``usable`` counts the specifications the curve is built from. A
        specification that ran but was excluded for an ``error``-severity
        diagnostic is in neither ``usable`` nor ``failed``: it is ``flagged``,
        because "the fit said these numbers are not reportable" is a different
        fact from "the fit crashed", and collapsing the two would hide the
        first behind the second.
        """
        return {"considered": self.considered, "run": self.run,
                "usable": self.usable, "failed": len(self.failed),
                "flagged": len(self.flagged)}

    # ---------------------------------------------------------- diagnostics

    @property
    def diagnostics(self) -> Diagnosis:
        """What a reader has to know before reading the curve.

        A multiverse reports a spread over the specifications that produced a
        number. Every specification that did not is a hole in that spread, and
        the count of holes is the first thing that decides whether the spread
        means anything: a grid with 60 of 64 specifications missing is not a
        narrow result, it is a broken analysis with a narrow-looking report.

        The codes are about the *grid*, not about any one fit. Per-fit findings
        live on the fits and are summarised in :attr:`flagged`.
        """
        items: list[Diagnostic] = []
        counts = self.counts()
        excluded = sum(1 for f in self.flagged if f["excluded"])
        if counts["usable"] == 0:
            items.append(Diagnostic(
                code="no_usable_specification", category="inference",
                severity="error",
                message=(f"none of the {counts['run']} specification(s) that ran "
                         f"produced a usable estimate ({counts['failed']} failed, "
                         f"{excluded} excluded)"),
                action=("there is no stability statistic to read; inspect "
                        "`failed` and `flagged` before the model or the data")))
            return Diagnosis(items)
        if counts["failed"]:
            items.append(Diagnostic(
                code="specification_failed", category="inference",
                severity="warning",
                message=(f"{counts['failed']} of {counts['run']} specification(s) "
                         "produced no estimate and are absent from the curve"),
                action=("read `failed`: the analyses that break are not a random "
                        "subset of the analyses that work")))
        if excluded:
            items.append(Diagnostic(
                code="specification_excluded", category="inference",
                severity="warning",
                message=(f"{excluded} of {counts['run']} specification(s) were left "
                         "out because their own fit reported an error"),
                action=("read `flagged`; pass require_clean=False to see them in "
                        "the curve as well")))
        return Diagnosis(items)

    def curve(self) -> list[dict[str, Any]]:
        """The usable specifications, ordered by estimate.

        Ordered rather than listed in grid order, because the question the curve
        answers is "where does the answer sit", not "in what order did the loop
        visit the cells".
        """
        rows = [{"id": self.ids[i], "estimate": self.estimates[i],
                 "ci_lb": self.ci_lb[i], "ci_ub": self.ci_ub[i],
                 **{k: v for k, v in self.choices[i].items()}}
                for i in range(self.usable)]
        return sorted(rows, key=lambda r: r["estimate"])

    def summary(self) -> dict[str, Any]:
        """The five statistics, with the caveats attached to the fragile ones."""
        est = np.asarray(self.estimates, dtype=float)
        if est.size == 0:
            # The same keys as the full block, so a caller reading `summary()`
            # does not have to know which branch produced it. An empty grid is
            # exactly when someone is most likely to be reading this.
            return {"counts": self.counts(), "failure_rate": self.failure_rate,
                    "flagged": [dict(f) for f in self.flagged],
                    "estimates": None,
                    "note": "no specification produced an estimate"}
        lb = np.asarray(self.ci_lb, dtype=float)
        ub = np.asarray(self.ci_ub, dtype=float)
        positive = int(np.sum(est > 0.0))
        negative = int(np.sum(est < 0.0))
        excludes_zero = int(np.sum((lb > 0.0) | (ub < 0.0)))
        return {
            "counts": self.counts(),
            "failure_rate": self.failure_rate,
            "flagged": [dict(f) for f in self.flagged],
            "estimates": {
                "min": float(est.min()),
                "max": float(est.max()),
                "range": float(est.max() - est.min()),
                "median": float(np.median(est)),
                "mean": float(est.mean()),
                "sd": float(est.std(ddof=1)) if est.size > 1 else float("nan"),
                "q25": float(np.percentile(est, 25)),
                "q75": float(np.percentile(est, 75)),
            },
            "sign": {
                "positive": positive,
                "negative": negative,
                "zero": int(est.size - positive - negative),
                "agreement": max(positive, negative) / est.size,
                "agreement_note": "the larger of the two shares; 0.5 means the "
                                  "sign is a coin flip across defensible "
                                  "analyses",
            },
            "excludes_zero": {
                "count": excludes_zero,
                "share": excludes_zero / est.size,
                "note": "not a probability that the effect is real: the analyses "
                        "are not independent experiments and the grid is not a "
                        "prior",
            },
        }

    def variance_axes(self) -> list[dict[str, Any]]:
        """How much of the spread each decision accounts for, descriptively.

        For one decision the statistic is the between-level variance of the mean
        estimate, weighted by how many usable specifications each level has,
        divided by the total variance of the estimates. It is a one-way
        decomposition, so on a balanced grid of additive decisions the shares
        add up to one; where decisions interact they do not, because a one-way
        decomposition has nowhere to put an interaction. That shortfall is a
        property of the statistic and not of the grid, and it is the second
        reason the result is called exploratory.
        """
        est = np.asarray(self.estimates, dtype=float)
        total = float(est.var()) if est.size > 1 else 0.0
        rows: list[dict[str, Any]] = []
        for name in self.decisions:
            groups: dict[str, list[float]] = {}
            for i in range(self.usable):
                key = str(self.choices[i].get(name))
                groups.setdefault(key, []).append(self.estimates[i])
            if len(groups) < 2:
                continue
            grand = float(est.mean())
            between = 0.0
            for values in groups.values():
                n = len(values)
                if n == 0:
                    continue
                between += n * (float(np.mean(values)) - grand) ** 2
            between /= est.size
            rows.append({
                "decision": name,
                "levels": len(groups),
                "between_variance": between,
                "share": (between / total) if total > 0 else float("nan"),
                "level_means": {k: float(np.mean(v))
                                for k, v in sorted(groups.items())},
                "note": "descriptive: an unbalanced grid and correlated "
                        "decisions both move this number",
            })
        return sorted(rows, key=lambda r: -r["between_variance"])

    def to_dict(self) -> dict:
        return {
            "counts": self.counts(),
            "failure_rate": self.failure_rate,
            "summary": self.summary(),
            "variance_axes": self.variance_axes(),
            "curve": self.curve(),
            "failed": [dict(f) for f in self.failed],
            "flagged": [dict(f) for f in self.flagged],
            "decisions": list(self.decisions),
        }

    def __str__(self) -> str:
        counts = self.counts()
        lines = [f"Multiverse over {counts['considered']} specification(s): "
                 f"{counts['usable']} usable, {counts['failed']} failed"
                 + (f", {counts['flagged']} flagged" if counts["flagged"] else "")]
        block = self.summary()
        est = block.get("estimates")
        if est is None:
            lines.append("  no specification produced an estimate")
        else:
            lines.append(f"  estimate range {est['min']:+.4f} to "
                         f"{est['max']:+.4f} (median {est['median']:+.4f})")
            lines.append(f"  sign agreement {block['sign']['agreement']:.2f}; "
                         f"{block['excludes_zero']['count']} of "
                         f"{self.usable} intervals exclude zero")
        for row in self.variance_axes():
            share = row["share"]
            shown = "n/a" if share != share else f"{share:.2f}"
            lines.append(f"  {row['decision']}: {shown} of the spread")
        if self.failed:
            lines.append("  failed specifications:")
            for item in self.failed:
                choices = ", ".join(f"{k}={v}" for k, v in item["choices"].items())
                lines.append(f"    {item['id'][:12]} {choices}: {item['error']}")
        if self.flagged:
            lines.append("  specifications carrying a diagnostic:")
            for item in self.flagged:
                choices = ", ".join(f"{k}={v}" for k, v in item["choices"].items())
                mark = "excluded" if item["excluded"] else "kept"
                lines.append(f"    {item['id'][:12]} {choices}: "
                             f"{', '.join(item['codes'])} ({mark})")
        return "\n".join(lines)


def multiverse(
    spec: AnalysisSpec,
    run: Callable[[dict[str, Any]], Any],
    *,
    extract: Callable[[Any], tuple[float, float, float]] | None = None,
    on_error: str = "record",
    specs: list[Specification] | None = None,
    require_clean: bool = True,
) -> MultiverseResult:
    """Run every specification and summarise how stable the answer is.

    Args:
        spec: the declared decisions.
        run: called with one ``{decision: value}`` mapping per specification.
        extract: turns a result into ``(estimate, ci_lb, ci_ub)``; defaults to
            :func:`DEFAULT_EXTRACT`, which reads the first coefficient.
        on_error: ``"record"`` (default) keeps failures; ``"raise"`` re-raises.
        specs: an explicit subset to run; defaults to the whole grid.
        require_clean: when true (default), a specification whose fit reports an
            ``error``-severity diagnostic is moved to ``flagged`` and kept out of
            ``usable``. A fit that says its own numbers are not reportable should
            not contribute to a stability statistic; set this to ``False`` to
            include such fits anyway, which flags them but does not drop them.

    Returns:
        :class:`MultiverseResult`. Failures are counted and named, never dropped.
        ``flagged`` and ``failed`` are disjoint, so
        ``run == usable + failed + excluded`` holds, where ``excluded`` counts
        the flagged entries with ``excluded`` set.

    Raises:
        ValueError: when a result cannot be turned into an estimate, because a
            silent skip would turn a broken specification into a missing row.
    """
    get = extract or DEFAULT_EXTRACT
    outcomes = execute(spec, run, on_error=on_error, specs=specs)

    estimates: list[float] = []
    ci_lb: list[float] = []
    ci_ub: list[float] = []
    ids: list[str] = []
    choices: list[dict[str, Any]] = []
    failed: list[dict[str, Any]] = []
    flagged: list[dict[str, Any]] = []
    for outcome in outcomes:
        if not outcome.ok:
            failed.append({"id": outcome.id, "choices": dict(outcome.choices),
                           "error": outcome.error})
            continue
        # The fit's own verdict, not a re-derivation of it: `execute` already
        # collected the codes, and the severity lives on the result.
        diag = getattr(outcome.result, "diagnostics", None)
        codes = tuple(str(c) for c in diag.codes()) \
            if diag is not None and hasattr(diag, "codes") \
            else tuple(outcome.diagnostics)
        worst = getattr(diag, "worst", None) if diag is not None else None
        excluded = bool(require_clean and worst == "error")
        try:
            est, lb, ub = get(outcome.result)
        except Exception as exc:  # noqa: BLE001
            # A result that ran but cannot be summarised is a failure with a
            # different cause, and it belongs in the same list.
            failed.append({"id": outcome.id, "choices": dict(outcome.choices),
                           "error": f"extract: {type(exc).__name__}: {exc}"})
            continue
        if not (math.isfinite(est) and math.isfinite(lb) and math.isfinite(ub)):
            failed.append({"id": outcome.id, "choices": dict(outcome.choices),
                           "error": "extract: a non-finite estimate"})
            continue
        # Recorded only once the row is known not to be a failure, so that
        # `flagged` and `failed` never both name the same specification and
        # `run == usable + failed + excluded` holds exactly.
        if codes:
            flagged.append({"id": outcome.id, "choices": dict(outcome.choices),
                            "codes": list(codes), "worst": worst,
                            "excluded": excluded})
        if excluded:
            continue
        ids.append(outcome.id)
        choices.append(dict(outcome.choices))
        estimates.append(float(est))
        ci_lb.append(float(lb))
        ci_ub.append(float(ub))

    return MultiverseResult(
        outcomes=outcomes, choices=choices, estimates=estimates,
        ci_lb=ci_lb, ci_ub=ci_ub, ids=ids, failed=failed,
        flagged=flagged,
        considered=len(outcomes), decisions=spec.names)


# --------------------------------------------------------------------------- #
# the report
# --------------------------------------------------------------------------- #
#: The caveats that travel with any multiverse table. Not configurable: a report
#: without them is the thing this module exists to prevent.
NOTES = (
    "The specifications are not independent experiments: they share the same "
    "studies. The share whose interval excludes zero is a frequency over "
    "analyses, not a probability that the effect is real.",
    "The variance shares are a one-way decomposition, so an interaction between "
    "two decisions is left unattributed and the shares sum to less than one.",
    "A specification that failed or was excluded is listed with its status. "
    "Reading only the rows that produced a number is the largest silent bias in "
    "a published multiverse.",
)


@dataclass(frozen=True)
class SpecificationReport:
    """Every specification of a grid, with its status and, where it has one, its number.

    The table is the point: ``mv.curve()`` is already the subset that worked, so
    a reader who pastes it into a paper has removed the failures by copying. This
    object renders all of them, in grid order, with the status beside the number.
    """

    rows: list[dict[str, Any]]
    counts: dict[str, int]
    title: str = ""
    provenance: str = ""
    notes: tuple[str, ...] = NOTES

    @property
    def markdown(self) -> str:
        """The table, the counts and the caveats, as Markdown."""
        out: list[str] = []
        if self.title:
            out += [f"# {self.title}", ""]
        if self.provenance:
            out += [self.provenance, ""]
        out += ["| specification | choices | estimate | 95% interval | status |",
                "|---|---|---|---|---|"]
        out += ["| " + " | ".join(self._cells(row)) + " |" for row in self.rows]
        out += ["",
                f"{self.counts['run']} of {self.counts['considered']} "
                f"specifications ran: {self.counts['usable']} usable, "
                f"{self.counts['failed']} failed, {self.counts['flagged']} "
                f"flagged ({self.excluded} excluded by their own diagnostics).",
                "", "**What this table does not say**", ""]
        out += [f"- {note}" for note in self.notes]
        return "\n".join(out) + "\n"

    @property
    def excluded(self) -> int:
        return sum(1 for row in self.rows if row["status"] == "excluded")

    def _cells(self, row: dict[str, Any]) -> list[str]:
        choices = ", ".join(f"{k}={v}" for k, v in row["choices"].items())
        if row["status"] == "failed":
            return [row["id"][:12], choices, "—", "—", f"failed: {row['error']}"]
        if row["status"] == "excluded":
            codes = ", ".join(row["codes"])
            return [row["id"][:12], choices, "—", "—", f"excluded: {codes}"]
        status = "ok" if not row["codes"] else "ok (" + ", ".join(row["codes"]) + ")"
        return [row["id"][:12], choices, f"{row['estimate']:+.4f}",
                f"{row['ci_lb']:+.4f} to {row['ci_ub']:+.4f}", status]

    def to_dict(self) -> dict[str, Any]:
        return {"title": self.title, "counts": dict(self.counts),
                "provenance": self.provenance,
                "rows": [dict(r) for r in self.rows], "notes": list(self.notes)}

    def save(self, path: str | Path) -> Path:
        """Write the Markdown to ``path`` and return the path."""
        target = Path(path)
        target.write_text(self.markdown, encoding="utf-8")
        return target

    def __str__(self) -> str:
        return self.markdown

    def __len__(self) -> int:
        return len(self.rows)


def specification_report(mv: MultiverseResult, *, title: str = "",
                         provenance: str = "") -> SpecificationReport:
    """Render every specification of ``mv`` as one table, failures included.

    Args:
        mv: the result of :func:`multiverse`.
        title: an optional heading; omitted when empty.
        provenance: an optional line naming the data and the date, so a pasted
            table can be traced to a run.

    Returns:
        :class:`SpecificationReport`; its ``markdown`` property is the table.

    The rows are in grid order, not ordered by estimate: the table audits the
    grid, and sorting it by the answer would re-introduce the selection it exists
    to expose. ``mv.curve()`` is the sorted view.
    """
    usable = {mid: i for i, mid in enumerate(mv.ids)}
    failed = {row["id"]: row for row in mv.failed}
    flagged = {row["id"]: row for row in mv.flagged}

    rows: list[dict[str, Any]] = []
    for outcome in mv.outcomes:
        mid = outcome.id
        codes = list(flagged[mid]["codes"]) if mid in flagged else []
        base = {"id": mid, "choices": dict(outcome.choices), "codes": codes}
        if mid in failed:
            rows.append({**base, "status": "failed",
                         "error": failed[mid]["error"]})
        elif mid in flagged and flagged[mid]["excluded"]:
            rows.append({**base, "status": "excluded"})
        else:
            index = usable[mid]
            rows.append({**base, "status": "ok",
                         "estimate": mv.estimates[index],
                         "ci_lb": mv.ci_lb[index], "ci_ub": mv.ci_ub[index]})

    return SpecificationReport(rows=rows, counts=mv.counts(), title=title,
                               provenance=provenance)
