"""Declared missing-data handling, instead of an implicit ``dropna``.

A model that silently drops rows has made an assumption on the analyst's
behalf: that the missingness is unrelated to the quantities being estimated.
That assumption is sometimes right and is never free, so this module makes the
handling a *declaration* attached to the result rather than a side effect of
`na.omit`.

Three strategies exist and they are not interchangeable:

``complete_case``
    Rows with any missing value are removed. Consistent under MCAR; biased under
    MAR even when the complete cases are a large majority.
``available_case``
    Each quantity uses whatever rows it can. Uses more data than complete-case
    and does **not** give a coherent covariance across quantities, because
    different rows contribute to different parts.
``likelihood``
    The rows are kept and the likelihood integrates over the missing values
    (full-information maximum likelihood). Valid under MAR given a correct
    model; this is the only strategy here that is.

Multiple imputation is **not** one of them. Under MAR an imputation procedure is
only as credible as its imputation model, and verifying it needs a coverage
study under a *known* missingness mechanism; adding it as a `method="mi"`
argument to six models at once would be the opposite of what this module is for.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from typing import Any

import numpy as np

from .missing import complete_cases, is_missing

__all__ = [
    "STRATEGIES",
    "MissingPolicy",
    "complete_case_policy",
    "policy_from_result",
]

#: The declared strategies, each with the assumption it needs in one line.
STRATEGIES: dict[str, str] = {
    "complete_case":
        "rows with any missing value are removed; consistent only when the "
        "missingness is unrelated to the quantities being estimated (MCAR)",
    "available_case":
        "each quantity uses the rows it has; more data than complete-case, but "
        "the quantities no longer share a coherent covariance",
    "likelihood":
        "rows are kept and the likelihood integrates over the missing values "
        "(full-information ML); valid under MAR given a correct model",
}


@dataclass(frozen=True)
class MissingPolicy:
    """What a fit did about missing values, and what that assumes.

    Attributes:
        strategy: one of :data:`STRATEGIES`.
        n_total: rows offered to the model.
        n_used: rows that entered it.
        dropped: 0-based indices of the rows that did not, in input order.
        mechanism: what the analyst is willing to assume -- ``"MCAR"``,
            ``"MAR"``, ``"MNAR"`` or ``"unknown"``. The default is ``"unknown"``
            because that is the truthful label when nothing was said, and a
            policy that claims MCAR without evidence is worse than one that
            admits it does not know.
        note: anything else worth carrying, e.g. which columns were affected.
    """

    strategy: str
    n_total: int
    n_used: int
    dropped: tuple[int, ...] = ()
    mechanism: str = "unknown"
    note: str = ""

    def __post_init__(self) -> None:
        if self.strategy not in STRATEGIES:
            raise ValueError(
                f"unknown strategy {self.strategy!r}; expected one of "
                f"{tuple(STRATEGIES)}")
        if self.mechanism not in ("MCAR", "MAR", "MNAR", "unknown"):
            raise ValueError(
                f"unknown mechanism {self.mechanism!r}; expected 'MCAR', "
                "'MAR', 'MNAR' or 'unknown'")
        if not 0 <= self.n_used <= self.n_total:
            raise ValueError(
                f"n_used ({self.n_used}) must be between 0 and n_total "
                f"({self.n_total})")

    @property
    def n_dropped(self) -> int:
        return self.n_total - self.n_used

    @property
    def is_complete(self) -> bool:
        return self.n_dropped == 0

    @property
    def assumption(self) -> str:
        """The one-line assumption this strategy needs."""
        return STRATEGIES[self.strategy]

    def to_dict(self) -> dict[str, Any]:
        return {
            "strategy": self.strategy,
            "assumption": self.assumption,
            "mechanism": self.mechanism,
            "n_total": int(self.n_total),
            "n_used": int(self.n_used),
            "n_dropped": int(self.n_dropped),
            "dropped": [int(i) for i in self.dropped],
            "note": self.note,
        }

    def __str__(self) -> str:
        head = f"{self.strategy}: {self.n_used} of {self.n_total} rows used"
        if self.is_complete:
            return head + "; nothing dropped"
        shown = ", ".join(str(i + 1) for i in self.dropped[:10])
        more = f", ... (+{len(self.dropped) - 10} more)" if len(self.dropped) > 10 else ""
        return (f"{head}; dropped {self.n_dropped} (row"
                f"{'s' if self.n_dropped > 1 else ''} {shown}{more}); "
                f"assumes {self.mechanism}: {self.assumption}")


def complete_case_policy(
    columns: Mapping[str, Any],
    *,
    mechanism: str = "unknown",
    warn: bool = False,
    context: str = "rows",
) -> MissingPolicy:
    """Declare the complete-case handling of ``columns`` without re-deriving it.

    Args:
        columns: ``{name: sequence}`` as passed to :func:`missing.complete_cases`.
        mechanism: the missingness assumption to record.
        warn: forward to :func:`missing.complete_cases`; off by default because
            the policy object is the declaration and a warning would duplicate
            it.
        context: noun used in that warning.

    Returns:
        The :class:`MissingPolicy` describing what complete-case handling did.
    """
    keep, dropped = complete_cases(columns, context=context, warn=warn)
    present = {k: v for k, v in columns.items() if v is not None}
    n_total = len(next(iter(present.values()))) if present else 0
    affected = [name for name, values in present.items()
                if bool(np.any(is_missing(values)))]
    note = (f"affected columns: {', '.join(affected)}" if affected else "")
    return MissingPolicy(
        strategy="complete_case",
        n_total=int(n_total),
        n_used=int(np.sum(keep)) if keep.size else 0,
        dropped=tuple(int(i) for i in dropped),
        mechanism=mechanism,
        note=note,
    )


def policy_from_result(result: Any) -> MissingPolicy | None:
    """Read the handling a fitted result already did, when it recorded one.

    Returns ``None`` rather than guessing when the result does not carry the
    counts: a made-up policy is worse than an absent one, because it would be
    printed next to a fit as if it had been observed.
    """
    k_all = getattr(result, "k_all", None)
    k = getattr(result, "k", None)
    if k_all is None or k is None:
        return None
    try:
        n_total, n_used = int(k_all), int(k)
    except (TypeError, ValueError):  # pragma: no cover - defensive
        return None
    if n_used > n_total or n_total <= 0:
        return None
    not_na = getattr(result, "not_na", None)
    dropped: Iterable[int] = ()
    if not_na is not None:
        flags = np.asarray(list(not_na), dtype=bool)
        if flags.size == n_total:
            dropped = tuple(int(i) for i in np.flatnonzero(~flags))
    return MissingPolicy(
        strategy="complete_case",
        n_total=n_total,
        n_used=n_used,
        dropped=tuple(dropped),
        mechanism="unknown",
    )
