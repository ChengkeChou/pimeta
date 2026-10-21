"""League table (``netmeta::netleague``).

Lower triangle: network estimates (column vs row). Upper triangle: direct
estimates (row vs column). Split out of :mod:`pimeta.netmeta`; the public names
are re-exported there.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:  # pragma: no cover - typing only
    from ..netmeta import NetmetaResult


def _format_right(mat: np.ndarray, digits: int) -> dict:
    """Replicate R ``format(x, justify = "right")`` on a numeric matrix.

    Each non-missing value is rendered with a fixed number of decimals
    (``f"{v:.{digits}f}"``), then right-justified to the width of the widest
    entry (including R's ``"NA"``, which occupies 3 columns) so that positive
    numbers are padded with leading spaces to line up the minus signs.
    Missing values are omitted from the result (R ``formatCI`` maps them to
    ``""``).
    """
    n = mat.shape[0]
    cells = {}
    width = 3  # "NA"
    for i in range(n):
        for j in range(n):
            v = mat[i, j]
            if np.isnan(v):
                continue
            s = f"{v:.{digits}f}"
            cells[(i, j)] = s
            width = max(width, len(s))
    return {k: s.rjust(width) for k, s in cells.items()}


def _league_cells(
    te_nma: np.ndarray,
    lo_nma: np.ndarray,
    up_nma: np.ndarray,
    te_dir: np.ndarray,
    lo_dir: np.ndarray,
    up_dir: np.ndarray,
    trts: list[str],
    digits: int,
    text_na: str,
    ci: bool,
) -> list[list[str]]:
    """Build a league-table matrix (list of string rows).

    Lower triangle (``r > c``) holds the network estimates (column vs row,
    i.e. ``theta[c] - theta[r]``); upper triangle (``r < c``) holds the direct
    estimates (row vs column, ``theta_dir[r] - theta_dir[c]``). The diagonal
    holds the treatment names. Matches ``netmeta::netleague`` orientation.
    """
    n = len(trts)

    def _fn(mat: np.ndarray) -> list[list[str]]:
        return [
            [
                text_na if np.isnan(mat[i, j]) else f"{mat[i, j]:.{digits}f}"
                for j in range(n)
            ]
            for i in range(n)
        ]

    te_n = _fn(te_nma)
    te_d = _fn(te_dir)

    lo_n = _format_right(lo_nma, digits) if ci else {}
    up_n = _format_right(up_nma, digits) if ci else {}
    lo_d = _format_right(lo_dir, digits) if ci else {}
    up_d = _format_right(up_dir, digits) if ci else {}

    cells = [[text_na] * n for _ in range(n)]
    for r in range(n):
        cells[r][r] = trts[r]
        for c in range(n):
            if r == c:
                continue
            if r < c:
                # Upper triangle: direct estimate (row vs column).
                if np.isnan(te_dir[r, c]):
                    cells[r][c] = text_na
                elif ci:
                    cells[r][c] = (
                        f"{te_d[r][c]} [{lo_d[(r, c)]}; {up_d[(r, c)]}]"
                    )
                else:
                    cells[r][c] = te_d[r][c]
            else:
                # Lower triangle: network estimate (column vs row).
                if np.isnan(te_nma[c, r]):
                    cells[r][c] = text_na
                elif ci:
                    cells[r][c] = (
                        f"{te_n[c][r]} [{lo_n[(c, r)]}; {up_n[(c, r)]}]"
                    )
                else:
                    cells[r][c] = te_n[c][r]
    return cells


@dataclass
class NetleagueResult:
    """League table (network estimates in the lower triangle, direct estimates
    in the upper triangle), replicating ``netmeta::netleague``.

    ``common`` and ``random`` are matrices (lists of string rows) of formatted
    cells: ``"TE [lower; upper]"``, ``"."`` for missing, treatment names on the
    diagonal. ``trts`` is the row/column order. ``details`` is the explanatory
    caption text used by R's print method.
    """

    trts: list[str]
    common: list[list[str]] = field(default_factory=list)
    random: list[list[str]] = field(default_factory=list)
    details: str = ""

    def to_dict(self) -> dict:
        return {
            "trts": self.trts,
            "common": self.common,
            "random": self.random,
            "details": self.details,
        }

    def to_string(self) -> str:
        """Render the league table(s) as fixed-width text, like R."""
        out = []
        if self.common:
            out.append("League table (common effects model):")
            out.append(_league_to_text(self.trts, self.common))
        if self.random:
            out.append("League table (random effects model):")
            out.append(_league_to_text(self.trts, self.random))
        if self.details:
            out.append(self.details)
        return "\n".join(out)


def _league_to_text(trts: list[str], cells: list[list[str]]) -> str:
    header = [""] + list(trts)
    rows = [[trts[i]] + list(cells[i]) for i in range(len(trts))]
    table = [header] + rows
    widths = [
        max(len(table[r][c]) for r in range(len(table)))
        for c in range(len(table[0]))
    ]
    lines = []
    for r in range(len(table)):
        lines.append(
            " ".join(table[r][c].rjust(widths[c]) for c in range(len(table[0])))
        )
    return "\n".join(lines)


def netleague(
    x: NetmetaResult,
    common: bool = True,
    random: bool = True,
    seq: list[str] | None = None,
    ci: bool = True,
    backtransf: bool = False,
    digits: int = 4,
    text_na: str = ".",
    details: bool = True,
) -> NetleagueResult:
    """League table for a network meta-analysis.

    Replicates ``netmeta::netleague`` for a single ``netmeta`` object: the
    lower triangle gives the network estimates (column vs row) and the upper
    triangle the direct estimates (row vs column).

    Args:
        x: a :class:`NetmetaResult` from :func:`netmeta`.
        common: include the common-effects table.
        random: include the random-effects table.
        seq: optional row/column treatment order (default: ``x.trts``).
        ci: include confidence intervals.
        backtransf: back-transform to the original scale (``exp`` for ratio
            measures; ``False`` for ``MD``).
        digits: number of decimals.
        text_na: string used for missing cells.
        details: include the explanatory caption.
    """
    if seq is None:
        seq = list(x.trts)
    else:
        seq = list(seq)
    idx = [x.trts.index(t) for t in seq]
    len(seq)

    def _pick(mat: np.ndarray) -> np.ndarray:
        return mat[np.ix_(idx, idx)]

    details_txt = (
        "\nLower triangle: results from network meta-analysis (column vs row)"
        "\nUpper triangle: results from direct comparisons (row vs column)\n"
        if details
        else ""
    )

    res = NetleagueResult(trts=seq, details=details_txt)

    if common:
        te_nma = _pick(x.TE_common)
        lo_nma = _pick(x.lower_common)
        up_nma = _pick(x.upper_common)
        te_dir = _pick(x.TE_direct_common)
        lo_dir = _pick(x.lower_direct_common)
        up_dir = _pick(x.upper_direct_common)
        if backtransf:
            te_nma, lo_nma, up_nma = _backtransf3(te_nma, lo_nma, up_nma, x.sm)
            te_dir, lo_dir, up_dir = _backtransf3(te_dir, lo_dir, up_dir, x.sm)
        res.common = _league_cells(
            te_nma, lo_nma, up_nma, te_dir, lo_dir, up_dir,
            seq, digits, text_na, ci,
        )

    if random:
        te_nma = _pick(x.TE_random)
        lo_nma = _pick(x.lower_random)
        up_nma = _pick(x.upper_random)
        te_dir = _pick(x.TE_direct_random)
        lo_dir = _pick(x.lower_direct_random)
        up_dir = _pick(x.upper_direct_random)
        if backtransf:
            te_nma, lo_nma, up_nma = _backtransf3(te_nma, lo_nma, up_nma, x.sm)
            te_dir, lo_dir, up_dir = _backtransf3(te_dir, lo_dir, up_dir, x.sm)
        res.random = _league_cells(
            te_nma, lo_nma, up_nma, te_dir, lo_dir, up_dir,
            seq, digits, text_na, ci,
        )

    return res


def _backtransf3(
    te: np.ndarray, lo: np.ndarray, up: np.ndarray, sm: str
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Back-transform point estimate and CI to the original scale."""
    if sm == "VE":
        te_t = -np.exp(-te)
        lo_t = -np.exp(-up)
        up_t = -np.exp(-lo)
        return te_t, lo_t, up_t
    return np.exp(te), np.exp(lo), np.exp(up)
