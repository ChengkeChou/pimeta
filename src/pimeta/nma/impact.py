"""````netmeta::netimpact`` — study-level influence on the network estimate``.

Re-fits the network with each study removed and reports how far the
pooled effect moves, so a single influential study is visible without
a manual leave-one-out loop.

Part of the network meta-analysis implementation; :mod:`pimeta.netmeta`
re-exports these names and remains the public entry point.
"""

from __future__ import annotations

import sys
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:                       # pragma: no cover - typing only
    pass

__all__ = ['NetimpactResult', 'netimpact']


# --------------------------------------------------------------------------- #
# netmeta::netimpact
# --------------------------------------------------------------------------- #

class NetimpactResult:
    """Result of :func:`netimpact` (``netmeta::netimpact``)."""

    __slots__ = ("impact_common", "impact_random", "impact_fixed",
                 "comparisons", "studies", "ignored_comparisons",
                 "seTE_ignore")

    def __init__(self, impact_common, impact_random, comparisons, studies,
                 ignored_comparisons, seTE_ignore):
        self.impact_common = impact_common
        self.impact_random = impact_random
        self.impact_fixed = impact_common
        self.comparisons = comparisons
        self.studies = studies
        self.ignored_comparisons = ignored_comparisons
        self.seTE_ignore = seTE_ignore

    def __repr__(self):
        return (f"NetimpactResult(studies={len(self.studies)}, "
                f"comparisons={len(self.comparisons)})")

    def as_dict(self):
        return {
            "impact.common": self.impact_common,
            "impact.random": self.impact_random,
            "impact.fixed": self.impact_common,
            "ignored.comparisons": self.ignored_comparisons,
            "seTE.ignore": self.seTE_ignore,
        }


def _comparison_list(trts):
    """Row-major ``i < j`` comparison labels — netmeta's ``x$comparisons``."""
    out = []
    for i in range(len(trts) - 1):
        for j in range(i + 1, len(trts)):
            out.append(f"{trts[i]}:{trts[j]}")
    return out


def netimpact(result, seTE_ignore=None):
    """``netmeta::netimpact`` — leave-one-study-out impact on all comparisons.

    Each study is down-weighted (its standard errors are replaced by
    ``seTE_ignore``) and the network is re-estimated with the original ``tau``
    as ``tau_preset``; the reported impact is ``1 - (seTE / seTE_removed)^2``.

    Returns:
        :class:`NetimpactResult` — ``impact_common``/``impact_random`` map
        study labels to ``{comparison: impact}`` dictionaries.
    """
    from ..netmeta import NetmetaResult, netmeta  # local import: avoid a cycle

    if not isinstance(result, NetmetaResult):
        raise TypeError("argument 'result' must be a NetmetaResult")

    se = np.asarray(result.seTE_raw, dtype=float)
    if seTE_ignore is None:
        seTE_ignore = 1000.0 * float(np.nanmax(se))
    seTE_ignore = float(seTE_ignore)

    te = list(result.TE_raw)
    t1 = list(result.treat1_raw)
    t2 = list(result.treat2_raw)
    stud = list(result.studlab_raw)
    trts = list(result.trts)
    studies = list(dict.fromkeys(stud))

    lower = [(i, j) for i in range(len(trts) - 1) for j in range(i + 1, len(trts))]
    comps = [f"{trts[i]}:{trts[j]}" for i, j in lower]

    def _flat(mat):
        mat = np.asarray(mat, dtype=float)
        return {(i, j): float(mat[i, j]) for i, j in lower}

    base_c = _flat(result.seTE_common)
    base_r = _flat(result.seTE_random)

    tiny = sys.float_info.epsilon ** 0.5  # .Machine$double.eps^0.5
    impact_c = {s: {} for s in studies}
    impact_r = {s: {} for s in studies}

    raw_comp = [f"{a}:{b}" for a, b in zip(t1, t2)]
    ignored = {s: [c for c, sv in zip(raw_comp, stud) if sv == s]
               for s in studies}

    for s in studies:
        se_i = [seTE_ignore if sv == s else se[k] for k, sv in enumerate(stud)]
        net_i = netmeta(te, se_i, t1, t2, stud, tau_preset=result.tau)
        new_c = _flat(net_i.seTE_common)
        new_r = _flat(net_i.seTE_random)
        for idx, key in enumerate(lower):
            name = comps[idx]
            ic = 1.0 - (base_c[key] / new_c[key]) ** 2
            ir = 1.0 - (base_r[key] / new_r[key]) ** 2
            impact_c[s][name] = 0.0 if abs(ic) < tiny else float(ic)
            impact_r[s][name] = 0.0 if abs(ir) < tiny else float(ir)

    return NetimpactResult(
        impact_common=impact_c,
        impact_random=impact_r,
        comparisons=comps,
        studies=studies,
        ignored_comparisons=ignored,
        seTE_ignore=seTE_ignore,
    )
