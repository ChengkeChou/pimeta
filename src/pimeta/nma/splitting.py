"""Split direct and indirect evidence (``netmeta::netsplit``).

Split out of :mod:`pimeta.netmeta`; the public names are re-exported there, so
``pimeta.netmeta.netsplit`` and ``pimeta.nma.splitting.netsplit`` are the same
object.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from scipy import stats

if TYPE_CHECKING:  # pragma: no cover - typing only
    from ..netmeta import NetmetaResult


@dataclass
class NetsplitResult:
    """Split of direct and indirect evidence per comparison.

    ``comparison`` lists every treatment pair in the upper-triangle order
    (``trts[i] : trts[j]`` with ``i < j``). ``k`` is the number of studies
    providing direct evidence (``A.matrix``); ``prop_common`` / ``prop_random``
    are the direct-evidence proportions. The ``common`` / ``direct_common`` /
    ``indirect_common`` / ``compare_common`` dicts (and their ``*_random``
    counterparts) hold aligned lists keyed by ``"TE"``, ``"seTE"``, ``"lower"``,
    ``"upper"``, ``"statistic"``, ``"p"`` (``compare_*`` also has ``"z"``).
    Missing values (R ``NA``) are ``None``.
    """

    comparison: list[str]
    k: list[int]
    prop_common: list[float]
    prop_random: list[float]
    method: str = "Back-calculation"

    common: dict = field(default_factory=dict)
    direct_common: dict = field(default_factory=dict)
    indirect_common: dict = field(default_factory=dict)
    compare_common: dict = field(default_factory=dict)

    random: dict = field(default_factory=dict)
    direct_random: dict = field(default_factory=dict)
    indirect_random: dict = field(default_factory=dict)
    compare_random: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "comparison": self.comparison,
            "k": self.k,
            "prop_common": self.prop_common,
            "prop_random": self.prop_random,
            "method": self.method,
            "common": self.common,
            "direct_common": self.direct_common,
            "indirect_common": self.indirect_common,
            "compare_common": self.compare_common,
            "random": self.random,
            "direct_random": self.direct_random,
            "indirect_random": self.indirect_random,
            "compare_random": self.compare_random,
        }


def _split_pooled(
    x: NetmetaResult, pooled: str, tol_direct: float, flip: list[bool]
) -> tuple:
    """Build nma / direct / indirect / compare dicts for one model.

    ``flip`` is aligned with the comparison list; a ``True`` entry means the
    comparison was expressed as ``trts[j] : trts[i]`` (reference placed on the
    right), so the effect, confidence limits and statistic are negated.
    """
    if pooled == "common":
        TE, se = x.TE_common, x.seTE_common
        lo, up = x.lower_common, x.upper_common
        st, pv = x.statistic_common, x.pval_common
        TE_d, se_d = x.TE_direct_common, x.seTE_direct_common
        TE_i, se_i = x.TE_indirect_common, x.seTE_indirect_common
        prop = x.prop_direct_common
    else:
        TE, se = x.TE_random, x.seTE_random
        lo, up = x.lower_random, x.upper_random
        st, pv = x.statistic_random, x.pval_random
        TE_d, se_d = x.TE_direct_random, x.seTE_direct_random
        TE_i, se_i = x.TE_indirect_random, x.seTE_indirect_random
        prop = x.prop_direct_random

    n = x.n
    a_mat = x.A_matrix
    z = stats.norm.ppf(1 - (1 - x.level) / 2)

    nma: dict[str, list] = {k: [] for k in ("TE", "seTE", "lower", "upper", "statistic", "p")}
    direct: dict[str, list] = {k: [] for k in ("TE", "seTE", "lower", "upper", "statistic", "p")}
    indirect: dict[str, list] = {k: [] for k in ("TE", "seTE", "lower", "upper", "statistic", "p")}
    compare: dict[str, list] = {k: [] for k in ("TE", "seTE", "lower", "upper", "statistic", "p", "z")}

    def _ci(te, se_):
        if te is None or se_ is None:
            return None, None, None, None
        lower = te - z * se_
        upper = te + z * se_
        stat = te / se_ if se_ != 0 else float("nan")
        p = 2.0 * stats.norm.sf(abs(stat))
        return lower, upper, stat, p

    kk = 0
    for i in range(n - 1):
        for j in range(i + 1, n):
            sgn = -1.0 if flip[kk] else 1.0
            p_ij = float(prop[i, j])
            # Network estimate.
            nma["TE"].append(sgn * float(TE[i, j]))
            nma["seTE"].append(float(se[i, j]))
            nma["lower"].append(sgn * float(lo[i, j]))
            nma["upper"].append(sgn * float(up[i, j]))
            if sgn < 0:
                nma["lower"][-1], nma["upper"][-1] = (
                    nma["upper"][-1], nma["lower"][-1],
                )
            nma["statistic"].append(sgn * float(st[i, j]))
            nma["p"].append(float(pv[i, j]))

            # Direct estimate (NA when there is (almost) no direct evidence).
            if abs(p_ij) < tol_direct:
                d_te = d_se = None
            else:
                d_te = sgn * float(TE_d[i, j])
                d_se = float(se_d[i, j])
            d_lo, d_up, d_st, d_pv = _ci(d_te, d_se)
            direct["TE"].append(d_te)
            direct["seTE"].append(d_se)
            direct["lower"].append(d_lo)
            direct["upper"].append(d_up)
            direct["statistic"].append(d_st)
            direct["p"].append(d_pv)

            # Indirect estimate (NA for pure direct evidence; equals the
            # network estimate when there is no direct evidence at all).
            if abs(p_ij - 1.0) < tol_direct:
                i_te = i_se = None
            else:
                i_te = sgn * float(TE_i[i, j])
                i_se = float(se_i[i, j])
                if a_mat[i, j] == 0:
                    i_te = sgn * float(TE[i, j])
                    i_se = float(se[i, j])
            i_lo, i_up, i_st, i_pv = _ci(i_te, i_se)
            indirect["TE"].append(i_te)
            indirect["seTE"].append(i_se)
            indirect["lower"].append(i_lo)
            indirect["upper"].append(i_up)
            indirect["statistic"].append(i_st)
            indirect["p"].append(i_pv)

            # Test of direct vs indirect (difference).
            if d_te is None or i_te is None or d_se is None or i_se is None:
                c_te = c_se = None
            else:
                c_te = d_te - i_te
                c_se = math.sqrt(d_se ** 2 + i_se ** 2)
            c_lo, c_up, c_st, c_pv = _ci(c_te, c_se)
            compare["TE"].append(c_te)
            compare["seTE"].append(c_se)
            compare["lower"].append(c_lo)
            compare["upper"].append(c_up)
            compare["statistic"].append(c_st)
            compare["p"].append(c_pv)
            compare["z"].append(c_st)

            kk += 1

    return nma, direct, indirect, compare


def netsplit(
    x: NetmetaResult,
    method: str = "Back-calculation",
    common: bool = True,
    random: bool = True,
    tol_direct: float = 5e-4,
) -> NetsplitResult:
    """Split direct and indirect evidence in a network meta-analysis.

    Replicates ``netmeta::netsplit`` with ``method = "Back-calculation"``
    (the default for non-binary ``netmeta`` objects). The indirect estimate is
    back-calculated from the network and direct estimates using the direct
    evidence proportion, already stored in ``x`` by :func:`netmeta`.

    Args:
        x: a :class:`NetmetaResult` from :func:`netmeta`.
        method: splitting method (only ``"Back-calculation"``).
        common: include common-effects results.
        random: include random-effects results.
        tol_direct: maximum deviation of the direct evidence proportion from 0
            or 1 to classify a comparison as indirect-only or direct-only.
    """
    if method != "Back-calculation":
        raise ValueError(
            f"method={method!r} not supported; only 'Back-calculation' "
            "is implemented"
        )

    n = x.n
    comparison = []
    flip = []
    k = []
    prop_c = []
    prop_r = []
    a_mat = x.A_matrix
    ref = x.reference_group
    base_ref = x.baseline_reference
    for i in range(n - 1):
        for j in range(i + 1, n):
            t1 = x.trts[i]
            t2 = x.trts[j]
            # netmeta::comptrts: with a reference group as baseline, comparisons
            # are expressed as "other treatment : reference" (reference on the
            # right), so any comparison with the reference first is reversed.
            fl = bool(base_ref and ref != "" and t1 == ref)
            if fl:
                comparison.append(f"{t2}:{t1}")
            else:
                comparison.append(f"{t1}:{t2}")
            flip.append(fl)
            k.append(int(a_mat[i, j]))
            prop_c.append(float(x.prop_direct_common[i, j]))
            prop_r.append(float(x.prop_direct_random[i, j]))

    res = NetsplitResult(
        comparison=comparison,
        k=k,
        prop_common=prop_c,
        prop_random=prop_r,
        method=method,
    )

    if common:
        nma, direct, indirect, compare = _split_pooled(
            x, "common", tol_direct, flip
        )
        res.common = nma
        res.direct_common = direct
        res.indirect_common = indirect
        res.compare_common = compare
    if random:
        nma, direct, indirect, compare = _split_pooled(
            x, "random", tol_direct, flip
        )
        res.random = nma
        res.direct_random = direct
        res.indirect_random = indirect
        res.compare_random = compare

    return res
