"""Pairwise (meta-analysis) view of a network (``netmeta::netpairwise``).

Re-expresses a network as a ``metagen``-style pairwise meta-analysis over the
direct comparisons, plus a per-pair direct-evidence table.  Split out of
:mod:`pimeta.netmeta`; the public names are re-exported there.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np
from scipy import stats

from .core import _cochran_q, _dl_tau2, _isquared, _iv_pool

if TYPE_CHECKING:  # pragma: no cover - typing only
    pass


@dataclass
class NetpairwiseResult:
    """Pairwise meta-analysis of the direct comparisons of a network.

    ``netmeta::netpairwise`` re-expresses a network meta-analysis as a plain
    pairwise meta-analysis (a ``metagen``-style object) over the direct
    comparisons, plus a per-pair direct-evidence table. Every comparison is
    oriented as "non-reference treatment vs reference treatment" when a
    reference group is set, and alphabetically (treat1 < treat2) otherwise.
    """

    trts: list[str]
    k: int                          # number of pairwise comparisons (edges)
    reference_group: str
    sm: str
    level: float
    method_tau: str

    # Overall pooled estimates (metagen on all comparisons, DL tau^2).
    TE_common: float = float("nan")
    seTE_common: float = float("nan")
    lower_common: float = float("nan")
    upper_common: float = float("nan")
    statistic_common: float = float("nan")
    pval_common: float = float("nan")
    TE_random: float = float("nan")
    seTE_random: float = float("nan")
    lower_random: float = float("nan")
    upper_random: float = float("nan")
    statistic_random: float = float("nan")
    pval_random: float = float("nan")
    tau2: float = float("nan")
    tau: float = float("nan")
    I2: float = float("nan")
    lower_I2: float = float("nan")
    upper_I2: float = float("nan")
    Q: float = float("nan")
    df_Q: float = float("nan")
    pval_Q: float = float("nan")

    # Oriented comparison-level data (sorted by treatment pair).
    TE: list = field(default_factory=list)
    seTE: list = field(default_factory=list)
    treat1: list = field(default_factory=list)
    treat2: list = field(default_factory=list)
    studlab: list = field(default_factory=list)

    # Per-pair direct evidence (one dict per pair with direct evidence).
    pairs: list = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "trts": self.trts,
            "k": self.k,
            "reference_group": self.reference_group,
            "sm": self.sm,
            "level": self.level,
            "method_tau": self.method_tau,
            "TE_common": self.TE_common,
            "seTE_common": self.seTE_common,
            "lower_common": self.lower_common,
            "upper_common": self.upper_common,
            "statistic_common": self.statistic_common,
            "pval_common": self.pval_common,
            "TE_random": self.TE_random,
            "seTE_random": self.seTE_random,
            "lower_random": self.lower_random,
            "upper_random": self.upper_random,
            "statistic_random": self.statistic_random,
            "pval_random": self.pval_random,
            "tau2": self.tau2,
            "tau": self.tau,
            "I2": self.I2,
            "lower_I2": self.lower_I2,
            "upper_I2": self.upper_I2,
            "Q": self.Q,
            "df_Q": self.df_Q,
            "pval_Q": self.pval_Q,
            "TE": self.TE,
            "seTE": self.seTE,
            "treat1": self.treat1,
            "treat2": self.treat2,
            "studlab": self.studlab,
            "pairs": self.pairs,
        }


def _orient_comparisons(te, se, treat1, treat2, studlab, reference_group, trts):
    """Orient and sort pairwise comparisons (netmeta::netpairwise direction).

    Each comparison is oriented as "non-reference vs reference" when one of the
    two treatments is ``reference_group``; otherwise treatments are ordered
    alphabetically (treat1 < treat2), flipping the sign of the effect when the
    order is swapped. The result is sorted by (treat1, treat2) position.
    """
    idx = {t: i for i, t in enumerate(trts)}
    out = []
    for k in range(len(te)):
        a, b = treat1[k], treat2[k]
        y = float(te[k])
        if reference_group and b == reference_group:
            pass
        elif reference_group and a == reference_group:
            a, b = b, a
            y = -y
        elif a > b:
            a, b = b, a
            y = -y
        out.append([a, b, y, float(se[k]), studlab[k]])
    out.sort(key=lambda r: (idx[r[0]], idx[r[1]]))
    return out


def netpairwise(
    x,
    separate_trts: bool = True,
    reference_group: str | None = None,
    sm: str = "",
    level: float = 0.95,
    method_tau: str = "DL",
    common: bool = True,
    random: bool = True,
) -> NetpairwiseResult:
    """Pairwise (meta-analysis) view of a network meta-analysis.

    Transcribes ``netmeta::netpairwise``. ``x`` is a :class:`NetmetaResult`
    (from :func:`netmeta`). The result is a metagen-style object whose overall
    pooled estimates come from a DerSimonian-Laird random-effects pairwise
    meta-analysis of all direct comparisons, plus a ``pairs`` table giving the
    per-comparison direct evidence (common/random effect, Q, I^2, tau^2).
    """
    from ..netmeta import NetmetaResult  # local import: cycle

    if isinstance(x, NetmetaResult):
        te = list(x.TE_raw)
        se = list(x.seTE_raw)
        t1 = list(x.treat1_raw)
        t2 = list(x.treat2_raw)
        stud = list(x.studlab_raw)
        trts = list(x.trts)
        reference_group = reference_group if reference_group else x.reference_group
        if not sm:
            sm = x.sm
        if level is None:
            level = x.level
    else:
        raise TypeError("netpairwise expects a NetmetaResult")

    if method_tau.upper() != "DL":
        raise ValueError("netpairwise only supports method_tau='DL'")

    oriented = _orient_comparisons(te, se, t1, t2, stud, reference_group, trts)
    te_arr = np.array([r[2] for r in oriented], dtype=float)
    se_arr = np.array([r[3] for r in oriented], dtype=float)
    k = len(te_arr)

    z = stats.norm.ppf(1 - (1 - level) / 2)

    result = NetpairwiseResult(
        trts=trts,
        k=k,
        reference_group=reference_group,
        sm=sm,
        level=level,
        method_tau=method_tau,
        TE=[r[2] for r in oriented],
        seTE=[r[3] for r in oriented],
        treat1=[r[0] for r in oriented],
        treat2=[r[1] for r in oriented],
        studlab=[r[4] for r in oriented],
    )

    q = _cochran_q(te_arr, se_arr)
    df_q = float(k - 1)
    result.Q = q
    result.df_Q = df_q
    result.pval_Q = float(stats.chi2.sf(q, df_q)) if df_q > 0 else float("nan")

    # Common-effects pooled estimate (inverse-variance, tau^2 = 0).
    if common:
        mu_c, se_c = _iv_pool(te_arr, se_arr, 0.0)
        result.TE_common = mu_c
        result.seTE_common = se_c
        result.lower_common = mu_c - z * se_c
        result.upper_common = mu_c + z * se_c
        result.statistic_common = mu_c / se_c
        result.pval_common = float(2.0 * stats.norm.sf(abs(mu_c / se_c)))

    # Random-effects pooled estimate (DerSimonian-Laird tau^2).
    tau2 = _dl_tau2(te_arr, se_arr)
    tau2_eff = 0.0 if math.isnan(tau2) else tau2
    result.tau2 = tau2
    result.tau = math.sqrt(tau2) if not math.isnan(tau2) else float("nan")
    i2, lower_i2, upper_i2 = _isquared(q, df_q, level)
    result.I2 = i2
    result.lower_I2 = lower_i2
    result.upper_I2 = upper_i2
    if random:
        mu_r, se_r = _iv_pool(te_arr, se_arr, tau2_eff)
        result.TE_random = mu_r
        result.seTE_random = se_r
        result.lower_random = mu_r - z * se_r
        result.upper_random = mu_r + z * se_r
        result.statistic_random = mu_r / se_r
        result.pval_random = float(2.0 * stats.norm.sf(abs(mu_r / se_r)))

    # Per-pair direct evidence (group consecutive oriented comparisons).
    pairs = []
    i = 0
    while i < k:
        j = i
        while j < k and oriented[j][0] == oriented[i][0] \
                and oriented[j][1] == oriented[i][1]:
            j += 1
        y = [r[2] for r in oriented[i:j]]
        s = [r[3] for r in oriented[i:j]]
        kk = j - i
        mu_c, se_c = _iv_pool(y, s, 0.0)
        qp = _cochran_q(y, s)
        t2p = _dl_tau2(y, s)
        t2p_eff = 0.0 if math.isnan(t2p) else t2p
        mu_r, se_r = _iv_pool(y, s, t2p_eff)
        i2p, _, _ = _isquared(qp, float(kk - 1), level)
        pairs.append({
            "treat1": oriented[i][0],
            "treat2": oriented[i][1],
            "k": kk,
            "TE_common": mu_c,
            "seTE_common": se_c,
            "lower_common": mu_c - z * se_c,
            "upper_common": mu_c + z * se_c,
            "TE_random": mu_r,
            "seTE_random": se_r,
            "lower_random": mu_r - z * se_r,
            "upper_random": mu_r + z * se_r,
            "Q": qp,
            "I2": i2p,
            "tau2": t2p,
        })
        i = j
    result.pairs = pairs

    return result
