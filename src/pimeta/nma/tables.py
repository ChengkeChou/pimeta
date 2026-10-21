"""Per-subgroup refits and per-comparison summaries.

``netmeta::subgroup`` and ``netmeta::nettable``.  Split out of
:mod:`pimeta.netmeta`; the public names are re-exported there.
"""

from __future__ import annotations

import numpy as np


class NetsubgroupResult:
    """Result of :func:`subgroup` for a netmeta object."""

    __slots__ = ("levels", "networks", "TE_common", "seTE_common",
                 "TE_random", "seTE_random", "tau2", "trts")

    def __init__(self, **kw):
        for k in self.__slots__:
            setattr(self, k, kw.get(k))

    def __repr__(self):
        return f"NetsubgroupResult(levels={self.levels})"

    def as_dict(self):
        return {"levels": self.levels, "networks": self.networks,
                "tau2": self.tau2}


def subgroup(result, subgroup, only_connected=False, method_tau=None,
             level_ma=None, common=True, random=True):
    """``netmeta::subgroup`` — refit the network within each subgroup.

    Args:
        result: a :class:`NetmetaResult`.
        subgroup: one subgroup label per **study**.
        only_connected: error out when a subgroup network is disconnected.
        method_tau: heterogeneity estimator (defaults to the model's).
        level_ma: confidence level (defaults to the model's).
        common, random: which models to record.

    Returns:
        :class:`NetsubgroupResult` with per-level 3-D arrays
        ``TE_common``/``seTE_common``/``TE_random``/``seTE_random`` of shape
        ``(n, n, n_levels)`` and the fitted networks.
    """
    from ..netmeta import netmeta  # local import: cycle

    trts = list(result.trts)
    n = len(trts)
    stud = [str(v) for v in result.studlab_raw]
    sg = [str(v) for v in np.asarray(subgroup).ravel()]

    # ``subgroup`` is a *study*-level variable, but the raw comparison rows
    # repeat each study.  Accept either one label per study (in order of first
    # appearance) or one label per comparison row.
    studies = []
    for s in stud:
        if s not in studies:
            studies.append(s)
    if len(sg) == len(studies):
        label_of = dict(zip(studies, sg))
    elif len(sg) == len(stud) and len(studies) == len(stud):
        label_of = dict(zip(studies, sg))
    elif len(sg) == len(stud):
        label_of = {}
        for s, g in zip(stud, sg):
            label_of.setdefault(s, g)
    else:
        raise ValueError(
            f"length of 'subgroup' ({len(sg)}) must equal the number of "
            f"studies ({len(studies)}) or comparisons ({len(stud)})")

    levels = sorted(set(label_of.values()))
    if method_tau is None:
        method_tau = result.method_tau
    if level_ma is None:
        level_ma = result.level

    te_c = np.full((n, n, len(levels)), float('nan'))
    se_c = np.full((n, n, len(levels)), float('nan'))
    te_r = np.full((n, n, len(levels)), float('nan'))
    se_r = np.full((n, n, len(levels)), float('nan'))
    tau2s = {}
    nets = {}

    t1 = list(result.treat1_raw)
    t2 = list(result.treat2_raw)
    tev = list(result.TE_raw)
    sev = list(result.seTE_raw)

    for k, lev in enumerate(levels):
        idx = [i for i, s in enumerate(stud) if label_of.get(s) == lev]
        if not idx:
            continue
        try:
            net_k = netmeta([tev[i] for i in idx], [sev[i] for i in idx],
                            [t1[i] for i in idx], [t2[i] for i in idx],
                            [stud[i] for i in idx], sm=result.sm,
                            level=level_ma, method_tau=method_tau,
                            reference_group=result.reference_group or "")
        except Exception:
            nets[lev] = None
            continue
        nets[lev] = net_k
        tau2s[lev] = net_k.tau2
        pos = {t: i for i, t in enumerate(net_k.trts)}
        for a in net_k.trts:
            for b in net_k.trts:
                i, j = pos[a], pos[b]
                gi, gj = trts.index(a), trts.index(b)
                te_c[gi, gj, k] = np.asarray(net_k.TE_common)[i, j]
                se_c[gi, gj, k] = np.asarray(net_k.seTE_common)[i, j]
                te_r[gi, gj, k] = np.asarray(net_k.TE_random)[i, j]
                se_r[gi, gj, k] = np.asarray(net_k.seTE_random)[i, j]

    return NetsubgroupResult(
        levels=levels, networks=nets, TE_common=te_c, seTE_common=se_c,
        TE_random=te_r, seTE_random=se_r, tau2=tau2s, trts=trts,
    )


class NettableResult:
    """Result of :func:`nettable` — per-comparison summary of the network."""

    __slots__ = ("rows", "trts")

    def __init__(self, rows, trts):
        self.rows = rows
        self.trts = trts

    def __repr__(self):
        return f"NettableResult(comparisons={len(self.rows)})"

    def as_dict(self):
        return {"rows": self.rows}

    def to_frame(self):
        try:
            import pandas as pd
        except ImportError:  # pragma: no cover
            return self.rows
        return pd.DataFrame(self.rows)


def nettable(result, level=0.95):
    """``netmeta::nettable`` — direct evidence per comparison.

    Returns one row per treatment comparison with the number of studies, the
    number of two-arm studies, the direct estimate and the network estimate.

    Args:
        result: a :class:`NetmetaResult`.
        level: confidence level (kept for API parity).

    Returns:
        :class:`NettableResult`.
    """
    trts = list(result.trts)
    t1 = [str(v) for v in result.treat1_raw]
    t2 = [str(v) for v in result.treat2_raw]
    np.asarray(result.TE_raw, dtype=float)
    np.asarray(result.seTE_raw, dtype=float)
    stud = [str(v) for v in result.studlab_raw]
    narms = np.asarray(result.narms_raw, dtype=float)

    {t: i for i, t in enumerate(trts)}
    groups = {}
    for i in range(len(t1)):
        key = tuple(sorted([str(t1[i]), str(t2[i])]))
        groups.setdefault(key, []).append(i)

    net_c = np.asarray(result.TE_common, dtype=float)
    net_r = np.asarray(result.TE_random, dtype=float)
    dir_c = np.asarray(result.TE_direct_common, dtype=float)
    dir_r = np.asarray(result.TE_direct_random, dtype=float)

    # netmeta lists *all* n(n-1)/2 comparisons; those without direct evidence
    # show "." in R and NA here.
    rows = []
    for i in range(len(trts)):
        for j in range(i + 1, len(trts)):
            a, b = trts[i], trts[j]
            idx = groups.get(tuple(sorted([a, b])), [])
            rows.append({
                "treat1": a, "treat2": b,
                "studies": len({stud[i] for i in idx}),
                "n.2arm": int(sum(1 for i in idx if narms[i] == 2)),
                "narms": int(max((narms[i] for i in idx), default=0)),
                "TE.direct.common": float(dir_c[i, j]),
                "TE.direct.random": float(dir_r[i, j]),
                "TE.network.common": float(net_c[i, j]),
                "TE.network.random": float(net_r[i, j]),
            })
    return NettableResult(rows, trts)
