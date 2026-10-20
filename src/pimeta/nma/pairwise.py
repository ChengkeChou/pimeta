"""``netmeta::pairwise`` — arm-level data to comparison-level data.

Each study with ``a`` arms contributes ``a(a-1)/2`` comparison rows.  Binary
data (``event`` / ``n``) supports OR, RR, RD and ASD; continuous data
(``mean`` / ``sd`` / ``n``) supports MD and SMD.  The continuity correction and
the SMD study-level SD pooling follow netmeta exactly.

This module is one section of the network meta-analysis implementation;
:mod:`pimeta.netmeta` re-exports it and remains the public entry point.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

__all__ = ["PairwiseData", "pairwise"]


# ---------------------------------------------------------------------------
# pairwise: arm-level data -> comparison-level data (netmeta::pairwise)
# ---------------------------------------------------------------------------

def pairwise(
    treat=None,
    n=None,
    event=None,
    mean=None,
    sd=None,
    time=None,
    studlab=None,
    data=None,
    sm: str = "",
    incr: float = 0.5,
    allincr: bool = False,
    addincr: bool = False,
    allstudies: bool = False,
) -> "PairwiseData":
    """Convert arm-level data to comparison-level data (netmeta::pairwise).

    Supports binary (``event`` / ``n``; sm OR/RR/RD) and continuous
    (``mean`` / ``sd`` / ``n``; sm MD/SMD) data. Each study with ``a`` arms
    contributes ``a(a-1)/2`` pairwise comparisons (treat1 vs treat2).

    ``data`` may be a dict / ``DataFrame`` whose columns carry the same names
    as the arguments (``treat``, ``n``, ``event``, ``mean``, ``sd``,
    ``studlab``); an explicitly passed argument wins over the column.

    Not supported: R's ``TE``/``seTE`` arguments, which feed contrast-level data
    straight through instead of deriving it from the arms.
    """
    if data is not None:
        # metafor's ``data=`` just evaluates the remaining arguments inside the
        # data frame.  Accept the same column names as the parameter names.
        if hasattr(data, "to_dict"):
            cols = data.to_dict(orient="list")
        elif isinstance(data, dict):
            cols = data
        elif hasattr(data, "__dict__") and not isinstance(data, type):
            cols = {k: v for k, v in vars(data).items()
                    if not k.startswith("_")}
        else:
            raise TypeError("argument 'data' must be a dict or a DataFrame")

        def _from(name, cur):
            if cur is not None:
                return cur
            return cols.get(name)

        treat = _from("treat", None)
        if treat is None:
            raise ValueError("'data' must contain a 'treat' column")
        n = _from("n", n)
        event = _from("event", event)
        mean = _from("mean", mean)
        sd = _from("sd", sd)
        time = _from("time", time)
        studlab = _from("studlab", studlab)

    def _vec(v):
        """Accept a scalar or a sequence; R vectors are per-arm."""
        if v is None:
            return None
        arr = np.atleast_1d(np.asarray(v, dtype=float)).ravel()
        if arr.size == 1 and k_arms > 1:
            arr = np.repeat(arr, k_arms)      # R recycles length-1 vectors
        return [float(x) for x in arr]

    treat = list(treat)
    k_arms = len(treat)

    n = _vec(n)
    event = _vec(event)
    mean = _vec(mean)
    sd = _vec(sd)
    time = _vec(time)

    for nm, v in (("n", n), ("event", event), ("mean", mean), ("sd", sd),
                  ("time", time)):
        if v is not None and len(v) != k_arms:
            raise ValueError(
                f"length of '{nm}' ({len(v)}) does not match 'treat' "
                f"({k_arms})")

    if event is not None:
        if n is None:
            raise ValueError("'n' is required for binary data")
        if sm == "":
            sm = "OR"
        out = _pairwise_binary(treat, event, n, studlab, sm, incr, allincr, addincr)
    else:
        if n is not None and mean is None and sd is None:
            raise ValueError(
                "no outcome data supplied: pass 'event' (binary) or "
                "'mean'/'sd' (continuous) alongside 'n'")
        if n is None or mean is None or sd is None:
            raise ValueError(
                "either 'event'/'n' (binary) or 'mean'/'sd'/'n' (continuous) "
                "must be supplied")
        if sm == "":
            sm = "MD"
        out = _pairwise_continuous(treat, n, mean, sd, studlab, sm)

    return PairwiseData(sm=sm, **out)


@dataclass
class PairwiseData:
    """Comparison-level data produced by :func:`pairwise`."""

    sm: str
    TE: list
    seTE: list
    treat1: list
    treat2: list
    studlab: list
    narms: list

    def to_dict(self) -> dict:
        return {
            "sm": self.sm,
            "TE": self.TE,
            "seTE": self.seTE,
            "treat1": self.treat1,
            "treat2": self.treat2,
            "studlab": self.studlab,
            "narms": self.narms,
        }


def _pairwise_continuous(treat, n, mean, sd, studlab, sm):
    te, se, t1, t2, stud, narms = [], [], [], [], [], []
    by_study = {}
    for i, s in enumerate(studlab if studlab is not None else range(len(treat))):
        by_study.setdefault(s, []).append(i)
    for s, inds in by_study.items():
        a = len(inds)
        n_i = sum(n[i] for i in inds)  # study total n (each arm counted once)
        # For SMD with >2 arms, netmeta replaces every arm's sd by the
        # study-level pooled SD before forming pairwise comparisons.
        sds = [sd[i] for i in inds]
        if sm.upper() == "SMD" and a > 2:
            num = sum((n[i] - 1) * sd[i] ** 2 for i in inds)
            den = sum(n[i] - 1 for i in inds)
            sdp = math.sqrt(num / den)
            sds = [sdp] * a
        for x in range(a):
            for y in range(x + 1, a):
                ix, iy = inds[x], inds[y]
                t1.append(treat[ix])
                t2.append(treat[iy])
                if sm.upper() == "SMD":
                    # Cohen's d with pooled SD (n-1 denominators).
                    nx, ny = n[ix], n[iy]
                    sp2 = ((nx - 1) * sds[x] ** 2 + (ny - 1) * sds[y] ** 2) \
                        / (nx + ny - 2)
                    d = (mean[ix] - mean[iy]) / math.sqrt(sp2)
                    # netmeta overrides seTE for SMD:
                    #   seTE = sqrt(1/n1 + 1/n2 + TE^2/(2 * n_i))
                    vte = 1.0 / nx + 1.0 / ny + d ** 2 / (2.0 * n_i)
                    te.append(d)
                    se.append(math.sqrt(vte))
                else:  # MD
                    te.append(mean[ix] - mean[iy])
                    se.append(math.sqrt(sd[ix] ** 2 / n[ix] + sd[iy] ** 2 / n[iy]))
                stud.append(s)
                narms.append(a)
    return {"TE": te, "seTE": se, "treat1": t1, "treat2": t2,
            "studlab": [str(s) for s in stud], "narms": narms}


def _pairwise_binary(treat, event, n, studlab, sm, incr, allincr, addincr):
    te, se, t1, t2, stud, narms = [], [], [], [], [], []
    by_study = {}
    for i, s in enumerate(studlab if studlab is not None else range(len(treat))):
        by_study.setdefault(s, []).append(i)
    sm = sm.upper()
    for s, inds in by_study.items():
        a = len(inds)
        # Study-level continuity correction (netmeta::pairwise).
        if sm == "ASD":
            incr_study = 0.0
        else:
            n_zeros = sum(1 for i in inds if event[i] == 0)
            n_all = sum(1 for i in inds if event[i] == n[i])
            sparse = (n_zeros > 0) or (n_all > 0)
            if addincr:
                incr_study = incr
            elif allincr:
                incr_study = incr if n_zeros > 0 else 0.0
            else:
                incr_study = incr if sparse else 0.0
        for x in range(a):
            for y in range(x + 1, a):
                ix, iy = inds[x], inds[y]
                a1, n1 = event[ix], n[ix]
                a2, n2 = event[iy], n[iy]
                ie = ic = incr_study
                t1.append(treat[ix])
                t2.append(treat[iy])
                if sm == "RR":
                    # meta::metabin RR (RR.Cochrane = FALSE).
                    allevents = (a1 == n1 and a2 == n2)
                    e1 = a1 + ie * (0.0 if allevents else 1.0)
                    e2 = a2 + ic * (0.0 if allevents else 1.0)
                    te.append(math.log((e1 / (n1 + ie)) / (e2 / (n2 + ic))))
                    se.append(math.sqrt(
                        1 / e1 - 1 / (n1 + ie) + 1 / e2 - 1 / (n2 + ic)))
                elif sm == "RD":
                    te.append(a1 / n1 - a2 / n2)
                    se.append(math.sqrt(
                        (a1 + ie) * (n1 - a1 + ie) / (n1 + 2 * ie) ** 3
                        + (a2 + ic) * (n2 - a2 + ic) / (n2 + 2 * ic) ** 3))
                else:  # OR
                    e1 = a1 + ie
                    e2 = a2 + ic
                    te.append(math.log(
                        (e1 * (n2 - a2 + ic)) / ((n1 - a1 + ie) * e2)))
                    se.append(math.sqrt(
                        1 / e1 + 1 / (n1 - a1 + ie) + 1 / e2 + 1 / (n2 - a2 + ic)))
                stud.append(s)
                narms.append(a)
    return {"TE": te, "seTE": se, "treat1": t1, "treat2": t2,
            "studlab": [str(s) for s in stud], "narms": narms}
