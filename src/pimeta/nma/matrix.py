"""Treatment x treatment matrices and comparison tables.

``netmatrix``, ``netcomparison`` and ``netbind`` — all of them read a fitted
object and return a table, so none of them touches the fitting code.  Split out
of :mod:`pimeta.netmeta`; the public names are re-exported there.
"""

from __future__ import annotations

import numpy as np
from scipy import stats

from .component import NetcombResult, _compsplit, _create_c_matrix


def _by_summary(x, index1, index2=None, func="mode", ties_method="first",
                sep=":", rng=None):
    """``netmeta:::bySummary`` with ``long = FALSE``.

    Returns a dict of columns ``indices``, ``sum``, ``min``, ``max``,
    ``mean``, ``median``, ``mode`` — one row per distinct index (pair).
    """
    xv = np.asarray(x, dtype=float).ravel()
    idx = ([f"{a}{sep}{b}" for a, b in zip(index1, index2)]
           if index2 is not None else [str(v) for v in index1])
    keys = list(dict.fromkeys(idx))
    out = {"indices": keys, "sum": [], "min": [], "max": [],
           "mean": [], "median": [], "mode": []}
    for key in keys:
        vals = xv[[i for i, v in enumerate(idx) if v == key]]
        vals = vals[~np.isnan(vals)]
        if vals.size == 0:
            for f in ("sum", "min", "max", "mean", "median", "mode"):
                out[f].append(float('nan'))
            continue
        out["sum"].append(float(np.sum(vals)))
        out["min"].append(float(np.min(vals)))
        out["max"].append(float(np.max(vals)))
        out["mean"].append(float(np.mean(vals)))
        out["median"].append(float(np.median(vals)))
        uniq, counts = np.unique(vals, return_counts=True)
        best = np.flatnonzero(counts == counts.max())
        if best.size > 1:
            if ties_method == "last":
                pick = best[-1]
            elif ties_method == "random":
                pick = int(rng.choice(best)) if rng is not None else best[0]
            else:
                pick = best[0]
        else:
            pick = best[0]
        out["mode"].append(float(uniq[pick]))
    return out


class NetmatrixResult:
    """Result of :func:`netmatrix` — a treatment x treatment summary matrix."""

    __slots__ = ("matrix", "trts", "func", "levels", "labels")

    def __init__(self, matrix, trts, func, levels, labels):
        self.matrix = matrix
        self.trts = trts
        self.func = func
        self.levels = levels
        self.labels = labels

    def __repr__(self):
        return f"NetmatrixResult(n={len(self.trts)}, func={self.func!r})"


def netmatrix(result, var, levels=None, labels=None, func="mode",
              ties_method="first", seed=None):
    """``netmeta::netmatrix`` — summarise a study-level variable per comparison.

    Builds an ``n x n`` matrix holding, for every pair of treatments that is
    compared directly, a summary (``"mode"``, ``"min"``, ``"max"``, ``"mean"``,
    ``"median"`` or ``"sum"``) of a study-level variable.  Unobserved pairs and
    the diagonal are ``NaN``.

    Args:
        result: a :class:`NetmetaResult`.
        var: one value per study/comparison.
        levels, labels: optional recoding of the summary values.
        func: summary function name.
        ties_method: ``"first"``, ``"last"`` or ``"random"`` (for the mode).
        seed: RNG seed used when ``ties_method="random"``.

    Returns:
        :class:`NetmatrixResult`.
    """
    if func not in ("mode", "min", "max", "mean", "median", "sum"):
        raise ValueError(
            "func must be one of mode/min/max/mean/median/sum"
        )
    if ties_method not in ("first", "last", "random"):
        raise ValueError("ties_method must be 'first', 'last' or 'random'")

    trts = list(result.trts)
    t1 = [str(v) for v in result.treat1_raw]
    t2 = [str(v) for v in result.treat2_raw]
    n = len(t1)
    v = np.asarray(var, dtype=float).ravel()
    if v.size == 1 and n > 1:
        v = np.repeat(v, n)
    if v.size != n:
        raise ValueError(
            f"length of 'var' ({v.size}) does not match the number of "
            f"comparisons ({n})"
        )

    rng = np.random.default_rng(seed) if seed is not None else None
    summ = _by_summary(v, t1, t2, func="mode", ties_method=ties_method,
                       sep=":", rng=rng)
    vals = np.asarray(summ[func], dtype=float)

    if levels is not None:
        if labels is None:
            labels = levels
        if len(levels) != len(labels):
            raise ValueError(
                "Different lengths of arguments 'levels' and 'labels'."
            )
        mapped = []
        for cur in vals:
            hit = [k for k, lv in enumerate(levels) if float(lv) == cur]
            mapped.append(labels[hit[0]] if hit else float('nan'))
        vals = np.asarray([float(m) if isinstance(m, (int, float)) else m
                           for m in mapped], dtype=object)

    mat = np.full((len(trts), len(trts)), float('nan'), dtype=object)
    pos = {t: i for i, t in enumerate(trts)}
    for j, key in enumerate(summ["indices"]):
        a, b = key.split(":", 1)
        if a not in pos or b not in pos:
            continue
        mat[pos[a], pos[b]] = vals[j]
        mat[pos[b], pos[a]] = vals[j]
    try:
        mat = mat.astype(float)
    except (TypeError, ValueError):
        pass
    return NetmatrixResult(mat, trts, func, levels, labels)


class NetcomparisonResult:
    """Result of :func:`netcomparison` (``netmeta::netcomparison``)."""

    __slots__ = ("comparison", "treat1", "treat2", "TE_common", "seTE_common",
                 "lower_common", "upper_common", "statistic_common",
                 "pval_common", "TE_random", "seTE_random", "lower_random",
                 "upper_random", "statistic_random", "pval_random",
                 "common", "random", "level", "trts", "comps", "inactive",
                 "B_matrix", "C_matrix", "add1", "add2")

    def __init__(self, **kw):
        for k in self.__slots__:
            setattr(self, k, kw.get(k))

    def __repr__(self):
        return f"NetcomparisonResult(k={len(self.comparison or [])})"

    def as_dict(self):
        return {k.replace("_", ".").replace("..", "."): getattr(self, k)
                for k in self.__slots__}


def netcomparison(result, treat1=None, treat2=None, common=True, random=True,
                  level=0.95, nchar_comps=None, backtransf=False):
    """``netmeta::netcomparison`` — estimate comparisons between combinations.

    Works on a :class:`NetcombResult`: the requested ``treat1`` vs ``treat2``
    combinations (e.g. ``"A+B"`` vs ``"plac"``) are expressed as contrasts of
    the estimated component effects.

    Args:
        result: a :class:`NetcombResult`.
        treat1, treat2: combinations to compare.  Defaults mirror R: all
            treatments vs the reference group.
        common, random: which models to report.
        level: confidence level.
        nchar_comps: component-name abbreviation length (unused, kept for API
            compatibility).
        backtransf: back-transform the estimates.

    Returns:
        :class:`NetcomparisonResult`.
    """
    if not isinstance(result, NetcombResult):
        raise TypeError("argument 'result' must be a NetcombResult")

    trts = list(result.trts)
    list(result.comps)
    inactive = list(result.inactive or [])
    sep_comps = result.sep_comps or "+"
    ref = result.reference_group or ""

    t1_orig, t2_orig = treat1, treat2
    if treat1 is None:
        treat1 = list(trts)
    elif isinstance(treat1, str):
        treat1 = [treat1]
    else:
        treat1 = list(treat1)
    if treat2 is None:
        treat2 = ref
        if treat2 == "":
            treat2 = treat1[0]
            treat1 = treat1[1:]
        else:
            treat1 = [t for t in treat1 if t != treat2]
    elif isinstance(treat2, str):
        treat2 = [treat2]
    else:
        treat2 = list(treat2)
    if len(treat1) == 1 and len(treat2) > 1:
        treat1 = treat1 * len(treat2)
    elif len(treat1) > 1 and len(treat2) == 1:
        treat2 = treat2 * len(treat1)
    if len(treat1) != len(treat2):
        raise ValueError("treat1 and treat2 must have the same length")

    ncomp = len(treat1)
    # reuse the C matrix of the netcomb fit (rows = combinations in result order)
    cnames = list(result.comps)
    cmat = np.asarray(result.C_matrix, dtype=float)
    if cmat.shape[1] != len(cnames):
        cmat, cnames, _ = _create_c_matrix(
            sorted(set(treat1) | set(treat2)),
            (result.inactive or [None])[0]
            if isinstance(result.inactive, list) else result.inactive,
            sep_comps,
        )

    b = np.zeros((ncomp, len(trts)))
    tpos = {t: i for i, t in enumerate(trts)}
    for i in range(ncomp):
        if treat1[i] in tpos:
            b[i, tpos[treat1[i]]] += 1.0
        if treat2[i] in tpos:
            b[i, tpos[treat2[i]]] -= 1.0
    x = b @ np.asarray(cmat, dtype=float)

    comp_c = np.asarray(result.Comp_common, dtype=float)
    comp_r = np.asarray(result.Comp_random, dtype=float)
    comp_na = np.isnan(comp_c)
    if comp_na.any():
        inest = np.abs(x[:, comp_na]).sum(axis=1) > 0
        x_use = x[:, ~comp_na]
    else:
        inest = np.zeros(ncomp, dtype=bool)
        x_use = x

    lplus_c = np.asarray(result.Lplus_matrix_common, dtype=float)
    lplus_r = np.asarray(result.Lplus_matrix_random, dtype=float)
    if comp_na.any():
        lplus_c = lplus_c[np.ix_(~comp_na, ~comp_na)]
        lplus_r = lplus_r[np.ix_(~comp_na, ~comp_na)]

    te_c = x_use @ comp_c[~comp_na]
    se_c = np.sqrt(np.diag(x_use @ lplus_c @ x_use.T))
    te_r = x_use @ comp_r[~comp_na]
    se_r = np.sqrt(np.diag(x_use @ lplus_r @ x_use.T))

    te_c[inest] = float('nan')
    se_c[inest] = float('nan')
    te_r[inest] = float('nan')
    se_r[inest] = float('nan')

    zc = te_c / se_c
    zr = te_r / se_r
    crit = float(stats.norm.isf((1.0 - level) / 2.0))
    # R labels each comparison by the components that differ between the arms
    comparison = []
    for a, b in zip(treat1, treat2):
        sa = set(_compsplit(a, sep_comps))
        sb = set(_compsplit(b, sep_comps))
        only1 = [c for c in _compsplit(a, sep_comps) if c not in sb]
        only2 = [c for c in _compsplit(b, sep_comps) if c not in sa]
        comb1 = sep_comps.join(only1)
        comb2 = sep_comps.join(only2)
        if comb1 == "":
            comparison.append(comb2)
        elif comb2 == "":
            comparison.append(comb1)
        else:
            comparison.append(f"{comb1}{sep_comps}{comb2}")

    return NetcomparisonResult(
        comparison=comparison, treat1=list(treat1), treat2=list(treat2),
        TE_common=[float(v) for v in te_c],
        seTE_common=[float(v) for v in se_c],
        lower_common=[float(v) for v in te_c - crit * se_c],
        upper_common=[float(v) for v in te_c + crit * se_c],
        statistic_common=[float(v) for v in zc],
        pval_common=[float(v) for v in 2 * stats.norm.sf(np.abs(zc))],
        TE_random=[float(v) for v in te_r],
        seTE_random=[float(v) for v in se_r],
        lower_random=[float(v) for v in te_r - crit * se_r],
        upper_random=[float(v) for v in te_r + crit * se_r],
        statistic_random=[float(v) for v in zr],
        pval_random=[float(v) for v in 2 * stats.norm.sf(np.abs(zr))],
        common=common, random=random, level=level, trts=trts,
        comps=[c for j, c in enumerate(cnames) if cmat[:, j].any()],
        inactive=inactive, B_matrix=b, C_matrix=np.asarray(cmat, dtype=float),
        add1=t1_orig, add2=t2_orig,
    )


class NetbindResult:
    """Result of :func:`netbind` (``netmeta::netbind``)."""

    __slots__ = ("common", "random", "sm", "reference_group",
                 "baseline_reference", "names", "nets", "fixed")

    def __init__(self, common, random, sm, reference_group,
                 baseline_reference, names, nets):
        self.common = common
        self.random = random
        self.fixed = common
        self.sm = sm
        self.reference_group = reference_group
        self.baseline_reference = baseline_reference
        self.names = names
        self.nets = nets

    def __repr__(self):
        return (f"NetbindResult(nets={len(self.names)}, "
                f"common_rows={len(self.common)})")

    def as_dict(self):
        return {"common": self.common, "random": self.random,
                "fixed": self.common, "sm": self.sm}


def netbind(*results, name=None, common=None, random=None,
            reference_group=None, baseline_reference=True):
    """``netmeta::netbind`` — collect reference-comparison estimates.

    For each network the estimates of every treatment against
    ``reference_group`` are extracted from the common-effect and random-effects
    matrices and stacked into one long table (used to draw several networks in
    a single forest plot).

    Args:
        *results: :class:`NetmetaResult` objects.
        name: labels for the networks (default ``"Network 1"``, ...).
        common, random: restrict the output to one model.
        reference_group: reference treatment (default: the first network's).
        baseline_reference: keep the reference as the *column* (``True``) or as
            the *row* (``False``).

    Returns:
        :class:`NetbindResult`.
    """
    from ..netmeta import NetmetaResult  # local import: cycle

    nets = list(results)
    if not nets:
        raise ValueError("argument 'result' is required")
    for r in nets:
        if not isinstance(r, NetmetaResult):
            raise TypeError("all arguments must be NetmetaResult objects")

    if name is None:
        name = [f"Network {i + 1}" for i in range(len(nets))]
    elif isinstance(name, str):
        name = [name]
    name = list(name)
    if len(name) != len(nets):
        raise ValueError("length of 'name' must match the number of networks")

    ref = reference_group
    if ref is None:
        ref = getattr(nets[0], "reference_group", "") or ""
    if ref == "":
        ref = list(nets[0].trts)[0]

    want_common = True if common is None else bool(common)
    want_random = True if random is None else bool(random)

    def _rows(model):
        out = []
        for k, r in enumerate(nets):
            trts = list(r.trts)
            if ref not in trts:
                raise ValueError(
                    f"reference treatment '{ref}' is not in network "
                    f"'{name[k]}'"
                )
            j = trts.index(ref)
            if model == "common":
                te = np.asarray(r.TE_common, dtype=float)
                se = np.asarray(r.seTE_common, dtype=float)
                lo = np.asarray(r.lower_common, dtype=float)
                up = np.asarray(r.upper_common, dtype=float)
                st = np.asarray(r.statistic_common, dtype=float)
                pv = np.asarray(r.pval_common, dtype=float)
            else:
                te = np.asarray(r.TE_random, dtype=float)
                se = np.asarray(r.seTE_random, dtype=float)
                lo = np.asarray(r.lower_random, dtype=float)
                up = np.asarray(r.upper_random, dtype=float)
                st = np.asarray(r.statistic_random, dtype=float)
                pv = np.asarray(r.pval_random, dtype=float)
            for i, t in enumerate(trts):
                out.append({
                    "name": name[k], "treat": t,
                    "TE": float(te[i, j] if baseline_reference else te[j, i]),
                    "seTE": float(se[i, j] if baseline_reference else se[j, i]),
                    "lower": float(lo[i, j] if baseline_reference else lo[j, i]),
                    "upper": float(up[i, j] if baseline_reference else up[j, i]),
                    "statistic": float(st[i, j] if baseline_reference
                                       else st[j, i]),
                    "pval": float(pv[i, j] if baseline_reference else pv[j, i]),
                    "method": str(r.method_tau),
                })
        return out

    return NetbindResult(
        common=_rows("common") if want_common else [],
        random=_rows("random") if want_random else [],
        sm=getattr(nets[0], "sm", ""),
        reference_group=ref,
        baseline_reference=baseline_reference,
        names=name, nets=nets,
    )
