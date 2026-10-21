"""Network complexity and component-combination effects.

``netcomplexity`` counts the independent cycles of a comparison graph (a pimeta
extra); ``netcomplex`` transcribes ``netmeta::netcomplex``.  Split out of
:mod:`pimeta.netmeta`; the public names are re-exported there.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .component import NetcombResult, _ci, _compsplit, _create_c_matrix


def netcomplexity(result) -> int:
    """Independent cycles of a comparison graph (a pimeta extra).

    The number of independent cycles is ``m - n + n_subnets`` (with ``m``
    comparisons, ``n`` treatments and ``n_subnets`` connected components).  A
    network without cycles has complexity 0.

    ``netmeta`` itself does not export such a function; the equivalent
    quantity in R is ``netconnection(...)$d``.  This helper is provided
    because it was historically (and misleadingly) called ``netcomplex`` here —
    that name now matches R's ``netmeta::netcomplex``, which computes
    combination effects of a component network meta-analysis.

    Args:
        result: a :class:`NetmetaResult` (or any object exposing ``m``, ``n``
            and ``n_subnets``).

    Returns:
        The network complexity as an ``int``.
    """
    m = int(result.m)
    n = int(result.n)
    ns = getattr(result, "n_subnets", None)
    if ns is None:
        ns = getattr(result, "nsubnets", 1)
    return m - n + int(ns if ns is not None else 1)


@dataclass
class NetcomplexResult:
    """Result of :func:`netcomplex` (``netmeta::netcomplex``)."""

    complex: list[str]
    comps: list[str]
    C_matrix: np.ndarray
    Comb_common: np.ndarray = field(default_factory=lambda: np.empty(0))
    seComb_common: np.ndarray = field(default_factory=lambda: np.empty(0))
    lower_Comb_common: np.ndarray = field(default_factory=lambda: np.empty(0))
    upper_Comb_common: np.ndarray = field(default_factory=lambda: np.empty(0))
    statistic_Comb_common: np.ndarray = field(default_factory=lambda: np.empty(0))
    pval_Comb_common: np.ndarray = field(default_factory=lambda: np.empty(0))
    Comb_random: np.ndarray = field(default_factory=lambda: np.empty(0))
    seComb_random: np.ndarray = field(default_factory=lambda: np.empty(0))
    lower_Comb_random: np.ndarray = field(default_factory=lambda: np.empty(0))
    upper_Comb_random: np.ndarray = field(default_factory=lambda: np.empty(0))
    statistic_Comb_random: np.ndarray = field(default_factory=lambda: np.empty(0))
    pval_Comb_random: np.ndarray = field(default_factory=lambda: np.empty(0))
    inactive: str | None = None
    level: float = 0.95
    common: bool = True
    random: bool = True

    def __repr__(self):
        return f"NetcomplexResult(complex={self.complex})"

    def _m(self, x):
        return x.tolist() if isinstance(x, np.ndarray) else x

    def to_dict(self) -> dict:
        return {
            "complex": self.complex,
            "comps": self.comps,
            "C_matrix": self._m(self.C_matrix),
            "Comb_common": self._m(self.Comb_common),
            "seComb_common": self._m(self.seComb_common),
            "lower_Comb_common": self._m(self.lower_Comb_common),
            "upper_Comb_common": self._m(self.upper_Comb_common),
            "statistic_Comb_common": self._m(self.statistic_Comb_common),
            "pval_Comb_common": self._m(self.pval_Comb_common),
            "Comb_random": self._m(self.Comb_random),
            "seComb_random": self._m(self.seComb_random),
            "lower_Comb_random": self._m(self.lower_Comb_random),
            "upper_Comb_random": self._m(self.upper_Comb_random),
            "statistic_Comb_random": self._m(self.statistic_Comb_random),
            "pval_Comb_random": self._m(self.pval_Comb_random),
            "inactive": self.inactive,
            "level": self.level,
            "common": self.common,
            "random": self.random,
        }


def netcomplex(result, complex=None, common=True, random=True, level=None):
    """``netmeta::netcomplex`` — combination effects of a component NMA.

    Estimates the effect of arbitrary treatment *complexes* (combinations of
    components) from a fitted :func:`netcomb` model::

        Comb = C %*% Comp
        seComb = sqrt(diag(C %*% Lplus %*% t(C)))

    Args:
        result: a :class:`NetcombResult`.
        complex: which complexes to evaluate.  ``None`` (the default) uses every
            treatment of the network; otherwise a sequence of complex labels
            such as ``["A+B", "A+C+B"]`` (a 2-D array is read rowwise).
        common, random: which models to fill in.
        level: confidence level (defaults to the model's).

    Returns:
        :class:`NetcomplexResult`.
    """
    if not isinstance(result, NetcombResult):
        raise TypeError("argument 'result' must be a NetcombResult")

    comps = list(result.comps)
    sep = result.sep_comps or "+"
    if level is None:
        level = result.level

    if complex is None:
        labels = list(result.trts)
        c_mat = np.asarray(result.C_matrix, dtype=float)
        if c_mat.shape[0] != len(labels):
            c_mat = _create_c_matrix(labels, result.inactive, sep)[0]
    else:
        arr = np.asarray(complex, dtype=object)
        if arr.ndim == 2:
            c_mat = np.asarray(arr, dtype=float)
            if c_mat.shape[1] == len(comps):
                labels = [str(i + 1) for i in range(c_mat.shape[0])]
            elif c_mat.shape[0] == len(comps):
                c_mat = c_mat.T
                labels = [str(i + 1) for i in range(c_mat.shape[0])]
            else:
                raise ValueError(
                    f"the complex matrix must have {len(comps)} columns "
                    f"(components), got {c_mat.shape[1]}")
        else:
            labels = [str(v) for v in arr.ravel()]
            c_mat = np.zeros((len(labels), len(comps)), dtype=float)
            for i, lab in enumerate(labels):
                for part in _compsplit(lab, sep):
                    if part not in comps:
                        raise ValueError(
                            f"unknown component {part!r} in complex {lab!r}; "
                            f"known components are {comps}")
                    c_mat[i, comps.index(part)] = 1.0

    out = NetcomplexResult(complex=labels, comps=comps, C_matrix=c_mat,
                           inactive=result.inactive, level=float(level),
                           common=bool(common), random=bool(random))

    for model in ("common", "random"):
        if not (common if model == "common" else random):
            continue
        comp = np.asarray(getattr(result, f"Comp_{model}"), dtype=float)
        lplus = np.asarray(getattr(result, f"Lplus_matrix_{model}"), dtype=float)
        if comp.size == 0:
            continue
        comb = c_mat @ comp
        var = np.einsum("ij,jk,ik->i", c_mat, lplus, c_mat)
        se = np.sqrt(np.clip(var, 0.0, None))
        lower, upper, stat, pval = _ci(comb, se, level)
        setattr(out, f"Comb_{model}", comb)
        setattr(out, f"seComb_{model}", se)
        setattr(out, f"lower_Comb_{model}", lower)
        setattr(out, f"upper_Comb_{model}", upper)
        setattr(out, f"statistic_Comb_{model}", stat)
        setattr(out, f"pval_Comb_{model}", pval)

    return out
