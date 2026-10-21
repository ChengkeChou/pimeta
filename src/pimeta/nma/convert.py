"""Adapters for foreign network meta-analysis objects.

``netmeta::crossnma2netmeta`` and ``netmeta::gemtc2netmeta`` turn a ``crossnma``
fit or a ``gemtc`` network into pimeta's :class:`~pimeta.netmeta.NetmetaResult`.
Split out of :mod:`pimeta.netmeta`; the public names are re-exported there.
"""

from __future__ import annotations

import math

import numpy as np


def _as_dict_obj(x, what):
    """Accept a dict or an object with ``__dict__`` / ``_asdict``."""
    if isinstance(x, dict):
        return x
    if hasattr(x, "_asdict"):
        return dict(x._asdict())
    if hasattr(x, "__dict__"):
        return {k: v for k, v in vars(x).items() if not k.startswith("_")}
    raise TypeError(f"argument '{what}' must be a dict or a mapping-like object")


def crossnma2netmeta(x, sm="", method_tau="DL", reference_group="",
                     level=0.95):
    """``netmeta::crossnma2netmeta`` — convert a ``crossnma`` fit to netmeta.

    ``crossnma`` (Hamza et al.) fits a network meta-analysis that combines
    randomised and non-randomised evidence.  Its fit object carries a study
    level table with the columns ``studlab``, ``treat1``, ``treat2``, ``TE`` and
    ``seTE`` (accessed here either directly or through a ``data`` entry).

    Args:
        x: the ``crossnma`` object as a dict / mapping.
        sm: effect measure label.
        method_tau: heterogeneity estimator.
        reference_group: reference treatment.
        level: confidence level.

    Returns:
        :class:`NetmetaResult`.
    """
    from ..netmeta import netmeta  # local import: cycle

    obj = _as_dict_obj(x, "x")
    tab = obj.get("data", obj)
    tab = _as_dict_obj(tab, "x['data']")

    def _col(*names):
        for n in names:
            if n in tab:
                return list(tab[n])
        raise ValueError(f"could not find any of the columns {names} in 'x'")

    stud = _col("studlab", "study", "study_id")
    t1 = _col("treat1", "treatment1", "t1")
    t2 = _col("treat2", "treatment2", "t2")
    te = [float(v) for v in _col("TE", "yi", "theta")]
    se = [float(v) for v in _col("seTE", "sei", "se")]

    return netmeta(te, se, t1, t2, stud, sm=obj.get("sm", sm) or sm,
                   level=level, method_tau=method_tau,
                   reference_group=reference_group or obj.get("reference.group", ""))


def gemtc2netmeta(x, sm="", method_tau="DL", reference_group="",
                  level=0.95):
    """``netmeta::gemtc2netmeta`` — convert a ``gemtc`` network/fit to netmeta.

    ``gemtc`` (van Valkenhoef & Kuiper) stores the evidence either arm-based
    (``data.ab``: one row per arm with ``study``, ``treatment`` and the outcome
    columns) or contrast-based (``data.re``: ``study``, ``treatment``,
    ``diff``, ``std.err``).  Contrast data are used directly; arm-based data are
    converted into consecutive within-study contrasts.

    Args:
        x: the ``gemtc`` object as a dict / mapping (a ``network`` or a fit).
        sm, method_tau, reference_group, level: forwarded to ``netmeta``.

    Returns:
        :class:`NetmetaResult`.
    """
    from ..netmeta import netmeta  # local import: cycle

    obj = _as_dict_obj(x, "x")
    net = _as_dict_obj(obj.get("network", obj), "x['network']")

    te, se, t1, t2, stud = [], [], [], [], []

    def _find(tab, *names):
        for n in names:
            if n in tab:
                return list(tab[n])
        return None

    re_tab = net.get("data.re") or obj.get("data.re")
    ab_tab = net.get("data.ab") or obj.get("data.ab")

    if re_tab:
        tab = _as_dict_obj(re_tab, "x['network']['data.re']")
        s = _find(tab, "study", "studlab")
        tr = _find(tab, "treatment", "treat")
        d = _find(tab, "diff", "TE", "mean")
        e = _find(tab, "std.err", "se", "seTE")
        if s is None or tr is None or d is None or e is None:
            raise ValueError("data.re must contain study/treatment/diff/std.err")
        # gemtc contrast rows are relative to the study's baseline treatment
        by_study = {}
        for i in range(len(s)):
            by_study.setdefault(str(s[i]), []).append(i)
        for st, idx in by_study.items():
            base = str(tr[idx[0]])
            for i in idx[1:]:
                t1.append(str(tr[i]))
                t2.append(base)
                te.append(float(d[i]))
                se.append(float(e[i]))
                stud.append(st)
    elif ab_tab:
        tab = _as_dict_obj(ab_tab, "x['network']['data.ab']")
        s = _find(tab, "study", "studlab")
        tr = _find(tab, "treatment", "treat")
        if s is None or tr is None:
            raise ValueError("data.ab must contain study and treatment columns")
        keys = [k for k in tab if k not in ("study", "studlab", "treatment",
                                            "treat")]
        if not keys:
            raise ValueError("data.ab must contain an outcome column")
        out = np.asarray(tab[keys[0]], dtype=float)
        by_study = {}
        for i in range(len(s)):
            by_study.setdefault(str(s[i]), []).append(i)
        for st, idx in by_study.items():
            if len(idx) < 2:
                continue
            base = idx[0]
            for i in idx[1:]:
                t1.append(str(tr[i]))
                t2.append(str(tr[base]))
                te.append(float(out[i] - out[base]))
                se.append(float("nan"))
                stud.append(st)
        if any(math.isnan(v) for v in se):
            raise ValueError(
                "arm-based gemtc data need a variance column to be converted; "
                "supply data.re or fitted estimates instead"
            )
    else:
        raise ValueError(
            "could not find 'data.re' or 'data.ab' in the gemtc network object"
        )

    return netmeta(te, se, t1, t2, stud, sm=sm, level=level,
                   method_tau=method_tau, reference_group=reference_group)
