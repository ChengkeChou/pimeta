"""Subsetting and previewing ``list.rma``-style result objects.

metafor's ``[.list.rma``, ``head.list.rma`` and ``tail.list.rma`` operate on
the *lists* that ``leave1out()``, ``cumul()``, ``confint()``, ``influence()``
and friends return: every element is a per-study vector, and ``x[i]`` slices
all of them together so the rows stay aligned.

Python has no ``[`` protocol to hook on a plain dataclass, so the same thing is
spelled with three explicit helpers:

* :func:`slice_rma` — ``x[i]``: keep the selected rows.
* :func:`head_rma` — ``head(x, n)``: the first ``n`` rows.
* :func:`tail_rma` — ``tail(x, n)``: the last ``n`` rows.

A "row" is any field whose value is a list/tuple/1-D array whose length equals
``x.k`` (the number of studies/estimates); ``k`` itself, scalar fields and the
pooled summary stay untouched, exactly like R's ``[.list.rma`` which subsets
each element and re-attaches the class.
"""

from __future__ import annotations

import copy
from dataclasses import fields, is_dataclass

import numpy as np

__all__ = ["slice_rma", "head_rma", "tail_rma", "as_data_frame_rma_list",
           "RowSlicable"]


class RowSlicable:
    """Mixin implementing R's ``[.list.rma`` as Python ``x[i]``.

    R's ``[`` is a *method* (``[.list.rma``) so ``x[1:3]`` subsets every
    per-study vector at once.  Python spells that as ``__getitem__``, which
    this mixin provides on top of :func:`slice_rma`, so ``x[i]``,
    ``x[1:3]`` and ``x[mask]`` all behave like their R counterparts.

    Result classes opt in by inheriting from it (``head``/``tail``/``slice_rma``
    keep working for classes that do not).
    """

    def __getitem__(self, index):
        return slice_rma(self, index)


def _row_lengths(obj, k):
    """Names of the fields that hold one value per row."""
    names = []
    if is_dataclass(obj):
        cand = [(f.name, getattr(obj, f.name, None)) for f in fields(obj)]
    else:
        cand = list(vars(obj).items()) if hasattr(obj, "__dict__") else []
    for name, val in cand:
        if isinstance(val, (list, tuple)):
            if k is None or len(val) == k:
                names.append(name)
        elif isinstance(val, np.ndarray):
            if val.ndim >= 1 and (k is None or val.shape[0] == k):
                names.append(name)
    return names


def slice_rma(x, index, copy_obj=True):
    """``x[index]`` — subset every per-study vector of a result object.

    Args:
        x: a result object (any dataclass or plain object) whose per-study
            vectors have length ``x.k``.
        index: a slice, an integer, a list of integers or a boolean mask
            (0-based, like Python — R's 1-based indices are shifted by the
            caller).
        copy_obj: return a shallow copy of ``x`` (default) rather than
            mutating it in place.

    Returns:
        A copy of ``x`` with the row-level fields subset and ``k`` updated.
    """
    k = getattr(x, "k", None)
    row_k = k if isinstance(k, int) else None
    names = _row_lengths(x, row_k)
    if not names:
        raise TypeError(
            f"{type(x).__name__} has no per-row fields to subset")

    out = copy.copy(x) if copy_obj else x
    for name in names:
        val = getattr(out, name)
        # ``np.atleast_1d`` keeps R's semantics: x[2] yields a length-1
        # result, never a bare scalar.
        if isinstance(val, tuple):
            new = tuple(np.atleast_1d(
                np.asarray(val, dtype=object)[index]).tolist())
        elif isinstance(val, list):
            new = list(np.atleast_1d(
                np.asarray(val, dtype=object)[index]).tolist())
        else:
            new = np.atleast_1d(np.asarray(val)[index])
        setattr(out, name, new)

    # keep `k` consistent when it is stored as a plain count
    if isinstance(getattr(out, "k", None), int):
        n = _n_rows(getattr(out, names[0]))
        out.k = n
    return out


def _n_rows(val):
    if isinstance(val, np.ndarray):
        return int(val.shape[0]) if val.ndim else 1
    try:
        return len(val)
    except TypeError:                      # pragma: no cover - scalars
        return 1


def head_rma(x, n=6):
    """``head(x, n)`` — the first ``n`` rows of a result object."""
    n = int(n)
    if n < 0:                              # pragma: no cover - R semantics
        raise ValueError("'n' must be non-negative")
    return slice_rma(x, slice(0, n))


def tail_rma(x, n=6):
    """``tail(x, n)`` — the last ``n`` rows of a result object."""
    n = int(n)
    if n < 0:                              # pragma: no cover - R semantics
        raise ValueError("'n' must be non-negative")
    return slice_rma(x, slice(-n, None) if n else slice(0, 0))


# R spells these ``head.list.rma`` / ``tail.list.rma``; ``head``/``tail`` are
# exported as package-level aliases in ``__init__`` because nothing else in
# pimeta needs the bare names.
head = head_rma
tail = tail_rma


def as_data_frame_rma_list(x, rows=None):
    """``as.data.frame(list.rma)`` — the per-study vectors as a dict of columns."""
    k = getattr(x, "k", None)
    names = _row_lengths(x, int(k) if isinstance(k, int) else None)
    out = {}
    for name in names:
        val = getattr(x, name)
        col = val.tolist() if isinstance(val, np.ndarray) else list(val)
        if rows is not None:
            col = [col[i] for i in np.atleast_1d(np.asarray(rows)).ravel()]
        out[name] = col
    return out
