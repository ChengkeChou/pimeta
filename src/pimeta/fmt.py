"""``fmt*``/``dfround`` — metafor's number-formatting helpers.

Faithful transcriptions of ``metafor::fmtp``, ``fmtp2``, ``fmtt``, ``fmtx`` and
``dfround`` (metafor 5.2-1). R's ``formatC`` semantics are reproduced with a
small helper (:func:`_formatc`) so that widths, flags, ``digits`` and
``drop0trailing`` behave identically.
"""

from __future__ import annotations

import math
import re

import numpy as np

__all__ = ["fmtp", "fmtp2", "fmtt", "fmtx", "dfround"]


def _formatc(x, fmt="f", digits=4, flag="", width=None, drop0trailing=False):
    """A minimal ``formatC`` for numeric vectors."""
    if isinstance(x, (int, np.integer)) and fmt == "d":
        s = str(int(x))
    elif fmt == "f":
        s = f"{float(x):.{int(digits)}f}"
    elif fmt == "e":
        s = f"{float(x):.{int(digits)}e}"
    elif fmt == "g":
        s = f"{float(x):.{int(digits)}g}"
    else:
        s = str(x)

    if drop0trailing and "." in s:
        s = s.rstrip("0").rstrip(".")

    if flag == "-":
        if width is not None:
            s = s.ljust(int(width))
    elif flag == "0":
        if width is not None:
            s = s.rjust(int(width), "0")
    elif width is not None:
        s = s.rjust(int(width))
    return s


def _digits_for_cutoff(cutoff):
    """metafor: ``nchar(formatC(cutoff, "f", 10, drop0trailing=TRUE)) - 2``."""
    return len(_formatc(cutoff, "f", 10, drop0trailing=True)) - 2


def fmtp(p, digits=4, pname="", equal=False, sep=False, add0=False, quote=False):
    """``fmtp`` — format p-values (``"<0.0001"`` style).

    Args:
        p: p-value(s).
        digits: number of decimals shown above the cut-off.
        pname: optional name prefix (e.g. ``"p"``).
        equal: insert ``"="`` before the value.
        sep: insert a space between the name/equal sign and the value.
        add0: keep the leading zero in ``"<0.0001"``.
        quote: keep the result as ``str`` (R sets ``noquote`` when ``False``).

    Returns:
        A formatted ``str`` (scalar input) or list of ``str``.
    """
    arr = np.atleast_1d(np.asarray(p, dtype=float))
    digits = max(int(digits), 1)
    cutoff = "." + "0" * (digits - 1) + "1"
    ncutoff = float(cutoff)
    eq = "=" if equal else ""
    if sep:
        pname = (pname + " ") if pname != "" else ""
        sep_s = " "
    else:
        sep_s = ""

    out = []
    for x in arr:
        if np.isnan(x):
            out.append(f"{pname}{eq}{sep_s}NA")
        elif x >= ncutoff:
            out.append(f"{pname}{eq}{sep_s}{_formatc(x, 'f', digits)}")
        else:
            out.append(f"{pname}<{sep_s}{'0' if add0 else ''}{cutoff}")
    if np.isscalar(p) or np.ndim(p) == 0:
        return out[0]
    return out


def fmtp2(p, cutoff=(0.001, 0.06), pname="p", sep=True, add0=False, quote=False):
    """``fmtp2`` — format p-values with two cut-offs (meta's ``pval7`` style)."""
    arr = np.atleast_1d(np.asarray(p, dtype=float))
    if len(cutoff) == 1:
        raise ValueError("argument 'cutoff' must be of length 2")
    cutoff = sorted(cutoff)
    if cutoff[0] == 0:
        raise ValueError("the lower 'cutoff' value must be > 0")
    digits1 = _digits_for_cutoff(cutoff[0])
    digits2 = _digits_for_cutoff(cutoff[1])
    if sep:
        pname = (pname + " ") if pname != "" else ""
        sep_s = " "
    else:
        sep_s = ""

    out = []
    for x in arr:
        if np.isnan(x):
            out.append(f"{pname}={sep_s}NA")
        elif x < cutoff[0]:
            out.append(f"{pname}<{sep_s}{_formatc(cutoff[0], 'f', digits1)}")
        elif x < cutoff[1]:
            out.append(f"{pname}={sep_s}{_formatc(x, 'f', digits1)}")
        else:
            out.append(f"{pname}={sep_s}{_formatc(x, 'f', digits2)}")
    if not add0:
        out = [o.replace("0.", ".") for o in out]
    if np.isscalar(p) or np.ndim(p) == 0:
        return out[0]
    return out


def fmtx(x, digits=4, flag="", quote=False, **kwargs):
    """``fmtx`` — format numbers with fixed decimals (``formatC`` semantics).

    Keyword options mirroring metafor: ``addwidth``, ``drop0ifint``, ``add0``,
    ``thresh``, ``postfix``.
    """
    arr = np.asarray(x, dtype=float)
    if arr.ndim == 2:
        digs = digits if np.ndim(digits) else [digits] * arr.shape[1]
        out = np.empty(arr.shape, dtype=object)
        for j in range(arr.shape[1]):
            col = fmtx(arr[:, j], digits=digs[j], flag=flag, quote=True, **kwargs)
            out[:, j] = np.atleast_1d(col)
        return out

    addwidth = kwargs.get("addwidth")
    width = None if addwidth is None else int(digits) + int(addwidth)
    drop0ifint = bool(kwargs.get("drop0ifint", False))
    add0 = bool(kwargs.get("add0", True))
    thresh = kwargs.get("thresh")
    postfix = kwargs.get("postfix", "")
    if thresh is not None:
        if arr.size != 1:
            raise ValueError("can only use 'thresh' when 'x' is a scalar")
        if abs(float(arr.ravel()[0])) <= thresh:
            digits = 0

    flat = arr.ravel()
    out = []
    for v in flat:
        if np.isnan(v):
            out.append(f"NA{postfix}")
            continue
        s = _formatc(
            v, "f", digits, flag=flag, width=width,
            drop0trailing=drop0ifint and isinstance(digits, (int, np.integer)),
        )
        if not add0:
            s = s.replace("0.", ".")
        out.append(s + postfix)

    res = out[0] if arr.ndim == 0 or (arr.size == 1 and np.ndim(x) == 0) else out
    return res


def fmtt(val, tname="", df=None, df1=None, df2=None, pval=None, digits=4,
         pname="p-val", format=1, sep=True, quote=False, call=False, **kwargs):
    """``fmtt`` — format a test statistic together with its p-value.

    Args:
        val: test statistic (scalar).
        tname: name of the statistic, e.g. ``"z"``, ``"t"``, ``"Q"``, ``"F"``.
        df: single degrees-of-freedom value (``format=1``).
        df1, df2: numerator/denominator df (used instead of ``df``).
        pval: the p-value.
        digits: int or ``(test_digits, pval_digits)``.
        pname: label used for the p-value.
        format: 1 → ``"t(df = x) = ..., p-val = ..."``, 2 → ``"t = ..., df = ..., p-val = ..."``.
        sep: use spaces around ``=``.
        quote: keep as ``str``.
    """
    if np.size(val) != 1:
        raise ValueError("argument 'val' must be a scalar")
    if format not in (1, 2):
        raise ValueError("argument 'format' can only be equal to 1 or 2")
    if pval is None:
        raise ValueError("must specify the 'pval' argument")
    sepset = sep
    sep_s = " " if sep else ""
    flag = kwargs.get("flag", "")

    if np.ndim(digits) == 0:
        digs = {"test": int(digits), "pval": int(digits)}
    else:
        digs = {"test": int(digits[0]), "pval": int(digits[1])}

    x = fmtx(val, digs["test"], flag=flag, quote=True)
    p = fmtp(pval, digs["pval"], equal=True, sep=sepset, quote=True)

    if format == 1:
        if df is None:
            if df1 is not None and df2 is not None:
                return (f"{tname}(df1{sep_s}={sep_s}{df1},"
                        f" df2{sep_s}={sep_s}{round(df2, 2)}){sep_s}={sep_s}{x}, "
                        f"{pname}{sep_s}{p}")
            return f"{tname}{sep_s}={sep_s}{x}, {pname}{sep_s}{p}"
        return f"{tname}(df{sep_s}={sep_s}{df}){sep_s}={sep_s}{x}, {pname}{sep_s}{p}"

    if df is None:
        if df1 is not None and df2 is not None:
            return (f"{tname}{sep_s}={sep_s}{x}, df1{sep_s}={sep_s}{df1}, "
                    f"df2{sep_s}={sep_s}{round(df2, 2)}, {pname}{sep_s}{p}")
        return f"{tname}{sep_s}={sep_s}{x}, {pname}{sep_s}{p}"
    return (f"{tname}{sep_s}={sep_s}{x}, df{sep_s}={sep_s}{df}, "
            f"{pname}{sep_s}{p}")


def dfround(x, digits=0, drop0=True):
    """``dfround`` — round the numeric columns of a table to ``digits``.

    Works on a ``{name: column}`` mapping (returned as a new dict) or on a pandas
    ``DataFrame`` when pandas happens to be installed (returned as a DataFrame).
    Non-numeric columns are passed through untouched.

    Args:
        x: mapping of column name -> values, or a pandas ``DataFrame``.
        digits: int or per-column sequence.
        drop0: round (``True``) or format as fixed-decimal strings (``False``).
    """
    is_pandas = hasattr(x, "columns") and hasattr(x, "copy")
    cols = x.columns if is_pandas else list(x.keys())
    p = len(cols)
    digs = [int(digits)] * p if np.ndim(digits) == 0 else [int(d) for d in digits]
    d0 = [bool(drop0)] * p if np.ndim(drop0) == 0 else [bool(d) for d in drop0]
    if p != len(digs):
        raise ValueError(
            f"number of columns in 'x' ({p}) does not match 'digits' ({len(digs)})"
        )
    if p != len(d0):
        raise ValueError(
            f"number of columns in 'x' ({p}) does not match 'drop0' ({len(d0)})"
        )

    if is_pandas:
        out = x.copy()
        for i, col in enumerate(cols):
            if not np.issubdtype(out[col].dtype, np.number):
                continue
            vals = out[col].to_numpy(dtype=float)
            out[col] = np.round(vals, digs[i]) if d0[i] else [
                _formatc(v, "f", digs[i]) for v in vals
            ]
        return out

    out = {}
    for i, col in enumerate(cols):
        vals = np.asarray(x[col])
        if not np.issubdtype(vals.dtype, np.number):
            out[col] = x[col]
            continue
        fv = vals.astype(float)
        out[col] = np.round(fv, digs[i]) if d0[i] else [
            _formatc(v, "f", digs[i]) for v in fv
        ]
    return out
