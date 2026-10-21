"""Treatment label abbreviation (``netmeta::treats``).

R's ``abbreviate()`` / ``stripchars()`` port, used whenever a treatment label
has to fit into a fixed-width table.  Pure string handling, no numerics.
"""

from __future__ import annotations

import numpy as np


def _isspace(ch):
    return ch in " \t\n\v\f\r"


def _first_char(s, i):
    """``#define FIRSTCHAR(i)`` — the preceding character is a space."""
    return _isspace(s[i - 1])


def _last_char(s, i):
    """``#define LASTCHAR(i)`` — end of a word."""
    return not _isspace(s[i - 1]) and (i + 1 >= len(s) or _isspace(s[i + 1]))


_VOWELS = frozenset("aeiou")


def _stripchars(name, minlen, usecl=True):
    """R's ``stripchars()`` (``src/main/character.c``) — one abbreviation.

    Characters are dropped from the end of words, at least one character of
    each word is kept: spaces first, then lower-case vowels, then lower-case
    consonants, then anything (index 0 is never touched).
    """
    s = name
    if len(s) < minlen:
        return s
    nspace = 0
    finished = False

    # pass 0: drop trailing spaces, remember the internal ones
    j = 1
    i = len(s) - 1
    while i > 0:
        if _isspace(s[i]):
            if j:
                s = s[:i]
            else:
                nspace += 1
        else:
            j = 0
        if len(s) - nspace <= minlen:
            finished = True
            break
        i -= 1

    passes = []
    if usecl:
        passes += [
            lambda t, k: t[k] in _VOWELS and _last_char(t, k),
            lambda t, k: t[k] in _VOWELS and not _first_char(t, k),
            lambda t, k: t[k].islower() and _last_char(t, k),
            lambda t, k: t[k].islower() and not _first_char(t, k),
        ]
    # brute force: anything that is not the first character of a word
    passes.append(lambda t, k: not _first_char(t, k) and not _isspace(t[k]))

    for pred in passes:
        if finished:
            break
        i = len(s) - 1
        while i > 0:
            if pred(s, i):
                s = s[:i] + s[i + 1:]
            if len(s) - nspace <= minlen:
                finished = True
                break
            i -= 1

    # finally drop the remaining internal spaces if still too long
    if len(s) > minlen:
        i = len(s) - 1
        while i > 0:
            if _isspace(s[i]):
                s = s[:i] + s[i + 1:]
            i -= 1
    return s


def _duplicated(x):
    """R's ``duplicated()`` — TRUE for elements seen earlier in the vector."""
    out = []
    seen = set()
    for v in x:
        out.append(v in seen)
        seen.add(v)
    return out


def _abbreviate(names, minlength=4, use_classes=True):
    """R's ``base::abbreviate`` (``method = "left.kept"``)."""
    minlength = int(minlength)
    if minlength <= 0:
        return ["" for _ in names]

    old = [str(v).strip(" \t\n\v\f\r") for v in names]

    uniq = []
    seen = set()
    for v in old:
        if v not in seen:
            seen.add(v)
            uniq.append(v)

    x = list(uniq)                 # current abbreviations (parallel to uniq)
    dup2 = [True] * len(uniq)
    these = list(uniq)             # names still being abbreviated
    while True:
        pos = [k for k, d in enumerate(dup2) if d]
        for k, v in zip(pos, these):
            x[k] = _stripchars(v, minlength, use_classes)
        dup2 = _duplicated(x)
        if not any(dup2):
            break
        minlength += 1
        sub = [x[k] for k, d in enumerate(dup2) if d]
        dup2 = [d or (v in sub) for v, d in zip(x, dup2)]
        these = [uniq[k] for k, d in enumerate(dup2) if d]

    if len(uniq) != len(old):
        x = [x[uniq.index(v)] for v in old]
    return x


def treats(x, nchar_trts=8, row=True):
    """``netmeta::treats`` — abbreviate treatment labels to ``nchar_trts`` chars.

    Args:
        x: a list of treatment labels, or a matrix whose row/column names are
            the treatments.
        nchar_trts: maximum label length.
        row: use the row names when ``x`` is a matrix.

    Returns:
        list of abbreviations.  When truncation would create duplicates the
        labels are abbreviated with R's ``abbreviate()`` (drop spaces, then
        lower-case vowels, then consonants, from the end of words), e.g.
        ``["aa1", "aa2", "bb"]`` -> ``["a1", "a2", "bb"]``.
    """
    a = np.asarray(x, dtype=object)
    if a.ndim == 2:
        names = None
        if hasattr(x, "index") or hasattr(x, "columns"):
            names = list(x.index) if row else list(x.columns)
        elif hasattr(x, "rownames"):
            names = list(x.rownames)
        if not names:
            names = [str(v) for v in range(a.shape[0 if row else 1])]
    else:
        names = [str(v) for v in a.ravel()]

    res = [s[:int(nchar_trts)] for s in names]
    if len(set(res)) != len(set(names)):
        res = _abbreviate(names, nchar_trts)
    return res
