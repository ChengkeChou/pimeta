"""Readers for Cochrane Review data — ``read.rm5`` / ``read.cdir`` / ``read.mtv``.

The three formats all describe a Cochrane/RevMan review as a set of
comparisons and outcomes with per-study data.  They are parsed into the same
long table that the R functions return, with the columns

``comp.no, outcome.no, group.no, overall, type, method, sm, studlab,``
``event.e, n.e, event.c, n.c, mean.e, sd.e, mean.c, sd.c,``
``O.E, V, TE, seTE, label.e, label.c, complab, outclab``

``type`` is ``"D"`` (binary), ``"C"`` (continuous), ``"P"`` (generic
O-E/V) or ``"I"`` (inverse variance).  The result is a
:class:`ReviewDataResult` whose ``data`` list of dicts can be handed straight
to :func:`~.meta_compose.metacr`.
"""

from __future__ import annotations

import csv
import gzip
import io
import os
import xml.etree.ElementTree as ET

__all__ = ["read_rm5", "read_cdir", "read_mtv", "ReviewDataResult",
           "REVIEW_COLUMNS"]

REVIEW_COLUMNS = [
    "comp.no", "outcome.no", "group.no", "overall", "type", "method", "sm",
    "studlab", "event.e", "n.e", "event.c", "n.c", "mean.e", "sd.e",
    "mean.c", "sd.c", "O.E", "V", "TE", "seTE", "label.e", "label.c",
    "complab", "outclab",
]

# RevMan effect-measure codes -> meta's `sm` labels
_RM5_SM = {
    "1": "OR", "2": "RR", "3": "RD", "4": "PETO_OR",
    "5": "MD", "6": "SMD", "7": "ROM",
    "8": "Peto O-E (OR)", "9": "Peto O-E (RR)",
    "10": "HR", "11": "IRR",
}

_RM5_METHOD = {"1": "MH", "2": "Inverse", "3": "Peto"}

_NUMERIC = {"event.e", "n.e", "event.c", "n.c", "mean.e", "sd.e", "mean.c",
            "sd.c", "O.E", "V", "TE", "seTE"}


class ReviewDataResult:
    """Result of :func:`read_rm5` / :func:`read_cdir` / :func:`read_mtv`.

    Attributes:
        data: list of dicts with the columns in :data:`REVIEW_COLUMNS`.
        title: review title.
        label_e, label_c: names of the experimental / control groups.
        comparisons: list of comparison labels.
        source: the file that was read.
    """

    __slots__ = ("data", "title", "label_e", "label_c", "comparisons",
                 "source")

    def __init__(self, data, title="", label_e="", label_c="",
                 comparisons=None, source=""):
        self.data = data
        self.title = title
        self.label_e = label_e
        self.label_c = label_c
        self.comparisons = comparisons or []
        self.source = source

    def __repr__(self):
        return (f"ReviewDataResult(rows={len(self.data)}, "
                f"comparisons={len(self.comparisons)})")

    def as_dict(self):
        return {"data": self.data, "title": self.title}

    def to_frame(self):
        try:
            import pandas as pd
        except ImportError:  # pragma: no cover
            return self.data
        return pd.DataFrame(self.data, columns=REVIEW_COLUMNS)


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #

def _blank_row(**kw):
    row = {c: None for c in REVIEW_COLUMNS}
    row.update({"overall": True, "method": "Inverse", "sm": ""})
    row.update(kw)
    return row


def _num(txt):
    if txt is None:
        return None
    t = str(txt).strip()
    if t == "":
        return None
    try:
        return float(t)
    except ValueError:
        return None


def _detect_type(txt):
    t = str(txt or "").strip().upper()
    if t in ("D", "B", "BINARY", "DICHOTOMOUS"):
        return "D"
    if t in ("C", "CONT", "CONTINUOUS"):
        return "C"
    if t in ("P", "GENERIC", "O-E"):
        return "P"
    if t in ("I", "IV", "INVERSE"):
        return "I"
    return ""


# --------------------------------------------------------------------------- #
# read.rm5 — RevMan 5 (gzipped or plain XML)
# --------------------------------------------------------------------------- #

def read_rm5(file, encoding="utf-8"):
    """``meta::read.rm5`` — read a RevMan 5 ``.rm5`` (XML) review file.

    RevMan 5 stores the review as XML (usually gzip-compressed)::

        <COCHRANE_REVIEW>
          <DATA_AND_ANALYSES>
            <COMPARISON>

    Args:
        file: path to the ``.rm5`` file, or an open binary file object.
        encoding: text encoding of the embedded XML.

    Returns:
        :class:`ReviewDataResult`.
    """
    if hasattr(file, "read"):
        raw = file.read()
        source = getattr(file, "name", "")
    else:
        source = str(file)
        if not os.path.exists(source):
            raise FileNotFoundError(source)
        with open(source, "rb") as fh:
            raw = fh.read()
    if isinstance(raw, str):
        raw = raw.encode(encoding)
    if raw[:2] == b"\x1f\x8b":
        raw = gzip.decompress(raw)

    root = ET.fromstring(raw.decode(encoding, errors="replace"))
    title = ""
    for tag in ("TITLE", "REVTITLE"):
        node = root.find(f".//{tag}")
        if node is not None and (node.text or "").strip():
            title = node.text.strip()
            break

    data = []
    comparisons = []
    label_e = label_c = ""
    comp_no = 0
    for comp in root.iter("COMPARISON"):
        comp_no += 1
        cname = ""
        for tag in ("NAME", "COMPARISON_NAME", "TITLE"):
            node = comp.find(tag)
            if node is not None and (node.text or "").strip():
                cname = node.text.strip()
                break
        comparisons.append(cname)
        outcome_no = 0
        for out in comp.iter("OUTCOME"):
            outcome_no += 1
            sm_code = ""
            for tag in ("EFFECT_MEASURE", "DATA_TYPE", "TYPE"):
                node = out.find(tag)
                if node is not None and (node.text or "").strip():
                    sm_code = node.text.strip()
                    break
            groups = list(out.iter("GROUP"))
            for gi, grp in enumerate(groups):
                le = grp.get("LABEL") or ""
                if gi == 0:
                    label_e = le or label_e
                elif gi == 1:
                    label_c = le or label_c
            is_binary = sm_code in ("1", "2", "3", "4", "8", "9")
            for grp in groups:
                for row in grp.iter("DATA"):
                    row_vals = {c.tag: (c.text or "") for c in row}
                    method = _RM5_METHOD.get(row_vals.get("METHOD", "2"),
                                             "Inverse")
                    sm_label = _RM5_SM.get(sm_code, "")
                    rec = _blank_row(
                        **{"comp.no": comp_no, "outcome.no": outcome_no,
                           "group.no": 1,
                           "type": "D" if is_binary else "C",
                           "method": method, "sm": sm_label}
                    )
                    rec["studlab"] = row_vals.get("STUDY", "")
                    for key, col in (("E1", "event.e"), ("N1", "n.e"),
                                     ("E2", "event.c"), ("N2", "n.c"),
                                     ("M1", "mean.e"), ("S1", "sd.e"),
                                     ("M2", "mean.c"), ("S2", "sd.c")):
                        rec[col] = _num(row_vals.get(key))
                    data.append(rec)
    return ReviewDataResult(data, title=title, label_e=label_e,
                            label_c=label_c, comparisons=comparisons,
                            source=source)


# --------------------------------------------------------------------------- #
# read.cdir — Cochrane Data Interchange Format
# --------------------------------------------------------------------------- #

def read_cdir(file, encoding="utf-8", sep=None):
    """``meta::read.cdir`` — read a Cochrane ``.cdir`` review file.

    ``.cdir`` is a delimited text file.  Sections are introduced by lines such
    as ``@comparison``, ``@outcome`` and ``@study``; the remaining lines are
    tabular with a header row whose column names follow the RevMan naming
    (``Event1``, ``Total1``, ``Event2``, ``Total2``, ``Mean1``, ``SD1``,
    ``Mean2``, ``SD2``, ``ES``, ``SE``, ``OE``, ``V``).

    Args:
        file: path or open text/binary file object.
        encoding: text encoding.
        sep: field separator (default: auto-detect tab / comma / semicolon).

    Returns:
        :class:`ReviewDataResult`.
    """
    if hasattr(file, "read"):
        txt = file.read()
        source = getattr(file, "name", "")
    else:
        source = str(file)
        if not os.path.exists(source):
            raise FileNotFoundError(source)
        with open(source, "rb") as fh:
            txt = fh.read()
    if isinstance(txt, bytes):
        txt = txt.decode(encoding, errors="replace")

    lines = [ln.rstrip("\r\n") for ln in txt.splitlines()]
    if sep is None:
        head = next((ln for ln in lines if ln.strip()), "")
        sep = "\t" if "\t" in head else (";" if ";" in head else ",")

    title = ""
    data = []
    comparisons = []
    label_e = label_c = ""
    comp_no = outcome_no = 0
    header = None
    sm = ""
    method = "Inverse"

    def _flush(rows, ctype):
        for rec in rows:
            rec["comp.no"] = comp_no
            rec["outcome.no"] = outcome_no
            rec["type"] = ctype
            rec["sm"] = sm
            rec["method"] = method
            data.append(rec)

    pending = []
    ctype = "D"
    for ln in lines:
        low = ln.strip().lower()
        if not low:
            continue
        if low.startswith("@"):
            if pending:
                _flush(pending, ctype)
                pending = []
            if "title" in low:
                title = ln.split(sep, 1)[-1].strip() if sep in ln else ""
            elif "comparison" in low:
                comp_no += 1
                outcome_no = 0
                parts = [p.strip() for p in ln.split(sep)]
                comparisons.append(parts[1] if len(parts) > 1 else "")
                if len(parts) > 2:
                    label_e, label_c = parts[1], parts[2]
            elif "outcome" in low:
                outcome_no += 1
            elif "measure" in low or "effect" in low:
                frag = ln.split(sep)[-1].strip() if sep in ln else ""
                sm = frag.upper()
            elif "method" in low:
                frag = ln.split(sep)[-1].strip() if sep in ln else ""
                for m in ("MH", "Peto", "Inverse"):
                    if frag.lower() == m.lower():
                        method = m
                        break
            elif "type" in low:
                ctype = _detect_type(ln.split(sep)[-1]) or "D"
            continue
        fields = [f.strip() for f in ln.split(sep)]
        upper = [f.upper() for f in fields]
        if "STUDY" in upper or "STUDYID" in upper or "EVENT1" in upper:
            header = upper
            continue
        if header is None:
            continue
        row = dict(zip(header, fields))
        rec = _blank_row(studlab=row.get("STUDY") or row.get("STUDYID")
                         or row.get("NAME") or "")
        for key, col in (("EVENT1", "event.e"), ("TOTAL1", "n.e"),
                         ("EVENT2", "event.c"), ("TOTAL2", "n.c"),
                         ("MEAN1", "mean.e"), ("SD1", "sd.e"),
                         ("MEAN2", "mean.c"), ("SD2", "sd.c"),
                         ("OE", "O.E"), ("V", "V"), ("ES", "TE"),
                         ("TE", "TE"), ("SE", "seTE")):
            if key in row:
                rec[col] = _num(row[key])
        pending.append(rec)
    if pending:
        _flush(pending, ctype)
    return ReviewDataResult(data, title=title, label_e=label_e,
                            label_c=label_c, comparisons=comparisons,
                            source=source)


# --------------------------------------------------------------------------- #
# read.mtv — meta-view / "metaview" long format
# --------------------------------------------------------------------------- #

def read_mtv(file, encoding="utf-8"):
    """``meta::read.mtv`` — read a ``.mtv`` (MetaView) data file.

    The MetaView format is a whitespace/comma separated table with the columns
    ``id, comp, outcome, treat, n, event`` (binary) or
    ``id, comp, outcome, treat, n, mean, sd`` (continuous).  Each study
    contributes one row per arm; the arms are paired within a study.

    Args:
        file: path or open file object.
        encoding: text encoding.

    Returns:
        :class:`ReviewDataResult`.
    """
    if hasattr(file, "read"):
        txt = file.read()
        source = getattr(file, "name", "")
    else:
        source = str(file)
        if not os.path.exists(source):
            raise FileNotFoundError(source)
        with open(source, "rb") as fh:
            txt = fh.read()
    if isinstance(txt, bytes):
        txt = txt.decode(encoding, errors="replace")

    rows = list(csv.reader(io.StringIO(txt), skipinitialspace=True))
    rows = [r for r in rows if any(f.strip() for f in r)]
    if not rows:
        return ReviewDataResult([], source=source)

    hdr = [h.strip().lower() for h in rows[0]]
    body = rows[1:]

    def _idx(*names):
        for n in names:
            if n in hdr:
                return hdr.index(n)
        return None

    i_id = _idx("id", "study", "studlab")
    i_comp = _idx("comp", "comparison", "comp.no")
    i_out = _idx("outcome", "outcome.no")
    i_trt = _idx("treat", "treat1", "arm")
    i_n = _idx("n", "total")
    i_ev = _idx("event", "events", "r")
    i_mean = _idx("mean", "y")
    i_sd = _idx("sd", "s")
    is_binary = i_ev is not None

    studies = []
    for r in body:
        studies.append({
            "id": r[i_id] if i_id is not None else "",
            "comp": int(float(r[i_comp])) if i_comp is not None else 1,
            "outcome": int(float(r[i_out])) if i_out is not None else 1,
            "treat": r[i_trt] if i_trt is not None else "",
            "n": _num(r[i_n]) if i_n is not None else None,
            "event": _num(r[i_ev]) if i_ev is not None else None,
            "mean": _num(r[i_mean]) if i_mean is not None else None,
            "sd": _num(r[i_sd]) if i_sd is not None else None,
        })

    groups = {}
    for rec in studies:
        groups.setdefault((rec["comp"], rec["outcome"], rec["id"]), []).append(rec)

    data = []
    comparisons = []
    for (comp, outcome, sid), arms in groups.items():
        while len(comparisons) < comp:
            comparisons.append("")
        if len(arms) < 2:
            continue
        a, b = arms[0], arms[1]
        row = _blank_row(**{
            "comp.no": comp, "outcome.no": outcome,
            "type": "D" if is_binary else "C",
            "sm": "" if is_binary else "MD", "method": "Inverse",
            "studlab": sid,
        })
        if is_binary:
            row.update({"event.e": a["event"], "n.e": a["n"],
                        "event.c": b["event"], "n.c": b["n"]})
        else:
            row.update({"mean.e": a["mean"], "sd.e": a["sd"], "n.e": a["n"],
                        "mean.c": b["mean"], "sd.c": b["sd"], "n.c": b["n"]})
        data.append(row)
    return ReviewDataResult(data, comparisons=comparisons, source=source)
