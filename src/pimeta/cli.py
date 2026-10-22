"""Command-line interface for pimeta.

Two entry points, same implementation::

    pimeta fit studies.csv --yi TE --vi seTE2 --method REML
    python -m pimeta fit studies.csv --yi TE --vi seTE2 --method REML

Commands
--------
``fit``      pool effect sizes (``rma``) and print the coefficient table
``escalc``   compute effect sizes from arm-level data
``network``  fit a network meta-analysis (``netmeta``) and print the league

Input is CSV/TSV/JSON; column names are passed as ``--yi``/``--vi``/... so the
same frame can feed every command.  Output is TSV on stdout (pipe it into
anything) or CSV/JSON with ``-o``/``--json``.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import warnings
from pathlib import Path
from typing import Any

import numpy as np

from . import __version__
from .record.table import read_table as read_table_impl

__all__ = ["main", "build_parser"]


# --------------------------------------------------------------------------- #
# input / output
# --------------------------------------------------------------------------- #
def read_table(path: str, delimiter: str | None = None) -> dict:
    """Read a CSV/TSV/JSON/JSONL/NPZ file into ``{column: list}``.

    The parsing lives in :func:`pimeta.record.table.read_table` so the CLI and
    the analysis record share one implementation; see that function for the
    format rules.  Cells stay as text here -- ``_to_float`` does the numeric
    conversion, which keeps the error message ("non-numeric value at row 3")
    in the CLI where it is actionable.

    Args:
        path: file to read; the extension selects the parser.
        delimiter: CSV delimiter; ``None`` uses the extension and otherwise
            sniffs.

    Returns:
        A dict of columns.  Blank fields become ``None``.
    """
    return read_table_impl(path, delimiter)


def _to_float(columns: dict, name: str) -> list[float]:
    """Convert ``columns[name]`` to floats, or exit with a clear message."""
    values = columns.get(name)
    if values is None:
        have = ", ".join(sorted(columns)) or "(none)"
        raise SystemExit(f"column {name!r} not found; available columns: {have}")
    out: list[float] = []
    for i, value in enumerate(values):
        if value is None:
            out.append(float("nan"))
            continue
        try:
            out.append(float(value))
        except (TypeError, ValueError):
            raise SystemExit(
                f"column {name!r} has a non-numeric value at row {i + 1}: {value!r}"
            ) from None
    return out


def _emit(columns: dict, out: str | None = None, fmt: str = "tsv") -> str:
    """Write ``columns`` to ``out`` (or return the text).

    Args:
        columns: ``{name: sequence}``.
        out: destination path; when ``None`` the text is returned.
        fmt: ``"tsv"``, ``"csv"`` or ``"json"``.

    Returns:
        The rendered text when ``out`` is ``None``, otherwise ``""``.
    """
    if out is not None and fmt == "tsv":
        suffix = Path(out).suffix.lower()
        if suffix == ".csv":
            fmt = "csv"
        elif suffix == ".json":
            fmt = "json"
    keys = list(columns)
    rows = len(columns[keys[0]]) if keys else 0

    if fmt == "json":
        text = json.dumps({k: list(columns[k]) for k in keys}, indent=2)
    elif fmt == "csv":
        import io

        buffer = io.StringIO()
        writer = csv.writer(buffer, lineterminator="\n")
        writer.writerow(keys)
        for i in range(rows):
            writer.writerow([_cell(columns[k][i]) for k in keys])
        text = buffer.getvalue()
    else:
        lines = ["\t".join(keys)]
        for i in range(rows):
            lines.append("\t".join(_cell(columns[k][i]) for k in keys))
        text = "\n".join(lines) + "\n"

    if out is None:
        return text
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    Path(out).write_text(text, encoding="utf-8", newline="\n")
    print(f"wrote {out}", file=sys.stderr)
    return ""


def _cell(value):
    if value is None:
        return "NA"
    if isinstance(value, float):
        return f"{value:.6g}"
    return str(value)


# --------------------------------------------------------------------------- #
# commands
# --------------------------------------------------------------------------- #
def _cmd_fit(args) -> int:
    import pimeta as ma

    cols = read_table(args.data, args.delimiter)
    yi = _to_float(cols, args.yi)
    if args.vi:
        vi = _to_float(cols, args.vi)
        sei = None
    else:
        if not args.sei:
            raise SystemExit("pass either --vi or --sei")
        vi = None
        sei = _to_float(cols, args.sei)

    result = ma.rma(
        yi, vi=vi, sei=sei,
        mods=args.mods,
        slab=args.slab,
        data=cols if (args.mods or args.slab) else None,
        method=args.method,
        test=args.test,
        level=args.level,
        warn_missing=not args.no_warn_missing,
    )

    table = result.tidy(backend="dict") if hasattr(result, "tidy") else {}
    text = _emit(table, out=args.output, fmt=args.format)
    if text:
        sys.stdout.write(text)

    if args.forest:
        ma.forest(result.yi, result.vi, studlab=result.slab,
                  pooled=result, filename=args.forest)
    if args.output:
        glance = result.glance(backend="dict") if hasattr(result, "glance") else {}
        _emit(glance, out=str(Path(args.output).with_name(
            Path(args.output).stem + "_glance" + Path(args.output).suffix)),
            fmt=args.format)
    return 0


def _cmd_escalc(args) -> int:
    import pimeta as ma

    cols = read_table(args.data, args.delimiter)
    kwargs: dict[str, Any] = {}
    for name in ("ai", "bi", "ci", "di", "n1i", "n2i", "m1i", "m2i",
                 "sd1i", "sd2i", "x1i", "x2i", "t1i", "t2i", "xi", "ni", "ri"):
        value = getattr(args, name, None)
        if value:
            kwargs[name] = _to_float(cols, value)

    res = ma.escalc(args.measure, add=args.add, drop00=not args.keep00, **kwargs)
    out = {"yi": list(res.yi), "vi": list(res.vi)}
    text = _emit(out, out=args.output, fmt=args.format)
    if text:
        sys.stdout.write(text)
    return 0


def _cmd_network(args) -> int:
    import pimeta as ma

    cols = read_table(args.data, args.delimiter)
    te = _to_float(cols, args.te)
    se = _to_float(cols, args.se)
    treat1 = cols.get(args.treat1)
    treat2 = cols.get(args.treat2)
    if treat1 is None or treat2 is None:
        raise SystemExit(
            f"columns {args.treat1!r}/{args.treat2!r} not found; "
            f"available columns: {', '.join(sorted(cols))}")

    res = ma.netmeta(
        te, se, treat1, treat2,
        cols.get(args.studlab) if args.studlab else None,
        sm=args.sm,
    )

    trts = list(res.trts)
    matrix = np.asarray(res.TE_random if args.random else res.TE_common, dtype=float)
    table: dict[str, list] = {"treatment": trts}
    for j, name in enumerate(trts):
        table[str(name)] = [float(matrix[i, j]) for i in range(len(trts))]

    text = _emit(table, out=args.output, fmt=args.format)
    if text:
        sys.stdout.write(text)
    if args.graph:
        ma.netgraph(res, filename=args.graph)
    return 0


# --------------------------------------------------------------------------- #
# parser
# --------------------------------------------------------------------------- #
def build_parser() -> argparse.ArgumentParser:
    """Build the argument parser (exposed so tests can inspect it)."""
    parser = argparse.ArgumentParser(
        prog="pimeta",
        description="Pure-Python meta-analysis (metafor / netmeta / meta).",
    )
    parser.add_argument("--version", action="version",
                        version=f"pimeta {__version__}")
    sub = parser.add_subparsers(dest="command", required=True)

    def _common(p):
        p.add_argument("data", help="CSV/TSV/JSON file with one row per study")
        p.add_argument("--delimiter", default=None,
                       help="CSV delimiter (default: infer from the extension/sniff)")
        p.add_argument("-o", "--output", default=None,
                       help="write results to this file instead of stdout")
        p.add_argument("--format", choices=("tsv", "csv", "json"), default="tsv",
                       help="output format (default: tsv)")
        p.add_argument("--no-warn-missing", action="store_true",
                       help="drop incomplete rows silently")

    fit = sub.add_parser("fit", help="pool effect sizes with rma()")
    _common(fit)
    fit.add_argument("--yi", required=True, help="effect-size column")
    fit.add_argument("--vi", help="variance column")
    fit.add_argument("--sei", help="standard-error column (instead of --vi)")
    fit.add_argument("--mods", help="moderator formula, e.g. '~ dose + year'")
    fit.add_argument("--slab", help="study-label column")
    fit.add_argument("--method", default="REML",
                     help="tau^2 estimator (default: REML)")
    fit.add_argument("--test", default="z", choices=("z", "knha", "t", "adhoc"),
                     help="coefficient test (default: z)")
    fit.add_argument("--level", type=float, default=0.95, help="confidence level")
    fit.add_argument("--forest", default=None, help="also draw a forest plot here")
    fit.set_defaults(func=_cmd_fit)

    esc = sub.add_parser("escalc", help="compute effect sizes from raw data")
    _common(esc)
    esc.add_argument("--measure", required=True, help="RR, OR, MD, SMD, ...")
    for name in ("ai", "bi", "ci", "di", "n1i", "n2i", "m1i", "m2i", "sd1i",
                 "sd2i", "x1i", "x2i", "t1i", "t2i", "xi", "ni", "ri"):
        esc.add_argument(f"--{name}", default=None, help=f"{name} column")
    esc.add_argument("--add", type=float, default=0.5, help="continuity correction")
    esc.add_argument("--keep00", action="store_true",
                     help="keep studies with zero events in both arms")
    esc.set_defaults(func=_cmd_escalc)

    net = sub.add_parser("network", help="fit a network meta-analysis")
    _common(net)
    net.add_argument("--te", required=True, help="contrast column")
    net.add_argument("--se", required=True, help="standard-error column")
    net.add_argument("--treat1", required=True, help="first treatment column")
    net.add_argument("--treat2", required=True, help="second treatment column")
    net.add_argument("--studlab", help="study-label column")
    net.add_argument("--sm", default="", help="summary measure (MD, OR, ...)")
    net.add_argument("--common", action="store_true", default=True,
                     help="report common-effects estimates (default)")
    net.add_argument("--random", action="store_true", default=True,
                     help="report random-effects estimates (default)")
    net.add_argument("--graph", default=None, help="also draw the network graph here")
    net.set_defaults(func=_cmd_network)
    return parser


def main(argv: list[str] | None = None) -> int:
    """Run the CLI.  Returns a process exit code (0 on success)."""
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return int(args.func(args))
    except BrokenPipeError:                      # pragma: no cover - shell piping
        return 0
    except (FileNotFoundError, ValueError, SystemExit) as exc:
        message = str(exc) or exc.__class__.__name__
        print(f"pimeta: {message}", file=sys.stderr)
        return 2


if __name__ == "__main__":                       # pragma: no cover
    warnings.simplefilter("default")
    raise SystemExit(main())
