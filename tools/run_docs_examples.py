"""Execute the Python examples in a Markdown document and refresh their output.

A block is considered an executable example when a ```` ```python ```` fence is
*immediately* followed by a ```` ```text ```` fence holding the recorded output.
Every other code block is left untouched, so illustrative snippets and shell
commands cost nothing.

Blocks share one namespace and run in document order, which is how the manual is
meant to be read.

Usage::

    python tools/run_docs_examples.py docs/zh/usage.md          # check
    python tools/run_docs_examples.py --update docs/zh/usage.md # rewrite output
    python tools/run_docs_examples.py docs/zh/usage.md --verbose

``tests/test_docs_examples.py`` runs the check so the examples cannot rot.
"""

from __future__ import annotations

import argparse
import contextlib
import io
import re
import sys
import warnings
from pathlib import Path

PY_OPEN = re.compile(r"^```python\s*$")
TEXT_OPEN = re.compile(r"^```text\s*$")
FENCE = "```"


class Example:
    """One ``python`` block plus the ``text`` block that follows it."""

    def __init__(self, code: str, first_line: int,
                 out_span: tuple[int, int] | None, output: str) -> None:
        self.code = code
        self.first_line = first_line          # 1-based, for messages
        self.out_span = out_span              # (start, end) line indexes, end exclusive
        self.output = output


def _scan(lines: list[str]) -> list[Example]:
    examples: list[Example] = []
    i = 0
    while i < len(lines):
        if not PY_OPEN.match(lines[i]):
            i += 1
            continue

        start = i + 1
        i = start
        while i < len(lines) and lines[i].rstrip() != FENCE:
            i += 1
        code = "".join(lines[start:i])
        code_end = i
        i += 1                                   # past the closing fence

        # Skip blank lines, then look for the recorded output block.
        j = i
        while j < len(lines) and not lines[j].strip():
            j += 1
        out_span: tuple[int, int] | None = None
        output = ""
        if j < len(lines) and TEXT_OPEN.match(lines[j]):
            out_start = j + 1
            k = out_start
            while k < len(lines) and lines[k].rstrip() != FENCE:
                k += 1
            out_span = (j, k + 1)
            output = "".join(lines[out_start:k])
            i = k + 1
        examples.append(Example(code, start, out_span, output))
        _ = code_end
    return examples


def _run(examples: list[Example]) -> list[str]:
    """Run every example in one namespace and return the captured stdout."""
    import matplotlib

    matplotlib.use("Agg")

    namespace: dict[str, object] = {"__name__": "__docs__"}
    results: list[str] = []
    for ex in examples:
        buf = io.StringIO()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            try:
                with contextlib.redirect_stdout(buf):
                    exec(compile(ex.code, f"<example@{ex.first_line}>", "exec"),
                         namespace)
            except Exception as exc:                     # noqa: BLE001
                raise RuntimeError(
                    f"example at line {ex.first_line} raised "
                    f"{type(exc).__name__}: {exc}") from exc
        results.append(_normalise(buf.getvalue()))
    return results


def _normalise(text: str) -> str:
    """Strip trailing spaces and collapse runs of blank lines."""
    lines = [ln.rstrip() for ln in text.splitlines()]
    out: list[str] = []
    for ln in lines:
        if ln or (out and out[-1]):
            out.append(ln)
    while out and not out[-1]:
        out.pop()
    return "".join(ln + "\n" for ln in out)


def process(path: Path, update: bool = False, verbose: bool = False) -> list[str]:
    """Check (or refresh) every example in *path*; return a list of problems."""
    text = path.read_text(encoding="utf-8")
    lines = text.splitlines(keepends=True)
    examples = _scan(lines)
    if not examples:
        return [f"{path}: no executable examples found"]

    produced = _run(examples)

    problems: list[str] = []
    if not update:
        for ex, got in zip(examples, produced, strict=True):
            if ex.out_span is None:
                problems.append(f"{path}:{ex.first_line}: example has no output block")
            elif ex.output != got:
                problems.append(
                    f"{path}:{ex.first_line}: recorded output does not match\n"
                    f"--- recorded ---\n{ex.output}--- actual ---\n{got}")
            elif verbose:
                print(f"{path}:{ex.first_line}: ok")
        return problems

    # Rewrite from the bottom so earlier indexes stay valid.
    for ex, got in reversed(list(zip(examples, produced, strict=True))):
        block = [FENCE + "text\n", got, FENCE + "\n"]
        if ex.out_span is None:
            problems.append(f"{path}:{ex.first_line}: example has no output block")
            continue
        lo, hi = ex.out_span
        lines[lo:hi] = block
        if verbose:
            print(f"{path}:{ex.first_line}: updated")
    path.write_text("".join(lines), encoding="utf-8", newline="\n")
    return problems


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("paths", nargs="+", type=Path)
    parser.add_argument("--update", action="store_true",
                        help="rewrite the recorded output instead of checking it")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args(argv)

    failures: list[str] = []
    for path in args.paths:
        failures.extend(process(path, update=args.update, verbose=args.verbose))

    for msg in failures:
        print(msg, file=sys.stderr)
    if not failures:
        print(f"ok: {len(args.paths)} document(s)")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
