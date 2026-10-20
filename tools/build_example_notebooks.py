"""Build the example notebooks in ``examples/`` and really execute them.

No Jupyter is required: cells run through :func:`exec` in one shared namespace,
stdout is captured, and every Matplotlib figure a cell creates is rendered to
PNG and embedded.  The result is an ordinary ``.ipynb`` that GitHub, VS Code and
Jupyter all render with the outputs visible.

    python tools/build_example_notebooks.py            # rebuild every notebook
    python tools/build_example_notebooks.py 01         # only names containing "01"

Cells are plain Python — no IPython magics — so ``tests/test_example_notebooks.py``
can execute them without a kernel and catch bit-rot.
"""

from __future__ import annotations

import ast
import base64
import contextlib
import io
import json
import sys
import traceback
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "examples"
NB_VERSION = {"nbformat": 4, "nbformat_minor": 5}


# --------------------------------------------------------------------------- #
# execution
# --------------------------------------------------------------------------- #
def _render(value: object) -> dict | None:
    """Turn a cell's last expression into a Jupyter ``execute_result`` bundle."""
    if value is None:
        return None
    data: dict[str, object] = {"text/plain": repr(value)}
    html = None
    if hasattr(value, "to_html"):
        try:
            html = value.to_html()
        except Exception:                            # pragma: no cover - safety
            html = None
    elif hasattr(value, "_repr_html_"):
        html = value._repr_html_()
    if html:
        data["text/html"] = html
    elif hasattr(value, "to_string"):
        data["text/plain"] = value.to_string()
    return {"data": data, "metadata": {}, "output_type": "execute_result"}


def run_cell(source: str, namespace: dict) -> list[dict]:
    """Execute one cell; return its Jupyter outputs."""
    tree = ast.parse(source)
    body = tree.body
    last = None
    if body and isinstance(body[-1], ast.Expr):
        last = body.pop()

    buf = io.StringIO()
    before = set(plt.get_fignums())
    with contextlib.redirect_stdout(buf):
        if body:
            exec(compile(ast.Module(body=body, type_ignores=[]), "<cell>", "exec"),
                 namespace)
        value = eval(compile(ast.Expression(last.value), "<cell>", "eval"), namespace) \
            if last is not None else None

    outputs: list[dict] = []
    text = buf.getvalue()
    if text:
        outputs.append({"name": "stdout", "output_type": "stream",
                        "text": text.splitlines(keepends=True)})

    for num in sorted(set(plt.get_fignums()) - before):
        fig = plt.figure(num)
        png = io.BytesIO()
        fig.savefig(png, format="png", dpi=110, bbox_inches="tight")
        outputs.append({
            "data": {"image/png": base64.b64encode(png.getvalue()).decode("ascii"),
                     "text/plain": ["<Figure>"]},
            "metadata": {},
            "output_type": "display_data",
        })
        plt.close(fig)

    rendered = _render(value)
    if rendered is not None:
        outputs.append(rendered)
    return outputs


# --------------------------------------------------------------------------- #
# notebook assembly
# --------------------------------------------------------------------------- #
def build(name: str, cells: list[tuple[str, str]]) -> tuple[dict, int]:
    namespace: dict[str, object] = {"__name__": "__main__"}
    out_cells: list[dict] = []
    count = 0
    errors = 0
    for index, (kind, source) in enumerate(cells):
        cell_id = f"{name.split('-')[0]}-{index:02d}"
        source_lines = source.splitlines(keepends=True)
        if kind == "markdown":
            out_cells.append({"cell_type": "markdown", "id": cell_id,
                              "metadata": {}, "source": source_lines})
            continue
        count += 1
        try:
            outputs = run_cell(source, namespace)
        except Exception:                            # noqa: BLE001
            errors += 1
            traceback.print_exc()
            print(f"  !! cell {cell_id} failed", file=sys.stderr)
            raise
        out_cells.append({"cell_type": "code", "execution_count": count,
                          "id": cell_id, "metadata": {}, "outputs": outputs,
                          "source": source_lines})
    notebook = {
        "cells": out_cells,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python",
                           "name": "python3"},
            "language_info": {"name": "python", "version": "3"},
        },
        **NB_VERSION,
    }
    return notebook, count


def write(path: Path, notebook: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(notebook, ensure_ascii=False, indent=1) + "\n",
                    encoding="utf-8", newline="\n")


def main(argv: list[str]) -> int:
    from notebook_sources import NOTEBOOKS

    wanted = argv
    built = 0
    for name, cells in NOTEBOOKS:
        if wanted and not any(w in name for w in wanted):
            continue
        notebook, code_cells = build(name, cells)
        write(OUT / name, notebook)
        images = sum(1 for c in notebook["cells"] if c["cell_type"] == "code"
                     for o in c["outputs"] if o["output_type"] == "display_data")
        print(f"wrote examples/{name}: {code_cells} code cells, {images} figure(s)")
        built += 1
    print(f"built {built} notebook(s)")
    return 0


if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).parent))
    sys.exit(main(sys.argv[1:]))
