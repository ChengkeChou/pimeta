"""The example notebooks must be valid, executed and still runnable.

Two things are checked, both without a Jupyter kernel:

- every notebook already carries outputs, so what is committed is what the user
  sees on GitHub (an unexecuted notebook would show no numbers at all);
- every code cell still runs.

A notebook that needs an optional dependency (pandas, scikit-learn, xarray,
networkx) is skipped when that dependency is absent, exactly like the rest of
the suite.

Regenerate the notebooks with::

    python tools/build_example_notebooks.py
"""

from __future__ import annotations

import ast
import contextlib
import io
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pytest  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
NOTEBOOKS = sorted((ROOT / "examples").glob("*.ipynb"))


def _imports(source: str) -> set[str]:
    """Top-level module names a code cell imports."""
    names: set[str] = set()
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.Import):
            names.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
            names.add(node.module.split(".")[0])
    return names


def _missing(code_cells: list[str]) -> list[str]:
    wanted: set[str] = set()
    for source in code_cells:
        wanted |= _imports(source)
    missing = []
    for name in sorted(wanted):
        if name in sys.builtin_module_names:
            continue
        try:
            if __import__(name) is None:            # pragma: no cover
                continue
        except ImportError:
            missing.append(name)
    return missing


def _cells(path: Path) -> tuple[list[str], list[dict]]:
    notebook = json.loads(path.read_text(encoding="utf-8"))
    assert notebook["nbformat"] == 4, f"{path.name}: unexpected nbformat"

    sources: list[str] = []
    code: list[dict] = []
    for cell in notebook["cells"]:
        assert cell["cell_type"] in {"markdown", "code"}, cell["cell_type"]
        if cell["cell_type"] != "code":
            continue
        source = "".join(cell["source"])
        assert source.strip(), f"{path.name}: empty code cell"
        sources.append(source)
        code.append(cell)
    return sources, code


@pytest.mark.parametrize("path", NOTEBOOKS, ids=lambda p: p.name)
def test_example_notebook_is_executed_and_runs(path: Path) -> None:
    sources, code_cells = _cells(path)

    missing = _missing(sources)
    if missing:
        pytest.skip(f"needs {', '.join(missing)}")

    unexecuted = [c["id"] for c in code_cells if c.get("execution_count") is None]
    assert not unexecuted, (
        f"{path.name}: cells {unexecuted} were never run — "
        "rebuild with tools/build_example_notebooks.py"
    )
    assert any(c["outputs"] for c in code_cells), (
        f"{path.name}: no recorded output at all — "
        "rebuild with tools/build_example_notebooks.py"
    )

    namespace: dict[str, object] = {"__name__": "__main__"}
    with contextlib.redirect_stdout(io.StringIO()):
        for index, source in enumerate(sources):
            try:
                exec(compile(source, f"{path.name}:cell{index}", "exec"), namespace)
            finally:
                plt.close("all")
