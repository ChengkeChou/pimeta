"""The manual's examples must still run and still produce their recorded output.

``docs/zh/usage.md`` pairs every runnable ``python`` block with the ``text``
block holding its real output.  ``tools/run_docs_examples.py`` regenerates those
blocks; this test keeps them honest, so the manual cannot drift away from the
code.

Comparison ignores runs of spaces, because column alignment in ``DataFrame``
representations legitimately moves between pandas versions; the values still
have to match exactly.
"""

from __future__ import annotations

import importlib.util
import re
from pathlib import Path
from types import ModuleType

import pytest

from pimeta import interop

ROOT = Path(__file__).resolve().parents[1]
TOOL = ROOT / "tools" / "run_docs_examples.py"
MANUALS = [ROOT / "docs" / "usage.md", ROOT / "docs" / "zh" / "usage.md"]

# The manuals deliberately show the ecosystem integrations, so they need the
# optional extras that test_interop.py also skips without.
_NEEDED = ("pandas", "sklearn", "xarray", "networkx")
_MISSING = [name for name in _NEEDED if not interop.available().get(name)]

pytestmark = [
    pytest.mark.skipif(
        not (TOOL.exists() and all(p.exists() for p in MANUALS)),
        reason="manual or tool missing",
    ),
    pytest.mark.skipif(
        bool(_MISSING), reason=f"manual examples need {', '.join(_MISSING)}"
    ),
]


def _load_tool() -> ModuleType:
    spec = importlib.util.spec_from_file_location("_run_docs_examples", TOOL)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _squeeze(text: str) -> str:
    """Collapse runs of spaces and drop trailing blanks on every line."""
    return "\n".join(
        re.sub(r"[ \t]+", " ", line).strip() for line in text.splitlines()
    ).strip()


@pytest.mark.parametrize("manual", MANUALS, ids=lambda p: p.parent.name or "en")
def test_manual_examples_run_and_match_their_recorded_output(manual: Path):
    tool = _load_tool()

    text = manual.read_text(encoding="utf-8")
    examples = tool._scan(text.splitlines(keepends=True))
    assert examples, f"no executable examples found in {manual.name}"

    produced = tool._run(examples)

    mismatches = []
    for ex, got in zip(examples, produced, strict=True):
        if ex.out_span is None:
            mismatches.append(f"line {ex.first_line}: no output block")
        elif _squeeze(ex.output) != _squeeze(got):
            mismatches.append(
                f"line {ex.first_line}: recorded output is stale\n"
                f"--- recorded ---\n{ex.output}--- actual ---\n{got}"
            )

    assert not mismatches, (
        "\n\n".join(mismatches)
        + "\n\nRegenerate with:\n"
        f"    python tools/run_docs_examples.py --update {manual.as_posix()}"
    )
