"""Guide figures: a referenced file has to exist, and look like a picture.

The specification curve shipped with twelve passing tests while drawing a single
visible dot: the axes were wrong, and no assertion in the suite was looking at
the image. The committed figures are therefore checked as *files* — that the
guide points at one, that the file is a real PNG of a plausible size — and the
pixel-level check is ``python tools/make_figures.py --check``, run deliberately
rather than in CI, because byte equality across matplotlib versions is not
something a dependency update should be able to fail.
"""

from __future__ import annotations

import importlib.util
import re
import struct
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
DOCS = ROOT / "docs"
IMAGES = DOCS / "guide" / "images"
TOOL = ROOT / "tools" / "make_figures.py"

#: Markdown image references, e.g. ``![the curve](images/curve.png)``.
_IMAGE = re.compile(r"!\[[^\]]*\]\((?P<path>[^)\s]+)\)")

#: A 4x4 inch figure at 100 dpi is a few kilobytes; anything this small is an
#: empty canvas that rendered without raising, which is the failure mode that
#: matters.
MIN_BYTES = 8000


def _references() -> list[tuple[Path, Path]]:
    found = []
    for page in sorted(DOCS.glob("guide/*.md")):
        for match in _IMAGE.finditer(page.read_text(encoding="utf-8")):
            target = (page.parent / match.group("path"))
            found.append((page, target))
    return found


def _png_size(data: bytes) -> tuple[int, int]:
    assert data[:8] == b"\x89PNG\r\n\x1a\n", "not a PNG"
    assert data[12:16] == b"IHDR", "no IHDR chunk"
    return struct.unpack(">II", data[16:24])


def test_the_guide_references_at_least_one_figure():
    # If this stops being true the rest of the file silently checks nothing.
    assert _references()


@pytest.mark.parametrize("page,target", _references(),
                         ids=[str(t.name) for _, t in _references()])
def test_every_referenced_figure_exists_and_has_content(page, target):
    assert target.exists(), f"{page.name} references a missing figure: {target}"
    data = target.read_bytes()
    assert len(data) >= MIN_BYTES, (
        f"{target.name} is {len(data)} bytes; a rendered figure that small is "
        "almost certainly an empty canvas")
    width, height = _png_size(data)
    assert width >= 400 and height >= 300, (
        f"{target.name} is {width}x{height}; the guide figures are rendered at "
        "a readable size")


def test_the_figure_tool_writes_where_the_guide_reads():
    # The tool and the guide have to agree on the directory, or a regenerated
    # figure lands somewhere nothing links to. `tools` is not an installed
    # package, so the module is loaded from its file.
    spec = importlib.util.spec_from_file_location("make_figures", TOOL)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    assert Path(module.IMAGES).resolve() == IMAGES.resolve()
    referenced = {t.name for _, t in _references()}
    assert referenced <= set(module.FIGURES), (
        f"referenced but never generated: {referenced - set(module.FIGURES)}")
