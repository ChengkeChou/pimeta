"""Guards for the documentation: coverage, links and API-name accuracy.

These tests keep the docs honest without requiring a documentation build:

* every public name has a docstring,
* the generated API reference covers every name in ``pimeta.__all__``,
* every relative Markdown link points at a file that exists,
* every ``ma.<name>`` referenced in the docs resolves to a real attribute,
* the front-door snippets in ``index.md`` / ``quickstart.md`` actually run,
* no file carries a U+FFFD replacement character, no Sphinx role syntax
  (``:class:``, ``:func:``, ...) leaks into the API reference, and every
  "Mirrors R's ..." name is a plain identifier.
"""

from __future__ import annotations

import ast
import inspect
import re
from pathlib import Path

import pytest

import pimeta as ma

ROOT = Path(__file__).resolve().parents[1]
DOCS = ROOT / "docs"

#: Names that are data, not callables, and legitimately have no docstring.
_DATA_ONLY = {"RATIO_TYPES", "DIFF_TYPES", "REVIEW_COLUMNS",
              "settings_meta", "settings_netmeta",
              "META_DEFAULTS", "NETMETA_DEFAULTS"}


def _md_files() -> list[Path]:
    return sorted(p for p in DOCS.rglob("*.md") if p.is_file())


def _python_blocks(path: Path) -> list[str]:
    blocks, current, inside = [], [], False
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip().startswith("```python"):
            inside, current = True, []
            continue
        if inside and line.strip().startswith("```"):
            blocks.append("\n".join(current))
            inside = False
            continue
        if inside:
            current.append(line)
    return blocks


# --------------------------------------------------------------------------- #
# Docstring coverage
# --------------------------------------------------------------------------- #
def test_every_public_callable_has_a_docstring():
    missing = []
    for name in ma.__all__:
        if name in _DATA_ONLY:
            continue
        obj = getattr(ma, name, None)
        if obj is None:
            missing.append(f"{name} (not exported)")
            continue
        if not (inspect.isfunction(obj) or inspect.isclass(obj)):
            continue
        if not (inspect.getdoc(obj) or "").strip():
            missing.append(name)
    assert not missing, f"missing docstrings: {missing}"


def test_every_public_name_is_importable():
    missing = [n for n in ma.__all__ if not hasattr(ma, n)]
    assert not missing, f"names in __all__ but not exported: {missing}"


# --------------------------------------------------------------------------- #
# Generated API reference
# --------------------------------------------------------------------------- #
def test_api_docs_are_checked_in():
    api = DOCS / "api"
    assert api.is_dir(), "run `python tools/gen_api_docs.py`"
    pages = sorted(p.name for p in api.glob("*.md"))
    assert "index.md" in pages
    # One page per documented topic.
    assert len(pages) >= 10, pages


def test_api_index_covers_every_public_name():
    text = (DOCS / "api" / "index.md").read_text(encoding="utf-8")
    missing = [n for n in ma.__all__ if f"[`{n}`]" not in text]
    assert not missing, f"not in docs/api/index.md: {missing[:20]}"


def test_r_method_map_is_generated():
    text = (DOCS / "reference" / "r-method-map.md").read_text(encoding="utf-8")
    missing = [k for k in ma.r_methods() if f"`{k}`" not in text]
    assert not missing, f"missing from the R map: {missing[:20]}"


# --------------------------------------------------------------------------- #
# Links
# --------------------------------------------------------------------------- #
_LINK = re.compile(r"\[[^\]]*\]\(([^)]+)\)")


def test_relative_markdown_links_resolve():
    broken = []
    for path in _md_files():
        for target in _LINK.findall(path.read_text(encoding="utf-8")):
            target = target.split("#", 1)[0].strip()
            if not target or target.startswith(("http://", "https://", "mailto:")):
                continue
            if not (path.parent / target).exists():
                broken.append(f"{path.relative_to(ROOT)} -> {target}")
    assert not broken, "broken links:\n" + "\n".join(broken)


def test_mkdocs_nav_files_exist():
    config = (ROOT / "mkdocs.yml").read_text(encoding="utf-8")
    refs = re.findall(r":\s*([\w./-]+\.md)\s*$", config, flags=re.MULTILINE)
    missing = [r for r in refs if not (DOCS / r).exists()]
    assert not missing, f"nav points at missing files: {missing}"


# --------------------------------------------------------------------------- #
# Names used in prose
# --------------------------------------------------------------------------- #
#: ``ma.<name>`` in prose — but not the ``level.ma.common`` settings keys.
_MA_ATTR = re.compile(r"(?<![\w.])ma\.([A-Za-z_][A-Za-z0-9_]*)")
#: ``interop.<name>`` in prose or an import; ``interop.md`` is a link, not a name.
_INTEROP_ATTR = re.compile(r"\binterop\.([A-Za-z_]\w*)\b")
_INTEROP_IMPORT = re.compile(r"from pimeta\.interop import\s+([^\n]+)")
_INTEROP_SKIP = {"md"}


def test_names_referenced_in_docs_exist():
    bad = set()
    for path in _md_files():
        text = path.read_text(encoding="utf-8")
        for name in _MA_ATTR.findall(text):
            if not hasattr(ma, name):
                bad.add(f"{path.relative_to(ROOT)}: ma.{name}")
    assert not bad, "unknown names in docs:\n" + "\n".join(sorted(bad))


def test_interop_names_referenced_in_docs_exist():
    import pimeta.interop as interop

    bad = set()
    for path in _md_files():
        text = path.read_text(encoding="utf-8")
        names = set(_INTEROP_ATTR.findall(text)) - _INTEROP_SKIP
        for line in _INTEROP_IMPORT.findall(text):
            names.update(part.strip().split(" as ")[0]
                         for part in line.split(","))
        for name in names:
            if name and not hasattr(interop, name):
                bad.add(f"{path.relative_to(ROOT)}: interop.{name}")
    assert not bad, "unknown interop names in docs:\n" + "\n".join(sorted(bad))


# --------------------------------------------------------------------------- #
# Front-door snippets must run
# --------------------------------------------------------------------------- #
def test_front_door_snippets_parse():
    for name in ("index.md", "quickstart.md"):
        for block in _python_blocks(DOCS / name):
            ast.parse(block)          # raises SyntaxError on a broken snippet


@pytest.mark.parametrize("name", ["index.md", "quickstart.md"])
def test_front_door_snippets_run(name, tmp_path, monkeypatch):
    """Execute the self-contained snippets from the landing page and quickstart."""
    monkeypatch.chdir(tmp_path)
    for index, block in enumerate(_python_blocks(DOCS / name)):
        namespace: dict = {}
        try:
            exec(compile(block, f"{name}#block{index}", "exec"), namespace)
        except Exception as exc:                     # pragma: no cover - failure path
            pytest.fail(f"{name} block {index} failed: {type(exc).__name__}: {exc}\n\n{block}")

# --------------------------------------------------------------------------- #
# Symbol integrity: no mangled characters, no leaked Sphinx roles
# --------------------------------------------------------------------------- #
_TEXTLIKE = {".md", ".py", ".toml", ".yml", ".yaml", ".ipynb", ".txt", ".cfg"}
_SKIP_DIRS = {".git", "build", "dist", "__pycache__", ".mypy_cache",
              ".pytest_cache", ".ruff_cache"}


def _text_files() -> list[Path]:
    """Every text file a reader can end up looking at."""
    roots = [ROOT / "src", ROOT / "docs", ROOT / "tests", ROOT / "tools",
             ROOT / "examples"]
    files = [ROOT / "README.md", ROOT / "CHANGELOG.md"]
    for root in roots:
        if not root.exists():
            continue
        for path in root.rglob("*"):
            if (path.is_file() and path.suffix in _TEXTLIKE
                    and not _SKIP_DIRS & set(path.parts)):
                files.append(path)
    return sorted(set(files))


def test_no_replacement_characters():
    """U+FFFD means a file was decoded with the wrong encoding at some point.

    It shows up as a stray question mark inside a sentence: one docstring in
    ``measures.py`` had fourteen of them where an em dash belonged, and the API
    reference faithfully copied every one.
    """
    bad = []
    for path in _text_files():
        for i, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            if "\ufffd" in line:
                bad.append(f"{path.relative_to(ROOT)}:{i}: {line.strip()[:80]}")
    assert not bad, "replacement characters (U+FFFD) found:\n" + "\n".join(bad)


def test_generated_pages_have_no_sphinx_roles():
    """Docstrings use ``:class:`X```; the reader must see just ``X``."""
    pattern = re.compile(r":(?:class|func|meth|mod|attr|data|exc|obj|ref):`")
    bad = [
        f"{path.relative_to(ROOT)}:{i}: {line.strip()[:80]}"
        for path in sorted((DOCS / "api").glob("*.md"))
        for i, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1)
        if pattern.search(line)
    ]
    assert not bad, "Sphinx roles leaked into the API reference:\n" + "\n".join(bad)


def test_mirror_names_are_plain_identifiers():
    """``*Mirrors R's `...`.*`` must never start with a colon.

    ``meta:::ci`` (R's internal accessor) used to be split on ``::`` and come
    out as ``:ci``.
    """
    pattern = re.compile(r"Mirrors R's `([^`]*)`")
    bad = []
    for path in sorted((DOCS / "api").glob("*.md")):
        for i, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            for match in pattern.finditer(line):
                name = match.group(1)
                if not name or not (name[0].isalpha() or name[0] in "[$._"):
                    bad.append(f"{path.relative_to(ROOT)}:{i}: {name!r}")
    assert not bad, "malformed mirror names:\n" + "\n".join(bad)


# --------------------------------------------------------------------------- #
# no comma-separated runs of inline code
# --------------------------------------------------------------------------- #
_SPAN = re.compile(r"`[^`\n]+`")
_PUNCT = set(" \t,;、，；|/()[]{}<>+-–—:.·*")


def _is_enumeration(line: str) -> bool:
    """Is this line nothing but three or more code spans glued by punctuation?

    ``\\`RR\\`, \\`OR\\`, \\`RD\\`, ...`` is the shape that breaks a renderer which
    treats ``<code>`` as an independent inline box: the commas stay put and the
    spans drift to the end of the paragraph, so the README reads as a run of
    commas followed by a pile of identifiers.  The question is therefore what
    is *left* once the spans are removed -- only punctuation means the line was
    an enumeration.  Prose that merely mentions a few identifiers keeps its
    words and is fine.
    """
    if len(_SPAN.findall(line)) < 3:
        return False
    rest = _SPAN.sub("", line)
    return len([ch for ch in rest if ch not in _PUNCT]) <= 3


def test_no_comma_separated_runs_of_inline_code():
    """Enumerations must be fenced blocks or tables, not inline-code chains."""
    targets = [ROOT / "README.md", *sorted(DOCS.rglob("*.md"))]
    bad = []
    for path in targets:
        if "changelog" in path.name or path.parent.name == "api":
            continue  # release history and generated pages keep the old shape
        for i, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            if line.lstrip().startswith(("```", "|")):
                continue
            if _is_enumeration(line):
                bad.append(f"{path.relative_to(ROOT)}:{i}: {line.strip()[:90]}")
    assert not bad, (
        "inline-code enumerations reorder badly in some renderers; use a "
        "```text block or a table instead:\n" + "\n".join(bad)
    )

