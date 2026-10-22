"""Generate the Markdown API reference from pimeta's docstrings.

Run from the repository root::

    python tools/gen_api_docs.py

The script walks ``pimeta.__all__``, groups the names by topic, and writes one
page per topic into ``docs/api/`` plus an index.  Everything is plain Markdown:
no mkdocstrings, no Sphinx directives, no build step required.  That means the
reference renders identically on GitHub, in VS Code's Markdown preview and under
MkDocs, and cannot drift from the code because it is regenerated from the
objects themselves.

Two companion files are produced as well:

``docs/reference/r-method-map.md``
    The ``R S3 method name -> pimeta callable`` table from ``r_methods()``.
``docs/api/index.md``
    A searchable list of every public name with a link to its section.
"""

from __future__ import annotations

import inspect
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
OUT = ROOT / "docs"
sys.path.insert(0, str(SRC))

import pimeta  # noqa: E402  (path manipulation must come first)

#: Module name -> (topic slug, topic title).  Anything not listed lands in
#: ``misc``.  Keeping this explicit means the reference never silently loses a
#: module when new ones are added.
MODULE_TOPICS: dict[str, tuple[str, str]] = {
    # effect sizes and measures
    "escalc": ("effect-sizes", "Effect sizes"),
    "transforms": ("effect-sizes", "Effect sizes"),
    "measures": ("effect-sizes", "Effect sizes"),
    "conv": ("effect-sizes", "Effect sizes"),
    "fmt": ("effect-sizes", "Effect sizes"),
    # models
    "pooling": ("models", "Models"),
    "tau2": ("models", "Models"),
    "rma_uni": ("models", "Models"),
    "rma_mv": ("models", "Models"),
    "rve": ("models", "Models"),
    "dta": ("models", "Models"),
    "rma_glmm": ("models", "Models"),
    "rma_ls": ("models", "Models"),
    "rma_peto_mh": ("models", "Models"),
    "selmodel": ("models", "Models"),
    "basis": ("models", "Models"),
    # high-level drivers
    "metagen": ("drivers", "High-level drivers"),
    "metabin": ("drivers", "High-level drivers"),
    "metacont": ("drivers", "High-level drivers"),
    "metacor": ("drivers", "High-level drivers"),
    "metainc": ("drivers", "High-level drivers"),
    "metaprop": ("drivers", "High-level drivers"),
    "metarate": ("drivers", "High-level drivers"),
    "metamean": ("drivers", "High-level drivers"),
    "metareg": ("drivers", "High-level drivers"),
    "meta_compose": ("drivers", "High-level drivers"),
    # network meta-analysis
    "netmeta": ("network", "Network meta-analysis"),
    "nma": ("network", "Network meta-analysis"),
    "nma.binary": ("network", "Network meta-analysis"),
    "nma.complexity": ("network", "Network meta-analysis"),
    "nma.component": ("network", "Network meta-analysis"),
    "nma.contrib": ("network", "Network meta-analysis"),
    "nma.convert": ("network", "Network meta-analysis"),
    "nma.core": ("network", "Network meta-analysis"),
    "nma.graph": ("network", "Network meta-analysis"),
    "nma.impact": ("network", "Network meta-analysis"),
    "nma.labels": ("network", "Network meta-analysis"),
    "nma.league": ("network", "Network meta-analysis"),
    "nma.matrix": ("network", "Network meta-analysis"),
    "nma.measures": ("network", "Network meta-analysis"),
    "nma.pairwise": ("network", "Network meta-analysis"),
    "nma.pairwise_table": ("network", "Network meta-analysis"),
    "nma.paths": ("network", "Network meta-analysis"),
    "nma.ranking": ("network", "Network meta-analysis"),
    "nma.splitting": ("network", "Network meta-analysis"),
    "nma.tables": ("network", "Network meta-analysis"),
    # publication bias / sensitivity
    "publication_bias": ("bias", "Publication bias"),
    "fsn": ("bias", "Publication bias"),
    "gosh": ("bias", "Publication bias"),
    "sensitivity": ("bias", "Publication bias"),
    "cumul": ("bias", "Publication bias"),
    "metainf": ("bias", "Publication bias"),
    # diagnostics and helpers
    "diagnostics": ("diagnostics", "Diagnostics and utilities"),
    "mv_diagnostics": ("diagnostics", "Diagnostics and utilities"),
    "missing": ("diagnostics", "Diagnostics and utilities"),
    "extras": ("diagnostics", "Diagnostics and utilities"),
    "utils": ("diagnostics", "Diagnostics and utilities"),
    "_firth": ("diagnostics", "Diagnostics and utilities"),
    "_numeric": ("diagnostics", "Diagnostics and utilities"),
    # plotting
    "plotting": ("plotting", "Figures"),
    # IO and ecosystem
    "readers": ("io", "Input, output and interoperability"),
    "data_compat": ("io", "Input, output and interoperability"),
    "interop": ("io", "Input, output and interoperability"),
    "interop.base": ("io", "Input, output and interoperability"),
    "interop.broom": ("io", "Input, output and interoperability"),
    "interop.estimator": ("io", "Input, output and interoperability"),
    "interop.tables": ("io", "Input, output and interoperability"),
    "interop.statsmodels": ("io", "Input, output and interoperability"),
    "interop.xarray_adapter": ("io", "Input, output and interoperability"),
    "interop.networkx_adapter": ("io", "Input, output and interoperability"),
    "record": ("io", "Input, output and interoperability"),
    "record.jsonio": ("io", "Input, output and interoperability"),
    "record.fingerprint": ("io", "Input, output and interoperability"),
    "record.table": ("io", "Input, output and interoperability"),
    "record.schema": ("io", "Input, output and interoperability"),
    "record.store": ("io", "Input, output and interoperability"),
    # S3 generics and reporting
    "s3": ("s3", "S3-style generics and reporting"),
    "rma_list": ("s3", "S3-style generics and reporting"),
    "report": ("s3", "S3-style generics and reporting"),
    "settings": ("s3", "S3-style generics and reporting"),
}

TOPIC_ORDER = (
    ("effect-sizes", "Effect sizes"),
    ("models", "Models"),
    ("drivers", "High-level drivers"),
    ("network", "Network meta-analysis"),
    ("bias", "Publication bias"),
    ("diagnostics", "Diagnostics and utilities"),
    ("plotting", "Figures"),
    ("io", "Input, output and interoperability"),
    ("s3", "S3-style generics and reporting"),
    ("misc", "Other"),
)

#: Data objects (sets, dicts, submodules) have no meaningful ``__module__``, so
#: they are routed by name instead.
NAME_TOPICS: dict[str, str] = {
    "RATIO_TYPES": "effect-sizes",
    "DIFF_TYPES": "effect-sizes",
    "REVIEW_COLUMNS": "io",
    "interop": "io",
    "record": "io",
    "RECORD_SCHEMA": "io",
    "STORE_SCHEMA": "io",
    "settings_meta": "s3",
    "settings_netmeta": "s3",
    "META_DEFAULTS": "s3",
    "NETMETA_DEFAULTS": "s3",
}

_SECTION = re.compile(
    r"^(Args|Arguments|Parameters|Returns|Return|Raises|Attributes|Yields|"
    r"Note|Notes|Example|Examples):\s*$")
_ITEM = re.compile(r"^(\S[^:]*?):\s*(.*)$")

#: A docstring's first line usually names the R function it mirrors, either as
#: ``metafor::rma.mv``, as a dotted name (``confint.rma.uni``) or as a leading
#: code span (``rma()``).  Bare prose code spans such as ``delta`` are ignored —
#: guessing on those produced more wrong labels than right ones.
_SPAN = re.compile(r"``([^`]+)``")
_R_IDENT = re.compile(r"^(?:[A-Za-z_][\w.]*|\[[^\]]*\]|\$[^ ]*)$")

#: Sphinx cross-reference roles that must not reach the Markdown output.
_RST_ROLE = re.compile(
    r":(?:class|func|meth|mod|attr|data|exc|obj|ref|py:class|py:func):`([^`]+)`")


# --------------------------------------------------------------------------- #
# docstring -> Markdown
# --------------------------------------------------------------------------- #
def _indent(text: str) -> int:
    for line in text.splitlines():
        if line.strip():
            return len(line) - len(line.lstrip())
    return 0


def _clean(lines: list[str]) -> list[str]:
    """Strip the common indentation and trailing blank lines.

    A line is never cut by more than its own indentation: a dedented paragraph
    that slipped into the block must keep its first word.
    """
    body = "\n".join(lines)
    pad = _indent(body)
    out: list[str] = []
    for line in body.splitlines():
        if not line.strip():
            out.append("")
        else:
            cut = min(pad, len(line) - len(line.lstrip()))
            out.append(line[cut:])
    while out and not out[-1].strip():
        out.pop()
    return out


def _render_section(name: str, body: list[str]) -> list[str]:
    """Render one ``Args:``-style block."""
    lines = _clean(body)
    if not lines:
        return []

    # After _clean() an item starts at column 0; continuation lines stay indented.
    items: list[tuple[str, list[str]]] = []
    for line in lines:
        if not line.strip():
            continue
        if not line.startswith(" "):
            match = _ITEM.match(line)
            if match:
                items.append((match.group(1).strip(), [match.group(2).strip()]))
                continue
            items.append(("", [line.strip()]))
        elif items:
            items[-1][1].append(line.strip())
        else:
            items.append(("", [line.strip()]))

    named = [term for term, _ in items if term]
    out = [f"**{name}**", ""]
    if not named:
        return out + [" ".join(line for line in lines if line.strip()), ""]
    for term, desc in items:
        text = " ".join(part for part in desc if part)
        if term:
            out.append(f"- `{term}` — {text}" if text else f"- `{term}`")
        elif text:
            out.append(text)
    out.append("")
    return out


def _strip_rst_roles(text: str) -> str:
    """Rewrite Sphinx roles as plain inline code: ``:func:`x``` -> ``` `x` ```.

    Docstrings are written in Sphinx style (``:class:`~pimeta.RmaResult```),
    which GitHub renders literally as a stray colon and backticks.  The ``~``
    shorthand and the ``pimeta.`` prefix are dropped so the reader sees the
    bare name.
    """
    def repl(match: re.Match[str]) -> str:
        target = match.group(1).lstrip("~")
        for prefix in ("pimeta.", "pimeta.interop.", "pimeta.nma."):
            if target.startswith(prefix):
                target = target[len(prefix):]
                break
        return f"`{target}`"

    return _RST_ROLE.sub(repl, text)


def doc_to_markdown(doc: str | None) -> str:
    """Convert a pimeta docstring (Google style) to Markdown."""
    if not doc:
        return ""
    lines = _strip_rst_roles(inspect.cleandoc(doc)).splitlines()
    out: list[str] = []
    i = 0
    while i < len(lines):
        line = lines[i]
        match = _SECTION.match(line.strip())
        if match:
            section = match.group(1)
            body: list[str] = []
            body_pad: int | None = None
            i += 1
            while i < len(lines):
                nxt = lines[i]
                if _SECTION.match(nxt.strip()):
                    break                       # the next field list
                if nxt.strip():
                    pad = len(nxt) - len(nxt.lstrip())
                    if body_pad is None:
                        body_pad = pad          # items line up with the first
                    elif pad < body_pad:
                        break                   # dedented: back to prose
                body.append(nxt)
                i += 1
            out.extend(_render_section(section, body))
            continue
        out.append(line)
        i += 1
    text = "\n".join(out)
    # Collapse runs of blank lines.
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def _signature(obj) -> str:
    try:
        sig = inspect.signature(obj)
    except (TypeError, ValueError):                   # pragma: no cover - builtins
        return ""
    name = getattr(obj, "__name__", "")
    return f"{name}{sig}"


def _is_callable(obj) -> bool:
    return inspect.isfunction(obj) or inspect.isclass(obj)


def render_value(obj) -> str:
    """Describe a public constant or submodule without leaking a builtin docstring."""
    if inspect.ismodule(obj):
        return (f"*Subpackage `{obj.__name__}` — see the "
                "[interoperability guide](../guide/interop.md).*")
    kind = type(obj).__name__
    if isinstance(obj, (set, frozenset, list, tuple)) and obj and all(
            isinstance(v, str) for v in obj):
        preview = ", ".join(f"`{v}`" for v in sorted(obj))
        return f"*Constant (`{kind}`):* {preview}"
    if isinstance(obj, dict) and obj:
        keys = ", ".join(f"`{k}`" for k in list(obj)[:10])
        more = " …" if len(obj) > 10 else ""
        return f"*Mapping (`{kind}`) with {len(obj)} keys:* {keys}{more}"
    return f"*Constant of type `{kind}`.*"


def _r_name(obj) -> str | None:
    """The R function a callable mirrors, or ``None`` when it does not say."""
    doc = inspect.getdoc(obj) or ""
    first = doc.splitlines()[0].strip() if doc else ""
    spans = [s.strip() for s in _SPAN.findall(first)]

    for raw in spans:                       # metafor::rma.mv is unambiguous
        if ":::" in raw:                    # meta:::ci is R's internal form
            return re.sub(r"\(\)$", "", raw.split(":::")[-1].lstrip(":"))
        if "::" in raw:
            return re.sub(r"\(\)$", "", raw.split("::")[-1].lstrip(":"))
    for raw in spans:                       # dotted names are R functions
        candidate = re.sub(r"\(\)$", "", raw)
        if "." in candidate and _R_IDENT.match(candidate):
            return candidate
    if "mirror" in first.lower() or first.startswith("``"):
        for raw in spans:
            candidate = re.sub(r"\(\)$", "", raw)
            if _R_IDENT.match(candidate):
                return candidate
    return None


# --------------------------------------------------------------------------- #
# page assembly
# --------------------------------------------------------------------------- #
def build_pages() -> dict[Path, str]:
    topics: dict[str, list[tuple[str, object]]] = {slug: [] for slug, _ in TOPIC_ORDER}
    for name in pimeta.__all__:
        obj = getattr(pimeta, name, None)
        if obj is None:
            continue
        module = getattr(obj, "__module__", "") or ""
        module = module.removeprefix("pimeta.")
        slug = NAME_TOPICS.get(name) or MODULE_TOPICS.get(module, ("misc", "Other"))[0]
        topics[slug].append((name, obj))

    pages: dict[Path, str] = {}
    index_rows: list[str] = []

    for slug, title in TOPIC_ORDER:
        entries = topics.get(slug) or []
        if not entries:
            continue
        head = [
            f"# {title}",
            "",
            f"*{len(entries)} public names, generated from `pimeta.__all__` and the "
            "docstrings in `src/pimeta/`. Do not edit this page by hand — run "
            "`python tools/gen_api_docs.py`.*",
            "",
            "| Name | Mirrors in R |",
            "|---|---|",
        ]
        body: list[str] = []
        for name, obj in entries:
            head.append(f"| [`{name}`](#{name.lower()}) | {_r_name(obj) or '—'} |")
            body.extend([f"## `{name}`", ""])
            if _is_callable(obj):
                sig = _signature(obj)
                if sig:
                    body.extend(["```python", sig, "```", ""])
                mirrored = _r_name(obj)
                if mirrored:
                    body.extend([f"*Mirrors R's `{mirrored}`.*", ""])
                text = doc_to_markdown(inspect.getdoc(obj))
                if text:
                    body.extend([text, ""])
            else:
                body.extend([render_value(obj), ""])
        pages[OUT / "api" / f"{slug}.md"] = "\n".join(head + [""] + body).rstrip() + "\n"

        for name, obj in entries:
            index_rows.append(f"| [`{name}`]({slug}.md#{name.lower()}) | "
                              f"{title} | {_r_name(obj) or '—'} |")

    pages[OUT / "api" / "index.md"] = "\n".join([
        "# API reference",
        "",
        f"All {sum(len(v) for v in topics.values())} public names in `pimeta.__all__`, "
        "generated from the source docstrings.",
        "",
        "| Name | Section | Mirrors in R |",
        "|---|---|---|",
        *index_rows,
        "",
    ])

    # The R S3 method map.
    methods = pimeta.r_methods()
    rows = "\n".join(f"| `{r}` | [`{py}`](../api/index.md#{py.lower()}) |"
                     for r, py in sorted(methods.items()))
    pages[OUT / "reference" / "r-method-map.md"] = "\n".join([
        "# R method-name map",
        "",
        "R writes its methods as `generic.class` (`print.rma.uni`,",
        "`cooks.distance.rma.mv`, `[.list.rma`).  Those strings are not valid",
        "Python identifiers, so pimeta exposes R-independent names and bridges the",
        "two with `pimeta.r_methods()` plus a module-level `__getattr__`:",
        "",
        "```python",
        'import pimeta as ma',
        'getattr(ma, "cooks.distance.rma.mv")   # -> ma.cooks_distance',
        'getattr(ma, "[.list.rma")              # -> ma.slice_rma',
        "```",
        "",
        f"{len(methods)} entries:",
        "",
        "| R | pimeta |",
        "|---|---|",
        rows,
        "",
    ])
    # The changelog is part of the documentation site; copy it so the nav entry
    # always resolves and the two never drift apart.
    changelog = ROOT / "CHANGELOG.md"
    if changelog.exists():
        pages[OUT / "changelog.md"] = changelog.read_text(encoding="utf-8")

    return pages


def main() -> int:
    pages = build_pages()
    for path, text in pages.items():
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8", newline="\n")

    # ``docs/api/`` is entirely generated.  Drop pages that this run did not
    # produce, otherwise a topic that loses its last name (or gets renamed)
    # leaves a stale copy behind that still shows up in a site search.
    generated = set(pages)
    for stale in sorted((OUT / "api").glob("*.md")):
        if stale not in generated:
            stale.unlink()
            print(f"removed stale page {stale.relative_to(ROOT)}")

    total = len(pimeta.__all__)
    print(f"wrote {len(pages)} pages covering {total} public names")
    for path in sorted(pages):
        print(f"  {path.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
