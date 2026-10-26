"""Write ``tests/data/MANIFEST.json``: what every fixture is and where it came from.

Why a manifest at all
---------------------
The test fixtures are load-bearing: a golden value computed from ``robma.csv``
is a claim about the algorithm only if the bytes that produced it are fixed. A
file that can change silently turns a regression test into a coin toss, so the
manifest records a SHA-256 and a byte count for every fixture and
``tests/test_data_manifest.py`` refuses to run a suite where they disagree.

Why the provenance fields are filled in the way they are
-------------------------------------------------------
Three different things live in ``tests/data`` and they carry different
obligations:

``generated``
    A committed script produces the file. Re-running that script has to
    reproduce the bytes, so the fixture is verifiable by anyone.
``r-reference``
    Dumped from an R session by a committed script, with the tool versions
    recorded. The numbers are a reference implementation's output, which is the
    only thing that can licence a golden value.
``upstream``
    A dataset redistributed from another project. The licence is that project's,
    not pimeta's, and this tool records where it comes from rather than
    asserting a licence it cannot verify.
``unverified``
    No producing script and no recorded source. These are honest gaps, listed
    here rather than left invisible: the alternative is a manifest that claims
    more than it knows, which is worse than no manifest.

The set of ``unverified`` files is written out in :data:`UNVERIFIED` and pinned
by a test. Adding a fixture without provenance therefore fails the suite, and so
does quietly moving one out of the unverified list without recording a source.

Usage::

    python tools/make_data_manifest.py           # rewrite the manifest
    python tools/make_data_manifest.py --check    # exit 1 when it is stale
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "tests" / "data"
MANIFEST = DATA / "MANIFEST.json"

SCHEMA = 1

#: The project's own licence, for files this repository generates.
OWN_LICENCE = "MIT"

#: Which fixture came from where. A file with no entry is recorded as
#: ``unverified`` rather than guessed at.
ORIGIN: dict[str, dict[str, Any]] = {
    "dynhet.csv": {
        "origin": "generated",
        "produced_by": "tools/make_dynhet_fixture.py",
        "licence": OWN_LICENCE,
        "note": "simulated time-varying heterogeneity; the generating script "
                "fixes the seed, so the bytes are reproducible",
    },
    "nma_bayes.csv": {
        "origin": "generated",
        "produced_by": "tools/make_nma_bayes_fixture.py",
        "licence": OWN_LICENCE,
        "note": "simulated arm-level network for the Bayesian NMA; checked "
                "against tools/quadrature_nma_bayes.py and multinma",
    },
    "robma.csv": {
        "origin": "generated",
        "produced_by": "tools/make_robma_fixture.py",
        "licence": OWN_LICENCE,
        "note": "simulated selection-model fixture used by the RoBMA and "
                "diagnostics tests",
    },
    "_r_senn2013.tsv": {
        "origin": "r-reference",
        "produced_by": "tools/r_reference/netmeta_senn2013.R",
        "licence": OWN_LICENCE,
        "note": "output of the reference implementation, whose versions are "
                "recorded in the file's own header",
    },
    "_konstantopoulos2011.csv": {
        "origin": "upstream",
        "source": "metafor::dat.konstantopoulos2011",
        "licence": "upstream",
        "note": "a published dataset bundled with metafor; the column set "
                "matches that dataset exactly. Redistribution is governed by "
                "metafor's licence and the original publication, not by this "
                "repository",
    },
    "_senn2013_comparison.csv": {
        "origin": "upstream",
        "source": "netmeta::senn2013",
        "licence": "upstream",
        "note": "the Senn 2013 network in contrast format, as distributed with "
                "netmeta",
    },
}

#: Fixtures whose origin is not recorded. Pinned by a test so the list can only
#: shrink by recording a source, and can only grow by someone editing this file.
UNVERIFIED = (
    "ipd_nma.csv",
    "nma_dose.csv",
    "nma_timecourse.csv",
    "onestage_slopes.csv",
    "prediction_joint.csv",
    "timecourse.csv",
    "tvhr_piecewise.csv",
    "_nc_common_ref.csv",
    "_nc_random_ref.csv",
)

_ORIGINS = ("generated", "r-reference", "upstream", "unverified")


def digest(path: Path) -> dict[str, Any]:
    data = path.read_bytes()
    return {"sha256": hashlib.sha256(data).hexdigest(), "bytes": len(data)}


def build() -> dict[str, Any]:
    files: dict[str, Any] = {}
    for path in sorted(DATA.iterdir()):
        if not path.is_file() or path.name == MANIFEST.name:
            continue
        entry = {"name": path.name, **digest(path)}
        known = ORIGIN.get(path.name)
        if known is None:
            entry.update({
                "origin": "unverified",
                "licence": "unverified",
                "note": "no producing script and no recorded source; confirm "
                        "before relying on this file as a reference",
            })
        else:
            entry.update(known)
        if entry["origin"] not in _ORIGINS:
            raise SystemExit(f"{path.name}: unknown origin {entry['origin']!r}")
        files[path.name] = entry
    return {
        "schema": SCHEMA,
        "note": "Generated by tools/make_data_manifest.py. Every fixture under "
                "tests/data is listed with its SHA-256, its provenance and the "
                "licence that governs it. 'unverified' means exactly that: no "
                "source is recorded in this repository, and the entry says so "
                "rather than asserting one.",
        "count": len(files),
        "unverified": sorted(n for n, e in files.items()
                             if e["origin"] == "unverified"),
        "files": files,
    }


def render(payload: dict[str, Any]) -> str:
    return json.dumps(payload, indent=2, sort_keys=True) + "\n"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--check", action="store_true",
                        help="do not write; exit 1 if the manifest is stale")
    args = parser.parse_args(argv)

    text = render(build())
    if args.check:
        current = MANIFEST.read_text(encoding="utf-8") if MANIFEST.exists() else ""
        if current != text:
            print("tests/data/MANIFEST.json is stale; run "
                  "`python tools/make_data_manifest.py`")
            return 1
        return 0
    MANIFEST.write_text(text, encoding="utf-8")
    payload = json.loads(text)
    print(f"wrote {MANIFEST.relative_to(ROOT)}: {payload['count']} files, "
          f"{len(payload['unverified'])} unverified")
    return 0


if __name__ == "__main__":
    sys.exit(main())
