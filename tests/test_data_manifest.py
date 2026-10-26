"""The fixture manifest: integrity and provenance of everything in tests/data.

Two separate claims are checked here, and they fail for different reasons:

* **Integrity.** Every fixture's bytes are pinned by SHA-256. A golden value in a
  test is only evidence while the input that produced it is fixed; a fixture that
  can change silently makes the test a coin toss.
* **Provenance.** Every fixture names where it came from, or says that this
  repository does not know. The ``unverified`` list is pinned, so it can only
  shrink by recording a real source — never by someone deciding the question is
  unimportant.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "tests" / "data"
MANIFEST = DATA / "MANIFEST.json"

ORIGINS = {"generated", "r-reference", "upstream", "unverified"}

#: The files whose origin this repository cannot state. Pinned deliberately:
#: see the module docstring.
UNVERIFIED = {
    "tvhr_piecewise.csv",
}


@pytest.fixture(scope="module")
def manifest() -> dict:
    assert MANIFEST.exists(), (
        "tests/data/MANIFEST.json is missing; run "
        "`python tools/make_data_manifest.py`")
    return json.loads(MANIFEST.read_text(encoding="utf-8"))


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_every_fixture_is_listed_and_nothing_else_is(manifest):
    on_disk = {p.name for p in DATA.iterdir()
               if p.is_file() and p.name != MANIFEST.name}
    listed = set(manifest["files"])
    assert not (on_disk - listed), (
        f"fixtures with no manifest entry: {sorted(on_disk - listed)}; run "
        "`python tools/make_data_manifest.py`")
    assert not (listed - on_disk), (
        f"manifest entries with no file: {sorted(listed - on_disk)}")


@pytest.mark.parametrize("name", sorted(json.loads(
    MANIFEST.read_text(encoding="utf-8"))["files"]) if MANIFEST.exists() else [])
def test_the_bytes_are_the_bytes_the_manifest_recorded(manifest, name):
    path = DATA / name
    entry = manifest["files"][name]
    assert path.stat().st_size == entry["bytes"], (
        f"{name} is {path.stat().st_size} bytes, the manifest says "
        f"{entry['bytes']}; the fixture changed, so the tests that depend on it "
        "are no longer testing what they recorded")
    assert _sha256(path) == entry["sha256"], f"{name} has changed content"


def test_every_entry_declares_a_known_origin_and_a_licence(manifest):
    for name, entry in manifest["files"].items():
        assert entry["origin"] in ORIGINS, f"{name}: {entry['origin']!r}"
        assert entry.get("licence"), f"{name} has no licence field"
        if entry["origin"] == "generated":
            assert entry.get("produced_by"), (
                f"{name} claims to be generated but names no script")
            assert (ROOT / entry["produced_by"]).exists(), (
                f"{name}: {entry['produced_by']} does not exist")
        if entry["origin"] == "upstream":
            assert entry.get("source"), (
                f"{name} claims an upstream source but does not name it")


def test_the_unverified_list_is_the_one_the_repository_admits_to(manifest):
    # Both directions. Growing the list is a deliberate edit, and shrinking it
    # requires recording a source above, which is the point.
    recorded = set(manifest["unverified"])
    assert recorded == UNVERIFIED, (
        f"manifest says unverified={sorted(recorded)}, the test says "
        f"{sorted(UNVERIFIED)}; update both, and record a source if one is now "
        "known")


def test_every_unverified_file_says_so_in_its_entry(manifest):
    for name in manifest["unverified"]:
        entry = manifest["files"][name]
        assert entry["origin"] == "unverified"
        assert entry["licence"] == "unverified", (
            f"{name} is unverified but claims licence {entry['licence']!r}")
        assert entry["note"].strip(), f"{name} has no explanation"


def test_the_manifest_is_regenerable_and_current():
    # The same check the tool offers, run here so a stale manifest cannot be
    # committed even by someone who never runs the tool.
    tool = ROOT / "tools" / "make_data_manifest.py"
    import importlib.util

    spec = importlib.util.spec_from_file_location("make_data_manifest", tool)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    assert module.render(module.build()) == MANIFEST.read_text(encoding="utf-8")


# --------------------------------------------------------------- regenerable
def test_a_generated_fixture_can_be_rebuilt_from_its_script(manifest):
    """A generator is only a source if running it reproduces the file.

    Scripts that advertise ``--check`` are actually executed; the rest are
    recorded but not claimed to be byte-reproducible, which is a weaker and
    honest statement.
    """
    import subprocess
    import sys

    generated = {n: e for n, e in manifest["files"].items()
                 if e["origin"] == "generated"}
    assert generated
    checked = 0
    for name, entry in generated.items():
        script = ROOT / entry["produced_by"]
        assert script.exists(), f"{name}: {entry['produced_by']} is missing"
        if '"--check"' not in script.read_text(encoding="utf-8"):
            continue
        proc = subprocess.run([sys.executable, str(script), "--check"],
                              cwd=ROOT, capture_output=True, text=True)
        assert proc.returncode == 0, (
            f"{name} does not match {entry['produced_by']}:\n"
            f"{proc.stdout}{proc.stderr}")
        checked += 1
    assert checked >= 1, "no generated fixture advertises --check"


def test_the_regenerated_bivariate_fixture_has_a_reference_script():
    # The R numbers in tests/test_prediction_joint.py were recomputed after the
    # fixture was rebuilt; the script that does it has to stay in the repository
    # or the next regeneration is guesswork.
    script = ROOT / "tools" / "r_reference" / "prediction_joint.R"
    assert script.exists()
    text = script.read_text(encoding="utf-8")
    for name in ("R_RHO40_BETA", "R_RHO40_TAU2", "R_SEPARATE"):
        assert name in text, name
