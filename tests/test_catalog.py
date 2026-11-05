"""The discovery layer has to stay honest, so it is checked like data.

The catalogue makes claims about what has been measured.  A claim that names a
file which does not exist, or a status outside the fixed vocabulary, would be
worse than having no catalogue at all -- so those two things are asserted here.
"""

from __future__ import annotations

import pathlib

import pytest

import pimeta as ma

REPO = pathlib.Path(__file__).resolve().parents[1]


def test_the_status_vocabulary_is_closed():
    for name in ma.list_models():
        assert ma.describe(name)["validation_status"] in ma.catalog.STATUSES


def test_a_shortfall_is_not_reported_as_validation():
    """The one measured failure must not be filed under "validated"."""
    assert ma.describe("predict_region")["validation_status"] != "validated"


def test_every_entry_names_an_entry_point_that_exists():
    for name in ma.list_models():
        entry = ma.describe(name)["entry_point"]
        if entry.startswith("("):
            continue
        assert hasattr(ma, entry) or entry in {"dose-response"}, (
            f"{name}: entry_point {entry!r} is not in the package namespace")


def test_evidence_points_at_a_file_that_exists():
    """Every claim must name something in this repository, not a memory."""
    for name in ma.list_models():
        evidence = ma.describe(name)["evidence"]
        # the path comes first; anything after ';' is commentary
        first = evidence.split(";")[0].split(",")[0].strip()
        if first.startswith("(no measurement"):
            continue
        candidates = [REPO / first,
                      REPO.parent / "metabench" / pathlib.Path(first).name,
                      REPO.parent / first]
        assert any(c.exists() for c in candidates), (
            f"{name}: evidence {first!r} does not exist")


def test_describe_refuses_an_unknown_name():
    with pytest.raises(KeyError) as excinfo:
        ma.describe("rma.typo")
    assert "known" in str(excinfo.value)


def test_describe_returns_copies():
    """A caller must not be able to edit the catalogue by accident."""
    first = ma.describe("rma")
    first["limitations"].append("mutated")
    assert "mutated" not in ma.describe("rma")["limitations"]


def test_the_lists_are_not_empty_and_agree():
    assert ma.list_estimators()
    assert ma.list_effect_sizes()
    assert ma.list_diagnostics()
    # every estimator offered by the two main models is in the list
    listed = set(ma.list_estimators())
    for method in ma.describe("rma")["methods"]:
        assert method in listed, f"rma offers {method!r}, which is not catalogued"


def test_the_known_gap_is_stated_rather_than_hidden():
    """The joint region is the one measured defect; it must say so."""
    info = ma.describe("predict_region")
    assert info["validation_status"] == "measured-shortfall"
    assert any("66.3%" in line for line in info["limitations"])
    assert "B_joint_region.md" in info["evidence"]
