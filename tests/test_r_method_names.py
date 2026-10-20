"""R's ``generic.class`` S3 method names as resolvable aliases.

R writes its methods as ``print.rma.uni``, ``cooks.distance.rma.mv``,
``[.list.rma`` ...  None of those strings are valid Python identifiers (``.``
would be attribute access, ``[``/``$`` are syntax), so pimeta exports
R-independent names and maps the R spellings onto them.  This lets a
mechanical R -> Python translation (which produces
``getattr(pimeta, "cooks.distance.rma.mv")``) resolve to the real callable.
"""

import pytest

import pimeta as ma


def test_r_methods_is_a_non_empty_map_of_callables():
    m = ma.r_methods()
    assert isinstance(m, dict) and len(m) >= 60
    for r_name, _py_name in m.items():
        assert "." in r_name or "[" in r_name or "$" in r_name
        assert callable(getattr(ma, r_name))


def test_r_methods_maps_the_documented_names():
    m = ma.r_methods()
    assert m["cooks.distance.rma.mv"] == "cooks_distance"
    assert m["profile.rma.mv"] == "profile_rmamv"
    assert m["profile.rma.ls"] == "profile_rma_ls"
    assert m["profile.rma.uni.selmodel"] == "profile_selmodel"
    assert m["hc.rma.uni"] == "hc_rma"
    assert m["[.list.rma"] == "slice_rma"
    assert m["head.list.rma"] == "head_rma"
    assert m["tail.list.rma"] == "tail_rma"
    assert m["print.rma.uni"] == "summary"
    assert m["ranktest.rma"] == "begg_test"


def test_getattr_resolves_r_names():
    assert getattr(ma, "print.rma.uni") is ma.summary
    assert getattr(ma, "[.list.rma") is ma.slice_rma
    assert getattr(ma, "cooks.distance.rma.mv") is ma.cooks_distance
    assert getattr(ma, "confint.rma.ls") is ma.confint_rma_ls


def test_unknown_r_name_still_raises_attribute_error():
    with pytest.raises(AttributeError, match="has no attribute"):
        ma.print.rma.uni            # noqa: B018 - attribute chaining must fail
    assert not hasattr(ma, "nope.rma")
    assert not hasattr(ma, "print.rma.nonexistent")


def test_every_r_alias_target_exists():
    """The map is filtered at import, so nothing may dangle."""
    for r_name, py_name in ma.r_methods().items():
        assert hasattr(ma, py_name), f"{r_name} -> {py_name} is missing"


def test_r_alias_does_not_shadow_real_attributes():
    """__getattr__ only runs for missing names, so real ones are untouched."""
    assert ma.summary is not ma.r_methods().get("summary")
    assert callable(ma.summary)
