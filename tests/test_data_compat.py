"""Tests for :mod:`pimeta.data_compat` — frames, formulas and table output."""

from __future__ import annotations

import numpy as np
import pytest

import pimeta as ma
from pimeta.data_compat import (
    FormulaSpec,
    as_columns,
    column_names,
    design_matrix,
    install_repr_html,
    lookup,
    model_frame,
    parse_formula,
    resolve,
    resolve_mods,
    to_dataframe,
    to_html,
)

# --------------------------------------------------------------------------- #
# Fixtures
# --------------------------------------------------------------------------- #
FRAME = {
    "yi": [0.20, 0.40, 0.10, 0.50, 0.30, 0.15],
    "vi": [0.04, 0.05, 0.03, 0.06, 0.02, 0.05],
    "dose": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
    "year": [1990.0, 1995.0, 2000.0, 2005.0, 2010.0, 2015.0],
    "arm": ["A", "B", "A", "B", "A", "B"],
    "slab": ["S1", "S2", "S3", "S4", "S5", "S6"],
}


# --------------------------------------------------------------------------- #
# Frame normalisation
# --------------------------------------------------------------------------- #
def test_as_columns_from_dict():
    cols = as_columns(FRAME)
    assert set(cols) == set(FRAME)
    assert isinstance(cols["yi"], np.ndarray)
    assert cols["yi"].tolist() == FRAME["yi"]


def test_as_columns_none_is_empty():
    assert as_columns(None) == {}


def test_as_columns_from_records():
    records = [{"yi": 0.1, "vi": 0.01}, {"yi": 0.2, "vi": 0.02}]
    cols = as_columns(records)
    assert cols["yi"].tolist() == [0.1, 0.2]
    assert cols["vi"].tolist() == [0.01, 0.02]


def test_as_columns_from_structured_array():
    arr = np.array([(0.1, 0.01), (0.2, 0.02)],
                   dtype=[("yi", "f8"), ("vi", "f8")])
    cols = as_columns(arr)
    assert cols["yi"].tolist() == [0.1, 0.2]


def test_as_columns_from_pandas():
    pd = pytest.importorskip("pandas")
    cols = as_columns(pd.DataFrame({"yi": [0.1, 0.2], "vi": [0.01, 0.02]}))
    assert cols["yi"].tolist() == [0.1, 0.2]


def test_as_columns_from_polars():
    pl = pytest.importorskip("polars")
    cols = as_columns(pl.DataFrame({"yi": [0.1, 0.2]}))
    assert cols["yi"].tolist() == [0.1, 0.2]


def test_as_columns_rejects_unknown_type():
    with pytest.raises(TypeError, match="must be a dict"):
        as_columns(42)


def test_column_names_and_lookup():
    assert column_names(FRAME) == list(FRAME)
    assert lookup(FRAME, "dose").tolist() == FRAME["dose"]


def test_lookup_missing_column_lists_alternatives():
    with pytest.raises(ValueError, match="not found in 'data'"):
        lookup(FRAME, "nope")


# --------------------------------------------------------------------------- #
# resolve
# --------------------------------------------------------------------------- #
def test_resolve_plain_vector():
    assert resolve([1, 2, 3], name="vi").tolist() == [1.0, 2.0, 3.0]


def test_resolve_column_name():
    assert resolve("dose", FRAME, name="mods").tolist() == FRAME["dose"]


def test_resolve_column_name_without_data_raises():
    with pytest.raises(ValueError, match="no 'data' was supplied"):
        resolve("dose", None, name="mods")


def test_resolve_list_of_column_names():
    out = resolve(["dose", "year"], FRAME, name="mods")
    assert isinstance(out, list) and len(out) == 2
    assert out[0].tolist() == FRAME["dose"]
    assert out[1].tolist() == FRAME["year"]


def test_resolve_string_list_is_data_when_not_columns():
    # Regression: a character vector such as study labels must stay data.
    labels = ["Study 1", "Study 2", "Study 3"]
    out = resolve(labels, None, name="slab", dtype=None)
    assert out.tolist() == labels


def test_resolve_string_list_with_frame_but_non_matching_names():
    labels = ["Study 1", "Study 2"]
    out = resolve(labels, FRAME, name="slab", dtype=None)
    assert out.tolist() == labels


def test_resolve_strings_as_names_forced_off():
    out = resolve(["dose", "year"], FRAME, name="x",
                  dtype=None, strings_as_names=False)
    assert out.tolist() == ["dose", "year"]


def test_resolve_length_check():
    with pytest.raises(ValueError, match="were expected"):
        resolve([1, 2], k=3, name="vi")


def test_resolve_none():
    assert resolve(None, name="vi") is None


# --------------------------------------------------------------------------- #
# Formulas
# --------------------------------------------------------------------------- #
def test_parse_formula_main_effects():
    spec = parse_formula("~ dose + year")
    assert spec == FormulaSpec(response=None, terms=["dose", "year"], intercept=True)


def test_parse_formula_interaction_expansion():
    assert parse_formula("~ dose*year").terms == ["dose", "year", "dose:year"]


def test_parse_formula_interaction_only():
    assert parse_formula("~ dose:year").terms == ["dose:year"]


def test_parse_formula_without_intercept():
    assert parse_formula("~ 0 + dose").intercept is False
    assert parse_formula("~ dose - 1").intercept is False
    assert parse_formula("~ dose + 0").intercept is False


def test_parse_formula_two_sided():
    spec = parse_formula("yi ~ dose")
    assert spec.response == "yi"
    assert spec.terms == ["dose"]


def test_parse_formula_dot():
    assert parse_formula("~ .").terms == ["."]


def test_parse_formula_backticks():
    assert parse_formula("~ `dose.gy`").terms == ["dose.gy"]


def test_parse_formula_rejects_empty():
    with pytest.raises(ValueError, match="empty formula"):
        parse_formula("   ")


def test_parse_formula_passthrough():
    spec = FormulaSpec(terms=["x"])
    assert parse_formula(spec) is spec


def test_formula_spec_repr():
    assert "~ 'dose + year'" in repr(parse_formula("~ dose + year"))
    assert "0 + " in repr(parse_formula("~ 0 + dose"))


# --------------------------------------------------------------------------- #
# design_matrix / model_frame
# --------------------------------------------------------------------------- #
def test_design_matrix_intercept_only():
    X, names = design_matrix("~ 1", FRAME)
    assert names == ["(Intercept)"]
    assert X.shape == (6, 1)


def test_design_matrix_numeric():
    X, names = design_matrix("~ dose + year", FRAME)
    assert names == ["(Intercept)", "dose", "year"]
    assert X.shape == (6, 3)
    assert np.allclose(X[:, 1], FRAME["dose"])


def test_design_matrix_no_intercept():
    X, names = design_matrix("~ 0 + dose", FRAME)
    assert names == ["dose"]
    assert X.shape == (6, 1)


def test_design_matrix_dummy_coding():
    X, names = design_matrix("~ arm", FRAME)
    # R drops the first level and names the rest arm[level].
    assert names == ["(Intercept)", "arm[B]"]
    assert X[:, 1].tolist() == [0.0, 1.0, 0.0, 1.0, 0.0, 1.0]


def test_design_matrix_interaction():
    X, names = design_matrix("~ dose:arm", FRAME)
    assert "dose:arm[B]" in names


def test_design_matrix_dot_expands_other_columns():
    X, names = design_matrix("~ .", FRAME)
    assert "dose" in names and "year" in names
    assert "arm[B]" in names


def test_design_matrix_missing_term_raises():
    with pytest.raises(ValueError, match="does not name a column"):
        design_matrix("~ nope", FRAME)


def test_model_frame_two_sided():
    y, X, names = model_frame("yi ~ dose", FRAME)
    assert y.tolist() == FRAME["yi"]
    assert names == ["(Intercept)", "dose"]


def test_model_frame_one_sided():
    y, X, _ = model_frame("~ dose", FRAME)
    assert y is None
    assert X.shape == (6, 2)


# --------------------------------------------------------------------------- #
# resolve_mods
# --------------------------------------------------------------------------- #
def test_resolve_mods_none():
    assert resolve_mods(None) == (None, True)


def test_resolve_mods_formula():
    arr, intercept = resolve_mods("~ dose + year", FRAME, k=6)
    assert arr.shape == (6, 2)
    assert intercept is True


def test_resolve_mods_formula_without_intercept():
    arr, intercept = resolve_mods("~ 0 + dose", FRAME, k=6)
    assert arr.shape == (6, 1)
    assert intercept is False


def test_resolve_mods_column_list():
    arr, intercept = resolve_mods(["dose", "year"], FRAME, k=6)
    assert arr.shape == (6, 2)
    assert intercept is True


def test_resolve_mods_array():
    arr, intercept = resolve_mods(np.array(FRAME["dose"]), None, k=6)
    assert arr.shape == (6, 1)
    assert intercept is True


def test_resolve_mods_two_sided_formula_rejected():
    with pytest.raises(ValueError, match="one-sided"):
        resolve_mods("yi ~ dose", FRAME, k=6)


def test_resolve_mods_formula_without_data_raises():
    with pytest.raises(ValueError, match="no 'data' was supplied"):
        resolve_mods("~ dose", None, k=6)


def test_resolve_mods_columns_without_data_raise():
    with pytest.raises(ValueError, match="no 'data'"):
        resolve_mods(["dose"], None, k=6)


# --------------------------------------------------------------------------- #
# Table output
# --------------------------------------------------------------------------- #
def test_to_dataframe_and_html():
    r = ma.rma(FRAME["yi"], FRAME["vi"])
    table = to_dataframe(r)
    assert table is not None
    html = to_html(r)
    assert html.startswith("<div") and "RmaResult" in html


def test_to_html_truncates():
    r = ma.rma(FRAME["yi"], FRAME["vi"])
    short = to_html(r, max_rows=2)
    assert "more fields" in short


def test_to_html_survives_a_nested_mapping_in_to_dict():
    # `RmaResult.to_dict()` carries a `diagnostics` block, which is a dict. The
    # flattener used to assume that every mapping it recursed into was an
    # object with a __dict__, so adding that block broke `to_html` for every
    # fit. Nothing in the diagnostics tests noticed, because they never render
    # a table.
    r = ma.rma(FRAME["yi"], FRAME["vi"])
    assert isinstance(r.to_dict()["diagnostics"], dict)
    html = to_html(r)
    assert "diagnostics." in html, (
        "nested mappings should reach the table with a dotted name")
    table = to_dataframe(r)
    assert table is not None


def test_install_repr_html_is_idempotent():
    namespace = {"RmaResult": ma.RmaResult}
    assert install_repr_html(namespace) == 0
    assert hasattr(ma.RmaResult, "_repr_html_")


# --------------------------------------------------------------------------- #
# End-to-end wiring
# --------------------------------------------------------------------------- #
def test_rma_accepts_column_names_and_formula():
    r = ma.rma("yi", "vi", mods="~ dose", data=FRAME)
    direct = ma.rma(FRAME["yi"], FRAME["vi"], mods=np.array(FRAME["dose"]))
    assert np.allclose(np.asarray(r.b), np.asarray(direct.b), atol=1e-12)


def test_rma_accepts_sei_column():
    frame = dict(FRAME, sei=[np.sqrt(v) for v in FRAME["vi"]])
    r = ma.rma("yi", sei="sei", data=frame)
    direct = ma.rma(FRAME["yi"], FRAME["vi"])
    assert np.allclose(np.asarray(r.b), np.asarray(direct.b), atol=1e-12)


def test_rma_still_accepts_string_slab():
    # Regression: slab is a character vector, not a column-name list.
    r = ma.rma(FRAME["yi"], FRAME["vi"], slab=["A", "B", "C", "D", "E", "F"])
    assert list(r.slab) == ["A", "B", "C", "D", "E", "F"]


def test_rma_slab_from_column():
    r = ma.rma("yi", "vi", slab="slab", data=FRAME)
    assert list(r.slab) == FRAME["slab"]


def test_rma_with_dataframe():
    pd = pytest.importorskip("pandas")
    r = ma.rma("yi", "vi", data=pd.DataFrame(FRAME))
    direct = ma.rma(FRAME["yi"], FRAME["vi"])
    assert np.allclose(np.asarray(r.b), np.asarray(direct.b), atol=1e-12)


def test_escalc_accepts_column_names():
    frame = {
        "ai": [10, 20, 15], "bi": [90, 80, 85],
        "ci": [12, 18, 14], "di": [88, 82, 86],
    }
    a = ma.escalc("RR", "ai", "bi", "ci", "di", data=frame)
    b = ma.escalc("RR", frame["ai"], frame["bi"], frame["ci"], frame["di"])
    assert np.allclose(np.asarray(a.yi), np.asarray(b.yi), atol=1e-12)


def test_metabin_accepts_column_names():
    frame = {"e1": [10, 20, 15], "n1": [100, 100, 100],
             "e2": [12, 18, 14], "n2": [100, 100, 100]}
    a = ma.metabin("e1", "n1", "e2", "n2", data=frame)
    b = ma.metabin(frame["e1"], frame["n1"], frame["e2"], frame["n2"])
    assert np.allclose(float(a.random.pooled_est), float(b.random.pooled_est), atol=1e-12)
    assert np.allclose(float(a.random.pooled_se), float(b.random.pooled_se), atol=1e-12)


def test_a_factor_term_without_an_intercept_keeps_every_level():
    """R's model.matrix(~ f - 1) is a full indicator matrix, not L-1 dummies.

    Dropping the reference level is only correct when an intercept is present;
    doing it unconditionally silently merges one level into another.
    """
    frame = {"g": ["a", "b", "c", "a", "b", "c"], "yi": [1.0, 2, 3, 4, 5, 6]}
    X, names = design_matrix("~ g - 1", frame)
    assert names == ["g[a]", "g[b]", "g[c]"]
    assert X.shape == (6, 3)
    assert np.allclose(X.sum(axis=1), 1.0)
    X, names = design_matrix("~ g", frame)
    assert names == ["(Intercept)", "g[b]", "g[c]"]
    assert X.shape == (6, 3)
