"""Non-linear meta-regression: bases, joint tests, curves and the SROC plot.

Every number in here comes from R 4.6.1 with ``rms`` 8.1.1 and ``metafor``
5.2.1:

* the spline matrices are ``rms::rcs(x, parms = c(...))``, printed to 15 digits;
* the fit is ``rma(yi, vi, mods = ~ rcs(year0, parms = c(2, 12, 26)))`` on
  ``dat.bcg`` under REML;
* the tests are ``anova(fit, btt = ...)``;
* the curve is ``predict(fit, newmods = <spline matrix>)``.

The knots are always written out explicitly, because ``pimeta`` places them by
its own documented rule rather than copying ``rms``'s default — the basis
*formula* is what is being verified, not another package's convention.
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pytest  # noqa: E402

import pimeta as ma  # noqa: E402

# --------------------------------------------------------------------------- #
# reference data
# --------------------------------------------------------------------------- #
X = np.array([1, 2, 4, 7, 9, 11, 15, 18, 22, 26, 31, 37, 44], dtype=float)

# rms::rcs(X, parms = c(9, 15, 22))[, 2]
RCS_3_COL2 = [
    0.0, 0.0, 0.0, 0.0, 0.0,
    0.0473372781065089, 1.27810650887574, 4.01690617075233,
    9.23076923076923, 14.7692307692308, 21.6923076923077,
    30.0, 39.6923076923078,
]

# rms::rcs(X, parms = c(4, 10, 18, 30))[, 2:3]
RCS_4_COL2 = [
    0.0, 0.0, 0.0,
    0.0399408284023669, 0.18491124260355, 0.507396449704142,
    1.9689349112426, 4.05917159763314, 8.42209072978304,
    14.1104536489152, 22.0769230769231, 31.7692307692308, 43.0769230769231,
]
RCS_4_COL3 = [
    0.0, 0.0, 0.0, 0.0, 0.0,
    0.0014792899408284, 0.18491124260355, 0.757396449704143,
    2.39842209072979, 4.79684418145957, 8.28402366863906,
    12.5443786982249, 17.5147928994083,
]

# metafor::dat.bcg, escalc(measure="RR") — the same 13 studies as dat.bcg.
YEAR0 = np.array([0, 1, 12, 29, 25, 5, 25, 32, 20, 13, 26, 21, 28], dtype=float)
BCG_YI = np.array([
    -0.889311333920205, -1.58538865720143, -1.34807314829969,
    -1.44155119002131, -0.217547322211296, -0.786115585818864,
    -1.62089822359839, 0.0119523335238405, -0.469417648738149,
    -1.37134480347278, -0.339358828338391, 0.445913400571379,
    -0.0173139482168798,
])
BCG_VI = np.array([
    0.325584765003961, 0.194581121398144, 0.415367965367965,
    0.0200100319022476, 0.0512101721696309, 0.00690561845590876,
    0.223017247572315, 0.00396157929781773, 0.056434210463249,
    0.0730247936130289, 0.0124122139715597, 0.532505845200153,
    0.0714046596839863,
])
BCG = {"year0": YEAR0, "yi": BCG_YI, "vi": BCG_VI}
KNOTS = [2.0, 12.0, 26.0]
TERM = "rcs(year0, knots=[2,12,26])"
MODS = "~ rcs(year0, knots=[2,12,26])"

# rma(yi, vi, mods = ~ rcs(year0, parms = c(2, 12, 26)), method = "REML")
R_COEF = [-1.16270701940409, 0.0127694163276215, 0.0182405485874188]
R_SE = [0.484558386570301, 0.0621575504882448, 0.0781539121763369]
R_TAU2 = 0.296250946333081
R_QE = 91.9112660639736
# anova(fit, btt = 2:3) — the whole spline term, in metafor's indexing
R_QM_TERM = 2.543057153982
R_QM_TERM_P = 0.280402676944108

# predict(fit, newmods = rms::rcs(c(0, 10, 26), parms = c(2, 12, 26)))
GRID = [0.0, 10.0, 26.0]
R_PRED = [-1.16270701940409, -1.01879903516128, -0.541893508918462]
R_PRED_SE = [0.484558386570301, 0.325196186240139, 0.207551116777214]
R_PRED_CI_LB = [-2.11242400548871, -1.65617184810173, -0.94868622275287]
R_PRED_CI_UB = [-0.212990033319459, -0.381426222220824, -0.135100795084055]
R_PRED_PI_LB = [-2.59099206258975, -2.26148962963997, -1.68360981473132]
R_PRED_PI_UB = [0.265578023781575, 0.223891559317423, 0.599822796894395]


def _bcg_fit(**kw):
    return ma.rma(BCG_YI, BCG_VI, mods=MODS, data=BCG, **kw)


# --------------------------------------------------------------------------- #
# restricted cubic splines
# --------------------------------------------------------------------------- #
def test_rcs_matches_rms_with_three_knots():
    basis = ma.rcs(X, knots=[9, 15, 22])
    assert basis.shape == (13, 2)
    assert basis.values[:, 0] == pytest.approx(X)
    assert basis.values[:, 1] == pytest.approx(RCS_3_COL2, rel=1e-13)


def test_rcs_matches_rms_with_four_knots():
    basis = ma.rcs(X, knots=[4, 10, 18, 30])
    assert basis.shape == (13, 3)
    assert basis.values[:, 1] == pytest.approx(RCS_4_COL2, rel=1e-13)
    assert basis.values[:, 2] == pytest.approx(RCS_4_COL3, rel=1e-13)


def test_the_default_knots_are_harrells_percentiles():
    knots = ma.default_knots(X, 4)
    expected = np.quantile(X, [0.05, 0.35, 0.65, 0.95])
    assert knots == pytest.approx(expected)
    # 4 knots give 3 columns, which is the df people usually quote
    assert ma.rcs(X, nk=4).shape == (13, 3)
    assert ma.rcs(X, df=3).shape == (13, 3)


def test_a_restricted_spline_is_linear_beyond_the_outer_knots():
    """That restriction is the whole reason to prefer a spline to a polynomial."""
    grid = np.array([30.0, 40.0, 50.0, 60.0, 70.0])
    basis = ma.rcs(grid, knots=[2.0, 12.0, 26.0])
    second_diff = np.diff(basis.values[:, 1], n=2)
    assert second_diff == pytest.approx(np.zeros(second_diff.size), abs=1e-10)


def test_rcs_rejects_impossible_knots():
    with pytest.raises(ValueError, match="at least 3 knots"):
        ma.rcs(X, knots=[1, 2])
    with pytest.raises(ValueError, match="strictly increasing"):
        ma.rcs(X, knots=[9, 9, 22])
    with pytest.raises(ValueError, match="distinct knots"):
        ma.rcs(np.array([1.0, 1.0, 1.0, 2.0, 3.0]), nk=5)


def test_a_basis_can_be_re_evaluated_at_new_values():
    basis = ma.rcs(X, knots=[2.0, 12.0, 26.0])
    again = basis.evaluate([0.0, 10.0, 26.0])
    assert again.shape == (3, 2)
    # The first column is x itself, so a re-evaluation is checkable by eye.
    assert again.values[:, 0] == pytest.approx([0.0, 10.0, 26.0])
    assert again.names == basis.names


# --------------------------------------------------------------------------- #
# fractional polynomials
# --------------------------------------------------------------------------- #
V = np.array([2, 3, 5, 8, 13, 21], dtype=float)


def test_fp_power_one_is_the_linear_term():
    basis = ma.fp(V, powers=[1])
    assert basis.values[:, 0] == pytest.approx(V)


def test_fp_power_zero_is_the_log():
    basis = ma.fp(V, powers=[0])
    assert basis.values[:, 0] == pytest.approx(np.log(V))


def test_a_repeated_power_multiplies_by_log():
    basis = ma.fp(V, powers=[1, 1])
    assert basis.values[:, 0] == pytest.approx(V)
    assert basis.values[:, 1] == pytest.approx(V * np.log(V))


def test_fp_half_a_power_repeated_is_the_sqrt_times_log():
    basis = ma.fp(V, powers=[0.5, 0.5])
    root = np.sqrt(V)
    assert basis.values[:, 0] == pytest.approx(root)
    assert basis.values[:, 1] == pytest.approx(root * np.log(V))


def test_fp_powers_are_sorted_like_the_literature():
    basis = ma.fp(V, powers=[2, -1])
    assert basis.detail["powers"] == [-1.0, 2.0]
    assert basis.values[:, 0] == pytest.approx(V ** -1)


def test_fp_scaling_does_not_change_the_fit_only_the_coefficients():
    plain = ma.fp(V, powers=[1, 2])
    scaled = ma.fp(V, powers=[1, 2], scale=21.0)
    assert not np.allclose(plain.values, scaled.values)
    # Both span the same column space, which is what the model sees.
    assert np.linalg.matrix_rank(np.column_stack([plain.values, scaled.values])) == 2


def test_fp_rejects_non_positive_values():
    with pytest.raises(ValueError, match="strictly positive"):
        ma.fp(np.array([0.0, 1.0, 2.0]), powers=[0])
    with pytest.raises(ValueError, match="at least one power"):
        ma.fp(V, powers=[])


# --------------------------------------------------------------------------- #
# formulas
# --------------------------------------------------------------------------- #
def test_a_basis_call_expands_inside_a_formula():
    fit = _bcg_fit()
    assert fit.coefnames == ["intrcpt", f"{TERM}[1]", f"{TERM}[2]"]


def test_a_spline_meta_regression_matches_r():
    fit = _bcg_fit()
    assert fit.b == pytest.approx(R_COEF, rel=1e-10)
    assert fit.se == pytest.approx(R_SE, rel=1e-9)
    assert fit.tau2 == pytest.approx(R_TAU2, rel=1e-9)
    assert fit.QE == pytest.approx(R_QE, rel=1e-10)


def test_whitespace_in_the_call_does_not_matter():
    tight = ma.rma(BCG_YI, BCG_VI, mods="~ rcs(year0,knots=[2,12,26])", data=BCG)
    loose = ma.rma(BCG_YI, BCG_VI, mods="~ rcs( year0 , knots = [2, 12, 26] )",
                   data=BCG)
    assert tight.b == pytest.approx(loose.b, rel=1e-14)


def test_negative_powers_survive_the_formula_parser():
    """A bare '-' would be read as a term separator without depth tracking."""
    data = {"x": V, "yi": BCG_YI[:6], "vi": BCG_VI[:6]}
    fit = ma.rma(data["yi"], data["vi"], mods="~ fp(x, [-1, 0])", data=data)
    assert fit.coefnames == ["intrcpt", "fp(x, [-1,0])[1]", "fp(x, [-1,0])[2]"]
    assert np.all(np.isfinite(fit.b))


def test_a_basis_object_can_be_passed_as_mods():
    basis = ma.rcs(YEAR0, knots=KNOTS, name=TERM)
    from_basis = ma.rma(BCG_YI, BCG_VI, mods=basis)
    from_formula = _bcg_fit()
    assert from_basis.b == pytest.approx(from_formula.b, rel=1e-12)
    assert from_basis.se == pytest.approx(from_formula.se, rel=1e-12)
    assert from_basis.coefnames == from_formula.coefnames


def test_an_unknown_basis_name_still_reads_as_a_missing_column():
    with pytest.raises(ValueError, match="does not name a column"):
        ma.rma(BCG_YI, BCG_VI, mods="~ nosuchbasis(year0)", data=BCG)


def test_a_basis_call_with_a_missing_column_says_so():
    with pytest.raises(ValueError, match="refers to column 'dose'"):
        ma.rma(BCG_YI, BCG_VI, mods="~ rcs(dose, 4)", data=BCG)


def test_a_basis_call_with_a_stray_keyword_says_so():
    with pytest.raises(ValueError, match="unexpected argument"):
        ma.rma(BCG_YI, BCG_VI, mods="~ rcs(year0, degree=4)", data=BCG)


# --------------------------------------------------------------------------- #
# joint tests
# --------------------------------------------------------------------------- #
def test_the_term_test_matches_metafors_anova():
    fit = _bcg_fit()
    test = ma.nonlinear_test(fit, TERM)
    # metafor's anova(fit, btt = 2:3) tests the two spline coefficients.
    assert test.overall.QM == pytest.approx(R_QM_TERM, rel=1e-9)
    assert test.overall.pval == pytest.approx(R_QM_TERM_P, rel=1e-9)
    assert test.overall.df == 2
    assert test.overall.test == "chi2"


def test_the_non_linearity_test_is_the_wald_square_of_the_extra_coefficient():
    fit = _bcg_fit()
    test = ma.nonlinear_test(fit, TERM)
    # A single coefficient: the joint Wald statistic is z^2.
    assert test.nonlinear.df == 1
    assert test.nonlinear.QM == pytest.approx(fit.zval[2] ** 2, rel=1e-12)


def test_a_linear_model_has_no_non_linear_part():
    fit = ma.rma(BCG_YI, BCG_VI, mods="~ fp(year0, [1])", data=BCG)
    with pytest.raises(ValueError, match="cannot be non-linear"):
        ma.nonlinear_test(fit, "fp(year0,[1])")


def test_wald_test_takes_one_based_positions_like_metafors_btt():
    fit = _bcg_fit()
    first = ma.wald_test(fit, btt=[1])
    assert first.QM == pytest.approx(fit.zval[0] ** 2, rel=1e-12)
    with pytest.raises(ValueError, match="outside 1..3"):
        ma.wald_test(fit, btt=[0])
    with pytest.raises(ValueError, match="outside 1..3"):
        ma.wald_test(fit, btt=[4])


def test_wald_test_accepts_names():
    fit = _bcg_fit()
    by_name = ma.wald_test(fit, names=[f"{TERM}[1]", f"{TERM}[2]"])
    assert by_name.QM == pytest.approx(R_QM_TERM, rel=1e-9)
    with pytest.raises(ValueError, match="no coefficient named"):
        ma.wald_test(fit, names=["dose"])


def test_wald_test_says_what_it_needs():
    with pytest.raises(ValueError, match="give 'btt' or 'names'"):
        ma.wald_test(_bcg_fit())


# --------------------------------------------------------------------------- #
# curves
# --------------------------------------------------------------------------- #
def test_the_predicted_curve_matches_r():
    fit = _bcg_fit()
    basis = ma.rcs(YEAR0, knots=KNOTS, name=TERM)
    curve = ma.predict_curve(fit, basis, grid=GRID)
    assert curve.fit == pytest.approx(R_PRED, rel=1e-10)
    assert curve.se == pytest.approx(R_PRED_SE, rel=1e-9)
    assert curve.ci_lb == pytest.approx(R_PRED_CI_LB, rel=1e-9)
    assert curve.ci_ub == pytest.approx(R_PRED_CI_UB, rel=1e-9)
    assert curve.pi_lb == pytest.approx(R_PRED_PI_LB, rel=1e-9)
    assert curve.pi_ub == pytest.approx(R_PRED_PI_UB, rel=1e-9)
    assert curve.level == pytest.approx(0.95)


def test_the_default_grid_spans_the_observed_range():
    fit = _bcg_fit()
    curve = ma.predict_curve(fit, ma.rcs(YEAR0, knots=KNOTS, name=TERM))
    assert curve.x.size == 100
    assert curve.x.min() == pytest.approx(YEAR0.min())
    assert curve.x.max() == pytest.approx(YEAR0.max())


def test_a_curve_needs_the_other_moderators_to_be_pinned():
    fit = ma.rma(BCG_YI, BCG_VI, mods=f"{MODS} + fp_other",
                 data={**BCG, "fp_other": np.linspace(0.1, 1.0, YEAR0.size)})
    other = ma.rcs(YEAR0, knots=KNOTS, name="rcs(year0, knots=[2,12,26])")
    with pytest.raises(ValueError, match="does not have"):
        ma.predict_curve(fit, other, grid=GRID, at={"nope": 1.0})
    # ... and holding it at a value works, and moves the whole curve by the
    # moderator's own coefficient -- the basis columns are unaffected.
    pinned = ma.predict_curve(fit, other, grid=GRID, at={"fp_other": 0.0})
    shifted = ma.predict_curve(fit, other, grid=GRID, at={"fp_other": 1.0})
    delta = shifted.fit - pinned.fit
    assert delta == pytest.approx(np.full(delta.size, delta[0]))
    index = fit.coefnames.index("fp_other")
    assert delta[0] == pytest.approx(fit.b[index], rel=1e-12)


def test_a_curve_rejects_a_basis_with_the_wrong_number_of_columns():
    fit = _bcg_fit()
    other = ma.rcs(YEAR0, knots=[2.0, 10.0, 18.0, 30.0], name=TERM)
    with pytest.raises(ValueError, match="column"):
        ma.predict_curve(fit, other, grid=GRID)


# --------------------------------------------------------------------------- #
# SROC plot
# --------------------------------------------------------------------------- #
AUDITC = {
    "TP": [47, 126, 19, 36, 130, 84, 68, 752, 59, 142, 137, 57, 34, 152],
    "FN": [9, 51, 10, 3, 19, 2, 0, 0, 5, 50, 24, 3, 1, 51],
    "FP": [101, 272, 12, 78, 211, 68, 112, 3226, 55, 571, 107, 103, 21, 88],
    "TN": [738, 1543, 192, 276, 959, 89, 423, 2977, 136, 2788, 358, 437, 56, 264],
}


@pytest.fixture
def auditc_fit():
    with pytest.warns(ma.DtaCorrectionWarning):
        return ma.reitsma(AUDITC)


def test_plot_sroc_draws_curve_points_and_summary(auditc_fit):
    ax = ma.plot_sroc(auditc_fit)
    assert isinstance(ax, plt.Axes)
    # diagonal + curve + two crosshair arms
    assert len(ax.lines) == 4
    # study points + summary point
    assert len(ax.collections) == 2
    assert ax.get_xlabel().startswith("False positive rate")
    assert ax.get_xlim() == pytest.approx((0.0, 1.0))
    labels = [t.get_text() for t in ax.texts]
    assert any(lbl.startswith("AUC = ") for lbl in labels)
    plt.close(ax.figure)


def test_plot_sroc_can_be_stripped_down(auditc_fit):
    ax = ma.plot_sroc(auditc_fit, curve=False, summary=False, ci=False,
                      show_auc=False, studlab=[])
    assert len(ax.lines) == 1          # the diagonal only
    assert len(ax.texts) == 0
    plt.close(ax.figure)


def test_plot_sroc_rejects_a_non_dta_result():
    with pytest.raises(TypeError, match="ReitsmaResult"):
        ma.plot_sroc(ma.rma(BCG_YI, BCG_VI))


def test_plot_sroc_writes_a_file(auditc_fit, tmp_path):
    out = tmp_path / "figs" / "sroc.png"
    ma.plot_sroc(auditc_fit, filename=str(out))
    assert out.exists() and out.stat().st_size > 1000
    plt.close("all")


# --------------------------------------------------------------------------- #
# the record rule: a non-linear model has to be reproducible too
# --------------------------------------------------------------------------- #
def test_a_spline_model_round_trips_through_a_record(tmp_path):
    from pimeta.record import load_record, save_record

    fit = _bcg_fit()
    path = tmp_path / "spline.json"
    save_record(
        path, fit, data=BCG, mode="bundle", function="rma",
        call={"yi": "yi", "vi": "vi", "mods": f"~ rcs(year0, knots={KNOTS})",
              "method": "REML"},
    )
    loaded = load_record(path)
    assert loaded.kind == "rma"
    assert loaded.analysis["settings"]["method"] == "REML"
    loaded.verify()
    again = loaded.replay()
    assert again.b == pytest.approx(fit.b, rel=1e-12)


def test_a_wald_test_is_recordable():
    fit = _bcg_fit()
    test = ma.nonlinear_test(fit, TERM)
    record = ma.build_record(test, function="nonlinear_test",
                             call={"term": TERM})
    payload = record.results["values"]
    assert payload["term"] == TERM
    assert payload["overall"]["QM"] == pytest.approx(R_QM_TERM, rel=1e-9)
