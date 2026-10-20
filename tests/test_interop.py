"""Tests for :mod:`pimeta.interop` — the ecosystem protocol layer."""

from __future__ import annotations

import numpy as np
import pytest

import pimeta as ma
from pimeta.interop import (
    BACKENDS,
    MetaEstimator,
    MetaRegressor,
    as_results,
    augment,
    available,
    glance,
    install_methods,
    protocol_report,
    supports,
    tidy,
    to_frame,
)
from pimeta.interop.tables import normalize_columns

# --------------------------------------------------------------------------- #
# Fixtures
# --------------------------------------------------------------------------- #
YI = np.array([0.20, 0.40, 0.10, 0.50, 0.30, 0.15])
VI = np.array([0.04, 0.05, 0.03, 0.06, 0.02, 0.05])
DOSE = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
YEAR = np.array([1990.0, 1995.0, 2000.0, 2005.0, 2010.0, 2015.0])


@pytest.fixture(scope="module")
def fit():
    return ma.rma(YI, VI, method="REML")


@pytest.fixture(scope="module")
def fit_mods():
    return ma.rma(YI, VI, mods=DOSE, method="REML")


# --------------------------------------------------------------------------- #
# Protocols
# --------------------------------------------------------------------------- #
def test_protocol_report_marks_tidy_table(fit):
    report = protocol_report(fit)
    assert report["tidy"] is True
    assert report["table"] is True
    # No estimator methods on a plain result object.
    assert report["estimator"] is False


def test_supports_unknown_protocol_raises(fit):
    with pytest.raises(KeyError, match="unknown protocol"):
        supports(fit, "does-not-exist")


def test_install_methods_touched_many_classes():
    # Every result dataclass with to_dict() should have received the methods.
    assert ma._interop_classes >= 20
    # Calling it again is a no-op (idempotent guard).
    assert install_methods(ma.__dict__) == 0


def test_available_reports_booleans():
    info = available()
    assert set(info) >= {"pandas", "polars", "pyarrow", "sklearn",
                         "statsmodels", "xarray", "networkx"}
    assert all(isinstance(v, bool) for v in info.values())


# --------------------------------------------------------------------------- #
# tidy
# --------------------------------------------------------------------------- #
def test_tidy_intercept_only(fit):
    table = tidy(fit, backend="dict")
    assert table["term"] == ["intrcpt"]
    assert table["estimate"][0] == pytest.approx(float(fit.b[0]), rel=0, abs=1e-12)
    assert table["std.error"][0] == pytest.approx(float(fit.se[0]), rel=0, abs=1e-12)
    assert table["conf.low"][0] == pytest.approx(float(fit.ci_lb[0]), rel=0, abs=1e-12)
    assert table["conf.high"][0] == pytest.approx(float(fit.ci_ub[0]), rel=0, abs=1e-12)
    assert table["p.value"][0] == pytest.approx(float(fit.pval[0]), rel=0, abs=1e-12)


def test_tidy_coefficient_names(fit_mods):
    table = tidy(fit_mods, backend="dict")
    assert table["term"] == ["intrcpt", "V1"]
    assert len(table["estimate"]) == 2
    assert table["estimate"][1] == pytest.approx(float(fit_mods.b[1]), rel=0, abs=1e-12)


def test_tidy_as_method_matches_function(fit):
    assert fit.tidy(backend="dict") == tidy(fit, backend="dict")


def test_tidy_conf_level_override(fit):
    narrow = tidy(fit, conf_level=0.90, backend="dict")
    wide = tidy(fit, conf_level=0.99, backend="dict")
    assert narrow["conf.low"][0] > wide["conf.low"][0]
    assert narrow["conf.high"][0] < wide["conf.high"][0]
    # The point estimate is unchanged by the level.
    assert narrow["estimate"][0] == pytest.approx(wide["estimate"][0], rel=0, abs=1e-12)


def test_tidy_conf_int_false_omits_columns(fit):
    table = tidy(fit, conf_int=False, backend="dict")
    assert "conf.low" not in table
    assert "conf.high" not in table


def test_tidy_exponentiate(fit):
    table = tidy(fit, exponentiate=True, backend="dict")
    plain = tidy(fit, backend="dict")
    assert table["estimate"][0] == pytest.approx(np.exp(plain["estimate"][0]), rel=1e-12)


def test_tidy_rowwise_result():
    loo = ma.leave1out(YI, VI)
    table = tidy(loo, backend="dict")
    assert len(table["estimate"]) == len(YI)
    assert table["term"] == [str(i + 1) for i in range(len(YI))]


def test_tidy_scale_component():
    r = ma.rma(YI, VI, mods=DOSE, scale=DOSE)
    alpha = tidy(r, component="alpha", backend="dict")
    assert len(alpha["estimate"]) == len(np.atleast_1d(r.alpha))
    assert alpha["estimate"][0] == pytest.approx(float(r.alpha[0]), rel=1e-12, abs=1e-12)
    both = tidy(r, component="both", backend="dict")
    assert len(both["estimate"]) == len(tidy(r, backend="dict")["estimate"]) + len(alpha["estimate"])
    assert set(both["component"]) == {"location", "scale"}


def test_tidy_invalid_component(fit):
    with pytest.raises(ValueError, match="component"):
        tidy(fit, component="nope")


def test_tidy_rejects_object_without_estimates():
    with pytest.raises(TypeError, match="no estimate vector"):
        tidy(object())


# --------------------------------------------------------------------------- #
# glance
# --------------------------------------------------------------------------- #
def test_glance_core_fields(fit):
    row = glance(fit, backend="dict")
    assert row["k"] == [len(YI)]
    assert row["tau2"] == [pytest.approx(float(fit.tau2), abs=1e-12)]
    assert row["I2"] == [pytest.approx(float(fit.I2), abs=1e-12)]
    assert row["method"] == ["REML"]
    assert "logLik" in row and "AIC" in row and "BIC" in row


def test_glance_tau_is_sqrt_tau2(fit):
    row = glance(fit, backend="dict")
    assert row["tau"][0] == pytest.approx(np.sqrt(max(float(fit.tau2), 0.0)), rel=1e-12)


def test_glance_is_single_row(fit_mods):
    row = glance(fit_mods, backend="dict")
    assert all(len(v) == 1 for v in row.values())


def test_glance_as_method(fit):
    assert fit.glance(backend="dict") == glance(fit, backend="dict")


# --------------------------------------------------------------------------- #
# augment
# --------------------------------------------------------------------------- #
def test_augment_basic_columns(fit):
    table = augment(fit, backend="dict")
    assert len(table["study"]) == len(YI)
    assert table["yi"] == pytest.approx(list(YI))
    assert table[".fitted"] == pytest.approx(list(fit.fitted))
    assert table[".resid"] == pytest.approx(list(fit.residuals))


def test_augment_diagnostics(fit):
    table = augment(fit, diagnostics=True, backend="dict")
    assert ".hat" in table and ".std.resid" in table and ".cooksd" in table


def test_augment_without_diagnostics(fit):
    table = augment(fit, diagnostics=False, backend="dict")
    assert ".hat" not in table


def test_augment_rejects_objects_without_yi():
    with pytest.raises(TypeError, match="no per-study"):
        augment(object())


# --------------------------------------------------------------------------- #
# tables
# --------------------------------------------------------------------------- #
def test_to_frame_dict_and_records():
    cols = {"a": [1, 2], "b": [3.0, 4.0]}
    assert to_frame(cols, backend="dict")["a"] == [1, 2]
    records = to_frame(cols, backend="records")
    assert records == [{"a": 1, "b": 3.0}, {"a": 2, "b": 4.0}]


def test_to_frame_unknown_backend():
    assert "dict" in BACKENDS
    with pytest.raises(ValueError, match="unknown backend"):
        to_frame({"a": [1]}, backend="excel")


def test_normalize_columns_broadcasts_scalars():
    out = normalize_columns({"metric": "SMD", "value": [1, 2, 3]})
    assert out["metric"] == ["SMD", "SMD", "SMD"]


def test_normalize_columns_rejects_mismatch():
    with pytest.raises(ValueError, match="inconsistent lengths"):
        normalize_columns({"a": [1, 2], "b": [1, 2, 3]})


def test_to_table_method(fit):
    table = fit.to_table(backend="dict")
    assert isinstance(table, dict) and table


@pytest.mark.skipif(available()["pandas"], reason="pandas is installed")
def test_explicit_pandas_backend_requires_pandas():
    with pytest.raises(ImportError, match="pandas"):
        to_frame({"a": [1]}, backend="pandas")


@pytest.mark.skipif(not available()["pandas"], reason="pandas not installed")
def test_pandas_backend_roundtrip(fit):
    frame = tidy(fit, backend="pandas")
    assert list(frame.columns) == ["term", "estimate", "std.error", "statistic",
                                   "p.value", "conf.low", "conf.high"]


# --------------------------------------------------------------------------- #
# estimator protocol
# --------------------------------------------------------------------------- #
def test_estimator_params_roundtrip():
    est = MetaRegressor(method="DL", test="knha", level=0.9, measure="SMD")
    assert est.get_params() == {"method": "DL", "test": "knha",
                                "level": 0.9, "measure": "SMD"}
    est.set_params(method="REML")
    assert est.method == "REML"
    with pytest.raises(ValueError, match="invalid parameter"):
        est.set_params(bogus=1)


def test_estimator_alias():
    assert MetaEstimator is MetaRegressor


def test_estimator_matches_direct_rma():
    est = MetaRegressor().fit(DOSE, YI, vi=VI)
    direct = ma.rma(YI, VI, mods=DOSE)
    assert np.allclose(est.coef_, np.asarray(direct.b), atol=1e-12)
    assert est.tau2_ == pytest.approx(float(direct.tau2), abs=1e-12)
    assert est.n_features_in_ == 1
    assert est.intercept_ == pytest.approx(float(direct.b[0]), abs=1e-12)


def test_estimator_sample_weight_is_inverse_variance():
    est = MetaRegressor().fit(DOSE, YI, sample_weight=1.0 / VI)
    ref = MetaRegressor().fit(DOSE, YI, vi=VI)
    assert np.allclose(est.coef_, ref.coef_, atol=1e-12)


def test_estimator_predict_matches_fitted():
    est = MetaRegressor().fit(DOSE, YI, vi=VI)
    assert np.allclose(est.predict(DOSE), np.asarray(est.result_.fitted), atol=1e-12)


def test_estimator_predict_intervals():
    est = MetaRegressor().fit(DOSE, YI, vi=VI)
    point, lower, upper = est.predict(DOSE, interval="confidence")
    assert np.all(lower <= point) and np.all(point <= upper)
    _, plow, pup = est.predict(DOSE, interval="prediction", vi=VI)
    assert np.all(plow <= lower + 1e-12)
    assert np.all(pup >= upper - 1e-12)


def test_estimator_predict_prediction_needs_vi():
    est = MetaRegressor().fit(DOSE, YI, vi=VI)
    with pytest.raises(ValueError, match="'vi' is required"):
        est.predict(DOSE, interval="prediction")


def test_estimator_predict_invalid_interval():
    est = MetaRegressor().fit(DOSE, YI, vi=VI)
    with pytest.raises(ValueError, match="'interval'"):
        est.predict(DOSE, interval="bogus")


def test_estimator_score_range():
    est = MetaRegressor().fit(DOSE, YI, vi=VI)
    value = est.score(DOSE, YI, sample_weight=1.0 / VI)
    assert np.isfinite(value) and value <= 1.0 + 1e-12


def test_estimator_intercept_only():
    est = MetaRegressor().fit(None, YI, vi=VI)
    direct = ma.rma(YI, VI)
    assert np.allclose(est.coef_, np.asarray(direct.b), atol=1e-12)
    assert est.n_features_in_ == 0


def test_estimator_feature_names_from_frame():
    pd = pytest.importorskip("pandas")
    frame = pd.DataFrame({"dose": DOSE, "year": YEAR})
    est = MetaRegressor().fit(frame, YI, vi=VI)
    assert est.n_features_in_ == 2
    assert list(est.feature_names_in_) == ["dose", "year"]


def test_estimator_validation_errors():
    with pytest.raises(ValueError, match="rows"):
        MetaRegressor().fit(DOSE[:3], YI, vi=VI)
    with pytest.raises(ValueError, match="strictly positive"):
        MetaRegressor().fit(DOSE, YI, sample_weight=np.zeros_like(YI))
    with pytest.raises(ValueError, match="'vi'"):
        MetaRegressor().fit(DOSE, YI, vi=VI[:3])


def test_estimator_clone_style_reconstruction():
    est = MetaRegressor(method="DL").fit(DOSE, YI, vi=VI)
    clone = type(est)(**est.get_params())
    assert clone.get_params() == est.get_params()
    assert clone.result_ if hasattr(clone, "result_") else True


def test_estimator_repr_and_summary():
    est = MetaRegressor().fit(DOSE, YI, vi=VI)
    assert "MetaRegressor" in repr(est)
    assert isinstance(est.summary(), str)
    with pytest.raises(AttributeError, match="not fitted"):
        MetaRegressor().summary()


# --------------------------------------------------------------------------- #
# statsmodels-shaped results
# --------------------------------------------------------------------------- #
def test_as_results_core_attributes(fit_mods):
    res = as_results(fit_mods)
    assert np.allclose(res.params, np.asarray(fit_mods.b), atol=1e-12)
    assert np.allclose(res.bse, np.asarray(fit_mods.se), atol=1e-12)
    assert np.allclose(res.tvalues, np.asarray(fit_mods.zval), atol=1e-12)
    assert np.allclose(res.pvalues, np.asarray(fit_mods.pval), atol=1e-12)
    assert np.allclose(res.cov_params(), np.asarray(fit_mods.vb), atol=1e-12)
    assert res.nobs == int(fit_mods.k)
    assert res.df_resid == pytest.approx(float(fit_mods.k - fit_mods.p))
    assert res.df_model == pytest.approx(float(fit_mods.p - 1))


def test_as_results_conf_int_matches_model(fit_mods):
    res = as_results(fit_mods)
    ci = res.conf_int()
    assert ci.shape == (2, 2)
    assert np.allclose(ci[:, 0], np.asarray(fit_mods.ci_lb), atol=1e-10)
    assert np.allclose(ci[:, 1], np.asarray(fit_mods.ci_ub), atol=1e-10)


def test_as_results_alpha_argument(fit_mods):
    res = as_results(fit_mods)
    wide = res.conf_int(alpha=0.01)
    narrow = res.conf_int(alpha=0.10)
    assert np.all(wide[:, 0] < narrow[:, 0])
    assert np.all(wide[:, 1] > narrow[:, 1])


def test_as_results_information_criteria(fit):
    res = as_results(fit)
    assert np.isfinite(res.llf)
    assert np.isfinite(res.aic)
    assert np.isfinite(res.bic)


def test_as_results_fitted_and_resid(fit):
    res = as_results(fit)
    assert np.allclose(res.fittedvalues, np.asarray(fit.fitted), atol=1e-12)
    assert np.allclose(res.resid, np.asarray(fit.residuals), atol=1e-12)


def test_as_results_model_info(fit):
    res = as_results(fit)
    assert res.model.nobs == int(fit.k)
    assert res.model.k_constant == 1
    assert res.model.exog.shape == (int(fit.k), int(fit.p))
    assert "MetaModelInfo" in repr(res.model)


def test_as_results_exog_names(fit_mods):
    res = as_results(fit_mods)
    assert res.model.exog_names == ["intrcpt", "V1"]
    assert res.model.endog_names == "yi"


def test_as_results_as_pandas(fit_mods):
    frame = as_results(fit_mods).as_pandas()
    if isinstance(frame, dict):
        assert "coef" in frame
        return
    assert list(frame.index) == ["intrcpt", "V1"]
    assert list(frame.columns)[:2] == ["coef", "std err"]


def test_as_results_rejects_non_coefficient_objects():
    with pytest.raises(TypeError, match="no 'b'/'se'"):
        as_results(object())


def test_as_results_repr(fit):
    assert "MetaResults" in repr(as_results(fit))
    assert "MetaResults" in repr(as_results(ma.rma(YI, VI, mods=DOSE)))


# --------------------------------------------------------------------------- #
# optional labelled / graph backends
# --------------------------------------------------------------------------- #
def _netmeta_fit():
    TE = np.array([0.10, -0.20, 0.30, 0.15, -0.05, 0.25])
    seTE = np.array([0.10, 0.12, 0.11, 0.09, 0.13, 0.10])
    treat1 = np.array(["A", "A", "B", "B", "C", "A"])
    treat2 = np.array(["B", "C", "C", "D", "D", "D"])
    studlab = np.array([f"S{i + 1}" for i in range(TE.size)])
    return ma.netmeta(TE, seTE, treat1, treat2, studlab, sm="MD")


@pytest.mark.skipif(not available()["xarray"], reason="xarray not installed")
def test_to_xarray_dataset():
    from pimeta.interop import to_xarray

    ds = to_xarray(_netmeta_fit())
    assert "TE_random" in ds.data_vars
    assert ds["TE_random"].dims == ("treat1", "treat2")
    assert list(ds["TE_random"].coords["treat1"].values) == list(ds.attrs["trts"]) if "trts" in ds.attrs else True


@pytest.mark.skipif(not available()["xarray"], reason="xarray not installed")
def test_to_xarray_requires_trts():
    from pimeta.interop import to_xarray

    with pytest.raises(TypeError, match="trts"):
        to_xarray(object())


@pytest.mark.skipif(not available()["networkx"], reason="networkx not installed")
def test_to_networkx_graph():
    from pimeta.interop import to_networkx

    graph = to_networkx(_netmeta_fit())
    assert set(graph.nodes) == {"A", "B", "C", "D"}
    assert graph.number_of_edges() >= 3
    edge = next(iter(graph.edges(data=True)))[2]
    assert "studies" in edge and "weight" in edge


@pytest.mark.skipif(not available()["networkx"], reason="networkx not installed")
def test_to_networkx_weight_modes():
    from pimeta.interop import to_networkx

    with pytest.raises(ValueError, match="unknown weight mode"):
        to_networkx(_netmeta_fit(), weight="bogus")


@pytest.mark.skipif(not available()["networkx"], reason="networkx not installed")
def test_to_networkx_requires_comparisons():
    from pimeta.interop import to_networkx

    with pytest.raises(TypeError, match="trts"):
        to_networkx(object())


@pytest.mark.skipif(available()["xarray"], reason="xarray is installed")
def test_xarray_missing_dependency_message():
    from pimeta.interop import to_xarray

    with pytest.raises(ImportError, match="xarray"):
        to_xarray(_netmeta_fit())


@pytest.mark.skipif(available()["networkx"], reason="networkx is installed")
def test_networkx_missing_dependency_message():
    from pimeta.interop import to_networkx

    with pytest.raises(ImportError, match="networkx"):
        to_networkx(_netmeta_fit())


# --------------------------------------------------------------------------- #
# sklearn integration (only when scikit-learn happens to be installed)
# --------------------------------------------------------------------------- #
@pytest.mark.skipif(not available()["sklearn"], reason="scikit-learn not installed")
def test_sklearn_clone_and_grid_search():
    from sklearn.base import clone
    from sklearn.model_selection import GridSearchCV

    est = MetaRegressor()
    assert clone(est).get_params() == est.get_params()
    search = GridSearchCV(est, {"method": ["DL", "REML"]}, cv=3)
    search.fit(DOSE.reshape(-1, 1), YI, sample_weight=1.0 / VI)
    assert search.best_params_["method"] in ("DL", "REML")


@pytest.mark.skipif(not available()["sklearn"], reason="scikit-learn not installed")
def test_sklearn_pipeline_and_cross_val_score():
    from sklearn.base import is_regressor
    from sklearn.model_selection import KFold, cross_val_score
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler

    assert is_regressor(MetaRegressor())
    pipe = Pipeline([("scale", StandardScaler()), ("meta", MetaRegressor())])
    pipe.fit(DOSE.reshape(-1, 1), YI, meta__sample_weight=1.0 / VI)
    scores = cross_val_score(pipe, DOSE.reshape(-1, 1), YI, cv=KFold(3))
    assert scores.shape == (3,)


# --------------------------------------------------------------------------- #
# statsmodels integration (only when statsmodels happens to be installed)
# --------------------------------------------------------------------------- #
@pytest.mark.skipif(not available()["statsmodels"], reason="statsmodels not installed")
def test_statsmodels_summary_col_accepts_adapter(fit_mods):
    from statsmodels.iolib.summary2 import summary_col

    text = str(summary_col([as_results(fit_mods)], stars=True))
    assert "intrcpt" in text
    assert "V1" in text
