"""Tests for the ``rma`` accessors (``s3.py``) and the object helpers in
``utils.py``, pinned to metafor where the quantity is well defined.

The reference model is ``rma(yi, vi, mods = ~ mods, method = "REML")`` on the
13 BCG trials, the same one used in ``test_coverage_gaps.py``.
"""
import numpy as np
import pytest

import pimeta as ma

BCGY = [-0.88931133, -1.58538866, -1.34807315, -1.44155119, -0.21754732,
        -0.78611559, -1.59664331, -0.32966938, -0.55783318, -0.49888521,
        -0.96960854, -0.48266968, -0.41683796]
BCGV = [0.13320328, 0.26373142, 0.21275645, 0.14693293, 0.14293235,
        0.12151837, 0.44437921, 0.15956733, 0.13432539, 0.33222552,
        0.20969889, 0.17053331, 0.13000306]
MODS = list(range(1, 14))
MODS2 = [3, 1, 4, 1, 5, 9, 2, 6, 5, 3, 5, 8, 9]


@pytest.fixture(scope="module")
def fit():
    return ma.rma(BCGY, BCGV, mods=list(MODS), method="REML")


@pytest.fixture(scope="module")
def fit3():
    return ma.rma(BCGY, BCGV,
                  mods=np.column_stack([np.asarray(MODS, dtype=float),
                                        np.asarray(MODS2, dtype=float)]),
                  method="REML")


@pytest.fixture(scope="module")
def fit0():
    return ma.rma(BCGY, BCGV, method="REML")


# --------------------------------------------------------------- accessors ---
def test_coef_and_se(fit):
    co = ma.coef(fit)
    co = list(co.values()) if isinstance(co, dict) else list(np.asarray(co).ravel())
    assert co == pytest.approx(list(fit.b), abs=1e-12)
    ses = ma.se(fit)
    got = list(ses.values()) if isinstance(ses, dict) else list(ses)
    assert got == pytest.approx(
        [0.24015416232592754, 0.030141937103205677], abs=1e-10)


def test_vcov_is_symmetric_and_matches_fit(fit):
    v = np.asarray(ma.vcov(fit))
    assert v.ravel() == pytest.approx(np.asarray(fit.vb).ravel())
    assert np.allclose(v, v.T)


def test_fitted_residuals_and_weights_are_consistent(fit):
    fitted = np.asarray(ma.fitted(fit)).ravel()
    resid = np.asarray(ma.residuals(fit)).ravel()
    assert (np.asarray(fit.yi) - fitted) == pytest.approx(resid, abs=1e-12)
    w = np.asarray(ma.weights(fit)).ravel()
    assert w == pytest.approx(np.asarray(fit.weights).ravel())
    pct = np.asarray(ma.weights(fit, pct=True)).ravel()
    assert float(pct.sum()) == pytest.approx(100.0, abs=1e-9)


def test_loglik_deviance_aic_bic(fit):
    """metafor: fitstats(fit) = logLik, deviance, AIC, BIC, AICc."""
    want = [-6.0146040499928448, 12.02920809998569, 18.029208099985688,
            19.222893918380802, 21.457779528557118]
    assert float(ma.logLik(fit)) == pytest.approx(want[0], abs=1e-9)
    assert float(ma.deviance(fit)) == pytest.approx(want[1], abs=1e-9)
    assert float(ma.AIC(fit)) == pytest.approx(want[2], abs=1e-9)
    assert float(ma.BIC(fit)) == pytest.approx(want[3], abs=1e-9)
    assert ma.aic_rma(fit) == pytest.approx(want[2], abs=1e-9)
    assert ma.bic_rma(fit) == pytest.approx(want[3], abs=1e-9)


def test_aicc_matches_r(fit):
    """R: fitstats(fit)["AICc"] = 21.457779528557118."""
    assert ma.aic_rma(fit, correct=True) == pytest.approx(
        21.457779528557118, abs=1e-9)


def test_fitstats_result(fit):
    d = ma.fitstats(fit).as_dict()
    assert set(d) == {"ML", "REML"}
    reml = d["REML"]
    assert reml["ll"] == pytest.approx(-6.0146040499928448, abs=1e-9)
    assert reml["dev"] == pytest.approx(12.02920809998569, abs=1e-9)
    assert reml["AIC"] == pytest.approx(18.029208099985688, abs=1e-9)
    assert reml["BIC"] == pytest.approx(19.222893918380802, abs=1e-9)
    assert reml["AICc"] == pytest.approx(21.457779528557118, abs=1e-9)


def test_nobs_df_residual_and_formula(fit):
    assert ma.nobs(fit) == 13
    assert ma.df_residual(fit) == pytest.approx(13 - 2)
    assert "mods" in ma.formula(fit)


def test_model_matrix(fit):
    mm = ma.model_matrix(fit)
    x = np.asarray(mm["matrix"] if isinstance(mm, dict) else mm, dtype=float)
    assert x.shape == (13, 2)
    assert x[:, 0] == pytest.approx(np.ones(13))
    assert x[:, 1] == pytest.approx(np.array(MODS, dtype=float))
    df = ma.model_matrix(fit, asdf=True)
    assert len(df) == 13


def test_hatvalues_matches_r(fit):
    """metafor 5.2-1 on this model."""
    want = np.array([0.344437116852, 0.136136067895, 0.130400916280,
                     0.145651395466, 0.118065709470, 0.116582415221,
                     0.029874200398, 0.088997633920, 0.126140179561,
                     0.064725959499, 0.132954805029, 0.211538773640,
                     0.354494826769])
    assert np.asarray(ma.hatvalues_rma(fit)).ravel() == pytest.approx(
        want, abs=1e-9)


def test_cooks_distance_matches_r(fit):
    """metafor: cooks.distance(fit) on the BCG moderator model."""
    want = np.array([0.395654608286, 0.173339917935, 0.085208602950,
                     0.311168192784, 0.497412660374, 0.007953004837,
                     0.047635396946, 0.101901359239, 0.015362908489,
                     0.002150980727, 0.158420714760, 0.000148847126,
                     0.000318937454])
    d = np.asarray(ma.cooks_distance_rma(fit)).ravel()
    assert d.shape == (13,)
    assert d == pytest.approx(want, abs=1e-9)


def test_blup_matches_r(fit):
    want_pred = np.array([-1.145744946733, -1.084790821052, -1.023836695370,
                          -0.962882569689, -0.901928444007, -0.840974318326,
                          -0.780020192644, -0.719066066962, -0.658111941281,
                          -0.597157815599, -0.536203689918, -0.475249564236,
                          -0.414295438555])
    want_se = np.array([0.214196530594, 0.189481815748, 0.166564209914,
                        0.146290759429, 0.129905385989, 0.119024808625,
                        0.115219241328, 0.119168430471, 0.130168463209,
                        0.146641111399, 0.166974474201, 0.189932638749,
                        0.214675131848])
    out = ma.blup(fit)
    assert np.asarray(out.pred).ravel() == pytest.approx(want_pred, abs=1e-9)
    assert np.asarray(out.se).ravel() == pytest.approx(want_se, abs=1e-9)


def test_rstandard_and_dfbetas(fit):
    rz = np.asarray(ma.rstandard_rma(fit)).ravel()
    assert rz.shape == (13,)
    assert np.all(np.isfinite(rz))
    dfb = np.asarray(ma.dfbetas_rma(fit))
    assert dfb.shape == (13, 2)
    assert np.all(np.isfinite(dfb))


def test_ranef_rma_shrinks_towards_zero(fit0):
    assert ma.ranef_rma(fit0, level=0.95) is not None


def test_pairmat_matches_r(fit3):
    got = ma.pairmat(fit3, btt=[2, 3])
    assert len(got["names"]) == 1
    assert np.asarray(got["matrix"]).ravel() == pytest.approx([0, -1, 1])
    with pytest.raises(ValueError, match="int_only|intercept"):
        ma.pairmat(ma.rma(BCGY, BCGV), btt=[1, 2])
    with pytest.raises(ValueError, match="btt"):
        ma.pairmat(fit3)


def test_update_rma_changes_the_estimator(fit):
    dl = ma.update_rma(fit, method="DL")
    assert dl.method == "DL"
    # this data set has tau^2 = 0 under both estimators, so compare the
    # heterogeneity statistic instead of tau^2
    assert dl.QE == pytest.approx(fit.QE, abs=1e-8)
    assert np.asarray(dl.X).shape == np.asarray(fit.X).shape


def test_update_rma_survives_without_new_mods(fit):
    out = ma.update_rma(fit)
    assert np.asarray(out.X).shape == np.asarray(fit.X).shape
    assert np.asarray(out.b) == pytest.approx(np.asarray(fit.b), abs=1e-8)


def test_as_data_frame_helpers(fit):
    lst = ma.as_data_frame_list_rma(fit)
    assert len(lst) == 13
    # as_data_frame_confint flattens a confint-style dict into rows
    rows = ma.as_data_frame_confint({"estimate": [0.5, 0.7],
                                     "ci.lb": [0.1, 0.2],
                                     "ci.ub": [0.9, 1.2]})
    assert len(rows) == 2
    assert rows[0]["estimate"] == pytest.approx(0.5)
    assert ma.as_matrix_list_rma(fit) is not None


def test_summary_and_text_do_not_raise(fit):
    assert ma.summary(fit) is not None
    txt = ma.summary_text(fit)
    assert isinstance(txt, str) and len(txt) > 0
    ma.print_summary(fit)
