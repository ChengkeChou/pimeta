"""DTA bivariate model, HSROC, SROC and AUC -- parity with ``mada`` 0.5.12.

Every reference number below was produced by ``mada`` 0.5.12 on R 4.6.1 with::

    fit <- mada::reitsma(dat, method = "reml")     # or "ml"
    coef(fit); fit$Psi; as.numeric(fit$logLik); AIC(fit); BIC(fit)
    AUC(fit); mada:::calc_hsroc_coef(fit)

The datasets are mada's own ``AuditC`` (14 studies) and ``Dementia``
(33 studies).  ``Dementia`` matters because its HSROC asymmetry parameter is
positive, so a sign error in the reparameterisation cannot slip through.

The likelihood is normalised the same way ``mvmeta`` normalises it, and
``loglik`` carries mada's Jacobian correction, because that is the number its
AIC is built on.
"""

from __future__ import annotations

import warnings

import numpy as np
import pytest

import pimeta as ma
from pimeta.dta import (
    DtaCorrectionWarning,
    DtaData,
    HsrocResult,
    ReitsmaResult,
    SrocResult,
    auc,
    escalc_dta,
    hsroc,
    reitsma,
    sroc,
)


def _mada(data, method="reml", **kwargs):
    """Fit exactly the way mada does, for comparing numbers with R."""
    return reitsma(data, method=method, correction_control="all", **kwargs)


# Most tests do not care that a correction fired; two of them assert on it
# explicitly with ``pytest.warns``, which sets its own filter.
pytestmark = pytest.mark.filterwarnings(
    "ignore::pimeta.dta.DtaCorrectionWarning"
)

#: mada::AuditC
AUDITC = {
    "TP": [47, 126, 19, 36, 130, 84, 68, 752, 59, 142, 137, 57, 34, 152],
    "FN": [9, 51, 10, 3, 19, 2, 0, 0, 5, 50, 24, 3, 1, 51],
    "FP": [101, 272, 12, 78, 211, 68, 112, 3226, 55, 571, 107, 103, 21, 88],
    "TN": [738, 1543, 192, 276, 959, 89, 423, 2977, 136, 2788, 358, 437, 56, 264],
}

#: mada::Dementia
DEMENTIA = {
    "TP": [66, 118, 48, 134, 24, 68, 64, 282, 14, 262, 144, 184, 22, 112, 152,
           30, 32, 10, 708, 182, 60, 74, 28, 40, 318, 388, 118, 44, 124, 26,
           74, 38, 78],
    "FN": [4, 12, 20, 8, 6, 16, 18, 64, 2, 20, 18, 34, 0, 0, 82, 26, 6, 4, 88,
           108, 30, 24, 12, 6, 52, 116, 66, 8, 46, 44, 32, 46, 34],
    "FP": [240, 10, 64, 28, 44, 48, 0, 20, 44, 30, 30, 34, 152, 590, 126, 26,
           4, 12, 1438, 18, 24, 16, 26, 76, 174, 16, 2, 34, 98, 4, 2, 0, 46],
    "TN": [870, 110, 990, 152, 292, 154, 72, 286, 286, 178, 124, 52, 140, 2092,
           1010, 236, 248, 334, 10448, 184, 74, 144, 210, 528, 578, 54, 44,
           396, 310, 172, 226, 440, 376],
}

AUDITC_REF = {
    "reml": {
        "coef": [2.099742247463, -1.263690935269],
        "Psi": [1.379582891589, 0.640294227640, 0.407208155633],
        "loglik": 31.563960673044,
        "loglik_raw": -33.909597129937,
        "AIC": -53.127921346089,
        "BIC": -46.466898795213,
        "auc": 0.886679347847,
        "pauc": 0.861191557507,
        "hsroc": [-0.083379254875, 3.262132642713, -0.610105993194,
                  0.694906171358, 0.218447774872],
        "sens": 0.890878124098,
        "fpr": 0.220339172665,
    },
    "ml": {
        "coef": [2.076259082149, -1.262447087459],
        "Psi": [1.223841766866, 0.583232920168, 0.374882421620],
        "loglik": 33.073238516259,
        "loglik_raw": -32.400319286722,
        "AIC": -56.146477032517,
        "BIC": -49.485454481641,
        "auc": 0.886293795818,
        "pauc": 0.860086061394,
        "hsroc": [-0.076164129580, 3.241584924511, -0.591568872410,
                  0.630289150614, 0.188224921782],
        "sens": 0.888574182676,
        "fpr": 0.220552927387,
    },
}

DEMENTIA_REF = {
    "reml": {
        "coef": [1.317331184285, -2.052318130442],
        "Psi": [0.639937463406, 0.433005926867, 0.877243668276],
        "loglik": 61.202660714713,
        "loglik_raw": -91.096651843880,
        "AIC": -112.405321429427,
        "BIC": -101.457047719295,
        "auc": 0.904415681260,
        "pauc": 0.843023116192,
        "hsroc": [-0.235644396228, 3.322114359728, 0.157707169295,
                  0.591129807061, 0.632495520775],
        "sens": 0.788737342451,
        "fpr": 0.113818356579,
    },
    "ml": {
        "coef": [1.315280675184, -2.048708866086],
        "Psi": [0.615413533658, 0.416492063105, 0.839265038040],
        "loglik": 63.188928075859,
        "loglik_raw": -89.110384482735,
        "AIC": -116.377856151717,
        "BIC": -105.429582441585,
        "auc": 0.904164483873,
        "pauc": 0.842520803761,
        "hsroc": [-0.237234570966, 3.317171941460, 0.155116049918,
                  0.567583986247, 0.604367692567],
        "sens": 0.788395462294,
        "fpr": 0.114182908202,
    },
}


# --------------------------------------------------------------------------- #
# data preparation
# --------------------------------------------------------------------------- #
def test_the_default_correction_touches_only_the_studies_that_need_it():
    data = escalc_dta(AUDITC)
    # Two studies have FN == 0.  Correcting the other twelve as well is what
    # mada does; it is not what this package does, and not what the method
    # says.
    assert data.corrected.sum() == 2
    assert data.tp[0] == 47
    assert data.fn[6] == pytest.approx(0.5)


def test_mada_global_correction_is_available_but_has_to_be_asked_for():
    with pytest.warns(DtaCorrectionWarning, match="all 14 studies"):
        data = escalc_dta(AUDITC, correction_control="all")
    assert data.corrected.all()
    assert data.tp[0] == pytest.approx(47.5)


def test_a_correction_is_never_applied_silently():
    with pytest.warns(DtaCorrectionWarning, match="2 of 14 studies"):
        escalc_dta(AUDITC)
    # No zero cell, nothing to say.
    clean = {"TP": [10, 20], "FN": [2, 3], "FP": [4, 5], "TN": [40, 50]}
    with warnings.catch_warnings():
        warnings.simplefilter("error", DtaCorrectionWarning)
        escalc_dta(clean)


def test_correction_control_single_touches_only_the_broken_studies():
    data = escalc_dta(AUDITC, correction_control="single")
    assert data.corrected.sum() == 2
    assert data.tp[0] == 47                       # untouched
    assert data.fn[6] == pytest.approx(0.5)       # corrected


def test_correction_control_none_leaves_the_counts_alone():
    data = escalc_dta(AUDITC, correction_control="none")
    assert not data.corrected.any()
    assert data.fn[6] == 0


def test_uncorrected_sensitivity_and_fpr_are_kept_for_the_partial_auc():
    data = escalc_dta(AUDITC, correction_control="all")
    assert data.fpr_raw[0] == pytest.approx(101 / (101 + 738))
    assert data.fpr[0] == pytest.approx(101.5 / (101.5 + 738.5))
    assert data.fpr_raw[0] != data.fpr[0]


def test_dta_data_rejects_mismatched_lengths():
    with pytest.raises(ValueError, match="same length"):
        escalc_dta([1, 2], [1], [1, 2], [1, 2])


def test_dta_data_rejects_a_bad_correction_mode():
    with pytest.raises(ValueError, match="correction_control"):
        escalc_dta(AUDITC, correction_control="sometimes")


def test_counts_can_arrive_as_a_mapping_or_as_arguments():
    from_mapping = escalc_dta(AUDITC)
    from_args = escalc_dta(AUDITC["TP"], AUDITC["FN"], AUDITC["FP"], AUDITC["TN"])
    assert np.allclose(from_mapping.sens, from_args.sens)


# --------------------------------------------------------------------------- #
# the bivariate model, against mada
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("dataset,ref", [
    ("auditc", AUDITC_REF),
    ("dementia", DEMENTIA_REF),
])
@pytest.mark.parametrize("method", ["reml", "ml"])
def test_reitsma_matches_mada(dataset, ref, method):
    data = AUDITC if dataset == "auditc" else DEMENTIA
    expected = ref[method]
    fit = _mada(data, method)

    # The means agree to 1e-7.  The variance components are allowed 1e-5
    # because mvmeta stops earlier than this implementation does: evaluating
    # the likelihood at mada's own parameters gives the identical value (see
    # test_likelihood_agrees_at_madas_optimum), so the gap is the stopping
    # rule, not a different model.
    #
    # mada fits logit(sensitivity) and logit(FPR); this package reports
    # logit(specificity), so the second component and the covariance come out
    # negated.  mu_fpr is the value mada prints.
    assert fit.mu_fpr == pytest.approx(expected["coef"], rel=1e-6)
    assert fit.mu[0] == pytest.approx(expected["coef"][0], rel=1e-6)
    assert fit.mu[1] == pytest.approx(-expected["coef"][1], rel=1e-6)
    assert fit.Psi_fpr[0, 1] == pytest.approx(expected["Psi"][1], rel=1e-5)
    assert fit.Psi[0, 0] == pytest.approx(expected["Psi"][0], rel=1e-5)
    assert fit.Psi[0, 1] == pytest.approx(-expected["Psi"][1], rel=1e-5)
    assert fit.Psi[1, 1] == pytest.approx(expected["Psi"][2], rel=1e-5)
    assert fit.loglik == pytest.approx(expected["loglik"], rel=1e-6)
    assert fit.loglik_raw == pytest.approx(expected["loglik_raw"], rel=1e-6)
    assert fit.AIC == pytest.approx(expected["AIC"], rel=1e-6)
    assert fit.BIC == pytest.approx(expected["BIC"], rel=1e-6)


@pytest.mark.parametrize("dataset,ref", [
    ("auditc", AUDITC_REF),
    ("dementia", DEMENTIA_REF),
])
@pytest.mark.parametrize("method", ["reml", "ml"])
def test_likelihood_agrees_at_madas_optimum(dataset, ref, method):
    """The model is the same one mada fits, to nine decimal places.

    This is the strong claim; the parameter comparison above is the weak one.
    Evaluating our profiled likelihood at *mada's* reported variance
    components reproduces mada's own -2 log-likelihood, which can only happen
    if the likelihood -- sampling variances, continuity correction, scale and
    all -- is the same function.
    """
    from pimeta.dta import _jacobian_term, _logit, _neg2_loglik

    data = AUDITC if dataset == "auditc" else DEMENTIA
    expected = ref[method]
    prepared = escalc_dta(data, correction_control="all")
    y = np.column_stack([_logit(prepared.sens), _logit(prepared.fpr)])
    V = np.zeros((prepared.k, 2, 2))
    V[:, 0, 0] = prepared.var_sens
    V[:, 1, 1] = prepared.var_fpr

    psi = expected["Psi"]
    sd1, sd2 = np.sqrt(psi[0]), np.sqrt(psi[2])
    rho = psi[1] / (sd1 * sd2)
    params = np.array([np.log(sd1), np.log(sd2), np.arctanh(rho)])

    ours = _neg2_loglik(params, y, V, method)
    # mada's logLik is -2ll/2 plus the Jacobian and the normalising constant;
    # invert that to recover its -2ll.
    dof = 2 * prepared.k - 2 if method == "reml" else 2 * prepared.k
    mada_raw = expected["loglik"] - _jacobian_term(prepared)
    mada_neg2ll = -2.0 * (mada_raw + 0.5 * dof * np.log(2.0 * np.pi))
    assert ours == pytest.approx(mada_neg2ll, abs=1e-6)


@pytest.mark.parametrize("dataset,ref", [
    ("auditc", AUDITC_REF),
    ("dementia", DEMENTIA_REF),
])
@pytest.mark.parametrize("method", ["reml", "ml"])
def test_hsroc_matches_madas_reparameterisation(dataset, ref, method):
    data = AUDITC if dataset == "auditc" else DEMENTIA
    expected = ref[method]
    fit = _mada(data, method)
    h = hsroc(fit)
    got = [h.Theta, h.Lambda, h.beta, h.sigma2theta, h.sigma2alpha]
    # Derived quantities amplify the variance-component gap: beta is a log
    # ratio of the two between-study SDs, so a 1e-6 difference in each become
    # ~1e-5 here.
    assert got == pytest.approx(expected["hsroc"], rel=1e-4, abs=1e-7)


@pytest.mark.parametrize("dataset,ref", [
    ("auditc", AUDITC_REF),
    ("dementia", DEMENTIA_REF),
])
@pytest.mark.parametrize("method", ["reml", "ml"])
def test_summary_point_matches_mada(dataset, ref, method):
    data = AUDITC if dataset == "auditc" else DEMENTIA
    expected = ref[method]
    fit = _mada(data, method)
    assert fit.summary_sens == pytest.approx(expected["sens"], rel=1e-6)
    assert fit.summary_fpr == pytest.approx(expected["fpr"], rel=1e-6)
    assert fit.summary_spec == pytest.approx(1 - expected["fpr"], rel=1e-6)


@pytest.mark.parametrize("dataset,ref", [
    ("auditc", AUDITC_REF),
    ("dementia", DEMENTIA_REF),
])
@pytest.mark.parametrize("method", ["reml", "ml"])
def test_auc_matches_mada(dataset, ref, method):
    data = AUDITC if dataset == "auditc" else DEMENTIA
    expected = ref[method]
    got = auc(_mada(data, method))
    assert got["auc"] == pytest.approx(expected["auc"], abs=1e-5)
    assert got["pauc"] == pytest.approx(expected["pauc"], abs=1e-5)


# --------------------------------------------------------------------------- #
# SROC
# --------------------------------------------------------------------------- #
def test_sroc_reproduces_madas_first_grid_point():
    # mada::sroc(fit) evaluated at fpr = 0.01 gives 0.01742666 for AuditC.
    curve = sroc(_mada(AUDITC))
    assert isinstance(curve, SrocResult)
    assert curve.fpr.size == 99
    assert curve.tpr[0] == pytest.approx(0.01742666, abs=1e-7)


def test_sroc_is_monotone_and_inside_the_unit_square():
    curve = sroc(reitsma(AUDITC))
    assert np.all(np.diff(curve.tpr) > 0)
    assert np.all((curve.tpr >= 0) & (curve.tpr <= 1))


def test_sroc_passes_near_the_summary_point():
    fit = reitsma(AUDITC)
    curve = sroc(fit)
    at_summary = float(np.interp(fit.summary_fpr, curve.fpr, curve.tpr))
    assert at_summary == pytest.approx(fit.summary_sens, abs=0.01)


def test_sroc_rejects_fpr_outside_the_open_unit_interval():
    fit = reitsma(AUDITC)
    with pytest.raises(ValueError, match="strictly between"):
        sroc(fit, fpr=[0.0, 0.5])


def test_sroc_rejects_an_unknown_type():
    fit = reitsma(AUDITC)
    with pytest.raises(ValueError, match="ruttergatsonis"):
        sroc(fit, type="binormal")


def test_auc_needs_a_reasonable_grid():
    fit = reitsma(AUDITC)
    with pytest.raises(ValueError, match="at least 10"):
        auc(fit, fpr=[0.1, 0.2, 0.3])


# --------------------------------------------------------------------------- #
# structural invariants
# --------------------------------------------------------------------------- #
def test_hsroc_accepts_raw_counts_and_round_trips():
    direct = hsroc(AUDITC)
    via_fit = hsroc(reitsma(AUDITC))
    assert direct.Theta == pytest.approx(via_fit.Theta)
    assert hsroc(direct) is direct


def test_hsroc_and_bivariate_share_the_likelihood():
    # Harbord et al.: the two models are reparameterisations, so the
    # likelihood cannot depend on which one you report.
    fit = reitsma(AUDITC)
    h = hsroc(fit)
    assert h.fit is fit
    assert fit.loglik == pytest.approx(h.fit.loglik)


def test_beta_zero_gives_a_symmetric_sroc():
    fit = reitsma(AUDITC)
    h = hsroc(fit)
    # Rebuild the curve by hand with beta forced to 0 and check the shape.
    from pimeta.dta import _expit, _logit

    grid = np.array([0.05, 0.2, 0.5, 0.8])
    sym = _expit(h.Lambda + _logit(grid))
    dor = h.dor
    assert np.allclose(sym / (1 - sym), dor * grid / (1 - grid))


def test_dor_is_exp_of_lambda():
    h = hsroc(reitsma(AUDITC))
    assert h.dor == pytest.approx(float(np.exp(h.Lambda)))


def test_summary_intervals_bracket_the_point():
    fit = reitsma(AUDITC)
    lo, hi = fit.sens_ci
    assert lo < fit.summary_sens < hi
    lo, hi = fit.fpr_ci
    assert lo < fit.summary_fpr < hi


def test_coef_table_has_one_row_per_coefficient():
    table = reitsma(AUDITC).coef_table()
    assert table["model"] == ["reitsma", "reitsma"]
    assert len(table["term"]) == 2
    assert table["ci_lb"][0] < table["estimate"][0] < table["ci_ub"][0]


def test_to_dict_is_json_ready():
    from pimeta.record.jsonio import dumps, loads

    payload = loads(dumps(reitsma(AUDITC).to_dict()))
    assert payload["k"] == 14
    assert payload["method"] == "reml"
    assert len(payload["Psi"]) == 2


def test_hsroc_to_dict_is_json_ready():
    from pimeta.record.jsonio import dumps, loads

    payload = loads(dumps(hsroc(reitsma(AUDITC)).to_dict()))
    assert payload["Lambda"] > 0
    assert "DOR" in payload


def test_dta_data_accepts_a_frame_like_object():
    class Frame:
        columns = ["TP", "FN", "FP", "TN"]

        def __getitem__(self, key):
            return AUDITC[key]

    data = escalc_dta(Frame())
    assert isinstance(data, DtaData)
    assert data.k == 14


def test_reitsma_rejects_an_unknown_method():
    with pytest.raises(ValueError, match="reml"):
        reitsma(AUDITC, method="vc")


def test_reitsma_needs_at_least_two_studies():
    tiny = {"TP": [10], "FN": [2], "FP": [3], "TN": [40]}
    with pytest.raises(ValueError, match="at least two studies"):
        reitsma(tiny)


def test_result_objects_expose_sroc_and_auc():
    fit = reitsma(AUDITC)
    assert isinstance(fit, ReitsmaResult)
    assert isinstance(fit.sroc(), SrocResult)
    assert fit.auc()["auc"] == pytest.approx(auc(fit)["auc"])
    h = hsroc(fit)
    assert isinstance(h, HsrocResult)
    assert h.auc()["auc"] == pytest.approx(auc(fit)["auc"])


# --------------------------------------------------------------------------- #
# integration with pimeta.record
# --------------------------------------------------------------------------- #
def _recorded(tmp_path, **kwargs):
    from pimeta.record import save_record

    fit = reitsma(AUDITC)
    path = tmp_path / "dta.json"
    save_record(path, fit, data=AUDITC, mode="bundle", function="reitsma",
                call={"method": "reml"}, pass_data=True, **kwargs)
    return fit, path


def test_a_dta_fit_round_trips_through_a_bundle_record(tmp_path):
    from pimeta.record import load_record

    fit, path = _recorded(tmp_path)
    loaded = load_record(path)

    assert loaded.kind == "reitsma"
    assert loaded.inputs["mode"] == "bundle"
    assert loaded.inputs["n_rows"] == 14
    assert loaded.analysis["pass_data"] is True
    assert loaded.environment["pimeta"] == ma.__version__

    # The bundled counts still hash to the recorded digests.
    fresh = loaded.verify()
    assert fresh["analysis"] == loaded.fingerprint["analysis"]

    # Re-fitting from the file reproduces the coefficients exactly.
    again = loaded.replay()
    assert again.mu == pytest.approx(fit.mu, rel=1e-12)
    assert again.Psi.ravel() == pytest.approx(fit.Psi.ravel(), rel=1e-12)
    assert again.AIC == pytest.approx(fit.AIC, rel=1e-12)


def test_replay_catches_an_edited_coefficient(tmp_path):
    from pimeta.record import RecordMismatch, load_record

    _, path = _recorded(tmp_path)
    loaded = load_record(path)
    loaded.results["values"]["mu"] = [9.9, -9.9]
    with pytest.raises(RecordMismatch, match="replayed"):
        loaded.replay()


def test_the_counts_are_not_copied_in_digest_mode(tmp_path):
    from pimeta.record import RecordError, build_record

    fit = reitsma(AUDITC)
    record = build_record(fit, data=AUDITC, mode="digest", function="reitsma",
                          call={"method": "reml"}, pass_data=True)
    assert "data" not in record.inputs
    with pytest.raises(RecordError, match="digest"):
        record.verify()
    # ... but the record can still be checked against the original table.
    assert record.verify(AUDITC)["source"] == record.fingerprint["source"]


def test_pass_data_defaults_to_false_when_no_column_names_are_used():
    from pimeta.record import build_record

    fit = reitsma(AUDITC)
    auto = build_record(fit, data=AUDITC, mode="bundle", function="reitsma",
                        call={"method": "reml"})
    # reitsma() takes the whole table, not column names, so the inference
    # cannot see that the data is needed -- which is why pass_data exists.
    assert auto.analysis["pass_data"] is False

    forced = build_record(fit, data=AUDITC, mode="bundle", function="reitsma",
                          call={"method": "reml"}, pass_data=True)
    assert forced.analysis["pass_data"] is True
    assert forced.replay().mu == pytest.approx(fit.mu, rel=1e-12)


def test_a_dta_fit_goes_into_an_estimates_store():
    from pimeta.record import EstimatesStore

    store = EstimatesStore("auditc")
    store.store("reml", _mada(AUDITC))
    store.store("ml", _mada(AUDITC, "ml"))

    table = store.compare()
    assert table["model"] == ["reml", "reml", "ml", "ml"]
    assert table["term"][0] == "(Intercept): logit(sens)"
    assert table["estimate"][0] == pytest.approx(2.0997422, abs=1e-6)
    assert table["estimate"][2] == pytest.approx(2.0762591, abs=1e-6)
    assert len(table["pval"]) == 4

