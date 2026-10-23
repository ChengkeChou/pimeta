"""One-stage individual-participant models: closed-form and quadrature.

Reference values come from R 4.6.1 with `lme4` 2.0.6, on the same deterministically
constructed participant-level data that `test_ipd.py` uses:

    lmer(ycont ~ treat + age_c + (1 | study))              # REML and ML
    glmer(ybin ~ treat + age_c + (1 | study), binomial)

The patient rows are built arithmetically, so R and Python construct identical
data with nothing to ship and no RNG stream to drift.

The binary reference is a *singular* fit: `glmer` reports a between-study
variance of exactly 0 and warns. It is still the right comparison — the same
model fitted to the same boundary — but the agreement is to the optimiser's
tolerance rather than to machine precision, which is why the coefficients are
compared at 1e-3 and not 1e-8.
"""

from __future__ import annotations

import numpy as np
import pytest

import pimeta as ma

PER_STUDY = 60
STUDIES = 3


def _data():
    ycont, ybin, treat, study, age = [], [], [], [], []
    for s in range(STUDIES):
        i = np.arange(PER_STUDY)
        t = i % 2
        age_c = (i % 13) * 2
        score = (i * 37 + s * 11) % 100
        risk = 0.20 + 0.10 * t + 0.004 * age_c + 0.02 * s
        ybin.append((score < risk * 100).astype(float))
        ycont.append(2 * t + 0.1 * age_c + s + ((i * 13) % 7 - 3) * 0.3)
        treat.append(t.astype(float))
        study.append(np.full(PER_STUDY, float(s)))
        age.append(age_c.astype(float))
    return (np.concatenate(ycont), np.concatenate(ybin),
            np.concatenate(treat), np.concatenate(study), np.concatenate(age))


YCONT, YBIN, TREAT, STUDY, AGE = _data()

# lmer(ycont ~ treat + age_c + (1 | study), REML = TRUE)
R_REML_COEF = [0.82745490324672932, 2.046129878648737, 0.11451295506723493]
R_REML_SE = [0.58319607683608443, 0.090503434371791869, 0.006145346962695219]
R_REML_TAU2 = 0.99385887929287986
R_REML_S2 = 0.36846837456328158
R_REML_LL = -176.97910010936769

# ... REML = FALSE
R_ML_SE = [0.47846620626795677, 0.089990662842424532, 0.0061105288479747421]
R_ML_TAU2 = 0.66059498734963351
R_ML_S2 = 0.36430488954588552
R_ML_LL = -171.5781277457495

# glmer(ybin ~ treat + age_c + (1 | study), family = binomial) -- singular fit
R_BIN_COEF = [-1.2585094886987582, 0.4620327941608412, 0.021513931639190284]
R_BIN_SE = [0.35172850315290555, 0.32438535882382291, 0.0219614972099848]
R_BIN_LL = -110.85502657676008


@pytest.fixture(scope="module")
def gaussian():
    return ma.ipd_one_stage(YCONT, study=STUDY, treat=TREAT,
                            covariates={"age_c": AGE})


def test_the_linear_one_stage_model_matches_lmer(gaussian):
    assert gaussian.b == pytest.approx(R_REML_COEF, rel=1e-10)
    assert gaussian.tau2 == pytest.approx(R_REML_TAU2, rel=1e-6)
    assert gaussian.sigma2 == pytest.approx(R_REML_S2, rel=1e-6)
    assert gaussian.loglik == pytest.approx(R_REML_LL, rel=1e-9)


def test_the_standard_errors_match_lmer(gaussian):
    assert gaussian.se == pytest.approx(R_REML_SE, rel=1e-4)


def test_ml_differs_from_reml_in_the_right_direction():
    ml = ma.ipd_one_stage(YCONT, study=STUDY, treat=TREAT,
                          covariates={"age_c": AGE}, method="ML")
    assert ml.b == pytest.approx(R_REML_COEF, rel=1e-10)   # same point
    assert ml.tau2 == pytest.approx(R_ML_TAU2, rel=1e-5)   # smaller variance
    assert ml.sigma2 == pytest.approx(R_ML_S2, rel=1e-5)
    assert ml.loglik == pytest.approx(R_ML_LL, rel=1e-9)
    assert ml.se == pytest.approx(R_ML_SE, rel=1e-4)


def test_the_result_reports_the_model_it_fitted(gaussian):
    assert gaussian.coefnames == ["(Intercept)", "treat", "age_c"]
    assert (gaussian.participants, gaussian.studies) == (180, 3)
    assert gaussian.family == "gaussian"
    assert gaussian.converged
    assert 0.0 < gaussian.ICC < 1.0
    assert len(gaussian.coef_table()) == 3


def test_the_binomial_one_stage_model_matches_glmer():
    fit = ma.ipd_one_stage(YBIN, study=STUDY, treat=TREAT,
                           covariates={"age_c": AGE}, family="binomial")
    assert fit.b == pytest.approx(R_BIN_COEF, abs=2e-3)
    assert fit.se == pytest.approx(R_BIN_SE, rel=1e-4)
    assert fit.loglik == pytest.approx(R_BIN_LL, rel=1e-5)
    assert fit.sigma2 is None          # the logistic link fixes the scale
    assert fit.ICC is None


def test_the_quadrature_grid_can_be_changed():
    coarse = ma.ipd_one_stage(YBIN, study=STUDY, treat=TREAT, family="binomial",
                              quadrature=7)
    fine = ma.ipd_one_stage(YBIN, study=STUDY, treat=TREAT, family="binomial",
                            quadrature=25)
    assert coarse.b[1] == pytest.approx(fine.b[1], abs=1e-3)


def test_a_constant_outcome_is_rejected():
    with pytest.raises(ValueError, match="constant"):
        ma.ipd_one_stage(np.ones(60), study=np.zeros(60), treat=np.tile([0.0, 1.0], 30))


def test_a_mis_coded_binary_outcome_is_rejected():
    with pytest.raises(ValueError, match="coded 0/1"):
        ma.ipd_one_stage(YBIN + 1.0, study=STUDY, treat=TREAT, family="binomial")


def test_the_inputs_have_to_line_up():
    with pytest.raises(ValueError, match="same length"):
        ma.ipd_one_stage([1.0, 2.0], study=[0.0], treat=[0.0, 1.0])
    with pytest.raises(ValueError, match="one value per participant"):
        ma.ipd_one_stage(YCONT, study=STUDY, treat=TREAT,
                         covariates={"age_c": AGE[:-1]})
    with pytest.raises(ValueError, match="unknown family"):
        ma.ipd_one_stage(YCONT, study=STUDY, treat=TREAT, family="poisson")
    with pytest.raises(ValueError, match="unknown method"):
        ma.ipd_one_stage(YCONT, study=STUDY, treat=TREAT, method="DL")


def test_a_one_stage_fit_round_trips_through_a_record(tmp_path):
    from pimeta.record import load_record, save_record

    fit = ma.ipd_one_stage(YCONT, study=STUDY, treat=TREAT)
    path = tmp_path / "onestage.json"
    save_record(path, fit, mode="digest", function="ipd_one_stage")
    loaded = load_record(path)
    assert loaded.kind == "ipd_one_stage"
    assert loaded.results["values"]["tau2"] == pytest.approx(fit.tau2, rel=1e-12)
    assert loaded.results["values"]["participants"] == 180
