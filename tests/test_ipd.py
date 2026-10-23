"""Individual participant data: the first stage, the second stage, and parity.

Reference values come from R 4.6.1:

* ``glm(y ~ treat + age_c, family = binomial)`` per study;
* ``metafor::rma(yi = <treatment log OR>, vi = <its variance>, method = "REML")``
  on the three study-level estimates.

The participant-level data are built arithmetically rather than drawn from a
generator, so the identical rows can be constructed in R and in Python with no
file to ship and no RNG whose stream might drift between versions.
"""

from __future__ import annotations

import numpy as np
import pytest

import pimeta as ma

STUDIES = 3
PER_STUDY = 60


def _participants():
    """The deterministic dataset: no RNG, identical in R and Python."""
    y, treat, study, age = [], [], [], []
    for s in range(STUDIES):
        i = np.arange(PER_STUDY)
        t = i % 2
        a = 40 + (i % 13) * 2
        score = (i * 37 + s * 11) % 100
        risk = 0.20 + 0.10 * t + 0.004 * (a - 40) + 0.02 * s
        y.append((score < risk * 100).astype(float))
        treat.append(t.astype(float))
        study.append(np.full(PER_STUDY, float(s)))
        age.append((a - 40).astype(float))
    return (np.concatenate(y), np.concatenate(study), np.concatenate(treat),
            np.concatenate(age))


Y, STUDY, TREAT, AGE = _participants()

# glm(y ~ treat + age_c, family = binomial), study 0
R_B0 = [-1.6041656107279185, 0.49445146271785118, 0.035376983104429709]
R_SE0 = [0.65032581366321329, 0.58389733114318032, 0.039566901482120664]
R_VB0_ROW1 = [0.42292366391672048, -0.1916302303243235, -0.019150101020658585]

# the three treatment log ORs
R_STUDY_B = [0.49445146271785118, 0.46205210174934108, 0.43968457156220514]
R_STUDY_SE = [0.58389733114318032, 0.56172083412465534, 0.54593020825896987]

# metafor::rma on those three
R_POOLED_B = 0.46416487241470605
R_POOLED_SE = 0.32516885700076131
R_POOLED_TAU2 = 0.0
R_POOLED_QE = 0.004715370119951017


# --------------------------------------------------------------------------- #
# the first stage
# --------------------------------------------------------------------------- #
def test_the_logistic_first_stage_matches_glm():
    n = PER_STUDY
    i = np.arange(n)
    treat = (i % 2).astype(float)
    age = (40 + (i % 13) * 2 - 40).astype(float)
    y = ((i * 37) % 100 < (0.20 + 0.10 * treat + 0.004 * age) * 100).astype(float)
    X = np.column_stack([np.ones(n), treat, age])

    b, vb, iterations, converged = ma.logistic_model(y, X)
    assert converged and iterations < 20
    assert b == pytest.approx(R_B0, rel=1e-10)
    # 1e-6: R's glm stops on a relative deviance change of 1e-8, one iteration
    # before the coefficients stop moving, so its vcov is evaluated slightly
    # earlier than ours.
    assert np.sqrt(np.diag(vb)) == pytest.approx(R_SE0, rel=1e-6)
    assert vb[0, :] == pytest.approx(R_VB0_ROW1, rel=1e-6)


def test_the_logistic_first_stage_rejects_a_bad_design():
    with pytest.raises(ValueError, match="coded 0/1"):
        ma.logistic_model([0.0, 1.0, 2.0], np.ones((3, 1)))
    with pytest.raises(ValueError, match="rank deficient"):
        ma.logistic_model([0.0, 1.0, 1.0], np.ones((3, 2)))
    with pytest.raises(ValueError, match="complete separation"):
        # every participant with x > 0 has the event: the MLE does not exist
        ma.logistic_model([0.0, 1.0, 1.0, 1.0],
                          np.column_stack([np.ones(4), [0.0, 1.0, 2.0, 3.0]]))


def test_the_linear_first_stage_is_ols():
    x = np.array([1.0, 2.0, 3.0, 4.0])
    y = 2.0 * x + 1.0
    X = np.column_stack([np.ones(4), x])
    b, vb = ma.linear_model(y, X)
    assert b == pytest.approx([1.0, 2.0], rel=1e-12)
    assert np.all(np.diag(vb) > 0)          # no residual df left, but finite


# --------------------------------------------------------------------------- #
# the second stage
# --------------------------------------------------------------------------- #
@pytest.fixture(scope="module")
def pooled():
    return ma.ipd_two_stage(Y, study=STUDY, treat=TREAT,
                            covariates={"age_c": AGE})


def test_the_two_stage_pooling_matches_r(pooled):
    assert pooled.b == pytest.approx(R_POOLED_B, rel=1e-8)
    assert pooled.se == pytest.approx(R_POOLED_SE, rel=1e-6)
    assert pooled.tau2 == pytest.approx(R_POOLED_TAU2, abs=1e-12)
    assert pooled.QE == pytest.approx(R_POOLED_QE, rel=1e-4)


def test_the_per_study_estimates_match_glm(pooled):
    assert [s.estimate for s in pooled.per_study] == pytest.approx(
        R_STUDY_B, rel=1e-8)
    assert [float(np.sqrt(s.variance)) for s in pooled.per_study] == \
        pytest.approx(R_STUDY_SE, rel=1e-6)
    assert [s.study for s in pooled.per_study] == ["0", "1", "2"]
    assert [s.events for s in pooled.per_study] == [17, 19, 21]


def test_the_two_stage_result_reports_what_it_is(pooled):
    assert pooled.term == "treat"
    assert (pooled.k, pooled.studies) == (3, 3)
    assert pooled.excluded == {}
    assert pooled.coefnames if hasattr(pooled, "coefnames") else True
    table = pooled.coef_table()
    assert table[0]["term"] == "treat"
    forest = pooled.per_study_table()
    assert len(forest) == 4
    assert forest[-1]["study"] == "Pooled"
    assert forest[-1]["n"] == 3 * PER_STUDY


def test_an_interaction_is_pooled_the_same_way():
    fit = ma.ipd_two_stage(Y, study=STUDY, treat=TREAT,
                           covariates={"age_c": AGE}, term="treat:age_c")
    assert fit.term == "treat:age_c"
    assert np.isfinite(fit.b)
    assert fit.per_study[0].coefnames == ["(Intercept)", "treat", "age_c",
                                          "treat:age_c"]


def test_a_study_without_events_is_excluded_and_named():
    y = np.concatenate([Y, np.zeros(PER_STUDY)])
    study = np.concatenate([STUDY, np.full(PER_STUDY, 99.0)])
    treat = np.concatenate([TREAT, (np.arange(PER_STUDY) % 2).astype(float)])
    age = np.concatenate([AGE, np.zeros(PER_STUDY)])
    fit = ma.ipd_two_stage(y, study=study, treat=treat,
                           covariates={"age_c": age})
    assert (fit.k, fit.studies) == (3, 4)
    assert "99" in fit.excluded
    assert "event" in fit.excluded["99"]


def test_the_stage_rejects_inputs_that_do_not_line_up():
    with pytest.raises(ValueError, match="same length"):
        ma.ipd_two_stage([0.0, 1.0], study=[0.0], treat=[0.0, 1.0])
    with pytest.raises(ValueError, match="one value per participant"):
        ma.ipd_two_stage(Y, study=STUDY, treat=TREAT,
                         covariates={"age_c": AGE[:-1]})
    with pytest.raises(ValueError, match="unknown term"):
        ma.ipd_two_stage(Y, study=STUDY, treat=TREAT, term="sex")


def test_a_continuous_outcome_uses_the_linear_first_stage():
    outcome = 2.0 * TREAT + 0.1 * AGE
    fit = ma.ipd_two_stage(outcome, study=STUDY, treat=TREAT,
                           covariates={"age_c": AGE}, family="gaussian")
    assert fit.b == pytest.approx(2.0, rel=1e-10)
    assert fit.per_study[0].events is None


def test_an_ipd_fit_round_trips_through_a_record(tmp_path):
    from pimeta.record import load_record, save_record

    fit = ma.ipd_two_stage(Y, study=STUDY, treat=TREAT)
    path = tmp_path / "ipd.json"
    save_record(path, fit, mode="digest", function="ipd_two_stage")
    loaded = load_record(path)
    assert loaded.kind == "ipd_two_stage"
    assert loaded.results["values"]["b"] == pytest.approx(fit.b, rel=1e-12)
    assert loaded.results["values"]["per_study"][0]["events"] == 17
