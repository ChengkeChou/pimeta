"""Joint pooling of interval-specific hazard ratios.

``piecewise_exp`` pools each interval with its own univariate model.  That
gives valid interval-specific estimates and no way to compare them: the
estimates share the between-study covariance, so "the hazard ratio falls after
six months" needs the joint covariance.  ``piecewise_joint`` fits the vector.

Three claims are checked here, and they are different kinds of claim:

1. **Against metafor.**  ``tests/data/tvhr_piecewise.csv`` is 16 simulated
   studies of 400 participants with a hazard ratio that falls from
   ``exp(0.60) = 1.82`` to ``exp(0) = 1.00`` over 24 months.  R's ``glm``
   per study and ``metafor::rma.mv`` on the stacked block reproduce every
   number, run with ``glm.control(epsilon = 1e-14)`` — the default stopping
   rule leaves the per-study variances wrong in the sixth digit, and a
   reference that stops early is not a reference.
2. **Against itself.**  With ``struct="DIAG"`` the joint fit must reproduce
   separate pooling exactly, because a diagonal within-study covariance and a
   diagonal between-study covariance make the intervals independent.  That
   check is what caught the between-study covariance coming back in
   factor-level order rather than interval order.
3. **By simulation.**  The trend test rejects when the hazard ratio really
   changes and does not when it does not.  That is the claim the tests exist
   for; agreement with metafor only says two programs compute the same thing.
"""

import pathlib
import warnings

import numpy as np
import pytest

import pimeta as ma

DATA = np.genfromtxt(
    pathlib.Path(__file__).resolve().parent / "data" / "tvhr_piecewise.csv",
    delimiter=",", skip_header=1)
STUDY = DATA[:, 0]
TREAT = DATA[:, 1]
TIME = DATA[:, 2]
EVENT = DATA[:, 3]
BREAKS = np.array([0.0, 6.0, 12.0, 24.0])

pytestmark = pytest.mark.filterwarnings(
    "ignore:the estimated between-interval correlation:RuntimeWarning")

# --- R 4.6.1, metafor 5.2.1, glm(epsilon = 1e-14) ------------------------
# study 1's own fit
R_STUDY1_LOG_HR = [0.4767983343, -0.1539744094, 0.0042736336]
R_STUDY1_VAR = [0.0174375094, 0.0678685048, 0.1388888889]

# struct = "DIAG"
R_DIAG_BETA = [0.5909320230, 0.2412065871, -0.0170264305]
R_DIAG_SE = [0.0522384335, 0.0609400437, 0.0906407100]
R_DIAG_TAU2 = [0.0212932560, 0.0000000001, 0.0247566031]

# struct = "UN"
R_UN_BETA = [0.5920080021, 0.2435418923, -0.0165547592]
R_UN_SE = [0.0532022401, 0.0702545423, 0.0911177147]
R_UN_TAU2 = [0.0230038320, 0.0194281842, 0.0263449990]
R_UN_RHO = [0.9997975338, 0.6503201172, 0.6654742678]
R_UN_LINEAR_EST = -0.6085627613
R_UN_LINEAR_SE = 0.0957289176


@pytest.fixture(scope="module")
def separate():
    return ma.piecewise_exp(TIME, EVENT, study=STUDY, treat=TREAT,
                            breaks=BREAKS)


@pytest.fixture(scope="module")
def joint_diag():
    return ma.piecewise_joint(TIME, EVENT, study=STUDY, treat=TREAT,
                              breaks=BREAKS, struct="DIAG")


@pytest.fixture(scope="module")
def joint_un():
    return ma.piecewise_joint(TIME, EVENT, study=STUDY, treat=TREAT,
                              breaks=BREAKS)


def test_the_per_study_stage_matches_r(separate):
    assert separate.studies[0] == 1
    assert list(separate.log_hr[0]) == pytest.approx(R_STUDY1_LOG_HR,
                                                     rel=1e-8)
    var = np.diag(np.asarray(separate.cov_hr)[0])
    assert list(var) == pytest.approx(R_STUDY1_VAR, rel=1e-8)


def test_the_per_study_covariance_is_diagonal(separate):
    """The claim the documentation makes: intervals partition follow-up."""
    cov = np.asarray(separate.cov_hr)
    worst = 0.0
    for i in range(cov.shape[0]):
        block = cov[i]
        off = block - np.diag(np.diag(block))
        worst = max(worst, float(np.nanmax(np.abs(off))))
    assert worst == 0.0


def test_diagonal_joint_pooling_reproduces_separate_pooling(separate,
                                                            joint_diag):
    """Independent intervals: the two routes must agree to the optimiser."""
    assert joint_diag.beta == pytest.approx(separate.beta, abs=1e-5)
    assert joint_diag.tau2 == pytest.approx(separate.tau2, abs=1e-5)


def test_the_diagonal_fit_matches_metafor(joint_diag):
    assert joint_diag.beta == pytest.approx(R_DIAG_BETA, abs=1e-6)
    assert joint_diag.se == pytest.approx(R_DIAG_SE, abs=1e-6)
    assert joint_diag.tau2 == pytest.approx(R_DIAG_TAU2, abs=1e-6)
    # the covariance is put back in interval order, not factor-level order
    # ("0-6" < "12-24" < "6-12" alphabetically)
    assert np.all(np.diag(joint_diag.Sigma) == pytest.approx(
        R_DIAG_TAU2, abs=1e-6))


def test_the_unstructured_fit_matches_metafor_loosely(joint_un):
    """Tolerance 1e-3, because this fit sits on the correlation ridge.

    ``rho`` between the first two intervals is 0.9998 under metafor and
    0.9999999 under pimeta: two correct implementations stopping at different
    points on the same ridge.  Pinning this to 1e-6 would pin the optimiser.
    """
    assert joint_un.beta == pytest.approx(R_UN_BETA, abs=1e-3)
    assert joint_un.se == pytest.approx(R_UN_SE, abs=1e-3)
    assert joint_un.tau2 == pytest.approx(R_UN_TAU2, abs=1e-3)
    sd = np.sqrt(np.diag(joint_un.Sigma))
    rho = (joint_un.Sigma / np.outer(sd, sd))[np.triu_indices(3, 1)]
    # the first correlation is the one that can be pinned: both implementations
    # put it at 1, which is where the ridge is
    assert rho[0] == pytest.approx(1.0, abs=1e-3)
    # the other two cannot.  On that ridge they trade off, and the two
    # implementations differ by 0.02 and 0.11 (pimeta 0.628 / 0.774 against
    # metafor 0.650 / 0.665).  Asserting 1e-2 here would pin the optimiser,
    # so the tolerance is what separates "the same model" from "the same
    # number", and only the first of those is true.
    assert rho[1:] == pytest.approx(R_UN_RHO[1:], abs=0.12)


def test_the_linear_trend_matches_metafor(joint_un):
    test = joint_un.trend_test("linear")
    assert test["estimate"] == pytest.approx(R_UN_LINEAR_EST, abs=1e-3)
    assert test["se"] == pytest.approx(R_UN_LINEAR_SE, abs=1e-3)
    assert test["df"] == 1


def test_the_wald_test_is_the_closed_form(joint_un):
    """Whatever the estimates are, the test must be w'beta with w'Vw."""
    w = np.array([-1.0, 0.0, 1.0])
    est = float(w @ joint_un.beta)
    var = float(w @ joint_un.vcov @ w)
    test = joint_un.trend_test("linear")
    assert test["estimate"] == pytest.approx(est, rel=1e-12)
    assert test["se"] == pytest.approx(np.sqrt(var), rel=1e-12)
    assert test["stat"] == pytest.approx(est ** 2 / var, rel=1e-12)
    # with three equally spaced intervals the "linear" weights are the
    # first-versus-last contrast, so the two tests coincide
    assert joint_un.trend_test("diff")["stat"] == pytest.approx(
        test["stat"], rel=1e-12)


def test_the_omnibus_test_has_two_degrees_of_freedom(joint_un):
    test = joint_un.trend_test("any")
    assert test["df"] == 2
    b = joint_un.beta
    C = np.array([[-1.0, 1.0, 0.0], [-1.0, 0.0, 1.0]])
    stat = float((C @ b) @ np.linalg.solve(C @ joint_un.vcov @ C.T,
                                           C @ b))
    assert test["stat"] == pytest.approx(stat, rel=1e-12)


def test_a_declining_hazard_ratio_is_detected(joint_diag, joint_un):
    """The fixture's hazard ratio really does fall: 1.81, 1.27, 0.98."""
    for fit in (joint_diag, joint_un):
        assert fit.trend_test("any")["pval"] < 1e-6
        assert fit.trend_test("linear")["pval"] < 1e-6
        assert fit.trend_test("diff")["pval"] < 1e-6
    hr = np.exp(joint_diag.beta)
    assert hr[0] > hr[1] > hr[2]


def test_a_constant_hazard_ratio_is_not_rejected():
    """The same simulation with the effect held fixed across intervals."""
    time, event, study, treat = _simulate([0.35, 0.35, 0.35], seed=7,
                                          k=14, n=250)
    fit = ma.piecewise_joint(time, event, study=study, treat=treat,
                             breaks=BREAKS, struct="DIAG")
    assert fit.trend_test("any")["pval"] > 0.05
    assert fit.trend_test("linear")["pval"] > 0.05
    hr = np.exp(fit.beta)
    assert np.max(hr) / np.min(hr) < 1.6


def test_a_declining_hazard_ratio_is_detected_by_the_fresh_simulation():
    time, event, study, treat = _simulate([0.60, 0.30, 0.00], seed=9,
                                          k=14, n=250)
    fit = ma.piecewise_joint(time, event, study=study, treat=treat,
                             breaks=BREAKS, struct="DIAG")
    assert fit.trend_test("any")["pval"] < 1e-4
    assert fit.trend_test("linear")["pval"] < 1e-4


def test_the_answer_does_not_depend_on_the_covariance_structure():
    """All three structures reject, and disagree about by how much.

    The qualitative answer survives; the statistic does not.  That is why the
    structure is a reported setting rather than a hidden default.
    """
    stats = {}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        for struct in ("UN", "HCS", "DIAG"):
            fit = ma.piecewise_joint(TIME, EVENT, study=STUDY, treat=TREAT,
                                     breaks=BREAKS, struct=struct)
            stats[struct] = fit.trend_test("any")["stat"]
            assert fit.trend_test("any")["pval"] < 1e-6
    assert len({round(v, 1) for v in stats.values()}) == 3


def test_a_boundary_correlation_warns():
    with pytest.warns(RuntimeWarning, match="ridge"):
        ma.piecewise_joint(TIME, EVENT, study=STUDY, treat=TREAT,
                           breaks=BREAKS, struct="UN")


def test_an_imposed_rho_changes_the_answer_and_is_checked():
    free = ma.piecewise_joint(TIME, EVENT, study=STUDY, treat=TREAT,
                              breaks=BREAKS, struct="DIAG")
    with_rho = ma.piecewise_joint(TIME, EVENT, study=STUDY, treat=TREAT,
                                  breaks=BREAKS, struct="DIAG", rho=0.5)
    assert with_rho.rho == 0.5
    assert with_rho.beta != pytest.approx(free.beta, abs=1e-6)
    # a correlation of 1 between intervals makes their covariance singular and
    # every comparison between them infinitely precise, so it cannot be allowed
    with pytest.raises(ValueError, match="positive definite"):
        ma.piecewise_joint(TIME, EVENT, study=STUDY, treat=TREAT,
                           breaks=BREAKS, struct="DIAG", rho=1.0)
    with pytest.raises(ValueError, match="rho"):
        ma.piecewise_joint(TIME, EVENT, study=STUDY, treat=TREAT,
                           breaks=BREAKS, rho=1.5)


def test_an_interval_with_no_events_is_reported_as_missing():
    """A dummy column of zeros would make the design singular.

    Nothing is followed beyond 24 months, so the [24, 30) interval has no rows
    at all: it gets ``NaN``, not a coefficient at minus infinity.
    """
    breaks = np.array([0.0, 6.0, 12.0, 24.0, 30.0])
    fit = ma.piecewise_joint(TIME, EVENT, study=STUDY, treat=TREAT,
                             breaks=breaks, struct="DIAG")
    assert np.isnan(fit.beta[3])
    assert fit.k[3] == 0
    assert np.all(np.isfinite(fit.beta[:3]))
    # the trend over the estimable intervals still works
    assert np.isfinite(fit.trend_test("linear")["stat"])


def test_an_intervals_with_too_few_events_is_refused_not_invented():
    with pytest.raises(ValueError, match="no interval"):
        ma.piecewise_joint(TIME, EVENT, study=STUDY, treat=TREAT,
                           breaks=BREAKS, min_events=100000)


def test_the_record_and_the_text(joint_diag):
    record = joint_diag.to_dict()
    assert record["labels"] == ["0-6", "6-12", "12-24"]
    assert record["k"] == [16, 16, 16]
    assert record["struct"] == "DIAG"
    assert len(record["vcov"]) == 3
    text = str(joint_diag)
    assert "joint pooling" in text
    assert "hazard ratio constant" in text
    assert "last vs first" in text


def test_the_structure_sensitivity_reproduces_the_warning(joint_un):
    """The warning quotes 52.1 / 57.3 / 40.6; those have to be computed.

    A sentence in a warning is a claim like any other.  This is the
    recomputation the warning points at, so the three numbers are checked
    against the fit rather than against the prose.
    """
    out = joint_un.structure_sensitivity()
    by = out["by_struct"]
    assert set(by) == {"UN", "HCS", "DIAG"}
    assert by["UN"]["tests"]["any"]["stat"] == pytest.approx(52.09, abs=0.05)
    assert by["HCS"]["tests"]["any"]["stat"] == pytest.approx(57.28, abs=0.05)
    assert by["DIAG"]["tests"]["any"]["stat"] == pytest.approx(40.58, abs=0.05)
    # the spread is exactly what the per-structure stats say it is
    stats = [r["tests"]["any"]["stat"] for r in by.values()]
    assert out["spread"]["any"]["min"] == pytest.approx(min(stats), rel=1e-12)
    assert out["spread"]["any"]["max"] == pytest.approx(max(stats), rel=1e-12)
    # and the refit under the fitted structure must agree with the fit itself
    assert by["UN"]["tests"]["linear"]["stat"] == pytest.approx(
        joint_un.trend_test("linear")["stat"], rel=1e-8)


def test_the_diagonal_structure_cannot_sit_on_the_ridge(joint_un):
    """The ridge is a property of ``UN``, which is why the spread exists."""
    by = joint_un.structure_sensitivity()["by_struct"]
    assert by["UN"]["boundary"] is True
    assert by["DIAG"]["boundary"] is False


def test_every_structure_agrees_the_hazard_ratio_falls(joint_un):
    """The claim that survives the ridge is the qualitative one."""
    by = joint_un.structure_sensitivity()["by_struct"]
    for struct, row in by.items():
        assert row["tests"]["any"]["pval"] < 1e-6, struct
        assert row["tests"]["linear"]["estimate"] < 0.0, struct
    # and the statistic itself is not stable to one decimal place
    assert (by["HCS"]["tests"]["any"]["stat"]
            - by["DIAG"]["tests"]["any"]["stat"]) > 10.0


def test_a_structure_that_fails_is_reported_not_raised(joint_un):
    out = joint_un.structure_sensitivity(structs=("UN", "not-a-structure"))
    assert "error" in out["by_struct"]["not-a-structure"]
    assert "tests" in out["by_struct"]["UN"]
    # the failing structure contributes nothing to the spread
    assert out["spread"]["any"]["min"] == pytest.approx(
        out["by_struct"]["UN"]["tests"]["any"]["stat"], rel=1e-12)


# -- the contrast route: one slope per study, one variance to estimate ------- #

@pytest.fixture(scope="module")
def contrast():
    return ma.contrast_trend(TIME, EVENT, study=STUDY, treat=TREAT,
                             breaks=BREAKS)


def test_the_contrast_slope_is_the_within_study_wls_slope(separate, contrast):
    """Every reported slope has to be c'V^-1 y / c'V^-1 c, with c centred.

    This is the whole model, so it is checked directly on one study rather than
    only through the pooled answer.
    """
    log_hr = np.asarray(separate.log_hr)
    cov = np.asarray(separate.cov_hr)
    points = np.array([3.0, 9.0, 18.0])       # midpoints of the three intervals
    worst = 0.0
    for i in range(log_hr.shape[0]):
        ok = np.isfinite(log_hr[i])
        if ok.sum() < 2:
            continue
        idx = np.flatnonzero(ok)
        info = np.linalg.inv(cov[i][np.ix_(idx, idx)])
        t = points[idx]
        w = info.sum()                        # 1' V^-1 1
        c = t - (np.ones(idx.size) @ info @ t) / w
        hand = float(c @ info @ log_hr[i][idx]) / float(c @ info @ c)
        worst = max(worst, abs(hand - contrast["slope"][i]))
    assert worst < 1e-12


def test_the_contrast_trend_agrees_with_the_joint_answer(contrast, joint_un):
    """Same question, different weighting: same sign, similar size.

    The point of the two routes is not that they give the same number -- the
    contrast route gives more weight to a study whose slope is well determined
    and none to its level, which is the nuisance dimension being removed -- but
    they must not disagree about the direction.
    """
    joint = joint_un.trend_test("linear")
    per_month = joint["estimate"] / 12.0      # the joint contrast spans 12 months
    assert contrast["beta"] < 0.0
    assert contrast["ci_ub"] < 0.0
    assert contrast["beta"] == pytest.approx(per_month, rel=0.35)


def test_the_contrast_model_has_one_variance_parameter(contrast):
    """No correlation to estimate, which is the reason for the route."""
    assert isinstance(contrast["tau2"], float)
    assert contrast["tau2"] >= 0.0
    assert contrast["df"] == contrast["k"] - 1        # Knapp-Hartung
    assert contrast["k"] == 16
    assert contrast["excluded"] == {}
    assert contrast["times"].tolist() == [3.0, 9.0, 18.0]


def test_the_contrast_answer_does_not_depend_on_the_between_study_structure():
    """The joint fit's statistic moves by 40% between structures; this one has
    no structure to move."""
    fit = ma.contrast_trend(TIME, EVENT, study=STUDY, treat=TREAT,
                            breaks=BREAKS)
    again = ma.contrast_trend(TIME, EVENT, study=STUDY, treat=TREAT,
                              breaks=BREAKS, rho=None)
    assert fit["beta"] == again["beta"]
    assert fit["tau2"] == again["tau2"]


def test_a_constant_hazard_ratio_shows_no_trend_through_the_contrast():
    time, event, study, treat = _simulate([0.35, 0.35, 0.35], seed=7,
                                          k=14, n=250)
    out = ma.contrast_trend(time, event, study=study, treat=treat,
                            breaks=BREAKS)
    assert out["p_value"] > 0.05
    assert abs(out["beta"]) < 0.02


def test_a_declining_hazard_ratio_shows_a_trend_through_the_contrast():
    time, event, study, treat = _simulate([0.60, 0.30, 0.00], seed=9,
                                          k=14, n=250)
    out = ma.contrast_trend(time, event, study=study, treat=treat,
                            breaks=BREAKS)
    assert out["beta"] < 0.0
    assert out["p_value"] < 0.01
    assert out["ratio"] == pytest.approx(float(np.exp(out["beta"])), rel=1e-12)


def test_the_contrast_refusals():
    with pytest.raises(ValueError, match="three or more boundaries"):
        ma.contrast_trend(TIME, EVENT, study=STUDY, treat=TREAT,
                          breaks=np.array([0.0, 6.0]))
    with pytest.raises(ValueError, match="one time per interval"):
        ma.contrast_trend(TIME, EVENT, study=STUDY, treat=TREAT,
                          breaks=BREAKS, times=[1.0, 2.0])
    with pytest.raises(ValueError, match="rho"):
        ma.contrast_trend(TIME, EVENT, study=STUDY, treat=TREAT,
                          breaks=BREAKS, rho=2.0)
    with pytest.raises(ValueError, match="0/1"):
        ma.contrast_trend(TIME, EVENT, study=STUDY, treat=TREAT + 1,
                          breaks=BREAKS)


def _simulate(effect, *, seed, k, n, breaks=BREAKS, sd_study=0.12,
              rho_study=0.5, base=0.08, mean_censor=40.0, horizon=24.0):
    """Piecewise-exponential survival data with a known log hazard ratio."""
    rng = np.random.default_rng(seed)
    k_int = len(effect)
    cov = np.full((k_int, k_int), rho_study * sd_study ** 2)
    np.fill_diagonal(cov, sd_study ** 2)
    out_t, out_e, out_s, out_tr = [], [], [], []
    for i in range(1, k + 1):
        u = rng.multivariate_normal(np.zeros(k_int), cov)
        rate0 = base * np.exp(rng.normal(0.0, 0.3))
        treat = np.tile([0.0, 1.0], n // 2)
        cens = rng.exponential(mean_censor, n)
        for b in range(n):
            t = 0.0
            event = 0.0
            for j in range(k_int):
                rate = rate0 * np.exp((effect[j] + u[j]) * treat[b])
                wait = rng.exponential(1.0 / rate)
                if t + wait <= min(cens[b], breaks[j + 1]):
                    t += wait
                    event = 1.0
                    break
                if cens[b] <= breaks[j + 1]:
                    t = cens[b]
                    break
                t = breaks[j + 1]
            t = min(t, horizon)
            out_t.append(t)
            out_e.append(event if t <= horizon else 0.0)
            out_s.append(float(i))
            out_tr.append(float(treat[b]))
    return (np.array(out_t), np.array(out_e), np.array(out_s),
            np.array(out_tr))
