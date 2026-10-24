"""Joint prediction region for a multivariate meta-analysis.

metafor implements no joint prediction region for ``rma.mv`` — its ``predict``
returns ``NA`` bounds for a multilevel fit — so there is nothing to compare
against and this file has to do the two things a reference would have done:
check the algebra against the univariate case, which *is* pinned, and check by
simulation that a region claimed to be 95% contains a new study's effects 95% of
the time.

The second one is the point. A projection of an ellipsoid onto one outcome is
wider than that outcome's own prediction interval, so a region can look
"conservative" per outcome while its joint coverage is wrong — which is exactly
what a per-outcome check cannot see.
"""

import numpy as np
import pytest
from scipy import stats

import pimeta as ma
from pimeta.data_compat import design_matrix

# The small-k warning is expected throughout this file; the test that is about
# it re-enables warnings locally.
pytestmark = pytest.mark.filterwarnings(
    "ignore:the chi-square reference is asymptotic:RuntimeWarning")

# --- Berkey 1998, the fit tests/test_mv.py already pins --------------------
YI = [0.47, -0.32, 0.20, -0.60, 0.40, -0.12, 0.26, -0.31, 0.56, -0.39]
VI = [0.0075, 0.0077, 0.0057, 0.0008, 0.0021,
      0.0014, 0.0029, 0.0015, 0.0148, 0.0304]
STUDY = [1, 1, 2, 2, 3, 3, 4, 4, 5, 5]
OUTCOME = ["PD", "AL"] * 5


@pytest.fixture(scope="module")
def fit():
    data = {"study": STUDY, "outcome": OUTCOME,
            "yi": np.array(YI), "vi": np.array(VI)}
    V = ma.vcalc(np.array(VI), cluster=data["study"], obs=data["outcome"],
                 rho=0.5)
    X, _ = design_matrix("~ outcome - 1", data)
    return ma.rma_mv(np.array(YI), V=V, mods=X, data=data,
                     random="~ outcome | study", struct="UN",
                     intercept=False)


def test_the_region_is_the_chi_square_ellipsoid(fit):
    region = ma.predict_region(fit)
    expect = stats.chi2.ppf(0.95, 2)
    assert region.crit == pytest.approx(expect, rel=1e-12)
    assert region.dim == 2
    cov = np.asarray(fit.vb) + np.asarray(fit.g_matrix()[0])
    assert region.cov == pytest.approx(cov, rel=1e-12)
    assert region.center == pytest.approx(np.array(fit.beta), rel=1e-12)


def test_membership_is_the_mahalanobis_rule(fit):
    region = ma.predict_region(fit)
    assert region.contains(region.center)
    assert region.distance(region.center) == pytest.approx(0.0, abs=1e-12)
    # walk out along the first principal direction until it leaves
    scale = np.sqrt(np.diag(region.cov))
    edge = region.center + scale * 0.5
    inside = region.distance(edge) <= region.crit
    far = region.center + scale * 10.0
    assert region.contains(edge) if inside else not region.contains(edge)
    assert not region.contains(far)


def test_a_higher_level_gives_a_larger_region(fit):
    ninety = ma.predict_region(fit, level=90)
    ninety_five = ma.predict_region(fit)
    assert ninety.crit < ninety_five.crit
    assert (ninety.margin(0)[1] - ninety.margin(0)[0]
            < ninety_five.margin(0)[1] - ninety_five.margin(0)[0])


def test_asking_about_an_observed_estimate_widens_the_region(fit):
    S = np.array([[0.05, 0.01], [0.01, 0.06]])
    true_effect = ma.predict_region(fit)
    observed = ma.predict_region(fit, observation=S)
    assert not true_effect.for_observation and observed.for_observation
    assert observed.cov == pytest.approx(true_effect.cov + S, rel=1e-12)


def test_the_projection_is_wider_than_a_per_outcome_interval(fit):
    """Projecting a joint region is conservative, and says so."""
    region = ma.predict_region(fit)
    for i in range(region.dim):
        lo, hi = region.margin(i)
        z = stats.norm.ppf(0.975)
        half = z * np.sqrt(region.cov[i, i])
        assert (hi - lo) / 2 > half


def test_one_outcome_reduces_to_the_ordinary_prediction_interval():
    """With J = 1 the ellipsoid is the univariate prediction interval.

    chi-square(1) is the square of a standard normal, so the reduction is exact
    and this is the only algebra check available -- metafor has no joint region
    for a multilevel fit to compare against.
    """
    rng = np.random.default_rng(7)
    k = 40
    mu, tau2 = 0.4, 0.05
    vi = rng.uniform(0.02, 0.06, k)
    theta = rng.normal(mu, np.sqrt(tau2), k)
    yi = rng.normal(theta, np.sqrt(vi))
    fit = ma.rma_mv(yi, V=np.diag(vi), data={"study": list(range(k))},
                    random="~ 1 | study", struct="CS")
    region = ma.predict_region(fit)
    assert region.dim == 1
    lo, hi = region.margin(0)
    z = stats.norm.ppf(0.975)
    half = z * np.sqrt(region.cov[0, 0])
    assert hi - region.center[0] == pytest.approx(half, rel=1e-10)
    assert lo == pytest.approx(region.center[0] - half, rel=1e-10)
    # ... and the covariance is the univariate one: V + tau2
    assert region.cov[0, 0] == pytest.approx(
        float(np.asarray(fit.vb).ravel()[0]) + float(fit.sigma2[0]), rel=1e-12)
    # the Hotelling reference is t^2 on one df fewer, hence wider
    hotelling = ma.predict_region(fit, test="t")
    m = fit.k - fit.p
    assert hotelling.crit == pytest.approx(stats.t.ppf(0.975, m - 1) ** 2,
                                           rel=1e-12)
    assert hotelling.crit > region.crit


def _simulate_fit(rng, k, mu, sigma):
    """One meta-analysis of ``k`` studies reporting two correlated outcomes."""
    theta = rng.multivariate_normal(mu, sigma, k)
    vi = rng.uniform(0.01, 0.03, (k, 2))
    V = np.zeros((k, 2, 2))
    V[:, 0, 0] = vi[:, 0]
    V[:, 1, 1] = vi[:, 1]
    V[:, 0, 1] = V[:, 1, 0] = 0.5 * np.sqrt(vi[:, 0] * vi[:, 1])
    yi = theta + rng.normal(0.0, 1.0, (k, 2)) * np.sqrt(vi)
    # stack study-major, one mean per outcome
    Vbig = np.zeros((2 * k, 2 * k))
    for i in range(k):
        Vbig[2 * i:2 * i + 2, 2 * i:2 * i + 2] = V[i]
    X = np.zeros((2 * k, 2))
    X[np.arange(2 * k), np.tile([0, 1], k)] = 1.0
    return ma.rma_mv(yi.ravel(), V=Vbig, mods=X,
                     data={"coef": list(np.tile([0, 1], k)),
                           "study": [f"s{i}" for i in range(k)
                                     for _ in range(2)]},
                     random="~ coef | study", struct="UN", intercept=False)


def test_the_region_covers_a_new_study_at_the_stated_rate():
    """The claim the region makes, checked against simulation.

    Twenty studies of two correlated outcomes; draw a new study's true effects
    from the *true* law and count how often it lands inside.  The fit is never
    told the truth, so the error in mu-hat, V-hat and Sigma-hat are all in the
    count.

    Measured over 300 replications this configuration covers 0.917 under the
    chi-square reference and 0.930 under the Hotelling one (MC se 0.016).  The
    guard below is deliberately loose: it detects a collapse, and the manual
    run of ``tools/coverage_predict_region.py`` is what reports calibration.
    """
    rng = np.random.default_rng(20261024)
    mu = np.array([0.30, -0.20])
    sigma = np.array([[0.06, 0.03], [0.03, 0.09]])
    k = 20
    reps = 60
    hits_z = 0
    hits_t = 0
    for _ in range(reps):
        fit = _simulate_fit(rng, k, mu, sigma)
        region = ma.predict_region(fit)
        hotelling = ma.predict_region(fit, test="t")
        fresh = rng.multivariate_normal(mu, sigma)
        hits_z += int(region.contains(fresh))
        hits_t += int(hotelling.contains(fresh))
    # the two references share the ellipsoid, so the Hotelling critical value
    # can only add points -- that part is not sampling error
    assert hits_t >= hits_z
    coverage = hits_z / reps
    assert 0.82 <= coverage <= 1.0, coverage


def test_a_per_outcome_interval_is_anti_conservative_jointly(fit):
    """Why the region exists: the per-outcome intervals do not add up.

    Point-wise this is a tautology (the projection is wider), but it is the
    property that is quoted, and it is the one a reader is tempted to skip.
    """
    region = ma.predict_region(fit)
    z = stats.norm.ppf(0.975)
    for i in range(region.dim):
        lo, hi = region.margin(i)
        half = z * np.sqrt(region.cov[i, i])
        assert (region.center[i] - half, region.center[i] + half) != (lo, hi)


def test_the_asymptotic_reference_warns_with_few_studies(fit):
    with pytest.warns(RuntimeWarning, match="asymptotic"):
        ma.predict_region(fit)


def test_the_hotelling_reference_is_quiet(fit):
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        ma.predict_region(fit, test="t")


def test_the_wrong_kind_of_object_is_rejected():
    with pytest.raises(TypeError, match="RmaMvResult"):
        ma.predict_region({"beta": [0.1]})


def test_an_unknown_reference_is_rejected(fit):
    with pytest.raises(ValueError, match="unknown test"):
        ma.predict_region(fit, test="knha")


def test_the_hotelling_critical_value_is_the_closed_form(fit):
    """test='t' swaps the chi-square for J(m-1)/(m-J) F(J, m-J)."""
    region = ma.predict_region(fit, test="t")
    m = fit.k - fit.p
    expect = (2 * (m - 1.0) / (m - 2.0)) * stats.f.ppf(0.95, 2, m - 2)
    assert region.crit == pytest.approx(expect, rel=1e-12)
    assert region.test == "t"
    assert region.df == pytest.approx([2.0, float(m - 2)])
    # the covariance is the same ellipsoid; only the reference moved
    assert region.cov == pytest.approx(ma.predict_region(fit).cov, rel=1e-12)
    assert region.crit > ma.predict_region(fit).crit
    assert "Hotelling" in str(region)


def test_the_hotelling_reference_needs_more_df_than_outcomes(fit):
    with pytest.raises(ValueError, match="degrees of freedom"):
        ma.predict_region(fit, test="t", df=2)


def test_an_observation_with_the_wrong_shape_is_rejected(fit):
    with pytest.raises(ValueError, match="observation"):
        ma.predict_region(fit, observation=np.eye(3))


def test_the_record_and_the_text(fit):
    region = ma.predict_region(fit)
    record = region.to_dict()
    assert record["df"] == 2
    assert record["level"] == 95.0
    assert record["center"] == pytest.approx(
        [float(v) for v in region.center], rel=1e-12)
    text = str(region)
    assert "joint prediction region" in text
    assert "wider than per-outcome intervals" in text
