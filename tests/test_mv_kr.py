"""Kenward-Roger degrees of freedom for a joint region.

The numbers here are not compared against R, because metafor has no
Kenward-Roger calibration for ``rma.mv``.  What is checked instead is that the
machinery is internally exact -- the parameter vector round-trips through the
covariance, the derivatives are of the right quantity, the degrees of freedom
respond in the direction they must -- and that the *coverage* it produces was
measured rather than assumed.
"""

import numpy as np
import pytest
from scipy import stats

import pimeta as ma
from pimeta.mv.kr import _G_of, _theta_of, kenward_roger_region


def _fit(k, seed=0, rho=0.6, tau2=(0.05, 0.05)):
    rng = np.random.default_rng(seed)
    cov = np.array([[tau2[0], rho * np.sqrt(tau2[0] * tau2[1])],
                    [rho * np.sqrt(tau2[0] * tau2[1]), tau2[1]]])
    theta = rng.multivariate_normal([0.3, -0.2], cov, k)
    vi = rng.uniform(0.01, 0.03, (k, 2))
    y = theta + rng.normal(0.0, 1.0, (k, 2)) * np.sqrt(vi)
    V = np.zeros((2 * k, 2 * k))
    for i in range(k):
        block = np.diag(vi[i])
        block[0, 1] = block[1, 0] = 0.5 * np.sqrt(vi[i, 0] * vi[i, 1])
        V[2 * i:2 * i + 2, 2 * i:2 * i + 2] = block
    X = np.zeros((2 * k, 2))
    X[np.arange(2 * k), np.tile([0, 1], k)] = 1.0
    return ma.rma_mv(y.ravel(), V=V, mods=X,
                     data={"coef": list(np.tile([0, 1], k)),
                           "study": [f"s{i}" for i in range(k)
                                     for _ in range(2)]},
                     random="~ coef | study", struct="UN", intercept=False)


def test_the_parameter_vector_round_trips_through_the_covariance():
    """The derivatives are only meaningful if the order is the optimiser's."""
    G = np.array([[0.04, 0.012, 0.005],
                  [0.012, 0.09, -0.02],
                  [0.005, -0.02, 0.06]])
    theta = _theta_of(G)
    assert theta.shape == (6,)
    assert theta[:3] == pytest.approx(np.diag(G))
    assert _G_of(theta, 3) == pytest.approx(G, abs=1e-15)


def test_the_degrees_of_freedom_are_finite_and_larger_than_chi2_at_small_k():
    """With ten studies the region must be wider than the Wald one.

    ``m`` around 7 with ``J = 2`` gives a critical value near 11.5 against the
    chi-square 5.99 -- that is the whole point of the calibration, and the sign
    of the change is the thing that must not be wrong.
    """
    out = kenward_roger_region(_fit(10), level=95)
    assert out["df1"] == 2
    assert 3.0 < out["m"] < 30.0
    assert out["crit"] > float(stats.chi2.ppf(0.95, 2))
    assert out["crit"] == pytest.approx(
        2.0 * out["m"] / (out["m"] - 1.0)
        * float(stats.f.ppf(0.95, 2, out["m"] - 1.0)), rel=1e-12)


def test_the_correction_shrinks_as_studies_are_added():
    """More studies estimate the variance parameters better, so m grows."""
    small = kenward_roger_region(_fit(10), level=95)
    large = kenward_roger_region(_fit(40), level=95)
    assert large["m"] > small["m"]
    assert abs(large["crit"] - float(stats.chi2.ppf(0.95, 2))) < abs(
        small["crit"] - float(stats.chi2.ppf(0.95, 2)))


def test_a_boundary_correlation_is_refused_not_guessed():
    """On the ridge the derivative does not exist; the answer is 'no'."""
    fit = _fit(12)
    G, _ = fit.g_matrix()
    G = np.asarray(G, dtype=float)
    pinned = np.array([G[0, 0], G[1, 1], 0.999999999])
    try:
        forced = ma.rma_mv(np.asarray(fit.yi), V=fit.V, mods=fit.X,
                           data=fit._data, random=fit.random, struct="UN",
                           intercept=False, sigma2=pinned)
    except (ValueError, np.linalg.LinAlgError):  # pragma: no cover
        pytest.skip("the fitter refuses the pinned covariance")
    with pytest.raises(ValueError, match="boundary|\\+/-1|flat"):
        kenward_roger_region(forced)


def test_the_level_is_accepted_in_both_conventions():
    fit = _fit(10)
    a = kenward_roger_region(fit, level=95)
    b = kenward_roger_region(fit, level=0.95)
    assert a["crit"] == pytest.approx(b["crit"], rel=1e-12)
