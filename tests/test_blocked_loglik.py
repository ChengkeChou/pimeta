"""The block-wise likelihood must be the same function, only cheaper.

The fast path is chosen automatically whenever the random effects and the
sampling covariances both live inside one grouping factor.  If the partition
were wrong the likelihood would still be finite, still be smooth, and still be
wrong -- so the two paths are compared point by point, on every shape of model
this suite has, rather than argued about.
"""

from __future__ import annotations

import numpy as np
import pytest

import pimeta as ma
from pimeta.rma_mv import _independent_blocks, _mv_loglik


def _konstantopoulos():
    """A nested three-level model: districts, years within them."""
    rng = np.random.default_rng(7)
    district = np.repeat(np.arange(12), 4)
    year = np.tile(np.arange(4), 12)
    yi = rng.normal(0.2, 0.1, district.size)
    vi = np.full(district.size, 0.04)
    return yi, vi, {"district": district, "year": year}


@pytest.mark.parametrize("random,struct", [
    ("~ 1 | district", None),
    ("~ year | district", "CS"),
    ("~ year | district", "DIAG"),
])
def test_the_blocked_likelihood_equals_the_full_one(random, struct):
    yi, vi, data = _konstantopoulos()
    kwargs = {"random": random}
    if struct is not None:
        kwargs["struct"] = struct
    fit = ma.rma_mv(yi, vi=vi, data=data, **kwargs)
    comp = fit.components[0]
    params = np.asarray(fit.params, dtype=float)
    groups = _independent_blocks(fit.components, fit.V, fit.X.shape[0],
                                 comp.make_D(params))
    assert groups is not None

    for scale in (1.0, 1.7, 0.4):
        piece = params * scale
        D = [comp.make_D(piece)]
        full = _mv_loglik(D, fit.yi, fit.V, fit.X, "REML")
        blocked = _mv_loglik(D, fit.yi, fit.V, fit.X, "REML", groups=groups)
        assert blocked == pytest.approx(full, rel=1e-10, abs=1e-10)


def test_a_multivariate_model_with_correlated_outcomes_is_blocked_correctly():
    """The D-family layout: two outcomes per study, V block diagonal."""
    rng = np.random.default_rng(3)
    studies, J = 8, 2
    V = np.zeros((studies * J, studies * J))
    yi = np.empty(studies * J)
    for s in range(studies):
        block = np.full((J, J), 0.02)
        np.fill_diagonal(block, 0.05)
        V[s * J:(s + 1) * J, s * J:(s + 1) * J] = block
        yi[s * J:(s + 1) * J] = rng.normal(0.1, 0.2, J)
    X = np.zeros((studies * J, J))
    X[np.arange(studies * J), np.tile(np.arange(J), studies)] = 1.0
    frame = {"study": [f"s{s}" for s in range(studies) for _ in range(J)],
             "outcome": [f"o{j}" for _ in range(studies) for j in range(J)]}
    fit = ma.rma_mv(yi, V=V, mods=X, data=frame,
                    random="~ outcome | study", struct="UN", intercept=False)
    comp = fit.components[0]
    params = np.asarray(fit.params, dtype=float)
    groups = _independent_blocks(fit.components, V, X.shape[0],
                                 comp.make_D(params))
    assert groups is not None
    assert groups.tolist() == list(np.repeat(np.arange(studies), J))

    D = [comp.make_D(params)]
    full = _mv_loglik(D, yi, V, X, "REML")
    blocked = _mv_loglik(D, yi, V, X, "REML", groups=groups)
    assert blocked == pytest.approx(full, rel=1e-10, abs=1e-10)
    assert full == pytest.approx(-fit.ll, abs=1e-8)


def test_a_sampling_covariance_that_crosses_studies_is_not_blocked():
    """Two rows from different groups that share a sampling covariance."""
    yi = np.array([0.1, 0.2, 0.3, 0.4])
    V = np.eye(4) * 0.05
    V[0, 2] = V[2, 0] = 0.01                  # study 0 and study 1 are linked
    fit = ma.rma_mv(yi, V=V, data={"study": [0, 0, 1, 1]},
                    random="~ 1 | study")
    comp = fit.components[0]
    params = np.asarray(fit.params, dtype=float)
    groups = _independent_blocks(fit.components, V, fit.X.shape[0],
                                 comp.make_D(params))
    assert groups is None


def test_two_random_terms_are_not_blocked():
    """Two components with different groupings do not share a partition."""
    yi, vi, data = _konstantopoulos()
    fit = ma.rma_mv(yi, vi=vi, data=data,
                    random=["~ 1 | district", "~ year | district"],
                    struct="CS")
    comp = fit.components[0]
    params = np.asarray(fit.params, dtype=float)
    groups = _independent_blocks(fit.components, fit.V, fit.X.shape[0],
                                 comp.make_D(params[: comp.n_params]))
    assert groups is None
