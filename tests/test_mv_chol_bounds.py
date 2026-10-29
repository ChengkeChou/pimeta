"""The Cholesky bounds, and what they made unreachable.

``_UNComp.chol_bounds`` binds the diagonal of the lower-triangular factor.  The
factor's vector is column-major, so "the diagonal" is not "the first ``J``
entries": binding the first ``J`` pinned ``L10``/``L20`` above zero and left
``L11``/``L22`` free, which made every correlation between the first inner
level and the others non-negative by construction.  A 60-study simulation with
true correlations (-0.60, 0.20, -0.30) returned ``rho = (0.0000, 0.2854,
-0.1272)``: exactly zero where the bound bit, estimated where it did not.
"""

import warnings

import numpy as np
import pytest

import pimeta as ma
from pimeta.data_compat import design_matrix
from pimeta.rma_mv import _UNComp

SEED = 20261029
STUDIES, LEVELS = 60, 3
SD = np.array([0.40, 0.40, 0.40])
RHO = np.array([-0.60, 0.20, -0.30])          # (1,0), (2,0), (2,1)
WITHIN_VAR = 0.01


def test_the_bounds_bind_the_diagonal_and_nothing_else():
    outer = np.repeat(np.arange(STUDIES), LEVELS)
    inner = np.tile(np.arange(LEVELS), STUDIES)
    comp = _UNComp(inner, outer, LEVELS)
    pairs = [(i, j) for j in range(LEVELS) for i in range(j, LEVELS)]
    assert [f"L{i}{j}" for i, j in pairs] == ["L00", "L10", "L20",
                                              "L11", "L21", "L22"]
    bounds = comp.chol_bounds()
    for (i, j), bound in zip(pairs, bounds):
        if i == j:
            assert bound == (0.0, None), (i, j, bound)
        else:
            assert bound == (None, None), (i, j, bound)


def test_a_negative_correlation_can_be_estimated():
    rng = np.random.default_rng(SEED)
    corr = np.eye(LEVELS)
    pairs = [(i, j) for j in range(LEVELS - 1) for i in range(j + 1, LEVELS)]
    for value, (i, j) in zip(RHO, pairs):
        corr[i, j] = corr[j, i] = value
    cov = np.outer(SD, SD) * corr
    assert np.linalg.eigvalsh(cov).min() > 0

    yi, vi, study, level = [], [], [], []
    for k in range(STUDIES):
        theta = rng.multivariate_normal(np.zeros(LEVELS), cov)
        for j in range(LEVELS):
            yi.append(float(theta[j] + rng.normal(0.0, np.sqrt(WITHIN_VAR))))
            vi.append(WITHIN_VAR)
            study.append(f"s{k:03d}")
            level.append(f"o{j}")

    data = {"yi": yi, "vi": vi, "study": study, "level": level}
    X, _ = design_matrix("~ level - 1", data)
    # diagonal sampling covariance: the only correlation in the model is the
    # between-study one, which is what the bounds act on
    V = ma.vcalc(np.array(vi), cluster=np.array(study), obs=np.array(level),
                 rho=0.0)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        fit = ma.rma_mv(np.array(yi), V=V, mods=X, data=data,
                        random="~ level | study", struct="UN",
                        intercept=False)

    rho = np.asarray(fit.rho, dtype=float)
    # the first pair is the one the old bounds clipped; it must now be free
    assert rho[0] < -0.2, rho
    assert abs(rho[0]) > 1e-6, "the first pair is still pinned at zero"
    # and the estimate should be in the neighbourhood of the truth
    assert abs(rho[0] - RHO[0]) < 0.4, (rho[0], RHO[0])


def test_the_shipped_fit_is_unchanged_by_the_wider_box():
    """Correcting the bounds must not move a fit whose optimum was inside.

    The tvhr fixture's `UN` optimum sits at `rho = (0.637, 0.864, 0.163)`, all
    strictly inside the cone, so the wider box has nothing to add and the fit
    has to land on exactly the numbers 1.1.4 reported. It did not, until the
    cone test was skipped on the Cholesky route: with the test on, the
    optimiser hit `|rho| = 1` on the way and the objective jumped to 1e5.
    """
    data = np.genfromtxt(
        __import__("pathlib").Path(__file__).resolve().parent
        / "data" / "tvhr_piecewise.csv", delimiter=",", skip_header=1)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        fit = ma.piecewise_joint(data[:, 2], data[:, 3], study=data[:, 0],
                                 treat=data[:, 1],
                                 breaks=np.array([0.0, 6.0, 12.0, 24.0]),
                                 struct="UN")
    assert fit.model.ll == pytest.approx(-5.293316572107862, rel=1e-9)
