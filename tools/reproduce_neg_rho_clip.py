"""Reproduce the ``chol_bounds`` clip: a negative correlation that cannot be fitted.

``_UNComp.chol_bounds`` binds the first ``J`` entries of the Cholesky vector,
but the vector is column-major, so those are ``L00, L10, L20`` rather than the
diagonal ``L00, L11, L22``.  The consequence is not a positive-definiteness
problem -- ``L L'`` is PSD for any ``L`` -- but a *reachability* problem: the
pair between the first inner level and the others can never be negative, and a
fit that starts there is clipped onto the bound at the first iteration.

Run it with the shipped code and with the one-line fix::

    python tools/reproduce_neg_rho_clip.py            # as shipped
    # then change chol_bounds to bind i == j and run it again

The simulation has known correlations (-0.60, 0.20, -0.30) and 60 studies:

* as shipped:  rho = ( 0.0000,  0.2854, -0.1272)
* with the fix: rho = (-0.3345,  0.3206, -0.2126)

The first pair is exactly zero as shipped, and the other two are estimated --
which is the signature of a bound on the first column and nowhere else.
"""

from __future__ import annotations

import warnings

import numpy as np

import pimeta as ma
from pimeta.data_compat import design_matrix

SEED = 20261029
K, J = 60, 3
SD = np.array([0.40, 0.40, 0.40])
RHO = np.array([-0.60, 0.20, -0.30])          # (1,0), (2,0), (2,1)
WITHIN_VAR = 0.01


def simulate():
    rng = np.random.default_rng(SEED)
    R = np.eye(J)
    pairs = [(i, j) for j in range(J - 1) for i in range(j + 1, J)]
    for value, (i, j) in zip(RHO, pairs):
        R[i, j] = R[j, i] = value
    G = np.outer(SD, SD) * R

    yi, vi, study, outcome = [], [], [], []
    for k in range(K):
        theta = rng.multivariate_normal(np.zeros(J), G)
        for j in range(J):
            yi.append(theta[j] + rng.normal(0.0, np.sqrt(WITHIN_VAR)))
            vi.append(WITHIN_VAR)
            study.append(f"s{k:03d}")
            outcome.append(f"o{j}")
    return np.array(yi), np.array(vi), np.array(study), np.array(outcome)


def fit():
    yi, vi, study, outcome = simulate()
    data = {"yi": yi, "vi": vi, "study": study, "outcome": outcome}
    X, _ = design_matrix("~ outcome - 1", data)
    # a diagonal sampling covariance: the only correlation in the model is the
    # between-study one, which is what the bounds act on
    V = ma.vcalc(vi, cluster=study, obs=outcome, rho=0.0)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return ma.rma_mv(yi, V=V, mods=X, data=data,
                         random="~ outcome | study", struct="UN",
                         intercept=False)


def main() -> int:
    result = fit()
    tau2 = np.asarray(result.tau2, dtype=float)
    rho = np.asarray(result.rho, dtype=float)
    print(f"true rho : {np.round(RHO, 4)}")
    print(f"fitted   : {np.round(rho, 4)}")
    print(f"tau2     : {np.round(tau2, 4)}")
    first_column = rho[0]
    if abs(first_column) < 1e-8:
        print("\nthe first pair is exactly zero: chol_bounds is clipping it")
        return 1
    print("\nthe first pair moved off the bound")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
