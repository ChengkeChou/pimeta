"""The exact posterior of the arm-level Bayesian NMA, without any sampler.

Run from the repository root::

    python tools/quadrature_nma_bayes.py

Why this exists.  Comparing two MCMC samplers on the same model answers "do the
samplers agree", not "is the model the one that was written down".  For this
model there is a stronger option, because the likelihood is Gaussian and the
priors on the treatment effects and the study baselines are Gaussian:

1. integrating each study's baseline :math:`\\mu_j` out of the likelihood leaves
   a Gaussian in :math:`d`, so :math:`p(d \\mid \\tau, y)` is **exactly**
   multivariate normal;
2. integrating :math:`d` out as well leaves :math:`p(y \\mid \\tau)`, a
   **scalar** function of :math:`\\tau`;
3. so the whole posterior is one one-dimensional integral over :math:`\\tau`,
   and
   :math:`E[d] = \\int E[d \\mid \\tau]\\, p(\\tau \\mid y)\\, d\\tau`
   with :math:`\\tau` on a grid.

No Metropolis, no NUTS, no warmup, no divergence, no effective sample size -- a
number that is exact to quadrature error.  That is what makes it the arbiter
when a sampler and a reference implementation disagree.

The derivation, so it can be checked rather than trusted:

.. math::

    y \\mid d, \\tau \\sim N(X d,\\ V_\\mu), \\qquad
    V_\\mu = \\Psi + s_\\mu^2 M M^{\\mathsf T},

with :math:`\\Psi` the block-diagonal arm variances (sampling variance plus
:math:`\\tau^2` on every arm except each study's pinned baseline arm) and
:math:`M` the study indicator matrix.  With :math:`d \\sim N(0, s_d^2 I)`,

.. math::

    \\log p(y \\mid \\tau) = -\\tfrac12\\left[
        \\log\\left|V_\\mu + s_d^2 X X^{\\mathsf T}\\right|
        + y^{\\mathsf T}(V_\\mu + s_d^2 XX^{\\mathsf T})^{-1} y\\right],

    p(d \\mid \\tau, y) = N(m, V), \\quad
    V = (X^{\\mathsf T}V_\\mu^{-1}X + s_d^{-2}I)^{-1}, \\quad
    m = V X^{\\mathsf T}V_\\mu^{-1} y.
"""

from __future__ import annotations

import csv
import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))

from pimeta.nma.bayes import _prepare  # noqa: E402

D_SD = 10.0
MU_SD = 100.0
TAU_SCALE = 0.5


def _design(data) -> np.ndarray:
    """The full study-indicator matrix that each baseline was integrated with."""
    m = data.y.size
    M = np.zeros((m, len(data.studies)))
    for j, s in enumerate(data.starts):
        end = data.starts[j + 1] if j + 1 < len(data.starts) else m
        M[s:end, j] = 1.0
    return M


def exact_posterior(y, v, study, treat, reference=None, *,
                    d_sd: float = D_SD, mu_sd: float = MU_SD,
                    tau_scale: float = TAU_SCALE,
                    random: bool = True, n_tau: int = 2001,
                    tau_max: float = 3.0) -> dict:
    """Posterior mean and SD of every parameter, by quadrature over ``tau``.

    Returns a dict with ``tau_mean``, ``tau_sd``, ``d_mean``, ``d_sd`` (one entry
    per basic parameter), ``treatments``, ``basic`` and ``reference``.
    """
    data = _prepare(y, None, v, study, treat, reference)
    X = data.X
    M = _design(data)
    one = ~data.is_baseline
    taus = np.array([0.0]) if not random else np.linspace(1e-8, tau_max, n_tau)

    logw = np.empty(taus.size)
    means = np.empty((taus.size, data.n_basic))
    covs = np.empty((taus.size, data.n_basic, data.n_basic))
    eye = np.eye(data.n_basic)
    for i, tau in enumerate(taus):
        psi = data.v + tau * tau * one
        V_mu = np.diag(psi) + mu_sd ** 2 * (M @ M.T)
        chol = np.linalg.cholesky(V_mu)
        S_inv_y = np.linalg.solve(chol.T, np.linalg.solve(chol, data.y))
        S_inv_X = np.linalg.solve(chol.T, np.linalg.solve(chol, X))

        # p(y | tau) after integrating d out as well
        quad_mat = V_mu + d_sd ** 2 * (X @ X.T)
        chol_q = np.linalg.cholesky(quad_mat)
        z = np.linalg.solve(chol_q, data.y)
        logw[i] = (-0.5 * float(np.sum(z * z))
                   - float(np.sum(np.log(np.diag(chol_q)))))

        prec = X.T @ S_inv_X + eye / d_sd ** 2
        cov = np.linalg.inv(prec)
        means[i] = cov @ (X.T @ S_inv_y)
        covs[i] = cov

    if random:
        log_prior = -0.5 * (taus / tau_scale) ** 2
        logw = logw + log_prior
    if taus.size == 1:
        wt = np.array([1.0])
    else:
        wt = np.full(taus.size, (taus[-1] - taus[0]) / (taus.size - 1))
        wt[0] *= 0.5
        wt[-1] *= 0.5
    w = np.exp(logw - logw.max())
    w = w / float(wt @ w)

    def integrate(f):
        return np.tensordot(wt * w, f, axes=(0, 0))

    d_mean = integrate(means)
    second = integrate(covs + means[:, :, None] * means[:, None, :])
    d_var = np.diag(second - np.outer(d_mean, d_mean))
    tau_mean = float(integrate(taus))
    tau_var = float(integrate((taus - tau_mean) ** 2))
    return {
        "treatments": data.treatments,
        "basic": data.basic,
        "reference": data.reference,
        "d_mean": d_mean,
        "d_sd": np.sqrt(np.maximum(d_var, 0.0)),
        "tau_mean": tau_mean if random else None,
        "tau_sd": np.sqrt(max(tau_var, 0.0)) if random else None,
    }


def main() -> None:
    root = pathlib.Path(__file__).resolve().parents[1]
    path = root / "tests" / "data" / "nma_bayes.csv"
    rows = list(csv.DictReader(path.open(encoding="utf-8")))
    y = np.array([float(r["yi"]) for r in rows])
    v = np.array([float(r["sei"]) for r in rows]) ** 2
    study = [r["study"] for r in rows]
    treat = [r["treat"] for r in rows]

    for label, random in (("fixed", False), ("random", True)):
        out = exact_posterior(y, v, study, treat, random=random)
        parts = "  ".join(f"d[{t}] {m:.4f} ({s:.4f})" for t, m, s in
                          zip(out["basic"], out["d_mean"], out["d_sd"]))
        print(f"{label:>6}: {parts}")
        if random:
            print(f"        tau {out['tau_mean']:.4f} ({out['tau_sd']:.4f})")


if __name__ == "__main__":
    main()
