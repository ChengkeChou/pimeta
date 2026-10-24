"""The exact posterior of the Bayesian location-scale model, without a sampler.

Run from the repository root::

    python tools/quadrature_dynhet.py

Why this exists.  Comparing a sampler with a sampler answers "do they agree",
not "is this the posterior that was written down".  For this model the answer
can be computed instead of sampled, for the same reason as in
``tools/quadrature_nma_bayes.py``:

1. given the scale coefficients :math:`\\alpha`, the model is Gaussian in the
   location coefficients :math:`\\beta`, so :math:`\\beta` can be integrated out
   under its Gaussian prior in closed form;
2. what is left is :math:`p(y \\mid \\alpha)`, an explicit function of two or
   three numbers;
3. so the posterior is a two- or three-dimensional integral on a grid.

No Metropolis, no warmup, no effective sample size.  The grid is chosen from a
frequentist fit (``tau_trend`` with ``test="z"``) as
:math:`\\hat\\alpha \\pm 8\\,se(\\hat\\alpha)`, wide enough that the tails carry
no mass and narrow enough that 401 points per axis resolve the mode;
``tests/test_dynhet.py`` checks that doubling the range changes nothing.
"""

from __future__ import annotations

import csv
import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))

from pimeta.dynamichet import (  # noqa: E402
    _BAYES_PRIORS,
    _design,
    tau_trend,
)

__all__ = ["exact_posterior"]


def exact_posterior(yi, vi, time, *, mu: str = "linear", tau: str = "linear",
                    prior_beta_sd: float | None = None,
                    prior_alpha0_sd: float | None = None,
                    prior_alpha_sd: float | None = None,
                    n_grid: int = 401, width: float = 8.0) -> dict:
    """Posterior mean and SD of every parameter, by quadrature over ``alpha``.

    Returns a dict with ``alpha_mean``, ``alpha_sd``, ``beta_mean``, ``beta_sd``,
    ``tau2_mean_at_centre``, ``mu_at`` (a callable taking times) and the grid it
    used.
    """
    y = np.asarray(yi, dtype=float).ravel()
    v = np.asarray(vi, dtype=float).ravel()
    t = np.asarray(time, dtype=float).ravel()
    centre = float(t.mean())
    X = _design(t, mu, centre)
    Z = _design(t, tau, centre)
    p = X.shape[1]
    q = Z.shape[1]

    beta_sd = _BAYES_PRIORS["beta_sd"] if prior_beta_sd is None else prior_beta_sd
    a0_sd = _BAYES_PRIORS["alpha0_sd"] if prior_alpha0_sd is None else prior_alpha0_sd
    a_sd = _BAYES_PRIORS["alpha_sd"] if prior_alpha_sd is None else prior_alpha_sd

    start = tau_trend(y, v, t, mu=mu, tau=tau, method="ML", test="z")
    centre_alpha = start.alpha.copy()
    spread = np.maximum(start.se_alpha * width, 0.25)
    axes = [np.linspace(centre_alpha[j] - spread[j],
                        centre_alpha[j] + spread[j], n_grid)
            for j in range(q)]
    mesh = np.meshgrid(*axes, indexing="ij")
    grid = np.stack([m.ravel() for m in mesh], axis=1)      # (n_points, q)

    def log_post(alpha: np.ndarray) -> float:
        d = v + np.exp(Z @ alpha)
        cov = np.diag(d) + beta_sd ** 2 * (X @ X.T)
        chol = np.linalg.cholesky(cov)
        z = np.linalg.solve(chol, y)
        value = -0.5 * float(z @ z) - float(np.sum(np.log(np.diag(chol))))
        value -= 0.5 * (alpha[0] / a0_sd) ** 2
        if q > 1:
            value -= 0.5 * float(alpha[1:] @ alpha[1:]) / a_sd ** 2
        return value

    logw = np.array([log_post(a) for a in grid])
    w = np.exp(logw - logw.max())
    if q == 1:
        dx = axes[0][1] - axes[0][0]
        volume = dx
    else:
        volume = float(np.prod([ax[1] - ax[0] for ax in axes]))
    w = w * volume
    total = float(w.sum())
    w = w / total

    alpha_mean = w @ grid
    alpha_sd = np.sqrt(w @ (grid - alpha_mean) ** 2)

    # E[beta] = E_{alpha}[ m(alpha) ], and Var = E[V(alpha)] + Var[m(alpha)]
    eye = np.eye(p)
    means = np.empty((grid.shape[0], p))
    covs = np.empty((grid.shape[0], p, p))
    for i, alpha in enumerate(grid):
        d = v + np.exp(Z @ alpha)
        prec = X.T @ (X / d[:, None]) + eye / beta_sd ** 2
        cov = np.linalg.inv(prec)
        means[i] = cov @ (X.T @ (y / d))
        covs[i] = cov
    beta_mean = w @ means
    second = np.einsum("k,kij->ij", w, covs) + np.einsum(
        "k,ki,kj->ij", w, means - beta_mean, means - beta_mean)
    beta_sd = np.sqrt(np.diag(second))
    tau2_mean = float(w @ np.exp(grid[:, 0]))

    def mu_at(times):
        Xn = _design(np.atleast_1d(np.asarray(times, dtype=float)), mu, centre)
        point = Xn @ beta_mean
        cov_pt = np.einsum("k,kij->ij", w, covs) + np.einsum(
            "k,ki,kj->ij", w, means - beta_mean, means - beta_mean)
        var = np.maximum(np.sum((Xn @ cov_pt) * Xn, axis=1), 0.0)
        return {"mu": point, "se": np.sqrt(var)}

    return {"alpha_mean": alpha_mean, "alpha_sd": alpha_sd,
            "beta_mean": beta_mean, "beta_sd": beta_sd,
            "tau2_mean_at_centre": tau2_mean, "mu_at": mu_at,
            "n_grid": int(grid.shape[0]), "width": float(width),
            "bounds": [(float(a[0]), float(a[-1])) for a in axes]}


def main() -> None:
    root = pathlib.Path(__file__).resolve().parents[1]
    rows = list(csv.DictReader((root / "tests" / "data" / "dynhet.csv").open(
        encoding="utf-8")))
    y = np.array([float(r["yi"]) for r in rows])
    v = np.array([float(r["vi"]) for r in rows])
    t = np.array([float(r["year"]) for r in rows])
    out = exact_posterior(y, v, t)
    print("exact alpha:", np.round(out["alpha_mean"], 4),
          "+-", np.round(out["alpha_sd"], 4))
    print("exact beta :", np.round(out["beta_mean"], 4),
          "+-", np.round(out["beta_sd"], 4))
    print("exact tau2 at the mean time:", round(out["tau2_mean_at_centre"], 4))
    print("grid:", out["n_grid"], "points, bounds", out["bounds"])


if __name__ == "__main__":
    main()
