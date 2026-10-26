"""Kenward-Roger degrees of freedom for a joint confidence region.

A joint region for a :math:`J`-dimensional pooled effect compares

.. math::

    (\\hat\\\\beta - \\\\beta)' \\\\hat V^{-1} (\\hat\\\\beta - \\\\beta)

against a reference distribution.  ``z`` and ``t`` both get the *denominator*
wrong: with ten studies, :math:`\\\\hat V` is estimated from ten clusters and the
between-study covariance is itself estimated, so the region is too narrow long
before the quantile matters.  Replacing :math:`k - p` by an exact Wishart
degrees of freedom over-corrects instead, because it charges the variance
parameters as if they were free.

Kenward & Roger (1997, *Biometrics* 53, 983–997) give the calibration in terms
of the *variance* parameters.  Write :math:`\\\\theta` for them (here the
between-study covariance's own parameters), :math:`W = \\\\mathcal{I}(\\\\hat\\\\theta)^{-1}`
for their asymptotic covariance and :math:`\\\\Phi = (X' M^{-1} X)^{-1}` for the
covariance of :math:`\\\\hat\\\\beta`.  Then

.. math::

    a_1 = \\\\sum_{rs} W_{rs} \\\\operatorname{tr}(\\\\Phi P_r)
                     \\\\operatorname{tr}(\\\\Phi P_s), \\\\qquad
    a_2 = \\\\sum_{rs} W_{rs} \\\\operatorname{tr}(\\\\Phi P_r \\\\Phi P_s),

with :math:`P_r = \\\\partial \\\\Phi^{-1} / \\\\partial \\\\theta_r`, and

.. math::

    B = \\\\frac{a_1 + 6a_2}{2J}, \\\\qquad
    g = \\\\frac{(J+1)a_1 - (J+4)a_2}{(J+2)a_2}, \\\\qquad
    m = 4 + \\\\frac{J+2}{JB - 1}.

The region is then referred to :math:`\\\\frac{Jm}{m-J+1} F_{J,\\\\, m-J+1}`
instead of :math:`\\\\chi^2_J`.  Everything is a trace of a `J x J` matrix: the
cost is a handful of refits to differentiate, not a new estimator.

Why this and not a parametric bootstrap: a bootstrap resamples at
:math:`\\\\hat\\\\theta`, so when :math:`\\\\hat\\\\theta` sits on a boundary — which with ten
studies it often does — every replicate is generated under too little
heterogeneity and the calibrated quantile comes out too small.  That failure
mode was measured for the univariate prediction interval (0.860 at `k = 5`), and
it is the reason this module takes the analytic route.

The derivatives are taken by refitting with the variance parameters pinned
(``rma_mv(..., sigma2=theta)``), so no hand-written tensor derivative is
needed and the same code path that fits the model supplies the curvature.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from scipy import stats

__all__ = ["kenward_roger_region"]


def _theta_of(G: np.ndarray) -> np.ndarray:
    """The free parameter vector ``(tau2_1..tau2_J, rho_pairs)`` of ``G``.

    ``rho_pairs`` follows ``_UNComp``'s column-major lower-triangular order,
    which is metafor's ``[lower.tri]``.
    """
    G = np.asarray(G, dtype=float)
    J = G.shape[0]
    tau2 = np.diag(G).astype(float)
    sd = np.sqrt(np.maximum(tau2, 0.0))
    rho = []
    for j in range(J - 1):
        for i in range(j + 1, J):
            denom = sd[i] * sd[j]
            rho.append(0.0 if denom <= 0.0 else float(G[i, j] / denom))
    return np.concatenate([tau2, np.asarray(rho, dtype=float)])


def _G_of(theta: np.ndarray, J: int) -> np.ndarray:
    """Rebuild ``G`` from the parameter vector, inverting :func:`_theta_of`."""
    tau2 = np.asarray(theta[:J], dtype=float)
    sd = np.sqrt(np.maximum(tau2, 0.0))
    G = np.zeros((J, J))
    G[np.diag_indices(J)] = tau2
    k = J
    for j in range(J - 1):
        for i in range(j + 1, J):
            G[i, j] = G[j, i] = theta[k] * sd[i] * sd[j]
            k += 1
    return G


def _refit(result: Any, theta: np.ndarray) -> Any:
    from ..rma_mv import rma_mv

    return rma_mv(np.asarray(result.yi, dtype=float), V=result.V,
                  mods=result.X, data=result._data, random=result.random,
                  struct=result.struct, intercept=result.intercept_incl,
                  method=result.method, sigma2=np.asarray(theta, dtype=float))


def _phi_inverse(result: Any) -> np.ndarray:
    """``X' M^{-1} X`` at the fitted point -- the inverse of ``Cov(beta_hat)``."""
    X = np.asarray(result.X, dtype=float)
    M = np.asarray(result.M, dtype=float)
    return X.T @ np.linalg.inv(M) @ X


def kenward_roger_region(result: Any, *, level: float = 0.95,
                         step: float | None = None) -> dict:
    """The KR-calibrated critical value for a joint region.

    Args:
        result: a fitted :class:`~pimeta.RmaMvResult` with an unstructured
            between-study covariance.  The parameter vector must be fully free
            — ``J`` variances and ``J(J-1)/2`` correlations — because the
            derivatives are taken with respect to all of them.
        level: confidence level, ``0.95`` or ``95``.
        step: finite-difference step for the variance parameters.  The default
            is ``1e-4 * max(1, |theta_r|)``, shrunk for a correlation so that
            ``rho +- step`` stays inside ``(-1, 1)``.

    Returns:
        A dict with ``crit`` (the critical value for the region statistic),
        ``df1``, ``df2``, ``m``, ``a1``, ``a2``, ``B``, ``g``, ``theta`` and
        ``level``.

    Raises:
        ValueError: the covariance is not fully free, the correlation sits on
            the boundary (where the derivative is not defined and the
            calibration has nothing to say), or the observed information is not
            positive definite.
    """
    Gmat, _names = result.g_matrix()
    if Gmat is None:
        raise ValueError(
            "the fit has no between-study covariance, so there are no variance "
            "parameters to differentiate")
    G = np.asarray(Gmat, dtype=float)
    J = G.shape[0]
    theta = _theta_of(G)
    n_var = J * (J + 1) // 2
    if theta.size != n_var:
        raise ValueError(
            f"expected {n_var} variance parameters for a J = {J} unstructured "
            f"covariance, got {theta.size}")
    # the round trip is the only check that the parameter order matches the
    # optimiser's; a silent mismatch would differentiate the wrong direction
    if not np.allclose(_G_of(theta, J), G, atol=1e-12, rtol=0.0):
        raise ValueError(
            "the between-study covariance does not round-trip through its "
            "parameter vector, so the derivatives would be taken in the wrong "
            "coordinates")

    for r in range(theta.size):
        if r >= J and abs(theta[r]) >= 1.0 - 1e-8:
            raise ValueError(
                "a between-study correlation sits on +/-1; the likelihood is "
                "flat in that direction and Kenward-Roger degrees of freedom "
                "are not defined there.  Use struct='HCS', or report the "
                "spread across structures (structure_sensitivity) instead")

    if step is None:
        steps = np.array([1e-4 * max(1.0, abs(t)) for t in theta])
    else:
        steps = np.full(theta.size, float(step))
    for r in range(J, theta.size):
        steps[r] = min(steps[r], 0.25 * (1.0 - abs(theta[r])))

    def ll(values: np.ndarray) -> float:
        return float(_refit(result, values).ll)

    def shifted(r: int, delta: float) -> np.ndarray:
        out = theta.copy()
        out[r] += delta
        return out

    # --- observed information for theta, by central differences ------------ #
    P = theta.size
    hess = np.zeros((P, P))
    for i in range(P):
        h = steps[i]
        hess[i, i] = (ll(shifted(i, h)) - 2.0 * ll(theta)
                      + ll(shifted(i, -h))) / (h * h)
        for j in range(i + 1, P):
            hj = steps[j]
            pp = theta.copy()
            pp[i] += h
            pp[j] += hj
            pm = theta.copy()
            pm[i] += h
            pm[j] -= hj
            mp = theta.copy()
            mp[i] -= h
            mp[j] += hj
            mm = theta.copy()
            mm[i] -= h
            mm[j] -= hj
            hess[i, j] = hess[j, i] = (ll(pp) - ll(pm) - ll(mp) + ll(mm)) / (
                4.0 * h * hj)
    if not np.all(np.isfinite(hess)):
        raise ValueError("the observed information is not finite")
    # ``ll`` is the log-likelihood, so the observed information is minus its
    # Hessian.  Deciding by the sign rather than assuming it means the same
    # function works for an ML and a REML criterion, and it fails loudly if the
    # point is a maximum of nothing.
    eigenvalues = np.linalg.eigvalsh(hess)
    if np.all(eigenvalues < 0.0):
        hess = -hess
    elif not np.all(eigenvalues > 0.0):
        raise ValueError(
            "the curvature of the likelihood at the fitted point is not "
            "definite, so the variance parameters have no asymptotic "
            "covariance to use")
    try:
        W = np.linalg.inv(hess)
    except np.linalg.LinAlgError as exc:
        raise ValueError(
            "the observed information for the variance parameters is "
            "singular, so their covariance is not available") from exc

    # --- P_r = d(X' M^-1 X)/d theta_r and Phi ----------------------------- #
    phi = np.linalg.inv(_phi_inverse(result))
    derivs = []
    for r in range(P):
        h = steps[r]
        derivs.append((_phi_inverse(_refit(result, shifted(r, h)))
                       - _phi_inverse(_refit(result, shifted(r, -h))))
                      / (2.0 * h))


    a1 = 0.0
    a2 = 0.0
    for r in range(P):
        for s in range(P):
            w = W[r, s]
            if w == 0.0:
                continue
            a1 += w * float(np.trace(phi @ derivs[r])) * float(
                np.trace(phi @ derivs[s]))
            a2 += w * float(np.trace(phi @ derivs[r] @ phi @ derivs[s]))
    if a2 <= 0.0:
        raise ValueError(
            "the Kenward-Roger second-order term is not positive, which means "
            "the variance parameters carry no information here")

    b = (a1 + 6.0 * a2) / (2.0 * J)
    g = ((J + 1.0) * a1 - (J + 4.0) * a2) / ((J + 2.0) * a2)
    alpha = 1.0 - float(level) / 100.0 if float(level) > 1.0 else 1.0 - float(
        level)
    denom = J * b - 1.0
    if denom <= 0.0:
        # JB <= 1 means the variance parameters are estimated precisely enough
        # that the correction has nothing left to say; the scaled F tends to
        # chi-square as m grows, so that is the honest limit rather than an
        # error.  ``m`` is reported as infinite so a caller can tell the two
        # routes apart.
        return {"crit": float(stats.chi2.ppf(1.0 - alpha, J)), "df1": J,
                "df2": float("inf"), "m": float("inf"), "a1": float(a1),
                "a2": float(a2), "B": float(b), "g": float(g),
                "theta": theta,
                "level": float(level) * (100.0 if float(level) <= 1.0 else 1.0)}
    m = 4.0 + (J + 2.0) / denom
    df2 = m - J + 1.0
    if df2 <= 0.0:
        raise ValueError("the Kenward-Roger denominator degrees of freedom "
                         f"is not positive ({df2:.3f})")
    crit = float(J * m / (m - J + 1.0)) * float(stats.f.ppf(1.0 - alpha, J, df2))
    return {"crit": crit, "df1": J, "df2": float(df2), "m": float(m),
            "a1": float(a1), "a2": float(a2), "B": float(b), "g": float(g),
            "theta": theta, "level": float(level) * (100.0 if float(level) <= 1.0 else 1.0)}
