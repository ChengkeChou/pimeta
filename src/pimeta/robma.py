"""Bayesian model averaging over publication-bias models (RoBMA).

The frequentist funnel-plot methods in :mod:`pimeta.pub_bias` answer one
question at a time — *is* the funnel asymmetric, *how much* would PET or PEESE
shift the estimate — and each answer is conditional on that one model being
right. RoBMA replaces the choice with a prior over models and reports the
posterior, so "there is no bias" and "the selection model fits better" compete
on the same footing instead of one being the null hypothesis.

This module reproduces the default model space of the R package `RoBMA`
(Bartoš, Maier, Quintana & Wagenmakers) version 4.0.0, **and checks itself
against that package**, but it does so by a different route: the marginal
likelihood of every model is computed by numerical integration, so the model
posterior probabilities are exact up to quadrature error rather than exact up to
Monte Carlo error.

The model space
---------------

A model is a triple:

* **Bias.** Nine possibilities — no bias, six Vevea–Hedges step selection
  functions, PET and PEESE.
* **Effect.** :math:`\\mu = 0` (point mass) or :math:`\\mu \\sim N(0, 1/2)`,
  with prior weights one to one.
* **Heterogeneity.** :math:`\\tau = 0` or
  :math:`\\tau \\sim N(0, 1/8)` on :math:`[0, \\infty)`, again one to one.

That is :math:`9 \\times 2 \\times 2 = 36` models — RoBMA's "PSMA" ensemble. The
prior weights over the nine bias models are the ones the reference uses: 1 for
no bias, :math:`1/12` for each step function and :math:`1/4` for PET and for
PEESE, which normalise to :math:`1/2`, :math:`1/24` and :math:`1/8`.

Why the priors are on the effect scale
--------------------------------------

`RoBMA` 4.0 takes a ``measure`` and an ``ni`` argument and the documentation
suggests the priors are rescaled by a unit-information standard deviation
:math:`\\sqrt{\\sum n_i / \\sum \\sigma_i^{-2}}`. They are not, in the default
call: with the same data and ``ni`` multiplied by 100 — which moves that
standard deviation from 2.0 to 20.0 — the posterior mean and standard deviation
of :math:`\\mu` are identical to three decimals, and the prior object records
``sd = 0.7071068`` in both fits. The priors are therefore constants on the
``yi`` scale, and that is what is implemented here. This is a measurement, not
an assumption, and `tests/test_robma.py` pins it.

The likelihood
--------------

Without selection the model is the ordinary random-effects model,
:math:`y_i \\sim N(m_i, \\sigma_i^2 + \\tau^2)`, with
:math:`m_i = \\mu` (no bias), :math:`\\mu + \\beta \\sigma_i` (PET) or
:math:`\\mu + \\beta \\sigma_i^2` (PEESE), and :math:`\\beta \\ge 0`.

A step selection function partitions the p-value scale into bins at ``steps``
and gives bin :math:`j` a weight :math:`\\omega_j`, so that a study in bin
:math:`j` is published with probability proportional to :math:`\\omega_j`. The
density is then

.. math::

    f_{\\text{sel}}(y_i) =
    \\frac{\\omega_{j(i)}\\,\\phi(y_i; m_i, s_i)}
         {\\sum_j \\omega_j \\pi_{ij}},
    \\qquad s_i^2 = \\sigma_i^2 + \\tau^2,

where :math:`\\pi_{ij}` is the probability that a study with standard error
:math:`s_i` falls in bin :math:`j`. The normalisation makes the weights
invariant to a common rescaling, and the reference most-significant bin is fixed
at :math:`\\omega_1 = 1`.

What this module does not do
----------------------------

It does not sample. The default space is small enough to integrate, and an
exact answer is worth more than a fast one here, because the whole point of the
method is a posterior *probability over models* — a quantity an MCMC run can
only estimate with an error that shows up as "Bayes factor MC error 21%" in the
reference's own output. Custom and larger spaces are not covered: pass a subset
through ``bias_models`` and the space shrinks, but it does not grow.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy import stats
from scipy.special import logsumexp

__all__ = [
    "RobmaResult",
    "WeightFunction",
    "robma",
    "step_weight_function",
]


# --------------------------------------------------------------------------- #
# the model space
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class WeightFunction:
    """One Vevea–Hedges step selection function.

    Attributes:
        side: ``"one-sided"`` or ``"two-sided"`` p-values.
        steps: the p-value cut points. They define ``len(steps) + 1`` bins and
            must increase inside ``(0, 1)``.
    """

    side: str
    steps: tuple[float, ...]

    def __post_init__(self) -> None:
        steps = tuple(float(s) for s in self.steps)
        object.__setattr__(self, "steps", steps)
        if self.side not in ("one-sided", "two-sided"):
            raise ValueError("'side' must be 'one-sided' or 'two-sided'")
        if not steps:
            raise ValueError("'steps' must contain at least one cut point")
        if any(not 0.0 < s < 1.0 for s in steps):
            raise ValueError("'steps' must lie strictly inside (0, 1)")
        if list(steps) != sorted(steps):
            raise ValueError("'steps' must be increasing")

    @property
    def n_bins(self) -> int:
        """Bins on the p-value scale."""
        return len(self.steps) + 1

    @property
    def name(self) -> str:
        return f"{self.side}: {', '.join(f'{s:g}' for s in self.steps)}"

    def __str__(self) -> str:
        return f"omega[{self.name}]"


def step_weight_function(side: str = "one-sided",
                         steps=(0.025, 0.05)) -> WeightFunction:
    """Build a step selection function.

    Args:
        side: ``"one-sided"`` or ``"two-sided"``.
        steps: p-value cut points, increasing, inside ``(0, 1)``.

    Returns:
        :class:`WeightFunction`.

    Example:
        >>> wf = ma.step_weight_function("two-sided", [0.05])
        >>> wf.n_bins
        2
    """
    return WeightFunction(side=side, steps=tuple(steps))


#: The six step functions of RoBMA's default ensemble, in the reference's order.
DEFAULT_STEP_FUNCTIONS: tuple[WeightFunction, ...] = (
    WeightFunction("two-sided", (0.05,)),
    WeightFunction("two-sided", (0.05, 0.1)),
    WeightFunction("one-sided", (0.05,)),
    WeightFunction("one-sided", (0.025, 0.05)),
    WeightFunction("one-sided", (0.05, 0.5)),
    WeightFunction("one-sided", (0.025, 0.05, 0.5)),
)

#: Bias-model names in the default ensemble, and their prior weights.
BIAS_PRIOR_WEIGHTS: dict[str, float] = {
    "none": 1.0,
    **{wf.name: 1.0 / 12.0 for wf in DEFAULT_STEP_FUNCTIONS},
    "PET": 1.0 / 4.0,
    "PEESE": 1.0 / 4.0,
}

#: Prior hyperparameters, read off `RoBMA` 4.0.0 rather than from the papers.
DEFAULTS: dict[str, float] = {
    "mu_sd": 0.7071067811865476,      # 1 / sqrt(2)
    "tau_sd": 0.35355339059327373,    # 1 / (2 sqrt(2))
    "pet_scale": 1.0,                 # half-Cauchy
    "peese_scale": 5.0,               # half-Cauchy
}

#: Grid resolution. The defaults hold four significant digits against a doubled
#: grid on the Gaussian models; the ``omega`` axis is the coarsest because it is
#: the one that multiplies, and a selection model with four bins is a
#: five-dimensional integral.
DEFAULT_GRID: dict[str, int] = {
    "tau": 121,
    "beta": 81,
    "mu": 61,
    "omega": 7,
}


# --------------------------------------------------------------------------- #
# numerics
# --------------------------------------------------------------------------- #
def _legendre(a: float, b: float, n: int) -> tuple[np.ndarray, np.ndarray]:
    """Gauss–Legendre nodes and weights on ``[a, b]``."""
    x, w = np.polynomial.legendre.leggauss(int(n))
    half = 0.5 * (b - a)
    return a + half * (x + 1.0), half * w


def _log_normal(y: np.ndarray, mean: np.ndarray, sd: np.ndarray) -> np.ndarray:
    z = (y - mean) / sd
    return -0.5 * z * z - np.log(sd) - 0.9189385332046728  # log sqrt(2 pi)


def _log_half_normal(x: float, sd: float) -> float:
    if x <= 0.0:
        return -np.inf
    return float(np.log(2.0) - 0.9189385332046728 - np.log(sd)
                 - 0.5 * (x / sd) ** 2)


def _log_half_cauchy(x: float, scale: float) -> float:
    if x < 0.0:
        return -np.inf
    return float(np.log(2.0) - np.log(np.pi * scale)
                 - np.log1p((x / scale) ** 2))


def _z_from_p(p: float) -> float:
    """The normal quantile whose upper tail is ``p``."""
    return float(stats.norm.isf(p))


def _bin_probabilities(mu, sei, s, wf: WeightFunction) -> np.ndarray:
    """Probability of each p-value bin, shape ``(..., n_bins)``.

    The bin edges are placed on the study's **own** p-value scale, so they sit at
    ``sei * z``, while the probability of falling between two of them is taken
    from the marginal distribution ``N(mu, s)`` with
    ``s**2 = sei**2 + tau**2``. The two scales are not interchangeable and
    confusing them produces plausible-looking numbers, which is why they are
    separate arguments.
    """
    arrays = np.broadcast_arrays(np.asarray(mu, dtype=float),
                                 np.asarray(sei, dtype=float),
                                 np.asarray(s, dtype=float))
    mu, sei, s = arrays
    z = np.array([_z_from_p(step) for step in wf.steps], dtype=float)
    edges = sei[..., None] * z.reshape((1,) * (sei.ndim) + (-1,))
    if wf.side == "two-sided":
        upper = stats.norm.sf(edges, mu[..., None], s[..., None])
        lower = stats.norm.cdf(-edges, mu[..., None], s[..., None])
        inside = upper + lower
    else:
        inside = stats.norm.sf(edges, mu[..., None], s[..., None])
    # cum[j] = P(p-value above cut j), so the bins are the differences of
    # (1, cum..., 0); dropping the trailing zero would silently lose the last
    # bin and leave the likelihood with the wrong number of columns
    cumulative = np.concatenate(
        [np.ones(mu.shape + (1,)), inside, np.zeros(mu.shape + (1,))],
        axis=-1)
    return -np.diff(cumulative, axis=-1)


def _observed_bin(yi: np.ndarray, sei: np.ndarray,
                  wf: WeightFunction) -> np.ndarray:
    """Which p-value bin each study's observed effect falls in."""
    if wf.side == "two-sided":
        p = 2.0 * stats.norm.sf(np.abs(yi) / sei)
    else:
        p = stats.norm.sf(yi / sei)
    return np.searchsorted(np.asarray(wf.steps), p, side="left")


def _omega_from_cube(u: np.ndarray, n_bins: int) -> np.ndarray:
    """Monotone decreasing weights from free coordinates in ``(0, 1)``.

    ``u`` has shape ``(..., n_bins - 1)``. The first bin is the reference and has
    weight one; each later weight is the previous one times a fraction in
    ``(0, 1)``, so the sequence decreases and the prior on ``u`` is uniform on
    the unit cube.
    """
    ones = np.ones(u.shape[:-1] + (1,))
    return np.concatenate([ones, np.cumprod(u, axis=-1)], axis=-1)


# --------------------------------------------------------------------------- #
# per-model marginal likelihoods
# --------------------------------------------------------------------------- #
def _loglik_selection(yi, sei, mu, tau2, omega, wf, obs_bin) -> np.ndarray:
    """Log likelihood of a step selection model, shape ``(n_tau, n_omega)``.

    ``mu`` is a scalar, ``tau2`` a vector of length ``n_tau`` and ``omega`` a
    matrix ``(n_omega, n_bins)`` of weight vectors. The bin edges sit on the
    study's own standard error while the distribution they are integrated
    against is the marginal ``N(mu, sqrt(sei^2 + tau2))``; the two scales are
    separate arguments because using one for both produces plausible numbers.
    """
    tau2 = np.atleast_1d(np.asarray(tau2, dtype=float))
    omega = np.atleast_2d(np.asarray(omega, dtype=float))
    s = np.sqrt(np.asarray(sei, dtype=float) ** 2 + tau2[:, None])   # (T, K)
    probs = _bin_probabilities(mu, sei, s, wf)                       # (T, K, B)
    denom = np.sum(omega[None, :, None, :] * probs[:, None, :, :],
                   axis=-1)                                          # (T, U, K)
    picked = omega[None, :, obs_bin]                                 # (1, U, K)
    with np.errstate(divide="ignore", invalid="ignore"):
        log_ratio = np.log(picked) - np.log(denom)
    base = _log_normal(yi, np.full_like(s, float(mu)), s)[:, None, :]  # (T,1,K)
    return np.sum(log_ratio + base, axis=-1)                         # (T, U)


def _mu_box(yi, sei, mu_sd: float) -> tuple[float, float]:
    """A start box wide enough that the prior and the data are both inside."""
    lo = min(-6.0 * mu_sd, float(np.min(yi)) - 6.0 * float(np.max(sei)))
    hi = max(6.0 * mu_sd, float(np.max(yi)) + 6.0 * float(np.max(sei)))
    return lo, hi


def _mu_grid(mu_sd: float, n: int, lo: float,
             hi: float) -> tuple[np.ndarray, np.ndarray]:
    """Nodes and log prior weights for ``mu`` on ``[lo, hi]``."""
    nodes, weights = _legendre(float(lo), float(hi), n)
    return nodes, _log_normal(nodes, np.zeros_like(nodes),
                              np.full_like(nodes, mu_sd)) + np.log(weights)


def _cube_grid(n: int, dim: int) -> tuple[np.ndarray, np.ndarray]:
    """Nodes and log weights on the unit cube, one axis per free weight."""
    if dim <= 0:
        return np.zeros((1, 0)), np.zeros(1)
    nodes, weights = _legendre(0.0, 1.0, int(n))
    grids = np.meshgrid(*([nodes] * dim), indexing="ij")
    wgrid = np.meshgrid(*([weights] * dim), indexing="ij")
    u = np.stack([g.reshape(-1) for g in grids], axis=-1)
    w = np.ones(u.shape[0])
    for g in wgrid:
        w = w * g.reshape(-1)
    return u, np.log(w)


def _tau_grid(tau_sd: float, n: int, lo: float,
              hi: float) -> tuple[np.ndarray, np.ndarray]:
    """Nodes and log prior weights for ``tau`` on ``[lo, hi]``."""
    nodes, weights = _legendre(float(lo), float(hi), n)
    log_prior = np.array([_log_half_normal(t, tau_sd) + np.log(w)
                          for t, w in zip(nodes, weights)])
    if not np.all(np.isfinite(log_prior)):
        raise RuntimeError("the tau prior is zero on part of its own grid")
    return nodes, log_prior


def _beta_grid(scale: float, n: int, lo: float,
               hi: float) -> tuple[np.ndarray, np.ndarray]:
    """Nodes and log prior weights for a non-negative PET/PEESE coefficient."""
    nodes, weights = _legendre(float(lo), float(hi), n)
    log_prior = np.array([_log_half_cauchy(b, scale) + np.log(w)
                          for b, w in zip(nodes, weights)])
    if not np.all(np.isfinite(log_prior)):
        raise RuntimeError("the PET/PEESE prior is zero on part of its grid")
    return nodes, log_prior


# --------------------------------------------------------------------------- #
# one model at a time
# --------------------------------------------------------------------------- #
_LOG_2PI = 1.8378770664093453


@dataclass
class _Fit:
    """What integrating one model produced.

    The effect is integrated out analytically, so its posterior is not a grid
    but a Gaussian mixture: one component per ``(tau, beta)`` node, with the
    node's posterior weight.
    """

    log_marginal: float
    mu_mean: np.ndarray | None = None
    mu_sd: np.ndarray | None = None
    mu_logw: np.ndarray | None = None
    mu_grid_nodes: np.ndarray | None = None
    mu_grid_logw: np.ndarray | None = None
    tau_nodes: np.ndarray | None = None
    tau_logw: np.ndarray | None = None


def _gauss_conditional(yi, sei, tau_nodes, x, beta_nodes, mu_sd, effect_free):
    """Marginalise ``mu`` analytically, and return its conditional posterior.

    With a normal prior on :math:`\\mu` and a normal likelihood the integral has
    a closed form, by the matrix determinant lemma applied to
    :math:`V + \\sigma_\\mu^2 \\mathbf{1}\\mathbf{1}'`. Integrating it out rather
    than gridding it is not an optimisation: the prior is ten times wider than
    the posterior on real data, so a uniform grid that covers the prior has no
    resolution left where the mass is.

    Returns arrays of shape ``(n_tau, n_beta)``.
    """
    tau2 = (np.asarray(tau_nodes, dtype=float) ** 2)[:, None, None]
    var = np.asarray(sei, dtype=float) ** 2 + tau2          # (n_tau, 1, K)
    resid = (np.asarray(yi, dtype=float)
             - np.asarray(beta_nodes, dtype=float)[None, :, None] * x)
    inv = 1.0 / var
    s1 = np.sum(inv, axis=-1)
    s2 = np.sum(resid * inv, axis=-1)
    q = np.sum(resid * resid * inv, axis=-1)
    log_det = np.sum(np.log(var), axis=-1)
    base = -0.5 * (np.size(yi) * _LOG_2PI + log_det)
    if not effect_free:
        return base - 0.5 * q, np.zeros_like(s1), np.ones_like(s1)
    a = mu_sd ** 2
    log_int = (base
               - 0.5 * np.log1p(a * s1)
               - 0.5 * (q - a * s2 ** 2 / (1.0 + a * s1)))
    precision = s1 + 1.0 / a
    # the conditional sd of mu does not depend on beta, so it arrives with the
    # tau axis only; broadcast it to the joint grid before it is stacked
    sd = np.broadcast_to(1.0 / np.sqrt(precision), s2.shape)
    return log_int, s2 / precision, sd


def _trim(nodes: np.ndarray, logmass: np.ndarray, *, lower_zero: bool,
          keep: float = 1.0 - 1e-7, pad: float = 0.15) -> tuple[float, float]:
    """Interval of ``nodes`` holding ``keep`` of the mass, widened a little."""
    m = np.exp(logmass - np.nanmax(logmass))
    total = float(np.sum(m))
    if total <= 0.0 or not np.isfinite(total):
        return float(nodes[0]), float(nodes[-1])
    cdf = np.cumsum(m) / total
    tail = 0.5 * (1.0 - keep)
    idx = np.clip(np.searchsorted(cdf, np.array([tail, 1.0 - tail])),
                  0, nodes.size - 1)
    lo = float(nodes[idx[0]])
    hi = float(nodes[idx[1]])
    if hi <= lo:
        hi = lo + max(abs(lo), 1e-3)
    width = (hi - lo) * pad
    lo, hi = lo - width, hi + width
    if lower_zero:
        lo = max(lo, 0.0)
    return lo, hi


def _fit_gaussian(yi, sei, *, kind: str, effect_free: bool,
                  hetero_free: bool, prior: dict, grid: dict) -> _Fit:
    """Integrate one Gaussian model: no bias, PET or PEESE."""
    k = yi.size
    if kind == "none":
        beta_nodes, logw_beta = np.zeros(1), np.zeros(1)
        x = np.zeros(k)
        beta_scale = None
    else:
        beta_scale = float(prior["pet_scale"] if kind == "PET"
                           else prior["peese_scale"])
        x = sei if kind == "PET" else sei ** 2
        lo, hi = 0.0, 20.0 * beta_scale
        beta_nodes, logw_beta = _beta_grid(beta_scale, grid["beta"], lo, hi)

    if hetero_free:
        tau_hi = max(6.0 * prior["tau_sd"], 2.0 * float(np.max(sei)))
        tau_nodes, logw_tau = _tau_grid(prior["tau_sd"], grid["tau"],
                                        0.0, tau_hi)
    else:
        tau_nodes, logw_tau = np.zeros(1), np.zeros(1)

    # Two passes: the second places both grids on the region that carries the
    # mass, which the prior alone does not identify.
    total = log_int = m_hat = s_hat = np.zeros(1)
    passes = 3
    for it in range(passes):
        log_int, m_hat, s_hat = _gauss_conditional(
            yi, sei, tau_nodes, x, beta_nodes, float(prior["mu_sd"]),
            effect_free)
        total = log_int + logw_tau[:, None] + logw_beta[None, :]
        if not hetero_free and len(beta_nodes) == 1:
            break
        if it == passes - 1:
            break          # the last pass is the answer; do not move the grid
        if hetero_free:
            lo, hi = _trim(tau_nodes, logsumexp(total, axis=1), lower_zero=True)
            tau_nodes, logw_tau = _tau_grid(prior["tau_sd"], grid["tau"],
                                            lo, hi)
        if len(beta_nodes) > 1:
            lo, hi = _trim(beta_nodes, logsumexp(total, axis=0),
                           lower_zero=True)
            assert beta_scale is not None
            beta_nodes, logw_beta = _beta_grid(beta_scale, grid["beta"],
                                               lo, hi)

    log_marginal = float(logsumexp(total))
    flat_w = total.reshape(-1)
    keep = flat_w > np.max(flat_w) - 60.0
    return _Fit(
        log_marginal=log_marginal,
        mu_mean=(m_hat.reshape(-1)[keep] if effect_free else None),
        mu_sd=(s_hat.reshape(-1)[keep] if effect_free else None),
        mu_logw=(flat_w[keep] - log_marginal if effect_free else None),
        tau_nodes=tau_nodes if hetero_free else None,
        tau_logw=(logsumexp(total, axis=1) if hetero_free else None),
    )


def _fit_model(yi, sei, *, bias, effect_free: bool, hetero_free: bool,
               prior: dict, grid: dict) -> _Fit:
    if isinstance(bias, WeightFunction):
        return _fit_selection(yi, sei, wf=bias, effect_free=effect_free,
                              hetero_free=hetero_free, prior=prior, grid=grid)
    if bias in ("none", "PET", "PEESE"):
        return _fit_gaussian(yi, sei, kind=bias, effect_free=effect_free,
                             hetero_free=hetero_free, prior=prior, grid=grid)
    raise ValueError(f"unknown bias model {bias!r}")


def _fit_selection(yi, sei, *, wf: WeightFunction, effect_free: bool,
                   hetero_free: bool, prior: dict, grid: dict,
                   omega_fixed=None) -> _Fit:
    """Integrate a step selection model over ``(mu, tau, omega)``.

    ``mu`` cannot be integrated out here the way it can when the model is
    Gaussian: the selection normalisation depends on it, so the marginal of
    ``mu`` is carried on a grid and combined later with the normal components
    that the bias-free models produce.
    """
    obs_bin = _observed_bin(yi, sei, wf)
    n_omega = wf.n_bins - 1
    tau_hi = max(6.0 * prior["tau_sd"], 2.0 * float(np.max(sei)))
    if hetero_free:
        tau_nodes, logw_tau = _tau_grid(prior["tau_sd"], grid["tau"],
                                        0.0, tau_hi)
    else:
        tau_nodes, logw_tau = np.zeros(1), np.zeros(1)
    if omega_fixed is None:
        u_nodes, logw_u = _cube_grid(grid["omega"], n_omega)
        omegas = _omega_from_cube(u_nodes, wf.n_bins)
    else:
        # a fixed weight vector removes omega from the model entirely, which is
        # how the kernel can be checked without a prior in the way
        omegas = np.atleast_2d(np.asarray(omega_fixed, dtype=float))
        logw_u = np.zeros(omegas.shape[0])

    lo, hi = _mu_box(yi, sei, float(prior["mu_sd"]))
    n_mu = int(grid["mu"]) if effect_free else 1
    mu_nodes, logw_mu = _mu_grid(float(prior["mu_sd"]), n_mu, lo, hi)

    total = np.zeros(1)
    tau_mass = np.full(1, -np.inf)
    passes = 3
    for it in range(passes):
        parts = []
        # log(0), not log(1): accumulating from a vector of zeros gives every tau
        # a unit of probability before any likelihood is added, and a log
        # likelihood of -100 is invisible next to it
        tau_mass = np.full(tau_nodes.size, -np.inf)
        for m, lwm in zip(mu_nodes, logw_mu):
            ll = _loglik_selection(yi, sei, float(m), tau_nodes ** 2,
                                   omegas, wf, obs_bin)
            joint = ll + logw_tau[:, None] + logw_u[None, :]
            parts.append(logsumexp(joint) + lwm)
            tau_mass = np.logaddexp(tau_mass, logsumexp(joint, axis=1))
        total = np.asarray(parts)
        if it == passes - 1:
            break          # the last pass is the answer; do not move the grid
        if effect_free:
            lo, hi = _trim(mu_nodes, total, lower_zero=False)
            mu_nodes, logw_mu = _mu_grid(float(prior["mu_sd"]), n_mu, lo, hi)
        if hetero_free:
            tlo, thi = _trim(tau_nodes, tau_mass, lower_zero=True)
            tau_nodes, logw_tau = _tau_grid(prior["tau_sd"], grid["tau"],
                                            tlo, thi)

    log_marginal = float(logsumexp(total))
    return _Fit(
        log_marginal=log_marginal,
        mu_grid_nodes=mu_nodes if effect_free else None,
        mu_grid_logw=(total - log_marginal if effect_free else None),
        tau_nodes=tau_nodes if hetero_free else None,
        tau_logw=(tau_mass if hetero_free else None),
    )


# --------------------------------------------------------------------------- #
# posterior summaries of a mixture with a point mass
# --------------------------------------------------------------------------- #
def _mixture_summary(means: list[np.ndarray], sds: list[np.ndarray],
                     weights: list[np.ndarray], point_mass: float,
                     point_value: float, level: float,
                     grid_terms=None) -> dict:
    """Mean, SD and equal-tailed interval of a mixed posterior plus a spike.

    Two kinds of continuous component arrive here. When the model is Gaussian
    the effect was integrated out analytically and each ``(tau, beta)`` node
    leaves a normal density behind; when the model selects on the p-value the
    normalisation depends on :math:`\\mu`, so its marginal is a grid. Both are
    handled. ``level`` is a fraction — 0.95, not 95.
    """
    tail = (1.0 - level) / 2.0
    grid_terms = list(grid_terms or [])
    if not means and not grid_terms:
        return {"mean": float(point_value), "sd": 0.0,
                "q_low": float(point_value), "q_high": float(point_value),
                "point_mass": float(point_mass), "prob_positive": 0.0}

    if means:
        comp_mu = np.concatenate(means)
        comp_sd = np.concatenate(sds)
        comp_w = np.exp(np.concatenate(weights))
    else:
        comp_mu = np.zeros(0)
        comp_sd = np.ones(0)
        comp_w = np.zeros(0)
    g_nodes = [np.asarray(t[0], dtype=float) for t in grid_terms]
    g_w = [np.exp(np.asarray(t[1], dtype=float)) for t in grid_terms]

    # the weights already carry each model's posterior probability, so the whole
    # continuous part is rescaled at once rather than per family
    continuous = float(np.sum(comp_w)) + sum(float(np.sum(w)) for w in g_w)
    scale = (1.0 - point_mass) / continuous
    comp_w = comp_w * scale
    g_w = [w * scale for w in g_w]

    mean = float(np.sum(comp_w * comp_mu)
                 + sum(float(np.sum(w * n)) for n, w in zip(g_nodes, g_w))
                 + point_mass * point_value)
    second = float(np.sum(comp_w * (comp_sd ** 2 + comp_mu ** 2))
                   + sum(float(np.sum(w * n ** 2))
                         for n, w in zip(g_nodes, g_w))
                   + point_mass * point_value ** 2)
    sd_total = float(np.sqrt(max(second - mean ** 2, 0.0)))

    def cdf(x: float) -> float:
        value = 0.0
        if comp_mu.size:
            value += float(np.sum(comp_w * stats.norm.cdf((x - comp_mu)
                                                          / comp_sd)))
        for n, w in zip(g_nodes, g_w):
            value += float(np.sum(w[n <= x]))
        if point_value <= x:
            value += point_mass
        return value

    if comp_mu.size:
        lo = float(np.min(comp_mu - 10.0 * comp_sd))
        hi = float(np.max(comp_mu + 10.0 * comp_sd))
    else:
        lo, hi = float("inf"), float("-inf")
    for n in g_nodes:
        lo = min(lo, float(np.min(n)))
        hi = max(hi, float(np.max(n)))
    q_low = _invert_cdf(cdf, tail, lo, hi)
    q_high = _invert_cdf(cdf, 1.0 - tail, lo, hi)
    return {"mean": mean, "sd": sd_total, "q_low": q_low, "q_high": q_high,
            "point_mass": float(point_mass),
            "prob_positive": 1.0 - cdf(0.0)}


def _invert_cdf(cdf, target: float, lo: float, hi: float, n: int = 200) -> float:
    """Quantile of a monotone cdf by bisection, bracketed inside a wide range."""
    lo, hi = float(lo), float(hi)
    for _ in range(n):
        mid = 0.5 * (lo + hi)
        if cdf(mid) < target:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


# --------------------------------------------------------------------------- #
# the public entry point
# --------------------------------------------------------------------------- #
@dataclass
class RobmaResult:
    """Model-averaged posterior over publication-bias models.

    Attributes:
        models: one dictionary per model, sorted by posterior probability, with
            ``bias``, ``effect``, ``heterogeneity``, ``log_marginal``,
            ``prior_prob`` and ``post_prob``.
        inclusion: for each component, the prior and posterior probability that
            it is present and the resulting Bayes factor.
        mu: model-averaged posterior of the pooled effect, with ``mean``, ``sd``,
            ``q_low``, ``q_high``, ``point_mass`` (the posterior probability
            that :math:`\\mu = 0`) and ``prob_positive``.
        tau: the same for the between-study standard deviation.
        k: studies used.
        level: credibility level in percent.
        priors: the prior hyperparameters actually integrated.
        grid: the quadrature resolution.
        bias_models: which bias models entered the ensemble.
    """

    models: list[dict]
    inclusion: dict[str, dict]
    mu: dict
    tau: dict
    k: int
    level: float = 95.0
    priors: dict = field(default_factory=dict)
    grid: dict = field(default_factory=dict)
    bias_models: tuple[str, ...] = ()

    # -- reporting --------------------------------------------------------- #
    def summary(self) -> list[dict]:
        """The model table, most probable first."""
        return [dict(m) for m in self.models]

    def to_dict(self) -> dict:
        return {
            "k": self.k,
            "level": self.level,
            "bias_models": list(self.bias_models),
            "priors": {k: float(v) for k, v in self.priors.items()},
            "grid": {k: int(v) for k, v in self.grid.items()},
            "inclusion": {k: dict(v) for k, v in self.inclusion.items()},
            "mu": dict(self.mu),
            "tau": dict(self.tau),
            "models": [dict(m) for m in self.models],
        }

    def __str__(self) -> str:
        top = max(self.models, key=lambda m: m["post_prob"])
        rows = ["RoBMA: model-averaged posterior over publication bias"]
        rows.append(f"  studies                 {self.k}")
        rows.append(f"  models in the ensemble  {len(self.models)}")
        rows.append(f"  pooled effect           {self.mu['mean']:+.4f} "
                    f"[{self.mu['q_low']:+.4f}, {self.mu['q_high']:+.4f}] "
                    f"({self.level:.0f}% credible)")
        rows.append(f"  P(mu = 0 | data)        {self.mu['point_mass']:.3f}")
        rows.append(f"  between-study SD        {self.tau['mean']:.4f} "
                    f"[{self.tau['q_low']:.4f}, {self.tau['q_high']:.4f}]")
        rows.append(f"  P(tau = 0 | data)       {self.tau['point_mass']:.3f}")
        for name in ("Effect", "Heterogeneity", "Publication bias"):
            inc = self.inclusion[name]
            rows.append(f"  {name:<22}  P = {inc['post_prob']:.3f} "
                        f"(prior {inc['prior_prob']:.3f}, "
                        f"BF = {inc['bf']:.3f})")
        rows.append(f"  most probable model     {top['bias']}"
                    f"{'' if top['effect'] else ' / mu = 0'}"
                    f"{'' if top['heterogeneity'] else ' / tau = 0'}"
                    f"  (P = {top['post_prob']:.3f})")
        return "\n".join(rows)


def robma(yi, sei=None, *, bias_models=None, prior=None, level: float = 95.0,
          grid=None) -> RobmaResult:
    """Bayesian model averaging over publication-bias models.

    Enumerates the model space, integrates the marginal likelihood of each
    model by Gauss–Legendre quadrature, and reports the posterior over models
    together with the model-averaged estimate. No sampling: the answer is exact
    up to quadrature error, which is checked in the test suite by refining the
    grid.

    The components are the ones the reference package uses. Each model combines

    * a **bias** model — no bias, PET (:math:`\\mu + \\beta \\sigma_i`) or
      PEESE (:math:`\\mu + \\beta \\sigma_i^2`), with
      :math:`\\beta \\sim \\text{half-Cauchy}(0, c)` and :math:`c = 1` for PET
      and :math:`5` for PEESE;
    * an **effect** indicator — :math:`\\mu = 0` or
      :math:`\\mu \\sim N(0, 1/2)`;
    * a **heterogeneity** indicator — :math:`\\tau = 0` or
      :math:`\\tau \\sim N(0, 1/8)` on :math:`[0, \\infty)`.

    Prior weights over the bias models are :math:`1` for no bias and
    :math:`1/4` for PET and PEESE, normalising to :math:`1/2` and :math:`1/8`;
    the two indicators are uniform, so P(:math:`\\mu = 0`) = 0.5 a priori. These
    are read off `RoBMA` 4.0.0 rather than from its papers, which differ.

    Args:
        yi: observed effects, or a fitted model whose ``.yi``/``.vi`` are used.
        sei: their standard errors.
        bias_models: which bias models to include, by name or by
            :class:`WeightFunction`. Defaults to the three that this module
            verifies, ``("none", "PET", "PEESE")``; the step selection functions
            raise until their kernel has been pinned against the reference.
        prior: overrides for ``mu_sd``, ``tau_sd``, ``pet_scale``,
            ``peese_scale``.
        level: credibility level in percent (``0.95`` and ``95`` both mean 95%).
        grid: quadrature nodes per axis: ``tau`` and ``beta``. The effect is
            integrated analytically and has no grid.

    Returns:
        :class:`RobmaResult`.

    Raises:
        ValueError: fewer than three studies, mismatched lengths, non-positive
            standard errors, or an unknown key in ``prior``/``grid``.
        NotImplementedError: when a step selection function is requested.

    Example:
        >>> fit = ma.robma(yi, sei=sei)
        >>> print(fit)                          # doctest: +SKIP
    """
    from .levels import level_to_alpha
    from .rma_uni import RmaResult

    if isinstance(yi, RmaResult):
        sei_arr = np.sqrt(np.asarray(yi.vi, dtype=float))
        yi_arr = np.asarray(yi.yi, dtype=float)
    else:
        if sei is None:
            raise ValueError("'sei' is required when 'yi' is an array")
        yi_arr = np.asarray(yi, dtype=float).ravel()
        sei_arr = np.asarray(sei, dtype=float).ravel()
    if yi_arr.size != sei_arr.size:
        raise ValueError("'yi' and 'sei' must be the same length")
    if yi_arr.size < 3:
        raise ValueError("at least three studies are needed")
    if np.any(sei_arr <= 0.0):
        raise ValueError("standard errors must be positive")
    if not np.all(np.isfinite(yi_arr)):
        raise ValueError("'yi' must be finite")

    priors = dict(DEFAULTS)
    if prior:
        unknown = set(prior) - set(priors)
        if unknown:
            raise ValueError(
                f"unknown prior(s): {', '.join(sorted(map(str, unknown)))}; "
                f"known keys are {', '.join(sorted(priors))}")
        priors.update({k: float(v) for k, v in prior.items()})
    resolution = dict(DEFAULT_GRID)
    if grid:
        unknown = set(grid) - set(resolution)
        if unknown:
            raise ValueError(
                f"unknown grid key(s): {', '.join(sorted(map(str, unknown)))}; "
                f"known keys are {', '.join(sorted(resolution))}")
        resolution.update({k: int(v) for k, v in grid.items()})
    if any(v < 3 for v in resolution.values()):
        raise ValueError("every grid axis needs at least three nodes")
    alpha = level_to_alpha(level)
    level_pct = 100.0 * (1.0 - alpha)

    specs = _normalise_bias_models(bias_models)
    labels = tuple(_bias_label(s) for s in specs)
    weights = {lab: _bias_weight(lab) for lab in labels}
    total_w = float(sum(weights.values()))

    models: list[dict] = []
    for spec in specs:
        label = _bias_label(spec)
        for effect_free in (True, False):
            for hetero_free in (True, False):
                fit = _fit_model(yi_arr, sei_arr, bias=spec,
                                 effect_free=effect_free,
                                 hetero_free=hetero_free, prior=priors,
                                 grid=resolution)
                models.append({
                    "bias": label,
                    "effect": bool(effect_free),
                    "heterogeneity": bool(hetero_free),
                    "log_marginal": fit.log_marginal,
                    "prior_prob": weights[label] / total_w * 0.25,
                    "_fit": fit,
                })

    log_joint = np.array([np.log(m["prior_prob"]) + m["log_marginal"]
                          for m in models])
    log_ev = float(logsumexp(log_joint))
    for m, value in zip(models, log_joint):
        m["post_prob"] = float(np.exp(value - log_ev))

    # -- model-averaged posteriors ----------------------------------------- #
    mu_means: list[np.ndarray] = []
    mu_sds: list[np.ndarray] = []
    mu_ws: list[np.ndarray] = []
    mu_grid_terms: list[tuple[np.ndarray, np.ndarray]] = []
    mu_spike = 0.0
    tau_terms: list[np.ndarray] = []
    tau_spike = 0.0
    tau_nodes = None
    for entry in models:
        model_fit: _Fit = entry.pop("_fit")
        p = entry["post_prob"]
        if entry["effect"]:
            if p > 0.0:
                if model_fit.mu_mean is not None \
                        and model_fit.mu_sd is not None:
                    mu_means.append(model_fit.mu_mean)
                    mu_sds.append(model_fit.mu_sd)
                    mu_ws.append(np.log(p) + np.asarray(model_fit.mu_logw))
                elif model_fit.mu_grid_logw is not None:
                    assert model_fit.mu_grid_nodes is not None
                    mu_grid_terms.append(
                        (model_fit.mu_grid_nodes,
                         np.log(p) + model_fit.mu_grid_logw))
        else:
            mu_spike += p
        if entry["heterogeneity"]:
            if p > 0.0:
                tau_terms.append(np.log(p) + model_fit.tau_logw
                                 - model_fit.log_marginal)
            tau_nodes = model_fit.tau_nodes
        else:
            tau_spike += p

    mu = _mixture_summary(mu_means, mu_sds, mu_ws, mu_spike, 0.0, 1.0 - alpha,
                          grid_terms=mu_grid_terms)
    tau = _grid_summary(tau_nodes, tau_terms, tau_spike, 0.0, 1.0 - alpha)

    # -- inclusion probabilities ------------------------------------------- #
    def _incl(pred) -> dict:
        post = float(sum(m["post_prob"] for m in models if pred(m)))
        prior_ = float(sum(m["prior_prob"] for m in models if pred(m)))
        return {"prior_prob": prior_, "post_prob": post,
                "bf": post / prior_ if prior_ > 0.0 else float("nan")}

    inclusion = {
        "Effect": _incl(lambda m: m["effect"]),
        "Heterogeneity": _incl(lambda m: m["heterogeneity"]),
        "Publication bias": _incl(lambda m: m["bias"] != "none"),
    }
    for label in labels:
        inclusion[f"bias:{label}"] = _incl(lambda m, n=label: m["bias"] == n)

    models.sort(key=lambda m: -m["post_prob"])
    return RobmaResult(models=models, inclusion=inclusion, mu=mu, tau=tau,
                       k=int(yi_arr.size), level=level_pct, priors=priors,
                       grid=resolution, bias_models=labels)


def _grid_summary(nodes, terms: list[np.ndarray], point_mass: float,
                  point_value: float, level: float) -> dict:
    """Mean, SD and interval of a grid posterior plus a spike at ``0``.

    Used for :math:`\tau`, which is one-dimensional and concentrated near zero;
    its posterior is carried on the quadrature nodes rather than as a mixture.    ``level`` is a fraction — 0.95, not 95.    """
    if nodes is None or not terms:
        return {"mean": float(point_value), "sd": 0.0,
                "q_low": float(point_value), "q_high": float(point_value),
                "point_mass": float(point_mass), "prob_positive": 0.0}
    logw = logsumexp(np.vstack(terms), axis=0)
    mass = np.exp(logw - logw.max())
    total = float(np.sum(mass))
    if total <= 0.0 or not np.isfinite(total):
        raise RuntimeError("the tau posterior has no mass on its own grid")
    mass = mass / total * (1.0 - point_mass)
    xs = np.concatenate([[point_value], nodes])
    ps = np.concatenate([[point_mass], mass])
    order = np.argsort(xs)
    xs, ps = xs[order], ps[order]
    mean = float(np.sum(xs * ps))
    sd = float(np.sqrt(np.sum((xs - mean) ** 2 * ps)))
    cdf = np.cumsum(ps)
    tail = (1.0 - level) / 2.0
    return {"mean": mean, "sd": sd,
            "q_low": float(np.interp(tail, cdf, xs)),
            "q_high": float(np.interp(1.0 - tail, cdf, xs)),
            "point_mass": float(point_mass),
            "prob_positive": float(np.sum(mass[nodes > 0.0]))}


def _bias_label(spec) -> str:
    """Printable name of a bias model, whether it is a keyword or a function."""
    return spec if isinstance(spec, str) else spec.name


def _bias_weight(label: str) -> float:
    """Prior model weight; an unlisted step function gets the reference's 1/12."""
    return BIAS_PRIOR_WEIGHTS.get(label, 1.0 / 12.0)


def _normalise_bias_models(bias_models) -> tuple:
    """Validate the requested bias models and return them in canonical order."""
    if bias_models is None:
        return ("none", "PET", "PEESE")
    requested: list = []
    seen: set[str] = set()
    for item in bias_models:
        label = _bias_label(item)
        if label in seen:
            continue
        if isinstance(item, WeightFunction):
            raise NotImplementedError(
                "the step selection functions are not verified. With the "
                "weights fixed at (1, 0.5) the kernel reproduces the reference's "
                "heterogeneity - tau 0.071 against 0.080 one-sided and 0.073 "
                "against 0.072 two-sided - but the pooled effect is offset by "
                "about 0.11 with an identical standard deviation (mu 0.354 "
                "against 0.236). A constant offset with matching spread points "
                "at how the location enters, not at the kernel, so that is what "
                "is being pinned next. Use 'none', 'PET' or 'PEESE' until it "
                "agrees.")
        if not isinstance(item, str):
            raise ValueError(
                f"a bias model must be a name or a WeightFunction, not "
                f"{type(item).__name__}")
        if item not in BIAS_PRIOR_WEIGHTS:
            raise ValueError(
                f"unknown bias model {item!r}; known are "
                f"{', '.join(sorted(BIAS_PRIOR_WEIGHTS))}")
        seen.add(label)
        requested.append(item)
    if not requested:
        raise ValueError("'bias_models' must name at least one model")
    order = list(BIAS_PRIOR_WEIGHTS)
    requested.sort(key=lambda s: order.index(str(s)))
    return tuple(requested)
