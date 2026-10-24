"""Markov chain Monte Carlo, written against numpy.

Some models in this package have no closed-form marginal likelihood, and no
amount of algebra removes the integral. The clearest case is a random effect on
a parameter a model is *not* linear in: a dose-response curve whose Emax and
ED50 vary between studies,

.. math::

    y_{i} = f_b(x_b; \\beta_b + b_{ib}) - f_a(x_a; \\beta_a + b_{ia}) + e_i,
    \\qquad b_{i} \\sim N(0, D),

has a marginal likelihood that is an integral over :math:`b_i` with no closed
form, so the REML machinery used elsewhere in the package does not apply.
Bayesian inference needs no such marginal: it samples :math:`b_i` alongside
:math:`\\beta` and integrates by averaging.

Why this module is **optional**
------------------------------

Nothing else in `pimeta` imports it, and no function requires it. It is written
against numpy, so it adds no dependency at all: `pip install pimeta` already
provides everything it uses, and `pip install pimeta[bayes]` is declared only so
that an accelerator can be added later without changing anybody's install
command. What it costs is time — an MCMC fit is seconds to minutes where a REML
fit is milliseconds — so it is opt-in per call (``inference="mcmc"``) rather
than a default.

What it is, and what it is not
------------------------------

It is a sampler you give a log-density and a parameterisation: adaptive
random-walk Metropolis with a multivariate normal proposal whose covariance is
learned from warmup draws (Haario, Saksman & Tamminen 2001), one chain at a
time, and the diagnostics that say whether the draws can be believed —
rank-normalised split :math:`\\hat{R}` and bulk effective sample size, both
following Vehtari et al. (2021), because the older versions of those numbers are
quietly too optimistic on exactly the models this module is for.

It is **not** a probabilistic programming language. There is no gradient
information, no NUTS, no automatic parameterisation. In exchange, a model here
cannot fail to compile and cannot silently sample a different model from the one
you meant — the log-density *is* the model. For hierarchical models with many
parameters a gradient sampler is markedly more efficient, and that is the trade
being made deliberately.

The one thing this module will not do is hide a bad chain. A fit that does not
mix returns `rhat` well above 1.01 and says so in `Posterior.summary`;
:meth:`Posterior.check` raises unless the caller asks not to be told.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from scipy import stats

__all__ = [
    "Posterior",
    "ess_bulk",
    "ess_tail",
    "rhat",
    "sample_log_posterior",
]


# --------------------------------------------------------------------------- #
# diagnostics
# --------------------------------------------------------------------------- #
def _as_chains(draws) -> np.ndarray:
    """``(n_chains, n_draws, n_par)`` from an array or a per-chain list."""
    arr = np.asarray(draws, dtype=float)
    if arr.ndim == 1:
        return arr.reshape(1, -1, 1)
    if arr.ndim == 2:
        return arr[:, :, None]
    if arr.ndim != 3:
        raise ValueError("draws must be 1-, 2- or 3-dimensional")
    return arr


def split_chains(draws) -> np.ndarray:
    """Cut every chain in half, so a drifting chain is visible as discord."""
    chains = _as_chains(draws)
    n_chains, n_draws, n_par = chains.shape
    half = n_draws // 2
    if half < 2:
        raise ValueError(
            "a chain needs at least four draws to be split; increase 'n_draws'")
    out = np.empty((2 * n_chains, half, n_par))
    out[0::2] = chains[:, :half, :]
    out[1::2] = chains[:, n_draws - half:, :]
    return out


def _rank_normalise(chains: np.ndarray) -> np.ndarray:
    """Map each parameter's pooled draws to normal scores.

    Ranking first is what makes the diagnostic usable on a skewed posterior — a
    tail explored in a few big jumps produces a large ``R-hat`` under the raw
    version even when the chains agree, and that is a false alarm the rank
    version does not raise.
    """
    n_chains, n_draws, n_par = chains.shape
    total = n_chains * n_draws
    flat = chains.reshape(total, n_par)
    out = np.empty_like(flat)
    for j in range(n_par):
        order = stats.rankdata(flat[:, j], method="average")
        out[:, j] = stats.norm.ppf((order - 0.375) / (total + 0.25))
    return out.reshape(n_chains, n_draws, n_par)


def _rhat_chains(chains: np.ndarray) -> np.ndarray:
    """The variance-ratio :math:`\\hat{R}`, per parameter."""
    n_chains, n_draws, n_par = chains.shape
    out = np.empty(n_par)
    for j in range(n_par):
        x = chains[:, :, j]
        between = n_draws * float(np.var(np.mean(x, axis=1), ddof=1))
        within = float(np.mean(np.var(x, axis=1, ddof=1)))
        if within <= 0.0:
            out[j] = np.inf if between > 0.0 else 1.0
            continue
        var_plus = (n_draws - 1.0) / n_draws * within + between / n_draws
        out[j] = float(np.sqrt(var_plus / within))
    return out


def _autocovariance(x: np.ndarray) -> np.ndarray:
    """Autocovariance of one chain by FFT, for the ESS estimate."""
    n = x.size
    centred = x - x.mean()
    size = 1
    while size < 2 * n:
        size *= 2
    spectrum = np.fft.rfft(centred, size)
    acov = np.fft.irfft(spectrum * np.conjugate(spectrum), size)[:n]
    return acov / n


def _ess_single(x: np.ndarray) -> float:
    """Bulk ESS from the initial positive sequence, Geyer (1992)."""
    n = x.size
    if n < 4 or np.allclose(x, x[0]):
        return 1.0
    acov = _autocovariance(x)
    if acov[0] <= 0.0:
        return 1.0
    rho = acov / acov[0]
    total = 0.0
    for k in range(1, n - 1, 2):
        pair = rho[k] + (rho[k + 1] if k + 1 < n else 0.0)
        if pair < 0.0:
            break
        total += pair
    tau = -1.0 + 2.0 * (1.0 + total)
    tau = max(tau, 1e-10)
    return float(n / tau)


def ess_bulk(draws) -> np.ndarray:
    """Bulk effective sample size per parameter, over all chains pooled.

    Computed on the rank-normalised draws, so a heavy tail does not flatter it.
    """
    chains = split_chains(draws)
    normal = _rank_normalise(chains)
    n_chains, n_draws, n_par = normal.shape
    out = np.empty(n_par)
    for j in range(n_par):
        out[j] = float(np.sum([_ess_single(normal[c, :, j])
                               for c in range(n_chains)]))
    return out


def ess_tail(draws) -> np.ndarray:
    """Effective sample size of the 5% and 95% quantiles, whichever is worse."""
    chains = split_chains(draws)
    n_chains, n_draws, n_par = chains.shape
    out = np.empty(n_par)
    for j in range(n_par):
        x = chains[:, :, j]
        lo = np.quantile(x, 0.05)
        hi = np.quantile(x, 0.95)
        low_ind = (x <= lo).astype(float)
        high_ind = (x >= hi).astype(float)
        out[j] = min(_ess_single(low_ind.reshape(-1)),
                     _ess_single(high_ind.reshape(-1)))
    return out


def rhat(draws) -> np.ndarray:
    """Rank-normalised split :math:`\\hat{R}`, per parameter.

    Values near 1.00 mean the chains are indistinguishable; above 1.01 they are
    not, and the draws should not be summarised. Both the location and the
    spread version are computed and the larger is reported, because a chain can
    agree on the centre and disagree in the tail.
    """
    chains = split_chains(draws)
    normal = _rank_normalise(chains)
    bulk = _rhat_chains(normal)
    folded = _rhat_chains(np.abs(normal - np.median(normal, axis=(0, 1))))
    return np.maximum(bulk, folded)


# --------------------------------------------------------------------------- #
# the sampler
# --------------------------------------------------------------------------- #
@dataclass
class Posterior:
    """Draws from a posterior, with the diagnostics that judge them.

    Attributes:
        chains: ``(n_chains, n_draws, n_par)`` draws after warmup.
        names: one name per parameter.
        logpost: ``(n_chains, n_draws)`` log-posterior at each draw.
        accept: acceptance rate per chain from the sampling phase.
        settings: what was run, for the record.
    """

    chains: np.ndarray = field(repr=False)
    names: list[str]
    logpost: np.ndarray = field(repr=False)
    accept: np.ndarray
    settings: dict = field(default_factory=dict)

    # -- shape ------------------------------------------------------------- #
    @property
    def n_chains(self) -> int:
        return int(self.chains.shape[0])

    @property
    def n_draws(self) -> int:
        return int(self.chains.shape[1])

    @property
    def n_par(self) -> int:
        return int(self.chains.shape[2])

    @property
    def draws(self) -> np.ndarray:
        """All draws stacked over chains, ``(n_chains * n_draws, n_par)``."""
        return self.chains.reshape(-1, self.n_par)

    def __getitem__(self, name: str) -> np.ndarray:
        return self.draws[:, self.names.index(name)]

    # -- diagnostics ------------------------------------------------------- #
    def rhat(self) -> np.ndarray:
        return rhat(self.chains)

    def ess(self) -> np.ndarray:
        return ess_bulk(self.chains)

    def ess_tail(self) -> np.ndarray:
        return ess_tail(self.chains)

    def check(self, *, rhat_max: float = 1.01, ess_min: float = 100.0,
              warn_only: bool = False) -> dict:
        """Judge the draws, raising unless ``warn_only``.

        The thresholds are the ones the Stan and Vehtari et al. literature
        settled on, not house numbers: ``R-hat`` below 1.01 and a bulk ESS of at
        least 100 per parameter are roughly where a posterior mean is stable to
        the precision anyone reports.
        """
        rh = self.rhat()
        es = self.ess()
        problems = []
        for name, r, e in zip(self.names, rh, es):
            if not np.isfinite(r) or r > rhat_max:
                problems.append(
                    f"{name}: R-hat is undefined (a chain never moved)"
                    if not np.isfinite(r) else
                    f"{name}: R-hat = {r:.3f} > {rhat_max} (the chains disagree)")
            if e < ess_min:
                problems.append(
                    f"{name}: effective sample size {e:.0f} < {ess_min:.0f}")
        report: dict[str, Any] = {
            "ok": not problems,
            "problems": problems,
            "rhat_max": float(np.max(rh)) if rh.size else float("nan"),
            "ess_min": float(np.min(es)) if es.size else float("nan"),
        }
        if problems and not warn_only:
            raise RuntimeError(
                "the sampler did not mix well enough to summarise these draws:\n"
                + "\n".join("  " + p for p in problems)
                + "\nIncrease 'n_draws' or 'warmup', reparameterise, or pass "
                  "warn_only=True to inspect the draws anyway.")
        return report

    def summary(self, level: float = 0.95) -> list[dict]:
        """Mean, SD, Monte Carlo error, interval, ESS and R-hat per parameter."""
        tail = (1.0 - level) / 2.0
        draws = self.draws
        rh = self.rhat()
        es = self.ess()
        rows = []
        for j, name in enumerate(self.names):
            x = draws[:, j]
            sd = float(np.std(x, ddof=1))
            rows.append({
                "term": name,
                "mean": float(np.mean(x)),
                "sd": sd,
                "mcse": sd / np.sqrt(max(es[j], 1.0)),
                "ess": float(es[j]),
                "rhat": float(rh[j]),
                "q_low": float(np.quantile(x, tail)),
                "median": float(np.median(x)),
                "q_high": float(np.quantile(x, 1.0 - tail)),
            })
        return rows

    def ci(self, name: str | None = None, level: float = 0.95):
        """Equal-tailed credible interval for one parameter, or for all."""
        tail = (1.0 - level) / 2.0
        if name is None:
            return [(float(np.quantile(self.draws[:, j], tail)),
                     float(np.quantile(self.draws[:, j], 1.0 - tail)))
                    for j in range(self.n_par)]
        x = self[name]
        return (float(np.quantile(x, tail)), float(np.quantile(x, 1.0 - tail)))

    def to_dict(self) -> dict:
        return {
            "names": list(self.names),
            "n_chains": self.n_chains,
            "n_draws": self.n_draws,
            "accept": [float(v) for v in self.accept],
            "settings": {
                k: (v if isinstance(v, (int, float, str, bool, type(None)))
                    else str(v))
                for k, v in self.settings.items()
            },
            "summary": self.summary(),
        }

    def draws_json(self) -> str:
        """The draws themselves, for a record that has to be reproducible."""
        return json.dumps({"names": list(self.names),
                           "chains": np.asarray(self.chains).tolist()})

    def __str__(self) -> str:
        rows = self.summary()
        width = max([len(r["term"]) for r in rows] + [8])
        body = [f"{'':<{width}}{'mean':>10}{'sd':>10}{'mcse':>10}"
                f"{'2.5%':>10}{'97.5%':>10}{'ess':>8}{'Rhat':>7}"]
        for r in rows:
            body.append(f"{r['term']:<{width}}{r['mean']:>10.4f}"
                        f"{r['sd']:>10.4f}{r['mcse']:>10.4f}"
                        f"{r['q_low']:>10.4f}{r['q_high']:>10.4f}"
                        f"{r['ess']:>8.0f}{r['rhat']:>7.3f}")
        body.append(f"  {self.n_chains} chains x {self.n_draws} draws, "
                    f"acceptance {np.mean(self.accept):.2f}")
        return "\n".join(body)


# --------------------------------------------------------------------------- #
# the sampler, continued
# --------------------------------------------------------------------------- #
def _proposal_covariance(draws: np.ndarray, fallback: np.ndarray) -> np.ndarray:
    """Proposal covariance from warmup draws, regularised to stay usable."""
    if draws.shape[0] <= draws.shape[1] + 1:
        return fallback
    cov = np.cov(draws, rowvar=False)
    if cov.ndim == 0:
        cov = np.array([[float(cov)]])
    if not np.all(np.isfinite(cov)) or np.trace(cov) <= 0.0:
        return fallback
    # a proposal built from too few draws is nearly singular; shrink it towards
    # its diagonal until Cholesky succeeds rather than crashing mid-run
    for shrink in (0.0, 0.05, 0.1, 0.25, 0.5):
        candidate = (1.0 - shrink) * cov + shrink * np.diag(np.diag(cov))
        try:
            np.linalg.cholesky(candidate + 1e-12 * np.eye(cov.shape[0]))
            return candidate
        except np.linalg.LinAlgError:
            continue
    return fallback


def _starting_points(init, chains: int) -> np.ndarray:
    if callable(init):
        rng = np.random.default_rng(0)
        return np.vstack([np.asarray(init(rng), dtype=float).ravel()
                          for _ in range(chains)])
    arr = np.asarray(init, dtype=float)
    if arr.ndim == 1:
        return np.vstack([arr] * chains)
    if arr.ndim == 2:
        if arr.shape[0] == chains:
            return arr
        if arr.shape[1] == chains:
            return arr.T
        raise ValueError(
            f"'init' is {arr.shape}; expected one row per chain (or a single "
            "vector to use for all of them)")
    raise ValueError("'init' must be a vector, a matrix or a callable")


def _evaluate(log_target, theta: np.ndarray, *, where: str | None) -> float:
    value = log_target(theta)
    if isinstance(value, (int, float, np.floating, np.integer)):
        return float(value)
    if isinstance(value, np.ndarray) and value.size == 1:
        return float(np.asarray(value).reshape(-1)[0])
    raise TypeError(
        "the log density must return a single number, not "
        f"{type(value).__name__}" + (f" (at {where})" if where else ""))


def _try_log_target(log_target, theta: np.ndarray) -> float:
    """The proposal's log density, with a singular region treated as rejected."""
    try:
        return _evaluate(log_target, theta, where=None)
    except np.linalg.LinAlgError:
        return float("-inf")
    except (ValueError, ZeroDivisionError, FloatingPointError):
        return float("-inf")


def sample_log_posterior(log_target: Callable[[np.ndarray], float],
                         init, *, names=None, n_draws: int = 4000,
                         warmup: int = 2000, chains: int = 4,
                         target_accept: float = 0.234,
                         init_sd: float | np.ndarray = 0.1,
                         seed=None, adapt: bool = True,
                         blocks=None, max_attempts: int = 1) -> Posterior:
    """Sample a log-density with adaptive random-walk Metropolis.

    One chain at a time, each with its own adaptive proposal. During warmup the
    proposal covariance is re-estimated from the states visited so far and scaled
    by the Haario factor ``2.38**2 / n_par``, with a bounded correction from the
    acceptance rate; after warmup everything is frozen, so the draws that are
    kept come from a genuine Markov chain with no adaptation left in it.

    Args:
        log_target: ``f(theta) -> float``, the log posterior up to an additive
            constant. It may return ``-inf`` (a rejected region) but must not
            raise for finite inputs.
        init: starting values, shape ``(n_par,)``; one row per chain; or a
            callable ``(rng) -> array``.
        names: parameter names, for the summary.
        n_draws: draws kept per chain, after warmup.
        warmup: adaptation draws per chain, discarded.
        chains: number of chains. Fewer than two cannot produce ``R-hat``.
        target_accept: the acceptance rate the correction aims for. 0.234 is the
            random-walk optimum for a Gaussian target in high dimensions.
        init_sd: initial proposal scale — a scalar, or one per parameter.
        seed: seeds the generator, so a fit is reproducible.
        adapt: ``False`` freezes the proposal at ``init_sd`` from the start,
            which is useful for showing that adaptation is what does the work.
        blocks: an optional update schedule: a list of index arrays, disjoint
            and covering every parameter, each updated in its own Metropolis
            step. A hierarchical model mixes far better this way, because a
            global random walk in a hundred dimensions crosses a narrow ridge
            that a per-group walk along it does not. ``None`` updates all
            parameters in one block.
        max_attempts: retry the whole run with a quarter of the proposal scale
            if a chain cannot move at all. ``1`` disables the retry.

    Returns:
        :class:`Posterior`.

    Raises:
        ValueError: for a bad ``init``, fewer than four draws, fewer than two
            chains, or a starting point with non-finite log density.
        TypeError: when ``log_target`` does not return a number.

    Example:
        >>> draws = sample_log_posterior(
        ...     lambda t: -0.5 * float(t @ t), [0.0, 0.0],
        ...     names=["a", "b"], n_draws=1000, warmup=500, chains=2, seed=1)
        >>> print(draws)                       # doctest: +SKIP
    """
    if n_draws < 4:
        raise ValueError("'n_draws' must be at least 4 to compute diagnostics")
    if chains < 2:
        raise ValueError(
            "'chains' must be at least 2: one chain cannot say anything about "
            "whether it has converged")
    start = _starting_points(init, chains)
    n_par = start.shape[1]
    if names is None:
        labels = [f"theta[{j + 1}]" for j in range(n_par)]
    else:
        labels = [str(v) for v in names]
        if len(labels) != n_par:
            raise ValueError(
                f"'names' has {len(labels)} entries but 'init' has {n_par} "
                "parameter(s)")
    scale = np.broadcast_to(np.asarray(init_sd, dtype=float),
                            (n_par,)).astype(float).copy()
    if np.any(scale <= 0.0):
        raise ValueError("'init_sd' must be positive")
    block_list = _as_blocks(blocks, n_par)

    root = seed if isinstance(seed, np.random.Generator) else \
        np.random.default_rng(seed)
    children = (root.spawn(chains) if hasattr(root, "spawn")
                else [np.random.default_rng(int(root.integers(2 ** 63)))
                      for _ in range(chains)])
    for c in range(chains):
        first = _evaluate(log_target, start[c], where=f"chain {c + 1} start")
        if not np.isfinite(first):
            raise ValueError(
                f"chain {c + 1} starts where the log density is {first}: a "
                "starting point has to be inside the support of the posterior")

    best: Posterior | None = None
    attempt = 0
    while True:
        attempt += 1
        out = _run_chains(log_target, start, children, scale, block_list,
                          n_draws=n_draws, warmup=warmup,
                          target_accept=target_accept, adapt=adapt)
        if best is None or float(np.min(out.accept)) > float(np.min(best.accept)):
            best = out
        if float(np.min(out.accept)) > 0.0 or attempt >= max_attempts:
            break
        scale = scale * 0.25

    assert best is not None
    best.names = labels
    best.settings = {
        "n_draws": n_draws,
        "warmup": warmup,
        "chains": chains,
        "n_blocks": len(block_list),
        "target_accept": float(target_accept),
        "adapt": bool(adapt),
        "inference": "adaptive random-walk Metropolis (numpy only)",
    }
    return best


def _run_chains(log_target, start, rngs, scale, block_list, *, n_draws, warmup,
                target_accept, adapt) -> Posterior:
    n_chains, n_par = start.shape
    draws = np.empty((n_chains, n_draws, n_par))
    logpost = np.empty((n_chains, n_draws))
    accept = np.empty(n_chains)
    window = 64
    for c in range(n_chains):
        rng = rngs[c]
        theta = start[c].copy()
        lp = _evaluate(log_target, theta, where=f"chain {c + 1} start")
        covs = [np.diag(scale[idx] ** 2) for idx in block_list]
        chols = [np.linalg.cholesky(cov) for cov in covs]
        multipliers = [1.0] * len(block_list)
        windows = [np.zeros(window) for _ in block_list]
        n_accept = np.zeros(len(block_list))
        visited = [np.empty((max(warmup, 1), idx.size)) for idx in block_list]
        for it in range(warmup + n_draws):
            in_warmup = it < warmup
            for b, idx in enumerate(block_list):
                if in_warmup:
                    visited[b][it] = theta[idx]
                step = (multipliers[b] * np.sqrt(2.38 ** 2 / idx.size)
                        * (chols[b] @ rng.standard_normal(idx.size)))
                previous = theta[idx].copy()
                theta[idx] += step
                lp_new = _try_log_target(log_target, theta)
                moved = (np.isfinite(lp_new)
                         and np.log(rng.random()) < lp_new - lp)
                if moved:
                    lp = lp_new
                    if not in_warmup:
                        n_accept[b] += 1
                else:
                    theta[idx] = previous
                if in_warmup and adapt:
                    # The proposal is the Haario rule -- 2.38^2/d times the
                    # covariance of the states this block has visited -- with a
                    # bounded correction from the acceptance rate of the last 64
                    # sweeps.  The bound is what keeps the tuning honest: a
                    # cumulative rate lags so far behind a proposal that has
                    # become too wide that an unbounded controller runs away,
                    # and the block then rejects everything for the rest of the
                    # run.
                    windows[b][it % window] = 1.0 if moved else 0.0
                    if it >= window // 2:
                        rate = float(np.mean(windows[b]))
                        gamma = 1.0 / (it + 1) ** 0.6
                        multipliers[b] = float(np.clip(
                            multipliers[b]
                            * np.exp(gamma * (rate - target_accept)), 0.25, 4.0))
                    if it >= 50 and it % 25 == 24:
                        covs[b] = _proposal_covariance(visited[b][:it + 1],
                                                       covs[b])
                        chols[b] = np.linalg.cholesky(
                            covs[b] + 1e-12 * np.eye(idx.size))
            if not in_warmup:
                k = it - warmup
                draws[c, k] = theta
                logpost[c, k] = lp
        accept[c] = float(np.mean(n_accept / max(n_draws, 1)))
    return Posterior(chains=draws, names=[], logpost=logpost, accept=accept)


def _as_blocks(blocks, n_par: int) -> list[np.ndarray]:
    """Validate an update schedule: disjoint index blocks covering everything."""
    if blocks is None:
        return [np.arange(n_par)]
    out = []
    seen = np.zeros(n_par, dtype=bool)
    for block in blocks:
        idx = np.asarray(block, dtype=int).ravel()
        if idx.size == 0:
            raise ValueError("a block cannot be empty")
        if np.any(idx < 0) or np.any(idx >= n_par):
            raise ValueError(
                f"a block indexes outside 0..{n_par - 1}: {idx.tolist()}")
        if np.any(seen[idx]):
            raise ValueError(
                "blocks must be disjoint: an index appears in more than one")
        seen[idx] = True
        out.append(idx)
    if not np.all(seen):
        missing = np.flatnonzero(~seen).tolist()
        raise ValueError(
            f"the blocks leave parameter(s) {missing} out, and a parameter "
            "that is never updated stays at its starting value")
    return out
