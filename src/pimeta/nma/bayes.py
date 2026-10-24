"""A Bayesian network meta-analysis: the same network, sampled not fitted.

The frequentist networking code in this package answers "how precise is
:math:`d_b - d_a`" with a point estimate and a standard error.  A Bayesian
answer is a distribution, and the reason to want one is not decoration: a
ranking probability, a probability that a contrast exceeds a clinically
important difference, or a prediction for a study that has not been done are
all functionals of the posterior that no single standard error carries.

The model
---------
Arm-level data :math:`(y_{jk}, \\sigma_{jk})` for arm :math:`k` of study
:math:`j` on treatment :math:`t_{jk}`, anchored on the network reference
treatment :math:`A`:

.. math::

    y_{jk} \\sim N(\\mu_j + \\theta_{jk},\\ \\sigma_{jk}^2), \\qquad
    \\theta_{jk} = (d_{t_{jk}} - d_A) + \\delta_{jk},\\quad
    \\delta_{jk} \\sim N(0, \\tau^2),\\quad \\delta_{j,b_j} = 0,

where :math:`b_j` is the arm of study :math:`j` that used the reference
treatment.  This is the aggregate-data model of Phillippo et al. (2020) and it
is the one ``multinma::nma()`` fits, which is what makes that package a
reference for it rather than merely another opinion.  Three consequences are
worth stating because they are easy to get wrong:

- the random effects sit on **arms**, not on contrasts, so a multi-arm study's
  two comparisons against its baseline get independent deviations.  A
  contrast-level model would attach one deviation per comparison, and the two
  agree only for two-arm studies;
- a study's baseline is the arm that used the **reference treatment**, not its
  first-listed arm, so row order is not part of the model.  That was checked
  against ``multinma`` rather than assumed: reordering a three-arm study's rows
  leaves its ``delta`` parameters and posterior untouched;
- a study that contains no reference arm has no arm pinned at zero and gets a
  deviation on every arm — with :math:`\\mu_j` free, the mean structure is the
  same as pinning one arm but the variance structure is not.  ``multinma`` does
  this, and the fixture contains two reference-free studies so that getting it
  wrong shows up as a disagreement instead of hiding.

Study baselines :math:`\\mu_j` are **integrated out analytically**, not
sampled.  They are a Gaussian block with a Gaussian prior, so the integral is
closed form (:math:`V_j = \\Psi_j + s_\\mu^2 \\mathbf{1}\\mathbf{1}'`,
:math:`\\Psi_j = \\operatorname{diag}(\\sigma_{jk}^2 + \\tau^2)` off the
baseline), and marginalising is exactly equivalent to sampling them while
removing one parameter per study from the chain.  With thirty studies that is
the difference between a six-dimensional posterior and a thirty-six-dimensional
one, and a random walk in the second is not a sampler anyone should trust.

Priors, stated rather than buried
---------------------------------
- :math:`d \\sim N(0, 10^2)` — weak on any log-effect scale.
- :math:`\\mu_j \\sim N(0, 100^2)` — as close to "uninformative" as a proper
  prior gets, and it is ``multinma``'s default, which lets the reference
  comparison hold the priors fixed.
- :math:`\\tau \\sim \\text{half-N}(0, 0.5^2)` — a between-study SD of 1 on a
  log scale is already enormous, so this is weakly informative by construction.
  It is deliberately narrower than ``multinma``'s ``half_normal(5)`` default;
  the reference script sets both packages to the same value, so the agreement
  measured there is agreement about a model, not about a default.

What is checked, and in which order
-----------------------------------
1. **Algebra.**  The marginal likelihood after integrating :math:`\\mu_j` has
   to equal a direct numerical integration over :math:`\\mu_j`; with
   :math:`\\tau = 0` the posterior mode of :math:`d` has to equal the
   fixed-effect network estimate, which :mod:`pimeta.netmeta` already matches to
   ``netmeta``; a single two-arm study has to reproduce the inverse-variance
   pooled estimate; and the answer must not depend on which treatment is chosen
   as the network reference.
2. **Model.**  ``multinma::nma()`` on the same fixture with the same priors,
   compared per parameter against the combined Monte Carlo error.
3. **Property.**  Data simulated from a known :math:`(d, \\tau)`: bias within
   three Monte Carlo standard errors and interval coverage near nominal.

A second sampler agreeing with the first does not by itself validate the model:
two implementations of the same wrong likelihood agree just as well.  Layer 1
is what makes the likelihood trustworthy, and it comes first for that reason.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from ..bayes import Posterior, sample_log_posterior
from ..levels import level_to_alpha

__all__ = ["BayesNmaResult", "bayes_nma"]

_DEFAULTS: dict[str, Any] = {
    "n_draws": 4000,
    "warmup": 2000,
    "chains": 4,
    "seed": None,
    "check": True,
    "rhat_max": 1.05,
    "ess_min": 100.0,
}


# --------------------------------------------------------------------------- #
# data
# --------------------------------------------------------------------------- #
@dataclass
class _ArmData:
    """Arm-level data laid out the way the likelihood wants it.

    Rows are grouped by study so the sufficient statistics of each study are
    contiguous slices ``[starts[j], starts[j+1])`` and ``np.add.reduceat`` can
    do the whole likelihood in one pass.  A Python loop over studies inside a
    Metropolis loop is the difference between a fixture that runs in two
    seconds and one that runs in two minutes.
    """

    y: np.ndarray            # (m,) arm means
    v: np.ndarray            # (m,) arm sampling variances
    X: np.ndarray            # (m, n_basic): e_{t_jk}, the reference column dropped
    starts: np.ndarray       # (n_studies,) first row of each study
    is_baseline: np.ndarray  # (m,) the arm on the reference treatment, if any
    treatments: list[str]
    basic: list[str]
    reference: str
    studies: list[str]
    n_basic: int = field(init=False)

    def __post_init__(self):
        self.n_basic = len(self.basic)


def _prepare(yi, sei, vi, study, treat, reference) -> _ArmData:
    """Validate the arms and lay them out in study blocks."""
    y = np.asarray(yi, dtype=float).ravel()
    m = y.size
    if m == 0:
        raise ValueError("'yi' is empty")
    if (sei is None) == (vi is None):
        raise ValueError("pass exactly one of 'sei' or 'vi'")
    v = (np.asarray(sei, dtype=float).ravel() ** 2 if sei is not None
         else np.asarray(vi, dtype=float).ravel())
    stud = [str(s) for s in np.asarray(study).ravel()]
    trt = [str(t) for t in np.asarray(treat).ravel()]
    for name, arr in (("sei/vi", v), ("study", stud), ("treat", trt)):
        if len(arr) != m:
            raise ValueError(
                f"'{name}' has {len(arr)} entries but 'yi' has {m}")
    if not np.all(np.isfinite(y)):
        raise ValueError("'yi' contains a non-finite value")
    if not np.all(np.isfinite(v)) or np.any(v <= 0.0):
        raise ValueError("every arm needs a finite, strictly positive variance")

    # a study is one contiguous block; its arms keep their given order
    order = list(range(m))
    studies: list[str] = []
    for s in stud:
        if s not in studies:
            studies.append(s)
    order.sort(key=lambda i: (studies.index(stud[i]), i))

    seen: set[tuple[str, str]] = set()
    for i in order:
        key = (stud[i], trt[i])
        if key in seen:
            raise ValueError(
                f"study {stud[i]!r} lists treatment {trt[i]!r} twice; each "
                "arm must appear once")
        seen.add(key)

    starts = np.zeros(len(studies), dtype=int)
    rank = {i: r for r, i in enumerate(order)}
    for j, s in enumerate(studies):
        starts[j] = min(rank[i] for i in order if stud[i] == s)
    sizes = np.diff(np.append(starts, m))
    if np.any(sizes < 2):
        single = [s for s, k in zip(studies, sizes) if k < 2]
        raise ValueError(
            "every study needs at least two arms; "
            f"{', '.join(map(repr, single))} has one")

    treatments = list(dict.fromkeys(trt))
    ref = treatments[0] if reference is None else str(reference)
    if ref not in treatments:
        raise ValueError(
            f"unknown reference treatment {reference!r}; known: {treatments}")
    basic = [t for t in treatments if t != ref]
    col = {t: j for j, t in enumerate(basic)}

    y_sorted = y[order]
    v_sorted = v[order]
    trt_sorted = [trt[i] for i in order]
    # One arm per study is pinned at zero and carries no deviation: the arm on
    # the reference treatment if the study has one, otherwise the study's
    # first arm.  The second case is not free choice -- a study without the
    # reference treatment would otherwise get a deviation on every arm, which
    # is a different model.  On a three-study sub-network containing one such
    # study the two choices give posterior means of d_B of 0.437 and 0.365, and
    # ``multinma`` gives 0.353, so it pins an arm in every study.
    is_baseline = np.zeros(m, dtype=bool)
    base_of = [""] * len(studies)
    for j, s in enumerate(starts):
        end = starts[j + 1] if j + 1 < len(starts) else m
        rows_j = list(range(s, end))
        pick = next((i for i in rows_j if trt_sorted[i] == ref), rows_j[0])
        is_baseline[pick] = True
        base_of[j] = trt_sorted[pick]
    # theta[jk] = d(t_jk) - d(b_j), so the baseline arm is identically zero and
    # mu_j is the baseline arm's own expected value -- the same quantity
    # multinma calls mu[study]
    X = np.zeros((m, len(basic)))
    for j, s in enumerate(starts):
        end = starts[j + 1] if j + 1 < len(starts) else m
        base = base_of[j]
        for i in range(s, end):
            if is_baseline[i]:
                continue
            if trt_sorted[i] in col:
                X[i, col[trt_sorted[i]]] += 1.0
            if base in col:
                X[i, col[base]] -= 1.0
    return _ArmData(y=y_sorted, v=v_sorted, X=X, starts=starts,
                    is_baseline=is_baseline, treatments=treatments,
                    basic=basic, reference=ref, studies=studies)


def _marginal_loglik(d: np.ndarray, tau2: float, data: _ArmData,
                     prior_intercept_sd: float) -> float:
    """Log likelihood of the arms with each study's baseline integrated out.

    With :math:`\\mu_j \\sim N(0, s^2)` and :math:`\\Psi_j` the arm-level
    covariance before the study mean is added, :math:`V_j = \\Psi_j + s^2
    \\mathbf{1}\\mathbf{1}'`, and the Woodbury identity gives both the
    determinant and the quadratic form in terms of three weighted sums that
    ``np.add.reduceat`` produces for every study at once.  The quadratic form is
    written as :math:`q - b^2/a + \\frac{b^2/a}{1 + s^2 a}` rather than
    :math:`q - \\frac{s^2b^2}{1+s^2a}` because the second form subtracts two
    large numbers and the first does not.
    """
    resid = data.y - data.X @ d
    v = data.v + tau2 * (~data.is_baseline)
    w = 1.0 / v
    a = np.add.reduceat(w, data.starts)
    b = np.add.reduceat(w * resid, data.starts)
    q = np.add.reduceat(w * resid * resid, data.starts)
    logdet = np.add.reduceat(np.log(v), data.starts)

    s2 = float(prior_intercept_sd) ** 2
    b2a = b * b / a
    quad = (q - b2a) + b2a / (1.0 + s2 * a)
    return -0.5 * float(np.sum(logdet + np.log1p(s2 * a) + quad))


def _starting_values(data: _ArmData, tau_start: float) -> np.ndarray:
    """A weighted least-squares start, so warmup is not spent finding the mode.

    Each study is centred on its first arm.  Any arm would do: the study mean
    is a free parameter that the centring cancels against, so this is a
    starting point and not a modelling choice.
    """
    sizes = np.diff(np.append(data.starts, data.y.size))
    anchor_y = np.repeat(data.y[data.starts], sizes)
    anchor_v = np.repeat(data.v[data.starts], sizes)
    anchor_X = np.repeat(data.X[data.starts], sizes, axis=0)
    keep = np.ones(data.y.size, dtype=bool)
    keep[data.starts] = False
    yc = (data.y - anchor_y)[keep]
    vc = (data.v + anchor_v)[keep]
    Xc = (data.X - anchor_X)[keep]
    w = 1.0 / vc
    gram = Xc.T @ (w[:, None] * Xc) + 1e-8 * np.eye(data.n_basic)
    rhs = Xc.T @ (w * yc)
    try:
        d0 = np.linalg.solve(gram, rhs)
    except np.linalg.LinAlgError:      # pragma: no cover - ridge makes it solid
        d0 = np.zeros(data.n_basic)
    return np.append(d0, float(np.log(tau_start)))


# --------------------------------------------------------------------------- #
# results
# --------------------------------------------------------------------------- #
@dataclass
class BayesNmaResult:
    """A sampled network meta-analysis.

    Attributes:
        posterior: the draws and their diagnostics.
        treatments: treatment labels, in parameter order.
        reference: the treatment whose effect is fixed at zero.
        basic: the treatments that carry a parameter (all but ``reference``).
        trt_effects: ``"random"`` or ``"fixed"``.
        tau: posterior mean of the between-study SD, or ``None`` when fixed.
        tau2: posterior mean of :math:`\\tau^2`, or ``None`` when fixed.
        tau_ci: equal-tailed interval for :math:`\\tau`.
        k: number of studies.
        n_arms: number of arms.
        level: the credible level the intervals use, as a fraction.
        priors: the priors the model was fitted with, for the record.
    """

    posterior: Posterior
    treatments: list[str]
    reference: str
    basic: list[str]
    trt_effects: str
    tau: float | None
    tau2: float | None
    tau_ci: tuple[float, float] | None
    k: int
    n_arms: int
    level: float
    priors: dict = field(default_factory=dict)
    _data: Any = field(default=None, repr=False)

    # -- draws ------------------------------------------------------------- #
    def _d_draws(self) -> np.ndarray:
        """``(n_draws, n_treatments)`` draws of every treatment effect."""
        draws = self.posterior.draws
        out = np.zeros((draws.shape[0], len(self.treatments)))
        for j, t in enumerate(self.basic):
            out[:, self.treatments.index(t)] = draws[:, j]
        return out

    # -- the quantities anyone actually asks for --------------------------- #
    def contrast(self, treat1: str, treat2: str, level=None) -> dict:
        """Posterior of ``d(treat1) - d(treat2)``, with its probability.

        The sign convention is the one ``pimeta`` uses everywhere: a positive
        value means ``treat1`` has the larger effect estimate than ``treat2``.
        """
        for t in (treat1, treat2):
            if t not in self.treatments:
                raise ValueError(f"unknown treatment {t!r}")
        alpha = level_to_alpha(self.level if level is None else level)
        d = self._d_draws()
        x = d[:, self.treatments.index(str(treat1))] \
            - d[:, self.treatments.index(str(treat2))]
        lo, hi = np.quantile(x, [alpha / 2.0, 1.0 - alpha / 2.0])
        ess = float(self.posterior.ess()[0])
        return {
            "treat1": str(treat1),
            "treat2": str(treat2),
            "mean": float(np.mean(x)),
            "sd": float(np.std(x, ddof=1)),
            "mcse": float(np.std(x, ddof=1) / np.sqrt(max(ess, 1.0))),
            "q_low": float(lo),
            "q_high": float(hi),
            "prob_positive": float(np.mean(x > 0.0)),
            "level": float(1.0 - alpha),
        }

    def league_table(self, level=None, digits: int = 3) -> dict:
        """Every pairwise contrast: posterior mean, interval and probability.

        Returns a dict of ``treatments`` plus nested ``mean``, ``q_low``,
        ``q_high`` and ``prob_positive`` matrices, row = ``treat1``, column =
        ``treat2``, with ``nan`` on the diagonal.
        """
        n = len(self.treatments)
        mean = np.full((n, n), np.nan)
        lo = np.full((n, n), np.nan)
        hi = np.full((n, n), np.nan)
        prob = np.full((n, n), np.nan)
        d = self._d_draws()
        alpha = level_to_alpha(self.level if level is None else level)
        tails = [alpha / 2.0, 1.0 - alpha / 2.0]
        for i in range(n):
            for j in range(n):
                if i == j:
                    continue
                x = d[:, i] - d[:, j]
                mean[i, j] = float(np.mean(x))
                lo[i, j], hi[i, j] = (float(v) for v in np.quantile(x, tails))
                prob[i, j] = float(np.mean(x > 0.0))
        return {"treatments": list(self.treatments), "mean": mean,
                "q_low": lo, "q_high": hi, "prob_positive": prob,
                "digits": int(digits)}

    def rank(self, small_values: str = "good", level=None) -> dict:
        """Ranking probabilities from the draws themselves.

        For every draw the treatments are ordered, so ``prob_best`` is the
        share of the posterior in which a treatment has the smallest (or
        largest, with ``small_values="bad"``) effect.  That is a probability
        about the parameter, not a p-value about a hypothesis, and it is
        available only because the whole posterior was sampled — which is the
        honest reason a Bayesian analysis ranks treatments more readily than a
        frequentist one does.

        ``sucra`` is the mean of :math:`P(\\text{rank} \\le r)` over
        :math:`r = 2 \\ldots n`, the same summary ``netmeta::netrank`` reports
        as SUCRA, so the two can be compared directly.
        """
        if small_values not in ("good", "bad"):
            raise ValueError("'small_values' must be 'good' or 'bad'")
        d = self._d_draws()
        n = len(self.treatments)
        ranks = np.argsort(np.argsort(d, axis=1), axis=1) + 1
        if small_values == "bad":
            ranks = n + 1 - ranks
        # share of draws in which each treatment took rank r, for every r
        share = np.stack([np.mean(ranks == r, axis=0) for r in range(1, n + 1)])
        cum = np.cumsum(share, axis=0)                 # P(rank <= r)
        sucra = cum[: n - 1].mean(axis=0) / (n - 1)
        return {
            "treatments": list(self.treatments),
            "prob_best": share[0].copy(),
            "prob_worst": share[n - 1].copy(),
            "mean_rank": ranks.mean(axis=0),
            "rank_share": share,
            "sucra": sucra,
            "small_values": small_values,
        }

    def intercept(self, study: str, level=None) -> dict:
        """Posterior mean and SD of one study's baseline mean :math:`\\mu_j`.

        The sampler never moved :math:`\\mu_j`: it was integrated out, which is
        what keeps the chain small.  Its posterior is therefore a mixture of the
        conditional Gaussians :math:`p(\\mu_j \\mid d, \\tau, y)`, one per draw,
        and the mixture's mean and variance are available exactly from the law
        of total variance.  They are reported as moments because that is what
        they are: an interval here would be a normal approximation wearing a
        credible interval's clothes.

        This exists because it is the check on the marginalisation.  ``multinma``
        *samples* :math:`\\mu_j`, so its posterior mean and SD for each study are
        a direct test of the analytic integral rather than of the same algebra
        twice.
        """
        if self._data is None:            # pragma: no cover - defensive
            raise RuntimeError("this result was built without its data")
        key = str(study)
        if key not in self._data.studies:
            raise ValueError(f"unknown study {study!r}")
        j = self._data.studies.index(key)
        cond_mean, cond_var = self._intercept_moments()
        mean = float(np.mean(cond_mean[:, j]))
        var = float(np.mean(cond_var[:, j]) + np.var(cond_mean[:, j], ddof=1))
        return {"study": key, "mean": mean, "sd": float(np.sqrt(max(var, 0.0)))}

    def _intercept_moments(self) -> tuple[np.ndarray, np.ndarray]:
        """Per draw, the conditional mean and variance of every study mean."""
        data = self._data
        draws = self.posterior.draws
        d = draws[:, : data.n_basic]
        if self.trt_effects == "random":
            tau2 = np.exp(draws[:, data.n_basic]) ** 2
        else:
            tau2 = np.zeros(draws.shape[0])
        resid = data.y[None, :] - d @ data.X.T                 # (S, m)
        w = 1.0 / (data.v[None, :] + tau2[:, None] * (~data.is_baseline)[None, :])
        a = np.add.reduceat(w, data.starts, axis=1)
        b = np.add.reduceat(w * resid, data.starts, axis=1)
        prec = 1.0 / float(self.priors["intercept_sd"]) ** 2 + a
        return b / prec, 1.0 / prec

    # -- report ------------------------------------------------------------ #
    def summary(self, level=None) -> list[dict]:
        """One row per parameter, in the order the sampler reported them."""
        alpha = level_to_alpha(self.level if level is None else level)
        return self.posterior.summary(level=1.0 - alpha)

    def fit(self) -> dict:
        """The diagnostics that say whether these draws can be summarised."""
        return {
            "n_chains": self.posterior.n_chains,
            "n_draws": self.posterior.n_draws,
            "accept": [float(v) for v in self.posterior.accept],
            "rhat_max": float(np.max(self.posterior.rhat())),
            "ess_min": float(np.min(self.posterior.ess())),
        }

    def to_dict(self) -> dict:
        return {
            "model": "arm-level Bayesian NMA with consistency",
            "treatments": list(self.treatments),
            "reference": self.reference,
            "basic": list(self.basic),
            "trt_effects": self.trt_effects,
            "tau": self.tau,
            "tau2": self.tau2,
            "tau_ci": list(self.tau_ci) if self.tau_ci is not None else None,
            "k": int(self.k),
            "n_arms": int(self.n_arms),
            "level": float(self.level),
            "priors": dict(self.priors),
            "posterior": self.posterior.to_dict(),
        }

    def __str__(self) -> str:
        alpha = 1.0 - self.level
        lo, hi = alpha / 2.0, 1.0 - alpha / 2.0
        width = max(len(t) for t in self.basic) + 4
        lines = [f"Bayesian network meta-analysis, {self.trt_effects} effects",
                 f"  {self.k} studies, {self.n_arms} arms, "
                 f"{len(self.treatments)} treatments, reference "
                 f"{self.reference!r}"]
        head = (f"{'effect':<{width}}{'mean':>9}{'sd':>8}"
                f"{f'{100 * lo:.1f}%':>9}{f'{100 * hi:.1f}%':>9}")
        lines.append(head)
        stats = {r["term"]: r for r in self.summary()}
        for t in self.basic:
            r = stats[f"d[{t}]"]
            lines.append(f"{'d[' + t + ']':<{width}}{r['mean']:>9.3f}"
                         f"{r['sd']:>8.3f}{r['q_low']:>9.3f}"
                         f"{r['q_high']:>9.3f}")
        if self.tau is not None:
            ci_note = ""
            if self.tau_ci is not None:
                ci_note = (f"  [{self.tau_ci[0]:.3f}, "
                           f"{self.tau_ci[1]:.3f}]")
            lines.append(f"{'tau':<{width}}{self.tau:>9.3f}{'':>8}"
                         f"{ci_note}")
        f = self.fit()
        lines.append(f"  {f['n_chains']} chains x {f['n_draws']} draws, "
                     f"acceptance {np.mean(f['accept']):.2f}, "
                     f"max R-hat {f['rhat_max']:.3f}, "
                     f"min ESS {f['ess_min']:.0f}")
        return "\n".join(lines)


# --------------------------------------------------------------------------- #
# the fit
# --------------------------------------------------------------------------- #
def bayes_nma(yi, *, study, treat, sei=None, vi=None, reference=None,
              trt_effects: str = "random", prior_d_sd: float = 10.0,
              prior_intercept_sd: float = 100.0, prior_tau_scale: float = 0.5,
              level=0.95, mcmc: dict | None = None) -> BayesNmaResult:
    """Fit a Bayesian network meta-analysis to arm-level data.

    Args:
        yi: arm-level effect estimates, one per arm.
        study: study label per arm.
        treat: treatment label per arm.  A study's arm on the reference
            treatment is its baseline and is pinned at zero; a study without
            that treatment gets a deviation on every arm.  Row order is not
            part of the model.
        sei: standard error per arm.
        vi: variance per arm; pass exactly one of ``sei`` and ``vi``.
        reference: the treatment whose effect is fixed at zero.  Defaults to
            the first treatment seen.  It changes the parameterisation and not
            the posterior of any contrast.
        trt_effects: ``"random"`` (default) or ``"fixed"``.
        prior_d_sd: SD of the normal prior on the basic treatment effects.
        prior_intercept_sd: SD of the normal prior on each study's baseline
            mean.  The baseline is integrated out analytically, so this is a
            proper prior with a closed-form marginal, not a device.
        prior_tau_scale: scale of the half-normal prior on the between-study
            SD.  Ignored when ``trt_effects="fixed"``.
        level: credible level, as ``0.95``, ``95`` or the alpha ``0.05``.
        mcmc: sampler overrides — ``n_draws``, ``warmup``, ``chains``,
            ``seed``, ``check``, ``rhat_max``, ``ess_min``.

    Returns:
        :class:`BayesNmaResult`.

    Raises:
        ValueError: for data that is not a network (a one-arm study, a repeated
            arm, a non-positive variance), for an unknown reference, or for an
            unknown ``mcmc`` key.
        RuntimeError: from :meth:`Posterior.check` when the draws did not mix.

    Example:
        >>> res = bayes_nma(                       # doctest: +SKIP
        ...     [0.1, 0.4, 0.2, 0.5], study=["s1", "s1", "s2", "s2"],
        ...     treat=["A", "B", "A", "C"], sei=[0.2, 0.2, 0.3, 0.3])
        >>> res.contrast("B", "A")["mean"]          # doctest: +SKIP
    """
    if trt_effects not in ("random", "fixed"):
        raise ValueError(
            "'trt_effects' must be 'random' or 'fixed', not "
            f"{trt_effects!r}")
    alpha = level_to_alpha(level)
    cfg = dict(_DEFAULTS)
    if mcmc:
        unknown = set(mcmc) - set(cfg)
        if unknown:
            raise ValueError(
                f"unknown 'mcmc' setting(s): {', '.join(sorted(map(str, unknown)))}; "
                f"known keys are {', '.join(sorted(cfg))}")
        cfg.update(mcmc)
    if prior_d_sd <= 0.0 or prior_intercept_sd <= 0.0:
        raise ValueError("prior scales must be positive")
    if trt_effects == "random" and prior_tau_scale <= 0.0:
        raise ValueError("'prior_tau_scale' must be positive")

    data = _prepare(yi, sei, vi, study, treat, reference)
    random = trt_effects == "random"
    n_par = data.n_basic + 1 if random else data.n_basic

    def log_post(theta: np.ndarray) -> float:
        d = theta[: data.n_basic]
        value = -0.5 * float(d @ d) / float(prior_d_sd) ** 2
        if random:
            log_tau = float(theta[data.n_basic])
            tau = float(np.exp(log_tau))
            if not np.isfinite(tau):
                return -np.inf
            value += -0.5 * (tau / float(prior_tau_scale)) ** 2 + log_tau
            tau2 = tau * tau
        else:
            tau2 = 0.0
        return value + _marginal_loglik(d, tau2, data,
                                        float(prior_intercept_sd))

    start = _starting_values(data, 0.1)
    if not random:
        start = start[: data.n_basic]
    for trial in (start, np.zeros(n_par)):
        if np.isfinite(log_post(trial)):
            start = trial
            break
    else:                                       # pragma: no cover - defensive
        raise ValueError(
            "the log density is not finite at any starting point; the data "
            "may be degenerate")

    blocks = [np.arange(data.n_basic)]
    names = [f"d[{t}]" for t in data.basic]
    if random:
        blocks.append(np.array([data.n_basic]))
        names.append("log_tau")

    post = sample_log_posterior(
        log_post, start, names=names,
        n_draws=int(cfg["n_draws"]), warmup=int(cfg["warmup"]),
        chains=int(cfg["chains"]), seed=cfg["seed"], init_sd=0.05,
        blocks=blocks)
    if cfg["check"]:
        post.check(rhat_max=float(cfg["rhat_max"]),
                   ess_min=float(cfg["ess_min"]))

    if random:
        tau_draws = np.exp(post.draws[:, data.n_basic])
        tau = float(np.mean(tau_draws))
        tau2 = float(np.mean(tau_draws ** 2))
        tail = alpha / 2.0
        tau_ci: tuple[float, float] | None = (float(np.quantile(tau_draws, tail)),
                                             float(np.quantile(tau_draws,
                                                               1.0 - tail)))
    else:
        tau = None
        tau2 = None
        tau_ci = None

    return BayesNmaResult(
        posterior=post,
        treatments=list(data.treatments),
        reference=data.reference,
        basic=list(data.basic),
        trt_effects=trt_effects,
        tau=tau,
        tau2=tau2,
        tau_ci=tau_ci,
        k=len(data.studies),
        n_arms=int(data.y.size),
        level=1.0 - alpha,
        priors={
            "d_sd": float(prior_d_sd),
            "intercept_sd": float(prior_intercept_sd),
            "tau_scale": float(prior_tau_scale) if random else None,
            "tau_distribution": "half-normal" if random else "none (fixed)",
        },
        _data=data,
    )
