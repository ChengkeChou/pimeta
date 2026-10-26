"""Data-generating processes and calibration summaries for simulation studies.

The point of this module is not to draw data. It is to answer the question a
method has to answer before anyone may rely on it: *when the truth is known, do
the intervals cover it, does the test reject at the rate it advertises, and does
the standard error mean what it says?*

Every number produced here is a frequency over independent replications, so
every number carries a Monte Carlo standard error. A coverage of 0.93 is not a
defect and a coverage of 0.87 is not a small defect; the MCSE is what tells them
apart, which is why it is reported next to every estimate instead of being left
to the reader.

What the intervals are checked against matters, and it is easy to get wrong:

* the confidence interval for the mean effect is checked against ``mu``, the
  fixed population mean;
* the prediction interval is checked against the true effect of a *new* study,
  drawn from ``N(mu, tau^2)`` -- **not** against ``mu``. A prediction interval
  that is checked against the mean will look conservative, because it is being
  asked the wrong question.

Both are drawn here per replication, so the check is the one the interval
claims to make.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from functools import partial
from typing import Any

import numpy as np

from .parallel import map_ordered, resolve_jobs

__all__ = [
    "SCHEMA_VERSION",
    "Scenario",
    "Dataset",
    "SCENARIOS",
    "simulate_study",
    "run_scenario",
    "summarise_fits",
]
SCHEMA_VERSION = 1

#: How the sampling standard errors are spread across studies. The spread
#: matters because a meta-analysis with equal precisions is the easy case;
#: unequal precisions are where weighting schemes actually differ.
SE_KINDS = ("equal", "lognormal", "wide", "one_precise")


@dataclass(frozen=True)
class Scenario:
    """A data-generating process with a known truth.

    Attributes:
        name: label written into the report.
        k: number of studies per replication.
        mu: true population mean effect on the analysis scale.
        tau2: true between-study variance; ``0.0`` gives a fixed-effect truth,
            where a random-effects estimator should put mass at the boundary.
        se: typical sampling standard error.
        se_kind: how the standard errors vary across studies --
            ``"equal"`` (all ``se``), ``"lognormal"`` (``se * exp(N(0, 0.5))``),
            ``"wide"`` (a geometric spread from ``se/2`` to ``2 se``) or
            ``"one_precise"`` (all ``se`` except one study at ``se/10``).
        outlier: when not ``None``, one study's effect is shifted by this many
            of its own standard errors, which no Gaussian model anticipates.
        note: free text recorded in the report.
    """

    name: str
    k: int = 20
    mu: float = 0.0
    tau2: float = 0.04
    se: float = 0.2
    se_kind: str = "equal"
    outlier: float | None = None
    note: str = ""

    def __post_init__(self) -> None:
        if self.k < 2:
            raise ValueError("a meta-analysis needs at least two studies")
        if self.tau2 < 0.0:
            raise ValueError("tau2 cannot be negative")
        if self.se <= 0.0:
            raise ValueError("se must be positive")
        if self.se_kind not in SE_KINDS:
            raise ValueError(
                f"se_kind must be one of {SE_KINDS}, got {self.se_kind!r}"
            )


@dataclass(frozen=True)
class Dataset:
    """One replication: the observed data plus the truth that produced it."""

    yi: np.ndarray
    sei: np.ndarray
    mu: float
    tau2: float
    #: True effect of a study that was *not* observed -- the target of a
    #: prediction interval.
    mu_new: float
    truth: dict[str, float] = field(default_factory=dict)

    @property
    def vi(self) -> np.ndarray:
        return self.sei ** 2

    def __len__(self) -> int:
        return int(self.yi.size)


def _sampling_ses(scenario: Scenario, rng: np.random.Generator) -> np.ndarray:
    k = scenario.k
    if scenario.se_kind == "equal":
        return np.full(k, scenario.se)
    if scenario.se_kind == "lognormal":
        return scenario.se * np.exp(rng.normal(0.0, 0.5, size=k))
    if scenario.se_kind == "wide":
        return scenario.se * np.exp(np.linspace(np.log(0.5), np.log(2.0), k))
    if scenario.se_kind == "one_precise":
        sei = np.full(k, scenario.se)
        sei[-1] = scenario.se / 10.0
        return sei
    raise ValueError(f"unknown se_kind {scenario.se_kind!r}")


def simulate_study(scenario: Scenario,
                   rng: np.random.Generator | int | None = None) -> Dataset:
    """Draw one replication from ``scenario``.

    The generative model is the one a random-effects meta-analysis assumes:

    .. math::

        \\mu_i \\sim N(\\mu, \\tau^2), \\qquad
        y_i \\sim N(\\mu_i, v_i)

    with ``v_i`` the sampling variance. When ``scenario.outlier`` is set, one
    study's observed effect is displaced by that many standard errors after
    generation, so the truth (``mu_i``) no longer matches the data for that
    study and no Gaussian likelihood is correct for it.
    """
    if not isinstance(rng, np.random.Generator):
        rng = np.random.default_rng(rng)
    sei = _sampling_ses(scenario, rng)
    mu_i = rng.normal(scenario.mu, np.sqrt(scenario.tau2), size=scenario.k)
    yi = rng.normal(mu_i, sei)
    mu_new = float(rng.normal(scenario.mu, np.sqrt(scenario.tau2)))
    if scenario.outlier is not None:
        j = int(rng.integers(scenario.k))
        yi[j] += scenario.outlier * sei[j]
    truth = {
        "mu": float(scenario.mu),
        "tau2": float(scenario.tau2),
        "tau": float(np.sqrt(scenario.tau2)),
        "mu_new": mu_new,
        "k": float(scenario.k),
    }
    return Dataset(yi=yi, sei=sei, mu=float(scenario.mu),
                   tau2=float(scenario.tau2), mu_new=mu_new, truth=truth)


# --------------------------------------------------------------------------
# summarising a set of replications
# --------------------------------------------------------------------------

def _mcse_mean(values: np.ndarray) -> float:
    n = values.size
    if n < 2:
        return float("nan")
    return float(values.std(ddof=1) / np.sqrt(n))


def _mcse_rate(p: float, n: int) -> float:
    if n < 2:
        return float("nan")
    return float(np.sqrt(max(p * (1.0 - p), 0.0) / n))


def _rate(hits: np.ndarray) -> dict[str, Any]:
    n = int(hits.size)
    p = float(hits.mean()) if n else float("nan")
    return {
        "estimate": p,
        "mcse": _mcse_rate(p, n),
        "hits": int(hits.sum()),
        "reps": n,
    }


def summarise_fits(estimates: list[dict[str, Any]], mu: float,
                   tau2: float) -> dict:
    """Reduce per-replication records to the six calibration blocks.

    Each record is a mapping with the keys ``b`` (mean estimate), ``se``
    (its standard error), ``ci_lb``, ``ci_ub``, ``pval``, ``tau2``,
    ``I2``, ``pi_lb``, ``pi_ub`` and ``mu_new``. Records whose ``b`` is not
    finite are excluded from every summary and counted in ``reliability`` --
    silently averaging over the replications that happened to work is how a
    broken estimator acquires a good-looking bias.
    """
    n_all = len(estimates)
    if n_all == 0:
        raise ValueError("no replications to summarise")
    ok = [r for r in estimates if np.isfinite(r["b"])]
    n = len(ok)
    if n == 0:
        raise ValueError("every replication produced a non-finite estimate")

    b = np.array([r["b"] for r in ok], dtype=float)
    se = np.array([r["se"] for r in ok], dtype=float)
    ci_lb = np.array([r["ci_lb"] for r in ok], dtype=float)
    ci_ub = np.array([r["ci_ub"] for r in ok], dtype=float)
    pval = np.array([r["pval"] for r in ok], dtype=float)
    tau2_hat = np.array([r["tau2"] for r in ok], dtype=float)
    i2 = np.array([r["I2"] for r in ok], dtype=float)
    pi_lb = np.array([np.nan if r["pi_lb"] is None else r["pi_lb"] for r in ok],
                     dtype=float)
    pi_ub = np.array([np.nan if r["pi_ub"] is None else r["pi_ub"] for r in ok],
                     dtype=float)
    pi_lb_t = np.array([np.nan if r.get("pi_lb_t") is None else r["pi_lb_t"]
                        for r in ok], dtype=float)
    pi_ub_t = np.array([np.nan if r.get("pi_ub_t") is None else r["pi_ub_t"]
                        for r in ok], dtype=float)
    mu_new = np.array([r["mu_new"] for r in ok], dtype=float)

    bias = float(b.mean() - mu)
    emp_sd = float(b.std(ddof=1)) if n > 1 else float("nan")
    mean_se = float(se.mean())
    rmse = float(np.sqrt(np.mean((b - mu) ** 2)))
    se_ratio = float(mean_se / emp_sd) if emp_sd > 0 else float("nan")
    # The ratio of the mean reported SE to the empirical SD is a ratio of two
    # noisy estimates, so its Monte Carlo error has to be propagated: a
    # tolerance taken from the flat rate MCSE would be far too tight and would
    # flag ordinary sampling noise as a change in behaviour.
    se_mcse = _mcse_mean(se)
    emp_sd_mcse = (float(emp_sd / np.sqrt(2.0 * (n - 1)))
                   if n > 1 and emp_sd > 0 else float("nan"))
    if emp_sd > 0 and mean_se > 0 and n > 1:
        rel = ((se_mcse / mean_se) ** 2 + (emp_sd_mcse / emp_sd) ** 2)
        se_ratio_mcse = float(se_ratio * np.sqrt(rel))
    else:
        se_ratio_mcse = float("nan")

    ci_hit = (ci_lb <= mu) & (mu <= ci_ub)
    pi_ok = np.isfinite(pi_lb) & np.isfinite(pi_ub)
    pi_t_ok = np.isfinite(pi_lb_t) & np.isfinite(pi_ub_t)
    # Coverage is a frequency over the replications that actually produced an
    # interval. Dividing by every replication would report a model that cannot
    # predict as having missed, which is a different diagnosis.
    pi_hit = pi_ok[pi_ok] & (pi_lb[pi_ok] <= mu_new[pi_ok]) & (
        mu_new[pi_ok] <= pi_ub[pi_ok])
    pi_t_hit = pi_t_ok[pi_t_ok] & (pi_lb_t[pi_t_ok] <= mu_new[pi_t_ok]) & (
        mu_new[pi_t_ok] <= pi_ub_t[pi_t_ok])
    reject = pval < 0.05

    tau2_bias = float(tau2_hat.mean() - tau2)
    tau2_rmse = float(np.sqrt(np.mean((tau2_hat - tau2) ** 2)))

    return {
        "counts": {"reps": n_all, "reps_used": n,
                   "reps_non_finite": n_all - n},
        "bias": {
            "mean_estimate": float(b.mean()),
            "truth": float(mu),
            "estimate": bias,
            "mcse": _mcse_mean(b),
            "relative": (float(bias / mu) if mu != 0.0 else None),
        },
        "dispersion": {
            "rmse": rmse,
            "rmse_mcse": _mcse_mean(np.sqrt((b - mu) ** 2)) / np.sqrt(2.0),
            "empirical_sd": emp_sd,
            "empirical_sd_mcse": emp_sd_mcse,
            "mean_reported_se": mean_se,
            "mean_reported_se_mcse": se_mcse,
            #: Should be 1.0. Above 1 means the reported SE is too big (and the
            #: intervals too wide); below 1 means the intervals are too narrow
            #: even though every individual one looks reasonable.
            "se_ratio": se_ratio,
            "se_ratio_mcse": se_ratio_mcse,
            "se_ratio_note": "mean reported SE divided by the empirical SD of "
                             "the estimates; 1.0 is calibrated",
        },
        "coverage": {
            "ci": _rate(ci_hit),
            "ci_target": 0.95,
            "pi": _rate(pi_hit),
            "pi_t": _rate(pi_t_hit),
            "pi_target": 0.95,
            "pi_defined_reps": int(pi_ok.sum()),
            "pi_target_note": "prediction intervals are checked against the true "
                              "effect of an unobserved new study, drawn from "
                              "N(mu, tau^2)",
            "pi_variants_note": "'pi' is the interval the default options give; "
                                "'pi_t' is the same fit with predtype='t', which "
                                "charges the t distribution for the uncertainty "
                                "in tau^2. They are reported side by side so the "
                                "cost of the default is a measurement, not an "
                                "opinion",
        },
        "tests": {
            "rejection_rate": _rate(reject),
            "alpha": 0.05,
            #: A rejection rate is a size when the truth is the null and a power
            #: when it is not; the field says which one this number is so it
            #: cannot be read as the wrong one.
            "role": "size" if mu == 0.0 else "power",
        },
        "heterogeneity": {
            "tau2_mean": float(tau2_hat.mean()),
            "tau2_truth": float(tau2),
            "tau2_bias": tau2_bias,
            "tau2_bias_mcse": _mcse_mean(tau2_hat),
            "tau2_rmse": tau2_rmse,
            "tau2_at_boundary": float(np.mean(tau2_hat <= 1e-10)),
            "i2_mean": float(i2.mean()),
        },
        "reliability": {
            "fraction_finite": float(n / n_all),
            "fraction_tau2_at_boundary": float(np.mean(tau2_hat <= 1e-10)),
            "note": "fraction_finite below 1 means some replications failed; "
                    "the summaries above describe only the rest",
        },
    }


# --------------------------------------------------------------------------
# running a scenario end to end
# --------------------------------------------------------------------------

def _record_for(data: Dataset, res: Any, level: float) -> dict[str, Any]:
    """One replication's fit as the record :func:`summarise_fits` reads.

    A model that cannot predict is recorded as having no interval rather than
    as having missed: those are different diagnoses and only the second one is
    about calibration.
    """
    if res is None:
        return {"b": float("nan"), "se": float("nan"),
                "ci_lb": float("nan"), "ci_ub": float("nan"),
                "pval": float("nan"), "tau2": float("nan"),
                "I2": float("nan"), "pi_lb": None, "pi_ub": None,
                "mu_new": data.mu_new}
    try:
        pred = res.predict(level)
        pi_lb, pi_ub = pred.pi_lb, pred.pi_ub
    except Exception:  # noqa: BLE001 - a model that cannot predict is data
        pi_lb, pi_ub = None, None
    try:
        pred_t = res.predict(level, predtype="t")
        pi_lb_t, pi_ub_t = pred_t.pi_lb, pred_t.pi_ub
    except Exception:  # noqa: BLE001
        pi_lb_t, pi_ub_t = None, None
    return {
        "b": float(res.b[0]),
        "se": float(res.se[0]),
        "ci_lb": float(res.ci_lb[0]),
        "ci_ub": float(res.ci_ub[0]),
        "pval": float(res.pval[0]),
        "tau2": float(res.tau2),
        "I2": float(res.I2),
        "pi_lb": None if pi_lb is None else float(pi_lb),
        "pi_ub": None if pi_ub is None else float(pi_ub),
        "pi_lb_t": None if pi_lb_t is None else float(pi_lb_t),
        "pi_ub_t": None if pi_ub_t is None else float(pi_ub_t),
        "mu_new": data.mu_new,
    }


def _default_fit_record(data: Dataset, *, method: str,
                        level: float) -> dict[str, Any]:
    """Fit one replication with :func:`pimeta.rma`.

    Module-level and keyword-only so a :class:`functools.partial` of it can be
    pickled and sent to a worker process; a closure would not be.
    """
    return _record_for(data, _default_fit(method=method, level=level)(data),
                       level)


def _record_for_estimator(data: Dataset, *, estimator: Callable[[Any], Any],
                          level: float) -> dict[str, Any]:
    """The same, for a caller-supplied estimator."""
    return _record_for(data, estimator(data), level)


def run_scenario(
    scenario: Scenario,
    reps: int = 1000,
    seed: int = 1,
    *,
    method: str = "REML",
    level: float = 0.95,
    fit: Callable[[Dataset], Any] | None = None,
    n_jobs: int | None = None,
) -> dict[str, Any]:
    """Simulate ``reps`` replications of ``scenario`` and summarise them.

    Args:
        scenario: the data-generating process.
        reps: number of independent replications.
        seed: seed for the replication generator; the same seed gives the same
            report, which is what makes a stored baseline meaningful.
        method: tau^2 estimator passed to :func:`pimeta.rma`.
        level: confidence level.
        fit: optional replacement for the estimator. It receives a
            :class:`Dataset` and must return an object carrying ``b``, ``se``,
            ``ci_lb``, ``ci_ub``, ``pval``, ``tau2``, ``I2`` and ``predict()``.
            Injecting it lets this module be used for models other than
            :func:`pimeta.rma` without importing them here.
        n_jobs: worker processes for the *fits*. ``None`` or ``1`` runs them in
            this process. All ``reps`` datasets are simulated first, serially,
            from the one seeded generator, so the random stream a replication
            consumes does not depend on how many workers there are: the report
            is identical to the serial one to the last bit, and
            ``tests/test_parallel.py`` checks that with ``float.hex()``. Only
            the fitting is spread out, and each fit receives a dataset rather
            than a generator. ``n_jobs > 1`` needs ``fit`` to be picklable
            (a module-level function or a ``partial`` of one).

    Returns:
        A JSON-serialisable report with the six calibration blocks plus the
        settings that produced them.
    """
    if reps < 2:
        raise ValueError("reps must be at least 2 for a Monte Carlo error")
    rng = np.random.default_rng(seed)
    # Simulate everything before fitting anything: that is what makes the
    # result independent of n_jobs.
    datasets = [simulate_study(scenario, rng) for _ in range(reps)]
    if fit is None:
        per_rep: Callable[[Dataset], dict[str, Any]] = partial(
            _default_fit_record, method=method, level=level)
    else:
        per_rep = partial(_record_for_estimator, estimator=fit, level=level)
    records = map_ordered(per_rep, datasets, n_jobs=n_jobs)

    report = summarise_fits(records, scenario.mu, scenario.tau2)
    report["schema_version"] = SCHEMA_VERSION
    report["scenario"] = _scenario_dict(scenario)
    report["settings"] = {"reps": reps, "seed": seed, "method": method,
                          "level": level,
                          "n_jobs": resolve_jobs(n_jobs, reps)}
    return report


def _scenario_dict(scenario: Scenario) -> dict[str, Any]:
    return {
        "name": scenario.name,
        "k": scenario.k,
        "mu": scenario.mu,
        "tau2": scenario.tau2,
        "se": scenario.se,
        "se_kind": scenario.se_kind,
        "outlier": scenario.outlier,
        "note": scenario.note,
    }


def _default_fit(*, method: str, level: float) -> Callable[[Dataset], Any]:
    from .rma_uni import rma

    def _fit(data: Dataset):
        return rma(data.yi, vi=data.vi, method=method, level=level)

    return _fit


#: Scenarios kept in the repository as baselines. Each one exists to expose a
#: specific failure mode, not to be a representative sample of meta-analyses.
SCENARIOS: dict[str, Scenario] = {
    "null": Scenario(
        name="null", k=20, mu=0.0, tau2=0.04, se=0.2,
        note="size: the mean effect is truly zero, mild heterogeneity",
    ),
    "alt": Scenario(
        name="alt", k=20, mu=0.3, tau2=0.04, se=0.2,
        note="power: a real effect of 0.3 with the same SE as a single study",
    ),
    "fixed": Scenario(
        name="fixed", k=20, mu=0.2, tau2=0.0, se=0.2,
        note="tau2 sits on the boundary; coverage must not collapse there",
    ),
    "tiny": Scenario(
        name="tiny", k=5, mu=0.2, tau2=0.04, se=0.2,
        note="k=5, where every variance estimator is unstable",
    ),
    "unequal": Scenario(
        name="unequal", k=20, mu=0.2, tau2=0.04, se=0.2, se_kind="wide",
        note="precisions spread by a factor of four",
    ),
    "outlier": Scenario(
        name="outlier", k=20, mu=0.2, tau2=0.04, se=0.2, outlier=5.0,
        note="one contaminated study; the Gaussian likelihood is wrong for it",
    ),
}
