# Validation

*6 public names, generated from `pimeta.__all__` and the docstrings in `src/pimeta/`. Do not edit this page by hand — run `python tools/gen_api_docs.py`.*

| Name | Mirrors in R |
|---|---|
| [`SCENARIOS`](#scenarios) | — |
| [`Scenario`](#scenario) | — |
| [`Dataset`](#dataset) | — |
| [`simulate_study`](#simulate_study) | — |
| [`summarise_fits`](#summarise_fits) | — |
| [`run_scenario`](#run_scenario) | — |

## `SCENARIOS`

*Mapping (`dict`) with 6 keys:* `null`, `alt`, `fixed`, `tiny`, `unequal`, `outlier`

## `Scenario`

```python
Scenario(name: 'str', k: 'int' = 20, mu: 'float' = 0.0, tau2: 'float' = 0.04, se: 'float' = 0.2, se_kind: 'str' = 'equal', outlier: 'float | None' = None, note: 'str' = '') -> None
```

A data-generating process with a known truth.

**Attributes**

- `name` — label written into the report.
- `k` — number of studies per replication.
- `mu` — true population mean effect on the analysis scale.
- `tau2` — true between-study variance; ``0.0`` gives a fixed-effect truth, where a random-effects estimator should put mass at the boundary.
- `se` — typical sampling standard error.
- `se_kind` — how the standard errors vary across studies -- ``"equal"`` (all ``se``), ``"lognormal"`` (``se * exp(N(0, 0.5))``), ``"wide"`` (a geometric spread from ``se/2`` to ``2 se``) or ``"one_precise"`` (all ``se`` except one study at ``se/10``).
- `outlier` — when not ``None``, one study's effect is shifted by this many of its own standard errors, which no Gaussian model anticipates.
- `note` — free text recorded in the report.

## `Dataset`

```python
Dataset(yi: 'np.ndarray', sei: 'np.ndarray', mu: 'float', tau2: 'float', mu_new: 'float', truth: 'dict[str, float]' = <factory>) -> None
```

One replication: the observed data plus the truth that produced it.

## `simulate_study`

```python
simulate_study(scenario: 'Scenario', rng: 'np.random.Generator | int | None' = None) -> 'Dataset'
```

Draw one replication from ``scenario``.

The generative model is the one a random-effects meta-analysis assumes:

.. math::

    \mu_i \sim N(\mu, \tau^2), \qquad
    y_i \sim N(\mu_i, v_i)

with ``v_i`` the sampling variance. When ``scenario.outlier`` is set, one
study's observed effect is displaced by that many standard errors after
generation, so the truth (``mu_i``) no longer matches the data for that
study and no Gaussian likelihood is correct for it.

## `summarise_fits`

```python
summarise_fits(estimates: 'list[dict[str, Any]]', mu: 'float', tau2: 'float') -> 'dict'
```

Reduce per-replication records to the six calibration blocks.

Each record is a mapping with the keys ``b`` (mean estimate), ``se``
(its standard error), ``ci_lb``, ``ci_ub``, ``pval``, ``tau2``,
``I2``, ``pi_lb``, ``pi_ub`` and ``mu_new``. Records whose ``b`` is not
finite are excluded from every summary and counted in ``reliability`` --
silently averaging over the replications that happened to work is how a
broken estimator acquires a good-looking bias.

## `run_scenario`

```python
run_scenario(scenario: 'Scenario', reps: 'int' = 1000, seed: 'int' = 1, *, method: 'str' = 'REML', level: 'float' = 0.95, fit: 'Callable[[Dataset], Any] | None' = None, n_jobs: 'int | None' = None) -> 'dict[str, Any]'
```

Simulate ``reps`` replications of ``scenario`` and summarise them.

**Args**

- `scenario` — the data-generating process.
- `reps` — number of independent replications.
- `seed` — seed for the replication generator; the same seed gives the same report, which is what makes a stored baseline meaningful.
- `method` — tau^2 estimator passed to `rma`.
- `level` — confidence level.
- `fit` — optional replacement for the estimator. It receives a `Dataset` and must return an object carrying ``b``, ``se``, ``ci_lb``, ``ci_ub``, ``pval``, ``tau2``, ``I2`` and ``predict()``. Injecting it lets this module be used for models other than `rma` without importing them here.
- `n_jobs` — worker processes for the *fits*. ``None`` or ``1`` runs them in this process. All ``reps`` datasets are simulated first, serially, from the one seeded generator, so the random stream a replication consumes does not depend on how many workers there are: the report is identical to the serial one to the last bit, and ``tests/test_parallel.py`` checks that with ``float.hex()``. Only the fitting is spread out, and each fit receives a dataset rather than a generator. ``n_jobs > 1`` needs ``fit`` to be picklable (a module-level function or a ``partial`` of one).

**Returns**

A JSON-serialisable report with the six calibration blocks plus the settings that produced them.
