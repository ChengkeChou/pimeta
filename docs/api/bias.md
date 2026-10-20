# Publication bias

*30 public names, generated from `pimeta.__all__` and the docstrings in `src/pimeta/`. Do not edit this page by hand — run `python tools/gen_api_docs.py`.*

| Name | Mirrors in R |
|---|---|
| [`egger_test`](#egger_test) | — |
| [`EggerResult`](#eggerresult) | — |
| [`begg_test`](#begg_test) | — |
| [`BeggResult`](#beggresult) | — |
| [`trimfill`](#trimfill) | — |
| [`TrimfillResult`](#trimfillresult) | — |
| [`influence`](#influence) | — |
| [`InfluenceResult`](#influenceresult) | — |
| [`leave1out`](#leave1out) | — |
| [`LeaveOneOutResult`](#leaveoneoutresult) | — |
| [`hatvalues`](#hatvalues) | — |
| [`cooks_distance`](#cooks_distance) | — |
| [`rstudent`](#rstudent) | — |
| [`cumul`](#cumul) | cumul |
| [`CumulResult`](#cumulresult) | — |
| [`fsn`](#fsn) | fsn |
| [`FsnResult`](#fsnresult) | — |
| [`gosh`](#gosh) | — |
| [`GoshResult`](#goshresult) | — |
| [`metainf`](#metainf) | metainf |
| [`MetainfResult`](#metainfresult) | metainf |
| [`metacum`](#metacum) | metacum |
| [`MetacumResult`](#metacumresult) | metacum |
| [`regtest`](#regtest) | — |
| [`RegtestResult`](#regtestresult) | — |
| [`metabias`](#metabias) | metabias |
| [`MetabiasResult`](#metabiasresult) | metabias |
| [`peters_test`](#peters_test) | — |
| [`baujat`](#baujat) | — |
| [`BaujatResult`](#baujatresult) | — |

## `egger_test`

```python
egger_test(y, v, model: 'str' = 'rma', method: 'str' = 'REML') -> 'EggerResult'
```

Egger's regression test (metafor::regtest).

``model="lm"``  : classical Egger test — weighted regression of ``y`` on
                  ``sqrt(v)`` with weights ``1/v``, multiplicative
                  dispersion, and a t-test on the *slope*.
``model="rma"`` : mixed-effects meta-regression version (metafor default)
                  — ``metareg(y, v, X=sqrt(v))`` with a z-test on the
                  *slope*.

The test statistic is the coefficient of ``sqrt(v)`` (the slope), exactly
as in metafor::regtest. A significant slope indicates funnel-plot
asymmetry; the intercept is the "limit estimate" (pooled effect under
infinite precision).

## `EggerResult`

```python
EggerResult(intercept: 'float', intercept_se: 'float', stat: 'float', p: 'float', slope: 'float', slope_se: 'float', k: 'int', model: 'str') -> None
```

Egger regression test output.

## `begg_test`

```python
begg_test(y, v) -> 'BeggResult'
```

Begg & Mazumdar rank correlation test (metafor::ranktest).

## `BeggResult`

```python
BeggResult(tau: 'float', z: 'float', p: 'float', k: 'int') -> None
```

Begg & Mazumdar rank correlation test output.

## `trimfill`

```python
trimfill(y, v, effect_type: 'str' = 'MD', estimator: 'str' = 'L0', method_tau: 'str' = 'REML', alpha: 'float' = 0.05) -> 'TrimfillResult'
```

Duval & Tweedie trim-and-fill (metafor::trimfill.rma.uni).

Replicates the exact iterative algorithm:

1. Determine the side of asymmetry from Egger's regression test
   (slope of ``y ~ sqrt(v)`` < 0 -> "right", else "left").
2. If "right", flip the signs so the problem is a left-side fill.
3. Sort by ascending effect; iteratively truncate the ``k0`` most
   extreme studies, re-estimating the centre until ``k0`` converges.
4. Impute ``k0`` mirrored studies (``2*centre - yi``) and re-pool.

``method_tau`` mirrors metafor's ``x$method`` (EE/FE/CE for fixed,
DL/REML/ML/... for random effects).

## `TrimfillResult`

```python
TrimfillResult(k0: 'int', estimator: 'str', side: 'str', se_k0: 'float', p_k0: 'float | None', filled_effect: 'float', filled_se: 'float', filled_ci_lower: 'float', filled_ci_upper: 'float', filled_tau2: 'float', original_effect: 'float', k: 'int') -> None
```

Trim-and-fill output.

## `influence`

```python
influence(y, v, X=None, X_names=None, method: 'str' = 'REML', tau2: 'float | None' = None, test: 'str' = 'z', alpha: 'float' = 0.05) -> 'InfluenceResult'
```

Leave-one-out influence diagnostics (metafor::influence).

**Args**

- `y, v` — analysis-scale effect and sampling variance.
- `X` — moderator design matrix (n, m) WITHOUT intercept, or None for an intercept-only model.
- `method` — tau^2 estimator ("FE", "DL", "ML" or "REML"; default REML).
- `tau2` — optional fixed tau^2 (when given, it is not re-estimated in the leave-one-out refits).
- `test` — "z" (default) or "knha".

## `InfluenceResult`

```python
InfluenceResult(rstudent: 'list[float]', dffits: 'list[float]', cook_d: 'list[float]', cov_r: 'list[float]', tau2_del: 'list[float]', qe_del: 'list[float]', hat: 'list[float]', weight: 'list[float]', dfbs: 'list[list[float]]', beta_del: 'list[list[float]]', se_del: 'list[list[float]]', is_infl: 'list[bool]', tau2: 'float', QE: 'float', k: 'int', p: 'int', names: 'list[str]' = <factory>) -> None
```

Leave-one-out / influence diagnostics (metafor::influence).

## `leave1out`

```python
leave1out(y, v=None, method: 'str' = 'REML', tau2: 'float | None' = None, alpha: 'float' = 0.05, cluster=None) -> 'LeaveOneOutResult'
```

Leave-one-out analysis for an intercept-only model (metafor::leave1out).

For each study, refits the model without it and records the pooled
estimate, SE, z, p, CI, QE and tau^2 / I^2 / H^2.  An ``rma.mh`` /
``rma.peto`` result can be passed directly, in which case metafor's
``leave1out.rma.mh`` / ``.peto`` is used (``cluster`` collapses studies).

## `LeaveOneOutResult`

```python
LeaveOneOutResult(estimate: 'list[float]', se: 'list[float]', z: 'list[float]', p: 'list[float]', ci_lower: 'list[float]', ci_upper: 'list[float]', QE: 'list[float]', QE_p: 'list[float]', tau2: 'list[float]', I2: 'list[float]', H2: 'list[float]') -> None
```

Leave-one-out pooled estimates (metafor::leave1out).

## `hatvalues`

```python
hatvalues(y, v=None, X=None, method: 'str' = 'REML', tau2: 'float | None' = None, type='diagonal')
```

Diagonal of the hat matrix (metafor::hatvalues).

An ``rma.mv`` fit can be passed instead of ``(y, v)``, in which case
metafor's ``hatvalues.rma.mv`` is used and ``type="matrix"`` returns the
full hat matrix.

## `cooks_distance`

```python
cooks_distance(y, v=None, X=None, method: 'str' = 'REML', tau2: 'float | None' = None, cluster=None)
```

Cook's distances (metafor::cooks.distance).

An ``rma.mv`` fit can be passed instead of ``(y, v)``, in which case
metafor's ``cooks.distance.rma.mv`` is used (``cluster`` collapses the
studies into groups).

## `rstudent`

```python
rstudent(y, v=None, X=None, method: 'str' = 'REML', tau2: 'float | None' = None, cluster=None)
```

Externally standardized (deleted) residuals (metafor::rstudent).

Accepts ``(y, v, ...)``, an ``rma.mh`` / ``rma.peto`` result or an
``rma.mv`` fit; the latter two use metafor's own methods and return a dict
(``resid``/``se``/``z``/``slab`` for mh/peto, ``delresid``/``sedelresid``/
``z``/``X2``/``slab`` for rma.mv).

## `cumul`

```python
cumul(yi, vi=None, n=None, effect_type: 'str' = 'MD', method: 'str' = 'DL', alpha: 'float' = 0.05, order=None, decreasing: 'bool' = False, collapse: 'bool' = False) -> 'CumulResult'
```

*Mirrors R's `cumul`.*

Cumulative (sequential) meta-analysis (mirrors ``meta::cumul``).

**Args**

- `yi, vi` — per-study effect and sampling variance (analysis scale).
- `n` — per-study sample size (for the HS tau2 estimator).
- `effect_type` — passed through to pooling (for back-transformation).
- `method` — tau-squared estimator (default DL).
`rma.mh`` / ``rma.peto`` result can be passed as ``yi``, in which case
for's ``cumul.rma.mh`` / ``.peto`` is used (``order``, ``decreasing``
``collapse`` then control the ordering).

## `CumulResult`

```python
CumulResult(estimate: 'list[float]', se: 'list[float]', z: 'list[float]', p: 'list[float]', ci_lower: 'list[float]', ci_upper: 'list[float]', q: 'list[float]', q_p: 'list[float]', tau2: 'list[float]', i2: 'list[float]', h2: 'list[float]', k: 'list[int]' = <factory>) -> None
```

Cumulative meta-analysis: one row per prefix of studies.

## `fsn`

```python
fsn(yi, vi=None, sei=None, type: 'str' = 'Rosenthal', alpha: 'float' = 0.05, target: 'float | None' = None) -> 'FsnResult'
```

*Mirrors R's `fsn`.*

Compute the fail-safe N (mirrors ``metafor::fsn``).

**Args**

- `yi` — per-study effect estimates.
- `vi` — per-study sampling variances (or ``sei`` standard errors).
- `type` — "Rosenthal" (default), "Rosenberg", "Orwin" or "Binomial".
- `alpha` — significance level (default 0.05).
- `target` — target effect size for Orwin (default est/2).

## `FsnResult`

```python
FsnResult(type: 'str', k: 'int', fsnum: 'float', pval: 'float', est: 'float | None', target: 'float | None', alpha: 'float') -> None
```

Fail-safe N result.

## `gosh`

```python
gosh(y, v, X=None, method: 'str' = 'REML', tau2: 'float | None' = None, subsets: 'float | None' = None, seed: 'int | None' = None) -> 'GoshResult'
```

Graphical display of study heterogeneity (metafor::gosh).

**Args**

- `y, v` — analysis-scale effect and sampling variance.
- `X` — moderator design matrix (n, m) WITHOUT intercept, or None for an intercept-only model.
- `method` — "FE", "DL", "ML" or "REML" (default REML).
- `tau2` — fixed tau^2 value (equivalent to ``rma(..., tau2=...)``); when given, tau^2 is not re-estimated for each subset.
- `subsets` — number of subsets to fit. If None, all possible subsets are fitted when that number is <= 1e6, otherwise 1e6 random subsets.
- `seed` — RNG seed for the random-subsets path (reproduces R's ``seed``).

**Returns**

GoshResult with one row per subset, sorted by subset size ``k``.

## `GoshResult`

```python
GoshResult(k: 'list[int]', QE: 'list[float]', I2: 'list[float]', H2: 'list[float]', tau2: 'list[float]', tau: 'list[float]', estimate: 'list[float]', beta: 'list[list[float]]', incl: 'list[list[bool]]', k_total: 'int', p: 'int', int_only: 'bool', method: 'str') -> None
```

Results of a GOSH analysis (one row per fitted subset).

## `metainf`

```python
metainf(TE, seTE, studlab=None, sm: 'str' = '', method_tau: 'str' = 'DL', pooled: 'str' = 'random', level: 'float' = 0.95) -> 'MetainfResult'
```

*Mirrors R's `metainf`.*

Leave-one-out influence analysis (``meta::metainf``).

**Args**

- `TE, seTE` — analysis-scale effects and SEs.
- `studlab` — study labels.
- `sm` — summary measure (recorded only).
- `method_tau` — tau^2 estimator (default DL).
- `pooled` — ``"random"`` (default) or ``"common"``.
- `level` — confidence level.

## `MetainfResult`

```python
MetainfResult(studlab: 'list[str]', TE: 'list[float]', seTE: 'list[float]', lower: 'list[float]', upper: 'list[float]', statistic: 'list[float]', p: 'list[float]', I2: 'list[float]', tau2: 'list[float]', Q: 'list[float]', df: 'list[int]', k: 'int', pooled: 'str') -> None
```

*Mirrors R's `metainf`.*

``meta::metainf`` — one row per study (that study left out).

## `metacum`

```python
metacum(TE, seTE, studlab=None, sm: 'str' = '', method_tau: 'str' = 'DL', pooled: 'str' = 'random', sortvar=None, level: 'float' = 0.95) -> 'MetacumResult'
```

*Mirrors R's `metacum`.*

Cumulative meta-analysis (``meta::metacum``).

**Args**

- `TE, seTE` — analysis-scale effects and SEs.
- `studlab` — study labels.
- `sm` — summary measure (recorded only).
- `method_tau` — tau^2 estimator (default DL).
- `pooled` — ``"random"`` (default) or ``"common"``.
- `sortvar` — optional numeric vector to sort the studies before cumulating (e.g. publication year).
- `level` — confidence level.

## `MetacumResult`

```python
MetacumResult(studlab: 'list[str]', TE: 'list[float]', seTE: 'list[float]', lower: 'list[float]', upper: 'list[float]', statistic: 'list[float]', p: 'list[float]', I2: 'list[float]', tau2: 'list[float]', Q: 'list[float]', df: 'list[int]', k: 'int', pooled: 'str') -> None
```

*Mirrors R's `metacum`.*

``meta::metacum`` — one row per cumulative prefix of studies.

## `regtest`

```python
regtest(y, v, ni=None, predictor: 'str' = 'sei', model: 'str' = 'rma', method: 'str' = 'REML', test: 'str' = 'z', alpha: 'float' = 0.05) -> 'RegtestResult'
```

Regression test for funnel-plot asymmetry (metafor::regtest).

Fits ``y ~ 1 + predictor`` (a meta-regression, or a weighted linear model
when ``model="lm"``) and tests the slope. ``predictor="sei"`` is Egger's
test; ``predictor="ninv"`` (1/n) is Peters' test; ``predictor="vi"`` is
the additive variant. When ``test="knha"`` the coefficient test uses the
Knapp-Hartung adjustment (recommended for Peters' test).

## `RegtestResult`

```python
RegtestResult(predictor: 'str', stat: 'float', p: 'float', model: 'str', test: 'str', method: 'str', intercept: 'float | None', intercept_ci_lower: 'float | None', intercept_ci_upper: 'float | None', k: 'int', slope: 'float | None' = None, slope_se: 'float | None' = None, intercept_se: 'float | None' = None, tau: 'float | None' = None, df: 'int | None' = None) -> None
```

metafor::regtest output (Egger / Peters / other predictors).

## `metabias`

```python
metabias(x, seTE=None, method_bias=None, plotit=False, correct=False, k_min=10, n_e=None, n_c=None, event_e=None, event_c=None, sm=None, method_tau=None, alpha=0.05)
```

*Mirrors R's `metabias`.*

Test for funnel-plot asymmetry (``meta::metabias``).

All nine methods meta supports are available:

===============  ========================================================
``Begg``         rank correlation of the standardised effect sizes with
                 their variances (``kentau``, with meta's tie corrections)
``Egger``        regression of the effect on its standard error (``lm``)
``Thompson``     the same regression fitted as a random-effects model
                 (additive; uses the analysis' tau^2 estimator)
``Schwarzer``    hypergeometric variant of the rank test (binary data)
``Harbord``      score test built from the 2x2 tables (odds-ratio scale)
``Peters``       regression on ``1/n`` with the Peters variances
``Deeks``        regression on ``1/sqrt(ESS)`` for diagnostic odds ratios
``Pustejovsky``  regression on ``sqrt(1/n_e + 1/n_c)``
``Macaskill``    regression weighted by the total sample size
===============  ========================================================

**Args**

- `x` — a result object (``metabin`` / ``metagen`` / ...) or, together with ``seTE``, the effect sizes themselves.
- `seTE` — standard errors when ``x`` holds the effect sizes.
- `method_bias` — one of the names above (prefix-matched; the pre-8.x aliases ``rank``/``linreg``/``mm``/``count``/``score`` also work). Defaults to ``"Harbord"`` for a binary odds-ratio analysis and ``"Egger"`` otherwise, matching meta.
- `plotit` — accepted for interface parity — draw the radial plot yourself with :func:`pimeta.plotting.radial`.
- `correct` — continuity correction for the rank tests.
- `k_min` — meta requires ``k >= k_min`` and ``k >= 3``; otherwise only ``k`` and ``k_min`` are reported, exactly like R.
- `n_e, n_c, event_e, event_c, sm` — override the values taken from ``x``.
- `method_tau` — tau^2 estimator used by the Thompson variant (defaults to the analysis' own estimator, i.e. R's ``x$method.tau``).
- `alpha` — significance level used for the regression CIs.

**Returns**

- `:class` — `MetabiasResult`.

## `MetabiasResult`

```python
MetabiasResult(method_bias: 'str', method: 'str' = '', var_model: 'str' = '', statistic: 'float | None' = None, pval: 'float | None' = None, p_value: 'float | None' = None, estimate: 'list[float] | None' = None, intercept: 'float | None' = None, se_intercept: 'float | None' = None, tau: 'float | None' = None, df: 'int | None' = None, k: 'int | None' = None, k_min: 'float' = 10) -> None
```

*Mirrors R's `metabias`.*

Result of :func:`metabias` (``meta::metabias``).

``statistic`` is the z of the rank tests and the t of the regression
tests; ``estimate`` holds ``[bias, se.bias]`` — the slope of the
regression of the effect on the chosen predictor — and ``intercept`` /
``se_intercept`` the "limit estimate".  When ``k < k_min`` (or ``k < 3``)
meta reports only ``k`` and ``k_min``; the remaining fields are ``None``.

## `peters_test`

```python
peters_test(y, v, ni, test: 'str' = 'knha', method: 'str' = 'REML') -> 'RegtestResult'
```

Peters' regression test: ``y ~ 1 + 1/n`` (metafor::regtest, ninv).

## `baujat`

```python
baujat(y, v, method: 'str' = 'REML', tau2: 'float | None' = None) -> 'BaujatResult'
```

Baujat diagnostic coordinates for an intercept-only model.

``x`` = ``(yi - mu)^2 / (tau^2 + vi)`` (squared Pearson residual);
``y`` = ``(mu_{-i} - mu)^2 / var(mu_{-i})`` (standardized influence of the
leave-one-out refit on the pooled estimate), exactly as metafor::baujat.

## `BaujatResult`

```python
BaujatResult(x: 'list[float]', y: 'list[float]', k: 'int') -> None
```

Baujat plot coordinates (metafor::baujat.rma).
