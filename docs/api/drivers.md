# High-level drivers

*31 public names, generated from `pimeta.__all__` and the docstrings in `src/pimeta/`. Do not edit this page by hand — run `python tools/gen_api_docs.py`.*

| Name | Mirrors in R |
|---|---|
| [`DosresResult`](#dosresresult) | — |
| [`dose_cov`](#dose_cov) | — |
| [`dosres`](#dosres) | — |
| [`greenland_counts`](#greenland_counts) | — |
| [`metabin`](#metabin) | metabin |
| [`MetabResult`](#metabresult) | — |
| [`metacont`](#metacont) | metacont |
| [`MetacontResult`](#metacontresult) | — |
| [`metareg`](#metareg) | rma(yi, vi, mods=X) |
| [`MetaregResult`](#metaregresult) | — |
| [`metacor`](#metacor) | metacor |
| [`MetacorResult`](#metacorresult) | — |
| [`metainc`](#metainc) | metainc |
| [`MetaincResult`](#metaincresult) | — |
| [`metamean`](#metamean) | metamean |
| [`MetameanResult`](#metameanresult) | — |
| [`metagen`](#metagen) | metagen |
| [`MetagenResult`](#metagenresult) | metagen |
| [`transf`](#transf) | — |
| [`backtransf`](#backtransf) | — |
| [`metaprop`](#metaprop) | metaprop |
| [`MetapropResult`](#metapropresult) | — |
| [`metarate`](#metarate) | metarate |
| [`MetarateResult`](#metarateresult) | — |
| [`metabind`](#metabind) | metabind |
| [`metaadd`](#metaadd) | metaadd |
| [`MetaaddResult`](#metaaddresult) | metaadd |
| [`metamerge`](#metamerge) | metamerge |
| [`MetabindResult`](#metabindresult) | — |
| [`metacr`](#metacr) | metacr |
| [`MetacrResult`](#metacrresult) | metacr |

## `DosresResult`

```python
DosresResult(b: 'list[float]', se: 'list[float]', zval: 'list[float]', pval: 'list[float]', ci_lb: 'list[float]', ci_ub: 'list[float]', vb: 'np.ndarray', coefnames: 'list[str]', tau2: 'float', QE: 'float', QE_df: 'int', QE_p: 'float', k: 'int', studies: 'int', method: 'str', level: 'float', basis: 'Basis | None', covariance: 'str', test: 'str', fit: 'Any', dose: 'np.ndarray' = <factory>) -> None
```

A pooled dose-response curve.

**Attributes**

- `b` — fixed-effect coefficients of the dose basis (intercept first).
- `se, zval, pval, ci_lb, ci_ub` — one entry per coefficient.
- `vb` — their covariance matrix.
- `coefnames` — names, e.g. ``["intrcpt", "dose[1]", "dose[2]"]``.
- `tau2` — between-study variance of the random intercept.
- `QE, QE_df, QE_p` — residual heterogeneity.
- `k` — number of contrasts fitted (reference levels excluded).
- `studies` — number of studies.
- `method` — τ² estimator.
- `level` — confidence level.
- `basis` — the `basis.Basis` that was fitted.
- `covariance` — ``"greenland"``, ``"constant"`` or ``"independent"``.
- `test` — reference distribution of the coefficient tests.
- `fit` — the underlying `rma_mv.RmaMvResult`.
- `dose` — the doses that were modelled, and ``study`` their study labels.

## `dose_cov`

```python
dose_cov(dose, events=None, totals=None, *, effects=None, variances=None, type: 'str' = 'cc', rho: 'float | None' = None, tol: 'float' = 1e-12, max_iter: 'int' = 500) -> 'np.ndarray'
```

Within-study covariance of the log contrasts of one dose-response study.

**Args**

- `dose` — the dose levels, ascending or not; used for ordering only.
- `events` — cases/events per level; required for the Greenland correction.
- `totals` — participants or person-time per level.
- `effects` — log relative effect per level (needed by the correction).
- `variances` — variance per level; zero/``NaN`` marks the reference level.
- `type` — ``"cc"``, ``"ci"`` or ``"ir"``.
- `rho` — use a constant within-study correlation instead of the correction. The covariance is then ``rho * sqrt(vi_i * vi_j)`` and `variances` must be given.
- `tol, max_iter` — passed to `greenland_counts`.

**Returns**

``(m, m)`` covariance matrix over the **non-reference** levels, in the input order.

**Raises**

- `ValueError` — when neither route is available, or the inputs disagree.

**Example**

>>> cov = ma.dose_cov(dose, events, totals, effects=logrr, variances=v) >>> cov.shape (3, 3)

## `dosres`

```python
dosres(yi, *, study, dose, vi=None, cases=None, n=None, type: 'str' = 'cc', basis: 'Basis | None' = None, knots=None, nk: 'int | None' = 3, df: 'int | None' = None, rho: 'float | None' = None, method: 'str' = 'REML', level: 'float' = 0.95, tol: 'float' = 1e-12) -> 'DosresResult'
```

Pool a dose-response curve across studies.

Two-stage in spirit and one-stage in form: each study contributes its own
within-study covariance block, and the studies are pooled by a random
intercept. That is the model the dose-response literature calls one-stage;
the alternative — fit each study and pool the coefficients — is available by
fitting the basis per study and handing the coefficients to ``rma_mv``.

**Args**

- `yi` — log relative effect per contrast.
- `study` — study identifier per row.
- `dose` — dose per row.
- `vi` — variance of ``yi``. Rows with zero or missing ``vi`` are the reference levels: they are used to build the covariance and then excluded from the fit, exactly as a published table's reference row is.
- `cases` — cases per dose level, for the Greenland correction.
- `n` — participants or person-time per dose level.
- `type` — ``"cc"``, ``"ci"`` or ``"ir"`` — see `dose_cov`.
- `basis` — a `basis.Basis` over dose. When omitted one is built by `basis.rcs` from ``knots``/``nk``/``df``.
- `knots, nk, df` — passed to `basis.rcs` when ``basis`` is not given.
- `rho` — constant within-study correlation, used instead of the Greenland correction.
- `method` — τ² estimator (``"REML"`` by default).
- `level` — confidence level.
- `tol` — Greenland convergence tolerance.

**Returns**

`DosresResult`.

**Raises**

- `ValueError` — when the rows do not line up, when a study has fewer than two levels, or when no covariance route is available.

**Example**

>>> fit = ma.dosres(logrr, study=id, dose=dose, vi=se**2, ...                 cases=cases, n=n, knots=[5, 15, 30])

## `greenland_counts`

```python
greenland_counts(events, totals, effects, variances=None, *, type: 'str' = 'cc', tol: 'float' = 1e-12, max_iter: 'int' = 500) -> 'np.ndarray'
```

Greenland–Longnecker adjusted case counts for one study.

A published table reports relative risks to two or three decimals, so the
reported reference count and the reported contrasts are not exactly
consistent. Greenland & Longnecker (1992) solve for the counts that *are*
consistent by iterating

.. math::

    A^{(m+1)} = A^{(m)} + H^{-1}\big[\, y + \log(A_0)
                + \log(n_i - A_i) - \log(A_i) - \log(n_0 - A_0)\big]

where :math:`H` is the covariance matrix implied by those counts. The
result is used to build the covariance and nothing else — the fitted
effects stay the ones the paper reported.

**Args**

- `events` — cases (``cc``/``ci``) or events (``ir``) per dose level.
- `totals` — total participants (``cc``/``ci``) or person-time (``ir``).
- `effects` — log relative effect per dose level. The reference level(s) may be given as ``0``.
- `variances` — variance of each log effect. ``0``, ``None`` or ``NaN`` marks the reference level, whose variance is taken to be zero.
- `type` — ``"cc"`` (case-control), ``"ci"`` (cumulative incidence) or ``"ir"`` (incidence rate).
- `tol` — convergence tolerance on the squared change in the adjusted counts. The default is tighter than ``dosresmeta``'s 1e-5, which leaves the counts good to about five decimals; the fixed point is the same.
- `max_iter` — iteration cap.

**Returns**

Adjusted counts, one per dose level, in the input order.

**Raises**

- `ValueError` — for fewer than two dose levels, one reference level, or a negative/zero count.

## `metabin`

```python
metabin(event_e, n_e, event_c, n_c, sm: 'str' = 'RR', method: 'str' = 'Inverse', random: 'bool' = True, method_tau: 'str' = 'REML', alpha: 'float' = 0.05, data=None) -> 'MetabResult'
```

*Mirrors R's `metabin`.*

Pool binary (2x2) study data (mirrors ``meta::metabin``).

**Args**

- `event_e, n_e` — treatment-arm events and group size (per study).
- `event_c, n_c` — control-arm events and group size (per study).
- `sm` — summary measure, one of RR/OR/RD/AS.
- `method` — "Inverse", "MH", "Peto" or "LRP" (penalised logistic regression, Firth/Jeffreys; ``sm="OR"`` only).
- `random` — also fit the random-effects model (method_tau tau^2 estimator). For ``method="LRP"`` the "random" result is the common estimate with its SE multiplied by the dispersion factor ``phi``.
- `method_tau` — DL/HE/HS/SJ/ML/REML/PM/EB (default DL, matching meta). Unused by "LRP", which has no tau^2 (``tau2`` is NaN, as in R).
- `data` — optional frame; the four column arguments may then be names.

## `MetabResult`

```python
MetabResult(sm: 'str', method: 'str', k: 'int', fixed: 'PooledResult', random: 'PooledResult | None', yi: 'list[float]', vi: 'list[float]', phi: 'float' = 1.0, method_tau: 'str' = 'REML', event_e: 'list[float] | None' = None, n_e: 'list[float] | None' = None, event_c: 'list[float] | None' = None, n_c: 'list[float] | None' = None) -> None
```

Binary meta-analysis result: fixed + random pooled effects.

## `metacont`

```python
metacont(n_e, mean_e, sd_e, n_c, mean_c, sd_c, sm: 'str' = 'MD', random: 'bool' = True, method_tau: 'str' = 'REML', alpha: 'float' = 0.05, data=None) -> 'MetacontResult'
```

*Mirrors R's `metacont`.*

Pool continuous (mean/SD) study data (mirrors ``meta::metacont``).

**Args**

- `n_e, mean_e, sd_e` — treatment-arm size, mean, SD (per study).
- `n_c, mean_c, sd_c` — control-arm size, mean, SD (per study).
- `sm` — "MD" (mean difference) or "SMD"/"SMDH" (standardised mean difference, Hedges' g by default like meta::metacont).
- `random` — also fit the random-effects model.
- `method_tau` — tau-squared estimator (default DL).
- `data` — optional frame; the six column arguments may then be names.

## `MetacontResult`

```python
MetacontResult(sm: 'str', k: 'int', fixed: 'PooledResult', random: 'PooledResult | None', yi: 'list[float]', vi: 'list[float]', n_e: 'list[float] | None' = None, n_c: 'list[float] | None' = None, method_tau: 'str' = 'REML') -> None
```

Continuous meta-analysis result: fixed + random pooled effects.

## `metareg`

```python
metareg(y, v, X, X_names=None, method: 'str' = 'REML', alpha: 'float' = 0.05, test: 'str' = 'z') -> 'MetaregResult'
```

*Mirrors R's `rma(yi, vi, mods=X)`.*

Fit a meta-regression (mirrors ``metafor::rma(yi, vi, mods=X)``).

**Args**

- `y, v` — analysis-scale effect and sampling variance (log scale for ratio measures).
- `X` — design matrix (n, p) WITHOUT an intercept column (one is added automatically), or (n, 1) single-moderator array.
- `X_names` — moderator names (len p).
- `method` — "FE" (fixed), "DL", "ML" or "REML" (default REML, metafor default).
- `test` — "z" (default), "t", "knha" (Knapp-Hartung / HKSJ) or "adhoc" (Sidik-Jonkman ad-hoc). Non-z tests use a t distribution with ``k-p`` df; "knha"/"adhoc" additionally scale the variance by ``s2w = RSS.knha/(k-p)`` (metafor's scalar implementation).

## `MetaregResult`

```python
MetaregResult(b: 'list[float]', se: 'list[float]', z: 'list[float]', p: 'list[float]', ci_lower: 'list[float]', ci_upper: 'list[float]', names: 'list[str]' = <factory>, tau2: 'float' = 0.0, method: 'str' = 'REML', k: 'int' = 0, p_mods: 'int' = 0, QM: 'float' = 0.0, QM_df: 'int' = 0, QM_p: 'float' = nan, QE: 'float' = 0.0, QE_df: 'int' = 0, QE_p: 'float' = nan, I2: 'float' = 0.0, H2: 'float' = 0.0, R2: 'float' = 0.0) -> None
```

Meta-regression coefficients + heterogeneity / model diagnostics.

## `metacor`

```python
metacor(cor, n, sm: 'str' = 'ZCOR', random: 'bool' = True, method_tau: 'str' = 'REML', alpha: 'float' = 0.05) -> 'MetacorResult'
```

*Mirrors R's `metacor`.*

Pool correlations (mirrors ``meta::metacor``).

**Args**

- `cor` — correlation coefficient per study.
- `n` — sample size per study.
- `sm` — "ZCOR" (Fisher's z, default) or "COR" (raw).
- `random` — also fit the random-effects model.
- `method_tau` — tau-squared estimator (default DL).

## `MetacorResult`

```python
MetacorResult(sm: 'str', k: 'int', fixed_cor: 'float', fixed_lower: 'float', fixed_upper: 'float', random_cor: 'float | None', random_lower: 'float | None', random_upper: 'float | None', yi: 'list[float]', vi: 'list[float]', fixed: 'PooledResult', random: 'PooledResult | None') -> None
```

Correlation meta-analysis result (fixed + random, correlation scale).

## `metainc`

```python
metainc(event_e, time_e, event_c, time_c, sm: 'str' = 'IRR', random: 'bool' = True, method: 'str' = 'MH', method_tau: 'str' = 'REML', incr: 'float' = 0.5, alpha: 'float' = 0.05) -> 'MetaincResult'
```

*Mirrors R's `metainc`.*

Pool incidence rates (mirrors ``meta::metainc``).

**Args**

- `event_e, time_e` — treatment-arm event count and person-time per study.
- `event_c, time_c` — control-arm event count and person-time per study.
- `sm` — "IRR" (rate ratio, default), "IRD" (rate difference) or "IRSD".
- `random` — also fit the random-effects model.
- `method` — common-effect method "MH" (default), "Cochran" or "Inverse".
- `method_tau` — tau-squared estimator (default DL).
- `incr` — continuity correction added when any study has zero events.

## `MetaincResult`

```python
MetaincResult(sm: 'str', method: 'str', k: 'int', fixed_effect: 'float', fixed_lower: 'float', fixed_upper: 'float', random_effect: 'float | None', random_lower: 'float | None', random_upper: 'float | None', fixed_rate: 'float', fixed_rate_lower: 'float', fixed_rate_upper: 'float', random_rate: 'float | None', random_rate_lower: 'float | None', random_rate_upper: 'float | None', yi: 'list[float]', vi: 'list[float]', fixed: 'PooledResult', random: 'PooledResult | None') -> None
```

Incidence-rate meta-analysis result (fixed + random).

``fixed_effect`` / ``random_effect`` are on the *analysis* scale (log for
IRR), matching ``meta``'s ``$TE.common`` / ``$TE.random``. The
back-transformed rate-scale values are exposed as ``fixed_rate`` /
``random_rate`` (what ``meta`` shows in printouts when ``backtransf=TRUE``).

## `metamean`

```python
metamean(n, mean, sd, sm: 'str' = 'MRAW', random: 'bool' = True, method_tau: 'str' = 'REML', alpha: 'float' = 0.05) -> 'MetameanResult'
```

*Mirrors R's `metamean`.*

Pool single-arm means (mirrors ``meta::metamean``).

**Args**

- `n` — sample size per study.
- `mean` — mean per study.
- `sd` — standard deviation per study.
- `sm` — "MRAW" (default) or "MLN" (log scale).
- `random` — also fit the random-effects model.
- `method_tau` — tau-squared estimator (default DL).

## `MetameanResult`

```python
MetameanResult(sm: 'str', k: 'int', fixed_mean: 'float', fixed_lower: 'float', fixed_upper: 'float', random_mean: 'float | None', random_lower: 'float | None', random_upper: 'float | None', yi: 'list[float]', vi: 'list[float]', fixed: 'PooledResult', random: 'PooledResult | None') -> None
```

Single-arm mean meta-analysis result (fixed + random, mean scale).

## `metagen`

```python
metagen(TE, seTE, studlab=None, sm: 'str' = '', method_tau: 'str' = 'REML', method_random_ci: 'str' = 'classic', method_predict: 'str' = 'V', common: 'bool' = True, random: 'bool' = True, prediction: 'bool' = False, level: 'float' = 0.95, level_predict: 'float' = 0.95, null_effect: 'float' = 0.0, exclude=None, warn_missing: 'bool' = True) -> 'MetagenResult'
```

*Mirrors R's `metagen`.*

Generic inverse-variance meta-analysis (mirrors ``meta::metagen``).

**Args**

- `TE` — treatment estimates, already on the analysis scale (e.g. ``lnRR``).
- `seTE` — standard errors (analysis scale).
- `studlab` — optional study labels.
- `sm` — summary measure (``""``, ``"RR"``, ``"OR"``, ...); only used for report-scale back-transformation and validation.
- `method_tau` — tau-squared estimator (``"REML"`` default, ``"DL"``, ``"ML"``, ``"HE"``, ``"HS"``, ``"SJ"``, ``"PM"``, ``"EB"``).
- `method_random_ci` — random-effects CI method (``"classic"`` default).
- `method_predict` — prediction-interval method (``"V"`` default).
- `common` — compute the common (fixed) effect model.
- `random` — compute the random-effects model.
- `prediction` — compute the prediction interval.
- `level` — confidence level for the pooled estimates.
- `level_predict` — confidence level for the prediction interval.
- `null_effect` — null hypothesis value on the analysis scale (default 0).

Returns a `MetagenResult` whose numeric fields are on the analysis
scale, exactly matching R's ``$TE.common``, ``$lower.random``, etc.

## `MetagenResult`

```python
MetagenResult(sm: 'str', null_effect: 'float', TE: 'list[float]', seTE: 'list[float]', statistic: 'list[float]', pval: 'list[float]', lower: 'list[float]', upper: 'list[float]', k: 'int', k_all: 'int', k_TE: 'int', w_common: 'list[float]', TE_common: 'float', seTE_common: 'float', statistic_common: 'float', pval_common: 'float', lower_common: 'float', upper_common: 'float', w_random: 'list[float]', TE_random: 'float', seTE_random: 'float', seTE_classic: 'float', statistic_random: 'float', pval_random: 'float', df_random: 'float', lower_random: 'float', upper_random: 'float', Q: 'float', df_Q: 'int', pval_Q: 'float', method_tau: 'str', tau2: 'float', tau: 'float', H: 'float', lower_H: 'float', upper_H: 'float', I2: 'float', lower_I2: 'float', upper_I2: 'float', method_predict: 'str', seTE_predict: 'float', df_predict: 'float', lower_predict: 'float', upper_predict: 'float', studlab: 'list[str]' = <factory>) -> None
```

*Mirrors R's `metagen`.*

``meta::metagen`` result (all numeric fields on the analysis scale).

## `transf`

```python
transf(x, sm: 'str')
```

meta::transf — transform a value to the analysis scale.

``x`` may be a scalar or an array; a NumPy array is returned.

## `backtransf`

```python
backtransf(x, sm: 'str', n=None, time=None)
```

meta::backtransf — back-transform an analysis-scale value to report scale.

``n`` is required for ``sm="PFT"`` and ``time`` for ``sm="IRFT"``. Returns
a NumPy array (or scalar-convertible value for scalar input).

## `metaprop`

```python
metaprop(event, n, sm: 'str' = 'PLOGIT', random: 'bool' = True, method_tau: 'str' = 'REML', incr: 'float' = 0.5, alpha: 'float' = 0.05) -> 'MetapropResult'
```

*Mirrors R's `metaprop`.*

Pool single-arm proportions (mirrors ``meta::metaprop``).

**Args**

- `event` — number of events per study.
- `n` — sample size per study.
- `sm` — "PFT" (Freeman-Tukey double arcsine, default), "PAS", "PLN", "PLOGIT" or "PRAW".
- `random` — also fit the random-effects model.
- `method_tau` — tau-squared estimator (default DL).
- `incr` — continuity correction for PLN/PLOGIT/PRAW (default 0.5).

## `MetapropResult`

```python
MetapropResult(sm: 'str', k: 'int', n: 'list[float]', fixed_prop: 'float', fixed_lower: 'float', fixed_upper: 'float', random_prop: 'float | None', random_lower: 'float | None', random_upper: 'float | None', yi: 'list[float]', vi: 'list[float]', fixed: 'PooledResult', random: 'PooledResult | None', event: 'list[float] | None' = None, method_tau: 'str' = 'REML') -> None
```

Proportion meta-analysis result (fixed + random, proportion scale).

## `metarate`

```python
metarate(event, time, sm: 'str' = 'IRLN', random: 'bool' = True, method_tau: 'str' = 'REML', incr: 'float' = 0.5, method_incr: 'str' = 'only0', alpha: 'float' = 0.05) -> 'MetarateResult'
```

*Mirrors R's `metarate`.*

Pool single-arm incidence rates (mirrors ``meta::metarate``).

**Args**

- `event` — number of events per study.
- `time` — person-time at risk per study.
- `sm` — "IRLN" (log rate, default), "IR", "IRS" or "IRFT".
- `random` — also fit the random-effects model.
- `method_tau` — tau-squared estimator (default REML).
- `incr` — continuity correction added to zero-event studies (default 0.5).
- `method_incr` — "only0" (default), "all", "if0all" or "user" (in which case ``incr`` may be a per-study vector).

## `MetarateResult`

```python
MetarateResult(sm: 'str', k: 'int', event: 'list[float]', time: 'list[float]', incr_event: 'list[float]', yi: 'list[float]', vi: 'list[float]', fixed_rate: 'float', fixed_lower: 'float', fixed_upper: 'float', random_rate: 'float | None', random_lower: 'float | None', random_upper: 'float | None', i2: 'float', q: 'float', q_df: 'int', q_p: 'float', tau2: 'float | None', fixed: 'PooledResult', random: 'PooledResult | None') -> None
```

Incidence-rate meta-analysis result (fixed + random, rate scale).

## `metabind`

```python
metabind(*results, method_tau='REML', sm=None, studlab=None, level=0.95)
```

*Mirrors R's `metabind`.*

``meta::metabind`` — combine several meta-analyses into a single one.

The study-level estimates of all inputs are concatenated and re-pooled with
`.metagen.metagen`, which is what ``meta::metabind`` does (it keeps
the individual study rows rather than the summary estimates).

**Args**

- `*results` — meta-analysis result objects exposing ``TE``/``seTE``.
- `method_tau` — ``tau^2`` estimator passed to `metagen`.
- `sm` — effect measure; defaults to the ``sm`` of the first input.
- `studlab` — optional study labels (one per input study).
- `level` — confidence level (0.95).

**Returns**

`MetabindResult`.

**Raises**

- `ValueError` — when no inputs are given or an input lacks study-level data.

## `metaadd`

```python
metaadd(result, type=None, TE=None, lower=None, upper=None, statistic=None, pval=None, df=None, se=None, method='', method_ci='', text='Added result', data=None)
```

*Mirrors R's `metaadd`.*

``meta::metaadd`` — append pooled results to a meta-analysis.

Either pass another meta-analysis as ``data`` (its common-effect,
random-effects, tau^2 and prediction results are appended) or supply the
values directly through ``type``/``TE``/``lower``/``upper``/``se``.

**Args**

- `result` — a `.metagen.MetagenResult`.
- `type` — ``"common"``, ``"random"``, ``"prediction"`` or ``"tau2"``.
- `TE, lower, upper, statistic, pval, df, se` — the result to append.
- `method, method_ci` — labels for the estimation method.
- `text` — label used when printing.
- `data` — alternatively a `.metagen.MetagenResult` whose pooled results are appended.

**Returns**

`MetaaddResult`.

## `MetaaddResult`

```python
MetaaddResult(base, added, common, random, prediction)
```

*Mirrors R's `metaadd`.*

Result of `metaadd` (``meta::metaadd``).

``meta::metaadd`` appends additional *pooled* results (from another
meta-analysis or supplied directly) to a ``meta`` object.  In R those
fields are plain vectors, so the accessors here return lists whose first
element is the original value and whose remaining elements are the added
ones.

**Attributes**

- `base` — the original `.metagen.MetagenResult`.
- `added` — list of ``dict`` rows, one per appended result, with the keys ``type``, ``TE``, ``se``, ``lower``, ``upper``, ``statistic``, ``pval``, ``df``, ``method``, ``method_ci``, ``text``.
- `common, random, prediction` — whether the corresponding model is present after the additions.
- `is_metaadd` — always ``True`` (mirrors the ``"metaadd"`` class tag).

## `metamerge`

```python
metamerge(x, y, method_tau='REML', level=0.95)
```

*Mirrors R's `metamerge`.*

``meta::metamerge`` — merge two meta-analyses of the same studies.

``meta::metamerge`` is used when the *same* studies were analysed twice
(e.g. two effect measures, or a subgroup split); the two study lists are
aligned by position and pooled jointly, and the separate analyses are kept
in the returned object.

**Args**

- `x, y` — meta-analysis result objects exposing ``TE``/``seTE``.
- `method_tau` — ``tau^2`` estimator.
- `level` — confidence level.

**Returns**

`MetabindResult` whose ``meta`` is the pooled combination and which additionally carries ``x``/``y`` for the two inputs.

## `MetabindResult`

```python
MetabindResult(meta, TE, seTE, studlab, sm, parts)
```

Result of `metabind` — a combined meta-analysis.

**Attributes**

- `meta` — the pooled `.metagen.MetagenResult`.
- `TE` — concatenated study effects.
- `seTE` — concatenated standard errors.
- `studlab` — concatenated study labels.
- `sm` — effect measure (taken from the first input).
- `parts` — number of input meta-analyses that were combined.

## `metacr`

```python
metacr(x, comp_no=1, outcome_no=1, method=None, sm=None, common=True, random=True, method_tau='DL', level=0.05, level_ma=0.95, swap_events=None, prediction=False, level_predict=0.05)
```

*Mirrors R's `metacr`.*

``meta::metacr`` — refit one comparison/outcome of a review.

**Args**

- `x` — the review's data — a ``pandas.DataFrame`` or a list of dicts with the RevMan column names.
- `comp_no, outcome_no` — which comparison and outcome to analyse.
- `method` — ``"Inverse"``, ``"MH"`` or ``"Peto"`` (default: from the data).
- `sm` — effect measure (default: from the data).  ``"PETO_OR"`` is mapped to ``"OR"`` exactly like R.
- `common, random, prediction` — which models to fit.
- `method_tau` — heterogeneity estimator (R's ``metacr`` default is ``"DL"``).
- `level, level_ma` — confidence levels for the study-level and pooled intervals.
- `swap_events` — swap the event counts of the two arms.
- `level_predict` — level of the prediction interval.

**Returns**

`MetacrResult`, or ``None`` when the selection is empty.

## `MetacrResult`

```python
MetacrResult(meta, comp_no, outcome_no, type, sm, method, swap_events, n_studies)
```

*Mirrors R's `metacr`.*

Result of `metacr` (``meta::metacr``).

``meta::metacr`` operates on a RevMan 5 (``.rm5``) or Cochrane data
(``.cdir``) object, selects one comparison and outcome, and re-runs the
appropriate ``meta`` analysis.  This port takes the *data frame* that such
an object carries (the columns ``comp.no``, ``outcome.no``, ``type``,
``sm``, ``method``, ``event.e``, ``n.e``, ``event.c``, ``n.c``,
``mean.e``, ``sd.e``, ``mean.c``, ``sd.c``, ``O.E``, ``V``, ``TE``,
``seTE``) instead of parsing the RevMan file.

**Attributes**

- `meta` — the fitted analysis object.
- `comp_no, outcome_no` — the selected comparison and outcome.
- `type` — ``"D"`` (binary), ``"C"`` (continuous), ``"P"`` (generic Peto / O-E), or ``"I"`` (inverse variance).
- `sm` — effect measure used.
- `method` — ``"Inverse"``, ``"MH"`` or ``"Peto"``.
- `swap_events` — whether the 2x2 tables were swapped.
- `n_studies` — number of studies in the selected comparison.
