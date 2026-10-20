# Models

*34 public names, generated from `pimeta.__all__` and the docstrings in `src/pimeta/`. Do not edit this page by hand — run `python tools/gen_api_docs.py`.*

| Name | Mirrors in R |
|---|---|
| [`StudyEffect`](#studyeffect) | — |
| [`PooledResult`](#pooledresult) | — |
| [`run_meta`](#run_meta) | rma |
| [`pool_yv`](#pool_yv) | — |
| [`result_to_dict`](#result_to_dict) | — |
| [`confint_tau2`](#confint_tau2) | confint.rma.uni |
| [`Tau2Confint`](#tau2confint) | confint |
| [`predict_rma`](#predict_rma) | predict.rma |
| [`PredictResult`](#predictresult) | predict.rma |
| [`hc_rma`](#hc_rma) | hc.rma.uni |
| [`HcResult`](#hcresult) | hc.rma.uni |
| [`estimate_tau2`](#estimate_tau2) | — |
| [`rma_glmm`](#rma_glmm) | — |
| [`RmaGlmmResult`](#rmaglmmresult) | rma.glmm |
| [`rma_mv`](#rma_mv) | rma.mv |
| [`RmaMvResult`](#rmamvresult) | rma.mv |
| [`anova_rma`](#anova_rma) | — |
| [`ranef`](#ranef) | — |
| [`robust`](#robust) | robust |
| [`rma_peto`](#rma_peto) | rma.peto |
| [`RmaPetoResult`](#rmapetoresult) | rma.peto |
| [`rma_mh`](#rma_mh) | rma.mh |
| [`RmaMhResult`](#rmamhresult) | rma.mh |
| [`rma`](#rma) | rma |
| [`RmaResult`](#rmaresult) | rma.uni |
| [`selmodel`](#selmodel) | — |
| [`SelModelResult`](#selmodelresult) | rma.uni.selmodel |
| [`SelProfileResult`](#selprofileresult) | profile.rma.uni.selmodel |
| [`profile_selmodel`](#profile_selmodel) | — |
| [`rma_ls`](#rma_ls) | — |
| [`RmaLsResult`](#rmalsresult) | rma.ls |
| [`profile_rma_ls`](#profile_rma_ls) | profile.rma.ls |
| [`RmaLsProfileResult`](#rmalsprofileresult) | profile.rma.ls |
| [`confint_rma_ls`](#confint_rma_ls) | confint.rma.ls |

## `StudyEffect`

```python
StudyEffect(effect_size: 'float', ci_lower: 'float | None' = None, ci_upper: 'float | None' = None, standard_error: 'float | None' = None, variance: 'float | None' = None, n: 'float | None' = None, study_id: 'str' = '') -> None
```

One study's effect estimate + dispersion (SE/variance/CI).

## `PooledResult`

```python
PooledResult(model: 'str', method: 'str', k: 'int', effect_type: 'str', pooled_effect: 'float', pooled_ci_lower: 'float', pooled_ci_upper: 'float', pooled_est: 'float', pooled_se: 'float', z: 'float', p_value: 'float', q: 'float', q_df: 'int', q_p: 'float', tau2: 'float', i2: 'float', h2: 'float', weights: 'list[float]' = <factory>, pi_lower: 'float | None' = None, pi_upper: 'float | None' = None) -> None
```

Pooled estimate + heterogeneity diagnostics (R-identical numbers).

## `run_meta`

```python
run_meta(effects: 'list[StudyEffect]', effect_type: 'str', model: 'str' = 'random', method: 'str' = 'DL', alpha: 'float' = 0.05) -> 'PooledResult'
```

*Mirrors R's `rma`.*

Pool a set of study effects (mirrors ``rma`` / ``metagen``).

**Args**

- `effects` — per-study effect estimates with SE/variance/CI.
- `effect_type` — one of RR/OR/HR/PETO/IRR/PLN/PLO (log scale) or MD/SMD/WMD/RD/AS/PAS/PFT (raw scale).
- `model` — "fixed" or "random".
- `method` — tau-squared estimator, one of DL/HE/HS/SJ/ML/REML/PM/EB (meta/metafor default is DL). Ignored for fixed-effect.
- `alpha` — significance level for CI/PI (0.05 -> 95%).

## `pool_yv`

```python
pool_yv(y: 'list[float]', v: 'list[float]', n: 'list[float] | None', effect_type: 'str', model: 'str' = 'random', method: 'str' = 'DL', alpha: 'float' = 0.05) -> 'PooledResult'
```

Pool analysis-scale ``(y, v)`` arrays directly.

This is the shared numeric core behind :func:`run_meta` and the
binary-data wrappers (metabin). ``y`` must already be on the analysis
scale (log scale for ratio types).

## `result_to_dict`

```python
result_to_dict(result: 'PooledResult') -> 'dict'
```

Serialize a PooledResult to a plain ``dict`` of Python scalars.

## `confint_tau2`

```python
confint_tau2(y: 'ArrayLike', v: 'ArrayLike', tau2: 'float', method: 'str' = 'REML', level: 'float' = 0.05, type: 'str' = 'QP', X: 'np.ndarray | None' = None) -> 'Tau2Confint'
```

*Mirrors R's `confint.rma.uni`.*

CI for tau^2 (transcribed from ``metafor::confint.rma.uni``).

**Args**

- `y` — analysis-scale effects.
- `v` — analysis-scale sampling variances.
- `tau2` — the estimated tau^2 (from the fitted model).
- `method` — "ML" or "REML" (used only by the "PL" profile-likelihood interval, which refits at fixed tau^2 with the same method).
- `level` — **alpha** level (0.05 -> 95% CI), matching metafor's internal ``x$level`` convention.
- `type` — "QP" (Q-profile, default), "PL" (profile-likelihood), or "HT" (Higgins-Thompson).
- `X` — the fitted design matrix (``k x p``, intercept included).  Omitted for an intercept-only model.  metafor refuses ``type="HT"`` when the model has moderators, and so do we.

## `Tau2Confint`

```python
Tau2Confint(type: 'str', tau2: 'float', tau2_lb: 'float', tau2_ub: 'float', tau_lb: 'float | None', tau_ub: 'float | None', i2: 'float', i2_lb: 'float', i2_ub: 'float', h2: 'float', h2_lb: 'float', h2_ub: 'float', lb_sign: 'str' = '', ub_sign: 'str' = '', lb_conv: 'bool' = False, ub_conv: 'bool' = False) -> None
```

*Mirrors R's `confint`.*

CI for tau^2 / tau / I^2 / H^2 (mirrors ``metafor::confint``).

## `predict_rma`

```python
predict_rma(est: 'float', se: 'float', tau2: 'float', k: 'int', p: 'int' = 1, parms: 'int | None' = None, level: 'float' = 0.05, predtype: 'str' = 'default', test: 'str' = 'z', newvi: 'float' = 0.0, transf: 'Callable[..., float] | None' = None, targs=None) -> 'PredictResult'
```

*Mirrors R's `predict.rma`.*

Prediction (+ CI/PI) for an intercept-only model (``metafor::predict.rma``).

Replicates ``predict.rma.uni`` for the ``k.new = 1`` (intercept-only) case.
``est``/``se``/``tau2`` are on the analysis scale (log scale for ratio
types); ``level`` is the alpha level (0.05 -> 95%).

``predtype`` controls the prediction interval reference distribution, exactly
as in metafor:

- ``"default"`` — uses the CI reference (normal for ``test="z"``).
- ``"simple"``  — ``crit = qnorm``, ``pi.se = sqrt(tau2 + newvi)`` (drops ``se``).
- ``"riley"``   — ``crit = qt(level/2, df = k - parms)``  (Riley et al. 2011).
- ``"t"``       — ``crit = qt(level/2, df = k - p)``.

``parms`` defaults to ``p + 1`` (rma.uni counts one extra parameter for
tau^2). ``test`` mirrors metafor's ``test`` (``"z"`` default, or
``"knha"``/``"t"`` for a t-based CI).

## `PredictResult`

```python
PredictResult(pred: 'float', se: 'float', ci_lb: 'float', ci_ub: 'float', pi_lb: 'float | None', pi_ub: 'float | None', pi_dist: 'str', ddf: 'float', pi_ddf: 'float | None', level: 'float') -> None
```

*Mirrors R's `predict.rma`.*

Predicted value + CI/PI (mirrors ``metafor::predict.rma``).

## `hc_rma`

```python
hc_rma(yi, vi, level: 'float' = 0.05, *, beta_rma: 'float | None' = None, se_rma: 'float | None' = None, ci_lb_rma: 'float | None' = None, ci_ub_rma: 'float | None' = None, tau2_rma: 'float | None' = None) -> 'HcResult'
```

*Mirrors R's `hc.rma.uni`.*

Hartung-Knapp interval for an intercept-only model (``metafor::hc.rma.uni``).

Replicates the *exact* Sidik & Jonkman (2002) / Hartung (1999) method: it
refits tau^2 with the DerSimonian-Laird estimator, then finds the inflated
critical value ``u0 = SDR * t0`` where ``t0`` solves

    int_t0^inf P(Q <= finv(r/t0)) phi(r) dr = level/2

with ``Q ~ Gamma(shape(r), scale(r))``. The same ``_zeroin`` (R
``uniroot``) root finder and a Gauss-Kronrod integration with R's
``integrate`` tolerance are used, so ``t0`` matches R's ``uniroot`` root.

## `HcResult`

```python
HcResult(beta: 'float', se: 'float', ci_lb: 'float', ci_ub: 'float', tau2: 'float', beta_rma: 'float', se_rma: 'float', ci_lb_rma: 'float', ci_ub_rma: 'float', tau2_rma: 'float', level: 'float') -> None
```

*Mirrors R's `hc.rma.uni`.*

Hartung-Knapp (Sidik-Jonkman exact) interval (``metafor::hc.rma.uni``).

## `estimate_tau2`

```python
estimate_tau2(y: 'list[float] | np.ndarray', v: 'list[float] | np.ndarray', method: 'str', n: 'list[float] | np.ndarray | None' = None) -> 'float'
```

Return tau-squared for a moment/iterative estimator.

``method`` is case-insensitive and must be one of DL/HE/HS/SJ/PM/EB (the
likelihood-based ML/REML are handled by metafor's Fisher-scoring loop in
``pooling``). ``n`` is the per-study total sample size, required only for
``HS``.

## `rma_glmm`

```python
rma_glmm(measure: 'str', ai=None, bi=None, ci=None, di=None, xi=None, ni=None, mods=None, data=None, slab=None, method: 'str' = 'ML', test: 'str' = 'z', level: 'float' = 0.95, btt=None, digits: 'int' = 4, nAGQ: 'int' = 7, add: 'float' = 0.5, to: 'str' = 'only0', drop00: 'bool' = True, verbose: 'bool' = False, control=None) -> 'RmaGlmmResult'
```

Fit a generalised linear mixed-effects meta-analysis model.

``measure`` is one of ``OR``/``RR``/``RD`` (two-arm; pass ``ai/bi/ci/di``)
or ``PLO``/``PR``/``PLN`` (single-arm; pass ``xi/ni``).  ``method`` is
``"ML"`` (random effects via AGQ) or ``"FE"`` (fixed effects).

## `RmaGlmmResult`

```python
RmaGlmmResult(b: 'list[float]', beta: 'list[float]', se: 'list[float]', zval: 'list[float]', pval: 'list[float]', ci_lb: 'list[float]', ci_ub: 'list[float]', vb: 'np.ndarray', tau2: 'float', se_tau2: 'float | None', sigma2: 'float', rho: 'float', I2: 'float', H2: 'float', vt: 'float', QE_Wld: 'float', QEp_Wld: 'float', QE_LRT: 'float', QEp_LRT: 'float', QE_df: 'int', QM: 'float', QMdf: 'list', QMp: 'float', k: 'int', k_f: 'int', k_yi: 'int', k_eff: 'int', k_all: 'int', p: 'int', p_eff: 'int', parms: 'int', int_only: 'bool', yi: 'list[float]', vi: 'list[float]', X: 'np.ndarray', fit_stats: 'dict', measure: 'str', method: 'str', model: 'str', test: 'str', digits: 'int', level: 'float', slab: 'list[str]') -> None
```

*Mirrors R's `rma.glmm`.*

Fitted ``rma.glmm`` model, mirroring the R ``$`` field set.

## `rma_mv`

```python
rma_mv(yi, vi=None, V=None, mods=None, data=None, random: 'str' = '~ 1 | study', struct: 'str' = 'CS', rho=None, method: 'str' = 'REML', level: 'float' = 95.0, test: 'str' = 'z', dfs: 'str | np.ndarray' = 'residual', intercept: 'bool' = True, sigma2=None, slab=None, cluster=None) -> 'RmaMvResult'
```

*Mirrors R's `rma.mv`.*

Fit a multilevel / multivariate random-effects model (``metafor::rma.mv``).

**Args**

- `yi` — effect estimates (length ``k``).
- `vi` — sampling variances (length ``k``), used when ``V`` is None.
- `V` — full sampling variance-covariance matrix (``k x k``).
- `mods` — fixed-effect design matrix (``k x m``, without intercept), or None for an intercept-only model.
- `data` — dict-like (or pandas DataFrame) providing the grouping factors named in ``random``.
- `random` — random-effects formula(s), e.g. ``"~ 1 | study"``, ``"~ 1 | district/school"`` (nested intercepts), ``"~ year | district"`` (compound-symmetric inner factor), or ``"~ year | district"`` with ``struct="GEN"`` for random slopes.
- `struct` — variance structure for an ``~ inner | outer`` term -- ``"CS"`` (default) or ``"GEN"`` (random slopes with unstructured covariance).  A list matches multiple ``~ inner | outer`` terms.
- `rho` — fixes the correlation of the structured term(s) instead of estimating it, as ``rma.mv`` does.  Scalar or one value per term.
- `method` — ``"REML"`` (default) or ``"ML"``.
- `level` — confidence level (95 -> 95% CI).
- `test` — ``"z"`` (default) or ``"t"`` / ``"knha"`` / ``"adhoc"``. ``"t"`` uses a t-distribution with ``dfs`` degrees of freedom; ``"knha"`` additionally scales the variance by ``RSS/(k-p)`` (Knapp-Hartung); ``"adhoc"`` scales by ``max(RSS/(k-p), 1)``.
- `dfs` — ``"residual"`` (default, ``k-p``), ``"contain"``, or a numeric value / array (one df per coefficient).
- `intercept` — prepend an intercept column to ``mods`` (``False`` matches R's ``mods = ~ x1 + x2 - 1``).
- `sigma2` — fix the variance components instead of estimating them (R's ``sigma2=``), which is what the leave-cluster-out influence diagnostics need.
- `slab` — study labels, kept on the result for the plotting methods.
- `cluster` — optional clustering variable, kept for the diagnostics.

## `RmaMvResult`

```python
RmaMvResult(method: 'str', k: 'int', p: 'int', sigma2: 'list[float]', beta: 'list[float]', se: 'list[float]', zval: 'list[float]', pval: 'list[float]', ci_lb: 'list[float]', ci_ub: 'list[float]', qe: 'float', qe_df: 'int', qe_p: 'float', random: 'str' = '', int_only: 'bool' = True, M: 'np.ndarray' = <factory>, vb: 'np.ndarray' = <factory>, X: 'np.ndarray' = <factory>, yi: 'np.ndarray' = <factory>, level: 'float' = 95.0, tau2: 'list[float] | None' = None, rho: 'list[float] | None' = None, ll: 'float | None' = None, parms: 'int | None' = None, fit_stats: 'dict | None' = None, ddf: 'list[float] | None' = None, s2w: 'float' = 1.0, components: 'list | None' = None, params: 'np.ndarray' = <factory>, V: 'np.ndarray' = <factory>, mods: 'np.ndarray | None' = None, struct: 'str' = 'CS', dfs: 'str | float | list | None' = None, intercept_incl: 'bool' = True, slab: 'list[str] | None' = None, cluster: 'list | None' = None, test: 'str' = 'z', _data: 'dict | None' = None) -> None
```

*Mirrors R's `rma.mv`.*

Multi-level model fit, mirroring the fields of ``rma.mv`` objects.

## `anova_rma`

```python
anova_rma(full: 'RmaMvResult', reduced: 'RmaMvResult', df: 'int | None' = None)
```

Likelihood-ratio test between two nested ``rma_mv`` fits.

Replicates ``metafor::anova.rma`` for two models.  The model with more
estimated parameters is taken as the *full* model; the test statistic is
``LRT = deviance(reduced) - deviance(full)`` evaluated under the fitted
``method`` (``ML`` by default in metafor; the ``ll.QE`` constant cancels,
so this equals ``-2 * (ll.reduced - ll.full)``).  ``pval`` uses a chi-square
with ``df = parms.full - parms.reduced``.

Returns a dict mirroring metafor's two-row table:
    ``{"full": {...}, "reduced": {...}, "LRT": ..., "pval": ..., "df": ...}``

## `ranef`

```python
ranef(result: 'RmaMvResult', level: 'float | None' = None) -> 'dict'
```

Best linear unbiased predictions (BLUP) of the random effects.

Replicates ``metafor::ranef.rma.mv``.  Returns a dict keyed by the random
effect term name (``"district"``, ``"district/school"``, ``"~ year |
district"``, ...).  Each entry holds ``labels`` plus the prediction vector
and its ``se`` / ``pi.lb`` / ``pi.ub`` (prediction interval).  For
``struct="GEN"`` terms metafor does not report ``se``; the entry then
holds a ``pred`` matrix (outer groups x random-effect variables) with
``var_names`` columns.

## `robust`

```python
robust(result: 'RmaMvResult', cluster=None, adjust: 'bool' = True, clubSandwich: 'bool' = False, vcov: 'str' = 'CR2', coef_test: 'str' = 'Satterthwaite', level: 'float | None' = None)
```

*Mirrors R's `robust`.*

Cluster-robust inference for an ``rma.mv`` fit (``metafor::robust``).

``cluster`` is required, exactly as in R (``robust.rma.mv`` has no
default); when the fit stored one, it is reused.

Default path (``clubSandwich=False``) — metafor's CR0/CR1 sandwich over
the full marginal var-cov matrix: ``bread = vb X' M^-1``, block-diagonal
meat from per-cluster residual outer products, then ``n/(n-p)`` scaling.

- ``adjust=True``  -> "CR1" (``n/(n-p)``)
- ``adjust=False`` -> "CR0"
- ``adjust="Stata"``/"Stata1" -> "CR1.S1" (``n/(n-1) * (k-1)/(k-p)``)
- ``adjust="Stata2"``         -> "CR1.S2" (``n/(n-1)``)

``clubSandwich=True`` reproduces ``clubSandwich::vcovCR`` (CR2/CR3) with
the working covariance equal to the full marginal matrix ``M``, plus
``coef_test`` in {"Satterthwaite", "naive-t", "naive-tp", "z"}.

## `rma_peto`

```python
rma_peto(ai, bi, ci, di, *, add: 'float' = 0.5, drop00: 'bool' = True, level: 'float' = 0.05, slab=None) -> 'RmaPetoResult'
```

*Mirrors R's `rma.peto`.*

Peto's one-step odds-ratio meta-analysis (``metafor::rma.peto``).

``ai/bi/ci/di`` are the treatment/control event and non-event counts.
The estimate is fixed-effect (``tau2 = 0``) and computed on the log scale.

## `RmaPetoResult`

```python
RmaPetoResult(beta: 'float', se: 'float', zval: 'float', pval: 'float', ci_lb: 'float', ci_ub: 'float', tau2: 'float', I2: 'float', H2: 'float', QE: 'float', QEp: 'float', k: 'int', k_pos: 'int', measure: 'str', method: 'str', level: 'float', ai: 'list[float] | None' = None, bi: 'list[float] | None' = None, ci: 'list[float] | None' = None, di: 'list[float] | None' = None, slab: 'list[str] | None' = None, add: 'float' = 0.5, drop00: 'bool' = True) -> None
```

*Mirrors R's `rma.peto`.*

``metafor::rma.peto`` — Peto's one-step odds ratio (fixed effects).

## `rma_mh`

```python
rma_mh(ai, bi, ci, di, *, measure: 'str' = 'OR', add: 'float' = 0.5, drop00: 'bool' = True, correct: 'bool' = True, level: 'float' = 0.05, slab=None) -> 'RmaMhResult'
```

*Mirrors R's `rma.mh`.*

Mantel-Haenszel meta-analysis (``metafor::rma.mh``).

Supports ``measure`` in ``{"OR", "RR", "RD"}``.  ``beta`` is on the log
scale for OR/RR and on the raw scale for RD.  For the OR, the
Cochran-Mantel-Haenszel (``MH``/``CO``), Breslow-Day (``BD``) and Tarone
(``TA``) tests are also returned.

## `RmaMhResult`

```python
RmaMhResult(beta: 'float', se: 'float', zval: 'float', pval: 'float', ci_lb: 'float', ci_ub: 'float', tau2: 'float', I2: 'float', H2: 'float', QE: 'float', QEp: 'float', k: 'int', measure: 'str', method: 'str', CO: 'float | None', COp: 'float | None', MH: 'float | None', MHp: 'float | None', BD: 'float | None', BDp: 'float | None', TA: 'float | None', TAp: 'float | None', k_pos: 'int | None', level: 'float', ai: 'list[float] | None' = None, bi: 'list[float] | None' = None, ci: 'list[float] | None' = None, di: 'list[float] | None' = None, slab: 'list[str] | None' = None, add: 'float' = 0.5, drop00: 'bool' = True, correct: 'bool' = True) -> None
```

*Mirrors R's `rma.mh`.*

``metafor::rma.mh`` — Mantel-Haenszel (fixed effects).

## `rma`

```python
rma(yi, vi=None, sei=None, mods=None, data=None, slab=None, measure: 'str' = 'GEN', method: 'str' = 'REML', test: 'str' = 'z', level: 'float' = 0.95, btt=None, digits: 'int' = 4, scale=None, scale_intercept: 'bool' = True, link: 'str' = 'log', optbeta: 'bool' = False, alpha=None, warn_missing: 'bool' = True) -> "'RmaResult | RmaLsResult'"
```

*Mirrors R's `rma`.*

Fit a univariate meta-analysis model (``metafor::rma``).

**Args**

- `yi` — analysis-scale effect sizes (e.g. log risk ratios).
- `vi` — sampling variances (alternative to ``sei``).
- `sei` — sampling standard errors (alternative to ``vi``).
- `mods` — moderator values — a 1-D vector for a single moderator or a 2-D array for several, WITHOUT an intercept column (one is added).
- `slab` — study labels.
- `measure` — effect-size measure (recorded, does not change the scale).
- `method` — tau^2 estimator: FE/DL/HE/HS/SJ/PM/MP/ML/REML/EB (default REML).
- `test` — "z" (default), "t", "knha" or "adhoc".
- `level` — confidence level (0.95 -> 95% CI).  alpha = 1 - level.
- `btt` — 1-based coefficient indices to include in the QM omnibus test (default: the intercept for intercept-only models, otherwise all moderator coefficients).
- `digits` — printing precision (recorded only).
- `scale, scale_intercept, link, optbeta, alpha` — when ``scale`` is given a location-scale model is fitted instead (``rma(yi, vi, mods=~x, scale=~z)`` in R, class ``rma.ls``); the fit is delegated to :func:`pimeta.rma_ls` and an :class:`~pimeta.RmaLsResult` is returned.
- `warn_missing` — emit a :class:`~pimeta.missing.MissingDataWarning` naming the studies dropped because ``yi``, ``vi`` or a moderator is missing.  R drops them silently through ``na.action``; pass ``False`` to drop them silently here too.

## `RmaResult`

```python
RmaResult(b: 'list[float]', se: 'list[float]', zval: 'list[float]', pval: 'list[float]', ci_lb: 'list[float]', ci_ub: 'list[float]', vb: 'np.ndarray', tau2: 'float', se_tau2: 'float | None', tau2_fix: 'bool', I2: 'float', H2: 'float', R2: 'float | None', vt: 'float', QE: 'float', QEp: 'float', QM: 'float', QMdf: 'list', QMp: 'float', k: 'int', k_f: 'int', k_eff: 'int', k_all: 'int', p: 'int', p_eff: 'int', parms: 'int', int_only: 'bool', int_incl: 'bool', yi: 'list[float]', vi: 'list[float]', X: 'np.ndarray', mods: 'np.ndarray | None', weights: 'list[float]', weights_pct: 'list[float]', fitted: 'list[float]', residuals: 'list[float]', method: 'str', test: 'str', dfs: 'float | None', ddf: 'float | None', s2w: 'float', measure: 'str', slab: 'list[str]', not_na: 'list[bool]', level: 'float', coefnames: 'list[str] | None' = None) -> None
```

*Mirrors R's `rma.uni`.*

Fitted ``rma.uni`` model, mirroring the R ``$`` field set.

## `selmodel`

```python
selmodel(x: 'RmaResult', type: 'str', alternative: 'str' = 'greater', prec=None, subset=None, delta=None, steps=None, decreasing: 'bool' = False, verbose: 'bool' = False, digits: 'int' = 4, control: 'dict | None' = None, **kwargs) -> 'SelModelResult'
```

Fit a publication-bias selection model on an existing ``rma`` fit.

**Args**

- `x` — fitted ``RmaResult`` (from :func:`rma`).
- `type` — selection model type (one of ``TYPE_OPTIONS``, prefix-matched).
- `alternative` — "greater" (default), "less" or "two.sided".
- `prec` — optional precision weights ("sei"/"vi"/"ninv"/"sqrtninv").
- `subset` — optional boolean vector selecting studies to model.
- `delta` — optional fixed values for the selection parameters.
- `steps` — optional p-value cut-points (required for ``stepfun``).
- `decreasing` — monotone decreasing step weights (``stepfun`` only).
- `verbose` — print progress.
- `digits` — printing precision (recorded only).
- `control` — dict of optimisation/Hessian control options.

**Returns**

- `:class` — `SelModelResult` with the full metafor ``$`` field set.

## `SelModelResult`

```python
SelModelResult(beta: 'list[float]', b: 'list[float]', se: 'list[float]', zval: 'list[float]', pval: 'list[float]', ci_lb: 'list[float]', ci_ub: 'list[float]', vb: 'np.ndarray', tau2: 'float', se_tau2: 'float', ci_lb_tau2: 'float', ci_ub_tau2: 'float', delta: 'list[float]', vd: 'np.ndarray', se_delta: 'list[float]', zval_delta: 'list[float]', pval_delta: 'list[float]', ci_lb_delta: 'list[float]', ci_ub_delta: 'list[float]', QM: 'float', QMp: 'float', ll: 'float', ll0: 'float', LRT: 'float', LRTdf: 'int', LRTp: 'float', LRT_tau2: 'float', LRTp_tau2: 'float', parms: 'int', fit_stats: 'dict', pvals: 'list[float]', type: 'str', alternative: 'str', steps: 'list[float]', pgrp: 'list', k0: 'int', k1: 'int', model: 'str', k: 'int', p: 'int', tau2_fix: 'bool', method: 'str', yi: 'list[float]', vi: 'list[float]', level: 'float') -> None
```

*Mirrors R's `rma.uni.selmodel`.*

Fitted ``rma.uni.selmodel`` model, mirroring the R ``$`` field set.

## `SelProfileResult`

```python
SelProfileResult(delta, ll, maxll, delta_hat, ci_lb, ci_ub, level, type_, k)
```

*Mirrors R's `profile.rma.uni.selmodel`.*

Result of :func:`profile_selmodel` (``profile.rma.uni.selmodel``).

``delta`` holds the profiled selection parameters, ``ll`` the corresponding
maximised log-likelihoods and ``ci_lb``/``ci_ub`` the profile-likelihood
limits (the ``delta`` values where ``ll`` drops by ``chi2(1, 1-alpha)/2``).

## `profile_selmodel`

```python
profile_selmodel(result, delta=None, n_points: 'int' = 20, span=None, level: 'float' = 0.95)
```

Profile the log-likelihood of a selection model over ``delta``.

``profile.rma.uni.selmodel`` refits the model on a grid of fixed ``delta``
values and records the maximised log-likelihood, then inverts the
likelihood-ratio test to obtain a profile CI for the selection parameter.
The refits here re-estimate tau^2 exactly as the rest of this module does;
R additionally fixes tau^2 at the full-model estimate, so the curve can
differ in the third or fourth digit.

**Args**

- `result` — :class:`SelModelResult` from :func:`selmodel`.
- `delta` — explicit grid (default: ``n_points`` values spanning ``delta_hat +/- span * |delta_hat|``, or +/-2 when ``delta_hat == 0``).
- `n_points` — grid size when ``delta`` is not given.
- `span` — half-width of the default grid, as a multiple of ``|delta_hat|``.
- `level` — confidence level for the profile limits.

**Returns**

- `:class` — `SelProfileResult`.

## `rma_ls`

```python
rma_ls(yi, vi=None, sei=None, mods=None, scale=None, data=None, slab=None, measure: 'str' = 'GEN', method: 'str' = 'REML', test: 'str' = 'z', level: 'float' = 0.95, intercept: 'bool' = True, scale_intercept: 'bool' = True, link: 'str' = 'log', optbeta: 'bool' = False, alpha=None, btt=None, att=None, warn_missing: 'bool' = True) -> 'RmaLsResult'
```

Fit a location-scale meta-analysis model (``rma(..., scale=~mods)``).

**Args**

- `yi` — analysis-scale effect sizes.
- `vi` — sampling variances (alternative to ``sei``).
- `sei` — sampling standard errors (alternative to ``vi``).
- `mods` — location moderators — 1-D or 2-D, WITHOUT an intercept (one is added when ``intercept=True``).
- `scale` — scale moderators — 1-D or 2-D, WITHOUT an intercept (added when ``scale_intercept=True``).  Required.
- `data` — dict / DataFrame supplying the columns named by string, so that ``mods="x"`` and ``scale="z"`` can be used like R formulas.
- `slab` — study labels.
- `measure` — effect-size measure (recorded only).
- `method` — ``"REML"`` (default) or ``"ML"`` — location-scale models have no closed-form tau^2 estimator.
- `test` — coefficient test for the location part (``"z"``, ``"t"``, ``"knha"``, ``"adhoc"``); the ``alpha`` tests use ``k - q`` df for the t variants, as metafor does.
- `level` — confidence level (95 -> 95% CI).
- `intercept, scale_intercept` — include the intercepts in the two designs.
- `link` — ``"log"`` (default, tau^2 = exp(z'alpha)) or ``"identity"``.
- `optbeta` — estimate the location coefficients jointly with ``alpha`` instead of profiling them out (R's ``control=list(optbeta=TRUE)``).
- `alpha` — fix (some of) the scale coefficients; ``NaN`` entries stay free.
- `btt, att` — 1-based indices for the omnibus tests of the location and the scale coefficients (default: all).
- `warn_missing` — emit a warning naming the studies dropped because a value is missing (R's ``na.action`` drops them silently).

## `RmaLsResult`

```python
RmaLsResult(b: 'list[float]', se: 'list[float]', zval: 'list[float]', pval: 'list[float]', ci_lb: 'list[float]', ci_ub: 'list[float]', vb: 'np.ndarray', alpha: 'list[float]', se_alpha: 'list[float]', zval_alpha: 'list[float]', pval_alpha: 'list[float]', ci_lb_alpha: 'list[float]', ci_ub_alpha: 'list[float]', va: 'np.ndarray', QS: 'float', QSdf: 'tuple[int, float]', QSp: 'float', tau2: 'list[float]', QE: 'float', QEp: 'float', I2: 'list[float] | None', H2: 'list[float] | None', vt: 'float', parms: 'int', ll: 'float', fit_stats: 'dict', k: 'int', p: 'int', q: 'int', alphas: 'int', method: 'str', model: 'str' = 'rma.ls', link: 'str' = 'log', test: 'str' = 'z', ddf: 'float' = nan, ddf_alpha: 'float' = nan, level: 'float' = 0.95, intercept: 'bool' = True, scale_intercept: 'bool' = True, optbeta: 'bool' = False, alpha_fix: 'list[bool]' = <factory>, Z_int_incl: 'bool' = True, Z_int_only: 'bool' = False, int_only: 'bool' = True, measure: 'str' = 'GEN', yi: 'list[float]' = <factory>, vi: 'list[float]' = <factory>, X: 'np.ndarray | None' = None, Z: 'np.ndarray | None' = None, slab: 'list[str]' = <factory>, QM: 'float' = nan, QMdf: 'tuple[int, float]' = (0, nan), QMp: 'float' = nan, R2: 'object' = None) -> None
```

*Mirrors R's `rma.ls`.*

Location-scale fit (mirrors metafor's ``rma.ls`` objects).

## `profile_rma_ls`

```python
profile_rma_ls(x, alpha=None, steps=20, xlim=None, plot=False, **kw)
```

*Mirrors R's `profile.rma.ls`.*

``profile.rma.ls`` — profile the log-likelihood over one scale
coefficient.

For every grid value of the chosen ``alpha`` the model is refitted with
that coefficient pinned while the others are re-estimated, which is what
metafor's ``.profile.rma.ls`` does (``rma(..., scale=Z, alpha=c(NA, v))``).

**Args**

- `x` — a :class:`RmaLsResult`.
- `alpha` — 1-based index of the scale coefficient to profile (metafor's ``alpha=``).  Defaults to the first non-fixed coefficient.
- `steps` — number of grid points, or an explicit sequence of values.
- `xlim` — profiling range; by default ``vc ± qnorm(0.995) * se.alpha`` around the estimate, like metafor.
- `plot` — draw the profile plot immediately.
- `**kw` — forwarded to :func:`pimeta.plotting.plot_mv_profile`.

**Returns**

- `:class` — `RmaLsProfileResult`.

## `RmaLsProfileResult`

```python
RmaLsProfileResult(component, index, alpha, ll, maxll, beta, ci_lb, ci_ub, vc, method, xlab, title, comps=1)
```

*Mirrors R's `profile.rma.ls`.*

Result of :func:`profile_rma_ls` (mirrors ``profile.rma.ls`` objects).

``alpha`` holds the profiling grid (the values of the chosen scale
coefficient), ``ll`` the maximised log-likelihood at each grid point and
``beta`` / ``ci_lb`` / ``ci_ub`` the location coefficients (one row per
grid point).  ``maxll`` is the log-likelihood of the unconstrained fit.

## `confint_rma_ls`

```python
confint_rma_ls(x, parm=None, level=None, digits=4)
```

*Mirrors R's `confint.rma.ls`.*

``confint.rma.ls`` — Wald intervals for the scale coefficients.

``parm`` selects the coefficients by name (``"z"`` / ``"z1"`` ...) or by
1-based index; the default is every coefficient.  Returns a dict with
``estimate``, ``ci_lb``, ``ci_ub``, ``level`` and ``parm``.

metafor additionally supports profile-likelihood intervals here; use
:func:`profile_rma_ls` and invert the profile yourself for those.
