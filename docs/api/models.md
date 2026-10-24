# Models

*90 public names, generated from `pimeta.__all__` and the docstrings in `src/pimeta/`. Do not edit this page by hand — run `python tools/gen_api_docs.py`.*

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
| [`Basis`](#basis) | — |
| [`CurveResult`](#curveresult) | — |
| [`NonlinearTestResult`](#nonlineartestresult) | — |
| [`WaldResult`](#waldresult) | — |
| [`default_knots`](#default_knots) | — |
| [`IpdResult`](#ipdresult) | — |
| [`StudyFit`](#studyfit) | — |
| [`ipd_two_stage`](#ipd_two_stage) | — |
| [`IpdCurve`](#ipdcurve) | — |
| [`IpdCurveResult`](#ipdcurveresult) | — |
| [`ipd_two_stage_curve`](#ipd_two_stage_curve) | — |
| [`linear_model`](#linear_model) | — |
| [`logistic_model`](#logistic_model) | — |
| [`OneStageResult`](#onestageresult) | — |
| [`ipd_one_stage`](#ipd_one_stage) | — |
| [`PredictionResult`](#predictionresult) | — |
| [`auc_logit`](#auc_logit) | — |
| [`oe_ratio`](#oe_ratio) | — |
| [`prediction_meta`](#prediction_meta) | — |
| [`KmResult`](#kmresult) | — |
| [`ReconstructedIpD`](#reconstructedipd) | — |
| [`RmstResult`](#rmstresult) | — |
| [`km_curve`](#km_curve) | — |
| [`km_reconstruct`](#km_reconstruct) | — |
| [`rmst`](#rmst) | — |
| [`piecewise_exp`](#piecewise_exp) | — |
| [`piecewise_joint`](#piecewise_joint) | — |
| [`PiecewiseJointResult`](#piecewisejointresult) | — |
| [`PiecewiseResult`](#piecewiseresult) | — |
| [`poisson_irls`](#poisson_irls) | — |
| [`split_followup`](#split_followup) | — |
| [`fp`](#fp) | — |
| [`nonlinear_test`](#nonlinear_test) | — |
| [`predict_curve`](#predict_curve) | — |
| [`rcs`](#rcs) | — |
| [`wald_test`](#wald_test) | anova |
| [`hc_rma`](#hc_rma) | hc.rma.uni |
| [`HcResult`](#hcresult) | hc.rma.uni |
| [`estimate_tau2`](#estimate_tau2) | — |
| [`rma_glmm`](#rma_glmm) | — |
| [`RmaGlmmResult`](#rmaglmmresult) | rma.glmm |
| [`rma_mv`](#rma_mv) | rma.mv |
| [`RmaMvResult`](#rmamvresult) | rma.mv |
| [`anova_rma`](#anova_rma) | — |
| [`ranef`](#ranef) | — |
| [`robust`](#robust) | rma.mv |
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
| [`vcalc`](#vcalc) | — |
| [`shared_control_corr`](#shared_control_corr) | — |
| [`tau_cor`](#tau_cor) | — |
| [`TauCorResult`](#taucorresult) | — |
| [`rho_sensitivity`](#rho_sensitivity) | — |
| [`RhoSensitivityResult`](#rhosensitivityresult) | — |
| [`confint_rma_mv`](#confint_rma_mv) | — |
| [`ConfintMvResult`](#confintmvresult) | — |
| [`predict_region`](#predict_region) | — |
| [`PredictionRegion`](#predictionregion) | — |
| [`DtaData`](#dtadata) | — |
| [`DtaCorrectionWarning`](#dtacorrectionwarning) | — |
| [`ReitsmaResult`](#reitsmaresult) | — |
| [`HsrocResult`](#hsrocresult) | — |
| [`SrocResult`](#srocresult) | — |
| [`escalc_dta`](#escalc_dta) | — |
| [`reitsma`](#reitsma) | reitsma |
| [`hsroc`](#hsroc) | — |
| [`sroc`](#sroc) | sroc |
| [`auc`](#auc) | AUC |

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

This is the shared numeric core behind `run_meta` and the
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

## `Basis`

```python
Basis(values: 'np.ndarray', names: 'list[str]', label: 'str', kind: 'str', x: 'np.ndarray', builder: 'Callable[[np.ndarray], np.ndarray]', detail: 'dict[str, Any]' = <factory>) -> None
```

An expanded moderator: the columns, their names, and how to rebuild them.

**Attributes**

- `values` — ``(n, m)`` matrix of the expanded moderator.
- `names` — one name per column, ``"<label>[j]"``.
- `label` — what the term is called in the model, e.g. ``"rcs(dose,4)"``.
- `kind` — ``"rcs"`` or ``"fp"``.
- `builder` — maps a 1-D vector of new values to a new ``(g, m)`` matrix.
- `detail` — the construction parameters (knots, powers, scale), for display and for the record.

## `CurveResult`

```python
CurveResult(x: 'np.ndarray', fit: 'np.ndarray', se: 'np.ndarray', ci_lb: 'np.ndarray', ci_ub: 'np.ndarray', pi_lb: 'np.ndarray | None', pi_ub: 'np.ndarray | None', level: 'float', term: 'str', at: 'dict[str, float]' = <factory>) -> None
```

A fitted curve with its confidence and prediction bands.

**Attributes**

- `x` — the grid the curve was evaluated on.
- `fit` — predicted effect at each grid point (analysis scale).
- `se` — standard error of the prediction.
- `ci_lb, ci_ub` — confidence band.
- `pi_lb, pi_ub` — prediction band (``None`` entries when ``tau2 == 0``, because then there is no between-study distribution to predict).
- `level` — confidence level, e.g. 0.95.
- `term` — the basis label.
- `at` — the values the other moderators were held at.

## `NonlinearTestResult`

```python
NonlinearTestResult(term: 'str', overall: 'WaldResult', nonlinear: 'WaldResult', k: 'int') -> None
```

Overall and non-linearity tests for one basis term.

**Attributes**

- `term` — the term label, e.g. ``"rcs(dose,4)"``.
- `overall` — joint test over every column of the term — "is there any association?".
- `nonlinear` — joint test over the non-linear columns only — "is the association non-linear?". For a spline this is all columns after the first, because the first column is ``x`` itself.
- `k` — number of studies.

## `WaldResult`

```python
WaldResult(QM: 'float', df: 'int', pval: 'float', names: 'list[str]', test: 'str', ddf: 'float | None', k: 'int') -> None
```

Joint test that a set of coefficients is zero.

Mirrors ``metafor::anova.rma``: ``QM`` is the quadratic form
``b' V^-1 b`` over the selected coefficients, referred to a chi-squared
distribution (``test="z"``) or an F distribution with ``df`` and ``ddf``
degrees of freedom (``test="knha"``/``"t"``/``"adhoc"``).

**Attributes**

- `QM` — the test statistic.
- `df` — number of coefficients tested (the numerator df).
- `pval` — p-value.
- `names` — the coefficients that were tested.
- `test` — ``"chi2"`` or ``"F"``.
- `ddf` — denominator df, or ``None`` for the chi-squared test.
- `k` — number of studies in the model.

## `default_knots`

```python
default_knots(x, nk: 'int' = 4) -> 'np.ndarray'
```

Harrell's default knot positions for ``nk`` knots.

Uses the published percentile table for 3–7 knots and an evenly spaced
grid of percentiles from 5% to 95% beyond that. Quantiles are the
``type = 7`` (linear interpolation) ones that R uses by default.

**Args**

- `x` — the moderator.
- `nk` — number of knots.

**Returns**

Sorted knot vector of length ``nk``.

**Raises**

- `ValueError` — for fewer than 3 knots, or more knots than distinct values.

## `IpdResult`

```python
IpdResult(term: 'str', b: 'float', se: 'float', zval: 'float', pval: 'float', ci_lb: 'float', ci_ub: 'float', tau2: 'float', QE: 'float', QE_df: 'int', QE_p: 'float', I2: 'float', k: 'int', studies: 'int', excluded: 'dict[str, str]', method: 'str', level: 'float', test: 'str', per_study: 'list[StudyFit]', fit: 'Any') -> None
```

A pooled individual-participant analysis.

**Attributes**

- `term` — name of the coefficient that was pooled.
- `b, se, zval, pval, ci_lb, ci_ub` — the pooled estimate.
- `tau2, QE, QE_df, QE_p, I2` — between-study heterogeneity.
- `k` — number of studies that contributed.
- `studies` — studies present in the data.
- `excluded` — studies that could not be fitted, with the reason.
- `method` — τ² estimator.
- `level` — confidence level.
- `test` — reference distribution (``"z"`` unless ``test=`` was given).
- `per_study` — one `StudyFit` per contributing study.
- `fit` — the underlying `rma_uni.RmaResult`.

## `StudyFit`

```python
StudyFit(study: 'str', n: 'int', events: 'int | None', coefnames: 'list[str]', b: 'list[float]', se: 'list[float]', vb: 'np.ndarray', term: 'int', converged: 'bool') -> None
```

One study's first-stage model.

**Attributes**

- `study` — the study label.
- `n` — participants in the analysis.
- `events` — outcome events (``None`` for a continuous outcome).
- `coefnames` — ``["(Intercept)", ...]``.
- `b, se` — coefficients and their standard errors.
- `vb` — covariance matrix.
- `term` — index of the coefficient being pooled.
- `converged` — whether the first stage converged.

## `ipd_two_stage`

```python
ipd_two_stage(y, *, study, treat, covariates=None, family: 'str' = 'binomial', term: 'str' = 'treat', method: 'str' = 'REML', level: 'float' = 0.95, test: 'str' = 'z', min_events: 'int' = 1) -> 'IpdResult'
```

Fit one model per study, then pool one coefficient.

**Args**

- `y` — the outcome, one entry per participant.
- `study` — the study label, one entry per participant.
- `treat` — the treatment indicator (0/1), one entry per participant. It is put first in the design, so ``term="treat"`` means the treatment effect.
- `covariates` — optional list of participant-level covariates, each one entry per participant.
- `family` — ``"binomial"`` (logistic) or ``"gaussian"`` (linear).
- `term` — which coefficient to pool — ``"treat"``, a covariate name, or a product such as ``"treat:age"``. The product is formed for you.
- `method` — τ² estimator for the second stage.
- `level` — confidence level.
- `test` — ``"z"``, ``"t"``, ``"knha"`` or ``"adhoc"`` for the second stage.
- `min_events` — studies with fewer events than this are excluded (a study with no events in one arm has no finite logistic estimate).

**Returns**

`IpdResult`.

**Raises**

- `ValueError` — when the inputs disagree in length, the study has too few participants, or the requested term does not exist.

**Example**

- `>>> fit = ma.ipd_two_stage(y, study=study, treat=treat)   # doctest` — +SKIP
- `>>> fit.b, fit.tau2, fit.k                                # doctest` — +SKIP

## `IpdCurve`

```python
IpdCurve(x: 'np.ndarray', est: 'np.ndarray', se: 'np.ndarray', ci_lb: 'np.ndarray', ci_ub: 'np.ndarray', pred_lb: 'np.ndarray', pred_ub: 'np.ndarray', outside: 'np.ndarray') -> None
```

The treatment-effect curve on a grid of covariate values.

## `IpdCurveResult`

```python
IpdCurveResult(names: 'list[str]', coef: 'list[float]', se: 'list[float]', vcov: 'np.ndarray', tau2: 'list[float]', Sigma: 'np.ndarray', k: 'int', studies: 'int', excluded: 'dict[str, str]', support: 'tuple[float, float]', method: 'str', level: 'float', test: 'str', basis: 'Basis', per_study: 'list[StudyFit]', fit: 'Any', center: 'float' = 0.0, _cr2: 'Any' = None) -> None
```

A joint pooling of a study-level treatment-effect block.

**Attributes**

- `names` — one name per pooled coefficient, ``["treat", "treat:B[1]", ...]``.
- `coef, se, vcov` — the pooled coefficients and their covariance.
- `tau2` — the diagonal of ``Sigma``, one per coefficient.
- `Sigma` — the between-study covariance matrix of the block.
- `k, studies, excluded` — contributing studies, studies present, failures.
- `support` — ``(min, max)`` of the covariate actually observed.
- `method, level, test` — the second-stage settings.
- `per_study` — one `ipd.StudyFit` per contributing study.
- `fit` — the underlying `rma_mv.RmaMvResult`.

## `ipd_two_stage_curve`

```python
ipd_two_stage_curve(y, *, study, treat, covariate, basis=None, n_knots: 'int' = 4, knots=None, center=None, struct: 'str' = 'UN', family: 'str' = 'binomial', covariates=None, method: 'str' = 'REML', level: 'float' = 0.95, test: 'str' = 'z', min_events: 'int' = 1) -> 'IpdCurveResult'
```

Pool a treatment-by-covariate spline jointly across studies.

**Args**

- `y` — the outcome, one entry per participant.
- `study` — the study label, one entry per participant.
- `treat` — the treatment indicator (0/1), one entry per participant.
- `covariate` — the effect modifier, one entry per participant.
- `basis` — a `basis.Basis` to use instead of building one; ``ma.fp`` gives fractional polynomials, ``ma.rcs`` splines.
- `n_knots` — knots for the default restricted cubic spline.
- `knots` — explicit knot positions.  The same knots are used in every study, which is what makes the coefficients comparable.
- `center` — value the covariate is centred at, by default its median.  The linear coefficient of the block is the treatment effect *at this value*, and without centring it is the effect at covariate = 0, which is usually an extrapolation (age 0, dose 0) and can have a standard error an order of magnitude too wide.  Pass ``0`` to switch centring off.
- `struct` — structure of the between-study covariance.  ``"UN"`` (default) estimates every variance and correlation, which needs a lot of studies; ``"HCS"`` gives each coefficient its own variance but one common correlation, and is what to use when ``k`` is small.  See `IpdCurveResult.diagnostics`.
- `family` — ``"binomial"`` (logistic) or ``"gaussian"`` (linear).
- `covariates` — further participant-level covariates, adjusted for but not pooled.
- `method` — τ² estimator for the second stage.
- `level` — confidence level.
- `test` — ``"z"`` (default) or ``"t"``.  ``"knha"``/``"adhoc"`` are refused: their degrees of freedom are computed from the stacked rows rather than from the studies, which makes the band narrower, not wider.
- `min_events` — studies with fewer events are excluded (no finite logistic estimate otherwise).

**Returns**

`IpdCurveResult`.

**Raises**

- `ValueError` — when the inputs disagree in length, no study can be fitted, or the basis has fewer than two columns.

**Example**

- `>>> fit = ma.ipd_two_stage_curve(  # doctest` — +SKIP
...     y, study=study, treat=treat, covariate=age)
- `>>> fit.test_nonlinearity()["pval"]  # doctest` — +SKIP
- `>>> fit.curve([45, 55, 65]).to_table()  # doctest` — +SKIP

## `linear_model`

```python
linear_model(y, X)
```

Ordinary least squares with the residual variance estimated.

**Args**

- `y` — the outcome.
- `X` — ``(n, p)`` design matrix, intercept column included.

**Returns**

``(b, vb)``.

**Raises**

- `ValueError` — when there are no residual degrees of freedom, or the design is rank deficient.

## `logistic_model`

```python
logistic_model(y, X, *, max_iter: 'int' = 200, tol: 'float' = 1e-10)
```

Maximum-likelihood logistic regression by iteratively reweighted least squares.

**Args**

- `y` — 0/1 outcome.
- `X` — ``(n, p)`` design matrix, intercept column included.
- `max_iter` — iteration cap.
- `tol` — convergence tolerance on the largest absolute coefficient change.

**Returns**

``(b, vb, iterations, converged)``.

**Raises**

- `ValueError` — on a non-binary outcome, a rank-deficient design, or complete separation (a study whose covariate perfectly predicts the outcome has no finite estimate, and inventing one would be worse than saying so).

## `OneStageResult`

```python
OneStageResult(b: 'list[float]', se: 'list[float]', zval: 'list[float]', pval: 'list[float]', ci_lb: 'list[float]', ci_ub: 'list[float]', vb: 'np.ndarray', coefnames: 'list[str]', tau2: 'float', sigma2: 'float | None', loglik: 'float', method: 'str', participants: 'int', studies: 'int', family: 'str', converged: 'bool', level: 'float', test: 'str' = 'z', D: 'np.ndarray | None' = None, re_names: 'list[str]' = <factory>, struct: 'str' = 'UN') -> None
```

A one-stage individual-participant model.

**Attributes**

- `b, se, zval, pval, ci_lb, ci_ub` — fixed effects.
- `vb` — their covariance matrix.
- `coefnames` — names, intercept first.
- `tau2` — between-study variance of the random intercept.
- `sigma2` — residual variance (``None`` for a binary outcome, where it is fixed at :math:`\pi^2/3` by the logistic link).
- `loglik` — the maximised (restricted) log-likelihood.
- `method` — ``"REML"`` or ``"ML"``.
- `participants` — rows used.
- `studies` — number of studies.
- `family` — ``"gaussian"`` or ``"binomial"``.
- `converged` — whether the optimiser reported success.
- `level` — confidence level.
- `test` — always ``"z"`` — a one-stage model's fixed effects are tested against the normal reference unless the user asks otherwise.

## `ipd_one_stage`

```python
ipd_one_stage(y, *, study, treat, covariates=None, family: 'str' = 'gaussian', method: 'str' = 'REML', level: 'float' = 0.95, quadrature: 'int' = 15, max_iter: 'int' = 500, re_slopes=None, struct: 'str' = 'UN') -> 'OneStageResult'
```

Fit one model to every participant, with participants nested in studies.

**Args**

- `y` — the outcome, one entry per participant.
- `study` — the study label, one entry per participant.
- `treat` — the treatment indicator, one entry per participant.
- `covariates` — optional participant-level covariates — a list of arrays (named ``cov1``, …) or a mapping ``{name: array}``.
- `family` — ``"gaussian"`` (linear mixed model, analytic likelihood) or ``"binomial"`` (logistic mixed model, Gauss–Hermite quadrature).
- `method` — ``"REML"`` (default) or ``"ML"``. Ignored for the binomial family, where the likelihood is a true likelihood.
- `level` — confidence level.
- `quadrature` — number of Gauss–Hermite nodes (binomial only).
- `max_iter` — optimiser iteration cap.
- `re_slopes` — names of fixed-effect columns whose coefficients vary randomly across studies, or a ``(participants, q - 1)`` matrix of such columns. The random intercept is always included, so ``re_slopes=["treat"]`` is the lme4 model ``(1 + treat | study)``.
- `struct` — ``"UN"`` (default) for a full between-study covariance of the random terms, or ``"DIAG"`` for independent random effects.

**Returns**

`OneStageResult`.

**Raises**

- `ValueError` — for mismatched lengths, a missing treatment indicator, or an unknown family.

**Example**

- `>>> fit = ma.ipd_one_stage(y, study=study, treat=treat)   # doctest` — +SKIP
- `>>> fit.b[1], fit.tau2, fit.ICC                           # doctest` — +SKIP

## `PredictionResult`

```python
PredictionResult(metric: 'str', yi: 'list[float]', vi: 'list[float]', b: 'float', se: 'float', ci_lb: 'float', ci_ub: 'float', pooled: 'float', pooled_ci_lb: 'float', pooled_ci_ub: 'float', tau2: 'float', QE: 'float', QE_df: 'int', QE_p: 'float', I2: 'float', pi_lb: 'float | None', pi_ub: 'float | None', k: 'int', method: 'str', level: 'float', scale: 'str', fit: 'Any') -> None
```

A pooled performance metric.

**Attributes**

- `metric` — as passed in.
- `yi, vi` — the per-study values on the analysis scale.
- `b, se, ci_lb, ci_ub` — the pooled estimate on the analysis scale.
- `pooled, pooled_ci_lb, pooled_ci_ub` — the same, back-transformed to the scale the studies reported.
- `tau2, QE, QE_df, QE_p, I2` — heterogeneity.
- `pi_lb, pi_ub` — prediction interval, back-transformed.
- `k` — studies pooled.
- `method` — τ² estimator.
- `level` — confidence level.
- `scale` — ``"logit"``, ``"log"`` or ``"identity"``.
- `fit` — the underlying `rma_uni.RmaResult`.

## `auc_logit`

```python
auc_logit(auc, *, n_event, n_nonevent)
```

Logit-transform an AUC and give it the Hanley–McNeil variance.

An AUC is a proportion, so it is pooled on the logit scale — that keeps the
back-transformed interval inside [0, 1] instead of letting it run past 1.
Its standard error comes from Hanley & McNeil (1982),

.. math::

    \mathrm{Var}(A) = \frac{A(1-A) + (n_1-1)(Q_1 - A^2)
                            + (n_0-1)(Q_2 - A^2)}{n_1 n_0},

with :math:`Q_1 = A/(2-A)` and :math:`Q_2 = 2A^2/(1+A)`, mapped to the logit
scale by the delta method. Note what this means: two studies reporting the
same AUC have different variances if their event counts differ.

**Args**

- `auc` — the reported C-statistic(s), strictly inside (0, 1).
- `n_event` — number of participants with the outcome.
- `n_nonevent` — number without it.

**Returns**

``(yi, vi)`` on the logit scale.

**Raises**

- `ValueError` — for an AUC outside (0, 1), or non-positive group sizes.

**Example**

>>> yi, vi = ma.auc_logit(0.75, n_event=100, n_nonevent=400)

## `oe_ratio`

```python
oe_ratio(observed, expected)
```

Log observed-to-expected ratio and its Poisson variance.

**Args**

- `observed` — observed events.
- `expected` — expected events under the model.

**Returns**

``(yi, vi)`` with ``yi = log(O / E)`` and ``vi = 1/O + 1/E``.

**Raises**

- `ValueError` — for a non-positive count.

**Example**

>>> yi, vi = ma.oe_ratio(120, 100)      # log(1.2), 1/120 + 1/100

## `prediction_meta`

```python
prediction_meta(yi, vi, *, metric: 'str', method: 'str' = 'REML', level: 'float' = 0.95, test: 'str' = 'z') -> 'PredictionResult'
```

Pool a prediction-model performance metric on its own scale.

**Args**

- `yi` — the transformed values — from `auc_logit` or `oe_ratio`, or the reported values themselves for a calibration slope/intercept or a Brier score.
- `vi` — their variances.
- `metric` — ``"auc"``, ``"oe_ratio"``, ``"calibration_slope"``, ``"calibration_intercept"`` or ``"brier"``. It decides the back-transformation, nothing else.
- `method` — τ² estimator.
- `level` — confidence level.
- `test` — reference distribution.

**Returns**

`PredictionResult`.

**Raises**

- `ValueError` — for an unknown metric or mismatched inputs.

**Example**

>>> yi, vi = ma.auc_logit([0.75, 0.80, 0.72], n_event=[100, 60, 140],
...                       n_nonevent=[400, 300, 500])
- `>>> ma.prediction_meta(yi, vi, metric="auc").pooled     # doctest` — +SKIP

## `KmResult`

```python
KmResult(time: 'np.ndarray', n_risk: 'np.ndarray', n_event: 'np.ndarray', n_censored: 'np.ndarray', surv: 'np.ndarray', se: 'np.ndarray', n: 'int') -> None
```

A Kaplan–Meier curve, with the risk sets it came from.

**Attributes**

- `time` — the event times, excluding time 0.
- `n_risk` — number at risk just before each of those times.
- `n_event` — events at each of those times.
- `n_censored` — censored just after each of those times.
- `surv` — the survival estimate at each time (``S(0) = 1`` implied).
- `se` — Greenwood standard error of ``surv``.
- `n` — number of participants.

## `ReconstructedIpD`

```python
ReconstructedIpD(time: 'np.ndarray', event: 'np.ndarray', n: 'int', n_event: 'int', km: 'KmResult') -> None
```

Individual times and event indicators rebuilt from a KM curve.

**Attributes**

- `time` — reconstructed follow-up time per participant.
- `event` — 1 for an event, 0 for a censoring.
- `n` — number of participants.
- `n_event` — number of events.
- `km` — the `KmResult` the reconstruction reproduces.

## `RmstResult`

```python
RmstResult(rmst: 'float', se: 'float | None', ci_lb: 'float | None', ci_ub: 'float | None', tau: 'float', level: 'float', surv: 'np.ndarray' = <factory>, time: 'np.ndarray' = <factory>, truncated: 'bool' = False) -> None
```

Restricted mean survival time up to a horizon.

**Attributes**

- `rmst` — the estimate.
- `se` — standard error (``None`` when the risk sets were not supplied).
- `ci_lb, ci_ub` — confidence interval (``None`` alongside ``se``).
- `tau` — the horizon.
- `level` — confidence level.
- `surv` — the survival values used.
- `time` — the times used.
- `truncated` — whether ``tau`` fell before the last reported time.

## `km_curve`

```python
km_curve(time, event) -> 'KmResult'
```

Kaplan–Meier survivor function from individual times.

**Args**

- `time` — follow-up time per participant.
- `event` — 1 for an event, 0 for a censoring.

**Returns**

`KmResult`.

**Raises**

- `ValueError` — for mismatched lengths, negative times, or an event indicator that is not 0/1.

**Example**

>>> km = ma.km_curve([1, 2, 2, 5], [1, 1, 0, 1]) >>> [round(v, 4) for v in km.surv] [0.75, 0.5, 0.5]

## `km_reconstruct`

```python
km_reconstruct(time, n_risk, n_event, *, total: 'int | None' = None) -> 'ReconstructedIpD'
```

Rebuild individual data from a published Kaplan–Meier summary.

**Args**

- `time` — the reported time points (strictly increasing, all positive).
- `n_risk` — number at risk just before each time point.
- `n_event` — events at each time point.
- `total` — participants at risk at time 0. Defaults to ``n_risk[0]``.

Censoring is spread evenly across the interval that produced it. It has to
fall strictly inside the interval: placing it at the interval's end would
put those participants back into the next risk set and change the curve.

**Returns**

`ReconstructedIpD`. Feeding ``time``/``event`` to `km_curve` reproduces ``n_risk``, ``n_event`` and the survival values implied by the input.

**Raises**

- `ValueError` — for inconsistent risk sets — a value that increases, or fewer at risk than the events plus the next risk set.

**Example**

>>> ipd = ma.km_reconstruct([2, 5, 9], [36, 30, 21], [4, 5, 6], total=40) >>> ipd.n, ipd.n_event (40, 15)

## `rmst`

```python
rmst(time, surv, *, tau: 'float | None' = None, n_risk=None, n_event=None, level: 'float' = 0.95) -> 'RmstResult'
```

Restricted mean survival time: the area under the survival curve.

**Args**

- `time` — the times at which the curve is reported (event times).
- `surv` — the survival estimate at each of those times. ``S(0) = 1`` is assumed and must not be included.
- `tau` — the horizon. Defaults to the last reported time.
- `n_risk` — number at risk just before each time; supplying it (with ``n_event``) adds the standard error.
- `n_event` — events at each time.
- `level` — confidence level.

**Returns**

`RmstResult`.

**Raises**

- `ValueError` — for mismatched lengths, times that are not increasing, or ``tau`` beyond the data with no curve to integrate.

**Example**

- `>>> ma.rmst([12, 24], [0.8, 0.6], tau=24).rmst      # doctest` — +SKIP

## `piecewise_exp`

```python
piecewise_exp(time, event, *, study, treat, breaks, method='REML', level=95.0, min_events=1)
```

Pooled interval-specific hazard ratios from participant-level data.

Each study is fitted separately by Poisson regression of the split
follow-up, with the interval as a stratifying factor and an
interval-by-treatment interaction.  The interval-specific log hazard
ratios are then pooled with a random-effects model per interval.

**Args**

- `time` — follow-up time per participant.
- `event` — 1 if the event was observed, 0 if censored.
- `study` — study identifier per participant.
- `treat` — treatment indicator per participant (0/1).
- `breaks` — interval boundaries, starting at 0 and strictly increasing.
- `method` — between-study variance estimator for the pooling.
- `level` — confidence level in percent (``95``) or as a proportion (``0.95``).
- `min_events` — an interval with fewer events than this is not fitted for that study; it stays ``NaN`` instead of returning a coefficient at minus infinity.

**Returns**

`PiecewiseResult`.

**Raises**

- `ValueError` — mismatched lengths, a treatment indicator that is not 0/1, or fewer than two studies.

## `piecewise_joint`

```python
piecewise_joint(time, event, *, study, treat, breaks, rho=None, method='REML', level=95.0, min_events=1, struct='UN')
```

Pool the interval-specific hazard ratios as one vector.

The difference from `piecewise_exp` is what is being estimated.  That
function pools each interval with its own univariate model: every reported
hazard ratio is valid on its own, and any statement comparing two intervals
is informal.  This one stacks a study's interval-specific log hazard ratios
into a vector, gives each study its own sampling covariance, puts a
between-study covariance on top and fits them together —so
`PiecewiseJointResult.trend_test` and
`predict_region` have the covariance they need.

**Args**

- `time, event, study, treat, breaks, min_events` — as in `piecewise_exp`.
- `rho` — within-study correlation between a study's interval-specific log hazard ratios.  ``None`` uses the covariance the per-study Poisson fits supply, which for this parameterisation is exactly diagonal — the intervals partition the follow-up, so the information matrix is block diagonal.  A non-zero value is for *aggregate* input, where the reported hazard ratios come from a model that does not have that structure and the correlation cannot be estimated from the data; it is an assumption, and it belongs in a sensitivity analysis.
- `method` — ``"REML"`` or ``"ML"``.
- `level` — confidence level.
- `struct` — between-study covariance structure passed to `rma_mv`; ``"UN"`` needs ``k(k+1)/2`` parameters for ``k`` intervals.

**Returns**

`PiecewiseJointResult`.

**Raises**

- `ValueError` — on mismatched lengths, a treatment indicator that is not 0/1, fewer than two studies, or a ``rho`` outside ``[-1, 1]``.

## `PiecewiseJointResult`

```python
PiecewiseJointResult(breaks: 'np.ndarray', labels: 'list[str]', beta: 'np.ndarray', se: 'np.ndarray', ci_lb: 'np.ndarray', ci_ub: 'np.ndarray', vcov: 'np.ndarray', Sigma: 'np.ndarray', tau2: 'np.ndarray', k: 'list[int]', studies: 'list', model: 'object' = None, dropped: 'list' = <factory>, rho: 'float | None' = None, method: 'str' = 'REML', level: 'float' = 95.0, struct: 'str' = 'UN') -> None
```

Interval-specific hazard ratios pooled as a vector.

Pooling each interval with its own univariate model gives valid
interval-specific estimates but makes every *comparison* between intervals
an informal one: the estimates share the between-study covariance, so
"the hazard ratio falls after six months" needs the joint covariance.

**Attributes**

- `breaks, labels` — the intervals.
- `beta, se, ci_lb, ci_ub` — the pooled log hazard ratio per interval.
- `vcov` — their joint covariance matrix.
- `Sigma` — the estimated between-study covariance of the interval-specific log hazard ratios, on the log scale.
- `tau2` — its diagonal.
- `k` — studies contributing per interval.
- `rho` — the within-study correlation that was imposed on the sampling covariance, or ``None`` when the per-study fits supplied it.
- `model` — the underlying `RmaMvResult`, so `predict_region` can be applied to it.
- `studies, dropped` — as in `PiecewiseResult`.
- `method, level, struct` — settings.

## `PiecewiseResult`

```python
PiecewiseResult(breaks: 'np.ndarray', labels: 'list[str]', beta: 'np.ndarray', se: 'np.ndarray', zval: 'np.ndarray', pval: 'np.ndarray', ci_lb: 'np.ndarray', ci_ub: 'np.ndarray', tau2: 'np.ndarray', QE: 'np.ndarray', QEp: 'np.ndarray', k: 'list[int]', log_hr: 'np.ndarray', se_hr: 'np.ndarray', studies: 'list', cov_hr: 'np.ndarray | None' = None, fits: 'list' = <factory>, dropped: 'list' = <factory>, method: 'str' = 'REML', level: 'float' = 95.0) -> None
```

Interval-specific hazard ratios pooled across studies.

**Attributes**

- `breaks` — the interval boundaries that were used.
- `labels` — a readable label per interval, e.g. ``"0-6"``.
- `beta` — pooled log hazard ratio per interval.
- `se` — its standard error.
- `zval, pval, ci_lb, ci_ub` — inference on each interval.
- `tau2` — between-study variance per interval.
- `QE, QEp` — residual heterogeneity per interval.
- `k` — number of studies contributing to each interval.
- `log_hr` — ``(n_study, n_interval)`` per-study log hazard ratios, ``NaN`` where a study could not be fitted.
- `se_hr` — the matching standard errors.
- `cov_hr` — ``(n_study, n_interval, n_interval)`` the covariance of each study's log hazard ratios.  Exactly diagonal, because the intervals partition the follow-up and a Poisson fit with a saturated interval-by-treatment interaction therefore has a block-diagonal information matrix; `piecewise_joint` checks that rather than trusting it.
- `studies` — study labels, in the row order of ``log_hr``.
- `fits` — the pooling fits, one per interval.
- `dropped` — ``(study, reason)`` for every study that could not be used; nothing is dropped silently.
- `method, level` — settings.

## `poisson_irls`

```python
poisson_irls(y, X, offset=None, max_iter=200, tol=1e-10)
```

Poisson regression by iteratively reweighted least squares.

**Args**

- `y` — counts.
- `X` — design matrix (``n x p``).
- `offset` — optional additive term on the log scale, e.g. ``log(exposure)``.
- `max_iter` — iteration cap.
- `tol` — convergence threshold on the largest coefficient change.

**Returns**

``(beta, vcov, iterations, converged)``.

**Raises**

- `ValueError` — mismatched shapes, negative counts, a rank-deficient design, or a non-finite fit.

## `split_followup`

```python
split_followup(time, event, breaks)
```

Split each participant's follow-up at the interval boundaries.

**Args**

- `time` — follow-up time per participant.
- `event` — 1 if the event was observed, 0 if censored.
- `breaks` — interval boundaries, starting at 0.  Follow-up is split into ``len(breaks) - 1`` intervals; the last one is open-ended.

**Returns**

``(rows, intervals, exposure, events)`` —``rows`` indexes the input participant, ``intervals`` is 0-based, ``exposure`` the time spent in that interval and ``events`` whether the event fell in it.

**Raises**

- `ValueError` — the breaks are not strictly increasing or do not start at zero, or a time is negative.

## `fp`

```python
fp(x, powers, scale: 'float | None' = None, center: 'bool' = False, name: 'str | None' = None) -> 'Basis'
```

Fractional-polynomial basis (Royston & Sauerbrei) for a moderator.

**Args**

- `x` — the moderator, strictly positive, 1-D.
- `powers` — the powers, e.g. ``[1, 2]``, ``[-1, 0]``, ``[0.5, 0.5]``. They are sorted (the literature always lists them ascending); a repeated power contributes a second term multiplied by ``log(x)``.
- `scale` — divides ``x`` before the transform. ``None`` (default) means 1, i.e. no rescaling. Scaling does not change the fit, only the coefficients.
- `center` — subtract each column's mean. Also fit-invariant; it only makes the intercept interpretable at the mean of the transformed values.
- `name` — label for the term; defaults to ``"fp"``.

**Returns**

`Basis` with one column per power.

**Raises**

- `ValueError` — for an empty power list, non-positive ``x``, or a missing value.

**Example**

>>> b = fp([1, 2, 3, 4], powers=[0, 1])      # log(x) and x
- `>>> [round(v, 4) for v in b.values[` — , 0]]
[0.0, 0.6931, 1.0986, 1.3863]

## `nonlinear_test`

```python
nonlinear_test(result, term: 'str') -> 'NonlinearTestResult'
```

Test whether the effect of a basis term is non-linear.

Runs two joint Wald tests over the columns that belong to ``term``: one
over all of them (``overall``) and one over all but the first
(``nonlinear``). The convention that the first column is the linear term
holds for both ``rcs`` and ``fp``, where the first power is the linear one
whenever the model has it.

**Args**

- `result` — a fitted model that used a basis as moderator.
- `term` — the basis label, e.g. ``"rcs(dose,4)"`` or ``"fp(dose,[1,2])"``.

**Returns**

`NonlinearTestResult`.

**Raises**

- `ValueError` — when no coefficient belongs to ``term``.

**Example**

- `>>> fit = ma.rma(yi, vi, mods="~ rcs(dose, 4)", data=df)   # doctest` — +SKIP
- `>>> print(ma.nonlinear_test(fit, "rcs(dose,4)"))           # doctest` — +SKIP

## `predict_curve`

```python
predict_curve(result, basis: 'Basis', grid=None, at: 'dict | None' = None, level: 'float | None' = None) -> 'CurveResult'
```

Evaluate a fitted basis term over a grid, with bands.

This is a thin wrapper around ``result.predict(newmods=...)``: the only
thing it adds is rebuilding the design matrix from a `Basis`, and
holding the *other* moderators at fixed values.

**Args**

- `result` — a fitted `rma_uni.RmaResult` whose moderators include ``basis``.
- `basis` — the basis used in the fit (or one rebuilt with the same knots).
- `grid` — values of the moderator to evaluate at. Defaults to 100 points spanning the observed range.
- `at` — values for the other moderators, by coefficient name. Anything not given is held at 0, which is the model's reference point.
- `level` — confidence level; defaults to the level of the fit.

**Returns**

`CurveResult`.

**Raises**

- `ValueError` — when no coefficient belongs to the basis, when ``at`` names an unknown moderator, or when a moderator is left unset.

**Example**

- `>>> b = ma.rcs(df["dose"], nk=4, name="rcs(dose,4)")     # doctest` — +SKIP
- `>>> fit = ma.rma(df["yi"], df["vi"], mods=b)             # doctest` — +SKIP
- `>>> curve = ma.predict_curve(fit, b)                     # doctest` — +SKIP

## `rcs`

```python
rcs(x, nk: 'int | None' = 4, df: 'int | None' = None, knots=None, name: 'str | None' = None) -> 'Basis'
```

Restricted cubic spline basis (Harrell) for a moderator.

**Args**

- `x` — the moderator, 1-D.
- `nk` — number of knots; 4 by default, which gives 3 columns.
- `df` — degrees of freedom — the number of *columns* — as an alternative to ``nk``. ``df=3`` is the same as ``nk=4``.
- `knots` — explicit knot positions. When given, ``nk`` and ``df`` are ignored.
- `name` — label for the term; defaults to ``"rcs"``. ``design_matrix`` passes the call itself, so a fitted coefficient is named ``rcs(dose,4)[2]``.

**Returns**

- ``Basis` with ``nk - 1`` columns` — ``x`` followed by the
``nk - 2`` non-linear terms.

**Raises**

- `ValueError` — for fewer than 3 knots, fewer than 4 distinct values, or knots that are not strictly increasing.

**Example**

- `>>> b = rcs([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], nk=4)   # doctest` — +SKIP
- `>>> b.shape                                          # doctest` — +SKIP
(10, 3)

## `wald_test`

```python
wald_test(result, btt: 'Sequence[int] | None' = None, names: 'Sequence[str] | None' = None) -> 'WaldResult'
```

*Mirrors R's `anova`.*

Joint Wald test for a subset of coefficients (``metafor::anova``).

**Args**

- `result` — a fitted `rma_uni.RmaResult` (or anything with ``b``/``beta``, ``vb``, ``test``, ``s2w`` and ``ddf``).
- `btt` — **1-based** coefficient positions, exactly as in ``metafor::anova(fit, btt=)``. Note the difference from Python indexing: ``btt=[2, 3]`` tests the second and third coefficients.
- `names` — coefficient names instead of positions. Matching is exact, so pass the names ``result.coefnames`` reports.

**Returns**

`WaldResult`.

**Raises**

- `ValueError` — when nothing is selected, or a position is out of range.

**Example**

- `>>> fit = ma.rma(yi, vi, mods="~ x", data=df)    # doctest` — +SKIP
- `>>> ma.wald_test(fit, btt=[2])                    # doctest` — +SKIP

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
rma_mv(yi, vi=None, V=None, mods=None, data=None, random: 'str' = '~ 1 | study', struct: 'str' = 'CS', rho=None, method: 'str' = 'REML', level: 'float' = 95.0, test: 'str' = 'z', dfs: 'str | np.ndarray' = 'residual', intercept: 'bool' = True, sigma2=None, slab=None, cluster=None, warn_missing: 'bool' = True) -> 'RmaMvResult'
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
- `warn_missing` — emit a warning naming the studies dropped because a value is missing (R's ``na.action`` drops them silently).

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
robust(result: 'Any', cluster=None, adjust: 'bool | str' = True, clubSandwich: 'bool' = False, vcov: 'str' = 'CR2', coef_test: 'str' = 'Satterthwaite', level: 'float | None' = None)
```

*Mirrors R's `rma.mv`.*

Cluster-robust inference for an ``rma`` or ``rma.mv`` fit.

**Args**

- `result` — a fitted result.  Both the univariate and the multilevel model are accepted; the sandwich is the same, only the marginal var-cov matrix differs.
- `cluster` — cluster label per estimate.  Required for a univariate fit (``robust.rma.uni`` has no default either); a multilevel fit reuses the ``cluster=`` it was fitted with when omitted.
- `adjust` — small-sample adjustment for the metafor sandwich. ``True`` (default) is CR1 (``n/(n-p)``), ``False`` is CR0, ``"Stata"``/``"Stata1"`` is CR1.S1 (``n/(n-1) * (k-1)/(k-p)``) and ``"Stata2"`` is CR1.S2 (``n/(n-1)``).  Ignored when ``clubSandwich=True``.
- `clubSandwich` — use the CR2/CR3 bias-reduced sandwich and the Satterthwaite degrees of freedom instead of metafor's own.
- `vcov` — ``"CR2"`` (default) or ``"CR3"``; only read when ``clubSandwich=True``.
- `coef_test` — ``"Satterthwaite"`` (default), ``"naive-t"``, ``"naive-tp"`` or ``"z"``.
- `level` — confidence level; ``None`` reuses the fit's.  Both ``0.95`` and ``95`` are accepted.

**Returns**

A dict with ``vbest``, ``beta``, ``se``, ``zval``, ``pval``, ``ci_lb``, ``ci_ub``, ``dfs``, ``n``, ``vb`` and -- for univariate fits -- the omnibus test ``QM``/``QMdf``/``QMp``.  The omnibus entries are ``None`` on the clubSandwich path, where clubSandwich's own small-sample reference distribution would be needed to report them honestly.

**Raises**

- `ValueError` — when no cluster is supplied, when the cluster vector does not match the number of estimates, or when there are not more clusters than coefficients.

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
rma(yi, vi=None, sei=None, mods=None, data=None, slab=None, measure: 'str' = 'GEN', method: 'str' = 'REML', test: 'str' = 'z', level: 'float' = 0.95, btt=None, digits: 'int' = 4, scale=None, scale_intercept: 'bool' = True, link: 'str' = 'log', optbeta: 'bool' = False, alpha=None, warn_missing: 'bool' = True) -> 'RmaResult | RmaLsResult'
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
- `scale, scale_intercept, link, optbeta, alpha` — when ``scale`` is given a location-scale model is fitted instead (``rma(yi, vi, mods=~x, scale=~z)`` in R, class ``rma.ls``); the fit is delegated to `rma_ls` and an `RmaLsResult` is returned.
- `warn_missing` — emit a `missing.MissingDataWarning` naming the studies dropped because ``yi``, ``vi`` or a moderator is missing.  R drops them silently through ``na.action``; pass ``False`` to drop them silently here too.

## `RmaResult`

```python
RmaResult(b: 'list[float]', se: 'list[float]', zval: 'list[float]', pval: 'list[float]', ci_lb: 'list[float]', ci_ub: 'list[float]', vb: 'np.ndarray', tau2: 'float', se_tau2: 'float | None', tau2_fix: 'bool', I2: 'float', H2: 'float', R2: 'float | None', vt: 'float', QE: 'float', QEp: 'float', QM: 'float', QMdf: 'list', QMp: 'float', k: 'int', k_f: 'int', k_eff: 'int', k_all: 'int', p: 'int', p_eff: 'int', parms: 'int', int_only: 'bool', int_incl: 'bool', yi: 'list[float]', vi: 'list[float]', X: 'np.ndarray', mods: 'np.ndarray | None', weights: 'list[float]', weights_pct: 'list[float]', fitted: 'list[float]', residuals: 'list[float]', method: 'str', test: 'str', dfs: 'float | None', ddf: 'float | None', s2w: 'float', measure: 'str', slab: 'list[str]', not_na: 'list[bool]', level: 'float', coefnames: 'list[str] | None' = None, btt: 'list[int] | None' = None) -> None
```

*Mirrors R's `rma.uni`.*

Fitted ``rma.uni`` model, mirroring the R ``$`` field set.

## `selmodel`

```python
selmodel(x: 'RmaResult', type: 'str', alternative: 'str' = 'greater', prec=None, subset=None, delta=None, steps=None, decreasing: 'bool' = False, verbose: 'bool' = False, digits: 'int' = 4, control: 'dict | None' = None, **kwargs) -> 'SelModelResult'
```

Fit a publication-bias selection model on an existing ``rma`` fit.

**Args**

- `x` — fitted ``RmaResult`` (from `rma`).
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

`SelModelResult` with the full metafor ``$`` field set.

## `SelModelResult`

```python
SelModelResult(beta: 'list[float]', b: 'list[float]', se: 'list[float]', zval: 'list[float]', pval: 'list[float]', ci_lb: 'list[float]', ci_ub: 'list[float]', vb: 'np.ndarray', tau2: 'float', se_tau2: 'float', ci_lb_tau2: 'float', ci_ub_tau2: 'float', delta: 'list[float]', vd: 'np.ndarray', se_delta: 'list[float]', zval_delta: 'list[float]', pval_delta: 'list[float]', ci_lb_delta: 'list[float]', ci_ub_delta: 'list[float]', QM: 'float', QMp: 'float', ll: 'float', ll0: 'float', LRT: 'float', LRTdf: 'int', LRTp: 'float', LRT_tau2: 'float', LRTp_tau2: 'float', parms: 'int', fit_stats: 'dict', pvals: 'list[float]', type: 'str', alternative: 'str', steps: 'list[float]', pgrp: 'list', k0: 'int', k1: 'int', model: 'str', k: 'int', p: 'int', tau2_fix: 'bool', method: 'str', yi: 'list[float]', vi: 'list[float]', level: 'float', _model: 'Any' = None) -> None
```

*Mirrors R's `rma.uni.selmodel`.*

Fitted ``rma.uni.selmodel`` model, mirroring the R ``$`` field set.

## `SelProfileResult`

```python
SelProfileResult(delta, ll, maxll, delta_hat, ci_lb, ci_ub, level, type_, k)
```

*Mirrors R's `profile.rma.uni.selmodel`.*

Result of `profile_selmodel` (``profile.rma.uni.selmodel``).

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

- `result` — `SelModelResult` from `selmodel`.
- `delta` — explicit grid (default: ``n_points`` values spanning ``delta_hat +/- span * |delta_hat|``, or +/-2 when ``delta_hat == 0``).
- `n_points` — grid size when ``delta`` is not given.
- `span` — half-width of the default grid, as a multiple of ``|delta_hat|``.
- `level` — confidence level for the profile limits.

**Returns**

`SelProfileResult`.

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

- `x` — a `RmaLsResult`.
- `alpha` — 1-based index of the scale coefficient to profile (metafor's ``alpha=``).  Defaults to the first non-fixed coefficient.
- `steps` — number of grid points, or an explicit sequence of values.
- `xlim` — profiling range; by default ``vc ± qnorm(0.995) * se.alpha`` around the estimate, like metafor.
- `plot` — draw the profile plot immediately.
- `**kw` — forwarded to `plotting.plot_mv_profile`.

**Returns**

`RmaLsProfileResult`.

## `RmaLsProfileResult`

```python
RmaLsProfileResult(component, index, alpha, ll, maxll, beta, ci_lb, ci_ub, vc, method, xlab, title, comps=1)
```

*Mirrors R's `profile.rma.ls`.*

Result of `profile_rma_ls` (mirrors ``profile.rma.ls`` objects).

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
`profile_rma_ls` and invert the profile yourself for those.

## `vcalc`

```python
vcalc(vi, cluster=None, subgroup=None, obs=None, type=None, time1=None, time2=None, grp1=None, grp2=None, w1=None, w2=None, rho=None, phi=None, rho_names=None, r_compat=False, check_pd=True)
```

Build the ``k x k`` sampling covariance matrix ``V``.

**Args**

- `vi` — sampling variances, length ``k``.
- `cluster` — cluster identifier (usually the study).  ``None`` treats every row as one cluster, which is a convenience pimeta adds; R requires the argument.
- `subgroup` — optional independent subgroups within a cluster; rows in different subgroups are independent even inside one cluster.
- `obs` — optional identifier of the *observation* of one construct, so several measures of the same thing can be told apart.
- `type` — optional identifier of the *construct*, when several different outcomes were measured on the same subjects.
- `time1` — the time point at which each effect size was measured.  With ``phi`` this turns the shared-arm correlation into an autocorrelation, and two estimates measured at the same time are perfectly correlated.
- `time2` — the time of the *second* condition a row compares, for a design in which the two conditions were measured at different times. Requires ``grp2``.
- `grp1, grp2` — the two conditions each effect size compares.  Rows that share one of them are correlated, with the correlation derived from the shared arm's weight.
- `w1, w2` — weights (group sizes, or inverse sampling variances) of the two conditions; default 1 each, which makes the correlation exactly 0.5.
- `rho` — correlation among estimates of the same construct (with ``obs``), among different constructs (with ``type``), or among all rows of a cluster (neither given).  A scalar, or a square matrix indexed by the factor's levels.
- `phi` — autocorrelation between two measurements one time unit apart, in ``[-1, 1]``.  Required when ``time1`` is given; two estimates at times $t_i$ and $t_j$ then correlate ``phi ** |t_i - t_j|``.
- `rho_names` — labels of the rows of a matrix ``rho``, in matrix order. Without it the matrix is indexed in first-appearance order of the factor, which is what R's ``unique()`` produces.
- `r_compat` — reproduce metafor exactly, including its silent discard of ``rho`` when neither ``obs`` nor ``type`` is given.  Default ``False``: the correlation that was asked for is the one used.
- `check_pd` — warn when a cluster's covariance block is not positive definite, as metafor's ``checkpd`` does.  ``phi = 1`` and ``r_compat`` both produce such a block.

**Returns**

``(k, k)`` float array.

**Raises**

- `ValueError` — inconsistent lengths, a forgotten correlation, a ``phi`` outside ``[-1, 1]``, a time argument without ``phi``, or a row that compares a condition with itself.

## `shared_control_corr`

```python
shared_control_corr(w_shared, w_a, w_b)
```

Correlation between two contrasts that share one condition.

Effect size ``a`` contrasts arm A against the shared arm C, effect size
``b`` contrasts arm B against the same C.  With homogeneous within-arm
variance, ``Cov = sigma^2/w_C`` and ``Var = sigma^2/w_A + sigma^2/w_C``,
so the variance cancels and the correlation is a function of the weights
alone.

**Args**

- `w_shared` — weight (or group size) of the shared condition.
- `w_a` — weight of the first contrast's other condition.
- `w_b` — weight of the second contrast's other condition.

**Returns**

The correlation in ``[0, 1]``; ``0.5`` when all three weights agree.

## `tau_cor`

```python
tau_cor(fit)
```

Between-outcome correlation matrix of a fitted ``rma_mv`` model.

**Args**

- `fit` — `rma_mv.RmaMvResult` from a model with a multivariate random term, e.g. ``random="~ outcome | study", struct="UN"``.

**Returns**

`TauCorResult`.

**Raises**

- `ValueError` — the model has no multivariate random term, so there is no between-outcome covariance to report.

## `TauCorResult`

```python
TauCorResult(names: 'list[str]', G: 'np.ndarray', R: 'np.ndarray', tau2: 'np.ndarray', rho: 'np.ndarray') -> None
```

Between-outcome covariance and correlation of the random effects.

**Attributes**

- `names` — outcome labels, in the order of the model's inner factor.
- `G` — the ``J x J`` random-effects covariance matrix.
- `R` — the corresponding correlation matrix.
- `tau2` — ``diag(G)``.
- `rho` — the off-diagonal correlations, lower triangle, row by row — the order metafor reports in ``$rho``.

## `rho_sensitivity`

```python
rho_sensitivity(yi, vi, *, study, outcome, mods=None, struct='UN', method='REML', rho_grid=(0.0, 0.25, 0.5, 0.75), level=95.0, test='z')
```

Refit a multivariate model along a grid of within-study correlations.

**Args**

- `yi` — effect estimates, length ``k``.
- `vi` — sampling variances, length ``k``.
- `study` — study identifier; defines the blocks of ``V``.
- `outcome` — outcome identifier; the inner factor of the random term.
- `mods` — fixed-effect design — a design matrix, a formula string such as ``"~ outcome - 1"``, or ``None`` for an intercept-only model.
- `struct` — variance structure passed to ``rma_mv``; ``"UN"`` estimates a free between-outcome correlation, ``"CS"`` one common value.
- `method` — ``"REML"`` (default) or ``"ML"``.
- `rho_grid` — the assumed within-study correlations to fit.
- `level` — confidence level in percent.
- `test` — ``"z"`` (default), ``"t"``, ``"knha"`` or ``"adhoc"``.

**Returns**

`RhoSensitivityResult`.

**Raises**

- `ValueError` — an empty grid, mismatched lengths, or a design that cannot be resolved.

## `RhoSensitivityResult`

```python
RhoSensitivityResult(rho: 'np.ndarray', names: 'list[str]', estimate: 'np.ndarray', se: 'np.ndarray', ci_lb: 'np.ndarray', ci_ub: 'np.ndarray', tau2: 'np.ndarray', fits: 'list' = <factory>) -> None
```

Pooled estimates along a grid of within-study correlations.

**Attributes**

- `rho` — the grid that was fitted.
- `names` — coefficient labels.
- `estimate` — ``(G, p)`` pooled estimates.
- `se` — ``(G, p)`` standard errors.
- `ci_lb, ci_ub` — ``(G, p)`` interval bounds.
- `tau2` — ``(G, q)`` variance components (``q`` depends on ``struct``).
- `fits` — the individual model fits, in grid order.

## `confint_rma_mv`

```python
confint_rma_mv(result, *, tau2=None, rho=None, level=None, tol=0.0001220703125, r_compat=False)
```

Profile-likelihood interval for one ``tau2`` or ``rho`` of an ``rma_mv``.

**Args**

- `result` — a `RmaMvResult` from `rma_mv`.
- `tau2` — 1-based index of the between-study variance component to profile, counting only the ``tau2`` components.
- `rho` — 1-based index of the correlation component to profile.
- `level` — confidence level, accepted as ``95``, ``0.95`` or ``0.05``; defaults to the level the model was fitted at.
- `tol` — convergence tolerance handed to the root finder.  The default is R's ``uniroot`` default, so the endpoint matches metafor's.
- `r_compat` — profile from the fitted values only, as metafor does.  The default re-starts the refit from the variance boundaries as well, because a component that collapses to zero can leave the likelihood higher than the interior branch and metafor's interval would then exclude a point with a larger profile likelihood. ``True`` reproduces metafor; ``False`` is the supremum over the parameter space.  See ``docs/reference/differences.md`` §4.

**Returns**

`ConfintMvResult`, whose ``curve`` holds every point the search evaluated.

**Raises**

- `ValueError` — when neither or both of ``tau2`` and ``rho`` are given, when the model has no such component, or when the index is out of range.

**Example**

>>> fit = ma.rma_mv(yi, V=V, mods=X, random="~ outcome | study", ...                 struct="UN") >>> ma.confint_rma_mv(fit, rho=1)

## `ConfintMvResult`

```python
ConfintMvResult(kind: 'str', index: 'int', estimate: 'float', ci_lb: 'float', ci_ub: 'float', level: 'float', ll_max: 'float', crit: 'float', method: 'str', at_bound_lb: 'bool' = False, at_bound_ub: 'bool' = False, curve: 'list' = <factory>) -> None
```

A profile-likelihood interval for one variance component.

## `predict_region`

```python
predict_region(result, *, level=None, observation=None, test='z', df=None) -> 'PredictionRegion'
```

Joint prediction region for a new study.

**Args**

- `result` — a `RmaMvResult` with a multivariate random term (``random="~ outcome | study"`` with ``struct="UN"`` or similar).
- `level` — confidence level, accepted as ``95``, ``0.95`` or ``0.05``; defaults to the level the model was fitted with.
- `observation` — the new study's sampling covariance (``J x J``), when the question is where its *observed* estimates would fall rather than where its true effects would.
- `test` — ``"z"`` (default) for the asymptotic :math:`\chi^2_J` reference, or ``"t"`` for the Hotelling :math:`F` small-sample one.
- `df` — denominator degrees of freedom for ``test="t"``; defaults to ``k - p``.  Ignored by ``test="z"``.

**Returns**

`PredictionRegion`.

**Raises**

- `ValueError` — when the fit has no between-study covariance matrix, when ``observation`` has the wrong shape, when that matrix is not positive definite, or when ``test="t"`` is asked for with too few degrees of freedom to form the reference.

Warns:
    RuntimeWarning: with ``test="z"`` and 20 or fewer residual degrees of
        freedom, quoting the measured coverage at small ``k``.

**Example**

- `>>> r = ma.predict_region(fit)                 # doctest` — +SKIP
- `>>> r.contains([-0.3, 0.4])                    # doctest` — +SKIP
- `>>> r.marginal()                               # doctest` — +SKIP

## `PredictionRegion`

```python
PredictionRegion(names: 'list[str]', center: 'np.ndarray', cov: 'np.ndarray', crit: 'float', level: 'float', k: 'int', for_observation: 'bool' = False, test: 'str' = 'z', df: 'list[float] | None' = None) -> None
```

An ellipsoidal prediction region for a new study's effect vector.

## `DtaData`

```python
DtaData(labels: 'list[str]', tp: 'np.ndarray', fn: 'np.ndarray', fp: 'np.ndarray', tn: 'np.ndarray', sens: 'np.ndarray', fpr: 'np.ndarray', var_sens: 'np.ndarray', var_fpr: 'np.ndarray', n_positive: 'np.ndarray', n_negative: 'np.ndarray', corrected: 'np.ndarray', correction: 'float' = 0.5, sens_raw: 'np.ndarray | None' = None, fpr_raw: 'np.ndarray | None' = None) -> None
```

Per-study diagnostic accuracy data, after any continuity correction.

**Attributes**

- `labels` — study labels.
- `tp, fn, fp, tn` — the four cells *after* correction.
- `sens` — sensitivity ``tp / (tp + fn)``.
- `fpr` — false positive rate ``fp / (fp + tn)``.
- `var_sens` — variance of ``logit(sens)``.
- `var_fpr` — variance of ``logit(fpr)``.
- `n_positive` — diseased participants per study.
- `n_negative` — non-diseased participants per study.
- `corrected` — which studies had a zero cell before correction.

## `DtaCorrectionWarning`

A continuity correction changed the counts.

Emitted whenever ``escalc_dta`` adds something to a cell, naming how many
studies were affected.  A correction is a modelling decision, not a
formatting detail, so it should never happen silently.

## `ReitsmaResult`

```python
ReitsmaResult(mu: 'list[float]', se: 'list[float]', ci_lb: 'list[float]', ci_ub: 'list[float]', Psi: 'np.ndarray', tau: 'list[float]', rho: 'float', vcov: 'np.ndarray', summary_sens: 'float', summary_fpr: 'float', summary_spec: 'float', sens_ci: 'list[float]', spec_ci: 'list[float]', fpr_ci: 'list[float]', k: 'int', method: 'str', loglik: 'float', loglik_raw: 'float', AIC: 'float', BIC: 'float', level: 'float', mu_fpr: 'list[float]', Psi_fpr: 'np.ndarray', vcov_fpr: 'np.ndarray', rho_fpr: 'float', data: 'DtaData') -> None
```

A fitted bivariate diagnostic accuracy model.

**Attributes**

- `mu` — ``(mu_1, mu_2)`` -- mean logit sensitivity and mean logit **specificity**.  Reporting specificity rather than its complement keeps ``rho`` on the scale the DTA literature uses: a threshold effect shows up as a *negative* correlation.
- `se` — standard errors of ``mu``.
- `ci_lb, ci_ub` — confidence limits of ``mu`` on the logit scale.
- `Psi` — the 2x2 study-level covariance matrix, in (sens, spec) coordinates.
- `tau` — its standard deviations.
- `rho` — the between-study correlation of logit sensitivity and logit specificity (negative under the usual threshold effect).
- `vcov` — covariance matrix of ``mu``.
- `summary_sens` — back-transformed ``mu_1``.
- `summary_spec` — back-transformed ``mu_2``.
- `summary_fpr` — ``1 - summary_spec``.
- `sens_ci, spec_ci, fpr_ci` — confidence intervals on the probability scale.
- `k` — number of studies.
- `method` — ``"reml"`` or ``"ml"``.
- `loglik` — log-likelihood with mada's Jacobian correction (what AIC uses).
- `loglik_raw` — the model's own log-likelihood.
- `AIC, BIC` — information criteria using ``2 + 3`` parameters and ``2k`` observations, as mada does.
- `mu_fpr, Psi_fpr, vcov_fpr, rho_fpr` — the same quantities on the false-positive-rate scale ``mada`` reports, so its numbers can be recovered exactly.
- `data` — the `DtaData` that was fitted.

## `HsrocResult`

```python
HsrocResult(Theta: 'float', Lambda: 'float', beta: 'float', sigma2theta: 'float', sigma2alpha: 'float', summary_sens: 'float', summary_fpr: 'float', summary_spec: 'float', fit: 'ReitsmaResult') -> None
```

HSROC parameters (Rutter & Gatsonis), derived from a bivariate fit.

**Attributes**

- `Theta` — mean threshold ``theta`` (positive = stricter threshold).
- `Lambda` — mean accuracy ``alpha``; ``exp(Lambda)`` is the diagnostic odds ratio at ``beta = 0``.
- `beta` — the asymmetry parameter; ``0`` means a symmetric SROC.
- `sigma2theta` — variance of the threshold random effect.
- `sigma2alpha` — variance of the accuracy random effect.
- `summary_sens, summary_fpr` — the summary point implied by the HSROC parameters.
- `fit` — the `ReitsmaResult` this was derived from.

## `SrocResult`

```python
SrocResult(fpr: 'np.ndarray', tpr: 'np.ndarray', type: 'str' = 'ruttergatsonis') -> None
```

A point on (or a grid of) the summary ROC curve.

**Attributes**

- `fpr` — false positive rates.
- `tpr` — the corresponding sensitivities.
- `type` — ``"ruttergatsonis"`` (the HSROC curve through the fitted parameters) or ``"naive"``.

## `escalc_dta`

```python
escalc_dta(tp: 'Any', fn: 'Any' = None, fp: 'Any' = None, tn: 'Any' = None, *, slab: 'Sequence[Any] | None' = None, correction: 'float' = 0.5, correction_control: 'str' = 'single') -> 'DtaData'
```

Turn 2x2 counts into logit sensitivity / logit FPR with variances.

**Args**

- `tp` — true positives, or a mapping with ``TP``/``FN``/``FP``/``TN`` keys (when the other three arguments are omitted).
- `fn` — false negatives.
- `fp` — false positives.
- `tn` — true negatives.
- `slab` — study labels; defaults to ``"Study 1"``, ``"Study 2"``, ...
- `correction` — value added to the cells, ``0.5`` by default.
- `correction_control` — ``"single"`` (default) adds ``correction`` only to the studies that have a zero cell, ``"none"`` never corrects, and ``"all"`` reproduces ``mada``, which adds it to *every* study as soon as any table anywhere has a zero cell. ``"all"`` is deliberately not the default here.  It is not a correction any more, it is a data-dependent perturbation of studies that were fine: adding one study with a zero cell moves every other study's estimate, and the bias is largest where the sample is smallest.  Use it only to reproduce ``mada``'s output exactly.

**Returns**

A `DtaData`.

**Raises**

- `ValueError` — for inconsistent input or an unknown ``correction_control``.

## `reitsma`

```python
reitsma(data: 'Any' = None, *, tp: 'Any' = None, fn: 'Any' = None, fp: 'Any' = None, tn: 'Any' = None, slab: 'Sequence[Any] | None' = None, method: 'str' = 'reml', correction: 'float' = 0.5, correction_control: 'str' = 'single', level: 'float' = 0.95) -> 'ReitsmaResult'
```

*Mirrors R's `reitsma`.*

Fit the bivariate diagnostic accuracy model (``mada::reitsma``).

**Args**

- `data` — a `DtaData`, or a mapping/frame with ``TP``/``FN``/``FP``/ ``TN`` columns (optionally ``study``).
- `tp, fn, fp, tn` — counts, used instead of ``data``.
- `slab` — study labels.
- `method` — ``"reml"`` (mada's default) or ``"ml"``.
- `correction` — continuity correction added to the cells.
- `correction_control` — ``"single"`` (default), ``"none"`` or ``"all"``. ``"all"`` reproduces ``mada``'s global correction; see `escalc_dta` for why it is not the default.
- `level` — confidence level (``0.95``).

**Returns**

A `ReitsmaResult`.

**Raises**

- `ValueError` — for an unknown method or unusable counts.

## `hsroc`

```python
hsroc(result: 'Any' = None, **kwargs: 'Any') -> 'HsrocResult'
```

HSROC parameters of a bivariate fit.

The two models are equivalent (Harbord et al. 2007), so this does not
re-fit: it applies the exact algebraic transform mada uses in
``calc_hsroc_coef``.  With the fit reported in (logit sensitivity, logit
**specificity**) coordinates, and ``c = sqrt(sd_sens / sd_spec)``::

    beta   = log(sd_spec / sd_sens)
    Theta  = (mu_sens / c - c * mu_spec) / 2
    Lambda = mu_sens / c + c * mu_spec
    sigma2theta = (sd_sens*sd_spec - Psi_12) / 2
    sigma2alpha = 2 * (sd_sens*sd_spec + Psi_12)

The signs differ from mada's version only because mada works on the
false-positive-rate scale; the resulting ``Theta``, ``Lambda``, ``beta``
and both variances are identical.

**Args**

- `result` — an `ReitsmaResult`, or ``None`` to fit from ``kwargs``.
- `**kwargs` — forwarded to `reitsma` when ``result`` is not a fit.

**Returns**

An `HsrocResult`.

**Raises**

- `TypeError` — when ``result`` is not a bivariate fit.

## `sroc`

```python
sroc(fit: 'ReitsmaResult', fpr: 'Any' = None, *, type: 'str' = 'ruttergatsonis') -> 'SrocResult'
```

*Mirrors R's `sroc`.*

The summary ROC curve (``mada::sroc``).

**Args**

- `fit` — a `ReitsmaResult`.
- `fpr` — false positive rates to evaluate; defaults to ``0.01, ..., 0.99``.
- `type` — ``"ruttergatsonis"`` (default) uses the HSROC curve implied by the fit, which is the line ``logit(TPR) = Lambda*exp(-beta/2) + exp(-beta)*logit(FPR)``. ``"naive"`` joins the observed points without smoothing.

**Returns**

A `SrocResult`.

**Raises**

- `ValueError` — for an unknown ``type``.

## `auc`

```python
auc(fit: 'ReitsmaResult', fpr: 'Any' = None) -> 'dict'
```

*Mirrors R's `AUC`.*

Area under the SROC curve (``mada::AUC``).

**Args**

- `fit` — a `ReitsmaResult`.
- `fpr` — grid of false positive rates; defaults to ``0.01, ..., 0.99``.

**Returns**

- ```{"auc"` — ..., "pauc": ...}``.  The partial AUC restricts the grid to
the range of FPRs actually observed, which is the number a reader
should quote when the studies cluster at one end of the range.

**Raises**

- `ValueError` — when fewer than 10 grid points are supplied.
