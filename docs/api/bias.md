# Publication bias

*40 public names, generated from `pimeta.__all__` and the docstrings in `src/pimeta/`. Do not edit this page by hand — run `python tools/gen_api_docs.py`.*

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
| [`limitmeta`](#limitmeta) | — |
| [`LimitmetaResult`](#limitmetaresult) | — |
| [`copas`](#copas) | — |
| [`CopasResult`](#copasresult) | — |
| [`copas_fit`](#copas_fit) | — |
| [`copas_loglik`](#copas_loglik) | — |
| [`copas_region`](#copas_region) | — |
| [`CopasRegion`](#copasregion) | — |
| [`bayes_copas`](#bayes_copas) | — |
| [`CopasBayesResult`](#copasbayesresult) | — |
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

An ``rma.mh`` / ``rma.peto`` result can be passed as ``yi``, in which case
metafor's ``cumul.rma.mh`` / ``.peto`` is used (``order``, ``decreasing``
and ``collapse`` then control the ordering).

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

## `limitmeta`

```python
limitmeta(result, method_adjust='beta0', level=None)
```

Adjust a pooled effect for small-study effects (Rücker et al. 2011).

**Args**

- `result` — fitted `rma_uni.RmaResult`, or a meta-analysis result holding one in ``.random``.  Its :math:`\tau^2` — from whichever estimator was used — drives the shrinkage, so it must be an intercept-only random-effects fit.
- `method_adjust` — ``"beta0"`` (default), ``"betalim"`` or ``"mulim"``.
- `level` — confidence level, either as a percentage (``95``) or a proportion (``0.95``).  Defaults to the level of the input model. (The package is not consistent here — ``RmaResult.level`` is a proportion while ``rma_mv(level=)`` is a percentage — so both are accepted rather than one of them silently misread.)

**Returns**

`LimitmetaResult`.

**Raises**

- `TypeError` — the argument is not a fitted model.
- `ValueError` — the model is not intercept-only, fewer than three studies, or an unknown ``method_adjust``.

## `LimitmetaResult`

```python
LimitmetaResult(method_adjust: 'str', TE: 'np.ndarray', seTE: 'np.ndarray', TE_limit: 'np.ndarray', seTE_limit: 'np.ndarray', TE_adjust: 'float', seTE_adjust: 'float', ci_lb: 'float', ci_ub: 'float', zval: 'float', pval: 'float', alpha: 'float', beta: 'float', tau: 'float', TE_random: 'float', seTE_random: 'float', Q: 'float', df_Q: 'int', Q_small: 'float', Q_resid: 'float', G_squared: 'float', k: 'int', level: 'float' = 95.0, slab: 'list[str] | None' = None) -> None
```

Adjusted effect estimate from a limit meta-analysis.

**Attributes**

- `method_adjust` — which adjusted estimate was reported.
- `TE, seTE` — the studies' effects and standard errors.
- `TE_limit, seTE_limit` — shrunken per-study estimates.
- `TE_adjust, seTE_adjust` — the adjusted overall effect and its SE.
- `ci_lb, ci_ub, zval, pval` — inference on the adjusted effect.
- `alpha, beta` — intercept and slope of the radial plot — the bias parameter and the effect at infinite precision.
- `tau` — between-study standard deviation used for the shrinkage.
- `TE_random, seTE_random` — the unadjusted random-effects estimate.
- `Q, df_Q` — Cochran's Q of the unadjusted model.
- `Q_small` — heterogeneity attributed to small-study effects.
- `Q_resid` — heterogeneity remaining beyond them.
- `G_squared` — proportion of the radial-plot variation left unexplained by the shrunken model.
- `k, level` — number of studies and the confidence level in percent.
- `slab` — study labels, when the input model had them.

## `copas`

```python
copas(yi, sei=None, *, tau=None, mu0=None, se0=None, ngrid=20, nlevels=10, levels=None, rho_bound=0.9999, left=None, slope=None, sign_rsb=0.1, level=95.0, gamma0_range=None, gamma1_range=None)
```

Copas selection-model analysis of a set of effect sizes.

**Args**

- `yi` — observed effects, or a fitted `rma_uni.RmaResult` whose ``.yi`` / ``.vi`` / ``.tau2`` are used.
- `sei` — standard errors, when ``yi`` is an array.
- `tau` — between-study SD to start from; defaults to the random-effects estimate.
- `mu0, se0` — the random-effects estimate and SE.  When ``yi`` is a fitted model these come from the fit.
- `ngrid` — grid size for the sensitivity surface.
- `nlevels` — number of contour levels when ``levels`` is not given.
- `levels` — explicit contour levels of the pooled effect.
- `rho_bound` — bound on the effect-selection correlation.
- `left` — whether the unpublished studies are assumed to lie on the left of the funnel (negative effects).  Defaults to the sign of Egger's intercept, as metasens does.
- `slope` — slope of the line orthogonal to the contours, in rescaled coordinates.  Defaults to the value derived from the contours.
- `sign_rsb` — significance level of the residual-selection test used to decide where along the path to stop.
- `level` — confidence level in percent.
- `gamma0_range, gamma1_range` — grid ranges; defaults follow metasens.

**Returns**

`CopasResult`.

**Raises**

- `ValueError` — too few studies, mismatched lengths, or non-positive standard errors.

## `CopasResult`

```python
CopasResult(yi: 'np.ndarray', seTE: 'np.ndarray', tau_random: 'float', TE_random: 'float', seTE_random: 'float', left: 'bool', rho_bound: 'float', gamma0_range: 'tuple', gamma1_range: 'tuple', gamma0: 'np.ndarray', gamma1: 'np.ndarray', te_contour: 'np.ndarray', loglik_contour: 'np.ndarray', levels: 'list', contour: 'dict', slope: 'float', gamma0_line: 'np.ndarray', gamma1_line: 'np.ndarray', publprob: 'np.ndarray', te_line: 'np.ndarray', se_line: 'np.ndarray', rho_line: 'np.ndarray', tau_line: 'np.ndarray', pval_rsb: 'np.ndarray', n_unpubl: 'np.ndarray', te_adjust: 'float', se_adjust: 'float', ci_lb: 'float', ci_ub: 'float', zval: 'float', pval: 'float', sign_rsb: 'float' = 0.1, level: 'float' = 95.0) -> None
```

A Copas selection-model analysis.

**Attributes**

- `yi, seTE` — the studies.
- `tau_random, TE_random, seTE_random` — the ordinary random-effects fit the analysis starts from.
- `left` — which side the missing studies were assumed to be on.
- `rho_bound` — the bound placed on the effect-selection correlation.
- `gamma0_range, gamma1_range` — the grid ranges.
- `gamma0, gamma1` — the grid itself.
- `te_contour` — ``(ngrid, ngrid)`` pooled effect at each grid point, with ``gamma0`` varying down the rows and ``gamma1`` across columns.
- `loglik_contour` — ``(ngrid, ngrid)`` maximised log-likelihood at each grid point — the surface `copas_region` turns into a confidence region.
- `levels` — the contour levels that were traced.
- `contour` — per level, the number of crossing points, the fitted slope, its standard error, the intercept and the adjusted R-squared.
- `slope` — slope of the line orthogonal to the contours, in rescaled coordinates.
- `gamma0_line, gamma1_line` — the path walked, in natural units.
- `publprob` — publication probability of the largest study at each point.
- `te_line, se_line, rho_line, tau_line` — the fit along the path.
- `pval_rsb` — p-value of the residual-selection test (beta = 0).
- `n_unpubl` — number of studies implied to be unpublished.
- `te_adjust, se_adjust, ci_lb, ci_ub, zval, pval` — the reported answer.
- `sign_rsb, level` — settings.

## `copas_fit`

```python
copas_fit(yi, sei, gamma0, gamma1, *, rho_bound=0.9999, mu0=None, rho0=None, tau0=None, with_beta=False, se_mu=True, rho_lim=None)
```

Fit the selection model at one ``(gamma0, gamma1)``.

**Args**

- `yi` — observed effects.
- `sei` — their standard errors.
- `gamma0, gamma1` — the selection parameters to condition on.
- `rho_bound` — the correlation is constrained to ``[-rho_bound, rho_bound]``.
- `mu0, rho0, tau0` — starting values.  metasens starts from the ordinary random-effects estimate, ``-rho_bound/2`` and the random-effects :math:`\hat\tau`; the same defaults are used here.
- `with_beta` — also estimate the residual small-study slope.
- `se_mu` — compute the standard error from the numerical Hessian.
- `rho_lim` — explicit ``(lower, upper)`` bounds for the correlation.  The residual-selection fit needs this: metasens restricts :math:`\rho` to the side of the funnel the missing studies were assumed to be on, which is narrower than the bound used for the surface and can matter for which optimum is reached.

**Returns**

`CopasFit`.

## `copas_loglik`

```python
copas_loglik(params, yi, sei, gamma0, gamma1)
```

Negative log-likelihood of the Copas selection model.

**Args**

- `params` — ``(mu, rho, tau)``, or ``(mu, rho, tau, beta)`` to include the residual small-study slope.
- `yi` — observed effects.
- `sei` — their standard errors.
- `gamma0, gamma1` — the selection parameters.

**Returns**

The negative log-likelihood, the quantity metasens' ``copas.loglik.without.beta`` returns (and ``.with.beta`` for the four-parameter form).

**Raises**

- `ValueError` — the parameter vector is not of length three or four.

## `copas_region`

```python
copas_region(result, level=95.0)
```

Likelihood region for the Copas selection parameters.

The model identifies only the combination ``gamma0 + gamma1/se``, so a
point estimate of the adjusted effect is not the honest output.  This
keeps the whole set of selection scenarios consistent with the data:

.. math::

    2\{\ell_{\max} - \ell(\gamma_0, \gamma_1)\}
    \le \chi^2_2(1-\alpha)

and reports the range of the adjusted effect over that set.  Projecting a
joint confidence set onto one coordinate gives a conservative interval —
wider than a profile that fixes the others — which is the right direction
to err in when the "others" are unidentified.

**Args**

- `result` — a `CopasResult` from `copas`; the surface is reused, so nothing is refitted.
- `level` — confidence level in percent (``95``) or as a proportion (``0.95``).

**Returns**

`CopasRegion`.

**Raises**

- `ValueError` — the level is not in ``(0, 100)``, or the surface has no finite point to anchor the maximum.

## `CopasRegion`

```python
CopasRegion(level: 'float', crit: 'float', gamma0_in: 'np.ndarray', gamma1_in: 'np.ndarray', mu_in: 'np.ndarray', publprob_in: 'np.ndarray', mu_best: 'float', mu_lb: 'float', mu_ub: 'float', publprob_lb: 'float', publprob_ub: 'float', no_selection_inside: 'bool', gamma1_min: 'float', sign_stable: 'bool', k: 'int') -> None
```

Which selection scenarios the data cannot rule out.

``metasens`` reduces the Copas analysis to one number by walking a path
drawn by hand.  This is the same model with an actual inference attached:
the set of ``(gamma0, gamma1)`` whose profile likelihood is within the
usual chi-square-2 threshold of the best value.  Everything the model can
say about the adjusted effect then follows from that set.

**Attributes**

- `level` — confidence level in percent.
- `crit` — the threshold ``2*(max loglik - loglik)`` must not exceed.
- `gamma0_in, gamma1_in` — the grid points inside the region.
- `mu_in` — the pooled effect at those points.
- `publprob_in` — publication probability of the largest study at those points — the interpretable coordinate.
- `mu_best` — the pooled effect at the single best point.
- `mu_lb, mu_ub` — **projection interval** for the adjusted effect: the range of the effect over the whole region.  Conservative by construction, in the same sense that a projection of a joint confidence set always is.
- `publprob_lb, publprob_ub` — the same for the publication probability.
- `no_selection_inside` — whether the region contains a scenario in which publication does not depend on precision (``gamma1 = 0``).  If it does, the data do not require any selection at all.
- `gamma1_min` — the smallest ``|gamma1|`` among admissible scenarios; zero means precision-dependent selection is not needed to explain the data.
- `sign_stable` — whether every admissible scenario has the effect on the same side of zero.  This is the question a reader actually asks, and it is answerable without any of the path machinery.
- `k` — number of studies.

## `bayes_copas`

```python
bayes_copas(yi, sei=None, *, gamma=None, prior=None, mcmc=None, level: 'float' = 95.0) -> 'CopasBayesResult'
```

Posterior of the Copas selection model.

The model is the one `copas` estimates, with the parameters it holds
fixed instead given priors:

.. math::

    y_i \mid \text{published} \sim \text{the Copas likelihood},\quad
    \mu \sim N(0, \sigma_\mu^2),\; \tau \sim \text{half-}N(0, s_\tau^2),\;
    \rho \sim U(-1, 1),\; \gamma_0 \sim N(0, s_0^2),\;
    \gamma_1 \sim \text{half-}N(0, s_1^2).

:math:`\gamma_1 \ge 0` is the same restriction the sensitivity grid uses,
and the *side* of the funnel the missing studies are on is carried by the
sign of :math:`\rho`.  Priors are arguments rather than constants so a
sensitivity analysis is a keyword, and they are recorded on the result.

Why a posterior rather than the contour heuristic: :math:`(\gamma_0,
\gamma_1)` are weakly identified, and a posterior says so in the width of
the interval instead of returning the point where a fitted line happened to
stop. `copas_region` makes the same point from the frequentist side.

**Args**

- `yi` — observed effects, or a fitted model whose ``.yi``/``.vi`` are used.
- `sei` — their standard errors.
- `gamma` — ``(gamma0, gamma1)`` to hold the selection parameters fixed. Used to compare against `copas_fit` at one grid point, and to ask "what if the selection mechanism were known". Left out, they are estimated like the rest.
- `prior` — overrides for ``mu_sd``, ``tau_scale``, ``gamma0_sd`` and ``gamma1_scale``.
- `mcmc` — sampler settings: ``n_draws``, ``warmup``, ``chains``, ``seed``, ``check``.
- `level` — credibility level in percent (``0.95``, ``95`` and ``0.05`` are all read as 95%).

**Returns**

`CopasBayesResult`.

**Raises**

- `ValueError` — fewer than three studies, mismatched lengths, non-positive standard errors, or an unknown key in ``prior``/``mcmc``.

**Example**

- `>>> fit = ma.bayes_copas(yi, sei=sei, mcmc={"seed"` — 1})
- `>>> print(fit)                          # doctest` — +SKIP

## `CopasBayesResult`

```python
CopasBayesResult(posterior: 'object', mu: 'float', tau: 'float', rho: 'float', mu_ci: 'tuple[float, float]', tau_ci: 'tuple[float, float]', rho_ci: 'tuple[float, float]', k: 'int', level: 'float' = 95.0, gamma0: 'float' = nan, gamma1: 'float' = nan, fixed_gamma: 'tuple[float, float] | None' = None, priors: 'dict' = <factory>) -> None
```

Posterior of the Copas selection model.

**Attributes**

- `posterior` — the `bayes.Posterior`, with its draws and diagnostics.
- `mu, tau, rho, gamma0, gamma1` — posterior medians.
- `mu_ci, tau_ci, rho_ci` — equal-tailed credible intervals.
- `k` — studies.
- `level` — credibility level in percent.
- `fixed_gamma` — the ``(gamma0, gamma1)`` that were held fixed, or ``None`` when the selection parameters were estimated.
- `priors` — what was put on the parameters, so the record is complete.

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
- `plotit` — accepted for interface parity — draw the radial plot yourself with `plotting.radial`.
- `correct` — continuity correction for the rank tests.
- `k_min` — meta requires ``k >= k_min`` and ``k >= 3``; otherwise only ``k`` and ``k_min`` are reported, exactly like R.
- `n_e, n_c, event_e, event_c, sm` — override the values taken from ``x``.
- `method_tau` — tau^2 estimator used by the Thompson variant (defaults to the analysis' own estimator, i.e. R's ``x$method.tau``).
- `alpha` — significance level used for the regression CIs.

**Returns**

`MetabiasResult`.

## `MetabiasResult`

```python
MetabiasResult(method_bias: 'str', method: 'str' = '', var_model: 'str' = '', statistic: 'float | None' = None, pval: 'float | None' = None, p_value: 'float | None' = None, estimate: 'list[float] | None' = None, intercept: 'float | None' = None, se_intercept: 'float | None' = None, tau: 'float | None' = None, df: 'int | None' = None, k: 'int | None' = None, k_min: 'float' = 10) -> None
```

*Mirrors R's `metabias`.*

Result of `metabias` (``meta::metabias``).

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
