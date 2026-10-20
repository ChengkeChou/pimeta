# Diagnostics and utilities

*58 public names, generated from `pimeta.__all__` and the docstrings in `src/pimeta/`. Do not edit this page by hand — run `python tools/gen_api_docs.py`.*

| Name | Mirrors in R |
|---|---|
| [`MvProfileResult`](#mvprofileresult) | profile.rma |
| [`profile_rmamv`](#profile_rmamv) | profile.rma.mv |
| [`deltamethod`](#deltamethod) | — |
| [`DeltaMethodResult`](#deltamethodresult) | deltamethod |
| [`tes`](#tes) | tes |
| [`TesResult`](#tesresult) | tes |
| [`vif`](#vif) | — |
| [`VifResult`](#vifresult) | vif |
| [`permutest`](#permutest) | permutest |
| [`PermutestResult`](#permutestresult) | permutest |
| [`simulate_rma`](#simulate_rma) | simulate.rma |
| [`vcalc`](#vcalc) | vcalc |
| [`hettest`](#hettest) | hettest |
| [`HettestResult`](#hettestresult) | — |
| [`matreg`](#matreg) | matreg |
| [`MatregResult`](#matregresult) | matreg |
| [`profile_rma`](#profile_rma) | profile.rma.uni |
| [`ProfileResult`](#profileresult) | profile.rma.uni |
| [`bldiag`](#bldiag) | bldiag |
| [`blsplit`](#blsplit) | blsplit |
| [`vec2mat`](#vec2mat) | vec2mat |
| [`replmiss`](#replmiss) | replmiss |
| [`cmicalc`](#cmicalc) | cmicalc |
| [`contrmat`](#contrmat) | contrmat |
| [`pairmat`](#pairmat) | pairmat |
| [`rcalc`](#rcalc) | rcalc |
| [`RcalcResult`](#rcalcresult) | — |
| [`fitstats`](#fitstats) | fitstats |
| [`FitstatsResult`](#fitstatsresult) | — |
| [`aic_rma`](#aic_rma) | AIC.rma |
| [`bic_rma`](#bic_rma) | BIC.rma |
| [`se_rma`](#se_rma) | se.rma |
| [`df_residual_rma`](#df_residual_rma) | df.residual.rma |
| [`model_matrix_rma`](#model_matrix_rma) | model.matrix.rma |
| [`as_data_frame_list_rma`](#as_data_frame_list_rma) | as.data.frame.list.rma |
| [`cooks_distance_rma`](#cooks_distance_rma) | cooks.distance.rma.uni |
| [`sigma_matreg`](#sigma_matreg) | sigma.matreg |
| [`update_rma`](#update_rma) | update.rma |
| [`blup`](#blup) | blup.rma.uni |
| [`BlupResult`](#blupresult) | blup.rma.uni |
| [`as_data_frame_confint`](#as_data_frame_confint) | as.data.frame.confint.rma |
| [`as_matrix_list_rma`](#as_matrix_list_rma) | as.matrix.list.rma |
| [`se`](#se) | se |
| [`as_data_frame_list_anova`](#as_data_frame_list_anova) | as.data.frame.list.anova.rma |
| [`as_data_frame_list_confint`](#as_data_frame_list_confint) | as.data.frame.confint.rma |
| [`as_data_frame_vif`](#as_data_frame_vif) | as.data.frame.vif.rma |
| [`inches2units`](#inches2units) | inches2units |
| [`units2inches`](#units2inches) | units2inches |
| [`jama_labels`](#jama_labels) | JAMAlabels |
| [`ci`](#ci) | ci |
| [`getmfopt`](#getmfopt) | getmfopt |
| [`setmfopt`](#setmfopt) | setmfopt |
| [`estimates`](#estimates) | estimates |
| [`EstimatesResult`](#estimatesresult) | estimates |
| [`forest_args`](#forest_args) | .forestArgs |
| [`MissingDataWarning`](#missingdatawarning) | — |
| [`complete_cases`](#complete_cases) | — |
| [`is_missing`](#is_missing) | — |

## `MvProfileResult`

```python
MvProfileResult(component, index, sigma2, ll, maxll, beta, ci_lb, ci_ub, vc, method, xlab, title, comps=1)
```

*Mirrors R's `profile.rma`.*

Result of `profile_rmamv` (mirrors ``profile.rma`` objects).

``sigma2`` holds the profiling grid, ``ll`` the maximised log-likelihood
at each grid point and ``beta`` / ``ci_lb`` / ``ci_ub`` the fixed effects
(one row per grid point).  ``maxll`` is the log-likelihood of the
unconstrained fit.

## `profile_rmamv`

```python
profile_rmamv(x, sigma2=None, steps=20, xlim=None, plot=False, **kw)
```

*Mirrors R's `profile.rma.mv`.*

``profile.rma.mv`` — profile the log-likelihood over one variance
component.

For every grid value ``v`` of the chosen component the model is refitted
with **that** component pinned to ``v`` while every other component is
re-estimated, which is exactly what metafor's ``.profile.rma.mv`` does
(``rma.mv(..., sigma2 = c(NA, ..., v, ..., NA))``).  The returned object
holds the maximised log-likelihood, the fixed effects and their confidence
intervals at each grid point.

**Args**

- `x` — an `RmaMvResult`.
- `sigma2` — 1-based index of the variance component to profile (metafor's ``sigma2=`` argument).  Defaults to the first component.
- `steps` — number of grid points, or an explicit sequence of values (metafor's ``steps`` argument — a vector of length ``>= 2`` is used as the grid itself).
- `xlim` — profiling range; by default ``[max(0, vc/4), max(0.1, vc*4)]`` around the fitted value ``vc``, like metafor.
- `plot` — draw the profile plot immediately.
- `**kw` — forwarded to `plotting.plot_mv_profile`.

**Returns**

`MvProfileResult`.

## `deltamethod`

```python
deltamethod(coef, vcov, fun, order=1, level=0.95, H0=0.0)
```

Delta-method standard errors for a transformation of the coefficients.

**Args**

- `coef` — coefficient vector.
- `vcov` — corresponding variance-covariance matrix.
- `fun` — callable mapping the coefficient vector (or, when its signature has one parameter per coefficient, one scalar argument per coefficient) to a transformed value (scalar or vector).
- `order` — ``1``, ``2`` or ``3`` — order of the delta-method approximation.
- `level` — confidence level (0.95 → 95% CIs).
- `H0` — null value(s) used for the z-test.

**Returns**

`DeltaMethodResult`.

**Note**

metafor computes the required derivatives with the ``calculus`` package; here they are obtained with high-accuracy central differences.

## `DeltaMethodResult`

```python
DeltaMethodResult(coef: 'list[float]', se: 'list[float]', zval: 'list[float]', pval: 'list[float]', ci_lb: 'list[float]', ci_ub: 'list[float]', vcov: 'np.ndarray', level: 'float', order: 'int', test: 'str' = 'z') -> None
```

*Mirrors R's `deltamethod`.*

Result of `deltamethod` (``metafor::deltamethod``).

## `tes`

```python
tes(yi, vi=None, sei=None, *, H0=0.0, alternative='two.sided', alpha=0.05, theta=None, tau2=None, test=None, tes_alternative='greater', tes_alpha=0.1, correct=False, find_lim=True, rel_tol=0.0001220703125, subdivisions=100, tau2_lb=0.0)
```

*Mirrors R's `tes`.*

Test of excess significance (``metafor::tes``).

**Args**

- `yi` — study estimates.
- `vi` — sampling variances (or supply ``sei``).
- `sei` — standard errors (used when ``vi`` is not given).
- `H0` — null value of the effect size (default 0).
- `alternative` — ``"two.sided"``, ``"greater"`` or ``"less"``.
- `alpha` — significance level used to declare a study significant.
- `theta` — assumed (common) true effect size(s); estimated when ``None``.
- `tau2` — between-study variance used for the power computation.
- `test` — ``"exact"``, ``"chi2"`` or ``"binom"`` (chosen automatically when ``None``).
- `tes_alternative` — alternative for the test of excess significance.
- `tes_alpha` — level used to derive ``theta_lim``.
- `correct` — apply Yates' continuity correction (``chi2`` test).
- `find_lim` — compute the ``theta`` value at which the test would no longer be significant.
- `rel_tol` — relative tolerance for the power integrals.
- `subdivisions` — maximum number of subdivisions for the integrals.
- `tau2_lb` — values of ``tau2`` at or below this are treated as zero.

**Returns**

`TesResult`.

## `TesResult`

```python
TesResult(k: 'int', O: 'int', E: 'float', OEratio: 'float', test: 'str', X2: 'float', pval: 'float', power: 'list[float]', sig: 'list[bool]', theta: 'object', theta_lim: 'object', tes_alternative: 'str', tes_alpha: 'float') -> None
```

*Mirrors R's `tes`.*

Result of `tes` (``metafor::tes``).

## `vif`

```python
vif(result, btt=None, intercept=False)
```

Generalized variance-inflation factors (Fox & Monette).

**Args**

- `result` — a fitted `RmaResult`.
- `btt` — indices (1-based, like metafor) or names of the coefficients to assess. Defaults to all non-intercept coefficients, or to all coefficients when the model has no intercept.
- `intercept` — keep the intercept when computing the VIFs.

**Returns**

`VifResult`.

**Raises**

- `ValueError` — for intercept-only models, where VIFs are undefined.

## `VifResult`

```python
VifResult(vifs: 'list[float]', sifs: 'list[float]', m: 'list[int]', specs: 'list[str]', coefs: 'list[str]', coefnames: 'list[str]', bttspec: 'bool') -> None
```

*Mirrors R's `vif`.*

Result of `vif` (``metafor::vif``).

## `permutest`

```python
permutest(result, exact=False, iter=1000, btt=None, permci=False, seed=None, alternative='two.sided', stat='test', p2defn='abs')
```

*Mirrors R's `permutest`.*

Permutation test for the model coefficients (``metafor::permutest``).

With moderators the rows of the design matrix are permuted; without them
the outcomes get random **sign flips**, which is the randomisation
distribution of a single-group meta-analysis (permuting the outcomes there
would leave the fit unchanged).  Each coefficient gets its own p-value from
its own permuted z-distribution, and the omnibus ``QM`` gets one too.

**Args**

- `result` — fitted `RmaResult`.
- `exact` — enumerate the whole permutation set (2^k sign patterns without moderators, k!/prod(c!) row permutations with them).  Ignored (and silently switched on/off) exactly as metafor does: if the exact count is no larger than ``iter`` the exact test is used anyway.
- `iter` — number of random permutations when the exact set is too large.
- `btt` — 1-based indices of the coefficients to test (metafor's ``btt``). Defaults to the moderator coefficients (or 1 when there are none).
- `permci` — also compute permutation confidence intervals for the coefficients (a root search on the permutation p-value, so this costs roughly ``2 * p`` full permutation distributions).
- `seed` — optional RNG seed.
- `alternative` — ``"two.sided"`` (default), ``"less"`` or ``"greater"``.
- `stat` — ``"test"`` (z-values, default) or ``"coef"`` (raw coefficients).
- `p2defn` — ``"abs"`` (default) or ``"px2"`` two-sided convention.

**Returns**

`PermutestResult`.

## `PermutestResult`

```python
PermutestResult(pval: 'list[float]', QM: 'float', QMp: 'float', QMp_perm: 'float', dfs: 'list[int]', k: 'int', p: 'int', nperm: 'int', exact: 'bool' = False, beta: 'list[float] | None' = None, se: 'list[float] | None' = None, ci_lb: 'list[float] | None' = None, ci_ub: 'list[float] | None' = None, zval: 'list[float] | None' = None, zval_perm: 'list[list[float]] | None' = None, QM_perm: 'list[float] | None' = None, coefnames: 'list[str] | None' = None) -> None
```

*Mirrors R's `permutest`.*

Result of `permutest` (``metafor::permutest``).

``pval`` holds one permutation p-value per model coefficient (``p``
entries), ``QMp_perm`` the permutation p-value of the omnibus statistic.
``ci_lb`` / ``ci_ub`` are only filled in when ``permci`` was requested.

## `simulate_rma`

```python
simulate_rma(result, nsim=1, seed=None)
```

*Mirrors R's `simulate.rma`.*

``metafor::simulate.rma`` — simulate effect sizes from a fitted model.

Draws ``theta_i ~ N(X_i beta, tau2)`` and ``y_i ~ N(theta_i, v_i)``, i.e.
new effect sizes with the same sampling variances and the estimated
heterogeneity.

**Args**

- `result` — fitted `RmaResult`.
- `nsim` — number of simulated datasets.
- `seed` — optional RNG seed.

**Returns**

``(k, nsim)`` array of simulated effect sizes.

## `vcalc`

```python
vcalc(vi, cluster=None, obs=None, rho=0.5, phi=0.0)
```

*Mirrors R's `vcalc`.*

``metafor::vcalc`` — build a block-diagonal sampling-variance matrix.

Within a cluster, the covariance between two effect sizes is
``rho * sqrt(vi_i * vi_j)``; between clusters it is zero.

**Args**

- `vi` — sampling variances (length ``k``).
- `cluster` — cluster identifiers (length ``k``); ``None`` puts everything in one cluster.
- `obs` — observation identifiers used to order the within-cluster blocks (optional; defaults to the input order).
- `rho` — within-cluster correlation.
- `phi` — optional additive heterogeneity factor (``phi * sqrt(vi_i*vi_j)`` used by some ``vcalc`` variants); default 0.

**Returns**

``(k, k)`` float array.

## `hettest`

```python
hettest(result, method=None, test='score', seed=None, iter=1000)
```

*Mirrors R's `hettest`.*

``metafor::hettest`` — test for residual heterogeneity.

Implements metafor's ``.hettest.score`` (Rao's score test evaluated with the
*estimated* ``tau^2`` and the random-effects weights)::

    vari = tau2_hat + vi;  wi = 1/vari;  S = sum(wi)
    Ui   = 0.5*wi^2*(yi-mu_hat)^2 - 0.5*wi + 0.5*wi^2/S
    I    = outer(-1/(sqrt(2)*S*vari^2)) with
           diag(I) = 0.5*(wi - 1/(S*vari^2))^2
    X2   = Ui' inv(I) Ui          ~ chi^2(k-1)

**Args**

- `result` — fitted `RmaResult` (intercept-only models only).
- `method` — ``"REML"`` (default) or ``"ML"`` — selects the score variant.
- `test` — ``"score"`` (metafor default) or ``"QE"`` (classical Cochran test, provided as a convenience).
- `seed, iter` — accepted for API compatibility (no bootstrap is run).

**Returns**

`HettestResult`.

## `HettestResult`

```python
HettestResult(method: 'str', test: 'str', QE: 'float', pval: 'float', k: 'int', p: 'int') -> None
```

Result of the (permutation/score) heterogeneity test.

## `matreg`

```python
matreg(y, x=None, R=None, n=None, V=None, cov=False, means=None, ztor=False, nearpd=False, level=95, digits=4, names=None, measure=None)
```

*Mirrors R's `matreg`.*

``metafor::matreg`` — regression from a correlation / covariance matrix.

Transcribes ``metafor::matreg``: the standardised path coefficients are
``b = R[x,x]^-1 R[x,y]`` and their standard errors come either from the
residual mean square (``n`` supplied) or from a delta-method propagation of
a sampling covariance matrix ``V`` over the unique elements of ``R``.

**Args**

- `y` — 1-based index of the outcome variable in ``R`` (or its name, see ``names``).
- `x` — 1-based indices of the predictors.  Defaults to every variable except ``y``.
- `R` — symmetric correlation matrix.
- `n` — sample size — exclusive with ``V``.
- `V` — sampling covariance matrix of the ``p(p-1)/2`` unique off-diagonal elements of ``R`` (column-major lower triangle, i.e. R's ``R[lower.tri(R)]``), or ``p(p+1)/2`` when ``cov=True``.
- `cov` — treat ``R`` as a covariance matrix and add an intercept.
- `means` — variable means, used to centre the intercept when ``cov=True``.
- `ztor` — ``R`` holds Fisher z values; transform back with ``tanh``.
- `nearpd` — repair a non-positive-definite ``R[x,x]`` by clipping its eigenvalues (R uses ``Matrix::nearPD``, so the repaired matrix and therefore the estimates need not agree to the last digit).
- `level` — confidence level (``95``, ``0.95`` or ``0.05``).  The result stores the tail area, exactly like R.
- `names` — variable names for ``R``, so that ``y``/``x`` can be strings.

**Returns**

`MatregResult`.

## `MatregResult`

```python
MatregResult(beta: 'list[float]', se: 'list[float]', pval: 'list[float]', ci_lb: 'list[float]', ci_ub: 'list[float]', vb: 'np.ndarray', names: 'list[str]', test: 'str', level: 'float', intercept: 'bool', p: 'int', parms: 'int', R2: 'float', tval: 'list[float] | None' = None, df: 'int | None' = None, df_residual: 'int | None' = None, R2adj: 'float | None' = None, F: 'list[float] | None' = None, Fdf: 'list[int] | None' = None, Fp: 'float | None' = None, sigma2_ml: 'float | None' = None, sigma2_reml: 'float | None' = None, nobs: 'int | None' = None, deviance: 'float | None' = None, fit_stats: 'dict | None' = None, zval: 'list[float] | None' = None, QM: 'float | None' = None, QMdf: 'list[int] | None' = None, QMp: 'float | None' = None, k: 'int' = 0, yi: 'list[float] | None' = None, vi: 'list[float] | None' = None, QE: 'float | None' = None, QEp: 'float | None' = None) -> None
```

*Mirrors R's `matreg`.*

Result of `matreg` (``metafor::matreg``).

``test`` is ``"t"`` when the model was fitted from ``n`` (the standard
error of the standardised path coefficients comes from the residual mean
square and the coefficient is referred to a t distribution) and ``"z"``
when a sampling covariance matrix ``V`` was supplied (delta-method
standard errors, normal reference distribution).

## `profile_rma`

```python
profile_rma(result, n_points=20, level=0.05)
```

*Mirrors R's `profile.rma.uni`.*

``metafor::profile.rma.uni`` — profile log-likelihood of ``tau^2``.

A `.selmodel.SelModelResult` is handled too (R's
``profile.rma.uni.selmodel``), in which case the profile is over the
selection parameter ``delta``.

Otherwise the (restricted) log-likelihood is evaluated on an evenly spaced
grid that runs from the QP lower to the QP upper confidence bound of
``tau^2`` — exactly the grid metafor uses (``n_points = 20`` by default).
At each grid point ``beta`` is re-estimated, so the curve is the profile
(not the conditional) likelihood.

**Args**

- `result` — fitted `RmaResult` or `SelModelResult`.
- `n_points` — number of grid points.
- `level` — alpha for the QP bounds (0.05 → 95% bounds).

**Returns**

`ProfileResult` (or ``SelProfileResult`` for a selection model).

## `ProfileResult`

```python
ProfileResult(tau2: 'list[float]', ll: 'list[float]', beta: 'list[float]', ci_lb: 'float', ci_ub: 'float', maxll: 'float', method: 'str', k: 'int') -> None
```

*Mirrors R's `profile.rma.uni`.*

Result of `profile_rma` (``metafor::profile.rma.uni``).

## `bldiag`

```python
bldiag(*blocks, order=None)
```

*Mirrors R's `bldiag`.*

``metafor::bldiag`` — build a block-diagonal matrix.

**Args**

- `*blocks` — matrices (block diagonal), vectors (treated as diagonals) or a single list of such blocks.
- `order` — optional permutation (0-based indices, or a permutation vector as returned by R's ``order()``) applied to rows and columns of the resulting square matrix.

**Returns**

``numpy.ndarray``.

## `blsplit`

```python
blsplit(x, cluster=None, fun=None, args=None, sort=False)
```

*Mirrors R's `blsplit`.*

``metafor::blsplit`` — split a (symmetric) matrix by cluster.

**Args**

- `x` — square symmetric matrix.
- `cluster` — grouping variable (one entry per row/column).
- `fun` — optional function applied to each sub-matrix.
- `args` — extra positional arguments for ``fun``.
- `sort` — sort the cluster labels.

**Returns**

``dict`` mapping cluster label to sub-matrix (or to ``fun``'s result).

## `vec2mat`

```python
vec2mat(x, diag=False, corr=None, dimnames=None)
```

*Mirrors R's `vec2mat`.*

``metafor::vec2mat`` — build a symmetric matrix from its lower triangle.

**Args**

- `x` — vector of lower-triangular entries (including the diagonal when ``diag=True``), in R's column-major order.
- `diag` — whether ``x`` includes the diagonal.
- `corr` — set the diagonal to 1 (default ``not diag``).
- `dimnames` — optional row/column labels.

**Returns**

``numpy.ndarray``.

## `replmiss`

```python
replmiss(x, y=None)
```

*Mirrors R's `replmiss`.*

``metafor::replmiss`` — replace missing values of ``x`` with ``y``.

## `cmicalc`

```python
cmicalc(mi, method='exact')
```

*Mirrors R's `cmicalc`.*

``metafor::cmicalc`` — continuity-correction multiplier (Elston 1991).

``exact``:  ``exp(lgamma(mi/2) - log(sqrt(mi/2)) - lgamma((mi-1)/2))``
``approx``: ``1 - 3/(4*mi - 1)``

## `contrmat`

```python
contrmat(data, grp1, grp2, last=None, shorten=False, minlen=2, check=True, append=True)
```

*Mirrors R's `contrmat`.*

``metafor::contrmat`` — treatment-contrast indicator columns.

``X = model.matrix(~0+grp1) - model.matrix(~0+grp2)`` over the treatment
levels, with the ``last`` level moved to the final column.

**Args**

- `data` — ``pandas.DataFrame`` or dict of columns.
- `grp1, grp2` — column names (or 0-based positions) of the two arms.
- `last` — level to be placed last (default: most frequent ``grp2`` level).
- `shorten, minlen` — shorten the level labels (first ``minlen`` characters).
- `check` — run R's ``make.names`` on the resulting column names.
- `append` — return ``data`` with the contrast columns appended.

**Returns**

dict of columns (or the contrast matrix when ``append=False``).

## `pairmat`

```python
pairmat(result, btt=None, fixed=False)
```

*Mirrors R's `pairmat`.*

``metafor::pairmat`` — pairwise-comparison contrast matrix.

**Args**

- `result` — a `.rma_uni.RmaResult`.
- `btt` — coefficients to compare (1-based like R).
- `fixed` — include the coefficient names as stored.

**Returns**

``(matrix, rownames)``.

## `rcalc`

```python
rcalc(x, ni, data=None, rtoz=False, nfun='min', upper=False, simplify=True, rowid=None, vnames=None, noid=False, dnames=None)
```

*Mirrors R's `rcalc`.*

``metafor::rcalc`` — turn correlation matrices into effect sizes + V.

**Args**

- `x` — a square correlation matrix, a list of them, or a ``pandas`` ``DataFrame`` with columns ``(ri, var1, var2, id)``.
- `ni` — sample size (scalar) or one per matrix / study.
- `rtoz` — apply Fisher's r-to-z transformation.
- `nfun` — how to summarise ``ni`` within a study (``"min"``, ``"harmonic"``, ``"mean"``).
- `upper` — read the upper rather than the lower triangle.
- `simplify` — concatenate the per-study results.
- `rowid` — reorder the rows of ``dat``/``V``.
- `vnames` — names for the four data columns.
- `noid` — drop the study-id column.

**Returns**

`RcalcResult` (or a list of them when ``simplify=False``).

## `RcalcResult`

```python
RcalcResult(dat, V)
```

Result of `rcalc` — ``dat`` (long table) and ``V`` (covariance).

## `fitstats`

```python
fitstats(result, REML=None)
```

*Mirrors R's `fitstats`.*

``metafor::fitstats`` / ``$fit.stats`` — logLik, deviance, AIC, BIC, AICc.

## `FitstatsResult`

```python
FitstatsResult(ML, REML, method, parms, k, p)
```

Fit statistics of a constant- or random-effects model.

Mirrors the ``$fit.stats`` matrix of a ``rma`` object: rows ``ll``, ``dev``,
``AIC``, ``BIC``, ``AICc``; columns ``ML`` and ``REML``.

## `aic_rma`

```python
aic_rma(result, k=2, correct=False)
```

*Mirrors R's `AIC.rma`.*

``metafor::AIC.rma`` / ``AIC.matreg`` — AIC (or AICc).

## `bic_rma`

```python
bic_rma(result)
```

*Mirrors R's `BIC.rma`.*

``metafor::BIC.rma`` / ``BIC.matreg`` — BIC.

## `se_rma`

```python
se_rma(result, type='beta')
```

*Mirrors R's `se.rma`.*

``metafor::se.rma`` — standard errors of the coefficients.

## `df_residual_rma`

```python
df_residual_rma(result)
```

*Mirrors R's `df.residual.rma`.*

``metafor::df.residual.rma`` — ``k.eff - p.eff``.

## `model_matrix_rma`

```python
model_matrix_rma(result, asdf=False)
```

*Mirrors R's `model.matrix.rma`.*

``metafor::model.matrix.rma`` — the design matrix used in the fit.

## `as_data_frame_list_rma`

```python
as_data_frame_list_rma(result)
```

*Mirrors R's `as.data.frame.list.rma`.*

``metafor::as.data.frame.list.rma`` — study-level result data frame.

## `cooks_distance_rma`

```python
cooks_distance_rma(result)
```

*Mirrors R's `cooks.distance.rma.uni`.*

``metafor::cooks.distance.rma.uni`` — Cook's distance per study.

## `sigma_matreg`

```python
sigma_matreg(result, REML=True)
```

*Mirrors R's `sigma.matreg`.*

``metafor::sigma.matreg`` — residual SD of a ``matreg`` fit.

## `update_rma`

```python
update_rma(result, mods=None, **kwargs)
```

*Mirrors R's `update.rma`.*

``metafor::update.rma`` — refit with modified arguments.

Only re-specification of the moderators (``mods``) and the simple scalar
arguments supported by `.rma_uni.rma` can be changed.

## `blup`

```python
blup(result, level=None)
```

*Mirrors R's `blup.rma.uni`.*

``metafor::blup.rma.uni`` — best linear unbiased predictions.

``pred_i = l_i y_i + (1 - l_i) X_i b`` with ``l_i = tau2 / (tau2 + v_i)``;
``var(pred_i) = l_i v_i + (1 - l_i)^2 X_i V X_i'``.

## `BlupResult`

```python
BlupResult(pred, se, pi_lb, pi_ub, slab)
```

*Mirrors R's `blup.rma.uni`.*

Result of `blup` (``metafor::blup.rma.uni``).

## `as_data_frame_confint`

```python
as_data_frame_confint(result, include_na=False)
```

*Mirrors R's `as.data.frame.confint.rma`.*

``metafor::as.data.frame.confint.rma`` — CIs as a data frame.

Accepts a `.pooling.Tau2Confint`, a `VifResult`, a
``matreg`` fit, an ``anova``-style result or a plain dict of columns;
returns a list of dicts with the columns ``estimate``/``ci.lb``/``ci.ub``
(metafor's names) when available.

## `as_matrix_list_rma`

```python
as_matrix_list_rma(result)
```

*Mirrors R's `as.matrix.list.rma`.*

``metafor::as.matrix.list.rma`` — study-level matrix of a fitted model.

## `se`

```python
se(result, type='beta')
```

*Mirrors R's `se`.*

``metafor::se`` / ``se.rma`` / ``se.default`` — coefficient SEs.

## `as_data_frame_list_anova`

```python
as_data_frame_list_anova(result)
```

*Mirrors R's `as.data.frame.list.anova.rma`.*

``metafor::as.data.frame.list.anova.rma`` — anova table as a data frame.

## `as_data_frame_list_confint`

```python
as_data_frame_confint(result, include_na=False)
```

*Mirrors R's `as.data.frame.confint.rma`.*

``metafor::as.data.frame.confint.rma`` — CIs as a data frame.

Accepts a `.pooling.Tau2Confint`, a `VifResult`, a
``matreg`` fit, an ``anova``-style result or a plain dict of columns;
returns a list of dicts with the columns ``estimate``/``ci.lb``/``ci.ub``
(metafor's names) when available.

## `as_data_frame_vif`

```python
as_data_frame_vif(result)
```

*Mirrors R's `as.data.frame.vif.rma`.*

``metafor::as.data.frame.vif.rma`` — VIF table as a data frame.

## `inches2units`

```python
inches2units(x, units='in', dpi=300)
```

*Mirrors R's `inches2units`.*

``meta::inches2units`` — convert inches to ``in``/``cm``/``mm``/``px``.

## `units2inches`

```python
units2inches(x, units='in', dpi=300)
```

*Mirrors R's `units2inches`.*

``meta::units2inches`` — inverse of `inches2units`.

## `jama_labels`

```python
jama_labels(author, year, citation)
```

*Mirrors R's `JAMAlabels`.*

``meta::JAMAlabels`` — JAMA-style study labels (``Author^ref (year)``).

## `ci`

```python
ci(te, seTE, level=0.95, df=None, null_effect=0.0)
```

*Mirrors R's `ci`.*

``meta:::ci`` — confidence interval and test for an effect estimate.

**Args**

- `te` — effect estimate(s).
- `seTE` — standard error(s).
- `level` — confidence level.
- `df` — degrees of freedom; ``None`` (or ``inf``) uses the normal distribution, a positive value the t distribution.
- `null_effect` — null value of the effect measure.

**Returns**

dict with ``TE``, ``seTE``, ``lower``, ``upper``, ``statistic``, ``p``, ``level``, ``df`` and ``null.effect``.

## `getmfopt`

```python
getmfopt(x=None, default=None)
```

*Mirrors R's `getmfopt`.*

``metafor::getmfopt`` — read a ``metafor`` global option.

## `setmfopt`

```python
setmfopt(**kwargs)
```

*Mirrors R's `setmfopt`.*

``metafor::setmfopt`` — set ``metafor`` global options.

## `estimates`

```python
estimates(result, study_results=True, common=True, random=True, prediction=True, backtransf=False)
```

*Mirrors R's `estimates`.*

``meta::estimates`` — tabulate study and pooled estimates.

**Args**

- `result` — a `.metagen.MetagenResult`.
- `study_results` — include the per-study rows.
- `common, random, prediction` — include the corresponding pooled rows.
- `backtransf` — report on the back-transformed scale.

**Returns**

`EstimatesResult`.

## `EstimatesResult`

```python
EstimatesResult(rows, sm, backtransf)
```

*Mirrors R's `estimates`.*

Result of `estimates` (``meta::estimates``).

A study-level table with the pooled rows appended.  ``rows`` is a list of
dicts with the columns ``studlab``, ``TE``, ``seTE``, ``lower``, ``upper``,
``statistic``, ``pval`` and ``type`` (``"study"``, ``"common"``,
``"random"`` or ``"prediction"``).

## `forest_args`

```python
forest_args(common=True, random=True, prediction=False, overall=False, text_common='Common effect model', text_random='Random effects model', text_predict='Prediction interval', text_overall='Overall effect')
```

*Mirrors R's `.forestArgs`.*

``meta::.forestArgs`` — labels/rows used by the forest plot.

Returns a dict with the row labels in the order they appear in a forest
plot, plus the ``common``/``random``/``prediction``/``overall`` flags.

## `MissingDataWarning`

Raised when rows are dropped because of missing values.

## `complete_cases`

```python
complete_cases(columns, *, context: 'str' = 'rows', warn: 'bool' = True)
```

Row mask that keeps only complete cases, plus the dropped indices.

**Args**

- `columns` — ``{name: sequence}`` — every column must have the same length.
- `context` — noun used in the warning message ("studies", "rows", ...).
- `warn` — emit `MissingDataWarning` when anything is dropped.

**Returns**

``(keep, dropped)`` where ``keep`` is a boolean mask over the original rows and ``dropped`` holds the 0-based indices that were removed.

**Raises**

- `ValueError` — when the columns do not all have the same length.

## `is_missing`

```python
is_missing(values) -> 'np.ndarray'
```

Boolean mask of missing entries in ``values``.

Numeric arrays use ``~np.isfinite`` (so ``nan``, ``inf`` and ``-inf`` all
count as missing, as R's ``NA`` does after arithmetic).  Object arrays treat
``None`` and ``float('nan')`` as missing.  Integer and boolean arrays have no
missing values and yield an all-``False`` mask.

**Args**

- `values` — a 1-D array-like.

**Returns**

A boolean ``ndarray`` with one entry per element.
