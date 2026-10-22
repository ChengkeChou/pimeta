# Multilevel and generalised models

## `rma.mv` — multilevel / multivariate

```python
import numpy as np
import pimeta as ma

yi = np.array([0.20, 0.40, 0.10, 0.50, 0.30, 0.15])
vi = np.array([0.04, 0.05, 0.03, 0.06, 0.02, 0.05])
study = np.array(["A", "A", "B", "B", "C", "C"])

fit = ma.rma_mv(yi, vi, random="~ 1 | study", data={"study": study})

print(fit.sigma2)                       # variance components
print(fit.beta, fit.se, fit.ci_lb, fit.ci_ub)
print(fit.qe, fit.qe_df, fit.qe_p)      # residual heterogeneity
print(fit.ll)                           # log-likelihood
```

`random` uses metafor's `~ inner | outer` syntax. Nesting takes several terms:

```python
fit = ma.rma_mv(yi, vi, random="~ 1 | district/school", data=dat)
```

Known-variance matrices go through `V=`:

```python
fit = ma.rma_mv(yi, V=V, random="~ 1 | study", data={"study": study})
```

Structure and correlation of the inner term are set with `struct` (`"CS"`,
`"HCS"`, `"UN"`, `"AR"`, …) and `rho`; `sigma2=` pins variance components
(pass `float("nan")` to leave a component free and fix the others).

> **`level` is a percentage here.** `rma_mv(..., level=95)` follows
> `metafor::rma.mv()`, whereas `rma(..., level=0.95)` follows
> `metafor::rma()`. That inconsistency is R's, and pimeta reproduces it.

### Diagnostics

The same generics that work on an `rma` fit dispatch on an `rma_mv` fit:

```python
import pimeta as ma

print(ma.hatvalues(fit))         # leverage per study
print(ma.weights(fit))           # k x k weight matrix
print(ma.rstandard(fit))         # standardised residuals by component
print(ma.dfbetas(fit))           # influence on each coefficient
print(ma.cooks_distance(fit))    # Cook's distance
```

### Profiling a variance component

```python
import pimeta as ma

prof = ma.profile_rmamv(fit, component=1, steps=20)
ma.plot_mv_profile(prof)
```

### Robust variance estimation

Cluster-robust (sandwich) standard errors work on both fit types — a `rma`
fit is the special case where the marginal covariance happens to be diagonal.

```python
import pimeta as ma

fit = ma.rma(yi, vi=vi, data=df)
rob = ma.robust(fit, cluster=df["study"])
print(rob["se"], rob["pval"], rob["ci_lb"], rob["ci_ub"])

# Small-sample variants (metafor's CR0 / CR1 / CR1.S1 / CR1.S2).
ma.robust(fit, cluster=df["study"], adjust=False)        # CR0
ma.robust(fit, cluster=df["study"], adjust="Stata2")     # CR1.S2

# Bias-reduced CR2/CR3 with Satterthwaite degrees of freedom
# (clubSandwich's default).
ma.robust(fit, cluster=df["study"], clubSandwich=True,
          vcov="CR2", coef_test="Satterthwaite")
```

The result is a dict: `vbest`, `beta`, `se`, `zval`, `pval`, `ci_lb`, `ci_ub`,
`dfs`, `n` and `vb`. For a `rma` fit on the metafor path it also carries the
omnibus test as `QM`, `QMdf` and `QMp`.

For an `rma.mv` fit the `cluster=` argument can be omitted when the model
stored one. For an `rma` fit it is required, as in R.

### Random effects and model comparison

```python
import pimeta as ma

print(ma.ranef(fit))               # BLUPs
print(ma.anova_rma(fit_a, fit_b))  # likelihood-ratio comparison
```

## `rma.glmm` — generalised linear mixed models

For binary and count data with exactly modelled within-study variability:

```python
import pimeta as ma

fit = ma.rma_glmm("OR", ai=ai, bi=bi, ci=ci, di=di)
print(fit.b[0], fit.se[0])
print(fit.tau2, fit.I2, fit.QE_Wld, fit.QEp_Wld)
```

`measure` may be `"OR"`, `"RR"`, `"RD"`, `"IRR"`, `"IRD"` or `"PLO"`; models
are `"UM.FS"` (default), `"UM.RS"`, `"CM.EL"` and `"CM.AL"` via `model=`.
`nAGQ=` controls the adaptive Gauss–Hermite quadrature points.

## Aggregating dependent effect sizes

If a study contributes several dependent effect sizes, `aggregate_escalc`
collapses them with a within-cluster correlation, exactly as
`metafor::aggregate.escalc()` does:

```python
import pimeta as ma

one_row_per_study = ma.aggregate_escalc(d, cluster=study, struct="CS", rho=0.5)
```

## Which model should I use?

| Situation | Model |
|---|---|
| independent studies, one effect each | `rma` |
| several effects per study, or nested clustering | `rma_mv` |
| binary counts with sparse cells | `rma_glmm` |
| between-study variance depends on moderators | `rma(..., scale=~ z)` — see [Selection and scale models](selection-models.md) |
| publication bias modelled explicitly | `selmodel` — see [Selection and scale models](selection-models.md) |
