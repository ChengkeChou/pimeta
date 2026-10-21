# Selection and scale models

Two model families let the *heterogeneity* rather than the mean vary across
studies.

## Location-scale models — `rma.ls`

metafor has no `rma.ls()` function: the model is fitted by passing
`scale = ~ z` to `rma()`. pimeta does the same and returns an `RmaLsResult`.

```python
import pimeta as ma

yi = [0.20, 0.40, 0.10, 0.50, 0.30, 0.15]
vi = [0.04, 0.05, 0.03, 0.06, 0.02, 0.05]
dose = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]

r = ma.rma(yi, vi, mods=dose, scale=dose, method="REML")

print(r.b)              # location coefficients
print(r.alpha)          # scale coefficients (log link by default)
print(r.se_alpha)
print(r.QS, r.QSp)      # test of the scale coefficients
print(r.tau2)           # per-study vector, not a scalar
```

The model is τ²ᵢ = exp(zᵢ′α) with weights 1 / (vᵢ + τ²ᵢ). Because τ² varies by
study, `tau2`, `I2` and `H2` are **vectors of length k**.

- `link="log"` (default) gives τ²ᵢ = exp(zᵢ′α); `link="identity"` gives zᵢ′α.
- `method` must be `"ML"` or `"REML"` (metafor imposes the same restriction).
- `scale_intercept=False` drops the scale intercept.
- `optbeta=True` estimates location and scale jointly instead of profiling β out.

### Profiling the scale coefficients

```python
import pimeta as ma

prof = ma.profile_rma_ls(r, steps=20)
ma.plot_mv_profile(prof)

print(ma.confint_rma_ls(r))
```

> **Convergence.** metafor stops `optim()` on a relative *function* tolerance, so
> its gradient at the reported estimate is around 1e-4 (log link) to 1e-3
> (identity). pimeta adds a damped Newton polish and drives the gradient to
> about 1e-9. The scale coefficients therefore differ from R in the fifth
> significant digit, and pimeta's log-likelihood is always ≥ R's. See
> [Differences from R](../reference/differences.md).

## Selection models — `rma.uni.selmodel`

`selmodel` fits `metafor::rma.uni.selmodel()` on an existing `rma` fit: the
probability that a study is published depends on its p-value.

```python
import pimeta as ma

yi = [0.20, 0.40, 0.10, 0.50, 0.30, 0.15]
vi = [0.04, 0.05, 0.03, 0.06, 0.02, 0.05]

base = ma.rma(yi, vi, method="REML")
fit = ma.selmodel(base, "beta")

print(fit.delta, fit.se_delta, fit.pval_delta)
print(fit.LRT, fit.LRTdf, fit.LRTp)     # test of selection
print(fit.ll, fit.ll0)                  # with and without selection
```

Supported `type` values (prefix-matched, as in R):

| Family | Types |
|---|---|
| one-parameter | `beta`, `halfnorm`, `negexp`, `logistic`, `power`, `negexppow` |
| with an extra shift | `halfnorm1`, `negexp1`, `logistic1`, `power1` |
| two-parameter | `halfnorm2`, `negexp2`, `logistic2`, `power2` |
| step functions | `stepfun`, `stepcon` (needs `steps=`) |
| truncation | `trunc`, `truncest` |

Other arguments:

```python
import pimeta as ma

fit = ma.selmodel(
    base, "stepfun",
    steps=[0.025, 0.05],       # p-value cut points
    alternative="two.sided",   # "greater" (default), "less"
    prec="sei",                # precision weights
    subset=mask,               # restrict the modelled studies
)
```

### Profiling the selection parameter

```python
import pimeta as ma

prof = ma.profile_selmodel(fit, n_points=20)
ma.plot_selprofile(prof)
```

> **Difference from R.** metafor fixes τ² at the full-model estimate on every
> grid point; pimeta re-estimates it, which is the proper profile likelihood.
> The curve can differ in the third or fourth digit.

### Convergence and the delta parameterisation

metafor optimises each `delta` through a map onto `[delta.min, delta.max]`.
There are two of them — `delta.transf.fun` (the exponential) and the `mapfun`
that is actually applied — and the exponential is only used when `delta.max`
is unbounded or the type forces it. Every smooth family has `delta.max = 100`,
so the map is the **scaled logistic** `lb + (ub - lb) * plogis(x)`.
`stepfun(decreasing=True)` and `stepcon` use the cumulative parameterisation
instead, which keeps the step weights monotone by construction; pimeta does the
same for both, so the two agree exactly. Setting `control={"delta.max": float("inf")}`
forces the exponential map, exactly as in R.

The truncated and step-function likelihoods are piecewise constant in the
cut-point: the score is exactly zero across whole intervals and `optim()` stops
on such a plateau. R's own `truncest` and `negexppow` fits are not the maxima of
their objectives. pimeta adds one derivative-free restart after the gradient
run, so its log-likelihood is ≥ R's for those types. The smooth families agree
to about $10^{-6}$ (the residual is R's adaptive-quadrature tolerance). See
[Differences from R](../reference/differences.md).

## Which one?

| Question | Model |
|---|---|
| Does the *effect* depend on a moderator? | `rma(..., mods=~ z)` — [Meta-regression](meta-regression.md) |
| Does the *heterogeneity* depend on a moderator? | `rma(..., scale=~ z)` |
| Is the effect over-estimated because small studies are missing? | `selmodel` |
| Do I just want to adjust for funnel asymmetry? | `trimfill`, `regtest` — [Publication bias](publication-bias.md) |
