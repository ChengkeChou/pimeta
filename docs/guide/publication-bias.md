# Publication bias

## Funnel and radial plots

```python
import pimeta as ma

yi = [0.20, 0.40, 0.10, 0.50, 0.30, 0.15, 0.25, 0.35, 0.05, 0.45]
vi = [0.04, 0.05, 0.03, 0.06, 0.02, 0.05, 0.04, 0.03, 0.06, 0.02]

ma.funnel(yi, vi)                       # funnel plot
ma.funnel(yi, vi, y_axis="precision")   # precision instead of SE
ma.radial(yi, vi)                       # Galbraith radial plot
ma.labbe(yi, vi)                        # L'Abbé plot
```

## Egger's test

`regtest` is `metafor::regtest()` — a weighted regression of the effect on a
predictor (standard error by default):

```python
import pimeta as ma

rt = ma.regtest(yi, vi, predictor="sei")
print(rt.stat, rt.p)           # z statistic and p-value
print(rt.slope, rt.intercept)  # regression coefficients
print(rt.tau)                  # residual heterogeneity

# The Egger variant used by metafor when a standard error is available.
ma.regtest(yi, vi, predictor="se")
```

## Begg's rank test

```python
import pimeta as ma

bg = ma.begg_test(yi, vi)
print(bg.tau, bg.z, bg.p)      # Kendall's tau, z, p
```

## Trim and fill

```python
import pimeta as ma

tf = ma.trimfill(yi, vi, estimator="L0")
print(tf.k0)                     # number of filled studies
print(tf.filled_effect)          # adjusted pooled estimate
print(tf.original_effect)        # unadjusted estimate
```

`estimator` is `"L0"` (default), `"R0"` or `"Q0"`.

## Failsafe N

```python
import pimeta as ma

print(ma.fsn(yi, vi, type="Rosenthal").fsnum)
print(ma.fsn(yi, vi, type="Orwin", target=0.1).fsnum)
print(ma.fsn(yi, vi, type="Rosenberg").fsnum)
```

## The nine `meta` tests

`metabias` ports `meta::metabias()`: all nine tests are available.

| Name | What it regresses on | Needs |
|---|---|---|
| `Begg` | rank correlation of standardised effects with variances | effect + SE |
| `Egger` | the effect on its standard error | effect + SE |
| `Thompson` | Egger, fitted as a random-effects model | effect + SE |
| `Schwarzer` | hypergeometric variant of the rank test | 2×2 counts |
| `Harbord` | score test from the 2×2 tables | 2×2 counts |
| `Peters` | regression on `1/n` | arm sizes |
| `Deeks` | regression on `1/sqrt(ESS)` | 2×2 counts |
| `Pustejovsky` | regression on `sqrt(1/n_e + 1/n_c)` | arm sizes |
| `Macaskill` | regression weighted by total sample size | arm sizes |

The first positional argument is either a fitted result or the effect sizes
themselves (with `seTE=`); counts travel through `event_e` / `event_c` /
`n_e` / `n_c`.

```python
import pimeta as ma

# Straight from effect sizes and standard errors (Egger by default).
res = ma.metabias(yi, seTE)
print(res.method_bias, res.statistic, res.pval)
print(res.intercept, res.se_intercept, res.tau, res.df)

# From a fitted binary model, choosing the test explicitly.
mb = ma.metabin(event_e, n_e, event_c, n_c, sm="OR")
res = ma.metabias(mb, method_bias="Harbord")
print(res.pval)
```

Tests that need sample sizes (`Peters`, `Macaskill`, `Harbord`, `Deeks`,
`Schwarzer`) are only available when arm-level counts are supplied — a
single-arm proportion study or an `rma` fit without counts raises a clear error
instead of returning a meaningless number.

## Excess significance and GOSH

```python
import pimeta as ma

print(ma.tes(yi, vi).pval)

g = ma.gosh(yi, vi, subsets=1000, seed=42)
ma.plot_gosh(g)
```

## Sensitivity to single studies

```python
import pimeta as ma

loo = ma.leave1out(yi, vi)
print(loo.estimate, loo.I2)
```

`leave1out` results slice natively (`loo[0:3]`, `ma.head(loo, 3)`), and
`influence()` gives Cook's distance, DFBS and hat values per study.
