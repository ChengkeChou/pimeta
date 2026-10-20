# Heterogeneity

Every quantity `metafor` reports for heterogeneity is on the fitted object:

```python
import math
import pimeta as ma

yi = [0.20, 0.40, 0.10, 0.50, 0.30, 0.15]
vi = [0.04, 0.05, 0.03, 0.06, 0.02, 0.05]

r = ma.rma(yi, vi, method="REML")

print(r.QE, r.QEp)              # Cochran's Q and its p-value (df = k - p)
print(r.tau2, math.sqrt(r.tau2))  # between-study variance / SD
print(r.I2, r.H2)               # percentages
print(r.vt)                     # typical within-study variance
print(r.R2)                     # only for models with moderators
```

> **Percentages.** `I2`, `H2` and `R2` are on the 0–100 scale, exactly like
> R's `r$I2`. Do not multiply by 100.

## Interpreting the numbers

| Quantity | Meaning |
|---|---|
| `QE` | Cochran's Q — weighted squared deviation from the pooled estimate |
| `I2` | share of total variability attributable to heterogeneity |
| `H2` | `QE / (k - p)` — the ratio form of `I2` |
| `R2` | share of the *between-study* variance explained by the moderators |
| `tau2` | absolute between-study variance; needed for prediction intervals |

## Subgroups

A factor moderator is just a dummy-coded design matrix. Use a formula and let
pimeta build it the way R does:

```python
import pimeta as ma

df = {
    "yi": yi, "vi": vi,
    "group": ["A", "A", "B", "B", "C", "C"],
}

r = ma.rma("yi", "vi", mods="~ group", data=df)
print(r.b)                 # intercept = group A, dummies for B and C
print(r.QM, r.QMdf, r.QMp) # omnibus test of the subgroup factor
print(r.R2)
```

To test only a subset of coefficients, use `btt` (1-based indices, as in R):

```python
import pimeta as ma

r = ma.rma(yi, vi, mods=[g1, g2, g3], btt=[2, 3])
print(r.QM, r.QMp)
```

## Permutation tests

`permutest` gives a permutation-based p-value for the moderator test, which
does not rely on the χ² approximation:

```python
import pimeta as ma

p = ma.permutest(r, iter=1000, seed=42)
print(p.pval)

ma.plot_permutest(p)     # distribution of the permuted statistic
```

## Leave-one-out and influence

```python
import pimeta as ma

loo = ma.leave1out(yi, vi, method="REML")
print(loo.estimate, loo.I2)

loo[0:3]          # slice rows; every vector stays aligned
ma.head(loo, 3)

inf = ma.influence(yi, vi)
print(inf.cook_d, inf.dfbs, inf.hat)
ma.plot_influence(inf)

# Cumulative meta-analysis in study order.
cu = ma.cumul(yi, vi, order=list(range(len(yi))))
print(cu.estimate, cu.k)
```

## Excess significance

```python
import pimeta as ma

print(ma.tes(yi, vi).pval)
```

## Multicollinearity

```python
import pimeta as ma

mod = ma.rma(yi, vi, mods=[[dose, year], [dose2, year2]])
print(ma.vif(mod))     # generalised variance inflation factors
```

`vif` needs at least two moderators; it raises on an intercept-only model.

## Visual checks

```python
import pimeta as ma

ma.qqnorm_rma(r)       # normal Q-Q plot of the residuals
ma.baujat(yi, vi)      # Baujat plot: contribution to Q vs influence
ma.plot_gosh(ma.gosh(yi, vi, subsets=1000, seed=1))
```

## Next

- [Meta-regression](meta-regression.md) — continuous moderators and `R2`.
- [Publication bias](publication-bias.md) — funnel plots and trim-and-fill.
