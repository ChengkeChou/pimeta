# Quick start

A complete analysis in five steps. Every snippet is self-contained — copy it
into a script or a notebook as-is.

## 1. Effect sizes from raw 2×2 data

`escalc` mirrors `metafor::escalc()`: give it arm-level counts and it returns
effect sizes (`yi`) and variances (`vi`).

```python
import pimeta as ma

ai = [3, 6, 5, 8, 4, 7]          # events, treatment
bi = [52, 48, 55, 47, 50, 45]    # non-events, treatment
ci = [8, 10, 9, 11, 7, 9]        # events, control
di = [47, 42, 51, 40, 49, 44]    # non-events, control

d = ma.escalc("RR", ai=ai, bi=bi, ci=ci, di=di)

print(d.yi)   # log risk ratios
print(d.vi)   # sampling variances
```

Ratio measures (`RR`, `OR`, `IRR`, `PLN`, `PLO`, `PETO`) come back on the log
scale, so back-transform with `math.exp()` when you report them.

## 2. Pool the studies

```python
import pimeta as ma

d = ma.escalc("RR", ai=[3, 6, 5, 8], bi=[52, 48, 55, 47],
              ci=[8, 10, 9, 11], di=[47, 42, 51, 40])

r = ma.rma(d.yi, d.vi, method="REML")

print(r.b[0])       # pooled log RR
print(r.ci_lb[0])   # lower bound
print(r.ci_ub[0])   # upper bound
print(r.tau2)       # between-study variance
print(r.I2)         # proportion of variability due to heterogeneity (%)
print(r.QE, r.QEp)  # Cochran's Q and its p-value
```

`method` selects the τ² estimator: `DL`, `HE`, `HS`, `SJ`, `ML`, `REML`, `PM`,
`EB`, `GENQ`, `GENQM`. Use `test="knha"` for the Hartung–Knapp adjustment.

### Common-effect model

```python
import pimeta as ma

yi = [0.20, 0.40, 0.10, 0.50]
vi = [0.04, 0.05, 0.03, 0.06]

fixed = ma.rma(yi, vi, method="FE")
print(fixed.b[0], fixed.ci_lb[0], fixed.ci_ub[0])
```

## 3. Draw a forest plot

```python
import pimeta as ma

yi = [0.20, 0.40, 0.10, 0.50]
vi = [0.04, 0.05, 0.03, 0.06]

r = ma.rma(yi, vi, method="REML")
ax = ma.forest(yi, vi, studlab=["Study A", "Study B", "Study C", "Study D"],
               pooled=r, effect_type="RR", filename="forest.png")
print(type(ax).__name__)    # Axes
```

Every figure function returns the `matplotlib.axes.Axes` it drew on, so you can
compose panels with `plt.subplots` or restyle any artist:

```python
import matplotlib.pyplot as plt
import pimeta as ma

yi = [0.20, 0.40, 0.10, 0.50]
vi = [0.04, 0.05, 0.03, 0.06]

fig, axes = plt.subplots(1, 2, figsize=(11, 4))
ma.forest(yi, vi, ax=axes[0])
ma.funnel(yi, vi, ax=axes[1])
```

## 4. Explain the heterogeneity

```python
import pimeta as ma

yi = [0.20, 0.40, 0.10, 0.50, 0.30, 0.15]
vi = [0.04, 0.05, 0.03, 0.06, 0.02, 0.05]
dose = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]

r = ma.rma(yi, vi, mods=dose, method="REML")

print(r.b)                 # intercept and slope
print(r.QM, r.QMp)         # test of the moderator
print(r.R2)                # proportion of heterogeneity explained
```

Or with a formula and a data frame, exactly like R:

```python
import pimeta as ma

df = {
    "yi": [0.20, 0.40, 0.10, 0.50, 0.30, 0.15],
    "vi": [0.04, 0.05, 0.03, 0.06, 0.02, 0.05],
    "dose": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
    "arm": ["A", "B", "A", "B", "A", "B"],
}

r = ma.rma("yi", "vi", mods="~ dose + arm", data=df)
print(r.b)
```

## 5. Check for publication bias

```python
import pimeta as ma

yi = [0.20, 0.40, 0.10, 0.50, 0.30, 0.15, 0.25, 0.35, 0.05, 0.45]
vi = [0.04, 0.05, 0.03, 0.06, 0.02, 0.05, 0.04, 0.03, 0.06, 0.02]

print(ma.regtest(yi, vi).p)                  # Egger's test
print(ma.begg_test(yi, vi).p)                # Begg's rank test
print(ma.trimfill(yi, vi).k0)                # trim-and-fill: filled studies
print(ma.fsn(yi, vi, type="Rosenthal").fsnum)
```

With raw 2×2 data, `metabias` runs any of the nine tests from the `meta`
package on a fitted binary model:

```python
import pimeta as ma

event_e = [3, 6, 5, 8, 4, 7, 2, 9, 5, 6]
n_e = [55, 54, 60, 55, 54, 52, 53, 53, 54, 52]
event_c = [8, 10, 9, 11, 7, 9, 10, 8, 11, 7]
n_c = [55, 52, 60, 51, 56, 53, 53, 55, 51, 55]

mb = ma.metabin(event_e, n_e, event_c, n_c, sm="OR")
print(ma.metabias(mb, method_bias="Harbord").pval)
```

## 6. Network meta-analysis

```python
import numpy as np
import pimeta as ma

TE = np.array([0.10, -0.20, 0.30, 0.15, -0.05, 0.25])
seTE = np.array([0.10, 0.12, 0.11, 0.09, 0.13, 0.10])
treat1 = np.array(["A", "A", "B", "B", "C", "A"])
treat2 = np.array(["B", "C", "C", "D", "D", "D"])
studlab = np.array([f"S{i + 1}" for i in range(TE.size)])

n = ma.netmeta(TE, seTE, treat1, treat2, studlab, sm="MD")

print(n.TE_random)                          # league-style contrast matrix
print(ma.netrank(n).ranking_random)         # P-scores
print(ma.netmeasures(n).proportion)         # direct-evidence proportions
ma.netgraph(n, filename="network.png")
```

## Next steps

- [Coming from R](migration-from-r.md) — how each R idiom maps.
- [Data frames and formulas](guide/data-frames.md) — pass column names instead
  of arrays.
- [Interoperability](guide/interop.md) — tidy tables, scikit-learn, statsmodels.
- [API reference](api/index.md) — the full surface.
