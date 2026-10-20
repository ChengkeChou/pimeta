# Effect sizes

`escalc` is a transcription of `metafor::escalc()`. It takes arm-level data and
returns effect sizes (`yi`) and sampling variances (`vi`).

```python
import pimeta as ma

d = ma.escalc("RR", ai=ai, bi=bi, ci=ci, di=di)
print(d.yi, d.vi)
```

## Measures

| Data shape | Measures |
|---|---|
| 2×2 counts (`ai, bi, ci, di`) | `RR`, `OR`, `RD`, `AS`, `PETO`, `D2OR`, `RPB`, `RBIS` |
| two-arm continuous (`n1i, m1i, sd1i, n2i, m2i, sd2i`) | `MD`, `SMD`, `SMDH`, `ROM`, `CVR` |
| two-arm counts over time (`x1i, t1i, x2i, t2i`) | `IRR`, `IRD`, `IRSD` |
| single proportion (`xi, ni`) | `PR`, `PLN`, `PLO`, `PAS`, `PFT`, `PRAW`, `PD` |
| correlation (`ri, ni`) | `COR`, `UCOR`, `ZCOR`, `ZPCOR` |
| means (`mi, ni, sdi`) | `MN`, `SMN`, `MRAW` |

Ratio measures (`RR`, `OR`, `IRR`, `PLN`, `PLO`, `PETO`) are returned on the
**log** scale. Back-transform with `math.exp()`.

## Continuity and zero cells

```python
import pimeta as ma

# Default: add 0.5 to the cells of studies that contain a zero.
d = ma.escalc("RR", ai=ai, bi=bi, ci=ci, di=di, add=0.5)

# No correction at all.
d = ma.escalc("RR", ai=ai, bi=bi, ci=ci, di=di, add=0)

# Drop studies with zero events in both arms (metafor's default for RR).
d = ma.escalc("RR", ai=ai, bi=bi, ci=ci, di=di, drop00=True)
```

`PETO` needs no correction and ignores `add`.

## Small-sample corrections

```python
import pimeta as ma

# Hedges' g (default, vtype="LS") or the "AV" variant.
g = ma.escalc("SMD", n1i=n1i, m1i=m1i, sd1i=sd1i,
              n2i=n2i, m2i=m2i, sd2i=sd2i, vtype="LS")
g_av = ma.escalc("SMD", n1i=n1i, m1i=m1i, sd1i=sd1i,
                 n2i=n2i, m2i=m2i, sd2i=sd2i, vtype="AV")

# Fisher's z for correlations.
z = ma.escalc("ZCOR", ri=ri, ni=ni)
```

## With a data frame

```python
import pimeta as ma

df = {"ai": [3, 6, 5], "bi": [52, 48, 55], "ci": [8, 10, 9], "di": [47, 42, 51]}
d = ma.escalc("RR", "ai", "bi", "ci", "di", data=df)
```

## Transformations

All 58 `metafor::transf.*` functions are available for presentation:

```python
import pimeta as ma

ma.transf_ilogit(0.5)       # logit -> probability
ma.transf_ztor(0.3)         # Fisher's z -> r
ma.transf_dtou1(0.4)        # Cohen's d -> U1 (percentile)
ma.transf_dtou3(0.4)        # Cohen's d -> U3
```

All of them take a scalar or an array and are vectorised with NumPy.

## Stacking and reshaping

```python
import pimeta as ma

stacked = ma.rbind_escalc(d1, d2)     # rbind()
wide = ma.cbind_escalc(d1, d2)        # cbind()

# Collapse dependent effect sizes within a cluster (rma.mv-style).
aggregated = ma.aggregate_escalc(d, cluster=study)
```

`aggregate_escalc` fits a multilevel model with a within-cluster correlation
(`struct="CS"`, `rho=0.5` by default) and returns one row per cluster, which is
what `metafor::aggregate.escalc()` does.

See the [API reference](../api/effect-sizes.md) for every measure and argument.
