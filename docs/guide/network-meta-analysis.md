# Network meta-analysis

`netmeta` ports `netmeta::netmeta()`. Estimates are `n × n` matrices indexed by
treatment name: `TE[i, j]` is the effect of treatment *i* relative to
treatment *j*.

```python
import numpy as np
import pimeta as ma

TE = np.array([0.10, -0.20, 0.30, 0.15, -0.05, 0.25])
seTE = np.array([0.10, 0.12, 0.11, 0.09, 0.13, 0.10])
treat1 = np.array(["A", "A", "B", "B", "C", "A"])
treat2 = np.array(["B", "C", "C", "D", "D", "D"])
studlab = np.array([f"S{i + 1}" for i in range(TE.size)])

n = ma.netmeta(TE, seTE, treat1, treat2, studlab, sm="MD")

print(n.trts)              # treatment order
print(n.TE_random)         # random-effects contrast matrix
print(n.seTE_random)
print(n.Q, n.Q_heterogeneity, n.Q_inconsistency)
print(n.tau2, n.I2)
```

## From a data frame

```python
import pimeta as ma

df = {"TE": TE, "seTE": seTE, "treat1": treat1, "treat2": treat2,
      "studlab": studlab}

n = ma.netmeta("TE", "seTE", "treat1", "treat2", "studlab", data=df, sm="MD")
```

## Ranking

```python
import pimeta as ma

rk = ma.netrank(n)
print(rk.ranking_random)     # P-scores (netmeta names the field "ranking.random")
print(rk.Pmatrix_random)     # pairwise probability matrix

ma.rankogram(n)              # rank-probability plot
ma.netposet(n)               # partial order of treatments
ma.hasse(n)
ma.traffic_light(n)          # certainty-of-evidence traffic lights
```

> **P-scores, not ranks.** Despite the name, `ranking_random` holds P-scores
> (higher is better), matching netmeta 3.7-0.

## Direct evidence

```python
import pimeta as ma

print(ma.netmeasures(n).proportion)     # direct-evidence proportions
print(ma.netcontrib(n))                 # contribution matrix
print(ma.netmatrix(n))                  # adjacency / weight matrix
print(ma.netcomparison(n))              # direct vs indirect per comparison
```

## Inconsistency

```python
import pimeta as ma

print(ma.decomp_design(n))              # Q decomposition: within vs between designs
print(ma.netsplit(n))                   # direct vs indirect by comparison
print(ma.netheat(n))                    # heat plot of inconsistency
ma.heatplot(n)
```

## Regression and impact

```python
import pimeta as ma

reg = ma.netmetareg(n, mods=dose)
print(reg.b, reg.pval)

print(ma.netimpact(n))                  # study-level influence
print(ma.netdistance(n))                # evidence distances
print(ma.netconnection(n))              # connectivity check
```

## Combining and simplifying

```python
import pimeta as ma

print(ma.netcomplexity(n))

combined = ma.netcomb(n, treat=["B", "C"])   # combine treatments
simple = ma.discomb(n, inactive="D")         # drop / merge nodes

merged = ma.netbind(n1, n2)                  # bind two networks
```

## Figures

```python
import pimeta as ma

ma.netgraph(n, filename="network.png")   # evidence graph
ma.netheat(n, filename="heat.png")       # inconsistency heat map
ma.rankogram(n, filename="rank.png")     # rank probabilities
ma.netposet(n)
```

Every figure returns a matplotlib `Axes`, so it composes with subplots like the
rest of the package.

## Interoperability

```python
from pimeta.interop import to_networkx, to_xarray

graph = to_networkx(n)     # networkx Graph: nodes = treatments, edges = comparisons
dataset = to_xarray(n)     # xarray Dataset with ("treat1", "treat2") dims
print(dataset["TE_random"].sel(treat1="A", treat2="D"))
```

See [Interoperability](interop.md).
