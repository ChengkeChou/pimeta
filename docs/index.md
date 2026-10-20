# pimeta

**Pure-Python meta-analysis — a numerically verified port of R's
[metafor](https://cran.r-project.org/package=metafor),
[netmeta](https://cran.r-project.org/package=netmeta) and
[meta](https://cran.r-project.org/package=meta).**

No R installation, no `rpy2`, no third-party meta-analysis library underneath:
the numeric engine is a direct NumPy/SciPy transcription of the R reference
implementations, so results reproduce R to floating-point tolerance.

```bash
pip install pimeta
```

```python
import pimeta as ma

yi = [0.20, 0.40, 0.10, 0.50]
vi = [0.04, 0.05, 0.03, 0.06]

r = ma.rma(yi, vi, method="REML")

print(r.b[0], r.ci_lb[0], r.ci_ub[0])   # pooled estimate and 95% interval
print(r.tau2, r.I2, r.QE, r.QEp)        # heterogeneity
```

!!! tip "New here?"
    The [**usage manual**](usage.md) walks from a first pool through effect
    sizes, τ² estimators, meta-regression, publication bias, multilevel models,
    network meta-analysis, result tables, ecosystem interop, plots and the CLI.
    Every example there is executed and its real output recorded — and four
    [notebooks](https://github.com/ChengkeChou/pimeta/tree/main/examples) ship
    already executed if you would rather read than type.

## What is covered

| Area | Highlights |
|---|---|
| **Effect sizes** | 30 `escalc` measures (`RR`, `OR`, `RD`, `SMD`, `SMDH`, `COR`, `ZCOR`, `IRR`, `PLO`, `PFT`, …), continuity corrections, `drop00`, and all 58 `metafor::transf.*` |
| **Models** | `rma` / `rma.uni`, `rma.mv`, `rma.glmm`, `rma.ls` (location-scale), `rma.peto`, `rma.mh`, `rma.uni.selmodel`, 11 τ² estimators, Hartung–Knapp, prediction intervals |
| **Drivers** | `metagen`, `metabin`, `metacont`, `metacor`, `metainc`, `metaprop`, `metarate`, `metamean`, `metareg`, `trimfill`, `regtest`, `metabias` |
| **Network meta-analysis** | `netmeta`, `netmetabin`, `netcomb`, `discomb`, `netrank`, `netimpact`, `netsplit`, `netmeasures`, `netleaguematrix`, `netgraph`, `netheat`, … |
| **Publication bias** | `funnel`, `radial`, `labbe`, `trimfill`, `fsn`, `regtest`, `begg_test`, `metabias` (9 tests), `gosh`, `permutest`, `tes` |
| **Diagnostics** | `leave1out`, `influence`, `cooks_distance`, `dfbetas`, `hatvalues`, `rstandard`, `rstudent`, `baujat`, `profile_rma` |
| **Figures** | `forest`, `funnel`, `radial`, `labbe`, `bubble`, `qqnorm_rma`, `netgraph`, `netheat`, `rankogram`, `traffic_light` — all returning a matplotlib `Axes` |
| **Readers** | RevMan `.rm5`, Cochrane `.cdir` / `.mtv` |
| **Interoperability** | data frames, R-style formulas, tidy tables, scikit-learn estimators, statsmodels results, xarray, networkx |

About 390 public names in total — see the [API reference](api/index.md).

## Design rules

1. **Same names as R.** Result objects expose the `$` fields of the R object
   (`r.b`, `r.se`, `r.tau2`, `r.I2`, `r.ci_lb`), so R code and R textbooks
   translate almost one-to-one. R's `generic.class` method names are bridged via
   `pimeta.r_methods()` — see [R method-name map](reference/r-method-map.md).
2. **Verified, not approximated.** Every quantity is compared against R
   (`metafor` 5.2-1, `netmeta` 3.7-0, `meta` 8.5-0) during development. Typical
   agreement is below `1e-12`; the largest honest deviation is `1e-5`, in
   `rma.ls`, where pimeta converges further than R does. Details in
   [Differences from R](reference/differences.md).
3. **No runtime surprises.** `import pimeta` needs only NumPy, SciPy and
   matplotlib. Everything else — pandas, polars, pyarrow, scikit-learn,
   statsmodels, xarray, networkx — is lazy and opt-in.

## Where to start

- **[Quick start](quickstart.md)** — a complete analysis in 15 minutes.
- **[Coming from R](migration-from-r.md)** — line-by-line translation guide.
- **[Guides](guide/pooling.md)** — one page per task.
- **[API reference](api/index.md)** — every public name, generated from the
  docstrings.
- **[中文说明](zh/index.md)** — 中文快速上手与约定说明。
