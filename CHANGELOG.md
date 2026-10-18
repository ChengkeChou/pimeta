# Changelog

All notable changes to `pimeta` are documented here.
The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/)
and the project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [1.0.0] — unreleased

First public release. Complete pure-Python reproduction of R's `metafor`
5.2-1, `netmeta` 3.7-0 and `meta` 8.5-0.

### Added

**Effect sizes and pooling**
- `escalc()` covering all `metafor` measures (`RR`, `OR`, `RD`, `AS`, `PETO`,
  `IRR`, `IRS`, `IRSD`, `MN`, `SMN`, `MRAW`, `SMCC`, `MD`, `SMD`, `SMDH`,
  `ROM`, `RPB`, `RBIS`, `D2OR`, `COR`, `UCOR`, `ZCOR`, `ZPCOR`, `PR`, `PLN`,
  `PLO`, `PAS`, `PFT`, `PRAW`, `PD`) with `add` / `to` / `drop00` handling.
- τ² estimators `DL`, `HE`, `HS`, `SJ`, `ML`, `REML`, `PM`, `EB`, `GENQ`,
  `GENQM`, `QGENQ`, `QGENQM`.
- `rma()`, `rma_uni()`, `rma_mv()`, `rma_glmm()`, `rma_peto()`, `rma_mh()`.
- Hartung–Knapp, Knapp–Hartung, prediction intervals, profile-likelihood τ² CI.

**R driver parity**
- `metagen`, `metabin`, `metacont`, `metacor`, `metainc`, `metaprop`,
  `metarate`, `metamean`, `metacum`, `metainf`, `metareg`, `metabind`,
  `metamerge`, `metaadd`, `metacr`.

**Network meta-analysis**
- `netmeta`, `netmetabin`, `netcomb`, `discomb`, `netcomplex`, `netmetareg`,
  `netmeasures`, `netrank`, `netimpact`, `netpath`, `netsplit`, `netmatrix`,
  `netcomparison`, `netbind`, `netcontrib`, `netdistance`, `nettable`,
  `decomp_design`, `invmat`, `hatmatrix`, `comps`, `combinations`, `treats`,
  `vikor`, `subgroup`, `createC`, `crossnma2netmeta`, `gemtc2netmeta`.

**Plots** (matplotlib, headless-safe `Agg` backend)
- `forest`, `funnel`, `netgraph`, `plot_baujat`, `plot_gosh`, `radial`,
  `labbe`, `llplot`, `bubble`, `drapery`, `qqnorm_rma`, `rankogram`,
  `netheat`, `hasse`, `netposet`, `traffic_light`, `heatplot`, `regplot`.

**Inference, diagnostics and sensitivity**
- `regtest`, `begg_test`, `trimfill`, `fsn`, `gosh`, `permutest`, `tes`,
  `vif`, `deltamethod`, `leave1out`, `influence`, `baujat`, `cooks_distance`,
  `hatvalues`, `rstudent`, `simulate_rma`, `vcalc`, `hettest`, `matreg`,
  `profile_rma`.

**Selection models**
- All 18 `rma.uni.selmodel` types, including `stepcon` (SLSQP).

**Reporting and layout (`meta` parity)**
- `cidprop`, `cilayout`, `format_ci`, `forest_dims`, `longarm`, `addpoly`,
  `reporter`, `rob`, `aggregate_escalc`, `cbind_escalc`, `rbind_escalc`,
  `estimates`, `emmprep`, `points_regplot`.

**S3-style generics**
- `coef`, `vcov`, `fitted`, `residuals`, `logLik`, `deviance`, `nobs`,
  `formula`, `AIC`, `BIC`, `weights`, `sigma`, `df_residual`, `model_matrix`,
  `simulate`, `confint`, `predict`, `hatvalues_rma`, `dfbetas_rma`,
  `rstandard_rma`, `ranef_rma`, `summary`, `summary_text`, `print_summary`,
  `plot`.

**Utilities (`metafor` internals exposed)**
- `bldiag`, `blsplit`, `vec2mat`, `replmiss`, `cmicalc`, `contrmat`, `pairmat`,
  `rcalc`, `fitstats`, `blup`, `sigma_matreg`, `update_rma`, `se`, `ci`,
  `getmfopt`, `setmfopt`, `jama_labels`, `inches2units`, `units2inches`,
  `as_data_frame_*` family.

**Readers**
- `read_rm5`, `read_cdir`, `read_mtv` for Cochrane RevMan and `.cdir` / `.mtv`
  exports.

**Transforms**
- All 58 `metafor::transf.*` functions plus `_int` / `_mode` variants.

### Notes
- Result objects are dataclasses whose field names mirror the R object's `$`
  fields, so R code translates almost one-to-one.
- R quirks (non-convergence → `NA`, `contrmat` sign conventions, `NA` subsets)
  are reproduced deliberately.
- No R, rpy2, or third-party meta-analysis package is used or required.

[1.0.0]: https://github.com/ChengkeChou/pmeta/releases/tag/v1.0.0
