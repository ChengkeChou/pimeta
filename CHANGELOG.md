# Changelog

All notable changes to `pimeta` are documented here.
The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/)
and the project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Fixed

- **`method="HE"` returned the DerSimonian-Laird estimate.** The HE slot in the
  τ² dispatch table was literally an alias for `_dl`, on the assumption that
  Hedges' estimator equals DL for intercept-only models. It does not:
  metafor's HE is `var(y, ddof=1) - mean(v)`. On the 13 BCG trials R gives
  0.32856385799576876 while `pimeta` returned 0.30876026286237612. `_he()` is
  now a real implementation and matches R to 6e-17.
- **`MetaregResult.R2` was a fraction instead of a percentage.** `I2` and `H2`
  on the same object are percentages (matching R), so `R2` was reporting 0.1490
  where metafor reports 14.9046. Now a percentage.
- **`permutest()` produced a degenerate test for intercept-only models.** It
  permuted the outcomes, but with no moderators the design matrix is a column of
  ones, so every permutation left the fit unchanged and every data set came back
  with p = 1. metafor flips the *signs* of the outcomes there (and enumerates
  all 2^k sign patterns when `exact = TRUE`); it permutes the rows of `X` only
  when moderators are present. Both schemes are now reproduced. It also returned
  one omnibus p-value repeated for every coefficient — each coefficient now gets
  its own p-value from its own permutation distribution, the omnibus `QM` gets a
  permutation p-value, and the `comptol = .Machine$double.eps^0.5` slack that
  metafor applies to the comparisons is included.
- **`netmeta()` accepted `method_tau` values it could not compute.** It raised
  `ValueError` for everything but `"DL"` even though `netmeta` documents `"ML"`
  and `"REML"`; both are implemented now (see below).
- **`rma_mv()` stalled ~4e-7 short of the maximum with a single variance
  component.** The profile likelihood is flat enough that L-BFGS-B stopped early;
  a 1-D golden-section polish now runs after it, which also tightened the
  existing `rma.mv` fits.
- **`netmetabin(method="NCH")` and `netmetareg(consistency=FALSE)` were
  advertised in the docstrings but raised `NotImplementedError`.** Both are
  implemented.
- **`netmetabin(method="LRP")` and `metabin(method="LRP")` were not
  available at all** (R needs `brglm2`). Both are implemented now.
- **`escalc(measure="SMD"/"SMDH")` had the wrong Hedges correction and the
  wrong variance.** It used the large-sample shortcut
  `1 - 3/(4*mi - 1)` where metafor uses the exact `cmicalc(mi)`, and it
  multiplied the *whole* variance by `J^2` when metafor only rescales the
  `yi^2/(2*n)` term: metafor's `vtype="LS"` is
  `1/n1 + 1/n2 + yi^2/(2*(n1+n2))`. `SMDH` was also an alias for `SMD` even
  though it standardises by `sqrt((sd1^2+sd2^2)/2)`. On the test data SMD was
  off by 4.8e-6 and SMDH by 7.2e-4; both now match R to 1e-12, and the other
  `vtype` choices (`LS2`, `LS3`, `UB`, `AV`, `H0`) are accepted.
- **`bldiag(..., order=...)` applied the wrong permutation.** The inverse of a
  plain index vector is `order(order)` in R, i.e. a single argsort of the
  0-based indices; the code applied two, so `bldiag(c(1,2,3), order=c(3,1,2))`
  returned `diag(2,3,1)` instead of `diag(3,1,2)`.
- **`cooks_distance_rma()`, `dfbetas_rma()` and `update_rma()` crashed on any
  model with moderators.** All three handed `influence()` (or `rma()`) a design
  matrix that already contained an intercept column, so a second one was
  prepended and every refit was singular (`LinAlgError`). `cooks_distance_rma`
  also passed `tau2=` explicitly, which *fixes* tau^2 in the leave-one-out
  refits where metafor re-estimates it. `cooks_distance_rma` now matches R to
  1e-9 and returns an array instead of a dict keyed on study names.
  `update_rma` also works again without an explicit `mods` argument.

### Added

- **`netmeta(method_tau="ML"/"REML")`.** netmeta does not use its own moments
  estimator here: it collapses the network into per-study stars, builds
  `contrmat` dummies and calls `metafor::rma.mv` with
  `mods = ~ V1 + ... + V(n-1) - 1`, `random = ~ factor(comparison) | studlab`
  and a *fixed* `rho = 0.5`, returning `sqrt(tau2)`. Reproduced, including the
  blocked `calcV` sampling covariance when `n1`/`n2`/`event*`/`mean*`/`sd*`/
  `time*` are supplied. Agrees with R to 2.6e-15 (DL), 6e-8 (ML), 8e-9 (REML).
- **`netcontrib(method="randomwalk")`** — Davies' random-walk contribution
  matrix, transcribed from `netmeta:::contribution.matrix.davies`. Matches R to
  5e-10 (limited by the printed reference).
- **`netmetabin(method="NCH")`** — the conditional (non-central hypergeometric)
  likelihood, maximised over the identifiable treatment contrasts with the
  covariance taken from the analytic observed information rather than `optim`'s
  finite differences.
- **`netmetareg(consistency=False)`** — the inconsistency model: one covariate
  interaction per comparison (`assumption="independent"`) or a single common one
  (`"common"`), with `direction1`/`direction2`, netmeta's `mkIval()` fallback,
  `max.ia`, and `check_Ival`-style validation of the direction assignment.
- **`metabin(method="LRP")` and `netmetabin(method="LRP")`** — Jeffreys-penalised
  (Firth) logistic regression, which is what R delegates to
  `brglm2::brglmFit(type="MPL_Jeffreys")`. The new `_firth.py` maximises
  `l(b) + 0.5 log|X' diag(n p (1-p)) X|` with Newton's method using the analytic
  score and Hessian, and returns `I^-1` at the estimate as the covariance (R's
  `vcov(res.lrp)`). `meta:::phi` — the c-hat dispersion factor that inflates the
  LRP standard errors, `(sum of squared Pearson residuals / df) / (1 + s.bar)`
  floored at 1 — is reproduced too.
  `netmetabin(method="LRP")` fits one logistic model over all arms with fixed
  study effects and expands the treatment coefficients with the same `H` matrix
  the MH and NCH branches use; it is the only `netmetabin` method with a
  random-effects counterpart (`TE_random` = common estimate, `seTE_random` =
  common SE x `phi`). `netmetabin` also reports `phi`.

  One honest difference: `brglm2` stops at `epsilon = 1e-8` on the penalised
  deviance and lands ~4e-7 from the optimum (`max|score|` 8e-6 on the 4-study
  test set). Our Newton iteration converges to `max|score| < 1e-12`, and with
  `brglm_control(epsilon = 1e-12)` R reproduces our numbers to 1e-13, which is
  asserted in the tests. Against R's *default* settings the agreement is 4e-7.
- **`RmaResult.confint()` now works with moderators**, and
  **`RmaResult.predict(newmods=...)`** is implemented. `_qe_func()` and
  `confint_tau2()` take an optional design matrix, so the Q-profile uses the
  general `P = W - W X (X'WX)^-1 X'W` and `vt = (k - p)/tr(P)`.
  `type="HT"` now raises the same error metafor does for moderator models
  instead of silently returning a wrong interval.
- **`permutest(permci=True)`** is computed instead of merely accepted: each
  bound is a root search on the permutation p-value of the shifted null model
  `yi - val * X[, j]`. The permutation set is drawn once and reused for the whole
  search (metafor redraws at every step, which leaves `uniroot` chasing a noisy
  step function).
- **`netcontrib` / `netmetabin` / `netmetareg` gained 25 R-pinned tests**, plus
  12 for `confint`/`predict`, 9 for `permutest`, 14 for `LRP`, 33 for the
  utility helpers / `escalc` and 18 for the `rma` accessors.
- `tests/test_utils_escalc.py` and `tests/test_rma_object_helpers.py` pin
  `bldiag`, `blsplit`, `vec2mat`, `replmiss`, `cmicalc`, `contrmat`, `pairmat`,
  `ci`, `units2inches`/`inches2units`, `jama_labels`, `getmfopt`/`setmfopt`,
  `escalc` across every supported measure, and the `rma` accessors
  (`coef`, `vcov`, `fitted`, `residuals`, `weights`, `logLik`, `deviance`,
  `AIC`, `BIC`, `nobs`, `formula`, `model_matrix`, `hatvalues`, `cooks.distance`,
  `dfbetas`, `rstandard`, `blup`, `summary`).
- `.github/workflows/ci.yml` — every push and pull request now runs the full
  suite on a 4-way matrix (Python 3.10/3.11 with `numpy<2`, Python 3.12/3.13 with
  `numpy` 2.x), reports coverage, and separately builds the sdist/wheel, runs
  `twine check` and executes `examples/bcg_meta_analysis.py` against the built
  wheel. Previously the only workflow fired on a GitHub Release, which was far
  too late to catch a regression.
- `tests/test_coverage_gaps.py` — 30 tests pinning the modules that had the
  thinnest coverage against R 4.6.1 / metafor 5.2-1 reference values generated by
  `_refgen_gaps.R`: all τ² estimators, `fsn` (Rosenthal/Rosenberg/Orwin),
  `regtest` with the `sei`/`vi`/`ni` predictors, `begg_test`, `trimfill`
  (L0/R0/Q0), `leave1out`, `influence` and its components (`cooks_distance`,
  `hatvalues`, `rstudent`), `baujat`, `cumul`, and `metareg`.
- `[tool.mypy]` configuration so contributors get a reproducible
  `mypy src/pimeta` run.

### Changed

- `rma_mv()` gained `intercept=False` (R's `mods = ~ x1 + x2 - 1`) and `rho=`
  (a *fixed* correlation, which is how `rma.mv` treats a supplied `rho`), both
  needed by the network `ML`/`REML` estimator. Both default to the previous
  behaviour.
- `confint_tau2()` and `_qe_func()` in `pooling.py` accept an optional design
  matrix. Passing `X=None` keeps the intercept-only formulation exactly as
  before.
- Coverage of the six previously thinnest modules: `fsn` 24 %→72 %,
  `publication_bias` 28 %→76 %, `sensitivity` 43 %→92 %, `cumul` 44 %→96 %,
  `metareg` 44 %→79 %, `tau2` 46 %→89 %. A second pass took `escalc` 59 %→86 %,
  `s3` 61 %→79 % and `utils` 58 %→75 %, for 78 %→80 % overall on 13740
  statements.
- Corrected several annotations that were objectively wrong rather than merely
  imprecise: `_prepare()` was declared to return a 2-tuple while returning three
  values; the τ² helpers in `tau2.py` and the array-taking helpers in
  `pooling.py` declared `list[float]` while being handed `np.ndarray`; and
  `RmaMvResult` advertised `np.ndarray` for fields it defaulted to `None`.

### Not implemented

- Nothing from `meta` / `metafor` / `netmeta` that `pimeta` advertises any
  more. `metabin`/`netmetabin` `method="LRP"` used to be the last gap and now
  works without `brglm2`.

### Fixed

- **`method="HS"`, `method="PM"` and `method="EB"` returned values that did
  not match metafor.** All three are now transcribed from `rma.uni`'s source and
  reproduce R to machine precision:

  | method | before | R 4.6.1 / metafor 5.2-1 | now |
  |---|---|---|---|
  | `HS` | 0.19030177854977914 | 0.22836286367690678 | 2.8e-17 |
  | `PM` | 0.31806845220502394 | 0.31809371186301511 | exact |
  | `EB` | 0.31806845220502394 | 0.31806851577130513 | exact |

  The HS bug was a wrong formula (`(Q-(k-1))/trP` instead of metafor's
  `(RSS-k)/sum(wi)`). The PM bug was a tolerance mismatch: metafor hands the
  problem to R's `uniroot` with `con$tol = .Machine$double.eps^0.25` (1.22e-4),
  so it stops ~2.5e-5 short of the true root — a tight Brent solve is *more*
  accurate but disagrees with R. EB needed metafor's actual Fisher-scoring
  iteration starting from the HE value, whose `adj` step is now reproduced
  exactly.
- **`method="HE"` returned the DerSimonian-Laird estimate.** The HE slot in the
  τ² dispatch table was literally an alias for `_dl`, on the assumption that
  Hedges' estimator equals DL for intercept-only models. It does not:
  metafor's HE is `var(y, ddof=1) - mean(v)`. On the 13 BCG trials R gives
  0.32856385799576876 while `pimeta` returned 0.30876026286237612. `_he()` is
  now a real implementation and matches R to 6e-17.
- **`MetaregResult.R2` was a fraction instead of a percentage.** `I2` and `H2`
  on the same object are percentages (matching R), so `R2` was reporting 0.1490
  where metafor reports 14.9046. Now a percentage.

All six moment estimators (`DL`, `HE`, `HS`, `SJ`, `PM`, `EB`) now agree with
`rma(yi, vi, method=…)$tau2` to at worst 5.6e-17.

### Removed

- The `py.typed` marker and the `Typing :: Typed` classifier. The package ships
  annotations, but `mypy src/pimeta` still reports ~55 structural
  inconsistencies in the numeric kernel; advertising PEP 561 support would
  promise users machine-checked types we cannot yet guarantee. The marker comes
  back once that count reaches zero.

## [1.0.1] — 2026-10-18

Documentation release. No library code changed; PyPI description is rebuilt from
`README.md`, so a version bump was needed to publish the corrected docs.

### Added

- `examples/bcg_meta_analysis.py` — a runnable end-to-end walkthrough on the
  13 BCG vaccine trials. It writes the study table to CSV, recomputes effect
  sizes with `escalc()`, pools with `rma(REML)`, draws a forest plot, runs
  Egger's test, trim-and-fill and Rosenthal's fail-safe N, fits a
  meta-regression on publication year, and closes with a small network
  meta-analysis — asserting every number against R's `metafor::dat.bcg` output.
- README badge row (PyPI, Python versions, licence, CI) and an "API conventions
  worth knowing" table covering: `I2`/`H2`/`R2` are percentages, `confint()`
  returns a dict, `result.b` is always an array, `escalc()` returns log-scale
  `yi` for ratio measures, `netrank(...).ranking_random` holds P-scores, plots
  return a matplotlib `Axes`.
- README: Requirements matrix stating that CI runs `numpy` 2.5 / `scipy` 1.18 /
  `matplotlib` 3.11 and that the same example passes on `numpy` 1.26 / `scipy`
  1.16.

### Fixed

- README listed five symbols that do not exist in the package: `ranktest`
  (the function is `begg_test`), `cooks.distance` (`cooks_distance`), bare
  `dfbetas` and `rstandard` (`dfbetas_rma`, `rstandard_rma`), and `transf_rtor`
  (removed — `metafor` has no `transf.rtor`). Every backticked API name in the
  README is now checked against `pimeta.__all__`.
- README claimed `netrank()` offers SUCRA and frequency rankings; it implements
  P-scores only, which is what netmeta 3.7-0's `netrank()` returns.
- README `escalc()` list repeated `RD` and omitted nothing else; corrected.
- Quick-start pooling example printed the wrong three numbers for its own input
  data.
- Documented that `metafor.news()` is the sole unimplemented `metafor` export
  (278/279).

## [1.0.0] — 2026-10-18

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

[1.0.0]: https://github.com/ChengkeChou/pimeta/releases/tag/v1.0.0
[1.0.1]: https://github.com/ChengkeChou/pimeta/releases/tag/v1.0.1