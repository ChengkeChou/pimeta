# Changelog

All notable changes to `pimeta` are documented here.
The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/)
and the project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- **An R parity fixture for the network side.**
  `tools/r_reference/netmeta_senn2013.R` dumps every quantity that matters from
  R (netmeta 3.7-0, metafor 5.2-1) on `Senn2013` into
  `tests/data/_r_senn2013.tsv`: the fitted model (k, m, n, Q, tau^2, I^2), both
  contrast matrices, the whole design-by-treatment decomposition
  (`Q.decomp`, `Q.het.design`, `Q.inc.detach`, the element-wise `Q.inc.design`,
  and `Q.inc.random` with its `tau.within`), the P-score ranking, `netmeasures`
  and `netsplit`. `tests/test_r_parity_senn2013.py` recomputes all of it —
  fixed-effect quantities must agree to 1e-12, random-effects ones to 1e-7
  because R's REML stopping rule leaves `tau^2` about 1e-8 off in relative
  terms and that propagates into every standard error.
- **Command-line interface.** `pimeta` (and `python -m pimeta`) with three
  subcommands, so the common analyses need no Python at all:
  - `pimeta fit data.csv --yi TE --vi seTE2 [--mods "~ dose + year"]` —
    pools with `rma`, prints the coefficient table, and optionally writes a
    forest plot and a `*_glance` file with model-level statistics.
  - `pimeta escalc raw.csv --measure RR --ai ai --bi bi --ci ci --di di` —
    arm-level to effect sizes.
  - `pimeta network nma.csv --te TE --se seTE --treat1 t1 --treat2 t2` — the
    treatment-by-treatment estimate matrix.
  Input is CSV/TSV/JSON; output is TSV on stdout (`--format csv|json`, `-o
  file.csv`), so it pipes.  Unknown columns, non-numeric values and unreadable
  files exit with code 2 and a message on stderr.  `docs/guide/cli.md` documents
  every option.
- **Missing values are dropped with a warning.** R discards incomplete cases
  silently through `na.action`; pimeta keeps the dropping but makes it visible:
  `MissingDataWarning` names the number of dropped studies and their rows.
  Wired into `rma` / `rma_ls` (including moderators and scale moderators, via
  the new `warn_missing=True` argument) and `metagen` (which already dropped
  them, silently, before).  `pimeta.missing.complete_cases` / `is_missing` are
  public so downstream code can apply the same rule.
- **Formula-derived coefficient names.** `rma("yi", "vi", mods="~ dose + arm",
  data=df)` now reports `dose` and `arm[B]` instead of `V1`/`V2`; the names are
  carried on `RmaResult.coefnames` and picked up by `tidy()`, `coef()` and the
  printed summary.  A string column whose entries all parse as numbers is
  treated as numeric rather than dummy-coded, which previously made a
  CSV-loaded moderator singular.
- **`py.typed` — inline types for downstream users.** The package now ships the
  PEP 561 marker, so `mypy`/`pyright`/IDE completion use the real annotations
  instead of `Any`. `mypy src/pimeta` went from **167 errors to 0** across 54
  modules with no behavioural change: the fixes were container annotations
  (`dict[str, Any]`, `list[tuple[float, float | None]]`), a typed `_new_ax()`
  helper replacing six duplicated untyped `plt.subplots` branches in
  `plotting.py`, `setattr` for dynamically attached methods, and `None`-safe
  guards in `rma_mv` / `netmeta` / `selmodel`.
- **`pimeta.nma` package skeleton.** The 7,467-line `netmeta.py` is being carved
  into reviewable sections without changing an import path.
  `netmeta::pairwise` (arm-level to comparison-level conversion) now lives in
  `pimeta/nma/pairwise.py`; `pimeta.netmeta.pairwise` and
  `pimeta.nma.pairwise` are the same object, and the package docstring maps the
  remaining fourteen banner-delimited sections.
- **CI and tooling.** `ci.yml` gains a `typecheck` job (`mypy src/pimeta`) and a
  `lint` job (`ruff check` + `ruff format --check` over `src`, `tests`,
  `tools`). A `.pre-commit-config.yaml` runs trailing-whitespace,
  end-of-file-fixer, `check-yaml`, `check-toml`, ruff and mypy.
- **Documentation site (`docs/`).** A full guide set plus a **generated** API
  reference. `tools/gen_api_docs.py` walks `pimeta.__all__` (390 names), groups
  the names by topic and renders one Markdown page per topic from the live
  docstrings — no mkdocstrings, no Sphinx, no build step, so the reference
  renders identically on GitHub, in VS Code and under MkDocs. It also
  regenerates the `R S3 method -> pimeta callable` table and copies
  `CHANGELOG.md` into the site.
  - `docs/index.md`, `docs/quickstart.md` — landing page and a complete
    walkthrough.
  - `docs/migration-from-r.md` — every R idiom (`$`, `generic.class`, `x[i]`,
    `mods=~`, `set.seed`) with its Python equivalent.
  - `docs/guide/` — eleven task pages: effect sizes, pooling, meta-regression,
    heterogeneity, publication bias, network meta-analysis, multilevel models,
    selection and scale models, plots, data frames and formulas,
    interoperability.
  - `docs/reference/` — API conventions, differences from R, R method-name map.
  - `docs/zh/index.md` — Chinese quick start and conventions.
  - `mkdocs.yml` (material theme) and a `docs` extra so the site can be built
    with `pip install pimeta[docs] && mkdocs serve`.
  - `tests/test_docs.py` (12 tests) keeps the docs honest: every public callable
    must have a docstring, the generated pages must cover every name in
    `__all__`, every relative link must resolve, every `ma.<name>` mentioned in
    prose must exist, and the snippets on the landing page and quickstart are
    executed.
- **`pimeta.interop` — ecosystem compatibility as protocols, not adapters.**
  A new subpackage that makes pimeta results consumable by the wider Python
  stack without adding a single runtime dependency (every optional import is
  lazy, so `import pimeta` still needs only NumPy/SciPy/matplotlib).
  - **Tidy tables (broom protocol).** `tidy()`, `glance()` and `augment()` give
    the one-row-per-coefficient / one-row-per-model / one-row-per-study tables
    that `broom` established and that every tidy data flow expects.  They are
    also attached as methods to all 43 result classes (`r.tidy()`,
    `r.glance()`, `r.augment()`, `r.to_table()`), so no import is needed.
    `tidy()` handles coefficient models (`term`/`estimate`/`std.error`/
    `statistic`/`p.value`/`conf.low`/`conf.high`), row-wise results
    (`leave1out`, `cumul`, `influence`) and scale models
    (`component="beta"|"alpha"|"both"` with a `component` column).
  - **Estimator protocol (`MetaRegressor`).** Duck-typed `fit`/`predict`/
    `get_params`/`set_params` with `sample_weight = 1 / vi`, so the model works
    inside `sklearn.pipeline.Pipeline`, `GridSearchCV`, `cross_val_score`,
    `sklearn.base.clone` and `joblib`/`dask`/`ray` backends — verified against
    `is_regressor()`, `clone()`, `Pipeline(StandardScaler, MetaRegressor)` and
    `cross_val_score`.  Adds `predict(interval="confidence"|"prediction")` and
    a weighted `score()`.
  - **statsmodels-shaped results (`as_results`).** `params`, `bse`, `tvalues`,
    `pvalues`, `conf_int()`, `cov_params()`, `llf`, `aic`, `bic`, `nobs`,
    `df_resid`, `df_model`, `fittedvalues`, `resid`, `model.exog_names` and
    `as_pandas()`.  `statsmodels.iolib.summary2.summary_col()` accepts the
    adapter directly (verified against statsmodels 0.14).
  - **Frame backends (`to_frame`).** `dict`, `records`, `pandas`, `polars` and
    `pyarrow`, with scalar broadcasting and explicit errors on ragged columns.
  - **Labelled and graph output.** `to_xarray()` turns the `n x n` NMA
    contrast matrices into a `DataArray`/`Dataset` with `("treat1", "treat2")`
    dims; `to_networkx()` turns the evidence network into a real `Graph` with
    per-edge studies/effects and per-node P-scores.
  - **Protocol introspection.** `supports(obj, "estimator")`,
    `protocol_report(obj)` and `available()` answer "can this object go into
    that tool" without guessing.
- **Location-scale models — the last missing model family.** metafor has no
  `rma.ls()` function: the model is fitted by passing `scale = ~ z` to
  `rma()`.  `pimeta.rma()` now does the same and returns an `RmaLsResult`, so
  `rma(yi, vi, mods=~x, scale=~z)` works like R.  The fit maximises
  τ²_i = g(z_i'α) with τ²_i = v_i + τ²_i in the weights, `link="log"`
  (τ² = exp(z'α), default) or `"identity"`, `method="ML"/"REML"` only (as in
  metafor), and the location coefficients profiled out (or estimated jointly
  with `optbeta=True`).  Standard errors of α come from the numerical Hessian
  of the negative log-likelihood, exactly as `numDeriv::hessian` does.
  Everything metafor reports is there: `b`/`se`/`zval`/`pval`/`ci`/`vb`, the
  per-study `tau2` vector, `alpha`/`se_alpha`/`zval_alpha`/`pval_alpha`/`ci`,
  `va`, `QS`/`QSdf`/`QSp`, `QM`/`QMdf`/`QMp`, `QE`/`QEp`, the per-study
  `I2`/`H2`, `parms = p + q`, `R2 = None`, and the full `fit.stats` matrix.
  Coefficients and standard errors agree with metafor to 1e-7 on the log link
  and the log-likelihood to 1e-6 on both links; α itself agrees to ~1e-5
  because metafor stops on a relative function tolerance on a very flat
  surface (its gradient there is ~1e-4, ours ~1e-9 — we reach the *better*
  optimum, which is pinned by a test).
- **`profile_rma_ls()` (`profile.rma.ls`)** profiles the log-likelihood over
  one scale coefficient, refitting with that coefficient pinned while the
  others stay free, over metafor's default window
  `vc ± qnorm(0.995) * se.alpha`.  On an explicit grid the log-likelihood
  matches R to 4e-14 and the coefficients to 1e-10.  `plot_mv_profile()` and
  `plot(x, which="profile")` draw it.  `confint_rma_ls()` gives the Wald
  intervals for the scale coefficients.
- **`[`, `head()` and `tail()` for `list.rma`-style results.** metafor's
  `[.list.rma`, `head.list.rma` and `tail.list.rma` slice every per-study
  vector of a `leave1out()` / `influence()` / `confint()` result at once so
  the rows stay aligned.  `InfluenceResult`, `LeaveOneOutResult`,
  `CumulResult` and `MetacumResult` now inherit a `RowSlicable` mixin, so the
  R idiom works as real Python slicing: `x[0:3]`, `x[2]` (still a length-1
  result, as in R), `x[[1, 3]]` and `x[mask]`.  `rma_list.py` also provides
  the explicit `slice_rma(x, index)`, `head_rma(x, n=6)`, `tail_rma(x, n=6)`
  and `as_data_frame_rma_list(x)`, exported as `pimeta.head` / `pimeta.tail` /
  `pimeta.slice_rma` for classes that do not opt in.  Scalars and the pooled
  summary are left alone and the input is never mutated.
- **R's `generic.class` method names resolve as aliases.** R spells its
  methods `print.rma.uni`, `cooks.distance.rma.mv`, `[.list.rma`, ... which
  can never be Python identifiers.  `pimeta.r_methods()` returns the
  66-entry `R name -> pimeta callable` map and a module-level `__getattr__`
  makes `getattr(pimeta, "cooks.distance.rma.mv")` resolve to
  `cooks_distance`, so a mechanical R -> Python translation works.  The map is
  filtered at import, so it can never point at a missing name, and unknown
  names still raise `AttributeError` (so `hasattr` keeps working).
- **`metabias()` — every small-study-effect test `meta` offers.** The whole
  function was missing, so `Begg`, `Egger`, `Thompson` and the six binary-data
  variants had no counterpart. All nine are now implemented as transcriptions
  of `meta:::metabias.meta` / `linregcore` / `kentau`:
  - `Begg` — rank correlation of the standardised effects with their variances,
    with meta's tie-corrected `kentau` (and the optional continuity
    correction);
  - `Egger` / `Thompson` — the multiplicative (`lm`) and additive
    (random-effects) regressions;
  - `Schwarzer` — the hypergeometric variant, including meta's **noncentral**
    hypergeometric null moments (`P(x) ∝ C(n1,x)C(n2,m1-x)psi^x`) driven by
    the MH common odds ratio;
  - `Harbord` — the 2x2 score test with separate OR and RR variance formulas;
  - `Peters`, `Deeks`, `Pustejovsky`, `Macaskill` — the four variance /
    sample-size predictors, with `n_e`/`n_c` taken from the fit or supplied.

  Defaults follow R exactly (`Harbord` for a binary odds-ratio analysis,
  `Egger` otherwise), the `k < k_min | k < 3` short circuit returns only
  `k`/`k_min`, the "all studies have the same precision" error is raised, and
  the pre-8.x aliases (`rank`, `linreg`, `mm`, `count`, `score`) are accepted.
  Every statistic, df, p-value, bias, SE, intercept, its SE and `tau` matches
  R 4.6.1 / meta 8.5-0 to every printed digit on a 10-study odds-ratio
  analysis, on the same data restricted to six studies, on a 10-study
  continuous (`metacont`) analysis, on a 10-study single-arm (`metaprop`)
  analysis, and on a generic `(TE, seTE)` vector.
  `metabin()` now also keeps the raw 2x2 counts
  (`event_e`, `n_e`, `event_c`, `n_c`) so the binary tests can run from the
  object, and `metacont()` keeps `n_e`/`n_c`/`method_tau` for the
  Macaskill/Pustejovsky variants, while `metaprop()` keeps `event`/`n` for the
  single-arm `Peters` variance `sqrt(1/event + 1/(n - event))` — mirroring R's
  `x$event.e`, `x$n.e`, `x$event`, `x$n` and `x$method.tau`. `metagen()` /
  `metacum()` results (which name their columns `TE`/`seTE`) are accepted too.
- **`profile_rmamv()` (`profile.rma.mv`).** Profiles the log-likelihood over
  one variance component of an `rma_mv` fit while re-estimating the others —
  exactly what metafor's `.profile.rma.mv` does — and returns
  `MvProfileResult` with the grid, the log-likelihood, the coefficients and
  their CIs at each point, plus metafor's default window
  `[max(0, vc/4), max(0.1, vc*4)]`. `plot_mv_profile` /
  `plot(x, which="profile")` draw it. To support this `rma_mv(sigma2=)` now
  accepts `NaN` entries, which pins the named components while the rest stay
  free (R's `sigma2 = c(NA, ..., v, ..., NA)`). On an explicit grid the
  profile matches R to 3e-15; with an auto-generated grid it agrees to ~1e-8,
  the accuracy limit of the variance-component optimiser.
- **The 14 `rma.mh` / `rma.peto` methods.** `weights()`, `cumul()`,
  `leave1out()`, `rstandard()`, `rstudent()`, `confint()` and `qqnorm()` were
  unavailable on those result objects (every one raised `TypeError`); they now
  work as they do in R. `rma_mh()`/`rma_peto()` keep their 2x2 tables and take
  a `slab=` argument, `weights(mh, pct=True)` returns R's percentages and
  `weights(mh)` the raw `diag(wi)`, `leave1out(mh, cluster=)` collapses studies
  into clusters, `cumul(mh, order=, decreasing=, collapse=)` reproduces
  `cumul.rma.mh`'s ordering (and labels the rows with the reordered `slab`),
  and `confint(mh, level=, transf=True)` back-transforms the ratio measures.
  All pinned against R 4.6.1 / metafor 5.2-1: residuals, standardised and
  deleted residuals, weights, CIs, `leave1out` (estimate/SE/z/p/CI/Q/Qp/I²/H²)
  and `cumul` match to every printed digit.
- **The `rma.mv` diagnostics.** New `mv_diagnostics.py` implements
  `hatvalues.rma.mv` (diagonal or the full hat matrix), `weights.rma.mv`
  (diagonal / matrix / row sums), `rstandard.rma.mv` (with the optional
  per-cluster `X2`), `cooks.distance.rma.mv`, `dfbetas.rma.mv` and
  `rstudent.rma.mv`, and `qqnorm_rma()` now accepts ``rma.mv`` fits. To support
  them `rma_mv()` gained `sigma2=` (fix the variance components like R),
  `slab=` and `cluster=`, and keeps `V`/`mods`/`struct`/`dfs`/`test` on the
  result.
- **`profile_selmodel()`** (`profile.rma.uni.selmodel`) profiles the
  log-likelihood over the selection parameter ``delta`` and returns
  ``SelProfileResult`` with the profile limits; `plot(sm, which="profile")` and
  `plot_selprofile()` draw it. `profile_rma()` routes a ``SelModelResult``
  there.
- **`robust()`** now falls back to the `cluster` stored on the fit and raises
  R's `Argument 'cluster' must be specified.` when there is none.

### Fixed

- **A self-comparison is rejected instead of silently fitted.** `netmeta()` and
  `netmetabin()` accepted a row whose two treatments were the same, which
  produced a meaningless fit and later surfaced as a bare `StopIteration`
  inside `decomp_design()`. Both now raise R's own message —
  `Treatments must be different (arguments 'treat1' and 'treat2').` — naming
  the offending comparison.
- **The diagonal of `netmeta()`'s `TE` matrices was `NaN` where R reports 0.**
  `TE.common`, `TE.random` and the matching `lower`/`upper` matrices are now 0
  on the diagonal, because a treatment against itself has effect 0 and no
  uncertainty; `seTE` was already right, and the `statistic`/`pval` diagonals
  stay `NaN` exactly as in R. Same correction for the random-effects bounds of
  `netmetabin(method="LRP")`. Found by the new Senn2013 parity fixture, which
  compares the full n x n matrices rather than a handful of contrasts.
- **`matreg()` was not metafor's `matreg`.** It pooled the y–x correlations in
  an intercept-only fixed-effect model instead of solving the standardised
  path model, so every number it returned differed from R — and the object
  carried fields (`QE`, `QM`) that `matreg` does not have. It is now a
  transcription of `metafor::matreg`, both branches:
  - `n` supplied (`test="t"`): `b = R[x,x]^-1 R[x,y]`, `sse = 1 - b'R[x,y]`,
    `vb = (sse/df) R[x,x]^-1` for the coefficients but
    `sigma^2 = sdy^2 (n-1)(1-R2)/df` for the fit statistics, `R2`, `R2adj`,
    `F`, `parms`, `deviance` and metafor's `fit.stats` matrix.
  - `V` supplied (`test="z"`): metafor's delta-method Jacobian `A` over the
    unique elements of `R` (column-major lower triangle, `p(p+1)/2` of them
    when `cov=True`), `vb = A V A'` and the Wald `QM`.

  Also new: `cov=`, `means=`, `ztor=`, `nearpd=`, `level=` (95/0.95/0.05, and
  the result stores the tail area like R), `names=` so `y`/`x` can be variable
  names, and the validations R performs (`V` xor `n`, `cov && ztor`, duplicated
  / out-of-range indices, `|R| > 1`, `n <= m`, `V` dimensions).

  On a 4-variable correlation matrix every printed digit now agrees with
  R 4.6.1 / metafor 5.2-1: coefficients, SEs, t/z values, p-values, CIs
  (including `level=90`), `vb`, `R2`/`R2adj`, `F`/`Fp`, `sigma2.ml`,
  `sigma2.reml`, `deviance`, all five `fit.stats` rows and `QM`.
  `fitstats()`, `aic_rma()`, `bic_rma()` and `sigma_matreg()` were rewired to
  read those fields (`AIC`/`BIC` default to the ML column, as in R).
- **`fitstats()` blew up on a saturated model.** With `k == p` the REML `BIC`
  evaluated `log(0)` and raised `ValueError`; R returns `-Inf` there, so
  `_fitstats_core()` now does the same.
- **`as_data_frame_confint()` silently returned an empty list** for any object
  whose point estimate is not called `estimate` (e.g. a `matreg` fit, which
  uses `beta`). It now maps `beta`/`coef`/`tau2` to metafor's `estimate` and
  `ci_lb`/`ci_ub` to `ci.lb`/`ci.ub`.
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
- **`treats()` did not reproduce R's `abbreviate()`.** It claimed to, but
  emitted `aa1`/`aa2`/`bb1` where R gives `a1`/`a2`/`bb`. `stripchars()` from
  `src/main/character.c` and `base::abbreviate()`'s `left.kept` loop are now
  transcribed: spaces first, then lower-case vowels, then lower-case
  consonants, then anything, always from the end of a word and never touching
  the first character. All six reference cases match R exactly.
- **`netcomplex()` was a different function than R's.** It returned the number
  of independent cycles of the comparison graph, but `netmeta::netcomplex()`
  computes the *combination effects* of a component network meta-analysis
  (`C %*% Comp`, `se = sqrt(diag(C %*% Lplus %*% t(C)))`). `netcomplex()` now
  implements the latter and returns a `NetcomplexResult`; the cycle count lives
  on as `netcomplexity()`.
- **`netpath()` crashed with `NameError: name 'scipy' is not defined`** whenever
  the path-incidence matrix was rank deficient, because `scipy.linalg` was never
  imported for the pivoted QR.
- **`netmeta()` raised `ValueError: could not broadcast input array from shape
  (2,) into shape (1,)` for a two-treatment network.** `fX(n)` is degenerate at
  `n == 2`; the single contrast is now `-theta_1` as in R.
- **`nettable()` omitted comparisons without direct evidence.** netmeta lists
  all `n(n-1)/2` rows and leaves the direct columns as `.` (NA here).
- **`subgroup()` silently dropped studies.** It zipped the study labels against
  the comparison rows, so a one-label-per-study vector (the normal case) only
  covered the first few rows and every subgroup fit used the wrong studies. It
  now accepts a study-level or a comparison-level vector and validates the
  length.
- **`hatmatrix()` defaulted to `method="Davies"`**; netmeta 3.7-0 defaults to
  `"Ruecker"`.
- **`pairwise()` iterated scalar `n`/`event`/`mean`/`sd`** (`TypeError: 'int'
  object is not iterable`) and produced a confusing error when no outcome data
  was supplied. Length-1 vectors are recycled like in R and the missing-data
  error names the alternatives.
- **`selmodel()` crashed on a scalar `delta`, or on a scalar
  `control={"delta.min"/"delta.max"}`** (`TypeError: len() of unsized object`
  and `IndexError: too many indices`); both are broadcast now, and a wrong
  length raises a clear error.
- **`selmodel()` stopped short of the maximum for the plateau models.** The
  truncated likelihoods are piecewise constant in the cut-point, so the score
  is exactly zero across whole intervals and BFGS halts on the plateau —
  `type="truncest"` returned a log-likelihood of 9.38 where R already reaches
  13.57 and the true maximum is 13.72. A derivative-free restart now runs for
  the types with flat regions (`trunc`, `truncest`, `stepfun`, `stepcon`,
  `negexppow`); it never makes the fit worse and it beats metafor's own
  `optim()` result on `truncest` (13.7243 vs 13.5676) and `negexppow`
  (29.3369 vs 29.3326).
- **`plot(x, which="baujat")` and `plot(x, which="gosh")` raised
  `KeyError: 1`.** They handed the fitted `RmaResult` straight to
  `plot_baujat()`/`plot_gosh()`, which consume a `BaujatResult`/`GoshResult`;
  the diagnostic is now computed first, matching metafor's
  `plot(baujat(x))` / `plot(gosh(x))` idiom.
- **`gosh(subsets=<n>)` crashed with `AttributeError: 'numpy.int64' object has
  no attribute 'upper'`.** The random-subset loop reused `m` as its loop
  variable, clobbering the effect-measure string that is stored on the result.
- **`rma_glmm()` gave a bare `TypeError` for a mismatched measure.** Passing
  `xi`/`ni` with `measure="OR"` died inside the raw-data block with
  `'NoneType' object is not iterable`; it now explains which arguments the
  chosen measure needs.
- **`rcalc()` dropped the caller's variable names on the formula-style path.**
  Every pair came back labelled `x1`/`x2`/`x3` even when `var1`/`var2` named
  the real variables; the per-study dimension names are threaded through now.
  `rtoz`/`upper` are also validated instead of accepting any truthy value.
- **`conv_wald(..., level=0.95)` returned a variance three orders of magnitude
  too large.** `level` is a *percentage* here, so 0.95 was read as "0.95%".
  Both conventions are accepted now (`95` and `0.95` agree), mirroring
  metafor's `chklevel()`, and an impossible level raises instead of silently
  producing nonsense.
- **`confint_tau2()`, `hc_rma()` and `predict_rma()` took `level` as an alpha
  while everything else takes a confidence level**, so `level=0.95` quietly
  produced a 5%-confidence interval. They accept `0.05`, `0.95` and `95`
  interchangeably now (values `>= 0.5` are read as a confidence level, below
  that as a tail probability), and an impossible level raises.
- **`rma_mv()` accepted an unknown `struct` silently** and died with
  `TypeError: 'NoneType' object is not subscriptable` when `random` named a
  grouping variable that was not supplied. An invalid `struct` raises, and a
  missing/unknown grouping variable names itself.
- **`blsplit()`'s `cluster` and `pairwise()`'s `treat` had no default**, so the
  documented `ValueError("Must specify the 'cluster' variable.")` was
  unreachable and `pairwise(data=...)` could not be called on its own.
- **`_coef_names()` labelled the moderator `V2` instead of `mod1`.** A single
  moderator stored as a `(k, 1)` array appended `n_mods - 1 = 0` names and then
  fell through to the `V2` fallback, and two moderators produced
  `['intrcpt', 'mod1', 'V3']`. Coefficients are now `intrcpt, mod1, mod2, ...`
  exactly like `rma()`'s row names, and `intrcpt` is dropped when there is no
  intercept.

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
  5e-10 (limited by the printed reference). Incomplete networks work too: the
  `type="full"` hat matrix is `n(n-1)/2` square with a zero column for every
  pair that has no direct evidence, so comparisons known only indirectly are
  dropped from the result exactly as R drops them (verified against R on a
  5-edge, 4-treatment network to 5e-14).
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
- **`plot_influence()` and `plot_permutest()`** — the last two `metafor` plots
  that had no equivalent (`plot.infl.rma.uni`, `plot.permutest.rma.uni`).
  `plot_influence` draws the eight panels of `plot.infl.rma.uni` with the same
  reference lines (`±qnorm(0.975)` for the residuals, `±3√(p/(k−p))` for
  DFFITS, `qchisq(0.5, p)` for Cook's distance, `1` for the covariance ratio,
  the full-model τ²/QE for the deletion panels, `p/k` and `3p/k` for the hat
  values and `100/k` for the weights), the covariance ratio on a log scale,
  `plotinf` as a logical or a vector of panel indices, and `plotdfbs` for the
  per-coefficient DFBETAS panels. `plot_permutest` draws the permutation
  distribution of each coefficient's test statistic with the observed value
  marked. `plot(x, which="influence")` and `plot(x, which="permutest")` route
  to them and compute the diagnostic first, like `plot(influence(x))` /
  `plot(permutest(x))`. `permutest()` now keeps `zval`, `zval_perm`, `QM_perm`
  and `coefnames` so the figure can be redrawn without repeating the fit.
- **`dfbetas()` / `rstandard()`** aliases for `dfbetas_rma()` / `rstandard_rma()`,
  matching the short names `metafor` exports next to `hatvalues`/`ranef`.
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
- `tests/test_transforms.py` — 64 tests pinning every `metafor:::transf.*`
  helper (the `*_int` integral transforms, the `*_mode`/`*_pct`/`*_d` families,
  `transf_iprev`, `transf_ilogit`, `transf_*_props`, and the argument checking)
  to R reference values.
- `tests/test_fmt_conv_extras.py` — 21 tests for `fmtx`/`formatx`/`dfround`,
  `conv_delta`/`conv_wald`/`conv_fivenum`, `matreg`, `simulate_rma`, `vcalc`,
  `hettest` and `profile_rma`.
- `tests/test_netmeta_api.py` — 94 tests pinning the rest of the netmeta API:
  `nettable`, `netrank` (P-scores), `netmeasures` (`minpar`, `minpar.study`,
  `proportion`), `netsplit` (direct/indirect and the inconsistency test),
  `decomp_design`, `netdistance`, `netmatrix`, `netconnection`, `hatmatrix`
  (`Davies` full/short/long and `Krahn`), `vikor`, `subgroup`, `netcomplex`,
  `pairwise`, `treats`, `comps`, `combinations`, `createC` and the
  `gemtc2netmeta`/`crossnma2netmeta` converters.
- `tests/test_selmodel_models.py` — 49 tests covering all 18 selection-model
  types, both alternative hypotheses, `prec`, fixed `delta`, `subset`,
  `control` overrides and the `LRT` against the plain model. The fits are
  memoised because each one costs seconds.
- `tests/test_diagnostics_extra.py` — 44 tests for the second- and third-order
  delta method, every `tes()` variant (`alternative`, `test`, `correct`,
  `theta`, `tau2`, `find_lim`), the `vif()` `btt` specifications (indices,
  names, joint sets, `intercept=True`) and `permutest()` with moderators
  (row permutations, `exact=True`, `permci=True`).
- `tests/test_s3_pooling.py` — 24 tests dispatching `plot(x, which=...)` to all
  eleven figures, `predict(newmods=)`, `confint()`, the `confint_tau2()`
  bounds, `hc_rma()` and `predict_rma()`.
- `tests/test_utils_rcalc.py` — 29 tests for every `rcalc()` input form, the
  `fitstats()` REML/ML split, `sigma_matreg()`, and `conv_wald` /
  `conv_delta` / `conv_fivenum`.
- `tests/test_last_public_api.py` — 30 tests for the last public functions that
  had none: `gosh(subsets=)`, `rma_glmm()` argument checking, `peters_test`,
  `lrp_common`, `metaadd` and `points_regplot`.
- `tests/test_utils_diag_netmatrix_mv.py` — 36 tests for the remaining
  `utils`/`diagnostics` branches (`blsplit`, `as_data_frame_*`), the `tes`
  integral path with `tau2 > 0`, `permutest` one-sided alternatives and exact
  sign flips, the six `netmatrix` summarisers, level recoding and tie-breaking,
  and the `rma_mv` variance structures.
- `tests/test_plots_influence_permutest.py` — 20 tests for the two new
  figures and for the coefficient labels.
- `tests/test_matreg_r_parity.py` — 19 tests pinning every printed digit of
  `matreg()`'s two branches against R, plus its validations and the
  `fitstats`/`AIC`/`BIC`/`sigma` methods.
- `tests/test_mh_peto_methods.py` — 36 tests for the `rma.mh` / `rma.peto`
  methods, pinned against R.
- `tests/test_rma_mv_methods.py` — 24 tests for the `rma.mv` diagnostics,
  pinned against R (tolerances reflect the ~5e-8 agreement of the
  variance-component optimiser).
- `tests/test_selmodel_profile.py` — 11 tests for `profile_selmodel()` and its
  figure (small grids, because every grid point is a full `selmodel` fit).
- `tests/test_utils_diagnostics_fill.py` — 36 tests for the `rcalc()` row
  reordering and `dnames`, the coefficient-label forms, `fitstats()`
  column selection and the saturated-model edge case, the unit conversions,
  `as_data_frame_confint()`'s renaming and every `tes`/`permutest` variant.
- **R argument parity**: `netmeta::comps()` gained its `trts` reference
  argument (components are validated against it and returned in that order) and
  `netmeta::pairwise()` gained `data=` (a dict / DataFrame carrying the same
  column names as the parameters; an explicit argument still wins). R's
  `TE`/`seTE` inputs remain unsupported and are documented as such.
- `[tool.mypy]` configuration so contributors get a reproducible
  `mypy src/pimeta` run.

- **Chinese usage manual with executable examples.** `docs/zh/usage.md`
  walks from a first pool through effect sizes, τ² estimators, meta-regression,
  publication bias, multilevel models, network meta-analysis, result tables,
  ecosystem interop, plots and the CLI. Every `python` block is followed by the
  output it really produced: `tools/run_docs_examples.py` runs the blocks (one
  shared namespace, document order) and rewrites the recorded output, and
  `tests/test_docs_examples.py` re-runs them so the manual cannot drift. The
  outputs are generated in one environment and cross-checked in another
  (pandas 2 and pandas 3, numpy 1 and 2).

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

- `ruff check` is now clean and actually verified: 304 findings became 0, and
  the earlier "unverified lint job" caveat is gone. Two rules are ignored with a
  stated reason — `B905` (every `zip()` pairs parallel arrays that are equal by
  construction) and `E741` (`O`, `I` and `l` are the observation counts,
  inertia and R source notation). `ruff format --check` is no longer enforced:
  the house style is uniform but not ruff-format's, and adopting it would
  rewrite 88 of 91 files.

### Not implemented

- Nothing from `meta` / `metafor` / `netmeta` that `pimeta` advertises any
  more. There is no `NotImplementedError` left anywhere in `src/pimeta`.
  `metabin`/`netmetabin` `method="LRP"` used to be the last gap and now works
  without `brglm2`; `netcontrib(method="randomwalk")` used to reject incomplete
  networks and now handles them.
- Still *deliberately* different from R, and documented at the call site:
  - `pairwise()` does not take `TE`/`seTE` inputs. R's `pairwise` is not
    exported by `netmeta` or `metafor` (it is only reachable through the
    attached search path), so its handling of pre-computed effects could not be
    verified against the source.
  - `netmatrix()` keeps `ties_method="first"` as the default instead of R's
    `"random"`, so results are reproducible.
  - `dfround(drop0=True)` returns numbers where R returns strings.

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

- **`MetaRegressor` was unusable with scikit-learn ≥ 1.6.** `is_regressor()`,
  `clone()`, `get_tags()` and every `Pipeline` raised
  `AttributeError: 'MetaRegressor' object has no attribute '__sklearn_tags__'`,
  because scikit-learn replaced the `_estimator_type` attribute with a tags
  protocol. `MetaRegressor` now implements `__sklearn_tags__` (importing
  scikit-learn only when it is asked) and still exposes `_estimator_type` for
  older versions. Verified against 1.9.1.
- **A rank-deficient design matrix now fails with an explanation.** Two collinear
  moderators used to surface as `numpy.linalg.LinAlgError: Singular matrix` from
  inside the τ² estimator; `rma()` now reports the offending columns and the
  usual cause (collinear moderators, or a factor coded with all its levels).
  The test fixture that had `year = 1990 + 5 * (dose - 1)` — exactly collinear
  with `dose` — is fixed too.

- **`netheat()` works now.** It used to raise `ValueError: result does not
  expose an inconsistency contribution matrix` because `netmeta()` never
  attached one. `pimeta.netmeta.netheat_matrix()` now transcribes the
  computation R performs inside `netheat()` (netmeta 3.7-0) — Krahn's `V`, the
  detach residuals and `Q.inc.design`, combined as
  `qtyp[, j] = (V^-1 x_j) * x_j`, `heat = qtyp - Q.inc.design` — and
  `netheat()` draws it with the design names on both axes. `random=` and
  `tau_preset=` follow R's branch structure. Five parity tests in
  `tests/test_netheat.py` pin the matrix cell by cell against R on networks
  whose detach residuals are not all zero (the plain three-treatment loop is
  degenerate and would not have caught a wrong formula). A network without a
  loop now warns and returns `None`, exactly like R.

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