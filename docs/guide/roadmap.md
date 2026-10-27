# Roadmap

What is left, in the order it will be done, and what would have to be true
before each stage counts as finished.

This page is a plan, not a promise. It exists so that the order can be argued
with, and so that "is it done?" has an answer that is not a feeling.

## The rule that does not change

Every stage below ends the same way: code, a test that fails if the code is
absent or wrong, a paragraph in the guide, an entry in the changelog. A feature
without something that can disagree with it is not shipped, and the
[capability matrix](capability-matrix.md) records what that has cost.

Two consequences worth stating up front, because they shape the order:

- A **reference implementation is a development-time oracle**, never a runtime
  dependency. Nothing in `src/` imports, shells out to, or requires R. The hard
  dependency set is `numpy`, `scipy`, `matplotlib` and it will stay that way.
- When a reference implementation is unavailable, **an exact posterior
  computed by quadrature outranks another sampler**. Two samplers agreeing is
  evidence about the samplers; quadrature is evidence about the model. This is
  the same protocol used for the network meta-analysis and the location-scale
  model, and it is why several results on the matrix are pinned without R.

## Where this starts

All sixteen rows of the capability matrix say **yes**, each with a stated scope.
What is genuinely open:

| Item | State |
|---|---|
| RoBMA — Bayesian model averaging over selection models | implemented (36-model space, exact quadrature); the **free-weight** pathway has no third-party check, because the R reference's two pathways disagree about the kernel orientation |
| Non-parametric scale for `tau_trend` (spline or Gaussian process) | **spline implemented**: orthonormal bending basis, one wiggliness hyperparameter estimated by Laplace marginal likelihood, `edf` folded into the Knapp–Hartung df, cross-checked against CmdStan on a bump fixture (`tools/r_reference/tau2t_spline.R`); a Gaussian-process prior is not implemented |
| HTE — formal inference on treatment-effect modification | skeleton only |
| IPD curve band | model-based 0.816–0.884; per-point Satterthwaite df gives 0.944–0.996 except at the edge of the covariate range (0.740 at `x = 65`, where the sandwich itself is wrong) |
| Joint prediction region | 0.767 at `k = 10` under chi-square, 0.827 under Hotelling and **0.903** under Kenward–Roger degrees of freedom; 0.903 is not 0.95 because a prediction region also carries the uncertainty of `Sigma` inside the ellipse |
| Time-varying hazard ratio interval | `contrast_trend()` avoids the `rho = 1` boundary entirely for a linear trend; the joint model reports the spread across structures instead of one number |

The last three are the reason stage 3 is not optional infrastructure. They are
defects that a user would meet, and they are already measured.

## The order

| # | Stage | Size | Why here |
|---|---|---|---|
| 0 | A reference environment | S | Unblocks the last matrix row, and every future oracle |
| 1 | Diagnostics as data | S | Stage 3 aggregates exactly these numbers |
| 2 | A versioned result schema | S | Stages 5 and 7 both emit results; retrofitting a schema breaks saved analyses |
| 3 | The validation harness and the corpus | L | Has three named defects to fix, not just scaffolding |
| 4 | Sensitivity analyses | M | Reuses the engine; needs stage 1 to know when a variant is invalid |
| 5 | RoBMA | L | The last matrix row; specification needs stage 0, verification needs stage 3 |
| 6 | The analysis specification | M | Needs the record system (already built) and the schema |
| 7 | Multiverse | M | Needs 1, 3, 6 — it reports a failure rate, which is a diagnostic aggregate |
| 8 | Performance | M | Baselines start at stage 3, the work happens last |

The obvious order is 1, 2, then the harness, and that is what the brief
proposed. Stage 1 and stage 3 are swapped here for one reason: the simulation
harness has to report convergence failures, boundary estimates, condition
numbers and boundary-hitting rates, and those are *aggregates of diagnostics*.
Built separately, the two definitions will disagree and the disagreement will be
discovered a year later.

---

## Stage 0 — a reference environment

**Status: done.** `RoBMA` 4.0.0 on JAGS 4.3.2, inside an Ubuntu 22.04 WSL2
distribution whose disk is `D:\wsl\Ubuntu\ext4.vhdx` and whose R is 4.6.1 from
the CRAN apt repository. A fifteen-study fit completes and returns posterior
summaries with `ESS` and `R_hat`, which is the proof that the blocker below is
gone.

Two things were learned that the papers do not say, and they matter for stage 5:
RoBMA 4.0.0 requires an explicit `measure`, and it requires `ni` or
`unit_information_sd` before it will build its priors at all. So its
specification is **not** the one written up in the 3.x papers — reading the model
set off the paper would have produced the wrong one, which is exactly the failure
mode this stage exists to prevent.

**The failure is environmental, twice over.** On this machine, `rstan` 2.32.7
fails with `sink(type = "output"): invalid connection`, and a real C++ error
usually precedes it; JAGS has no `win-64` build on conda-forge and the
SourceForge installer host is unreachable. Neither failure says anything about
the statistical models. It says the Windows-native R toolchain in the Anaconda
environment cannot be trusted to produce oracles.

**Also worth stating plainly: JAGS is needed for RoBMA's *numbers*, not for our
implementation.** Writing the sampler does not require JAGS. What JAGS buys is a
second opinion on the *specification* — which models are in the default model
space, which weight functions, which priors — and that is the one thing exact
arithmetic cannot settle, because arithmetic can only check the model set it is
given.

**The Windows-native route is closed for a reason that can be checked rather
than guessed.** conda-forge *does* publish a Windows JAGS, `m2w64-jags` 4.3.0
for `win-64`, so "JAGS is not available for Windows" would be a false statement.
Its dependencies are `m2w64-gcc-libs`, `m2w64-openblas` and
`msys2-conda-epoch`, and its build timestamp is December 2020: that makes it an
**msvcrt** build. R 4.6.1 on Windows is **UCRT**, as every R from 4.2 onwards
is, and a UCRT R cannot safely link an msvcrt static library. `r-rjags` has no
conda-forge build in any flavour. Independently, `C:\rtools45` is installed but
invisible to R — `Sys.which(c("make", "gcc", "g++"))` returns empty strings
inside the Anaconda R — so any Windows compile route would have to repair the
toolchain first as well.

**Which means a second CRAN R does not help either**, and this is where the
brief's step 1 diverges from the conclusion: the problem is not which R is
installed, it is that the only Windows JAGS available is an msvcrt build and R
4.6 is UCRT. A fresh CRAN R 4.6 reproduces the mismatch; the only R that would
match is old enough to be a poor reference in 2026.

**Decision: build the reference environment on WSL2.** The WSL2 kernel is
already installed (`docker-desktop`, WSL version 2), so an Ubuntu image is the
short path. It is not only about RoBMA: a Linux reference box also removes the
"cannot compile a new Stan model" limit that has forced several earlier
validations onto deterministic references, and it brings `MBNMAdose`,
`MBNMAtime`, `RoBMA` and JAGS in one move.

That leaves two attempts:

1. **WSL2 Ubuntu.** `wsl --install -d Ubuntu` is one elevated command and it is
   the user's to run — nothing in this repository will ever trigger a UAC
   prompt. Then `apt install jags`, R from the CRAN apt repository, and
   `install.packages("RoBMA")`. A compiler is always present on Linux.
2. **`m2w64-jags` plus Rtools45**, kept as a bounded experiment rather than a
   plan: ten minutes, no administrator, and a failure is recorded as a measured
   dead end instead of an assumption. The ABI argument above says it is unlikely
   to link.

Done means: `library(RoBMA)` loads and one default `RoBMA()` call returns a
posterior on a fixture we can also fit in Python. If stage 0 fails entirely,
stage 5 proceeds without it and the missing pin is recorded rather than hidden.

## Stage 1 — diagnostics as data

Every fitted result gains `.diagnostics`, a tuple of records with `code`,
`category`, `severity`, `message`, `action` and `detail`. A warning printed to
stderr is not a diagnostic: it cannot be exported, counted, or aggregated.

| Category | Codes in the first batch |
|---|---|
| Identifiability | rank-deficient design; disconnected network; a parameter with no data support |
| Optimisation | not converged; different starts reach different optima; Hessian not positive definite; iteration limit |
| Boundary | `tau2 = 0`; a correlation at ±1; a singular covariance; a scale coefficient at the `max_ratio` cap |
| Data | `vi <= 0` or non-finite; duplicated study or arm; effect size inconsistent with its standard error |
| Inference | a Wald interval where the likelihood is far from quadratic; extrapolation beyond the observed covariate range; a prediction interval whose coverage has not been verified |

Three rules the implementation must keep:

1. **Computed, never asserted.** Every code has a definition and two tests: one
   that constructs the condition and checks the code fires, one that checks it
   does *not* fire on a clean fixture. A diagnostic that only ever appears in
   code comments is decoration.
2. **`error` means the numbers are not reportable as if valid**, and
   `to_dict()` marks them so. `warning` means reportable with a caveat. `info`
   means worth knowing.
3. **Existing `RuntimeWarning`s become diagnostics**, with the warning kept for
   compatibility. Nothing that used to be announced becomes silent.

**Why before the harness:** stage 3's "numerical reliability" block *is* the
aggregate of these codes. Two definitions of "converged" in one package is a bug
that takes a year to notice.

Done means: every result type in the package has `.diagnostics`, the boundary
and identifiability codes fire on purpose-built fixtures, and the docs list each
code with the action it implies.

### Where this actually stands

Partly done, and the honest accounting is more useful than the plan.

| | |
|---|---|
| result classes | **13 of 114** carry `.diagnostics` |
| codes | **12**, each pinned in `tests/test_diagnostic_registry.py` |
| codes from the table above that do **not** exist | rank-deficient design, disconnected network, differing optima, non-PD Hessian, iteration limit, duplicated study, effect size inconsistent with its SE, extrapolation, unverified prediction coverage |

Two of those are not missing, they are **not diagnosable at fit time**: `rma`
raises on a rank-deficient design and `netmeta` raises on a disconnected network,
so no fitted object can ever carry those states. A code that can never fire is
worse than no code — `studies_dropped` was one for months — so they belong in the
documentation of what the constructor refuses, not in the diagnostics module.

The rollout is deliberately curated. `attach_diagnostics` makes the property one
line, and it is applied only where each check is *true* of the family: firing
`few_studies` on a heterogeneity test was the first false positive this produced,
and it is why that check now requires an estimated `tau2`. The remaining classes
are mostly derived views (rankograms, predictor tables, profile likelihoods) where
a blanket property would emit diagnostics that are wrong.

## Stage 2 — a versioned result schema

`to_dict()` and the JSON export gain `schema_version`, `diagnostics`,
`failures`, `settings` and `software` (pimeta, Python, numpy and scipy
versions). One JSON Schema file, one golden-JSON test per result family, so a
renamed field cannot slip through silently.

The interop page gains the table the brief asked for, made explicit: which
fields are **the same statistic as** the corresponding `metafor` / `netmeta`
field, which are merely **the same name**, and which have **no counterpart**.
Interoperability by name is how two packages come to disagree without anyone
noticing.

Done means: a saved result from an older version can be read and its schema
version checked; renaming a field breaks a test rather than a user's script.

### Where this actually stands

The schema half is done: `schema_version`, `software`, `diagnostics` and the
golden-JSON tests are in place. The field map is written as
[Field map: pimeta against metafor and netmeta](../reference/field-map.md), and
`tests/test_field_map.py` verifies every claim in it that a Python repository can
verify — each pimeta field exists on a real fit, each rename is the mechanical
one, each "absent" field is genuinely absent. The R side of the claim rests on
the parity tests and cannot be checked from here, which the page says.

One row of that table is worth reading twice: R's `x$weight` is the *percentage*
while `x$weights` is the raw weight. pimeta calls them `weights_pct` and
`weights`, so a ported `x$weight → x.weight` fails loudly instead of quietly
porting the wrong number.

## Stage 3 — the validation harness and the corpus

Three objects, no framework. The first two exist; the third exists for six
scenarios.

- **`pimeta.validation`** — data-generating processes that return data *and* an
  explicit `truth` dictionary. The name is `validation` and not `simulate`
  because `pimeta.simulate` is already taken: it is the R-compatible
  `simulate.rma.uni` generic, and importing a submodule of that name would
  shadow it in the package namespace. A scenario is one `Scenario` value, not
  one script.
- **`tools/validate.py --scenario NAME --reps N --seed S`** — runs one scenario
  and writes one JSON with six blocks: bias, RMSE, empirical standard deviation
  against the mean estimated standard error, coverage for confidence and
  prediction intervals, test size and power, and reliability (finite-estimate
  rate, `tau2` at the boundary, `I^2`). Every block carries its Monte Carlo
  standard error, so a number can be read instead of interpreted.
- **`tests/validation/baseline/*.json`** — committed numbers, compared on every
  run. The tolerance is derived from the two runs' Monte Carlo errors rather
  than chosen by hand, so a baseline taken at 2000 replications does not flag
  noise as a regression.

**Two tiers, deliberately.** `fast` scenarios run in the default test suite with
a fixed seed and few replications; they are regression tests for the code.
`full` scenarios take minutes and are run as a task; their output is committed
and reviewed, and they are regression tests for the *statistics*.

**What the harness can and cannot see.** A coverage estimate at 400
replications has a Monte Carlo standard error of 0.011 at a nominal 0.95. The
harness can therefore detect gross miscalibration and nothing finer, and each
scenario states the departure it is designed to detect and the replications that
departure needs. A validation framework that quietly reports 0.947 and calls it
a pass is worse than no framework.

**It already found something.** Six baselines at 2000 replications, REML:

| scenario | k | bias | mean SE / empirical SD | CI coverage | PI coverage | PI (`predtype="t"`) | size or power |
|---|---|---|---|---|---|---|---|
| `null` | 20 | −0.001 | 0.985 | 0.9335 | 0.8815 | 0.8990 | size 0.067 |
| `alt` | 20 | −0.001 | 0.985 | 0.9335 | 0.8815 | 0.8990 | power 0.998 |
| `fixed` (`tau2=0`) | 20 | 0.000 | 1.057 | 0.9535 | 0.9685 | 0.9785 | power 0.989 |
| `tiny` | 5 | 0.001 | 1.019 | 0.9230 | 0.8040 | 0.8935 | power 0.388 |
| `unequal` | 20 | −0.001 | 0.964 | 0.9275 | 0.8825 | 0.9030 | power 0.865 |
| `outlier` | 20 | 0.049 | 1.264 | 0.9430 | 0.9770 | 0.9840 | power 0.907 |

The confidence interval at `k = 20` covers 0.9335 ± 0.0056 against a nominal
0.95, and the test rejects 0.0665 ± 0.0056 against a nominal 0.05. Both are the
known behaviour of the normal-reference interval, which charges nothing for the
uncertainty in `tau2`; they are recorded rather than fixed, because changing
the default would be a change of method and not a bug fix.

The prediction interval is the one that does not hold up: 0.8815 at `k = 20`
and 0.8040 at `k = 5`. Switching to the `t` reference recovers about 1.8 and 8.7
points respectively and still leaves a shortfall of 5.1 and 5.7 points. That is
the measured size of the defect the literature attributes to the standard
prediction-interval formula, and it is now the first entry in the debt list
below.

Scenario axes, chosen because they are the ones that break estimators: `tau2` at
zero, small and large; `k` at 5, 10, 40 and 80; balanced against unbalanced
arms; correlated parameters near their boundary; a sampling covariance that is
*estimated* rather than known; a misspecified functional form; and for the
network models a disconnected network against a well-connected one.

**The three debts this stage exists to repay** — each ends either fixed, with the
mechanism named, or documented as a measured limitation with the scenario that
shows it:

- IPD curve band: model-based coverage 0.816–0.884 against a nominal 0.95;
  per-point Satterthwaite degrees of freedom lift it to 0.944–0.996 except at
  the edge of the covariate range, where the CR1 sandwich itself is wrong
  (0.740 at `x = 65`, with 34% of fits on a variance boundary).
- Joint prediction region: 0.767 at `k = 10` under the chi-square reference,
  0.827 under Hotelling, 0.903 under Kenward–Roger degrees of freedom. The
  remaining gap is structural rather than a quantile that has not been found
  yet: a prediction region also carries the uncertainty of `Sigma` inside the
  ellipse, and a denominator degrees of freedom does not calibrate that.
- Time-varying hazard ratio: `contrast_trend()` removes the `rho = 1` boundary
  for a linear trend by construction; the joint model reports the spread across
  structures instead of one number.
- **New:** the standard prediction interval, 0.8815 at `k = 20` and 0.8040 at
  `k = 5` — see the table above.

**The corpus, and a licence constraint that is not negotiable.** Three tiers:
textbook data small enough to check by hand, public datasets for comparison
against mature software, and simulated data with known truth. `pimeta` is MIT,
so **no `metafor` or `netmeta` data file may be committed**. What is committed
is our own derived reference table, its provenance, and the script that produced
it. That is already the practice; this stage turns it into a rule with a test.

Done means: every model family has at least one fast scenario in the default
suite, the three debts are resolved one way or the other, and the corpus has a
manifest naming source, licence, applicable models, reference values and
tolerance.

## Stage 4 — sensitivity analyses

Sensitivity asks one question: *what happens to this estimate if one assumption
changes?* It is not a multiverse and it does not average anything.

`ma.sensitivity(fit, over=...)` returns a table of the same estimand under each
variant, with a one-line reason per variant: leave-one-out and influence, a
different heterogeneity estimator, a different effect-size method, a different
publication-bias assumption, and a grid over an unknown within-study correlation
(`dose_cov`'s `rho` already parameterises this).

**Missing data, scoped honestly.** Every model declares which strategies it
supports, and the declaration is part of the result: complete-case,
available-case, and full-information maximum likelihood where the likelihood
admits it (the multilevel and one-stage IPD likelihoods do). A `dropna` switch
hides the assumption; a declared policy does not.

**Multiple imputation is not in this cycle.** Under MAR, an imputation procedure
is only as credible as its imputation model, and verifying it means a coverage
study under a *known* missingness mechanism — a stage-3-sized project of its own.
If it is built, it is built as its own stage with its own harness, not as a
`method="mi"` argument added to six models at once.

Done means: `ma.sensitivity` covers the five variants above, each variant's
assumptions are stated, and a variant that is statistically invalid for a given
fit is refused with a reason rather than attempted.

**Status.** Three of the five are in `ma.sensitivity_table` (`tau2`,
`leave1out`, `bias`), each with a one-line reason per row and a `refused` list
that is part of the result; see [Sensitivity analyses](sensitivity.md).
`"rho"` is accepted only to be refused for a univariate fit, because the
assumption does not exist there.

The missing-data half is in `ma.MissingPolicy`: every fit declares its strategy
(`complete_case`, `available_case` or `likelihood`), how many rows it used and
which it dropped, and the assumption that strategy needs, with `mechanism`
defaulting to `"unknown"` rather than to the optimistic label. Multiple
imputation is deliberately **not** one of the strategies — see the paragraph
above for why.

Still open: nothing in stage 4's list. The effect-size method is in as the
`measure` family, which takes the alternative fits from the caller because a
result cannot re-derive the 2x2 counts a Peto or Mantel-Haenszel estimate needs;
each alternative must carry a one-line reason saying why the estimand is
unchanged.

## Stage 5 — RoBMA

**The structural point that de-risks this stage.** With independent model
indicators, the posterior over a model set is the normalised marginal
likelihoods. So for the models in the set — no bias, PET, PEESE, and selection
models over several weight functions, each crossed with fixed and random
effects — everything except the sampler's mobility is checkable by quadrature:

- PET and PEESE put a Gaussian prior-like term on the effect; conditional on the
  bias parameter the model is Gaussian, so that parameter integrates out
  analytically or on a one-dimensional grid.
- A selection model's likelihood is a product of truncated-normal density
  ratios. Given `(mu, tau)` it is computable pointwise, smooth, and cheap, so a
  two-dimensional quadrature over `(mu, log tau)` gives its marginal.

That means the exact posterior model probabilities and the exact model-averaged
estimate are both available without JAGS. **The plan:**

1. **Write the model set down before writing code**: the exact list of bias
   models, weight functions and priors, read off the paper rather than guessed,
   recorded as a table with the count. If the count cannot be pinned, that is
   the finding, and it is reported before any sampling starts.
2. `tools/quadrature_robma.py` — exact per-model marginals, exact model
   probabilities, exact model-averaged estimate and its interval.
3. `bayes_robma(yi, sei, *, model_space=..., priors=..., mcmc=...)` — **product-space
   sampling**: the model index is a discrete parameter sampled alongside the
   continuous ones, and the marginal likelihoods are available, so reversible
   jump is not needed. This is both simpler and easier to check than a
   transdimensional sampler.
4. Verification: (a) sampler against quadrature within Monte Carlo error, on
   more than one fixture; (b) the posterior model probabilities sum to one and
   recover the truth in a simulation where the generating model is in the set;
   (c) if stage 0 succeeded, the R `RoBMA` package agrees on the *specification*
   — the model set and the priors — even where its numbers differ by MCMC error.

Ship condition: (a) and (b). (c) is desirable; its absence would be recorded in
the changelog, not glossed.

**The sentence the docs must contain, in these words:** the output is a
posterior probability over models and an average over them, not a test for
publication bias. A posterior probability of 0.6 for "some bias" is not a 60 %
chance that the literature is biased; it is a statement conditional on the model
space, the priors and the data.

Done means: the model set is documented with its priors, quadrature and sampler
agree, and the guide explains what the output does and does not mean.

## Stage 6 — the analysis specification

Four separate objects, as the brief proposed:

| Object | Holds |
|---|---|
| `AnalysisSpec` | the declared decisions, each with its admissible values and a one-line statistical justification |
| run record | what actually ran — the existing `pimeta.record` machinery (`source_fingerprint`, `analysis_fingerprint`, digest, bundle, replay) |
| result | the fit, its diagnostics and its failures |
| report | the rendered summary |

Rules that keep this from becoming a combinatorics toy: **a decision enters the
specification only if more than one choice is statistically defensible**, and
the number of specifications is a consequence, never a target.
`spec.expand()` enumerates the grid and gives each combination an id derived
from its fingerprint, so a rerun lands on the same ids and two runs can be
diffed.

Done means: a specification can be written to a file, expanded, executed and
replayed, and each result names the specification that produced it.

**Status.** `ma.AnalysisSpec` is in, with `Decision` (a value list, a reason per
value, a declared default), fingerprint-derived specification ids, a file format
that records its own size, and `ma.execute` which keeps failures as outcomes
instead of dropping them. A saved record can name the specification that
produced it (`save_record(..., specification=outcome.id)`), in the object, the id
or a mapping, and attaching one leaves the analysis fingerprint untouched. See
[Reproducible analyses](reproducibility.md#declaring-the-decisions-analysisspec).
Still open: the report object that renders a specification's results as one
table. **Done since:** `ma.specification_report(mv)` renders every specification
of a grid — including the ones that failed and the ones excluded by their own
diagnostics — as one Markdown table in grid order, with the caveats attached
below it. It adds no statistics; it renders the numbers that exist and refuses to
show a number for a specification whose fit called it an error.

## Stage 7 — multiverse

The statistics of stability, not a forest plot of everything:

- sign agreement rate, the range and distribution of the estimate, and the
  proportion of intervals excluding zero;
- the specification curve, ordered by estimate, with the decision margins
  underneath;
- **the failure rate, with the failed specifications named** — dropping them is
  the largest silent bias in published multiverses;
- the count of specifications *considered*, *run* and *usable*;
- a variance decomposition across decision axes, labelled exploratory.

Two corrections that the guide will state, because they are routinely got wrong:
the `M` analyses are not `M` independent experiments, so the proportion of
significant specifications is **not** a probability that the effect is real; and
which decisions "matter" is confounded by an unbalanced grid and by correlated
decisions, so that decomposition is descriptive.

**Multiverse, sensitivity and model averaging stay three functions.** They share
the specification, the run record and the report, and they do not share an
interpretation. Averaging over a multiverse would be averaging over analyses
whose weights nobody defined.

Done means: the five quantities above are produced from a specification, failed
specifications are in the output rather than absent from it, and the guide states
what the numbers do not mean.

**Status.** `ma.multiverse` produces all five, with failures counted and named;
see [Multiverse analysis](multiverse.md). The specification curve is ordered by
estimate and each row keeps the choices that produced it, and
`ma.plot_specification_curve` draws it with the decision margins underneath, so a
cluster of similar estimates can be traced to the decision that defines it.
What is still missing is nothing in stage 7's list, but the decomposition remains
one-way: it leaves interactions unattributed by construction, which is why the
figure — not the decomposition — is where interactions are meant to be seen.

## Stage 8 — performance

Measurement first, and stored, because an optimisation without a baseline is an
anecdote.

1. **`tools/bench.py`** — wall time, peak memory, `k`, parameter dimension,
   optimiser iterations, failure rate, and a phase split (preparation,
   objective evaluation, linear algebra, plotting). Baseline committed as JSON.
2. **Algorithm and numerical linear algebra** — the known hot spots are
   factorisations rebuilt inside optimiser loops (the multilevel likelihood
   reassembles the block-diagonal covariance on every evaluation) and gradients
   taken by finite differences where an analytic one exists. `tau_trend` already
   has an analytic gradient; several others do not. This usually beats
   parallelising.
3. **Vectorisation, JIT, parallelism — last, and only where step 1 proves it.**
   `numba` as an *optional* extra, so the hard dependency list does not grow.
   `concurrent.futures` for multiverse, bootstrap and simulation tasks, with
   per-task seeds drawn from a `SeedSequence` so serial and parallel runs are
   bit-identical.
4. **The statistical equivalence gate** — every optimisation keeps the stage-3
   fast baselines green and the R-parity numbers at their recorded tolerance. No
   speed change ships with a changed estimate.

Done means: a baseline exists for every model family, each optimisation is
justified by the profile rather than by intuition, and the equivalence gate runs
in the default test suite.

**Status.** `tools/bench.py` records wall time and peak traced memory for seven
cases covering `rma` (three estimators, two sizes), a meta-regression, a
multivariate fit, RoBMA and `tau_trend`, and compares a fresh run against
`tests/perf/baseline.json`.

It paid for itself immediately, and it also caught a fault in itself. The first
version timed the runs with `tracemalloc` active, which inflates allocation-heavy
code unevenly; it reported `rma(method="DL")` at 22.9 ms where the untraced time
is 2.5 ms. Timing and tracing are now two separate passes.

With a trustworthy number, the probe found two quadratic steps, both of the shape
the stage predicted — a `k x k` object rebuilt where a `p x p` expression would
do:

- `_proj` formed `P = W - W X (X'WX)^-1 X'W` explicitly for `tr(P)`, `tr(P^2)`
  and `y'Py`;
- the Fisher-scoring loop in `pooling` built `np.eye(k) - X (X'X)^-1 X'` and, per
  iteration, `np.diag(w)` followed by two `k x k` products.

Both now go through `pimeta._projection`, which computes the four quantities from
`p x p` matrices in `O(kp)`. `y'Py` is evaluated as `sum(w r^2)` with the weighted
residual, which is also numerically better than expanding the product.

| | before | after |
|---|---|---|
| `rma(method="FE")`, k = 2000 | 134 ms | 1.7 ms |
| `rma(method="DL")`, k = 2000 | 389 ms | 1.7 ms |
| `rma(method="REML")`, k = 2000 | 507 ms | 2.3 ms |
| `rma(method="REML")`, k = 20000 | `MemoryError`, 2.98 GiB | 31 ms |
| `rma(method="REML")`, k = 200000 | — | 288 ms |

The equivalence gate held: `tools/validate.py --all` reproduces every stored
baseline with deltas of 0.0000 at the reported precision, and the full suite is
unchanged. `tests/test_projection_helpers.py` compares each `O(kp)` expression
against the explicit `k x k` one, so the algebra is checked rather than assumed.

Not done, and not justified yet: vectorisation beyond what numpy already gives,
an optional `numba` path, and process-level parallelism for multiverse and
simulation runs. The profile no longer points at them — 200 000 studies in 288 ms
is not the binding constraint for this library.

## Not planned

- **A general workflow engine.** The composable API plus the run record is the
  deliverable; a higher-level engine should follow observed demand rather than
  precede it.
- **More file-format converters.** Schema and traceability first.
- **Automatic selection of knots or fractional-polynomial powers.** Both are
  model-selection procedures with their own multiplicity problems, and neither
  is a one-liner done honestly.
- **Multiple imputation in this cycle** — see stage 4.

## Carried-over debt

- **HTE formal inference** — subgroup and interaction tests exist as a skeleton;
  the formal treatment (a test of modification with the right reference
  distribution under a random-effects model) belongs with stage 4, because it is
  a sensitivity question about a specified interaction.
- **Non-parametric `tau2(t)`** — a spline or Gaussian process on
  `log tau2(t)` needs its own reference, its own small-`k` inference story and a
  shrinkage prior whose behaviour has been measured. It follows stage 5, and it
  is the only place where the parametric-scale advice in
  [Dynamic heterogeneity](dynamic-heterogeneity.md) stops applying.
- **State-space `tau2(t)`** — the same prerequisite as the non-parametric case.
- **Prediction-interval coverage.** 0.8815 ± 0.0072 at `k = 20` and
  0.8040 ± 0.0089 at `k = 5`, measured by the `null` and `tiny` baselines. Two
  alternatives were measured rather than assumed. `predtype="t"` gave 0.8990 and
  0.8935 at the same settings — never worse, and 9 points better at `k = 5`, so
  the one keyword is worth it. A parametric bootstrap over the sampling
  distribution of `tau2` gave 0.910 at `k = 20` and 0.860 at `k = 5`, so it does
  not earn its cost as a default and is kept in `tools/coverage_pi.py` as the
  evidence rather than shipped. What remains is the variance that accounts for
  the uncertainty in `tau2`; closing it is a change of method, not a correction.
  Until then the default is documented as a lower bound on the uncertainty, in
  the `predict` docstring and in [Pooling](pooling.md#prediction-intervals).

## What could reorder this

- If stage 0's WSL environment works, stage 5 gains its specification pin and
  should move ahead of stage 4.
- A user-facing defect in an existing model jumps the queue; the matrix is a
  statement about what works, not a schedule.
- If stage 3 finds that its scenarios cannot separate a defect from Monte Carlo
  noise at feasible replication counts, both stage 3 and the scenario design
  grow, and stages 6 and 7 wait.

## See also

- [Capability matrix](capability-matrix.md) — what is done, with the test that
  protects each row.
- [Reproducible analyses](reproducibility.md) — the record, digest and replay
  machinery stages 6 and 7 build on.
- [Dynamic heterogeneity](dynamic-heterogeneity.md) — the worked example of the
  validation protocol this roadmap generalises.
