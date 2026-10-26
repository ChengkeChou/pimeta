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
| RoBMA — Bayesian model averaging over selection models | not implemented; blocked on a JAGS reference |
| Non-parametric scale for `tau_trend` (spline or Gaussian process) | not implemented; no reference, no measured shrinkage prior |
| HTE — formal inference on treatment-effect modification | skeleton only |
| IPD curve band | coverage measured at 0.74–0.87 against a nominal 0.95 |
| Joint prediction region | coverage 0.767 at `k = 10` |
| Time-varying hazard ratio interval | estimate sits on the `rho = 1` boundary; the interval does not account for it |

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

- IPD curve band: coverage 0.74–0.87 against a nominal 0.95.
- Joint prediction region: 0.767 at `k = 10`.
- Time-varying hazard ratio: the estimate sits on `rho = 1`; the interval does
  not.
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
  0.8040 ± 0.0140 at `k = 5`, measured by the `null` and `tiny` baselines. The
  `t` reference closes part of the gap; closing the rest needs a variance that
  accounts for the uncertainty in `tau2`, which is a change of method rather
  than a correction. Until then the default prediction interval is a lower
  bound on the uncertainty and this page says so.

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
