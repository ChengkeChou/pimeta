# Changelog

All notable changes to `pimeta` are documented here.
The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/)
and the project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Fixed

- **Copas validation against `metasens`, separated into its two halves.** The
  selection model's pipeline was audited stage by stage by feeding pimeta's
  sensitivity surface to metasens' own downstream code (`contourLines`, the
  per-contour least squares, the `nobs`-weighted slope, the orthogonal line):
  it returns `slope = 0.14937955364276298`, the value pimeta computes to the
  last digit, with identical per-contour piece sizes. So every stage after the
  surface is the reference's algorithm exactly, and the whole residual
  disagreement is the surface — 1.4e-5 at the worst grid point, from two
  optimisers stopping in different places, giving 2e-8 on the adjusted effect
  and 4e-5 relative on its standard error. The same audit found that
  `metasens`' contour tracer emits duplicated polylines (one level three
  times) and 2–5 point boundary fragments with adjusted R² down to −0.33 on
  the synthetic fixture, and those are fitted as contour lines and weighted by
  `nobs`; that fixture therefore cannot be a parity target, and
  `docs/reference/differences.md` now says so with the measured numbers.
- **One confidence-level convention, everywhere.** `pimeta` had three
  spellings of the same argument: `0.95` (a confidence level), `95` (percent)
  and `0.05` (the alpha). The defaults were internally consistent, so nothing
  broke by default — but crossing conventions did: `rma_peto(level=0.95)`
  returned an interval 31 times too narrow, and `rma_peto(level=95)` returned
  `nan`, both silently. Every entry point now normalises through
  `ma.level_to_alpha`, which accepts all three and is idempotent on the alpha
  so it can be applied at more than one layer. `tests/test_level_consistency.py`
  pins that all three spellings give the same interval, for `rma_peto`,
  `rma_mh`, `mh_confint`, `confint_tau2`, `predict_rma` and `hc_rma`, and that
  the interval each reports really is the 95% one.

### Added

- **Markov chain Monte Carlo, as an optional engine that adds no dependency.**
  `pimeta.bayes` is a complete sampler written against numpy: adaptive
  random-walk Metropolis with a multivariate normal proposal built by the
  Haario rule (`2.38**2 / d` times the covariance of the states visited, with a
  bounded correction from a 64-step acceptance window), optional per-block
  updates for hierarchical models, and the diagnostics that say whether the
  draws can be believed — rank-normalised split `R-hat` and bulk and tail
  effective sample size, following Vehtari et al. (2021) rather than the older
  versions of those numbers, which are quietly too optimistic on exactly the
  models this is for.

  `pip install pimeta[bayes]` installs exactly what `pip install pimeta`
  installs. The extra is declared so that a deliberately optional feature is
  *stated* rather than implied — and so an accelerator has somewhere to go
  later without changing anyone's install command. What the feature costs is
  time, not dependencies, which is why it is opt-in per call.

  The engine is checked three ways, because a sampler has no reference
  implementation to be compared with. Against **exact posteriors**: a Gaussian
  with a known covariance, and a beta-binomial whose posterior is a Beta, both
  reproduced within Monte Carlo error including the interval. Against its own
  **failure modes**, in both directions: chains started twelve standard
  deviations apart must raise, and on a heavy-tailed target the statistic must
  be invariant to a monotone transformation of the draws — which is exactly what
  the raw variance-ratio version is not. Against **refusals**: one chain cannot
  report convergence, four draws cannot be split, a log density that returns a
  list is a bug in the caller, and a starting point outside the support is
  reported rather than sampled from.

  Two things the implementation had to get right that are easy to get wrong.
  The acceptance rate that drives the tuning is a **window**, not the running
  average: a cumulative rate lags so far behind a proposal that has become too
  wide that the scale runs away, and the chain then rejects everything for the
  remainder of the run — measured, before the fix, as a final acceptance of
  0.027 on a target it had earlier sampled correctly. And adaptation has to be
  per **block**: on a 14-parameter hierarchical model a single global proposal
  reached `R-hat = 1.23` where per-group blocks reached `1.014`, on the same
  800 draws and the same seed.

- **Time-course network meta-analysis.** `pimeta.nma.timecourse` fits
  `y = g_b(t) - g_a(t) + e` across a network, with the same parameter vector
  used wherever a treatment appears — so consistency cancels at **every** time,
  including times no study measured, as an identity rather than as a property a
  lucky fit might have.

  Four bases over time (`constant`, `linear`, `quadratic`, `rcs`), four
  heterogeneity structures (fixed, a random study level, a random trajectory,
  both), and an explicit reference treatment: every contrast is identified only
  up to a common shift of all trajectories, so one of them is fixed at zero and
  left out of `curves` rather than estimated as a flat direction.

  Two things are specific to time. **The baseline**: with a constant term the
  difference at `t = 0` is the ordinary NMA effect, so `basis="constant"` on one
  row per study *is* standard network meta-analysis — pinned against `netmeta`,
  with effects, standard errors and `tau2` all agreeing to 1e-8. **The
  correlation**: a study reporting the effect at 3, 6 and 12 months measured the
  same participants three times, so `rho` (constant) or `phi` (AR(1) decay
  `phi**|t_i - t_j|`) is required rather than defaulted.

  Validated against `metafor::rma.mv` with exactly that AR(1) covariance for all
  three linear bases: coefficients to 1e-7, standard errors to 1e-6 relative,
  `tau2` to 2e-7.

  One refusal is a genuine modelling statement rather than input validation: a
  study that contributes contrasts for **more than one pair of treatments**
  cannot be described by a single `rho` or `phi`. Two arms measured at the same
  time share the control, so their contrasts are correlated through it in a way
  that depends on the arm sizes; the module says so and points at `vcov`.

  Building it changed one test. The restricted cubic spline nests the straight
  line, so its likelihood "should" be at least as good — and under REML it is
  not, because restricted likelihoods of models with different fixed effects are
  not comparable. The comparison now uses ML, which is the only place it is
  meaningful.

- **Dose-response network meta-analysis.** `pimeta.nma.dose` fits a dose
  function per treatment to a network of contrasts,
  `y_i = f_b(x_b; β_b) - f_a(x_a; β_a) + e_i`, with six dose functions
  (`linear`, `quadratic`, `loglinear`, `rcs`, and the two saturating forms
  `emax` and `exponential`), a choice of three heterogeneity structures, and a
  league table at any pair of doses through `contrast()`.

  The parameterisation is what makes the consistency equations hold: the same
  `β_t` is used wherever treatment *t* appears, in either position, so summing
  `d_ab(x) + d_bc(x) - d_ac(x)` around a circuit cancels **as an identity, at
  any dose** — including doses nobody studied. A model that stored one
  parameter per comparison would satisfy the equations only at the observed
  doses and only if the fit happened to. The test asserts it at doses that
  appear nowhere in the data, and asserts that the same linear functional has
  the same Jacobian whichever way it is assembled, so no route through the
  network can produce a different variance for the same number.

  Every curve is anchored at a zero dose, and that is an assumption rather than
  a convenience: without an anchor, adding a constant to every curve leaves
  every contrast unchanged, so the model is not identified. `anchor=` changes
  the anchor dose; a treatment seen only at the anchor gets no curve at all
  rather than parameters the data cannot support.

  Validated at the level of the model, on an 16-study / 23-contrast fixture
  simulated from known Emax curves:

  - every dose function that is linear in its parameters against
    `metafor::rma.mv(yi, V, mods = ~0 + basis, random = ~1 | study)`, with
    coefficients to 1e-7, standard errors to 1e-6 relative and `tau2` to 5e-8;
  - `emax` and `exponential` against `nls(..., weights = 1/vi)` with the PORT
    algorithm: parameters to 1e-4, standard errors to 1e-4 relative, and the
    residual sum of squares to 1e-8, the last recomputed through the public
    `contrast()` API rather than read from the fit;
  - the restricted cubic spline basis itself against `rms::rcs`, to 1e-9.

  Two things the validation settled that are worth recording. `rma.uni` is
  **not** a reference for this model: it treats a study's several contrasts as
  independent, so on the same fixture it reports `tau2 = 0.2216` where the
  model with a study-level random effect gives `0.3013`; `rma.mv` gives
  `0.3013134977` against pimeta's `0.3013134873`. And `nls` estimates a
  dispersion from the residuals, which is right when the variances are
  unknown and wrong here; the comparison is against its covariance divided by
  `summary()$sigma`, which isolates the known-variance answer.

  Refusals rather than quiet approximations: a random effect on the curve with
  a dose function that is not linear in its parameters (`emax`, `exponential`
  — the marginal likelihood would be an integral with no closed form), a curve
  with fewer distinct doses than parameters, a covariance that is not positive
  definite, a between-study variance with one study, and an unknown dose
  function or structure. A boundary `tau2` warns and quotes its size against
  the mean sampling variance, because a zero heterogeneity estimate is a
  statement about the data and not a detail of the fit.

- **Individual-participant-data network meta-analysis.** `pimeta.nma.ipd`
  fits the one-stage IPD network model: a study intercept per study, treatment
  effects against a reference, optional study-centred covariates as main
  effects, optional treatment-by-covariate interactions, and a random
  treatment effect per study with a full between-study covariance (`struct`).
  The design is built from the parameterisation layer, so the same contrast
  matrix the algebra tests check is the one the likelihood uses —
  `contrast(a, b)` is a row of `B`, and the league table is `B` against the
  reference vector.

  Covariates are centred **within study**, which is what keeps the treatment
  parameters on their within-study scale and lets the study intercepts absorb
  between-study differences in the covariate distribution. Interaction terms
  are formed from the centred covariate, so `delta` is the change in the
  treatment effect per unit *within-study* deviation, not a between-study
  comparison that confounding could produce.

  Validated against `lme4::lmer(y ~ 0 + study + A + x + (0 + A | study))` on a
  3960-participant, 8-study, three-treatment fixture: treatment effects and
  the covariate coefficient to 1e-8, standard errors to 7e-6, `sigma2` to
  9e-8, the between-study covariance to 1.4e-5 on its flattest direction, the
  log-likelihood to 1e-10. Two degenerations are pinned as well — a two-arm
  network reproduces `rma` (effect to 1e-8, `tau2` to 3e-8) and a one-study
  network reproduces weighted least squares exactly — because those relate the
  new code to implementations that are already trusted.

  `multinma` 0.9.1 was also fitted to the same fixture and gives posterior
  means `d[B] = 0.07`, `d[C] = 0.16`, which is corroboration on the mean
  structure and nothing more: it is **not** the same model, so its posterior
  SDs (0.17, 0.18) are not comparable to these standard errors (0.112,
  0.140). `multinma`'s IPD normal model puts one shared heterogeneity SD on
  every relative effect — it estimated `tau = 0.44` where this fixture
  supports treatment-specific variances of 0.095 and 0.115 — and it fits a
  residual SD per study-arm (22 of them here, against one `sigma2`). Neither
  choice is wrong, but a Bayesian fit of a differently parameterised model
  cannot be a tolerance target, so the pinned numbers come from `lme4`, which
  fits the same likelihood deterministically.

- **A network parameterisation layer, with algebraic tests.**
  `pimeta.nma.design` holds the part of a network meta-analysis that every
  network model shares and none of them should re-invent: the comparison
  graph, the signed contrast matrix, the basic-parameter design matrix, the
  fundamental circuits, the unrelated-mean-effects parameterisation, the
  node-splitting decomposition and the exact multi-arm covariance.

  It exists because the failure mode it guards is not in any likelihood. Two
  programs can fit the same model correctly and both map treatments to
  parameters wrongly; the likelihood then looks fine and every consistency
  statement derived from it is wrong. The tests in
  `tests/test_nma_design.py` check the layer against graph theory rather than
  against a second implementation of the same formula: circuits are
  independent and number `m - n + c`, the design has rank `n - c`, the null
  space is the components' indicator vectors, and ``B d = X_b d_b`` holds for
  arbitrary `d` and every choice of reference.

  Writing it found two real errors, both in the same function, both in the
  sign. The multi-arm covariance had been assembled by *adding* the shared
  arm's variance without regard to sign: `A vs B` and `B vs C` share arm `B`
  with opposite signs, so they covary by `-Var(B)`, and the first version
  reported `+Var(B)`. A simulation of arm means caught it, which is exactly
  the check a formula restated in code cannot catch. And the shared-arm
  variance was taken from `variances` by *length* — a three-arm study has
  three arms and three comparisons, so the length cannot say which is meant;
  `per="arm" | "comparison"` is now explicit and the ambiguous case raises.

  The multi-arm block is singular by construction: a `k`-arm study's
  `k(k-1)/2` comparisons live in `k - 1` dimensions, because
  `(A-B) - (A-C) = C - B`. That is not repaired, it is reported (`rank`) and
  tested, because it is the reason a multi-arm study cannot be entered as `m`
  independent observations.

- **Time-course meta-analysis from aggregate data.** `ma.timecourse` pools a
  treatment effect reported at several time points per study. The within-study
  correlation is spelled out — `rho` for a constant one, `phi` for AR(1) decay
  `phi**|t_i - t_j|`, which `vcalc` builds — and required rather than
  defaulted, because treating a study's three estimates as independent is the
  error that makes a time trend look more precise than it is. The trend's
  standard error differs between the two assumptions, and the tests pin that.

  The pooled object is a curve over time, linear by default and a spline when
  asked for. `curve()` returns a confidence band and a prediction band for a
  new study's curve, `test_trend()` asks whether the effect changes with time
  at all, and `test_nonlinear()` whether the spline was needed. Pinned against
  `metafor::rma.mv` with the same block-diagonal `V` and `struct="GEN"`:
  coefficients to 1e-9, standard errors to 1e-6, variance components to 1e-6,
  the curve to 1e-9.

  Building it found a bug in code shipped earlier this session.
  `RmaMvResult.g_matrix()` returned `None` for every `struct="GEN"` fit,
  because the GEN component stores its dimension as `q` and the accessor looked
  only for `J`. Every random-slope fit therefore reported a between-study
  covariance of zero, and `predict_region` — which adds that covariance —
  reported a prediction region no wider than the confidence one. The accessor
  now reads both, uses the component's `var_names` for labels, and the
  regression test is the first thing in `tests/test_timecourse.py` that could
  have failed.

  Scope: the pairwise model, treatment against a common comparator. A network
  version needs the consistency equations on top and is not attempted.

- **IPD plus aggregate data.** `ipd_two_stage(..., aggregate=...)` adds studies
  that reported only the pooled effect as extra studies in the second stage.
  Not a new estimator — the same second stage with two more rows — but the
  difference between a question about the trials that shared their data and a
  question about all of them. Aggregate estimates must be on the same scale as
  the pooled term, a study cannot appear both as IPD and as aggregate, and the
  two kinds of contributor are counted separately (`n_ipd`, `n_aggregate`) and
  labelled in `per_study_table()`.

  One caveat is emitted rather than documented away: an aggregate estimate and
  a covariate-adjusted IPD estimate are the same estimand only if the
  aggregate model adjusted for the same covariables, and for an odds ratio a
  marginal and a conditional effect differ even when both are right. A fit
  with covariates warns, because silently pooling a conditional with a
  marginal is the failure mode the feature creates.

  Pinned by `tests/test_ipd_aggregate.py`: the mixed pool equals `rma` over the
  combined estimate and variance vectors to 1e-12.

- **Joint inference on discrimination and calibration.**
  `ma.prediction_joint` pools a C-statistic and a calibration slope as a pair.
  Two separate intervals cannot answer either question that matters: "is this
  model both discriminative and calibrated" is a joint hypothesis, and "where
  would a new study's pair fall" is an ellipse. It returns the pooled metrics
  on their own scales, `test_discrimination()` (AUC = 0.5),
  `test_calibration()` (slope = 1), `test_both()` (the 2-df joint test), a
  confidence region for the pooled pair and — through `predict_region` — a
  prediction region for a new study.

  The within-study covariance between the two metrics is the obstacle and is
  not recoverable from published summaries, so `rho` is an explicit argument
  and `sensitivity()` refits along a grid: on the pinned fixture the pooled AUC
  moves by less than 0.01 across `rho` in [0, 0.75] while the calibration slope
  moves monotonically, so a conclusion about discrimination survives the
  assumption and one about the slope's third decimal does not.

  Pinned against `metafor::rma.mv` on constructed blocks: coefficients and
  standard errors to 1e-6, variance components to 1e-5, the joint statistic to
  1e-5. The check worth more than the pin is internal: with `struct="DIAG"` and
  `rho = 0` the joint fit collapses to the two marginal pools, and it does —
  but `struct="UN"` does not, because it also estimates the *between*-study
  correlation, which is -0.12 here with the within-study one set to zero.
  Wrongly assuming the two agree would have made the joint model pointless.

- **Joint pooling of interval-specific hazard ratios, with trend tests.**
  `ma.piecewise_joint` stacks a study's interval-specific log hazard ratios
  into a vector, gives each study its own sampling covariance and a
  between-study covariance on top, and fits them together — so
  `trend_test(kind="linear" | "diff" | "any")` and `predict_region` have the
  covariance a comparison between intervals needs. `ma.piecewise_exp` pools the
  same estimates one interval at a time and remains the right tool for
  reporting each interval on its own.

  The correlation between a study's interval-specific estimates is where this
  model is usually misdescribed, and the documentation said it wrong. The
  intervals *partition* follow-up, so a Poisson fit with a saturated
  interval-by-treatment interaction estimates each interval from disjoint rows
  and the information matrix is block diagonal: the estimates are **exactly
  uncorrelated** (measured at 0.0, not argued). A within-study `rho` therefore
  has no role for participant-level input; it has one for aggregate input,
  where the published hazard ratios come from a model without that structure
  and the correlation cannot be estimated, so `rho` is an assumption to vary.

  Pinned against `metafor::rma.mv` on the stacked block
  (`tests/test_tvhr_joint.py`): per-study sampling covariance to 1e-8, the
  diagonal pooling to 1e-6, the trend estimate and its standard error to 1e-3.
  Generating the reference needed `glm.control(epsilon = 1e-14)`: with R's
  default the per-study standard errors are wrong in the sixth digit, and a
  reference that stops early is not a reference.

  The unstructured pooling is pinned only to 1e-3, and the reason is a result.
  With 16 studies the estimate of the between-interval covariance lands on a
  boundary where one correlation is 1; the likelihood is flat along the ridge,
  and every statistic that contrasts intervals depends on where the optimiser
  stopped. The omnibus test of a constant hazard ratio is 40.6 with
  `struct="DIAG"`, 52.1 with `"UN"` and 57.3 with `"HCS"`. The qualitative
  answer survives that spread; the number does not, so a fit with a boundary
  correlation now warns and quotes the three values.

- **Random slopes in the one-stage IPD model.** `re_slopes=` puts
  fixed-design coefficients into the random part, so `re_slopes=["treat"]` is
  lme4's `(1 + treat | study)` and `struct="DIAG"` is
  `(1 | study) + (0 + treat | study)`. This is a new likelihood rather than a
  new argument: `V_i = sigma2 (I + Z_i Lambda Z_i')` does not invert by a
  rank-one update, so the closed form is gone. Woodbury keeps every step at
  `q x q` — `Omega_i^-1 = I - Z_i M_i^-1 Z_i'` with `M_i = Lambda^-1 + Z_i'Z_i`
  — so a study of 10 000 participants costs what a study of ten costs, and
  `sigma2` is still profiled out analytically.

  `OneStageResult.contrast(L)` reports a confidence band for linear
  combinations and, because the interaction can be random, a prediction band
  for a new study's curve — the one-stage counterpart of
  `ipd_two_stage_curve` for a non-linear treatment effect modifier.

  Pinned by `tests/test_onestage_slopes.py` against `lme4` 2.0.6 on 24 studies
  of 165–275 participants: fixed effects to 1e-6 relative, variance components
  to 1e-5 absolute, log-likelihood to 1e-6, under REML with an unstructured
  covariance, under REML with a diagonal one, and under ML.

  The fixture had to be rebuilt to be worth pinning. A first attempt with 25–60
  participant studies put both implementations on the correlation boundary
  (`rho-hat = 0.99998`), because the sampling noise in each study's own
  intercept and slope is comparable to the variance being estimated — two
  implementations agreeing on a boundary is not evidence. The pinned fixture
  estimates `rho-hat = 0.339` against 0.5 true.

  Still not implemented: a residual variance that differs by study, which is one
  variance per study against a handful of participants each.

- **A joint prediction region for `rma.mv`.** `ma.predict_region` returns the
  ellipsoid $(\theta - \hat\mu)'(V+\Sigma)^{-1}(\theta-\hat\mu) \le c_{1-\alpha}$
  for a new study's true effect vector, `observation=` switches the question to
  its *observed* estimates, and the projection onto one outcome is reported
  (wider than a per-outcome interval, as projecting a joint region is). With
  `J = 1` it is exactly the univariate prediction interval — the only algebra
  check available, because metafor returns `NA` prediction bounds for a
  multilevel fit.

  The reference distribution was not assumed.
  `tools/coverage_predict_region.py` fits 300 replications of two correlated
  outcomes and draws a fresh study from the *true* law, so the count includes
  the error in $\hat\mu$, $\hat{V}$ and $\hat\Sigma$:

  | k | chi-square | Hotelling | two per-outcome z intervals |
  |---|---|---|---|
  | 10 | 0.767 | 0.827 | 0.777 |
  | 20 | 0.917 | 0.930 | 0.897 |
  | 40 | 0.937 | 0.947 | 0.910 |
  | 80 | 0.940 | 0.940 | 0.897 |

  The chi-square reference is asymptotic in the number of studies, so a fit with
  20 or fewer residual degrees of freedom now warns and quotes these numbers.
  `test="t"` swaps in the Hotelling `J(m-1)/(m-J) F(J, m-J)` reference, better
  wherever it differs and identical at `k = 80`; with `J = 1` it is `t^2` on one
  degree of freedom fewer than a univariate prediction interval uses.

  The last column is why the region exists: per-outcome intervals never reach
  0.95 at *any* `k`, including `k = 80` where the ellipsoid has converged,
  because independent intervals cannot be joint.

- **A Monte-Carlo coverage study of the IPD curve, and what it found.**
  `tools/coverage_ipd_curve.py` simulates the two-stage IPD analysis with a
  known true curve and reports how often the band contains it. At 120
  replications of 12 studies the nominal 95% confidence band covered **0.74 to
  0.87**; the prediction band, being wider, covered 0.88 to 0.94. The cause is
  not the curve arithmetic — it is the Wald interval conditioning on the
  covariance estimate, plus a third of the fits placing a variance at zero while
  the truth was positive.

  `test="t"` leaves that unchanged (44 residual degrees of freedom is not a
  small-sample correction). `test="knha"` made it **worse**: 0.31 to 0.51. Its
  degrees of freedom are counted from the stacked rows of the multivariate fit,
  so the t reference is essentially z while `s2w` shrinks the variance. That
  option is now refused with the reason, instead of shipping a band that is
  narrower than the one it was meant to fix. A 25-replication guard runs in the
  test suite so a regression to gross miscalibration fails there.

  `curve(robust=True)` adds the CR1 sandwich on the observed within-study
  residuals with a t reference on `k - p` clusters. Measured over the same 120
  replications it fixes the middle of the covariate range (0.86-0.98 against
  0.81-0.87 model-based) and is *worse* at the edge (0.62 against 0.74 at
  x = 65), where a spline curve is least stable. Neither band is calibrated
  everywhere at k = 12, so the default stays model-based and both are reported
  with their numbers rather than one being sold as the fix.

- **Joint pooling of a non-linear treatment effect modifier in two-stage IPD.**
  `ma.ipd_two_stage_curve` pools a spline interaction the way it has to be
  pooled: the whole treatment block `(treat, treat:B_1, ..., treat:B_K)` as a
  vector, with the study's own coefficient covariance as its sampling
  covariance and an unstructured between-study covariance on top. Pooling the
  coefficients one at a time — which is all the univariate `term=` route can
  do — discards the correlation between them, and then
  `Var{Δ(x)} = b(x)' Cov(μ̂) b(x)` is computed from variances that do not know
  about each other.

  The result carries the average curve with a confidence band and a prediction
  band for a new study (which adds the between-study covariance), a joint test
  of effect modification, and a joint test of non-linearity. The covariate is
  centred at its median by default and the basis is built once with common
  knots, both of which Riley et al. (2020) require and neither of which is
  cosmetic: without centring the linear coefficient is the treatment effect at
  covariate zero, an extrapolation whose standard error was an order of
  magnitude too wide in the reference cohort.

  Pinned by `tests/test_ipd_curve.py` against one `glm` per study and
  `metafor::rma.mv` on the stacked block.

- **Fixed: a balanced study was thrown out of every IPD analysis.** The
  exclusion rule tested whether the number of events equalled the number of
  non-events *in the whole study*, which dropped perfectly usable studies and
  said nothing about the arms. A study is only unusable when an arm has no
  events or only events. Found by the differential test above, which expected
  twelve studies and got eleven.

- **Profile-likelihood intervals for `rma_mv` variance components.**
  `ma.confint_rma_mv` fills the one hole metafor itself leaves in a
  multivariate fit: `rma.mv` reports no standard error for `tau2` or `rho`, and
  an asymptotic one would be wrong at the boundaries anyway. The new function
  computes the interval metafor's `confint` computes — fix the component,
  re-estimate the rest, find where the likelihood has dropped by half a
  chi-square quantile — and reproduces metafor's profile to 1e-14 at every
  point it can evaluate, with endpoints agreeing to about 1e-5.

  It differs from metafor in one place, on purpose. A between-study variance
  can collapse to zero, and then the correlation is not a parameter of the
  model: the likelihood is the same for every `rho`. Where that flat
  likelihood is above the threshold, the point belongs in the interval, and
  metafor's interior-only search reports an endpoint that excludes it. The
  profile here is the supremum over the parameter space, re-started from the
  variance boundaries; `r_compat=True` keeps metafor's answer reachable.

- **Within-subject covariance in `ma.vcalc`.** `time1` and `time2` with `phi`
  are now implemented rather than refused, so a design in which the same
  subjects are measured at several time points no longer has to be approximated
  by a constant correlation. The correlation of two estimates one unit apart in
  time is `phi`, two units apart `phi²`, and two estimates at the same time are
  the same estimate. This is metafor's AR(1) structure.

  The change also reorganised the function around the construction metafor
  actually uses: the correlation entry is the product of an *outcome* factor
  (`obs` / `type`) and an *arm-and-time* factor
  (`grp1` / `grp2` / `time1` / `time2` / `w1` / `w2`), either of which is 1
  when its arguments are absent. Previously the two were alternative branches,
  so a design that needed both could not be expressed. Pinned against metafor
  5.2.1 on the Ishak et al. (2007) dataset its own `vcalc` example uses
  (agreement 1e-16) and on a design using every argument at once. `vcalc`
  gained metafor's positive-definiteness warning as `check_pd`, which is what
  catches a `phi` of 1; a row whose two conditions are the same is now
  rejected instead of producing a variance of zero.

- **Copas selection model.** `ma.copas` implements the selection model of
  Copas and Shi (2000, 2001): a study is published when
  `gamma0 + gamma1/se + delta > 0`, so large studies almost surely and small
  ones only when their result is favourable. Only the combination
  `gamma0 + gamma1/se` is identified, which is why the answer is a surface
  rather than a number.
  - the likelihood, the fits at a fixed `(gamma0, gamma1)`, the sensitivity
    surface, the publication probability and the implied number of
    unpublished studies reproduce `metasens` 1.5.3;
  - `ma.copas_loglik` and `ma.copas_fit` are public so the objective and a
    single fit can be checked on their own;
  - the adjusted estimate follows metasens' path through the surface — the
    contour-orthogonal direction, stopping where residual selection stops
    being significant — and agrees to about 1e-5. The contour tracer splits a
    level into a different number of pieces, which is documented in
    `docs/reference/differences.md` section 3 rather than glossed over;
  - a standard error is withheld (NaN) where `rho` sits on its bound, because
    a constrained fit has no meaningful curvature in that direction;
  - pinned by `tests/test_copas.py` on Fleiss 1993 (the example in metasens'
    own documentation, where no adjustment is needed) and on a synthetic set
    where selection is detected and the estimate does move.
- **Time-varying hazard ratios: piecewise-exponential meta-analysis.**
  `ma.piecewise_exp` splits follow-up at the analyst's cut points, fits each
  study as a Poisson regression with the interval as a stratifying factor — so
  each study gets its own baseline hazard and its own log hazard ratio per
  interval — and pools those interval by interval.
  - `ma.split_followup` expands `(time, event)` into per-interval exposure and
    events; `ma.poisson_irls` is the first-stage solver. Both are public so the
    split and the fit can be audited on their own.
  - A cell with no events is removed rather than fitted: it has no finite
    hazard ratio, and returning a coefficient at minus infinity would look
    like evidence.
  - Two limitations are deliberate: the cut points are the analyst's (choosing
    them is model selection), and pooling each interval separately ignores the
    correlation between a study's interval-specific estimates, so no trend
    test is reported.
  - Pinned by `tests/test_tvhr.py` against `glm(family = poisson)` and
    `metafor::rma`. R's default GLM tolerance leaves its own standard errors
    off by 1.4e-5, so the reference uses `epsilon = 1e-14`.

- **Limit meta-analysis: an effect estimate adjusted for small-study
  effects.** `ma.limitmeta` implements Rücker et al. (2011). The model lets
  the effect depend on the standard error, so standardising by the total
  standard error $\sqrt{se^2+\tau^2}$ turns the likelihood into a straight
  line — the generalised radial plot — whose intercept is the bias parameter
  and whose slope is the effect at infinite precision. Everything is closed
  form, and the standard errors are the ordinary least-squares ones (valid as
  they stand: the standardised residuals have variance exactly 1).
  - three adjusted estimates: `beta0` (the limit effect shifted by the bias),
    `betalim` (radial-plot slope of the shrunken estimates) and `mulim`
    (common-effect estimate of the shrunken estimates);
  - heterogeneity is decomposed into `Q.small` and `Q.resid`, plus $G^2$;
  - `pimeta.limitmeta.radial_regression` is exposed on its own for checking a
    radial plot.
  - Pinned by `tests/test_limitmeta.py` against `metasens` 1.5.3 on Moore 1998
    and on a synthetic dataset; all three adjustments agree to 1e-15.

- **Multivariate reporting layer (`pimeta.mv`).** The estimator was already
  complete; what was missing was everything around it.
  - `ma.vcalc` now implements all four dependency structures `metafor::vcalc`
    documents: the same construct measured several ways (`obs` + `rho`),
    different constructs (`type` + `rho`), independent subgroups inside a
    study (`subgroup`), and contrasts sharing a condition (`grp1` / `grp2`
    with `w1` / `w2`). `rho` also accepts a full correlation matrix, so
    per-pair correlations are available.
  - `ma.shared_control_corr` exposes the shared-arm formula
    `(1/w_C)/sqrt((1/w_A + 1/w_C)(1/w_B + 1/w_C))`. It equals 0.5 exactly for
    equal arms — the 0.5 metafor prints is derived, not assumed.
  - `ma.tau_cor` returns the between-outcome covariance `G` of a
    `struct="UN"` fit and the correlation matrix on top of it, with outcome
    labels.
  - `ma.rho_sensitivity` refits the model along a grid of assumed
    within-study correlations and reports each pooled estimate with its
    interval, which is the sensitivity analysis a scalar `rho` obliges.
  - `tests/test_mv.py` pins all of it against metafor 5.2.1 on Berkey 1998.

- **Prediction-model performance: discrimination and calibration.**
  `pimeta.prediction` pools five metrics, each on the scale and with the
  variance its own literature specifies:
  - `auc_logit` puts an AUC on the logit scale with the Hanley–McNeil variance
    and the delta method. It asks for the event and non-event counts, because
    an AUC's precision is not a function of the AUC alone — two studies
    reporting 0.75 do not have the same estimate.
  - `oe_ratio` puts an observed-to-expected ratio on the log scale with the
    Poisson variance `1/O + 1/E`.
  - `prediction_meta` pools a transformed metric with the existing engine and
    back-transforms the estimate, its interval and the prediction interval, so
    a proportion cannot leave [0, 1]. Calibration slope and intercept, and the
    Brier score, are pooled on the scale they were reported on.
  - An AUC and a calibration slope are deliberately different `metric=`
    values: they are on different scales and mean different things.
  - The Hanley–McNeil variance is checked against a simulation rather than
    against the same formula written twice.
- **One-stage individual-participant-data meta-analysis.** `ma.ipd_one_stage`
  fits one model to every participant with a random intercept per study, the
  route that estimates the treatment effect across all participants at once
  rather than pooling study-level estimates:
  - a continuous outcome uses the closed-form marginal likelihood
    (`V_i = sigma^2 I + tau^2 J`, computed from per-study cross-products, so
    nothing n-by-n is ever formed) and is maximised over the residual and
    between-study variances under REML or ML;
  - a binary outcome uses adaptive Gauss–Hermite quadrature, and the standard
    errors come from the fixed-effect block of the observed information.
  - Verified against `lme4` 2.0.6 — `lmer` under REML and ML, `glmer` for the
    binary model.
  - The two-stage and one-stage routes are different models and can disagree;
    both are here so the disagreement can be looked at rather than assumed
    away. Deliberately absent: study-specific residual variances, random
    treatment slopes, and non-linear covariate terms.
- **Two-stage individual-participant-data meta-analysis.** `pimeta.ipd` runs a
  model inside each study and pools its coefficient:
  - `logistic_model` is maximum likelihood by iteratively reweighted least
    squares, and refuses to invent an estimate when a study has complete
    separation; `linear_model` is OLS with the residual variance estimated.
  - `ipd_two_stage` fits those per study, pools the requested coefficient with
    `rma`, and reports each study's estimate, its participants and its events.
    `term="treat:age_c"` builds the covariate interaction inside every study
    and pools that, which is the analysis most IPD reviews are for.
  - A study that cannot be fitted is named in `excluded` with the reason rather
    than being dropped quietly.
  - Verified against `glm(family = binomial)` and `metafor::rma`.
- **Survival outcomes: RMST and Kaplan–Meier reconstruction.** `pimeta.surv`
  adds the two things a published curve can be turned into:
  - `km_curve` builds a Kaplan–Meier from individual times; `km_reconstruct`
    rebuilds individual times and event indicators from a published (time, at
    risk, events) table, reproducing the curve exactly at every reported time
    point — which is how IPD is obtained for most published trials, and the
    input `ipd_two_stage` expects.
  - `rmst` is the restricted mean survival time with the Uno standard error,
    the summary that stays meaningful when proportional hazards does not hold.
  - RMST checked against R integrating the same step function two ways
    (a hand-written step sum and `integrate(stepfun(...))`), and against a
    closed form for a single-event curve.
- **Dose-response meta-analysis.** A dose-response study contributes several
  contrasts against the same reference group, so the inverse-variance pool of
  them is wrong. `pimeta.dose` supplies the two missing pieces:
  - `dose_cov` builds the within-study covariance. Given `cases` and `n` it
    applies the Greenland–Longnecker correction — re-deriving the reference
    group's case count from the reported relative risks, because a published
    table rounds them and the naive count is not consistent with them — and
    `greenland_counts` exposes the adjusted counts on their own. Given `rho` it
    builds a constant-correlation block instead, for a paper that reports no
    counts.
  - `dosres` assembles the block-diagonal covariance, puts a spline (or
    fractional polynomial, or anything else from `pimeta.basis`) over dose, and
    fits the model with a random intercept per study. The result is an ordinary
    multilevel fit, so `wald_test`, `nonlinear_test`, `predict_curve` and the
    record layer all work on it.
  - Verified against `dosresmeta` 2.2.0 for the adjustment and `metafor` 5.2.1
    `rma.mv` for the fit. `dosresmeta`'s default also randomises the spline
    coefficients, which is a different estimator; the reference here is the
    `rma.mv` call with the same covariance.
- **Non-linear meta-regression: restricted cubic splines and fractional
  polynomials.** The estimator is linear in the parameters, not in the
  moderator, so a curved dose-response only needs a basis expansion — which is
  what the new `pimeta.basis` module provides:
  - `rcs` is Harrell's restricted cubic spline, with knots at his recommended
    percentiles unless they are given explicitly, and the fit forced linear
    beyond the outer knots.
  - `fp` is the Royston–Sauerbrei fractional polynomial: arbitrary powers
    including negative and fractional ones, with `log(x)` for the power 0 and a
    second term multiplied by `log(x)` when a power repeats.
  - Both return a `Basis`, which is array-like (so it can go straight to
    `mods=`) and can re-evaluate itself at new values (so a curve can be
    predicted from it).
  - Both can be written *inside a formula*: `mods="~ rcs(dose, 4)"` or
    `mods="~ fp(dose, [-1, 0]) + year"`. The formula parser now tracks
    parentheses, so a negative number inside a call no longer splits the term,
    and the coefficients are named `rcs(dose, 4)[2]` rather than `mods[2]`.
  - `wald_test` is the joint test that a subset of coefficients is zero
    (`metafor::anova`), and `nonlinear_test` splits it into "is there any
    effect?" and "is the effect non-linear?".
  - `predict_curve` evaluates a fitted basis over a grid and returns the curve
    with a confidence band and a prediction band.
  - Verified against `rms` 8.1.1 (basis matrices, 13 digits), `metafor` 5.2.1
    (coefficients, τ², Q, `anova`, `predict`), and against R's own
    `rma(mods=~x)` / `rma(mods=~log(x))` equivalents for the fractional
    polynomials.
- **`plot_sroc`: the picture the DTA model implies.** One dot per study sized by
  the number of diseased participants, the Rutter–Gatsonis summary curve, and
  the summary point as a crosshair whose arms are the confidence interval on
  each axis, with the AUC and partial AUC annotated. `mada` splits this across
  `crosshair` and `ROCellipse`; here it is one axes.
- **A capability matrix** (`docs/guide/capability-matrix.md`): twelve method
  areas marked implemented / partial / absent, each naming the function and the
  guard test. It was written by reading the source rather than the changelog,
  and it changed the roadmap — the formula parser, the design matrix,
  `predict(newmods=...)`, the mixed-model kernel, `netcomb` and the multivariate
  machinery were all already present.

### Changed

- **Network meta-analysis records now capture the τ² estimator.** `_SETTING_ATTRS`
  gained `method_tau`, `reference_group` and `baseline_reference`, so a DL fit
  and a REML fit of the same table get different `analysis` fingerprints and
  different `settings` blocks. Without them the two were indistinguishable by
  digest even though they are different analyses. `netmeta()` takes column
  names, so it needed no other change: `save_record(..., function="netmeta",
  call={"TE": ..., "treat1": ...})` verifies and replays as it stands, comparing
  every numeric field of the fitted network object.
- **A fifth example notebook: diagnostic accuracy and robust variance.**
  `examples/05-diagnostic-accuracy.ipynb` walks through `escalc_dta` /
  `reitsma` / `hsroc` on `mada::AuditC`, the Rutter–Gatsonis SROC with AUC and
  partial AUC, the four cluster-robust variance variants side by side, and a
  record that verifies and replays itself. It is generated by
  `tools/build_example_notebooks.py` like the other four, so
  `tests/test_example_notebooks.py` re-executes it and it cannot go stale.
- **The four example notebooks are now bilingual.** Every markdown cell leads
  with English and keeps the original Chinese underneath as a `**中文**`
  blockquote, so the runnable examples read the same way as the rest of the
  documentation. Code cells and their stored outputs are untouched. The
  `!!! note` admonition in `03-network-meta-analysis.ipynb` became a plain
  blockquote, which renders in Jupyter and on GitHub rather than only under
  MkDocs Material.

### Added

- **Diagnostic test accuracy: the bivariate model and HSROC.** `pimeta.dta`
  synthesises 2x2 diagnostic tables the way `mada` does:
  - `escalc_dta` turns TP/FN/FP/TN into logit sensitivity and logit
    specificity with delta-method variances. A zero cell gives an infinite
    variance, so `correction_control` decides what happens: the default
    (`"single"`) corrects only the studies that need it, `"none"` never, and
    `"all"` reproduces `mada`'s behaviour. Every correction raises a
    `DtaCorrectionWarning` rather than changing the counts silently.
  - `reitsma` fits the bivariate random-effects model by REML or ML,
    reporting the mean logit pair, the 2x2 study-level covariance, the
    summary sensitivity/specificity/FPR with intervals, and the
    Jacobian-corrected log-likelihood, AIC and BIC that `mada` reports.
  - `hsroc` returns the Rutter–Gatsonis parameters (threshold, accuracy,
    asymmetry, both variance components). Because the two models are
    reparameterisations of one another, it applies the exact algebraic
    transform instead of re-fitting, so both report the same likelihood.
  - `sroc` draws the Rutter–Gatsonis summary curve and `auc` computes the
    AUC and the partial AUC over the observed FPR range, reproducing
    `mada::AUC`'s grid convention exactly.
  - Verified against `mada` 0.5.12 on its own `AuditC` (14 studies) and
    `Dementia` (33 studies) datasets, REML and ML: coefficients, `Psi`,
    log-likelihood, AIC/BIC, HSROC parameters, summary point and both AUCs.
    `tests/test_dta.py` additionally evaluates the likelihood at mada's
    reported variance components and matches it to 1e-6, which shows the two
    implementations are optimising the same function.
  - Two places diverge from `mada` on purpose, because copying it would be
    wrong:
    1. **Continuity correction is per study.** `mada` applies it to every
       study once any table has a zero cell (an `ifelse` where an `if` was
       meant). That breaks locality — adding one study changes every other
       study's contribution — and hits hardest where the sample is smallest,
       i.e. exactly where no correction was needed. On `AuditC`, two of
       fourteen studies need it and the global rule moves the other twelve by
       up to 0.013 in sensitivity and `tau_sens` by 6.7%. Available as
       `correction_control="all"` for exact reproduction.
    2. **The second component is specificity, not FPR.** Same model, sign
       flip, but specificity puts `rho` on the scale the literature uses: a
       threshold effect is a *negative* correlation. `mada`'s FPR-scale values
       are still reported as `mu_fpr`, `Psi_fpr`, `vcov_fpr` and `rho_fpr`,
       so its printed output can be compared term by term.
  - A DTA fit drops straight into `pimeta.record`: `save_record(..., mode=
    "bundle", function="reitsma", call=..., pass_data=True)` gives a record
    that can be `verify()`-ed and `replay()`-ed without the original counts,
    and `EstimatesStore` compares several fits in one table. `pass_data` is a
    new `build_record()` argument for functions that take the whole table as
    their `data=` rather than column names — the column-name inference cannot
    see that case. `examples/dta_record.py` runs the whole loop.
- **Cluster-robust variance estimation for `rma`.** `robust()` previously only
  accepted an `rma.mv` fit; `metafor::robust.rma.uni` had no counterpart, so
  calling it on a univariate fit raised `AttributeError`. The estimator moved
  to the new `pimeta.rve` module and now serves both fit classes through one
  sandwich implementation — a univariate fit is only the case where the
  marginal var-cov matrix is diagonal. CR0/CR1/CR1.S1/CR1.S2 and the
  clubSandwich CR2/CR3 + Satterthwaite path all reproduce metafor 5.2.1 and
  clubSandwich 0.7.0 to the printed precision, including the robust omnibus
  test. `rma.uni` fits now record their `btt`, so the omnibus covers the same
  coefficients metafor would test.
- **`pimeta.record` — standardised, traceable, reproducible input and output.**
  A new subpackage that makes a result carry its own evidence instead of
  relying on a saved script:
  - `save_record` / `load_record` write and read a single JSON file with five
    blocks: `inputs` (输入数据), `analysis` (分析配置), `process` (运算过程),
    `environment` (运行环境) and `results` (结果与诊断).
  - Two storage modes. `mode="digest"` (default) keeps fingerprints, column
    names and row count only, so no participant-level data is copied;
    `mode="bundle"` stores the table as well, which makes the record
    self-verifying (`record.verify()`) and self-replaying (`record.replay()`,
    which re-fits and compares every numeric field).
  - **Two independent SHA-256 fingerprints**, because one hash cannot answer
    both questions. `source_fingerprint` covers every column in file order, so
    it changes when a study is relabelled. `analysis_fingerprint` covers only
    the columns that entered the model (sorted by name) plus the settings that
    can move a number, so relabelling `slab` — or reordering columns, or adding
    an unused one — leaves it unchanged while changing `method` or a value does
    not. Digests are computed over a canonical form, so they are identical
    across platforms.
  - `pimeta.record.jsonio` writes **strict JSON**: `NaN` and `±Infinity` are
    emitted as tagged `{"$pimeta": "nan"}` objects instead of the invalid
    `NaN`/`Infinity` tokens that `json.dumps` produces. The file is readable by
    any strict parser *and* lossless — `tau2 = 0`, `I2 = nan` and `H2 = inf`
    stay three different facts. Writes are atomic and line endings are forced
    to `\n`.
  - `read_table` / `write_table` handle CSV, TSV, JSON, JSONL and NPZ in both
    directions. Text output defaults to the shortest round-trip float
    representation instead of truncating at six significant digits, so a table
    written and read back reproduces the analysis. `.npz` stores numeric and
    string columns and refuses object columns rather than enabling `pickle`.
  - **`EstimatesStore`** is the `estimates store` equivalent: named fits,
    `compare()` across models, `restore()`, and JSON persistence. A loaded store
    keeps the records but not the live objects, and says so instead of
    pretending otherwise.
  - `pimeta.cli.read_table` now delegates to the shared implementation, so the
    CLI and the record layer parse files identically.
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

### Changed

- **`netmeta.py` is down to 2,300 lines, from 7,088.** The ten zero-dependency
  sections — everything that only *reads* a fitted object — now live in their
  own modules under `pimeta/nma/`:

  ```text
  splitting       netsplit          league         netleague
  pairwise_table  netpairwise       complexity     netcomplex(ity)
  convert         crossnma2netmeta  gemtc2netmeta  matrix   netmatrix
                  netcomparison     netbind        paths    netpath
  labels          treats            ranking        vikor
  tables          subgroup          nettable
  ```

  `pimeta.netmeta` re-exports all of them, so `pimeta.netmeta.netsplit` and
  `pimeta.nma.splitting.netsplit` are the same object and not one import path
  changed; `pimeta/nma/__init__.py` carries the map. What remains in
  `netmeta.py` is the fitting code plus `netmetareg`, the heat matrix and the
  `createC`/`invmat`/`hatmatrix` aliases. The split was done by AST span rather
  than by hand, and the whole suite passed without a single test edit.
- **The README no longer enumerates names as comma-separated inline code.**
  Runs such as `` `RR`, `OR`, `RD`, ... `` reorder badly in renderers that treat
  `<code>` as an independent inline box: the commas stay put and the identifiers
  drift to the end of the paragraph, which made the "What is covered" section
  read as a string of commas. Those lists are now fenced `text` blocks (and, in
  the Requirements section, a table), and
  `tests/test_docs.py::test_no_comma_separated_runs_of_inline_code` keeps them
  that way.

### Fixed

- **`conv_delta()` / `conv_wald()` silently collapsed to one element when
  `transf` was a scalar function.** On NumPy 1.x `math.exp(np.array([a, b]))`
  returns `exp(a)` instead of raising, so the `except TypeError` fallback in
  `_apply` never ran: the result had length one and — because NumPy emits a
  `DeprecationWarning` for that conversion — the test suite, which promotes
  those to errors, failed on the `numpy1` CI jobs while `numpy2` passed (there
  the same call raises). The callable is now classified by the *shape* it
  returns rather than by whether the call succeeded, and the probe's warning is
  suppressed. `conv_wald()` used the same pattern and had the same defect.
- **`selmodel()` used the wrong delta map, so every smooth family could stop
  short of the maximum.** metafor keeps two maps apart: `delta.transf.fun` (the
  exponential, used only when `delta.max` contains an infinity or the type
  forces it) and the `mapfun` that is actually applied, which is `NA` for every
  family with a finite `delta.max` and therefore falls back to the scaled
  logistic `lb + (ub - lb) * plogis(x)`. pimeta used the exponential everywhere.
  On the 20-study reference data this left `power2` at delta = (0.0014, 100) with
  ll = 19.13 instead of R's (0.00046, 58.24) with ll = 20.2552; `beta`,
  `halfnorm`, `negexp`, `logistic`, `power`, the `*1` and `*2` families,
  `negexppow` and `trunc` were affected the same way.
- **`selmodel()` no longer produces `ll = nan` from an overflowing weight.** The
  power-family ratio `(1-x)**e / (1-s0)**e` is evaluated as a single `exp` of a
  difference of logarithms (the direct form is `0/0` at `s0 = 1` and `inf/inf`
  for a large negative exponent), `exp(tau2)` is capped, `1/(1+exp(-x))` became
  `expit`, and a non-finite log-likelihood is replaced by a large finite
  penalty instead of being handed to the optimiser, which used to keep whatever
  NaN point it happened to be standing on.
- **`stepcon` is fitted with the monotone parameterisation instead of a
  constraint.** metafor solves this model with an SQP solver (Rsolnp) under the
  constraint `delta[j] >= delta[j+1]`; SciPy's SLSQP is the same class of
  solver, but it reports failure and still returns the iterate it stopped on,
  so pimeta could hand back `delta = (1, 0.066, 0.0995, ...)` — a sequence no
  `stepcon` model can produce — together with an `ll` above the true maximum.
  `stepcon` and `stepfun(decreasing=TRUE)` are the same model, so the
  cumulative parameterisation the latter uses is now used for both: the
  monotone cone is reached by construction and the boundary optimum (two
  adjacent weights equal, the usual answer) is found. On `dat.bcg` with
  `steps=(0.025, 0.5, 1)` the fit now reproduces R: ll = -12.6129799 vs
  -12.6129800, se(beta) = 0.207145 vs 0.207146, delta = (1, 1, 0.68944) vs
  (1, 0.99999977, 0.68945369).
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