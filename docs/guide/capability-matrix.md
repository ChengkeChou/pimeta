# Capability matrix

What `pimeta` can do today, what it does half-way, and what it does not do at
all — with the code and the test that back every row.

This page exists because a package that claims breadth is hard to trust. The
rows below were written by reading the source, not by reading the changelog:
each one names the function, and where a guard test exists it names the file.
A row with no test is a row that nothing protects, and that is worth knowing
too.

## How to read it

| Mark | Meaning |
|---|---|
| **yes** | implemented and pinned against an R implementation |
| **partial** | the estimator exists, part of the workflow around it does not |
| **no** | nothing in the package |

## The methods

### 1. Non-linear meta-regression — **yes**

Wanted: restricted cubic splines, fractional polynomials, a test for
non-linearity, and a predicted curve.

All four are in `pimeta.basis`. `ma.rcs` gives Harrell's restricted cubic
spline with its knots at the recommended percentiles, `ma.fp` the
Royston–Sauerbrei fractional polynomial, and both can be written inside a
formula (`~ rcs(dose, 4)`, `~ fp(dose, [-1, 0])`) or passed to `mods=` as
objects. `ma.nonlinear_test` separates "is there an effect?" from "is it
non-linear?", `ma.wald_test` runs the same joint test over any subset of
coefficients, and `ma.predict_curve` returns the curve with a confidence band
and a prediction band.

Pinned by `tests/test_basis.py` against `rms` 8.1.1 for the basis matrices and
`metafor` 5.2.1 for the fit, the `anova` tests and `predict`.

What the audit changed: the formula parser, the design-matrix builder and
`predict(newmods=...)` were already in the tree, so this was a basis and two
tests rather than a new regression layer.

Still missing, on purpose: automatic selection of the number of knots or of the
power set. Both are model-selection procedures (with their own multiplicity
problems), not bases, and neither is a one-liner done honestly.

### 2. Dose–response meta-analysis — **yes**

Wanted: the non-linear dose–response curve with the within-study covariance
handled.

`pimeta.dose` has the two pieces that make this a different problem from
ordinary meta-regression:

- `dose_cov` builds the within-study covariance of contrasts that share a
  control group. With `cases` and `n` it applies the Greenland–Longnecker
  correction, which re-derives the reference group's case count from the
  reported relative risks (a published table reports them rounded, so the naive
  count is not consistent with them). With `rho` it builds a constant-correlation
  block for a paper that prints no counts. `greenland_counts` is exposed on its
  own for checking a table.
- `dosres` assembles the block-diagonal covariance, puts a spline over dose,
  and fits the whole thing as a multilevel model with a random intercept per
  study — then reports the coefficients, the between-study variance and the
  residual heterogeneity.

Pinned by `tests/test_dose.py` against `dosresmeta` 2.2.0 (the adjustment) and
against `metafor` 5.2.1 `rma.mv` with the same covariance (the fit).

One deliberate difference: `dosresmeta`'s default also puts random effects on
the spline coefficients, so its `Psi` is a *matrix* over the basis rather than a
single variance. That is a different estimator, not a different implementation,
and `pimeta` uses the random-intercept model; the `random=` argument of
`rma_mv` is available if a random-slope version is wanted.

Still missing: selecting the number of knots and the reference dose
automatically, and dose–response *network* meta-analysis.

### 3. Two-stage IPD — **yes**

Wanted: fit a model inside each study, then pool the coefficient vectors.

`pimeta.ipd` does exactly that. `logistic_model` and `linear_model` are the
first stage (maximum likelihood by IRLS, and OLS with the residual variance
estimated); `ipd_two_stage` runs them per study, reports every study's estimate
and its events, and pools one coefficient with `rma`. A covariate interaction
is pooled the same way — `term="treat:age_c"` builds the product inside each
study, which is the analysis most IPD reviews are actually for.

Studies that cannot be fitted are not silently dropped: a study where an arm has
no variation in the outcome, or where a covariate separates it, is named in
`result.excluded` with the reason.

Pinned by `tests/test_ipd.py` against `glm(family = binomial)` and
`metafor::rma`.

**Non-linear modification** is a different problem, and `ma.ipd_two_stage_curve`
is the answer to it. A spline interaction is several coefficients from the same
study, they are correlated, and pooling them one at a time — all the univariate
`term=` route can do — throws that correlation away; the curve is then built
from variances that do not know about each other, so its band is not a band. The
new function pools the whole treatment block as a vector, with an unstructured
between-study covariance, and evaluates

.. math::

    \\Delta(x) = b(x)'\\hat\\mu,
    \\qquad \\operatorname{Var}\\{\\hat\\Delta(x)\\}
    = b(x)' \\operatorname{Cov}(\\hat\\mu)\\, b(x)

It returns two bands — the average curve, and where a *new study's* curve would
fall, which adds `Sigma` — plus a joint test of effect modification and a joint
test of non-linearity.

Three details it refuses to paper over. The covariate is centred at its median
by default: the linear coefficient is the treatment effect at that value, and
without centring it is the effect at covariate zero, an extrapolation whose
standard error in the reference cohort was an order of magnitude too wide. The
basis and its knots are built **once** and evaluated in every study, so the
coefficients pooled are the same quantity in each. And a study whose arm has no
variation is excluded *with the reason*, while the curve flags points outside the
observed covariate range.

Pinned by `tests/test_ipd_curve.py` against one `glm` per study and
`metafor::rma.mv` on the stacked block: pooled coefficients to 1e-6, the curve to
1e-6, the two joint tests to 1e-5. That cohort's between-study covariance
estimates to zero, which is a property of 30 studies with noisy coefficients and
not of the implementation, so a second test builds heterogeneous studies directly
to check that the prediction band really is wider than the confidence band.

**When the covariance cannot be identified**, `diagnostics()` says so instead of
leaving it to be discovered later: it reports how many studies each variance
parameter is estimated from, which `tau2` components sit at zero, which
correlations sit on ±1 (which makes `Sigma` singular and the likelihood flat, so
another implementation may stop elsewhere on the same ridge), and the condition
number of `Sigma`. A fit that trips any of these emits a `RuntimeWarning`.
`struct="HCS"` — one common correlation instead of `K(K+1)/2` of them — is the
fallback for a small `k`, and `struct="UN"` stays the default because it is the
honest one when the studies can pay for it.

**What the band actually covers** was measured rather than assumed
(`tools/coverage_ipd_curve.py`, 120 replications of 12 studies, nominal 95%):

| x | model-based (default) | CR1 + t (``robust=True``) |
|---|---|---|
| 45 | 0.758 | 0.875 |
| 50 | 0.867 | 0.983 |
| 55 | 0.808 | 0.975 |
| 60 | 0.817 | 0.858 |
| 65 | 0.742 | 0.617 |

Neither is calibrated, and the failure is not where a simple story would put
it. The model-based Wald band under-covers throughout, because it conditions on
the covariance estimate and a third of the fits put a variance at zero while the
truth was positive. The cluster-robust CR1 band (sandwich on the observed
within-study residuals, t on ``k - p`` clusters) fixes the middle — 0.86 to 0.98
— but is *worse* at the edge of the covariate range, where a spline curve is
least stable. So no default is offered as "the calibrated one"; both are
reported, `robust=True` is available and documented with these numbers, and the
honest reading is that the edge of the range needs more studies, not a different
quantile.

The prediction band is wider and therefore over-covers under CR1 (0.98 to 1.00)
while reaching 0.88 to 0.94 model-based. `test="t"` changes nothing (44 residual
degrees of freedom is not a small-sample correction) and `test="knha"` made the
model-based band **worse** — 0.31 to 0.51 — because `rma_mv` computes its
residual degrees of freedom from the *stacked rows*, so the t reference is
nearly z while `s2w` shrinks the variance. It is refused rather than shipped.

**IPD plus aggregate data.** `ipd_two_stage(..., aggregate=...)` adds studies
that reported only the pooled effect, as extra studies in the second stage —
which is not a new estimator but is the difference between a question about the
trials that shared their data and a question about all of them. The aggregate
estimates have to be on the same scale as the pooled term (a log odds ratio for
a logistic first stage), a study cannot appear both as IPD and as aggregate,
and the two kinds of contributor are counted separately (`n_ipd`,
`n_aggregate`) and labelled in `per_study_table()`.

One caveat is emitted rather than documented away: an aggregate estimate and a
**covariate-adjusted** IPD estimate are the same estimand only if the aggregate
model adjusted for the same covariables, and for an odds ratio a marginal and a
conditional effect differ even when both are right. A fit with covariates
therefore warns. The alternative — silently pooling a conditional with a
marginal — is the failure mode this feature creates, so it is the one to shout
about.

Pinned by `tests/test_ipd_aggregate.py`: the mixed pool equals `rma` over the
combined estimate and variance vectors to 1e-12, which is the only arithmetic
claim there is.

### 4. One-stage IPD — **yes**, with a stated model

Wanted: participant-level data in one model, with participants nested in
studies.

`ma.ipd_one_stage` fits
:math:`g(E[y_{ij}]) = x_{ij}'\beta + u_i` with :math:`u_i \sim N(0,\tau^2)`.
For a continuous outcome the integral over the random effect is analytic, so
the likelihood is closed form and `V_i = \sigma^2 I + \tau^2 J` is never
inverted — GLS and the profile likelihood are computed from per-study
cross-products, which keeps it linear in the number of participants. For a
binary outcome the integral is approximated by adaptive Gauss–Hermite
quadrature.

Pinned by `tests/test_onestage.py` against `lme4` 2.0.6: `lmer` under REML and
ML, and `glmer` for the binary model.

**Random slopes.** `re_slopes=` puts fixed-design coefficients into the random
part, so `re_slopes=["treat"]` is `lmer`'s `(1 + treat | study)` and
`struct="DIAG"` is `(1 | study) + (0 + treat | study)`. That is a new
likelihood, not a new argument: $V_i = \sigma^2 (I + Z_i \Lambda Z_i')$ does
not invert by a rank-one update, so the closed form is gone. Woodbury keeps
every step at $q \times q$ —
$\Omega_i^{-1} = I - Z_i M_i^{-1} Z_i'$ with $M_i = \Lambda^{-1} + Z_i'Z_i$
and $\log|\Omega_i| = \log|\Lambda| + \log|M_i|$ — so a study of 10 000
participants costs what a study of ten costs, and $\sigma^2$ is still profiled
out analytically.

Pinned by `tests/test_onestage_slopes.py` on 24 studies of 165–275
participants, against `lme4` 2.0.6: fixed effects to 1e-6 relative, variance
components to 1e-5 absolute, log-likelihood to 1e-6 — under REML with an
unstructured covariance, under REML with a diagonal one, and under ML. The
fixture is large on purpose. A first attempt with 25–60 participant studies
put both implementations on the correlation boundary ($\hat\rho = 0.99998$)
because the sampling noise in each study's own intercept and slope is
comparable to the variance being estimated, and two implementations agreeing
on a boundary is not evidence. The pinned fixture estimates
$\hat\rho = 0.339$ against 0.5 true.

**A non-linear interaction** is the same code plus a contrast: pass the basis
columns as `covariates=`, name them in `re_slopes=`, and read the curve off
with `fit.contrast(L)`, which returns a confidence band for the average curve
and — because the interaction is random — a prediction band for a new study's
curve. That is the one-stage counterpart of `ma.ipd_two_stage_curve`, and the
same caveats apply: the Wald band is asymptotic in the number of studies, and
at 12 studies it does not cover what it claims.

Deliberately not implemented: a residual variance that differs by study. That
is one variance per study against a handful of participants each, and the
profile likelihood answers it with a boundary.

The two-stage and one-stage routes can disagree. That is a result, not a bug —
compare them with `ma.ipd_two_stage` when it matters.

### 5. Diagnostic test accuracy — **yes**

`ma.escalc_dta` prepares the 2x2 counts, `ma.reitsma` fits the bivariate
random-effects model by REML or ML, `ma.hsroc` gives the Rutter–Gatsonis
parameters without re-fitting, `ma.sroc` returns the summary curve, and
`ma.auc` the AUC and partial AUC.

Pinned by `tests/test_dta.py` against `mada` 0.5.12 on two datasets under both
estimators. Two behaviours are deliberately corrected rather than copied; see
[the differences page](../reference/differences.md) §4.

Not a gap: `mada`'s function surface. `pimeta` implements the methods, not
another package's API.

`ma.plot_sroc` draws the picture `mada` splits across two functions: one dot per
study, the summary curve, and the summary point as a crosshair with the
confidence interval on each axis.

### 6. Robust variance estimation — **yes**

`ma.robust` serves both `ma.rma` and `ma.rma_mv` fits through one sandwich
implementation: CR0, CR1, CR1.S1, CR1.S2, and the clubSandwich CR2/CR3 path
with Satterthwaite degrees of freedom and a robust omnibus test.

Pinned by `tests/test_rve.py` against `clubSandwich` 0.7.0.

### 7. Survival outcomes — **yes**

Already there, in more depth than "pool a log hazard ratio":

- a pooled log HR is an ordinary effect size — `ma.metagen` treats `"HR"` as a
  relative effect, and `ma.rd` converts a pooled HR into an absolute risk
  difference;
- `ma.rmst` is the restricted mean survival time, the area under a Kaplan–Meier
  curve up to a horizon, with the Uno standard error when the risk sets are
  supplied. It is the summary that stays meaningful when proportional hazards
  fails;
- `ma.km_curve` builds a Kaplan–Meier curve from individual times, and
  `ma.km_reconstruct` rebuilds individual times and event indicators from a
  published (time, at risk, events) table — which is how most "IPD" is actually
  obtained, and it feeds `ma.ipd_two_stage` directly;
- `ma.piecewise_exp` gives the **time-varying hazard ratio**. Follow-up is
  split at the analyst's cut points and each study is fitted as a Poisson
  regression with the interval as a stratifying factor, so every study
  estimates its own baseline hazard and its own log hazard ratio per interval;
  those are then pooled interval by interval. `ma.split_followup` and
  `ma.poisson_irls` are exposed for checking the split and the fit.
- `ma.piecewise_joint` pools the same estimates **as a vector**: each study's
  interval-specific log hazard ratios enter with their own sampling covariance
  and a between-study covariance on top, which is what a comparison between
  intervals needs. `trend_test(kind=...)` reports three Wald tests —
  `"linear"` (the usual monotone trend), `"diff"` (last interval against the
  first) and `"any"` (the omnibus test that the hazard ratio is constant) — and
  `ma.predict_region` applies to the underlying fit.

On the correlation between a study's interval-specific estimates, which is
where this model is usually misdescribed. The intervals *partition* each
participant's follow-up, so a Poisson fit with a saturated
interval-by-treatment interaction estimates each interval's log hazard ratio
from disjoint rows and the information matrix is block diagonal: the estimates
are **exactly uncorrelated**, measured at 0.0 rather than argued. A
within-study `rho` therefore has no role for participant-level input. It has
one for *aggregate* input, where the published hazard ratios come from a model
without that structure and the correlation cannot be estimated; there `rho` is
an assumption to be varied.

Two limitations, stated rather than papered over. The cut points are not
chosen by the data — that is model selection with its own multiplicity, the
same problem as choosing the number of spline knots. And the between-study
correlation is not always identifiable: with few studies `Sigma` lands on a
boundary where one correlation is ±1, the likelihood is flat along a ridge, and
every statistic that contrasts intervals depends on where the optimiser
stopped. On the 16-study fixture the omnibus statistic is 40.6 with
`struct="DIAG"`, 52.1 with `"UN"` and 57.3 with `"HCS"` — the qualitative answer
survives that spread, the number does not, and a fit with a boundary
correlation warns.

Pinned by `tests/test_tvhr.py` against `glm(family = poisson)` in R and then
`metafor::rma` per interval, and by `tests/test_tvhr_joint.py` against
`metafor::rma.mv` on the stacked block: coefficients to 1e-6, the per-study
sampling covariance to 1e-8, the diagonal pooling to 1e-6. The unstructured
pooling is pinned only to 1e-3, and the test says why — that fit is on the
ridge. One detail worth knowing: R's default `epsilon = 1e-8` leaves the GLM
standard errors off by 1.4e-5, so the reference is generated with
`epsilon = 1e-14`; the difference looks like an implementation gap and is a
convergence tolerance.

Nothing here claims that pooling log HRs handles non-proportional hazards —
it does not, which is why RMST and this module exist.

### 8. Prediction-model performance — **yes**

Wanted: meta-analysis of discrimination and calibration.

`pimeta.prediction` pools five metrics, each on the scale and with the variance
its own literature specifies:

| metric | pooled on | variance from |
|---|---|---|
| discrimination (AUC) | logit | Hanley–McNeil, then the delta method |
| calibration slope | as reported | the study's own |
| calibration intercept | as reported | the study's own |
| O:E ratio | log | Poisson, `1/O + 1/E` |
| Brier score | as reported | the study's own |

The point of the module is that an AUC cannot be pooled with a calibration
slope — different scale, different meaning — and that an AUC's variance is not
a function of the AUC alone: `ma.auc_logit` asks for the event and non-event
counts because two studies reporting 0.75 have different precision.

Pinned by `tests/test_prediction.py`, which checks the transforms exactly,
checks that the pooling *is* `rma`, and checks the Hanley–McNeil variance
against a simulation rather than against the same formula written twice.

**Joint inference.** `ma.prediction_joint` pools the C-statistic and the
calibration slope as a pair, because two separate intervals cannot answer
either question that matters: "is this model both discriminative and
calibrated" is a joint hypothesis, and "where would a new study land" is an
ellipse. It returns the pooled metrics on their own scales,
`test_discrimination()` (AUC = 0.5), `test_calibration()` (slope = 1),
`test_both()` (the 2-df joint test), a confidence region for the pooled pair,
and — through `ma.predict_region` — a prediction region for a new study.

The within-study covariance between the two metrics is the obstacle, and it is
not recoverable from published summaries. `rho` is therefore an explicit
argument, defaulting to 0 only so that it cannot be silently invented, and
`sensitivity()` refits along a grid and reports what the assumption is worth:
on the pinned fixture the pooled AUC moves by less than 0.01 across
$\rho \in [0, 0.75]$ while the calibration slope moves monotonically, so a
conclusion that rests on the AUC survives and one that rests on the slope's
third decimal does not.

Pinned by `tests/test_prediction_joint.py` against `metafor::rma.mv` on the
constructed blocks: coefficients and standard errors to 1e-6, variance
components to 1e-5, the joint statistic to 1e-5. One check is internal rather
than external and worth more than the pin: with `struct="DIAG"` and `rho = 0`
the joint fit must collapse to the two marginal pools, and it does — but
`struct="UN"` does *not*, because it estimates the between-study correlation
too (here −0.12 with the within-study one set to zero). That distinction is
the module's point in miniature.

### 9. Multivariate meta-analysis — **yes**

Already there: the estimator is complete. `ma.rma_mv` implements every metafor
variance structure (CS, HCS, AR, TOEP, UN, GEN), random slopes, custom weight
matrices, and moderator formulas. `ma.matreg` regresses one outcome on others
between studies. `ma.vcalc` builds the sampling covariance. `ma.aggregate_escalc`
aggregates dependent effect sizes within a cluster. `ma.sigma_matreg` recovers
the coefficient covariance.

The reporting layer is in `pimeta.mv`:

- **the sampling covariance** (`ma.vcalc`) covers all four dependency
  structures metafor documents — the same construct measured several ways
  (`obs` + `rho`), different constructs (`type` + `rho`), independent subgroups
  within a study (`subgroup`), and contrasts that share a condition
  (`grp1` / `grp2`). The last one is derived, not assumed: two contrasts
  against the same control share that arm's sampling variance, so
  $\rho = (1/w_C)/\sqrt{(1/w_A+1/w_C)(1/w_B+1/w_C)}$, which is exactly 0.5
  for equal arms. `ma.shared_control_corr` exposes that formula.
- **`ma.tau_cor`** returns the between-outcome covariance `G` of a
  `struct="UN"` fit and the correlation matrix on top of it, with outcome
  labels — the number a reader wants, since `G` itself is on the effect-size
  scale. It reproduces `cov2cor(fit$G)` exactly.
- **`ma.rho_sensitivity`** refits the model along a grid of assumed
  within-study correlations and reports every pooled estimate with its
  interval. This is the obligation the scalar `rho` creates: the correlation
  cannot be estimated from the data (each study contributes a handful of
  estimates), so what has to be shown is what the assumption is worth.

`rma.mv` reports no `se.tau2` / `se.rho` for these models, and that is not an
omission to be filled with an asymptotic standard error: the approximation is
unreliable for a variance at its zero boundary and for a correlation at ±1.
What the parameters do admit is a **profile-likelihood interval**, and
`ma.confint_rma_mv` computes one — fix the component at a value, re-estimate
the rest, and read off where the likelihood has fallen by half a chi-square
quantile. That is what `metafor::confint` does, and the profile itself is
reproduced to 1e-14 at every point metafor can evaluate; the endpoints agree
to about 1e-5, which is the tolerance of the root finder both use.

One endpoint is deliberately different. At 99% on the Berkey fit metafor
reports `[-0.835, 1]`, but a model in which one between-study variance is zero
has likelihood 1.3813 at *every* correlation — the correlation is then not a
parameter of the model — and 1.3813 is above that level's threshold of 0.2755.
An interval excluding it would exclude points with a higher profile
likelihood, so `pimeta` reports the whole range. `r_compat=True` follows
metafor's interior branch. See [the differences page](../reference/differences.md) §4.

A region for a *new* study is a different object, and `ma.predict_region`
provides it.  For pooled effects $\hat\mu$ with covariance $V$ and a
between-study covariance $\Sigma$, a new study's true effects satisfy

$$(\theta_{\text{new}} - \hat\mu)' (V + \Sigma)^{-1}
  (\theta_{\text{new}} - \hat\mu) \le c_{1-\alpha}$$

which is the ellipsoid to report in place of one interval per outcome.  Passing
the new study's sampling covariance as `observation=` adds it and answers the
question about its *observed* estimates instead.  With $J = 1$ the region is
exactly the univariate prediction interval — the only algebra check available,
since metafor returns `NA` prediction bounds for a multilevel fit.

Two numbers have to be reported with it, and both were measured
(`tools/coverage_predict_region.py`, 300 replications, two correlated
outcomes, nominal 0.95):

| k | chi-square reference | Hotelling reference | two per-outcome z intervals |
|---|---|---|---|
| 10 | 0.767 | 0.827 | 0.777 |
| 20 | 0.917 | 0.930 | 0.897 |
| 40 | 0.937 | 0.947 | 0.910 |
| 80 | 0.940 | 0.940 | 0.897 |

The chi-square reference is asymptotic in the number of studies, so at small
$k$ the region covers less than it claims; `test="t"` swaps in the Hotelling
$J(m-1)/(m-J)\,F(J, m-J)$ reference with $m = k - p$, which is better
everywhere it differs and identical at $k = 80$.  A fit with 20 or fewer
residual degrees of freedom warns and quotes these numbers rather than letting
a reader assume the nominal level.

The third column is the point.  The per-outcome intervals the region replaces
never reach 0.95 at *any* $k$ — including $k = 80$, where the ellipsoid has
converged — because independent intervals cannot be joint: a point can sit
inside both and outside the region they are supposed to describe.

Pinned by `tests/test_mv.py` against `metafor` 5.2.1 on the Berkey 1998
dataset — the example metafor's own `vcalc` documentation uses.

One behaviour is corrected rather than copied: metafor silently discards `rho`
when neither `obs` nor `type` is given, returning a correlation of 1. See
[the differences page](../reference/differences.md) §4.

The fifth structure is a within-subject design: `time1` (and optionally
`time2`) with `phi` replaces the fixed correlation by an autocorrelation,
$\rho_{ij} = \phi^{|t_i - t_j|}$, so two measurements one unit apart correlate
`phi` and two at the same time are the same estimate. `vcalc` warns when a
cluster's block stops being positive definite, which is what happens at
`phi = 1`; metafor's `checkpd` does the same. Pinned against metafor on the
Ishak et al. (2007) dataset its own example uses, and on a design with every
argument at once.

metafor builds the correlation as a *product* of two factors — the outcome
correlation (`obs` / `type`) and the arm-and-time correlation (`grp1` /
`grp2` / `time1` / `time2` / `w1` / `w2`) — and so does `pimeta`. Either
factor is 1 when its arguments are absent.

Not a gap: correlation structures beyond the five listed at the top of this
section.  Anything else has to be given as an explicit matrix in `rho`.

### 10. Component network meta-analysis — **yes**

`ma.netcomb` is the additive component model (Rücker et al. 2020): a
combination's effect is the sum of its components' effects, $\delta = B\theta$
with $\theta = C\beta$, estimated by weighted least squares, with the additive
$Q$ and the component-difference test. `ma.netcomplex` derives the effect of any
requested combination from the fitted components, and `ma.discomb` handles
disconnected components. All three are pinned against `netmeta` 3.7.0.

Missing: the component *interaction* (inconsistency) model and time-course NMA.
(IPD-NMA and dose-response NMA are separate areas and are done; see items 13
and 14.)

### 11. Publication bias — **yes**

Already there, and larger than "a funnel plot": `ma.selmodel` with eighteen
selection functions (beta, half-normal, negative-exponential, logistic, power,
their truncated variants, step functions, and the truncation family), plus
`ma.profile_selmodel` for the profile of a fitted selection model. The classic
tools are present too: `ma.trimfill` for trim-and-fill, `ma.fsn` for the
file-drawer, and `ma.regtest` and `ma.egger_test` for the regression test. All
of it is pinned against `metafor` and `meta`.

`ma.limitmeta` adds the adjusted estimate of Rücker et al. (2011). It is not a
selection model but an extended random-effects model in which the effect may
depend on the standard error; standardising by the total standard error turns
the likelihood into a line, so the bias parameter is an intercept and the
limit effect a slope, and both come out in closed form. Three adjusted
estimates are available (`beta0`, `betalim`, `mulim`), and the heterogeneity is
decomposed into the part explained by small-study effects and the part that is
not. Pinned by `tests/test_limitmeta.py` against `metasens` 1.5.3 on Moore
1998 and on a synthetic set.

Missing: any Bayesian model averaging.

`ma.copas` adds the selection model itself (Copas and Shi 2000, 2001). It is
the one place in the package where the answer is deliberately a *surface*
rather than a number: only the combination $\gamma_0 + \gamma_1/se$ is
identified from the observed studies, so the analysis reports the pooled
effect along a path through the ( $\gamma_0$, $\gamma_1$ ) plane, the
publication probability and the number of studies the implied selection would
leave unpublished, and the residual-selection test at each step.

The likelihood, the fits at a fixed $(\gamma_0, \gamma_1)$, the sensitivity
surface, the publication probabilities and the implied number of unpublished
studies reproduce `metasens` 1.5.3 exactly or near it. Everything *after* the
surface — the contour tracing, the per-contour least squares, the weighted
slope, the orthogonal path — was shown to be the reference's algorithm step for
step: metasens' own downstream code, run on pimeta's surface, returns
`slope = 0.14937955364276298`, the value pimeta computes to the last digit,
with identical per-contour piece sizes. What is left is the surface itself,
where each grid point is a three-parameter fit and the two optimisers stop in
different places (1.4e-5 at the worst point), giving 2e-8 on the adjusted
effect and 4e-5 relative on its standard error. That separation is spelled out
in `docs/reference/differences.md` §3, and the tests pin both halves.

That path is not inference, though, and the capability matrix row is marked
"yes" because of `ma.copas_region`, not because of it. The region is the
likelihood-ratio set of $(\gamma_0, \gamma_1)$ at a chosen confidence level —
the selection scenarios the data cannot rule out — projected onto the adjusted
effect to give an interval, along with whether a scenario with no
precision-dependent selection is admissible and whether every admissible
scenario agrees on the sign. Projecting a joint confidence set is conservative
by construction, and it is documented that way; the coverage was not
simulated, and the tests do not claim it was.

Not offered: Bayesian model averaging (RoBMA). It needs an MCMC sampler, which
the dependency policy rules out; a half-validated one would be worse than the
gap.

### 12. Time-varying effects — **no**

Two different things are easy to conflate here, and only one of them is done.

A time-varying **hazard ratio** *is* implemented: `ma.piecewise_exp` (row 7)
splits follow-up at given break points, fits a Poisson model with a log-time
offset per interval, and pools the interval-specific log hazard ratios. That is
a treatment effect that changes over time, and it is pinned against `survival`
and `metafor` in `tests/test_tvhr.py`. `ma.piecewise_joint` pools the same
estimates as a vector and reports the trend tests that comparison needs, pinned
against `metafor::rma.mv` in `tests/test_tvhr_joint.py`.

What is absent is a time-varying **pooled effect size from aggregate data** —
dynamic heterogeneity, where the pooled estimate and $\tau^2$ are functions of
a continuous time index, and the usual candidates are a Gaussian process or a
state-space model on the coefficients. There is no reference implementation in
R to check one against, and this package does not ship models it cannot
verify. It is a research direction rather than a missing feature.

**Time-course meta-analysis from aggregate data** *is* implemented, and it is
the case most reviews actually meet: a trial reports the treatment effect at 3,
6 and 12 months. `ma.timecourse` pools them with the within-study correlation
spelled out — `rho` for a constant one, `phi` for AR(1) decay
`phi**|t_i - t_j|`, which `ma.vcalc` builds — and a curve over time that can be
linear or a spline. `curve()` returns a confidence band for the average curve
and a prediction band for a new study's curve, `test_trend()` asks whether the
effect changes with time at all, and `test_nonlinear()` whether a spline was
needed.

The correlation is required rather than defaulted, because the alternative —
treating a study's three estimates as independent — is the error that makes a
time trend look more precise than it is; the difference in the trend's standard
error is pinned in the tests. Pinned against `metafor::rma.mv` with the same
block-diagonal `V` and `struct="GEN"`: coefficients to 1e-9, standard errors to
1e-6, variance components to 1e-6, the curve to 1e-9.

Scope: this is the **pairwise** model, treatment against a common comparator.
The network version — consistency on the whole treatment difference over time —
is item 15.

### 13. Network meta-analysis with individual participant data — **yes**

`ma.nma.ipd` fits the one-stage IPD network model: one intercept per study,
treatment effects against a chosen reference, study-centred covariates as main
effects, optional treatment-by-covariate interactions, and a random treatment
effect per study with a full between-study covariance. It is the class of model
`multinma` fits, except that the between-treatment covariance is estimated
rather than constrained to a single shared heterogeneity SD.

Covariates are centred **within study**. That is not cosmetic: it keeps the
treatment parameters on their within-study scale, lets the study intercepts
absorb between-study differences in the covariate distribution, and makes the
interaction term a within-study effect modification rather than a
between-study comparison an ecological bias could produce.

The parameterisation comes from `pimeta.nma.design`, the network layer this was
built on, so the contrast matrix the algebra tests check is the one the
likelihood uses: `contrast(a, b)` is a row combination of `B`, and the league
table is `B` applied to the reference vector.

Pinned against `lme4::lmer(y ~ 0 + study + A + x + (0 + A | study))` on an
8-study, 3960-participant, three-treatment fixture: effects and the covariate
coefficient to 1e-8, standard errors to 7e-6, `sigma2` to 9e-8, the
between-study covariance to 1.4e-5 on its flattest direction, the
log-likelihood to 1e-10. Two degenerations are pinned too — a two-treatment
network reproduces `rma`, and a one-study network reproduces weighted least
squares exactly.

`multinma` 0.9.1 was fitted to the same fixture as an independent check and
its posterior means agree (0.07 and 0.16 against 0.0702 and 0.1491), but it is
not the same model, so its posterior SDs are not comparable: it places one
shared heterogeneity SD on every relative effect (`tau = 0.44` where this
fixture supports treatment-specific SDs of 0.31 and 0.34) and fits a residual
SD per study-arm rather than one `sigma2`. A differently parameterised
Bayesian fit cannot be a tolerance target, which is why the pinned numbers come
from `lme4`.

### 14. Dose-response network meta-analysis — **yes**

`ma.nma_dose` fits a dose function per treatment to a network of contrasts:
`y = f_b(x_b) - f_a(x_a) + e`. Six dose functions — `linear`, `quadratic`,
`loglinear`, `rcs`, `emax`, `exponential` — three heterogeneity structures (no
between-study variation, a random study level, a random **curve**), and a
league table at any pair of doses through `contrast()`.

The reason this is a network model rather than a pairwise one is the
consistency equation. Because the same parameter vector is used wherever a
treatment appears, `d_ab(x) + d_bc(x) - d_ac(x)` cancels identically at **any**
dose, including doses nobody studied. A model storing one parameter per
comparison would satisfy it only at observed doses. The tests assert the
identity at unseen doses and assert that the same functional has the same
Jacobian whichever route through the network assembles it.

Every curve is anchored at dose zero. That is an assumption, not a
convention: without it, adding a constant to every curve leaves every contrast
unchanged and the model is not identified.

Pinned against `metafor::rma.mv` for the four dose functions linear in their
parameters (coefficients to 1e-7, standard errors to 1e-6 relative, `tau2` to
5e-8), against `nls` with the PORT algorithm for `emax` and `exponential`
(parameters to 1e-4, residual sum of squares to 1e-8), and against `rms::rcs`
for the spline basis itself (1e-9).

Not attempted, and why: a random effect on `(emax, ed50)`. The marginal
likelihood is then an integral with no closed form, so it needs MCMC, which the
dependency policy rules out. `MBNMAdose` is the reference implementation that
does exactly that, and it could not be installed here because it needs `rjags`
and a JAGS binary, which requires an elevated install — so the validation went
to implementations that are deterministic instead.

### 15. Time-course network meta-analysis — **yes**

`ma.nma_timecourse` fits `y = g_b(t) - g_a(t) + e` with the same parameter
vector used wherever a treatment appears, so the network's consistency equation
cancels at **every** time, including times nobody measured. Four bases over
time (`constant`, `linear`, `quadratic`, `rcs`), four heterogeneity structures,
and a reference treatment whose trajectory is identically zero — necessary
because a common shift of every trajectory leaves every contrast unchanged.

The two things specific to time:

- **The baseline.** With a constant term, the difference at $t = 0$ is the
  ordinary NMA effect, so `basis="constant"` on one row per study *is*
  standard network meta-analysis. The tests pin that against `netmeta`: same
effects to 1e-8, same standard errors to 1e-8 relative, same `tau2` to 1e-8.
- **The correlation.** A study reporting the effect at 3, 6 and 12 months
  measured the same participants three times. `rho` gives a constant
  within-study correlation, `phi` the AR(1) decay `phi**|t_i - t_j|`. It is
  required rather than defaulted, and a study contributing more than one
  *treatment pair* is **refused** when `rho`/`phi` is given: two arms measured
  at the same time share the control, so their contrasts are correlated through
  it in a way that depends on the arm sizes, and no single number expresses
  that. Build the covariance and pass `vcov`.

Pinned against `metafor::rma.mv` with exactly that AR(1) covariance, for all
three linear bases: coefficients to 1e-7, standard errors to 1e-6 relative,
`tau2` to 2e-7.

Not attempted: random effects on a non-linear time parameter. Every basis here
is linear in its parameters, so a random trajectory is a linear mixed model and
needs no integration; a model with, say, a free asymptote or a free half-life
would need MCMC. `MBNMAtime` fits exactly that class, and it could not be
installed here for the same reason as `MBNMAdose`: `rjags` needs a JAGS binary
and an elevated install.

### Not on the list: IPD plus aggregate data — **yes**

`ipd_two_stage(..., aggregate=...)`; see item 3. It needed nothing beyond the
two-stage model, which is why it could be done as soon as that existed.

## Summary

| # | Area | Status | Guard test |
|---|---|---|---|
| 1 | non-linear meta-regression | yes | `tests/test_basis.py` |
| 2 | dose–response | yes | `tests/test_dose.py` |
| 3 | two-stage IPD | yes | `tests/test_ipd.py`, `tests/test_ipd_aggregate.py` |
| 4 | one-stage IPD | yes | `tests/test_onestage.py`, `tests/test_onestage_slopes.py` |
| 5 | diagnostic test accuracy | yes | `tests/test_dta.py` |
| 6 | robust variance estimation | yes | `tests/test_rve.py` |
| 7 | survival outcomes | yes | `tests/test_surv.py`, `tests/test_tvhr.py`, `tests/test_tvhr_joint.py` |
| 8 | prediction-model performance | yes | `tests/test_prediction.py` |
| 9 | multivariate | yes | `tests/test_mv.py`, `tests/test_mv_confint.py`, `tests/test_mv_region.py` |
| 10 | component NMA | yes | `tests/test_netmeta_api.py`, `tests/test_pimeta.py` |
| 11 | publication bias | yes | `tests/test_selmodel_models.py`, `tests/test_metabias.py`, `tests/test_limitmeta.py`, `tests/test_copas.py`, `tests/test_copas_region.py` |
| 12 | time-varying effects (dynamic heterogeneity) | partly — time-course from aggregate data is done; the state-space version is not | `tests/test_timecourse.py` |
| 13 | IPD network meta-analysis | yes | `tests/test_nma_design.py`, `tests/test_ipd_nma.py` |
| 14 | dose-response network meta-analysis | yes — every dose function except a random effect on a non-linear parameter | `tests/test_nma_dose.py` |
| 15 | time-course network meta-analysis | yes — every linear-in-parameters basis; a random effect on a non-linear time parameter is not available | `tests/test_nma_timecourse.py` |

Five of these turned out to be further along than the roadmap assumed: items
5, 6 and 10 are finished, item 1 needed only a basis rather than a regression
layer, and item 9 was missing only its reporting layer. All of items 1, 5 and 6
were completed in the current development cycle, which is why the planning
documents written before it overstate the work remaining.

Items 13–15 are the opposite case and are worth separating from the rest: they
are **new models**, not reporting layers, and each one needed its own
likelihood, its own parameterisation argument and its own reference. What made
them tractable is the shared parameterisation layer (`pimeta.nma.design`) and
the observation that all three are the same generalised least squares problem —
a per-treatment basis over a continuous index, a known within-study covariance
and a between-study covariance to profile out. The dose and time-course models
differ in their basis functions and their correlation assumptions, not in their
estimator; IPD-NMA differs only in taking individual rows instead of contrasts.
A fourth member of that family, a model with a random effect on a **non-linear**
basis parameter, is not available, and the reason is structural rather than a
missing feature: the marginal likelihood stops being closed-form.

## Dependency policy

`pimeta` depends on `numpy`, `scipy` and `matplotlib`, and nothing else. That
is a hard constraint, and it decides which items above are worth attempting:

- **Allowed** — anything that reduces to linear algebra plus an optimiser over a
  handful of variance parameters. Items 1, 2, 3, 8, 9 and the HR half of 7 are
  in this class.
- **Allowed but expensive** — a new likelihood for a named model family, as in
  items 4 and 5. Each one is written for its own family and validated against
  its own R implementation.
- **Allowed, and now present, as an opt-in** — Bayesian inference, which needs a
  sampler rather than a dependency. `pimeta.bayes` is written against numpy and
  adds nothing to the install; it is opt-in per call because it is slow, not
  because it is absent. That reverses the earlier position on this page, which
  treated a general MCMC sampler as out of scope; the reversal is deliberate and
  the engine is tested against exact posteriors and against its own failure
  modes before being used for anything.
- **Still not allowed** — a general-purpose mixed-model solver (the models here
  have known within-study covariances and a handful of variance parameters, so a
  general solver would be a dependency bought for nothing) and a Bayesian
  model-averaging engine as a black box: the model set has to be written out and
  checked, which is what the changelog calls RoBMA-not-yet rather than
  RoBMA-someday.
- **Not allowed** — a general-purpose mixed-model solver, a general MCMC
  sampler, or a Bayesian model-averaging engine. Those are the parts of items 4
  and 11 that a dependency-free package cannot do well, and a half-validated
  numerical solver is worse than an honest gap.

Where a new method cannot be validated against an R implementation, it is not
added. The reference implementations are listed on
[the differences page](../reference/differences.md).

## What this page changes about the roadmap

The original plan sequenced features. The audit says the shared primitive
(formula, design matrix, prediction with intervals, sampled covariance, mixed
model) is already present, so the order should be by *what is genuinely new
numerics*, not by topic:

| Stage | Work | Why here |
|---|---|---|
| 0 | this page | cheap, and it stops the next three stages from being built twice |
| 1 | `rcs` and fractional-polynomial bases, the non-linearity test, `predict_curve` | **done** — item 1 is finished; it unblocks items 2 and 3 |
| 2 | dose–response | item 1 plus the covariance construction; **done** — it needed the covariance, not a new estimator |
| 3 | IPD data layer and two-stage | **done** — it needed a first stage and a data layer, not a new estimator |
| 4 | one-stage IPD, continuous then binary | **done** — the only item that needed a genuinely new likelihood; random slopes added on top of it |
| 5 | KM reconstruction and RMST | **done** — data layer, strictly verifiable, feeds item 3 |
| 6 | prediction-model performance | **done** — four metrics, each on its own scale and variance, pooled by the existing engine |
| — | RoBMA, time-varying effects | declined, see the dependency policy |

> **中文**　本页是能力矩阵：逐项核对「已有 / 部分 / 无」，并写明证明它的测试。结论是原路线图高估了工作量 —— 公式解析、设计矩阵、带置信区间与预测区间的 `predict`、混合模型内核**都已经在库里**，非线性回归只差基函数与非线性检验；DTA 与稳健方差已经完成；`netcomb` 就是成分 NMA。真正的缺口是剂量–反应的协方差构造、IPD 数据层、单阶段 IPD 的逐个体似然、KM 重建、预测模型性能。依赖政策写死：不引入通用混合模型求解器、不引入 MCMC。
