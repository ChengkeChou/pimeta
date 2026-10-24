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

Studies that cannot be fitted are not silently dropped: a study with no events,
or one where a covariate separates the outcome, is named in `result.excluded`
with the reason.

Pinned by `tests/test_ipd.py` against `glm(family = binomial)` and
`metafor::rma`.

Missing: the one-stage model (item 4), and IPD plus aggregate data.

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

Deliberately not implemented: study-specific residual variances, random
treatment slopes, and non-linear covariate terms. A random intercept with fixed
effects is the model the Cochrane guidance describes as the standard one-stage
analysis; the rest are extensions with their own identification problems, and a
package that has no dependency to lean on should not claim them.

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

Two limitations, stated rather than papered over. The cut points are not
chosen by the data — that is model selection with its own multiplicity, the
same problem as choosing the number of spline knots. And pooling each interval
separately ignores the correlation between a study's interval-specific
estimates (they come from the same participants): each interval's pooled
estimate is valid, but *comparing* intervals needs a model that carries that
correlation, so no trend test is reported.

Pinned by `tests/test_tvhr.py` against `glm(family = poisson)` in R and then
`metafor::rma` per interval. One detail worth knowing: R's default
`epsilon = 1e-8` leaves the GLM standard errors off by 1.4e-5, so the
reference is generated with `epsilon = 1e-14`; the difference looks like an
implementation gap and is a convergence tolerance.

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

Missing: the within-study covariance between an AUC and a calibration slope
reported by the same study, which a joint model would need. Pooling them
separately avoids the problem rather than solving it.

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

Still not offered: a joint **prediction** region. A joint confidence ellipsoid
for the pooled effects is standard (Jackson, Riley and White 2014), but a
region for a *new* study is a different object and no implementation provides
one to check against.

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

Missing: the component *interaction* (inconsistency) model, dose–response NMA,
time-course NMA, and IPD-NMA.

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
and `metafor` in `tests/test_tvhr.py`.

What is absent is a time-varying **effect size** — dynamic heterogeneity, where
the pooled estimate and $\tau^2$ are functions of a continuous time index, and
the usual candidates are a Gaussian process or a state-space model on the
coefficients. There is no reference implementation in R to check one against,
and this package does not ship models it cannot verify. It is a research
direction rather than a missing feature.

### Not on the list: IPD plus aggregate data — **no**

Depends on items 3 and 4. Doing it before them would mean building the hybrid
model twice.

## Summary

| # | Area | Status | Guard test |
|---|---|---|---|
| 1 | non-linear meta-regression | yes | `tests/test_basis.py` |
| 2 | dose–response | yes | `tests/test_dose.py` |
| 3 | two-stage IPD | yes | `tests/test_ipd.py` |
| 4 | one-stage IPD | yes | `tests/test_onestage.py` |
| 5 | diagnostic test accuracy | yes | `tests/test_dta.py` |
| 6 | robust variance estimation | yes | `tests/test_rve.py` |
| 7 | survival outcomes | yes | `tests/test_surv.py`, `tests/test_tvhr.py` |
| 8 | prediction-model performance | yes | `tests/test_prediction.py` |
| 9 | multivariate | yes | `tests/test_mv.py` |
| 10 | component NMA | yes | `tests/test_netmeta_api.py`, `tests/test_pimeta.py` |
| 11 | publication bias | yes | `tests/test_selmodel_models.py`, `tests/test_metabias.py`, `tests/test_limitmeta.py`, `tests/test_copas.py`, `tests/test_copas_region.py` |
| 12 | time-varying effects (dynamic heterogeneity) | no | — |

Five of these twelve turned out to be further along than the roadmap assumed:
items 5, 6 and 10 are finished, item 1 needed only a basis rather than a
regression layer, and item 9 was missing only its reporting layer. All of
items 1, 5 and 6 were completed in the current development cycle, which is why
the planning documents written before it overstate the work remaining.

## Dependency policy

`pimeta` depends on `numpy`, `scipy` and `matplotlib`, and nothing else. That
is a hard constraint, and it decides which items above are worth attempting:

- **Allowed** — anything that reduces to linear algebra plus an optimiser over a
  handful of variance parameters. Items 1, 2, 3, 8, 9 and the HR half of 7 are
  in this class.
- **Allowed but expensive** — a new likelihood for a named model family, as in
  items 4 and 5. Each one is written for its own family and validated against
  its own R implementation.
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
| 4 | one-stage IPD, continuous then binary | **done** — the only item that needed a genuinely new likelihood |
| 5 | KM reconstruction and RMST | **done** — data layer, strictly verifiable, feeds item 3 |
| 6 | prediction-model performance | **done** — four metrics, each on its own scale and variance, pooled by the existing engine |
| — | RoBMA, time-varying effects | declined, see the dependency policy |

> **中文**　本页是能力矩阵：逐项核对「已有 / 部分 / 无」，并写明证明它的测试。结论是原路线图高估了工作量 —— 公式解析、设计矩阵、带置信区间与预测区间的 `predict`、混合模型内核**都已经在库里**，非线性回归只差基函数与非线性检验；DTA 与稳健方差已经完成；`netcomb` 就是成分 NMA。真正的缺口是剂量–反应的协方差构造、IPD 数据层、单阶段 IPD 的逐个体似然、KM 重建、预测模型性能。依赖政策写死：不引入通用混合模型求解器、不引入 MCMC。
