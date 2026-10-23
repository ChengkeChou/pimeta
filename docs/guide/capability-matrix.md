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

### 7. Survival outcomes — **partial**

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
  obtained, and it feeds `ma.ipd_two_stage` directly.

Missing: time-dependent hazard ratios. Note that nothing here claims that
pooling log HRs handles non-proportional hazards — it does not, which is why
RMST is there.

### 8. Prediction-model performance — **no**

Wanted: meta-analysis of discrimination and calibration.

Nothing exists: no calibration slope or intercept, no observed-to-expected
ratio, no Brier score, no logit-AUC. The keyword search over the source returns
nothing.

These are not one effect size with four names — each has its own sampling
variance and its own transformation, and when two are reported by the same
study their estimates are correlated, which needs a covariance the study does
not print.

### 9. Multivariate meta-analysis — **partial**

Already there: the estimator is complete. `ma.rma_mv` implements every metafor
variance structure (CS, HCS, AR, TOEP, UN, GEN), random slopes, custom weight
matrices, and moderator formulas. `ma.matreg` regresses one outcome on others
between studies. `ma.vcalc` builds the sampling covariance. `ma.aggregate_escalc`
aggregates dependent effect sizes within a cluster. `ma.sigma_matreg` recovers
the coefficient covariance.

Missing, and small in each case: `vcalc` takes one `rho`, not one per comparison
pair; there is no helper that returns the between-study correlation matrix with
intervals; and there is no multivariate prediction region.

### 10. Component network meta-analysis — **yes**

`ma.netcomb` is the additive component model (Rücker et al. 2020): a
combination's effect is the sum of its components' effects, $\delta = B\theta$
with $\theta = C\beta$, estimated by weighted least squares, with the additive
$Q$ and the component-difference test. `ma.netcomplex` derives the effect of any
requested combination from the fitted components, and `ma.discomb` handles
disconnected components. All three are pinned against `netmeta` 3.7.0.

Missing: the component *interaction* (inconsistency) model, dose–response NMA,
time-course NMA, and IPD-NMA.

### 11. Publication bias — **partial**

Already there, and larger than "a funnel plot": `ma.selmodel` with eighteen
selection functions (beta, half-normal, negative-exponential, logistic, power,
their truncated variants, step functions, and the truncation family), plus
`ma.profile_selmodel` for the profile of a fitted selection model. The classic
tools are present too: `ma.trimfill` for trim-and-fill, `ma.fsn` for the
file-drawer, and `ma.regtest` and `ma.egger_test` for the regression test. All
of it is pinned against `metafor` and `meta`.

Missing: `limitmeta` (a different estimator, from `metasens`), the Copas
selection model, and any Bayesian model averaging.

### 12. Time-varying effects — **no**

Nothing. No time index, no dynamic heterogeneity, no Gaussian process. This is
a research direction rather than a missing feature.

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
| 7 | survival outcomes | partial | `tests/test_surv.py` |
| 8 | prediction-model performance | no | — |
| 9 | multivariate | partial | `tests/test_rma_mv_methods.py`, `tests/test_matreg_r_parity.py` |
| 10 | component NMA | yes | `tests/test_netmeta_api.py`, `tests/test_pimeta.py` |
| 11 | publication bias | partial | `tests/test_selmodel_models.py`, `tests/test_metabias.py` |
| 12 | time-varying effects | no | — |

Five of these twelve turned out to be further along than the roadmap assumed:
items 5, 6 and 10 are finished, item 1 needed only a basis rather than a
regression layer, and item 9 is missing only its convenience layer. All of
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
| 6 | prediction-model performance | four effect sizes and their covariances; different audience, can run in parallel |
| — | RoBMA, time-varying effects | declined, see the dependency policy |

> **中文**　本页是能力矩阵：逐项核对「已有 / 部分 / 无」，并写明证明它的测试。结论是原路线图高估了工作量 —— 公式解析、设计矩阵、带置信区间与预测区间的 `predict`、混合模型内核**都已经在库里**，非线性回归只差基函数与非线性检验；DTA 与稳健方差已经完成；`netcomb` 就是成分 NMA。真正的缺口是剂量–反应的协方差构造、IPD 数据层、单阶段 IPD 的逐个体似然、KM 重建、预测模型性能。依赖政策写死：不引入通用混合模型求解器、不引入 MCMC。
