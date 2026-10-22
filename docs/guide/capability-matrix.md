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

### 1. Non-linear meta-regression — **partial**

Wanted: restricted cubic splines, fractional polynomials, a test for
non-linearity, and a predicted curve.

Already there: meta-regression is complete, including the part people assume is
missing. There is a formula parser and a design-matrix builder in
`pimeta.data_compat`: `parse_formula` splits a string into terms,
`design_matrix` expands them into columns, and `resolve_mods` is the entry
point `ma.rma` uses. So `ma.rma(yi, vi, mods="~ dose + year", data=df)` and
interactions (`a*b`, `a:b`), dummy coding, `0 +` and `.` expansion all work, and
`pimeta.record` captures the call. `predict(newmods=...)` already returns a
point estimate with a confidence *and* a prediction interval per row, so a
curve plus bands needs only a basis matrix, not new prediction code.

Missing: the basis itself. There is no `rcs()` and no fractional-polynomial
expansion, and the formula parser splits on `+` and `:` without understanding
function calls, so `~ rcs(dose, 4)` cannot be written. Also missing: the
non-linearity test (Wald over the whole block, and the linear-versus-non-linear
partition), and a `predict_curve()` convenience wrapper.

Estimated cost: small. The expensive half of this feature is already in the
tree.

### 2. Dose–response meta-analysis — **no**

Wanted: the non-linear dose–response curve with the within-study covariance
handled.

Already there: the two pieces it would sit on — `ma.vcalc` builds a
block-diagonal sampling covariance matrix from an assumed within-study
correlation, and `ma.rma_mv` fits it with every metafor variance structure.

Missing: the Greenland–Longnecker covariance for contrasts that share a
control group (only a single `rho` is supported, not one per comparison pair),
the reference-dose convention, the dose grid, and the spline design over dose.

### 3. Two-stage IPD — **no**

Wanted: fit a model inside each study, then pool the coefficient vectors.

Already there: everything downstream of the per-study fit. A study-level
$(\hat\beta_i, \hat V_i)$ is just a multivariate effect size, and `ma.rma_mv`
with `struct="UN"` pools correlated coefficients correctly.

Missing: a public per-study GLM (there is an internal PIRLS binomial fitter
inside `rma_glmm`, but it is not exported and is shaped for two-arm aggregate
data), an IPD data layer with the checks that matter (which studies actually
provide a randomised contrast), and the assembly that turns study fits into one
stacked model.

### 4. One-stage IPD — **no**

Wanted: participant-level data in one model, with participants nested in
studies.

Already there: `ma.rma_mv` is a mixed model in which the sampling variances are
*known*. A continuous one-stage IPD model differs from it by exactly one free
parameter — the residual scale — so the optimiser and the variance-component
machinery carry over.

Missing: that unknown scale, and any participant-level likelihood. A binary or
time-to-event one-stage model needs the likelihood written per participant
rather than per study-arm, which is new numerical code, not a re-parameterisation.
This is the most expensive item on the list and the one where writing a general
mixed-model solver would be a mistake.

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

There is no SROC plotting helper yet, which is a small genuine gap: the curve
exists, drawing it is three lines of matplotlib.

### 6. Robust variance estimation — **yes**

`ma.robust` serves both `ma.rma` and `ma.rma_mv` fits through one sandwich
implementation: CR0, CR1, CR1.S1, CR1.S2, and the clubSandwich CR2/CR3 path
with Satterthwaite degrees of freedom and a robust omnibus test.

Pinned by `tests/test_rve.py` against `clubSandwich` 0.7.0.

### 7. Survival outcomes — **partial**

Already there: a pooled log hazard ratio is an ordinary effect size. `ma.metagen`
treats `"HR"` as a relative effect (for log transformation and plotting), and
`ma.rd` converts a pooled HR into an absolute risk difference.

Missing: everything that needs more than one number per study. There is no
Kaplan–Meier reconstruction, no restricted mean survival time, and no
time-dependent hazard ratio. Note that nothing here claims that pooling log HRs
handles non-proportional hazards — it does not.

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
| 1 | non-linear meta-regression | partial | `tests/test_data_compat.py` (formulas), `tests/test_coverage_gaps.py` (`predict`) |
| 2 | dose–response | no | — |
| 3 | two-stage IPD | no | — |
| 4 | one-stage IPD | no | — |
| 5 | diagnostic test accuracy | yes | `tests/test_dta.py` |
| 6 | robust variance estimation | yes | `tests/test_rve.py` |
| 7 | survival outcomes | partial | — |
| 8 | prediction-model performance | no | — |
| 9 | multivariate | partial | `tests/test_rma_mv_methods.py`, `tests/test_matreg_r_parity.py` |
| 10 | component NMA | yes | `tests/test_netmeta_api.py`, `tests/test_pimeta.py` |
| 11 | publication bias | partial | `tests/test_selmodel_models.py`, `tests/test_metabias.py` |
| 12 | time-varying effects | no | — |

Five of these twelve turned out to be further along than the roadmap assumed:
items 5 and 6 are finished, item 10 was already finished, item 9 is missing only
its convenience layer, and item 1 is missing only a basis matrix. Items 5 and 6
were completed in the current development cycle, which is why the planning
documents written before it overstate the work remaining.

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
| 1 | `rcs` and fractional-polynomial bases, the non-linearity test, `predict_curve` | item 1 is 80% done; this is the remaining 20% and it unblocks items 2 and 3 |
| 2 | dose–response | item 1 plus one covariance construction; same readers as item 3 |
| 3 | IPD data layer and two-stage | reuses stage 1 for the non-linear case, nothing else new |
| 4 | one-stage IPD, continuous then binary | first genuinely new likelihood |
| 5 | KM reconstruction and RMST | data layer, strictly verifiable, feeds item 3 |
| 6 | prediction-model performance | four effect sizes and their covariances; different audience, can run in parallel |
| — | RoBMA, time-varying effects | declined, see the dependency policy |

> **中文**　本页是能力矩阵：逐项核对「已有 / 部分 / 无」，并写明证明它的测试。结论是原路线图高估了工作量 —— 公式解析、设计矩阵、带置信区间与预测区间的 `predict`、混合模型内核**都已经在库里**，非线性回归只差基函数与非线性检验；DTA 与稳健方差已经完成；`netcomb` 就是成分 NMA。真正的缺口是剂量–反应的协方差构造、IPD 数据层、单阶段 IPD 的逐个体似然、KM 重建、预测模型性能。依赖政策写死：不引入通用混合模型求解器、不引入 MCMC。
