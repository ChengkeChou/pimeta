# Sensitivity analyses

Sensitivity analysis asks one question: **what happens to this estimate if one
assumption changes?** It is not a multiverse and it does not average anything.
Each row of the table is the same estimand under one variant, and the point is
to see how far the answer travels when a choice that the data did not force is
changed.

```python
import numpy as np
import pimeta as ma

rng = np.random.default_rng(3)
yi = rng.normal(0.3, 0.25, 12)
vi = rng.uniform(0.02, 0.09, 12)
fit = ma.rma(yi, vi=vi)
res = ma.sensitivity_table(fit)
print(res)
```

```text
Sensitivity of the pooled estimate (k = 12, p = 1, 95% intervals)

  baseline                          +0.2611  [+0.0612, +0.4609]

tau2: 8 variant(s), estimate from +0.2404 to +0.2636
  tau2 = DL                         +0.2605  [+0.0650, +0.4560]
  tau2 = HE                         +0.2633  [+0.0418, +0.4847]
  ...
  tau2 = FE                         +0.2404  [+0.1166, +0.3641]
```

The exact numbers are not the point — they come from the seeded data above so
that the example can be reproduced character for character. Read the `estimate
from ... to ...` line of each family.

## The three families

| family | what changes | rows |
|---|---|---|
| `tau2` | the heterogeneity estimator | one per estimator in `METHODS`, excluding the one the fit used |
| `leave1out` | which study is included | one per study, with its Cook's distance and hat value |
| `bias` | the publication-bias assumption | limit meta-analysis at three extrapolation points, a step selection model, the RoBMA posterior mean |

`MP` is absent from `METHODS` because `metafor` treats it as an alias of `PM`;
a table that lists the same estimator twice looks more thorough than it is. The
`EB` row is present but says in its `reason` that empirical Bayes converges to
`PM` by construction, so that row is a convergence check rather than an
alternative.

The `tau2` family always includes `FE`. "What if there were no heterogeneity at
all" is a different model, but it is also the assumption a reader will silently
make, and showing the fixed-effect interval next to the random-effects one is
the cheapest way to stop that.

## Refusals are part of the answer

A variant that is statistically invalid for a given fit is refused **with a
reason**, in `res.refused`, rather than attempted and dropped:

```python
res = ma.sensitivity_table(fit, over=("bias",))
res.refused
# [{'family': 'bias',
#   'reason': 'every publication-bias adjustment here is a property of a pooled '
#             'estimate; this fit has 3 coefficients, so there is no single '
#             'estimate for it to adjust'}]
```

A table that quietly omits the rows it could not compute is worse than no table,
because the reader cannot tell an omission from a null result. Asking only for a
refused family returns that refusal and an empty row list — the reason the
variant is invalid *is* the answer, so it is not turned into an exception the
caller has to guess at.

`"rho"` is accepted in `over` only to be refused: a univariate fit has one
outcome and therefore no within-study correlation. Use
`rho_sensitivity()` on a multivariate fit, where the assumption exists and can be
varied.

## Reading the table

Two things are worth looking at, and neither is the point estimate:

- **The range, not the width.** `res.range("tau2")` gives the smallest and
  largest estimate in a family. If the choice of heterogeneity estimator moves
  the answer as much as the confidence interval is wide, the estimate is not
  supporting any conclusion sharper than "positive".
- **Whether the movement is explained.** A `leave1out` row that moves the
  estimate carries a `cook_d` and `influence` flag in `detail`. An estimate that
  moves because of one study is a different finding from one that moves because
  of the whole body of evidence, and the table separates them.

## What this is not

- **It is not multiple imputation.** Under MAR an imputation procedure is only
  as credible as its imputation model, and verifying it needs a coverage study
  under a *known* missingness mechanism. That is its own project, not a
  `method="mi"` argument.
- **It is not a specification curve.** That averages and orders across analyses,
  which is the multiverse ([roadmap](roadmap.md)) and a later stage.
- **It does not weight the variants.** No probability is attached to a variant,
  because none has been earned: the prior over "which estimator is right" is not
  a quantity the data supplies.

## See also

- [Publication bias](publication-bias.md) — the estimators behind the `bias` rows.
- [Heterogeneity](heterogeneity.md) — what `tau2` means and why the estimator choice matters.
- [Pooling](pooling.md) — where the baseline interval comes from.
