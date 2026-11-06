# Validation evidence

Every entry in the catalogue answers "how far is this trusted?" with a
`validation_status` and a list of the numbers behind it. This page is where those
numbers are written down **inside this repository**, so that `ma.describe()` can
always be checked against a file that ships with the package.

The full studies — data-generating mechanisms, replicate counts, Monte Carlo
errors, and the corrections that were measured and rejected — live in the
`metabench` measurement harness and are cited next to each table below.

## The status vocabulary

| status | meaning |
|---|---|
| `validated` | measured here and the claim held |
| `conditional` | measured, and it holds within the stated conditions |
| `measured-shortfall` | measured, and it does **not** meet the nominal claim; the numbers are quoted |
| `parity` | agrees with a reference implementation, which says nothing about calibration |
| `unvalidated` | no measurement exists in this repository |

## Coverage, where the nominal level is 0.95

Replicates are per cell. "MC" is the Monte Carlo half-width, 1.96 times the
standard error of the proportion.

| quantity | cells | measured coverage | MC | verdict |
|---|---|---|---|---|
| `rma` confidence interval | k = 5 … 100 | 0.885–0.949 | ±0.011–0.018 | mostly calibrated; the two k = 5 cells are short |
| `predict_rma` prediction interval | k = 5, 10, 20, 50, 100 | 0.797, 0.836, 0.888, 0.919, 0.931 | ±0.011–0.018 | **measured-shortfall** |
| `predict_rma(predtype="t")` | the same cells | 0.893, 0.867, 0.900, 0.922, 0.934 | ±0.011–0.014 | better everywhere, still short below k = 20 |
| `predict_region(test="z")` | k = 5, 10, 20, 50 | 0.670, 0.690, 0.768, 0.870 | ±0.020–0.038 | **measured-shortfall** |
| `predict_region(test="t")` | the same cells | 0.825, 0.741, 0.793, 0.877 | ±0.017–0.037 | **measured-shortfall** |
| `predict_region(test="hk")` | the same cells | 0.972, 0.863, 0.846, 0.893 | ±0.007–0.035 | **measured-shortfall**; covers at k = 5 by being 4.9x the volume |
| `predict_region(test="bootstrap")` | k = 5, 10, 20 | 0.810, 0.880, 0.860 | ±0.064–0.077 | **measured-shortfall** |
| `predict_region`, region for the *mean* | every cell above | 0.96–1.00 | — | calibrated, which localises the defect to the prediction step |

Two prediction steps fail for the same reason, measured rather than assumed: the
width of both collapses when the heterogeneity estimate collapses, and the
distance it has to cover does not. For the univariate interval the correlation
between the half-width and `tau^2_hat` is 0.97, while the correlation between the
half-width and the error it must cover is 0.03.

## Corrections that were measured

| correction | where it applies | measured result | price |
|---|---|---|---|
| `crit_scale="measured"` (a multiplier on the `hk` critical value) | `predict_region` | 0.947–0.956 at k = 10–20 on a seed that was not used to calibrate it | 3.2–4.1x the chi-square volume |
| `tau^2_hat + 0.2(tau^2_ub − tau^2_hat)` | univariate prediction interval | 0.947 at k = 10 off-sample, 0.938 at k = 20, 0.977 at k = 5 | 1.2–2.2x the width |
| `test="bootstrap"` | `predict_region` | 0.810–0.880: an improvement that does not reach nominal | 3.7–4.6x the volume |

A fixed scale factor was measured and **rejected**: even at twice the width, the
univariate interval reaches only 0.925 at k = 5. The multiplier is not a function
of `k` either — it is 2.11 at k = 10 with two outcomes, 1.59 at the same k with
`rho = 0.7`, and 1.53 with three outcomes — which is why
`crit_scale="measured"` refuses a `J` it was not calibrated for instead of
interpolating into one.

## Where the full studies are

| family | report | contents |
|---|---|---|
| B, joint region, screening | `metabench/reports/B_joint_region.md` | 6 cells x 3 references, boundary split |
| B, joint region, formal | `metabench/reports/B2_joint_region_formal.md` | 2000 replicates, the bootstrap, the scaled reference validated off-sample, J = 3 and J = 5, the effective-df prototype |
| C, prediction interval | `metabench/reports/C_prediction_interval.md` | the k sweep, the mechanism, the rules measured and rejected |
| D, covariance | `metabench/reports/D_covariance.md` | multiple starts, boundary and flat-ridge rates, the 1.1.7 optimisation defect |

Each report states its seed, its replicate count and its Monte Carlo error, and
each can be reproduced from the scripts named in it. The harness is a separate
repository on purpose: a measurement that the measured code can edit is not a
measurement.

## Parity and reference checks

Correctness against other implementations is reported separately, because
agreeing with `metafor` says nothing about whether the interval covers:

- per-value parity against R-generated reference values (17 significant digits)
  lives in `tests/data/` and is asserted in `tests/`;
- an independently written reference likelihood lives in `metabench/reference/`;
- `tests/test_coverage_reproduction.py` re-runs a scaled-down version of the
  coverage studies inside this repository, so a regression in a shared kernel is
  caught even when the full harness is not checked out.
