# Diagnostic test accuracy: bivariate model and HSROC

A diagnostic meta-analysis does not pool one effect size. Each study reports
two numbers — how often the test is positive in diseased people (sensitivity)
and in healthy people (false positive rate) — and those two numbers are
correlated across studies, because a study that uses a lower threshold gains
sensitivity at the cost of specificity. Pooling them separately throws that
away.

`pimeta.dta` fits them jointly.

## Preparing the data

```python
import pimeta as ma

data = ma.escalc_dta(
    tp=[47, 126, 19, 36],
    fn=[9, 51, 10, 3],
    fp=[101, 272, 12, 78],
    tn=[738, 1543, 192, 276],
    slab=["Study 1", "Study 2", "Study 3", "Study 4"],
)
```

`escalc_dta` returns a `DtaData` holding the four cells, sensitivity, FPR and
the variance of each on the logit scale ($1/TP + 1/FN$ and $1/FP + 1/TN$).

### The continuity correction has a surprising default

A zero cell gives an infinite logit variance. `mada`'s default,
`correction_control="all"`, adds the correction to **every** study as soon as
**any** table in the dataset has a zero cell — not only to the studies that
need it. That changes the estimates, so `pimeta` reproduces it:

```python
ma.escalc_dta(tp, fn, fp, tn)                              # "all" (mada's default)
ma.escalc_dta(tp, fn, fp, tn, correction_control="single")  # only the broken ones
ma.escalc_dta(tp, fn, fp, tn, correction_control="none")    # never
```

## The bivariate model

```python
fit = ma.reitsma(data)                 # REML, mada's default
print(fit.mu)                          # mean logit sensitivity, mean logit FPR
print(fit.Psi)                         # 2x2 study-level covariance
print(fit.summary_sens, fit.summary_fpr, fit.summary_spec)
print(fit.AIC)
```

The summary point is the back-transformed mean pair, and its interval comes
from the fixed-effects covariance:

```python
print(fit.sens_ci, fit.fpr_ci)
```

Use `method="ml"` for maximum likelihood. Other estimation methods that
`mada` offers (`"vc"`, `"mm"`, `"fixed"`) are not implemented; REML and ML are
the two that are used in practice.

## HSROC

The HSROC model (Rutter & Gatsonis) is a different parameterisation of the
same likelihood: a threshold random effect $\theta_i$, an accuracy random
effect $\alpha_i$ and a fixed asymmetry $\beta$.

```python
h = ma.hsroc(fit)
print(h.Theta, h.Lambda, h.beta)
print(h.sigma2theta, h.sigma2alpha)
print(h.dor)                  # exp(Lambda)
```

`hsroc` does not re-fit. Harbord et al. showed the two models are equivalent,
so `hsroc` applies the exact algebraic transform — which means both report the
same log-likelihood and the same AIC, as they must.

| | bivariate | HSROC |
|---|---|---|
| Mean structure | `mu = (mu_sens, mu_fpr)` | `Theta`, `Lambda` |
| Heterogeneity | `tau_sens`, `tau_fpr`, `rho` | `sigma_theta`, `sigma_alpha` |
| Asymmetry | carried by `rho` | `beta` |
| Parameters | 5 | 5 |

`beta = 0` means a symmetric SROC and `text{DOR} = e^{\Lambda}`.

## SROC and AUC

```python
curve = ma.sroc(fit)                   # fpr 0.01 ... 0.99
print(curve.fpr[:5], curve.tpr[:5])

print(ma.auc(fit))                     # {"auc": ..., "pauc": ...}
```

The default SROC is the Rutter–Gatsonis curve through the fitted parameters,

$$\text{logit}(\text{TPR}) = \Lambda e^{-\beta/2} + e^{-\beta}\,\text{logit}(\text{FPR}),$$

drawn on a 99-point grid. The partial AUC restricts the grid to the FPR range
the studies actually cover, which is the number worth quoting when every study
sits at one end of the range.

`ma.auc` reproduces `mada::AUC`'s numerical convention exactly, including the
detail that the trapezoid sum is divided by the number of grid points rather
than by the number of intervals.

## Using a fit in a record

Both result types carry a `to_dict()`, so they drop straight into
`pimeta.record`:

```python
from pimeta.record import save_record

save_record("dta.json", fit, data=data.to_dict(), function="reitsma",
            call={"method": "reml"})
```

## Parity with `mada`

The numbers are checked against `mada` 0.5.12 on its own `AuditC` (14 studies)
and `Dementia` (33 studies) datasets, under both REML and ML:
coefficients, the study-level covariance, the Jacobian-corrected
log-likelihood, AIC/BIC, the HSROC parameters, the summary point and both
AUCs. `tests/test_dta.py` also evaluates the likelihood at `mada`'s reported
variance components and asserts it matches to $10^{-6}$, which is the strong
statement: the two implementations are optimising the same function, and the
small parameter differences are `mvmeta`'s stopping tolerance.

## See also

- [Multilevel models](multilevel.md) — `rma.mv`, which underlies the bivariate fit
- [Effect sizes](effect-sizes.md) — `escalc` for the non-diagnostic measures
- [Reproducible analyses](reproducibility.md) — recording a fit
