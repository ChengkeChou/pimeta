# Pooling

`rma` is the workhorse — a port of `metafor::rma()`, returning the same `$`
fields as the R object.

```python
import pimeta as ma

yi = [0.20, 0.40, 0.10, 0.50]
vi = [0.04, 0.05, 0.03, 0.06]

r = ma.rma(yi, vi, method="REML")

r.b[0]        # pooled estimate (analysis scale)
r.se[0]       # standard error
r.ci_lb[0], r.ci_ub[0]
r.tau2        # between-study variance
r.I2, r.H2    # heterogeneity as percentages
r.QE, r.QEp   # Cochran's Q, p-value
```

## Common vs random effects

```python
import pimeta as ma

fixed = ma.rma(yi, vi, method="FE")        # common-effect
random = ma.rma(yi, vi, method="REML")     # random-effects
```

`method` names the τ² estimator. The default is `"REML"`, matching `metafor`.

| Estimator | Notes |
|---|---|
| `DL` | DerSimonian–Laird; the `meta` package default |
| `HE`, `HS`, `SJ` | Hedges, Hunter–Schmidt, Sidik–Jonkman |
| `ML`, `REML` | maximum likelihood; `metafor`'s default |
| `PM`, `EB` | Paule–Mandel, empirical Bayes |
| `GENQ`, `GENQM`, `QGENQ`, `QGENQM` | generalised Q with a specified `tau2` target |

## Standard errors

```python
import pimeta as ma

# Hartung-Knapp (recommended when k is small).
r_hk = ma.rma(yi, vi, method="REML", test="knha")
print(r_hk.se[0], r_hk.ci_lb[0], r_hk.ci_ub[0])

# Ad hoc Knapp-Hartung variants.
r_adhoc = ma.rma(yi, vi, test="adhoc")
```

With `test="knha"` the statistic switches to a *t* with `k - p` degrees of
freedom, exactly as in `metafor`.

## Confidence interval for τ²

`confint(r)` returns a dict — R's `confint.rma()` gives the τ² interval, not the
coefficient interval:

```python
import pimeta as ma

ci = ma.confint(r, level=0.95, type="QP")
print(ci["tau2"], ci["ci_lb"], ci["ci_ub"])

# Profile-likelihood instead of Q-profile.
ci_pl = ma.confint(r, type="PL")

# Exact (t-distribution based) interval.
ci_exact = ma.confint(r, type="HT")
```

## Prediction intervals

```python
import pimeta as ma

pi = ma.predict(r, level=0.95)
print(pi.pred, pi.pi_lb, pi.pi_ub)

# For new moderator values, in a meta-regression.
pi_new = ma.predict(r, newmods=[[4.0]])
```

**The default interval is `predtype="t"`, not metafor's plug-in, and the size of
the difference has been measured.** The plug-in interval is
`est ± z * sqrt(se² + tau2)`, which treats the estimated between-study variance
as if it were known; it is what `metafor::predict` reports, and it covers less
than it claims. Over 2000 replications per scenario, checking against the true
effect of a new study drawn from `N(mu, tau²)`:

| scenario | k | metafor's interval (`predtype="default"`) | `predtype="t"` (our default) | parametric bootstrap |
|---|---|---|---|---|
| `null` | 20 | 0.8815 ± 0.0072 | 0.8990 ± 0.0067 | — |
| `tiny` | 5 | 0.8040 ± 0.0089 | 0.8935 ± 0.0069 | — |
| `unequal` | 20 | 0.8825 ± 0.0072 | 0.9030 ± 0.0066 | — |

At 500 replications including a bootstrap interval built from the sampling
distribution of `tau2`:

| scenario | k | metafor's interval | `t` | bootstrap |
|---|---|---|---|---|
| `null` | 20 | 0.900 | 0.910 | 0.910 |
| `tiny` | 5 | 0.830 | 0.904 | 0.860 |
| `unequal` | 20 | 0.912 | 0.918 | 0.900 |

Three conclusions follow, and the third one changed the default.

A `t` quantile on `k - p` degrees of freedom is never worse than the plug-in
interval and is materially better at `k = 5` (0.904 against 0.830). A parametric
bootstrap, which is the only one of the three that reflects the sampling
distribution of `tau2` directly, does **not** beat `t` at `k = 5` (0.860), so it
does not earn its cost as a default; it is kept in `tools/coverage_pi.py` as the
measurement rather than shipped as a code path. Neither reference reaches 0.95,
and the residual is the uncertainty in `tau2` that no plug-in variance includes.

**Which is why the default is `t`.** An interval that misses its nominal level by
12 points at `k = 5` is not a defensible default, and "the reference does it this
way" is a statement about the reference. `predtype="default"` still reproduces
`metafor::predict` exactly — that is what the parity tests use — but it is now
something you ask for rather than something you get.

```python
# What R does, when reproducing R is the point.
pi = ma.predict(r, level=0.95, predtype="default")
```

Regenerate the table with `python tools/coverage_pi.py --reps 500`; the
`null`, `tiny` and `unequal` baselines carry the same measurement under
`coverage.pi` and `coverage.pi_t`, each with its own Monte Carlo error.

## Profiling τ²

```python
import pimeta as ma

prof = ma.profile_rma(r, n_points=40)
print(prof.tau2, prof.ll)

ax = ma.plot(prof)     # draws the log-likelihood curve
```

## By-hand pooling

If you already have analysis-scale `(y, v)` arrays and want the same numbers
without the R-shaped result object:

```python
import pimeta as ma

res = ma.pool_yv(y, v, None, effect_type="MD", model="random", method="REML")
print(res.pooled_effect, res.pooled_se, res.q, res.i2)
```

## Fixed-effect-only models

`rma.mh` (Mantel–Haenszel) and `rma.peto` are fixed-effect-only, mirror their
metafor counterparts, and support weights, cumul, leave1out, rstandard,
rstudent, confint and qqnorm:

```python
import pimeta as ma

mh = ma.rma_mh(ai=ai, bi=bi, ci=ci, di=di, measure="RR")
print(mh.b[0], mh.ci_lb[0], mh.ci_ub[0])

peto = ma.rma_peto(ai=ai, bi=bi, ci=ci, di=di)
print(peto.b[0])
```

## Which output you get

| Call | Result class | Mirrors |
|---|---|---|
| `rma(...)` | `RmaResult` | `rma.uni` |
| `rma(..., scale=~ z)` | `RmaLsResult` | location-scale (`rma.ls`) |
| `rma_mv(...)` | `RmaMvResult` | `rma.mv` |
| `rma_glmm(...)` | `RmaGlmmResult` | `rma.glmm` |
| `metagen(...)` | `PooledResult` | `meta::metagen` |
| `metabin`, `metacont`, … | `MetabResult`, `MetacontResult`, … | `meta::metabin` etc. |

All of them expose `to_dict()`, `tidy()`, `glance()` and a Jupyter HTML table.
