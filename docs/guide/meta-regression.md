# Meta-regression

Moderators go through `mods`. `rma` prepends the intercept for you, so pass only
the moderator columns — exactly like R's `mods = ~ x`.

```python
import pimeta as ma

yi = [0.20, 0.40, 0.10, 0.50, 0.30, 0.15]
vi = [0.04, 0.05, 0.03, 0.06, 0.02, 0.05]
dose = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]

r = ma.rma(yi, vi, mods=dose, method="REML")

print(r.b)                     # [intercept, slope]
print(r.se, r.ci_lb, r.ci_ub)
print(r.QM, r.QMdf, r.QMp)     # omnibus test of the moderators
print(r.R2)                    # heterogeneity explained
```

## With a formula and a data frame

```python
import pimeta as ma

df = {
    "yi": yi, "vi": vi, "dose": dose,
    "year": [1990, 1995, 2000, 2005, 2010, 2015],
    "arm": ["A", "B", "A", "B", "A", "B"],
}

r = ma.rma("yi", "vi", mods="~ dose + year + arm", data=df)
print(r.b)
```

Formulas are expanded the way R's `model.matrix` does it:

| Formula | Columns |
|---|---|
| `~ dose + year` | `dose`, `year` |
| `~ dose*year` | `dose`, `year`, `dose:year` |
| `~ dose:year` | the interaction only |
| `~ 0 + dose` | no intercept (and `~ dose - 1` is the same) |
| `~ .` | every other column of `data` |

A string column becomes treatment-contrast dummies named `arm[B]` with the first
level dropped, so coefficients line up with a `metafor` fit.

## Testing a subset of coefficients

`btt` uses **1-based** indices, as in R. The intercept is coefficient 1.

```python
import pimeta as ma

r = ma.rma(yi, vi, mods=[dose, year], btt=[2])   # slope of dose only
print(r.QM, r.QMp)
```

## Predictions

```python
import pimeta as ma

r = ma.rma(yi, vi, mods=dose)

pred = ma.predict(r, newmods=[[7.0]])
print(pred[0]["pred"], pred[0]["se"], pred[0]["ci.lb"], pred[0]["ci.ub"])
```

`newmods` must supply **only the moderator columns** (no intercept column).
For a prediction interval on the pooled effect rather than on a new moderator
value, use `ma.predict_rma(r)`, which returns `pred`, `se`, `pi_lb` and `pi_ub`.

## Plots

```python
import pimeta as ma

ma.bubble(r)        # fitted line with study-size bubbles
ma.regplot(r)       # Egger-style regression plot for the first moderator
ma.drapery(r)       # shaded likelihood surface of the pooled estimate
```

## Non-linear effects: splines and fractional polynomials

The estimator is linear in the *parameters*, not in the moderator. Hand it a
column for `(x - 5)**3` and it fits it. So a curved dose-response or age effect
needs no new model — only a basis expansion of the moderator:

```python
import pimeta as ma

# 4 knots -> 3 columns: x, plus two non-linear terms
r = ma.rma(yi, vi, mods="~ rcs(dose, 4)", data=df)
print(r.coefnames)      # intrcpt, rcs(dose, 4)[1], rcs(dose, 4)[2], ...

print(ma.nonlinear_test(r, "rcs(dose, 4)"))
```

`rcs` places its knots at Harrell's recommended percentiles (5/35/65/95% for
four knots) unless you pass `knots=`. Beyond the outer knots the fit is forced
to be linear, which is what stops a curve from turning up at the edge of the
data — the failure mode of an ordinary polynomial. `fp` is the
Royston–Sauerbrei alternative: a couple of powers, possibly negative or
fractional, with `log(x)` standing in for the power 0.

```python
r = ma.rma(yi, vi, mods="~ fp(dose, [-1, 0]) + year", data=df)
```

Three things come with the basis:

| What | Call |
|---|---|
| Is there any effect? | `ma.nonlinear_test(r, term).overall` |
| Is it non-linear? | `ma.nonlinear_test(r, term).nonlinear` |
| The curve, with bands | `ma.predict_curve(r, basis)` |

`nonlinear_test` runs joint Wald tests over the term's columns: `overall` over
all of them, `nonlinear` over all but the first (the linear one). For a joint
test over any other set of coefficients, use `ma.wald_test(r, btt=[...])` with
**1-based** positions, as in R.

`predict_curve` takes the basis used in the fit and evaluates it over a grid,
returning the fitted value with a confidence band and a prediction band. If the
model has other moderators, pin them with `at=`:

```python
age = ma.rcs(df["age"], nk=4, name="rcs(age, 4)")
curve = ma.predict_curve(r, age, grid=range(20, 80), at={"year": 2010})
print(curve.fit[:3], curve.pi_lb[:3])
```

A `Basis` can also be passed straight to `mods=`, which is handy in a loop over
candidate knot placements:

```python
basis = ma.rcs(df["dose"], nk=5, name="rcs(dose, 5)")
r = ma.rma(df["yi"], df["vi"], mods=basis)
print(r.b)
```

Everything downstream is unchanged: the result is an ordinary `RmaResult`, so
`predict`, `coef_table`, `record` and the plots all work. The numbers below were
checked against `metafor` 5.2.1 driving `rms::rcs`, and against R's own
`rma(mods=~x)` / `rma(mods=~log(x))` equivalents for the fractional polynomials.

!!! note "Knots and labels"
    The basis columns are named `<label>[1]`, `<label>[2]`, … so a term can be
    found again from the fitted coefficients. `nonlinear_test` and
    `predict_curve` match on that label and ignore whitespace, but the *knots*
    must be the ones used in the fit — a different placement is a different
    basis, and `predict_curve` refuses one with the wrong number of columns.

## As a scikit-learn estimator

The same model is available with the estimator protocol, which unlocks
scikit-learn's Pipeline, GridSearchCV and cross_val_score:

```python
from pimeta.interop import MetaRegressor

est = MetaRegressor(method="REML").fit(dose, yi, sample_weight=1 / vi)
print(est.coef_, est.tau2_)
```

See [Interoperability](interop.md).

## Related

- [Heterogeneity](heterogeneity.md) — `R2`, `QM`, permutation tests.
- [Selection models](selection-models.md) — when the *heterogeneity* itself
  depends on moderators (`scale = ~ z`).
