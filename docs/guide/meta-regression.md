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
