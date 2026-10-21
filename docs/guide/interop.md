# Interoperability

Compatibility is implemented as **protocols**, not as adapters for individual
libraries. One implementation therefore serves many downstream tools, and no
optional dependency is imported until you actually use it.

| Protocol | What you call | What it unlocks |
|---|---|---|
| tidy tables (broom) | `r.tidy()`, `r.glance()`, `r.augment()` | one row per coefficient / model / study; any pandas, polars or CSV pipeline |
| estimator (scikit-learn) | `MetaRegressor().fit(X, y, sample_weight=1/vi)` | `Pipeline`, `GridSearchCV`, `cross_val_score`, `clone`, `joblib`/`dask`/`ray`, `optuna`, `mlflow`, `shap` |
| results (statsmodels) | `as_results(r).conf_int()` | `summary_col`, report builders, anything reading `params` / `bse` / `pvalues` |
| frames | `r.to_table(backend=...)` | `dict`, `records`, `pandas`, `polars`, `pyarrow` |
| labelled arrays | `to_xarray(n)` | `DataArray` / `Dataset` with `treat1` / `treat2` dims |
| graphs | `to_networkx(n)` | centrality, communities, layouts, GraphML / GEXF export |

## Tidy tables

```python
import pimeta as ma

r = ma.rma(yi, vi, mods=dose, method="REML")

r.tidy()      # term / estimate / std.error / statistic / p.value / conf.low / conf.high
r.glance()    # tau2, I2, QE, QEp, QM, logLik, AIC, BIC, ...
r.augment()   # one row per study, plus .hat / .std.resid / .cooksd
```

The same functions are importable:

```python
from pimeta.interop import tidy, glance, augment

table = tidy(r, conf_level=0.99, exponentiate=True, backend="dict")
```

`tidy` understands three shapes:

- **coefficient models** — one row per coefficient;
- **row-wise results** (`leave1out`, `cumul`, `influence`) — one row per study;
- **scale models** (`rma_ls`) — pass `component="alpha"` or `"both"`; the
  latter adds a `component` column.

## Frame backends

```python
from pimeta.interop import to_frame

to_frame({"a": [1, 2], "b": [3.0, 4.0]}, backend="dict")
to_frame({"a": [1, 2], "b": [3.0, 4.0]}, backend="records")
to_frame({"a": [1, 2], "b": [3.0, 4.0]}, backend="pandas")
to_frame({"a": [1, 2], "b": [3.0, 4.0]}, backend="polars")
to_frame({"a": [1, 2], "b": [3.0, 4.0]}, backend="arrow")
```

Scalars are broadcast to the row count; ragged columns raise.

## scikit-learn

```python
import numpy as np
from pimeta.interop import MetaRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score

est = MetaRegressor(method="REML")
est.fit(dose, yi, sample_weight=1 / vi)     # inverse variance = weight

print(est.coef_, est.intercept_, est.tau2_)
print(est.predict([[7.0]], interval="confidence"))

pipe = Pipeline([("scale", StandardScaler()), ("meta", MetaRegressor())])
pipe.fit(X, yi, meta__sample_weight=1 / vi)
print(cross_val_score(pipe, X, yi))
```

`sample_weight` is the sklearn convention (bigger weight = more influence), so
`sample_weight = 1 / vi` is exactly inverse-variance weighting. `vi=` also works
and takes precedence.

The class does **not** import scikit-learn: it implements the duck-typed
contract, so it stays a plain Python object when sklearn is absent and gains
clone, Pipeline, GridSearchCV, cross_val_score, joblib and optuna when it is
present.

## statsmodels

```python
from pimeta.interop import as_results
from statsmodels.iolib.summary2 import summary_col

res = as_results(r)

res.params, res.bse, res.tvalues, res.pvalues
res.conf_int(alpha=0.05), res.cov_params()
res.llf, res.aic, res.bic, res.nobs
res.fittedvalues, res.resid
res.model.exog_names
res.as_pandas()

print(summary_col([res], stars=True))
```

`tvalues` holds the *t* statistic under Hartung–Knapp and *z* otherwise —
statsmodels calls the slot `tvalues` either way.

## xarray and networkx

```python
from pimeta.interop import to_xarray, to_networkx

ds = to_xarray(netmeta_result)          # ("treat1", "treat2") dims
print(ds["TE_random"].sel(treat1="A", treat2="D"))

g = to_networkx(netmeta_result)         # nodes = treatments, edges = comparisons
print(g.nodes["A"]["pscore"])
print(next(iter(g.edges(data=True))))
```

## Asking what is supported

```python
import pimeta as ma

ma.interop.protocol_report(r)
# {'estimator': False, 'tidy': True, 'statsmodels': False,
#  'frame': False, 'table': True, 'array': False}

ma.interop.supports(r, "tidy")     # True
ma.interop.available()             # which optional backends are installed
```

## Installing backends

```bash
pip install pimeta[pandas]        # DataFrame output
pip install pimeta[polars]
pip install pimeta[arrow]
pip install pimeta[sklearn]
pip install pimeta[statsmodels]
pip install pimeta[xarray]
pip install pimeta[networkx]
pip install pimeta[interop]       # all of the above
```
