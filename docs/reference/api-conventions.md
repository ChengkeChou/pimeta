# API conventions

A few things differ from what a Python-first user might assume. Each one is
deliberate: it matches the R object being mirrored.

| Convention | Detail |
|---|---|
| `result.I2` is a **percentage** | `92.22`, not `0.9222` — identical to R's `r$I2`. Same for `H2` and `R2`. |
| `confint(result)` returns a **dict** | `{"tau2", "ci_lb", "ci_ub", "level"}` — the QP/profile confidence interval for τ², mirroring `confint.rma()`. |
| `result.b` is always an **array** | Even intercept-only models give an `np.ndarray` of length 1, like R's `r$b`. Use `float(r.b[0])`. |
| `escalc()` returns **log-scale** `yi` for ratio measures | `RR` / `OR` / `IRR` are log-transformed; use `math.exp()` to get the reported ratio. |
| `netrank(...).ranking_random` holds **P-scores**, not ranks | netmeta 3.7-0 names the field `ranking.random` even though it contains P-scores. `Pmatrix_random` holds the pairwise probability matrix. |
| `rma_mv(..., level=95)` vs `rma(..., level=0.95)` | metafor itself uses a percentage for `rma.mv` and a fraction for `rma`; pimeta reproduces the inconsistency. |
| `rma_ls` reports `tau2` / `I2` / `H2` as **vectors** | τ² varies by study in a location-scale model, so these are length `k`, not scalars. |
| `btt` / `att` are **1-based** | As in R: the intercept is coefficient 1. |
| Plots return a matplotlib `Axes` | Pass `filename="x.png"` to also write the figure, otherwise it stays in memory. The backend falls back to `Agg`, so nothing blocks a headless script. |

## Result objects

Every fitted model is a dataclass that mirrors the R object's `$` fields:

```python
import pimeta as ma

r = ma.rma(yi, vi, method="REML")

r.b, r.se, r.zval, r.pval, r.ci_lb, r.ci_ub   # r$b, r$se, ... r$ci.ub
r.tau2, r.I2, r.H2, r.QE, r.QEp               # r$tau2, r$I2, ...
r.weights, r.fitted, r.residuals              # r$weights, r$fitted, ...
r.method, r.test, r.level, r.k, r.p            # settings and dimensions
```

All of them also provide:

| Member | Purpose |
|---|---|
| `to_dict()` | plain `dict` of scalars and lists — JSON-serialisable |
| `tidy()` / `glance()` / `augment()` | broom-style tables |
| `to_table(backend=...)` | the same as a frame |
| `_repr_html_` | formatted table in Jupyter |

## Units and scales

| Quantity | Scale |
|---|---|
| `b`, `yi`, `fitted`, `residuals` | analysis scale (log for ratio measures) |
| `vi`, `se`, `vb`, `weights` | variance / standard error / inverse-variance weight |
| `QE`, `QM` | Cochran's Q and the omnibus moderator statistic |
| `I2`, `H2`, `R2` | percentages (0–100) |
| `tau2` | variance; `tau` is reported by `netmeta` only |

## Versions

`pimeta.__version__` gives the installed version. The suite is tested on Python
3.10–3.13 with NumPy 1.26 and 2.x, SciPy 1.16–1.18 and matplotlib 3.11.
