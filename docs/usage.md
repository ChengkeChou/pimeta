# Usage manual

This is the "type it and it runs" companion to the API reference. **Every
Python block below is followed by the output it actually produced.** Those
blocks are generated and checked by `tools/run_docs_examples.py`, and
`tests/test_docs_examples.py` re-runs them in CI, so the examples cannot drift
away from the code.

!!! note "The examples run in order"
    All blocks share one Python namespace, so a later example can reuse the data
    and the fitted model from an earlier one. To run a single block on its own,
    copy the `import pimeta as ma` line and the data block with it.

!!! tip "Prefer a notebook?"
    [`examples/`](https://github.com/ChengkeChou/pimeta/tree/main/examples) holds
    four notebooks that are already executed, figures included: `01-quickstart`
    (pool to forest plot), `02-meta-regression` (moderators and publication
    bias), `03-network-meta-analysis` (NMA) and `04-ecosystem` (pandas,
    scikit-learn, xarray).

!!! abstract "中文版"
    这份手册还有[中文版](zh/usage.md)。

## 1. Your first pool

The usual entry point is `rma()`: hand it effect sizes `yi` and variances `vi`
and it pools them under a random-effects model.

```python
import pimeta as ma

yi = [0.10, 0.65, 0.20, 0.90, 0.15, 1.10, 0.05, 0.75, 0.40, 1.00]
vi = [0.04, 0.06, 0.05, 0.08, 0.04, 0.09, 0.05, 0.07, 0.06, 0.08]

r = ma.rma(yi, vi, method="REML")

print(f"k             = {r.k}")
print(f"pooled effect = {r.b[0]:.6f}")
print(f"std. error    = {r.se[0]:.6f}")
print(f"95% CI        = [{r.ci_lb[0]:.6f}, {r.ci_ub[0]:.6f}]")
print(f"tau^2         = {r.tau2:.6f}")
print(f"I^2 (%)       = {r.I2:.2f}")
print(f"Q             = {r.QE:.4f} (df={r.k - r.p}, p={r.QEp:.4f})")
```

```text
k             = 10
pooled effect = 0.491362
std. error    = 0.123249
95% CI        = [0.249799, 0.732926]
tau^2         = 0.091675
I^2 (%)       = 61.23
Q             = 22.7202 (df=9, p=0.0069)
```

`r.b` is always an array (even for an intercept-only model), so read it with
`float(r.b[0])`. The field names mirror R's `$` exactly, so r$b becomes r.b,
r$ci.lb becomes r.ci_lb and r$tau2 becomes r.tau2.

## 2. Where the data comes from

Three input styles, all equivalent; pick whichever fits what you have.

```python
import numpy as np

# (a) plain arrays, as above

# (b) column names plus a dict, so long lists cannot get out of sync
dat = {
    "study": ["Adams 2019", "Baker 2020", "Chen 2021", "Diaz 2022", "Evans 2022",
              "Ferrari 2023", "Garcia 2023", "Huang 2024", "Ito 2024", "Jensen 2025"],
    "yi": yi,
    "vi": vi,
    "dose": [10, 20, 30, 40, 50, 60, 70, 80, 90, 100],
}
r2 = ma.rma("yi", "vi", data=dat)

# (c) the formula interface: mods is written as "~ 1 + dose"
r3 = ma.rma("yi", "vi", mods="~ dose", data=dat)

print("(b) matches (a):", np.allclose(r2.b, r.b))
print("(c) coefficient names:", r3.coefnames)
print("(c) coefficients:", [round(float(z), 6) for z in r3.b])
```

```text
(b) matches (a): True
(c) coefficient names: ['intrcpt', 'dose']
(c) coefficients: [0.247874, 0.004535]
```

With pandas you can pass the DataFrame directly and use the same column names:

```python
import pandas as pd

df = pd.DataFrame(dat)
r4 = ma.rma("yi", "vi", data=df)
print(r4.k, round(float(r4.b[0]), 6))
```

```text
10 0.491362
```

## 3. Effect sizes: `escalc()`

Start with the most common case, a 2×2 table. For **ratio** measures
(OR/RR/IRR) `escalc()` returns the **log** scale; exponentiate to report the
ratio itself.

```python
ai = [12, 15, 9, 20]      # events, treatment arm
bi = [88, 85, 91, 80]     # non-events, treatment arm
ci = [20, 22, 14, 25]     # events, control arm
di = [80, 78, 86, 75]     # non-events, control arm

es_or = ma.escalc("OR", ai=ai, bi=bi, ci=ci, di=di)

print("measure:", es_or.measure, "| k =", es_or.k)
print("log OR :", [round(float(v), 6) for v in es_or.yi])
print("OR     :", [round(float(v), 4) for v in es_or.yi])
print("var    :", [round(float(v), 6) for v in es_or.vi])
```

```text
measure: OR | k = 4
log OR : [-0.606136, -0.468935, -0.498345, -0.287682]
OR     : [-0.6061, -0.4689, -0.4983, -0.2877]
var    : [0.157197, 0.136706, 0.205157, 0.115833]
```

Continuous outcomes and single-arm proportions:

```python
es_md = ma.escalc("MD",
                  n1i=[30, 25, 40], m1i=[5.2, 4.9, 6.1], sd1i=[1.1, 1.3, 0.9],
                  n2i=[28, 26, 38], m2i=[4.6, 4.4, 5.3], sd2i=[1.0, 1.2, 1.1])
es_pr = ma.escalc("PLO", xi=[3, 7, 12], ni=[40, 50, 60])

print("MD      :", [round(float(v), 6) for v in es_md.yi])
print("log-odds:", [round(float(v), 6) for v in es_pr.yi])
print("back to proportions:",
      [round(float(v), 4) for v in 1 / (1 + np.exp(-np.asarray(es_pr.yi)))])
```

```text
MD      : [0.6, 0.5, 0.8]
log-odds: [-2.512306, -1.81529, -1.386294]
back to proportions: [0.075, 0.14, 0.2]
```

`escalc()` covers the main metafor measures: binary `OR`/`RR`/`RD`/`AS`/`PETO`,
continuous `MD`/`SMD`/`SMDH`/`ROM`/`CVR`, single-arm proportions
`PLN`/`PLO`/`PAS`/`PFT`/`PR`, correlations `ZCOR`/`COR`, and rate ratios
`IRR`/`IRLN`/`IRS`. See the API reference for the full signatures.

## 4. `rma()` options

The three things you change most often are the τ² estimator, the test, and the
confidence level.

```python
for method in ("DL", "REML", "PM", "HS", "EB", "ML"):
    fit = ma.rma(yi, vi, method=method)
    print(f"{method:5s} tau2={fit.tau2:.6f}  I2={fit.I2:6.2f}%  b={fit.b[0]:.6f}")

# Knapp-Hartung widens the interval, which is safer with few studies
kh = ma.rma(yi, vi, test="knha")
print(f"\nknha: t={kh.zval[0]:.4f} p={kh.pval[0]:.4f} "
      f"CI=[{kh.ci_lb[0]:.4f}, {kh.ci_ub[0]:.4f}]")
```

```text
DL    tau2=0.088490  I2= 60.39%  b=0.490553
REML  tau2=0.091675  I2= 61.23%  b=0.491362
PM    tau2=0.094703  I2= 62.00%  b=0.492102
HS    tau2=0.073218  I2= 55.78%  b=0.486144
EB    tau2=0.094704  I2= 62.00%  b=0.492102
ML    tau2=0.075659  I2= 56.59%  b=0.486914

knha: t=3.9484 p=0.0034 CI=[0.2098, 0.7729]
```

## 5. Heterogeneity and prediction intervals

```python
print(f"H^2   = {r.H2:.4f}")
print(f"I^2   = {r.I2:.2f}%")
print(f"tau   = {np.sqrt(r.tau2):.6f}")

# interval for the effect in a future study (metafor's predict())
pred = ma.predict(r)
print("estimate:", f"{pred.pred:.4f}", " 95% CI:", f"[{pred.ci_lb:.4f}, {pred.ci_ub:.4f}]")
print("prediction interval:", f"[{pred.pi_lb:.4f}, {pred.pi_ub:.4f}]")

# confidence interval for tau^2 itself
ci_tau2 = ma.confint(r)
print("tau^2 CI:", {k: (round(v, 6) if isinstance(v, float) else v)
                    for k, v in ci_tau2.items()})
```

```text
H^2   = 2.5793
I^2   = 61.23%
tau   = 0.302779
estimate: 0.4914  95% CI: [0.2498, 0.7329]
prediction interval: [-0.2481, 1.2309]
tau^2 CI: {'tau2': 0.091675, 'ci_lb': 0.011901, 'ci_ub': 0.465019, 'level': 0.95}
```

## 6. Subgroups and meta-regression

```python
reg = ma.rma(yi, vi, mods="~ dose", data=dat)
print("intercept:", round(float(reg.b[0]), 6))
print("dose     :", round(float(reg.b[1]), 6))
print("QM       :", f"{reg.QM:.4f}", "df =", reg.QMdf[0], "p =", f"{reg.QMp:.4f}")
print("tau^2    :", f"{r.tau2:.6f}", "->", f"{reg.tau2:.6f}")

# categorical moderators are dummy-coded the way R does it
dat2 = dict(dat, arm=["A", "B", "A", "B", "A", "B", "A", "B", "A", "B"])
reg2 = ma.rma("yi", "vi", mods="~ arm", data=dat2)
print("coefficient names:", reg2.coefnames)
```

```text
intercept: 0.247874
dose     : 0.004535
QM       : 1.1431 df = 1 p = 0.2850
tau^2    : 0.091675 -> 0.089292
coefficient names: ['intrcpt', 'arm[B]']
```

## 7. Publication bias

```python
rt = ma.regtest(yi, vi)          # Egger's regression test
tf = ma.trimfill(yi, vi)         # trim-and-fill
mb = ma.metabias(r)              # test on an rma() result

print("regtest  p =", f"{rt.p:.4f}", " slope =", f"{rt.slope:.4f}")
print("trimfill filled =", tf.k0, " adjusted effect =", f"{tf.filled_effect:.6f}")
print("metabias p =", f"{mb.pval:.4g}" if mb.pval is not None
      else f"k={mb.k} < k_min={mb.k_min}, not reported by convention")
```

```text
regtest  p = 0.0000  slope = 10.4959
trimfill filled = 1  adjusted effect = 0.438025
metabias p = 3.12e-05
```

Publication-bias tests are conventionally only reported with `k >= 10`, which is
exactly the size of this example.

## 8. Multilevel / multivariate models: `rma_mv()`

When a study contributes several effect sizes, the within-study correlation has
to enter the model.

```python
study = ["A", "A", "B", "B", "C", "C", "D", "D", "E", "E", "F", "F"]
y_mv = [0.10, 0.35, 0.55, 0.20, 0.80, 0.65,
        0.15, 0.45, 0.95, 0.70, 0.30, 0.60]
v_mv = [0.05, 0.04, 0.06, 0.05, 0.07, 0.06,
        0.04, 0.05, 0.08, 0.06, 0.05, 0.05]

mv = ma.rma_mv(y_mv, v_mv, random="~ 1 | study", data={"study": study})

print("k =", mv.k, "effect sizes from", len(set(study)), "studies")
print("beta =", f"{float(np.atleast_1d(mv.beta)[0]):.6f}")
print("sigma^2 =", [round(float(t), 6) for t in np.atleast_1d(mv.sigma2)])
print("random =", mv.random, "| struct =", mv.struct)
```

```text
k = 12 effect sizes from 6 studies
beta = 0.458086
sigma^2 = [0.023807]
random = ~ 1 | study | struct = CS
```

!!! warning "`rma_mv` calls the variance component `sigma2`"
    This follows R's `rma.mv`: a multivariate model stores its components in
    `$sigma2` (one per random term) and does not use `$tau2`. Only the
    univariate `rma()` uses `tau2`.

## 9. Network meta-analysis: `netmeta()`

```python
nm = ma.netmeta(
    TE=[0.30, -0.20, 0.50, 0.10, -0.15, 0.25],
    seTE=[0.12, 0.15, 0.18, 0.14, 0.11, 0.13],
    treat1=["A", "B", "A", "C", "B", "C"],
    treat2=["B", "A", "C", "A", "C", "B"],
    studlab=["S1", "S2", "S3", "S4", "S5", "S6"],
    sm="RR",
)

print("treatments:", list(nm.trts))
print("comparisons:", nm.m, " treatments:", nm.n)
print("tau^2 =", f"{nm.tau2:.6f}")
print("Q =", f"{nm.Q:.4f}", " df =", nm.df_Q, " p =", f"{nm.pval_Q:.4f}")
print("Q split: heterogeneity", f"{nm.Q_heterogeneity:.4f}",
      "+ inconsistency", f"{nm.Q_inconsistency:.4f}")
print("TE_random shape:", np.shape(nm.TE_random), "(treatment x treatment)")
print("A vs B:", f"{float(nm.TE_random[0, 1]):.4f}",
      f"[{float(nm.lower_random[0, 1]):.4f}, {float(nm.upper_random[0, 1]):.4f}]")
```

```text
treatments: ['A', 'B', 'C']
comparisons: 6  treatments: 3
tau^2 = 0.016798
Q = 7.6544  df = 4.0  p = 0.1051
Q split: heterogeneity 7.5389 + inconsistency 0.1155
TE_random shape: (3, 3) (treatment x treatment)
A vs B: 0.2866 [0.0739, 0.4993]
```

Ranking, league table and node splitting:

```python
rk = ma.netrank(nm)
print("P-score (in treatment order):", [round(float(v), 4) for v in rk.pscore_random])

lg = ma.netleague(nm)
print("league treatments:", len(lg.trts),
      "| random matrix shape:", np.asarray(lg.random).shape)

sp = ma.netsplit(nm)
print("split comparisons:", list(sp.comparison))
print("compare_random fields:", list(sp.compare_random))
print("direct vs indirect:", [round(float(v), 4) for v in sp.compare_random["TE"]])
```

```text
P-score (in treatment order): [0.0763, 0.9707, 0.4531]
league treatments: 3 | random matrix shape: (3, 3)
split comparisons: ['A:B', 'A:C', 'B:C']
compare_random fields: ['TE', 'seTE', 'lower', 'upper', 'statistic', 'p', 'z']
direct vs indirect: [-0.0956, 0.0956, -0.0956]
```

`ma.netheat(nm)` draws the design-by-treatment heat plot; the matrix behind it
is `ma.netmeta.netheat_matrix(nm)`, transcribed from R's `netheat()`.

## 10. Raw binary network data: `netmetabin()`

```python
nb = ma.netmetabin(
    event1=[12, 15, 9, 20], n1=[100, 100, 100, 140],
    event2=[20, 22, 14, 25], n2=[100, 100, 100, 150],
    treat1=["A", "B", "A", "C"], treat2=["B", "A", "C", "A"],
    studlab=["S1", "S2", "S3", "S4"], sm="OR",
)
print("method:", nb.method, "| designs:", nb.d)
print("first two log ORs:", np.round(np.asarray(nb.TE, dtype=float)[:2], 4))
print("Q =", f"{nb.Q:.4f}", "p =", f"{nb.pval_Q:.4f}", "| I^2 =", f"{nb.I2:.2f}%")
```

```text
method: MH | designs: 2
first two log ORs: [-0.6061  0.4689]
Q = nan p = nan | I^2 = nan%
```

## 11. Getting results out

Three levels of detail: one row per coefficient (`tidy`), model-level numbers
(`glance`), and the study-level table (`to_dataframe`).

```python
t = ma.interop.tidy(r)
print("tidy type:", type(t).__name__, "| columns:", list(t))
print(t.to_string(index=False))

g = ma.interop.glance(r)
print("\nglance keys:", list(g)[:8], "...")
print("k =", int(g["k"].iloc[0]),
      "| tau2 =", round(float(g["tau2"].iloc[0]), 6),
      "| I2 =", round(float(g["I2"].iloc[0]), 2),
      "| QEp =", round(float(g["QEp"].iloc[0]), 6))

print("\nto_dataframe(r) columns:", list(ma.to_dataframe(r)))
```

```text
tidy type: DataFrame | columns: ['term', 'estimate', 'std.error', 'statistic', 'p.value', 'conf.low', 'conf.high']
   term  estimate  std.error  statistic  p.value  conf.low  conf.high
intrcpt  0.491362   0.123249   3.986743 0.000067  0.249799   0.732926

glance keys: ['k', 'p', 'tau2', 'I2', 'H2', 'R2', 'QE', 'QEp'] ...
k = 10 | tau2 = 0.091675 | I2 = 61.23 | QEp = 0.006856

to_dataframe(r) columns: ['yi', 'vi', 'X', 'weights', 'weights_pct', 'fitted', 'residuals', 'slab', 'not_na']
```

`to_table()` gives you a pandas `DataFrame` by default, or `dict` / `records` /
`polars` / `arrow`:

```python
tbl = r.to_table()
print("default type:", type(tbl).__name__, "| shape:", getattr(tbl, "shape", None))
print("dict backend keys:", list(r.to_table(backend="dict"))[:6])
print("Markdown output available:", hasattr(tbl, "to_markdown"))
```

```text
default type: DataFrame | shape: (10, 9)
dict backend keys: ['yi', 'vi', 'X', 'weights', 'weights_pct', 'fitted']
Markdown output available: True
```

## 12. Working with the Python ecosystem

`pimeta` does not import sklearn or statsmodels, but it **implements their
protocols**, so results drop straight into a Pipeline:

```python
from sklearn.base import clone, is_regressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

est = ma.interop.MetaRegressor(method="REML")
X = np.asarray(dat["dose"], dtype=float).reshape(-1, 1)
w = 1.0 / np.asarray(vi)
est.fit(X, np.array(yi), sample_weight=w)

print("is_regressor:", is_regressor(est))
print("predict dose=55:", round(float(est.predict(np.array([[55.0]]))[0]), 6))
print("clone works:", clone(est).get_params()["method"])

pipe = Pipeline([("scale", StandardScaler()),
                 ("meta", ma.interop.MetaRegressor(method="DL"))])
pipe.fit(X, np.array(yi), meta__sample_weight=w)
print("Pipeline predict:", round(float(pipe.predict(np.array([[55.0]]))[0]), 6))
```

```text
is_regressor: True
predict dose=55: 0.497301
clone works: REML
Pipeline predict: 0.49666
```

Export to xarray, networkx or a statsmodels-style results object:

```python
ds = ma.interop.to_xarray(nm)
print("xarray dims:", dict(ds.dims))

g = ma.interop.to_networkx(nm)
print("networkx:", g.number_of_nodes(), "nodes", g.number_of_edges(), "edges")

sm_res = ma.interop.as_results(r)
print("statsmodels-style fields:", ["params", "bse", "pvalues", "aic"])
print("  params =", sm_res.params.round(6), " aic =", round(sm_res.aic, 4))
```

```text
xarray dims: {'treat1': 3, 'treat2': 3, 'treatment': 3}
networkx: 3 nodes 3 edges
statsmodels-style fields: ['params', 'bse', 'pvalues', 'aic']
  params = [0.491362]  aic = 12.8138
```

## 13. Plots

Every plot can be called directly (`ma.forest(...)` returns a Matplotlib
`Axes`) or through the dispatcher `ma.plot(result, which=...)`; pass
`filename=` to write it to disk as well.

```python
import tempfile
from pathlib import Path

out = Path(tempfile.mkdtemp())

ax = ma.plot(r, which="forest", filename=str(out / "forest.png"))
print("forest  ->", type(ax).__name__, "| written:", (out / "forest.png").stat().st_size > 0)

for which in ("funnel", "baujat", "radial", "qqnorm"):
    obj = ma.plot(r, which=which, filename=str(out / f"{which}.png"))
    print(f"{which:7s} ->", type(obj).__name__,
          "| written:", (out / f"{which}.png").stat().st_size > 0)

ax2 = ma.forest(yi, vi, studlab=dat["study"])
print("ma.forest returns:", type(ax2).__name__,
      "| figure size:", tuple(float(v) for v in ax2.figure.get_size_inches()))
```

```text
forest  -> Axes | written: True
funnel  -> Axes | written: True
baujat  -> Axes | written: True
radial  -> PlotResult | written: True
qqnorm  -> PlotResult | written: True
ma.forest returns: Axes | figure size: (7.0, 6.5)
```

## 14. Command line

Installing the package also installs the `pimeta` command, which is handy in
shell scripts and Makefiles.

```bash
pimeta fit results.csv --yi yi --vi vi --method REML -o pooled.tsv
pimeta fit results.csv --yi yi --vi vi --mods "~ dose" --forest forest.png
pimeta escalc results.csv --measure OR --ai ai --bi bi --ci ci --di di
pimeta network net.csv --TE TE --seTE seTE --treat1 treat1 --treat2 treat2
```

```text
```

Besides the main table, `-o` also writes `<name>_glance.tsv` next to it with the
model-level numbers (k, τ², I², Q, p). `python -m pimeta` is equivalent to
`pimeta` when the console script is not on the PATH.

## 15. Traps

| Symptom | Cause | Fix |
|---|---|---|
| the pooled value looks wrong | `escalc` returns the log scale for OR/RR/IRR | report `math.exp(r.b[0])` |
| `r.b` prints as `[0.29]` | `b` is always an array | `float(r.b[0])` |
| `I2` is 92.2, not 0.922 | deliberate, matches R | read it as a percentage |
| `rma_mv(level=0.95)` fails | this argument takes a percentage | `level=95`; `rma()` takes `0.95` |
| `slab=["Study 1", ...]` treated as column names | a list of strings is only read as names when every string is a real column | pass `strings_as_names=False` to force |
| `netmeta` says comparisons were dropped | missing values | clean the data, or silence with `warn_missing=False` |
| meta-regression coefficients are `V1` | you passed arrays | use `mods="~ dose", data=...` |
| `netheat` warns and returns `None` | the network has no loop (no between-design df) | it needs at least two independent paths between treatments |
