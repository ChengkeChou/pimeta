# pimeta

[![PyPI](https://img.shields.io/pypi/v/pimeta.svg)](https://pypi.org/project/pimeta/)
[![Python versions](https://img.shields.io/pypi/pyversions/pimeta.svg)](https://pypi.org/project/pimeta/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Tests](https://img.shields.io/badge/tests-1050%20passing-brightgreen.svg)](https://github.com/ChengkeChou/pimeta/actions/workflows/publish.yml)

**Pure-Python meta-analysis — a faithful, numerically verified port of R's
[`metafor`](https://cran.r-project.org/package=metafor),
[`netmeta`](https://cran.r-project.org/package=netmeta) and
[`meta`](https://cran.r-project.org/package=meta).**

No R installation. No rpy2. No third-party meta-analysis library under the hood —
the numeric engine is a direct NumPy/SciPy transcription of the R reference
implementations, so results reproduce R to floating-point tolerance.

```bash
pip install pimeta
```

```python
import pimeta as ma

r = ma.rma(yi=[0.20, 0.40, 0.10, 0.50], vi=[0.04, 0.05, 0.03, 0.06], method="REML")
print(r.b, r.se, r.tau2, r.I2)     # pooled estimate, SE, tau^2, I^2
```

---

## Why this package

| | R (`metafor`/`netmeta`/`meta`) | **pimeta** |
|---|---|---|
| Runtime | R + C/C++ toolchain | Python + NumPy/SciPy |
| Integration | subprocess / rpy2 | native Python objects |
| Output | R list-like S3 objects | dataclasses with the **same field names** |
| Plots | base graphics / grid | matplotlib `Axes` you can further style |
| Reproducibility | reference implementation | 375 exported functions verified against R |

Every result object exposes the **same attribute names as the R object's `$`
fields**, so code and documentation written for R translate almost one-to-one:

```python
r = ma.rma(yi, vi, method="REML")
r.b, r.se, r.zval, r.pval, r.ci_lb, r.ci_ub   # R: r$b, r$se, r$zval, r$pval, r$ci.lb, r$ci.ub
r.tau2, r.I2, r.H2, r.QE, r.QEp               # R: r$tau2, r$I2, r$H2, r$QE, r$QEp
r.weights, r.fitted, r.residuals              # R: r$weights, r$fitted, r$residuals
```

---

## What is covered

### Effect sizes — `escalc()`
`RR`, `OR`, `RD`, `AS`, `PETO`, `IRR`, `IRS`, `IRSD`, `MN`, `SMN`,
`MRAW`, `SMCC`, `MD`, `SMD`, `SMDH`, `ROM`, `RPB`, `RBIS`, `D2OR`, `COR`, `UCOR`,
`ZCOR`, `ZPCOR`, `PR`, `PLN`, `PLO`, `PAS`, `PFT`, `PRAW`, `PD` — plus cell
continuity corrections (`add`, `to = "only0" | "all" | "IF0"`) and `drop00`.

### Pooling and heterogeneity
- Fixed-effect and random-effects (`common` / `random`) pooling
- τ² estimators: `DL`, `HE`, `HS`, `SJ`, `ML`, `REML`, `PM`, `EB`, `GENQ`,
  `GENQM`, `QGENQ`, `QGENQM`
- Hartung–Knapp adjustment, Knapp–Hartung (`test = "knha"`, `"adhoc"`)
- Prediction intervals, `QE`/`I²`/`H²`/`R²`, `QE`-based CIs, profile-likelihood τ² CI

### High-level drivers (R-function parity)
`rma()`, `rma.uni` (`rma_uni`), `rma.mv` (`rma_mv`), `rma.glmm` (`rma_glmm`),
`rma.peto`, `rma.mh`, `rma.uni.selmodel`, `rma.ls` (`rma(..., scale=~z)`),
`metagen`, `metabin`, `metacont`,
`metacor`, `metainc`, `metaprop`, `metarate`, `metamean`, `metacum`, `metainf`,
`metareg`, `trimfill`, `regtest`, `metabias`, `begg_test`, `funnel`, `forest`,
`labbe`, `radial`, `baujat`, `leave1out`, `influence`, `cooks_distance`,
`rstudent`, `hatvalues`, `gosh`, `fsn`, `permutest`, `tes`, `vif`,
`profile_rma`, `profile_rmamv`, `profile_rma_ls`, `confint_rma_ls`.

`leave1out` / `influence` / `confint` results support R's ``x[i]``,
`head(x)` and `tail(x)` idioms natively: `x[0:3]`, `x[2]`, `x[[1, 3]]` and
`ma.head(x, 6)` / `ma.tail(x, 6)`.  R's `generic.class` method names
(`print.rma.uni`, `cooks.distance.rma.mv`, `[.list.rma`, ...) are not valid
Python identifiers, but `getattr(ma, "cooks.distance.rma.mv")` resolves to the
implementing callable — see `ma.r_methods()` for the full 66-entry map.

`rma.peto` / `rma.mh` carry metafor's own methods: `weights`, `cumul`,
`leave1out`, `rstandard`, `rstudent`, `confint` and `qqnorm` all accept the
fitted object, and `rma.mv` fits support `hatvalues`, `weights`, `rstandard`,
`cooks_distance`, `dfbetas`, `rstudent`, `ranef`, `robust`, `confint`,
`predict`, `qqnorm` and `profile_rmamv` (metafor's `profile.rma.mv`).

### Command line

The package installs a `pimeta` command (and `python -m pimeta` works too), so
the common analyses need no script at all:

```bash
pimeta fit studies.csv --yi TE --vi seTE2 --method REML
pimeta fit studies.csv --yi TE --vi seTE2 --mods "~ dose + year" -o coef.csv
pimeta escalc raw.csv --measure RR --ai ai --bi bi --ci ci --di di
pimeta network nma.csv --te TE --se seTE --treat1 t1 --treat2 t2 --sm MD
```

Output is TSV on stdout, `--format csv|json` or `-o file.csv` otherwise.  See
[the guide](docs/guide/cli.md).

### Network meta-analysis (`netmeta`)
`netmeta`, `netmetabin`, `netcomb`, `discomb`, `netcomplex`, `netmetareg`,
`netmeasures`, `netrank` (P-scores), `netimpact`, `netpath`,
`netsplit`, `netmatrix`, `netcomparison`, `netbind`, `netcontrib`,
`netdistance`, `netheat`, `nettable`, `netgraph`, `netposet`, `hasse`,
`rankogram`, `decomp_design`, `invmat`, `hatmatrix`, `comps`, `combinations`,
`treats`, `vikor`, `subgroup`, `crossnma2netmeta`, `gemtc2netmeta`, `createC`.

### Publication bias & sensitivity
`regtest` (Egger), `begg_test` (Begg rank test), `metabias` — all nine `meta`
tests: `Begg`, `Egger`, `Thompson`, `Schwarzer`, `Harbord`, `Peters`, `Deeks`,
`Pustejovsky`, `Macaskill` — `trimfill` (L0 / R0 / Q0), `fsn`, `funnel`,
`gosh`, `influence`, `baujat`, `leave1out`, `permutest`.

### Reporting helpers (`meta` package parity)
`cidprop`, `cilayout`, `format_ci`, `forest_dims`, `longarm`, `addpoly`,
`reporter`, `rob`, `aggregate_escalc`, `cbind_escalc`, `rbind_escalc`,
`estimates`, `emmprep`, `points_regplot`.

### S3-style generics
`coef`, `vcov`, `fitted`, `residuals`, `logLik`, `deviance`, `nobs`, `formula`,
`AIC`, `BIC`, `weights`, `sigma`, `df_residual`, `model_matrix`, `simulate`,
`confint`, `predict`, `hatvalues`, `hatvalues_rma`, `dfbetas`, `dfbetas_rma`,
`rstandard`, `rstandard_rma`, `ranef`, `summary`, `summary_text`,
`print_summary`, `plot`.

### Figures
`forest`, `funnel`, `radial`, `labbe`, `llplot`, `bubble`, `drapery`,
`qqnorm_rma`, `plot_baujat`, `plot_gosh`, `plot_influence` (metafor's
`plot.infl.rma.uni`), `plot_permutest` (metafor's `plot.permutest.rma.uni`),
`plot_selprofile` (metafor's `plot.profile.rma`), `plot_mv_profile`
(metafor's `plot.profile.rma` for `rma.mv` variance components), `heatplot`,
`regplot`,
`netgraph`, `netheat`, `rankogram`, `hasse`,
`netposet`, `traffic_light` — plus `plot(x, which=...)` to reach all of them
from a fitted object.

### Readers
`read_rm5`, `read_cdir`, `read_mtv` — parse Cochrane RevMan `.rm5` files and
Cochrane `.cdir` / `.mtv` exports directly into analysis-ready objects.

### Transforms (all 58 of `metafor::transf.*`)
`transf_logit`, `transf_ilogit`, `transf_probit`, `transf_arcsin`,
`transf_ztor`, `transf_ztor2`, `transf_dtocliffd`,
`transf_pft`, `transf_ipft`, `transf_ahw`, `transf_iahw`, `transf_abt`,
`transf_iabt`, `transf_rpbtod`, `transf_dtou1`, `transf_dtou2`,
`transf_dtou3`, and the `_int` / `_mode` variants.

---

## Quick start

### 1. Effect sizes from raw 2×2 data

```python
import pimeta as ma

d = ma.escalc(
    measure="RR",
    ai=[10, 20, 15], bi=[90, 80, 85],     # events / non-events, treatment
    ci=[12, 18, 14], di=[88, 82, 86],     # events / non-events, control
)
print(d.yi)   # log risk ratios
print(d.vi)   # sampling variances
```

### 2. Random-effects pooling + forest plot

```python
r = ma.rma(d.yi, d.vi, method="REML")
print(r.b, r.ci_lb, r.ci_ub)   # 0.0266  -0.3557  0.4090   (log RR; exp() -> 1.027)
print(r.tau2, r.I2, r.QE, r.QEp)  # 0.0  0.0  0.355  0.837

ma.forest(d.yi, d.vi, studlab=["Study A", "Study B", "Study C"],
          pooled=r, effect_type="RR", filename="forest.png")
```

### 3. Meta-regression

```python
# yi, vi come from your study table; dose / baseline_risk are the moderators
meta_reg = ma.rma(yi, vi, mods=[dose, baseline_risk], method="REML")
print(meta_reg.b)          # intercept + slopes
print(meta_reg.QM, meta_reg.QMp)
```

### 4. Network meta-analysis

```python
n = ma.netmeta(
    TE, seTE, treat1, treat2, studlab,
    sm="MD", reference_group="placebo",
)
print(n.TE_random)                       # league-style contrast matrix
print(ma.netrank(n).ranking_random)      # P-scores per treatment
print(ma.netmeasures(n).proportion)      # direct-evidence proportions
ma.netgraph(n, filename="network.png")   # evidence graph
```

### 5. Publication bias

```python
print(ma.regtest(d.yi, d.vi).p)          # Egger's test p-value
print(ma.trimfill(d.yi, d.vi).k0)        # trim-and-fill, L0 estimator
print(ma.fsn(d.yi, d.vi, type="Rosenthal").fsnum)
```

---

## Requirements

- Python ≥ 3.10 (tested on 3.10 – 3.13)
- `numpy` ≥ 1.24, `scipy` ≥ 1.10, `matplotlib` ≥ 3.7
- `pandas` is optional — only needed for DataFrame-shaped output
  (`pip install pimeta[pandas]`)

CI runs the suite on Python 3.12 with `numpy` 2.5, `scipy` 1.18 and
`matplotlib` 3.11; the same example script passes locally with `numpy` 1.26 /
`scipy` 1.16, so both sides of the NumPy 2.0 boundary are covered. Figures use
the headless `Agg` backend and never need a display.

---

## Numerical validation

The library is developed against a differential test suite: identical datasets
are fed to R (`metafor` 5.2-1, `netmeta` 3.7-0, `meta` 8.5-0) and to `pimeta`,
and every reported quantity is compared. Typical agreement:

| Family | Max abs. deviation from R |
|---|---|
| `escalc` / effect sizes | 0 (bit-identical cells) |
| `rma`, `metagen`, τ² estimators | < 1e-12 |
| `rma.mv` / multilevel | < 1e-6 |
| `netmeta` NMA (Q, τ², contrasts) | < 1e-6 |
| `netmetareg`, `netmeasures`, `netimpact` | < 1e-8 |
| `transf.*`, `fmt.*`, measures | < 1e-10 |

Where R has quirks (non-convergence, `NA` subsets, sign conventions of
`contrmat`), the behaviour is reproduced rather than "fixed".

Coverage: `meta` 59/59, `netmeta` 38/38 and `metafor` 278/279 exported
functions. The single exception is `metafor.news()`, a changelog viewer with no
analytic content.

---

## API conventions worth knowing

A few things differ from what a Python-first user might assume. Each one is
deliberate: it matches the R object being mirrored.

| Convention | Detail |
|---|---|
| `result.I2` is a **percentage** | `92.22`, not `0.9222` — identical to R's `r$I2`. Same for `H2` and `R2`. |
| `confint(result)` returns a **dict** | `{"tau2", "ci_lb", "ci_ub", "level"}` — the QP/profile confidence interval for τ², mirroring `confint.rma()`. |
| `result.b` is always an **array** | Even intercept-only models give `np.ndarray` of length 1, like R's `r$b`. Use `float(r.b[0])`. |
| `escalc()` returns **log-scale** `yi` for ratio measures | `RR` / `OR` / `IRR` are log-transformed; use `math.exp()` to get the reported ratio. |
| `netrank(...).ranking_random` holds **P-scores**, not ranks | netmeta 3.7-0 names the field `ranking.random` even though it contains P-scores. `Pmatrix_random` holds the pairwise probability matrix. |
| Plots return a matplotlib `Axes` | Pass `filename="x.png"` to also write the figure, otherwise it stays in memory. The backend is forced to `Agg`, so nothing blocks a headless script. |

`pimeta.__version__` gives the installed version.

---

## Differences from R

Nothing in the analytic surface is missing. Every deviation below is either
forced by Python's grammar or is a place where an R quirk was deliberately not
reproduced. Everything else is verified against R — see
[Numerical validation](#numerical-validation).

### 1. Names that cannot exist in Python

R encodes the class inside the function name (`print.rma.uni`,
`cooks.distance.rma.mv`). In Python `.` is attribute access, so those strings
are not identifiers; and `[`, `$`, `[<-` are operators, not names at all. The
behaviour is fully implemented — only the spelling changes.

| R | pimeta |
|---|---|
| `print(x)` / `summary(x)` | `print(x)` / `x.summary()`; `str(x)` and `summary_text(x)` give the text block |
| `cooks.distance(x)`, `ranktest(x)`, `hc(x)` | `ma.cooks_distance(x)`, `ma.begg_test(x)`, `ma.hc_rma(x)` |
| `x[1:3]`, `x[[1,3]]`, `head(x)`, `tail(x)` | `x[0:3]`, `x[[1, 3]]`, `ma.head(x)`, `ma.tail(x)` |
| `x$ci.lb`, `x$tau2` | `x.ci_lb`, `x.tau2` |
| `x$b <- 0` | `x.b = np.array([0.0])` — result objects are plain mutable objects |
| `update(x, mods = ~z)` | `ma.update_rma(x, mods=z)` — refits, like R |
| `as.data.frame(x)` | `ma.as_data_frame_rma_list(x)` or `x.to_dict()` |

If you are mechanically translating R code, `ma.r_methods()` returns the full
66-entry map from R method name to pimeta callable, and a module-level
`__getattr__` resolves the R spelling directly:

```python
import pimeta as ma

getattr(ma, "cooks.distance.rma.mv")   # -> ma.cooks_distance
getattr(ma, "print.rma.uni")           # -> ma.summary
getattr(ma, "profile.rma.ls")          # -> ma.profile_rma_ls
getattr(ma, "[.list.rma")              # -> ma.slice_rma

getattr(ma, "cooks.distance.rma.mv") is ma.cooks_distance   # True
hasattr(ma, "no.such.method")                                # False
```

Unknown names still raise `AttributeError`, so `hasattr()` and star-imports
behave normally. Two families resist a literal mapping: the R replacement
functions (`$<-`, `[<-`) have no Python call syntax — attribute assignment
(`x.b = ...`) works because results are ordinary mutable objects, but there is no
in-place item assignment, so rebuild the object instead; and `print.*` maps to
`summary` / `print_summary` — the multi-block R console layout is not reproduced
byte-for-byte.

### 2. Where numbers can differ

| Family | Agreement with R | Why |
|---|---|---|
| `escalc` / effect sizes | bit-identical | closed-form arithmetic |
| `rma`, `rma.uni`, τ² estimators | < 1e-12 | same iterative schemes |
| `rma.mv`, `rma.glmm` | < 1e-6 | different optimiser (`scipy.optimize` vs `optim`), same objective |
| `rma.ls` coefficients | ~1e-5 | see below |
| `selmodel` (smooth families) | < 1e-6 | same likelihood, different quadrature tolerance |
| `selmodel` (`truncest`, `negexppow`) | ours is higher | R's `optim()` stalls on a plateau; see below |
| `rma.mv` diagnostics / profile | ~1e-8 | finite-difference Hessian and grid refits |
| `netmeta`, `netmetareg`, `netmeasures` | < 1e-6 … 1e-8 | same algebra, different linear-algebra backend |
| `transf.*`, `fmt.*` | < 1e-10 | closed form |

**`rma.ls` is the clearest place pimeta reports a better optimum than R.** metafor
stops `optim()` on a relative *function* tolerance, so the gradient at its
reported estimate sits around $10^{-4}$ (log link) to $10^{-3}$ (identity link).
pimeta adds a damped Newton polish on the free scale parameters and drives the
gradient to about $10^{-9}$. Our log-likelihood is therefore always
$\ge$ R's, and the scale coefficients $\alpha$ can differ in the fifth
significant digit. Both are valid local optima; ours is the more converged one.
The suite pins this as `r.ll >= ref.ll - 1e-9` instead of exact equality.

The same caveat applies to `selmodel`. The truncated and step-function
likelihoods are piecewise constant in the cut-point, so the score is exactly
zero across whole intervals and `optim()` stops on such a plateau — metafor's own
`truncest` and `negexppow` fits are not the maxima of their own objectives.
pimeta adds one derivative-free restart, so its log-likelihood is $\ge$ R's for
these types too. The smooth families (`beta`, `halfnorm`, `negexp`, `logistic`,
`power`, the `*1`/`*2` variants, `stepfun`, `stepcon`) agree to about $10^{-6}$,
the residual being R's adaptive-quadrature tolerance.

### 3. Deliberate non-portings

| R behaviour | pimeta |
|---|---|
| `metabias(..., plotit=TRUE)` draws a radial plot as a side effect | `plotit` is accepted for signature parity but draws nothing — call `ma.radial()` / `ma.regplot()` yourself. Fitting and testing are unaffected. |
| `netmeta::pairwise(TE=, seTE=)` passes contrast-level data straight through | not implemented; arm-level `treat` / `n` / `event` / `mean` / `sd` input is fully supported |
| `netmatrix(..., ties.method=)` tie handling | fixed at `"first"`; only matters when reference selection hits an exact tie |
| `dfround()` returns an R `data.frame` | returns a `dict` of columns, or a pandas `DataFrame` when pandas is installed |
| `print.*` reproduce the multi-block R console layout | `str(x)` / `summary_text(x)` return a compact block; the exact R layout is not reproduced |
| `options()`, `metafor::setmfopt()`, `meta::setvals()` are global state | `ma.setvals()` / `ma.gs()` exist for API parity, but every function also takes its options as explicit arguments — no silent carry-over between calls |
| `set.seed()` seeds a global RNG | pass `seed=` to the routines that actually draw random numbers: `gosh`, `permutest`, `hettest`, `simulate`, `regplot`, and `netmatrix(ties_method="random")` |
| `profile.rma.uni.selmodel` fixes τ² at the full-model estimate on every grid point | τ² is re-estimated at each point, which is the proper profile likelihood; the curve can differ in the third or fourth digit |
| `metafor::news()` (changelog viewer) | not implemented — no analytic content |

---

## Where pimeta goes beyond R

A faithful port is the floor, not the ceiling. These are places where the
Python version is genuinely better suited to the task than the R original.

**No runtime to install.** `pip install pimeta` pulls only NumPy, SciPy and
matplotlib. No R, no rpy2 bridge, no compilation step — the same code runs on a
laptop, in CI, in a container, in AWS Lambda, and inside Pyodide/WASM in a
browser.

**Results are plain Python objects.** Every fitted model is a dataclass with the
R field names, so `vars(x)`, `x.to_dict()` and `json.dumps(x.to_dict())` all
work. Shipping a meta-analysis to a web front end is a one-liner, not an
S3-marshalling exercise.

```python
r = ma.rma(yi, vi, method="REML")
payload = r.to_dict()          # plain dict of scalars / arrays
```

**Plots are `matplotlib.Axes`, not side effects.** Every figure function returns
the `Axes` it drew on, so you can compose panels with `plt.subplots`, restyle
any artist, or hand the figure to `savefig`, `FigureCanvasAgg`, a notebook, or a
report builder. Pass `filename=` only when you want a file written.

```python
fig, axes = plt.subplots(1, 2, figsize=(11, 4))
ma.forest(yi, vi, ax=axes[0])          # composition instead of a global device
ma.funnel(yi, vi, ax=axes[1])
```

**List-like results slice natively.** `leave1out`, `influence`, `cumul` and
`metacum` results implement `__getitem__`, so R's `x[i]` / `head()` / `tail()`
idioms work directly on every per-study vector at once, with R's semantics
(`x[2]` returns a length-1 *result*, not a bare scalar):

```python
loo = ma.leave1out(yi, vi)
loo[0:3]          # first three rows, all vectors sliced together
loo[[1, 3]]       # arbitrary rows
loo[mask]         # boolean selection
ma.head(loo, 6)
```

**Tighter convergence, reported honestly.** Where metafor's optimiser stops early
(`rma.ls`), pimeta keeps polishing and lands on a higher likelihood — and the
test suite asserts the inequality rather than papering over the difference.

**Numerical care where R relies on doubles being enough.** The Schwarzer
hypergeometric mean/variance is evaluated in log space with `gammaln`, so
large-table binary tests do not overflow; several profile and grid routines are
written in log-likelihood space for the same reason.

**Explicit randomness.** Stochastic procedures accept `seed=` (an integer or a
`numpy.random.Generator`) and draw from a local generator instead of mutating a
process-global stream, so two analyses can run concurrently without stepping on
each other.

**R names still work when you need them.** The `r_methods()` map plus the
module-level `__getattr__` mean a script translated line-by-line from R keeps
running, without giving up idiomatic Python names for everyone else.

---

## Ecosystem integration

Compatibility is implemented as **protocols**, not as one-off adapters, so a
single implementation serves many downstream tools. Everything lives in
`pimeta.interop`, every optional import is lazy, and `import pimeta` still needs
only NumPy / SciPy / matplotlib.

| Protocol | What you call | What it unlocks |
|---|---|---|
| tidy tables (broom) | `r.tidy()`, `r.glance()`, `r.augment()` | one row per coefficient / model / study — any pandas, polars or CSV pipeline |
| estimator (scikit-learn) | `MetaRegressor().fit(X, y, sample_weight=1/vi)` | `Pipeline`, `GridSearchCV`, `cross_val_score`, `clone`, `joblib` / `dask` / `ray`, `optuna`, `mlflow`, `shap` |
| results (statsmodels) | `as_results(r).conf_int()` | `summary_col`, report builders, anything that reads `params` / `bse` / `pvalues` |
| frames | `r.to_table(backend=...)` | `dict`, `records`, `pandas`, `polars`, `pyarrow` |
| labelled arrays | `to_xarray(n)` | `DataArray` / `Dataset` with `treat1` / `treat2` dims |
| graphs | `to_networkx(n)` | centrality, communities, layouts, GraphML / GEXF export |

```python
import pimeta as ma
from pimeta.interop import MetaRegressor, as_results, to_networkx

r = ma.rma(yi, vi, mods=dose, method="REML")
r.tidy()      # term / estimate / std.error / statistic / p.value / conf.low / conf.high
r.glance()    # tau2, I2, QE, QEp, QM, logLik, AIC, BIC, ...
r.augment()   # one row per study, plus .hat / .std.resid / .cooksd

print(as_results(r).conf_int(alpha=0.05))

est = MetaRegressor().fit(dose, yi, sample_weight=1 / vi)
est.predict(new_dose, interval="prediction", vi=new_vi)
```

Ask an object what it supports instead of guessing:

```python
ma.interop.protocol_report(r)
# {'estimator': False, 'tidy': True, 'statsmodels': False, ...}
ma.interop.available()      # which optional backends are installed
```

Backends are opt-in: `pip install pimeta[pandas]`, `pimeta[polars]`,
`pimeta[arrow]`, `pimeta[sklearn]`, `pimeta[statsmodels]`, `pimeta[xarray]`,
`pimeta[networkx]` — or `pimeta[interop]` for all of them.

### Data frames and formulas

Every data-consuming entry point also accepts a **frame plus column names**, and
`mods` accepts an R-style **formula**, so R code translates without unpacking
columns by hand:

```python
df = {"yi": yi, "vi": vi, "dose": dose, "arm": arm, "slab": slab}

r = ma.rma("yi", "vi", mods="~ dose + arm", data=df)
d = ma.escalc("RR", "ai", "bi", "ci", "di", data=df)
```

Accepted frames: `dict`, pandas, polars, NumPy structured arrays and lists of
records. Formulas support `a + b`, `a*b`, `a:b`, `0 + a` / `- 1` and `~ .`;
string columns are dummy-coded the way R does it (first level dropped, `arm[B]`
naming), so the coefficients line up with a `metafor` fit.

---

## Examples

`examples/bcg_meta_analysis.py` is a complete runnable script. It writes the 13
BCG vaccine trials to a CSV, recomputes the effect sizes with `escalc()`, pools
them with `rma()`, draws a forest plot, runs Egger's test and trim-and-fill, fits
a meta-regression on publication year, and finishes with a small network
meta-analysis — checking the numbers against the values R returns for
`metafor::dat.bcg` at every step.

```bash
python examples/bcg_meta_analysis.py
```

### Notebooks (already executed)

Four Jupyter notebooks ship with their outputs and figures baked in, so they
read like a tutorial on GitHub without running anything:

| Notebook | What it shows |
|---|---|
| [`01-quickstart.ipynb`](examples/01-quickstart.ipynb) | 2×2 counts → `escalc()` → `rma()` → heterogeneity → prediction interval → forest plot → `tidy()` table |
| [`02-meta-regression.ipynb`](examples/02-meta-regression.ipynb) | moderator, subgroup split, Egger's test, trim-and-fill, funnel and Baujat plots |
| [`03-network-meta-analysis.ipynb`](examples/03-network-meta-analysis.ipynb) | `netmeta()`: n×n effect matrix, P-score ranking, league table, direct vs indirect |
| [`04-ecosystem.ipynb`](examples/04-ecosystem.ipynb) | pandas in/out, scikit-learn `Pipeline` + `GridSearchCV`, xarray and networkx export |

They are generated and really run by `tools/build_example_notebooks.py` (no
Jupyter needed), and `tests/test_example_notebooks.py` re-executes every cell so
they cannot go stale.

---

## Documentation

A full documentation site lives in [`docs/`](docs/index.md) and is published
from it:

- [Quick start](docs/quickstart.md) — a complete analysis in 15 minutes
- [Coming from R](docs/migration-from-r.md) — line-by-line translation guide
- [Guides](docs/guide/pooling.md) — one page per task
- [API reference](docs/api/index.md) — all 390 public names, generated from the
  docstrings by `python tools/gen_api_docs.py`
- [中文说明](docs/zh/index.md)

Build it locally with `pip install pimeta[docs] && mkdocs serve`.

In-line help is always available:

```python
import pimeta as ma

print(ma.__version__)
help(ma.rma)
help(ma.netmeta)
help(ma.escalc)
```

Every public symbol carries a docstring that names the R function it mirrors.

- Source and issues: <https://github.com/ChengkeChou/pimeta>
- Package index: <https://pypi.org/project/pimeta/>
- Changelog: <https://github.com/ChengkeChou/pimeta/blob/main/CHANGELOG.md>

---

## License

MIT. This package contains an independent Python implementation; it does not
contain or link against R or any GPL R package source code.

## Citation

If you use this package in academic work, please also cite the original R
packages and methods:

- Viechtbauer, W. (2010). Conducting meta-analyses in R with the metafor
  package. *Journal of Statistical Software*, 36(3), 1–48.
- Balduzzi, S., Rücker, G., Nikolakopoulou, A., Papakonstantinou, T.,
  Salanti, G., Efthimiou, O., Schwarzer, G. (2023). netmeta: An R package for
  network meta-analysis using frequentist methods. *Journal of Statistical
  Software*, 106(5), 1–40.
- Balduzzi, S., Rücker, G., Schwarzer, G. (2019). How to perform a
  meta-analysis with R: a practical tutorial. *Evidence-Based Mental Health*,
  22, 153–160.

---

## 中文说明

**pimeta 是 R 生态里 `metafor` + `netmeta` + `meta` 三个包的纯 Python 复现。**
不装 R、不用 rpy2、不依赖任何第三方 meta 分析库，数值引擎是用 NumPy/SciPy 直接转写的，
结果与 R 逐位对齐（多数函数误差 < 1e-12）。

```python
import pimeta as ma

# 效应量
d = ma.escalc(measure="RR", ai=[10, 20, 15], bi=[90, 80, 85],
              ci=[12, 18, 14], di=[88, 82, 86])

# 合并 + 异质性
r = ma.rma(d.yi, d.vi, method="REML")
print(r.b, r.se, r.tau2, r.I2, r.QE, r.QEp)

# 森林图
ma.forest(d.yi, d.vi, studlab=["A", "B", "C"], pooled=r,
          effect_type="RR", filename="forest.png")

# 网状 meta 分析
n = ma.netmeta(TE, seTE, treat1, treat2, studlab, sm="MD")
print(ma.netrank(n).ranking_random)   # 各处理 P-score

# 发表偏倚
print(ma.regtest(d.yi, d.vi).p)     # Egger
print(ma.trimfill(d.yi, d.vi).k0)   # 剪补法
```

**结果对象的字段名和 R 完全一致**，R 代码里的 `r$b`、`r$tau2`、`r$I2`、
`r$ci.lb` 在 Python 里就是 `r.b`、`r.tau2`、`r.I2`、`r.ci_lb`，
已有的 R 脚本和教材可以近乎一对一翻译。

**几个需要知道的约定**（都是为了和 R 保持一致）：

| 约定 | 说明 |
|---|---|
| `r.I2` 是**百分数** | `92.22` 而不是 `0.9222`，和 R 的 `r$I2` 相同；`H2`、`R2` 同理 |
| `confint(r)` 返回 **dict** | `{"tau2", "ci_lb", "ci_ub", "level"}`，对应 R 的 `confint.rma()` 给出的 τ² 置信区间 |
| `r.b` 永远是**数组** | 即使只有截距也是长度为 1 的 `np.ndarray`，取值用 `float(r.b[0])` |
| `escalc()` 对 RR/OR/IRR 返回**对数尺度** | 要报告比值本身请用 `math.exp()` |
| 画图返回 matplotlib `Axes` | 加 `filename="x.png"` 才会落盘；后端强制 `Agg`，脚本里不会弹窗卡住 |

完整可运行示例见 [`examples/bcg_meta_analysis.py`](examples/bcg_meta_analysis.py)：从 CSV 读入 13 个卡介苗试验 → 算效应量 → 合并 → 森林图 → Egger / 剪补法 →
元回归 → 网状 meta 分析，每一步都和 R 的 `metafor::dat.bcg` 结果对拍。

#### 已经跑好的 notebook（打开就能看结果和图）

| Notebook | 内容 |
|---|---|
| [`01-quickstart.ipynb`](examples/01-quickstart.ipynb) | 2×2 计数 → `escalc()` → `rma()` → 异质性 → 预测区间 → 森林图 → `tidy()` 表 |
| [`02-meta-regression.ipynb`](examples/02-meta-regression.ipynb) | 调节变量、亚组、Egger 检验、剪补法、漏斗图与 Baujat 图 |
| [`03-network-meta-analysis.ipynb`](examples/03-network-meta-analysis.ipynb) | `netmeta()`：n×n 效应矩阵、P-score 排名、联赛表、直接 vs 间接证据 |
| [`04-ecosystem.ipynb`](examples/04-ecosystem.ipynb) | pandas 进出、scikit-learn `Pipeline` + `GridSearchCV`、xarray / networkx 导出 |

由 `tools/build_example_notebooks.py` 真实执行后生成（不需要装 Jupyter），
`tests/test_example_notebooks.py` 会重跑每个单元格，保证示例不会过期。

覆盖范围：`meta` 59/59、`netmeta` 38/38、`metafor` 278/279 个导出函数
（唯一未实现的是 `metafor.news()`，它只是查看更新日志，不含分析功能）。

### 和 R 不一样的地方

**分析功能没有缺口。** 下面每一条差异，要么是 Python 语法强制的，要么是我们
故意没有照抄的 R 怪癖。其余部分全部与 R 对拍过（见上表）。

#### 一、写不出来的名字

R 把类名编进函数名（`print.rma.uni`、`cooks.distance.rma.mv`）。Python 里
`.` 是属性访问，这些字符串**不是合法标识符**；`[`、`$`、`[<-` 更是运算符，
连标识符都构不成。功能全在，只是写法变了。

```python
import numpy as np
import pimeta as ma

yi, vi = [0.20, 0.40, 0.10, 0.50], [0.04, 0.05, 0.03, 0.06]   # 任意一组数据
z = [1, 2, 3, 4]                                               # 任意调节变量

# 1) R 的 generic.class 名字 → 可直接解析
getattr(ma, "cooks.distance.rma.mv")   # -> ma.cooks_distance
getattr(ma, "print.rma.uni")           # -> ma.summary
getattr(ma, "profile.rma.ls")          # -> ma.profile_rma_ls
getattr(ma, "[.list.rma")              # -> ma.slice_rma
ma.r_methods()                         # 完整的 66 条映射表

# 2) R 的 x[i] / head / tail → 原生切片
loo = ma.leave1out(yi, vi)
loo[0:3]      # 所有逐研究向量同步切
loo[2]        # 长度 1 的结果（R 语义，不是裸标量）
loo[[1, 3]]   # 指定行
loo[mask]     # 布尔掩码
ma.head(loo, 6); ma.tail(loo, 6)

# 3) R 的 x$b <- 0 → 直接属性赋值（结果对象是普通的可变对象）
x = ma.rma(yi, vi)
x.b = np.array([0.0])

# 4) R 的 update(x, mods = ~z) → 重新拟合
x = ma.update_rma(ma.rma(yi, vi), mods=z)
```

未映射的名字仍然抛 `AttributeError`，`hasattr()` 和 `from pimeta import *`
行为正常。只有两类无法字面对应：R 的替换函数（`$<-`、`[<-`）在 Python 里没有
调用语法——属性赋值（`x.b = ...`）可用，因为结果是普通的可变对象，但没有原地
下标赋值，需要重建对象；`print.*` 映射到 `summary` / `print_summary`，R 那个
多段式控制台排版没有逐字节照抄。

#### 二、数值可能不同的地方

| 系列 | 与 R 的差异 | 原因 |
|---|---|---|
| `escalc` / 效应量 | 逐位相同 | 闭式算术 |
| `rma`、`rma.uni`、τ² 估计 | < 1e-12 | 同一套迭代格式 |
| `rma.mv`、`rma.glmm` | < 1e-6 | 优化器不同（`scipy.optimize` vs `optim`），目标函数相同 |
| `rma.ls` 的 α 系数 | ~1e-5 | 见下 |
| `selmodel`（平滑家族） | < 1e-6 | 同一个似然，自适应积分容差不同 |
| `selmodel`（`truncest`、`negexppow`） | 我们的更高 | R 的 `optim()` 停在平台区；见下 |
| `rma.mv` 诊断量 / profile | ~1e-8 | 数值 Hessian 与网格重拟合 |
| `netmeta`、`netmetareg`、`netmeasures` | < 1e-6 … 1e-8 | 同一套代数，线性代数后端不同 |
| `transf.*`、`fmt.*` | < 1e-10 | 闭式 |

**`rma.ls` 是最明显的一处：我们的解比 R 更收敛。** metafor 用相对**函数值**
容差停 `optim()`，所以它报告的估计处梯度在 $10^{-4}$（log 连接）到
$10^{-3}$（identity 连接）量级。pimeta 额外做了一步阻尼牛顿抛光，把梯度压到
$10^{-9}$ 左右。因此我们的对数似然**恒 ≥ R**，α 系数可能在第五位有效数字上
不同。两者都是合法的局部最优，我们的是收敛更彻底的那个。测试里钉的是
`r.ll >= ref.ll - 1e-9`，而不是精确相等。

`selmodel` 也有同样的情形。截尾与阶梯类型的似然对切点是**分段常数**，整段区间
上得分函数恰好为零，`optim()` 就停在这样的平台上——metafor 自己的 `truncest`
与 `negexppow` 结果并非其目标函数的极大点。pimeta 在此基础上加了一次导数
无关的重启，所以这两类模型的对数似然也**恒 ≥ R**。平滑家族（`beta`、
`halfnorm`、`negexp`、`logistic`、`power`、`*1`/`*2` 变体、`stepfun`、
`stepcon`）与 R 的差异在 $10^{-6}$ 量级，残差来自 R 自适应积分的容差。

#### 三、故意不移植的部分

| R 的行为 | pimeta 的处理 |
|---|---|
| `metabias(..., plotit=TRUE)` 顺带画 radial 图 | `plotit` 为签名兼容而保留，但不画图；请自己调 `ma.radial()` / `ma.regplot()`。拟合与检验不受影响 |
| `netmeta::pairwise(TE=, seTE=)` 直接透传对照级数据 | 未实现；臂级 `treat` / `n` / `event` / `mean` / `sd` 输入完整支持 |
| `netmatrix(..., ties.method=)` 的并列处理 | 固定为 `"first"`；只在参照选择出现精确并列时才有影响 |
| `dfround()` 返回 R `data.frame` | 返回列的 `dict`；装了 pandas 时返回 `DataFrame` |
| `print.*` 复刻多段式 R 控制台排版 | `str(x)` / `summary_text(x)` 给出紧凑排版；R 的原样版式未复刻 |
| `options()`、`setmfopt()`、`setvals()` 是全局状态 | `ma.setvals()` / `ma.gs()` 为 API 兼容而存在，但每个函数同时也接受显式参数——调用之间不会有隐式状态串味 |
| `set.seed()` 种的是全局随机流 | 随机过程用 `seed=` 参数 + 局部 `Generator` |
| `profile.rma.uni.selmodel` 在每个格点固定 τ² | 每个格点**重新估计** τ²，这是标准的 profile 似然；曲线可能有三、四位小数的差异 |
| `metafor::news()`（看更新日志） | 未实现——不含分析功能 |

### 比 R 更好的地方

**1. 零运行时依赖。** `pip install pimeta` 只拉 NumPy / SciPy / matplotlib。
不装 R、不架 rpy2、不需要编译——笔记本、CI、容器、AWS Lambda、甚至浏览器里的
Pyodide/WASM 跑的都是同一份代码。

**2. 结果是普通的 Python 对象。** 每个拟合结果都是 dataclass，字段名照抄 R，
所以 `vars(x)`、`x.to_dict()`、`json.dumps(x.to_dict())` 直接可用。把一次
meta 分析接到 Web 前端是一行代码，而不是一场 S3 对象封送。

```python
r = ma.rma(yi, vi, method="REML")
payload = r.to_dict()          # 标量/数组组成的普通 dict
```

**3. 画图返回 `matplotlib.Axes`，而不是副作用。** 每个绘图函数都把画布返回，
于是你可以用 `plt.subplots` 拼版、改任何 artist、交给 `savefig` / 报告生成器 /
notebook。只有想落盘时才传 `filename=`。

```python
fig, axes = plt.subplots(1, 2, figsize=(11, 4))
ma.forest(yi, vi, ax=axes[0])
ma.funnel(yi, vi, ax=axes[1])
```

**4. 列表型结果原生可切片。** `leave1out` / `influence` / `cumul` / `metacum`
实现了 `__getitem__`，R 的 `x[i]`、`head()`、`tail()` 直接可用，且保持 R 的语义
（`x[2]` 返回长度 1 的**结果对象**，不是裸标量）。

**5. 收敛更彻底，而且如实报告。** metafor 提前停下的地方（`rma.ls`），pimeta
继续抛光、落到更高的似然上——测试断言的是不等式，而不是把差异掩盖过去。

**6. 在 R 靠 double 硬扛的地方做了数值处理。** Schwarzer 检验的超几何均值/方差
在对数空间用 `gammaln` 计算，大表二分类数据不会溢出；若干 profile 与网格过程
同样在对数似然空间里写。

**7. 随机性显式化。** 随机过程接受 `seed=`（整数或 `numpy.random.Generator`），
用局部生成器抽随机数，不改进程全局随机流，两个分析可以并发跑而互不干扰。

**8. 需要时 R 名字照样能用。** `r_methods()` + 模块级 `__getattr__` 让逐行翻译
过来的 R 脚本继续跑，同时不妨碍其他人用 Pythonic 的名字。

### 生态对接：用协议，不用适配器

兼容是按**协议**实现的，不是给每个库单独写适配器——写一次，多个生态都能用。
全部放在 `pimeta.interop`，可选依赖全是懒加载，`import pimeta` 依然只需要
NumPy / SciPy / matplotlib。

| 协议 | 调用 | 解锁什么 |
|---|---|---|
| tidy 表格（broom） | `r.tidy()`、`r.glance()`、`r.augment()` | 每个系数 / 每个模型 / 每个研究一行，任何 pandas / polars / CSV 流程都能接 |
| 估计器（sklearn） | `MetaRegressor().fit(X, y, sample_weight=1/vi)` | `Pipeline`、`GridSearchCV`、`cross_val_score`、`clone`、`joblib`/`dask`/`ray`、`optuna`、`mlflow`、`shap` |
| 结果对象（statsmodels） | `as_results(r).conf_int()` | `summary_col`、报告生成器、任何读 `params`/`bse`/`pvalues` 的代码 |
| 表格后端 | `r.to_table(backend=...)` | `dict`、`records`、`pandas`、`polars`、`pyarrow` |
| 带标签数组 | `to_xarray(n)` | 带 `treat1`/`treat2` 维的 `DataArray`/`Dataset` |
| 图 | `to_networkx(n)` | 中心性、社区发现、布局、GraphML/GEXF 导出 |

```python
import pimeta as ma
from pimeta.interop import MetaRegressor, as_results, to_networkx

r = ma.rma(yi, vi, mods=dose, method="REML")
r.tidy()      # term / estimate / std.error / statistic / p.value / conf.low / conf.high
r.glance()    # tau2、I2、QE、QEp、QM、logLik、AIC、BIC ……
r.augment()   # 每个研究一行，附 .hat / .std.resid / .cooksd

est = MetaRegressor().fit(dose, yi, sample_weight=1 / vi)
est.predict(new_dose, interval="prediction", vi=new_vi)

ma.interop.protocol_report(r)   # 这个对象支持哪些协议
ma.interop.available()          # 装了哪些可选后端
```

按需安装后端：`pip install pimeta[pandas]`、`pimeta[polars]`、`pimeta[arrow]`、
`pimeta[sklearn]`、`pimeta[statsmodels]`、`pimeta[xarray]`、`pimeta[networkx]`，
或一次装齐 `pimeta[interop]`。

**数据框与公式。** 所有吃数据的入口都支持 **frame + 列名**，`mods` 还额外支持
R 风格公式，R 代码不用手工拆列：

```python
df = {"yi": yi, "vi": vi, "dose": dose, "arm": arm, "slab": slab}

r = ma.rma("yi", "vi", mods="~ dose + arm", data=df)
d = ma.escalc("RR", "ai", "bi", "ci", "di", data=df)
```

支持的 frame：`dict`、pandas、polars、NumPy 结构化数组、记录字典列表。
公式支持 `a + b`、`a*b`、`a:b`、`0 + a` / `- 1`、`~ .`；字符串列按 R 的方式
做哑变量编码（丢弃第一层，列名 `arm[B]`），所以系数与 `metafor` 对齐。
