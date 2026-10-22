# Differences from R

Nothing in the analytic surface is missing. Every deviation below is either
forced by Python's grammar or a place where an R quirk was deliberately not
reproduced. Everything else is verified against R — see
[Numerical validation](#numerical-validation).

## 1. Names that cannot exist in Python

R encodes the class inside the function name (`print.rma.uni`,
`cooks.distance.rma.mv`). In Python `.` is attribute access, so those strings are
not identifiers; `[`, `$` and `[<-` are operators, not names at all. The
behaviour is fully implemented — only the spelling changes.

| R | pimeta |
|---|---|
| `print(x)` / `summary(x)` | `print(x)` / `x.summary()`; `str(x)` and `summary_text(x)` give the text block |
| `cooks.distance(x)`, `ranktest(x)`, `hc(x)` | `ma.cooks_distance(x)`, `ma.begg_test(x)`, `ma.hc_rma(x)` |
| `x[1:3]`, `x[[1,3]]`, `head(x)`, `tail(x)` | `x[0:3]`, `x[[1, 3]]`, `ma.head(x)`, `ma.tail(x)` |
| `x$ci.lb`, `x$tau2` | `x.ci_lb`, `x.tau2` |
| `x$b <- 0` | `x.b = np.array([0.0])` — results are plain mutable objects |
| `update(x, mods = ~z)` | `ma.update_rma(x, mods=z)` — refits, like R |
| `as.data.frame(x)` | `ma.as_data_frame_rma_list(x)` or `x.to_dict()` |

`ma.r_methods()` returns the full 66-entry map from R method name to pimeta
callable, and a module-level `__getattr__` resolves the R spelling directly:

```python
import pimeta as ma

getattr(ma, "cooks.distance.rma.mv") is ma.cooks_distance   # True
getattr(ma, "print.rma.uni")           is ma.summary        # True
getattr(ma, "[.list.rma")              is ma.slice_rma      # True
hasattr(ma, "no.such.method")                               # False
```

Two families resist a literal mapping: the R replacement functions (`$<-`,
`[<-`) have no Python call syntax — attribute assignment works because results
are ordinary mutable objects, but there is no in-place item assignment, so
rebuild the object instead; and `print.*` maps to `summary` / `print_summary`,
where the multi-block R console layout is not reproduced byte-for-byte.

## 2. Where numbers can differ

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
reported estimate sits around 1e-4 (log link) to 1e-3 (identity link). pimeta
adds a damped Newton polish on the free scale parameters and drives the gradient
to about 1e-9. Our log-likelihood is therefore always ≥ R's, and the scale
coefficients α can differ in the fifth significant digit. Both are valid local
optima; ours is the more converged one. The suite pins this as
`r.ll >= ref.ll - 1e-9` instead of exact equality.

The same caveat applies to `selmodel`. The truncated and step-function
likelihoods are piecewise constant in the cut-point, so the score is exactly
zero across whole intervals and `optim()` stops on such a plateau; metafor's own
`truncest` and `negexppow` fits are not the maxima of their own objectives.
pimeta adds one derivative-free restart, so its log-likelihood is ≥ R's for
these types as well. The smooth families (`beta`, `halfnorm`, `negexp`,
`logistic`, `power`, the `*1`/`*2` variants, `stepfun`, `stepcon`) agree to
about 1e-6, the residual being R's adaptive-quadrature tolerance.

## 3. Deliberate non-portings

| R behaviour | pimeta |
|---|---|
| `metabias(..., plotit=TRUE)` draws a radial plot as a side effect | `plotit` is accepted for signature parity but draws nothing — call `ma.radial()` / `ma.regplot()` yourself |
| `netmeta::pairwise(TE=, seTE=)` passes contrast-level data straight through | not implemented; arm-level `treat` / `n` / `event` / `mean` / `sd` input is fully supported |
| `netmatrix(..., ties.method=)` tie handling | fixed at `"first"`; only matters when reference selection hits an exact tie |
| `dfround()` returns an R `data.frame` | returns a `dict` of columns, or a pandas `DataFrame` when pandas is installed |
| `print.*` reproduce the multi-block R console layout | `str(x)` / `summary_text(x)` return a compact block |
| `options()`, `metafor::setmfopt()`, `meta::setvals()` are global state | `ma.setvals()` / `ma.gs()` exist for API parity, but every function also takes its options as explicit arguments |
| `set.seed()` seeds a global RNG | pass `seed=` to the routines that draw random numbers: `gosh`, `permutest`, `hettest`, `simulate`, `regplot`, `netmatrix(ties_method="random")` |
| `profile.rma.uni.selmodel` fixes τ² at the full-model estimate on every grid point | τ² is re-estimated at each point, which is the proper profile likelihood |
| `metafor::news()` (changelog viewer) | not implemented — no analytic content |

### `mada` is not ported

`pimeta.dta` implements the **methods** — bivariate random-effects model, HSROC,
SROC, AUC / partial AUC — not `mada`'s function surface. The package's scope is
`meta`, `netmeta` and `metafor`; adding a fourth dependency on another package's
API would make "which numbers should match R" unanswerable, and several `mada`
exports are display code (`print.*`, `summary.*`, `vcov.*`) that pimeta already
covers with `coef_table()` / `summary_text()` / `to_dict()`.

| `mada` export | pimeta |
|---|---|
| `madad`, `madauni` | not ported — per-outcome and multivariate diagnostics already exist as `rma`, `rma_mv` and `mv_diagnostics` |
| `print.*`, `summary.*`, `vcov.*` | S3 display and accessors; use `coef_table()`, `summary_text()`, `to_dict()`, `.vcov` |
| `talpha`, `CIrho`, `sens`, `spec`, `fpr`, `SummaryPts` | numbers, not names: `ReitsmaResult` already carries `summary_sens`, `summary_spec`, `summary_fpr`, `sens_ci`, `spec_ci`, `fpr_ci`, `rho` |
| `cochran.Q` | use `rma_mv` on the prepared table |
| `phm` (proportional hazards DTA) | not ported — a separate likelihood family, outside the scope above |
| `predv_d`, `predv_r` (posterior-predictive checks) | not ported; the standardised-residual and BLUP diagnostics in `mv_diagnostics` cover the same ground |
| `forest`, `forestmada`, `crosshair`, `ROCellipse`, `mcsroc`, `mslSROC`, `rsSROC` | `plot_forest()`, `plot_sroc()`, `plot_baujat()` — pimeta's plotting convention (returns an `Axes`) |

Two things *are* deliberately different from `mada`, on scientific grounds:
the continuity correction touches only the studies that need it (and warns),
and the model is parameterised on specificity so `rho` is negative under a
threshold effect. `correction_control="all"` and the `*_fpr` mirror fields
reproduce `mada` exactly when that is what you need.

## Numerical validation

The library is developed against a differential test suite: identical datasets
are fed to R (`metafor` 5.2-1, `netmeta` 3.7-0, `meta` 8.5-0) and to pimeta, and
every reported quantity is compared.

Coverage: `meta` 59/59, `netmeta` 38/38 and `metafor` 278/279 exported
functions. The single exception is `metafor.news()`, a changelog viewer with no
analytic content.

Where R has quirks (non-convergence, `NA` subsets, sign conventions of
`contrmat`), the behaviour is reproduced rather than "fixed" — the reasoning
being that a port which silently improves on its reference is impossible to
verify. The one exception is `mada`, which is not a port and is corrected
rather than copied; see above.
