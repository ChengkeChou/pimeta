# Coming from R

pimeta mirrors the R packages closely enough that most code translates
mechanically. This page lists every construct where it does not.

## Library names

| R | pimeta |
|---|---|
| `library(metafor)` | `import pimeta as ma` |
| `library(meta)` | `import pimeta as ma` |
| `library(netmeta)` | `import pimeta as ma` |

One namespace, three packages. Functions keep the R spellings where they are
valid Python: `escalc`, `rma`, `rma_mv`, `metabin`, `netmeta`, `forest`.

## Object fields

R's `$` becomes `.`:

```python
r = ma.rma(yi, vi, method="REML")

r.b          # r$b
r.se         # r$se
r.zval       # r$zval
r.pval       # r$pval
r.ci_lb      # r$ci.lb
r.ci_ub      # r$ci.ub
r.tau2       # r$tau2
r.I2         # r$I2
r.QE, r.QEp  # r$QE, r$QEp
```

Dots inside a field name become underscores: `ci.lb` → `ci_lb`,
`pval.random` → `pval_random`.

## Method names

R encodes the class in the function name. Those strings are not valid Python
identifiers, so pimeta exposes plain names **and** bridges the R spellings:

```python
import pimeta as ma

r = ma.rma(yi, vi)

r.summary()                                # summary(r)
ma.cooks_distance(r)                       # cooks.distance(r)
ma.begg_test(r)                            # ranktest(r)
ma.hc_rma(r)                               # hc(r)

# Mechanical translation: resolve R method names directly.
getattr(ma, "cooks.distance.rma.mv")       # -> ma.cooks_distance
getattr(ma, "print.rma.uni")               # -> ma.summary
getattr(ma, "[.list.rma")                  # -> ma.slice_rma
```

`ma.r_methods()` returns the full map; see the
[R method-name map](reference/r-method-map.md).

## Subsetting and `head`/`tail`

`[.list.rma`, `head.list.rma` and `tail.list.rma` slice every per-study vector
at once so rows stay aligned. Python results implement `__getitem__`:

```python
loo = ma.leave1out(yi, vi)

loo[0:3]        # 1:3  (0-based in Python)
loo[2]          # a length-1 result, not a bare number
loo[[1, 3]]     # c(2, 4)
loo[mask]       # logical index
ma.head(loo, 6) # head(x)
ma.tail(loo, 6) # tail(x)
```

## Formulas and `data=`

R evaluates arguments inside a data frame. Python has no non-standard
evaluation, so pass a **column name** where a vector is expected and hand the
frame in with `data=`:

```python
import pimeta as ma

df = {"yi": yi, "vi": vi, "dose": dose, "arm": arm}

ma.rma("yi", "vi", data=df)                       # rma(yi, vi, data=df)
ma.rma("yi", "vi", mods="~ dose + arm", data=df)  # mods = ~ dose + arm
ma.escalc("RR", "ai", "bi", "ci", "di", data=df)  # escalc(..., data=df)
```

Formulas support `a + b`, `a*b`, `a:b`, `0 + a` / `- 1`, `~ .` and backticked
names. String columns are dummy-coded exactly as R's `model.matrix` does it
(first level dropped, `arm[B]` naming), so coefficients line up.

Two things cannot be translated:

- **`update(x, . ~ . + z)`** — R's dot shorthand has no Python equivalent. Pass
  the new moderators explicitly: `ma.update_rma(x, mods=new_mods)`.
- **`x$b <- value`** — attribute assignment works (`x.b = ...`) because results
  are ordinary mutable objects, but there is no in-place item assignment on
  list-style results.

## Global state

| R | pimeta |
|---|---|
| `options(...)`, `metafor::setmfopt()` | pass arguments explicitly; `ma.setvals()` / `ma.gs()` exist for `meta::settings` parity |
| `set.seed(42)` | pass `seed=42` to `gosh`, `permutest`, `hettest`, `simulate`, `regplot`, `netmatrix(ties_method="random")` |

Nothing in pimeta silently changes the behaviour of a later call.

## Printing

| R | pimeta |
|---|---|
| `print(x)` | `print(x)` — `str(x)` gives the text block |
| `summary(x)` | `x.summary()` — returns the result object with a richer `str` |
| `print(summary(x))` | `ma.print_summary(x)` |
| `as.data.frame(x)` | `ma.as_data_frame_rma_list(x)` or `x.to_table()` |
| `tidy(x)`, `glance(x)`, `augment(x)` (broom) | `x.tidy()`, `x.glance()`, `x.augment()` |

## Conventions that differ

These match R on purpose, and surprise people on first contact:

| Convention | Detail |
|---|---|
| `I2`, `H2`, `R2` are **percentages** | `92.22`, not `0.9222` |
| `b` is always an **array** | even intercept-only models; use `float(r.b[0])` |
| `confint(r)` returns a **dict** | `{"tau2", "ci_lb", "ci_ub", "level"}` — the τ² interval |
| `escalc` returns **log scale** for ratio measures | back-transform with `math.exp()` |

## Differences in results

Everything is verified against R; the few places where numbers can differ (and
the one place pimeta converges further) are listed in
[Differences from R](reference/differences.md).
