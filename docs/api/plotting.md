# Figures

*25 public names, generated from `pimeta.__all__` and the docstrings in `src/pimeta/`. Do not edit this page by hand — run `python tools/gen_api_docs.py`.*

| Name | Mirrors in R |
|---|---|
| [`plot_mv_profile`](#plot_mv_profile) | plot.profile.rma |
| [`forest`](#forest) | — |
| [`funnel`](#funnel) | — |
| [`netgraph`](#netgraph) | — |
| [`plot_baujat`](#plot_baujat) | — |
| [`plot_specification_curve`](#plot_specification_curve) | — |
| [`plot_gosh`](#plot_gosh) | — |
| [`plot_sroc`](#plot_sroc) | — |
| [`plot_influence`](#plot_influence) | plot.infl.rma.uni |
| [`plot_permutest`](#plot_permutest) | plot.permutest.rma.uni |
| [`plot_selprofile`](#plot_selprofile) | plot.profile.rma |
| [`PlotResult`](#plotresult) | — |
| [`heatplot`](#heatplot) | heatplot.netpath |
| [`regplot`](#regplot) | regplot.rma |
| [`radial`](#radial) | radial |
| [`labbe`](#labbe) | labbe |
| [`llplot`](#llplot) | llplot |
| [`bubble`](#bubble) | regplot |
| [`drapery`](#drapery) | — |
| [`qqnorm_rma`](#qqnorm_rma) | qqnorm.rma.uni |
| [`rankogram`](#rankogram) | rankogram |
| [`netheat`](#netheat) | netheat |
| [`hasse`](#hasse) | hasse |
| [`netposet`](#netposet) | netposet |
| [`traffic_light`](#traffic_light) | traffic_light |

## `plot_mv_profile`

```python
plot_mv_profile(profile, filename=None, figsize=None, title=None, xlab=None, ylab='Log-likelihood', ylim=None, ax=None)
```

*Mirrors R's `plot.profile.rma`.*

``metafor::plot.profile.rma`` for a profiled parameter.

Used for the variance components of an ``rma.mv`` fit
(`.mv_diagnostics.profile_rmamv`) and for the scale coefficients of
a location-scale model (`.rma_ls.profile_rma_ls`); the x grid is
read from ``profile.sigma2`` or, failing that, ``profile.alpha``.

**Args**

- `profile` — anything exposing ``sigma2`` or ``alpha``, plus ``ll``, ``maxll``, ``vc``, ``xlab`` and ``title``.

**Returns**

`PlotResult`.

## `forest`

```python
forest(yi, vi=None, studlab=None, pooled=None, effect_type: 'str | None' = None, log_scale: 'bool | None' = None, null: 'float' = 0.0, xlab: 'str | None' = None, sort_by_effect: 'bool' = False, filename: 'str | None' = None, figsize: 'tuple[float, float] | None' = None, title: 'str | None' = None, ax: 'plt.Axes | None' = None) -> 'plt.Axes'
```

Draw a forest plot.

``yi`` / ``vi`` are per-study effect sizes and sampling variances on the
*analysis* scale (log scale for ratio types). ``pooled`` is an optional
`pooling.PooledResult` (or any object
exposing ``pooled_est``, ``pooled_se`` / ``pooled_ci_lower``,
``pooled_ci_upper`` and ``weights``) used to draw the summary diamond and
size the study boxes by weight.

A diamond needs ``pooled``. Without it only the per-study rows are drawn.

## `funnel`

```python
funnel(yi, vi=None, studlab=None, effect_type: 'str | None' = None, xlab: 'str | None' = None, filename: 'str | None' = None, figsize: 'tuple[float, float]' = (6.0, 6.0), title: 'str | None' = None, ax: 'plt.Axes | None' = None) -> 'plt.Axes'
```

Draw a funnel plot of effect size vs. standard error.

Inverted (metafor) convention: the y axis is the standard error so more
precise studies sit at the top.

## `netgraph`

```python
netgraph(result=None, trts=None, treat1=None, treat2=None, studlab=None, filename: 'str | None' = None, figsize: 'tuple[float, float]' = (7.0, 7.0), title: 'str | None' = None, node_size: 'float' = 900.0, ax: 'plt.Axes | None' = None) -> 'plt.Axes'
```

Draw a network evidence graph.

Accepts either a `NetmetaResult` (reads ``trts``, ``treat1_raw``,
``treat2_raw``, ``studlab_raw``) or explicit ``trts`` / ``treat1`` /
``treat2`` lists. Edge width encodes the number of studies comparing the
pair. Layout is netmeta's default circular arrangement (treatments placed
evenly around a circle, starting from the top).

## `plot_baujat`

```python
plot_baujat(result, studlab=None, filename: 'str | None' = None, figsize: 'tuple[float, float]' = (6.0, 6.0), title: 'str | None' = None, ax: 'plt.Axes | None' = None) -> 'plt.Axes'
```

Draw a Baujat plot (heterogeneity contribution vs. influence).

``result`` is a `BaujatResult` (fields ``x`` and ``y``).

## `plot_specification_curve`

```python
plot_specification_curve(mv, *, filename=None, figsize=None, title=None, mark_default=None) -> 'PlotResult'
```

The specification curve, with the decisions underneath it.

Both kinds of panel share one ordinal axis: the x position is the rank of a
specification in the estimate ordering, so a column can be read straight down
from an interval to every decision that produced it. The upper panel puts the
estimate on the y axis with its interval; the lower ones show, for each
decision, which value each specification took.

Why the margins and not a table: the curve alone invites the reader to
attribute the movement to whichever decision they already suspected. Putting
the values underneath makes a cluster visible instead, and clusters are what
interactions look like.

The first version of this figure plotted the estimate on the x axis with the
specifications stacked on y, and shared that axis with the margins -- so the
margins were drawn at positions 0..M-1 on an axis measured in effect sizes,
and all but the first marker fell outside the view. The tests passed; only
rendering it showed the bug. The markers are now on the ordinal axis they
belong to, and a test pins the shared x range to the number of
specifications.

**Args**

- `mv` — a `multiverse.MultiverseResult`. Anything with ``curve()``, ``decisions`` and ``counts()`` works.
- `filename` — save the figure here (300 dpi); ``None`` leaves it in memory.
- `figsize` — figure size in inches; the default grows with the number of decisions, because a fixed height would crush them together.
- `title` — figure title; defaults to the counts and the range.
- `mark_default` — an optional specification id to highlight in the curve -- the primary analysis inside its own multiverse.

**Returns**

`PlotResult` whose ``ax`` is the upper (curve) panel.

**Raises**

- `ValueError` — when no specification produced a usable estimate, because a specification curve of nothing is not a diagnostic.

## `plot_gosh`

```python
plot_gosh(result, x: 'str' = 'tau2', y: 'str' = 'estimate', filename: 'str | None' = None, figsize: 'tuple[float, float]' = (6.0, 6.0), title: 'str | None' = None, ax: 'plt.Axes | None' = None) -> 'plt.Axes'
```

Draw a GOSH plot (Goodman-Schulz-Hastie heterogeneity).

``result`` is a `GoshResult`; ``x``/``y`` select two of the stored
per-subset vectors: ``estimate``, ``tau2``, ``QE``, ``I2``, ``H2``.

## `plot_sroc`

```python
plot_sroc(fit, studlab=None, curve: 'bool' = True, summary: 'bool' = True, ci: 'bool' = True, show_auc: 'bool' = True, size_by: 'str' = 'diseased', filename: 'str | None' = None, figsize: 'tuple[float, float]' = (6.0, 6.0), title: 'str | None' = None, ax: 'plt.Axes | None' = None) -> 'plt.Axes'
```

Draw an SROC: study points, summary curve and the summary point.

The picture `mada` splits across ``crosshair`` and ``ROCellipse``, drawn in
one axes: one dot per study at (FPR, TPR) with area proportional to the
number of diseased participants, the Rutter-Gatsonis summary curve, and the
back-transformed mean pair as a crosshair whose arms are the confidence
interval on each axis.

**Args**

- `fit` — a `dta.ReitsmaResult` from `dta.reitsma`.
- `studlab` — study labels; defaults to ``Study 1`` ... ``Study k``. Pass ``[]`` to draw no labels.
- `curve` — draw the summary curve.
- `summary` — draw the summary point.
- `ci` — draw the confidence interval on each axis of the summary point.
- `show_auc` — annotate the AUC and the partial AUC.
- `size_by` — ``"diseased"`` (number of participants with the target condition), ``"total"`` or ``"equal"``.
- `filename` — when given, save at 300 dpi.
- `figsize, title, ax` — standard plotting arguments.

**Returns**

The `matplotlib.axes.Axes`.

**Raises**

- `TypeError` — when ``fit`` is not a ``ReitsmaResult``.

**Example**

- `>>> fit = ma.reitsma(counts)            # doctest` — +SKIP
- `>>> ma.plot_sroc(fit)                   # doctest` — +SKIP

## `plot_influence`

```python
plot_influence(x, plotinf=True, plotdfbs=False, logcov=True, crit=None, ncol=4, figsize=None, psize=None, col='black', col_infl='red', slab=None, filename=None)
```

*Mirrors R's `plot.infl.rma.uni`.*

Influence diagnostics as a panel grid (``metafor::plot.infl.rma.uni``).

R draws eight panels against the study index: the externally standardized
residuals, DFFITS, Cook's distance, the covariance ratio, the leave-one-out
tau^2, the leave-one-out QE, the hat value and the weight, each with the
reference lines ``plot.infl.rma.uni`` adds (``+/- qnorm(0.975)`` for the
residuals, ``+/- 3 sqrt(p/(k-p))`` for DFFITS, ``qchisq(0.5, p)`` for Cook,
``1`` for the covariance ratio, the full-model tau^2 and QE for the deletion
panels, ``p/k`` and ``3p/k`` for the hat values and ``100/k`` for the
weights).  Studies metafor flags as influential are drawn in red.

**Args**

- `x` — an `sensitivity.InfluenceResult`.
- `plotinf` — ``True`` for all eight panels, or a list of 1-based panel indices (1 rstudent, 2 dffits, 3 cook.d, 4 cov.r, 5 tau2.del, 6 QE.del, 7 hat, 8 weight).
- `plotdfbs` — also draw one panel per coefficient holding its DFBETAS.
- `logcov` — put the covariance-ratio panel on a log scale (R's default).
- `crit` — residual reference value (default ``qnorm(0.975)``).
- `filename` — optional path to save the figure to.

**Returns**

`PlotResult`.

## `plot_permutest`

```python
plot_permutest(x, btt=None, nbin=None, figsize=None, col='0.8', col_obs='blue', filename=None)
```

*Mirrors R's `plot.permutest.rma.uni`.*

Permutation distributions (``metafor::plot.permutest.rma.uni``).

One panel per coefficient: a histogram of the permutation distribution of
the test statistic with the observed value marked, which is the picture
R draws for ``plot(permutest(res))``.

**Args**

- `x` — a `diagnostics.PermutestResult`.
- `btt` — 1-based indices of the coefficients to draw (default all).
- `nbin` — number of histogram bins (default ``ceiling(sqrt(nperm))``).

**Returns**

`PlotResult`.

## `plot_selprofile`

```python
plot_selprofile(profile, filename=None, figsize=None, title='Profile log-likelihood', xlab='delta', ylab='Log-likelihood', ax=None)
```

*Mirrors R's `plot.profile.rma`.*

``metafor::plot.profile.rma`` for a selection model.

Draws the profiled log-likelihood against the selection parameter
``delta``, marks the maximum, and shades the profile-likelihood interval.

**Args**

- `profile` — a `.selmodel.SelProfileResult` (or anything with ``delta``/``ll``/``maxll``/``ci_lb``/``ci_ub``).

**Returns**

`PlotResult`.

## `PlotResult`

```python
PlotResult(ax: 'plt.Axes', figure: 'plt.Figure', filename: 'str | None' = None) -> None
```

Outcome of a plotting call (axes + optional saved path).

## `heatplot`

```python
heatplot(path_result, xlab='Randomized comparison', ylab='', cex_lab=1.0, col=None, figsize=(8, 2.6), filename=None)
```

*Mirrors R's `heatplot.netpath`.*

``netmeta::heatplot.netpath`` — standardised heat map of path evidence.

Takes the dict returned by `.netmeta.netpath` and draws a 1-row heat
map of ``theta_p`` standardised by its standard error
(``theta_p / sqrt(diag(Sigma))``), i.e. the evidence flow of every simple
path between the two nodes.

**Args**

- `path_result` — the dict returned by ``netpath``.
- `xlab, ylab, cex_lab` — axis labels / label size.
- `col` — optional matplotlib colormap name.
- `figsize, filename` — figure size and output path.

**Returns**

`PlotResult`.

## `regplot`

```python
regplot(result, mod=None, labels=None, xlab='Moderator', ylab='Effect size', ci=True, pi=False, level=None, predlim=None, len=100, perm=None, seed=None, pch=19, psize=None, col=None, figsize=(7, 5), filename=None)
```

*Mirrors R's `regplot.rma`.*

``metafor::regplot.rma`` — meta-regression plot with a reference region.

Draws the fitted regression line with its confidence (and optionally
prediction) band.  With ``perm`` the permutation-based reference region of
metafor is added: the moderator is permuted ``perm`` times, the model is
refitted with ``tau2`` held fixed, and the envelope of the fitted lines is
shaded.

**Args**

- `result` — a `.rma_uni.RmaResult` with exactly one moderator.
- `mod` — the moderator values (defaults to the second column of ``X``).
- `labels` — optional study labels.
- `ci, pi` — draw the confidence / prediction band.
- `level` — confidence level (defaults to the model's).
- `predlim` — ``(min, max)`` of the moderator grid.
- `len` — number of grid points.
- `perm` — number of permutations for the reference region.
- `seed` — RNG seed for the permutations.
- `pch, psize, col, figsize, filename` — plotting options.

**Returns**

`PlotResult`.

## `radial`

```python
radial(yi, vi=None, sei=None, pooled: 'float' = 0.0, pooled_se: 'float | None' = None, studlab=None, null: 'float' = 0.0, xlab: 'str' = 'Inverse standard error', ylab: 'str' = 'Standardized estimate', filename: 'str | None' = None, figsize: 'tuple[float, float] | None' = None, title: 'str | None' = None, ax: 'plt.Axes | None' = None)
```

*Mirrors R's `radial`.*

``metafor::radial`` — Galbraith radial plot.

Plots ``yi/sei`` against ``1/sei``; a study's vertical distance from the
reference line is proportional to its standardized residual, and studies on
the same horizontal line carry the same precision.

**Returns**

`PlotResult`.

## `labbe`

```python
labbe(event_e, n_e, event_c, n_c, studlab=None, pooled=None, filename: 'str | None' = None, figsize: 'tuple[float, float] | None' = None, title: 'str | None' = None, ax: 'plt.Axes | None' = None)
```

*Mirrors R's `labbe`.*

``metafor::labbe`` — L'Abbé plot (treatment risk vs control risk).

**Args**

- `event_e, n_e` — events and totals in the experimental arms.
- `event_c, n_c` — events and totals in the control arms.
- `studlab` — optional study labels.
- `pooled` — optional ``(risk_e, risk_c)`` point for the pooled effect.

**Returns**

`PlotResult`.

## `llplot`

```python
llplot(result, tau2=None, refline: 'bool' = True, xlab: 'str | None' = None, filename: 'str | None' = None, figsize: 'tuple[float, float] | None' = None, title: 'str' = 'Log-likelihood plot', ax: 'plt.Axes | None' = None)
```

*Mirrors R's `llplot`.*

``metafor::llplot`` — profile log-likelihood of ``tau^2``.

Recomputes the (restricted) log-likelihood on a grid of ``tau^2`` values
around the estimate and plots it, rescaled so that the maximum is 0.

**Returns**

`PlotResult`.

## `bubble`

```python
bubble(result, mod=None, xlab: 'str | None' = None, filename: 'str | None' = None, figsize: 'tuple[float, float] | None' = None, title: 'str' = 'Bubble plot', ax: 'plt.Axes | None' = None)
```

*Mirrors R's `regplot`.*

``metafor::regplot``/``bubble`` — bubble plot for meta-regression.

**Args**

- `result` — fitted `RmaResult` with exactly one moderator.
- `mod` — moderator values (defaults to the single column of ``result.X``).
- `xlab` — x-axis label.

**Returns**

`PlotResult`.

## `drapery`

```python
drapery(result, tau2=None, level: 'float' = 0.95, prediction: 'bool' = True, filename: 'str | None' = None, figsize: 'tuple[float, float] | None' = None, title: 'str' = 'Drapery plot', ax: 'plt.Axes | None' = None)
```

Drapery plot — p-value / confidence interval curves across ``tau^2``.

Draws, for the overall effect, the two-sided p-value as a function of the
assumed heterogeneity (the "drapery"), together with the confidence and
prediction interval bounds.

**Returns**

`PlotResult`.

## `qqnorm_rma`

```python
qqnorm_rma(result, type: 'str' = 'resid', filename: 'str | None' = None, figsize: 'tuple[float, float] | None' = None, title: 'str' = 'Normal Q-Q plot', ax: 'plt.Axes | None' = None)
```

*Mirrors R's `qqnorm.rma.uni`.*

``metafor::qqnorm.rma.uni`` — normal Q-Q plot of residuals or effects.

``qqnorm.rma.mh`` / ``qqnorm.rma.peto`` are covered too: pass an
``rma.mh`` / ``rma.peto`` result and the matching residuals are computed
first.

**Args**

- `result` — fitted `RmaResult` (or an ``rma.mh``/``rma.peto`` fit).
- `type` — ``"resid"`` (raw residuals), ``"rstandard"`` (standardized) or ``"rstudent"`` (studentized).

**Returns**

`PlotResult`.

## `rankogram`

```python
rankogram(result, cumulative: 'bool' = False, filename: 'str | None' = None, figsize: 'tuple[float, float] | None' = None, title: 'str' = 'Rankogram', ax: 'plt.Axes | None' = None)
```

*Mirrors R's `rankogram`.*

``netmeta::rankogram`` — ranking-probability plot.

**Args**

- `result` — a `NetrankResult` (or any object exposing ``ranking``/``rank_matrix``).
- `cumulative` — plot cumulative probabilities instead of pointwise ones.

**Returns**

`PlotResult`.

## `netheat`

```python
netheat(result, random: 'bool' = False, tau_preset: 'float | None' = None, filename: 'str | None' = None, figsize: 'tuple[float, float] | None' = None, title: 'str' = 'Heat plot', ax: 'plt.Axes | None' = None)
```

*Mirrors R's `netheat`.*

``netmeta::netheat`` — heat plot of the contribution to inconsistency.

Each cell ``[i, j]`` is how much design *i* contributes to the
inconsistency that remains once design *j* is detached, so a row that
stands out marks a design involved in the network's contradictions.
Blue cells (negative) mean the design removes inconsistency, red ones
(positive) mean it adds some.

**Args**

- `result` — a `netmeta.NetmetaResult`, or any object exposing a precomputed ``heat_matrix``.
- `random` — use the within-design tau (R's ``netheat(x, random = TRUE)``).
- `tau_preset` — fix the heterogeneity variance instead of using 0 (common effect) or the within-design estimate.
- `filename` — write the figure here as well.
- `figsize` — matplotlib figure size.
- `title` — plot title; pass ``""`` to leave it blank.
- `ax` — draw into an existing axes.

**Returns**

`PlotResult`, or ``None`` when the network has no loop (R also warns and draws nothing in that case).

## `hasse`

```python
hasse(treatments=None, edges=None, filename: 'str | None' = None, figsize: 'tuple[float, float] | None' = None, title: 'str' = 'Hasse diagram', ax: 'plt.Axes | None' = None)
```

*Mirrors R's `hasse`.*

``netmeta::hasse`` — Hasse diagram of the partial order of treatments.

**Args**

- `treatments` — treatment labels (list of str).
- `edges` — iterable of ``(upper, lower)`` pairs describing the covering relations of the partial order.
- `title` — plot title.

**Returns**

`PlotResult`.

## `netposet`

```python
netposet(treatments=None, edges=None, filename: 'str | None' = None, figsize: 'tuple[float, float] | None' = None, title: 'str' = 'Partial order of treatments', ax: 'plt.Axes | None' = None)
```

*Mirrors R's `netposet`.*

``netmeta::netposet`` — partial-order (poset) diagram of treatments.

Draws the same covering-relation graph as `hasse` but is exposed under
netmeta's name; ``netposet`` additionally accepts pre-computed coordinates
through ``edges``.

**Args**

- `treatments` — treatment labels.
- `edges` — iterable of ``(upper, lower)`` covering relations.
- `title` — plot title.

**Returns**

`PlotResult`.

## `traffic_light`

```python
traffic_light(labels=None, judgements=None, domains=None, filename: 'str | None' = None, figsize: 'tuple[float, float] | None' = None, title: 'str' = 'Risk of bias', ax: 'plt.Axes | None' = None)
```

*Mirrors R's `traffic_light`.*

``meta::traffic_light`` — risk-of-bias traffic-light plot.

**Args**

- `labels` — study labels.
- `judgements` — 2-D structure ``studies x domains`` of judgements; each entry is one of ``"low"``, ``"unclear"``/``"some"``, ``"high"`` (case-insensitive) or a numeric score in ``{0, 1, 2}``.
- `domains` — domain names.
- `title` — plot title.

**Returns**

`PlotResult`.
