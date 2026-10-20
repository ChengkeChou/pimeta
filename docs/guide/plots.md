# Plots

Every figure function returns the `matplotlib.axes.Axes` it drew on. Nothing is
written to disk unless you pass `filename=`.

```python
import pimeta as ma

ax = ma.forest(yi, vi, studlab=labels, pooled=r)
ax.set_title("My meta-analysis")     # ordinary matplotlib from here

ma.funnel(yi, vi, filename="funnel.png")   # write a file as well
```

## Composing panels

```python
import matplotlib.pyplot as plt
import pimeta as ma

fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))

ma.forest(yi, vi, pooled=r, ax=axes[0])
ma.funnel(yi, vi, ax=axes[1])
ma.radial(yi, vi, ax=axes[2])

fig.tight_layout()
```

## Available figures

| Function | Mirrors | Notes |
|---|---|---|
| `forest` | `forest.rma` | `pooled=`, `effect_type=`, `sort_by_effect=` |
| `funnel` | `funnel.rma` | `y_axis="se"` / `"precision"` |
| `radial` | `radial.rma` | Galbraith plot |
| `labbe` | `labbe.rma` | L'Abbé plot |
| `qqnorm_rma` | `qqnorm.rma.uni` | `type="resid"` / `"rstandard"` |
| `bubble` | `regplot` | meta-regression bubble plot |
| `regplot` | `regplot.rma` | regression plot with CI/PI bands |
| `drapery` | `drapery` | likelihood surface of the pooled estimate |
| `baujat` | `baujat` | contribution vs influence |
| `plot_influence` | `plot.infl.rma.uni` | Cook's distance, DFBS, hat |
| `plot_gosh` | `plot.gosh.rma` | GOSH diagnostic scatter |
| `plot_permutest` | `plot.permutest.rma.uni` | permutation distribution |
| `plot_mv_profile` | `plot.profile.rma` | variance-component profile |
| `plot_selprofile` | `plot.profile.rma` | selection-model profile |
| `llplot` | `llplot` | likelihood plot for individual studies |

Network meta-analysis:

| Function | Notes |
|---|---|
| `netgraph` | evidence graph; `seq=` controls layout order |
| `netheat` / `heatplot` | inconsistency heat map |
| `rankogram` | rank probabilities |
| `netposet`, `hasse` | partial order of treatments |
| `traffic_light` | certainty-of-evidence table |

## One entry point

`plot(fitted_object, which=...)` routes to the right figure:

```python
import pimeta as ma

ma.plot(r, which="forest")
ma.plot(r, which="funnel")
ma.plot(r, which="baujat")
ma.plot(ma.profile_rma(r), which="profile")

# Profile objects are auto-detected.
ma.plot(ma.profile_rmamv(fit_mv))
ma.plot(ma.profile_selmodel(fit_sel))
```

## Headless and file output

The backend is forced to `Agg` when no display is available, so scripts and CI
never block. `filename=` accepts anything matplotlib's `savefig` understands
(`.png`, `.pdf`, `.svg`, …).

```python
import pimeta as ma

ma.forest(yi, vi, filename="forest.pdf", figsize=(8, 6))
```

## Styling

Because the return value is a real `Axes`, every matplotlib technique applies:

```python
import matplotlib as mpl
import pimeta as ma

mpl.rcParams["font.size"] = 9

ax = ma.forest(yi, vi)
for label in ax.get_yticklabels():
    label.set_fontstyle("italic")
```
