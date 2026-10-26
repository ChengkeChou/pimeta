"""Plotting layer for meta-analysis (mirrors ``metafor`` / ``meta`` / ``netmeta``).

Built directly on the matplotlib primitives (no seaborn / networkx / forestplot
dependency) to draw the standard meta-analysis figures:

- :func:`forest`   — forest plot (metafor::forest, classic diamond style)
- :func:`funnel`   — funnel plot (metafor::funnel)
- :func:`netgraph` — network evidence graph (netmeta::netgraph)
- :func:`plot_baujat` — Baujat influence plot (metafor::plot.baujat.rma)
- :func:`plot_gosh`    — GOSH heterogeneity plot (metafor::plot.gosh.rma)

All functions use the non-interactive ``Agg`` backend so they are safe to call
headlessly, and every one accepts an optional ``filename`` to persist the figure
(PNG/PDF/...) and returns the matplotlib ``Axes``.
"""

from __future__ import annotations

import math
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import cast

import matplotlib

matplotlib.use("Agg")  # headless-safe before importing pyplot
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Polygon
from scipy import stats as _stats

__all__ = [
    "forest",
    "funnel",
    "netgraph",
    "plot_baujat",
    "plot_gosh",
    "plot_sroc",
    "radial",
    "labbe",
    "llplot",
    "bubble",
    "drapery",
    "qqnorm_rma",
    "rankogram",
    "netheat",
    "hasse",
    "netposet",
    "traffic_light",
    "heatplot",
    "regplot",
    "plot_influence",
    "plot_permutest",
    "plot_selprofile",
    "plot_mv_profile",
    "PlotResult",
]

# Ratio effect types are pooled on the log scale; their null line sits at 0
# on the analysis scale (i.e. exp(0) = 1 on the raw scale).
_RATIO_TYPES = {"RR", "OR", "HR", "PETO", "IRR", "PLN", "PLO"}


@dataclass
class PlotResult:
    """Outcome of a plotting call (axes + optional saved path)."""

    ax: plt.Axes
    figure: plt.Figure
    filename: str | None = None


def _arr(values, name: str) -> np.ndarray:
    arr = np.asarray(list(values), dtype=float).ravel()
    if arr.size == 0:
        raise ValueError(f"{name} must not be empty")
    return arr


def _studlab(studlab, k: int) -> list[str]:
    if studlab is None:
        return [f"Study {i + 1}" for i in range(k)]
    return [str(s) for s in list(studlab)[:k]]


def _is_ratio(effect_type: str | None) -> bool:
    return (effect_type or "").upper() in _RATIO_TYPES


def _save(fig: plt.Figure, filename: str | None) -> str | None:
    if filename is None:
        return None
    path = Path(filename)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=300, bbox_inches="tight")
    return str(path)


def _effect_label(effect_type: str | None, log_scale: bool) -> str:
    et = (effect_type or "Effect").upper()
    if log_scale:
        return f"{et} (log scale)"
    return et


def forest(
    yi,
    vi=None,
    studlab=None,
    pooled=None,
    effect_type: str | None = None,
    log_scale: bool | None = None,
    null: float = 0.0,
    xlab: str | None = None,
    sort_by_effect: bool = False,
    filename: str | None = None,
    figsize: tuple[float, float] | None = None,
    title: str | None = None,
    ax: plt.Axes | None = None,
) -> plt.Axes:
    """Draw a forest plot.

    ``yi`` / ``vi`` are per-study effect sizes and sampling variances on the
    *analysis* scale (log scale for ratio types). ``pooled`` is an optional
    :class:`~pimeta.pooling.PooledResult` (or any object
    exposing ``pooled_est``, ``pooled_se`` / ``pooled_ci_lower``,
    ``pooled_ci_upper`` and ``weights``) used to draw the summary diamond and
    size the study boxes by weight.

    A diamond needs ``pooled``. Without it only the per-study rows are drawn.
    """
    yi = _arr(yi, "yi")
    k = yi.size

    if vi is None:
        raise ValueError("vi (per-study variance) is required for a forest plot")
    vi = _arr(vi, "vi")
    if vi.size != k:
        raise ValueError("yi and vi must have the same length")

    ratio = _is_ratio(effect_type) if log_scale is None else bool(log_scale)
    labels = _studlab(studlab, k)

    # Effective variances: fold the between-study variance in when available.
    tau2 = 0.0
    if pooled is not None and hasattr(pooled, "tau2"):
        tau2 = float(pooled.tau2) or 0.0
    se_i = np.sqrt(vi + tau2)

    lo = yi - 1.95996398454 * np.sqrt(vi)
    hi = yi + 1.95996398454 * np.sqrt(vi)

    order = np.arange(k)
    if sort_by_effect:
        order = np.argsort(yi)

    # Reuse a provided axis or create a new figure.
    h = max(4.0, 0.45 * k + 2.0)
    fig, ax = _new_ax(ax, figsize or (7.0, h))

    ypos = np.arange(k)[::-1].astype(float)  # first study at the top

    # per-study boxes (size ~ weight) + CI whiskers
    if pooled is not None and hasattr(pooled, "weights") and pooled.weights:
        w = np.asarray(pooled.weights, dtype=float).ravel()
        if w.size == k:
            w = w / max(w.max(), 1e-12)
        else:
            w = 1.0 / se_i
    else:
        w = 1.0 / se_i
    w = w / max(w.max(), 1e-12)
    sizes = 20.0 + 160.0 * np.asarray(w).ravel()[order]

    ax.hlines(ypos, lo[order], hi[order], color="0.35", linewidth=1.0, zorder=2)
    ax.scatter(
        yi[order], ypos, s=sizes, c="k", edgecolors="k", zorder=3, linewidths=0.5
    )

    # summary diamond
    if pooled is not None:
        p_est = getattr(pooled, "pooled_est", None)
        if p_est is None and hasattr(pooled, "pooled_effect"):
            p_est = math.log(pooled.pooled_effect) if ratio else pooled.pooled_effect
        if p_est is not None:
            ci_l = getattr(pooled, "pooled_ci_lower", None)
            ci_u = getattr(pooled, "pooled_ci_upper", None)
            p_se = getattr(pooled, "pooled_se", None)
            if ratio and ci_l is not None and ci_u is not None:
                ci_l, ci_u = math.log(float(ci_l)), math.log(float(ci_u))
            elif p_se is not None:
                ci_l = p_est - 1.95996398454 * p_se
                ci_u = p_est + 1.95996398454 * p_se
            if ci_l is not None and ci_u is not None:
                dy = -0.6
                diamond = Polygon(
                    [
                        (ci_l, dy),
                        (p_est, dy + 0.55),
                        (ci_u, dy),
                        (p_est, dy - 0.55),
                    ],
                    closed=True,
                    facecolor="0.6",
                    edgecolor="k",
                    zorder=4,
                )
                ax.add_patch(diamond)
                ax.vlines(p_est, dy - 1.0, dy + 1.0, color="k", linewidth=0.8, zorder=3)

    ax.axvline(null, color="k", linewidth=0.8, linestyle="--", zorder=1)
    ax.set_yticks(ypos)
    ax.set_yticklabels([labels[i] for i in order], fontsize=9)
    ax.set_xlabel(xlab or _effect_label(effect_type, ratio))
    if ratio:
        from matplotlib.ticker import FuncFormatter

        ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{math.exp(v):g}"))
    if title:
        ax.set_title(title)
    ax.margins(x=0.05)

    _save(fig, filename)
    return ax


def funnel(
    yi,
    vi=None,
    studlab=None,
    effect_type: str | None = None,
    xlab: str | None = None,
    filename: str | None = None,
    figsize: tuple[float, float] = (6.0, 6.0),
    title: str | None = None,
    ax: plt.Axes | None = None,
) -> plt.Axes:
    """Draw a funnel plot of effect size vs. standard error.

    Inverted (metafor) convention: the y axis is the standard error so more
    precise studies sit at the top.
    """
    yi = _arr(yi, "yi")
    k = yi.size
    if vi is None:
        raise ValueError("vi (per-study variance) is required for a funnel plot")
    vi = _arr(vi, "vi")
    if vi.size != k:
        raise ValueError("yi and vi must have the same length")

    se = np.sqrt(vi)
    ratio = _is_ratio(effect_type)

    fig, ax = _new_ax(ax, figsize)

    ax.scatter(yi, se, s=40, c="0.2", edgecolors="k", linewidths=0.5, zorder=3)

    # pseudo 95% confidence triangle around the pooled (fixed-effect) mean
    mu = float(np.sum(yi / vi) / np.sum(1.0 / vi))
    se_max = float(se.max()) * 1.15
    se_grid = np.linspace(0.0, se_max, 200)
    ax.plot(mu + 1.96 * se_grid, se_grid, color="0.4", linestyle="--", linewidth=1.0)
    ax.plot(mu - 1.96 * se_grid, se_grid, color="0.4", linestyle="--", linewidth=1.0)
    ax.axvline(mu, color="0.5", linewidth=0.8)

    ax.set_xlabel(xlab or _effect_label(effect_type, ratio))
    ax.set_ylabel("Standard Error")
    ax.invert_yaxis()
    if title:
        ax.set_title(title)
    ax.margins(x=0.05)

    _save(fig, filename)
    return ax


def netgraph(
    result=None,
    trts=None,
    treat1=None,
    treat2=None,
    studlab=None,
    filename: str | None = None,
    figsize: tuple[float, float] = (7.0, 7.0),
    title: str | None = None,
    node_size: float = 900.0,
    ax: plt.Axes | None = None,
) -> plt.Axes:
    """Draw a network evidence graph.

    Accepts either a :class:`NetmetaResult` (reads ``trts``, ``treat1_raw``,
    ``treat2_raw``, ``studlab_raw``) or explicit ``trts`` / ``treat1`` /
    ``treat2`` lists. Edge width encodes the number of studies comparing the
    pair. Layout is netmeta's default circular arrangement (treatments placed
    evenly around a circle, starting from the top).
    """
    if result is not None:
        trts = getattr(result, "trts", trts)
        treat1 = getattr(result, "treat1_raw", treat1)
        treat2 = getattr(result, "treat2_raw", treat2)
        studlab = getattr(result, "studlab_raw", studlab)

    trts = list(trts)
    treat1 = list(treat1)
    treat2 = list(treat2)
    if not trts or len(treat1) != len(treat2):
        raise ValueError("netgraph needs trts and aligned treat1/treat2")

    pair_count: dict[tuple[str, str], int] = {}
    for a, b in zip(treat1, treat2):
        ta, tb = str(a), str(b)
        key: tuple[str, str] = (ta, tb) if ta <= tb else (tb, ta)
        pair_count[key] = pair_count.get(key, 0) + 1

    edges = []
    widths = []
    for (a, b), cnt in pair_count.items():
        if a in trts and b in trts and a != b:
            edges.append((a, b))
            widths.append(cnt)

    fig, ax = _new_ax(ax, figsize)

    # Circular layout: treatments evenly spaced, starting from the top.
    n = len(trts)
    theta = np.linspace(0.0, 2.0 * np.pi, n, endpoint=False)
    theta = np.pi / 2.0 - theta
    pos = {t: (float(np.cos(th)), float(np.sin(th))) for t, th in zip(trts, theta)}

    for (a, b), w in zip(edges, widths):
        x1, y1 = pos[a]
        x2, y2 = pos[b]
        ax.plot([x1, x2], [y1, y2], color="0.4",
                linewidth=1.5 + 2.5 * w, alpha=0.9, zorder=1)

    xs = [pos[t][0] for t in trts]
    ys = [pos[t][1] for t in trts]
    ax.scatter(xs, ys, s=node_size, c="0.85", edgecolors="k", linewidths=1.0,
               zorder=2)

    for t in trts:
        x, y = pos[t]
        ax.annotate(str(t), (x, y), ha="center", va="center", fontsize=10,
                    zorder=3)

    if title:
        ax.set_title(title)
    ax.axis("off")

    _save(fig, filename)
    return ax


# --------------------------------------------------------------------------- #
# Additional metafor / meta / netmeta figures
# --------------------------------------------------------------------------- #

def _new_ax(ax: plt.Axes | None = None,
            figsize=None) -> tuple[plt.Figure, plt.Axes]:
    """Return ``(figure, axes)``, reusing ``ax`` when one was supplied.

    ``Axes.figure`` is typed ``Figure | SubFigure | None`` because an axes can
    live on a subfigure; a standalone axes always has a real figure, so the
    cast is exact.  Keeping this in one place means the plotting functions do
    not each repeat an untyped ``plt.subplots`` branch.
    """
    if ax is None:
        fig, new_ax = plt.subplots(figsize=figsize or (6.5, 5.0))
        return fig, new_ax
    return cast(plt.Figure, ax.figure), ax


def radial(
    yi,
    vi=None,
    sei=None,
    pooled: float = 0.0,
    pooled_se: float | None = None,
    studlab=None,
    null: float = 0.0,
    xlab: str = "Inverse standard error",
    ylab: str = "Standardized estimate",
    filename: str | None = None,
    figsize: tuple[float, float] | None = None,
    title: str | None = None,
    ax: plt.Axes | None = None,
):
    """``metafor::radial`` — Galbraith radial plot.

    Plots ``yi/sei`` against ``1/sei``; a study's vertical distance from the
    reference line is proportional to its standardized residual, and studies on
    the same horizontal line carry the same precision.

    Returns:
        :class:`PlotResult`.
    """
    yi = _arr(yi, "yi")
    if sei is None:
        if vi is None:
            raise ValueError("must specify 'vi' or 'sei'")
        sei = np.sqrt(_arr(vi, "vi"))
    else:
        sei = _arr(sei, "sei")

    fig, ax = _new_ax(ax, figsize)
    x = 1.0 / sei
    y = (yi - null) / sei

    # reference line through the pooled estimate with slope = pooled effect
    xs = np.linspace(0.0, float(np.max(x)) * 1.05, 50)
    ax.plot(xs, pooled * xs, color="C0", lw=1.2, label="pooled")
    if pooled_se is not None:
        for sign, style in ((-1, "--"), (1, "--")):
            ax.plot(xs, pooled * xs + sign * 1.96 * np.sqrt(1.0 + pooled_se ** 2 * xs ** 2),
                    color="C3", lw=1.0, ls=style)
    ax.scatter(x, y, s=26, c="0.4", zorder=3)
    if studlab is not None:
        for xi, yi_, lab in zip(x, y, _studlab(studlab, x.size)):
            ax.annotate(str(lab), (xi, yi_), xytext=(4, 2),
                        textcoords="offset points", fontsize=7)
    ax.set_xlabel(xlab)
    ax.set_ylabel(ylab)
    if title:
        ax.set_title(title)
    ax.grid(alpha=0.25)
    _save(fig, filename)
    return PlotResult(ax=ax, figure=fig, filename=filename)


def labbe(
    event_e,
    n_e,
    event_c,
    n_c,
    studlab=None,
    pooled=None,
    filename: str | None = None,
    figsize: tuple[float, float] | None = None,
    title: str | None = None,
    ax: plt.Axes | None = None,
):
    """``metafor::labbe`` — L'Abbé plot (treatment risk vs control risk).

    Args:
        event_e, n_e: events and totals in the experimental arms.
        event_c, n_c: events and totals in the control arms.
        studlab: optional study labels.
        pooled: optional ``(risk_e, risk_c)`` point for the pooled effect.

    Returns:
        :class:`PlotResult`.
    """
    ee, ne = _arr(event_e, "event_e"), _arr(n_e, "n_e")
    ec, nc = _arr(event_c, "event_c"), _arr(n_c, "n_c")
    if not (ee.size == ne.size == ec.size == nc.size):
        raise ValueError("event/n vectors must have the same length")
    pe = ee / ne
    pc = ec / nc

    fig, ax = _new_ax(ax, figsize)
    sizes = 18.0 + 60.0 * (ne + nc) / float(np.max(ne + nc))
    ax.scatter(pc, pe, s=sizes, c="0.45", alpha=0.75, edgecolors="k",
               linewidths=0.6, zorder=3)
    lim = max(float(np.max(pc)), float(np.max(pe)), 1e-6) * 1.08
    ax.plot([0, lim], [0, lim], color="0.6", lw=1.0, ls="--")
    if studlab is not None:
        for a, b, lab in zip(pc, pe, _studlab(studlab, pc.size)):
            ax.annotate(str(lab), (a, b), xytext=(4, 2),
                        textcoords="offset points", fontsize=7)
    if pooled is not None:
        ax.scatter([pooled[1]], [pooled[0]], marker="D", s=60, c="C3", zorder=4,
                   label="pooled")
        ax.legend(frameon=False, fontsize=8)
    ax.set_xlim(0, lim)
    ax.set_ylim(0, lim)
    ax.set_xlabel("Risk in control group")
    ax.set_ylabel("Risk in experimental group")
    if title:
        ax.set_title(title)
    ax.grid(alpha=0.25)
    _save(fig, filename)
    return PlotResult(ax=ax, figure=fig, filename=filename)


def llplot(
    result,
    tau2=None,
    refline: bool = True,
    xlab: str | None = None,
    filename: str | None = None,
    figsize: tuple[float, float] | None = None,
    title: str = "Log-likelihood plot",
    ax: plt.Axes | None = None,
):
    """``metafor::llplot`` — profile log-likelihood of ``tau^2``.

    Recomputes the (restricted) log-likelihood on a grid of ``tau^2`` values
    around the estimate and plots it, rescaled so that the maximum is 0.

    Returns:
        :class:`PlotResult`.
    """
    from .rma_uni import RmaResult

    if not isinstance(result, RmaResult):
        raise TypeError("argument 'result' must be an RmaResult")
    tau2_hat = float(result.tau2)
    if tau2 is None:
        hi = max(4.0 * tau2_hat, tau2_hat + 0.5, 0.5)
        grid = np.linspace(0.0, hi, 120)
    else:
        grid = _arr(tau2, "tau2")

    yi = np.asarray(result.yi, dtype=float)
    vi = np.asarray(result.vi, dtype=float)
    X = np.asarray(result.X, dtype=float)

    def _ll(t2):
        w = 1.0 / (vi + t2)
        WX = X * w[:, None]
        try:
            inv = np.linalg.inv(X.T @ WX)
        except np.linalg.LinAlgError:
            inv = np.linalg.pinv(X.T @ WX)
        beta = inv @ (X.T @ (w * yi))
        r = yi - X @ beta
        if result.method in ("FE", "EE", "CE"):
            return -0.5 * float(r @ (w * r))
        det_term = float(np.sum(np.log(vi + t2)))
        return -0.5 * (float(r @ (w * r)) + det_term - math.log(max(abs(np.linalg.det(X.T @ WX)), 1e-300)))

    lls = np.array([_ll(t) for t in grid])
    lls = lls - np.max(lls)

    fig, ax = _new_ax(ax, figsize)
    ax.plot(grid, lls, color="C0", lw=1.4)
    if refline:
        ax.axvline(tau2_hat, color="C3", lw=1.0, ls="--")
        ax.axhline(-1.920729, color="0.6", lw=1.0, ls=":")  # 95% cutoff
    ax.set_xlabel(xlab or "tau^2")
    ax.set_ylabel("Relative log-likelihood")
    if title:
        ax.set_title(title)
    ax.grid(alpha=0.25)
    _save(fig, filename)
    return PlotResult(ax=ax, figure=fig, filename=filename)


def bubble(
    result,
    mod=None,
    xlab: str | None = None,
    filename: str | None = None,
    figsize: tuple[float, float] | None = None,
    title: str = "Bubble plot",
    ax: plt.Axes | None = None,
):
    """``metafor::regplot``/``bubble`` — bubble plot for meta-regression.

    Args:
        result: fitted :class:`RmaResult` with exactly one moderator.
        mod: moderator values (defaults to the single column of ``result.X``).
        xlab: x-axis label.

    Returns:
        :class:`PlotResult`.
    """
    from .rma_uni import RmaResult

    if not isinstance(result, RmaResult):
        raise TypeError("argument 'result' must be an RmaResult")
    X = np.asarray(result.X, dtype=float)
    if X.shape[1] < 2:
        raise ValueError("bubble plots require a model with moderators")
    if mod is None:
        if X.shape[1] != 2:
            raise ValueError("specify 'mod' when the model has >1 moderator")
        mod = X[:, 1]
    mod = _arr(mod, "mod")
    yi = np.asarray(result.yi, dtype=float)
    vi = np.asarray(result.vi, dtype=float)
    beta = np.asarray(result.b, dtype=float)
    tau2 = float(result.tau2)

    w = 1.0 / (vi + tau2)
    sizes = 12.0 + 120.0 * w / float(np.max(w))

    fig, ax = _new_ax(ax, figsize)
    ax.scatter(mod, yi, s=sizes, c="C0", alpha=0.55, edgecolors="k",
               linewidths=0.5, zorder=3)
    xs = np.linspace(float(np.min(mod)), float(np.max(mod)), 100)
    ax.plot(xs, beta[0] + beta[1] * xs, color="C3", lw=1.6, zorder=4)
    ax.set_xlabel(xlab or "Moderator")
    ax.set_ylabel("Observed outcome")
    if title:
        ax.set_title(title)
    ax.grid(alpha=0.25)
    _save(fig, filename)
    return PlotResult(ax=ax, figure=fig, filename=filename)


def drapery(
    result,
    tau2=None,
    level: float = 0.95,
    prediction: bool = True,
    filename: str | None = None,
    figsize: tuple[float, float] | None = None,
    title: str = "Drapery plot",
    ax: plt.Axes | None = None,
):
    """Drapery plot — p-value / confidence interval curves across ``tau^2``.

    Draws, for the overall effect, the two-sided p-value as a function of the
    assumed heterogeneity (the "drapery"), together with the confidence and
    prediction interval bounds.

    Returns:
        :class:`PlotResult`.
    """
    from scipy import stats as _st

    from .rma_uni import RmaResult

    if not isinstance(result, RmaResult):
        raise TypeError("argument 'result' must be an RmaResult")
    yi = np.asarray(result.yi, dtype=float)
    vi = np.asarray(result.vi, dtype=float)
    tau2_hat = float(result.tau2)
    grid = np.linspace(0.0, max(3.0 * tau2_hat, 0.5), 120) if tau2 is None else _arr(tau2, "tau2")

    betas, ses = [], []
    for t2 in grid:
        w = 1.0 / (vi + t2)
        b = float(np.sum(w * yi) / np.sum(w))
        betas.append(b)
        ses.append(math.sqrt(1.0 / float(np.sum(w))))
    beta_arr = np.array(betas)
    se_arr = np.array(ses)
    crit = _st.norm.isf((1.0 - level) / 2.0)

    fig, ax = _new_ax(ax, figsize)
    ax.plot(grid, beta_arr, color="C0", lw=1.5, label="pooled estimate")
    ax.fill_between(grid, beta_arr - crit * se_arr, beta_arr + crit * se_arr,
                    color="C0", alpha=0.15, label=f"{int(level*100)}% CI")
    if prediction:
        pse = np.sqrt(se_arr ** 2 + grid)
        ax.plot(grid, beta_arr - crit * pse, color="C3", lw=1.0, ls="--",
                label="prediction interval")
        ax.plot(grid, beta_arr + crit * pse, color="C3", lw=1.0, ls="--")
    ax.axvline(tau2_hat, color="0.4", lw=1.0, ls=":")
    ax.set_xlabel("tau^2")
    ax.set_ylabel("Effect size")
    if title:
        ax.set_title(title)
    ax.legend(frameon=False, fontsize=8)
    ax.grid(alpha=0.25)
    _save(fig, filename)
    return PlotResult(ax=ax, figure=fig, filename=filename)


def qqnorm_rma(
    result,
    type: str = "resid",
    filename: str | None = None,
    figsize: tuple[float, float] | None = None,
    title: str = "Normal Q-Q plot",
    ax: plt.Axes | None = None,
):
    """``metafor::qqnorm.rma.uni`` — normal Q-Q plot of residuals or effects.

    ``qqnorm.rma.mh`` / ``qqnorm.rma.peto`` are covered too: pass an
    ``rma.mh`` / ``rma.peto`` result and the matching residuals are computed
    first.

    Args:
        result: fitted :class:`RmaResult` (or an ``rma.mh``/``rma.peto`` fit).
        type: ``"resid"`` (raw residuals), ``"rstandard"`` (standardized) or
            ``"rstudent"`` (studentized).

    Returns:
        :class:`PlotResult`.
    """

    from .rma_uni import RmaResult

    if str(type) not in ("resid", "rstandard", "rstudent"):
        raise ValueError("'type' must be 'resid', 'rstandard' or 'rstudent'")
    from .rma_peto_mh import is_mh_result, mh_rstandard, mh_rstudent

    if is_mh_result(result):
        slab = list(result.slab)
        if str(type) == "rstudent":
            resid = np.asarray(mh_rstudent(result)["z"], dtype=float)
        elif str(type) == "rstandard":
            resid = np.asarray(mh_rstandard(result)["z"], dtype=float)
        else:
            resid = np.asarray(mh_rstandard(result)["resid"], dtype=float)
        return _qq_panel(resid, slab, filename, figsize, title, ax)
    from .rma_mv import RmaMvResult

    if isinstance(result, RmaMvResult):
        from .mv_diagnostics import mv_rstandard, mv_rstudent

        if str(type) == "rstudent":
            resid = np.asarray(mv_rstudent(result)["z"], dtype=float)
        elif str(type) == "rstandard":
            resid = np.asarray(mv_rstandard(result)["z"], dtype=float)
        else:
            resid = np.asarray(mv_rstandard(result)["resid"], dtype=float)
        return _qq_panel(resid, list(result.slab or []), filename, figsize, title, ax)
    if not isinstance(result, RmaResult):
        raise TypeError("argument 'result' must be an RmaResult")
    resid = np.asarray(result.residuals, dtype=float)
    if type == "rstandard":
        resid = resid / np.sqrt(np.asarray(result.vi, dtype=float) + float(result.tau2))
    elif type == "rstudent":
        vi = np.asarray(result.vi, dtype=float)
        tau2 = float(result.tau2)
        resid = resid / np.sqrt(np.maximum(vi + tau2 - np.diag(np.asarray(result.vb, dtype=float))[0], 0.0))
    else:
        pass

    return _qq_panel(resid, None, filename, figsize, title, ax)


def _qq_panel(resid, slab=None, filename=None, figsize=None, title=None,
              ax=None):
    """Draw the normal Q-Q panel shared by the ``qqnorm`` methods."""
    from scipy import stats as _st

    resid = np.asarray(resid, dtype=float)
    ok = np.isfinite(resid)
    resid = resid[ok]
    n = resid.size
    p = (np.arange(1, n + 1) - 0.5) / n
    theo = _st.norm.ppf(p)
    obs = np.sort(resid)
    slope, intercept = np.polyfit(theo, obs, 1)

    fig, ax = _new_ax(ax, figsize)
    ax.scatter(theo, obs, s=24, c="0.35", zorder=3)
    ax.plot(theo, intercept + slope * theo, color="C3", lw=1.3)
    ax.set_xlabel("Theoretical quantiles")
    ax.set_ylabel("Sample quantiles")
    if title:
        ax.set_title(title)
    ax.grid(alpha=0.25)
    _save(fig, filename)
    return PlotResult(ax=ax, figure=fig, filename=filename)


def plot_selprofile(profile, filename=None, figsize=None,
                    title="Profile log-likelihood",
                    xlab="delta", ylab="Log-likelihood", ax=None):
    """``metafor::plot.profile.rma`` for a selection model.

    Draws the profiled log-likelihood against the selection parameter
    ``delta``, marks the maximum, and shades the profile-likelihood interval.

    Args:
        profile: a :class:`~.selmodel.SelProfileResult` (or anything with
            ``delta``/``ll``/``maxll``/``ci_lb``/``ci_ub``).

    Returns:
        :class:`PlotResult`.
    """
    d = np.asarray(profile.delta, dtype=float)
    ll = np.asarray(profile.ll, dtype=float)
    ok = np.isfinite(ll)
    if not ok.any():
        raise ValueError("the profile contains no finite log-likelihood")
    d, ll = d[ok], ll[ok]
    order = np.argsort(d)
    d, ll = d[order], ll[order]
    maxll = float(getattr(profile, "maxll", np.max(ll)))

    fig, ax = _new_ax(ax, figsize)
    lo = getattr(profile, "ci_lb", float("nan"))
    hi = getattr(profile, "ci_ub", float("nan"))
    if np.isfinite(lo) and np.isfinite(hi):
        ax.axvspan(lo, hi, color="0.9", zorder=0)
    ax.plot(d, ll, color="C0", linewidth=1.6, zorder=3)
    ax.axhline(maxll, color="0.5", linewidth=0.8, linestyle="--", zorder=1)
    hat = getattr(profile, "delta_hat", None)
    if hat is not None and np.isfinite(float(hat)):
        ax.axvline(float(hat), color="C3", linewidth=1.0, zorder=2)
    ax.set_xlabel(xlab)
    ax.set_ylabel(ylab)
    if title:
        ax.set_title(title)
    ax.grid(alpha=0.25)
    _save(fig, filename)
    return PlotResult(ax=ax, figure=fig, filename=filename)


def plot_mv_profile(profile, filename=None, figsize=None,
                    title=None, xlab=None, ylab="Log-likelihood",
                    ylim=None, ax=None):
    """``metafor::plot.profile.rma`` for a profiled parameter.

    Used for the variance components of an ``rma.mv`` fit
    (:func:`~.mv_diagnostics.profile_rmamv`) and for the scale coefficients of
    a location-scale model (:func:`~.rma_ls.profile_rma_ls`); the x grid is
    read from ``profile.sigma2`` or, failing that, ``profile.alpha``.

    Args:
        profile: anything exposing ``sigma2`` or ``alpha``, plus ``ll``,
            ``maxll``, ``vc``, ``xlab`` and ``title``.

    Returns:
        :class:`PlotResult`.
    """
    grid = getattr(profile, "sigma2", None)
    if grid is None:
        grid = getattr(profile, "alpha", None)
    vc = np.asarray(grid, dtype=float)
    ll = np.asarray(profile.ll, dtype=float)
    ok = np.isfinite(ll)
    if not ok.any():
        raise ValueError("the profile contains no finite log-likelihood")
    vc, ll = vc[ok], ll[ok]
    order = np.argsort(vc)
    vc, ll = vc[order], ll[order]
    maxll = float(getattr(profile, "maxll", np.max(ll)))

    fig, ax = _new_ax(ax, figsize)
    ax.plot(vc, ll, color="C0", linewidth=1.6, zorder=3)
    ax.axhline(maxll, color="0.5", linewidth=0.8, linestyle="--", zorder=1)
    hat = getattr(profile, "vc", None)
    if hat is not None and np.isfinite(float(hat)):
        ax.axvline(float(hat), color="C3", linewidth=1.0, zorder=2)

    if xlab is None:
        xlab = getattr(profile, "xlab", None) or r"$\sigma^2$ Value"
    if title is None:
        title = getattr(profile, "title", None) or "Profile Plot"
    ax.set_xlabel(xlab)
    ax.set_ylabel(ylab)
    if ylim is not None:
        ax.set_ylim(*np.sort(np.asarray(ylim, dtype=float)))
    if title:
        ax.set_title(title)
    ax.grid(alpha=0.25)
    _save(fig, filename)
    return PlotResult(ax=ax, figure=fig, filename=filename)


def rankogram(
    result,
    cumulative: bool = False,
    filename: str | None = None,
    figsize: tuple[float, float] | None = None,
    title: str = "Rankogram",
    ax: plt.Axes | None = None,
):
    """``netmeta::rankogram`` — ranking-probability plot.

    Args:
        result: a :class:`NetrankResult` (or any object exposing
            ``ranking``/``rank_matrix``).
        cumulative: plot cumulative probabilities instead of pointwise ones.

    Returns:
        :class:`PlotResult`.
    """
    probs = getattr(result, "rank_matrix", None)
    trts = getattr(result, "treatments", None)
    if probs is None:
        probs = getattr(result, "ranking", None)
    if probs is None:
        raise ValueError("result does not expose ranking probabilities")
    probs = np.asarray(probs, dtype=float)
    if probs.ndim == 1:
        probs = probs.reshape(1, -1)
    if cumulative:
        probs = np.cumsum(probs, axis=1)
    n_rank = probs.shape[1]
    if trts is None:
        trts = [f"T{i+1}" for i in range(probs.shape[0])]

    fig, ax = _new_ax(ax, figsize)
    xs = np.arange(1, n_rank + 1)
    for i, t in enumerate(trts):
        ax.plot(xs, probs[i], marker="o", ms=4, lw=1.2, label=str(t))
    ax.set_xlabel("Rank")
    ax.set_ylabel("Cumulative probability" if cumulative else "Probability")
    ax.set_xticks(xs)
    if title:
        ax.set_title(title)
    ax.legend(frameon=False, fontsize=7, ncol=2)
    ax.grid(alpha=0.25)
    _save(fig, filename)
    return PlotResult(ax=ax, figure=fig, filename=filename)


def netheat(
    result,
    random: bool = False,
    tau_preset: float | None = None,
    filename: str | None = None,
    figsize: tuple[float, float] | None = None,
    title: str = "Heat plot",
    ax: plt.Axes | None = None,
):
    """``netmeta::netheat`` — heat plot of the contribution to inconsistency.

    Each cell ``[i, j]`` is how much design *i* contributes to the
    inconsistency that remains once design *j* is detached, so a row that
    stands out marks a design involved in the network's contradictions.
    Blue cells (negative) mean the design removes inconsistency, red ones
    (positive) mean it adds some.

    Args:
        result: a :class:`~pimeta.netmeta.NetmetaResult`, or any object
            exposing a precomputed ``heat_matrix``.
        random: use the within-design tau (R's ``netheat(x, random = TRUE)``).
        tau_preset: fix the heterogeneity variance instead of using 0
            (common effect) or the within-design estimate.
        filename: write the figure here as well.
        figsize: matplotlib figure size.
        title: plot title; pass ``""`` to leave it blank.
        ax: draw into an existing axes.

    Returns:
        :class:`PlotResult`, or ``None`` when the network has no loop (R
        also warns and draws nothing in that case).
    """
    mat = getattr(result, "heat_matrix", None)
    labels = getattr(result, "designs", None)
    if mat is None:
        from .netmeta import netheat_matrix  # local import: avoid a cycle

        computed = netheat_matrix(result, random=random, tau_preset=tau_preset)
        if computed is None:
            warnings.warn(
                "Net heat plot not available: the network needs at least one "
                "loop, i.e. one degree of freedom between designs.",
                stacklevel=2,
            )
            return None
        mat, labels = computed

    mat = np.asarray(mat, dtype=float)
    if mat.ndim == 1:
        mat = mat.reshape(1, -1)

    fig, ax = _new_ax(ax, figsize)
    finite = mat[np.isfinite(mat)]
    limit = float(np.max(np.abs(finite))) if finite.size else 1.0
    im = ax.imshow(mat, aspect="auto", cmap="RdBu_r",
                   interpolation="nearest", vmin=-limit, vmax=limit)
    fig.colorbar(im, ax=ax, shrink=0.8)
    if labels is not None and len(labels) == mat.shape[0] == mat.shape[1]:
        ticks = range(mat.shape[0])
        ax.set_xticks(ticks)
        ax.set_yticks(ticks)
        ax.set_xticklabels([str(v) for v in labels], fontsize=7, rotation=90)
        ax.set_yticklabels([str(v) for v in labels], fontsize=7)
        ax.set_xlabel("detached design")
        ax.set_ylabel("design")
    if title:
        ax.set_title(title)
    _save(fig, filename)
    return PlotResult(ax=ax, figure=fig, filename=filename)


def hasse(
    treatments=None,
    edges=None,
    filename: str | None = None,
    figsize: tuple[float, float] | None = None,
    title: str = "Hasse diagram",
    ax: plt.Axes | None = None,
):
    """``netmeta::hasse`` — Hasse diagram of the partial order of treatments.

    Args:
        treatments: treatment labels (list of str).
        edges: iterable of ``(upper, lower)`` pairs describing the covering
            relations of the partial order.
        title: plot title.

    Returns:
        :class:`PlotResult`.
    """
    if treatments is None:
        raise ValueError("must specify 'treatments'")
    trts = [str(t) for t in treatments]
    edges = list(edges or [])
    # longest-path layering: level(v) = 1 + max(level of predecessors)
    level = {t: 0 for t in trts}
    for _ in range(len(trts)):
        changed = False
        for a, b in edges:
            if a in level and b in level and level[b] < level[a] + 1:
                level[b] = level[a] + 1
                changed = True
        if not changed:
            break
    # spread nodes within a level
    by_level: dict[int, list[str]] = {}
    for t in trts:
        by_level.setdefault(level[t], []).append(t)
    pos = {}
    for lv, nodes in by_level.items():
        n = len(nodes)
        for i, t in enumerate(nodes):
            pos[t] = ((i + 0.5) / max(n, 1), -lv)

    fig, ax = _new_ax(ax, figsize)
    for a, b in edges:
        if a in pos and b in pos:
            ax.plot([pos[a][0], pos[b][0]], [pos[a][1], pos[b][1]],
                    color="0.6", lw=1.0, zorder=1)
    for t in trts:
        if t in pos:
            ax.scatter([pos[t][0]], [pos[t][1]], s=380, c="0.92",
                       edgecolors="k", linewidths=1.0, zorder=2)
            ax.annotate(t, pos[t], ha="center", va="center", fontsize=9, zorder=3)
    ax.axis("off")
    if title:
        ax.set_title(title)
    _save(fig, filename)
    return PlotResult(ax=ax, figure=fig, filename=filename)


def netposet(
    treatments=None,
    edges=None,
    filename: str | None = None,
    figsize: tuple[float, float] | None = None,
    title: str = "Partial order of treatments",
    ax: plt.Axes | None = None,
):
    """``netmeta::netposet`` — partial-order (poset) diagram of treatments.

    Draws the same covering-relation graph as :func:`hasse` but is exposed under
    netmeta's name; ``netposet`` additionally accepts pre-computed coordinates
    through ``edges``.

    Args:
        treatments: treatment labels.
        edges: iterable of ``(upper, lower)`` covering relations.
        title: plot title.

    Returns:
        :class:`PlotResult`.
    """
    return hasse(
        treatments=treatments, edges=edges, filename=filename,
        figsize=figsize, title=title, ax=ax,
    )


def traffic_light(
    labels=None,
    judgements=None,
    domains=None,
    filename: str | None = None,
    figsize: tuple[float, float] | None = None,
    title: str = "Risk of bias",
    ax: plt.Axes | None = None,
):
    """``meta::traffic_light`` — risk-of-bias traffic-light plot.

    Args:
        labels: study labels.
        judgements: 2-D structure ``studies x domains`` of judgements; each
            entry is one of ``"low"``, ``"unclear"``/``"some"``, ``"high"``
            (case-insensitive) or a numeric score in ``{0, 1, 2}``.
        domains: domain names.
        title: plot title.

    Returns:
        :class:`PlotResult`.
    """
    if judgements is None:
        raise ValueError("must specify 'judgements'")
    j = np.asarray(judgements, dtype=object)
    if j.ndim == 1:
        j = j.reshape(1, -1)
    n_study, n_dom = j.shape
    labels = _studlab(labels, n_study)
    if domains is None:
        domains = [f"D{i+1}" for i in range(n_dom)]

    colours = {"low": "#2ca02c", "unclear": "#ffd92f", "some": "#ffd92f",
               "high": "#d62728"}
    fig, ax = _new_ax(ax, figsize)
    for i in range(n_study):
        for k in range(n_dom):
            v = j[i, k]
            if isinstance(v, (int, float, np.integer, np.floating)):
                key = "low" if v == 0 else ("unclear" if v == 1 else "high")
            else:
                key = str(v).strip().lower()
                key = key if key in colours else "unclear"
            ax.add_patch(plt.Circle((k + 0.5, n_study - i - 0.5), 0.30,
                                    color=colours[key], ec="k", lw=0.6))
    ax.set_xlim(0, n_dom)
    ax.set_ylim(0, n_study)
    ax.set_xticks(np.arange(n_dom) + 0.5)
    ax.set_xticklabels([str(d) for d in domains], rotation=30, ha="right", fontsize=8)
    ax.set_yticks(np.arange(n_study) + 0.5)
    ax.set_yticklabels(labels[::-1], fontsize=8)
    ax.set_aspect("equal")
    if title:
        ax.set_title(title)
    _save(fig, filename)
    return PlotResult(ax=ax, figure=fig, filename=filename)


def plot_baujat(
    result,
    studlab=None,
    filename: str | None = None,
    figsize: tuple[float, float] = (6.0, 6.0),
    title: str | None = None,
    ax: plt.Axes | None = None,
) -> plt.Axes:
    """Draw a Baujat plot (heterogeneity contribution vs. influence).

    ``result`` is a :class:`BaujatResult` (fields ``x`` and ``y``).
    """
    x = np.asarray(list(result.x), dtype=float).ravel()
    y = np.asarray(list(result.y), dtype=float).ravel()
    k = x.size
    labels = _studlab(studlab, k)

    fig, ax = _new_ax(ax, figsize)

    ax.scatter(x, y, s=50, c="0.2", edgecolors="k", linewidths=0.5, zorder=3)
    for xi, yi, lab in zip(x, y, labels):
        ax.annotate(lab, (xi, yi), textcoords="offset points", xytext=(4, 3),
                    fontsize=7)

    ax.set_xlabel("Contribution to overall heterogeneity")
    ax.set_ylabel("Influence on overall result")
    if title:
        ax.set_title(title)

    _save(fig, filename)
    return ax


def plot_gosh(
    result,
    x: str = "tau2",
    y: str = "estimate",
    filename: str | None = None,
    figsize: tuple[float, float] = (6.0, 6.0),
    title: str | None = None,
    ax: plt.Axes | None = None,
) -> plt.Axes:
    """Draw a GOSH plot (Goodman-Schulz-Hastie heterogeneity).

    ``result`` is a :class:`GoshResult`; ``x``/``y`` select two of the stored
    per-subset vectors: ``estimate``, ``tau2``, ``QE``, ``I2``, ``H2``.
    """
    def _col(name):
        if name == "estimate" and not getattr(result, "int_only", True):
            raise ValueError("'estimate' is only available for intercept-only GOSH")
        vals = getattr(result, name, None)
        if vals is None:
            raise ValueError(f"GoshResult has no field {name!r}")
        return np.asarray(list(vals), dtype=float).ravel()

    xv = _col(x)
    yv = _col(y)

    fig, ax = _new_ax(ax, figsize)

    ax.scatter(xv, yv, s=25, c="0.3", edgecolors="k", linewidths=0.3, alpha=0.8)
    ax.set_xlabel(x)
    ax.set_ylabel(y)
    if title:
        ax.set_title(title)

    _save(fig, filename)
    return ax


def plot_sroc(
    fit,
    studlab=None,
    curve: bool = True,
    summary: bool = True,
    ci: bool = True,
    show_auc: bool = True,
    size_by: str = "diseased",
    filename: str | None = None,
    figsize: tuple[float, float] = (6.0, 6.0),
    title: str | None = None,
    ax: plt.Axes | None = None,
) -> plt.Axes:
    """Draw an SROC: study points, summary curve and the summary point.

    The picture `mada` splits across ``crosshair`` and ``ROCellipse``, drawn in
    one axes: one dot per study at (FPR, TPR) with area proportional to the
    number of diseased participants, the Rutter-Gatsonis summary curve, and the
    back-transformed mean pair as a crosshair whose arms are the confidence
    interval on each axis.

    Args:
        fit: a :class:`~pimeta.dta.ReitsmaResult` from :func:`~pimeta.dta.reitsma`.
        studlab: study labels; defaults to ``Study 1`` ... ``Study k``. Pass
            ``[]`` to draw no labels.
        curve: draw the summary curve.
        summary: draw the summary point.
        ci: draw the confidence interval on each axis of the summary point.
        show_auc: annotate the AUC and the partial AUC.
        size_by: ``"diseased"`` (number of participants with the target
            condition), ``"total"`` or ``"equal"``.
        filename: when given, save at 300 dpi.
        figsize, title, ax: standard plotting arguments.

    Returns:
        The :class:`matplotlib.axes.Axes`.

    Raises:
        TypeError: when ``fit`` is not a ``ReitsmaResult``.

    Example:
        >>> fit = ma.reitsma(counts)            # doctest: +SKIP
        >>> ma.plot_sroc(fit)                   # doctest: +SKIP
    """
    from .dta import ReitsmaResult, sroc

    if not isinstance(fit, ReitsmaResult):
        raise TypeError(
            "plot_sroc expects a ReitsmaResult (the output of reitsma()); "
            f"got {type(fit).__name__}")

    data = fit.data
    fpr = np.asarray(data.fpr, dtype=float).ravel()
    tpr = np.asarray(data.sens, dtype=float).ravel()
    k = fpr.size
    if studlab is None:
        labels = _studlab(None, k)
    else:
        labels = [str(s) for s in list(studlab)]

    key = size_by.lower()
    if key == "diseased":
        weight = np.asarray(data.n_positive, dtype=float).ravel()
    elif key == "total":
        weight = (np.asarray(data.n_positive, dtype=float).ravel()
                  + np.asarray(data.n_negative, dtype=float).ravel())
    elif key == "equal":
        weight = np.ones(k)
    else:
        raise ValueError(
            "size_by must be 'diseased', 'total' or 'equal', "
            f"not {size_by!r}")
    span = weight.max() - weight.min()
    sizes = 40.0 + (140.0 * (weight - weight.min()) / span) if span > 0 else None
    if sizes is None:
        sizes = np.full(k, 90.0)

    fig, ax = _new_ax(ax, figsize)

    ax.plot([0.0, 1.0], [0.0, 1.0], color="0.7", linewidth=0.8,
            linestyle="--", zorder=1)

    if curve:
        smry = sroc(fit)
        ax.plot(smry.fpr, smry.tpr, color="C0", linewidth=1.6, zorder=2)

    ax.scatter(fpr, tpr, s=sizes, c="0.25", edgecolors="k", linewidths=0.5,
               alpha=0.75, zorder=3)
    if labels:
        for x, y, lab in zip(fpr, tpr, labels):
            ax.annotate(lab, (x, y), textcoords="offset points", xytext=(5, 3),
                        fontsize=7)

    if summary:
        sx = float(fit.summary_fpr)
        sy = float(fit.summary_sens)
        if ci:
            lo, hi = fit.fpr_ci
            ax.plot([lo, hi], [sy, sy], color="C3", linewidth=1.2, zorder=4)
            lo, hi = fit.sens_ci
            ax.plot([sx, sx], [lo, hi], color="C3", linewidth=1.2, zorder=4)
        ax.scatter([sx], [sy], s=90, marker="D", c="C3", edgecolors="k",
                   linewidths=0.6, zorder=5)

    if show_auc:
        stats = _auc_dict(fit)
        ax.text(
            0.98, 0.02,
            f"AUC = {stats['auc']:.3f}\npartial AUC = {stats['pauc']:.3f}",
            transform=ax.transAxes, ha="right", va="bottom", fontsize=8,
        )

    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.0)
    ax.set_xlabel("False positive rate (1 - specificity)")
    ax.set_ylabel("True positive rate (sensitivity)")
    if title:
        ax.set_title(title)

    _save(fig, filename)
    return ax


def _auc_dict(fit) -> dict:
    from .dta import auc

    return auc(fit)


# --------------------------------------------------------------------------- #
# heatplot.netpath  /  regplot.rma
# --------------------------------------------------------------------------- #

def heatplot(path_result, xlab="Randomized comparison", ylab="",
             cex_lab=1.0, col=None, figsize=(8, 2.6), filename=None):
    """``netmeta::heatplot.netpath`` — standardised heat map of path evidence.

    Takes the dict returned by :func:`~.netmeta.netpath` and draws a 1-row heat
    map of ``theta_p`` standardised by its standard error
    (``theta_p / sqrt(diag(Sigma))``), i.e. the evidence flow of every simple
    path between the two nodes.

    Args:
        path_result: the dict returned by ``netpath``.
        xlab, ylab, cex_lab: axis labels / label size.
        col: optional matplotlib colormap name.
        figsize, filename: figure size and output path.

    Returns:
        :class:`PlotResult`.
    """
    theta = path_result.get("theta_p")
    sigma = path_result.get("Sigma")
    if theta is None or sigma is None:
        raise ValueError(
            "no path found — 'theta_p' / 'Sigma' are missing from the netpath "
            "result"
        )
    theta = np.asarray(theta, dtype=float).ravel()
    se = np.sqrt(np.clip(np.diag(np.asarray(sigma, dtype=float)), 0, None))
    with np.errstate(divide="ignore", invalid="ignore"):
        z = theta / se

    fig, ax = plt.subplots(figsize=figsize)
    data = z.reshape(1, -1)
    vmax = float(np.nanmax(np.abs(data))) if np.isfinite(data).any() else 1.0
    im = ax.imshow(data, aspect="auto", cmap=col or "RdBu_r",
                   vmin=-vmax, vmax=vmax)
    ax.set_yticks([])
    ax.set_xticks(range(len(z)))
    labels = path_result.get("results") or []
    ax.set_xticklabels([f"Path {r['path_index']}" for r in labels]
                       if labels else [f"Path {i + 1}" for i in range(len(z))])
    for i, val in enumerate(z):
        if np.isfinite(val):
            ax.text(i, 0, f"{val:.2f}", ha="center", va="center",
                    fontsize=8, color="black")
    ax.set_xlabel(xlab, fontsize=11 * cex_lab)
    ax.set_ylabel(ylab, fontsize=11 * cex_lab)
    ax.set_title(f"{path_result.get('comparison', '')}", fontsize=11)
    fig.colorbar(im, ax=ax, orientation="vertical", shrink=0.8,
                 label="standardised evidence flow")
    fig.tight_layout()
    if filename:
        fig.savefig(filename)
    return PlotResult(ax=ax, figure=fig, filename=filename)


def regplot(result, mod=None, labels=None, xlab="Moderator",
            ylab="Effect size", ci=True, pi=False, level=None, predlim=None,
            len=100, perm=None, seed=None, pch=19, psize=None, col=None,
            figsize=(7, 5), filename=None):
    """``metafor::regplot.rma`` — meta-regression plot with a reference region.

    Draws the fitted regression line with its confidence (and optionally
    prediction) band.  With ``perm`` the permutation-based reference region of
    metafor is added: the moderator is permuted ``perm`` times, the model is
    refitted with ``tau2`` held fixed, and the envelope of the fitted lines is
    shaded.

    Args:
        result: a :class:`~.rma_uni.RmaResult` with exactly one moderator.
        mod: the moderator values (defaults to the second column of ``X``).
        labels: optional study labels.
        ci, pi: draw the confidence / prediction band.
        level: confidence level (defaults to the model's).
        predlim: ``(min, max)`` of the moderator grid.
        len: number of grid points.
        perm: number of permutations for the reference region.
        seed: RNG seed for the permutations.
        pch, psize, col, figsize, filename: plotting options.

    Returns:
        :class:`PlotResult`.
    """
    x = np.asarray(result.X, dtype=float)
    if x.shape[1] < 2:
        raise ValueError(
            "regplot needs a model with exactly one moderator (plus intercept)"
        )
    if mod is None:
        mod = x[:, 1]
    mod = np.asarray(mod, dtype=float).ravel()
    yi = np.asarray(result.yi, dtype=float).ravel()
    vi = np.asarray(result.vi, dtype=float).ravel()
    b = np.asarray(result.b, dtype=float).ravel()
    vb = np.asarray(result.vb, dtype=float)
    tau2 = float(result.tau2)
    if level is None:
        level = float(getattr(result, "level", 0.95))
    crit = float(_stats.norm.isf((1.0 - level) / 2.0))

    if predlim is None:
        lo, hi = float(np.min(mod)), float(np.max(mod))
    else:
        lo, hi = float(predlim[0]), float(predlim[1])
    grid = np.linspace(lo, hi, len)
    xg = np.column_stack([np.ones_like(grid), grid])
    pred = xg @ b
    se_fit = np.sqrt(np.einsum("ij,jk,ik->i", xg, vb, xg))
    ci_lb, ci_ub = pred - crit * se_fit, pred + crit * se_fit
    se_pi = np.sqrt(se_fit ** 2 + tau2)
    pi_lb, pi_ub = pred - crit * se_pi, pred + crit * se_pi

    fig, ax = plt.subplots(figsize=figsize)

    if perm:
        rng = np.random.default_rng(seed)
        w = 1.0 / (vi + tau2)
        lines = []
        for _ in range(int(perm)):
            mp = rng.permutation(mod)
            xp = np.column_stack([np.ones_like(mp), mp])
            xtwx = xp.T @ (w[:, None] * xp)
            try:
                bp = np.linalg.solve(xtwx, xp.T @ (w * yi))
            except np.linalg.LinAlgError:
                continue
            lines.append(xg @ bp)
        if lines:
            lines = np.vstack(lines)
            ax.fill_between(grid, lines.min(axis=0), lines.max(axis=0),
                            color="0.85", zorder=0,
                            label=f"reference region ({int(perm)} perms)")

    if pi:
        ax.fill_between(grid, pi_lb, pi_ub, color="0.75", alpha=0.6,
                        label=f"{int(round(100 * level))}% PI", zorder=1)
    if ci:
        ax.fill_between(grid, ci_lb, ci_ub, color="0.55", alpha=0.5,
                        label=f"{int(round(100 * level))}% CI", zorder=2)

    ax.scatter(mod, yi, s=psize if psize is not None else 36,
               marker="o" if pch == 19 else pch, color=col or "black",
               zorder=4)
    ax.plot(grid, pred, color=col or "black", linewidth=1.6, zorder=5)

    if labels is not None:
        for xi, yi_i, lab in zip(mod, yi, list(labels)[:mod.size]):
            ax.annotate(str(lab), (xi, yi_i), textcoords="offset points",
                        xytext=(4, 4), fontsize=7)

    ax.set_xlabel(xlab)
    ax.set_ylabel(ylab)
    if perm:
        ax.legend(loc="best", fontsize=8)
    fig.tight_layout()
    if filename:
        fig.savefig(filename)
    return PlotResult(ax=ax, figure=fig, filename=filename)


#: metafor's influence panels, in ``plot.infl.rma.uni``'s order.
_INFL_PANELS = (
    ("rstudent", "rstudent"),
    ("dffits", "dffits"),
    ("cook_d", "cook.d"),
    ("cov_r", "cov.r"),
    ("tau2_del", "tau2.del"),
    ("qe_del", "QE.del"),
    ("hat", "hat"),
    ("weight", "weight"),
)


def _infl_refs(key, x, crit):
    """Horizontal reference lines R draws in each influence panel."""
    k, p = int(x.k), int(x.p)
    if key == "rstudent":
        return [(0.0, "--"), (crit, ":"), (-crit, ":")]
    if key == "dffits":
        lim = 3.0 * math.sqrt(p / max(k - p, 1))
        return [(0.0, "--"), (lim, ":"), (-lim, ":")]
    if key == "cook_d":
        return [(_stats.chi2.ppf(0.5, p), ":")]
    if key == "cov_r":
        return [(1.0, "--")]
    if key == "tau2_del":
        return [(float(x.tau2), "--")]
    if key == "qe_del":
        return [(float(x.QE), "--")]
    if key == "hat":
        return [(float(k - p), ":"), (p / k, "--"), (3.0 * p / k, ":")]
    if key == "weight":
        return [(100.0 / k, "--")]
    return []


def _infl_ylim(key, vals, x, crit):
    """Per-panel y limits, following ``zi.max`` in ``plot.infl.rma.uni``."""
    k, p = int(x.k), int(x.p)
    a = np.abs(np.asarray(vals, dtype=float))
    a = a[np.isfinite(a)]
    top = float(a.max()) if a.size else 1.0
    pad = 1.05
    if key == "rstudent":
        return (-pad * max(top, 2.0), pad * max(top, 2.0))
    if key == "dffits":
        lim = pad * max(top, 3.0 * math.sqrt(p / max(k - p, 1)))
        return (-lim, lim)
    if key == "cov_r":
        return (None, pad * max(top, 1.0))
    if key == "hat":
        return (None, pad * max(top, 3.0 * p / k))
    return (None, pad * top if top > 0 else None)


def plot_influence(x, plotinf=True, plotdfbs=False, logcov=True, crit=None,
                   ncol=4, figsize=None, psize=None, col="black",
                   col_infl="red", slab=None, filename=None):
    """Influence diagnostics as a panel grid (``metafor::plot.infl.rma.uni``).

    R draws eight panels against the study index: the externally standardized
    residuals, DFFITS, Cook's distance, the covariance ratio, the leave-one-out
    tau^2, the leave-one-out QE, the hat value and the weight, each with the
    reference lines ``plot.infl.rma.uni`` adds (``+/- qnorm(0.975)`` for the
    residuals, ``+/- 3 sqrt(p/(k-p))`` for DFFITS, ``qchisq(0.5, p)`` for Cook,
    ``1`` for the covariance ratio, the full-model tau^2 and QE for the deletion
    panels, ``p/k`` and ``3p/k`` for the hat values and ``100/k`` for the
    weights).  Studies metafor flags as influential are drawn in red.

    Args:
        x: an :class:`~pimeta.sensitivity.InfluenceResult`.
        plotinf: ``True`` for all eight panels, or a list of 1-based panel
            indices (1 rstudent, 2 dffits, 3 cook.d, 4 cov.r, 5 tau2.del,
            6 QE.del, 7 hat, 8 weight).
        plotdfbs: also draw one panel per coefficient holding its DFBETAS.
        logcov: put the covariance-ratio panel on a log scale (R's default).
        crit: residual reference value (default ``qnorm(0.975)``).
        filename: optional path to save the figure to.

    Returns:
        :class:`PlotResult`.
    """
    if plotinf is True:
        wanted = list(range(1, 9))
    elif plotinf is False:
        wanted = []
    else:
        wanted = [int(v) for v in np.atleast_1d(plotinf)]
        wanted = [v for v in dict.fromkeys(wanted) if 1 <= v <= 8]
        if not wanted:
            raise ValueError("'plotinf' must be TRUE or index the 8 panels 1..8")
    if not plotdfbs and not wanted:
        raise ValueError("at least one of 'plotinf'/'plotdfbs' must be true")

    crit = float(_stats.norm.isf(0.025)) if crit is None else float(crit)
    panels = [_INFL_PANELS[i - 1] for i in wanted]
    dfbs = np.asarray(x.dfbs, dtype=float) if plotdfbs else None
    if dfbs is not None and dfbs.ndim == 2:
        # InfluenceResult.names holds the *coefficient* names (metafor's
        # x$dfbs columns), which is exactly what the DFBETAS panels are titled
        # with in plot.infl.rma.uni.
        coef_names = list(x.names or [])
        if len(coef_names) != dfbs.shape[1]:
            coef_names = [f"b{j + 1}" for j in range(dfbs.shape[1])]
        panels += [(f"__dfbs{j}", f"dfbs: {coef_names[j]}")
                   for j in range(dfbs.shape[1])]

    n = len(panels)
    ncol = max(1, min(int(ncol), n))
    nrow = int(math.ceil(n / ncol))
    if figsize is None:
        figsize = (3.2 * ncol, 2.4 * nrow)
    fig, axes = plt.subplots(nrow, ncol, figsize=figsize, squeeze=False)
    flat = axes.ravel()

    k = int(x.k)
    idx = np.arange(1, k + 1)
    labels = list(slab) if slab is not None else [str(i) for i in idx]
    if len(labels) != k:
        labels = [str(i) for i in idx]
    infl = np.asarray(x.is_infl, dtype=bool) if x.is_infl else np.zeros(k, bool)

    for ax, (key, label) in zip(flat, panels):
        if key.startswith("__dfbs"):
            j = int(key[6:])
            vals = dfbs[:, j]
            refs = [(0.0, "--"), (1.0, ":"), (-1.0, ":")]
            ylim = (None, 1.05 * max(float(np.nanmax(np.abs(vals))), 1.0)
                    if np.isfinite(vals).any() else None)
        else:
            vals = np.asarray(getattr(x, key), dtype=float).ravel()
            refs = _infl_refs(key, x, crit)
            ylim = _infl_ylim(key, vals, x, crit)
        for h, style in refs:
            ax.axhline(h, color="0.4", linewidth=0.8, linestyle=style, zorder=0)
        ax.vlines(idx, 0.0, vals, color="0.75", linewidth=0.6, zorder=1)
        ax.scatter(idx[~infl], vals[~infl], s=psize or 22, color=col,
                   zorder=3, linewidths=0.4, edgecolors="0.3")
        if infl.any():
            ax.scatter(idx[infl], vals[infl], s=psize or 22, color=col_infl,
                       zorder=4, linewidths=0.4, edgecolors="0.3")
        if key == "cov_r" and logcov:
            ax.set_yscale("log")
        if ylim != (None, None):
            ax.set_ylim(*ylim)
        ax.set_title(label, fontsize=8)
        ax.set_xticks(idx)
        ax.set_xticklabels(labels, rotation=90, fontsize=6)
        ax.tick_params(axis="y", labelsize=7)
        ax.grid(axis="y", color="0.92", linewidth=0.5)

    for ax in flat[n:]:
        ax.set_visible(False)

    fig.tight_layout()
    if filename:
        fig.savefig(filename)
    return PlotResult(ax=flat[0], figure=fig, filename=filename)


def plot_permutest(x, btt=None, nbin=None, figsize=None, col="0.8",
                   col_obs="blue", filename=None):
    """Permutation distributions (``metafor::plot.permutest.rma.uni``).

    One panel per coefficient: a histogram of the permutation distribution of
    the test statistic with the observed value marked, which is the picture
    R draws for ``plot(permutest(res))``.

    Args:
        x: a :class:`~pimeta.diagnostics.PermutestResult`.
        btt: 1-based indices of the coefficients to draw (default all).
        nbin: number of histogram bins (default ``ceiling(sqrt(nperm))``).

    Returns:
        :class:`PlotResult`.
    """
    if getattr(x, "zval_perm", None) is None:
        raise TypeError(
            "plot(x, which='permutest') needs the permutation distributions; "
            "the result was not produced by permutest()"
        )
    zperm = np.asarray(x.zval_perm, dtype=float)
    zobs = np.asarray(x.zval if x.zval is not None else x.beta, dtype=float)
    p = zperm.shape[1]
    coefs = list(range(p)) if btt is None else [int(b) - 1 for b in np.atleast_1d(btt)]
    n = len(coefs)
    fig, axes = plt.subplots(1, n, figsize=figsize or (3.4 * n, 3.0),
                             squeeze=False)
    flat = axes.ravel()
    names = list(x.coefnames or [f"b{j + 1}" for j in range(p)])
    if len(names) != p:
        names = [f"b{j + 1}" for j in range(p)]
    nb = int(nbin) if nbin is not None else max(5, int(math.ceil(math.sqrt(x.nperm))))

    for ax, j in zip(flat, coefs):
        arr = zperm[:, j]
        arr = arr[np.isfinite(arr)]
        if arr.size:
            ax.hist(arr, bins=nb, color=col, edgecolor="0.5", linewidth=0.4,
                    zorder=1)
        ax.axvline(zobs[j], color=col_obs, linewidth=1.4, zorder=3)
        pv = x.pval[j] if j < len(x.pval) else float("nan")
        ax.set_title(f"{names[j]}  (p = {pv:.4g})", fontsize=8)
        ax.set_xlabel("Test statistic", fontsize=7)
        ax.set_ylabel("Frequency", fontsize=7)
        ax.tick_params(labelsize=7)

    for ax in flat[n:]:
        ax.set_visible(False)
    fig.tight_layout()
    if filename:
        fig.savefig(filename)
    return PlotResult(ax=flat[0], figure=fig, filename=filename)


def plot_specification_curve(
    mv,
    *,
    filename=None,
    figsize=None,
    title=None,
    mark_default=None,
) -> PlotResult:
    """The specification curve, with the decisions underneath it.

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

    Args:
        mv: a :class:`~pimeta.multiverse.MultiverseResult`. Anything with
            ``curve()``, ``decisions`` and ``counts()`` works.
        filename: save the figure here (300 dpi); ``None`` leaves it in memory.
        figsize: figure size in inches; the default grows with the number of
            decisions, because a fixed height would crush them together.
        title: figure title; defaults to the counts and the range.
        mark_default: an optional specification id to highlight in the curve --
            the primary analysis inside its own multiverse.

    Returns:
        :class:`PlotResult` whose ``ax`` is the upper (curve) panel.

    Raises:
        ValueError: when no specification produced a usable estimate, because a
            specification curve of nothing is not a diagnostic.
    """
    rows = list(mv.curve())
    if not rows:
        raise ValueError(
            "no specification produced a usable estimate; there is no curve to "
            "draw (see MultiverseResult.failed for the reasons)")
    decisions = list(mv.decisions)
    counts = mv.counts()

    estimates = np.array([row["estimate"] for row in rows], dtype=float)
    ci_lb = np.array([row["ci_lb"] for row in rows], dtype=float)
    ci_ub = np.array([row["ci_ub"] for row in rows], dtype=float)
    positions = np.arange(estimates.size)

    n_panels = 1 + len(decisions)
    if figsize is None:
        figsize = (7.4, 2.4 + 1.05 * n_panels)
    # Constrained layout, not tight_layout: tight_layout warns (and lays out
    # wrongly) for a figure built from add_gridspec with shared axes.
    fig = plt.figure(figsize=figsize, layout="constrained")
    grid = fig.add_gridspec(
        n_panels, 1, height_ratios=[3.0] + [1.0] * len(decisions))
    ax_curve = fig.add_subplot(grid[0])

    ax_curve.errorbar(
        positions, estimates,
        yerr=[estimates - ci_lb, ci_ub - estimates],
        fmt="o", markersize=3.4, color="C0", ecolor="0.55",
        elinewidth=1.0, capsize=0, zorder=2)
    ax_curve.axhline(0.0, color="0.3", linewidth=0.9, linestyle="--", zorder=0)
    median = float(np.median(estimates))
    ax_curve.axhline(median, color="C3", linewidth=0.9, linestyle=":", zorder=0,
                     label=f"median {median:+.4f}")
    ax_curve.set_ylabel("estimate (interval per specification)", fontsize=8)
    ax_curve.tick_params(labelsize=7)
    ax_curve.legend(fontsize=7, loc="lower right", frameon=True,
                    framealpha=0.88, edgecolor="none")

    if mark_default is not None:
        hits = [i for i, row in enumerate(rows) if row["id"] == mark_default]
        if hits:
            i = hits[0]
            ax_curve.plot([positions[i]], [estimates[i]], "o", markersize=9,
                          markerfacecolor="none", markeredgecolor="C3",
                          markeredgewidth=1.4, zorder=3,
                          label="the primary analysis")
            ax_curve.annotate(f"{estimates[i]:+.4f}", (positions[i], estimates[i]),
                              textcoords="offset points", xytext=(8, 6),
                              fontsize=7, color="C3")

    markers = ("o", "s", "^", "D", "v", "P", "X", "*", "<", ">")
    for j, name in enumerate(decisions):
        ax = fig.add_subplot(grid[j + 1], sharex=ax_curve)
        levels: list = []
        for row in rows:
            value = row.get(name)
            if value not in levels:
                levels.append(value)
        for k, level in enumerate(levels):
            where = [i for i, row in enumerate(rows) if row.get(name) == level]
            ax.plot(where, np.full(len(where), k), markersize=3.4,
                    linestyle="none", marker=markers[k % len(markers)],
                    color=f"C{k % 10}")
        ax.set_yticks(range(len(levels)))
        ax.set_yticklabels([str(v) for v in levels], fontsize=7)
        ax.set_ylabel(name, fontsize=8, rotation=0, ha="right", va="center")
        ax.tick_params(labelsize=7)
        ax.grid(axis="x", color="0.9", linewidth=0.6)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)

    # The shared axis is ordinal: from the first specification to the last, with
    # a little room either side. Anything else would draw the margins against a
    # scale measured in effect sizes.
    ax_curve.set_xlim(-0.6, max(0.6, estimates.size - 0.4))
    low = min(float(ci_lb.min()), 0.0)
    high = max(float(ci_ub.max()), 0.0)
    pad = 0.06 * (high - low if high > low else 1.0)
    ax_curve.set_ylim(low - pad, high + pad)
    ax_curve.set_title(
        title if title is not None
        else (f"{counts['usable']} usable of {counts['considered']} "
              f"specification(s); estimate {estimates.min():+.4f} to "
              f"{estimates.max():+.4f}"
              + (f"; {counts['failed']} failed" if counts["failed"] else "")),
        fontsize=9, loc="left")
    for ax in fig.axes[:-1]:
        ax.tick_params(labelbottom=False)
    fig.axes[-1].set_xlabel("specification, ordered by estimate", fontsize=8)
    if filename:
        fig.savefig(filename, dpi=300, bbox_inches="tight")
    return PlotResult(ax=ax_curve, figure=fig, filename=filename)
