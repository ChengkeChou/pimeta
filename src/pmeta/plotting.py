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
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import matplotlib

matplotlib.use("Agg")  # headless-safe before importing pyplot
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Polygon

__all__ = [
    "forest",
    "funnel",
    "netgraph",
    "plot_baujat",
    "plot_gosh",
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
    :class:`~pmeta.pooling.PooledResult` (or any object
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
    if ax is None:
        h = max(4.0, 0.45 * k + 2.0)
        fig, ax = plt.subplots(figsize=figsize or (7.0, h))
    else:
        fig = ax.figure

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
            if ratio and ci_l is not None:
                ci_l, ci_u = math.log(ci_l), math.log(ci_u)
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
    labels = _studlab(studlab, k)

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.figure

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
        key = tuple(sorted((str(a), str(b))))
        pair_count[key] = pair_count.get(key, 0) + 1

    edges = []
    widths = []
    for (a, b), cnt in pair_count.items():
        if a in trts and b in trts and a != b:
            edges.append((a, b))
            widths.append(cnt)

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.figure

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

def _new_ax(ax, figsize):
    if ax is not None:
        return ax.figure, ax
    fig, ax = plt.subplots(figsize=figsize or (6.5, 5.0))
    return fig, ax


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
    from .rma_uni import RmaResult
    from scipy import stats as _st

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
    betas = np.array(betas)
    ses = np.array(ses)
    crit = _st.norm.isf((1.0 - level) / 2.0)

    fig, ax = _new_ax(ax, figsize)
    ax.plot(grid, betas, color="C0", lw=1.5, label="pooled estimate")
    ax.fill_between(grid, betas - crit * ses, betas + crit * ses,
                    color="C0", alpha=0.15, label=f"{int(level*100)}% CI")
    if prediction:
        pse = np.sqrt(ses ** 2 + grid)
        ax.plot(grid, betas - crit * pse, color="C3", lw=1.0, ls="--",
                label="prediction interval")
        ax.plot(grid, betas + crit * pse, color="C3", lw=1.0, ls="--")
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

    Args:
        result: fitted :class:`RmaResult`.
        type: ``"resid"`` (raw residuals), ``"rstandard"`` (standardized) or
            ``"rstudent"`` (studentized).

    Returns:
        :class:`PlotResult`.
    """
    from .rma_uni import RmaResult
    from scipy import stats as _st

    if not isinstance(result, RmaResult):
        raise TypeError("argument 'result' must be an RmaResult")
    resid = np.asarray(result.residuals, dtype=float)
    if type == "rstandard":
        resid = resid / np.sqrt(np.asarray(result.vi, dtype=float) + float(result.tau2))
    elif type == "rstudent":
        from scipy import stats as _s2
        vi = np.asarray(result.vi, dtype=float)
        tau2 = float(result.tau2)
        resid = resid / np.sqrt(np.maximum(vi + tau2 - np.diag(np.asarray(result.vb, dtype=float))[0], 0.0))
    elif type != "resid":
        raise ValueError("'type' must be 'resid', 'rstandard' or 'rstudent'")

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
    filename: str | None = None,
    figsize: tuple[float, float] | None = None,
    title: str = "Heat plot",
    ax: plt.Axes | None = None,
):
    """``netmeta::netheat`` — heat plot of the contribution to inconsistency.

    Renders the ``(design x comparison)`` matrix of the influence on the
    network's inconsistency as a colour map.

    Returns:
        :class:`PlotResult`.
    """
    mat = getattr(result, "heat_matrix", None)
    if mat is None:
        mat = getattr(result, "Q_inc_design", None)
    if mat is None:
        raise ValueError("result does not expose an inconsistency contribution matrix")
    mat = np.asarray(mat, dtype=float)
    if mat.ndim == 1:
        mat = mat.reshape(1, -1)

    fig, ax = _new_ax(ax, figsize)
    im = ax.imshow(mat, aspect="auto", cmap="RdYlGn_r", interpolation="nearest")
    fig.colorbar(im, ax=ax, shrink=0.8)
    labels = getattr(result, "designs", None)
    if labels is not None and len(labels) == mat.shape[0]:
        ax.set_yticks(range(mat.shape[0]))
        ax.set_yticklabels([str(v) for v in labels], fontsize=7)
    ax.set_xlabel("Comparison")
    ax.set_ylabel("Design")
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

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.figure

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

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.figure

    ax.scatter(xv, yv, s=25, c="0.3", edgecolors="k", linewidths=0.3, alpha=0.8)
    ax.set_xlabel(x)
    ax.set_ylabel(y)
    if title:
        ax.set_title(title)

    _save(fig, filename)
    return ax


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
    from scipy import stats as _stats

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
