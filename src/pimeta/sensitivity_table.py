"""A sensitivity table over one fitted model.

Sensitivity asks one question: *what happens to this estimate if one assumption
changes?* It is not a multiverse and it does not average anything -- each row is
the same estimand under one variant, and the point is to see how far the answer
travels when a choice that was not forced by the data is changed.

Three families are covered:

``tau2``
    The heterogeneity estimator. Every method refits the same data; the row
    difference is only the estimator, which is the choice most often left
    unexamined and least often defended.
``leave1out``
    One study removed at a time, with that study's Cook's distance and hat
    value attached, so a row that moves the estimate can be recognised as
    influential rather than merely different.
``bias``
    Publication-bias adjustments that keep ``mu`` as the estimand: the limit
    meta-analysis of Rücker et al. at two extrapolation points, a step selection
    model, and the RoBMA posterior mean.

A variant that is statistically invalid for a given fit is **refused with a
reason**, in ``refused``, rather than attempted and reported. A table that
quietly omits the rows it could not compute is worse than no table: the reader
cannot tell an omission from a null result.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from scipy import stats

from .levels import level_to_alpha, level_to_percent
from .rma_list import RowSlicable
from .sensitivity import influence

__all__ = ["SensitivityRow", "SensitivityResult", "FAMILIES", "sensitivity_table"]

#: Estimators a ``tau2`` row can be built from. ``FE`` is included on purpose:
#: "what if there were no heterogeneity at all" is a sensitivity question even
#: though it is a different model. ``MP`` is absent because metafor treats it as
#: an alias of ``PM``, and a table that lists the same estimator twice is a
#: table that looks more thorough than it is.
METHODS = ("REML", "DL", "HE", "HS", "SJ", "PM", "ML", "EB", "FE")

#: Families this function knows about.
FAMILIES = ("tau2", "leave1out", "bias")


@dataclass
class SensitivityRow:
    """One variant of the same estimand.

    Attributes:
        family: ``"baseline"``, ``"tau2"``, ``"leave1out"`` or ``"bias"``.
        variant: what changed, in words a reader can act on.
        estimate, se, ci_lb, ci_ub: the estimand under this variant.
        tau2: the heterogeneity estimate under this variant, when it has one.
        detail: anything else worth keeping (Cook's distance, hat value, ...).
        reason: why this variant is worth looking at.
    """

    family: str
    variant: str
    estimate: float
    se: float
    ci_lb: float
    ci_ub: float
    tau2: float | None = None
    detail: dict[str, Any] = field(default_factory=dict)
    reason: str = ""

    def to_dict(self) -> dict:
        return {
            "family": self.family,
            "variant": self.variant,
            "estimate": float(self.estimate),
            "se": float(self.se),
            "ci_lb": float(self.ci_lb),
            "ci_ub": float(self.ci_ub),
            "tau2": None if self.tau2 is None else float(self.tau2),
            "detail": dict(self.detail),
            "reason": self.reason,
        }


@dataclass
class SensitivityResult(RowSlicable):
    """A baseline estimate and the rows that perturb it."""

    baseline: SensitivityRow
    rows: list[SensitivityRow]
    refused: list[dict[str, str]]
    families: list[str]
    level: float
    k: int
    p: int

    def family(self, name: str) -> list[SensitivityRow]:
        return [r for r in self.rows if r.family == name]

    def range(self, name: str) -> tuple[float, float]:
        """Smallest and largest estimate among the rows of one family."""
        rows = self.family(name)
        if not rows:
            raise ValueError(f"no rows for family {name!r}")
        values = [r.estimate for r in rows]
        return float(min(values)), float(max(values))

    def to_dict(self) -> dict:
        return {
            "baseline": self.baseline.to_dict(),
            "rows": [r.to_dict() for r in self.rows],
            "refused": [dict(d) for d in self.refused],
            "families": list(self.families),
            "level": float(self.level),
            "k": int(self.k),
            "p": int(self.p),
        }

    def __str__(self) -> str:
        b = self.baseline
        width = max([len(r.variant) for r in self.rows] + [len("baseline")])
        lines = [f"Sensitivity of the pooled estimate (k = {self.k}, "
                 f"p = {self.p}, "
                 f"{level_to_percent(self.level):g}% intervals)", ""]
        lines.append(f"  {'baseline':<{width}}  {b.estimate:+.4f}  "
                     f"[{b.ci_lb:+.4f}, {b.ci_ub:+.4f}]")
        current = None
        for r in self.rows:
            if r.family != current:
                current = r.family
                lo, hi = self.range(current)
                lines.append("")
                lines.append(f"{current}: "
                             f"{len(self.family(current))} variant(s), "
                             f"estimate from {lo:+.4f} to {hi:+.4f}")
            lines.append(f"  {r.variant:<{width}}  {r.estimate:+.4f}  "
                         f"[{r.ci_lb:+.4f}, {r.ci_ub:+.4f}]")
        for item in self.refused:
            lines.append("")
            lines.append(f"refused ({item['family']}): {item['reason']}")
        return "\n".join(lines)


def _row_from_fit(fit, family: str, variant: str, reason: str,
                  alpha: float) -> SensitivityRow:
    se = float(fit.se[0])
    est = float(fit.b[0])
    crit = float(stats.norm.ppf(1.0 - alpha / 2.0))
    return SensitivityRow(
        family=family, variant=variant, estimate=est, se=se,
        ci_lb=float(fit.ci_lb[0]), ci_ub=float(fit.ci_ub[0]),
        tau2=float(fit.tau2), detail={"crit": crit}, reason=reason)


def _tau2_rows(fit, methods, level: float) -> list[SensitivityRow]:
    from .rma_uni import rma

    out: list[SensitivityRow] = []
    for method in methods:
        if str(method).upper() == str(fit.method).upper():
            continue
        refit = rma(np.asarray(fit.yi, dtype=float),
                    vi=np.asarray(fit.vi, dtype=float),
                    mods=None if fit.mods is None else np.asarray(fit.mods),
                    method=method, level=level,
                    test=fit.test if fit.test in ("z", "t", "knha", "adhoc")
                    else "z")
        if str(method).upper() == "EB":
            why = ("iterative empirical Bayes, which converges to PM by "
                   "construction, so this row is a convergence check rather "
                   "than an alternative")
        else:
            why = (f"the same data under the {method} heterogeneity estimator; "
                   f"the fit itself used {fit.method}")
        out.append(_row_from_fit(refit, "tau2", f"tau2 = {method}", why,
                                 level_to_alpha(level)))
    return out


def _leave1out_rows(fit, alpha: float,
                    refused: list[dict[str, str]]) -> list[SensitivityRow]:
    # influence() takes the moderators *without* an intercept column: it adds
    # one itself. Passing fit.X (which already has one) makes X'X singular.
    mods = None if fit.mods is None else np.asarray(fit.mods, dtype=float)
    method = str(fit.method).upper()
    if method not in ("FE", "DL", "ML", "REML"):
        refused.append({
            "family": "leave1out",
            "reason": f"the leave-one-out refits are only implemented for "
                      f"FE/DL/ML/REML and this fit used {fit.method}; the "
                      "rows below are REML, which is a different estimator "
                      "from the one that produced the baseline"})
        method = "REML"
    inf = influence(np.asarray(fit.yi, dtype=float),
                    np.asarray(fit.vi, dtype=float),
                    X=mods, method=method,
                    test=fit.test if fit.test in ("z", "knha", "adhoc")
                    else "z",
                    alpha=alpha)
    crit = float(stats.norm.ppf(1.0 - alpha / 2.0))
    labels = list(getattr(fit, "slab", []) or [])
    out: list[SensitivityRow] = []
    for i in range(inf.k):
        est = float(inf.beta_del[i][0])
        se = float(inf.se_del[i][0])
        name = labels[i] if i < len(labels) else f"study {i + 1}"
        out.append(SensitivityRow(
            family="leave1out", variant=f"without {name}",
            estimate=est, se=se,
            ci_lb=est - crit * se, ci_ub=est + crit * se,
            tau2=float(inf.tau2_del[i]),
            detail={"cook_d": float(inf.cook_d[i]), "hat": float(inf.hat[i]),
                    "cooks_distance_ok": crit,
                    "influential": bool(inf.is_infl[i])},
            reason="one study removed; an estimate that moves here moves "
                   "because of that study, not because of the evidence"))
    return out


def _bias_rows(fit, level: float, refused: list[dict[str, str]]
               ) -> list[SensitivityRow]:
    from .robma import robma
    from .selmodel import selmodel

    pct = level_to_percent(level)
    out: list[SensitivityRow] = []
    limitmeta_fn: Callable[..., Any] | None
    try:
        from .limitmeta import limitmeta
        limitmeta_fn = limitmeta
    except ImportError:  # pragma: no cover - module is part of the package
        refused.append({"family": "bias",
                        "reason": "the limit meta-analysis is not available"})
        limitmeta_fn = None
    if limitmeta_fn is not None:
        for adjust, why in (
            ("beta0", "the limit estimate, extrapolating to infinite precision"),
            ("betalim", "the limit estimate with the slope fixed at the "
                         "radial-regression value"),
            ("mulim", "the limit estimate under the random-effects model"),
        ):
            try:
                adj = limitmeta_fn(fit, method_adjust=adjust, level=pct)
            except Exception as exc:  # noqa: BLE001 - a refusal is a result
                refused.append({"family": "bias",
                                "reason": f"limitmeta({adjust}): {exc}"})
                continue
            out.append(SensitivityRow(
                family="bias", variant=f"limit meta-analysis ({adjust})",
                estimate=float(adj.TE_adjust), se=float(adj.seTE_adjust),
                ci_lb=float(adj.ci_lb), ci_ub=float(adj.ci_ub),
                tau2=float(adj.tau), reason=why))
    try:
        sel = selmodel(fit, type="step", steps=[0.05])
        out.append(SensitivityRow(
            family="bias", variant="selection model (step, p < 0.05)",
            estimate=float(sel.b[0]), se=float(sel.se[0]),
            ci_lb=float(sel.ci_lb[0]), ci_ub=float(sel.ci_ub[0]),
            tau2=float(sel.tau2),
            detail={"p_selected": float(sel.pvals[0]) if sel.pvals else None},
            reason="the estimate if smaller studies are down-weighted by a "
                   "one-step selection function"))
    except Exception as exc:  # noqa: BLE001
        refused.append({"family": "bias",
                        "reason": f"selection model: {exc}"})
    try:
        rb = robma(np.asarray(fit.yi, dtype=float),
                   sei=np.sqrt(np.asarray(fit.vi, dtype=float)),
                   level=pct)
        mu = rb.mu
        out.append(SensitivityRow(
            family="bias", variant="RoBMA posterior mean",
            estimate=float(mu["mean"]), se=float(mu["sd"]),
            ci_lb=float(mu["q_low"]), ci_ub=float(mu["q_high"]),
            tau2=float(rb.tau["mean"]),
            detail={"p_mu_zero": float(mu["point_mass"]),
                    "p_tau_zero": float(rb.tau["point_mass"])},
            reason="model-averaged over no bias, PET, PEESE and selection, "
                   "with the posterior probability of no effect and of no "
                   "heterogeneity reported alongside it"))
    except Exception as exc:  # noqa: BLE001
        refused.append({"family": "bias", "reason": f"RoBMA: {exc}"})
    return out


def sensitivity_table(fit, *, over=FAMILIES, methods=None, level=None
                      ) -> SensitivityResult:
    """The same estimand under each assumption that could have been different.

    Args:
        fit: an :class:`~pimeta.RmaResult` from :func:`~pimeta.rma`.
        over: which families to compute; a subset of ``("tau2", "leave1out",
            "bias")``.  ``"rho"`` is accepted only to be refused with a reason,
            because a univariate fit has no within-study correlation.
        methods: heterogeneity estimators for the ``tau2`` family; defaults to
            every entry of ``METHODS`` except the one the fit used.
        level: confidence level; defaults to the fit's own.

    Returns:
        :class:`SensitivityResult` with a baseline row, one row per variant and
        a ``refused`` list naming each variant that was not attempted and why.
        Asking only for a family that is refused returns that refusal and an
        empty row list: the reason the variant is invalid *is* the answer, so
        it is not turned into an exception the caller has to guess at.

    Raises:
        TypeError: when ``fit`` is not an ``RmaResult``.
        ValueError: for an unknown family, for a fit that carries no data, or
            for a zero-information ``over``.

    Example:
        >>> res = sensitivity_table(fit)                      # doctest: +SKIP
        >>> res.range("tau2")                                 # doctest: +SKIP
        >>> print(res)                                        # doctest: +SKIP
    """
    from .rma_uni import RmaResult

    if not isinstance(fit, RmaResult):
        raise TypeError("'fit' must be an RmaResult from rma()")
    wanted = [str(o) for o in over]
    unknown = [o for o in wanted if o not in FAMILIES and o != "rho"]
    if unknown:
        raise ValueError(
            f"unknown family {unknown[0]!r}; expected a subset of "
            f"{FAMILIES} (plus 'rho', which is refused for a fit like this)")
    if getattr(fit, "yi", None) is None or getattr(fit, "vi", None) is None:
        raise ValueError(
            "the fit does not carry its data, so nothing can be refitted")

    conf = fit.level if level is None else float(level)
    # ``RmaResult.level`` is a proportion (0.95), and the hand-written
    # normalisation that used to be here read it as an alpha, which produced
    # leave-one-out intervals of width +/-0.006 instead of +/-0.18.
    alpha = level_to_alpha(conf)
    refused: list[dict[str, str]] = []

    se0 = float(fit.se[0])
    baseline = SensitivityRow(
        family="baseline", variant="as fitted",
        estimate=float(fit.b[0]), se=se0,
        ci_lb=float(fit.ci_lb[0]), ci_ub=float(fit.ci_ub[0]),
        tau2=float(fit.tau2),
        detail={"method": fit.method, "test": fit.test},
        reason="the reported estimate")

    rows: list[SensitivityRow] = []
    computed: list[str] = []
    if "tau2" in wanted:
        chosen = METHODS if methods is None else tuple(methods)
        rows.extend(_tau2_rows(fit, chosen, level=conf))
        computed.append("tau2")
    if "leave1out" in wanted:
        rows.extend(_leave1out_rows(fit, alpha=alpha, refused=refused))
        computed.append("leave1out")
    if "bias" in wanted:
        if int(fit.p) > 1:
            refused.append({
                "family": "bias",
                "reason": "every publication-bias adjustment here is a "
                          "property of a pooled estimate; this fit has "
                          f"{int(fit.p)} coefficients, so there is no single "
                          "estimate for it to adjust"})
        elif int(fit.k) < 8:
            refused.append({
                "family": "bias",
                "reason": f"k = {int(fit.k)}: with fewer than eight studies "
                          "the selection function and the radial slope are not "
                          "identifiable, and an adjusted number would be a "
                          "choice of prior rather than a finding"})
        else:
            rows.extend(_bias_rows(fit, conf, refused))
            computed.append("bias")
    if "rho" in wanted:
        refused.append({
            "family": "rho",
            "reason": "a univariate fit has one outcome and therefore no "
                      "within-study correlation; use rho_sensitivity() on a "
                      "multivariate fit, where the assumption exists"})
    if not rows and not refused:
        raise ValueError(
            "no family was computed and none was refused; ask for at least "
            f"one of {FAMILIES}")

    return SensitivityResult(
        baseline=baseline, rows=rows, refused=refused, families=computed,
        level=conf, k=int(fit.k), p=int(fit.p))
