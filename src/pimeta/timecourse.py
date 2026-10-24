"""Time-course meta-analysis: one effect per study per time point.

A trial that reports the treatment effect at 3, 6 and 12 months reports three
numbers that are not independent — the same participants produce all three.  A
time-course analysis therefore has two things to get right at once:

- the **within-study covariance** between the estimates at different times, and
  :func:`~pimeta.mv.vcov.vcalc` supplies it: ``rho`` for a constant
  correlation or ``phi`` for an AR(1) decay ``phi**|t_i - t_j|``.  It cannot be
  estimated from the reported summaries, so it is an assumption and the
  documentation says so;
- the **shape over time**, which is a curve rather than a set of point
  estimates, and the coefficients describing it are estimated jointly so that
  a contrast between two time points has a covariance.

The model is

.. math::

    y_{it} = b(t)'\\beta + b(t)'u_i + e_{it}, \\qquad
    u_i \\sim N(0, \\Sigma), \\quad e_i \\sim N(0, V_i)

with :math:`b(t)` a basis over time — linear by default, a restricted cubic
spline when ``knots`` or ``n_knots`` is given — :math:`V_i` the within-study
covariance built from the reported standard errors and ``rho`` or ``phi``, and
:math:`\\Sigma` estimated across studies.  ``timecourse(...).curve()`` returns
the pooled curve with a confidence band and a prediction band for a new
study's curve, and :meth:`TimeCourseResult.test_trend` tests whether the effect
changes with time at all.

Scope, stated rather than implied: this is the **pairwise** model, treatment
against a common comparator.  A network version needs the consistency equations
of a network meta-analysis on top and is not attempted here.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from scipy import stats

from .basis import Basis, rcs
from .levels import level_to_alpha, level_to_percent
from .rma_mv import rma_mv

__all__ = ["TimeCourseCurve", "TimeCourseResult", "timecourse"]


@dataclass
class TimeCourseCurve:
    """The pooled effect over time, with both bands."""

    x: np.ndarray = field(repr=False)
    est: np.ndarray = field(repr=False)
    se: np.ndarray = field(repr=False)
    ci_lb: np.ndarray = field(repr=False)
    ci_ub: np.ndarray = field(repr=False)
    pred_lb: np.ndarray = field(repr=False)
    pred_ub: np.ndarray = field(repr=False)
    outside: np.ndarray = field(repr=False)

    def to_dict(self) -> dict:
        return {
            "x": self.x.tolist(),
            "est": self.est.tolist(),
            "se": self.se.tolist(),
            "ci_lb": self.ci_lb.tolist(),
            "ci_ub": self.ci_ub.tolist(),
            "pred_lb": self.pred_lb.tolist(),
            "pred_ub": self.pred_ub.tolist(),
            "outside": self.outside.tolist(),
        }


@dataclass
class TimeCourseResult:
    """A pooled time-course model.

    Attributes:
        names: coefficient names, intercept first.
        coef, se: the pooled curve's coefficients.
        vcov: their joint covariance.
        tau2: the between-study variances, one per basis column.
        Sigma: the between-study covariance of the curve's coefficients.
        k: studies contributing.
        studies: their labels.
        support: the observed range of time.
        rho, phi: which within-study correlation was assumed, or ``None``.
        method, level, struct: settings.
        basis: the basis over time.
        fit: the underlying :class:`~pimeta.RmaMvResult`.
    """

    names: list[str]
    coef: list[float]
    se: list[float]
    vcov: np.ndarray = field(repr=False)
    tau2: list[float]
    Sigma: np.ndarray = field(repr=False)
    k: int
    studies: list
    support: tuple[float, float]
    method: str
    level: float
    struct: str
    basis: Basis | None = field(repr=False)
    fit: Any = field(repr=False)
    center: float = 0.0
    rho: float | None = None
    phi: float | None = None

    def _design(self, x) -> np.ndarray:
        z = np.asarray(x, dtype=float).ravel() - self.center
        if self.basis is None:
            return np.column_stack([np.ones(z.size), z])
        b = self.basis.evaluate(z).values
        return np.column_stack([np.ones(b.shape[0]), b])

    def _crit(self, level) -> float:
        alpha = level_to_alpha(self.level if level is None else level)
        return float(stats.norm.ppf(1.0 - alpha / 2.0))

    def curve(self, x=None, n: int = 50, level=None) -> TimeCourseCurve:
        """The pooled effect over time, with a confidence and a prediction band."""
        if x is None:
            lo, hi = self.support
            xv = np.linspace(lo, hi, int(n))
        else:
            xv = np.asarray(x, dtype=float).ravel()
        design = self._design(xv)
        est = design @ np.asarray(self.coef, dtype=float)
        vcov = np.asarray(self.vcov, dtype=float)
        var = np.einsum("gi,ij,gj->g", design, vcov, design)
        var_pred = np.einsum("gi,ij,gj->g", design, vcov + self.Sigma, design)
        se = np.sqrt(np.maximum(var, 0.0))
        se_pred = np.sqrt(np.maximum(var_pred, 0.0))
        crit = self._crit(level)
        lo, hi = self.support
        return TimeCourseCurve(
            x=xv, est=est, se=se,
            ci_lb=est - crit * se, ci_ub=est + crit * se,
            pred_lb=est - crit * se_pred, pred_ub=est + crit * se_pred,
            outside=(xv < lo) | (xv > hi),
        )

    def _joint(self, rows) -> dict:
        idx = np.asarray(rows, dtype=int)
        b = np.asarray(self.coef, dtype=float)[idx]
        V = np.asarray(self.vcov, dtype=float)[np.ix_(idx, idx)]
        stat = float(b @ np.linalg.solve(V, b))
        return {"stat": stat, "df": int(idx.size),
                "pval": float(stats.chi2.sf(stat, idx.size))}

    def test_trend(self) -> dict:
        """Does the effect change with time at all?

        The null is that every coefficient beyond the intercept is zero, tested
        jointly.  With a linear basis that is one degree of freedom; with a
        spline it is K, and testing the columns one at a time would be testing
        the parameterisation rather than the shape.
        """
        return self._joint(range(1, len(self.coef)))

    def test_nonlinear(self) -> dict:
        """Are the non-linear basis columns needed?  Linear basis: nothing to test."""
        if len(self.coef) <= 2:
            raise ValueError(
                "the basis has fewer than three columns, so there is no "
                "non-linear term to test; pass knots= or n_knots= for a spline")
        return self._joint(range(2, len(self.coef)))

    def coef_table(self) -> list[dict]:
        rows = []
        for name, b, se in zip(self.names, self.coef, self.se):
            rows.append({"term": name, "estimate": float(b), "se": float(se),
                         "zval": float(b / se) if se > 0 else float("nan"),
                         "pval": float(2.0 * stats.norm.sf(abs(b / se)))
                         if se > 0 else float("nan")})
        return rows

    def to_dict(self) -> dict:
        return {
            "names": list(self.names),
            "coef": list(self.coef),
            "se": list(self.se),
            "vcov": np.asarray(self.vcov).tolist(),
            "tau2": list(self.tau2),
            "Sigma": np.asarray(self.Sigma).tolist(),
            "k": self.k,
            "studies": list(self.studies),
            "support": list(self.support),
            "method": self.method,
            "level": self.level,
            "struct": self.struct,
            "center": self.center,
            "rho": self.rho,
            "phi": self.phi,
            "basis": None if self.basis is None else self.basis.to_dict(),
        }

    def __str__(self) -> str:
        lo, hi = self.support
        lines = [
            "time-course meta-analysis",
            f"  {self.k} studies, time {lo:g} to {hi:g}, "
            f"basis {', '.join(self.names)}",
        ]
        if self.phi is not None:
            lines.append(f"  within-study correlation: AR(1) with phi = {self.phi:g}")
        elif self.rho is not None:
            lines.append(f"  within-study correlation: rho = {self.rho:g}")
        lines.append("")
        for row in self.coef_table():
            lines.append(f"  {row['term']:<12}{row['estimate']:>10.4f}"
                         f"{row['se']:>9.4f}")
        try:
            trend = self.test_trend()
            lines.append("")
            lines.append(f"  effect changes with time: chi2({trend['df']}) = "
                         f"{trend['stat']:.3f}, p = {trend['pval']:.4g}")
        except ValueError:
            pass
        return "\n".join(lines)


def timecourse(yi, *, study, time, se=None, vi=None, rho=None, phi=None,
               knots=None, n_knots: int = 0, center=None,
               method: str = "REML", level: float = 0.95,
               test: str = "z") -> TimeCourseResult:
    """Pool treatment effects reported at several time points per study.

    Args:
        yi: the effect sizes, one row per study and time point.
        study: the study label for each row.
        time: the time of each estimate.
        se: its standard error; alternatively ``vi``.
        vi: the sampling variance, if ``se`` is not given.
        rho: a constant within-study correlation between a study's estimates;
            one of ``rho`` or ``phi`` is required, because the alternative —
            treating them as independent when they are not — silently
            overstates the precision of the curve.
        phi: an AR(1) decay, giving ``phi**|t_i - t_j|``.
        knots: knot positions for the spline basis over time.
        n_knots: knots for a spline basis; ``0`` (the default) means a linear
            trend.  A spline fits a shape the data may not support: with three
            time points and a dozen studies the extra coefficients buy a
            curvature the between-study covariance cannot pay for.
        center: time at which the intercept is the effect; the median by
            default, because at time zero the intercept is an extrapolation.
        method: ``"REML"`` or ``"ML"``.
        level: confidence level.
        test: reference distribution for the coefficient tests.

    Returns:
        :class:`TimeCourseResult`.

    Raises:
        ValueError: for mismatched lengths, a study with one time point and no
            correlation (nothing to correlate), neither ``rho`` nor ``phi``, a
            ``phi`` outside ``[-1, 1]``, or fewer than two studies.
    """
    yi = np.asarray(yi, dtype=float).ravel()
    time = np.asarray(time, dtype=float).ravel()
    labels = [str(s) for s in np.asarray(study).ravel()]
    n = yi.size
    if not (time.size == len(labels) == n):
        raise ValueError("'yi', 'study' and 'time' must be the same length")
    if se is None and vi is None:
        raise ValueError("give 'se' or 'vi'")
    if vi is not None:
        vi = np.asarray(vi, dtype=float).ravel()
        if vi.size != n:
            raise ValueError("'vi' must be the same length as 'yi'")
        if np.any(vi <= 0.0):
            raise ValueError("every sampling variance must be positive")
        se = np.sqrt(vi)
    else:
        se = np.asarray(se, dtype=float).ravel()
        if se.size != n:
            raise ValueError("'se' must be the same length as 'yi'")
        if np.any(se <= 0.0):
            raise ValueError("every standard error must be positive")
    if rho is None and phi is None:
        raise ValueError(
            "give 'rho' (a constant correlation) or 'phi' (AR(1) decay): a "
            "study's estimates at different times are not independent, and "
            "treating them as such is the error this model exists to avoid")
    if phi is not None and not -1.0 <= float(phi) <= 1.0:
        raise ValueError("'phi' must lie in [-1, 1]")
    if rho is not None and not -1.0 <= float(rho) <= 1.0:
        raise ValueError("'rho' must lie in [-1, 1]")
    if test not in ("z", "t"):
        raise ValueError(f"unknown test: {test!r}; expected 'z' or 't'")

    studies = list(dict.fromkeys(labels))
    if len(studies) < 2:
        raise ValueError("at least two studies are needed to pool")

    sh = 0.0 if center is None else float(center)
    if center is None:
        sh = float(np.median(time))
    if knots is not None:
        basis = rcs(time - sh, knots=np.asarray(knots, float).ravel() - sh,
                    name="time")
    elif n_knots and n_knots > 2:
        basis = rcs(time - sh, nk=int(n_knots), name="time")
    else:
        basis = None
    if basis is None:
        B_all = (time - sh).reshape(-1, 1)
        basis_names = ["time"]
    else:
        B_all = np.asarray(basis.evaluate(time - sh).values, dtype=float)
        basis_names = list(basis.names)
    K = int(B_all.shape[1])
    if K < 1:
        raise ValueError("the basis has no columns")

    V = np.zeros((n, n))
    rows_by_study: dict[str, np.ndarray] = {}
    for g in studies:
        idx = np.array([i for i, lab in enumerate(labels) if lab == g])
        rows_by_study[g] = idx
        block = np.zeros((idx.size, idx.size))
        for a in range(idx.size):
            for b in range(idx.size):
                if a == b:
                    block[a, b] = se[idx[a]] ** 2
                else:
                    dt = abs(time[idx[a]] - time[idx[b]])
                    corr = (float(phi) ** dt if phi is not None
                            else float(rho))
                    block[a, b] = corr * se[idx[a]] * se[idx[b]]
        V[np.ix_(idx, idx)] = block

    names = ["(Intercept)"] + basis_names
    X = np.column_stack([np.ones(n), B_all])
    # the random-effects formula needs one column per basis function, and the
    # spline's own names (time, time', ...) are not formula-safe
    col_names = [f"b{j + 1}" for j in range(K)]
    data = {"study": labels}
    for j, nm in enumerate(col_names):
        data[nm] = B_all[:, j]
    # struct="GEN" is the structure that means what is wanted here: the basis
    # columns as random slopes with an unstructured covariance, which is the
    # random curve b(t)'u_i itself.
    random = "~ " + " + ".join(col_names) + " | study"
    fit = rma_mv(yi, V=V, mods=X, data=data, random=random, struct="GEN",
                 intercept=False, method=method, level=level)
    n_vc = (K + 1) * (K + 2) // 2
    if len(studies) < 2 * n_vc:
        warnings.warn(
            f"{len(studies)} studies for {n_vc} variance parameters in the "
            "random curve, so the between-study covariance is unlikely to be "
            "identified; a linear basis or fewer knots is the usual answer",
            RuntimeWarning, stacklevel=2)

    coef = list(np.asarray(fit.beta, dtype=float))
    vcov = np.asarray(fit.vb, dtype=float)
    G, _gnames = fit.g_matrix()
    Sigma = (np.asarray(G, dtype=float) if G is not None
             else np.zeros((K + 1, K + 1)))
    return TimeCourseResult(
        names=names, coef=[float(v) for v in coef],
        se=[float(v) for v in np.sqrt(np.diag(vcov))],
        vcov=vcov, tau2=list(np.diag(Sigma)), Sigma=Sigma,
        k=len(studies), studies=studies,
        support=(float(np.min(time)), float(np.max(time))),
        method=method, level=level_to_percent(level), struct="GEN",
        basis=basis, fit=fit, center=sh,
        rho=None if rho is None else float(rho),
        phi=None if phi is None else float(phi),
    )
