"""Time-course network meta-analysis.

A time-course network asks how the *difference* between two treatments evolves,
not just whether there is one.  Each study contributes contrasts at the times it
measured,

.. math::

    y_{i,t} = g_b(t; \\beta_b) - g_a(t; \\beta_a) + e_{i,t},

and the same :math:`\\beta_t` is used wherever treatment *t* appears, so the
network's consistency equation cancels at **every** time — including the times
no study measured.  That is the whole reason the parameterisation is per
treatment rather than per comparison: see :mod:`pimeta.nma.dose`, which makes
the same argument for dose.

Two things are specific to time.

**The baseline.**  With a constant term in :math:`g`, the difference at
:math:`t = 0` is the ordinary network meta-analysis treatment effect, and the
model at :math:`t = 0` *is* standard NMA — that is the time-invariant limit and
the tests pin it against `netmeta`.  Dropping the constant instead asserts that
two treatments cannot differ at baseline, which is true by randomisation and is
a real restriction on the fit rather than a convenience.
``basis="constant"`` keeps only the constant term, which is standard NMA with
the usual interpretation.

**The correlation.**  A study that reports the effect at 3, 6 and 12 months
measured the same participants three times, so its contrasts are correlated and
the within-study covariance is not diagonal.  The correlation cannot be
estimated from the reported summaries, so it is an assumption, spelled out:
``rho`` for a constant correlation, or ``phi`` for the AR(1) decay
``phi**|t_i - t_j|`` that :func:`~pimeta.mv.vcov.vcalc` builds.  It is required
rather than defaulted, because treating a study's repeated measurements as
independent makes a time trend look more precise than the data can support.

Bases over time
---------------

``constant``
    One parameter per treatment: standard network meta-analysis.
``linear``
    ``b0 + b1 t``, the default and the form most time-course analyses report.
``quadratic``
    ``b0 + b1 t + b2 t**2``.
``rcs``
    A restricted cubic spline in time, knots given or placed on the observed
    times; the constant term is included, so the spline is unconstrained at
    zero.

Heterogeneity
-------------

``struct="FE"`` treats the studies as fixed.  ``struct="study"`` (the default)
adds a random effect on the study's level, which is what a time-course
meta-analysis usually means by random effects.  ``struct="coef"`` adds a random
effect on the **curve**, so a study's whole trajectory may deviate, and
``"coef+study"`` both.  Because every time basis here is linear in its
parameters, all four are the same generalised least squares problem with
different between-study covariances, and no numerical integration is involved.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from scipy import stats
from scipy.optimize import minimize, minimize_scalar

from ..basis import _rcs_matrix, default_knots
from ..levels import level_to_alpha
from .dose import _chol, _log_cholesky, _quadratic_form

__all__ = [
    "TimeCourseNmaCurve",
    "TimeCourseNmaResult",
    "nma_timecourse",
    "time_bases",
]

_BASES = ("constant", "linear", "quadratic", "rcs")
_PARAMETER_NAMES = {
    "constant": ("intercept",),
    "linear": ("intercept", "time"),
    "quadratic": ("intercept", "time", "time^2"),
}


def time_bases() -> list[str]:
    """The bases over time this module can fit.

    Example:
        >>> ma.nma.timecourse.time_bases()       # doctest: +SKIP
        ['constant', 'linear', 'quadratic', 'rcs']
    """
    return list(_BASES)


def _design(basis: str, time, *, knots=None) -> np.ndarray:
    """The design row(s) ``g(t)`` — the constant term first, then the shape."""
    t = np.asarray(time, dtype=float).ravel()
    if not np.all(np.isfinite(t)):
        raise ValueError("'time' contains missing or infinite values")
    if basis == "constant":
        return np.ones((t.size, 1))
    if basis == "linear":
        return np.column_stack([np.ones(t.size), t])
    if basis == "quadratic":
        return np.column_stack([np.ones(t.size), t, t ** 2])
    if basis == "rcs":
        if knots is None:
            raise ValueError("the 'rcs' time basis needs 'knots'")
        return np.column_stack([np.ones(t.size),
                                _rcs_matrix(t, np.asarray(knots, dtype=float))])
    raise ValueError(
        f"unknown time basis {basis!r}; use one of "
        + ", ".join(repr(b) for b in _BASES))


def _n_parameters(basis: str, *, n_knots: int | None = None) -> int:
    if basis == "rcs":
        if n_knots is None:
            raise ValueError("the 'rcs' time basis needs 'n_knots'")
        return 1 + int(n_knots)
    return len(_PARAMETER_NAMES[basis])


def _names(basis: str, *, n_knots: int | None = None,
           label: str | None = None) -> list[str]:
    prefix = "" if label is None else f"{label}:"
    if basis == "rcs":
        if n_knots is None:
            raise ValueError("the 'rcs' time basis needs 'n_knots'")
        return ([f"{prefix}intercept"]
                + [f"{prefix}rcs[{j + 1}]" for j in range(int(n_knots))])
    return [f"{prefix}{name}" for name in _PARAMETER_NAMES[basis]]


@dataclass
class TimeCourseNmaCurve:
    """A treatment's trajectory, or a contrast between two of them."""

    labels: list[str] = field(default_factory=list)
    time: np.ndarray = field(repr=False, default_factory=lambda: np.empty(0))
    est: np.ndarray = field(repr=False, default_factory=lambda: np.empty(0))
    se: np.ndarray = field(repr=False, default_factory=lambda: np.empty(0))
    ci_lb: np.ndarray = field(repr=False, default_factory=lambda: np.empty(0))
    ci_ub: np.ndarray = field(repr=False, default_factory=lambda: np.empty(0))
    pred_lb: np.ndarray = field(repr=False, default_factory=lambda: np.empty(0))
    pred_ub: np.ndarray = field(repr=False, default_factory=lambda: np.empty(0))
    outside: np.ndarray = field(repr=False, default_factory=lambda: np.empty(0))

    def to_dict(self) -> dict:
        return {
            "labels": list(self.labels),
            "time": self.time.tolist(),
            "est": self.est.tolist(),
            "se": self.se.tolist(),
            "ci_lb": self.ci_lb.tolist(),
            "ci_ub": self.ci_ub.tolist(),
            "pred_lb": self.pred_lb.tolist(),
            "pred_ub": self.pred_ub.tolist(),
            "outside": self.outside.tolist(),
        }

    def __str__(self) -> str:
        head = "  ".join(self.labels) if self.labels else "time course"
        rows = [f"{head:<28}{'time':>8}{'est':>10}{'se':>9}{'ci_lb':>10}"
                f"{'ci_ub':>10}"]
        for i in range(self.time.size):
            rows.append(f"{'':<28}{self.time[i]:>8.4g}{self.est[i]:>10.4f}"
                        f"{self.se[i]:>9.4f}{self.ci_lb[i]:>10.4f}"
                        f"{self.ci_ub[i]:>10.4f}")
        return "\n".join(rows)


@dataclass
class TimeCourseNmaResult:
    """A fitted time-course network.

    Attributes:
        basis: the basis over time.
        treatments: every treatment in the network.
        curves: the treatments with a fitted trajectory.  A treatment seen at a
            single time has one, but only as many parameters as its data can
            support.
        names: coefficient names, one block per treatment, constant first.
        coef, se: the coefficients and their standard errors.
        vcov: their covariance.
        tau2: the between-study variance of the study level, or ``NaN`` when
            ``struct`` has no level term.
        D: the between-study covariance of the trajectories, or ``None``.
        k: contrasts fitted.
        studies: the study labels.
        times: the observed times per treatment.
        rho, phi: the within-study correlation that was assumed.
        struct, coef_struct, method, level: settings.
        loglik: the maximised (restricted) log-likelihood.
    """

    basis: str
    treatments: list[str]
    curves: list[str]
    names: list[str]
    coef: list[float]
    se: list[float]
    vcov: np.ndarray = field(repr=False, default_factory=lambda: np.empty((0, 0)))
    reference: str = ""
    tau2: float = float("nan")
    D: np.ndarray | None = field(repr=False, default=None)
    k: int = 0
    studies: list = field(default_factory=list)
    times: dict = field(default_factory=dict)
    rho: float | None = None
    phi: float | None = None
    struct: str = "study"
    coef_struct: str = "DIAG"
    method: str = "REML"
    level: float = 95.0
    loglik: float = float("nan")
    n_knots: int | None = None
    knots: list[float] | None = None
    fit: Any = field(repr=False, default=None)

    # -- prediction -------------------------------------------------------- #
    def _block_of(self, treat) -> int | None:
        name = str(treat)
        if name not in self.treatments:
            raise ValueError(f"{treat!r} is not a treatment in this network")
        return self.curves.index(name) if name in self.curves else None

    def _params(self, block: int) -> np.ndarray:
        p = _n_parameters(self.basis, n_knots=self.n_knots)
        return np.asarray(self.coef, dtype=float)[block * p:(block + 1) * p]

    def _value(self, treat, time) -> np.ndarray:
        """``g_t(t)``: zero for the reference treatment, by construction."""
        block = self._block_of(treat)
        time = np.asarray(time, dtype=float).ravel()
        if block is None:
            return np.zeros(time.size)
        return _design(self.basis, time, knots=self.knots) @ self._params(block)

    def _jacobian(self, treat, time) -> np.ndarray:
        p = _n_parameters(self.basis, n_knots=self.n_knots)
        time = np.asarray(time, dtype=float).ravel()
        out = np.zeros((time.size, len(self.coef)))
        block = self._block_of(treat)
        if block is None:
            return out
        out[:, block * p:(block + 1) * p] = _design(
            self.basis, time, knots=self.knots)
        return out

    def _crit(self) -> float:
        return float(stats.norm.ppf(0.5 + self.level / 200.0))

    def curve(self, treat, times=None, *, level: float | None = None,
              n: int = 50) -> TimeCourseNmaCurve:
        """One treatment's effect over time, against the reference.

        ``g_t(t)`` is measured from the reference treatment's trajectory, so
        ``curve`` and ``contrast`` report the same kind of quantity and the
        reference treatment itself is a flat zero by definition.
        """
        if times is None:
            seen = self.times.get(str(treat)) or [0.0]
            grid = np.linspace(min(seen), max(seen), int(n))
        else:
            grid = np.asarray(times, dtype=float).ravel()
        return self._band([(str(treat), grid, 1.0)], level=level,
                          labels=[f"{treat}"])

    def contrast(self, treat_a, time_a, treat_b, time_b, *,
                 level: float | None = None) -> TimeCourseNmaCurve:
        """``g_b(time_b) - g_a(time_a)`` at one pair of times."""
        return self._band(
            [(str(treat_b), np.array([float(time_b)]), 1.0),
             (str(treat_a), np.array([float(time_a)]), -1.0)],
            level=level,
            labels=[f"{treat_b}@{time_b:g} vs {treat_a}@{time_a:g}"])

    def _band(self, parts, *, level: float | None = None,
              labels=None) -> TimeCourseNmaCurve:
        alpha = (level_to_alpha(level) if level is not None
                 else level_to_alpha(self.level))
        crit = float(stats.norm.ppf(1.0 - alpha / 2.0))
        size = parts[0][1].size
        est = np.zeros(size)
        jac = np.zeros((size, len(self.coef)))
        for treat, t, sign in parts:
            est += sign * self._value(treat, t)
            jac += sign * self._jacobian(treat, t)
        var = np.maximum(np.einsum("ij,jk,ik->i", jac, np.asarray(self.vcov),
                                   jac), 0.0)
        se = np.sqrt(var)
        extra = self._prediction_variance(jac)
        seen = self.times.get(parts[0][0]) or [0.0]
        lo, hi = min(seen), max(seen)
        return TimeCourseNmaCurve(
            labels=list(labels) if labels is not None else [parts[0][0]],
            time=parts[0][1].copy(),
            est=est,
            se=se,
            ci_lb=est - crit * se,
            ci_ub=est + crit * se,
            pred_lb=est - crit * np.sqrt(var + extra),
            pred_ub=est + crit * np.sqrt(var + extra),
            outside=(parts[0][1] < lo) | (parts[0][1] > hi),
        )

    def _prediction_variance(self, jac: np.ndarray) -> np.ndarray:
        out = np.zeros(jac.shape[0])
        if np.isfinite(self.tau2) and self.tau2 > 0.0:
            out += self.tau2
        if self.D is not None and np.size(self.D):
            out += np.einsum("ij,jk,ik->i", jac, np.asarray(self.D), jac)
        return out

    # -- tests ------------------------------------------------------------- #
    def _wald(self, columns: list[int]) -> dict:
        beta = np.asarray(self.coef, dtype=float)[columns]
        vcov = np.asarray(self.vcov, dtype=float)[np.ix_(columns, columns)]
        # a near-flat direction is not a failure of the test, it is a parameter
        # the data cannot pin down, and pinv reports exactly that
        stat = float(beta @ np.linalg.pinv(vcov, rcond=1e-10) @ beta)
        df = int(np.linalg.matrix_rank(vcov, tol=1e-10))
        return {
            "statistic": stat,
            "df": df,
            "pval": float(stats.chi2.sf(stat, df)) if df else float("nan"),
            "test": "chi2",
        }

    def test_trend(self) -> dict:
        """Wald test that no treatment's effect changes with time."""
        columns = [i for block in range(len(self.curves))
                   for i in self._time_columns(block)]
        if not columns:
            raise ValueError(
                f"the {self.basis!r} basis is constant in time, so there is no "
                "trend to test: that model is standard network meta-analysis")
        return self._wald(columns)

    def _time_columns(self, block: int) -> list[int]:
        p = _n_parameters(self.basis, n_knots=self.n_knots)
        if self.basis == "constant":
            return []
        return list(range(block * p + 1, (block + 1) * p))

    def test_nonlinear(self) -> dict:
        """Wald test that the curve is linear where the basis allows more."""
        if self.basis in ("constant", "linear"):
            raise ValueError(
                f"the {self.basis!r} basis has no curvature, so there is no "
                "non-linearity to test")
        p = _n_parameters(self.basis, n_knots=self.n_knots)
        columns = [block * p + 2 + j for block in range(len(self.curves))
                   for j in range(p - 2)]
        return self._wald(columns)

    def test_effect(self) -> dict:
        """Standard NMA: no treatment differs at time zero from the reference.

        The intercepts are the treatment effects at ``t = 0``, so this is the
        ordinary network meta-analysis effect test — which is why
        ``basis="constant"`` reduces the whole model to standard NMA.
        """
        columns = [block * _n_parameters(self.basis, n_knots=self.n_knots)
                   for block in range(len(self.curves))]
        return self._wald(columns)

    # -- reporting --------------------------------------------------------- #
    def coef_table(self) -> list[dict]:
        crit = self._crit()
        rows = []
        for name, est, se in zip(self.names, self.coef, self.se):
            z = est / se if se > 0 else float("nan")
            rows.append({
                "term": name,
                "estimate": float(est),
                "se": float(se),
                "zval": float(z),
                "pval": (float(2.0 * stats.norm.sf(abs(z))) if se > 0
                         else float("nan")),
                "ci_lb": float(est - crit * se),
                "ci_ub": float(est + crit * se),
            })
        return rows

    def to_dict(self) -> dict:
        return {
            "basis": self.basis,
            "treatments": list(self.treatments),
            "curves": list(self.curves),
            "reference": self.reference,
            "names": list(self.names),
            "coef": [float(v) for v in self.coef],
            "se": [float(v) for v in self.se],
            "tau2": None if not np.isfinite(self.tau2) else float(self.tau2),
            "D": None if self.D is None else np.asarray(self.D).tolist(),
            "k": self.k,
            "studies": list(self.studies),
            "times": {k: [float(v) for v in vs]
                      for k, vs in self.times.items()},
            "rho": self.rho,
            "phi": self.phi,
            "struct": self.struct,
            "coef_struct": self.coef_struct,
            "method": self.method,
            "level": self.level,
            "loglik": self.loglik,
            "n_knots": self.n_knots,
            "knots": None if self.knots is None else [float(v) for v in self.knots],
        }

    def __str__(self) -> str:
        parts = [
            f"Time-course network meta-analysis ({self.basis} basis over time)",
            f"  treatments: {', '.join(self.treatments)}",
            f"  reference: {self.reference}",
            f"  {self.k} contrasts from {len(self.studies)} studies",
            f"  heterogeneity: {self.struct}"
            + (f" ({self.coef_struct})" if self.D is not None else ""),
            (f"  tau2 = {self.tau2:.6f} "
             f"(tau = {np.sqrt(max(self.tau2, 0.0)):.6f})"
             if np.isfinite(self.tau2) else "  tau2 = 0 (fixed effect)"),
            "  within-study correlation: "
            + ("none" if self.rho is None and self.phi is None
               else (f"rho = {self.rho:g}" if self.rho is not None
                     else f"phi = {self.phi:g}")),
            f"  log-likelihood ({self.method}) = {self.loglik:.6f}",
        ]
        width = max([len(n) for n in self.names] + [6])
        for name, est, se in zip(self.names, self.coef, self.se):
            parts.append(f"  {name:<{width}}  {est:>10.6f}  ({se:.6f})")
        return "\n".join(parts)


def _within_study_covariance(vi, sid, n_study, times, *,
                             rho, phi) -> np.ndarray:
    """The within-study covariance of a study's repeated contrasts."""
    V = np.diag(vi)
    if rho is None and phi is None:
        return V
    for s in range(n_study):
        sel = np.flatnonzero(sid == s)
        if sel.size < 2:
            continue
        sd = np.sqrt(vi[sel])
        if phi is not None:
            gaps = np.abs(times[sel][:, None] - times[sel][None, :])
            corr = float(phi) ** gaps
        else:
            corr = np.full((sel.size, sel.size), float(rho))
        block = corr * np.outer(sd, sd)
        block[np.diag_indices_from(block)] = vi[sel]
        V[np.ix_(sel, sel)] = block
    return V


def nma_timecourse(yi, *, study, treat1, treat2, time, time2=None, vi=None,
                   vcov=None, rho: float | None = None,
                   phi: float | None = None, basis: str = "linear",
                   reference: str | None = None,
                   knots=None, n_knots: int | None = None,
                   struct: str = "study", coef_struct: str = "DIAG",
                   method: str = "REML", level: float = 0.95,
                   max_iter: int = 500, tol: float = 1e-9,
                   start=None) -> TimeCourseNmaResult:
    """Fit a time-course model to a network of contrasts.

    Args:
        yi: the observed contrast per row, on the linear-predictor scale.
        study: study label per row.
        treat1, treat2: the two treatments compared, ``treat2`` minus
            ``treat1`` in sign.
        time: the time of the ``treat2`` arm.
        time2: the time of the ``treat1`` arm; ``time`` for both when omitted,
            which is the usual case of two arms measured on the same schedule.
        vi: variance of ``yi`` per row; ignored when ``vcov`` is given.
        vcov: full covariance of ``yi``.  Rows from one study are correlated
            whenever they come from the same participants, which is the normal
            case for repeated measurements.
        rho: a constant within-study correlation between a study's contrasts.
        phi: an AR(1) correlation ``phi**|t_i - t_j|`` between them.  Give
            ``rho`` or ``phi`` or ``vcov``; all three at once is refused.
        basis: ``"constant"``, ``"linear"``, ``"quadratic"`` or ``"rcs"``; see
            the module docstring.
        reference: the treatment whose trajectory is identically zero.  Every
            comparison is identified only up to a common shift of all the
            trajectories, so one of them has to be fixed; left out, the
            treatment that appears in ``treat1`` most often is used, which is
            the control in a normally-oriented dataset.
        knots, n_knots: for ``basis="rcs"``.
        struct: ``"FE"``, ``"study"`` (default), ``"coef"`` or ``"coef+study"``.
        coef_struct: ``"DIAG"`` or ``"UN"`` for the trajectory's random effects.
        method: ``"REML"`` (default) or ``"ML"``.
        level: confidence level; ``0.95``, ``95`` and ``0.05`` are all read as
            the same 95%.
        max_iter: optimiser iteration cap.
        tol: convergence tolerance on the variance parameters.
        start: optional starting values for the coefficients.

    Returns:
        :class:`TimeCourseNmaResult`.

    Raises:
        ValueError: for mismatched inputs, an unknown basis, more than one
            correlation assumption, a study with fewer distinct times than the
            basis has parameters, or a covariance that is not positive
            definite.

    Example:
        >>> fit = ma.nma_timecourse(effect, study=study, treat1=arm1,
        ...                         treat2=arm2, time=month, vi=vi, phi=0.6)
        >>> print(fit.contrast("placebo", 0, "drug", 12))
    """
    name = str(basis).lower()
    if name not in _BASES:
        raise ValueError(
            f"unknown time basis {basis!r}; use one of "
            + ", ".join(repr(b) for b in _BASES))
    if struct not in ("FE", "study", "coef", "coef+study"):
        raise ValueError(
            f"unknown 'struct' {struct!r}; use 'FE', 'study', 'coef' or "
            "'coef+study'")
    if coef_struct not in ("DIAG", "UN"):
        raise ValueError(
            f"unknown 'coef_struct' {coef_struct!r}; use 'DIAG' or 'UN'")
    given = sum(x is not None for x in (vcov, rho, phi))
    if given > 1:
        raise ValueError(
            "give at most one of 'vcov', 'rho' and 'phi': they are three ways "
            "of saying the same thing and only one can be true")
    method_name = str(method).upper()
    if method_name not in ("REML", "ML"):
        raise ValueError(f"unknown 'method' {method!r}; use 'REML' or 'ML'")
    level = 100.0 * (1.0 - level_to_alpha(level))

    y = np.asarray(yi, dtype=float).ravel()
    k = y.size
    if k == 0:
        raise ValueError("no rows to fit")
    if not np.all(np.isfinite(y)):
        raise ValueError("'yi' contains missing or infinite values")
    studies = [str(s) for s in np.asarray(study).ravel()]
    a1 = [str(t) for t in np.asarray(treat1).ravel()]
    a2 = [str(t) for t in np.asarray(treat2).ravel()]
    t2v = np.asarray(time, dtype=float).ravel()
    t1v = np.asarray(time, dtype=float).ravel() if time2 is None \
        else np.asarray(time2, dtype=float).ravel()
    for label, values in (("study", studies), ("treat1", a1), ("treat2", a2),
                          ("time", t2v), ("time2", t1v)):
        if len(values) != k:
            raise ValueError(
                f"'{label}' has {len(values)} entries, expected {k}")

    studies_unique = list(dict.fromkeys(studies))
    index = {s: i for i, s in enumerate(studies_unique)}
    sid = np.array([index[s] for s in studies])

    times_all = np.concatenate([t1v, t2v])
    if name == "rcs":
        if knots is None:
            nk = 3 if n_knots is None else int(n_knots)
            if nk < 1:
                raise ValueError("'n_knots' must be at least 1")
            if np.unique(times_all).size < max(nk, 4):
                raise ValueError(
                    f"a restricted cubic spline with {nk} column(s) needs at "
                    f"least {max(nk, 4)} distinct times; found "
                    f"{np.unique(times_all).size}")
            knots = default_knots(times_all, nk + 1)
        knots = [float(v) for v in np.sort(np.asarray(knots, dtype=float).ravel())]
        p = 1 + len(knots) - 1
    else:
        p = _n_parameters(name)
        knots = None

    treatments = sorted(set(a1) | set(a2))
    if len(a1) != len(set(zip(studies, a1, t1v, a2, t2v, strict=True))):
        raise ValueError(
            "duplicate rows: a study-treatment-time cell appears more than once")
    if reference is None:
        counts: dict[str, int] = {}
        for name_ in a1:
            counts[name_] = counts.get(name_, 0) + 1
        reference = sorted(counts, key=lambda t: (-counts[t], t))[0]
    reference = str(reference)
    if reference not in treatments:
        raise ValueError(
            f"the reference {reference!r} is not a treatment in this network")
    # a common shift of every trajectory leaves every contrast unchanged, so one
    # trajectory is fixed at zero and the rest are measured from it
    curves = [t for t in treatments if t != reference]
    if not curves:
        raise ValueError(
            "every row compares the reference with itself; there is nothing to "
            "estimate")
    n_blocks = len(curves)
    n_par = n_blocks * p
    names: list[str] = []
    for treat in curves:
        names.extend(_names(name, n_knots=p - 1, label=treat))

    # --- within-study covariance ------------------------------------------ #
    if vcov is not None:
        V = np.asarray(vcov, dtype=float)
        if V.shape != (k, k):
            raise ValueError(f"'vcov' is {V.shape}, expected ({k}, {k})")
        V = 0.5 * (V + V.T)
    else:
        if vi is None:
            raise ValueError(
                "give 'vi' (with 'rho' or 'phi') or 'vcov'; the within-study "
                "correlation of repeated measurements cannot be estimated from "
                "the summaries")
        v = np.asarray(vi, dtype=float).ravel()
        if v.size != k:
            raise ValueError(f"'vi' has {v.size} entries, expected {k}")
        if np.any(~np.isfinite(v)) or np.any(v <= 0.0):
            raise ValueError("'vi' must be positive and finite")
        for corr in (rho, phi):
            if corr is not None and not 0.0 <= float(corr) < 1.0:
                raise ValueError("'rho' and 'phi' must be in [0, 1)")
        if rho is not None or phi is not None:
            # a study's repeated measurements are the same two arms measured at
            # two times, which is exactly what a time-decayed correlation
            # describes.  A study contributing several *treatment pairs* is a
            # different correlation problem: two arms measured at the same time
            # share the control, so their contrasts are correlated in a way that
            # depends on the arm sizes, and no single rho or phi can express it.
            pairs: dict[int, set] = {}
            for i in range(k):
                pairs.setdefault(int(sid[i]), set()).add((a1[i], a2[i]))
            multi = [studies_unique[s] for s, ps in pairs.items() if len(ps) > 1]
            if multi:
                raise ValueError(
                    f"{', '.join(sorted(multi))} compare more than one pair of "
                    "treatments, so no single 'rho' or 'phi' describes their "
                    "within-study covariance: contrasts of two arms at the same "
                    "time share the control and are correlated through it in a "
                    "way that depends on the arm sizes. Build the covariance "
                    "and pass it as 'vcov'.")
        V = _within_study_covariance(v, sid, len(studies_unique), t2v,
                                     rho=None if rho is None else float(rho),
                                     phi=None if phi is None else float(phi))
    try:
        np.linalg.cholesky(V)
    except np.linalg.LinAlgError as exc:
        raise ValueError(
            "the within-study covariance is not positive definite: with phi "
            "close to 1 a study's block stops being a correlation matrix") from exc

    # --- the design -------------------------------------------------------- #
    X = np.zeros((k, n_par))
    block = {t: i for i, t in enumerate(curves)}
    for i in range(k):
        for treat, when, sign in ((a2[i], t2v[i], 1.0), (a1[i], t1v[i], -1.0)):
            if treat == reference:
                continue
            lo = block[treat] * p
            X[i, lo:lo + p] += sign * _design(
                name, np.array([when]), knots=knots)[0]
    rank = int(np.linalg.matrix_rank(X))
    if rank < n_par:
        counts2: dict[str, set] = {}
        for i in range(k):
            for treat, when in ((a2[i], t2v[i]), (a1[i], t1v[i])):
                counts2.setdefault(treat, set()).add(round(float(when), 10))
        thin = [t for t in curves if len(counts2.get(t, ())) < p]
        raise ValueError(
            f"the time-course design has rank {rank} but {n_par} parameter(s)"
            + (f": {', '.join(thin)} has fewer distinct times than the {p} "
               f"parameter(s) the {name!r} basis needs" if thin else ""))

    Zs = np.zeros((k, len(studies_unique)))
    Zs[np.arange(k), sid] = 1.0
    share = Zs @ Zs.T
    use_study = struct in ("study", "coef+study")
    use_coef = struct in ("coef", "coef+study")
    if use_study and len(studies_unique) < 2:
        raise ValueError(
            "a between-study variance needs at least two studies; use "
            "struct='FE' for a single study")

    block_sizes = [(b * p, (b + 1) * p) for b in range(n_blocks)]
    n_theta = (1 if use_study else 0) + (
        n_blocks * (p if coef_struct == "DIAG" else p * (p + 1) // 2)
        if use_coef else 0)

    def unpack(theta: np.ndarray):
        pos = 0
        tau2 = float(np.exp(2.0 * theta[0])) if use_study else 0.0
        pos += 1 if use_study else 0
        D = None
        if use_coef:
            D = np.zeros((n_par, n_par))
            if coef_struct == "DIAG":
                for lo, hi in block_sizes:
                    D[lo:hi, lo:hi] = np.diag(np.exp(2.0 * theta[pos:pos + p]))
                    pos += p
            else:
                per = p * (p + 1) // 2
                for lo, hi in block_sizes:
                    D[lo:hi, lo:hi] = _log_cholesky(theta[pos:pos + per], p)
                    pos += per
        return tau2, D

    def sigma_of(theta: np.ndarray) -> np.ndarray:
        tau2, D = unpack(theta)
        S = V.copy()
        if use_study:
            S += tau2 * share
        if D is not None:
            S += X @ D @ X.T
        return S

    def evaluate(theta: np.ndarray):
        S = sigma_of(theta)
        chol, logdet = _chol(S)
        w = np.linalg.solve(chol, np.eye(k))
        omega = w.T @ w
        xtwx = X.T @ omega @ X
        beta = np.linalg.solve(xtwx, X.T @ omega @ y)
        resid = y - X @ beta
        value = logdet + _quadratic_form(chol, resid)
        if method_name == "REML":
            sign, ld = np.linalg.slogdet(xtwx)
            if sign <= 0:
                raise np.linalg.LinAlgError("the design is singular")
            value += ld
        return 0.5 * value, beta, xtwx

    def objective(theta: np.ndarray) -> float:
        try:
            return float(evaluate(theta)[0])
        except np.linalg.LinAlgError:
            return 1e20

    beta = (np.zeros(n_par) if start is None
            else np.asarray(start, dtype=float).ravel())
    if beta.size != n_par:
        raise ValueError(f"'start' has {beta.size} value(s), expected {n_par}")
    theta_hat = np.zeros(n_theta)
    if n_theta:
        theta0 = np.zeros(n_theta)
        if use_study:
            theta0[0] = -3.0
            fe = evaluate(theta0)[1]
            resid = y - X @ fe
            df = max(k - n_par, 1)
            theta0[0] = 0.5 * np.log(max(float(resid @ resid) / df, 1e-6))
        if use_coef:
            theta0[(1 if use_study else 0):] = -2.0
        bounds = [(-8.0, 8.0)] * n_theta
        if n_theta == 1:
            best = np.array([float(minimize_scalar(
                lambda t: objective(np.array([t])),
                bounds=(-8.0, 8.0), method="bounded",
                options={"xatol": max(tol, 1e-9)}).x)])
        else:
            fit = minimize(objective, theta0, method="L-BFGS-B",
                           bounds=bounds,
                           options={"maxiter": max_iter, "ftol": tol})
            best = np.asarray(fit.x, dtype=float)
        polished = minimize(objective, best, method="Nelder-Mead",
                            options={"maxiter": max_iter * 4, "xatol": 1e-9,
                                     "fatol": 1e-12})
        if polished.fun <= objective(best):
            best = np.asarray(polished.x, dtype=float)
        theta_hat = best
    value, beta, xtwx = evaluate(theta_hat)
    tau2, D = unpack(theta_hat)
    vcov = np.linalg.inv(xtwx)
    vcov = 0.5 * (vcov + vcov.T)
    se = np.sqrt(np.maximum(np.diag(vcov), 0.0))

    if use_study and (tau2 <= 1e-6 * float(np.mean(np.diag(V)))
                      or float(theta_hat[0]) <= -7.5):
        warnings.warn(
            f"the between-study variance is estimated at the boundary "
            f"(tau2 = {tau2:.3g}, against a mean sampling variance of "
            f"{float(np.mean(np.diag(V))):.3g}); the intervals are then the "
            "fixed-effect ones and overstate the precision",
            RuntimeWarning, stacklevel=2)
    if D is not None and np.min(np.diag(D)) < 1e-8:
        warnings.warn(
            "a between-study variance of the trajectory's coefficients is "
            "estimated at zero; the trajectory term adds nothing to the model",
            RuntimeWarning, stacklevel=2)

    span: dict[str, tuple[float, float]] = {}
    for treat in treatments:
        seen = [float(t) for i in range(k)
                for tr, t in ((a2[i], t2v[i]), (a1[i], t1v[i])) if tr == treat]
        span[treat] = (min(seen), max(seen)) if seen else (0.0, 0.0)

    return TimeCourseNmaResult(
        basis=name,
        treatments=treatments,
        curves=curves,
        reference=reference,
        names=names,
        coef=[float(v) for v in beta],
        se=[float(v) for v in se],
        vcov=vcov,
        tau2=tau2 if use_study else float("nan"),
        D=D,
        k=k,
        studies=studies_unique,
        times={k2: list(v) for k2, v in _observed_times(a1, a2, t1v, t2v,
                                                        treatments).items()},
        rho=None if rho is None else float(rho),
        phi=None if phi is None else float(phi),
        struct=struct,
        coef_struct=coef_struct,
        method=method_name,
        level=level,
        loglik=-float(value),
        n_knots=(p - 1) if name == "rcs" else None,
        knots=knots,
        fit={"theta": theta_hat.tolist(), "loglik": -float(value)},
    )


def _observed_times(a1, a2, t1v, t2v, treatments) -> dict:
    found: dict[str, set] = {}
    for i, (x, z) in enumerate(zip(a1, a2)):
        found.setdefault(x, set()).add(round(float(t1v[i]), 10))
        found.setdefault(z, set()).add(round(float(t2v[i]), 10))
    for treat in treatments:
        found.setdefault(treat, set())
    return {t: sorted(v) for t, v in found.items()}
