"""Dose-response meta-analysis: the within-study covariance, and pooling.

A dose-response study contributes several correlated contrasts against the same
reference group — log RR at 10 g/day, at 20, at 30 — so the effects are not
independent and an inverse-variance pool of them is wrong. Two things are
missing from a generic meta-regression, and both are here:

* **the covariance.** Two contrasts that share a control group share the
  sampling error of that control, so :func:`dose_cov` builds the within-study
  block. The Greenland–Longnecker correction re-derives the reference group's
  case count from the reported relative risks, because a published table
  rounds them and the naive count would not be consistent with them.
* **the design.** :func:`dosres` puts a spline over dose, assembles the
  block-diagonal covariance and fits the whole thing as a multilevel model with
  a random intercept per study.

The estimator is :func:`~pimeta.rma_mv.rma_mv`: a dose-response model is a
mixed model whose residual covariance is known up to the within-study blocks.
Nothing new is estimated, so everything downstream — the non-linearity test,
the predicted curve, the record — works as it does for any other fit.

    >>> fit = ma.dosres(yi, study=study, dose=dose, vi=vi, knots=[5, 15, 30])
    >>> print(fit.coef_table())
    >>> ma.predict_curve(fit, ma.rcs(dose, knots=[5, 15, 30], name="dose"))

Two ways to get the covariance:

``cases`` and ``n`` given
    Greenland–Longnecker, adjusted for the reference group.
``rho`` given
    A constant within-study correlation, ``rho * sqrt(vi_i * vi_j)``, for a
    paper that reports the contrasts but not the counts. If neither is given
    the off-diagonals are zero, which is what an ordinary meta-regression
    assumes — and is usually wrong for dose-response data.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
from scipy import stats

from .basis import Basis
from .checks import attach_diagnostics
from .pooling import predict_rma
from .utils import bldiag

__all__ = [
    "DosresResult",
    "dose_cov",
    "dosres",
    "greenland_counts",
]


# --------------------------------------------------------------------------- #
# the reference group, adjusted
# --------------------------------------------------------------------------- #
def _icc(counts: np.ndarray, totals: np.ndarray, kind: str) -> np.ndarray:
    """Variance of a log proportion (``cc``/``ci``) or a log rate (``ir``)."""
    if kind == "ir":
        return 1.0 / counts
    return 1.0 / counts + 1.0 / (totals - counts)


def greenland_counts(events, totals, effects, variances=None, *,
                     type: str = "cc", tol: float = 1e-12,
                     max_iter: int = 500) -> np.ndarray:
    """Greenland–Longnecker adjusted case counts for one study.

    A published table reports relative risks to two or three decimals, so the
    reported reference count and the reported contrasts are not exactly
    consistent. Greenland & Longnecker (1992) solve for the counts that *are*
    consistent by iterating

    .. math::

        A^{(m+1)} = A^{(m)} + H^{-1}\\big[\\, y + \\log(A_0)
                    + \\log(n_i - A_i) - \\log(A_i) - \\log(n_0 - A_0)\\big]

    where :math:`H` is the covariance matrix implied by those counts. The
    result is used to build the covariance and nothing else — the fitted
    effects stay the ones the paper reported.

    Args:
        events: cases (``cc``/``ci``) or events (``ir``) per dose level.
        totals: total participants (``cc``/``ci``) or person-time (``ir``).
        effects: log relative effect per dose level. The reference level(s)
            may be given as ``0``.
        variances: variance of each log effect. ``0``, ``None`` or ``NaN``
            marks the reference level, whose variance is taken to be zero.
        type: ``"cc"`` (case-control), ``"ci"`` (cumulative incidence) or
            ``"ir"`` (incidence rate).
        tol: convergence tolerance on the squared change in the adjusted
            counts. The default is tighter than ``dosresmeta``'s 1e-5, which
            leaves the counts good to about five decimals; the fixed point is
            the same.
        max_iter: iteration cap.

    Returns:
        Adjusted counts, one per dose level, in the input order.

    Raises:
        ValueError: for fewer than two dose levels, one reference level, or a
            negative/zero count.
    """
    kind = _as_type(type)
    events = np.asarray(events, dtype=float).ravel()
    totals = np.asarray(totals, dtype=float).ravel()
    effects = np.asarray(effects, dtype=float).ravel()
    if not (events.size == totals.size == effects.size):
        raise ValueError("'events', 'totals' and 'effects' must be the same length")
    if events.size < 2:
        raise ValueError(
            "a dose-response study needs at least two dose levels")
    if variances is None:
        ref = np.zeros(events.size, dtype=bool)
        ref[0] = True
    else:
        v = np.asarray(variances, dtype=float).ravel()
        v = np.where(np.isfinite(v), v, 0.0)
        ref = v <= 0.0
    n_ref = int(ref.sum())
    if n_ref != 1:
        raise ValueError(
            f"exactly one reference dose level is needed (one row with a zero "
            f"variance); found {n_ref}")
    if np.any(events <= 0.0) or np.any(totals <= 0.0):
        raise ValueError("counts and totals must be positive; add a correction upstream")

    idx = ~ref
    observed = events[idx]
    total_i = totals[idx]
    total_0 = float(totals[ref][0])
    log_effect = effects[idx]

    adjusted = events.copy()
    for _ in range(int(max_iter)):
        a0 = float(events.sum() - adjusted[idx].sum())
        c0 = float(_icc(np.array([a0]), np.array([total_0]), kind)[0])
        ci = _icc(adjusted[idx], total_i, kind)
        h = np.full((observed.size, observed.size), c0)
        h[np.diag_indices_from(h)] = ci + c0
        if kind == "ir":
            residual = (log_effect + np.log(a0) + np.log(total_i)
                        - np.log(adjusted[idx]) - np.log(total_0))
        else:
            residual = (log_effect + np.log(a0) + np.log(total_i - adjusted[idx])
                        - np.log(adjusted[idx]) - np.log(total_0 - a0))
        step = np.linalg.solve(h, residual)
        previous = adjusted[idx].copy()
        adjusted[idx] = previous + step
        adjusted[ref] = a0
        if float(np.sum((adjusted[idx] - previous) ** 2)) < tol:
            break
    return adjusted


def _as_type(type_: str) -> str:
    kind = str(type_).lower()
    if kind not in ("cc", "ci", "ir"):
        raise ValueError(
            f"unknown data type {type_!r}; use 'cc', 'ci' or 'ir' "
            "(case-control, cumulative incidence, incidence rate)")
    return kind


# --------------------------------------------------------------------------- #
# the within-study covariance
# --------------------------------------------------------------------------- #
def dose_cov(dose, events=None, totals=None, *, effects=None, variances=None,
             type: str = "cc", rho: float | None = None,
             tol: float = 1e-12, max_iter: int = 500) -> np.ndarray:
    """Within-study covariance of the log contrasts of one dose-response study.

    Args:
        dose: the dose levels, ascending or not; used for ordering only.
        events: cases/events per level; required for the Greenland correction.
        totals: participants or person-time per level.
        effects: log relative effect per level (needed by the correction).
        variances: variance per level; zero/``NaN`` marks the reference level.
        type: ``"cc"``, ``"ci"`` or ``"ir"``.
        rho: use a constant within-study correlation instead of the correction.
            The covariance is then ``rho * sqrt(vi_i * vi_j)`` and `variances`
            must be given.
        tol, max_iter: passed to :func:`greenland_counts`.

    Returns:
        ``(m, m)`` covariance matrix over the **non-reference** levels, in the
        input order.

    Raises:
        ValueError: when neither route is available, or the inputs disagree.

    Example:
        >>> cov = ma.dose_cov(dose, events, totals, effects=logrr, variances=v)
        >>> cov.shape
        (3, 3)
    """
    dose = np.asarray(dose, dtype=float).ravel()
    if variances is not None:
        v = np.asarray(variances, dtype=float).ravel()
        v = np.where(np.isfinite(v), v, 0.0)
        ref = v <= 0.0
    else:
        v = None
        ref = np.zeros(dose.size, dtype=bool)
        ref[0] = True

    kind = _as_type(type)
    if rho is not None:
        if v is None:
            raise ValueError("rho= needs 'variances'")
        idx = ~ref
        vi = v[idx]
        sd = np.sqrt(vi)
        corr = np.full((vi.size, vi.size), float(rho))
        np.fill_diagonal(corr, 1.0)
        return np.outer(sd, sd) * corr

    if events is None or totals is None or effects is None:
        raise ValueError(
            "give either 'rho=' (a constant within-study correlation), or "
            "'events', 'totals' and 'effects' for the Greenland-Longnecker "
            "correction")
    adjusted = greenland_counts(events, totals, effects, v, type=kind, tol=tol,
                                max_iter=max_iter)
    idx = ~ref
    totals = np.asarray(totals, dtype=float).ravel()
    c0 = float(_icc(adjusted[ref], totals[ref], kind)[0])
    ci = _icc(adjusted[idx], totals[idx], kind)
    cov = np.full((int(idx.sum()), int(idx.sum())), c0)
    cov[np.diag_indices_from(cov)] = ci + c0
    return cov


# --------------------------------------------------------------------------- #
# the model
# --------------------------------------------------------------------------- #
@attach_diagnostics
@dataclass
class DosresResult:
    """A pooled dose-response curve.

    Attributes:
        b: fixed-effect coefficients of the dose basis (intercept first).
        se, zval, pval, ci_lb, ci_ub: one entry per coefficient.
        vb: their covariance matrix.
        coefnames: names, e.g. ``["intrcpt", "dose[1]", "dose[2]"]``.
        tau2: between-study variance of the random intercept.
        QE, QE_df, QE_p: residual heterogeneity.
        k: number of contrasts fitted (reference levels excluded).
        studies: number of studies.
        method: τ² estimator.
        level: confidence level.
        basis: the :class:`~pimeta.basis.Basis` that was fitted.
        covariance: ``"greenland"``, ``"constant"`` or ``"independent"``.
        test: reference distribution of the coefficient tests.
        fit: the underlying :class:`~pimeta.rma_mv.RmaMvResult`.
        dose: the doses that were modelled, and ``study`` their study labels.
    """

    b: list[float]
    se: list[float]
    zval: list[float]
    pval: list[float]
    ci_lb: list[float]
    ci_ub: list[float]
    vb: np.ndarray = field(repr=False)
    coefnames: list[str]
    tau2: float
    QE: float
    QE_df: int
    QE_p: float
    k: int
    studies: int
    method: str
    level: float
    basis: Basis | None = field(repr=False)
    covariance: str
    test: str
    fit: Any = field(repr=False)
    dose: np.ndarray = field(repr=False, default_factory=lambda: np.empty(0))

    def to_dict(self) -> dict:
        return {
            "b": list(self.b),
            "se": list(self.se),
            "zval": list(self.zval),
            "pval": list(self.pval),
            "ci_lb": list(self.ci_lb),
            "ci_ub": list(self.ci_ub),
            "coefnames": list(self.coefnames),
            "tau2": self.tau2,
            "QE": self.QE,
            "QE_df": self.QE_df,
            "QE_p": self.QE_p,
            "k": self.k,
            "studies": self.studies,
            "method": self.method,
            "level": self.level,
            "covariance": self.covariance,
        }

    # -- the interface predict_curve() expects ----------------------------- #
    def predict(self, level: float | None = None, newmods=None, **kw):
        """Predicted log effect at new moderator rows, with CI and PI."""
        if newmods is None:
            new = np.zeros((1, len(self.b)))
            new[0, 0] = 1.0
        else:
            new = np.asarray(newmods, dtype=float)
            if new.ndim == 1:
                new = new.reshape(1, -1)
            new = np.column_stack([np.ones(new.shape[0]), new])
        if new.shape[1] != len(self.b):
            raise ValueError(
                f"'newmods' has {new.shape[1] - 1} column(s); the model has "
                f"{len(self.b) - 1}")
        alpha = (1.0 - self.level) if level is None else float(level)
        est = new @ np.asarray(self.b, dtype=float)
        var = np.einsum("ij,jk,ik->i", new, np.asarray(self.vb), new)
        out = []
        for i in range(new.shape[0]):
            out.append(predict_rma(
                est=float(est[i]),
                se=float(np.sqrt(max(var[i], 0.0))),
                tau2=self.tau2,
                k=self.k,
                p=len(self.b),
                parms=len(self.b) + 1,
                level=alpha,
                test=self.test,
            ))
        return out[0] if len(out) == 1 else out

    def coef_table(self) -> list[dict]:
        return [
            {
                "term": name,
                "estimate": float(b),
                "se": float(se),
                "zval": float(z),
                "pval": float(p),
                "ci_lb": float(lo),
                "ci_ub": float(hi),
            }
            for name, b, se, z, p, lo, hi in zip(
                self.coefnames, self.b, self.se, self.zval, self.pval,
                self.ci_lb, self.ci_ub)
        ]


def dosres(yi, *, study, dose, vi=None, cases=None, n=None, type: str = "cc",
           basis: Basis | None = None, knots=None, nk: int | None = 3,
           df: int | None = None, rho: float | None = None,
           method: str = "REML", level: float = 0.95,
           tol: float = 1e-12) -> DosresResult:
    """Pool a dose-response curve across studies.

    Two-stage in spirit and one-stage in form: each study contributes its own
    within-study covariance block, and the studies are pooled by a random
    intercept. That is the model the dose-response literature calls one-stage;
    the alternative — fit each study and pool the coefficients — is available by
    fitting the basis per study and handing the coefficients to ``rma_mv``.

    Args:
        yi: log relative effect per contrast.
        study: study identifier per row.
        dose: dose per row.
        vi: variance of ``yi``. Rows with zero or missing ``vi`` are the
            reference levels: they are used to build the covariance and then
            excluded from the fit, exactly as a published table's reference
            row is.
        cases: cases per dose level, for the Greenland correction.
        n: participants or person-time per dose level.
        type: ``"cc"``, ``"ci"`` or ``"ir"`` — see :func:`dose_cov`.
        basis: a :class:`~pimeta.basis.Basis` over dose. When omitted one is
            built by :func:`~pimeta.basis.rcs` from ``knots``/``nk``/``df``.
        knots, nk, df: passed to :func:`~pimeta.basis.rcs` when ``basis`` is not
            given.
        rho: constant within-study correlation, used instead of the Greenland
            correction.
        method: τ² estimator (``"REML"`` by default).
        level: confidence level.
        tol: Greenland convergence tolerance.

    Returns:
        :class:`DosresResult`.

    Raises:
        ValueError: when the rows do not line up, when a study has fewer than
            two levels, or when no covariance route is available.

    Example:
        >>> fit = ma.dosres(logrr, study=id, dose=dose, vi=se**2,
        ...                 cases=cases, n=n, knots=[5, 15, 30])
    """
    from .rma_mv import rma_mv

    yi = np.asarray(yi, dtype=float).ravel()
    dose = np.asarray(dose, dtype=float).ravel()
    labels = [str(s) for s in np.asarray(study).ravel()]
    if not (yi.size == dose.size == len(labels)):
        raise ValueError("'yi', 'study' and 'dose' must be the same length")
    v = (np.zeros(yi.size) if vi is None
         else np.where(np.isfinite(np.asarray(vi, dtype=float).ravel()),
                       np.asarray(vi, dtype=float).ravel(), 0.0))
    if vi is not None and v.size != yi.size:
        raise ValueError("'vi' must be the same length as 'yi'")

    groups = list(dict.fromkeys(labels))
    blocks: list[np.ndarray] = []
    row_order: list[np.ndarray] = []
    cov_kind = "independent"
    if rho is not None:
        cov_kind = "constant"
    elif cases is not None and n is not None:
        cov_kind = "greenland"
    cases_arr = None if cases is None else np.asarray(cases, dtype=float).ravel()
    n_arr = None if n is None else np.asarray(n, dtype=float).ravel()
    if cov_kind == "greenland" and (cases_arr is None or n_arr is None):
        raise ValueError("'cases' and 'n' must both be given")

    for g in groups:
        idx = np.array([i for i, lab in enumerate(labels) if lab == g])
        if idx.size < 2:
            raise ValueError(
                f"study {g!r} has a single row; a dose-response study needs at "
                "least two contrasts")
        if cov_kind == "independent":
            blocks.append(np.diag(v[idx]))
            row_order.append(idx)
            continue
        block = dose_cov(
            dose[idx],
            None if cases_arr is None else cases_arr[idx],
            None if n_arr is None else n_arr[idx],
            effects=yi[idx], variances=v[idx], type=type, rho=rho, tol=tol,
        )
        blocks.append(block)
        row_order.append(idx[v[idx] > 0.0])

    # The blocks and the rows have to be in the same order.  Building the row
    # order from the blocks (rather than a mask) makes interleaved input safe.
    order = np.concatenate(row_order)
    V = bldiag(*blocks)

    basis_obj = basis
    if basis_obj is None:
        basis_obj = _rcs_over(dose, knots=knots, nk=nk, df=df)
    expanded = basis_obj.values
    if expanded.shape[0] != yi.size:
        raise ValueError(
            "the basis was built from a different number of rows than 'yi'")
    mods = expanded[order]

    model = rma_mv(
        yi[order], V=V, mods=mods,
        data={"study": [labels[i] for i in order]},
        random="~ 1 | study",
        method=method,
    )

    names = ["intrcpt"] + list(basis_obj.names)
    b = [float(x) for x in np.asarray(model.beta, dtype=float).ravel()]
    se = np.sqrt(np.diag(np.asarray(model.vb)))
    test = str(getattr(model, "test", "z"))
    if test in ("t", "knha", "adhoc") and getattr(model, "ddf", None):
        crit = float(stats.t.ppf(
            0.5 + level / 2.0, float(np.asarray(model.ddf, dtype=float).ravel()[0])))
    else:
        crit = float(stats.norm.ppf(0.5 + level / 2.0))
    z = b / se
    return DosresResult(
        b=b,
        se=[float(x) for x in se],
        zval=[float(x) for x in z],
        pval=[float(2.0 * stats.norm.sf(abs(x))) for x in z],
        ci_lb=[float(bi - crit * sei) for bi, sei in zip(b, se)],
        ci_ub=[float(bi + crit * se_i) for bi, se_i in zip(b, se)],
        vb=np.asarray(model.vb, dtype=float),
        coefnames=names,
        tau2=float(np.asarray(model.sigma2).ravel()[0]),
        QE=float(model.qe),
        QE_df=int(model.qe_df),
        QE_p=float(model.qe_p),
        k=int(model.k),
        studies=len(groups),
        method=str(model.method),
        level=float(level),
        basis=basis_obj,
        covariance=cov_kind,
        test=test,
        fit=model,
        dose=dose[order],
    )


def _rcs_over(dose, *, knots, nk, df) -> Basis:
    """A spline over ``dose`` named ``dose``; knots from the *observed* doses."""
    from .basis import rcs

    positive = np.asarray(dose, dtype=float).ravel()
    if knots is None:
        positive = positive[positive > 0.0]
        if positive.size < 4:
            raise ValueError(
                "too few distinct doses to place knots; pass 'knots=' or "
                "'basis='")
    return rcs(np.asarray(dose, dtype=float).ravel(), nk=nk, df=df, knots=knots,
               name="dose")
