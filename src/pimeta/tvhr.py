"""Time-varying hazard ratios from piecewise-exponential models.

When proportional hazards fails, a single pooled hazard ratio is the wrong
summary: it averages over a hazard ratio that changes.  The usual answer is a
piecewise-exponential model — constant hazard within each interval of
follow-up, free to differ between intervals — which turns the problem into a
Poisson regression per study:

.. math::

    d_{ik} \\sim \\text{Poisson}(\\lambda_{ik} e_{ik}), \\qquad
    \\log\\lambda_{ik} = \\alpha_{s(i),k} + \\beta_k\\,\\text{treat}_i

where :math:`e_{ik}` is the exposure of participant :math:`i` in interval
:math:`k`.  The study and interval enter as a stratifying factor, so each
study estimates its own baseline hazard; :math:`\\beta_k` is that study's
log hazard ratio in interval :math:`k`.

The per-study estimates are then pooled interval by interval with the ordinary
random-effects engine, which is what "pooled hazard ratio at 0-6 months,
6-12 months, ..." means.

Two limitations are deliberate and worth stating:

- **Pooling each interval separately ignores the correlation between a study's
  interval-specific estimates** (they come from the same participants).  The
  intervals' pooled estimates are each valid; *comparing* them — "the hazard
  ratio falls after six months" — needs a model that carries that correlation,
  which this is not.  No trend test is reported for that reason.
- **The cut points are not chosen by the data.**  Picking where to break
  follow-up is a model-selection problem with its own multiplicity, exactly as
  choosing the number of spline knots is; the break points are the analyst's.

Inputs are participant-level: ``time``, ``event``, ``study`` and ``treat``.
:func:`pimeta.surv.km_reconstruct` turns a published (time, at-risk, events)
table into exactly this, which is how most "IPD" is actually obtained.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

__all__ = ["PiecewiseResult", "piecewise_exp", "poisson_irls", "split_followup"]


def split_followup(time, event, breaks):
    """Split each participant's follow-up at the interval boundaries.

    Args:
        time: follow-up time per participant.
        event: 1 if the event was observed, 0 if censored.
        breaks: interval boundaries, starting at 0.  Follow-up is split into
            ``len(breaks) - 1`` intervals; the last one is open-ended.

    Returns:
        ``(rows, intervals, exposure, events)`` — ``rows`` indexes the input
        participant, ``intervals`` is 0-based, ``exposure`` the time spent in
        that interval and ``events`` whether the event fell in it.

    Raises:
        ValueError: the breaks are not strictly increasing or do not start
            at zero, or a time is negative.
    """
    time = np.asarray(time, dtype=float).ravel()
    event = np.asarray(event).ravel()
    breaks = np.asarray(breaks, dtype=float).ravel()
    if breaks.size < 2:
        raise ValueError("'breaks' needs at least two boundaries")
    if breaks[0] != 0.0:
        raise ValueError("'breaks' must start at 0: follow-up starts there")
    if np.any(np.diff(breaks) <= 0):
        raise ValueError("'breaks' must be strictly increasing")
    if np.any(time < 0):
        raise ValueError("follow-up times must be non-negative")
    k_int = breaks.size - 1
    rows: list[int] = []
    ints: list[int] = []
    expos: list[float] = []
    evts: list[int] = []
    for i in range(time.size):
        t_end = float(time[i])
        hit = int(event[i]) == 1
        for j in range(k_int):
            lo, hi = breaks[j], breaks[j + 1]
            if t_end <= lo:
                break
            rows.append(i)
            ints.append(j)
            expos.append(min(t_end, hi) - lo)
            evts.append(1 if (hit and t_end <= hi) else 0)
    return (np.array(rows, dtype=int), np.array(ints, dtype=int),
            np.array(expos, dtype=float), np.array(evts, dtype=int))


def poisson_irls(y, X, offset=None, max_iter=200, tol=1e-10):
    """Poisson regression by iteratively reweighted least squares.

    Args:
        y: counts.
        X: design matrix (``n x p``).
        offset: optional additive term on the log scale, e.g.
            ``log(exposure)``.
        max_iter: iteration cap.
        tol: convergence threshold on the largest coefficient change.

    Returns:
        ``(beta, vcov, iterations, converged)``.

    Raises:
        ValueError: mismatched shapes, negative counts, a rank-deficient
            design, or a non-finite fit.
    """
    y = np.asarray(y, dtype=float).ravel()
    X = np.asarray(X, dtype=float)
    if X.ndim != 2:
        raise ValueError("'X' must be a two-dimensional design matrix")
    n, p = X.shape
    if y.size != n:
        raise ValueError("'y' and 'X' must have the same number of rows")
    if np.any(y < 0):
        raise ValueError("Poisson counts must be non-negative")
    off = np.zeros(n) if offset is None else np.asarray(offset, dtype=float).ravel()
    if off.size != n:
        raise ValueError("'offset' must have one entry per row")
    if np.linalg.matrix_rank(X) < p:
        raise ValueError("the design matrix is rank deficient")

    beta = np.zeros(p)
    # start from a log-scale average so the first step is not wildly off
    mean_rate = max(float(np.sum(y)) / max(float(np.sum(np.exp(off))), 1e-12),
                    1e-8)
    beta[0] = math.log(mean_rate)
    converged = False
    iterations = 0
    for count in range(int(max_iter)):
        iterations = count + 1
        eta = X @ beta + off
        eta = np.clip(eta, -700.0, 700.0)
        mu = np.exp(eta)
        if not np.all(np.isfinite(mu)):
            raise ValueError("the fit diverged; the model may be saturated")
        w = np.maximum(mu, 1e-12)
        z = X @ beta + (y - mu) / w
        xtwx = X.T @ (X * w[:, None])
        try:
            step = np.linalg.solve(xtwx, X.T @ (w * z))
        except np.linalg.LinAlgError as exc:      # pragma: no cover
            raise ValueError("the Poisson fit is singular") from exc
        delta = float(np.max(np.abs(step - beta)))
        beta = step
        if delta < tol:
            converged = True
            break

    eta = np.clip(X @ beta + off, -700.0, 700.0)
    w = np.maximum(np.exp(eta), 1e-12)
    xtwx = X.T @ (X * w[:, None])
    try:
        vcov = np.linalg.inv(xtwx)
    except np.linalg.LinAlgError as exc:          # pragma: no cover
        raise ValueError("the Poisson fit is singular") from exc
    return beta, vcov, iterations, converged


@dataclass
class PiecewiseResult:
    """Interval-specific hazard ratios pooled across studies.

    Attributes:
        breaks: the interval boundaries that were used.
        labels: a readable label per interval, e.g. ``"0-6"``.
        beta: pooled log hazard ratio per interval.
        se: its standard error.
        zval, pval, ci_lb, ci_ub: inference on each interval.
        tau2: between-study variance per interval.
        QE, QEp: residual heterogeneity per interval.
        k: number of studies contributing to each interval.
        log_hr: ``(n_study, n_interval)`` per-study log hazard ratios, ``NaN``
            where a study could not be fitted.
        se_hr: the matching standard errors.
        studies: study labels, in the row order of ``log_hr``.
        fits: the pooling fits, one per interval.
        dropped: ``(study, reason)`` for every study that could not be used;
            nothing is dropped silently.
        method, level: settings.
    """

    breaks: np.ndarray = field(repr=False)
    labels: list[str]
    beta: np.ndarray = field(repr=False)
    se: np.ndarray = field(repr=False)
    zval: np.ndarray = field(repr=False)
    pval: np.ndarray = field(repr=False)
    ci_lb: np.ndarray = field(repr=False)
    ci_ub: np.ndarray = field(repr=False)
    tau2: np.ndarray = field(repr=False)
    QE: np.ndarray = field(repr=False)
    QEp: np.ndarray = field(repr=False)
    k: list[int]
    log_hr: np.ndarray = field(repr=False)
    se_hr: np.ndarray = field(repr=False)
    studies: list
    fits: list = field(repr=False, default_factory=list)
    dropped: list = field(repr=False, default_factory=list)
    method: str = "REML"
    level: float = 95.0

    def coef_table(self) -> list[dict]:
        """One row per interval, with the hazard ratio on its own scale."""
        rows = []
        for j, label in enumerate(self.labels):
            rows.append({
                "interval": label,
                "log_hr": float(self.beta[j]),
                "se": float(self.se[j]),
                "hr": float(np.exp(self.beta[j])),
                "hr_lb": float(np.exp(self.ci_lb[j])),
                "hr_ub": float(np.exp(self.ci_ub[j])),
                "zval": float(self.zval[j]),
                "pval": float(self.pval[j]),
                "tau2": float(self.tau2[j]),
                "k": int(self.k[j]),
            })
        return rows

    def to_dict(self) -> dict:
        return {
            "breaks": self.breaks.tolist(),
            "labels": list(self.labels),
            "beta": self.beta.tolist(),
            "se": self.se.tolist(),
            "zval": self.zval.tolist(),
            "pval": self.pval.tolist(),
            "ci_lb": self.ci_lb.tolist(),
            "ci_ub": self.ci_ub.tolist(),
            "tau2": self.tau2.tolist(),
            "QE": self.QE.tolist(),
            "QEp": self.QEp.tolist(),
            "k": list(self.k),
            "log_hr": self.log_hr.tolist(),
            "se_hr": self.se_hr.tolist(),
            "studies": list(self.studies),
            "dropped": [[s, why] for s, why in self.dropped],
            "method": self.method,
            "level": self.level,
        }

    def __str__(self) -> str:
        rows = ["time-varying hazard ratio (piecewise exponential)"]
        rows.append(f"  {'interval':<14}{'HR':>8}{'95% CI':>20}{'k':>5}"
                    f"{'tau2':>9}")
        for row in self.coef_table():
            ci = f"[{row['hr_lb']:.3f}, {row['hr_ub']:.3f}]"
            rows.append(f"  {row['interval']:<14}{row['hr']:>8.3f}{ci:>20}"
                        f"{row['k']:>5}{row['tau2']:>9.4f}")
        return "\n".join(rows)


def _interval_labels(breaks):
    labels = []
    for j in range(breaks.size - 1):
        lo, hi = breaks[j], breaks[j + 1]
        labels.append(f"{lo:g}-{hi:g}")
    return labels


def piecewise_exp(time, event, *, study, treat, breaks, method="REML",
                  level=95.0, min_events=1):
    """Pooled interval-specific hazard ratios from participant-level data.

    Each study is fitted separately by Poisson regression of the split
    follow-up, with the interval as a stratifying factor and an
    interval-by-treatment interaction.  The interval-specific log hazard
    ratios are then pooled with a random-effects model per interval.

    Args:
        time: follow-up time per participant.
        event: 1 if the event was observed, 0 if censored.
        study: study identifier per participant.
        treat: treatment indicator per participant (0/1).
        breaks: interval boundaries, starting at 0 and strictly increasing.
        method: between-study variance estimator for the pooling.
        level: confidence level in percent (``95``) or as a proportion
            (``0.95``).
        min_events: an interval with fewer events than this is not fitted for
            that study; it stays ``NaN`` instead of returning a coefficient at
            minus infinity.

    Returns:
        :class:`PiecewiseResult`.

    Raises:
        ValueError: mismatched lengths, a treatment indicator that is not
            0/1, or fewer than two studies.
    """
    from .rma_uni import rma

    time = np.asarray(time, dtype=float).ravel()
    event = np.asarray(event).ravel()
    study = np.asarray(study).ravel()
    treat = np.asarray(treat).ravel()
    breaks = np.asarray(breaks, dtype=float).ravel()
    n = time.size
    if not (event.size == study.size == treat.size == n):
        raise ValueError("'time', 'event', 'study' and 'treat' must be the "
                         "same length")
    uniq_treat = np.unique(treat)
    if not set(np.asarray(uniq_treat).tolist()) <= {0, 1}:
        raise ValueError("'treat' must be a 0/1 indicator")

    level = float(level)
    if level <= 1.0:
        level *= 100.0

    k_int = breaks.size - 1
    labels = _interval_labels(breaks)
    studies = list(np.unique(study))
    if len(studies) < 2:
        raise ValueError("at least two studies are needed to pool")

    log_hr = np.full((len(studies), k_int), np.nan)
    se_hr = np.full((len(studies), k_int), np.nan)
    dropped: list[tuple] = []

    for row, s in enumerate(studies):
        sel = study == s
        rows, ints, exposure, events = split_followup(
            time[sel], event[sel], breaks)
        if rows.size == 0:
            dropped.append((s, "no follow-up after time 0"))
            continue
        tr = treat[sel][rows]
        # a cell with fewer than 'min_events' events carries no information
        # about the treatment effect, and fitting it would return a
        # coefficient at minus infinity.  Those rows are removed, not fitted.
        per_interval = np.bincount(ints, weights=events, minlength=k_int)
        usable = [j for j in range(k_int)
                  if per_interval[j] >= min_events]
        if not usable:
            dropped.append((s, f"no interval has {min_events} event(s)"))
            continue
        keep = np.isin(ints, usable)
        rows, ints = rows[keep], ints[keep]
        exposure, events, tr = exposure[keep], events[keep], tr[keep]
        used = sorted(usable)
        slot = {j: pos for pos, j in enumerate(used)}
        codes = np.array([slot[j] for j in ints], dtype=int)
        n_used = len(used)
        # design: the used interval dummies, then their interaction with treat
        X = np.zeros((rows.size, 2 * n_used))
        X[np.arange(rows.size), codes] = 1.0
        X[np.arange(rows.size), n_used + codes] = tr
        try:
            beta, vcov, _iters, converged = poisson_irls(
                events, X, offset=np.log(exposure))
        except ValueError as exc:
            dropped.append((s, f"Poisson fit failed: {exc}"))
            continue
        if not converged:
            dropped.append((s, "Poisson fit did not converge"))
            continue
        for pos, j in enumerate(used):
            col = n_used + pos
            est = float(beta[col])
            var = float(vcov[col, col])
            if var <= 0.0 or not np.isfinite(var):
                continue
            log_hr[row, j] = est
            se_hr[row, j] = math.sqrt(var)

    beta = np.full(k_int, np.nan)
    se = np.full(k_int, np.nan)
    zval = np.full(k_int, np.nan)
    pval = np.full(k_int, np.nan)
    ci_lb = np.full(k_int, np.nan)
    ci_ub = np.full(k_int, np.nan)
    tau2 = np.full(k_int, np.nan)
    qe = np.full(k_int, np.nan)
    qep = np.full(k_int, np.nan)
    ks: list[int] = []
    fits: list = []
    for j in range(k_int):
        ok = np.isfinite(log_hr[:, j])
        n_ok = int(np.sum(ok))
        ks.append(n_ok)
        if n_ok < 2:
            fits.append(None)
            continue
        fit = rma(log_hr[ok, j].tolist(), sei=se_hr[ok, j].tolist(),
                  method=method, level=level / 100.0)
        fits.append(fit)
        beta[j] = float(fit.b[0])
        se[j] = float(fit.se[0])
        zval[j] = float(fit.zval[0])
        pval[j] = float(fit.pval[0])
        ci_lb[j] = float(fit.ci_lb[0])
        ci_ub[j] = float(fit.ci_ub[0])
        tau2[j] = float(fit.tau2)
        qe[j] = float(fit.QE)
        qep[j] = float(fit.QEp)

    return PiecewiseResult(
        breaks=breaks, labels=labels, beta=beta, se=se, zval=zval,
        pval=pval, ci_lb=ci_lb, ci_ub=ci_ub, tau2=tau2, QE=qe, QEp=qep,
        k=ks, log_hr=log_hr, se_hr=se_hr, studies=studies, fits=fits,
        dropped=dropped, method=method, level=level,
    )
