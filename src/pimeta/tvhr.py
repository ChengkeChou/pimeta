"""Time-varying hazard ratios from piecewise-exponential models.

When proportional hazards fails, a single pooled hazard ratio is the wrong
summary: it averages over a hazard ratio that changes.  The usual answer is a
piecewise-exponential model —constant hazard within each interval of
follow-up, free to differ between intervals —which turns the problem into a
Poisson regression per study:

.. math::

    d_{ik} \\sim \\text{Poisson}(\\lambda_{ik} e_{ik}), \\qquad
    \\log\\lambda_{ik} = \\alpha_{s(i),k} + \\beta_k\\,\\text{treat}_i

where :math:`e_{ik}` is the exposure of participant :math:`i` in interval
:math:`k`.  The study and interval enter as a stratifying factor, so each
study estimates its own baseline hazard; :math:`\\beta_k` is that study's
log hazard ratio in interval :math:`k`.

The per-study estimates are then pooled.  Three routes, and the difference
between them is what is being estimated:

- :func:`piecewise_exp` pools each interval with its own univariate model.
  Every reported hazard ratio is valid on its own, and any statement comparing
  two intervals is informal.
- :func:`piecewise_joint` stacks a study's interval-specific log hazard ratios
  into a vector, gives each study its own sampling covariance, puts a
  between-study covariance on top and fits them together — so a trend test and
  a joint region have the covariance they need.  Its weakness is that the
  between-study covariance has :math:`J(J+1)/2` parameters to estimate from a
  handful of studies.
- :func:`contrast_trend` reduces each study to **one** number first — the slope
  of its log hazard ratio against time — and pools those by an ordinary
  univariate random-effects model.  The study-level level is shared by all
  intervals, and a centred contrast annihilates it exactly, so the correlation
  that goes to :math:`\\pm 1` in the joint model cannot appear at all: one
  variance parameter remains, and Knapp–Hartung and Q-profile intervals, which
  need a scalar, become available.

On the within-study correlation, which is where this model is usually
misdescribed.  The intervals partition each participant's follow-up, so a
Poisson fit with a saturated interval-by-treatment interaction estimates each
interval's log hazard ratio from disjoint rows and the information matrix is
block diagonal: a study's interval-specific estimates are **exactly
uncorrelated**, which ``tests/test_tvhr_joint.py`` checks rather than assumes.
A within-study ``rho`` therefore has no role when the input is participant
level.  It does when the input is *aggregate*: published interval-specific
hazard ratios come from a model that does not have that structure, and the
correlation cannot be estimated from the data, so ``rho`` is an assumption to
be varied rather than a quantity to be reported.

Two limitations are deliberate and worth stating:

- **The between-study correlation is not always identifiable.**  With a few
  studies the estimate of :math:`\\Sigma` lands on a boundary where one
  correlation is :math:`\\pm 1`; the likelihood is then flat along a ridge and
  every statistic that contrasts intervals — the omnibus test especially —
  depends on where the optimiser stopped.  On the shipped 16-study fixture the
  omnibus statistic is 40.6 with ``struct="DIAG"``, 52.1 with ``"UN"`` and 57.3
  with ``"HCS"``: the qualitative answer survives, the number does not, and a
  fit with a boundary correlation warns.  :func:`contrast_trend` avoids the
  problem instead of reporting around it: it never forms the covariance.
- **The cut points are not chosen by the data.**  Picking where to break
  follow-up is a model-selection problem with its own multiplicity, exactly as
  choosing the number of spline knots is; the break points are the analyst's.

Inputs are participant-level: ``time``, ``event``, ``study`` and ``treat``.
:func:`pimeta.surv.km_reconstruct` turns a published (time, at-risk, events)
table into exactly this, which is how most "IPD" is actually obtained.
"""

from __future__ import annotations

import math
import warnings
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, cast

import numpy as np
from scipy import stats

if TYPE_CHECKING:  # pragma: no cover - import cycle at runtime
    from .rma_mv import RmaMvResult

__all__ = ["PiecewiseJointResult", "PiecewiseResult", "contrast_trend",
           "piecewise_exp", "piecewise_joint", "poisson_irls",
           "split_followup"]


def split_followup(time, event, breaks):
    """Split each participant's follow-up at the interval boundaries.

    Args:
        time: follow-up time per participant.
        event: 1 if the event was observed, 0 if censored.
        breaks: interval boundaries, starting at 0.  Follow-up is split into
            ``len(breaks) - 1`` intervals; the last one is open-ended.

    Returns:
        ``(rows, intervals, exposure, events)`` —``rows`` indexes the input
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
        cov_hr: ``(n_study, n_interval, n_interval)`` the covariance of each
            study's log hazard ratios.  Exactly diagonal, because the intervals
            partition the follow-up and a Poisson fit with a saturated
            interval-by-treatment interaction therefore has a block-diagonal
            information matrix; :func:`piecewise_joint` checks that rather than
            trusting it.
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
    cov_hr: np.ndarray | None = field(repr=False, default=None)
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


def _fit_study_blocks(time, event, study, treat, breaks, min_events):
    """One Poisson fit per study; return the estimates *and* their covariance.

    The intervals partition each participant's follow-up, so a study's
    interval-specific log hazard ratios are estimated from disjoint rows of the
    design and their covariance block is exactly diagonal.  That is checked
    numerically in the tests rather than assumed, and it is the reason a
    within-study ``rho`` only matters for aggregate data, where the estimates
    come from a model that does not have this structure.
    """
    k_int = breaks.size - 1
    studies = list(np.unique(study))
    log_hr = np.full((len(studies), k_int), np.nan)
    se_hr = np.full((len(studies), k_int), np.nan)
    cov_hr = np.full((len(studies), k_int, k_int), np.nan)
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
            for pos2, j2 in enumerate(used):
                cov_hr[row, j, j2] = float(vcov[col, n_used + pos2])
    return studies, log_hr, se_hr, cov_hr, dropped


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
    if len(np.unique(study)) < 2:
        raise ValueError("at least two studies are needed to pool")
    studies, log_hr, se_hr, cov_hr, dropped = _fit_study_blocks(
        time, event, study, treat, breaks, min_events)

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
        k=ks, log_hr=log_hr, se_hr=se_hr, studies=studies, cov_hr=cov_hr,
        fits=fits, dropped=dropped, method=method, level=level,
    )


# --------------------------------------------------------------------------- #
# joint pooling: the interval-specific log hazard ratios as one vector
# --------------------------------------------------------------------------- #
def _wald_test(weights, beta, vcov):
    """A one-degree-of-freedom Wald test of ``weights @ beta == 0``."""
    w = np.asarray(weights, dtype=float).ravel()
    beta = np.asarray(beta, dtype=float).ravel()
    vcov = np.asarray(vcov, dtype=float)
    if w.size != beta.size:
        raise ValueError("the contrast must cover every interval")
    ok = np.isfinite(beta)
    wk = w[ok]
    var = float(wk @ vcov[np.ix_(ok, ok)] @ wk)
    if var <= 0.0:
        return {"stat": float("nan"), "df": 1, "pval": float("nan")}
    est = float(wk @ beta[ok])
    stat = est ** 2 / var
    return {"stat": float(stat), "df": 1,
            "pval": float(stats.chi2.sf(stat, 1)),
            "estimate": est, "se": math.sqrt(var)}


def _trend_stat(beta, vcov, kind):
    """The trend statistics from an estimate vector and its covariance.

    Separated from :meth:`PiecewiseJointResult.trend_test` so the same test can
    be recomputed from a *refit* under a different between-study structure
    without building a fake result object to hold it.
    """
    beta = np.asarray(beta, dtype=float).ravel()
    vcov = np.asarray(vcov, dtype=float)
    label = str(kind).lower()
    ok = np.isfinite(beta)
    n = int(np.sum(ok))
    if n < 2:
        raise ValueError("a trend needs at least two estimable intervals")
    if label == "linear":
        pos = np.arange(beta.size, dtype=float)
        return _wald_test(pos - pos[ok].mean(), beta, vcov)
    if label == "diff":
        w = np.zeros(beta.size)
        idx = np.flatnonzero(ok)
        w[idx[0]] = -1.0
        w[idx[-1]] = 1.0
        return _wald_test(w, beta, vcov)
    if label == "any":
        b = beta[ok]
        C = np.zeros((n - 1, n))
        for i in range(1, n):
            C[i - 1, i] = 1.0
            C[i - 1, 0] = -1.0
        diff = C @ b
        middle = C @ vcov[np.ix_(ok, ok)] @ C.T
        stat = float(diff @ np.linalg.solve(middle, diff))
        return {"stat": stat, "df": n - 1,
                "pval": float(stats.chi2.sf(stat, n - 1))}
    raise ValueError(
        f"unknown trend kind {kind!r}; use 'linear', 'diff' or 'any'")


@dataclass
class PiecewiseJointResult:
    """Interval-specific hazard ratios pooled as a vector.

    Pooling each interval with its own univariate model gives valid
    interval-specific estimates but makes every *comparison* between intervals
    an informal one: the estimates share the between-study covariance, so
    "the hazard ratio falls after six months" needs the joint covariance.

    Attributes:
        breaks, labels: the intervals.
        beta, se, ci_lb, ci_ub: the pooled log hazard ratio per interval.
        vcov: their joint covariance matrix.
        Sigma: the estimated between-study covariance of the interval-specific
            log hazard ratios, on the log scale.
        tau2: its diagonal.
        k: studies contributing per interval.
        rho: the within-study correlation that was imposed on the sampling
            covariance, or ``None`` when the per-study fits supplied it.
        model: the underlying :class:`~pimeta.RmaMvResult`, so
            :func:`~pimeta.predict_region` can be applied to it.
        studies, dropped: as in :class:`PiecewiseResult`.
        method, level, struct: settings.
    """

    breaks: np.ndarray = field(repr=False)
    labels: list[str]
    beta: np.ndarray = field(repr=False)
    se: np.ndarray = field(repr=False)
    ci_lb: np.ndarray = field(repr=False)
    ci_ub: np.ndarray = field(repr=False)
    vcov: np.ndarray = field(repr=False)
    Sigma: np.ndarray = field(repr=False)
    tau2: np.ndarray = field(repr=False)
    k: list[int]
    studies: list
    model: object = field(repr=False, default=None)
    dropped: list = field(repr=False, default_factory=list)
    rho: float | None = None
    method: str = "REML"
    level: float = 95.0
    struct: str = "UN"

    def coef_table(self) -> list[dict]:
        """One row per interval, with the hazard ratio on its own scale."""
        rows = []
        for j, label in enumerate(self.labels):
            if not np.isfinite(self.beta[j]):
                rows.append({"interval": label, "hr": float("nan"),
                             "hr_lb": float("nan"), "hr_ub": float("nan"),
                             "log_hr": float("nan"), "se": float("nan"),
                             "k": int(self.k[j]), "tau2": float("nan")})
                continue
            rows.append({
                "interval": label,
                "log_hr": float(self.beta[j]),
                "se": float(self.se[j]),
                "hr": float(np.exp(self.beta[j])),
                "hr_lb": float(np.exp(self.ci_lb[j])),
                "hr_ub": float(np.exp(self.ci_ub[j])),
                "k": int(self.k[j]),
                "tau2": float(self.tau2[j]),
            })
        return rows

    def _wald(self, weights):
        """A one-degree-of-freedom Wald test of ``weights @ beta == 0``."""
        res = _wald_test(weights, self.beta, self.vcov)
        if "estimate" in res:
            crit = self._crit()
            res["ci_lb"] = res["estimate"] - crit * res["se"]
            res["ci_ub"] = res["estimate"] + crit * res["se"]
        return res

    def _crit(self) -> float:
        return float(stats.norm.ppf(0.5 + self.level / 200.0))

    def structure_sensitivity(self, structs: tuple[str, ...] = ("UN", "HCS",
                                                              "DIAG"),
                              level: float | None = None) -> dict:
        """Refit the stacked model under other between-study structures.

        ``struct="UN"`` estimates one variance and one correlation per interval
        pair, and with three intervals on a handful of studies the correlation
        estimate can reach +/-1.  When it does, the between-study covariance is
        singular and any joint test that contrasts the intervals is decided by
        where on that ridge the optimiser stopped -- not by the data.  The
        qualitative answer survives that; the *statistic* does not, and a
        single number reported without the spread is a false precision.

        This refits the same stacked data and design under each structure and
        recomputes the trend statistics, so the range is a computation rather
        than a sentence.  ``"UN"`` makes the fewest assumptions and is the
        least stable; ``"HCS"`` assumes one common correlation and ``"DIAG"``
        assumes none, which is the most restrictive and the most stable.  A
        conclusion that holds under all three is a conclusion about the data;
        one that holds only under ``"UN"`` is a conclusion about the optimiser.

        Args:
            structs: the structures to refit.  ``"CS"`` is accepted too, as
                are any others :func:`~pimeta.rma_mv` supports.
            level: confidence level for the recomputed intervals; defaults to
                the level the result was fitted at.

        Returns:
            A dict with ``by_struct`` -- one entry per structure holding
            ``tests`` (the same keys :meth:`trend_test` returns, for
            ``"any"``, ``"linear"`` and ``"diff"``), ``tau2``, ``boundary``
            (whether that fit landed on the correlation ridge) and ``error``
            if the refit failed -- and ``spread``, which gives the smallest and
            largest statistic each trend test takes across the structures that
            converged.

        Raises:
            ValueError: the result was built without a fitted model, or the
                covariance is not positive definite under every structure.
        """
        from .rma_mv import rma_mv

        if self.model is None:
            raise ValueError("this result carries no fitted model to refit")
        model = cast("RmaMvResult", self.model)
        lvl = self.level if level is None else float(level)
        if lvl <= 1.0:
            lvl *= 100.0
        by_struct: dict[str, dict] = {}
        for struct in structs:
            try:
                fit = rma_mv(np.asarray(model.yi, dtype=float), V=model.V,
                             mods=model.X, data=model._data,
                             random=model.random, struct=str(struct),
                             intercept=model.intercept_incl,
                             method=model.method, level=lvl / 100.0)
            except (ValueError, np.linalg.LinAlgError) as exc:
                by_struct[str(struct)] = {"error": str(exc)}
                continue
            tests: dict[str, dict] = {}
            for kind in ("any", "linear", "diff"):
                try:
                    tests[kind] = _trend_stat(fit.beta, fit.vb, kind)
                except ValueError:
                    continue
            G, _ = fit.g_matrix()
            sigma = (np.asarray(G, dtype=float) if G is not None
                     else np.zeros((0, 0)))
            boundary = False
            if sigma.size and sigma.shape[0] > 1:
                sd = np.sqrt(np.maximum(np.diag(sigma), 0.0))
                with np.errstate(divide="ignore", invalid="ignore"):
                    corr = sigma / np.outer(sd, sd)
                off = corr[~np.eye(sigma.shape[0], dtype=bool)]
                boundary = bool(np.any(np.abs(off[np.isfinite(off)]) >= 0.999))
            by_struct[str(struct)] = {
                "tests": tests,
                "tau2": np.diag(sigma).copy() if sigma.size else np.zeros(0),
                "boundary": boundary,
            }
        spread: dict[str, dict] = {}
        for kind in ("any", "linear", "diff"):
            vals = [r["tests"][kind]["stat"] for r in by_struct.values()
                    if "tests" in r and kind in r["tests"]
                    and np.isfinite(r["tests"][kind]["stat"])]
            if vals:
                spread[kind] = {"min": float(min(vals)),
                                "max": float(max(vals))}
        return {"by_struct": by_struct, "spread": spread, "level": lvl}

    def trend_test(self, kind: str = "linear") -> dict:
        """Is the hazard ratio changing over follow-up?

        Args:
            kind: ``"linear"`` weighs the intervals by their position, which is
                the usual test for a monotone trend; ``"diff"`` contrasts the
                last interval with the first, which is the question a reader
                actually asks; ``"any"`` tests every interval against the first
                jointly, which is the omnibus test that the hazard ratio is
                constant.

        Returns:
            A dict with ``stat``, ``df``, ``pval`` and, for the one-degree-of-
            freedom versions, ``estimate``, ``se``, ``ci_lb``, ``ci_ub`` on the
            log scale.
        """
        result = _trend_stat(self.beta, self.vcov, kind)
        if "estimate" in result:
            crit = self._crit()
            result["ci_lb"] = result["estimate"] - crit * result["se"]
            result["ci_ub"] = result["estimate"] + crit * result["se"]
        return result

    def to_dict(self) -> dict:
        return {
            "breaks": self.breaks.tolist(),
            "labels": list(self.labels),
            "beta": self.beta.tolist(),
            "se": self.se.tolist(),
            "ci_lb": self.ci_lb.tolist(),
            "ci_ub": self.ci_ub.tolist(),
            "vcov": self.vcov.tolist(),
            "Sigma": self.Sigma.tolist(),
            "tau2": self.tau2.tolist(),
            "k": list(self.k),
            "studies": list(self.studies),
            "rho": self.rho,
            "method": self.method,
            "level": self.level,
            "struct": self.struct,
            "dropped": [[s, why] for s, why in self.dropped],
        }

    def __str__(self) -> str:
        rows = ["time-varying hazard ratio, joint pooling"]
        rows.append(f"  {'interval':<14}{'HR':>8}{'95% CI':>20}{'k':>5}"
                    f"{'tau2':>9}")
        for row in self.coef_table():
            ci = f"[{row['hr_lb']:.3f}, {row['hr_ub']:.3f}]"
            rows.append(f"  {row['interval']:<14}{row['hr']:>8.3f}{ci:>20}"
                        f"{row['k']:>5}{row['tau2']:>9.4f}")
        for kind, name in (("any", "hazard ratio constant"),
                           ("linear", "linear trend"),
                           ("diff", "last vs first")):
            try:
                test = self.trend_test(kind)
            except ValueError:
                continue
            rows.append(f"  {name:<22} chi2({test['df']}) = "
                        f"{test['stat']:.3f}, p = {test['pval']:.4g}")
        return "\n".join(rows)


def piecewise_joint(time, event, *, study, treat, breaks, rho=None,
                    method="REML", level=95.0, min_events=1,
                    struct="UN"):
    """Pool the interval-specific hazard ratios as one vector.

    The difference from :func:`piecewise_exp` is what is being estimated.  That
    function pools each interval with its own univariate model: every reported
    hazard ratio is valid on its own, and any statement comparing two intervals
    is informal.  This one stacks a study's interval-specific log hazard ratios
    into a vector, gives each study its own sampling covariance, puts a
    between-study covariance on top and fits them together —so
    :meth:`PiecewiseJointResult.trend_test` and
    :func:`~pimeta.predict_region` have the covariance they need.

    Args:
        time, event, study, treat, breaks, min_events: as in
            :func:`piecewise_exp`.
        rho: within-study correlation between a study's interval-specific log
            hazard ratios.  ``None`` uses the covariance the per-study Poisson
            fits supply, which for this parameterisation is exactly diagonal —
            the intervals partition the follow-up, so the information matrix is
            block diagonal.  A non-zero value is for *aggregate* input, where
            the reported hazard ratios come from a model that does not have that
            structure and the correlation cannot be estimated from the data;
            it is an assumption, and it belongs in a sensitivity analysis.
        method: ``"REML"`` or ``"ML"``.
        level: confidence level.
        struct: between-study covariance structure passed to
            :func:`~pimeta.rma_mv`; ``"UN"`` needs ``k(k+1)/2`` parameters for
            ``k`` intervals.

    Returns:
        :class:`PiecewiseJointResult`.

    Raises:
        ValueError: on mismatched lengths, a treatment indicator that is not
            0/1, fewer than two studies, or a ``rho`` outside ``[-1, 1]``.
    """
    from .rma_mv import rma_mv

    time = np.asarray(time, dtype=float).ravel()
    event = np.asarray(event).ravel()
    study = np.asarray(study).ravel()
    treat = np.asarray(treat).ravel()
    breaks = np.asarray(breaks, dtype=float).ravel()
    n = time.size
    if not (event.size == study.size == treat.size == n):
        raise ValueError("'time', 'event', 'study' and 'treat' must be the "
                         "same length")
    if not set(np.unique(treat).tolist()) <= {0, 1}:
        raise ValueError("'treat' must be a 0/1 indicator")
    if rho is not None and not -1.0 <= float(rho) <= 1.0:
        raise ValueError("'rho' must lie in [-1, 1]")
    level = float(level)
    if level <= 1.0:
        level *= 100.0
    k_int = breaks.size - 1
    labels = _interval_labels(breaks)
    if len(np.unique(study)) < 2:
        raise ValueError("at least two studies are needed to pool")
    studies, log_hr, se_hr, cov_hr, dropped = _fit_study_blocks(
        time, event, study, treat, breaks, min_events)

    yi: list[float] = []
    blocks: list[np.ndarray] = []
    k_per = np.zeros(k_int, dtype=int)
    for i in range(len(studies)):
        ok = np.isfinite(log_hr[i])
        if not ok.any():
            continue
        idx = np.flatnonzero(ok)
        block = np.array(cov_hr[i])[np.ix_(idx, idx)]
        if rho is not None:
            sd = np.sqrt(np.diag(block))
            corr = np.full(block.shape, float(rho))
            np.fill_diagonal(corr, 1.0)
            block = corr * np.outer(sd, sd)
        try:
            np.linalg.cholesky(block)
        except np.linalg.LinAlgError as exc:
            raise ValueError(
                f"study {studies[i]!r}: the implied within-study covariance is "
                f"not positive definite; rho={rho} is too large for its "
                "standard errors") from exc
        for j in idx:
            yi.append(float(log_hr[i, j]))
            k_per[j] += 1
        blocks.append(block)

    # an interval no study could be fitted in has no column: a dummy of zeros
    # makes the design singular, and the honest answer for it is NaN, not a
    # coefficient at minus infinity.
    used = [j for j in range(k_int) if k_per[j] > 0]
    if not used:
        raise ValueError("no interval has enough events in any study")
    col_of = {j: pos for pos, j in enumerate(used)}
    total = sum(b.shape[0] for b in blocks)
    V = np.zeros((total, total))
    X = np.zeros((total, len(used)))
    labels_row: list[str] = []
    study_row: list[str] = []
    start = 0
    pos = 0
    for i in range(len(studies)):
        idx = np.flatnonzero(np.isfinite(log_hr[i]))
        if idx.size == 0:
            continue
        n_i = idx.size
        V[start:start + n_i, start:start + n_i] = blocks[pos]
        X[start + np.arange(n_i), [col_of[j] for j in idx]] = 1.0
        labels_row.extend(labels[j] for j in idx)
        study_row.extend([str(studies[i])] * n_i)
        start += n_i
        pos += 1

    data = {"interval": labels_row, "study": study_row}
    fit = rma_mv(np.array(yi, dtype=float), V=V, mods=X, data=data,
                 random="~ interval | study", struct=struct,
                 intercept=False, method=method, level=level / 100.0)
    sub_beta = np.asarray(fit.beta, dtype=float)
    sub_vcov = np.asarray(fit.vb, dtype=float)
    G, names = fit.g_matrix()
    sub_sigma = (np.asarray(G, dtype=float) if G is not None
                 else np.zeros((len(used), len(used))))
    # rma_mv orders the levels of the random term by their sort order, so the
    # intervals are not necessarily in the order they were asked for
    # ("0-6" < "12-24" < "6-12").  Put Sigma back in interval order.
    if names:
        order = [list(names).index(labels[j]) for j in used]
    else:
        order = list(range(len(used)))
    sub_sigma = sub_sigma[np.ix_(order, order)]
    beta = np.full(k_int, np.nan)
    vcov = np.full((k_int, k_int), np.nan)
    Sigma = np.zeros((k_int, k_int))
    for j in used:
        beta[j] = sub_beta[col_of[j]]
        for j2 in used:
            vcov[j, j2] = sub_vcov[col_of[j], col_of[j2]]
            Sigma[j, j2] = sub_sigma[col_of[j], col_of[j2]]
    sigma2 = np.diag(Sigma)
    if len(used) > 1:
        sd = np.sqrt(np.maximum(sigma2[used], 0.0))
        with np.errstate(divide="ignore", invalid="ignore"):
            corr = Sigma[np.ix_(used, used)] / np.outer(sd, sd)
        off = corr[~np.eye(len(used), dtype=bool)]
        if np.any(np.abs(off[np.isfinite(off)]) >= 0.999):
            warnings.warn(
                "the estimated between-interval correlation is at +/-1, so the "
                "between-study covariance is singular and any joint test that "
                "contrasts the intervals is decided by where on that ridge the "
                "optimiser stopped: on the shipped fixture struct='UN' gives "
                "chi2 = 52.1 for a constant hazard ratio where struct='HCS' "
                "gives 57.3 and struct='DIAG' gives 40.6.  The qualitative "
                "answer survives that spread; the statistic does not.  "
                "structure_sensitivity() recomputes the spread for this fit",
                RuntimeWarning, stacklevel=2)
    se = np.sqrt(np.diag(vcov))
    crit = float(stats.norm.ppf(0.5 + level / 200.0))
    return PiecewiseJointResult(
        breaks=breaks, labels=labels, beta=beta, se=se,
        ci_lb=beta - crit * se, ci_ub=beta + crit * se,
        vcov=vcov, Sigma=Sigma, tau2=np.diag(Sigma),
        k=k_per.tolist(), studies=studies, model=fit, dropped=dropped,
        rho=None if rho is None else float(rho), method=method, level=level,
        struct=struct,
    )


def contrast_trend(time, event, *, study, treat, breaks, times=None,
                   rho=None, method="REML", test="knha", level=95.0,
                   min_events=1):
    """The change in the hazard ratio over follow-up, as one slope per study.

    :func:`piecewise_joint` estimates a between-study covariance over the
    interval-specific log hazard ratios and contrasts it afterwards.  With a
    handful of studies that covariance can be singular — one correlation
    reaches :math:`\\pm 1`, the likelihood is flat along the ridge, and every
    statistic that contrasts intervals depends on where the optimiser stopped.
    This function never forms that matrix.

    Each study is reduced to **one** number inside the study before anything is
    pooled: the slope of its log hazard ratios against time, centred on that
    study's own information-weighted mean time :math:`\\bar t_i`,

    .. math::

        \\hat\\beta_{1i}
            = \\frac{\\mathbf{c}_i' \\mathbf{V}_i^{-1} \\mathbf{y}_i}
                   {\\mathbf{c}_i' \\mathbf{V}_i^{-1} \\mathbf{c}_i},
        \\qquad
        s_{1i}^2 = \\frac{1}{\\mathbf{c}_i' \\mathbf{V}_i^{-1} \\mathbf{c}_i},
        \\qquad \\mathbf{c}_i = \\mathbf{t}_i - \\bar t_i .

    Those slopes are then pooled by an ordinary random-effects meta-analysis,
    so the only between-study parameter is one scalar
    :math:`\\tau^2_{\\text{trend}}`.

    Why the ridge cannot reappear.  The cause of :math:`\\rho \\to 1` is the
    study-level *level*: a random effect shared by all of a study's intervals,
    moving them up and down together.  A centred contrast satisfies
    :math:`\\mathbf{c}_i' \\mathbf{1} = 0`, so that effect cancels **exactly**
    rather than approximately.  Centring on the information-weighted mean also
    makes the slope orthogonal to the intercept within each study, which is why
    the level and the trend can be estimated separately at all.

    What is given up: this is a test of a *linear* change in the log hazard
    ratio.  A hazard ratio that falls and then flattens has a slope near zero
    even though it is not constant, so use :func:`piecewise_joint`'s
    ``trend_test("any")`` when the shape itself is the question, and read
    :meth:`PiecewiseJointResult.structure_sensitivity` with it.

    Args:
        time, event, study, treat, breaks, min_events: as in
            :func:`piecewise_exp`.
        times: time to attach to each interval, one value per interval.  The
            default is the midpoint.  Pass explicit values when the cut points
            were chosen unevenly and the midpoints are not what the slope should
            be taken against.
        rho: within-study correlation imposed on the sampling covariance
            before the slope is taken, for aggregate input.  It changes
            :math:`s_{1i}^2` — a within-study model assumption — and is not a
            between-study correlation, because there is none to estimate.
        method: ``tau^2`` estimator passed to :func:`~pimeta.rma`.
        test: ``"knha"`` (default; Knapp–Hartung), ``"t"``, ``"z"`` or
            ``"adhoc"``.  Knapp–Hartung is the default here because it is the
            reason for taking this route: it needs a single variance, which the
            joint model does not have.
        level: confidence level.
        min_events: as in :func:`piecewise_exp`.

    Returns:
        A dict with ``k``, ``studies``, ``slope`` and ``slope_var`` (per
        study), ``beta``, ``se``, ``ci_lb``, ``ci_ub``, ``p_value``,
        ``tau2``, ``I2``, ``Q``, ``df``, ``ratio`` (the hazard ratio per unit
        of time), ``times``, ``excluded`` (study -> why), and ``model``, the
        underlying :class:`~pimeta.RmaResult`.

    Raises:
        ValueError: on mismatched lengths, a treatment indicator that is not
            0/1, fewer than three boundaries, fewer than two studies, a
            ``rho`` outside ``[-1, 1]``, a ``times`` of the wrong length, or
            fewer than two studies with two or more estimable intervals.
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
    if not set(np.unique(treat).tolist()) <= {0, 1}:
        raise ValueError("'treat' must be a 0/1 indicator")
    if rho is not None and not -1.0 <= float(rho) <= 1.0:
        raise ValueError("'rho' must lie in [-1, 1]")
    if breaks.size < 3:
        raise ValueError("a trend needs at least two intervals, so 'breaks' "
                         "must have three or more boundaries")
    level = float(level)
    if level <= 1.0:
        level *= 100.0
    k_int = breaks.size - 1
    labels = _interval_labels(breaks)
    if times is None:
        # ``split_followup`` truncates each study at ``breaks[-1]``, so every
        # interval is closed and the midpoint is well defined for all of them.
        points = np.asarray(0.5 * (breaks[:-1] + breaks[1:]), dtype=float)
    else:
        points = np.asarray(times, dtype=float).ravel()
        if points.size != k_int:
            raise ValueError(f"'times' must give one time per interval "
                             f"({k_int}), not {points.size}")
    if len(np.unique(study)) < 2:
        raise ValueError("at least two studies are needed to pool")

    studies, log_hr, se_hr, cov_hr, dropped = _fit_study_blocks(
        time, event, study, treat, breaks, min_events)

    slopes: list[float] = []
    variances: list[float] = []
    kept: list = []
    excluded: dict[str, str] = {str(s): why for s, why in dropped}
    for i, s in enumerate(studies):
        ok = np.isfinite(log_hr[i])
        if int(ok.sum()) < 2:
            excluded[str(s)] = "fewer than two intervals with an estimate"
            continue
        idx = np.flatnonzero(ok)
        block = np.array(cov_hr[i], dtype=float)[np.ix_(idx, idx)]
        if rho is not None:
            sd = np.sqrt(np.diag(block))
            corr = np.full(block.shape, float(rho))
            np.fill_diagonal(corr, 1.0)
            block = corr * np.outer(sd, sd)
        try:
            info = np.linalg.inv(block)
        except np.linalg.LinAlgError:
            excluded[str(s)] = "the within-study covariance is not invertible"
            continue
        t = points[idx]
        ones = np.ones(idx.size)
        weight = float(ones @ info @ ones)
        if weight <= 0.0:
            excluded[str(s)] = "the within-study covariance is not positive "
            excluded[str(s)] += "definite"
            continue
        centred = t - float(ones @ info @ t) / weight
        denom = float(centred @ info @ centred)
        if denom <= 1e-14:
            excluded[str(s)] = "the interval times are degenerate for this study"
            continue
        slopes.append(float(centred @ info @ log_hr[i][idx]) / denom)
        variances.append(1.0 / denom)
        kept.append(s)

    if len(kept) < 2:
        raise ValueError(
            "fewer than two studies have two or more estimable intervals, so "
            "there is no between-study variance to estimate; the study-level "
            "slopes are in 'excluded'")

    fit = rma(np.array(slopes, dtype=float), vi=np.array(variances, dtype=float),
              method=method, test=test, level=level / 100.0,
              slab=[str(s) for s in kept])
    return {
        "k": len(kept),
        "studies": kept,
        "slope": np.array(slopes, dtype=float),
        "slope_var": np.array(variances, dtype=float),
        "beta": float(fit.b[0]),
        "se": float(fit.se[0]),
        "ci_lb": float(fit.ci_lb[0]),
        "ci_ub": float(fit.ci_ub[0]),
        "p_value": float(fit.pval[0]),
        "df": fit.dfs,
        "tau2": float(fit.tau2),
        "I2": float(fit.I2),
        "Q": float(fit.QE),
        "ratio": float(np.exp(fit.b[0])),
        "times": points.copy(),
        "labels": labels,
        "excluded": excluded,
        "rho": None if rho is None else float(rho),
        "method": method,
        "test": test,
        "level": level,
        "model": fit,
    }


