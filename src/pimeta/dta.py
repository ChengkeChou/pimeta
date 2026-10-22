"""Diagnostic test accuracy: the bivariate model and HSROC.

Two ways to synthesise a 2x2 diagnostic accuracy table, and they turn out to
be the same model in different clothes.

**Bivariate model** (:func:`reitsma`, after Reitsma et al. 2005)
    Each study contributes a pair ``(logit sensitivity, logit FPR)``.  The
    pair has a known sampling covariance (taken as diagonal: the diseased and
    non-diseased groups are different patients) and an unknown study-level
    covariance::

        mu = (mu_1, mu_2)'        mean logit sensitivity, mean logit FPR
        Psi = [[tau1^2, rho*tau1*tau2], [rho*tau1*tau2, tau2^2]]

**HSROC model** (Rutter & Gatsonis 2001)
    The same data parametrised by a threshold random effect ``theta`` and an
    accuracy random effect ``alpha``, plus a fixed asymmetry ``beta``::

        logit(sens_i) = (theta_i + alpha_i/2) * exp(-beta/2)
        logit(fpr_i)  = (theta_i - alpha_i/2) * exp(+beta/2)

Harbord et al. (2007) showed the two are reparameterisations of one another,
so :func:`hsroc` does not fit anything: it applies mada's exact algebraic
transform to the bivariate fit.  Both therefore report the same likelihood and
the same AIC, which is the correct answer rather than a coincidence.

Parity, and where this module deliberately differs from ``mada``
--------------------------------------------------------------
The estimates match ``mada`` 0.5.12 exactly (see ``tests/test_dta.py``), but
two of mada's behaviours are **not** adopted, because they are not defensible:

* **The continuity correction is per study by default.**  mada's
  ``correction.control="all"`` adds the correction to *every* study as soon as
  any table anywhere has a zero cell.  That is a global scalar ``if`` wrapped
  around a vectorised ``any()`` -- almost certainly an ``ifelse`` written as
  ``if``, since the ``"single"`` branch right beside it is element-wise.  It
  also breaks locality (adding one study moves everyone else's data) and it
  hits hardest where the sample is smallest, i.e. exactly where the correction
  is not needed.  Use ``correction_control="all"`` to reproduce mada bit for
  bit; the default here is ``"single"``, and any correction that changes the
  counts raises :class:`DtaCorrectionWarning`.
* **The second component is the specificity, not the false positive rate.**
  The two are the same model with a sign flip, but reporting specificity keeps
  the between-study correlation on the scale the DTA literature uses: a
  threshold effect is a *negative* ``rho``, not a positive one.  The
  FPR-scale values remain available as :attr:`ReitsmaResult.mu_fpr`,
  ``Psi_fpr``, ``vcov_fpr`` and ``rho_fpr``, which is what mada prints.

``mada`` reports a log-likelihood with a Jacobian correction added, the change
of variables from the logit scale back to the probability scale.  That is what
its AIC is built on, so :attr:`ReitsmaResult.loglik` keeps it and
:attr:`ReitsmaResult.loglik_raw` exposes the model's own value.
"""

from __future__ import annotations

import warnings
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from scipy import optimize, stats

__all__ = [
    "DtaData",
    "DtaCorrectionWarning",
    "HsrocResult",
    "ReitsmaResult",
    "SrocResult",
    "auc",
    "escalc_dta",
    "hsroc",
    "reitsma",
    "sroc",
]


class DtaCorrectionWarning(UserWarning):
    """A continuity correction changed the counts.

    Emitted whenever ``escalc_dta`` adds something to a cell, naming how many
    studies were affected.  A correction is a modelling decision, not a
    formatting detail, so it should never happen silently.
    """


def _warn(message: str) -> None:
    warnings.warn(message, DtaCorrectionWarning, stacklevel=3)

_EPS = np.finfo(float).eps
#: A variance this small is indistinguishable from zero and would make the
#: study-level covariance singular.
_TAU_FLOOR = 1e-6


# --------------------------------------------------------------------------- #
# data preparation
# --------------------------------------------------------------------------- #
@dataclass
class DtaData:
    """Per-study diagnostic accuracy data, after any continuity correction.

    Attributes:
        labels: study labels.
        tp, fn, fp, tn: the four cells *after* correction.
        sens: sensitivity ``tp / (tp + fn)``.
        fpr: false positive rate ``fp / (fp + tn)``.
        var_sens: variance of ``logit(sens)``.
        var_fpr: variance of ``logit(fpr)``.
        n_positive: diseased participants per study.
        n_negative: non-diseased participants per study.
        corrected: which studies had a zero cell before correction.
    """

    labels: list[str]
    tp: np.ndarray
    fn: np.ndarray
    fp: np.ndarray
    tn: np.ndarray
    sens: np.ndarray
    fpr: np.ndarray
    var_sens: np.ndarray
    var_fpr: np.ndarray
    n_positive: np.ndarray
    n_negative: np.ndarray
    corrected: np.ndarray
    correction: float = 0.5
    #: Sensitivity and FPR computed from the *uncorrected* counts.  mada uses
    #: these when it decides which part of the SROC curve was observed, so the
    #: partial AUC depends on them rather than on the corrected values.
    sens_raw: np.ndarray | None = field(default=None, repr=False)
    fpr_raw: np.ndarray | None = field(default=None, repr=False)

    def __len__(self) -> int:
        return int(self.sens.size)

    @property
    def k(self) -> int:
        """Number of studies."""
        return len(self)

    @property
    def spec(self) -> np.ndarray:
        """Specificity, ``1 - fpr``."""
        return 1.0 - self.fpr

    def to_dict(self) -> dict:
        """A JSON-able summary of the prepared data."""
        return {
            "k": self.k,
            "labels": list(self.labels),
            "tp": self.tp.tolist(),
            "fn": self.fn.tolist(),
            "fp": self.fp.tolist(),
            "tn": self.tn.tolist(),
            "sens": self.sens.tolist(),
            "fpr": self.fpr.tolist(),
            "var_sens": self.var_sens.tolist(),
            "var_fpr": self.var_fpr.tolist(),
            "corrected": [bool(c) for c in self.corrected],
            "correction": self.correction,
        }


def _as_array(values: Any, name: str) -> np.ndarray:
    arr = np.asarray(values, dtype=float).ravel()
    if arr.size == 0:
        raise ValueError(f"{name!r} is empty")
    if np.any(~np.isfinite(arr)) or np.any(arr < 0):
        raise ValueError(f"{name!r} must be finite and non-negative")
    return arr


def escalc_dta(
    tp: Any,
    fn: Any = None,
    fp: Any = None,
    tn: Any = None,
    *,
    slab: Sequence[Any] | None = None,
    correction: float = 0.5,
    correction_control: str = "single",
) -> DtaData:
    """Turn 2x2 counts into logit sensitivity / logit FPR with variances.

    Args:
        tp: true positives, or a mapping with ``TP``/``FN``/``FP``/``TN`` keys
            (when the other three arguments are omitted).
        fn: false negatives.
        fp: false positives.
        tn: true negatives.
        slab: study labels; defaults to ``"Study 1"``, ``"Study 2"``, ...
        correction: value added to the cells, ``0.5`` by default.
        correction_control: ``"single"`` (default) adds ``correction`` only to
            the studies that have a zero cell, ``"none"`` never corrects, and
            ``"all"`` reproduces ``mada``, which adds it to *every* study as
            soon as any table anywhere has a zero cell.

            ``"all"`` is deliberately not the default here.  It is not a
            correction any more, it is a data-dependent perturbation of studies
            that were fine: adding one study with a zero cell moves every other
            study's estimate, and the bias is largest where the sample is
            smallest.  Use it only to reproduce ``mada``'s output exactly.

    Returns:
        A :class:`DtaData`.

    Raises:
        ValueError: for inconsistent input or an unknown ``correction_control``.
    """
    if fn is None and fp is None and tn is None:
        counts = _counts_from(tp)
        tp, fn, fp, tn = counts["tp"], counts["fn"], counts["fp"], counts["tn"]
        if slab is None:
            slab = counts.get("labels")

    tp = _as_array(tp, "tp")
    fn = _as_array(fn, "fn")
    fp = _as_array(fp, "fp")
    tn = _as_array(tn, "tn")
    if not (tp.size == fn.size == fp.size == tn.size):
        raise ValueError("tp, fn, fp and tn must all have the same length")
    if correction_control not in ("all", "single", "none"):
        raise ValueError(
            f"correction_control must be 'all', 'single' or 'none', "
            f"not {correction_control!r}"
        )
    if correction < 0:
        raise ValueError("correction must be non-negative")

    k = tp.size
    zero_cell = (tp == 0) | (fn == 0) | (fp == 0) | (tn == 0)
    add = np.zeros(k)
    if correction > 0 and zero_cell.any():
        if correction_control == "all":
            # mada: one zero cell anywhere means every study is corrected.
            add[:] = correction
            n_clean = int(k - zero_cell.sum())
            _warn(
                f"continuity correction {correction:g} applied to all {k} "
                f"studies because {int(zero_cell.sum())} had a zero cell"
                + (f", including {n_clean} that did not need it" if n_clean else "")
                + "; this reproduces mada's default - pass "
                "correction_control='single' to correct only the studies that "
                "need it"
            )
        elif correction_control == "single":
            add[zero_cell] = correction
            _warn(
                f"continuity correction {correction:g} applied to "
                f"{int(zero_cell.sum())} of {k} studies with a zero cell"
            )

    tpc, fnc, fpc, tnc = tp + add, fn + add, fp + add, tn + add
    n_pos = tpc + fnc
    n_neg = fpc + tnc
    sens = tpc / n_pos
    fpr = fpc / n_neg

    # delta method on the logit scale: var = 1/tp + 1/fn for sensitivity,
    # 1/fp + 1/tn for the false positive rate.  With correction="none" a zero
    # cell gives an infinite variance, which is the honest answer rather than
    # a warning worth printing.
    with np.errstate(divide="ignore", invalid="ignore"):
        var_sens = 1.0 / tpc + 1.0 / fnc
        var_fpr = 1.0 / fpc + 1.0 / tnc

    if slab is None:
        labels = [f"Study {i + 1}" for i in range(k)]
    else:
        labels = [str(s) for s in np.asarray(slab, dtype=object).ravel()]
        if len(labels) != k:
            raise ValueError("slab must have one entry per study")

    return DtaData(
        labels=labels,
        tp=tpc,
        fn=fnc,
        fp=fpc,
        tn=tnc,
        sens=sens,
        fpr=fpr,
        var_sens=var_sens,
        var_fpr=var_fpr,
        n_positive=n_pos,
        n_negative=n_neg,
        corrected=add > 0,
        correction=float(correction),
        sens_raw=np.where(tp + fn > 0, tp / np.maximum(tp + fn, 1), np.nan),
        fpr_raw=np.where(fp + tn > 0, fp / np.maximum(fp + tn, 1), np.nan),
    )


def _lookup(data: Any, *names: str):
    """Fetch a column from a mapping or a frame-like object."""
    for name in names:
        if isinstance(data, Mapping):
            for key in (name, name.lower(), name.upper(), name.capitalize()):
                if key in data:
                    return data[key]
        else:
            if hasattr(data, "__getitem__"):
                try:
                    return data[name]
                except (KeyError, IndexError, TypeError):
                    pass
            if hasattr(data, "columns") and name in data.columns:
                return data[name]
    return None


def _counts_from(data: Any) -> dict:
    """Extract ``tp/fn/fp/tn`` (and labels) from a dict or data frame."""
    if isinstance(data, DtaData):
        return {"tp": data.tp, "fn": data.fn, "fp": data.fp, "tn": data.tn,
                "labels": data.labels}
    found = {k: _lookup(data, *aliases) for k, aliases in (
        ("tp", ("TP", "tp", "tpos", "true_positive")),
        ("fn", ("FN", "fn", "fneg", "false_negative")),
        ("fp", ("FP", "fp", "fpos", "false_positive")),
        ("tn", ("TN", "tn", "tneg", "true_negative")),
    )}
    missing = [k for k, v in found.items() if v is None]
    if missing:
        raise ValueError(
            f"could not find the cell column(s) {missing}; expected TP/FN/FP/TN"
        )
    labels = _lookup(data, "study", "Study", "slab", "id")
    found["labels"] = labels
    return found


# --------------------------------------------------------------------------- #
# the bivariate random-effects fit
# --------------------------------------------------------------------------- #
@dataclass
class ReitsmaResult:
    """A fitted bivariate diagnostic accuracy model.

    Attributes:
        mu: ``(mu_1, mu_2)`` -- mean logit sensitivity and mean logit
            **specificity**.  Reporting specificity rather than its complement
            keeps ``rho`` on the scale the DTA literature uses: a threshold
            effect shows up as a *negative* correlation.
        se: standard errors of ``mu``.
        ci_lb, ci_ub: confidence limits of ``mu`` on the logit scale.
        Psi: the 2x2 study-level covariance matrix, in (sens, spec)
            coordinates.
        tau: its standard deviations.
        rho: the between-study correlation of logit sensitivity and logit
            specificity (negative under the usual threshold effect).
        vcov: covariance matrix of ``mu``.
        summary_sens: back-transformed ``mu_1``.
        summary_spec: back-transformed ``mu_2``.
        summary_fpr: ``1 - summary_spec``.
        sens_ci, spec_ci, fpr_ci: confidence intervals on the probability
            scale.
        k: number of studies.
        method: ``"reml"`` or ``"ml"``.
        loglik: log-likelihood with mada's Jacobian correction (what AIC uses).
        loglik_raw: the model's own log-likelihood.
        AIC, BIC: information criteria using ``2 + 3`` parameters and
            ``2k`` observations, as mada does.
        mu_fpr, Psi_fpr, vcov_fpr, rho_fpr: the same quantities on the
            false-positive-rate scale ``mada`` reports, so its numbers can be
            recovered exactly.
        data: the :class:`DtaData` that was fitted.
    """

    mu: list[float]
    se: list[float]
    ci_lb: list[float]
    ci_ub: list[float]
    Psi: np.ndarray
    tau: list[float]
    rho: float
    vcov: np.ndarray
    summary_sens: float
    summary_fpr: float
    summary_spec: float
    sens_ci: list[float]
    spec_ci: list[float]
    fpr_ci: list[float]
    k: int
    method: str
    loglik: float
    loglik_raw: float
    AIC: float
    BIC: float
    level: float
    mu_fpr: list[float] = field(repr=False)
    Psi_fpr: np.ndarray = field(repr=False)
    vcov_fpr: np.ndarray = field(repr=False)
    rho_fpr: float = field(repr=False)
    data: DtaData = field(repr=False)

    # -- derived views ----------------------------------------------------- #
    @property
    def coefnames(self) -> list[str]:
        """Names of the two coefficients."""
        return ["(Intercept): logit(sens)", "(Intercept): logit(spec)"]

    def coef_table(self) -> dict:
        """The coefficient table as ``{column: list}``."""
        zval = [m / s for m, s in zip(self.mu, self.se)]
        return {
            "model": ["reitsma"] * 2,
            "term": self.coefnames,
            "estimate": list(self.mu),
            "se": list(self.se),
            "statistic": zval,
            "pval": [float(2 * stats.norm.sf(abs(z))) for z in zval],
            "ci_lb": list(self.ci_lb),
            "ci_ub": list(self.ci_ub),
        }

    def sroc(self, fpr: Any = None, *, type: str = "ruttergatsonis") -> SrocResult:
        """The summary ROC curve; see :func:`sroc`."""
        return sroc(self, fpr=fpr, type=type)

    def auc(self, fpr: Any = None) -> dict:
        """Area under the SROC curve; see :func:`auc`."""
        return auc(self, fpr=fpr)

    def to_dict(self) -> dict:
        """A JSON-able snapshot."""
        return {
            "coefnames": self.coefnames,
            "mu": list(self.mu),
            "se": list(self.se),
            "zval": [m / s for m, s in zip(self.mu, self.se)],
            "pval": [float(2 * stats.norm.sf(abs(m / s)))
                     for m, s in zip(self.mu, self.se)],
            "ci_lb": list(self.ci_lb),
            "ci_ub": list(self.ci_ub),
            "Psi": np.asarray(self.Psi).tolist(),
            "tau": list(self.tau),
            "rho": self.rho,
            "summary_sens": self.summary_sens,
            "summary_fpr": self.summary_fpr,
            "summary_spec": self.summary_spec,
            "sens_ci": list(self.sens_ci),
            "fpr_ci": list(self.fpr_ci),
            "k": self.k,
            "method": self.method,
            "loglik": self.loglik,
            "loglik_raw": self.loglik_raw,
            "AIC": self.AIC,
            "BIC": self.BIC,
        }


@dataclass
class SrocResult:
    """A point on (or a grid of) the summary ROC curve.

    Attributes:
        fpr: false positive rates.
        tpr: the corresponding sensitivities.
        type: ``"ruttergatsonis"`` (the HSROC curve through the fitted
            parameters) or ``"naive"``.
    """

    fpr: np.ndarray
    tpr: np.ndarray
    type: str = "ruttergatsonis"

    def to_dict(self) -> dict:
        """A JSON-able snapshot."""
        return {
            "type": self.type,
            "fpr": np.asarray(self.fpr).tolist(),
            "tpr": np.asarray(self.tpr).tolist(),
        }

    def to_frame(self) -> dict:
        """``{"fpr": [...], "sens": [...]}`` for tabular output."""
        return {"fpr": np.asarray(self.fpr).tolist(),
                "sens": np.asarray(self.tpr).tolist()}


def _logit(p: np.ndarray) -> np.ndarray:
    p = np.clip(np.asarray(p, dtype=float), 1e-300, 1 - 1e-16)
    return np.log(p) - np.log1p(-p)


def _jacobian_term(data: DtaData) -> float:
    """mada's Jacobian correction to the log-likelihood.

    Fitting on the logit scale means the likelihood is for the transformed
    data; ``sum(log jacobian)`` puts it back on the probability scale.  It is
    a constant with respect to the parameters, so it changes AIC and BIC but
    not the estimates.
    """
    return float(np.sum(-np.log(data.sens) - np.log1p(-data.sens)
                        - np.log(data.fpr) - np.log1p(-data.fpr)))


def _expit(x: Any) -> Any:
    return 1.0 / (1.0 + np.exp(-np.asarray(x, dtype=float)))


def _neg2_loglik(
    params: np.ndarray,
    y: np.ndarray,
    V: np.ndarray,
    method: str,
) -> float:
    """Profiled -2 log-likelihood for the given variance components.

    ``params`` is ``(log tau1, log tau2, atanh rho)``, which keeps the
    optimisation unconstrained.  The mean vector is profiled out by GLS, so the
    optimiser only has to move three parameters.
    """
    tau = np.exp(params[:2])
    rho = np.tanh(params[2]) * (1 - 1e-12)
    cov = rho * tau[0] * tau[1]
    Psi = np.array([[tau[0] ** 2, cov], [cov, tau[1] ** 2]])
    if not np.all(np.isfinite(Psi)):
        return np.inf

    M = V + Psi                                   # (k, 2, 2)
    try:
        chol = np.linalg.cholesky(M)
    except np.linalg.LinAlgError:
        return np.inf
    logdet = 2.0 * np.sum(np.log(np.diagonal(chol, axis1=1, axis2=2)))

    Minv = np.linalg.inv(M)
    sxx = Minv.sum(axis=0)                        # sum of M_i^-1
    sxy = np.einsum("kij,kj->i", Minv, y)
    try:
        mu = np.linalg.solve(sxx, sxy)
    except np.linalg.LinAlgError:
        return np.inf
    resid = y - mu
    quad = np.einsum("ki,kij,kj->", resid, Minv, resid)

    total = float(logdet.sum() + quad)
    if method == "reml":
        sign, logdet_sxx = np.linalg.slogdet(sxx)
        if sign <= 0:
            return np.inf
        total += logdet_sxx
    if not np.isfinite(total):
        return np.inf
    return total


def _fit_bivariate(y: np.ndarray, V: np.ndarray, method: str):
    """Fit ``y_i ~ N(mu, V_i + Psi)`` by ML or REML.

    Returns ``(mu, Psi, loglik_raw)``.
    """
    k = y.shape[0]
    sd = np.maximum(y.std(axis=0, ddof=1), 0.1)
    starts = [
        np.array([np.log(sd[0]), np.log(sd[1]), 0.0]),
        np.array([np.log(sd[0]), np.log(sd[1]), np.arctanh(0.5)]),
        np.array([np.log(sd[0] * 0.5), np.log(sd[1] * 0.5), np.arctanh(-0.5)]),
    ]

    best = None
    for start in starts:
        # Three stages, because each stalls where the next one is comfortable:
        # Nelder-Mead survives a flat, nearly singular surface; Powell is
        # derivative-free but quadratically accurate near the optimum; BFGS
        # finishes. Stopping after the first stage leaves ~1e-7 in the
        # coefficients, which is enough to break an R comparison.
        res = optimize.minimize(
            _neg2_loglik, start, args=(y, V, method), method="Nelder-Mead",
            options={"maxiter": 4000, "xatol": 1e-10, "fatol": 1e-12},
        )
        res = optimize.minimize(
            _neg2_loglik, res.x, args=(y, V, method), method="Powell",
            options={"xtol": 1e-12, "ftol": 1e-15, "maxiter": 5000},
        )
        res = optimize.minimize(
            _neg2_loglik, res.x, args=(y, V, method), method="BFGS",
            options={"gtol": 1e-12, "maxiter": 2000},
        )
        # A gradient method stops when its own gradient estimate looks small,
        # which on this near-singular surface happens around 1e-7.  Solving
        # score = 0 instead recovers most of the remaining digits.
        polished = optimize.root(
            lambda x: optimize.approx_fprime(x, _neg2_loglik, 1e-6, y, V, method),
            res.x, method="hybr", tol=1e-12,
        )
        if polished.success and np.isfinite(_neg2_loglik(polished.x, y, V, method)):
            res = optimize.OptimizeResult(x=polished.x,
                                          fun=_neg2_loglik(polished.x, y, V, method))
        if not np.isfinite(res.fun):
            continue
        if best is None or res.fun < best.fun:
            best = res
    if best is None:                             # pragma: no cover
        raise RuntimeError("the bivariate model could not be fitted")

    tau = np.exp(best.x[:2])
    rho = float(np.tanh(best.x[2]))
    cov = rho * tau[0] * tau[1]
    Psi = np.array([[tau[0] ** 2, cov], [cov, tau[1] ** 2]])

    M = V + Psi
    Minv = np.linalg.inv(M)
    mu = np.linalg.solve(Minv.sum(axis=0), np.einsum("kij,kj->i", Minv, y))

    # The normalising constant of the multivariate normal likelihood.  ML sees
    # 2k observations of dimension 2; REML replaces them with the 2k - p
    # residual degrees of freedom after profiling out the two means.
    n_obs = 2 * k
    dof = n_obs - 2 if method == "reml" else n_obs
    loglik = -0.5 * float(best.fun) - 0.5 * dof * np.log(2.0 * np.pi)
    return mu, Psi, float(loglik), k


def reitsma(
    data: Any = None,
    *,
    tp: Any = None,
    fn: Any = None,
    fp: Any = None,
    tn: Any = None,
    slab: Sequence[Any] | None = None,
    method: str = "reml",
    correction: float = 0.5,
    correction_control: str = "single",
    level: float = 0.95,
) -> ReitsmaResult:
    """Fit the bivariate diagnostic accuracy model (``mada::reitsma``).

    Args:
        data: a :class:`DtaData`, or a mapping/frame with ``TP``/``FN``/``FP``/
            ``TN`` columns (optionally ``study``).
        tp, fn, fp, tn: counts, used instead of ``data``.
        slab: study labels.
        method: ``"reml"`` (mada's default) or ``"ml"``.
        correction: continuity correction added to the cells.
        correction_control: ``"single"`` (default), ``"none"`` or ``"all"``.
            ``"all"`` reproduces ``mada``'s global correction; see
            :func:`escalc_dta` for why it is not the default.
        level: confidence level (``0.95``).

    Returns:
        A :class:`ReitsmaResult`.

    Raises:
        ValueError: for an unknown method or unusable counts.
    """
    method = str(method).lower()
    if method not in ("reml", "ml"):
        raise ValueError(f"method must be 'reml' or 'ml', not {method!r}")

    if isinstance(data, DtaData):
        prepared = data
    else:
        source = data if data is not None else {"tp": tp, "fn": fn, "fp": fp, "tn": tn}
        counts = _counts_from(source)
        if slab is None:
            slab = counts.get("labels")
        prepared = escalc_dta(
            counts["tp"], counts["fn"], counts["fp"], counts["tn"],
            slab=slab, correction=correction,
            correction_control=correction_control,
        )
    if prepared.k < 2:
        raise ValueError("the bivariate model needs at least two studies")

    y = np.column_stack([_logit(prepared.sens), _logit(prepared.fpr)])
    V = np.zeros((prepared.k, 2, 2))
    V[:, 0, 0] = np.maximum(prepared.var_sens, _TAU_FLOOR)
    V[:, 1, 1] = np.maximum(prepared.var_fpr, _TAU_FLOOR)

    mu_fpr, Psi_fpr, loglik_raw, k = _fit_bivariate(y, V, method)

    M = V + Psi_fpr
    Minv = np.linalg.inv(M)
    vcov_fpr = np.linalg.inv(Minv.sum(axis=0))

    # Report in the conventional coordinates: the second component is the
    # specificity, not the false positive rate.  Flip v -> -v.  The model is
    # untouched -- only the sign of mu_2, of the covariance and of rho changes --
    # so a threshold effect shows up as the negative correlation that the DTA
    # literature reports instead of its mirror image.
    flip = np.array([1.0, -1.0])
    outer = np.outer(flip, flip)
    mu = mu_fpr * flip
    Psi = Psi_fpr * outer
    vcov = vcov_fpr * outer

    se = np.sqrt(np.diag(vcov))

    z = stats.norm.ppf(1 - (1 - level) / 2)
    ci_lb = mu - z * se
    ci_ub = mu + z * se

    # mada adds the Jacobian of the logit transform back to the probability
    # scale, which is the likelihood its AIC is built on.
    loglik = float(loglik_raw + _jacobian_term(prepared))
    n_par = 5                                   # 2 means + 3 variance components
    aic = -2.0 * loglik + 2.0 * n_par
    bic = -2.0 * loglik + np.log(2.0 * k) * n_par

    tau = np.sqrt(np.diag(Psi))
    sens_lo, sens_hi = float(_expit(ci_lb[0])), float(_expit(ci_ub[0]))
    spec_lo, spec_hi = float(_expit(ci_lb[1])), float(_expit(ci_ub[1]))
    return ReitsmaResult(
        mu=[float(v) for v in mu],
        se=[float(v) for v in se],
        ci_lb=[float(v) for v in ci_lb],
        ci_ub=[float(v) for v in ci_ub],
        Psi=Psi,
        tau=[float(v) for v in tau],
        rho=float(Psi[0, 1] / (tau[0] * tau[1])),
        vcov=vcov,
        summary_sens=float(_expit(mu[0])),
        summary_spec=float(_expit(mu[1])),
        summary_fpr=float(1 - _expit(mu[1])),
        sens_ci=[sens_lo, sens_hi],
        spec_ci=[spec_lo, spec_hi],
        # 1 - spec reverses the interval, so the limits swap.
        fpr_ci=[1 - spec_hi, 1 - spec_lo],
        k=k,
        method=method,
        loglik=loglik,
        loglik_raw=float(loglik_raw),
        AIC=float(aic),
        BIC=float(bic),
        level=float(level),
        mu_fpr=[float(v) for v in mu_fpr],
        Psi_fpr=Psi_fpr,
        vcov_fpr=vcov_fpr,
        rho_fpr=float(
            Psi_fpr[0, 1] / np.sqrt(Psi_fpr[0, 0] * Psi_fpr[1, 1])
        ),
        data=prepared,
    )


# --------------------------------------------------------------------------- #
# HSROC: the same fit, reparameterised
# --------------------------------------------------------------------------- #
@dataclass
class HsrocResult:
    """HSROC parameters (Rutter & Gatsonis), derived from a bivariate fit.

    Attributes:
        Theta: mean threshold ``theta`` (positive = stricter threshold).
        Lambda: mean accuracy ``alpha``; ``exp(Lambda)`` is the diagnostic odds
            ratio at ``beta = 0``.
        beta: the asymmetry parameter; ``0`` means a symmetric SROC.
        sigma2theta: variance of the threshold random effect.
        sigma2alpha: variance of the accuracy random effect.
        summary_sens, summary_fpr: the summary point implied by the HSROC
            parameters.
        fit: the :class:`ReitsmaResult` this was derived from.
    """

    Theta: float
    Lambda: float
    beta: float
    sigma2theta: float
    sigma2alpha: float
    summary_sens: float
    summary_fpr: float
    summary_spec: float
    fit: ReitsmaResult = field(repr=False, compare=False)

    @property
    def dor(self) -> float:
        """Diagnostic odds ratio at the mean threshold."""
        return float(np.exp(self.Lambda))

    def to_dict(self) -> dict:
        """A JSON-able snapshot."""
        return {
            "Theta": self.Theta,
            "Lambda": self.Lambda,
            "beta": self.beta,
            "sigma2theta": self.sigma2theta,
            "sigma2alpha": self.sigma2alpha,
            "summary_sens": self.summary_sens,
            "summary_fpr": self.summary_fpr,
            "summary_spec": self.summary_spec,
            "DOR": self.dor,
        }

    def sroc(self, fpr: Any = None) -> SrocResult:
        """The HSROC summary curve."""
        return sroc(self.fit, fpr=fpr, type="ruttergatsonis")

    def auc(self, fpr: Any = None) -> dict:
        """Area under the HSROC summary curve."""
        return auc(self.fit, fpr=fpr)


def hsroc(result: Any = None, **kwargs: Any) -> HsrocResult:
    """HSROC parameters of a bivariate fit.

    The two models are equivalent (Harbord et al. 2007), so this does not
    re-fit: it applies the exact algebraic transform mada uses in
    ``calc_hsroc_coef``.  With the fit reported in (logit sensitivity, logit
    **specificity**) coordinates, and ``c = sqrt(sd_sens / sd_spec)``::

        beta   = log(sd_spec / sd_sens)
        Theta  = (mu_sens / c - c * mu_spec) / 2
        Lambda = mu_sens / c + c * mu_spec
        sigma2theta = (sd_sens*sd_spec - Psi_12) / 2
        sigma2alpha = 2 * (sd_sens*sd_spec + Psi_12)

    The signs differ from mada's version only because mada works on the
    false-positive-rate scale; the resulting ``Theta``, ``Lambda``, ``beta``
    and both variances are identical.

    Args:
        result: an :class:`ReitsmaResult`, or ``None`` to fit from ``kwargs``.
        **kwargs: forwarded to :func:`reitsma` when ``result`` is not a fit.

    Returns:
        An :class:`HsrocResult`.

    Raises:
        TypeError: when ``result`` is not a bivariate fit.
    """
    if isinstance(result, HsrocResult):
        return result
    if isinstance(result, ReitsmaResult):
        fit = result
    else:
        fit = reitsma(result, **kwargs)

    mu_sens, mu_spec = fit.mu
    sd_sens, sd_spec = fit.tau
    if sd_sens <= 0 or sd_spec <= 0:             # pragma: no cover
        raise ValueError(
            "HSROC parameters need positive between-study standard deviations; "
            "this fit collapsed one of them to zero"
        )

    covariance = float(fit.Psi[0, 1])
    c = np.sqrt(sd_sens / sd_spec)
    beta = float(np.log(sd_spec / sd_sens))
    Theta = float(0.5 * (mu_sens / c - c * mu_spec))
    Lambda = float(mu_sens / c + c * mu_spec)
    sigma2theta = float(0.5 * (sd_sens * sd_spec - covariance))
    sigma2alpha = float(2.0 * (sd_sens * sd_spec + covariance))

    # The HSROC summary point: threshold at Theta, accuracy at Lambda.
    half = np.exp(-beta / 2.0)
    logit_sens = (Theta + Lambda / 2.0) * half
    logit_spec = -(Theta - Lambda / 2.0) / half
    spec = float(_expit(logit_spec))

    return HsrocResult(
        Theta=Theta,
        Lambda=Lambda,
        beta=beta,
        sigma2theta=sigma2theta,
        sigma2alpha=sigma2alpha,
        summary_sens=float(_expit(logit_sens)),
        summary_spec=spec,
        summary_fpr=float(1 - spec),
        fit=fit,
    )


# --------------------------------------------------------------------------- #
# SROC and AUC
# --------------------------------------------------------------------------- #
def _sroc_tpr(fit: ReitsmaResult, fpr: np.ndarray) -> np.ndarray:
    """Rutter-Gatsonis SROC through the fitted parameters."""
    hs = hsroc(fit)
    return _expit(hs.Lambda * np.exp(-hs.beta / 2.0)
                  + np.exp(-hs.beta) * _logit(fpr))


def sroc(
    fit: ReitsmaResult,
    fpr: Any = None,
    *,
    type: str = "ruttergatsonis",
) -> SrocResult:
    """The summary ROC curve (``mada::sroc``).

    Args:
        fit: a :class:`ReitsmaResult`.
        fpr: false positive rates to evaluate; defaults to ``0.01, ..., 0.99``.
        type: ``"ruttergatsonis"`` (default) uses the HSROC curve implied by
            the fit, which is the line
            ``logit(TPR) = Lambda*exp(-beta/2) + exp(-beta)*logit(FPR)``.
            ``"naive"`` joins the observed points without smoothing.

    Returns:
        A :class:`SrocResult`.

    Raises:
        ValueError: for an unknown ``type``.
    """
    if type not in ("ruttergatsonis", "naive"):
        raise ValueError(f"type must be 'ruttergatsonis' or 'naive', not {type!r}")
    grid = np.arange(1, 100) / 100.0 if fpr is None else np.asarray(fpr, dtype=float)
    if np.any((grid <= 0) | (grid >= 1)):
        raise ValueError("fpr values must lie strictly between 0 and 1")
    if type == "naive":
        order = np.argsort(fit.data.fpr)
        tpr = np.interp(grid, fit.data.fpr[order], fit.data.sens[order])
    else:
        tpr = _sroc_tpr(fit, grid)
    return SrocResult(fpr=grid, tpr=np.asarray(tpr, dtype=float), type=type)


def _trapezoid_mada(sens: np.ndarray) -> float:
    """mada's ``AUC.default``: trapezoids on ``n`` points divided by ``n``.

    Note the divisor is ``n``, not ``n - 1``: the grid runs 0.01 ... 0.99 and
    the endpoints are not evaluated, so mada treats the gap as one more cell.
    Reproduced exactly so the numbers agree with R.
    """
    n = sens.size
    return float((sens[0] / 2 + sens[1:-1].sum() + sens[-1] / 2) / n)


def auc(fit: ReitsmaResult, fpr: Any = None) -> dict:
    """Area under the SROC curve (``mada::AUC``).

    Args:
        fit: a :class:`ReitsmaResult`.
        fpr: grid of false positive rates; defaults to ``0.01, ..., 0.99``.

    Returns:
        ``{"auc": ..., "pauc": ...}``.  The partial AUC restricts the grid to
        the range of FPRs actually observed, which is the number a reader
        should quote when the studies cluster at one end of the range.

    Raises:
        ValueError: when fewer than 10 grid points are supplied.
    """
    grid = np.arange(1, 100) / 100.0 if fpr is None else np.asarray(fpr, dtype=float)
    if grid.size < 10:
        raise ValueError("specify at least 10 FPR values")

    observed = np.asarray(
        fit.data.fpr_raw if fit.data.fpr_raw is not None else fit.data.fpr,
        dtype=float,
    )
    observed = observed[np.isfinite(observed)]
    lo = max(0.01, float(observed.min()))
    hi = min(0.99, float(observed.max()))
    partial_grid = np.linspace(lo, hi, 99)

    return {
        "auc": _trapezoid_mada(_sroc_tpr(fit, grid)),
        "pauc": _trapezoid_mada(_sroc_tpr(fit, partial_grid)),
    }
