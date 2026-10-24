"""Individual participant data: one model per study, then pool.

The two-stage route to IPD meta-analysis. Each study is analysed on its own
participants, producing an estimate and its variance; the estimates are then
pooled by :func:`~pimeta.rma_uni.rma`. It is the route that reuses a
meta-analysis engine unchanged, and its weakness is well known — a study with
few events contributes a wide interval, and a study that cannot be fitted
contributes nothing.

The one-stage route (a single model over all participants with participants
nested in studies) is a different model, not a refinement of this one, and it
needs a participant-level likelihood; it is not here.

    >>> fit = ma.ipd_two_stage(y, study=study, treat=treat, covariates=[age])
    >>> print(fit.coef_table())
    >>> for s in fit.per_study:
    ...     print(s.study, s.n, s.events, s.b[1], s.se[1])

A covariate is centred option — the interaction between treatment and a
covariate is the point of much IPD work, and it is obtained by naming the
product as the term to pool:

    >>> fit = ma.ipd_two_stage(y, study=study, treat=treat, covariates=[age_c],
    ...                        term="treat:age_c")
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

import numpy as np

__all__ = [
    "IpdResult",
    "StudyFit",
    "ipd_two_stage",
    "linear_model",
    "logistic_model",
]


def _label(value) -> str:
    """A study label that reads ``0`` rather than ``0.0``."""
    try:
        number = float(value)
    except (TypeError, ValueError):
        return str(value)
    return str(int(number)) if number.is_integer() else str(value)


def _expit(eta):
    out = np.empty_like(eta, dtype=float)
    pos = eta >= 0
    out[pos] = 1.0 / (1.0 + np.exp(-eta[pos]))
    ex = np.exp(eta[~pos])
    out[~pos] = ex / (1.0 + ex)
    return out


def logistic_model(y, X, *, max_iter: int = 200, tol: float = 1e-10):
    """Maximum-likelihood logistic regression by iteratively reweighted least squares.

    Args:
        y: 0/1 outcome.
        X: ``(n, p)`` design matrix, intercept column included.
        max_iter: iteration cap.
        tol: convergence tolerance on the largest absolute coefficient change.

    Returns:
        ``(b, vb, iterations, converged)``.

    Raises:
        ValueError: on a non-binary outcome, a rank-deficient design, or
            complete separation (a study whose covariate perfectly predicts the
            outcome has no finite estimate, and inventing one would be worse
            than saying so).
    """
    y = np.asarray(y, dtype=float).ravel()
    X = np.asarray(X, dtype=float)
    if not np.all((y == 0.0) | (y == 1.0)):
        raise ValueError("'y' must be coded 0/1")
    n, p = X.shape
    if n <= p:
        raise ValueError(
            f"study has {n} participants and {p} parameters; the model cannot "
            "be fitted")
    if np.linalg.matrix_rank(X) < p:
        raise ValueError("the design matrix is rank deficient")

    b = np.zeros(p)
    converged = False
    iterations = 0
    for count in range(1, int(max_iter) + 1):
        iterations = count
        eta = X @ b
        mu = _expit(eta)
        w = mu * (1.0 - mu)
        if np.any(w <= 0.0):
            raise ValueError(
                "complete separation: a fitted probability is exactly 0 or 1, "
                "so the maximum-likelihood estimate is infinite")
        xtw = X.T * w
        h = xtw @ X
        step = np.linalg.solve(h, xtw @ (eta + (y - mu) / w))
        change = float(np.max(np.abs(step - b)))
        b = step
        if change < tol:
            converged = True
            break
    eta = X @ b
    mu = _expit(eta)
    w = mu * (1.0 - mu)
    vb = np.linalg.inv((X.T * w) @ X)
    return b, vb, iterations, converged


def linear_model(y, X):
    """Ordinary least squares with the residual variance estimated.

    Args:
        y: the outcome.
        X: ``(n, p)`` design matrix, intercept column included.

    Returns:
        ``(b, vb)``.

    Raises:
        ValueError: when there are no residual degrees of freedom, or the
            design is rank deficient.
    """
    y = np.asarray(y, dtype=float).ravel()
    X = np.asarray(X, dtype=float)
    n, p = X.shape
    if n <= p:
        raise ValueError(
            f"study has {n} participants and {p} parameters; the model cannot "
            "be fitted")
    if np.linalg.matrix_rank(X) < p:
        raise ValueError("the design matrix is rank deficient")
    xtx_inv = np.linalg.inv(X.T @ X)
    b = xtx_inv @ X.T @ y
    resid = y - X @ b
    sigma2 = float(resid @ resid) / (n - p)
    return b, sigma2 * xtx_inv


@dataclass
class StudyFit:
    """One study's first-stage model.

    Attributes:
        study: the study label.
        n: participants in the analysis.
        events: outcome events (``None`` for a continuous outcome).
        coefnames: ``["(Intercept)", ...]``.
        b, se: coefficients and their standard errors.
        vb: covariance matrix.
        term: index of the coefficient being pooled.
        converged: whether the first stage converged.
    """

    study: str
    n: int
    events: int | None
    coefnames: list[str]
    b: list[float]
    se: list[float]
    vb: np.ndarray = field(repr=False)
    term: int
    converged: bool

    @property
    def estimate(self) -> float:
        return self.b[self.term]

    @property
    def variance(self) -> float:
        return float(self.vb[self.term, self.term])

    def to_dict(self) -> dict:
        return {
            "study": self.study,
            "n": self.n,
            "events": self.events,
            "coefnames": list(self.coefnames),
            "b": list(self.b),
            "se": list(self.se),
            "converged": self.converged,
        }


@dataclass
class IpdResult:
    """A pooled individual-participant analysis.

    Attributes:
        term: name of the coefficient that was pooled.
        b, se, zval, pval, ci_lb, ci_ub: the pooled estimate.
        tau2, QE, QE_df, QE_p, I2: between-study heterogeneity.
        k: number of studies that contributed.
        studies: studies present in the data.
        excluded: studies that could not be fitted, with the reason.
        method: τ² estimator.
        level: confidence level.
        test: reference distribution (``"z"`` unless ``test=`` was given).
        per_study: one :class:`StudyFit` per contributing study.
        fit: the underlying :class:`~pimeta.rma_uni.RmaResult`.
    """

    term: str
    b: float
    se: float
    zval: float
    pval: float
    ci_lb: float
    ci_ub: float
    tau2: float
    QE: float
    QE_df: int
    QE_p: float
    I2: float
    k: int
    studies: int
    excluded: dict[str, str]
    method: str
    level: float
    test: str
    per_study: list[StudyFit] = field(repr=False)
    fit: Any = field(repr=False)

    def to_dict(self) -> dict:
        return {
            "term": self.term,
            "b": self.b,
            "se": self.se,
            "zval": self.zval,
            "pval": self.pval,
            "ci_lb": self.ci_lb,
            "ci_ub": self.ci_ub,
            "tau2": self.tau2,
            "QE": self.QE,
            "QE_df": self.QE_df,
            "QE_p": self.QE_p,
            "I2": self.I2,
            "k": self.k,
            "studies": self.studies,
            "excluded": dict(self.excluded),
            "method": self.method,
            "level": self.level,
            "term_fit": "pooled across studies",
            "per_study": [s.to_dict() for s in self.per_study],
        }

    def coef_table(self) -> list[dict]:
        return [{
            "term": self.term,
            "estimate": self.b,
            "se": self.se,
            "zval": self.zval,
            "pval": self.pval,
            "ci_lb": self.ci_lb,
            "ci_ub": self.ci_ub,
        }]

    def per_study_table(self) -> list[dict]:
        """The forest-plot table: one row per study plus the pooled estimate."""
        rows = [
            {
                "study": s.study,
                "n": s.n,
                "events": s.events,
                "estimate": s.estimate,
                "se": float(np.sqrt(s.variance)),
                "ci_lb": s.estimate - 1.96 * float(np.sqrt(s.variance)),
                "ci_ub": s.estimate + 1.96 * float(np.sqrt(s.variance)),
            }
            for s in self.per_study
        ]
        rows.append({
            "study": "Pooled",
            "n": sum(s.n for s in self.per_study),
            "events": (None if self.per_study[0].events is None
                       else sum(s.events or 0 for s in self.per_study)),
            "estimate": self.b,
            "se": self.se,
            "ci_lb": self.ci_lb,
            "ci_ub": self.ci_ub,
        })
        return rows


def ipd_two_stage(y, *, study, treat, covariates=None, family: str = "binomial",
                  term: str = "treat", method: str = "REML",
                  level: float = 0.95, test: str = "z",
                  min_events: int = 1) -> IpdResult:
    """Fit one model per study, then pool one coefficient.

    Args:
        y: the outcome, one entry per participant.
        study: the study label, one entry per participant.
        treat: the treatment indicator (0/1), one entry per participant. It is
            put first in the design, so ``term="treat"`` means the treatment
            effect.
        covariates: optional list of participant-level covariates, each one
            entry per participant.
        family: ``"binomial"`` (logistic) or ``"gaussian"`` (linear).
        term: which coefficient to pool — ``"treat"``, a covariate name, or a
            product such as ``"treat:age"``. The product is formed for you.
        method: τ² estimator for the second stage.
        level: confidence level.
        test: ``"z"``, ``"t"``, ``"knha"`` or ``"adhoc"`` for the second stage.
        min_events: studies with fewer events than this are excluded (a study
            with no events in one arm has no finite logistic estimate).

    Returns:
        :class:`IpdResult`.

    Raises:
        ValueError: when the inputs disagree in length, the study has too few
            participants, or the requested term does not exist.

    Example:
        >>> fit = ma.ipd_two_stage(y, study=study, treat=treat)   # doctest: +SKIP
        >>> fit.b, fit.tau2, fit.k                                # doctest: +SKIP
    """
    from .rma_uni import rma

    y = np.asarray(y, dtype=float).ravel()
    labels = [_label(s) for s in np.asarray(study).ravel()]
    treat = np.asarray(treat, dtype=float).ravel()
    if not (y.size == len(labels) == treat.size):
        raise ValueError("'y', 'study' and 'treat' must be the same length")
    if covariates is None:
        cov_map: dict[str, np.ndarray] = {}
    elif isinstance(covariates, Mapping):
        cov_map = {str(k): np.asarray(v, dtype=float).ravel()
                   for k, v in covariates.items()}
    else:
        cov_map = {f"cov{i + 1}": np.asarray(v, dtype=float).ravel()
                   for i, v in enumerate(covariates)}
    for c in cov_map.values():
        if c.size != y.size:
            raise ValueError("each covariate needs one value per participant")

    fam = str(family).lower()
    if fam not in ("binomial", "gaussian"):
        raise ValueError(
            f"unknown family {family!r}; use 'binomial' or 'gaussian'")

    groups = list(dict.fromkeys(labels))
    per_study: list[StudyFit] = []
    excluded: dict[str, str] = {}
    for g in groups:
        rows = np.array([i for i, lab in enumerate(labels) if lab == g])
        yg = y[rows]
        tg = treat[rows]
        names = ["(Intercept)", "treat"] + list(cov_map)
        cols = [np.ones(rows.size), tg] + [c[rows] for c in cov_map.values()]
        if ":" in term:
            left, _, right = term.partition(":")
            if left not in names or right not in names:
                raise ValueError(
                    f"cannot build {term!r}; the design has {names}")
            left_values = cols[names.index(left)]
            right_values = cols[names.index(right)]
            names.append(term)
            cols.append(left_values * right_values)
        X = np.column_stack(cols)
        if term not in names:
            raise ValueError(
                f"unknown term {term!r}; the design has {names}")
        index = names.index(term)
        events = None
        if fam == "binomial":
            events = int(np.sum(yg == 1.0))
            if events < int(min_events):
                excluded[g] = (
                    f"{events} event(s) in {yg.size} participants; "
                    f"fewer than min_events={int(min_events)}")
                continue
            # A logistic model needs both arms to vary in the outcome.  The
            # check used to be ``events == non-events``, which threw away
            # perfectly balanced studies and said nothing about the arms.
            arms = [yg[tg == 1.0], yg[tg == 0.0]]
            if any(a.size == 0 for a in arms):
                excluded[g] = "only one treatment arm is present"
                continue
            if any(np.all(a == a[0]) for a in arms):
                excluded[g] = (
                    "an arm has 0 or all events, so the logistic "
                    "maximum-likelihood estimate is infinite")
                continue
        try:
            if fam == "binomial":
                b, vb, _iters, converged = logistic_model(yg, X)
            else:
                b, vb = linear_model(yg, X)
                converged = True
        except ValueError as exc:
            excluded[g] = str(exc)
            continue
        per_study.append(StudyFit(
            study=g, n=int(rows.size), events=events, coefnames=names,
            b=[float(v) for v in b],
            se=[float(np.sqrt(max(vb[i, i], 0.0))) for i in range(b.size)],
            vb=np.asarray(vb, dtype=float), term=index, converged=converged,
        ))

    if not per_study:
        raise ValueError(
            f"no study could be fitted (excluded: {excluded})")

    estimates = np.array([s.estimate for s in per_study])
    variances = np.array([s.variance for s in per_study])
    model = rma(estimates, vi=variances, method=method, level=level,
                test=test, slab=[s.study for s in per_study])
    return IpdResult(
        term=term,
        b=float(model.b[0]),
        se=float(model.se[0]),
        zval=float(model.zval[0]),
        pval=float(model.pval[0]),
        ci_lb=float(model.ci_lb[0]),
        ci_ub=float(model.ci_ub[0]),
        tau2=float(np.asarray(model.tau2, dtype=float).ravel()[0]),
        QE=float(model.QE),
        QE_df=int(model.k - model.p),
        QE_p=float(model.QEp if hasattr(model, "QEp") else float("nan")),
        I2=float(getattr(model, "I2", float("nan"))),
        k=len(per_study),
        studies=len(groups),
        excluded=excluded,
        method=str(model.method),
        level=float(level),
        test=str(getattr(model, "test", "z")),
        per_study=per_study,
        fit=model,
    )
