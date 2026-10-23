"""One-stage individual-participant-data meta-analysis.

The alternative to :mod:`pimeta.ipd`, and a different model rather than a
refinement of it: every participant enters one model, with participants nested
in studies, so a study's own level is a random intercept and the treatment
effect is estimated across all of them at once.

    .. math::

        g(E[y_{ij}]) = x_{ij}'\\beta + u_i, \\qquad u_i \\sim N(0, \\tau^2)

For a continuous outcome the integral over :math:`u_i` is analytic, so the
likelihood is closed form and is maximised over :math:`(\\sigma^2, \\tau^2)`
directly — that is the ``family="gaussian"`` path. For a binary outcome the
integral is approximated by adaptive Gauss–Hermite quadrature, which is what
``family="binomial"`` does.

Why this is not a one-liner: a single-stage model needs a likelihood per
participant and a variance component, which is exactly what an inverse-variance
meta-analysis engine does not have. It is also why the two-stage route exists —
see :func:`~pimeta.ipd.ipd_two_stage` — and the two can disagree, which is a
result rather than a bug.

What this model does *not* do: study-specific residual variances, random
treatment slopes, and non-linear covariate effects. A random intercept and
fixed effects is the model the Cochrane guidance describes as the standard
one-stage analysis; the rest are extensions with their own identification
problems.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field

import numpy as np
from scipy import stats
from scipy.optimize import minimize

__all__ = ["OneStageResult", "ipd_one_stage"]


def _label(value) -> str:
    """A study label that reads ``0`` rather than ``0.0``."""
    try:
        number = float(value)
    except (TypeError, ValueError):
        return str(value)
    return str(int(number)) if number.is_integer() else str(value)


@dataclass
class OneStageResult:
    """A one-stage individual-participant model.

    Attributes:
        b, se, zval, pval, ci_lb, ci_ub: fixed effects.
        vb: their covariance matrix.
        coefnames: names, intercept first.
        tau2: between-study variance of the random intercept.
        sigma2: residual variance (``None`` for a binary outcome, where it is
            fixed at :math:`\\pi^2/3` by the logistic link).
        loglik: the maximised (restricted) log-likelihood.
        method: ``"REML"`` or ``"ML"``.
        participants: rows used.
        studies: number of studies.
        family: ``"gaussian"`` or ``"binomial"``.
        converged: whether the optimiser reported success.
        level: confidence level.
        test: always ``"z"`` — a one-stage model's fixed effects are tested
            against the normal reference unless the user asks otherwise.
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
    sigma2: float | None
    loglik: float
    method: str
    participants: int
    studies: int
    family: str
    converged: bool
    level: float
    test: str = "z"

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
            "sigma2": self.sigma2,
            "loglik": self.loglik,
            "method": self.method,
            "participants": self.participants,
            "studies": self.studies,
            "family": self.family,
            "converged": self.converged,
            "level": self.level,
        }

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

    @property
    def ICC(self) -> float | None:
        """Intraclass correlation, for a continuous outcome."""
        if self.sigma2 is None:
            return None
        total = self.tau2 + self.sigma2
        return float(self.tau2 / total) if total > 0 else None


# --------------------------------------------------------------------------- #
# the continuous outcome: closed form
# --------------------------------------------------------------------------- #
def _study_sums(X, y, rows):
    """Per-study cross-products, so the profile likelihood is cheap."""
    out = []
    for idx in rows:
        Xi = X[idx]
        yi = y[idx]
        n_i = idx.size
        one = np.ones(n_i)
        out.append({
            "n": n_i,
            "xtx": Xi.T @ Xi,
            "xty": Xi.T @ yi,
            "yty": float(yi @ yi),
            "xt1": Xi.T @ one,          # X'1
            "y1": float(yi @ one),      # y'1
        })
    return out


def _gls(sums, s2, tau2):
    """``X'V^-1 X``, ``X'V^-1 y``, ``y'V^-1 y`` and ``sum log|V_i|``.

    Uses ``V_i = s2 I + tau2 J``, whose inverse is
    ``s2^-1 (I - tau2 / (s2 + n_i tau2) J)``, so nothing is inverted and no
    participant-by-participant matrix is ever formed.
    """
    p = next(iter(sums))["xtx"].shape[0]
    xtvx = np.zeros((p, p))
    xtvy = np.zeros(p)
    ytvy = 0.0
    logdet = 0.0
    a = 1.0 / s2
    for s in sums:
        n_i = s["n"]
        # V_i^-1 = a I - b J, with a = 1/s2 and b = tau2 / (s2 (s2 + n_i tau2)).
        denom = s2 * (s2 + n_i * tau2)
        b = tau2 / denom if denom > 0 else 0.0
        xtvx += a * s["xtx"] - b * np.outer(s["xt1"], s["xt1"])
        xtvy += a * s["xty"] - b * s["xt1"] * s["y1"]
        ytvy += a * s["yty"] - b * s["y1"] * s["y1"]
        logdet += (n_i - 1) * np.log(s2) + np.log(s2 + n_i * tau2)
    return xtvx, xtvy, ytvy, logdet


def _gaussian_loglik(params, sums, n_total, p, reml):
    s2 = float(np.exp(params[0]))
    tau2 = float(np.exp(params[1]))
    xtvx, xtvy, ytvy, logdet = _gls(sums, s2, tau2)
    sign, logdet_xvx = np.linalg.slogdet(xtvx)
    if sign <= 0:
        return -1e10
    b = np.linalg.solve(xtvx, xtvy)
    rss = float(ytvy - b @ xtvy)
    if reml:
        # lme4 counts the normalising constant with the residual degrees of
        # freedom, so the value is comparable rather than merely proportional.
        value = -0.5 * (logdet + logdet_xvx + rss
                        + (n_total - p) * np.log(2.0 * np.pi))
    else:
        value = -0.5 * (logdet + rss + n_total * np.log(2.0 * np.pi))
    # A finite floor keeps the optimiser's finite differences meaningful; -inf
    # makes them nan and the line search gives up.
    return float(value) if np.isfinite(value) else -1e10


# --------------------------------------------------------------------------- #
# the binary outcome: adaptive Gauss-Hermite quadrature
# --------------------------------------------------------------------------- #
def _aghq_nodes(k: int):
    x, w = np.polynomial.hermite.hermgauss(int(k))
    return np.sqrt(2.0) * x, w / np.sqrt(np.pi)


def _binomial_loglik(params, Xs, ys, nodes, weights):
    beta = np.asarray(params[:-1], dtype=float)
    tau = float(np.exp(params[-1]))
    total = 0.0
    with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
        for X, y in zip(Xs, ys):
            eta = X @ beta                    # (n_i,)
            # (K, n_i): one row per quadrature node
            lin = eta[None, :] + tau * nodes[:, None]
            # log p(y | u) = y*lin - log(1 + exp(lin))
            ll = y[None, :] * lin - np.logaddexp(0.0, lin)
            per_node = np.log(weights) + ll.sum(axis=1)
            top = per_node.max()
            total += top + np.log(np.exp(per_node - top).sum())
    return float(total)


def _num_hessian(fun, x, step=1e-4):
    n = x.size
    H = np.zeros((n, n))
    f0 = fun(x)
    for i in range(n):
        for j in range(i, n):
            ei = np.zeros(n)
            ej = np.zeros(n)
            ei[i] = step
            ej[j] = step
            fpp = fun(x + ei + ej)
            fpm = fun(x + ei - ej)
            fmp = fun(x - ei + ej)
            fmm = fun(x - ei - ej)
            H[i, j] = H[j, i] = (fpp - fpm - fmp + fmm) / (4.0 * step * step)
    return H, f0


# --------------------------------------------------------------------------- #
# the public entry point
# --------------------------------------------------------------------------- #
def ipd_one_stage(y, *, study, treat, covariates=None, family: str = "gaussian",
                  method: str = "REML", level: float = 0.95,
                  quadrature: int = 15, max_iter: int = 500) -> OneStageResult:
    """Fit one model to every participant, with participants nested in studies.

    Args:
        y: the outcome, one entry per participant.
        study: the study label, one entry per participant.
        treat: the treatment indicator, one entry per participant.
        covariates: optional participant-level covariates — a list of arrays
            (named ``cov1``, …) or a mapping ``{name: array}``.
        family: ``"gaussian"`` (linear mixed model, analytic likelihood) or
            ``"binomial"`` (logistic mixed model, Gauss–Hermite quadrature).
        method: ``"REML"`` (default) or ``"ML"``. Ignored for the binomial
            family, where the likelihood is a true likelihood.
        level: confidence level.
        quadrature: number of Gauss–Hermite nodes (binomial only).
        max_iter: optimiser iteration cap.

    Returns:
        :class:`OneStageResult`.

    Raises:
        ValueError: for mismatched lengths, a missing treatment indicator, or
            an unknown family.

    Example:
        >>> fit = ma.ipd_one_stage(y, study=study, treat=treat)   # doctest: +SKIP
        >>> fit.b[1], fit.tau2, fit.ICC                           # doctest: +SKIP
    """
    y = np.asarray(y, dtype=float).ravel()
    treat = np.asarray(treat, dtype=float).ravel()
    labels = [_label(s) for s in np.asarray(study).ravel()]
    if not (y.size == treat.size == len(labels)):
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
    if fam not in ("gaussian", "binomial"):
        raise ValueError(
            f"unknown family {family!r}; use 'gaussian' or 'binomial'")
    method = str(method).upper()
    if fam == "gaussian" and method not in ("REML", "ML"):
        raise ValueError(f"unknown method {method!r}; use 'REML' or 'ML'")

    names = ["(Intercept)", "treat"] + list(cov_map)
    X = np.column_stack([np.ones(y.size), treat, *cov_map.values()])
    groups = list(dict.fromkeys(labels))
    rows = [np.array([i for i, lab in enumerate(labels) if lab == g])
            for g in groups]
    p = X.shape[1]

    if fam == "gaussian":
        if np.all(y == y[0]):
            raise ValueError("the outcome is constant; there is nothing to fit")
        sums = _study_sums(X, y, rows)
        best = None
        for starting in ((np.log(np.var(y) / 2), np.log(np.var(y) / 2)),
                         (np.log(np.var(y) * 0.9), np.log(np.var(y) * 0.1))):
            res = minimize(
                lambda q: -_gaussian_loglik(q, sums, y.size, p,
                                            method == "REML"),
                np.asarray(starting, dtype=float), method="L-BFGS-B",
                bounds=[(np.log(1e-12), np.log(1e12))] * 2,
                options={"maxiter": int(max_iter), "ftol": 1e-15,
                         "gtol": 1e-10},
            )
            if best is None or res.fun < best.fun:
                best = res
        assert best is not None
        s2 = float(np.exp(best.x[0]))
        tau2 = float(np.exp(best.x[1]))
        xtvx, xtvy, _ytvy, _logdet = _gls(sums, s2, tau2)
        b = np.linalg.solve(xtvx, xtvy)
        vb = np.linalg.inv(xtvx)
        loglik = -float(best.fun)
        sigma2: float | None = s2
        converged = bool(best.success)
        tau2 = 0.0 if tau2 < 1e-10 else tau2
    else:
        if not np.all((y == 0.0) | (y == 1.0)):
            raise ValueError("a binomial outcome must be coded 0/1")
        nodes, weights = _aghq_nodes(quadrature)
        Xs = [X[idx] for idx in rows]
        ys = [y[idx] for idx in rows]

        def neg(params):
            return -_binomial_loglik(params, Xs, ys, nodes, weights)

        start = np.zeros(p + 1)
        start[-1] = np.log(0.5)
        res = minimize(neg, start, method="L-BFGS-B",
                       bounds=[(-50.0, 50.0)] * p + [(np.log(1e-8), np.log(50.0))],
                       options={"maxiter": int(max_iter), "ftol": 1e-15,
                                "gtol": 1e-10})
        beta = np.asarray(res.x[:-1], dtype=float)
        tau2 = float(np.exp(res.x[-1])) ** 2
        loglik = -float(res.fun)
        # The Hessian of the *negative* log-likelihood is the observed
        # information at the optimum, which is what the variances come from.
        hess, _ = _num_hessian(neg, np.asarray(res.x))
        # Only the fixed-effect block is wanted, and it is the block that stays
        # well conditioned when the variance component sits on its boundary.
        fixed_block = hess[:-1, :-1]
        try:
            vb = np.linalg.inv(fixed_block)
            if not np.all(np.diag(vb) > 0):
                raise np.linalg.LinAlgError("negative variance")
        except np.linalg.LinAlgError:      # pragma: no cover - ill-conditioned
            vb = np.linalg.pinv(fixed_block)
        b = beta
        sigma2 = None
        converged = bool(res.success)
        tau2 = 0.0 if tau2 < 1e-8 else tau2

    se = np.sqrt(np.diag(vb))
    crit = float(stats.norm.ppf(0.5 + level / 2.0))
    z = b / se
    return OneStageResult(
        b=[float(v) for v in b],
        se=[float(v) for v in se],
        zval=[float(v) for v in z],
        pval=[float(2.0 * stats.norm.sf(abs(v))) for v in z],
        ci_lb=[float(bi - crit * sei) for bi, sei in zip(b, se)],
        ci_ub=[float(bi + crit * sei) for bi, sei in zip(b, se)],
        vb=np.asarray(vb, dtype=float),
        coefnames=names,
        tau2=float(tau2),
        sigma2=sigma2,
        loglik=float(loglik),
        method=("ML" if fam == "binomial" else method),
        participants=int(y.size),
        studies=len(groups),
        family=fam,
        converged=converged,
        level=float(level),
    )
