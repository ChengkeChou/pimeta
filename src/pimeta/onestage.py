"""One-stage individual-participant-data meta-analysis.

The alternative to :mod:`pimeta.ipd`, and a different model rather than a
refinement of it: every participant enters one model, with participants nested
in studies, so a study's own level is a random intercept and the treatment
effect is estimated across all of them at once.

    .. math::

        g(E[y_{ij}]) = x_{ij}'\\beta + z_{ij}'u_i, \\qquad u_i \\sim N(0, D)

For a continuous outcome with an intercept-only random term the integral over
:math:`u_i` is analytic, so the likelihood is closed form and is maximised over
:math:`(\\sigma^2, \\tau^2)` directly — that is the ``family="gaussian"`` path.
For a binary outcome the integral is approximated by adaptive Gauss–Hermite
quadrature, which is what ``family="binomial"`` does.

Random slopes
-------------
``re_slopes=`` puts coefficients of the fixed design into :math:`z_{ij}`, so
``re_slopes=["treat"]`` is the lme4 model ``(1 + treat | study)``.  That is not
a cosmetic addition: ``V_i = sigma2 (I + Z_i Lambda Z_i')`` no longer inverts by
a rank-one update, so the closed form is gone and the likelihood is evaluated
one study at a time — with Woodbury, at ``q x q``, so a study of 10 000
participants costs what a study of ten costs.  ``sigma2`` is still profiled out
analytically, which is why the model has ``q(q+1)/2`` parameters rather than one
more.

A random slope on a *non-linear* interaction is the same code: give the basis
columns as covariates, put them in ``re_slopes``, and read the curve off with
:meth:`OneStageResult.contrast`, which reports a confidence band for the average
and a prediction band for a new study's curve.

What this model does *not* do: a residual variance that differs by study.  That
is one variance per study against a handful of participants each, and the
profile likelihood answers it with a boundary.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field

import numpy as np
from scipy import stats
from scipy.linalg import cho_solve
from scipy.optimize import minimize

from .checks import attach_diagnostics

__all__ = ["OneStageResult", "ipd_one_stage"]


def _label(value) -> str:
    """A study label that reads ``0`` rather than ``0.0``."""
    try:
        number = float(value)
    except (TypeError, ValueError):
        return str(value)
    return str(int(number)) if number.is_integer() else str(value)


@dataclass
@attach_diagnostics
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
    # Random slopes: the between-study covariance of the random terms and their
    # names. ``D`` is None for the random-intercept model, whose variance is
    # ``tau2``.
    D: np.ndarray | None = field(repr=False, default=None)
    re_names: list[str] = field(default_factory=list)
    struct: str = "UN"

    def contrast(self, L, *, level: float | None = None,
                 prediction: bool = True) -> list[dict]:
        """Estimate linear combinations ``L @ b`` with intervals.

        Args:
            L: a ``(m, p)`` matrix (or a single row) of contrast weights, in
                the order of :attr:`coefnames`.
            level: confidence level; defaults to the fit's.
            prediction: also report the band for a **new study**, which adds
                the between-study covariance ``D``. That is what a non-linear
                treatment effect needs when the interaction is random.

        Returns:
            One dict per row: ``est``, ``se``, ``ci_lb``, ``ci_ub`` and, when
            asked for, ``pred_lb`` / ``pred_ub``.
        """
        M = np.atleast_2d(np.asarray(L, dtype=float))
        if M.shape[1] != len(self.b):
            raise ValueError(
                f"'L' needs one column per coefficient ({len(self.b)}), got "
                f"{M.shape[1]}")
        lvl = self.level if level is None else level
        crit = float(stats.norm.ppf(0.5 + lvl / 2.0))
        est = M @ np.asarray(self.b, dtype=float)
        var = np.einsum("ij,jk,ik->i", M, np.asarray(self.vb), M)
        se = np.sqrt(var)
        rows = [
            {"est": float(est[i]), "se": float(se[i]),
             "ci_lb": float(est[i] - crit * se[i]),
             "ci_ub": float(est[i] + crit * se[i])}
            for i in range(est.size)
        ]
        if prediction and self.D is not None:
            # the random terms are the first coefficients of the design, in the
            # order re_names lists them; a slope passed as a bare array has no
            # coefficient to add, and then there is no prediction band.
            if all(nm in self.coefnames for nm in self.re_names):
                idx = [self.coefnames.index(nm) for nm in self.re_names]
                Mre = M[:, idx]
                extra = np.einsum("ij,jk,ik->i", Mre, np.asarray(self.D), Mre)
                se_p = np.sqrt(var + extra)
                for i, row in enumerate(rows):
                    row["pred_lb"] = float(est[i] - crit * se_p[i])
                    row["pred_ub"] = float(est[i] + crit * se_p[i])
        return rows

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
            "struct": self.struct,
            "re_names": list(self.re_names),
            "D": None if self.D is None else [[float(v) for v in row]
                                              for row in np.asarray(self.D)],
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
# random slopes: a general linear mixed model on the stacked participants
# --------------------------------------------------------------------------- #
def _chol_lambda(theta, q: int, struct: str):
    """``Lambda = D / sigma2`` from unconstrained parameters.

    Parameterised through a Cholesky factor so every candidate is a valid
    covariance matrix: the log-diagonal is free on the real line and the
    off-diagonals are unconstrained correlations-in-waiting.
    """
    if struct == "DIAG":
        return np.diag(np.exp(np.asarray(theta, dtype=float)))
    L = np.zeros((q, q))
    k = 0
    for i in range(q):
        for j in range(i + 1):
            if i == j:
                L[i, j] = np.exp(float(theta[k]))
            else:
                L[i, j] = float(theta[k])
            k += 1
    return L @ L.T


def n_lambda_params(q: int, struct: str) -> int:
    """How many numbers ``_chol_lambda`` needs."""
    return q if struct == "DIAG" else q * (q + 1) // 2


def _random_blocks(X, y, Z, rows):
    """Per-study cross-products against both the fixed and the random design."""
    out = []
    for idx in rows:
        Xi = X[idx]
        Zi = Z[idx]
        yi = y[idx]
        out.append({
            "n": int(idx.size),
            "xtx": Xi.T @ Xi,
            "xty": Xi.T @ yi,
            "yty": float(yi @ yi),
            "xtz": Xi.T @ Zi,
            "zty": Zi.T @ yi,
            "ztz": Zi.T @ Zi,
        })
    return out


def _random_gls(blocks, lam):
    """The GLS pieces under ``V_i = sigma2 (I + Z_i Lambda Z_i')``.

    Woodbury keeps every matrix at ``q x q``:
    ``Omega_i^-1 = I - Z_i M_i^-1 Z_i'`` with ``M_i = Lambda^-1 + Z_i'Z_i``,
    and ``log|Omega_i| = log|Lambda| + log|M_i|`` by Sylvester.  So nothing of
    size ``n_i`` is ever factorised, and a study with 10 000 participants costs
    the same as one with ten.
    """
    q = lam.shape[0]
    chol_lam = np.linalg.cholesky(lam)
    lam_inv = cho_solve((chol_lam, True), np.eye(q))
    logdet_lam = 2.0 * float(np.sum(np.log(np.diag(chol_lam))))
    p = blocks[0]["xtx"].shape[0]
    A = np.zeros((p, p))
    bvec = np.zeros(p)
    c = 0.0
    logdet = 0.0
    for s in blocks:
        M = lam_inv + s["ztz"]
        chol_m = np.linalg.cholesky(0.5 * (M + M.T))
        m_inv = cho_solve((chol_m, True), np.eye(q))
        logdet += logdet_lam + 2.0 * float(np.sum(np.log(np.diag(chol_m))))
        A += s["xtx"] - s["xtz"] @ m_inv @ s["xtz"].T
        bvec += s["xty"] - s["xtz"] @ m_inv @ s["zty"]
        c += s["yty"] - float(s["zty"] @ m_inv @ s["zty"])
    return A, bvec, c, logdet


def _random_loglik(theta, blocks, n_total, p, reml, struct, q):
    """Profile likelihood in ``sigma2``, which is why it is not a parameter."""
    lam = _chol_lambda(theta, q, struct)
    A, bvec, c, logdet = _random_gls(blocks, lam)
    sign, logdet_a = np.linalg.slogdet(A)
    if sign <= 0:
        return -1e10
    beta = np.linalg.solve(A, bvec)
    rss = float(c - beta @ bvec)
    df = n_total - p if reml else n_total
    if not np.isfinite(rss) or rss <= 0.0 or df <= 0:
        return -1e10
    s2 = rss / df
    # -2l = N log s2 + log|Omega| + rss/s2 + (-p log s2 for REML) + ...
    value = -0.5 * (logdet + df * (np.log(s2) + 1.0) + np.log(2.0 * np.pi) * df
                    + (logdet_a if reml else 0.0))
    return float(value) if np.isfinite(value) else -1e10


def _as_slope_columns(values, n: int) -> list[np.ndarray]:
    """A list of random-slope columns from an array or a sequence of them."""
    if isinstance(values, (list, tuple)) and len(values) > 0:
        cols = [np.asarray(v, dtype=float).ravel() for v in values]
        if all(c.size == n for c in cols):
            return cols
    raw = np.asarray(values, dtype=float)
    mat: np.ndarray = raw.reshape(raw.shape[0], -1)
    return [mat[:, j] for j in range(mat.shape[1])]


def _diag_positions(q: int, struct: str) -> list[int]:
    """Where the log-variances sit in the parameter vector."""
    if struct == "DIAG":
        return list(range(q))
    return [i * (i + 1) // 2 + i for i in range(q)]


def _fit_random_slopes(X, y, Z, rows, *, method, max_iter, struct, q):
    """Maximise the profile likelihood of the random-slope model."""
    blocks = _random_blocks(X, y, Z, rows)
    p = X.shape[1]
    n_par = n_lambda_params(q, struct)
    diag_at = _diag_positions(q, struct)

    def objective(theta):
        return -_random_loglik(theta, blocks, y.size, p, method == "REML",
                               struct, q)

    var_y = float(np.var(y)) if np.var(y) > 0 else 1.0
    best = None
    bounds = [(-12.0, 12.0)] * n_par
    for share in (0.5, 0.1, 0.9):
        theta = np.zeros(n_par)
        for j in diag_at:
            theta[j] = 0.5 * np.log(var_y * share)
        res = minimize(objective, theta, method="L-BFGS-B", bounds=bounds,
                       options={"maxiter": int(max_iter), "ftol": 1e-15,
                                "gtol": 1e-10})
        if best is None or res.fun < best.fun:
            best = res
    assert best is not None
    # The correlation ridge of a random-intercept-and-slope model is nearly
    # flat, so a gradient step stops short of the maximum by more than the
    # likelihood's own curvature would suggest.  A derivative-free polish is
    # cheap at q x q and closes that gap.
    polished = minimize(objective, np.asarray(best.x, dtype=float),
                        method="Nelder-Mead",
                        options={"maxiter": 4000, "xatol": 1e-10,
                                 "fatol": 1e-12})
    if polished.fun < best.fun:
        best = polished
    lam = _chol_lambda(best.x, q, struct)
    A, bvec, c, _logdet = _random_gls(blocks, lam)
    beta = np.linalg.solve(A, bvec)
    rss = float(c - beta @ bvec)
    df = y.size - p if method == "REML" else y.size
    s2 = rss / df if df > 0 else float("nan")
    return {
        "beta": beta,
        "vb": s2 * np.linalg.inv(A),
        "sigma2": s2,
        "D": s2 * lam,
        "loglik": -float(best.fun),
        "converged": bool(best.success),
    }


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
                  quadrature: int = 15, max_iter: int = 500,
                  re_slopes=None, struct: str = "UN") -> OneStageResult:
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
        re_slopes: names of fixed-effect columns whose coefficients vary
            randomly across studies, or a ``(participants, q - 1)`` matrix of
            such columns. The random intercept is always included, so
            ``re_slopes=["treat"]`` is the lme4 model ``(1 + treat | study)``.
        struct: ``"UN"`` (default) for a full between-study covariance of the
            random terms, or ``"DIAG"`` for independent random effects.

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

    struct = str(struct).upper()
    if struct not in ("UN", "DIAG"):
        raise ValueError(f"unknown struct {struct!r}; use 'UN' or 'DIAG'")
    slope_cols: list[np.ndarray] = []
    slope_names: list[str] = []
    if re_slopes is not None:
        if fam == "binomial":
            raise ValueError(
                "random slopes are implemented for family='gaussian'; the "
                "logistic likelihood would need quadrature over the whole "
                "random vector, one dimension per slope")
        if isinstance(re_slopes, str):
            re_slopes = [re_slopes]
        if isinstance(re_slopes, Mapping):
            slope_names = [str(k) for k in re_slopes]
            slope_cols = [np.asarray(v, dtype=float).ravel()
                          for v in re_slopes.values()]
        elif (isinstance(re_slopes, (list, tuple)) and len(re_slopes) > 0
              and all(isinstance(v, str) for v in re_slopes)):
            for name in re_slopes:
                if name not in names:
                    raise ValueError(
                        f"unknown column {name!r} for re_slopes; the "
                        f"fixed-effect design has {names}")
                slope_names.append(str(name))
                slope_cols.append(X[:, names.index(name)])
        else:
            slope_cols = _as_slope_columns(re_slopes, y.size)
            slope_names = [f"re{j + 1}" for j in range(len(slope_cols))]
        for c in slope_cols:
            if c.size != y.size:
                raise ValueError(
                    "each random-slope column needs one value per participant")
    re_names = ["(Intercept)"] + slope_names

    if fam == "gaussian":
        if np.all(y == y[0]):
            raise ValueError("the outcome is constant; there is nothing to fit")
        if slope_cols:
            Z = np.column_stack([np.ones(y.size), *slope_cols])
            q = Z.shape[1]
            if len(groups) <= q:
                raise ValueError(
                    f"{len(groups)} studies cannot identify {q} random "
                    "effects; use fewer slopes or struct='DIAG'")
            if np.linalg.matrix_rank(Z) < q:
                raise ValueError(
                    "the random-effect design is collinear, so its covariance "
                    "is not identified; a slope that repeats the intercept "
                    "(or a constant covariate) is already counted by the "
                    "random intercept")
            out = _fit_random_slopes(X, y, Z, rows, method=method,
                                     max_iter=max_iter, struct=struct, q=q)
            b = np.asarray(out["beta"], dtype=float)
            vb = np.asarray(out["vb"], dtype=float)
            loglik = float(out["loglik"])
            sigma2_r = float(out["sigma2"])
            converged = bool(out["converged"])
            D = np.asarray(out["D"], dtype=float)
            tau2 = float(D[0, 0])
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
                vb=vb,
                coefnames=names,
                tau2=tau2,
                sigma2=sigma2_r,
                loglik=loglik,
                method=method,
                participants=int(y.size),
                studies=len(groups),
                family=fam,
                converged=converged,
                level=float(level),
                D=D,
                re_names=re_names,
                struct=struct,
            )
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
