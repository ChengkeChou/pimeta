"""One-stage individual-participant network meta-analysis.

The model every IPD-NMA is a special case of:

.. math::

    g(E[y_{ij}]) = \\mu_{s(i)} + \\delta_{s(i),t_{ij}}
                    + x_{ij}'\\gamma + x_{ij}'\\delta^{(x)}_{t_{ij}},
    \\qquad \\delta_s \\sim N(d, \\Sigma)

with one intercept per study (so the network's comparisons are *within* study),
the treatment effects :math:`\\delta_{s}` a vector over the non-reference
treatments, and the interaction between a covariate and treatment shared across
studies by default.  ``d`` is the consistency-model treatment effect; it is
exactly the basic parameter of :class:`~pimeta.nma.design.TreatmentNetwork`, so
the network parameterisation is not re-derived here — it is imported, and every
consistency statement is the one the algebraic tests already cover.

For a Gaussian outcome the likelihood is exact and the model is a linear mixed
model, so it is maximised directly (with :math:`\\sigma^2` profiled out and the
between-study covariance in a Cholesky parameterisation, as in
:mod:`pimeta.onestage`).  That is not a simplification: it is the case in which
the answer can be checked against ``lme4`` to machine precision, which is the
only way to know the network parameterisation survived contact with a
likelihood.

Scope, stated rather than implied: the Gaussian family only, a shared (not
study-specific) interaction, and random effects on the treatment contrast.  A
binomial outcome needs quadrature or a Laplace approximation over a random
vector and is not attempted here; the point of this module is the network layer
underneath, and shipping an unverified logistic version would obscure it.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
from scipy import stats
from scipy.optimize import minimize

from ..checks import attach_diagnostics
from ..levels import level_to_percent
from .design import TreatmentNetwork

__all__ = ["IpdNmaResult", "ipd_nma"]


@dataclass
@attach_diagnostics
class IpdNmaResult:
    """A one-stage IPD network meta-analysis.

    Attributes:
        treatments: all treatments in the network, reference first.
        reference: the treatment every effect is relative to.
        d: treatment effects against the reference, on the linear-predictor
            scale, in the order of :attr:`estimable`.
        se: their standard errors.
        vcov: their covariance matrix.
        Sigma: the between-study covariance of the treatment effects.
        tau2: its diagonal.
        gamma: covariate main effects (shared across studies).
        delta: treatment-by-covariate interaction (shared), or ``None``.
        k, participants: studies and rows used.
        network: the :class:`~pimeta.nma.design.TreatmentNetwork` that
            generated the parameterisation.
        sigma2: residual variance (Gaussian).
        loglik: the maximised log-likelihood.
        method, level, struct, interaction: settings.
        fit: the raw optimiser output, for diagnosing a boundary.
    """

    treatments: list[str]
    reference: str
    estimable: list[str]
    d: list[float]
    se: list[float]
    vcov: np.ndarray = field(repr=False)
    Sigma: np.ndarray = field(repr=False)
    tau2: list[float]
    gamma: list[float]
    delta: list[float] | None
    k: int
    participants: int
    network: TreatmentNetwork | None = field(repr=False, default=None)
    sigma2: float = float("nan")
    loglik: float = float("nan")
    method: str = "REML"
    level: float = 95.0
    struct: str = "UN"
    interaction: list[str] | None = None
    fit: Any = field(repr=False, default=None)

    def _crit(self) -> float:
        return float(stats.norm.ppf(0.5 + self.level / 200.0))

    def coef_table(self) -> list[dict]:
        crit = self._crit()
        rows = []
        for name, est, se in zip(self.estimable, self.d, self.se):
            rows.append({
                "treatment": name,
                "vs": self.reference,
                "estimate": float(est),
                "se": float(se),
                "ci_lb": float(est - crit * se),
                "ci_ub": float(est + crit * se),
                "zval": float(est / se) if se > 0 else float("nan"),
                "pval": (float(2.0 * stats.norm.sf(abs(est / se)))
                         if se > 0 else float("nan")),
            })
        return rows

    def contrast(self, a: str, b: str, *, level: float | None = None) -> dict:
        """One pairwise contrast ``a - b`` from the fitted network.

        Consistency is not assumed here: the contrast is an exact linear
        combination of the estimated basic parameters, so it equals what the
        consistency model implies and inherits its variance.
        """
        a, b = str(a), str(b)
        for name in (a, b):
            if name != self.reference and name not in self.estimable:
                raise ValueError(f"unknown treatment {name!r}")
        w = np.zeros(len(self.estimable))
        if a != self.reference:
            w[self.estimable.index(a)] += 1.0
        if b != self.reference:
            w[self.estimable.index(b)] -= 1.0
        est = float(w @ np.asarray(self.d, dtype=float))
        var = float(w @ np.asarray(self.vcov, dtype=float) @ w)
        lvl = self.level if level is None else level_to_percent(level)
        crit = float(stats.norm.ppf(0.5 + lvl / 200.0))
        return {
            "comparison": (a, b),
            "estimate": est,
            "se": float(np.sqrt(max(var, 0.0))),
            "ci_lb": est - crit * float(np.sqrt(max(var, 0.0))),
            "ci_ub": est + crit * float(np.sqrt(max(var, 0.0))),
        }

    def league_table(self, *, level: float | None = None) -> list[dict]:
        """Every pairwise contrast, one row each."""
        rows = []
        for i, a in enumerate(self.treatments):
            for b in self.treatments[i + 1:]:
                out = self.contrast(a, b, level=level)
                rows.append({
                    "treat1": a, "treat2": b,
                    "estimate": out["estimate"], "se": out["se"],
                    "ci_lb": out["ci_lb"], "ci_ub": out["ci_ub"],
                })
        return rows

    def to_dict(self) -> dict:
        return {
            "treatments": list(self.treatments),
            "reference": self.reference,
            "estimable": list(self.estimable),
            "d": list(self.d),
            "se": list(self.se),
            "vcov": np.asarray(self.vcov).tolist(),
            "Sigma": np.asarray(self.Sigma).tolist(),
            "tau2": list(self.tau2),
            "gamma": list(self.gamma),
            "delta": None if self.delta is None else list(self.delta),
            "k": self.k,
            "participants": self.participants,
            "sigma2": self.sigma2,
            "loglik": self.loglik,
            "method": self.method,
            "level": self.level,
            "struct": self.struct,
            "interaction": (None if self.interaction is None
                            else list(self.interaction)),
            "df_inconsistency": (self.network.df_inconsistency
                                 if self.network is not None else None),
            "coef": self.coef_table(),
        }

    def __str__(self) -> str:
        lines = [
            "one-stage IPD network meta-analysis",
            f"  {self.k} studies, {self.participants} participants, "
            f"{len(self.treatments)} treatments, reference {self.reference!r}",
        ]
        if self.network is not None:
            lines.append(
                f"  {self.network.n_comparisons} comparisons, "
                f"{self.network.df_inconsistency} degree(s) of freedom for "
                "inconsistency")
        lines.append("")
        lines.append(f"  {'treatment':<14}{'vs':<8}{'estimate':>10}{'se':>9}")
        for row in self.coef_table():
            lines.append(f"  {row['treatment']:<14}{row['vs']:<8}"
                         f"{row['estimate']:>10.4f}{row['se']:>9.4f}")
        lines.append("")
        lines.append("  between-study variance: " + ", ".join(
            f"{v:.4f}" for v in self.tau2))
        return "\n".join(lines)


def _network_from(study, treat, reference):
    """Build the comparison network from participant-level arm assignments."""
    labels = [str(t) for t in np.asarray(treat).ravel()]
    studies = [str(s) for s in np.asarray(study).ravel()]
    order: list[str] = []
    for name in labels:
        if name not in order:
            order.append(name)
    if reference is not None:
        ref = str(reference)
        if ref not in order:
            raise ValueError(f"reference treatment {ref!r} is not in the data")
        order = [ref] + [t for t in order if t != ref]
    else:
        ref = order[0]
    comparisons = []
    for s in dict.fromkeys(studies):
        rows = [i for i, lab in enumerate(studies) if lab == s]
        arms = list(dict.fromkeys(labels[i] for i in rows))
        if len(arms) < 2:
            continue
        base = ref if ref in arms else arms[0]
        for arm in arms:
            if arm != base:
                comparisons.append((s, arm, base))
    return TreatmentNetwork(treatments=order, comparisons=comparisons)


def ipd_nma(y, *, study, treat, reference=None, covariates=None,
            interaction=None, family: str = "gaussian",
            method: str = "REML", level: float = 0.95, struct: str = "UN",
            max_iter: int = 500) -> IpdNmaResult:
    """One-stage network meta-analysis of individual participant data.

    Args:
        y: the outcome, one entry per participant.
        study: study label per participant.
        treat: treatment arm label per participant.
        reference: the treatment that defines the basic parameters; the first
            one seen if omitted.
        covariates: optional mapping ``{name: array}`` of participant-level
            covariates entering as shared main effects.
        interaction: names among ``covariates`` whose effect is allowed to
            differ by treatment.  The interaction is shared across studies,
            which is the usual assumption and the one that keeps it estimable
            with a handful of studies.
        family: ``"gaussian"`` only; see the module docstring.
        method: ``"REML"`` or ``"ML"``.
        level: confidence level.
        struct: ``"UN"`` for an unstructured between-study covariance of the
            treatment effects, or ``"DIAG"`` for independent ones.
        max_iter: optimiser iteration cap.

    Returns:
        :class:`IpdNmaResult`.

    Raises:
        ValueError: mismatched lengths, an unknown family, a disconnected
            network, a reference not in the data, or a covariate requested for
            interaction that was not supplied.

    Example:
        >>> fit = ma.ipd_nma(y, study=study, treat=treat)   # doctest: +SKIP
        >>> fit.contrast("B", "A")                          # doctest: +SKIP
    """
    from ..onestage import _random_blocks, _random_gls, _random_loglik

    fam = str(family).lower()
    if fam != "gaussian":
        raise ValueError(
            "only family='gaussian' is implemented: a binomial one-stage "
            "network model needs quadrature or a Laplace approximation over "
            "the whole random vector, and an unverified version would be "
            "worse than the gap")
    settings = str(method).upper()
    if settings not in ("REML", "ML"):
        raise ValueError(f"unknown method {method!r}; use 'REML' or 'ML'")
    struct = str(struct).upper()
    if struct not in ("UN", "DIAG"):
        raise ValueError(f"unknown struct {struct!r}; use 'UN' or 'DIAG'")

    y = np.asarray(y, dtype=float).ravel()
    studies = [str(s) for s in np.asarray(study).ravel()]
    if not (len(studies) == y.size):
        raise ValueError("'y' and 'study' must be the same length")
    network = _network_from(study, treat, reference)
    if not network.is_connected():
        raise ValueError(
            "the network is disconnected: "
            f"{len(network.components())} separate components "
            f"{network.components()}; a network meta-analysis needs one")
    ref = network.treatments[0]
    estimable = network.treatments[1:]
    q = len(estimable)

    labels = [str(t) for t in np.asarray(treat).ravel()]
    pos = {t: j for j, t in enumerate(estimable)}
    A = np.zeros((y.size, q))
    for i, name in enumerate(labels):
        if name in pos:
            A[i, pos[name]] = 1.0

    cov_map: dict[str, np.ndarray] = {}
    if covariates is not None:
        cov_map = {str(k): np.asarray(v, dtype=float).ravel()
                   for k, v in covariates.items()}
        for k, v in cov_map.items():
            if v.size != y.size:
                raise ValueError(
                    f"covariate {k!r} needs one value per participant")
    interaction = [] if interaction is None else [str(v) for v in interaction]
    for name in interaction:
        if name not in cov_map:
            raise ValueError(
                f"interaction term {name!r} is not among the covariates "
                f"{list(cov_map)}")

    groups = list(dict.fromkeys(studies))
    S = np.zeros((y.size, len(groups)))
    for i, name in enumerate(studies):
        S[i, groups.index(name)] = 1.0

    # covariates are centred within study, so the treatment parameters keep
    # their within-study interpretation and the study intercepts absorb the
    # between-study differences in the covariate distribution
    centred = {}
    for name, values in cov_map.items():
        c = values.astype(float).copy()
        for g in groups:
            idx = [i for i, lab in enumerate(studies) if lab == g]
            c[idx] -= float(np.mean(c[idx]))
        centred[name] = c

    blocks = [S, A]
    for name in cov_map:
        blocks.append(centred[name].reshape(-1, 1))
    for name in interaction:
        blocks.append(A * centred[name].reshape(-1, 1))
    X = np.column_stack(blocks)

    if np.linalg.matrix_rank(X) < X.shape[1]:
        raise ValueError(
            "the fixed-effect design is rank deficient, so the network is not "
            "estimable as specified; check that every treatment is compared "
            "with the reference inside at least one study")

    rows = [np.array([i for i, lab in enumerate(studies) if lab == g])
            for g in groups]
    Z = A
    dat_blocks = _random_blocks(X, y, Z, rows)
    p = X.shape[1]
    n_par = q if struct == "DIAG" else q * (q + 1) // 2

    def objective(theta):
        return -_random_loglik(theta, dat_blocks, y.size, p,
                               settings == "REML", struct, q)

    var_y = float(np.var(y)) if float(np.var(y)) > 0 else 1.0
    diag_at = (list(range(q)) if struct == "DIAG"
               else [i * (i + 1) // 2 + i for i in range(q)])
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
    polished = minimize(objective, np.asarray(best.x, dtype=float),
                        method="Nelder-Mead",
                        options={"maxiter": 4000, "xatol": 1e-10,
                                 "fatol": 1e-12})
    if polished.fun < best.fun:
        best = polished

    from ..onestage import _chol_lambda

    lam = _chol_lambda(best.x, q, struct)
    Amat, bvec, ctot, _logdet = _random_gls(dat_blocks, lam)
    beta = np.linalg.solve(Amat, bvec)
    rss = float(ctot - beta @ bvec)
    df = y.size - p if settings == "REML" else y.size
    sigma2 = rss / df if df > 0 else float("nan")
    vcov_full = sigma2 * np.linalg.inv(Amat)
    Sigma = sigma2 * lam

    off = len(groups)
    d = beta[off:off + q]
    vb = vcov_full[off:off + q, off:off + q]
    gamma = beta[off + q:off + q + len(cov_map)]
    delta = beta[off + q + len(cov_map):]

    return IpdNmaResult(
        treatments=list(network.treatments), reference=ref,
        estimable=estimable, d=[float(v) for v in d],
        se=[float(v) for v in np.sqrt(np.diag(vb))],
        vcov=np.asarray(vb, dtype=float), Sigma=np.asarray(Sigma, float),
        tau2=[float(v) for v in np.diag(Sigma)],
        gamma=[float(v) for v in gamma],
        delta=None if delta.size == 0 else [float(v) for v in delta],
        k=len(groups), participants=int(y.size), network=network,
        sigma2=float(sigma2), loglik=-float(best.fun), method=settings,
        level=level_to_percent(level), struct=struct,
        interaction=list(interaction) or None, fit=best,
    )
