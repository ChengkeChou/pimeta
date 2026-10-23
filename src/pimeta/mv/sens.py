"""How much does the answer depend on the assumed within-study correlation?

The within-study correlation among dependent effect sizes cannot be estimated
from a meta-analysis: each study contributes a handful of estimates, so the
correlation is a nuisance parameter that has to be *assumed*.  That is not a
defect — it is a fact about the data — but it does create an obligation: the
report has to show what the assumption is worth.

``rho_sensitivity`` refits the model over a grid of correlations and returns
the pooled estimates with their intervals side by side.  The interesting output
is not any single row; it is whether the conclusion survives the whole grid.

This mirrors what the Cochrane Handbook asks for (a sensitivity analysis over
``rho``, typically 0, 0.4, 0.8) but does it with the actual model rather than
with a single alternative fit.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .vcov import vcalc

__all__ = ["RhoSensitivityResult", "rho_sensitivity"]


@dataclass
class RhoSensitivityResult:
    """Pooled estimates along a grid of within-study correlations.

    Attributes:
        rho: the grid that was fitted.
        names: coefficient labels.
        estimate: ``(G, p)`` pooled estimates.
        se: ``(G, p)`` standard errors.
        ci_lb, ci_ub: ``(G, p)`` interval bounds.
        tau2: ``(G, q)`` variance components (``q`` depends on ``struct``).
        fits: the individual model fits, in grid order.
    """

    rho: np.ndarray = field(repr=False)
    names: list[str]
    estimate: np.ndarray = field(repr=False)
    se: np.ndarray = field(repr=False)
    ci_lb: np.ndarray = field(repr=False)
    ci_ub: np.ndarray = field(repr=False)
    tau2: np.ndarray = field(repr=False)
    fits: list = field(repr=False, default_factory=list)

    def to_table(self) -> list[dict]:
        """One row per grid point, with one column pair per coefficient."""
        rows = []
        for g, rho in enumerate(self.rho):
            row: dict = {"rho": float(rho)}
            for j, name in enumerate(self.names):
                row[name] = float(self.estimate[g, j])
                row[f"{name}_se"] = float(self.se[g, j])
                row[f"{name}_ci_lb"] = float(self.ci_lb[g, j])
                row[f"{name}_ci_ub"] = float(self.ci_ub[g, j])
            rows.append(row)
        return rows

    def to_dict(self) -> dict:
        return {
            "rho": self.rho.tolist(),
            "names": list(self.names),
            "estimate": self.estimate.tolist(),
            "se": self.se.tolist(),
            "ci_lb": self.ci_lb.tolist(),
            "ci_ub": self.ci_ub.tolist(),
            "tau2": self.tau2.tolist(),
        }

    def range_of(self, name: str) -> dict:
        """Spread of one coefficient's estimate and interval over the grid."""
        j = self.names.index(name)
        est = self.estimate[:, j]
        return {
            "coefficient": name,
            "estimate_min": float(est.min()),
            "estimate_max": float(est.max()),
            "estimate_spread": float(est.max() - est.min()),
            "ci_lb_min": float(self.ci_lb[:, j].min()),
            "ci_ub_max": float(self.ci_ub[:, j].max()),
            "sign_stable": bool((est > 0).all() or (est < 0).all()),
        }

    def __str__(self) -> str:
        lines = ["within-study correlation sensitivity"]
        for row in self.to_table():
            cells = "  ".join(
                f"{row[n]:+.4f} [{row[n + '_ci_lb']:+.4f}, "
                f"{row[n + '_ci_ub']:+.4f}]"
                for n in self.names
            )
            lines.append(f"  rho={row['rho']:.2f}  {cells}")
        return "\n".join(lines)


def rho_sensitivity(yi, vi, *, study, outcome, mods=None, struct="UN",
                    method="REML", rho_grid=(0.0, 0.25, 0.5, 0.75),
                    level=95.0, test="z"):
    """Refit a multivariate model along a grid of within-study correlations.

    Args:
        yi: effect estimates, length ``k``.
        vi: sampling variances, length ``k``.
        study: study identifier; defines the blocks of ``V``.
        outcome: outcome identifier; the inner factor of the random term.
        mods: fixed-effect design — a design matrix, a formula string such as
            ``"~ outcome - 1"``, or ``None`` for an intercept-only model.
        struct: variance structure passed to ``rma_mv``; ``"UN"`` estimates a
            free between-outcome correlation, ``"CS"`` one common value.
        method: ``"REML"`` (default) or ``"ML"``.
        rho_grid: the assumed within-study correlations to fit.
        level: confidence level in percent.
        test: ``"z"`` (default), ``"t"``, ``"knha"`` or ``"adhoc"``.

    Returns:
        :class:`RhoSensitivityResult`.

    Raises:
        ValueError: an empty grid, mismatched lengths, or a design that cannot
            be resolved.
    """
    from ..data_compat import design_matrix
    from ..rma_mv import rma_mv

    yi = np.asarray(yi, dtype=float).ravel()
    vi = np.asarray(vi, dtype=float).ravel()
    study = np.asarray(study).ravel()
    outcome = np.asarray(outcome).ravel()
    k = yi.size
    if not (vi.size == study.size == outcome.size == k):
        raise ValueError("'yi', 'vi', 'study' and 'outcome' must be the same "
                         "length")
    grid = np.asarray(rho_grid, dtype=float).ravel()
    if grid.size == 0:
        raise ValueError("'rho_grid' must not be empty")
    if np.any((grid < -1.0) | (grid > 1.0)):
        raise ValueError("'rho_grid' entries must lie in [-1, 1]")

    data = {"yi": yi, "study": study, "outcome": outcome}
    if mods is None:
        X, xnames = None, ["intercept"]
        intercept = True
    elif isinstance(mods, str):
        X, xnames = design_matrix(mods, data)
        intercept = False
        if X.shape[1] == 0:
            raise ValueError("'mods' resolves to a model with no columns")
    else:
        X = np.asarray(mods, dtype=float)
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        xnames = [f"x{j + 1}" for j in range(X.shape[1])]
        intercept = False
    if X is not None and X.shape[0] != k:
        raise ValueError("'mods' must have one row per effect size")

    estimates, errors, lows, highs, tau2s, fits = [], [], [], [], [], []
    for rho in grid:
        V = vcalc(vi, cluster=study, obs=outcome, rho=float(rho))
        fit = rma_mv(yi, V=V, mods=X, data=data, random="~ outcome | study",
                     struct=struct, method=method, level=level, test=test,
                     intercept=intercept)
        estimates.append(np.asarray(fit.beta, dtype=float).ravel())
        errors.append(np.asarray(fit.se, dtype=float).ravel())
        lows.append(np.asarray(fit.ci_lb, dtype=float).ravel())
        highs.append(np.asarray(fit.ci_ub, dtype=float).ravel())
        tau2s.append(np.asarray(fit.tau2 if fit.tau2 is not None
                                else fit.sigma2, dtype=float).ravel())
        fits.append(fit)

    names = list(xnames)
    return RhoSensitivityResult(
        rho=grid,
        names=names,
        estimate=np.vstack(estimates),
        se=np.vstack(errors),
        ci_lb=np.vstack(lows),
        ci_ub=np.vstack(highs),
        tau2=np.vstack(tau2s),
        fits=fits,
    )
