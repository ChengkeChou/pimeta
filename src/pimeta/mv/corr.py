"""Between-outcome correlations of a fitted multivariate model.

A multivariate random-effects model estimates a ``J x J`` covariance matrix
``G`` for the outcomes.  ``G`` itself is hard to compare across fits, because
its diagonal is on the effect-size scale.  The correlation matrix
``R = D^-1 G D^-1`` (``D = diag(sqrt(diag G))``) is unitless and is what a
reader actually wants: *do the outcomes move together across studies, and how
strongly?*

The numbers here are exactly metafor's ``cov2cor(fit$G)``.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

__all__ = ["TauCorResult", "tau_cor"]


@dataclass
class TauCorResult:
    """Between-outcome covariance and correlation of the random effects.

    Attributes:
        names: outcome labels, in the order of the model's inner factor.
        G: the ``J x J`` random-effects covariance matrix.
        R: the corresponding correlation matrix.
        tau2: ``diag(G)``.
        rho: the off-diagonal correlations, lower triangle, row by row — the
            order metafor reports in ``$rho``.
    """

    names: list[str]
    G: np.ndarray = field(repr=False)
    R: np.ndarray = field(repr=False)
    tau2: np.ndarray = field(repr=False)
    rho: np.ndarray = field(repr=False)

    @property
    def J(self) -> int:
        return len(self.names)

    def coef_table(self) -> list[dict]:
        """One row per outcome and one per outcome pair."""
        rows = [
            {"outcome": self.names[i], "tau2": float(self.tau2[i]),
             "sqrt_tau2": float(np.sqrt(max(self.tau2[i], 0.0)))}
            for i in range(self.J)
        ]
        rows += [
            {"outcome1": self.names[i], "outcome2": self.names[j],
             "rho": float(self.R[i, j])}
            for i in range(self.J) for j in range(i + 1, self.J)
        ]
        return rows

    def to_dict(self) -> dict:
        return {
            "names": list(self.names),
            "G": self.G.tolist(),
            "R": self.R.tolist(),
            "tau2": self.tau2.tolist(),
            "rho": self.rho.tolist(),
        }

    def __str__(self) -> str:
        lines = ["between-outcome correlations"]
        lines.append("  " + " " * 12 + "".join(f"{n:>10}" for n in self.names))
        for i, name in enumerate(self.names):
            cells = "".join(
                "         -" if i == j else f"{self.R[i, j]:>10.4f}"
                for j in range(self.J)
            )
            lines.append(f"  {name:<12}{cells}")
        return "\n".join(lines)


def tau_cor(fit):
    """Between-outcome correlation matrix of a fitted ``rma_mv`` model.

    Args:
        fit: :class:`~pimeta.rma_mv.RmaMvResult` from a model with a
            multivariate random term, e.g.
            ``random="~ outcome | study", struct="UN"``.

    Returns:
        :class:`TauCorResult`.

    Raises:
        ValueError: the model has no multivariate random term, so there is no
            between-outcome covariance to report.
    """
    G, names = fit.g_matrix()
    if G is None:
        raise ValueError(
            "the model has no multivariate random term: fit it with "
            "random='~ outcome | study' (and struct='UN' for a free "
            "correlation matrix) to estimate between-outcome dependence")
    G = np.asarray(G, dtype=float)
    J = G.shape[0]
    if not names or len(names) != J:
        names = [f"y{i + 1}" for i in range(J)]
    d = np.sqrt(np.clip(np.diag(G), 0.0, None))
    R = np.eye(J)
    for i in range(J):
        for j in range(J):
            if i == j:
                continue
            denom = d[i] * d[j]
            R[i, j] = 0.0 if denom <= 0.0 else float(np.clip(G[i, j] / denom,
                                                            -1.0, 1.0))
    rho = np.array([R[i, j] for i in range(1, J) for j in range(i)])
    return TauCorResult(names=names, G=G, R=R, tau2=np.diag(G).copy(),
                        rho=rho)
