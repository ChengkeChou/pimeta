"""Joint prediction region for a multivariate meta-analysis.

A prediction interval for one outcome asks where a *new study's* true effect
will fall.  With several outcomes the same question is a region, and the
temptation is to report one interval per outcome and call it joint.  It is not:
those intervals ignore the correlation between the outcomes, so a point can sit
inside every one of them and outside the region they are supposed to describe.

The region
----------
For a fit with pooled effects :math:`\\hat\\mu` and covariance
:math:`V`, and a between-study covariance :math:`\\Sigma`, a new study's true
effects are distributed as

.. math::

    \\theta_{\\text{new}} \\sim N(\\mu, \\Sigma), \\qquad
    \\hat\\mu \\sim N(\\mu, V)

so their difference is centred at zero with covariance :math:`V + \\Sigma` and

.. math::

    (\\theta - \\hat\\mu)' (V + \\Sigma)^{-1} (\\theta - \\hat\\mu)
    \\le \\chi^2_{1-\\alpha}(J)

is a :math:`1-\\alpha` region for the new study's **true** effects.  Passing the
new study's sampling covariance as ``observation=`` adds it, which gives the
region for its **observed** estimates instead — a wider region, and a different
question.

Two properties are worth stating rather than discovering:

- the region is an ellipsoid, so its projection onto one outcome is
  :math:`\\hat\\mu_j \\pm \\sqrt{\\chi^2_{1-\\alpha}(J)\\,(V+\\Sigma)_{jj}}`,
  which is **wider** than the marginal prediction interval
  :math:`\\pm z_{1-\\alpha/2}`.  Projecting a joint region is conservative by
  construction; the margins are reported as exactly that.
- the chi-square reference is asymptotic in the number of studies.  What it
  covers at a given ``k`` was measured by ``tools/coverage_predict_region.py``
  rather than assumed, because metafor does not implement a joint prediction
  region for ``rma.mv`` and there is nothing to compare against.  Two
  correlated outcomes, 300 replications, nominal 0.95:

  ======  ==========  ==========  ==========
  k       chi-square  Hotelling   two z PIs
  ======  ==========  ==========  ==========
  10      0.767       0.827       0.777
  20      0.917       0.930       0.897
  40      0.937       0.947       0.910
  80      0.940       0.940       0.897
  ======  ==========  ==========  ==========

  So the region is conservative per outcome and still under-covers jointly at
  small ``k``, while the per-outcome intervals it replaces never reach nominal
  at any ``k`` -- they ignore the correlation by construction.  A fit with 20
  or fewer residual degrees of freedom warns and quotes these numbers, and
  ``test="t"`` swaps in the Hotelling reference

  .. math::

      \\frac{J(m-1)}{m-J}\\,F_{1-\\alpha}(J, m-J), \\qquad m = k - p,

  which with :math:`J = 1` is :math:`t^2_{m-1}` -- the univariate correction,
  one degree of freedom dearer than the :math:`t_m` a univariate prediction
  interval uses.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field

import numpy as np
from scipy import stats

from ..levels import level_to_alpha, level_to_percent
from ..rma_mv import RmaMvResult

__all__ = ["PredictionRegion", "predict_region"]


@dataclass
class PredictionRegion:
    """An ellipsoidal prediction region for a new study's effect vector."""

    names: list[str]
    center: np.ndarray = field(repr=False)
    cov: np.ndarray = field(repr=False)
    crit: float
    level: float
    k: int
    for_observation: bool = False
    test: str = "z"
    df: list[float] | None = None

    @property
    def dim(self) -> int:
        return int(self.center.size)

    def contains(self, point) -> bool:
        """Whether ``point`` lies inside the region."""
        delta = np.asarray(point, dtype=float).ravel() - self.center
        if delta.size != self.dim:
            raise ValueError(
                f"expected {self.dim} values, got {delta.size}")
        return bool(delta @ np.linalg.solve(self.cov, delta) <= self.crit)

    def distance(self, point) -> float:
        """The squared Mahalanobis distance; compare it with ``crit``."""
        delta = np.asarray(point, dtype=float).ravel() - self.center
        return float(delta @ np.linalg.solve(self.cov, delta))

    def margin(self, i: int) -> tuple[float, float]:
        """The conservative interval the region implies for outcome ``i``."""
        half = float(np.sqrt(self.crit * self.cov[i, i]))
        return float(self.center[i] - half), float(self.center[i] + half)

    def marginal(self) -> list[dict]:
        """One row per outcome: the projected interval and where it comes from."""
        rows = []
        for i, name in enumerate(self.names):
            lo, hi = self.margin(i)
            rows.append({
                "outcome": name,
                "estimate": float(self.center[i]),
                "se": float(np.sqrt(self.cov[i, i])),
                "pred_lb": lo,
                "pred_ub": hi,
            })
        return rows

    def _reference(self) -> str:
        if self.test == "t":
            df = self.df or []
            return f"Hotelling F({self.dim}, {df[1]:g})" if len(df) > 1 \
                else "Hotelling F"
        return f"chi-square({self.dim})"

    def to_dict(self) -> dict:
        return {
            "names": list(self.names),
            "center": [float(v) for v in self.center],
            "cov": [[float(v) for v in row] for row in self.cov],
            "crit": float(self.crit),
            "level": float(self.level),
            "df": self.dim,
            "k": self.k,
            "for_observation": bool(self.for_observation),
            "test": self.test,
            "ref_df": list(self.df) if self.df else None,
        }

    def __str__(self) -> str:
        what = ("a new study's observed estimates" if self.for_observation
                else "a new study's true effects")
        lines = [
            f"{level_to_percent(self.level):g}% joint prediction region for "
            f"{what}",
            f"{self._reference()} critical value {self.crit:.4f}, "
            f"from {self.k} studies",
            "",
            f"{'outcome':>12}{'estimate':>10}{'pred_lb':>10}{'pred_ub':>10}",
        ]
        for row in self.marginal():
            lines.append(
                f"{row['outcome']:>12}{row['estimate']:>10.4f}"
                f"{row['pred_lb']:>10.4f}{row['pred_ub']:>10.4f}")
        lines.append("")
        lines.append("the two bounds are the projection of the ellipsoid and "
                     "are therefore wider than per-outcome intervals")
        return "\n".join(lines)


def _univariate_sigma(result):
    """``(Sigma, names)`` for a one-coefficient model, or ``(None, [])``.

    A single random intercept has ``J = 1``, so ``g_matrix()`` skips it; the
    between-study variance is then the sum of the variance components, which is
    what an ordinary univariate prediction interval adds.  Keeping this path
    makes the region reduce exactly to that interval when ``J = 1`` -- the one
    algebra check available, since metafor has no joint region to compare with.
    """
    p = int(np.asarray(result.beta).size)
    if p != 1:
        return None, []
    comps = result.sigma2 or []
    if not comps:
        return None, []
    tau2 = float(np.sum(np.asarray(comps, dtype=float)))
    return np.array([[tau2]], dtype=float), []


def predict_region(result, *, level=None, observation=None, test="z",
                   df=None) -> PredictionRegion:
    """Joint prediction region for a new study.

    Args:
        result: a :class:`~pimeta.RmaMvResult` with a multivariate random term
            (``random="~ outcome | study"`` with ``struct="UN"`` or similar).
        level: confidence level, accepted as ``95``, ``0.95`` or ``0.05``;
            defaults to the level the model was fitted with.
        observation: the new study's sampling covariance (``J x J``), when the
            question is where its *observed* estimates would fall rather than
            where its true effects would.
        test: ``"z"`` (default) for the asymptotic :math:`\\chi^2_J`
            reference, or ``"t"`` for the Hotelling :math:`F` small-sample one.
        df: denominator degrees of freedom for ``test="t"``; defaults to
            ``k - p``.  Ignored by ``test="z"``.

    Returns:
        :class:`PredictionRegion`.

    Raises:
        ValueError: when the fit has no between-study covariance matrix, when
            ``observation`` has the wrong shape, when that matrix is not
            positive definite, or when ``test="t"`` is asked for with too few
            degrees of freedom to form the reference.

    Warns:
        RuntimeWarning: with ``test="z"`` and 20 or fewer residual degrees of
            freedom, quoting the measured coverage at small ``k``.

    Example:
        >>> r = ma.predict_region(fit)                 # doctest: +SKIP
        >>> r.contains([-0.3, 0.4])                    # doctest: +SKIP
        >>> r.marginal()                               # doctest: +SKIP
    """
    if not isinstance(result, RmaMvResult):
        raise TypeError("'result' must be an RmaMvResult from rma_mv()")
    if test not in ("z", "t"):
        raise ValueError(
            f"unknown test: {test!r}; expected 'z' or 't'")
    G, names = result.g_matrix()
    if G is None:
        G, names = _univariate_sigma(result)
    if G is None:
        raise ValueError(
            "the fit has no between-study covariance matrix to predict from; "
            "it needs a multivariate random term such as "
            "random='~ outcome | study' with struct='UN'")
    mu = np.asarray(result.beta, dtype=float).ravel()
    V = np.asarray(result.vb, dtype=float)
    Sigma = np.asarray(G, dtype=float)
    cov = V + Sigma
    for_observation = False
    if observation is not None:
        S = np.asarray(observation, dtype=float)
        if S.shape != cov.shape:
            raise ValueError(
                f"'observation' must be {cov.shape[0]}x{cov.shape[1]}, "
                f"not {S.shape[0]}x{S.shape[1]}")
        cov = cov + S
        for_observation = True

    alpha = level_to_alpha(result.level if level is None else level)
    J = int(mu.size)
    ref_df = None
    if test == "z":
        crit = float(stats.chi2.ppf(1.0 - alpha, J))
        if int(result.k) - int(result.p) <= 20:
            warnings.warn(
                "the chi-square reference is asymptotic in the number of "
                f"studies, and this fit has {int(result.k)} studies and "
                f"{int(result.p)} coefficients.  Measured over 300 "
                "replications of two correlated outcomes, the region covered "
                "0.77 at k=10 and 0.92 at k=20 against a claimed 0.95, rising "
                "to 0.94 at k=40; the Hotelling reference (test='t') gave "
                "0.83, 0.93 and 0.95 at the same settings",
                RuntimeWarning, stacklevel=2)
    else:
        m = float(int(result.k) - int(result.p) if df is None else df)
        if not np.isfinite(m) or m <= J:
            raise ValueError(
                f"test='t' needs more than {J} denominator degrees of freedom "
                f"for {J} outcomes and got {m:g}; use test='z' or pass df=")
        crit = float(J * (m - 1.0) / (m - J)
                     * stats.f.ppf(1.0 - alpha, J, m - J))
        ref_df = [float(J), float(m - J)]
    cov = 0.5 * (cov + cov.T)
    try:
        np.linalg.cholesky(cov)
    except np.linalg.LinAlgError as exc:
        raise ValueError(
            "the prediction covariance is not positive definite, so the "
            "region is unbounded; the between-study covariance is probably "
            "singular (see confint_rma_mv for the same diagnostic)") from exc

    return PredictionRegion(
        names=list(names) if names else [f"outcome {i + 1}"
                                         for i in range(mu.size)],
        center=mu,
        cov=cov,
        crit=crit,
        level=level_to_percent(result.level if level is None else level),
        k=int(result.k),
        for_observation=for_observation,
        test=test,
        df=ref_df,
    )
