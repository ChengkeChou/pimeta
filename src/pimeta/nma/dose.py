"""Dose-response network meta-analysis.

A dose-response network asks for a curve per treatment, not a number per
comparison, and the two things that makes hard are both structural:

- the effect of a contrast depends on **both** arms' doses.  Study *i* comparing
  treatment *b* at dose :math:`x_b` with treatment *a* at dose :math:`x_a`
  contributes

  .. math::

      y_i = f_b(x_b; \\beta_b) - f_a(x_a; \\beta_a) + e_i ,

  so a comparison between two nonzero doses carries information about the
  *difference* of the two curves, not about either curve alone;
- the model must reproduce the consistency equations of the network for **every**
  dose, not just at the observed ones.  That is automatic here, and it is worth
  being explicit about why: the same parameter vector :math:`\\beta_t` is used
  wherever treatment *t* appears, in either position.  Summing
  :math:`d_{ab}(x) + d_{bc}(x) - d_{ac}(x)` around a circuit gives zero as an
  algebraic identity, at any dose, including doses nobody studied.  A model that
  stored one parameter per *comparison* would not have this property, which is
  the reason the parameterisation is per treatment rather than per edge.

Each curve is anchored at a zero dose, :math:`f_t(0) = 0` for every *t*.  That is
an assumption, and it is the assumption that makes the model identified: without
an anchor, adding a constant to every curve leaves every contrast unchanged.  It
says that a dose of zero means no drug effect, whichever drug it is — true by
construction for a placebo arm, and an approximation for an active-control study
with no zero-dose arm.

Dose functions
--------------

``linear``
    :math:`\\beta_1 x`; one parameter per treatment.
``quadratic``
    :math:`\\beta_1 x + \\beta_2 x^2`.
``loglinear``
    :math:`\\beta_1 \\log(1 + x)`, the form used when the dose scale is
    multiplicative.
``rcs``
    A restricted cubic spline in dose, anchored at zero, with the knots given
    by the user or placed on the observed doses.
``emax``
    :math:`E\\,x / (ED_{50} + x)` — the hyperbola the pharmacology literature
    calls Emax, with :math:`ED_{50}` estimated on the log scale so it stays
    positive.
``exponential``
    :math:`E\\,(1 - e^{-x/\\lambda})`, which approaches its asymptote faster
    than Emax near zero and is the usual alternative to it.

Heterogeneity
-------------

Three structures, because they answer different questions and are not
interchangeable:

``struct="FE"``
    No between-study variation; the intervals are conditional on the studies at
    hand.
``struct="study"``
    A random effect on the study's level, :math:`\\Sigma = V + \\tau^2`.  This
    is the model the pairwise dose-response literature fits (``dosresmeta``'s
    random intercept), generalised to a network, and it is the default because
    it works for every dose function.
``struct="coef"``
    A random effect on the **curve**, :math:`\\Sigma = V + XDX'`, so a study's
    whole dose-response relationship may deviate, not just its level.
    ``coef_struct="DIAG"`` gives each coefficient its own variance,
    ``"UN"`` a full covariance within a treatment's block.  Only available for
    dose functions that are linear in their parameters: a random effect on
    :math:`(E, ED_{50})` would make the marginal likelihood an integral with no
    closed form, and this package does not sample.

``struct="coef+study"`` adds the level term to the curve term.

Estimation is by generalised least squares with the between-study covariance
profiled out: for a trial value of the variance parameters the coefficients are
weighted least squares (or non-linear least squares for ``emax`` and
``exponential``), and the variance parameters are chosen by restricted maximum
likelihood.  Nothing new is estimated beyond the dose-response parameters and a
handful of variances.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from scipy import stats
from scipy.optimize import minimize, minimize_scalar

from ..basis import _rcs_matrix, default_knots
from ..bayes import sample_log_posterior
from ..levels import level_to_alpha

__all__ = ["NmaDoseCurve", "NmaDoseResult", "dose_functions", "nma_dose"]


# --------------------------------------------------------------------------- #
# the dose functions
# --------------------------------------------------------------------------- #
_LINEAR_IN_PARAMETERS = ("linear", "quadratic", "loglinear", "rcs")

_PARAMETER_NAMES = {
    "linear": ("dose",),
    "quadratic": ("dose", "dose^2"),
    "loglinear": ("log1p(dose)",),
    "emax": ("emax", "ed50"),
    "exponential": ("emax", "scale"),
}


def dose_functions() -> list[str]:
    """The dose functions this module can fit, with ``rcs`` last.

    Example:
        >>> ma.nma.dose.dose_functions()          # doctest: +SKIP
        ['linear', 'quadratic', 'loglinear', 'emax', 'exponential', 'rcs']
    """
    return ["linear", "quadratic", "loglinear", "emax", "exponential", "rcs"]


def _check_family(family: str) -> str:
    name = str(family).lower()
    if name not in _PARAMETER_NAMES and name != "rcs":
        raise ValueError(
            f"unknown dose function {family!r}; use one of "
            + ", ".join(repr(f) for f in dose_functions()))
    return name


def _basis(family: str, dose, *, knots=None) -> np.ndarray:
    """The anchored design row(s) ``c(x)`` for the linear dose functions.

    Every column is zero at ``dose == 0``, which is what anchors the curve.  A
    restricted cubic spline is not zero at zero by construction — its basis is
    built from the pooled dose vector — so the value at zero is subtracted.
    """
    x = np.asarray(dose, dtype=float).ravel()
    if family == "linear":
        return x.reshape(-1, 1)
    if family == "quadratic":
        return np.column_stack([x, x ** 2])
    if family == "loglinear":
        if np.any(x < 0.0):
            raise ValueError(
                "the 'loglinear' dose function needs non-negative doses")
        return np.log1p(x).reshape(-1, 1)
    if family == "rcs":
        if knots is None:
            raise ValueError("the 'rcs' dose function needs 'knots'")
        z = _rcs_matrix(x, np.asarray(knots, dtype=float))
        zero = _rcs_matrix(np.zeros(1), np.asarray(knots, dtype=float))
        return z - zero
    raise ValueError(
        f"{family!r} is not linear in its parameters; there is no design row")


def _value(family: str, dose, params) -> np.ndarray:
    """``f(x; params)`` for the dose functions that are non-linear in them."""
    x = np.asarray(dose, dtype=float).ravel()
    p = np.asarray(params, dtype=float).ravel()
    # the scale parameters are on the log scale, and the optimiser is allowed
    # to wander far outside the data before it is pulled back; clipping keeps
    # that exploration from overflowing rather than changing the model, since
    # a scale of exp(30) is already unidentifiable
    scale = float(np.exp(np.clip(p[1], -30.0, 30.0)))
    if family == "emax":
        return p[0] * x / (scale + x)
    if family == "exponential":
        return p[0] * (1.0 - np.exp(-x / scale))
    raise ValueError(
        f"{family!r} is linear in its parameters; use the design row")


def _value_rows(family: str, dose, params) -> np.ndarray:
    """``f(x; params)`` with one parameter vector per row.

    The nonlinear dose functions are the reason a random effect on the curve
    needs a sampler, so they also need a form that takes a *per-row* parameter
    vector: a study's Emax and ED50 are its own.
    """
    x = np.asarray(dose, dtype=float).ravel()
    p = np.atleast_2d(np.asarray(params, dtype=float))
    if p.shape[0] == 1:
        p = np.broadcast_to(p, (x.size, p.shape[1]))
    scale = np.exp(np.clip(p[:, 1], -30.0, 30.0))
    if family == "emax":
        return p[:, 0] * x / (scale + x)
    if family == "exponential":
        return p[:, 0] * (1.0 - np.exp(-x / scale))
    raise ValueError(
        f"{family!r} is linear in its parameters; use the design row")


def _n_parameters(family: str, *, n_knots: int | None = None) -> int:
    if family == "rcs":
        if n_knots is None:
            raise ValueError("the 'rcs' dose function needs 'n_knots'")
        return int(n_knots)
    return len(_PARAMETER_NAMES[family])


def _parameter_names(family: str, *, n_knots: int | None = None,
                     label: str | None = None) -> list[str]:
    prefix = "" if label is None else f"{label}:"
    if family == "rcs":
        if n_knots is None:
            raise ValueError("the 'rcs' dose function needs 'n_knots'")
        return [f"{prefix}rcs[{j + 1}]" for j in range(int(n_knots))]
    return [f"{prefix}{name}" for name in _PARAMETER_NAMES[family]]


# --------------------------------------------------------------------------- #
# small linear-algebra helpers
# --------------------------------------------------------------------------- #
def _chol(matrix: np.ndarray) -> tuple[np.ndarray, float]:
    """Cholesky factor and log-determinant; raises if not positive definite."""
    chol = np.linalg.cholesky(matrix)
    return chol, 2.0 * float(np.sum(np.log(np.diag(chol))))


def _quadratic_form(chol: np.ndarray, vector: np.ndarray) -> float:
    """``v' M^-1 v`` from the Cholesky factor of ``M``."""
    sol = np.linalg.solve(chol, vector)
    return float(sol @ sol)


def _log_cholesky(theta: np.ndarray, p: int) -> np.ndarray:
    """``L L'`` from an unconstrained vector: diagonal logged, rest free."""
    lower = np.zeros((p, p))
    idx = 0
    for i in range(p):
        lower[i, i] = float(np.exp(theta[idx]))
        idx += 1
        for j in range(i):
            lower[i, j] = float(theta[idx])
            idx += 1
    return lower @ lower.T


# --------------------------------------------------------------------------- #
# results
# --------------------------------------------------------------------------- #
@dataclass
class NmaDoseCurve:
    """A treatment's dose-response curve, or a contrast between two curves."""

    labels: list[str] = field(default_factory=list)
    dose: np.ndarray = field(repr=False, default_factory=lambda: np.empty(0))
    est: np.ndarray = field(repr=False, default_factory=lambda: np.empty(0))
    se: np.ndarray = field(repr=False, default_factory=lambda: np.empty(0))
    ci_lb: np.ndarray = field(repr=False, default_factory=lambda: np.empty(0))
    ci_ub: np.ndarray = field(repr=False, default_factory=lambda: np.empty(0))
    pred_lb: np.ndarray = field(repr=False, default_factory=lambda: np.empty(0))
    pred_ub: np.ndarray = field(repr=False, default_factory=lambda: np.empty(0))
    outside: np.ndarray = field(repr=False, default_factory=lambda: np.empty(0))

    def to_dict(self) -> dict:
        return {
            "labels": list(self.labels),
            "dose": self.dose.tolist(),
            "est": self.est.tolist(),
            "se": self.se.tolist(),
            "ci_lb": self.ci_lb.tolist(),
            "ci_ub": self.ci_ub.tolist(),
            "pred_lb": self.pred_lb.tolist(),
            "pred_ub": self.pred_ub.tolist(),
            "outside": self.outside.tolist(),
        }

    def __str__(self) -> str:
        head = "  ".join(self.labels) if self.labels else "dose-response"
        rows = [f"{head:<28}{'dose':>9}{'est':>10}{'se':>9}{'ci_lb':>10}"
                f"{'ci_ub':>10}"]
        for i in range(self.dose.size):
            rows.append(f"{'':<28}{self.dose[i]:>9.4g}{self.est[i]:>10.4f}"
                        f"{self.se[i]:>9.4f}{self.ci_lb[i]:>10.4f}"
                        f"{self.ci_ub[i]:>10.4f}")
        return "\n".join(rows)


@dataclass
class NmaDoseResult:
    """A fitted dose-response network.

    Attributes:
        family: the dose function.
        treatments: every treatment in the network.
        curves: the treatments whose curve was estimated.  A treatment seen only
            at the anchor dose has no curve — its effect is zero by definition —
            and is reported here as omitted rather than given a parameter the
            data cannot identify.
        names: coefficient names, one block per treatment.
        coef, se: the dose-response parameters and their standard errors.
        vcov: their covariance.
        tau2: the between-study variance of the study level, or ``NaN`` when
            ``struct`` has no level term.
        D: the between-study covariance of the curve's coefficients, or
            ``None``.
        k: contrasts fitted.
        studies: the study labels.
        dose_range: the observed range of dose, per treatment.
        struct, coef_struct: the heterogeneity structure.
        method, level: settings.
        loglik: the maximised (restricted) log-likelihood, or for a Bayesian
            fit the log posterior at the draw with the highest density.
        anchor: the dose at which every curve is zero.
        inference: ``"reml"`` or ``"mcmc"``.
        posterior: the :class:`~pimeta.bayes.Posterior` when ``inference`` is
            ``"mcmc"``, else ``None``.  When it is present, ``coef`` and ``se``
            hold the posterior mean and standard deviation, and every interval
            the result reports is a draw-based credible interval rather than a
            delta-method one.
        fit: the raw optimiser output, for diagnosing a boundary.
    """

    family: str
    treatments: list[str]
    curves: list[str]
    names: list[str]
    coef: list[float]
    se: list[float]
    vcov: np.ndarray = field(repr=False, default_factory=lambda: np.empty((0, 0)))
    tau2: float = float("nan")
    D: np.ndarray | None = field(repr=False, default=None)
    k: int = 0
    studies: list = field(default_factory=list)
    dose_range: dict = field(default_factory=dict)
    struct: str = "study"
    coef_struct: str = "DIAG"
    method: str = "REML"
    level: float = 95.0
    loglik: float = float("nan")
    anchor: float = 0.0
    n_knots: int | None = None
    knots: list[float] | None = None
    treatment_specific: bool = True
    inference: str = "reml"
    posterior: Any = field(repr=False, default=None)
    fit: Any = field(repr=False, default=None)

    # -- the machinery a prediction needs, rebuilt from the stored settings -- #
    def _blocks(self) -> list[str]:
        if not self.treatment_specific:
            return ["shared"]
        return list(self.curves)

    def _block_of(self, treat) -> int | None:
        name = str(treat)
        if name not in self.treatments:
            raise ValueError(f"{treat!r} is not a treatment in this network")
        blocks = self._blocks()
        if not self.treatment_specific:
            return 0
        return blocks.index(name) if name in blocks else None

    def _params(self, block: int) -> np.ndarray:
        p = _n_parameters(self.family, n_knots=self.n_knots)
        lo = block * p
        return np.asarray(self.coef, dtype=float)[lo:lo + p]

    def _curve_draws(self, treat, dose, sign: float = 1.0) -> np.ndarray | None:
        """``f_t(x)`` for every posterior draw: ``(n_draws, n_dose)``.

        A Bayesian fit does not need a delta method: pushing each draw through
        the curve and taking quantiles is exact, and it stays honest when the
        posterior is skewed, which for a ratio like ED50 it usually is.
        """
        if self.posterior is None:
            return None
        x = np.asarray(dose, dtype=float).ravel()
        block = self._block_of(treat)
        draws = np.asarray(self.posterior.draws)
        if block is None:
            return np.zeros((draws.shape[0], x.size))
        p = _n_parameters(self.family, n_knots=self.n_knots)
        params = draws[:, block * p:(block + 1) * p]
        if self.family in _LINEAR_IN_PARAMETERS:
            return sign * (_basis(self.family, x, knots=self.knots) @ params.T).T
        return sign * np.vstack([
            _value(self.family, x, row) for row in params])

    def _curve_value(self, treat, dose) -> np.ndarray:
        """``f_t(x)`` on the fitted parameters: zero where there is no curve."""
        x = np.asarray(dose, dtype=float).ravel()
        block = self._block_of(treat)
        if block is None:
            return np.zeros(x.size)
        params = self._params(block)
        if self.family in _LINEAR_IN_PARAMETERS:
            return _basis(self.family, x, knots=self.knots) @ params
        return _value(self.family, x, params)

    def _curve_jacobian(self, treat, dose) -> np.ndarray:
        """``d f_t(x) / d beta`` with columns over the whole parameter vector."""
        x = np.asarray(dose, dtype=float).ravel()
        p = _n_parameters(self.family, n_knots=self.n_knots)
        out = np.zeros((x.size, len(self.coef)))
        block = self._block_of(treat)
        if block is None:
            return out
        if self.family in _LINEAR_IN_PARAMETERS:
            out[:, block * p:(block + 1) * p] = _basis(
                self.family, x, knots=self.knots)
            return out
        params = self._params(block)
        for j in range(p):
            step = 1e-6 * max(1.0, abs(float(params[j])))
            up = params.copy()
            down = params.copy()
            up[j] += step
            down[j] -= step
            out[:, block * p + j] = (
                _value(self.family, x, up) - _value(self.family, x, down)
            ) / (2.0 * step)
        return out

    def _crit(self) -> float:
        return float(stats.norm.ppf(0.5 + self.level / 200.0))

    # -- what a reader asks for -------------------------------------------- #
    def coef_table(self) -> list[dict]:
        crit = self._crit()
        rows = []
        for name, est, se in zip(self.names, self.coef, self.se):
            z = est / se if se > 0 else float("nan")
            rows.append({
                "term": name,
                "estimate": float(est),
                "se": float(se),
                "zval": float(z),
                "pval": float(2.0 * stats.norm.sf(abs(z))) if se > 0 else float("nan"),
                "ci_lb": float(est - crit * se),
                "ci_ub": float(est + crit * se),
            })
        return rows

    def curve(self, treat, doses=None, *, level: float | None = None,
              n: int = 50) -> NmaDoseCurve:
        """One treatment's curve: effect at dose ``x`` against its zero dose.

        Args:
            treat: the treatment label.
            doses: the doses to evaluate at; defaults to a grid over the
                treatment's observed range.
            level: confidence level for this call; the fit's by default.
            n: grid size when ``doses`` is None.

        Returns:
            :class:`NmaDoseCurve` with a confidence band and a prediction band
            for a new study.  The two coincide under ``struct="FE"``, which is
            reported rather than hidden.
        """
        if doses is None:
            lo, hi = self.dose_range.get(str(treat), (self.anchor, self.anchor))
            grid = np.linspace(lo, hi, int(n))
        else:
            grid = np.asarray(doses, dtype=float).ravel()
        return self._band([(str(treat), grid)], level=level)

    def contrast(self, treat_a, dose_a, treat_b, dose_b, *,
                 level: float | None = None) -> NmaDoseCurve:
        """``f_b(dose_b) - f_a(dose_a)`` at a single pair of doses."""
        return self._band(
            [(str(treat_b), np.array([float(dose_b)])),
             (str(treat_a), np.array([float(dose_a)]))],
            level=level,
            signed=[1.0, -1.0],
            labels=[f"{treat_b}@{dose_b:g} vs {treat_a}@{dose_a:g}"],
        )

    def _band(self, parts, *, level: float | None = None,
              signed=None, labels=None) -> NmaDoseCurve:
        alpha = level_to_alpha(level) if level is not None else \
            level_to_alpha(self.level)
        crit = float(stats.norm.ppf(1.0 - alpha / 2.0))
        signs = [1.0] * len(parts) if signed is None else list(signed)
        size = parts[0][1].size
        lo, hi = self.dose_range.get(parts[0][0], (self.anchor, self.anchor))
        outside = (parts[0][1] < lo) | (parts[0][1] > hi)
        if self.posterior is not None:
            return self._band_from_draws(parts, signs, alpha, labels, outside)
        est = np.zeros(size)
        jac = np.zeros((size, len(self.coef)))
        for (treat, x), s in zip(parts, signs):
            est += s * self._curve_value(treat, x)
            jac += s * self._curve_jacobian(treat, x)
        var = np.einsum("ij,jk,ik->i", jac, np.asarray(self.vcov), jac)
        var = np.maximum(var, 0.0)
        se = np.sqrt(var)
        extra = self._prediction_variance(jac)
        return NmaDoseCurve(
            labels=list(labels) if labels is not None
            else [f"{parts[0][0]} vs dose {self.anchor:g}"],
            dose=parts[0][1].copy(),
            est=est,
            se=se,
            ci_lb=est - crit * se,
            ci_ub=est + crit * se,
            pred_lb=est - crit * np.sqrt(var + extra),
            pred_ub=est + crit * np.sqrt(var + extra),
            outside=outside,
        )

    def _band_from_draws(self, parts, signs, alpha, labels,
                         outside) -> NmaDoseCurve:
        """Credible band: push every draw through the curve, then take quantiles.

        The prediction band adds one draw of the between-study deviation per
        posterior draw, so it is the interval a *new* study's curve would fall
        in — built by sampling, not by adding a variance to an interval, which
        for a curve that is nonlinear in its parameters would be a different
        number.
        """
        curves = None
        for (treat, x), s in zip(parts, signs):
            piece = self._curve_draws(treat, x, sign=s)
            assert piece is not None
            curves = piece if curves is None else curves + piece
        assert curves is not None
        n_draw = curves.shape[0]
        p = _n_parameters(self.family, n_knots=self.n_knots)
        predict = curves.copy()
        if self.D is not None and np.size(self.D):
            rng = np.random.default_rng(0)
            chol = np.linalg.cholesky(
                np.asarray(self.D) + 1e-12 * np.eye(np.asarray(self.D).shape[0]))
            noise = rng.standard_normal((n_draw, chol.shape[0])) @ chol.T
            for (treat, x), s in zip(parts, signs):
                block = self._block_of(treat)
                if block is None:
                    continue
                params = noise[:, block * p:(block + 1) * p]
                xv = np.asarray(x, dtype=float)
                if self.family in _LINEAR_IN_PARAMETERS:
                    basis = _basis(self.family, xv, knots=self.knots)
                    predict += s * (basis @ params.T).T
                else:
                    predict += s * np.vstack([
                        _value(self.family, xv, row) for row in params])
        if np.isfinite(self.tau2) and self.tau2 > 0.0:
            rng = np.random.default_rng(1)
            predict = predict + rng.normal(0.0, np.sqrt(self.tau2), (n_draw, 1))
        tail = alpha / 2.0
        return NmaDoseCurve(
            labels=list(labels) if labels is not None
            else [f"{parts[0][0]} vs dose {self.anchor:g}"],
            dose=parts[0][1].copy(),
            est=np.mean(curves, axis=0),
            se=np.std(curves, axis=0, ddof=1),
            ci_lb=np.quantile(curves, tail, axis=0),
            ci_ub=np.quantile(curves, 1.0 - tail, axis=0),
            pred_lb=np.quantile(predict, tail, axis=0),
            pred_ub=np.quantile(predict, 1.0 - tail, axis=0),
            outside=outside,
        )

    def _prediction_variance(self, jac: np.ndarray) -> np.ndarray:
        """Between-study variance a new study's curve would add."""
        out = np.zeros(jac.shape[0])
        if np.isfinite(self.tau2) and self.tau2 > 0.0:
            out += self.tau2
        if self.D is not None and np.size(self.D):
            out += np.einsum("ij,jk,ik->i", jac, np.asarray(self.D), jac)
        return out

    def test_dose(self) -> dict:
        """Wald test that no curve has any dose effect at all."""
        beta = np.asarray(self.coef, dtype=float)
        vcov = np.asarray(self.vcov, dtype=float)
        if beta.size == 0:
            raise ValueError("this fit has no dose-response parameters")
        stat = float(beta @ np.linalg.solve(vcov, beta))
        df = int(beta.size)
        return {
            "statistic": stat,
            "df": df,
            "pval": float(stats.chi2.sf(stat, df)),
            "test": "chi2",
        }

    def to_dict(self) -> dict:
        return {
            "family": self.family,
            "treatments": list(self.treatments),
            "curves": list(self.curves),
            "names": list(self.names),
            "coef": [float(v) for v in self.coef],
            "se": [float(v) for v in self.se],
            "tau2": None if not np.isfinite(self.tau2) else float(self.tau2),
            "D": None if self.D is None else np.asarray(self.D).tolist(),
            "k": self.k,
            "studies": list(self.studies),
            "dose_range": {k: [float(v[0]), float(v[1])]
                           for k, v in self.dose_range.items()},
            "struct": self.struct,
            "coef_struct": self.coef_struct,
            "method": self.method,
            "level": self.level,
            "loglik": self.loglik,
            "anchor": self.anchor,
            "n_knots": self.n_knots,
            "knots": None if self.knots is None else [float(v) for v in self.knots],
            "treatment_specific": self.treatment_specific,
        }

    def __str__(self) -> str:
        parts = [
            f"Dose-response network meta-analysis ({self.family} dose function)",
            f"  treatments: {', '.join(self.treatments)}",
            f"  curves estimated for: {', '.join(self.curves) or 'none'}",
            f"  {self.k} contrasts from {len(self.studies)} studies",
            f"  heterogeneity: {self.struct}"
            + (f" ({self.coef_struct})" if self.D is not None else ""),
        ]
        if np.isfinite(self.tau2):
            if self.tau2 > 0.0:
                parts.append(
                    f"  tau2 = {self.tau2:.6f} (tau = {np.sqrt(self.tau2):.6f})")
            else:
                parts.append("  tau2 = 0 (no between-study variation)")
        else:
            parts.append("  no study-level term in this structure")
        parts.append(f"  log-likelihood ({self.method}) = {self.loglik:.6f}")
        width = max([len(n) for n in self.names] + [6])
        for name, est, se in zip(self.names, self.coef, self.se):
            parts.append(f"  {name:<{width}}  {est:>10.6f}  ({se:.6f})")
        return "\n".join(parts)


# --------------------------------------------------------------------------- #
# the estimator
# --------------------------------------------------------------------------- #
def _as_vector(values, name: str, n: int | None = None) -> np.ndarray:
    out = np.asarray(values, dtype=float).ravel()
    if n is not None and out.size != n:
        raise ValueError(f"'{name}' has {out.size} entries, expected {n}")
    return out


def nma_dose(yi, *, study, treat1, treat2, dose1, dose2, vi=None, vcov=None,
             rho=None, family: str = "linear", treatment_specific: bool = True,
             knots=None, n_knots: int | None = None, struct: str = "study",
             coef_struct: str = "DIAG", method: str = "REML",
             level: float = 0.95, anchor: float = 0.0, max_iter: int = 500,
             start=None, tol: float = 1e-9, inference: str = "reml",
             mcmc: dict | None = None) -> NmaDoseResult:
    """Fit a dose-response model to a network of contrasts.

    Args:
        yi: the observed contrast of each row, on the linear-predictor scale.
        study: study label per row.
        treat1, treat2: the two treatments compared, ``treat2`` minus
            ``treat1`` in sign.  Which is which matters for the dose, not for
            the model.
        dose1, dose2: the dose of each arm.  ``anchor`` (zero by default) means
            no drug.
        vi: variance of ``yi`` per row; ignored when ``vcov`` is given.
        vcov: full covariance of ``yi``.  A study with several contrasts —
            several doses against one control, or a multi-arm study — has
            correlated rows, and this is where that goes.
        rho: constant within-study correlation, used to build ``vcov`` from
            ``vi`` as ``rho * sqrt(vi_i * vi_j)`` inside a study.  An
            assumption, not an estimate; supplied rather than defaulted for
            that reason.
        family: the dose function, see :func:`dose_functions`.
        treatment_specific: ``True`` (default) gives every treatment its own
            curve; ``False`` fits one curve shared by all of them, which is the
            "class" model.
        knots, n_knots: for ``family="rcs"``.  ``n_knots`` is the number of
            spline columns, placed on the observed doses; ``knots`` gives them
            explicitly.
        struct: ``"FE"``, ``"study"`` (default), ``"coef"`` or
            ``"coef+study"``.
        coef_struct: ``"DIAG"`` or ``"UN"`` for the curve's random effects.
        method: ``"REML"`` (default) or ``"ML"``.
        level: confidence level; ``0.95``, ``95`` and ``0.05`` are all read as
            the same 95%.
        anchor: the dose at which every curve is zero.
        max_iter: optimiser iteration cap.
        start: optional starting values for the dose-response parameters.
        tol: convergence tolerance on the variance parameters.
        inference: ``"reml"`` (default) profiles the variance parameters by
            restricted maximum likelihood; ``"mcmc"`` samples the posterior
            instead, which is the only way to fit a **random effect on the
            curve** with a dose function that is not linear in its parameters —
            the marginal likelihood is then an integral with no closed form. A
            Bayesian fit also reports draw-based credible and prediction
            intervals rather than delta-method ones.
        mcmc: sampler settings and priors for ``inference="mcmc"``. Recognised
            keys are ``n_draws``, ``warmup``, ``chains``, ``seed``,
            ``prior_beta`` (scale of the normal prior on the curve parameters)
            and ``prior_het`` (scale of the half-normal prior on every
            between-study standard deviation).

    Returns:
        :class:`NmaDoseResult`.

    Raises:
        ValueError: for mismatched inputs, an unknown dose function, a negative
            dose under ``loglinear``, a covariance that is not positive
            definite, or a design that cannot identify the curves asked for.

    Example:
        >>> fit = ma.nma_dose(yi, study=study, treat1=arm1, treat2=arm2,
        ...                   dose1=d1, dose2=d2, vi=vi, family="emax")
        >>> print(fit.curve("drugB", [0, 5, 10]))
    """
    family = _check_family(family)
    if str(inference).lower() == "mcmc":
        return _nma_dose_mcmc(
            yi, study=study, treat1=treat1, treat2=treat2, dose1=dose1,
            dose2=dose2, vi=vi, vcov=vcov, rho=rho, family=family,
            treatment_specific=treatment_specific, knots=knots,
            n_knots=n_knots, struct=struct, coef_struct=coef_struct,
            level=level, anchor=anchor, settings=mcmc)
    if str(inference).lower() != "reml":
        raise ValueError(
            f"unknown 'inference' {inference!r}; use 'reml' or 'mcmc'")
    if struct not in ("FE", "study", "coef", "coef+study"):
        raise ValueError(
            f"unknown 'struct' {struct!r}; use 'FE', 'study', 'coef' or "
            "'coef+study'")
    if coef_struct not in ("DIAG", "UN"):
        raise ValueError(
            f"unknown 'coef_struct' {coef_struct!r}; use 'DIAG' or 'UN'")
    use_coef = struct in ("coef", "coef+study")
    if use_coef and family not in _LINEAR_IN_PARAMETERS:
        raise ValueError(
            f"a random effect on the curve needs a dose function that is "
            f"linear in its parameters, and {family!r} is not: a random "
            f"effect on (emax, ed50) makes the marginal likelihood an integral "
            f"with no closed form. Use struct='study' or struct='FE' with "
            f"{family!r}, or a linear/quadratic/loglinear/rcs dose function "
            f"with struct='coef'.")
    method_name = str(method).upper()
    if method_name not in ("REML", "ML"):
        raise ValueError(f"unknown 'method' {method!r}; use 'REML' or 'ML'")
    level = 100.0 * (1.0 - level_to_alpha(level))

    y = _as_vector(yi, "yi")
    k = y.size
    if k == 0:
        raise ValueError("no rows to fit")
    studies = [str(s) for s in np.asarray(study).ravel()]
    a1 = [str(t) for t in np.asarray(treat1).ravel()]
    a2 = [str(t) for t in np.asarray(treat2).ravel()]
    d1 = _as_vector(dose1, "dose1", k)
    d2 = _as_vector(dose2, "dose2", k)
    for name, values in (("study", studies), ("treat1", a1), ("treat2", a2)):
        if len(values) != k:
            raise ValueError(
                f"'{name}' has {len(values)} entries, expected {k}")
    if not np.all(np.isfinite(y)):
        raise ValueError("'yi' contains missing or infinite values")
    if np.any(d1 < 0.0) or np.any(d2 < 0.0):
        raise ValueError("doses must be non-negative")
    if np.any(d1 < anchor) or np.any(d2 < anchor):
        raise ValueError(f"doses must be at least the anchor ({anchor:g})")

    # --- the covariance of the observed contrasts ------------------------- #
    studies_unique = list(dict.fromkeys(studies))
    idx = {s: i for i, s in enumerate(studies_unique)}
    sid = np.array([idx[s] for s in studies])
    if vcov is not None:
        V = np.asarray(vcov, dtype=float)
        if V.shape != (k, k):
            raise ValueError(
                f"'vcov' is {V.shape}, expected ({k}, {k})")
        V = 0.5 * (V + V.T)
    else:
        if vi is None:
            raise ValueError("give 'vi' or 'vcov'")
        v = _as_vector(vi, "vi", k)
        if np.any(~np.isfinite(v)) or np.any(v <= 0.0):
            raise ValueError("'vi' must be positive and finite")
        V = np.diag(v)
        if rho is not None:
            r = float(rho)
            if not 0.0 <= r < 1.0:
                raise ValueError("'rho' must be in [0, 1)")
            for s in studies_unique:
                sel = np.flatnonzero(sid == idx[s])
                if sel.size < 2:
                    continue
                sd = np.sqrt(v[sel])
                V[np.ix_(sel, sel)] = r * np.outer(sd, sd)
                V[sel, sel] = v[sel]
    try:
        np.linalg.cholesky(V)
    except np.linalg.LinAlgError as exc:
        raise ValueError(
            "the within-study covariance 'V' is not positive definite: check "
            "the standard errors and the correlation you assumed") from exc

    # --- who has a curve -------------------------------------------------- #
    treatments = sorted(set(a1) | set(a2))
    seen_nontrivial = {}
    for i in range(k):
        for treat, dose in ((a2[i], float(d2[i])), (a1[i], float(d1[i]))):
            if not np.isclose(dose, anchor):
                seen_nontrivial[treat] = True
    curves = [t for t in treatments if seen_nontrivial.get(t, False)]
    if not curves:
        raise ValueError(
            f"every dose equals the anchor ({anchor:g}), so no treatment has a "
            "dose-response function to estimate")
    if not treatment_specific:
        blocks: list[str] = ["shared"]
    else:
        blocks = list(curves)
    n_blocks = len(blocks)

    if family == "rcs":
        if knots is None:
            all_dose = np.concatenate([d1, d2])
            all_dose = all_dose[~np.isclose(all_dose, anchor)]
            nk = 3 if n_knots is None else int(n_knots)
            if nk < 1:
                raise ValueError("'n_knots' must be at least 1")
            if np.unique(all_dose).size < max(nk, 4):
                raise ValueError(
                    f"a restricted cubic spline with {nk} column(s) needs at "
                    f"least {max(nk, 4)} distinct doses; found "
                    f"{np.unique(all_dose).size}")
            knots = default_knots(all_dose, nk + 1)
        knots = [float(v) for v in np.sort(np.asarray(knots, dtype=float).ravel())]
        p = len(knots) - 1
    else:
        p = _n_parameters(family)
        knots = None
    names: list[str] = []
    for label in blocks:
        names.extend(_parameter_names(
            family, n_knots=p, label=None if label == "shared" else label))
    n_par = n_blocks * p

    def block_of(treat: str) -> int | None:
        if not treatment_specific:
            return 0
        return blocks.index(treat) if treat in blocks else None

    # --- the fixed design (linear families) ------------------------------- #
    X = np.zeros((k, n_par))
    if family in _LINEAR_IN_PARAMETERS:
        for i in range(k):
            for treat, dose, sign in ((a2[i], d2[i], 1.0),
                                      (a1[i], d1[i], -1.0)):
                block = block_of(treat)
                if block is None:
                    continue
                X[i, block * p:(block + 1) * p] += sign * _basis(
                    family, np.array([dose]), knots=knots)[0]
        rank = int(np.linalg.matrix_rank(X))
        if rank < n_par:
            offenders = []
            for block, treat in enumerate(blocks):
                cols = X[:, block * p:(block + 1) * p]
                if cols.size and np.linalg.matrix_rank(cols) < p:
                    offenders.append(treat)
            detail = (f" (the curve of {', '.join(offenders)} cannot be "
                      f"identified: it needs more distinct doses)" if offenders
                      else "")
            raise ValueError(
                f"the dose-response design has rank {rank} but {n_par} "
                f"parameter(s){detail}")

    # every curve needs more distinct non-zero doses than it has parameters,
    # whichever family it is: two parameters cannot be told apart from one dose
    for treat in curves:
        seen = {round(float(d), 10) for i in range(k)
                for t, d in ((a2[i], d2[i]), (a1[i], d1[i]))
                if t == treat and not np.isclose(float(d), anchor)}
        if len(seen) < p:
            raise ValueError(
                f"the curve of {treat!r} has {p} parameter(s) but that "
                f"treatment is only ever seen at {len(seen)} non-zero dose(s)"
                + (f" ({', '.join(f'{v:g}' for v in sorted(seen))})" if seen
                   else "")
                + "; the curve cannot be identified from this data")

    span = {}
    for treat in treatments:
        seen_doses = [float(d) for i in range(k) for t, d in
                      ((a2[i], d2[i]), (a1[i], d1[i])) if t == treat]
        span[treat] = ((min(seen_doses), max(seen_doses)) if seen_doses
                       else (anchor, anchor))

    # --- the random-effect design ----------------------------------------- #
    Zs = np.zeros((k, len(studies_unique)))
    Zs[np.arange(k), sid] = 1.0
    use_study = struct in ("study", "coef+study")
    share = Zs @ Zs.T
    if use_study and len(studies_unique) < 2:
        raise ValueError(
            "a between-study variance needs at least two studies; use "
            "struct='FE' for a single study")

    # which curve block each arm belongs to; -1 when the arm has no curve
    blk2 = np.full(k, -1, dtype=int)
    blk1 = np.full(k, -1, dtype=int)
    for i in range(k):
        found = block_of(a2[i])
        if found is not None:
            blk2[i] = found
        found = block_of(a1[i])
        if found is not None:
            blk1[i] = found

    def curve_mean(beta: np.ndarray) -> np.ndarray:
        if family in _LINEAR_IN_PARAMETERS:
            return X @ beta
        mu = np.zeros(k)
        for block in range(n_blocks):
            params = beta[block * p:(block + 1) * p]
            up = blk2 == block
            down = blk1 == block
            if up.any():
                mu[up] += _value(family, d2[up], params)
            if down.any():
                mu[down] -= _value(family, d1[down], params)
        return mu

    def curve_jac(beta: np.ndarray) -> np.ndarray:
        if family in _LINEAR_IN_PARAMETERS:
            return X
        out = np.zeros((k, n_par))
        for j in range(n_par):
            step = 1e-6 * max(1.0, abs(float(beta[j])))
            up = beta.copy()
            down = beta.copy()
            up[j] += step
            down[j] -= step
            out[:, j] = (curve_mean(up) - curve_mean(down)) / (2.0 * step)
        return out

    # --- theta -> the between-study covariance ----------------------------- #
    block_sizes = [(b * p, (b + 1) * p) for b in range(n_blocks)]
    n_diag = n_blocks * p
    n_un = n_blocks * p * (p + 1) // 2
    n_coef = n_diag if coef_struct == "DIAG" else n_un
    n_theta = (1 if use_study else 0) + (n_coef if use_coef else 0)

    def unpack(theta: np.ndarray):
        pos = 0
        tau2 = float(np.exp(2.0 * theta[0])) if use_study else 0.0
        pos += 1 if use_study else 0
        D = None
        if use_coef:
            D = np.zeros((n_par, n_par))
            if coef_struct == "DIAG":
                for lo, hi in block_sizes:
                    D[lo:hi, lo:hi] = np.diag(np.exp(2.0 * theta[pos:pos + p]))
                    pos += p
            else:
                per = p * (p + 1) // 2
                for lo, hi in block_sizes:
                    D[lo:hi, lo:hi] = _log_cholesky(theta[pos:pos + per], p)
                    pos += per
        return tau2, D

    def sigma_of(theta: np.ndarray) -> np.ndarray:
        tau2, D = unpack(theta)
        S = V.copy()
        if use_study:
            S += tau2 * share
        if D is not None:
            S += X @ D @ X.T
        return S

    # --- profile the dose-response parameters out ------------------------- #
    def fit_beta(theta: np.ndarray, beta0: np.ndarray | None):
        S = sigma_of(theta)
        chol, logdet = _chol(S)
        if family in _LINEAR_IN_PARAMETERS:
            w = np.linalg.solve(chol, np.eye(k))
            xtwx = X.T @ (w.T @ w) @ X
            beta = np.linalg.solve(xtwx, X.T @ (w.T @ w) @ y)
            resid = y - X @ beta
            return beta, resid, chol, logdet, xtwx
        # non-linear: least squares in the GLS metric, warm-started from the
        # previous value of beta.  A full simplex search at every trial value of
        # the variance parameters would make the profile likelihood hundreds of
        # times more expensive than the fit it is profiling, so the simplex is
        # used only to find a start and, once, to polish the optimum.
        def objective(b):
            if not np.all(np.isfinite(b)):
                return 1e20
            r = y - curve_mean(b)
            if not np.all(np.isfinite(r)):
                return 1e20
            value = _quadratic_form(chol, r)
            return value if np.isfinite(value) else 1e20

        b0 = np.zeros(n_par) if beta0 is None else np.asarray(beta0, dtype=float)
        best = minimize(objective, b0, method="L-BFGS-B",
                        options={"maxiter": max_iter, "ftol": 1e-14,
                                 "gtol": 1e-10})
        if not np.isfinite(best.fun) or best.fun >= 1e20:
            best = minimize(objective, b0, method="Nelder-Mead",
                            options={"maxiter": max_iter * 2, "xatol": 1e-10,
                                     "fatol": 1e-12})
        beta = np.asarray(best.x, dtype=float)
        resid = y - curve_mean(beta)
        jac = curve_jac(beta)
        w = np.linalg.solve(chol, np.eye(k))
        jwj = jac.T @ (w.T @ w) @ jac
        return beta, resid, chol, logdet, jwj

    def polish_beta(theta: np.ndarray, beta0: np.ndarray, *, thorough: bool):
        """Re-solve for ``beta`` at a fixed covariance, exactly at the optimum."""
        if family in _LINEAR_IN_PARAMETERS:
            return fit_beta(theta, beta0)[0]
        S = sigma_of(theta)
        chol, _ = _chol(S)

        def objective(b):
            if not np.all(np.isfinite(b)):
                return 1e20
            r = y - curve_mean(b)
            if not np.all(np.isfinite(r)):
                return 1e20
            value = _quadratic_form(chol, r)
            return value if np.isfinite(value) else 1e20

        best = minimize(objective, beta0, method="L-BFGS-B",
                        options={"maxiter": max_iter * 4, "ftol": 1e-16,
                                 "gtol": 1e-12})
        if thorough:
            start = best.x if np.isfinite(best.fun) else beta0
            simplex = minimize(
                objective, start, method="Nelder-Mead",
                options={"maxiter": max_iter * 10, "xatol": 1e-12,
                         "fatol": 1e-14})
            if simplex.fun <= best.fun:
                best = simplex
            again = minimize(objective, best.x, method="L-BFGS-B",
                             options={"maxiter": max_iter * 4, "ftol": 1e-16,
                                      "gtol": 1e-12})
            if again.fun <= best.fun:
                best = again
        return np.asarray(best.x, dtype=float)

    def objective_theta(theta: np.ndarray, beta0):
        try:
            beta, resid, chol, logdet, jwj = fit_beta(theta, beta0)
        except np.linalg.LinAlgError:
            return 1e20, None, None, None
        value = logdet + _quadratic_form(chol, resid)
        if method_name == "REML":
            sign, ld = np.linalg.slogdet(jwj)
            if sign <= 0:
                return 1e20, None, None, None
            value += ld
        return 0.5 * value, beta, resid, (chol, logdet, jwj)

    beta = np.zeros(n_par) if start is None else np.asarray(start, float).ravel()
    if beta.size != n_par:
        raise ValueError(f"'start' has {beta.size} value(s), expected {n_par}")
    info = None
    theta_hat = np.zeros(n_theta)
    if n_theta:
        theta0 = np.zeros(n_theta)
        if use_study:
            theta0[0] = -3.0
        if use_coef:
            theta0[(1 if use_study else 0):] = -2.0
        if use_study:
            # a crude moment start for tau, so the optimiser does not have to
            # find the scale of the problem from zero
            fe = fit_beta(theta0, beta)[0]
            resid = y - curve_mean(fe)
            kk = k - n_par
            if kk > 1:
                q = float(resid @ resid)
                theta0[0] = 0.5 * np.log(max(q / kk, 1e-6))
        best = None
        bounds = [(-8.0, 8.0)] * n_theta
        current = theta0
        previous = np.inf
        if n_theta == 1:
            # a bounded scalar search costs a tenth of what a numerical
            # gradient does here, because every gradient pair is a full
            # refit of the dose-response parameters
            scalar = minimize_scalar(
                lambda t, b=beta: objective_theta(np.array([t]), b)[0],
                bounds=(-8.0, 8.0), method="bounded",
                options={"xatol": max(tol, 1e-8)})
            current = np.array([float(scalar.x)])
            passes = 1
        else:
            passes = 3
        for _ in range(passes):
            res = minimize(
                lambda t, b=beta: objective_theta(t, b)[0], current,
                method="L-BFGS-B", bounds=bounds,
                options={"maxiter": max_iter, "ftol": tol})
            if best is None or res.fun < best.fun:
                best = res
            if np.isfinite(previous) and abs(previous - res.fun) < tol:
                break
            previous = float(res.fun)
            current = np.asarray(res.x, dtype=float)
            beta = objective_theta(current, beta)[1]
        assert best is not None
        polish = minimize(
            lambda t: objective_theta(t, beta)[0], np.asarray(best.x, float),
            method="Nelder-Mead",
            options={"maxiter": max_iter * 4, "xatol": 1e-8, "fatol": 1e-10})
        if polish.fun <= best.fun:
            best = polish
        theta_hat = np.asarray(best.x, dtype=float)
        if beta is None:
            beta = np.zeros(n_par)
        beta = polish_beta(theta_hat, beta, thorough=True)
        value, beta, resid, info = objective_theta(theta_hat, beta)
    else:
        beta = polish_beta(theta_hat, beta, thorough=True)
        value, beta, resid, info = objective_theta(theta_hat, beta)
    if info is None:
        raise RuntimeError(
            "the likelihood could not be evaluated at the optimum; the "
            "covariance structure is probably not positive definite")
    chol, logdet, jwj = info
    tau2, D = unpack(theta_hat)
    vcov = np.linalg.inv(jwj)
    vcov = 0.5 * (vcov + vcov.T)
    se = np.sqrt(np.maximum(np.diag(vcov), 0.0))

    if use_study and (tau2 <= 1e-6 * float(np.mean(np.diag(V)))
                      or float(theta_hat[0]) <= -7.5):
        warnings.warn(
            f"the between-study variance is estimated at the boundary "
            f"(tau2 = {tau2:.3g}, against a mean sampling variance of "
            f"{float(np.mean(np.diag(V))):.3g}); the intervals are then the "
            "fixed-effect ones and overstate the precision",
            RuntimeWarning, stacklevel=2)
    if D is not None and np.min(np.diag(D)) < 1e-8:
        warnings.warn(
            "a between-study variance of the curve's coefficients is "
            "estimated at zero; the curve term adds nothing to the model",
            RuntimeWarning, stacklevel=2)

    return NmaDoseResult(
        family=family,
        treatments=treatments,
        curves=curves,
        names=names,
        coef=[float(v) for v in beta],
        se=[float(v) for v in se],
        vcov=vcov,
        tau2=tau2 if use_study else float("nan"),
        D=D,
        k=k,
        studies=studies_unique,
        dose_range=span,
        struct=struct,
        coef_struct=coef_struct,
        method=method_name,
        level=level,
        loglik=-float(value),
        anchor=float(anchor),
        n_knots=p if family == "rcs" else None,
        knots=knots,
        treatment_specific=bool(treatment_specific),
        fit={"theta": theta_hat.tolist(), "loglik": -float(value)},
    )


# --------------------------------------------------------------------------- #
# the Bayesian fit
# --------------------------------------------------------------------------- #
_MCMC_DEFAULTS: dict[str, Any] = {
    "n_draws": 4000,
    "warmup": 2000,
    "chains": 4,
    "seed": None,
    "prior_beta": 10.0,
    "prior_het": 0.5,
    "check": True,
}


def _nma_dose_mcmc(yi, *, study, treat1, treat2, dose1, dose2, vi, vcov, rho,
                   family, treatment_specific, knots, n_knots, struct,
                   coef_struct, level, anchor,
                   settings: dict | None = None) -> NmaDoseResult:
    """Fit the hierarchical dose-response model by sampling the posterior.

    The model is the same one :func:`nma_dose` fits, with the random effect
    where the closed-form version cannot have it: on the **curve parameters
    themselves**,

    .. math::

        \\beta_{it} = \\beta_t + b_{it}, \\qquad
        b_{it} \\sim N(0, \\operatorname{diag}(\\sigma_t^2)),

    so a study's whole dose-response relationship may deviate, not just its
    level. For a dose function that is linear in its parameters the marginal
    likelihood of that model is closed-form and REML handles it; for ``emax`` or
    ``exponential`` it is an integral, and this function samples :math:`b_{it}`
    instead of integrating it out.

    The deviations are sampled **non-centred** (:math:`b = \\sigma z` with
    :math:`z \\sim N(0, 1)`), because a centred random walk struggles when the
    between-study scale is much smaller than the fixed effect — the same
    funnel that makes hierarchical models need reparameterisation everywhere
    else.

    Priors, stated rather than buried:

    - :math:`\\beta_t \\sim N(0, 10^2)` on the scale the model is written on, so
      an effect and a log-ED50 are both given a weak prior of the same width.
    - every between-study standard deviation, including the study level when
      ``struct`` has one, :math:`\\sigma \\sim \\text{half-N}(0, 0.5^2)`. On a
      log-effect scale a between-study SD of 1 is already enormous, so this is
      weakly informative by construction rather than tuned here.

    ``coef_struct`` is limited to ``"DIAG"``: a full covariance between a
    curve's coefficients would need its own parameterisation and its own
    validation, and a half-checked one is worse than a stated limit.
    """
    cfg = dict(_MCMC_DEFAULTS)
    if settings:
        unknown = set(settings) - set(cfg)
        if unknown:
            raise ValueError(
                f"unknown 'mcmc' setting(s): {', '.join(sorted(map(str, unknown)))}; "
                f"known keys are {', '.join(sorted(cfg))}")
        cfg.update(settings)
    if coef_struct != "DIAG":
        raise ValueError(
            "inference='mcmc' supports coef_struct='DIAG' only: a full "
            "between-curve covariance needs a parameterisation this engine does "
            "not implement, and an unvalidated one would be worse than the limit")

    # the frequentist fixed-effect fit does all the input validation, the design
    # and the identifiability checks, and gives the sampler somewhere to start
    fe = nma_dose(yi, study=study, treat1=treat1, treat2=treat2, dose1=dose1,
                  dose2=dose2, vi=vi, vcov=vcov, rho=rho, family=family,
                  treatment_specific=treatment_specific, knots=knots,
                  n_knots=n_knots, struct="FE", level=level, anchor=anchor)
    y = np.asarray(yi, dtype=float).ravel()
    k = y.size
    studies = [str(s) for s in np.asarray(study).ravel()]
    studies_unique = list(dict.fromkeys(studies))
    index = {s: i for i, s in enumerate(studies_unique)}
    sid = np.array([index[s] for s in studies])
    a1 = [str(t) for t in np.asarray(treat1).ravel()]
    a2 = [str(t) for t in np.asarray(treat2).ravel()]
    d1 = np.asarray(dose1, dtype=float).ravel()
    d2 = np.asarray(dose2, dtype=float).ravel()
    if vcov is not None:
        V = np.asarray(vcov, dtype=float)
    else:
        v = np.asarray(vi, dtype=float).ravel()
        V = np.diag(v)
        if rho is not None:
            r = float(rho)
            for s in studies_unique:
                sel = np.flatnonzero(sid == index[s])
                if sel.size < 2:
                    continue
                sd = np.sqrt(v[sel])
                V[np.ix_(sel, sel)] = r * np.outer(sd, sd)
                V[sel, sel] = v[sel]
    chol_V, logdet_V = _chol(V)

    blocks_names = fe._blocks()
    p = _n_parameters(family, n_knots=fe.n_knots)
    n_par = len(blocks_names) * p
    block_of = {t: i for i, t in enumerate(blocks_names)} \
        if treatment_specific else None

    def block_index(treat: str) -> int | None:
        if not treatment_specific:
            return 0
        assert block_of is not None
        return block_of.get(treat)

    blk2 = np.full(k, -1, dtype=int)
    blk1 = np.full(k, -1, dtype=int)
    for i in range(k):
        found = block_index(a2[i])
        if found is not None:
            blk2[i] = found
        found = block_index(a1[i])
        if found is not None:
            blk1[i] = found
    if family in _LINEAR_IN_PARAMETERS:
        Xd = np.zeros((k, n_par))
        for i in range(k):
            for treat, dose, sign in ((a2[i], d2[i], 1.0), (a1[i], d1[i], -1.0)):
                b = block_index(treat)
                if b is None:
                    continue
                Xd[i, b * p:(b + 1) * p] += sign * _basis(
                    family, np.array([dose]), knots=fe.knots)[0]
    else:
        Xd = None

    use_study = struct in ("study", "coef+study")
    use_coef = struct in ("coef", "coef+study")
    n_sd = (1 if use_study else 0) + (n_par if use_coef else 0)
    n_z = len(studies_unique) * n_par
    prior_beta = float(cfg["prior_beta"])
    prior_het = float(cfg["prior_het"])

    def unpack(theta: np.ndarray):
        beta = theta[:n_par]
        pos = n_par
        tau = float(np.exp(theta[pos])) if use_study else 0.0
        pos += 1 if use_study else 0
        sd = np.exp(theta[pos:pos + n_par]) if use_coef else np.full(n_par, 0.0)
        pos += n_par if use_coef else 0
        z = theta[pos:pos + n_z].reshape(len(studies_unique), n_par)
        return beta, tau, sd, z

    def mean_vector(beta, sd, z):
        # a study's parameters are the pooled ones plus its own deviation, and
        # ``sd`` is zero unless the curve itself carries a random effect
        per_study = beta[None, :] + z * sd[None, :]
        if Xd is not None:
            return np.sum(Xd * per_study[sid], axis=1)
        mu = np.zeros(k)
        for b in range(len(blocks_names)):
            params_study = per_study[:, b * p:(b + 1) * p]
            up = blk2 == b
            down = blk1 == b
            if up.any():
                mu[up] += _value_rows(family, d2[up], params_study[sid[up]])
            if down.any():
                mu[down] -= _value_rows(family, d1[down], params_study[sid[down]])
        return mu

    def logpost(theta: np.ndarray) -> float:
        beta, tau, sd, z = unpack(theta)
        if not np.all(np.isfinite(theta)):
            return -np.inf
        mu = mean_vector(beta, sd, z)
        resid = y - mu
        value = -0.5 * logdet_V - 0.5 * _quadratic_form(chol_V, resid)
        value += -0.5 * float(beta @ beta) / prior_beta ** 2
        if use_study:
            value += (-0.5 * (tau / prior_het) ** 2
                      + float(np.log(max(tau, 1e-300))))
        if use_coef:
            value += float(np.sum(-0.5 * (sd / prior_het) ** 2
                                  + np.log(np.maximum(sd, 1e-300))))
        value += float(np.sum(-0.5 * z * z))
        return float(value)

    start = np.zeros(n_par + n_sd + n_z)
    start[:n_par] = np.asarray(fe.coef, dtype=float)
    pos = n_par
    if use_study:
        start[pos] = np.log(0.1)
        pos += 1
    if use_coef:
        start[pos:pos + n_par] = np.log(0.1)
    schedule = ([np.arange(n_par)]
                + ([np.arange(n_par, n_par + n_sd)] if n_sd else [])
                + [np.array([n_par + n_sd + i * n_par + j for j in range(n_par)])
                   for i in range(len(studies_unique))])
    post = sample_log_posterior(
        logpost, start,
        names=(fe.names
               + (["log_sd_study"] if use_study else [])
               + ([f"log_sd[{name}]" for name in fe.names] if use_coef else [])
               + [f"z[{s}:{name}]" for s in studies_unique for name in fe.names]),
        n_draws=int(cfg["n_draws"]), warmup=int(cfg["warmup"]),
        chains=int(cfg["chains"]), seed=cfg["seed"],
        init_sd=0.1, blocks=schedule)
    if cfg["check"]:
        post.check(rhat_max=1.05, ess_min=100.0)

    draws = post.draws
    beta_draws = draws[:, :n_par]
    coef = np.mean(beta_draws, axis=0)
    vcov_beta = np.cov(beta_draws, rowvar=False)
    vcov_beta = np.atleast_2d(vcov_beta)
    se = np.sqrt(np.maximum(np.diag(vcov_beta), 0.0))
    pos = n_par
    tau2 = float(np.mean(np.exp(2.0 * draws[:, pos]))) if use_study \
        else float("nan")
    pos += 1 if use_study else 0
    D = None
    if use_coef:
        sd_draws = np.exp(draws[:, pos:pos + n_par])
        D = np.diag(np.mean(sd_draws ** 2, axis=0))
    return NmaDoseResult(
        family=family,
        treatments=fe.treatments,
        curves=list(fe.curves),
        names=list(fe.names),
        coef=[float(v) for v in coef],
        se=[float(v) for v in se],
        vcov=vcov_beta,
        tau2=tau2,
        D=D,
        k=k,
        studies=studies_unique,
        dose_range=dict(fe.dose_range),
        struct=struct,
        coef_struct=coef_struct,
        method="MCMC",
        level=fe.level,
        loglik=float(np.max(post.logpost)),
        anchor=float(anchor),
        n_knots=fe.n_knots,
        knots=fe.knots,
        treatment_specific=bool(treatment_specific),
        inference="mcmc",
        posterior=post,
        fit={"accept": [float(v) for v in post.accept],
             "rhat_max": float(np.max(post.rhat())),
             "ess_min": float(np.min(post.ess()))},
    )
