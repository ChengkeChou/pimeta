"""Two-stage IPD with a non-linear treatment effect modifier.

The univariate :func:`~pimeta.ipd.ipd_two_stage` pools one coefficient.  That
is enough for a linear interaction and wrong for anything else: a restricted
cubic spline interaction is several coefficients from the *same* study, they are
correlated, and pooling them one at a time throws that correlation away — the
curve is then built from variances that do not know about each other, so its
confidence band is not a band.

This module pools the whole treatment-effect block as a vector::

    theta_i = (treat, treat:B_1, ..., treat:B_K)_i
    theta_i ~ N(theta, S_i)          S_i from the study's own fit
    theta   ~ N(mu, Sigma)           Sigma free (``struct="UN"``)

and then evaluates

.. math::

    \\Delta(x) = \\theta_0 + \\sum_k \\theta_k B_k(x),
    \\quad \\operatorname{Var}\\{\\hat\\Delta(x)\\}
    = b(x)' \\operatorname{Cov}(\\hat\\mu) b(x)

with ``b(x) = (1, B_1(x), ..., B_K(x))'``.  The covariance is what carries the
correlation between coefficients; a per-coefficient pooling would set its
off-diagonal to zero.

Two bands are reported, and they answer different questions:

- ``ci_lb`` / ``ci_ub`` -- uncertainty of the **average** curve, the analogue
  of a confidence interval for a pooled effect;
- ``pred_lb`` / ``pred_ub`` -- where a **new study's** curve would fall, which
  adds ``Sigma``.  The difference is usually the honest one to look at when
  asking whether two studies could have produced the same shape.

The basis is built once with common knots and then evaluated in every study, as
Riley et al. (2020) require; a study whose design cannot identify the whole
block is excluded with the reason rather than having zeros filled in, and the
curve reports the covariate range the data actually support.
"""

from __future__ import annotations

import warnings
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from scipy import stats

from .basis import Basis, rcs
from .ipd import StudyFit, _label, linear_model, logistic_model
from .levels import level_to_alpha, level_to_percent

__all__ = ["IpdCurve", "IpdCurveResult", "ipd_two_stage_curve"]


@dataclass
class IpdCurve:
    """The treatment-effect curve on a grid of covariate values."""

    x: np.ndarray
    est: np.ndarray
    se: np.ndarray
    ci_lb: np.ndarray
    ci_ub: np.ndarray
    pred_lb: np.ndarray
    pred_ub: np.ndarray
    outside: np.ndarray = field(repr=False)

    def to_table(self) -> list[dict]:
        return [
            {
                "x": float(self.x[i]),
                "estimate": float(self.est[i]),
                "se": float(self.se[i]),
                "ci_lb": float(self.ci_lb[i]),
                "ci_ub": float(self.ci_ub[i]),
                "pred_lb": float(self.pred_lb[i]),
                "pred_ub": float(self.pred_ub[i]),
                "extrapolated": bool(self.outside[i]),
            }
            for i in range(self.x.size)
        ]


@dataclass
class IpdCurveResult:
    """A joint pooling of a study-level treatment-effect block.

    Attributes:
        names: one name per pooled coefficient, ``["treat", "treat:B[1]", ...]``.
        coef, se, vcov: the pooled coefficients and their covariance.
        tau2: the diagonal of ``Sigma``, one per coefficient.
        Sigma: the between-study covariance matrix of the block.
        k, studies, excluded: contributing studies, studies present, failures.
        support: ``(min, max)`` of the covariate actually observed.
        method, level, test: the second-stage settings.
        per_study: one :class:`~pimeta.ipd.StudyFit` per contributing study.
        fit: the underlying :class:`~pimeta.rma_mv.RmaMvResult`.
    """

    names: list[str]
    coef: list[float]
    se: list[float]
    vcov: np.ndarray = field(repr=False)
    tau2: list[float]
    Sigma: np.ndarray = field(repr=False)
    k: int
    studies: int
    excluded: dict[str, str]
    support: tuple[float, float]
    method: str
    level: float
    test: str
    basis: Basis = field(repr=False)
    per_study: list[StudyFit] = field(repr=False)
    fit: Any = field(repr=False)
    center: float = 0.0

    # -- the curve -------------------------------------------------------- #
    def curve(self, x=None, n: int = 50, level=None) -> IpdCurve:
        """Evaluate the treatment-effect curve.

        Args:
            x: covariate values; ``None`` uses a grid over the observed range.
            n: number of grid points when ``x`` is ``None``.
            level: confidence level, accepted as ``95``, ``0.95`` or ``0.05``;
                defaults to the level the model was fitted at.

        Returns:
            :class:`IpdCurve` with a confidence band for the average curve and
            a prediction band for a new study's curve.
        """
        if x is None:
            lo, hi = self.support
            xv = np.linspace(lo, hi, int(n))
        else:
            xv = np.asarray(x, dtype=float).ravel()
        design = self._design(xv)
        est = design @ np.asarray(self.coef, dtype=float)
        var = np.einsum("gi,ij,gj->g", design, self.vcov, design)
        se = np.sqrt(np.maximum(var, 0.0))
        var_pred = np.einsum(
            "gi,ij,gj->g", design, self.vcov + self.Sigma, design)
        se_pred = np.sqrt(np.maximum(var_pred, 0.0))
        crit = self._crit(level)
        lo, hi = self.support
        return IpdCurve(
            x=xv, est=est, se=se,
            ci_lb=est - crit * se, ci_ub=est + crit * se,
            pred_lb=est - crit * se_pred, pred_ub=est + crit * se_pred,
            outside=(xv < lo) | (xv > hi),
        )

    def _design(self, x) -> np.ndarray:
        """``[1, B_1(x), ..., B_K(x)]`` -- the curve's contrast vector."""
        basis = self.basis.evaluate(np.asarray(x, dtype=float).ravel()
                                    - self.center).values
        return np.column_stack([np.ones(basis.shape[0]), basis])

    def _crit(self, level) -> float:
        alpha = level_to_alpha(self.level if level is None else level)
        if self.test in ("t", "knha", "adhoc"):
            # the model's coefficient tests use their own degrees of freedom;
            # a curve combines all of them, so the smallest is the safe one
            df = float(np.min(np.asarray(self.fit.ddf, dtype=float)))
            return float(stats.t.ppf(1.0 - alpha / 2.0, df))
        return float(stats.norm.ppf(1.0 - alpha / 2.0))

    # -- the two joint tests ---------------------------------------------- #
    def test_interaction(self) -> dict:
        """Does the treatment effect vary with the covariate at all?

        The null is that every ``treat:B_k`` coefficient is zero.  This is the
        test a subgroup analysis is trying to approximate and usually cannot.
        """
        return self._wald(slice(1, self.K + 1), "treatment effect modification")

    def test_nonlinearity(self) -> dict:
        """Is the modification non-linear?

        The null is that the coefficients on the non-linear basis columns are
        zero, leaving ``treat + treat:x``: a straight line.  A spline that is
        not tested this way is a spline that always looks curved.
        """
        if self.K < 2:
            raise ValueError(
                "the basis has no non-linear column, so there is nothing to "
                "test; rcs() needs at least 3 knots")
        return self._wald(slice(2, self.K + 1), "non-linearity")

    def _wald(self, block: slice, what: str) -> dict:
        b = np.asarray(self.coef, dtype=float)[block]
        v = np.asarray(self.vcov, dtype=float)[block, block]
        stat = float(b @ np.linalg.solve(v, b))
        df = int(b.size)
        return {
            "test": what,
            "coefficients": list(b),
            "stat": stat,
            "df": df,
            "pval": float(stats.chi2.sf(stat, df)),
        }

    @property
    def K(self) -> int:
        return len(self.names) - 1

    # -- can the covariance be believed? ----------------------------------- #
    def diagnostics(self) -> dict:
        """Whether the between-study covariance is identifiable at this ``k``.

        A covariance matrix over ``K + 1`` coefficients has ``(K+1)(K+2)/2``
        parameters, and every one of them has to be estimated from whole
        studies.  With a dozen studies the estimate routinely lands on a
        boundary -- a variance at zero, or a correlation at +/-1 -- and the
        likelihood is then flat in that direction: two correct implementations
        can pick different points on it, and the prediction band inherits the
        difference.  This method says whether that happened.

        Returns:
            A dict with ``k``, ``n_variance_parameters``,
            ``studies_per_parameter``, ``tau2_at_zero``, ``rho_at_bound``,
            ``condition`` (of ``Sigma``), ``boundary`` and ``notes`` (a list of
            sentences, empty when nothing is wrong).
        """
        tau2 = np.asarray(self.tau2, dtype=float)
        rho = np.asarray(
            self.fit.rho if self.fit.rho is not None else [], dtype=float)
        n_vc = int((self.fit.parms or (self.fit.p + tau2.size)) - self.fit.p)
        at_zero = [i + 1 for i, v in enumerate(tau2) if v <= 1e-10]
        at_bound = [i + 1 for i, v in enumerate(rho)
                    if abs(abs(v) - 1.0) <= 1e-6]
        singular = np.linalg.svd(np.asarray(self.Sigma, dtype=float),
                                 compute_uv=False)
        condition = (float(singular[0] / singular[-1])
                     if singular[-1] > 0.0 else float("inf"))

        notes = []
        if tau2.size and len(at_zero) == tau2.size:
            notes.append(
                "every between-study variance is zero, so no heterogeneity "
                "was detected and the prediction band necessarily equals the "
                "confidence band; that is an estimate, not a failure")
        elif at_zero:
            notes.append(
                f"between-study variance component(s) {at_zero} sit at zero; "
                "the curve for those directions is fixed, not estimated")
        if at_bound:
            notes.append(
                f"correlation(s) {at_bound} sit on +/-1, which makes Sigma "
                "singular: the likelihood is flat there and another "
                "implementation may stop at a different point on the same "
                "ridge")
        if n_vc and self.k < 3 * n_vc:
            notes.append(
                f"{self.k} studies for {n_vc} variance parameters; consider "
                "struct='HCS' (a common correlation) or fewer basis columns")
        return {
            "k": self.k,
            "n_variance_parameters": n_vc,
            "studies_per_parameter": (self.k / n_vc) if n_vc else float("nan"),
            "tau2_at_zero": at_zero,
            "rho_at_bound": at_bound,
            "condition": condition,
            "boundary": bool(at_zero or at_bound),
            "notes": notes,
        }

    # -- reporting -------------------------------------------------------- #
    def coef_table(self) -> list[dict]:
        return [
            {"term": n, "estimate": self.coef[i], "se": self.se[i]}
            for i, n in enumerate(self.names)
        ]

    def to_dict(self) -> dict:
        return {
            "names": list(self.names),
            "coef": list(self.coef),
            "se": list(self.se),
            "tau2": list(self.tau2),
            "k": self.k,
            "studies": self.studies,
            "excluded": dict(self.excluded),
            "support": [float(self.support[0]), float(self.support[1])],
            "method": self.method,
            "level": self.level,
            "test": self.test,
            "basis": self.basis.to_dict(),
            "center": float(self.center),
            "per_study": [s.to_dict() for s in self.per_study],
        }


def ipd_two_stage_curve(y, *, study, treat, covariate, basis=None,
                        n_knots: int = 4, knots=None, center=None,
                        struct: str = "UN",
                        family: str = "binomial",
                        covariates=None, method: str = "REML",
                        level: float = 0.95, test: str = "z",
                        min_events: int = 1) -> IpdCurveResult:
    """Pool a treatment-by-covariate spline jointly across studies.

    Args:
        y: the outcome, one entry per participant.
        study: the study label, one entry per participant.
        treat: the treatment indicator (0/1), one entry per participant.
        covariate: the effect modifier, one entry per participant.
        basis: a :class:`~pimeta.basis.Basis` to use instead of building one;
            ``ma.fp`` gives fractional polynomials, ``ma.rcs`` splines.
        n_knots: knots for the default restricted cubic spline.
        knots: explicit knot positions.  The same knots are used in every study,
            which is what makes the coefficients comparable.
        center: value the covariate is centred at, by default its median.  The
            linear coefficient of the block is the treatment effect *at this
            value*, and without centring it is the effect at covariate = 0,
            which is usually an extrapolation (age 0, dose 0) and can have a
            standard error an order of magnitude too wide.  Pass ``0`` to
            switch centring off.
        struct: structure of the between-study covariance.  ``"UN"`` (default)
            estimates every variance and correlation, which needs a lot of
            studies; ``"HCS"`` gives each coefficient its own variance but one
            common correlation, and is what to use when ``k`` is small.  See
            :meth:`IpdCurveResult.diagnostics`.
        family: ``"binomial"`` (logistic) or ``"gaussian"`` (linear).
        covariates: further participant-level covariates, adjusted for but not
            pooled.
        method: τ² estimator for the second stage.
        level: confidence level.
        test: ``"z"`` (default) or ``"t"``.  ``"knha"``/``"adhoc"`` are refused:
            their degrees of freedom are computed from the stacked rows rather
            than from the studies, which makes the band narrower, not wider.
        min_events: studies with fewer events are excluded (no finite logistic
            estimate otherwise).

    Returns:
        :class:`IpdCurveResult`.

    Raises:
        ValueError: when the inputs disagree in length, no study can be fitted,
            or the basis has fewer than two columns.

    Example:
        >>> fit = ma.ipd_two_stage_curve(  # doctest: +SKIP
        ...     y, study=study, treat=treat, covariate=age)
        >>> fit.test_nonlinearity()["pval"]  # doctest: +SKIP
        >>> fit.curve([45, 55, 65]).to_table()  # doctest: +SKIP
    """
    from .rma_mv import rma_mv

    y = np.asarray(y, dtype=float).ravel()
    labels = [_label(s) for s in np.asarray(study).ravel()]
    treat = np.asarray(treat, dtype=float).ravel()
    mod = np.asarray(covariate, dtype=float).ravel()
    if not (y.size == len(labels) == treat.size == mod.size):
        raise ValueError(
            "'y', 'study', 'treat' and 'covariate' must be the same length")
    if not np.all(np.isfinite(mod)):
        raise ValueError("'covariate' contains missing or infinite values")

    if covariates is None:
        cov_map: dict[str, np.ndarray] = {}
    elif isinstance(covariates, Mapping):
        cov_map = {str(k): np.asarray(v, dtype=float).ravel()
                   for k, v in covariates.items()}
    else:
        cov_map = {f"cov{i + 1}": np.asarray(v, dtype=float).ravel()
                   for i, v in enumerate(covariates)}
    for name, values in cov_map.items():
        if values.size != y.size:
            raise ValueError(
                f"covariate {name!r} needs one value per participant")

    fam = str(family).lower()
    if fam not in ("binomial", "gaussian"):
        raise ValueError(
            f"unknown family {family!r}; use 'binomial' or 'gaussian'")
    if str(test).lower() in ("knha", "adhoc"):
        # `rma_mv`'s residual degrees of freedom count *rows*, and a stacked
        # multivariate fit has K rows per study -- so the t reference is almost
        # z while the variance is scaled by s2w.  Simulated coverage with
        # test="knha" was 0.31 to 0.51 for a nominal 0.95, i.e. worse than the
        # z interval it was meant to fix.  Refusing is better than shipping it.
        raise ValueError(
            "test='knha' is not available here: its degrees of freedom are "
            "computed from the stacked rows rather than from the studies, so "
            "the interval is narrower than the z one instead of wider. Use "
            "test='z' (default) or 't', and see "
            "tools/coverage_ipd_curve.py for what the band actually covers")

    if basis is None:
        if center is None:
            center = float(np.median(mod))
        z = mod - float(center)
        if knots is not None:
            knots = np.asarray(knots, dtype=float).ravel() - float(center)
        basis = rcs(z, nk=n_knots, knots=knots, name="rcs")
    else:
        center = 0.0 if center is None else float(center)
    # The basis is defined once and evaluated in every study.  Building it per
    # study would give each one its own knots, and then the coefficients pooled
    # in the second stage would not be the same quantity.
    shift = float(center)
    B_all = np.asarray(basis.evaluate(mod - shift).values, dtype=float)
    K = int(B_all.shape[1])
    if K < 1:
        raise ValueError("the basis has no columns")

    names = ["treat"] + [f"treat:{n}" for n in basis.names]
    block = slice(1, K + 2)          # design columns 1..K+1

    groups = list(dict.fromkeys(labels))
    per_study: list[StudyFit] = []
    excluded: dict[str, str] = {}
    for g in groups:
        rows = np.array([i for i, lab in enumerate(labels) if lab == g])
        yg = y[rows]
        tg = treat[rows]
        Bg = B_all[rows]
        design_names = (["(Intercept)", "treat"]
                        + list(basis.names) + list(cov_map))
        cols = [np.ones(rows.size), tg, Bg * tg[:, None]]
        cols += [c[rows] for c in cov_map.values()]
        X = np.column_stack(cols)

        events = None
        if fam == "binomial":
            events = int(np.sum(yg == 1.0))
            if events < int(min_events):
                excluded[g] = (
                    f"{events} event(s) in {yg.size} participants; "
                    f"fewer than min_events={int(min_events)}")
                continue
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
            study=g, n=int(rows.size), events=events,
            coefnames=design_names,
            b=[float(v) for v in b],
            se=[float(np.sqrt(max(vb[i, i], 0.0))) for i in range(b.size)],
            vb=np.asarray(vb, dtype=float), term=1, converged=converged,
        ))

    if len(per_study) < 2:
        raise ValueError(
            "at least two studies must be fittable for a joint pooling "
            f"(excluded: {excluded})")

    # ---- second stage: one mean per coefficient, free covariance ----------
    n_coef = K + 1
    stacked = np.concatenate(
        [np.asarray(s.b, dtype=float)[block] for s in per_study])
    blocks = [np.asarray(s.vb, dtype=float)[block, block] for s in per_study]
    V = np.zeros((stacked.size, stacked.size))
    offset = 0
    for Si in blocks:
        m = Si.shape[0]
        V[offset:offset + m, offset:offset + m] = Si
        offset += m

    index = np.tile(np.arange(n_coef), len(per_study))
    study_labels = [s.study for s in per_study for _ in range(n_coef)]
    X_one = np.zeros((stacked.size, n_coef))
    X_one[np.arange(stacked.size), index] = 1.0
    data = {"coef": index.tolist(), "study": study_labels}

    fit = rma_mv(stacked, V=V, mods=X_one, data=data,
                 random="~ coef | study", struct=struct, intercept=False,
                 method=method, level=level_to_percent(level), test=test)

    coef = [float(v) for v in fit.beta]
    vcov = np.asarray(fit.vb, dtype=float)
    G, gnames = fit.g_matrix()
    if G is None:                                    # pragma: no cover
        raise ValueError("the second stage has no between-study covariance")
    Sigma = np.asarray(G, dtype=float)
    tau2 = [float(v) for v in np.diag(Sigma)]

    result = IpdCurveResult(
        names=list(names),
        coef=coef,
        se=[float(v) for v in fit.se],
        vcov=vcov,
        tau2=tau2,
        Sigma=Sigma,
        k=len(per_study),
        studies=len(groups),
        excluded=excluded,
        support=(float(mod.min()), float(mod.max())),
        method=str(fit.method),
        level=level_to_percent(level),
        test=str(getattr(fit, "test", "z")),
        basis=basis,
        center=shift,
        per_study=per_study,
        fit=fit,
    )
    # A boundary estimate is not an error, but it is not something to discover
    # three sections later either: on the boundary the likelihood is flat in
    # that direction and the prediction band is a choice of stopping point.
    problems = result.diagnostics()["notes"]
    if problems:
        warnings.warn(
            "the between-study covariance of the treatment block is not well "
            "identified: " + " | ".join(problems),
            RuntimeWarning, stacklevel=2)
    return result
