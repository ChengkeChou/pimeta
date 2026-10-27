"""Structured diagnostics: what is wrong with *this* fit, in a form a program
can read.

The distinction that matters here is between a simulation study and a
diagnostic. A simulation answers "is the method reliable across repeated
samples?"; a diagnostic answers "should I believe the number in front of me?".
This module does the second, and it returns records rather than printing
warnings, because a warning cannot be exported, counted, or aggregated.

Why it exists, in one example
-----------------------------

While building :mod:`pimeta.robma` a marginal likelihood accumulated from
``log(1)`` instead of ``log(0)``, which gave every value of :math:`\\tau` a unit
of probability before any likelihood was added. The fit returned a posterior
mean, a standard deviation, an interval and model probabilities, all finite and
all plausible-looking, and not one warning. The parameter had simply stopped
being informed by the data.

No general-purpose test catches that, but a specific one does: compare the
posterior standard deviation with the prior standard deviation. If they are the
same, the data did not update the parameter, whatever the reason. That is
:func:`diagnose` code ``posterior_equals_prior``, and it is the reason this
module is worth having.

Scope of the first release
--------------------------

Three checks, chosen because they are (a) computable from a fitted object
without refitting and (b) failure modes that have actually been met here.
Coverage of every model is deliberately not attempted: a diagnostic that is
wrong is worse than a diagnostic that is missing, because it is believed.

Nothing is attached to the result classes yet. ``diagnose`` is a function so
that adding it cannot change any existing behaviour, and so that a check can be
dropped without touching the models.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

__all__ = ["Diagnostic", "SCHEMA_VERSION", "diagnose", "software_versions"]

#: Bumped whenever a field is renamed or removed from a result's ``to_dict``.
#: Adding a field does not bump it, so a reader can tell a missing key from a
#: changed one.
SCHEMA_VERSION = 1

#: Severities, worst first. ``error`` means the numbers are not reportable as if
#: valid, ``warning`` means reportable with a caveat, ``info`` is worth knowing.
SEVERITIES = ("error", "warning", "info")


@dataclass(frozen=True)
class Diagnostic:
    """One thing worth knowing about a fit.

    Attributes:
        code: a stable machine-readable identifier, e.g. ``tau2_zero``. Codes
            are the contract; messages may be reworded.
        category: one of ``identifiability``, ``optimisation``, ``boundary``,
            ``data`` or ``inference``.
        severity: ``error``, ``warning`` or ``info``.
        message: what was found, with the numbers that show it.
        action: what a user should do about it, or why there is nothing to do.
    """

    code: str
    category: str
    severity: str
    message: str
    action: str

    def __post_init__(self) -> None:
        if self.severity not in SEVERITIES:
            raise ValueError(
                f"'severity' must be one of {', '.join(SEVERITIES)}")
        if not self.code or not self.code.replace("_", "").isalnum():
            raise ValueError("'code' must be a snake_case identifier")

    def to_dict(self) -> dict:
        return {"code": self.code, "category": self.category,
                "severity": self.severity, "message": self.message,
                "action": self.action}

    def __str__(self) -> str:
        return f"[{self.severity}] {self.code}: {self.message}"


@dataclass
class Diagnosis:
    """The diagnostics for one fit, with the worst severity first."""

    items: list[Diagnostic] = field(default_factory=list)

    def __iter__(self):
        return iter(self.items)

    def __len__(self) -> int:
        return len(self.items)

    def codes(self) -> list[str]:
        return [d.code for d in self.items]

    @property
    def worst(self) -> str | None:
        """The highest severity present, or ``None`` for a clean fit."""
        for level in SEVERITIES:
            if any(d.severity == level for d in self.items):
                return level
        return None

    def of(self, severity: str) -> list[Diagnostic]:
        return [d for d in self.items if d.severity == severity]

    def to_dict(self) -> dict:
        return {"worst": self.worst, "items": [d.to_dict() for d in self.items]}

    def __str__(self) -> str:
        if not self.items:
            return "no diagnostics"
        return "\n".join(str(d) for d in self.items)


# --------------------------------------------------------------------------- #
# individual checks
# --------------------------------------------------------------------------- #
def _check_tau2_zero(fit) -> Diagnostic | None:
    tau2 = getattr(fit, "tau2", None)
    if tau2 is None:
        return None
    # rma.mv exposes tau2 as a list or a matrix, so a bare float() here would
    # make diagnose() raise on a whole model family - worse than not diagnosing
    values = np.atleast_1d(np.asarray(tau2, dtype=float)).ravel()
    values = values[np.isfinite(values)]
    if values.size == 0 or float(np.max(values)) > 1e-10:
        return None
    return Diagnostic(
        code="tau2_zero", category="boundary", severity="warning",
        message="every between-study variance is estimated at "
                f"{float(np.max(values)):.3g}",
        action="the random-effects model has collapsed to a fixed-effect model; "
               "report this rather than the random-effects interval, and note "
               "that the between-study variance is not identified from these data")


def _check_variance_component(fit) -> Diagnostic | None:
    """A variance component pinned at zero in a model that has several."""
    sigma2 = getattr(fit, "sigma2", None)
    if sigma2 is None:
        return None
    values = np.atleast_1d(np.asarray(sigma2, dtype=float)).ravel()
    if values.size < 2:
        return None                      # a single component is tau2_zero
    zero = int(np.sum(values <= 1e-10))
    if zero == 0:
        return None
    return Diagnostic(
        code="variance_component_zero", category="boundary", severity="warning",
        message=f"{zero} of {values.size} variance components are estimated at "
                "zero",
        action="the corresponding random term is not needed by these data; "
               "report the model without it rather than the boundary estimate, "
               "and do not compare information criteria across the boundary")


def _check_ill_conditioned(fit) -> Diagnostic | None:
    """A fixed-effect covariance that is close to singular."""
    vb = getattr(fit, "vb", None)
    if vb is None or np.size(vb) == 0:
        return None
    matrix = np.asarray(vb, dtype=float)
    if matrix.ndim != 2 or matrix.shape[0] < 2:
        return None
    cond = float(np.linalg.cond(matrix))
    if not np.isfinite(cond) or cond < 1e10:
        return None
    return Diagnostic(
        code="ill_conditioned", category="identifiability", severity="warning",
        message=f"the coefficient covariance has condition number {cond:.3g}",
        action="some coefficients are nearly linear combinations of the others; "
               "their standard errors are large for a structural reason, not "
               "because the data are uninformative, and the point estimates "
               "should not be interpreted separately")


def _check_boundary_correlation(fit) -> Diagnostic | None:
    """A correlation estimated at +/-1, where the likelihood is not regular."""
    rho = getattr(fit, "rho", None)
    if rho is None:
        return None
    values = np.atleast_1d(np.asarray(rho, dtype=float)).ravel()
    values = values[np.isfinite(values)]
    if values.size == 0 or float(np.min(1.0 - np.abs(values))) > 1e-6:
        return None
    return Diagnostic(
        code="boundary_correlation", category="boundary", severity="warning",
        message="a correlation is estimated at the edge of its range "
                f"(rho = {float(values[np.argmax(np.abs(values))]):+.6f})",
        action="the usual interval and test are not valid at the boundary; "
               "report the profile or a likelihood-ratio interval instead")


def _check_non_finite_estimate(fit) -> Diagnostic | None:
    """A coefficient or standard error that is not a number."""
    # Families name their coefficient vector differently: `rma` uses `b`,
    # `matreg` uses `beta`. Reading only `b` made the check silently skip whole
    # model families, which is the failure mode this package keeps meeting.
    raw = None
    for attribute in ("b", "beta", "coef"):
        raw = getattr(fit, attribute, None)
        if raw is not None:
            break
    se = getattr(fit, "se", None)
    if raw is None or se is None:
        return None
    parts = [np.atleast_1d(np.asarray(raw, dtype=float)).ravel(),
             np.atleast_1d(np.asarray(se, dtype=float)).ravel()]
    values = np.concatenate([p for p in parts if p.size])
    if values.size == 0 or np.all(np.isfinite(values)):
        return None
    bad = int(np.sum(~np.isfinite(values)))
    return Diagnostic(
        code="non_finite_estimate", category="optimisation", severity="error",
        message=f"{bad} of {values.size} estimates are not finite",
        action="do not report these numbers. A non-finite coefficient or "
               "standard error means the fit failed, not that the effect is "
               "large; inspect the design matrix and the study-level variances")


def _check_zero_weight_study(fit) -> Diagnostic | None:
    """A study that received no weight at all."""
    weights = getattr(fit, "weights", None)
    if weights is None:
        return None
    values = np.atleast_1d(np.asarray(weights, dtype=float)).ravel()
    values = values[np.isfinite(values)]
    if values.size < 2 or float(np.min(values)) > 0.0:
        return None
    zero = int(np.sum(values <= 0.0))
    return Diagnostic(
        code="zero_weight_study", category="data", severity="warning",
        message=f"{zero} of {values.size} studies carry zero weight",
        action="those studies contribute nothing to the estimate, which "
               "usually means an infinite or missing variance; report them "
               "rather than implying the whole sample was used")


def _check_negative_variance(fit) -> Diagnostic | None:
    """A negative sampling variance, which the fitter does not refuse.

    Measured, not assumed: ``rma`` accepts ``vi = -0.01`` alongside five
    ordinary variances and returns ``b = 0.2091``, ``se = 0.0786`` -- a
    plausible-looking meta-analysis computed from weights ``1 / (vi + tau2)``
    that are not all positive. Nothing in the output says so. A negative
    variance is impossible for every effect-size measure, so its presence means
    the input is wrong, not that the data are unusual.
    """
    vi = getattr(fit, "vi", None)
    if vi is None:
        return None
    values = np.atleast_1d(np.asarray(vi, dtype=float)).ravel()
    values = values[np.isfinite(values)]
    if values.size == 0 or float(np.min(values)) >= 0.0:
        return None
    bad = int(np.sum(values < 0.0))
    return Diagnostic(
        code="negative_variance", category="data", severity="error",
        message=(f"{bad} of {values.size} sampling variances are negative "
                 f"(smallest {float(np.min(values)):.6g})"),
        action="do not report these numbers. A negative variance makes the "
               "weight 1/(vi + tau2) non-positive, so the estimate is not a "
               "weighted mean of anything; the usual cause is an effect size "
               "computed from a degenerate table or a mis-specified column")


def _check_zero_variance(fit) -> Diagnostic | None:
    """A sampling variance of exactly zero: a study treated as known exactly."""
    vi = getattr(fit, "vi", None)
    if vi is None:
        return None
    values = np.atleast_1d(np.asarray(vi, dtype=float)).ravel()
    values = values[np.isfinite(values)]
    if values.size == 0 or not np.any(values == 0.0):
        return None
    zero = int(np.sum(values == 0.0))
    return Diagnostic(
        code="zero_variance", category="data", severity="warning",
        message=(f"{zero} of {values.size} sampling variances are exactly zero, "
                 "so those studies have weight 1/tau2 and take the whole "
                 "weight when the heterogeneity is estimated at zero"),
        action="report this: a study with no sampling error is either a "
               "data-entry error or a deliberate assumption, and either way "
               "the pooled estimate is driven by it")


def _check_not_converged(fit) -> Diagnostic | None:
    """The optimiser says it did not converge.

    Everything else in the fit is then a snapshot of a search that stopped early,
    which is a different object from an estimate.
    """
    converged = getattr(fit, "converged", None)
    if converged is None or converged:
        return None
    return Diagnostic(
        code="not_converged", category="optimisation", severity="error",
        message="the optimiser reports that it did not converge",
        action="do not report these numbers as estimates. Increase the iteration "
               "limit, change the starting values, or simplify the model; a "
               "non-converged fit is a failed search, not a wide interval")


def _check_studies_dropped(fit) -> Diagnostic | None:
    """Studies removed before fitting, which changes what the estimate covers.

    ``rma`` already emits a ``MissingDataWarning``; this is the same condition as
    a record, so an exported result says how many studies it actually used
    instead of leaving that to whoever read the console.
    """
    k_all = getattr(fit, "k_all", None)
    k = getattr(fit, "k", None)
    if k_all is None or k is None or int(k_all) <= int(k):
        return None
    dropped = int(k_all) - int(k)
    return Diagnostic(
        code="studies_dropped", category="data", severity="warning",
        message=f"{dropped} of {int(k_all)} studies were dropped before "
                "fitting",
        action="the estimate covers the remaining studies only; report which "
               "were dropped and why, because a result that silently covers "
               "fewer studies than the review contains is not the same result")


def _check_few_studies(fit) -> Diagnostic | None:
    """Few studies, in a model that has to estimate a heterogeneity from them.

    The condition is about identifying the between-study variance, so a result
    that does not estimate one -- a heterogeneity test, a regression table, a
    rankogram -- is not a ``few_studies`` problem however small its ``k``. Firing
    there was the first false positive this rollout produced, on ``matreg`` with
    two variables: the message talked about a variance the model never had.

    A Bayesian fit stores the parameter as ``tau`` rather than ``tau2``, and it
    estimates the same thing, so both count.
    """
    if (getattr(fit, "tau2", None) is None
            and getattr(fit, "tau", None) is None):
        return None
    k = getattr(fit, "k", None)
    if k is None or int(k) >= 5:
        return None
    return Diagnostic(
        code="few_studies", category="identifiability", severity="warning",
        message=f"only {int(k)} studies: a between-study variance estimated "
                "from so few is barely identified",
        action="treat the heterogeneity estimate as a guess, prefer a "
               "fixed-effect analysis or a prior on tau, and do not read the "
               "prediction interval as calibrated")


def _check_posterior_equals_prior(fit) -> Diagnostic | None:
    """Posterior spread equal to the prior's means the data did not update it.

    This is the check that would have caught the ``log(1)`` bug in
    :mod:`pimeta.robma`, and it needs no knowledge of why the update failed.
    """
    tau = getattr(fit, "tau", None)
    priors = getattr(fit, "priors", None)
    if not isinstance(tau, dict) or not isinstance(priors, dict):
        return None
    prior_sd = priors.get("tau_sd")
    post_sd = tau.get("sd")
    if prior_sd is None or post_sd is None:
        return None
    prior_sd, post_sd = float(prior_sd), float(post_sd)
    if prior_sd <= 0.0:
        return None
    # the posterior is a mixture with a point mass at zero, and the continuous
    # part is a half-normal, whose own SD is prior_sd * sqrt(1 - 2/pi), not
    # prior_sd. A parameter the data never touched therefore still has
    #   sd = prior_sd * sqrt((1 - p) - (1 - p)^2 * 2/pi),
    # and comparing against anything simpler never fires
    point = float(tau.get("point_mass", 0.0))
    rest = max(1.0 - point, 1e-12)
    expected = prior_sd * float(np.sqrt(max(rest - rest ** 2 * 2.0 / np.pi,
                                            1e-12)))
    if expected <= 0.0:
        return None
    if abs(post_sd - expected) / expected >= 0.05:
        return None
    return Diagnostic(
        code="posterior_equals_prior", category="optimisation",
        severity="warning",
        message=f"the posterior SD of tau is {post_sd:.4g} against a prior SD "
                f"of {prior_sd:.4g}: the data did not update this parameter",
        action="do not report this parameter as estimated. Either the data are "
               "uninformative about it, or the likelihood is being evaluated "
               "somewhere it should not be")


def _check_near_rank_deficient(fit) -> Diagnostic | None:
    """A design that is full rank but nearly collinear.

    The *exact* case is not diagnosable: `rma` raises ``ValueError`` when the
    design matrix is rank deficient, so no fitted object can carry it. The
    near case is the one that reaches a fit, and it is the one that produces
    standard errors nobody can interpret. The threshold is a condition number of
    ``1e6`` on the design matrix itself, which is where double precision starts
    to lose about six digits in the projection.
    """
    design = getattr(fit, "X", None)
    if design is None or np.size(design) == 0:
        return None
    matrix = np.asarray(design, dtype=float)
    if matrix.ndim != 2 or matrix.shape[1] < 2:
        return None
    cond = float(np.linalg.cond(matrix))
    if not np.isfinite(cond) or cond <= 1e6:
        return None
    return Diagnostic(
        code="near_rank_deficient", category="identifiability",
        severity="warning",
        message=f"the design matrix has condition number {cond:.3g}",
        action="the moderators are nearly collinear: the fit exists, but the "
               "coefficients and their standard errors are not separately "
               "interpretable. Drop or combine the offending columns")


def _check_flat_likelihood(fit) -> Diagnostic | None:
    """A heterogeneity estimate that is not a maximum in the curvature sense.

    With the other coefficients profiled out, ``tau2`` is a single parameter, so
    "the Hessian is not positive definite" reduces to a scalar: the second
    derivative of the log-likelihood must be negative at the estimate. A
    non-negative curvature at an interior estimate means the likelihood has no
    proper maximum there -- the optimiser stopped on a flat or rising surface --
    and the standard error of ``tau2``, which is computed from that curvature,
    is meaningless. At ``tau2 = 0`` the curvature test does not apply; that is
    the boundary case and :func:`_check_tau2_zero` covers it.
    """
    tau2 = getattr(fit, "tau2", None)
    yi = getattr(fit, "yi", None)
    vi = getattr(fit, "vi", None)
    if tau2 is None or yi is None or vi is None:
        return None
    value = float(np.ravel(np.asarray(tau2, dtype=float))[0])
    if not np.isfinite(value) or value <= 1e-8:
        return None                      # boundary, not a failure to identify
    y = np.asarray(yi, dtype=float).ravel()
    v = np.asarray(vi, dtype=float).ravel()
    design = getattr(fit, "X", None)
    if design is None or y.size < 3:
        return None
    X = np.asarray(design, dtype=float)
    method = str(getattr(fit, "method", "REML")).upper()
    if method not in ("REML", "ML"):
        return None
    from .pooling import _reml_ml_loglik

    step = max(1e-4, 1e-3 * value)
    try:
        # `_reml_ml_loglik` returns the log-likelihood itself, not its negative
        ll0 = float(_reml_ml_loglik(y, v, X, value, method))
        ll_up = float(_reml_ml_loglik(y, v, X, value + step, method))
        ll_dn = float(_reml_ml_loglik(y, v, X, max(value - step, 1e-12),
                                      method))
    except Exception:                    # noqa: BLE001 - a check must not raise
        return None
    curvature = (ll_up - 2.0 * ll0 + ll_dn) / (step ** 2)
    if not np.isfinite(curvature) or curvature < 0.0:
        return None
    return Diagnostic(
        code="flat_likelihood", category="optimisation", severity="warning",
        message=f"the log-likelihood is flat or concave-up at tau2 = "
                f"{value:.4g} (curvature {curvature:+.3g})",
        action="the heterogeneity estimate is not a maximum of the surface the "
               "optimiser searched: treat it as unestimated and do not quote "
               "its standard error")


def _fisher_report(fit) -> tuple[int, bool] | None:
    """Iterations the ML/REML heterogeneity search used, and whether it ran out."""
    method = str(getattr(fit, "method", "")).upper()
    if method not in ("ML", "REML"):
        return None
    yi = getattr(fit, "yi", None)
    vi = getattr(fit, "vi", None)
    design = getattr(fit, "X", None)
    if yi is None or vi is None or design is None:
        return None
    from .pooling import tau2_iterations

    try:
        return tau2_iterations(np.asarray(yi, dtype=float),
                               np.asarray(vi, dtype=float),
                               np.asarray(design, dtype=float), method)
    except Exception:                    # noqa: BLE001 - a check must not raise
        return None


def _check_max_iter_reached(fit) -> Diagnostic | None:
    """The heterogeneity search stopped at its iteration limit.

    ``not_converged`` covers an optimiser that says so; this is the other half,
    a search that hit its budget and returned its last iterate without any flag.
    The estimate is then wherever the last step landed, and the standard error
    of ``tau^2`` comes from a curvature that was never reached.
    """
    report = _fisher_report(fit)
    if report is None:
        return None
    n_iter, hit = report
    if not hit:
        return None
    return Diagnostic(
        code="max_iter_reached", category="optimisation", severity="warning",
        message=f"the Fisher scoring for tau^2 stopped at its {n_iter}-iteration "
                "limit instead of at the convergence threshold",
        action="the estimate is whatever the last step produced. Raise the "
               "iteration limit, or fix tau^2 at a defensible value; do not "
               "read the standard error of tau^2 as if the search had converged")


def _check_multi_modal_likelihood(fit) -> Diagnostic | None:
    """More than one local maximum on the tau^2 profile.

    Every optimiser returns *one* peak; the question this answers is whether
    there was another.  The profile is scanned on a fixed grid derived from the
    data (see :func:`pimeta.pooling.profile_modes`), and only interior maxima
    are counted, because ``tau^2 = 0`` is the boundary and
    ``_check_tau2_zero`` already reports it.
    """
    method = str(getattr(fit, "method", "")).upper()
    if method not in ("ML", "REML"):
        return None
    yi = getattr(fit, "yi", None)
    vi = getattr(fit, "vi", None)
    design = getattr(fit, "X", None)
    if yi is None or vi is None or design is None:
        return None
    from .pooling import profile_modes

    try:
        peaks = profile_modes(np.asarray(yi, dtype=float),
                              np.asarray(vi, dtype=float),
                              np.asarray(design, dtype=float), method)
    except Exception:                    # noqa: BLE001 - a check must not raise
        return None
    if len(peaks) < 2:
        return None
    shown = ", ".join(f"{peak:.4g}" for peak in peaks)
    return Diagnostic(
        code="multi_modal_likelihood", category="optimisation",
        severity="warning",
        message=f"the tau^2 profile has {len(peaks)} local maxima (at {shown})",
        action="the likelihood does not single out one heterogeneity: report "
               "that the estimate depends on where the search started, and "
               "consider the models at both peaks instead of only the one that "
               "was returned")


_CHECKS = (_check_non_finite_estimate, _check_negative_variance,
           _check_not_converged, _check_max_iter_reached, _check_tau2_zero,
           _check_variance_component, _check_ill_conditioned,
           _check_near_rank_deficient, _check_boundary_correlation,
           _check_studies_dropped, _check_zero_weight_study, _check_zero_variance,
           _check_flat_likelihood, _check_multi_modal_likelihood,
           _check_few_studies, _check_posterior_equals_prior)


def software_versions() -> dict:
    """The versions a saved result needs in order to be reproducible later."""
    import platform

    import scipy

    from . import __version__ as pimeta_version

    return {"pimeta": pimeta_version, "python": platform.python_version(),
            "numpy": np.__version__, "scipy": scipy.__version__}


def attach_diagnostics(cls):
    """Give a result class the standard ``diagnostics`` property.

    Used as a class decorator, after ``@dataclass``::

        @dataclass
        @attach_diagnostics
        class SomeResult:
            ...

    One decorator rather than a copy of the same six lines in twenty files,
    because copies drift: three result classes already had this property written
    by hand, and one of them had a ``diagnostics()`` method returning a dict
    instead. A class only gets the decorator where ``diagnose``'s checks are
    meaningful for it — a predictor table or a heterogeneity test is not a
    meta-analysis fit, and firing ``few_studies`` on one would be a diagnostic
    that is wrong, which is worse than one that is missing.

    The property never raises: every check reads with ``getattr`` and skips what
    the object does not carry.
    """

    def diagnostics(self) -> Diagnosis:
        """The structured diagnostics for this fit.

        A property rather than a stored field, so a check can be added or
        removed without changing what the fit computes. Never raises: each
        check reads what the object carries and skips the rest.
        """
        return diagnose(self)

    cls.diagnostics = property(diagnostics)
    return cls


def diagnose(fit) -> Diagnosis:
    """Run every applicable check against a fitted object.

    Args:
        fit: any fitted result. Checks that do not apply — because the object
            does not carry the state they read — are skipped rather than
            guessed at.

    Returns:
        :class:`Diagnosis`, worst severity first, empty when nothing fired.

    Example:
        >>> fit = ma.rma(yi, vi)
        >>> for d in ma.diagnose(fit):          # doctest: +SKIP
        ...     print(d)
    """
    items = [d for d in (check(fit) for check in _CHECKS) if d is not None]
    items.sort(key=lambda d: SEVERITIES.index(d.severity))
    return Diagnosis(items=items)
