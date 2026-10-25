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
    b = getattr(fit, "b", None)
    se = getattr(fit, "se", None)
    if b is None or se is None:
        return None
    parts = [np.atleast_1d(np.asarray(b, dtype=float)).ravel(),
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


_CHECKS = (_check_non_finite_estimate, _check_tau2_zero,
           _check_variance_component, _check_ill_conditioned,
           _check_boundary_correlation, _check_studies_dropped,
           _check_zero_weight_study, _check_few_studies,
           _check_posterior_equals_prior)


def software_versions() -> dict:
    """The versions a saved result needs in order to be reproducible later."""
    import platform

    import scipy

    from . import __version__ as pimeta_version

    return {"pimeta": pimeta_version, "python": platform.python_version(),
            "numpy": np.__version__, "scipy": scipy.__version__}


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
