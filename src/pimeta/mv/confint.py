"""Profile-likelihood intervals for the variance components of ``rma_mv``.

metafor reports no standard error for ``tau2`` or ``rho`` in an ``rma.mv``
fit, and that is not an omission: the asymptotic approximation is unreliable
for a variance at its zero boundary and for a correlation at +/-1.  What the
parameters do admit is a *profile* interval -- fix the parameter at a value,
re-estimate everything else, and read off the values where the likelihood has
dropped by half a chi-square quantile::

    2 * (max loglik - profile loglik) = qchisq(1 - alpha, 1)

``metafor::confint`` does exactly this and so does :func:`confint_rma_mv`, down
to Brent's method from ``uniroot`` and its tolerance, so the two endpoints can
be compared rather than merely resembled.

Two properties of the answer are worth stating, because they are properties of
profile likelihood rather than of this implementation:

- the interval is **not symmetric** about the estimate;
- when the profile is still above the threshold at a bound, the endpoint *is*
  that bound.  A correlation of 1 is the common case -- the data are
  consistent with perfect correlation over that whole side -- and the result
  says so with ``at_bound_ub`` rather than pretending to have found a limit.

A flat profile or several peaks is what unidentifiable or badly converged
models look like; :attr:`ConfintMvResult.curve` holds every evaluated point so
that this can be seen instead of guessed.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy import stats
from scipy.optimize import minimize, minimize_scalar

from .._numeric import UNIROOT_TOL, zeroin
from ..levels import level_to_alpha, level_to_percent
from ..rma_mv import RmaMvResult, _mv_loglik, _mv_optimise

__all__ = ["ConfintMvResult", "confint_rma_mv"]


@dataclass
class ConfintMvResult:
    """A profile-likelihood interval for one variance component."""

    kind: str                       # "tau2" or "rho"
    index: int                      # 1-based, as metafor
    estimate: float
    ci_lb: float
    ci_ub: float
    level: float                    # percent, whichever spelling was given
    ll_max: float
    crit: float                     # half the chi-square(1) quantile
    method: str                     # "REML" or "ML"
    at_bound_lb: bool = False
    at_bound_ub: bool = False
    curve: list = field(default_factory=list, repr=False)

    @property
    def label(self) -> str:
        return f"{self.kind}.{self.index}" if self.index > 1 else self.kind

    def to_table(self) -> list[dict]:
        """The interval as a one-row table, in metafor's column names."""
        return [{
            "name": self.label,
            "estimate": self.estimate,
            "ci.lb": self.ci_lb,
            "ci.ub": self.ci_ub,
        }]

    def __str__(self) -> str:
        width = max(len(self.label), 5)
        head = f"{'':>{width}}{'estimate':>10}{'ci.lb':>10}{'ci.ub':>10}"
        row = (f"{self.label:>{width}}{self.estimate:>10.4f}"
               f"{self.ci_lb:>10.4f}{self.ci_ub:>10.4f}")
        note = ""
        if self.at_bound_lb or self.at_bound_ub:
            note = "\n(an endpoint is a bound; the profile never left the interval)"
        return (f"{head}\n{row}\n\n"
                f"Level {level_to_percent(self.level):g}%, profile likelihood, "
                f"{self.method}{note}")


def _parameter_slots(result):
    """Flatten ``[[name, (lo, hi)], ...]`` over the model's parameter vector."""
    slots = []
    for comp in result.components:
        for name, bound in zip(comp.vc_names, comp.bounds(), strict=True):
            slots.append((name, bound))
    return slots


def _profile_starts(result, free, bounds, r_compat):
    """Starting points for the refit at a pinned parameter.

    The fitted values alone are not enough.  A between-study variance that the
    rest of the model does not need can collapse to its zero boundary, and from
    the interior optimum the optimiser never goes there -- metafor's does not
    either.  That matters because such a point can have a *higher* likelihood
    than the interior one at the same correlation, which makes the profile
    discontinuous-looking and any interval built from the interior branch
    internally inconsistent (a point with a higher profile likelihood must be
    inside the interval).

    So: start from the fit, from the fit with each variance component pushed to
    its lower bound, and from a scaled version of the fit.  ``r_compat`` uses
    the fitted values only, which is what reproduces metafor.
    """
    base = np.asarray(result.params, dtype=float)
    starts = [base[free].copy()]
    if r_compat:
        return starts
    for pos, slot in enumerate(free):
        lo, _hi = bounds[slot]
        if lo is not None and lo == 0.0:
            pushed = base[free].copy()
            pushed[pos] = 0.0
            starts.append(pushed)
    for factor in (0.1, 10.0):
        starts.append(base[free] * factor)
    return starts


def _profile_loglik(result, index, value, bounds, cache, r_compat=False):
    """Profile log-likelihood: parameter ``index`` pinned at ``value``.

    Everything else is re-estimated, which is what makes this a profile and not
    a slice.  Results are cached because the endpoint search evaluates the same
    point more than once.
    """
    key = float(value)
    if key in cache:
        return cache[key]
    base = np.asarray(result.params, dtype=float).copy()
    free = [j for j in range(base.size) if j != index]
    lo, hi = bounds[index]
    if lo is not None and value < lo - 1e-12:
        raise ValueError(f"the profile was asked for {value} but the "
                         f"parameter cannot go below {lo}")
    if hi is not None and value > hi + 1e-12:
        raise ValueError(f"the profile was asked for {value} but the "
                         f"parameter cannot go above {hi}")

    def objective(theta):
        full = base.copy()
        full[free] = theta
        full[index] = value
        D_list = []
        offset = 0
        for comp in result.components:
            D_list.append(comp.make_D(full[offset:offset + comp.n_params]))
            offset += comp.n_params
        return _mv_loglik(D_list, result.yi, result.V, result.X, result.method)

    free_bounds = [bounds[j] for j in free]
    best_val = np.inf
    for start in _profile_starts(result, free, bounds, r_compat):
        candidate = _mv_optimise(objective, start, free_bounds,
                                 result.components, minimize, minimize_scalar)
        best_val = min(best_val, float(objective(candidate)))
    ll = -best_val
    cache[key] = ll
    return ll


def _endpoint(profile, start, target, limit, direction, tol):
    """Walk outward from ``start`` until the profile crosses ``target``.

    Returns ``(value, at_bound)``.  The step doubles, so a wide interval costs a
    handful of fits rather than a grid, but it is never allowed to jump the
    last half of the distance to the limit: the profile can dip below the
    threshold and come back up (a variance component collapsing to zero does
    exactly that), and walking straight to the bound would report "the interval
    reaches the bound" when it does not.  The crossing is then refined with
    Brent's method, as R's ``uniroot`` does it.
    """
    if abs(limit - start) == 0.0:
        return limit, True
    step = max(abs(limit - start) * 0.05, 1e-3)
    inner = x = start
    for _ in range(200):
        remaining = abs(limit - x)
        step = min(step, 0.5 * remaining) if remaining > 0.0 else 0.0
        x = x + direction * step
        if direction * (x - limit) > 0.0:
            x = limit
        if profile(x) <= target:
            a, b = (inner, x) if direction > 0 else (x, inner)
            if abs(b - a) <= tol:
                return x, x == limit
            return zeroin(lambda t: profile(t) - target, a, b, tol=tol), False
        inner = x
        if x == limit:
            return limit, True
        step *= 2.0
    return limit, True


def confint_rma_mv(result, *, tau2=None, rho=None, level=None,
                   tol=UNIROOT_TOL, r_compat=False):
    """Profile-likelihood interval for one ``tau2`` or ``rho`` of an ``rma_mv``.

    Args:
        result: a :class:`~pimeta.RmaMvResult` from :func:`pimeta.rma_mv`.
        tau2: 1-based index of the between-study variance component to profile,
            counting only the ``tau2`` components.
        rho: 1-based index of the correlation component to profile.
        level: confidence level, accepted as ``95``, ``0.95`` or ``0.05``;
            defaults to the level the model was fitted at.
        tol: convergence tolerance handed to the root finder.  The default is
            R's ``uniroot`` default, so the endpoint matches metafor's.
        r_compat: profile from the fitted values only, as metafor does.  The
            default re-starts the refit from the variance boundaries as well,
            because a component that collapses to zero can leave the
            likelihood higher than the interior branch and metafor's interval
            would then exclude a point with a larger profile likelihood.
            ``True`` reproduces metafor; ``False`` is the supremum over the
            parameter space.  See ``docs/reference/differences.md`` §4.

    Returns:
        :class:`ConfintMvResult`, whose ``curve`` holds every point the search
        evaluated.

    Raises:
        ValueError: when neither or both of ``tau2`` and ``rho`` are given, when
            the model has no such component, or when the index is out of range.

    Example:
        >>> fit = ma.rma_mv(yi, V=V, mods=X, random="~ outcome | study",
        ...                 struct="UN")
        >>> ma.confint_rma_mv(fit, rho=1)
    """
    if not isinstance(result, RmaMvResult):
        raise TypeError("'result' must be an RmaMvResult from rma_mv()")
    if (tau2 is None) == (rho is None):
        raise ValueError("give exactly one of 'tau2' and 'rho', as metafor's "
                         "confint() does: it profiles one component at a time")
    kind, index = ("tau2", tau2) if tau2 is not None else ("rho", rho)
    index = int(index)
    if index < 1:
        raise ValueError(f"'{kind}' is 1-based")

    slots = _parameter_slots(result)
    bounds = [b for _, b in slots]
    positions = [i for i, (name, _) in enumerate(slots) if name == kind]
    if not positions:
        raise ValueError(
            f"the model has no '{kind}' component to profile; it has "
            f"{sorted({name for name, _ in slots})}")
    if index > len(positions):
        raise ValueError(
            f"'{kind}' has {len(positions)} component(s), not {index}")
    slot = positions[index - 1]

    alpha = level_to_alpha(result.level if level is None else level)
    est = float(np.asarray(result.params, dtype=float)[slot])
    ll_max = float(result.ll) if result.ll is not None else None
    if ll_max is None:                                  # pragma: no cover
        raise ValueError("the fit carries no log-likelihood to profile")
    crit = float(0.5 * stats.chi2.ppf(1.0 - alpha, 1))
    target = ll_max - crit

    cache = {est: ll_max}
    profile = lambda v: _profile_loglik(result, slot, v, bounds, cache,  # noqa: E731
                                        r_compat=r_compat)

    lo, hi = bounds[slot]
    search_lo = lo if lo is not None else est - max(10.0, 10.0 * abs(est))
    if hi is not None:
        search_hi = hi
    else:
        # metafor's rule for an unbounded variance component
        search_hi = est + max(10.0, 10.0 * max(abs(est), 1e-8))

    ci_lb, at_lb = _endpoint(profile, est, target, search_lo, -1.0, tol)
    ci_ub, at_ub = _endpoint(profile, est, target, search_hi, 1.0, tol)

    curve = sorted(cache.items())
    return ConfintMvResult(
        kind=kind,
        index=index,
        estimate=est,
        ci_lb=float(ci_lb),
        ci_ub=float(ci_ub),
        level=level_to_percent(result.level if level is None else level),
        ll_max=ll_max,
        crit=crit,
        method=result.method,
        at_bound_lb=bool(at_lb),
        at_bound_ub=bool(at_ub),
        curve=curve,
    )
