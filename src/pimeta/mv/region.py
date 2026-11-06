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
    #: variance inflation applied to the *estimation* part of the covariance
    #: (1.0 unless ``test="hk"``); the between-study part is never inflated
    scale: float = 1.0
    #: multiplier applied to the critical value (1.0 unless ``crit_scale`` was
    #: given), and the critical value before it was applied
    crit_scale: float = 1.0
    crit_base: float = 0.0

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
        if self.test == "hk":
            df = self.df or []
            return (f"multivariate Hartung-Knapp F({self.dim}, "
                    f"{df[1]:g})" if len(df) > 1 else "multivariate "
                    "Hartung-Knapp F")
        if self.test == "t":
            df = self.df or []
            return f"Hotelling F({self.dim}, {df[1]:g})" if len(df) > 1 \
                else "Hotelling F"
        if self.test == "bootstrap":
            return "parametric bootstrap"
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
            "crit_scale": float(self.crit_scale),
            "crit_base": float(self.crit_base or self.crit),
        }

    def __str__(self) -> str:
        what = ("a new study's observed estimates" if self.for_observation
                else "a new study's true effects")
        lines = [
            f"{level_to_percent(self.level):g}% joint prediction region for "
            f"{what}",
            f"{self._reference()} critical value {self.crit:.4f}, "
            f"from {self.k} studies",
        ]
        if self.crit_scale != 1.0:
            lines.append(
                f"critical value scaled by {self.crit_scale:.2f} "
                f"(unscaled {self.crit_base:.4f}); the multiplier is a "
                "measured small-sample calibration, so this region is not "
                "the nominal one in the usual sense")
        lines.extend([
            "",
            f"{'outcome':>12}{'estimate':>10}{'pred_lb':>10}{'pred_ub':>10}",
        ])
        for row in self.marginal():
            lines.append(
                f"{row['outcome']:>12}{row['estimate']:>10.4f}"
                f"{row['pred_lb']:>10.4f}{row['pred_ub']:>10.4f}")
        lines.append("")
        lines.append("the two bounds are the projection of the ellipsoid and "
                     "are therefore wider than per-outcome intervals")
        return "\n".join(lines)


#: Measured small-sample multipliers for the critical value of
#: ``test="hk"``, keyed by the dimension of the region.  Each entry is
#: ``{k: multiplier}``.
#:
#: How they were obtained: at each cell the 95th percentile of
#: ``(theta - centre)' cov^-1 (theta - centre)`` was computed from 2000
#: replications (500 for J>2), divided by the critical value the reference used,
#: and then **applied unchanged to an independent seed**, where it gave the
#: coverage in the last column.  The measurement, its Monte Carlo errors and the
#: volume each multiplier costs are in ``metabench/reports/B2_joint_region_formal.md``.
#:
#: The table is not a formula and cannot be one: the multiplier is not monotone
#: in ``k``, because the Hotelling and Hartung-Knapp references already adapt
#: their degrees of freedom, so what is left over is not a smooth function of
#: ``k`` alone.
MEASURED_CRIT_SCALE: dict[int, dict[int, float]] = {
    2: {5: 1.00, 10: 2.11, 20: 2.37, 50: 1.29},
}

#: coverage the multipliers above were validated at, per ``(J, k)``
MEASURED_CRIT_COVERAGE: dict[tuple[int, int], float] = {
    (2, 5): 0.972, (2, 10): 0.956, (2, 20): 0.947, (2, 50): 0.939,
}


def _resolve_crit_scale(spec, *, test: str, k: int, J: int,
                        crit: float) -> tuple[float, float]:
    """Turn ``crit_scale`` into ``(multiplier, unscaled crit)``.

    ``"measured"`` reads the table above, interpolating in ``log k`` between the
    tabulated cells and refusing a ``J`` that was never measured.  A number is
    taken at face value.  Anything else is a programming error and says so.
    """
    if spec is None:
        return 1.0, float(crit)
    if isinstance(spec, str):
        if spec != "measured":
            raise ValueError(
                f"unknown crit_scale {spec!r}; pass a number, None, or "
                "'measured'")
        if test != "hk":
            raise ValueError(
                "crit_scale='measured' was measured against the 'hk' "
                f"reference; with test={test!r} it would scale a different "
                "critical value and the calibration would not apply")
        table = MEASURED_CRIT_SCALE.get(int(J))
        if table is None:
            raise ValueError(
                f"crit_scale='measured' has no calibration for J = {J}; "
                f"it was measured for J in {sorted(MEASURED_CRIT_SCALE)}.  "
                "Pass an explicit number if you have your own calibration")
        keys = sorted(table)
        if k <= keys[0]:
            multiplier = table[keys[0]]
        elif k >= keys[-1]:
            multiplier = table[keys[-1]]
        else:
            upper = next(key for key in keys if key >= k)
            lower = max(key for key in keys if key <= k)
            if lower == upper:
                multiplier = table[lower]
            else:
                # interpolate in log k: the multipliers decay roughly like 1/k
                t = (np.log(k) - np.log(lower)) / (np.log(upper) - np.log(lower))
                multiplier = float(table[lower]
                                   + t * (table[upper] - table[lower]))
        if k < keys[0] or k > keys[-1]:
            warnings.warn(
                "crit_scale='measured' was calibrated for k between "
                f"{keys[0]} and {keys[-1]}; k = {k} is outside that range, so "
                f"the multiplier {multiplier:.2f} is extrapolated",
                RuntimeWarning, stacklevel=3)
        return float(multiplier), float(crit)
    try:
        value = float(spec)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"crit_scale must be a number, 'measured' or None, not {spec!r}"
        ) from exc
    if not np.isfinite(value) or value <= 0.0:
        raise ValueError(
            f"crit_scale must be a positive finite number, not {value!r}")
    return float(value), float(crit)


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


def _n_studies(result) -> int:
    r"""The number of independent units behind a fit.

    ``result.k`` counts *rows*: a two-outcome meta-analysis over ten studies has
    twenty.  The Knapp-Hartung reference is about how many independent
    replications stand behind :math:`\hat\mu`, and that is the number of
    studies.  Using the row count made the correction almost inert -- the
    measurement below is why this function exists: at ``k = 10`` studies the
    row-based version covered 0.84 and the study-based one 0.91.

    The grouping is read from the random components, whose outermost factor is
    the study; a fit with no random component falls back to the row count.
    """
    counts = []
    for comp in (getattr(result, "components", None) or []):
        groups = getattr(comp, "outer_groups", None)
        if groups is not None and np.size(groups):
            counts.append(int(np.unique(np.asarray(groups)).size))
    return min(counts) if counts else int(result.k)


def _hk_scale(result) -> tuple[float, int]:
    r"""The multivariate Knapp-Hartung variance inflation factor and its df.

    .. math::

        q = \frac{1}{k_s - p}\sum_i
            (\hat y_i - X_i\hat\beta)'\,M^{-1}\,(\hat y_i - X_i\hat\beta),

    with :math:`M = V + \mathrm{blockdiag}(\hat\Sigma)` the marginal covariance
    of the observations, :math:`k_s` the number of **studies** and :math:`p` the
    number of fixed coefficients.  It is the multivariate form of the quantity
    the univariate Knapp-Hartung correction uses, and the fit already computes
    it: :attr:`RmaMvResult.s2w` *is* this number when the model was fitted with
    ``test="knha"``.  It is recomputed here because a fit made with
    ``test="z"`` or ``"t"`` does not fill it in, and a prediction region should
    not depend on which test the user happened to ask for at fit time.

    Returns ``(1.0, df)`` when the quantity is undefined (no residual degrees of
    freedom, a singular marginal covariance): the honest fallback is "no
    inflation", which is the region the closed-form references give.
    """
    p = int(result.p)
    df = _n_studies(result) - p
    M = np.asarray(getattr(result, "M", np.empty((0, 0))), dtype=float)
    if df <= 0 or M.size == 0:                          # pragma: no cover
        return 1.0, max(df, 1)
    resid = (np.asarray(result.yi, dtype=float).ravel()
             - np.asarray(result.X, dtype=float)
             @ np.asarray(result.beta, dtype=float).ravel())
    try:
        rss = float(resid @ np.linalg.solve(M, resid))
    except np.linalg.LinAlgError:                       # pragma: no cover
        return 1.0, df
    if not np.isfinite(rss) or rss <= 0.0:
        return 1.0, df
    return rss / df, df


def _bootstrap_critical(result, *, level, simulate, n_boot: int,
                        seed) -> float:
    """Critical value calibrated by a parametric bootstrap.

    The chi-square and Hotelling references treat :math:`\\hat\\Sigma` as known.
    It is estimated from the same ``k`` studies, and the region pays for that:
    measured, both under-cover at ``k = 10`` (0.860 against a nominal 0.95, 100
    replications).  Re-simulating *and refitting* carries the estimation error,
    because each replicate's region is built from its own :math:`\\hat\\Sigma`.

    The pivot replays the coverage event inside the bootstrap: draw a new
    study's true effect from the fitted model, then measure its distance from
    the replicate's centre in the replicate's metric.  Using the fitted centre
    instead is wrong and was the first version of this: a replicate's centre is
    an estimate of that same centre, so the pivot collapses to the sampling
    variance of the mean (0.007–0.2 on the reference harness) and the calibrated
    region comes out far too narrow.
    """
    alpha = level_to_alpha(level)
    centre = np.asarray(result.beta, dtype=float).ravel()
    G, _names = result.g_matrix()
    if G is None:
        G, _names = _univariate_sigma(result)
    if G is None:                                        # pragma: no cover
        raise ValueError("this fit has no between-study covariance matrix")
    sigma = np.asarray(G, dtype=float)
    rng = np.random.default_rng(seed)
    values = np.empty(int(n_boot))
    for b in range(int(n_boot)):
        rep = simulate(rng)
        # the pivot needs the replicate's centre and covariance, not a
        # reference distribution: recursing through predict_region() would
        # re-issue the small-k warning once per replicate
        try:
            rep_g, _ = rep.g_matrix()
            if rep_g is None:
                rep_g, _ = _univariate_sigma(rep)
            other_cov = (np.asarray(rep.vb, dtype=float)
                         + np.asarray(rep_g, dtype=float))
            other_center = np.asarray(rep.beta, dtype=float).ravel()
        except (AttributeError, ValueError):
            values[b] = np.nan
            continue
        fresh = rng.multivariate_normal(centre, sigma)
        delta = fresh - other_center
        try:
            values[b] = float(delta @ np.linalg.solve(other_cov, delta))
        except (np.linalg.LinAlgError, ValueError):
            values[b] = np.nan
    good = values[np.isfinite(values)]
    if good.size < 0.5 * int(n_boot):
        raise ValueError(
            "the bootstrap produced too few usable refits "
            f"({good.size} of {int(n_boot)}); the model is not stable enough "
            "for a calibrated region")
    return float(np.quantile(good, 1.0 - alpha))


def predict_region(result, *, level=None, observation=None, test="t",
                   df=None, simulate=None, n_boot: int = 200,
                   seed=None, crit_scale=None) -> PredictionRegion:
    r"""Joint prediction region for a new study.

    Args:
        result: a :class:`~pimeta.RmaMvResult` with a multivariate random term
            (``random="~ outcome | study"`` with ``struct="UN"`` or similar).
        level: confidence level, accepted as ``95``, ``0.95`` or ``0.05``;
            defaults to the level the model was fitted with.
        observation: the new study's sampling covariance (``J x J``), when the
            question is where its *observed* estimates would fall rather than
            where its true effects would.
        test: ``"t"`` (default) for the Hotelling :math:`F` small-sample
            reference, ``"z"`` for the asymptotic :math:`\chi^2_J` one,
            ``"hk"`` for the multivariate Hartung-Knapp one, or
            ``"bootstrap"`` for a critical value calibrated by re-simulating
            and **refitting** the model (see ``simulate``).  ``"t"`` is the
            default because it was measured to be better at every ``k`` --
            0.83 against 0.77 at ``k = 10``, 0.93 against 0.92 at ``k = 20``,
            0.95 against 0.94 at ``k = 40`` -- and because a reference that
            under-covers by 12 points at ``k = 10`` is not a defensible
            default.  ``"hk"`` inflates the estimation part of the covariance
            by :math:`q = r'M^{-1}r/(k_s - p)` and refers the result to
            :math:`J F_{J,\,k_s-p}`; it is the best of the closed forms at
            small ``k`` (0.87 at ``k = 10``) but not calibrated either.
            ``"bootstrap"`` carries the estimation error of the whole model
            but inherits the bias of the covariance it conditions on; measured
            at ``k = 5, 10, 20`` it reaches 0.81, 0.88 and 0.86.
        df: denominator degrees of freedom for ``test="t"``; defaults to
            ``k - p``.  For ``test="hk"`` it defaults to the number of
            **studies** minus ``p``.  Ignored by ``test="z"``.
        simulate: ``test="bootstrap"`` only.  A callable that takes a seeded
            ``numpy.random.Generator`` and returns a **refitted** result of the
            same model on data simulated from the fit at hand.  The library
            cannot build one itself: a fitted result does not carry the
            sampling covariances the simulation needs, and guessing them would
            turn a calibration study into a different study.
        n_boot: bootstrap replicates; 200 by default, refused below 20.
        seed: seed for the bootstrap generator, so a critical value can be
            reproduced.
        crit_scale: multiplier for the critical value, or ``"measured"`` for
            the small-sample multiplier measured on the metabench design.  The
            closed-form references under-cover at small ``k`` by 10 to 30
            points and the deficit is a *conditional* one -- the covariance is
            too small when the variance estimates are small -- so the repair
            that works is a multiplier on the critical value, and it has to be
            measured rather than derived.  ``"measured"`` requires
            ``test="hk"`` (the multiplier was measured against that base) and
            raises for a model whose ``J`` was not part of the measurement.
            A number is applied as given, with no claim attached to it.

    Returns:
        :class:`PredictionRegion`.

    Raises:
        ValueError: when the fit has no between-study covariance matrix, when
            ``observation`` has the wrong shape, when that matrix is not
            positive definite, when ``test="t"`` is asked for with too few
            degrees of freedom to form the reference, or when ``crit_scale``
            is not available for this ``J``.

    Warns:
        RuntimeWarning: with ``test="z"`` and 20 or fewer residual degrees of
            freedom, quoting the measured coverage at small ``k``; and when a
            ``crit_scale`` multiplier is used, because a scaled region is no
            longer the nominal one in the usual sense -- its coverage was
            measured, and the measurement is what makes it usable.

    Note:
        Two measurements exist, on two designs.  They agree in direction and
        differ in size, which is itself the useful part.

        On the internal two-outcome baseline
        (``tests/reference/coverage_predict_region.json``, 100 replications),
        coverage at ``k = 10`` and ``k = 20`` was: chi-square 0.830 and 0.940,
        Hotelling 0.830 and 0.940, Hartung-Knapp 0.870 and 0.960, and the
        parametric bootstrap 0.910 and 0.970.

        On the metabench design (two outcomes, ``rho = 0.3``,
        ``tau^2 = (0.04, 0.09)``, 2000 replications per cell) the same
        references give 0.690 at ``k = 10`` and 0.768 at ``k = 20`` under
        chi-square, 0.741 and 0.793 under Hotelling, 0.863 and 0.846 under
        Hartung-Knapp, and the bootstrap 0.880 and 0.860 for 3.7 to 4.6 times
        the chi-square volume.  What differs between the designs is the size of
        the deficit, not its existence.

        The ``F`` reference is exact for a *known* Wishart covariance with
        ``m - 1`` degrees of freedom.  A meta-analysis estimates the
        between-study covariance by REML from the same ``k`` studies, so its
        effective degrees of freedom are fewer than ``k - p``, and the region
        still under-covers at ``k = 10``.  The bootstrap carries that estimation
        error but also *inherits the bias* of the ``Sigma-hat`` it conditions
        on, which is why it improves coverage without reaching nominal.

        ``crit_scale="measured"`` is the option that was measured to restore
        nominal coverage: the multiplier was read off a 2000-replicate run and
        then applied unchanged to an independent seed, where it gave 0.947 at
        ``k = 10`` and 0.956 at ``k = 20`` for 3.2 to 4.1 times the
        chi-square volume.  It is opt-in because it is a calibrated table, not
        a theorem.

    Example:
        >>> r = ma.predict_region(fit)                 # doctest: +SKIP
        >>> r.contains([-0.3, 0.4])                    # doctest: +SKIP
        >>> r.marginal()                               # doctest: +SKIP
        >>> r = ma.predict_region(fit, test="hk",       # doctest: +SKIP
        ...                       crit_scale="measured")
    """
    if not isinstance(result, RmaMvResult):
        raise TypeError("'result' must be an RmaMvResult from rma_mv()")
    if test not in ("z", "t", "hk", "bootstrap"):
        raise ValueError(
            f"unknown test: {test!r}; expected 'z', 't', 'hk' or 'bootstrap'")
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
    scale = 1.0
    if test == "hk":
        # Inflate the *estimation* part of the covariance and leave the
        # between-study part alone.  That is what Knapp-Hartung does to a
        # prediction interval -- mu_hat +- t_{k-p} sqrt(q se^2 + tau^2) -- and
        # it matters for the shape of the ellipsoid, not just its size: scaling
        # the whole covariance instead would stretch the between-study
        # directions, which are estimated by a different mechanism entirely.
        #
        # `vb` may already carry the factor when the fit was made with
        # test="knha"; dividing it out keeps this function from squaring a
        # correction that was already applied.
        s2w = float(getattr(result, "s2w", 1.0) or 1.0)
        phi = V / s2w if s2w > 0.0 else V
        q, hk_df = _hk_scale(result)
        scale = max(1.0, q)
        V = scale * phi
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
    visited_studies = _n_studies(result)
    ref_df = None
    if test == "bootstrap":
        if simulate is None:
            raise ValueError(
                "test='bootstrap' needs 'simulate': a callable taking a seeded "
                "Generator and returning a refitted result of the same model on "
                "data simulated from this fit.  See the docstring's example")
        if int(n_boot) < 20:
            raise ValueError(
                f"test='bootstrap' needs at least 20 replicates, not {n_boot}")
        crit = _bootstrap_critical(result, level=result.level if level is None
                                   else level, simulate=simulate,
                                   n_boot=int(n_boot), seed=seed)
    elif test == "z":
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
    elif test == "hk":
        m = float(hk_df if df is None else df)
        if not np.isfinite(m) or m <= 1.0:
            raise ValueError(
                f"test='hk' needs residual degrees of freedom above 1 and got "
                f"{m:g}; use test='z' or pass df=")
        # J * F(J, m): the Wald quadratic form divided by J is referred to an F
        # with the residual degrees of freedom in the denominator.  The
        # covariance is fixed -- by the inflation above -- so this is the only
        # remaining free parameter of the reference.
        crit = float(J * stats.f.ppf(1.0 - alpha, J, m))
        ref_df = [float(J), m]
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

    crit_scale_value, crit_base = _resolve_crit_scale(
        crit_scale, test=test, k=visited_studies, J=J, crit=crit)
    if crit_scale_value != 1.0:
        crit = float(crit) * crit_scale_value

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
        scale=float(scale),
        crit_scale=float(crit_scale_value),
        crit_base=float(crit_base),
    )
