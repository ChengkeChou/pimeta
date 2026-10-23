"""Non-linear meta-regression: spline and fractional-polynomial bases.

The estimator in :mod:`pimeta.metareg` is linear *in the parameters*, not in
the moderator. Give it a column for ``(x - 5)**3`` and it fits it. A non-linear
dose-response or age effect is therefore a **basis expansion** of the moderator
plus one extra test, which is what this module provides:

``rcs``
    Restricted cubic spline (Harrell). ``K`` knots give ``K - 1`` columns: the
    linear term plus ``K - 2`` non-linear terms. The fit is forced to be linear
    beyond the outer knots, so the curve cannot turn up at the edge of the data
    the way an ordinary polynomial does. This is the usual choice for
    dose-response and for "is the effect monotone?" questions.

``fp``
    Fractional polynomials (Royston & Sauerbrei). A small set of powers —
    possibly negative, possibly fractional, with ``log(x)`` standing in for the
    power 0 — instead of a polynomial's 1, 2, 3. Two terms usually beat a
    cubic.

Both return a :class:`Basis`: an array-like object that remembers how it was
built, so it can be re-evaluated at new values. That is what
:func:`predict_curve` needs, and it is why a basis can be handed to
``predict`` without recomputing knots by hand.

    >>> import pimeta as ma
    >>> fit = ma.rma(yi, vi, mods="~ rcs(dose, 4)", data=df)
    >>> ma.nonlinear_test(fit, "rcs(dose,4)")
    >>> ma.predict_curve(fit, ma.rcs(df["dose"], nk=4, name="rcs(dose,4)"), grid)

The columns are named ``"<label>[1]"``, ``"<label>[2]"``, … so a term can be
found again from the fitted coefficient names. ``nonlinear_test`` and
``predict_curve`` both use that convention.

Nothing here changes the estimator: the basis is ordinary moderator data, the
fit is the same weighted least squares with a random effect, and the result is
an ordinary :class:`~pimeta.rma_uni.RmaResult`. That also means every model
built this way is recordable and replayable like any other.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from scipy import stats

__all__ = [
    "Basis",
    "CurveResult",
    "NonlinearTestResult",
    "WaldResult",
    "default_knots",
    "fp",
    "nonlinear_test",
    "predict_curve",
    "rcs",
    "wald_test",
]


# --------------------------------------------------------------------------- #
# the basis object
# --------------------------------------------------------------------------- #
@dataclass
class Basis:
    """An expanded moderator: the columns, their names, and how to rebuild them.

    Attributes:
        values: ``(n, m)`` matrix of the expanded moderator.
        names: one name per column, ``"<label>[j]"``.
        label: what the term is called in the model, e.g. ``"rcs(dose,4)"``.
        kind: ``"rcs"`` or ``"fp"``.
        builder: maps a 1-D vector of new values to a new ``(g, m)`` matrix.
        detail: the construction parameters (knots, powers, scale), for display
            and for the record.
    """

    values: np.ndarray
    names: list[str]
    label: str
    kind: str
    x: np.ndarray = field(repr=False)
    builder: Callable[[np.ndarray], np.ndarray] = field(repr=False)
    detail: dict[str, Any] = field(default_factory=dict)

    # -- array protocol, so a Basis can be passed straight to mods= ---------- #
    def __array__(self, dtype=None, copy=None) -> np.ndarray:
        return np.array(self.values, dtype=dtype, copy=copy)

    @property
    def shape(self) -> tuple[int, int]:
        return self.values.shape

    @property
    def ndim(self) -> int:
        return 2

    def __len__(self) -> int:
        return int(self.values.shape[0])

    def __getitem__(self, item):
        return self.values[item]

    # -- the reason this is a class and not a bare array -------------------- #
    def evaluate(self, x) -> Basis:
        """Rebuild the same basis at new values of the moderator."""
        new = np.asarray(x, dtype=float).ravel()
        return Basis(
            values=self.builder(new),
            names=list(self.names),
            label=self.label,
            kind=self.kind,
            x=new,
            builder=self.builder,
            detail=dict(self.detail),
        )

    def to_dict(self) -> dict:
        return {
            "label": self.label,
            "kind": self.kind,
            "names": list(self.names),
            **self.detail,
        }


def _column_names(label: str, m: int) -> list[str]:
    return [f"{label}[{j + 1}]" for j in range(m)]


def _squash(text: str) -> str:
    """Whitespace-insensitive form, so ``rcs(x, 4)`` matches ``rcs(x,4)``."""
    return "".join(str(text).split())


# --------------------------------------------------------------------------- #
# restricted cubic splines
# --------------------------------------------------------------------------- #
#: Harrell's recommended percentile placement, from ``rms::rcspline.eval``.
_HARRELL = {
    3: (10.0, 50.0, 90.0),
    4: (5.0, 35.0, 65.0, 95.0),
    5: (5.0, 27.5, 50.0, 72.5, 95.0),
    6: (5.0, 23.0, 41.0, 59.0, 77.0, 95.0),
    7: (2.5, 18.33, 34.17, 50.0, 65.83, 81.67, 97.5),
}


def default_knots(x, nk: int = 4) -> np.ndarray:
    """Harrell's default knot positions for ``nk`` knots.

    Uses the published percentile table for 3–7 knots and an evenly spaced
    grid of percentiles from 5% to 95% beyond that. Quantiles are the
    ``type = 7`` (linear interpolation) ones that R uses by default.

    Args:
        x: the moderator.
        nk: number of knots.

    Returns:
        Sorted knot vector of length ``nk``.

    Raises:
        ValueError: for fewer than 3 knots, or more knots than distinct values.
    """
    x = np.asarray(x, dtype=float).ravel()
    x = x[np.isfinite(x)]
    if nk < 3:
        raise ValueError(
            f"a restricted cubic spline needs at least 3 knots, not {nk}")
    percentiles = _HARRELL.get(nk)
    if percentiles is None:
        percentiles = tuple(np.linspace(5.0, 95.0, nk))
    knots = np.quantile(x, np.asarray(percentiles) / 100.0)
    n_distinct = int(np.unique(knots).size)
    if n_distinct < nk:
        raise ValueError(
            f"cannot place {nk} distinct knots: the moderator has only "
            f"{int(np.unique(x).size)} distinct value(s). Use fewer knots, or "
            "pass explicit 'knots='."
        )
    return knots


def _rcs_matrix(x: np.ndarray, knots: np.ndarray) -> np.ndarray:
    """Harrell's restricted cubic spline basis, normalised by the knot range^2.

    Column 1 is ``x`` itself; column ``j + 1`` is

    .. math::

        \\frac{(x-t_j)_+^3 - w_j (x-t_{K-1})_+^3
               + w'_j (x-t_K)_+^3}{(t_K - t_1)^2}

    with :math:`w_j = (t_K - t_j)/(t_K - t_{K-1})` and
    :math:`w'_j = (t_{K-1} - t_j)/(t_K - t_{K-1})`. Beyond the outer knots the
    quadratic and cubic terms cancel by construction, which is what makes the
    fit linear there.
    """
    t = np.asarray(knots, dtype=float)
    x = np.asarray(x, dtype=float)
    k = t.size
    denom = (t[-1] - t[0]) ** 2
    ratio_hi = (t[-1] - t[:-2]) / (t[-1] - t[-2])
    ratio_lo = (t[-2] - t[:-2]) / (t[-1] - t[-2])
    cols = [x]
    for j in range(k - 2):
        cube = (np.clip(x - t[j], 0.0, None) ** 3
                - ratio_hi[j] * np.clip(x - t[-2], 0.0, None) ** 3
                + ratio_lo[j] * np.clip(x - t[-1], 0.0, None) ** 3) / denom
        cols.append(cube)
    return np.column_stack(cols)


def rcs(x, nk: int | None = 4, df: int | None = None, knots=None,
        name: str | None = None) -> Basis:
    """Restricted cubic spline basis (Harrell) for a moderator.

    Args:
        x: the moderator, 1-D.
        nk: number of knots; 4 by default, which gives 3 columns.
        df: degrees of freedom — the number of *columns* — as an alternative to
            ``nk``. ``df=3`` is the same as ``nk=4``.
        knots: explicit knot positions. When given, ``nk`` and ``df`` are
            ignored.
        name: label for the term; defaults to ``"rcs"``. ``design_matrix``
            passes the call itself, so a fitted coefficient is named
            ``rcs(dose,4)[2]``.

    Returns:
        :class:`Basis` with ``nk - 1`` columns: ``x`` followed by the
        ``nk - 2`` non-linear terms.

    Raises:
        ValueError: for fewer than 3 knots, fewer than 4 distinct values, or
            knots that are not strictly increasing.

    Example:
        >>> b = rcs([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], nk=4)   # doctest: +SKIP
        >>> b.shape                                          # doctest: +SKIP
        (10, 3)
    """
    xv = np.asarray(x, dtype=float).ravel()
    if not np.all(np.isfinite(xv)):
        raise ValueError("'x' contains missing or infinite values")
    if df is not None:
        nk = int(df) + 1
    if knots is None:
        nk = 4 if nk is None else int(nk)
        kn = default_knots(xv, nk)
    else:
        kn = np.sort(np.asarray(knots, dtype=float).ravel())
        if kn.size < 3:
            raise ValueError(
                f"a restricted cubic spline needs at least 3 knots, "
                f"not {kn.size}")
        if np.any(np.diff(kn) <= 0):
            raise ValueError("'knots' must be strictly increasing")
    label = "rcs" if name is None else str(name)
    values = _rcs_matrix(xv, kn)
    return Basis(
        values=values,
        names=_column_names(label, values.shape[1]),
        label=label,
        kind="rcs",
        x=xv,
        builder=lambda new: _rcs_matrix(np.asarray(new, dtype=float).ravel(), kn),
        detail={"knots": [float(v) for v in kn], "nk": int(kn.size)},
    )


# --------------------------------------------------------------------------- #
# fractional polynomials
# --------------------------------------------------------------------------- #
def _fp_matrix(x: np.ndarray, powers: Sequence[float], scale: float,
               center: bool) -> np.ndarray:
    """Royston-Sauerbrei fractional-polynomial terms.

    The transform of power ``p`` is ``z**p`` for ``p != 0`` and ``log(z)`` for
    ``p == 0``, where ``z = x / scale``. A power repeated for the second time
    is multiplied by ``log(z)``, which is the limit of
    ``(z**(p + d) - z**p)/d`` as ``d -> 0`` and hence the natural second term
    of a two-term fractional polynomial with equal powers.
    """
    z = np.asarray(x, dtype=float) / float(scale)
    needs_positive = any(p <= 0.0 for p in powers)
    if needs_positive and np.any(z <= 0.0):
        raise ValueError(
            "'fp' needs strictly positive values when a power is zero or "
            "negative: log(x) and x**p must exist")
    if not needs_positive and np.any(z < 0.0):
        raise ValueError("'fp' needs non-negative values")
    # log(0) is only reached when a log term exists, which 'needs_positive'
    # excludes; guarding the zeros keeps numpy quiet and gives the correct
    # limit (0) for a repeated positive power at x = 0.
    logz = np.log(np.where(z > 0.0, z, 1.0))
    seen: dict[float, int] = {}
    cols = []
    for p in powers:
        base = logz if p == 0.0 else z ** float(p)
        repeat = seen.get(float(p), 0)
        cols.append(base * (logz ** repeat))
        seen[float(p)] = repeat + 1
    out = np.column_stack(cols)
    if center:
        # Centring changes the intercept only; it keeps the transformed
        # columns on a comparable scale. metafor users usually centre.
        out = out - out.mean(axis=0)
    return out


def fp(x, powers, scale: float | None = None, center: bool = False,
       name: str | None = None) -> Basis:
    """Fractional-polynomial basis (Royston & Sauerbrei) for a moderator.

    Args:
        x: the moderator, strictly positive, 1-D.
        powers: the powers, e.g. ``[1, 2]``, ``[-1, 0]``, ``[0.5, 0.5]``. They
            are sorted (the literature always lists them ascending); a repeated
            power contributes a second term multiplied by ``log(x)``.
        scale: divides ``x`` before the transform. ``None`` (default) means 1,
            i.e. no rescaling. Scaling does not change the fit, only the
            coefficients.
        center: subtract each column's mean. Also fit-invariant; it only makes
            the intercept interpretable at the mean of the transformed values.
        name: label for the term; defaults to ``"fp"``.

    Returns:
        :class:`Basis` with one column per power.

    Raises:
        ValueError: for an empty power list, non-positive ``x``, or a missing
            value.

    Example:
        >>> b = fp([1, 2, 3, 4], powers=[0, 1])      # log(x) and x
        >>> [round(v, 4) for v in b.values[:, 0]]
        [0.0, 0.6931, 1.0986, 1.3863]
    """
    xv = np.asarray(x, dtype=float).ravel()
    if not np.all(np.isfinite(xv)):
        raise ValueError("'x' contains missing or infinite values")
    pw = [float(p) for p in np.sort(np.asarray(list(powers), dtype=float).ravel())]
    if not pw:
        raise ValueError("'powers' must contain at least one power")
    sc = 1.0 if scale is None else float(scale)
    label = "fp" if name is None else str(name)
    values = _fp_matrix(xv, pw, sc, center)
    return Basis(
        values=values,
        names=_column_names(label, values.shape[1]),
        label=label,
        kind="fp",
        x=xv,
        builder=lambda new: _fp_matrix(
            np.asarray(new, dtype=float).ravel(), pw, sc, center),
        detail={
            "powers": [float(p) for p in pw],
            "scale": sc,
            "center": bool(center),
        },
    )


# --------------------------------------------------------------------------- #
# joint Wald tests
# --------------------------------------------------------------------------- #
@dataclass
class WaldResult:
    """Joint test that a set of coefficients is zero.

    Mirrors ``metafor::anova.rma``: ``QM`` is the quadratic form
    ``b' V^-1 b`` over the selected coefficients, referred to a chi-squared
    distribution (``test="z"``) or an F distribution with ``df`` and ``ddf``
    degrees of freedom (``test="knha"``/``"t"``/``"adhoc"``).

    Attributes:
        QM: the test statistic.
        df: number of coefficients tested (the numerator df).
        pval: p-value.
        names: the coefficients that were tested.
        test: ``"chi2"`` or ``"F"``.
        ddf: denominator df, or ``None`` for the chi-squared test.
        k: number of studies in the model.
    """

    QM: float
    df: int
    pval: float
    names: list[str]
    test: str
    ddf: float | None
    k: int

    def to_dict(self) -> dict:
        return {
            "QM": self.QM,
            "df": self.df,
            "pval": self.pval,
            "names": list(self.names),
            "test": self.test,
            "ddf": self.ddf,
            "k": self.k,
        }


def _coef_vector(result) -> np.ndarray:
    """Coefficients, under either naming: ``rma`` uses ``b``, ``rma_mv`` ``beta``."""
    for name in ("b", "beta"):
        value = getattr(result, name, None)
        if value is not None:
            return np.asarray(value, dtype=float).ravel()
    return np.empty(0)


def _coef_names(result) -> list[str]:
    names = getattr(result, "coefnames", None)
    b = _coef_vector(result)
    if names is None or len(names) != b.size:
        names = [f"b{i + 1}" for i in range(b.size)]
    return list(names)


def wald_test(result, btt: Sequence[int] | None = None,
              names: Sequence[str] | None = None) -> WaldResult:
    """Joint Wald test for a subset of coefficients (``metafor::anova``).

    Args:
        result: a fitted :class:`~pimeta.rma_uni.RmaResult` (or anything with
            ``b``/``beta``, ``vb``, ``test``, ``s2w`` and ``ddf``).
        btt: **1-based** coefficient positions, exactly as in
            ``metafor::anova(fit, btt=)``. Note the difference from Python
            indexing: ``btt=[2, 3]`` tests the second and third coefficients.
        names: coefficient names instead of positions. Matching is exact, so
            pass the names ``result.coefnames`` reports.

    Returns:
        :class:`WaldResult`.

    Raises:
        ValueError: when nothing is selected, or a position is out of range.

    Example:
        >>> fit = ma.rma(yi, vi, mods="~ x", data=df)    # doctest: +SKIP
        >>> ma.wald_test(fit, btt=[2])                    # doctest: +SKIP
    """
    b = _coef_vector(result)
    vb = np.asarray(getattr(result, "vb", np.empty((0, 0))), dtype=float)
    if b.size == 0 or vb.shape != (b.size, b.size):
        raise TypeError(
            "wald_test needs a fitted model with 'b' and 'vb' "
            f"(got {type(result).__name__})")
    all_names = _coef_names(result)

    if btt is not None and names is not None:
        raise ValueError("give either 'btt' or 'names', not both")
    if btt is not None:
        idx = [int(i) - 1 for i in btt]
        out_of_range = [i + 1 for i in idx if i < 0 or i >= b.size]
        if out_of_range:
            raise ValueError(
                f"'btt' positions {out_of_range} are outside 1..{b.size} "
                "(btt is 1-based, as in metafor)")
    elif names is not None:
        wanted = [str(n) for n in names]
        missing = [n for n in wanted if n not in all_names]
        if missing:
            raise ValueError(
                f"no coefficient named {missing}. Available: {all_names}")
        idx = [all_names.index(n) for n in wanted]
    else:
        raise ValueError("give 'btt' or 'names' to say which coefficients to test")
    if not idx:
        raise ValueError("no coefficients selected")

    bq = b[idx]
    vq = vb[np.ix_(idx, idx)]
    qm = float(bq @ np.linalg.solve(vq, bq))
    df = len(idx)

    test_name = str(getattr(result, "test", "z")).lower()
    if test_name in ("t", "knha", "adhoc"):
        # metafor's anova divides the quadratic form by the Knapp-Hartung
        # scaling factor and refers it to an F distribution.
        s2w = float(getattr(result, "s2w", 1.0) or 1.0)
        ddf = getattr(result, "ddf", None)
        if ddf is None:
            k = int(getattr(result, "k", 0))
            p = int(getattr(result, "p", b.size))
            ddf = float(k - p)
        stat = qm / s2w
        return WaldResult(
            QM=qm, df=df, pval=float(stats.f.sf(stat / df, df, ddf)),
            names=[all_names[i] for i in idx], test="F", ddf=float(ddf),
            k=int(getattr(result, "k", 0)),
        )
    return WaldResult(
        QM=qm, df=df, pval=float(stats.chi2.sf(qm, df)),
        names=[all_names[i] for i in idx], test="chi2", ddf=None,
        k=int(getattr(result, "k", 0)),
    )


@dataclass
class NonlinearTestResult:
    """Overall and non-linearity tests for one basis term.

    Attributes:
        term: the term label, e.g. ``"rcs(dose,4)"``.
        overall: joint test over every column of the term — "is there any
            association?".
        nonlinear: joint test over the non-linear columns only — "is the
            association non-linear?". For a spline this is all columns after
            the first, because the first column is ``x`` itself.
        k: number of studies.
    """

    term: str
    overall: WaldResult
    nonlinear: WaldResult
    k: int

    def to_dict(self) -> dict:
        return {
            "term": self.term,
            "overall": self.overall.to_dict(),
            "nonlinear": self.nonlinear.to_dict(),
            "k": self.k,
        }

    def __str__(self) -> str:
        return (
            f"{self.term}: overall chi2({self.overall.df}) = "
            f"{self.overall.QM:.4f}, p = {self.overall.pval:.4g}; "
            f"non-linearity chi2({self.nonlinear.df}) = "
            f"{self.nonlinear.QM:.4f}, p = {self.nonlinear.pval:.4g}"
        )


def _term_positions(result, term: str) -> list[int]:
    """1-based positions of the coefficients belonging to ``term``."""
    wanted = _squash(term)
    all_names = _coef_names(result)
    pos = [i + 1 for i, n in enumerate(all_names)
           if _squash(n) == wanted or _squash(n).startswith(wanted + "[")]
    if not pos:
        raise ValueError(
            f"no coefficient belongs to term {term!r}. Available: {all_names}")
    return pos


def nonlinear_test(result, term: str) -> NonlinearTestResult:
    """Test whether the effect of a basis term is non-linear.

    Runs two joint Wald tests over the columns that belong to ``term``: one
    over all of them (``overall``) and one over all but the first
    (``nonlinear``). The convention that the first column is the linear term
    holds for both ``rcs`` and ``fp``, where the first power is the linear one
    whenever the model has it.

    Args:
        result: a fitted model that used a basis as moderator.
        term: the basis label, e.g. ``"rcs(dose,4)"`` or ``"fp(dose,[1,2])"``.

    Returns:
        :class:`NonlinearTestResult`.

    Raises:
        ValueError: when no coefficient belongs to ``term``.

    Example:
        >>> fit = ma.rma(yi, vi, mods="~ rcs(dose, 4)", data=df)   # doctest: +SKIP
        >>> print(ma.nonlinear_test(fit, "rcs(dose,4)"))           # doctest: +SKIP
    """
    pos = _term_positions(result, term)
    overall = wald_test(result, btt=pos)
    if len(pos) < 2:
        raise ValueError(
            f"term {term!r} has a single column, so it cannot be non-linear")
    nonlinear = wald_test(result, btt=pos[1:])
    return NonlinearTestResult(
        term=term, overall=overall, nonlinear=nonlinear,
        k=int(getattr(result, "k", 0)),
    )


# --------------------------------------------------------------------------- #
# predicted curves
# --------------------------------------------------------------------------- #
@dataclass
class CurveResult:
    """A fitted curve with its confidence and prediction bands.

    Attributes:
        x: the grid the curve was evaluated on.
        fit: predicted effect at each grid point (analysis scale).
        se: standard error of the prediction.
        ci_lb, ci_ub: confidence band.
        pi_lb, pi_ub: prediction band (``None`` entries when ``tau2 == 0``,
            because then there is no between-study distribution to predict).
        level: confidence level, e.g. 0.95.
        term: the basis label.
        at: the values the other moderators were held at.
    """

    x: np.ndarray
    fit: np.ndarray
    se: np.ndarray
    ci_lb: np.ndarray
    ci_ub: np.ndarray
    pi_lb: np.ndarray | None
    pi_ub: np.ndarray | None
    level: float
    term: str
    at: dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "x": np.asarray(self.x, dtype=float).tolist(),
            "fit": np.asarray(self.fit, dtype=float).tolist(),
            "se": np.asarray(self.se, dtype=float).tolist(),
            "ci_lb": np.asarray(self.ci_lb, dtype=float).tolist(),
            "ci_ub": np.asarray(self.ci_ub, dtype=float).tolist(),
            "pi_lb": None if self.pi_lb is None else np.asarray(self.pi_lb).tolist(),
            "pi_ub": None if self.pi_ub is None else np.asarray(self.pi_ub).tolist(),
            "level": self.level,
            "term": self.term,
            "at": dict(self.at),
        }


def predict_curve(result, basis: Basis, grid=None, at: dict | None = None,
                  level: float | None = None) -> CurveResult:
    """Evaluate a fitted basis term over a grid, with bands.

    This is a thin wrapper around ``result.predict(newmods=...)``: the only
    thing it adds is rebuilding the design matrix from a :class:`Basis`, and
    holding the *other* moderators at fixed values.

    Args:
        result: a fitted :class:`~pimeta.rma_uni.RmaResult` whose moderators
            include ``basis``.
        basis: the basis used in the fit (or one rebuilt with the same knots).
        grid: values of the moderator to evaluate at. Defaults to 100 points
            spanning the observed range.
        at: values for the other moderators, by coefficient name. Anything not
            given is held at 0, which is the model's reference point.
        level: confidence level; defaults to the level of the fit.

    Returns:
        :class:`CurveResult`.

    Raises:
        ValueError: when no coefficient belongs to the basis, when ``at`` names
            an unknown moderator, or when a moderator is left unset.

    Example:
        >>> b = ma.rcs(df["dose"], nk=4, name="rcs(dose,4)")     # doctest: +SKIP
        >>> fit = ma.rma(df["yi"], df["vi"], mods=b)             # doctest: +SKIP
        >>> curve = ma.predict_curve(fit, b)                     # doctest: +SKIP
    """
    xv = np.asarray(basis.x if grid is None else grid, dtype=float).ravel()
    if grid is None:
        lo, hi = float(np.min(xv)), float(np.max(xv))
        xv = np.linspace(lo, hi, 100)
    evaluated = basis.evaluate(xv)

    all_names = _coef_names(result)
    mods_names = [n for n in all_names if n != "intrcpt"]
    if not mods_names:
        raise ValueError("the model has no moderators")
    prefix = f"{_squash(basis.label)}["
    term_pos = [i for i, n in enumerate(mods_names)
                if _squash(n).startswith(prefix)]
    if not term_pos:
        raise ValueError(
            f"no moderator belongs to basis {basis.label!r}. Available: "
            f"{mods_names}. Rebuild the basis with the same 'name=' you used "
            "in the formula.")
    if len(term_pos) != evaluated.values.shape[1]:
        raise ValueError(
            f"the basis has {evaluated.values.shape[1]} column(s) but the model "
            f"gives term {basis.label!r} {len(term_pos)}; the knots or powers "
            "differ from the fitted ones")

    held = dict(at or {})
    unknown = [n for n in held if n not in mods_names]
    if unknown:
        raise ValueError(
            f"'at' names moderators the model does not have: {unknown}. "
            f"Available: {mods_names}")

    newmods = np.zeros((xv.size, len(mods_names)))
    for j in term_pos:
        newmods[:, j] = evaluated.values[:, term_pos.index(j)]
    for j, name in enumerate(mods_names):
        if name in held:
            newmods[:, j] = float(held[name])

    preds = result.predict(newmods=newmods, level=level)
    if not isinstance(preds, list):
        preds = [preds]

    pi_lb = [p.pi_lb for p in preds]
    pi_ub = [p.pi_ub for p in preds]
    has_pi = all(v is not None for v in pi_lb)
    return CurveResult(
        x=xv,
        fit=np.array([p.pred for p in preds], dtype=float),
        se=np.array([p.se for p in preds], dtype=float),
        ci_lb=np.array([p.ci_lb for p in preds], dtype=float),
        ci_ub=np.array([p.ci_ub for p in preds], dtype=float),
        pi_lb=np.array(pi_lb, dtype=float) if has_pi else None,
        pi_ub=np.array(pi_ub, dtype=float) if has_pi else None,
        level=1.0 - float(preds[0].level),
        term=basis.label,
        at={k: float(v) for k, v in held.items()},
    )
