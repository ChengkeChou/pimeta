"""Multi-level / multivariate random-effects models, replicating
``metafor::rma.mv`` and ``metafor::robust``.

Scope
-----
The full ``rma.mv`` in R is ~2000 lines and supports arbitrary variance
structures (CS/AR/UN/...), random slopes, custom weight matrices ``W`` and
sparse matrices. The medically relevant core is the **random-intercepts
multilevel model**, i.e. ::

    rma.mv(yi, vi, random = ~ 1 | study)          # single random intercept
    rma.mv(yi, vi, random = ~ 1 | district/school) # nested random intercepts

which is what this module reproduces exactly (REML/ML estimation of the
variance components via the same marginal-likelihood objective metafor
minimises, plus the cluster-robust ``robust()`` sandwich estimator).

``~ 1 | a/b`` expands to ``~ 1 | a + 1 | a:b`` (``/`` = nesting), so the
second random effect is the interaction ``a:b``, exactly as in R.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from scipy import stats
from scipy.optimize import minimize, minimize_scalar

from .missing import complete_cases

Z_975 = stats.norm.ppf(0.975)


@dataclass
class RmaMvResult:
    """Multi-level model fit, mirroring the fields of ``rma.mv`` objects."""

    @property
    def diagnostics(self):
        """The structured diagnostics for this fit.

        A property rather than a stored field, so that a check can be added or
        removed without changing what a fit computes.
        """
        from .checks import diagnose

        return diagnose(self)

    def to_dict(self) -> dict:
        """Flat export, mirroring :meth:`RmaResult.to_dict`.

        The field set is the dataclass's own, not the ``s3`` tidy/glance shape:
        the point of this method is to round-trip a fit exactly, and the tidy
        frames are a reporting view that drops the internals.
        """
        from .checks import SCHEMA_VERSION, software_versions

        return {
            "schema_version": SCHEMA_VERSION,
            "software": software_versions(),
            "diagnostics": self.diagnostics.to_dict(),
            "method": self.method,
            "k": self.k,
            "p": self.p,
            "sigma2": list(self.sigma2),
            "beta": list(self.beta),
            "se": list(self.se),
            "zval": list(self.zval),
            "pval": list(self.pval),
            "ci_lb": list(self.ci_lb),
            "ci_ub": list(self.ci_ub),
            "qe": self.qe,
            "qe_df": self.qe_df,
            "qe_p": self.qe_p,
            "tau2": None if self.tau2 is None else list(self.tau2),
            "rho": None if self.rho is None else list(self.rho),
            "ll": self.ll,
            "random": self.random,
            "level": self.level,
        }

    method: str                     # "REML" | "ML"
    k: int
    p: int
    sigma2: list[float]             # variance components (one per random term)
    beta: list[float]               # fixed-effect coefficients
    se: list[float]
    zval: list[float]
    pval: list[float]
    ci_lb: list[float]
    ci_ub: list[float]
    qe: float                       # residual heterogeneity Q
    qe_df: int
    qe_p: float
    random: str = ""
    int_only: bool = True
    # internals consumed by robust() (not serialised). These are always supplied
    # by rma_mv(), so an empty-array factory keeps them non-Optional.
    M: np.ndarray = field(                          # marginal var-cov matrix
        repr=False, default_factory=lambda: np.empty((0, 0))
    )
    vb: np.ndarray = field(                         # (X'W X)^-1 model vcov
        repr=False, default_factory=lambda: np.empty((0, 0))
    )
    X: np.ndarray = field(repr=False, default_factory=lambda: np.empty((0, 0)))
    yi: np.ndarray = field(repr=False, default_factory=lambda: np.empty(0))
    level: float = 95.0
    # `~ inner | outer` variance/correlation components (mirrors rma.mv's
    # `$tau2` / `$rho`); None for models with only `~ 1 | factor` terms.
    tau2: list[float] | None = None
    rho: list[float] | None = None
    # fit statistics mirroring rma.mv's `$fit.stats` (both ML and REML columns)
    ll: float | None = None         # logLik under the fitted method
    parms: int | None = None        # p + number of random-effect parameters
    fit_stats: dict | None = None   # {"ML": {...}, "REML": {...}}
    ddf: list[float] | None = None  # denominator df (test="t"/"knha"/"adhoc")
    s2w: float = 1.0                # Knapp-Hartung variance scaling factor
    # internals consumed by ranef() (not serialised)
    components: list | None = field(repr=False, default=None)
    params: np.ndarray = field(repr=False, default_factory=lambda: np.empty(0))
    # inputs kept so that metafor's rma.mv methods (hatvalues, weights,
    # rstandard, cooks.distance, dfbetas, rstudent, qqnorm) can recompute on
    # subsets of the data.
    V: np.ndarray = field(repr=False, default_factory=lambda: np.empty((0, 0)))
    mods: np.ndarray | None = field(repr=False, default=None)
    struct: str = "CS"
    dfs: str | float | list | None = None
    intercept_incl: bool = True
    slab: list[str] | None = None
    cluster: list | None = None
    test: str = "z"
    # grouping variables used to fit the model, needed by the refit-based
    # diagnostics (stored under a private name so repr stays short).
    _data: dict | None = field(repr=False, default=None)

    def g_matrix(self):
        """Between-study covariance of the first multivariate random term.

        Returns:
            ``(G, names)`` — the ``q x q`` covariance matrix of the random
            coefficients and their names, or ``(None, [])`` when the model has
            no random term with more than one coefficient.

        Both random-slope structures are covered: ``struct="UN"`` over an inner
        factor (``J`` levels) and ``struct="GEN"`` over the design columns
        (``q`` coefficients).  ``GEN`` was silently skipped until a
        differential test against ``metafor`` reported a between-study
        covariance of zero for every random-slope fit.
        """
        if not self.components:
            return None, []
        offset = 0
        for comp in self.components:
            n = comp.n_params
            block = self.params[offset:offset + n]
            offset += n
            J = int(getattr(comp, "J", getattr(comp, "q", 1)))
            if J < 2:
                continue
            labels = list(getattr(comp, "inner_labels", None)
                          or getattr(comp, "var_names", None) or [])
            if all(labels):
                names = [str(x) for x in labels]
            else:
                names = [f"y{i + 1}" for i in range(J)]
            if hasattr(comp, "_G"):
                return np.asarray(comp._G(block), dtype=float), names
            # compound symmetry: tau2 on the diagonal, rho off it
            _s2, tau2, rho = comp.report(block)
            tau2_val = float(tau2[0]) if tau2 else 0.0
            rho_val = float(rho[0]) if rho else 0.0
            G = np.full((J, J), rho_val * tau2_val)
            np.fill_diagonal(G, tau2_val)
            return G, names
        return None, []


def _encode_levels(values):
    """Map ``values`` to 0-based level indices (sorted order, R ``factor``)."""
    levels = sorted(set(values))
    index = {v: i for i, v in enumerate(levels)}
    idx = np.array([index[v] for v in values], dtype=int)
    return idx, levels


def _one_hot(values):
    """Encode ``values`` as a 0/1 indicator matrix (levels in sorted order)."""
    idx, levels = _encode_levels(values)
    z = np.zeros((len(values), len(levels)), dtype=float)
    z[np.arange(len(values)), idx] = 1.0
    return z


def _get_col(data, name):
    """Pull a column from a dict-like or pandas DataFrame ``data``."""
    if data is None:
        raise ValueError(
            f"the random-effects formula refers to the grouping variable "
            f"{name!r}, but no 'data' was supplied; pass a data frame (or a "
            f"dict) holding that column")
    if hasattr(data, "iloc"):  # pandas DataFrame
        col = data[name]
        return [v for v in col]
    try:
        return [v for v in data[name]]
    except KeyError:
        raise ValueError(
            f"grouping variable {name!r} is not a column of 'data' "
            f"(available: {sorted(data) if hasattr(data, '__iter__') else data})"
        ) from None


def _parse_random_factors(random: str) -> list[str]:
    """Expand ``~ 1 | a/b/c`` into ``["a", "a:b", "a:b:c"]``."""
    rhs = random.split("|")[-1].strip()
    parts = [p.strip() for p in rhs.split("/") if p.strip()]
    factors: list[str] = []
    acc: list[str] = []
    for p in parts:
        acc.append(p)
        factors.append(":".join(acc))
    return factors


def _split_top_plus(rhs: str) -> list[str]:
    """Split a formula RHS on top-level ``+`` (outside any parentheses)."""
    parts: list[str] = []
    depth = 0
    cur: list[str] = []
    for ch in rhs:
        if ch == "(":
            depth += 1
        elif ch == ")":
            depth -= 1
        if ch == "+" and depth == 0:
            parts.append("".join(cur).strip())
            cur = []
        else:
            cur.append(ch)
    if cur:
        parts.append("".join(cur).strip())
    return [p for p in parts if p]


def _component_from_inner_outer(inner: str, outer: str, struct: str) -> list[dict]:
    """Turn one ``inner | outer`` fragment into component descriptors.

    ``inner == "1"`` yields one random-intercept component per nesting level
    of ``outer`` (``a/b`` -> ``a``, ``a:b``).  Otherwise the fragment is an
    ``~ inner | outer`` term whose variance structure is governed by
    ``struct``: ``"CS"`` (default) treats ``inner`` as a factor with a
    compound-symmetric variance matrix, ``"GEN"`` treats the ``inner``
    variables as predictors producing random slopes with an unstructured
    variance-covariance matrix.
    """
    outer_parts = [p.strip() for p in outer.split("/") if p.strip()]

    if inner.strip() == "1":
        comps: list[dict] = []
        acc: list[str] = []
        for p in outer_parts:
            acc.append(p)
            comps.append({"kind": "intercept", "factor": ":".join(acc),
                          "name": "/".join(acc)})
        return comps

    s = str(struct).upper()
    name = f"~ {inner.strip()} | {outer_parts[-1]}"

    if s == "GEN":
        vars_ = [v.strip() for v in _split_top_plus(inner) if v.strip()]
        return [{"kind": "gen", "vars": vars_, "outer": outer_parts[-1],
                 "name": name}]

    if s == "UN":
        # unstructured G over the *inner factor* levels (unlike GEN, which
        # treats inner as random-slope predictors)
        return [{"kind": "un", "inner": inner.strip(), "outer": outer_parts[-1],
                 "name": name}]

    return [{"kind": "struct", "inner": inner.strip(), "outer": outer_parts[-1],
             "struct": s, "name": name}]


def _parse_random_components(random, struct="CS") -> list[dict]:
    """Parse ``random`` into a list of component descriptors.

    Handles ``~ 1 | a/b`` (nested intercepts), ``~ 1 | a + 1 | b`` (multiple
    intercepts) and ``~ inner | outer`` (structured random effects / random
    slopes).
    """
    _known = ("CS", "HCS", "UN", "ID", "DIAG", "GEN", "AR", "HAR", "CAR",
              "SPEXP", "SPGAU", "SPLONG", "EXP", "GAU")
    if struct is not None and str(struct).upper() not in _known:
        raise ValueError(
            f"unknown 'struct' {struct!r}; supported: {', '.join(_known)}")
    struct = str(struct).upper()
    if isinstance(random, (list, tuple)):
        structs = struct if isinstance(struct, (list, tuple)) else [struct] * len(random)
        comps: list[dict] = []
        for r, s in zip(random, structs):
            comps.extend(_parse_random_components(r, s))
        return comps

    rhs = str(random).strip()
    if rhs.startswith("~"):
        rhs = rhs[1:].strip()

    if rhs.count("|") == 1:
        inner, outer = rhs.split("|", 1)
        return _component_from_inner_outer(inner.strip(), outer.strip(), struct)

    # multiple ``... | ...`` terms joined by ``+``
    comps = []
    for term in _split_top_plus(rhs):
        if "|" not in term:
            raise ValueError(
                f"random-effects formula {random!r} has no grouping factor; "
                f"use the form '~ 1 | study' (or '~ inner | outer')")
        inner, outer = term.split("|", 1)
        comps.extend(_component_from_inner_outer(inner.strip(), outer.strip(), struct))
    return comps


def _factor_values(data, name: str):
    """Resolve a (possibly interacting) factor into its level values."""
    cols = name.split(":")
    if len(cols) == 1:
        return _get_col(data, cols[0])
    columns = [_get_col(data, c) for c in cols]
    return list(zip(*columns))


class _InterceptComp:
    """``~ 1 | factor`` — single variance component scaled identity."""

    kind = "intercept"

    def __init__(self, Z: np.ndarray, labels: list[str] | None = None,
                 name: str | None = None):
        self.Z = Z
        self.labels = labels if labels is not None else [str(i) for i in range(Z.shape[1])]
        self.name = name if name is not None else "intercept"
        self._ZZ = Z @ Z.T

    def contain_groups(self) -> list[np.ndarray]:
        """Group label vectors used for ``dfs="contain"``."""
        return [np.argmax(self.Z, axis=1)]

    def blup(self, p: np.ndarray, W: np.ndarray, r: np.ndarray,
             X: np.ndarray, XWX_inv: np.ndarray):
        """BLUP of the random intercepts: ``b = tau2 Z' W r``."""
        tau2 = float(p[0])
        Z = self.Z
        b = tau2 * (Z.T @ (W @ r)).ravel()
        ZWZ = Z.T @ W @ Z
        ZWX = Z.T @ W @ X
        var = tau2 - tau2 ** 2 * np.diag(ZWZ) + tau2 ** 2 * np.diag(ZWX @ XWX_inv @ ZWX.T)
        se = np.sqrt(np.maximum(var, 0.0))
        return self.labels, b, se

    @property
    def n_params(self) -> int:
        return 1

    @property
    def vc_names(self) -> list[str]:
        """What each entry of ``p`` is, in order (metafor's parameter names)."""
        return ["tau2"]

    @property
    def n_vc(self) -> int:
        return 1

    def make_D(self, p: np.ndarray) -> np.ndarray:
        return float(p[0]) * self._ZZ

    def init(self, tau2_val: float) -> np.ndarray:
        return np.array([tau2_val])

    def bounds(self) -> list:
        return [(0.0, None)]

    def report(self, p: np.ndarray):
        s2 = float(p[0])
        if s2 <= np.finfo(np.float64).eps * 10:
            s2 = 0.0
        return [s2], [], []


class _CSComp:
    """``~ inner | outer`` with compound symmetry (``tau2`` + ``rho``)."""

    kind = "cs"

    def __init__(self, inner_idx: np.ndarray, outer_groups: np.ndarray, J: int,
                 inner_labels: list[str] | None = None,
                 outer_labels: list[str] | None = None,
                 name: str | None = None):
        self.inner_idx = inner_idx
        self.outer_groups = outer_groups
        self.J = J
        self.inner_labels = inner_labels if inner_labels is not None else [str(i) for i in range(J)]
        n_outer = int(outer_groups.max()) + 1
        self.outer_labels = outer_labels if outer_labels is not None else [str(i) for i in range(n_outer)]
        self.name = name if name is not None else "cs"
        # boolean "same outer group" indicator (k x k)
        self._same = (outer_groups[:, None] == outer_groups[None, :]).astype(float)

    def contain_groups(self) -> list[np.ndarray]:
        return [self.inner_idx, self.outer_groups]

    def blup(self, p: np.ndarray, W: np.ndarray, r: np.ndarray,
             X: np.ndarray, XWX_inv: np.ndarray):
        """BLUP of ``~ inner | outer`` under compound symmetry.

        Returns the per-observation (``k``-length) predictions and SEs with
        ``"inner | outer"`` labels; ``ranef()`` deduplicates the unique
        ``inner x outer`` combinations (metafor's ``duplicated(r.names)``).
        """
        tau2, rho = p
        J = self.J
        k = self.inner_idx.shape[0]
        G = tau2 * ((1.0 - rho) * np.eye(J) + rho * np.ones((J, J)))
        Z1 = np.eye(J)[self.inner_idx]                       # k x J
        N = int(self.outer_groups.max()) + 1
        Z2 = np.eye(N)[self.outer_groups]                    # k x N
        Gfull = (Z1 @ G @ Z1.T) * (Z2 @ Z2.T)                # k x k
        GW = Gfull @ W
        pred = (GW @ r).ravel()                              # k
        I = np.eye(k)
        H = X @ XWX_inv @ X.T @ W
        vpred = Gfull - GW @ (I - H) @ Gfull
        se = np.sqrt(np.maximum(np.diag(vpred), 0.0))
        labels = [f"{self.inner_labels[self.inner_idx[i]]} | {self.outer_labels[self.outer_groups[i]]}"
                  for i in range(k)]
        return labels, pred, se

    @property
    def n_params(self) -> int:
        return 2

    @property
    def vc_names(self) -> list[str]:
        return ["tau2", "rho"]

    @property
    def n_vc(self) -> int:
        return 1

    def make_D(self, p: np.ndarray) -> np.ndarray:
        tau2, rho = p
        G = tau2 * ((1.0 - rho) * np.eye(self.J) + rho * np.ones((self.J, self.J)))
        Gi = G[np.ix_(self.inner_idx, self.inner_idx)]
        return Gi * self._same

    def init(self, tau2_val: float) -> np.ndarray:
        return np.array([tau2_val, 0.0])

    def bounds(self) -> list:
        # rho in (-1, 1); a value of exactly -1 makes G singular so keep a
        # small margin while still allowing near-degenerate fits.
        return [(0.0, None), (-1.0 + 1e-9, 1.0 - 1e-9)]

    def report(self, p: np.ndarray):
        tau2, rho = p
        return [0.0], [float(tau2)], [float(rho)]


class _StructComp:
    """``~ inner | outer`` under one of metafor's variance structures.

    Covers ``"CS"``, ``"ID"``, ``"DIAG"``, ``"HCS"``, ``"AR"``, ``"HAR"``
    and ``"CAR"`` (``"CS"`` is also routed here; the standalone
    :class:`_CSComp` is kept for the ``struct="CS"`` default path).  The
    variance matrix is ``D = (Z1 G Z1') * (Z2 Z2')`` where ``G`` is the
    ``J x J`` inner covariance matrix parameterised per structure exactly as
    metafor's ``.process.G.afterrmna`` does it.

    The free parameter vector ``p`` holds only the *free* parameters:
    ``n_free_tau2`` variance components followed by ``rho`` (if the structure
    has one).  Inner levels whose factor has ``k == 1`` (only one distinct
    outer level) have their ``tau2`` fixed to 0 (metafor), so they are
    excluded from ``p`` and re-inserted as 0 in ``_full_tau2``.
    """

    kind = "struct"

    def __init__(self, inner_idx: np.ndarray, outer_groups: np.ndarray, J: int,
                 struct: str,
                 inner_values: np.ndarray | None = None,
                 inner_labels: list[str] | None = None,
                 outer_labels: list[str] | None = None,
                 name: str | None = None,
                 fixed_tau2: np.ndarray | None = None,
                 fixed_rho: float | None = None):
        self.inner_idx = inner_idx
        self.outer_groups = outer_groups
        self.J = J
        self.struct = str(struct).upper()
        # metafor treats a user-supplied 'rho' as fixed, not as a start value.
        self.fixed_rho = fixed_rho
        self.inner_labels = inner_labels if inner_labels is not None else [str(i) for i in range(J)]
        n_outer = int(outer_groups.max()) + 1
        self.outer_labels = outer_labels if outer_labels is not None else [str(i) for i in range(n_outer)]
        self.name = name if name is not None else "struct"
        self._same = (outer_groups[:, None] == outer_groups[None, :]).astype(float)
        if inner_values is not None:
            # CAR ``g.values``: sorted unique (rounded) inner level values.
            self.inner_values = np.asarray(inner_values, dtype=float).reshape(-1)
        else:
            self.inner_values = np.arange(J, dtype=float)
        if fixed_tau2 is not None:
            self._fixed_tau2 = np.asarray(fixed_tau2, dtype=bool).reshape(-1)
        else:
            self._fixed_tau2 = np.zeros(J, dtype=bool)
        self._free_tau2 = ~self._fixed_tau2
        self._n_free_tau2 = int(self._free_tau2.sum())

    # -- parameterisation -------------------------------------------------

    @property
    def n_tau2(self) -> int:
        if self.struct in ("DIAG", "HCS", "HAR"):
            return self.J
        return 1

    @property
    def has_rho(self) -> bool:
        if self.fixed_rho is not None:
            return False
        return self.struct in ("CS", "HCS", "AR", "HAR", "CAR")

    def _rho(self, p: np.ndarray) -> float:
        """Correlation used by ``_G``: fixed when supplied, else estimated."""
        if self.fixed_rho is not None:
            return float(self.fixed_rho)
        return float(p[-1])

    @property
    def n_params(self) -> int:
        if self.n_tau2 == self.J:
            n = self._n_free_tau2
        else:
            n = 1
        return n + (1 if self.has_rho else 0)

    @property
    def vc_names(self) -> list[str]:
        n_tau2 = self.n_params - (1 if self.has_rho else 0)
        return ["tau2"] * n_tau2 + (["rho"] if self.has_rho else [])

    @property
    def n_vc(self) -> int:
        return self.n_tau2

    def _full_tau2(self, p: np.ndarray) -> np.ndarray:
        """Full length-``J`` tau2 vector with fixed (k=1) entries set to 0."""
        tau2 = np.zeros(self.J)
        tau2[self._free_tau2] = p[: self._n_free_tau2]
        return tau2

    def _toeplitz(self, rho: float) -> np.ndarray:
        d = np.abs(np.subtract.outer(np.arange(self.J), np.arange(self.J)))
        return rho ** d

    def _G(self, p: np.ndarray) -> np.ndarray:
        J = self.J
        s = self.struct
        if s == "CS":
            tau2, rho = p[0], self._rho(p)
            G = np.full((J, J), rho * tau2)
            np.fill_diagonal(G, tau2)
            return G
        if s == "ID":
            return float(p[0]) * np.eye(J)
        if s == "DIAG":
            return np.diag(self._full_tau2(p))
        if s == "HCS":
            tau2 = self._full_tau2(p)
            rho = self._rho(p)
            sq = np.sqrt(tau2)
            G = rho * np.outer(sq, sq)
            np.fill_diagonal(G, tau2)
            return G
        if s == "AR":
            tau2, rho = p[0], self._rho(p)
            return tau2 * self._toeplitz(rho)
        if s == "HAR":
            tau2 = self._full_tau2(p)
            rho = self._rho(p)
            sq = np.sqrt(tau2)
            G = np.outer(sq, sq) * self._toeplitz(rho)
            np.fill_diagonal(G, tau2)
            return G
        if s == "CAR":
            tau2, rho = p[0], self._rho(p)
            x = self.inner_values
            cor = rho ** np.abs(x[:, None] - x[None, :])
            return tau2 * cor
        raise ValueError(f"unsupported struct {s!r}")

    # -- component protocol ----------------------------------------------

    def contain_groups(self) -> list[np.ndarray]:
        return [self.inner_idx, self.outer_groups]

    def blup(self, p: np.ndarray, W: np.ndarray, r: np.ndarray,
             X: np.ndarray, XWX_inv: np.ndarray):
        G = self._G(p)
        J = self.J
        k = self.inner_idx.shape[0]
        Z1 = np.eye(J)[self.inner_idx]
        N = int(self.outer_groups.max()) + 1
        Z2 = np.eye(N)[self.outer_groups]
        Gfull = (Z1 @ G @ Z1.T) * (Z2 @ Z2.T)
        GW = Gfull @ W
        pred = (GW @ r).ravel()
        I = np.eye(k)
        H = X @ XWX_inv @ X.T @ W
        vpred = Gfull - GW @ (I - H) @ Gfull
        se = np.sqrt(np.maximum(np.diag(vpred), 0.0))
        labels = [f"{self.inner_labels[self.inner_idx[i]]} | {self.outer_labels[self.outer_groups[i]]}"
                  for i in range(k)]
        return labels, pred, se

    def make_D(self, p: np.ndarray) -> np.ndarray:
        G = self._G(p)
        Gi = G[np.ix_(self.inner_idx, self.inner_idx)]
        return Gi * self._same

    def init(self, tau2_val: float) -> np.ndarray:
        if self.n_tau2 == self.J:
            n_tau = self._n_free_tau2
        else:
            n_tau = 1
        out = np.full(n_tau, float(tau2_val))
        if self.has_rho:
            out = np.append(out, 0.5)
        return out

    def bounds(self) -> list:
        if self.n_tau2 == self.J:
            n_tau = self._n_free_tau2
        else:
            n_tau = 1
        b: list[tuple[float, float | None]] = [(0.0, None)] * n_tau
        if self.has_rho:
            if self.struct == "CAR":
                b.append((1e-9, 1.0 - 1e-9))
            else:
                b.append((-1.0 + 1e-9, 1.0 - 1e-9))
        return b

    def report(self, p: np.ndarray):
        if self.n_tau2 == self.J:
            tau2 = [float(x) for x in self._full_tau2(p)]
        else:
            tau2 = [float(p[0])]
        if self.has_rho:
            rho = [float(p[-1])]
        elif self.fixed_rho is not None:
            rho = [float(self.fixed_rho)]
        else:
            rho = [0.0]  # ID / DIAG: rho fixed at 0 (metafor)
        return [0.0], tau2, rho


class _GENComp:
    """``~ vars | outer`` — random slopes, unstructured G.

    Parameterised exactly as metafor does for ``struct = "GEN"``: the free
    parameters are the elements of the *lower-triangular Cholesky factor*
    ``L`` of the ``q x q`` covariance matrix ``G`` (``G = L L'``).  metafor
    switches on ``cholesky = TRUE`` for ``UN``/``UNR``/``GEN`` and optimises
    ``diag(L)`` plus the ``lower.tri(L)`` entries, so ``G`` is automatically
    non-negative-definite and the degenerate (``rho = -1``) boundary can be
    reached without an explicit correlation bound.  ``Z_cols`` holds one
    ``k x N`` design block per random effect (intercept then each predictor),
    so ``D = sum_ab G[a,b] * Z_a Z_b'`` equals metafor's
    ``(Z.G1 %*% G %*% t(Z.G1)) * tcrossprod(Z.G2)`` construction.
    """

    kind = "gen"

    def __init__(self, Z_cols: list[np.ndarray], q: int, outer: np.ndarray | None = None,
                 outer_labels: list[str] | None = None, var_names: list[str] | None = None,
                 name: str | None = None):
        self.Z_cols = Z_cols
        self.q = q
        self._k = Z_cols[0].shape[0]
        self._n_rho = q * (q - 1) // 2
        self._outer = np.asarray(
            outer if outer is not None else np.argmax(Z_cols[0], axis=1))
        n_lev = int(np.max(self._outer)) + 1
        self.outer_labels = outer_labels if outer_labels is not None else [str(i) for i in range(n_lev)]
        self.var_names = var_names if var_names is not None else ["intrcpt"] + [f"v{i}" for i in range(1, q)]
        self.name = name if name is not None else "gen"

    def contain_groups(self) -> list[np.ndarray]:
        return [self._outer]

    def blup(self, p: np.ndarray, W: np.ndarray, r: np.ndarray,
             X: np.ndarray, XWX_inv: np.ndarray):
        """BLUP of the random slopes (metafor: no SE for ``struct="GEN"``).

        Returns the ``N x q`` matrix of predictions with outer-group labels;
        ``se`` is ``None`` (metafor does not compute it for GEN).
        """
        G = self._G(p)
        ZG1 = np.hstack([col.sum(axis=1).reshape(-1, 1) for col in self.Z_cols])
        levels = np.unique(self._outer)
        b = np.zeros((len(levels), self.q))
        for j, lvl in enumerate(levels):
            incl = np.where(self._outer == lvl)[0]
            b[j] = (G @ ZG1[incl].T @ W[np.ix_(incl, incl)] @ r[incl].reshape(-1, 1)).ravel()
        return [self.outer_labels[j] for j in levels], b, None

    @property
    def n_params(self) -> int:
        return self.q + self._n_rho

    @property
    def vc_names(self) -> list[str]:
        # the lower-triangular Cholesky factor, as metafor's cholesky=TRUE
        return ["chol"] * self.n_params

    @property
    def n_vc(self) -> int:
        return self.q

    def _G(self, p: np.ndarray) -> np.ndarray:
        L = np.zeros((self.q, self.q))
        np.fill_diagonal(L, p[: self.q])
        idx = self.q
        for i in range(1, self.q):
            for j in range(i):
                L[i, j] = p[idx]
                idx += 1
        return L @ L.T

    def make_D(self, p: np.ndarray) -> np.ndarray:
        G = self._G(p)
        D = np.zeros((self._k, self._k))
        for a in range(self.q):
            Za = self.Z_cols[a]
            for b in range(self.q):
                D += G[a, b] * (Za @ self.Z_cols[b].T)
        return D

    def init(self, tau2_val: float) -> np.ndarray:
        # metafor: tau2.init = rep(total/(#variance components), q),
        #          rho.init  = rep(0.5, q(q-1)/2), then G <- chol(G).
        G = np.full((self.q, self.q), 0.0)
        np.fill_diagonal(G, tau2_val)
        for i in range(1, self.q):
            for j in range(i):
                G[i, j] = G[j, i] = 0.5 * tau2_val
        L = np.linalg.cholesky(G)
        p = np.zeros(self.n_params)
        p[: self.q] = np.diag(L)
        idx = self.q
        for i in range(1, self.q):
            for j in range(i):
                p[idx] = L[i, j]
                idx += 1
        return p

    def bounds(self) -> list:
        return [(None, None)] * self.n_params

    def report(self, p: np.ndarray):
        G = self._G(p)
        tau2 = [float(x) for x in np.diag(G)]
        rho = []
        for i in range(1, self.q):
            for j in range(i):
                denom = math.sqrt(max(G[i, i] * G[j, j], 0.0))
                if denom <= 0.0:
                    rho.append(0.0)
                else:
                    rho.append(float(G[i, j] / denom))
        return [], tau2, rho


class _UNComp:
    """``~ inner | outer`` with an *unstructured* covariance matrix ``G`` over
    the inner factor levels (``struct = "UN"``).

    ``G`` is reconstructed as ``G = S R S`` (metafor's ``.con.vcov.UN``)
    where ``S = diag(sqrt(tau2))`` and ``R`` is the correlation matrix whose
    lower-triangular entries hold ``rho`` in metafor's ``[lower.tri]``
    (column-major) order.  The free parameter vector ``p`` holds only the
    free parameters: inner levels with ``k == 1`` have their ``tau2`` fixed
    to 0 and inner-level pairs that never co-occur have their ``rho`` fixed
    to 0 (metafor ``.process.G.afterrmna``), so both are excluded from ``p``
    and re-inserted as 0.  ``D = (Z1 G Z1') * (Z2 Z2')``.
    """

    kind = "un"

    def __init__(self, inner_idx: np.ndarray, outer_groups: np.ndarray, J: int,
                 inner_labels: list[str] | None = None,
                 outer_labels: list[str] | None = None,
                 name: str | None = None,
                 fixed_tau2: np.ndarray | None = None,
                 fixed_rho: np.ndarray | bool | None = None):
        self.inner_idx = inner_idx
        self.outer_groups = outer_groups
        self.J = J
        self.inner_labels = inner_labels if inner_labels is not None else [str(i) for i in range(J)]
        n_outer = int(outer_groups.max()) + 1
        self.outer_labels = outer_labels if outer_labels is not None else [str(i) for i in range(n_outer)]
        self.name = name if name is not None else "un"
        self._same = (outer_groups[:, None] == outer_groups[None, :]).astype(float)
        self._n_rho = J * (J - 1) // 2
        # column-major lower-triangular (i, j) pairs, matching R's [lower.tri]
        self._rho_pairs = [(i, j) for j in range(J - 1) for i in range(j + 1, J)]
        if fixed_tau2 is not None:
            self._fixed_tau2 = np.asarray(fixed_tau2, dtype=bool).reshape(-1)
        else:
            self._fixed_tau2 = np.zeros(J, dtype=bool)
        if fixed_rho is not None:
            self._fixed_rho = np.asarray(fixed_rho, dtype=bool).reshape(-1)
        else:
            self._fixed_rho = np.zeros(self._n_rho, dtype=bool)
        self._free_tau2 = ~self._fixed_tau2
        self._free_rho = ~self._fixed_rho
        self._n_free_tau2 = int(self._free_tau2.sum())
        self._n_free_rho = int(self._free_rho.sum())

    def contain_groups(self) -> list[np.ndarray]:
        return [self.inner_idx, self.outer_groups]

    def _full_tau2(self, p: np.ndarray) -> np.ndarray:
        tau2 = np.zeros(self.J)
        tau2[self._free_tau2] = p[: self._n_free_tau2]
        return tau2

    def _full_rho(self, p: np.ndarray) -> np.ndarray:
        rho = np.zeros(self._n_rho)
        rho[self._free_rho] = p[self._n_free_tau2:self._n_free_tau2 + self._n_free_rho]
        return rho

    def _G(self, p: np.ndarray) -> np.ndarray:
        tau2 = self._full_tau2(p)
        rho = self._full_rho(p)
        sq = np.sqrt(tau2)
        R = np.eye(self.J)
        for k, (i, j) in enumerate(self._rho_pairs):
            R[i, j] = R[j, i] = rho[k]
        return np.outer(sq, sq) * R

    def blup(self, p: np.ndarray, W: np.ndarray, r: np.ndarray,
             X: np.ndarray, XWX_inv: np.ndarray):
        G = self._G(p)
        k = self.inner_idx.shape[0]
        Z1 = np.eye(self.J)[self.inner_idx]
        N = int(self.outer_groups.max()) + 1
        Z2 = np.eye(N)[self.outer_groups]
        Gfull = (Z1 @ G @ Z1.T) * (Z2 @ Z2.T)
        GW = Gfull @ W
        pred = (GW @ r).ravel()
        I = np.eye(k)
        H = X @ XWX_inv @ X.T @ W
        vpred = Gfull - GW @ (I - H) @ Gfull
        se = np.sqrt(np.maximum(np.diag(vpred), 0.0))
        labels = [f"{self.inner_labels[self.inner_idx[i]]} | {self.outer_labels[self.outer_groups[i]]}"
                  for i in range(k)]
        return labels, pred, se

    @property
    def n_params(self) -> int:
        return self._n_free_tau2 + self._n_free_rho

    @property
    def vc_names(self) -> list[str]:
        return ["tau2"] * self._n_free_tau2 + ["rho"] * self._n_free_rho

    @property
    def n_vc(self) -> int:
        return self.J

    def make_D(self, p: np.ndarray) -> np.ndarray:
        G = self._G(p)
        Gi = G[np.ix_(self.inner_idx, self.inner_idx)]
        return Gi * self._same

    def init(self, tau2_val: float) -> np.ndarray:
        return np.concatenate([
            np.full(self._n_free_tau2, float(tau2_val)),
            np.full(self._n_free_rho, 0.5),
        ])

    def bounds(self) -> list:
        return (
            [(0.0, None)] * self._n_free_tau2
            + [(-1.0 + 1e-9, 1.0 - 1e-9)] * self._n_free_rho
        )

    # -- the correlation cone: checking it, and staying inside it ---------- #
    def corr_matrix(self, p: np.ndarray) -> np.ndarray:
        """The implied correlation matrix, without checking whether it is one."""
        R = np.eye(self.J)
        rho = self._full_rho(p)
        for k, (i, j) in enumerate(self._rho_pairs):
            R[i, j] = R[j, i] = rho[k]
        return R

    def is_valid(self, p: np.ndarray) -> bool:
        """Whether the implied correlation matrix is positive definite.

        The per-pair box bounds are necessary and not sufficient: with three or
        more inner levels a vector of correlations that is inside ``(-1, 1)``
        term by term can still imply an indefinite ``R``, and the likelihood is
        then evaluated at a point that is not a covariance matrix at all.
        Nothing in the box constraint stops the optimiser walking there, which
        is how a fit can end up reporting two correlations of exactly one and a
        third one negative.
        """
        if self._n_free_rho == 0:
            return True
        try:
            np.linalg.cholesky(self.corr_matrix(p))
        except np.linalg.LinAlgError:
            return False
        return True

    def to_chol_vec(self, p: np.ndarray) -> np.ndarray:
        """External ``(tau2, rho)`` -> the column-major lower Cholesky vector.

        Only meaningful when every parameter is free: ``G = L L'`` is then
        positive semidefinite by construction for any ``L``, the map is smooth
        everywhere including the boundary, and a variance going to zero is the
        corresponding row of ``L`` going to zero -- no dangling correlations.
        """
        tau2 = self._full_tau2(p)
        rho = self._full_rho(p)
        sd = np.sqrt(np.maximum(tau2, 1e-6))
        R = np.eye(self.J)
        for k, (i, j) in enumerate(self._rho_pairs):
            R[i, j] = R[j, i] = rho[k]
        eigenvalues, eigenvectors = np.linalg.eigh(R)
        if eigenvalues.min() < 1e-4:
            # an invalid starting point is snapped to the cone, because the
            # conversion has to be possible for any ``init``
            eigenvalues = np.maximum(eigenvalues, 1e-4)
            R = eigenvectors @ np.diag(eigenvalues) @ eigenvectors.T
            d = np.sqrt(np.diag(R))
            R = R / np.outer(d, d)
        L = np.linalg.cholesky(np.outer(sd, sd) * R)
        return np.array([L[i, j] for j in range(self.J)
                         for i in range(j, self.J)], dtype=float)

    def from_chol_vec(self, l_vec: np.ndarray) -> np.ndarray:
        """The Cholesky vector -> external ``(tau2, rho)``."""
        L = np.zeros((self.J, self.J))
        idx = 0
        for j in range(self.J):
            for i in range(j, self.J):
                L[i, j] = l_vec[idx]
                idx += 1
        G = L @ L.T
        tau2 = np.diag(G).copy()
        sd = np.sqrt(tau2)
        rho = np.zeros(self._n_rho)
        for k, (i, j) in enumerate(self._rho_pairs):
            denom = sd[i] * sd[j]
            if denom > 1e-12:
                rho[k] = float(np.clip(G[i, j] / denom, -1.0, 1.0))
        out = np.zeros(self._n_free_tau2 + self._n_free_rho)
        out[: self._n_free_tau2] = tau2[self._free_tau2]
        out[self._n_free_tau2:] = rho[self._free_rho]
        return out

    def chol_bounds(self) -> list:
        """Bounds for the Cholesky vector: non-negative diagonal, free elsewhere."""
        return (
            [(0.0, None)] * self._n_free_tau2
            + [(None, None)] * self._n_free_rho
        )

    def report(self, p: np.ndarray):
        tau2 = [float(x) for x in self._full_tau2(p)]
        rho = [float(x) for x in self._full_rho(p)]
        return [], tau2, rho


def _mv_loglik(
    D_list: list[np.ndarray],
    Y: np.ndarray,
    V: np.ndarray,
    X: np.ndarray,
    method: str,
) -> float:
    """Marginal (restricted) log-likelihood for the multilevel model.

    ``M = V + sum_j D_j`` where each ``D_j`` is the (already scaled) random
    effect contribution.  Returns the *positive* objective (negative
    log-likelihood) for minimisation, matching metafor's ``.ll.rma.mv``
    (with ``REMLf = TRUE``, metafor's default).
    """
    k = X.shape[0]
    p = X.shape[1]
    M = V + sum(D_list)
    sign, logdetM = np.linalg.slogdet(M)
    if sign <= 0 or not np.isfinite(logdetM):
        return 1e300
    W = np.linalg.inv(M)
    XWX = X.T @ W @ X
    try:
        stXWX = np.linalg.inv(XWX)
    except np.linalg.LinAlgError:
        return 1e300
    beta = stXWX @ X.T @ W @ Y
    r = Y - X @ beta
    rss = float((r.T @ W @ r).item())
    if method == "ML":
        ll = -0.5 * k * math.log(2 * math.pi) - 0.5 * logdetM - 0.5 * rss
    else:  # REML, REMLf = TRUE
        _, logdetXX = np.linalg.slogdet(X.T @ X)
        _, logdetXWX = np.linalg.slogdet(XWX)
        ll = (
            -0.5 * (k - p) * math.log(2 * math.pi)
            + 0.5 * logdetXX
            - 0.5 * logdetM
            - 0.5 * logdetXWX
            - 0.5 * rss
        )
    return -float(ll)


def _fmt_level(lv) -> str:
    """Format a factor level (a single value or a tuple) as a label."""
    if isinstance(lv, tuple):
        return "/".join(str(x) for x in lv)
    return str(lv)


def _levels_k(inner_idx: np.ndarray, outer_groups: np.ndarray, J: int) -> np.ndarray:
    """metafor ``g.levels.k``: number of distinct outer levels per inner level."""
    out = np.zeros(J, dtype=int)
    for j in range(J):
        out[j] = len(set(outer_groups[inner_idx == j]))
    return out


def _cooccur_comb_k(inner_idx: np.ndarray, outer_groups: np.ndarray, J: int) -> np.ndarray:
    """metafor ``g.levels.comb.k``: co-occurrence counts for the ``J(J-1)/2``
    lower-triangular inner-level pairs, in R's ``[lower.tri]`` (column-major)
    order."""
    n_outer = int(outer_groups.max()) + 1
    C = np.zeros((n_outer, J), dtype=float)
    np.add.at(C, (outer_groups, inner_idx), 1.0)
    Cb = (C > 0).astype(float)
    comb = Cb.T @ Cb  # J x J: comb[i, j] = # outer groups containing both i and j
    out = []
    for j in range(J - 1):
        for i in range(j + 1, J):
            out.append(comb[i, j])
    return np.array(out)


def _build_components(descs: list[dict], data, k: int,
                      rho=None) -> list:
    """Instantiate random-effect component objects from descriptors.

    ``rho`` (scalar or per-term list) fixes the correlation of the structured
    ``~ inner | outer`` terms, mirroring ``rma.mv`` where a supplied ``rho`` is
    a fixed value rather than a starting value.
    """
    comps: list[Any] = []
    rho_list = None
    if rho is not None:
        rho_list = list(rho) if isinstance(rho, (list, tuple, np.ndarray)) else [rho]
    rho_pos = 0
    for desc in descs:
        kind = desc["kind"]
        if kind == "intercept":
            vals = _factor_values(data, desc["factor"])
            idx, levels = _encode_levels(vals)
            Z = np.zeros((len(vals), len(levels)), dtype=float)
            Z[np.arange(len(vals)), idx] = 1.0
            comps.append(_InterceptComp(Z, [_fmt_level(lv) for lv in levels],
                                        name=desc.get("name")))
        elif kind == "cs":
            inner_vals = _get_col(data, desc["inner"])
            outer_vals = _get_col(data, desc["outer"])
            inner_idx, inner_levels = _encode_levels(inner_vals)
            outer_groups, outer_levels = _encode_levels(outer_vals)
            comps.append(_CSComp(inner_idx, outer_groups, int(inner_idx.max()) + 1,
                                 inner_labels=[str(x) for x in inner_levels],
                                 outer_labels=[str(x) for x in outer_levels],
                                 name=desc.get("name")))
        elif kind == "struct":
            inner_vals = _get_col(data, desc["inner"])
            outer_vals = _get_col(data, desc["outer"])
            inner_idx, inner_levels = _encode_levels(inner_vals)
            outer_groups, outer_levels = _encode_levels(outer_vals)
            struct = desc["struct"]
            J = int(inner_idx.max()) + 1
            inner_values = None
            if struct == "CAR":
                # g.values: sorted unique (rounded) inner level values
                try:
                    inner_values = np.array([round(float(v), 8) for v in inner_levels],
                                            dtype=float)
                except (TypeError, ValueError):
                    inner_values = None
            fixed_tau2 = None
            if struct in ("DIAG", "HCS", "HAR"):
                # k=1 inner levels -> tau2 fixed to 0
                fixed_tau2 = _levels_k(inner_idx, outer_groups, J) == 1
            fixed_rho = None
            if rho_list:
                fixed_rho = float(rho_list[min(rho_pos, len(rho_list) - 1)])
                rho_pos += 1
            comps.append(_StructComp(
                inner_idx, outer_groups, J, struct,
                inner_values=inner_values,
                inner_labels=[str(x) for x in inner_levels],
                outer_labels=[str(x) for x in outer_levels],
                name=desc.get("name"),
                fixed_tau2=fixed_tau2,
                fixed_rho=fixed_rho))
        elif kind == "un":
            inner_vals = _get_col(data, desc["inner"])
            outer_vals = _get_col(data, desc["outer"])
            inner_idx, inner_levels = _encode_levels(inner_vals)
            outer_groups, outer_levels = _encode_levels(outer_vals)
            J = int(inner_idx.max()) + 1
            fixed_tau2 = _levels_k(inner_idx, outer_groups, J) == 1
            fixed_rho_un = _cooccur_comb_k(inner_idx, outer_groups, J) == 0
            comps.append(_UNComp(inner_idx, outer_groups, J,
                                 inner_labels=[str(x) for x in inner_levels],
                                 outer_labels=[str(x) for x in outer_levels],
                                 name=desc.get("name"),
                                 fixed_tau2=fixed_tau2,
                                 fixed_rho=fixed_rho_un))
        elif kind == "gen":
            vars_ = desc["vars"]
            no_intercept = any(v in ("0", "-1", "- 1") for v in vars_)
            pred_vars = [v for v in vars_ if v not in ("0", "-1", "- 1")]
            outer_vals = _get_col(data, desc["outer"])
            one_hot_outer = _one_hot(outer_vals)
            _, outer_levels = _encode_levels(outer_vals)
            cols = [] if no_intercept else [one_hot_outer]
            for v in pred_vars:
                x = np.asarray(_get_col(data, v), dtype=float).reshape(-1, 1)
                cols.append(one_hot_outer * x)
            var_names = ([] if no_intercept else ["intrcpt"]) + pred_vars
            comps.append(_GENComp(cols, len(cols), outer=np.argmax(one_hot_outer, axis=1),
                                  outer_labels=[str(x) for x in outer_levels],
                                  var_names=var_names, name=desc.get("name")))
        else:
            raise ValueError(f"unknown random-component kind {kind!r}")
    return comps


def _contain_ddf(components: list, X: np.ndarray, k: int, p: int) -> np.ndarray:
    """Containment denominator degrees of freedom (metafor ``.ddf.calc``).

    For each fixed-effect coefficient ``j``, the ddf is the minimum number of
    levels among the random-effect factors that *contain* ``X[:, j]`` (i.e.
    ``X[:, j]`` is constant within every level of that factor), minus ``p``,
    floored at 1.  Factors that do not contain a coefficient leave its ddf at
    the residual ``k - p``.
    """
    groups: list[np.ndarray] = []
    for c in components:
        groups.extend(c.contain_groups())
    ddf = np.full(p, float(k))
    for j in range(p):
        xj = X[:, j]
        for g in groups:
            nlev = int(np.unique(g).size)
            contained = True
            for lvl in np.unique(g):
                vals = xj[g == lvl]
                if vals.size and np.ptp(vals) > 1e-12:
                    contained = False
                    break
            if contained and nlev < ddf[j]:
                ddf[j] = nlev
    ddf = ddf - p
    ddf[ddf < 1] = 1.0
    return ddf


def rma_mv(
    yi,
    vi=None,
    V=None,
    mods=None,
    data=None,
    random: str = "~ 1 | study",
    struct: str = "CS",
    rho=None,
    method: str = "REML",
    level: float = 95.0,
    test: str = "z",
    dfs: str | np.ndarray = "residual",
    intercept: bool = True,
    sigma2=None,
    slab=None,
    cluster=None,
    warn_missing: bool = True,
    lkj_eta: float = 0.0,
) -> RmaMvResult:
    """Fit a multilevel / multivariate random-effects model (``metafor::rma.mv``).

    Args:
        yi: effect estimates (length ``k``).
        vi: sampling variances (length ``k``), used when ``V`` is None.
        V: full sampling variance-covariance matrix (``k x k``).
        mods: fixed-effect design matrix (``k x m``, without intercept), or
            None for an intercept-only model.
        data: dict-like (or pandas DataFrame) providing the grouping factors
            named in ``random``.
        random: random-effects formula(s), e.g. ``"~ 1 | study"``,
            ``"~ 1 | district/school"`` (nested intercepts),
            ``"~ year | district"`` (compound-symmetric inner factor), or
            ``"~ year | district"`` with ``struct="GEN"`` for random slopes.
        struct: variance structure for an ``~ inner | outer`` term --
            ``"CS"`` (default) or ``"GEN"`` (random slopes with unstructured
            covariance).  A list matches multiple ``~ inner | outer`` terms.
        rho: fixes the correlation of the structured term(s) instead of
            estimating it, as ``rma.mv`` does.  Scalar or one value per term.
        method: ``"REML"`` (default) or ``"ML"``.
        level: confidence level (95 -> 95% CI).
        test: ``"z"`` (default) or ``"t"`` / ``"knha"`` / ``"adhoc"``.
            ``"t"`` uses a t-distribution with ``dfs`` degrees of freedom;
            ``"knha"`` additionally scales the variance by
            ``RSS/(k-p)`` (Knapp-Hartung); ``"adhoc"`` scales by
            ``max(RSS/(k-p), 1)``.
        dfs: ``"residual"`` (default, ``k-p``), ``"contain"``, or a numeric
            value / array (one df per coefficient).
        intercept: prepend an intercept column to ``mods`` (``False`` matches
            R's ``mods = ~ x1 + x2 - 1``).
        sigma2: fix the variance components instead of estimating them (R's
            ``sigma2=``), which is what the leave-cluster-out influence
            diagnostics need.
        slab: study labels, kept on the result for the plotting methods.
        cluster: optional clustering variable, kept for the diagnostics.
        warn_missing: emit a warning naming the studies dropped because a
            value is missing (R's ``na.action`` drops them silently).
        lkj_eta: strength of an LKJ penalty on every estimated correlation
            matrix, :math:`p(R) \\propto \\det(R)^{\\eta - 1}`, added to the
            likelihood as :math:`-(\\eta - 1)\\log\\det R`.  ``0`` (default)
            is metafor's unpenalised fit.  ``\\eta = 2`` mildly prefers
            independence and, more to the point, costs nothing at the boundary
            and a great deal there: with a dozen studies the unpenalised
            estimate sits on :math:`\\pm 1` in most fits, and a region built
            from a singular :math:`\\Sigma` is built from nothing.  It is a
            penalty, not a prior — the interval is still a Wald interval and
            does not carry the penalty as if it were information.
    """
    Y = np.asarray(yi, dtype=float).reshape(-1, 1)
    k = Y.shape[0]

    if V is not None:
        V = np.asarray(V, dtype=float)
    elif vi is not None:
        V = np.diag(np.asarray(vi, dtype=float))
    else:
        raise ValueError("either 'vi' or 'V' must be supplied")

    if V.shape != (k, k):
        raise ValueError(f"V must be ({k}, {k}), got {V.shape}")

    # Drop incomplete cases (R's na.action) before anything depends on k.
    checked: dict = {"yi": Y.ravel(), "vi": np.diag(V)}
    mods_pre = None if mods is None else np.asarray(mods, dtype=float)
    if mods_pre is not None and mods_pre.ndim == 2:
        for j in range(mods_pre.shape[1]):
            checked[f"mods[{j}]"] = mods_pre[:, j]
    keep, _dropped = complete_cases(checked, context="studies", warn=warn_missing)
    if not keep.all():
        Y = Y[keep]
        V = V[np.ix_(keep, keep)]
        k = Y.shape[0]
        if mods is not None and mods_pre is not None:
            mods = mods_pre[keep]
        if slab is not None and len(list(slab)) == keep.size:
            slab = [s for s, ok in zip(slab, keep) if ok]
        if cluster is not None and np.asarray(cluster).shape[0] == keep.size:
            cluster = np.asarray(cluster)[keep]
        if data is not None:
            # The random-effects factors come from ``data``; keep it aligned.
            from .data_compat import as_columns

            data = {name: np.asarray(values)[keep]
                    for name, values in as_columns(data).items()}

    # fixed-effect design matrix
    if mods is not None:
        mods = np.asarray(mods, dtype=float)
        if mods.ndim == 1:
            mods = mods.reshape(-1, 1)
        X = np.hstack([np.ones((k, 1)), mods]) if intercept else mods
    elif intercept:
        X = np.ones((k, 1))
    else:
        raise ValueError("'intercept=False' requires 'mods'")
    p = X.shape[1]

    method = method.upper()
    if method not in ("REML", "ML"):
        raise ValueError("method must be 'REML' or 'ML'")

    lkj_eta = float(lkj_eta)
    if not np.isfinite(lkj_eta) or lkj_eta < 0.0:
        raise ValueError("'lkj_eta' must be finite and non-negative")

    # --- random-effects components ---------------------------------------
    components = _build_components(
        _parse_random_components(random, struct), data, k, rho=rho)
    n_terms = sum(c.n_params for c in components)
    n_vc = sum(c.n_vc for c in components)

    # --- initial values (metafor's total / #variance-components scheme) --
    try:
        W0 = np.linalg.inv(V)
    except np.linalg.LinAlgError:
        W0 = np.diag(1.0 / np.maximum(np.diag(V), 1e-12))
    stXW0X = np.linalg.inv(X.T @ W0 @ X)
    beta_fe = stXW0X @ X.T @ W0 @ Y
    resid = Y - X @ beta_fe
    total = max(
        0.001 * n_vc,
        float(np.var(resid, ddof=1)) - 1.0 / np.mean(1.0 / np.diag(V)),
    )
    tau2_val = total / n_vc

    init = np.concatenate([c.init(tau2_val) for c in components])
    bounds = [b for c in components for b in c.bounds()]

    def obj(params):
        D_list = []
        offset = 0
        penalty = 0.0
        for c in components:
            piece = params[offset : offset + c.n_params]
            offset += c.n_params
            valid = getattr(c, "is_valid", None)
            if valid is not None and not valid(piece):
                # Only reachable when parameters are pinned (the free path is
                # optimised on the Cholesky vector, which cannot leave the
                # cone).  A smooth quadratic barrier pushes the iterate back
                # inside instead of a hard wall, which would stall the line
                # search on a non-smooth cliff.
                lam = float(np.linalg.eigvalsh(c.corr_matrix(piece))[0])
                if lam <= 1e-6:
                    return 1e5 + 1e7 * (1e-6 - lam) ** 2
                return 1e5
            if lkj_eta > 1.0 and getattr(c, "kind", None) == "un":
                corr = c.corr_matrix(piece)
                sign, logdet = np.linalg.slogdet(corr)
                if sign <= 0 or not np.isfinite(logdet):
                    return 1e5
                # -log p(R) for p(R) ∝ det(R)^(eta - 1); at eta = 1 it is a
                # constant and the fit is the unpenalised one
                penalty += -(lkj_eta - 1.0) * float(logdet)
            D_list.append(c.make_D(piece))
        return _mv_loglik(D_list, Y, V, X, method) + penalty

    if sigma2 is not None:
        # Variance components given by the user.  ``NaN`` entries stay free
        # (this is how ``profile.rma.mv`` pins one component at a time while
        # the others are still estimated, matching R's ``sigma2=NA``).
        fixed = np.asarray(sigma2, dtype=float).ravel()
        if fixed.size != init.size:
            raise ValueError(
                f"'sigma2' has {fixed.size} value(s) but the model has "
                f"{init.size} variance component(s)")
        pinned = np.isfinite(fixed)
        if pinned.all():
            params = fixed
        else:
            def _expand(theta, _pinned=pinned, _fixed=fixed):
                out = np.empty(_fixed.size)
                out[_pinned] = _fixed[_pinned]
                out[~_pinned] = theta
                return out

            params = _expand(_mv_optimise(
                lambda th: obj(_expand(th)), init[~pinned],
                [b for b, pin in zip(bounds, pinned) if not pin],
                components, minimize, minimize_scalar))
    else:
        # The free path optimises the Cholesky vector ``G = L L'``: every
        # ``L`` gives a positive semidefinite covariance by construction, the
        # map is smooth all the way to the boundary, and a variance going to
        # zero is the corresponding row of ``L`` going to zero.  This is the
        # parametrisation metafor uses internally, and it is what keeps a fit
        # from reporting a correlation matrix with negative eigenvalues.  The
        # external ``(tau2, rho)`` vector is only assembled afterwards.
        #
        # Only needed for J >= 3: with two inner levels the single correlation
        # and its box bounds already describe the cone, and keeping the old
        # surface there preserves the measured parity with metafor exactly.
        free_un = [c for c in components
                   if getattr(c, "kind", None) == "un" and c.J >= 3
                   and not np.any(c._fixed_tau2)
                   and not np.any(c._fixed_rho)]
        if len(components) == 1 and len(free_un) == 1:
            comp = free_un[0]
            l0 = comp.to_chol_vec(init)
            l_opt = _mv_optimise(
                lambda l: obj(comp.from_chol_vec(l)),
                l0, comp.chol_bounds(), components, minimize, minimize_scalar)
            params = comp.from_chol_vec(l_opt)
        else:
            params = _mv_optimise(
                obj, init, bounds, components, minimize, minimize_scalar)

    # final fit at the optimum
    D_list = []
    offset = 0
    for c in components:
        D_list.append(c.make_D(params[offset : offset + c.n_params]))
        offset += c.n_params
    M = V + sum(D_list)
    W = np.linalg.inv(M)
    XWX = X.T @ W @ X
    vb = np.linalg.inv(XWX)
    beta = vb @ X.T @ W @ Y
    resid_m = Y - X @ beta
    rss = float((resid_m.T @ W @ resid_m).item())  # Knapp-Hartung numerator

    # --- fixed-effect tests (z / t / knha / adhoc) ----------------------
    test = test.lower()
    if test == "hksj":
        test = "knha"
    if test not in ("z", "t", "knha", "adhoc"):
        raise ValueError("test must be 'z', 't', 'knha' or 'adhoc'")

    # denominator degrees of freedom (metafor .ddf.calc)
    if isinstance(dfs, str):
        dfs = dfs.lower()
        if dfs == "residual":
            ddf = np.full(p, float(k - p))
        elif dfs == "contain":
            ddf = _contain_ddf(components, X, k, p)
        else:
            raise ValueError("dfs must be 'residual', 'contain' or numeric")
    else:
        ddf = np.broadcast_to(np.asarray(dfs, dtype=float), (p,)).astype(float)

    # Knapp-Hartung / Sidik-Jonkman variance scaling
    s2w = 1.0
    if test in ("knha", "adhoc"):
        s2w = rss / (k - p) if (k - p) > 0 and rss > np.finfo(float).eps else 0.0
        if test == "adhoc":
            s2w = max(s2w, 1.0)
        vb = s2w * vb

    se = np.sqrt(np.diag(vb))
    zval = (beta.ravel()) / se
    if test in ("t", "knha", "adhoc"):
        pval = np.array([2 * stats.t.sf(np.abs(zval[i]), df=ddf[i]) if ddf[i] > 0 else np.nan for i in range(p)])
        crit = np.array([stats.t.ppf(1 - (1 - level / 100.0) / 2.0, df=ddf[i]) if ddf[i] > 0 else np.nan for i in range(p)])
    else:
        pval = 2 * stats.norm.sf(np.abs(zval))
        crit = np.full(p, stats.norm.ppf(level / 100.0 + (1 - level / 100.0) / 2.0))
    ci_lb = beta.ravel() - crit * se
    ci_ub = beta.ravel() + crit * se

    # residual heterogeneity Q (fixed-effect scale)
    qe = float((resid.T @ W0 @ resid).item())
    qe_df = k - p
    qe_p = float(stats.chi2.sf(qe, qe_df))

    # --- fit statistics (metafor rma.mv $fit.stats) ---------------------
    parms = p + n_terms
    _, logdetV = np.linalg.slogdet(V)
    ll_qe = -0.5 * k * math.log(2 * math.pi) - 0.5 * logdetV  # V-only model
    ll_ml = -_mv_loglik(D_list, Y, V, X, "ML")
    ll_reml = -_mv_loglik(D_list, Y, V, X, "REML")

    def _aicc(n, lp):
        denom = max(n, parms + 2) - parms - 1
        return -2 * lp + 2 * parms * max(n, parms + 2) / denom

    fit_stats = {
        "ML": {
            "ll": ll_ml,
            "dev": -2 * (ll_ml - ll_qe),
            "AIC": -2 * ll_ml + 2 * parms,
            "BIC": -2 * ll_ml + parms * math.log(k),
            "AICc": _aicc(k, ll_ml),
        },
        "REML": {
            "ll": ll_reml,
            "dev": -2 * ll_reml,
            "AIC": -2 * ll_reml + 2 * parms,
            "BIC": -2 * ll_reml + parms * math.log(k - p),
            "AICc": _aicc(k - p, ll_reml),
        },
    }
    ll_fitted = fit_stats[method]["ll"]

    # collect variance-component summaries
    sigma2_vals: list[float] = []
    tau2_vals: list[float] = []
    rho_vals: list[float] = []
    offset = 0
    for c in components:
        s2, t2, r = c.report(params[offset : offset + c.n_params])
        offset += c.n_params
        sigma2_vals.extend(s2)
        tau2_vals.extend(t2)
        rho_vals.extend(r)

    return RmaMvResult(
        method=method,
        k=k,
        p=p,
        sigma2=sigma2_vals,
        beta=[float(b) for b in beta.ravel()],
        se=[float(s) for s in se],
        zval=[float(z) for z in zval],
        pval=[float(pv) for pv in pval],
        ci_lb=[float(c) for c in ci_lb],
        ci_ub=[float(c) for c in ci_ub],
        qe=qe,
        qe_df=qe_df,
        qe_p=qe_p,
        random=str(random),
        int_only=(p == 1 and intercept),
        M=M,
        vb=vb,
        X=X,
        yi=Y.ravel(),
        level=level,
        tau2=tau2_vals or None,
        rho=rho_vals or None,
        ll=ll_fitted,
        parms=parms,
        fit_stats=fit_stats,
        ddf=[float(d) for d in ddf],
        s2w=float(s2w),
        components=components,
        params=params,
        V=V,
        mods=mods,
        struct=str(struct),
        dfs=(dfs.tolist() if isinstance(dfs, np.ndarray) else dfs),
        intercept_incl=bool(intercept),
        slab=([str(v) for v in slab] if slab is not None
              else [f"Study {i + 1}" for i in range(k)]),
        cluster=(list(cluster) if cluster is not None else None),
        test=test,
        _data=_as_data_dict(data),
    )


def _as_data_dict(data):
    """Normalise the grouping data to a plain ``{name: list}`` dict."""
    if data is None:
        return None
    if isinstance(data, dict):
        return {str(k): (list(v) if hasattr(v, "__len__") else v)
                for k, v in data.items()}
    cols = getattr(data, "columns", None)
    if cols is not None:
        return {str(c): list(data[c]) for c in cols}
    raise TypeError("argument 'data' must be a dict or a DataFrame")


def _mv_optimise(obj, init, bounds, components, minimize, minimize_scalar):
    """Maximise the marginal likelihood over the variance components.

    Split out of :func:`rma_mv` so that the diagnostics can refit on subsets
    while the user-facing code path stays readable.
    """
    lbfgsb_opts = {"ftol": 1e-12, "gtol": 1e-10, "maxiter": 20000, "maxls": 60}
    opt = minimize(obj, init, method="L-BFGS-B", bounds=bounds,
                   options=lbfgsb_opts)

    # GEN (unstructured random slopes) is ill-conditioned on the raw scale of
    # the inner predictors: the marginal likelihood has a shallow ridge near
    # the degenerate (rho = -1) boundary where L-BFGS-B stalls in a poor local
    # optimum.  metafor's nlminb traverses this ridge; Nelder-Mead is the most
    # reliable scipy fallback and is only needed when a GEN term is present.
    #
    # The same stall happens to a UN covariance, and it is worse there: the
    # optimiser can stop with one variance component *at zero* and the other
    # interior, which is a boundary point rather than a maximum.  Measured on a
    # two-metric fixture (tests/data/prediction_joint.csv, regenerated
    # 2026-10-26): REML returned Sigma = [[0, 0], [0, 0.1003]] while the
    # univariate fit of the same first metric estimated tau2 = 0.0241, and
    # metafor estimated [[0.0240, -0.0361], [-0.0361, 0.0799]]. ML on the same
    # data found the interior point, so the likelihood is fine and the search
    # was not.  A solution with at least one parameter on its lower bound and at
    # least one interior is the signature of that stall, and only those fits pay
    # for the fallback.
    stalled = False
    if any(c.kind == "un" for c in components):
        lo = np.array([b[0] if b[0] is not None else -np.inf for b in bounds])
        at_lo = np.isclose(np.asarray(opt.x, dtype=float), lo, atol=1e-10)
        stalled = bool(at_lo.any() and not at_lo.all())

    if stalled or any(c.kind == "gen" for c in components):
        opt_nm = minimize(
            obj, init, method="Nelder-Mead",
            options={"xatol": 1e-10, "fatol": 1e-12,
                     "maxiter": 100000, "maxfev": 100000},
        )
        if obj(opt_nm.x) < obj(opt.x):
            opt = minimize(obj, opt_nm.x, method="L-BFGS-B", bounds=bounds,
                           options=lbfgsb_opts)

    # With a single variance component the profile likelihood is very flat
    # around its maximum and L-BFGS-B stops a few 1e-8 short of it.  Golden
    # section on the 1-D objective polishes the estimate to machine precision.
    if len(init) == 1 and bounds[0][1] is None:
        best = float(opt.x[0])
        best_f = float(obj(opt.x))
        lo = max(0.0, best - max(1e-3, 0.5 * abs(best)))
        hi = best + max(1e-3, 0.5 * abs(best))
        for _ in range(4):
            if not hi > lo:
                break
            sc = minimize_scalar(
                lambda t: float(obj(np.array([t]))),
                bounds=(lo, hi),
                method="bounded",
                options={"xatol": 1e-14 * max(1.0, abs(hi))},
            )
            if sc.fun < best_f:
                best, best_f = float(sc.x), float(sc.fun)
            span = max(1e-8, 0.02 * max(1.0, abs(best)))
            lo, hi = max(0.0, best - span), best + span
        return np.array([best])
    return opt.x


# The sandwich estimator moved to ``pimeta.rve`` once it had to serve the
# univariate fit too; a univariate ``rma`` is just an ``rma.mv`` whose marginal
# var-cov matrix happens to be diagonal, so one implementation covers both.
# Re-exported here because ``robust()`` used to be reachable as
# ``pimeta.rma_mv.robust``.
from .rve import robust  # noqa: E402,F401


def anova_rma(full: RmaMvResult, reduced: RmaMvResult, df: int | None = None):
    """Likelihood-ratio test between two nested ``rma_mv`` fits.

    Replicates ``metafor::anova.rma`` for two models.  The model with more
    estimated parameters is taken as the *full* model; the test statistic is
    ``LRT = deviance(reduced) - deviance(full)`` evaluated under the fitted
    ``method`` (``ML`` by default in metafor; the ``ll.QE`` constant cancels,
    so this equals ``-2 * (ll.reduced - ll.full)``).  ``pval`` uses a chi-square
    with ``df = parms.full - parms.reduced``.

    Returns a dict mirroring metafor's two-row table:
        ``{"full": {...}, "reduced": {...}, "LRT": ..., "pval": ..., "df": ...}``
    """
    if full.method != reduced.method:
        raise ValueError("full and reduced model must use the same method")
    if full.k != reduced.k:
        raise ValueError("models must be fitted to the same data")

    parms_f = int(full.parms or 0)
    parms_r = int(reduced.parms or 0)
    if parms_f < parms_r:  # ensure 'full' has more parameters
        full, reduced = reduced, full
        parms_f, parms_r = parms_r, parms_f

    method = full.method
    stats_f = full.fit_stats or {}
    stats_r = reduced.fit_stats or {}
    dev_f = float(stats_f.get(method, {}).get("dev") or 0.0)
    dev_r = float(stats_r.get(method, {}).get("dev") or 0.0)
    lrt = max(dev_r - dev_f, 0.0)
    parms_diff = (parms_f - parms_r) if df is None else df
    pval = float(stats.chi2.sf(lrt, parms_diff))

    def _row(r: RmaMvResult) -> dict:
        fs = (r.fit_stats or {}).get(method, {})
        return {
            "df": r.parms,
            "AIC": fs["AIC"],
            "BIC": fs["BIC"],
            "AICc": fs["AICc"],
            "logLik": fs["ll"],
            "QE": r.qe,
        }

    return {
        "full": _row(full),
        "reduced": _row(reduced),
        "LRT": lrt,
        "pval": pval,
        "df": parms_diff,
    }


def ranef(result: RmaMvResult, level: float | None = None) -> dict:
    """Best linear unbiased predictions (BLUP) of the random effects.

    Replicates ``metafor::ranef.rma.mv``.  Returns a dict keyed by the random
    effect term name (``"district"``, ``"district/school"``, ``"~ year |
    district"``, ...).  Each entry holds ``labels`` plus the prediction vector
    and its ``se`` / ``pi.lb`` / ``pi.ub`` (prediction interval).  For
    ``struct="GEN"`` terms metafor does not report ``se``; the entry then
    holds a ``pred`` matrix (outer groups x random-effect variables) with
    ``var_names`` columns.
    """
    if result.components is None or result.params is None:
        raise ValueError("ranef() requires a rma_mv() fit (BLUP info not stored)")

    M = result.M
    X = result.X
    beta = np.asarray(result.beta, dtype=float).reshape(-1, 1)
    Y = result.yi.reshape(-1, 1)
    W = np.linalg.inv(M)
    XWX_inv = np.linalg.inv(X.T @ W @ X)
    r = Y - X @ beta

    lvl = result.level if level is None else level
    crit = stats.norm.ppf(1 - (1 - lvl / 100.0) / 2.0)

    out: dict = {}
    offset = 0
    for c in result.components:
        p = result.params[offset : offset + c.n_params]
        offset += c.n_params
        labels, b, se = c.blup(p, W, r, X, XWX_inv)
        if c.kind == "intercept":
            pi_lb = b - crit * se
            pi_ub = b + crit * se
            out[c.name] = {
                "labels": list(labels),
                "intrcpt": np.asarray(b, dtype=float),
                "se": np.asarray(se, dtype=float),
                "pi_lb": np.asarray(pi_lb, dtype=float),
                "pi_ub": np.asarray(pi_ub, dtype=float),
            }
        elif c.kind in ("cs", "struct", "un"):
            # deduplicate the per-observation predictions by unique inner|outer
            seen: dict = {}
            keep_idx: list[int] = []
            for i, lab in enumerate(labels):
                if lab not in seen:
                    seen[lab] = i
                    keep_idx.append(i)
            keep = np.asarray(keep_idx, dtype=int)
            pi_lb = b[keep] - crit * se[keep]
            pi_ub = b[keep] + crit * se[keep]
            out[c.name] = {
                "labels": [labels[i] for i in keep],
                "intrcpt": np.asarray(b[keep], dtype=float),
                "se": np.asarray(se[keep], dtype=float),
                "pi_lb": np.asarray(pi_lb, dtype=float),
                "pi_ub": np.asarray(pi_ub, dtype=float),
            }
        elif c.kind == "gen":
            out[c.name] = {
                "labels": list(labels),
                "var_names": list(c.var_names),
                "pred": np.asarray(b, dtype=float),
                "se": None,
            }
    return out
