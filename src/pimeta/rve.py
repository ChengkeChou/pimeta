"""Cluster-robust (sandwich) variance estimation -- ``metafor::robust``.

Two estimators live here:

``robust(fit, cluster=...)`` with ``clubSandwich=False``
    metafor's own sandwich.  The bread is the model's variance-covariance
    matrix times the weighted design, the meat is the block-diagonal matrix of
    within-cluster residual outer products, and ``adjust`` selects the
    small-sample factor: ``True`` gives CR1, ``False`` gives CR0,
    ``"Stata"``/``"Stata1"`` gives CR1.S1 and ``"Stata2"`` gives CR1.S2.

``robust(fit, cluster=..., clubSandwich=True)``
    the CR2/CR3 bias-reduced sandwiches of Bell & McCaffrey, as implemented in
    ``clubSandwich::vcovCR``, with the Satterthwaite degrees of freedom from
    ``clubSandwich::coef_test``.

Both fit classes go through the same code: a univariate ``rma`` fit is only a
special case of ``rma.mv`` in which the marginal var-cov matrix happens to be
diagonal.  :class:`_View` is the small adapter that makes that explicit, so
there is one sandwich implementation rather than two that can drift apart.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from scipy import stats

__all__ = ["robust"]

#: Below this diagonal element the robust variance is treated as degenerate and
#: the row/column is blanked to NaN, as metafor does.
_DEGENERATE = 100.0 * np.finfo(float).eps


class _View:
    """The handful of quantities a sandwich needs, from either fit class.

    Attributes:
        X: fixed-effects design matrix (``k x p``).
        yi: outcome vector (``k``).
        beta: coefficient column vector (``p x 1``).
        M: marginal var-cov matrix of the estimates (``k x k``).
        vb: the model's own var-cov matrix (``p x p``).
        p: number of coefficients.
        k: number of estimates.
        level: confidence level, normalised to a percentage (``95``).
        btt: 1-based indices tested by the omnibus statistic.
        multilevel: ``True`` for ``rma.mv`` fits (chi-square omnibus) and
            ``False`` for ``rma.uni`` fits (F omnibus).
    """

    __slots__ = ("X", "yi", "beta", "M", "vb", "p", "k", "level", "btt",
                 "multilevel", "stored_cluster")

    def __init__(self, X, yi, beta, M, vb, p, k, level, btt, multilevel,
                 stored_cluster):
        self.X = np.asarray(X, dtype=float)
        self.yi = np.asarray(yi, dtype=float).ravel()
        self.beta = np.asarray(beta, dtype=float).reshape(-1, 1)
        self.M = np.asarray(M, dtype=float)
        self.vb = np.asarray(vb, dtype=float)
        self.p = int(p)
        self.k = int(k)
        self.level = _percent(level)
        self.btt = [int(i) for i in btt]
        self.multilevel = bool(multilevel)
        self.stored_cluster = stored_cluster

    @classmethod
    def of(cls, result: Any) -> _View:
        """Build the adapter for a fitted ``rma`` or ``rma.mv`` result."""
        # rma.mv stores the marginal matrix and a column vector of coefficients;
        # rma.uni stores neither, but its weights pin the matrix down exactly
        # (W = 1/(vi + tau2), so M = diag(1/w)).
        if hasattr(result, "M") and hasattr(result, "beta"):
            return cls(
                X=result.X,
                yi=result.yi,
                beta=result.beta,
                M=result.M,
                vb=result.vb,
                p=result.p,
                k=result.k,
                level=result.level,
                btt=getattr(result, "btt", None) or _default_btt(result),
                multilevel=True,
                stored_cluster=getattr(result, "cluster", None),
            )

        weights = np.asarray(result.weights, dtype=float).ravel()
        if np.any(weights <= 0):
            raise ValueError(
                "the fit has non-positive weights, so its marginal var-cov "
                "matrix is not invertible; robust() cannot proceed"
            )
        return cls(
            X=result.X,
            yi=result.yi,
            beta=result.b,
            M=np.diag(1.0 / weights),
            vb=result.vb,
            p=result.p,
            k=result.k,
            level=result.level,
            btt=getattr(result, "btt", None) or _default_btt(result),
            multilevel=False,
            stored_cluster=getattr(result, "cluster", None),
        )


def _percent(level: float) -> float:
    """Accept both ``0.95`` and ``95`` as the confidence level."""
    value = float(level)
    return value * 100.0 if value <= 1.0 else value


def _default_btt(result: Any) -> list[int]:
    """metafor's default omnibus test: the intercept, or all slopes."""
    if getattr(result, "int_only", False):
        return [1]
    return list(range(2, int(result.p) + 1))


# --------------------------------------------------------------------------- #
# small linear-algebra helpers
# --------------------------------------------------------------------------- #
def _cluster_groups(cluster: np.ndarray) -> list[np.ndarray]:
    """Row indices of each cluster, in first-appearance order."""
    seen: list = []
    idx: dict = {}
    for c in cluster:
        if c not in idx:
            idx[c] = len(seen)
            seen.append(c)
    groups: list[list] = [[] for _ in seen]
    for i, c in enumerate(cluster):
        groups[idx[c]].append(i)
    return [np.asarray(g, dtype=int) for g in groups]


def _block_diag(blocks: list[np.ndarray]) -> np.ndarray:
    """Assemble a block-diagonal matrix from square blocks."""
    total = sum(b.shape[0] for b in blocks)
    out = np.zeros((total, total))
    offset = 0
    for block in blocks:
        size = block.shape[0]
        out[offset : offset + size, offset : offset + size] = block
        offset += size
    return out


def _sym_matrix_power(A: np.ndarray, power: float) -> np.ndarray:
    """Symmetric matrix power via eigendecomposition (clubSandwich ``matrix_power``)."""
    evals, evecs = np.linalg.eigh(A)
    evals_p = np.where(evals > 1e-12, evals**power, 0.0)
    return (evecs * evals_p) @ evecs.T


def _cr_vcov(view: _View, cluster: np.ndarray, cr_type: str):
    """clubSandwich CR2/CR3 sandwich plus the Satterthwaite intermediates.

    Replicates ``clubSandwich::vcovCR`` with ``inverse_var=TRUE``: the working
    covariance is the full marginal matrix ``M``.

    Returns:
        ``(vb, E_list, X_list, Omega_list, B)`` where the lists hold one block
        per cluster and ``B`` is the bread ``(X' M^-1 X)^-1``.
    """
    X, M, yi, beta, p = view.X, view.M, view.yi, view.beta, view.p

    W = np.linalg.inv(M)
    B = np.linalg.inv(X.T @ W @ X)

    E_list: list[np.ndarray] = []
    X_list: list[np.ndarray] = []
    Omega_list: list[np.ndarray] = []
    meat = np.zeros((p, p))
    for g in _cluster_groups(cluster):
        Xj = X[g]
        Mj = M[np.ix_(g, g)]
        Wj = W[np.ix_(g, g)]
        nj = len(g)
        XWj = Xj.T @ Wj                      # p x n_j
        Hj = Xj @ B @ XWj                    # n_j x n_j leverage
        IHj = np.eye(nj) - Hj
        if cr_type == "CR2":
            L = np.linalg.cholesky(Mj)       # lower; R (upper) = L.T
            Gj = L.T @ IHj @ Mj @ L
            Aj = L @ _sym_matrix_power(Gj, -0.5) @ L.T
        elif cr_type == "CR3":
            Aj = np.linalg.inv(IHj)
        else:
            raise ValueError(f"unknown vcov type: {cr_type}")
        Ej = XWj @ Aj                        # p x n_j
        ej = yi[g] - (Xj @ beta).ravel()
        cj = Ej @ ej                         # p x 1
        meat += np.outer(cj, cj)
        E_list.append(Ej)
        X_list.append(Xj)
        Omega_list.append(Mj)

    return B @ meat @ B, E_list, X_list, Omega_list, B


def _satterthwaite_dof(B, E_list, X_list, Omega_list) -> np.ndarray:
    """clubSandwich Satterthwaite degrees of freedom (inverse_var=TRUE path).

    ``nu_i = tr(P_i)^2 / sum(P_i^2)`` with ``P_i`` built from the ``get_GH`` /
    ``get_P_array`` quantities for the case where no augmented model matrix is
    supplied.
    """
    p = B.shape[0]
    J = len(E_list)
    H_array = np.zeros((p, p, J))
    G_list: list[np.ndarray] = []
    for j in range(J):
        ME = B @ E_list[j]                   # p x n_j
        Lj = np.linalg.cholesky(Omega_list[j])
        G_list.append(ME @ Lj)               # p x n_j
        H_array[:, :, j] = ME @ X_list[j] @ np.linalg.cholesky(B)

    dof = np.empty(p)
    for i in range(p):
        Hi = H_array[i, :, :]                # (u=p, J)
        P = -Hi.T @ Hi
        diag_term = np.array([np.sum(G_list[j][i, :] ** 2) for j in range(J)])
        P[np.diag_indices(J)] += diag_term
        tr = np.trace(P)
        sumsq = np.sum(P**2)
        dof[i] = tr * tr / sumsq
    return dof


def _coef_dof(coef_test: str, p: int, n: int, vb: np.ndarray | None = None,
              B=None, E_list=None, X_list=None, Omega_list=None) -> np.ndarray:
    """Denominator degrees of freedom for the chosen coefficient test."""
    if coef_test == "Satterthwaite":
        return _satterthwaite_dof(B, E_list, X_list, Omega_list)
    if coef_test == "naive-t":
        return np.full(p, float(n - 1))
    if coef_test == "naive-tp":
        return np.full(p, float(n - p))
    if coef_test == "z":
        return np.full(p, np.inf)
    raise ValueError(f"unknown coef_test: {coef_test!r}")


def _omnibus(view: _View, vb: np.ndarray, dfs) -> tuple:
    """Wald omnibus test of ``btt``, with metafor's reference distribution."""
    idx = [i - 1 for i in view.btt if 0 < i <= view.p]
    if not idx:
        return None, None, None
    b = view.beta.ravel()[idx]
    vb_tt = vb[np.ix_(idx, idx)]
    try:
        stat = float(b @ np.linalg.solve(vb_tt, b))
    except np.linalg.LinAlgError:
        return float("nan"), None, float("nan")
    m = len(idx)
    if view.multilevel:
        return stat, m, float(stats.chi2.sf(stat, m))
    stat = stat / m
    denom = _min_finite(dfs)
    return stat, [m, denom], float(stats.f.sf(stat, m, denom))


def _min_finite(dfs) -> float:
    arr = np.asarray(dfs, dtype=float).ravel()
    finite = arr[np.isfinite(arr)]
    return float(finite.min()) if finite.size else np.inf


# --------------------------------------------------------------------------- #
# public entry point
# --------------------------------------------------------------------------- #
def robust(
    result: Any,
    cluster=None,
    adjust: bool | str = True,
    clubSandwich: bool = False,
    vcov: str = "CR2",
    coef_test: str = "Satterthwaite",
    level: float | None = None,
):
    """Cluster-robust inference for an ``rma`` or ``rma.mv`` fit.

    Args:
        result: a fitted result.  Both the univariate and the multilevel model
            are accepted; the sandwich is the same, only the marginal
            var-cov matrix differs.
        cluster: cluster label per estimate.  Required for a univariate fit
            (``robust.rma.uni`` has no default either); a multilevel fit
            reuses the ``cluster=`` it was fitted with when omitted.
        adjust: small-sample adjustment for the metafor sandwich.
            ``True`` (default) is CR1 (``n/(n-p)``), ``False`` is CR0,
            ``"Stata"``/``"Stata1"`` is CR1.S1 (``n/(n-1) * (k-1)/(k-p)``)
            and ``"Stata2"`` is CR1.S2 (``n/(n-1)``).  Ignored when
            ``clubSandwich=True``.
        clubSandwich: use the CR2/CR3 bias-reduced sandwich and the
            Satterthwaite degrees of freedom instead of metafor's own.
        vcov: ``"CR2"`` (default) or ``"CR3"``; only read when
            ``clubSandwich=True``.
        coef_test: ``"Satterthwaite"`` (default), ``"naive-t"``,
            ``"naive-tp"`` or ``"z"``.
        level: confidence level; ``None`` reuses the fit's.  Both ``0.95`` and
            ``95`` are accepted.

    Returns:
        A dict with ``vbest``, ``beta``, ``se``, ``zval``, ``pval``, ``ci_lb``,
        ``ci_ub``, ``dfs``, ``n``, ``vb`` and -- for univariate fits -- the
        omnibus test ``QM``/``QMdf``/``QMp``.  The omnibus entries are ``None``
        on the clubSandwich path, where clubSandwich's own small-sample
        reference distribution would be needed to report them honestly.

    Raises:
        ValueError: when no cluster is supplied, when the cluster vector does
            not match the number of estimates, or when there are not more
            clusters than coefficients.
    """
    view = _View.of(result)
    if cluster is None:
        cluster = view.stored_cluster
    if cluster is None:
        raise ValueError("Argument 'cluster' must be specified.")
    cluster = np.asarray(cluster).ravel()
    if cluster.shape[0] != view.k:
        raise ValueError(
            "cluster must have one entry per observation"
            f" (got {cluster.shape[0]}, need {view.k})"
        )

    n = len(np.unique(cluster))
    p = view.p
    k = view.k
    dfs = n - p
    lvl = view.level if level is None else _percent(level)
    ddf_out: list[float] | float

    if clubSandwich:
        vb_robust, E_list, X_list, Omega_list, B = _cr_vcov(view, cluster, vcov)
        ddf = _coef_dof(coef_test, p, n, B=B, E_list=E_list, X_list=X_list,
                        Omega_list=Omega_list)
        vbest = vcov
        ddf_out = [float(d) for d in ddf]
    else:
        if dfs <= 0:
            raise ValueError(
                f"the number of clusters ({n}) must exceed the number of "
                f"coefficients ({p})"
            )

        # Bread: the model's own var-cov times the weighted design.
        ocl = np.argsort(cluster, kind="stable")
        W = np.linalg.inv(view.M[np.ix_(ocl, ocl)])
        bread = view.vb @ view.X[ocl, :].T @ W

        # Meat: block-diagonal within-cluster residual outer products.
        ei = view.yi - (view.X @ view.beta).ravel()
        groups = _cluster_groups(cluster[ocl])
        meat = _block_diag([np.outer(ei[ocl][g], ei[ocl][g]) for g in groups])

        vb_robust = bread @ meat @ bread.T
        vbest = "CR0"
        if adjust is True:
            vb_robust = (n / dfs) * vb_robust
            vbest = "CR1"
        elif isinstance(adjust, str) and adjust in ("Stata", "Stata1"):
            vb_robust = (n / (n - 1) * (k - 1) / (k - p)) * vb_robust
            vbest = "CR1.S1"
        elif isinstance(adjust, str) and adjust == "Stata2":
            vb_robust = (n / (n - 1)) * vb_robust
            vbest = "CR1.S2"
        ddf = np.full(p, float(dfs))
        ddf_out = float(dfs)

    # A degenerate variance means that coefficient is not identified by the
    # data at hand; blanking the row keeps it out of every derived quantity
    # instead of letting a 1e-17 standard error masquerade as precision.
    degenerate = np.diag(vb_robust) < _DEGENERATE
    if np.any(degenerate):
        vb_robust = vb_robust.copy()
        vb_robust[degenerate, :] = np.nan
        vb_robust[:, degenerate] = np.nan

    beta = view.beta.ravel()
    with np.errstate(invalid="ignore", divide="ignore"):
        se = np.sqrt(np.diag(vb_robust))
        zval = beta / se
        alpha = 1.0 - lvl / 100.0
        crit = stats.t.ppf(1 - alpha / 2.0, df=ddf)
        pval = 2.0 * stats.t.sf(np.abs(zval), df=ddf)
        ci_lb = beta - crit * se
        ci_ub = beta + crit * se

    if clubSandwich:
        qm, qm_df, qm_p = None, None, None
    else:
        qm, qm_df, qm_p = _omnibus(view, vb_robust, ddf)

    return {
        "vbest": vbest,
        "beta": [float(b) for b in beta],
        "se": [float(s) for s in se],
        "zval": [float(z) for z in np.atleast_1d(zval)],
        "pval": [float(pv) for pv in np.atleast_1d(pval)],
        "ci_lb": [float(c) for c in np.atleast_1d(ci_lb)],
        "ci_ub": [float(c) for c in np.atleast_1d(ci_ub)],
        "dfs": ddf_out,
        "n": n,
        "vb": vb_robust,
        "QM": qm,
        "QMdf": qm_df,
        "QMp": qm_p,
    }
