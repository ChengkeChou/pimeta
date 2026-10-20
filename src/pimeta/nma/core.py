"""Graph-theory and numeric primitives shared by the NMA code.

These are the leaves of the network meta-analysis implementation: incidence
matrices, the Laplacian pseudoinverse, the I^2 transform, the Cochran Q and
the DerSimonian-Laird tau^2.  They are pure NumPy/SciPy and depend on nothing
else in ``pimeta``, so every other module may import them at module level
without creating a cycle.

``pimeta.netmeta`` re-exports all of them, because the rest of that module and
several tests refer to the private names directly.
"""

from __future__ import annotations

import math

import numpy as np
from scipy import stats

Z_975 = stats.norm.ppf(0.975)

# netmeta's is_zero() uses .Machine$double.eps^0.5 ~ 1.49e-8.
_ZERO = 1e-8


def _create_b(treat1_pos, treat2_pos, n):
    """Edge-vertex incidence matrix B (m x n).

    ``B[i, treat1_pos[i]] = 1`` and ``B[i, treat2_pos[i]] = -1``, all other
    entries 0. Mirrors ``netmeta::createB``.
    """
    m = len(treat1_pos)
    b = np.zeros((m, n), dtype=float)
    for i in range(m):
        b[i, treat1_pos[i]] = 1.0
        b[i, treat2_pos[i]] = -1.0
    return b


def _create_b_full(n):
    """Full edge-vertex incidence matrix for all n(n-1)/2 unordered pairs."""
    pairs = [(i, j) for i in range(n) for j in range(i + 1, n)]
    t1 = [p[0] for p in pairs]
    t2 = [p[1] for p in pairs]
    return _create_b(t1, t2, n)


def _invmat(x):
    """Moore-Penrose pseudoinverse of a (weighted) Laplacian.

    ``solve(X - J/n) + J/n`` with J the all-ones matrix — exact transcription
    of ``netmeta::invmat`` (valid for connected networks, rank n-1).
    """
    n = x.shape[0]
    j = np.ones((n, n), dtype=float)
    return np.linalg.solve(x - j / n, np.eye(n)) + j / n


def _multiarm(r):
    """Adjust the comparison variances of a single multi-arm study.

    Exact transcription of ``netmeta::multiarm`` (Rücker 2012, appendix).
    ``r`` holds the k(k-1)/2 comparison variances (se^2 + tau^2) ordered
    (1,2), (1,3), ..., (1,k), (2,3), ..., (k-1,k) by sorted treatment. The
    returned vector gives the adjusted variances in the same order, so the
    study's internal comparisons are down-weighted consistently.
    """
    r = np.asarray(r, dtype=float)
    m = len(r)
    k = int(round((1.0 + math.sqrt(8.0 * m + 1.0)) / 2.0))

    b = _create_b_full(k)                     # complete incidence matrix
    rr = np.diag(r)

    # R[i,j] = resistance (variance) between arms i,j of the complete graph.
    tmp = b.T @ rr @ b
    rmat = np.diag(np.diag(tmp)) - tmp

    # Lt = -0.5 * (B'B) R (B'B) / k^2 ; L = invmat(Lt).
    bb = b.T @ b
    lt = -0.5 * bb @ rmat @ bb / float(k * k)
    l = _invmat(lt)

    # W = diag(diag(L)) - L ; zero out tiny negative weights.
    w = np.diag(np.diag(l)) - l
    lower = np.tril_indices(k, -1)
    denom = float(np.sum(np.abs(w[lower])))
    if denom > 0:
        small_neg = (w < 0.0) & (np.abs(w) / denom < 0.001)
        w = np.where(small_neg, 0.0, w)

    # V = 1 / W element-wise (R's ``V <- 1 / W`` is element-wise).
    with np.errstate(divide="ignore", invalid="ignore"):
        vmat = 1.0 / w

    v = np.zeros(m, dtype=float)
    e = 0
    for i in range(k - 1):
        for j in range(i + 1, k):
            v[e] = vmat[i, j]
            e += 1
    return v


def _adjusted_variances(se_arr, tau2, treat1_pos, treat2_pos, studlab):
    """Per-comparison adjusted variances (se^2), replicating netmeta::prepare.

    Two-arm studies contribute ``se^2 + tau2`` unchanged. Multi-arm studies
    have their comparison variances reweighted by ``_multiarm`` (Rücker 2012)
    so the internally-correlated comparisons are down-weighted consistently.
    """
    se2 = se_arr ** 2 + tau2
    adj = se2.copy()

    by_study = {}
    for i, s in enumerate(studlab):
        by_study.setdefault(s, []).append(i)

    for inds in by_study.values():
        if len(inds) <= 1:
            continue
        # Order this study's comparisons as (1,2),(1,3),...,(k-1,k).
        order = sorted(inds, key=lambda i: (treat1_pos[i], treat2_pos[i]))
        v = _multiarm(se2[order])
        for i, vv in zip(order, v):
            adj[i] = vv

    return adj


def _connected(adj):
    """Return True if the 0/1 adjacency graph is connected (BFS)."""
    n = adj.shape[0]
    seen = [False] * n
    seen[0] = True
    stack = [0]
    while stack:
        u = stack.pop()
        for v in range(n):
            if not seen[v] and adj[u, v]:
                seen[v] = True
                stack.append(v)
    return all(seen)


def _isquared(q, df, level):
    """I^2 with confidence limits (Higgins & Thompson 2002), via calcH.

    ``H = sqrt(Q/df)``, ``I^2 = (H^2 - 1) / H^2``; CI back-computed from a
    normal CI for ``log(max(H, 1))``. Exact transcription of
    ``meta::isquared`` / ``meta::calcH``.
    """
    if df <= 0 or math.isnan(q):
        return float("nan"), float("nan"), float("nan")

    k = df + 1
    if k > 1:
        h = math.sqrt(q / (k - 1))
    else:
        h = float("nan")

    if q > k:
        if k >= 2:
            selog_h = (0.5 * (math.log(q) - math.log(k - 1))
                       / (math.sqrt(2 * q) - math.sqrt(2 * k - 3)))
        else:
            selog_h = float("nan")
    else:
        if k > 2:
            selog_h = math.sqrt(1.0 / (2.0 * (k - 2))
                                * (1.0 - 1.0 / (3.0 * (k - 2) ** 2)))
        else:
            selog_h = float("nan")

    z = stats.norm.ppf(1 - (1 - level) / 2)
    log_h = math.log(max(h, 1.0))

    def _f(x):
        return (x ** 2 - 1.0) / x ** 2

    i2 = _f(math.exp(log_h))
    if math.isnan(selog_h):
        lower = upper = float("nan")
    else:
        lower = _f(max(math.exp(log_h - z * selog_h), 1.0))
        upper = _f(max(math.exp(log_h + z * selog_h), 1.0))
    return i2, lower, upper


def _cochran_q(y, se):
    """Cochran's Q for fixed (inverse-variance) weights 1/se^2."""
    y = np.asarray(y, dtype=float)
    se = np.asarray(se, dtype=float)
    w = 1.0 / se ** 2
    mu = float(np.sum(w * y) / np.sum(w))
    return float(np.sum(w * (y - mu) ** 2))


def _dl_tau2(y, se):
    """DerSimonian-Laird tau^2 (pairwise metagen)."""
    y = np.asarray(y, dtype=float)
    se = np.asarray(se, dtype=float)
    k = len(y)
    if k <= 1:
        return float("nan")
    w = 1.0 / se ** 2
    sw = float(np.sum(w))
    q = _cochran_q(y, se)
    c = sw - float(np.sum(w ** 2)) / sw
    return max(0.0, (q - (k - 1)) / c)


def _iv_pool(y, se, tau2):
    """Inverse-variance pooled estimate with a *fixed* tau^2 added to se^2."""
    y = np.asarray(y, dtype=float)
    se = np.asarray(se, dtype=float)
    w = 1.0 / (se ** 2 + tau2)
    sw = float(np.sum(w))
    mu = float(np.sum(w * y) / sw)
    se_mu = math.sqrt(1.0 / sw)
    return mu, se_mu


def _pair_index(i, j, n):
    """Flattened 0-based row index of pair (i, j) with i < j in ``X.full``.

    Rows of the design matrix follow ``trts.poss``:
    t0:t1, t0:t2, ..., t0:t(n-1), t1:t2, ..., t(n-2):t(n-1).
    """
    idx = 0
    for a in range(i):
        idx += n - 1 - a
    return idx + (j - i - 1)


def _fx(n):
    """Full design matrix ``fX(n)`` from ``netmeta::nma.krahn``.

    ``possK = n*(n-1)/2`` rows (one per possible contrast) by ``n-1`` columns
    (treatment effects relative to the first treatment). Exact 0-based
    transcription of the R ``fX`` helper.
    """
    poss_k = n * (n - 1) // 2
    x = np.zeros((poss_k, n - 1), dtype=float)
    if n <= 2:
        # a single contrast A:B is just -theta_1
        x[0, 0] = -1.0
        return x
    x[0:(n - 1), 0:(n - 1)] = np.diag(np.full(n - 1, -1.0))
    x[poss_k - 1, (n - 3):(n - 1)] = [1.0, -1.0]
    if poss_k - (n - 1) > 1:
        l = n                       # 1-based row start
        j = n - 2
        u = n + j - 1               # 1-based row end
        for k in range(1, n - 2):   # k = 1 .. n-3
            block = np.hstack([
                np.ones((n - k - 1, 1)),
                np.diag(np.full(n - k - 1, -1.0)),
            ])
            x[(l - 1):u, (k - 1):(n - 1)] = block
            j -= 1
            l = u + 1
            u = l + j - 1
    return x


def _block_diag(blocks):
    """Block-diagonal matrix from a list of scalars/vectors/matrices."""
    mats = []
    for b in blocks:
        arr = np.asarray(b, dtype=float)
        if arr.ndim == 0:
            arr = arr.reshape(1, 1)
        elif arr.ndim == 1:
            arr = arr.reshape(1, -1)
        mats.append(arr)
    sizes = [m.shape[0] for m in mats]
    total = sum(sizes)
    out = np.zeros((total, total), dtype=float)
    r = 0
    for m in mats:
        c = m.shape[0]
        out[r:r + c, r:r + c] = m
        r += c
    return out
