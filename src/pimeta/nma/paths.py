"""Path-by-path inconsistency (``netmeta::netpath``).

Builds the evidence-flow network from the Davies hat matrix, enumerates every
simple path between two treatments and tests the direct evidence along those
paths.  Split out of :mod:`pimeta.netmeta`; the public names are re-exported
there.
"""

from __future__ import annotations

import math
import sys
from typing import TYPE_CHECKING

import numpy as np
from scipy import stats

from .core import _create_b, _create_b_full, _invmat
from .impact import _comparison_list

if TYPE_CHECKING:  # pragma: no cover - typing only
    pass


def _hatmatrix_aggr(result, model="random", type="full"):
    """``netmeta::hatmatrix.aggr`` — the Davies hat matrix.

    ``type`` is ``"short"`` (e x e, observed comparisons), ``"long"``
    (all x observed) or ``"full"`` (all x all).
    """
    trts = list(result.trts)
    n = len(trts)
    m = len(result.TE_raw)
    t1p = list(result.treat1_pos)
    t2p = list(result.treat2_pos)
    se_adj = np.asarray(
        result.seTE_adj_random if model == "random" else result.seTE_adj_common,
        dtype=float,
    )
    B = _create_b(t1p, t2p, n)                    # raw m x n

    observed = set()
    for a, b in zip(result.treat1_raw, result.treat2_raw):
        i, j = trts.index(a), trts.index(b)
        observed.add((min(i, j), max(i, j)))
    obs_pairs = [(i, j) for i in range(n - 1) for j in range(i + 1, n)
                 if (i, j) in observed]
    obs_comps = [f"{trts[i]}:{trts[j]}" for i, j in obs_pairs]

    e = len(obs_pairs)
    W = np.zeros((e, e))
    for r, (i, j) in enumerate(obs_pairs):
        w = 0.0
        for k in range(m):
            if B[k, i] == 1.0 and B[k, j] == -1.0:
                w += 1.0 / se_adj[k] ** 2
        W[r, r] = w

    B_aggr = _create_b([p[0] for p in obs_pairs], [p[1] for p in obs_pairs], n)
    L = B_aggr.T @ W @ B_aggr
    L_plus = _invmat(L)
    L_plus[np.abs(L_plus) < sys.float_info.epsilon] = 0.0

    H_short = B_aggr @ L_plus @ B_aggr.T @ W
    B_full = _create_b_full(n)
    H_long = B_full @ L_plus @ B_aggr.T @ W

    c = n * (n - 1) // 2
    H_full = np.zeros((c, c))
    l_idx = 0
    for k in range(c):
        if l_idx < e:
            i, j = obs_pairs[l_idx]
            bi = int(np.argmax(B_full[k] == 1.0))
            bj = int(np.argmax(B_full[k] == -1.0))
            if bi == i and bj == j:
                H_full[:, k] = H_long[:, l_idx]
                l_idx += 1

    if type == "short":
        return {"matrix": H_short, "rows": obs_comps, "cols": obs_comps}
    if type == "long":
        return {"matrix": H_long, "rows": _comparison_list(trts),
                "cols": obs_comps}
    return {"matrix": H_full, "rows": _comparison_list(trts),
            "cols": _comparison_list(trts)}


def _all_simple_paths(edges, source, target):
    """All simple paths (igraph ``all_simple_paths``) as lists of node names."""
    adj = {}
    for a, b in edges:
        adj.setdefault(a, []).append(b)
        adj.setdefault(b, [])

    paths = []

    def _dfs(node, path, seen):
        if node == target:
            paths.append(list(path))
            return
        for nxt in adj.get(node, []):
            if nxt not in seen:
                seen.add(nxt)
                path.append(nxt)
                _dfs(nxt, path, seen)
                path.pop()
                seen.discard(nxt)

    if source in adj:
        _dfs(source, [source], {source})
    return paths


def _get_edges_from_path(nodes, sep=":"):
    """``get_edges_from_path`` — unordered consecutive edge labels."""
    if len(nodes) < 2:
        return []
    return [sep.join(sorted([nodes[k], nodes[k + 1]]))
            for k in range(len(nodes) - 1)]


def netpath(result, node1, node2, random=True):
    """``netmeta::netpath`` — path-by-path inconsistency (Koenig et al. 2013).

    Builds the evidence-flow network from the row of the Davies hat matrix that
    corresponds to ``node1`` vs ``node2``, enumerates every simple path between
    the two nodes, and tests whether the direct evidence along those paths is
    consistent with the network estimate.

    Args:
        result: a :class:`NetmetaResult`.
        node1, node2: the treatment contrast of interest.
        random: use the random-effects model.

    Returns:
        A dict with ``results`` (per-path ``Q``, ``df``, ``pval``),
        ``A_matrix``, ``Sigma`` and ``theta_p`` — the fields of the R
        ``netpath`` object.
    """
    from ..netmeta import NetmetaResult  # local import: cycle

    if not isinstance(result, NetmetaResult):
        raise TypeError("argument 'result' must be a NetmetaResult")
    trts = list(result.trts)
    for nd in (node1, node2):
        if nd not in trts:
            raise ValueError(f"'{nd}' is not a treatment of the network")
    if node1 == node2:
        raise ValueError("'node1' and 'node2' must differ")

    sep = ":"
    hm = _hatmatrix_aggr(result, "random" if random else "common", "full")
    comp_intr = f"{node1}{sep}{node2}"
    if comp_intr not in hm["rows"]:
        comp_intr = sep.join(sorted([node1, node2]))

    # --- directed evidence-flow graph from the hat-matrix row -------------- #
    row = hm["matrix"][hm["rows"].index(comp_intr), :]
    edges = []
    for c_idx, col_label in enumerate(hm["cols"]):
        parts = col_label.split(sep)
        if len(parts) != 2:
            continue
        i, j = parts
        val = row[c_idx]
        if val > 0:
            edges.append((i, j))
        elif val < 0:
            edges.append((j, i))

    all_paths = _all_simple_paths(edges, node1, node2)
    if not all_paths:
        return {
            "comparison": comp_intr, "results": [], "A_matrix": None,
            "Sigma": None, "theta_p": None, "note": "No path found",
            "random": random, "node1": node1, "node2": node2,
            "trts": trts, "sep.trts": sep,
        }

    edge_sets = [_get_edges_from_path(p, sep) for p in all_paths]
    npth = len(edge_sets)
    A = np.zeros((npth, npth))
    for i in range(npth):
        for j in range(npth):
            if i == j:
                A[i, j] = len(edge_sets[i])
            else:
                A[i, j] = len(set(edge_sets[i]) & set(edge_sets[j]))

    # --- drop linearly dependent paths (reduce_to_full_rank) --------------- #
    if abs(np.linalg.det(A)) < 1e-6:
        from scipy.linalg import qr as _qr
        _, _, piv = _qr(A, pivoting=True)
        rank = int(np.linalg.matrix_rank(A, tol=1e-8))
        kept = sorted(int(p) for p in piv[:rank])
    else:
        kept = list(range(npth))

    kept_paths = [all_paths[k] for k in kept]
    kept_edges = [_get_edges_from_path(p, sep) for p in kept_paths]

    comp_edges = _comparison_list(trts)
    se_dir = np.asarray(result.seTE_direct_common, dtype=float)
    te_dir = np.asarray(result.TE_direct_common, dtype=float)
    v_diag = []
    for ce in comp_edges:
        i, j = (trts.index(x) for x in ce.split(sep))
        v_diag.append(se_dir[i, j] ** 2)

    idx_c = [k for k, v in enumerate(v_diag) if not math.isnan(v)]
    C = np.zeros((len(kept), len(comp_edges)))
    for i, es in enumerate(kept_edges):
        for j, ce in enumerate(comp_edges):
            if ce in es:
                C[i, j] = 1.0
    C = C[:, idx_c]
    V = np.diag([v_diag[k] for k in idx_c])

    Sigma = C @ V @ C.T
    if Sigma.shape[0] > 0 and abs(np.linalg.det(Sigma)) < 1e-6:
        Sigma_inv = np.linalg.pinv(Sigma)
        note = "Matrix Sigma singular, pseudo-inverse used"
    else:
        Sigma_inv = np.linalg.inv(Sigma)
        note = ""

    theta_p = []
    for p in kept_paths:
        s = 0.0
        for k in range(len(p) - 1):
            a, b = p[k], p[k + 1]
            v = te_dir[trts.index(a), trts.index(b)]
            if not math.isnan(v):
                s += v
        theta_p.append(s)
    theta_p = np.asarray(theta_p, dtype=float)

    i1, i2 = trts.index(node1), trts.index(node2)
    theta_diff = theta_p - float(np.asarray(result.TE_common)[i1, i2])
    Q = float(theta_diff @ Sigma_inv @ theta_diff)
    df = int(Sigma.shape[0] - 1)
    pval = float(stats.chi2.sf(Q, df)) if df > 0 else float("nan")

    return {
        "comparison": comp_intr,
        "results": [{"path_index": kept[k] + 1, "path": kept_paths[k],
                     "Q": Q, "df": df, "pval": pval, "note": note}
                    for k in range(len(kept))],
        "A_matrix": A,
        "Sigma": Sigma,
        "theta_p": theta_p,
        "note": note,
        "random": random,
        "node1": node1,
        "node2": node2,
        "trts": trts,
        "sep.trts": sep,
    }
