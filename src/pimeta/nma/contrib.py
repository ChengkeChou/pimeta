"""Contribution of each study to each network estimate (netmeta::netcontrib).

``netcontrib`` decomposes every network estimate into the contribution of the
direct comparisons it flows through -- Papakonstantinou et al. (2018)'s
shortest-path method by default, or a random walk over the network.

``NetcontribResult`` and ``netcontrib`` are re-exported from
:mod:`pimeta.netmeta`; the annotation-only use of ``NetmetaResult`` is a
``TYPE_CHECKING`` import so this module stays cycle-free.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

from .core import _ZERO, _create_b_full, _invmat

if TYPE_CHECKING:                                    # pragma: no cover
    from ..netmeta import NetmetaResult



@dataclass
class NetcontribResult:
    """Contribution matrix of a network meta-analysis (netmeta::netcontrib).

    ``common[r, c]`` (and ``random[r, c]``) is the proportion of the pooled
    estimate for comparison ``comparisons[r]`` (a treatment pair, row) that is
    contributed by the direct comparison ``directs[c]`` (column). Rows sum to
    1 (within numerical precision). Uses the default "shortestpath"
    (Papakonstantinou et al. 2018) flow-decomposition method.
    """

    trts: list[str]
    comparisons: list[str]
    directs: list[str]
    common: np.ndarray
    random: np.ndarray
    method: str = "shortestpath"

    def to_dict(self) -> dict:
        return {
            "trts": self.trts,
            "comparisons": self.comparisons,
            "directs": self.directs,
            "common": self.common.tolist(),
            "random": self.random.tolist(),
            "method": self.method,
        }


def _shortest_path_edges(adj, source, target):
    """Unweighted shortest path (BFS) from source to target.

    ``adj`` maps a node to a list of ``(neighbor, edge_index)`` tuples. Returns
    the ordered list of edge indices on one shortest path (empty if no path).
    """
    from collections import deque

    if source == target:
        return []
    parent = {source: (None, None)}
    queue = deque([source])
    while queue:
        u = queue.popleft()
        if u == target:
            break
        for v, eidx in adj.get(u, ()):
            if v not in parent:
                parent[v] = (u, eidx)
                queue.append(v)
    if target not in parent:
        return []
    path = []
    cur = target
    while parent[cur][0] is not None:
        prev, eidx = parent[cur]
        path.append(eidx)
        cur = prev
    path.reverse()
    return path


def _hatmatrix_aggr_long(treat1_pos, treat2_pos, b_full_m, se_adj, n):
    """Aggregated hat matrix in "long" form (netmeta::hatmatrix.aggr).

    Returns ``(H_long, pairs)`` where ``H_long`` is (c x e): rows = all
    n(n-1)/2 treatment pairs (upper-triangle order), columns = the e unique
    direct comparisons (in ascending node-pair order). ``pairs`` gives the
    direct-comparison node pairs for each column.
    """
    pairs = sorted(set(zip(treat1_pos, treat2_pos)))
    e = len(pairs)
    b_aggr = np.zeros((e, n), dtype=float)
    for r, (i, j) in enumerate(pairs):
        b_aggr[r, i] = 1.0
        b_aggr[r, j] = -1.0

    m = b_full_m.shape[0]
    w = np.zeros((e, e), dtype=float)
    for i in range(e):
        idx_a = idx_b = None
        for j in range(n):
            if b_aggr[i, j] == 1.0:
                idx_a = j
            elif b_aggr[i, j] == -1.0:
                idx_b = j
        wab = 0.0
        for k in range(m):
            if b_full_m[k, idx_a] == 1.0 and b_full_m[k, idx_b] == -1.0:
                wab += 1.0 / (se_adj[k] ** 2)
        w[i, i] = wab

    l = b_aggr.T @ w @ b_aggr
    lplus = _invmat(l)
    lplus[np.abs(lplus) < _ZERO] = 0.0

    b_full = _create_b_full(n)
    h_long = b_full @ lplus @ b_aggr.T @ w
    return h_long, pairs


def _contribution_matrix(treat1_pos, treat2_pos, b_full_m, se_adj, n):
    """Shortest-path flow decomposition (netmeta::contribution.matrix.tpapak)."""
    h_long, pairs = _hatmatrix_aggr_long(treat1_pos, treat2_pos, b_full_m, se_adj, n)
    e = h_long.shape[1]
    c = h_long.shape[0]
    weights = np.zeros((c, e), dtype=float)

    all_pairs = [(i, j) for i in range(n) for j in range(i + 1, n)]
    for row, (source, target) in enumerate(all_pairs):
        h_row = h_long[row, :]
        flows = [abs(float(h_row[k])) for k in range(e)]
        # Directed edge list in column order; H > 0 keeps (di, dj), otherwise
        # reversed. Zero-flow edges stay in the graph (igraph includes them).
        edges = []
        for k in range(e):
            di, dj = pairs[k]
            u, v = (di, dj) if h_row[k] > 0.0 else (dj, di)
            edges.append((u, v))
        # igraph assigns vertex ids in first-appearance order and its BFS
        # iterates neighbours by ascending vertex id; replicate both here.
        vid = {}
        for u, v in edges:
            if u not in vid:
                vid[u] = len(vid)
            if v not in vid:
                vid[v] = len(vid)
        active = [True] * e
        wrow = np.zeros(e, dtype=float)
        while True:
            adj = {}
            for k in range(e):
                if not active[k]:
                    continue
                u, v = edges[k]
                adj.setdefault(u, []).append((v, k))
            for u in adj:
                adj[u].sort(key=lambda item: vid[item[0]])
            path = _shortest_path_edges(adj, source, target)
            if not path:
                break
            pl = float(len(path))
            flow = min(flows[k] for k in path)
            for k in path:
                flows[k] -= flow
                wrow[k] += flow / pl
                if flows[k] == 0.0:
                    active[k] = False
        wrow[np.abs(wrow) < _ZERO] = 0.0
        weights[row, :] = wrow
    return weights, pairs


def _simple_paths_randomwalk(adj: np.ndarray, source: int, target: int) -> list[list[int]]:
    """Every simple directed path ``source -> target`` in ``adj`` (igraph equivalent)."""
    n = adj.shape[0]
    out: list[list[int]] = []
    stack: list[tuple[int, list[int]]] = [(source, [source])]
    while stack:
        node, path = stack.pop()
        if node == target:
            out.append(path)
            continue
        for nxt in range(n):
            if adj[node, nxt] != 0.0 and nxt not in path:
                stack.append((nxt, path + [nxt]))
    return out


def _contribution_matrix_randomwalk(
    result, model: str
) -> tuple[np.ndarray, list[tuple[int, int]]]:
    """Davies' random-walk contribution matrix (netmeta::contribution.matrix.davies).

    For each treatment pair the aggregated hat matrix supplies a directed
    "flow" matrix ``Q``; normalising its rows gives a transition matrix that is
    made absorbing at the target. Every simple path from source to target then
    contributes ``(1/L) * prod(P)`` where ``L`` is the path length.
    """
    n = int(result.n)
    treat1_pos = [int(v) for v in result.treat1_pos]
    treat2_pos = [int(v) for v in result.treat2_pos]
    b_full_m = np.asarray(result.B_matrix, dtype=float)
    se_adj = np.asarray(
        result.seTE_adj_random if model == "random" else result.seTE_adj_common,
        dtype=float,
    )
    # hatmatrix.aggr(x, model, type = "full") is c x c with c = n(n-1)/2: the
    # column of a pair with direct evidence receives the matching column of the
    # "long" hat matrix, every other column stays zero (so a comparison that is
    # only ever indirect contributes nothing and is dropped below).
    h_long, hpairs = _hatmatrix_aggr_long(
        treat1_pos, treat2_pos, b_full_m, se_adj, n
    )
    comps = n * (n - 1) // 2
    pairs_all = [(i, j) for i in range(n) for j in range(i + 1, n)]
    col_of = {pair: k for k, pair in enumerate(pairs_all)}
    h_full = np.zeros((comps, comps), dtype=float)
    for l, pair in enumerate(hpairs):
        h_full[:, col_of[pair]] = h_long[:, l]
    ep = 2.220446049250313e-16 * 1000.0  # R's is_zero(x, n = 1000)

    weights = np.zeros((comps, comps), dtype=float)
    r = 0
    for t1 in range(n - 1):
        for t2 in range(t1 + 1, n):
            q = np.zeros((n, n), dtype=float)
            idx = 0
            for i in range(n - 1):
                for j in range(i + 1, n):
                    q[i, j] = h_full[r, idx]
                    idx += 1
            for i in range(n):
                for j in range(n):
                    if i != j:
                        q[j, i] = -q[i, j]
            for i in range(n):
                for j in range(n):
                    if i != j and q[i, j] <= 0.0:
                        q[i, j] = 0.0

            # P = ginv(diag(rowSums(Q))) %*% Q, absorbing at t2
            p = np.zeros((n, n), dtype=float)
            row_sums = q.sum(axis=1)
            for i in range(n):
                if row_sums[i] != 0.0:
                    p[i, :] = q[i, :] / row_sums[i]
            p[t2, :] = 0.0
            p[t2, t2] = 1.0
            p[np.abs(p) < ep] = 0.0

            rmat = np.zeros((n, n), dtype=float)
            for path in _simple_paths_randomwalk(p, t1, t2):
                length = len(path) - 1
                if length <= 0:
                    continue
                prob = 1.0 / length
                for i in range(length):
                    prob *= p[path[i], path[i + 1]]
                for i in range(length):
                    rmat[path[i], path[i + 1]] += prob
                for i in range(n - 1):
                    for j in range(i + 1, n):
                        v = max(rmat[i, j], rmat[j, i])
                        rmat[i, j] = v
                        rmat[j, i] = v

            # R fills uppertri() in column-major order but labels the columns
            # with hatmatrix.aggr's ascending pair order, so write the entries
            # straight into ascending order here.
            for lex_pos, (i, j) in enumerate(
                (a, b) for a in range(n) for b in range(a + 1, n)
            ):
                weights[r, lex_pos] = rmat[i, j]
            r += 1

    weights[np.abs(weights) < ep] = 0.0
    keep = weights.sum(axis=0) > 0.0
    # R labels the columns with hatmatrix.aggr's row names (ascending pair
    # order) while the values were filled in upper-triangle column-major order.
    kept_pairs = [pair for pair, k in zip(pairs_all, keep) if k]
    return weights[:, keep], kept_pairs


def netcontrib(x: NetmetaResult, method: str = "shortestpath") -> NetcontribResult:
    """Contribution matrix of a network meta-analysis.

    Transcribes ``netmeta::netcontrib`` (default "shortestpath" method). Each
    row of the returned ``common`` / ``random`` matrices corresponds to a
    treatment pair; each column to a direct comparison (edge). Rows sum to 1.
    """
    if method not in ("shortestpath", "randomwalk"):
        raise ValueError("method must be 'shortestpath' or 'randomwalk'")

    n = x.n
    trts = list(x.trts)
    treat1_pos = [int(v) for v in x.treat1_pos]
    treat2_pos = [int(v) for v in x.treat2_pos]
    b_full_m = np.asarray(x.B_matrix, dtype=float)
    se_common = np.asarray(x.seTE_adj_common, dtype=float)
    se_random = np.asarray(x.seTE_adj_random, dtype=float)

    if method == "randomwalk":
        common, pairs = _contribution_matrix_randomwalk(x, "common")
        random, _ = _contribution_matrix_randomwalk(x, "random")
    else:
        common, pairs = _contribution_matrix(
            treat1_pos, treat2_pos, b_full_m, se_common, n)
        random, _ = _contribution_matrix(
            treat1_pos, treat2_pos, b_full_m, se_random, n)

    sep = ":"
    comparisons = [f"{trts[i]}{sep}{trts[j]}" for i in range(n) for j in range(i + 1, n)]
    directs = [f"{trts[i]}{sep}{trts[j]}" for (i, j) in pairs]

    return NetcontribResult(
        trts=trts,
        comparisons=comparisons,
        directs=directs,
        common=common,
        random=random,
        method=method,
    )
