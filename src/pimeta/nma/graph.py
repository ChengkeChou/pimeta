"""Graph structure of a network (netmeta::netconnection / netdistance).

``netconnection`` builds the (weighted) adjacency matrix of a network and
labels the edges with the number of direct comparisons; ``netdistance`` turns
that adjacency into a shortest-path distance matrix, sorted the way R's
``hclust(dist(D2))`` sorts it.

Both public names are re-exported from :mod:`pimeta.netmeta`.  The one name
this module still borrows from there -- ``NetmetaResult``, used for an
``isinstance`` check -- is imported inside the function that needs it.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np


def _netdistance_matrix(a: np.ndarray) -> np.ndarray:
    """Graph distance matrix (shortest path, in number of edges).

    Exact transcription of ``netmeta::netdistance.default``: ``D = sign(A)``,
    then all off-diagonal zero entries become ``Inf``, followed by a layered
    breadth-first relaxation that turns every entry into the number of edges
    on the shortest path (``Inf`` when the two nodes are in different
    subnetworks).
    """
    a = np.asarray(a, dtype=float)
    n = a.shape[0]
    d = np.sign(a)
    for i in range(n - 1):
        for j in range(i + 1, n):
            if d[i, j] == 0:
                d[i, j] = np.inf
                d[j, i] = np.inf
    for dist in range(1, n):
        for i in range(n):
            for j in range(n):
                if d[i, j] == dist:
                    for k in range(n):
                        akj = d[k, i] + dist
                        if akj < d[k, j]:
                            d[k, j] = akj
    return d


def _hclust_order(d2: np.ndarray) -> list[int]:
    """Leaf order of ``hclust(dist(D2))`` (complete linkage, Euclidean).

    Replicates R's ``hclust(dist(D2))$order`` used by ``netmeta::netdistance``
    to sort the nodes of a network. Ties are broken by SciPy's complete-linkage
    implementation, which agrees with R for the common (non-degenerate) cases;
    the order only affects the *display* ordering of disconnected components
    and never the numeric results.
    """
    from scipy.cluster.hierarchy import leaves_list, linkage
    from scipy.spatial.distance import pdist

    d2 = np.asarray(d2, dtype=float)
    cond = pdist(d2, metric="euclidean")
    z = linkage(cond, method="complete")
    return [int(i) for i in leaves_list(z)]


@dataclass
class NetdistanceResult:
    """Graph distance matrix (netmeta::netdistance).

    ``D[i, j]`` is the number of edges on the shortest path between nodes
    ``i`` and ``j`` (``inf`` if they lie in different subnetworks), diagonal
    zero. ``order`` is the ``hclust`` leaf order R attaches for display
    purposes; ``labels`` are the row/column names.
    """

    D: np.ndarray
    labels: list[str] = field(default_factory=list)
    order: list[int] = field(default_factory=list)

    def to_dict(self) -> dict:
        d = np.asarray(self.D, dtype=object)
        d[np.isinf(np.asarray(self.D, dtype=float))] = None
        return {
            "D": d.tolist(),
            "labels": self.labels,
            "order": self.order,
        }


@dataclass
class NetconnectionResult:
    """Network connectivity structure (netmeta::netconnection).

    Groups the treatment network into its connected subnetworks and reports
    the graph distance matrix ``D_matrix``, the weighted adjacency matrix
    ``A_matrix`` (``A[i,j]`` = number of studies directly comparing ``i`` and
    ``j``) and the Laplacian ``L_matrix``, all indexed by ``seq``.
    """

    treat1: list = field(default_factory=list)
    treat2: list = field(default_factory=list)
    studlab: list = field(default_factory=list)
    design: list = field(default_factory=list)          # per-comparison design
    subnet: list = field(default_factory=list)          # per-comparison subnet
    k: int = 0                                          # number of studies
    m: int = 0                                          # number of comparisons
    n: int = 0                                          # number of treatments
    n_subnets: int = 0
    d: int = 0                                          # number of designs
    seq: list[str] = field(default_factory=list)
    D_matrix: np.ndarray = field(default_factory=lambda: np.empty((0, 0)))
    A_matrix: np.ndarray = field(default_factory=lambda: np.empty((0, 0)))
    L_matrix: np.ndarray = field(default_factory=lambda: np.empty((0, 0)))
    designs: list[str] = field(default_factory=list)
    comparisons: list[str] = field(default_factory=list)
    subnet_comparisons: list[int] = field(default_factory=list)

    def to_dict(self) -> dict:
        def _m(x: np.ndarray):
            if x.size == 0:
                return []
            arr = np.asarray(x, dtype=float)
            out = arr.astype(object)
            out[np.isinf(arr)] = None
            return out.tolist()

        return {
            "treat1": self.treat1,
            "treat2": self.treat2,
            "studlab": self.studlab,
            "design": self.design,
            "subnet": self.subnet,
            "k": self.k, "m": self.m, "n": self.n,
            "n_subnets": self.n_subnets, "d": self.d,
            "seq": self.seq,
            "D_matrix": _m(self.D_matrix),
            "A_matrix": self.A_matrix.tolist() if self.A_matrix.size else [],
            "L_matrix": _m(self.L_matrix),
            "designs": self.designs,
            "comparisons": self.comparisons,
            "subnet_comparisons": self.subnet_comparisons,
        }


def netconnection(
    treat1,
    treat2,
    studlab=None,
    sep_trts: str = ":",
    title: str = "",
    details_disconnected: bool = False,
) -> NetconnectionResult:
    """Determine the (sub)network structure of a network meta-analysis.

    Replicates ``netmeta::netconnection``: builds the weighted adjacency matrix
    from ``treat1``/``treat2``, computes the graph-distance matrix, splits the
    network into its connected subnetworks and reports the number of studies
    (``k``), pairwise comparisons (``m``), treatments (``n``), subnetworks
    (``n_subnets``) and designs (``d``), together with the ordered distance /
    adjacency / Laplacian matrices.

    Args:
        treat1, treat2: the two treatments of each pairwise comparison
            (``m`` vectors, e.g. ``d["treat1"]`` / ``d["treat2"]``).
        studlab: study labels; ``None`` treats every comparison as its own
            study.
        sep_trts: separator used in the ``comparisons`` labels.
    """
    treat1 = [str(t) for t in treat1]
    treat2 = [str(t) for t in treat2]
    m = len(treat1)
    if len(treat2) != m:
        raise ValueError("Arguments 'treat1' and 'treat2' must have the same length.")
    if studlab is None:
        studlab = [str(i + 1) for i in range(m)]
    else:
        studlab = [str(s) for s in studlab]
        if len(studlab) != m:
            raise ValueError("Arguments 'treat1' and 'studlab' must have the same length.")
    for a, b in zip(treat1, treat2):
        if a == b:
            raise ValueError("Treatments must be different (arguments 'treat1' and 'treat2').")

    labels = sorted(set(treat1) | set(treat2))
    n = len(labels)
    pos = {t: i for i, t in enumerate(labels)}

    # Weighted adjacency: A[i,j] = number of studies comparing i and j.
    a_mat = np.zeros((n, n), dtype=float)
    for a, b in zip(treat1, treat2):
        a_mat[pos[a], pos[b]] += 1.0
        a_mat[pos[b], pos[a]] += 1.0

    d_mat = _netdistance_matrix(a_mat)
    l_mat = np.diag(a_mat.sum(axis=1)) - a_mat

    # hclust leaf order (matches R's hclust(dist(D2))$order).
    maxdist = n
    d2 = np.where(np.isinf(d_mat), maxdist, d_mat)
    order = _hclust_order(d2)

    # Sort D / A / L by the hclust order.
    d_sorted = d_mat[np.ix_(order, order)]
    a_sorted = a_mat[np.ix_(order, order)]
    l_mat[np.ix_(order, order)]
    order_labels = [labels[i] for i in order]

    # Walk through the sorted distance matrix to carve out the subnetworks.
    id_treats = []
    id_subnets = []
    subnet_i = 0
    dat_subnet = []  # (subnet, comparison, treat1, treat2)
    a_loop = a_sorted.copy()
    d_loop = d_sorted.copy()
    o_loop = list(order_labels)
    while d_loop.shape[0] > 0:
        subnet_i += 1
        nrows = d_loop.shape[0]
        inf_pos = [i for i in range(nrows) if np.isinf(d_loop[i, 0])]
        next_subnet = inf_pos[0] if inf_pos else nrows
        sel = list(range(next_subnet))
        for s in sel:
            id_treats.append(o_loop[s])
            id_subnets.append(subnet_i)
        sub_labels = [o_loop[s] for s in sel]
        sort_idx = sorted(range(len(sub_labels)), key=lambda x: sub_labels[x])
        a_i = a_loop[np.ix_(sel, sel)][np.ix_(sort_idx, sort_idx)]
        sub_labels_sorted = [sub_labels[x] for x in sort_idx]
        nci = a_i.shape[0]
        for row_i in range(nci - 1):
            for col_i in range(1, nci):
                if col_i > row_i and a_i[row_i, col_i] > 0:
                    t1 = sub_labels_sorted[row_i]
                    t2 = sub_labels_sorted[col_i]
                    dat_subnet.append((subnet_i, f"{t1}{sep_trts}{t2}", t1, t2))
        keep = [i for i in range(nrows) if i not in set(sel)]
        a_loop = a_loop[np.ix_(keep, keep)]
        d_loop = d_loop[np.ix_(keep, keep)]
        o_loop = [o_loop[i] for i in keep]

    n_subnets = subnet_i

    # seq: for each subnet, the treatments in order of first appearance over
    # the (alphabetically ordered) comparisons, treat1 before treat2.
    seq = []
    for comp in range(1, n_subnets + 1):
        t1_list = [d[2] for d in dat_subnet if d[0] == comp]
        t2_list = [d[3] for d in dat_subnet if d[0] == comp]
        seen = set()
        for x in t1_list + t2_list:
            if x not in seen:
                seen.add(x)
                seq.append(x)

    # Reorder matrices by seq.
    seq_pos = {t: i for i, t in enumerate(seq)}
    idx = [seq_pos[t] for t in labels]
    d_final = d_mat[np.ix_(idx, idx)]
    a_final = a_mat[np.ix_(idx, idx)]
    l_final = l_mat[np.ix_(idx, idx)]

    # Designs: sorted unique treatments of each study joined by ':' (R's
    # designs() default separator, independent of sep.trts).
    study_treats: dict[Any, set[Any]] = {}
    for a, b, s in zip(treat1, treat2, studlab):
        study_treats.setdefault(s, set()).update([a, b])
    design_per_study = {s: ":".join(sorted(study_treats[s])) for s in study_treats}
    design = [design_per_study[s] for s in studlab]
    designs = sorted(set(design))

    treat_to_subnet = dict(zip(id_treats, id_subnets))
    subnet = [treat_to_subnet[t] for t in treat1]

    comparisons = [d[1] for d in dat_subnet]
    subnet_comparisons = [d[0] for d in dat_subnet]

    return NetconnectionResult(
        treat1=treat1,
        treat2=treat2,
        studlab=studlab,
        design=design,
        subnet=subnet,
        k=len(set(studlab)),
        m=m,
        n=n,
        n_subnets=n_subnets,
        d=len(designs),
        seq=seq,
        D_matrix=d_final,
        A_matrix=a_final,
        L_matrix=l_final,
        designs=designs,
        comparisons=comparisons,
        subnet_comparisons=subnet_comparisons,
    )


def netdistance(x, sort: bool = True) -> NetdistanceResult:
    """Graph distance matrix (netmeta::netdistance).

    Accepts either a weighted/0-1 adjacency ``ndarray``, a
    :class:`NetmetaResult`, or a :class:`NetconnectionResult`. For a
    ``NetmetaResult`` the adjacency is sorted by the ``netconnection`` order
    when ``sort`` is true, matching ``netmeta::netdistance.netmeta``.
    """
    from ..netmeta import NetmetaResult  # local import: cycle

    if isinstance(x, NetmetaResult):
        labels = list(x.trts)
        a_mat = np.zeros((x.n, x.n), dtype=float)
        for i in range(x.n - 1):
            for j in range(i + 1, x.n):
                if not np.isnan(x.A_matrix[i, j]):
                    a_mat[i, j] = a_mat[j, i] = float(x.A_matrix[i, j])
        if sort:
            seq = netconnection(x.treat1_raw, x.treat2_raw, x.studlab_raw).seq
            idx = [labels.index(t) for t in seq]
            a_mat = a_mat[np.ix_(idx, idx)]
            labels = seq
        return NetdistanceResult(D=_netdistance_matrix(a_mat), labels=labels)
    if isinstance(x, NetconnectionResult):
        return NetdistanceResult(
            D=_netdistance_matrix(x.A_matrix.astype(float)),
            labels=list(x.seq),
        )
    a = np.asarray(x, dtype=float)
    d = _netdistance_matrix(a)
    d2 = np.where(np.isinf(d), a.shape[0], d)
    order = _hclust_order(d2)
    return NetdistanceResult(D=d, order=order)
