"""``networkx`` output — the evidence network as a real graph object.

Once the evidence network is a ``networkx`` graph you get, for free, everything
the graph community has built: centrality measures, community detection, several
dozen layouts, and export to GraphML/GEXF/GML for Gephi, Cytoscape or D3.

The adapter is deliberately thin: nodes are treatments, edges are direct
comparisons (aggregated across studies), and both carry the numbers a downstream
algorithm or a visual encoder needs.
"""

from __future__ import annotations

from collections import defaultdict

import numpy as np

__all__ = ["to_networkx"]


def _require_networkx():
    try:
        import networkx as nx
    except ImportError as exc:                        # pragma: no cover - optional
        raise ImportError(
            "graph output requires networkx (pip install pimeta[networkx])") from exc
    return nx


def to_networkx(result, weight: str = "inverse_variance", directed: bool = False):
    """Build a ``networkx`` graph of the evidence network.

    Args:
        result: a :class:`pimeta.netmeta.NetmetaResult`.
        weight: how to weight an edge.
            ``"inverse_variance"`` (default) uses :math:`1/\\overline{se}^2`
            across the studies comparing that pair; ``"studies"`` uses the number
            of studies; ``"none"`` stores no weight.
        directed: keep the contrast direction as stored (``True``) or collapse
            each pair into one undirected edge (``False``, the default — the
            usual convention for an evidence network).

    Returns:
        A ``networkx.Graph`` (or ``DiGraph``) whose nodes carry ``pscore`` and
        whose edges carry ``studies``, ``TE``, ``seTE`` and ``weight``.

    Raises:
        ImportError: when networkx is not installed.
        TypeError: when ``result`` is not a network meta-analysis result.
        ValueError: for an unknown ``weight`` mode.
    """
    nx = _require_networkx()
    if weight not in ("inverse_variance", "studies", "none"):
        raise ValueError(
            f"unknown weight mode {weight!r}; "
            "choose 'inverse_variance', 'studies' or 'none'")

    trts = getattr(result, "trts", None)
    if trts is None:
        raise TypeError(f"{type(result).__name__} has no 'trts'; "
                        "to_networkx() expects a network meta-analysis result")
    names = [str(t) for t in trts]

    graph = nx.DiGraph() if directed else nx.Graph()
    graph.add_nodes_from(names)

    pscore = getattr(result, "pscore_random", None)
    if pscore is None or np.asarray(pscore).size != len(names):
        pscore = getattr(result, "pscore_common", None)
    if pscore is not None and np.asarray(pscore).size == len(names):
        for name, value in zip(names, np.asarray(pscore, dtype=float)):
            graph.nodes[name]["pscore"] = float(value)

    treat1 = list(getattr(result, "treat1_raw", []) or [])
    treat2 = list(getattr(result, "treat2_raw", []) or [])
    te = list(getattr(result, "TE_raw", []) or [])
    sete = list(getattr(result, "seTE_raw", []) or [])
    studlab = list(getattr(result, "studlab_raw", []) or [])
    if not treat1 or not treat2:
        raise TypeError(
            f"{type(result).__name__} carries no 'treat1_raw'/'treat2_raw' "
            "comparison data")

    # Aggregate studies by (comparison) key.
    buckets: dict[tuple[str, str], dict] = defaultdict(
        lambda: {"n": 0, "te": [], "se": [], "studies": []})
    for i, (a, b) in enumerate(zip(treat1, treat2)):
        a, b = str(a), str(b)
        # sorted() would be typed as an arbitrary-length tuple; a pair is exact.
        key: tuple[str, str] = (a, b) if directed else (min(a, b), max(a, b))
        bucket = buckets[key]
        bucket["n"] += 1
        if i < len(te):
            bucket["te"].append(float(te[i]))
        if i < len(sete) and sete[i] is not None and np.isfinite(float(sete[i])):
            bucket["se"].append(float(sete[i]))
        if i < len(studlab):
            bucket["studies"].append(str(studlab[i]))

    for (a, b), bucket in buckets.items():
        attrs: dict = {"studies": int(bucket["n"])}
        if bucket["te"]:
            # Inverse-variance weighted mean effect for this direct comparison.
            if bucket["se"] and len(bucket["se"]) == len(bucket["te"]):
                w = 1.0 / np.square(np.asarray(bucket["se"], dtype=float))
                attrs["TE"] = float(np.sum(w * np.asarray(bucket["te"])) / np.sum(w))
                attrs["seTE"] = float(np.sqrt(1.0 / np.sum(w)))
            else:
                attrs["TE"] = float(np.mean(bucket["te"]))
        if bucket["studies"]:
            attrs["study_labels"] = bucket["studies"]
        if weight == "studies":
            attrs["weight"] = float(bucket["n"])
        elif weight == "inverse_variance":
            if "seTE" in attrs and attrs["seTE"] > 0:
                attrs["weight"] = float(1.0 / attrs["seTE"] ** 2)
            else:
                attrs["weight"] = float(bucket["n"])
        graph.add_edge(a, b, **attrs)

    return graph
