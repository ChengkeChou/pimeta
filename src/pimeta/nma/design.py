"""The network parameterisation layer: how contrasts map to parameters.

Every network meta-analysis — standard, dose-response, time-course, IPD — is
the same object underneath: a set of *treatment parameters*, a set of
*observed comparisons*, and a linear map between them.  Getting that map wrong
is the failure mode that no amount of likelihood engineering rescues, and it
is the one thing that can be checked without a likelihood at all.

This module is that map, on its own, with no estimator attached:

``TreatmentNetwork``
    treatments, studies, arms, connectivity, components, a spanning forest and
    the fundamental circuits (independent loops) of the comparison graph.

``contrast_matrix``
    :math:`B` (``m x n``): row ``i`` compares ``treat1`` against ``treat2``,
    ``+1`` for the first and ``-1`` for the second.  One sign convention,
    stated once, used by every model in the package.

``basic_parameters`` / ``design_matrix``
    :math:`d_{ab} = d_b - d_a` with one treatment fixed at zero.  The design
    matrix :math:`X_b` (:math:`m \\times (n-1)`) satisfies
    :math:`B d = X_b d_b` for every :math:`d_b`, which is what ``netmeta``'s
    ``createB``/``createX`` pair does.

``circuits`` / ``consistency_residuals``
    A network is consistent on a circuit when the signed sum of its edge
    effects is zero.  The circuit space has dimension
    ``m - n + components``, which is also the number of degrees of freedom an
    inconsistency model spends — so this one number is both the algebraic
    statement and the model-selection one.

``ume_parameters`` / ``ume_design_matrix``
    The unrelated-mean-effects model: every comparison type gets its own
    parameter, so no consistency is imposed.  Its parameter count minus
    ``n - 1`` is the inconsistency degrees of freedom.

``split_design``
    The direct/indirect decomposition of one comparison: which edges feed each
    side, and with what weights, so a node-splitting test can be checked
    against the arithmetic rather than against a second implementation of it.

``multiarm_structure``
    Within-study covariance of a multi-arm study's comparisons.  Two contrasts
    against the same arm share that arm's sampling variance, so the study's
    comparisons are correlated and cannot be treated as separate observations —
    the single mistake that most often makes a network analysis too precise.

What this module deliberately does not do: estimate anything.  The estimators
in :mod:`pimeta.netmeta`, and the dose-response, time-course and IPD network
models, all consume this layer, so a consistency bug is found once, here,
by an algebraic test rather than three times in three likelihoods.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

__all__ = [
    "MultiArmStructure",
    "TreatmentNetwork",
    "consistency_residuals",
    "multiarm_structure",
]

_ZERO = 1e-10


# --------------------------------------------------------------------------- #
# the graph
# --------------------------------------------------------------------------- #
@dataclass
class TreatmentNetwork:
    """The comparison graph of a network meta-analysis.

    Attributes:
        treatments: treatment labels, in the order the parameters are indexed.
        comparisons: one row per study comparison, ``(study, treat1, treat2)``
            with the *first* treatment named first, so ``+1`` always means
            "favours `treat1` relative to `treat2`" on the analysis scale.
        studies: the study labels, in order of first appearance.
    """

    treatments: list[str]
    comparisons: list[tuple[str, str, str]]
    studies: list[str] = field(default_factory=list)

    def __post_init__(self):
        self.treatments = [str(t) for t in self.treatments]
        if len(set(self.treatments)) != len(self.treatments):
            raise ValueError("treatment labels must be unique")
        index = {t: i for i, t in enumerate(self.treatments)}
        rows = []
        for rec in self.comparisons:
            if len(rec) != 3:
                raise ValueError(
                    "each comparison must be (study, treat1, treat2)")
            study, t1, t2 = (str(rec[0]), str(rec[1]), str(rec[2]))
            if t1 not in index or t2 not in index:
                raise ValueError(
                    f"unknown treatment in comparison {rec!r}; known: "
                    f"{self.treatments}")
            if t1 == t2:
                raise ValueError(
                    f"comparison {rec!r} compares a treatment with itself")
            rows.append((study, t1, t2))
        self.comparisons = rows
        if not self.studies:
            seen: list[str] = []
            for study, _t1, _t2 in rows:
                if study not in seen:
                    seen.append(study)
            self.studies = seen
        self._index = index

    # -- basic shape ------------------------------------------------------ #
    @property
    def n_treatments(self) -> int:
        return len(self.treatments)

    @property
    def n_comparisons(self) -> int:
        return len(self.comparisons)

    def position(self, treatment: str) -> int:
        """Index of a treatment in :attr:`treatments`."""
        key = str(treatment)
        if key not in self._index:
            raise ValueError(f"unknown treatment {treatment!r}")
        return self._index[key]

    def study_arms(self) -> dict[str, list[str]]:
        """``{study: [arm, ...]}`` in first-appearance order, de-duplicated."""
        out: dict[str, list[str]] = {}
        for study, t1, t2 in self.comparisons:
            arms = out.setdefault(study, [])
            for t in (t1, t2):
                if t not in arms:
                    arms.append(t)
        return out

    def n_arms(self, study: str) -> int:
        """Arms in a study; a k-arm study contributes ``k(k-1)/2`` comparisons."""
        return len(self.study_arms()[str(study)])

    def multiarm_studies(self) -> list[str]:
        return [s for s, arms in self.study_arms().items() if len(arms) > 2]

    # -- connectivity ----------------------------------------------------- #
    def components(self) -> list[list[str]]:
        """Connected components of the comparison graph, largest first."""
        parent = {t: t for t in self.treatments}

        def find(a):
            while parent[a] != a:
                parent[a] = parent[parent[a]]
                a = parent[a]
            return a

        for _study, t1, t2 in self.comparisons:
            ra, rb = find(t1), find(t2)
            if ra != rb:
                parent[ra] = rb
        groups: dict[str, list[str]] = {}
        for t in self.treatments:
            groups.setdefault(find(t), []).append(t)
        out = sorted(groups.values(), key=len, reverse=True)
        return out

    def is_connected(self) -> bool:
        return len(self.components()) == 1

    def spanning_forest(self) -> list[tuple[int, str, str]]:
        """Edge indices of a spanning forest, in the order they join sets."""
        parent = {t: t for t in self.treatments}

        def find(a):
            while parent[a] != a:
                parent[a] = parent[parent[a]]
                a = parent[a]
            return a

        tree: list[tuple[int, str, str]] = []
        for i, (_study, t1, t2) in enumerate(self.comparisons):
            ra, rb = find(t1), find(t2)
            if ra != rb:
                parent[ra] = rb
                tree.append((i, t1, t2))
        return tree

    def circuits(self) -> list[list[tuple[int, str, str]]]:
        """Fundamental circuits: one per non-tree edge, as signed edge lists.

        Each circuit is the cycle closed by a non-tree edge, with the sign of
        every edge relative to its own ``treat1 -> treat2`` direction.  The
        number of circuits is ``m - n + components``, the dimension of the
        cycle space.
        """
        tree = self.spanning_forest()
        adjacency: dict[str, list[tuple[str, int, int]]] = {}
        for idx, t1, t2 in tree:
            adjacency.setdefault(t1, []).append((t2, idx, 1))
            adjacency.setdefault(t2, []).append((t1, idx, -1))

        def path(start: str, goal: str) -> list[tuple[int, int]]:
            """Edges ``(index, sign)`` of the tree path, sign relative to the edge."""
            prev: dict[str, tuple[str, int, int]] = {start: (start, -1, 0)}
            stack = [start]
            while stack:
                node = stack.pop()
                if node == goal:
                    break
                for nxt, idx, sign in adjacency.get(node, []):
                    if nxt not in prev:
                        prev[nxt] = (node, idx, sign)
                        stack.append(nxt)
            if goal not in prev:
                return []
            out: list[tuple[int, int]] = []
            node = goal
            while node != start:
                parent, idx, sign = prev[node]
                out.append((idx, sign))
                node = parent
            return out

        in_tree = {idx for idx, _t1, _t2 in tree}
        out: list[list[tuple[int, str, str]]] = []
        for i, (_study, t1, t2) in enumerate(self.comparisons):
            if i in in_tree:
                continue
            cycle: list[tuple[int, str, str]] = [(i, t1, t2)]
            for idx, sign in path(t2, t1):
                s1, s2 = self.comparisons[idx][1], self.comparisons[idx][2]
                if sign > 0:
                    cycle.append((idx, s1, s2))
                else:
                    cycle.append((idx, s2, s1))
            out.append(cycle)
        return out

    @property
    def df_inconsistency(self) -> int:
        """Degrees of freedom an inconsistency model would spend.

        ``m - n + components``: the cycle-space dimension, which equals the
        NMA residual degrees of freedom ``m - (n - 1)`` for a connected
        network, and which an UME model spends only when no comparison type
        repeats.
        """
        return (self.n_comparisons - self.n_treatments
                + len(self.components()))

    # -- matrices --------------------------------------------------------- #
    def contrast_matrix(self) -> np.ndarray:
        """``B`` (``m x n``): ``+1`` for treat1, ``-1`` for treat2."""
        B = np.zeros((self.n_comparisons, self.n_treatments))
        for i, (_study, t1, t2) in enumerate(self.comparisons):
            B[i, self.position(t1)] = 1.0
            B[i, self.position(t2)] = -1.0
        return B

    def basic_parameters(self, reference=None) -> list[str]:
        """Treatments that get a parameter, with ``reference`` fixed at zero."""
        ref = self.treatments[0] if reference is None else str(reference)
        if ref not in self._index:
            raise ValueError(f"unknown reference treatment {reference!r}")
        return [t for t in self.treatments if t != ref]

    def design_matrix(self, reference=None) -> np.ndarray:
        """``X_b`` (``m x (n-1)``) with ``B d = X_b d_b`` for every ``d``.

        Column ``j`` is the contrast of basic parameter ``j`` against the
        reference: ``+1`` when the comparison's treat1 is that parameter,
        ``-1`` when its treat2 is, ``0`` otherwise.  For a closed network the
        columns are linearly independent (rank ``n - 1`` per component), which
        is what ``rank`` reports and what makes the parameters estimable.
        """
        params = self.basic_parameters(reference)
        pos = {t: j for j, t in enumerate(params)}
        X = np.zeros((self.n_comparisons, len(params)))
        for i, (_study, t1, t2) in enumerate(self.comparisons):
            if t1 in pos:
                X[i, pos[t1]] += 1.0
            if t2 in pos:
                X[i, pos[t2]] -= 1.0
        return X

    def design_rank(self, reference=None) -> int:
        """Rank of the basic-parameter design: ``n - components``."""
        X = self.design_matrix(reference)
        if X.size == 0:
            return 0
        return int(np.linalg.matrix_rank(X))

    def null_space(self) -> np.ndarray:
        """Basis of the constraint space of ``B`` (columns ``x`` with ``B x = 0``).

        For a connected network this is the single all-ones vector — adding a
        constant to every treatment effect changes no contrast — so its
        dimension tells you how many components there are, and any other basis
        vector would mean the parameterisation is not identifiable.
        """
        from scipy.linalg import null_space

        B = self.contrast_matrix()
        if B.size == 0:
            return np.zeros((self.n_treatments, 0))
        return null_space(B, rcond=1e-10)

    # -- inconsistency models --------------------------------------------- #
    def comparison_types(self) -> list[tuple[str, str]]:
        """Distinct unordered comparisons, in first-appearance order."""
        seen: list[tuple[str, str]] = []
        for _study, t1, t2 in self.comparisons:
            key = (t1, t2)
            if key not in seen:
                seen.append(key)
        return seen

    def ume_parameters(self) -> list[tuple[str, str]]:
        """The unrelated-mean-effects parameterisation: one per comparison type.

        The consistency model has ``n - 1`` parameters.  This one has one per
        observed comparison *type*, so it spends ``t - (n - 1)`` more — which
        equals :attr:`df_inconsistency` exactly when no pair of treatments is
        compared by more than one study.  Repeated types make ``t < m``, and
        then UME absorbs less than the cycle space allows: two studies of the
        same pair are forced to agree even under UME.  That is a property of
        the model, not of the network, and it is why the two numbers are
        reported separately rather than one being derived from the other.
        """
        return self.comparison_types()

    def ume_design_matrix(self) -> np.ndarray:
        """``X_u`` (``m x t``): a ``1`` in the column of the comparison's type.

        Fitting a model with this design imposes no consistency between types:
        each is its own parameter, so every circuit through differently-typed
        edges can be non-zero.
        """
        types = self.ume_parameters()
        pos = {pair: j for j, pair in enumerate(types)}
        X = np.zeros((self.n_comparisons, len(types)))
        for i, (_study, t1, t2) in enumerate(self.comparisons):
            X[i, pos[(t1, t2)]] += 1.0
        return X

    def split_design(self, treat1: str, treat2: str) -> dict:
        """How much of one comparison comes from direct evidence.

        The question node-splitting answers, expressed as arithmetic rather
        than as a second estimator: which comparisons feed the direct estimate
        of ``treat1`` vs ``treat2``, and which feed the indirect one.

        Args:
            treat1: first treatment of the comparison of interest.
            treat2: second treatment.

        Returns:
            ``direct`` — indices of comparisons that are exactly this
            comparison (either orientation); ``indirect`` — indices that lie
            on a path between the two treatments through at least one other
            treatment; ``df`` — ``1`` when both sides are non-empty, else 0.

        Raises:
            ValueError: when the two treatments are the same, or the
        comparison is not in the network.
        """
        a, b = str(treat1), str(treat2)
        if a == b:
            raise ValueError("a comparison needs two different treatments")
        direct = []
        for i, (_study, t1, t2) in enumerate(self.comparisons):
            if {t1, t2} == {a, b}:
                direct.append(i)
        # the indirect side: a spanning tree path that avoids the direct edges
        adjacency: dict[str, list[tuple[str, int]]] = {}
        for i, (_study, t1, t2) in enumerate(self.comparisons):
            if i in direct:
                continue
            adjacency.setdefault(t1, []).append((t2, i))
            adjacency.setdefault(t2, []).append((t1, i))
        prev: dict[str, tuple[str, int]] = {a: (a, -1)}
        stack = [a]
        while stack:
            node = stack.pop()
            if node == b:
                break
            for nxt, idx in adjacency.get(node, []):
                if nxt not in prev:
                    prev[nxt] = (node, idx)
                    stack.append(nxt)
        indirect: list[int] = []
        if b in prev:
            node = b
            while node != a:
                parent, idx = prev[node]
                indirect.append(idx)
                node = parent
            indirect.reverse()
        if not direct:
            raise ValueError(
                f"{a!r} vs {b!r} is not a comparison in this network, so it "
                "has no direct evidence to split off")
        return {
            "comparison": (a, b),
            "direct": direct,
            "indirect": indirect,
            "df": 1 if indirect else 0,
        }


def consistency_residuals(network: TreatmentNetwork, effects) -> np.ndarray:
    """Signed sums around every circuit: zero for a consistent network.

    Args:
        network: the comparison graph.
        effects: one effect per comparison, on the analysis scale, in the order
            of ``network.comparisons`` and with the sign convention of
            :meth:`TreatmentNetwork.contrast_matrix`.

    Returns:
        One residual per circuit.  All zero by construction when the effects
        come from treatment parameters; non-zero exactly when they do not.
    """
    eff = np.asarray(effects, dtype=float).ravel()
    if eff.size != network.n_comparisons:
        raise ValueError(
            f"expected {network.n_comparisons} effects, got {eff.size}")
    out = []
    for cycle in network.circuits():
        total = 0.0
        for idx, t1, _t2 in cycle:
            sign = 1.0 if network.comparisons[idx][1] == t1 else -1.0
            total += sign * eff[idx]
        out.append(total)
    return np.array(out, dtype=float)


# --------------------------------------------------------------------------- #
# multi-arm structure
# --------------------------------------------------------------------------- #
@dataclass
class MultiArmStructure:
    """Within-study covariance of a multi-arm study's comparisons.

    The ``m = k(k-1)/2`` comparisons of a ``k``-arm study live in a
    ``k - 1``-dimensional space: ``(A-B) - (A-C) = C - B``, so the matrix is
    **singular by construction**, with ``m - (k - 1)`` zero eigenvalues.  That
    is not a defect to be repaired, it is the reason a multi-arm study cannot
    be entered as ``m`` independent observations.

    Attributes:
        arms: the study's arms, in order.
        pairs: the comparisons, ``(i, j)`` indices into ``arms``.
        covariance: ``(m x m)`` covariance of the comparisons' effect estimates.
        correlation: the matching correlation matrix.
    """

    arms: list[str]
    pairs: list[tuple[int, int]]
    covariance: np.ndarray = field(repr=False)
    correlation: np.ndarray = field(repr=False)

    @property
    def rank(self) -> int:
        """Rank of the covariance: ``k - 1`` for ``k`` arms."""
        return int(np.linalg.matrix_rank(np.asarray(self.covariance)))

    def to_dict(self) -> dict:
        return {
            "arms": list(self.arms),
            "pairs": [[int(i), int(j)] for i, j in self.pairs],
            "covariance": np.asarray(self.covariance).tolist(),
            "correlation": np.asarray(self.correlation).tolist(),
            "rank": self.rank,
        }


def multiarm_structure(arms, variances, *, per: str = "arm",
                       rho=None) -> MultiArmStructure:
    """Covariance of every pairwise comparison inside one multi-arm study.

    Two contrasts that share an arm share that arm's sampling variance: with
    contrast coefficients ``a`` and ``b`` over the arms,

    .. math::

        \\mathrm{Cov}(y_a, y_b) = \\sum_k a_k b_k \\mathrm{Var}(A_k),

    so ``A vs B`` and ``A vs C`` covary by ``+Var(A)`` while ``A vs B`` and
    ``B vs C`` covary by ``-Var(B)``.  The sign is the part that is easy to
    get wrong and easy to test: a simulation of arm means reproduces the whole
    matrix, and ``tests/test_nma_design.py`` does exactly that.

    Args:
        arms: arm labels of the study, in order.
        variances: one variance per arm (``per="arm"``, the default) or one
            per comparison in the order ``(0,1), (0,2), ..., (1,2), ...``.
        per: which of the two ``variances`` means.  It is explicit because a
            three-arm study has three arms *and* three comparisons, so the
            length alone cannot say.
        rho: used only with ``per="comparison"``, where the shared-arm
            variance cannot be recovered; the correlation is then ``rho`` with
            the sign of the shared arm.

    Returns:
        :class:`MultiArmStructure`.

    Raises:
        ValueError: wrong length, non-positive variance, an unknown ``per``, or
            a covariance that is not positive definite.
    """
    arms = [str(a) for a in arms]
    k = len(arms)
    if k < 2:
        raise ValueError("a multi-arm structure needs at least two arms")
    pairs = [(i, j) for i in range(k - 1) for j in range(i + 1, k)]
    m = len(pairs)
    which = str(per).lower()
    if which not in ("arm", "comparison"):
        raise ValueError(
            f"unknown per={per!r}; use 'arm' or 'comparison'")
    v = np.asarray(variances, dtype=float).ravel()
    want = k if which == "arm" else m
    if v.size != want:
        raise ValueError(
            f"per={which!r} needs {want} variances, got {v.size}")
    if np.any(v <= 0.0):
        raise ValueError("every variance must be positive")

    # coefficients of each comparison over the arms: +1 first, -1 second
    coef = np.zeros((m, k))
    for a, (i, j) in enumerate(pairs):
        coef[a, i] = 1.0
        coef[a, j] = -1.0

    if which == "arm":
        cov = coef @ np.diag(v) @ coef.T
    else:
        if rho is None:
            raise ValueError(
                "comparison-level variances cannot recover the shared-arm "
                "covariance; give one variance per arm, or 'rho' explicitly")
        cov = np.diag(v)
        sd = np.sqrt(v)
        for a in range(m):
            for b in range(m):
                if a != b and np.any(coef[a] * coef[b] < 0.0):
                    # one shared arm with opposite signs: negative covariance
                    cov[a, b] = -float(rho) * sd[a] * sd[b]
                elif a != b and np.any(coef[a] * coef[b] > 0.0):
                    cov[a, b] = float(rho) * sd[a] * sd[b]
    cov = 0.5 * (cov + cov.T)
    evals = np.linalg.eigvalsh(cov)
    scale = max(float(np.max(np.abs(evals))) if evals.size else 1.0, 1.0)
    if evals.size and float(np.min(evals)) < -1e-8 * scale:
        raise ValueError(
            "the multi-arm covariance has a negative eigenvalue; the "
            "variances or rho are inconsistent with a shared-arm structure")
    sd = np.sqrt(np.diag(cov))
    corr = cov / np.outer(sd, sd)
    return MultiArmStructure(arms=arms, pairs=pairs, covariance=cov,
                             correlation=corr)
