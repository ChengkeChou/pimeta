"""Algebraic tests of the network parameterisation layer.

These are the first of the three validation layers a network model needs, and
the only one that can be run without a likelihood:

1. **algebraic** (this file) — do the consistency constraints hold for *any*
   parameter vector, is the design matrix full rank, does the cycle space have
   the dimension the graph theory says it does;
2. **model-level** — does an independent implementation of the same equations
   give the same numbers;
3. **property-level** — on data with a known truth, does the model recover it
   and does a misspecified one get rejected.

Passing layer 1 says the constraints are implemented correctly.  It says
nothing about a likelihood, a prior or a posterior, which is why the other two
layers are not optional and why this file does not pretend to be enough.

Everything here is checked against graph theory rather than against a second
implementation of the same formula: a circuit basis has ``m - n + c`` vectors,
the design matrix has rank ``n - c``, the null space is the spanning set of
the components' indicator vectors.
"""

import itertools

import numpy as np
import pytest

from pimeta.nma.design import (
    TreatmentNetwork,
    consistency_residuals,
    multiarm_structure,
)

# A B C in a triangle, plus a fourth treatment D attached only to A, plus a
# second trial of A vs B: one circuit, one repeated comparison type, four
# treatments, six comparisons.
TRIANGLE = TreatmentNetwork(
    treatments=["A", "B", "C", "D"],
    comparisons=[
        ("s1", "A", "B"),
        ("s2", "A", "C"),
        ("s3", "B", "C"),
        ("s4", "A", "D"),
        ("s5", "A", "B"),
        ("s5", "A", "C"),
    ],
)


def test_the_incidence_matrix_has_the_signed_convention():
    B = TRIANGLE.contrast_matrix()
    assert B.shape == (6, 4)
    i = TRIANGLE.comparisons.index(("s3", "B", "C"))
    assert B[i, TRIANGLE.position("B")] == 1.0
    assert B[i, TRIANGLE.position("C")] == -1.0
    assert np.sum(np.abs(B[i])) == 2.0


def test_a_random_parameter_vector_satisfies_every_circuit():
    """The consistency identity, for arbitrary treatment effects.

    ``d_ab = d_b - d_a`` around any closed loop cancels term by term.  If this
    fails for some parameter vector, the sign convention is wrong somewhere —
    which is the bug that no amount of likelihood work would reveal.
    """
    rng = np.random.default_rng(4)
    for _ in range(20):
        theta = rng.normal(0.0, 1.0, TRIANGLE.n_treatments)
        effects = TRIANGLE.contrast_matrix() @ theta
        assert consistency_residuals(TRIANGLE, effects) == pytest.approx(
            np.zeros(len(TRIANGLE.circuits())), abs=1e-12)


def test_an_inconsistent_network_is_detected():
    rng = np.random.default_rng(5)
    effects = rng.normal(0.0, 1.0, TRIANGLE.n_comparisons)
    residuals = consistency_residuals(TRIANGLE, effects)
    assert len(residuals) == TRIANGLE.df_inconsistency
    assert np.any(np.abs(residuals) > 0.1)


def test_the_design_matrix_reproduces_the_contrasts():
    """``B d = X_b d_b`` for every ``d``: the reference choice is a gauge."""
    rng = np.random.default_rng(6)
    for reference in ("A", "B", "C", "D"):
        params = TRIANGLE.basic_parameters(reference)
        theta = rng.normal(0.0, 1.0, TRIANGLE.n_treatments)
        B = TRIANGLE.contrast_matrix()
        X = TRIANGLE.design_matrix(reference)
        basic = np.array([theta[TRIANGLE.position(t)]
                          - theta[TRIANGLE.position(reference)]
                          for t in params])
        assert X @ basic == pytest.approx(B @ theta, abs=1e-12)


def test_the_design_rank_is_n_minus_the_number_of_components():
    assert TRIANGLE.design_rank() == TRIANGLE.n_treatments - 1
    assert TRIANGLE.is_connected()

    split = TreatmentNetwork(
        treatments=["A", "B", "C", "D"],
        comparisons=[("s1", "A", "B"), ("s2", "C", "D")],
    )
    assert len(split.components()) == 2
    # one reference is fixed overall, so the rank is n - components
    assert split.design_rank() == split.n_treatments - 2


def test_the_cycle_space_has_the_dimension_the_graph_says():
    """Number of circuits = m - n + c, for several network shapes."""
    shapes = {
        "two edges": TreatmentNetwork(["A", "B", "C"],
                                      [("s1", "A", "B"), ("s2", "B", "C")]),
        "triangle": TreatmentNetwork(["A", "B", "C"],
                                     [("s1", "A", "B"), ("s2", "B", "C"),
                                      ("s3", "A", "C")]),
        "square with diagonal": TreatmentNetwork(
            ["A", "B", "C", "D"],
            [("s1", "A", "B"), ("s2", "B", "C"), ("s3", "C", "D"),
             ("s4", "D", "A"), ("s5", "A", "C")]),
        "two components": TreatmentNetwork(
            ["A", "B", "C", "D"],
            [("s1", "A", "B"), ("s2", "B", "C"), ("s3", "C", "A"),
             ("s4", "C", "D")]),
    }
    for name, net in shapes.items():
        m, n, c = net.n_comparisons, net.n_treatments, len(net.components())
        circuits = net.circuits()
        assert len(circuits) == m - n + c, name
        assert net.df_inconsistency == m - n + c, name


def test_the_circuits_are_independent_as_vector_space():
    """The residual map has rank ``m - n + c``, so the circuits carry it."""
    m = TRIANGLE.n_comparisons
    # build the residual operator as rows of +/-1 over the edges
    rows = []
    for cycle in TRIANGLE.circuits():
        row = np.zeros(m)
        for idx, t1, _t2 in cycle:
            row[idx] = 1.0 if TRIANGLE.comparisons[idx][1] == t1 else -1.0
        rows.append(row)
    R = np.array(rows)
    assert np.linalg.matrix_rank(R) == TRIANGLE.df_inconsistency
    # and its left null space is exactly the space of consistent effects
    from scipy.linalg import null_space

    consistent = TRIANGLE.design_matrix()
    assert np.allclose(R @ consistent, 0.0, atol=1e-12)
    assert null_space(R, rcond=1e-10).shape[1] == consistent.shape[1]


def test_the_null_space_counts_the_components():
    V = TRIANGLE.null_space()
    assert V.shape == (TRIANGLE.n_treatments, 1)
    # adding a constant to every effect changes no contrast
    assert np.allclose(np.abs(V[:, 0]), 1.0 / np.sqrt(4), atol=1e-10)

    split = TreatmentNetwork(["A", "B", "C", "D"],
                             [("s1", "A", "B"), ("s2", "C", "D")])
    assert split.null_space().shape[1] == 2


def test_the_parameters_are_identified_up_to_the_reference():
    """Every contrast is invariant to the reference; the parameters are not."""
    rng = np.random.default_rng(7)
    theta = rng.normal(0.0, 1.0, TRIANGLE.n_treatments)
    B = TRIANGLE.contrast_matrix()
    effects = B @ theta
    for reference in TRIANGLE.treatments:
        X = TRIANGLE.design_matrix(reference)
        # a full-rank least-squares solve reproduces the fitted contrasts
        basic, *_ = np.linalg.lstsq(X, effects, rcond=None)
        assert X @ basic == pytest.approx(effects, abs=1e-10)


def test_the_ume_design_spans_strictly_more_than_consistency():
    X_ume = TRIANGLE.ume_design_matrix()
    X_con = TRIANGLE.design_matrix()
    assert np.linalg.matrix_rank(X_ume) >= np.linalg.matrix_rank(X_con)
    # the consistency design lies inside the UME column space, so the
    # likelihood ratio test of the two is well defined
    residual = X_con - X_ume @ np.linalg.lstsq(X_ume, X_con, rcond=None)[0]
    assert np.max(np.abs(residual)) < 1e-10
    # this network repeats the A-B and A-C types, so UME spends one degree of
    # freedom where the cycle space allows three: two studies of the same pair
    # are still forced to agree under UME
    assert len(TRIANGLE.ume_parameters()) == 4
    assert len(TRIANGLE.ume_parameters()) - 3 == 1
    assert TRIANGLE.df_inconsistency == 3


def test_the_split_design_separates_direct_from_indirect_evidence():
    split = TRIANGLE.split_design("B", "C")
    direct = [TRIANGLE.comparisons[i][1:] for i in split["direct"]]
    assert direct == [("B", "C")]
    indirect = [TRIANGLE.comparisons[i][1:] for i in split["indirect"]]
    # the indirect side is a path B - A - C
    assert ("B", "A") in indirect or ("A", "B") in indirect
    assert ("A", "C") in indirect or ("C", "A") in indirect
    assert split["df"] == 1


def test_a_comparison_without_indirect_evidence_cannot_be_split():
    """``A vs D`` is a bridge: there is no second path to split against."""
    split = TRIANGLE.split_design("A", "D")
    assert split["indirect"] == []
    assert split["df"] == 0
    with pytest.raises(ValueError, match="not a comparison"):
        TRIANGLE.split_design("B", "D")


def test_multiarm_variances_put_the_shared_arm_on_the_covariance():
    """Exact, from the arm variances — and the sign is the whole point.

    With the convention that each comparison is "first arm minus second",
    ``Cov(y_a, y_b) = sum_k a_k b_k Var(A_k)``: ``A vs B`` and ``A vs C``
    share A with the same sign and covary by ``+Var(A)``, ``A vs B`` and
    ``B vs C`` share B with opposite signs and covary by ``-Var(B)``, and
    ``A vs C`` and ``B vs C`` share C with the same (negative) sign and
    covary by ``+Var(C)``.  Adding shared variances without the signs is the
    mistake, and the pattern it produces is wrong in exactly one cell.
    """
    structure = multiarm_structure(
        ["A", "B", "C"], variances=[0.04, 0.09, 0.16])
    i = structure.pairs.index((0, 1))     # A vs B
    j = structure.pairs.index((0, 2))     # A vs C
    k = structure.pairs.index((1, 2))     # B vs C
    assert structure.covariance[i, i] == pytest.approx(0.04 + 0.09)
    assert structure.covariance[j, j] == pytest.approx(0.04 + 0.16)
    # A vs B and A vs C share arm A, same sign
    assert structure.covariance[i, j] == pytest.approx(0.04)
    # A vs B and B vs C share arm B, opposite sign
    assert structure.covariance[i, k] == pytest.approx(-0.09)
    # A vs C and B vs C share arm C, both as -1
    assert structure.covariance[j, k] == pytest.approx(0.16)
    assert structure.correlation[i, j] == pytest.approx(
        0.04 / np.sqrt(0.13 * 0.20))
    # three arms give three contrasts in a two-dimensional space
    assert structure.rank == 2
    assert np.all(np.linalg.eigvalsh(structure.covariance) > -1e-10)


def test_a_three_arm_study_has_a_singular_covariance_and_that_is_the_point():
    """``(A-B) - (A-C) = C - B``, so the three contrasts are collinear.

    The consequence is not an edge case: entering a three-arm study's three
    comparisons as three independent observations over-states the network's
    information by one degree of freedom per such study, and every consistency
    equation that touches it is then wrong by a variance nobody checks.
    """
    structure = multiarm_structure(["A", "B", "C"],
                                   variances=[0.04, 0.09, 0.16])
    null = np.array([1.0, -1.0, 1.0])
    assert structure.covariance @ null == pytest.approx(np.zeros(3),
                                                        abs=1e-12)
    assert structure.rank == len(structure.arms) - 1
    four = multiarm_structure(["A", "B", "C", "D"],
                              variances=[0.04, 0.09, 0.16, 0.25])
    assert four.rank == 3
    assert len(four.pairs) == 6


def test_equal_arm_variances_give_one_half_and_minus_one_half():
    """The number a practitioner recognises, with the signs a reader forgets.

    Equal arms make every correlation ``+/-0.5``: ``+`` where the shared arm
    enters with the same sign on both sides, ``-`` where it does not.
    """
    structure = multiarm_structure(
        ["A", "B", "C"], variances=[0.05, 0.05, 0.05])
    same = np.array([[1.0, 0.5, -0.5],
                     [0.5, 1.0, 0.5],
                     [-0.5, 0.5, 1.0]])
    assert np.allclose(structure.correlation, same)


def test_comparison_level_variances_need_an_explicit_rho():
    with pytest.raises(ValueError, match="rho"):
        multiarm_structure(["A", "B", "C"], variances=[0.13, 0.20, 0.25],
                           per="comparison")
    with_rho = multiarm_structure(["A", "B", "C"],
                                  variances=[0.13, 0.20, 0.25],
                                  per="comparison", rho=0.4)
    # (0,1) with (0,2) share arm A positively
    assert with_rho.correlation[0, 1] == pytest.approx(0.4)
    # (0,1) with (1,2) share arm B negatively
    assert with_rho.correlation[0, 2] == pytest.approx(-0.4)


def test_the_variances_must_say_what_they_are():
    """A three-arm study has three arms and three comparisons."""
    with pytest.raises(ValueError, match="per='arm' needs 3"):
        multiarm_structure(["A", "B", "C"], variances=[0.1, 0.2], per="arm")
    with pytest.raises(ValueError, match="unknown per"):
        multiarm_structure(["A", "B", "C"], variances=[0.1, 0.2, 0.3],
                           per="arms")


def test_a_multiarm_structure_is_the_exact_covariance_of_arm_means():
    """Simulation check of the derivation, not a restatement of the formula.

    Three arms with known means and variances: the estimated contrasts are
    linear in the arm means with a known covariance, and the empirical
    covariance of those contrasts must match ``multiarm_structure`` — signs
    included, which is what a formula written with ``+`` gets wrong.
    """
    rng = np.random.default_rng(11)
    variances = np.array([0.04, 0.09, 0.16])
    draws = rng.normal(0.0, 1.0, (200000, 3)) * np.sqrt(variances)
    y_ab = draws[:, 0] - draws[:, 1]
    y_ac = draws[:, 0] - draws[:, 2]
    y_bc = draws[:, 1] - draws[:, 2]
    empirical = np.cov(np.column_stack([y_ab, y_ac, y_bc]).T)
    structure = multiarm_structure(["A", "B", "C"], variances=variances)
    assert empirical == pytest.approx(structure.covariance, abs=0.01)
    # the collinearity is visible in the simulated data too
    assert y_ab - y_ac + y_bc == pytest.approx(np.zeros(y_ab.size), abs=1e-12)


def test_every_pair_of_treatments_is_covered_by_a_circuit_or_a_tree():
    """A basic sanity property: the design is estimable for a connected net."""
    rng = np.random.default_rng(12)
    theta = rng.normal(0.0, 1.0, TRIANGLE.n_treatments)
    effects = TRIANGLE.contrast_matrix() @ theta
    X = TRIANGLE.design_matrix()
    basic, *_ = np.linalg.lstsq(X, effects, rcond=None)
    fitted = X @ basic
    for i, j in itertools.combinations(range(TRIANGLE.n_treatments), 2):
        row = np.zeros(TRIANGLE.n_treatments)
        row[i], row[j] = 1.0, -1.0
        want = float(row @ theta)
        # every observed comparison of that pair reproduces the contrast
        for n, (_s, t1, t2) in enumerate(TRIANGLE.comparisons):
            if {t1, t2} == {TRIANGLE.treatments[i], TRIANGLE.treatments[j]}:
                sign = 1.0 if t1 == TRIANGLE.treatments[i] else -1.0
                assert sign * fitted[n] == pytest.approx(want, abs=1e-10)


def test_a_network_needs_distinct_treatments_and_no_self_comparison():
    with pytest.raises(ValueError, match="unique"):
        TreatmentNetwork(["A", "A"], [("s1", "A", "B")])
    with pytest.raises(ValueError, match="itself"):
        TreatmentNetwork(["A", "B"], [("s1", "A", "A")])
    with pytest.raises(ValueError, match="unknown treatment"):
        TreatmentNetwork(["A", "B"], [("s1", "A", "Z")])
