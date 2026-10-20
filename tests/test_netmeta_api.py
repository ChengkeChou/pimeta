"""netmeta API surface: nettable, netrank, netmeasures, netsplit, vikor, ...

Every expected value in this file comes from R 4.6.1 / netmeta 3.7-0; see
``tests/data`` for the benchmark CSVs and the module docstrings of
``_refgen_nmapi4.R`` / ``_refgen_netcomplex3.R`` in the scratch area.
"""

from __future__ import annotations

import numpy as np
import pytest

import pimeta as ma

# ---------------------------------------------------------------------------
# Shared network: 5 treatments, 6 studies, two 3-arm studies
# ---------------------------------------------------------------------------
NMA_TE = [0.10, -0.30, -0.40, 0.55, -0.25, 0.20, 0.40, 0.20, 0.50, -0.10]
NMA_SE = [0.05, 0.06, 0.07, 0.10, 0.12, 0.09, 0.11, 0.13, 0.15, 0.14]
NMA_T1 = ["A", "A", "B", "A", "A", "D", "D", "E", "B", "B"]
NMA_T2 = ["B", "C", "C", "D", "D", "E", "C", "C", "E", "E"]
NMA_ST = ["k1", "k1", "k1", "k2", "k3", "k4", "k4", "k4", "k5", "k6"]

# netrank(netmeta(...))$ranking.common / $ranking.random
R_PSCORE_COMMON = [0.37567114774317462, 0.75989786785063695,
                   0.00689534984050824, 0.39084163142860018,
                   0.96669400313708043]
R_PSCORE_RANDOM = [0.396220823655634, 0.544264937962689, 0.464079538784379,
                   0.430444273402454, 0.664990426194844]

# netmeasures(...)$minpar / $minpar.study
R_MINPAR = {
    "A:B": 1.47513645230031, "A:C": 1.59376454173211, "A:D": 1.32089130932702,
    "A:E": 1.74228800867963, "B:C": 1.57869443585506, "B:D": 1.72264201773935,
    "B:E": 1.32945560970671, "C:D": 1.62472128340332, "C:E": 1.62766346966942,
    "D:E": 1.49481854200917,
}
R_MINPAR_STUDY_D = {
    "A:D": 2.61609808734216, "B:E": 2.64246184401183,
}

# netsplit(...)$compare.common / $compare.random — the direct-vs-indirect test
R_COMPARE_COMMON = {
    "A:B": (-0.0654086404407453, 0.163800207537668),
    "A:C": (-0.9241253451464360, 0.144112317343926),
    "A:D": (0.5973585574233823, 0.126477724521696),
    "B:C": (-0.9423017565578831, 0.173101335405281),
    "B:E": (0.2051810882237991, 0.154598121682016),
    "C:D": (-0.9186023660463277, 0.143262859207011),
    "C:E": (-0.9448392366317171, 0.172710038099353),
    "D:E": (0.0205144603432108, 0.163134814104017),
}
R_COMPARE_RANDOM = {
    "A:B": (-0.0626741494589314, 0.814141029055840),
    "D:E": (0.0516380808087190, 0.814151607357930),
}
R_PROP_COMMON = [0.906822487260051, 0.826659399286910, 0.631069484514081,
                 0.0, 0.836470825867087, 0.0, 0.561723618383146,
                 0.410453396838232, 0.433432648483008, 0.695637099550925]
R_PROP_RANDOM = [0.676564984841621, 0.604963320140607, 0.757064561587197,
                 0.0, 0.609431120856224, 0.0, 0.752187581667814,
                 0.587984805271410, 0.587166361081043, 0.668124931430945]

# netdistance(netmeta(...))
R_NETDISTANCE = [
    [0, 1, 1, 1, 2],
    [1, 0, 1, 2, 1],
    [1, 1, 0, 1, 1],
    [1, 2, 1, 0, 1],
    [2, 1, 1, 1, 0],
]

# netmatrix(netmeta(...), TE.random[upper.tri(TE.random)])
R_NETMATRIX = [
    [None, 0.120271, 0.053626, 0.024216, None],
    [0.120271, None, -0.066645, None, 0.182863],
    [0.053626, -0.066645, None, 0.207078, 0.086807],
    [0.024216, None, 0.207078, None, -0.029411],
    [None, 0.182863, 0.086807, -0.029411, None],
]

# hatmatrix(x, method = "Davies", type = ...)$common — column sums
R_HAT_COLSUM = {
    "full": [0.54695415, 0.92534658, 2.52769926, 0.00000000, 0.18282388,
             0.00000000, 2.36413027, 0.72493685, 0.38323361, 1.25263612],
    "short": [0.714657144, 0.661600577, 1.623742279, 0.065995069,
              1.648662074, 0.445006831, 0.282588816, 1.068749110],
    "long": [0.54695415, 0.92534658, 2.52769926, 0.18282388, 2.36413027,
             0.72493685, 0.38323361, 1.25263612],
}
R_HAT_SHAPE = {"full": (10, 10), "short": (8, 8), "long": (10, 8)}


@pytest.fixture(scope="module")
def nma():
    return ma.netmeta(NMA_TE, NMA_SE, NMA_T1, NMA_T2, NMA_ST, sm="MD")


# ---------------------------------------------------------------------------
# treats / comps / combinations / createC / invmat
# ---------------------------------------------------------------------------
def test_treats_short_names_are_truncated():
    assert ma.treats(["A", "B", "C", "D", "E"], nchar_trts=2) == \
        ["A", "B", "C", "D", "E"]


@pytest.mark.parametrize("names, nchar, want", [
    # R: abbreviate(c("aa1","aa2","bb"), 2)
    (["aa1", "aa2", "bb"], 2, ["a1", "a2", "bb"]),
    # R: abbreviate(c("Placebo","Metformin","Metformin+Placebo"), 8)
    (["Placebo", "Metformin", "Metformin+Placebo"], 8,
     ["Placebo", "Metformn", "Mtfrmn+P"]),
    (["Placebo", "Metformin", "Metformin+Placebo"], 3, ["Plc", "Mtf", "M+P"]),
    # the abbreviation has to grow until it is unique
    (["abcd", "abce", "abcf"], 2, ["abcd", "abce", "abcf"]),
    (["aaaa", "bbbb"], 2, ["aa", "bb"]),
])
def test_treats_matches_r_abbreviate(names, nchar, want):
    assert ma.treats(names, nchar_trts=nchar) == want


def test_treats_uses_row_or_column_names():
    class _Frame:
        def __init__(self):
            self.index = ["aa1", "aa2"]
            self.columns = ["xx", "yy"]
            self._a = np.zeros((2, 2))
            self.shape = (2, 2)
            self.ndim = 2

        def __array__(self, dtype=None):
            return self._a

    frame = _Frame()
    assert ma.treats(frame, nchar_trts=2, row=True) == ["a1", "a2"]
    assert ma.treats(frame, nchar_trts=2, row=False) == ["xx", "yy"]


def test_treats_on_a_plain_matrix_falls_back_to_indices():
    assert ma.treats(np.zeros((2, 2)), nchar_trts=2) == ["0", "1"]


def test_comps_splits_components():
    assert ma.comps("A+B") == ["A", "B"]
    assert ma.comps(["A+B", "C"]) == [["A", "B"], ["C"]]


def test_comps_orders_by_the_trts_argument():
    """netmeta::comps() takes a reference list of components."""
    assert ma.comps("B+A", trts=["A", "B", "C"]) == ["A", "B"]
    assert ma.comps(["B+A", "C"], trts=["A", "B", "C"]) == [["A", "B"], ["C"]]


def test_comps_rejects_a_component_outside_trts():
    with pytest.raises(ValueError, match="not in 'trts'"):
        ma.comps("A+Z", trts=["A", "B"])


def test_combinations_returns_sorted_unique_components():
    assert ma.combinations("A+B") == ["A", "B"]
    assert ma.combinations(["A+B", "C", "A"]) == ["A", "C", "A+B"]


def test_create_c_matrix_drops_the_inactive_component():
    c_mat, comps, names = ma.createC(["A+B", "C+D", "A+C"], inactive="D")
    assert comps == ["A", "B", "C"]
    assert names == ["A+B", "C+D", "A+C"]
    assert c_mat.tolist() == [[1.0, 1.0, 0.0],
                              [0.0, 0.0, 1.0],
                              [1.0, 0.0, 1.0]]


def test_create_c_matrix_without_inactive_keeps_every_component():
    c_mat, comps, _ = ma.createC(["A+B", "C+D"])
    assert comps == ["A", "B", "C", "D"]
    assert c_mat.shape == (2, 4)


def test_create_c_matrix_partial_match_of_inactive():
    """netmeta::setchar() first tries an exact match, then a partial one."""
    _, comps, _ = ma.createC(["A+B", "Cmet"], inactive="Cm")
    assert comps == ["A", "B"]


def test_create_c_matrix_rejects_ambiguous_inactive():
    with pytest.raises(ValueError, match="multiple components"):
        ma.createC(["Aa+X", "Ab+Y"], inactive="A")


def test_create_c_matrix_rejects_unknown_inactive():
    with pytest.raises(ValueError, match="not a treatment component"):
        ma.createC(["A+B", "C"], inactive="Z")


def test_invmat_inverts_a_laplacian():
    # Laplacian of the path A-B-C
    lap = np.array([[1.0, -1.0, 0.0], [-1.0, 2.0, -1.0], [0.0, -1.0, 1.0]])
    inv = ma.invmat(lap)
    # netmeta::invmat is the Moore-Penrose pseudoinverse of the Laplacian
    assert inv == pytest.approx(np.linalg.pinv(lap), abs=1e-12)
    assert inv.sum(axis=0) == pytest.approx(np.zeros(3), abs=1e-12)


# ---------------------------------------------------------------------------
# nettable
# ---------------------------------------------------------------------------
def test_nettable_lists_every_comparison(nma):
    """netmeta lists all n(n-1)/2 comparisons, even without direct evidence."""
    tab = ma.nettable(nma)
    assert len(tab.rows) == 10
    assert [(r["treat1"], r["treat2"]) for r in tab.rows] == [
        ("A", "B"), ("A", "C"), ("A", "D"), ("A", "E"), ("B", "C"),
        ("B", "D"), ("B", "E"), ("C", "D"), ("C", "E"), ("D", "E"),
    ]


def test_nettable_counts_multi_arm_studies(nma):
    rows = {(r["treat1"], r["treat2"]): r for r in ma.nettable(nma).rows}
    # A:D is studied by k2 and by k3, both 2-arm studies
    assert rows[("A", "D")]["studies"] == 2
    assert rows[("A", "D")]["n.2arm"] == 2
    assert rows[("A", "D")]["narms"] == 2
    # A:B is a single 3-arm study
    assert rows[("A", "B")]["studies"] == 1
    assert rows[("A", "B")]["narms"] == 3
    assert rows[("A", "B")]["n.2arm"] == 0
    # A:E and B:D are indirect only -> no direct row
    assert rows[("A", "E")]["studies"] == 0
    assert rows[("A", "E")]["narms"] == 0
    assert np.isnan(rows[("A", "E")]["TE.direct.common"])


def test_nettable_network_estimates_match_the_model(nma):
    rows = {(r["treat1"], r["treat2"]): r for r in ma.nettable(nma).rows}
    pos = {t: i for i, t in enumerate(nma.trts)}
    for (a, b), row in rows.items():
        i, j = pos[a], pos[b]
        assert row["TE.network.common"] == pytest.approx(
            nma.TE_common[i, j], abs=1e-12)
        assert row["TE.network.random"] == pytest.approx(
            nma.TE_random[i, j], abs=1e-12)


def test_nettable_direct_estimates_match_r(nma):
    rows = {(r["treat1"], r["treat2"]): r for r in ma.nettable(nma).rows}
    assert rows[("A", "B")]["TE.direct.common"] == pytest.approx(0.1, abs=1e-12)
    assert rows[("B", "E")]["TE.direct.random"] == pytest.approx(
        0.1981, abs=1e-4)
    assert rows[("A", "B")]["TE.network.random"] == pytest.approx(
        0.12027101448028808, abs=1e-12)


# ---------------------------------------------------------------------------
# netrank
# ---------------------------------------------------------------------------
def test_netrank_pscores_match_r(nma):
    res = ma.netrank(nma)
    assert res.trts == list(nma.trts)
    assert list(res.ranking_common) == pytest.approx(R_PSCORE_COMMON, abs=1e-14)
    assert list(res.ranking_random) == pytest.approx(R_PSCORE_RANDOM, abs=1e-14)


def test_netrank_pscore_always_sums_to_half(nma):
    res = ma.netrank(nma)
    # R: sum(netrank(x)$ranking.common) == 2.5 for this 5-treatment network
    assert sum(res.ranking_common) == pytest.approx(2.5, abs=1e-12)
    assert sum(res.ranking_random) == pytest.approx(2.5, abs=1e-12)


def test_netrank_rejects_unknown_small_values(nma):
    with pytest.raises(ValueError, match="small_values"):
        ma.netrank(nma, small_values="good")


# ---------------------------------------------------------------------------
# netmeasures
# ---------------------------------------------------------------------------
def test_netmeasures_minpar_matches_r(nma):
    res = ma.netmeasures(nma)
    assert res.minpar == pytest.approx(R_MINPAR, abs=1e-11)


def test_netmeasures_minpar_study_uses_within_design_routes(nma):
    """minpar.study re-routes within a design, so A:D and B:E differ."""
    res = ma.netmeasures(nma)
    for comp, val in R_MINPAR_STUDY_D.items():
        assert res.minpar_study[comp] == pytest.approx(val, abs=1e-11)
    # the other comparisons have only one route and are unchanged
    for comp in ("A:B", "A:C", "C:D"):
        assert res.minpar_study[comp] == pytest.approx(res.minpar[comp],
                                                       abs=1e-12)


def test_netmeasures_proportion_is_per_comparison(nma):
    """netmeasures()$proportion is the random-effects direct-evidence share."""
    res = ma.netmeasures(nma)
    assert isinstance(res.proportion, dict)
    assert list(res.proportion) == list(res.minpar)
    assert list(res.proportion.values()) == pytest.approx(
        R_PROP_RANDOM, abs=1e-11)
    assert res.proportion["A:E"] == 0.0


# ---------------------------------------------------------------------------
# netsplit
# ---------------------------------------------------------------------------
def test_netsplit_lists_every_comparison(nma):
    res = ma.netsplit(nma)
    assert res.comparison == ["A:B", "A:C", "A:D", "A:E", "B:C", "B:D",
                              "B:E", "C:D", "C:E", "D:E"]


def test_netsplit_indirect_vs_direct_matches_r(nma):
    res = ma.netsplit(nma)
    comps = list(res.comparison)
    for comp, (te, se) in R_COMPARE_COMMON.items():
        k = comps.index(comp)
        assert res.compare_common["TE"][k] == pytest.approx(te, abs=1e-10), comp
        assert res.compare_common["seTE"][k] == pytest.approx(se, abs=1e-11), comp


def test_netsplit_direct_estimates_match_the_network(nma):
    res = ma.netsplit(nma)
    comps = list(res.comparison)
    pos = {t: i for i, t in enumerate(nma.trts)}
    for comp in ("A:B", "C:E", "D:E"):
        a, b = comp.split(":")
        k = comps.index(comp)
        assert res.direct_common["TE"][k] == pytest.approx(
            nma.TE_direct_common[pos[a], pos[b]], abs=1e-12)
        assert res.indirect_common["TE"][k] + res.compare_common["TE"][k] == \
            pytest.approx(res.direct_common["TE"][k], abs=1e-10)


def test_netsplit_random_effects_matches_r(nma):
    res = ma.netsplit(nma)
    comps = list(res.comparison)
    for comp, (te, se) in R_COMPARE_RANDOM.items():
        k = comps.index(comp)
        assert res.compare_random["TE"][k] == pytest.approx(te, abs=1e-10), comp
        assert res.compare_random["seTE"][k] == pytest.approx(se, abs=1e-11), comp


def test_netsplit_proportion_of_direct_evidence_matches_r(nma):
    res = ma.netsplit(nma)
    assert list(res.prop_common) == pytest.approx(R_PROP_COMMON, abs=1e-11)
    assert list(res.prop_random) == pytest.approx(R_PROP_RANDOM, abs=1e-11)


def test_netsplit_comparisons_without_direct_evidence_are_none(nma):
    res = ma.netsplit(nma)
    comps = list(res.comparison)
    for comp in ("A:E", "B:D"):
        k = comps.index(comp)
        assert res.direct_common["TE"][k] is None
        assert res.prop_common[k] == 0.0
        assert res.indirect_common["TE"][k] is not None


# ---------------------------------------------------------------------------
# decomp_design
# ---------------------------------------------------------------------------
def test_decomp_design_matches_r(nma):
    dd = ma.decomp_design(nma)
    q = dd.Q_decomp
    assert q["Total"] == pytest.approx(
        (75.9059606003131, 4.0, 1.28165393453076e-15), abs=1e-12)
    assert q["Within designs"] == pytest.approx(
        (34.7805770803318, 2.0, 2.80216463878911e-08), abs=1e-9)
    assert q["Between designs"] == pytest.approx(
        (41.1253835199813, 2.0, 1.17418474995086e-09), abs=1e-10)

    het = {r["design"]: r for r in dd.Q_het_design}
    assert het["A:D"]["Q"] == pytest.approx(26.22950819672131, abs=1e-10)
    assert het["B:E"]["Q"] == pytest.approx(8.55106888361045, abs=1e-10)
    assert het["A:B:C"]["df"] == 0.0
    assert np.isnan(het["A:B:C"]["pval"])

    det = {r["design"]: r for r in dd.Q_inc_detach}
    assert det["A:D"]["Q"] == pytest.approx(18.8183360237329, abs=1e-9)
    assert det["B:E"]["Q"] == pytest.approx(39.3639501018292, abs=1e-9)


def test_decomp_design_inconsistency_random_matches_r(nma):
    inc = ma.decomp_design(nma).Q_inc_random
    assert inc["Q"] == pytest.approx(1.4773340956349, abs=1e-10)
    assert inc["df"] == 2.0
    assert inc["pval"] == pytest.approx(0.477750309603268, abs=1e-12)
    assert inc["tau.within"] == pytest.approx(0.503174284818149, abs=1e-12)


def test_decomp_design_matches_the_netmeta_q_statistics(nma):
    dd = ma.decomp_design(nma)
    assert dd.Q_decomp["Total"][0] == pytest.approx(nma.Q, abs=1e-10)
    assert dd.Q_decomp["Total"][1] == pytest.approx(nma.df_Q, abs=1e-12)
    within = dd.Q_decomp["Within designs"][0]
    between = dd.Q_decomp["Between designs"][0]
    assert within + between == pytest.approx(nma.Q, abs=1e-9)


def test_decomp_design_tau_preset_changes_the_between_designs_test(nma):
    dd = ma.decomp_design(nma, tau_preset=0.1)
    assert dd.tau_preset == 0.1
    assert dd.Q_inc_random_preset["tau.preset"] == 0.1
    assert dd.Q_inc_random_preset["Q"] == pytest.approx(
        19.47864381614158, abs=1e-10)


def test_decomp_design_rejects_a_negative_tau_preset(nma):
    with pytest.raises(ValueError, match="non-negative"):
        ma.decomp_design(nma, tau_preset=-1.0)


def test_decomp_design_warns_for_a_two_treatment_network():
    m = ma.netmeta([0.1, 0.2], [0.05, 0.06], ["A", "A"], ["B", "B"],
                   ["s1", "s2"], sm="MD")
    assert m.n == 2
    with pytest.warns(UserWarning, match="two treatments"):
        assert ma.decomp_design(m) is None


# ---------------------------------------------------------------------------
# netdistance / netmatrix / netconnection / netcomplexity
# ---------------------------------------------------------------------------
def test_netdistance_matches_r(nma):
    res = ma.netdistance(nma)
    assert res.labels == ["A", "B", "C", "D", "E"]
    assert np.asarray(res.D).tolist() == R_NETDISTANCE


def test_netdistance_is_symmetric_with_a_zero_diagonal(nma):
    d = np.asarray(ma.netdistance(nma).D, dtype=float)
    assert np.allclose(d, d.T)
    assert np.all(np.diag(d) == 0.0)


def test_netdistance_order_is_available(nma):
    res = ma.netdistance(nma)
    assert isinstance(res.order, list)
    assert np.asarray(res.D).shape == (nma.n, nma.n)
    assert list(res.labels) == list(nma.trts)


def test_netmatrix_reproduces_the_study_level_variable(nma):
    """var holds one value per *comparison row*; the result is symmetric."""
    var = np.asarray(nma.TE_raw, dtype=float)
    mat = np.asarray(ma.netmatrix(nma, var).matrix, dtype=float)
    pos = {t: i for i, t in enumerate(nma.trts)}
    seen = {}
    for a, b, v in zip(nma.treat1_raw, nma.treat2_raw, var):
        seen.setdefault(frozenset((a, b)), []).append(v)
    for pair, vals in seen.items():
        a, b = tuple(pair)
        if len(vals) == 1:            # pairs studied once map straight through
            assert mat[pos[a], pos[b]] == pytest.approx(vals[0], abs=1e-12)
            assert mat[pos[b], pos[a]] == pytest.approx(vals[0], abs=1e-12)
        else:                         # shared pairs are summarised
            assert mat[pos[a], pos[b]] in vals
            assert mat[pos[a], pos[b]] == mat[pos[b], pos[a]]
    # A:E is never studied directly
    assert np.isnan(mat[pos["A"], pos["E"]])
    assert np.all(np.isnan(np.diag(mat)))


def test_netmatrix_rejects_a_mismatched_var(nma):
    with pytest.raises(ValueError, match="number of comparisons"):
        ma.netmatrix(nma, [1.0, 2.0])


def test_netconnection_matches_r():
    res = ma.netconnection(NMA_T1, NMA_T2, NMA_ST)
    assert (res.n, res.k, res.m, res.d) == (5, 6, 10, 4)
    assert res.n_subnets == 1


def test_netcomplexity_counts_independent_cycles(nma):
    assert ma.netcomplexity(nma) == nma.m - nma.n + 1


# ---------------------------------------------------------------------------
# hatmatrix
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("type_", ["full", "short", "long"])
def test_hatmatrix_davies_matches_r(nma, type_):
    out = ma.hatmatrix(nma, method="Davies", type=type_)
    h = np.asarray(out["common"], dtype=float)
    assert h.shape == R_HAT_SHAPE[type_]
    assert h.sum(axis=1) == pytest.approx(R_HAT_COLSUM[type_], abs=1e-7) or \
        h.sum(axis=0) == pytest.approx(R_HAT_COLSUM[type_], abs=1e-7)


def test_hatmatrix_ruecker_is_the_default(nma):
    """netmeta 3.7-0's default is method = "Ruecker"."""
    out = ma.hatmatrix(nma)
    assert out["method"] == "Ruecker"
    assert np.asarray(out["common"]).shape == (10, 10)
    assert np.asarray(out["random"]).shape == (10, 10)


def test_hatmatrix_krahn_uses_the_design_decomposition(nma):
    out = ma.hatmatrix(nma, method="Krahn", type="full")
    assert out["method"] == "Krahn"
    assert np.asarray(out["common"]).ndim == 2
    assert np.asarray(out["common"]).shape[0] == nma.m


def test_hatmatrix_krahn_studies(nma):
    out = ma.hatmatrix(nma, method="Krahn", type="studies")
    assert np.asarray(out["common"]).shape[0] == nma.m


def test_hatmatrix_rejects_an_unknown_method(nma):
    with pytest.raises(ValueError, match="Ruecker"):
        ma.hatmatrix(nma, method="Nope")


# ---------------------------------------------------------------------------
# vikor
# ---------------------------------------------------------------------------
R_VIKOR_Q = {"E": 0.0, "B": 0.232810254383247, "D": 0.557385832600217,
             "A": 0.590189364023190, "C": 1.0}


def test_vikor_on_a_dataframe_matches_r(nma):
    class _Frame:
        def __init__(self):
            self.index = list(nma.trts)
            self._a = np.column_stack([R_PSCORE_COMMON, R_PSCORE_RANDOM])

        def to_numpy(self):
            return self._a

    res = ma.vikor(_Frame())
    assert res.trts == ["E", "B", "D", "A", "C"]
    assert res.Q == pytest.approx(R_VIKOR_Q, abs=1e-13)


def test_vikor_uses_the_pscores_of_a_netmeta_result(nma):
    res = ma.vikor(nma)
    assert res.trts[0] == "E" and res.trts[-1] == "C"


def test_vikor_on_a_plain_array_labels_rows_by_index():
    res = ma.vikor([[0.9, 0.8], [0.5, 0.6], [0.2, 0.1]])
    assert res.trts == ["1", "2", "3"]


def test_vikor_flattens_a_one_dimensional_input():
    res = ma.vikor([0.9, 0.5, 0.2])
    assert res.Q["1"] == pytest.approx(0.0)
    assert res.Q["3"] == pytest.approx(1.0)


def test_vikor_normalises_the_weights():
    res = ma.vikor([[0.9, 0.8], [0.5, 0.6], [0.1, 0.3]], weights=[2.0, 2.0])
    assert res.weights == pytest.approx([0.5, 0.5])


def test_vikor_rejects_a_wrong_number_of_weights():
    with pytest.raises(ValueError, match="Number of weights"):
        ma.vikor([[0.9, 0.8], [0.5, 0.6]], weights=[1.0])


def test_vikor_rejects_negative_weights():
    with pytest.raises(ValueError, match="non-negative"):
        ma.vikor([[0.9, 0.8], [0.5, 0.6]], weights=[-1.0, 2.0])


def test_vikor_rejects_all_zero_weights():
    with pytest.raises(ValueError, match="larger than 0"):
        ma.vikor([[0.9, 0.8], [0.5, 0.6]], weights=[0.0, 0.0])


def test_vikor_drops_incomplete_treatments():
    res = ma.vikor([[0.9, 0.8], [np.nan, 0.6], [0.2, 0.1]])
    assert "2" not in res.trts
    assert res.dropped == ["2"]


def test_vikor_needs_two_complete_treatments():
    with pytest.raises(ValueError, match="at least two"):
        ma.vikor([[0.9, np.nan], [np.nan, 0.6]])


def test_vikor_rejects_identical_regret():
    with pytest.raises(ValueError, match="statistic R"):
        ma.vikor([[0.0, 0.9], [0.0, 0.9], [0.0, 0.9]])


def test_vikor_rejects_identical_group_utility():
    # every treatment has the same worst-case regret contribution vector
    with pytest.raises(ValueError, match="statistic S"):
        ma.vikor([[0.9, 0.1], [0.9, 0.1], [0.5, 0.5]])


def test_vikor_v_controls_the_strategy_weight():
    x = [[0.9, 0.1], [0.5, 0.5], [0.2, 0.9]]
    # v = 1 is pure group utility (S) and v = 0 pure regret (R); both are
    # rescaled to [0, 1], so compare the induced ordering instead
    assert list(ma.vikor(x, v=1.0).Q) == list(ma.vikor(x, v=1.0).S)
    assert list(ma.vikor(x, v=0.0).Q) == list(ma.vikor(x, v=0.0).R)


# ---------------------------------------------------------------------------
# subgroup
# ---------------------------------------------------------------------------
def test_subgroup_refits_each_level(nma):
    res = ma.subgroup(nma, ["x", "x", "y", "y", "x", "y"])
    assert res.levels == ["x", "y"]
    assert np.asarray(res.TE_random).shape == (nma.n, nma.n, 2)


def test_subgroup_single_level_reproduces_the_whole_network(nma):
    res = ma.subgroup(nma, ["x"] * 6)
    assert res.levels == ["x"]
    net = res.networks["x"]
    assert net.k == nma.k and net.tau2 == pytest.approx(nma.tau2, abs=1e-12)


def test_subgroup_leaves_missing_comparisons_nan(nma):
    # "x" = {k1, k2, k3} covers A, B, C, D only; "y" = {k4, k5, k6} covers
    # B, C, D, E -> A never appears in "y" and E never appears in "x"
    res = ma.subgroup(nma, ["x", "x", "x", "y", "y", "y"])
    te = np.asarray(res.TE_random, dtype=float)
    pos = {t: i for i, t in enumerate(nma.trts)}
    assert np.isfinite(te[pos["A"], pos["B"], 0])
    assert np.isnan(te[pos["A"], pos["B"], 1])
    assert np.isnan(te[pos["A"], pos["E"], 0])
    assert np.isnan(te[pos["A"], pos["E"], 1])
    assert np.isfinite(te[pos["D"], pos["E"], 1])


def test_subgroup_records_tau2_per_level(nma):
    res = ma.subgroup(nma, ["x"] * 6)
    assert res.tau2["x"] == pytest.approx(nma.tau2, abs=1e-12)


def test_subgroup_only_connected_is_accepted(nma):
    res = ma.subgroup(nma, ["x"] * 6, only_connected=True)
    assert res.levels == ["x"]


# ---------------------------------------------------------------------------
# netcomplex (netmeta::netcomplex — combination effects)
# ---------------------------------------------------------------------------
CMP_TE = [0.10, -0.20, 0.30, -0.10, 0.20, 0.15, -0.25, 0.05]
CMP_SE = [0.10, 0.11, 0.12, 0.13, 0.14, 0.15, 0.16, 0.17]
CMP_T1 = ["A", "A+B", "A+B+C", "A", "A+B", "B", "C", "A+C"]
CMP_T2 = ["A+B", "A+B+C", "C", "C", "C", "B+C", "B+C", "B"]
CMP_ST = ["s1", "s2", "s3", "s4", "s5", "s6", "s7", "s8"]

R_COMP_COMMON = [0.1207660168285, 0.0893797592132, 0.0885933984589]
R_COMP_RANDOM = [0.105653303335, 0.113549255467, 0.078789957336]
R_COMB_COMMON = [0.1207660168285, 0.2101457760417, 0.2987391745007,
                 0.2093594152874, 0.0893797592132, 0.1779731576721,
                 0.0885933984589]
R_SECOMB_COMMON = [0.0747836634588, 0.0839001126172, 0.1242578483168,
                   0.1103092908711, 0.0635983991076, 0.1007670782717,
                   0.0654523265850]
R_COMB_RANDOM = [0.105653303335, 0.219202558803, 0.297992516139,
                 0.184443260671, 0.113549255467, 0.192339212803,
                 0.078789957336]
R_SECOMB_RANDOM = [0.1026167553528, 0.1222845419851, 0.1805737834847,
                   0.1519828649330, 0.0929582315876, 0.1499152141124,
                   0.0935336799399]


@pytest.fixture(scope="module")
def netcomb_result():
    m = ma.netmeta(CMP_TE, CMP_SE, CMP_T1, CMP_T2, CMP_ST, sm="MD")
    return ma.netcomb(m, sep_comps="+")


def test_netcomb_component_effects_matches_r(netcomb_result):
    assert netcomb_result.comps == ["A", "B", "C"]
    assert np.asarray(netcomb_result.Comp_common) == pytest.approx(
        R_COMP_COMMON, abs=1e-11)
    assert np.asarray(netcomb_result.Comp_random) == pytest.approx(
        R_COMP_RANDOM, abs=1e-10)


def test_netcomplex_combination_effects_match_r(netcomb_result):
    res = ma.netcomplex(netcomb_result)
    assert res.complex == ["A", "A+B", "A+B+C", "A+C", "B", "B+C", "C"]
    assert res.Comb_common == pytest.approx(R_COMB_COMMON, abs=1e-11)
    assert res.seComb_common == pytest.approx(R_SECOMB_COMMON, abs=1e-11)
    assert res.Comb_random == pytest.approx(R_COMB_RANDOM, abs=1e-10)
    assert res.seComb_random == pytest.approx(R_SECOMB_RANDOM, abs=1e-10)


def test_netcomplex_selects_requested_complexes(netcomb_result):
    res = ma.netcomplex(netcomb_result, complex=["A+B", "B+C"])
    assert res.complex == ["A+B", "B+C"]
    assert res.Comb_common == pytest.approx([0.210145776042, 0.177973157672],
                                            abs=1e-11)
    assert res.seComb_common == pytest.approx([0.0839001126172,
                                               0.1007670782717], abs=1e-11)


def test_netcomplex_common_equals_the_additive_component_sum(netcomb_result):
    res = ma.netcomplex(netcomb_result)
    k = res.complex.index("A+B+C")
    assert res.Comb_common[k] == pytest.approx(
        sum(R_COMP_COMMON), abs=1e-11)


def test_netcomplex_rejects_an_unknown_component(netcomb_result):
    with pytest.raises(ValueError, match="unknown component"):
        ma.netcomplex(netcomb_result, complex=["A+Z"])


def test_netcomplex_requires_a_netcomb_result(nma):
    with pytest.raises(TypeError, match="NetcombResult"):
        ma.netcomplex(nma)


def test_netcomplex_ci_uses_the_requested_level(netcomb_result):
    res = ma.netcomplex(netcomb_result, level=0.90)
    assert res.level == 0.90
    assert np.all(res.lower_Comb_common < res.Comb_common)
    assert np.all(res.upper_Comb_common > res.Comb_common)


# ---------------------------------------------------------------------------
# pairwise
# ---------------------------------------------------------------------------
def test_pairwise_binary_two_arm_matches_r():
    res = ma.pairwise(treat=["A", "B", "A", "C"], n=[40, 50, 45, 55],
                      event=[10, 12, 11, 14],
                      studlab=["s1", "s1", "s2", "s2"], sm="OR")
    assert res.TE == pytest.approx([0.0540672212703, -0.0539505147287],
                                   abs=1e-11)
    assert res.seTE == pytest.approx([0.492932506679, 0.464908239275],
                                     abs=1e-11)
    assert res.treat1 == ["A", "A"] and res.treat2 == ["B", "C"]


def test_pairwise_binary_three_arm_matches_r():
    res = ma.pairwise(treat=["A", "B", "C"], n=[40, 50, 60],
                      event=[10, 12, 15], studlab=["s1", "s1", "s1"], sm="OR")
    assert res.TE == pytest.approx([0.0540672212703, 0.0, -0.0540672212703],
                                   abs=1e-11)
    assert res.seTE == pytest.approx([0.492932506679, 0.471404520791,
                                      0.445576044796], abs=1e-11)
    assert list(zip(res.treat1, res.treat2)) == [("A", "B"), ("A", "C"),
                                                 ("B", "C")]


def test_pairwise_continuous_matches_r():
    res = ma.pairwise(treat=["A", "B"], n=[40, 50], mean=[10, 12],
                      sd=[3, 4], studlab=["s1", "s1"], sm="MD")
    assert res.TE == pytest.approx([-2.0], abs=1e-12)
    assert res.seTE == pytest.approx([0.738241153012], abs=1e-11)


def test_pairwise_defaults_the_effect_measure():
    assert ma.pairwise(treat=["A", "B"], n=[10, 10], event=[2, 3],
                       studlab=["s", "s"]).sm == "OR"
    assert ma.pairwise(treat=["A", "B"], n=[10, 10], mean=[1, 2], sd=[1, 1],
                       studlab=["s", "s"]).sm == "MD"


def test_pairwise_accepts_a_scalar_n():
    res = ma.pairwise(treat=["A", "B"], n=10.0, event=[2, 3],
                      studlab=["s", "s"])
    assert len(res.TE) == 1


def test_pairwise_rejects_a_mismatched_vector():
    with pytest.raises(ValueError, match="length of 'n'"):
        ma.pairwise(treat=["A", "B"], n=[1, 2, 3], event=[1, 2],
                    studlab=["s", "s"])


def test_pairwise_needs_outcome_data():
    with pytest.raises(ValueError, match="no outcome data"):
        ma.pairwise(treat=["A", "B"], n=[10, 10], studlab=["s", "s"])


def test_pairwise_needs_a_full_continuous_outcome():
    with pytest.raises(ValueError, match="event.*or.*mean"):
        ma.pairwise(treat=["A", "B"], studlab=["s", "s"])


def test_pairwise_rejects_data_frames():
    with pytest.raises(TypeError, match="dict or a DataFrame"):
        ma.pairwise(treat=["A", "B"], data=object())


def test_pairwise_accepts_a_data_mapping():
    """metafor's ``data=`` evaluates the arguments inside the data frame."""
    res = ma.pairwise(data={"treat": ["A", "B"], "n": [10, 10],
                            "event": [2, 3], "studlab": ["s", "s"]})
    assert res.sm == "OR"
    assert len(res.TE) == 1


def test_pairwise_explicit_arguments_win_over_the_data_columns():
    res = ma.pairwise(data={"treat": ["A", "B"], "n": [10, 10],
                            "event": [2, 3], "studlab": ["s", "s"]},
                      n=[20, 20])
    ref = ma.pairwise(treat=["A", "B"], n=[20, 20], event=[2, 3],
                      studlab=["s", "s"])
    assert res.TE == pytest.approx(ref.TE)


def test_pairwise_data_needs_a_treat_column():
    with pytest.raises(ValueError, match="'treat' column"):
        ma.pairwise(data={"n": [10, 10], "event": [2, 3]})


def test_pairwise_result_round_trips():
    res = ma.pairwise(treat=["A", "B"], n=[10, 10], event=[2, 3],
                      studlab=["s", "s"])
    d = res.to_dict()
    assert d["sm"] == "OR" and len(d["TE"]) == 1


# ---------------------------------------------------------------------------
# gemtc2netmeta / crossnma2netmeta
# ---------------------------------------------------------------------------
def test_gemtc2netmeta_reads_contrast_data():
    obj = {"treatments": ["A", "B", "C"],
           "data.re": {"study": ["s1", "s1", "s2", "s2"],
                       "treatment": ["A", "B", "A", "C"],
                       "diff": [0.0, 0.2, 0.0, -0.3],
                       "std.err": [0.0, 0.1, 0.0, 0.12]}}
    res = ma.gemtc2netmeta(obj)
    assert res.trts == ["A", "B", "C"]
    # the first row of each study is the baseline
    assert list(zip(res.treat1_raw, res.treat2_raw)) == [("B", "A"),
                                                         ("C", "A")]


def test_gemtc2netmeta_rejects_a_network_without_data():
    with pytest.raises(ValueError, match="data.re"):
        ma.gemtc2netmeta({"treatments": ["A", "B"]})


def test_gemtc2netmeta_rejects_an_arm_based_table_without_variance():
    obj = {"treatments": ["A", "B"],
           "data.ab": {"study": ["s1", "s1"], "treatment": ["A", "B"],
                       "mean": [0.1, 0.3]}}
    with pytest.raises(ValueError, match="variance column"):
        ma.gemtc2netmeta(obj)


def test_crossnma2netmeta_reads_a_study_table():
    obj = {"studlab": ["s1", "s2"], "treat1": ["A", "A"],
           "treat2": ["B", "C"], "TE": [0.2, -0.3], "seTE": [0.1, 0.12]}
    res = ma.crossnma2netmeta(obj)
    assert res.trts == ["A", "B", "C"]


def test_crossnma2netmeta_accepts_alternative_column_names():
    obj = {"study_id": ["s1"], "treatment1": ["A"], "treatment2": ["B"],
           "yi": [0.2], "sei": [0.1]}
    assert ma.crossnma2netmeta(obj).trts == ["A", "B"]


def test_crossnma2netmeta_rejects_a_missing_study_column():
    with pytest.raises(ValueError, match="studlab"):
        ma.crossnma2netmeta({"treat1": ["A"], "treat2": ["B"],
                             "TE": [0.1], "seTE": [0.1]})


def test_converter_rejects_a_non_mapping():
    with pytest.raises(TypeError):
        ma.crossnma2netmeta(object())
