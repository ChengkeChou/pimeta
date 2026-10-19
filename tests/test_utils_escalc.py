"""R-pinned tests for the utility helpers (metafor::bldiag & friends) and
``escalc`` across every supported effect-size measure.

Reference values were produced with metafor 5.2-1 / meta 8.5-0 on R 4.6.1.
"""
import numpy as np
import pytest

import pimeta as ma

# --------------------------------------------------------------------------- #
# metafor::bldiag / blsplit / vec2mat / replmiss / cmicalc
# --------------------------------------------------------------------------- #

def test_bldiag_matches_r():
    got = ma.bldiag([1, 2], [[1, 3], [2, 4]], 3)
    want = np.array([[1, 0, 0, 0, 0],
                     [0, 2, 0, 0, 0],
                     [0, 0, 1, 3, 0],
                     [0, 0, 2, 4, 0],
                     [0, 0, 0, 0, 3]], dtype=float)
    assert got == pytest.approx(want)


def test_bldiag_validates_the_order_argument():
    with pytest.raises(ValueError, match="square"):
        ma.bldiag(np.zeros((2, 3)), order=[1, 0])
    with pytest.raises(ValueError, match="Length of the 'order'"):
        ma.bldiag([1, 2], order=[1, 0, 2])


def test_bldiag_applies_the_order_permutation():
    """R: out[sort.vec, sort.vec] <- out, sort.vec <- order(order)."""
    # bldiag(c(1,2,3), order = c(3,1,2)) gives diag(3, 1, 2)
    perm = ma.bldiag([1.0, 2.0, 3.0], order=[2, 0, 1])
    assert np.diag(perm) == pytest.approx([3.0, 1.0, 2.0])
    identity = ma.bldiag([1.0, 2.0, 3.0], order=[0, 1, 2])
    assert np.diag(identity) == pytest.approx([1.0, 2.0, 3.0])


def test_blsplit_matches_r():
    x = np.array([[1, 2, 3], [2, 5, 6], [3, 6, 9]], dtype=float)
    parts = ma.blsplit(x, ["a", "b", "a"])
    assert list(parts) == ["a", "b"]
    assert np.asarray(parts["a"]).ravel() == pytest.approx([1, 3, 3, 9])
    assert np.asarray(parts["b"]).ravel() == pytest.approx([5])


def test_vec2mat_matches_r():
    assert ma.vec2mat([0.3, 0.4, 0.5], diag=True).ravel() == pytest.approx(
        [0.3, 0.4, 0.4, 0.5])
    assert ma.vec2mat([0.3, 0.4, 0.5]).ravel() == pytest.approx(
        [1.0, 0.3, 0.4, 0.3, 1.0, 0.5, 0.4, 0.5, 1.0])


def test_vec2mat_rejects_a_bad_length():
    with pytest.raises(ValueError, match="Length of 'x'"):
        ma.vec2mat([0.1, 0.2])


def test_replmiss_matches_r():
    assert ma.replmiss([1, np.nan, 3], [9, 9, 9]) == pytest.approx([1, 9, 3])
    # without 'y' the input is returned (NaN still in place)
    assert np.isnan(ma.replmiss([1, np.nan, 3])[1])
    with pytest.raises(ValueError, match="not the same"):
        ma.replmiss([1, np.nan, 3], [9, 9])


def test_cmicalc_matches_r():
    exact = [float("nan"), 0.56418958354775628, 0.72360125455826774,
             0.79788456080286529, 0.84074868245968926]
    approx = [float("nan"), 0.5714285714285714, 0.72727272727272729,
              0.80000000000000004, 0.84210526315789469]
    assert ma.cmicalc([1, 2, 3, 4, 5]) == pytest.approx(exact, nan_ok=True)
    assert ma.cmicalc([1, 2, 3, 4, 5], "approx") == pytest.approx(
        approx, nan_ok=True)
    with pytest.raises(ValueError, match="exact"):
        ma.cmicalc([1, 2], "wrong")


def test_contrmat_matches_r():
    data = {"g1": ["a", "a", "b"], "g2": ["b", "c", "b"]}
    got = ma.contrmat(data, "g1", "g2", append=False)
    # R: columns a, c, b (the most frequent 'g2' level is moved last)
    assert got["names"] == ["a", "c", "b"]
    assert np.asarray(got["matrix"]).ravel() == pytest.approx(
        [1, 0, -1, 1, -1, 0, 0, 0, 0])


# --------------------------------------------------------------------------- #
# meta::ci / units / JAMA labels / options
# --------------------------------------------------------------------------- #

def test_ci_matches_r():
    out = ma.ci([0.6113, -0.2], [0.27, 0.1])
    assert out["lower"] == pytest.approx([0.08210972, -0.39599640], abs=1e-7)
    assert out["upper"] == pytest.approx([1.140490276, -0.004003602], abs=1e-8)
    assert out["statistic"] == pytest.approx([2.264074, -2.0], abs=1e-6)
    assert out["p"] == pytest.approx([0.02356956, 0.04550026], abs=1e-8)
    assert out["df"] == float("inf")


def test_ci_with_t_distribution_and_null_effect_matches_r():
    out = ma.ci(0.6113, 0.27, level=0.95, df=10, null_effect=0.5)
    assert out["lower"] == pytest.approx(0.00970251, abs=1e-7)
    assert out["upper"] == pytest.approx(1.212897, abs=1e-6)
    assert out["statistic"] == pytest.approx(0.4122222, abs=1e-6)
    assert out["p"] == pytest.approx(0.6888743, abs=1e-7)
    assert out["df"] == 10.0


def test_unit_conversions():
    assert ma.units2inches([2.54, 25.4], "cm") == pytest.approx([1.0, 10.0])
    assert ma.units2inches([1, 10], "mm") == pytest.approx(
        [0.03937007874015748, 0.39370078740157483])
    assert ma.units2inches([1, 10], "px", dpi=300) == pytest.approx(
        [1 / 300, 10 / 300])
    assert ma.inches2units([1, 2], "cm") == pytest.approx([2.54, 5.08])
    with pytest.raises(ValueError, match="units must be"):
        ma.units2inches(1, "furlong")


def test_jama_labels():
    assert ma.jama_labels(["Smith JA", "Doe B"], [2001, 2003], [1, 2]) == [
        "Smith JA^1 (2001)", "Doe B^2 (2003)"]
    with pytest.raises(ValueError, match="same length"):
        ma.jama_labels(["A"], [2000, 2001], [1])


def test_setmfopt_getmfopt_roundtrip():
    before = ma.getmfopt()
    try:
        ma.setmfopt(space=True, theme="dark")
        assert ma.getmfopt("space") is True
        assert ma.getmfopt("theme") == "dark"
        assert ma.getmfopt("no_such_option", default="x") == "x"
    finally:
        ma.setmfopt(**before)
    with pytest.raises(ValueError, match="'space' must be"):
        ma.setmfopt(space="yes")
    with pytest.raises(ValueError, match="'theme' must be"):
        ma.setmfopt(theme="neon")


# --------------------------------------------------------------------------- #
# escalc — every supported measure, pinned to metafor
# --------------------------------------------------------------------------- #
ES_BIN = dict(ai=[9, 12, 7], bi=[51, 53, 48], ci=[5, 6, 4], di=[53, 56, 46])
ES_CONT = dict(m1i=[5.5, 6.1, 4.8], m2i=[4.9, 5.8, 4.4],
               sd1i=[1.2, 1.4, 1.1], sd2i=[1.3, 1.5, 1.0],
               n1i=[30, 40, 25], n2i=[28, 38, 27])
ES_RATE = dict(x1i=[12, 8, 15], x2i=[7, 5, 9],
               t1i=[120, 90, 150], t2i=[110, 85, 140])
ES_PROP = dict(xi=[12, 8, 15], ni=[120, 90, 150])

ESCALC_BINARY = {
    "RR": ([0.55388511322643774, 0.64589429570940005, 0.46430560813109789],
           [0.27720306513409965, 0.21848635235732011, 0.35467532467532464]),
    "OR": ([0.62625294572991508, 0.74820695774297263, 0.51705617351662658],
           [0.34958687877666789, 0.28672506738544473, 0.43542960662525876]),
    "RD": ([0.063793103448275851, 0.087841191066997532, 0.047272727272727258],
           [0.0034831942679076633, 0.0037257070563829547,
            0.0034915341848234414]),
    "AS": ([0.099698195498314934, 0.1277910993284192, 0.0780333550764436],
           [0.0084770114942528743, 0.0078784119106699753,
            0.0095454545454545445]),
    "PETO": ([0.60489532019704439, 0.71642953081248439, 0.49924388957270982],
             [0.32152093596059111, 0.25702415370956355,
              0.40323544927026561]),
}

ESCALC_CONT = {
    "MD": ([0.59999999999999964, 0.29999999999999982, 0.39999999999999947],
           [0.10835714285714287, 0.10821052631578947, 0.085437037037037034]),
    "SMD": ([0.47383568894629557, 0.20491106974779177, 0.37549415775895389],
            [0.070983138531405351, 0.051584946823076419,
             0.078392766484259219]),
    "SMDH": ([0.47315937949117853, 0.20472526163232613, 0.37477986957079007],
             [0.073749478519402334, 0.053041765876900973,
              0.081863012772626073]),
}

ESCALC_PROP = {
    "PLN": ([-2.3025850929940455, -2.4203681286504293, -2.3025850929940455],
            [0.074999999999999997, 0.11388888888888889, 0.059999999999999998]),
    "PLO": ([-2.1972245773362191, -2.3272777055844172, -2.1972245773362191],
            [0.092592592592592587, 0.13719512195121952, 0.07407407407407407]),
    "PAS": ([0.32175055439664219, 0.30274595162300438, 0.32175055439664219],
            [0.0020833333333333333, 0.0027777777777777779,
             0.0016666666666666668]),
    "PFT": ([0.32716031781657939, 0.31047001543750796, 0.32610095917219323],
            [0.0020746887966804979, 0.0027624309392265192,
             0.0016611295681063123]),
    "PR": ([0.10000000000000001, 0.088888888888888892, 0.10000000000000001],
           [0.00075000000000000012, 0.0008998628257887518,
            0.00060000000000000006]),
}


@pytest.mark.parametrize("measure", sorted(ESCALC_BINARY))
def test_escalc_binary_measures_match_r(measure):
    want_yi, want_vi = ESCALC_BINARY[measure]
    r = ma.escalc(measure, **ES_BIN)
    assert list(r.yi) == pytest.approx(want_yi, abs=1e-12)
    assert list(r.vi) == pytest.approx(want_vi, abs=1e-12)


@pytest.mark.parametrize("measure", sorted(ESCALC_CONT))
def test_escalc_continuous_measures_match_r(measure):
    want_yi, want_vi = ESCALC_CONT[measure]
    r = ma.escalc(measure, **ES_CONT)
    assert list(r.yi) == pytest.approx(want_yi, abs=1e-12)
    assert list(r.vi) == pytest.approx(want_vi, abs=1e-12)


@pytest.mark.parametrize("measure", sorted(ESCALC_PROP))
def test_escalc_proportion_measures_match_r(measure):
    want_yi, want_vi = ESCALC_PROP[measure]
    r = ma.escalc(measure, **ES_PROP)
    assert list(r.yi) == pytest.approx(want_yi, abs=1e-12)
    assert list(r.vi) == pytest.approx(want_vi, abs=1e-12)


def test_escalc_irr_matches_r():
    r = ma.escalc("IRR", **ES_RATE)
    assert list(r.yi) == pytest.approx(
        [0.45198512374305766, 0.41284521540578689, 0.44183275227903973],
        abs=1e-12)
    assert list(r.vi) == pytest.approx(
        [0.22619047619047616, 0.32500000000000001, 0.17777777777777776],
        abs=1e-12)


def test_escalc_zcor_matches_r():
    r = ma.escalc("ZCOR", ri=[0.5, -0.2, 0.7], ni=[100, 80, 120])
    assert list(r.yi) == pytest.approx(
        [0.54930614433405489, -0.20273255405408219, 0.86730052769405319],
        abs=1e-12)
    assert list(r.vi) == pytest.approx(
        [0.010309278350515464, 0.012987012987012988, 0.0085470085470085479],
        abs=1e-12)


def test_escalc_derives_the_2x2_table_from_events_and_sizes():
    from_sizes = ma.escalc("OR", ai=[9, 12, 7], ci=[5, 6, 4],
                           n1i=[60, 65, 55], n2i=[58, 62, 50])
    from_cells = ma.escalc("OR", **ES_BIN)
    assert list(from_sizes.yi) == pytest.approx(list(from_cells.yi), abs=1e-12)
    assert list(from_sizes.vi) == pytest.approx(list(from_cells.vi), abs=1e-12)


def test_escalc_add_zero_matches_r_for_nonzero_cells():
    """Without zero cells the continuity increment must not matter."""
    with_add = ma.escalc("OR", add=0.5, **ES_BIN)
    no_add = ma.escalc("OR", add=0, **ES_BIN)
    assert list(no_add.yi) == pytest.approx(list(with_add.yi), abs=1e-12)
    assert list(no_add.vi) == pytest.approx(list(with_add.vi), abs=1e-12)


def test_escalc_drop00_removes_double_zero_studies():
    a = [0, 9]
    b = [60, 51]
    c = [0, 5]
    d = [58, 53]
    kept = ma.escalc("RR", ai=a, bi=b, ci=c, di=d, drop00=True)
    assert kept.k == 1
    allrows = ma.escalc("RR", ai=a, bi=b, ci=c, di=d, drop00=False)
    assert allrows.k == 2


def test_escalc_rejects_unknown_and_incomplete_measures():
    with pytest.raises(ValueError, match="unsupported measure"):
        ma.escalc("XYZ")
    with pytest.raises(ValueError, match="ai/bi/ci/di"):
        ma.escalc("OR", ai=[1, 2])
    with pytest.raises(ValueError, match="m1i"):
        ma.escalc("MD", m1i=[1])
    with pytest.raises(ValueError, match="IRR"):
        ma.escalc("IRR", x1i=[1])
    with pytest.raises(ValueError, match="proportion"):
        ma.escalc("PLN", xi=[1])
    with pytest.raises(ValueError, match="ZCOR"):
        ma.escalc("ZCOR", ri=[0.1])
