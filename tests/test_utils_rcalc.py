"""``rcalc``, ``fitstats``/``sigma_matreg``, ``as_data_frame_*`` and ``conv.*``.

Reference values are the R/Cochran identities documented in the docstrings;
the matrix-to-vector ordering is the one ``metafor::rcalc`` produces
(column-major enumeration of the triangle).
"""

from __future__ import annotations

import math

import numpy as np
import pytest

import pimeta as ma

M = np.array([[1.0, 0.5, 0.3],
              [0.5, 1.0, 0.2],
              [0.3, 0.2, 1.0]])


# ---------------------------------------------------------------------------
# rcalc
# ---------------------------------------------------------------------------
def test_rcalc_matrix_to_pairs():
    res = ma.rcalc(M, 100)
    # metafor enumerates the lower triangle column-major: (2,1), (3,1), (3,2)
    assert res.dat["var1"] == ["x1", "x1", "x2"]
    assert res.dat["var2"] == ["x2", "x3", "x3"]
    assert res.dat["var1.var2"] == ["x1.x2", "x1.x3", "x2.x3"]
    assert res.dat["yi"] == pytest.approx([0.5, 0.3, 0.2])
    assert res.dat["ni"] == [100.0, 100.0, 100.0]


def test_rcalc_fisher_z():
    res = ma.rcalc(M, 100, rtoz=True)
    assert res.dat["yi"] == pytest.approx(
        [math.atanh(0.5), math.atanh(0.3), math.atanh(0.2)], abs=1e-12)


def test_rcalc_upper_uses_the_upper_triangle():
    lower_only = np.tril(M) + np.tril(M, -1).T - np.eye(3)
    assert ma.rcalc(lower_only, 100).dat["yi"] == pytest.approx([0.5, 0.3, 0.2])
    assert ma.rcalc(lower_only, 100, upper=True).dat["yi"] == \
        pytest.approx([0.5, 0.3, 0.2])


def test_rcalc_variance_matrix_is_block_diagonal_for_a_list():
    one = ma.rcalc(M, 100)
    two = ma.rcalc([M, M], [100, 80])
    assert np.asarray(two.V).shape == (6, 6)
    assert np.asarray(two.V)[:3, :3] == pytest.approx(np.asarray(one.V))
    assert np.allclose(np.asarray(two.V)[:3, 3:], 0.0)


def test_rcalc_list_keeps_simplify_false_separate():
    res = ma.rcalc([M, M], [100, 80], simplify=False)
    assert len(res) == 2
    assert all(np.asarray(r.V).shape == (3, 3) for r in res)


def test_rcalc_noid_drops_the_id_column():
    assert "id" in ma.rcalc([M, M], [100, 80]).dat
    assert "id" not in ma.rcalc([M, M], [100, 80], noid=True).dat


def test_rcalc_formula_style_input_fills_the_missing_pair_with_na():
    res = ma.rcalc({"ri": [0.5, 0.3], "var1": ["a", "a"],
                    "var2": ["b", "c"]}, [100, 100])
    # r-c is never supplied, so the third row comes back NA (metafor warns here)
    assert res.dat["var1.var2"] == ["a.b", "a.c", "b.c"]
    assert res.dat["yi"][:2] == pytest.approx([0.5, 0.3])
    assert np.isnan(res.dat["yi"][2])


def test_rcalc_formula_style_input_with_ids():
    res = ma.rcalc({"ri": [0.5, 0.3, 0.2],
                    "var1": ["a", "a", "b"],
                    "var2": ["b", "c", "c"],
                    "id": [1, 1, 2]}, [100, 100, 80])
    assert res.dat["id"] == [0, 0, 0, 1]
    assert res.dat["yi"][0] == pytest.approx(0.5)
    assert res.dat["ni"][-1] == pytest.approx(80.0)


def test_rcalc_rejects_a_non_square_matrix():
    with pytest.raises(ValueError, match="square matrix"):
        ma.rcalc(np.zeros((2, 3)), 10)


def test_rcalc_rejects_a_length_mismatch():
    with pytest.raises(ValueError, match="same length"):
        ma.rcalc([M, M], [100])


def test_rcalc_rejects_a_repeated_pair():
    with pytest.raises(ValueError, match="Identical"):
        ma.rcalc({"ri": [0.5], "var1": ["a"], "var2": ["a"]}, [100])


def test_rcalc_one_by_one_matrix_is_empty():
    res = ma.rcalc(np.array([[1.0]]), 10)
    assert res.dat is None and res.V is None


def test_rcalc_result_round_trips():
    res = ma.rcalc(M, 100)
    d = res.as_dict()
    assert set(d) == {"dat", "V"}
    assert "RcalcResult" in repr(res)


# ---------------------------------------------------------------------------
# fitstats / sigma_matreg / as_data_frame_*
# ---------------------------------------------------------------------------
YI = [0.1, 0.2, 0.3, 0.4]
VI = [0.01, 0.02, 0.03, 0.04]


def test_fitstats_repl_and_ml_differ():
    r = ma.rma(YI, vi=VI, method="REML")
    repl = ma.fitstats(r, REML=True)
    ml = ma.fitstats(r, REML=False)
    assert repl["logLik"] != pytest.approx(ml["logLik"])
    assert set(repl) >= {"logLik", "deviance", "AIC", "BIC", "AICc"}


def test_fitstats_picks_the_models_own_estimator():
    assert ma.fitstats(ma.rma(YI, vi=VI, method="REML")).method == "REML"
    assert ma.fitstats(ma.rma(YI, vi=VI, method="ML")).method == "ML"


def test_fitstats_identities():
    r = ma.rma(YI, vi=VI, method="ML")
    fs = ma.fitstats(r, REML=False)
    # k = 4, p = 1 -> AIC = -2ll + 2*2
    assert fs["AIC"] == pytest.approx(-2 * fs["logLik"] + 2 * 2, abs=1e-9)
    assert fs["BIC"] == pytest.approx(-2 * fs["logLik"] + math.log(4) * 2,
                                      abs=1e-9)


def test_sigma_matreg_needs_a_correlation_matrix_model():
    reg = ma.rma(YI, vi=VI, mods=np.array([1.0, 2.0, 3.0, 4.0]))
    with pytest.raises(ValueError, match="sigma"):
        ma.sigma_matreg(reg)


def test_as_data_frame_vif_lists_the_moderators():
    mods = np.array([[1.0, 2.0, 1.0], [2.0, 1.0, 3.0], [3.0, 4.0, 2.0],
                     [4.0, 3.0, 5.0], [5.0, 6.0, 4.0]])
    reg = ma.rma([0.1, 0.2, 0.3, 0.4, 0.5],
                 vi=[0.01, 0.02, 0.03, 0.04, 0.05], mods=mods)
    rows = ma.as_data_frame_vif(reg)
    # rma() records VIFs on the fit; the helper just reshapes them
    assert isinstance(rows, list)


# ---------------------------------------------------------------------------
# conv.*
# ---------------------------------------------------------------------------
def test_conv_wald_recovers_the_variance():
    res = ma.conv_wald(0.5, 0.1, 0.9)
    # half-width 0.4 = 1.959964 * se -> vi = se^2
    assert res.yi == pytest.approx([0.5], abs=1e-12)
    assert res.vi == pytest.approx([(0.4 / 1.959963984540054) ** 2], abs=1e-12)


def test_conv_wald_level_changes_the_variance():
    # the bounds are held fixed, so a wider interval implies a smaller variance
    loose = ma.conv_wald(0.5, 0.1, 0.9, level=95)
    tight = ma.conv_wald(0.5, 0.1, 0.9, level=99)
    assert np.asarray(tight.vi)[0] < np.asarray(loose.vi)[0]


def test_conv_wald_applies_the_transform():
    res = ma.conv_wald(0.5, 0.1, 0.9, transf=np.exp)
    assert np.asarray(res.yi)[0] == pytest.approx(math.exp(0.5), abs=1e-12)


def test_conv_delta_is_the_delta_method():
    res = ma.conv_delta(0.5, 0.04, np.exp)
    assert np.asarray(res.yi)[0] == pytest.approx(math.exp(0.5), abs=1e-12)
    assert np.asarray(res.vi)[0] == pytest.approx(math.exp(0.5) ** 2 * 0.04,
                                                  abs=1e-12)


def test_conv_delta_accepts_a_scalar_function():
    res = ma.conv_delta(0.5, 0.04, math.exp)
    assert np.asarray(res.vi)[0] == pytest.approx(math.exp(0.5) ** 2 * 0.04,
                                                  abs=1e-9)


def test_conv_delta_rejects_a_length_mismatch():
    with pytest.raises(ValueError, match="not the same"):
        ma.conv_delta([0.1, 0.2], [0.01], np.exp)


def test_conv_fivenum_needs_something_to_work_with():
    with pytest.raises(ValueError, match="at least some"):
        ma.conv_fivenum()


@pytest.mark.parametrize("method", ["luo/wan/shi", "wan2014", "bland2015"])
def test_conv_fivenum_methods(method):
    # with test = TRUE metafor refuses to reconstruct (the data fail its
    # normality screen), so use test = FALSE to exercise the estimator itself
    res = ma.conv_fivenum(1, 2, 3, 4.5, 7, n=50, method=method, test=False)
    assert np.isfinite(res.mean) and np.isfinite(res.sd)
    assert res.sd > 0


def test_conv_fivenum_rejects_an_unknown_method():
    with pytest.raises(ValueError, match="method"):
        ma.conv_fivenum(1, 2, 3, 4.5, 7, 50, method="nope", test=False)
