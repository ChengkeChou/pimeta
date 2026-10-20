"""Extra branch coverage for ``publication_bias`` (regtest / egger_test / ...).

The regression machinery is already pinned against R in
``test_metabias.py``; these tests only exercise the alternate entry points and
validation branches that the ``metabias`` path does not reach.
"""

import math

import numpy as np
import pytest

from pimeta import egger_test, regtest, trimfill
from pimeta.publication_bias import (
    _bias_method,
    _kentau,
    _linregcore,
    _predictor_values,
)

TE = [0.1, -0.2, 0.05, -0.15, 0.3, 0.25, 0.12, -0.08, 0.2, 0.02]
SE = [0.10, 0.12, 0.11, 0.13, 0.14, 0.15, 0.09, 0.10, 0.12, 0.11]
V = [s ** 2 for s in SE]


# --------------------------------------------------------------------------- #
# regtest(model="lm") must agree with meta's linregcore
# --------------------------------------------------------------------------- #

def test_regtest_lm_matches_metas_linregcore():
    lm = regtest(TE, V, predictor="sei", model="lm")
    core = _linregcore(TE, SE, None, "lm")
    assert lm.slope == pytest.approx(core["slope"], rel=0, abs=1e-12)
    assert lm.slope_se == pytest.approx(core["se.slope"], rel=0, abs=1e-12)
    assert lm.intercept == pytest.approx(core["intercept"], rel=0, abs=1e-12)
    assert lm.intercept_se == pytest.approx(core["se.intercept"], rel=0,
                                            abs=1e-12)
    assert lm.tau == pytest.approx(core["tau"], rel=0, abs=1e-12)
    assert lm.df == 8
    assert lm.stat == pytest.approx(core["statistic"], rel=0, abs=1e-12)


def test_regtest_lm_matches_r():
    # R: regtest(rma.uni(yi, sei=seTE, method="REML", test="t"),
    #            predictor="sei", model="lm")
    # -> t = 0.51777846236912, p = 0.61862277663171, limit estimate -0.119043...
    res = regtest(TE, V, predictor="sei", model="lm")
    assert res.stat == pytest.approx(0.51777846236912, rel=0, abs=1e-9)
    assert res.p == pytest.approx(0.61862277663171, rel=0, abs=1e-12)
    assert res.intercept == pytest.approx(-0.11904347867467, rel=0, abs=1e-9)
    assert res.slope == pytest.approx(1.5237087200236, rel=0, abs=1e-9)
    assert res.model == "lm"
    assert res.method == "lm"


def test_regtest_rejects_unknown_model():
    with pytest.raises(ValueError, match="unknown model"):
        regtest(TE, V, model="ols")


def test_regtest_rejects_unknown_predictor():
    with pytest.raises(ValueError, match="unknown predictor"):
        regtest(TE, V, predictor="nonsense")


def test_regtest_ni_predictor_has_no_limit_estimate():
    res = regtest(TE, V, ni=[10, 20, 30, 40, 50, 60, 70, 80, 90, 100],
                  predictor="ni")
    assert res.intercept is None
    assert res.intercept_ci_lower is None
    assert res.slope is not None


# --------------------------------------------------------------------------- #
# egger_test: both models
# --------------------------------------------------------------------------- #

def test_egger_test_lm_branch():
    res = egger_test(TE, V, model="lm")
    assert res.model == "lm"
    assert res.k == 10
    assert res.stat == pytest.approx(0.51777846236912, rel=0, abs=1e-9)
    assert res.intercept == pytest.approx(-0.11904347867467, rel=0, abs=1e-9)
    assert res.slope == pytest.approx(1.5237087200236, rel=0, abs=1e-9)


def test_egger_test_rma_branch():
    res = egger_test(TE, V)
    assert res.model == "rma"
    assert res.slope == pytest.approx(2.1301704039775, rel=0, abs=1e-9)


def test_egger_test_rejects_unknown_model():
    with pytest.raises(ValueError, match="unknown model"):
        egger_test(TE, V, model="ols")


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #

def test_predictor_values_all_variants():
    ni = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
    v = np.asarray(V)
    assert _predictor_values("sei", v, None)[0] == pytest.approx(SE[0])
    assert _predictor_values("vi", v, None)[0] == pytest.approx(V[0])
    assert _predictor_values("ni", v, ni)[0] == pytest.approx(10)
    assert _predictor_values("ninv", v, ni)[0] == pytest.approx(0.1)
    assert _predictor_values("sqrtni", v, ni)[0] == pytest.approx(math.sqrt(10))
    assert _predictor_values("sqrtninv", v, ni)[0] == pytest.approx(
        1 / math.sqrt(10))


def test_predictor_values_need_sample_sizes():
    with pytest.raises(ValueError, match="sample sizes"):
        _predictor_values("ni", np.asarray(V), None)


def test_linregcore_rejects_unknown_model():
    with pytest.raises(ValueError, match="model must be"):
        _linregcore(TE, SE, None, "ols")


def test_bias_method_none_returns_none():
    assert _bias_method(None) is None


def test_kentau_validations():
    with pytest.raises(ValueError, match="must be equal"):
        _kentau([1, 2, 3], [1, 2])
    with pytest.raises(ValueError, match="at least three"):
        _kentau([1, 2], [1, 2])


def test_kentau_continuity_correction_follows_the_sign_of_ks():
    """``ks - sign(ks)``: subtracts one for positive ks, adds one for negative."""
    x = [1.0, 2.0, 3.0, 4.0]
    y = [1.0, 2.0, 3.0, 4.0]
    ks0 = _kentau(x, y, correct=False)[0]
    assert ks0 > 0
    assert _kentau(x, y, correct=True)[0] == pytest.approx(ks0 - 1.0)
    x2 = [4.0, 3.0, 2.0, 1.0]
    assert _kentau(x2, y, correct=False)[0] == pytest.approx(-ks0)
    assert _kentau(x2, y, correct=True)[0] == pytest.approx(-ks0 + 1.0)


def test_trimfill_rejects_unknown_estimator():
    with pytest.raises(ValueError, match="unknown estimator"):
        trimfill(TE, V, estimator="X9")
