"""selection models (metafor::selmodel / rma.uni.selmodel).

Reference values come from R 4.6.1 / metafor 5.2-1 on the 20-study data set
below (see the scratch file ``_refgen_selmodel2.R``).

Each ``selmodel()`` fit costs a few seconds, so the fits are memoised: every
distinct call happens exactly once per test session.
"""

from __future__ import annotations

import numpy as np
import pytest

import pimeta as ma
from pimeta.selmodel import TYPE_OPTIONS

YI = [0.20, 0.35, 0.10, 0.55, 0.80, 0.05, 0.42, 0.30, 0.15, 0.65,
      0.25, 0.48, 0.12, 0.70, 0.33, 0.18, 0.58, 0.28, 0.40, 0.22]
VI = [0.008, 0.020, 0.006, 0.045, 0.090, 0.005, 0.030, 0.014, 0.009,
      0.055, 0.011, 0.026, 0.007, 0.075, 0.017, 0.010, 0.040, 0.013,
      0.024, 0.012]

#: metafor keeps the best result its optim() run reaches; we additionally do a
#: derivative-free restart, so our log-likelihood is never lower and the
#: parameters can sit slightly away from R's (better) optimum.
R_LL = {
    "beta": 18.3144458122617,
    "halfnorm": 12.7466971115375,
    "negexppow": 29.3325688590139,
    "truncest": 13.5676238719807,
    "trunc": 7.7328292372705,
    "stepcon": 12.0055255810418,
}

_CACHE = {}
_BASE = {}


def base_fit():
    if "v" not in _BASE:
        _BASE["v"] = ma.rma(YI, vi=VI, method="ML")
    return _BASE["v"]


def fit(**kwargs):
    """Memoised ``selmodel(base, **kwargs)``."""
    key = repr(sorted((k, tuple(v) if isinstance(v, list) else v)
                      for k, v in kwargs.items()))
    if key not in _CACHE:
        _CACHE[key] = ma.selmodel(base_fit(), **kwargs)
    return _CACHE[key]


# ---------------------------------------------------------------------------
# every selection-model type fits
# ---------------------------------------------------------------------------
SIMPLE_TYPES = [t for t in TYPE_OPTIONS if t not in ("stepfun", "stepcon")]


@pytest.mark.parametrize("type_", SIMPLE_TYPES)
def test_every_selection_model_type_fits(type_):
    f = fit(type=type_)
    assert f.type == type_
    assert np.all(np.isfinite(np.asarray(f.b, dtype=float)))
    assert np.all(np.isfinite(np.asarray(f.delta, dtype=float)))
    assert np.isfinite(f.ll)
    # a selection model cannot fit worse than the plain ML fit on the same data
    assert f.ll >= float(ma.logLik(base_fit())) - 1e-8


@pytest.mark.parametrize("type_", ["stepfun", "stepcon"])
def test_step_selection_models_need_steps(type_):
    with pytest.raises(ValueError, match="steps"):
        ma.selmodel(base_fit(), type=type_)


@pytest.mark.parametrize("type_", ["stepfun", "stepcon"])
def test_step_selection_models_with_steps(type_):
    f = fit(type=type_, steps=[0.05, 0.5, 1.0])
    assert f.ll == pytest.approx(12.0055255810418, abs=1e-4)
    assert float(np.asarray(f.b)[0]) == pytest.approx(0.126996008711,
                                                      abs=1e-4)
    # the first weight is fixed at one by metafor
    assert float(np.asarray(f.delta)[0]) == pytest.approx(1.0, abs=1e-12)


def test_unknown_type_is_rejected():
    with pytest.raises(ValueError, match="unknown 'type'"):
        ma.selmodel(base_fit(), type="nope")


def test_type_is_prefix_matched():
    """metafor's setchar() accepts unambiguous prefixes."""
    f = fit(type="halfn")
    assert f.type.startswith("halfn")


# ---------------------------------------------------------------------------
# the well-behaved models reproduce R's estimates
# ---------------------------------------------------------------------------
def test_beta_selection_model_matches_r():
    f = fit(type="beta")
    assert f.ll >= R_LL["beta"] - 1e-9
    # R: delta = (1.76737056476432, 21.66368533787754), b = 0.773463799485337
    assert float(np.asarray(f.delta)[0]) == pytest.approx(1.76737, abs=2e-3)
    assert float(np.asarray(f.delta)[1]) == pytest.approx(21.66369, abs=1e-1)
    assert float(np.asarray(f.b)[0]) == pytest.approx(0.773464, abs=8e-3)


def test_halfnorm_selection_model_matches_r():
    f = fit(type="halfnorm")
    # R: delta = 56.1075661127925, b = 0.128419692814956
    assert float(np.asarray(f.delta)[0]) == pytest.approx(56.1076, rel=1e-3)
    assert float(np.asarray(f.b)[0]) == pytest.approx(0.128420, abs=1e-3)


def test_halfnorm1_is_the_same_model():
    a = fit(type="halfnorm")
    b = fit(type="halfnorm1")
    assert np.asarray(a.delta) == pytest.approx(np.asarray(b.delta), abs=1e-10)
    assert np.asarray(a.b) == pytest.approx(np.asarray(b.b), abs=1e-10)


def test_halfnorm_two_sided_matches_r():
    # R: delta = 13.7859704424, b = 0.17073656056, ll = 11.5647024948224
    f = fit(type="halfnorm", alternative="two.sided")
    assert f.ll >= 11.5647024948224 - 1e-6
    assert float(np.asarray(f.b)[0]) == pytest.approx(0.17073, abs=2e-3)


def test_halfnorm_with_empirical_precision_matches_r():
    # R: delta = 99.90997475, b = 0.142222783701, ll = 14.9138979283696
    f = fit(type="halfnorm", prec="sei")
    assert f.ll >= 14.9138979283696 - 1e-6
    assert float(np.asarray(f.b)[0]) == pytest.approx(0.14222, abs=3e-3)


def test_prec_requires_sample_sizes():
    with pytest.raises(ValueError, match="sample size"):
        ma.selmodel(base_fit(), type="halfnorm", prec="ninv")


def test_negexppow_log_likelihood_beats_r():
    """metafor's optim() stalls here; our restart reaches a higher maximum."""
    f = fit(type="negexppow")
    assert f.ll >= R_LL["negexppow"] - 1e-9


def test_truncest_log_likelihood_beats_r():
    """R's truncest fit is also a stalled optim() run."""
    f = fit(type="truncest")
    assert f.ll >= R_LL["truncest"] - 1e-9
    weight, cutoff = (float(v) for v in np.asarray(f.delta))
    assert 0.0 < weight < 0.2
    assert YI[0] < cutoff < max(YI)


def test_trunc_selection_model_matches_r():
    f = fit(type="trunc")
    # R: b = 0.241021504371, ll = 7.7328292372705 (R's integrate() tolerance is
    # only 1.2e-4 relative, so the likelihood agrees to ~4e-4)
    assert float(np.asarray(f.b)[0]) == pytest.approx(0.241021504371,
                                                      abs=1e-4)
    assert f.ll == pytest.approx(7.7328292372705, abs=1e-3)


def test_trunc_with_steps_matches_r():
    f = fit(type="trunc", steps=[0.05])
    # R: delta = 0.324285750814, b = 0.240435624413
    assert float(np.asarray(f.delta)[0]) == pytest.approx(0.324285750814,
                                                          abs=1e-4)
    assert float(np.asarray(f.b)[0]) == pytest.approx(0.240435624413,
                                                      abs=1e-4)


def test_stepfun_decreasing_equals_stepcon_when_monotone():
    a = fit(type="stepfun", steps=[0.05, 0.5, 1.0])
    b = fit(type="stepcon", steps=[0.05, 0.5, 1.0])
    assert np.asarray(a.delta) == pytest.approx(np.asarray(b.delta), abs=1e-6)


def test_stepfun_monotone_flag():
    f = fit(type="stepfun", steps=[0.05, 0.5, 1.0], decreasing=True)
    assert np.isfinite(f.ll)


# ---------------------------------------------------------------------------
# alternative hypotheses
# ---------------------------------------------------------------------------
def test_alternative_greater_is_the_default():
    assert fit(type="beta").alternative == "greater"


def test_alternative_less_still_fits():
    f = fit(type="halfnorm", alternative="less")
    assert f.alternative == "less"
    assert np.isfinite(f.ll)


def test_two_sided_power_model():
    f = fit(type="power", alternative="two.sided")
    assert np.isfinite(f.ll)
    assert f.alternative == "two.sided"


def test_trunc_rejects_two_sided():
    with pytest.raises(ValueError):
        ma.selmodel(base_fit(), type="trunc", alternative="two.sided")


def test_beta_alternative_less_swaps_the_deltas():
    """metafor reorders the two beta parameters for the 'less' alternative."""
    a = fit(type="beta")
    b = fit(type="beta", alternative="less")
    assert float(np.asarray(b.delta)[0]) == pytest.approx(
        float(np.asarray(a.delta)[1]), abs=2e-2)
    assert float(np.asarray(b.b)[0]) == pytest.approx(
        float(np.asarray(a.b)[0]), abs=1e-3)


# ---------------------------------------------------------------------------
# fixed parameters, subsets, options
# ---------------------------------------------------------------------------
def test_fixed_delta_matches_r():
    # R: b = 0.243031242188, ll = 7.59465412025874
    f = fit(type="halfnorm", delta=2.0)
    assert float(np.asarray(f.delta)[0]) == pytest.approx(2.0, abs=1e-12)
    assert float(np.asarray(f.b)[0]) == pytest.approx(0.243031, abs=1e-4)
    assert f.ll == pytest.approx(7.59465412025874, abs=1e-6)


def test_fixed_delta_vector_for_truncest():
    f = fit(type="truncest", delta=[0.05, 0.48])
    assert np.asarray(f.delta) == pytest.approx([0.05, 0.48], abs=1e-12)
    assert np.isfinite(f.ll)


def test_fixed_delta_wrong_length_is_rejected():
    with pytest.raises(ValueError, match="length"):
        ma.selmodel(base_fit(), type="beta", delta=[1.0, 2.0, 3.0])


def test_subset_restricts_the_modelled_studies():
    sub = [i % 2 == 0 for i in range(len(YI))]
    f = fit(type="halfnorm", subset=sub)
    assert f.k1 == sum(sub)
    assert f.k0 == len(sub) - sum(sub)
    assert f.k == len(sub)


def test_verbose_is_accepted():
    f = fit(type="beta", verbose=True)
    assert np.isfinite(f.ll)


def test_control_tau2_max_is_honoured():
    f = fit(type="beta", control={"tau2.max": 1e-6})
    assert float(f.tau2) <= 1e-6 + 1e-12


def test_control_delta_max_is_honoured():
    f = fit(type="halfnorm", control={"delta.max": 5.0})
    assert float(np.asarray(f.delta)[0]) <= 5.0 + 1e-8


# ---------------------------------------------------------------------------
# result contents
# ---------------------------------------------------------------------------
def test_result_fields_and_dict():
    f = fit(type="beta")
    for name in ("b", "se", "zval", "pval", "ci_lb", "ci_ub", "tau2",
                 "se_tau2", "delta", "se_delta", "zval_delta",
                 "pval_delta", "ci_lb_delta", "ci_ub_delta", "QM", "QMp",
                 "ll", "ll0", "LRT", "LRTdf", "LRTp", "k", "k0", "k1"):
        assert hasattr(f, name), name
    d = f.to_dict()
    assert "delta" in d and "ll" in d


def test_selection_model_moves_the_effect_estimate_away_from_zero():
    f = fit(type="beta")
    assert float(np.asarray(f.b)[0]) > float(np.asarray(base_fit().b)[0])


def test_lrt_compares_against_the_plain_model():
    f = fit(type="beta")
    assert f.ll0 == pytest.approx(float(ma.logLik(base_fit())), abs=1e-6)
    assert f.LRT == pytest.approx(2 * (f.ll - f.ll0), abs=1e-8)
