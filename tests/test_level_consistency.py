"""One confidence-level convention, checked across every entry point.

``pimeta`` grew up with three spellings of "how wide should this interval be" —
``0.95`` (a confidence level), ``95`` (the same in percent) and ``0.05`` (the
alpha).  Every entry point now normalises through :func:`pimeta.levels.level_to_alpha`,
so all three spellings have to give the *same* interval.  That is what this
file pins.

It exists because the alternative — remembering which function wants which —
already failed once: ``rma_peto(level=0.95)`` returned an interval 31 times too
narrow (a 5% interval) and ``rma_peto(level=95)`` returned ``nan``, both
silently.
"""

import numpy as np
import pytest
from scipy import stats

import pimeta as ma
from pimeta.levels import level_to_alpha, level_to_percent
from pimeta.rma_peto_mh import mh_confint

Z_975 = stats.norm.ppf(0.975)

# a small, well-behaved 2x2 set
A = [3, 5, 2, 7, 4]
B = [20, 18, 25, 15, 22]
C = [8, 6, 9, 10, 7]
D = [19, 20, 16, 18, 21]

YI = np.array([0.20, 0.40, 0.10, 0.50, 0.30, 0.15, 0.45, 0.25])
VI = np.array([0.04, 0.05, 0.03, 0.06, 0.02, 0.05, 0.04, 0.03])

SPELLINGS = (0.05, 0.95, 95)


# --------------------------------------------------------------------------- #
# the normaliser itself
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("value", SPELLINGS)
def test_every_spelling_maps_to_the_same_alpha(value):
    assert level_to_alpha(value) == pytest.approx(0.05, abs=1e-12)


def test_the_normaliser_is_idempotent_on_an_alpha():
    """That property is what makes it safe to apply at several layers."""
    once = level_to_alpha(0.95)
    assert level_to_alpha(once) == pytest.approx(once, rel=1e-12)


def test_percent_and_fraction_agree():
    assert level_to_percent(95) == pytest.approx(95.0, rel=1e-12)
    assert level_to_percent(0.95) == pytest.approx(95.0, rel=1e-12)
    assert level_to_percent(0.05) == pytest.approx(95.0, rel=1e-12)


def test_an_impossible_level_is_rejected():
    with pytest.raises(ValueError, match="level"):
        level_to_alpha(0.0)
    with pytest.raises(ValueError, match="level"):
        level_to_alpha(120)


# --------------------------------------------------------------------------- #
# the fixed-effect binary methods (the ones that were broken)
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("value", SPELLINGS)
def test_rma_peto_accepts_every_spelling(value):
    base = ma.rma_peto(A, B, C, D)
    got = ma.rma_peto(A, B, C, D, level=value)
    assert got.ci_lb == pytest.approx(base.ci_lb, rel=1e-12)
    assert got.ci_ub == pytest.approx(base.ci_ub, rel=1e-12)
    # the interval it reports really is the 95% one
    assert (got.ci_ub - got.ci_lb) / (2.0 * got.se) == pytest.approx(
        Z_975, rel=1e-10)


@pytest.mark.parametrize("value", SPELLINGS)
def test_rma_mh_accepts_every_spelling(value):
    base = ma.rma_mh(A, B, C, D)
    got = ma.rma_mh(A, B, C, D, level=value)
    assert got.ci_lb == pytest.approx(base.ci_lb, rel=1e-12)
    assert got.ci_ub == pytest.approx(base.ci_ub, rel=1e-12)
    assert (got.ci_ub - got.ci_lb) / (2.0 * got.se) == pytest.approx(
        Z_975, rel=1e-10)


def test_a_narrower_level_gives_a_narrower_interval():
    wide = ma.rma_peto(A, B, C, D, level=95)
    narrow = ma.rma_peto(A, B, C, D, level=90)
    assert (narrow.ci_ub - narrow.ci_lb) < (wide.ci_ub - wide.ci_lb)


def test_mh_confint_accepts_every_spelling():
    fit = ma.rma_peto(A, B, C, D)
    base = mh_confint(fit)["ci.lb"]
    for value in SPELLINGS:
        assert mh_confint(fit, level=value)["ci.lb"] == pytest.approx(
            base, rel=1e-12)


# --------------------------------------------------------------------------- #
# the pooling entry points that historically took alpha
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("value", SPELLINGS)
def test_confint_tau2_accepts_every_spelling(value):
    base = ma.confint_tau2(YI, VI, 0.02, level=0.05)
    got = ma.confint_tau2(YI, VI, 0.02, level=value)
    assert got.tau2_lb == pytest.approx(base.tau2_lb, rel=1e-10)
    assert got.tau2_ub == pytest.approx(base.tau2_ub, rel=1e-10)


@pytest.mark.parametrize("value", SPELLINGS)
def test_predict_rma_accepts_every_spelling(value):
    base = ma.predict_rma(0.3, 0.08, 0.02, k=8, level=0.05)
    got = ma.predict_rma(0.3, 0.08, 0.02, k=8, level=value)
    assert got.ci_lb == pytest.approx(base.ci_lb, rel=1e-10)
    assert got.pi_lb == pytest.approx(base.pi_lb, rel=1e-10)


@pytest.mark.parametrize("value", SPELLINGS)
def test_hc_rma_accepts_every_spelling(value):
    base = ma.hc_rma(YI, VI, level=0.05)
    got = ma.hc_rma(YI, VI, level=value)
    assert got.ci_lb == pytest.approx(base.ci_lb, rel=1e-10)
    assert got.ci_ub == pytest.approx(base.ci_ub, rel=1e-10)


# --------------------------------------------------------------------------- #
# the rest of the package speaks the same language by default
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("fit,se,lo,hi", [("rma", "se", "ci_lb", "ci_ub")])
def test_the_ordinary_fit_reports_a_95_percent_interval(fit, se, lo, hi):
    m = ma.rma(YI, vi=VI)
    assert (getattr(m, hi)[0] - getattr(m, lo)[0]) / (2.0 * m.se[0]) == \
        pytest.approx(Z_975, rel=1e-10)


def test_rma_mv_reports_a_95_percent_interval():
    m = ma.rma_mv(YI, vi=VI, random="~ 1 | s",
                  data={"s": [1, 1, 2, 2, 3, 3, 4, 4]})
    assert (m.ci_ub[0] - m.ci_lb[0]) / (2.0 * m.se[0]) == pytest.approx(
        Z_975, rel=1e-8)


def test_a_knha_fit_uses_the_t_quantile_not_the_z_one():
    m = ma.rma(YI, vi=VI, test="knha")
    ratio = (m.ci_ub[0] - m.ci_lb[0]) / (2.0 * m.se[0])
    assert ratio > Z_975          # the t critical value is wider
    assert ratio == pytest.approx(float(stats.t.ppf(0.975, m.k - m.p)),
                                  rel=1e-10)
