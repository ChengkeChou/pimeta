"""IPD plus aggregate data in the two-stage model.

An IPD meta-analysis that can only take the trials that shared their data is
answering a question about those trials.  Adding the ones that reported only an
effect is not a new estimator — it is the same second stage with two more rows —
so the whole of this file is about the contract rather than the algebra: the
two kinds of estimate have to be on the same scale, they cannot be the same
study twice, and a covariate-adjusted IPD estimate is not the same estimand as
an unadjusted aggregate one.
"""

import numpy as np
import pytest

import pimeta as ma


def _ipd(seed=3, k=6, n=200, effect=0.4):
    rng = np.random.default_rng(seed)
    study = np.repeat(np.arange(1, k + 1), n)
    treat = np.tile(np.r_[np.zeros(n // 2), np.ones(n - n // 2)], k)
    u = rng.normal(0.0, 0.3, k)
    eta = -0.3 + (effect + u[study - 1]) * treat
    y = (rng.random(eta.size) < 1.0 / (1.0 + np.exp(-eta))).astype(float)
    return y, study, treat


@pytest.fixture(scope="module")
def parts():
    return _ipd()


AGG = {"study": ["a1", "a2"], "estimate": [0.42, 0.31],
       "se": [0.15, 0.18]}


def test_the_mixed_pool_is_the_plain_pool_of_the_two_vectors(parts):
    y, study, treat = parts
    ipd = ma.ipd_two_stage(y, study=study, treat=treat)
    mixed = ma.ipd_two_stage(y, study=study, treat=treat, aggregate=AGG)
    est = np.array([s.estimate for s in ipd.per_study] + AGG["estimate"])
    var = np.array([s.variance for s in ipd.per_study]
                   + [se ** 2 for se in AGG["se"]])
    manual = ma.rma(est, vi=var, method="REML")
    assert mixed.b == pytest.approx(float(manual.b[0]), rel=1e-12)
    assert mixed.se == pytest.approx(float(manual.se[0]), rel=1e-12)
    assert mixed.tau2 == pytest.approx(float(manual.tau2), rel=1e-12)
    assert mixed.k == ipd.k + 2


def test_the_contributors_are_counted_and_labelled(parts):
    y, study, treat = parts
    mixed = ma.ipd_two_stage(y, study=study, treat=treat, aggregate=AGG)
    assert mixed.n_ipd == 6
    assert mixed.n_aggregate == 2
    assert mixed.k == 8
    rows = mixed.per_study_table()
    assert [row["source"] for row in rows] == ["ipd"] * 6 + ["aggregate"] * 2 \
        + ["pooled"]
    record = mixed.to_dict()
    assert record["n_ipd"] == 6
    assert record["n_aggregate"] == 2
    assert record["per_study"][-1]["source"] == "aggregate"


def test_a_single_record_and_a_list_of_records_agree(parts):
    y, study, treat = parts
    one = ma.ipd_two_stage(y, study=study, treat=treat,
                           aggregate={"study": "a1",
                                      "estimate": 0.42, "se": 0.15})
    listed = ma.ipd_two_stage(y, study=study, treat=treat,
                              aggregate=[{"study": "a1",
                                          "estimate": 0.42, "se": 0.15}])
    assert one.k == listed.k == 7
    assert one.b == pytest.approx(listed.b, rel=1e-12)


def test_adding_aggregate_studies_narrows_the_estimate(parts):
    """Not a property of the method, a property of this fixture: it should."""
    y, study, treat = parts
    ipd = ma.ipd_two_stage(y, study=study, treat=treat)
    mixed = ma.ipd_two_stage(y, study=study, treat=treat, aggregate=AGG)
    assert mixed.se < ipd.se
    assert mixed.b > ipd.b          # the aggregate studies sit above the IPD pool


def test_the_same_study_cannot_enter_twice(parts):
    y, study, treat = parts
    with pytest.raises(ValueError, match="cannot enter twice"):
        ma.ipd_two_stage(y, study=study, treat=treat,
                         aggregate={"study": ["1"], "estimate": [0.4],
                                    "se": [0.1]})


def test_a_missing_or_impossible_variance_is_refused(parts):
    y, study, treat = parts
    with pytest.raises(ValueError, match="positive standard error"):
        ma.ipd_two_stage(y, study=study, treat=treat,
                         aggregate={"study": ["a1"], "estimate": [0.4],
                                    "se": [0.0]})
    with pytest.raises(ValueError, match="needs a 'se' entry"):
        ma.ipd_two_stage(y, study=study, treat=treat,
                         aggregate={"study": ["a1"], "estimate": [0.4]})
    with pytest.raises(ValueError, match="different lengths"):
        ma.ipd_two_stage(y, study=study, treat=treat,
                         aggregate={"study": ["a1", "a2"],
                                    "estimate": [0.4, 0.5], "se": [0.1]})


def test_an_adjusted_ipd_analysis_says_the_estimands_may_differ(parts):
    """The caveat that matters, emitted rather than buried in a docstring."""
    y, study, treat = parts
    rng = np.random.default_rng(11)
    age = rng.normal(60.0, 10.0, y.size)
    with pytest.warns(RuntimeWarning, match="marginal and a conditional"):
        ma.ipd_two_stage(y, study=study, treat=treat,
                         covariates={"age": age}, aggregate=AGG)


def test_an_unadjusted_ipd_analysis_does_not_warn(parts):
    import warnings as _warnings
    y, study, treat = parts
    with _warnings.catch_warnings():
        _warnings.simplefilter("error")
        ma.ipd_two_stage(y, study=study, treat=treat, aggregate=AGG)
