"""Parallel execution must not change the answer.

The claim being tested is not "it is faster" -- that depends on the machine --
but "``n_jobs=2`` gives the same numbers as ``n_jobs=1``, to the last bit".  A
speed-up that changes the third decimal is a different analysis with a nicer
wall-clock, so equivalence is the property worth pinning.
"""

import functools
import json

import numpy as np
import pytest

from pimeta.parallel import check_picklable, map_ordered, resolve_jobs


def _square(x):
    """Module-level so it can be pickled to a worker process."""
    return x * x


def _noisy_square(x):
    """Same value, different cost per item, to shuffle completion order."""
    import time

    time.sleep(0.01 * (5 - x % 5))
    return x * x


def test_resolve_jobs_counts():
    assert resolve_jobs(None, 10) == 1
    assert resolve_jobs(0, 10) == 1
    assert resolve_jobs(1, 10) == 1
    assert resolve_jobs(4, 10) == 4
    # never more workers than items: the extra ones would only start up
    assert resolve_jobs(8, 3) == 3
    assert resolve_jobs(4, 1) == 1
    assert resolve_jobs(-1, 3) == 3
    assert resolve_jobs(-1, 0) == 1


def test_the_serial_path_is_the_list_comprehension():
    items = list(range(20))
    assert map_ordered(_square, items) == [x * x for x in items]
    assert map_ordered(_square, items, n_jobs=1) == [x * x for x in items]


def test_the_parallel_path_agrees_with_the_serial_path():
    items = list(range(24))
    serial = map_ordered(_square, items, n_jobs=1)
    for jobs in (2, 3):
        assert map_ordered(_square, items, n_jobs=jobs) == serial


def test_the_order_is_the_input_order_not_the_completion_order():
    """The point of going through ``Executor.map`` rather than ``as_completed``.

    ``_noisy_square`` makes low indices slow, so a collection in completion
    order would come back nearly reversed.
    """
    items = list(range(20))
    serial = map_ordered(_square, items, n_jobs=1)
    assert map_ordered(_noisy_square, items, n_jobs=4) == serial


def test_a_picklable_partial_is_accepted():
    """A ``partial`` of a module-level function is the supported idiom."""
    fn = functools.partial(_square)
    assert map_ordered(fn, [1, 2, 3], n_jobs=2) == [1, 4, 9]


def test_a_closure_is_refused_before_any_work_starts():
    scale = 3

    def _closure(x):  # noqa: ANN001 - deliberately unpicklable
        return x * scale

    with pytest.raises(ValueError, match="n_jobs=1"):
        map_ordered(_closure, [1, 2, 3], n_jobs=2)
    # and refused at the check itself, without going through map_ordered
    with pytest.raises(ValueError, match="pickle"):
        check_picklable(_closure)
    # the serial path never needs the check
    assert map_ordered(_closure, [1, 2, 3]) == [3, 6, 9]


def test_an_empty_sequence_is_not_an_error():
    assert map_ordered(_square, [], n_jobs=4) == []


def test_identical_floating_point_results_to_the_last_bit():
    """Not ``approx``: the same float, because there is no reduction at all."""
    items = [np.linspace(0.0, 1.0, 50) + i for i in range(12)]
    serial = map_ordered(float_pipeline, items, n_jobs=1)
    parallel = map_ordered(float_pipeline, items, n_jobs=3)
    assert serial == parallel
    for a, b in zip(serial, parallel):
        assert float(a).hex() == float(b).hex()


def float_pipeline(values):
    """A numpy pipeline with an order-dependent reduction inside one item.

    Threads or processes must not touch this: the reduction happens *within* an
    item, so its order is fixed, and the parallel loop never reorders items.
    """
    return float(np.sum(np.exp(-np.asarray(values, dtype=float)) / 3.0))


# --------------------------------------------------------------------------
# the call sites: a simulation study and a parametric bootstrap
# --------------------------------------------------------------------------

def test_run_scenario_is_identical_with_two_workers():
    """The claim the ``n_jobs`` argument makes, checked on the report itself.

    Every block of the report is compared, including the Monte Carlo standard
    errors, so a single moved digit anywhere fails.
    """
    from pimeta.validation import Scenario, run_scenario

    s = Scenario(name="quick", k=12, mu=0.1, tau2=0.04, se=0.2)
    serial = run_scenario(s, reps=12, seed=4242, n_jobs=1)
    parallel = run_scenario(s, reps=12, seed=4242, n_jobs=2)
    # the worker count is recorded, everything else must be identical
    assert serial["settings"].pop("n_jobs") == 1
    assert parallel["settings"].pop("n_jobs") == 2
    assert json.dumps(serial, sort_keys=True) == json.dumps(parallel, sort_keys=True)


def test_the_bootstrap_p_value_is_identical_with_two_workers():
    """Same for the dynamic-heterogeneity parametric bootstrap.

    The bootstrap samples are drawn before the loop, so the p-value is a
    function of the samples and not of the scheduling.
    """
    from pimeta.dynamichet import tau_trend

    rng = np.random.default_rng(11)
    k = 16
    t = rng.uniform(1990.0, 2020.0, k)
    # Heterogeneity has to be real for this test to mean anything: with tau^2
    # driven to zero the observed information is singular, and both refits
    # return the same likelihood, so two workers would agree on a degenerate
    # number.  0.35 against sampling SDs near 0.2 keeps tau^2 interior.
    y = 0.2 + 0.01 * (t - 2005.0) + rng.normal(0.0, 0.35, k)
    v = rng.uniform(0.02, 0.06, k)
    fit = tau_trend(y, v, t)
    # tau^2 is a function of t here, so the check is on its smallest value
    assert float(np.min(np.asarray(fit.tau2))) > 0.01, (
        "the fit is degenerate; the comparison is vacuous")
    serial = fit.test_trend("bootstrap", n_boot=19, seed=3, n_jobs=1)
    parallel = fit.test_trend("bootstrap", n_boot=19, seed=3, n_jobs=2)
    assert 0.0 < serial["p_value"] < 1.0
    assert serial["p_value"] == parallel["p_value"]
    assert serial["statistic"] == parallel["statistic"]
    assert float(serial["p_value"]).hex() == float(parallel["p_value"]).hex()
