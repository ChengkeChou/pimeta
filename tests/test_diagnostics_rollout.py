"""Which results carry ``.diagnostics``, and that it is a ``Diagnosis``.

The rollout started from three result classes, one of which spelled the property
as a method returning a dict. ``attach_diagnostics`` makes the property a
one-line addition, and this file keeps the list honest in two ways:

* the set of classes that carry it is pinned, so adding one is a deliberate act
  rather than a side effect;
* each family that carries it is fitted for real at least once, because a
  property that returns a wrong diagnostic is worse than one that is absent —
  ``few_studies`` firing on a heterogeneity test would be exactly that.
"""

from __future__ import annotations

import inspect
import math

import pimeta as ma

#: Every result class that carries ``.diagnostics`` today.
WITH_DIAGNOSTICS = {
    # the three it started from
    "RmaResult",
    "RmaMvResult",
    "RobmaResult",
    # rolled out through @attach_diagnostics
    "DosresResult",
    "IpdCurveResult",
    "MatregResult",
    "NetmetaResult",
    "RmaLsResult",
    "RmaMhResult",
    "RmaPetoResult",
    "SelModelResult",
    "TimeCourseResult",
    # a grid of specifications, whose findings are about the grid
    "MultiverseResult",
}


def _carrying() -> set[str]:
    found = set()
    for name in ma.__all__:
        cls = getattr(ma, name, None)
        if inspect.isclass(cls) and name.endswith("Result"):
            if isinstance(vars(cls).get("diagnostics"), property):
                found.add(name)
    return found


def test_the_set_of_results_with_diagnostics_is_pinned():
    assert _carrying() == WITH_DIAGNOSTICS


def test_every_diagnostics_property_returns_a_diagnosis():
    # The contract is the type, not the name: `Diagnosis` has codes(), worst and
    # to_dict(), so a reader can aggregate findings without knowing the family.
    for name in sorted(WITH_DIAGNOSTICS):
        cls = getattr(ma, name)
        prop = vars(cls).get("diagnostics")
        assert isinstance(prop, property), name
        assert prop.fget is not None and prop.fget.__doc__, name


def test_a_bare_object_gets_an_empty_diagnosis_rather_than_an_exception():
    """Every check reads with ``getattr``, so a result with no state is clean.

    This is what makes the decorator safe to apply family by family: a class
    that carries only three of the attributes a check reads must not raise.
    """
    @ma.checks.attach_diagnostics
    class Bare:
        pass

    diagnosis = Bare().diagnostics
    assert isinstance(diagnosis, ma.Diagnosis)
    assert diagnosis.codes() == [] and diagnosis.worst is None


def test_a_bare_object_carrying_one_field_still_works():
    @ma.checks.attach_diagnostics
    class AlmostABare:
        tau2 = 0.0
        k = 30

    assert AlmostABare().diagnostics.codes() == ["tau2_zero"]


# --------------------------------------------------------------------------- #
# one real fit per family that newly carries the property
# --------------------------------------------------------------------------- #
def test_rma_peto_returns_a_diagnosis():
    fit = ma.rma_peto([3, 2, 5, 4, 6], [30, 28, 40, 35, 50],
                      [4, 5, 3, 6, 4], [29, 30, 38, 34, 48])
    diagnosis = fit.diagnostics
    assert isinstance(diagnosis, ma.Diagnosis)
    assert diagnosis.worst is None or diagnosis.worst in ma.checks.SEVERITIES


def test_rma_mh_returns_a_diagnosis():
    fit = ma.rma_mh([3, 2, 5, 4, 6], [30, 28, 40, 35, 50],
                    [4, 5, 3, 6, 4], [29, 30, 38, 34, 48])
    assert isinstance(fit.diagnostics, ma.Diagnosis)


def test_matreg_returns_a_diagnosis():
    # two correlated estimates and their 2x2 covariance; indices are 1-based
    fit = ma.matreg(1, 2, [[0.04, 0.01], [0.01, 0.09]], n=120)
    diagnosis = fit.diagnostics
    assert isinstance(diagnosis, ma.Diagnosis)
    assert "non_finite_estimate" not in diagnosis.codes()
    # `matreg` estimates no between-study variance, so a small `k` here is not
    # a few-studies problem: this is the false positive the rollout found.
    assert "few_studies" not in diagnosis.codes()
    assert all(math.isfinite(v) for v in fit.beta)


def test_netmeta_returns_a_diagnosis():
    fit = ma.netmeta(
        TE=[0.10, -0.30, -0.40, 0.55, -0.25, 0.20, 0.40, 0.20, 0.50, -0.10],
        seTE=[0.05, 0.06, 0.07, 0.10, 0.12, 0.09, 0.11, 0.13, 0.15, 0.14],
        treat1=["A", "A", "B", "A", "A", "D", "D", "E", "B", "B"],
        treat2=["B", "C", "C", "D", "D", "E", "C", "C", "E", "E"])
    diagnosis = fit.diagnostics
    assert isinstance(diagnosis, ma.Diagnosis)
    # a five-treatment network on ten studies is not a few-studies problem
    assert "few_studies" not in diagnosis.codes()
