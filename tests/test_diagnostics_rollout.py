"""Which results carry ``.diagnostics``, what fires on them, and what does not.

The rollout started from three result classes, one of which spelled the property
as a method returning a dict. ``attach_diagnostics`` makes the property a
one-line addition, and this file keeps the rollout honest in three ways:

* the set of classes that carry it is pinned, so adding one is a deliberate act
  rather than a side effect, and each entry says *why* the checks are meaningful
  for it;
* the classes deliberately left alone are pinned too, with the reason, because
  "not this one, and here is why" is the harder and more useful half of the
  judgement -- ``few_studies`` firing on a heterogeneity test is a false
  diagnostic, which is worse than a missing one;
* each family is fitted for real at least once, at two study counts, because a
  property that returns a *wrong* diagnostic is worse than one that is absent.

The other thing this file pins is the storage. The package keeps the same two
facts (how many studies, what is tau2) in four layouts:

* ``k`` and a scalar ``tau2`` (``rma``, ``netmetabin``, ``metarate``, ...);
* ``k`` and ``tau2`` behind ``random.tau2``, which is R's own nesting on the
  RevMan-shaped results (``attach_tau2`` flattens it);
* no ``k`` at all, but ``studies`` as a count (``ipd_one_stage``);
* ``k`` as a *list* of per-interval counts (``piecewise_exp``), where the check
  has to decline rather than guess.

The last of those is why the rollout now has behavioural tests rather than a
list of names: attaching the decorator to ``PiecewiseResult`` raised
``TypeError`` from inside a property documented never to raise, because
``int(k)`` met a list. Only a real fit finds that.
"""

from __future__ import annotations

import dataclasses
import inspect
import math
import warnings

import numpy as np
import pytest

import pimeta as ma
from pimeta.checks import Diagnosis

#: Every result class that carries ``.diagnostics``, and what makes the checks
#: meaningful for it. The keys are the contract; the values are documentation.
DECORATED: dict[str, str] = {
    # the three the rollout started from
    "RmaResult": "the reference fit",
    "RmaMvResult": "multivariate fit with a between-study covariance",
    "RobmaResult": "robust Bayesian model averaging",
    # rolled out through @attach_diagnostics
    "DosresResult": "dose-response fit",
    "IpdCurveResult": "two-stage IPD dose-response curve",
    "MatregResult": "meta-regression on a covariance matrix",
    "NetmetaResult": "frequentist network meta-analysis",
    "RmaLsResult": "location-scale fit",
    "RmaMhResult": "Mantel-Haenszel pooling",
    "RmaPetoResult": "Peto odds ratio",
    "SelModelResult": "selection model",
    "TimeCourseResult": "time-course network fit",
    "MultiverseResult": "a grid of specifications, each checked",
    # the 1.1.1 rollout: fits that estimate a heterogeneity and know their k
    "RmaGlmmResult": "mixed-effects binomial/Poisson pooling",
    "IpdResult": "two-stage IPD pooling",
    "IpdNmaResult": "one-stage network meta-analysis of IPD",
    "OneStageResult": "one-stage IPD model",
    "NetmetabinResult": "network meta-analysis of binary data",
    "NetmetaregResult": "network meta-regression",
    "PiecewiseResult": "piecewise exponential pooling",
    "MetabResult": "RevMan-shaped binary pooling",
    "MetacontResult": "RevMan-shaped continuous pooling",
    "MetacorResult": "RevMan-shaped correlation pooling",
    "MetagenResult": "RevMan-shaped generic inverse-variance pooling",
    "MetaincResult": "RevMan-shaped incidence-rate pooling",
    "MetameanResult": "RevMan-shaped single-arm mean pooling",
    "MetapropResult": "RevMan-shaped proportion pooling",
    "MetarateResult": "RevMan-shaped rate pooling",
}

#: Classes deliberately left alone, with the reason. Not an omission list: each
#: one would get a diagnostic that is true but useless, or false.
NOT_DECORATED: dict[str, str] = {
    "BaujatResult": "a per-study contribution table, not a fit",
    "BeggResult": "a rank test, not a fit: no variance is estimated",
    "EggerResult": "a regression test on the funnel plot, not a fit",
    "HettestResult": "a heterogeneity test, whose subject is the heterogeneity",
    "PermutestResult": "a permutation test",
    "RegtestResult": "a regression test on the funnel plot",
    "TauCorResult": "a test between two heterogeneity estimates",
    "TesResult": "a test, not a fit",
    "LimitmetaResult": "an adjusted estimate whose variance is not estimated",
    "FsnResult": "a fail-safe N, not a fit",
    "TrimfillResult": "a fill-in count attached to an existing fit",
    "LeaveOneOutResult": "a collection of fits: the collection has no single k",
    "CumulResult": "a cumulative sequence of fits",
    "InfluenceResult": "a collection of leave-one-out fits",
    "MetainfResult": "a collection of leave-one-out fits",
    "ProfileResult": "a profile of one fit's tau2, not a fit",
    "SensitivityResult": "a grid of refits",
    "RhoSensitivityResult": "a grid of refits: tau2 is a vector, not a scalar",
}


@pytest.fixture(scope="module", autouse=True)
def _quiet():
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        yield


def _carrying() -> set[str]:
    found = set()
    for name in ma.__all__:
        cls = getattr(ma, name, None)
        if inspect.isclass(cls) and name.endswith("Result"):
            if isinstance(vars(cls).get("diagnostics"), property):
                found.add(name)
    return found


def test_the_set_of_results_with_diagnostics_is_pinned():
    assert _carrying() == set(DECORATED), (
        "the list of classes with diagnostics changed; say in DECORATED why "
        "the new one is meaningful, or take the decorator off")


def test_the_skipped_classes_are_named_and_are_not_decorated():
    assert not set(NOT_DECORATED) & _carrying(), (
        "documented as skipped but decorated")
    for name in NOT_DECORATED:
        assert hasattr(ma, name), f"{name} is not a public result class"


def test_every_diagnostics_property_returns_a_diagnosis():
    # The contract is the type, not the name: `Diagnosis` has codes(), worst and
    # to_dict(), so a reader can aggregate findings without knowing the family.
    for name in sorted(DECORATED):
        prop = vars(getattr(ma, name)).get("diagnostics")
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
# one real fit per family that carries the property, at two study counts
# --------------------------------------------------------------------------- #
E1 = [3, 4, 5, 6, 7, 8]
N1 = [40, 45, 50, 55, 60, 65]
E2 = [6, 8, 9, 11, 12, 15]
N2 = [42, 44, 52, 58, 62, 66]
T1 = ["A", "A", "A", "B", "B", "C"]
T2 = ["B", "C", "B", "C", "C", "B"]


def _slices(values, k):
    return list(values)[:k]


def _ipd(k, rng_seed=0):
    """A k-study, two-arm, Gaussian IPD set with a real heterogeneity."""
    rng = np.random.default_rng(rng_seed)
    study = np.repeat(np.arange(1, k + 1), 40)
    treat = np.tile(np.repeat([0, 1], 20), k)
    shift = np.repeat(rng.normal(0.0, 0.4, k), 40)
    y = rng.normal(0.0, 1.0, 40 * k) + 0.5 * treat + shift
    return {"y": y, "study": study, "treat": treat}


def _ipd_binary(k, rng_seed=1):
    rng = np.random.default_rng(rng_seed)
    study = np.repeat(np.arange(1, k + 1), 40)
    treat = np.tile(np.repeat([0, 1], 20), k)
    y = rng.binomial(1, 0.3, 40 * k)
    return {"y": y, "study": study, "treat": treat}


def _nma(k):
    """A k-study, three-treatment network with a treatment effect on each arm."""
    rng = np.random.default_rng(3)
    study = np.repeat(np.arange(1, k + 1), 30)
    treat = np.tile(np.repeat(["A", "B", "C"], 10), k)
    y = (rng.normal(0.0, 1.0, 30 * k) + 0.4 * (treat == "B")
         + 0.6 * (treat == "C"))
    return {"y": y, "study": study, "treat": treat}


def _netmetareg(k):
    net = ma.netmeta(
        TE=_slices([0.1, 0.2, 0.3, 0.15, 0.25, 0.35], k),
        seTE=[0.1] * k,
        treat1=_slices(["A", "B", "A", "B", "C", "A"], k),
        treat2=_slices(["B", "C", "C", "C", "B", "B"], k),
        studlab=list(range(1, k + 1)))
    return ma.netmetareg(net, np.linspace(0.0, 1.0, k))


def _piecewise(k, rng_seed=2):
    rng = np.random.default_rng(rng_seed)
    total = 20 * k
    time = rng.exponential(2.0, total) + 0.01
    event = rng.binomial(1, 0.6, total)
    study = np.repeat(np.arange(1, k + 1), 20)
    treat = np.tile(np.repeat([0, 1], 10), k)
    return ma.piecewise_exp(time, event, study=study, treat=treat,
                            breaks=[0, 1, 2, 4])


#: ``(name, build at k=2, build at k=6)`` for every class where ``few_studies``
#: is the check that has to fire. Two studies cannot identify a variance; six
#: can, and a check that keeps firing at six is a check reading nothing.
FEW_STUDIES = [
    ("RmaGlmmResult",
     lambda: ma.rma_glmm("PLO", xi=[10, 12], ni=[50, 52]),
     lambda: ma.rma_glmm("PLO", xi=[10, 12, 8, 15, 9, 20],
                         ni=[50, 52, 45, 60, 48, 70])),
    ("IpdResult",
     lambda: ma.ipd_two_stage(**_ipd_binary(2), family="binomial"),
     lambda: ma.ipd_two_stage(**_ipd_binary(6), family="binomial")),
    ("OneStageResult",
     lambda: ma.ipd_one_stage(**_ipd(2), family="gaussian"),
     lambda: ma.ipd_one_stage(**_ipd(6), family="gaussian")),
    ("IpdNmaResult",
     lambda: ma.ipd_nma(**_nma(2)),
     lambda: ma.ipd_nma(**_nma(6))),
    ("NetmetabinResult",
     lambda: ma.netmetabin(_slices(E1, 2), _slices(N1, 2), _slices(E2, 2),
                           _slices(N2, 2), _slices(T1, 2), _slices(T2, 2)),
     lambda: ma.netmetabin(E1, N1, E2, N2, T1, T2)),
    ("NetmetaregResult", lambda: _netmetareg(3), lambda: _netmetareg(6)),
    ("MetabResult",
     lambda: ma.metabin(_slices(E1, 2), _slices(N1, 2), _slices(E2, 2),
                        _slices(N2, 2)),
     lambda: ma.metabin(E1, N1, E2, N2)),
    ("MetacontResult",
     lambda: ma.metacont(n_e=[30, 35], mean_e=[1.2, 1.0], sd_e=[1.0, 1.1],
                         n_c=[32, 33], mean_c=[1.6, 1.3], sd_c=[1.0, 1.0]),
     lambda: ma.metacont(n_e=np.linspace(25, 50, 6),
                         sd_e=np.linspace(0.9, 1.2, 6),
                         mean_e=np.linspace(0.9, 1.4, 6),
                         n_c=np.linspace(26, 48, 6),
                         sd_c=np.linspace(0.9, 1.1, 6),
                         mean_c=np.linspace(1.2, 1.8, 6))),
    ("MetacorResult",
     lambda: ma.metacor([0.3, 0.5], [40, 50]),
     lambda: ma.metacor(np.linspace(-0.3, 0.7, 6), np.linspace(40, 60, 6))),
    ("MetagenResult",
     lambda: ma.metagen(TE=[0.1, 0.3], seTE=[0.1, 0.12]),
     lambda: ma.metagen(TE=[0.1, 0.3, 0.2, 0.5, 0.4, 0.15],
                        seTE=[0.1, 0.12, 0.09, 0.2, 0.1, 0.11])),
    ("MetaincResult",
     lambda: ma.metainc(_slices(E1, 2), _slices(N1, 2), _slices(E2, 2),
                        _slices(N2, 2)),
     lambda: ma.metainc(E1, N1, E2, N2)),
    ("MetameanResult",
     lambda: ma.metamean(n=[30, 40], mean=[1.0, 1.2], sd=[0.5, 0.6]),
     lambda: ma.metamean(n=np.linspace(30, 50, 6),
                         mean=np.linspace(0.9, 1.3, 6),
                         sd=np.linspace(0.4, 0.65, 6))),
    ("MetapropResult",
     lambda: ma.metaprop([3, 5], [30, 40]),
     lambda: ma.metaprop(np.linspace(3, 12, 6), np.linspace(30, 55, 6))),
    ("MetarateResult",
     lambda: ma.metarate([3, 5], [30, 40]),
     lambda: ma.metarate(np.linspace(3, 12, 6), np.linspace(30, 55, 6))),
]


def _codes(fit) -> list[str]:
    result = fit.diagnostics
    assert isinstance(result, Diagnosis)
    return [d.code for d in result]


@pytest.mark.parametrize("name,build_small,build_large", FEW_STUDIES,
                         ids=[row[0] for row in FEW_STUDIES])
def test_few_studies_fires_on_two_studies_and_not_on_six(
        name, build_small, build_large):
    small, large = build_small(), build_large()
    assert type(small).__name__ == name and type(large).__name__ == name
    assert "few_studies" in _codes(small), (
        f"{name} did not report few_studies at k=2: the check is reading a "
        "study count the class does not carry")
    assert "few_studies" not in _codes(large), (
        f"{name} still reported few_studies at k=6")


def test_a_one_stage_fit_reports_its_study_count_as_studies_not_k():
    fit = ma.ipd_one_stage(**_ipd(2), family="gaussian")
    assert not hasattr(fit, "k") or fit.k is None
    assert fit.studies == 2
    assert "few_studies" in _codes(fit)


def test_a_piecewise_fit_declines_rather_than_guessing_its_k():
    # `k` is one count per interval: intervals are informed by different numbers
    # of studies, so there is no single k to compare against five.
    fit = _piecewise(6)
    assert np.size(fit.k) > 1, "k is expected to be per-interval"
    assert "few_studies" not in _codes(fit)
    # what it can report, it does: the checks that read its own stored numbers
    assert set(_codes(fit)) <= {"tau2_zero", "non_finite_estimate"}


def test_the_flattened_tau2_is_the_random_effects_one():
    fit = ma.metabin(E1, N1, E2, N2)
    assert fit.tau2 == fit.random.tau2
    assert dataclasses.replace(fit, random=None).tau2 is None, (
        "a fixed-effect-only fit did not estimate a variance, so it must not "
        "look as if it did")


def test_a_class_without_a_random_effects_component_is_not_few_studies():
    # The point of returning None rather than 0.0: `few_studies` is about
    # identifying a variance, and there is none to identify here.
    fit = dataclasses.replace(ma.metabin(E1, N1, E2, N2), random=None, k=2)
    assert "few_studies" not in _codes(fit)


def test_the_metagen_result_names_the_studies_it_dropped():
    fit = ma.metagen(TE=[0.1, np.nan, 0.2, 0.5, 0.4, 0.15],
                     seTE=[0.1, 0.12, 0.09, 0.2, 0.1, 0.11])
    assert "studies_dropped" in _codes(fit)
    assert fit.k == 5 and fit.k_all == 6


# --------------------------------------------------------------------------- #
# the families the rollout started from, still fitted for real
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
    codes = _codes(fit)
    assert "non_finite_estimate" not in codes
    # `matreg` estimates no between-study variance, so a small `k` here is not
    # a few-studies problem: this is the false positive the rollout found.
    assert "few_studies" not in codes
    assert all(math.isfinite(v) for v in fit.beta)


def test_netmeta_returns_a_diagnosis():
    fit = ma.netmeta(
        TE=[0.10, -0.30, -0.40, 0.55, -0.25, 0.20, 0.40, 0.20, 0.50, -0.10],
        seTE=[0.05, 0.06, 0.07, 0.10, 0.12, 0.09, 0.11, 0.13, 0.15, 0.14],
        treat1=["A", "A", "B", "A", "A", "D", "D", "E", "B", "B"],
        treat2=["B", "C", "C", "D", "D", "E", "C", "C", "E", "E"])
    # a five-treatment network on ten studies is not a few-studies problem
    assert "few_studies" not in _codes(fit)
