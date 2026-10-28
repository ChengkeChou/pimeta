"""The multilevel checks: variance components, conditioning, boundary rho.

The shared point of these three is that they read state every ``rma_mv`` fit
already carries, so they cost nothing and cannot change a fit. Each has a fire
test and a silent test.
"""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np

import pimeta as ma


def _stub(**kw):
    """A fitted-result stand-in carrying only the fields a check reads."""
    base = {"tau2": None, "sigma2": None, "vb": None, "rho": None,
            "k": 10, "priors": {}, "tau": {}}
    base.update(kw)
    return SimpleNamespace(**base)


def _multilevel_data():
    """Six clusters of four studies: enough for a two-component model."""
    rng = np.random.default_rng(3)
    n_clusters, per = 6, 4
    cluster = np.repeat(np.arange(n_clusters), per)
    idv = np.arange(cluster.size)
    vi = rng.uniform(0.02, 0.06, cluster.size)
    # a genuine cluster-level offset, otherwise every variance component is
    # estimated at zero and the silent test cannot be silent
    offset = rng.normal(0.0, 0.25, n_clusters)[cluster]
    yi = 0.3 + offset + rng.normal(0.0, 0.05, cluster.size)
    # a mapping, not a DataFrame: pandas is optional, and importing it at module
    # level turns a missing optional dependency into a collection error
    return {"yi": yi, "vi": vi, "cluster": cluster, "id": idv}


def _fit_multilevel(d, mods=None):
    return ma.rma_mv(np.asarray(d["yi"], dtype=float),
                     vi=np.asarray(d["vi"], dtype=float),
                     mods=mods, data=d, random="~ 1 | cluster / id",
                     method="REML")


# --------------------------------------------------------------------------- #
# the covariance really is a list-shaped field
# --------------------------------------------------------------------------- #
def test_diagnose_does_not_raise_on_a_multilevel_fit():
    """A diagnostics function that crashes on one model family is worse than none.

    ``tau2`` on an ``rma_mv`` result can be a list, where a bare float() would
    raise; this is the regression test for that.
    """
    fit = _fit_multilevel(_multilevel_data())
    diagnosis = ma.diagnose(fit)          # must not raise
    assert isinstance(diagnosis.to_dict(), dict)


def test_variance_component_zero_fires_when_a_component_is_unneeded():
    assert "variance_component_zero" in ma.diagnose(
        _stub(sigma2=[0.04, 0.0])).codes()


def test_variance_component_zero_stays_silent_when_all_are_positive():
    assert "variance_component_zero" not in ma.diagnose(
        _stub(sigma2=[0.04, 0.01])).codes()


# --------------------------------------------------------------------------- #
# conditioning
# --------------------------------------------------------------------------- #
def test_ill_conditioned_fires_on_a_near_singular_covariance():
    vb = np.array([[1.0, 1.0], [1.0, 1.0 + 1e-14]])
    assert "ill_conditioned" in ma.diagnose(_stub(vb=vb)).codes()


def test_ill_conditioned_stays_silent_on_a_well_conditioned_covariance():
    assert "ill_conditioned" not in ma.diagnose(_stub(vb=np.eye(2))).codes()


# --------------------------------------------------------------------------- #
# a correlation at the edge of its range
# --------------------------------------------------------------------------- #
def test_boundary_correlation_fires_when_rho_is_pinned_at_one():
    fit = _fit_multilevel(_multilevel_data())
    fit.rho = [1.0]
    assert "boundary_correlation" in ma.diagnose(fit).codes()


def test_boundary_correlation_stays_silent_when_rho_is_interior():
    fit = _fit_multilevel(_multilevel_data())
    fit.rho = [0.35]
    assert "boundary_correlation" not in ma.diagnose(fit).codes()


def test_boundary_correlation_is_skipped_when_the_model_has_no_rho():
    fit = _fit_multilevel(_multilevel_data())
    fit.rho = None
    assert "boundary_correlation" not in ma.diagnose(fit).codes()
