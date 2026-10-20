"""Guards for the ``pimeta.nma`` package skeleton.

The network meta-analysis implementation lives in ``pimeta/netmeta.py`` and is
being carved into ``pimeta/nma/`` one section at a time.  ``pimeta.netmeta``
must keep re-exporting every moved name, so these tests pin the identity
relationship rather than the location.
"""

from __future__ import annotations

import sys

import pimeta as ma
import pimeta.nma as nma


def test_pairwise_is_the_same_object_in_both_namespaces():
    from pimeta.netmeta import PairwiseData
    from pimeta.netmeta import pairwise as from_netmeta

    assert from_netmeta is nma.pairwise
    assert nma.PairwiseData is PairwiseData
    assert ma.pairwise is nma.pairwise
    assert ma.PairwiseData is PairwiseData


def test_moved_pairwise_still_computes():
    res = ma.pairwise(
        treat=["A", "B", "A", "C"],
        n=[10, 10, 12, 12],
        event=[1, 3, 2, 4],
        studlab=["S1", "S1", "S2", "S2"],
    )
    assert res.sm == "OR"
    assert len(res.TE) == 2
    assert len(res.seTE) == 2
    assert res.narms == [2, 2]
    assert res.to_dict()["treat1"] == ["A", "A"]


def test_moved_pairwise_keeps_the_continuous_path():
    res = ma.pairwise(
        treat=["A", "B", "C"],
        n=[20, 22, 18],
        mean=[5.0, 5.5, 6.0],
        sd=[1.0, 1.1, 0.9],
        studlab=["S1", "S1", "S1"],
    )
    assert res.sm == "MD"
    assert len(res.TE) == 3          # three arms -> three comparisons
    assert res.narms == [3, 3, 3]


def test_moved_sections_stay_identity_equal():
    import pimeta.netmeta as netmeta_mod

    netmeta_mod = sys.modules["pimeta.netmeta"]
    for name in ("netmeasures", "NetmeasuresResult", "netimpact",
                 "NetimpactResult"):
        assert getattr(netmeta_mod, name) is getattr(nma, name), name
        assert getattr(ma, name) is getattr(nma, name), name


def test_moved_netmeasures_still_computes():
    meas = ma.netmeasures(_smoking_network())
    assert set(meas.proportion) == {"A:B", "A:C", "B:C"}
    assert all(0.0 <= p <= 1.0 for p in meas.proportion.values())
    assert set(meas.as_dict()) == {"proportion", "meanpath", "minpar",
                                   "minpar.study", "H.tilde", "random",
                                   "tau.preset"}


def test_moved_netimpact_still_computes():
    res = _smoking_network()
    impact = ma.netimpact(res)
    assert impact.studies == list(res.studlab_raw)
    assert impact.comparisons == ["A:B", "A:C", "B:C"]
    assert set(impact.impact_common) == set(res.studlab_raw)
    assert set(impact.as_dict()) >= {"impact.common", "impact.random"}


def _smoking_network():
    # Four-arm smoking cessation network: A = no intervention, B = counselling,
    # C = self-help, D = both.
    return ma.netmeta(
        treat1=["A", "B", "A", "C", "B", "C"],
        treat2=["B", "A", "C", "A", "C", "B"],
        TE=[0.30, -0.20, 0.50, 0.10, -0.15, 0.25],
        seTE=[0.12, 0.15, 0.18, 0.14, 0.11, 0.13],
        studlab=["S1", "S2", "S3", "S4", "S5", "S6"],
        data=None,
        sm="RR",
    )
