"""Guards for the ``pimeta.nma`` package skeleton.

The network meta-analysis implementation lives in ``pimeta/netmeta.py`` and is
being carved into ``pimeta/nma/`` one section at a time.  ``pimeta.netmeta``
must keep re-exporting every moved name, so these tests pin the identity
relationship rather than the location.
"""

from __future__ import annotations

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
