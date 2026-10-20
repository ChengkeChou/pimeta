"""``[`` / ``head`` / ``tail`` for ``list.rma``-style result objects.

metafor's ``[.list.rma``, ``head.list.rma`` and ``tail.list.rma`` slice every
per-study vector of a leave-one-out / influence / cumulative result at once so
the rows stay aligned.  Python has no ``[`` hook for plain dataclasses, so
:mod:`pimeta.rma_list` exposes the same behaviour as explicit helpers.
"""

import numpy as np
import pytest

import pimeta as ma
from pimeta.rma_list import (
    as_data_frame_rma_list,
    head_rma,
    slice_rma,
    tail_rma,
)

YI = [0.10, -0.05, 0.22, 0.08, -0.12, 0.31, 0.04, 0.19, -0.02, 0.27]
VI = [0.010, 0.014, 0.020, 0.011, 0.018, 0.025, 0.013, 0.017, 0.012, 0.022]


@pytest.fixture(scope="module")
def loo():
    return ma.leave1out(YI, VI)


@pytest.fixture(scope="module")
def infl():
    return ma.influence(YI, VI)


# --------------------------------------------------------------------------- #
# head / tail
# --------------------------------------------------------------------------- #

def test_head_rma_keeps_the_first_rows(loo):
    h = head_rma(loo, 3)
    assert len(h.estimate) == 3
    assert h.estimate == pytest.approx(loo.estimate[:3])
    assert h.se == pytest.approx(loo.se[:3])
    assert h.ci_lower == pytest.approx(loo.ci_lower[:3])
    assert h.QE == pytest.approx(loo.QE[:3])


def test_tail_rma_keeps_the_last_rows(loo):
    t = tail_rma(loo, 2)
    assert len(t.estimate) == 2
    assert t.estimate == pytest.approx(loo.estimate[-2:])


def test_head_and_tail_do_not_mutate_the_input(loo):
    before = list(loo.estimate)
    head_rma(loo, 4)
    tail_rma(loo, 4)
    assert loo.estimate == before


def test_head_rma_defaults_to_six(loo):
    assert len(head_rma(loo).estimate) == 6
    assert len(tail_rma(loo).estimate) == 6


def test_head_rma_edge_cases(loo):
    assert len(head_rma(loo, 0).estimate) == 0
    assert len(head_rma(loo, 99).estimate) == 10
    assert len(tail_rma(loo, 0).estimate) == 0
    assert len(tail_rma(loo, 99).estimate) == 10
    with pytest.raises(ValueError, match="non-negative"):
        head_rma(loo, -1)


# --------------------------------------------------------------------------- #
# [  (slice_rma)
# --------------------------------------------------------------------------- #

def test_slice_rma_with_a_list(loo):
    s = slice_rma(loo, [1, 3])
    assert s.estimate == pytest.approx([loo.estimate[1], loo.estimate[3]])
    assert s.p == pytest.approx([loo.p[1], loo.p[3]])


def test_slice_rma_with_a_slice(loo):
    s = slice_rma(loo, slice(2, 5))
    assert s.estimate == pytest.approx(loo.estimate[2:5])


def test_slice_rma_with_a_boolean_mask(loo):
    mask = np.array([True, False] * 5)
    s = slice_rma(loo, mask)
    assert len(s.estimate) == 5
    assert s.estimate == pytest.approx([loo.estimate[i] for i in range(0, 10, 2)])


def test_slice_rma_updates_k_when_present(infl):
    assert infl.k == 10
    assert head_rma(infl, 4).k == 4
    assert tail_rma(infl, 3).k == 3
    assert len(head_rma(infl, 4).rstudent) == 4
    assert len(head_rma(infl, 4).dfbs) == 4


def test_slice_rma_in_place(loo):
    h = head_rma(loo, 2)
    same = slice_rma(h, [0], copy_obj=False)
    assert same is h
    assert len(h.estimate) == 1


def test_slice_rma_rejects_objects_without_rows():
    class Empty:
        pass

    with pytest.raises(TypeError, match="no per-row fields"):
        slice_rma(Empty(), [0])


# --------------------------------------------------------------------------- #
# as.data.frame
# --------------------------------------------------------------------------- #

def test_as_data_frame_rma_list(loo):
    df = as_data_frame_rma_list(loo)
    assert "estimate" in df and "se" in df and "ci_lower" in df
    assert len(df["estimate"]) == 10
    assert df["estimate"] == pytest.approx(loo.estimate)


def test_as_data_frame_rma_list_rows(loo):
    df = as_data_frame_rma_list(loo, rows=[0, 1])
    assert len(df["estimate"]) == 2


# --------------------------------------------------------------------------- #
# package-level aliases
# --------------------------------------------------------------------------- #

def test_head_and_tail_are_exported():
    assert ma.head is head_rma
    assert ma.tail is tail_rma
    assert "head" in ma.__all__ and "tail" in ma.__all__
    assert "slice_rma" in ma.__all__


def test_head_works_on_a_rma_result_table(loo):
    h = ma.head(loo, 3)
    assert isinstance(h, type(loo))
    assert len(h.estimate) == 3


# --------------------------------------------------------------------------- #
# x[i] — R's [.list.rma as a real Python protocol
# --------------------------------------------------------------------------- #

def test_getitem_matches_r_subsetting(loo):
    s = loo[0:3]
    assert isinstance(s, type(loo))
    assert s.estimate == pytest.approx(loo.estimate[:3])
    assert s.se == pytest.approx(loo.se[:3])


def test_getitem_scalar_keeps_length_one(loo):
    """R's x[2] is a length-1 list, never a bare scalar."""
    s = loo[2]
    assert s.estimate == pytest.approx([loo.estimate[2]])
    assert s.QE == pytest.approx([loo.QE[2]])


def test_getitem_with_a_list_of_indices(loo):
    s = loo[[1, 3]]
    assert s.estimate == pytest.approx([loo.estimate[1], loo.estimate[3]])


def test_getitem_updates_k(infl):
    assert infl[0:4].k == 4
    assert len(infl[0:4].rstudent) == 4
    assert len(infl[0:4].beta_del) == 4


def test_getitem_on_cyumul_keeps_the_per_row_k():
    cu = ma.cumul(YI, VI)
    s = cu[0:3]
    assert s.estimate == pytest.approx(cu.estimate[:3])
    assert s.k == [1, 2, 3]


def test_getitem_on_metacum():
    mc = ma.metacum(YI, VI)
    s = mc[0:2]
    assert s.TE == pytest.approx(mc.TE[:2])
    assert s.k == 2

