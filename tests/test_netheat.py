"""R parity for the ``netheat`` matrix (netmeta::netheat, netmeta 3.7-0).

The reference values come from R itself.  ``netheat()`` returns ``NULL`` and
only draws, so the numbers were produced by running its internal computation:

    nmak  <- netmeta:::nma_krahn(x, tau.preset = tau)
    dct   <- netmeta:::decomp.tau(x, tau.preset = tau)
    qtyp  <- apply(dct$residuals.inc.detach, 2,
                   function(x) t(x) %*% solve(nmak$V) * x)
    inc   <- matrix(dct$Q.inc.design, nrow(qtyp), ncol(qtyp))
    t1    <- -(inc - qtyp)

Both the common-effect (tau = 0) and the random-effects (tau = tau.within)
branch are pinned, on networks whose detach residuals are not all zero — the
3-treatment loop alone is degenerate and would not catch a wrong formula.
"""

from __future__ import annotations

import warnings

import numpy as np
import pytest

import pimeta as ma
from pimeta.netmeta import netheat_matrix

# Two loops sharing the A:B edge -> non-zero detachment residuals.
TWO_LOOPS = dict(
    TE=[0.30, -0.20, 0.90, 0.40, 0.10],
    seTE=[0.12, 0.15, 0.18, 0.14, 0.13],
    treat1=["A", "B", "A", "A", "B"],
    treat2=["B", "C", "C", "D", "D"],
    studlab=["S1", "S2", "S3", "S4", "S5"],
)

# Two studies in the A:B design disagree, so tau.within > 0 and the
# random-effects branch differs from the common-effect one.
WITHIN_HET = dict(
    TE=[0.10, 1.50, 0.30, 0.40, 0.10, 0.55],
    seTE=[0.05, 0.05, 0.12, 0.14, 0.13, 0.16],
    treat1=["A", "A", "A", "A", "B", "B"],
    treat2=["B", "B", "C", "D", "C", "D"],
    studlab=["S1", "S2", "S3", "S4", "S5", "S6"],
)

DESIGNS = ["A:B", "A:C", "A:D", "B:C", "B:D"]

# R: round(-(inc - qtyp), 10)
R_TWO_LOOPS = np.array([
    [-1.113908, -1.113908, 0.805096, -1.113908, 0.805096],
    [-2.391782, -4.873958, -0.556198, -4.873958, -0.556198],
    [1.265580, -0.235984, -0.235984, -0.235984, -0.235984],
    [-1.660960, -3.384693, -0.386248, -3.384693, -0.386248],
    [1.091239, -0.203476, -0.203476, -0.203476, -0.203476],
])

R_WITHIN_HET_COMMON = np.array([
    [-1.771218, -1.2483575, -1.3464903, -1.2483575, -1.3464903],
    [-4.192651, -4.4940738, 0.3987886, -4.4940738, 0.3987886],
    [-7.411470, 0.3767125, -7.8217393, 0.3767125, -7.8217393],
    [-4.920542, -5.2742950, 0.4680228, -5.2742950, 0.4680228],
    [-9.680287, 0.4920326, -10.2161493, 0.4920326, -10.2161493],
])

R_WITHIN_HET_RANDOM = np.array([
    [-0.13295956, -0.06164555, -0.10419272, -0.06164555, -0.10419272],
    [-0.02223548, -0.02988123, 0.02835106, -0.02988123, 0.02835106],
    [-0.11261116, 0.02481972, -0.12029700, 0.02481972, -0.12029700],
    [-0.02229152, -0.02995655, 0.02842252, -0.02995655, 0.02842252],
    [-0.11328880, 0.02496907, -0.12102088, 0.02496907, -0.12102088],
])


def _fit(spec: dict):
    return ma.netmeta(sm="RR", reference_group="A", **spec)


def test_netheat_matrix_matches_r_common_effect():
    heat, labels = netheat_matrix(_fit(TWO_LOOPS))
    assert labels == DESIGNS
    assert heat.shape == (5, 5)
    assert np.allclose(heat, R_TWO_LOOPS, atol=1e-6)


def test_netheat_matrix_matches_r_on_within_design_heterogeneity():
    heat, labels = netheat_matrix(_fit(WITHIN_HET))
    assert labels == DESIGNS
    assert np.allclose(heat, R_WITHIN_HET_COMMON, atol=1e-6)


def test_netheat_matrix_matches_r_random_effects():
    result = _fit(WITHIN_HET)
    # R: tau.within(x) = 0.988685996664
    assert ma.decomp_design(result).tau_within == pytest.approx(0.988686, abs=1e-6)

    heat, labels = netheat_matrix(result, random=True)
    assert labels == DESIGNS
    assert np.allclose(heat, R_WITHIN_HET_RANDOM, atol=1e-7)

    common, _ = netheat_matrix(result, random=False)
    assert not np.allclose(heat, common), "random and common must differ here"


def test_netheat_draws_the_matrix_and_labels_the_designs(tmp_path):
    plot = ma.netheat(_fit(TWO_LOOPS), filename=str(tmp_path / "heat.png"))
    assert plot is not None
    assert (tmp_path / "heat.png").stat().st_size > 0
    ticks = [t.get_text() for t in plot.ax.get_yticklabels()]
    assert ticks == DESIGNS
    assert [t.get_text() for t in plot.ax.get_xticklabels()] == DESIGNS


def test_netheat_without_a_loop_warns_and_returns_none():
    result = ma.netmeta(TE=[0.30, -0.20], seTE=[0.12, 0.15],
                        treat1=["A", "B"], treat2=["B", "A"],
                        studlab=["S1", "S2"], sm="RR")
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        assert ma.netheat(result) is None
    assert any("loop" in str(w.message) for w in caught)
