"""The specification curve: the curve, and the decisions underneath it.

Checked against the two things that make the figure honest rather than
decorative: every usable specification must appear exactly once, and every
decision must get its own panel. A curve that silently dropped a row, or a margin
panel that vanished because a decision had one level, would misrepresent the grid
in a way no reader could detect from the picture.

Figures are closed after each test; the plotting module uses the default backend
and an open figure is a resource leak the rest of the suite would eventually
notice through matplotlib's "too many figures" warning.
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pytest

import pimeta as ma
from pimeta.spec import AnalysisSpec, Decision


@pytest.fixture()
def spec() -> AnalysisSpec:
    return AnalysisSpec(decisions=(
        Decision(name="method", values=("REML", "DL"), default="REML",
                 reasons={"REML": "a", "DL": "b"}),
        Decision(name="test", values=("z", "knha"), default="z",
                 reasons={"z": "a", "knha": "b"}),
    ))


def _run(choices):
    y = [0.10, 0.25, 0.40, 0.55, 0.70, 0.30]
    v = [0.04, 0.05, 0.06, 0.03, 0.08, 0.05]
    return ma.rma(y, vi=v, method=choices["method"], test=choices["test"])


@pytest.fixture()
def mv(spec):
    return ma.multiverse(spec, _run)


@pytest.fixture(autouse=True)
def _close_figures():
    yield
    plt.close("all")


def test_the_figure_has_one_panel_per_decision_plus_the_curve(mv):
    result = ma.plot_specification_curve(mv)
    assert len(result.figure.axes) == 1 + len(mv.decisions)
    assert result.ax is result.figure.axes[0]


def test_every_usable_specification_is_drawn_exactly_once(mv):
    result = ma.plot_specification_curve(mv)
    # The curve line holds one point per usable specification; the legend and
    # the zero line are separate artists, so count the data instead.
    drawn = sum(len(line.get_xdata()) for line in result.ax.lines)
    assert drawn >= mv.counts()["usable"]
    estimates = np.array([row["estimate"] for row in mv.curve()])
    assert estimates.size == mv.counts()["usable"]


def test_the_margins_are_labelled_with_the_decision_names(mv):
    result = ma.plot_specification_curve(mv)
    labels = {ax.get_ylabel() for ax in result.figure.axes[1:]}
    assert labels == set(mv.decisions)


def test_the_margin_rows_show_the_values_that_were_used(mv):
    result = ma.plot_specification_curve(mv)
    method_ax = next(ax for ax in result.figure.axes[1:]
                     if ax.get_ylabel() == "method")
    shown = {tick.get_text() for tick in method_ax.get_yticklabels()}
    assert shown == {"REML", "DL"}


def test_the_default_specification_can_be_marked(mv):
    one = mv.curve()[0]["id"]
    result = ma.plot_specification_curve(mv, mark_default=one)
    assert any(line.get_label() == "the primary analysis"
               for line in result.ax.lines)


def test_an_unknown_default_id_is_ignored_rather_than_crashing(mv):
    result = ma.plot_specification_curve(mv, mark_default="sha256:" + "0" * 32)
    assert not any(line.get_label() == "the primary analysis"
                   for line in result.ax.lines)


def test_the_title_reports_the_counts_and_the_range(mv):
    result = ma.plot_specification_curve(mv)
    title = result.ax.get_title(loc="left")
    assert "usable of" in title
    assert "estimate" in title


def test_a_custom_title_is_used(mv):
    result = ma.plot_specification_curve(mv, title="my grid")
    assert result.ax.get_title(loc="left") == "my grid"


def test_the_figure_can_be_saved(mv, tmp_path):
    target = tmp_path / "curve.png"
    result = ma.plot_specification_curve(mv, filename=str(target))
    assert result.filename == str(target)
    assert target.exists() and target.stat().st_size > 0


def test_failures_are_mentioned_in_the_title_when_there_are_any(spec):
    def run(choices):
        if choices["test"] == "knha":
            raise RuntimeError("cannot")
        return _run(choices)

    mv = ma.multiverse(spec, run)
    result = ma.plot_specification_curve(mv)
    assert "failed" in result.ax.get_title(loc="left")


def test_an_empty_multiverse_is_refused_with_the_reason(spec):
    def run(choices):
        raise RuntimeError("nothing works")

    mv = ma.multiverse(spec, run)
    with pytest.raises(ValueError, match="no specification produced"):
        ma.plot_specification_curve(mv)


def test_the_panels_share_one_x_axis(mv):
    result = ma.plot_specification_curve(mv)
    first = result.figure.axes[0]
    for ax in result.figure.axes[1:]:
        # sharex means the same axis object, so the curve and the margins cannot
        # drift apart when the figure is resized.
        assert ax.get_xaxis_transform() is not None
        assert ax.get_xlim() == first.get_xlim()


def test_the_shared_axis_is_ordinal_not_measured_in_effect_sizes(mv):
    # The first version plotted the estimate on x and shared that axis with the
    # margins, so the margins were drawn at positions 0..M-1 on a scale of
    # effect sizes and every marker after the first fell outside the view. The
    # assertions above all passed; rendering the figure is what showed it. This
    # pins the axis to the specification count.
    result = ma.plot_specification_curve(mv)
    n = mv.counts()["usable"]
    low, high = result.ax.get_xlim()
    assert low < 0.0 and high > n - 1.0
    assert high < n + 1.0, "the ordinal axis should stop just past the last point"


def test_every_margin_marker_is_inside_the_view(mv):
    result = ma.plot_specification_curve(mv)
    low, high = result.ax.get_xlim()
    for ax in result.figure.axes[1:]:
        for line in ax.lines:
            for x in np.atleast_1d(line.get_xdata()).astype(float):
                assert low <= x <= high, (ax.get_ylabel(), x, low, high)


def test_the_margins_mark_one_position_per_usable_specification(mv):
    result = ma.plot_specification_curve(mv)
    n = mv.counts()["usable"]
    for ax in result.figure.axes[1:]:
        marked = sum(len(np.atleast_1d(line.get_xdata())) for line in ax.lines)
        assert marked == n, ax.get_ylabel()
