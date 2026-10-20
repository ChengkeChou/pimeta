"""``profile.rma.uni.selmodel`` and its figure.

``profile_selmodel`` refits the selection model on a grid of fixed ``delta``
values and inverts the likelihood-ratio test to get a profile interval.  Every
grid point is a full ``selmodel`` fit (several seconds each), so the grid is
computed once per module and the cheaper helpers are asserted against it.
"""

import numpy as np
import pytest

import pimeta as ma

YI = [0.1, -0.2, 0.05, -0.15, 0.3, 0.25, 0.12, -0.05]
VI = [0.01, 0.0144, 0.0121, 0.0169, 0.0196, 0.0225, 0.01, 0.0121]


@pytest.fixture(scope="module")
def sel():
    return ma.selmodel(ma.rma(YI, vi=VI), type="beta")


@pytest.fixture(scope="module")
def prof(sel):
    return ma.profile_selmodel(sel, delta=[0.3, 0.6, 0.9])


def test_profile_grid_and_statistics(prof):
    assert prof.delta == [0.3, 0.6, 0.9]
    assert len(prof.ll) == 3
    assert all(np.isfinite(prof.ll))
    assert prof.maxll == pytest.approx(max(prof.ll))
    assert prof.type == "beta"
    assert prof.k == 8


def test_profile_delta_hat_is_the_fitted_value(sel, prof):
    assert prof.delta_hat == pytest.approx(float(sel.delta[0]))
    # the likelihood is highest near the estimate
    assert abs(prof.delta[int(np.argmax(prof.ll))] - prof.delta_hat) <= 0.6


def test_profile_round_trips_to_a_dict(prof):
    d = prof.to_dict()
    assert set(d) >= {"delta", "ll", "maxll", "delta_hat", "ci_lb", "ci_ub"}
    assert "SelProfileResult" in repr(prof)


def test_profile_default_grid(sel):
    p = ma.profile_selmodel(sel, n_points=3)
    assert len(p.delta) == 3
    assert p.delta[0] < p.delta_hat < p.delta[-1]


def test_profile_rejects_a_non_selection_model():
    with pytest.raises(TypeError, match="SelModelResult"):
        ma.profile_selmodel(ma.rma(YI, vi=VI))


def test_profile_rma_routes_a_selection_model(sel):
    assert isinstance(ma.profile_rma(sel, n_points=2), ma.SelProfileResult)


def test_profile_rma_still_profiles_tau2():
    assert ma.profile_rma(ma.rma(YI, vi=VI), n_points=4).method == "REML"


def test_plot_selprofile(prof, tmp_path):
    path = tmp_path / "prof.png"
    out = ma.plot_selprofile(prof, filename=str(path))
    assert path.exists() and path.stat().st_size > 0
    assert out.ax.get_xlabel() == "delta"


def test_plot_selprofile_needs_a_finite_likelihood():
    class Bare:
        delta = [0.0, 1.0]
        ll = [float("nan"), float("nan")]
        maxll = float("nan")

    with pytest.raises(ValueError, match="no finite log-likelihood"):
        ma.plot_selprofile(Bare())


def test_plot_which_profile_for_a_selection_model(sel):
    out = ma.plot(sel, which="profile", n_points=2)
    assert out.ax.get_xlabel() == "delta"


def test_plot_which_profile_for_a_plain_model():
    assert ma.plot(ma.rma(YI, vi=VI), which="profile").figure is not None
