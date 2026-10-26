"""The O(kp) trace and quadratic-form helpers must equal the k-by-k matrices.

``_proj`` builds ``P = W - W X (X'WX)^-1 X'W`` explicitly, which is O(k^2) in
memory. Everything the estimators need from ``P`` is a trace, or ``y'Py``, and
those are computed from p-by-p matrices instead. That is a real speed-up and a
real opportunity to be subtly wrong, so the two forms are compared here on
random data rather than trusted because the algebra looks right.

The comparison uses the *uncentred* quadratic form; the estimators pass centred
data. Both forms are compared on the same input, so the choice does not matter
for the test's purpose.
"""

from __future__ import annotations

import numpy as np
import pytest

from pimeta.rma_uni import _proj, _quad_form, _trace_P, _trace_P2


def _data(k: int, p: int, seed: int = 0):
    rng = np.random.default_rng(seed)
    X = np.column_stack([np.ones(k), rng.normal(size=(k, p - 1))])
    w = 1.0 / (0.02 + 0.1 * rng.random(k))
    y = rng.normal(0.3, 0.5, k)
    return X, w, y


@pytest.mark.parametrize("k,p", [(5, 1), (6, 2), (20, 1), (20, 3), (50, 4)])
def test_quadratic_form_matches_the_explicit_projection(k, p):
    X, w, y = _data(k, p)
    P = _proj(X, w)
    assert _quad_form(X, w, y) == pytest.approx(float(y @ P @ y), rel=1e-10)


@pytest.mark.parametrize("k,p", [(5, 1), (6, 2), (20, 1), (20, 3), (50, 4)])
def test_trace_matches_the_explicit_projection(k, p):
    X, w, _ = _data(k, p)
    P = _proj(X, w)
    assert _trace_P(X, w) == pytest.approx(float(np.trace(P)), rel=1e-10)


@pytest.mark.parametrize("k,p", [(5, 1), (6, 2), (20, 1), (20, 3), (50, 4)])
def test_trace_of_the_square_matches_the_explicit_projection(k, p):
    X, w, _ = _data(k, p)
    P = _proj(X, w)
    assert _trace_P2(X, w) == pytest.approx(float(np.sum(P * P)), rel=1e-9)


def test_the_projection_is_idempotent_in_the_weights_it_should_be():
    # P X = 0 for any weights, which is the property the traces rely on.
    X, w, _ = _data(12, 2)
    P = _proj(X, w)
    assert np.allclose(P @ X, 0.0, atol=1e-10)


def test_the_quadratic_form_does_not_form_a_k_by_k_matrix():
    # A k of 4000 with an explicit k-by-k projection allocates 128 MB; the
    # helper must stay small enough to be a rounding error. The threshold is
    # loose because it is a shape check, not a benchmark.
    k = 4000
    rng = np.random.default_rng(1)
    X = np.column_stack([np.ones(k), rng.normal(size=k)])
    w = 1.0 / (0.02 + 0.1 * rng.random(k))
    y = rng.normal(0.3, 0.5, k)
    import tracemalloc

    tracemalloc.start()
    value = _quad_form(X, w, y)
    peak = tracemalloc.get_traced_memory()[1]
    tracemalloc.stop()
    assert np.isfinite(value)
    assert peak < 4 * 1024 * 1024, f"peak {peak / 1024:.0f} KB suggests a k-by-k matrix"


def test_the_estimators_still_agree_with_a_hand_computed_dl():
    # DL in closed form, written out independently of the module's helpers:
    # tau2 = (Q - (k-1)) / (sum(w) - sum(w^2)/sum(w)).
    import pimeta as ma

    y = np.array([0.10, 0.25, 0.40, 0.55, 0.70])
    v = np.array([0.04, 0.05, 0.06, 0.03, 0.08])
    w = 1.0 / v
    k = y.size
    ybar = float(np.sum(w * y) / np.sum(w))
    Q = float(np.sum(w * (y - ybar) ** 2))
    tau2 = max(0.0, (Q - (k - 1)) / (np.sum(w) - np.sum(w ** 2) / np.sum(w)))
    fit = ma.rma(y, vi=v, method="DL")
    assert fit.tau2 == pytest.approx(tau2, rel=1e-12)
