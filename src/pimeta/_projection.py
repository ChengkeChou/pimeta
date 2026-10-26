"""Quadratics of the weighted projection without forming it.

The estimators need three numbers from
``P = W - W X (X'WX)^-1 X'W`` with ``W = diag(w)``: the traces ``tr(P)`` and
``tr(P^2)``, and the quadratic forms ``y'Py`` and ``y'P^2y``. Forming ``P``
costs ``O(k^2)`` memory and ``O(k^2 p)`` time, and a ``k`` of 20000 is 3 GiB.

All four have closed forms in ``p x p`` matrices. They live here rather than in
:mod:`pimeta.rma_uni` because :mod:`pimeta.pooling` needs them too and already
imports this module, while ``rma_uni`` imports ``pooling``: keeping them in
``rma_uni`` would make a cycle.

Two numerical choices are worth recording.

``y'Py`` is computed as ``sum(w * r^2)`` with ``r = y - X beta`` rather than by
expanding the product. The expanded form subtracts two large nearly equal
numbers whenever the model fits well, and loses most of its digits there.

``y'P^2y`` is ``sum(w^2 * r^2)`` because ``P y = W r`` exactly: ``P`` is a
weighted residual-maker, so squaring it just squares the weights.
"""

from __future__ import annotations

import numpy as np

__all__ = [
    "quad_form",
    "quad_form_squared",
    "trace_projection",
    "trace_projection_squared",
]


def _wtm(X: np.ndarray, w: np.ndarray) -> np.ndarray:
    return X.T @ (w[:, None] * X)


def _residuals(X: np.ndarray, w: np.ndarray, y: np.ndarray) -> np.ndarray:
    stXWX = np.linalg.inv(_wtm(X, w))
    beta = stXWX @ (X.T @ (w * y))
    return y - X @ beta


def quad_form(X: np.ndarray, w: np.ndarray, y: np.ndarray) -> float:
    """``y' P y``."""
    r = _residuals(X, w, y)
    return float(np.sum(w * r * r))


def quad_form_squared(X: np.ndarray, w: np.ndarray, y: np.ndarray) -> float:
    """``y' P^2 y``, which is ``sum(w^2 r^2)`` because ``P y = W r``."""
    r = _residuals(X, w, y)
    return float(np.sum(w * w * r * r))


def trace_projection(X: np.ndarray, w: np.ndarray) -> float:
    """``tr(P) = sum(w) - tr((X'WX)^-1 X'W^2 X)``."""
    stXWX = np.linalg.inv(_wtm(X, w))
    return float(np.sum(w) - np.trace(stXWX @ _wtm(X, w * w)))


def trace_projection_squared(X: np.ndarray, w: np.ndarray) -> float:
    """``tr(P^2) = sum(w^2) - 2 tr(M X'W^3X) + tr(B^2)``, ``B = M X'W^2X``."""
    stXWX = np.linalg.inv(_wtm(X, w))
    B = stXWX @ _wtm(X, w * w)
    return float(np.sum(w * w) - 2.0 * np.trace(stXWX @ _wtm(X, w * w * w))
                 + np.trace(B @ B))
