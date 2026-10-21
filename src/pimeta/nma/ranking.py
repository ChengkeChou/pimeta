"""VIKOR multicriteria ranking (``netmeta::vikor``).

Split out of :mod:`pimeta.netmeta`; the public name is re-exported there.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:  # pragma: no cover - typing only
    pass


class VikorResult:
    """Result of :func:`vikor` — VIKOR ranking statistics."""

    __slots__ = ("Q", "S", "R", "trts", "weights", "v", "dropped")

    def __init__(self, Q, S, R, trts, weights, v, dropped):
        self.Q = Q
        self.S = S
        self.R = R
        self.trts = trts
        self.weights = weights
        self.v = v
        self.dropped = dropped

    def __repr__(self):
        return f"VikorResult(n={len(self.trts)})"

    def as_dict(self):
        return {"Q": self.Q, "S": self.S, "R": self.R}

    def to_frame(self):
        try:
            import pandas as pd
        except ImportError:  # pragma: no cover
            return [{"treat": t, "Q": self.Q[t], "S": self.S[t],
                     "R": self.R[t]} for t in self.trts]
        return pd.DataFrame({"Q": self.Q, "S": self.S, "R": self.R})


def vikor(x, weights=None, v=0.5):
    """``netmeta::vikor`` — VIKOR multicriteria ranking.

    ``x`` holds one score in ``[0, 1]`` per treatment and outcome (higher is
    better).  Equations (transcribed from ``netmeta:::vikor_internal``)::

        dist = 1 - x;  wnm = dist * weights
        S_i  = sum_j wnm_ij            (group utility)
        R_i  = max_j wnm_ij            (individual regret)
        Q_i  = v·(S_i - min S)/(max S - min S)
               + (1 - v)·(R_i - min R)/(max R - min R)

    Args:
        x: ``NetmetaResult`` (P-scores are used), ``pandas.DataFrame`` or a
            2-D array-like of scores; row names are the treatments.
        weights: outcome weights (default: equal).
        v: weight of the group-utility strategy (``0.5`` = balanced).

    Returns:
        :class:`VikorResult` sorted by increasing ``Q`` (best first).
    """
    from ..netmeta import NetmetaResult  # local import: cycle

    if isinstance(x, NetmetaResult):
        trts = list(x.trts)
        mat = np.column_stack([np.asarray(x.pscore_common, dtype=float),
                               np.asarray(x.pscore_random, dtype=float)])
    elif hasattr(x, "to_numpy"):
        trts = list(x.index)
        mat = np.asarray(x.to_numpy(), dtype=float)
    else:
        mat = np.asarray(x, dtype=float)
        if mat.ndim == 1:
            mat = mat.reshape(-1, 1)
        trts = [str(i + 1) for i in range(mat.shape[0])]

    n_out = mat.shape[1]
    if weights is None:
        weights = np.full(n_out, 1.0 / n_out)
    else:
        weights = np.asarray(weights, dtype=float).ravel()
        if weights.size != n_out:
            raise ValueError(
                "Number of weights is different from the number of outcomes."
            )
        if np.any(weights < 0):
            raise ValueError("weights must be non-negative")
        if np.isclose(weights.sum(), 0):
            raise ValueError("Sum of weights must be larger than 0.")
        if not np.isclose(weights.sum(), 1.0):
            weights = weights / weights.sum()

    dist = 1.0 - mat
    wnm = dist * weights

    dropped = [trts[i] for i in range(mat.shape[0])
               if np.isnan(wnm[i]).any()]
    keep = [i for i in range(mat.shape[0]) if i not in
            [j for j, t in enumerate(trts) if t in dropped]]
    if len(keep) < 2:
        raise ValueError(
            "VIKOR method requires at least two treatments with complete "
            "rankings."
        )
    sub = wnm[keep]
    S = sub.sum(axis=1)
    R = sub.max(axis=1)
    if np.isclose(R.min(), R.max()):
        raise ValueError(
            "VIKOR method not applicable as all values of ranking statistic R "
            "are identical."
        )
    if np.isclose(S.min(), S.max()):
        raise ValueError(
            "VIKOR method not applicable as all values of ranking statistic S "
            "are identical."
        )
    Q = (v * (S - S.min()) / (S.max() - S.min())
         + (1 - v) * (R - R.min()) / (R.max() - R.min()))

    names = [trts[i] for i in keep]
    order = np.argsort(Q, kind="stable")
    return VikorResult(
        Q={names[i]: float(Q[i]) for i in order},
        S={names[i]: float(S[i]) for i in order},
        R={names[i]: float(R[i]) for i in order},
        trts=[names[i] for i in order],
        weights=weights.tolist(), v=float(v), dropped=dropped,
    )
