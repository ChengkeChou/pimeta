"""metainf / metacum (meta::metainf / meta::metacum).

Both repeatedly refit :func:`.metagen` on subsets of the studies:

- ``metainf``  leave-one-out influence: for each study ``i``, refit without it
  and record the pooled estimate, SE, CI, statistic, I^2 and tau^2.
- ``metacum``  cumulative (sequential) meta-analysis: for each prefix of the
  studies (in the supplied order, optionally sorted by ``sortvar``), record the
  pooled estimate, SE, CI, statistic, I^2 and tau^2.

The heterogeneity statistics (``I2``, ``tau2``, ``Q``) follow meta's
Q-based definition, exactly as :func:`.metagen` computes them.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from scipy import stats

from .metagen import MetagenResult, metagen
from .rma_list import RowSlicable


# --------------------------------------------------------------------------- #
# result objects
# --------------------------------------------------------------------------- #
@dataclass
class MetainfResult:
    """``meta::metainf`` — one row per study (that study left out)."""

    studlab: list[str]
    TE: list[float]
    seTE: list[float]
    lower: list[float]
    upper: list[float]
    statistic: list[float]
    p: list[float]
    I2: list[float]
    tau2: list[float]
    Q: list[float]
    df: list[int]
    k: int
    pooled: str

    def to_dict(self) -> dict:
        return {
            "studlab": self.studlab,
            "TE": self.TE,
            "seTE": self.seTE,
            "lower": self.lower,
            "upper": self.upper,
            "statistic": self.statistic,
            "p": self.p,
            "I2": self.I2,
            "tau2": self.tau2,
            "Q": self.Q,
            "df": self.df,
            "k": self.k,
            "pooled": self.pooled,
        }


@dataclass
class MetacumResult(RowSlicable):
    """``meta::metacum`` — one row per cumulative prefix of studies."""

    studlab: list[str]
    TE: list[float]
    seTE: list[float]
    lower: list[float]
    upper: list[float]
    statistic: list[float]
    p: list[float]
    I2: list[float]
    tau2: list[float]
    Q: list[float]
    df: list[int]
    k: int
    pooled: str

    def to_dict(self) -> dict:
        return {
            "studlab": self.studlab,
            "TE": self.TE,
            "seTE": self.seTE,
            "lower": self.lower,
            "upper": self.upper,
            "statistic": self.statistic,
            "p": self.p,
            "I2": self.I2,
            "tau2": self.tau2,
            "Q": self.Q,
            "df": self.df,
            "k": self.k,
            "pooled": self.pooled,
        }


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def _pick(m: MetagenResult, pooled: str):
    """Return (TE, seTE, lower, upper, statistic) for the requested model."""
    if pooled == "common":
        return (
            m.TE_common, m.seTE_common, m.lower_common, m.upper_common,
            m.statistic_common,
        )
    return (
        m.TE_random, m.seTE_random, m.lower_random, m.upper_random,
        m.statistic_random,
    )


def _p_from_stat(stat: float) -> float:
    return float(2.0 * stats.norm.sf(abs(stat)))


# --------------------------------------------------------------------------- #
# metainf
# --------------------------------------------------------------------------- #
def metainf(
    TE,
    seTE,
    studlab=None,
    sm: str = "",
    method_tau: str = "DL",
    pooled: str = "random",
    level: float = 0.95,
) -> MetainfResult:
    """Leave-one-out influence analysis (``meta::metainf``).

    Args:
        TE, seTE: analysis-scale effects and SEs.
        studlab: study labels.
        sm: summary measure (recorded only).
        method_tau: tau^2 estimator (default DL).
        pooled: ``"random"`` (default) or ``"common"``.
        level: confidence level.
    """
    TE = np.asarray(TE, dtype=float).ravel()
    seTE = np.asarray(seTE, dtype=float).ravel()
    k = len(TE)
    if k < 2:
        raise ValueError("metainf requires at least two studies")

    if studlab is None:
        studlab = [f"Study {i + 1}" for i in range(k)]
    studlab = list(studlab)

    out_TE, out_se, out_lo, out_up = [], [], [], []
    out_stat, out_p = [], []
    out_i2, out_tau2, out_q, out_df = [], [], [], []

    for i in range(k):
        idx = [j for j in range(k) if j != i]
        m = metagen(
            TE[idx].tolist(), seTE[idx].tolist(),
            studlab=[studlab[j] for j in idx],
            sm=sm, method_tau=method_tau, level=level,
        )
        te, se, lo, up, stat = _pick(m, pooled)
        out_TE.append(float(te))
        out_se.append(float(se))
        out_lo.append(float(lo))
        out_up.append(float(up))
        out_stat.append(float(stat))
        out_p.append(_p_from_stat(float(stat)))
        out_i2.append(float(m.I2))
        out_tau2.append(float(m.tau2))
        out_q.append(float(m.Q))
        out_df.append(int(m.df_Q))

    return MetainfResult(
        studlab=studlab,
        TE=out_TE, seTE=out_se, lower=out_lo, upper=out_up,
        statistic=out_stat, p=out_p,
        I2=out_i2, tau2=out_tau2, Q=out_q, df=out_df,
        k=k, pooled=pooled,
    )


# --------------------------------------------------------------------------- #
# metacum
# --------------------------------------------------------------------------- #
def metacum(
    TE,
    seTE,
    studlab=None,
    sm: str = "",
    method_tau: str = "DL",
    pooled: str = "random",
    sortvar=None,
    level: float = 0.95,
) -> MetacumResult:
    """Cumulative meta-analysis (``meta::metacum``).

    Args:
        TE, seTE: analysis-scale effects and SEs.
        studlab: study labels.
        sm: summary measure (recorded only).
        method_tau: tau^2 estimator (default DL).
        pooled: ``"random"`` (default) or ``"common"``.
        sortvar: optional numeric vector to sort the studies before cumulating
            (e.g. publication year).
        level: confidence level.
    """
    TE = np.asarray(TE, dtype=float).ravel()
    seTE = np.asarray(seTE, dtype=float).ravel()
    k = len(TE)
    if k == 0:
        raise ValueError("no studies supplied")

    if studlab is None:
        studlab = [f"Study {i + 1}" for i in range(k)]
    studlab = list(studlab)

    order = list(range(k))
    if sortvar is not None:
        sv = np.asarray(sortvar, dtype=float).ravel()
        order = sorted(range(k), key=lambda i: (sv[i], i))

    out_TE, out_se, out_lo, out_up = [], [], [], []
    out_stat, out_p = [], []
    out_i2, out_tau2, out_q, out_df = [], [], [], []
    out_studlab = []

    for n in range(1, k + 1):
        idx = order[:n]
        m = metagen(
            TE[idx].tolist(), seTE[idx].tolist(),
            studlab=[studlab[j] for j in idx],
            sm=sm, method_tau=method_tau, level=level,
        )
        te, se, lo, up, stat = _pick(m, pooled)
        out_TE.append(float(te))
        out_se.append(float(se))
        out_lo.append(float(lo))
        out_up.append(float(up))
        out_stat.append(float(stat))
        out_p.append(_p_from_stat(float(stat)) if not math.isnan(float(stat)) else math.nan)
        out_i2.append(float(m.I2))
        out_tau2.append(float(m.tau2))
        out_q.append(float(m.Q))
        out_df.append(int(m.df_Q))
        out_studlab.append(f"Adding {studlab[idx[-1]]} (k={n})")

    return MetacumResult(
        studlab=out_studlab,
        TE=out_TE, seTE=out_se, lower=out_lo, upper=out_up,
        statistic=out_stat, p=out_p,
        I2=out_i2, tau2=out_tau2, Q=out_q, df=out_df,
        k=k, pooled=pooled,
    )
