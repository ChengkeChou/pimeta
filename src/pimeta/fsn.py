"""Fail-safe N (metafor::fsn).

Replicates the classic publication-bias fail-safe N statistics:

- ``Rosenthal``  Rosenthal's method (Stouffer-combined z, the default)
- ``Rosenberg``  Rosenberg's weighted method
- ``Orwin``      Orwin's method (target effect size)
- ``Binomial``   sign test based on the binomial distribution

Formulas are exact transcriptions of ``metafor::fsn.rma.uni``.
"""

from __future__ import annotations

import math
import sys
from dataclasses import dataclass

import numpy as np
from scipy import stats

_TYPES = {"Rosenthal", "Rosenberg", "Orwin", "Binomial"}


def _rnd_fsn(fsnum: float) -> float:
    """metafor's ``.rnd.fsn``: ceiling unless already (near-)integral."""
    if not math.isfinite(fsnum):
        return float(fsnum)
    if abs(fsnum - round(fsnum)) >= math.sqrt(sys.float_info.epsilon):
        return float(math.ceil(fsnum))
    return float(round(fsnum))


@dataclass
class FsnResult:
    """Fail-safe N result."""

    type: str
    k: int
    fsnum: float          # fail-safe N
    pval: float           # combined p-value (Rosenthal/Rosenberg/Binomial)
    est: float | None     # weighted mean (Orwin/Rosenberg)
    target: float | None  # target effect (Orwin)
    alpha: float

    def to_dict(self) -> dict:
        return {
            "type": self.type,
            "k": self.k,
            "fsnum": self.fsnum,
            "pval": self.pval,
            "est": self.est,
            "target": self.target,
            "alpha": self.alpha,
        }


def fsn(
    yi,
    vi=None,
    sei=None,
    type: str = "Rosenthal",
    alpha: float = 0.05,
    target: float | None = None,
) -> FsnResult:
    """Compute the fail-safe N (mirrors ``metafor::fsn``).

    Args:
        yi: per-study effect estimates.
        vi: per-study sampling variances (or ``sei`` standard errors).
        type: "Rosenthal" (default), "Rosenberg", "Orwin" or "Binomial".
        alpha: significance level (default 0.05).
        target: target effect size for Orwin (default est/2).
    """
    type_opts = {"rosenthal": "Rosenthal", "binomial": "Binomial",
                 "orwin": "Orwin", "rosenberg": "Rosenberg"}
    key = type_opts.get(type.lower())
    if key is None:
        raise ValueError(f"unknown type {type!r}")
    type = key

    yi = np.asarray(yi, dtype=float)
    k = len(yi)

    if vi is None:
        if sei is None:
            if type in ("Rosenthal", "Rosenberg"):
                raise ValueError("must specify vi or sei for this type")
            vi = np.ones(k)
        else:
            vi = np.asarray(sei, dtype=float) ** 2
    else:
        vi = np.asarray(vi, dtype=float)

    if len(vi) != k:
        raise ValueError("length of yi and vi differ")

    z_crit = stats.norm.isf(alpha)          # qnorm(alpha, lower.tail=FALSE)
    z_crit2 = stats.norm.isf(alpha / 2.0)   # qnorm(alpha/2, lower.tail=FALSE)

    if type == "Rosenthal":
        zi = yi / np.sqrt(vi)
        z_avg = abs(float(np.sum(zi)) / math.sqrt(k))
        pval = float(stats.norm.sf(z_avg))
        fsnum = _rnd_fsn(max(0.0, k * (z_avg / z_crit) ** 2 - k))
        est = target = None

    elif type == "Rosenberg":
        wi = 1.0 / vi
        est = float(np.sum(wi * yi) / np.sum(wi))
        zval = est / math.sqrt(1.0 / float(np.sum(wi)))
        pval = 2.0 * float(stats.norm.sf(abs(zval)))
        vt = 1.0 / float(np.mean(1.0 / vi))
        fsnum = _rnd_fsn(max(0.0, ((float(np.sum(wi * yi)) / z_crit2) ** 2 - float(np.sum(wi))) * vt))
        target = None

    elif type == "Orwin":
        wi = 1.0 / vi
        est = float(np.sum(wi * yi) / np.sum(wi))
        if target is None:
            target = est / 2.0
        if target == 0.0:
            fsnum = math.inf
        else:
            if np.sign(target) != np.sign(est):
                target = -target
            fsnum = max(0.0, k * (est - target) / target)
        fsnum = _rnd_fsn(fsnum)
        pval = math.nan

    else:  # Binomial
        kpos = int(np.sum(yi > 0))
        pval = float(stats.binomtest(kpos, k).pvalue)
        fsnum = 0.0
        if pval < alpha:
            pvalnew = pval
            while pvalnew < alpha:
                fsnum += 2
                pvalnew = float(stats.binomtest(int(kpos + fsnum / 2), int(k + fsnum)).pvalue)
        fsnum = _rnd_fsn(fsnum)
        est = target = None

    return FsnResult(
        type=type,
        k=k,
        fsnum=float(fsnum),
        pval=pval,
        est=est,
        target=target,
        alpha=alpha,
    )
