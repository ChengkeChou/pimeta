"""Evidence about a subgroup difference, not just a p-value.

An interaction p-value above 0.05 means "the data do not rule out no
difference". It does not distinguish "the subgroups are the same" from "the
subgroups were too small to tell", and those are different findings with
different consequences. Two closed-form summaries separate them, and both are
computed from the same two numbers an interaction test already produces.

Bayes factor for a point null
----------------------------
With ``Delta_hat ~ N(Delta, se^2)`` and a normal prior ``Delta ~ N(0, s0^2)``,
the marginal likelihoods give

.. math::

    \\mathrm{BF}_{10} = \\sqrt{\\frac{se^2}{se^2 + s_0^2}}
        \\exp\\left(\\frac{\\hat\\Delta^2}{2\\,se^2}
        \\frac{s_0^2}{se^2 + s_0^2}\\right)

which is exact, needs no sampling, and is the ratio of the two models' predictive
densities at the observed estimate. ``BF10 = 0.3`` is evidence *for* equality,
``BF10 = 3`` evidence for a difference, and anything in between is "the data do
not decide" -- which is a statement a p-value cannot make at all.

The prior scale matters and is reported rather than hidden: ``s0`` is the size of
subgroup difference that was considered plausible *before* seeing the data, and
the default (0.5 on the effect scale) is a default, not a finding.

Equivalence proportion
----------------------
``P(|Delta| < mcid | data)``, the posterior mass inside the region that would
count as clinically irrelevant. Unlike a TOST p-value it says how confident the
equivalence is, on the same scale as a confidence level, and it degrades
gracefully: with a wide ``se`` it heads to 0 rather than to a verdict.
"""

from __future__ import annotations

import math

import numpy as np
from scipy import stats

__all__ = ["bayes_factor_normal", "subgroup_evidence"]

#: Jeffreys' thresholds. The band between 1/3 and 3 is "the data do not
#: decide", which is a verdict rather than an absence of one.
_EVIDENCE = (
    (100.0, "extreme evidence for a difference"),
    (30.0, "very strong evidence for a difference"),
    (10.0, "strong evidence for a difference"),
    (3.0, "moderate evidence for a difference"),
)
_EQUIVALENCE = (
    (10.0, "moderate evidence for equivalence"),
    (100.0, "strong evidence for equivalence"),
)


def bayes_factor_normal(est: float, se: float, *, prior_sd: float = 0.5
                        ) -> float:
    """``BF10`` for ``Delta = 0`` against ``Delta ~ N(0, prior_sd^2)``.

    Args:
        est: the observed subgroup difference.
        se: its standard error.
        prior_sd: the plausible size of the difference under the alternative.

    Returns:
        The Bayes factor. Above 1 supports a difference, below 1 supports
        equality.

    Raises:
        ValueError: a non-positive standard error or prior scale.

    Example:
        >>> round(bayes_factor_normal(0.0, 0.2), 3)
        0.371
        >>> round(bayes_factor_normal(0.4, 0.2), 3)
        2.083
    """
    if not (se > 0.0) or not np.isfinite(se):
        raise ValueError("'se' must be finite and positive")
    if not (prior_sd > 0.0) or not np.isfinite(prior_sd):
        raise ValueError("'prior_sd' must be finite and positive")
    est = float(est)
    ratio = se * se / (se * se + prior_sd * prior_sd)
    exponent = (est * est / (2.0 * se * se)) * (prior_sd * prior_sd
                                                / (se * se + prior_sd * prior_sd))
    return float(math.sqrt(ratio) * math.exp(exponent))


def _verdict(bf: float) -> str:
    for threshold, text in _EVIDENCE:
        if bf >= threshold:
            return text
    if bf > 1.0 / 3.0:
        return "no evidence either way"
    inverse = 1.0 / bf if bf > 0.0 else float("inf")
    for threshold, text in _EQUIVALENCE:
        if inverse >= threshold:
            return text
    return "very strong evidence for equivalence"


def subgroup_evidence(est: float, se: float, *, df: float | None = None,
                      prior_sd: float = 0.5, mcid: float | None = None,
                      level: float = 0.95) -> dict:
    """Everything that can be said about a subgroup difference, in closed form.

    Args:
        est: the observed difference between the subgroup effects.
        se: its standard error.
        df: degrees of freedom, when the interval was built with a ``t``
            reference. ``None`` uses the normal.
        prior_sd: prior scale for the Bayes factor; see
            :func:`bayes_factor_normal`.
        mcid: the smallest difference that would matter. When given, the
            equivalence proportion is reported; when not, it is ``None`` rather
            than a default nobody chose.
        level: confidence level for the interval, as a proportion or percentage.

    Returns:
        Dictionary with ``est``, ``se``, ``df``, ``ci_lb``, ``ci_ub``, ``bf10``,
        ``bf01``, ``evidence`` (a sentence), ``prior_sd``,
        ``prob_wrong_sign`` (the posterior mass on the opposite sign to the
        estimate), and ``prob_equivalence`` (``None`` without ``mcid``).

    Example:
        >>> out = subgroup_evidence(0.35, 0.16, mcid=0.2)
        >>> out["bf10"] > 1 and 0.0 <= out["prob_equivalence"] <= 1.0
        True
    """
    from .levels import level_to_alpha

    alpha = level_to_alpha(level)
    dist = stats.norm if df is None else stats.t
    crit = (float(dist.ppf(1.0 - alpha / 2.0)) if df is None
            else float(dist.ppf(1.0 - alpha / 2.0, df)))
    scale = float(se)
    if not (scale > 0.0):
        raise ValueError("'se' must be positive")
    bf10 = bayes_factor_normal(est, scale, prior_sd=prior_sd)
    if mcid is None:
        prob_equivalence = None
    else:
        if not (mcid > 0.0):
            raise ValueError("'mcid' must be positive")
        if df is None:
            prob_equivalence = float(stats.norm.cdf((mcid - est) / scale)
                                     - stats.norm.cdf((-mcid - est) / scale))
        else:
            prob_equivalence = float(stats.t.cdf((mcid - est) / scale, df)
                                     - stats.t.cdf((-mcid - est) / scale, df))
    if df is None:
        prob_wrong_sign = float(stats.norm.cdf(-abs(est) / scale))
    else:
        prob_wrong_sign = float(stats.t.cdf(-abs(est) / scale, df))
    return {
        "est": float(est),
        "se": scale,
        "df": None if df is None else float(df),
        "ci_lb": float(est) - crit * scale,
        "ci_ub": float(est) + crit * scale,
        "bf10": bf10,
        "bf01": 1.0 / bf10 if bf10 > 0.0 else float("inf"),
        "evidence": _verdict(bf10),
        "prior_sd": float(prior_sd),
        "mcid": None if mcid is None else float(mcid),
        "prob_wrong_sign": prob_wrong_sign,
        "prob_equivalence": prob_equivalence,
    }
