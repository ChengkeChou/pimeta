"""``netmeta::netmeasures`` — direct-evidence proportions.

Reports how much of each network estimate comes from direct rather than
indirect evidence, and the heterogeneity of the direct evidence.

Part of the network meta-analysis implementation; :mod:`pimeta.netmeta`
re-exports these names and remains the public entry point.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:                       # pragma: no cover - typing only
    pass

__all__ = ["NetmeasuresResult", "netmeasures"]


# --------------------------------------------------------------------------- #
# netmeta::netmeasures
# --------------------------------------------------------------------------- #

class NetmeasuresResult:
    """Result of :func:`netmeasures` (``netmeta::netmeasures``)."""

    __slots__ = ("proportion", "meanpath", "minpar", "minpar_study",
                 "H_tilde", "random", "tau_preset")

    def __init__(self, proportion, meanpath, minpar, minpar_study,
                 H_tilde, random, tau_preset):
        self.proportion = proportion
        self.meanpath = meanpath
        self.minpar = minpar
        self.minpar_study = minpar_study
        self.H_tilde = H_tilde
        self.random = random
        self.tau_preset = tau_preset

    def __repr__(self):
        return (f"NetmeasuresResult(k={len(self.proportion)}, "
                f"random={self.random})")

    def as_dict(self):
        """R-style ``$`` access."""
        return {
            "proportion": self.proportion,
            "meanpath": self.meanpath,
            "minpar": self.minpar,
            "minpar.study": self.minpar_study,
            "H.tilde": self.H_tilde,
            "random": self.random,
            "tau.preset": self.tau_preset,
        }


def _htilde(H, groups):
    """``H.tilde`` — absolute-sum aggregation within design (or study) blocks.

    R::

        l <- lapply(split(H, <block index>), function(x) matrix(x, nrow(H)))
        H.tilde <- sapply(l, function(x) apply(x, 1, function(x)
                       0.5 * (abs(sum(x)) + sum(abs(x)))))
    """
    order = list(dict.fromkeys(groups))
    out = np.zeros((H.shape[0], len(order)))
    for j, g in enumerate(order):
        cols = [r for r, v in enumerate(groups) if v == g]
        blk = H[:, cols]
        out[:, j] = 0.5 * (np.abs(blk.sum(axis=1)) + np.abs(blk).sum(axis=1))
    return out, order


def netmeasures(result, random=None, tau_preset=None):
    """``netmeta::netmeasures`` — network path measures (Koenig et al. 2013).

    Args:
        result: a :class:`NetmetaResult`.
        random: use the random-effects model.  Defaults to ``True``; forced to
            ``True`` when ``tau_preset`` is given.
        tau_preset: fixed heterogeneity variance.

    Returns:
        :class:`NetmeasuresResult` with ``proportion`` (share of the network
        estimate's variance that is direct evidence), ``meanpath``, ``minpar``
        and ``minpar_study``.
    """
    from ..netmeta import NetmetaResult, _nma_krahn_full  # local import: cycle

    if not isinstance(result, NetmetaResult):
        raise TypeError("argument 'result' must be a NetmetaResult")

    if random is None:
        random = True
    if tau_preset is not None:
        tau_preset = float(tau_preset)
        if tau_preset < 0:
            raise ValueError("argument 'tau_preset' must be non-negative")
        random = True

    if random:
        t2 = result.tau if tau_preset is None else tau_preset
        tau_preset_out = result.tau if tau_preset is None else tau_preset
    else:
        t2 = 0.0
        tau_preset_out = None

    nmak = _nma_krahn_full(
        result.TE_raw, result.seTE_raw, result.treat1_raw, result.treat2_raw,
        result.studlab_raw, result.narms_raw, list(result.trts),
        "", float(t2),
    )

    if nmak["n"] == 2:
        designs = list(dict.fromkeys(d["design"] for d in nmak["design"]))
        prop = {d: 1.0 for d in designs}
        return NetmeasuresResult(prop, dict(prop), None, None, None,
                                 random, tau_preset_out)

    H = nmak["H"]
    x_obs_s = nmak["x_obs_studies"]
    v_studies = nmak["V_studies"]
    v_studies_inv = np.linalg.pinv(v_studies)
    H_studies = (nmak["X_full"]
                 @ np.linalg.pinv(x_obs_s.T @ v_studies_inv @ x_obs_s)
                 @ x_obs_s.T
                 @ v_studies_inv)

    poss = list(nmak["trts_poss"])
    proportion = {name: 0.0 for name in poss}
    network = nmak["network"]
    for d in nmak["direct"]:
        comp = d["comparison"]
        if comp in proportion and not math.isnan(d["seTE"]):
            proportion[comp] = (network[comp][1] ** 2) / (d["seTE"] ** 2)

    H_tilde, designs = _htilde(H, nmak["design_col"])
    H_tilde_s, studies = _htilde(H_studies, nmak["studlab_studies"])

    meanpath = {n: float(H_tilde[i, :].sum()) for i, n in enumerate(poss)}
    minpar = {}
    for i, n in enumerate(poss):
        mx = float(np.abs(H_tilde[i, :]).max())
        minpar[n] = float("inf") if mx == 0.0 else 1.0 / mx
    minpar_study = {}
    for i, n in enumerate(poss):
        mx = float(np.abs(H_tilde_s[i, :]).max())
        minpar_study[n] = float("inf") if mx == 0.0 else 1.0 / mx

    return NetmeasuresResult(
        proportion=proportion,
        meanpath=meanpath,
        minpar=minpar,
        minpar_study=minpar_study,
        H_tilde={"rows": poss, "cols": designs, "values": H_tilde},
        random=random,
        tau_preset=tau_preset_out,
    )
