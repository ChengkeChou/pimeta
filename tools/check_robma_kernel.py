"""Rung 1: check the selection kernel with the weights held fixed.

With omega fixed there is no prior on the weights and no parameterisation to
disagree about, so any difference from the reference is in the kernel itself.

Targets, from ``tools/reference_robma_bsel.R`` (3 chains, ESS > 9000):

===============  ==================  ==================
model            mu                  tau
===============  ==================  ==================
one-sided        0.236 (0.067)       0.080 (0.060)
two-sided        0.248 (0.062)       0.072 (0.053)
===============  ==================  ==================
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

import pimeta as ma
from pimeta.robma import DEFAULT_GRID as GRID
from pimeta.robma import DEFAULTS as PRIOR
from pimeta.robma import _fit_selection

ROOT = Path(__file__).resolve().parents[1]
TARGETS = {
    "one-sided": (0.236, 0.067, 0.080, 0.060),
    "two-sided": (0.248, 0.062, 0.072, 0.053),
}


def summarise(nodes, logw):
    """Mean, SD and 95% interval of a grid posterior."""
    w = np.exp(logw - np.max(logw))
    w = w / w.sum()
    order = np.argsort(nodes)
    x, p = nodes[order], w[order]
    mean = float(np.sum(x * p))
    sd = float(np.sqrt(np.sum((x - mean) ** 2 * p)))
    cdf = np.cumsum(p)
    return (mean, sd, float(np.interp(0.025, cdf, x)),
            float(np.interp(0.975, cdf, x)))


def main() -> None:
    d = pd.read_csv(ROOT / "tests" / "data" / "robma.csv")
    yi = d["yi"].to_numpy(float)
    sei = d["sei"].to_numpy(float)

    print(f"{'model':<12}{'quantity':<8}{'pimeta':>22}"
          f"{'reference':>22}")
    for side, target in TARGETS.items():
        wf = ma.step_weight_function(side, [0.05])
        fit = _fit_selection(yi, sei, wf=wf, effect_free=True,
                             hetero_free=True, prior=PRIOR, grid=GRID,
                             omega_fixed=(1.0, 0.5))
        mu = summarise(fit.mu_grid_nodes, fit.mu_grid_logw)
        tau = summarise(fit.tau_nodes, fit.tau_logw)
        ref_mu = f"{target[0]:0.3f} ({target[1]:0.3f})"
        ref_tau = f"{target[2]:0.3f} ({target[3]:0.3f})"
        print(f"{side:<12}{'mu':<8}"
              f"{mu[0]:>10.3f} ({mu[1]:0.3f})  [{mu[2]:0.3f},{mu[3]:0.3f}]"
              f"{ref_mu:>22}")
        print(f"{'':<12}{'tau':<8}"
              f"{tau[0]:>10.3f} ({tau[1]:0.3f})  [{tau[2]:0.3f},{tau[3]:0.3f}]"
              f"{ref_tau:>22}")


if __name__ == "__main__":
    main()
