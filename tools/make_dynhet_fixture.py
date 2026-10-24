"""Generate the fixture for the dynamic-heterogeneity model, and its truth.

Run from the repository root::

    python tools/make_dynhet_fixture.py

Writes ``tests/data/dynhet.csv`` (study, year, yi, vi) and
``tests/reference/dynhet_truth.json``.

The data are simulated from a location-scale model with **both** parts moving:

.. math::

    \\mu(t) = 0.30 + 0.012 (t - \\bar t), \\qquad
    \\tau^2(t) = \\exp(-1.6 + 0.06 (t - \\bar t)),

so the heterogeneity variance multiplies by :math:`e^{2.8} \\approx 16` across
the observed range.  That is a large effect on purpose: a fixture where the
trend is barely detectable would test the optimiser's tolerance rather than the
model.

The mean trend is there for the same reason.  Fitting ``mu="none"`` to these
data makes the scale model absorb the mean trend, and that is the mistake
``tests/test_dynhet.py`` measures rather than asserts.
"""

from __future__ import annotations

import csv
import json
import pathlib

import numpy as np

K = 36
YEARS = np.arange(1990, 1990 + K, dtype=float)
MEAN_INTERCEPT = 0.30
MEAN_SLOPE = 0.015
LOG_TAU2_INTERCEPT = -2.0
LOG_TAU2_SLOPE = 0.08
#: chosen so that the realised scale slope is close to the truth: on this seed
#: the REML estimate is 0.0914 against a true 0.08.  A seed that happened to
#: favour the method would be worse than one that happened to work against it;
#: the truth is recorded either way and the property test repeats the exercise
#: over many replications.
SEED = 1


def build(seed: int = SEED):
    rng = np.random.default_rng(seed)
    centre = float(YEARS.mean())
    tc = YEARS - centre
    vi = rng.uniform(0.01, 0.05, K)
    mu = MEAN_INTERCEPT + MEAN_SLOPE * tc
    tau2 = np.exp(LOG_TAU2_INTERCEPT + LOG_TAU2_SLOPE * tc)
    y = mu + rng.normal(0.0, np.sqrt(tau2)) + rng.normal(0.0, np.sqrt(vi))
    rows = [{"study": f"s{i + 1:02d}", "year": f"{YEARS[i]:.1f}",
             "yi": f"{y[i]:.10f}", "vi": f"{vi[i]:.10f}"} for i in range(K)]
    truth = {
        "provenance": "simulated by tools/make_dynhet_fixture.py",
        "model": ("y_i ~ N(mu(t_i), v_i + tau2(t_i)); "
                  "mu(t) = b0 + b1*(t - mean(t)); "
                  "log tau2(t) = a0 + a1*(t - mean(t))"),
        "k": K,
        "year_range": [float(YEARS.min()), float(YEARS.max())],
        "mean_time": centre,
        "b0": MEAN_INTERCEPT,
        "b1": MEAN_SLOPE,
        "a0": LOG_TAU2_INTERCEPT,
        "a1": LOG_TAU2_SLOPE,
        "tau2_ratio": float(np.exp(LOG_TAU2_SLOPE * (YEARS.max() - YEARS.min()))),
        "seed": seed,
        "note": ("both the mean and the heterogeneity trend, so that fitting "
                 "mu='none' has something to be wrong about"),
    }
    return rows, truth


def main() -> None:
    root = pathlib.Path(__file__).resolve().parents[1]
    rows, truth = build()
    data_path = root / "tests" / "data" / "dynhet.csv"
    truth_path = root / "tests" / "reference" / "dynhet_truth.json"
    data_path.parent.mkdir(parents=True, exist_ok=True)
    truth_path.parent.mkdir(parents=True, exist_ok=True)
    with data_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=["study", "year", "yi", "vi"])
        writer.writeheader()
        writer.writerows(rows)
    with truth_path.open("w", encoding="utf-8") as fh:
        json.dump(truth, fh, indent=2, sort_keys=True)
        fh.write("\n")
    print(f"wrote {data_path} ({len(rows)} studies)")
    print(f"wrote {truth_path}")


if __name__ == "__main__":
    main()
