"""Generate the arm-level fixture for the Bayesian NMA, and its truth.

Run from the repository root::

    python tools/make_nma_bayes_fixture.py

Writes ``tests/data/nma_bayes.csv`` (arm-level: study, treat, yi, sei) and
``tests/reference/nma_bayes_truth.json`` (the parameter values the data were
simulated from).

The network is deliberately not a set of two-arm studies, because two-arm
studies are exactly the case where an arm-level and a contrast-level
heterogeneity model coincide.  It contains three three-arm studies and two
studies with no reference treatment at all, so that:

- the arm-level random effect is exercised (a study's two comparisons against
  its baseline share the baseline's sampling error but not their deviations);
- the "first-listed arm is the baseline" rule is exercised on a study that
  does not contain the network reference;
- a parameter is identified by indirect evidence only (``B vs C`` via
  ``B vs A`` and ``C vs A``).

Each study's first-listed arm is its baseline; the row order in the CSV is the
model specification, not a formatting choice.
"""

from __future__ import annotations

import csv
import json
import pathlib

import numpy as np

STUDIES = [
    ("s01", ("A", "B")),
    ("s02", ("A", "C")),
    ("s03", ("A", "D")),
    ("s04", ("A", "B", "C")),
    ("s05", ("A", "B", "D")),
    ("s06", ("A", "C", "D")),
    ("s07", ("B", "C")),
    ("s08", ("A", "B")),
    ("s09", ("A", "C")),
    ("s10", ("A", "D")),
    ("s11", ("B", "D")),
    ("s12", ("A", "B", "C")),
    ("s13", ("A", "C")),
    ("s14", ("A", "B")),
]

# the effects the data are generated from, on the analysis scale
TRUE_D = {"A": 0.0, "B": 0.30, "C": 0.55, "D": 0.15}
TRUE_TAU = 0.15
TRUE_STUDY_MEAN_SD = 1.0

# per-arm sampling standard errors, in the order the arms are listed
SE = [0.18, 0.22, 0.20, 0.25, 0.24, 0.19, 0.30, 0.26,
      0.21, 0.23, 0.28, 0.20, 0.17, 0.24, 0.30, 0.27,
      0.19, 0.22, 0.25, 0.21, 0.29, 0.23, 0.20, 0.26,
      0.18, 0.24, 0.22, 0.28, 0.21, 0.25, 0.23, 0.19, 0.27, 0.30, 0.20]


def build(seed: int = 20240617) -> tuple[list[dict], dict]:
    rng = np.random.default_rng(seed)
    rows: list[dict] = []
    arm = 0
    for study, arms in STUDIES:
        mu = float(rng.normal(0.0, TRUE_STUDY_MEAN_SD))
        base = arms[0]
        for k, treat in enumerate(arms):
            se = SE[arm % len(SE)]
            arm += 1
            dev = 0.0 if k == 0 else float(rng.normal(0.0, TRUE_TAU))
            y = mu + TRUE_D[treat] - TRUE_D[base] + dev + float(rng.normal(0.0, se))
            rows.append({"study": study, "treat": treat,
                         "yi": f"{y:.10f}", "sei": f"{se:.10f}"})
    truth = {
        "provenance": "simulated by tools/make_nma_bayes_fixture.py",
        "model": ("y[jk] ~ N(mu[j] + d[t_jk] - d[t_jb] + delta[jk], se^2); "
                  "delta ~ N(0, tau^2) independent across non-baseline arms"),
        "reference": "A",
        "d": TRUE_D,
        "tau": TRUE_TAU,
        "study_mean_sd": TRUE_STUDY_MEAN_SD,
        "seed": seed,
        "k": len(STUDIES),
        "arms": len(rows),
        "baseline_rule": "each study's first-listed arm",
        "note": ("the CSV row order is part of the model: for a multi-arm "
                 "study the random effects sit on the arms other than the "
                 "baseline"),
    }
    return rows, truth


def main() -> None:
    root = pathlib.Path(__file__).resolve().parents[1]
    rows, truth = build()
    data_path = root / "tests" / "data" / "nma_bayes.csv"
    truth_path = root / "tests" / "reference" / "nma_bayes_truth.json"
    data_path.parent.mkdir(parents=True, exist_ok=True)
    truth_path.parent.mkdir(parents=True, exist_ok=True)
    with data_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=["study", "treat", "yi", "sei"])
        writer.writeheader()
        writer.writerows(rows)
    with truth_path.open("w", encoding="utf-8") as fh:
        json.dump(truth, fh, indent=2, sort_keys=True)
        fh.write("\n")
    print(f"wrote {data_path} ({len(rows)} arms, {truth['k']} studies)")
    print(f"wrote {truth_path}")


if __name__ == "__main__":
    main()
