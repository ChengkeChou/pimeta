"""The fixture behind ``tests/test_robma.py``.

Fifteen studies with standard errors spread unevenly and a true effect of 0.30,
built so that the funnel is mildly asymmetric: the three most precise studies
sit above the pooled value. That is enough for the bias models to be given real
work to do without any of them dominating, which is what a reference comparison
needs.

The numbers are written to ``tests/data/robma.csv``; nothing is randomised at
run time, so the R reference and the Python test read the same fifteen rows.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def make() -> dict[str, np.ndarray]:
    """The fixture, as columns."""
    sei = np.array([
        0.150, 0.160, 0.170, 0.195, 0.205, 0.225, 0.240, 0.255,
        0.270, 0.290, 0.310, 0.330, 0.355, 0.380, 0.410,
    ])
    yi = np.array([
        0.512, 0.447, 0.398, 0.301, 0.264, 0.355, 0.198, 0.242,
        0.094, 0.176, 0.021, 0.088, 0.143, -0.041, 0.207,
    ])
    ni = np.round(4.0 / sei ** 2)
    return {"study": np.arange(1, yi.size + 1), "yi": yi, "sei": sei,
            "ni": ni}


def main() -> None:
    cols = make()
    out = ROOT / "tests" / "data" / "robma.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    header = ",".join(cols)
    rows = [f"{s},{y:0.6f},{e:0.6f},{n}" for s, y, e, n in zip(
        cols["study"], cols["yi"], cols["sei"], cols["ni"])]
    out.write_text(header + "\n" + "\n".join(rows) + "\n", encoding="utf-8")
    print(f"wrote {out} with {len(rows)} studies")


if __name__ == "__main__":
    main()
