"""Regenerate the figures committed under ``docs/guide/images/``.

A figure in a guide is a claim. Asserting that a plotting function returns a
``Figure`` is not the same as looking at it — the specification-curve plot passed
twelve tests while drawing a single visible dot, because its x axis was measured
in effect sizes and its margin positions were ordinal. The only check that
catches that is rendering the file and looking at it, which is why the output is
committed rather than generated at docs build time: what is in the repository is
what a reader sees.

The data is synthetic and fixed. It is not a meta-analysis anyone should quote;
it exists so the shape of the picture is reproducible and a change to the
plotting code shows up as a changed file in `git diff` rather than as a surprise
in the rendered docs.

Usage::

    python tools/make_figures.py           # rewrite every figure
    python tools/make_figures.py --check    # fail if any file would change

``--check`` is what CI runs: a stale figure is a documentation bug, and the way
to notice it is to make it a failing check rather than a memory exercise.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import matplotlib  # noqa: E402

matplotlib.use("Agg")

from pimeta.multiverse import multiverse  # noqa: E402
from pimeta.plotting import plot_specification_curve  # noqa: E402
from pimeta.rma_uni import rma  # noqa: E402
from pimeta.spec import AnalysisSpec, Decision  # noqa: E402

IMAGES = ROOT / "docs" / "guide" / "images"

#: Eight studies, fixed, with a real but modest effect and honest disagreement.
YI = np.array([0.62, 0.11, 0.44, 0.05, 0.73, 0.28, -0.19, 0.51])
SEI = np.array([0.21, 0.18, 0.30, 0.15, 0.42, 0.24, 0.26, 0.33])


def guide_spec() -> AnalysisSpec:
    """The decisions the guide declares, in the same order."""
    return AnalysisSpec(decisions=(
        Decision(name="method", values=("REML", "DL", "FE"), default="REML",
                 reasons={"REML": "the default here; less biased at small k",
                          "DL": "what most published analyses used",
                          "FE": "the assumption of no heterogeneity"}),
        Decision(name="test", values=("z", "knha"), default="z",
                 reasons={"z": "the nominal reference",
                          "knha": "wider when the studies disagree"}),
    ))


def specification_curve(out: Path) -> Path:
    """The figure in ``docs/guide/multiverse.md``."""
    spec = guide_spec()
    yi, sei = YI, SEI
    mv = multiverse(spec, lambda c: rma(yi, sei=sei, **c))
    default = spec.default()
    plot_specification_curve(
        mv, filename=str(out), title="Specification curve, eight synthetic studies",
        mark_default=default.id if default is not None else None)
    return out


FIGURES = {"specification_curve.png": specification_curve}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--check", action="store_true",
                        help="do not write; exit 1 if a committed figure is stale")
    args = parser.parse_args(argv)

    IMAGES.mkdir(parents=True, exist_ok=True)
    stale = 0
    for name, build in FIGURES.items():
        path = IMAGES / name
        before = path.read_bytes() if path.exists() else None
        tmp = path.with_suffix(".tmp.png")
        build(tmp)
        after = tmp.read_bytes()
        if args.check:
            if before != after:
                stale += 1
                print(f"stale: {path.relative_to(ROOT)}")
            tmp.unlink()
        else:
            tmp.replace(path)
            print(f"wrote: {path.relative_to(ROOT)} ({len(after) / 1024:.0f} KiB)")
    if args.check and stale:
        print(f"{stale} figure(s) do not match the plotting code; "
              "run `python tools/make_figures.py`")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
