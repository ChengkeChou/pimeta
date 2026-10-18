"""A complete `pimeta` walkthrough on the classic BCG vaccine dataset.

Run it with::

    python examples/bcg_meta_analysis.py

It writes its inputs and figures to ``./pimeta-example-output/`` and checks every
number against what R returns for the same data:

    library(metafor)
    dat <- escalc(measure = "RR", ai = tpos, bi = tneg, ci = cpos, di = cneg,
                  data = dat.bcg)
    rma(yi, vi, data = dat, method = "REML")

If a check fails the script exits non-zero, so it doubles as a smoke test.
"""
from __future__ import annotations

import csv
import math
import sys
from pathlib import Path

import numpy as np

import pimeta as ma

OUT = Path("pimeta-example-output")
FAILS: list[str] = []


def check(label: str, ok: bool, detail: str = "") -> None:
    print(f"  [{'ok' if ok else 'FAIL'}] {label}" + (f"  {detail}" if detail else ""))
    if not ok:
        FAILS.append(label)


def main() -> int:
    OUT.mkdir(exist_ok=True)

    print("=" * 72)
    print(f"pimeta {ma.__version__} — BCG vaccine meta-analysis walkthrough")
    print("=" * 72)

    # ----------------------------------------------------------------- step 1
    # Real study data lives in a CSV, not in the script. Keep the raw 2x2
    # counts so `escalc()` does the effect-size maths (and gets it right).
    # trial, year, tpos, tneg, cpos, cneg  == metafor::dat.bcg
    studies = [
        ("Aronson 1948", 1948, 4, 119, 11, 128),
        ("Ferguson & Simes 1949", 1949, 6, 300, 29, 274),
        ("Rosenthal et al 1960", 1960, 3, 228, 11, 209),
        ("Hart & Sutherland 1977", 1977, 62, 13536, 248, 12619),
        ("Frimodt-Moller et al 1973", 1973, 33, 5036, 47, 5761),
        ("Stein & Aronson 1953", 1953, 180, 1361, 372, 1079),
        ("Vandiviere et al 1973", 1973, 8, 2537, 10, 619),
        ("TPT Madras 1980", 1980, 505, 87886, 499, 87892),
        ("Coetzee & Berjak 1968", 1968, 29, 7470, 45, 7232),
        ("Rosenthal et al 1961", 1961, 17, 1699, 65, 1600),
        ("Comstock et al 1974", 1974, 186, 50448, 141, 27197),
        ("Comstock & Webster 1969", 1969, 5, 2493, 3, 2338),
        ("Comstock et al 1976", 1976, 27, 16886, 29, 17825),
    ]
    csv_path = OUT / "bcg.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(["study", "year", "tpos", "tneg", "cpos", "cneg"])
        writer.writerows(studies)

    with csv_path.open(newline="", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))

    studlab = [r["study"] for r in rows]
    year = np.array([int(r["year"]) for r in rows], dtype=float)
    tpos = np.array([int(r["tpos"]) for r in rows], dtype=float)
    tneg = np.array([int(r["tneg"]) for r in rows], dtype=float)
    cpos = np.array([int(r["cpos"]) for r in rows], dtype=float)
    cneg = np.array([int(r["cneg"]) for r in rows], dtype=float)

    print(f"\n1. read {len(rows)} studies from {csv_path}")

    # ----------------------------------------------------------------- step 2
    # Effect sizes. For ratio measures such as RR, yi is on the LOG scale.
    es = ma.escalc(measure="RR", ai=tpos, bi=tneg, ci=cpos, di=cneg)
    print("\n2. escalc(measure='RR')")
    print(f"   first study: log RR = {float(es.yi[0]):.6f}  ->  RR = {math.exp(float(es.yi[0])):.4f}")

    # Values printed by R for the same call.
    r_yi = [-0.889311333920205, -1.585388657201431, -1.348073148299693,
            -1.441551190021305, -0.217547322211296, -0.786115585818864,
            -1.620898223598392, 0.011952333523841, -0.469417648738149,
            -1.371344803472784, -0.339358828338391, 0.445913400571379,
            -0.017313948216880]
    r_vi = [0.325584765003961, 0.194581121398144, 0.415367965367965,
            0.020010031902248, 0.051210172169631, 0.006905618455909,
            0.223017247572315, 0.003961579297818, 0.056434210463249,
            0.073024793613029, 0.012412213971560, 0.532505845200153,
            0.071404659683986]
    dy = max(abs(float(a) - b) for a, b in zip(es.yi, r_yi))
    dv = max(abs(float(a) - b) for a, b in zip(es.vi, r_vi))
    check("yi matches R", dy < 1e-12, f"max|dyi|={dy:.1e}")
    check("vi matches R", dv < 1e-12, f"max|dvi|={dv:.1e}")

    # ----------------------------------------------------------------- step 3
    # Random-effects pooling. r.I2 is a percentage, exactly like R's r$I2.
    res = ma.rma(es.yi, es.vi, slab=studlab, method="REML")
    print("\n3. rma(method='REML')")
    print(f"   RR = {math.exp(float(res.b[0])):.4f} "
          f"[{math.exp(float(res.ci_lb[0])):.4f}; {math.exp(float(res.ci_ub[0])):.4f}]")
    print(f"   tau2 = {float(res.tau2):.4f}   I2 = {float(res.I2):.2f}%   "
          f"Q = {float(res.QE):.3f} (p = {float(res.QEp):.2e})")

    check("beta   = -0.7145323", abs(float(res.b[0]) - -0.7145323) < 1e-6,
          f"{float(res.b[0]):.7f}")
    check("tau2   =  0.3132433", abs(float(res.tau2) - 0.3132433) < 1e-6,
          f"{float(res.tau2):.7f}")
    check("I2     = 92.2214", abs(float(res.I2) - 92.22144) < 1e-3,
          f"{float(res.I2):.5f}")
    check("QE     = 152.2330", abs(float(res.QE) - 152.2330) < 1e-3,
          f"{float(res.QE):.5f}")

    print()
    print(ma.summary_text(res, digits=4))

    # ----------------------------------------------------------------- step 4
    # Forest plot. Pass filename= to also write the PNG.
    print("\n4. forest plot")
    forest_png = OUT / "forest.png"
    ma.forest(es.yi, es.vi, studlab=studlab, pooled=res, effect_type="RR",
              log_scale=True, filename=str(forest_png))
    check("forest.png written", forest_png.stat().st_size > 5_000,
          f"{forest_png.stat().st_size:,} bytes")

    # ----------------------------------------------------------------- step 5
    # Publication bias: Egger's test, trim-and-fill, Rosenthal's fail-safe N.
    print("\n5. publication bias")
    egger = ma.regtest(es.yi, es.vi)
    trim = ma.trimfill(es.yi, es.vi)
    fail_safe = ma.fsn(es.yi, es.vi, type="Rosenthal")
    funnel_png = OUT / "funnel.png"
    ma.funnel(es.yi, es.vi, studlab=studlab, effect_type="RR",
              filename=str(funnel_png))
    check("Egger's test", 0.0 <= float(egger.p) <= 1.0, f"p = {float(egger.p):.4f}")
    check("trim-and-fill", trim.k0 >= 0, f"k0 = {trim.k0} imputed")
    check("fail-safe N", float(fail_safe.fsnum) > 0, f"{float(fail_safe.fsnum):.0f}")

    # ----------------------------------------------------------------- step 6
    # Meta-regression: does the effect change with publication year?
    print("\n6. meta-regression on publication year")
    reg = ma.rma(es.yi, es.vi, mods=year, method="REML")
    beta = np.asarray(reg.b, dtype=float)
    print(f"   intercept = {beta[0]:.4f}, slope = {beta[1]:.5f}, "
          f"QM p = {float(reg.QMp):.4f}")
    check("two coefficients", beta.size == 2, beta.round(4).tolist())
    bubble_png = OUT / "bubble.png"
    ma.regplot(reg, mod=year, filename=str(bubble_png))
    check("bubble.png written", bubble_png.stat().st_size > 5_000)

    # ----------------------------------------------------------------- step 7
    # Network meta-analysis on a small synthetic network (illustrative only).
    print("\n7. network meta-analysis (synthetic 4-treatment network)")
    nma = [
        ("s1", 0.10, 0.10, "Placebo", "DrugA"),
        ("s2", -0.20, 0.12, "Placebo", "DrugB"),
        ("s3", 0.05, 0.11, "DrugA", "DrugB"),
        ("s4", -0.15, 0.13, "DrugA", "DrugC"),
        ("s5", 0.30, 0.14, "DrugB", "DrugC"),
        ("s6", 0.25, 0.15, "DrugC", "Placebo"),
    ]
    nm = ma.netmeta(
        TE=[te for _, te, _, _, _ in nma],
        seTE=[se for _, _, se, _, _ in nma],
        treat1=[t1 for _, _, _, t1, _ in nma],
        treat2=[t2 for _, _, _, _, t2 in nma],
        studlab=[s for s, _, _, _, _ in nma],
        sm="MD",
        reference_group="Placebo",
    )
    ranking = ma.netrank(nm)
    print(f"   Q = {float(nm.Q):.4f} between {len(nm.trts)} treatments")
    for trt, pscore in zip(ranking.trts, ranking.ranking_random):
        print(f"     {trt:<8s} P-score = {float(pscore):.3f}")
    check("4 treatments ranked", len(ranking.trts) == 4)
    net_png = OUT / "network.png"
    ma.netgraph(nm, filename=str(net_png))
    check("network.png written", net_png.stat().st_size > 5_000)

    # ----------------------------------------------------------------- wrap up
    print("\n" + "=" * 72)
    print(f"figures and inputs in: {OUT.resolve()}")
    for f in sorted(OUT.iterdir()):
        print(f"  {f.name:<14s} {f.stat().st_size:>9,} bytes")
    print("=" * 72)

    if FAILS:
        print(f"\n{len(FAILS)} check(s) FAILED: {FAILS}")
        return 1
    print("\nAll checks passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
