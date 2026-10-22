"""An end-to-end traceable diagnostic accuracy analysis.

Run it::

    python examples/dta_record.py

It fits a bivariate DTA model on mada's ``AuditC`` data, writes the whole
analysis to ``dta_record.json``, then reopens the file and re-runs the
analysis from what the file contains -- no original data needed, because the
record is a bundle.

The point is not the numbers (those are in the guide). The point is that the
file is enough: an auditor can see which data went in, which settings were
used, which software produced it, and whether re-running gives the same
answer.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import pimeta as ma
from pimeta.record import EstimatesStore, load_record, save_record

# mada::AuditC -- 14 studies of a liquid-based cytology test.
COUNT = {
    "TP": [47, 126, 19, 36, 130, 84, 68, 752, 59, 142, 137, 57, 34, 152],
    "FN": [9, 51, 10, 3, 19, 2, 0, 0, 5, 50, 24, 3, 1, 51],
    "FP": [101, 272, 12, 78, 211, 68, 112, 3226, 55, 571, 107, 103, 21, 88],
    "TN": [738, 1543, 192, 276, 959, 89, 423, 2977, 136, 2788, 358, 437, 56, 264],
}


def main() -> None:
    out = Path(tempfile.mkdtemp(prefix="pimeta-dta-"))

    # ---------------------------------------------------------------- fit --
    fit = ma.reitsma(COUNT)
    print("bivariate model, REML")
    print(f"  mean logit(sens) = {fit.mu[0]:.4f}  (se {fit.se[0]:.4f})")
    print(f"  mean logit(fpr)  = {fit.mu[1]:.4f}  (se {fit.se[1]:.4f})")
    print(f"  summary point    = sens {fit.summary_sens:.3f}, "
          f"spec {fit.summary_spec:.3f}")
    print(f"  tau              = {fit.tau[0]:.3f}, {fit.tau[1]:.3f}   rho = {fit.rho:.3f}")
    print(f"  AIC = {fit.AIC:.3f}")

    h = ma.hsroc(fit)
    print(f"  HSROC            = Theta {h.Theta:.3f}, Lambda {h.Lambda:.3f}, "
          f"beta {h.beta:.3f}")
    print(f"  AUC              = {ma.auc(fit)['auc']:.4f}")

    # ------------------------------------------------------------- record --
    # mode="bundle" copies the counts into the file, which is what makes the
    # record self-verifying and self-replaying.  pass_data=True is needed
    # because reitsma() takes the whole table, not column names.
    path = out / "dta_record.json"
    record = save_record(
        path,
        fit,
        data=COUNT,
        mode="bundle",
        function="reitsma",
        call={"method": "reml"},
        pass_data=True,
    )
    print(f"\nwrote {path}  ({path.stat().st_size:,} bytes)")
    print("  " + record.summary().replace("\n", "\n  "))

    # ------------------------------------------------------------- verify --
    reopened = load_record(path)
    fresh = reopened.verify()
    print(f"\nverify: the bundled counts still hash to "
          f"{fresh['analysis'][:23]}...")

    # ------------------------------------------------------------- replay --
    again = reopened.replay()          # raises RecordMismatch on any drift
    print(f"replay: mu {again.mu[0]:.10f} / {again.mu[1]:.10f} "
          f"(original {fit.mu[0]:.10f} / {fit.mu[1]:.10f})")

    # -------------------------------------------------------------- store --
    models = EstimatesStore("auditc")
    models.store("reml", fit, data=COUNT, mode="bundle",
                 function="reitsma", call={"method": "reml"}, pass_data=True)
    models.store("ml", ma.reitsma(COUNT, method="ml"), data=COUNT,
                 mode="bundle", function="reitsma", call={"method": "ml"},
                 pass_data=True)
    models.save(out / "estimates.json")

    print("\nestimates store")
    table = models.compare()
    for row in zip(table["model"], table["term"], table["estimate"], table["se"]):
        print(f"  {row[0]:5s} {row[1]:26s} {row[2]:+.4f}  (se {row[3]:.4f})")

    print(f"\neverything is under {out}")


if __name__ == "__main__":
    main()
