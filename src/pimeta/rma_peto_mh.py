"""metafor's fixed-effect binary methods ``rma.peto`` and ``rma.mh``.

These replicate ``metafor::rma.peto`` (Peto's one-step odds ratio) and
``metafor::rma.mh`` (Mantel-Haenszel for OR/RR/RD/IRR/IRD) exactly.  The MH
point estimates and variances are the Robins-Breslow-Greenland (OR/RD) and
Greenland-Robins (RR) estimators — identical to the ``meta`` package's
``metabin(method="MH")`` — but metafor additionally reports the
Cochran-Mantel-Haenszel test (continuity-corrected), Breslow-Day and Tarone
heterogeneity tests for the OR, and a MH-style risk-difference variance.

``level`` is the alpha level (0.05 -> 95%), consistent with the rest of the
package (``confint_tau2``, ``predict_rma``, ``hc_rma``).
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from scipy import stats

from .escalc import escalc
from .metabin import _mh_or, _mh_rr, _mh_rd, _peto_or

_Z_975 = stats.norm.ppf(0.975)


@dataclass
class RmaPetoResult:
    """``metafor::rma.peto`` — Peto's one-step odds ratio (fixed effects)."""

    beta: float          # log OR
    se: float
    zval: float
    pval: float
    ci_lb: float
    ci_ub: float
    tau2: float          # always 0 (FE)
    I2: float
    H2: float
    QE: float
    QEp: float
    k: int
    k_pos: int
    measure: str         # "PETO"
    method: str          # "FE"
    level: float         # alpha


@dataclass
class RmaMhResult:
    """``metafor::rma.mh`` — Mantel-Haenszel (fixed effects)."""

    beta: float          # log OR / log RR / RD / log IRR / IRD
    se: float
    zval: float
    pval: float
    ci_lb: float
    ci_ub: float
    tau2: float          # always 0 (FE)
    I2: float
    H2: float
    QE: float
    QEp: float
    k: int
    measure: str
    method: str          # "FE"
    CO: float | None     # Cochran-Mantel-Haenszel (uncorrected var), OR/IRR
    COp: float | None
    MH: float | None     # Cochran-Mantel-Haenszel (corrected var), OR/IRR
    MHp: float | None
    BD: float | None     # Breslow-Day, OR only
    BDp: float | None
    TA: float | None     # Tarone, OR only
    TAp: float | None
    k_pos: int | None
    level: float         # alpha


def _fe_heterogeneity(qe: float, k: int) -> tuple[float, float, float, float]:
    """metafor's FE I2/H2/QEp block (rma.peto / rma.mh)."""
    if k > 1:
        qep = float(stats.chi2.sf(qe, k - 1))
        i2 = max(0.0, 100.0 * (qe - (k - 1)) / qe)
        h2 = qe / (k - 1)
    else:
        qep = 1.0
        i2 = 0.0
        h2 = 1.0
    return qep, i2, h2, qe


def rma_peto(
    ai,
    bi,
    ci,
    di,
    *,
    add: float = 0.5,
    drop00: bool = True,
    level: float = 0.05,
) -> RmaPetoResult:
    """Peto's one-step odds-ratio meta-analysis (``metafor::rma.peto``).

    ``ai/bi/ci/di`` are the treatment/control event and non-event counts.
    The estimate is fixed-effect (``tau2 = 0``) and computed on the log scale.
    """
    a = np.asarray(ai, dtype=float)
    b = np.asarray(bi, dtype=float)
    c = np.asarray(ci, dtype=float)
    d = np.asarray(di, dtype=float)
    n1 = a + b
    n2 = c + d
    N = a + b + c + d

    xt = a + c
    yt = b + d
    Ei = xt * n1 / N
    Vi = xt * yt * (n1 / N) * (n2 / N) / (N - 1.0)
    sumVi = float(np.sum(Vi))
    if sumVi == 0.0:
        raise ValueError(
            "One of the two outcomes never occurred in any table; "
            "Peto's method cannot be used."
        )

    beta = float(np.sum(a - Ei) / sumVi)
    se = math.sqrt(1.0 / sumVi)
    zval = beta / se
    pval = 2.0 * float(stats.norm.sf(abs(zval)))
    crit = float(stats.norm.ppf(1.0 - level / 2.0))
    ci_lb = beta - crit * se
    ci_ub = beta + crit * se

    k_pos = int(np.sum(Vi > 0.0))
    mask = Vi > 0.0
    Vi2 = np.where(mask, Vi, np.nan)
    oe = a - Ei
    qe = max(0.0, float(np.nansum(oe**2 / Vi2) - np.nansum(oe) ** 2 / np.nansum(Vi2)))
    k = a.shape[0]
    qep, i2, h2, _ = _fe_heterogeneity(qe, k)

    return RmaPetoResult(
        beta=beta, se=se, zval=zval, pval=pval, ci_lb=ci_lb, ci_ub=ci_ub,
        tau2=0.0, I2=i2, H2=h2, QE=qe, QEp=qep, k=k, k_pos=k_pos,
        measure="PETO", method="FE", level=level,
    )


def rma_mh(
    ai,
    bi,
    ci,
    di,
    *,
    measure: str = "OR",
    add: float = 0.5,
    drop00: bool = True,
    correct: bool = True,
    level: float = 0.05,
) -> RmaMhResult:
    """Mantel-Haenszel meta-analysis (``metafor::rma.mh``).

    Supports ``measure`` in ``{"OR", "RR", "RD"}``.  ``beta`` is on the log
    scale for OR/RR and on the raw scale for RD.  For the OR, the
    Cochran-Mantel-Haenszel (``MH``/``CO``), Breslow-Day (``BD``) and Tarone
    (``TA``) tests are also returned.
    """
    measure = measure.upper()
    if measure not in ("OR", "RR", "RD"):
        raise ValueError("rma.mh supports measures OR, RR, RD")

    a = np.asarray(ai, dtype=float)
    b = np.asarray(bi, dtype=float)
    c = np.asarray(ci, dtype=float)
    d = np.asarray(di, dtype=float)
    n1 = a + b
    n2 = c + d
    N = a + b + c + d
    k = a.shape[0]
    crit = float(stats.norm.ppf(1.0 - level / 2.0))

    CO = COp = MH = MHp = BD = BDp = TA = TAp = None
    k_pos = None

    if measure == "OR":
        beta, var = _mh_or(a, b, c, d, n1, n2)
        se = math.sqrt(var)
        zval = beta / se
        pval = 2.0 * float(stats.norm.sf(abs(zval)))
        ci_lb = beta - crit * se
        ci_ub = beta + crit * se

        # Cochran-Mantel-Haenszel tests (continuity-corrected MH).
        xt = a + c
        yt = b + d
        corr = 0.5 if correct else 0.0
        if float(np.sum(xt)) == 0.0 or float(np.sum(yt)) == 0.0:
            CO = COp = MH = MHp = float("nan")
        else:
            num = (abs(float(np.sum(a - (n1 / N) * xt))) - corr) ** 2
            CO = num / float(np.sum((n1 / N) * (n2 / N) * (xt * (yt / N))))
            COp = float(stats.chi2.sf(CO, 1))
            MH = num / float(np.sum((n1 / N) * (n2 / N) * (xt * (yt / (N - 1.0)))))
            MHp = float(stats.chi2.sf(MH, 1))

        # Breslow-Day / Tarone.
        beta_exp = math.exp(beta)
        if beta_exp == 1.0:
            N11 = (n1 / N) * xt
        else:
            A = beta_exp * (n1 + xt) + (n2 - xt)
            B = np.sqrt(A**2 - 4.0 * n1 * xt * beta_exp * (beta_exp - 1.0))
            N11 = (A - B) / (2.0 * (beta_exp - 1.0))
        pos = (N11 > 0.0) & (xt > 0.0) & (yt > 0.0)
        k_pos = int(np.sum(pos))
        N11p = N11[pos]
        N12 = n1[pos] - N11p
        N21 = xt[pos] - N11p
        N22 = N11p - n1[pos] - xt[pos] + N[pos]
        harm = 1.0 / (1.0 / N11p + 1.0 / N12 + 1.0 / N21 + 1.0 / N22)
        ap = a[pos]
        BD = max(0.0, float(np.sum((ap - N11p) ** 2 / harm)))
        TA = max(0.0, BD - float(np.sum(ap - N11p) ** 2 / np.sum(harm)))
        if k_pos > 1:
            BDp = float(stats.chi2.sf(BD, k_pos - 1))
            TAp = float(stats.chi2.sf(TA, k_pos - 1))
        else:
            BDp = TAp = float("nan")

    elif measure == "RR":
        beta, var = _mh_rr(a, b, c, d, n1, n2)
        se = math.sqrt(var)
        zval = beta / se
        pval = 2.0 * float(stats.norm.sf(abs(zval)))
        ci_lb = beta - crit * se
        ci_ub = beta + crit * se

    else:  # RD
        C = float(np.sum(n1 * (n2 / N)))
        beta = float(np.sum(a * (n2 / N) - c * (n1 / N))) / C
        A = float(np.sum(
            c * (n1 / N) ** 2 - a * (n2 / N) ** 2
            + (n1 / N) * (n2 / N) * (n2 - n1) / 2.0
        ))
        B = float(np.sum(a * (n2 - c) / N + c * (n1 - a) / N))
        var = (beta * A + B / 2.0) / C**2
        se = math.sqrt(var)
        zval = beta / se
        pval = 2.0 * float(stats.norm.sf(abs(zval)))
        ci_lb = beta - crit * se
        ci_ub = beta + crit * se

    # Cochran Q + I2/H2, centered on the MH estimate (metafor uses the
    # escalc yi/vi — with add=0.5 to="only0" — and beta on the same scale).
    esc = escalc(measure, ai=list(a), bi=list(b), ci=list(c), di=list(d),
                 add=add, drop00=drop00)
    yi = np.asarray(esc.yi, dtype=float)
    vi = np.asarray(esc.vi, dtype=float)
    k_yi = yi.shape[0]
    wi = 1.0 / vi
    qe = max(0.0, float(np.sum(wi * (yi - beta) ** 2)))
    qep, i2, h2, _ = _fe_heterogeneity(qe, k_yi)

    return RmaMhResult(
        beta=beta, se=se, zval=zval, pval=pval, ci_lb=ci_lb, ci_ub=ci_ub,
        tau2=0.0, I2=i2, H2=h2, QE=qe, QEp=qep, k=k_yi, measure=measure,
        method="FE", CO=CO, COp=COp, MH=MH, MHp=MHp, BD=BD, BDp=BDp,
        TA=TA, TAp=TAp, k_pos=k_pos, level=level,
    )
