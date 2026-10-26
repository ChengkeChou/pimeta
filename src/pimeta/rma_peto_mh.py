"""metafor's fixed-effect binary methods ``rma.peto`` and ``rma.mh``.

These replicate ``metafor::rma.peto`` (Peto's one-step odds ratio) and
``metafor::rma.mh`` (Mantel-Haenszel for OR/RR/RD/IRR/IRD) exactly.  The MH
point estimates and variances are the Robins-Breslow-Greenland (OR/RD) and
Greenland-Robins (RR) estimators — identical to the ``meta`` package's
``metabin(method="MH")`` — but metafor additionally reports the
Cochran-Mantel-Haenszel test (continuity-corrected), Breslow-Day and Tarone
heterogeneity tests for the OR, and a MH-style risk-difference variance.

``level`` is accepted as a confidence level (``95`` or ``0.95``) or as the
corresponding alpha (``0.05``); see :mod:`pimeta.levels`.  It used to take
alpha only, which made ``level=0.95`` return a 5%-wide interval and
``level=95`` return ``nan``.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from scipy import stats

from .checks import attach_diagnostics
from .escalc import escalc
from .levels import level_to_alpha
from .metabin import _mh_or, _mh_rr

_Z_975 = stats.norm.ppf(0.975)


@attach_diagnostics
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
    # raw tables, kept so that metafor's cumul()/leave1out()/rstudent()/
    # weights()/confint()/qqnorm() methods can recompute the estimator.
    ai: list[float] | None = None
    bi: list[float] | None = None
    ci: list[float] | None = None
    di: list[float] | None = None
    slab: list[str] | None = None
    add: float = 0.5
    drop00: bool = True


@attach_diagnostics
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
    # raw tables, kept so that metafor's cumul()/leave1out()/rstudent()/
    # weights()/confint()/qqnorm() methods can recompute the estimator.
    ai: list[float] | None = None
    bi: list[float] | None = None
    ci: list[float] | None = None
    di: list[float] | None = None
    slab: list[str] | None = None
    add: float = 0.5
    drop00: bool = True
    correct: bool = True


def _slab(slab, k):
    """Study labels, defaulting to ``Study 1..k``."""
    if slab is None:
        return [f"Study {i + 1}" for i in range(k)]
    slab = [str(v) for v in slab]
    if len(slab) != k:
        raise ValueError(
            f"length of 'slab' ({len(slab)}) does not match the number of "
            f"studies ({k})")
    return slab


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
    slab=None,
) -> RmaPetoResult:
    """Peto's one-step odds-ratio meta-analysis (``metafor::rma.peto``).

    ``ai/bi/ci/di`` are the treatment/control event and non-event counts.
    The estimate is fixed-effect (``tau2 = 0``) and computed on the log scale.
    """
    a = np.asarray(ai, dtype=float)
    b = np.asarray(bi, dtype=float)
    c = np.asarray(ci, dtype=float)
    d = np.asarray(di, dtype=float)
    alpha = level_to_alpha(level)
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
    crit = float(stats.norm.ppf(1.0 - alpha / 2.0))
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
        measure="PETO", method="FE", level=alpha,
        ai=[float(v) for v in a], bi=[float(v) for v in b],
        ci=[float(v) for v in c], di=[float(v) for v in d],
        slab=_slab(slab, k), add=float(add), drop00=bool(drop00),
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
    slab=None,
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
    alpha = level_to_alpha(level)
    n1 = a + b
    n2 = c + d
    N = a + b + c + d
    a.shape[0]
    crit = float(stats.norm.ppf(1.0 - alpha / 2.0))

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
        TA=TA, TAp=TAp, k_pos=k_pos, level=alpha,
        ai=[float(v) for v in a], bi=[float(v) for v in b],
        ci=[float(v) for v in c], di=[float(v) for v in d],
        slab=_slab(slab, k_yi), add=float(add), drop00=bool(drop00),
        correct=bool(correct),
    )



# --------------------------------------------------------------------------- #
# metafor's rma.mh / rma.peto methods
# --------------------------------------------------------------------------- #
#
# metafor gives both models their own cumul(), leave1out(), rstandard(),
# rstudent(), weights(), confint() and qqnorm() implementations; each simply
# refits the (very cheap, closed-form) estimator on subsets of the 2x2 tables.
# They are transcribed here and wired up through the package's generic
# dispatch, so ``weights(mh)``, ``cumul(mh)``, ``leave1out(mh)``, ... work
# exactly as they do in R.

_MH_RESULTS = (RmaPetoResult, RmaMhResult)


def is_mh_result(x) -> bool:
    """True for the ``rma.mh`` / ``rma.peto`` result objects."""
    return isinstance(x, _MH_RESULTS)


def _mh_tables(x):
    """``(a, b, c, d)`` as float arrays."""
    return (np.asarray(x.ai, dtype=float), np.asarray(x.bi, dtype=float),
            np.asarray(x.ci, dtype=float), np.asarray(x.di, dtype=float))


def _mh_refit(x, keep):
    """Refit ``rma.peto`` / ``rma.mh`` on the tables selected by ``keep``."""
    a, b, c, d = _mh_tables(x)
    a, b, c, d = a[keep], b[keep], c[keep], d[keep]
    if isinstance(x, RmaPetoResult):
        return rma_peto(a, b, c, d, add=x.add, drop00=x.drop00, level=x.level)
    return rma_mh(a, b, c, d, measure=x.measure, add=x.add, drop00=x.drop00,
                  correct=x.correct, level=x.level)


def _mh_yi_vi(x):
    """The ``yi``/``vi`` that ``rma.peto`` / ``rma.mh`` actually pool.

    For Peto's method these are the *per-study one-step* log odds ratios
    ``(a - E) / V`` with variance ``1 / V`` (hypergeometric ``V``), not the
    ``a - E`` differences themselves.
    """
    a, b, c, d = _mh_tables(x)
    if isinstance(x, RmaPetoResult):
        n1 = a + b
        n2 = c + d
        N = a + b + c + d
        xt = a + c
        yt = b + d
        E = xt * n1 / N
        V = xt * yt * (n1 / N) * (n2 / N) / (N - 1.0)
        with np.errstate(invalid="ignore", divide="ignore"):
            return (a - E) / V, 1.0 / V
    esc = escalc(x.measure, ai=list(a), bi=list(b), ci=list(c), di=list(d),
                 add=x.add, drop00=x.drop00)
    return np.asarray(esc.yi, dtype=float), np.asarray(esc.vi, dtype=float)


def _mh_wi(x):
    """metafor's per-study weights ``wi`` (``weights.rma.mh`` / ``.peto``)."""
    a, b, c, d = _mh_tables(x)
    n1 = a + b
    n2 = c + d
    N = a + b + c + d
    if isinstance(x, RmaPetoResult):
        xt = a + c
        yt = b + d
        return xt * yt * (n1 / N) * (n2 / N) / (N - 1.0)
    measure = str(x.measure).upper()
    if measure == "OR":
        return (b / N) * c
    if measure == "RR":
        return (c / N) * n1
    if measure == "RD":
        return (n1 / N) * n2
    raise ValueError(f"rma.mh measure {measure!r} has no weight formula")


def mh_weights(x, type="diagonal"):
    """``weights.rma.mh`` / ``weights.rma.peto``.

    ``type="diagonal"`` returns the weights as percentages of their sum,
    ``type="matrix"`` the diagonal weight matrix as a ``numpy`` array.
    """
    if not is_mh_result(x):
        raise TypeError(f"weights() is not defined for {type(x).__name__}")
    wi = _mh_wi(x)
    kind = str(type).lower()
    if kind.startswith("diag"):
        return [float(v) for v in wi / np.sum(wi) * 100.0]
    if kind == "matrix":
        return np.diag(wi)
    raise ValueError("'type' must be 'diagonal' or 'matrix'")


def mh_confint(x, level=None, transf=None):
    """``confint.rma.mh`` / ``confint.rma.peto`` — estimate plus Wald CI.

    ``level`` accepts ``95``, ``0.95`` or ``0.05``; ``transf=True`` applies
    ``exp()`` for the ratio measures, exactly like R.
    """
    if not is_mh_result(x):
        raise TypeError(f"confint() is not defined for {type(x).__name__}")
    alpha = float(x.level) if level is None else level_to_alpha(level)
    crit = float(stats.norm.isf(alpha / 2.0))
    beta = float(x.beta)
    lo = beta - crit * float(x.se)
    hi = beta + crit * float(x.se)
    if transf is True:
        transf = (math.exp if str(x.measure).upper() in ("OR", "RR", "IRR")
                  else None)
    if callable(transf):
        beta, lo, hi = float(transf(beta)), float(transf(lo)), float(transf(hi))
    return {"estimate": beta, "ci.lb": min(lo, hi), "ci.ub": max(lo, hi)}


def mh_rstandard(x):
    """``rstandard.rma.mh`` / ``rstandard.rma.peto`` — standardised residuals."""
    if not is_mh_result(x):
        raise TypeError(f"rstandard() is not defined for {type(x).__name__}")
    yi, vi = _mh_yi_vi(x)
    resid = yi - float(x.beta)
    resid[np.abs(resid) < 100 * 2.220446049250313e-16] = 0.0
    se = np.sqrt(vi)
    with np.errstate(invalid="ignore", divide="ignore"):
        z = resid / se
    return {"resid": [float(v) for v in resid],
            "se": [float(v) for v in se],
            "z": [float(v) for v in z],
            "slab": list(x.slab)}


def mh_rstudent(x):
    """``rstudent.rma.mh`` / ``rstudent.rma.peto`` — deleted residuals.

    Every table is dropped in turn, the estimator is refitted on the rest, and
    the deleted residual ``yi - beta(-i)`` is divided by
    ``sqrt(vi + var(beta(-i)))``.
    """
    if not is_mh_result(x):
        raise TypeError(f"rstudent() is not defined for {type(x).__name__}")
    yi, vi = _mh_yi_vi(x)
    k = int(x.k)
    delpred = np.full(k, np.nan)
    vdelpred = np.full(k, np.nan)
    for i in range(k):
        try:
            res = _mh_refit(x, np.arange(k) != i)
        except (ValueError, ZeroDivisionError, FloatingPointError):
            continue
        delpred[i] = float(res.beta)
        vdelpred[i] = float(res.se) ** 2
    resid = yi - delpred
    resid[np.abs(resid) < 100 * 2.220446049250313e-16] = 0.0
    se = np.sqrt(vi + vdelpred)
    with np.errstate(invalid="ignore", divide="ignore"):
        z = resid / se
    return {"resid": [float(v) for v in resid],
            "se": [float(v) for v in se],
            "z": [float(v) for v in z],
            "slab": list(x.slab)}


def mh_leave1out(x, cluster=None):
    """``leave1out.rma.mh`` / ``leave1out.rma.peto``.

    Refits the estimator once per study (or once per ``cluster``) and returns a
    :class:`~pimeta.sensitivity.LeaveOneOutResult`; the ``slab`` attribute holds
    R's ``"-study"`` labels.
    """
    from .sensitivity import LeaveOneOutResult

    if not is_mh_result(x):
        raise TypeError(f"leave1out() is not defined for {type(x).__name__}")
    k = int(x.k)
    if k == 1:
        raise ValueError("Stopped because k = 1.")
    labels = list(x.slab)
    if cluster is None:
        groups = [[i] for i in range(k)]
        names = [f"-{labels[i]}" for i in range(k)]
    else:
        cl = list(cluster)
        if len(cl) != k:
            raise ValueError(
                f"Length of the variable specified via 'cluster' ({len(cl)}) "
                f"does not match the length of the data ({k}).")
        if any(v is None for v in cl):
            raise ValueError("No missing values allowed in 'cluster' variable.")
        order = list(dict.fromkeys(cl))
        groups = [[i for i in range(k) if cl[i] == g] for g in order]
        names = [f"-{g}" for g in order]

    est, se, z, p, lo, hi, q, qp, i2, h2 = ([] for _ in range(10))
    for grp in groups:
        keep = np.ones(k, dtype=bool)
        keep[grp] = False
        try:
            res = _mh_refit(x, keep)
        except (ValueError, ZeroDivisionError, FloatingPointError):
            for lst in (est, se, z, p, lo, hi, q, qp, i2, h2):
                lst.append(float("nan"))
            continue
        est.append(float(res.beta))
        se.append(float(res.se))
        z.append(float(res.zval))
        p.append(float(res.pval))
        lo.append(float(res.ci_lb))
        hi.append(float(res.ci_ub))
        q.append(float(res.QE))
        qp.append(float(res.QEp))
        i2.append(float(res.I2))
        h2.append(float(res.H2))
    out = LeaveOneOutResult(
        estimate=est, se=se, z=z, p=p, ci_lower=lo, ci_upper=hi,
        QE=q, QE_p=qp, tau2=[0.0] * len(est), I2=i2, H2=h2,
    )
    out.slab = names
    return out


def mh_cumul(x, order=None, decreasing=False, collapse=False):
    """``cumul.rma.mh`` / ``cumul.rma.peto`` — cumulative estimates.

    ``order`` supplies the ordering; consecutive tables are added one at a
    time, or one distinct ``order`` value at a time when ``collapse=True``.
    """
    from .cumul import CumulResult

    if not is_mh_result(x):
        raise TypeError(f"cumul() is not defined for {type(x).__name__}")
    k = int(x.k)
    labels = list(x.slab)
    if order is None:
        orvar = list(range(k))
        collapse = False
    else:
        orvar = list(order)
        if len(orvar) != k:
            raise ValueError(
                f"Length of the 'order' argument ({len(orvar)}) does not "
                f"correspond to the size of the original dataset ({k}).")
    idx = sorted(range(k), key=lambda i: orvar[i], reverse=bool(decreasing))
    uor = list(dict.fromkeys(orvar[i] for i in idx))
    steps = len(uor) if collapse else len(idx)

    kk, est, se, z, p, lo, hi, q, qp, i2, h2, used = ([] for _ in range(12))
    for pos in range(steps):
        if collapse:
            keep = np.array([orvar[i] in uor[:pos + 1] for i in range(k)])
            label = str(uor[pos])
        else:
            keep = np.zeros(k, dtype=bool)
            keep[idx[:pos + 1]] = True
            label = str(labels[idx[pos]])
        try:
            res = _mh_refit(x, keep)
        except (ValueError, ZeroDivisionError, FloatingPointError):
            for lst in (kk, est, se, z, p, lo, hi, q, qp, i2, h2):
                lst.append(float("nan"))
            used.append("")
            continue
        kk.append(int(np.sum(keep)))
        est.append(float(res.beta))
        se.append(float(res.se))
        z.append(float(res.zval))
        p.append(float(res.pval))
        lo.append(float(res.ci_lb))
        hi.append(float(res.ci_ub))
        q.append(float(res.QE))
        qp.append(float(res.QEp))
        i2.append(float(res.I2))
        h2.append(float(res.H2))
        used.append(label)
    out = CumulResult(
        estimate=est, se=se, z=z, p=p, ci_lower=lo, ci_upper=hi,
        q=q, q_p=qp, tau2=[0.0] * len(est), i2=i2, h2=h2, k=kk,
    )
    out.slab = used
    return out
