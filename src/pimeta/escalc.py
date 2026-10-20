"""Effect-size computation (metafor ``escalc`` / meta ``metabin``/``metacont``).

Computes an effect estimate ``yi`` and sampling variance ``vi`` from raw study
data, reproducing metafor's ``escalc()`` formulas exactly. This is the numeric
foundation for ``metabin`` (2x2 binary), ``metacont`` (continuous) and
``metaprop`` (single-arm proportions).

All ratio-type measures (RR/OR/PETO/IRR/PLN/PLO) return ``yi`` on the **log**
scale, matching metafor, so they drop straight into inverse-variance pooling.

Implemented measures:
    binary 2x2 : RR, OR, RD, AS, PETO
    continuous : MD, SMD (Hedges' g), SMDH
    rates      : IRR
    proportions: PLN, PLO, PAS, PFT
    correlation: ZCOR (Fisher's z)
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from .data_compat import resolve

# Measures pooled on the log scale (back-transformed with exp after pooling).
LOG_MEASURES = {"RR", "OR", "PETO", "IRR", "PLN", "PLO"}

_BINARY_MEASURES = {"RR", "OR", "RD", "AS", "PETO"}
_CONTINUOUS_MEASURES = {"MD", "SMD", "SMDH"}
_RATE_MEASURES = {"IRR"}
_PROP_MEASURES = {"PLN", "PLO", "PAS", "PFT", "PR"}
_CORR_MEASURES = {"ZCOR"}

SUPPORTED_MEASURES = (
    _BINARY_MEASURES | _CONTINUOUS_MEASURES | _RATE_MEASURES
    | _PROP_MEASURES | _CORR_MEASURES
)


@dataclass
class EscalcResult:
    """Vectorized effect-size output (one row per study)."""

    measure: str
    yi: list[float]                 # effect (analysis scale)
    vi: list[float]                 # sampling variance (analysis scale)
    ni: list[float]                 # per-study total sample size (for HS tau2)
    k: int

    def to_studies(self) -> list[tuple[float, float, float]]:
        """Return list of (yi, vi, ni) triples."""
        return list(zip(self.yi, self.vi, self.ni))


def _vec(values, name: str) -> list[float] | None:
    if values is None:
        return None
    return [float(x) for x in values]


def escalc(
    measure: str,
    ai=None,
    bi=None,
    ci=None,
    di=None,
    n1i=None,
    n2i=None,
    m1i=None,
    m2i=None,
    sd1i=None,
    sd2i=None,
    x1i=None,
    x2i=None,
    t1i=None,
    t2i=None,
    xi=None,
    ni=None,
    ri=None,
    add: float = 0.5,
    drop00: bool = True,
    vtype: str = "LS",
    data=None,
) -> EscalcResult:
    """Compute effect sizes and variances.

    Column inputs are parallel lists. For 2x2 binary measures pass ``ai, bi,
    ci, di`` (treatment/control events and non-events) or ``ai, ci, n1i, n2i``
    (events plus group sizes). ``add`` is the continuity correction for OR
    (0.5 by default, matching metafor). ``drop00`` drops rows with zero events
    in both arms (metafor's default for RR).  ``vtype`` selects the variance
    estimator for SMD/SMDH (metafor's ``vtype``, default ``"LS"``).

    ``data`` lets every column argument be given as a *name* instead of a
    vector (``escalc("RR", "ai", "bi", "ci", "di", data=df)``); it accepts a
    dict, a pandas/polars ``DataFrame``, a structured array or a list of
    records.
    """
    measure = measure.upper()
    if measure not in SUPPORTED_MEASURES:
        raise ValueError(f"unsupported measure: {measure!r}")

    if data is not None:
        (ai, bi, ci, di, n1i, n2i, m1i, m2i, sd1i, sd2i, x1i, x2i, t1i, t2i,
         xi, ni, ri) = (
            resolve(v, data, name=n, dtype=None)
            for v, n in ((ai, "ai"), (bi, "bi"), (ci, "ci"), (di, "di"),
                         (n1i, "n1i"), (n2i, "n2i"), (m1i, "m1i"), (m2i, "m2i"),
                         (sd1i, "sd1i"), (sd2i, "sd2i"), (x1i, "x1i"),
                         (x2i, "x2i"), (t1i, "t1i"), (t2i, "t2i"), (xi, "xi"),
                         (ni, "ni"), (ri, "ri")))

    ai = _vec(ai, "ai")
    bi = _vec(bi, "bi")
    ci = _vec(ci, "ci")
    di = _vec(di, "di")
    n1i = _vec(n1i, "n1i")
    n2i = _vec(n2i, "n2i")

    if measure in _BINARY_MEASURES:
        # Derive the full 2x2 table if only events + group sizes were given.
        if bi is None and n1i is not None:
            bi = [n1 - a for n1, a in zip(n1i, ai)]
        if di is None and n2i is not None:
            di = [n2 - c for n2, c in zip(n2i, ci)]
        if any(x is None for x in (ai, bi, ci, di)):
            raise ValueError(
                f"measure {measure} needs ai/bi/ci/di or ai/ci/n1i/n2i"
            )
        rows = list(zip(ai, bi, ci, di))
        if drop00:
            keep = [i for i, r in enumerate(rows) if not (r[0] == 0 and r[2] == 0)]
            rows = [rows[i] for i in keep]
        return _escalc_binary(measure, rows, add)

    if measure in _CONTINUOUS_MEASURES:
        m1i = _vec(m1i, "m1i")
        m2i = _vec(m2i, "m2i")
        sd1i = _vec(sd1i, "sd1i")
        sd2i = _vec(sd2i, "sd2i")
        n1i = _vec(n1i, "n1i")
        n2i = _vec(n2i, "n2i")
        if any(x is None for x in (m1i, m2i, sd1i, sd2i, n1i, n2i)):
            raise ValueError("continuous measures need m1i/m2i/sd1i/sd2i/n1i/n2i")
        return _escalc_continuous(measure, m1i, m2i, sd1i, sd2i, n1i, n2i,
                                  vtype=vtype)

    if measure in _RATE_MEASURES:
        x1i = _vec(x1i, "x1i")
        x2i = _vec(x2i, "x2i")
        t1i = _vec(t1i, "t1i")
        t2i = _vec(t2i, "t2i")
        if any(x is None for x in (x1i, x2i, t1i, t2i)):
            raise ValueError("IRR needs x1i/x2i/t1i/t2i")
        yi = [math.log((x1 / t1) / (x2 / t2)) for x1, x2, t1, t2 in zip(x1i, x2i, t1i, t2i)]
        vi = [1.0 / x1 + 1.0 / x2 for x1, x2 in zip(x1i, x2i)]
        ni = [t1 + t2 for t1, t2 in zip(t1i, t2i)]
        return EscalcResult(measure, yi, vi, ni, len(yi))

    if measure in _PROP_MEASURES:
        xi = _vec(xi, "xi")
        ni = _vec(ni, "ni")
        if xi is None or ni is None:
            raise ValueError("proportion measures need xi/ni")
        return _escalc_prop(measure, xi, ni)

    # correlation
    ri = _vec(ri, "ri")
    ni = _vec(ni, "ni")
    if ri is None or ni is None:
        raise ValueError("ZCOR needs ri/ni")
    yi = [0.5 * math.log((1.0 + r) / (1.0 - r)) for r in ri]
    vi = [1.0 / (n - 3.0) for n in ni]
    return EscalcResult(measure, yi, vi, list(ni), len(yi))


def _escalc_binary(measure: str, rows: list, add: float) -> EscalcResult:
    yi: list[float] = []
    vi: list[float] = []
    ni: list[float] = []
    for a, b, c, d in rows:
        # metafor `to="only0"`: for any 2x2 table with at least one zero
        # cell, add `add` to ALL four cells (not just the zero ones).
        if a == 0 or b == 0 or c == 0 or d == 0:
            A, B, C, D = a + add, b + add, c + add, d + add
        else:
            A, B, C, D = a, b, c, d
        n1, n2 = A + B, C + D
        N = n1 + n2
        if measure == "RR":
            y = math.log((A / n1) / (C / n2))
            v = 1.0 / A - 1.0 / n1 + 1.0 / C - 1.0 / n2
        elif measure == "OR":
            y = math.log((A * D) / (B * C))
            v = 1.0 / A + 1.0 / B + 1.0 / C + 1.0 / D
        elif measure == "RD":
            y = A / n1 - C / n2
            v = A * B / n1 ** 3 + C * D / n2 ** 3
        elif measure == "AS":
            y = math.asin(math.sqrt(A / n1)) - math.asin(math.sqrt(C / n2))
            v = 1.0 / (4.0 * n1) + 1.0 / (4.0 * n2)
        else:  # PETO
            O, E = A, (A + B) * (A + C) / N
            V = (A + B) * (C + D) * (A + C) * (B + D) / (N * N * (N - 1.0))
            y = (O - E) / V
            v = 1.0 / V
        yi.append(y)
        vi.append(v)
        ni.append(float(N))
    return EscalcResult(measure, yi, vi, ni, len(yi))


def _cmicalc(mi: float) -> float:
    """metafor's ``cmicalc(mi, method="exact")`` — the exact Hedges correction.

    ``exp(lgamma(mi/2) - log(sqrt(mi/2)) - lgamma((mi-1)/2))``.  The common
    large-sample shortcut ``1 - 3/(4*mi - 1)`` is *not* what metafor uses, and
    differs in the 6th decimal for realistic group sizes.
    """
    from scipy.special import gammaln

    if mi <= 1:
        return float("nan")
    return float(math.exp(gammaln(mi / 2.0) - math.log(math.sqrt(mi / 2.0))
                          - gammaln((mi - 1.0) / 2.0)))


#: Which ``vtype`` values metafor accepts for each standardised measure.
_SMD_VTYPES = {"LS", "LS2", "LS3", "UB", "AV", "H0"}
_SMDH_VTYPES = {"LS", "LS2", "LS3"}


def _escalc_continuous(
    measure: str, m1i, m2i, sd1i, sd2i, n1i, n2i, vtype: str = "LS"
) -> EscalcResult:
    yi: list[float] = []
    vi: list[float] = []
    ni: list[float] = []
    vt = str(vtype).upper()
    if measure == "SMD" and vt not in _SMD_VTYPES:
        raise ValueError(
            f"vtype must be one of {sorted(_SMD_VTYPES)} for SMD"
        )
    if measure == "SMDH" and vt not in _SMDH_VTYPES:
        raise ValueError(
            f"vtype must be one of {sorted(_SMDH_VTYPES)} for SMDH"
        )
    # metafor's vtype="AV" standardises by the sample-size-weighted mean g.
    if measure == "SMD" and vt == "AV":
        raw = []
        for m1, m2, sd1, sd2, n1, n2 in zip(m1i, m2i, sd1i, sd2i, n1i, n2i):
            mi = n1 + n2 - 2
            sp = math.sqrt(((n1 - 1) * sd1 ** 2 + (n2 - 1) * sd2 ** 2) / mi)
            raw.append((m1 - m2) / sp * _cmicalc(mi))
        wsum = sum(w for w in (n1i[i] + n2i[i] for i in range(len(raw))))
        mnwyi = sum(raw[i] * (n1i[i] + n2i[i]) for i in range(len(raw))) / wsum

    for m1, m2, sd1, sd2, n1, n2 in zip(m1i, m2i, sd1i, sd2i, n1i, n2i):
        mi = n1 + n2 - 2
        if measure == "MD":
            y = m1 - m2
            v = sd1 ** 2 / n1 + sd2 ** 2 / n2
        elif measure == "SMD":
            sp = math.sqrt(((n1 - 1) * sd1 ** 2 + (n2 - 1) * sd2 ** 2) / mi)
            d = (m1 - m2) / sp
            y = _cmicalc(mi) * d
            if vt == "LS":
                v = 1.0 / n1 + 1.0 / n2 + y ** 2 / (2.0 * (n1 + n2))
            elif vt == "LS2":
                v = _cmicalc(mi) ** 2 * (
                    1.0 / n1 + 1.0 / n2 + d ** 2 / (2.0 * (n1 + n2)))
            elif vt == "LS3":
                v = 1.0 / n1 + 1.0 / n2 + y ** 2 / (2.0 * (n1 + n2 - 3.94))
            elif vt == "UB":
                cmi = _cmicalc(mi)
                v = (1.0 / n1 + 1.0 / n2
                     + (1.0 - (mi - 2.0) / (mi * cmi ** 2)) * y ** 2)
            elif vt == "AV":
                v = 1.0 / n1 + 1.0 / n2 + mnwyi ** 2 / (2.0 * (n1 + n2))
            else:  # H0
                cmi = _cmicalc(mi)
                v = (cmi ** 2 * (1.0 / n1 + 1.0 / n2) * mi / (mi - 2.0)
                     if mi > 2 else float("nan"))
        else:  # SMDH — standardised by the average of the two SDs
            sdpi = math.sqrt((sd1 ** 2 + sd2 ** 2) / 2.0)
            d = (m1 - m2) / sdpi
            y = _cmicalc(mi) * d
            if vt == "LS":
                v = (y ** 2 * (sd1 ** 4 / (n1 - 1) + sd2 ** 4 / (n2 - 1))
                     / (8.0 * sdpi ** 4)
                     + (sd1 ** 2 / (n1 - 1) + sd2 ** 2 / (n2 - 1)) / sdpi ** 2)
            elif vt == "LS2":
                v = (sd1 ** 2 / (n1 * sdpi ** 2) + sd2 ** 2 / (n2 * sdpi ** 2)
                     + y ** 2 / (8.0 * sdpi ** 4) * (sd1 ** 4 / n1 + sd2 ** 4 / n2))
            else:  # LS3
                v = (sd1 ** 2 / (n1 * sdpi ** 2) + sd2 ** 2 / (n2 * sdpi ** 2)
                     + y ** 2 / (8.0 * sdpi ** 4)
                     * (sd1 ** 4 / (n1 - 1) + sd2 ** 4 / (n2 - 1)))
        yi.append(y)
        vi.append(v)
        ni.append(float(n1 + n2))
    return EscalcResult(measure, yi, vi, ni, len(yi))


def _escalc_prop(measure: str, xi, ni) -> EscalcResult:
    yi: list[float] = []
    vi: list[float] = []
    for x, n in zip(xi, ni):
        p = x / n
        if measure == "PLN":
            y, v = math.log(p), (1.0 - p) / (n * p)
        elif measure == "PLO":
            y, v = math.log(p / (1.0 - p)), 1.0 / (n * p * (1.0 - p))
        elif measure == "PAS":
            y, v = math.asin(math.sqrt(p)), 1.0 / (4.0 * n)
        elif measure == "PR":
            y, v = p, p * (1.0 - p) / n
        else:  # PFT (Freeman-Tukey double arcsine)
            y = 0.5 * (
                math.asin(math.sqrt(x / (n + 1.0)))
                + math.asin(math.sqrt((x + 1.0) / (n + 1.0)))
            )
            v = 1.0 / (4.0 * n + 2.0)
        yi.append(y)
        vi.append(v)
    return EscalcResult(measure, yi, vi, [float(n) for n in ni], len(yi))
