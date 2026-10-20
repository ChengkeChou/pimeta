"""Binary-outcome meta-analysis (meta::metabin / metafor::rma.mh).

Reproduces the three canonical pooling methods for 2x2 data:

- ``Inverse``  : inverse-variance on the effect sizes from :mod:`.escalc`.
- ``MH``       : Mantel-Haenszel (Greenland & Robins 1985; Robins, Breslow &
                 Greenland 1986; Sato 1989 for RD). Fixed effect only pooled
                 via the MH weights; random effects reuse the DL-family tau^2
                 on the study-level log estimates.
- ``Peto``     : Peto's one-step odds ratio (Yusuf et al. 1985), OR only.

All ratio measures (RR/OR/PETO) are pooled on the log scale and returned
back-transformed, exactly like the R packages.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from scipy import stats

from .data_compat import resolve
from .escalc import escalc
from .pooling import PooledResult, _back_transform, pool_yv

Z_975 = stats.norm.ppf(0.975)

# Which pooling methods support which summary measures.
_MH_SM = {"RR", "OR", "RD"}
_PETO_SM = {"OR"}
_SM = {"RR", "OR", "RD", "AS"}


@dataclass
class MetabResult:
    """Binary meta-analysis result: fixed + random pooled effects."""

    sm: str                       # summary measure
    method: str                   # "Inverse" | "MH" | "Peto" | "LRP"
    k: int
    fixed: PooledResult
    random: PooledResult | None
    yi: list[float]               # per-study effect (analysis scale)
    vi: list[float]               # per-study sampling variance (analysis scale)
    phi: float = 1.0              # dispersion factor (method="LRP" only)
    # tau^2 estimator actually used (R's x$method.tau; metabias' Thompson
    # variant reuses it)
    method_tau: str = "REML"
    # raw 2x2 counts, kept so that metabias() can run the tests that need them
    # (Harbord, Peters, Deeks, Schwarzer).
    event_e: list[float] | None = None
    n_e: list[float] | None = None
    event_c: list[float] | None = None
    n_c: list[float] | None = None

    def to_dict(self) -> dict:
        return {
            "sm": self.sm,
            "method": self.method,
            "k": self.k,
            "phi": self.phi,
            "method_tau": self.method_tau,
            "fixed": _result_dict(self.fixed),
            "random": _result_dict(self.random) if self.random else None,
            "yi": self.yi,
            "vi": self.vi,
        }


def _result_dict(r: PooledResult | None) -> dict | None:
    if r is None:
        return None
    return {
        "pooled_effect": r.pooled_effect,
        "ci_lower": r.pooled_ci_lower,
        "ci_upper": r.pooled_ci_upper,
        "est": r.pooled_est,
        "se": r.pooled_se,
        "z": r.z,
        "p": r.p_value,
        "q": r.q,
        "q_df": r.q_df,
        "q_p": r.q_p,
        "tau2": r.tau2,
        # meta package stores $I2 as a proportion (0-1); PooledResult.i2 is
        # the percentage (0-100, metafor convention), so divide by 100.
        "i2": r.i2 / 100.0,
        "h2": r.h2,
        "weights": r.weights,
        "pi_lower": r.pi_lower,
        "pi_upper": r.pi_upper,
    }


def _cochran_q(
    y: list[float], v: list[float], center: float | None = None
) -> tuple[float, float]:
    """Cochran's Q and its p-value (fixed-effect heterogeneity).

    ``center`` optionally pins the reference value that the squared deviations
    are measured from. R's ``meta`` package uses this for Mantel-Haenszel +
    DerSimonian-Laird (its ``Q.Cochrane`` option): the Q and DL tau^2 are then
    centered on the MH common-effect estimate instead of the IV mean.
    """
    w = [1.0 / vi for vi in v]
    sw = sum(w)
    if center is None:
        mu = sum(wi * yi for wi, yi in zip(w, y)) / sw
    else:
        mu = center
    q = sum(wi * (yi - mu) ** 2 for wi, yi in zip(w, y))
    df = len(y) - 1
    p = float(stats.chi2.sf(q, df)) if df > 0 else math.nan
    return q, p


def _fixed_result(
    est: float,
    se: float,
    sm: str,
    k: int,
    q: float,
    q_p: float,
    tau2: float = 0.0,
) -> PooledResult:
    """Build a fixed-effect PooledResult from an externally pooled (est, se)."""
    z = est / se
    p = 2.0 * float(stats.norm.cdf(-abs(z)))
    ci_l = est - Z_975 * se
    ci_u = est + Z_975 * se
    q_df = k - 1
    i2 = max(0.0, 100.0 * (q - q_df) / q) if q > 0 else 0.0
    h2 = q / q_df if q_df > 0 else math.nan
    return PooledResult(
        model="fixed",
        method="",
        k=k,
        effect_type=sm,
        pooled_effect=_back_transform(est, sm),
        pooled_ci_lower=_back_transform(ci_l, sm),
        pooled_ci_upper=_back_transform(ci_u, sm),
        pooled_est=est,
        pooled_se=se,
        z=z,
        p_value=p,
        q=q,
        q_df=q_df,
        q_p=q_p,
        tau2=tau2,
        i2=i2,
        h2=h2,
    )


def _mh_or(a, b, c, d, n1, n2):
    """Mantel-Haenszel log OR and variance (Robins, Breslow & Greenland 1986)."""
    N = n1 + n2
    F = a * d / N
    G = b * c / N
    H = (a + d) / N
    K = (b + c) / N
    SF = float(np.sum(F))
    SG = float(np.sum(G))
    est = math.log(SF / SG)
    var = (
        float(np.sum(F * H)) / (2 * SF ** 2)
        + float(np.sum(F * K + G * H)) / (2 * SF * SG)
        + float(np.sum(G * K)) / (2 * SG ** 2)
    )
    return est, var


def _mh_rr(a, b, c, d, n1, n2):
    """Mantel-Haenszel log RR and variance (Greenland & Robins 1985)."""
    N = n1 + n2
    R = a * n2 / N
    S = c * n1 / N
    SR = float(np.sum(R))
    SS = float(np.sum(S))
    est = math.log(SR / SS)
    var = float(np.sum((n1 * n2 * (a + c) - a * c * N) / N ** 2)) / (SR * SS)
    return est, var


def _mh_rd(a, b, c, d, n1, n2):
    """Mantel-Haenszel risk difference and variance (Sato 1989)."""
    N = n1 + n2
    w = n1 * n2 / N
    Sw = float(np.sum(w))
    est = float(np.sum((a * n2 - c * n1) / N)) / Sw
    var = float(np.sum((a * b * n2 ** 3 + c * d * n1 ** 3) / (n1 * n2 * N ** 2))) / Sw ** 2
    return est, var


def lrp_common(event_e, n_e, event_c, n_c) -> tuple[float, float, float]:
    """Penalised-likelihood log OR and its SE (``meta:::runLRP``).

    Fits ``cbind(events, nonevents) ~ treat + study`` by Jeffreys-penalised
    logistic regression (Firth), i.e. exactly what ``glm(..., method =
    brglm2::brglmFit, type = "MPL_Jeffreys")`` does.  Study fixed effects make
    every study contribute its own baseline risk, so the treatment coefficient
    is the common (conditional-on-study) log OR.

    Returns ``(est, se, phi)`` where ``phi`` is meta's c-hat dispersion factor
    (1 for a single study or when there is no overdispersion).
    """
    from ._firth import dispersion_phi, firth_logit, treatment_design

    a = np.asarray(event_e, dtype=float)
    n1 = np.asarray(n_e, dtype=float)
    c = np.asarray(event_c, dtype=float)
    n2 = np.asarray(n_c, dtype=float)
    k = a.size

    # Long format: per study the treatment arm first (labelled "B"), control
    # second (labelled "A"), so R's factor() puts "A" (the control) first and
    # the "B" coefficient is the treat-vs-control log OR.
    trt_idx: list[int] = []
    study_idx: list[int] = []
    succ: list[float] = []
    tot: list[float] = []
    for i in range(k):
        trt_idx += [1, 0]
        study_idx += [i, i]
        succ += [float(a[i]), float(c[i])]
        tot += [float(n1[i]), float(n2[i])]

    x = treatment_design(trt_idx, study_idx, 2, k)
    beta, vcov, mu = firth_logit(x, succ, tot)
    est = float(beta[1])
    se = float(np.sqrt(vcov[1, 1]))
    phi = 1.0 if k == 1 else dispersion_phi(succ, tot, mu, x.shape[1])
    return est, se, phi


def _peto_or(a, b, c, d, n1, n2):
    """Peto log OR, variance and heterogeneity Q (Yusuf et al. 1985)."""
    N = n1 + n2
    n = a + c                       # total events per study
    E = n1 * n / N                  # expected treatment events
    V = n1 * n2 * n * (b + d) / (N ** 2 * (N - 1))
    O = a
    OE = float(np.sum(O - E))
    Vsum = float(np.sum(V))
    est = OE / Vsum
    var = 1.0 / Vsum
    q = float(np.sum((O - E) ** 2 / V)) - OE ** 2 / Vsum
    return est, var, q


def _pool_mh_dl(
    y: list[float],
    v: list[float],
    sm: str,
    center: float,
    alpha: float,
) -> PooledResult:
    """Random-effects pool for MH + DL (R ``meta`` ``Q.Cochrane``).

    ``meta``'s default ``Q.Cochrane`` setting turns on whenever
    ``method == "MH"`` and ``method.tau == "DL"``. In that case the Cochran Q
    and the DerSimonian-Laird tau^2 are centered on the Mantel-Haenszel
    common-effect estimate (``TE.common``) rather than the inverse-variance
    mean, while the pooled random-effects estimate still uses ordinary
    inverse-variance weights ``1/(vi + tau2)``.
    """
    k = len(y)
    w = [1.0 / vi for vi in v]
    sw = sum(w)
    q = sum(wi * (yi - center) ** 2 for wi, yi in zip(w, y))
    c = sw - sum(wi ** 2 for wi in w) / sw
    tau2 = max(0.0, (q - (k - 1)) / c) if k > 1 else 0.0

    w2 = [1.0 / (vi + tau2) for vi in v]
    sw2 = sum(w2)
    est = sum(wi * yi for wi, yi in zip(w2, y)) / sw2
    se = math.sqrt(1.0 / sw2)
    z = est / se
    p = 2.0 * float(stats.norm.cdf(-abs(z)))
    ci_l = est - Z_975 * se
    ci_u = est + Z_975 * se
    q_p = float(stats.chi2.sf(q, k - 1)) if k > 1 else math.nan

    # tau2-based I^2 / H^2 (metafor convention, shared by meta::metabin)
    tr_p = sw - sum(wi ** 2 for wi in w) / sw
    vt = (k - 1) / tr_p if k > 1 else 0.0
    i2 = 100.0 * tau2 / (vt + tau2)
    h2 = tau2 / vt + 1.0

    pi_lower = pi_upper = None
    if k > 2:
        t_crit = stats.t.ppf(1 - alpha / 2, df=k - 2)
        pi_se = math.sqrt(tau2 + se ** 2)
        pi_lower = est - t_crit * pi_se
        pi_upper = est + t_crit * pi_se

    return PooledResult(
        model="random",
        method="DL",
        k=k,
        effect_type=sm,
        pooled_effect=_back_transform(est, sm),
        pooled_ci_lower=_back_transform(ci_l, sm),
        pooled_ci_upper=_back_transform(ci_u, sm),
        pooled_est=est,
        pooled_se=se,
        z=z,
        p_value=p,
        q=q,
        q_df=k - 1,
        q_p=q_p,
        tau2=tau2,
        i2=i2,
        h2=h2,
        weights=w2,
        pi_lower=_back_transform(pi_lower, sm) if pi_lower is not None else None,
        pi_upper=_back_transform(pi_upper, sm) if pi_upper is not None else None,
    )


def metabin(
    event_e,
    n_e,
    event_c,
    n_c,
    sm: str = "RR",
    method: str = "Inverse",
    random: bool = True,
    method_tau: str = "REML",
    alpha: float = 0.05,
    data=None,
) -> MetabResult:
    """Pool binary (2x2) study data (mirrors ``meta::metabin``).

    Args:
        event_e, n_e: treatment-arm events and group size (per study).
        event_c, n_c: control-arm events and group size (per study).
        sm: summary measure, one of RR/OR/RD/AS.
        method: "Inverse", "MH", "Peto" or "LRP" (penalised logistic
            regression, Firth/Jeffreys; ``sm="OR"`` only).
        random: also fit the random-effects model (method_tau tau^2 estimator).
            For ``method="LRP"`` the "random" result is the common estimate with
            its SE multiplied by the dispersion factor ``phi``.
        method_tau: DL/HE/HS/SJ/ML/REML/PM/EB (default DL, matching meta).
            Unused by "LRP", which has no tau^2 (``tau2`` is NaN, as in R).
        data: optional frame; the four column arguments may then be names.
    """
    if data is not None:
        event_e, n_e, event_c, n_c = (
            resolve(v, data, name=n) for v, n in
            ((event_e, "event_e"), (n_e, "n_e"),
             (event_c, "event_c"), (n_c, "n_c")))
    sm = sm.upper()
    method = method.capitalize()
    if sm not in _SM:
        raise ValueError(f"sm must be one of {sorted(_SM)}")
    if method not in ("Inverse", "Mh", "Peto", "Lrp"):
        raise ValueError("method must be 'Inverse', 'MH', 'Peto' or 'LRP'")
    if method == "Lrp" and sm != "OR":
        raise ValueError(
            "logistic regression with penalised likelihood only possible with "
            "sm='OR'"
        )

    a = np.asarray(event_e, dtype=float)
    n1 = np.asarray(n_e, dtype=float)
    c = np.asarray(event_c, dtype=float)
    n2 = np.asarray(n_c, dtype=float)
    b = n1 - a
    d = n2 - c
    N = n1 + n2
    k = len(a)

    if k == 0:
        raise ValueError("no studies supplied")
    counts = dict(
        event_e=[float(v) for v in a], n_e=[float(v) for v in n1],
        event_c=[float(v) for v in c], n_c=[float(v) for v in n2],
        method_tau=str(method_tau).upper(),
    )

    # Study-level effect sizes (inverse-variance estimates; also used for
    # the random-effects tau^2 and the Cochran Q).
    esc = escalc(sm, ai=a, bi=b, ci=c, di=d, add=0.5)
    yi, vi, ni = esc.yi, esc.vi, esc.ni

    if method == "Inverse":
        fixed = pool_yv(yi, vi, ni, sm, model="fixed", alpha=alpha)
    elif method == "Mh":
        if sm == "OR":
            mh_est, var = _mh_or(a, b, c, d, n1, n2)
        elif sm == "RR":
            mh_est, var = _mh_rr(a, b, c, d, n1, n2)
        elif sm == "RD":
            mh_est, var = _mh_rd(a, b, c, d, n1, n2)
        else:
            raise ValueError(f"MH supports RR/OR/RD, not {sm}")
        # meta::metabin defaults Q.Cochrane = TRUE iff method=="MH" &&
        # method.tau=="DL": Cochran Q and DL tau^2 are then centered on the
        # MH common-effect estimate instead of the inverse-variance mean.
        use_q_cochrane = method_tau.upper() == "DL"
        q, q_p = _cochran_q(yi, vi, center=mh_est if use_q_cochrane else None)
        fixed = _fixed_result(mh_est, math.sqrt(var), sm, k, q, q_p)
    elif method == "Lrp":
        lrp_est, lrp_se, phi_lrp = lrp_common(a, n1, c, n2)
        # R still reports the usual Cochran Q / I^2 for LRP, but no tau^2.
        q, q_p = _cochran_q(yi, vi)
        lrp_fixed = _fixed_result(lrp_est, lrp_se, sm, k, q, q_p, tau2=math.nan)
        if not random:
            return MetabResult(sm, method, k, lrp_fixed, None, yi, vi, phi_lrp,
                               **counts)
        # "Random" for LRP is the common estimate with the SE inflated by phi.
        random_result = _fixed_result(
            lrp_est, lrp_se * phi_lrp, sm, k, q, q_p, tau2=math.nan
        )
        random_result.model = "random"
        random_result.method = "LRP"
        return MetabResult(sm, method, k, lrp_fixed, random_result, yi, vi,
                           phi_lrp, **counts)
    else:  # Peto
        if sm != "OR":
            raise ValueError("Peto method supports OR only")
        est, var, q = _peto_or(a, b, c, d, n1, n2)
        q_p = float(stats.chi2.sf(q, k - 1)) if k > 1 else math.nan
        fixed = _fixed_result(est, math.sqrt(var), sm, k, q, q_p)

    if random:
        if method == "Mh" and method_tau.upper() == "DL":
            random_result = _pool_mh_dl(yi, vi, sm, mh_est, alpha)
        else:
            random_result = pool_yv(
                yi, vi, ni, sm, model="random", method=method_tau, alpha=alpha
            )
    else:
        random_result = None

    return MetabResult(sm, method, k, fixed, random_result, yi, vi, 1.0,
                       **counts)
