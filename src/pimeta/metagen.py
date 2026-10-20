"""Generic inverse-variance meta-analysis (``meta::metagen``).

Replicates ``meta::metagen`` — the flagship generic meta-analysis function of
the R ``meta`` package — by transcribing its source directly. The function
pools pre-computed treatment estimates and standard errors with
inverse-variance weighting, on either the fixed (common) effect or
random-effects model.

Scale conventions (exactly like R):

* The input ``TE`` is already on the **analysis scale** (``transf = TRUE`` by
  default): ``lnRR`` for ``sm="RR"``, ``lnOR`` for ``sm="OR"``, logit for
  ``sm="PLOGIT"``, Fisher-z for ``sm="ZCOR"``, and so on.
* Every numeric field of the returned result is reported on the **analysis
  scale** (identical to R's ``$TE.common``, ``$lower.random``, ...). The
  report-scale back-transformation is exposed via :func:`backtransf` and the
  :meth:`MetagenResult.report` / :meth:`MetagenResult.to_dict_report` helpers.

Heterogeneity (``meta``'s Q-based convention, *not* metafor's tau^2-based one):

* ``H = sqrt(Q / (k - 1))``
* ``I2 = (H^2 - 1) / H^2``  (equivalently ``(Q - (k-1)) / Q``), a proportion
  in ``[0, 1]``.
* ``H``/``I2`` confidence intervals come from Higgins-Thompson's approximation
  on the log-H scale (``meta:::calcH`` + ``meta:::isquared``).
"""

from __future__ import annotations

import math
import warnings
from dataclasses import dataclass, field

import numpy as np

from .missing import MissingDataWarning as _MissingDataWarning
from scipy import stats

from .metaprop import _asin2p
from .metarate import _asin2ir
from .pooling import _metafor_fisher_scoring_tau2
from .tau2 import estimate_tau2

Z_975 = stats.norm.ppf(0.975)  # qnorm(0.975)


def _qt(p: float, df: float) -> float:
    """High-precision Student-t quantile, matching R's ``qt``.

    ``scipy.stats.t.ppf`` loses up to ~1e-10 relative accuracy for small ``df``
    (enough to perturb a prediction interval in the 12th digit). R inverts the
    regularised incomplete beta function instead; we do the same via
    ``scipy.special.betaincinv`` and reproduce R's value to ~1e-15.

    For ``p > 0.5``:  ``t = sqrt(df * (1/x - 1))``  where
    ``x = I^{-1}(df/2, 1/2 ; 2*(1-p))``  (the CDF tail relation of the t
    distribution). The lower tail follows by symmetry.
    """
    from scipy.special import betaincinv

    if df <= 0.0 or not math.isfinite(df):
        return math.nan
    p = float(p)
    if p < 0.0 or p > 1.0:
        return math.nan
    if p == 0.5:
        return 0.0
    if p < 0.5:
        return -_qt(1.0 - p, df)
    x = betaincinv(df / 2.0, 0.5, 2.0 * (1.0 - p))
    return math.sqrt(df * (1.0 / x - 1.0))

# ---------------------------------------------------------------------------
# summary-measure categories (meta::gs)
# ---------------------------------------------------------------------------
SM_BIN = {"OR", "RD", "RR", "ASD", "DOR", "VE"}
SM_CONT = {"MD", "SMD", "ROM"}
SM_COR = {"ZCOR", "COR"}
SM_INC = {"IRR", "IRD", "IRSD", "VE"}
SM_MEAN = {"MRAW", "MLN"}
SM_PROP = {"PLOGIT", "PLN", "PRAW", "PAS", "PFT"}
SM_RATE = {"IR", "IRLN", "IRS", "IRFT"}

ALL_SM = SM_BIN | SM_CONT | SM_COR | SM_INC | SM_MEAN | SM_PROP | SM_RATE

# meta:::is_relative_effect  +  meta:::is_log_effect
RELATIVE_EFFECT = {"HR", "OR", "RR", "IRR", "ROM", "DOR"}
LOG_EFFECT = {"PLN", "IRLN", "MLN"}
_LOG_SM = RELATIVE_EFFECT | LOG_EFFECT


# ---------------------------------------------------------------------------
# transformation helpers (meta source, transcribed literally)
# ---------------------------------------------------------------------------
def _cor2z(x):
    """meta::cor2z — Fisher's r-to-z transform."""
    return 0.5 * np.log((1.0 + x) / (1.0 - x))


def _z2cor(x):
    """meta::z2cor — inverse Fisher transform."""
    return np.tanh(x)


def _p2logit(x):
    """meta::p2logit — qlogis(x)."""
    return np.log(x / (1.0 - x))


def _logit2p(x):
    """meta::logit2p — plogis(x)."""
    return 1.0 / (1.0 + np.exp(-x))


def _p2asin(x):
    """meta::p2asin."""
    return np.arcsin(np.sqrt(x))


def _ve2logvr(x):
    """meta::VE2logVR."""
    return np.log(1.0 - x / 100.0)


def _logvr2ve(x):
    """meta::logVR2VE."""
    return 100.0 * (1.0 - np.exp(x))


def transf(x, sm: str):
    """meta::transf — transform a value to the analysis scale.

    ``x`` may be a scalar or an array; a NumPy array is returned.
    """
    x = np.asarray(x, dtype=float)
    sm = (sm or "").upper()
    if sm in _LOG_SM:
        return np.log(x)
    if sm == "ZCOR":
        return _cor2z(x)
    if sm == "PLOGIT":
        return _p2logit(x)
    if sm == "PAS":
        return _p2asin(x)
    if sm == "IRS":
        return np.sqrt(x)
    if sm == "VE":
        return _ve2logvr(x)
    return x


def backtransf(x, sm: str, n=None, time=None):
    """meta::backtransf — back-transform an analysis-scale value to report scale.

    ``n`` is required for ``sm="PFT"`` and ``time`` for ``sm="IRFT"``. Returns
    a NumPy array (or scalar-convertible value for scalar input).
    """
    x = np.asarray(x, dtype=float)
    sm = (sm or "").upper()
    if sm in _LOG_SM:
        res = np.exp(x)
    elif sm == "ZCOR":
        res = _z2cor(x)
    elif sm == "PLOGIT":
        res = _logit2p(x)
    elif sm == "PAS":
        res = np.sin(x) ** 2
    elif sm == "PFT":
        if n is None:
            raise ValueError(
                "Argument 'n' must be provided to back transform "
                "Freeman-Tukey transformed proportions."
            )
        res = np.array([_asin2p(xi, ni) for xi, ni in _zip_nt(x, n)])
    elif sm == "IRS":
        res = x ** 2
    elif sm == "IRFT":
        if time is None:
            raise ValueError(
                "Argument 'time' must be provided to back transform "
                "Freeman-Tukey transformed incidence rates."
            )
        res = np.array([_asin2ir(xi, ti) for xi, ti in _zip_nt(x, time)])
    elif sm == "VE":
        res = _logvr2ve(x)
    else:
        res = x
    # meta clamps the back-transformed value back into the natural range.
    if sm == "PRAW":
        res = np.clip(res, 0.0, 1.0)
    if sm == "PLN":
        res = np.clip(res, 0.0, 1.0)
    if sm == "IR":
        res = np.maximum(res, 0.0)
    return res


def _zip_nt(x, nt):
    """Broadcast a scalar/array n (or time) against the x array element-wise."""
    x = np.asarray(x)
    if np.isscalar(nt) or (isinstance(nt, (list, tuple, np.ndarray)) and len(np.atleast_1d(nt)) == 1):
        val = float(np.atleast_1d(nt)[0])
        return [(float(xi), val) for xi in x]
    return [(float(xi), float(ti)) for xi, ti in zip(x, nt)]


# ---------------------------------------------------------------------------
# heterogeneity statistics (meta:::calcH, meta:::isquared, meta:::hetcalc)
# ---------------------------------------------------------------------------
def _calc_h(Q: float, df: int, level: float) -> dict:
    """meta:::calcH — H = max(sqrt(Q/df), 1) with an approximate CI.

    Mirrors ``meta:::calcH``: ``H`` is floored at 1 before the log-scale CI, so
    ``I^2`` is never negative (matches R exactly for Q < k-1).
    """
    k = df + 1
    if math.isnan(Q):
        return {"TE": math.nan, "lower": math.nan, "upper": math.nan}
    if k > 1:
        h = math.sqrt(Q / (k - 1))
    else:
        h = math.nan

    if Q > k and k >= 2:
        selog_h = 0.5 * (math.log(Q) - math.log(k - 1)) / (
            math.sqrt(2.0 * Q) - math.sqrt(2.0 * k - 3.0)
        )
    elif k > 2:
        selog_h = math.sqrt(1.0 / (2.0 * (k - 2)) * (1.0 - 1.0 / (3.0 * (k - 2) ** 2)))
    else:
        selog_h = math.nan

    if math.isnan(h):
        return {"TE": math.nan, "lower": math.nan, "upper": math.nan}

    log_h = math.log(max(h, 1.0))
    te = math.exp(log_h)
    if math.isnan(selog_h):
        return {"TE": te, "lower": math.nan, "upper": math.nan}

    z = stats.norm.ppf(1.0 - (1.0 - level) / 2.0)
    return {
        "TE": te,
        "lower": max(math.exp(log_h - z * selog_h), 1.0),
        "upper": max(math.exp(log_h + z * selog_h), 1.0),
    }


def _isquared(Q: float, df: int, level: float) -> dict:
    """meta:::isquared — I^2 = (H^2 - 1)/H^2."""
    t = _calc_h(Q, df, level)
    f = lambda hh: (hh ** 2 - 1.0) / hh ** 2  # noqa: E731
    return {
        "TE": f(t["TE"]) if not math.isnan(t["TE"]) else math.nan,
        "lower": f(t["lower"]) if not math.isnan(t["lower"]) else math.nan,
        "upper": f(t["upper"]) if not math.isnan(t["upper"]) else math.nan,
    }


def _hetcalc(y, v, method_tau: str, level: float) -> dict:
    """meta:::hetcalc for the default (no subgroup, no ``TE.tau``) case.

    ``tau2`` comes from ``metafor::rma.uni`` (so DL is the DerSimonian-Laird
    moment estimator, ML/REML the Fisher-scoring loop); ``Q`` is Cochran's Q.
    """
    y = np.asarray(y, dtype=float).ravel()
    v = np.asarray(v, dtype=float).ravel()
    sel = ~(np.isnan(y) | np.isnan(v))
    y, v = y[sel], v[sel]
    k = int(y.shape[0])

    if k < 2:
        Q = 0.0 if k == 1 else math.nan
        df_q = 0 if k == 1 else 0
        pval_q = math.nan
        tau2 = math.nan
        tau = math.nan
        h = _calc_h(Q, df_q, level)
        i2 = _isquared(Q, df_q, level)
        return {
            "k": k, "Q": Q, "df_Q": df_q, "pval_Q": pval_q,
            "tau2": tau2, "tau": tau,
            "H": h["TE"], "lower_H": h["lower"], "upper_H": h["upper"],
            "I2": i2["TE"], "lower_I2": i2["lower"], "upper_I2": i2["upper"],
        }

    method_tau = (method_tau or "REML").upper()
    if method_tau in ("ML", "REML"):
        X = np.ones((k, 1))
        tau2 = _metafor_fisher_scoring_tau2(y.tolist(), v.tolist(), X, method_tau)
    else:
        tau2 = estimate_tau2(y.tolist(), v.tolist(), method_tau, None)
    tau = math.sqrt(tau2)

    w = 1.0 / v
    mu = float(np.sum(w * y) / np.sum(w))
    Q = float(np.sum(w * (y - mu) ** 2))
    df_q = k - 1
    pval_q = float(stats.chi2.sf(Q, df_q))

    h = _calc_h(Q, df_q, level)
    i2 = _isquared(Q, df_q, level)
    return {
        "k": k, "Q": Q, "df_Q": df_q, "pval_Q": pval_q,
        "tau2": tau2, "tau": tau,
        "H": h["TE"], "lower_H": h["lower"], "upper_H": h["upper"],
        "I2": i2["TE"], "lower_I2": i2["lower"], "upper_I2": i2["upper"],
    }


# ---------------------------------------------------------------------------
# result object
# ---------------------------------------------------------------------------
@dataclass
class MetagenResult:
    """``meta::metagen`` result (all numeric fields on the analysis scale)."""

    sm: str
    null_effect: float
    # study-level (analysis scale)
    TE: list[float]
    seTE: list[float]
    statistic: list[float]
    pval: list[float]
    lower: list[float]
    upper: list[float]
    # counts
    k: int
    k_all: int
    k_TE: int
    # common (fixed) effect
    w_common: list[float]
    TE_common: float
    seTE_common: float
    statistic_common: float
    pval_common: float
    lower_common: float
    upper_common: float
    # random effects
    w_random: list[float]
    TE_random: float
    seTE_random: float
    seTE_classic: float
    statistic_random: float
    pval_random: float
    df_random: float
    lower_random: float
    upper_random: float
    # heterogeneity
    Q: float
    df_Q: int
    pval_Q: float
    method_tau: str
    tau2: float
    tau: float
    H: float
    lower_H: float
    upper_H: float
    I2: float
    lower_I2: float
    upper_I2: float
    # prediction interval
    method_predict: str
    seTE_predict: float
    df_predict: float
    lower_predict: float
    upper_predict: float
    # reporting helpers
    studlab: list[str] = field(default_factory=list)

    # -- back-transformation helpers ----------------------------------------
    def report(self, n=None, time=None):
        """Back-transform the pooled estimates to the report scale.

        Returns a dict with ``common`` / ``random`` / ``prediction`` entries
        holding ``(est, lower, upper)`` on the report (e.g. RR, OR) scale.
        """
        bt = lambda v: float(backtransf(v, self.sm, n=n, time=time))  # noqa: E731
        return {
            "common": {
                "est": bt(self.TE_common),
                "lower": bt(self.lower_common),
                "upper": bt(self.upper_common),
            },
            "random": {
                "est": bt(self.TE_random),
                "lower": bt(self.lower_random),
                "upper": bt(self.upper_random),
            },
            "prediction": {
                "est": bt(self.TE_random),
                "lower": bt(self.lower_predict),
                "upper": bt(self.upper_predict),
            },
        }

    def to_dict(self) -> dict:
        """Serialize to R-``$``-named keys (analysis scale)."""
        return {
            "sm": self.sm,
            "null.effect": self.null_effect,
            "TE": self.TE,
            "seTE": self.seTE,
            "statistic": self.statistic,
            "pval": self.pval,
            "lower": self.lower,
            "upper": self.upper,
            "k": self.k,
            "k.all": self.k_all,
            "k.TE": self.k_TE,
            "w.common": self.w_common,
            "TE.common": self.TE_common,
            "seTE.common": self.seTE_common,
            "statistic.common": self.statistic_common,
            "pval.common": self.pval_common,
            "lower.common": self.lower_common,
            "upper.common": self.upper_common,
            "w.random": self.w_random,
            "TE.random": self.TE_random,
            "seTE.random": self.seTE_random,
            "seTE.classic": self.seTE_classic,
            "statistic.random": self.statistic_random,
            "pval.random": self.pval_random,
            "df.random": self.df_random,
            "lower.random": self.lower_random,
            "upper.random": self.upper_random,
            "Q": self.Q,
            "df.Q": self.df_Q,
            "pval.Q": self.pval_Q,
            "method.tau": self.method_tau,
            "tau2": self.tau2,
            "tau": self.tau,
            "H": self.H,
            "lower.H": self.lower_H,
            "upper.H": self.upper_H,
            "I2": self.I2,
            "lower.I2": self.lower_I2,
            "upper.I2": self.upper_I2,
            "method.predict": self.method_predict,
            "seTE.predict": self.seTE_predict,
            "df.predict": self.df_predict,
            "lower.predict": self.lower_predict,
            "upper.predict": self.upper_predict,
            "studlab": self.studlab,
        }


# ---------------------------------------------------------------------------
# main entry point
# ---------------------------------------------------------------------------
def metagen(
    TE,
    seTE,
    studlab=None,
    sm: str = "",
    method_tau: str = "REML",
    method_random_ci: str = "classic",
    method_predict: str = "V",
    common: bool = True,
    random: bool = True,
    prediction: bool = False,
    level: float = 0.95,
    level_predict: float = 0.95,
    null_effect: float = 0.0,
    exclude=None,
    warn_missing: bool = True,
) -> MetagenResult:
    """Generic inverse-variance meta-analysis (mirrors ``meta::metagen``).

    Args:
        TE: treatment estimates, already on the analysis scale (e.g. ``lnRR``).
        seTE: standard errors (analysis scale).
        studlab: optional study labels.
        sm: summary measure (``""``, ``"RR"``, ``"OR"``, ...); only used for
            report-scale back-transformation and validation.
        method_tau: tau-squared estimator (``"REML"`` default, ``"DL"``, ``"ML"``,
            ``"HE"``, ``"HS"``, ``"SJ"``, ``"PM"``, ``"EB"``).
        method_random_ci: random-effects CI method (``"classic"`` default).
        method_predict: prediction-interval method (``"V"`` default).
        common: compute the common (fixed) effect model.
        random: compute the random-effects model.
        prediction: compute the prediction interval.
        level: confidence level for the pooled estimates.
        level_predict: confidence level for the prediction interval.
        null_effect: null hypothesis value on the analysis scale (default 0).

    Returns a :class:`MetagenResult` whose numeric fields are on the analysis
    scale, exactly matching R's ``$TE.common``, ``$lower.random``, etc.
    """
    sm = (sm or "").upper()
    if sm not in ALL_SM and sm != "":
        raise ValueError(f"unknown sm: {sm!r}")

    TE = np.asarray(TE, dtype=float).ravel()
    seTE = np.asarray(seTE, dtype=float).ravel()
    if TE.shape[0] != seTE.shape[0] or TE.shape[0] == 0:
        raise ValueError("TE and seTE must be non-empty equal-length arrays")

    k_all = int(TE.shape[0])
    if exclude is None:
        exclude = np.zeros(k_all, dtype=bool)
    else:
        exclude = np.asarray(exclude, dtype=bool).ravel()

    if studlab is None:
        studlab = [f"Study {i + 1}" for i in range(k_all)]
    studlab = list(studlab)

    # meta ignores infinite values in the pooling (sel.noInf).
    use = ~(np.isnan(TE) | np.isnan(seTE)) & ~np.isinf(TE) & ~np.isinf(seTE) & ~exclude
    k = int(np.sum(use))
    k_TE = int(np.sum(~np.isnan(TE)))

    dropped = int(k_all - k)
    if warn_missing and dropped:
        rows = np.flatnonzero(~use)
        shown = ", ".join(str(int(i) + 1) for i in rows[:10])
        more = f", ... (+{rows.size - 10} more)" if rows.size > 10 else ""
        warnings.warn(
            f"dropped {dropped} of {k_all} studies with missing or infinite "
            f"'TE'/'seTE' (row{'s' if rows.size > 1 else ''} {shown}{more}); "
            "pass warn_missing=False to silence this",
            _MissingDataWarning,
            stacklevel=2,
        )

    z = stats.norm.ppf(1.0 - (1.0 - level) / 2.0)  # qnorm((1+level)/2)

    # study-level CI (analysis scale, z reference, meta:::ci with df=NULL).
    statistic = (TE - null_effect) / seTE
    pval = 2.0 * stats.norm.sf(np.abs(statistic))
    lower = TE - z * seTE
    upper = TE + z * seTE

    def _nan():
        return math.nan

    if k == 0:
        empty = [0.0] * k_all
        return MetagenResult(
            sm=sm, null_effect=null_effect,
            TE=TE.tolist(), seTE=seTE.tolist(),
            statistic=statistic.tolist(), pval=pval.tolist(),
            lower=lower.tolist(), upper=upper.tolist(),
            k=0, k_all=k_all, k_TE=k_TE,
            w_common=empty, TE_common=_nan(), seTE_common=_nan(),
            statistic_common=_nan(), pval_common=_nan(),
            lower_common=_nan(), upper_common=_nan(),
            w_random=empty, TE_random=_nan(), seTE_random=_nan(),
            seTE_classic=_nan(), statistic_random=_nan(), pval_random=_nan(),
            df_random=_nan(), lower_random=_nan(), upper_random=_nan(),
            Q=_nan(), df_Q=0, pval_Q=_nan(), method_tau=method_tau.upper(),
            tau2=_nan(), tau=_nan(), H=_nan(), lower_H=_nan(), upper_H=_nan(),
            I2=_nan(), lower_I2=_nan(), upper_I2=_nan(),
            method_predict=method_predict, seTE_predict=_nan(), df_predict=_nan(),
            lower_predict=_nan(), upper_predict=_nan(), studlab=studlab,
        )

    hc = _hetcalc(TE[use], seTE[use] ** 2, method_tau, level)
    tau2_calc = 0.0 if math.isnan(hc["tau2"]) else hc["tau2"]

    v = seTE ** 2
    # ---- common effect ---------------------------------------------------
    if common and k >= 1:
        w_common = np.where(use, 1.0 / v, 0.0)
        sw = float(np.sum(w_common))
        te_common = float(np.sum(w_common * TE) / sw)
        se_common = math.sqrt(1.0 / sw)
        stat_common = (te_common - null_effect) / se_common
        pval_common = 2.0 * stats.norm.sf(abs(stat_common))
        lower_common = te_common - z * se_common
        upper_common = te_common + z * se_common
    else:
        w_common = np.zeros(k_all)
        te_common = se_common = stat_common = pval_common = _nan()
        lower_common = upper_common = _nan()

    # ---- random effects --------------------------------------------------
    if random and k >= 1:
        w_random = np.where(use, 1.0 / (v + tau2_calc), 0.0)
        swr = float(np.sum(w_random))
        te_random = float(np.sum(w_random * TE) / swr)
        se_classic = math.sqrt(1.0 / swr)
        se_random = se_classic
        stat_random = (te_random - null_effect) / se_random
        pval_random = 2.0 * stats.norm.sf(abs(stat_random))
        lower_random = te_random - z * se_random
        upper_random = te_random + z * se_random
        df_random = math.inf
    else:
        w_random = np.zeros(k_all)
        te_random = se_random = se_classic = stat_random = pval_random = _nan()
        lower_random = upper_random = _nan()
        df_random = _nan()

    # ---- prediction interval ---------------------------------------------
    # meta always computes the prediction interval (default method.predict="V")
    # when the random-effects model is fitted; the ``prediction`` flag only
    # controls whether it is *printed*.
    if random and k >= 2:
        se_predict = math.sqrt(se_classic ** 2 + tau2_calc)
        df_predict = float(k - 1)
        tcrit = _qt(1.0 - (1.0 - level_predict) / 2.0, df_predict)
        lower_predict = te_random - tcrit * se_predict
        upper_predict = te_random + tcrit * se_predict
    else:
        se_predict = df_predict = lower_predict = upper_predict = _nan()

    return MetagenResult(
        sm=sm, null_effect=null_effect,
        TE=TE.tolist(), seTE=seTE.tolist(),
        statistic=statistic.tolist(), pval=pval.tolist(),
        lower=lower.tolist(), upper=upper.tolist(),
        k=k, k_all=k_all, k_TE=k_TE,
        w_common=w_common.tolist(), TE_common=te_common, seTE_common=se_common,
        statistic_common=stat_common, pval_common=pval_common,
        lower_common=lower_common, upper_common=upper_common,
        w_random=w_random.tolist(), TE_random=te_random, seTE_random=se_random,
        seTE_classic=se_classic, statistic_random=stat_random, pval_random=pval_random,
        df_random=df_random, lower_random=lower_random, upper_random=upper_random,
        Q=hc["Q"], df_Q=hc["df_Q"], pval_Q=hc["pval_Q"],
        method_tau=method_tau.upper(),
        tau2=hc["tau2"], tau=hc["tau"],
        H=hc["H"], lower_H=hc["lower_H"], upper_H=hc["upper_H"],
        I2=hc["I2"], lower_I2=hc["lower_I2"], upper_I2=hc["upper_I2"],
        method_predict=method_predict, seTE_predict=se_predict, df_predict=df_predict,
        lower_predict=lower_predict, upper_predict=upper_predict,
        studlab=studlab,
    )
