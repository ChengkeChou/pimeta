"""S3-style generics for the result objects.

R's packages dispatch ``print(x)``, ``coef(x)``, ``plot(x)`` … on
``class(x)``.  Python has no such mechanism, so this module reproduces it with
:func:`functools.singledispatch`::

    coef(result)        # -> {'intrcpt': -0.7145, ...}
    vcov(result)        # -> ndarray
    summary(result)     # -> dict of the printed sections
    plot(result)        # -> the matching figure

Registered handlers cover ``RmaResult``, ``RmaMvResult``, ``RmaGlmmResult``,
``SelModelResult``, ``MatregResult``, ``MetagenResult`` (and every ``meta``
family result), ``NetmetaResult``, ``ProfileResult``, ``PermutestResult``,
``VifResult``, ``DeltaMethodResult``, ``TesResult`` and ``HettestResult``.  An
``object`` fallback resolves the common R field names, so new result types work
without registration.
"""

from __future__ import annotations

import math
from functools import singledispatch

import numpy as np

__all__ = [
    "coef", "vcov", "fitted", "residuals", "logLik", "deviance", "nobs",
    "formula", "AIC", "BIC", "weights", "sigma", "df_residual",
    "model_matrix", "simulate", "confint", "predict", "hatvalues_rma",
    "dfbetas_rma", "rstandard_rma", "ranef_rma",
    "summary", "summary_text", "print_summary", "plot",
]

_NAN = float("nan")


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #

def _first(obj, *names, default=None):
    for n in names:
        if isinstance(obj, dict) and n in obj:
            return obj[n]
        v = getattr(obj, n, None)
        if v is not None:
            return v
    return default


def _opt(obj, name):
    return _first(obj, name, default=None)


def _coefnames(obj, p):
    names = _opt(obj, "coefnames") or _opt(obj, "names") or _opt(obj, "trts")
    if names is not None and len(names) >= p:
        return [str(n) for n in list(names)[:p]]
    if p == 1:
        return ["intrcpt"]
    return ["intrcpt"] + [f"V{i + 1}" for i in range(p - 1)]


def _p(obj, b):
    p = _opt(obj, "p")
    return int(p) if p is not None else int(np.asarray(b).size)


# --------------------------------------------------------------------------- #
# A. value accessors
# --------------------------------------------------------------------------- #

@singledispatch
def coef(x, type="beta"):
    """``coef`` — model coefficients as ``{name: value}``.

    Mirrors ``coef.rma`` / ``coef.matreg`` / ``coef.permutest.rma.uni``: the
    ``"beta"`` coefficients for location models, the ``"alpha"`` coefficients
    for scale models and the ``"delta"`` selection parameters.
    """
    if type == "delta":
        d = _opt(x, "delta")
        if d is None:
            raise ValueError("model has no 'delta' coefficients")
        d = np.atleast_1d(np.asarray(d, dtype=float))
        names = (["delta"] if d.size == 1
                 else [f"delta.{i + 1}" for i in range(d.size)])
        return dict(zip(names, d.tolist()))
    b = _first(x, "b", "beta", "coefficients", "TE_common", "TE")
    if b is None:
        raise TypeError(f"coef() is not defined for {type(x).__name__}")
    b = np.atleast_1d(np.asarray(b, dtype=float))
    return dict(zip(_coefnames(x, b.size), b.tolist()))


@singledispatch
def vcov(x, type="beta"):
    """``vcov`` — variance-covariance matrix of the coefficients."""
    vb = _first(x, "vb", "vcov", "vcov_matrix")
    if vb is None:
        raise TypeError(f"vcov() is not defined for {type(x).__name__}")
    return np.atleast_2d(np.asarray(vb, dtype=float))


@singledispatch
def fitted(x):
    """``fitted`` — fitted values."""
    v = _first(x, "fitted", "fitted_values", "fittedvalues")
    if v is None:
        X = _first(x, "X", "X_matrix")
        b = _first(x, "b", "beta")
        if X is not None and b is not None:
            return np.asarray(X, dtype=float) @ np.asarray(b, dtype=float)
        raise TypeError(f"fitted() is not defined for {type(x).__name__}")
    return np.asarray(v, dtype=float)


@singledispatch
def residuals(x):
    """``residuals`` — raw residuals."""
    v = _first(x, "residuals", "resid")
    if v is not None:
        return np.asarray(v, dtype=float)
    return np.asarray(_first(x, "yi", "TE"), dtype=float) - fitted(x)


@singledispatch
def logLik(x):
    """``logLik`` — maximised (restricted) log-likelihood."""
    ll = _first(x, "ll", "logLik", "loglik")
    if ll is not None:
        return float(np.asarray(ll).ravel()[0])
    from .rma_uni import RmaResult
    if isinstance(x, RmaResult):
        from .utils import fitstats
        fs = fitstats(x)
        return float(fs._get("ll", fs.method == "REML"))
    raise TypeError(f"logLik() is not defined for {type(x).__name__}")


@singledispatch
def deviance(x):
    """``deviance`` — ``-2 * logLik`` under the fitted model."""
    from .rma_uni import RmaResult
    if isinstance(x, RmaResult):
        from .utils import fitstats
        fs = fitstats(x)
        return float(fs._get("dev", fs.method == "REML"))
    return -2.0 * logLik(x)


@singledispatch
def nobs(x):
    """``nobs`` — number of observations used in the fit."""
    for name in ("k", "k_f", "k_eff"):
        v = _opt(x, name)
        if v is not None:
            return int(v)
    raise TypeError(f"nobs() is not defined for {type(x).__name__}")


@singledispatch
def formula(x):
    """``formula`` — the model formula as text."""
    f = _opt(x, "formula_str")
    if f is not None:
        return f
    if _opt(x, "int_only"):
        return "~ 1"
    mods = _opt(x, "mods")
    if isinstance(mods, str):
        return f"~ {mods}"
    if isinstance(mods, (list, tuple)):
        return "~ " + " + ".join(str(m) for m in mods)
    p = _opt(x, "p")
    if p is not None:
        return "~ 1" if int(p) == 1 else "~ mods"
    return "~ 1"


@singledispatch
def AIC(x, k=2, correct=False):
    """``AIC`` — Akaike information criterion (``correct=True`` → AICc)."""
    from .utils import aic_rma
    return aic_rma(x, k=k, correct=correct)


@singledispatch
def BIC(x):
    """``BIC`` — Bayesian information criterion."""
    from .utils import bic_rma
    return bic_rma(x)


@singledispatch
def weights(x, pct=False, type=None):
    """``weights`` — study weights (raw or as percentages)."""
    from .rma_mv import RmaMvResult
    from .rma_peto_mh import is_mh_result, mh_weights

    if is_mh_result(x):
        # weights.rma.mh / weights.rma.peto: percentages by default.
        return mh_weights(x, type="diagonal" if pct is not False else "matrix")
    if isinstance(x, RmaMvResult):
        from .mv_diagnostics import mv_weights

        want = type or ("diagonal" if pct is not False else "matrix")
        return mv_weights(x, type=want)
    w = _first(x, "weights", "w")
    if w is None:
        raise TypeError(f"weights() is not defined for {type(x).__name__}")
    w = np.asarray(w, dtype=float)
    return 100.0 * w / w.sum() if pct else w


@singledispatch
def sigma(x, REML=True):
    """``sigma`` — residual standard deviation (``matreg``)."""
    from .utils import sigma_matreg
    return sigma_matreg(x, REML=REML)


@singledispatch
def df_residual(x):
    """``df.residual`` — residual degrees of freedom."""
    from .utils import df_residual_rma
    return df_residual_rma(x)


@singledispatch
def model_matrix(x, asdf=False):
    """``model.matrix`` — the design matrix used in the fit."""
    from .utils import model_matrix_rma
    return model_matrix_rma(x, asdf=asdf)


@singledispatch
def simulate(x, nsim=1, seed=None):
    """``simulate`` — simulate new effect sizes from the fitted model."""
    from .extras import simulate_rma
    return simulate_rma(x, nsim=nsim, seed=seed)


@singledispatch
def confint(x, level=0.95, type="QP", transf=None):
    """``confint`` — confidence interval of tau^2 (or of the coefficients).

    ``transf`` is only used by the ``rma.mh`` / ``rma.peto`` methods, where
    ``transf=True`` back-transforms the ratio measures with ``exp()``.
    """
    from .rma_peto_mh import is_mh_result, mh_confint

    if is_mh_result(x):
        return mh_confint(x, level=level, transf=transf)
    tau2 = _opt(x, "tau2")
    yi = _opt(x, "yi")
    vi = _opt(x, "vi")
    if tau2 is not None and yi is not None and vi is not None:
        from .pooling import confint_tau2
        method = _opt(x, "method") or "REML"
        r = confint_tau2(list(yi), list(vi), float(tau2), method=method,
                         level=1.0 - level if level > 0.5 else level,
                         type=type)
        return {"tau2": float(tau2), "ci_lb": float(r.tau2_lb),
                "ci_ub": float(r.tau2_ub), "level": level}
    b = coef(x)
    se = _first(x, "se")
    if se is None:
        raise TypeError(f"confint() is not defined for {type(x).__name__}")
    from scipy import stats as _st
    crit = float(_st.norm.isf((1.0 - level) / 2.0))
    se = np.asarray(se, dtype=float)
    return {k: {"estimate": v, "ci.lb": v - crit * se[i],
                "ci.ub": v + crit * se[i]}
            for i, (k, v) in enumerate(b.items())}


@singledispatch
def predict(x, newmods=None, level=0.95, predtype="t"):
    """``predict`` — predicted effect(s) for new moderator values."""
    from .pooling import predict_rma
    b = np.asarray(_first(x, "b", "beta"), dtype=float)
    se = np.asarray(_first(x, "se"), dtype=float)
    tau2 = float(_opt(x, "tau2") or 0.0)
    k = nobs(x)
    p = _p(x, b)
    if newmods is None:
        if p > 1:
            raise ValueError("'newmods' is required for models with moderators")
        return predict_rma(float(b[0]), float(se[0]), tau2, k, p=p,
                           level=1.0 - level, predtype=predtype)
    X = np.asarray(newmods, dtype=float)
    if X.ndim == 1:
        X = X.reshape(1, -1)
    if X.shape[1] == p - 1:
        X = np.column_stack([np.ones(X.shape[0]), X])
    vb = vcov(x)
    out = []
    for i in range(X.shape[0]):
        xi = X[i]
        est = float(xi @ b)
        var = float(xi @ vb @ xi)
        out.append({"pred": est, "se": math.sqrt(var),
                    "ci.lb": est - 1.959963984540054 * math.sqrt(var),
                    "ci.ub": est + 1.959963984540054 * math.sqrt(var)})
    return out


@singledispatch
def hatvalues_rma(x, type="diagonal"):
    """``hatvalues`` — leverages of the studies."""
    from .rma_mv import RmaMvResult

    if isinstance(x, RmaMvResult):
        from .mv_diagnostics import mv_hatvalues

        return mv_hatvalues(x, type=type)
    w = _first(x, "weights", "w")
    X = _first(x, "X", "X_matrix")
    if w is None or X is None:
        raise TypeError(f"hatvalues() is not defined for {type(x).__name__}")
    w = np.asarray(w, dtype=float)
    X = np.asarray(X, dtype=float)
    xtwx = X.T @ (w[:, None] * X)
    return np.einsum("ij,jk,ik->i", X, np.linalg.pinv(xtwx), X) * w


@singledispatch
def dfbetas_rma(x, cluster=None):
    """``dfbetas`` — change in the coefficients when dropping each study."""
    from .rma_mv import RmaMvResult

    if isinstance(x, RmaMvResult):
        from .mv_diagnostics import mv_dfbetas

        return np.asarray(mv_dfbetas(x, cluster=cluster), dtype=float)
    from .sensitivity import influence
    yi = _first(x, "yi")
    vi = _first(x, "vi")
    X = _first(x, "X", "X_matrix")
    if yi is None or vi is None or X is None:
        raise TypeError(f"dfbetas() is not defined for {type(x).__name__}")
    # influence() prepends its own intercept column, so hand it the moderator
    # matrix without the model's intercept (otherwise the refit is singular).
    X = np.asarray(X, dtype=float)
    if X.ndim == 1:
        X = X.reshape(-1, 1)
    if X.shape[1] > 1 and np.allclose(X[:, 0], 1.0):
        X = X[:, 1:]
    elif X.shape[1] == 1 and np.allclose(X[:, 0], 1.0):
        X = None
    inf = influence(list(yi), list(vi), X=X,
                    method=_opt(x, "method") or "REML",
                    tau2=(float(_opt(x, "tau2"))
                          if _opt(x, "tau2_fix") else None),
                    test=_opt(x, "test") or "z")
    return np.asarray(inf.dfbs, dtype=float)


@singledispatch
def rstandard_rma(x, cluster=None):
    """``rstandard`` — standardised residuals."""
    from .rma_mv import RmaMvResult
    from .rma_peto_mh import is_mh_result, mh_rstandard

    if is_mh_result(x):
        return mh_rstandard(x)
    if isinstance(x, RmaMvResult):
        from .mv_diagnostics import mv_rstandard

        return mv_rstandard(x, cluster=cluster)
    res = residuals(x)
    vi = _first(x, "vi")
    tau2 = float(_opt(x, "tau2") or 0.0)
    if vi is None:
        raise TypeError(f"rstandard() is not defined for {type(x).__name__}")
    return res / np.sqrt(np.asarray(vi, dtype=float) + tau2)


@singledispatch
def ranef_rma(x, level=None):
    """``ranef`` — random effects (BLUPs) of the studies."""
    from .utils import blup
    return blup(x)


# --------------------------------------------------------------------------- #
# B. summary / printing
# --------------------------------------------------------------------------- #

@singledispatch
def summary(x):
    """``summary`` — the information ``print``/``summary`` would show.

    Returns a nested dict rather than a formatted string, so it can be fed to
    any renderer.  ``summary_text()`` produces the R-style text block.
    """
    from .metagen import MetagenResult

    out = {"class": type(x).__name__}
    if isinstance(x, MetagenResult):
        out.update({
            "sm": getattr(x, "sm", ""),
            "k": int(x.k),
            "common": {"estimate": x.TE_common, "se": x.seTE_common,
                       "ci.lb": x.lower_common, "ci.ub": x.upper_common,
                       "statistic": x.statistic_common, "pval": x.pval_common},
            "random": {"estimate": x.TE_random, "se": x.seTE_random,
                       "ci.lb": x.lower_random, "ci.ub": x.upper_random,
                       "statistic": x.statistic_random, "pval": x.pval_random},
            "heterogeneity": {"Q": x.Q, "df.Q": x.df_Q, "pval.Q": x.pval_Q,
                              "tau2": x.tau2, "I2": x.I2, "H": getattr(x, "H", None)},
            "method.tau": getattr(x, "method_tau", ""),
        })
        return out

    b = None
    try:
        b = coef(x)
    except (TypeError, ValueError):
        pass
    se = _first(x, "se")
    tau2 = _opt(x, "tau2")
    if b is not None:
        crit = 1.959963984540054
        se_a = np.asarray(se, dtype=float) if se is not None else None
        beta_tab = {}
        for i, (kname, v) in enumerate(b.items()):
            row = {"estimate": v, "se": float(se_a[i]) if se_a is not None
                   and i < se_a.size else _NAN}
            if row["se"] == row["se"]:
                row["ci.lb"] = v - crit * row["se"]
                row["ci.ub"] = v + crit * row["se"]
                row["zval"] = v / row["se"]
                row["pval"] = float(2 * _norm_sf(abs(row["zval"])))
            beta_tab[kname] = row
        out["beta"] = beta_tab
    if tau2 is not None:
        out["tau2"] = float(np.asarray(tau2).ravel()[0])
    for name in ("method", "test", "measure", "level", "k", "p"):
        v = _opt(x, name)
        if v is not None:
            out[name] = v
    for name in ("QE", "QEp", "QM", "QMp", "I2", "H2", "R2", "sigma2"):
        v = _opt(x, name)
        if v is not None:
            out[name] = v
    return out


def _norm_sf(z):
    from scipy import stats as _st
    return float(_st.norm.sf(z))


def summary_text(x, digits=4, level=None):
    """R-style ``print(summary(x))`` text block."""
    s = summary(x)
    f = lambda v: (f"{v:.{digits}f}" if isinstance(v, float) and v == v  # noqa: E731
                   else str(v))
    lines = [f"Class: {s.get('class', '')}"]
    if "sm" in s and s["sm"]:
        lines.append(f"Effect measure: {s['sm']}")
    if "k" in s:
        lines.append(f"Number of studies: {s['k']}")
    if "beta" in s:
        lines.append("")
        lines.append("Model Results:")
        lines.append(f"{'':<12}{'estimate':>10}{'se':>10}{'ci.lb':>10}"
                     f"{'ci.ub':>10}{'pval':>10}")
        for name, row in s["beta"].items():
            lines.append(
                f"{name:<12}{f(row['estimate']):>10}{f(row['se']):>10}"
                f"{f(row.get('ci.lb')):>10}{f(row.get('ci.ub')):>10}"
                f"{f(row.get('pval')):>10}"
            )
    if "common" in s:
        c, r = s["common"], s["random"]
        lines.append("")
        lines.append(f"{'':<22}{'estimate':>10}{'ci.lb':>10}{'ci.ub':>10}")
        lines.append(f"{'Common effect model':<22}{f(c['estimate']):>10}"
                     f"{f(c['ci.lb']):>10}{f(c['ci.ub']):>10}")
        lines.append(f"{'Random effects model':<22}{f(r['estimate']):>10}"
                     f"{f(r['ci.lb']):>10}{f(r['ci.ub']):>10}")
        h = s["heterogeneity"]
        lines.append("")
        lines.append("Heterogeneity:")
        lines.append(f"  Q = {f(h['Q'])}, df = {h['df.Q']}, "
                     f"p = {f(h['pval.Q'])}")
        lines.append(f"  tau^2 = {f(h['tau2'])}, I^2 = {f(h['I2'])}")
        lines.append(f"  tau^2 estimator: {s.get('method.tau', '')}")
    elif "tau2" in s:
        lines.append("")
        lines.append(f"tau^2 = {f(s['tau2'])}")
        for key in ("QE", "QEp", "I2", "H2", "R2"):
            if key in s:
                lines.append(f"{key} = {f(s[key])}")
    return "\n".join(lines)


def print_summary(x, digits=4):
    """Print the R-style summary block and return it."""
    txt = summary_text(x, digits=digits)
    print(txt)
    return txt


# --------------------------------------------------------------------------- #
# C. plot dispatch
# --------------------------------------------------------------------------- #

@singledispatch
def plot(x, which="forest", **kwargs):
    """``plot`` — dispatch to the matching figure for ``x``.

    ``which`` selects the figure (``"forest"``, ``"funnel"``, ``"radial"``,
    ``"labbe"``, ``"qqnorm"``, ``"baujat"``, ``"gosh"``, ``"profile"``,
    ``"bubble"``, ``"drapery"``, ``"regplot"``, ``"influence"``,
    ``"permutest"``).  The study-level ``yi``/``vi`` are extracted from the
    fitted object automatically.

    Passing an already-computed profile object (``plot(profile_rmamv(x))``
    or ``plot(profile_selmodel(x))``) draws the profile even with the default
    ``which``, matching R's S3 dispatch.
    """
    from . import plotting

    name = str(which).lower()
    if name == "forest" and getattr(x, "yi", None) is None and (
            hasattr(x, "sigma2") or hasattr(x, "delta")
            or hasattr(x, "alpha")) and hasattr(x, "ll"):
        name = "profile"
    yi = _opt(x, "yi")
    vi = _opt(x, "vi")
    slab = _opt(x, "slab")
    effect_type = _opt(x, "measure") or _opt(x, "sm") or None

    if name == "forest":
        if yi is None:
            raise TypeError("plot(x, which='forest') needs study-level yi/vi")
        return plotting.forest(yi, vi, studlab=slab, effect_type=effect_type,
                               **kwargs)
    if name == "funnel":
        if yi is None:
            raise TypeError("plot(x, which='funnel') needs study-level yi/vi")
        return plotting.funnel(yi, vi, studlab=slab, effect_type=effect_type,
                               **kwargs)
    if name == "radial":
        if yi is None:
            raise TypeError("plot(x, which='radial') needs study-level yi/vi")
        b = np.asarray(_first(x, "b", "beta"), dtype=float).ravel()
        se = np.asarray(_first(x, "se"), dtype=float).ravel()
        return plotting.radial(yi, vi, pooled=float(b[0]),
                               pooled_se=float(se[0]), studlab=slab, **kwargs)
    if name == "labbe":
        ev_e = _opt(x, "event_e") or _opt(x, "event")
        if ev_e is None:
            raise TypeError(
                "plot(x, which='labbe') needs binary event counts")
        return plotting.labbe(ev_e, _opt(x, "n_e"), _opt(x, "event_c"),
                              _opt(x, "n_c"), studlab=slab, **kwargs)
    if name == "qqnorm":
        return plotting.qqnorm_rma(x, **kwargs)
    if name == "baujat":
        # plot_baujat() consumes a BaujatResult, not the fitted model, so the
        # diagnostic has to be computed first (metafor's plot(baujat(x)) idiom).
        if yi is None or vi is None:
            raise TypeError("plot(x, which='baujat') needs study-level yi/vi")
        from .plotting import plot_baujat
        from .sensitivity import baujat
        return plot_baujat(
            baujat(np.asarray(yi, dtype=float), np.asarray(vi, dtype=float)),
            **kwargs)
    if name == "gosh":
        if yi is None or vi is None:
            raise TypeError("plot(x, which='gosh') needs study-level yi/vi")
        from .gosh import gosh
        from .plotting import plot_gosh
        return plot_gosh(
            gosh(np.asarray(yi, dtype=float), np.asarray(vi, dtype=float)),
            **kwargs)
    if name == "profile":
        from .mv_diagnostics import MvProfileResult, profile_rmamv
        from .selmodel import SelModelResult

        if isinstance(x, SelModelResult):
            # profile.rma.uni.selmodel: profile over the selection parameter
            from .selmodel import profile_selmodel

            prof_keys = ("delta", "n_points", "span", "level")
            prof_kw = {k: kwargs.pop(k) for k in prof_keys if k in kwargs}
            prof_kw.setdefault("n_points", 12)
            return plotting.plot_selprofile(profile_selmodel(x, **prof_kw),
                                            **kwargs)
        if isinstance(x, MvProfileResult):
            # plot() of an already-computed profile.rma.mv
            return plotting.plot_mv_profile(x, **kwargs)
        from .rma_ls import RmaLsProfileResult, RmaLsResult, profile_rma_ls

        if isinstance(x, RmaLsProfileResult):
            return plotting.plot_mv_profile(x, **kwargs)
        if isinstance(x, RmaLsResult):
            # plot(x, which="profile") on a location-scale fit
            prof_keys = ("alpha", "steps", "xlim")
            prof_kw = {k: kwargs.pop(k) for k in prof_keys if k in kwargs}
            return plotting.plot_mv_profile(profile_rma_ls(x, **prof_kw),
                                            **kwargs)
        from .rma_mv import RmaMvResult

        if isinstance(x, RmaMvResult):
            # plot(x, which="profile") on a fit == plot(profile(x))
            prof_keys = ("sigma2", "steps", "xlim")
            prof_kw = {k: kwargs.pop(k) for k in prof_keys if k in kwargs}
            return plotting.plot_mv_profile(profile_rmamv(x, **prof_kw),
                                            **kwargs)
        return plotting.llplot(x, **kwargs)
    if name == "bubble":
        return plotting.bubble(x, **kwargs)
    if name == "drapery":
        return plotting.drapery(x, **kwargs)
    if name == "regplot":
        return plotting.regplot(x, **kwargs)
    if name in ("influence", "influential", "infl"):
        # plot(x, which="influence") == metafor's plot(influence(x))
        from .plotting import plot_influence
        from .sensitivity import influence

        if hasattr(x, "is_infl"):
            res = x
        else:
            if yi is None or vi is None:
                raise TypeError(
                    "plot(x, which='influence') needs study-level yi/vi")
            mods = _opt(x, "X")
            if mods is not None:
                mods = np.asarray(mods, dtype=float)[:, 1:]
            res = influence(
                np.asarray(yi, dtype=float), np.asarray(vi, dtype=float),
                X=mods, X_names=None,
                method=_opt(x, "method") or "REML",
                test=(_opt(x, "test") if _opt(x, "test") in ("z", "knha")
                      else "z"),
            )
        if slab is None:
            slab = _opt(x, "slab")
        return plot_influence(res, slab=slab, **kwargs)
    if name in ("permutest", "permutation"):
        # plot(x, which="permutest") == metafor's plot(permutest(x))
        if getattr(x, "zval_perm", None) is None:
            from .diagnostics import permutest as _permutest

            # permutest()'s own arguments are forwarded; whatever is left goes
            # to the plotting routine.
            perm_keys = ("exact", "iter", "permci", "seed", "alternative",
                         "stat", "p2defn")
            perm_kw = {k: kwargs.pop(k) for k in perm_keys if k in kwargs}
            perm_btt = kwargs.pop("btt", None)
            x = _permutest(x, btt=perm_btt, **perm_kw)
            kwargs["btt"] = perm_btt
        return plotting.plot_permutest(x, **kwargs)
    raise ValueError(f"unknown 'which' value: {which!r}")
