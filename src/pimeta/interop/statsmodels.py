"""statsmodels-style result adapter.

`statsmodels <https://www.statsmodels.org/>`_ is not just a modelling library —
it is where a large part of the scientific-Python ecosystem looks for a *result
object*.  Table renderers, ``summary_col``, ``compare_lr_test``, report builders
and a lot of user code all assume the same small surface::

    res.params   res.bse   res.tvalues   res.pvalues
    res.conf_int()   res.cov_params()
    res.llf   res.aic   res.bic   res.nobs
    res.fittedvalues   res.resid

:func:`as_results` wraps any pimeta fit in an object that provides exactly that,
so those tools work without a special case.  No statsmodels import is required
to build or use the adapter — it is plain duck typing.
"""

from __future__ import annotations

import numpy as np

__all__ = ["MetaResults", "MetaModelInfo", "as_results"]


class MetaModelInfo:
    """Minimal ``.model`` companion, matching what statsmodels consumers read.

    Attributes:
        endog (ndarray): the effect sizes (``yi``).
        exog (ndarray): the design matrix, intercept column included.
        k_constant (int): 1 when the model has an intercept, else 0.
        exog_names (list): coefficient labels, in design-matrix order.
        endog_names (str): name of the response (``"yi"``).
    """

    def __init__(self, endog, exog, k_constant, exog_names=None, endog_names="yi"):
        self.endog = np.asarray(endog, dtype=float)
        self.exog = np.asarray(exog, dtype=float)
        self.k_constant = int(k_constant)
        self.endog_names = str(endog_names)
        p = self.exog.shape[1]
        if exog_names is not None and len(exog_names) == p:
            self.exog_names = [str(v) for v in exog_names]
        else:
            self.exog_names = (["intrcpt"] if self.k_constant
                               else []) + [f"V{i + 1}" for i in range(p - self.k_constant)]

    @property
    def nobs(self) -> int:
        """Number of observations."""
        return int(self.endog.size)

    @property
    def df_resid(self) -> float:
        """Residual degrees of freedom."""
        return float(self.nobs - self.exog.shape[1])

    @property
    def df_model(self) -> float:
        """Model degrees of freedom."""
        return float(self.exog.shape[1] - self.k_constant)

    def __repr__(self) -> str:
        return (f"MetaModelInfo(nobs={self.nobs}, "
                f"k={self.exog.shape[1]}, k_constant={self.k_constant})")


class MetaResults:
    """A statsmodels-shaped view of a pimeta fit.

    Built by :func:`as_results`; do not instantiate directly.

    Attributes:
        result: the underlying pimeta result object.
        params (ndarray): coefficient estimates.
        bse (ndarray): standard errors.
        tvalues (ndarray): test statistics (``t`` under Hartung–Knapp,
            ``z`` otherwise — statsmodels calls the slot ``tvalues`` regardless).
        pvalues (ndarray): two-sided p-values.
        llf (float): maximised log-likelihood, when the model defines one.
        aic (float): Akaike information criterion.
        bic (float): Bayesian information criterion.
        nobs (int): number of studies.
        model (MetaModelInfo): endog/exog companion object.
    """

    def __init__(self, result):
        from ..s3 import AIC, BIC, logLik

        self.result = result
        b = np.atleast_1d(np.asarray(result.b, dtype=float)).ravel()
        self.params = b
        self.bse = np.atleast_1d(np.asarray(result.se, dtype=float)).ravel()
        stat = getattr(result, "zval", None)
        self.tvalues = (np.atleast_1d(np.asarray(stat, dtype=float)).ravel()
                        if stat is not None else self.params / self.bse)
        pval = getattr(result, "pval", None)
        if pval is not None:
            self.pvalues = np.atleast_1d(np.asarray(pval, dtype=float)).ravel()
        else:
            from scipy import stats

            if str(getattr(result, "test", "z")) in ("knha", "adhoc", "t") and getattr(result, "ddf", None):
                self.pvalues = 2.0 * stats.t.sf(np.abs(self.tvalues), float(result.ddf))
            else:
                self.pvalues = 2.0 * stats.norm.sf(np.abs(self.tvalues))

        self._cov = np.asarray(getattr(result, "vb", np.eye(b.size)), dtype=float)
        self.nobs = int(getattr(result, "k", b.size))
        self.level = float(getattr(result, "level", 0.95) or 0.95)

        X = getattr(result, "X", None)
        k_constant = 1 if bool(getattr(result, "int_incl", False)) else 0
        try:
            from .base import coefficient_names

            names = coefficient_names(result, b.size)
        except Exception:                            # pragma: no cover - defensive
            names = None
        self.model = MetaModelInfo(getattr(result, "yi", []),
                                   np.asarray(X) if X is not None else np.ones((self.nobs, 1)),
                                   k_constant, exog_names=names)

        self.llf = self._quiet(logLik)
        self.aic = self._quiet(AIC)
        self.bic = self._quiet(BIC)
        rt2 = getattr(result, "R2", None)
        self.rsquared = float(rt2) if rt2 is not None else float("nan")

    # -- helpers ------------------------------------------------------------ #
    def _quiet(self, func) -> float:
        """Call a pimeta generic, returning ``nan`` when the model lacks it."""
        try:
            return float(func(self.result))
        except Exception:                       # pragma: no cover - model-dependent
            return float("nan")

    # -- statsmodels surface ------------------------------------------------ #
    @property
    def df_resid(self) -> float:
        """Residual degrees of freedom (``nobs - k``)."""
        return self.model.df_resid

    @property
    def df_model(self) -> float:
        """Model degrees of freedom (``k - 1`` with an intercept)."""
        return self.model.df_model

    @property
    def fittedvalues(self) -> np.ndarray:
        """Fitted values on the analysis scale."""
        return np.asarray(getattr(self.result, "fitted", []), dtype=float)

    @property
    def resid(self) -> np.ndarray:
        """Residuals on the analysis scale."""
        return np.asarray(getattr(self.result, "residuals", []), dtype=float)

    def cov_params(self) -> np.ndarray:
        """Coefficient covariance matrix."""
        return self._cov

    def conf_int(self, alpha: float = 0.05, cols=None) -> np.ndarray:
        """Confidence intervals for the coefficients.

        Args:
            alpha: tail probability, so ``alpha=0.05`` gives a 95% interval
                (statsmodels convention).
            cols: optional column selection.

        Returns:
            An ``(p, 2)`` array of ``[lower, upper]`` rows.
        """
        from scipy import stats

        1.0 - float(alpha)
        if str(getattr(self.result, "test", "z")) in ("knha", "adhoc", "t") and getattr(self.result, "ddf", None):
            crit = float(stats.t.isf(alpha / 2.0, float(self.result.ddf)))
        else:
            crit = float(stats.norm.isf(alpha / 2.0))
        lower = self.params - crit * self.bse
        upper = self.params + crit * self.bse
        out = np.column_stack([lower, upper])
        if cols is not None:
            out = out[:, cols] if isinstance(cols, (list, tuple, np.ndarray, slice)) else out[cols]
        return out

    def summary(self):
        """Return the pimeta text summary (string-like)."""
        text = getattr(self.result, "summary_text", None)
        if callable(text):
            return text()
        return str(self.result)

    def as_pandas(self):
        """Return the coefficient table as a pandas ``DataFrame``.

        Columns are ``coef``, ``std err``, the test statistic, the two-sided
        p-value and the interval — the layout statsmodels' own ``summary()``
        uses — indexed by coefficient name so it joins cleanly with other result
        tables.

        Returns:
            A ``pandas.DataFrame``, or a ``dict`` of columns when pandas is absent.
        """
        ci = self.conf_int()
        alpha = 1.0 - self.level
        lo_label = f"{alpha / 2:.3f}"
        hi_label = f"{1.0 - alpha / 2:.3f}"
        ddf = getattr(self.result, "ddf", None)
        stat_name = ("t" if (str(getattr(self.result, "test", "z"))
                             in ("knha", "adhoc", "t") and ddf) else "z")
        columns = {
            "coef": self.params,
            "std err": self.bse,
            stat_name: self.tvalues,
            f"P>|{stat_name}|": self.pvalues,
            f"[{lo_label}, {hi_label}]": [f"{row[0]:.3f}, {row[1]:.3f}" for row in ci],
        }
        try:
            import pandas as pd
        except ImportError:                           # pragma: no cover - optional
            return columns
        return pd.DataFrame(columns, index=self.model.exog_names)

    def __repr__(self) -> str:
        tau2 = getattr(self.result, "tau2", None)
        extra = f", tau2={float(tau2):.6g}" if isinstance(tau2, (int, float)) else ""
        return f"MetaResults(nobs={self.nobs}, k={self.params.size}{extra})"


def as_results(result) -> MetaResults:
    """Wrap a pimeta fit in a statsmodels-shaped result object.

    Args:
        result: a fitted result carrying ``b``/``se`` (``rma``, ``rma_mv``,
            ``rma_ls``, ``metagen``, ``metabin``, ...).

    Returns:
        A :class:`MetaResults` exposing ``params``, ``bse``, ``tvalues``,
        ``pvalues``, ``conf_int()``, ``cov_params()``, ``llf``, ``aic``,
        ``bic``, ``nobs``, ``fittedvalues`` and ``resid``.

    Raises:
        TypeError: when ``result`` has no coefficient vector.
    """
    if not hasattr(result, "b") or not hasattr(result, "se"):
        raise TypeError(
            f"{type(result).__name__} has no 'b'/'se' pair; "
            "as_results() needs a coefficient-based model")
    return MetaResults(result)
