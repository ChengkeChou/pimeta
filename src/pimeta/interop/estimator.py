"""scikit-learn compatible estimator — without importing scikit-learn.

``scikit-learn``'s ecosystem is glued together by a *duck-typed* contract, not by
inheritance: ``clone()`` only needs ``get_params``/``set_params``,
``GridSearchCV`` additionally needs ``fit``, and anything that scores needs
``predict``.  Implementing those four methods therefore makes a pimeta model
usable by

* ``sklearn.pipeline.Pipeline``
* ``GridSearchCV`` / ``RandomizedSearchCV`` / ``cross_val_score`` /
  ``cross_validate`` / ``learning_curve`` / ``validation_curve``
* ``sklearn.base.clone``
* ``joblib`` / ``dask`` / ``ray`` parallel backends (they clone and pickle)
* ``optuna`` / ``hyperopt`` / ``mlflow`` / ``shap`` integrations

with **zero added runtime dependencies**.  The class is also a perfectly
ordinary Python object, so it works without scikit-learn installed at all.

Why meta-regression maps onto this interface so naturally: in a meta-analysis
the inverse sampling variance *is* the weight, so
``sample_weight = 1 / vi`` and everything else is a weighted least-squares
regression with a random effect.
"""

from __future__ import annotations

import numpy as np

from .base import as_2d

__all__ = ["MetaRegressor", "MetaEstimator"]


class MetaRegressor:
    """Random-effects meta-regression as a scikit-learn estimator.

    The design is deliberately minimal: ``__init__`` only stores parameters
    (sklearn's ``clone`` compares arguments by identity), and all validation
    happens in :meth:`fit`.

    Args:
        method: τ² estimator; ``"REML"`` (default), ``"DL"``, ``"ML"``, ``"PM"``,
            ``"EB"``, ``"SJ"``, ``"HS"``, ``"HE"``, ``"GENQ"``, ``"GENQM"``.
        test: ``"z"`` (default) or ``"knha"`` for the Hartung–Knapp adjustment.
        level: confidence level for the coefficient intervals.
        measure: effect-size label carried through to the result object.

    Attributes:
        result_ (RmaResult): the full pimeta fit, for anything the estimator API
            does not expose (heterogeneity, prediction intervals, plots).
        coef_ (ndarray): estimated coefficients, intercept first.
        intercept_ (float): the intercept, or ``0.0`` when the model has none.
        tau2_ (float): estimated between-study variance.
        n_features_in_ (int): number of moderators.
        feature_names_in_ (ndarray): moderator names, when a labelled frame was
            passed to :meth:`fit`.

    Example:
        >>> import numpy as np
        >>> from pimeta.interop import MetaRegressor
        >>> yi = np.array([0.20, 0.40, 0.10, 0.50])
        >>> vi = np.array([0.04, 0.05, 0.03, 0.06])
        >>> dose = np.array([1.0, 2.0, 3.0, 4.0])
        >>> est = MetaRegressor().fit(dose, yi, sample_weight=1 / vi)
        >>> est.coef_.shape
        (2,)
    """

    #: Tells ``sklearn.base.is_regressor`` what kind of estimator this is.
    #: sklearn < 1.6 reads this attribute directly.
    _estimator_type = "regressor"

    def __sklearn_tags__(self):
        """sklearn >= 1.6 estimator tags (``get_tags`` / ``is_regressor``).

        sklearn deprecated ``_estimator_type`` in favour of this protocol, so
        both are provided.  The import is local: pimeta never requires sklearn,
        it only answers when sklearn asks.
        """
        from sklearn.base import BaseEstimator

        tags = BaseEstimator().__sklearn_tags__()
        tags.estimator_type = "regressor"
        tags.target_tags.required = True
        tags.target_tags.one_d_labels = True
        return tags

    def __init__(self, method: str = "REML", test: str = "z",
                 level: float = 0.95, measure: str = "GEN"):
        self.method = method
        self.test = test
        self.level = level
        self.measure = measure

    # -- sklearn parameter protocol ---------------------------------------- #
    def get_params(self, deep: bool = True) -> dict:
        """Return the constructor parameters (``sklearn.base.BaseEstimator`` API).

        Args:
            deep: accepted for API compatibility; pimeta has no nested estimators.

        Returns:
            ``{name: value}`` for every constructor argument.
        """
        return {"method": self.method, "test": self.test,
                "level": self.level, "measure": self.measure}

    def set_params(self, **params):
        """Set constructor parameters in place (``BaseEstimator`` API).

        Args:
            **params: parameter name / value pairs.

        Returns:
            ``self``, so calls can be chained.

        Raises:
            ValueError: for an unknown parameter name.
        """
        valid = self.get_params()
        for key in params:
            if key not in valid:
                raise ValueError(
                    f"invalid parameter {key!r} for {type(self).__name__}; "
                    f"valid parameters are {sorted(valid)}")
        for key, value in params.items():
            setattr(self, key, value)
        return self

    # -- fitting ------------------------------------------------------------ #
    def fit(self, X, y, sample_weight=None, vi=None, **fit_params):
        """Fit the meta-regression.

        Args:
            X: moderators — ``None`` for an intercept-only model, a 1-D vector, a
                2-D array, or a labelled pandas/polars frame.
            y: effect sizes (``yi``), on whatever scale the analysis uses.
            sample_weight: inverse sampling variances (``1 / vi``).  This is the
                sklearn convention (larger weight = more influence), and it is
                what :meth:`predict` and :meth:`score` expect too.
            vi: sampling variances.  Takes precedence over ``sample_weight``.
            **fit_params: ignored, so a ``Pipeline`` can pass extra keys through.

        Returns:
            ``self``.

        Raises:
            ValueError: on a length mismatch between ``X``, ``y`` and the weights.
        """
        from ..rma_uni import rma

        yv = np.asarray(y, dtype=float).ravel()
        Xa, names = as_2d(X, yv.size)
        if Xa.shape[0] != yv.size:
            raise ValueError(
                f"'X' has {Xa.shape[0]} rows but 'y' has {yv.size} values")

        if vi is None:
            if sample_weight is None:
                vi_arr = np.ones_like(yv)
            else:
                w = np.asarray(sample_weight, dtype=float).ravel()
                if w.size != yv.size:
                    raise ValueError(
                        f"'sample_weight' has {w.size} values but 'y' has {yv.size}")
                if np.any(w <= 0):
                    raise ValueError("'sample_weight' must be strictly positive")
                vi_arr = 1.0 / w
        else:
            vi_arr = np.asarray(vi, dtype=float).ravel()
            if vi_arr.size != yv.size:
                raise ValueError(f"'vi' has {vi_arr.size} values but 'y' has {yv.size}")

        self.n_features_in_ = int(Xa.shape[1])
        if names is not None and len(names) == self.n_features_in_:
            self.feature_names_in_ = np.asarray(names, dtype=object)
        else:
            self.__dict__.pop("feature_names_in_", None)

        mods = Xa if Xa.shape[1] else None
        self.result_ = rma(yv, vi_arr, mods=mods, method=self.method,
                           test=self.test, level=self.level,
                           measure=self.measure)
        self.coef_ = np.asarray(self.result_.b, dtype=float)
        self.intercept_ = float(self.coef_[0]) if self._has_intercept() else 0.0
        self.tau2_ = float(getattr(self.result_, "tau2", np.nan))
        return self

    # -- prediction --------------------------------------------------------- #
    def _has_intercept(self) -> bool:
        return self.coef_.size == self.n_features_in_ + 1

    def _design(self, X) -> np.ndarray:
        Xa, _ = as_2d(X, None)
        if Xa.shape[1] != self.n_features_in_:
            raise ValueError(
                f"'X' has {Xa.shape[1]} features, but {type(self).__name__} "
                f"was fitted with {self.n_features_in_}")
        if self._has_intercept():
            return np.column_stack([np.ones(Xa.shape[0]), Xa])
        return Xa

    def predict(self, X, interval: str | None = None, level: float | None = None,
                vi=None):
        """Predict the pooled effect for new moderator values.

        Args:
            X: moderators, same width as at fit time.
            interval: ``None`` for point predictions, ``"confidence"`` for the
                interval of the fitted mean, or ``"prediction"`` for a
                prediction interval of a future study (which also needs ``vi``).
            level: confidence level; defaults to the fitted model's ``level``.
            vi: sampling variances of the new studies, required when
                ``interval="prediction"``.

        Returns:
            An ``ndarray`` of predictions, or a ``(fit, lower, upper)`` tuple when
            ``interval`` is given.

        Raises:
            ValueError: for an unknown ``interval`` or a missing ``vi``.
        """
        design = self._design(X)
        fit = design @ self.coef_
        if interval is None:
            return fit
        if interval not in ("confidence", "prediction"):
            raise ValueError("'interval' must be None, 'confidence' or 'prediction'")

        lvl = float(level if level is not None else self.level)
        vb = np.asarray(self.result_.vb, dtype=float)
        var = np.einsum("ij,jk,ik->i", design, vb, design)
        if interval == "prediction":
            if vi is None:
                raise ValueError("'vi' is required when interval='prediction'")
            var = var + float(self.tau2_) + np.asarray(vi, dtype=float).ravel()

        from scipy import stats

        if str(self.test) in ("knha", "adhoc", "t") and getattr(self.result_, "ddf", None):
            crit = float(stats.t.isf((1.0 - lvl) / 2.0, float(self.result_.ddf)))
        else:
            crit = float(stats.norm.isf((1.0 - lvl) / 2.0))
        se = np.sqrt(np.maximum(var, 0.0))
        return fit, fit - crit * se, fit + crit * se

    def score(self, X, y, sample_weight=None) -> float:
        """Weighted coefficient of determination.

        ``1 - sum(w * res^2) / sum(w * (y - weighted_mean)^2)`` with
        ``w = 1 / vi``.  This is the natural analogue of ``R^2`` for a
        meta-regression, so it slots into sklearn's model-selection helpers.

        Args:
            X: moderators.
            y: observed effect sizes.
            sample_weight: inverse sampling variances; defaults to equal weights.

        Returns:
            The weighted :math:`R^2`, or ``nan`` when the denominator is zero.
        """
        yv = np.asarray(y, dtype=float).ravel()
        w = (np.ones_like(yv) if sample_weight is None
             else np.asarray(sample_weight, dtype=float).ravel())
        pred = np.asarray(self.predict(X), dtype=float).ravel()
        resid = yv - pred
        mean = float(np.sum(w * yv) / np.sum(w))
        ss_res = float(np.sum(w * resid ** 2))
        ss_tot = float(np.sum(w * (yv - mean) ** 2))
        if ss_tot <= 0:
            return float("nan")
        return 1.0 - ss_res / ss_tot

    # -- convenience -------------------------------------------------------- #
    def summary(self) -> str:
        """Return the pimeta text summary of the fitted model."""
        result = getattr(self, "result_", None)
        if result is None:
            raise AttributeError("estimator is not fitted yet; call fit() first")
        text = getattr(result, "summary_text", None)
        if callable(text):
            return str(text())
        return str(result)

    def __repr__(self) -> str:
        params = ", ".join(f"{k}={v!r}" for k, v in self.get_params().items())
        return f"{type(self).__name__}({params})"


#: Alias kept for callers who use "estimator" as the generic term.
MetaEstimator = MetaRegressor
