"""Joint inference on discrimination and calibration.

A prognostic-model meta-analysis pools a C-statistic and a calibration slope
from the same studies.  :func:`~pimeta.prediction.prediction_meta` pools them
one at a time, which answers two separate questions and cannot answer either of
the questions that matter:

- *is this model both discriminative and calibrated?* — a joint hypothesis
  about two parameters, for which two marginal intervals are not a test;
- *where would a new study's pair of metrics fall?* — an ellipse, for the same
  reason a joint prediction region replaces two prediction intervals.

Both need the correlation between the two metrics.  Within a study the
C-statistic and the calibration slope are estimated from the same patients and
are **not** independent, and that correlation is essentially never reported.
So it is an assumption here, named ``rho`` and required rather than defaulted
silently, and :meth:`PredictionJointResult.sensitivity` shows what it is worth.

The model is the bivariate one:

.. math::

    \\begin{pmatrix} \\text{logit}(A_i) \\\\ s_i \\end{pmatrix}
    \\sim N\\!\\left(\\mu, V_i + \\Sigma\\right)

with :math:`V_i` the study's own sampling covariance — Hanley–McNeil mapped to
the logit scale for the AUC, the reported standard error for the slope, and
:math:`\\rho` between them — and :math:`\\Sigma` estimated across studies.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field

import numpy as np
from scipy import stats

from .levels import level_to_alpha, level_to_percent
from .prediction import _expit, auc_logit
from .rma_mv import rma_mv

__all__ = ["PredictionJointResult", "prediction_joint"]


@dataclass
class PredictionJointResult:
    """A bivariate pool of a C-statistic and a calibration slope.

    Attributes:
        auc, slope: the pooled values on their own scales.
        auc_lb, auc_ub: the interval for the AUC, back-transformed.
        slope_lb, slope_ub: the interval for the calibration slope.
        beta: the pooled parameters on the analysis scale
            ``(logit AUC, calibration slope)``.
        vcov: their joint covariance.
        Sigma: the between-study covariance of the two metrics.
        rho: the within-study correlation that was assumed.
        k, studies: studies contributing.
        model: the underlying :class:`~pimeta.RmaMvResult`.
        method, level, struct: settings.
    """

    auc: float
    slope: float
    auc_lb: float
    auc_ub: float
    slope_lb: float
    slope_ub: float
    beta: np.ndarray = field(repr=False)
    vcov: np.ndarray = field(repr=False)
    Sigma: np.ndarray = field(repr=False)
    rho: float
    k: int
    studies: list
    model: object = field(repr=False, default=None)
    method: str = "REML"
    level: float = 95.0
    struct: str = "UN"

    @property
    def dim(self) -> int:
        return int(np.asarray(self.beta).size)

    @property
    def se(self) -> np.ndarray:
        """Standard errors on the analysis scale."""
        return np.sqrt(np.diag(np.asarray(self.vcov, dtype=float)))

    def _crit(self) -> float:
        return float(stats.norm.ppf(0.5 + self.level / 200.0))

    def _wald(self, weights, target=0.0) -> dict:
        w = np.asarray(weights, dtype=float).ravel()
        b = np.asarray(self.beta, dtype=float)
        V = np.asarray(self.vcov, dtype=float)
        diff = float(w @ b) - float(target)
        var = float(w @ V @ w)
        if var <= 0.0:
            return {"stat": float("nan"), "df": 1, "pval": float("nan")}
        stat = diff ** 2 / var
        crit = self._crit()
        return {
            "estimate": float(w @ b),
            "se": float(np.sqrt(var)),
            "ci_lb": float(w @ b - crit * np.sqrt(var)),
            "ci_ub": float(w @ b + crit * np.sqrt(var)),
            "stat": float(stat),
            "df": 1,
            "pval": float(stats.chi2.sf(stat, 1)),
        }

    def test_discrimination(self, null: float = 0.5) -> dict:
        """Wald test of ``AUC = null`` on the logit scale (default chance)."""
        logit = float(np.log(null / (1.0 - null)))
        out = self._wald([1.0, 0.0], target=logit)
        out["auc"] = float(_expit(out["estimate"]))
        out["auc_lb"] = float(_expit(out["ci_lb"]))
        out["auc_ub"] = float(_expit(out["ci_ub"]))
        return out

    def test_calibration(self, null: float = 1.0) -> dict:
        """Wald test of ``slope = null``; 1 is perfect calibration."""
        return self._wald([0.0, 1.0], target=float(null))

    def test_both(self, auc: float = 0.5, slope: float = 1.0) -> dict:
        """Joint Wald test that the pooled point equals ``(auc, slope)``.

        Two marginal intervals are not a test of this: a model can fail the
        joint hypothesis while each metric's interval excludes its own null, and
        the reverse.  The statistic is
        :math:`(\\hat\\beta-\\beta_0)'V^{-1}(\\hat\\beta-\\beta_0)` on 2 df.
        """
        target = np.array([np.log(auc / (1.0 - auc)), float(slope)])
        b = np.asarray(self.beta, dtype=float)
        V = np.asarray(self.vcov, dtype=float)
        diff = b - target
        stat = float(diff @ np.linalg.solve(V, diff))
        return {"stat": stat, "df": self.dim,
                "pval": float(stats.chi2.sf(stat, self.dim))}

    def confidence_region(self) -> dict:
        """The ellipsoid for the pooled pair, in the metrics' own scales.

        A confidence region, not a prediction region: it describes where the
        *average* model's pair lies, which is the question "is this model any
        good" rather than "what would a new study find".
        """
        b = np.asarray(self.beta, dtype=float)
        V = np.asarray(self.vcov, dtype=float)
        alpha = level_to_alpha(self.level)
        return {
            "center": b,
            "cov": V,
            "crit": float(stats.chi2.ppf(1.0 - alpha, self.dim)),
            "level": self.level,
        }

    def prediction_region(self):
        """Where a *new* study's pair would fall; adds ``Sigma``."""
        from .mv.region import predict_region

        return predict_region(self.model, level=self.level)

    def sensitivity(self, rhos=(0.0, 0.25, 0.5, 0.75)) -> list[dict]:
        """Refit along a grid of within-study correlations.

        The correlation cannot be estimated from published summaries, so the
        obligation it creates is to show what the assumption is worth.  This
        returns one row per value of ``rho`` with the pooled AUC, the pooled
        slope and the p-value of the joint test.
        """
        rows = []
        for rho in rhos:
            fit = prediction_joint(
                auc=self._auc_input, n_event=self._n_event,
                n_nonevent=self._n_nonevent, auc_se=self._auc_se,
                slope=self._slope_input, slope_se=self._slope_se,
                rho=float(rho), method=self.method, level=self.level,
                struct=self.struct)
            rows.append({
                "rho": float(rho),
                "auc": fit.auc,
                "auc_lb": fit.auc_lb,
                "auc_ub": fit.auc_ub,
                "slope": fit.slope,
                "slope_lb": fit.slope_lb,
                "slope_ub": fit.slope_ub,
                "p_joint": fit.test_both()["pval"],
                "p_discrimination": fit.test_discrimination()["pval"],
                "p_calibration": fit.test_calibration()["pval"],
            })
        return rows

    # the inputs are kept for `sensitivity`, which has to refit
    _auc_input: np.ndarray = field(repr=False, default_factory=lambda:
                                   np.empty(0))
    _n_event: np.ndarray | None = field(repr=False, default=None)
    _n_nonevent: np.ndarray | None = field(repr=False, default=None)
    _auc_se: np.ndarray | None = field(repr=False, default=None)
    _slope_input: np.ndarray = field(repr=False, default_factory=lambda:
                                     np.empty(0))
    _slope_se: np.ndarray = field(repr=False, default_factory=lambda:
                                  np.empty(0))

    def to_dict(self) -> dict:
        return {
            "auc": self.auc,
            "auc_lb": self.auc_lb,
            "auc_ub": self.auc_ub,
            "slope": self.slope,
            "slope_lb": self.slope_lb,
            "slope_ub": self.slope_ub,
            "beta": np.asarray(self.beta).tolist(),
            "vcov": np.asarray(self.vcov).tolist(),
            "Sigma": np.asarray(self.Sigma).tolist(),
            "rho": self.rho,
            "k": self.k,
            "studies": list(self.studies),
            "method": self.method,
            "level": self.level,
            "struct": self.struct,
        }

    def coef_table(self) -> list[dict]:
        return [
            {"metric": "auc", "estimate": self.auc,
             "ci_lb": self.auc_lb, "ci_ub": self.auc_ub},
            {"metric": "calibration_slope", "estimate": self.slope,
             "ci_lb": self.slope_lb, "ci_ub": self.slope_ub},
        ]

    def __str__(self) -> str:
        pct = level_to_percent(self.level)
        lines = [
            "joint pooling of discrimination and calibration",
            f"  within-study correlation assumed: rho = {self.rho:g}",
            "",
            f"  {'metric':<20}{'estimate':>10}{f'{pct:g}% interval':>22}",
            f"  {'AUC':<20}{self.auc:>10.4f}"
            f"{f'[{self.auc_lb:.4f}, {self.auc_ub:.4f}]':>22}",
            f"  {'calibration slope':<20}{self.slope:>10.4f}"
            f"{f'[{self.slope_lb:.4f}, {self.slope_ub:.4f}]':>22}",
            "",
            f"  AUC = 0.5:            p = {self.test_discrimination()['pval']:.4g}",
            f"  slope = 1:            p = {self.test_calibration()['pval']:.4g}",
            f"  both at once:         p = {self.test_both()['pval']:.4g}",
        ]
        return "\n".join(lines)


def prediction_joint(*, auc, slope, n_event=None, n_nonevent=None, auc_se=None,
                     slope_se=None, rho=0.0, method="REML", level=0.95,
                     struct="UN", studies=None) -> PredictionJointResult:
    """Pool a C-statistic and a calibration slope *jointly*.

    Args:
        auc: per-study C-statistics, or ``NaN`` where a study reports none.
        slope: per-study calibration slopes, or ``NaN``.
        n_event, n_nonevent: group sizes for the Hanley–McNeil variance of the
            AUC.  Needed unless ``auc_se`` is given.
        auc_se: standard errors of the AUC **on the AUC scale**, for studies
            that report them instead of the group sizes.
        slope_se: standard errors of the calibration slope.
        rho: within-study correlation between the logit AUC and the slope.
            Defaults to 0, which is an assumption and not a measurement — use
            :meth:`PredictionJointResult.sensitivity` to see what it costs.
        method: ``"REML"`` or ``"ML"``.
        level: confidence level.
        struct: between-study covariance structure for the two metrics.
        studies: study labels, for reporting.

    Returns:
        :class:`PredictionJointResult`.

    Raises:
        ValueError: mismatched lengths, a study with a metric but no variance,
            a ``rho`` outside ``[-1, 1]``, or fewer than two usable studies.

    Example:
        >>> fit = ma.prediction_joint(auc=a, slope=s, n_event=n1,
        ...                           n_nonevent=n0, slope_se=se_s,
        ...                           rho=0.4)          # doctest: +SKIP
        >>> fit.test_both()                            # doctest: +SKIP
    """
    auc = np.asarray(auc, dtype=float).ravel()
    slope = np.asarray(slope, dtype=float).ravel()
    n = auc.size
    if slope.size != n:
        raise ValueError("'auc' and 'slope' must be the same length")
    if not -1.0 <= float(rho) <= 1.0:
        raise ValueError("'rho' must lie in [-1, 1]")
    if studies is None:
        studies = [f"study {i + 1}" for i in range(n)]
    studies = list(studies)
    if len(studies) != n:
        raise ValueError("'studies' needs one label per study")
    slope_se = (np.full(n, np.nan) if slope_se is None
                else np.asarray(slope_se, dtype=float).ravel())
    if slope_se.size != n:
        raise ValueError("'slope_se' needs one value per study")

    # the AUC variance: Hanley-McNeil on the logit scale, or a reported se
    if auc_se is not None:
        se = np.asarray(auc_se, dtype=float).ravel()
        if se.size != n:
            raise ValueError("'auc_se' needs one value per study")
        ok = np.isfinite(auc) & np.isfinite(se) & (se > 0.0)
        logit = np.full(n, np.nan)
        logit[ok] = np.log(auc[ok] / (1.0 - auc[ok]))
        # delta method from the AUC scale to the logit scale
        var_a = (se / (auc * (1.0 - auc))) ** 2
        var_a[~ok] = np.nan
    else:
        if n_event is None or n_nonevent is None:
            raise ValueError(
                "give either 'n_event' and 'n_nonevent' (for the "
                "Hanley-McNeil variance) or 'auc_se'")
        n1 = np.asarray(n_event, dtype=float).ravel()
        n0 = np.asarray(n_nonevent, dtype=float).ravel()
        if n1.size != n or n0.size != n:
            raise ValueError("'n_event' and 'n_nonevent' need one value each")
        ok = np.isfinite(auc) & np.isfinite(n1) & np.isfinite(n0)
        logit = np.full(n, np.nan)
        var_a = np.full(n, np.nan)
        if np.any(ok):
            y, v = auc_logit(auc[ok], n_event=n1[ok], n_nonevent=n0[ok])
            logit[ok] = y
            var_a[ok] = v

    keep_a = np.isfinite(logit) & np.isfinite(var_a) & (var_a > 0.0)
    keep_s = np.isfinite(slope) & np.isfinite(slope_se) & (slope_se > 0.0)
    if np.sum(keep_a | keep_s) < 2:
        raise ValueError("at least two studies are needed to pool")
    if int(np.sum(keep_a & keep_s)) < 2:
        warnings.warn(
            "fewer than two studies report both metrics, so the "
            "between-metric covariance cannot be separated from the "
            "within-study one; it is fixed at rho * sd_a * sd_s",
            RuntimeWarning, stacklevel=2)

    used = [i for i in range(n) if keep_a[i] or keep_s[i]]
    rows: list[float] = []
    labels: list[str] = []
    study_rows: list[str] = []
    blocks: list[np.ndarray] = []
    for i in used:
        if keep_a[i] and keep_s[i]:
            cov = rho * np.sqrt(var_a[i]) * slope_se[i]
            blocks.append(np.array([[var_a[i], cov], [cov, slope_se[i] ** 2]]))
            rows.extend([logit[i], slope[i]])
            labels.extend(["auc", "calibration_slope"])
            study_rows.extend([studies[i], studies[i]])
        elif keep_a[i]:
            blocks.append(np.array([[var_a[i]]]))
            rows.append(logit[i])
            labels.append("auc")
            study_rows.append(studies[i])
        else:
            blocks.append(np.array([[slope_se[i] ** 2]]))
            rows.append(slope[i])
            labels.append("calibration_slope")
            study_rows.append(studies[i])

    total = sum(b.shape[0] for b in blocks)
    V = np.zeros((total, total))
    start = 0
    for b in blocks:
        n_i = b.shape[0]
        V[start:start + n_i, start:start + n_i] = b
        start += n_i

    names = ["auc", "calibration_slope"]
    present = [nm for nm in names if nm in labels]
    if len(present) < 2:
        raise ValueError(
            "only one of the two metrics is reported often enough to pool; "
            "use prediction_meta for a single metric")
    X = np.zeros((total, len(present)))
    for j, nm in enumerate(present):
        for k, lab in enumerate(labels):
            if lab == nm:
                X[k, j] = 1.0
    data = {"metric": labels, "study": study_rows}
    fit = rma_mv(np.array(rows, dtype=float), V=V, mods=X, data=data,
                 random="~ metric | study", struct=struct,
                 intercept=False, method=method, level=level)
    beta = np.asarray(fit.beta, dtype=float)
    vcov = np.asarray(fit.vb, dtype=float)
    G, gnames = fit.g_matrix()
    Sigma = np.asarray(G, dtype=float) if G is not None else np.zeros(
        (len(present), len(present)))
    if gnames and list(gnames) != present:
        order = [list(gnames).index(nm) for nm in present]
        Sigma = Sigma[np.ix_(order, order)]

    crit = float(stats.norm.ppf(0.5 + level_to_percent(level) / 200.0))
    se = np.sqrt(np.diag(vcov))
    pos_a = present.index("auc")
    pos_s = present.index("calibration_slope")

    result = PredictionJointResult(
        auc=float(_expit(beta[pos_a])),
        auc_lb=float(_expit(beta[pos_a] - crit * se[pos_a])),
        auc_ub=float(_expit(beta[pos_a] + crit * se[pos_a])),
        slope=float(beta[pos_s]),
        slope_lb=float(beta[pos_s] - crit * se[pos_s]),
        slope_ub=float(beta[pos_s] + crit * se[pos_s]),
        beta=beta, vcov=vcov, Sigma=Sigma, rho=float(rho),
        k=len(used), studies=[studies[i] for i in used], model=fit,
        method=method, level=level_to_percent(level), struct=struct,
        _auc_input=auc, _n_event=None if n_event is None else np.asarray(
            n_event, dtype=float).ravel(),
        _n_nonevent=None if n_nonevent is None else np.asarray(
            n_nonevent, dtype=float).ravel(),
        _auc_se=None if auc_se is None else np.asarray(
            auc_se, dtype=float).ravel(),
        _slope_input=slope, _slope_se=slope_se,
    )
    return result
