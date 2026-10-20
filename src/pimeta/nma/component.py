"""Component network meta-analysis -- ``netmeta::netcomb`` / ``discomb``.

Additive component NMA (Rücker et al. 2020) models the effect of a
combination treatment as the sum of its components, ``delta = B theta`` with
``theta = C beta``.  ``netcomb`` starts from a fitted :class:`NetmetaResult`
and therefore needs a connected network; ``discomb`` takes raw per-comparison
data and also handles disconnected ones.

``NetcombResult``, ``netcomb`` and ``discomb`` are re-exported from
:mod:`pimeta.netmeta`.  The two names that still live there
(``NetmetaResult`` and the ``netmeta`` driver) are imported inside the
functions that need them, because ``pimeta.netmeta`` imports this module.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from scipy import stats

from .core import (
    _ZERO,
    _adjusted_variances,
    _connected,
    _create_b,
    _create_b_full,
    _isquared,
)


def _compsplit(x, sep_comps):
    """Split a treatment name into its components (netmeta::compsplit)."""
    import re
    parts = re.split(re.escape(sep_comps), x)
    return [p.strip() for p in parts if p.strip() != ""]


def _create_c_matrix(trts, inactive=None, sep_comps="+"):
    """C matrix mapping treatments -> components (netmeta::createC.netmeta).

    Returns ``(C, components, row_names, inactive)``. ``C`` has one row per
    retained treatment (in ``trts`` order) and one column per component; the
    entry is 1.0 when the treatment contains that component, else 0.0.
    """
    trts_all = list(trts)
    all_comps = []
    for t in trts:
        all_comps.extend(_compsplit(t, sep_comps))
    components = sorted(set(all_comps))

    avail_inactive = inactive is not None
    if avail_inactive:
        # netmeta::setchar (exact, then unique partial match).
        if inactive not in components:
            partial = [c for c in components if c.startswith(inactive)]
            if len(partial) == 1:
                inactive = partial[0]
            elif len(partial) > 1:
                raise ValueError(
                    f"inactive={inactive!r} matches multiple components: "
                    f"{partial}")
            else:
                raise ValueError(
                    f"inactive={inactive!r} is not a treatment component")
        trts_kept = [t for t in trts if t != inactive]
        if not trts_kept:
            raise ValueError(
                "All treatments equal the inactive treatment component")
        components = [c for c in components if c != inactive]
    else:
        trts_kept = list(trts)

    c = len(components)
    c_mat = np.zeros((len(trts_kept), c), dtype=float)
    for i, t in enumerate(trts_kept):
        parts = _compsplit(t, sep_comps)
        for j, comp in enumerate(components):
            if comp in parts:
                c_mat[i, j] = 1.0

    if avail_inactive and inactive in trts_all:
        c_mat = np.vstack([c_mat, np.zeros((1, c), dtype=float)])
        row_names = trts_kept + [inactive]
    else:
        row_names = trts_kept
    return c_mat, components, row_names, inactive


def _ci(te, se, level):
    """Confidence intervals / p-values (meta::ci, df = NULL branch)."""
    te = np.asarray(te, dtype=float)
    se = np.asarray(se, dtype=float)
    z = stats.norm.ppf(1 - (1 - level) / 2)
    lower = te - z * se
    upper = te + z * se
    with np.errstate(divide="ignore", invalid="ignore"):
        statistic = te / se
    pval = 2.0 * stats.norm.sf(np.abs(statistic))
    return lower, upper, statistic, pval


def _nma_additive(te, w_diag, studlab, level, x_mat, c_mat, b_mat,
                  df_q_additive, n):
    """Weighted-least-squares additive CNMA (netmeta::nma_additive).

    The model is ``delta = B theta = B C beta = X beta`` with component
    parameters ``beta`` estimated by WLS (weights ``w_diag`` on the study
    comparisons). ``x_mat = B @ C``, ``c_mat = C``, ``b_mat = B``.
    """
    te = np.asarray(te, dtype=float)
    w_diag = np.asarray(w_diag, dtype=float)
    m = len(te)
    w = np.diag(w_diag)

    l = x_mat.T @ w @ x_mat
    lplus = np.linalg.pinv(l)
    lplus[np.abs(lplus) < _ZERO] = 0.0

    h = x_mat @ lplus @ x_mat.T @ w
    h[np.abs(h) < _ZERO] = 0.0

    beta = lplus @ x_mat.T @ w @ te
    se_beta = np.sqrt(np.diag(lplus))

    theta = c_mat @ beta
    se_theta = np.sqrt(np.diag(c_mat @ lplus @ c_mat.T))

    delta = x_mat @ beta
    se_delta = np.sqrt(np.diag(x_mat @ lplus @ x_mat.T))

    # All pairwise contrasts (netmeta::nma_additive's B.full / X.all).
    b_full = _create_b_full(n)
    x_all = b_full @ c_mat
    cov = x_all @ lplus @ x_all.T
    cov[np.abs(cov) < _ZERO] = 0.0
    delta_all = x_all @ beta
    se_delta_all = np.sqrt(np.diag(x_all @ lplus @ x_all.T))

    delta_all_mat = np.zeros((n, n), dtype=float)
    se_delta_all_mat = np.zeros((n, n), dtype=float)
    k = 0
    for i in range(n - 1):
        for j in range(i + 1, n):
            delta_all_mat[i, j] = delta_all[k]
            delta_all_mat[j, i] = -delta_all[k]
            se_delta_all_mat[i, j] = se_delta_all_mat[j, i] = se_delta_all[k]
            k += 1

    cmp_lower, cmp_upper, cmp_stat, cmp_p = _ci(delta, se_delta, level)
    all_lower, all_upper, all_stat, all_p = _ci(delta_all_mat, se_delta_all_mat,
                                                level)
    comps_lower, comps_upper, comps_stat, comps_p = _ci(beta, se_beta, level)
    comb_lower, comb_upper, comb_stat, comb_p = _ci(theta, se_theta, level)

    resid = delta - te
    q_additive = float(resid @ w @ resid)
    if math.isnan(df_q_additive) or df_q_additive == 0:
        pval_q_additive = float("nan")
    else:
        pval_q_additive = float(stats.chi2.sf(q_additive, df_q_additive))

    i_m = np.eye(m)
    e = np.array(
        [[1.0 if studlab[i] == studlab[j] else 0.0 for j in range(m)]
         for i in range(m)],
        dtype=float,
    )
    if df_q_additive == 0:
        tau2 = tau = i2 = lower_i2 = upper_i2 = float("nan")
    else:
        denom = float(np.trace((i_m - h) @ ((b_mat @ b_mat.T) * (e / 2.0)) @ w))
        tau2 = max(0.0, (q_additive - df_q_additive) / denom) \
            if denom > 0 else float("nan")
        tau = math.sqrt(tau2) if not math.isnan(tau2) else float("nan")
        i2, lower_i2, upper_i2 = _isquared(q_additive, df_q_additive, level)

    return {
        "comparisons": {"TE": delta, "seTE": se_delta, "lower": cmp_lower,
                        "upper": cmp_upper, "statistic": cmp_stat, "p": cmp_p},
        "all_comparisons": {"TE": delta_all_mat, "seTE": se_delta_all_mat,
                            "lower": all_lower, "upper": all_upper,
                            "statistic": all_stat, "p": all_p},
        "components": {"TE": beta, "seTE": se_beta, "lower": comps_lower,
                       "upper": comps_upper, "statistic": comps_stat,
                       "p": comps_p},
        "combinations": {"TE": theta, "seTE": se_theta, "lower": comb_lower,
                         "upper": comb_upper, "statistic": comb_stat,
                         "p": comb_p},
        "Q_additive": q_additive,
        "pval_Q_additive": pval_q_additive,
        "tau": tau, "I2": i2, "lower_I2": lower_i2, "upper_I2": upper_i2,
        "L": l, "Lplus": lplus, "H": h, "Cov": cov,
    }


def _get_unident(x_mat):
    """Identify non-identifiable components (netmeta::get_unident).

    Components whose columns of ``X`` lie in the null space of ``X`` cannot be
    uniquely estimated; the null space is read off the eigenvectors of the
    projection ``pinv(X) @ X`` with (near-)zero eigenvalue.  Returns the
    sorted column indices of the non-identifiable components.
    """
    xplus = np.linalg.pinv(x_mat)
    proj = xplus @ x_mat
    eigvals, eigvecs = np.linalg.eigh(proj)
    tol = 100.0 * np.finfo(float).eps
    unident = set()
    for i in range(len(eigvals)):
        if abs(eigvals[i]) < tol:
            v = eigvecs[:, i]
            for j in range(len(v)):
                if abs(v[j]) > tol:
                    unident.add(j)
    return sorted(unident)


@dataclass
class NetcombResult:
    """Additive component network meta-analysis (netmeta::netcomb)."""

    trts: list[str]
    comps: list[str]
    k: int
    m: int
    n: int
    d: int
    c: int
    sm: str
    level: float
    method_tau: str
    reference_group: str
    inactive: str | None
    sep_comps: str

    # Common-effects all-pairs contrast matrices (n x n, diagonal 0).
    TE_common: np.ndarray = field(default_factory=lambda: np.empty((0, 0)))
    seTE_common: np.ndarray = field(default_factory=lambda: np.empty((0, 0)))
    lower_common: np.ndarray = field(default_factory=lambda: np.empty((0, 0)))
    upper_common: np.ndarray = field(default_factory=lambda: np.empty((0, 0)))
    statistic_common: np.ndarray = field(default_factory=lambda: np.empty((0, 0)))
    pval_common: np.ndarray = field(default_factory=lambda: np.empty((0, 0)))

    TE_random: np.ndarray = field(default_factory=lambda: np.empty((0, 0)))
    seTE_random: np.ndarray = field(default_factory=lambda: np.empty((0, 0)))
    lower_random: np.ndarray = field(default_factory=lambda: np.empty((0, 0)))
    upper_random: np.ndarray = field(default_factory=lambda: np.empty((0, 0)))
    statistic_random: np.ndarray = field(default_factory=lambda: np.empty((0, 0)))
    pval_random: np.ndarray = field(default_factory=lambda: np.empty((0, 0)))

    # Component (incremental) effects (length c).
    Comp_common: np.ndarray = field(default_factory=lambda: np.empty(0))
    seComp_common: np.ndarray = field(default_factory=lambda: np.empty(0))
    lower_Comp_common: np.ndarray = field(default_factory=lambda: np.empty(0))
    upper_Comp_common: np.ndarray = field(default_factory=lambda: np.empty(0))
    statistic_Comp_common: np.ndarray = field(default_factory=lambda: np.empty(0))
    pval_Comp_common: np.ndarray = field(default_factory=lambda: np.empty(0))
    Comp_random: np.ndarray = field(default_factory=lambda: np.empty(0))
    seComp_random: np.ndarray = field(default_factory=lambda: np.empty(0))
    lower_Comp_random: np.ndarray = field(default_factory=lambda: np.empty(0))
    upper_Comp_random: np.ndarray = field(default_factory=lambda: np.empty(0))
    statistic_Comp_random: np.ndarray = field(default_factory=lambda: np.empty(0))
    pval_Comp_random: np.ndarray = field(default_factory=lambda: np.empty(0))

    # Combination (per-treatment) effects (length n).
    Comb_common: np.ndarray = field(default_factory=lambda: np.empty(0))
    seComb_common: np.ndarray = field(default_factory=lambda: np.empty(0))
    lower_Comb_common: np.ndarray = field(default_factory=lambda: np.empty(0))
    upper_Comb_common: np.ndarray = field(default_factory=lambda: np.empty(0))
    statistic_Comb_common: np.ndarray = field(default_factory=lambda: np.empty(0))
    pval_Comb_common: np.ndarray = field(default_factory=lambda: np.empty(0))
    Comb_random: np.ndarray = field(default_factory=lambda: np.empty(0))
    seComb_random: np.ndarray = field(default_factory=lambda: np.empty(0))
    lower_Comb_random: np.ndarray = field(default_factory=lambda: np.empty(0))
    upper_Comb_random: np.ndarray = field(default_factory=lambda: np.empty(0))
    statistic_Comb_random: np.ndarray = field(default_factory=lambda: np.empty(0))
    pval_Comb_random: np.ndarray = field(default_factory=lambda: np.empty(0))

    # Standard-NMA treatment estimates (copied from the netmeta input).
    TE_nma_common: np.ndarray = field(default_factory=lambda: np.empty((0, 0)))
    seTE_nma_common: np.ndarray = field(default_factory=lambda: np.empty((0, 0)))
    lower_nma_common: np.ndarray = field(default_factory=lambda: np.empty((0, 0)))
    upper_nma_common: np.ndarray = field(default_factory=lambda: np.empty((0, 0)))
    TE_nma_random: np.ndarray = field(default_factory=lambda: np.empty((0, 0)))
    seTE_nma_random: np.ndarray = field(default_factory=lambda: np.empty((0, 0)))
    lower_nma_random: np.ndarray = field(default_factory=lambda: np.empty((0, 0)))
    upper_nma_random: np.ndarray = field(default_factory=lambda: np.empty((0, 0)))

    # Study-level fitted (additive) effects (length m).
    TE_cnma_common: np.ndarray = field(default_factory=lambda: np.empty(0))
    seTE_cnma_common: np.ndarray = field(default_factory=lambda: np.empty(0))
    lower_cnma_common: np.ndarray = field(default_factory=lambda: np.empty(0))
    upper_cnma_common: np.ndarray = field(default_factory=lambda: np.empty(0))
    TE_cnma_random: np.ndarray = field(default_factory=lambda: np.empty(0))
    seTE_cnma_random: np.ndarray = field(default_factory=lambda: np.empty(0))
    lower_cnma_random: np.ndarray = field(default_factory=lambda: np.empty(0))
    upper_cnma_random: np.ndarray = field(default_factory=lambda: np.empty(0))

    # Heterogeneity / inconsistency.
    Q_additive: float = float("nan")
    df_Q_additive: float = float("nan")
    pval_Q_additive: float = float("nan")
    Q_standard: float = float("nan")
    df_Q_standard: float = float("nan")
    pval_Q_standard: float = float("nan")
    Q_diff: float = float("nan")
    df_Q_diff: float = float("nan")
    pval_Q_diff: float = float("nan")
    tau: float = float("nan")
    tau2: float = float("nan")
    I2: float = float("nan")
    lower_I2: float = float("nan")
    upper_I2: float = float("nan")

    # Matrices.
    A_matrix: np.ndarray = field(default_factory=lambda: np.empty((0, 0)))
    B_matrix: np.ndarray = field(default_factory=lambda: np.empty((0, 0)))
    X_matrix: np.ndarray = field(default_factory=lambda: np.empty((0, 0)))
    C_matrix: np.ndarray = field(default_factory=lambda: np.empty((0, 0)))
    L_matrix_common: np.ndarray = field(default_factory=lambda: np.empty((0, 0)))
    Lplus_matrix_common: np.ndarray = field(default_factory=lambda: np.empty((0, 0)))
    H_matrix_common: np.ndarray = field(default_factory=lambda: np.empty((0, 0)))
    Cov_common: np.ndarray = field(default_factory=lambda: np.empty((0, 0)))
    L_matrix_random: np.ndarray = field(default_factory=lambda: np.empty((0, 0)))
    Lplus_matrix_random: np.ndarray = field(default_factory=lambda: np.empty((0, 0)))
    H_matrix_random: np.ndarray = field(default_factory=lambda: np.empty((0, 0)))
    Cov_random: np.ndarray = field(default_factory=lambda: np.empty((0, 0)))

    def _m(self, x):
        return x.tolist() if isinstance(x, np.ndarray) else x

    def to_dict(self) -> dict:
        return {
            "trts": self.trts, "comps": self.comps,
            "k": self.k, "m": self.m, "n": self.n, "d": self.d, "c": self.c,
            "sm": self.sm, "level": self.level, "method_tau": self.method_tau,
            "reference_group": self.reference_group,
            "inactive": self.inactive, "sep_comps": self.sep_comps,
            "TE_common": self._m(self.TE_common),
            "seTE_common": self._m(self.seTE_common),
            "lower_common": self._m(self.lower_common),
            "upper_common": self._m(self.upper_common),
            "statistic_common": self._m(self.statistic_common),
            "pval_common": self._m(self.pval_common),
            "TE_random": self._m(self.TE_random),
            "seTE_random": self._m(self.seTE_random),
            "lower_random": self._m(self.lower_random),
            "upper_random": self._m(self.upper_random),
            "statistic_random": self._m(self.statistic_random),
            "pval_random": self._m(self.pval_random),
            "Comp_common": self._m(self.Comp_common),
            "seComp_common": self._m(self.seComp_common),
            "lower_Comp_common": self._m(self.lower_Comp_common),
            "upper_Comp_common": self._m(self.upper_Comp_common),
            "statistic_Comp_common": self._m(self.statistic_Comp_common),
            "pval_Comp_common": self._m(self.pval_Comp_common),
            "Comp_random": self._m(self.Comp_random),
            "seComp_random": self._m(self.seComp_random),
            "lower_Comp_random": self._m(self.lower_Comp_random),
            "upper_Comp_random": self._m(self.upper_Comp_random),
            "statistic_Comp_random": self._m(self.statistic_Comp_random),
            "pval_Comp_random": self._m(self.pval_Comp_random),
            "Comb_common": self._m(self.Comb_common),
            "seComb_common": self._m(self.seComb_common),
            "lower_Comb_common": self._m(self.lower_Comb_common),
            "upper_Comb_common": self._m(self.upper_Comb_common),
            "statistic_Comb_common": self._m(self.statistic_Comb_common),
            "pval_Comb_common": self._m(self.pval_Comb_common),
            "Comb_random": self._m(self.Comb_random),
            "seComb_random": self._m(self.seComb_random),
            "lower_Comb_random": self._m(self.lower_Comb_random),
            "upper_Comb_random": self._m(self.upper_Comb_random),
            "statistic_Comb_random": self._m(self.statistic_Comb_random),
            "pval_Comb_random": self._m(self.pval_Comb_random),
            "TE_nma_common": self._m(self.TE_nma_common),
            "seTE_nma_common": self._m(self.seTE_nma_common),
            "lower_nma_common": self._m(self.lower_nma_common),
            "upper_nma_common": self._m(self.upper_nma_common),
            "TE_nma_random": self._m(self.TE_nma_random),
            "seTE_nma_random": self._m(self.seTE_nma_random),
            "lower_nma_random": self._m(self.lower_nma_random),
            "upper_nma_random": self._m(self.upper_nma_random),
            "TE_cnma_common": self._m(self.TE_cnma_common),
            "seTE_cnma_common": self._m(self.seTE_cnma_common),
            "lower_cnma_common": self._m(self.lower_cnma_common),
            "upper_cnma_common": self._m(self.upper_cnma_common),
            "TE_cnma_random": self._m(self.TE_cnma_random),
            "seTE_cnma_random": self._m(self.seTE_cnma_random),
            "lower_cnma_random": self._m(self.lower_cnma_random),
            "upper_cnma_random": self._m(self.upper_cnma_random),
            "Q_additive": self.Q_additive,
            "df_Q_additive": self.df_Q_additive,
            "pval_Q_additive": self.pval_Q_additive,
            "Q_standard": self.Q_standard,
            "df_Q_standard": self.df_Q_standard,
            "pval_Q_standard": self.pval_Q_standard,
            "Q_diff": self.Q_diff,
            "df_Q_diff": self.df_Q_diff,
            "pval_Q_diff": self.pval_Q_diff,
            "tau": self.tau, "tau2": self.tau2,
            "I2": self.I2, "lower_I2": self.lower_I2, "upper_I2": self.upper_I2,
            "A_matrix": self._m(self.A_matrix),
            "B_matrix": self._m(self.B_matrix),
            "X_matrix": self._m(self.X_matrix),
            "C_matrix": self._m(self.C_matrix),
            "L_matrix_common": self._m(self.L_matrix_common),
            "Lplus_matrix_common": self._m(self.Lplus_matrix_common),
            "H_matrix_common": self._m(self.H_matrix_common),
            "Cov_common": self._m(self.Cov_common),
            "L_matrix_random": self._m(self.L_matrix_random),
            "Lplus_matrix_random": self._m(self.Lplus_matrix_random),
            "H_matrix_random": self._m(self.H_matrix_random),
            "Cov_random": self._m(self.Cov_random),
        }


def netcomb(
    x,
    inactive: str | None = None,
    sep_comps: str = "+",
    common: bool | None = None,
    random: bool | None = None,
    tau_preset: float | None = None,
) -> NetcombResult:
    """Additive component network meta-analysis (netmeta::netcomb).

    ``x`` is a :class:`NetmetaResult` from :func:`netmeta` whose treatment
    labels encode combination treatments (e.g. ``"A+B"``). The additive CNMA
    model assumes a combination's effect is the sum of its components'
    effects: ``delta = B theta`` with ``theta = C beta`` (Rücker et al. 2020),
    estimated by weighted least squares.
    """
    from ..netmeta import NetmetaResult  # local import: cycle

    if not isinstance(x, NetmetaResult):
        raise TypeError("netcomb expects a NetmetaResult")

    trts = list(x.trts)
    n = x.n
    m = x.m
    level = x.level

    if common is None:
        common = True
    if random is None:
        random = True

    b_matrix = np.asarray(x.B_matrix, dtype=float)  # m x n, normalized order

    # C matrix (component composition) and X = B C.
    c_mat_full, components, row_names, inactive = _create_c_matrix(
        trts, inactive, sep_comps)
    # Reorder rows to treatment order (netcomb indexes C by x$trts).
    order_idx = [row_names.index(t) for t in trts]
    c_mat = c_mat_full[order_idx, :]
    c = c_mat.shape[1]

    x_mat = b_matrix @ c_mat  # m x c

    rank_x = np.linalg.matrix_rank(x_mat)
    df_q_additive = x.df_Q + n - 1 - rank_x
    df_q_diff = n - 1 - rank_x

    # Normalize comparisons to treat1 < treat2 (matching B_matrix rows).
    te = np.array(x.TE_raw, dtype=float)
    se = np.array(x.seTE_raw, dtype=float)
    t1 = list(x.treat1_raw)
    t2 = list(x.treat2_raw)
    stud = list(x.studlab_raw)
    idx = {t: i for i, t in enumerate(trts)}
    for i in range(m):
        if t1[i] > t2[i]:
            t1[i], t2[i] = t2[i], t1[i]
            te[i] = -te[i]
    t1_pos = [idx[t] for t in t1]
    t2_pos = [idx[t] for t in t2]

    # Common-effects: W = diag(1 / adjusted se^2) with tau = 0.
    var_c = _adjusted_variances(se, 0.0, t1_pos, t2_pos, stud)
    w_c = 1.0 / var_c
    res_c = _nma_additive(te, w_c, stud, level, x_mat, c_mat, b_matrix,
                          df_q_additive, n)

    q_additive = res_c["Q_additive"]
    if tau_preset is not None:
        tau = float(tau_preset)
    else:
        tau = res_c["tau"]
    tau2_eff = 0.0 if (tau is None or math.isnan(tau)) else tau * tau

    # Random-effects: W = diag(1 / (se^2 + tau^2) adjusted).
    var_r = _adjusted_variances(se, tau2_eff, t1_pos, t2_pos, stud)
    w_r = 1.0 / var_r
    res_r = _nma_additive(te, w_r, stud, level, x_mat, c_mat, b_matrix,
                          df_q_additive, n)

    q_diff = q_additive - x.Q
    if not math.isnan(q_diff) and abs(q_diff) < np.finfo(float).eps ** 0.75:
        q_diff = 0.0
    if math.isnan(df_q_diff) or df_q_diff == 0:
        pval_q_diff = float("nan")
    else:
        pval_q_diff = float(stats.chi2.sf(q_diff, df_q_diff))

    tau2 = tau * tau if not math.isnan(tau) else float("nan")

    return NetcombResult(
        trts=trts, comps=components, k=x.k, m=m, n=n, d=x.d if hasattr(x, "d") else 0,
        c=c, sm=x.sm, level=level, method_tau=x.method_tau,
        reference_group=x.reference_group, inactive=inactive, sep_comps=sep_comps,
        TE_common=res_c["all_comparisons"]["TE"],
        seTE_common=res_c["all_comparisons"]["seTE"],
        lower_common=res_c["all_comparisons"]["lower"],
        upper_common=res_c["all_comparisons"]["upper"],
        statistic_common=res_c["all_comparisons"]["statistic"],
        pval_common=res_c["all_comparisons"]["p"],
        TE_random=res_r["all_comparisons"]["TE"],
        seTE_random=res_r["all_comparisons"]["seTE"],
        lower_random=res_r["all_comparisons"]["lower"],
        upper_random=res_r["all_comparisons"]["upper"],
        statistic_random=res_r["all_comparisons"]["statistic"],
        pval_random=res_r["all_comparisons"]["p"],
        Comp_common=res_c["components"]["TE"],
        seComp_common=res_c["components"]["seTE"],
        lower_Comp_common=res_c["components"]["lower"],
        upper_Comp_common=res_c["components"]["upper"],
        statistic_Comp_common=res_c["components"]["statistic"],
        pval_Comp_common=res_c["components"]["p"],
        Comp_random=res_r["components"]["TE"],
        seComp_random=res_r["components"]["seTE"],
        lower_Comp_random=res_r["components"]["lower"],
        upper_Comp_random=res_r["components"]["upper"],
        statistic_Comp_random=res_r["components"]["statistic"],
        pval_Comp_random=res_r["components"]["p"],
        Comb_common=res_c["combinations"]["TE"],
        seComb_common=res_c["combinations"]["seTE"],
        lower_Comb_common=res_c["combinations"]["lower"],
        upper_Comb_common=res_c["combinations"]["upper"],
        statistic_Comb_common=res_c["combinations"]["statistic"],
        pval_Comb_common=res_c["combinations"]["p"],
        Comb_random=res_r["combinations"]["TE"],
        seComb_random=res_r["combinations"]["seTE"],
        lower_Comb_random=res_r["combinations"]["lower"],
        upper_Comb_random=res_r["combinations"]["upper"],
        statistic_Comb_random=res_r["combinations"]["statistic"],
        pval_Comb_random=res_r["combinations"]["p"],
        TE_nma_common=x.TE_common,
        seTE_nma_common=x.seTE_common,
        lower_nma_common=x.lower_common,
        upper_nma_common=x.upper_common,
        TE_nma_random=x.TE_random,
        seTE_nma_random=x.seTE_random,
        lower_nma_random=x.lower_random,
        upper_nma_random=x.upper_random,
        TE_cnma_common=res_c["comparisons"]["TE"],
        seTE_cnma_common=res_c["comparisons"]["seTE"],
        lower_cnma_common=res_c["comparisons"]["lower"],
        upper_cnma_common=res_c["comparisons"]["upper"],
        TE_cnma_random=res_r["comparisons"]["TE"],
        seTE_cnma_random=res_r["comparisons"]["seTE"],
        lower_cnma_random=res_r["comparisons"]["lower"],
        upper_cnma_random=res_r["comparisons"]["upper"],
        Q_additive=q_additive,
        df_Q_additive=df_q_additive,
        pval_Q_additive=res_c["pval_Q_additive"],
        Q_standard=x.Q,
        df_Q_standard=x.df_Q,
        pval_Q_standard=x.pval_Q,
        Q_diff=q_diff,
        df_Q_diff=df_q_diff,
        pval_Q_diff=pval_q_diff,
        tau=tau, tau2=tau2,
        I2=res_c["I2"], lower_I2=res_c["lower_I2"], upper_I2=res_c["upper_I2"],
        A_matrix=x.A_matrix,
        B_matrix=b_matrix,
        X_matrix=x_mat,
        C_matrix=c_mat,
        L_matrix_common=res_c["L"],
        Lplus_matrix_common=res_c["Lplus"],
        H_matrix_common=res_c["H"],
        Cov_common=res_c["Cov"],
        L_matrix_random=res_r["L"],
        Lplus_matrix_random=res_r["Lplus"],
        H_matrix_random=res_r["H"],
        Cov_random=res_r["Cov"],
    )


def discomb(
    TE,
    seTE,
    treat1,
    treat2,
    studlab=None,
    inactive: str | None = None,
    sep_comps: str = "+",
    sm: str = "",
    level: float = 0.95,
    common: bool = True,
    random: bool = True,
    reference_group: str = "",
    tau_preset: float | None = None,
) -> NetcombResult:
    """Additive CNMA for possibly disconnected networks (netmeta::discomb).

    Unlike :func:`netcomb` (which starts from a :class:`NetmetaResult` and
    therefore requires a connected network), ``discomb`` takes raw
    per-comparison data and works even when the network is disconnected.  The
    additive degrees of freedom are ``df.Q.additive = n.a - k - rank(X)`` with
    ``n.a`` the total number of study arms and ``k`` the number of studies.
    When the network is connected the standard heterogeneity Q is taken from an
    internal :func:`netmeta` fit; otherwise it is undefined (NaN).
    """
    TE = list(TE)
    seTE = list(seTE)
    treat1 = list(treat1)
    treat2 = list(treat2)
    m = len(TE)
    if not (len(seTE) == len(treat1) == len(treat2) == m):
        raise ValueError("TE, seTE, treat1, treat2 must have equal length")

    if studlab is None:
        studlab = [f"study{i + 1}" for i in range(m)]
    else:
        studlab = list(studlab)

    trts = sorted(set(treat1) | set(treat2))
    n = len(trts)
    idx = {t: i for i, t in enumerate(trts)}

    te = [float(v) for v in TE]
    se = [float(v) for v in seTE]
    t1 = list(treat1)
    t2 = list(treat2)
    stud = list(studlab)

    # Normalize to treat1 < treat2 (flip the sign of TE accordingly).
    for i in range(m):
        if t1[i] > t2[i]:
            t1[i], t2[i] = t2[i], t1[i]
            te[i] = -te[i]

    t1_pos = [idx[t] for t in t1]
    t2_pos = [idx[t] for t in t2]

    # Arm count per study and total number of arms.
    studies = sorted(set(stud))
    study_to_trts: dict[Any, set[Any]] = {s: set() for s in studies}
    for s, a, b in zip(stud, t1, t2):
        study_to_trts[s].add(a)
        study_to_trts[s].add(b)
    narms_per_study = {s: len(study_to_trts[s]) for s in studies}
    n_a = sum(narms_per_study.values())
    k = len(studies)

    # Incidence / design / C matrices.
    b_mat = _create_b(t1_pos, t2_pos, n)
    c_mat_full, components, row_names, inactive = _create_c_matrix(
        trts, inactive, sep_comps)
    order_idx = [row_names.index(t) for t in trts]
    c_mat = c_mat_full[order_idx, :]
    c = c_mat.shape[1]
    x_mat = b_mat @ c_mat
    a_mat = np.diag(np.diag(b_mat.T @ b_mat)) - b_mat.T @ b_mat

    rank_x = np.linalg.matrix_rank(x_mat)
    df_q_additive = n_a - k - rank_x

    # Non-identifiable components (only when X is rank deficient).
    unident_idx = _get_unident(x_mat) if rank_x < c else []

    # Standard NMA heterogeneity Q (defined only for connected networks).
    adj = np.zeros((n, n), dtype=bool)
    for a, b in zip(t1_pos, t2_pos):
        adj[a, b] = adj[b, a] = True
    connected = (n <= 1) or _connected(adj)
    if connected:
        from ..netmeta import netmeta  # local import: cycle

        net = netmeta(TE, seTE, treat1, treat2, studlab,
                      sm=sm, level=level, common=True, random=True)
        q_standard = float(net.Q)
        df_q_standard = float(net.df_Q)
        pval_q_standard = float(net.pval_Q)
        df_q_diff = n - 1 - rank_x
    else:
        q_standard = df_q_standard = pval_q_standard = df_q_diff = float("nan")

    se_arr = np.asarray(se, dtype=float)
    te_arr = np.asarray(te, dtype=float)

    # Common-effects additive model (W = diag(1 / adjusted se^2), tau = 0).
    var_c = _adjusted_variances(se_arr, 0.0, t1_pos, t2_pos, stud)
    w_c = 1.0 / var_c
    res_c = _nma_additive(te_arr, w_c, stud, level, x_mat, c_mat, b_mat,
                          df_q_additive, n)

    q_additive = res_c["Q_additive"]
    if tau_preset is not None:
        tau = float(tau_preset)
    else:
        tau = res_c["tau"]
    tau2_eff = 0.0 if (tau is None or math.isnan(tau)) else tau * tau

    # Random-effects additive model.
    var_r = _adjusted_variances(se_arr, tau2_eff, t1_pos, t2_pos, stud)
    w_r = 1.0 / var_r
    res_r = _nma_additive(te_arr, w_r, stud, level, x_mat, c_mat, b_mat,
                          df_q_additive, n)

    # Set non-identifiable components/combinations to NA (netmeta na.unident).
    if unident_idx:
        unident_names = {components[j] for j in unident_idx}
        unident_comb_idx = [
            i for i, t in enumerate(trts)
            if any(p in unident_names for p in _compsplit(t, sep_comps))
        ]
        _stats = ("TE", "seTE", "lower", "upper", "statistic", "p")
        for res in (res_c, res_r):
            for j in unident_idx:
                for stat in _stats:
                    res["components"][stat][j] = float("nan")
            for j in unident_comb_idx:
                for stat in _stats:
                    res["combinations"][stat][j] = float("nan")

    q_diff = q_additive - q_standard
    if not math.isnan(q_diff) and abs(q_diff) < np.finfo(float).eps ** 0.75:
        q_diff = 0.0
    if math.isnan(df_q_diff) or df_q_diff == 0:
        pval_q_diff = float("nan")
    else:
        pval_q_diff = float(stats.chi2.sf(q_diff, df_q_diff))

    tau2 = tau * tau if not math.isnan(tau) else float("nan")

    if reference_group == "":
        reference_group = trts[0]

    # Number of unique designs (netmeta::designs -> distinct treatment sets).
    design_keys = {tuple(sorted(study_to_trts[s])) for s in studies}
    d = len(design_keys)

    # discomb sets the standard-NMA estimates to NA (length m, like TE.cnma).
    nma_nan = np.full(m, np.nan)

    return NetcombResult(
        trts=trts, comps=components, k=k, m=m, n=n, d=d,
        c=c, sm=sm, level=level, method_tau="DL",
        reference_group=reference_group, inactive=inactive, sep_comps=sep_comps,
        TE_common=res_c["all_comparisons"]["TE"],
        seTE_common=res_c["all_comparisons"]["seTE"],
        lower_common=res_c["all_comparisons"]["lower"],
        upper_common=res_c["all_comparisons"]["upper"],
        statistic_common=res_c["all_comparisons"]["statistic"],
        pval_common=res_c["all_comparisons"]["p"],
        TE_random=res_r["all_comparisons"]["TE"],
        seTE_random=res_r["all_comparisons"]["seTE"],
        lower_random=res_r["all_comparisons"]["lower"],
        upper_random=res_r["all_comparisons"]["upper"],
        statistic_random=res_r["all_comparisons"]["statistic"],
        pval_random=res_r["all_comparisons"]["p"],
        Comp_common=res_c["components"]["TE"],
        seComp_common=res_c["components"]["seTE"],
        lower_Comp_common=res_c["components"]["lower"],
        upper_Comp_common=res_c["components"]["upper"],
        statistic_Comp_common=res_c["components"]["statistic"],
        pval_Comp_common=res_c["components"]["p"],
        Comp_random=res_r["components"]["TE"],
        seComp_random=res_r["components"]["seTE"],
        lower_Comp_random=res_r["components"]["lower"],
        upper_Comp_random=res_r["components"]["upper"],
        statistic_Comp_random=res_r["components"]["statistic"],
        pval_Comp_random=res_r["components"]["p"],
        Comb_common=res_c["combinations"]["TE"],
        seComb_common=res_c["combinations"]["seTE"],
        lower_Comb_common=res_c["combinations"]["lower"],
        upper_Comb_common=res_c["combinations"]["upper"],
        statistic_Comb_common=res_c["combinations"]["statistic"],
        pval_Comb_common=res_c["combinations"]["p"],
        Comb_random=res_r["combinations"]["TE"],
        seComb_random=res_r["combinations"]["seTE"],
        lower_Comb_random=res_r["combinations"]["lower"],
        upper_Comb_random=res_r["combinations"]["upper"],
        statistic_Comb_random=res_r["combinations"]["statistic"],
        pval_Comb_random=res_r["combinations"]["p"],
        TE_nma_common=nma_nan,
        seTE_nma_common=nma_nan,
        lower_nma_common=nma_nan,
        upper_nma_common=nma_nan,
        TE_nma_random=nma_nan,
        seTE_nma_random=nma_nan,
        lower_nma_random=nma_nan,
        upper_nma_random=nma_nan,
        TE_cnma_common=res_c["comparisons"]["TE"],
        seTE_cnma_common=res_c["comparisons"]["seTE"],
        lower_cnma_common=res_c["comparisons"]["lower"],
        upper_cnma_common=res_c["comparisons"]["upper"],
        TE_cnma_random=res_r["comparisons"]["TE"],
        seTE_cnma_random=res_r["comparisons"]["seTE"],
        lower_cnma_random=res_r["comparisons"]["lower"],
        upper_cnma_random=res_r["comparisons"]["upper"],
        Q_additive=q_additive,
        df_Q_additive=df_q_additive,
        pval_Q_additive=res_c["pval_Q_additive"],
        Q_standard=q_standard,
        df_Q_standard=df_q_standard,
        pval_Q_standard=pval_q_standard,
        Q_diff=q_diff,
        df_Q_diff=df_q_diff,
        pval_Q_diff=pval_q_diff,
        tau=tau, tau2=tau2,
        I2=res_c["I2"], lower_I2=res_c["lower_I2"], upper_I2=res_c["upper_I2"],
        A_matrix=a_mat,
        B_matrix=b_mat,
        X_matrix=x_mat,
        C_matrix=c_mat,
        L_matrix_common=res_c["L"],
        Lplus_matrix_common=res_c["Lplus"],
        H_matrix_common=res_c["H"],
        Cov_common=res_c["Cov"],
        L_matrix_random=res_r["L"],
        Lplus_matrix_random=res_r["Lplus"],
        H_matrix_random=res_r["H"],
        Cov_random=res_r["Cov"],
    )
