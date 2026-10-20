"""Binary network meta-analysis -- the ``netmeta::netmetabin`` twin.

Arm-level 2x2 counts in, network estimates out.  Four estimators share this
code path:

* ``"MH"`` -- Mantel-Haenszel odds ratios with Stijnen et al. (2010)'s
  random-effects model (hypergeometric variance correction);
* ``"NCH"`` -- conditional non-central hypergeometric likelihood;
* ``"LRP"`` -- Jeffreys-penalised logistic regression (Firth), the only
  method with a genuine random-effects counterpart;
* ``"Inverse"`` -- delegates to :func:`pimeta.netmeta`, i.e. it treats the
  per-comparison log-ORs as effect sizes.

``NetmetabinResult`` and ``netmetabin`` are re-exported from
:mod:`pimeta.netmeta`, so ``pimeta.netmetabin`` and
``pimeta.netmeta.netmetabin`` resolve to the same objects as before the
split.  ``_cochran_q`` / ``_dl_tau2`` / ``netmeta`` still live in
``pimeta.netmeta`` and are imported inside the functions that need them,
because that module imports this one.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
from scipy import stats

from .core import _cochran_q, _dl_tau2


@dataclass
class NetmetabinResult:
    """Binary network meta-analysis (netmeta::netmetabin), common-effects.

    Implements the Mantel-Haenszel NMA of Stijnen et al. (2010): per-design
    log-OR contrasts ``L.xy`` with variances/covariances ``V`` are combined by
    weighted least squares into basic contrasts ``TE.basic``, then expanded to
    all treatment pairs via the hat-like matrix ``H``.  All effect quantities
    are reported on the **log-OR** scale (matching ``netmetabin(...)$TE.common``
    etc. before back-transformation).
    """

    trts: list
    k: int                          # number of studies
    m: int                          # number of comparisons
    n: int                          # number of treatments
    d: int                          # number of designs
    designs: list
    sm: str
    method: str
    level: float
    level_ma: float
    reference_group: str
    cc_pooled: bool

    # Common effects (log-OR scale)
    TE_common: np.ndarray = field(default_factory=lambda: np.empty((0, 0)))
    seTE_common: np.ndarray = field(default_factory=lambda: np.empty((0, 0)))
    lower_common: np.ndarray = field(default_factory=lambda: np.empty((0, 0)))
    upper_common: np.ndarray = field(default_factory=lambda: np.empty((0, 0)))
    statistic_common: np.ndarray = field(default_factory=lambda: np.empty((0, 0)))
    pval_common: np.ndarray = field(default_factory=lambda: np.empty((0, 0)))

    # Direct (per-comparison Mantel-Haenszel) effects
    TE_direct_common: np.ndarray = field(default_factory=lambda: np.empty((0, 0)))
    seTE_direct_common: np.ndarray = field(default_factory=lambda: np.empty((0, 0)))
    lower_direct_common: np.ndarray = field(default_factory=lambda: np.empty((0, 0)))
    upper_direct_common: np.ndarray = field(default_factory=lambda: np.empty((0, 0)))
    statistic_direct_common: np.ndarray = field(default_factory=lambda: np.empty((0, 0)))
    pval_direct_common: np.ndarray = field(default_factory=lambda: np.empty((0, 0)))
    Q_direct: np.ndarray = field(default_factory=lambda: np.empty((0, 0)))
    tau2_direct: np.ndarray = field(default_factory=lambda: np.empty((0, 0)))
    I2_direct: np.ndarray = field(default_factory=lambda: np.empty((0, 0)))

    # Inconsistency (Stijnen Q); overall Q/tau^2/I^2 are undefined for MH.
    Q: float = float("nan")
    df_Q: float = float("nan")
    pval_Q: float = float("nan")
    Q_inconsistency: float = float("nan")
    df_Q_inconsistency: float = float("nan")
    pval_Q_inconsistency: float = float("nan")
    I2: float = float("nan")
    tau: float = float("nan")
    tau2: float = float("nan")

    # Random effects (method="LRP" only: the common estimate with the SE
    # multiplied by the dispersion factor `phi`).
    TE_random: np.ndarray = field(default_factory=lambda: np.empty((0, 0)))
    seTE_random: np.ndarray = field(default_factory=lambda: np.empty((0, 0)))
    lower_random: np.ndarray = field(default_factory=lambda: np.empty((0, 0)))
    upper_random: np.ndarray = field(default_factory=lambda: np.empty((0, 0)))
    phi: float = float("nan")

    A_matrix: np.ndarray = field(default_factory=lambda: np.empty((0, 0)))
    H_matrix_common: np.ndarray = field(default_factory=lambda: np.empty((0, 0)))

    # Raw per-comparison data
    TE: list = field(default_factory=list)          # per-comparison log OR
    seTE: list = field(default_factory=list)
    treat1: list = field(default_factory=list)
    treat2: list = field(default_factory=list)
    studlab: list = field(default_factory=list)
    event1: list = field(default_factory=list)
    event2: list = field(default_factory=list)
    n1: list = field(default_factory=list)
    n2: list = field(default_factory=list)

    def _m(self, x):
        return x.tolist() if isinstance(x, np.ndarray) else x

    def to_dict(self) -> dict:
        return {
            "trts": self.trts, "k": self.k, "m": self.m, "n": self.n,
            "d": self.d, "designs": self.designs,
            "sm": self.sm, "method": self.method,
            "level": self.level, "level_ma": self.level_ma,
            "reference_group": self.reference_group, "cc_pooled": self.cc_pooled,
            "TE_common": self._m(self.TE_common),
            "seTE_common": self._m(self.seTE_common),
            "lower_common": self._m(self.lower_common),
            "upper_common": self._m(self.upper_common),
            "statistic_common": self._m(self.statistic_common),
            "pval_common": self._m(self.pval_common),
            "TE_direct_common": self._m(self.TE_direct_common),
            "seTE_direct_common": self._m(self.seTE_direct_common),
            "lower_direct_common": self._m(self.lower_direct_common),
            "upper_direct_common": self._m(self.upper_direct_common),
            "statistic_direct_common": self._m(self.statistic_direct_common),
            "pval_direct_common": self._m(self.pval_direct_common),
            "Q_direct": self._m(self.Q_direct),
            "tau2_direct": self._m(self.tau2_direct),
            "I2_direct": self._m(self.I2_direct),
            "Q": self.Q, "df_Q": self.df_Q, "pval_Q": self.pval_Q,
            "Q_inconsistency": self.Q_inconsistency,
            "df_Q_inconsistency": self.df_Q_inconsistency,
            "pval_Q_inconsistency": self.pval_Q_inconsistency,
            "I2": self.I2, "tau": self.tau, "tau2": self.tau2,
            "TE_random": self._m(self.TE_random),
            "seTE_random": self._m(self.seTE_random),
            "lower_random": self._m(self.lower_random),
            "upper_random": self._m(self.upper_random),
            "phi": self.phi,
            "A_matrix": self._m(self.A_matrix),
            "H_matrix_common": self._m(self.H_matrix_common),
            "TE": self.TE, "seTE": self.seTE,
            "treat1": self.treat1, "treat2": self.treat2,
            "studlab": self.studlab,
            "event1": self.event1, "event2": self.event2,
            "n1": self.n1, "n2": self.n2,
        }


def _netmetabin_nch(event1, n1, event2, n2, treat1, treat2, studlab, trts):
    """Conditional (non-central hypergeometric) network estimate.

    netmetabin(method = "NCH") maximises the study-conditional likelihood

        logL(x) = sum_i [ sum_{k >= 2} r_ik * eta_ik
                          - N1_i * log(n_i1 + sum_{k >= 2} n_ik exp(eta_ik)) ]

    with ``eta_ik = x[t_ik] - x[t_i1] * (t_i1 != 1)``, i.e. each study's total
    number of events is conditioned on, which removes the study-specific
    baseline risks.  Treatment 1 (the alphabetically first) is the implicit
    reference: ``x[1]`` cancels out of every ``eta`` and is left at 0.

    Returns ``(TE_basic, cov_basic)`` where ``TE_basic[k] = -x[k+1]`` and
    ``cov_basic`` is the inverse observed information of ``x[2:]`` -- exactly
    R's ``TE.basic`` and ``W``.
    """
    from scipy.optimize import minimize

    n_treat = len(trts)
    tindex = {t: i for i, t in enumerate(trts)}

    # Arm-level data grouped by study, arms ordered by treatment index.
    arms: dict[str, dict[str, tuple[float, float]]] = {}
    for s, t, e, nn in zip(studlab, treat1, event1, n1):
        arms.setdefault(s, {})[t] = (float(e), float(nn))
    for s, t, e, nn in zip(studlab, treat2, event2, n2):
        arms.setdefault(s, {})[t] = (float(e), float(nn))

    studies = []
    for s in sorted(arms):
        ordered = sorted(arms[s], key=lambda t: tindex[t])
        studies.append([(tindex[t], arms[s][t][0], arms[s][t][1]) for t in ordered])

    def _terms(x):
        """Per-study (loglik, gradient, observed information)."""
        ll = 0.0
        grad = np.zeros(n_treat)
        info = np.zeros((n_treat, n_treat))
        for arm_list in studies:
            t0, r0, n0 = arm_list[0]
            cond = 1.0 if t0 != 0 else 0.0
            total_events = r0
            a = []
            for (tk, rk, nk) in arm_list[1:]:
                eta = x[tk] - cond * x[t0]
                ak = nk * math.exp(eta)
                ll += rk * eta
                total_events += rk
                a.append((tk, rk, ak))
            s_tot = n0 + sum(ak for _tk, _rk, ak in a)
            ll -= total_events * math.log(s_tot)
            w = [(tk, ak / s_tot) for tk, _rk, ak in a]

            wbar = np.zeros(n_treat)
            outer = np.zeros((n_treat, n_treat))
            for k, (tk, wk) in enumerate(w):
                v = np.zeros(n_treat)
                v[tk] += 1.0
                v[t0] -= cond
                grad += (a[k][1] - total_events * wk) * v
                wbar += wk * v
                outer += wk * np.outer(v, v)
            info += total_events * (outer - np.outer(wbar, wbar))
        return ll, grad, info

    x0 = np.zeros(n_treat)
    opt = minimize(
        lambda z: -_terms(np.concatenate([[0.0], z]))[0],
        x0[1:],
        jac=lambda z: -_terms(np.concatenate([[0.0], z]))[1][1:],
        method="L-BFGS-B",
        options={"ftol": 1e-16, "gtol": 1e-12, "maxiter": 10000, "maxls": 60},
    )
    x = np.concatenate([[0.0], opt.x])

    # Newton polish on the analytic information: L-BFGS-B stops a little short
    # on this flat conditional likelihood.
    for _ in range(50):
        _ll, grad, info = _terms(x)
        sub = info[1:, 1:]
        try:
            step = np.linalg.solve(sub, grad[1:])
        except np.linalg.LinAlgError:
            break
        if not np.all(np.isfinite(step)) or np.max(np.abs(step)) < 1e-13:
            break
        x[1:] += step
    _ll, grad, info = _terms(x)

    te_basic = -x[1:]
    cov_basic = np.linalg.inv(info[1:, 1:])
    return te_basic, cov_basic


def _netmetabin_lrp(event1, n1, event2, n2, treat1, treat2, studlab, trts):
    """Penalised-likelihood network estimate (netmetabin method = "LRP").

    Fits one logistic regression over all arms,

        logit(p_ik) = alpha + (treatment effects) + (study effects)

    by Jeffreys-penalised (Firth) maximum likelihood -- R's
    ``glm(cbind(event, non.event) ~ as.factor(treat) + as.factor(studlab),
    method = brglm2::brglmFit, type = "MPL_Jeffreys")``.  The study effects
    absorb each study's baseline risk, so the treatment coefficients are the
    common log ORs against the alphabetically first treatment.

    Returns ``(TE_basic, cov_basic, phi)`` with ``TE_basic[k] = -(theta_k -
    theta_1)``, i.e. the same sign convention the MH and NCH branches use.
    """
    from .._firth import dispersion_phi, firth_logit, treatment_design

    tindex = {t: i for i, t in enumerate(trts)}
    arms: dict[str, dict[str, tuple[float, float]]] = {}
    for s, t, e, nn in zip(studlab, treat1, event1, n1):
        arms.setdefault(s, {})[t] = (float(e), float(nn))
    for s, t, e, nn in zip(studlab, treat2, event2, n2):
        arms.setdefault(s, {})[t] = (float(e), float(nn))

    studies = sorted(arms)
    sindex = {s: i for i, s in enumerate(studies)}
    trt_idx: list[int] = []
    study_idx: list[int] = []
    succ: list[float] = []
    tot: list[float] = []
    for s in studies:
        for t in sorted(arms[s], key=lambda v: tindex[v]):
            trt_idx.append(tindex[t])
            study_idx.append(sindex[s])
            succ.append(arms[s][t][0])
            tot.append(arms[s][t][1])

    x = treatment_design(trt_idx, study_idx, len(trts), len(studies))
    beta, vcov, mu = firth_logit(x, succ, tot)
    n_par = len(trts) - 1
    te_basic = -np.asarray(beta[1:1 + n_par], dtype=float)
    cov_basic = np.asarray(vcov[1:1 + n_par, 1:1 + n_par], dtype=float)
    phi = 1.0 if len(studies) == 1 else dispersion_phi(succ, tot, mu, x.shape[1])
    return te_basic, cov_basic, phi


def _netmetabin_mh(event1, n1, event2, n2, treat1, treat2, studlab,
                   level, level_ma, reference_group, cc_pooled, method="MH"):
    """Mantel-Haenszel network meta-analysis (Stijnen et al. 2010).

    ``event1/n1`` describe the ``treat1`` arm and ``event2/n2`` the ``treat2``
    arm of each comparison (already normalized so ``treat1 < treat2``).  Returns
    a dict of all quantities needed to build a :class:`NetmetabinResult`.
    """
    from ..metabin import _mh_or

    m = len(event1)
    trts = sorted(set(treat1) | set(treat2))
    n_treat = len(trts)
    treat_idx = {t: i for i, t in enumerate(trts)}

    # --- arm-level data -----------------------------------------------------
    # study -> {treat: (event, n)}
    study_arms = {}
    for s, t, e, nn in zip(studlab, treat1, event1, n1):
        study_arms.setdefault(s, {})[t] = (float(e), float(nn))
    for s, t, e, nn in zip(studlab, treat2, event2, n2):
        study_arms.setdefault(s, {})[t] = (float(e), float(nn))

    # Design = sorted treatment set of a study.
    study_design = {s: tuple(sorted(arms)) for s, arms in study_arms.items()}
    designs = sorted(set(study_design.values()))
    d = len(designs)
    design_studies = {dsg: sorted(s for s in study_design
                                   if study_design[s] == dsg)
                      for dsg in designs}
    n_d = [len(dsg) for dsg in designs]                       # treats / design
    k_d = [len(design_studies[dsg]) for dsg in designs]       # studies / design

    length_y = sum(nn - 1 for nn in n_d)
    y = np.zeros(length_y)
    V1 = np.zeros((length_y, length_y))
    V2 = np.zeros((length_y, length_y))
    X = np.zeros((length_y, n_treat - 1))
    list1 = np.zeros((length_y, 2), dtype=int)
    N_j = np.zeros(d, dtype=int)
    for i in range(1, d):
        N_j[i] = N_j[i - 1] + n_d[i - 1] - 1

    counter1 = counter2 = counter3 = 0

    for i in range(d):
        nd = n_d[i]
        kd = k_d[i]
        treats = list(designs[i])                             # sorted names
        studies = design_studies[designs[i]]                  # sorted names

        ev = np.zeros((kd, nd))                               # event
        nn_arm = np.zeros((kd, nd))                           # n
        ne = np.zeros((kd, nd))                               # non.event
        n_study_arr = np.zeros(kd)
        for si, s in enumerate(studies):
            n_study_arr[si] = sum(study_arms[s][t][1] for t in treats)
            for ti, t in enumerate(treats):
                e, n_ = study_arms[s][t]
                ev[si, ti] = e
                nn_arm[si, ti] = n_
                ne[si, ti] = n_ - e

        c_xy = np.zeros((nd, nd, kd))
        d_xy = np.zeros((nd, nd, kd))
        for st in range(kd):
            for t1 in range(nd):
                for t2 in range(nd):
                    c_xy[t1, t2, st] = ev[st, t1] * ne[st, t2] / n_study_arr[st]
                    d_xy[t1, t2, st] = (ev[st, t1] + ne[st, t2]) / n_study_arr[st]

        C_xy = c_xy.sum(axis=2)
        with np.errstate(divide="ignore", invalid="ignore"):
            L_xy = np.log(C_xy / C_xy.T)

        U_xyy = np.zeros((nd, nd))
        for j in range(nd):
            for k in range(nd):
                U_xyy[j, k] = (
                    (c_xy[j, k, :] * d_xy[j, k, :]).sum()
                    / (2.0 * C_xy[j, k] ** 2)
                    + (c_xy[j, k, :] * d_xy[k, j, :]
                       + c_xy[k, j, :] * d_xy[j, k, :]).sum()
                    / (2.0 * C_xy[j, k] * C_xy[k, j])
                    + (c_xy[k, j, :] * d_xy[k, j, :]).sum()
                    / (2.0 * C_xy[k, j] ** 2)
                )

        U_xyz = np.zeros((nd, nd, nd))
        for t1 in range(nd):
            for t2 in range(nd):
                for t3 in range(nd):
                    if t1 == t2 or t1 == t3 or t2 == t3:
                        continue
                    ps1 = ps2 = ps3 = ps4 = 0.0
                    for st in range(kd):
                        ns2 = n_study_arr[st] ** 2
                        ps1 += ev[st, t1] / ns2 * ne[st, t2] * ne[st, t3]
                        ps2 += nn_arm[st, t1] / ns2 * ne[st, t2] * ev[st, t3]
                        ps3 += nn_arm[st, t1] / ns2 * ev[st, t2] * ne[st, t3]
                        ps4 += ne[st, t1] / ns2 * ev[st, t2] * ev[st, t3]
                    U_xyz[t1, t2, t3] = (
                        ps1 / (3.0 * C_xy[t1, t2] * C_xy[t1, t3])
                        + ps2 / (3.0 * C_xy[t1, t2] * C_xy[t3, t1])
                        + ps3 / (3.0 * C_xy[t2, t1] * C_xy[t1, t3])
                        + ps4 / (3.0 * C_xy[t2, t1] * C_xy[t3, t1])
                    )

        L_bar_xy = np.zeros((nd, nd))
        for t1 in range(nd):
            for t2 in range(nd):
                L_bar_xy[t1, t2] = (L_xy[t1, :].sum() - L_xy[t2, :].sum()) / nd

        for t1 in range(nd):
            U_xyy[t1, t1] = 0.0

        U_plus_xx = np.zeros(nd)
        for t1 in range(nd):
            U_plus_xx[t1] = U_xyy[t1, :].sum() + U_xyz[t1, :, :].sum()

        U_new = np.zeros((nd, nd, nd))
        for t1 in range(nd):
            for t2 in range(nd):
                for t3 in range(nd):
                    U_new[t1, t2, t3] = (
                        ((t1 != t2) and (t1 != t3) and (t2 != t3))
                        * U_xyz[t1, t2, t3]
                        + ((t1 != t2) and (t2 == t3)) * U_xyy[t1, t2]
                    )

        U_plus_xy = np.zeros((nd, nd))
        for t1 in range(nd):
            for t2 in range(nd):
                U_plus_xy[t1, t2] = (
                    U_new[:, t1, t2].sum()
                    - U_new[t1, t2, :].sum()
                    - U_new[t2, t1, :].sum()
                    + U_new[t1, t2, t2]
                )

        Var_Lbar = np.zeros((nd, nd))
        for t1 in range(nd):
            for t2 in range(nd):
                Var_Lbar[t1, t2] = (
                    U_plus_xx[t1] - 2.0 * U_plus_xy[t1, t2] + U_plus_xx[t2]
                ) / nd ** 2

        CoVar_Lbar = np.zeros((nd, nd))
        for t1 in range(nd):
            for t2 in range(nd):
                CoVar_Lbar[t1, t2] = (
                    U_plus_xx[0] - U_plus_xy[0, t2] - U_plus_xy[t1, 0]
                    + U_plus_xy[t1, t2]
                ) / nd ** 2

        id_t2 = [treat_idx[t] for t in treats]

        for j in range(nd - 1):
            y[j + counter1] = L_bar_xy[0, j + 1]
        counter1 += nd - 1

        for j in range(nd - 1):
            for k in range(nd - 1):
                V1[j + counter2, k + counter2] = (k == j) * Var_Lbar[0, j + 1]
        counter2 += nd - 1

        for j in range(1, nd):
            for k in range(1, nd):
                V2[j + counter3 - 1, k + counter3 - 1] = (k != j) * CoVar_Lbar[j, k]
        counter3 += nd - 1

        for j in range(nd - 1):
            list1[N_j[i] + j, 0] = id_t2[0]
            list1[N_j[i] + j, 1] = id_t2[j + 1]

    V = V1 + V2
    for i in range(length_y):
        for k in range(n_treat - 1):
            if list1[i, 0] == k + 1:
                X[i, k] = -1.0
            if list1[i, 1] == k + 1:
                X[i, k] = 1.0

    W = np.linalg.inv(V)
    XWX = X.T @ W @ X
    XWy = X.T @ W @ y

    # Hat-like matrix expanding basic contrasts to all pairwise contrasts.
    nrows_H = n_treat * (n_treat - 1) // 2
    H = np.zeros((nrows_H, n_treat - 1))
    for r in range(n_treat - 1):
        H[r, r] = 1.0
    if n_treat > 2:
        h1 = []
        for i in range(1, n_treat - 1):
            for j in range(i + 1, n_treat):
                h1.append((i, j))
        for idx_, (i, j) in enumerate(h1):
            for k in range(n_treat - 1):
                H[idx_ + n_treat - 1, k] = -float(i == k + 1) + float(j == k + 1)

    phi = float("nan")
    if method == "NCH":
        TE_basic, cov_basic = _netmetabin_nch(
            event1, n1, event2, n2, treat1, treat2, studlab, trts)
        cov_d_hat = H @ cov_basic @ H.T
    elif method == "LRP":
        TE_basic, cov_basic, phi = _netmetabin_lrp(
            event1, n1, event2, n2, treat1, treat2, studlab, trts)
        cov_d_hat = H @ cov_basic @ H.T
    else:
        TE_basic = np.linalg.solve(XWX, XWy)
        cov_d_hat = H @ np.linalg.solve(XWX, H.T)

    d_hat = H @ TE_basic

    # Expand to the n x n TE.common / seTE.common matrices.
    idx_lower = [(r, c) for c in range(n_treat) for r in range(c + 1, n_treat)]
    TE_common = np.full((n_treat, n_treat), np.nan)
    for pos, (r, c) in enumerate(idx_lower):
        TE_common[r, c] = d_hat[pos]
    TE_common = TE_common.T
    for pos, (r, c) in enumerate(idx_lower):
        TE_common[r, c] = -d_hat[pos]
    np.fill_diagonal(TE_common, 0.0)

    se_diag = np.sqrt(np.diag(cov_d_hat))
    seTE_common = np.full((n_treat, n_treat), np.nan)
    for pos, (r, c) in enumerate(idx_lower):
        seTE_common[r, c] = se_diag[pos]
    seTE_common = seTE_common.T
    for pos, (r, c) in enumerate(idx_lower):
        seTE_common[r, c] = se_diag[pos]
    np.fill_diagonal(seTE_common, 0.0)

    # Stijnen Q (inconsistency) -- only defined for the Mantel-Haenszel fit.
    if method == "MH":
        resid = y - X @ TE_basic
        Q = float(resid @ np.linalg.solve(V, resid))
        df_Q = sum(nn - 1 for nn in n_d) - (n_treat - 1)
        pval_Q = float(stats.chi2.sf(Q, df_Q)) if df_Q > 0 else float("nan")
    else:
        Q = df_Q = pval_Q = float("nan")

    # Confidence intervals (netmeta::ci).
    z = stats.norm.ppf(1.0 - (1.0 - level_ma) / 2.0)
    lower_common = TE_common - z * seTE_common
    upper_common = TE_common + z * seTE_common
    with np.errstate(divide="ignore", invalid="ignore"):
        statistic_common = TE_common / seTE_common
    pval_common = 2.0 * stats.norm.cdf(-np.abs(statistic_common))
    for i in range(n_treat):
        statistic_common[i, i] = float("nan")
        pval_common[i, i] = float("nan")

    # Adjacency (number of direct comparisons) matrix.
    A = np.zeros((n_treat, n_treat))
    for a, b in zip(treat1, treat2):
        A[treat_idx[a], treat_idx[b]] += 1
        A[treat_idx[b], treat_idx[a]] += 1

    # Direct per-comparison Mantel-Haenszel estimates (Q.Cochrane = FALSE).
    TE_direct = np.full((n_treat, n_treat), np.nan)
    seTE_direct = np.full((n_treat, n_treat), np.nan)
    Q_direct = np.full((n_treat, n_treat), np.nan)
    tau2_direct = np.full((n_treat, n_treat), np.nan)
    I2_direct = np.full((n_treat, n_treat), np.nan)
    for i in range(n_treat):
        for j in range(i + 1, n_treat):
            sel = [k for k in range(m)
                   if treat_idx[treat1[k]] == i and treat_idx[treat2[k]] == j]
            if not sel:
                continue
            a = np.array([event1[k] for k in sel], dtype=float)
            nn1 = np.array([n1[k] for k in sel], dtype=float)
            c = np.array([event2[k] for k in sel], dtype=float)
            nn2 = np.array([n2[k] for k in sel], dtype=float)
            b = nn1 - a
            dd = nn2 - c
            if method == "LRP":
                from ..metabin import lrp_common

                pair_est, pair_se, _pair_phi = lrp_common(a, nn1, c, nn2)
            else:
                pair_est, _mh_var = _mh_or(a, b, c, dd, nn1, nn2)
                pair_se = math.sqrt(_mh_var)
            TE_direct[i, j] = pair_est
            TE_direct[j, i] = -pair_est
            seTE_direct[i, j] = seTE_direct[j, i] = pair_se
            # Cochran Q centered on the inverse-variance mean (Q.Cochrane=FALSE).
            yi = np.log((a * dd) / (b * c))
            vi = 1.0 / a + 1.0 / b + 1.0 / c + 1.0 / dd
            se_i = np.sqrt(vi)
            q_pair = _cochran_q(yi, se_i)
            k_pair = len(sel)
            Q_direct[i, j] = Q_direct[j, i] = q_pair
            tau2_direct[i, j] = tau2_direct[j, i] = (
                float("nan") if method == "LRP" else _dl_tau2(yi, se_i)
            )
            I2_direct[i, j] = I2_direct[j, i] = (
                max(0.0, (q_pair - (k_pair - 1)) / q_pair) * 100.0
                if q_pair > 0 else 0.0
            )

    lower_direct = TE_direct - z * seTE_direct
    upper_direct = TE_direct + z * seTE_direct
    with np.errstate(divide="ignore", invalid="ignore"):
        statistic_direct = TE_direct / seTE_direct
    pval_direct = 2.0 * stats.norm.cdf(-np.abs(statistic_direct))
    for i in range(n_treat):
        statistic_direct[i, i] = float("nan")
        pval_direct[i, i] = float("nan")

    # Per-comparison log OR (metabin method="Inverse" TE) for the res$TE slot.
    TE_per = []
    seTE_per = []
    for k in range(m):
        a = event1[k]
        b = n1[k] - event1[k]
        c = event2[k]
        dd = n2[k] - event2[k]
        TE_per.append(math.log(a * dd / (b * c)))
        seTE_per.append(math.sqrt(1.0 / a + 1.0 / b + 1.0 / c + 1.0 / dd))

    return {
        "trts": trts, "k": len(study_arms), "m": m, "n": n_treat, "d": d,
        "designs": [":".join(dsg) for dsg in designs],
        "TE_common": TE_common, "seTE_common": seTE_common,
        "lower_common": lower_common, "upper_common": upper_common,
        "statistic_common": statistic_common, "pval_common": pval_common,
        "TE_direct": TE_direct, "seTE_direct": seTE_direct,
        "lower_direct": lower_direct, "upper_direct": upper_direct,
        "statistic_direct": statistic_direct, "pval_direct": pval_direct,
        "Q_direct": Q_direct, "tau2_direct": tau2_direct, "I2_direct": I2_direct,
        "Q": Q, "df_Q": df_Q, "pval_Q": pval_Q,
        "A": A, "H": H, "phi": phi,
        "TE": TE_per, "seTE": seTE_per,
    }


def netmetabin(
    event1,
    n1,
    event2,
    n2,
    treat1,
    treat2,
    studlab=None,
    sm: str = "OR",
    method: str = "MH",
    level: float = 0.95,
    level_ma: float | None = None,
    common: bool = True,
    random: bool = True,
    reference_group: str = "",
    baseline_reference: bool = True,
    cc_pooled: bool = False,
    incr: float = 0.5,
    tau_preset: float | None = None,
):
    """Binary network meta-analysis (netmeta::netmetabin).

    Args:
        event1, n1: events and group size of the ``treat1`` arm (per study).
        event2, n2: events and group size of the ``treat2`` arm (per study).
        treat1, treat2: treatment labels for each comparison.
        studlab: study labels. If None, each comparison is its own study.
        sm: summary measure ("OR" for MH; "OR"/"RR" for Inverse).
        method: "MH" (Mantel-Haenszel, default), "NCH" (conditional
            non-central hypergeometric likelihood), "LRP" (Jeffreys-penalised
            logistic regression, Firth -- the only method with a random-effects
            counterpart, obtained by inflating the SEs by ``phi``), or
            "Inverse" (classic NMA via :func:`netmeta`).
        level, level_ma: confidence levels.
        common, random: fit common / random effects (Inverse method only).
        cc_pooled: pool zero cells across studies (MH method).

    The "MH" method reproduces Stijnen et al. (2010); "Inverse" delegates to
    :func:`netmeta` after computing per-comparison log-OR estimates.
    """
    method = method.upper()
    if method not in ("MH", "INVERSE", "NCH", "LRP"):
        raise ValueError(
            "method must be one of 'MH', 'NCH', 'LRP' or 'Inverse'"
        )

    sm = sm.upper()
    if method in ("MH", "NCH"):
        if sm != "OR":
            sm = "OR"
        random = False
    elif method == "LRP":
        if sm not in ("", "OR"):
            raise ValueError(
                "logistic regression with penalised likelihood only possible "
                "with sm='OR'"
            )
        sm = "OR"

    event1 = np.asarray(event1, dtype=float)
    n1 = np.asarray(n1, dtype=float)
    event2 = np.asarray(event2, dtype=float)
    n2 = np.asarray(n2, dtype=float)
    treat1 = [str(t) for t in treat1]
    treat2 = [str(t) for t in treat2]
    m = len(event1)
    if not (len(n1) == len(event2) == len(n2) == len(treat1) == len(treat2) == m):
        raise ValueError("event1, n1, event2, n2, treat1, treat2 must match")

    if studlab is None:
        studlab = [f"{i + 1}" for i in range(m)]
    else:
        studlab = [str(s) for s in studlab]

    if level_ma is None:
        level_ma = level

    # Normalize to treat1 < treat2 (swap arm data, not a sign flip).
    for i in range(m):
        if treat1[i] > treat2[i]:
            treat1[i], treat2[i] = treat2[i], treat1[i]
            event1[i], event2[i] = event2[i], event1[i]
            n1[i], n2[i] = n2[i], n1[i]

    if reference_group == "":
        reference_group = sorted(set(treat1) | set(treat2))[0]

    if method == "INVERSE":
        from ..metabin import metabin
        from ..netmeta import netmeta  # local import: cycle

        mb = metabin(
            list(event1), list(n1), list(event2), list(n2),
            sm=sm, method="Inverse", random=False,
        )
        te = list(mb.yi)
        se = [math.sqrt(v) for v in mb.vi]
        return netmeta(
            te, se, treat1, treat2, studlab, sm=sm, level=level,
            common=common, random=random, method_tau="DL",
            tau_preset=tau_preset, reference_group=reference_group,
            baseline_reference=baseline_reference,
        )

    res = _netmetabin_mh(
        list(event1), list(n1), list(event2), list(n2),
        treat1, treat2, studlab, level, level_ma, reference_group, cc_pooled,
        method=method,
    )

    te_r = se_r = lo_r = up_r = None
    if method == "LRP":
        zz = stats.norm.ppf(1.0 - (1.0 - level_ma) / 2.0)
        te_r = np.asarray(res["TE_common"], dtype=float)
        se_r = np.asarray(res["seTE_common"], dtype=float) * res["phi"]
        lo_r = te_r - zz * se_r
        up_r = te_r + zz * se_r
        np.fill_diagonal(se_r, 0.0)
        np.fill_diagonal(lo_r, 0.0)
        np.fill_diagonal(up_r, 0.0)

    return NetmetabinResult(
        trts=res["trts"], k=res["k"], m=res["m"], n=res["n"], d=res["d"],
        designs=res["designs"], sm="OR", method=method,
        level=level, level_ma=level_ma, reference_group=reference_group,
        cc_pooled=cc_pooled,
        TE_common=res["TE_common"], seTE_common=res["seTE_common"],
        lower_common=res["lower_common"], upper_common=res["upper_common"],
        statistic_common=res["statistic_common"], pval_common=res["pval_common"],
        TE_direct_common=res["TE_direct"], seTE_direct_common=res["seTE_direct"],
        lower_direct_common=res["lower_direct"], upper_direct_common=res["upper_direct"],
        statistic_direct_common=res["statistic_direct"],
        pval_direct_common=res["pval_direct"],
        Q_direct=res["Q_direct"], tau2_direct=res["tau2_direct"],
        I2_direct=res["I2_direct"],
        Q=float("nan"), df_Q=float("nan"), pval_Q=float("nan"),
        Q_inconsistency=res["Q"], df_Q_inconsistency=res["df_Q"],
        pval_Q_inconsistency=res["pval_Q"],
        I2=float("nan"), tau=float("nan"), tau2=float("nan"),
        TE_random=te_r if te_r is not None else np.empty((0, 0)),
        seTE_random=se_r if se_r is not None else np.empty((0, 0)),
        lower_random=lo_r if lo_r is not None else np.empty((0, 0)),
        upper_random=up_r if up_r is not None else np.empty((0, 0)),
        phi=res["phi"],
        A_matrix=res["A"], H_matrix_common=res["H"],
        TE=res["TE"], seTE=res["seTE"],
        treat1=treat1, treat2=treat2, studlab=studlab,
        event1=list(event1), event2=list(event2), n1=list(n1), n2=list(n2),
    )
