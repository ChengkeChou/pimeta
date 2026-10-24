"""Network meta-analysis, replicating R's ``netmeta`` (R眉cker 2012).

Implements the frequentist graph-theoretical approach of R眉cker (2012) and the
Senn et al. (2013) Aitken estimator, matching ``netmeta::netmeta`` exactly:

- Edge-vertex incidence matrix ``B`` (``createB``)
- Weighted Laplacian ``L = B' W B`` (Kirchhoff matrix)
- Moore-Penrose pseudoinverse ``Lplus`` via ``solve(L - J/n) + J/n``
- Hat matrix ``H = B Lplus B' W``; fitted effects ``v = H TE``
- Resistance-distance variances ``R[i,j] = Lplus[i,i] + Lplus[j,j] - 2 Lplus[i,j]``
- Variance-covariance of all contrasts ``Cov = B.full Lplus B.full'``
- Q (heterogeneity + inconsistency) statistic with ``df = 2*sum(1/narms) - (n-1)``
- tau^2 via the Jackson et al. (2012) method-of-moments estimator
- I^2 via Higgins & Thompson (2002) (through ``calcH`` / ``isquared``)
- Direct estimates per design via inverse-variance metagen with ``tau.preset``
- P-scores via ``netrank`` (one-sided p-value row means)

Numeric engine: numpy linear algebra only (no LLM, no hallucination).
"""

from __future__ import annotations

import math
import warnings
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from scipy import stats

from .data_compat import resolve
from .missing import complete_cases
from .nma.binary import (  # noqa: F401  (re-export: pip.api and docs use these)
    NetmetabinResult,
    netmetabin,
)
from .nma.complexity import (  # noqa: F401
    NetcomplexResult,
    netcomplex,
    netcomplexity,
)
from .nma.component import (  # noqa: F401  (re-export: documented public API;
    NetcombResult,  # _create_c_matrix/_compsplit/_ci are shared
    _ci,  # with netcomplex/netcomparison below)
    _compsplit,
    _create_c_matrix,
    discomb,
    netcomb,
)
from .nma.contrib import (  # noqa: F401  (re-export: documented public API)
    NetcontribResult,
    netcontrib,
)
from .nma.convert import crossnma2netmeta, gemtc2netmeta  # noqa: F401
from .nma.core import (  # noqa: F401  (re-export: the code below uses them all)
    _ZERO,
    Z_975,
    _adjusted_variances,
    _block_diag,
    _cochran_q,
    _connected,
    _create_b,
    _create_b_full,
    _dl_tau2,
    _fx,
    _invmat,
    _isquared,
    _iv_pool,
    _multiarm,
    _pair_index,
)

# The remaining sections live in their own modules.  They are re-exported here
# so ``pimeta.netmeta.<name>`` keeps working; nothing below this block uses
# them except ``hatmatrix()``, which shares ``_hatmatrix_aggr`` with ``netpath``.
from .nma.design import (  # noqa: F401  (network parameterisation layer)
    MultiArmStructure,
    TreatmentNetwork,
    consistency_residuals,
    multiarm_structure,
)
from .nma.graph import (  # noqa: F401  (re-export: documented public API)
    NetconnectionResult,
    NetdistanceResult,
    netconnection,
    netdistance,
)
from .nma.impact import (  # noqa: F401  (re-export; _comparison_list is shared
    NetimpactResult,  # with _hatmatrix_aggr further down this module)
    _comparison_list,
    netimpact,
)
from .nma.labels import treats  # noqa: F401
from .nma.league import NetleagueResult, netleague  # noqa: F401
from .nma.matrix import (  # noqa: F401
    NetbindResult,
    NetcomparisonResult,
    NetmatrixResult,
    netbind,
    netcomparison,
    netmatrix,
)
from .nma.measures import NetmeasuresResult, netmeasures  # noqa: F401
from .nma.pairwise import PairwiseData, pairwise  # noqa: F401
from .nma.pairwise_table import NetpairwiseResult, netpairwise  # noqa: F401
from .nma.paths import _hatmatrix_aggr, netpath  # noqa: F401
from .nma.ranking import VikorResult, vikor  # noqa: F401
from .nma.splitting import NetsplitResult, netsplit  # noqa: F401
from .nma.tables import (  # noqa: F401
    NetsubgroupResult,
    NettableResult,
    nettable,
    subgroup,
)

# ---------------------------------------------------------------------------
# Core R眉cker (2012) algorithm
# ---------------------------------------------------------------------------

def _nma_ruecker(te, se_adj, treat1_pos, treat2_pos, narms, studlab, n, level):
    """One pass of the R眉cker (2012) graph-theoretical NMA.

    ``se_adj`` is the per-comparison standard error *after* multi-arm variance
    adjustment (``_adjusted_variances``); the weight matrix is
    ``W = diag(1 / se_adj^2)``. Returns a dict of all statistics.
    """
    te = np.asarray(te, dtype=float)
    se_adj = np.asarray(se_adj, dtype=float)
    m = len(te)
    w = np.diag(1.0 / se_adj ** 2)

    b = _create_b(treat1_pos, treat2_pos, n)
    b_full = _create_b_full(n)

    # Laplacian and pseudoinverse.
    l = b.T @ w @ b
    lplus = _invmat(l)
    lplus[np.abs(lplus) < _ZERO] = 0.0

    # Resistance distances R (n x n) and effective variances V (m).
    d = np.diag(lplus)
    r = d[:, None] + d[None, :] - 2.0 * lplus
    v_vec = np.array([r[t1, t2] for t1, t2 in zip(treat1_pos, treat2_pos)])

    g = b @ lplus @ b.T
    h = g @ w
    h[np.abs(h) < _ZERO] = 0.0

    cov = b_full @ lplus @ b_full.T

    # Fitted effects on observed edges, and Q.
    fitted = h @ te
    resid = te - fitted
    q = float(resid @ w @ resid)

    # Degrees of freedom.
    df1 = 2.0 * sum(1.0 / float(a) for a in narms)
    df_q = df1 - (n - 1)
    pval_q = stats.chi2.sf(q, df_q) if df_q > 0 else float("nan")

    # tau^2 via Jackson et al. (2012) method of moments.
    i_m = np.eye(m)
    e = np.array(
        [[1.0 if studlab[i] == studlab[j] else 0.0 for j in range(m)]
         for i in range(m)],
        dtype=float,
    )
    if df_q == 0:
        tau2 = tau = float("nan")
        i2 = lower_i2 = upper_i2 = float("nan")
    else:
        denom = float(np.trace((i_m - h) @ ((b @ b.T) * (e / 2.0)) @ w))
        tau2 = max(0.0, (q - df_q) / denom) if denom > 0 else float("nan")
        tau = math.sqrt(tau2) if not math.isnan(tau2) else float("nan")
        i2, lower_i2, upper_i2 = _isquared(q, df_q, level)

    # Treatment potentials and full n x n contrast matrices.
    theta = lplus @ b.T @ w @ te
    te_pooled = theta[:, None] - theta[None, :]
    se_pooled = np.sqrt(r)

    z = stats.norm.ppf(1 - (1 - level) / 2)
    lower = te_pooled - z * se_pooled
    upper = te_pooled + z * se_pooled
    with np.errstate(divide="ignore", invalid="ignore"):
        statistic = te_pooled / se_pooled
    pval = 2.0 * stats.norm.sf(np.abs(statistic))

    # Diagonal, matching R: a treatment against itself has effect 0 and no
    # uncertainty, so TE / lower / upper are 0.  The z statistic and p-value
    # are 0/0 and stay NaN (netmeta prints NaN there too).
    np.fill_diagonal(te_pooled, 0.0)
    np.fill_diagonal(lower, 0.0)
    np.fill_diagonal(upper, 0.0)
    np.fill_diagonal(statistic, np.nan)
    np.fill_diagonal(pval, np.nan)

    # Adjacency matrix: A[i,j] = number of direct comparisons (D - M).
    mmat = b.T @ b
    dmat = np.diag(np.diag(mmat))
    a_mat = dmat - mmat

    # Q decomposition into pairwise heterogeneity + residual inconsistency.
    q_matrix = np.zeros((n, n), dtype=float)
    for i in range(n - 1):
        for j in range(i + 1, n):
            if a_mat[i, j] > 1:
                sel = [k for k in range(m)
                       if (treat1_pos[k] == i and treat2_pos[k] == j)
                       or (treat1_pos[k] == j and treat2_pos[k] == i)]
                q_matrix[i, j] = _cochran_q(te[sel], se_adj[sel])

    q_het = float(np.sum(q_matrix))
    q_inc = q - q_het

    return {
        "b": b,
        "b_full": b_full,
        "l": l,
        "lplus": lplus,
        "r": r,
        "v_vec": v_vec,
        "g": g,
        "h": h,
        "cov": cov,
        "fitted": fitted,
        "q": q,
        "df_q": df_q,
        "pval_q": pval_q,
        "tau2": tau2,
        "tau": tau,
        "i2": i2,
        "lower_i2": lower_i2,
        "upper_i2": upper_i2,
        "theta": theta,
        "te_pooled": te_pooled,
        "se_pooled": se_pooled,
        "lower": lower,
        "upper": upper,
        "statistic": statistic,
        "pval": pval,
        "a_mat": a_mat,
        "q_matrix": q_matrix,
        "q_het": q_het,
        "q_inc": q_inc,
        "w": w,
    }


# ---------------------------------------------------------------------------
# Krahn (2013) design-based decomposition of Cochran's Q
# ---------------------------------------------------------------------------


def _nma_krahn(te, se, treat1_pos, treat2_pos, studlab, narms, trts):
    """Krahn et al. (2013) design-based decomposition of Cochran's Q.

    Common-effects model (``tau.preset = 0``). Replicates
    ``netmeta::nma_krahn`` plus the within/between-design decomposition of
    ``netmeta::decomp.tau``. ``te`` / ``treat1_pos`` / ``treat2_pos`` are
    already normalized so that ``treat1 < treat2``.

    Returns a dict with ``Q_net``, ``Q_het``, ``Q_inc`` and the corresponding
    degrees of freedom ``df_net``, ``df_het``, ``df_inc``.
    """
    m = len(te)
    n = len(trts)
    te = np.asarray(te, dtype=float)
    se = np.asarray(se, dtype=float)

    narms_by_study = {}
    for k in range(m):
        narms_by_study[studlab[k]] = int(narms[k])
    selmulti = np.array(
        [narms_by_study[studlab[k]] > 2 for k in range(m)], dtype=bool)

    # --- Direct (pairwise) estimates per comparison pair ---
    comp_idx = {}
    for k in range(m):
        key = (int(treat1_pos[k]), int(treat2_pos[k]))
        comp_idx.setdefault(key, []).append(k)
    direct = {}
    for key, idx in comp_idx.items():
        te_i = te[idx]
        se_i = se[idx]
        te_all, _ = _iv_pool(te_i, se_i, 0.0)
        idx2 = [k for k in idx if not selmulti[k]]
        if idx2:
            te_2, se_2 = _iv_pool(te[idx2], se[idx2], 0.0)
            n2 = len(idx2)
        else:
            te_2 = se_2 = float("nan")
            n2 = 0
        direct[key] = {
            "TE.all": te_all,
            "TE.2arm": te_2, "seTE.2arm": se_2, "n.2arm": n2,
        }

    # Two-arm designs (comparisons with at least one two-arm study).
    twoarm_designs = sorted(k for k, d in direct.items() if d["n.2arm"] > 0)

    # --- Multi-arm study covariance blocks (star parameterization) ---
    multiarm_studies = [s for s in sorted(set(studlab))
                        if narms_by_study[s] > 2]
    covs = {}
    basis_idx = {}
    for s in multiarm_studies:
        idx = [k for k in range(m) if studlab[k] == s]
        idx_sorted = sorted(idx, key=lambda k: (int(treat1_pos[k]),
                                                int(treat2_pos[k])))
        cnt = {}
        for k in idx_sorted:
            t1 = int(treat1_pos[k])
            cnt[t1] = cnt.get(t1, 0) + 1
        most_t1 = max(cnt, key=lambda x: cnt[x])
        front = [k for k in idx_sorted if int(treat1_pos[k]) == most_t1]
        back = [k for k in idx_sorted if int(treat1_pos[k]) != most_t1]
        order = front + back
        nn = narms_by_study[s]
        se_o = np.array([se[k] for k in order])
        c = np.full((nn - 1, nn - 1), np.nan)
        kk = 0
        for i in range(nn - 2):
            for j in range(i + 1, nn - 1):
                val = (se_o[i] ** 2 + se_o[j] ** 2
                       - se_o[nn - 1 + kk] ** 2) / 2.0
                c[i, j] = c[j, i] = val
                kk += 1
        for i in range(nn - 1):
            c[i, i] = se_o[i] ** 2
        covs[s] = c
        basis_idx[s] = order[:nn - 1]

    # Group multi-arm studies by design (design name = t1 of first basis +
    # all t2 of basis contrasts).
    design_studies = {}
    for s in multiarm_studies:
        b = basis_idx[s]
        t1 = int(treat1_pos[b[0]])
        t2s = sorted(int(treat2_pos[k]) for k in b)
        dn = ":".join([trts[t1]] + [trts[t] for t in t2s])
        design_studies.setdefault(dn, []).append(s)
    multicomp = sorted(design_studies.keys())

    # --- Study-level TE vector, TE.dir and V.studies ---
    # Order matches R: two-arm contrasts (by studlab), then multi-arm basis
    # contrasts (by studlab, then design grouping).
    study_order = sorted(range(m), key=lambda k: studlab[k])
    twoarm_order = [k for k in study_order if not selmulti[k]]
    multi_order = []
    for dn in multicomp:
        for s in design_studies[dn]:
            multi_order.extend(basis_idx[s])

    te_vec = np.array([te[k] for k in twoarm_order + multi_order])
    blocks = [[se[k] ** 2] for k in twoarm_order]
    for dn in multicomp:
        for s in design_studies[dn]:
            blocks.append(covs[s])
    v_studies = _block_diag(blocks)

    # Study-level TE.dir: two-arm -> TE.2arm; multi-arm basis -> TE.agg
    # (each multi-arm study replicates the design-level TE.agg for its
    #  (narms - 1) basis contrasts).
    te_dir_twoarm = [direct[(int(treat1_pos[k]), int(treat2_pos[k]))]["TE.2arm"]
                     for k in twoarm_order]
    te_agg_map = {}
    v3_agg_blocks = []
    for dn in multicomp:
        studs = design_studies[dn]
        dim = narms_by_study[studs[0]] - 1
        lsum = np.zeros((dim, dim))
        weighted = np.zeros((dim, 1))
        for s in studs:
            c_inv = np.linalg.inv(covs[s])
            lsum += c_inv
            te_basis = np.array([te[k] for k in basis_idx[s]])
            weighted += c_inv @ te_basis.reshape(-1, 1)
        covs3 = np.linalg.inv(lsum)
        te_agg = (covs3 @ weighted).ravel()
        te_agg_map[dn] = te_agg
        v3_agg_blocks.append(covs3)
    te_dir_multi = []
    for dn in multicomp:
        for _s in design_studies[dn]:
            te_dir_multi.extend(te_agg_map[dn])
    te_dir_study = np.concatenate(
        [np.array(te_dir_twoarm), np.array(te_dir_multi)])

    # --- Design-level V, TE.dir and X.obs ---
    v_design = np.diag([direct[d]["seTE.2arm"] ** 2 for d in twoarm_designs])
    v_design_block = v_design if v_design.size else np.zeros((0, 0))
    v = _block_diag([v_design_block] + v3_agg_blocks)

    te_dir_design = np.concatenate(
        [[direct[d]["TE.2arm"] for d in twoarm_designs]]
        + [te_agg_map[dn] for dn in multicomp])

    obs_rows = list(twoarm_designs)
    for dn in multicomp:
        s = design_studies[dn][0]
        obs_rows.extend([(int(treat1_pos[k]), int(treat2_pos[k]))
                         for k in basis_idx[s]])
    x_full = _fx(n)
    x_obs = np.vstack([x_full[_pair_index(i, j, n)] for (i, j) in obs_rows])

    # --- Network estimates (design level) and study-level mapping ---
    v_inv = np.linalg.inv(v)
    xt_v_x = x_obs.T @ v_inv @ x_obs
    xt_v_x_inv = np.linalg.inv(xt_v_x)
    h = x_full @ xt_v_x_inv @ x_obs.T @ v_inv
    te_net_full = h @ te_dir_design

    te_net_study = np.array([
        te_net_full[_pair_index(int(treat1_pos[k]), int(treat2_pos[k]), n)]
        for k in twoarm_order + multi_order])

    # --- Q decomposition (Krahn 2013 formulas (7) and (8)) ---
    v_studies_inv = np.linalg.inv(v_studies)
    d_het = te_vec - te_dir_study
    q_het = float(d_het @ v_studies_inv @ d_het)
    d_net = te_vec - te_net_study
    q_net = float(d_net @ v_studies_inv @ d_net)
    q_inc = q_net - q_het

    # --- Degrees of freedom ---
    df_net = len(te_vec) - (n - 1)
    df_het = 0.0
    for d in twoarm_designs:
        df_het += direct[d]["n.2arm"] - 1
    for dn in multicomp:
        studs = design_studies[dn]
        freq_d = sum(narms_by_study[s] - 1 for s in studs)
        narms_d = narms_by_study[studs[0]]
        df_het += freq_d - (narms_d - 1)
    df_inc = df_net - df_het

    return {
        "Q_net": q_net, "Q_het": q_het, "Q_inc": q_inc,
        "df_net": df_net, "df_het": df_het, "df_inc": df_inc,
    }


def _nma_krahn_full(te_raw, se_raw, treat1_raw, treat2_raw, studlab_raw,
                    narms_raw, trts_sorted, reference_group="",
                    tau_preset=0.0, sep_trts=":"):
    """Full transcription of ``netmeta::nma_krahn`` (Krahn et al. 2013).

    Unlike :func:`_nma_krahn` (which normalizes comparisons to
    ``treat1 < treat2`` and only returns the aggregate Q decomposition), this
    reproduces the *exact* direction convention of the R function:

    * ``TE = -x$TE``, then the reference group is swapped into ``treat1``
      position (comparisons involving the reference become ``ref:other``),
      so the stored ``TE`` equals ``treat2 - treat1`` (reversed).
    * ``trts`` are ordered ``c(reference, others alphabetically)``.

    Returns a dict with the design/study data frames and the design- and
    study-level covariance matrices needed by ``decomp.tau`` / ``tau.within``.
    """
    from collections import Counter

    m = len(te_raw)
    n = len(trts_sorted)
    te_raw = [float(v) for v in te_raw]
    se_raw = [float(v) for v in se_raw]
    # netmeta stores x$treat1/x$treat2 already normalized (treat1 < treat2);
    # normalize the raw comparisons here to match, flipping TE accordingly.
    treat1_n = list(treat1_raw)
    treat2_n = list(treat2_raw)
    for k in range(m):
        if treat1_n[k] > treat2_n[k]:
            treat1_n[k], treat2_n[k] = treat2_n[k], treat1_n[k]
            te_raw[k] = -te_raw[k]

    # --- treatment order: reference first ---
    if reference_group == "":
        trts = list(trts_sorted)
    else:
        trts = [reference_group] + [t for t in trts_sorted
                                    if t != reference_group]
    {t: i for i, t in enumerate(trts)}

    # --- studies.pre: TE = -x$TE, seTE = sqrt(se^2 + tau.preset^2) ---
    studies = []
    for k in range(m):
        studies.append({
            "studlab": studlab_raw[k],
            "treat1": treat1_n[k],
            "treat2": treat2_n[k],
            "TE": -te_raw[k],
            "seTE": math.sqrt(se_raw[k] ** 2 + tau_preset ** 2),
            "narms": int(narms_raw[k]),
        })
    # sort by studlab (stable)
    studies.sort(key=lambda r: r["studlab"])
    studies_pre = [dict(r) for r in studies]

    # --- swap reference into treat1 position ---
    for k in range(m):
        if studies_pre[k]["treat2"] == reference_group:
            studies[k]["treat1"] = studies_pre[k]["treat2"]
            studies[k]["treat2"] = studies_pre[k]["treat1"]
            studies[k]["TE"] = -studies_pre[k]["TE"]

    for r in studies:
        r["comparison"] = f"{r['treat1']}{sep_trts}{r['treat2']}"

    # --- direct estimates (metagen common-effects pooling per comparison) ---
    comp_to_rows = {}
    for k, r in enumerate(studies):
        comp_to_rows.setdefault(r["comparison"], []).append(k)
    comparisons = sorted(comp_to_rows.keys())

    direct = []
    for comp in comparisons:
        rows = comp_to_rows[comp]
        te_i = [studies[k]["TE"] for k in rows]
        se_i = [studies[k]["seTE"] for k in rows]
        te_all, se_all = _iv_pool(te_i, se_i, 0.0)
        rows2 = [k for k in rows if studies[k]["narms"] == 2]
        if rows2:
            te_2, se_2 = _iv_pool([studies[k]["TE"] for k in rows2],
                                  [studies[k]["seTE"] for k in rows2], 0.0)
            n2 = len(rows2)
        else:
            te_2 = se_2 = float("nan")
            n2 = 0
        direct.append({
            "comparison": comp, "TE": te_all, "seTE": se_all,
            "TE.2arm": te_2, "seTE.2arm": se_2, "n.2arm": n2,
        })
    direct_by_comp = {d["comparison"]: d for d in direct}

    # --- multi-arm study basis / covariance blocks ---
    selmulti = [r["narms"] > 2 for r in studies]
    multiarm = any(selmulti)
    multi_studlabs = sorted(set(
        r["studlab"] for r in studies if r["narms"] > 2))

    design_name = {}   # studlab -> design name
    basis_rows = {}    # studlab -> basis row indices (treat1-most, sort treat2)
    fb_rows = {}       # studlab -> front + back row indices
    covs = {}          # studlab -> (narms-1)x(narms-1) covariance

    if multiarm:
        for s in multi_studlabs:
            rows = [k for k in range(m) if studies[k]["studlab"] == s]
            t1_counts = Counter(studies[k]["treat1"] for k in rows)
            most_t1 = max(t1_counts, key=lambda t: t1_counts[t])
            b = [k for k in rows if studies[k]["treat1"] == most_t1]
            b.sort(key=lambda k: studies[k]["treat2"])
            basis_rows[s] = b
            dn = sep_trts.join([studies[b[0]]["treat1"]]
                               + [studies[k]["treat2"] for k in b])
            design_name[s] = dn
            # front + back (for covariance block)
            rows_sorted = sorted(
                rows, key=lambda k: (studies[k]["treat1"],
                                     studies[k]["treat2"]))
            front = [k for k in rows_sorted
                     if studies[k]["treat1"] == most_t1]
            back = [k for k in rows_sorted
                    if studies[k]["treat1"] != most_t1]
            fb_rows[s] = front + back
            order = fb_rows[s]
            nn = studies[order[0]]["narms"]
            se_o = [studies[k]["seTE"] for k in order]
            c = np.full((nn - 1, nn - 1), np.nan)
            kk = 0
            for i in range(nn - 2):
                for j in range(i + 1, nn - 1):
                    val = (se_o[i] ** 2 + se_o[j] ** 2
                           - se_o[nn - 1 + kk] ** 2) / 2.0
                    c[i, j] = c[j, i] = val
                    kk += 1
            for i in range(nn - 1):
                c[i, i] = se_o[i] ** 2
            covs[s] = c

    multicomp = sorted(set(design_name.values())) if multiarm else []

    # --- V.studies (study-level covariance) ---
    twoarm_rows = [k for k in range(m) if not selmulti[k]]
    v_studies_blocks = [[studies[k]["seTE"] ** 2] for k in twoarm_rows]
    v_studies_names = [studies[k]["comparison"] for k in twoarm_rows]
    if multiarm:
        for s in multi_studlabs:
            v_studies_blocks.append(covs[s])
            for _ in basis_rows[s]:
                v_studies_names.append(design_name[s])
    v_studies = _block_diag(v_studies_blocks)

    # --- direct2 (comparisons with two-arm evidence) ---
    direct2 = [d for d in direct if not math.isnan(d["seTE.2arm"])]
    twoarm_comps = [d["comparison"] for d in direct2]

    v_design = np.diag([d["seTE.2arm"] ** 2 for d in direct2])

    # --- multi-arm design-level aggregation ---
    v3_agg_blocks = []
    te_agg_flat = []
    if multiarm:
        for dn in multicomp:
            studs = [s for s in multi_studlabs if design_name[s] == dn]
            dim = studies[basis_rows[studs[0]][0]]["narms"] - 1
            lsum = np.zeros((dim, dim))
            weighted = np.zeros((dim, 1))
            for s in studs:
                c_inv = np.linalg.inv(covs[s])
                lsum += c_inv
                te_basis = np.array([studies[k]["TE"]
                                     for k in basis_rows[s]])
                weighted += c_inv @ te_basis.reshape(-1, 1)
            covs3 = np.linalg.inv(lsum)
            te_agg = (covs3 @ weighted).ravel()
            v3_agg_blocks.append(covs3)
            te_agg_flat.extend(te_agg)

    v = _block_diag([v_design] + v3_agg_blocks)

    te_dir_design = np.concatenate(
        [[d["TE.2arm"] for d in direct2]] + [np.asarray(te_agg_flat)])

    # --- X.full / X.obs ---
    x_full = _fx(n)
    trts_poss = []
    for i in range(n - 1):
        for j in range(i + 1, n):
            trts_poss.append(f"{trts[i]}{sep_trts}{trts[j]}")
    xfull_row = {name: k for k, name in enumerate(trts_poss)}

    obs_names = list(twoarm_comps)
    obs_row_comps = list(twoarm_comps)
    for dn in multicomp:
        s = [s for s in multi_studlabs if design_name[s] == dn][0]
        for k in basis_rows[s]:
            obs_names.append(dn)  # rownames of X.obs use the *design* name
            obs_row_comps.append(studies[k]["comparison"])
    x_obs = np.vstack([x_full[xfull_row[cp]] for cp in obs_row_comps])

    # --- network estimates (design level) ---
    v_inv = np.linalg.inv(v)
    xt_v_x = x_obs.T @ v_inv @ x_obs
    xt_v_x_inv = np.linalg.inv(xt_v_x)
    h = x_full @ xt_v_x_inv @ x_obs.T @ v_inv
    te_net_full = h @ te_dir_design

    cov_net_base = xt_v_x_inv
    co = []
    for i in range(n - 2):
        for j in range(1, n - 1):
            if i != j and i < j:
                co.append(cov_net_base[i, i] + cov_net_base[j, j]
                          - 2.0 * cov_net_base[i, j])
    cov_net = list(np.diag(cov_net_base)) + co
    se_net_full = np.sqrt(np.asarray(cov_net))

    net_by_name = {nm: (te_net_full[k], se_net_full[k])
                   for k, nm in enumerate(trts_poss)}

    # --- design data frame ---
    design_rows = []
    if multiarm:
        n_arms_per_design = [2] * len(direct2) + [
            studies[basis_rows[s][0]]["narms"] for dn in multicomp
            for s in [next(s2 for s2 in multi_studlabs
                           if design_name[s2] == dn)]]
        freq_per_design = [d["n.2arm"] for d in direct2]
        for dn in multicomp:
            studs = [s for s in multi_studlabs if design_name[s] == dn]
            freq_per_design.append(len(studs))
        # len.designs: how many rows each design occupies
        len_designs = [1] * len(direct2) + [
            studies[basis_rows[next(s2 for s2 in multi_studlabs
                                    if design_name[s2] == dn)][0]]["narms"] - 1
            for dn in multicomp]
        freq_col = []
        narms_col = []
        for idx_d, rep in enumerate(len_designs):
            for _ in range(rep):
                freq_col.append(freq_per_design[idx_d])
                narms_col.append(n_arms_per_design[idx_d])
        design_col = list(twoarm_comps) + [
            dn for dn in multicomp
            for _ in range(studies[basis_rows[next(
                s2 for s2 in multi_studlabs
                if design_name[s2] == dn)][0]]["narms"] - 1)]
        comparison_col = list(twoarm_comps)
        for dn in multicomp:
            s = next(s2 for s2 in multi_studlabs if design_name[s2] == dn)
            for k in basis_rows[s]:
                comparison_col.append(studies[k]["comparison"])
    else:
        len_designs = [1] * len(direct2)
        freq_col = [d["n.2arm"] for d in direct2]
        narms_col = [2] * len(direct2)
        design_col = list(twoarm_comps)
        comparison_col = list(twoarm_comps)

    se_dir_design = np.sqrt(np.diag(v))
    for r, (dn, cp, na, fr) in enumerate(
            zip(design_col, comparison_col, narms_col, freq_col)):
        te_net, se_net = net_by_name[cp]
        design_rows.append({
            "design": dn, "comparison": cp, "narms": na, "freq": fr,
            "TE.dir": float(te_dir_design[r]),
            "seTE.dir": float(se_dir_design[r]),
            "TE.net": float(te_net), "seTE.net": float(se_net),
        })

    # --- studies data frame (two-arm + multi-arm basis) ---
    study_rows = []
    base = [dict(r) for r in studies]  # studlab,treat1,treat2,TE,seTE,narms,comparison
    for k in twoarm_rows:
        study_rows.append(base[k])
    if multiarm:
        for dn in multicomp:
            s = next(s2 for s2 in multi_studlabs if design_name[s2] == dn)
            for k in basis_rows[s]:
                study_rows.append(base[k])
    # merge design-level columns (design, freq, TE.dir, ...) by (design, comparison)
    for r in study_rows:
        dname = design_name.get(r["studlab"], r["comparison"])
        r["design"] = dname
        match = next(d for d in design_rows
                     if d["design"] == dname
                     and d["comparison"] == r["comparison"])
        r["freq"] = match["freq"]
        r["TE.dir"] = match["TE.dir"]
        r["seTE.dir"] = match["seTE.dir"]
        r["TE.net"] = match["TE.net"]
        r["seTE.net"] = match["seTE.net"]
    # sort: two-arm by studlab, multi-arm by (studlab, treat1, treat2)
    lim = [r for r in study_rows if r["narms"] == 2]
    mult = [r for r in study_rows if r["narms"] > 2]
    lim.sort(key=lambda r: r["studlab"])
    mult.sort(key=lambda r: (r["studlab"], r["treat1"], r["treat2"]))
    study_rows = lim + mult

    d = len(set(dd["design"] for dd in design_rows))
    return {
        "n": n, "d": d, "trts": trts, "multicomp": multicomp,
        "design": design_rows, "studies": study_rows,
        "V": v, "V_studies": v_studies, "V_studies_names": v_studies_names,
        "X_obs": x_obs, "X_obs_names": obs_names, "X_full": x_full,
        "trts_poss": trts_poss,
        # --- extras used by netmeasures() -------------------------------- #
        "direct": direct, "direct_by_comp": direct_by_comp,
        "network": net_by_name, "H": h, "design_col": design_col,
        "x_obs_studies": np.vstack(
            [x_full[xfull_row[studies[k]["comparison"]]]
             for k in twoarm_rows]
            + ([x_full[xfull_row[studies[k]["comparison"]]]
                for dn in multicomp
                for k in basis_rows[next(s2 for s2 in multi_studlabs
                                         if design_name[s2] == dn)]]
               if multiarm else [])
        ),
        "studlab_studies": [studies[k]["studlab"] for k in twoarm_rows]
        + ([studies[k]["studlab"] for dn in multicomp
            for k in basis_rows[next(s2 for s2 in multi_studlabs
                                     if design_name[s2] == dn)]]
           if multiarm else []),
    }


def _decomp_tau(te_raw, se_raw, treat1_raw, treat2_raw, studlab_raw,
                narms_raw, trts_sorted, reference_group="", tau_preset=0.0):
    """Transcription of ``netmeta::decomp.tau`` (per-design Q decomposition)."""
    nmak = _nma_krahn_full(te_raw, se_raw, treat1_raw, treat2_raw,
                           studlab_raw, narms_raw, trts_sorted,
                           reference_group, tau_preset)
    if nmak["n"] == 2:
        return {"d": 1, "n": 2}

    design = nmak["design"]
    studies = nmak["studies"]
    v = nmak["V"]
    v_studies = nmak["V_studies"]
    v_studies_names = nmak["V_studies_names"]
    x_obs = nmak["X_obs"]
    x_obs_names = nmak["X_obs_names"]
    multicomp = nmak["multicomp"]
    multiarm = len(multicomp) > 0

    designs = list(dict.fromkeys(dd["design"] for dd in design))

    # --- detach loop: TE.net.minus ---
    n_rows = x_obs.shape[0]
    n_cols = x_obs.shape[1]
    te_net_minus = np.full((n_rows, len(designs)), np.nan)
    df_net_minus = np.zeros(len(designs))
    v_inv = np.linalg.inv(v)
    te_dir = np.array([dd["TE.dir"] for dd in design])
    for di, dname in enumerate(designs):
        narms = next(dd["narms"] for dd in design if dd["design"] == dname)
        xb = np.hstack([x_obs, np.zeros((n_rows, narms - 1))])
        sel = np.array([name == dname for name in x_obs_names])
        xb[sel, :] = 0.0
        xb[sel, n_cols:(n_cols + narms - 1)] = np.eye(narms - 1)
        xvx = xb.T @ v_inv @ xb
        try:
            xvx_inv = np.linalg.inv(xvx)
            te_net_minus[:, di] = (xb @ xvx_inv @ xb.T @ v_inv @ te_dir)
        except np.linalg.LinAlgError:
            te_net_minus[:, di] = np.nan
        df_net_minus[di] = np.linalg.matrix_rank(xb)

    # --- within-design Q (indexed by *design row*, matching R's design$design) ---
    study_designs = [r["design"] for r in studies]
    q_het_design = np.full(len(design), np.nan)
    freq_d = np.full(len(design), np.nan)
    for dname in designs:
        idx = [k for k, dn in enumerate(study_designs) if dn == dname]
        te_i = np.array([studies[k]["TE"] for k in idx])
        te_dir_i = np.array([studies[k]["TE.dir"] for k in idx])
        sel_v = [k for k, nm in enumerate(v_studies_names) if nm == dname]
        vs = v_studies[np.ix_(sel_v, sel_v)]
        d_i = te_i - te_dir_i
        pos = next(i for i, dd in enumerate(design) if dd["design"] == dname)
        q_het_design[pos] = float(d_i @ np.linalg.inv(vs) @ d_i)
        freq_d[pos] = len(idx)

    narms_design = np.array([dd["narms"] for dd in design], dtype=float)
    nafd = np.isnan(freq_d)
    narms_design[nafd] = np.nan
    df_het_design = freq_d - (narms_design - 1)
    pval_het_design = np.array([
        stats.chi2.sf(q_het_design[i], df_het_design[i])
        if df_het_design[i] > 0 else np.nan
        for i in range(len(design))])

    # --- total Q.het / Q.net / Q.inc ---
    te_all = np.array([r["TE"] for r in studies])
    te_dir_all = np.array([r["TE.dir"] for r in studies])
    te_net_all = np.array([r["TE.net"] for r in studies])
    v_studies_inv = np.linalg.inv(v_studies)
    d_het = te_all - te_dir_all
    q_het = float(d_het @ v_studies_inv @ d_het)
    d_net = te_all - te_net_all
    q_net = float(d_net @ v_studies_inv @ d_net)
    q_inc = q_net - q_het

    df_het = float(np.nansum(df_het_design))
    df_net = float(np.nansum(freq_d) - x_obs.shape[1])
    df_inc = df_net - df_het

    # --- Q.inc.design (element-wise decomposition, R: t(a) %*% solve(V) * a) ---
    te_dir_design = np.array([dd["TE.dir"] for dd in design])
    te_net_design = np.array([dd["TE.net"] for dd in design])
    diff_d = te_dir_design - te_net_design
    qid = (diff_d @ v_inv) * diff_d
    q_inc_design = qid
    q_inc_design_names = [dd["design"] for dd in design]  # V's colnames

    # --- residuals / Q.inc.detach ---
    # residuals starts as (n_rows x n_designs) with columns = designs (15);
    # then reorder columns to design$design (16, multi-arm duplicated).
    residuals = np.full((n_rows, len(designs)), np.nan)
    for di in range(len(designs)):
        residuals[:, di] = te_dir_design - te_net_minus[:, di]
    designs_idx = {dn: di for di, dn in enumerate(designs)}
    design_names = [dd["design"] for dd in design]
    residuals = residuals[:, [designs_idx[nm] for nm in design_names]]
    np.fill_diagonal(residuals, 0.0)
    if multiarm:
        for mc in multicomp:
            sel_mc = [k for k, nm in enumerate(design_names) if nm == mc]
            for r_ in sel_mc:
                for c_ in sel_mc:
                    residuals[r_, c_] = 0.0
    q_inc_detach = np.array([
        float(residuals[:, di] @ v_inv @ residuals[:, di])
        for di in range(len(design_names))])
    q_inc_detach[nafd] = np.nan
    df_inc_detach = np.full(len(design), np.nan)
    df_inc_detach[~nafd] = n_rows - df_net_minus
    pval_inc_detach = np.array([
        stats.chi2.sf(q_inc_detach[i], df_inc_detach[i])
        if not np.isnan(df_inc_detach[i]) and df_inc_detach[i] > 0 else np.nan
        for i in range(len(design))])

    q_decomp = {
        "Total": (q_net, df_net, stats.chi2.sf(q_net, df_net)
                  if df_net > 0 else np.nan),
        "Within designs": (q_het, df_het, stats.chi2.sf(q_het, df_het)
                           if df_het > 0 else np.nan),
        "Between designs": (q_inc, df_inc, stats.chi2.sf(q_inc, df_inc)
                            if df_inc > 0 else np.nan),
    }

    # Q.het.design rows (16 -> deduplicated; Q rounded to 15 decimals)
    het_rows = []
    for i in range(len(design)):
        het_rows.append({
            "design": design[i]["design"],
            "Q": float(round(q_het_design[i], 15)),
            "df": float(df_het_design[i]),
            "pval": float(pval_het_design[i])
            if not np.isnan(pval_het_design[i]) else np.nan,
        })
    seen = set()
    dedup = []
    for r in het_rows:
        if r["design"] not in seen:
            seen.add(r["design"])
            dedup.append(r)
    het_rows = dedup

    detach_rows = []
    for i in range(len(design)):
        detach_rows.append({
            "design": design[i]["design"],
            "Q": float(q_inc_detach[i]),
            "df": float(df_inc_detach[i]),
            "pval": float(pval_inc_detach[i])
            if not np.isnan(pval_inc_detach[i]) else np.nan,
        })
    seen = set()
    dedup = []
    for r in detach_rows:
        if r["design"] not in seen:
            seen.add(r["design"])
            dedup.append(r)
    detach_rows = dedup
    detach_rows = [r for r in detach_rows if not np.isnan(r["Q"])]

    return {
        "Q_decomp": q_decomp, "Q_het_design": het_rows,
        "Q_inc_detach": detach_rows, "Q_inc_design": q_inc_design,
        "Q_inc_design_names": q_inc_design_names,
        "residuals_inc_detach": residuals, "d": nmak["d"],
    }


def _tau_within(te_raw, se_raw, treat1_raw, treat2_raw, studlab_raw,
                narms_raw, trts_sorted, reference_group=""):
    """Transcription of ``netmeta::tau.within`` (DL tau under the
    inconsistency model)."""
    nmak = _nma_krahn_full(te_raw, se_raw, treat1_raw, treat2_raw,
                           studlab_raw, narms_raw, trts_sorted,
                           reference_group, 0.0)
    if nmak["n"] == 2:
        return float("nan")

    design = nmak["design"]
    studies = nmak["studies"]
    v_studies = nmak["V_studies"]

    # identity matrix t (design-level) with rownames "comparison_design"
    nd = len(design)
    t = np.eye(nd)
    names = [f"{dd['comparison']}_{dd['design']}" for dd in design]
    # X_incon_studies: rows of t for each study's "comparison_design"
    study_names = [f"{r['comparison']}_{r['design']}" for r in studies]
    t_by_name = {nm: i for i, nm in enumerate(names)}
    x_incon = np.vstack([t[t_by_name[nm]] for nm in study_names])
    y = np.array([r["TE"] for r in studies])

    def diagonal_blocks(v):
        # list of block index ranges for the block-diagonal v
        n = v.shape[0]
        blocks = []
        i = 0
        while i < n:
            # find block extent: nonzero row span
            nz = np.where(np.abs(v[i, :]) > 0)[0]
            if nz.size == 0:
                blocks.append([i])
                i += 1
            else:
                lo = int(nz[0])
                hi = int(nz[-1])
                blocks.append(list(range(lo, hi + 1)))
                i = hi + 1
        return blocks

    def v0(m):
        return np.diag(np.full(m, 0.5)) + np.full((m, m), 0.5)

    def w_fun(v, tausq):
        n = v.shape[0]
        w = np.eye(n)
        for b in diagonal_blocks(v):
            idx = np.ix_(b, b)
            blk = v[idx] + tausq * v0(len(b))
            w[idx] = np.linalg.inv(blk)
        return w

    def tau_dl(x, yv, v):
        w0 = w_fun(v, 0.0)
        c = np.linalg.inv(x.T @ w0 @ x)
        theta = c @ x.T @ w0 @ yv
        r = yv - x @ theta
        q = float(r @ w0 @ r)
        v0m = np.eye(v.shape[0])
        for b in diagonal_blocks(v):
            idx = np.ix_(b, b)
            v0m[idx] = v0(len(b))
        trace1 = float(np.trace(w0 @ v0m))
        trace2 = float(np.trace(c @ x.T @ w0 @ v0m @ w0 @ x))
        if trace1 == trace2:
            tausq = 0.0
        else:
            tausq = max(0.0, (q - v.shape[0] + x.shape[1]) / (trace1 - trace2))
        return math.sqrt(tausq)

    return tau_dl(x_incon, y, v_studies)


@dataclass
class DecompDesignResult:
    """Decomposition of designs (netmeta::decomp.design)."""

    Q_decomp: dict
    Q_het_design: list[dict]
    Q_inc_detach: list[dict]
    Q_inc_design: np.ndarray
    Q_inc_design_names: list[str]
    Q_inc_random: dict
    Q_inc_random_preset: dict | None
    residuals_inc_detach: np.ndarray
    tau_preset: float | None
    tau_within: float
    d: int

    def to_dict(self) -> dict:
        return {
            "Q_decomp": self.Q_decomp,
            "Q_het_design": self.Q_het_design,
            "Q_inc_detach": self.Q_inc_detach,
            "Q_inc_design": self.Q_inc_design.tolist(),
            "Q_inc_design_names": self.Q_inc_design_names,
            "Q_inc_random": self.Q_inc_random,
            "Q_inc_random_preset": self.Q_inc_random_preset,
            "residuals_inc_detach": self.residuals_inc_detach.tolist(),
            "tau_preset": self.tau_preset,
            "tau_within": self.tau_within,
            "d": self.d,
        }


def decomp_design(result: NetmetaResult, tau_preset=None) -> DecompDesignResult | None:
    """Decomposition of the Q statistic (netmeta::decomp.design, Krahn 2013).

    ``result`` must be a :class:`NetmetaResult` produced with the *raw*
    (un-normalized) comparison directions retained.
    """
    if result.n == 2:
        warnings.warn(
            "No decomposition possible for network meta-analysis with only "
            "two treatments.", stacklevel=2)
        return None
    if tau_preset is not None:
        if not isinstance(tau_preset, (int, float)) or tau_preset < 0:
            raise ValueError("tau_preset must be a non-negative numeric scalar")
        tau_preset = float(tau_preset)

    te_raw = result.TE_raw
    se_raw = result.seTE_raw
    treat1_raw = result.treat1_raw
    treat2_raw = result.treat2_raw
    studlab_raw = result.studlab_raw
    narms_raw = result.narms_raw
    trts = result.trts
    reference_group = result.reference_group

    tau_within = _tau_within(te_raw, se_raw, treat1_raw, treat2_raw,
                             studlab_raw, narms_raw, trts, reference_group)
    decomp_random = _decomp_tau(te_raw, se_raw, treat1_raw, treat2_raw,
                                studlab_raw, narms_raw, trts,
                                reference_group, tau_within)
    between = decomp_random["Q_decomp"]["Between designs"]
    q_inc_random = {
        "Q": float(between[0]), "df": float(between[1]),
        "pval": float(between[2]), "tau.within": float(tau_within),
    }

    q_inc_random_preset = None
    if tau_preset is not None:
        decomp_preset = _decomp_tau(te_raw, se_raw, treat1_raw, treat2_raw,
                                    studlab_raw, narms_raw, trts,
                                    reference_group, tau_preset)
        between_p = decomp_preset["Q_decomp"]["Between designs"]
        q_inc_random_preset = {
            "Q": float(between_p[0]), "df": float(between_p[1]),
            "pval": float(between_p[2]), "tau.preset": float(tau_preset),
        }

    dct = _decomp_tau(te_raw, se_raw, treat1_raw, treat2_raw, studlab_raw,
                      narms_raw, trts, reference_group, 0.0)
    return DecompDesignResult(
        Q_decomp=dct["Q_decomp"],
        Q_het_design=dct["Q_het_design"],
        Q_inc_detach=dct["Q_inc_detach"],
        Q_inc_design=dct["Q_inc_design"],
        Q_inc_design_names=dct["Q_inc_design_names"],
        Q_inc_random=q_inc_random,
        Q_inc_random_preset=q_inc_random_preset,
        residuals_inc_detach=dct["residuals_inc_detach"],
        tau_preset=tau_preset,
        tau_within=float(tau_within),
        d=dct["d"],
    )


def netheat_matrix(result: NetmetaResult, random: bool = False,
                   tau_preset: float | None = None
                   ) -> tuple[np.ndarray, list[str]] | None:
    """Design x design matrix drawn by ``netmeta::netheat`` (netmeta 3.7-0).

    R computes it inside ``netheat()`` from the Krahn decomposition:

    .. code-block:: r

        Q.inc.design.typ <- apply(residuals, 2, function(x)
                                  t(x) %*% solve(V) * x)
        diff <- Q.inc.design - Q.inc.design.typ
        t1   <- -diff

    so entry ``[i, j]`` is how much design ``i`` contributes to the
    inconsistency left over once design ``j`` is detached.

    Args:
        result: a :class:`NetmetaResult`.
        random: use the within-design tau (``netheat(x, random = TRUE)``).
        tau_preset: fix the heterogeneity variance instead of using 0
            (common effect) or ``tau.within``.

    Returns:
        ``(matrix, design labels)``, or ``None`` when the network has no loop
        (no between-design degrees of freedom), like R's warning.
    """
    if result.n == 2:
        return None

    args = (result.TE_raw, result.seTE_raw, result.treat1_raw,
            result.treat2_raw, result.studlab_raw, result.narms_raw,
            result.trts, result.reference_group)
    tau_within = _tau_within(*args)
    if tau_preset is not None:
        tau = float(tau_preset)
    elif random:
        tau = tau_within
    else:
        tau = 0.0

    nmak = _nma_krahn_full(*args, tau_preset=tau)
    dct = _decomp_tau(*args, tau)
    between_df = float(dct["Q_decomp"]["Between designs"][1])
    if between_df == 0.0:
        return None

    v = np.asarray(nmak["V"], dtype=float)
    residuals = np.asarray(dct["residuals_inc_detach"], dtype=float)
    q_inc = np.asarray(dct["Q_inc_design"], dtype=float)
    labels = list(dct["Q_inc_design_names"])

    # solve(V, residuals) is V^-1 x_j for every column j at once; R then
    # multiplies elementwise by x_j, keeping the row-vector orientation.
    solved = np.linalg.solve(v, residuals)
    q_typ = solved * residuals
    heat = q_typ - q_inc[:, None]

    # Drop designs whose whole column is NA (R: ``wi <- which(apply(t1, 2, ...))``).
    keep = [j for j in range(heat.shape[1])
            if not bool(np.all(np.isnan(heat[:, j])))]
    heat = heat[np.ix_(keep, keep)]
    labels = [labels[j] for j in keep]
    if heat.size == 0 or bool(np.all(np.isnan(heat))) or heat.shape[0] < 2:
        return None
    return heat, labels


# ---------------------------------------------------------------------------
# Result dataclasses
# ---------------------------------------------------------------------------

@dataclass
class NetmetaResult:
    """Results of a (common + random effects) network meta-analysis.

    Treatment-effect quantities are n x n matrices indexed by treatment name
    (rows/columns in the sorted treatment order). ``TE[i, j]`` is the network
    estimate of the effect of treatment *i* relative to treatment *j*
    (i.e. ``theta[i] - theta[j]``).
    """

    trts: list[str]
    k: int                          # number of studies
    m: int                          # number of pairwise comparisons (edges)
    n: int                          # number of treatments
    sm: str
    level: float
    method_tau: str

    # Common effects
    TE_common: np.ndarray = field(default_factory=lambda: np.empty((0, 0)))
    seTE_common: np.ndarray = field(default_factory=lambda: np.empty((0, 0)))
    lower_common: np.ndarray = field(default_factory=lambda: np.empty((0, 0)))
    upper_common: np.ndarray = field(default_factory=lambda: np.empty((0, 0)))
    statistic_common: np.ndarray = field(default_factory=lambda: np.empty((0, 0)))
    pval_common: np.ndarray = field(default_factory=lambda: np.empty((0, 0)))

    # Random effects
    TE_random: np.ndarray = field(default_factory=lambda: np.empty((0, 0)))
    seTE_random: np.ndarray = field(default_factory=lambda: np.empty((0, 0)))
    lower_random: np.ndarray = field(default_factory=lambda: np.empty((0, 0)))
    upper_random: np.ndarray = field(default_factory=lambda: np.empty((0, 0)))
    statistic_random: np.ndarray = field(default_factory=lambda: np.empty((0, 0)))
    pval_random: np.ndarray = field(default_factory=lambda: np.empty((0, 0)))

    # Direct evidence
    TE_direct_common: np.ndarray = field(default_factory=lambda: np.empty((0, 0)))
    seTE_direct_common: np.ndarray = field(default_factory=lambda: np.empty((0, 0)))
    TE_direct_random: np.ndarray = field(default_factory=lambda: np.empty((0, 0)))
    seTE_direct_random: np.ndarray = field(default_factory=lambda: np.empty((0, 0)))
    lower_direct_common: np.ndarray = field(default_factory=lambda: np.empty((0, 0)))
    upper_direct_common: np.ndarray = field(default_factory=lambda: np.empty((0, 0)))
    lower_direct_random: np.ndarray = field(default_factory=lambda: np.empty((0, 0)))
    upper_direct_random: np.ndarray = field(default_factory=lambda: np.empty((0, 0)))
    Q_direct: np.ndarray = field(default_factory=lambda: np.empty((0, 0)))
    tau2_direct: np.ndarray = field(default_factory=lambda: np.empty((0, 0)))
    I2_direct: np.ndarray = field(default_factory=lambda: np.empty((0, 0)))

    # Direct evidence proportion (netmeta::netmeasures) and indirect estimates
    # (back-calculation, netmeta::netsplit). All are n x n matrices indexed by
    # treatment name with the diagonal NA.
    prop_direct_common: np.ndarray = field(default_factory=lambda: np.empty((0, 0)))
    prop_direct_random: np.ndarray = field(default_factory=lambda: np.empty((0, 0)))
    TE_indirect_common: np.ndarray = field(default_factory=lambda: np.empty((0, 0)))
    seTE_indirect_common: np.ndarray = field(default_factory=lambda: np.empty((0, 0)))
    lower_indirect_common: np.ndarray = field(default_factory=lambda: np.empty((0, 0)))
    upper_indirect_common: np.ndarray = field(default_factory=lambda: np.empty((0, 0)))
    statistic_indirect_common: np.ndarray = field(default_factory=lambda: np.empty((0, 0)))
    pval_indirect_common: np.ndarray = field(default_factory=lambda: np.empty((0, 0)))
    TE_indirect_random: np.ndarray = field(default_factory=lambda: np.empty((0, 0)))
    seTE_indirect_random: np.ndarray = field(default_factory=lambda: np.empty((0, 0)))
    lower_indirect_random: np.ndarray = field(default_factory=lambda: np.empty((0, 0)))
    upper_indirect_random: np.ndarray = field(default_factory=lambda: np.empty((0, 0)))
    statistic_indirect_random: np.ndarray = field(default_factory=lambda: np.empty((0, 0)))
    pval_indirect_random: np.ndarray = field(default_factory=lambda: np.empty((0, 0)))

    # Matrices
    A_matrix: np.ndarray = field(default_factory=lambda: np.empty((0, 0)))
    B_matrix: np.ndarray = field(default_factory=lambda: np.empty((0, 0)))
    L_matrix_common: np.ndarray = field(default_factory=lambda: np.empty((0, 0)))
    Lplus_matrix_common: np.ndarray = field(default_factory=lambda: np.empty((0, 0)))
    L_matrix_random: np.ndarray = field(default_factory=lambda: np.empty((0, 0)))
    Lplus_matrix_random: np.ndarray = field(default_factory=lambda: np.empty((0, 0)))
    H_matrix_common: np.ndarray = field(default_factory=lambda: np.empty((0, 0)))
    H_matrix_random: np.ndarray = field(default_factory=lambda: np.empty((0, 0)))
    Cov_common: np.ndarray = field(default_factory=lambda: np.empty((0, 0)))
    Cov_random: np.ndarray = field(default_factory=lambda: np.empty((0, 0)))

    # Heterogeneity / inconsistency
    Q: float = float("nan")
    df_Q: float = float("nan")
    pval_Q: float = float("nan")
    I2: float = float("nan")
    lower_I2: float = float("nan")
    upper_I2: float = float("nan")
    tau: float = float("nan")
    tau2: float = float("nan")
    Q_heterogeneity: float = float("nan")
    df_Q_heterogeneity: float = float("nan")
    pval_Q_heterogeneity: float = float("nan")
    Q_inconsistency: float = float("nan")
    df_Q_inconsistency: float = float("nan")
    pval_Q_inconsistency: float = float("nan")
    Q_decomp: list[dict] = field(default_factory=list)

    # P-scores
    P_common: np.ndarray = field(default_factory=lambda: np.empty((0, 0)))
    P_random: np.ndarray = field(default_factory=lambda: np.empty((0, 0)))
    pscore_common: np.ndarray = field(default_factory=lambda: np.empty(0))
    pscore_random: np.ndarray = field(default_factory=lambda: np.empty(0))

    reference_group: str = ""
    baseline_reference: bool = True

    # Raw comparison data (un-normalized directions), needed by
    # decomp.design / nma.krahn (which uses its own direction convention).
    TE_raw: list = field(default_factory=list)
    seTE_raw: list = field(default_factory=list)
    treat1_raw: list = field(default_factory=list)
    treat2_raw: list = field(default_factory=list)
    studlab_raw: list = field(default_factory=list)
    narms_raw: list = field(default_factory=list)

    # Normalized treatment positions (0-based) and adjusted standard errors
    # (common/random), needed by netcontrib's hat-matrix aggregation.
    treat1_pos: list = field(default_factory=list)
    treat2_pos: list = field(default_factory=list)
    seTE_adj_common: list = field(default_factory=list)
    seTE_adj_random: list = field(default_factory=list)

    def _m(self, x):
        return x.tolist() if isinstance(x, np.ndarray) else x

    def to_dict(self) -> dict:
        return {
            "trts": self.trts,
            "k": self.k, "m": self.m, "n": self.n,
            "sm": self.sm, "level": self.level, "method_tau": self.method_tau,
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
            "TE_direct_common": self._m(self.TE_direct_common),
            "seTE_direct_common": self._m(self.seTE_direct_common),
            "TE_direct_random": self._m(self.TE_direct_random),
            "seTE_direct_random": self._m(self.seTE_direct_random),
            "lower_direct_common": self._m(self.lower_direct_common),
            "upper_direct_common": self._m(self.upper_direct_common),
            "lower_direct_random": self._m(self.lower_direct_random),
            "upper_direct_random": self._m(self.upper_direct_random),
            "Q_direct": self._m(self.Q_direct),
            "tau2_direct": self._m(self.tau2_direct),
            "I2_direct": self._m(self.I2_direct),
            "prop_direct_common": self._m(self.prop_direct_common),
            "prop_direct_random": self._m(self.prop_direct_random),
            "TE_indirect_common": self._m(self.TE_indirect_common),
            "seTE_indirect_common": self._m(self.seTE_indirect_common),
            "lower_indirect_common": self._m(self.lower_indirect_common),
            "upper_indirect_common": self._m(self.upper_indirect_common),
            "statistic_indirect_common": self._m(self.statistic_indirect_common),
            "pval_indirect_common": self._m(self.pval_indirect_common),
            "TE_indirect_random": self._m(self.TE_indirect_random),
            "seTE_indirect_random": self._m(self.seTE_indirect_random),
            "lower_indirect_random": self._m(self.lower_indirect_random),
            "upper_indirect_random": self._m(self.upper_indirect_random),
            "statistic_indirect_random": self._m(self.statistic_indirect_random),
            "pval_indirect_random": self._m(self.pval_indirect_random),
            "A_matrix": self._m(self.A_matrix),
            "B_matrix": self._m(self.B_matrix),
            "Q": self.Q, "df_Q": self.df_Q, "pval_Q": self.pval_Q,
            "I2": self.I2, "lower_I2": self.lower_I2, "upper_I2": self.upper_I2,
            "tau": self.tau, "tau2": self.tau2,
            "Q_heterogeneity": self.Q_heterogeneity,
            "df_Q_heterogeneity": self.df_Q_heterogeneity,
            "pval_Q_heterogeneity": self.pval_Q_heterogeneity,
            "Q_inconsistency": self.Q_inconsistency,
            "df_Q_inconsistency": self.df_Q_inconsistency,
            "pval_Q_inconsistency": self.pval_Q_inconsistency,
            "Q_decomp": self.Q_decomp,
            "P_common": self._m(self.P_common),
            "P_random": self._m(self.P_random),
            "pscore_common": self._m(self.pscore_common),
            "pscore_random": self._m(self.pscore_random),
            "reference_group": self.reference_group,
            "baseline_reference": self.baseline_reference,
            "TE_raw": self.TE_raw,
            "seTE_raw": self.seTE_raw,
            "treat1_raw": self.treat1_raw,
            "treat2_raw": self.treat2_raw,
            "studlab_raw": self.studlab_raw,
            "narms_raw": self.narms_raw,
            "treat1_pos": self.treat1_pos,
            "treat2_pos": self.treat2_pos,
            "seTE_adj_common": self.seTE_adj_common,
            "seTE_adj_random": self.seTE_adj_random,
        }


@dataclass
class NetrankResult:
    """P-scores from a network meta-analysis (``netmeta::netrank``).

    Field naming follows ``netrank()`` in netmeta 3.7-0, where - despite the
    name - ``ranking.common`` / ``ranking.random`` hold **P-scores**, and the
    pairwise probability matrices live in ``Pmatrix.common`` /
    ``Pmatrix.random``. ``Pscore.common`` does not exist in R.

    Both the R spelling (``ranking_common``, ``Pmatrix_common``, ``method``,
    ``nsim``, ...) and the earlier snake_case spelling (``pscore_common``,
    ``P_common``, ...) are available.
    """

    trts: list[str]
    pscore_common: np.ndarray
    pscore_random: np.ndarray
    P_common: np.ndarray
    P_random: np.ndarray
    small_values: str

    # -- R-compatible accessors ---------------------------------------------
    @property
    def ranking_common(self) -> np.ndarray:
        """P-scores, common effects (R: ``ranking.common``)."""
        return self.pscore_common

    @property
    def ranking_random(self) -> np.ndarray:
        """P-scores, random effects (R: ``ranking.random``)."""
        return self.pscore_random

    @property
    def ranking_fixed(self) -> np.ndarray:
        """P-scores, fixed effects (R: ``ranking.fixed``; = common)."""
        return self.pscore_common

    @property
    def Pscore_fixed(self) -> np.ndarray:
        """Deprecated alias of :attr:`ranking_fixed` (R: ``Pscore.fixed``)."""
        return self.pscore_common

    @property
    def Pscore_random(self) -> np.ndarray:
        """Deprecated alias of :attr:`ranking_random` (R: ``Pscore.random``)."""
        return self.pscore_random

    @property
    def Pmatrix_common(self) -> np.ndarray:
        """Pairwise probability matrix, common (R: ``Pmatrix.common``)."""
        return self.P_common

    @property
    def Pmatrix_random(self) -> np.ndarray:
        """Pairwise probability matrix, random (R: ``Pmatrix.random``)."""
        return self.P_random

    @property
    def Pmatrix_fixed(self) -> np.ndarray:
        """Pairwise probability matrix, fixed (R: ``Pmatrix.fixed``; = common)."""
        return self.P_common

    method: str = "P-score"
    nsim: int | None = None

    @property
    def common(self) -> bool:
        """Whether the common-effects model was fitted."""
        return bool(np.asarray(self.pscore_common).size)

    @property
    def random(self) -> bool:
        """Whether the random-effects model was fitted."""
        return bool(np.asarray(self.pscore_random).size)

    def to_dict(self) -> dict:
        return {
            "trts": self.trts,
            "pscore_common": self.pscore_common.tolist(),
            "pscore_random": self.pscore_random.tolist(),
            "P_common": self.P_common.tolist(),
            "P_random": self.P_random.tolist(),
            "small_values": self.small_values,
            "ranking_common": self.pscore_common.tolist(),
            "ranking_random": self.pscore_random.tolist(),
            "ranking_fixed": self.pscore_common.tolist(),
            "Pmatrix_common": self.P_common.tolist(),
            "Pmatrix_random": self.P_random.tolist(),
            "Pmatrix_fixed": self.P_common.tolist(),
            "method": self.method,
            "nsim": self.nsim,
        }


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def netrank(
    result: NetmetaResult,
    small_values: str = "desirable",
) -> NetrankResult:
    """P-scores (R眉cker & Schwarzer 2015), exact transcription of netrank.R.

    ``P[i, j]`` = probability treatment *i* is better than treatment *j*,
    computed from the two-sided p-value matrix as a one-sided p-value and
    averaged over rows. ``small_values`` = "desirable" means smaller effects
    are beneficial.
    """
    if small_values not in ("desirable", "undesirable"):
        raise ValueError("small_values must be 'desirable' or 'undesirable'")

    def _p_scores(te, pval):
        te = np.asarray(te, dtype=float)
        pval = np.asarray(pval, dtype=float)
        w = (1.0 + np.sign(te)) / 2.0
        if small_values == "desirable":
            p_mat = w * (pval / 2.0) + (1.0 - w) * (1.0 - pval / 2.0)
        else:
            p_mat = w * (1.0 - pval / 2.0) + (1.0 - w) * (pval / 2.0)
        scores = np.nanmean(p_mat, axis=1)
        return p_mat, scores

    p_common, score_common = _p_scores(result.TE_common, result.pval_common)
    p_random, score_random = _p_scores(result.TE_random, result.pval_random)

    return NetrankResult(
        trts=result.trts,
        pscore_common=score_common,
        pscore_random=score_random,
        P_common=p_common,
        P_random=p_random,
        small_values=small_values,
    )


def _calc_v_block(x: dict, sm: str) -> np.ndarray:
    """netmeta:::calcV 鈥?within-study covariance of one (multi-arm) study.

    Off-diagonals all equal the *reference arm's* variance (arm 2 after the
    star-shaped re-orientation); diagonals are ``seTE^2``.
    """
    k = len(x["seTE"])
    incr = float(x["incr2"][0]) if "incr2" in x else 0.0
    n2 = float(x["n2"][0]) + 2.0 * incr
    e2 = float(x["event2"][0]) + incr
    p2 = e2 / n2 if n2 > 0 else float("nan")
    sm = str(sm).upper()
    if sm == "OR":
        v = 1.0 / e2 + 1.0 / (float(x["n2"][0]) - float(x["event2"][0]) + incr)
    elif sm == "RR":
        v = (1.0 - p2) / (n2 * p2)
    elif sm == "RD":
        v = p2 * (1.0 - p2) / n2
    elif sm == "ASD":
        v = 0.25 / n2
    elif sm == "MD":
        v = float(x["sd2"][0]) ** 2 / n2
    elif sm == "SMD":
        v = 1.0 / n2
    elif sm == "IRR":
        v = 1.0 / e2
    elif sm == "IRD":
        v = e2 / float(x["time2"][0]) ** 2
    elif sm == "IRSD":
        v = 0.25 / float(x["time2"][0])
    elif k == 1:
        v = float(x["seTE"][0]) ** 2
    else:
        v = 0.0
    mat = np.full((k, k), v, dtype=float)
    np.fill_diagonal(mat, np.asarray(x["seTE"], dtype=float) ** 2)
    return mat


def _nma_tau_ml_reml(
    studlab,
    treat1,
    treat2,
    TE,
    seTE,
    reference_group: str,
    method: str,
    sm: str = "",
    arm_data: dict | None = None,
) -> float:
    """ML / REML network heterogeneity SD (netmeta's ``method.tau`` branch).

    R reduces the network to one *star* per study 鈥?every comparison that
    touches the study's reference treatment 鈥?expresses each as
    "comparison vs reference", then fits ``metafor::rma.mv`` with

    * ``mods = ~ V1 + ... + V(n-1) - 1`` where ``Vj`` are the ``contrmat``
      dummies for the non-reference treatments,
    * ``random = ~ factor(comparison) | studlab`` with ``struct = "CS"`` and
      ``rho = 0.5`` fixed,

    and returns ``sqrt(tau2)``.

    ``arm_data`` may carry the arm-level columns netmeta uses to build a
    blocked ``V`` (``n``, ``event``, ``mean``, ``sd``, ``time``, ``incr2``);
    without them ``V = diag(seTE^2)``, exactly as R does.
    """
    from collections import Counter

    from .rma_mv import rma_mv as _rma_mv

    m = len(TE)
    cols: dict[str, list] = {
        "studlab": list(studlab),
        "treat1": list(treat1),
        "treat2": list(treat2),
        "TE": list(TE),
        "seTE": list(seTE),
    }
    if arm_data:
        for name in ("n", "event", "mean", "sd", "time"):
            a, b = arm_data.get(name + "1"), arm_data.get(name + "2")
            if a is not None and b is not None:
                cols[name + "1"] = [float(v) for v in a]
                cols[name + "2"] = [float(v) for v in b]
        if "event1" in cols and arm_data.get("incr2") is not None:
            cols["incr2"] = [float(v) for v in arm_data["incr2"]]

    # 1. R's order(studlab, treat1, treat2) 鈥?lexicographic, stable.
    order_idx = sorted(range(m), key=lambda i: (cols["studlab"][i],
                                                cols["treat1"][i],
                                                cols["treat2"][i]))
    for key in cols:
        cols[key] = [cols[key][i] for i in order_idx]

    # 2. Keep only comparisons touching the study's reference arm, and
    #    re-orient them so that treat2 is always the reference.
    keep: list[bool] = []
    wo: list[bool] = []
    pos = 0
    while pos < m:
        study = cols["studlab"][pos]
        end = pos
        while end < m and cols["studlab"][end] == study:
            end += 1
        trts_i = sorted({cols["treat1"][i] for i in range(pos, end)}
                        | {cols["treat2"][i] for i in range(pos, end)})
        ref_i = reference_group if reference_group in trts_i else trts_i[-1]
        for i in range(pos, end):
            keep.append(not (cols["treat1"][i] != ref_i
                             and cols["treat2"][i] != ref_i))
            wo.append(cols["treat1"][i] == ref_i)
        pos = end

    sel = [i for i, k in enumerate(keep) if k]
    wo = [wo[i] for i in sel]
    for key in cols:
        cols[key] = [cols[key][i] for i in sel]

    for i, flip in enumerate(wo):
        if not flip:
            continue
        cols["TE"][i] = -cols["TE"][i]
        cols["treat1"][i], cols["treat2"][i] = cols["treat2"][i], cols["treat1"][i]
        for name in ("event", "n", "mean", "sd"):
            a, b = name + "1", name + "2"
            if a in cols:
                cols[a][i], cols[b][i] = cols[b][i], cols[a][i]
        # netmeta has a copy-paste slip here: time2 is left untouched while
        # time1 receives the old time2.  Reproduced verbatim.
        if "time1" in cols:
            cols["time1"][i], cols["time2"][i] = cols["time2"][i], cols["time2"][i]

    # 3. contrmat(): full dummies for treat1 minus treat2, with the most
    #    frequent treat2 level moved last and its column dropped.
    levels_ = sorted(set(cols["treat1"]) | set(cols["treat2"]))
    counts = Counter(cols["treat2"])
    last = sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))[0][0]
    lvls = [v for v in levels_ if v != last] + [last]

    k = len(cols["TE"])
    mods = np.zeros((k, max(len(lvls) - 1, 1)), dtype=float)
    for j, level in enumerate(lvls[:-1]):
        for i in range(k):
            mods[i, j] = (1.0 if cols["treat1"][i] == level else 0.0) \
                       - (1.0 if cols["treat2"][i] == level else 0.0)
    mods = mods[:, : len(lvls) - 1]

    # 4. Sampling covariance.
    if cols.get("n1") is not None and (
        "event1" in cols or "time1" in cols or "sd1" in cols
    ):
        blocks = []
        pos = 0
        while pos < k:
            end = pos
            study = cols["studlab"][pos]
            while end < k and cols["studlab"][end] == study:
                end += 1
            sub = {key: cols[key][pos:end] for key in cols}
            blocks.append(_calc_v_block(sub, sm))
            pos = end
        V = np.zeros((k, k), dtype=float)
        offset = 0
        for block in blocks:
            nb = block.shape[0]
            V[offset:offset + nb, offset:offset + nb] = block
            offset += nb
    else:
        V = np.diag(np.asarray(cols["seTE"], dtype=float) ** 2)

    if k == 1:
        # R fits rma.uni here and sets tau <- NA.
        return float("nan")

    comparison = [f"{cols['treat1'][i]} vs {cols['treat2'][i]}" for i in range(k)]
    res = _rma_mv(
        yi=cols["TE"],
        V=V,
        mods=mods,
        data={"comparison": comparison, "studlab": cols["studlab"]},
        random="~ comparison | studlab",
        struct="CS",
        rho=0.5,
        method=method.upper(),
        intercept=False,
    )
    if not res.tau2:
        return float("nan")
    return float(math.sqrt(max(res.tau2[0], 0.0)))


def netmeta(
    TE,
    seTE,
    treat1,
    treat2,
    studlab=None,
    sm: str = "",
    level: float = 0.95,
    common: bool = True,
    random: bool = True,
    method_tau: str = "DL",
    tau_preset: float | None = None,
    reference_group: str = "",
    baseline_reference: bool = True,
    small_values: str = "desirable",
    n1=None,
    n2=None,
    event1=None,
    event2=None,
    incr2=None,
    mean1=None,
    mean2=None,
    sd1=None,
    sd2=None,
    time1=None,
    time2=None,
    data=None,
    warn_missing: bool = True,
) -> NetmetaResult:
    """Frequentist graph-theoretical network meta-analysis (R眉cker 2012).

    Args:
        TE: per-comparison treatment effects (m).
        seTE: per-comparison standard errors (m).
        treat1, treat2: treatment labels for each comparison (m).
        studlab: study labels (m). If None, each comparison is its own
            (two-arm) study.
        sm: summary measure (informational, e.g. "MD", "RR").
        level: confidence level (0.95 -> 95%).
        common: fit the common-effects model.
        random: fit the random-effects model.
        method_tau: tau^2 estimator.  ``"DL"`` (default) is Jackson et al.'s
            network method of moments; ``"ML"`` and ``"REML"`` reproduce
            netmeta's ``metafor::rma.mv`` fit on the star-shaped per-study
            design (needs ``reference_group`` semantics identical to R, and
            optionally the arm-level data below for the blocked ``V``).
        tau_preset: optional preset tau (square-root of tau^2).
        reference_group: reference treatment label (informational).
        baseline_reference: whether the reference group is the baseline
            (comparisons are expressed as "other treatment vs reference").
        small_values: "desirable" if smaller effects are beneficial (P-scores).
        n1, n2: arm sample sizes (only used by ``method_tau="ML", "REML"``).
        event1, event2: arm event counts.
        incr2: continuity increment applied to the reference arm.
        mean1, mean2, sd1, sd2: arm means / SDs.
        time1, time2: arm person-times.
    """
    if data is not None:
        TE = resolve(TE, data, name="TE")
        seTE = resolve(seTE, data, name="seTE")
        treat1 = resolve(treat1, data, name="treat1", dtype=None)
        treat2 = resolve(treat2, data, name="treat2", dtype=None)
        studlab = resolve(studlab, data, name="studlab", dtype=None)
        n1 = resolve(n1, data, name="n1")
        n2 = resolve(n2, data, name="n2")
        event1 = resolve(event1, data, name="event1")
        event2 = resolve(event2, data, name="event2")
        incr2 = resolve(incr2, data, name="incr2")
        mean1 = resolve(mean1, data, name="mean1")
        mean2 = resolve(mean2, data, name="mean2")
        sd1 = resolve(sd1, data, name="sd1")
        sd2 = resolve(sd2, data, name="sd2")
        time1 = resolve(time1, data, name="time1")
        time2 = resolve(time2, data, name="time2")

    TE = list(TE)
    seTE = list(seTE)
    treat1 = list(treat1)
    treat2 = list(treat2)
    # Keep the raw (un-normalized) directions for decomp.design / nma.krahn.
    m = len(TE)
    if not (len(seTE) == len(treat1) == len(treat2) == m):
        raise ValueError("TE, seTE, treat1, treat2 must have equal length")

    # Drop incomplete comparisons (R's na.action).
    def _num(vals):
        return np.array([np.nan if v is None else float(v) for v in vals],
                        dtype=float)

    keep, _dropped = complete_cases(
        {"TE": _num(TE), "seTE": _num(seTE)},
        context="comparisons", warn=warn_missing)
    if not keep.all():
        TE = [v for v, ok in zip(TE, keep) if ok]
        seTE = [v for v, ok in zip(seTE, keep) if ok]
        treat1 = [v for v, ok in zip(treat1, keep) if ok]
        treat2 = [v for v, ok in zip(treat2, keep) if ok]
        if studlab is not None:
            studlab = [v for v, ok in zip(list(studlab), keep) if ok]
        m = len(TE)
        if m == 0:
            raise ValueError("no comparisons left after dropping missing values")

    TE_raw = list(TE)
    seTE_raw = list(seTE)
    treat1_raw = list(treat1)
    treat2_raw = list(treat2)

    # R: "Treatments must be different (arguments 'treat1' and 'treat2')."
    # Without this guard a self-comparison slips through the fit and later
    # trips a bare StopIteration inside the Krahn decomposition.
    same = [i for i in range(m) if treat1[i] == treat2[i]]
    if same:
        raise ValueError(
            "Treatments must be different (arguments 'treat1' and 'treat2'). "
            f"Comparison {same[0] + 1} has {treat1[same[0]]!r} on both sides.")

    if method_tau.upper() not in ("DL", "ML", "REML"):
        raise ValueError(
            f"method_tau={method_tau!r} not supported; use 'DL' (Jackson "
            "2012 method of moments), 'ML' or 'REML'"
        )

    # Treatment ordering: sorted alphabetically (netmeta default).
    trts = sorted(set(treat1) | set(treat2))
    n = len(trts)
    idx = {t: i for i, t in enumerate(trts)}

    # Normalize every comparison to treat1 < treat2 (netmeta::prepare), flipping
    # the sign of TE so it stays consistent with the incidence matrix B.
    for i in range(m):
        if treat1[i] > treat2[i]:
            treat1[i], treat2[i] = treat2[i], treat1[i]
            TE[i] = -TE[i]

    treat1_pos = [idx[t] for t in treat1]
    treat2_pos = [idx[t] for t in treat2]

    # Study labels (default: one study per comparison) and arm counts.
    if studlab is None:
        studlab = [f"study{i + 1}" for i in range(m)]
    else:
        studlab = list(studlab)
    studies = sorted(set(studlab))
    study_to_trts: dict[Any, set[Any]] = {s: set() for s in studies}
    for s, t1, t2 in zip(studlab, treat1, treat2):
        study_to_trts[s].add(t1)
        study_to_trts[s].add(t2)
    narms = [len(study_to_trts[s]) for s in studlab]
    studlab_raw = list(studlab)
    narms_raw = list(narms)

    # Check connectivity (the invmat pseudoinverse requires a connected graph).
    adj = np.zeros((n, n), dtype=bool)
    for t1, t2 in zip(treat1_pos, treat2_pos):
        adj[t1, t2] = adj[t2, t1] = True
    if n > 1 and not _connected(adj):
        raise ValueError("network is disconnected; cannot compute Laplacian "
                         "pseudoinverse (R眉cker 2012 requires a connected network)")

    se_arr = np.asarray(seTE, dtype=float)
    te_arr = np.asarray(TE, dtype=float)

    # Common effects: adjust multi-arm variances (tau2 = 0), then W = diag(1/se_adj^2).
    se_adj_c = np.sqrt(_adjusted_variances(
        se_arr, 0.0, treat1_pos, treat2_pos, studlab))
    res_c = _nma_ruecker(
        te_arr, se_adj_c, treat1_pos, treat2_pos, narms, studlab, n, level,
    )

    # Random effects: tau from DL (or preset).
    if tau_preset is not None:
        tau = float(tau_preset)
        tau2 = tau * tau
    elif method_tau.upper() in ("ML", "REML"):
        tau = _nma_tau_ml_reml(
            studlab_raw, treat1_raw, treat2_raw, TE_raw, seTE_raw,
            reference_group=str(reference_group or ""),
            method=method_tau.upper(),
            sm=sm,
            arm_data={
                "n1": n1, "n2": n2,
                "event1": event1, "event2": event2,
                "incr2": incr2,
                "mean1": mean1, "mean2": mean2,
                "sd1": sd1, "sd2": sd2,
                "time1": time1, "time2": time2,
            },
        )
        tau2 = tau * tau
    else:
        tau = res_c["tau"]
        tau2 = res_c["tau2"]

    tau2_eff = tau2 if not math.isnan(tau2) else 0.0
    se_adj_r = np.sqrt(_adjusted_variances(
        se_arr, tau2_eff, treat1_pos, treat2_pos, studlab))
    res_r = _nma_ruecker(
        te_arr, se_adj_r, treat1_pos, treat2_pos, narms, studlab, n, level,
    )

    # Direct estimates (per unique design / treatment pair).
    te_direct_c = np.full((n, n), np.nan)
    se_direct_c = np.full((n, n), np.nan)
    te_direct_r = np.full((n, n), np.nan)
    se_direct_r = np.full((n, n), np.nan)
    q_direct = np.full((n, n), np.nan)
    tau2_direct = np.full((n, n), np.nan)
    i2_direct = np.full((n, n), np.nan)

    for i in range(n - 1):
        for j in range(i + 1, n):
            sel = [k for k in range(m)
                   if treat1_pos[k] == i and treat2_pos[k] == j]
            if not sel:
                continue
            te_sub = te_arr[sel]
            se_sub = se_arr[sel]
            # Common-effects direct estimate (tau.direct = 0).
            te_c, se_c = _iv_pool(te_sub, se_sub, 0.0)
            te_direct_c[i, j] = te_c
            te_direct_c[j, i] = -te_c
            se_direct_c[i, j] = se_direct_c[j, i] = se_c
            # Random-effects direct estimate (tau.direct = network tau).
            if not math.isnan(tau2):
                te_r, se_r = _iv_pool(te_sub, se_sub, tau2)
                te_direct_r[i, j] = te_r
                te_direct_r[j, i] = -te_r
                se_direct_r[i, j] = se_direct_r[j, i] = se_r
            # Standard DL metagen on this pair (Q / tau2 / I2).
            q_pair = _cochran_q(te_sub, se_sub)
            tau2_pair = _dl_tau2(te_sub, se_sub)
            k_pair = len(sel)
            i2_pair = max(0.0, (q_pair - (k_pair - 1)) / q_pair) * 100.0 \
                if q_pair > 0 else 0.0
            q_direct[i, j] = q_direct[j, i] = q_pair
            tau2_direct[i, j] = tau2_direct[j, i] = tau2_pair
            i2_direct[i, j] = i2_direct[j, i] = i2_pair

    # Confidence intervals for the direct estimates (netmeta::netleague).
    zz = stats.norm.ppf(1 - (1 - level) / 2)
    lower_direct_c = te_direct_c - zz * se_direct_c
    upper_direct_c = te_direct_c + zz * se_direct_c
    lower_direct_r = te_direct_r - zz * se_direct_r
    upper_direct_r = te_direct_r + zz * se_direct_r

    # Q decomposition: pairwise heterogeneity statistics, computed from the
    # *adjusted* weights (netmeta::nma.ruecker 鈥?Q.matrix uses 1/sqrt(w.pooled)).
    # Q.decomp includes every pair with direct evidence (df = A[i,j] - 1).
    q_decomp = []
    a_mat_c = res_c["a_mat"]
    q_matrix_c = res_c["q_matrix"]
    for i in range(n - 1):
        for j in range(i + 1, n):
            if a_mat_c[i, j] > 0:
                q_val = float(q_matrix_c[i, j])
                df_val = int(a_mat_c[i, j] - 1)
                q_decomp.append({
                    "treat1": trts[i],
                    "treat2": trts[j],
                    "Q": q_val,
                    "df": df_val,
                    "pval.Q": float(stats.chi2.sf(q_val, df_val)) if df_val > 0 else None,
                })

    # Krahn (2013) design-based decomposition (netmeta::decomp.design).
    # When there is more than one design, Q.heterogeneity / Q.inconsistency
    # are overwritten with the "Within designs" / "Between designs" rows of
    # decomp.design 鈥?NOT the internal nma.ruecker pairwise sums.
    if n > 1:
        krahn = _nma_krahn(
            te_arr, se_arr, treat1_pos, treat2_pos, studlab, narms, trts)
        q_het = krahn["Q_het"]
        df_het = krahn["df_het"]
        q_inc = krahn["Q_inc"]
        df_inc = krahn["df_inc"]
    else:
        q_het = res_c["q_het"]
        df_het = float(sum(d["df"] for d in q_decomp))
        q_inc = res_c["q_inc"]
        df_inc = res_c["df_q"] - df_het
    pval_het = stats.chi2.sf(q_het, df_het) if df_het > 0 else float("nan")
    pval_inc = stats.chi2.sf(q_inc, df_inc) if df_inc > 0 else float("nan")

    # A.matrix with NA diagonal (netmeta convention).
    a_out = res_c["a_mat"].astype(float)
    np.fill_diagonal(a_out, np.nan)

    # Direct evidence proportion and indirect (back-calculated) estimates,
    # matching netmeta::netmeasures / netmeta::netsplit:
    #   prop.direct = seTE.network^2 / seTE.direct^2   (NA -> 0)
    #   TE.indirect = (TE.network - P * TE.direct) / (1 - P)
    #   seTE.indirect = sqrt(seTE.network^2 / (1 - P))
    # with P set to NA when P is (numerically) 1 or above (pure direct
    # evidence -> no indirect estimate).
    tol_p = math.sqrt(np.finfo(float).eps)

    def _backcalc(te_pooled, se_pooled, te_direct, se_direct):
        prop = np.full((n, n), np.nan)
        for i in range(n - 1):
            for j in range(i + 1, n):
                sd = se_direct[i, j]
                pc = (se_pooled[i, j] ** 2) / (sd ** 2) \
                    if not math.isnan(sd) else 0.0
                prop[i, j] = prop[j, i] = pc
        te_d = te_direct.copy()
        te_d[np.abs(prop) < tol_p] = 0.0
        p = prop.copy()
        p[np.abs(p - 1.0) < tol_p] = np.nan
        p[p > 1.0] = np.nan
        with np.errstate(divide="ignore", invalid="ignore"):
            te_ind = (te_pooled - p * te_d) / (1.0 - p)
            se_ind = np.sqrt(se_pooled ** 2 / (1.0 - p))
        zz = stats.norm.ppf(1 - (1 - level) / 2)
        lower = te_ind - zz * se_ind
        upper = te_ind + zz * se_ind
        with np.errstate(divide="ignore", invalid="ignore"):
            statistic = te_ind / se_ind
        pval = 2.0 * stats.norm.sf(np.abs(statistic))
        return prop, te_ind, se_ind, lower, upper, statistic, pval

    prop_c, te_ind_c, se_ind_c, lo_c, up_c, st_c, pv_c = _backcalc(
        res_c["te_pooled"], res_c["se_pooled"], te_direct_c, se_direct_c)
    prop_r, te_ind_r, se_ind_r, lo_r, up_r, st_r, pv_r = _backcalc(
        res_r["te_pooled"], res_r["se_pooled"], te_direct_r, se_direct_r)

    # P-scores.
    result = NetmetaResult(
        trts=trts,
        k=len(studies),
        m=m,
        n=n,
        sm=sm,
        level=level,
        method_tau=method_tau,
        TE_common=res_c["te_pooled"],
        seTE_common=res_c["se_pooled"],
        lower_common=res_c["lower"],
        upper_common=res_c["upper"],
        statistic_common=res_c["statistic"],
        pval_common=res_c["pval"],
        TE_random=res_r["te_pooled"],
        seTE_random=res_r["se_pooled"],
        lower_random=res_r["lower"],
        upper_random=res_r["upper"],
        statistic_random=res_r["statistic"],
        pval_random=res_r["pval"],
        TE_direct_common=te_direct_c,
        seTE_direct_common=se_direct_c,
        TE_direct_random=te_direct_r,
        seTE_direct_random=se_direct_r,
        lower_direct_common=lower_direct_c,
        upper_direct_common=upper_direct_c,
        lower_direct_random=lower_direct_r,
        upper_direct_random=upper_direct_r,
        Q_direct=q_direct,
        tau2_direct=tau2_direct,
        I2_direct=i2_direct,
        prop_direct_common=prop_c,
        prop_direct_random=prop_r,
        TE_indirect_common=te_ind_c,
        seTE_indirect_common=se_ind_c,
        lower_indirect_common=lo_c,
        upper_indirect_common=up_c,
        statistic_indirect_common=st_c,
        pval_indirect_common=pv_c,
        TE_indirect_random=te_ind_r,
        seTE_indirect_random=se_ind_r,
        lower_indirect_random=lo_r,
        upper_indirect_random=up_r,
        statistic_indirect_random=st_r,
        pval_indirect_random=pv_r,
        A_matrix=a_out,
        B_matrix=res_c["b"],
        L_matrix_common=res_c["l"],
        Lplus_matrix_common=res_c["lplus"],
        L_matrix_random=res_r["l"],
        Lplus_matrix_random=res_r["lplus"],
        H_matrix_common=res_c["h"],
        H_matrix_random=res_r["h"],
        Cov_common=res_c["cov"],
        Cov_random=res_r["cov"],
        Q=res_c["q"],
        df_Q=res_c["df_q"],
        pval_Q=res_c["pval_q"],
        I2=res_c["i2"],
        lower_I2=res_c["lower_i2"],
        upper_I2=res_c["upper_i2"],
        tau=tau,
        tau2=tau2,
        Q_heterogeneity=q_het,
        df_Q_heterogeneity=df_het,
        pval_Q_heterogeneity=pval_het,
        Q_inconsistency=q_inc,
        df_Q_inconsistency=df_inc,
        pval_Q_inconsistency=pval_inc,
        Q_decomp=q_decomp,
        reference_group=reference_group,
        baseline_reference=baseline_reference,
        TE_raw=TE_raw,
        seTE_raw=seTE_raw,
        treat1_raw=treat1_raw,
        treat2_raw=treat2_raw,
        studlab_raw=studlab_raw,
        narms_raw=narms_raw,
        treat1_pos=treat1_pos,
        treat2_pos=treat2_pos,
        seTE_adj_common=se_adj_c.tolist(),
        seTE_adj_random=se_adj_r.tolist(),
    )

    nr = netrank(result, small_values=small_values)
    result.P_common = nr.P_common
    result.P_random = nr.P_random
    result.pscore_common = nr.pscore_common
    result.pscore_random = nr.pscore_random

    return result


# ---------------------------------------------------------------------------
# netcontrib: network contribution matrix (netmeta::netcontrib)
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# netsplit: split direct and indirect evidence (netmeta::netsplit)
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# League table (netmeta::netleague)
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Network connectivity & distances (netmeta::netconnection / netmeta::netdistance)
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# netpairwise: pairwise (meta-analysis) view of a network (netmeta::netpairwise)
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# netcomb / discomb: additive component network meta-analysis (CNMA)
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# netmetabin: binary network meta-analysis (netmeta::netmetabin)
# ---------------------------------------------------------------------------


# --------------------------------------------------------------------------- #
# netcomplex
# --------------------------------------------------------------------------- #


# --------------------------------------------------------------------------- #
# netmetareg 鈥?network meta-regression
# --------------------------------------------------------------------------- #

@dataclass
class NetmetaregResult:
    """Result of :func:`netmetareg` (``netmeta::netmetareg``)."""

    beta: list[float]
    se: list[float]
    zval: list[float]
    pval: list[float]
    ci_lb: list[float]
    ci_ub: list[float]
    vb: np.ndarray
    tau2: float
    names: list[str]
    coef_na: list[bool]
    QE: float
    QEdf: int
    QM: float
    QMdf: int
    QMp: float
    k: int
    p: int
    trts: list[str]
    covar: np.ndarray
    level: float


def _nmr_fit(X, y, se, study_idx, n_study, tau2, rho=0.5):
    """Fit the WLS/REML network meta-regression for a given ``tau2``."""
    n = y.size
    R = np.zeros((n, n))
    for g in range(n_study):
        idx = np.where(study_idx == g)[0]
        R[np.ix_(idx, idx)] = rho
        R[idx, idx] = 1.0
    M = np.diag(se ** 2) + tau2 * R
    Mi = np.linalg.inv(M)
    XtMi = X.T @ Mi
    XtMiX = XtMi @ X
    try:
        inv = np.linalg.inv(XtMiX)
        logdet_xtmix = math.log(abs(np.linalg.det(XtMiX)))
    except np.linalg.LinAlgError:
        inv = np.linalg.pinv(XtMiX)
        logdet_xtmix = math.log(max(abs(np.linalg.det(inv)), 1e-300)) * -1.0
    beta = inv @ (XtMi @ y)
    resid = y - X @ beta
    rss = float(resid @ Mi @ resid)
    logdet_m = float(np.linalg.slogdet(M)[1])
    return beta, inv, rss, logdet_m, logdet_xtmix


def netmetareg(result, covar, method_tau="REML", reference_group=None,
               level=0.95, rho=0.5, assumption="independent",
               consistency=True, covar_name="covar",
               direction1=None, direction2=None, max_ia=False):
    """``netmeta::netmetareg`` 鈥?network meta-regression.

    Fits the treatment-by-covariate interaction model that netmeta uses::

        theta_k = (mu_t2 - mu_t1) + (beta_t2 - beta_t1) * x_k

    Comparisons are first re-oriented so that every study contributes contrasts
    against its own reference treatment (the network reference when the study
    contains it, otherwise the alphabetically last treatment of that study), the
    design matrix is ``[I(treat) | I(treat) * x]`` over the non-reference
    treatments, and the model is estimated by (restricted) maximum likelihood
    with a block-equicorrelated within-study covariance
    ``V = diag(se^2) + tau2 * CS(rho)`` 鈥?the same structure as metafor's
    ``rma.mv(..., random = ~ comparison | study, rho = 0.5)``.

    Args:
        result: a :class:`NetmetaResult`.
        covar: study-level covariate (one value per raw comparison row, like
            ``netmetareg(net, covar = vector)`` in R).
        method_tau: ``"REML"`` (default), ``"ML"`` or ``"FE"``.
        reference_group: reference treatment; defaults to the network's
            reference (or the first treatment).
        level: confidence level.
        rho: within-study correlation of the random effects (metafor default
            0.5).
        assumption: ``"independent"`` (treatment-specific interactions,
            the default) or ``"common"`` (a single common interaction).
        consistency: ``True`` (default) fits the consistency model; ``False``
            fits the inconsistency model, which adds one covariate interaction
            per comparison (``assumption="independent"``) or a single common
            one (``assumption="common"``).
        covar_name: label used for the interaction terms (R derives it from the
            deparsed ``covar`` expression).
        direction1, direction2: per-comparison direction-of-effect values
            (-1/0/1) for the two arms, only used with ``consistency=False``.
            Without them netmeta falls back to treatment order
            (``mkIval``: the alphabetically first treatment of each study is
            the direction-0 arm).
        max_ia: keep every comparison-level interaction column, including those
            contributed by a single study (R's ``max.ia``).

    Returns:
        :class:`NetmetaregResult`.

    Note:
        Exactly collinear design columns (which ``rma.mv`` drops) are removed
        before fitting; ``coef_na`` marks the dropped positions in the full
        column set.
    """
    from scipy import optimize as _opt

    if not isinstance(result, NetmetaResult):
        raise TypeError("argument 'result' must be a NetmetaResult")
    method_tau = str(method_tau).upper()
    if method_tau not in ("REML", "ML", "FE"):
        raise ValueError("method_tau must be 'REML', 'ML' or 'FE'")
    assumption = str(assumption)
    if assumption not in ("independent", "common"):
        raise ValueError("assumption must be 'independent' or 'common'")

    trts = list(result.trts)
    len(trts)
    {t: i for i, t in enumerate(trts)}

    te = np.asarray(result.TE_raw, dtype=float)
    se = np.asarray(result.seTE_raw, dtype=float)
    t1 = list(result.treat1_raw)
    t2 = list(result.treat2_raw)
    st = list(result.studlab_raw)
    cov = np.asarray(covar, dtype=float).ravel()
    if cov.size != te.size:
        raise ValueError(
            f"length of 'covar' ({cov.size}) does not match the number of "
            f"comparisons ({te.size})"
        )

    ref = reference_group
    if ref is None:
        ref = getattr(result, "reference_group", None) or ""
    if ref not in trts:
        ref = trts[0]

    studies = list(dict.fromkeys(st))
    study_idx = np.array([studies.index(s) for s in st])
    n_study = len(studies)

    # --- direction-of-effect values (ival1/ival2) --------------------------- #
    # consistency model: 0 for the study's reference arm, 1 otherwise, so a
    # comparison is kept when exactly one arm is the reference.
    # inconsistency model: straight from direction1/direction2, or netmeta's
    # mkIval() default (the alphabetically first treatment of the study).
    n_all = te.size
    ival1 = np.zeros(n_all)
    ival2 = np.zeros(n_all)
    if consistency:
        for s in studies:
            idx = [i for i, v in enumerate(st) if v == s]
            trts_i = sorted({t1[i] for i in idx} | {t2[i] for i in idx})
            ref_i = ref if ref in trts_i else trts_i[-1]
            for i in idx:
                ival1[i] = 0.0 if t1[i] == ref_i else 1.0
                ival2[i] = 0.0 if t2[i] == ref_i else 1.0
    else:
        if (direction1 is None) != (direction2 is None):
            raise ValueError(
                "'direction1' and 'direction2' must be supplied together"
            )
        if direction1 is not None:
            d1 = np.asarray(direction1, dtype=float).ravel()
            d2 = np.asarray(direction2, dtype=float).ravel()
            if d1.size != n_all or d2.size != n_all:
                raise ValueError(
                    "'direction1'/'direction2' must have one value per comparison"
                )
            t1_norm = list(getattr(result, "treat1", t1))
            flipped = [t1[i] != t1_norm[i] for i in range(n_all)]
            for i in range(n_all):
                if flipped[i]:
                    ival1[i], ival2[i] = d2[i], d1[i]
                else:
                    ival1[i], ival2[i] = d1[i], d2[i]
        else:
            for s in studies:
                idx = [i for i, v in enumerate(st) if v == s]
                ref_i = sorted({t1[i] for i in idx} | {t2[i] for i in idx})[0]
                for i in idx:
                    ival1[i] = 0.0 if t1[i] == ref_i else 1.0
                    ival2[i] = 0.0 if t2[i] == ref_i else 1.0
            warnings.warn(
                "No directionality parameters specified. "
                "Defaulting to treatment order.",
                stacklevel=2,
            )
        for name, arr in (("Ival1", ival1), ("Ival2", ival2)):
            bad = [v for v in arr if v not in (-1.0, 0.0, 1.0)]
            if bad:
                raise ValueError(
                    f"Error in directionality assignment. {name} only accepts "
                    "values of -1, 0, or 1."
                )
        seen: dict[tuple[str, str], set] = {}
        for i in range(n_all):
            seen.setdefault((st[i], t1[i]), set()).add(ival1[i])
            seen.setdefault((st[i], t2[i]), set()).add(ival2[i])
        clash = sorted({s for (s, _t), v in seen.items() if len(v) > 1})
        if clash:
            raise ValueError(
                "Error in directionality assignment. Multiple directions "
                "defined for identical treatments in the following studies: "
                + ", ".join(clash)
            )

    # --- re-orient each study's comparisons against its own reference ------- #
    rows = []
    for i in range(n_all):
        if ival1[i] == ival2[i]:
            continue
        if ival1[i] == 0.0:
            rows.append((i, t2[i], t1[i], -te[i], ival2[i] - ival1[i]))
        else:
            rows.append((i, t1[i], t2[i], te[i], ival1[i] - ival2[i]))

    row_idx = np.array([r[0] for r in rows])
    a_trt = [r[1] for r in rows]
    b_trt = [r[2] for r in rows]
    y = np.array([r[3] for r in rows], dtype=float)
    ival_r = np.array([r[4] for r in rows], dtype=float)
    se_r = se[row_idx]
    cov_r = cov[row_idx]
    study_r = study_idx[row_idx]
    n_row = y.size

    # --- design matrix ------------------------------------------------------ #
    # metafor's contrmat() codes +1 for treat1 and -1 for treat2; after the
    # re-orientation above treat1 is the study's reference treatment, so the
    # non-reference treatment enters with -1.
    non_ref = [t for t in trts if t != ref]
    X_main = np.zeros((n_row, len(non_ref)))
    for r_i in range(n_row):
        if b_trt[r_i] in non_ref:
            X_main[r_i, non_ref.index(b_trt[r_i])] -= 1.0
        if a_trt[r_i] in non_ref:
            X_main[r_i, non_ref.index(a_trt[r_i])] += 1.0
    if not consistency:
        # Inconsistency model: ~ 0 + trts + <comparison>:Ival:covar
        if assumption == "common":
            X_full = np.hstack([X_main, (ival_r * cov_r)[:, None]])
            names_full = list(non_ref) + [f"Ival:{covar_name}"]
        else:
            comps = [f"{min(a, b)}_vs_{max(a, b)}" for a, b in zip(a_trt, b_trt)]
            counts: dict[str, int] = {}
            for c in comps:
                counts[c] = counts.get(c, 0) + 1
            used = [c for c in sorted(counts) if max_ia or counts[c] >= 2]
            X_int = np.zeros((n_row, len(used)))
            for r_i, c in enumerate(comps):
                if c in used:
                    X_int[r_i, used.index(c)] = ival_r[r_i] * cov_r[r_i]
            X_full = np.hstack([X_main, X_int])
            names_full = list(non_ref) + [
                f".comp_{c}:Ival:{covar_name}" for c in used
            ]
    elif assumption == "common":
        # ~ 0 + trts + nonref:covar, where nonref marks the rows that contain
        # the network reference treatment (the reference-group contrast column
        # of contrmat() is non-zero exactly there).
        nonref = np.array([1.0 if a_trt[r_i] == ref else 0.0
                           for r_i in range(n_row)])
        X_full = np.hstack([X_main, (nonref * cov_r)[:, None]])
        names_full = list(non_ref) + [f"nonref:{covar_name}"]
    else:
        X_int = X_main * cov_r[:, None]
        X_full = np.hstack([X_main, X_int])
        names_full = list(non_ref) + [f"{t}:{covar_name}" for t in non_ref]

    # --- drop exactly collinear columns (rma.mv does the same) -------------- #
    keep = []
    for j in range(X_full.shape[1]):
        cand = keep + [j]
        if np.linalg.matrix_rank(X_full[:, cand], tol=1e-8) == len(cand):
            keep.append(j)
    coef_na = [j not in keep for j in range(X_full.shape[1])]
    X = X_full[:, keep]
    names = [names_full[j] for j in keep]

    # --- estimate tau2 and the coefficients --------------------------------- #
    N, p = X.shape

    def _negll(t2, reml=True):
        b, inv, rss, logdet_m, logdet_xtmix = _nmr_fit(
            X, y, se_r, study_r, n_study, t2, rho
        )
        if not reml:
            return 0.5 * (N * math.log(2.0 * math.pi) + logdet_m + rss)
        logdet_xtx = math.log(max(abs(np.linalg.det(X.T @ X)), 1e-300))
        return 0.5 * ((N - p) * math.log(2.0 * math.pi) - logdet_xtx
                      + logdet_m + logdet_xtmix + rss)

    if method_tau == "FE":
        tau2 = 0.0
    else:
        reml = method_tau == "REML"
        res = _opt.minimize_scalar(
            lambda t2: _negll(t2, reml), bounds=(0.0, 50.0), method="bounded",
            options={"xatol": 1e-12, "maxiter": 1000},
        )
        tau2 = float(max(res.x, 0.0))
        if tau2 < 1e-10:
            tau2 = 0.0

    beta, vb, rss, _, _ = _nmr_fit(X, y, se_r, study_r, n_study, tau2, rho)
    se_b = np.sqrt(np.diag(vb))
    zval = beta / se_b
    crit = stats.norm.isf((1.0 - level) / 2.0)

    # metafor computes QE from the *initial* V (diag(se^2), no tau2), i.e. the
    # fixed-effect weighted residual sum of squares of the fitted model.
    QE = float(_nmr_fit(X, y, se_r, study_r, n_study, 0.0, rho)[2])
    QEdf = N - p
    QM = float(beta @ np.linalg.solve(vb, beta)) if p > 0 else np.nan

    return NetmetaregResult(
        beta=[float(v) for v in beta],
        se=[float(v) for v in se_b],
        zval=[float(v) for v in zval],
        pval=[float(v) for v in 2.0 * stats.norm.sf(np.abs(zval))],
        ci_lb=[float(v) for v in beta - crit * se_b],
        ci_ub=[float(v) for v in beta + crit * se_b],
        vb=vb,
        tau2=tau2,
        names=names,
        coef_na=coef_na,
        QE=QE,
        QEdf=int(QEdf),
        QM=QM,
        QMdf=int(p),
        QMp=float(stats.chi2.sf(QM, p)) if p > 0 else np.nan,
        k=n_row,
        p=int(p),
        trts=trts,
        covar=cov_r,
        level=level,
    )


# --------------------------------------------------------------------------- #
# netmeta::netpath / hatmatrix(method = "Davies", type = "full")
# --------------------------------------------------------------------------- #


# --------------------------------------------------------------------------- #
# netmeta::bySummary / netmatrix / netcomparison / netbind
# --------------------------------------------------------------------------- #


# --------------------------------------------------------------------------- #
# netmeta::crossnma2netmeta / gemtc2netmeta  (foreign-object adapters)
# --------------------------------------------------------------------------- #


# --------------------------------------------------------------------------- #
# netmeta::treats / vikor / subgroup / nettable  + public helpers
# --------------------------------------------------------------------------- #


# --------------------------------------------------------------------------- #
# public aliases for the internal helpers (netmeta::createC / invmat /
# hatmatrix / comps / combinations)
# --------------------------------------------------------------------------- #

def createC(trts, inactive=None, sep_comps="+"):
    """``netmeta::createC`` 鈥?component incidence matrix ``(C, comps, names)``."""
    return _create_c_matrix(trts, inactive, sep_comps)[:3]


def invmat(x):
    """``netmeta::invmat`` 鈥?Moore-Penrose pseudoinverse of a Laplacian."""
    return _invmat(np.asarray(x, dtype=float))


def hatmatrix(x, method="Ruecker", type="full", common=None, random=None):
    """``netmeta::hatmatrix`` 鈥?hat matrix of a network.

    Supported: ``method="Ruecker"`` (the default, as in netmeta 3.7-0 鈥?the hat
    matrix stored on the fitted object), ``method="Davies"`` with ``type`` in
    ``short``/``long``/``full`` (via ``hatmatrix.aggr``), and
    ``method="Krahn"`` (the design-based hat matrix from the decomposition).
    """
    if method == "Ruecker":
        return {"common": np.asarray(x.H_matrix_common, dtype=float),
                "random": np.asarray(x.H_matrix_random, dtype=float),
                "method": method, "type": ""}
    if method == "Davies":
        out = {"method": method, "type": type}
        for model in ("common", "random"):
            res = _hatmatrix_aggr(x, model, type)
            out[model] = res["matrix"]
            out[f"{model}.rows"] = res["rows"]
            out[f"{model}.cols"] = res["cols"]
        return out
    if method == "Krahn":
        nmak = _nma_krahn_full(
            x.TE_raw, x.seTE_raw, x.treat1_raw, x.treat2_raw, x.studlab_raw,
            x.narms_raw, list(x.trts), x.reference_group or "",
            0.0 if type == "common" else float(x.tau),
        )
        if type == "studies":
            xs = nmak["x_obs_studies"]
            vs = nmak["V_studies"]
            vsi = np.linalg.pinv(vs)
            h = (nmak["X_full"] @ np.linalg.pinv(xs.T @ vsi @ xs)
                 @ xs.T @ vsi)
            return {"common": h, "random": h, "method": method,
                    "type": type, "cols": nmak["studlab_studies"]}
        return {"common": nmak["H"], "random": nmak["H"], "method": method,
                "type": type, "cols": nmak["design_col"]}
    raise ValueError("method must be 'Ruecker', 'Krahn' or 'Davies'")


def comps(x, trts=None, sep_comps="+"):
    """``netmeta::comps`` 鈥?split treatment names into components.

    Args:
        x: a treatment label, or a sequence of them.
        trts: optional reference list of *components*.  When given, every
            component found in ``x`` must appear there (netmeta's ``setref``
            check) and the components are returned in ``trts`` order, which is
            what ``netmeta:::comps`` does with its ``trts`` argument.
        sep_comps: component separator.

    Returns:
        The component list for a single label, or a list of component lists.
    """
    def _norm(parts):
        if trts is None:
            return parts
        ref = [str(v) for v in trts]
        for p in parts:
            if p not in ref:
                raise ValueError(
                    f"component {p!r} is not in 'trts' ({ref})")
        return [c for c in ref if c in parts]

    if isinstance(x, str):
        return _norm(_compsplit(x, sep_comps))
    out = []
    for t in np.asarray(x, dtype=object).ravel():
        out.append(_norm(_compsplit(str(t), sep_comps)))
    return out


def combinations(x, sep_comps="+"):
    """``netmeta::combinations`` 鈥?all treatment combinations of ``x``."""
    if isinstance(x, str):
        return _compsplit(x, sep_comps)
    vals = [str(v) for v in np.asarray(x, dtype=object).ravel()]
    return sorted(set(vals),
                  key=lambda s: (len(_compsplit(s, sep_comps)), s))
