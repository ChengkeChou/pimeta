"""Network meta-analysis, replicating R's ``netmeta`` (Rücker 2012).

Implements the frequentist graph-theoretical approach of Rücker (2012) and the
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
import sys
import warnings
from dataclasses import dataclass, field

import numpy as np
from scipy import stats

Z_975 = stats.norm.ppf(0.975)

# netmeta's is_zero() uses .Machine$double.eps^0.5 ~ 1.49e-8.
_ZERO = 1e-8


# ---------------------------------------------------------------------------
# Graph-theory primitives (transcribed from netmeta::createB / invmat)
# ---------------------------------------------------------------------------

def _create_b(treat1_pos, treat2_pos, n):
    """Edge-vertex incidence matrix B (m x n).

    ``B[i, treat1_pos[i]] = 1`` and ``B[i, treat2_pos[i]] = -1``, all other
    entries 0. Mirrors ``netmeta::createB``.
    """
    m = len(treat1_pos)
    b = np.zeros((m, n), dtype=float)
    for i in range(m):
        b[i, treat1_pos[i]] = 1.0
        b[i, treat2_pos[i]] = -1.0
    return b


def _create_b_full(n):
    """Full edge-vertex incidence matrix for all n(n-1)/2 unordered pairs."""
    pairs = [(i, j) for i in range(n) for j in range(i + 1, n)]
    t1 = [p[0] for p in pairs]
    t2 = [p[1] for p in pairs]
    return _create_b(t1, t2, n)


def _invmat(x):
    """Moore-Penrose pseudoinverse of a (weighted) Laplacian.

    ``solve(X - J/n) + J/n`` with J the all-ones matrix — exact transcription
    of ``netmeta::invmat`` (valid for connected networks, rank n-1).
    """
    n = x.shape[0]
    j = np.ones((n, n), dtype=float)
    return np.linalg.solve(x - j / n, np.eye(n)) + j / n


def _multiarm(r):
    """Adjust the comparison variances of a single multi-arm study.

    Exact transcription of ``netmeta::multiarm`` (Rücker 2012, appendix).
    ``r`` holds the k(k-1)/2 comparison variances (se^2 + tau^2) ordered
    (1,2), (1,3), ..., (1,k), (2,3), ..., (k-1,k) by sorted treatment. The
    returned vector gives the adjusted variances in the same order, so the
    study's internal comparisons are down-weighted consistently.
    """
    r = np.asarray(r, dtype=float)
    m = len(r)
    k = int(round((1.0 + math.sqrt(8.0 * m + 1.0)) / 2.0))

    b = _create_b_full(k)                     # complete incidence matrix
    rr = np.diag(r)

    # R[i,j] = resistance (variance) between arms i,j of the complete graph.
    tmp = b.T @ rr @ b
    rmat = np.diag(np.diag(tmp)) - tmp

    # Lt = -0.5 * (B'B) R (B'B) / k^2 ; L = invmat(Lt).
    bb = b.T @ b
    lt = -0.5 * bb @ rmat @ bb / float(k * k)
    l = _invmat(lt)

    # W = diag(diag(L)) - L ; zero out tiny negative weights.
    w = np.diag(np.diag(l)) - l
    lower = np.tril_indices(k, -1)
    denom = float(np.sum(np.abs(w[lower])))
    if denom > 0:
        small_neg = (w < 0.0) & (np.abs(w) / denom < 0.001)
        w = np.where(small_neg, 0.0, w)

    # V = 1 / W element-wise (R's ``V <- 1 / W`` is element-wise).
    with np.errstate(divide="ignore", invalid="ignore"):
        vmat = 1.0 / w

    v = np.zeros(m, dtype=float)
    e = 0
    for i in range(k - 1):
        for j in range(i + 1, k):
            v[e] = vmat[i, j]
            e += 1
    return v


def _adjusted_variances(se_arr, tau2, treat1_pos, treat2_pos, studlab):
    """Per-comparison adjusted variances (se^2), replicating netmeta::prepare.

    Two-arm studies contribute ``se^2 + tau2`` unchanged. Multi-arm studies
    have their comparison variances reweighted by ``_multiarm`` (Rücker 2012)
    so the internally-correlated comparisons are down-weighted consistently.
    """
    se2 = se_arr ** 2 + tau2
    adj = se2.copy()

    by_study = {}
    for i, s in enumerate(studlab):
        by_study.setdefault(s, []).append(i)

    for s, inds in by_study.items():
        if len(inds) <= 1:
            continue
        # Order this study's comparisons as (1,2),(1,3),...,(k-1,k).
        order = sorted(inds, key=lambda i: (treat1_pos[i], treat2_pos[i]))
        v = _multiarm(se2[order])
        for i, vv in zip(order, v):
            adj[i] = vv

    return adj


def _connected(adj):
    """Return True if the 0/1 adjacency graph is connected (BFS)."""
    n = adj.shape[0]
    seen = [False] * n
    seen[0] = True
    stack = [0]
    while stack:
        u = stack.pop()
        for v in range(n):
            if not seen[v] and adj[u, v]:
                seen[v] = True
                stack.append(v)
    return all(seen)


def _isquared(q, df, level):
    """I^2 with confidence limits (Higgins & Thompson 2002), via calcH.

    ``H = sqrt(Q/df)``, ``I^2 = (H^2 - 1) / H^2``; CI back-computed from a
    normal CI for ``log(max(H, 1))``. Exact transcription of
    ``meta::isquared`` / ``meta::calcH``.
    """
    if df <= 0 or math.isnan(q):
        return float("nan"), float("nan"), float("nan")

    k = df + 1
    if k > 1:
        h = math.sqrt(q / (k - 1))
    else:
        h = float("nan")

    if q > k:
        if k >= 2:
            selog_h = (0.5 * (math.log(q) - math.log(k - 1))
                       / (math.sqrt(2 * q) - math.sqrt(2 * k - 3)))
        else:
            selog_h = float("nan")
    else:
        if k > 2:
            selog_h = math.sqrt(1.0 / (2.0 * (k - 2))
                                * (1.0 - 1.0 / (3.0 * (k - 2) ** 2)))
        else:
            selog_h = float("nan")

    z = stats.norm.ppf(1 - (1 - level) / 2)
    log_h = math.log(max(h, 1.0))

    def _f(x):
        return (x ** 2 - 1.0) / x ** 2

    i2 = _f(math.exp(log_h))
    if math.isnan(selog_h):
        lower = upper = float("nan")
    else:
        lower = _f(max(math.exp(log_h - z * selog_h), 1.0))
        upper = _f(max(math.exp(log_h + z * selog_h), 1.0))
    return i2, lower, upper


def _cochran_q(y, se):
    """Cochran's Q for fixed (inverse-variance) weights 1/se^2."""
    y = np.asarray(y, dtype=float)
    se = np.asarray(se, dtype=float)
    w = 1.0 / se ** 2
    mu = float(np.sum(w * y) / np.sum(w))
    return float(np.sum(w * (y - mu) ** 2))


def _dl_tau2(y, se):
    """DerSimonian-Laird tau^2 (pairwise metagen)."""
    y = np.asarray(y, dtype=float)
    se = np.asarray(se, dtype=float)
    k = len(y)
    if k <= 1:
        return float("nan")
    w = 1.0 / se ** 2
    sw = float(np.sum(w))
    q = _cochran_q(y, se)
    c = sw - float(np.sum(w ** 2)) / sw
    return max(0.0, (q - (k - 1)) / c)


def _iv_pool(y, se, tau2):
    """Inverse-variance pooled estimate with a *fixed* tau^2 added to se^2."""
    y = np.asarray(y, dtype=float)
    se = np.asarray(se, dtype=float)
    w = 1.0 / (se ** 2 + tau2)
    sw = float(np.sum(w))
    mu = float(np.sum(w * y) / sw)
    se_mu = math.sqrt(1.0 / sw)
    return mu, se_mu


# ---------------------------------------------------------------------------
# Core Rücker (2012) algorithm
# ---------------------------------------------------------------------------

def _nma_ruecker(te, se_adj, treat1_pos, treat2_pos, narms, studlab, n, level):
    """One pass of the Rücker (2012) graph-theoretical NMA.

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

    # Diagonal: netmeta leaves TE on the diagonal NA (the ``all`` matrix is
    # never filled on its diagonal); seTE diagonal is sqrt(0) = 0.
    np.fill_diagonal(te_pooled, np.nan)
    np.fill_diagonal(lower, np.nan)
    np.fill_diagonal(upper, np.nan)
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

def _pair_index(i, j, n):
    """Flattened 0-based row index of pair (i, j) with i < j in ``X.full``.

    Rows of the design matrix follow ``trts.poss``:
    t0:t1, t0:t2, ..., t0:t(n-1), t1:t2, ..., t(n-2):t(n-1).
    """
    idx = 0
    for a in range(i):
        idx += n - 1 - a
    return idx + (j - i - 1)


def _fx(n):
    """Full design matrix ``fX(n)`` from ``netmeta::nma.krahn``.

    ``possK = n*(n-1)/2`` rows (one per possible contrast) by ``n-1`` columns
    (treatment effects relative to the first treatment). Exact 0-based
    transcription of the R ``fX`` helper.
    """
    poss_k = n * (n - 1) // 2
    x = np.zeros((poss_k, n - 1), dtype=float)
    x[0:(n - 1), 0:(n - 1)] = np.diag(np.full(n - 1, -1.0))
    x[poss_k - 1, (n - 3):(n - 1)] = [1.0, -1.0]
    if poss_k - (n - 1) > 1:
        l = n                       # 1-based row start
        j = n - 2
        u = n + j - 1               # 1-based row end
        for k in range(1, n - 2):   # k = 1 .. n-3
            block = np.hstack([
                np.ones((n - k - 1, 1)),
                np.diag(np.full(n - k - 1, -1.0)),
            ])
            x[(l - 1):u, (k - 1):(n - 1)] = block
            j -= 1
            l = u + 1
            u = l + j - 1
    return x


def _block_diag(blocks):
    """Block-diagonal matrix from a list of scalars/vectors/matrices."""
    mats = []
    for b in blocks:
        arr = np.asarray(b, dtype=float)
        if arr.ndim == 0:
            arr = arr.reshape(1, 1)
        elif arr.ndim == 1:
            arr = arr.reshape(1, -1)
        mats.append(arr)
    sizes = [m.shape[0] for m in mats]
    total = sum(sizes)
    out = np.zeros((total, total), dtype=float)
    r = 0
    for m in mats:
        c = m.shape[0]
        out[r:r + c, r:r + c] = m
        r += c
    return out


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
    idx_new = {t: i for i, t in enumerate(trts)}

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

    network = {"TE": te_net_full, "seTE": se_net_full,
               "names": trts_poss}
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


def decomp_design(result: NetmetaResult, tau_preset=None) -> DecompDesignResult:
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
    """P-scores (Rücker & Schwarzer 2015), exact transcription of netrank.R.

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
    """netmeta:::calcV — within-study covariance of one (multi-arm) study.

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

    R reduces the network to one *star* per study — every comparison that
    touches the study's reference treatment — expresses each as
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

    # 1. R's order(studlab, treat1, treat2) — lexicographic, stable.
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
) -> NetmetaResult:
    """Frequentist graph-theoretical network meta-analysis (Rücker 2012).

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
    TE = list(TE)
    seTE = list(seTE)
    treat1 = list(treat1)
    treat2 = list(treat2)
    # Keep the raw (un-normalized) directions for decomp.design / nma.krahn.
    TE_raw = list(TE)
    seTE_raw = list(seTE)
    treat1_raw = list(treat1)
    treat2_raw = list(treat2)
    m = len(TE)
    if not (len(seTE) == len(treat1) == len(treat2) == m):
        raise ValueError("TE, seTE, treat1, treat2 must have equal length")

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
    study_to_trts = {s: set() for s in studies}
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
                         "pseudoinverse (Rücker 2012 requires a connected network)")

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
    # *adjusted* weights (netmeta::nma.ruecker — Q.matrix uses 1/sqrt(w.pooled)).
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
    # decomp.design — NOT the internal nma.ruecker pairwise sums.
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

@dataclass
class NetcontribResult:
    """Contribution matrix of a network meta-analysis (netmeta::netcontrib).

    ``common[r, c]`` (and ``random[r, c]``) is the proportion of the pooled
    estimate for comparison ``comparisons[r]`` (a treatment pair, row) that is
    contributed by the direct comparison ``directs[c]`` (column). Rows sum to
    1 (within numerical precision). Uses the default "shortestpath"
    (Papakonstantinou et al. 2018) flow-decomposition method.
    """

    trts: list[str]
    comparisons: list[str]
    directs: list[str]
    common: np.ndarray
    random: np.ndarray
    method: str = "shortestpath"

    def to_dict(self) -> dict:
        return {
            "trts": self.trts,
            "comparisons": self.comparisons,
            "directs": self.directs,
            "common": self.common.tolist(),
            "random": self.random.tolist(),
            "method": self.method,
        }


def _shortest_path_edges(adj, source, target):
    """Unweighted shortest path (BFS) from source to target.

    ``adj`` maps a node to a list of ``(neighbor, edge_index)`` tuples. Returns
    the ordered list of edge indices on one shortest path (empty if no path).
    """
    from collections import deque

    if source == target:
        return []
    parent = {source: (None, None)}
    queue = deque([source])
    while queue:
        u = queue.popleft()
        if u == target:
            break
        for v, eidx in adj.get(u, ()):
            if v not in parent:
                parent[v] = (u, eidx)
                queue.append(v)
    if target not in parent:
        return []
    path = []
    cur = target
    while parent[cur][0] is not None:
        prev, eidx = parent[cur]
        path.append(eidx)
        cur = prev
    path.reverse()
    return path


def _hatmatrix_aggr_long(treat1_pos, treat2_pos, b_full_m, se_adj, n):
    """Aggregated hat matrix in "long" form (netmeta::hatmatrix.aggr).

    Returns ``(H_long, pairs)`` where ``H_long`` is (c x e): rows = all
    n(n-1)/2 treatment pairs (upper-triangle order), columns = the e unique
    direct comparisons (in ascending node-pair order). ``pairs`` gives the
    direct-comparison node pairs for each column.
    """
    pairs = sorted(set(zip(treat1_pos, treat2_pos)))
    e = len(pairs)
    b_aggr = np.zeros((e, n), dtype=float)
    for r, (i, j) in enumerate(pairs):
        b_aggr[r, i] = 1.0
        b_aggr[r, j] = -1.0

    m = b_full_m.shape[0]
    w = np.zeros((e, e), dtype=float)
    for i in range(e):
        idx_a = idx_b = None
        for j in range(n):
            if b_aggr[i, j] == 1.0:
                idx_a = j
            elif b_aggr[i, j] == -1.0:
                idx_b = j
        wab = 0.0
        for k in range(m):
            if b_full_m[k, idx_a] == 1.0 and b_full_m[k, idx_b] == -1.0:
                wab += 1.0 / (se_adj[k] ** 2)
        w[i, i] = wab

    l = b_aggr.T @ w @ b_aggr
    lplus = _invmat(l)
    lplus[np.abs(lplus) < _ZERO] = 0.0

    b_full = _create_b_full(n)
    h_long = b_full @ lplus @ b_aggr.T @ w
    return h_long, pairs


def _contribution_matrix(treat1_pos, treat2_pos, b_full_m, se_adj, n):
    """Shortest-path flow decomposition (netmeta::contribution.matrix.tpapak)."""
    h_long, pairs = _hatmatrix_aggr_long(treat1_pos, treat2_pos, b_full_m, se_adj, n)
    e = h_long.shape[1]
    c = h_long.shape[0]
    weights = np.zeros((c, e), dtype=float)

    all_pairs = [(i, j) for i in range(n) for j in range(i + 1, n)]
    for row, (source, target) in enumerate(all_pairs):
        h_row = h_long[row, :]
        flows = [abs(float(h_row[k])) for k in range(e)]
        # Directed edge list in column order; H > 0 keeps (di, dj), otherwise
        # reversed. Zero-flow edges stay in the graph (igraph includes them).
        edges = []
        for k in range(e):
            di, dj = pairs[k]
            u, v = (di, dj) if h_row[k] > 0.0 else (dj, di)
            edges.append((u, v))
        # igraph assigns vertex ids in first-appearance order and its BFS
        # iterates neighbours by ascending vertex id; replicate both here.
        vid = {}
        for u, v in edges:
            if u not in vid:
                vid[u] = len(vid)
            if v not in vid:
                vid[v] = len(vid)
        active = [True] * e
        wrow = np.zeros(e, dtype=float)
        while True:
            adj = {}
            for k in range(e):
                if not active[k]:
                    continue
                u, v = edges[k]
                adj.setdefault(u, []).append((v, k))
            for u in adj:
                adj[u].sort(key=lambda item: vid[item[0]])
            path = _shortest_path_edges(adj, source, target)
            if not path:
                break
            pl = float(len(path))
            flow = min(flows[k] for k in path)
            for k in path:
                flows[k] -= flow
                wrow[k] += flow / pl
                if flows[k] == 0.0:
                    active[k] = False
        wrow[np.abs(wrow) < _ZERO] = 0.0
        weights[row, :] = wrow
    return weights, pairs


def _simple_paths_randomwalk(adj: np.ndarray, source: int, target: int) -> list[list[int]]:
    """Every simple directed path ``source -> target`` in ``adj`` (igraph equivalent)."""
    n = adj.shape[0]
    out: list[list[int]] = []
    stack: list[tuple[int, list[int]]] = [(source, [source])]
    while stack:
        node, path = stack.pop()
        if node == target:
            out.append(path)
            continue
        for nxt in range(n):
            if adj[node, nxt] != 0.0 and nxt not in path:
                stack.append((nxt, path + [nxt]))
    return out


def _contribution_matrix_randomwalk(
    result, model: str
) -> tuple[np.ndarray, list[tuple[int, int]]]:
    """Davies' random-walk contribution matrix (netmeta::contribution.matrix.davies).

    For each treatment pair the aggregated hat matrix supplies a directed
    "flow" matrix ``Q``; normalising its rows gives a transition matrix that is
    made absorbing at the target. Every simple path from source to target then
    contributes ``(1/L) * prod(P)`` where ``L`` is the path length.
    """
    n = int(result.n)
    treat1_pos = [int(v) for v in result.treat1_pos]
    treat2_pos = [int(v) for v in result.treat2_pos]
    b_full_m = np.asarray(result.B_matrix, dtype=float)
    se_adj = np.asarray(
        result.seTE_adj_random if model == "random" else result.seTE_adj_common,
        dtype=float,
    )
    # hatmatrix.aggr(type="full") equals the long form whenever every treatment
    # pair has direct evidence, which is the only case R's randomwalk handles
    # without first collapsing duplicate edges.
    h_full, _hpairs = _hatmatrix_aggr_long(
        treat1_pos, treat2_pos, b_full_m, se_adj, n
    )
    if h_full.shape[1] != n * (n - 1) // 2:
        raise NotImplementedError(
            "netcontrib(method='randomwalk') on an incomplete network is not "
            "supported yet; every treatment pair must have direct evidence"
        )
    comps = n * (n - 1) // 2
    ep = 2.220446049250313e-16 * 1000.0  # R's is_zero(x, n = 1000)

    weights = np.zeros((comps, comps), dtype=float)
    r = 0
    for t1 in range(n - 1):
        for t2 in range(t1 + 1, n):
            q = np.zeros((n, n), dtype=float)
            idx = 0
            for i in range(n - 1):
                for j in range(i + 1, n):
                    q[i, j] = h_full[r, idx]
                    idx += 1
            for i in range(n):
                for j in range(n):
                    if i != j:
                        q[j, i] = -q[i, j]
            for i in range(n):
                for j in range(n):
                    if i != j and q[i, j] <= 0.0:
                        q[i, j] = 0.0

            # P = ginv(diag(rowSums(Q))) %*% Q, absorbing at t2
            p = np.zeros((n, n), dtype=float)
            row_sums = q.sum(axis=1)
            for i in range(n):
                if row_sums[i] != 0.0:
                    p[i, :] = q[i, :] / row_sums[i]
            p[t2, :] = 0.0
            p[t2, t2] = 1.0
            p[np.abs(p) < ep] = 0.0

            rmat = np.zeros((n, n), dtype=float)
            for path in _simple_paths_randomwalk(p, t1, t2):
                length = len(path) - 1
                if length <= 0:
                    continue
                prob = 1.0 / length
                for i in range(length):
                    prob *= p[path[i], path[i + 1]]
                for i in range(length):
                    rmat[path[i], path[i + 1]] += prob
                for i in range(n - 1):
                    for j in range(i + 1, n):
                        v = max(rmat[i, j], rmat[j, i])
                        rmat[i, j] = v
                        rmat[j, i] = v

            # R fills uppertri() in column-major order but labels the columns
            # with hatmatrix.aggr's ascending pair order, so write the entries
            # straight into ascending order here.
            for lex_pos, (i, j) in enumerate(
                (a, b) for a in range(n) for b in range(a + 1, n)
            ):
                weights[r, lex_pos] = rmat[i, j]
            r += 1

    weights[np.abs(weights) < ep] = 0.0
    keep = weights.sum(axis=0) > 0.0
    # R labels the columns with hatmatrix.aggr's row names (ascending pair
    # order) while the values were filled in upper-triangle column-major order.
    pairs_all = [(i, j) for i in range(n) for j in range(i + 1, n)]
    kept_pairs = [pair for pair, k in zip(pairs_all, keep) if k]
    return weights[:, keep], kept_pairs


def netcontrib(x: NetmetaResult, method: str = "shortestpath") -> NetcontribResult:
    """Contribution matrix of a network meta-analysis.

    Transcribes ``netmeta::netcontrib`` (default "shortestpath" method). Each
    row of the returned ``common`` / ``random`` matrices corresponds to a
    treatment pair; each column to a direct comparison (edge). Rows sum to 1.
    """
    if method not in ("shortestpath", "randomwalk"):
        raise ValueError("method must be 'shortestpath' or 'randomwalk'")

    n = x.n
    trts = list(x.trts)
    treat1_pos = [int(v) for v in x.treat1_pos]
    treat2_pos = [int(v) for v in x.treat2_pos]
    b_full_m = np.asarray(x.B_matrix, dtype=float)
    se_common = np.asarray(x.seTE_adj_common, dtype=float)
    se_random = np.asarray(x.seTE_adj_random, dtype=float)

    if method == "randomwalk":
        common, pairs = _contribution_matrix_randomwalk(x, "common")
        random, _ = _contribution_matrix_randomwalk(x, "random")
    else:
        common, pairs = _contribution_matrix(
            treat1_pos, treat2_pos, b_full_m, se_common, n)
        random, _ = _contribution_matrix(
            treat1_pos, treat2_pos, b_full_m, se_random, n)

    sep = ":"
    comparisons = [f"{trts[i]}{sep}{trts[j]}" for i in range(n) for j in range(i + 1, n)]
    directs = [f"{trts[i]}{sep}{trts[j]}" for (i, j) in pairs]

    return NetcontribResult(
        trts=trts,
        comparisons=comparisons,
        directs=directs,
        common=common,
        random=random,
        method=method,
    )


# ---------------------------------------------------------------------------
# netsplit: split direct and indirect evidence (netmeta::netsplit)
# ---------------------------------------------------------------------------

@dataclass
class NetsplitResult:
    """Split of direct and indirect evidence per comparison.

    ``comparison`` lists every treatment pair in the upper-triangle order
    (``trts[i] : trts[j]`` with ``i < j``). ``k`` is the number of studies
    providing direct evidence (``A.matrix``); ``prop_common`` / ``prop_random``
    are the direct-evidence proportions. The ``common`` / ``direct_common`` /
    ``indirect_common`` / ``compare_common`` dicts (and their ``*_random``
    counterparts) hold aligned lists keyed by ``"TE"``, ``"seTE"``, ``"lower"``,
    ``"upper"``, ``"statistic"``, ``"p"`` (``compare_*`` also has ``"z"``).
    Missing values (R ``NA``) are ``None``.
    """

    comparison: list[str]
    k: list[int]
    prop_common: list[float]
    prop_random: list[float]
    method: str = "Back-calculation"

    common: dict = field(default_factory=dict)
    direct_common: dict = field(default_factory=dict)
    indirect_common: dict = field(default_factory=dict)
    compare_common: dict = field(default_factory=dict)

    random: dict = field(default_factory=dict)
    direct_random: dict = field(default_factory=dict)
    indirect_random: dict = field(default_factory=dict)
    compare_random: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "comparison": self.comparison,
            "k": self.k,
            "prop_common": self.prop_common,
            "prop_random": self.prop_random,
            "method": self.method,
            "common": self.common,
            "direct_common": self.direct_common,
            "indirect_common": self.indirect_common,
            "compare_common": self.compare_common,
            "random": self.random,
            "direct_random": self.direct_random,
            "indirect_random": self.indirect_random,
            "compare_random": self.compare_random,
        }


def _split_pooled(
    x: NetmetaResult, pooled: str, tol_direct: float, flip: list[bool]
) -> tuple:
    """Build nma / direct / indirect / compare dicts for one model.

    ``flip`` is aligned with the comparison list; a ``True`` entry means the
    comparison was expressed as ``trts[j] : trts[i]`` (reference placed on the
    right), so the effect, confidence limits and statistic are negated.
    """
    if pooled == "common":
        TE, se = x.TE_common, x.seTE_common
        lo, up = x.lower_common, x.upper_common
        st, pv = x.statistic_common, x.pval_common
        TE_d, se_d = x.TE_direct_common, x.seTE_direct_common
        TE_i, se_i = x.TE_indirect_common, x.seTE_indirect_common
        prop = x.prop_direct_common
    else:
        TE, se = x.TE_random, x.seTE_random
        lo, up = x.lower_random, x.upper_random
        st, pv = x.statistic_random, x.pval_random
        TE_d, se_d = x.TE_direct_random, x.seTE_direct_random
        TE_i, se_i = x.TE_indirect_random, x.seTE_indirect_random
        prop = x.prop_direct_random

    n = x.n
    a_mat = x.A_matrix
    z = stats.norm.ppf(1 - (1 - x.level) / 2)

    nma = {k: [] for k in ("TE", "seTE", "lower", "upper", "statistic", "p")}
    direct = {k: [] for k in ("TE", "seTE", "lower", "upper", "statistic", "p")}
    indirect = {k: [] for k in ("TE", "seTE", "lower", "upper", "statistic", "p")}
    compare = {k: [] for k in ("TE", "seTE", "lower", "upper", "statistic", "p", "z")}

    def _ci(te, se_):
        if te is None or se_ is None:
            return None, None, None, None
        lower = te - z * se_
        upper = te + z * se_
        stat = te / se_ if se_ != 0 else float("nan")
        p = 2.0 * stats.norm.sf(abs(stat))
        return lower, upper, stat, p

    kk = 0
    for i in range(n - 1):
        for j in range(i + 1, n):
            sgn = -1.0 if flip[kk] else 1.0
            p_ij = float(prop[i, j])
            # Network estimate.
            nma["TE"].append(sgn * float(TE[i, j]))
            nma["seTE"].append(float(se[i, j]))
            nma["lower"].append(sgn * float(lo[i, j]))
            nma["upper"].append(sgn * float(up[i, j]))
            if sgn < 0:
                nma["lower"][-1], nma["upper"][-1] = (
                    nma["upper"][-1], nma["lower"][-1],
                )
            nma["statistic"].append(sgn * float(st[i, j]))
            nma["p"].append(float(pv[i, j]))

            # Direct estimate (NA when there is (almost) no direct evidence).
            if abs(p_ij) < tol_direct:
                d_te = d_se = None
            else:
                d_te = sgn * float(TE_d[i, j])
                d_se = float(se_d[i, j])
            d_lo, d_up, d_st, d_pv = _ci(d_te, d_se)
            direct["TE"].append(d_te)
            direct["seTE"].append(d_se)
            direct["lower"].append(d_lo)
            direct["upper"].append(d_up)
            direct["statistic"].append(d_st)
            direct["p"].append(d_pv)

            # Indirect estimate (NA for pure direct evidence; equals the
            # network estimate when there is no direct evidence at all).
            if abs(p_ij - 1.0) < tol_direct:
                i_te = i_se = None
            else:
                i_te = sgn * float(TE_i[i, j])
                i_se = float(se_i[i, j])
                if a_mat[i, j] == 0:
                    i_te = sgn * float(TE[i, j])
                    i_se = float(se[i, j])
            i_lo, i_up, i_st, i_pv = _ci(i_te, i_se)
            indirect["TE"].append(i_te)
            indirect["seTE"].append(i_se)
            indirect["lower"].append(i_lo)
            indirect["upper"].append(i_up)
            indirect["statistic"].append(i_st)
            indirect["p"].append(i_pv)

            # Test of direct vs indirect (difference).
            if d_te is None or i_te is None:
                c_te = c_se = None
            else:
                c_te = d_te - i_te
                c_se = math.sqrt(d_se ** 2 + i_se ** 2)
            c_lo, c_up, c_st, c_pv = _ci(c_te, c_se)
            compare["TE"].append(c_te)
            compare["seTE"].append(c_se)
            compare["lower"].append(c_lo)
            compare["upper"].append(c_up)
            compare["statistic"].append(c_st)
            compare["p"].append(c_pv)
            compare["z"].append(c_st)

            kk += 1

    return nma, direct, indirect, compare


def netsplit(
    x: NetmetaResult,
    method: str = "Back-calculation",
    common: bool = True,
    random: bool = True,
    tol_direct: float = 5e-4,
) -> NetsplitResult:
    """Split direct and indirect evidence in a network meta-analysis.

    Replicates ``netmeta::netsplit`` with ``method = "Back-calculation"``
    (the default for non-binary ``netmeta`` objects). The indirect estimate is
    back-calculated from the network and direct estimates using the direct
    evidence proportion, already stored in ``x`` by :func:`netmeta`.

    Args:
        x: a :class:`NetmetaResult` from :func:`netmeta`.
        method: splitting method (only ``"Back-calculation"``).
        common: include common-effects results.
        random: include random-effects results.
        tol_direct: maximum deviation of the direct evidence proportion from 0
            or 1 to classify a comparison as indirect-only or direct-only.
    """
    if method != "Back-calculation":
        raise ValueError(
            f"method={method!r} not supported; only 'Back-calculation' "
            "is implemented"
        )

    n = x.n
    comparison = []
    flip = []
    k = []
    prop_c = []
    prop_r = []
    a_mat = x.A_matrix
    ref = x.reference_group
    base_ref = x.baseline_reference
    for i in range(n - 1):
        for j in range(i + 1, n):
            t1 = x.trts[i]
            t2 = x.trts[j]
            # netmeta::comptrts: with a reference group as baseline, comparisons
            # are expressed as "other treatment : reference" (reference on the
            # right), so any comparison with the reference first is reversed.
            fl = bool(base_ref and ref != "" and t1 == ref)
            if fl:
                comparison.append(f"{t2}:{t1}")
            else:
                comparison.append(f"{t1}:{t2}")
            flip.append(fl)
            k.append(int(a_mat[i, j]))
            prop_c.append(float(x.prop_direct_common[i, j]))
            prop_r.append(float(x.prop_direct_random[i, j]))

    res = NetsplitResult(
        comparison=comparison,
        k=k,
        prop_common=prop_c,
        prop_random=prop_r,
        method=method,
    )

    if common:
        nma, direct, indirect, compare = _split_pooled(
            x, "common", tol_direct, flip
        )
        res.common = nma
        res.direct_common = direct
        res.indirect_common = indirect
        res.compare_common = compare
    if random:
        nma, direct, indirect, compare = _split_pooled(
            x, "random", tol_direct, flip
        )
        res.random = nma
        res.direct_random = direct
        res.indirect_random = indirect
        res.compare_random = compare

    return res


# ---------------------------------------------------------------------------
# League table (netmeta::netleague)
# ---------------------------------------------------------------------------

def _format_right(mat: np.ndarray, digits: int) -> dict:
    """Replicate R ``format(x, justify = "right")`` on a numeric matrix.

    Each non-missing value is rendered with a fixed number of decimals
    (``f"{v:.{digits}f}"``), then right-justified to the width of the widest
    entry (including R's ``"NA"``, which occupies 3 columns) so that positive
    numbers are padded with leading spaces to line up the minus signs.
    Missing values are omitted from the result (R ``formatCI`` maps them to
    ``""``).
    """
    n = mat.shape[0]
    cells = {}
    width = 3  # "NA"
    for i in range(n):
        for j in range(n):
            v = mat[i, j]
            if np.isnan(v):
                continue
            s = f"{v:.{digits}f}"
            cells[(i, j)] = s
            width = max(width, len(s))
    return {k: s.rjust(width) for k, s in cells.items()}


def _league_cells(
    te_nma: np.ndarray,
    lo_nma: np.ndarray,
    up_nma: np.ndarray,
    te_dir: np.ndarray,
    lo_dir: np.ndarray,
    up_dir: np.ndarray,
    trts: list[str],
    digits: int,
    text_na: str,
    ci: bool,
) -> list[list[str]]:
    """Build a league-table matrix (list of string rows).

    Lower triangle (``r > c``) holds the network estimates (column vs row,
    i.e. ``theta[c] - theta[r]``); upper triangle (``r < c``) holds the direct
    estimates (row vs column, ``theta_dir[r] - theta_dir[c]``). The diagonal
    holds the treatment names. Matches ``netmeta::netleague`` orientation.
    """
    n = len(trts)

    def _fn(mat: np.ndarray) -> list[list[str]]:
        return [
            [
                text_na if np.isnan(mat[i, j]) else f"{mat[i, j]:.{digits}f}"
                for j in range(n)
            ]
            for i in range(n)
        ]

    te_n = _fn(te_nma)
    te_d = _fn(te_dir)

    lo_n = _format_right(lo_nma, digits) if ci else {}
    up_n = _format_right(up_nma, digits) if ci else {}
    lo_d = _format_right(lo_dir, digits) if ci else {}
    up_d = _format_right(up_dir, digits) if ci else {}

    cells = [[text_na] * n for _ in range(n)]
    for r in range(n):
        cells[r][r] = trts[r]
        for c in range(n):
            if r == c:
                continue
            if r < c:
                # Upper triangle: direct estimate (row vs column).
                if np.isnan(te_dir[r, c]):
                    cells[r][c] = text_na
                elif ci:
                    cells[r][c] = (
                        f"{te_d[r][c]} [{lo_d[(r, c)]}; {up_d[(r, c)]}]"
                    )
                else:
                    cells[r][c] = te_d[r][c]
            else:
                # Lower triangle: network estimate (column vs row).
                if np.isnan(te_nma[c, r]):
                    cells[r][c] = text_na
                elif ci:
                    cells[r][c] = (
                        f"{te_n[c][r]} [{lo_n[(c, r)]}; {up_n[(c, r)]}]"
                    )
                else:
                    cells[r][c] = te_n[c][r]
    return cells


@dataclass
class NetleagueResult:
    """League table (network estimates in the lower triangle, direct estimates
    in the upper triangle), replicating ``netmeta::netleague``.

    ``common`` and ``random`` are matrices (lists of string rows) of formatted
    cells: ``"TE [lower; upper]"``, ``"."`` for missing, treatment names on the
    diagonal. ``trts`` is the row/column order. ``details`` is the explanatory
    caption text used by R's print method.
    """

    trts: list[str]
    common: list[list[str]] = field(default_factory=list)
    random: list[list[str]] = field(default_factory=list)
    details: str = ""

    def to_dict(self) -> dict:
        return {
            "trts": self.trts,
            "common": self.common,
            "random": self.random,
            "details": self.details,
        }

    def to_string(self) -> str:
        """Render the league table(s) as fixed-width text, like R."""
        out = []
        if self.common:
            out.append("League table (common effects model):")
            out.append(_league_to_text(self.trts, self.common))
        if self.random:
            out.append("League table (random effects model):")
            out.append(_league_to_text(self.trts, self.random))
        if self.details:
            out.append(self.details)
        return "\n".join(out)


def _league_to_text(trts: list[str], cells: list[list[str]]) -> str:
    header = [""] + list(trts)
    rows = [[trts[i]] + list(cells[i]) for i in range(len(trts))]
    table = [header] + rows
    widths = [
        max(len(table[r][c]) for r in range(len(table)))
        for c in range(len(table[0]))
    ]
    lines = []
    for r in range(len(table)):
        lines.append(
            " ".join(table[r][c].rjust(widths[c]) for c in range(len(table[0])))
        )
    return "\n".join(lines)


def netleague(
    x: NetmetaResult,
    common: bool = True,
    random: bool = True,
    seq: list[str] | None = None,
    ci: bool = True,
    backtransf: bool = False,
    digits: int = 4,
    text_na: str = ".",
    details: bool = True,
) -> NetleagueResult:
    """League table for a network meta-analysis.

    Replicates ``netmeta::netleague`` for a single ``netmeta`` object: the
    lower triangle gives the network estimates (column vs row) and the upper
    triangle the direct estimates (row vs column).

    Args:
        x: a :class:`NetmetaResult` from :func:`netmeta`.
        common: include the common-effects table.
        random: include the random-effects table.
        seq: optional row/column treatment order (default: ``x.trts``).
        ci: include confidence intervals.
        backtransf: back-transform to the original scale (``exp`` for ratio
            measures; ``False`` for ``MD``).
        digits: number of decimals.
        text_na: string used for missing cells.
        details: include the explanatory caption.
    """
    if seq is None:
        seq = list(x.trts)
    else:
        seq = list(seq)
    idx = [x.trts.index(t) for t in seq]
    n = len(seq)

    def _pick(mat: np.ndarray) -> np.ndarray:
        return mat[np.ix_(idx, idx)]

    details_txt = (
        "\nLower triangle: results from network meta-analysis (column vs row)"
        "\nUpper triangle: results from direct comparisons (row vs column)\n"
        if details
        else ""
    )

    res = NetleagueResult(trts=seq, details=details_txt)

    if common:
        te_nma = _pick(x.TE_common)
        lo_nma = _pick(x.lower_common)
        up_nma = _pick(x.upper_common)
        te_dir = _pick(x.TE_direct_common)
        lo_dir = _pick(x.lower_direct_common)
        up_dir = _pick(x.upper_direct_common)
        if backtransf:
            te_nma, lo_nma, up_nma = _backtransf3(te_nma, lo_nma, up_nma, x.sm)
            te_dir, lo_dir, up_dir = _backtransf3(te_dir, lo_dir, up_dir, x.sm)
        res.common = _league_cells(
            te_nma, lo_nma, up_nma, te_dir, lo_dir, up_dir,
            seq, digits, text_na, ci,
        )

    if random:
        te_nma = _pick(x.TE_random)
        lo_nma = _pick(x.lower_random)
        up_nma = _pick(x.upper_random)
        te_dir = _pick(x.TE_direct_random)
        lo_dir = _pick(x.lower_direct_random)
        up_dir = _pick(x.upper_direct_random)
        if backtransf:
            te_nma, lo_nma, up_nma = _backtransf3(te_nma, lo_nma, up_nma, x.sm)
            te_dir, lo_dir, up_dir = _backtransf3(te_dir, lo_dir, up_dir, x.sm)
        res.random = _league_cells(
            te_nma, lo_nma, up_nma, te_dir, lo_dir, up_dir,
            seq, digits, text_na, ci,
        )

    return res


def _backtransf3(
    te: np.ndarray, lo: np.ndarray, up: np.ndarray, sm: str
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Back-transform point estimate and CI to the original scale."""
    if sm == "VE":
        te_t = -np.exp(-te)
        lo_t = -np.exp(-up)
        up_t = -np.exp(-lo)
        return te_t, lo_t, up_t
    return np.exp(te), np.exp(lo), np.exp(up)


# ---------------------------------------------------------------------------
# Network connectivity & distances (netmeta::netconnection / netmeta::netdistance)
# ---------------------------------------------------------------------------

def _netdistance_matrix(a: np.ndarray) -> np.ndarray:
    """Graph distance matrix (shortest path, in number of edges).

    Exact transcription of ``netmeta::netdistance.default``: ``D = sign(A)``,
    then all off-diagonal zero entries become ``Inf``, followed by a layered
    breadth-first relaxation that turns every entry into the number of edges
    on the shortest path (``Inf`` when the two nodes are in different
    subnetworks).
    """
    a = np.asarray(a, dtype=float)
    n = a.shape[0]
    d = np.sign(a)
    for i in range(n - 1):
        for j in range(i + 1, n):
            if d[i, j] == 0:
                d[i, j] = np.inf
                d[j, i] = np.inf
    for dist in range(1, n):
        for i in range(n):
            for j in range(n):
                if d[i, j] == dist:
                    for k in range(n):
                        akj = d[k, i] + dist
                        if akj < d[k, j]:
                            d[k, j] = akj
    return d


def _hclust_order(d2: np.ndarray) -> list[int]:
    """Leaf order of ``hclust(dist(D2))`` (complete linkage, Euclidean).

    Replicates R's ``hclust(dist(D2))$order`` used by ``netmeta::netdistance``
    to sort the nodes of a network. Ties are broken by SciPy's complete-linkage
    implementation, which agrees with R for the common (non-degenerate) cases;
    the order only affects the *display* ordering of disconnected components
    and never the numeric results.
    """
    from scipy.cluster.hierarchy import leaves_list, linkage
    from scipy.spatial.distance import pdist

    d2 = np.asarray(d2, dtype=float)
    cond = pdist(d2, metric="euclidean")
    z = linkage(cond, method="complete")
    return [int(i) for i in leaves_list(z)]


@dataclass
class NetdistanceResult:
    """Graph distance matrix (netmeta::netdistance).

    ``D[i, j]`` is the number of edges on the shortest path between nodes
    ``i`` and ``j`` (``inf`` if they lie in different subnetworks), diagonal
    zero. ``order`` is the ``hclust`` leaf order R attaches for display
    purposes; ``labels`` are the row/column names.
    """

    D: np.ndarray
    labels: list[str] = field(default_factory=list)
    order: list[int] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "D": np.where(np.isinf(self.D), None, self.D).tolist(),
            "labels": self.labels,
            "order": self.order,
        }


@dataclass
class NetconnectionResult:
    """Network connectivity structure (netmeta::netconnection).

    Groups the treatment network into its connected subnetworks and reports
    the graph distance matrix ``D_matrix``, the weighted adjacency matrix
    ``A_matrix`` (``A[i,j]`` = number of studies directly comparing ``i`` and
    ``j``) and the Laplacian ``L_matrix``, all indexed by ``seq``.
    """

    treat1: list = field(default_factory=list)
    treat2: list = field(default_factory=list)
    studlab: list = field(default_factory=list)
    design: list = field(default_factory=list)          # per-comparison design
    subnet: list = field(default_factory=list)          # per-comparison subnet
    k: int = 0                                          # number of studies
    m: int = 0                                          # number of comparisons
    n: int = 0                                          # number of treatments
    n_subnets: int = 0
    d: int = 0                                          # number of designs
    seq: list[str] = field(default_factory=list)
    D_matrix: np.ndarray = field(default_factory=lambda: np.empty((0, 0)))
    A_matrix: np.ndarray = field(default_factory=lambda: np.empty((0, 0)))
    L_matrix: np.ndarray = field(default_factory=lambda: np.empty((0, 0)))
    designs: list[str] = field(default_factory=list)
    comparisons: list[str] = field(default_factory=list)
    subnet_comparisons: list[int] = field(default_factory=list)

    def to_dict(self) -> dict:
        def _m(x: np.ndarray):
            if x.size == 0:
                return []
            return np.where(np.isinf(x), None, x).tolist()

        return {
            "treat1": self.treat1,
            "treat2": self.treat2,
            "studlab": self.studlab,
            "design": self.design,
            "subnet": self.subnet,
            "k": self.k, "m": self.m, "n": self.n,
            "n_subnets": self.n_subnets, "d": self.d,
            "seq": self.seq,
            "D_matrix": _m(self.D_matrix),
            "A_matrix": self.A_matrix.tolist() if self.A_matrix.size else [],
            "L_matrix": _m(self.L_matrix),
            "designs": self.designs,
            "comparisons": self.comparisons,
            "subnet_comparisons": self.subnet_comparisons,
        }


def netconnection(
    treat1,
    treat2,
    studlab=None,
    sep_trts: str = ":",
    title: str = "",
    details_disconnected: bool = False,
) -> NetconnectionResult:
    """Determine the (sub)network structure of a network meta-analysis.

    Replicates ``netmeta::netconnection``: builds the weighted adjacency matrix
    from ``treat1``/``treat2``, computes the graph-distance matrix, splits the
    network into its connected subnetworks and reports the number of studies
    (``k``), pairwise comparisons (``m``), treatments (``n``), subnetworks
    (``n_subnets``) and designs (``d``), together with the ordered distance /
    adjacency / Laplacian matrices.

    Args:
        treat1, treat2: the two treatments of each pairwise comparison
            (``m`` vectors, e.g. ``d["treat1"]`` / ``d["treat2"]``).
        studlab: study labels; ``None`` treats every comparison as its own
            study.
        sep_trts: separator used in the ``comparisons`` labels.
    """
    treat1 = [str(t) for t in treat1]
    treat2 = [str(t) for t in treat2]
    m = len(treat1)
    if len(treat2) != m:
        raise ValueError("Arguments 'treat1' and 'treat2' must have the same length.")
    if studlab is None:
        studlab = [str(i + 1) for i in range(m)]
    else:
        studlab = [str(s) for s in studlab]
        if len(studlab) != m:
            raise ValueError("Arguments 'treat1' and 'studlab' must have the same length.")
    for a, b in zip(treat1, treat2):
        if a == b:
            raise ValueError("Treatments must be different (arguments 'treat1' and 'treat2').")

    labels = sorted(set(treat1) | set(treat2))
    n = len(labels)
    pos = {t: i for i, t in enumerate(labels)}

    # Weighted adjacency: A[i,j] = number of studies comparing i and j.
    a_mat = np.zeros((n, n), dtype=float)
    for a, b in zip(treat1, treat2):
        a_mat[pos[a], pos[b]] += 1.0
        a_mat[pos[b], pos[a]] += 1.0

    d_mat = _netdistance_matrix(a_mat)
    l_mat = np.diag(a_mat.sum(axis=1)) - a_mat

    # hclust leaf order (matches R's hclust(dist(D2))$order).
    maxdist = n
    d2 = np.where(np.isinf(d_mat), maxdist, d_mat)
    order = _hclust_order(d2)

    # Sort D / A / L by the hclust order.
    d_sorted = d_mat[np.ix_(order, order)]
    a_sorted = a_mat[np.ix_(order, order)]
    l_sorted = l_mat[np.ix_(order, order)]
    order_labels = [labels[i] for i in order]

    # Walk through the sorted distance matrix to carve out the subnetworks.
    id_treats = []
    id_subnets = []
    subnet_i = 0
    dat_subnet = []  # (subnet, comparison, treat1, treat2)
    a_loop = a_sorted.copy()
    d_loop = d_sorted.copy()
    o_loop = list(order_labels)
    while d_loop.shape[0] > 0:
        subnet_i += 1
        nrows = d_loop.shape[0]
        inf_pos = [i for i in range(nrows) if np.isinf(d_loop[i, 0])]
        next_subnet = inf_pos[0] if inf_pos else nrows
        sel = list(range(next_subnet))
        for s in sel:
            id_treats.append(o_loop[s])
            id_subnets.append(subnet_i)
        sub_labels = [o_loop[s] for s in sel]
        sort_idx = sorted(range(len(sub_labels)), key=lambda x: sub_labels[x])
        a_i = a_loop[np.ix_(sel, sel)][np.ix_(sort_idx, sort_idx)]
        sub_labels_sorted = [sub_labels[x] for x in sort_idx]
        nci = a_i.shape[0]
        for row_i in range(nci - 1):
            for col_i in range(1, nci):
                if col_i > row_i and a_i[row_i, col_i] > 0:
                    t1 = sub_labels_sorted[row_i]
                    t2 = sub_labels_sorted[col_i]
                    dat_subnet.append((subnet_i, f"{t1}{sep_trts}{t2}", t1, t2))
        keep = [i for i in range(nrows) if i not in set(sel)]
        a_loop = a_loop[np.ix_(keep, keep)]
        d_loop = d_loop[np.ix_(keep, keep)]
        o_loop = [o_loop[i] for i in keep]

    n_subnets = subnet_i

    # seq: for each subnet, the treatments in order of first appearance over
    # the (alphabetically ordered) comparisons, treat1 before treat2.
    seq = []
    for comp in range(1, n_subnets + 1):
        t1_list = [d[2] for d in dat_subnet if d[0] == comp]
        t2_list = [d[3] for d in dat_subnet if d[0] == comp]
        seen = set()
        for x in t1_list + t2_list:
            if x not in seen:
                seen.add(x)
                seq.append(x)

    # Reorder matrices by seq.
    seq_pos = {t: i for i, t in enumerate(seq)}
    idx = [seq_pos[t] for t in labels]
    d_final = d_mat[np.ix_(idx, idx)]
    a_final = a_mat[np.ix_(idx, idx)]
    l_final = l_mat[np.ix_(idx, idx)]

    # Designs: sorted unique treatments of each study joined by ':' (R's
    # designs() default separator, independent of sep.trts).
    study_treats: dict[str, set] = {}
    for a, b, s in zip(treat1, treat2, studlab):
        study_treats.setdefault(s, set()).update([a, b])
    design_per_study = {s: ":".join(sorted(study_treats[s])) for s in study_treats}
    design = [design_per_study[s] for s in studlab]
    designs = sorted(set(design))

    treat_to_subnet = dict(zip(id_treats, id_subnets))
    subnet = [treat_to_subnet[t] for t in treat1]

    comparisons = [d[1] for d in dat_subnet]
    subnet_comparisons = [d[0] for d in dat_subnet]

    return NetconnectionResult(
        treat1=treat1,
        treat2=treat2,
        studlab=studlab,
        design=design,
        subnet=subnet,
        k=len(set(studlab)),
        m=m,
        n=n,
        n_subnets=n_subnets,
        d=len(designs),
        seq=seq,
        D_matrix=d_final,
        A_matrix=a_final,
        L_matrix=l_final,
        designs=designs,
        comparisons=comparisons,
        subnet_comparisons=subnet_comparisons,
    )


def netdistance(x, sort: bool = True) -> NetdistanceResult:
    """Graph distance matrix (netmeta::netdistance).

    Accepts either a weighted/0-1 adjacency ``ndarray``, a
    :class:`NetmetaResult`, or a :class:`NetconnectionResult`. For a
    ``NetmetaResult`` the adjacency is sorted by the ``netconnection`` order
    when ``sort`` is true, matching ``netmeta::netdistance.netmeta``.
    """
    if isinstance(x, NetmetaResult):
        labels = list(x.trts)
        a_mat = np.zeros((x.n, x.n), dtype=float)
        for i in range(x.n - 1):
            for j in range(i + 1, x.n):
                if not np.isnan(x.A_matrix[i, j]):
                    a_mat[i, j] = a_mat[j, i] = float(x.A_matrix[i, j])
        if sort:
            seq = netconnection(x.treat1_raw, x.treat2_raw, x.studlab_raw).seq
            idx = [labels.index(t) for t in seq]
            a_mat = a_mat[np.ix_(idx, idx)]
            labels = seq
        return NetdistanceResult(D=_netdistance_matrix(a_mat), labels=labels)
    if isinstance(x, NetconnectionResult):
        return NetdistanceResult(
            D=_netdistance_matrix(x.A_matrix.astype(float)),
            labels=list(x.seq),
        )
    a = np.asarray(x, dtype=float)
    d = _netdistance_matrix(a)
    d2 = np.where(np.isinf(d), a.shape[0], d)
    order = _hclust_order(d2)
    return NetdistanceResult(D=d, order=order)


# ---------------------------------------------------------------------------
# netpairwise: pairwise (meta-analysis) view of a network (netmeta::netpairwise)
# ---------------------------------------------------------------------------

@dataclass
class NetpairwiseResult:
    """Pairwise meta-analysis of the direct comparisons of a network.

    ``netmeta::netpairwise`` re-expresses a network meta-analysis as a plain
    pairwise meta-analysis (a ``metagen``-style object) over the direct
    comparisons, plus a per-pair direct-evidence table. Every comparison is
    oriented as "non-reference treatment vs reference treatment" when a
    reference group is set, and alphabetically (treat1 < treat2) otherwise.
    """

    trts: list[str]
    k: int                          # number of pairwise comparisons (edges)
    reference_group: str
    sm: str
    level: float
    method_tau: str

    # Overall pooled estimates (metagen on all comparisons, DL tau^2).
    TE_common: float = float("nan")
    seTE_common: float = float("nan")
    lower_common: float = float("nan")
    upper_common: float = float("nan")
    statistic_common: float = float("nan")
    pval_common: float = float("nan")
    TE_random: float = float("nan")
    seTE_random: float = float("nan")
    lower_random: float = float("nan")
    upper_random: float = float("nan")
    statistic_random: float = float("nan")
    pval_random: float = float("nan")
    tau2: float = float("nan")
    tau: float = float("nan")
    I2: float = float("nan")
    lower_I2: float = float("nan")
    upper_I2: float = float("nan")
    Q: float = float("nan")
    df_Q: float = float("nan")
    pval_Q: float = float("nan")

    # Oriented comparison-level data (sorted by treatment pair).
    TE: list = field(default_factory=list)
    seTE: list = field(default_factory=list)
    treat1: list = field(default_factory=list)
    treat2: list = field(default_factory=list)
    studlab: list = field(default_factory=list)

    # Per-pair direct evidence (one dict per pair with direct evidence).
    pairs: list = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "trts": self.trts,
            "k": self.k,
            "reference_group": self.reference_group,
            "sm": self.sm,
            "level": self.level,
            "method_tau": self.method_tau,
            "TE_common": self.TE_common,
            "seTE_common": self.seTE_common,
            "lower_common": self.lower_common,
            "upper_common": self.upper_common,
            "statistic_common": self.statistic_common,
            "pval_common": self.pval_common,
            "TE_random": self.TE_random,
            "seTE_random": self.seTE_random,
            "lower_random": self.lower_random,
            "upper_random": self.upper_random,
            "statistic_random": self.statistic_random,
            "pval_random": self.pval_random,
            "tau2": self.tau2,
            "tau": self.tau,
            "I2": self.I2,
            "lower_I2": self.lower_I2,
            "upper_I2": self.upper_I2,
            "Q": self.Q,
            "df_Q": self.df_Q,
            "pval_Q": self.pval_Q,
            "TE": self.TE,
            "seTE": self.seTE,
            "treat1": self.treat1,
            "treat2": self.treat2,
            "studlab": self.studlab,
            "pairs": self.pairs,
        }


def _orient_comparisons(te, se, treat1, treat2, studlab, reference_group, trts):
    """Orient and sort pairwise comparisons (netmeta::netpairwise direction).

    Each comparison is oriented as "non-reference vs reference" when one of the
    two treatments is ``reference_group``; otherwise treatments are ordered
    alphabetically (treat1 < treat2), flipping the sign of the effect when the
    order is swapped. The result is sorted by (treat1, treat2) position.
    """
    idx = {t: i for i, t in enumerate(trts)}
    out = []
    for k in range(len(te)):
        a, b = treat1[k], treat2[k]
        y = float(te[k])
        if reference_group and b == reference_group:
            pass
        elif reference_group and a == reference_group:
            a, b = b, a
            y = -y
        elif a > b:
            a, b = b, a
            y = -y
        out.append([a, b, y, float(se[k]), studlab[k]])
    out.sort(key=lambda r: (idx[r[0]], idx[r[1]]))
    return out


def netpairwise(
    x,
    separate_trts: bool = True,
    reference_group: str | None = None,
    sm: str = "",
    level: float = 0.95,
    method_tau: str = "DL",
    common: bool = True,
    random: bool = True,
) -> NetpairwiseResult:
    """Pairwise (meta-analysis) view of a network meta-analysis.

    Transcribes ``netmeta::netpairwise``. ``x`` is a :class:`NetmetaResult`
    (from :func:`netmeta`). The result is a metagen-style object whose overall
    pooled estimates come from a DerSimonian-Laird random-effects pairwise
    meta-analysis of all direct comparisons, plus a ``pairs`` table giving the
    per-comparison direct evidence (common/random effect, Q, I^2, tau^2).
    """
    if isinstance(x, NetmetaResult):
        te = list(x.TE_raw)
        se = list(x.seTE_raw)
        t1 = list(x.treat1_raw)
        t2 = list(x.treat2_raw)
        stud = list(x.studlab_raw)
        trts = list(x.trts)
        reference_group = reference_group if reference_group else x.reference_group
        if not sm:
            sm = x.sm
        if level is None:
            level = x.level
    else:
        raise TypeError("netpairwise expects a NetmetaResult")

    if method_tau.upper() != "DL":
        raise ValueError("netpairwise only supports method_tau='DL'")

    oriented = _orient_comparisons(te, se, t1, t2, stud, reference_group, trts)
    te_arr = np.array([r[2] for r in oriented], dtype=float)
    se_arr = np.array([r[3] for r in oriented], dtype=float)
    k = len(te_arr)

    z = stats.norm.ppf(1 - (1 - level) / 2)

    result = NetpairwiseResult(
        trts=trts,
        k=k,
        reference_group=reference_group,
        sm=sm,
        level=level,
        method_tau=method_tau,
        TE=[r[2] for r in oriented],
        seTE=[r[3] for r in oriented],
        treat1=[r[0] for r in oriented],
        treat2=[r[1] for r in oriented],
        studlab=[r[4] for r in oriented],
    )

    q = _cochran_q(te_arr, se_arr)
    df_q = float(k - 1)
    result.Q = q
    result.df_Q = df_q
    result.pval_Q = float(stats.chi2.sf(q, df_q)) if df_q > 0 else float("nan")

    # Common-effects pooled estimate (inverse-variance, tau^2 = 0).
    if common:
        mu_c, se_c = _iv_pool(te_arr, se_arr, 0.0)
        result.TE_common = mu_c
        result.seTE_common = se_c
        result.lower_common = mu_c - z * se_c
        result.upper_common = mu_c + z * se_c
        result.statistic_common = mu_c / se_c
        result.pval_common = float(2.0 * stats.norm.sf(abs(mu_c / se_c)))

    # Random-effects pooled estimate (DerSimonian-Laird tau^2).
    tau2 = _dl_tau2(te_arr, se_arr)
    tau2_eff = 0.0 if math.isnan(tau2) else tau2
    result.tau2 = tau2
    result.tau = math.sqrt(tau2) if not math.isnan(tau2) else float("nan")
    i2, lower_i2, upper_i2 = _isquared(q, df_q, level)
    result.I2 = i2
    result.lower_I2 = lower_i2
    result.upper_I2 = upper_i2
    if random:
        mu_r, se_r = _iv_pool(te_arr, se_arr, tau2_eff)
        result.TE_random = mu_r
        result.seTE_random = se_r
        result.lower_random = mu_r - z * se_r
        result.upper_random = mu_r + z * se_r
        result.statistic_random = mu_r / se_r
        result.pval_random = float(2.0 * stats.norm.sf(abs(mu_r / se_r)))

    # Per-pair direct evidence (group consecutive oriented comparisons).
    pairs = []
    i = 0
    while i < k:
        j = i
        while j < k and oriented[j][0] == oriented[i][0] \
                and oriented[j][1] == oriented[i][1]:
            j += 1
        y = [r[2] for r in oriented[i:j]]
        s = [r[3] for r in oriented[i:j]]
        kk = j - i
        mu_c, se_c = _iv_pool(y, s, 0.0)
        qp = _cochran_q(y, s)
        t2p = _dl_tau2(y, s)
        t2p_eff = 0.0 if math.isnan(t2p) else t2p
        mu_r, se_r = _iv_pool(y, s, t2p_eff)
        i2p, _, _ = _isquared(qp, float(kk - 1), level)
        pairs.append({
            "treat1": oriented[i][0],
            "treat2": oriented[i][1],
            "k": kk,
            "TE_common": mu_c,
            "seTE_common": se_c,
            "lower_common": mu_c - z * se_c,
            "upper_common": mu_c + z * se_c,
            "TE_random": mu_r,
            "seTE_random": se_r,
            "lower_random": mu_r - z * se_r,
            "upper_random": mu_r + z * se_r,
            "Q": qp,
            "I2": i2p,
            "tau2": t2p,
        })
        i = j
    result.pairs = pairs

    return result


# ---------------------------------------------------------------------------
# pairwise: arm-level data -> comparison-level data (netmeta::pairwise)
# ---------------------------------------------------------------------------

def pairwise(
    treat,
    n=None,
    event=None,
    mean=None,
    sd=None,
    time=None,
    studlab=None,
    data=None,
    sm: str = "",
    incr: float = 0.5,
    allincr: bool = False,
    addincr: bool = False,
    allstudies: bool = False,
) -> "PairwiseData":
    """Convert arm-level data to comparison-level data (netmeta::pairwise).

    Supports binary (``event`` / ``n``; sm OR/RR/RD) and continuous
    (``mean`` / ``sd`` / ``n``; sm MD/SMD) data. Each study with ``a`` arms
    contributes ``a(a-1)/2`` pairwise comparisons (treat1 vs treat2).
    """
    if data is not None:
        raise NotImplementedError("data-frame input not supported; pass vectors")

    treat = list(treat)
    k_arms = len(treat)

    if event is not None:
        event = [float(e) for e in event]
        n = [float(nv) for nv in n]
        if sm == "":
            sm = "OR"
        out = _pairwise_binary(treat, event, n, studlab, sm, incr, allincr, addincr)
    else:
        n = [float(nv) for nv in n]
        mean = [float(mv) for mv in mean]
        sd = [float(sv) for sv in sd]
        if sm == "":
            sm = "MD"
        out = _pairwise_continuous(treat, n, mean, sd, studlab, sm)

    return PairwiseData(sm=sm, **out)


@dataclass
class PairwiseData:
    """Comparison-level data produced by :func:`pairwise`."""

    sm: str
    TE: list
    seTE: list
    treat1: list
    treat2: list
    studlab: list
    narms: list

    def to_dict(self) -> dict:
        return {
            "sm": self.sm,
            "TE": self.TE,
            "seTE": self.seTE,
            "treat1": self.treat1,
            "treat2": self.treat2,
            "studlab": self.studlab,
            "narms": self.narms,
        }


def _pairwise_continuous(treat, n, mean, sd, studlab, sm):
    te, se, t1, t2, stud, narms = [], [], [], [], [], []
    by_study = {}
    for i, s in enumerate(studlab if studlab is not None else range(len(treat))):
        by_study.setdefault(s, []).append(i)
    for s, inds in by_study.items():
        a = len(inds)
        n_i = sum(n[i] for i in inds)  # study total n (each arm counted once)
        # For SMD with >2 arms, netmeta replaces every arm's sd by the
        # study-level pooled SD before forming pairwise comparisons.
        sds = [sd[i] for i in inds]
        if sm.upper() == "SMD" and a > 2:
            num = sum((n[i] - 1) * sd[i] ** 2 for i in inds)
            den = sum(n[i] - 1 for i in inds)
            sdp = math.sqrt(num / den)
            sds = [sdp] * a
        for x in range(a):
            for y in range(x + 1, a):
                ix, iy = inds[x], inds[y]
                t1.append(treat[ix])
                t2.append(treat[iy])
                if sm.upper() == "SMD":
                    # Cohen's d with pooled SD (n-1 denominators).
                    nx, ny = n[ix], n[iy]
                    sp2 = ((nx - 1) * sds[x] ** 2 + (ny - 1) * sds[y] ** 2) \
                        / (nx + ny - 2)
                    d = (mean[ix] - mean[iy]) / math.sqrt(sp2)
                    # netmeta overrides seTE for SMD:
                    #   seTE = sqrt(1/n1 + 1/n2 + TE^2/(2 * n_i))
                    vte = 1.0 / nx + 1.0 / ny + d ** 2 / (2.0 * n_i)
                    te.append(d)
                    se.append(math.sqrt(vte))
                else:  # MD
                    te.append(mean[ix] - mean[iy])
                    se.append(math.sqrt(sd[ix] ** 2 / n[ix] + sd[iy] ** 2 / n[iy]))
                stud.append(s)
                narms.append(a)
    return {"TE": te, "seTE": se, "treat1": t1, "treat2": t2,
            "studlab": [str(s) for s in stud], "narms": narms}


def _pairwise_binary(treat, event, n, studlab, sm, incr, allincr, addincr):
    te, se, t1, t2, stud, narms = [], [], [], [], [], []
    by_study = {}
    for i, s in enumerate(studlab if studlab is not None else range(len(treat))):
        by_study.setdefault(s, []).append(i)
    sm = sm.upper()
    for s, inds in by_study.items():
        a = len(inds)
        # Study-level continuity correction (netmeta::pairwise).
        if sm == "ASD":
            incr_study = 0.0
        else:
            n_zeros = sum(1 for i in inds if event[i] == 0)
            n_all = sum(1 for i in inds if event[i] == n[i])
            sparse = (n_zeros > 0) or (n_all > 0)
            if addincr:
                incr_study = incr
            elif allincr:
                incr_study = incr if n_zeros > 0 else 0.0
            else:
                incr_study = incr if sparse else 0.0
        for x in range(a):
            for y in range(x + 1, a):
                ix, iy = inds[x], inds[y]
                a1, n1 = event[ix], n[ix]
                a2, n2 = event[iy], n[iy]
                ie = ic = incr_study
                t1.append(treat[ix])
                t2.append(treat[iy])
                if sm == "RR":
                    # meta::metabin RR (RR.Cochrane = FALSE).
                    allevents = (a1 == n1 and a2 == n2)
                    e1 = a1 + ie * (0.0 if allevents else 1.0)
                    e2 = a2 + ic * (0.0 if allevents else 1.0)
                    te.append(math.log((e1 / (n1 + ie)) / (e2 / (n2 + ic))))
                    se.append(math.sqrt(
                        1 / e1 - 1 / (n1 + ie) + 1 / e2 - 1 / (n2 + ic)))
                elif sm == "RD":
                    te.append(a1 / n1 - a2 / n2)
                    se.append(math.sqrt(
                        (a1 + ie) * (n1 - a1 + ie) / (n1 + 2 * ie) ** 3
                        + (a2 + ic) * (n2 - a2 + ic) / (n2 + 2 * ic) ** 3))
                else:  # OR
                    e1 = a1 + ie
                    e2 = a2 + ic
                    te.append(math.log(
                        (e1 * (n2 - a2 + ic)) / ((n1 - a1 + ie) * e2)))
                    se.append(math.sqrt(
                        1 / e1 + 1 / (n1 - a1 + ie) + 1 / e2 + 1 / (n2 - a2 + ic)))
                stud.append(s)
                narms.append(a)
    return {"TE": te, "seTE": se, "treat1": t1, "treat2": t2,
            "studlab": [str(s) for s in stud], "narms": narms}


# ---------------------------------------------------------------------------
# netcomb / discomb: additive component network meta-analysis (CNMA)
# ---------------------------------------------------------------------------

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
    study_to_trts = {s: set() for s in studies}
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


# ---------------------------------------------------------------------------
# netmetabin: binary network meta-analysis (netmeta::netmetabin)
# ---------------------------------------------------------------------------

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

def _netmetabin_mh(event1, n1, event2, n2, treat1, treat2, studlab,
                   level, level_ma, reference_group, cc_pooled, method="MH"):
    """Mantel-Haenszel network meta-analysis (Stijnen et al. 2010).

    ``event1/n1`` describe the ``treat1`` arm and ``event2/n2`` the ``treat2``
    arm of each comparison (already normalized so ``treat1 < treat2``).  Returns
    a dict of all quantities needed to build a :class:`NetmetabinResult`.
    """
    from .metabin import _mh_or

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

    if method == "NCH":
        TE_basic, cov_basic = _netmetabin_nch(
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
            mh_est, mh_var = _mh_or(a, b, c, dd, nn1, nn2)
            TE_direct[i, j] = mh_est
            TE_direct[j, i] = -mh_est
            seTE_direct[i, j] = seTE_direct[j, i] = math.sqrt(mh_var)
            # Cochran Q centered on the inverse-variance mean (Q.Cochrane=FALSE).
            yi = np.log((a * dd) / (b * c))
            vi = 1.0 / a + 1.0 / b + 1.0 / c + 1.0 / dd
            se_i = np.sqrt(vi)
            q_pair = _cochran_q(yi, se_i)
            k_pair = len(sel)
            Q_direct[i, j] = Q_direct[j, i] = q_pair
            tau2_direct[i, j] = tau2_direct[j, i] = _dl_tau2(yi, se_i)
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
        a = event1[k]; b = n1[k] - event1[k]
        c = event2[k]; dd = n2[k] - event2[k]
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
        "A": A, "H": H,
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
            non-central hypergeometric likelihood), "Inverse" (classic NMA
            via :func:`netmeta`), or "LRP" (needs brglm2, not available).
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
    if method == "LRP":
        raise NotImplementedError(
            "netmetabin method='LRP' needs brglm2's Firth-penalised logistic "
            "regression, which has no pimeta equivalent yet; use 'MH', "
            "'NCH' or 'Inverse'"
        )

    sm = sm.upper()
    if method in ("MH", "NCH"):
        if sm != "OR":
            sm = "OR"
        random = False

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
        from .metabin import metabin
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
        A_matrix=res["A"], H_matrix_common=res["H"],
        TE=res["TE"], seTE=res["seTE"],
        treat1=treat1, treat2=treat2, studlab=studlab,
        event1=list(event1), event2=list(event2), n1=list(n1), n2=list(n2),
    )


# --------------------------------------------------------------------------- #
# netcomplex
# --------------------------------------------------------------------------- #

def netcomplex(result) -> int:
    """``netmeta::netcomplex`` — complexity of a treatment network.

    The complexity is the number of independent cycles in the comparison graph,
    ``m - n + n_subnets`` (with ``m`` comparisons, ``n`` treatments and
    ``n_subnets`` connected components). A network without cycles has
    complexity 0.

    Args:
        result: a :class:`NetmetaResult` (or any object exposing ``m``, ``n``
            and ``n_subnets``).

    Returns:
        The network complexity as an ``int``.
    """
    m = int(getattr(result, "m"))
    n = int(getattr(result, "n"))
    ns = getattr(result, "n_subnets", None)
    if ns is None:
        ns = getattr(result, "nsubnets", 1)
    return m - n + int(ns)


# --------------------------------------------------------------------------- #
# netmetareg — network meta-regression
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
    """``netmeta::netmetareg`` — network meta-regression.

    Fits the treatment-by-covariate interaction model that netmeta uses::

        theta_k = (mu_t2 - mu_t1) + (beta_t2 - beta_t1) * x_k

    Comparisons are first re-oriented so that every study contributes contrasts
    against its own reference treatment (the network reference when the study
    contains it, otherwise the alphabetically last treatment of that study), the
    design matrix is ``[I(treat) | I(treat) * x]`` over the non-reference
    treatments, and the model is estimated by (restricted) maximum likelihood
    with a block-equicorrelated within-study covariance
    ``V = diag(se^2) + tau2 * CS(rho)`` — the same structure as metafor's
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
    n_trts = len(trts)
    t_index = {t: i for i, t in enumerate(trts)}

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
# netmeta::netmeasures
# --------------------------------------------------------------------------- #

class NetmeasuresResult:
    """Result of :func:`netmeasures` (``netmeta::netmeasures``)."""

    __slots__ = ("proportion", "meanpath", "minpar", "minpar_study",
                 "H_tilde", "random", "tau_preset")

    def __init__(self, proportion, meanpath, minpar, minpar_study,
                 H_tilde, random, tau_preset):
        self.proportion = proportion
        self.meanpath = meanpath
        self.minpar = minpar
        self.minpar_study = minpar_study
        self.H_tilde = H_tilde
        self.random = random
        self.tau_preset = tau_preset

    def __repr__(self):
        return (f"NetmeasuresResult(k={len(self.proportion)}, "
                f"random={self.random})")

    def as_dict(self):
        """R-style ``$`` access."""
        return {
            "proportion": self.proportion,
            "meanpath": self.meanpath,
            "minpar": self.minpar,
            "minpar.study": self.minpar_study,
            "H.tilde": self.H_tilde,
            "random": self.random,
            "tau.preset": self.tau_preset,
        }


def _htilde(H, groups):
    """``H.tilde`` — absolute-sum aggregation within design (or study) blocks.

    R::

        l <- lapply(split(H, <block index>), function(x) matrix(x, nrow(H)))
        H.tilde <- sapply(l, function(x) apply(x, 1, function(x)
                       0.5 * (abs(sum(x)) + sum(abs(x)))))
    """
    order = list(dict.fromkeys(groups))
    out = np.zeros((H.shape[0], len(order)))
    for j, g in enumerate(order):
        cols = [r for r, v in enumerate(groups) if v == g]
        blk = H[:, cols]
        out[:, j] = 0.5 * (np.abs(blk.sum(axis=1)) + np.abs(blk).sum(axis=1))
    return out, order


def netmeasures(result, random=None, tau_preset=None):
    """``netmeta::netmeasures`` — network path measures (Koenig et al. 2013).

    Args:
        result: a :class:`NetmetaResult`.
        random: use the random-effects model.  Defaults to ``True``; forced to
            ``True`` when ``tau_preset`` is given.
        tau_preset: fixed heterogeneity variance.

    Returns:
        :class:`NetmeasuresResult` with ``proportion`` (share of the network
        estimate's variance that is direct evidence), ``meanpath``, ``minpar``
        and ``minpar_study``.
    """
    if not isinstance(result, NetmetaResult):
        raise TypeError("argument 'result' must be a NetmetaResult")

    if random is None:
        random = True
    if tau_preset is not None:
        tau_preset = float(tau_preset)
        if tau_preset < 0:
            raise ValueError("argument 'tau_preset' must be non-negative")
        random = True

    if random:
        t2 = result.tau if tau_preset is None else tau_preset
        tau_preset_out = result.tau if tau_preset is None else tau_preset
    else:
        t2 = 0.0
        tau_preset_out = None

    nmak = _nma_krahn_full(
        result.TE_raw, result.seTE_raw, result.treat1_raw, result.treat2_raw,
        result.studlab_raw, result.narms_raw, list(result.trts),
        "", float(t2),
    )

    if nmak["n"] == 2:
        designs = list(dict.fromkeys(d["design"] for d in nmak["design"]))
        prop = {d: 1.0 for d in designs}
        return NetmeasuresResult(prop, dict(prop), None, None, None,
                                 random, tau_preset_out)

    H = nmak["H"]
    x_obs_s = nmak["x_obs_studies"]
    v_studies = nmak["V_studies"]
    v_studies_inv = np.linalg.pinv(v_studies)
    H_studies = (nmak["X_full"]
                 @ np.linalg.pinv(x_obs_s.T @ v_studies_inv @ x_obs_s)
                 @ x_obs_s.T
                 @ v_studies_inv)

    poss = list(nmak["trts_poss"])
    proportion = {name: 0.0 for name in poss}
    network = nmak["network"]
    for d in nmak["direct"]:
        comp = d["comparison"]
        if comp in proportion and not math.isnan(d["seTE"]):
            proportion[comp] = (network[comp][1] ** 2) / (d["seTE"] ** 2)

    H_tilde, designs = _htilde(H, nmak["design_col"])
    H_tilde_s, studies = _htilde(H_studies, nmak["studlab_studies"])

    meanpath = {n: float(H_tilde[i, :].sum()) for i, n in enumerate(poss)}
    minpar = {}
    for i, n in enumerate(poss):
        mx = float(np.abs(H_tilde[i, :]).max())
        minpar[n] = float("inf") if mx == 0.0 else 1.0 / mx
    minpar_study = {}
    for i, n in enumerate(poss):
        mx = float(np.abs(H_tilde_s[i, :]).max())
        minpar_study[n] = float("inf") if mx == 0.0 else 1.0 / mx

    return NetmeasuresResult(
        proportion=proportion,
        meanpath=meanpath,
        minpar=minpar,
        minpar_study=minpar_study,
        H_tilde={"rows": poss, "cols": designs, "values": H_tilde},
        random=random,
        tau_preset=tau_preset_out,
    )


# --------------------------------------------------------------------------- #
# netmeta::netimpact
# --------------------------------------------------------------------------- #

class NetimpactResult:
    """Result of :func:`netimpact` (``netmeta::netimpact``)."""

    __slots__ = ("impact_common", "impact_random", "impact_fixed",
                 "comparisons", "studies", "ignored_comparisons",
                 "seTE_ignore")

    def __init__(self, impact_common, impact_random, comparisons, studies,
                 ignored_comparisons, seTE_ignore):
        self.impact_common = impact_common
        self.impact_random = impact_random
        self.impact_fixed = impact_common
        self.comparisons = comparisons
        self.studies = studies
        self.ignored_comparisons = ignored_comparisons
        self.seTE_ignore = seTE_ignore

    def __repr__(self):
        return (f"NetimpactResult(studies={len(self.studies)}, "
                f"comparisons={len(self.comparisons)})")

    def as_dict(self):
        return {
            "impact.common": self.impact_common,
            "impact.random": self.impact_random,
            "impact.fixed": self.impact_common,
            "ignored.comparisons": self.ignored_comparisons,
            "seTE.ignore": self.seTE_ignore,
        }


def _comparison_list(trts):
    """Row-major ``i < j`` comparison labels — netmeta's ``x$comparisons``."""
    out = []
    for i in range(len(trts) - 1):
        for j in range(i + 1, len(trts)):
            out.append(f"{trts[i]}:{trts[j]}")
    return out


def netimpact(result, seTE_ignore=None):
    """``netmeta::netimpact`` — leave-one-study-out impact on all comparisons.

    Each study is down-weighted (its standard errors are replaced by
    ``seTE_ignore``) and the network is re-estimated with the original ``tau``
    as ``tau_preset``; the reported impact is ``1 - (seTE / seTE_removed)^2``.

    Returns:
        :class:`NetimpactResult` — ``impact_common``/``impact_random`` map
        study labels to ``{comparison: impact}`` dictionaries.
    """
    if not isinstance(result, NetmetaResult):
        raise TypeError("argument 'result' must be a NetmetaResult")

    se = np.asarray(result.seTE_raw, dtype=float)
    if seTE_ignore is None:
        seTE_ignore = 1000.0 * float(np.nanmax(se))
    seTE_ignore = float(seTE_ignore)

    te = list(result.TE_raw)
    t1 = list(result.treat1_raw)
    t2 = list(result.treat2_raw)
    stud = list(result.studlab_raw)
    trts = list(result.trts)
    studies = list(dict.fromkeys(stud))

    lower = [(i, j) for i in range(len(trts) - 1) for j in range(i + 1, len(trts))]
    comps = [f"{trts[i]}:{trts[j]}" for i, j in lower]

    def _flat(mat):
        mat = np.asarray(mat, dtype=float)
        return {(i, j): float(mat[i, j]) for i, j in lower}

    base_c = _flat(result.seTE_common)
    base_r = _flat(result.seTE_random)

    tiny = sys.float_info.epsilon ** 0.5  # .Machine$double.eps^0.5
    impact_c = {s: {} for s in studies}
    impact_r = {s: {} for s in studies}

    raw_comp = [f"{a}:{b}" for a, b in zip(t1, t2)]
    ignored = {s: [c for c, sv in zip(raw_comp, stud) if sv == s]
               for s in studies}

    for s in studies:
        se_i = [seTE_ignore if sv == s else se[k] for k, sv in enumerate(stud)]
        net_i = netmeta(te, se_i, t1, t2, stud, tau_preset=result.tau)
        new_c = _flat(net_i.seTE_common)
        new_r = _flat(net_i.seTE_random)
        for idx, key in enumerate(lower):
            name = comps[idx]
            ic = 1.0 - (base_c[key] / new_c[key]) ** 2
            ir = 1.0 - (base_r[key] / new_r[key]) ** 2
            impact_c[s][name] = 0.0 if abs(ic) < tiny else float(ic)
            impact_r[s][name] = 0.0 if abs(ir) < tiny else float(ir)

    return NetimpactResult(
        impact_common=impact_c,
        impact_random=impact_r,
        comparisons=comps,
        studies=studies,
        ignored_comparisons=ignored,
        seTE_ignore=seTE_ignore,
    )


# --------------------------------------------------------------------------- #
# netmeta::netpath / hatmatrix(method = "Davies", type = "full")
# --------------------------------------------------------------------------- #

def _hatmatrix_aggr(result, model="random", type="full"):
    """``netmeta::hatmatrix.aggr`` — the Davies hat matrix.

    ``type`` is ``"short"`` (e x e, observed comparisons), ``"long"``
    (all x observed) or ``"full"`` (all x all).
    """
    trts = list(result.trts)
    n = len(trts)
    m = len(result.TE_raw)
    t1p = list(result.treat1_pos)
    t2p = list(result.treat2_pos)
    se_adj = np.asarray(
        result.seTE_adj_random if model == "random" else result.seTE_adj_common,
        dtype=float,
    )
    B = _create_b(t1p, t2p, n)                    # raw m x n

    observed = set()
    for a, b in zip(result.treat1_raw, result.treat2_raw):
        i, j = trts.index(a), trts.index(b)
        observed.add((min(i, j), max(i, j)))
    obs_pairs = [(i, j) for i in range(n - 1) for j in range(i + 1, n)
                 if (i, j) in observed]
    obs_comps = [f"{trts[i]}:{trts[j]}" for i, j in obs_pairs]

    e = len(obs_pairs)
    W = np.zeros((e, e))
    for r, (i, j) in enumerate(obs_pairs):
        w = 0.0
        for k in range(m):
            if B[k, i] == 1.0 and B[k, j] == -1.0:
                w += 1.0 / se_adj[k] ** 2
        W[r, r] = w

    B_aggr = _create_b([p[0] for p in obs_pairs], [p[1] for p in obs_pairs], n)
    L = B_aggr.T @ W @ B_aggr
    L_plus = _invmat(L)
    L_plus[np.abs(L_plus) < sys.float_info.epsilon] = 0.0

    H_short = B_aggr @ L_plus @ B_aggr.T @ W
    B_full = _create_b_full(n)
    H_long = B_full @ L_plus @ B_aggr.T @ W

    c = n * (n - 1) // 2
    H_full = np.zeros((c, c))
    l_idx = 0
    for k in range(c):
        if l_idx < e:
            i, j = obs_pairs[l_idx]
            bi = int(np.argmax(B_full[k] == 1.0))
            bj = int(np.argmax(B_full[k] == -1.0))
            if bi == i and bj == j:
                H_full[:, k] = H_long[:, l_idx]
                l_idx += 1

    if type == "short":
        return {"matrix": H_short, "rows": obs_comps, "cols": obs_comps}
    if type == "long":
        return {"matrix": H_long, "rows": _comparison_list(trts),
                "cols": obs_comps}
    return {"matrix": H_full, "rows": _comparison_list(trts),
            "cols": _comparison_list(trts)}


def _all_simple_paths(edges, source, target):
    """All simple paths (igraph ``all_simple_paths``) as lists of node names."""
    adj = {}
    for a, b in edges:
        adj.setdefault(a, []).append(b)
        adj.setdefault(b, [])

    paths = []

    def _dfs(node, path, seen):
        if node == target:
            paths.append(list(path))
            return
        for nxt in adj.get(node, []):
            if nxt not in seen:
                seen.add(nxt)
                path.append(nxt)
                _dfs(nxt, path, seen)
                path.pop()
                seen.discard(nxt)

    if source in adj:
        _dfs(source, [source], {source})
    return paths


def _get_edges_from_path(nodes, sep=":"):
    """``get_edges_from_path`` — unordered consecutive edge labels."""
    if len(nodes) < 2:
        return []
    return [sep.join(sorted([nodes[k], nodes[k + 1]]))
            for k in range(len(nodes) - 1)]


def netpath(result, node1, node2, random=True):
    """``netmeta::netpath`` — path-by-path inconsistency (Koenig et al. 2013).

    Builds the evidence-flow network from the row of the Davies hat matrix that
    corresponds to ``node1`` vs ``node2``, enumerates every simple path between
    the two nodes, and tests whether the direct evidence along those paths is
    consistent with the network estimate.

    Args:
        result: a :class:`NetmetaResult`.
        node1, node2: the treatment contrast of interest.
        random: use the random-effects model.

    Returns:
        A dict with ``results`` (per-path ``Q``, ``df``, ``pval``),
        ``A_matrix``, ``Sigma`` and ``theta_p`` — the fields of the R
        ``netpath`` object.
    """
    if not isinstance(result, NetmetaResult):
        raise TypeError("argument 'result' must be a NetmetaResult")
    trts = list(result.trts)
    for nd in (node1, node2):
        if nd not in trts:
            raise ValueError(f"'{nd}' is not a treatment of the network")
    if node1 == node2:
        raise ValueError("'node1' and 'node2' must differ")

    sep = ":"
    hm = _hatmatrix_aggr(result, "random" if random else "common", "full")
    comp_intr = f"{node1}{sep}{node2}"
    if comp_intr not in hm["rows"]:
        comp_intr = sep.join(sorted([node1, node2]))

    # --- directed evidence-flow graph from the hat-matrix row -------------- #
    row = hm["matrix"][hm["rows"].index(comp_intr), :]
    edges = []
    for c_idx, col_label in enumerate(hm["cols"]):
        parts = col_label.split(sep)
        if len(parts) != 2:
            continue
        i, j = parts
        val = row[c_idx]
        if val > 0:
            edges.append((i, j))
        elif val < 0:
            edges.append((j, i))

    all_paths = _all_simple_paths(edges, node1, node2)
    if not all_paths:
        return {
            "comparison": comp_intr, "results": [], "A_matrix": None,
            "Sigma": None, "theta_p": None, "note": "No path found",
            "random": random, "node1": node1, "node2": node2,
            "trts": trts, "sep.trts": sep,
        }

    edge_sets = [_get_edges_from_path(p, sep) for p in all_paths]
    npth = len(edge_sets)
    A = np.zeros((npth, npth))
    for i in range(npth):
        for j in range(npth):
            if i == j:
                A[i, j] = len(edge_sets[i])
            else:
                A[i, j] = len(set(edge_sets[i]) & set(edge_sets[j]))

    # --- drop linearly dependent paths (reduce_to_full_rank) --------------- #
    if abs(np.linalg.det(A)) < 1e-6:
        _, _, piv = scipy.linalg.qr(A, pivoting=True)
        rank = int(np.linalg.matrix_rank(A, tol=1e-8))
        kept = sorted(int(p) for p in piv[:rank])
    else:
        kept = list(range(npth))

    kept_paths = [all_paths[k] for k in kept]
    kept_edges = [_get_edges_from_path(p, sep) for p in kept_paths]

    comp_edges = _comparison_list(trts)
    se_dir = np.asarray(result.seTE_direct_common, dtype=float)
    te_dir = np.asarray(result.TE_direct_common, dtype=float)
    v_diag = []
    for ce in comp_edges:
        i, j = (trts.index(x) for x in ce.split(sep))
        v_diag.append(se_dir[i, j] ** 2)

    idx_c = [k for k, v in enumerate(v_diag) if not math.isnan(v)]
    C = np.zeros((len(kept), len(comp_edges)))
    for i, es in enumerate(kept_edges):
        for j, ce in enumerate(comp_edges):
            if ce in es:
                C[i, j] = 1.0
    C = C[:, idx_c]
    V = np.diag([v_diag[k] for k in idx_c])

    Sigma = C @ V @ C.T
    if Sigma.shape[0] > 0 and abs(np.linalg.det(Sigma)) < 1e-6:
        Sigma_inv = np.linalg.pinv(Sigma)
        note = "Matrix Sigma singular, pseudo-inverse used"
    else:
        Sigma_inv = np.linalg.inv(Sigma)
        note = ""

    theta_p = []
    for p in kept_paths:
        s = 0.0
        for k in range(len(p) - 1):
            a, b = p[k], p[k + 1]
            v = te_dir[trts.index(a), trts.index(b)]
            if not math.isnan(v):
                s += v
        theta_p.append(s)
    theta_p = np.asarray(theta_p, dtype=float)

    i1, i2 = trts.index(node1), trts.index(node2)
    theta_diff = theta_p - float(np.asarray(result.TE_common)[i1, i2])
    Q = float(theta_diff @ Sigma_inv @ theta_diff)
    df = int(Sigma.shape[0] - 1)
    pval = float(stats.chi2.sf(Q, df)) if df > 0 else float("nan")

    return {
        "comparison": comp_intr,
        "results": [{"path_index": kept[k] + 1, "path": kept_paths[k],
                     "Q": Q, "df": df, "pval": pval, "note": note}
                    for k in range(len(kept))],
        "A_matrix": A,
        "Sigma": Sigma,
        "theta_p": theta_p,
        "note": note,
        "random": random,
        "node1": node1,
        "node2": node2,
        "trts": trts,
        "sep.trts": sep,
    }


# --------------------------------------------------------------------------- #
# netmeta::bySummary / netmatrix / netcomparison / netbind
# --------------------------------------------------------------------------- #

def _by_summary(x, index1, index2=None, func="mode", ties_method="first",
                sep=":", rng=None):
    """``netmeta:::bySummary`` with ``long = FALSE``.

    Returns a dict of columns ``indices``, ``sum``, ``min``, ``max``,
    ``mean``, ``median``, ``mode`` — one row per distinct index (pair).
    """
    xv = np.asarray(x, dtype=float).ravel()
    idx = ([f"{a}{sep}{b}" for a, b in zip(index1, index2)]
           if index2 is not None else [str(v) for v in index1])
    keys = list(dict.fromkeys(idx))
    out = {"indices": keys, "sum": [], "min": [], "max": [],
           "mean": [], "median": [], "mode": []}
    for key in keys:
        vals = xv[[i for i, v in enumerate(idx) if v == key]]
        vals = vals[~np.isnan(vals)]
        if vals.size == 0:
            for f in ("sum", "min", "max", "mean", "median", "mode"):
                out[f].append(float('nan'))
            continue
        out["sum"].append(float(np.sum(vals)))
        out["min"].append(float(np.min(vals)))
        out["max"].append(float(np.max(vals)))
        out["mean"].append(float(np.mean(vals)))
        out["median"].append(float(np.median(vals)))
        uniq, counts = np.unique(vals, return_counts=True)
        best = np.flatnonzero(counts == counts.max())
        if best.size > 1:
            if ties_method == "last":
                pick = best[-1]
            elif ties_method == "random":
                pick = int(rng.choice(best)) if rng is not None else best[0]
            else:
                pick = best[0]
        else:
            pick = best[0]
        out["mode"].append(float(uniq[pick]))
    return out


class NetmatrixResult:
    """Result of :func:`netmatrix` — a treatment x treatment summary matrix."""

    __slots__ = ("matrix", "trts", "func", "levels", "labels")

    def __init__(self, matrix, trts, func, levels, labels):
        self.matrix = matrix
        self.trts = trts
        self.func = func
        self.levels = levels
        self.labels = labels

    def __repr__(self):
        return f"NetmatrixResult(n={len(self.trts)}, func={self.func!r})"


def netmatrix(result, var, levels=None, labels=None, func="mode",
              ties_method="first", seed=None):
    """``netmeta::netmatrix`` — summarise a study-level variable per comparison.

    Builds an ``n x n`` matrix holding, for every pair of treatments that is
    compared directly, a summary (``"mode"``, ``"min"``, ``"max"``, ``"mean"``,
    ``"median"`` or ``"sum"``) of a study-level variable.  Unobserved pairs and
    the diagonal are ``NaN``.

    Args:
        result: a :class:`NetmetaResult`.
        var: one value per study/comparison.
        levels, labels: optional recoding of the summary values.
        func: summary function name.
        ties_method: ``"first"``, ``"last"`` or ``"random"`` (for the mode).
        seed: RNG seed used when ``ties_method="random"``.

    Returns:
        :class:`NetmatrixResult`.
    """
    if func not in ("mode", "min", "max", "mean", "median", "sum"):
        raise ValueError(
            "func must be one of mode/min/max/mean/median/sum"
        )
    if ties_method not in ("first", "last", "random"):
        raise ValueError("ties_method must be 'first', 'last' or 'random'")

    trts = list(result.trts)
    t1 = [str(v) for v in result.treat1_raw]
    t2 = [str(v) for v in result.treat2_raw]
    n = len(t1)
    v = np.asarray(var, dtype=float).ravel()
    if v.size == 1 and n > 1:
        v = np.repeat(v, n)
    if v.size != n:
        raise ValueError(
            f"length of 'var' ({v.size}) does not match the number of "
            f"comparisons ({n})"
        )

    rng = np.random.default_rng(seed) if seed is not None else None
    summ = _by_summary(v, t1, t2, func="mode", ties_method=ties_method,
                       sep=":", rng=rng)
    vals = np.asarray(summ[func], dtype=float)

    if levels is not None:
        if labels is None:
            labels = levels
        if len(levels) != len(labels):
            raise ValueError(
                "Different lengths of arguments 'levels' and 'labels'."
            )
        mapped = []
        for cur in vals:
            hit = [k for k, lv in enumerate(levels) if float(lv) == cur]
            mapped.append(labels[hit[0]] if hit else float('nan'))
        vals = np.asarray([float(m) if isinstance(m, (int, float)) else m
                           for m in mapped], dtype=object)

    mat = np.full((len(trts), len(trts)), float('nan'), dtype=object)
    pos = {t: i for i, t in enumerate(trts)}
    for j, key in enumerate(summ["indices"]):
        a, b = key.split(":", 1)
        if a not in pos or b not in pos:
            continue
        mat[pos[a], pos[b]] = vals[j]
        mat[pos[b], pos[a]] = vals[j]
    try:
        mat = mat.astype(float)
    except (TypeError, ValueError):
        pass
    return NetmatrixResult(mat, trts, func, levels, labels)


class NetcomparisonResult:
    """Result of :func:`netcomparison` (``netmeta::netcomparison``)."""

    __slots__ = ("comparison", "treat1", "treat2", "TE_common", "seTE_common",
                 "lower_common", "upper_common", "statistic_common",
                 "pval_common", "TE_random", "seTE_random", "lower_random",
                 "upper_random", "statistic_random", "pval_random",
                 "common", "random", "level", "trts", "comps", "inactive",
                 "B_matrix", "C_matrix", "add1", "add2")

    def __init__(self, **kw):
        for k in self.__slots__:
            setattr(self, k, kw.get(k))

    def __repr__(self):
        return f"NetcomparisonResult(k={len(self.comparison or [])})"

    def as_dict(self):
        return {k.replace("_", ".").replace("..", "."): getattr(self, k)
                for k in self.__slots__}


def netcomparison(result, treat1=None, treat2=None, common=True, random=True,
                  level=0.95, nchar_comps=None, backtransf=False):
    """``netmeta::netcomparison`` — estimate comparisons between combinations.

    Works on a :class:`NetcombResult`: the requested ``treat1`` vs ``treat2``
    combinations (e.g. ``"A+B"`` vs ``"plac"``) are expressed as contrasts of
    the estimated component effects.

    Args:
        result: a :class:`NetcombResult`.
        treat1, treat2: combinations to compare.  Defaults mirror R: all
            treatments vs the reference group.
        common, random: which models to report.
        level: confidence level.
        nchar_comps: component-name abbreviation length (unused, kept for API
            compatibility).
        backtransf: back-transform the estimates.

    Returns:
        :class:`NetcomparisonResult`.
    """
    if not isinstance(result, NetcombResult):
        raise TypeError("argument 'result' must be a NetcombResult")

    trts = list(result.trts)
    comps = list(result.comps)
    inactive = list(result.inactive or [])
    sep_comps = result.sep_comps or "+"
    ref = result.reference_group or ""

    t1_orig, t2_orig = treat1, treat2
    if treat1 is None:
        treat1 = list(trts)
    elif isinstance(treat1, str):
        treat1 = [treat1]
    else:
        treat1 = list(treat1)
    if treat2 is None:
        treat2 = ref
        if treat2 == "":
            treat2 = treat1[0]
            treat1 = treat1[1:]
        else:
            treat1 = [t for t in treat1 if t != treat2]
    elif isinstance(treat2, str):
        treat2 = [treat2]
    else:
        treat2 = list(treat2)
    if len(treat1) == 1 and len(treat2) > 1:
        treat1 = treat1 * len(treat2)
    elif len(treat1) > 1 and len(treat2) == 1:
        treat2 = treat2 * len(treat1)
    if len(treat1) != len(treat2):
        raise ValueError("treat1 and treat2 must have the same length")

    ncomp = len(treat1)
    # reuse the C matrix of the netcomb fit (rows = combinations in result order)
    cnames = list(result.comps)
    cmat = np.asarray(result.C_matrix, dtype=float)
    if cmat.shape[1] != len(cnames):
        cmat, cnames, _ = _create_c_matrix(
            sorted(set(treat1) | set(treat2)),
            (result.inactive or [None])[0]
            if isinstance(result.inactive, list) else result.inactive,
            sep_comps,
        )

    b = np.zeros((ncomp, len(trts)))
    tpos = {t: i for i, t in enumerate(trts)}
    for i in range(ncomp):
        if treat1[i] in tpos:
            b[i, tpos[treat1[i]]] += 1.0
        if treat2[i] in tpos:
            b[i, tpos[treat2[i]]] -= 1.0
    x = b @ np.asarray(cmat, dtype=float)

    comp_c = np.asarray(result.Comp_common, dtype=float)
    comp_r = np.asarray(result.Comp_random, dtype=float)
    comp_na = np.isnan(comp_c)
    if comp_na.any():
        inest = np.abs(x[:, comp_na]).sum(axis=1) > 0
        x_use = x[:, ~comp_na]
    else:
        inest = np.zeros(ncomp, dtype=bool)
        x_use = x

    lplus_c = np.asarray(result.Lplus_matrix_common, dtype=float)
    lplus_r = np.asarray(result.Lplus_matrix_random, dtype=float)
    if comp_na.any():
        lplus_c = lplus_c[np.ix_(~comp_na, ~comp_na)]
        lplus_r = lplus_r[np.ix_(~comp_na, ~comp_na)]

    te_c = x_use @ comp_c[~comp_na]
    se_c = np.sqrt(np.diag(x_use @ lplus_c @ x_use.T))
    te_r = x_use @ comp_r[~comp_na]
    se_r = np.sqrt(np.diag(x_use @ lplus_r @ x_use.T))

    te_c[inest] = float('nan')
    se_c[inest] = float('nan')
    te_r[inest] = float('nan')
    se_r[inest] = float('nan')

    zc = te_c / se_c
    zr = te_r / se_r
    crit = float(stats.norm.isf((1.0 - level) / 2.0))
    # R labels each comparison by the components that differ between the arms
    comparison = []
    for a, b in zip(treat1, treat2):
        sa = set(_compsplit(a, sep_comps))
        sb = set(_compsplit(b, sep_comps))
        only1 = [c for c in _compsplit(a, sep_comps) if c not in sb]
        only2 = [c for c in _compsplit(b, sep_comps) if c not in sa]
        comb1 = sep_comps.join(only1)
        comb2 = sep_comps.join(only2)
        if comb1 == "":
            comparison.append(comb2)
        elif comb2 == "":
            comparison.append(comb1)
        else:
            comparison.append(f"{comb1}{sep_comps}{comb2}")

    return NetcomparisonResult(
        comparison=comparison, treat1=list(treat1), treat2=list(treat2),
        TE_common=[float(v) for v in te_c],
        seTE_common=[float(v) for v in se_c],
        lower_common=[float(v) for v in te_c - crit * se_c],
        upper_common=[float(v) for v in te_c + crit * se_c],
        statistic_common=[float(v) for v in zc],
        pval_common=[float(v) for v in 2 * stats.norm.sf(np.abs(zc))],
        TE_random=[float(v) for v in te_r],
        seTE_random=[float(v) for v in se_r],
        lower_random=[float(v) for v in te_r - crit * se_r],
        upper_random=[float(v) for v in te_r + crit * se_r],
        statistic_random=[float(v) for v in zr],
        pval_random=[float(v) for v in 2 * stats.norm.sf(np.abs(zr))],
        common=common, random=random, level=level, trts=trts,
        comps=[c for j, c in enumerate(cnames) if cmat[:, j].any()],
        inactive=inactive, B_matrix=b, C_matrix=np.asarray(cmat, dtype=float),
        add1=t1_orig, add2=t2_orig,
    )


class NetbindResult:
    """Result of :func:`netbind` (``netmeta::netbind``)."""

    __slots__ = ("common", "random", "sm", "reference_group",
                 "baseline_reference", "names", "nets", "fixed")

    def __init__(self, common, random, sm, reference_group,
                 baseline_reference, names, nets):
        self.common = common
        self.random = random
        self.fixed = common
        self.sm = sm
        self.reference_group = reference_group
        self.baseline_reference = baseline_reference
        self.names = names
        self.nets = nets

    def __repr__(self):
        return (f"NetbindResult(nets={len(self.names)}, "
                f"common_rows={len(self.common)})")

    def as_dict(self):
        return {"common": self.common, "random": self.random,
                "fixed": self.common, "sm": self.sm}


def netbind(*results, name=None, common=None, random=None,
            reference_group=None, baseline_reference=True):
    """``netmeta::netbind`` — collect reference-comparison estimates.

    For each network the estimates of every treatment against
    ``reference_group`` are extracted from the common-effect and random-effects
    matrices and stacked into one long table (used to draw several networks in
    a single forest plot).

    Args:
        *results: :class:`NetmetaResult` objects.
        name: labels for the networks (default ``"Network 1"``, ...).
        common, random: restrict the output to one model.
        reference_group: reference treatment (default: the first network's).
        baseline_reference: keep the reference as the *column* (``True``) or as
            the *row* (``False``).

    Returns:
        :class:`NetbindResult`.
    """
    nets = list(results)
    if not nets:
        raise ValueError("argument 'result' is required")
    for r in nets:
        if not isinstance(r, NetmetaResult):
            raise TypeError("all arguments must be NetmetaResult objects")

    if name is None:
        name = [f"Network {i + 1}" for i in range(len(nets))]
    elif isinstance(name, str):
        name = [name]
    name = list(name)
    if len(name) != len(nets):
        raise ValueError("length of 'name' must match the number of networks")

    ref = reference_group
    if ref is None:
        ref = getattr(nets[0], "reference_group", "") or ""
    if ref == "":
        ref = list(nets[0].trts)[0]

    want_common = True if common is None else bool(common)
    want_random = True if random is None else bool(random)

    def _rows(model):
        out = []
        for k, r in enumerate(nets):
            trts = list(r.trts)
            if ref not in trts:
                raise ValueError(
                    f"reference treatment '{ref}' is not in network "
                    f"'{name[k]}'"
                )
            j = trts.index(ref)
            if model == "common":
                te = np.asarray(r.TE_common, dtype=float)
                se = np.asarray(r.seTE_common, dtype=float)
                lo = np.asarray(r.lower_common, dtype=float)
                up = np.asarray(r.upper_common, dtype=float)
                st = np.asarray(r.statistic_common, dtype=float)
                pv = np.asarray(r.pval_common, dtype=float)
            else:
                te = np.asarray(r.TE_random, dtype=float)
                se = np.asarray(r.seTE_random, dtype=float)
                lo = np.asarray(r.lower_random, dtype=float)
                up = np.asarray(r.upper_random, dtype=float)
                st = np.asarray(r.statistic_random, dtype=float)
                pv = np.asarray(r.pval_random, dtype=float)
            for i, t in enumerate(trts):
                out.append({
                    "name": name[k], "treat": t,
                    "TE": float(te[i, j] if baseline_reference else te[j, i]),
                    "seTE": float(se[i, j] if baseline_reference else se[j, i]),
                    "lower": float(lo[i, j] if baseline_reference else lo[j, i]),
                    "upper": float(up[i, j] if baseline_reference else up[j, i]),
                    "statistic": float(st[i, j] if baseline_reference
                                       else st[j, i]),
                    "pval": float(pv[i, j] if baseline_reference else pv[j, i]),
                    "method": str(r.method_tau),
                })
        return out

    return NetbindResult(
        common=_rows("common") if want_common else [],
        random=_rows("random") if want_random else [],
        sm=getattr(nets[0], "sm", ""),
        reference_group=ref,
        baseline_reference=baseline_reference,
        names=name, nets=nets,
    )


# --------------------------------------------------------------------------- #
# netmeta::crossnma2netmeta / gemtc2netmeta  (foreign-object adapters)
# --------------------------------------------------------------------------- #

def _as_dict_obj(x, what):
    """Accept a dict or an object with ``__dict__`` / ``_asdict``."""
    if isinstance(x, dict):
        return x
    if hasattr(x, "_asdict"):
        return dict(x._asdict())
    if hasattr(x, "__dict__"):
        return {k: v for k, v in vars(x).items() if not k.startswith("_")}
    raise TypeError(f"argument '{what}' must be a dict or a mapping-like object")


def crossnma2netmeta(x, sm="", method_tau="DL", reference_group="",
                     level=0.95):
    """``netmeta::crossnma2netmeta`` — convert a ``crossnma`` fit to netmeta.

    ``crossnma`` (Hamza et al.) fits a network meta-analysis that combines
    randomised and non-randomised evidence.  Its fit object carries a study
    level table with the columns ``studlab``, ``treat1``, ``treat2``, ``TE`` and
    ``seTE`` (accessed here either directly or through a ``data`` entry).

    Args:
        x: the ``crossnma`` object as a dict / mapping.
        sm: effect measure label.
        method_tau: heterogeneity estimator.
        reference_group: reference treatment.
        level: confidence level.

    Returns:
        :class:`NetmetaResult`.
    """
    obj = _as_dict_obj(x, "x")
    tab = obj.get("data", obj)
    tab = _as_dict_obj(tab, "x['data']")

    def _col(*names):
        for n in names:
            if n in tab:
                return list(tab[n])
        raise ValueError(f"could not find any of the columns {names} in 'x'")

    stud = _col("studlab", "study", "study_id")
    t1 = _col("treat1", "treatment1", "t1")
    t2 = _col("treat2", "treatment2", "t2")
    te = [float(v) for v in _col("TE", "yi", "theta")]
    se = [float(v) for v in _col("seTE", "sei", "se")]

    return netmeta(te, se, t1, t2, stud, sm=obj.get("sm", sm) or sm,
                   level=level, method_tau=method_tau,
                   reference_group=reference_group or obj.get("reference.group", ""))


def gemtc2netmeta(x, sm="", method_tau="DL", reference_group="",
                  level=0.95):
    """``netmeta::gemtc2netmeta`` — convert a ``gemtc`` network/fit to netmeta.

    ``gemtc`` (van Valkenhoef & Kuiper) stores the evidence either arm-based
    (``data.ab``: one row per arm with ``study``, ``treatment`` and the outcome
    columns) or contrast-based (``data.re``: ``study``, ``treatment``,
    ``diff``, ``std.err``).  Contrast data are used directly; arm-based data are
    converted into consecutive within-study contrasts.

    Args:
        x: the ``gemtc`` object as a dict / mapping (a ``network`` or a fit).
        sm, method_tau, reference_group, level: forwarded to ``netmeta``.

    Returns:
        :class:`NetmetaResult`.
    """
    obj = _as_dict_obj(x, "x")
    net = _as_dict_obj(obj.get("network", obj), "x['network']")

    te, se, t1, t2, stud = [], [], [], [], []

    def _find(tab, *names):
        for n in names:
            if n in tab:
                return list(tab[n])
        return None

    re_tab = net.get("data.re") or obj.get("data.re")
    ab_tab = net.get("data.ab") or obj.get("data.ab")

    if re_tab:
        tab = _as_dict_obj(re_tab, "x['network']['data.re']")
        s = _find(tab, "study", "studlab")
        tr = _find(tab, "treatment", "treat")
        d = _find(tab, "diff", "TE", "mean")
        e = _find(tab, "std.err", "se", "seTE")
        if s is None or tr is None or d is None or e is None:
            raise ValueError("data.re must contain study/treatment/diff/std.err")
        # gemtc contrast rows are relative to the study's baseline treatment
        by_study = {}
        for i in range(len(s)):
            by_study.setdefault(str(s[i]), []).append(i)
        for st, idx in by_study.items():
            base = str(tr[idx[0]])
            for i in idx[1:]:
                t1.append(str(tr[i]))
                t2.append(base)
                te.append(float(d[i]))
                se.append(float(e[i]))
                stud.append(st)
    elif ab_tab:
        tab = _as_dict_obj(ab_tab, "x['network']['data.ab']")
        s = _find(tab, "study", "studlab")
        tr = _find(tab, "treatment", "treat")
        if s is None or tr is None:
            raise ValueError("data.ab must contain study and treatment columns")
        keys = [k for k in tab if k not in ("study", "studlab", "treatment",
                                            "treat")]
        if not keys:
            raise ValueError("data.ab must contain an outcome column")
        out = np.asarray(tab[keys[0]], dtype=float)
        by_study = {}
        for i in range(len(s)):
            by_study.setdefault(str(s[i]), []).append(i)
        for st, idx in by_study.items():
            if len(idx) < 2:
                continue
            base = idx[0]
            for i in idx[1:]:
                t1.append(str(tr[i]))
                t2.append(str(tr[base]))
                te.append(float(out[i] - out[base]))
                se.append(float("nan"))
                stud.append(st)
        if any(math.isnan(v) for v in se):
            raise ValueError(
                "arm-based gemtc data need a variance column to be converted; "
                "supply data.re or fitted estimates instead"
            )
    else:
        raise ValueError(
            "could not find 'data.re' or 'data.ab' in the gemtc network object"
        )

    return netmeta(te, se, t1, t2, stud, sm=sm, level=level,
                   method_tau=method_tau, reference_group=reference_group)


# --------------------------------------------------------------------------- #
# netmeta::treats / vikor / subgroup / nettable  + public helpers
# --------------------------------------------------------------------------- #

def treats(x, nchar_trts=8, row=True):
    """``netmeta::treats`` — abbreviate treatment labels to ``nchar_trts`` chars.

    Args:
        x: a list of treatment labels, or a matrix whose row/column names are
            the treatments.
        nchar_trts: maximum label length.
        row: use the row names when ``x`` is a matrix.

    Returns:
        list of abbreviations.  When truncation would create duplicates the
        labels are abbreviated in R's ``abbreviate()`` style (``a1``, ``b1``,
        ``a2``, ...).
    """
    a = np.asarray(x, dtype=object)
    if a.ndim == 2:
        names = list(x.index if row and hasattr(x, "index") else
                     (x.columns if hasattr(x, "columns") else []))
        if not names:
            names = [str(v) for v in range(a.shape[0])]
    else:
        names = [str(v) for v in a.ravel()]
    res = [s[:int(nchar_trts)] for s in names]
    if len(set(res)) != len(set(names)):
        # R's abbreviate(): build a short prefix + digit suffix per group
        seen = {}
        out = []
        for s in names:
            key = s.lower()
            prefix = key[:int(nchar_trts)]
            if prefix not in seen:
                seen[prefix] = 0
            seen[prefix] += 1
            out.append(f"{prefix}{seen[prefix]}")
        res = out
    return res


class VikorResult:
    """Result of :func:`vikor` — VIKOR ranking statistics."""

    __slots__ = ("Q", "S", "R", "trts", "weights", "v", "dropped")

    def __init__(self, Q, S, R, trts, weights, v, dropped):
        self.Q = Q
        self.S = S
        self.R = R
        self.trts = trts
        self.weights = weights
        self.v = v
        self.dropped = dropped

    def __repr__(self):
        return f"VikorResult(n={len(self.trts)})"

    def as_dict(self):
        return {"Q": self.Q, "S": self.S, "R": self.R}

    def to_frame(self):
        try:
            import pandas as pd
        except ImportError:  # pragma: no cover
            return [{"treat": t, "Q": self.Q[t], "S": self.S[t],
                     "R": self.R[t]} for t in self.trts]
        return pd.DataFrame({"Q": self.Q, "S": self.S, "R": self.R})


def vikor(x, weights=None, v=0.5):
    """``netmeta::vikor`` — VIKOR multicriteria ranking.

    ``x`` holds one score in ``[0, 1]`` per treatment and outcome (higher is
    better).  Equations (transcribed from ``netmeta:::vikor_internal``)::

        dist = 1 - x;  wnm = dist * weights
        S_i  = sum_j wnm_ij            (group utility)
        R_i  = max_j wnm_ij            (individual regret)
        Q_i  = v·(S_i - min S)/(max S - min S)
               + (1 - v)·(R_i - min R)/(max R - min R)

    Args:
        x: ``NetmetaResult`` (P-scores are used), ``pandas.DataFrame`` or a
            2-D array-like of scores; row names are the treatments.
        weights: outcome weights (default: equal).
        v: weight of the group-utility strategy (``0.5`` = balanced).

    Returns:
        :class:`VikorResult` sorted by increasing ``Q`` (best first).
    """
    if isinstance(x, NetmetaResult):
        trts = list(x.trts)
        mat = np.column_stack([np.asarray(x.pscore_common, dtype=float),
                               np.asarray(x.pscore_random, dtype=float)])
    elif hasattr(x, "to_numpy"):
        trts = list(x.index)
        mat = np.asarray(x.to_numpy(), dtype=float)
    else:
        mat = np.asarray(x, dtype=float)
        if mat.ndim == 1:
            mat = mat.reshape(-1, 1)
        trts = [str(i + 1) for i in range(mat.shape[0])]

    n_out = mat.shape[1]
    if weights is None:
        weights = np.full(n_out, 1.0 / n_out)
    else:
        weights = np.asarray(weights, dtype=float).ravel()
        if weights.size != n_out:
            raise ValueError(
                "Number of weights is different from the number of outcomes."
            )
        if np.any(weights < 0):
            raise ValueError("weights must be non-negative")
        if np.isclose(weights.sum(), 0):
            raise ValueError("Sum of weights must be larger than 0.")
        if not np.isclose(weights.sum(), 1.0):
            weights = weights / weights.sum()

    dist = 1.0 - mat
    wnm = dist * weights

    dropped = [trts[i] for i in range(mat.shape[0])
               if np.isnan(wnm[i]).any()]
    keep = [i for i in range(mat.shape[0]) if i not in
            [j for j, t in enumerate(trts) if t in dropped]]
    if len(keep) < 2:
        raise ValueError(
            "VIKOR method requires at least two treatments with complete "
            "rankings."
        )
    sub = wnm[keep]
    S = sub.sum(axis=1)
    R = sub.max(axis=1)
    if np.isclose(R.min(), R.max()):
        raise ValueError(
            "VIKOR method not applicable as all values of ranking statistic R "
            "are identical."
        )
    if np.isclose(S.min(), S.max()):
        raise ValueError(
            "VIKOR method not applicable as all values of ranking statistic S "
            "are identical."
        )
    Q = (v * (S - S.min()) / (S.max() - S.min())
         + (1 - v) * (R - R.min()) / (R.max() - R.min()))

    names = [trts[i] for i in keep]
    order = np.argsort(Q, kind="stable")
    return VikorResult(
        Q={names[i]: float(Q[i]) for i in order},
        S={names[i]: float(S[i]) for i in order},
        R={names[i]: float(R[i]) for i in order},
        trts=[names[i] for i in order],
        weights=weights.tolist(), v=float(v), dropped=dropped,
    )


class NetsubgroupResult:
    """Result of :func:`subgroup` for a netmeta object."""

    __slots__ = ("levels", "networks", "TE_common", "seTE_common",
                 "TE_random", "seTE_random", "tau2", "trts")

    def __init__(self, **kw):
        for k in self.__slots__:
            setattr(self, k, kw.get(k))

    def __repr__(self):
        return f"NetsubgroupResult(levels={self.levels})"

    def as_dict(self):
        return {"levels": self.levels, "networks": self.networks,
                "tau2": self.tau2}


def subgroup(result, subgroup, only_connected=False, method_tau=None,
             level_ma=None, common=True, random=True):
    """``netmeta::subgroup`` — refit the network within each subgroup.

    Args:
        result: a :class:`NetmetaResult`.
        subgroup: one subgroup label per **study**.
        only_connected: error out when a subgroup network is disconnected.
        method_tau: heterogeneity estimator (defaults to the model's).
        level_ma: confidence level (defaults to the model's).
        common, random: which models to record.

    Returns:
        :class:`NetsubgroupResult` with per-level 3-D arrays
        ``TE_common``/``seTE_common``/``TE_random``/``seTE_random`` of shape
        ``(n, n, n_levels)`` and the fitted networks.
    """
    trts = list(result.trts)
    n = len(trts)
    stud = [str(v) for v in result.studlab_raw]
    sg = [str(v) for v in np.asarray(subgroup).ravel()]

    # expand study-level subgroup to comparison-level
    by_study = {}
    for s, g in zip(stud, sg):
        by_study.setdefault(s, g)
    if len(by_study) != len(set(stud)):
        pass
    levels = sorted(set(sg))
    if method_tau is None:
        method_tau = result.method_tau
    if level_ma is None:
        level_ma = result.level

    te_c = np.full((n, n, len(levels)), float('nan'))
    se_c = np.full((n, n, len(levels)), float('nan'))
    te_r = np.full((n, n, len(levels)), float('nan'))
    se_r = np.full((n, n, len(levels)), float('nan'))
    tau2s = {}
    nets = {}

    t1 = list(result.treat1_raw)
    t2 = list(result.treat2_raw)
    tev = list(result.TE_raw)
    sev = list(result.seTE_raw)

    for k, lev in enumerate(levels):
        idx = [i for i, s in enumerate(stud) if by_study.get(s) == lev]
        if not idx:
            continue
        try:
            net_k = netmeta([tev[i] for i in idx], [sev[i] for i in idx],
                            [t1[i] for i in idx], [t2[i] for i in idx],
                            [stud[i] for i in idx], sm=result.sm,
                            level=level_ma, method_tau=method_tau,
                            reference_group=result.reference_group or "")
        except Exception:
            nets[lev] = None
            continue
        nets[lev] = net_k
        tau2s[lev] = net_k.tau2
        pos = {t: i for i, t in enumerate(net_k.trts)}
        for a in net_k.trts:
            for b in net_k.trts:
                i, j = pos[a], pos[b]
                gi, gj = trts.index(a), trts.index(b)
                te_c[gi, gj, k] = np.asarray(net_k.TE_common)[i, j]
                se_c[gi, gj, k] = np.asarray(net_k.seTE_common)[i, j]
                te_r[gi, gj, k] = np.asarray(net_k.TE_random)[i, j]
                se_r[gi, gj, k] = np.asarray(net_k.seTE_random)[i, j]

    return NetsubgroupResult(
        levels=levels, networks=nets, TE_common=te_c, seTE_common=se_c,
        TE_random=te_r, seTE_random=se_r, tau2=tau2s, trts=trts,
    )


class NettableResult:
    """Result of :func:`nettable` — per-comparison summary of the network."""

    __slots__ = ("rows", "trts")

    def __init__(self, rows, trts):
        self.rows = rows
        self.trts = trts

    def __repr__(self):
        return f"NettableResult(comparisons={len(self.rows)})"

    def as_dict(self):
        return {"rows": self.rows}

    def to_frame(self):
        try:
            import pandas as pd
        except ImportError:  # pragma: no cover
            return self.rows
        return pd.DataFrame(self.rows)


def nettable(result, level=0.95):
    """``netmeta::nettable`` — direct evidence per comparison.

    Returns one row per treatment comparison with the number of studies, the
    number of two-arm studies, the direct estimate and the network estimate.

    Args:
        result: a :class:`NetmetaResult`.
        level: confidence level (kept for API parity).

    Returns:
        :class:`NettableResult`.
    """
    trts = list(result.trts)
    t1 = [str(v) for v in result.treat1_raw]
    t2 = [str(v) for v in result.treat2_raw]
    tev = np.asarray(result.TE_raw, dtype=float)
    sev = np.asarray(result.seTE_raw, dtype=float)
    stud = [str(v) for v in result.studlab_raw]
    narms = np.asarray(result.narms_raw, dtype=float)

    pos = {t: i for i, t in enumerate(trts)}
    groups = {}
    for i in range(len(t1)):
        key = tuple(sorted([t1[i], t2[i]]))
        groups.setdefault(key, []).append(i)

    net_c = np.asarray(result.TE_common, dtype=float)
    net_r = np.asarray(result.TE_random, dtype=float)
    dir_c = np.asarray(result.TE_direct_common, dtype=float)
    dir_r = np.asarray(result.TE_direct_random, dtype=float)

    rows = []
    for (a, b), idx in groups.items():
        rows.append({
            "treat1": a, "treat2": b,
            "studies": len({stud[i] for i in idx}),
            "n.2arm": int(sum(1 for i in idx if narms[i] == 2)),
            "narms": int(max(narms[i] for i in idx)),
            "TE.direct.common": float(dir_c[pos[a], pos[b]]),
            "TE.direct.random": float(dir_r[pos[a], pos[b]]),
            "TE.network.common": float(net_c[pos[a], pos[b]]),
            "TE.network.random": float(net_r[pos[a], pos[b]]),
        })
    return NettableResult(rows, trts)


# --------------------------------------------------------------------------- #
# public aliases for the internal helpers (netmeta::createC / invmat /
# hatmatrix / comps / combinations)
# --------------------------------------------------------------------------- #

def createC(trts, inactive=None, sep_comps="+"):
    """``netmeta::createC`` — component incidence matrix ``(C, comps, names)``."""
    return _create_c_matrix(trls if False else trts, inactive, sep_comps)[:3]


def invmat(x):
    """``netmeta::invmat`` — Moore-Penrose pseudoinverse of a Laplacian."""
    return _invmat(np.asarray(x, dtype=float))


def hatmatrix(x, method="Davies", type="full", common=None, random=None):
    """``netmeta::hatmatrix`` — hat matrix of a network.

    Supported: ``method="Davies"`` with ``type`` in ``short``/``long``/``full``
    (via ``hatmatrix.aggr``), and ``method="Ruecker"`` (the hat matrix stored
    on the fitted object).  ``method="Krahn"`` needs the design decomposition
    and is available through :func:`decomp_design`.
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


def comps(x, sep_comps="+"):
    """``netmeta::comps`` — split treatment names into components."""
    if isinstance(x, str):
        return _compsplit(x, sep_comps)
    out = []
    for t in np.asarray(x, dtype=object).ravel():
        out.append(_compsplit(str(t), sep_comps))
    return out


def combinations(x, sep_comps="+"):
    """``netmeta::combinations`` — all treatment combinations of ``x``."""
    if isinstance(x, str):
        return _compsplit(x, sep_comps)
    vals = [str(v) for v in np.asarray(x, dtype=object).ravel()]
    return sorted(set(vals),
                  key=lambda s: (len(_compsplit(s, sep_comps)), s))
