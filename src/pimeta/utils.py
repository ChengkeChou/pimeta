"""Small metafor helpers — matrix utilities, fit statistics and accessors.

Transcribes the following ``metafor`` exports:

* ``bldiag`` / ``blsplit`` / ``vec2mat`` / ``replmiss`` / ``cmicalc``
* ``contrmat`` / ``pairmat`` / ``rcalc``
* ``fitstats`` / ``AIC.rma`` / ``BIC.rma`` / ``se.rma`` /
  ``df.residual.rma`` / ``model.matrix.rma`` / ``as.data.frame.list.rma`` /
  ``cooks.distance.rma.uni`` / ``blup.rma.uni`` / ``sigma.matreg`` /
  ``update.rma``

plus ``meta::estimates`` and the internal ``meta::.forestArgs``.
"""

from __future__ import annotations

import math

import numpy as np
from scipy import stats

from .metagen import MetagenResult

__all__ = [
    "bldiag", "blsplit", "vec2mat", "replmiss", "cmicalc",
    "contrmat", "pairmat", "rcalc",
    "fitstats", "aic_rma", "bic_rma", "se_rma", "df_residual_rma",
    "model_matrix_rma", "as_data_frame_list_rma", "cooks_distance_rma",
    "blup", "sigma_matreg", "update_rma", "estimates", "forest_args",
    "RcalcResult", "BlupResult", "FitstatsResult", "EstimatesResult",
    "as_data_frame_confint", "as_matrix_list_rma", "se",
    "as_data_frame_list_anova", "as_data_frame_list_confint",
    "as_data_frame_vif", "inches2units", "units2inches",
    "jama_labels", "ci", "getmfopt", "setmfopt",
]

_NAN = float("nan")


# --------------------------------------------------------------------------- #
# bldiag
# --------------------------------------------------------------------------- #

def bldiag(*blocks, order=None):
    """``metafor::bldiag`` — build a block-diagonal matrix.

    Args:
        *blocks: matrices (block diagonal), vectors (treated as diagonals) or a
            single list of such blocks.
        order: optional permutation (0-based indices, or a permutation vector as
            returned by R's ``order()``) applied to rows and columns of the
            resulting square matrix.

    Returns:
        ``numpy.ndarray``.
    """
    if len(blocks) == 1 and isinstance(blocks[0], (list, tuple)):
        blocks = tuple(blocks[0])

    mats = []
    for b in blocks:
        a = np.asarray(b, dtype=float)
        if a.ndim == 0:
            a = np.diag(a.reshape(1))
        elif a.ndim == 1:
            a = np.diag(a)
        elif a.ndim != 2:
            raise ValueError("each block must be a vector or a matrix")
        mats.append(a)

    mats = [m for m in mats if m.shape != (0, 0)]
    if not mats:
        return np.zeros((0, 0))

    rows = sum(m.shape[0] for m in mats)
    cols = sum(m.shape[1] for m in mats)
    out = np.zeros((rows, cols))
    r = 0
    for m in mats:
        out[r:r + m.shape[0], 0:m.shape[1]] = 0.0
        r += m.shape[0]
    r = 0
    c = 0
    for m in mats:
        out[r:r + m.shape[0], c:c + m.shape[1]] = m
        r += m.shape[0]
        c += m.shape[1]

    if order is not None:
        if out.shape[0] != out.shape[1]:
            raise ValueError("Can only use 'order' argument for square matrices.")
        idx = np.asarray(order)
        if idx.size != out.shape[0]:
            raise ValueError(
                f"Length of the 'order' argument ({idx.size}) does not "
                f"correspond to the dimensions of the matrix "
                f"({out.shape[0]}x{out.shape[1]})."
            )
        if idx.dtype == bool or idx.size == 0:
            idx = np.flatnonzero(idx)
        # R: out[sort.vec, sort.vec] <- out with sort.vec <- order(order) for a
        # plain index vector, i.e. the *inverse* permutation (one argsort on the
        # 0-based indices, not two).
        inv = np.argsort(idx)
        tmp = np.zeros_like(out)
        tmp[np.ix_(inv, inv)] = out
        out = tmp
    return out


# --------------------------------------------------------------------------- #
# blsplit
# --------------------------------------------------------------------------- #

def blsplit(x, cluster=None, fun=None, args=None, sort=False):
    """``metafor::blsplit`` — split a (symmetric) matrix by cluster.

    Args:
        x: square symmetric matrix.
        cluster: grouping variable (one entry per row/column).
        fun: optional function applied to each sub-matrix.
        args: extra positional arguments for ``fun``.
        sort: sort the cluster labels.

    Returns:
        ``dict`` mapping cluster label to sub-matrix (or to ``fun``'s result).
    """
    a = np.asarray(x, dtype=float)
    if a.ndim != 2:
        raise ValueError("Argument 'x' must be a matrix.")
    if a.shape[0] != a.shape[1]:
        raise ValueError("Argument 'x' must be a square matrix.")
    if cluster is None:
        raise ValueError("Must specify the 'cluster' variable.")
    cl = list(cluster)
    if len(cl) != a.shape[0]:
        raise ValueError(
            f"Length of the variable specified via 'cluster' ({len(cl)}) does "
            f"not correspond to the dimensions of the matrix ({a.shape[0]}x"
            f"{a.shape[1]})."
        )
    keys = list(dict.fromkeys(cl))
    if sort:
        keys = sorted(keys)
    out = {}
    for key in keys:
        idx = [i for i, v in enumerate(cl) if v == key]
        sub = a[np.ix_(idx, idx)]
        if fun is not None:
            if args is None:
                sub = fun(sub)
            else:
                sub = fun(sub, *args)
        out[key] = sub
    return out


# --------------------------------------------------------------------------- #
# vec2mat
# --------------------------------------------------------------------------- #

def vec2mat(x, diag=False, corr=None, dimnames=None):
    """``metafor::vec2mat`` — build a symmetric matrix from its lower triangle.

    Args:
        x: vector of lower-triangular entries (including the diagonal when
            ``diag=True``), in R's column-major order.
        diag: whether ``x`` includes the diagonal.
        corr: set the diagonal to 1 (default ``not diag``).
        dimnames: optional row/column labels.

    Returns:
        ``numpy.ndarray``.
    """
    if corr is None:
        corr = not diag
    vals = np.asarray(x, dtype=float).ravel()
    p = vals.size
    dims = math.sqrt(2.0 * p + 0.25) + (-0.5 if diag else 0.5)
    if abs(dims - round(dims)) >= math.sqrt(np.finfo(float).eps):
        raise ValueError("Length of 'x' does not correspond to a square matrix.")
    dims = int(round(dims))
    if dimnames is not None and len(dimnames) != dims:
        raise ValueError(
            f"Length of 'dimnames' ({len(dimnames)}) does not correspond to "
            f"the dimensions of the matrix ({dims})."
        )

    r = np.full((dims, dims), _NAN)
    rows, cols = np.tril_indices(dims, 0 if diag else -1)
    # np.tril_indices is row-major; R fills column-major -> reorder.
    order = np.lexsort((rows, cols))
    r[rows[order], cols[order]] = vals
    ut = np.triu_indices(dims, 0 if diag else 1)
    r[ut] = r.T[ut]
    if corr:
        np.fill_diagonal(r, 1.0)
    return r


# --------------------------------------------------------------------------- #
# replmiss
# --------------------------------------------------------------------------- #

def replmiss(x, y=None):
    """``metafor::replmiss`` — replace missing values of ``x`` with ``y``."""
    xv = np.asarray(x, dtype=float).ravel().copy()
    if y is None:
        return xv
    yv = np.asarray(y, dtype=float).ravel()
    if yv.size == 0:
        return xv
    if xv.size == 0:
        xv = np.full(yv.size, _NAN)
    if yv.size == 1:
        yv = np.repeat(yv, xv.size)
    if xv.size != yv.size:
        raise ValueError("Length of 'x' and 'y' are not the same.")
    na = np.isnan(xv)
    xv[na] = yv[na]
    return xv


# --------------------------------------------------------------------------- #
# cmicalc
# --------------------------------------------------------------------------- #

def cmicalc(mi, method="exact"):
    """``metafor::cmicalc`` — continuity-correction multiplier (Elston 1991).

    ``exact``:  ``exp(lgamma(mi/2) - log(sqrt(mi/2)) - lgamma((mi-1)/2))``
    ``approx``: ``1 - 3/(4*mi - 1)``
    """
    if isinstance(method, bool):
        method = "exact" if method else "none"
    method = str(method)
    if method not in ("exact", "approx"):
        raise ValueError("Correction method must be either 'exact' or 'approx'.")
    m = np.asarray(mi, dtype=float).ravel()
    if method == "exact":
        out = np.where(
            m <= 1.0, _NAN,
            np.exp(_lgamma(m / 2.0) - np.log(np.sqrt(m / 2.0))
                   - _lgamma((m - 1.0) / 2.0)),
        )
    elif method == "approx":
        out = np.where(m <= 1.0, _NAN, 1.0 - 3.0 / (4.0 * m - 1.0))
    else:
        out = np.ones_like(m)
    return out


def _lgamma(x):
    from scipy.special import gammaln
    return gammaln(x)


# --------------------------------------------------------------------------- #
# contrmat
# --------------------------------------------------------------------------- #

def contrmat(data, grp1, grp2, last=None, shorten=False, minlen=2,
             check=True, append=True):
    """``metafor::contrmat`` — treatment-contrast indicator columns.

    ``X = model.matrix(~0+grp1) - model.matrix(~0+grp2)`` over the treatment
    levels, with the ``last`` level moved to the final column.

    Args:
        data: ``pandas.DataFrame`` or dict of columns.
        grp1, grp2: column names (or 0-based positions) of the two arms.
        last: level to be placed last (default: most frequent ``grp2`` level).
        shorten, minlen: shorten the level labels (first ``minlen`` characters).
        check: run R's ``make.names`` on the resulting column names.
        append: return ``data`` with the contrast columns appended.

    Returns:
        dict of columns (or the contrast matrix when ``append=False``).
    """

    if data is None:
        raise ValueError("Argument 'data' must be specified.")
    if hasattr(data, "to_dict") and hasattr(data, "columns"):
        cols = {c: list(data[c]) for c in data.columns}
    elif isinstance(data, dict):
        cols = {k: list(v) for k, v in data.items()}
    else:
        raise TypeError("argument 'data' must be a DataFrame or a dict")

    names = list(cols)
    n = len(next(iter(cols.values()))) if cols else 0

    def _pick(g):
        if isinstance(g, str):
            if g not in names:
                raise ValueError(
                    "Could not find or uniquely identify variable specified via "
                    "the group argument."
                )
            return list(cols[g])
        pos = int(round(g))
        return list(cols[names[pos]])

    a = [str(v) for v in _pick(grp1)]
    b = [str(v) for v in _pick(grp2)]
    if any(v is None for v in a) or any(v is None for v in b):
        raise ValueError("group variables should not contain missing values.")

    lvls = sorted(set(a) | set(b))
    if last is None:
        counts = {}
        for v in b:
            counts[v] = counts.get(v, 0) + 1
        best = max(counts.values()) if counts else 0
        last = sorted(k for k, v in counts.items() if v == best)[0]
    last = str(last)
    if last not in lvls:
        raise ValueError(
            "Could not find or uniquely identify group specified via the "
            "'last' argument."
        )
    lvls = [v for v in lvls if v != last] + [last]

    x = np.zeros((n, len(lvls)))
    pos = {v: i for i, v in enumerate(lvls)}
    for i in range(n):
        x[i, pos[a[i]]] += 1.0
        x[i, pos[b[i]]] -= 1.0

    labels = lvls
    if shorten:
        labels = [v[:minlen] for v in labels]
    if check:
        labels = [_make_names(v) for v in labels]
    if not append:
        return {"matrix": x, "names": labels}
    out = dict(cols)
    for j, lab in enumerate(labels):
        out[lab] = list(x[:, j])
    return out


def _make_names(s):
    """Approximate R's ``make.names``."""
    import re
    t = re.sub(r"[^0-9a-zA-Z._]", ".", s)
    if not t or not (t[0].isalpha() or t[0] == "."):
        t = "X" + t
    return t


# --------------------------------------------------------------------------- #
# pairmat
# --------------------------------------------------------------------------- #

def pairmat(result, btt=None, fixed=False):
    """``metafor::pairmat`` — pairwise-comparison contrast matrix.

    Args:
        result: a :class:`~.rma_uni.RmaResult`.
        btt: coefficients to compare (1-based like R).
        fixed: include the coefficient names as stored.

    Returns:
        ``(matrix, rownames)``.
    """
    if getattr(result, "int_only", False):
        raise ValueError(
            "Cannot construct contrast matrices for intercept-only models."
        )
    p = int(result.p)
    if btt is None:
        raise ValueError("Need to specify the 'btt' argument.")
    idx = [int(v) - 1 for v in np.asarray(btt).ravel()]
    names = _coef_names(result)
    k = len(idx)
    if k == 1:
        raise ValueError(
            "Need to specify multiple coefficients via argument 'btt' for "
            "pairwise comparisons."
        )
    rows = k * (k - 1) // 2
    x = np.zeros((rows, p))
    connames = []
    r = 0
    for i in range(k - 1):
        for j in range(i + 1, k):
            x[r, idx[i]] = -1.0
            x[r, idx[j]] = +1.0
            connames.append(f"{names[idx[i]]}-{names[idx[j]]}")
            r += 1
    return {"matrix": x, "names": connames}


def _coef_names(result):
    """R-style coefficient labels for a fitted model (``intrcpt``, ``mod1``...)."""
    p = int(result.p)
    mods = getattr(result, "mods", None)
    mod_names: list[str] = []
    if isinstance(mods, (list, tuple)):
        if mods and all(isinstance(m, str) for m in mods):
            mod_names = [str(m) for m in mods]
        else:
            mod_names = [f"mod{i + 1}" for i in range(len(mods))]
    elif isinstance(mods, str):
        mod_names = [m.strip() for m in mods.replace("+", " ").split() if m.strip()]
    elif mods is not None:
        arr = np.asarray(mods)
        n_mods = 1 if arr.ndim <= 1 else int(arr.shape[1])
        mod_names = [f"mod{i + 1}" for i in range(n_mods)]
    int_incl = bool(getattr(result, "int_incl", True))
    names = (["intrcpt"] if int_incl else []) + mod_names
    while len(names) < p:
        names.append(f"V{len(names) + 1}")
    return names[:p]


# --------------------------------------------------------------------------- #
# rcalc
# --------------------------------------------------------------------------- #

class RcalcResult:
    """Result of :func:`rcalc` — ``dat`` (long table) and ``V`` (covariance)."""

    __slots__ = ("dat", "V", "row_labels")

    def __init__(self, dat, V):
        self.dat = dat
        self.V = V
        self.row_labels = []

    def __repr__(self):
        n = 0 if self.dat is None else len(self.dat["yi"])
        return f"RcalcResult(m={n})"

    def as_dict(self):
        return {"dat": self.dat, "V": self.V}


def rcalc(x, ni, data=None, rtoz=False, nfun="min", upper=False,
          simplify=True, rowid=None, vnames=None, noid=False, dnames=None):
    """``metafor::rcalc`` — turn correlation matrices into effect sizes + V.

    Args:
        x: a square correlation matrix, a list of them, or a ``pandas``
            ``DataFrame`` with columns ``(ri, var1, var2, id)``.
        ni: sample size (scalar) or one per matrix / study.
        rtoz: apply Fisher's r-to-z transformation.
        nfun: how to summarise ``ni`` within a study (``"min"``,
            ``"harmonic"``, ``"mean"``).
        upper: read the upper rather than the lower triangle.
        simplify: concatenate the per-study results.
        rowid: reorder the rows of ``dat``/``V``.
        vnames: names for the four data columns.
        noid: drop the study-id column.

    Returns:
        :class:`RcalcResult` (or a list of them when ``simplify=False``).
    """
    if isinstance(x, list):
        if ni is None or len(x) != len(np.asarray(ni).ravel()):
            raise ValueError(
                "Argument 'ni' must be of the same length as there are "
                "elements in 'x'."
            )
        ni_v = np.asarray(ni, dtype=float).ravel()
        res = []
        for i, xi in enumerate(x):
            dn = None
            if dnames is not None:
                dn = dnames[i] if isinstance(dnames, (list, tuple)) and \
                    dnames and isinstance(dnames[0], (list, tuple)) else dnames
            res.append(rcalc(xi, ni_v[i], rtoz=rtoz, nfun=nfun, upper=upper,
                             simplify=True, dnames=dn))
        if not simplify:
            return res
        dat = {"id": [], "var1": [], "var2": [], "var1.var2": [],
               "yi": [], "ni": []}
        blocks = []
        labels = []
        for k, r in enumerate(res):
            for key in ("var1", "var2", "var1.var2", "yi", "ni"):
                dat[key].extend(r.dat[key])
            dat["id"].extend([k] * len(r.dat["yi"]))
            blocks.append(r.V)
            labels.extend(getattr(r, "row_labels", []))
        V = bldiag(blocks) if blocks else np.zeros((0, 0))
        if rowid is not None:
            keys = [f"{a}.{b}.{a}.{c}" for a, b, c in
                    zip(dat["id"], dat["var1"], dat["var2"])]
            idx = [keys.index(str(rv)) for rv in rowid]
            dat = {k: [v[i] for i in idx] for k, v in dat.items()}
            V = V[np.ix_(idx, idx)]
        if vnames is not None:
            pass
        if noid:
            dat.pop("id", None)
        return RcalcResult(dat, V)

    if isinstance(x, dict):
        # formula-style input: ri, var1, var2, [id]
        ri = [float(v) for v in x["ri"]]
        v1 = [str(v) for v in x["var1"]]
        v2 = [str(v) for v in x["var2"]]
        ids = [str(v) for v in x["id"]] if "id" in x else None
        ni_v = np.asarray(ni, dtype=float).ravel()
        groups = list(dict.fromkeys(ids)) if ids else ["1"]
        mats, sizes, per_dnames = [], [], []
        for g in groups:
            sel = [i for i in range(len(ri)) if ids is None or ids[i] == g]
            vars_ = sorted({v1[i] for i in sel} | {v2[i] for i in sel})
            m = np.full((len(vars_), len(vars_)), _NAN)
            np.fill_diagonal(m, 1.0)
            pi = {v: i for i, v in enumerate(vars_)}
            for i in sel:
                if v1[i] == v2[i]:
                    raise ValueError(f"Identical var1-var2 pair in study {g}.")
                m[pi[v1[i]], pi[v2[i]]] = ri[i]
                m[pi[v2[i]], pi[v1[i]]] = ri[i]
            mats.append(m)
            sizes.append(float(np.min(ni_v[sel])))
            per_dnames.append(vars_)
        out = rcalc(mats, sizes, rtoz=rtoz, simplify=simplify,
                    dnames=per_dnames)
        return out

    a = np.asarray(x, dtype=float)
    if a.ndim != 2 or a.shape[0] != a.shape[1]:
        raise ValueError("Argument 'x' must be a square matrix.")
    dims = a.shape[0]
    if upper not in (True, False) or rtoz not in (True, False):
        raise ValueError("Arguments 'rtoz' and 'upper' must be TRUE or FALSE.")
    dnames = ([str(v) for v in dnames] if dnames is not None
              else [f"x{i + 1}" for i in range(dims)])
    if len(dnames) != dims:
        raise ValueError(
            f"length of 'dnames' ({len(dnames)}) does not match 'x' ({dims})")
    if dims <= 1:
        return RcalcResult(None, None)

    a = a.copy()
    if upper:
        lt = np.tril_indices(dims, -1)
        a[lt] = a.T[lt]
        tri = np.triu_indices(dims, 1)
    else:
        ut = np.triu_indices(dims, 1)
        a[ut] = a.T[ut]
        tri = np.tril_indices(dims, -1)
    # R enumerates triangles column-major
    order = np.lexsort((tri[0], tri[1]))
    rr, cc = tri[0][order], tri[1][order]
    ri = a[rr, cc]
    if np.any(np.abs(ri) > 1):
        raise ValueError("One or more correlations are > 1 or < -1.")
    n_val = float(np.asarray(ni, dtype=float).ravel()[0])
    if n_val <= 0:
        raise ValueError("One or more sample sizes are <= 0.")
    if rtoz:
        ri = np.arctanh(ri)

    # R: I <- matrix(seq_len(dims), dims, dims)  (recycled -> row index)
    #    J <- matrix(seq_len(dims), dims, dims, byrow = TRUE)  (-> column index)
    im = np.broadcast_to(np.arange(1, dims + 1)[:, None], (dims, dims))
    jm = np.broadcast_to(np.arange(1, dims + 1)[None, :], (dims, dims))
    sel_i = im[tri[0][order], tri[1][order]]
    sel_j = jm[tri[0][order], tri[1][order]]

    m = ri.size
    V = np.full((m, m), _NAN)
    for ro in range(m):
        i = sel_i[ro] - 1
        j = sel_j[ro] - 1
        for co in range(m):
            k = sel_i[co] - 1
            l = sel_j[co] - 1
            v = (0.5 * a[i, j] * a[k, l]
                 * (a[i, k] ** 2 + a[i, l] ** 2 + a[j, k] ** 2 + a[j, l] ** 2)
                 + a[i, k] * a[j, l] + a[i, l] * a[j, k]
                 - (a[i, j] * a[i, k] * a[i, l]
                    + a[j, i] * a[j, k] * a[j, l]
                    + a[k, i] * a[k, j] * a[k, l]
                    + a[l, i] * a[l, j] * a[l, k]))
            if rtoz:
                v = v / ((1 - a[i, j] ** 2) * (1 - a[k, l] ** 2))
            V[ro, co] = v
    if n_val >= 5:
        V = V / ((n_val - 3) if rtoz else (n_val - 1))
    else:
        V = V * _NAN

    var1 = [dnames[i] for i in sel_i - 1]
    var2 = [dnames[j] for j in sel_j - 1]
    pairs = [tuple(sorted([u, v])) for u, v in zip(var1, var2)]
    var1 = [p[0] for p in pairs]
    var2 = [p[1] for p in pairs]
    labels = [f"{u}.{v}" for u, v in zip(var1, var2)]

    dat = {"var1": var1, "var2": var2, "var1.var2": labels,
           "yi": [float(v) for v in ri], "ni": [n_val] * m}
    out = RcalcResult(dat, V)
    out.row_labels = labels
    return out


# --------------------------------------------------------------------------- #
# fit statistics
# --------------------------------------------------------------------------- #

class FitstatsResult:
    """Fit statistics of a constant- or random-effects model.

    Mirrors the ``$fit.stats`` matrix of a ``rma`` object: rows ``ll``, ``dev``,
    ``AIC``, ``BIC``, ``AICc``; columns ``ML`` and ``REML``.
    """

    __slots__ = ("ML", "REML", "method", "parms", "k", "p")

    def __init__(self, ML, REML, method, parms, k, p):
        self.ML = ML
        self.REML = REML
        self.method = method
        self.parms = parms
        self.k = k
        self.p = p

    def __repr__(self):
        return f"FitstatsResult(method={self.method!r}, parms={self.parms})"

    def column(self, REML=None):
        """Return the selected column as ``{rowname: value}``."""
        if REML is None:
            REML = self.method == "REML"
        return dict(self.REML if REML else self.ML)

    def as_dict(self):
        return {"ML": dict(self.ML), "REML": dict(self.REML)}

    def _get(self, row, REML):
        return (self.REML if REML else self.ML)[row]


def _fitstats_core(result):
    yi = np.asarray(result.yi, dtype=float)
    vi = np.asarray(result.vi, dtype=float)
    x = np.asarray(result.X, dtype=float)
    b = np.asarray(result.b, dtype=float).ravel()
    tau2 = float(result.tau2) if not getattr(result, "tau2_fix", False) else 0.0
    k = yi.size
    p = x.shape[1]
    method = str(result.method)
    parms = p + (0 if method in ("FE", "EE", "CE") else 1)

    vari = vi + tau2
    w = 1.0 / vari
    resid = yi - x @ b
    rss = float(np.sum(w * resid ** 2))
    logdet_w = float(np.sum(np.log(vari)))

    ll_ml = -0.5 * k * math.log(2 * math.pi) - 0.5 * logdet_w - 0.5 * rss
    xtwx = x.T @ (w[:, None] * x)
    sign, logdet_xtwx = np.linalg.slogdet(xtwx)
    sign2, logdet_xtx = np.linalg.slogdet(x.T @ x)
    ll_reml = (-0.5 * (k - p) * math.log(2 * math.pi)
               + (0.5 * logdet_xtx if sign2 > 0 else _NAN)
               - 0.5 * logdet_w
               - (0.5 * logdet_xtwx if sign > 0 else _NAN)
               - 0.5 * rss)

    if k > p and np.all(vi > 0):
        ll_sat = float(np.sum(-0.5 * np.log(2 * math.pi * vi)))
        dev_ml = -2 * (ll_ml - ll_sat)
    elif k > p:
        dev_ml = -2 * ll_ml
    else:
        dev_ml = 0.0

    ml = {
        "ll": ll_ml, "dev": dev_ml,
        "AIC": -2 * ll_ml + 2 * parms,
        "BIC": -2 * ll_ml + parms * math.log(k),
        "AICc": -2 * ll_ml + (2 * parms * max(k, parms + 2)
                              / (max(k, parms + 2) - parms - 1)),
    }
    n_r = k - p
    # R's log(0) is -Inf rather than an error, so a saturated model reports
    # BIC = -Inf instead of blowing up here.
    bic_reml = (-2 * ll_reml + parms * math.log(n_r) if n_r > 0
                else (-math.inf if parms > 0 else 0.0))
    reml = {
        "ll": ll_reml, "dev": -2 * ll_reml,
        "AIC": -2 * ll_reml + 2 * parms,
        "BIC": bic_reml,
        "AICc": -2 * ll_reml + (2 * parms * max(n_r, parms + 2)
                                / (max(n_r, parms + 2) - parms - 1)),
    }
    return FitstatsResult(ml, reml, method, parms, k, p)


def _is_matreg(result):
    return (not hasattr(result, "X") and hasattr(result, "beta")
            and hasattr(result, "R2"))


def fitstats(result, REML=None):
    """``metafor::fitstats`` / ``$fit.stats`` — logLik, deviance, AIC, BIC, AICc."""
    fs = (_matreg_fitstats(result) if _is_matreg(result)
          else _fitstats_core(result))
    if REML is None:
        return fs
    col = fs.REML if REML else fs.ML
    names = {"ll": "logLik", "dev": "deviance", "AIC": "AIC",
             "BIC": "BIC", "AICc": "AICc"}
    return {names[k]: v for k, v in col.items()}


def aic_rma(result, k=2, correct=False):
    """``metafor::AIC.rma`` / ``AIC.matreg`` — AIC (or AICc)."""
    fs = (_matreg_fitstats(result) if _is_matreg(result)
          else _fitstats_core(result))
    row = "AICc" if correct else "AIC"
    return float(fs._get(row, fs.method == "REML"))


def bic_rma(result):
    """``metafor::BIC.rma`` / ``BIC.matreg`` — BIC."""
    fs = (_matreg_fitstats(result) if _is_matreg(result)
          else _fitstats_core(result))
    return float(fs._get("BIC", fs.method == "REML"))


# --------------------------------------------------------------------------- #
# accessors
# --------------------------------------------------------------------------- #

def se_rma(result, type="beta"):
    """``metafor::se.rma`` — standard errors of the coefficients."""
    names = _coef_names(result)
    return dict(zip(names, [float(v) for v in np.asarray(result.se).ravel()]))


def df_residual_rma(result):
    """``metafor::df.residual.rma`` — ``k.eff - p.eff``."""
    k_eff = getattr(result, "k_eff", None) or result.k
    p_eff = getattr(result, "p_eff", None) or result.p
    return int(k_eff) - int(p_eff)


def model_matrix_rma(result, asdf=False):
    """``metafor::model.matrix.rma`` — the design matrix used in the fit."""
    x = np.asarray(result.X, dtype=float)
    names = _coef_names(result)
    if not asdf:
        return {"matrix": x, "names": names}
    return [{names[j]: float(x[i, j]) for j in range(x.shape[1])}
            for i in range(x.shape[0])]


def as_data_frame_list_rma(result):
    """``metafor::as.data.frame.list.rma`` — study-level result data frame."""
    names = _coef_names(result)
    yi = np.asarray(result.yi, dtype=float)
    vi = np.asarray(result.vi, dtype=float)
    slab = getattr(result, "slab", None) or [f"Study {i + 1}"
                                             for i in range(yi.size)]
    w = np.asarray(result.weights, dtype=float)
    w_pct = np.asarray(result.weights_pct, dtype=float)
    fit = np.asarray(result.fitted, dtype=float)
    res = np.asarray(result.residuals, dtype=float)
    out = []
    for i in range(yi.size):
        row = {"yi": float(yi[i]), "vi": float(vi[i]), "sei": math.sqrt(vi[i])}
        for j, n in enumerate(names):
            row[n] = float(np.asarray(result.X)[i, j])
        row.update({
            "fitted": float(fit[i]) if fit.size else _NAN,
            "resid": float(res[i]) if res.size else _NAN,
            "weight": float(w[i]) if w.size else _NAN,
            "weight_pct": float(w_pct[i]) if w_pct.size else _NAN,
            "slab": slab[i] if i < len(slab) else str(i + 1),
        })
        out.append(row)
    return out


def cooks_distance_rma(result):
    """``metafor::cooks.distance.rma.uni`` — Cook's distance per study."""
    from .sensitivity import influence

    yi = np.asarray(result.yi, dtype=float)
    vi = np.asarray(result.vi, dtype=float)
    # influence() takes the moderator matrix WITHOUT an intercept (it prepends
    # one itself); passing result.X verbatim duplicated the intercept column and
    # made every refit singular.
    X = None
    if not getattr(result, "int_only", False):
        x = np.asarray(result.X, dtype=float)
        if x.ndim == 1:
            x = x.reshape(-1, 1)
        X = x[:, 1:] if np.allclose(x[:, 0], 1.0) else x
    inf = influence(yi, vi, X, method=str(result.method),
                    tau2=(float(result.tau2)
                          if getattr(result, "tau2_fix", False) else None),
                    test=str(getattr(result, "test", "z")))
    return np.asarray([float(v) for v in inf.cook_d], dtype=float)


class BlupResult:
    """Result of :func:`blup` (``metafor::blup.rma.uni``)."""

    __slots__ = ("pred", "se", "pi_lb", "pi_ub", "slab")

    def __init__(self, pred, se, pi_lb, pi_ub, slab):
        self.pred = pred
        self.se = se
        self.pi_lb = pi_lb
        self.pi_ub = pi_ub
        self.slab = slab

    def __repr__(self):
        return f"BlupResult(k={len(self.pred)})"

    def as_dict(self):
        return {"pred": self.pred, "se": self.se,
                "pi.lb": self.pi_lb, "pi.ub": self.pi_ub,
                "slab": self.slab}


def blup(result, level=None):
    """``metafor::blup.rma.uni`` — best linear unbiased predictions.

    ``pred_i = l_i y_i + (1 - l_i) X_i b`` with ``l_i = tau2 / (tau2 + v_i)``;
    ``var(pred_i) = l_i v_i + (1 - l_i)^2 X_i V X_i'``.
    """
    yi = np.asarray(result.yi, dtype=float)
    vi = np.asarray(result.vi, dtype=float)
    x = np.asarray(result.X, dtype=float)
    b = np.asarray(result.b, dtype=float).ravel()
    vb = np.asarray(result.vb, dtype=float)
    tau2 = float(result.tau2)
    if level is None:
        level = float(getattr(result, "level", 0.95))

    test = str(getattr(result, "test", "z"))
    if test in ("knha", "adhoc", "t"):
        crit = float(stats.t.isf((1.0 - level) / 2.0, int(result.ddf)))
    else:
        crit = float(stats.norm.isf((1.0 - level) / 2.0))

    li = np.ones(yi.size) if math.isinf(tau2) else tau2 / (tau2 + vi)
    pred = np.empty(yi.size)
    vpred = np.empty(yi.size)
    for i in range(yi.size):
        xi = x[i, :]
        pred[i] = li[i] * yi[i] + (1 - li[i]) * float(xi @ b)
        if li[i] == 1.0:
            vpred[i] = li[i] * vi[i]
        else:
            vpred[i] = (li[i] * vi[i]
                        + (1 - li[i]) ** 2 * float(xi @ vb @ xi))
    se = np.sqrt(vpred)
    pi_lb = pred - crit * se
    pi_ub = pred + crit * se
    slab = getattr(result, "slab", None) or [f"Study {i + 1}"
                                            for i in range(yi.size)]
    return BlupResult([float(v) for v in pred], [float(v) for v in se],
                      [float(v) for v in pi_lb], [float(v) for v in pi_ub],
                      list(slab))


def sigma_matreg(result, REML=True):
    """``metafor::sigma.matreg`` — residual SD of a ``matreg`` fit."""
    if str(getattr(result, "test", "")) != "t":
        raise ValueError("Model does not contain a 'sigma' estimate.")
    v = getattr(result, "sigma2_reml" if REML else "sigma2_ml", None)
    if v is None:
        raise ValueError("Model does not contain a 'sigma' estimate.")
    return math.sqrt(float(v))


def update_rma(result, mods=None, **kwargs):
    """``metafor::update.rma`` — refit with modified arguments.

    Only re-specification of the moderators (``mods``) and the simple scalar
    arguments supported by :func:`~.rma_uni.rma` can be changed.
    """
    from .rma_uni import rma

    yi = np.asarray(result.yi, dtype=float)
    vi = np.asarray(result.vi, dtype=float)
    opts = {
        "method": str(result.method),
        "test": str(getattr(result, "test", "z")),
        "level": float(getattr(result, "level", 0.95)),
        "slab": getattr(result, "slab", None),
        "measure": getattr(result, "measure", None),
    }
    if mods is None:
        if getattr(result, "int_only", False):
            mods_arr = None
        else:
            x = np.asarray(result.X, dtype=float)
            # rma() prepends its own intercept column, so drop the model's one
            # (passing the full X duplicated it and made the refit singular).
            mods_arr = x[:, 1:] if np.allclose(x[:, 0], 1.0) else x
    else:
        mods_arr = mods
    opts.update(kwargs)
    return rma(yi, vi=vi, mods=mods_arr, **opts)


# --------------------------------------------------------------------------- #
# meta::estimates (compact form)
# --------------------------------------------------------------------------- #

class EstimatesResult:
    """Result of :func:`estimates` (``meta::estimates``).

    A study-level table with the pooled rows appended.  ``rows`` is a list of
    dicts with the columns ``studlab``, ``TE``, ``seTE``, ``lower``, ``upper``,
    ``statistic``, ``pval`` and ``type`` (``"study"``, ``"common"``,
    ``"random"`` or ``"prediction"``).
    """

    __slots__ = ("rows", "sm", "backtransf")

    def __init__(self, rows, sm, backtransf):
        self.rows = rows
        self.sm = sm
        self.backtransf = backtransf

    def __repr__(self):
        return f"EstimatesResult(rows={len(self.rows)})"

    def as_dict(self):
        return {"rows": self.rows, "sm": self.sm}

    def to_frame(self):
        try:
            import pandas as pd
        except ImportError:  # pragma: no cover
            return self.rows
        return pd.DataFrame(self.rows)


def estimates(result, study_results=True, common=True, random=True,
              prediction=True, backtransf=False):
    """``meta::estimates`` — tabulate study and pooled estimates.

    Args:
        result: a :class:`~.metagen.MetagenResult`.
        study_results: include the per-study rows.
        common, random, prediction: include the corresponding pooled rows.
        backtransf: report on the back-transformed scale.

    Returns:
        :class:`EstimatesResult`.
    """
    if not isinstance(result, MetagenResult):
        raise TypeError("argument 'result' must be a MetagenResult")

    sm = str(getattr(result, "sm", "") or "")
    rows = []
    if study_results:
        yi = np.asarray(result.TE, dtype=float)
        sei = np.asarray(result.seTE, dtype=float)
        slab = getattr(result, "slab", None)
        if slab is None:
            slab = [f"Study {i + 1}" for i in range(yi.size)]
        for i in range(yi.size):
            te = float(yi[i])
            se = float(sei[i])
            stat = te / se if se > 0 else _NAN
            rows.append({
                "studlab": slab[i],
                "TE": _bt(te, sm, backtransf),
                "seTE": se,
                "lower": _bt(te - 1.959963984540054 * se, sm, backtransf),
                "upper": _bt(te + 1.959963984540054 * se, sm, backtransf),
                "statistic": stat,
                "pval": float(2 * stats.norm.sf(abs(stat))),
                "type": "study",
            })
    if common:
        rows.append({
            "studlab": "Common effect model",
            "TE": _bt(result.TE_common, sm, backtransf),
            "seTE": result.seTE_common,
            "lower": _bt(result.lower_common, sm, backtransf),
            "upper": _bt(result.upper_common, sm, backtransf),
            "statistic": result.statistic_common,
            "pval": result.pval_common,
            "type": "common",
        })
    if random:
        rows.append({
            "studlab": "Random effects model",
            "TE": _bt(result.TE_random, sm, backtransf),
            "seTE": result.seTE_random,
            "lower": _bt(result.lower_random, sm, backtransf),
            "upper": _bt(result.upper_random, sm, backtransf),
            "statistic": result.statistic_random,
            "pval": result.pval_random,
            "type": "random",
        })
    if prediction and getattr(result, "seTE_predict", None) is not None:
        rows.append({
            "studlab": "Prediction interval",
            "TE": _NAN,
            "seTE": result.seTE_predict,
            "lower": _bt(result.lower_predict, sm, backtransf),
            "upper": _bt(result.upper_predict, sm, backtransf),
            "statistic": _NAN, "pval": _NAN,
            "type": "prediction",
        })
    return EstimatesResult(rows, sm, backtransf)


def _bt(v, sm, backtransf):
    if not backtransf:
        return v
    from .metagen import backtransf as _backtransf
    try:
        return _backtransf(v, sm)
    except Exception:
        return v


# --------------------------------------------------------------------------- #
# meta::.forestArgs
# --------------------------------------------------------------------------- #

def forest_args(common=True, random=True, prediction=False, overall=False,
                text_common="Common effect model", text_random="Random effects model",
                text_predict="Prediction interval", text_overall="Overall effect"):
    """``meta::.forestArgs`` — labels/rows used by the forest plot.

    Returns a dict with the row labels in the order they appear in a forest
    plot, plus the ``common``/``random``/``prediction``/``overall`` flags.
    """
    labels = []
    if random:
        labels.append(text_random)
    if common:
        labels.append(text_common)
    if prediction:
        labels.append(text_predict)
    if overall:
        labels.append(text_overall)
    return {
        "labels": labels,
        "common": bool(common),
        "random": bool(random),
        "prediction": bool(prediction),
        "overall": bool(overall),
        "text.common": text_common,
        "text.random": text_random,
        "text.predict": text_predict,
        "text.overall": text_overall,
    }

# --------------------------------------------------------------------------- #
# additional conversions (as.data.frame / as.matrix variants) and matreg AIC
# --------------------------------------------------------------------------- #

def _fitstats_parts(yi, vi, x, b, tau2, method):
    """Fit statistics from raw arrays (used for ``matreg`` objects)."""
    class _P:
        pass

    p = _P()
    p.yi = np.asarray(yi, dtype=float)
    p.vi = np.asarray(vi, dtype=float)
    p.X = np.asarray(x, dtype=float)
    p.b = np.asarray(b, dtype=float)
    p.tau2 = float(tau2)
    p.tau2_fix = method in ("FE", "EE", "CE")
    p.method = method
    return _fitstats_core(p)


def _matreg_fitstats(result):
    """``fitstats.matreg`` — metafor stores both columns on the object."""
    fs = getattr(result, "fit_stats", None)
    if fs is None:
        raise ValueError(
            "Model does not contain fit statistics; 'matreg' only computes "
            "them when the model was fitted from 'n' rather than 'V'.")
    return FitstatsResult(dict(fs["ML"]), dict(fs["REML"]), "ML",
                          int(result.parms), int(result.nobs),
                          int(result.p) + 1)


def as_data_frame_confint(result, include_na=False):
    """``metafor::as.data.frame.confint.rma`` — CIs as a data frame.

    Accepts a :class:`~.pooling.Tau2Confint`, a :class:`VifResult`, a
    ``matreg`` fit, an ``anova``-style result or a plain dict of columns;
    returns a list of dicts with the columns ``estimate``/``ci.lb``/``ci.ub``
    (metafor's names) when available.
    """
    rows = []
    d = result if isinstance(result, dict) else getattr(result, "__dict__", {})
    if "estimate" not in d:
        # metafor's rows are estimate/ci.lb/ci.ub; our objects call the point
        # estimate `beta` (matreg), `tau2` (Tau2Confint) or `coef`.
        for alt in ("beta", "coef", "tau2", "estimate"):
            if alt in d:
                d = dict(d)
                d["estimate"] = d.pop(alt)
                break
    if "ci_lb" in d and "ci.lb" not in d:
        d = dict(d)
        d["ci.lb"] = d.pop("ci_lb")
        d["ci.ub"] = d.pop("ci_ub")
    keys = [k for k in d if k not in ("vb", "names", "fit_stats")]
    n = 1
    for k in keys:
        v = d[k]
        if isinstance(v, (list, tuple, np.ndarray)) and not isinstance(v, str):
            n = max(n, len(v))
    for i in range(n):
        row = {}
        for k in keys:
            v = d[k]
            if isinstance(v, (list, tuple, np.ndarray)) and not isinstance(v, str):
                row[k] = v[i] if i < len(v) else None
            elif i == 0:
                row[k] = v
        rows.append(row)
    if not include_na:
        rows = [r for r in rows
                if not all(v is None for k, v in r.items()
                           if k in ("estimate", "ci.lb", "ci.ub"))]
    return rows


def as_matrix_list_rma(result):
    """``metafor::as.matrix.list.rma`` — study-level matrix of a fitted model."""
    rows = as_data_frame_list_rma(result)
    names = [k for k in rows[0] if k not in ("slab",)] if rows else []
    return {
        "matrix": np.array([[r.get(k) or float("nan") for k in names]
                            for r in rows], dtype=float) if rows else None,
        "names": names,
        "slab": [r["slab"] for r in rows],
    }

# --------------------------------------------------------------------------- #
# S3-style naming aliases (se, as.data.frame.*)
# --------------------------------------------------------------------------- #

def se(result, type="beta"):
    """``metafor::se`` / ``se.rma`` / ``se.default`` — coefficient SEs."""
    return se_rma(result, type=type)


def as_data_frame_list_anova(result):
    """``metafor::as.data.frame.list.anova.rma`` — anova table as a data frame."""
    d = result if isinstance(result, dict) else getattr(result, "__dict__", {})
    cols = [k for k in d if not k.startswith("_")]
    lens = [len(d[k]) for k in cols
            if isinstance(d[k], (list, tuple, np.ndarray))
            and not isinstance(d[k], str)]
    n = max(lens) if lens else 1
    rows = []
    for i in range(n):
        row = {}
        for k in cols:
            v = d[k]
            if isinstance(v, (list, tuple, np.ndarray)) and not isinstance(v, str):
                row[k] = v[i] if i < len(v) else None
            elif i == 0:
                row[k] = v
        rows.append(row)
    return rows


as_data_frame_list_confint = as_data_frame_confint


def as_data_frame_vif(result):
    """``metafor::as.data.frame.vif.rma`` — VIF table as a data frame."""
    d = result if isinstance(result, dict) else getattr(result, "__dict__", {})
    names = d.get("names") or getattr(result, "names", None) or []
    vif = d.get("vif") or getattr(result, "vif", None) or []
    sif = d.get("sif") or getattr(result, "sif", None) or []
    rows = []
    for i in range(len(vif)):
        rows.append({"coef": names[i] if i < len(names) else f"coef{i + 1}",
                     "vif": float(vif[i]),
                     "sif": float(sif[i]) if i < len(sif) else None})
    return rows

# --------------------------------------------------------------------------- #
# meta plotting/layout helpers and metafor option accessors
# --------------------------------------------------------------------------- #

_UNIT_FACTOR = {"in": 1.0, "cm": 2.54, "mm": 25.4}

_METAFOR_OPTS = {"space": True}


def inches2units(x, units="in", dpi=300):
    """``meta::inches2units`` — convert inches to ``in``/``cm``/``mm``/``px``."""
    if units not in ("in", "cm", "mm", "px"):
        raise ValueError("units must be 'in', 'cm', 'mm' or 'px'")
    factor = dpi if units == "px" else _UNIT_FACTOR[units]
    return np.asarray(x, dtype=float) * factor


def units2inches(x, units="in", dpi=300):
    """``meta::units2inches`` — inverse of :func:`inches2units`."""
    if units not in ("in", "cm", "mm", "px"):
        raise ValueError("units must be 'in', 'cm', 'mm' or 'px'")
    factor = dpi if units == "px" else _UNIT_FACTOR[units]
    return np.asarray(x, dtype=float) / factor


def jama_labels(author, year, citation):
    """``meta::JAMAlabels`` — JAMA-style study labels (``Author^ref (year)``)."""
    a = list(np.asarray(author).ravel())
    y = list(np.asarray(year).ravel())
    c = list(np.asarray(citation).ravel())
    if not (len(a) == len(y) == len(c)):
        raise ValueError(
            "Arguments 'author', 'year' and 'citation' must be of same length."
        )
    return [f"{ai}^{ci} ({yi})" for ai, yi, ci in zip(a, y, c)]


def ci(te, seTE, level=0.95, df=None, null_effect=0.0):
    """``meta:::ci`` — confidence interval and test for an effect estimate.

    Args:
        te: effect estimate(s).
        seTE: standard error(s).
        level: confidence level.
        df: degrees of freedom; ``None`` (or ``inf``) uses the normal
            distribution, a positive value the t distribution.
        null_effect: null value of the effect measure.

    Returns:
        dict with ``TE``, ``seTE``, ``lower``, ``upper``, ``statistic``, ``p``,
        ``level``, ``df`` and ``null.effect``.
    """
    te = np.asarray(te, dtype=float)
    se = np.asarray(seTE, dtype=float)
    alpha = 1.0 - level
    statistic = (te - null_effect) / se
    if df is None or (np.isscalar(df) and (df is None
                                           or (isinstance(df, float)
                                               and math.isinf(df)))):
        crit = float(stats.norm.isf(alpha / 2.0))
        lower = te - crit * se
        upper = te + crit * se
        pval = 2 * stats.norm.sf(np.abs(statistic))
        df_out = float("inf")
    else:
        df_arr = np.asarray(df, dtype=float)
        crit = stats.t.isf(alpha / 2.0, df_arr)
        lower = te - crit * se
        upper = te + crit * se
        pval = 2 * stats.t.sf(np.abs(statistic), df_arr)
        df_out = df_arr
    return {"TE": te, "seTE": se, "lower": lower, "upper": upper,
            "statistic": statistic, "p": pval, "level": level,
            "df": df_out, "null.effect": null_effect}


def getmfopt(x=None, default=None):
    """``metafor::getmfopt`` — read a ``metafor`` global option."""
    if x is None:
        return dict(_METAFOR_OPTS)
    return _METAFOR_OPTS.get(str(x), default)


def setmfopt(**kwargs):
    """``metafor::setmfopt`` — set ``metafor`` global options."""
    if ("space" in kwargs and kwargs["space"] is not None
            and not isinstance(kwargs["space"], bool)):
        raise ValueError("'space' must be a logical.")
    if "theme" in kwargs and kwargs["theme"] is not None:
        allowed = {"default", "light", "dark", "auto", "custom",
                   "default2", "light2", "dark2", "auto2", "custom2"}
        if kwargs["theme"] not in allowed:
            raise ValueError(
                "'theme' must be either 'default(2)', 'light(2)', 'dark(2)', "
                "'auto(2)', or 'custom(2)'."
            )
    for k, v in kwargs.items():
        _METAFOR_OPTS[k] = v
    return dict(_METAFOR_OPTS)
