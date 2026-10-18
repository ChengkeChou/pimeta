"""GOSH plot diagnostics (metafor::gosh.rma).

Goodman-Schulz-Hastie style exhaustive (or random) subset analysis: refit the
model on every subset of studies of size ``p`` up to ``k`` and collect the
pooled estimate, Cochran's QE, I^2, H^2 and tau^2 for each subset. Used to
detect influential studies / outliers and to visualise heterogeneity.

Reference: metafor::gosh.rma (Viechtbauer, 2020).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from itertools import product

import numpy as np

from .metareg import _dl_tau2, _i2_h2, _wls
from .pooling import _reml_ml_loglik

_MAX_EXACT = 1_000_000  # metafor's hard cap for the exact (exhaustive) path


@dataclass
class GoshResult:
    """Results of a GOSH analysis (one row per fitted subset)."""

    k: list[int]                 # number of studies in each subset
    QE: list[float]              # Cochran's Q (fixed-effect weights)
    I2: list[float]              # I^2 statistic (percent)
    H2: list[float]              # H^2 statistic
    tau2: list[float]            # estimate of tau^2
    tau: list[float]             # sqrt(tau^2)
    estimate: list[float]        # pooled estimate (intercept-only models)
    beta: list[list[float]]      # (n_subset, p) coefficients (models with mods)
    incl: list[list[bool]]       # study-inclusion indicator, one row per subset
    k_total: int                 # number of studies in the full data set
    p: int                       # number of model coefficients (incl. intercept)
    int_only: bool
    method: str

    def to_dict(self) -> dict:
        out = {
            "k": self.k,
            "QE": self.QE,
            "I2": self.I2,
            "H2": self.H2,
            "tau2": self.tau2,
            "tau": self.tau,
            "incl": self.incl,
            "k_total": self.k_total,
            "p": self.p,
            "int_only": self.int_only,
            "method": self.method,
        }
        if self.int_only:
            out["estimate"] = self.estimate
        else:
            out["beta"] = self.beta
        return out


def _fs_tau2_converged(y, v, X, method):
    """ML/REML tau^2 via metafor's Fisher scoring, with convergence flag.

    Returns ``(tau2, converged)``. Identical to
    :func:`.pooling._metafor_fisher_scoring_tau2` except it mirrors R's
    ``while (change > threshold)`` loop and flags non-convergence (R raises
    ``"Fisher scoring algorithm did not converge"``, which GOSH turns into NA).
    """
    y = np.asarray(y, dtype=float).ravel()
    v = np.asarray(v, dtype=float).ravel()
    X = np.asarray(X, dtype=float)
    k = y.shape[0]
    p = X.shape[1]

    ymc = y - y.mean()
    stXX = np.linalg.inv(X.T @ X)
    P = np.eye(k) - X @ stXX @ X.T
    rss = float(ymc @ P @ ymc)
    tr_pv = float(np.sum(np.diag(P) * v))
    tau2 = max(0.0, (rss - tr_pv) / (k - p))

    threshold = 1e-5
    change = threshold + 1.0
    it = 0
    converged = True
    while change > threshold:
        it += 1
        old2 = tau2
        wi = 1.0 / (v + tau2)
        W = np.diag(wi)
        stXWX = np.linalg.inv((X.T * wi) @ X)
        P = W - W @ X @ stXWX @ X.T @ W
        PP = P @ P
        yppy = float(ymc @ PP @ ymc)
        if method == "ML":
            adj = (yppy - float(wi.sum())) / float(np.sum(wi ** 2))
        else:  # REML
            adj = (yppy - float(np.trace(P))) / float(np.trace(PP))
        if np.isnan(adj):
            adj = 0.0
        while tau2 + adj < 0.0:
            adj /= 2.0
        tau2 = tau2 + adj
        change = abs(old2 - tau2)
        if it > 100:
            converged = False
            break

    if converged and bool(np.all(v > 0)) and tau2 > threshold:
        ll0 = _reml_ml_loglik(y, v, X, 0.0, method)
        ll = _reml_ml_loglik(y, v, X, tau2, method)
        tol = float(np.finfo(np.float64).eps) ** 0.25
        if ll0 - ll > tol:
            tau2 = 0.0

    return float(tau2), converged


def _profile_subset(y, v, Xd, method, model, tau2_fixed):
    """Fit one subset and return (beta, k, QE, I2, H2, tau2) or None on failure.

    Mirrors ``metafor:::.profile.rma.uni`` with ``subset = TRUE``. For
    intercept-only FE/DL models (``model`` 1/2) the closed-form fast path is
    used; otherwise the full ``rma.uni`` fit is reproduced.
    """
    k = len(y)
    p = Xd.shape[1]
    wi = 1.0 / v
    sumwi = float(np.sum(wi))
    est = float(np.sum(wi * y) / sumwi)
    m = method.upper()

    if model == 1:  # fixed-effect fast path (int-only)
        q = float(np.sum(wi * (y - est) ** 2))
        i2 = 0.0
        h2 = 1.0
        if k > 1:
            i2 = max(0.0, 100.0 * (q - (k - 1)) / q)
            h2 = q / (k - 1)
        return np.array([est]), k, q, i2, h2, 0.0

    if model == 2:  # DerSimonian-Laird fast path (int-only)
        q = float(np.sum(wi * (y - est) ** 2))
        i2 = 0.0
        h2 = 1.0
        tau2 = 0.0
        if k > 1:
            i2 = max(0.0, 100.0 * (q - (k - 1)) / q)
            h2 = q / (k - 1)
            tau2 = max(0.0, (q - (k - 1)) / (sumwi - float(np.sum(wi ** 2)) / sumwi))
            w2 = 1.0 / (v + tau2)
            est = float(np.sum(w2 * y) / np.sum(w2))
        return np.array([est]), k, q, i2, h2, tau2

    # model == 0: full rma.uni refit. Mirror R's k/p error checks, which turn
    # into NA in GOSH when they trigger.
    if p == 1 and k == 1:
        # R's special "k1" path: intercept-only with a single study.
        return np.array([est]), k, 0.0, 0.0, 1.0, 0.0

    if m in ("FE", "EE", "CE"):
        if p > k:
            return None
    elif tau2_fixed is not None:
        if p > k:
            return None
    else:
        if (p + 1) > k:
            return None

    if tau2_fixed is not None:
        tau2 = float(tau2_fixed)
        converged = True
    elif m in ("FE", "EE", "CE"):
        tau2 = 0.0
        converged = True
    elif m == "DL":
        tau2 = _dl_tau2(y, v, Xd)
        converged = True
    else:  # ML / REML
        tau2, converged = _fs_tau2_converged(y, v, Xd, m)
        if not converged:
            return None

    w = 1.0 / (v + tau2)
    beta, _cov, _ = _wls(Xd, w, y)

    if k > p:
        _, _, qe = _wls(Xd, 1.0 / v, y)
        if m in ("FE", "EE", "CE"):
            df = k - p
            i2 = max(0.0, 100.0 * (qe - df) / qe) if qe > 0 else 0.0
            h2 = qe / df if df > 0 else 1.0
        else:
            i2, h2 = _i2_h2(tau2, v, Xd)
    else:  # k == p (saturated): R sets QE=0, I2=0, H2=1
        qe = 0.0
        i2 = 0.0
        h2 = 1.0
    return beta, k, qe, i2, h2, tau2


def gosh(
    y,
    v,
    X=None,
    method: str = "REML",
    tau2: float | None = None,
    subsets: float | None = None,
    seed: int | None = None,
) -> GoshResult:
    """Graphical display of study heterogeneity (metafor::gosh).

    Args:
        y, v: analysis-scale effect and sampling variance.
        X: moderator design matrix (n, m) WITHOUT intercept, or None for an
            intercept-only model.
        method: "FE", "DL", "ML" or "REML" (default REML).
        tau2: fixed tau^2 value (equivalent to ``rma(..., tau2=...)``); when
            given, tau^2 is not re-estimated for each subset.
        subsets: number of subsets to fit. If None, all possible subsets are
            fitted when that number is <= 1e6, otherwise 1e6 random subsets.
        seed: RNG seed for the random-subsets path (reproduces R's ``seed``).

    Returns:
        GoshResult with one row per subset, sorted by subset size ``k``.
    """
    y = np.asarray(y, dtype=float).ravel()
    v = np.asarray(v, dtype=float).ravel()
    k = len(y)
    if k <= 1:
        raise ValueError("GOSH requires k >= 2 studies.")

    if X is None:
        Xd = np.ones((k, 1))
        int_only = True
    else:
        X = np.asarray(X, dtype=float)
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        Xd = np.column_stack([np.ones(k), X])
        int_only = False
    p = Xd.shape[1]

    m = method.upper()
    if m not in ("FE", "EE", "CE", "DL", "ML", "REML"):
        raise ValueError("method must be FE/DL/ML/REML")
    if int_only and m in ("FE", "EE", "CE"):
        model = 1
    elif int_only and m == "DL":
        model = 2
    else:
        model = 0

    tau2_fixed = float(tau2) if tau2 is not None else None

    n_tot = sum(math.comb(k, i) for i in range(p, k + 1))
    if subsets is None:
        exact = n_tot <= _MAX_EXACT
        if not exact:
            n_tot = _MAX_EXACT
    else:
        subsets = int(round(subsets))
        if subsets <= 1:
            raise ValueError("Argument 'subsets' must be >= 2.")
        exact = n_tot <= subsets
        if not exact:
            n_tot = subsets

    if n_tot == math.inf or n_tot > _MAX_EXACT * 10:
        raise ValueError("Too many iterations required for all combinations.")

    if exact:
        # R's expand.grid() makes the FIRST column (study 1) vary fastest, so
        # study j corresponds to bit (j-1) of the row index. itertools.product
        # makes the first position vary slowest, so we reverse each combo to
        # match R's enumeration order exactly (required for stable-sort parity).
        incl = [
            list(combo[::-1])
            for combo in product([False, True], repeat=k)
            if sum(combo) >= p
        ]
    else:
        rng = np.random.default_rng(seed)
        sizes = np.arange(p, k + 1)
        probs = np.array([math.comb(k, int(s)) for s in sizes], dtype=float)
        probs = probs / probs.sum()
        j = rng.choice(sizes, size=n_tot, replace=True, p=probs)
        incl = []
        for m in j:
            sel = rng.choice(k, size=int(m), replace=False)
            row = [False] * k
            for i in sel:
                row[int(i)] = True
            incl.append(row)

    n_rows = len(incl)
    k_out = [0] * n_rows
    qe_out = [np.nan] * n_rows
    i2_out = [np.nan] * n_rows
    h2_out = [np.nan] * n_rows
    tau2_out = [np.nan] * n_rows
    est_out = [np.nan] * n_rows
    beta_out = [[np.nan] * p for _ in range(n_rows)] if not int_only else []

    for r, combo in enumerate(incl):
        idx = [i for i in range(k) if combo[i]]
        yi = y[idx]
        vi = v[idx]
        result = _profile_subset(yi, vi, Xd[idx], m, model, tau2_fixed)
        if result is None:
            # Model fit failed (e.g. Fisher scoring did not converge): leave NA.
            k_out[r] = len(idx)
            continue
        beta_r, kk, qe, i2, h2, t2 = result
        k_out[r] = kk
        qe_out[r] = qe
        i2_out[r] = i2
        h2_out[r] = h2
        tau2_out[r] = t2
        if int_only:
            est_out[r] = float(beta_r[0])
        else:
            beta_out[r] = [float(b) for b in beta_r]

    tau_out = [math.sqrt(t2) if t2 == t2 and t2 >= 0 else np.nan
               for t2 in tau2_out]

    # Stable sort by subset size, matching R's ``res[order(res$k), ]``.
    order = sorted(range(n_rows), key=lambda r: k_out[r])
    k_out = [k_out[r] for r in order]
    qe_out = [qe_out[r] for r in order]
    i2_out = [i2_out[r] for r in order]
    h2_out = [h2_out[r] for r in order]
    tau2_out = [tau2_out[r] for r in order]
    tau_out = [tau_out[r] for r in order]
    incl = [incl[r] for r in order]
    if int_only:
        est_out = [est_out[r] for r in order]
    else:
        beta_out = [beta_out[r] for r in order]

    return GoshResult(
        k=k_out,
        QE=qe_out,
        I2=i2_out,
        H2=h2_out,
        tau2=tau2_out,
        tau=tau_out,
        estimate=est_out if int_only else [],
        beta=beta_out,
        incl=incl,
        k_total=k,
        p=p,
        int_only=int_only,
        method=m,
    )
