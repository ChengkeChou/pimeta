"""Sampling covariance matrix of dependent effect sizes.

Port of ``metafor::vcalc``.  Dependency arises in four ways, and each one has
its own argument:

===========================  ==========================================
same cluster                 ``cluster`` (and ``subgroup`` for
                             independent sub-samples)
same construct, several      ``obs`` distinguishes the estimates;
measures                     ``rho`` is their correlation
different constructs         ``type`` distinguishes them; ``rho`` is
                             the between-construct correlation
contrasts sharing one        ``grp1`` / ``grp2`` name the two
condition                    conditions compared; ``w1`` / ``w2``
                             their inverse-variance weights
===========================  ==========================================

The shared-condition correlation is *derived*, not assumed: two contrasts
against the same control share the control arm's sampling variance, so

.. math::

    \\rho = \\frac{1/w_C}
                 {\\sqrt{(1/w_A + 1/w_C)(1/w_B + 1/w_C)}}

which equals exactly 0.5 when the three arms have equal weight.  That is why
metafor's output shows 0.5 everywhere; it is not a default correlation.

One behaviour is deliberately **not** reproduced: when neither ``obs`` nor
``type`` is given, metafor discards ``rho`` without a word and returns a
correlation of 1, i.e. a singular block.  ``pimeta`` applies the ``rho`` that
was asked for, and raises when there is nothing to apply.  Pass
``r_compat=True`` for the metafor behaviour; see
``docs/reference/differences.md`` §4.

A fifth structure is a *within-subject* design, where the same subjects are
measured at several time points: ``time1`` (and optionally ``time2``) with
``phi`` replaces the fixed correlation by an autocorrelation

.. math::

    \\rho_{ij} = \\phi^{|t_i - t_j|}

so two measurements one unit apart correlate ``phi``, two units apart
``phi**2``, and two measurements at the same time are the same estimate and
correlate 1.  ``phi`` must lie in ``[-1, 1]``; ``phi = 1`` makes the block
singular, and the positive-definiteness check warns rather than returning it
quietly.

metafor builds the correlation as a *product* of two factors -- how related
the outcomes are (``obs`` / ``type``) and how related the arms and measurement
times are (``grp1`` / ``grp2`` / ``time1`` / ``time2`` / ``w1`` / ``w2``) --
and so does this function.  Either factor is 1 when its arguments are absent,
which is why they can be given independently.
"""

from __future__ import annotations

import math
import warnings
from collections.abc import Callable

import numpy as np

__all__ = ["vcalc", "shared_control_corr"]


def shared_control_corr(w_shared, w_a, w_b):
    """Correlation between two contrasts that share one condition.

    Effect size ``a`` contrasts arm A against the shared arm C, effect size
    ``b`` contrasts arm B against the same C.  With homogeneous within-arm
    variance, ``Cov = sigma^2/w_C`` and ``Var = sigma^2/w_A + sigma^2/w_C``,
    so the variance cancels and the correlation is a function of the weights
    alone.

    Args:
        w_shared: weight (or group size) of the shared condition.
        w_a: weight of the first contrast's other condition.
        w_b: weight of the second contrast's other condition.

    Returns:
        The correlation in ``[0, 1]``; ``0.5`` when all three weights agree.
    """
    w_shared = float(w_shared)
    w_a = float(w_a)
    w_b = float(w_b)
    if w_shared <= 0.0:
        raise ValueError("weights must be positive")
    denom = math.sqrt((1.0 / w_a + 1.0 / w_shared) * (1.0 / w_b + 1.0 / w_shared))
    return (1.0 / w_shared) / denom


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #

def _as_vector(value, k, name):
    """Broadcast a scalar, or check a vector, to length ``k``."""
    if value is None:
        return None
    arr = np.asarray(value).ravel()
    if arr.size == 1:
        return np.repeat(arr, k)
    if arr.size != k:
        raise ValueError(
            f"length of {name!r} ({arr.size}) does not match 'vi' ({k})")
    return arr


def _index_levels(values, name):
    """Map values to 0-based codes in **first-appearance** order.

    R's ``unique()`` keeps first appearance rather than sorting, and ``vcalc``
    builds its correlation matrices in that order, so this has to match for a
    supplied correlation matrix to mean the same thing in both languages.
    """
    if values is None:
        return None, None
    arr = list(values)
    levels: list = []
    for v in arr:
        if v not in levels:
            levels.append(v)
    pos = {v: i for i, v in enumerate(levels)}
    codes = np.array([pos[v] for v in arr], dtype=int)
    if len(levels) < 2:
        raise ValueError(
            f"{name!r} must have at least two distinct values to be useful")
    return codes, levels


def _pair_corr(rho, levels, name, rho_names=None):
    """Normalise ``rho`` into a symmetric lookup over level pairs.

    Returns a closure ``f(i, j) -> corr`` used for *different* levels; the
    "same level means perfectly correlated" rule is applied by the caller.
    ``levels`` is the factor's level order, which decides how a supplied
    matrix is read unless ``rho_names`` says otherwise.
    """
    if rho is None:
        return None
    if np.ndim(rho) == 0:
        val = float(rho)
        if not -1.0 <= val <= 1.0:
            raise ValueError(f"{name}: correlations must lie in [-1, 1]")
        return lambda i, j: val
    arr = np.asarray(rho, dtype=float)
    if arr.ndim != 2:
        raise ValueError(
            f"{name}: 'rho' must be a scalar or a square matrix")
    n_levels = len(levels)
    if arr.shape != (n_levels, n_levels):
        raise ValueError(
            f"{name}: 'rho' is {arr.shape[0]}x{arr.shape[1]} but the factor "
            f"has {n_levels} levels")
    if not np.allclose(arr, arr.T):
        raise ValueError(f"{name}: 'rho' must be symmetric")
    if np.any(np.abs(arr) > 1.0):
        raise ValueError(f"{name}: correlations must lie in [-1, 1]")
    if rho_names is not None:
        names = list(rho_names)
        if len(names) != n_levels:
            raise ValueError(
                f"{name}: 'rho_names' has {len(names)} entries but the factor "
                f"has {n_levels} levels")
        order = {n: i for i, n in enumerate(names)}
        missing = [lv for lv in levels if lv not in order]
        if missing:
            raise ValueError(
                f"{name}: 'rho_names' does not cover the level(s) {missing}")
        pos = [order[lv] for lv in levels]
        arr = arr[np.ix_(pos, pos)]
    return lambda i, j: float(arr[i, j])


def _construct_block(idx, obs_codes, typ_codes, pair_obs, pair_typ, const):
    """Correlation implied by *which outcomes* the estimates measure.

    One factor of metafor's correlation entry.  With ``type`` and ``obs`` the
    two estimates of one construct are perfectly correlated, two measures of
    that construct take the within-construct correlation, and two different
    constructs take the between-construct correlation.  Each of the four
    combinations of the two arguments has its own reading of ``rho``, and the
    caller has already resolved which lookup to pass.
    """
    n = idx.size
    out = np.empty((n, n), dtype=float)
    if typ_codes is None:
        if obs_codes is None:
            out.fill(const)
            return out
        od = obs_codes[idx]
        for a in range(n):
            for b in range(n):
                out[a, b] = (1.0 if od[a] == od[b]
                             else pair_obs(od[a], od[b]))
        return out
    td = typ_codes[idx]
    if obs_codes is None:
        for a in range(n):
            for b in range(n):
                out[a, b] = (1.0 if td[a] == td[b]
                             else pair_typ(td[a], td[b]))
        return out
    od = obs_codes[idx]
    for a in range(n):
        for b in range(n):
            if td[a] != td[b]:
                out[a, b] = pair_typ(td[a], td[b])
            elif od[a] == od[b]:
                out[a, b] = 1.0
            else:
                out[a, b] = pair_obs(od[a], od[b])
    return out


def _arms_block(idx, g1, g2, t1, t2, w1v, w2v, phi):
    """Correlation implied by *which arms, at which time points*, a row compares.

    Two contrasts that share an arm share that arm's sampling error, and two
    measurements taken at different times share less of it:

    .. math::

        \\operatorname{Cov}(e_i, e_j) =
            \\sum_{(a, b)} \\pm\\, \\phi^{|t_a - t_b|}
            \\sqrt{\\frac{1}{w_{a,i}}\\frac{1}{w_{b,j}}}

    over the four arm matches, divided by the two contrasts' arm-scale
    standard deviations.  With every time equal and every weight 1 this is
    exactly 0.5 for two contrasts on a common control, which is where
    metafor's famous 0.5 comes from.

    The lower triangle is the one evaluated and then mirrored onto the upper
    triangle, because the four terms are not symmetric when rows disagree
    about which condition is ``grp1``.  That is what R's ``vcalc`` does.
    """
    i = idx[:, None]
    j = idx[None, :]

    def proximity(a, b):
        return np.where(a == b, 1.0, phi ** np.abs(a - b))

    num = (
        np.where(g1[i] == g1[j], proximity(t1[i], t1[j]), 0.0)
        * np.sqrt((1.0 / w1v[i]) * (1.0 / w1v[j]))
        - np.where(g1[i] == g2[j], proximity(t1[i], t2[j]), 0.0)
        * np.sqrt((1.0 / w1v[i]) * (1.0 / w2v[j]))
        - np.where(g2[i] == g1[j], proximity(t2[i], t1[j]), 0.0)
        * np.sqrt((1.0 / w2v[i]) * (1.0 / w1v[j]))
        + np.where(g2[i] == g2[j], proximity(t2[i], t2[j]), 0.0)
        * np.sqrt((1.0 / w2v[i]) * (1.0 / w2v[j]))
    )
    var = 1.0 / w1v[idx] + 1.0 / w2v[idx]
    out = num / np.sqrt(np.outer(var, var))
    return np.tril(out) + np.tril(out, -1).T


def _check_positive_definite(blocks, labels, V):
    """Warn about a cluster whose covariance block cannot be inverted.

    metafor does this by default (``checkpd = TRUE``) and it is the only
    warning that catches a ``phi`` of 1, which makes every measurement in a
    cluster perfectly correlated with every other.
    """
    for block, label in zip(blocks, labels, strict=True):
        if len(block) < 2:
            continue
        idx = np.asarray(block, dtype=int)
        sub = V[np.ix_(idx, idx)]
        if not np.all(np.isfinite(sub)):
            warnings.warn(
                f"the sampling covariance of cluster {label!r} is not finite",
                RuntimeWarning, stacklevel=3)
            continue
        ev = np.linalg.eigvalsh(sub)
        if ev[0] <= 1e-10 * abs(ev[-1]):
            warnings.warn(
                "the var-cov matrix appears to be not positive definite in "
                f"cluster {label!r}", RuntimeWarning, stacklevel=3)


# --------------------------------------------------------------------------- #
# vcalc
# --------------------------------------------------------------------------- #

def vcalc(vi, cluster=None, subgroup=None, obs=None, type=None,
          time1=None, time2=None, grp1=None, grp2=None, w1=None, w2=None,
          rho=None, phi=None, rho_names=None, r_compat=False,
          check_pd=True):
    """Build the ``k x k`` sampling covariance matrix ``V``.

    Args:
        vi: sampling variances, length ``k``.
        cluster: cluster identifier (usually the study).  ``None`` treats every
            row as one cluster, which is a convenience pimeta adds; R requires
            the argument.
        subgroup: optional independent subgroups within a cluster; rows in
            different subgroups are independent even inside one cluster.
        obs: optional identifier of the *observation* of one construct, so
            several measures of the same thing can be told apart.
        type: optional identifier of the *construct*, when several different
            outcomes were measured on the same subjects.
        time1: the time point at which each effect size was measured.  With
            ``phi`` this turns the shared-arm correlation into an
            autocorrelation, and two estimates measured at the same time are
            perfectly correlated.
        time2: the time of the *second* condition a row compares, for a design
            in which the two conditions were measured at different times.
            Requires ``grp2``.
        grp1, grp2: the two conditions each effect size compares.  Rows that
            share one of them are correlated, with the correlation derived
            from the shared arm's weight.
        w1, w2: weights (group sizes, or inverse sampling variances) of the
            two conditions; default 1 each, which makes the correlation
            exactly 0.5.
        rho: correlation among estimates of the same construct (with ``obs``),
            among different constructs (with ``type``), or among all rows of a
            cluster (neither given).  A scalar, or a square matrix indexed by
            the factor's levels.
        phi: autocorrelation between two measurements one time unit apart,
            in ``[-1, 1]``.  Required when ``time1`` is given; two estimates
            at times $t_i$ and $t_j$ then correlate ``phi ** |t_i - t_j|``.
        rho_names: labels of the rows of a matrix ``rho``, in matrix order.
            Without it the matrix is indexed in first-appearance order of the
            factor, which is what R's ``unique()`` produces.
        r_compat: reproduce metafor exactly, including its silent discard of
            ``rho`` when neither ``obs`` nor ``type`` is given.  Default
            ``False``: the correlation that was asked for is the one used.
        check_pd: warn when a cluster's covariance block is not positive
            definite, as metafor's ``checkpd`` does.  ``phi = 1`` and
            ``r_compat`` both produce such a block.

    Returns:
        ``(k, k)`` float array.

    Raises:
        ValueError: inconsistent lengths, a forgotten correlation, a ``phi``
            outside ``[-1, 1]``, a time argument without ``phi``, or a row
            that compares a condition with itself.
    """
    vi = np.asarray(vi, dtype=float).ravel()
    k = vi.size
    if k == 0:
        raise ValueError("'vi' must not be empty")
    if np.any(vi < 0):
        raise ValueError("'vi' must be non-negative")
    if time2 is not None and time1 is None:
        raise ValueError(
            "either give only 'time1', both 'time1' and 'time2', or neither")
    if time2 is not None and grp2 is None:
        raise ValueError("'grp2' is required when 'time2' is given")
    if time1 is not None and phi is None:
        raise ValueError(
            "'phi' is required when 'time1' is given: it is the "
            "autocorrelation between two measurements one unit apart")
    if phi is not None and abs(float(phi)) > 1.0:
        raise ValueError("'phi' must lie in [-1, 1]")

    cl = _as_vector(cluster, k, "cluster")
    if cl is None:
        cl = np.zeros(k, dtype=int)
    sg = _as_vector(subgroup, k, "subgroup")
    # a cluster is the pair (cluster, subgroup)
    if sg is not None:
        keys = list(zip(cl.tolist(), sg.tolist(), strict=True))
    else:
        keys = cl.tolist()

    obs_codes, obs_levels = _index_levels(obs, "obs")
    typ_codes, typ_levels = _index_levels(type, "type")

    if (grp1 is None) != (grp2 is None):
        raise ValueError("'grp1' and 'grp2' must be given together")

    sd = np.sqrt(vi)

    # ---- the arm / time factor -------------------------------------------
    if grp1 is None:
        g1 = np.ones(k)
        g2 = np.full(k, 2.0)
    else:
        g1 = np.asarray(_as_vector(grp1, k, "grp1"))
        g2 = np.asarray(_as_vector(grp2, k, "grp2"))
    wa = _as_vector(w1, k, "w1")
    wb = _as_vector(w2, k, "w2")
    wa = np.ones(k) if wa is None else np.asarray(wa, dtype=float)
    wb = np.ones(k) if wb is None else np.asarray(wb, dtype=float)
    if np.any(wa <= 0.0) or np.any(wb <= 0.0):
        raise ValueError("'w1' and 'w2' must be positive")
    if np.any(g1 == g2):
        raise ValueError(
            "'grp1' and 'grp2' name the two conditions a row compares; a row "
            "that compares a condition with itself is not an effect size and "
            "has no sampling variance")
    t1 = _as_vector(time1, k, "time1")
    t2 = _as_vector(time2, k, "time2")
    t1 = np.ones(k) if t1 is None else np.asarray(t1, dtype=float)
    t2 = t1 if t2 is None else np.asarray(t2, dtype=float)
    phi_val = 1.0 if phi is None else float(phi)
    arms_given = any(x is not None for x in (grp1, time1, w1, w2))

    # ---- the construct / observation factor ------------------------------
    const: float = 1.0
    pair_obs: Callable[[int, int], float] | None = None
    pair_typ: Callable[[int, int], float] | None = None

    if obs_codes is None and typ_codes is None:
        if r_compat:
            # metafor: 'rho' is dropped and every pair is perfectly
            # correlated, which makes the block singular.
            const = 1.0
        elif rho is not None:
            const = float(np.asarray(rho).ravel()[0])
            if not -1.0 <= const <= 1.0:
                raise ValueError("correlations must lie in [-1, 1]")
        elif arms_given:
            # the dependence is already carried by the shared arms and their
            # time points, so there is nothing to invent
            const = 1.0
        else:
            raise ValueError(
                "'rho' is required when neither 'obs' nor 'type' is given: "
                "two different effect sizes from one cluster are not "
                "perfectly correlated, so there is no defensible default. "
                "Pass 'rho=' for a constant within-cluster correlation, or "
                "'obs=' so that the estimates can be told apart. "
                "r_compat=True reproduces metafor's silent correlation of 1.")

    elif obs_codes is not None and typ_codes is None:
        pair_obs = _pair_corr(rho, obs_levels, "obs", rho_names)
        if pair_obs is None:
            raise ValueError("'rho' is required when 'obs' is given")

    elif typ_codes is not None and obs_codes is None:
        pair_typ = _pair_corr(rho, typ_levels, "type", rho_names)
        if pair_typ is None:
            raise ValueError("'rho' is required when 'type' is given")

    else:
        # ---- both 'type' and 'obs' ---------------------------------------
        if rho is None:
            raise ValueError(
                "'rho' is required when 'type' and 'obs' are given")
        if isinstance(rho, (list, tuple)) and len(rho) == 2:
            within_spec, between_spec = rho
        else:
            seq = (np.asarray(rho, dtype=float).ravel()
                   if np.ndim(rho) else None)
            if seq is None or seq.size != 2:
                raise ValueError(
                    "when 'type' and 'obs' are both given, 'rho' must supply "
                    "the within-construct and between-construct correlations, "
                    "e.g. rho=(0.6, 0.4)")
            within_spec, between_spec = float(seq[0]), float(seq[1])
        pair_obs = _pair_corr(within_spec, obs_levels, "obs", rho_names)
        pair_typ = _pair_corr(between_spec, typ_levels, "type", rho_names)

    # ---- assemble --------------------------------------------------------
    blocks: list[list[int]] = []
    index: dict = {}
    for i in range(k):
        if keys[i] not in index:
            index[keys[i]] = len(blocks)
            blocks.append([])
        blocks[index[keys[i]]].append(i)
    labels = list(index.keys())

    V = np.zeros((k, k))
    for block in blocks:
        idx = np.asarray(block, dtype=int)
        Rb = (_construct_block(idx, obs_codes, typ_codes, pair_obs, pair_typ,
                               const)
              * _arms_block(idx, g1, g2, t1, t2, wa, wb, phi_val))
        np.fill_diagonal(Rb, 1.0)
        V[np.ix_(idx, idx)] = Rb * np.outer(sd[idx], sd[idx])
    if check_pd:
        _check_positive_definite(blocks, labels, V)
    return V
