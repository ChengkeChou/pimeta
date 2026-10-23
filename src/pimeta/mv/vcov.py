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

Not implemented: within-subject designs (``time1`` / ``time2`` with ``phi``).
Those need the autocorrelation structure rather than a correlation matrix, and
an approximate version would be worse than the gap.  Asking for them raises.
"""

from __future__ import annotations

import math

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


# --------------------------------------------------------------------------- #
# vcalc
# --------------------------------------------------------------------------- #

def vcalc(vi, cluster=None, subgroup=None, obs=None, type=None,
          time1=None, time2=None, grp1=None, grp2=None, w1=None, w2=None,
          rho=None, phi=0.0, rho_names=None, r_compat=False):
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
        time1, time2: not supported; see the module docstring.
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
        phi: accepted for signature compatibility; only meaningful with
            ``time1`` / ``time2``, which are not implemented.  A nonzero value
            without a time index is an error rather than a silent no-op.
        rho_names: labels of the rows of a matrix ``rho``, in matrix order.
            Without it the matrix is indexed in first-appearance order of the
            factor, which is what R's ``unique()`` produces.
        r_compat: reproduce metafor exactly, including its silent discard of
            ``rho`` when neither ``obs`` nor ``type`` is given.  Default
            ``False``: the correlation that was asked for is the one used.

    Returns:
        ``(k, k)`` float array.

    Raises:
        ValueError: inconsistent lengths, a forgotten correlation, or a
            singular block that no argument explains.
        NotImplementedError: ``time1`` / ``time2`` were supplied.
    """
    vi = np.asarray(vi, dtype=float).ravel()
    k = vi.size
    if k == 0:
        raise ValueError("'vi' must not be empty")
    if np.any(vi < 0):
        raise ValueError("'vi' must be non-negative")
    if time1 is not None or time2 is not None:
        raise NotImplementedError(
            "within-subject designs (time1/time2 with phi) are not "
            "implemented; build the correlation matrix explicitly instead")
    if phi not in (0, 0.0, None):
        raise ValueError(
            "'phi' is the autocorrelation argument of a within-subject design "
            "and needs 'time1'/'time2', which are not implemented")

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
    V = np.diag(vi)

    if grp1 is not None:
        g1 = _as_vector(grp1, k, "grp1")
        g2 = _as_vector(grp2, k, "grp2")
        wa = _as_vector(w1, k, "w1")
        wb = _as_vector(w2, k, "w2")
        if wa is None:
            wa = np.ones(k)
        if wb is None:
            wb = np.ones(k)
        for i in range(k):
            for j in range(i + 1, k):
                if keys[i] != keys[j]:
                    continue
                if g1[i] == g1[j] and g2[i] == g2[j]:
                    corr = 1.0
                elif g1[i] == g1[j]:
                    corr = shared_control_corr(wa[i], wb[i], wb[j])
                elif g2[i] == g2[j]:
                    corr = shared_control_corr(wb[i], wa[i], wa[j])
                else:
                    corr = 0.0
                if corr != 0.0:
                    V[i, j] = V[j, i] = corr * sd[i] * sd[j]
        return V

    # ---- construct / observation correlations -----------------------------
    if obs_codes is None and typ_codes is None:
        if r_compat:
            # metafor: 'rho' is dropped and every pair is perfectly
            # correlated, which makes the block singular.
            for i in range(k):
                for j in range(i + 1, k):
                    if keys[i] == keys[j]:
                        V[i, j] = V[j, i] = sd[i] * sd[j]
            return V
        if rho is None:
            raise ValueError(
                "'rho' is required when neither 'obs' nor 'type' is given: "
                "two different effect sizes from one cluster are not "
                "perfectly correlated, so there is no defensible default. "
                "Pass 'rho=' for a constant within-cluster correlation, or "
                "'obs=' so that the estimates can be told apart. "
                "r_compat=True reproduces metafor's silent correlation of 1.")
        val = float(np.asarray(rho).ravel()[0])
        if not -1.0 <= val <= 1.0:
            raise ValueError("correlations must lie in [-1, 1]")
        for i in range(k):
            for j in range(i + 1, k):
                if keys[i] == keys[j]:
                    V[i, j] = V[j, i] = val * sd[i] * sd[j]
        return V

    if obs_codes is not None and typ_codes is None:
        pair = _pair_corr(rho, obs_levels, "obs", rho_names)
        if pair is None:
            raise ValueError("'rho' is required when 'obs' is given")
        for i in range(k):
            for j in range(i + 1, k):
                if keys[i] != keys[j]:
                    continue
                corr = (1.0 if obs_codes[i] == obs_codes[j]
                        else pair(obs_codes[i], obs_codes[j]))
                V[i, j] = V[j, i] = corr * sd[i] * sd[j]
        return V

    if typ_codes is not None and obs_codes is None:
        pair = _pair_corr(rho, typ_levels, "type", rho_names)
        if pair is None:
            raise ValueError("'rho' is required when 'type' is given")
        for i in range(k):
            for j in range(i + 1, k):
                if keys[i] != keys[j]:
                    continue
                corr = (1.0 if typ_codes[i] == typ_codes[j]
                        else pair(typ_codes[i], typ_codes[j]))
                V[i, j] = V[j, i] = corr * sd[i] * sd[j]
        return V

    # ---- both 'type' and 'obs' -------------------------------------------
    if rho is None:
        raise ValueError("'rho' is required when 'type' and 'obs' are given")
    if isinstance(rho, (list, tuple)) and len(rho) == 2:
        within_spec, between_spec = rho
    else:
        seq = np.asarray(rho, dtype=float).ravel() if np.ndim(rho) else None
        if seq is None or seq.size != 2:
            raise ValueError(
                "when 'type' and 'obs' are both given, 'rho' must supply the "
                "within-construct and between-construct correlations, e.g. "
                "rho=(0.6, 0.4)")
        within_spec, between_spec = float(seq[0]), float(seq[1])
    within = _pair_corr(within_spec, obs_levels, "obs", rho_names)
    between = _pair_corr(between_spec, typ_levels, "type", rho_names)
    for i in range(k):
        for j in range(i + 1, k):
            if keys[i] != keys[j]:
                continue
            if typ_codes[i] != typ_codes[j]:
                corr = between(typ_codes[i], typ_codes[j])
            elif obs_codes[i] == obs_codes[j]:
                corr = 1.0
            else:
                corr = within(obs_codes[i], obs_codes[j])
            V[i, j] = V[j, i] = corr * sd[i] * sd[j]
    return V
