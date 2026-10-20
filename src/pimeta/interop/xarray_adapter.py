"""``xarray`` output — labelled multi-dimensional results.

A network meta-analysis produces an ``n x n`` matrix of treatment contrasts.
As a bare ``numpy`` array that matrix is unusable without a side-table of
treatment names; as an ``xarray.DataArray`` the names travel with the data, and
``sel``, ``to_dataframe``, ``plot``, ``groupby`` and arithmetic all work.

Nothing here is imported at package import time — ``xarray`` is only touched
inside the two public functions, which raise a clear ``ImportError`` when it is
missing.
"""

from __future__ import annotations

import numpy as np

__all__ = ["to_xarray", "array_to_dataarray"]

#: Fields that hold an ``n x n`` treatment contrast matrix.
_MATRIX_FIELDS = (
    "TE_common", "seTE_common", "lower_common", "upper_common",
    "statistic_common", "pval_common",
    "TE_random", "seTE_random", "lower_random", "upper_random",
    "statistic_random", "pval_random",
    "TE_direct_common", "seTE_direct_common", "TE_direct_random",
    "seTE_direct_random", "lower_direct_common", "upper_direct_common",
    "lower_direct_random", "upper_direct_random",
    "Q_direct", "tau2_direct", "I2_direct",
    "prop_direct_common", "prop_direct_random",
    "TE_indirect_common", "seTE_indirect_common",
    "TE_indirect_random", "seTE_indirect_random",
    "P_common", "P_random",
    "Cov_common", "Cov_random",
)


def _require_xarray():
    try:
        import xarray as xr
    except ImportError as exc:                        # pragma: no cover - optional
        raise ImportError(
            "xarray output requires xarray (pip install pimeta[xarray])") from exc
    return xr


def array_to_dataarray(matrix, trts, name: str = "value"):
    """Wrap an ``n x n`` treatment matrix in a labelled ``DataArray``.

    Args:
        matrix: an ``(n, n)`` array-like.
        trts: treatment names, length ``n``.
        name: name to give the resulting array.

    Returns:
        An ``xarray.DataArray`` with dims ``("treat1", "treat2")``.

    Raises:
        ValueError: when the matrix is not square or does not match ``trts``.
    """
    xr = _require_xarray()
    arr = np.asarray(matrix, dtype=float)
    names = [str(t) for t in trts]
    if arr.ndim != 2 or arr.shape[0] != arr.shape[1]:
        raise ValueError(f"'matrix' must be square, got shape {arr.shape}")
    if arr.shape[0] != len(names):
        raise ValueError(
            f"'matrix' is {arr.shape[0]}x{arr.shape[1]} but 'trts' has {len(names)} names")
    return xr.DataArray(arr, dims=("treat1", "treat2"),
                        coords={"treat1": names, "treat2": names}, name=name)


def to_xarray(result, variables=None):
    """Convert a network meta-analysis result to a labelled ``xarray.Dataset``.

    Every ``n x n`` contrast matrix on the result becomes a variable with dims
    ``("treat1", "treat2")``; the P-score vector becomes a variable over a
    ``treatment`` dimension.  Scalars (Q, τ², I²) are attached as dataset
    attributes so they survive ``.to_netcdf()`` round-trips.

    Args:
        result: a :class:`pimeta.netmeta.NetmetaResult` (or any object exposing
            ``trts`` plus ``n x n`` matrix fields).
        variables: optional iterable of field names to include; defaults to
            every recognised matrix field present on the object.

    Returns:
        An ``xarray.Dataset``.

    Raises:
        ImportError: when xarray is not installed.
        TypeError: when ``result`` has no ``trts`` attribute.
        ValueError: when no matrix field is found.
    """
    xr = _require_xarray()
    trts = getattr(result, "trts", None)
    if trts is None:
        raise TypeError(f"{type(result).__name__} has no 'trts'; "
                        "to_xarray() expects a network meta-analysis result")
    names = [str(t) for t in trts]

    fields = list(variables) if variables is not None else [
        f for f in _MATRIX_FIELDS if isinstance(getattr(result, f, None), np.ndarray)
        and getattr(result, f).ndim == 2 and getattr(result, f).shape[0] == len(names)
    ]
    data_vars = {f: array_to_dataarray(getattr(result, f), names, f) for f in fields}

    for vec in ("pscore_common", "pscore_random"):
        values = getattr(result, vec, None)
        if values is not None and np.asarray(values).size == len(names):
            data_vars[vec] = xr.DataArray(
                np.asarray(values, dtype=float), dims=("treatment",),
                coords={"treatment": names}, name=vec)

    if not data_vars:
        raise ValueError(
            f"{type(result).__name__} exposes no '{len(names)}x{len(names)}' "
            "contrast matrix; pass 'variables=' explicitly")

    ds = xr.Dataset(data_vars)
    for scalar in ("k", "m", "n", "sm", "level", "method_tau", "Q", "tau2",
                   "I2", "Q_heterogeneity", "Q_inconsistency", "reference_group"):
        value = getattr(result, scalar, None)
        if isinstance(value, (str, int, float)) and not isinstance(value, bool):
            ds.attrs[scalar] = value
    return ds
