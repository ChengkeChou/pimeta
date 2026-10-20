"""Protocol detection and shared helpers for :mod:`pimeta.interop`.

The design principle of the whole ``interop`` package is that compatibility is
expressed as **protocols**, not as integrations with individual libraries.
Implementing the estimator protocol once makes a model usable by
``scikit-learn``'s ``Pipeline``/``GridSearchCV``, by ``joblib``/``dask``/``ray``
parallel backends, and by ``optuna``/``mlflow``/``shap`` — without importing any
of them.  The same holds for the tidy-table protocol (``tidy``/``glance``/
``augment``) and for the statsmodels result protocol.

Nothing in this module imports an optional third-party package at module import
time; every backend is resolved lazily so ``import pimeta`` stays cheap and
dependency-free.
"""

from __future__ import annotations

from typing import Any

import numpy as np

__all__ = [
    "PROTOCOLS",
    "available",
    "supports",
    "protocol_report",
    "as_2d",
    "coefficient_names",
]


#: Protocol name -> the attributes/methods a object must expose.
#:
#: ``all`` means every listed attribute must be present.  Protocols are
#: intentionally conservative: they describe what downstream tools actually
#: touch, not every method a real implementation happens to have.
PROTOCOLS: dict[str, tuple[str, ...]] = {
    # scikit-learn's duck-typed estimator contract.  ``GridSearchCV`` and
    # ``clone`` only need these four; the private tags are optional.
    "estimator": ("fit", "predict", "get_params", "set_params"),
    # broom-style tidy tables (R's broom / tidymodels convention).
    "tidy": ("tidy", "glance", "augment"),
    # The read-only subset of statsmodels' results API.
    "statsmodels": ("params", "bse", "conf_int", "cov_params", "nobs"),
    # Plain table extraction.
    "frame": ("to_dataframe",),
    "table": ("to_dict",),
    # NumPy array protocol.
    "array": ("__array__",),
}


def available() -> dict[str, bool]:
    """Report which optional frame backends are importable.

    Returns:
        A ``{module: bool}`` mapping for ``pandas``, ``polars`` and ``pyarrow``.
        ``dict`` and ``records`` are always available and therefore not listed.
    """
    out: dict[str, bool] = {}
    for name in ("pandas", "polars", "pyarrow", "sklearn", "statsmodels",
                 "xarray", "networkx"):
        try:                                          # pragma: no branch - trivial
            __import__(name)
            out[name] = True
        except Exception:                             # pragma: no cover - env dependent
            out[name] = False
    return out


def supports(obj: Any, protocol: str = "table") -> bool:
    """Return whether ``obj`` satisfies ``protocol``.

    Args:
        obj: any object.
        protocol: a key of :data:`PROTOCOLS`.

    Returns:
        ``True`` when every attribute named in the protocol exists.
    """
    attrs = PROTOCOLS.get(protocol)
    if attrs is None:
        raise KeyError(f"unknown protocol {protocol!r}; "
                       f"choices: {', '.join(sorted(PROTOCOLS))}")
    return all(hasattr(obj, a) for a in attrs)


def protocol_report(obj: Any) -> dict[str, bool]:
    """Return ``{protocol: supported}`` for every known protocol.

    Useful for debugging "why won't this object go into a Pipeline?".

    Args:
        obj: any object.

    Returns:
        A mapping from protocol name to a boolean.
    """
    return {name: supports(obj, name) for name in PROTOCOLS}


def as_2d(x: Any, n: int | None = None) -> tuple[np.ndarray, list[str] | None]:
    """Coerce ``x`` to a 2-D float matrix, plus optional column names.

    ``None`` becomes a ``(n, 0)`` matrix (intercept-only model).  A 1-D sequence
    becomes a single column.  A pandas/polars frame contributes its column names.

    Args:
        x: ``None``, a 1-D/2-D array-like, or a pandas/polars ``DataFrame``.
        n: required row count when ``x`` is ``None``.

    Returns:
        ``(matrix, column_names)`` where ``column_names`` is ``None`` unless a
        frame with string column labels was passed.
    """
    names: list[str] | None = None
    if x is None:
        if n is None:
            raise ValueError("'n' is required when 'x' is None")
        return np.empty((int(n), 0), dtype=float), None

    # pandas / polars / anything exposing .to_numpy or .columns
    if hasattr(x, "columns") and hasattr(x, "to_numpy"):
        names = [str(c) for c in x.columns]
        arr = np.asarray(x.to_numpy(), dtype=float)
    elif hasattr(x, "columns") and hasattr(x, "to_dict") and not isinstance(x, dict):
        names = [str(c) for c in x.columns]
        from ..data_compat import as_columns

        cols = as_columns(x)
        arr = np.column_stack([np.asarray(cols[c], dtype=float) for c in names])
    else:
        arr = np.asarray(x, dtype=float)

    if arr.ndim == 0:
        arr = arr.reshape(1, 1)
    elif arr.ndim == 1:
        arr = arr.reshape(-1, 1)
    elif arr.ndim > 2:
        raise ValueError(f"'x' must be at most 2-dimensional, got {arr.ndim}")
    return arr, names


def coefficient_names(obj: Any, p: int) -> list[str]:
    """Return ``p`` coefficient labels for ``obj``.

    Mirrors metafor's naming: an intercept is ``"intrcpt"`` and unlabelled
    moderators are ``"V1"``, ``"V2"``, ...  Honours ``coefnames``/``names``/
    ``trts`` when the result object carries them (set by ``rma(..., mods=...)``
    or by a formula-expanded design matrix).

    Args:
        obj: a fitted result object.
        p: number of coefficients.

    Returns:
        A list of ``p`` strings.
    """
    from ..s3 import _coefnames

    return _coefnames(obj, p)
