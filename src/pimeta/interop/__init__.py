"""Ecosystem interoperability — one set of protocols, many libraries.

The module is organised by **protocol**, not by vendor:

============================  ==========================================
:mod:`.estimator`             scikit-learn's duck-typed estimator contract
:mod:`.broom`                 ``tidy`` / ``glance`` / ``augment`` tables
:mod:`.statsmodels`           statsmodels-shaped results (``params``, ...)
:mod:`.tables`                pandas / polars / Arrow / dict / records
:mod:`.xarray_adapter`        labelled :math:`n \\times n` contrast arrays
:mod:`.networkx_adapter`      the evidence network as a graph
============================  ==========================================

Every optional dependency is resolved lazily, so ``import pimeta`` never pulls
in pandas, polars, pyarrow, scikit-learn, statsmodels, xarray or networkx.
:func:`install_methods` attaches the tidy-table protocol to the result classes
so that, for example, ``r.tidy()`` works directly on a fitted model.
"""

from __future__ import annotations

from .base import PROTOCOLS, as_2d, available, protocol_report, supports
from .broom import augment, glance, tidy
from .estimator import MetaEstimator, MetaRegressor
from .networkx_adapter import to_networkx
from .statsmodels import MetaModelInfo, MetaResults, as_results
from .tables import BACKENDS, to_frame
from .xarray_adapter import array_to_dataarray, to_xarray

__all__ = [
    # protocols
    "PROTOCOLS", "supports", "protocol_report", "available",
    "install_methods",
    # broom
    "tidy", "glance", "augment",
    # tables
    "BACKENDS", "to_frame",
    # estimator
    "MetaRegressor", "MetaEstimator",
    # statsmodels
    "MetaResults", "MetaModelInfo", "as_results",
    # labelled / graph output
    "to_xarray", "array_to_dataarray", "to_networkx",
    # helpers
    "as_2d",
]


# --------------------------------------------------------------------------- #
# tidy-table methods on result objects
# --------------------------------------------------------------------------- #
def _m_tidy(self, **kwargs):
    return tidy(self, **kwargs)


def _m_glance(self, **kwargs):
    return glance(self, **kwargs)


def _m_augment(self, **kwargs):
    return augment(self, **kwargs)


def _m_to_table(self, backend: str = "auto", rows=None):
    from ..data_compat import to_dataframe

    table = to_dataframe(self, rows=rows)
    if hasattr(table, "to_dict"):                    # a pandas DataFrame
        table = table.to_dict(orient="list")
    return to_frame(table, backend=backend)


def install_methods(namespace, *, overwrite: bool = False) -> int:
    """Attach ``tidy``/``glance``/``augment``/``to_table`` to result classes.

    Called once from :mod:`pimeta.__init__` so the tidy-table protocol is
    available on every result object without repeating the code in ~40 dataclass
    definitions.

    Args:
        namespace: a module ``globals()`` mapping (pimeta's own).
        overwrite: replace methods that already exist (default ``False``).

    Returns:
        The number of classes that received the methods.
    """
    count = 0
    for obj in list(namespace.values()):
        if not isinstance(obj, type):
            continue
        if not getattr(obj, "__module__", "").startswith("pimeta"):
            continue
        if not hasattr(obj, "to_dict"):
            continue
        if not overwrite and getattr(obj, "_interop_installed", False):
            continue
        for name, func in (("tidy", _m_tidy), ("glance", _m_glance),
                           ("augment", _m_augment), ("to_table", _m_to_table)):
            if overwrite or name not in vars(obj):
                setattr(obj, name, func)
        setattr(obj, "_interop_installed", True)
        count += 1
    return count
