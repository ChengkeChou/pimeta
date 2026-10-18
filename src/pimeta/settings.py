"""Global settings — ``meta::settings.meta`` / ``gs`` / ``setvals`` parity.

The R packages keep a large, mutable global option list. This module provides a
small, explicit equivalent: a plain ``dict`` per package plus helpers to read
(``gs``) and bulk-assign (``setvals``) settings. Nothing is truly global — the
dictionaries live in this module and callers can pass overrides explicitly.
"""

from __future__ import annotations

import copy

__all__ = ["gs", "setvals", "settings_meta", "settings_netmeta",
           "META_DEFAULTS", "NETMETA_DEFAULTS"]

#: Defaults for the ``meta`` package (subset of ``meta::settings.meta()``).
META_DEFAULTS = {
    "method.tau": "REML",
    "method.random.ci": "classic",
    "method.predict": "V",
    "method.common.ci": "classic",
    "level.ma": 0.95,
    "level.ma.common": 0.95,
    "level.ma.random": 0.95,
    "level.predict": 0.95,
    "hakn": False,
    "adhoc.hakn": "",
    "common": True,
    "random": True,
    "prediction": False,
    "overall": True,
    "sm": "",
    "incr": 0.5,
    "method.incr": "only0",
    "allincr": False,
    "addincr": False,
    "MH.exact": False,
    "Q.Cochrane": True,
    "tau.common": False,
    "backtransf": True,
    "digits": 4,
    "digits.se": 4,
    "digits.tau2": 4,
    "digits.I2": 1,
    "digits.pval": 4,
    "digits.CI": 4,
    "warn": True,
    "warn.deprecated": True,
}

#: Defaults for the ``netmeta`` package (subset of ``settings.netmeta()``).
NETMETA_DEFAULTS = {
    "method.tau": "REML",
    "level.ma": 0.95,
    "reference.group": "",
    "baseline.reference": True,
    "common": True,
    "random": True,
    "sm": "",
    "incr": 0.5,
    "method.incr": "only0",
    "allincr": False,
    "addincr": False,
    "seq": "",
    "nchar.trts": 666,
    "nchar.studlab": 666,
    "digits": 4,
    "digits.se": 4,
    "digits.tau2": 4,
    "digits.I2": 1,
    "digits.pval": 4,
    "digits.CI": 4,
    "warn": True,
    "warn.deprecated": True,
}

# Mutable, per-package setting stores.
settings_meta: dict = copy.deepcopy(META_DEFAULTS)
settings_netmeta: dict = copy.deepcopy(NETMETA_DEFAULTS)


def gs(name=None, package="meta"):
    """``meta::gs`` — get a (package) setting.

    Args:
        name: setting name. ``None`` returns a copy of the whole setting dict.
            Both the R spelling (``"method.tau"``) and the Python spelling
            (``"method_tau"``) are accepted.
        package: ``"meta"`` or ``"netmeta"``.

    Returns:
        The setting value (or the full ``dict``).

    Raises:
        KeyError: when the setting is unknown.
        ValueError: for an unsupported ``package``.
    """
    store = _store(package)
    if name is None:
        return copy.deepcopy(store)
    key = _norm(name)
    if key not in store:
        raise KeyError(f"unknown setting {name!r} for package {package!r}")
    return store[key]


def setvals(*args, package="meta", **kwargs):
    """``meta::setvals`` — bulk-assign (package) settings.

    Accepts ``name=value`` keyword arguments and/or ``(name, value)`` pairs, in
    the same spirit as ``meta::setvals()``. Underscores in names are normalised
    to dots so that ``setvals(method_tau="DL")`` sets ``"method.tau"``.

    Args:
        *args: optional ``(name, value)`` pairs.
        package: ``"meta"`` or ``"netmeta"``.
        **kwargs: setting name/value pairs.

    Returns:
        A copy of the updated setting ``dict``.
    """
    store = _store(package)
    for pair in args:
        if not (isinstance(pair, (tuple, list)) and len(pair) == 2):
            raise ValueError("positional arguments must be (name, value) pairs")
        store[_norm(pair[0])] = pair[1]
    for k, v in kwargs.items():
        store[_norm(k)] = v
    return copy.deepcopy(store)


def _norm(name):
    return str(name).replace("_", ".")


def _store(package):
    pkg = str(package).lower()
    if pkg == "meta":
        return settings_meta
    if pkg == "netmeta":
        return settings_netmeta
    raise ValueError("package must be 'meta' or 'netmeta'")
