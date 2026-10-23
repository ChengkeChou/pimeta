"""One place that decides what a "confidence level" argument means.

A meta-analysis API has to name the same quantity in three places — how wide an
interval should be — and ``pimeta`` grew up with three spellings of it:

======================  =========================================
``0.95``                the confidence level as a fraction, matching
                        metafor's public arguments and R's ``conf.level``
``95``                  the same thing as a percentage, which is how
                        ``meta`` spells it
``0.05``                the *alpha*, which is how metafor stores it
                        internally as ``x$level``
======================  =========================================

A caller cannot be expected to remember which function wants which, and getting
it wrong is silent: passing ``0.95`` to a function that reads alpha produces a
5%-wide interval, and passing ``95`` produces ``nan``. Both happened.

So every entry point normalises through :func:`level_to_alpha` or
:func:`level_to_percent`. The disambiguation rule is the only one that works:

- ``>= 1``    a percentage (``95``),
- ``> 0.5``   a confidence level as a fraction (``0.95``),
- otherwise   already an alpha (``0.05``).

The rule is **idempotent on alphas**, so a value can pass through it twice
without changing; that is what lets it be applied at every boundary without
having to trace which layer resolved it already.
"""

from __future__ import annotations

__all__ = ["level_to_alpha", "level_to_percent"]


def level_to_alpha(level) -> float:
    """Normalise any of the three spellings to a tail probability.

    Args:
        level: ``0.95``, ``95`` or ``0.05``.

    Returns:
        The alpha, e.g. ``0.05`` for all three of those inputs.

    Raises:
        ValueError: the value is not a usable confidence specification.
    """
    value = float(level)
    if not (value > 0.0 and value < 100.0) or value in (0.5, 1.0):
        raise ValueError(
            "argument 'level' must be a confidence level (95 or 0.95) or an "
            f"alpha (0.05), got {level!r}")
    if value >= 1.0:
        return 1.0 - value / 100.0
    if value > 0.5:
        return 1.0 - value
    return value


def level_to_percent(level) -> float:
    """Normalise any of the three spellings to a percentage (``95``)."""
    return 100.0 * (1.0 - level_to_alpha(level))
