"""Shared numeric primitives transcribed from R's C internals.

Kept in its own module so :mod:`.pooling` and :mod:`.tau2` can both use them
without a circular import.
"""

from __future__ import annotations

# metafor's convergence constants (``rma.uni``'s ``con`` list).
UNIROOT_TOL = 0.0001220703125   # .Machine$double.eps^0.25, R's uniroot tol
TAU2_THRESHOLD = 1e-5           # con$threshold
TAU2_MIN = 0.0                  # con$tau2.min
TAU2_MAX = 100.0                # con$tau2.max


def zeroin(
    f,
    ax: float,
    bx: float,
    tol: float = UNIROOT_TOL,
    maxiter: int = 1000,
) -> float:
    """Replicate R's ``uniroot`` (``C_zeroin2`` / Netlib ``zeroin``).

    Brent's method with the *exact* convergence criterion R uses
    (``tol_act = 2*eps*|b| + tol/2``, stop when ``|new_step| <= tol_act``), so
    the root matches ``uniroot(...)$root`` bit-for-bit. ``tol`` defaults to
    metafor's ``.Machine$double.eps^0.25``.
    """
    eps = 2.220446049250313e-16
    a = ax
    b = bx
    fa = f(a)
    fb = f(b)
    c = a
    fc = fa
    maxit = maxiter + 1

    if fa == 0.0:
        return a
    if fb == 0.0:
        return b

    while maxit:
        maxit -= 1
        prev_step = b - a
        if abs(fc) < abs(fb):
            # R's C_zeroin2 does this SEQUENTIALLY: ``a = b; b = c; c = a``,
            # so ``c`` ends up equal to the OLD ``b`` (``a`` was reassigned to
            # old ``b`` before the third statement runs). A naive 3-way tuple
            # rotate would instead give ``c = old a``, which is wrong.
            a, b, c = b, c, b
            fa, fb, fc = fb, fc, fb
        tol_act = 2.0 * eps * abs(b) + tol / 2.0
        new_step = (c - b) / 2.0

        if abs(new_step) <= tol_act or fb == 0.0:
            return b

        if abs(prev_step) >= tol_act and abs(fa) > abs(fb):
            cb = c - b
            if a == c:
                t1 = fb / fa
                p = cb * t1
                q = 1.0 - t1
            else:
                q = fa / fc
                t1 = fb / fc
                t2 = fb / fa
                p = t2 * (cb * q * (q - t1) - (b - a) * (t1 - 1.0))
                q = (q - 1.0) * (t1 - 1.0) * (t2 - 1.0)
            if p > 0.0:
                q = -q
            else:
                p = -p
            if (
                p < (0.75 * cb * q - abs(tol_act * q) / 2.0)
                and p < abs(prev_step * q) / 2.0
            ):
                new_step = p / q

        if abs(new_step) < tol_act:
            new_step = tol_act if new_step > 0.0 else -tol_act

        a = b
        fa = fb
        b += new_step
        fb = f(b)
        if (fb > 0.0 and fc > 0.0) or (fb < 0.0 and fc < 0.0):
            c = a
            fc = fa

    return b  # failed to converge (matches R's behaviour of returning b)
