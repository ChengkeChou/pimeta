"""Parallel execution that cannot change the answer.

The only parallel execution this package does is over **independent
repetitions**: replications of a simulation, bootstrap draws, permutations.  A
repetition is a pure function of data that was drawn before the loop started,
so running the loop in a different order, or in a different process, cannot
change a single number.

That property is the point, and it is what separates this module from
``n_jobs`` flags elsewhere.  Parallelising an accumulation -- a running
meta-analysis, a leave-one-out sweep, anything whose value depends on the order
of the additions -- changes the last bits of the result, because floating-point
addition is not associative.  A user who is told "``n_jobs=8``" and silently gets
a different third decimal has been given a speed-up they did not ask for at the
price of a number they cannot reproduce.  Nothing here does that.

The contract for a callable passed to :func:`map_ordered`:

* it must depend only on its argument -- no module-level mutable state, no
  generator held in a closure;
* it must be picklable, which in practice means a module-level function or a
  :class:`functools.partial` of one (:func:`check_picklable` says so before any
  work starts, rather than failing in a child process).

Any randomness must be drawn **before** the call, one stream per repetition, so
that the stream a repetition consumes does not depend on how many workers there
are.  The call sites here do exactly that.
"""

from __future__ import annotations

import os
import pickle
from collections.abc import Callable, Iterable, Iterator, Sequence
from concurrent.futures import ProcessPoolExecutor
from contextlib import contextmanager
from typing import Any

__all__ = ["check_picklable", "map_ordered", "resolve_jobs"]

#: Environment variables that size a BLAS/OpenMP thread pool.  A worker here
#: runs one repetition at a time, so an inner pool of one thread per call is
#: what it wants; the parallelism is across repetitions.
_BLAS_VARS = ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS")


@contextmanager
def _single_threaded_workers() -> Iterator[None]:
    """Give child processes one BLAS thread each, then restore the caller's env.

    Inherited through the environment, which is why this has to happen before
    the pool is created: a child initialises its BLAS at import time, long
    before any ``initializer`` could run.  Without it, four workers on a
    many-core machine each allocate a full thread pool, and on Windows the
    allocation failure kills the worker mid-task -- ``map_ordered`` then raises
    :class:`~concurrent.futures.process.BrokenProcessPool` instead of returning
    numbers, which is a silent wrong answer with a visible crash on top.
    """
    saved = {name: os.environ.get(name) for name in _BLAS_VARS}
    try:
        for name in _BLAS_VARS:
            os.environ.setdefault(name, "1")
        yield
    finally:
        for name, value in saved.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value


def resolve_jobs(n_jobs: int | None, count: int) -> int:
    """Decide how many workers to use.

    Args:
        n_jobs: ``None`` or ``1`` for the serial loop, ``-1`` for every core,
            otherwise that many workers.
        count: how many items there are to run.  More workers than items would
            only add start-up cost.

    Returns:
        At least 1, at most ``count``.
    """
    if n_jobs is None:
        return 1
    jobs = int(n_jobs)
    if jobs < 0:
        jobs = os.cpu_count() or 1
    if jobs <= 1 or count <= 1:
        return 1
    return max(1, min(jobs, int(count)))


def check_picklable(fn: Callable[..., Any]) -> None:
    """Raise a readable error if ``fn`` cannot be sent to a worker process.

    Called before any worker starts, so the failure is reported where the caller
    can act on it rather than inside a child process as a traceback about a
    closure.
    """
    try:
        pickle.dumps(fn)
    except Exception as exc:  # noqa: BLE001 - any pickling failure is the point
        raise ValueError(
            "the callable cannot be pickled, so it cannot be run in another "
            "process: a lambda, a nested function or a closure over local "
            "state will not work.  Pass a module-level function, or a "
            "functools.partial of one, or use n_jobs=1") from exc


def map_ordered(
    fn: Callable[[Any], Any],
    items: Iterable[Any],
    *,
    n_jobs: int | None = None,
) -> list[Any]:
    """``[fn(x) for x in items]``, possibly in parallel, always in order.

    Args:
        fn: the per-item callable; see the module docstring for the contract.
        items: the independent repetitions.
        n_jobs: worker processes; ``None`` or ``1`` runs the plain loop in this
            process, which needs no pickling and is what the tests compare
            against.

    Returns:
        The results, in the order of ``items``.  With ``n_jobs > 1`` the order
        of *completion* differs, which is why the collection goes through
        :meth:`~concurrent.futures.Executor.map`, whose results are yielded in
        submission order.

    Raises:
        ValueError: ``fn`` cannot be pickled and ``n_jobs > 1``.
    """
    seq: Sequence[Any] = list(items)
    jobs = resolve_jobs(n_jobs, len(seq))
    if jobs == 1:
        return [fn(item) for item in seq]
    check_picklable(fn)
    with _single_threaded_workers():
        with ProcessPoolExecutor(max_workers=jobs) as pool:
            return list(pool.map(fn, seq))
