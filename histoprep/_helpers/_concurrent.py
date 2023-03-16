from __future__ import annotations

__all__ = ["worker_init", "prepare_worker_pool", "close_pool"]

from collections.abc import Callable, Iterable
from functools import partial
from pathlib import Path

from mpire import WorkerPool


def worker_init(worker_state, reader_class, path: Path, backend: str) -> None:  # noqa
    """Worker initialization function for concurrent functions with reader."""
    worker_state["reader"] = reader_class(path, backend)


def prepare_worker_pool(
    reader,  # noqa
    worker_fn: Callable,
    iterable_of_args: Iterable,
    iterable_length: int,
    num_workers: int,
) -> tuple[WorkerPool | None, Iterable]:
    """Prepare worker pool and iterable."""
    if num_workers <= 1:
        return None, (worker_fn({"reader": reader}, *args) for args in iterable_of_args)
    # Prepare pool.
    init_fn = partial(
        worker_init,
        reader_class=reader.__class__,
        path=reader.path,
        backend=reader.backend.BACKEND_NAME,
    )
    pool = WorkerPool(n_jobs=num_workers, use_worker_state=True)
    iterable_of_args = pool.imap(
        func=worker_fn,
        iterable_of_args=iterable_of_args,
        iterable_len=iterable_length,
        worker_init=init_fn,
    )
    return pool, iterable_of_args


def close_pool(pool: WorkerPool | None) -> None:
    if pool is not None:
        pool.terminate()
