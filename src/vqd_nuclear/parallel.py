"""Parallel execution helpers with safe worker resolution."""
from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from typing import Callable, Iterable, TypeVar

T = TypeVar("T")
U = TypeVar("U")


def resolve_n_jobs(n_jobs: int | str | None, available_workers: int, task_count: int | None = None) -> int:
    """Return the number of workers to use for independent tasks."""
    available = max(1, int(available_workers))
    if n_jobs is None or str(n_jobs).lower() == "all":
        requested = available
    else:
        requested = max(1, int(n_jobs))
    if task_count is not None:
        requested = min(requested, max(1, int(task_count)))
    return requested


def chunksize(n_items: int, workers: int) -> int:
    """Small but nonzero chunksize for process pools."""
    return max(1, int(n_items) // max(1, int(workers) * 4))


def parallel_map(function: Callable[[T], U], items: Iterable[T], *, n_jobs: int | str | None, available_workers: int) -> list[U]:
    """Evaluate independent items with threads while preserving input order."""
    values = list(items)
    if not values:
        return []
    workers = resolve_n_jobs(n_jobs, available_workers, len(values))
    if workers == 1:
        return [function(item) for item in values]
    with ThreadPoolExecutor(max_workers=workers) as executor:
        return list(executor.map(function, values))


def parallel_map_processes(function: Callable[[T], U], items: Iterable[T], *, n_jobs: int | str | None, available_workers: int) -> list[U]:
    """Evaluate independent CPU-heavy top-level tasks with processes."""
    values = list(items)
    if not values:
        return []
    workers = resolve_n_jobs(n_jobs, available_workers, len(values))
    if workers == 1:
        return [function(item) for item in values]
    with ProcessPoolExecutor(max_workers=workers) as executor:
        return list(executor.map(function, values, chunksize=chunksize(len(values), workers)))
