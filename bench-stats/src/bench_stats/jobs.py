"""
Job management module
"""

import logging
import logging.handlers
import multiprocessing
from collections.abc import Callable, Hashable, Iterator, Mapping
from concurrent.futures import Executor, ProcessPoolExecutor
from contextlib import contextmanager
from queue import Queue
from typing import NamedTuple

logger = logging.getLogger(__name__)

# A unit of work that `run_jobs` runs, with a label that names it in the logs
type Job[R] = tuple[str, Callable[[], R]]

# Identifies a job that `run_jobs` runs
type JobKey = Hashable


def setup_worker_logging(log_queue: Queue[logging.LogRecord], level: int) -> None:
    """Put a worker's log records on `log_queue` for its parent process to write out"""
    root_logger = logging.getLogger()
    root_logger.setLevel(level)
    root_logger.addHandler(logging.handlers.QueueHandler(log_queue))
    logging.captureWarnings(True)


@contextmanager
def worker_pool(n_jobs: int) -> Iterator[Executor | None]:
    """Start a pool that runs up to `n_jobs` jobs at once, or no pool when `n_jobs` is 1"""
    if n_jobs == 1:
        yield None
        return

    # Writing one log file from several processes is not supported, so the workers
    # send their records here and this process does all the writing.
    root_logger = logging.getLogger()
    with multiprocessing.Manager() as manager:
        listener = logging.handlers.QueueListener(
            log_queue := manager.Queue(),
            *root_logger.handlers,
            respect_handler_level=True,
        )
        listener.start()
        try:
            with ProcessPoolExecutor(
                max_workers=n_jobs,
                initializer=setup_worker_logging,
                initargs=(log_queue, root_logger.level),
            ) as executor:
                yield executor
        finally:
            listener.stop()


class JobResults[K: JobKey, R](NamedTuple):
    """The results of the jobs that succeeded, by key, and the number that failed"""

    succeeded: dict[K, R]
    n_failures: int


def run_jobs[K: JobKey, R](
    jobs: Mapping[K, Job[R]],
    executor: Executor | None,
) -> JobResults[K, R]:
    """
    Run every job, in this process or spread over `executor`.

    A job that raises is logged under its label and left out of the results.
    The results of the other jobs keep the order of `jobs`.
    """
    futures = None
    if executor is not None:
        futures = {key: executor.submit(job) for key, (_, job) in jobs.items()}

    succeeded: dict[K, R] = {}
    n_failures = 0
    for key, (label, job) in jobs.items():
        try:
            succeeded[key] = job() if futures is None else futures[key].result()
        except Exception:
            logger.exception("Failed: %s", label)
            n_failures += 1

    return JobResults(succeeded, n_failures)
