import asyncio
import logging
import math
import os
import pickle
import sys
import time
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
from functools import cache
from multiprocessing import get_context
from pathlib import Path

from dial_rag.utils import int_env_var


def _cgroup_cpu_quota() -> int | None:
    """Number of cpus the container is allowed to use, or None if unrestricted.

    os.cpu_count() reports the cpus of the host and ignores the cgroup quota,
    so on its own it would size the pools for the whole node instead of for
    the limits of the pod.
    """
    try:
        # cgroup v2: "<quota> <period>", or "max <period>" when unrestricted
        quota, period = Path("/sys/fs/cgroup/cpu.max").read_text().split()
        return None if quota == "max" else max(1, math.ceil(int(quota) / int(period)))
    except (OSError, ValueError):
        pass

    try:
        # cgroup v1: a negative quota means unrestricted
        quota = int(Path("/sys/fs/cgroup/cpu/cpu.cfs_quota_us").read_text())
        period = int(Path("/sys/fs/cgroup/cpu/cpu.cfs_period_us").read_text())
        if quota > 0 and period > 0:
            return max(1, math.ceil(quota / period))
    except (OSError, ValueError):
        pass

    return None


def _available_cpu_count() -> int:
    host_cpus = os.cpu_count() or 1
    quota = _cgroup_cpu_quota()
    return min(host_cpus, quota) if quota is not None else host_cpus


CPU_COUNT = _available_cpu_count()


# 1 thread here, because inference is already parallelized inside the model,
# and we expect the batching to be done by the caller
EMBED_DOCUMENTS_WORKERS = int_env_var("EMBED_DOCUMENTS_WORKERS", 1)

# We do not want question answering to be blocked by the documents indexing
EMBED_QUERY_WORKERS = int_env_var("EMBED_QUERY_WORKERS", 1)

# Legacy way to set number of workers
DOCUMENT_LOADERS_WORKERS = int_env_var("DOCUMENT_LOADERS_WORKERS", max(1, CPU_COUNT - 2))

# The spawned indexing workers import the whole document loading stack
# (unstructured, docarray, pdfplumber, ...) on their first task and keep it
# resident for as long as they live. Recycling them after a number of tasks
# returns that memory to the OS instead of holding it until the pod restarts.
# 0 keeps the workers for the whole lifetime of the pool.
INDEXING_POOL_MAX_TASKS_PER_CHILD = int_env_var("INDEXING_POOL_MAX_TASKS_PER_CHILD", 16)

# Recycling alone only releases memory while documents keep being indexed, so
# the pool is also shut down completely once nothing has been indexed for a
# while, which brings an idle pod back to its baseline. 0 disables the idle
# shutdown and keeps the pool until the process exits.
INDEXING_POOL_IDLE_TIMEOUT = int_env_var("INDEXING_POOL_IDLE_TIMEOUT", 300)

IDLE_CHECK_INTERVAL = 30


logger = logging.getLogger(__name__)


def _resolve_max_tasks_per_child() -> int | None:
    """Tasks after which an indexing worker is replaced, None to never replace it.

    ProcessPoolExecutor deadlocks when the only worker of the pool reaches the
    limit and has to be replaced, because nothing is left to pick the queued
    tasks up. It is fixed in python 3.13, before that the recycling needs at
    least two workers to be safe.
    """
    if INDEXING_POOL_MAX_TASKS_PER_CHILD <= 0:
        return None

    if DOCUMENT_LOADERS_WORKERS < 2 and sys.version_info < (3, 13):
        logger.warning(
            f"Indexing workers recycling is disabled: it deadlocks with "
            f"{DOCUMENT_LOADERS_WORKERS} worker before python 3.13. "
            f"The pool is still shut down when idle to release its memory."
        )
        return None

    return INDEXING_POOL_MAX_TASKS_PER_CHILD


class UnpicklableExceptionError(RuntimeError):
    pass


def _run_in_process_wrapper(func, *args, **kwargs):
    try:
        return func(*args, **kwargs)
    except Exception as e:
        try:
            # python has an issue if unpicklable exception will be passed between processes
            # https://github.com/python/cpython/issues/120810
            # The exception created with kwargs could cause the issue
            pickle.loads(pickle.dumps(e))
        except Exception as pe:
            logger.exception(pe)
            # Unpicklable exception could break the process pool and cause the following error:
            # `concurrent.futures.process.BrokenProcessPool: A child process terminated abruptly, the process pool is not usable anymore`
            # To avoid this, we raise a custom exception with the original traceback in the __cause__ attribute
            raise UnpicklableExceptionError("Unpicklable exception raised in subprocess") from e
        raise


class CpuPools:
    indexing_embeddings_pool: ThreadPoolExecutor
    query_embeddings_pool: ThreadPoolExecutor

    def __init__(self) -> None:
        # Using process pool for indexing to avoid GIL limitations.
        # It is created on demand and shut down when idle, so the memory of
        # the imports its workers do is not held for the lifetime of the pod.
        self._indexing_cpu_pool: ProcessPoolExecutor | None = None
        self._indexing_pool_lock = asyncio.Lock()
        self._indexing_tasks_in_flight = 0
        self._indexing_idle_since: float | None = None
        self._idle_watchdog: asyncio.Task | None = None

        self.indexing_embeddings_pool = ThreadPoolExecutor(
            max_workers=EMBED_DOCUMENTS_WORKERS,
            thread_name_prefix="indexing_embeddings"
        )

        # TODO: Do we need a separate pool for query embeddings?
        self.query_embeddings_pool = ThreadPoolExecutor(
            max_workers=EMBED_QUERY_WORKERS,
            thread_name_prefix="query_embeddings"
        )

    def _run_in_pool(self, pool, func, *args, **kwargs):
        return asyncio.get_running_loop().run_in_executor(pool, func, *args, **kwargs)

    async def _get_indexing_cpu_pool(self) -> ProcessPoolExecutor:
        async with self._indexing_pool_lock:
            if self._indexing_cpu_pool is None:
                max_tasks_per_child = _resolve_max_tasks_per_child()
                logger.info(
                    f"Starting indexing cpu pool: {DOCUMENT_LOADERS_WORKERS} workers, "
                    f"max_tasks_per_child={max_tasks_per_child}"
                )
                self._indexing_cpu_pool = ProcessPoolExecutor(
                    max_workers=DOCUMENT_LOADERS_WORKERS,
                    # Spawn is used to avoid inheriting the file descriptors from the parent process
                    mp_context=get_context("spawn"),
                    max_tasks_per_child=max_tasks_per_child,
                )
                self._start_idle_watchdog()
            return self._indexing_cpu_pool

    def _start_idle_watchdog(self) -> None:
        if INDEXING_POOL_IDLE_TIMEOUT <= 0:
            return
        if self._idle_watchdog is not None and not self._idle_watchdog.done():
            return
        self._idle_watchdog = asyncio.create_task(self._shutdown_indexing_pool_when_idle())

    async def _shutdown_indexing_pool_when_idle(self) -> None:
        """Shut the indexing pool down once nothing has been indexed for a while.

        Exits after the shutdown, a new watchdog is started together with the
        next pool.
        """
        try:
            while True:
                await asyncio.sleep(min(IDLE_CHECK_INTERVAL, INDEXING_POOL_IDLE_TIMEOUT))

                async with self._indexing_pool_lock:
                    if self._indexing_cpu_pool is None:
                        return
                    if self._indexing_tasks_in_flight or self._indexing_idle_since is None:
                        continue
                    idle_for = time.monotonic() - self._indexing_idle_since
                    if idle_for < INDEXING_POOL_IDLE_TIMEOUT:
                        continue

                    pool = self._indexing_cpu_pool
                    self._indexing_cpu_pool = None
                    self._indexing_idle_since = None

                logger.info(
                    f"Indexing cpu pool was idle for {idle_for:.0f}s, "
                    f"shutting it down to release the memory of its workers"
                )
                # shutdown() waits for the workers to exit, so it is done
                # outside of the lock to not block the tasks starting a new pool
                await asyncio.get_running_loop().run_in_executor(None, pool.shutdown)
                return
        except asyncio.CancelledError:
            raise
        except Exception as e:
            logger.exception(e)

    async def run_in_indexing_cpu_pool(self, func, *args, **kwargs):
        pool = await self._get_indexing_cpu_pool()
        self._indexing_tasks_in_flight += 1
        try:
            return await self._run_in_pool(pool, _run_in_process_wrapper, func, *args, **kwargs)
        finally:
            self._indexing_tasks_in_flight -= 1
            self._indexing_idle_since = time.monotonic()

    def run_in_indexing_embeddings_pool(self, func, *args, **kwargs):
        return self._run_in_pool(self.indexing_embeddings_pool, func, *args, **kwargs)

    def run_in_query_embeddings_pool(self, func, *args, **kwargs):
        return self._run_in_pool(self.query_embeddings_pool, func, *args, **kwargs)

    @staticmethod
    @cache
    def instance():
        return CpuPools()


async def warmup_cpu_pools():
    """Warm up the pools to avoid the first call overhead"""
    cpu_pools = CpuPools.instance()
    await cpu_pools.run_in_indexing_cpu_pool(sum, range(10))
    await cpu_pools.run_in_indexing_embeddings_pool(sum, range(10))
    await cpu_pools.run_in_query_embeddings_pool(sum, range(10))


def run_in_indexing_cpu_pool(func, *args, **kwargs):
    return CpuPools.instance().run_in_indexing_cpu_pool(func, *args, **kwargs)


def run_in_indexing_embeddings_pool(func, *args, **kwargs):
    return CpuPools.instance().run_in_indexing_embeddings_pool(func, *args, **kwargs)


def run_in_query_embeddings_pool(func, *args, **kwargs):
    return CpuPools.instance().run_in_query_embeddings_pool(func, *args, **kwargs)
