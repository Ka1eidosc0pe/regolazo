"""Runs engine calls off the event loop on a small dedicated pool, with optional serialization and a result cache.

Engine calls (fork, rollout, 1,000-sample distributions) can take seconds and may not be thread-safe, so they
never run on the event loop or on the thread pool that FastAPI uses for other work. By default calls are
serialized with a lock; set ``engine_thread_safe`` to allow up to ``engine_concurrency`` at once. Results of
rollout / distribution requests that carry an explicit seed are deterministic and cached (small LRU).
"""

import asyncio
import threading
from collections import OrderedDict
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor

from pydantic import BaseModel


class EngineRunner:
    def __init__(self, concurrency: int = 2, thread_safe: bool = False, cache_size: int = 32) -> None:
        self._executor = ThreadPoolExecutor(max_workers=max(1, concurrency), thread_name_prefix="engine")
        self._lock: threading.Lock | None = None if thread_safe else threading.Lock()
        self._cache: OrderedDict[str, BaseModel] = OrderedDict()
        self._cache_size = cache_size
        self._cache_lock = threading.Lock()

    def _call[T](self, fn: Callable[[], T]) -> T:
        if self._lock is None:
            return fn()
        with self._lock:
            return fn()

    async def run[T](self, fn: Callable[[], T]) -> T:
        return await asyncio.get_running_loop().run_in_executor(self._executor, self._call, fn)

    async def run_cached[M: BaseModel](self, key: str | None, fn: Callable[[], M]) -> M:
        """Like ``run``, but results are memoized under ``key`` (``None`` = do not cache)."""
        if key is None or self._cache_size <= 0:
            return await self.run(fn)
        with self._cache_lock:
            hit = self._cache.get(key)
            if hit is not None:
                self._cache.move_to_end(key)
        if hit is not None:
            return hit  # type: ignore[return-value]
        result = await self.run(fn)
        with self._cache_lock:
            self._cache[key] = result
            while len(self._cache) > self._cache_size:
                self._cache.popitem(last=False)
        return result

    def close(self) -> None:
        self._executor.shutdown(wait=False, cancel_futures=True)
