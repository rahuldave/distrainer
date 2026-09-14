"""distrainer.loader: per-rank prefetching iterator over lanes of blocks (backend B).

Spec sections 4 and 5. The consumer hands in a generator of ``(segment, lane)`` pairs, where a
lane is the ``(position, BlockRef)`` list from :func:`distrainer.planner.lane`; the generator may
also be a callable taking a ``threading.Event`` so that a blocking ``BlockLog.wait_segment`` inside
it can be interrupted by ``close()``. A producer thread walks the generator (so waiting for the
next segment never stalls blocks that are already fetched), a small thread pool reads the Parquet
files, and a bounded queue of ``prefetch`` in-flight blocks provides backpressure. Items come out
in lane order. A loader can be iterated once.
"""

from __future__ import annotations

import queue
import threading
from collections.abc import Callable, Iterator
from concurrent.futures import Future, ThreadPoolExecutor
from typing import Any, cast

import pyarrow as pa
import pyarrow.fs as pafs

from distrainer.block import BlockRef, read_block
from distrainer.log import Segment

Lane = list[tuple[int, BlockRef]]
Lanes = Iterator[tuple[Segment, Lane]]
Item = tuple[Segment, tuple[int, BlockRef, pa.Table]]

_DONE = object()


class LaneLoader:
    def __init__(
        self,
        fs: pafs.FileSystem,
        root: str,
        lanes: Lanes | Callable[[threading.Event], Lanes],
        prefetch: int = 2,
        threads: int = 2,
    ):
        if prefetch <= 0 or threads <= 0:
            raise ValueError("prefetch and threads must be positive")
        self.fs = fs
        self.root = root
        self.stop_event = threading.Event()
        self._lanes: Lanes = cast(Lanes, lanes(self.stop_event) if callable(lanes) else lanes)
        self._queue: queue.Queue[Any] = queue.Queue(maxsize=prefetch)
        self._pool = ThreadPoolExecutor(max_workers=threads, thread_name_prefix="distrainer-load")
        self._producer: threading.Thread | None = None
        self._closed = False
        self._lock = threading.Lock()

    # ---- producer side ----

    def _produce(self) -> None:
        try:
            for segment, lane in self._lanes:
                for position, ref in lane:
                    if self.stop_event.is_set():
                        return
                    fut = self._pool.submit(read_block, self.fs, self.root, ref)
                    if not self._put((segment, position, ref, fut)):
                        fut.cancel()
                        return
                if self.stop_event.is_set():
                    return
            self._put(_DONE)
        except BaseException as exc:  # surfaced to the consumer
            self._put(exc)
        finally:
            close = getattr(self._lanes, "close", None)
            if close is not None:
                close()  # the producer thread owns the generator, so this is safe here

    def _put(self, item: Any) -> bool:
        while not self.stop_event.is_set():
            try:
                self._queue.put(item, timeout=0.1)
                return True
            except queue.Full:
                continue
        return False

    # ---- consumer side ----

    def __iter__(self) -> Iterator[Item]:
        with self._lock:
            if self._closed or self._producer is not None:
                raise RuntimeError("a LaneLoader can be iterated once; create a new one to restart")
            self._producer = threading.Thread(
                target=self._produce, name="distrainer-lanes", daemon=True
            )
            self._producer.start()
        try:
            while True:
                item = self._queue.get()
                if item is _DONE or self.stop_event.is_set():
                    return
                if isinstance(item, BaseException):
                    raise item
                segment, position, ref, fut = item
                table = fut.result()
                yield segment, (position, ref, table)
        finally:
            self.close()

    def close(self) -> None:
        """Stop prefetching. Safe to call from any thread and more than once."""
        with self._lock:
            if self._closed:
                return
            self._closed = True
        self.stop_event.set()
        # drop pending items, cancel their reads, then wake a consumer blocked on get()
        while True:
            try:
                item = self._queue.get_nowait()
            except queue.Empty:
                break
            if isinstance(item, tuple):
                fut: Future[pa.Table] = item[3]
                fut.cancel()
        try:
            self._queue.put_nowait(_DONE)
        except queue.Full:
            pass
        if self._producer is not None and self._producer is not threading.current_thread():
            self._producer.join(timeout=5)
        self._pool.shutdown(wait=False, cancel_futures=True)
