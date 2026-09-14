"""distrainer.loader: per-rank prefetching iterator over lanes of blocks (backend B).

Spec sections 4 and 5. The consumer hands in a generator of ``(segment, lane)`` pairs, where a
lane is the ``(position, BlockRef)`` list from :func:`distrainer.planner.lane`. A producer
thread walks that generator (so a blocking ``wait_segment`` inside it never stalls blocks that
are already fetched), a small thread pool reads the Parquet files, and a bounded queue of
``prefetch`` in-flight blocks provides backpressure. Items come out in lane order.
"""

from __future__ import annotations

import queue
import threading
from collections.abc import Iterator
from concurrent.futures import Future, ThreadPoolExecutor
from typing import Any

import pyarrow as pa
import pyarrow.fs as pafs

from distrainer.block import BlockRef, read_block
from distrainer.log import Segment

Lane = list[tuple[int, BlockRef]]
Item = tuple[Segment, tuple[int, BlockRef, pa.Table]]

_DONE = object()


class LaneLoader:
    def __init__(
        self,
        fs: pafs.FileSystem,
        root: str,
        lanes: Iterator[tuple[Segment, Lane]],
        prefetch: int = 2,
        threads: int = 2,
    ):
        if prefetch <= 0 or threads <= 0:
            raise ValueError("prefetch and threads must be positive")
        self.fs = fs
        self.root = root
        self._lanes = lanes
        self._queue: queue.Queue[Any] = queue.Queue(maxsize=prefetch)
        self._pool = ThreadPoolExecutor(max_workers=threads, thread_name_prefix="distrainer-load")
        self._stop = threading.Event()
        self._producer: threading.Thread | None = None
        self._closed = False

    # ---- producer side ----

    def _produce(self) -> None:
        try:
            for segment, lane in self._lanes:
                for position, ref in lane:
                    if self._stop.is_set():
                        return
                    fut = self._pool.submit(read_block, self.fs, self.root, ref)
                    self._put((segment, position, ref, fut))
            self._put(_DONE)
        except BaseException as exc:  # surfaced to the consumer
            self._put(exc)

    def _put(self, item: Any) -> None:
        while not self._stop.is_set():
            try:
                self._queue.put(item, timeout=0.1)
                return
            except queue.Full:
                continue

    # ---- consumer side ----

    def __iter__(self) -> Iterator[Item]:
        if self._producer is None:
            self._producer = threading.Thread(
                target=self._produce, name="distrainer-lanes", daemon=True
            )
            self._producer.start()
        try:
            while True:
                item = self._queue.get()
                if item is _DONE:
                    return
                if isinstance(item, BaseException):
                    raise item
                segment, position, ref, fut = item
                table = fut.result()
                yield segment, (position, ref, table)
        finally:
            self.close()

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        self._stop.set()
        # unblock a producer stuck on a full queue, then drop whatever is pending
        while True:
            try:
                item = self._queue.get_nowait()
            except queue.Empty:
                break
            if isinstance(item, tuple):
                fut: Future[pa.Table] = item[3]
                fut.cancel()
        if self._producer is not None and self._producer is not threading.current_thread():
            self._producer.join(timeout=5)
        self._pool.shutdown(wait=False, cancel_futures=True)
