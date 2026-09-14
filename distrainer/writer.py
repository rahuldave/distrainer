"""distrainer.writer: turn blocks into a log.

Spec sections 2.2 and 4. ``BatchWriter`` emits a finished corpus at t=0, one pass per epoch:
each pass permutes the *whole* block list with ``Random(hash((seed, pass)))`` and only then
cuts it into segments of ``W`` (``BlockLog.append`` re-shuffles within each segment).
``StreamingWriter`` is the helper base for producers that run alongside training: it buffers
``k * W`` blocks and commits each segment by sampling ``W`` of them.
"""

from __future__ import annotations

import random
from typing import Literal

from distrainer.block import BlockRef
from distrainer.log import BlockLog, Segment

Tail = Literal["error", "drop", "wrap"]


def pass_order(blocks: list[BlockRef], seed: int, pass_idx: int) -> list[BlockRef]:
    """Global permutation of ``blocks`` for one pass; deterministic in ``(seed, pass_idx)``."""
    order = list(blocks)
    random.Random(hash((seed, pass_idx))).shuffle(order)
    return order


def cut_segments(order: list[BlockRef], W: int, tail: Tail = "error") -> list[list[BlockRef]]:
    """Cut a permuted block list into chunks of exactly ``W``.

    ``tail`` says what to do with the last ``len % W`` blocks: ``error`` (default), ``drop``
    them for this pass, or ``wrap`` the chunk up with the first blocks of the same pass so no
    segment is short (those blocks are seen twice in the pass).
    """
    n = len(order)
    if n == 0:
        return []
    remainder = n % W
    chunks = [order[i : i + W] for i in range(0, n - remainder, W)]
    if remainder:
        if tail == "error":
            raise ValueError(
                f"{n} blocks is not a multiple of W={W}; pass tail='drop' or tail='wrap'"
            )
        if tail == "wrap":
            last = order[n - remainder :]
            fill = [b for b in order if b not in last][: W - remainder]
            if len(fill) < W - remainder:
                raise ValueError(f"cannot wrap: only {n} blocks for W={W}")
            chunks.append(last + fill)
    return chunks


class BatchWriter:
    """Write ``passes`` passes over ``blocks`` into ``log`` and end it."""

    def __init__(
        self,
        log: BlockLog,
        blocks: list[BlockRef],
        passes: int = 1,
        tail: Tail = "error",
        verify_blocks: bool = True,
    ):
        if passes <= 0:
            raise ValueError(f"passes must be positive, got {passes}")
        if len({b.block_id for b in blocks}) != len(blocks):
            raise ValueError("duplicate block_id in corpus")
        self.log = log
        self.blocks = list(blocks)
        self.passes = passes
        self.tail = tail
        self.verify_blocks = verify_blocks

    def plan(self) -> list[tuple[int, list[BlockRef]]]:
        """``(pass_idx, chunk)`` for every segment that ``run`` would append, in order."""
        W = self.log.W
        seed = self.log.meta().seed
        return [
            (p, chunk)
            for p in range(self.passes)
            for chunk in cut_segments(pass_order(self.blocks, seed, p), W, self.tail)
        ]

    def run(self) -> list[Segment]:
        segments = [
            self.log.append(
                chunk, pass_idx=p, meta={"writer": "batch"}, verify_blocks=self.verify_blocks
            )
            for p, chunk in self.plan()
        ]
        self.log.end()
        return segments


class StreamingWriter:
    """Producer-side buffer: ``push`` blocks as they arrive, segments are committed as they fill.

    With ``shuffle_buffer_segments = k`` the writer holds up to ``k * W`` blocks and each committed
    segment is a random sample of ``W`` of them (``k = 1`` is the plain window). ``flush`` commits
    what is left in full segments (``tail`` decides the remainder) and ``close`` also ends the log.
    """

    def __init__(
        self,
        log: BlockLog,
        shuffle_buffer_segments: int = 1,
        pass_idx: int = 0,
        verify_blocks: bool = True,
    ):
        if shuffle_buffer_segments <= 0:
            raise ValueError("shuffle_buffer_segments must be positive")
        self.log = log
        self.k = shuffle_buffer_segments
        self.pass_idx = pass_idx
        self.verify_blocks = verify_blocks
        self._buffer: list[BlockRef] = []
        self._rng = random.Random(hash((log.meta().seed, "stream")))
        self.committed: list[Segment] = []

    @property
    def buffered(self) -> int:
        return len(self._buffer)

    def push(self, block: BlockRef) -> Segment | None:
        """Add one block; returns the segment committed by this push, if any."""
        self._buffer.append(block)
        if len(self._buffer) >= self.k * self.log.W:
            return self._commit_one()
        return None

    def _commit_one(self) -> Segment:
        W = self.log.W
        idx = sorted(self._rng.sample(range(len(self._buffer)), W))
        chosen = [self._buffer[i] for i in idx]
        drop = set(idx)
        self._buffer = [b for i, b in enumerate(self._buffer) if i not in drop]
        seg = self.log.append(
            chosen,
            pass_idx=self.pass_idx,
            meta={"writer": "stream"},
            verify_blocks=self.verify_blocks,
        )
        self.committed.append(seg)
        return seg

    def flush(self, tail: Tail = "drop") -> list[Segment]:
        """Commit remaining full segments; ``tail`` handles the last partial one."""
        W = self.log.W
        out: list[Segment] = []
        while len(self._buffer) >= W:
            out.append(self._commit_one())
        if self._buffer:
            if tail == "error":
                raise ValueError(f"{len(self._buffer)} blocks left in the buffer, fewer than W={W}")
            if tail == "wrap":
                seen = [b for s in self.committed for b in s.blocks]
                chunk = cut_segments(self._buffer + seen, W, "drop")[:1]
                if chunk:
                    out.append(
                        self.log.append(
                            chunk[0],
                            pass_idx=self.pass_idx,
                            meta={"writer": "stream", "wrapped": True},
                            verify_blocks=self.verify_blocks,
                        )
                    )
                    self.committed.append(out[-1])
            self._buffer = []
        return out

    def close(self, tail: Tail = "drop") -> list[Segment]:
        out = self.flush(tail)
        self.log.end()
        return out
