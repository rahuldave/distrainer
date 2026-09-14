"""distrainer.log: the block log, an append-only sequence of immutable segments.

Spec sections 2.2, 3.1 and 4. This module holds the ``Segment`` and ``LogMeta`` records and
their JSON forms, and ``BlockLog``: the reader (list, wait, read), the single writer (buffer,
shuffle, commit atomically) and retention (``gc``).
"""

from __future__ import annotations

import json
import random
import re
import time
from collections.abc import Iterator
from dataclasses import dataclass, field
from typing import Any

import pyarrow.fs as pafs

from distrainer.block import BlockRef, block_num_rows
from distrainer.storage import (
    delete,
    ensure_dir,
    exists,
    join,
    list_names,
    read_bytes,
    write_atomic,
)

SCHEMA_VERSION = 1
LOG_DIR = "log"
END_MARKER = "_END"
META_FILE = "_meta.json"


def segment_filename(seq: int) -> str:
    return f"{seq:08d}.json"


@dataclass(frozen=True)
class LogMeta:
    W: int
    seed: int
    created_by: str = "distrainer"
    schema_version: int = SCHEMA_VERSION

    def to_json(self) -> str:
        return json.dumps(
            {
                "W": self.W,
                "seed": self.seed,
                "created_by": self.created_by,
                "schema_version": self.schema_version,
            },
            sort_keys=True,
        )

    @classmethod
    def from_json(cls, text: str) -> LogMeta:
        d = json.loads(text)
        return cls(
            W=int(d["W"]),
            seed=int(d["seed"]),
            created_by=str(d.get("created_by", "distrainer")),
            schema_version=int(d.get("schema_version", SCHEMA_VERSION)),
        )


@dataclass(frozen=True)
class Segment:
    seq: int
    W: int
    seed: int
    pass_idx: int
    blocks: list[BlockRef]
    meta: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if len(self.blocks) != self.W:
            raise ValueError(
                f"segment {self.seq} has {len(self.blocks)} blocks, expected W={self.W}"
            )

    def positions(self) -> range:
        return range(self.seq * self.W, (self.seq + 1) * self.W)

    def to_json(self) -> str:
        return json.dumps(
            {
                "seq": self.seq,
                "W": self.W,
                "seed": self.seed,
                "pass": self.pass_idx,
                "blocks": [b.to_dict() for b in self.blocks],
                "meta": self.meta,
                "schema_version": SCHEMA_VERSION,
            },
            sort_keys=True,
        )

    @classmethod
    def from_json(cls, text: str) -> Segment:
        d = json.loads(text)
        return cls(
            seq=int(d["seq"]),
            W=int(d["W"]),
            seed=int(d["seed"]),
            pass_idx=int(d.get("pass", 0)),
            blocks=[BlockRef.from_dict(b) for b in d["blocks"]],
            meta=dict(d.get("meta") or {}),
        )


_SEGMENT_RE = re.compile(r"^(\d{8})\.json$")


class BlockLog:
    """The segment log. Readers and writers share this class; only one writer per log.

    Layout under ``<root>/log/``: ``_meta.json`` (W, seed), ``00000000.json`` ... (segments in
    final order), optional ``_END``. Commit protocol: block files first, then the segment file
    written atomically; readers trust only committed segment files.
    """

    def __init__(self, fs: pafs.FileSystem, root: str):
        self.fs = fs
        self.root = root
        self.log_dir = join(root, LOG_DIR)
        self._meta: LogMeta | None = None
        self._segments: dict[int, Segment] = {}  # bounded cache, see _cache
        self._next_seq: int | None = None

    SEGMENT_CACHE_SIZE = 8

    def _cache(self, seg: Segment) -> Segment:
        self._segments[seg.seq] = seg
        while len(self._segments) > self.SEGMENT_CACHE_SIZE:
            self._segments.pop(next(iter(self._segments)))
        return seg

    # ---- creation / metadata ----

    @classmethod
    def create(
        cls, fs: pafs.FileSystem, root: str, W: int, seed: int, created_by: str = "distrainer"
    ) -> BlockLog:
        """Initialise an empty log. Fails if ``log/_meta.json`` already exists."""
        if W <= 0:
            raise ValueError(f"W must be positive, got {W}")
        log = cls(fs, root)
        ensure_dir(fs, log.log_dir)
        if exists(fs, log._meta_path):
            raise FileExistsError(f"log already exists at {log.log_dir}")
        write_atomic(
            fs, log._meta_path, LogMeta(W=W, seed=seed, created_by=created_by).to_json().encode()
        )
        log._next_seq = 0
        return log

    @classmethod
    def open(cls, fs: pafs.FileSystem, root: str) -> BlockLog:
        log = cls(fs, root)
        log.meta()
        return log

    @property
    def _meta_path(self) -> str:
        return join(self.log_dir, META_FILE)

    @property
    def _end_path(self) -> str:
        return join(self.log_dir, END_MARKER)

    def exists(self) -> bool:
        return exists(self.fs, self._meta_path)

    def meta(self) -> LogMeta:
        if self._meta is None:
            if not self.exists():
                raise FileNotFoundError(f"no block log at {self.log_dir} (missing {META_FILE})")
            self._meta = LogMeta.from_json(read_bytes(self.fs, self._meta_path).decode())
        return self._meta

    @property
    def W(self) -> int:
        return self.meta().W

    # ---- reader side ----

    def segment_path(self, seq: int) -> str:
        return join(self.log_dir, segment_filename(seq))

    def has_segment(self, seq: int) -> bool:
        return seq in self._segments or exists(self.fs, self.segment_path(seq))

    def read_segment(self, seq: int) -> Segment:
        seg = self._segments.get(seq)
        if seg is None:
            if not exists(self.fs, self.segment_path(seq)):
                raise FileNotFoundError(f"segment {seq} not committed in {self.log_dir}")
            text = read_bytes(self.fs, self.segment_path(seq)).decode()
            version = int(json.loads(text).get("schema_version", SCHEMA_VERSION))
            if version != SCHEMA_VERSION:
                raise ValueError(
                    f"segment {seq} has schema_version {version}, expected {SCHEMA_VERSION}"
                )
            seg = Segment.from_json(text)
            if seg.seq != seq or seg.W != self.W:
                raise ValueError(f"segment file {seq} is inconsistent: seq={seg.seq}, W={seg.W}")
            self._cache(seg)
        return seg

    def ended(self) -> bool:
        return exists(self.fs, self._end_path)

    def wait_segment(
        self, seq: int, poll_s: float = 1.0, timeout_s: float | None = None
    ) -> Segment | None:
        """Block until segment ``seq`` is committed.

        Returns ``None`` only when the log has ended and ``seq`` does not exist. Raises
        ``TimeoutError`` after ``timeout_s`` seconds of waiting (``None`` = wait forever).
        """
        deadline = None if timeout_s is None else time.monotonic() + timeout_s
        while True:
            if self.has_segment(seq):
                return self.read_segment(seq)
            if self.ended():
                # a segment may have been committed between the two checks
                return self.read_segment(seq) if self.has_segment(seq) else None
            if deadline is not None and time.monotonic() >= deadline:
                raise TimeoutError(f"segment {seq} did not appear within {timeout_s}s")
            time.sleep(poll_s)

    def committed_seqs(self) -> list[int]:
        """All committed segment numbers from one directory listing."""
        return sorted(
            int(m.group(1))
            for n in list_names(self.fs, self.log_dir)
            if (m := _SEGMENT_RE.match(n))
        )

    def first_seq(self) -> int | None:
        """Lowest committed segment number (after ``gc`` this is no longer 0)."""
        seqs = self.committed_seqs()
        return seqs[0] if seqs else None

    def last_seq(self) -> int | None:
        """Highest committed segment number (one directory listing), or ``None`` if empty."""
        seqs = [
            int(m.group(1))
            for n in list_names(self.fs, self.log_dir)
            if (m := _SEGMENT_RE.match(n))
        ]
        return max(seqs) if seqs else None

    def segments(self, start: int | None = None) -> Iterator[Segment]:
        """Committed segments in order from ``start`` (default: the lowest), up to the first gap."""
        seq = self.first_seq() if start is None else start
        if seq is None:
            return
        while self.has_segment(seq):
            yield self.read_segment(seq)
            seq += 1

    def next_pass_differs(self, segment: Segment) -> bool:
        """True at a pass boundary: the next segment belongs to another pass or the log ended.

        In streaming mode, if the next segment is not committed yet and the log has not ended,
        the answer is ``False`` (unknown counts as "same pass").
        """
        if self.has_segment(segment.seq + 1):
            return self.read_segment(segment.seq + 1).pass_idx != segment.pass_idx
        return self.ended()

    # ---- writer side ----

    def append(
        self,
        blocks: list[BlockRef],
        *,
        pass_idx: int = 0,
        meta: dict[str, Any] | None = None,
        verify_blocks: bool = True,
    ) -> Segment:
        """Shuffle ``blocks`` with ``Random(hash((seed, seq)))`` and commit them as the next
        segment."""
        W = self.W
        if len(blocks) != W:
            raise ValueError(f"append needs exactly W={W} blocks, got {len(blocks)}")
        if self.ended():
            raise RuntimeError("cannot append to an ended log")
        if verify_blocks:
            self._verify_blocks(blocks)
        seq = self._allocate_seq()
        if exists(self.fs, self.segment_path(seq)):
            raise FileExistsError(
                f"segment {seq} is already committed in {self.log_dir}: another writer is active "
                "(one writer per log) or this instance is stale"
            )
        order = list(blocks)
        random.Random(hash((self.meta().seed, seq))).shuffle(order)
        segment = Segment(
            seq=seq,
            W=W,
            seed=self.meta().seed,
            pass_idx=pass_idx,
            blocks=order,
            meta={"created_at": time.time(), **(meta or {})},
        )
        write_atomic(self.fs, self.segment_path(seq), segment.to_json().encode())
        self._cache(segment)
        self._next_seq = seq + 1
        return segment

    def _verify_blocks(self, blocks: list[BlockRef]) -> None:
        """Every block file must exist and its Parquet footer must agree with the ref."""
        bad: list[str] = []
        for b in blocks:
            try:
                rows = block_num_rows(self.fs, self.root, b)
            except Exception:  # missing, truncated, or unreadable
                bad.append(b.block_id)
                continue
            if rows != b.num_rows:
                bad.append(b.block_id)
        if bad:
            raise FileNotFoundError(
                f"block files missing or corrupt for {bad[:5]}{'...' if len(bad) > 5 else ''}"
            )

    def _allocate_seq(self) -> int:
        if self._next_seq is None:
            last = self.last_seq()
            self._next_seq = 0 if last is None else last + 1
        return self._next_seq

    def end(self) -> None:
        """Mark the producer finished (idempotent)."""
        if not self.ended():
            write_atomic(self.fs, self._end_path, b"")

    # ---- retention ----

    def gc(self, keep_from_seq: int, delete_blocks: bool = True) -> list[int]:
        """Delete segments ``< keep_from_seq`` and, if ``delete_blocks``, the block files only
        they reference.

        Blocks referenced by any *committed* later segment are left alone. A producer that will
        re-reference old blocks in segments it has not written yet (a multi-pass hook) must call
        with ``delete_blocks=False``. Returns the deleted segment numbers.
        """
        last = self.last_seq()
        if last is None or keep_from_seq <= 0:
            return []
        keep_from_seq = min(keep_from_seq, last + 1)
        kept_locators = {
            b.locator
            for seq in range(keep_from_seq, last + 1)
            if self.has_segment(seq)
            for b in self.read_segment(seq).blocks
        }
        deleted: list[int] = []
        for seq in range(keep_from_seq):
            if not self.has_segment(seq):
                continue
            if delete_blocks:
                for b in self.read_segment(seq).blocks:
                    if b.locator not in kept_locators:
                        delete(self.fs, join(self.root, b.locator))
                        kept_locators.add(b.locator)  # deleted once even if repeated below
            delete(self.fs, self.segment_path(seq))
            delete(self.fs, join(self.log_dir, f"{seq:08d}.rows.parquet"))
            self._segments.pop(seq, None)
            deleted.append(seq)
        return deleted
