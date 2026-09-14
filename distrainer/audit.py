"""distrainer.audit: the per-rank JSONL trail of consumed blocks the verification harness reads.

Spec section 2.2: every consumed block is appended as one JSON line to
``<root>/audit/<run_name>/<attempt>-<rank>.jsonl``. On a local filesystem lines are appended and
flushed one by one (tail-able); on object stores the file is rewritten on ``flush``.
"""

from __future__ import annotations

import json
import re
import time
from dataclasses import asdict, dataclass
from typing import Any

import pyarrow.fs as pafs

from distrainer.storage import (
    ensure_dir,
    join,
    list_names,
    local_path,
    read_bytes,
    write_bytes,
)

AUDIT_DIR = "audit"
_FILE_RE = re.compile(r"^(\d+)-(\d+)(?:\.(\d+))?\.jsonl$")
PART_LINES = 10_000  # object stores: start a new part file after this many lines


@dataclass(frozen=True)
class AuditRecord:
    attempt: int
    rank: int
    world_size: int
    segment: int
    step: int
    position: int
    block_id: str
    ts: float

    def to_json(self) -> str:
        return json.dumps(asdict(self), sort_keys=True)

    @classmethod
    def from_json(cls, line: str) -> AuditRecord:
        d = json.loads(line)
        return cls(
            attempt=int(d["attempt"]),
            rank=int(d["rank"]),
            world_size=int(d["world_size"]),
            segment=int(d["segment"]),
            step=int(d["step"]),
            position=int(d["position"]),
            block_id=str(d["block_id"]),
            ts=float(d["ts"]),
        )


def audit_dir(root: str, run_name: str) -> str:
    return join(root, AUDIT_DIR, run_name)


def audit_filename(attempt: int, rank: int, part: int | None = None) -> str:
    return f"{attempt}-{rank}.jsonl" if part is None else f"{attempt}-{rank}.{part}.jsonl"


def next_attempt(fs: pafs.FileSystem, root: str, run_name: str) -> int:
    """One more than the highest attempt id already recorded for ``run_name`` (0 for a new run).

    Rank 0 calls this and broadcasts the result so every restart of the worker group, even from
    the same checkpoint, gets its own audit files.
    """
    attempts = [
        int(m.group(1))
        for n in list_names(fs, audit_dir(root, run_name))
        if (m := _FILE_RE.match(n))
    ]
    return max(attempts) + 1 if attempts else 0


class AuditWriter:
    def __init__(
        self,
        fs: pafs.FileSystem,
        root: str,
        run_name: str,
        attempt: int,
        rank: int,
        part_lines: int = PART_LINES,
    ):
        if "/" in run_name or not run_name:
            raise ValueError(f"invalid run_name {run_name!r}")
        self.fs = fs
        self.attempt = attempt
        self.rank = rank
        self.dir = audit_dir(root, run_name)
        self.path = join(self.dir, audit_filename(attempt, rank))
        ensure_dir(fs, self.dir)
        os_path = local_path(fs, self.path)
        self._local = os_path is not None
        self._closed = False
        # object stores: lines of the current part are buffered and the part is rewritten on
        # flush; a new part starts after part_lines so the rewrite stays bounded
        self._part_lines = part_lines
        self._part = 0
        self._lines: list[str] = []
        if not self._local:
            parts = [
                int(m.group(3) or 0)
                for n in list_names(fs, self.dir)
                if (m := _FILE_RE.match(n))
                and (int(m.group(1)), int(m.group(2))) == (attempt, rank)
            ]
            self._part = max(parts) + 1 if parts else 0
            self.path = join(self.dir, audit_filename(attempt, rank, self._part))
            write_bytes(fs, self.path, b"")  # claim the attempt now, not at the first flush
        self._handle: Any = open(os_path, "a", encoding="utf-8") if os_path is not None else None

    def append(
        self,
        world_size: int,
        segment: int,
        step: int,
        position: int,
        block_id: str,
        ts: float | None = None,
    ) -> AuditRecord:
        rec = AuditRecord(
            attempt=self.attempt,
            rank=self.rank,
            world_size=world_size,
            segment=segment,
            step=step,
            position=position,
            block_id=block_id,
            ts=time.time() if ts is None else ts,
        )
        if self._closed:
            raise ValueError("audit writer is closed")
        line = rec.to_json()
        if self._local:
            self._handle.write(line + "\n")
            self._handle.flush()
        else:
            self._lines.append(line)
            if len(self._lines) >= self._part_lines:
                self.flush()
                self._part += 1
                self._lines = []
                self.path = join(self.dir, audit_filename(self.attempt, self.rank, self._part))
        return rec

    def flush(self) -> None:
        if self._closed:
            return
        if self._local:
            self._handle.flush()
        elif self._lines:
            write_bytes(self.fs, self.path, ("\n".join(self._lines) + "\n").encode())

    def close(self) -> None:
        if self._closed:
            return
        self.flush()
        self._closed = True
        if self._local:
            self._handle.close()


def read_audit(fs: pafs.FileSystem, root: str, run_name: str) -> list[AuditRecord]:
    """All records of a run over all attempts and ranks, in (attempt, rank, file order)."""
    d = audit_dir(root, run_name)
    keyed: list[tuple[tuple[int, int, int, int], AuditRecord]] = []
    for name in list_names(fs, d):
        m = _FILE_RE.match(name)
        if not m:
            continue
        attempt, rank, part = int(m.group(1)), int(m.group(2)), int(m.group(3) or 0)
        text = read_bytes(fs, join(d, name)).decode()
        for idx, line in enumerate(ln for ln in text.splitlines() if ln.strip()):
            keyed.append(((attempt, rank, part, idx), AuditRecord.from_json(line)))
    keyed.sort(key=lambda kv: kv[0])
    return [rec for _, rec in keyed]
