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
    exists,
    join,
    list_names,
    local_path,
    read_bytes,
    write_bytes,
)

AUDIT_DIR = "audit"
_FILE_RE = re.compile(r"^(\d+)-(\d+)\.jsonl$")


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


def audit_filename(attempt: int, rank: int) -> str:
    return f"{attempt}-{rank}.jsonl"


class AuditWriter:
    def __init__(self, fs: pafs.FileSystem, root: str, run_name: str, attempt: int, rank: int):
        self.fs = fs
        self.attempt = attempt
        self.rank = rank
        self.dir = audit_dir(root, run_name)
        self.path = join(self.dir, audit_filename(attempt, rank))
        ensure_dir(fs, self.dir)
        os_path = local_path(fs, self.path)
        self._local = os_path is not None
        self._lines: list[str] = []
        if not self._local and exists(fs, self.path):
            self._lines = read_bytes(fs, self.path).decode().splitlines()
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
        line = rec.to_json()
        if self._local:
            self._handle.write(line + "\n")
            self._handle.flush()
        else:
            self._lines.append(line)
        return rec

    def flush(self) -> None:
        if self._local:
            self._handle.flush()
        else:
            write_bytes(self.fs, self.path, ("\n".join(self._lines) + "\n").encode())

    def close(self) -> None:
        self.flush()
        if self._local:
            self._handle.close()


def read_audit(fs: pafs.FileSystem, root: str, run_name: str) -> list[AuditRecord]:
    """All records of a run over all attempts and ranks, ordered by (attempt, rank, line)."""
    d = audit_dir(root, run_name)
    out: list[AuditRecord] = []
    for name in list_names(fs, d):
        m = _FILE_RE.match(name)
        if not m:
            continue
        text = read_bytes(fs, join(d, name)).decode()
        out.extend(AuditRecord.from_json(line) for line in text.splitlines() if line.strip())
    out.sort(key=lambda r: (r.attempt, r.rank, r.ts, r.position))
    return out
