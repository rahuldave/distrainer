"""distrainer.block: ``BlockRef`` and Parquet read/write of one block on a pyarrow filesystem.

Spec sections 2.2 and 4. A block is one rank's batch stored as one Parquet file under
``<root>/blocks/<block_id>.parquet``; every row carries the ``block_id`` column.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import pyarrow as pa
import pyarrow.fs as pafs
import pyarrow.parquet as pq

from distrainer.storage import ensure_dir, join, write_atomic

BLOCK_ID_COLUMN = "block_id"
BLOCKS_DIR = "blocks"


@dataclass(frozen=True)
class BlockRef:
    block_id: str
    locator: str
    num_rows: int
    meta: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "block_id": self.block_id,
            "locator": self.locator,
            "num_rows": self.num_rows,
            "meta": dict(self.meta),
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> BlockRef:
        return cls(
            block_id=str(d["block_id"]),
            locator=str(d["locator"]),
            num_rows=int(d["num_rows"]),
            meta=dict(d.get("meta") or {}),
        )


def validate_block_id(block_id: str) -> str:
    if not isinstance(block_id, str) or not block_id or "/" in block_id or block_id.startswith("."):
        raise ValueError(f"invalid block_id {block_id!r}: non-empty, no '/', no leading '.'")
    return block_id


def block_locator(block_id: str) -> str:
    return f"{BLOCKS_DIR}/{validate_block_id(block_id)}.parquet"


def write_block(
    fs: pafs.FileSystem,
    root: str,
    block_id: str,
    table: pa.Table,
    meta: dict[str, Any] | None = None,
) -> BlockRef:
    """Write ``table`` as block ``block_id``; adds or validates the ``block_id`` column."""
    validate_block_id(block_id)
    if BLOCK_ID_COLUMN in table.column_names:
        ids = table.column(BLOCK_ID_COLUMN).unique().to_pylist()
        if ids and ids != [block_id]:
            raise ValueError(
                f"table column {BLOCK_ID_COLUMN!r} has values {ids!r}, expected only {block_id!r}"
            )
    else:
        table = table.append_column(
            BLOCK_ID_COLUMN, pa.array([block_id] * table.num_rows, type=pa.string())
        )
    locator = block_locator(block_id)
    ensure_dir(fs, join(root, BLOCKS_DIR))
    sink = pa.BufferOutputStream()
    pq.write_table(table, sink)
    write_atomic(fs, join(root, locator), sink.getvalue().to_pybytes())
    return BlockRef(
        block_id=block_id, locator=locator, num_rows=table.num_rows, meta=dict(meta or {})
    )


def validate_locator(locator: str) -> str:
    parts = locator.split("/")
    if not locator or locator.startswith("/") or ".." in parts or "" in parts:
        raise ValueError(f"invalid locator {locator!r}: must be a relative path inside the store")
    return locator


def read_block(fs: pafs.FileSystem, root: str, ref: BlockRef) -> pa.Table:
    return pq.read_table(join(root, validate_locator(ref.locator)), filesystem=fs)


def block_num_rows(fs: pafs.FileSystem, root: str, ref: BlockRef) -> int:
    """Row count from the Parquet footer; raises if the file is missing or truncated."""
    return pq.read_metadata(join(root, validate_locator(ref.locator)), filesystem=fs).num_rows
