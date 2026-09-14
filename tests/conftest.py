import pyarrow as pa
import pytest

from distrainer.block import BlockRef, write_block
from distrainer.log import Segment
from distrainer.storage import StorageConfig, build_filesystem


@pytest.fixture
def store(tmp_path):
    """(fs, root) for a fresh local store."""
    return build_filesystem(StorageConfig(kind="local", path=str(tmp_path / "store")))


def make_table(n_rows: int, offset: int = 0) -> pa.Table:
    return pa.table({"x": [float(i) for i in range(offset, offset + n_rows)]})


def make_refs(n: int, rows: int = 4) -> list[BlockRef]:
    return [BlockRef(f"b{i:04d}", f"blocks/b{i:04d}.parquet", rows) for i in range(n)]


def make_segment(seq: int, W: int, pass_idx: int = 0, seed: int = 1) -> Segment:
    return Segment(seq=seq, W=W, seed=seed, pass_idx=pass_idx, blocks=make_refs(W))


@pytest.fixture
def written_blocks(store):
    fs, root = store
    return fs, root, [write_block(fs, root, f"b{i}", make_table(3, i * 10)) for i in range(3)]
