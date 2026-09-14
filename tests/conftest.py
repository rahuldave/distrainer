import fsspec
import pyarrow as pa
import pyarrow.fs as pafs
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


class MemoryFS(pafs.PyFileSystem):
    """A non-local pyarrow filesystem (fsspec memory) standing in for an object store."""

    def __init__(self):
        fs = fsspec.filesystem("memory")
        fs.store.clear()
        super().__init__(pafs.FSSpecHandler(fs))


class AppendNextSegments:
    """A ``SegmentHook`` for tests (entry ``conftest:AppendNextSegments``): at every segment end
    it appends a segment of fresh blocks (ids ``s<seq>b<i>``) and ends the log once ``segments``
    segments exist. Every call is recorded on the class so tests can inspect what rank 0 saw."""

    calls: list[dict] = []

    def __init__(self, config=None, segments: int = 2, rows: int = 2):
        self.config = config
        self.segments = segments
        self.rows = rows

    def on_segment_end(self, model, ledger, log, ctx) -> None:
        type(self).calls.append(
            {"model": model, "ledger": ledger, "log": log, "ctx": ctx, "config": self.config}
        )
        nxt = ledger.segment + 1
        if log.ended() or log.has_segment(nxt):
            return  # segment ends are delivered at least once (restarts replay them)
        if nxt < self.segments:
            refs = [
                write_block(log.fs, log.root, f"s{nxt}b{i}", make_table(self.rows, i))
                for i in range(log.W)
            ]
            log.append(refs, pass_idx=0, meta={"writer": "hook"})
        else:
            log.end()


@pytest.fixture(autouse=True)
def _clear_hook_calls():
    AppendNextSegments.calls.clear()
    yield
    AppendNextSegments.calls.clear()


class NotAHook:
    """Accepts the factory call but has no ``on_segment_end``."""

    def __init__(self, config=None, **kwargs):
        self.config = config
