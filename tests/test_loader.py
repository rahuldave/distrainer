import threading
import time

import pytest
from conftest import make_table

from distrainer.block import write_block
from distrainer.loader import LaneLoader
from distrainer.log import BlockLog
from distrainer.planner import lane


def build(store, n_segments=2, W=4, rank=0, n=2):
    fs, root = store
    log = BlockLog.create(fs, root, W=W, seed=1)
    for s in range(n_segments):
        refs = [write_block(fs, root, f"s{s}b{i}", make_table(2, s * 100 + i)) for i in range(W)]
        log.append(refs)
    return fs, root, log


def lanes_for(log, rank, n, start_seq=0, start_step=0, poll=0.02, timeout=5):
    s, first = start_seq, start_step
    while (segment := log.wait_segment(s, poll_s=poll, timeout_s=timeout)) is not None:
        yield segment, lane(segment, rank, n, first)
        s, first = s + 1, 0


def test_yields_lane_in_order_with_tables(store):
    fs, root, log = build(store, n_segments=2, W=4)
    log.end()
    out = list(LaneLoader(fs, root, lanes_for(log, rank=1, n=2), prefetch=2, threads=2))
    positions = [p for _, (p, _, _) in out]
    assert positions == [1, 3, 5, 7]
    for seg, (p, ref, table) in out:
        assert seg.seq == p // 4
        assert table.num_rows == 2
        assert table.column("block_id").to_pylist() == [ref.block_id] * 2
        assert ref is seg.blocks[p - seg.seq * 4]


def test_resume_start_skips_positions(store):
    fs, root, log = build(store, n_segments=2, W=4)
    log.end()
    out = list(LaneLoader(fs, root, lanes_for(log, 0, 2, start_seq=0, start_step=1)))
    assert [p for _, (p, _, _) in out] == [2, 4, 6]


def test_prefetch_spans_segment_boundary_while_writer_is_behind(store):
    fs, root, log = build(store, n_segments=1, W=4)
    consumed = []

    def writer():
        time.sleep(0.4)
        refs = [write_block(fs, root, f"late{i}", make_table(1, i)) for i in range(4)]
        log.append(refs)
        log.end()

    threading.Thread(target=writer, daemon=True).start()
    t0 = time.monotonic()
    for _seg, (p, _, _) in LaneLoader(fs, root, lanes_for(log, 0, 1), prefetch=2):
        consumed.append((p, time.monotonic() - t0))
    assert [p for p, _ in consumed] == [0, 1, 2, 3, 4, 5, 6, 7]
    # the first segment's blocks were delivered before the late writer committed
    assert consumed[3][1] < 0.4
    assert consumed[4][1] >= 0.35


def test_close_stops_early_and_errors_propagate(store):
    fs, root, log = build(store, n_segments=3, W=4)
    log.end()
    loader = LaneLoader(fs, root, lanes_for(log, 0, 1), prefetch=1, threads=1)
    it = iter(loader)
    first = next(it)
    assert first[1][0] == 0
    loader.close()
    loader.close()  # idempotent

    def broken():
        yield from lanes_for(log, 0, 1)
        raise RuntimeError("boom after lanes")

    with pytest.raises(RuntimeError, match="boom"):
        list(LaneLoader(fs, root, broken()))
    with pytest.raises(ValueError):
        LaneLoader(fs, root, iter([]), prefetch=0)


def test_missing_block_file_raises_in_consumer(store):
    fs, root, log = build(store, n_segments=1, W=4)
    log.end()
    seg = log.read_segment(0)
    fs.delete_file(f"{root}/{seg.blocks[2].locator}")
    with pytest.raises(OSError):
        list(LaneLoader(fs, root, lanes_for(log, 0, 1)))
