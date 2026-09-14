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


def lanes_for(log, rank, n, start_seq=0, start_step=0, poll=0.02, timeout=5, stop=None):
    s, first = start_seq, start_step
    while (segment := log.wait_segment(s, poll_s=poll, timeout_s=timeout, stop=stop)) is not None:
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
    # the boundary is where the consumer waited: a visible gap between positions 3 and 4,
    # and no comparable gap inside either segment
    gaps = [b - a for (_, a), (_, b) in zip(consumed, consumed[1:], strict=False)]
    assert gaps[3] == max(gaps) and gaps[3] >= 0.2


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


def test_close_from_another_thread_unblocks_a_waiting_consumer(store):
    fs, root, log = build(store, n_segments=1, W=4)  # not ended: the lanes generator will wait
    loader = LaneLoader(fs, root, lambda stop: lanes_for(log, 0, 1, stop=stop, timeout=None))
    got = []
    done = threading.Event()

    def consume():
        for _, (p, _, _) in loader:
            got.append(p)
        done.set()

    threading.Thread(target=consume, daemon=True).start()
    time.sleep(0.3)  # consumer has drained segment 0 and is blocked on the next segment
    t0 = time.monotonic()
    loader.close()
    assert done.wait(2), "consumer never returned after close()"
    assert time.monotonic() - t0 < 1.5
    assert got == [0, 1, 2, 3]
    assert not [t for t in threading.enumerate() if t.name == "distrainer-lanes" and t.is_alive()]


def test_loader_iterates_once(store):
    fs, root, log = build(store, n_segments=1, W=4)
    log.end()
    loader = LaneLoader(fs, root, lanes_for(log, 0, 1))
    assert len(list(loader)) == 4
    with pytest.raises(RuntimeError):
        list(loader)


def test_error_on_first_next_of_lanes_generator(store):
    fs, root = store

    def broken_from_start():
        raise RuntimeError("no lanes")
        yield  # pragma: no cover

    with pytest.raises(RuntimeError, match="no lanes"):
        list(LaneLoader(fs, root, broken_from_start()))


def test_backpressure_bounds_reads_in_flight(store, monkeypatch):
    fs, root, log = build(store, n_segments=2, W=8)
    log.end()
    import distrainer.loader as loader_mod

    in_flight = {"now": 0, "max": 0}
    lock = threading.Lock()
    real_read = loader_mod.read_block

    def slow_read(*args):
        with lock:
            in_flight["now"] += 1
            in_flight["max"] = max(in_flight["max"], in_flight["now"])
        try:
            time.sleep(0.02)
            return real_read(*args)
        finally:
            with lock:
                in_flight["now"] -= 1

    monkeypatch.setattr(loader_mod, "read_block", slow_read)
    out = list(LaneLoader(fs, root, lanes_for(log, 0, 1), prefetch=2, threads=2))
    assert len(out) == 16
    assert in_flight["max"] <= 2  # never more reads running than worker threads
    assert LaneLoader(fs, root, iter([]), prefetch=2, threads=2)._queue.maxsize == 2


@pytest.mark.parametrize("n", [3, 4])
def test_union_of_all_ranks_lanes_is_the_segment(store, n):
    fs, root, log = build(store, n_segments=2, W=12)
    log.end()
    seen = []
    for rank in range(n):
        seen += [p for _, (p, _, _) in LaneLoader(fs, root, lanes_for(log, rank, n))]
    assert sorted(seen) == list(range(24))
