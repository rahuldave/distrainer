import pytest
from conftest import make_refs, make_table

from distrainer.block import write_block
from distrainer.log import BlockLog
from distrainer.writer import BatchWriter, StreamingWriter, cut_segments, pass_order


def corpus(store, n, W=4, seed=3):
    fs, root = store
    log = BlockLog.create(fs, root, W=W, seed=seed)
    refs = [write_block(fs, root, f"b{i:03d}", make_table(1, i)) for i in range(n)]
    return log, refs


def test_pass_order_is_deterministic_and_differs_between_passes():
    refs = make_refs(8)
    p0, p0_again, p1 = pass_order(refs, 5, 0), pass_order(refs, 5, 0), pass_order(refs, 5, 1)
    assert p0 == p0_again and sorted(p0, key=lambda r: r.block_id) == refs
    assert p0 != p1


def test_cut_segments_tail_modes():
    refs = make_refs(10)
    with pytest.raises(ValueError):
        cut_segments(refs, 4)
    assert [len(c) for c in cut_segments(refs, 4, "drop")] == [4, 4]
    wrapped = cut_segments(refs, 4, "wrap")
    assert [len(c) for c in wrapped] == [4, 4, 4]
    assert wrapped[2] == refs[8:] + refs[:2]
    assert cut_segments([], 4) == []
    assert [len(c) for c in cut_segments(make_refs(8), 4)] == [4, 4]
    small = cut_segments(make_refs(3), 8, "wrap")
    assert len(small) == 1 and len(small[0]) == 8
    assert [b.block_id for b in small[0]][:3] == ["b0000", "b0001", "b0002"]


def test_batch_writer_two_passes_regroup_segments(store):
    log, refs = corpus(store, 8, W=4)
    writer = BatchWriter(log, refs, passes=2)
    plan = writer.plan()
    assert [p for p, _ in plan] == [0, 0, 1, 1]
    segments = writer.run()
    assert [s.seq for s in segments] == [0, 1, 2, 3]
    assert [s.pass_idx for s in segments] == [0, 0, 1, 1]
    assert log.ended()
    ids = lambda s: sorted(b.block_id for b in s.blocks)  # noqa: E731
    # each pass covers the corpus exactly once
    assert sorted(ids(segments[0]) + ids(segments[1])) == [r.block_id for r in refs]
    assert sorted(ids(segments[2]) + ids(segments[3])) == [r.block_id for r in refs]
    # membership changes between passes (global permutation before cutting)
    groups = lambda segs: {frozenset(ids(s)) for s in segs}  # noqa: E731
    assert groups(segments[:2]) != groups(segments[2:])
    assert log.next_pass_differs(segments[1]) and not log.next_pass_differs(segments[0])


def test_batch_writer_validation(store):
    log, refs = corpus(store, 8, W=4)
    with pytest.raises(ValueError):
        BatchWriter(log, refs, passes=0)
    with pytest.raises(ValueError):
        BatchWriter(log, refs + refs[:1])
    with pytest.raises(ValueError):
        BatchWriter(log, refs[:6]).run()  # 6 % 4 != 0 and tail=error
    assert len(BatchWriter(log, refs[:6], tail="drop").run()) == 1


def test_streaming_writer_window_and_flush(store):
    log, refs = corpus(store, 10, W=4)
    sw = StreamingWriter(log)
    committed = [seg for r in refs if (seg := sw.push(r)) is not None]
    assert [s.seq for s in committed] == [0, 1]
    assert sw.buffered == 2
    assert sorted(b.block_id for b in committed[0].blocks) == [r.block_id for r in refs[:4]]
    with pytest.raises(ValueError):
        sw.flush(tail="error")
    assert sw.close(tail="drop") == [] and log.ended() and sw.buffered == 0


def test_streaming_writer_shuffle_buffer_mixes_beyond_window(store):
    log, refs = corpus(store, 12, W=4)
    sw = StreamingWriter(log, shuffle_buffer_segments=2)
    committed = [seg for r in refs if (seg := sw.push(r)) is not None]
    # first commit happens once 8 blocks are buffered and samples 4 of them
    assert len(committed) == 2 and sw.buffered == 4
    first = {b.block_id for b in committed[0].blocks}
    assert first <= {r.block_id for r in refs[:8]}
    assert first != {r.block_id for r in refs[:4]}, "a plain window would take the first 4"
    rest = sw.close()
    assert len(rest) == 1
    all_ids = sorted(b.block_id for s in committed + rest for b in s.blocks)
    assert all_ids == [r.block_id for r in refs]


def test_streaming_writer_wrap_tail(store):
    log, refs = corpus(store, 6, W=4)
    sw = StreamingWriter(log)
    for r in refs:
        sw.push(r)
    out = sw.close(tail="wrap")
    assert len(out) == 1 and out[0].meta.get("wrapped") is True
    assert log.last_seq() == 1
    with pytest.raises(ValueError):
        StreamingWriter(log, shuffle_buffer_segments=0)


def test_streaming_sampling_is_reproducible_and_rejects_duplicates(store, tmp_path):
    from distrainer.storage import StorageConfig, build_filesystem

    runs = []
    for i in range(2):
        st = build_filesystem(StorageConfig(kind="local", path=str(tmp_path / f"s{i}")))
        log, refs = corpus(st, 8, W=4)
        sw = StreamingWriter(log, shuffle_buffer_segments=2)
        for r in refs:
            sw.push(r)
        runs.append([[b.block_id for b in s.blocks] for s in sw.close()])
    assert runs[0] == runs[1]
    log, refs = corpus(store, 4, W=4)
    sw = StreamingWriter(log)
    sw.push(refs[0])
    with pytest.raises(ValueError):
        sw.push(refs[0])
    with pytest.raises(ValueError):
        sw.flush(tail="sideways")


def test_batch_writer_corpus_smaller_than_w(store):
    log, refs = corpus(store, 3, W=4)
    with pytest.raises(ValueError):
        BatchWriter(log, refs, tail="drop").run()  # would produce an empty, ended log
    assert not log.ended()
    segs = BatchWriter(log, refs, tail="wrap").run()
    assert len(segs) == 1 and len(segs[0].blocks) == 4 and log.ended()
    with pytest.raises(ValueError):
        cut_segments(refs, 4, "sideways")


def test_batch_writer_on_ended_log_fails(store):
    log, refs = corpus(store, 4, W=4)
    BatchWriter(log, refs).run()
    with pytest.raises(RuntimeError):
        BatchWriter(log, refs).run()
    fresh, refs2 = corpus((log.fs, f"{log.root}/again"), 4, W=4)
    fresh.append(refs2)
    assert BatchWriter(fresh, refs2).run()[0].seq == 1  # continues after existing segments


def test_streaming_wrap_needs_enough_blocks(store):
    log, refs = corpus(store, 2, W=4)
    sw = StreamingWriter(log)
    sw.push(refs[0])
    with pytest.raises(ValueError):
        sw.flush(tail="wrap")
