import json
import threading
import time

import pytest
from conftest import make_refs, make_table

from distrainer.block import BlockRef, write_block
from distrainer.log import BlockLog, LogMeta, Segment, segment_filename
from distrainer.storage import exists, join, list_names, read_bytes, write_bytes


def new_log(store, W=4, seed=7, n_blocks=None):
    fs, root = store
    log = BlockLog.create(fs, root, W=W, seed=seed)
    refs = [write_block(fs, root, f"b{i:03d}", make_table(2, i)) for i in range(n_blocks or W)]
    return log, refs


def test_create_open_meta(store):
    fs, root = store
    log, _ = new_log(store, W=4, seed=7)
    assert log.meta() == LogMeta(W=4, seed=7)
    assert log.W == 4
    reopened = BlockLog.open(fs, root)
    assert reopened.meta() == log.meta()
    with pytest.raises(FileExistsError):
        BlockLog.create(fs, root, W=4, seed=7)
    with pytest.raises(FileNotFoundError):
        BlockLog.open(fs, join(root, "elsewhere"))
    assert not BlockLog(fs, join(root, "elsewhere")).exists()
    with pytest.raises(ValueError):
        BlockLog.create(fs, join(root, "bad"), W=0, seed=1)


def test_append_shuffles_deterministically_and_commits_atomically(store):
    fs, root = store
    log, refs = new_log(store, W=4)
    seg0 = log.append(refs, pass_idx=0, meta={"note": "first"})
    assert seg0.seq == 0 and seg0.W == 4 and seg0.pass_idx == 0
    assert sorted(b.block_id for b in seg0.blocks) == [r.block_id for r in refs]
    assert seg0.meta["note"] == "first" and "created_at" in seg0.meta
    assert list(seg0.positions()) == [0, 1, 2, 3]
    # same seed and seq -> same order in an independent process/instance
    other = BlockLog.open(fs, root)
    assert other.read_segment(0).blocks == seg0.blocks
    # segment file is complete and no temp files are left behind
    names = list_names(fs, log.log_dir)
    assert sorted(names) == sorted(["_meta.json", segment_filename(0)])
    on_disk = Segment.from_json(read_bytes(fs, log.segment_path(0)).decode())
    assert on_disk == seg0
    seg1 = log.append(refs, pass_idx=1)
    assert seg1.seq == 1 and seg1.blocks != seg0.blocks  # different seq -> different order
    assert log.last_seq() == 1
    assert [s.seq for s in log.segments()] == [0, 1]


def test_append_validation(store):
    fs, root = store
    log, refs = new_log(store, W=4)
    with pytest.raises(ValueError):
        log.append(refs[:3])
    with pytest.raises(FileNotFoundError):
        log.append(make_refs(4))  # block files never written
    assert log.append(make_refs(4), verify_blocks=False).seq == 0
    log.end()
    log.end()  # idempotent
    assert log.ended()
    with pytest.raises(RuntimeError):
        log.append(refs)


def test_writer_resumes_sequence_from_listing(store):
    fs, root = store
    log, refs = new_log(store, W=4)
    log.append(refs)
    # a temp file left by a crashed writer must not confuse discovery
    write_bytes(fs, join(log.log_dir, "00000001.json.tmp-dead"), b"{}")
    writer2 = BlockLog.open(fs, root)
    assert writer2.last_seq() == 0
    assert writer2.append(refs).seq == 1
    assert writer2.has_segment(1) and not writer2.has_segment(2)
    with pytest.raises(FileNotFoundError):
        writer2.read_segment(2)


def test_read_segment_rejects_inconsistent_file(store):
    fs, root = store
    log, refs = new_log(store, W=4)
    seg = log.append(refs)
    d = json.loads(seg.to_json())
    d["seq"] = 5
    write_bytes(fs, log.segment_path(1), json.dumps(d).encode())
    with pytest.raises(ValueError):
        BlockLog.open(fs, root).read_segment(1)


def test_wait_segment_blocks_until_commit_then_none_after_end(store):
    fs, root = store
    log, refs = new_log(store, W=4)
    reader = BlockLog.open(fs, root)
    t0 = time.monotonic()

    def producer():
        time.sleep(0.3)
        log.append(refs)
        time.sleep(0.2)
        log.end()

    threading.Thread(target=producer, daemon=True).start()
    seg = reader.wait_segment(0, poll_s=0.05, timeout_s=5)
    assert seg is not None and seg.seq == 0 and time.monotonic() - t0 >= 0.25
    assert reader.wait_segment(1, poll_s=0.05, timeout_s=5) is None
    assert reader.ended()


def test_wait_segment_timeout(store):
    log, _ = new_log(store, W=4)
    with pytest.raises(TimeoutError):
        log.wait_segment(0, poll_s=0.01, timeout_s=0.1)


def test_next_pass_differs(store):
    log, refs = new_log(store, W=4)
    s0 = log.append(refs, pass_idx=0)
    assert log.next_pass_differs(s0) is False  # unknown: nothing committed yet, not ended
    s1 = log.append(refs, pass_idx=0)
    assert log.next_pass_differs(s0) is False
    s2 = log.append(refs, pass_idx=1)
    assert log.next_pass_differs(s1) is True
    assert log.next_pass_differs(s2) is False
    log.end()
    assert log.next_pass_differs(s2) is True


def test_gc_keeps_blocks_still_referenced_by_kept_segments(store):
    fs, root = store
    log, refs = new_log(store, W=4, n_blocks=8)
    log.append(refs[:4], pass_idx=0)  # seg 0: b000..b003
    log.append(refs[4:], pass_idx=0)  # seg 1: b004..b007
    log.append(refs[:4], pass_idx=1)  # seg 2: b000..b003 again (next pass)
    assert log.gc(0) == []
    assert log.gc(2) == [0, 1]
    assert not log.has_segment(0) and not log.has_segment(1) and log.has_segment(2)
    for r in refs[:4]:
        assert exists(fs, join(root, r.locator)), "blocks of a kept segment survive"
    for r in refs[4:]:
        assert not exists(fs, join(root, r.locator)), "blocks only seg 1 referenced are gone"
    assert log.gc(99) == [2]  # keep_from beyond the end is clamped
    assert BlockLog.open(fs, root).last_seq() is None


# ---- review follow-ups (phase 1-2 adversarial review) ----


def test_golden_permutation_pins_the_shuffle_formula(store):
    """Random(hash((seed, seq))) with seed=7: a tripwire against accidental formula changes."""
    fs, root = store
    log = BlockLog.create(fs, root, W=8, seed=7)
    refs = [write_block(fs, root, f"g{i}", make_table(1, i)) for i in range(8)]
    s0, s1 = log.append(refs), log.append(refs)
    assert [int(b.block_id[1:]) for b in s0.blocks] == [4, 1, 5, 7, 6, 2, 0, 3]
    assert [int(b.block_id[1:]) for b in s1.blocks] == [7, 6, 4, 0, 5, 1, 3, 2]


def test_append_refuses_to_overwrite_a_committed_segment(store):
    fs, root = store
    log, refs = new_log(store, W=4)
    log.append(refs)  # seq 0; this instance now believes the next seq is 1
    other = BlockLog.open(fs, root)
    other.append(refs)  # discovers seq 0 from the listing and commits seq 1
    with pytest.raises(FileExistsError):
        log.append(refs)  # stale instance must not overwrite seq 1
    assert BlockLog.open(fs, root).read_segment(1) == other.read_segment(1)


def test_segments_and_first_seq_after_gc(store):
    log, refs = new_log(store, W=4)
    for _ in range(3):
        log.append(refs)
    assert log.gc(2, delete_blocks=False) == [0, 1]
    assert log.first_seq() == 2 and log.committed_seqs() == [2]
    assert [s.seq for s in log.segments()] == [2]
    assert [s.seq for s in log.segments(start=0)] == []
    assert log.gc(2) == []  # idempotent
    for r in refs:
        assert exists(log.fs, join(log.root, r.locator))  # delete_blocks=False kept them


def test_verify_blocks_rejects_truncated_block_file(store):
    fs, root = store
    log, refs = new_log(store, W=4)
    path = join(root, refs[1].locator)
    data = read_bytes(fs, path)
    write_bytes(fs, path, data[: len(data) // 2])
    with pytest.raises(FileNotFoundError, match="corrupt"):
        log.append(refs)
    # a wrong row count in the ref is caught too
    bad = [BlockRef(r.block_id, r.locator, r.num_rows + 1) for r in refs]
    write_bytes(fs, path, data)
    with pytest.raises(FileNotFoundError):
        log.append(bad)


def test_end_race_between_the_two_probes_still_returns_the_segment(store, monkeypatch):
    log, refs = new_log(store, W=4)
    reader = BlockLog.open(log.fs, log.root)
    calls = {"n": 0}
    real_has = reader.has_segment

    def has_segment(seq):
        calls["n"] += 1
        if calls["n"] == 1:
            # writer commits the segment and _END right after the first negative probe
            log.append(refs)
            log.end()
            return False
        return real_has(seq)

    monkeypatch.setattr(reader, "has_segment", has_segment)
    seg = reader.wait_segment(0, poll_s=0.01, timeout_s=1)
    assert seg is not None and seg.seq == 0


def test_read_segment_rejects_unknown_schema_version(store):
    log, refs = new_log(store, W=4)
    seg = log.append(refs)
    d = json.loads(seg.to_json())
    d["schema_version"] = 99
    write_bytes(log.fs, log.segment_path(1), json.dumps(d).encode())
    with pytest.raises(ValueError, match="schema_version"):
        BlockLog.open(log.fs, log.root).read_segment(1)


def test_segment_cache_is_bounded(store):
    log, refs = new_log(store, W=4)
    for _ in range(BlockLog.SEGMENT_CACHE_SIZE + 3):
        log.append(refs)
    assert len(log._segments) == BlockLog.SEGMENT_CACHE_SIZE
    assert log.read_segment(0).seq == 0  # evicted entries are re-read from disk
