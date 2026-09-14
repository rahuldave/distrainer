"""The streaming producer example (S11): blocks streamed into a log with sleeps and _END."""

from pathlib import Path

import pytest
from test_train_loop import fake_ray  # noqa: F401 (fixture)

from distrainer.log import BlockLog
from distrainer.storage import exists, join
from examples.streaming_producer.produce import main, produce


def test_produce_streams_exactly_the_requested_segments_for_any_buffer(store):
    fs, root = store
    segs = produce(fs, root, W=4, seed=1, segments=3, rows=2, features=3)
    assert [s.seq for s in segs] == [0, 1, 2]
    log = BlockLog.open(fs, root)
    assert log.committed_seqs() == [0, 1, 2] and log.ended()
    assert log.meta().created_by == "streaming_producer"
    # a plain window: segment k holds arrivals kW..kW+W-1
    assert sorted(b.block_id for b in segs[0].blocks) == [f"p{i:06d}" for i in range(4)]
    assert all(exists(fs, join(root, b.locator)) for s in segs for b in s.blocks)
    assert segs[1].meta["writer"] == "stream" and "created_at" in segs[1].meta
    with pytest.raises(FileExistsError):
        produce(fs, root, W=4, seed=1, segments=1)
    with pytest.raises(ValueError):
        produce(fs, join(root, "x"), W=4, seed=1, segments=0)


def test_produce_with_a_shuffle_buffer_flushes_the_tail_at_close(store):
    fs, root = store
    messages = []
    segs = produce(fs, root, W=4, seed=1, segments=3, shuffle_buffer=2, progress=messages.append)
    assert [s.seq for s in segs] == [0, 1, 2]  # the last one comes out of close()
    ids = sorted(b.block_id for s in segs for b in s.blocks)
    assert ids == [f"p{i:06d}" for i in range(12)]  # every block exactly once
    assert messages[-1].startswith("log ended after 3")
    assert [int(m.split()[1]) for m in messages if m.startswith("segment ")] == [0, 1, 2]
    assert BlockLog.open(fs, root).ended()


def test_main_reads_defaults_from_the_trainer_config(tmp_path):
    cfg = tmp_path / "c.yaml"
    cfg.write_text(
        f"run_name: s\nstorage_path: {tmp_path / 'runs'}\nstore_root: {tmp_path / 'blocks'}\n"
        "seed: 5\nlog: {W: 4, shuffle_buffer_segments: 1}\n"
        "train: {rows_per_block: 3, features: 2}\n"
    )
    assert main(["--config", str(cfg), "--segments", "2", "--sleep-s", "0"]) == 0
    from distrainer.storage import resolve

    fs, root = resolve(str(tmp_path / "blocks"), create=False)
    log = BlockLog.open(fs, root)
    assert log.W == 4 and log.meta().seed == 5 and log.committed_seqs() == [0, 1]
    from distrainer.block import read_block

    table = read_block(fs, root, log.read_segment(0).blocks[0])
    assert table.num_rows == 3 and table.column_names == ["x_0", "x_1", "y", "block_id"]
    # --store and --W override the config; --store alone needs the rest from defaults
    assert (
        main(["--store", str(tmp_path / "other"), "--W", "2", "--segments", "1", "--sleep-s", "0"])
        == 0
    )
    assert BlockLog.open(*resolve(str(tmp_path / "other"), create=False)).W == 2
    with pytest.raises(SystemExit):
        main(["--segments", "1"])


def test_trainer_waits_for_a_live_producer_and_gc_keeps_the_window(tmp_path, fake_ray):  # noqa: F811
    """End to end on a fake Ray Train: a producer thread streams 5 segments (shuffle buffer 2,
    0.4 s between commits) while one rank trains with gc on; the rank waits, consumes every
    position once, and the log ends up two segments long."""
    import threading

    from distrainer.audit import read_audit
    from distrainer.config import DistrainerConfig
    from distrainer.storage import list_names
    from distrainer.trainer import DistTrainer, train_loop
    from examples.hello_blocks.train import build_model, train_step
    from integration_tests.cluster.check_audit import (
        check_retention,
        check_s11,
        check_streaming_run,
        segment_starts,
    )

    cfg = DistrainerConfig.from_dict(
        {
            "run_name": "stream",
            "storage_path": str(tmp_path / "runs"),
            "store_root": str(tmp_path / "blocks"),
            "seed": 7,
            "log": {"W": 4, "wait_poll_s": 0.01, "gc": True, "retention_segments": 1},
            "checkpoint": {"policy": "any", "every_k": 2, "num_to_keep": None},
            "scaling": {"num_workers": 1},
            "train": {"features": 8},
        }
    )
    fs, root = cfg.store_fs()
    produced: list = []
    messages: list[str] = []
    import time

    def trainer_is_waiting() -> None:
        # the trainer opens its audit file before it waits for segment 0: only push blocks once
        # it is there, so the rank really waits at every boundary whatever the machine's load
        deadline = time.monotonic() + 30
        while not (Path(root) / "audit" / "stream" / "0-0.jsonl").exists():
            assert time.monotonic() < deadline, "trainer did not start"
            time.sleep(0.01)

    thread = threading.Thread(
        target=lambda: produced.extend(
            produce(
                fs,
                root,
                W=4,
                seed=7,
                segments=5,
                sleep_s=0.4,
                shuffle_buffer=2,
                rows=4,
                progress=messages.append,
                on_created=trainer_is_waiting,
            )
        ),
        daemon=True,
    )
    thread.start()
    deadline = time.monotonic() + 10
    while not BlockLog(fs, root).exists():  # as the scenario runner does before train.py
        assert time.monotonic() < deadline, "producer did not create the log"
        time.sleep(0.01)
    fake_ray["checkpoint"], fake_ray["broadcast"], fake_ray["n"], fake_ray["rank"] = (
        None,
        None,
        1,
        0,
    )
    train_loop(DistTrainer(train_step, build_model, cfg).loop_config())
    thread.join(timeout=10)
    assert [s.seq for s in produced] == [0, 1, 2, 3, 4]
    records = read_audit(fs, root, "stream")
    committed_at = {  # as the scenario runner does: from the producer's printed lines
        int(m.split()[1]): float(m.rsplit(" ", 1)[1])
        for m in messages
        if m.startswith("segment ") and " committed at " in m
    }
    assert sorted(committed_at) == [0, 1, 2, 3, 4]
    assert check_s11(records, 4, committed_at, min_gap_s=0.25, expected_segments=5) == []
    starts = segment_starts(records)
    assert max(starts[k + 1] - starts[k] for k in range(4)) >= 0.25  # the rank waited
    # the same checks packaged for the scenario runner (local here, inside the head on S3);
    # the fake Ray Train keeps no run directory, so build one from the reported checkpoints
    import json

    from distrainer.trainer import CheckpointIO

    run_dir = tmp_path / "runs" / "stream"
    for _, _, name, ckpt in fake_ray["reports"]:
        if ckpt is not None:
            (run_dir / name).mkdir(parents=True, exist_ok=True)
            ledger = CheckpointIO.read_ledger(ckpt).asdict()
            (run_dir / name / ".metadata.json").write_text(json.dumps({"ledger": ledger}))
    problems, info = check_streaming_run(
        fs, root, "stream", committed_at, 0.25, 5, str(tmp_path / "runs" / "stream"), 1
    )
    assert problems == [] and info["kept"] == [3, 4] and info["last_ckpt"] == 4
    assert len(info["gaps"]) == 4 and "5 blocks" not in info["summary"]
    log = BlockLog.open(fs, root)
    kept = list(log.segments())
    assert [s.seq for s in kept] == [3, 4]  # last checkpoint in segment 4, retention 1
    block_files = {f"blocks/{n}" for n in list_names(fs, join(root, "blocks"))}
    assert (
        check_retention(
            [s.seq for s in kept], block_files, {b.locator for s in kept for b in s.blocks}, 4, 1
        )
        == []
    )
    assert len(block_files) == 8
    bad, _ = check_streaming_run(fs, root, "stream", committed_at, 0.25, 5, str(tmp_path / "x"), 1)
    assert any("no checkpoint ledgers" in p for p in bad)
