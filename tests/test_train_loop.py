"""Drive distrainer.trainer.train_loop rank by rank with a fake ray.train (no cluster)."""

import threading

import pyarrow as pa
import pytest
import torch
from conftest import AppendNextSegments, make_table

from distrainer.audit import next_attempt, read_audit
from distrainer.block import write_block
from distrainer.config import DistrainerConfig
from distrainer.log import BlockLog
from distrainer.storage import exists, join
from distrainer.trainer import CheckpointIO, DistTrainer, TrainInfo, train_loop
from integration_tests.cluster.check_audit import check_dealing, check_s1


def build_model(info: TrainInfo):
    torch.manual_seed(0)
    m = torch.nn.Linear(1, 1)
    return m, torch.optim.SGD(m.parameters(), lr=0.1)


def train_step(model, optimizer, table: pa.Table, info: TrainInfo):
    x = torch.tensor(table.column("x").to_numpy(), dtype=torch.float32).unsqueeze(1)
    optimizer.zero_grad()
    loss = (model(x) ** 2).mean()
    loss.backward()
    optimizer.step()
    return {"loss": float(loss.item()), "rows": table.num_rows}


def failing_step(model, optimizer, table, info):
    if info.position == 3:
        raise RuntimeError("boom at position 3")
    return train_step(model, optimizer, table, info)


class FakeContext:
    def __init__(self, rank, n):
        self.rank, self.n = rank, n

    def get_world_rank(self):
        return self.rank

    def get_world_size(self):
        return self.n


@pytest.fixture
def fake_ray(monkeypatch):
    """Patches the ray.train entry points train_loop uses; returns the recorded reports."""
    import ray.train
    import ray.train.collective
    import ray.train.torch

    state = {"reports": [], "checkpoint": None, "broadcast": None, "barriers": 0}

    def report(metrics, checkpoint=None, checkpoint_dir_name=None, **kw):
        state["reports"].append((state["rank"], dict(metrics), checkpoint_dir_name, checkpoint))

    def broadcast(value):
        if value is not None:
            state["broadcast"] = value
        return state["broadcast"]

    monkeypatch.setattr(ray.train, "get_context", lambda: FakeContext(state["rank"], state["n"]))
    monkeypatch.setattr(ray.train, "get_checkpoint", lambda: state["checkpoint"])
    monkeypatch.setattr(ray.train, "report", report)
    monkeypatch.setattr(ray.train.torch, "prepare_model", lambda m: m)
    monkeypatch.setattr(ray.train.torch, "get_device", lambda: torch.device("cpu"))
    monkeypatch.setattr(
        ray.train.collective,
        "barrier",
        lambda: state.__setitem__("barriers", state["barriers"] + 1),
    )
    monkeypatch.setattr(ray.train.collective, "broadcast_from_rank_zero", broadcast)
    return state


def make_store(tmp_path, W=8, segments=2, run_name="fake", end=True, **cfg_overrides):
    """A batch store of ``segments`` segments; ``end=False`` leaves the log open (streaming)."""
    cfg = DistrainerConfig.from_dict(
        {
            "run_name": run_name,
            "storage_path": str(tmp_path / "runs"),
            "store_root": str(tmp_path / "blocks"),
            "log": {"W": W, "passes": 1, "wait_poll_s": 0.01},
            "checkpoint": {"policy": "any", "every_k": 2, "num_to_keep": None},
            "scaling": {"num_workers": [1, 2]},
            **cfg_overrides,
        }
    )
    fs, root = cfg.store_fs()
    log = BlockLog.create(fs, root, W=W, seed=1)
    for s in range(segments):
        log.append([write_block(fs, root, f"s{s}b{i}", make_table(2, i)) for i in range(W)])
    if end:
        log.end()
    return cfg


def run_world(fake_ray, cfg, n, step=train_step, checkpoint=None, loop_extra=None):
    fake_ray["checkpoint"] = checkpoint
    fake_ray["broadcast"] = None
    loop_config = {**DistTrainer(step, build_model, cfg).loop_config(), **(loop_extra or {})}
    fake_ray["n"] = n
    for rank in range(n):
        fake_ray["rank"] = rank
        train_loop(loop_config)


def test_two_ranks_report_identically_and_audit_satisfies_s1(tmp_path, fake_ray):
    cfg = make_store(tmp_path, W=8, segments=2)
    run_world(fake_ray, cfg, n=2)
    fs, root = cfg.store_fs()
    records = read_audit(fs, root, "fake")
    assert check_s1(records, 8) == []
    per_rank = {}
    for rank, metrics, name, ckpt in fake_ray["reports"]:
        per_rank.setdefault(rank, []).append((name, metrics["cursor"], ckpt is not None))
    # 4 steps per segment, every_k=2 + segment end -> cursors 2 and 4 in each of 2 segments
    assert [c for _, c, _ in per_rank[0]] == [2, 4, 2, 4]
    assert [n for n, _, _ in per_rank[0]] == [n for n, _, _ in per_rank[1]]
    assert per_rank[0][0][0] == "checkpoint_g000000_p000004_n02_a00"
    assert all(has for _, _, has in per_rank[0]) and not any(has for _, _, has in per_rank[1])
    assert (
        fake_ray["reports"][0][1]["steps_in_report"] == 2
        and "loss_mean" in fake_ray["reports"][0][1]
    )
    assert [m["reports"] for r, m, _, _ in fake_ray["reports"] if r == 0] == [1, 2, 3, 4]
    assert fake_ray["barriers"] == 4  # segment ends x ranks
    assert next_attempt(fs, root, "fake") == 1


def test_resume_at_a_different_world_size_continues_from_the_ledger(tmp_path, fake_ray):
    cfg = make_store(tmp_path, W=8, segments=2)
    run_world(fake_ray, cfg, n=2)
    ckpt = fake_ray["reports"][0][3]  # rank 0's first checkpoint: segment 0, cursor 2, n=2
    assert CheckpointIO.read_ledger(ckpt).done_positions() == 4
    fake_ray["reports"].clear()
    run_world(fake_ray, cfg, n=4, checkpoint=ckpt)
    fs, root = cfg.store_fs()
    records = [r for r in read_audit(fs, root, "fake") if r.attempt == 1]
    assert records, "second attempt has its own audit files"
    assert sorted(r.position for r in records) == list(range(4, 16))
    assert check_dealing(records, 8) == []
    firsts = {}
    for r in records:
        firsts[r.rank] = min(firsts.get(r.rank, 10**9), r.position)
    assert firsts == {0: 4, 1: 5, 2: 6, 3: 7}  # step 1 of segment 0 under the new world size
    assert {r.step for r in records if r.segment == 0} == {1}
    names = {name for _, _, name, _ in fake_ray["reports"]}
    assert "checkpoint_g000000_p000008_n04_a01" in names
    assert all(name.endswith("_n04_a01") for name in names)


def test_initial_checkpoint_from_loop_config_when_ray_has_none(tmp_path, fake_ray):
    cfg = make_store(tmp_path, W=8, segments=2)
    run_world(fake_ray, cfg, n=2)
    last = [c for _, _, _, c in fake_ray["reports"] if c is not None][-1]  # end of segment 1
    cfg2 = make_store(tmp_path / "second", W=8, segments=3, run_name="resumed")
    fake_ray["reports"].clear()
    run_world(fake_ray, cfg2, n=2, loop_extra={"initial_checkpoint": last})
    fs, root = cfg2.store_fs()
    records = read_audit(fs, root, "resumed")
    assert sorted(r.position for r in records) == list(range(16, 24))  # segment 2 only


def test_train_step_error_closes_loader_and_audit(tmp_path, fake_ray):
    cfg = make_store(tmp_path, W=8, segments=1)
    with pytest.raises(RuntimeError, match="boom"):
        run_world(fake_ray, cfg, n=1, step=failing_step)
    fs, root = cfg.store_fs()
    assert [r.position for r in read_audit(fs, root, "fake")] == [0, 1, 2]
    assert not [t for t in threading.enumerate() if t.name == "distrainer-lanes" and t.is_alive()]


def test_report_every_step_option(tmp_path, fake_ray):
    cfg = make_store(
        tmp_path, W=8, segments=1, checkpoint={"policy": "segment_end", "report_every_step": True}
    )
    run_world(fake_ray, cfg, n=1)
    assert len(fake_ray["reports"]) == 8
    assert sum(c is not None for _, _, _, c in fake_ray["reports"]) == 1


def test_dist_trainer_config_mapping_and_picklable_loop_config(tmp_path, monkeypatch):
    cfg = make_store(tmp_path, W=8, segments=1)
    dt = DistTrainer(train_step, build_model, cfg)
    sc = dt.scaling_config()
    assert sc.num_workers == (1, 2) and sc.resources_per_worker == {"CPU": 1}
    rc = dt.run_config()
    assert (
        rc.name == "fake" and rc.storage_filesystem is None and rc.storage_path == cfg.storage_path
    )
    assert rc.checkpoint_config.num_to_keep is None and rc.failure_config.max_failures == 3
    import ray.cloudpickle as cp

    assert cp.loads(cp.dumps(dt.loop_config()))["config"]["run_name"] == "fake"

    class FakeS3(__import__("pyarrow.fs").fs.LocalFileSystem):
        def __init__(self, **kw):
            super().__init__()

    monkeypatch.setattr("distrainer.storage.pafs.S3FileSystem", FakeS3)
    cfg_s3 = DistrainerConfig.from_dict(
        {
            "storage_path": "b/runs",
            "store_root": "b/blocks",
            "storage": {"kind": "s3", "anonymous": True},
        }
    )
    rc3 = DistTrainer(train_step, build_model, cfg_s3).run_config()
    assert rc3.storage_filesystem is not None and rc3.storage_path == "b/runs"


# ---- M4: segment hooks and gc ----


def test_segment_hook_runs_on_rank_zero_with_a_ledger_snapshot_and_its_own_log(tmp_path, fake_ray):
    cfg = make_store(tmp_path, W=8, segments=1, end=False)
    run_world(fake_ray, cfg, n=2, loop_extra={"hooks": [AppendNextSegments(segments=3)]})
    fs, root = cfg.store_fs()
    records = read_audit(fs, root, "fake")
    assert check_s1(records, 8) == []
    assert sorted({r.segment for r in records}) == [0, 1, 2]
    assert {r.block_id[:2] for r in records if r.segment == 1} == {"s1"}
    calls = AppendNextSegments.calls
    assert [c["ctx"].rank for c in calls] == [0, 0, 0]  # rank 1 never runs hooks
    assert [c["ledger"].segment for c in calls] == [0, 1, 2]  # snapshots, not the live ledger
    assert all(c["ledger"].cursor == 4 and c["ledger"].world_size == 2 for c in calls)
    assert all(isinstance(c["model"], torch.nn.Linear) for c in calls)  # unwrapped
    assert len({id(c["log"]) for c in calls}) == 1 and calls[0]["log"].root == root
    assert [c["ctx"].segment for c in calls] == [0, 1, 2]
    log = BlockLog.open(fs, root)
    assert log.ended()
    assert fake_ray["barriers"] == 6  # 3 segment ends x 2 ranks
    # the loader's producer thread was waiting for segment seq+1 while the hook committed it:
    # every hook-made segment was committed before its first block was consumed
    for seq in (1, 2):
        first_ts = min(r.ts for r in records if r.segment == seq)
        assert log.read_segment(seq).meta["created_at"] <= first_ts


def test_hooks_from_config_are_built_by_entry_on_rank_zero(tmp_path, fake_ray):
    cfg = make_store(
        tmp_path,
        W=8,
        segments=1,
        end=False,
        hooks={"append": {"entry": "conftest:AppendNextSegments", "segments": 2}},
    )
    run_world(fake_ray, cfg, n=2)
    fs, root = cfg.store_fs()
    records = read_audit(fs, root, "fake")
    assert check_s1(records, 8) == [] and max(r.segment for r in records) == 1
    calls = AppendNextSegments.calls
    assert [c["ctx"].rank for c in calls] == [0, 0]
    assert calls[0]["config"].run_name == "fake"  # the factory received the worker's config


def test_gc_at_segment_end_keeps_retention_behind_the_last_checkpoint(tmp_path, fake_ray):
    log_cfg = {"W": 8, "passes": 1, "wait_poll_s": 0.01, "gc": True, "retention_segments": 1}
    cfg = make_store(tmp_path, W=8, segments=4, log=log_cfg)
    run_world(fake_ray, cfg, n=1)
    fs, root = cfg.store_fs()
    log = BlockLog.open(fs, root)
    # the any policy checkpoints at every segment end: at the end of segment s the last
    # checkpoint is in s, so segments < s - 1 are dropped; 0 and 1 are gone after segment 3
    assert log.committed_seqs() == [2, 3]
    assert not exists(fs, join(root, "blocks/s0b0.parquet"))
    assert not exists(fs, join(root, "blocks/s1b7.parquet"))
    assert exists(fs, join(root, "blocks/s2b0.parquet"))
    assert exists(fs, join(root, "blocks/s3b7.parquet"))
    assert sorted(r.position for r in read_audit(fs, root, "fake")) == list(range(32))

    # gc_blocks: false drops the segment files only
    cfg2 = make_store(tmp_path / "keep", W=8, segments=4, log={**log_cfg, "gc_blocks": False})
    run_world(fake_ray, cfg2, n=1)
    fs2, root2 = cfg2.store_fs()
    assert BlockLog.open(fs2, root2).committed_seqs() == [2, 3]
    assert exists(fs2, join(root2, "blocks/s0b0.parquet"))

    # without a checkpoint nothing is ever behind the retention window
    cfg3 = make_store(
        tmp_path / "never", W=8, segments=4, log=log_cfg, checkpoint={"policy": "never"}
    )
    run_world(fake_ray, cfg3, n=1)
    fs3, root3 = cfg3.store_fs()
    assert BlockLog.open(fs3, root3).committed_seqs() == [0, 1, 2, 3]


def test_resume_into_a_gc_window_fails_loudly_and_a_kept_segment_resumes(tmp_path, fake_ray):
    log_cfg = {"W": 8, "passes": 1, "wait_poll_s": 0.01, "gc": True, "retention_segments": 1}
    cfg = make_store(tmp_path, W=8, segments=4, log=log_cfg)
    run_world(fake_ray, cfg, n=1)
    fs, root = cfg.store_fs()
    assert BlockLog.open(fs, root).committed_seqs() == [2, 3]
    ckpts = {name: c for _, _, name, c in fake_ray["reports"] if c is not None}
    old = ckpts["checkpoint_g000001_p000004_n01_a00"]  # segment 1: gone
    with pytest.raises(RuntimeError, match="garbage-collected"):
        run_world(fake_ray, cfg, n=1, checkpoint=old)
    kept = ckpts["checkpoint_g000002_p000004_n01_a00"]  # segment 2, half done: still there
    fake_ray["reports"].clear()
    run_world(fake_ray, cfg, n=1, checkpoint=kept)
    records = [r for r in read_audit(fs, root, "fake") if r.attempt == 2]
    assert sorted(r.position for r in records) == list(range(20, 32))
    # a hole below the last committed segment is an error too, not a silent end
    cfg2 = make_store(tmp_path / "hole", W=8, segments=3)
    fs2, root2 = cfg2.store_fs()
    from distrainer.storage import delete

    delete(fs2, BlockLog.open(fs2, root2).segment_path(1))  # 0 and 2 remain, then _END
    with pytest.raises(RuntimeError, match="segment 1 is missing"):
        run_world(fake_ray, cfg2, n=1)


def test_dist_trainer_rejects_an_unimportable_hook_entry(tmp_path):
    cfg = make_store(tmp_path, W=8, segments=1, hooks={"h": "no.such.module:Hook"})
    with pytest.raises(ValueError, match="cannot import"):
        DistTrainer(train_step, build_model, cfg)


class DieBeforeAppending:
    """Simulates rank 0 dying between the segment-end checkpoint report and the hook."""

    def on_segment_end(self, model, ledger, log, ctx):
        raise RuntimeError("node died before the hook appended the next segment")


def test_restart_at_a_segment_boundary_replays_the_pending_hook_call(tmp_path, fake_ray):
    cfg = make_store(tmp_path, W=8, segments=1, end=False)
    with pytest.raises(RuntimeError, match="node died"):
        run_world(fake_ray, cfg, n=1, loop_extra={"hooks": [DieBeforeAppending()]})
    ckpts = {name: c for _, _, name, c in fake_ray["reports"] if c is not None}
    boundary = ckpts["checkpoint_g000000_p000008_n01_a00"]  # segment 0 complete, no segment 1
    fs, root = cfg.store_fs()
    assert BlockLog.open(fs, root).committed_seqs() == [0]
    hook = AppendNextSegments(segments=3)
    run_world(fake_ray, cfg, n=1, checkpoint=boundary, loop_extra={"hooks": [hook]})
    calls = AppendNextSegments.calls
    assert [c["ledger"].segment for c in calls] == [0, 1, 2]  # the replayed end of segment 0 first
    assert calls[0]["ctx"].attempt == 1 and calls[0]["ledger"].cursor == 8
    records = [r for r in read_audit(fs, root, "fake") if r.attempt == 1]
    assert sorted(r.position for r in records) == list(range(8, 24))
    assert BlockLog.open(fs, root).ended()
    # nothing to replay when the previous attempt did append (or the log is a finished batch)
    fake_ray["reports"].clear()
    run_world(fake_ray, cfg, n=1, checkpoint=boundary, loop_extra={"hooks": [hook]})
    assert [c["ledger"].segment for c in calls[3:]] == [1, 2]
