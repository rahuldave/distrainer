"""Drive distrainer.trainer.train_loop rank by rank with a fake ray.train (no cluster)."""

import threading

import pyarrow as pa
import pytest
import torch
from conftest import make_table

from distrainer.audit import next_attempt, read_audit
from distrainer.block import write_block
from distrainer.config import DistrainerConfig
from distrainer.log import BlockLog
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


def make_store(tmp_path, W=8, segments=2, run_name="fake", **cfg_overrides):
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
