"""train_loop on a real Gloo process group (tests/procgroup.py): the audit facts of the fake
fixture hold under DDP, and what the fake fixture cannot see (the all-reduce, the other rank
waiting at a segment end) is asserted here."""

import os

import pytest
import torch
from conftest import AppendNextSegments
from procgroup import run_world
from test_train_loop import build_model, make_store, train_step

from distrainer.audit import next_attempt, read_audit
from distrainer.ledger import Ledger
from distrainer.log import BlockLog
from distrainer.trainer import CheckpointIO, unwrap
from integration_tests.cluster.check_audit import check_dealing, check_s1

# the local_sgd and diloco stores below keep checkpoint.policy any with every_k on purpose (the
# mid-segment assertions), which the config warns about under the drifting kinds
pytestmark = pytest.mark.filterwarnings("ignore:parallel.kind:UserWarning")


def recording_step(model, optimizer, table, info):
    """``train_step`` with the weight saved before and after the step, per (rank, position)."""
    d = info.train["record_dir"]
    os.makedirs(d, exist_ok=True)
    w = unwrap(model).weight
    torch.save(w.detach().clone(), os.path.join(d, f"r{info.rank}_p{info.position:04d}_pre.pt"))
    out = train_step(model, optimizer, table, info)
    torch.save(w.detach().clone(), os.path.join(d, f"r{info.rank}_p{info.position:04d}_post.pt"))
    return out


def load_weights(d) -> dict[tuple[int, int, str], torch.Tensor]:
    """``{(rank, position, "pre" | "post"): weight}`` from ``recording_step``'s files."""
    out = {}
    for name in os.listdir(d):
        rank, pos, tag = name[:-3].split("_")  # r<rank>_p<position>_<pre|post>.pt
        out[(int(rank[1:]), int(pos[1:]), tag)] = torch.load(os.path.join(d, name))
    return out


def load_weight(report) -> torch.Tensor:
    """The weight in a checkpoint of the test's ``Linear(1, 1)``."""
    fresh = torch.nn.Linear(1, 1)
    CheckpointIO.load(report.checkpoint(), fresh)
    return fresh.weight.detach().clone()


def test_two_ranks_under_real_ddp_report_identically_and_audit_satisfies_s1(tmp_path):
    cfg = make_store(tmp_path, W=8, segments=2)
    world = run_world(cfg, 2, train_step, build_model, str(tmp_path / "out"))
    fs, root = cfg.store_fs()
    records = read_audit(fs, root, "fake")
    assert check_s1(records, 8) == []
    r0, r1 = world.by_rank(0), world.by_rank(1)
    # 4 steps per segment, every_k=2 + segment end -> cursors 2 and 4 in each of 2 segments
    assert [r.metrics["cursor"] for r in r0] == [2, 4, 2, 4]
    assert [r.checkpoint_dir_name for r in r0] == [r.checkpoint_dir_name for r in r1]
    assert r0[0].checkpoint_dir_name == "checkpoint_g000000_p000004_n02_a00"
    assert all(r.has_checkpoint for r in r0) and not any(r.has_checkpoint for r in r1)
    assert r0[0].metrics["steps_in_report"] == 2 and "loss_mean" in r0[0].metrics
    assert [r.metrics["reports"] for r in r0] == [1, 2, 3, 4]
    assert next_attempt(fs, root, "fake") == 1
    # the checkpoint holds the unwrapped module: it loads into a plain Linear
    fresh = torch.nn.Linear(1, 1)
    assert CheckpointIO.load(r0[-1].checkpoint(), fresh) == Ledger(1, 4, 2, 0, 0)


def test_ddp_averages_the_gradients_so_the_replicas_stay_identical(tmp_path):
    cfg = make_store(tmp_path, W=8, segments=1, train={"record_dir": str(tmp_path / "w")})
    run_world(cfg, 2, recording_step, build_model, str(tmp_path / "out"))
    w = load_weights(tmp_path / "w")
    # rank 0 took positions 0, 2, 4, 6 and rank 1 took 1, 3, 5, 7 at the same four steps
    assert sorted(k[:2] for k in w if k[2] == "post") == [
        (0, 0),
        (0, 2),
        (0, 4),
        (0, 6),
        (1, 1),
        (1, 3),
        (1, 5),
        (1, 7),
    ]
    for step in range(4):
        assert torch.allclose(
            w[(0, 2 * step, "post")], w[(1, 2 * step + 1, "post")], atol=0, rtol=0
        )
    # parallel.kind none: no wrap, and the different blocks drive the replicas apart at once
    cfg2 = make_store(
        tmp_path / "plain",
        W=8,
        segments=1,
        train={"record_dir": str(tmp_path / "w2")},
        parallel={"kind": "none"},
    )
    run_world(cfg2, 2, recording_step, build_model, str(tmp_path / "out2"))
    w2 = load_weights(tmp_path / "w2")
    assert not torch.equal(w2[(0, 0, "post")], w2[(1, 1, "post")])
    assert not torch.equal(
        w2[(0, 0, "post")], w[(0, 0, "post")]
    )  # DDP's step is neither replica's own


def test_resume_at_a_different_world_size_under_the_real_group(tmp_path):
    cfg = make_store(tmp_path, W=8, segments=2)
    world = run_world(cfg, 2, train_step, build_model, str(tmp_path / "out"))
    first = world.by_rank(0)[0]  # segment 0, cursor 2, n=2
    assert CheckpointIO.read_ledger(first.checkpoint()).done_positions() == 4
    world2 = run_world(
        cfg, 4, train_step, build_model, str(tmp_path / "out2"), checkpoint=first.checkpoint_path
    )
    fs, root = cfg.store_fs()
    records = [r for r in read_audit(fs, root, "fake") if r.attempt == 1]
    assert sorted(r.position for r in records) == list(range(4, 16))
    assert check_dealing(records, 8) == []
    firsts = {rank: min(r.position for r in records if r.rank == rank) for rank in range(4)}
    assert firsts == {0: 4, 1: 5, 2: 6, 3: 7}
    names = {r.checkpoint_dir_name for r in world2.reports}
    assert "checkpoint_g000000_p000008_n04_a01" in names
    assert all(name.endswith("_n04_a01") for name in names)
    assert all(r.metrics["attempt"] == 1 for r in world2.reports)


def test_segment_hook_appends_on_rank_zero_while_the_other_rank_waits(tmp_path):
    cfg = make_store(tmp_path, W=8, segments=1, end=False)
    world = run_world(
        cfg,
        2,
        train_step,
        build_model,
        str(tmp_path / "out"),
        loop_extra={"hooks": [AppendNextSegments(segments=3)]},
    )
    fs, root = cfg.store_fs()
    records = read_audit(fs, root, "fake")
    assert check_s1(records, 8) == []
    assert sorted({r.segment for r in records}) == [0, 1, 2]
    assert {r.rank for r in records if r.segment == 2} == {0, 1}
    log = BlockLog.open(fs, root)
    assert log.ended()
    # rank 1 polled for every hook-made segment and never read a block before its commit
    for seq in (1, 2):
        first_ts = min(r.ts for r in records if r.segment == seq)
        assert log.read_segment(seq).meta["created_at"] <= first_ts
    assert [r.metrics["cursor"] for r in world.by_rank(1)] == [2, 4, 2, 4, 2, 4]


# ---- M9: the segment-end sync (local SGD, DiLoCo) ----


def test_local_sgd_trains_alone_and_averages_at_the_segment_end(tmp_path):
    cfg = make_store(
        tmp_path,
        W=8,
        segments=2,
        train={"record_dir": str(tmp_path / "w")},
        parallel={"kind": "local_sgd"},
    )
    world = run_world(cfg, 2, recording_step, build_model, str(tmp_path / "out"))
    w = load_weights(tmp_path / "w")
    # inside a segment the replicas drift apart (no collective per step)
    assert not torch.equal(w[(0, 0, "post")], w[(1, 1, "post")])
    # the sync: segment 1 starts on both ranks from the average of the replicas' last weights
    avg = (w[(0, 6, "post")] + w[(1, 7, "post")]) / 2
    assert torch.allclose(w[(0, 8, "pre")], avg) and torch.allclose(w[(1, 9, "pre")], avg)
    # the segment-end checkpoint is taken after the sync: it holds the averaged weights, while
    # the mid-segment one holds rank 0's own drifted weights
    ckpts = world.checkpoints()
    assert torch.allclose(load_weight(ckpts["checkpoint_g000000_p000008_n02_a00"]), avg)
    assert torch.equal(load_weight(ckpts["checkpoint_g000000_p000004_n02_a00"]), w[(0, 2, "post")])
    fs, root = cfg.store_fs()
    assert check_s1(read_audit(fs, root, "fake"), 8) == []


def test_diloco_with_outer_lr_one_and_no_momentum_equals_local_sgd(tmp_path):
    finals = {}
    for kind, extra in (("local_sgd", {}), ("diloco", {"outer_lr": 1.0, "outer_momentum": 0.0})):
        cfg = make_store(tmp_path / kind, W=8, segments=2, parallel={"kind": kind, **extra})
        world = run_world(cfg, 2, train_step, build_model, str(tmp_path / kind / "out"))
        finals[kind] = load_weight(world.checkpoints()["checkpoint_g000001_p000008_n02_a00"])
    assert torch.allclose(finals["local_sgd"], finals["diloco"])


def test_diloco_outer_state_round_trips_through_a_checkpoint(tmp_path):
    cfg = make_store(tmp_path, W=8, segments=3, parallel={"kind": "diloco", "outer_momentum": 0.9})
    world = run_world(cfg, 2, train_step, build_model, str(tmp_path / "out"))
    ckpts = world.checkpoints()
    end1 = ckpts["checkpoint_g000001_p000008_n02_a00"]  # segment 1 complete: synced weights
    assert end1.checkpoint_path is not None
    assert os.path.exists(os.path.join(end1.checkpoint_path, CheckpointIO.PARALLEL))
    # a segment-end resume is exact: every rank restarts from the synced weights, the anchor
    # and the outer momentum come back, and the sync at the end of segment 2 uses them
    world2 = run_world(
        cfg, 2, train_step, build_model, str(tmp_path / "out2"), checkpoint=end1.checkpoint_path
    )
    resumed = load_weight(world2.checkpoints()["checkpoint_g000002_p000008_n02_a01"])
    assert torch.equal(resumed, load_weight(ckpts["checkpoint_g000002_p000008_n02_a00"]))
    # a mid-segment checkpoint holds rank 0's drifted replica: a resume restarts every rank
    # from it (row-exact positions, the other replicas' drift lost) with the sync state restored,
    # and without that state (the file removed) the outcome differs
    mid = ckpts["checkpoint_g000001_p000004_n02_a00"]
    assert mid.checkpoint_path is not None
    world3 = run_world(
        cfg, 2, train_step, build_model, str(tmp_path / "out3"), checkpoint=mid.checkpoint_path
    )
    fs, root = cfg.store_fs()
    records = [r for r in read_audit(fs, root, "fake") if r.attempt == 2]
    assert sorted(r.position for r in records) == list(range(12, 24))
    assert check_dealing(records, 8) == []
    os.remove(os.path.join(mid.checkpoint_path, CheckpointIO.PARALLEL))
    world4 = run_world(
        cfg, 2, train_step, build_model, str(tmp_path / "out4"), checkpoint=mid.checkpoint_path
    )
    with_state = load_weight(world3.checkpoints()["checkpoint_g000001_p000008_n02_a02"])
    without = load_weight(world4.checkpoints()["checkpoint_g000001_p000008_n02_a03"])
    assert not torch.equal(with_state, without)


# ---- M9: fsdp and the sharded checkpoint shape ----


def mlp_model(info):
    """A model with enough parameters to shard, and an optimizer with state."""
    torch.manual_seed(0)
    m = torch.nn.Sequential(torch.nn.Linear(1, 8), torch.nn.Tanh(), torch.nn.Linear(8, 1))
    return m, torch.optim.Adam(m.parameters(), lr=0.01)


def load_mlp(checkpoint_dir: str) -> torch.nn.Module:
    """A driver-side load (no process group) of either checkpoint shape into a plain MLP."""
    from ray.train import Checkpoint

    m, _ = mlp_model(None)
    CheckpointIO.load(Checkpoint.from_directory(checkpoint_dir), m)
    return m


def flat(m: torch.nn.Module) -> torch.Tensor:
    return torch.cat([p.detach().flatten() for p in m.parameters()])


def test_fsdp_shards_the_model_and_every_rank_writes_its_shard(tmp_path):
    finals = {}
    for kind in ("ddp", "fsdp"):
        cfg = make_store(tmp_path / kind, W=8, segments=2, parallel={"kind": kind})
        world = run_world(cfg, 2, train_step, mlp_model, str(tmp_path / kind / "out"))
        name = "checkpoint_g000001_p000008_n02_a00"
        if kind == "ddp":
            assert [r.has_checkpoint for r in world.by_rank(1)] == [False] * 4
            finals[kind] = flat(load_mlp(world.checkpoints()[name].checkpoint_path))
        else:
            assert all(r.has_checkpoint for r in world.reports)  # every rank reports its shard
            merged = world.merge_checkpoint(name, str(tmp_path / kind / "merged"))
            files = sorted(os.listdir(merged))
            assert "__0_0.distcp" in files and "__1_0.distcp" in files and ".metadata" in files
            assert "ledger.json" in files and "model.pt" not in files
            from ray.train import Checkpoint

            assert CheckpointIO.shape(Checkpoint.from_directory(merged)) == ("sharded", 2)
            assert CheckpointIO.read_ledger(Checkpoint.from_directory(merged)) == Ledger(
                1, 4, 2, 0, 0
            )
            finals[kind] = flat(load_mlp(merged))
        fs, root = cfg.store_fs()
        assert check_s1(read_audit(fs, root, "fake"), 8) == []
    # the same batches and the same effective weights: fsdp trains like ddp
    assert torch.allclose(finals["ddp"], finals["fsdp"], atol=1e-5)


def test_sharded_checkpoint_resumes_exactly_at_the_same_world_size_and_reshards_at_another(
    tmp_path,
):
    cfg = make_store(tmp_path, W=8, segments=3, parallel={"kind": "fsdp"})
    world = run_world(cfg, 2, train_step, mlp_model, str(tmp_path / "out"))
    end0 = world.merge_checkpoint("checkpoint_g000000_p000008_n02_a00", str(tmp_path / "m0"))
    mid1 = world.merge_checkpoint("checkpoint_g000001_p000004_n02_a00", str(tmp_path / "m1"))
    final = world.merge_checkpoint("checkpoint_g000002_p000008_n02_a00", str(tmp_path / "m2"))
    # the same world size, from a mid-segment checkpoint (the replicas agree under fsdp): exact,
    # the optimizer's moments included
    world2 = run_world(cfg, 2, train_step, mlp_model, str(tmp_path / "out2"), checkpoint=mid1)
    resumed = world2.merge_checkpoint("checkpoint_g000002_p000008_n02_a01", str(tmp_path / "r2"))
    assert torch.allclose(flat(load_mlp(resumed)), flat(load_mlp(final)), atol=1e-6)
    # four ranks from the two-rank segment-end checkpoint: the shards are re-cut on load
    world4 = run_world(cfg, 4, train_step, mlp_model, str(tmp_path / "out4"), checkpoint=end0)
    fs, root = cfg.store_fs()
    records = [r for r in read_audit(fs, root, "fake") if r.attempt == 2]
    assert sorted(r.position for r in records) == list(range(8, 24))
    assert check_dealing(records, 8) == []
    r4 = world4.merge_checkpoint("checkpoint_g000002_p000008_n04_a02", str(tmp_path / "r4"))
    from ray.train import Checkpoint

    assert CheckpointIO.shape(Checkpoint.from_directory(r4)) == ("sharded", 4)
    assert torch.isfinite(flat(load_mlp(r4))).all()
    # and one rank from the same checkpoint (a shrink): one shard
    world1 = run_world(cfg, 1, train_step, mlp_model, str(tmp_path / "out1"), checkpoint=end0)
    r1 = world1.merge_checkpoint("checkpoint_g000002_p000008_n01_a03", str(tmp_path / "r1"))
    assert CheckpointIO.shape(Checkpoint.from_directory(r1)) == ("sharded", 1)


def test_a_full_checkpoint_resumes_under_fsdp(tmp_path):
    cfg = make_store(tmp_path, W=8, segments=2, parallel={"kind": "ddp"})
    world = run_world(cfg, 2, train_step, mlp_model, str(tmp_path / "out"))
    end0 = world.checkpoints()["checkpoint_g000000_p000008_n02_a00"]
    assert end0.checkpoint_path is not None
    cfg2 = make_store(tmp_path / "f", W=8, segments=2, parallel={"kind": "fsdp"})
    world2 = run_world(
        cfg2, 2, train_step, mlp_model, str(tmp_path / "out2"), checkpoint=end0.checkpoint_path
    )
    fs, root = cfg2.store_fs()
    records = read_audit(fs, root, "fake")
    assert sorted(r.position for r in records) == list(range(8, 16))  # segment 1 only
    merged = world2.merge_checkpoint("checkpoint_g000001_p000008_n02_a00", str(tmp_path / "m"))
    assert torch.isfinite(flat(load_mlp(merged))).all()
