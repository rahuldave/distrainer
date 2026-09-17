"""train_loop on a real Gloo process group (tests/procgroup.py): the audit facts of the fake
fixture hold under DDP, and what the fake fixture cannot see (the all-reduce, the other rank
waiting at a segment end) is asserted here."""

import os

import torch
from conftest import AppendNextSegments
from procgroup import run_world
from test_train_loop import build_model, make_store, train_step

from distrainer.audit import next_attempt, read_audit
from distrainer.ledger import Ledger
from distrainer.log import BlockLog
from distrainer.trainer import CheckpointIO, unwrap
from integration_tests.cluster.check_audit import check_dealing, check_s1


def recording_step(model, optimizer, table, info):
    """``train_step``, then the weight after the step saved per (rank, position)."""
    out = train_step(model, optimizer, table, info)
    d = info.train["record_dir"]
    os.makedirs(d, exist_ok=True)
    torch.save(
        unwrap(model).weight.detach().clone(),
        os.path.join(d, f"r{info.rank}_p{info.position:04d}.pt"),
    )
    return out


def load_weights(d) -> dict[tuple[int, int], torch.Tensor]:
    out = {}
    for name in os.listdir(d):
        rank, pos = name[1:-3].split("_p")
        out[(int(rank), int(pos))] = torch.load(os.path.join(d, name))
    return out


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
    assert sorted(w) == [(0, 0), (0, 2), (0, 4), (0, 6), (1, 1), (1, 3), (1, 5), (1, 7)]
    for step in range(4):
        assert torch.allclose(w[(0, 2 * step)], w[(1, 2 * step + 1)], atol=0, rtol=0)
    # without the wrap the different blocks drive the replicas apart from the first step
    cfg2 = make_store(
        tmp_path / "plain", W=8, segments=1, train={"record_dir": str(tmp_path / "w2")}
    )
    run_world(cfg2, 2, recording_step, build_model, str(tmp_path / "out2"), prepare=False)
    w2 = load_weights(tmp_path / "w2")
    assert not torch.equal(w2[(0, 0)], w2[(1, 1)])
    assert not torch.equal(w2[(0, 0)], w[(0, 0)])  # and DDP's first step is neither replica's own


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
