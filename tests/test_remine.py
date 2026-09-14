"""The re-mining hook: anchors per segment, embedding-space negatives, streaming the toy log."""

import numpy as np
import pytest
import torch
from test_train_loop import fake_ray, run_world  # noqa: F401 (fixture)

from distrainer.audit import read_audit
from distrainer.config import DistrainerConfig
from distrainer.log import BlockLog
from examples.toy_contrastive.make_blocks import make_blocks, make_corpus, mine_rows
from examples.toy_contrastive.model import Encoder
from examples.toy_contrastive.remine import (
    RemineHook,
    embedded_neighbours,
    initial_log,
    remine_args,
    segment_anchors,
)
from examples.toy_contrastive.train import build_model, train_step
from integration_tests.cluster.check_audit import check_s6

TRAIN = {
    "n_items": 96,
    "features": 6,
    "clusters": 8,
    "anchors_per_block": 4,
    "hard_negatives": 3,
    "hidden": 8,
    "embed": 4,
}


def remine_cfg(tmp_path, segments=3, initial_segments=1, W=4, **extra):
    return DistrainerConfig.from_dict(
        {
            "run_name": "remine",
            "storage_path": str(tmp_path / "runs"),
            "store_root": str(tmp_path / "blocks"),
            "seed": 3,
            "log": {"W": W, "passes": 1, "wait_poll_s": 0.01},
            "checkpoint": {"policy": "any", "every_k": 2},
            "scaling": {"num_workers": 1},
            "hooks": {
                "remine": {
                    "entry": "examples.toy_contrastive.remine:RemineHook",
                    "segments": segments,
                    "initial_segments": initial_segments,
                }
            },
            "train": TRAIN,
            **extra,
        }
    )


def test_segment_anchors_slice_per_pass_permutations_and_wrap():
    p0, a0 = segment_anchors(96, 16, 0, seed=3)
    p5, a5 = segment_anchors(96, 16, 5, seed=3)
    p6, a6 = segment_anchors(96, 16, 6, seed=3)
    assert (p0, p5, p6) == (0, 0, 1)
    covered = np.concatenate([segment_anchors(96, 16, s, 3)[1] for s in range(6)])
    assert sorted(covered) == list(range(96))  # one pass covers the corpus exactly once
    assert len(a6) == 16 and sorted(np.concatenate([a0, a5])) != sorted(a6)
    # a slice straddling the pass boundary takes the tail of one permutation and the head of
    # the next, and is labelled with the pass of its first item
    p, a = segment_anchors(10, 4, 2, seed=1)
    assert p == 0 and len(a) == 4 and len(set(a[:2])) == 2
    with pytest.raises(ValueError):
        segment_anchors(0, 4, 0, 1)


def test_mine_rows_for_a_subset_of_anchors_matches_the_full_mining_rules():
    cfg = DistrainerConfig.from_dict({"seed": 3, "train": TRAIN})
    items, cluster, centroids = make_corpus(cfg)
    from examples.toy_contrastive.make_blocks import hard_negative_clusters

    near = hard_negative_clusters(centroids, 3)
    anchors = np.array([5, 17, 40, 2])
    mined = mine_rows(items, cluster, near, 3, np.random.default_rng(0), anchors=anchors)
    assert mined["positive"].shape == (4,) and mined["negatives"].shape == (4, 3)
    for row, i in enumerate(anchors):
        assert cluster[mined["positive"][row]] == cluster[i]
        assert mined["positive"][row] != i or (cluster == cluster[i]).sum() == 1
        for j in range(3):
            assert cluster[mined["negatives"][row, j]] == near[cluster[i]][j]


def test_embedded_neighbours_uses_the_encoder_and_restores_train_mode():
    cfg = DistrainerConfig.from_dict({"seed": 3, "train": TRAIN})
    items, cluster, centroids = make_corpus(cfg)
    torch.manual_seed(0)
    enc = Encoder(6, 8, 4)
    near = embedded_neighbours(enc, items, cluster, 8, 3)
    assert near.shape == (8, 3) and all(c not in near[c] for c in range(8))
    assert enc.training  # left as it was
    enc.eval()
    embedded_neighbours(enc, items, cluster, 8, 3)
    assert not enc.training


def test_initial_log_streams_only_the_first_segments(tmp_path):
    cfg = remine_cfg(tmp_path, segments=4, initial_segments=2)
    assert remine_args(cfg) == {"segments": 4, "initial_segments": 2}
    refs = make_blocks(cfg)  # streaming: no Ray Data, no _END
    fs, root = cfg.store_fs()
    log = BlockLog.open(fs, root)
    assert log.committed_seqs() == [0, 1] and not log.ended()
    assert len(refs) == 8 and all(r.block_id.startswith("s00000") for r in refs)
    assert sorted(r.block_id for r in refs)[0] == "s000000a00b00000"
    assert log.read_segment(1).meta["space"] == "input"
    assert make_blocks(cfg) == sorted(refs, key=lambda r: r.block_id)  # idempotent
    with pytest.raises(ValueError):
        initial_log(DistrainerConfig.from_dict({"train": TRAIN}))
    with pytest.raises(ValueError):
        RemineHook(cfg, segments=1, initial_segments=2)
    with pytest.raises(ValueError, match="multiple of W"):  # 96 items, 5 x 4 anchors per segment
        RemineHook(remine_cfg(tmp_path / "w5", W=5), segments=2)


def test_write_segment_refuses_a_wrong_sequence_number_or_w(tmp_path):
    cfg = remine_cfg(tmp_path, segments=3)
    hook = RemineHook(cfg, segments=3)
    fs, root = cfg.store_fs()
    log = BlockLog.create(fs, root, W=4, seed=3)
    from examples.toy_contrastive.make_blocks import hard_negative_clusters

    near = hard_negative_clusters(hook.centroids, 3)
    with pytest.raises(RuntimeError, match="not the next one"):
        hook.write_segment(log, 1, near, np.random.default_rng(0))
    seg = hook.write_segment(log, 0, near, np.random.default_rng(0), attempt=2)
    assert all(b.block_id.startswith("s000000a02b") for b in seg.blocks)
    other = BlockLog.create(fs, root + "/w8", W=8, seed=3)
    with pytest.raises(ValueError, match="differs"):
        hook.write_segment(other, 0, near, np.random.default_rng(0))


def test_mining_survives_an_empty_cluster():
    cfg = DistrainerConfig.from_dict({"seed": 3, "train": TRAIN})
    items, cluster, centroids = make_corpus(cfg)
    cluster = cluster.copy()
    cluster[cluster == 7] = 0  # cluster 7 has no items now
    torch.manual_seed(0)
    near = embedded_neighbours(Encoder(6, 8, 4), items, cluster, 8, 3)
    assert not (near == 7).any(), "an absent cluster is never a hard negative"
    forced = np.full((8, 3), 7)  # neighbours that all point at the empty cluster
    mined = mine_rows(items, cluster, forced, 3, np.random.default_rng(0))
    assert (cluster[mined["negatives"]] != cluster[:, None]).all(), "fallback avoids own cluster"


def test_hook_appends_mined_segments_then_ends_the_log(tmp_path, fake_ray):  # noqa: F811
    cfg = remine_cfg(tmp_path, segments=8, initial_segments=1)
    make_blocks(cfg)
    run_world(fake_ray, cfg, n=1, step=train_step, loop_extra={"build_model": build_model})
    fs, root = cfg.store_fs()
    log = BlockLog.open(fs, root)
    assert log.committed_seqs() == list(range(8)) and log.ended()
    segments = list(log.segments())
    # 16 anchors per segment over 96 items: segments 6 and 7 belong to the second pass
    assert [s.pass_idx for s in segments] == [0] * 6 + [1] * 2
    assert all(
        s.meta["space"] == "embedding" and s.meta["writer"] == "remine" for s in segments[1:]
    )
    assert [s.meta["mined_after_segment"] for s in segments[1:]] == list(range(7))
    records = read_audit(fs, root, "remine")
    assert check_s6(records, segments, 4, initial_segments=1, expected_segments=8) == []
    assert {r.block_id[:10] for r in records if r.segment == 3} == {"s000003a00"}
    # the hook's blocks carry positives of the anchor's cluster
    from distrainer.block import read_block

    table = read_block(fs, root, segments[3].blocks[0])
    assert table.num_rows == 4 and "neg_2" in table.column_names and "anchor" in table.column_names


def test_hook_is_idempotent_across_restarts(tmp_path):
    cfg = remine_cfg(tmp_path, segments=3, initial_segments=1)
    make_blocks(cfg)
    fs, root = cfg.store_fs()
    log = BlockLog.open(fs, root)
    hook = RemineHook(cfg, segments=3)
    model, _ = build_model(
        __import__("distrainer.trainer", fromlist=["TrainInfo"]).TrainInfo(0, 1, cfg)
    )
    from distrainer.ledger import Ledger

    hook.on_segment_end(model, Ledger(segment=0, cursor=4, world_size=1), log, None)
    assert log.committed_seqs() == [0, 1] and hook.mined == [1]
    hook.on_segment_end(model, Ledger(segment=0, cursor=4, world_size=1), log, None)  # replayed
    assert log.committed_seqs() == [0, 1] and hook.mined == [1]
    hook.on_segment_end(model, Ledger(segment=1, cursor=4, world_size=1), log, None)
    hook.on_segment_end(model, Ledger(segment=2, cursor=4, world_size=1), log, None)
    assert log.committed_seqs() == [0, 1, 2] and log.ended()
    hook.on_segment_end(model, Ledger(segment=2, cursor=4, world_size=1), log, None)  # no-op
    assert log.ended() and hook.mined == [1, 2]


def test_check_s6_flags_late_commits_and_reused_blocks(tmp_path):
    from conftest import make_refs

    from distrainer.audit import AuditRecord
    from distrainer.log import Segment

    W = 2
    base = Segment(0, W, 1, 0, make_refs(2), {"created_at": 0.0})
    hook_meta = {"created_at": 10.0, "writer": "remine", "mined_after_segment": 0}
    mined = Segment(1, W, 1, 0, [r for r in make_refs(4)[2:]], hook_meta)
    recs = [
        AuditRecord(0, 0, 1, 0, 0, 0, "b0000", 1.0),
        AuditRecord(0, 0, 1, 0, 1, 1, "b0001", 2.0),
        AuditRecord(0, 0, 1, 1, 0, 2, "b0002", 11.0),
        AuditRecord(0, 0, 1, 1, 1, 3, "b0003", 12.0),
    ]
    assert check_s6(recs, [base, mined], W, 1, expected_segments=2) == []
    late = Segment(1, W, 1, 0, mined.blocks, {**hook_meta, "created_at": 11.5})
    assert any("hook too late" in p for p in check_s6(recs, [base, late], W, 1))
    early = Segment(1, W, 1, 0, mined.blocks, {**hook_meta, "created_at": 1.5})  # rank 0 not done
    assert any("not mined at the segment end" in p for p in check_s6(recs, [base, early], W, 1))
    not_hook = Segment(1, W, 1, 0, mined.blocks, {"created_at": 10.0, "writer": "batch"})
    assert any("not written by the hook" in p for p in check_s6(recs, [base, not_hook], W, 1))
    reused = recs[:2] + [
        AuditRecord(0, 0, 1, 1, 0, 2, "b0000", 11.0),
        AuditRecord(0, 0, 1, 1, 1, 3, "b0001", 12.0),
    ]
    probs = check_s6(reused, [base, Segment(1, W, 1, 0, make_refs(2), hook_meta)], W, 1)
    assert any("re-used" in p for p in probs)
    assert any("expected 0..2" in p for p in check_s6(recs, [base, mined], W, 1, 3))
    assert any("not in the log" in p for p in check_s6(recs, [base], W, 1))
    assert check_s6([], [base], W, 1) == ["no audit records"]
