"""examples/image_contrastive at CPU size on synthetic images: blocks, views, step, probe."""

from pathlib import Path

import numpy as np
import pytest
import torch
from test_train_loop import fake_ray, run_world  # noqa: F401 (fixture)

from distrainer.audit import read_audit
from distrainer.block import read_block
from distrainer.config import DistrainerConfig, load_config
from distrainer.log import BlockLog
from distrainer.trainer import TrainInfo
from examples.image_contrastive.data import (
    decode_column,
    decode_png,
    encode_png,
    images_table,
    load_images,
    synthetic_images,
)
from examples.image_contrastive.make_blocks import make_blocks, read_probe_split
from examples.image_contrastive.model import Encoder, augment, normalize, nt_xent
from examples.image_contrastive.probe import bank_from_log, embed, knn_accuracy, run_probe
from examples.image_contrastive.train import (
    build_model,
    probe_checkpoint,
    step_generator,
    train_step,
)
from integration_tests.cluster.check_audit import check_s1

TRAIN = {
    "dataset": "synthetic",
    "n_items": 64,
    "n_test": 24,
    "rows_per_block": 8,
    "image_size": 16,
    "width": 4,
    "layers": [1, 1, 1, 1],
    "embed": 8,
    "proj_hidden": 8,
    "probe_train": 32,
    "probe_test": 24,
    "probe_k": 5,
}


def small_cfg(tmp_path, **extra):
    return DistrainerConfig.from_dict(
        {
            "run_name": "images",
            "storage_path": str(tmp_path / "runs"),
            "store_root": str(tmp_path / "blocks"),
            "seed": 3,
            "log": {"W": 4, "passes": 1, "wait_poll_s": 0.01},
            "checkpoint": {"policy": "any", "every_k": 2, "num_to_keep": None},
            "scaling": {"num_workers": [1, 2]},
            "train": {**TRAIN, **extra.pop("train", {})},
            **extra,
        }
    )


def test_synthetic_images_are_deterministic_uint8_and_classes_separate_in_pixel_space():
    images, labels = synthetic_images(40, seed=1, size=16)
    images2, labels2 = synthetic_images(40, seed=1, size=16)
    assert images.shape == (40, 16, 16, 3) and images.dtype == np.uint8
    assert labels.shape == (40,) and labels.dtype == np.int64 and labels.max() < 10
    assert np.array_equal(images, images2) and np.array_equal(labels, labels2)
    assert not np.array_equal(images, synthetic_images(40, seed=2, size=16)[0])
    # colours plus stripes: a pixel-space kNN already beats chance by a wide margin
    bank, bank_labels = synthetic_images(200, seed=5, size=16)

    def flat(x: np.ndarray) -> torch.Tensor:
        pixels = torch.from_numpy(x.reshape(len(x), -1).astype(np.float32))
        return torch.nn.functional.normalize(pixels, dim=-1)

    acc = knn_accuracy(
        flat(bank), torch.from_numpy(bank_labels), flat(images), torch.from_numpy(labels), k=5
    )
    assert acc > 0.5


def test_png_round_trip_and_table_columns():
    images, labels = synthetic_images(3, seed=0, size=8)
    assert np.array_equal(decode_png(encode_png(images[0])), images[0])
    table = images_table(images, labels, np.array([7, 8, 9]))
    assert table.column_names == ["item_id", "label", "image"]
    assert table.column("item_id").to_pylist() == [7, 8, 9]
    assert np.array_equal(decode_column(table), images)


def test_load_images_dispatches_and_rejects_unknown_datasets(tmp_path):
    cfg = small_cfg(tmp_path)
    train_images, train_labels = load_images(cfg, "train")
    test_images, _ = load_images(cfg, "test")
    assert train_images.shape == (64, 16, 16, 3) and len(train_labels) == 64
    assert test_images.shape == (24, 16, 16, 3)
    assert not np.array_equal(train_images[:24], test_images)  # a different draw
    with pytest.raises(ValueError, match="cifar10 or synthetic"):
        load_images(small_cfg(tmp_path, train={"dataset": "imagenet"}), "train")


def test_make_blocks_writes_png_rows_the_probe_split_and_a_full_log(tmp_path):
    cfg = small_cfg(tmp_path)
    refs = make_blocks(cfg)
    fs, root = cfg.store_fs()
    assert [r.block_id for r in refs] == [f"b{i:05d}" for i in range(8)]
    assert all(r.num_rows == 8 and r.meta == {"dataset": "synthetic"} for r in refs)
    log = BlockLog.open(fs, root)
    assert log.W == 4 and log.committed_seqs() == [0, 1] and log.ended()
    table = read_block(fs, root, refs[0])
    assert set(table.column_names) == {"item_id", "label", "image", "block_id"}
    images = decode_column(table)
    assert images.shape == (8, 16, 16, 3)
    # rows are a seeded permutation of the corpus: item ids cover it exactly once
    ids = sorted(i for r in refs for i in read_block(fs, root, r).column("item_id").to_pylist())
    assert ids == list(range(64))
    test_images, test_labels = read_probe_split(fs, root)
    assert test_images.shape == (24, 16, 16, 3) and test_labels.shape == (24,)
    assert read_probe_split(fs, root, limit=5)[0].shape[0] == 5
    # idempotent: a second call returns the committed refs without rewriting
    assert [r.block_id for r in make_blocks(cfg)] == [r.block_id for r in refs]


def test_make_blocks_rejects_shapes_that_do_not_fill_segments(tmp_path):
    with pytest.raises(ValueError, match="multiple of rows_per_block"):
        make_blocks(small_cfg(tmp_path, train={"n_items": 60}))
    with pytest.raises(ValueError, match="whole segments"):
        make_blocks(small_cfg(tmp_path / "b", train={"n_items": 48}))  # 6 blocks, W=4


def test_augment_makes_normalised_per_sample_views_reproducible_from_the_generator():
    images, _ = synthetic_images(6, seed=0, size=16)
    x = torch.from_numpy(images)
    v1 = augment(x, torch.Generator().manual_seed(1))
    v2 = augment(x, torch.Generator().manual_seed(1))
    v3 = augment(x, torch.Generator().manual_seed(2))
    assert v1.shape == (6, 3, 16, 16) and v1.dtype == torch.float32
    assert torch.equal(v1, v2) and not torch.equal(v1, v3)
    assert torch.isfinite(v1).all()
    # different samples get different crops: the views are not one transform of the batch
    plain = normalize(x)
    diffs = [(v1[i] - plain[i]).abs().mean().item() for i in range(6)]
    assert len({round(d, 4) for d in diffs}) > 1


def test_encoder_shapes_and_nt_xent_prefers_matching_views():
    torch.manual_seed(0)
    enc = Encoder(width=4, layers=[1, 1, 1, 1], embed=8, proj_hidden=8)
    x = torch.randn(5, 3, 16, 16)
    assert enc.features(x).shape == (5, 32) and enc.backbone.feature_dim == 32
    z = enc(x)
    assert z.shape == (5, 8) and torch.allclose(z.norm(dim=-1), torch.ones(5), atol=1e-5)
    close = nt_xent(z, z + 0.01 * torch.randn_like(z), temperature=0.1)
    far = nt_xent(z, torch.nn.functional.normalize(torch.randn_like(z), dim=-1), temperature=0.1)
    assert close.item() < far.item()
    close.backward()
    assert all(p.grad is not None for p in enc.parameters())
    with pytest.raises(ValueError):
        Encoder(width=4, layers=[1, 1, 1])


def test_train_step_on_a_two_row_table_updates_the_model(tmp_path):
    cfg = small_cfg(tmp_path)
    info = TrainInfo(rank=0, world_size=1, config=cfg, position=3)
    model, optimizer = build_model(info)
    before = [p.detach().clone() for p in model.parameters()]
    images, labels = synthetic_images(2, seed=0, size=16)
    metrics = train_step(model, optimizer, images_table(images, labels, np.arange(2)), info)
    assert set(metrics) == {"loss"} and np.isfinite(metrics["loss"])
    assert any(not torch.equal(a, b) for a, b in zip(before, model.parameters(), strict=True))
    # the views of a position are a function of the seed and the position only
    g1, g2 = step_generator(info), step_generator(info)
    assert torch.equal(torch.rand(3, generator=g1), torch.rand(3, generator=g2))
    other = TrainInfo(rank=1, world_size=2, config=cfg, position=4, attempt=2)
    assert not torch.equal(
        torch.rand(3, generator=step_generator(info)),
        torch.rand(3, generator=step_generator(other)),
    )


def test_knn_accuracy_is_exact_on_separable_features_and_bounded():
    bank = torch.eye(4).repeat(5, 1)  # 20 one-hot features, 4 classes
    bank_labels = torch.arange(4).repeat(5)
    queries = torch.eye(4)
    assert knn_accuracy(bank, bank_labels, queries, torch.arange(4), k=3, classes=4) == 1.0
    assert (
        knn_accuracy(bank, bank_labels, queries, torch.tensor([1, 2, 3, 0]), k=3, classes=4) == 0.0
    )
    with pytest.raises(ValueError):
        knn_accuracy(bank[:0], bank_labels[:0], queries, torch.arange(4), classes=4)


def test_embed_and_run_probe_read_the_store(tmp_path):
    cfg = small_cfg(tmp_path)
    make_blocks(cfg)
    fs, root = cfg.store_fs()
    bank_images, bank_labels = bank_from_log(fs, root, limit=20)
    assert bank_images.shape == (20, 16, 16, 3) and bank_labels.shape == (20,)
    model, _ = build_model(TrainInfo(rank=0, world_size=1, config=cfg))
    assert isinstance(model, Encoder)
    feats = embed(model, bank_images, torch.device("cpu"), batch=7)
    assert feats.shape == (20, 32) and torch.allclose(feats.norm(dim=-1), torch.ones(20), atol=1e-5)
    assert model.training  # left as it was
    result = run_probe(cfg, model)
    assert 0.0 <= result["knn_acc"] <= 1.0 and result["chance"] == 0.1
    assert result["probe_train"] == 32 and result["probe_test"] == 24
    assert result["device"] == "cpu"


def test_the_loop_runs_the_example_end_to_end_and_the_probe_learns(tmp_path, fake_ray):  # noqa: F811
    cfg = small_cfg(tmp_path, train={"n_items": 96, "lr": 0.01})  # 12 blocks: 3 segments of 4
    make_blocks(cfg)
    fs, root = cfg.store_fs()
    model, _ = build_model(TrainInfo(rank=0, world_size=1, config=cfg))
    assert isinstance(model, Encoder)
    untrained = run_probe(cfg, model)["knn_acc"]
    run_world(fake_ray, cfg, n=2, step=train_step, loop_extra={"build_model": build_model})
    records = read_audit(fs, root, "images")
    assert check_s1(records, 4) == []
    assert all(np.isfinite(m["loss"]) for _, m, _, _ in fake_ray["reports"])
    ckpt = [c for r, _, _, c in fake_ray["reports"] if r == 0 and c is not None][-1]
    trained = probe_checkpoint(cfg, ckpt)["knn_acc"]  # what train.py does after fit()
    assert trained >= untrained - 0.2  # a handful of steps must not wreck the features
    assert trained > 0.1  # and the classes are colours: above chance after 6 steps per rank


def test_the_shipped_configs_load_and_fill_whole_segments():
    root = Path(__file__).resolve().parents[1] / "examples" / "image_contrastive"
    for name in ("local.yaml", "local-synthetic.yaml", "harness-s3.yaml"):
        cfg = load_config(str(root / name))
        t = cfg.train
        assert t["dataset"] in ("cifar10", "synthetic"), name
        per_segment = int(t["rows_per_block"]) * cfg.log.W
        assert int(t["n_items"]) % per_segment == 0, f"{name}: n_items must fill segments"
        assert int(t["n_test"]) >= int(t["probe_test"]), name
    gpu = load_config(str(root / "harness-s3.yaml"))
    assert gpu.scaling.use_gpu and gpu.scaling.resources_per_worker == {"GPU": 1, "trainer": 1}
    assert gpu.storage.kind == "s3" and gpu.ray_address == "auto"
