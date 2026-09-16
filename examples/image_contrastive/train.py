"""image_contrastive: SimCLR on CIFAR-10 blocks, with a kNN probe at the end (``just images``).

Run: ``just images`` (= ``train.py --config examples/image_contrastive/local.yaml``: a tiny
encoder on a CIFAR-10 subset, CPU) or ``just images
examples/image_contrastive/local-synthetic.yaml`` (no download). ``harness-s3.yaml`` is the GPU
shape for the RunPod pods (tutorial 6). Every step
decodes one block of PNG rows, makes two augmented views of every image on the worker's device,
and minimises NT-Xent between them; the other images of the block (and of the other ranks with
``all_gather``) are the negatives. After training the driver loads the final checkpoint and
prints the weighted kNN accuracy of the backbone on the held-out split.
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path
from typing import Any

import pyarrow as pa
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from distrainer.audit import read_audit  # noqa: E402
from distrainer.config import DistrainerConfig, load_config  # noqa: E402
from distrainer.log import BlockLog  # noqa: E402
from distrainer.trainer import CheckpointIO, DistTrainer, TrainInfo, init_ray  # noqa: E402
from examples.image_contrastive.data import decode_column  # noqa: E402
from examples.image_contrastive.model import Encoder, augment, nt_xent  # noqa: E402
from integration_tests.cluster.check_audit import check_s1, summarize  # noqa: E402


def build_model(info: TrainInfo) -> tuple[torch.nn.Module, torch.optim.Optimizer]:
    t = info.train
    torch.manual_seed(info.config.seed)
    model = Encoder(
        width=int(t.get("width", 64)),
        layers=[int(n) for n in t.get("layers", [2, 2, 2, 2])],
        embed=int(t.get("embed", 128)),
        proj_hidden=int(t["proj_hidden"]) if t.get("proj_hidden") else None,
    )
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=float(t.get("lr", 1e-3)),
        weight_decay=float(t.get("weight_decay", 0)),
    )
    return model, optimizer


def step_generator(info: TrainInfo) -> torch.Generator:
    """The views of a block depend on the run's seed and the position only, so a replayed
    position after a restart sees the same augmentations."""
    return torch.Generator().manual_seed((info.config.seed * 1_000_003 + info.position) % 2**63)


def train_step(
    model: torch.nn.Module, optimizer: torch.optim.Optimizer, table: pa.Table, info: TrainInfo
) -> dict[str, float]:
    t = info.train
    x = torch.from_numpy(decode_column(table)).to(info.device, non_blocking=True)
    gen = step_generator(info)
    min_scale = float(t.get("crop_min_scale", 0.2))
    v1, v2 = augment(x, gen, min_scale=min_scale), augment(x, gen, min_scale=min_scale)
    amp = bool(t.get("amp", True)) and info.device.type == "cuda"
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=amp):
        z = model(torch.cat([v1, v2]))  # one forward: BatchNorm sees both views, as in SimCLR
    z1, z2 = z.float().chunk(2)
    loss = nt_xent(
        z1,
        z2,
        temperature=float(t.get("temperature", 0.1)),
        all_gather=bool(t.get("all_gather", True)),
    )
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()
    return {"loss": float(loss.item())}


def entry(cfg: DistrainerConfig):
    """For ``distrainer resume --entry examples.image_contrastive.train:entry``."""
    return train_step, build_model


def probe_checkpoint(cfg: DistrainerConfig, checkpoint: Any) -> dict[str, Any]:
    """Load the encoder of ``checkpoint`` on the driver and run the kNN probe on the store."""
    from examples.image_contrastive.probe import run_probe

    model, _ = build_model(TrainInfo(rank=0, world_size=1, config=cfg))
    CheckpointIO.load(checkpoint, model)
    assert isinstance(model, Encoder)
    return run_probe(cfg, model)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--keep", action="store_true")
    ap.add_argument("--no-check", action="store_true")
    ap.add_argument("--no-probe", action="store_true")
    ap.add_argument(
        "--rebuild",
        action="store_true",
        help="wipe the local store (blocks, log, probe split) first: a changed shape or seed "
        "would otherwise reuse the existing log silently",
    )
    ap.add_argument("--set", action="append", default=[], metavar="KEY=VALUE")
    args = ap.parse_args(argv)
    cfg = load_config(args.config, overrides=args.set)
    _, runs_root = cfg.runs_fs()
    fs, store_root = cfg.store_fs()
    if args.rebuild:
        if cfg.storage.kind != "local":
            raise SystemExit(
                "--rebuild wipes a local store only; delete the bucket prefix yourself"
            )
        shutil.rmtree(store_root, ignore_errors=True)
    if not args.keep and cfg.storage.kind == "local":
        shutil.rmtree(Path(runs_root) / cfg.run_name, ignore_errors=True)
        shutil.rmtree(Path(store_root) / "audit" / cfg.run_name, ignore_errors=True)
    init_ray(cfg)
    if not BlockLog(fs, store_root).exists():
        from examples.image_contrastive.make_blocks import make_blocks

        refs = make_blocks(cfg)
        print(f"wrote {len(refs)} blocks")
    result = DistTrainer(train_step, build_model, cfg).fit()
    ledger = CheckpointIO.read_ledger(result.checkpoint) if result.checkpoint else None
    print(f"final metrics: {result.metrics}")
    print(f"final ledger: {ledger}")
    if result.checkpoint is not None and not args.no_probe:
        probe = probe_checkpoint(cfg, result.checkpoint)
        print(
            f"probe: knn_acc={probe['knn_acc']:.3f} (chance {probe['chance']:.2f}) on "
            f"{probe['probe_test']} held-out images against {probe['probe_train']} "
            f"training images, {probe['device']}"
        )
    records = read_audit(fs, store_root, cfg.run_name)
    print(summarize(records))
    if args.no_check:
        return 0
    problems = check_s1(records, BlockLog.open(fs, store_root).W)
    for p in problems:
        print("S1 FAIL:", p)
    print("S1 PASS" if not problems else f"S1: {len(problems)} problem(s)")
    return 1 if problems else 0


if __name__ == "__main__":
    raise SystemExit(main())
