"""toy_contrastive: train the MLP encoder with InfoNCE on the mined blocks (``just contrastive``).

Run: ``just contrastive`` (= ``train.py --config examples/toy_contrastive/local.yaml``).
"""

from __future__ import annotations

import argparse
import logging
import os
import shutil
import sys
from pathlib import Path

import numpy as np
import pyarrow as pa
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from distrainer.audit import read_audit  # noqa: E402
from distrainer.config import DistrainerConfig, load_config  # noqa: E402
from distrainer.log import BlockLog  # noqa: E402
from distrainer.trainer import CheckpointIO, DistTrainer, TrainInfo  # noqa: E402
from examples.toy_contrastive.model import Encoder, info_nce  # noqa: E402
from integration_tests.cluster.check_audit import check_s1, summarize  # noqa: E402


def build_model(info: TrainInfo) -> tuple[torch.nn.Module, torch.optim.Optimizer]:
    t = info.train
    torch.manual_seed(info.config.seed)
    model = Encoder(int(t.get("features", 32)), int(t.get("hidden", 64)), int(t.get("embed", 16)))
    return model, torch.optim.Adam(model.parameters(), lr=float(t.get("lr", 1e-3)))


def _stack(table: pa.Table, column: str) -> torch.Tensor:
    return torch.from_numpy(np.stack(table.column(column).to_numpy(zero_copy_only=False)))


def train_step(
    model: torch.nn.Module, optimizer: torch.optim.Optimizer, table: pa.Table, info: TrainInfo
) -> dict[str, float]:
    t = info.train
    k = int(t.get("hard_negatives", 4))
    anchor = _stack(table, "anchor")
    positive = _stack(table, "positive")
    negatives = torch.stack([_stack(table, f"neg_{j}") for j in range(k)], dim=1)
    optimizer.zero_grad()
    loss = info_nce(
        model(anchor),
        model(positive),
        model(negatives.reshape(-1, negatives.shape[-1])).reshape(negatives.shape[0], k, -1),
        temperature=float(t.get("temperature", 0.1)),
        all_gather=bool(t.get("all_gather", False)),
    )
    loss.backward()
    optimizer.step()
    return {"loss": float(loss.item())}


def entry(cfg: DistrainerConfig):
    """For ``distrainer resume --entry examples.toy_contrastive.train:entry``."""
    return train_step, build_model


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--keep", action="store_true")
    ap.add_argument("--no-check", action="store_true")
    ap.add_argument("--set", action="append", default=[], metavar="KEY=VALUE")
    args = ap.parse_args(argv)
    cfg = load_config(args.config, overrides=args.set)
    _, runs_root = cfg.runs_fs()
    fs, store_root = cfg.store_fs()
    if not args.keep:
        shutil.rmtree(Path(runs_root) / cfg.run_name, ignore_errors=True)
        shutil.rmtree(Path(store_root) / "audit" / cfg.run_name, ignore_errors=True)
    os.environ.setdefault("RAY_ENABLE_UV_RUN_RUNTIME_ENV", "0")
    import ray

    ray.init(address=cfg.ray_address, ignore_reinit_error=True, logging_level=logging.ERROR)
    if not BlockLog(fs, store_root).exists():
        from examples.toy_contrastive.make_blocks import make_blocks

        refs = make_blocks(cfg)
        print(f"wrote {len(refs)} blocks")
    result = DistTrainer(train_step, build_model, cfg).fit()
    ledger = CheckpointIO.read_ledger(result.checkpoint) if result.checkpoint else None
    print(f"final metrics: {result.metrics}")
    print(f"final ledger: {ledger}")
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
