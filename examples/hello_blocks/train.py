"""hello_blocks: train a linear model on a block log with DistTrainer (the ``just smoke`` gate).

Run: ``uv run python examples/hello_blocks/train.py --config examples/hello_blocks/local.yaml``.
Builds the blocks if the store is empty, starts a local Ray cluster (or joins ``ray_address``),
trains, then checks the audit trail with the S1 assertions and prints the final ledger.
"""

from __future__ import annotations

import argparse
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
from distrainer.trainer import CheckpointIO, DistTrainer, TrainInfo, init_ray  # noqa: E402
from examples.hello_blocks.make_blocks import feature_columns, make_blocks  # noqa: E402
from integration_tests.cluster.check_audit import (  # noqa: E402
    check_report_count,
    check_s1,
    expected_reports,
    summarize,
)


def build_model(info: TrainInfo) -> tuple[torch.nn.Module, torch.optim.Optimizer]:
    d = int(info.train.get("features", 8))
    torch.manual_seed(info.config.seed)
    model = torch.nn.Linear(d, 1)
    return model, torch.optim.SGD(model.parameters(), lr=float(info.train.get("lr", 0.05)))


def train_step(
    model: torch.nn.Module, optimizer: torch.optim.Optimizer, table: pa.Table, info: TrainInfo
) -> dict[str, float]:
    d = int(info.train.get("features", 8))
    x = torch.from_numpy(
        np.column_stack([table.column(c).to_numpy() for c in feature_columns(d)]).astype("float32")
    )
    y = torch.from_numpy(table.column("y").to_numpy().astype("float32")).unsqueeze(1)
    optimizer.zero_grad()
    loss = torch.nn.functional.mse_loss(model(x), y)
    loss.backward()  # DDP all-reduces the gradients here
    optimizer.step()
    return {"loss": float(loss.item())}


def entry(cfg: DistrainerConfig):
    """For ``distrainer resume --entry examples.hello_blocks.train:entry``."""
    return train_step, build_model


def ensure_blocks(cfg: DistrainerConfig) -> None:
    fs, root = cfg.store_fs()
    if not BlockLog(fs, root).exists():
        make_blocks(cfg)


def reset_run(cfg: DistrainerConfig) -> None:
    """A fresh run: previous run dir and audit trail of this run_name are removed (local only)."""
    if cfg.storage.kind != "local":
        return
    _, runs_root = cfg.runs_fs()
    _, store_root = cfg.store_fs()
    shutil.rmtree(Path(runs_root) / cfg.run_name, ignore_errors=True)
    shutil.rmtree(Path(store_root) / "audit" / cfg.run_name, ignore_errors=True)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--keep", action="store_true", help="do not wipe a previous run of this name")
    ap.add_argument("--no-check", action="store_true", help="skip the S1 audit assertions")
    ap.add_argument(
        "--set",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="override a config value, dotted keys, YAML values (e.g. checkpoint.num_to_keep=null)",
    )
    args = ap.parse_args(argv)
    cfg = load_config(args.config, overrides=args.set)
    if not args.keep:
        reset_run(cfg)
    ensure_blocks(cfg)

    init_ray(cfg)
    result = DistTrainer(train_step, build_model, cfg).fit()
    ledger = CheckpointIO.read_ledger(result.checkpoint) if result.checkpoint else None
    print(f"final metrics: {result.metrics}")
    print(f"final ledger: {ledger}")

    fs, store_root = cfg.store_fs()
    records = read_audit(fs, store_root, cfg.run_name)
    print(summarize(records))
    if args.no_check:
        return 0
    W = BlockLog.open(fs, store_root).W
    problems = check_s1(records, W)
    n_reports = int(
        (result.metrics or {}).get("reports", 0)
    )  # metrics_dataframe only keeps kept checkpoints
    expected = expected_reports(len(records) // W, W, cfg.scaling.max_workers, cfg.checkpoint)
    problems += check_report_count(n_reports, expected)
    print(f"reports per rank: {n_reports} (expected {expected})")
    for p in problems:
        print("S1 FAIL:", p)
    print("S1 PASS" if not problems else f"S1: {len(problems)} problem(s)")
    return 1 if problems else 0


if __name__ == "__main__":
    raise SystemExit(main())
