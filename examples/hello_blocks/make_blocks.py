"""hello_blocks: the smallest possible block log.

Synthetic linear-regression rows ``y = x . w + noise`` with ``d`` features, ``rows_per_block``
rows per block, ``n_blocks`` blocks, written as Parquet files plus a log with ``W`` blocks per
segment and ``passes`` passes. No Ray needed here; ``train.py`` calls this when the store is empty.
"""

from __future__ import annotations

import argparse

import numpy as np
import pyarrow as pa

from distrainer.block import BlockRef, write_block
from distrainer.config import DistrainerConfig, load_config
from distrainer.log import BlockLog
from distrainer.writer import BatchWriter


def feature_columns(d: int) -> list[str]:
    return [f"x_{i}" for i in range(d)]


def make_blocks(cfg: DistrainerConfig) -> list[BlockRef]:
    """Write blocks and the log for ``cfg``; returns the block refs. Idempotent per store."""
    t = cfg.train
    n_blocks = int(t.get("n_blocks", 48))
    rows = int(t.get("rows_per_block", 32))
    d = int(t.get("features", 8))
    fs, root = cfg.store_fs()
    log = BlockLog.create(fs, root, W=cfg.log.W, seed=cfg.seed)
    rng = np.random.default_rng(cfg.seed)
    w_true = rng.normal(size=d)
    refs = []
    for i in range(n_blocks):
        x = rng.normal(size=(rows, d)).astype(np.float32)
        y = (x @ w_true + 0.05 * rng.normal(size=rows)).astype(np.float32)
        cols = {name: pa.array(x[:, j]) for j, name in enumerate(feature_columns(d))}
        cols["y"] = pa.array(y)
        refs.append(write_block(fs, root, f"b{i:05d}", pa.table(cols)))
    BatchWriter(log, refs, passes=cfg.log.passes, tail="error").run()
    return refs


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    cfg = load_config(ap.parse_args().config)
    refs = make_blocks(cfg)
    fs, root = cfg.store_fs()
    log = BlockLog.open(fs, root)
    print(f"wrote {len(refs)} blocks and {log.last_seq() + 1} segments under {root}")


if __name__ == "__main__":
    main()
