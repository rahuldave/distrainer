"""image_contrastive: CIFAR-10 (or synthetic) images -> PNG rows in Parquet blocks -> a log.

Every block holds ``rows_per_block`` images drawn at random (a seeded permutation of the
``n_items`` training images), as ``(item_id, label, image)`` with the image as PNG bytes; the
``label`` rides along for the probe only, training never reads it. The test split goes to
``<store_root>/probe/test.parquet`` (outside the log, so retention gc never touches it) for the
kNN probe at the end of a run. ``BatchWriter`` then writes ``log.passes`` passes of ``log.W``
blocks per segment. No Ray needed; ``train.py`` calls this when the store is empty.
"""

from __future__ import annotations

import argparse

import numpy as np
import pyarrow as pa
import pyarrow.fs as pafs
import pyarrow.parquet as pq

from distrainer.block import BlockRef, write_block
from distrainer.config import DistrainerConfig, load_config
from distrainer.log import BlockLog
from distrainer.storage import ensure_dir, exists, join, write_atomic
from distrainer.writer import BatchWriter
from examples.image_contrastive.data import decode_column, images_table, load_images

PROBE_DIR = "probe"
PROBE_TEST = "test.parquet"


def write_probe_split(
    fs: pafs.FileSystem, root: str, images: np.ndarray, labels: np.ndarray
) -> str:
    """The held-out images as one Parquet file under ``<root>/probe/``; returns its path."""
    ensure_dir(fs, join(root, PROBE_DIR))
    path = join(root, PROBE_DIR, PROBE_TEST)
    sink = pa.BufferOutputStream()
    pq.write_table(images_table(images, labels, np.arange(len(images))), sink)
    write_atomic(fs, path, sink.getvalue().to_pybytes())
    return path


def read_probe_split(
    fs: pafs.FileSystem, root: str, limit: int | None = None
) -> tuple[np.ndarray, np.ndarray]:
    """``(images, labels)`` of the held-out split, the first ``limit`` rows if given."""
    path = join(root, PROBE_DIR, PROBE_TEST)
    if not exists(fs, path):
        raise FileNotFoundError(f"no probe split at {path}: run make_blocks.py first")
    table = pq.read_table(path, filesystem=fs)
    if limit is not None:
        table = table.slice(0, limit)
    return decode_column(table), table.column("label").to_numpy().astype(np.int64)


def make_blocks(cfg: DistrainerConfig) -> list[BlockRef]:
    """Write the blocks, the probe split and the log for ``cfg``. Idempotent per store."""
    t = cfg.train
    rows = int(t.get("rows_per_block", 256))
    fs, root = cfg.store_fs()
    existing = BlockLog(fs, root)
    if existing.exists():
        if not existing.committed_seqs():
            raise RuntimeError(
                f"the log under {root} exists but has no segments (an interrupted make_blocks?): "
                "delete the store, or run train.py --rebuild"
            )
        seen: dict[str, BlockRef] = {}
        for seg in existing.segments():
            for b in seg.blocks:
                seen.setdefault(b.block_id, b)
        return [seen[k] for k in sorted(seen)]
    images, labels = load_images(cfg, "train")
    N = len(images)
    if rows <= 0:
        raise ValueError(f"rows_per_block must be positive, got {rows}")
    if N % rows:
        raise ValueError(f"{N} training images are not a multiple of rows_per_block={rows}")
    n_blocks = N // rows
    if n_blocks % cfg.log.W:
        raise ValueError(
            f"{n_blocks} blocks of {rows} rows do not fill whole segments of W={cfg.log.W}: "
            f"choose n_items as a multiple of {rows * cfg.log.W}"
        )
    test_images, test_labels = load_images(cfg, "test")
    write_probe_split(fs, root, test_images, test_labels)
    order = np.random.default_rng(cfg.seed).permutation(N)
    log = BlockLog.create(fs, root, W=cfg.log.W, seed=cfg.seed)
    dataset = str(t.get("dataset", "cifar10"))
    refs: list[BlockRef] = []
    for i in range(n_blocks):
        idx = order[i * rows : (i + 1) * rows]
        table = images_table(images[idx], labels[idx], idx)
        refs.append(write_block(fs, root, f"b{i:05d}", table, meta={"dataset": dataset}))
    BatchWriter(log, refs, passes=cfg.log.passes, tail="error").run()
    return refs


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--set", action="append", default=[], metavar="KEY=VALUE")
    args = ap.parse_args()
    cfg = load_config(args.config, overrides=args.set)
    refs = make_blocks(cfg)
    fs, root = cfg.store_fs()
    log = BlockLog.open(fs, root)
    print(f"{len(refs)} blocks and {len(log.committed_seqs())} segments under {root}")


if __name__ == "__main__":
    main()
