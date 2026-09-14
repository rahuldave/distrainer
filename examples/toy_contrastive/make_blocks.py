"""toy_contrastive: synthetic clustered items -> contrastive blocks with hard negatives -> log.

Spec section 8. ``N`` items with ``d`` features from ``C`` Gaussian clusters. Each row of a block
is one anchor with a positive (another item of the same cluster) and ``k`` hard negatives (items
from the ``k`` nearest *other* clusters by centroid distance). Blocks of ``B`` anchors are built
with Ray Data (``groupby("batch_id").map_groups``), then ``BatchWriter`` writes the log.
"""

from __future__ import annotations

import argparse

import numpy as np
import pyarrow as pa

from distrainer.block import BlockRef, write_block
from distrainer.config import DistrainerConfig, load_config
from distrainer.log import BlockLog
from distrainer.trainer import init_ray
from distrainer.writer import BatchWriter


def make_corpus(cfg: DistrainerConfig) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """``(items[N, d], cluster_of_item[N], centroids[C, d])`` from the ``train:`` section."""
    t = cfg.train
    N, d, C = int(t.get("n_items", 7680)), int(t.get("features", 32)), int(t.get("clusters", 64))
    rng = np.random.default_rng(cfg.seed)
    centroids = rng.normal(size=(C, d)).astype(np.float32) * 3.0
    cluster = rng.integers(0, C, size=N)
    items = (centroids[cluster] + rng.normal(size=(N, d))).astype(np.float32)
    return items, cluster, centroids


def hard_negative_clusters(centroids: np.ndarray, k: int) -> np.ndarray:
    """``[C, k]`` indices of the k nearest other clusters per cluster."""
    dist = np.linalg.norm(centroids[:, None, :] - centroids[None, :, :], axis=-1)
    np.fill_diagonal(dist, np.inf)
    return np.argsort(dist, axis=1)[:, :k]


def mine_rows(
    items: np.ndarray,
    cluster: np.ndarray,
    near: np.ndarray,
    k: int,
    rng: np.random.Generator,
) -> dict[str, np.ndarray]:
    """One positive and k hard negatives per item; returns columnar arrays (row i = anchor i)."""
    N = len(items)
    by_cluster: dict[int, np.ndarray] = {
        c: np.flatnonzero(cluster == c) for c in np.unique(cluster)
    }
    pos = np.empty(N, dtype=np.int64)
    negs = np.empty((N, k), dtype=np.int64)
    for i in range(N):
        c = int(cluster[i])
        members = by_cluster[c]
        pos[i] = i if len(members) == 1 else rng.choice(members[members != i])
        for j, other in enumerate(near[c]):
            pool = by_cluster.get(int(other))
            negs[i, j] = rng.choice(pool) if pool is not None and len(pool) else rng.integers(0, N)
    return {"positive": pos, "negatives": negs}


def make_blocks(cfg: DistrainerConfig) -> list[BlockRef]:
    """Build all blocks with Ray Data and write the log. Idempotent per store."""
    import ray
    import ray.data

    t = cfg.train
    B, k = int(t.get("anchors_per_block", 32)), int(t.get("hard_negatives", 4))
    items, cluster, centroids = make_corpus(cfg)
    N, d = items.shape
    if N % B:
        raise ValueError(f"n_items={N} must be a multiple of anchors_per_block={B}")
    rng = np.random.default_rng(cfg.seed + 1)
    mined = mine_rows(items, cluster, hard_negative_clusters(centroids, k), k, rng)
    order = rng.permutation(N)
    batch_id = np.empty(N, dtype=np.int64)
    batch_id[order] = np.arange(N) // B  # B random anchors per block

    fs, root = cfg.store_fs()
    log = BlockLog.create(fs, root, W=cfg.log.W, seed=cfg.seed)
    rows = {
        "item_id": np.arange(N),
        "batch_id": batch_id,
        "cluster": cluster,
        "anchor": list(items),
        "positive": list(items[mined["positive"]]),
        **{f"neg_{j}": list(items[mined["negatives"][:, j]]) for j in range(k)},
    }
    ds = ray.data.from_arrow(pa.table(rows))

    def write_group(group: pa.Table) -> pa.Table:
        bid = int(group.column("batch_id")[0].as_py())
        ref = write_block(fs, root, f"b{bid:05d}", group, meta={"anchors": group.num_rows})
        return pa.table({"block_id": [ref.block_id], "num_rows": [ref.num_rows]})

    written = ds.groupby("batch_id").map_groups(write_group, batch_format="pyarrow").to_arrow_refs()
    tables = [ray.get(r) for r in written]
    ids = sorted(bid for tbl in tables for bid in tbl.column("block_id").to_pylist())
    refs = [BlockRef(bid, f"blocks/{bid}.parquet", B, {"anchors": B}) for bid in ids]
    BatchWriter(log, refs, passes=cfg.log.passes, tail="error").run()
    return refs


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    cfg = load_config(ap.parse_args().config)
    init_ray(cfg)
    refs = make_blocks(cfg)
    fs, root = cfg.store_fs()
    print(
        f"wrote {len(refs)} blocks, {BlockLog.open(fs, root).last_seq() + 1} segments under {root}"
    )


if __name__ == "__main__":
    main()
