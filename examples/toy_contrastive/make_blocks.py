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
    anchors: np.ndarray | None = None,
) -> dict[str, np.ndarray]:
    """One positive and k hard negatives per anchor (every item by default); returns columnar
    arrays aligned with ``anchors`` (row j belongs to anchor ``anchors[j]``)."""
    N = len(items)
    idx = np.arange(N) if anchors is None else np.asarray(anchors)
    by_cluster: dict[int, np.ndarray] = {
        c: np.flatnonzero(cluster == c) for c in np.unique(cluster)
    }
    others: dict[int, np.ndarray] = {}  # fallback pool per cluster: every item of another cluster
    pos = np.empty(len(idx), dtype=np.int64)
    negs = np.empty((len(idx), k), dtype=np.int64)
    for row, i in enumerate(idx):
        c = int(cluster[i])
        members = by_cluster[c]
        pos[row] = i if len(members) == 1 else rng.choice(members[members != i])
        for j, other in enumerate(near[c]):
            pool = by_cluster.get(int(other))
            if pool is None or not len(pool):  # the neighbour cluster has no items
                pool = others.setdefault(c, np.flatnonzero(cluster != c))
            negs[row, j] = rng.choice(pool) if len(pool) else rng.integers(0, N)
    return {"positive": pos, "negatives": negs}


def make_blocks(cfg: DistrainerConfig) -> list[BlockRef]:
    """Build all blocks with Ray Data and write the log. Idempotent per store.

    With the ``remine`` hook configured the log is streamed instead: only the first
    ``initial_segments`` segments are written and the log is left open for the hook
    (``remine.initial_log``); Ray Data is not needed for that.
    """
    t = cfg.train
    B, k = int(t.get("anchors_per_block", 32)), int(t.get("hard_negatives", 4))
    fs, root = cfg.store_fs()
    existing = BlockLog(fs, root)
    if existing.exists():
        seen: dict[str, BlockRef] = {}
        for seg in existing.segments():
            for b in seg.blocks:
                seen.setdefault(b.block_id, b)
        return [seen[k] for k in sorted(seen)]
    if "remine" in cfg.hooks:
        from examples.toy_contrastive.remine import initial_log

        return initial_log(cfg)
    import ray
    import ray.data

    items, cluster, centroids = make_corpus(cfg)
    N, d = items.shape
    if N % B:
        raise ValueError(f"n_items={N} must be a multiple of anchors_per_block={B}")
    rng = np.random.default_rng(cfg.seed + 1)
    mined = mine_rows(items, cluster, hard_negative_clusters(centroids, k), k, rng)
    order = rng.permutation(N)
    batch_id = np.empty(N, dtype=np.int64)
    batch_id[order] = np.arange(N) // B  # B random anchors per block

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
        return pa.table(
            {"block_id": [ref.block_id], "locator": [ref.locator], "num_rows": [ref.num_rows]}
        )

    rows_out = ds.groupby("batch_id").map_groups(write_group, batch_format="pyarrow").take_all()
    refs = sorted(
        (
            BlockRef(r["block_id"], r["locator"], int(r["num_rows"]), {"anchors": B})
            for r in rows_out
        ),
        key=lambda ref: ref.block_id,
    )
    BatchWriter(log, refs, passes=cfg.log.passes, tail="error").run()
    return refs


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    cfg = load_config(ap.parse_args().config)
    init_ray(cfg)
    refs = make_blocks(cfg)
    fs, root = cfg.store_fs()
    n_seg = len(BlockLog.open(fs, root).committed_seqs())
    print(f"{len(refs)} blocks, {n_seg} segments under {root}")


if __name__ == "__main__":
    main()
