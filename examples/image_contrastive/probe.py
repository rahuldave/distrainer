"""image_contrastive.probe: a weighted k-nearest-neighbour accuracy as the run's sanity metric.

The encoder never sees a label; this asks whether its backbone features separate the classes
anyway. Training images (the first ``probe_train`` rows of the log, labels included) are the
memory bank, the held-out split (``probe/test.parquet``, the first ``probe_test`` rows) the
queries, and every query votes among its ``k`` nearest bank features by cosine similarity with
weights ``exp(sim / temperature)`` (the kNN monitor of instance-discrimination and SimCLR
reproductions). Chance is ``1 / classes``; an untrained encoder sits a little above it, a few
segments of SimCLR on CIFAR-10 well above.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pyarrow.fs as pafs
import torch
import torch.nn.functional as F

from distrainer.block import read_block
from distrainer.config import DistrainerConfig
from distrainer.log import BlockLog
from examples.image_contrastive.data import CIFAR10_CLASSES, decode_column
from examples.image_contrastive.make_blocks import read_probe_split
from examples.image_contrastive.model import Encoder, normalize


def embed(
    encoder: Encoder, images: np.ndarray, device: torch.device, batch: int = 512
) -> torch.Tensor:
    """L2-normalised backbone features ``[N, F]`` (eval mode, no gradients, left as it was)."""
    was_training = encoder.training
    encoder.eval()
    try:
        out = []
        with torch.no_grad():
            for i in range(0, len(images), batch):
                x = torch.from_numpy(np.ascontiguousarray(images[i : i + batch])).to(device)
                out.append(F.normalize(encoder.features(normalize(x)), dim=-1).cpu())
    finally:
        encoder.train(was_training)
    return torch.cat(out) if out else torch.zeros(0, encoder.backbone.feature_dim)


def knn_accuracy(
    bank: torch.Tensor,
    bank_labels: torch.Tensor,
    queries: torch.Tensor,
    query_labels: torch.Tensor,
    k: int = 20,
    classes: int = CIFAR10_CLASSES,
    temperature: float = 0.1,
    chunk: int = 1024,
) -> float:
    """Weighted kNN accuracy of ``queries [M, F]`` against ``bank [N, F]`` (both normalised)."""
    if len(bank) == 0 or len(queries) == 0:
        raise ValueError("the bank and the queries must be non-empty")
    if int(bank_labels.min()) < 0 or int(bank_labels.max()) >= classes:
        raise ValueError(
            f"bank labels must lie in [0, {classes}), got {bank_labels.unique().tolist()}"
        )
    bank_labels = bank_labels.to(bank.device)
    query_labels = query_labels.to(bank.device)
    k = min(k, len(bank))
    correct = 0
    for i in range(0, len(queries), chunk):
        q = queries[i : i + chunk].to(bank.device)
        sim, idx = (q @ bank.T).topk(k, dim=1)  # [m, k]
        weights = torch.exp(sim / temperature)
        votes = torch.zeros(len(q), classes, dtype=weights.dtype, device=weights.device)
        votes.scatter_add_(1, bank_labels[idx], weights)
        correct += int((votes.argmax(1) == query_labels[i : i + chunk]).sum())
    return correct / len(queries)


def bank_from_log(fs: pafs.FileSystem, root: str, limit: int) -> tuple[np.ndarray, np.ndarray]:
    """The first ``limit`` training images (and labels) in log order, from the oldest kept
    segment on (retention gc may have dropped earlier ones)."""
    log = BlockLog.open(fs, root)
    images, labels, n = [], [], 0
    for segment in log.segments():
        for ref in segment.blocks:
            if n >= limit:
                break
            table = read_block(fs, root, ref)
            images.append(decode_column(table))
            labels.append(table.column("label").to_numpy().astype(np.int64))
            n += table.num_rows
        if n >= limit:
            break
    if not images:
        raise ValueError(f"the log under {root} has no blocks to build a bank from")
    return np.concatenate(images)[:limit], np.concatenate(labels)[:limit]


def probe_device(cfg: DistrainerConfig) -> torch.device:
    name = str(cfg.train.get("probe_device", "auto"))
    if name == "auto":
        name = "cuda" if torch.cuda.is_available() else "cpu"
    return torch.device(name)


def run_probe(cfg: DistrainerConfig, encoder: Encoder) -> dict[str, Any]:
    """The kNN accuracy of ``encoder`` for the store of ``cfg``: ``knn_acc``, the sizes, chance."""
    t = cfg.train
    fs, root = cfg.store_fs()
    device = probe_device(cfg)
    n_train, n_test = int(t.get("probe_train", 5000)), int(t.get("probe_test", 2000))
    classes = int(t.get("classes", CIFAR10_CLASSES))
    bank_images, bank_labels = bank_from_log(fs, root, n_train)
    test_images, test_labels = read_probe_split(fs, root, limit=n_test)
    encoder = encoder.to(device)
    acc = knn_accuracy(
        embed(encoder, bank_images, device),
        torch.from_numpy(bank_labels),
        embed(encoder, test_images, device),
        torch.from_numpy(test_labels),
        k=int(t.get("probe_k", 20)),
        classes=classes,
    )
    return {
        "knn_acc": acc,
        "chance": 1.0 / classes,
        "probe_train": len(bank_images),
        "probe_test": len(test_images),
        "device": str(device),
    }
