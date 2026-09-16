"""image_contrastive.data: where the images come from, and the PNG codec for block rows.

Two sources, chosen by ``train.dataset``: ``cifar10`` (the torchvision download, cached under
``train.data_root``) and ``synthetic`` (class-coloured striped noise, no download; the unit tests
and a network-less smoke). Both give ``(images uint8 [N, H, W, 3], labels int64 [N])``; blocks
store every image as PNG bytes so a row is self-contained and the store stays small.
"""

from __future__ import annotations

import io

import numpy as np
import pyarrow as pa
from PIL import Image

from distrainer.config import DistrainerConfig

CIFAR10_CLASSES = 10
CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD = (0.2470, 0.2435, 0.2616)
IMAGE_COLUMN = "image"
LABEL_COLUMN = "label"

_PALETTE = np.array(
    [
        [220, 40, 40],
        [40, 200, 60],
        [50, 70, 230],
        [230, 210, 40],
        [200, 60, 200],
        [40, 210, 210],
        [240, 140, 30],
        [120, 80, 40],
        [150, 150, 150],
        [30, 30, 30],
    ],
    dtype=np.float32,
)


def synthetic_images(
    n: int, seed: int, size: int = 32, classes: int = CIFAR10_CLASSES, noise: float = 24.0
) -> tuple[np.ndarray, np.ndarray]:
    """``n`` images of ``size`` by ``size``: each class has a colour and a stripe frequency, every
    image its own phase, brightness and Gaussian noise, so a kNN in pixel space is well above
    chance and a small encoder learns the classes in a few steps."""
    if classes > len(_PALETTE):
        raise ValueError(f"synthetic supports up to {len(_PALETTE)} classes, got {classes}")
    rng = np.random.default_rng(seed)
    labels = rng.integers(0, classes, size=n)
    phase = rng.uniform(0, 2 * np.pi, size=n)
    gain = rng.uniform(0.6, 1.0, size=n)
    y = np.arange(size, dtype=np.float32)[None, :, None, None]  # [1, H, 1, 1]
    freq = (labels % 5 + 1).astype(np.float32)[:, None, None, None]
    stripes = 0.5 + 0.5 * np.sin(2 * np.pi * freq * y / size + phase[:, None, None, None])
    base = _PALETTE[labels][:, None, None, :] * gain[:, None, None, None]  # [n, 1, 1, 3]
    images = base * (0.5 + 0.5 * stripes) + rng.normal(scale=noise, size=(n, size, size, 3))
    return np.clip(images, 0, 255).astype(np.uint8), labels.astype(np.int64)


def load_cifar10(root: str, split: str) -> tuple[np.ndarray, np.ndarray]:
    """The CIFAR-10 ``train`` (50 000) or ``test`` (10 000) split, downloaded once into ``root``."""
    import os
    import sys

    from torchvision.datasets import CIFAR10

    if not sys.stdout.isatty():  # the download's progress bar is 36 KB of \r noise in a log file
        os.environ.setdefault("TQDM_DISABLE", "1")
    if split not in ("train", "test"):
        raise ValueError(f"split must be train or test, got {split!r}")
    ds = CIFAR10(root, train=split == "train", download=True)
    return np.asarray(ds.data, dtype=np.uint8), np.asarray(ds.targets, dtype=np.int64)


def load_images(cfg: DistrainerConfig, split: str) -> tuple[np.ndarray, np.ndarray]:
    """The ``train:`` section's dataset. ``n_items`` (train) and ``n_test`` (test) take a seeded
    random subset of CIFAR-10 (``null`` or 0 keeps the whole split) and size the synthetic draw."""
    t = cfg.train
    dataset = str(t.get("dataset", "cifar10"))
    n_key = "n_items" if split == "train" else "n_test"
    n = t.get(n_key)
    n = int(n) if n else 0
    seed = cfg.seed if split == "train" else cfg.seed + 1
    if dataset == "synthetic":
        return synthetic_images(
            n or 512,
            seed,
            size=int(t.get("image_size", 32)),
            classes=int(t.get("classes", CIFAR10_CLASSES)),
        )
    if dataset != "cifar10":
        raise ValueError(f"train.dataset must be cifar10 or synthetic, got {dataset!r}")
    images, labels = load_cifar10(str(t.get("data_root", ".harness/datasets")), split)
    if n > len(images):
        raise ValueError(
            f"train.{n_key}={n} exceeds the {len(images)} images of the cifar10 {split} split"
        )
    if n and n < len(images):
        idx = np.sort(np.random.default_rng(seed).permutation(len(images))[:n])
        images, labels = images[idx], labels[idx]
    return images, labels


def encode_png(image: np.ndarray) -> bytes:
    buf = io.BytesIO()
    Image.fromarray(np.ascontiguousarray(image, dtype=np.uint8)).save(buf, format="PNG")
    return buf.getvalue()


def decode_png(data: bytes) -> np.ndarray:
    with Image.open(io.BytesIO(data)) as im:
        return np.asarray(im.convert("RGB"), dtype=np.uint8)


def images_table(images: np.ndarray, labels: np.ndarray, item_ids: np.ndarray) -> pa.Table:
    """Rows ``(item_id, label, image)`` with the images as PNG bytes."""
    return pa.table(
        {
            "item_id": pa.array(np.asarray(item_ids, dtype=np.int64)),
            LABEL_COLUMN: pa.array(np.asarray(labels, dtype=np.int64)),
            IMAGE_COLUMN: pa.array([encode_png(im) for im in images], type=pa.binary()),
        }
    )


def decode_column(table: pa.Table, column: str = IMAGE_COLUMN) -> np.ndarray:
    """``uint8 [B, H, W, 3]`` from a block's PNG column (``[0, 0, 0, 3]`` for no rows)."""
    if table.num_rows == 0:
        return np.empty((0, 0, 0, 3), dtype=np.uint8)
    return np.stack([decode_png(b) for b in table.column(column).to_pylist()])
