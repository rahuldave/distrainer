"""image_contrastive.model: a CIFAR-style ResNet encoder, batched SimCLR augmentations, NT-Xent.

The encoder is ResNet-18's topology with the CIFAR stem (a 3x3 convolution, no max-pool) that
SimCLR uses for 32x32 images, with ``width`` and ``layers`` as knobs so the laptop configs run a
tiny one (``width: 16, layers: [1, 1, 1, 1]``) and the GPU config the real thing (``width: 64,
layers: [2, 2, 2, 2]``). A two-layer projection head follows, and the loss is computed on the
L2-normalised projection. ``augment`` draws SimCLR's random resized crop, flip, colour jitter and
grayscale *per sample* as batched tensor ops on the device (one ``grid_sample``, a few
broadcasts), so two views of a 256-row block cost a few milliseconds on a GPU. NT-Xent is the toy
example's ``info_nce`` with no mined negatives and ``all_gather`` for the other ranks' positives.
"""

from __future__ import annotations

import math
from collections.abc import Sequence

import torch
import torch.nn.functional as F
from torch import nn

from examples.image_contrastive.data import CIFAR10_MEAN, CIFAR10_STD
from examples.toy_contrastive.model import info_nce


class BasicBlock(nn.Module):
    def __init__(self, c_in: int, c_out: int, stride: int):
        super().__init__()
        self.conv1 = nn.Conv2d(c_in, c_out, 3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(c_out)
        self.conv2 = nn.Conv2d(c_out, c_out, 3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(c_out)
        self.shortcut: nn.Module = nn.Identity()
        if stride != 1 or c_in != c_out:
            self.shortcut = nn.Sequential(
                nn.Conv2d(c_in, c_out, 1, stride=stride, bias=False), nn.BatchNorm2d(c_out)
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return F.relu(out + self.shortcut(x))


class ResNet(nn.Module):
    """``width`` channels in the first stage, doubling per stage; ``layers`` blocks per stage.
    ``(64, [2, 2, 2, 2])`` is ResNet-18 with the CIFAR stem; features are ``8 * width`` wide."""

    def __init__(self, width: int = 64, layers: Sequence[int] = (2, 2, 2, 2), in_channels: int = 3):
        super().__init__()
        if width <= 0 or len(layers) != 4 or any(n <= 0 for n in layers):
            raise ValueError(f"width must be positive and layers four positive counts: {layers}")
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, width, 3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(width),
            nn.ReLU(inplace=True),
        )
        stages: list[nn.Module] = []
        c_in = width
        for i, n in enumerate(layers):
            c_out, stride = width * 2**i, 1 if i == 0 else 2
            blocks = [BasicBlock(c_in, c_out, stride)] + [
                BasicBlock(c_out, c_out, 1) for _ in range(n - 1)
            ]
            stages.append(nn.Sequential(*blocks))
            c_in = c_out
        self.stages = nn.Sequential(*stages)
        self.feature_dim = c_in

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.flatten(F.adaptive_avg_pool2d(self.stages(self.stem(x)), 1), 1)


class Encoder(nn.Module):
    """The backbone plus a projection head; ``forward`` gives the normalised projection (the
    loss space), ``features`` the backbone output (the probe space)."""

    def __init__(
        self,
        width: int = 64,
        layers: Sequence[int] = (2, 2, 2, 2),
        embed: int = 128,
        proj_hidden: int | None = None,
    ):
        super().__init__()
        self.backbone = ResNet(width, layers)
        d = self.backbone.feature_dim
        hidden = proj_hidden or d
        self.head = nn.Sequential(
            nn.Linear(d, hidden), nn.ReLU(inplace=True), nn.Linear(hidden, embed)
        )

    def features(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.normalize(self.head(self.backbone(x)), dim=-1)


def normalize(x: torch.Tensor) -> torch.Tensor:
    """``uint8 [B, H, W, 3]`` to the float ``[B, 3, H, W]`` the encoder eats (CIFAR statistics)."""
    x = x.permute(0, 3, 1, 2).to(torch.float32) / 255.0
    mean = torch.tensor(CIFAR10_MEAN, device=x.device).view(1, 3, 1, 1)
    std = torch.tensor(CIFAR10_STD, device=x.device).view(1, 3, 1, 1)
    return (x - mean) / std


def _rand(n: int, gen: torch.Generator | None, device: torch.device) -> torch.Tensor:
    # random parameters are drawn on the CPU (a CPU generator serves every device) and moved
    return torch.rand(n, generator=gen).to(device)


def augment(
    x: torch.Tensor,
    gen: torch.Generator | None = None,
    min_scale: float = 0.2,
    jitter: float = 0.4,
    p_jitter: float = 0.8,
    p_gray: float = 0.2,
) -> torch.Tensor:
    """One SimCLR view of every image: a random resized crop (area in ``[min_scale, 1]``, aspect
    in ``[3/4, 4/3]``) resampled to the input size, a horizontal flip with probability one half,
    brightness, contrast and saturation jitter of strength ``jitter`` with probability
    ``p_jitter``, grayscale with probability ``p_gray``; then CIFAR normalisation.

    ``x`` is ``uint8 [B, H, W, 3]`` on the target device; every parameter is drawn per sample
    from ``gen`` (a CPU generator: seed it per step for reproducible views).
    """
    B, H, W, _ = x.shape
    dev = x.device
    img = x.permute(0, 3, 1, 2).to(torch.float32) / 255.0  # [B, 3, H, W] in [0, 1]

    # crop: the window's half-extents (w, h) in normalised coordinates and its centre (cx, cy)
    area = min_scale + (1.0 - min_scale) * _rand(B, gen, dev)
    log_ratio = (math.log(4 / 3) - math.log(3 / 4)) * _rand(B, gen, dev) + math.log(3 / 4)
    ratio = torch.exp(log_ratio)
    w = torch.sqrt(area * ratio).clamp(max=1.0)
    h = torch.sqrt(area / ratio).clamp(max=1.0)
    cx = (2 * _rand(B, gen, dev) - 1) * (1 - w)
    cy = (2 * _rand(B, gen, dev) - 1) * (1 - h)
    flip = torch.where(_rand(B, gen, dev) < 0.5, -1.0, 1.0)
    theta = torch.zeros(B, 2, 3, device=dev)
    theta[:, 0, 0], theta[:, 0, 2] = w * flip, cx
    theta[:, 1, 1], theta[:, 1, 2] = h, cy
    grid = F.affine_grid(theta, [B, 3, H, W], align_corners=False)
    img = F.grid_sample(img, grid, mode="bilinear", padding_mode="reflection", align_corners=False)

    # colour jitter (brightness, contrast, saturation in a fixed order) on a random subset
    def factor() -> torch.Tensor:
        return (1.0 + jitter * (2 * _rand(B, gen, dev) - 1)).view(B, 1, 1, 1)

    gray_w = torch.tensor([0.299, 0.587, 0.114], device=dev).view(1, 3, 1, 1)
    jittered = (img * factor()).clamp_(0, 1)
    mean = (jittered * gray_w).sum(1, keepdim=True).mean((2, 3), keepdim=True)
    jittered = ((jittered - mean) * factor() + mean).clamp_(0, 1)
    gray = (jittered * gray_w).sum(1, keepdim=True)
    jittered = ((jittered - gray) * factor() + gray).clamp_(0, 1)
    use_jitter = (_rand(B, gen, dev) < p_jitter).view(B, 1, 1, 1)
    img = torch.where(use_jitter, jittered, img)

    use_gray = (_rand(B, gen, dev) < p_gray).view(B, 1, 1, 1)
    img = torch.where(use_gray, (img * gray_w).sum(1, keepdim=True).expand_as(img), img)

    mean_c = torch.tensor(CIFAR10_MEAN, device=dev).view(1, 3, 1, 1)
    std_c = torch.tensor(CIFAR10_STD, device=dev).view(1, 3, 1, 1)
    return (img - mean_c) / std_c


def nt_xent(
    z1: torch.Tensor, z2: torch.Tensor, temperature: float = 0.1, all_gather: bool = False
) -> torch.Tensor:
    """The symmetric NT-Xent of two normalised views ``[B, e]``: ``info_nce`` with no mined
    negatives (the other positives of the batch, and with ``all_gather`` of every rank, are the
    negatives), averaged over both directions."""
    empty = z1.new_zeros(z1.shape[0], 0, z1.shape[1])
    a = info_nce(z1, z2, empty, temperature=temperature, all_gather=all_gather)
    b = info_nce(z2, z1, empty, temperature=temperature, all_gather=all_gather)
    return 0.5 * (a + b)
