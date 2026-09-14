"""toy_contrastive: a two-layer MLP encoder and InfoNCE over in-block hard negatives.

With ``all_gather`` enabled the positives and negatives of every rank are gathered so each anchor
also sees the other ranks' candidates (the loss-side collective of spec section 8).
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


class Encoder(torch.nn.Module):
    def __init__(self, d_in: int, d_hidden: int = 64, d_out: int = 16):
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Linear(d_in, d_hidden), torch.nn.ReLU(), torch.nn.Linear(d_hidden, d_out)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.normalize(self.net(x), dim=-1)


def info_nce(
    anchor: torch.Tensor,
    positive: torch.Tensor,
    negatives: torch.Tensor,
    temperature: float = 0.1,
    all_gather: bool = False,
) -> torch.Tensor:
    """``anchor [B, e]``, ``positive [B, e]``, ``negatives [B, k, e]``: cross-entropy of the
    positive against the k hard negatives plus the other anchors' positives in the batch."""
    if all_gather and torch.distributed.is_available() and torch.distributed.is_initialized():
        world = torch.distributed.get_world_size()
        gathered = [torch.zeros_like(positive) for _ in range(world)]
        torch.distributed.all_gather(gathered, positive.detach())
        rank = torch.distributed.get_rank()
        gathered[rank] = positive  # keep the local gradient path
        pool = torch.cat(gathered, dim=0)  # [B*world, e]
    else:
        pool = positive
    pos_sim = (anchor * positive).sum(-1, keepdim=True)  # [B, 1]
    neg_sim = torch.einsum("be,bke->bk", anchor, negatives)  # [B, k]
    pool_sim = anchor @ pool.T  # [B, B*world]: in-batch negatives (includes own positive)
    logits = torch.cat([pos_sim, neg_sim, pool_sim], dim=1) / temperature
    # the positive column is index 0; own positive also appears in pool_sim, mask it out
    own = torch.arange(anchor.shape[0], device=anchor.device)
    offset = 1 + negatives.shape[1]
    if all_gather and torch.distributed.is_initialized():
        own = own + torch.distributed.get_rank() * anchor.shape[0]
    logits[torch.arange(anchor.shape[0]), offset + own] = float("-inf")
    target = torch.zeros(anchor.shape[0], dtype=torch.long, device=anchor.device)
    return F.cross_entropy(logits, target)
