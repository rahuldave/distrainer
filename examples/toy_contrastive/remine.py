"""toy_contrastive.remine: the re-mining segment hook (spec section 8, scenario S6).

At every segment end rank 0 embeds the whole corpus with the current encoder, recomputes each
cluster's nearest other clusters *in embedding space*, mines a positive and ``k`` hard negatives
for the anchors of the next segment, writes those ``W`` blocks and appends them as segment
``seq + 1``. The log is therefore produced in streaming mode: ``make_blocks.py`` writes only the
first ``initial_segments`` segments (mined in input space, no ``_END``) and the hook writes
``_END`` once ``segments`` segments exist.

Config (``hooks.remine``): ``entry: examples.toy_contrastive.remine:RemineHook``, ``segments``
(segments in the finished log), ``initial_segments`` (written by ``make_blocks.py``, default 1).
The corpus is rebuilt from ``seed`` with ``make_corpus``, so nothing rides in the checkpoint, and
the anchors of segment ``seq`` are a fixed slice of a per-pass permutation of the corpus:
``pass_idx`` advances when the slices wrap around the corpus.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pyarrow as pa
import torch

from distrainer.block import BlockRef, write_block
from distrainer.config import DistrainerConfig
from distrainer.hooks import hook_specs
from distrainer.ledger import Ledger
from distrainer.log import BlockLog, Segment
from examples.toy_contrastive.make_blocks import hard_negative_clusters, make_corpus, mine_rows

HOOK_NAME = "remine"


def remine_args(cfg: DistrainerConfig) -> dict[str, Any] | None:
    """The ``hooks.remine`` arguments if the hook is configured, else ``None``."""
    for name, _entry, args in hook_specs(cfg.hooks):
        if name == HOOK_NAME:
            return args
    return None


def segment_anchors(n_items: int, per_segment: int, seq: int, seed: int) -> tuple[int, np.ndarray]:
    """``(pass_idx, item indices)`` of the anchors segment ``seq`` covers.

    Segments take consecutive slices of ``per_segment`` items from a per-pass permutation of
    the corpus (``default_rng([seed, pass])``); a slice that runs past the end continues into
    the next pass's permutation. ``pass_idx`` is the pass of the slice's first item.
    """
    if n_items <= 0 or per_segment <= 0 or seq < 0:
        raise ValueError("n_items and per_segment must be positive, seq non-negative")
    start = seq * per_segment
    parts: list[np.ndarray] = []
    pos, taken = start, 0
    while taken < per_segment:
        p, offset = divmod(pos, n_items)
        perm = np.random.default_rng([seed, p]).permutation(n_items)
        take = min(per_segment - taken, n_items - offset)
        parts.append(perm[offset : offset + take])
        pos, taken = pos + take, taken + take
    return start // n_items, np.concatenate(parts)


def embedded_neighbours(
    model: torch.nn.Module, items: np.ndarray, cluster: np.ndarray, n_clusters: int, k: int
) -> np.ndarray:
    """``[C, k]`` nearest other clusters per cluster, by centroid distance in the encoder's
    output space (the model is run in eval mode without gradients and left as it was)."""
    was_training = model.training
    model.eval()
    try:
        with torch.no_grad():
            device = next(model.parameters()).device
            z = model(torch.from_numpy(items).to(device)).cpu().numpy()
    finally:
        model.train(was_training)
    # embeddings are L2-normalised, so a zero centroid for a cluster without items would be
    # nearer to everything than real centroids are to each other: park absent clusters far away
    cent = np.full((n_clusters, z.shape[1]), 1e6, dtype=np.float32)
    for c in range(n_clusters):
        members = cluster == c
        if members.any():
            cent[c] = z[members].mean(axis=0)
    return hard_negative_clusters(cent, k)


class RemineHook:
    """Rank 0's writer for the toy contrastive log (``SegmentHook``)."""

    def __init__(self, config: DistrainerConfig, segments: int, initial_segments: int = 1):
        if segments <= 0 or initial_segments <= 0 or initial_segments > segments:
            raise ValueError(
                f"remine needs 0 < initial_segments <= segments, got {initial_segments}, {segments}"
            )
        self.cfg = config
        self.segments = int(segments)
        self.initial_segments = int(initial_segments)
        t = config.train
        self.B = int(t.get("anchors_per_block", 32))
        self.k = int(t.get("hard_negatives", 4))
        self.items, self.cluster, self.centroids = make_corpus(config)
        if len(self.items) % self.per_segment:
            raise ValueError(
                f"n_items={len(self.items)} must be a multiple of W*anchors_per_block="
                f"{self.per_segment} so a segment never straddles two passes"
            )
        self.mined: list[int] = []  # segments this instance appended

    @property
    def per_segment(self) -> int:
        return self.cfg.log.W * self.B

    def write_segment(
        self,
        log: BlockLog,
        seq: int,
        near: np.ndarray,
        rng: np.random.Generator,
        meta: dict[str, Any] | None = None,
        attempt: int = 0,
    ) -> Segment:
        """Mine and write the ``W`` blocks of segment ``seq`` with the neighbour table ``near``
        and append them. ``seq`` must be the log's next sequence number; block ids carry the
        attempt so a stale rank 0 of an earlier attempt can never overwrite committed blocks."""
        if log.W != self.cfg.log.W:
            raise ValueError(f"log W={log.W} differs from config log.W={self.cfg.log.W}")
        last = log.last_seq()
        expected = 0 if last is None else last + 1
        if seq != expected:
            raise RuntimeError(f"segment {seq} is not the next one: the log ends at {last}")
        pass_idx, anchors = segment_anchors(len(self.items), self.per_segment, seq, self.cfg.seed)
        mined = mine_rows(self.items, self.cluster, near, self.k, rng, anchors=anchors)
        refs: list[BlockRef] = []
        for i in range(log.W):
            sl = slice(i * self.B, (i + 1) * self.B)
            idx = anchors[sl]
            table = pa.table(
                {
                    "item_id": idx,
                    "cluster": self.cluster[idx],
                    "anchor": list(self.items[idx]),
                    "positive": list(self.items[mined["positive"][sl]]),
                    **{
                        f"neg_{j}": list(self.items[mined["negatives"][sl, j]])
                        for j in range(self.k)
                    },
                }
            )
            refs.append(
                write_block(
                    log.fs,
                    log.root,
                    f"s{seq:06d}a{attempt:02d}b{i:05d}",
                    table,
                    meta={"anchors": self.B},
                )
            )
        segment = log.append(refs, pass_idx=pass_idx, meta={"writer": "remine", **(meta or {})})
        self.mined.append(segment.seq)
        return segment

    def on_segment_end(self, model: Any, ledger: Ledger, log: BlockLog, ctx: Any) -> None:
        nxt = ledger.segment + 1
        if log.ended() or log.has_segment(nxt):
            return  # a restarted attempt re-runs segment ends the previous one already served
        if nxt >= self.segments:
            log.end()
            return
        near = embedded_neighbours(model, self.items, self.cluster, len(self.centroids), self.k)
        rng = np.random.default_rng([self.cfg.seed, nxt, ledger.run_attempt])
        self.write_segment(
            log,
            nxt,
            near,
            rng,
            {"space": "embedding", "mined_after_segment": ledger.segment},
            attempt=ledger.run_attempt,
        )


def initial_log(cfg: DistrainerConfig) -> list[BlockRef]:
    """Streaming-mode ``make_blocks``: create the log and write the first ``initial_segments``
    segments mined in input space (centroid distance), without ``_END``; the hook does the rest."""
    args = remine_args(cfg)
    if args is None:
        raise ValueError("initial_log needs hooks.remine in the config")
    hook = RemineHook(cfg, **args)
    fs, root = cfg.store_fs()
    log = BlockLog.create(fs, root, W=cfg.log.W, seed=cfg.seed)
    near = hard_negative_clusters(hook.centroids, hook.k)
    refs: list[BlockRef] = []
    for seq in range(hook.initial_segments):
        seg = hook.write_segment(
            log, seq, near, np.random.default_rng([cfg.seed, seq]), {"space": "input"}
        )
        refs += seg.blocks
    return refs
