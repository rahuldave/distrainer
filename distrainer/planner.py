"""distrainer.planner: pure functions that deal segment positions to ranks and locate a resume.

Spec section 2.2 (assignment rule, resume rule) and section 4. No I/O here.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from distrainer.ledger import Ledger

if TYPE_CHECKING:
    from distrainer.block import BlockRef
    from distrainer.log import Segment


def steps_per_segment(W: int, world_size: int) -> int:
    if world_size <= 0:
        raise ValueError(f"world_size must be positive, got {world_size}")
    if W % world_size != 0:
        raise ValueError(f"W={W} must be a multiple of world_size={world_size}")
    return W // world_size


def lane(
    segment: Segment, rank: int, world_size: int, start_step: int = 0
) -> list[tuple[int, BlockRef]]:
    """``(global position, BlockRef)`` pairs rank ``rank`` consumes in ``segment`` from
    ``start_step``.

    Position ``p`` (0-based within the segment) goes to rank ``p % world_size`` at step
    ``p // world_size``; global position = ``segment.seq * W + p``.
    """
    W = segment.W
    n_steps = steps_per_segment(W, world_size)
    if not 0 <= rank < world_size:
        raise ValueError(f"rank {rank} outside [0, {world_size})")
    if not 0 <= start_step <= n_steps:
        raise ValueError(f"start_step {start_step} outside [0, {n_steps}]")
    base = segment.seq * W
    return [
        (base + step * world_size + rank, segment.blocks[step * world_size + rank])
        for step in range(start_step, n_steps)
    ]


def resume_start(ledger: Ledger, world_size_now: int, W: int | None = None) -> tuple[int, int]:
    """``(segment, start_step)`` for the current world size after a checkpoint.

    ``done = cursor * world_size_old`` positions of ``ledger.segment`` are complete. The start
    step is ``done // world_size_now``: the resume rounds *down* to the last position that is a
    multiple of the new world size and replays the remainder (at most ``world_size_now - 1``
    blocks). If ``W`` is given and the segment was completed, the next segment starts at step 0.
    """
    if world_size_now <= 0:
        raise ValueError(f"world_size_now must be positive, got {world_size_now}")
    done = ledger.done_positions()
    if W is not None:
        steps_per_segment(W, world_size_now)
        if done > W:
            raise ValueError(f"ledger says {done} positions done but W={W}")
        if done == W:
            return ledger.segment + 1, 0
    return ledger.segment, done // world_size_now


def replayed_positions(ledger: Ledger, world_size_now: int) -> int:
    """How many already-trained positions the resume rule trains again."""
    _, start = resume_start(ledger, world_size_now)
    return ledger.done_positions() - start * world_size_now
