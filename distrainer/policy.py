"""distrainer.policy: which step boundaries become checkpoints.

Spec section 4. Every step boundary is a legal checkpoint; a policy chooses. Index-based policies
need no communication. ``TimeBudget`` is decided by rank 0 and broadcast so every rank calls
``ray.train.report`` with a checkpoint the same number of times. ``Any`` evaluates *all* its
members on every step (no short-circuit) so collective calls stay aligned across ranks.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any as _Any
from typing import Protocol, runtime_checkable


@dataclass(frozen=True)
class StepContext:
    """What a policy may look at after one step. ``step_in_segment`` counts completed steps."""

    position: int
    step_in_segment: int
    segment: int = 0
    pass_idx: int = 0
    segment_end: bool = False
    pass_end: bool = False
    elapsed_s: float = 0.0
    rank: int = 0
    world_size: int = 1


@runtime_checkable
class CheckpointPolicy(Protocol):
    def should_checkpoint(self, ctx: StepContext) -> bool: ...


class Never:
    def should_checkpoint(self, ctx: StepContext) -> bool:
        return False


class EveryKSteps:
    """Checkpoint when the number of completed steps in the segment is a multiple of ``k``."""

    def __init__(self, k: int):
        if k <= 0:
            raise ValueError(f"k must be positive, got {k}")
        self.k = k

    def should_checkpoint(self, ctx: StepContext) -> bool:
        return ctx.step_in_segment > 0 and ctx.step_in_segment % self.k == 0


class SegmentEnd:
    def should_checkpoint(self, ctx: StepContext) -> bool:
        return ctx.segment_end


class PassEnd:
    def should_checkpoint(self, ctx: StepContext) -> bool:
        return ctx.pass_end


Broadcast = Callable[[_Any], _Any]


def _ray_broadcast(value: _Any) -> _Any:
    from ray.train.collective import broadcast_from_rank_zero

    return broadcast_from_rank_zero(value)


class TimeBudget:
    """Checkpoint once at least ``seconds`` have passed since the last one.

    Evaluated every ``poll_every`` steps; rank 0's answer is broadcast to all ranks through
    ``broadcast`` (default ``ray.train.collective.broadcast_from_rank_zero``). Between polls the
    answer is ``False`` everywhere, so no rank calls the collective alone.
    """

    def __init__(self, seconds: float, poll_every: int = 1, broadcast: Broadcast | None = None):
        if seconds <= 0 or poll_every <= 0:
            raise ValueError("seconds and poll_every must be positive")
        self.seconds = seconds
        self.poll_every = poll_every
        self._broadcast = broadcast or _ray_broadcast
        self._last_checkpoint_s = 0.0
        self._steps_seen = 0

    def should_checkpoint(self, ctx: StepContext) -> bool:
        self._steps_seen += 1
        if self._steps_seen % self.poll_every != 0:
            return False
        local = ctx.elapsed_s - self._last_checkpoint_s >= self.seconds if ctx.rank == 0 else None
        decision = bool(self._broadcast(local) if ctx.world_size > 1 else local)
        if decision:
            self._last_checkpoint_s = ctx.elapsed_s
        return decision


class Any:
    """OR of several policies; every member is evaluated on every step."""

    def __init__(self, policies: Sequence[CheckpointPolicy]):
        if not policies:
            raise ValueError("Any needs at least one policy")
        self.policies = list(policies)

    def should_checkpoint(self, ctx: StepContext) -> bool:
        results = [p.should_checkpoint(ctx) for p in self.policies]
        return any(results)


def build_policy(cfg: dict[str, _Any], broadcast: Broadcast | None = None) -> CheckpointPolicy:
    """Policy from the ``checkpoint:`` config section (spec section 7).

    ``policy``: ``every_k`` | ``segment_end`` | ``pass_end`` | ``time`` | ``any`` | ``never``.
    ``any`` is the union of the configured pieces: ``every_k`` if set, ``segment_end``,
    ``pass_end``, and ``time`` if ``time_budget_s`` is set.
    """
    kind = str(cfg.get("policy", "any"))
    every_k = cfg.get("every_k")
    budget = cfg.get("time_budget_s")
    poll = int(cfg.get("time_poll_every", 1))
    if kind == "never":
        return Never()
    if kind == "every_k":
        if every_k is None:
            raise ValueError("policy every_k needs checkpoint.every_k")
        return EveryKSteps(int(every_k))
    if kind == "segment_end":
        return SegmentEnd()
    if kind == "pass_end":
        return PassEnd()
    if kind == "time":
        if budget is None:
            raise ValueError("policy time needs checkpoint.time_budget_s")
        return TimeBudget(float(budget), poll, broadcast)
    if kind == "any":
        members: list[CheckpointPolicy] = []
        if every_k is not None:
            members.append(EveryKSteps(int(every_k)))
        members.extend([SegmentEnd(), PassEnd()])
        if budget is not None:
            members.append(TimeBudget(float(budget), poll, broadcast))
        return Any(members)
    raise ValueError(f"unknown checkpoint policy {kind!r}")
