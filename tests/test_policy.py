import pytest

from distrainer.policy import (
    Any,
    EveryKSteps,
    Never,
    PassEnd,
    SegmentEnd,
    StepContext,
    TimeBudget,
    build_policy,
)


def ctx(step, **kw):
    return StepContext(position=step, step_in_segment=step, **kw)


def test_index_based_policies():
    every2 = EveryKSteps(2)
    assert [every2.should_checkpoint(ctx(s)) for s in range(1, 5)] == [False, True, False, True]
    assert not every2.should_checkpoint(ctx(0))
    assert SegmentEnd().should_checkpoint(ctx(4, segment_end=True))
    assert not SegmentEnd().should_checkpoint(ctx(4))
    assert PassEnd().should_checkpoint(ctx(4, pass_end=True))
    assert not Never().should_checkpoint(ctx(4, segment_end=True, pass_end=True))
    with pytest.raises(ValueError):
        EveryKSteps(0)


def test_any_evaluates_every_member_without_short_circuit():
    calls = []

    class Spy:
        def __init__(self, answer):
            self.answer = answer

        def should_checkpoint(self, c):
            calls.append(self.answer)
            return self.answer

    policy = Any([Spy(True), Spy(False), Spy(True)])
    assert policy.should_checkpoint(ctx(1))
    assert calls == [True, False, True]
    assert not Any([Spy(False)]).should_checkpoint(ctx(1))
    with pytest.raises(ValueError):
        Any([])


def test_time_budget_polls_and_broadcasts_rank_zero_decision():
    broadcasts = []

    def fake_broadcast(v):
        broadcasts.append(v)
        return True if v is None else v  # non-zero ranks receive rank 0's answer

    tb = TimeBudget(seconds=5, poll_every=2, broadcast=fake_broadcast)
    # rank 0, world 2: step 1 not polled, step 2 polled (elapsed 3 < 5) -> False
    assert not tb.should_checkpoint(ctx(1, elapsed_s=1.0, rank=0, world_size=2))
    assert not tb.should_checkpoint(ctx(2, elapsed_s=3.0, rank=0, world_size=2))
    assert broadcasts == [False]
    assert not tb.should_checkpoint(ctx(3, elapsed_s=6.0, rank=0, world_size=2))  # not a poll step
    assert tb.should_checkpoint(ctx(4, elapsed_s=6.0, rank=0, world_size=2))
    assert broadcasts == [False, True]
    # timer resets: 6 + 4 = 10 < 11 -> False; 12 >= 11 -> True
    assert not tb.should_checkpoint(ctx(6, elapsed_s=10.0, rank=0, world_size=2))
    assert tb.should_checkpoint(ctx(8, elapsed_s=12.0, rank=0, world_size=2))
    # a non-zero rank passes None into the broadcast and takes whatever comes back
    tb1 = TimeBudget(seconds=5, poll_every=1, broadcast=fake_broadcast)
    assert tb1.should_checkpoint(ctx(1, elapsed_s=0.0, rank=1, world_size=2)) is True
    assert broadcasts[-1] is None
    # single worker: decided locally, no broadcast call
    n = len(broadcasts)
    assert TimeBudget(seconds=1, broadcast=fake_broadcast).should_checkpoint(ctx(1, elapsed_s=2.0))
    assert len(broadcasts) == n
    with pytest.raises(ValueError):
        TimeBudget(0)


def test_build_policy_from_config():
    assert isinstance(build_policy({"policy": "every_k", "every_k": 4}), EveryKSteps)
    assert isinstance(build_policy({"policy": "segment_end"}), SegmentEnd)
    assert isinstance(build_policy({"policy": "pass_end"}), PassEnd)
    assert isinstance(build_policy({"policy": "never"}), Never)
    tb = build_policy(
        {"policy": "time", "time_budget_s": 5, "time_poll_every": 3}, broadcast=lambda v: v
    )
    assert isinstance(tb, TimeBudget) and tb.poll_every == 3
    anyp = build_policy({"policy": "any", "every_k": 4, "time_budget_s": None})
    assert isinstance(anyp, Any)
    assert [type(p).__name__ for p in anyp.policies] == ["EveryKSteps", "SegmentEnd", "PassEnd"]
    assert [type(p).__name__ for p in build_policy({}).policies] == ["SegmentEnd", "PassEnd"]
    with pytest.raises(ValueError):
        build_policy({"policy": "every_k"})
    with pytest.raises(ValueError):
        build_policy({"policy": "time"})
    with pytest.raises(ValueError):
        build_policy({"policy": "weekly"})


def test_any_resets_time_budget_when_a_sibling_fires():
    tb = TimeBudget(seconds=5, broadcast=lambda v: v)
    policy = Any([SegmentEnd(), tb])
    assert policy.should_checkpoint(ctx(4, elapsed_s=4.0, segment_end=True))  # SegmentEnd fired
    assert not tb.should_checkpoint(ctx(5, elapsed_s=6.0))  # budget restarted at 4.0
    assert tb.should_checkpoint(ctx(6, elapsed_s=9.0))
    from distrainer.policy import notify_checkpoint

    notify_checkpoint(EveryKSteps(1), ctx(1))  # policies without the hook are fine
    notify_checkpoint(tb, ctx(7, elapsed_s=20.0))
    assert not tb.should_checkpoint(ctx(8, elapsed_s=24.0))
