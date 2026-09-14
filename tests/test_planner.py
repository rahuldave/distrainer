import math

import pytest
from conftest import make_segment
from hypothesis import given, settings
from hypothesis import strategies as st

from distrainer.ledger import Ledger
from distrainer.planner import lane, replayed_positions, resume_start, steps_per_segment


def test_worked_example_from_spec():
    # W=12, n=3: rank 0 gets positions 24,27,30,33 of segment 2 (blocks 0,3,6,9)
    seg = make_segment(2, 12)
    r0 = lane(seg, 0, 3)
    assert [p for p, _ in r0] == [24, 27, 30, 33]
    assert [b.block_id for _, b in r0] == ["b0000", "b0003", "b0006", "b0009"]
    assert [p for p, _ in lane(seg, 2, 3, start_step=2)] == [32, 35]
    assert lane(seg, 1, 3, start_step=4) == []


def test_resume_worked_example_and_resize():
    ledger = Ledger(segment=2, cursor=2, world_size=3)  # positions 24..29 done
    assert resume_start(ledger, 3) == (2, 2)
    assert resume_start(ledger, 2) == (2, 3)  # 6 done -> step 3 of 2
    assert resume_start(ledger, 4) == (2, 1)  # rounds down: 4 done, replays 2
    assert replayed_positions(ledger, 4) == 2
    assert resume_start(Ledger(), 4) == (0, 0)
    assert resume_start(Ledger(segment=2, cursor=4, world_size=3), 2, W=12) == (3, 0)


def test_argument_validation():
    seg = make_segment(0, 12)
    with pytest.raises(ValueError):
        steps_per_segment(12, 5)
    with pytest.raises(ValueError):
        lane(seg, 3, 3)
    with pytest.raises(ValueError):
        lane(seg, 0, 3, start_step=5)
    with pytest.raises(ValueError):
        resume_start(Ledger(segment=0, cursor=5, world_size=3), 3, W=12)
    with pytest.raises(ValueError):
        resume_start(Ledger(), 0)


@st.composite
def worlds(draw):
    n_old = draw(st.integers(1, 6))
    n_new = draw(st.integers(1, 6))
    W = math.lcm(n_old, n_new) * draw(st.integers(1, 4))
    cursor = draw(st.integers(0, W // n_old))
    seq = draw(st.integers(0, 5))
    return W, n_old, n_new, cursor, seq


@settings(max_examples=300)
@given(worlds())
def test_lanes_partition_the_segment(w):
    W, n, _, _, seq = w
    seg = make_segment(seq, W)
    seen = {}
    for rank in range(n):
        ln = lane(seg, rank, n)
        assert len(ln) == W // n
        assert [p for p, _ in ln] == sorted(p for p, _ in ln)
        for step, (p, ref) in enumerate(ln):
            local = p - seq * W
            assert local % n == rank and local // n == step
            assert ref is seg.blocks[local]
            assert p not in seen
            seen[p] = rank
    assert sorted(seen) == list(seg.positions())


@settings(max_examples=300)
@given(worlds())
def test_resume_rounds_down_and_covers_the_tail(w):
    W, n_old, n_new, cursor, seq = w
    ledger = Ledger(segment=seq, cursor=cursor, world_size=n_old)
    done = cursor * n_old
    s, start = resume_start(ledger, n_new, W=W)
    if done == W:
        assert (s, start) == (seq + 1, 0)
        return
    assert s == seq and start == done // n_new
    replay = done - start * n_new
    assert 0 <= replay < n_new
    assert replay == replayed_positions(ledger, n_new)
    seg = make_segment(seq, W)
    tail = sorted(p for r in range(n_new) for p, _ in lane(seg, r, n_new, start))
    assert tail == list(range(seq * W + start * n_new, (seq + 1) * W))
    # same world size never replays anything
    assert replayed_positions(ledger, n_old) == 0
