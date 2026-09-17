"""distrainer.parallel without a process group: the kinds' factories and the DiLoCo arithmetic."""

import pytest
import torch

from distrainer.config import DistrainerConfig
from distrainer.parallel import DiLoCo, LocalSGD, build_sync


def small() -> torch.nn.Linear:
    torch.manual_seed(0)
    return torch.nn.Linear(2, 1)


def params(m: torch.nn.Module) -> list[torch.Tensor]:
    return [p.detach().clone() for p in m.parameters()]


def cfg(**parallel) -> DistrainerConfig:
    return DistrainerConfig.from_dict({"parallel": parallel} if parallel else {})


def test_build_sync_per_kind():
    assert build_sync(cfg(), small()) is None  # ddp
    assert build_sync(cfg(kind="none"), small()) is None
    assert isinstance(build_sync(cfg(kind="local_sgd"), small()), LocalSGD)
    d = build_sync(cfg(kind="diloco", outer_lr=0.5, outer_momentum=0.0), small())
    assert isinstance(d, DiLoCo)
    assert d.outer.param_groups[0]["lr"] == 0.5 and d.outer.param_groups[0]["momentum"] == 0.0


def test_local_sgd_without_a_group_is_the_identity_and_has_no_state():
    m = small()
    before = params(m)
    LocalSGD().on_segment_sync(m, None, None)
    assert all(torch.equal(a, b) for a, b in zip(before, m.parameters(), strict=True))
    assert LocalSGD().state_dict() == {}


def test_diloco_outer_step_without_a_group():
    m = small()
    sync = DiLoCo(m, outer_lr=1.0, outer_momentum=0.0)
    anchor0 = [a.clone() for a in sync.anchor]
    with torch.no_grad():
        for p in m.parameters():
            p.add_(1.0)  # the inner steps moved every parameter by +1
    sync.on_segment_sync(m, None, None)
    # outer_lr 1 and no momentum: the anchor moves by the change and the model holds the anchor
    for a0, a, p in zip(anchor0, sync.anchor, m.parameters(), strict=True):
        assert torch.allclose(a, a0 + 1.0) and torch.equal(p, a)
    # with Nesterov momentum the same change applied twice moves further the second time
    m2 = small()
    sync2 = DiLoCo(m2, outer_lr=1.0, outer_momentum=0.9)
    moves = []
    for _ in range(2):
        before = params(m2)
        with torch.no_grad():
            for p in m2.parameters():
                p.add_(1.0)
        sync2.on_segment_sync(m2, None, None)
        moves.append(float((next(m2.parameters()).detach() - before[0]).mean()))
    assert moves[1] > moves[0] > 1.0


def test_diloco_state_round_trips_and_reset_follows_the_model():
    m = small()
    sync = DiLoCo(m, outer_lr=0.7, outer_momentum=0.9)
    with torch.no_grad():
        for p in m.parameters():
            p.add_(0.5)
    sync.on_segment_sync(m, None, None)  # momentum buffers exist now
    state = sync.state_dict()
    assert set(state) == {"anchor", "outer"}
    other = DiLoCo(small(), outer_lr=0.7, outer_momentum=0.9)
    other.load_state_dict(state)
    assert all(torch.equal(a, b) for a, b in zip(sync.anchor, other.anchor, strict=True))
    assert len(other.outer.state) == len(sync.outer.state) > 0
    # a checkpoint without the sync's state (a DDP run resumed under diloco): the anchor becomes
    # the loaded parameters and the outer optimizer starts afresh
    fresh = small()
    with torch.no_grad():
        for p in fresh.parameters():
            p.fill_(3.0)
    other.reset(fresh)
    assert all(torch.equal(a, p) for a, p in zip(other.anchor, fresh.parameters(), strict=True))
    assert len(other.outer.state) == 0


def test_diloco_rejects_bad_outer_settings():
    with pytest.raises(ValueError):
        DiLoCo(small(), outer_lr=0.0)
    with pytest.raises(ValueError):
        DiLoCo(small(), outer_momentum=1.0)
