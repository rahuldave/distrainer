"""distrainer.parallel: how the replicas agree, beyond DDP (spec section 5).

``parallel.kind`` in the config chooses the agreement. ``ddp`` and ``none`` are Ray Train's
``prepare_model`` wraps. ``fsdp`` is :func:`wrap_fsdp`, FSDP2's ``fully_shard`` over a mesh of
the ranks (the parameters become DTensors, one shard per rank; every rank then writes its own
checkpoint shard, see ``CheckpointIO``). ``local_sgd`` and ``diloco`` train every rank
alone for a segment (no wrap, no collective per step) and agree at the segment end, on every
rank, through a :class:`SegmentSync`: the loop calls ``on_segment_sync(model, optimizer, info)``
after the last step of a segment, before the policy's checkpoint (so a segment-end checkpoint
holds the synced weights) and before rank 0's writer hooks. ``H``, the number of local steps
between two syncs, is therefore ``W / n``: the segment is the sync unit, and a resize at a
segment end hands the new ranks the synced weights like any other resume.

A sync's state (nothing for local SGD; the anchor and the outer optimizer for DiLoCo) is saved
beside the ledger by ``CheckpointIO`` and restored on every rank; a checkpoint without it (a run
under another kind) resets the sync to the loaded parameters. A segment-end checkpoint resumes
exactly (every rank restarts from the synced weights). A mid-segment checkpoint holds rank 0's
replica, which has drifted from the others': a resume from it restarts every rank from that
replica, with the positions exact and the other replicas' drift lost; ``checkpoint.policy:
segment_end`` avoids that under these kinds.
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

import torch
import torch.distributed as dist

if TYPE_CHECKING:
    from distrainer.config import DistrainerConfig


@runtime_checkable
class SegmentSync(Protocol):
    """Called on every rank at a segment end with the (possibly wrapped) model, the inner
    optimizer and the ``TrainInfo`` of the last step; the same collectives on every rank."""

    def on_segment_sync(self, model: Any, optimizer: Any, info: Any) -> None: ...

    def state_dict(self) -> dict[str, Any]: ...

    def load_state_dict(self, state: dict[str, Any]) -> None: ...

    def reset(self, model: Any) -> None: ...


def _module(model: Any) -> torch.nn.Module:
    return getattr(model, "module", model)


def group_size() -> int:
    """The process group's world size, 1 without a group (a driver, the fake fixture)."""
    return dist.get_world_size() if dist.is_available() and dist.is_initialized() else 1


def average_(tensors: Iterable[torch.Tensor]) -> None:
    """In place: every tensor becomes its mean over the ranks (a no-op without a group)."""
    n = group_size()
    if n == 1:
        return
    for t in tensors:
        dist.all_reduce(t)
        t.div_(n)


def _float_buffers(module: torch.nn.Module) -> list[torch.Tensor]:
    return [b for b in module.buffers() if b.is_floating_point()]


class LocalSGD:
    """Average the parameters, and the floating-point buffers (BatchNorm's statistics), at the
    segment end; the inner optimizer's state stays local to each rank."""

    def on_segment_sync(self, model: Any, optimizer: Any, info: Any) -> None:
        m = _module(model)
        with torch.no_grad():
            average_(list(m.parameters()) + _float_buffers(m))

    def state_dict(self) -> dict[str, Any]:
        return {}

    def load_state_dict(self, state: dict[str, Any]) -> None:
        pass

    def reset(self, model: Any) -> None:
        pass


class DiLoCo:
    """Douillard et al. 2023. Every rank keeps an *anchor*, the parameters at the last sync. At
    the sync the average over the ranks of ``anchor - parameters`` (the inner change, as a
    descent direction) is the gradient of an outer SGD with Nesterov momentum over the anchor;
    the model then takes the anchor. Buffers are averaged as in local SGD."""

    def __init__(
        self,
        model: Any,
        outer_lr: float = 0.7,
        outer_momentum: float = 0.9,
        nesterov: bool = True,
    ):
        if outer_lr <= 0:
            raise ValueError(f"outer_lr must be positive, got {outer_lr}")
        if not 0 <= outer_momentum < 1:
            raise ValueError(f"outer_momentum must be in [0, 1), got {outer_momentum}")
        self.settings: dict[str, Any] = {
            "lr": outer_lr,
            "momentum": outer_momentum,
            "nesterov": nesterov and outer_momentum > 0,
        }
        self.anchor: list[torch.Tensor] = []
        self.outer: torch.optim.SGD
        self.reset(model)

    def reset(self, model: Any) -> None:
        """The anchor becomes the model's parameters; the outer optimizer starts afresh."""
        self.anchor = [p.detach().clone() for p in _module(model).parameters()]
        self.outer = torch.optim.SGD(self.anchor, **self.settings)

    def on_segment_sync(self, model: Any, optimizer: Any, info: Any) -> None:
        m = _module(model)
        params = list(m.parameters())
        with torch.no_grad():
            deltas = [a - p for a, p in zip(self.anchor, params, strict=True)]
            average_(deltas)
            for a, d in zip(self.anchor, deltas, strict=True):
                a.grad = d
            self.outer.step()
            for a in self.anchor:
                a.grad = None
            for a, p in zip(self.anchor, params, strict=True):
                p.copy_(a)
            average_(_float_buffers(m))

    def state_dict(self) -> dict[str, Any]:
        return {
            "anchor": [a.detach().cpu().clone() for a in self.anchor],
            "outer": self.outer.state_dict(),
        }

    def load_state_dict(self, state: dict[str, Any]) -> None:
        with torch.no_grad():
            for a, s in zip(self.anchor, state["anchor"], strict=True):
                a.copy_(s)
        self.outer.load_state_dict(state["outer"])


DTYPES = {"fp32": torch.float32, "bf16": torch.bfloat16, "fp16": torch.float16}


def is_sharded(model: Any) -> bool:
    """True when the module's parameters are DTensors (``fully_shard`` applied)."""
    from torch.distributed.tensor import DTensor

    return any(isinstance(p, DTensor) for p in _module(model).parameters())


def wrap_fsdp(
    model: torch.nn.Module,
    parallel: Any,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None = None,
) -> torch.nn.Module:
    """FSDP2 over the process group: every direct child with parameters is a shard unit, then the
    root. ``fully_shard`` swaps the parameters for sharded ones under the same names, so an
    ``optimizer`` built by ``build_model`` over the originals is re-pointed at them (it has no
    state yet). Without a process group (a driver, the fake fixture) the model stays as it is."""
    if not (dist.is_available() and dist.is_initialized()):
        return model
    from torch.distributed.device_mesh import init_device_mesh
    from torch.distributed.fsdp import MixedPrecisionPolicy, fully_shard

    names = {id(p): n for n, p in model.named_parameters()}
    mesh = init_device_mesh(device.type, (dist.get_world_size(),))
    policy = MixedPrecisionPolicy(
        param_dtype=DTYPES.get(parallel.param_dtype or ""),
        reduce_dtype=DTYPES.get(parallel.reduce_dtype or ""),
    )
    kwargs: dict[str, Any] = {
        "mesh": mesh,
        "reshard_after_forward": parallel.reshard_after_forward,
        "mp_policy": policy,
    }
    for child in model.children():
        if any(True for _ in child.parameters(recurse=True)):
            fully_shard(child, **kwargs)
    fully_shard(model, **kwargs)
    if optimizer is not None:
        if optimizer.state:
            raise ValueError("parallel.kind fsdp: build_model's optimizer must have no state yet")
        sharded = dict(model.named_parameters())
        for group in optimizer.param_groups:
            try:
                group["params"] = [sharded[names[id(p)]] for p in group["params"]]
            except KeyError as exc:
                raise ValueError(
                    "parallel.kind fsdp: the optimizer holds a parameter that is not the model's"
                ) from exc
    return model


def build_sync(cfg: DistrainerConfig, model: Any) -> SegmentSync | None:
    """The segment-end sync of ``cfg.parallel.kind``, or None for the kinds that need none."""
    p = cfg.parallel
    if p.kind == "local_sgd":
        return LocalSGD()
    if p.kind == "diloco":
        return DiLoCo(
            model, outer_lr=p.outer_lr, outer_momentum=p.outer_momentum, nesterov=p.outer_nesterov
        )
    return None
