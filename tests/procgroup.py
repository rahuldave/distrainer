"""A real process group for the trainer tests: ``n`` ranks as processes, Gloo on localhost.

The fake-Ray fixture in ``test_train_loop.py`` drives ``train_loop`` rank by rank in one process
with the collectives stubbed out, which cannot test anything *about* collectives. ``run_world``
here spawns ``n`` processes; each joins a Gloo group on a free localhost port, patches the
``ray.train`` entry points the loop uses onto the group (``prepare_model`` becomes the real DDP
wrap, ``barrier`` and ``broadcast_from_rank_zero`` become ``torch.distributed`` calls, ``report``
records), runs ``train_loop`` and hands its reports back through a file. The block stores are
the ones ``test_train_loop.make_store`` builds; the audit facts must be the same.

Every callable that crosses into a child (``train_step``, ``build_model``, hooks) has to be
picklable, so module-level functions and classes only.
"""

from __future__ import annotations

import os
import pickle
import socket
import sys
import time
from dataclasses import dataclass, field
from datetime import timedelta
from typing import Any

import torch
import torch.multiprocessing as mp

from distrainer.config import DistrainerConfig
from distrainer.trainer import DistTrainer

JOIN_TIMEOUT_S = int(os.environ.get("PROCGROUP_TIMEOUT_S", "180"))
COLLECTIVE_TIMEOUT_S = 60
DEBUG = bool(os.environ.get("PROCGROUP_DEBUG"))


def _log(rank: int, msg: str) -> None:
    """``PROCGROUP_DEBUG=1`` prints each rank's progress to stderr (where a hang is, in short)."""
    if DEBUG:
        print(f"[procgroup rank {rank}] {msg}", file=sys.stderr, flush=True)


@dataclass
class Report:
    rank: int
    metrics: dict[str, Any]
    checkpoint_dir_name: str | None
    checkpoint_path: str | None  # rank 0's saved directory, or None

    @property
    def has_checkpoint(self) -> bool:
        return self.checkpoint_path is not None

    def checkpoint(self) -> Any:
        from ray.train import Checkpoint

        assert self.checkpoint_path is not None
        return Checkpoint.from_directory(self.checkpoint_path)


@dataclass
class World:
    n: int
    reports: list[Report] = field(default_factory=list)

    def by_rank(self, rank: int) -> list[Report]:
        return [r for r in self.reports if r.rank == rank]

    def checkpoints(self) -> dict[str, Report]:
        """The reports that carried a checkpoint, by directory name."""
        return {r.checkpoint_dir_name or "": r for r in self.reports if r.has_checkpoint}


def free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return int(s.getsockname()[1])


class _Context:
    def __init__(self, rank: int, n: int):
        self.rank, self.n = rank, n

    def get_world_rank(self) -> int:
        return self.rank

    def get_local_rank(self) -> int:
        return self.rank

    def get_world_size(self) -> int:
        return self.n


def _patch_ray(
    rank: int, n: int, checkpoint_path: str | None, reports: list, prepare: bool
) -> None:
    """Point the ``ray.train`` names ``train_loop`` looks up at call time onto the Gloo group."""
    import ray.train
    import ray.train.collective
    import ray.train.torch
    import torch.distributed as dist
    from ray.train import Checkpoint

    def report(metrics, checkpoint=None, checkpoint_dir_name=None, **kw):
        reports.append(
            (rank, dict(metrics), checkpoint_dir_name, checkpoint.path if checkpoint else None)
        )

    def prepare_model(model, **kw):
        if prepare and n > 1:
            return torch.nn.parallel.DistributedDataParallel(model)
        return model

    def broadcast_from_rank_zero(data):
        box = [data]
        dist.broadcast_object_list(box, src=0)
        return box[0]

    ray.train.get_context = lambda: _Context(rank, n)
    ray.train.get_checkpoint = lambda: (
        Checkpoint.from_directory(checkpoint_path) if checkpoint_path else None
    )
    ray.train.report = report
    ray.train.torch.prepare_model = prepare_model
    ray.train.torch.get_device = lambda: torch.device("cpu")
    ray.train.collective.barrier = dist.barrier
    ray.train.collective.broadcast_from_rank_zero = broadcast_from_rank_zero


def _worker(
    rank: int,
    n: int,
    port: int,
    loop_config: dict[str, Any],
    checkpoint_path: str | None,
    out_dir: str,
    prepare: bool,
) -> None:
    import faulthandler

    import torch.distributed as dist

    _log(rank, "worker entered")
    # a rank stuck in a collective dumps every thread's stack and dies instead of timing out mutely
    faulthandler.dump_traceback_later(max(JOIN_TIMEOUT_S - 10, 5), exit=True)
    os.environ["TMPDIR"] = os.path.join(out_dir, "tmp")  # CheckpointIO's scratch stays in tmp_path
    os.makedirs(os.environ["TMPDIR"], exist_ok=True)
    torch.set_num_threads(1)
    dist.init_process_group(
        "gloo",
        init_method=f"tcp://127.0.0.1:{port}",
        rank=rank,
        world_size=n,
        timeout=timedelta(seconds=COLLECTIVE_TIMEOUT_S),
    )
    _log(rank, "process group up")
    reports: list = []
    try:
        _patch_ray(rank, n, checkpoint_path, reports, prepare)
        _log(rank, "ray.train patched")
        from distrainer.trainer import train_loop

        train_loop(loop_config)
        _log(rank, "train_loop returned")
    finally:
        with open(os.path.join(out_dir, f"rank{rank}.pkl"), "wb") as f:
            pickle.dump(reports, f)
        # no rank tears the group down while another may still be inside a collective
        dist.barrier()
        dist.destroy_process_group()
        _log(rank, "process group down")
        faulthandler.cancel_dump_traceback_later()


def run_world(
    cfg: DistrainerConfig,
    n: int,
    step: Any,
    build_model: Any,
    out_dir: str,
    checkpoint: str | None = None,
    loop_extra: dict[str, Any] | None = None,
    prepare: bool = True,
) -> World:
    """Run ``train_loop`` on ``n`` ranks over a Gloo group; ``checkpoint`` is a saved directory.

    ``prepare=False`` leaves the model unwrapped (no all-reduce), which is how a test shows what
    DDP does. ``out_dir`` receives the ranks' report files and the checkpoint scratch.
    """
    os.makedirs(out_dir, exist_ok=True)
    loop_config = {**DistTrainer(step, build_model, cfg).loop_config(), **(loop_extra or {})}
    ctx = mp.start_processes(
        _worker,
        args=(n, free_port(), loop_config, checkpoint, out_dir, prepare),
        nprocs=n,
        join=False,
        start_method="spawn",
    )
    deadline = time.monotonic() + JOIN_TIMEOUT_S
    try:
        # join(timeout) returns as soon as one rank is done; it is True only when all are
        while not ctx.join(timeout=max(deadline - time.monotonic(), 0.1)):
            if time.monotonic() > deadline:
                raise TimeoutError(f"{n} ranks did not finish within {JOIN_TIMEOUT_S} s")
    except Exception:
        for p in ctx.processes:
            if p.is_alive():
                p.kill()
        raise
    world = World(n=n)
    for rank in range(n):
        with open(os.path.join(out_dir, f"rank{rank}.pkl"), "rb") as f:
            world.reports.extend(Report(*r) for r in pickle.load(f))
    return world
