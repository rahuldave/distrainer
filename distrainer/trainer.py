"""distrainer.trainer: ``DistTrainer`` wraps Ray Train's ``TorchTrainer`` around the block loop.

Spec sections 5 and 6. Every rank runs :func:`train_loop`: open the log, restore the ledger from
the latest checkpoint, deal the remaining positions over the current world size, and for each
block call the user's ``train_step``; after every step all ranks call ``ray.train.report`` (a
barrier), with a checkpoint attached on rank 0 whenever the policy says so. Segment ends run the
hooks on rank 0 and then a collective barrier.
"""

from __future__ import annotations

import os
import shutil
import tempfile
import time
import uuid
from collections.abc import Callable, Iterator, Sequence
from dataclasses import dataclass, field
from typing import Any

import pyarrow as pa
import torch

from distrainer import __version__
from distrainer.audit import AuditWriter
from distrainer.config import DistrainerConfig
from distrainer.hooks import SegmentHook
from distrainer.ledger import Ledger
from distrainer.loader import LaneLoader
from distrainer.log import BlockLog, Segment
from distrainer.planner import lane, resume_start, steps_per_segment
from distrainer.policy import CheckpointPolicy, StepContext, build_policy, notify_checkpoint

# ---- user-facing contracts ----


@dataclass(frozen=True)
class TrainInfo:
    """What ``build_model`` and ``train_step`` get to know about the run."""

    rank: int
    world_size: int
    config: DistrainerConfig
    device: torch.device = field(default_factory=lambda: torch.device("cpu"))
    # per step (zero in build_model):
    position: int = 0
    segment: int = 0
    step_in_segment: int = 0
    pass_idx: int = 0
    attempt: int = 0

    @property
    def train(self) -> dict[str, Any]:
        """The free-form ``train:`` section of the config."""
        return self.config.train


BuildModel = Callable[[TrainInfo], tuple[torch.nn.Module, torch.optim.Optimizer]]
TrainStep = Callable[
    [torch.nn.Module, torch.optim.Optimizer, pa.Table, TrainInfo], dict[str, float]
]


def unwrap(model: torch.nn.Module) -> torch.nn.Module:
    """The user's module inside a DDP wrapper (or the module itself)."""
    return getattr(model, "module", model)


# ---- checkpoint files ----


def checkpoint_dir_name(ledger: Ledger) -> str:
    return f"checkpoint_g{ledger.segment:06d}_s{ledger.cursor:04d}"


class CheckpointIO:
    """Write and read distrainer checkpoints: ``model.pt``, ``optimizer.pt``, ``ledger.json``.

    ``save`` writes into a non-temporary worker-local directory (required for ASYNC upload) and
    attaches the ledger as cheap ``Checkpoint`` metadata; ``load`` restores states and returns
    the ledger; ``read_ledger`` reads only the metadata.
    """

    MODEL = "model.pt"
    OPTIMIZER = "optimizer.pt"

    def __init__(self, scratch_dir: str | None = None, run_name: str = "run"):
        base = scratch_dir or os.path.join(tempfile.gettempdir(), "distrainer", run_name)
        self.scratch_dir = os.path.join(base, uuid.uuid4().hex[:8])
        os.makedirs(self.scratch_dir, exist_ok=True)

    def save(
        self,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer | None,
        ledger: Ledger,
        extra: dict[str, Any] | None = None,
    ) -> Any:
        from ray.train import Checkpoint

        path = os.path.join(
            self.scratch_dir, checkpoint_dir_name(ledger) + "_" + uuid.uuid4().hex[:6]
        )
        os.makedirs(path, exist_ok=True)
        torch.save(unwrap(model).state_dict(), os.path.join(path, self.MODEL))
        if optimizer is not None:
            torch.save(optimizer.state_dict(), os.path.join(path, self.OPTIMIZER))
        ledger.save(path)
        checkpoint = Checkpoint.from_directory(path)
        checkpoint.set_metadata(
            {"ledger": ledger.asdict(), "distrainer": __version__, **(extra or {})}
        )
        return checkpoint

    @staticmethod
    def load(
        checkpoint: Any,
        model: torch.nn.Module | None = None,
        optimizer: torch.optim.Optimizer | None = None,
    ) -> Ledger:
        with checkpoint.as_directory() as path:
            if model is not None:
                state = torch.load(os.path.join(path, CheckpointIO.MODEL), map_location="cpu")
                unwrap(model).load_state_dict(state)
            opt_path = os.path.join(path, CheckpointIO.OPTIMIZER)
            if optimizer is not None and os.path.exists(opt_path):
                optimizer.load_state_dict(torch.load(opt_path, map_location="cpu"))
            return Ledger.load(path)

    @staticmethod
    def read_ledger(checkpoint: Any) -> Ledger:
        meta = checkpoint.get_metadata()
        if "ledger" in meta:
            return Ledger.from_dict(meta["ledger"])
        with checkpoint.as_directory() as path:
            return Ledger.load(path)

    def cleanup(self) -> None:
        shutil.rmtree(self.scratch_dir, ignore_errors=True)


# ---- the per-rank loop ----


def lanes_from(
    log: BlockLog, rank: int, world_size: int, seq: int, start_step: int, poll_s: float
) -> Callable[[Any], Iterator[tuple[Segment, list]]]:
    """The ``lanes`` generator factory of spec section 5 (prefetch spans segments)."""

    def gen(stop: Any) -> Iterator[tuple[Segment, list]]:
        s, first = seq, start_step
        while (segment := log.wait_segment(s, poll_s=poll_s, stop=stop)) is not None:
            yield segment, lane(segment, rank, world_size, first)
            s, first = s + 1, 0

    return gen


def train_loop(loop_config: dict[str, Any]) -> None:
    """Runs on every Ray Train worker. ``loop_config`` carries the config and user callables."""
    import ray.train
    import ray.train.torch
    from ray.train import CheckpointUploadMode
    from ray.train.collective import barrier

    cfg = DistrainerConfig.from_dict(loop_config["config"])
    build_model: BuildModel = loop_config["build_model"]
    train_step: TrainStep = loop_config["train_step"]
    hooks: Sequence[SegmentHook] = loop_config.get("hooks") or ()
    policy: CheckpointPolicy = loop_config.get("policy") or build_policy(
        cfg.checkpoint.as_policy_dict()
    )
    upload_mode = (
        CheckpointUploadMode.ASYNC
        if cfg.checkpoint.upload_mode == "async"
        else CheckpointUploadMode.SYNC
    )

    ctx = ray.train.get_context()
    rank, n = ctx.get_world_rank(), ctx.get_world_size()
    fs, store_root = cfg.store_fs()
    log = BlockLog.open(fs, store_root)
    W = log.W
    steps = steps_per_segment(W, n)  # raises if W % n != 0

    info = TrainInfo(rank=rank, world_size=n, config=cfg)
    model, optimizer = build_model(info)
    model = ray.train.torch.prepare_model(model)

    ledger = Ledger(world_size=n)
    checkpoint = ray.train.get_checkpoint()
    if checkpoint is not None:
        ledger = CheckpointIO.load(checkpoint, model, optimizer)
        ledger.run_attempt += 1
    attempt = ledger.run_attempt
    seq, start_step = resume_start(ledger, n, W=W)
    ledger = Ledger(
        segment=seq, cursor=start_step, world_size=n, pass_idx=ledger.pass_idx, run_attempt=attempt
    )

    io = CheckpointIO(run_name=cfg.run_name)
    audit = AuditWriter(fs, store_root, cfg.run_name, attempt, rank)
    loader = LaneLoader(
        fs,
        store_root,
        lanes_from(log, rank, n, seq, start_step, cfg.log.wait_poll_s),
        prefetch=cfg.loader.prefetch,
        threads=cfg.loader.threads,
    )
    t0 = time.monotonic()
    try:
        for segment, (position, ref, table) in loader:
            if segment.seq != ledger.segment:
                ledger.segment, ledger.cursor, ledger.pass_idx = segment.seq, 0, segment.pass_idx
            step_info = TrainInfo(
                rank=rank,
                world_size=n,
                config=cfg,
                position=position,
                segment=segment.seq,
                step_in_segment=ledger.cursor,
                pass_idx=segment.pass_idx,
                attempt=attempt,
            )
            metrics = dict(train_step(model, optimizer, table, step_info))
            audit.append(n, segment.seq, ledger.cursor, position, ref.block_id)
            ledger.cursor += 1
            ledger.world_size = n
            ledger.pass_idx = segment.pass_idx
            segment_end = ledger.cursor == steps
            pass_end = segment_end and log.next_pass_differs(segment)
            sctx = StepContext(
                position=position,
                step_in_segment=ledger.cursor,
                segment=segment.seq,
                pass_idx=segment.pass_idx,
                segment_end=segment_end,
                pass_end=pass_end,
                elapsed_s=time.monotonic() - t0,
                rank=rank,
                world_size=n,
            )
            metrics.update(
                position=position,
                segment=segment.seq,
                cursor=ledger.cursor,
                world_size=n,
                attempt=attempt,
            )
            if policy.should_checkpoint(sctx):
                notify_checkpoint(policy, sctx)
                ckpt = io.save(model, optimizer, ledger) if rank == 0 else None
                ray.train.report(
                    metrics,
                    checkpoint=ckpt,
                    checkpoint_dir_name=checkpoint_dir_name(ledger),
                    checkpoint_upload_mode=upload_mode,
                    delete_local_checkpoint_after_upload=True,
                )
            else:
                ray.train.report(metrics)
            if segment_end:
                if rank == 0:
                    for hook in hooks:
                        hook.on_segment_end(model, ledger, log, step_info)
                barrier()
    finally:
        loader.close()
        audit.close()


# ---- the driver-side wrapper ----


class DistTrainer:
    """Build the Ray Train ``TorchTrainer`` for a config and run it."""

    def __init__(
        self,
        train_step: TrainStep,
        build_model: BuildModel,
        config: DistrainerConfig,
        policy: CheckpointPolicy | None = None,
        hooks: Sequence[SegmentHook] = (),
        resume_from_checkpoint: Any = None,
        scaling_config: Any = None,
        run_config: Any = None,
    ):
        self.train_step = train_step
        self.build_model = build_model
        self.config = config
        self.policy = policy
        self.hooks = list(hooks)
        self.resume_from_checkpoint = resume_from_checkpoint
        self._scaling_config = scaling_config
        self._run_config = run_config

    def scaling_config(self) -> Any:
        from ray.train.v2.api.config import ScalingConfig

        if self._scaling_config is not None:
            return self._scaling_config
        s = self.config.scaling
        return ScalingConfig(
            num_workers=s.num_workers,
            use_gpu=s.use_gpu,
            resources_per_worker=dict(s.resources_per_worker),
            elastic_resize_monitor_interval_s=s.elastic_resize_monitor_interval_s,
        )

    def run_config(self) -> Any:
        from ray.train.v2.api.config import CheckpointConfig, FailureConfig, RunConfig

        if self._run_config is not None:
            return self._run_config
        cfg = self.config
        fs, runs_root = cfg.runs_fs()
        kwargs: dict[str, Any] = {"storage_path": runs_root}
        if cfg.storage.kind != "local":
            kwargs["storage_filesystem"] = fs
        return RunConfig(
            name=cfg.run_name,
            failure_config=FailureConfig(max_failures=cfg.failure.max_failures),
            checkpoint_config=CheckpointConfig(num_to_keep=cfg.checkpoint.num_to_keep),
            **kwargs,
        )

    def loop_config(self) -> dict[str, Any]:
        return {
            "config": self.config.asdict(),
            "build_model": self.build_model,
            "train_step": self.train_step,
            "policy": self.policy,
            "hooks": self.hooks,
        }

    def trainer(self) -> Any:
        from ray.train.v2.torch.torch_trainer import TorchTrainer

        return TorchTrainer(
            train_loop,
            train_loop_config=self.loop_config(),
            scaling_config=self.scaling_config(),
            run_config=self.run_config(),
            resume_from_checkpoint=self.resume_from_checkpoint,
        )

    def fit(self) -> Any:
        return self.trainer().fit()

    def run_dir(self) -> tuple[Any, str]:
        """``(fs, path)`` of the Ray Train run directory holding the checkpoints."""
        fs, runs_root = self.config.runs_fs()
        return fs, f"{runs_root}/{self.config.run_name}"
