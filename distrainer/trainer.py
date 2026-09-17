"""distrainer.trainer: ``DistTrainer`` wraps Ray Train's ``TorchTrainer`` around the block loop.

Spec sections 5 and 6. Every rank runs :func:`train_loop`: open the log, restore the ledger from
the latest checkpoint, deal the remaining positions over the current world size, and for each
block call the user's ``train_step``; after every step all ranks call ``ray.train.report`` (a
barrier), with a checkpoint attached on rank 0 whenever the policy says so. Segment ends run the
hooks on rank 0 (the ``hooks`` given to ``DistTrainer`` plus those built from ``cfg.hooks``),
then retention ``gc`` if ``log.gc`` is set, then a collective barrier.
"""

from __future__ import annotations

import os
import shutil
import tempfile
import time
import uuid
from collections.abc import Callable, Iterator, Sequence
from dataclasses import dataclass, field, replace
from typing import Any

import pyarrow as pa
import torch

from distrainer import __version__
from distrainer.audit import AuditWriter, next_attempt
from distrainer.config import DistrainerConfig, ParallelConfig
from distrainer.hooks import SegmentHook, build_hooks, hook_specs, load_entry
from distrainer.ledger import Ledger
from distrainer.loader import LaneLoader
from distrainer.log import BlockLog, Segment
from distrainer.planner import lane, resume_start, steps_per_segment
from distrainer.policy import CheckpointPolicy, StepContext, build_policy, notify_checkpoint

# ---- user-facing contracts ----


@dataclass(frozen=True)
class TrainInfo:
    """What ``build_model`` and ``train_step`` get to know about the run.

    ``device`` is Ray Train's device for this worker (CPU here, the GPU with ``use_gpu``). Per-step
    fields are zero in ``build_model``; ``step_in_segment`` counts the steps completed *before*
    this one (``StepContext.step_in_segment`` seen by policies counts this one as done).
    """

    rank: int
    world_size: int
    config: DistrainerConfig
    device: torch.device = field(default_factory=lambda: torch.device("cpu"))
    position: int = 0
    segment: int = 0
    step_in_segment: int = 0
    pass_idx: int = 0
    attempt: int = 0

    @property
    def train(self) -> dict[str, Any]:
        """The free-form ``train:`` section of the config."""
        return self.config.train

    @property
    def parallel(self) -> ParallelConfig:
        """The ``parallel:`` section: which kind of wrap the model got (``kind``)."""
        return self.config.parallel


BuildModel = Callable[[TrainInfo], tuple[torch.nn.Module, torch.optim.Optimizer]]
TrainStep = Callable[
    [torch.nn.Module, torch.optim.Optimizer, pa.Table, TrainInfo], dict[str, float]
]


def unwrap(model: torch.nn.Module) -> torch.nn.Module:
    """The user's module inside a DDP wrapper (or the module itself)."""
    return getattr(model, "module", model)


def parallel_strategy(parallel: ParallelConfig) -> str | None:
    """Ray Train's ``prepare_model(parallel_strategy=...)`` argument for a ``parallel:`` kind."""
    return {"ddp": "ddp", "none": None}[parallel.kind]


def init_ray(cfg: DistrainerConfig, **kwargs: Any) -> None:
    """``ray.init`` for distrainer drivers.

    Address from the config, quiet driver logging, the ``uv run`` hook off (workers use this
    interpreter directly), and the Train v2 controller poll interval
    (``RAY_TRAIN_HEALTH_CHECK_INTERVAL_S``) lowered so ``report`` does not cap throughput.
    """
    import logging
    import os

    os.environ.setdefault("RAY_ENABLE_UV_RUN_RUNTIME_ENV", "0")
    env_vars: dict[str, str] = {}
    if cfg.ray_health_check_interval_s is not None:
        env_vars["RAY_TRAIN_HEALTH_CHECK_INTERVAL_S"] = str(cfg.ray_health_check_interval_s)
    import ray

    if ray.is_initialized():
        return
    runtime_env = dict(kwargs.pop("runtime_env", {}) or {})
    if env_vars:
        runtime_env["env_vars"] = {**runtime_env.get("env_vars", {}), **env_vars}
    ray.init(
        address=cfg.ray_address,
        logging_level=logging.ERROR,
        runtime_env=runtime_env or None,
        **kwargs,
    )


class MetricAggregator:
    """Mean of numeric metrics since the last ``report`` (so fewer reports lose no signal)."""

    def __init__(self) -> None:
        self._sums: dict[str, float] = {}
        self._count = 0

    def add(self, metrics: dict[str, Any]) -> None:
        for k, v in metrics.items():
            if isinstance(v, bool) or not isinstance(v, int | float):
                continue
            self._sums[k] = self._sums.get(k, 0.0) + float(v)
        self._count += 1

    def flush(self, last: dict[str, Any]) -> dict[str, Any]:
        """``last`` step metrics plus ``<key>_mean`` over the window and ``steps_in_report``."""
        out = dict(last)
        if self._count:
            out.update({f"{k}_mean": s / self._count for k, s in self._sums.items()})
            out["steps_in_report"] = self._count
        self._sums, self._count = {}, 0
        return out


# ---- checkpoint files ----


def checkpoint_dir_name(ledger: Ledger) -> str:
    """Unique per data position, world size and attempt: ``cursor`` alone is ambiguous across
    resizes (cursor 2 at n=4 is 8 positions, at n=2 it is 4) and Ray trims checkpoints by path."""
    return (
        f"checkpoint_g{ledger.segment:06d}_p{ledger.done_positions():06d}"
        f"_n{ledger.world_size:02d}_a{ledger.run_attempt:02d}"
    )


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

    def cleanup_uploaded(self) -> None:
        """Remove the directories Ray already emptied after upload; leave in-flight ones."""
        try:
            for name in os.listdir(self.scratch_dir):
                path = os.path.join(self.scratch_dir, name)
                if os.path.isdir(path) and not os.listdir(path):
                    os.rmdir(path)
            if not os.listdir(self.scratch_dir):
                os.rmdir(self.scratch_dir)
        except OSError:
            pass


# ---- the per-rank loop ----


def lanes_from(
    log: BlockLog, rank: int, world_size: int, seq: int, start_step: int, poll_s: float
) -> Callable[[Any], Iterator[tuple[Segment, list]]]:
    """The ``lanes`` generator factory of spec section 5 (prefetch spans segments)."""

    def gen(stop: Any) -> Iterator[tuple[Segment, list]]:
        s, first = seq, start_step
        lowest = log.first_seq()
        if lowest is not None and s < lowest:
            raise RuntimeError(
                f"cannot resume at segment {s}: the log starts at {lowest} (garbage-collected); "
                "resume from a newer checkpoint or raise log.retention_segments"
            )
        while (segment := log.wait_segment(s, poll_s=poll_s, stop=stop)) is not None:
            yield segment, lane(segment, rank, world_size, first)
            s, first = s + 1, 0
        last = log.last_seq()
        if last is not None and s <= last and not (stop is not None and stop.is_set()):
            raise RuntimeError(f"segment {s} is missing from the log (segments up to {last} exist)")

    return gen


def train_loop(loop_config: dict[str, Any]) -> None:
    """Runs on every Ray Train worker. ``loop_config`` carries the config and user callables."""
    import ray.train
    import ray.train.torch
    from ray.train import CheckpointUploadMode
    from ray.train.collective import barrier, broadcast_from_rank_zero

    cfg = DistrainerConfig.from_dict(loop_config["config"])
    build_model: BuildModel = loop_config["build_model"]
    train_step: TrainStep = loop_config["train_step"]
    hooks: list[SegmentHook] = list(loop_config.get("hooks") or ())
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
    device = ray.train.torch.get_device()

    info = TrainInfo(rank=rank, world_size=n, config=cfg, device=device)
    model, optimizer = build_model(info)
    model = ray.train.torch.prepare_model(model, parallel_strategy=parallel_strategy(cfg.parallel))

    # hooks and gc are writers and run on rank 0 only; they get a BlockLog of their own rather
    # than the reader instance the loader's producer thread polls
    hook_log: BlockLog | None = None
    if rank == 0:
        hooks += build_hooks(cfg)
        if hooks or cfg.log.gc:
            hook_log = BlockLog.open(fs, store_root)

    # every restart of the worker group gets its own attempt id, even from the same checkpoint
    attempt = int(
        broadcast_from_rank_zero(next_attempt(fs, store_root, cfg.run_name) if rank == 0 else None)
    )
    ledger = Ledger(world_size=n)
    # Ray hands back the latest reported checkpoint after a failure or resize; a brand-new run
    # may start from an explicit checkpoint instead (Train v2 deprecated resume_from_checkpoint)
    checkpoint = ray.train.get_checkpoint() or loop_config.get("initial_checkpoint")
    if checkpoint is not None:
        ledger = CheckpointIO.load(checkpoint, model, optimizer)
    last_ckpt_segment = ledger.segment  # gc keeps retention_segments behind this
    seq, start_step = resume_start(ledger, n, W=W)
    resumed_at_segment_end = checkpoint is not None and ledger.done_positions() == W
    ckpt_ledger = replace(ledger)
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
    if hook_log is not None and resumed_at_segment_end and hooks:
        # a segment-end checkpoint is reported before the hooks run: if the previous attempt
        # died in between, the segment end was never served (hooks are at-least-once), and
        # without the next segment every rank would wait forever
        if not hook_log.ended() and not hook_log.has_segment(seq):
            replay_info = TrainInfo(
                rank=rank,
                world_size=n,
                config=cfg,
                device=device,
                position=seq * W - 1,
                segment=ckpt_ledger.segment,
                step_in_segment=ckpt_ledger.cursor,
                pass_idx=ckpt_ledger.pass_idx,
                attempt=attempt,
            )
            for hook in hooks:
                hook.on_segment_end(unwrap(model), replace(ckpt_ledger), hook_log, replay_info)
    t0 = time.monotonic()
    agg = MetricAggregator()
    reported_last = True
    n_reports = 0  # carried in every report so Result.metrics["reports"] is the per-rank count
    metrics: dict[str, Any] = {}
    try:
        for segment, (position, ref, table) in loader:
            if segment.seq != ledger.segment:
                ledger.segment, ledger.cursor, ledger.pass_idx = segment.seq, 0, segment.pass_idx
            step_info = TrainInfo(
                rank=rank,
                world_size=n,
                config=cfg,
                device=device,
                position=position,
                segment=segment.seq,
                step_in_segment=ledger.cursor,
                pass_idx=segment.pass_idx,
                attempt=attempt,
            )
            metrics = dict(train_step(model, optimizer, table, step_info))
            agg.add(metrics)
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
            # report is a barrier and, in Train v2, a one-slot queue drained at the controller's
            # poll interval; by default it is called only where a checkpoint is taken (the policy
            # decides identically on every rank) and metrics are averaged in between
            if policy.should_checkpoint(sctx):
                notify_checkpoint(policy, sctx)
                ckpt = io.save(model, optimizer, ledger) if rank == 0 else None
                last_ckpt_segment = ledger.segment
                n_reports += 1
                ray.train.report(
                    agg.flush({**metrics, "reports": n_reports}),
                    checkpoint=ckpt,
                    checkpoint_dir_name=checkpoint_dir_name(ledger),
                    checkpoint_upload_mode=upload_mode,
                    delete_local_checkpoint_after_upload=True,
                )
                audit.flush()  # object-store audit buffers survive a SIGKILL up to here
                reported_last = True
            elif cfg.checkpoint.report_every_step:
                n_reports += 1
                ray.train.report(agg.flush({**metrics, "reports": n_reports}))
                reported_last = True
            else:
                reported_last = False
            if segment_end:
                if hook_log is not None:  # rank 0
                    snapshot = replace(ledger)  # hooks see the position, never the live ledger
                    for hook in hooks:
                        hook.on_segment_end(unwrap(model), snapshot, hook_log, step_info)
                    if cfg.log.gc:
                        # measured from the last checkpoint rank 0 *saved*; the controller may
                        # restart from the one before it (ASYNC upload still in flight), which
                        # is why retention_segments must be at least 1 when gc is on
                        keep_from = last_ckpt_segment - cfg.log.retention_segments
                        if keep_from > 0:
                            hook_log.gc(keep_from, delete_blocks=cfg.log.gc_blocks)
                barrier()
        if not reported_last and metrics:
            n_reports += 1  # every rank took the same number of steps, so counts stay equal
            ray.train.report(agg.flush({**metrics, "reports": n_reports}))
    finally:
        loader.close()
        audit.close()
        io.cleanup_uploaded()


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
        for name, entry, _args in hook_specs(config.hooks):
            try:
                load_entry(entry)  # fail on the driver, not inside rank 0 with the others waiting
            except Exception as exc:  # ImportError, AttributeError, or the module's own error
                raise ValueError(f"hooks.{name}: cannot import {entry!r}: {exc}") from exc
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
            "initial_checkpoint": self.resume_from_checkpoint,
        }

    def trainer(self) -> Any:
        from ray.train.v2._internal.constants import is_v2_enabled

        if not is_v2_enabled():
            raise RuntimeError(
                "distrainer needs Ray Train v2 (the default since Ray 2.58); "
                "unset RAY_TRAIN_V2_ENABLED=0"
            )
        from ray.train.v2.torch.torch_trainer import TorchTrainer

        return TorchTrainer(
            train_loop,
            train_loop_config=self.loop_config(),
            scaling_config=self.scaling_config(),
            run_config=self.run_config(),
        )

    def fit(self) -> Any:
        return self.trainer().fit()

    def run_dir(self) -> tuple[Any, str]:
        """``(fs, path)`` of the Ray Train run directory holding the checkpoints."""
        fs, runs_root = self.config.runs_fs()
        return fs, f"{runs_root}/{self.config.run_name}"
