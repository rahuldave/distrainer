"""distrainer.config: the YAML configuration (spec section 7) as validated dataclasses.

``storage:`` describes the filesystem (kind, endpoint, credentials); ``storage_path`` (runs,
checkpoints) and ``store_root`` (blocks and log) are paths on that filesystem, so one config
covers a local folder and an S3-compatible bucket alike.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field, replace
from typing import Any

import pyarrow.fs as pafs
import yaml

from distrainer.storage import StorageConfig, build_filesystem


def _check_keys(section: str, d: dict[str, Any], cls: Any) -> None:
    unknown = set(d) - set(cls.__dataclass_fields__)
    if unknown:
        raise ValueError(f"unknown keys in {section}: {sorted(unknown)}")


@dataclass
class LogConfig:
    W: int = 16
    passes: int = 2
    wait_poll_s: float = 1.0
    retention_segments: int = 4
    shuffle_buffer_segments: int = 1


@dataclass
class CheckpointConfig:
    policy: str = "any"
    every_k: int | None = 4
    time_budget_s: float | None = None
    time_poll_every: int = 1
    num_to_keep: int | None = 3

    def as_policy_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class LoaderConfig:
    prefetch: int = 2
    threads: int = 2


@dataclass
class ScalingSpec:
    num_workers: int | tuple[int, int] = 2
    resources_per_worker: dict[str, float] = field(default_factory=lambda: {"CPU": 1})
    use_gpu: bool = False
    elastic_resize_monitor_interval_s: float = 15.0

    @property
    def min_workers(self) -> int:
        return self.num_workers if isinstance(self.num_workers, int) else self.num_workers[0]

    @property
    def max_workers(self) -> int:
        return self.num_workers if isinstance(self.num_workers, int) else self.num_workers[1]

    @property
    def elastic(self) -> bool:
        return not isinstance(self.num_workers, int)


@dataclass
class FailureSpec:
    max_failures: int = 3


@dataclass
class DistrainerConfig:
    run_name: str = "run"
    storage_path: str = "runs"
    store_root: str = "blocks"
    seed: int = 1234
    storage: StorageConfig = field(default_factory=StorageConfig)
    log: LogConfig = field(default_factory=LogConfig)
    checkpoint: CheckpointConfig = field(default_factory=CheckpointConfig)
    loader: LoaderConfig = field(default_factory=LoaderConfig)
    scaling: ScalingSpec = field(default_factory=ScalingSpec)
    failure: FailureSpec = field(default_factory=FailureSpec)
    hooks: dict[str, Any] = field(default_factory=dict)
    train: dict[str, Any] = field(default_factory=dict)

    # ---- construction ----

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> DistrainerConfig:
        d = dict(d or {})
        _check_keys("config", d, cls)
        sections: dict[str, type] = {
            "log": LogConfig,
            "checkpoint": CheckpointConfig,
            "loader": LoaderConfig,
            "scaling": ScalingSpec,
            "failure": FailureSpec,
        }
        kwargs: dict[str, Any] = {}
        for key, value in d.items():
            if key in sections:
                sub = dict(value or {})
                _check_keys(key, sub, sections[key])
                if key == "scaling" and isinstance(sub.get("num_workers"), list):
                    sub["num_workers"] = tuple(int(x) for x in sub["num_workers"])
                kwargs[key] = sections[key](**sub)
            elif key == "storage":
                sub = dict(value or {})
                if "path" in sub:
                    raise ValueError(
                        "storage.path is not used; set storage_path and store_root at the top level"
                    )
                sub["path"] = d.get("storage_path", "runs")
                kwargs[key] = StorageConfig.from_dict(sub)
            else:
                kwargs[key] = value
        return cls(**kwargs)

    @classmethod
    def from_yaml(cls, path: str) -> DistrainerConfig:
        with open(path, encoding="utf-8") as f:
            return cls.from_dict(yaml.safe_load(f) or {})

    def __post_init__(self) -> None:
        self.validate()

    def asdict(self) -> dict[str, Any]:
        """Plain dict (YAML-safe: ``num_workers`` becomes a list, ``storage.path`` is dropped)."""
        d = asdict(self)
        d["storage"] = {k: v for k, v in self.storage.asdict().items() if k != "path"}
        if isinstance(self.scaling.num_workers, tuple):
            d["scaling"]["num_workers"] = list(self.scaling.num_workers)
        return d

    # ---- validation ----

    def allowed_world_sizes(self) -> range:
        return range(self.scaling.min_workers, self.scaling.max_workers + 1)

    def validate(self) -> None:
        s = self.scaling
        if isinstance(s.num_workers, tuple):
            if (
                len(s.num_workers) != 2
                or s.num_workers[0] <= 0
                or s.num_workers[0] > s.num_workers[1]
            ):
                raise ValueError(
                    f"scaling.num_workers must be int or [min, max], got {s.num_workers}"
                )
        elif s.num_workers <= 0:
            raise ValueError("scaling.num_workers must be positive")
        if self.log.W <= 0:
            raise ValueError("log.W must be positive")
        lcm = math.lcm(*self.allowed_world_sizes())
        if self.log.W % lcm != 0:
            raise ValueError(
                f"log.W={self.log.W} must be a multiple of every allowed world size "
                f"{list(self.allowed_world_sizes())} (lcm {lcm})"
            )
        if self.log.passes <= 0 or self.log.retention_segments < 0:
            raise ValueError("log.passes must be positive and log.retention_segments >= 0")
        if self.log.shuffle_buffer_segments <= 0 or self.log.wait_poll_s <= 0:
            raise ValueError("log.shuffle_buffer_segments and log.wait_poll_s must be positive")
        c = self.checkpoint
        if c.policy not in ("any", "every_k", "segment_end", "pass_end", "time", "never"):
            raise ValueError(f"unknown checkpoint.policy {c.policy!r}")
        if c.policy == "every_k" and c.every_k is None:
            raise ValueError("checkpoint.policy every_k needs checkpoint.every_k")
        if c.policy == "time" and c.time_budget_s is None:
            raise ValueError("checkpoint.policy time needs checkpoint.time_budget_s")
        if c.every_k is not None and c.every_k <= 0:
            raise ValueError("checkpoint.every_k must be positive or null")
        if c.time_budget_s is not None and c.time_budget_s <= 0:
            raise ValueError("checkpoint.time_budget_s must be positive or null")
        if c.num_to_keep is not None and c.num_to_keep <= 0:
            raise ValueError("checkpoint.num_to_keep must be positive or null")
        if self.loader.prefetch <= 0 or self.loader.threads <= 0:
            raise ValueError("loader.prefetch and loader.threads must be positive")
        if self.failure.max_failures < 0:
            raise ValueError("failure.max_failures must be >= 0")
        if not self.run_name or "/" in self.run_name:
            raise ValueError("run_name must be non-empty and contain no '/'")

    # ---- filesystems ----

    def _fs_for(self, path: str) -> tuple[pafs.FileSystem, str]:
        return build_filesystem(replace(self.storage, path=path))

    def runs_fs(self) -> tuple[pafs.FileSystem, str]:
        """``(fs, root)`` for checkpoints and Ray Train run state (``storage_path``)."""
        return self._fs_for(self.storage_path)

    def store_fs(self) -> tuple[pafs.FileSystem, str]:
        """``(fs, root)`` for blocks, the log and audit trails (``store_root``)."""
        return self._fs_for(self.store_root)


def load_config(path: str) -> DistrainerConfig:
    return DistrainerConfig.from_yaml(path)
