"""distrainer.hooks: segment-end hooks (spec section 4).

A hook runs on rank 0 after the last step of a segment, before the barrier that releases the
other ranks; it may append the next segment(s) to the log, so hooks are writers. The trainer
hands every hook the unwrapped model, a snapshot of the ledger, a ``BlockLog`` of its own
(separate from the loader's reader instance) and the ``TrainInfo`` of the step just completed.

Hooks come from the ``hooks:`` config section, ``{name: {entry: "pkg.module:factory", ...args}}``
or ``{name: "pkg.module:factory"}``; ``factory(config, **args)`` must return a ``SegmentHook``.
``build_hooks`` runs on rank 0 inside the worker, so a config hook's state never crosses the
wire (hooks passed to ``DistTrainer(hooks=...)`` are cloudpickled to every rank instead); the
driver resolves every entry up front so a typo fails before the cluster is engaged.

Segment ends are delivered *at least once*: after a failure or resize the new attempt resumes
from a checkpoint and re-runs the segment ends the dead attempt already served. A hook that
appends must therefore check ``log.ended()`` and ``log.has_segment(ledger.segment + 1)`` first
(``BlockLog.append`` would otherwise allocate a fresh sequence number and drift the log).
"""

from __future__ import annotations

import importlib
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

if TYPE_CHECKING:
    from distrainer.config import DistrainerConfig
    from distrainer.ledger import Ledger
    from distrainer.log import BlockLog


@runtime_checkable
class SegmentHook(Protocol):
    """``model`` is the unwrapped module, ``ledger`` a snapshot (``ledger.segment`` just ended),
    ``log`` rank 0's own writer instance, ``ctx`` the ``TrainInfo`` of the last step. Called on
    rank 0 only, at least once per segment end (see the module docstring)."""

    def on_segment_end(self, model: Any, ledger: Ledger, log: BlockLog, ctx: Any) -> None: ...


HookSpec = tuple[str, str, dict[str, Any]]


def hook_specs(hooks: dict[str, Any] | None) -> list[HookSpec]:
    """``(name, entry, args)`` per configured hook, validating the ``pkg.module:attr`` shape."""
    if hooks is None:
        return []
    if not isinstance(hooks, dict):
        raise ValueError("hooks must be a mapping of name -> entry or {entry, ...args}")
    out: list[HookSpec] = []
    for name, spec in hooks.items():
        entry: Any
        args: dict[str, Any]
        if isinstance(spec, str):
            entry, args = spec, {}
        elif isinstance(spec, dict):
            args = dict(spec)
            entry = args.pop("entry", None)
        else:
            raise ValueError(f"hooks.{name} must be an entry string or a mapping with 'entry'")
        if not isinstance(entry, str) or entry.count(":") != 1 or not all(entry.split(":")):
            raise ValueError(f"hooks.{name}: entry must look like 'pkg.module:attr', got {entry!r}")
        out.append((str(name), entry, args))
    return out


def load_entry(entry: str) -> Any:
    """The object named by ``pkg.module:attr`` (a dotted ``attr`` walks nested attributes)."""
    mod_name, _, attr = entry.partition(":")
    obj: Any = importlib.import_module(mod_name)
    for part in attr.split("."):
        obj = getattr(obj, part)
    return obj


def build_hooks(cfg: DistrainerConfig) -> list[SegmentHook]:
    """Instantiate the hooks of ``cfg.hooks`` in config order."""
    hooks: list[SegmentHook] = []
    for name, entry, args in hook_specs(cfg.hooks):
        hook = load_entry(entry)(cfg, **args)
        if not isinstance(hook, SegmentHook):
            raise TypeError(
                f"hooks.{name}: {entry} returned a {type(hook).__name__} without on_segment_end"
            )
        hooks.append(hook)
    return hooks
