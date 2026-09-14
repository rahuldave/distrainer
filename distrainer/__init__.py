"""distrainer: block-native distributed training on Ray Train.

See docs/introduction.md for the ideas and docs/distrainer-spec.md for the contract.
"""

from distrainer.audit import AuditRecord, AuditWriter, read_audit
from distrainer.block import BlockRef, read_block, write_block
from distrainer.config import DistrainerConfig, load_config
from distrainer.ledger import Ledger
from distrainer.loader import LaneLoader
from distrainer.log import BlockLog, LogMeta, Segment
from distrainer.planner import lane, resume_start
from distrainer.policy import (
    Any,
    CheckpointPolicy,
    EveryKSteps,
    PassEnd,
    SegmentEnd,
    StepContext,
    TimeBudget,
    build_policy,
)
from distrainer.storage import StorageConfig, build_filesystem, resolve
from distrainer.writer import BatchWriter, StreamingWriter

__version__ = "0.0.1"

__all__ = [
    "Any",
    "AuditRecord",
    "AuditWriter",
    "BatchWriter",
    "BlockLog",
    "BlockRef",
    "CheckpointPolicy",
    "DistrainerConfig",
    "EveryKSteps",
    "LaneLoader",
    "Ledger",
    "LogMeta",
    "PassEnd",
    "Segment",
    "SegmentEnd",
    "StepContext",
    "StorageConfig",
    "StreamingWriter",
    "TimeBudget",
    "__version__",
    "build_filesystem",
    "build_policy",
    "lane",
    "load_config",
    "read_audit",
    "read_block",
    "resolve",
    "resume_start",
    "write_block",
]
