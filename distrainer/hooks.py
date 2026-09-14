"""distrainer.hooks: segment-end hooks (spec section 4).

A hook runs on rank 0 after the last step of a segment, before the barrier that releases the
other ranks; it may append the next segment(s) to the log, so hooks are writers. M4 ships the
re-mining hook; this module only fixes the protocol the trainer calls.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

if TYPE_CHECKING:
    from distrainer.ledger import Ledger
    from distrainer.log import BlockLog


@runtime_checkable
class SegmentHook(Protocol):
    def on_segment_end(self, model: Any, ledger: Ledger, log: BlockLog, ctx: Any) -> None: ...
