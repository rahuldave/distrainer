"""distrainer.ledger: the data position saved in every checkpoint.

Spec section 2.2. ``Ledger(segment, cursor, world_size)`` means: at this checkpoint every rank had
consumed exactly positions ``[segment*W, segment*W + cursor*world_size)``. ``pass_idx`` is copied
from the segment for hooks and policies; ``run_attempt`` counts Ray Train restarts.
"""

from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass
from typing import Any

LEDGER_FILENAME = "ledger.json"


@dataclass
class Ledger:
    segment: int = 0
    cursor: int = 0
    world_size: int = 0
    pass_idx: int = 0
    run_attempt: int = 0

    def done_positions(self) -> int:
        """Positions of ``segment`` completed at this checkpoint."""
        return self.cursor * self.world_size

    def asdict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> Ledger:
        known = set(cls.__dataclass_fields__)
        return cls(**{k: int(v) for k, v in d.items() if k in known})

    def to_json(self) -> str:
        return json.dumps(self.asdict(), sort_keys=True)

    @classmethod
    def from_json(cls, text: str) -> Ledger:
        return cls.from_dict(json.loads(text))

    def save(self, dir: str) -> str:
        path = os.path.join(dir, LEDGER_FILENAME)
        with open(path, "w", encoding="utf-8") as f:
            f.write(self.to_json())
        return path

    @classmethod
    def load(cls, dir: str) -> Ledger:
        with open(os.path.join(dir, LEDGER_FILENAME), encoding="utf-8") as f:
            return cls.from_json(f.read())
