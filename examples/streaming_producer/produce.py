"""streaming_producer: a writer process that streams hello_blocks-style blocks into a log (S11).

Run it alongside the trainer. It creates the log, pushes blocks one by one through
``StreamingWriter`` (a segment commits every ``W`` blocks, or a sample of ``W`` out of
``shuffle_buffer_segments * W`` buffered blocks), sleeps ``--sleep-s`` after every committed
segment, and ends the log after ``--segments`` segments (``close`` writes ``_END``). Blocks have
the hello_blocks schema (``x_0..x_{d-1}``, ``y``) so ``examples/hello_blocks/train.py`` trains on
them unchanged.

    python examples/streaming_producer/produce.py \\
        --config examples/hello_blocks/harness-stream.yaml --segments 8 --sleep-s 6

``--config`` is the trainer's YAML: ``store_root``, ``seed``, ``log.W``,
``log.shuffle_buffer_segments``, ``train.rows_per_block`` and ``train.features`` are taken from
it so both sides agree; every one of them can be overridden on the command line. The store must
not hold a log yet (the scenario runner wipes it first): a producer cannot resume a stream.
"""

from __future__ import annotations

import argparse
import sys
import time
from collections.abc import Callable
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.fs as pafs

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from distrainer.block import write_block  # noqa: E402
from distrainer.config import load_config  # noqa: E402
from distrainer.log import BlockLog, Segment  # noqa: E402
from distrainer.storage import resolve, s3_options_from_env  # noqa: E402
from distrainer.writer import StreamingWriter  # noqa: E402
from examples.hello_blocks.make_blocks import feature_columns  # noqa: E402


def produce(
    fs: pafs.FileSystem,
    root: str,
    *,
    W: int,
    seed: int,
    segments: int,
    sleep_s: float = 0.0,
    shuffle_buffer: int = 1,
    rows: int = 32,
    features: int = 8,
    progress: Callable[[str], None] | None = None,
    on_created: Callable[[], None] | None = None,
) -> list[Segment]:
    """Stream ``segments * W`` blocks into a new log under ``root`` and end it.

    Exactly ``segments`` segments come out for any ``shuffle_buffer``: the writer holds up to
    ``shuffle_buffer * W`` blocks and ``close`` flushes the full segments still buffered. The
    producer sleeps ``sleep_s`` after each segment committed while pushing (the ``shuffle_buffer
    - 1`` segments flushed at the end commit back to back). Block ids are ``p<arrival index>``;
    every commit is reported through ``progress`` as ``segment N committed at <time>``.
    ``on_created`` runs once the log exists and before the first block (a test uses it to let
    the trainer open the log first).
    """
    if segments <= 0 or W <= 0:
        raise ValueError("segments and W must be positive")
    say = progress or (lambda _msg: None)
    if BlockLog(fs, root).exists():
        raise FileExistsError(f"a log already exists under {root}; a stream cannot be resumed")
    log = BlockLog.create(fs, root, W=W, seed=seed, created_by="streaming_producer")
    if on_created is not None:
        on_created()
    writer = StreamingWriter(log, shuffle_buffer_segments=shuffle_buffer)
    rng = np.random.default_rng(seed)
    w_true = rng.normal(size=features)
    committed: list[Segment] = []
    for i in range(segments * W):
        x = rng.normal(size=(rows, features)).astype(np.float32)
        y = (x @ w_true + 0.05 * rng.normal(size=rows)).astype(np.float32)
        cols = {name: pa.array(x[:, j]) for j, name in enumerate(feature_columns(features))}
        cols["y"] = pa.array(y)
        ref = write_block(fs, root, f"p{i:06d}", pa.table(cols), meta={"produced_at": time.time()})
        seg = writer.push(ref)
        if seg is not None:
            committed.append(seg)
            say(f"segment {seg.seq} committed at {seg.meta['created_at']:.3f}")
            if sleep_s > 0 and i + 1 < segments * W:
                time.sleep(sleep_s)
    for seg in writer.close(tail="drop"):  # the (shuffle_buffer - 1) segments still buffered
        committed.append(seg)
        say(f"segment {seg.seq} committed at {seg.meta['created_at']:.3f}")
    say(f"log ended after {len(committed)} segments")
    return committed


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--config", default=None, help="trainer YAML the defaults are read from")
    ap.add_argument("--store", default=None, help="store root (path or s3://bucket/prefix)")
    ap.add_argument("--W", type=int, default=None)
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--segments", type=int, default=8)
    ap.add_argument("--sleep-s", type=float, default=5.0)
    ap.add_argument("--shuffle-buffer", type=int, default=None, help="log.shuffle_buffer_segments")
    ap.add_argument("--rows", type=int, default=None, help="train.rows_per_block")
    ap.add_argument("--features", type=int, default=None, help="train.features")
    args = ap.parse_args(argv)
    if args.config:
        cfg = load_config(args.config)
        fs, root = cfg.store_fs()
        defaults = {
            "W": cfg.log.W,
            "seed": cfg.seed,
            "shuffle_buffer": cfg.log.shuffle_buffer_segments,
            "rows": int(cfg.train.get("rows_per_block", 32)),
            "features": int(cfg.train.get("features", 8)),
        }
    else:
        if not args.store:
            ap.error("--store is required without --config")
        fs, root = None, None
        defaults = {"W": 24, "seed": 7, "shuffle_buffer": 1, "rows": 32, "features": 8}
    if args.store:
        fs, root = resolve(args.store, **s3_options_from_env())
    assert fs is not None and root is not None
    pick = lambda name: getattr(args, name) if getattr(args, name) is not None else defaults[name]  # noqa: E731
    produce(
        fs,
        root,
        W=int(pick("W")),
        seed=int(pick("seed")),
        segments=args.segments,
        sleep_s=args.sleep_s,
        shuffle_buffer=int(pick("shuffle_buffer")),
        rows=int(pick("rows")),
        features=int(pick("features")),
        progress=lambda msg: print(msg, flush=True),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
