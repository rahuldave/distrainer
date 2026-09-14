"""distrainer command line (spec section 6.4).

``inspect <uri>``      print the ledger from a checkpoint's metadata (no weights downloaded)
``export <uri> <dir>`` copy a checkpoint to a local directory
``resume <uri> --config cfg.yaml --entry module:function`` start a new run from a checkpoint
``log-ls <store>``     list the block log (meta, committed segments, _END)
``gc <store> --keep-from N`` delete segments below N (and their unreferenced blocks)

``<uri>`` is a local path or ``s3://bucket/prefix``; for S3-compatible stores set ``S3_ENDPOINT``
(and the credential variables) in the environment.
"""

from __future__ import annotations

import argparse
import importlib
import logging
import os
import sys
from collections.abc import Sequence
from typing import Any

from distrainer.config import load_config
from distrainer.log import BlockLog
from distrainer.storage import resolve


def _s3_options() -> dict[str, Any]:
    endpoint = os.environ.get("S3_ENDPOINT")
    return {"endpoint": endpoint, "region": os.environ.get("S3_REGION", "auto")} if endpoint else {}


def _checkpoint(uri: str) -> Any:
    from ray.train import Checkpoint

    fs, path = resolve(uri, create=False, **_s3_options())
    return Checkpoint(path=path, filesystem=fs)


def cmd_inspect(args: argparse.Namespace) -> int:
    from distrainer.trainer import CheckpointIO

    ckpt = _checkpoint(args.uri)
    meta = ckpt.get_metadata()
    ledger = CheckpointIO.read_ledger(ckpt)
    print(f"checkpoint: {args.uri}")
    print(f"ledger: {ledger.asdict()}")
    print(f"done positions in segment {ledger.segment}: {ledger.done_positions()}")
    if "distrainer" in meta:
        print(f"written by distrainer {meta['distrainer']}")
    return 0


def cmd_export(args: argparse.Namespace) -> int:
    path = _checkpoint(args.uri).to_directory(args.dir)
    print(f"exported to {path}")
    return 0


def cmd_resume(args: argparse.Namespace) -> int:
    """Start a new run from ``uri``. ``--entry pkg.module:function`` must return
    ``(train_step, build_model)`` for the config."""
    from distrainer.trainer import DistTrainer

    cfg = load_config(args.config)
    if args.run_name:
        cfg.run_name = args.run_name
    if args.seed is not None:
        cfg.seed = args.seed
    mod_name, _, fn_name = args.entry.partition(":")
    factory = getattr(importlib.import_module(mod_name), fn_name)
    train_step, build_model = factory(cfg)
    os.environ.setdefault("RAY_ENABLE_UV_RUN_RUNTIME_ENV", "0")
    import ray

    ray.init(address=cfg.ray_address, ignore_reinit_error=True, logging_level=logging.ERROR)
    result = DistTrainer(
        train_step, build_model, cfg, resume_from_checkpoint=_checkpoint(args.uri)
    ).fit()
    print(f"run finished: {result.path}")
    print(f"final metrics: {result.metrics}")
    return 0


def cmd_log_ls(args: argparse.Namespace) -> int:
    fs, root = resolve(args.store, create=False, **_s3_options())
    log = BlockLog(fs, root)
    if not log.exists():
        print(f"no block log under {args.store}")
        return 1
    meta = log.meta()
    seqs = log.committed_seqs()
    print(f"log: {log.log_dir}")
    print(f"W={meta.W} seed={meta.seed} schema_version={meta.schema_version} by {meta.created_by}")
    print(f"segments: {len(seqs)}" + (f" ({seqs[0]}..{seqs[-1]})" if seqs else ""))
    print(f"ended: {log.ended()}")
    if args.verbose:
        for seg in log.segments():
            ids = [b.block_id for b in seg.blocks]
            head = ", ".join(ids[:4]) + (", ..." if len(ids) > 4 else "")
            print(
                f"  {seg.seq:08d} pass={seg.pass_idx} positions {seg.positions().start}.."
                f"{seg.positions().stop - 1}: {head}"
            )
    return 0


def cmd_gc(args: argparse.Namespace) -> int:
    fs, root = resolve(args.store, create=False, **_s3_options())
    log = BlockLog.open(fs, root)
    deleted = log.gc(args.keep_from, delete_blocks=not args.keep_blocks)
    print(f"deleted segments: {deleted}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(prog="distrainer", description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("inspect", help="print the ledger of a checkpoint")
    p.add_argument("uri")
    p.set_defaults(fn=cmd_inspect)
    p = sub.add_parser("export", help="copy a checkpoint to a local directory")
    p.add_argument("uri")
    p.add_argument("dir")
    p.set_defaults(fn=cmd_export)
    p = sub.add_parser("resume", help="start a new run from a checkpoint")
    p.add_argument("uri")
    p.add_argument("--config", required=True)
    p.add_argument(
        "--entry", required=True, help="pkg.module:function returning (train_step, build_model)"
    )
    p.add_argument("--run-name", default=None)
    p.add_argument("--seed", type=int, default=None, dest="seed")
    p.set_defaults(fn=cmd_resume)
    p = sub.add_parser("log-ls", help="list a block log")
    p.add_argument("store")
    p.add_argument("-v", "--verbose", action="store_true")
    p.set_defaults(fn=cmd_log_ls)
    p = sub.add_parser("gc", help="delete segments below --keep-from")
    p.add_argument("store")
    p.add_argument("--keep-from", type=int, required=True)
    p.add_argument("--keep-blocks", action="store_true", help="delete segment files only")
    p.set_defaults(fn=cmd_gc)
    return ap


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(sys.argv[1:] if argv is None else list(argv))
    return int(args.fn(args))


if __name__ == "__main__":
    raise SystemExit(main())
