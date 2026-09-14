"""Assertions over distrainer audit trails (spec section 10). Pure functions plus a small CLI.

Every check returns a list of problem strings; an empty list means the scenario passed.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from collections.abc import Sequence
from typing import Any

from distrainer.audit import AuditRecord, read_audit
from distrainer.storage import join, list_names, read_bytes, resolve, s3_options_from_env


def by_attempt(records: Sequence[AuditRecord]) -> dict[int, list[AuditRecord]]:
    out: dict[int, list[AuditRecord]] = defaultdict(list)
    for r in records:
        out[r.attempt].append(r)
    return dict(out)


def summarize(records: Sequence[AuditRecord]) -> str:
    if not records:
        return "audit: no records"
    parts = []
    for attempt, recs in sorted(by_attempt(records).items()):
        ranks = sorted({r.rank for r in recs})
        n = {r.world_size for r in recs}
        segs = sorted({r.segment for r in recs})
        parts.append(
            f"attempt {attempt}: world_size {sorted(n)}, ranks {ranks}, {len(recs)} blocks, "
            f"segments {segs[0]}..{segs[-1]}, positions {min(r.position for r in recs)}.."
            f"{max(r.position for r in recs)}"
        )
    return "audit: " + "; ".join(parts)


def check_dealing(records: Sequence[AuditRecord], W: int) -> list[str]:
    """Within one attempt: rank r consumed position seq*W + step*n + r for every record."""
    problems = []
    for r in records:
        expected = r.segment * W + r.step * r.world_size + r.rank
        if r.position != expected:
            problems.append(
                f"attempt {r.attempt} rank {r.rank} segment {r.segment} step {r.step}: "
                f"position {r.position}, expected {expected}"
            )
    return problems


def check_s1(records: Sequence[AuditRecord], W: int) -> list[str]:
    """Happy path: one attempt, every segment fully consumed exactly once, dealing rule holds,
    every rank consumed the same number of blocks."""
    problems: list[str] = []
    if not records:
        return ["no audit records"]
    attempts = by_attempt(records)
    if len(attempts) != 1:
        problems.append(f"expected one attempt, found {sorted(attempts)}")
    problems += check_dealing(records, W)
    per_segment: dict[int, list[int]] = defaultdict(list)
    for r in records:
        per_segment[r.segment].append(r.position)
    for seq, positions in sorted(per_segment.items()):
        expected = list(range(seq * W, (seq + 1) * W))
        if sorted(positions) != expected:
            missing = sorted(set(expected) - set(positions))
            extra = sorted(p for p in positions if p not in expected)
            dupes = sorted({p for p in positions if positions.count(p) > 1})
            problems.append(
                f"segment {seq}: missing {missing[:8]}, extra {extra[:8]}, duplicated {dupes[:8]}"
            )
    counts = defaultdict(int)
    for r in records:
        counts[r.rank] += 1
    if len(set(counts.values())) > 1:
        problems.append(f"ranks consumed unequal block counts: {dict(sorted(counts.items()))}")
    return problems


def check_s7(a: Sequence[AuditRecord], b: Sequence[AuditRecord]) -> list[str]:
    """Determinism: two runs with the same seed have identical per-rank sequences."""

    def key(recs: Sequence[AuditRecord]) -> dict[int, list[tuple[int, int, int, str]]]:
        out: dict[int, list[tuple[int, int, int, str]]] = defaultdict(list)
        for r in recs:
            out[r.rank].append((r.segment, r.step, r.position, r.block_id))
        return dict(out)

    ka, kb = key(a), key(b)
    problems = []
    if set(ka) != set(kb):
        problems.append(f"rank sets differ: {sorted(ka)} vs {sorted(kb)}")
    for rank in sorted(set(ka) & set(kb)):
        if ka[rank] != kb[rank]:
            first = next(
                (i for i, (x, y) in enumerate(zip(ka[rank], kb[rank], strict=False)) if x != y),
                min(len(ka[rank]), len(kb[rank])),
            )
            problems.append(f"rank {rank}: sequences diverge at index {first}")
    return problems


def check_resume(records: Sequence[AuditRecord], W: int, start_position: int) -> list[str]:
    """A run resumed from a ledger: positions are exactly ``start_position`` up to the end of the
    last segment touched, each once, dealt by the assignment rule (S9/S10 and the CLI resume)."""
    if not records:
        return ["no audit records"]
    problems = check_dealing(records, W)
    positions = sorted(r.position for r in records)
    last_seg = max(r.segment for r in records)
    expected = list(range(start_position, (last_seg + 1) * W))
    if positions != expected:
        missing = sorted(set(expected) - set(positions))
        extra = sorted(set(positions) - set(expected))
        dupes = sorted({p for p in positions if positions.count(p) > 1})
        problems.append(
            f"resumed run: missing {missing[:8]}, extra {extra[:8]}, duplicated {dupes[:8]}"
        )
    return problems


def checkpoint_ledgers(run_uri: str) -> dict[str, dict]:
    """``{checkpoint dir name: ledger dict}`` from each checkpoint's ``.metadata.json``."""
    fs, run_dir = resolve(run_uri, create=False, **s3_options_from_env())
    out: dict[str, dict] = {}
    selector_names = list_names(fs, run_dir)
    # checkpoints are directories, so list the run dir with a selector instead
    import pyarrow.fs as pafs

    infos = fs.get_file_info(pafs.FileSelector(run_dir, recursive=False, allow_not_found=True))
    names = sorted(i.path.rsplit("/", 1)[-1] for i in infos if i.type == pafs.FileType.Directory)
    for name in names + [n for n in selector_names if n.startswith("checkpoint_")]:
        if not name.startswith("checkpoint_"):
            continue
        meta_path = join(run_dir, name, ".metadata.json")
        try:
            meta = json.loads(read_bytes(fs, meta_path).decode())
        except Exception:
            continue
        if "ledger" in meta:
            out[name] = meta["ledger"]
    return out


def check_s5(run_uri: str, W: int, every_k: int | None, expected_count: int | None) -> list[str]:
    """Checkpoint cadence: each ledger cursor is a multiple of k or the segment end; count."""
    ledgers = checkpoint_ledgers(run_uri)
    problems = []
    if not ledgers:
        return [f"no checkpoints with ledger metadata under {run_uri}"]
    for name, led in sorted(ledgers.items()):
        cursor, n = int(led["cursor"]), int(led["world_size"])
        at_end = cursor == W // n
        ok = at_end or (every_k is not None and cursor % every_k == 0)
        if not ok:
            problems.append(f"{name}: cursor {cursor} is neither a multiple of {every_k} nor W/n")
        expected_name = (
            f"checkpoint_g{int(led['segment']):06d}_p{cursor * n:06d}"
            f"_n{n:02d}_a{int(led.get('run_attempt', 0)):02d}"
        )
        if name != expected_name:
            problems.append(f"{name}: directory name does not match ledger {led}")
    if expected_count is not None and len(ledgers) != expected_count:
        problems.append(f"expected {expected_count} checkpoints, found {len(ledgers)}")
    return problems


def expected_checkpoints(
    n_segments: int, W: int, n: int, every_k: int | None, segment_end: bool = True
) -> int:
    """Checkpoints a run of ``n_segments`` writes: every ``every_k`` steps within a segment and,
    if ``segment_end``, at the last step (the ``any`` policy; bare ``every_k`` has no end)."""
    steps = W // n
    per_segment = sum(
        1
        for c in range(1, steps + 1)
        if (segment_end and c == steps) or (every_k and c % every_k == 0)
    )
    return n_segments * per_segment


def expected_reports(n_segments: int, W: int, n: int, checkpoint_cfg: Any) -> int:
    """``ray.train.report`` calls per rank for a completed run with the default cadence: one per
    checkpoint plus a final metrics-only report when the last step is not a checkpoint."""
    every_k = checkpoint_cfg.every_k if checkpoint_cfg.policy in ("any", "every_k") else None
    segment_end = checkpoint_cfg.policy in ("any", "segment_end", "pass_end")
    if checkpoint_cfg.report_every_step:
        return n_segments * (W // n)
    count = expected_checkpoints(n_segments, W, n, every_k, segment_end)
    steps = W // n
    last_is_checkpoint = segment_end or (every_k is not None and steps % every_k == 0)
    return count + (0 if last_is_checkpoint else 1)


def check_report_count(n_reports: int, expected: int) -> list[str]:
    return [] if n_reports == expected else [f"expected {expected} reports, Result has {n_reports}"]


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--store-root", required=True)
    ap.add_argument("--run-name", required=True)
    ap.add_argument("--scenario", default="S1", choices=["S1", "S5", "S7", "resume"])
    ap.add_argument("--start-position", type=int, default=None, help="resume: first position")
    ap.add_argument("--W", type=int, required=True)
    ap.add_argument("--every-k", type=int, default=None)
    ap.add_argument("--run-uri", default=None, help="S5: run directory holding checkpoints")
    ap.add_argument("--expected-checkpoints", type=int, default=None)
    ap.add_argument("--other-run-name", default=None, help="S7: run to compare against")
    args = ap.parse_args(argv)
    fs, root = resolve(args.store_root, create=False, **s3_options_from_env())
    records = read_audit(fs, root, args.run_name)
    print(summarize(records))
    if args.scenario == "S1":
        problems = check_s1(records, args.W)
    elif args.scenario == "S5":
        problems = check_s5(args.run_uri, args.W, args.every_k, args.expected_checkpoints)
    elif args.scenario == "resume":
        if args.start_position is None:
            ap.error("--start-position is required for the resume check")
        problems = check_resume(records, args.W, args.start_position)
    else:
        other = read_audit(fs, root, args.other_run_name)
        problems = check_s7(records, other)
    for p in problems:
        print(f"{args.scenario} FAIL: {p}")
    print(f"{args.scenario} {'PASS' if not problems else 'FAIL'}")
    return 1 if problems else 0


if __name__ == "__main__":
    raise SystemExit(main())
