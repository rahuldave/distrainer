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


def check_s6(
    records: Sequence[AuditRecord],
    segments: Sequence[Any],
    W: int,
    initial_segments: int,
    expected_segments: int | None = None,
    writer: str = "remine",
    clock_tolerance_s: float = 0.05,
) -> list[str]:
    """Segment hook (streaming through re-mining): S1 holds; every segment from
    ``initial_segments`` on was written by the hook (``meta.writer``, ``mined_after_segment``
    is the previous one), committed (``meta.created_at``) after rank 0's last record of the
    previous segment and before the first record that consumed it, and the records of such a
    segment name exactly its blocks, none of which the base corpus (the segments below
    ``initial_segments``) had. ``segments`` are the log's ``Segment`` objects; the timestamps
    are all ``time.time()`` on the containers' host, ``clock_tolerance_s`` absorbs skew."""
    problems = check_s1(records, W)
    if not records:
        return problems
    by_seq = {s.seq: s for s in segments}
    base_ids = {b.block_id for s in segments if s.seq < initial_segments for b in s.blocks}
    first_ts: dict[int, float] = {}
    last_ts_rank0: dict[int, float] = {}
    consumed: dict[int, set[str]] = defaultdict(set)
    for r in records:
        first_ts[r.segment] = min(first_ts.get(r.segment, r.ts), r.ts)
        if r.rank == 0:
            last_ts_rank0[r.segment] = max(last_ts_rank0.get(r.segment, r.ts), r.ts)
        consumed[r.segment].add(r.block_id)
    if expected_segments is not None and sorted(by_seq) != list(range(expected_segments)):
        problems.append(f"log has segments {sorted(by_seq)}, expected 0..{expected_segments - 1}")
    for seq, ts in sorted(first_ts.items()):
        seg = by_seq.get(seq)
        if seg is None:
            problems.append(f"segment {seq} was consumed but is not in the log")
            continue
        if seq < initial_segments:
            continue
        if seg.meta.get("writer") != writer or seg.meta.get("mined_after_segment") != seq - 1:
            problems.append(f"segment {seq}: not written by the hook after segment {seq - 1}")
        created = seg.meta.get("created_at")
        if created is None or float(created) > ts + clock_tolerance_s:
            problems.append(
                f"segment {seq}: committed at {created}, first consumed at {ts} (hook too late)"
            )
        elif (
            seq - 1 in last_ts_rank0 and float(created) + clock_tolerance_s < last_ts_rank0[seq - 1]
        ):
            problems.append(
                f"segment {seq}: committed at {created} before rank 0 finished segment "
                f"{seq - 1} at {last_ts_rank0[seq - 1]} (not mined at the segment end)"
            )
        ids = {b.block_id for b in seg.blocks}
        if consumed[seq] != ids:
            problems.append(f"segment {seq}: consumed {sorted(consumed[seq])[:4]}... != its blocks")
        reused = consumed[seq] & base_ids
        if reused:
            problems.append(f"segment {seq}: re-used base corpus blocks {sorted(reused)[:4]}")
    return problems


def segment_starts(records: Sequence[AuditRecord]) -> dict[int, float]:
    """Earliest audit ``ts`` per segment."""
    first: dict[int, float] = {}
    for r in records:
        first[r.segment] = min(first.get(r.segment, r.ts), r.ts)
    return first


def check_s11(
    records: Sequence[AuditRecord],
    W: int,
    committed_at: dict[int, float],
    min_gap_s: float,
    expected_segments: int | None = None,
    clock_tolerance_s: float = 0.05,
) -> list[str]:
    """Streaming producer: S1 holds; no segment was consumed before the producer committed it
    (``committed_at``: seq -> commit time, parsed from the producer's output so gc'd segments
    count too); every segment up to ``expected_segments`` was consumed (the run ended at
    ``_END``); and the ranks waited for the producer at least once: some segment was committed
    only after the trainer had finished the previous one (the direct signal), and some two
    consecutive segments started at least ``min_gap_s`` apart (the producer's sleep minus the
    poll interval and a tolerance; the steady state of a trainer faster than its producer).
    ``clock_tolerance_s`` absorbs skew between the producer's and the ranks' clocks."""
    problems = check_s1(records, W)
    if not records:
        return problems
    starts = segment_starts(records)
    ends: dict[int, float] = {}
    for r in records:
        ends[r.segment] = max(ends.get(r.segment, r.ts), r.ts)
    seqs = sorted(starts)
    if expected_segments is not None and seqs != list(range(expected_segments)):
        problems.append(f"consumed segments {seqs}, expected 0..{expected_segments - 1}")
    for seq in seqs:
        if seq not in committed_at:
            problems.append(f"segment {seq}: no commit time known")
        elif committed_at[seq] > starts[seq] + clock_tolerance_s:
            problems.append(
                f"segment {seq}: consumed at {starts[seq]:.3f} before its commit at "
                f"{committed_at[seq]:.3f}"
            )
    waited = [
        seq
        for seq in seqs[1:]
        if seq in committed_at and committed_at[seq] > ends[seq - 1] + clock_tolerance_s
    ]
    if not waited:
        problems.append(
            "ranks never waited for the producer: no segment was committed after the trainer "
            "had finished the previous one"
        )
    gaps = [starts[b] - starts[a] for a, b in zip(seqs, seqs[1:], strict=False)]
    if gaps and max(gaps) < min_gap_s:
        problems.append(
            f"segment start gaps {[round(g, 2) for g in gaps]} all below {min_gap_s:.2f}s: the "
            "trainer never reached the producer's pace"
        )
    return problems


def check_retention(
    kept_seqs: Sequence[int],
    block_files: set[str],
    kept_locators: set[str],
    last_checkpoint_segment: int,
    retention_segments: int,
) -> list[str]:
    """After a run with ``log.gc``: the log starts exactly ``retention_segments`` behind the
    last checkpoint's segment (or at 0), is contiguous, and the block directory holds exactly
    the blocks the kept segments reference (deleted segments' blocks are gone, kept ones intact).
    """
    problems: list[str] = []
    seqs = sorted(kept_seqs)
    if not seqs:
        return ["no segments left in the log"]
    expected_first = max(0, last_checkpoint_segment - retention_segments)
    if seqs[0] != expected_first:
        problems.append(
            f"log starts at segment {seqs[0]}, expected {expected_first} "
            f"(last checkpoint segment {last_checkpoint_segment} - retention {retention_segments})"
        )
    if seqs != list(range(seqs[0], seqs[-1] + 1)):
        problems.append(f"kept segments are not contiguous: {seqs}")
    missing = sorted(kept_locators - block_files)
    orphans = sorted(block_files - kept_locators)
    if missing:
        problems.append(f"blocks of kept segments missing: {missing[:5]}")
    if orphans:
        problems.append(f"blocks of deleted segments still present: {orphans[:5]}")
    return problems


def check_s8(
    records: Sequence[AuditRecord],
    W: int,
    n_reports: int,
    ledgers: dict[str, dict],
    poll_every: int = 1,
) -> list[str]:
    """Time-budget policy: one attempt with S1 dealing (the run finishing at all proves every
    rank called ``report`` equally often, Ray Train v2 enforces it inside ``report``); the budget
    fired more than once; rank 0's ``reports`` count equals the number of checkpoints written,
    plus one when the last step was a metrics-only report (so the run must keep every
    checkpoint: ``num_to_keep: null``); consecutive checkpoints are at least ``poll_every``
    steps apart (the policy only decides on poll steps); each ledger is a valid step boundary
    and its directory name matches."""
    problems = check_s1(records, W)
    if len(ledgers) < 2:
        problems.append(f"expected at least 2 time-budget checkpoints, found {len(ledgers)}")
    if n_reports not in (len(ledgers), len(ledgers) + 1):
        problems.append(f"rank 0 reported {n_reports} times for {len(ledgers)} checkpoints")
    positions = sorted(
        int(led["segment"]) * W + int(led["cursor"]) * int(led["world_size"])
        for led in ledgers.values()
    )
    for a, b in zip(positions, positions[1:], strict=False):
        n = next(int(led["world_size"]) for led in ledgers.values())
        if b - a < poll_every * n:
            problems.append(
                f"checkpoints at positions {a} and {b} are closer than {poll_every} poll steps"
            )
    for name, led in sorted(ledgers.items()):
        cursor, n = int(led["cursor"]), int(led["world_size"])
        if cursor <= 0 or cursor * n > W:
            problems.append(f"{name}: cursor {cursor} at world size {n} is not a step boundary")
        expected_name = (
            f"checkpoint_g{int(led['segment']):06d}_p{cursor * n:06d}"
            f"_n{n:02d}_a{int(led.get('run_attempt', 0)):02d}"
        )
        if name != expected_name:
            problems.append(f"{name}: directory name does not match ledger {led}")
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


def resume_start_position(segment: int, positions_done: int, W: int, world_size_now: int) -> int:
    """First position a run resumed from a ledger with ``positions_done`` positions of ``segment``
    consumes with ``world_size_now`` ranks (the library's own resume rule)."""
    from distrainer.ledger import Ledger
    from distrainer.planner import resume_start

    ledger = Ledger(segment=segment, cursor=positions_done, world_size=1)
    seg, step = resume_start(ledger, world_size_now, W=W)
    return seg * W + step * world_size_now


def check_resume(
    records: Sequence[AuditRecord],
    W: int,
    start_position: int | None = None,
    ledger: tuple[int, int] | None = None,
    expected_segments: int | None = None,
) -> list[str]:
    """A run resumed from a ledger: positions are exactly the resume start up to the end of the
    last segment touched, each once, dealt by the assignment rule (S9/S10 and the CLI resume).
    Give either ``start_position`` or ``ledger=(segment, positions_done)``; with the ledger the
    start is derived with the world size the resumed run actually had (round-down rule)."""
    if not records:
        return ["no audit records"]
    problems = check_dealing(records, W)
    first_attempt = min(r.attempt for r in records)
    n_now = next(r.world_size for r in records if r.attempt == first_attempt)
    if ledger is not None:
        start_position = resume_start_position(ledger[0], ledger[1], W, n_now)
    if start_position is None:
        return ["check_resume needs start_position or ledger"]
    positions = sorted(r.position for r in records)
    last_seg = max(r.segment for r in records)
    if expected_segments is not None and last_seg + 1 != expected_segments:
        problems.append(
            f"resumed run touched {last_seg + 1} segments, expected {expected_segments}"
        )
    expected = list(range(start_position, (last_seg + 1) * W))
    if positions != expected:
        missing = sorted(set(expected) - set(positions))
        extra = sorted(set(positions) - set(expected))
        dupes = sorted({p for p in positions if positions.count(p) > 1})
        problems.append(
            f"resumed run: missing {missing[:8]}, extra {extra[:8]}, duplicated {dupes[:8]}"
        )
    return problems


def check_recovery(
    records: Sequence[AuditRecord],
    W: int,
    every_k: int | None = None,
    expected_world_sizes: Sequence[int] | None = None,
    min_attempts: int = 2,
    expected_segments: int | None = None,
) -> list[str]:
    """Failure / resize scenarios (S2, S3, S4): several attempts, one log.

    Per attempt the dealing rule holds for that attempt's world size. Attempt ``i`` starts at a
    position that is ``<=`` the position after the last one attempt ``i-1`` consumed (no gap),
    lies on a step boundary of the new world size, and the replayed tail is bounded by
    ``2 * every_k * n_old + n_new`` blocks: one checkpoint interval since the last checkpoint the
    controller registered, plus one more that may still have been in flight (ASYNC upload and
    the controller's poll) when the group died, plus the round-down remainder. Over all attempts
    the union of positions covers every segment touched, from position 0 to the end of the last
    segment, with no gaps.
    """
    if not records:
        return ["no audit records"]
    problems = check_dealing(records, W)
    attempts = by_attempt(records)
    ids = sorted(attempts)
    if len(ids) < min_attempts:
        problems.append(f"expected at least {min_attempts} attempts, found {ids}")
    sizes = []
    for a in ids:
        recs = attempts[a]
        n = {r.world_size for r in recs}
        if len(n) != 1:
            problems.append(f"attempt {a}: mixed world sizes {sorted(n)}")
        sizes.append(min(n))
        seen = [r.position for r in recs]
        dupes = sorted({p for p in seen if seen.count(p) > 1})
        if dupes:
            problems.append(f"attempt {a}: positions consumed twice within the attempt {dupes[:8]}")
    if expected_world_sizes is not None and sizes != list(expected_world_sizes):
        problems.append(f"world sizes per attempt {sizes}, expected {list(expected_world_sizes)}")
    for prev, cur in zip(ids, ids[1:], strict=False):
        p_recs, c_recs = attempts[prev], attempts[cur]
        n_old, n_new = p_recs[0].world_size, c_recs[0].world_size
        last_prev = max(r.position for r in p_recs)
        first_cur = min(r.position for r in c_recs)
        seg = first_cur // W
        if first_cur > last_prev + 1:
            problems.append(f"attempt {cur} starts at {first_cur}, gap after {last_prev}")
        if (first_cur - seg * W) % n_new != 0:
            problems.append(
                f"attempt {cur} starts at {first_cur}: not a step boundary for n={n_new}"
            )
        replayed = sum(1 for r in p_recs if r.position >= first_cur)
        bound = 2 * (every_k or 1) * n_old + n_new
        if replayed > bound:
            problems.append(f"attempt {cur} replays {replayed} positions (> {bound})")
    positions = sorted({r.position for r in records})
    last_seg = max(r.segment for r in records)
    if expected_segments is not None and last_seg + 1 != expected_segments:
        problems.append(f"run touched {last_seg + 1} segments, expected {expected_segments}")
    if positions != list(range(0, (last_seg + 1) * W)):
        missing = sorted(set(range(0, (last_seg + 1) * W)) - set(positions))
        problems.append(f"union of attempts misses positions {missing[:8]}")
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
    ap.add_argument("--ledger-segment", type=int, default=None, help="resume: checkpoint segment")
    ap.add_argument("--ledger-positions", type=int, default=None, help="resume: positions done")
    ap.add_argument("--expected-segments", type=int, default=None)
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
        if args.start_position is None and args.ledger_segment is None:
            ap.error("--start-position or --ledger-segment/--ledger-positions is required")
        if (args.ledger_segment is None) != (args.ledger_positions is None):
            ap.error("--ledger-segment and --ledger-positions go together")
        ledger = (
            (args.ledger_segment, args.ledger_positions)
            if args.ledger_segment is not None
            else None
        )
        problems = check_resume(
            records, args.W, args.start_position, ledger, expected_segments=args.expected_segments
        )
    else:
        other = read_audit(fs, root, args.other_run_name)
        problems = check_s7(records, other)
    for p in problems:
        print(f"{args.scenario} FAIL: {p}")
    print(f"{args.scenario} {'PASS' if not problems else 'FAIL'}")
    return 1 if problems else 0


if __name__ == "__main__":
    raise SystemExit(main())
