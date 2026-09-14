from distrainer.audit import AuditRecord
from integration_tests.cluster.check_audit import (
    check_dealing,
    check_resume,
    check_s1,
    check_s7,
    expected_checkpoints,
    summarize,
)


def rec(attempt, rank, n, seg, step, pos, block=None, ts=0.0):
    return AuditRecord(attempt, rank, n, seg, step, pos, block or f"b{pos}", ts)


def happy(W=8, n=2, segments=2, attempt=0):
    out = []
    for seg in range(segments):
        for step in range(W // n):
            for rank in range(n):
                out.append(rec(attempt, rank, n, seg, step, seg * W + step * n + rank))
    return out


def test_s1_passes_on_a_correct_trail():
    assert check_s1(happy(), 8) == []
    assert "48" not in summarize(happy()) and "16 blocks" in summarize(happy())


def test_s1_detects_missing_duplicate_and_misdealt_positions():
    recs = happy()
    assert any("missing" in p for p in check_s1(recs[:-1], 8))
    dup = recs + [recs[0]]
    assert any("duplicated" in p for p in check_s1(dup, 8))
    wrong = recs[:]
    wrong[3] = rec(0, 1, 2, 0, 1, 2)  # rank 1 step 1 should be position 3
    assert check_dealing(wrong, 8) and any("expected 3" in p for p in check_dealing(wrong, 8))
    two_attempts = recs + [rec(1, 0, 2, 5, 0, 40)]
    assert any("one attempt" in p for p in check_s1(two_attempts, 8))
    unequal = recs + [rec(0, 0, 2, 2, 0, 16)]
    assert any("unequal" in p for p in check_s1(unequal, 8))
    assert check_s1([], 8) == ["no audit records"]


def test_s7_compares_per_rank_sequences():
    a, b = happy(), happy()
    assert check_s7(a, b) == []
    b[5] = rec(0, 1, 2, 0, 2, 5, block="other")
    assert any("rank 1" in p and "index 2" in p for p in check_s7(a, b))
    assert any("rank sets" in p for p in check_s7(a, [r for r in b if r.rank == 0]))


def test_check_resume_accepts_a_partial_first_segment():
    recs = [r for r in happy(W=8, n=2, segments=2) if r.position >= 6]
    assert check_resume(recs, 8, 6) == []
    assert any("missing" in p for p in check_resume(recs, 8, 4))
    assert any("extra" in p for p in check_resume(recs, 8, 7))
    assert check_resume([], 8, 0) == ["no audit records"]


def test_expected_checkpoints():
    assert expected_checkpoints(4, 12, 2, 2) == 12  # 6 steps: 2,4,6 per segment
    assert expected_checkpoints(4, 12, 2, None) == 4  # segment ends only
    assert expected_checkpoints(1, 12, 3, 5) == 1  # 4 steps: only the end
    assert expected_checkpoints(4, 12, 2, 4, segment_end=False) == 4  # bare every_k: step 4 only
    assert expected_checkpoints(4, 12, 2, 1, segment_end=False) == 24


def test_check_recovery_accepts_a_resized_restart_and_flags_gaps():
    from integration_tests.cluster.check_audit import check_recovery

    W = 8
    first = [r for r in happy(W=W, n=2, segments=2, attempt=0) if r.position < 6]  # died in seg 0
    # checkpoint after step 1 (cursor 2, n=2 -> 4 done); restart with 4 ranks from position 4
    second = []
    for seg in range(2):
        for step in range(W // 4):
            for rank in range(4):
                pos = seg * W + step * 4 + rank
                if pos >= 4:
                    second.append(rec(1, rank, 4, seg, step, pos))
    assert check_recovery(first + second, W, every_k=2, expected_world_sizes=[2, 4]) == []
    assert any("world sizes" in p for p in check_recovery(first + second, W, 2, [2, 2]))
    assert any("at least 2" in p for p in check_recovery(first, W, 2))
    skipped = [r for r in second if r.position >= 6]  # restart skips 4 and 5: not a step boundary
    assert any("step boundary" in p for p in check_recovery(first + skipped, W, 2))
    gap = [r for r in second if r.position >= 8]  # restart at 8: 6 and 7 never consumed
    problems = check_recovery(first + gap, W, 2)
    assert any("gap" in p for p in problems) and any("misses" in p for p in problems)
    off = [rec(1, r.rank, 4, r.segment, r.step, r.position) for r in second]
    off[0] = rec(1, 0, 4, 0, 1, 5)  # first position not on a step boundary for n=4
    problems = check_recovery(first + off, W, 2)
    assert problems


def test_check_recovery_allows_one_in_flight_checkpoint_interval():
    from integration_tests.cluster.check_audit import check_recovery

    W = 24
    first = [r for r in happy(W=W, n=3, segments=3, attempt=0) if r.position < 30]
    second = []
    for seg in range(3):
        for step in range(W // 2):
            for rank in range(2):
                pos = seg * W + step * 2 + rank
                if pos >= 18:  # resumed from the checkpoint before the in-flight one (24)
                    second.append(rec(1, rank, 2, seg, step, pos))
    assert check_recovery(first + second, W, every_k=2, expected_world_sizes=[3, 2]) == []
    too_far = [r for r in second if r.position >= 12] + [
        rec(1, 0, 2, 0, 6, 12),
        rec(1, 1, 2, 0, 6, 13),
    ]
    assert any("replays" in p for p in check_recovery(first + too_far, W, every_k=2))


def test_check_recovery_rejects_short_runs_and_duplicates_within_an_attempt():
    from integration_tests.cluster.check_audit import check_recovery

    W = 8
    first = [r for r in happy(W=W, n=2, segments=2, attempt=0) if r.position < 6]
    second = [
        rec(1, r.rank, 2, r.segment, r.step, r.position)
        for r in happy(W=W, n=2, segments=2)
        if r.position >= 4
    ]
    assert check_recovery(first + second, W, every_k=2, expected_segments=2) == []
    assert any("expected 3" in p for p in check_recovery(first + second, W, 2, expected_segments=3))
    dup = first + second + [second[0]]
    assert any("twice within" in p for p in check_recovery(dup, W, 2))


def test_check_resume_derives_the_start_from_the_ledger_and_world_size():
    from integration_tests.cluster.check_audit import check_resume, resume_start_position

    W = 12
    # checkpoint (segment 1, 8 positions done at n=2); resumed with 3 ranks -> rounds down to 6
    assert resume_start_position(1, 8, W, 3) == 12 + 6
    assert resume_start_position(1, 12, W, 3) == 24  # completed segment -> next one
    recs = []
    for seg in range(1, 3):
        for step in range(W // 3):
            for rank in range(3):
                pos = seg * W + step * 3 + rank
                if pos >= 18:
                    recs.append(rec(0, rank, 3, seg, step, pos))
    assert check_resume(recs, W, ledger=(1, 8), expected_segments=3) == []
    assert any("expected 4" in p for p in check_resume(recs, W, ledger=(1, 8), expected_segments=4))
    assert check_resume(recs, W) == ["check_resume needs start_position or ledger"]


def test_expected_reports():
    from types import SimpleNamespace

    from integration_tests.cluster.check_audit import expected_reports

    any_cfg = SimpleNamespace(policy="any", every_k=2, report_every_step=False)
    assert expected_reports(4, 12, 2, any_cfg) == 12  # 3 per segment, last step is a checkpoint
    k4 = SimpleNamespace(policy="every_k", every_k=4, report_every_step=False)
    assert expected_reports(4, 12, 2, k4) == 4 + 1  # step 4 of 6 only, plus the final report
    every = SimpleNamespace(policy="segment_end", every_k=None, report_every_step=True)
    assert expected_reports(4, 12, 2, every) == 24


def test_checkpoint_ledgers_check_s5_and_the_cli_resume_branch(tmp_path, capsys):
    import json

    from distrainer.audit import AuditWriter
    from distrainer.storage import StorageConfig, build_filesystem
    from integration_tests.cluster.check_audit import check_s5, checkpoint_ledgers, main

    run = tmp_path / "runs" / "r"
    for seg, cursor, n in [(0, 2, 2), (0, 4, 2), (1, 2, 2)]:
        d = run / f"checkpoint_g{seg:06d}_p{cursor * n:06d}_n{n:02d}_a00"
        d.mkdir(parents=True)
        (d / ".metadata.json").write_text(
            json.dumps(
                {"ledger": {"segment": seg, "cursor": cursor, "world_size": n, "run_attempt": 0}}
            )
        )
    (run / "checkpoint_manager_snapshot.json").write_text("{}")
    (run / "checkpoint_g000001_p000008_n02_a00").mkdir()  # no metadata: a partial upload
    ledgers = checkpoint_ledgers(str(run))
    assert sorted(ledgers) == [
        "checkpoint_g000000_p000004_n02_a00",
        "checkpoint_g000000_p000008_n02_a00",
        "checkpoint_g000001_p000004_n02_a00",
    ]
    assert check_s5(str(run), W=8, every_k=2, expected_count=3) == []
    assert any("expected 4" in p for p in check_s5(str(run), 8, 2, 4))
    assert any("multiple" in p for p in check_s5(str(run), 8, 3, None))

    fs, root = build_filesystem(StorageConfig(kind="local", path=str(tmp_path / "store")))
    w = AuditWriter(fs, root, "res", 0, 0)
    w1 = AuditWriter(fs, root, "res", 0, 1)
    for step in range(2):
        for rank, writer in enumerate((w, w1)):
            writer.append(2, 1, 2 + step, 12 + step * 2 + rank, f"b{step}{rank}")
    w.close()
    w1.close()
    rc = main(
        [
            "--store-root",
            root,
            "--run-name",
            "res",
            "--scenario",
            "resume",
            "--W",
            "8",
            "--ledger-segment",
            "1",
            "--ledger-positions",
            "4",
            "--expected-segments",
            "2",
        ]
    )
    out = capsys.readouterr().out
    assert rc == 0 and out.strip().endswith("resume PASS")


# ---- M4: S8, S11 and retention ----


def test_check_s8_counts_reports_against_time_budget_checkpoints():
    from integration_tests.cluster.check_audit import check_s8

    recs = happy(W=8, n=2, segments=2)
    ledgers = {
        "checkpoint_g000000_p000006_n02_a00": {"segment": 0, "cursor": 3, "world_size": 2},
        "checkpoint_g000001_p000004_n02_a00": {"segment": 1, "cursor": 2, "world_size": 2},
    }
    assert check_s8(recs, 8, 3, ledgers) == []  # final metrics-only report after the last one
    assert check_s8(recs, 8, 2, ledgers) == []
    assert any("reported 5" in p for p in check_s8(recs, 8, 5, ledgers))
    one = dict(list(ledgers.items())[:1])
    assert any("at least 2" in p for p in check_s8(recs, 8, 1, one))
    bad = {"checkpoint_g000000_p000010_n02_a00": {"segment": 0, "cursor": 5, "world_size": 2}}
    assert any("step boundary" in p for p in check_s8(recs, 8, 2, {**bad, **one}))
    renamed = {"checkpoint_x": ledgers["checkpoint_g000000_p000006_n02_a00"], **one}
    assert any("does not match" in p for p in check_s8(recs, 8, 2, renamed))


def test_check_s11_wants_commits_before_consumption_and_a_real_wait():
    from integration_tests.cluster.check_audit import check_s11, segment_starts

    W, n = 4, 2
    recs = []
    for seg in range(3):
        for step in range(W // n):
            for rank in range(n):
                pos = seg * W + step * n + rank
                recs.append(rec(0, rank, n, seg, step, pos, ts=10.0 * seg + step))
    assert segment_starts(recs) == {0: 0.0, 1: 10.0, 2: 20.0}
    committed = {0: -1.0, 1: 9.5, 2: 19.0}
    assert check_s11(recs, W, committed, min_gap_s=8.0, expected_segments=3) == []
    assert any("never waited" in p for p in check_s11(recs, W, committed, min_gap_s=11.0))
    late = {**committed, 2: 20.5}
    assert any("before its commit" in p for p in check_s11(recs, W, late, 8.0))
    assert any("no commit time" in p for p in check_s11(recs, W, {0: 0.0, 1: 9.0}, 8.0))
    assert any("expected 0..3" in p for p in check_s11(recs, W, committed, 8.0, 4))
    assert check_s11([], W, committed, 8.0) == ["no audit records"]


def test_check_retention_pins_the_window_and_the_block_directory():
    from integration_tests.cluster.check_audit import check_retention

    kept = {"blocks/p7.parquet", "blocks/p8.parquet", "blocks/p9.parquet"}
    assert check_retention([7, 8, 9], set(kept), kept, 9, 2) == []
    assert check_retention([0, 1], {"blocks/a"}, {"blocks/a"}, 1, 4) == []  # nothing to drop yet
    assert any("starts at segment 6" in p for p in check_retention([6, 7, 8, 9], kept, kept, 9, 2))
    assert any("not contiguous" in p for p in check_retention([7, 9], kept, kept, 9, 2))
    assert any(
        "still present" in p
        for p in check_retention([7, 8, 9], kept | {"blocks/p1.parquet"}, kept, 9, 2)
    )
    assert any(
        "missing" in p for p in check_retention([7, 8, 9], {"blocks/p7.parquet"}, kept, 9, 2)
    )
    assert check_retention([], set(), set(), 3, 1) == ["no segments left in the log"]
