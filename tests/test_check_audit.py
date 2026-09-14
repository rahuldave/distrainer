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
