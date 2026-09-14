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
