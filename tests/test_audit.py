import pytest
from conftest import MemoryFS

from distrainer.audit import AuditRecord, AuditWriter, audit_dir, read_audit
from distrainer.storage import join, list_names


def test_append_is_tailable_and_read_back_sorted(store):
    fs, root = store
    w0 = AuditWriter(fs, root, "run", attempt=1, rank=0)
    w1 = AuditWriter(fs, root, "run", attempt=1, rank=1)
    w0.append(2, 0, 0, 0, "b0", ts=1.0)
    w1.append(2, 0, 0, 1, "b1", ts=1.5)
    w0.append(2, 0, 1, 2, "b2", ts=2.0)
    # visible before close (line-buffered append on local fs)
    with open(w0.path, encoding="utf-8") as f:
        assert len(f.read().splitlines()) == 2
    w0.close()
    w1.close()
    recs = read_audit(fs, root, "run")
    assert [(r.attempt, r.rank, r.position) for r in recs] == [(1, 0, 0), (1, 0, 2), (1, 1, 1)]
    assert recs[0] == AuditRecord(1, 0, 2, 0, 0, 0, "b0", 1.0)
    assert sorted(list_names(fs, audit_dir(root, "run"))) == ["1-0.jsonl", "1-1.jsonl"]
    # a second attempt appends to its own file; unknown files are ignored
    w2 = AuditWriter(fs, root, "run", attempt=2, rank=0)
    w2.append(1, 0, 0, 2, "b2")
    w2.close()
    with open(join(audit_dir(root, "run"), "notes.txt"), "w") as f:
        f.write("ignored")
    assert [r.attempt for r in read_audit(fs, root, "run")] == [1, 1, 1, 2]
    assert read_audit(fs, root, "other") == []


def test_reopen_appends_and_json_roundtrip(store):
    fs, root = store
    AuditWriter(fs, root, "r", 1, 0).append(1, 0, 0, 0, "a").to_json()
    w = AuditWriter(fs, root, "r", 1, 0)
    w.append(1, 0, 1, 1, "b")
    w.close()
    recs = read_audit(fs, root, "r")
    assert [r.block_id for r in recs] == ["a", "b"]
    assert AuditRecord.from_json(recs[1].to_json()) == recs[1]


def test_object_store_writer_buffers_parts_and_reopens():
    fs = MemoryFS()
    fs.create_dir("root")
    w = AuditWriter(fs, "root", "s3run", 1, 0, part_lines=2)
    w.append(1, 0, 0, 0, "a")
    assert read_audit(fs, "root", "s3run") == []  # nothing written until flush
    w.flush()
    assert [r.block_id for r in read_audit(fs, "root", "s3run")] == ["a"]
    w.append(1, 0, 1, 1, "b")  # reaches part_lines: part 0 is written, part 1 starts
    w.append(1, 0, 2, 2, "c")
    w.close()
    w.close()  # idempotent
    with pytest.raises(ValueError):
        w.append(1, 0, 3, 3, "d")
    assert sorted(list_names(fs, audit_dir("root", "s3run"))) == ["1-0.0.jsonl", "1-0.1.jsonl"]
    assert [r.block_id for r in read_audit(fs, "root", "s3run")] == ["a", "b", "c"]
    w2 = AuditWriter(fs, "root", "s3run", 1, 0)  # reopen continues with a new part
    w2.flush()  # empty flush writes nothing
    w2.append(1, 0, 3, 3, "d")
    w2.close()
    assert [r.block_id for r in read_audit(fs, "root", "s3run")] == ["a", "b", "c", "d"]
    assert "1-0.2.jsonl" in list_names(fs, audit_dir("root", "s3run"))


def test_ordering_is_file_order_not_timestamp(store):
    fs, root = store
    w = AuditWriter(fs, root, "clock", 1, 0)
    w.append(1, 0, 0, 0, "first", ts=100.0)
    w.append(1, 0, 1, 1, "second", ts=50.0)  # clock stepped backwards
    w.append(1, 0, 2, 2, "third", ts=50.0)
    w.close()
    w.close()
    assert [r.block_id for r in read_audit(fs, root, "clock")] == ["first", "second", "third"]
    with pytest.raises(ValueError):
        AuditWriter(fs, root, "bad/name", 1, 0)
