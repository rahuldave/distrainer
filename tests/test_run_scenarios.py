"""The cluster scenario runner's process bookkeeping (the scenarios themselves need docker)."""

import os
import subprocess
import sys

from integration_tests.cluster import run_scenarios as rs


def test_finish_returns_the_process_output_and_final_metrics_parses_it(tmp_path):
    log_path = tmp_path / "p.log"
    log = open(log_path, "w")  # noqa: SIM115 (finish closes it)
    proc = subprocess.Popen(
        [
            sys.executable,
            "-c",
            "print('hello'); print(\"final metrics: {'reports': 3, 'x': 1.5}\")",
        ],
        stdout=log,
        stderr=subprocess.STDOUT,
        text=True,
    )
    rs._logs[proc.pid] = (log_path, log)
    rs._running.append(proc)
    out = rs.finish(proc, name="p")
    assert "hello" in out and proc not in rs._running and proc.pid not in rs._logs
    assert log.closed
    assert rs.final_metrics(out) == {"reports": 3, "x": 1.5}
    try:
        rs.final_metrics("nothing here")
    except RuntimeError as exc:
        assert "final metrics" in str(exc)
    else:
        raise AssertionError("final_metrics must fail without the line")


def test_log_exists_probes_a_local_store(tmp_path):
    from distrainer.log import BlockLog
    from distrainer.storage import resolve

    assert not rs.log_exists(str(tmp_path))
    BlockLog.create(*resolve(str(tmp_path)), W=4, seed=1)
    assert rs.log_exists(str(tmp_path))


# ---- where the runner reads a scenario's trail: the shared mount, or the bucket ----


def fake_driver(tmp_path, shared: str, endpoint: str):
    """A driver script that answers `shared` and `endpoint` only."""
    script = tmp_path / "driver.sh"
    script.write_text(
        "#!/usr/bin/env bash\n"
        'case "$1" in\n'
        f'  shared) printf "%s\\n" "{shared}" ;;\n'
        f'  endpoint) printf "%b\\n" "{endpoint}" ;;\n'
        '  *) echo "unexpected verb $1" >&2; exit 2 ;;\n'
        "esac\n"
    )
    script.chmod(0o755)
    return script


def test_minio_endpoint_parses_the_endpoint_verb():
    out = "dashboard=http://h:8265\nminio=http://192.168.139.221:9000 console=http://h:9001\n"
    assert rs.minio_endpoint(out) == "http://192.168.139.221:9000"
    try:
        rs.minio_endpoint("dashboard=http://h:8265\n")
    except RuntimeError as exc:
        assert "minio" in str(exc)
    else:
        raise AssertionError("no minio line must be an error")


def test_store_is_the_shared_mount_when_the_driver_has_one(tmp_path, monkeypatch):
    monkeypatch.setattr(rs, "DRIVER", fake_driver(tmp_path, str(tmp_path / "shared"), ""))
    st = rs.store_for_driver()
    assert not st.on_bucket
    assert st.blocks == str(tmp_path / "shared" / "blocks")
    assert st.runs == str(tmp_path / "shared" / "runs")
    assert st.cfg == rs.HARNESS_CFG and st.env == {} and st.s3 == {}


def test_store_is_the_bucket_when_the_driver_has_no_shared_mount(tmp_path, monkeypatch):
    endpoint = "dashboard=http://h:8265\\nminio=http://h:9000 console=http://h:9001"
    monkeypatch.setattr(rs, "DRIVER", fake_driver(tmp_path, "", endpoint))
    st = rs.store_for_driver()
    assert st.on_bucket
    assert st.blocks == rs.S3_BLOCKS and st.runs == rs.S3_RUNS
    assert st.cfg == rs.MINIO_CFG and st.env == {"DISTRAINER_MINIO": "1"}
    assert st.s3["endpoint"] == "http://h:9000"  # the bucket as seen from the Mac


def test_local_store_reads_a_trail_and_fresh_removes_the_run(tmp_path):
    from distrainer.audit import AuditWriter

    st = rs.Store(
        blocks=str(tmp_path / "blocks"), runs=str(tmp_path / "runs"), cfg=rs.HARNESS_CFG,
        env={}, s3={}, segments=10,
    )  # fmt: skip
    (tmp_path / "runs" / "s2").mkdir(parents=True)
    (tmp_path / "blocks").mkdir()
    fs, root = st.fs(st.blocks)
    writer = AuditWriter(fs, root, "s2", attempt=0, rank=0)
    writer.append(world_size=2, segment=0, step=0, position=0, block_id="b0")
    writer.close()
    assert [r.block_id for r in st.records("s2")] == ["b0"]
    st.wait_for_blocks("s2", 1, timeout_s=2)
    try:
        st.wait_for_blocks("s2", 2, timeout_s=0.2)
    except TimeoutError:
        pass
    else:
        raise AssertionError("one record must not satisfy a wait for two")
    st.fresh("s2")
    assert not (tmp_path / "runs" / "s2").exists()
    assert not (tmp_path / "blocks" / "audit" / "s2").exists()
    st.fresh("s2")  # a second time is fine: nothing to remove


def test_shared_skips_the_scenario_when_the_driver_prints_nothing(tmp_path, monkeypatch):
    monkeypatch.setattr(rs, "DRIVER", fake_driver(tmp_path, "", ""))
    try:
        rs.shared()
    except rs.SkipScenario as exc:
        assert "shared mount" in str(exc)
    else:
        raise AssertionError("an empty `shared` must skip the scenario")
    monkeypatch.setattr(rs, "DRIVER", fake_driver(tmp_path, str(tmp_path), ""))
    assert rs.shared() == tmp_path


def test_mac_env_sets_the_bucket_for_the_mac_only_and_restores(monkeypatch):
    monkeypatch.setenv("S3_ENDPOINT", "http://containers-see-this:9000")
    monkeypatch.delenv("S3_ACCESS_KEY", raising=False)
    st = rs.Store(
        blocks=rs.S3_BLOCKS, runs=rs.S3_RUNS, cfg=rs.MINIO_CFG, env={"DISTRAINER_MINIO": "1"},
        s3={"endpoint": "http://mac-sees-this:9000", "region": "auto"}, segments=10,
        credentials={"S3_ACCESS_KEY": "k", "S3_SECRET_KEY": "s"},
    )  # fmt: skip
    with st.mac_env():
        assert os.environ["S3_ENDPOINT"] == "http://mac-sees-this:9000"
        assert os.environ["S3_ACCESS_KEY"] == "k"
    assert os.environ["S3_ENDPOINT"] == "http://containers-see-this:9000"
    assert "S3_ACCESS_KEY" not in os.environ
    local = rs.Store(blocks="/b", runs="/r", cfg=rs.HARNESS_CFG, env={}, s3={}, segments=10)
    with local.mac_env():  # a shared-mount store touches nothing
        assert os.environ["S3_ENDPOINT"] == "http://containers-see-this:9000"


def test_main_reports_a_skipped_scenario_without_failing(monkeypatch, capsys):
    def skipper():
        raise rs.SkipScenario("needs a shared mount")

    monkeypatch.setattr(rs, "SCENARIOS", {"S6": skipper, "S2": lambda: []})
    monkeypatch.setattr(rs, "driver", lambda *a, **k: "")  # the final `down`
    assert rs.main(["--scenario", "all"]) == 0
    out = capsys.readouterr().out
    assert "S6: SKIP (needs a shared mount)" in out and "S2: PASS" in out
    monkeypatch.setattr(rs, "SCENARIOS", {"S2": lambda: ["bad"]})
    assert rs.main(["--scenario", "S2"]) == 1
