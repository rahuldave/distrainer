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


def test_store_endpoint_parses_the_endpoint_verb():
    out = "dashboard=http://h:8265\nminio=http://192.168.139.221:9000 console=http://h:9001\n"
    assert rs.store_endpoint(out) == ("minio", "http://192.168.139.221:9000")
    out = "dashboard=http://203.0.113.7:8265\ns3=https://s3.us-east-1.amazonaws.com\n"
    assert rs.store_endpoint(out) == ("s3", "https://s3.us-east-1.amazonaws.com")
    try:
        rs.store_endpoint("dashboard=http://h:8265\n")
    except RuntimeError as exc:
        assert "minio= or s3=" in str(exc)
    else:
        raise AssertionError("no store line must be an error")
    # kuberay's placeholder while nothing runs: MinIO is the store, its URL comes after `up`
    out = "minio=http://<no minio service>:9000 console=http://<no minio service>:9001\n"
    assert rs.store_endpoint(out) == ("minio", None)
    try:  # a store outside the cluster must be a URL
        rs.store_endpoint("s3=https://<unset>\n")
    except RuntimeError as exc:
        assert "no usable store" in str(exc)
    else:
        raise AssertionError("a placeholder external URL must be an error")


def test_read_dotenv_reads_values_as_the_shell_would(tmp_path):
    f = tmp_path / "env"
    f.write_text(
        "A=bare  # a comment\nB='quoted # not a comment'\nC=\"dq\"\nexport D=x\n# E=no\nF=\n"
    )
    values = rs.read_dotenv(f, {k: "dflt" for k in "ABCDEF"})
    assert values == {
        "A": "bare",
        "B": "quoted # not a comment",
        "C": "dq",
        "D": "x",
        "E": "dflt",
        "F": "",
    }


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
    assert not st.external and st.bucket == "distrainer"


S3_CFG_ENDPOINT = rs.load_config(str(rs.ROOT / rs.S3_CFG)).storage.endpoint


def test_store_is_the_external_bucket_when_the_driver_prints_s3(tmp_path, monkeypatch):
    """`s3=` names a store outside the cluster: harness-s3.yaml, no MinIO in the driver env."""
    monkeypatch.setattr(rs, "DOTENV", tmp_path / "no-dotenv")
    endpoint = f"dashboard=http://203.0.113.7:8265\\ns3={S3_CFG_ENDPOINT}"
    monkeypatch.setattr(rs, "DRIVER", fake_driver(tmp_path, "", endpoint))
    st = rs.store_for_driver()
    cfg = rs.load_config(str(rs.ROOT / rs.S3_CFG))
    assert st.external and st.on_bucket and st.env == {}
    assert st.cfg == rs.S3_CFG
    assert st.blocks == f"s3://{cfg.store_root}" and st.runs == f"s3://{cfg.storage_path}"
    assert st.bucket == cfg.store_root.split("/")[0]
    assert st.s3 == {"endpoint": S3_CFG_ENDPOINT, "region": cfg.storage.region}
    assert st.segments == rs.MINIO_SEGMENTS and st.W == 24
    assert st.credentials == {"S3_ACCESS_KEY": "distrainer", "S3_SECRET_KEY": "distrainer123"}
    assert st.lag_intervals > 0  # the controller registers checkpoints behind the workers there
    with st.mac_env():
        assert os.environ["S3_REGION"] == cfg.storage.region


def test_external_store_must_match_the_config_endpoint(tmp_path, monkeypatch):
    monkeypatch.setattr(rs, "DOTENV", tmp_path / "no-dotenv")
    endpoint = "dashboard=http://h:8265\\ns3=https://elsewhere.example"
    monkeypatch.setattr(rs, "DRIVER", fake_driver(tmp_path, "", endpoint))
    try:
        rs.bucket_store()
    except RuntimeError as exc:
        assert rs.S3_CFG in str(exc) and "elsewhere.example" in str(exc)
    else:
        raise AssertionError("a config that stores elsewhere than the driver says must fail")


def recording_driver(tmp_path, endpoint: str):
    """A driver that answers `shared` (nothing) and `endpoint`, and logs every other verb."""
    script, log = tmp_path / "driver.sh", tmp_path / "verbs.log"
    script.write_text(
        "#!/usr/bin/env bash\n"
        'case "$1" in\n'
        "  shared) ;;\n"
        f'  endpoint) printf "%b\\n" "{endpoint}" ;;\n'
        f'  *) echo "$*" >> "{log}" ;;\n'
        "esac\n"
    )
    script.chmod(0o755)
    return script, log


def test_up_bucket_reads_minio_url_after_the_deploy_when_the_driver_had_none(tmp_path, monkeypatch):
    """kuberay prints a placeholder MinIO URL until the service exists: bucket_store() before
    `up` is still the MinIO store, and up_bucket returns one that knows the URL."""
    monkeypatch.setattr(rs, "DOTENV", tmp_path / "no-dotenv")
    monkeypatch.setattr(rs, "wait_for_trainers", lambda n, timeout_s=0: None)
    script, log = tmp_path / "driver.sh", tmp_path / "verbs.log"
    script.write_text(
        "#!/usr/bin/env bash\n"
        'case "$1" in\n'
        "  shared) ;;\n"
        f'  endpoint) if [ -e "{tmp_path}/up-done" ]; then echo "minio=http://10.0.0.5:9000 x"; '
        'else echo "minio=http://<no minio service>:9000 x"; fi ;;\n'
        f'  up) touch "{tmp_path}/up-done"; echo "$*" >> "{log}" ;;\n'
        f'  *) echo "$*" >> "{log}" ;;\n'
        "esac\n"
    )
    script.chmod(0o755)
    monkeypatch.setattr(rs, "DRIVER", script)
    st = rs.bucket_store()
    assert not st.external and st.s3["endpoint"] is None
    try:
        st.fs(st.blocks)
    except RuntimeError as exc:
        assert "before `up`" in str(exc)
    else:
        raise AssertionError("a store without a URL must refuse to build a client")
    st = rs.up_bucket(2, st)
    assert st.s3["endpoint"] == "http://10.0.0.5:9000"
    assert log.read_text().splitlines() == ["up 2 minio", "mkbucket distrainer"]


def test_up_bucket_deploys_minio_only_when_minio_is_the_store(tmp_path, monkeypatch):
    monkeypatch.setattr(rs, "DOTENV", tmp_path / "no-dotenv")
    monkeypatch.setattr(rs, "wait_for_trainers", lambda n, timeout_s=0: None)
    script, log = recording_driver(tmp_path, f"dashboard=http://h:8265\\ns3={S3_CFG_ENDPOINT}")
    monkeypatch.setattr(rs, "DRIVER", script)
    st = rs.up_store(2)
    assert st.external
    assert log.read_text().splitlines() == ["up 2"]  # no MinIO, no mkbucket: the bucket exists
    log.unlink()
    script, log = recording_driver(tmp_path, "dashboard=http://h:8265\\nminio=http://h:9000 x")
    monkeypatch.setattr(rs, "DRIVER", script)
    st = rs.up_store(3)
    assert not st.external
    assert log.read_text().splitlines() == ["up 3 minio", "mkbucket distrainer"]


def test_s3_settings_follow_the_env_file_dotenv_points_at(tmp_path, monkeypatch):
    """The bootstrap's env file (DISTRAINER_ENV_FILE) is read after .env, as the driver does."""
    monkeypatch.setattr(rs, "ROOT", tmp_path)
    monkeypatch.setenv("DISTRAINER_DRIVER", "uncloud")
    monkeypatch.delenv("DISTRAINER_UNCLOUD_PROVIDER", raising=False)
    dotenv = tmp_path / ".env"
    monkeypatch.setattr(rs, "DOTENV", dotenv)
    assert rs.s3_settings_from_dotenv()["S3_ACCESS_KEY"] == "distrainer"  # no .env: the defaults
    dotenv.write_text("S3_ACCESS_KEY=minio-user\nexport S3_REGION='auto'\n# S3_SECRET_KEY=no\n")
    settings = rs.s3_settings_from_dotenv()
    assert settings == {
        "S3_ACCESS_KEY": "minio-user",
        "S3_SECRET_KEY": "distrainer123",
        "S3_REGION": "auto",
    }
    (tmp_path / "aws").mkdir()
    (tmp_path / "aws" / "env").write_text(
        'S3_ACCESS_KEY="AKIA"\nS3_SECRET_KEY=s\nS3_REGION=us-east-1\n'
    )
    dotenv.write_text("S3_ACCESS_KEY=minio-user\nDISTRAINER_ENV_FILE=aws/env\n")  # relative to ROOT
    settings = rs.s3_settings_from_dotenv()
    assert settings == {"S3_ACCESS_KEY": "AKIA", "S3_SECRET_KEY": "s", "S3_REGION": "us-east-1"}
    dotenv.write_text("DISTRAINER_ENV_FILE=/nowhere/env\n")
    try:
        rs.s3_settings_from_dotenv()
    except RuntimeError as exc:
        assert "DISTRAINER_ENV_FILE" in str(exc)
    else:
        raise AssertionError("a pointer to a missing file must fail, not fall back silently")
    monkeypatch.setenv(
        "DISTRAINER_ENV_FILE", str(tmp_path / "aws" / "env")
    )  # the shell's pointer wins
    assert rs.s3_settings_from_dotenv()["S3_ACCESS_KEY"] == "AKIA"
    monkeypatch.setenv("DISTRAINER_DRIVER", "compose")  # the env file belongs to an uncloud bed
    assert rs.s3_settings_from_dotenv()["S3_ACCESS_KEY"] == "distrainer"  # .env has no key here
    monkeypatch.setenv("DISTRAINER_DRIVER", "uncloud")
    (tmp_path / "aws" / "env").write_text(
        'DISTRAINER_UNCLOUD_PROVIDER=aws\nS3_ACCESS_KEY="AKIA"\nS3_SECRET_KEY=s\nS3_REGION=us-east-1\n'
    )
    monkeypatch.setenv("DISTRAINER_UNCLOUD_PROVIDER", "orbstack")  # another bed pinned: ignored
    assert rs.s3_settings_from_dotenv()["S3_ACCESS_KEY"] == "distrainer"
    monkeypatch.setenv("DISTRAINER_UNCLOUD_PROVIDER", "aws")
    assert rs.s3_settings_from_dotenv()["S3_ACCESS_KEY"] == "AKIA"
    monkeypatch.delenv("DISTRAINER_UNCLOUD_PROVIDER")
    dotenv.unlink()  # and works without any .env
    assert rs.s3_settings_from_dotenv()["S3_REGION"] == "us-east-1"
    monkeypatch.setenv("S3_REGION", "eu-west-1")  # the shell wins over both files
    assert rs.s3_settings_from_dotenv()["S3_REGION"] == "eu-west-1"


def test_recovery_scenarios_pass_the_store_allowance_to_check_recovery(tmp_path, monkeypatch):
    """S2, S3 and S4 hand the store's lag_intervals to check_recovery: the wiring, not the bound."""
    seen: list[int] = []
    monkeypatch.setattr(rs, "check_recovery", lambda *a, **k: seen.append(k["lag_intervals"]) or [])
    monkeypatch.setattr(rs, "check_transition", lambda *a, **k: [])
    monkeypatch.setattr(rs, "summarize", lambda recs: "")
    st = rs.Store(blocks="s3://b/blocks", runs="s3://b/runs", cfg=rs.S3_CFG, env={}, s3={},
                  segments=10, external=True)  # fmt: skip
    monkeypatch.setattr(rs, "up_store", lambda n: st)
    monkeypatch.setattr(rs, "ensure_blocks", lambda cfg: None)
    monkeypatch.setattr(rs.Store, "fresh", lambda self, run: None)
    monkeypatch.setattr(rs.Store, "wait_for_blocks", lambda self, *a, **k: None)
    monkeypatch.setattr(rs.Store, "records", lambda self, run: [])
    monkeypatch.setattr(rs, "driver", lambda *a, **k: "")
    monkeypatch.setattr(rs, "start_train", lambda *a, **k: object())
    monkeypatch.setattr(rs, "finish", lambda *a, **k: "")
    for scenario in (rs.scenario_s2, rs.scenario_s3, rs.scenario_s4):
        scenario()
    assert seen == [st.lag_intervals] * 3 and st.lag_intervals == 4


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
        assert os.environ["S3_REGION"] == "auto"  # real S3 signs by region: it must travel too
    assert st.lag_intervals == 0
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
