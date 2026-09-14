"""Cluster scenarios (spec section 10) driven through the deploy/driver.sh verbs.

S2 worker kill mid-run, S3 elastic scale up, S4 scale down, S9 cold restore and S10 head loss
against MinIO. Each scenario brings the cluster to the state it needs, runs hello_blocks inside
the head container, injects the failure while training runs, waits for the run to finish, and
checks the audit trail on the shared mount. Run: ``just integration S2`` (or ``all``).
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from distrainer.audit import read_audit  # noqa: E402
from distrainer.storage import resolve  # noqa: E402
from integration_tests.cluster.check_audit import (  # noqa: E402
    check_recovery,
    summarize,
)

DRIVER = ROOT / "deploy" / "driver.sh"
HARNESS_CFG = "examples/hello_blocks/harness.yaml"
MINIO_CFG = "examples/hello_blocks/harness-minio.yaml"
W = 24


def driver(*args: str, env: dict[str, str] | None = None, check: bool = True) -> str:
    proc = subprocess.run(
        [str(DRIVER), *args],
        cwd=ROOT,
        capture_output=True,
        text=True,
        env={**os.environ, **(env or {})},
    )
    if check and proc.returncode != 0:
        raise RuntimeError(
            f"driver {' '.join(args)} failed:\n{proc.stdout[-1500:]}\n{proc.stderr[-1500:]}"
        )
    return proc.stdout


def shared() -> Path:
    return Path(driver("shared").strip())


def audit_dir(run_name: str) -> Path:
    return shared() / "blocks" / "audit" / run_name


def start_train(cfg: str, *overrides: str, env: dict[str, str] | None = None) -> subprocess.Popen:
    cmd = [
        str(DRIVER),
        "exec-head",
        "python",
        "examples/hello_blocks/train.py",
        "--config",
        cfg,
        "--no-check",
    ]
    for o in overrides:
        cmd += ["--set", o]
    return subprocess.Popen(
        cmd,
        cwd=ROOT,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        env={**os.environ, **(env or {})},
    )


def wait_for_blocks(run_name: str, count: int, timeout_s: float = 300) -> None:
    """Block until at least ``count`` audit records exist for the run (training is under way)."""
    fs, root = resolve(str(shared() / "blocks"), create=False)
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        try:
            if len(read_audit(fs, root, run_name)) >= count:
                return
        except Exception:
            pass
        time.sleep(1.0)
    raise TimeoutError(f"{run_name}: fewer than {count} blocks consumed after {timeout_s}s")


def finish(proc: subprocess.Popen, timeout_s: float = 600, name: str = "train") -> str:
    out, _ = proc.communicate(timeout=timeout_s)
    logs = ROOT / ".harness" / "logs"
    logs.mkdir(parents=True, exist_ok=True)
    (logs / f"{name}.log").write_text(out)
    print(f"driver output: .harness/logs/{name}.log")
    if proc.returncode != 0:
        raise RuntimeError(f"training exited {proc.returncode}:\n{out[-2500:]}")
    return out


def records_for(run_name: str):
    fs, root = resolve(str(shared() / "blocks"), create=False)
    return read_audit(fs, root, run_name)


def fresh(run_name: str) -> None:
    import shutil

    shutil.rmtree(shared() / "runs" / run_name, ignore_errors=True)
    shutil.rmtree(audit_dir(run_name), ignore_errors=True)


def ensure_blocks(cfg: str = HARNESS_CFG) -> None:
    driver("exec-head", "python", "examples/hello_blocks/make_blocks.py", "--config", cfg)


# ---- scenarios ----


def scenario_s2() -> list[str]:
    """Worker kill mid-run: the container restarts, Ray restarts the group, positions continue."""
    driver("up", "2")
    ensure_blocks()
    fresh("s2")
    proc = start_train(HARNESS_CFG, "run_name=s2")
    wait_for_blocks("s2", 12)  # a few steps into segment 0
    driver("kill-worker", "2")
    finish(proc, name="s2")
    recs = records_for("s2")
    print(summarize(recs))
    return check_recovery(recs, W, every_k=2, expected_world_sizes=[2, 2])


def scenario_s3() -> list[str]:
    """Elastic scale up 2 -> 3 while training; the tail is re-dealt over 3 ranks."""
    driver("up", "2")
    ensure_blocks()
    fresh("s3")
    proc = start_train(HARNESS_CFG, "run_name=s3")
    wait_for_blocks("s3", 12)
    driver("scale", "3")
    finish(proc, name="s3")
    recs = records_for("s3")
    print(summarize(recs))
    return check_recovery(recs, W, every_k=2, expected_world_sizes=[2, 3])


def scenario_s4() -> list[str]:
    """Scale down 3 -> 2 while training (the removed worker is stopped, not restarted)."""
    driver("up", "3")
    ensure_blocks()
    fresh("s4")
    proc = start_train(HARNESS_CFG, "run_name=s4")
    wait_for_blocks("s4", 12)
    driver("scale", "2")
    finish(proc, name="s4")
    recs = records_for("s4")
    print(summarize(recs))
    return check_recovery(recs, W, every_k=2, expected_world_sizes=[3, 2])


def head_python(code: str, env: dict[str, str] | None = None) -> str:
    return driver("exec-head", "python", "-c", code, env=env)


def latest_checkpoint(run_uri: str, env: dict[str, str]) -> tuple[str, int]:
    """``(checkpoint uri, first position after it)`` for the newest checkpoint of a run on S3;
    listed from inside the head container, which has the S3 credentials."""
    out = (
        head_python(
            "import re, pyarrow.fs as pafs\n"
            "from distrainer.storage import resolve, s3_options_from_env\n"
            f"fs, root = resolve({run_uri!r}, create=False, **s3_options_from_env())\n"
            "infos = fs.get_file_info(pafs.FileSelector(root, recursive=False))\n"
            "names = sorted(i.path.rsplit('/', 1)[-1] for i in infos if 'checkpoint_g' in i.path)\n"
            "print(names[-1] if names else '')\n",
            env=env,
        )
        .strip()
        .splitlines()[-1]
    )
    if not out:
        raise RuntimeError(f"no checkpoints under {run_uri}")
    import re

    m = re.match(r"checkpoint_g(\d+)_p(\d+)_n(\d+)_a(\d+)", out)
    assert m, out
    start = int(m.group(1)) * W + int(m.group(2))
    return f"{run_uri}/{out}", start


def resume_check(run_name: str, start: int, env: dict[str, str]) -> list[str]:
    out = driver(
        "exec-head",
        "python",
        "integration_tests/cluster/check_audit.py",
        "--store-root",
        "s3://distrainer/blocks",
        "--run-name",
        run_name,
        "--scenario",
        "resume",
        "--W",
        str(W),
        "--start-position",
        str(start),
        env=env,
        check=False,
    )
    lines = [ln for ln in out.splitlines() if ln.strip()]
    print("\n".join(lines[-3:]))
    return (
        []
        if lines and lines[-1].endswith("PASS")
        else [ln for ln in lines if "FAIL" in ln] or ["resume check failed"]
    )


def cli_resume(ckpt_uri: str, run_name: str, env: dict[str, str]) -> None:
    driver(
        "exec-head",
        "python",
        "-m",
        "distrainer.cli",
        "resume",
        ckpt_uri,
        "--config",
        MINIO_CFG,
        "--entry",
        "examples.hello_blocks.train:entry",
        "--run-name",
        run_name,
        env=env,
    )


def scenario_s9() -> list[str]:
    """Cold restore: a full run on MinIO, the cluster torn down and the shared mount wiped, the
    cluster brought back, `distrainer resume` from the newest checkpoint URI into a new run."""
    env = {"DISTRAINER_MINIO": "1"}
    driver("up", "2", "minio", env=env)
    driver("mkbucket", "distrainer", env=env)
    driver(
        "exec-head",
        "python",
        "examples/hello_blocks/make_blocks.py",
        "--config",
        MINIO_CFG,
        env=env,
    )
    finish(start_train(MINIO_CFG, "run_name=s9", env=env), name="s9")
    driver("down", env=env)
    driver("wipe-shared", env=env)
    driver("up", "2", "minio", env=env)
    ckpt, start = latest_checkpoint("s3://distrainer/runs/s9", env)
    print(f"S9 resuming from {ckpt} (position {start})")
    cli_resume(ckpt, "s9_resume", env)
    return resume_check("s9_resume", start, env)


def scenario_s10() -> list[str]:
    """Head loss mid-run: the head container (Ray head, Train controller, driver) is killed,
    brought back, and the run continues in a new run from its latest checkpoint on MinIO."""
    env = {"DISTRAINER_MINIO": "1"}
    driver("up", "2", "minio", env=env)
    driver("mkbucket", "distrainer", env=env)
    driver(
        "exec-head",
        "python",
        "examples/hello_blocks/make_blocks.py",
        "--config",
        MINIO_CFG,
        env=env,
    )
    proc = start_train(MINIO_CFG, "run_name=s10", env=env)
    time.sleep(25)  # into the run: the audit is on S3, so wait by time rather than by file
    subprocess.run(["docker", "kill", "distrainer-head-1"], check=True, capture_output=True)
    proc.communicate(timeout=120)  # the exec dies with the head
    driver("up", "2", "minio", env=env)  # recreates the head; workers rejoin it
    time.sleep(10)
    ckpt, start = latest_checkpoint("s3://distrainer/runs/s10", env)
    print(f"S10 resuming from {ckpt} (position {start})")
    cli_resume(ckpt, "s10_resume", env)
    return resume_check("s10_resume", start, env)


SCENARIOS = {
    "S2": scenario_s2,
    "S3": scenario_s3,
    "S4": scenario_s4,
    "S9": scenario_s9,
    "S10": scenario_s10,
}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenario", default="all", choices=["all", *SCENARIOS])
    ap.add_argument("--keep-up", action="store_true", help="leave the cluster running afterwards")
    args = ap.parse_args(argv)
    names = list(SCENARIOS) if args.scenario == "all" else [args.scenario]
    failed = 0
    try:
        for name in names:
            t0 = time.monotonic()
            try:
                problems = SCENARIOS[name]()
            except Exception as exc:  # report and continue with the next scenario
                problems = [f"runner error: {exc}"]
            for p in problems:
                print(f"{name} FAIL: {p}")
            print(f"{name}: {'PASS' if not problems else 'FAIL'} ({time.monotonic() - t0:.0f}s)")
            failed += bool(problems)
    finally:
        if not args.keep_up:
            driver("down", check=False)
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
