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
from distrainer.config import load_config  # noqa: E402
from distrainer.storage import resolve  # noqa: E402
from integration_tests.cluster.check_audit import (  # noqa: E402
    check_recovery,
    summarize,
)

DRIVER = ROOT / "deploy" / "driver.sh"
HARNESS_CFG = "examples/hello_blocks/harness.yaml"
MINIO_CFG = "examples/hello_blocks/harness-minio.yaml"
_cfg = load_config(str(ROOT / HARNESS_CFG))
_minio_cfg = load_config(str(ROOT / MINIO_CFG))
W = _cfg.log.W
SEGMENTS = int(_cfg.train["n_blocks"]) // W  # segments per pass in the harness workload
S3_BLOCKS = f"s3://{_minio_cfg.store_root}"
S3_RUNS = f"s3://{_minio_cfg.storage_path}"


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


def wait_for_trainers(n: int, timeout_s: float = 180) -> None:
    """Block until Ray reports ``n`` `trainer` resources (all worker containers have joined)."""
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        out = driver(
            "exec-head",
            "python",
            "-c",
            "import ray; ray.init(address='auto', logging_level='ERROR'); "
            "print(int(ray.cluster_resources().get('trainer', 0)))",
            check=False,
        )
        lines = [ln for ln in out.strip().splitlines() if ln.strip().isdigit()]
        if lines and int(lines[-1]) >= n:
            return
        time.sleep(3)
    raise TimeoutError(f"fewer than {n} trainer resources after {timeout_s}s")


def up(n: int, *extra: str, env: dict[str, str] | None = None) -> None:
    driver("up", str(n), *extra, env=env)
    wait_for_trainers(n)


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
    proc = subprocess.Popen(
        cmd,
        cwd=ROOT,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        env={**os.environ, **(env or {})},
    )
    _running.append(proc)
    return proc


_running: list[subprocess.Popen] = []  # training processes to kill if a scenario aborts


def _check_alive(proc: subprocess.Popen | None) -> None:
    if proc is not None and proc.poll() is not None and proc.returncode != 0:
        out = proc.stdout.read() if proc.stdout else ""
        raise RuntimeError(f"training exited early ({proc.returncode}):\n{out[-2000:]}")


def wait_for_blocks(
    run_name: str, count: int, timeout_s: float = 300, proc: subprocess.Popen | None = None
) -> None:
    """Block until at least ``count`` audit records exist for the run (training is under way)."""
    fs, root = resolve(str(shared() / "blocks"), create=False)
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        _check_alive(proc)
        try:
            if len(read_audit(fs, root, run_name)) >= count:
                return
        except Exception:
            pass
        time.sleep(1.0)
    raise TimeoutError(f"{run_name}: fewer than {count} blocks consumed after {timeout_s}s")


def wait_for_blocks_s3(
    run_name: str,
    count: int,
    env: dict[str, str],
    timeout_s: float = 300,
    proc: subprocess.Popen | None = None,
) -> None:
    """``wait_for_blocks`` for a run whose audit trail is on the bucket (read inside the head)."""
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        _check_alive(proc)
        out = (
            head_python(
                "from distrainer.audit import read_audit\n"
                "from distrainer.storage import resolve, s3_options_from_env\n"
                f"fs, root = resolve({S3_BLOCKS!r}, create=False, **s3_options_from_env())\n"
                f"print(len(read_audit(fs, root, {run_name!r})))\n",
                env=env,
            )
            .strip()
            .splitlines()
        )
        if out and out[-1].isdigit() and int(out[-1]) >= count:
            return
        time.sleep(2.0)
    raise TimeoutError(f"{run_name}: fewer than {count} blocks on S3 after {timeout_s}s")


def finish(proc: subprocess.Popen, timeout_s: float = 600, name: str = "train") -> str:
    try:
        out, _ = proc.communicate(timeout=timeout_s)
    except subprocess.TimeoutExpired:
        proc.kill()
        out, _ = proc.communicate()
        raise RuntimeError(f"training did not finish within {timeout_s}s:\n{out[-2000:]}") from None
    finally:
        if proc in _running:
            _running.remove(proc)
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
    up(2)
    ensure_blocks()
    fresh("s2")
    proc = start_train(HARNESS_CFG, "run_name=s2")
    wait_for_blocks("s2", 12)  # a few steps into segment 0
    driver("kill-worker", "2")
    finish(proc, name="s2")
    recs = records_for("s2")
    print(summarize(recs))
    # Ray may need more than one restart while the dead node's heartbeat times out; every
    # attempt must run at world size 2 and the trail must be consistent across all of them
    problems = check_recovery(recs, W, every_k=2, expected_segments=SEGMENTS)
    sizes = sorted({r.world_size for r in recs})
    if sizes != [2]:
        problems.append(f"world sizes {sizes}, expected only 2")
    return problems


def scenario_s3() -> list[str]:
    """Elastic scale up 2 -> 3 while training; the tail is re-dealt over 3 ranks."""
    up(2)
    ensure_blocks()
    fresh("s3")
    proc = start_train(HARNESS_CFG, "run_name=s3")
    wait_for_blocks("s3", 12, proc=proc)
    driver("scale", "3")
    finish(proc, name="s3")
    recs = records_for("s3")
    print(summarize(recs))
    return check_recovery(
        recs, W, every_k=2, expected_world_sizes=[2, 3], expected_segments=SEGMENTS
    )


def scenario_s4() -> list[str]:
    """Scale down 3 -> 2 while training (the removed worker is stopped, not restarted)."""
    up(3)
    ensure_blocks()
    fresh("s4")
    proc = start_train(HARNESS_CFG, "run_name=s4")
    wait_for_blocks("s4", 12, proc=proc)
    driver("scale", "2")
    finish(proc, name="s4")
    recs = records_for("s4")
    print(summarize(recs))
    return check_recovery(
        recs, W, every_k=2, expected_world_sizes=[3, 2], expected_segments=SEGMENTS
    )


def s3_rm(prefixes: list[str], env: dict[str, str]) -> None:
    """Delete run state and audit trails of earlier scenario runs from the bucket."""
    head_python(
        "from distrainer.storage import resolve, s3_options_from_env\n"
        f"for uri in {prefixes!r}:\n"
        "    fs, root = resolve(uri, create=False, **s3_options_from_env())\n"
        "    try:\n"
        "        fs.delete_dir(root)\n"
        "    except FileNotFoundError:\n"
        "        pass\n"
        "print('cleaned')\n",
        env=env,
    )


def head_python(code: str, env: dict[str, str] | None = None) -> str:
    return driver("exec-head", "python", "-c", code, env=env)


def pick_checkpoint(
    run_uri: str, env: dict[str, str], which: str = "latest"
) -> tuple[str, int, int]:
    """``(checkpoint uri, ledger segment, ledger positions)`` for a checkpoint of a run on S3,
    chosen among those with a readable ``.metadata.json`` ledger (a partial upload left by a
    killed head has none). ``which``: ``latest`` or ``middle``, by data position."""
    out = (
        head_python(
            "import json\n"
            "from integration_tests.cluster.check_audit import checkpoint_ledgers\n"
            f"led = checkpoint_ledgers({run_uri!r})\n"
            "rows = sorted(\n"
            "    (int(v['segment']), int(v['cursor']) * int(v['world_size']), k)\n"
            "    for k, v in led.items()\n"
            ")\n"
            "print(json.dumps(rows))\n",
            env=env,
        )
        .strip()
        .splitlines()
    )
    import json

    rows = json.loads(out[-1]) if out else []
    if not rows:
        raise RuntimeError(f"no checkpoints with a ledger under {run_uri}")
    seg, positions, name = rows[-1] if which == "latest" else rows[len(rows) // 2]
    return f"{run_uri}/{name}", int(seg), int(positions)


def resume_check(run_name: str, segment: int, positions: int, env: dict[str, str]) -> list[str]:
    """Run check_audit --scenario resume inside the head (the audit is on S3); the start position
    is derived from the ledger and the world size the resumed run actually had."""
    proc = subprocess.run(
        [
            str(DRIVER),
            "exec-head",
            "python",
            "integration_tests/cluster/check_audit.py",
            "--store-root",
            S3_BLOCKS,
            "--run-name",
            run_name,
            "--scenario",
            "resume",
            "--W",
            str(W),
            "--ledger-segment",
            str(segment),
            "--ledger-positions",
            str(positions),
            "--expected-segments",
            str(SEGMENTS),
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
        env={**os.environ, **env},
    )
    lines = [ln for ln in (proc.stdout + proc.stderr).splitlines() if ln.strip()]
    print("\n".join(lines[-3:]))
    if proc.returncode == 0 and lines and lines[-1].endswith("PASS"):
        return []
    return [ln for ln in lines if "FAIL" in ln or "Error" in ln][-5:] or ["resume check failed"]


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
    up(2, "minio", env=env)
    driver("mkbucket", "distrainer", env=env)
    s3_rm(
        [
            f"{S3_RUNS}/s9",
            f"{S3_RUNS}/s9_resume",
            f"{S3_BLOCKS}/audit/s9",
            f"{S3_BLOCKS}/audit/s9_resume",
        ],
        env,
    )
    driver(
        "exec-head",
        "python",
        "examples/hello_blocks/make_blocks.py",
        "--config",
        MINIO_CFG,
        env=env,
    )
    finish(start_train(MINIO_CFG, "run_name=s9", "checkpoint.num_to_keep=null", env=env), name="s9")
    driver("down", env=env)
    driver("wipe-shared", env=env)
    up(2, "minio", env=env)
    # a completed run's newest checkpoint is the end of the log: resume from the middle one
    ckpt, seg, positions = pick_checkpoint(f"{S3_RUNS}/s9", env, which="middle")
    print(f"S9 resuming from {ckpt} (segment {seg}, {positions} positions done)")
    cli_resume(ckpt, "s9_resume", env)
    return resume_check("s9_resume", seg, positions, env)


def scenario_s10() -> list[str]:
    """Head loss mid-run: the head container (Ray head, Train controller, driver) is killed,
    brought back, and the run continues in a new run from its latest checkpoint on MinIO."""
    env = {"DISTRAINER_MINIO": "1"}
    up(2, "minio", env=env)
    driver("mkbucket", "distrainer", env=env)
    s3_rm(
        [
            f"{S3_RUNS}/s10",
            f"{S3_RUNS}/s10_resume",
            f"{S3_BLOCKS}/audit/s10",
            f"{S3_BLOCKS}/audit/s10_resume",
        ],
        env,
    )
    driver(
        "exec-head",
        "python",
        "examples/hello_blocks/make_blocks.py",
        "--config",
        MINIO_CFG,
        env=env,
    )
    proc = start_train(MINIO_CFG, "run_name=s10", env=env)
    wait_for_blocks_s3("s10", 24, env, proc=proc)  # a segment in, with checkpoints registered
    driver("kill-head", env=env)
    proc.communicate(timeout=120)  # the exec dies with the head
    _running.remove(proc)
    up(2, "minio", env=env)  # recreates the head; workers rejoin it
    ckpt, seg, positions = pick_checkpoint(f"{S3_RUNS}/s10", env)
    print(f"S10 resuming from {ckpt} (segment {seg}, {positions} positions done)")
    cli_resume(ckpt, "s10_resume", env)
    return resume_check("s10_resume", seg, positions, env)


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
            finally:
                for proc in list(_running):  # a scenario that raised must not leave a run behind
                    proc.kill()
                    _running.remove(proc)
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
