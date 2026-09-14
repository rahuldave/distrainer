"""Single-node scenarios on a local Ray cluster with examples/hello_blocks (spec section 10).

S1 happy path, S5 checkpoint cadence, S7 determinism. Each scenario is a fresh run of
``examples/hello_blocks/train.py`` with config overrides, followed by the audit assertions.
Run: ``uv run python integration_tests/single_node/run_scenarios.py [--scenario S5]``.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from distrainer.audit import read_audit  # noqa: E402
from distrainer.config import load_config  # noqa: E402
from distrainer.log import BlockLog  # noqa: E402
from integration_tests.cluster.check_audit import (  # noqa: E402
    check_s1,
    check_s5,
    check_s7,
    expected_checkpoints,
)

CONFIG = ROOT / "examples" / "hello_blocks" / "local.yaml"


def run_hello(run_name: str, *overrides: str) -> None:
    cmd = [
        sys.executable,
        str(ROOT / "examples" / "hello_blocks" / "train.py"),
        "--config",
        str(CONFIG),
        "--no-check",
        "--set",
        f"run_name={run_name}",
    ]
    for o in overrides:
        cmd += ["--set", o]
    proc = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True)
    if proc.returncode != 0:
        tail = "\n".join(proc.stdout.splitlines()[-15:] + proc.stderr.splitlines()[-15:])
        raise RuntimeError(f"{run_name} failed (exit {proc.returncode}):\n{tail}")


def scenario_s1() -> list[str]:
    cfg = load_config(str(CONFIG), ["run_name=s1"])
    run_hello("s1")
    fs, root = cfg.store_fs()
    return check_s1(read_audit(fs, root, "s1"), BlockLog.open(fs, root).W)


def scenario_s5() -> list[str]:
    problems = []
    for every_k, policy in ((1, "every_k"), (4, "every_k"), (None, "segment_end")):
        name = f"s5_{policy}_{every_k}"
        overrides = [f"checkpoint.policy={policy}", "checkpoint.num_to_keep=null"]
        if every_k is not None:
            overrides.append(f"checkpoint.every_k={every_k}")
        cfg = load_config(str(CONFIG), [f"run_name={name}", *overrides])
        run_hello(name, *overrides)
        fs, root = cfg.store_fs()
        W = BlockLog.open(fs, root).W
        n = cfg.scaling.max_workers
        n_segments = len(read_audit(fs, root, name)) // W
        _, runs_root = cfg.runs_fs()
        expected = expected_checkpoints(n_segments, W, n, every_k, segment_end=policy != "every_k")
        problems += [f"{name}: {p}" for p in check_s5(f"{runs_root}/{name}", W, every_k, expected)]
    return problems


def scenario_s7() -> list[str]:
    cfg = load_config(str(CONFIG), ["run_name=s7a"])
    run_hello("s7a")
    run_hello("s7b")
    fs, root = cfg.store_fs()
    return check_s7(read_audit(fs, root, "s7a"), read_audit(fs, root, "s7b"))


SCENARIOS = {"S1": scenario_s1, "S5": scenario_s5, "S7": scenario_s7}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenario", default="all", choices=["all", *SCENARIOS])
    args = ap.parse_args(argv)
    names = list(SCENARIOS) if args.scenario == "all" else [args.scenario]
    failed = 0
    for name in names:
        problems = SCENARIOS[name]()
        for p in problems:
            print(f"{name} FAIL: {p}")
        print(f"{name}: {'PASS' if not problems else 'FAIL'}")
        failed += bool(problems)
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
