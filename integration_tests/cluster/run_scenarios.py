"""Cluster scenarios (spec section 10) driven through the deploy/driver.sh verbs.

S2 worker kill mid-run, S3 elastic scale up, S4 scale down, S6 re-mining hook, S8 time-budget
policy, S9 cold restore and S10 head loss against MinIO, S11 streaming producer (S11s3: the same
with the log and the run on MinIO, so the segment put is the commit). Each scenario
brings the cluster to the state it needs, runs an example inside the head container, injects
the failure while training runs, waits for the run to finish, and checks the audit trail on the
shared mount, or on the bucket when the driver has no shared mount (``shared`` prints nothing:
S2, S3, S4 and S8 then run on the bucket the driver's ``endpoint`` names: MinIO in the cluster,
or a store outside it such as S3 when it prints ``s3=``; S6 and S11 need the mount and are
skipped). Run: ``just integration S2`` (or ``all``).
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from distrainer.audit import AuditRecord, read_audit  # noqa: E402
from distrainer.config import load_config  # noqa: E402
from distrainer.log import BlockLog  # noqa: E402
from distrainer.storage import exists, resolve  # noqa: E402
from integration_tests.cluster.check_audit import (  # noqa: E402
    by_attempt,
    check_recovery,
    check_s6,
    check_s8,
    check_streaming_run,
    checkpoint_ledgers,
    summarize,
)

DRIVER = ROOT / "deploy" / "driver.sh"
HARNESS_CFG = "examples/hello_blocks/harness.yaml"
MINIO_CFG = "examples/hello_blocks/harness-minio.yaml"
S3_CFG = "examples/hello_blocks/harness-s3.yaml"  # the same on a store outside the cluster
REMINE_CFG = "examples/toy_contrastive/harness-remine.yaml"
STREAM_CFG = "examples/hello_blocks/harness-stream.yaml"
STREAM_MINIO_CFG = "examples/hello_blocks/harness-stream-minio.yaml"
STREAM_S3_CFG = "examples/hello_blocks/harness-stream-s3.yaml"
DOTENV = ROOT / ".env"
_cfg = load_config(str(ROOT / HARNESS_CFG))
_minio_cfg = load_config(str(ROOT / MINIO_CFG))
W = _cfg.log.W
SEGMENTS = int(_cfg.train["n_blocks"]) // W  # segments per pass in the harness workload
MINIO_SEGMENTS = int(_minio_cfg.train["n_blocks"]) // _minio_cfg.log.W
S3_BLOCKS = f"s3://{_minio_cfg.store_root}"
S3_RUNS = f"s3://{_minio_cfg.storage_path}"


def driver(*args: str, env: dict[str, str] | None = None, check: bool = True) -> str:
    proc = subprocess.run(
        [str(DRIVER), *args],
        cwd=ROOT,
        capture_output=True,
        text=True,
        env={**os.environ, **(env or {})},
        timeout=600,
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


class SkipScenario(Exception):
    """The scenario needs something this driver does not offer (a shared mount)."""


def shared() -> Path:
    """The host path of the shared mount; a scenario that reads it cannot run without one."""
    out = driver("shared").strip()
    if not out:
        raise SkipScenario("needs a shared mount, and this driver has none (storage is S3 only)")
    return Path(out)


def store_endpoint(endpoint_output: str) -> tuple[str, str | None]:
    """``("minio", url)`` or ``("s3", url)`` from the output of the driver's ``endpoint`` verb:
    MinIO in the cluster (``minio=http://h:9000 ...``), or a store outside it that the containers
    reach directly (``s3=https://...``, the uncloud driver with S3_ENDPOINT set in .env). A
    MinIO line without a usable host (kuberay prints ``minio=http://<no minio service>:9000``
    while nothing is deployed) is ``("minio", None)``: the store is MinIO, its URL comes after
    ``up``. A store outside the cluster must be a URL."""
    from urllib.parse import urlparse

    for line in endpoint_output.splitlines():
        for kind in ("minio", "s3"):
            if line.startswith(f"{kind}="):
                url = line.split()[0][len(kind) + 1 :]
                host = urlparse(url).hostname or ""
                if not host or not all(c.isalnum() or c in ".-_" for c in host):
                    if kind == "minio":
                        return kind, None
                    raise RuntimeError(f"the driver's endpoint verb names no usable store: {line}")
                return kind, url
    raise RuntimeError(
        f"the driver's endpoint verb names no store (minio= or s3=):\n{endpoint_output}"
    )


def read_dotenv(path: Path, values: dict[str, str]) -> dict[str, str]:
    """Fill ``values`` from the ``KEY=value`` lines of an env file (comments and ``export``
    tolerated, quotes stripped); keys absent from ``values`` are ignored."""
    for raw in path.read_text().splitlines():
        line = raw.strip()
        if line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[len("export ") :]
        key, sep, value = line.partition("=")
        if not sep or key.strip() not in values:
            continue
        value = value.strip()
        if (
            value[:1] in ("'", '"') and value.count(value[0]) >= 2
        ):  # quoted: up to the closing quote
            value = value[1 : value.index(value[0], 1)]
        else:  # bare: an inline comment ends it, as it would for the shell
            value = value.split(" #", 1)[0].split("\t#", 1)[0].strip()
        values[key.strip()] = value
    return values


def s3_settings_from_dotenv() -> dict[str, str]:
    """The S3 credentials and region the containers use: .env when present (the drivers source
    it), then, under the uncloud driver, the file DISTRAINER_ENV_FILE names (in the environment,
    or in .env: a bootstrap's env file, read after .env as that driver does, so its settings
    win; skipped when the caller pins DISTRAINER_UNCLOUD_PROVIDER to another bed than the file
    declares), else the compose defaults. Returned, not exported: the driver subprocesses
    inherit this process's environment and compose interpolates these names."""
    values = {"S3_ACCESS_KEY": "distrainer", "S3_SECRET_KEY": "distrainer123", "S3_REGION": "auto"}
    pointer = os.environ.get("DISTRAINER_ENV_FILE", "")
    if DOTENV.exists():
        pointer = pointer or read_dotenv(DOTENV, {"DISTRAINER_ENV_FILE": ""})["DISTRAINER_ENV_FILE"]
        read_dotenv(DOTENV, values)
    if pointer and os.environ.get("DISTRAINER_DRIVER", "compose") != "uncloud":
        pointer = ""  # the env file belongs to an uncloud bed: the other drivers never read it
    if pointer:
        extra = Path(pointer) if pointer.startswith("/") else ROOT / pointer
        if not extra.exists():
            raise RuntimeError(f"DISTRAINER_ENV_FILE={pointer} does not exist")
        pinned = os.environ.get("DISTRAINER_UNCLOUD_PROVIDER", "")
        declared = read_dotenv(extra, {"DISTRAINER_UNCLOUD_PROVIDER": ""})[
            "DISTRAINER_UNCLOUD_PROVIDER"
        ]
        if not pinned or not declared or pinned == declared:
            read_dotenv(extra, values)
    for key in values:  # what the shell exports wins over both files, as it does for the drivers
        if key in os.environ:
            values[key] = os.environ[key]
    return values


@dataclass(frozen=True)
class Store:
    """Where the runner reads a scenario's audit trail, run state and checkpoints: the shared
    mount when the driver has one, else a bucket reached from the Mac at the URL of the driver's
    ``endpoint`` verb: MinIO in the cluster (``harness-minio.yaml``), or a store outside it
    (``harness-s3.yaml``, ``external``) that the containers reach directly and that exists before
    the cluster does. ``check_audit`` does not care."""

    blocks: str  # URI of the block store; the audit trail is under <blocks>/audit/<run>
    runs: str  # URI of the run state; the checkpoints are under <runs>/<run>
    cfg: str  # the hello_blocks config that writes there
    env: dict[str, str]  # driver environment (DISTRAINER_MINIO=1 when MinIO is the store)
    s3: dict[str, Any]  # resolve() options from the Mac (endpoint, region) on the bucket
    segments: int  # segments per pass of that config's workload
    credentials: dict[str, str] | None = (
        None  # S3_ACCESS_KEY / S3_SECRET_KEY for the Mac-side client
    )
    external: bool = (
        False  # the bucket is outside the cluster: no MinIO to deploy or bucket to make
    )

    @property
    def lag_intervals(self) -> int:
        """Extra checkpoint intervals a restarted attempt may replay (``check_recovery``): on a
        store outside the cluster the Train controller registers checkpoints a couple of reports
        behind the workers (its snapshot write and ``num_to_keep`` deletions are S3 round trips
        per report) and the teardown lets them run a few more steps; measured on AWS: 24 and 18
        positions against bounds of 11 and 14, so 4 (bounds 27 and 38) is the smallest value
        that passes with some headroom. The same mechanism applies to every restart there (S2
        passed within the base bound once). Zero when the store is in the cluster."""
        return 4 if self.external else 0

    @property
    def on_bucket(self) -> bool:
        return self.blocks.startswith("s3://")

    @property
    def bucket(self) -> str:
        return self.blocks[len("s3://") :].split("/", 1)[0]

    @property
    def W(self) -> int:  # noqa: N802 (the spec's name)
        return load_config(str(ROOT / self.cfg)).log.W

    def fs(self, uri: str):
        if self.on_bucket and not self.s3.get("endpoint"):
            raise RuntimeError(
                "the bucket's URL is not known before `up` (up_bucket returns the store)"
            )
        with self.mac_env():  # the S3 client takes its credentials from the environment
            return resolve(uri, create=False, **self.s3)

    def records(self, run_name: str) -> list[AuditRecord]:
        fs, root = self.fs(self.blocks)
        return read_audit(fs, root, run_name)

    def wait_for_blocks(
        self,
        run_name: str,
        count: int,
        timeout_s: float = 300,
        proc: subprocess.Popen | None = None,
    ) -> None:
        """Block until at least ``count`` audit records exist for the run (training runs)."""
        deadline = time.monotonic() + timeout_s
        seen, last_error = 0, ""
        while time.monotonic() < deadline:
            _check_alive(proc)
            try:
                seen = len(self.records(run_name))
                if seen >= count:
                    return
            except Exception as exc:  # no trail yet, or the store unreachable: keep polling
                last_error = f"{type(exc).__name__}: {str(exc)[:200]}"
            time.sleep(2.0 if self.on_bucket else 1.0)
        raise TimeoutError(
            f"{run_name}: fewer than {count} blocks consumed after {timeout_s}s ({seen} seen"
            + (f"; last read error {last_error}" if last_error else "")
            + ")"
        )

    def fresh(self, run_name: str) -> None:
        """Remove the run state and audit trail of an earlier run with this name (a run name
        that exists is restored by Ray Train, not started afresh). An unreachable store is an
        error, not "nothing there": a stale run left behind would be restored silently."""
        for uri in (f"{self.runs}/{run_name}", f"{self.blocks}/audit/{run_name}"):
            fs, root = self.fs(uri)
            if exists(fs, root):
                fs.delete_dir(root)

    @contextmanager
    def mac_env(self):
        """The bucket as the Mac sees it, in the environment: ``S3_ENDPOINT`` and the credentials
        for a check that resolves URIs through ``s3_options_from_env`` (``checkpoint_ledgers``)
        and for this store's own client. Never around a driver call: the driver subprocesses
        inherit the environment and compose interpolates these names into the containers."""
        if not self.on_bucket:
            yield
            return
        values = {  # real S3 signs by region and answers 400 to a wrong one (MinIO never cared)
            "S3_ENDPOINT": self.s3["endpoint"],
            "S3_REGION": str(self.s3.get("region") or "auto"),
            **(self.credentials or {}),
        }
        before = {k: os.environ.get(k) for k in values}
        os.environ.update(values)
        try:
            yield
        finally:
            for k, v in before.items():
                if v is None:
                    os.environ.pop(k, None)
                else:
                    os.environ[k] = v


def bucket_store() -> Store:
    """The store on a bucket, whichever the driver's ``endpoint`` names: MinIO in the cluster
    (``harness-minio.yaml``; ``up_bucket`` deploys it and makes the bucket), or a store outside
    it (``harness-s3.yaml``, whose endpoint must be the one the driver prints: the containers
    read the config, the driver the environment). Works before the cluster is up."""
    env = {"DISTRAINER_MINIO": "1"}
    kind, endpoint = store_endpoint(driver("endpoint", env=env))
    settings = s3_settings_from_dotenv()
    credentials = {k: settings[k] for k in ("S3_ACCESS_KEY", "S3_SECRET_KEY")}
    if kind == "s3":
        cfg = load_config(str(ROOT / S3_CFG))
        if cfg.storage.endpoint != endpoint:
            raise RuntimeError(
                f"{S3_CFG} stores at {cfg.storage.endpoint} but the driver's endpoint "
                f"(S3_ENDPOINT in .env) is {endpoint}; make them agree"
            )
        return Store(
            blocks=f"s3://{cfg.store_root}", runs=f"s3://{cfg.storage_path}", cfg=S3_CFG, env={},
            s3={"endpoint": endpoint, "region": cfg.storage.region or settings["S3_REGION"]},
            segments=int(cfg.train["n_blocks"]) // cfg.log.W, credentials=credentials,
            external=True,
        )  # fmt: skip
    return Store(
        blocks=S3_BLOCKS, runs=S3_RUNS, cfg=MINIO_CFG, env=env,
        s3={"endpoint": endpoint, "region": settings["S3_REGION"]},
        segments=MINIO_SEGMENTS, credentials=credentials,
    )  # fmt: skip


def store_for_driver() -> Store:
    """The shared mount if the driver prints one, else the bucket."""
    out = driver("shared").strip()
    if out:  # has_shared(), keeping the path
        base = Path(out)
        return Store(
            blocks=str(base / "blocks"), runs=str(base / "runs"), cfg=HARNESS_CFG, env={}, s3={},
            segments=SEGMENTS,
        )  # fmt: skip
    return bucket_store()


def has_shared() -> bool:
    return bool(driver("shared").strip())


def up_bucket(n: int, st: Store) -> Store:
    """Bring the cluster to ``n`` workers with what a bucket store needs: MinIO and the bucket
    when MinIO is the store; nothing more when the store is outside the cluster (it exists).
    Returns the store to use: the same one, or, when the driver could not name MinIO's URL
    before the deploy (kuberay), the store read again now that the service exists."""
    if st.external:
        up(n)
        return st
    up(n, "minio", env=st.env)
    driver("mkbucket", st.bucket, env=st.env)
    return st if st.s3.get("endpoint") else bucket_store()


def up_store(n: int) -> Store:
    """Bring the cluster to ``n`` workers with what the store needs and return the store."""
    if has_shared():
        up(n)
        return store_for_driver()
    return up_bucket(n, bucket_store())


def start_train(
    cfg: str,
    *overrides: str,
    env: dict[str, str] | None = None,
    name: str = "train",
    script: str = "examples/hello_blocks/train.py",
) -> subprocess.Popen:
    """Start a train.py inside the head (see ``start_head``)."""
    args = ["--config", cfg, "--no-check"]
    for o in overrides:
        args += ["--set", o]
    return start_head(script, args, env=env, name=name)


def start_head(
    script: str, args: list[str], env: dict[str, str] | None = None, name: str = "head"
) -> subprocess.Popen:
    """Start a Python script inside the head; its output streams to .harness/logs/<name>.log (a
    pipe that nobody drains while waiting would fill up and stall the process)."""
    cmd = [str(DRIVER), "exec-head", "python", script, *args]
    logs = ROOT / ".harness" / "logs"
    logs.mkdir(parents=True, exist_ok=True)
    log = open(logs / f"{name}.log", "w")  # noqa: SIM115 (closed in finish)
    proc = subprocess.Popen(
        cmd,
        cwd=ROOT,
        stdout=log,
        stderr=subprocess.STDOUT,
        text=True,
        env={**os.environ, **(env or {})},
    )
    _logs[proc.pid] = (logs / f"{name}.log", log)
    _running.append(proc)
    return proc


_running: list[subprocess.Popen] = []  # training processes to kill if a scenario aborts


_logs: dict[int, tuple[Path, Any]] = {}  # pid -> (log path, open file)


def _output(proc: subprocess.Popen) -> str:
    entry = _logs.get(proc.pid)
    if entry is None:
        return ""
    path, log_file = entry
    log_file.flush()
    return path.read_text() if path.exists() else ""


def _check_alive(proc: subprocess.Popen | None) -> None:
    if proc is not None and proc.poll() is not None and proc.returncode != 0:
        raise RuntimeError(f"training exited early ({proc.returncode}):\n{_output(proc)[-2000:]}")


def wait_for_blocks_s3(
    run_name: str,
    count: int,
    env: dict[str, str],
    blocks: str,
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
                f"fs, root = resolve({blocks!r}, create=False, **s3_options_from_env())\n"
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
    """Wait for a process started by ``start_head``; returns everything it printed."""
    try:
        proc.wait(timeout=timeout_s)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait()
        raise RuntimeError(
            f"training did not finish within {timeout_s}s:\n{_output(proc)[-2000:]}"
        ) from None
    finally:
        if proc in _running:
            _running.remove(proc)
    out = _output(proc)  # before the log entry is dropped, or there is nothing to read
    entry = _logs.pop(proc.pid, None)
    if entry is not None:
        entry[1].close()
    print(f"driver output: .harness/logs/{name}.log")
    if proc.returncode != 0:
        raise RuntimeError(f"training exited {proc.returncode}:\n{out[-2500:]}")
    return out


def fresh(run_name: str, store: str) -> None:
    """Remove the run state and audit trail of an earlier run from a store on the shared mount
    (S6 and S11 stream into a store of their own; the hello_blocks store is ``Store.fresh``)."""
    import shutil

    shutil.rmtree(shared() / "runs" / run_name, ignore_errors=True)
    shutil.rmtree(shared() / store / "audit" / run_name, ignore_errors=True)


def fresh_store(store: str) -> None:
    """Remove a whole block store under the shared mount (a streamed log must start empty)."""
    import shutil

    shutil.rmtree(shared() / store, ignore_errors=True)


def ensure_blocks(cfg: str = HARNESS_CFG) -> None:
    driver("exec-head", "python", "examples/hello_blocks/make_blocks.py", "--config", cfg)


# ---- scenarios ----


def check_transition(records, first: int, last: int) -> list[str]:
    """The first attempt ran at ``first`` ranks and the last at ``last``; extra restarts in
    between (Ray may need more than one while a node's death propagates) are allowed."""
    sizes = [attempts[0].world_size for _, attempts in sorted(by_attempt(records).items())]
    if not sizes or sizes[0] != first or sizes[-1] != last or set(sizes) - {first, last}:
        return [f"world sizes per attempt {sizes}, expected {first} -> ... -> {last}"]
    return []


def scenario_s2() -> list[str]:
    """Worker kill mid-run: the container restarts, Ray restarts the group, positions continue."""
    st = up_store(2)
    ensure_blocks(st.cfg)
    st.fresh("s2")
    proc = start_train(st.cfg, "run_name=s2", name="s2", env=st.env)
    st.wait_for_blocks("s2", 12, proc=proc)  # a few steps into segment 0
    driver("kill-worker", "2")
    finish(proc, name="s2")
    recs = st.records("s2")
    print(summarize(recs))
    # Ray may need more than one restart while the dead node's heartbeat times out; every
    # attempt must run at world size 2 and the trail must be consistent across all of them
    problems = check_recovery(
        recs, st.W, every_k=2, expected_segments=st.segments, lag_intervals=st.lag_intervals
    )
    sizes = sorted({r.world_size for r in recs})
    if sizes != [2]:
        problems.append(f"world sizes {sizes}, expected only 2")
    return problems


def scenario_s3() -> list[str]:
    """Elastic scale up 2 -> 3 while training; the tail is re-dealt over 3 ranks."""
    st = up_store(2)
    ensure_blocks(st.cfg)
    st.fresh("s3")
    proc = start_train(st.cfg, "run_name=s3", name="s3", env=st.env)
    st.wait_for_blocks("s3", 12, proc=proc)
    driver("scale", "3")
    finish(proc, name="s3")
    recs = st.records("s3")
    print(summarize(recs))
    return check_recovery(
        recs, st.W, every_k=2, expected_segments=st.segments, lag_intervals=st.lag_intervals
    ) + check_transition(recs, first=2, last=3)


def scenario_s4() -> list[str]:
    """Scale down 3 -> 2 while training (the removed worker is stopped, not restarted)."""
    st = up_store(3)
    ensure_blocks(st.cfg)
    st.fresh("s4")
    proc = start_train(st.cfg, "run_name=s4", name="s4", env=st.env)
    st.wait_for_blocks("s4", 12, proc=proc)
    driver("scale", "2")
    finish(proc, name="s4")
    recs = st.records("s4")
    print(summarize(recs))
    return check_recovery(
        recs, st.W, every_k=2, expected_segments=st.segments, lag_intervals=st.lag_intervals
    ) + check_transition(recs, first=3, last=2)


def scenario_s6() -> list[str]:
    """Re-mining hook: segment 0 is built by make_blocks.py, every later segment is mined by
    rank 0 with the current encoder at the previous segment's end; the log is streamed."""
    cfg = load_config(str(ROOT / REMINE_CFG))
    store = Path(cfg.store_root).name  # /shared/<store>
    remine = cfg.hooks["remine"]
    shared()  # a shared-mount scenario (the config stores under /shared): skip before `up`
    up(2)
    fresh_store(store)
    fresh("s6", store)
    proc = start_train(
        REMINE_CFG, "run_name=s6", name="s6", script="examples/toy_contrastive/train.py"
    )
    finish(proc, name="s6")
    fs, root = resolve(str(shared() / store), create=False)
    recs = read_audit(fs, root, "s6")
    print(summarize(recs))
    log = BlockLog.open(fs, root)
    problems = check_s6(
        recs,
        list(log.segments()),
        log.W,
        int(remine.get("initial_segments", 1)),
        expected_segments=int(remine["segments"]),
    )
    if not log.ended():
        problems.append("the hook did not end the log")
    return problems


def final_metrics(output: str) -> dict[str, Any]:
    """The ``final metrics: {...}`` dict a train.py printed."""
    import ast

    for line in output.splitlines():
        if line.startswith("final metrics: "):
            return dict(ast.literal_eval(line[len("final metrics: ") :]))
    raise RuntimeError("no 'final metrics:' line in the driver output")


def scenario_s8() -> list[str]:
    """Time-budget policy: rank 0 decides every ``time_poll_every`` steps whether
    ``time_budget_s`` have passed and broadcasts it, so every rank reports the same number of
    checkpoints (Ray Train v2 would otherwise deadlock inside report)."""
    st = up_store(2)
    ensure_blocks(st.cfg)
    st.fresh("s8")
    out = finish(
        start_train(
            st.cfg,
            "run_name=s8",
            "checkpoint.policy=time",
            "checkpoint.time_budget_s=5",
            "checkpoint.time_poll_every=2",
            "checkpoint.num_to_keep=null",
            name="s8",
            env=st.env,
        ),
        name="s8",
    )
    recs = st.records("s8")
    print(summarize(recs))
    n_reports = int(final_metrics(out).get("reports", 0))
    with st.mac_env():
        ledgers = checkpoint_ledgers(f"{st.runs}/s8")
    print(f"S8: {len(ledgers)} time-budget checkpoints, {n_reports} reports by rank 0")
    return check_s8(recs, st.W, n_reports, ledgers, poll_every=2)


def scenario_s11() -> list[str]:
    """Streaming producer: a separate process in the head creates the log and commits a segment
    every few seconds while the ranks train and wait; gc keeps the log short."""
    return streaming_scenario(STREAM_CFG, "s11")


def scenario_s11s3() -> list[str]:
    """S11 with the store and the run on a bucket (MinIO, or the store outside the cluster):
    every segment commit is one S3 put, the ranks poll the bucket, gc deletes objects behind the
    retention window."""
    st = bucket_store()
    return streaming_scenario(STREAM_S3_CFG if st.external else STREAM_MINIO_CFG, "s11s3", st)


def streaming_scenario(cfg_path: str, run_name: str, store: Store | None = None) -> list[str]:
    cfg = load_config(str(ROOT / cfg_path))
    s3 = cfg.storage.kind == "s3"
    env = store.env if store is not None else None
    # the check needs the trainer to catch up with the producer and wait: over an object store
    # across machines a segment costs the ranks 4 to 6 s (block reads and checkpoint puts over
    # the mesh) against the producer's 6.7 s cadence, and Ray Train takes 20 s to start, so the
    # bucket variant gets more segments for the trainer to close that gap
    segments, sleep_s = (14 if s3 else 10), 6.0
    if s3:
        if store is None:
            raise RuntimeError("a bucket config needs the bucket store")
        store = up_bucket(2, store)
        store_uri = f"s3://{cfg.store_root}"  # the whole streamed store belongs to this run
        run_uri = f"s3://{cfg.storage_path}/{run_name}"
        s3_rm([store_uri, run_uri], env or {})
    else:
        shared()  # a shared-mount scenario (the config stores under /shared): skip before `up`
        up(2)
        store_dir = Path(cfg.store_root).name  # /shared/<store>
        fresh_store(store_dir)
        fresh(run_name, store_dir)
        store_uri = str(shared() / store_dir)
        run_uri = str(shared() / "runs" / run_name)
    producer = start_head(
        "examples/streaming_producer/produce.py",
        ["--config", cfg_path, "--segments", str(segments), "--sleep-s", str(sleep_s)],
        env=env,
        name=f"{run_name}-producer",
    )
    deadline = time.monotonic() + (180 if s3 else 60)  # a bucket probe is a docker exec
    while not log_exists(store_uri, env):  # the trainer must find the log, not build a batch one
        _check_alive(producer)
        if time.monotonic() > deadline:
            raise TimeoutError("the producer did not create the log")
        time.sleep(0.5)
    proc = start_train(cfg_path, f"run_name={run_name}", env=env, name=run_name)
    producer_out = finish(producer, name=f"{run_name}-producer")
    finish(proc, name=run_name)
    committed_at = {
        int(ln.split()[1]): float(ln.rsplit(" ", 1)[1])
        for ln in producer_out.splitlines()
        if ln.startswith("segment ") and " committed at " in ln
    }
    args = (
        run_name,
        committed_at,
        sleep_s - cfg.log.wait_poll_s - 1.0,
        segments,
        run_uri,
        cfg.log.retention_segments,
    )
    if s3:  # the trail, the log and the checkpoints are on the bucket: check inside the head
        import json

        out = head_python(
            "import json\n"
            "from distrainer.storage import resolve, s3_options_from_env\n"
            "from integration_tests.cluster.check_audit import check_streaming_run\n"
            f"fs, root = resolve({store_uri!r}, create=False, **s3_options_from_env())\n"
            f"print(json.dumps(check_streaming_run(fs, root, *{args!r})))\n",
            env=env,
        )
        problems, info = json.loads(out.strip().splitlines()[-1])
    else:
        fs, root = resolve(store_uri, create=False)
        problems, info = check_streaming_run(fs, root, *args)
    print(info["summary"])
    print(f"{run_name.upper()} segment start gaps: {info['gaps']}")
    if info["last_ckpt"] is not None:
        print(
            f"{run_name.upper()}: log keeps segments {info['kept']}, "
            f"last checkpoint segment {info['last_ckpt']}"
        )
    return problems


def log_exists(store_uri: str, env: dict[str, str] | None = None) -> bool:
    """Whether a block log exists under ``store_uri`` (a bucket is probed inside the head; a
    failing probe counts as "not yet")."""
    if store_uri.startswith("s3://"):
        try:
            out = head_python(
                "from distrainer.log import BlockLog\n"
                "from distrainer.storage import resolve, s3_options_from_env\n"
                f"fs, root = resolve({store_uri!r}, create=False, **s3_options_from_env())\n"
                "print('LOG_EXISTS' if BlockLog(fs, root).exists() else 'NO_LOG')\n",
                env=env,
            )
        except RuntimeError:
            return False
        return "LOG_EXISTS" in out.split()
    return BlockLog(*resolve(store_uri, create=False)).exists()


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


def resume_check(
    run_name: str, segment: int, positions: int, env: dict[str, str], st: Store
) -> list[str]:
    """Run check_audit --scenario resume inside the head (the audit is on the bucket); the start
    position is derived from the ledger and the world size the resumed run actually had."""
    proc = subprocess.run(
        [
            str(DRIVER),
            "exec-head",
            "python",
            "integration_tests/cluster/check_audit.py",
            "--store-root",
            st.blocks,
            "--run-name",
            run_name,
            "--scenario",
            "resume",
            "--W",
            str(st.W),
            "--ledger-segment",
            str(segment),
            "--ledger-positions",
            str(positions),
            "--expected-segments",
            str(st.segments),
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


def cli_resume(ckpt_uri: str, run_name: str, env: dict[str, str], cfg: str) -> None:
    driver(
        "exec-head",
        "python",
        "-m",
        "distrainer.cli",
        "resume",
        ckpt_uri,
        "--config",
        cfg,
        "--entry",
        "examples.hello_blocks.train:entry",
        "--run-name",
        run_name,
        env=env,
    )


def scenario_s9() -> list[str]:
    """Cold restore: a full run on the bucket (MinIO, or the store outside the cluster), the
    cluster torn down and the shared mount wiped, the cluster brought back, `distrainer resume`
    from the newest checkpoint URI into a new run."""
    st = bucket_store()
    env = st.env
    st = up_bucket(2, st)
    s3_rm(
        [
            f"{st.runs}/s9",
            f"{st.runs}/s9_resume",
            f"{st.blocks}/audit/s9",
            f"{st.blocks}/audit/s9_resume",
        ],
        env,
    )
    ensure_blocks(st.cfg)
    finish(
        start_train(st.cfg, "run_name=s9", "checkpoint.num_to_keep=null", env=env, name="s9"),
        name="s9",
    )
    driver("down", env=env)
    driver("wipe-shared", env=env)
    st = up_bucket(2, st)
    # a completed run's newest checkpoint is the end of the log: resume from the middle one
    ckpt, seg, positions = pick_checkpoint(f"{st.runs}/s9", env, which="middle")
    print(f"S9 resuming from {ckpt} (segment {seg}, {positions} positions done)")
    cli_resume(ckpt, "s9_resume", env, st.cfg)
    return resume_check("s9_resume", seg, positions, env, st)


def scenario_s10() -> list[str]:
    """Head loss mid-run: the head container (Ray head, Train controller, driver) is killed,
    brought back, and the run continues in a new run from its latest checkpoint on the bucket."""
    st = bucket_store()
    env = st.env
    st = up_bucket(2, st)
    s3_rm(
        [
            f"{st.runs}/s10",
            f"{st.runs}/s10_resume",
            f"{st.blocks}/audit/s10",
            f"{st.blocks}/audit/s10_resume",
        ],
        env,
    )
    ensure_blocks(st.cfg)
    proc = start_train(st.cfg, "run_name=s10", env=env, name="s10")
    wait_for_blocks_s3("s10", 24, env, st.blocks, proc=proc)  # a segment in, checkpoints registered
    driver("kill-head", env=env)
    proc.communicate(timeout=120)  # the exec dies with the head
    _running.remove(proc)
    st = up_bucket(2, st)  # recreates the head; workers rejoin it
    ckpt, seg, positions = pick_checkpoint(f"{st.runs}/s10", env)
    print(f"S10 resuming from {ckpt} (segment {seg}, {positions} positions done)")
    cli_resume(ckpt, "s10_resume", env, st.cfg)
    return resume_check("s10_resume", seg, positions, env, st)


SCENARIOS = {
    "S2": scenario_s2,
    "S3": scenario_s3,
    "S4": scenario_s4,
    "S6": scenario_s6,
    "S8": scenario_s8,
    "S9": scenario_s9,
    "S10": scenario_s10,
    "S11": scenario_s11,
    "S11s3": scenario_s11s3,
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
            skipped = None
            try:
                problems = SCENARIOS[name]()
            except SkipScenario as exc:
                problems, skipped = [], str(exc)
            except Exception as exc:  # report and continue with the next scenario
                problems = [f"runner error: {exc}"]
            finally:
                if _running:  # a scenario that raised must not leave a run behind
                    for proc in list(_running):
                        proc.kill()
                        _running.remove(proc)
                    # killing the local `docker compose exec` client does not reach the process
                    # inside the container
                    driver(
                        "exec-head", "pkill", "-f", "examples/.*/(train|produce).py", check=False
                    )
            for p in problems:
                print(f"{name} FAIL: {p}")
            if skipped:
                print(f"{name}: SKIP ({skipped})")
            else:
                print(
                    f"{name}: {'PASS' if not problems else 'FAIL'} ({time.monotonic() - t0:.0f}s)"
                )
            failed += bool(problems)
    finally:
        if not args.keep_up:
            driver("down", check=False)
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
