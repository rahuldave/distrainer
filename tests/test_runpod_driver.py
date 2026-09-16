"""deploy/drivers/runpod.sh against a stub of the REST v2 API: a fake `curl` logs every call and
answers from canned JSON files, fake `ssh`/`scp` log their arguments. The driver runs from a
scratch tree (no .env, its own .harness/runpod), so the real account is never touched."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import time
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
DRIVER = ROOT / "deploy" / "drivers" / "runpod.sh"

CURL_STUB = r"""#!/bin/bash
# a RunPod v2 stub: logs "METHOD PATH BODY", answers with RESP/<METHOD>_<path>[.<n>].json (n = the
# n-th call of that method and path) or {}, and the status from a matching .code file (200
# otherwise)
method=GET; url=""; body=""
while [ $# -gt 0 ]; do
  case "$1" in
    -X) method="$2"; shift 2 ;;
    -d) body="$2"; shift 2 ;;
    -H|-w|-o) shift 2 ;;
    http*) url="$1"; shift ;;
    *) shift ;;
  esac
done
path="${url#*/v2}"; path="${path%%\?*}"
printf '%s %s %s\n' "$method" "$path" "$body" >> "$LOG"
n=$(grep -c "^$method $path " "$LOG")
key="${method}_$(printf '%s' "$path" | sed 's|^/||; s|/|_|g')"
code=200
for c in "$RESP/$key.$n.code" "$RESP/$key.code"; do
  [ -f "$c" ] && { code=$(cat "$c"); break; }
done
for f in "$RESP/$key.$n.json" "$RESP/$key.json"; do
  [ -f "$f" ] && { cat "$f"; printf '\n%s' "$code"; exit 0; }
done
printf '{}\n%s' "$code"
"""


def pod(name, pid, marker="distrainer", head=None, status="RUNNING", dc="EU-RO-1", cost=0.24):
    env = {"S3_ENDPOINT": "https://s3.example"}
    if marker:
        env["DISTRAINER_CLUSTER"] = marker
    if head:
        env["RAY_HEAD_ADDRESS"] = f"{head}.runpod.internal:6379"
    return {
        "id": pid,
        "name": name,
        "status": status,
        "env": env,
        "gpu": {"id": "NVIDIA RTX 2000 Ada Generation", "count": 1},
        "dataCenterId": dc,
        "cost": cost,
        "globalNetworking": {
            "enabled": True,
            "ip": "10.0.0.5",
            "internalDns": f"{pid}.runpod.internal",
        },
        "ssh": {"direct": {"host": "1.2.3.4", "port": 10341}, "proxy": None},
    }


OTHERS = [  # what the account also holds: a teammate's pod, and a look-alike without the marker
    pod("other-head", "teammate1", marker=None, status="EXITED", cost=0),
    pod("distrainer-head", "lookalike", marker=None, status="EXITED", cost=0),
]
CATALOG = {
    "gpus": [
        {
            "id": "NVIDIA RTX 2000 Ada Generation",
            "price": {"secure": 0.24, "community": 0.5},
            "secure": True,
            "community": False,
        },
        {
            "id": "NVIDIA RTX A4000",
            "price": {"secure": 0.25, "community": 0.17},
            "secure": True,
            "community": True,
        },
        {
            "id": "NVIDIA A40",
            "price": {"secure": 0.49, "community": 0.35},
            "secure": True,
            "community": False,
        },
    ]
}
DATACENTERS = {
    "dataCenters": [
        {"id": "EU-RO-1", "globalNetwork": True},
        {"id": "US-TX-1", "globalNetwork": False},
        {"id": "CA-MTL-1", "globalNetwork": True},
    ]
}


class Bed:
    """A scratch tree with the driver, the stubs and the canned responses."""

    def __init__(self, tmp_path: Path):
        self.root = tmp_path / "repo"
        (self.root / "deploy" / "drivers").mkdir(parents=True)
        shutil.copy(DRIVER, self.root / "deploy" / "drivers" / "runpod.sh")
        self.bin = tmp_path / "bin"
        self.bin.mkdir()
        self.log = tmp_path / "calls.log"
        self.log.write_text("")
        self.resp = tmp_path / "resp"
        self.resp.mkdir()
        (self.bin / "curl").write_text(CURL_STUB)
        for name in ("ssh", "scp"):
            (self.bin / name).write_text(
                "#!/bin/bash\n"
                f'printf \'%s %s\\n\' {name} "$*" >> "$LOG"\n'
                'printf \'%s\' "${@: -1}" > "$LOG.remote"\n'  # what the pod's shell would run
            )
        for f in self.bin.iterdir():
            f.chmod(0o755)
        self.respond("GET", "/catalog/gpus", CATALOG)
        self.respond("GET", "/catalog/datacenters", DATACENTERS)

    def respond(
        self, method: str, path: str, body: dict, n: int | None = None, code: int | None = None
    ):
        key = f"{method}_{path.lstrip('/').replace('/', '_')}" + (f".{n}" if n else "")
        (self.resp / f"{key}.json").write_text(json.dumps(body))
        if code:
            (self.resp / f"{key}.code").write_text(str(code))

    def pods(self, *pods: dict):
        self.respond("GET", "/pods", {"pods": [*OTHERS, *pods]})
        for p in pods:
            self.respond("GET", f"/pods/{p['id']}", p)

    def run(
        self, *args: str, env: dict | None = None, ok: bool = True
    ) -> subprocess.CompletedProcess:
        base = {
            "PATH": f"{self.bin}:{os.environ['PATH']}",
            "HOME": str(self.root),
            "LOG": str(self.log),
            "RESP": str(self.resp),
            "RUNPOD_KEY": "test-key",
            "S3_ENDPOINT": "https://s3.us-east-1.amazonaws.com",
            "S3_ACCESS_KEY": "ak",
            "S3_SECRET_KEY": "sk",
            "S3_REGION": "us-east-1",
        }
        proc = subprocess.run(
            ["bash", str(self.root / "deploy" / "drivers" / "runpod.sh"), *args],
            env={**base, **(env or {})}, capture_output=True, text=True, timeout=60,
        )  # fmt: skip
        if ok:
            assert proc.returncode == 0, proc.stdout + proc.stderr
        return proc

    def calls(self) -> list[tuple[str, str, dict | None]]:
        out = []
        for line in self.log.read_text().splitlines():
            method, path, body = line.split(" ", 2)  # "METHOD PATH BODY", the body may be empty
            try:
                out.append((method, path, json.loads(body) if body.strip() else None))
            except json.JSONDecodeError:  # a line still being written by a background create
                continue
        return out

    def posts(self, path: str = "/pods") -> list[dict]:
        return [b for m, p, b in self.calls() if m == "POST" and p == path]

    def terminated(self) -> list[str]:
        return [
            p.split("/")[2]
            for m, p, b in self.calls()
            if m == "POST" and p.endswith("/action") and b == {"action": "terminate"}
        ]


@pytest.fixture
def bed(tmp_path):
    return Bed(tmp_path)


def test_endpoint_names_the_bucket_and_needs_it(bed):
    out = bed.run("endpoint").stdout.splitlines()
    assert out == ["dashboard=(no head; run 'up')", "s3=https://s3.us-east-1.amazonaws.com"]
    proc = bed.run("endpoint", env={"S3_ENDPOINT": ""}, ok=False)
    assert proc.returncode == 2 and "S3_ENDPOINT" in proc.stderr
    assert bed.run("shared").stdout == ""  # nothing spans pods


def test_up_creates_a_head_then_workers_that_dial_it_in_its_data_center(bed):
    bed.pods()  # nothing of ours yet
    head = pod("distrainer-head", "headid", status="PROVISIONING")
    running = pod("distrainer-head", "headid")
    bed.respond("POST", "/pods", head, n=1)
    bed.respond("GET", "/pods/headid", running)
    bed.respond("POST", "/pods", pod("distrainer-worker-1", "w1", head="headid"), n=2)
    bed.respond("POST", "/pods", pod("distrainer-worker-2", "w2", head="headid"), n=3)
    (bed.root / "key.pub").write_text("ssh-ed25519 AAAATEST comment\n")
    (bed.root / "key").write_text("private\n")
    out = bed.run("up", "2", env={"DISTRAINER_RUNPOD_SSH_KEY": str(bed.root / "key")})
    posts = bed.posts()
    assert all(p["env"]["PUBLIC_KEY"] == "ssh-ed25519 AAAATEST comment" for p in posts)
    assert [p["name"] for p in posts] == [
        "distrainer-head",
        "distrainer-worker-1",
        "distrainer-worker-2",
    ]
    h = posts[0]
    assert h["image"] == "ghcr.io/rahuldave/distrainer-gpu:latest" and h["args"] == "head"
    assert h["ports"] == ["22/tcp"] and h["globalNetworking"] is True and h["startSsh"] is True
    assert h["cloud"] == "SECURE" and h["disk"] == 20
    assert h["dataCenterIds"] == ["EU-RO-1", "CA-MTL-1"]  # every global-networking data center
    assert h["gpu"] == {
        "id": "NVIDIA RTX 2000 Ada Generation",
        "count": 1,
        "minRamPerGpu": 8,
        "minVcpuCountPerGpu": 2,
    }
    assert h["env"]["DISTRAINER_CLUSTER"] == "distrainer" and h["env"]["DISTRAINER_ROLE"] == "head"
    assert (
        h["env"]["S3_ENDPOINT"] == "https://s3.us-east-1.amazonaws.com"
        and h["env"]["S3_SECRET_KEY"] == "sk"
    )
    assert h["env"]["RAY_health_check_period_ms"] == "1000" and "RAY_HEAD_ADDRESS" not in h["env"]
    for w in posts[1:]:
        assert (
            w["args"] == "worker" and w["env"]["RAY_HEAD_ADDRESS"] == "headid.runpod.internal:6379"
        )
        assert w["dataCenterIds"] == ["EU-RO-1"]  # the head's data center, not the whole list
    assert (
        bed.root / ".harness" / "runpod" / "distrainer-head.ssh"
    ).read_text() == "1.2.3.4 10341\n"
    assert (
        "created distrainer-head: headid on NVIDIA RTX 2000 Ada Generation in EU-RO-1 at 0.24 USD/h"
        in out.stderr
    )
    assert out.stdout.startswith("NAME\tID")  # the ps at the end
    assert bed.terminated() == []
    proc = bed.run("up", "2", "minio", ok=False)
    assert proc.returncode == 2 and "no MinIO" in proc.stderr
    proc = bed.run("up", "0", env={"DISTRAINER_RUNPOD_SSH_KEY": str(bed.root / "nokey")}, ok=False)
    assert proc.returncode == 1 and "has no" in proc.stderr and ".pub" in proc.stderr


def test_spend_guard_skips_pricey_types_and_capacity_errors_fall_through(bed):
    bed.pods()
    # A4000 (0.25) is above a 0.245 cap: skipped; the 2000 Ada (0.24) is taken
    bed.respond("POST", "/pods", pod("distrainer-head", "headid"), n=1)
    bed.respond("GET", "/pods/headid", pod("distrainer-head", "headid"))
    types = "NVIDIA RTX A4000,NVIDIA RTX 2000 Ada Generation,NVIDIA A40"
    proc = bed.run(
        "up",
        "0",
        env={"DISTRAINER_RUNPOD_GPU_TYPES": types, "DISTRAINER_RUNPOD_MAX_GPU_HOURLY": "0.245"},
    )
    assert [p["gpu"]["id"] for p in bed.posts()] == ["NVIDIA RTX 2000 Ada Generation"]
    assert "NVIDIA RTX A4000 is 0.25 USD/h" in proc.stderr and "skipped" in proc.stderr
    # nothing affordable: no pod, exit 1
    bed2 = Bed(bed.root.parent / "two")
    bed2.pods()
    proc = bed2.run("up", "0", env={"DISTRAINER_RUNPOD_MAX_GPU_HOURLY": "0.10"}, ok=False)
    assert proc.returncode == 1 and bed2.posts() == [] and "could be rented" in proc.stderr
    # the community cloud does not sell the A40: skipped, nothing rented
    bed4 = Bed(bed.root.parent / "four")
    bed4.pods()
    proc = bed4.run(
        "up",
        "0",
        env={"DISTRAINER_RUNPOD_CLOUD": "COMMUNITY", "DISTRAINER_RUNPOD_GPU_TYPES": "NVIDIA A40"},
        ok=False,
    )
    assert proc.returncode == 1 and "no COMMUNITY price" in proc.stderr and bed4.posts() == []
    # no capacity for the first type (HTTP 400): the next type is tried
    bed3 = Bed(bed.root.parent / "three")
    bed3.pods()
    bed3.respond("POST", "/pods", {"title": "no capacity"}, n=1, code=400)
    bed3.respond("POST", "/pods", pod("distrainer-head", "headid"), n=2)
    bed3.respond("GET", "/pods/headid", pod("distrainer-head", "headid"))
    proc = bed3.run("up", "0")
    assert [p["gpu"]["id"] for p in bed3.posts()] == [
        "NVIDIA RTX 2000 Ada Generation",
        "NVIDIA RTX A4000",
    ]
    assert "no NVIDIA RTX 2000 Ada Generation pod" in proc.stderr


def test_down_terminates_only_the_clusters_pods(bed):
    bed.pods(
        pod("distrainer-head", "headid"),
        pod("distrainer-worker-1", "w1", head="headid"),
        pod("distrainer-worker-2", "w2", head="headid"),
    )
    (bed.root / ".harness" / "runpod").mkdir(parents=True)
    (bed.root / ".harness" / "runpod" / "distrainer-head.ssh").write_text("1.2.3.4 10341\n")
    out = bed.run("down").stdout
    assert sorted(bed.terminated()) == ["headid", "w1", "w2"]  # never teammate1 or the look-alike
    assert "terminating distrainer-head (headid, RUNNING)" in out
    assert not (bed.root / ".harness" / "runpod" / "distrainer-head.ssh").exists()
    bed.run("nuke")  # the same under another name
    assert len(bed.terminated()) == 6


def test_scale_adds_or_removes_the_highest_workers(bed):
    bed.pods(
        pod("distrainer-head", "headid"),
        pod("distrainer-worker-1", "w1", head="headid"),
        pod("distrainer-worker-2", "w2", head="headid"),
    )
    bed.run("scale", "1")
    assert bed.terminated() == ["w2"]
    bed.respond("POST", "/pods", pod("distrainer-worker-3", "w3", head="headid"), n=1)
    bed.respond("POST", "/pods", pod("distrainer-worker-4", "w4", head="headid"), n=2)
    bed.run("scale", "4")
    assert [p["name"] for p in bed.posts()] == ["distrainer-worker-3", "distrainer-worker-4"]
    assert all(p["env"]["RAY_HEAD_ADDRESS"] == "headid.runpod.internal:6379" for p in bed.posts())


def test_kill_worker_terminates_and_recreates_the_same_name_kill_head_terminates_the_head(bed):
    bed.pods(
        pod("distrainer-head", "headid"),
        pod("distrainer-worker-1", "w1", head="headid"),
        pod("distrainer-worker-2", "w2", head="headid"),
    )
    out = bed.run("kill-worker", "2", env={"DISTRAINER_RESTART_DELAY": "0"}).stdout
    assert bed.terminated() == ["w2"] and "worker 2 killed" in out and bed.posts() == []
    bed.respond("POST", "/pods", pod("distrainer-worker-2", "w2b", head="headid"), n=1)
    out = bed.run("kill-worker", "2", env={"DISTRAINER_RESTART_DELAY": "1"}).stdout
    assert "created in 1s" in out
    deadline = time.monotonic() + 10
    while time.monotonic() < deadline and not bed.posts():
        time.sleep(0.5)
    (created,) = bed.posts()
    assert (
        created["name"] == "distrainer-worker-2"
        and created["env"]["RAY_HEAD_ADDRESS"] == "headid.runpod.internal:6379"
    )
    assert created["dataCenterIds"] == ["EU-RO-1"]
    (bed.root / ".harness" / "runpod" / "distrainer-head.ssh").write_text("1.2.3.4 10341\n")
    out = bed.run("kill-head").stdout
    headless = Bed(bed.root.parent / "headless")
    headless.pods(pod("distrainer-worker-1", "w1", head="gone"))
    proc = headless.run("kill-worker", "1", ok=False)
    assert proc.returncode == 1 and "no head pod" in proc.stderr and headless.terminated() == []
    assert bed.terminated()[-1] == "headid" and "new workers" in out
    assert not (bed.root / ".harness" / "runpod" / "distrainer-head.ssh").exists()
    bed.run("stop-worker", "1")
    assert ("POST", "/pods/w1/action", {"action": "stop"}) in bed.calls()


def test_up_after_a_head_death_replaces_workers_made_for_the_old_head(bed):
    bed.pods(
        pod("distrainer-worker-1", "w1", head="oldhead"),
        pod("distrainer-worker-2", "w2", head="oldhead"),
    )
    bed.respond("POST", "/pods", pod("distrainer-head", "newhead"), n=1)
    bed.respond("GET", "/pods/newhead", pod("distrainer-head", "newhead"))
    bed.respond("POST", "/pods", pod("distrainer-worker-1", "w1b", head="newhead"), n=2)
    bed.respond("POST", "/pods", pod("distrainer-worker-2", "w2b", head="newhead"), n=3)
    out = bed.run("up", "2").stdout
    assert sorted(bed.terminated()) == ["w1", "w2"] and "made for another head" in out
    assert [p["name"] for p in bed.posts()] == [
        "distrainer-head",
        "distrainer-worker-1",
        "distrainer-worker-2",
    ]
    assert all(
        p["env"]["RAY_HEAD_ADDRESS"] == "newhead.runpod.internal:6379" for p in bed.posts()[1:]
    )


def test_exec_head_is_a_login_shell_in_app_over_the_cached_ssh_port(bed):
    (bed.root / ".harness" / "runpod").mkdir(parents=True)
    (bed.root / ".harness" / "runpod" / "distrainer-head.ssh").write_text("1.2.3.4 10341\n")
    bed.run(
        "exec-head",
        "python",
        "-c",
        "import ray; print(ray.__version__)",
        env={"DISTRAINER_RUNPOD_SSH_KEY": "/k/id"},
    )
    (line,) = [ln for ln in bed.log.read_text().splitlines() if ln.startswith("ssh ")]
    assert "-T -p 10341" in line and "root@1.2.3.4" in line and "-i /k/id" in line
    assert not any(m == "GET" for m, _, _ in bed.calls())  # the cache spared the API
    # what the pod's login shell runs: argv with quotes, $ and spaces must arrive intact
    argv = ["python", "-c", 'print("$HOME", "a\\"b", \'x\')', "plain arg", ""]
    bed.run("exec-head", *argv)
    remote = (bed.log.parent / "calls.log.remote").read_text()
    assert remote.startswith("bash -lc ")
    app = bed.root / "app"
    app.mkdir()
    (app / "python").write_text('#!/bin/bash\nfor a in "$@"; do printf \'%s\\n\' "$a"; done\n')
    (app / "python").chmod(0o755)
    proc = subprocess.run(
        ["bash", "-c", remote.replace("cd\\ /app", "cd\\ " + str(app).replace("/", "\\/"))],
        env={"PATH": f"{app}:/usr/bin:/bin", "HOME": str(bed.root)}, capture_output=True, text=True,
    )  # fmt: skip
    assert proc.returncode == 0, proc.stderr
    assert proc.stdout.split("\n")[:-1] == argv[1:]
    bed.run("cp-from-head", "/tmp/a.json", "/tmp/b.json")
    (scp,) = [ln for ln in bed.log.read_text().splitlines() if ln.startswith("scp ")]
    assert "-P 10341" in scp and "root@1.2.3.4:/tmp/a.json /tmp/b.json" in scp
    # no cache: the head's ssh block from the API
    (bed.root / ".harness" / "runpod" / "distrainer-head.ssh").unlink()
    bed.pods(pod("distrainer-head", "headid"))
    bed.run("exec-head", "true")
    assert (
        bed.root / ".harness" / "runpod" / "distrainer-head.ssh"
    ).read_text() == "1.2.3.4 10341\n"


def test_ps_lists_the_cluster_with_its_hourly_total(bed):
    bed.pods(
        pod("distrainer-head", "headid"), pod("distrainer-worker-1", "w1", head="headid", cost=0.25)
    )
    out = bed.run("ps").stdout.splitlines()
    assert out[0].startswith("NAME\tID\tSTATUS\tGPU\tDC\tCOST\tINTERNAL")
    assert out[1].startswith(
        "distrainer-head\theadid\tRUNNING\tNVIDIA RTX 2000 Ada Generation\tEU-RO-1\t0.24 USD/h"
        "\theadid.runpod.internal"
    )
    assert out[-1] == "total: 0.49 USD/h"
    assert "teammate1" not in "\n".join(out) and "lookalike" not in "\n".join(out)


def test_build_needs_nothing_and_the_key_is_required_for_the_api(bed):
    out = bed.run(
        "build", env={"PATH": f"{bed.bin}:/usr/bin:/bin"}
    ).stdout  # no docker on that PATH: no manifest check
    assert "gpu-image workflow" in out and bed.calls() == []
    proc = bed.run("ps", env={"RUNPOD_KEY": ""}, ok=False)
    assert proc.returncode != 0 and "RUNPOD_KEY" in proc.stderr


def test_up_replaces_dead_pods_first_and_a_failed_listing_never_looks_empty(bed):
    # a stopped worker and an errored head are dead nodes: terminated, then made anew
    bed.pods(
        pod("distrainer-head", "oldhead", status="ERROR"),
        pod("distrainer-worker-1", "w1", head="oldhead", status="EXITED"),
    )
    bed.respond("POST", "/pods", pod("distrainer-head", "newhead"), n=1)
    bed.respond("GET", "/pods/newhead", pod("distrainer-head", "newhead"))
    bed.respond("POST", "/pods", pod("distrainer-worker-1", "w1b", head="newhead"), n=2)
    out = bed.run("up", "1").stdout
    assert sorted(bed.terminated()) == ["oldhead", "w1"] and "(oldhead, ERROR)" in out
    assert [p["name"] for p in bed.posts()] == ["distrainer-head", "distrainer-worker-1"]
    # the listing fails: down must not pretend the cluster is empty
    broken = Bed(bed.root.parent / "broken")
    broken.respond("GET", "/pods", {"title": "boom"}, code=500)
    proc = broken.run("down", ok=False)
    assert proc.returncode == 1 and "HTTP 500" in proc.stderr and broken.terminated() == []
    # a terminate that fails does not stop the others, and down reports it
    partial = Bed(bed.root.parent / "partial")
    partial.pods(pod("distrainer-head", "headid"), pod("distrainer-worker-1", "w1", head="headid"))
    partial.respond("POST", "/pods/headid/action", {"title": "busy"}, code=429)
    proc = partial.run("down", ok=False)
    assert proc.returncode == 1 and "headid still running" in proc.stderr
    assert sorted(partial.terminated()) == ["headid", "w1"]  # both attempted, one refused


def test_scale_fills_gaps_and_logs_reads_the_event_stream(bed):
    bed.pods(
        pod("distrainer-head", "headid"),
        pod("distrainer-worker-1", "w1", head="headid"),
        pod("distrainer-worker-3", "w3", head="headid"),
    )
    bed.respond("POST", "/pods", pod("distrainer-worker-2", "w2", head="headid"), n=1)
    bed.run("scale", "2")
    assert bed.terminated() == ["w3"] and [p["name"] for p in bed.posts()] == [
        "distrainer-worker-2"
    ]
    (bed.resp / "GET_pods_headid_logs.json").write_text(
        'data: {"line":"Ray runtime started."}\ndata: {"line":"ok"}\n'
    )
    assert bed.run("logs").stdout.splitlines() == ["Ray runtime started.", "ok"]
    assert bed.run("cost").stdout.startswith("total: 0.72 USD/h")
