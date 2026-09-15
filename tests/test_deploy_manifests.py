"""The KubeRay manifests under deploy/k8s and the uncloud compose file keep the harness contract
of spec sections 9 and 11: the head offers no `trainer` resource, every worker offers one, the
placeholders the KubeRay driver renders are exactly the ones the manifests use, MinIO is reachable
where the harness configs expect it, and the three drivers describe the same nodes and implement
the same verbs. Static checks only; nothing here talks to a cluster."""

from __future__ import annotations

import json
import re
from pathlib import Path

import yaml

from distrainer.config import load_config

ROOT = Path(__file__).resolve().parents[1]
K8S = ROOT / "deploy" / "k8s"
DRIVER = ROOT / "deploy" / "drivers" / "kuberay.sh"
PLACEHOLDER = re.compile(r"__[A-Z]+__")


S3_DEFAULTS = {  # what the driver substitutes without a .env (docker-compose.yml's defaults)
    "S3_ENDPOINT": "http://minio:9000",
    "S3_REGION": "auto",
    "S3_ACCESS_KEY": "distrainer",
    "S3_SECRET_KEY": "distrainer123",
}


def rendered(name: str, **values: str) -> list[dict]:
    text = (K8S / name).read_text()
    for key, value in {**S3_DEFAULTS, **values}.items():
        text = text.replace(f"__{key}__", value)
    assert not PLACEHOLDER.search(text), f"{name}: unrendered placeholder"
    return [doc for doc in yaml.safe_load_all(text) if doc]


def test_driver_renders_every_placeholder_the_manifests_use():
    used = {m for f in K8S.glob("*.yaml") for m in PLACEHOLDER.findall(f.read_text())}
    substituted = set(re.findall(r"s\|(__[A-Z]+__)\|", DRIVER.read_text()))  # the sed -e pairs
    assert used <= substituted, used - substituted


def raycluster(**values: str) -> dict:
    defaults = {"SHARED": "/tmp/shared", "ROOT": "/repo", "IMAGE": "img:t", "REPLICAS": "2"}
    docs = {d["kind"]: d for d in rendered("raycluster.yaml", **{**defaults, **values})}
    return docs["RayCluster"]


def test_raycluster_head_has_no_trainer_and_each_worker_offers_one():
    cluster = raycluster(REPLICAS="3")
    assert cluster["kind"] == "RayCluster"
    head = cluster["spec"]["headGroupSpec"]
    assert "resources" not in head["rayStartParams"]
    (group,) = cluster["spec"]["workerGroupSpecs"]
    assert group["replicas"] == 3
    assert group["minReplicas"] <= 2 <= 3 <= group["maxReplicas"]  # the scenarios' 2 <-> 3
    # KubeRay passes the value through bash: a JSON object inside a double-quoted string
    resources = json.loads(json.loads(group["rayStartParams"]["resources"]))
    assert resources == {"trainer": 1}
    assert group["rayStartParams"]["num-cpus"] == "1"


def test_raycluster_mounts_shared_storage_and_the_source_tree_in_every_pod():
    cluster = raycluster(SHARED="/host/shared")
    templates = [cluster["spec"]["headGroupSpec"]["template"]] + [
        g["template"] for g in cluster["spec"]["workerGroupSpecs"]
    ]
    for template in templates:
        # scale-down and stop-worker end in a node death after a docker-stop-like notice period
        assert template["spec"]["terminationGracePeriodSeconds"] <= 15
        # a lookup CoreDNS forwards outside the cluster must not cost a 15 s resolver timeout
        options = {o["name"]: o["value"] for o in template["spec"]["dnsConfig"]["options"]}
        assert all(isinstance(v, str) for v in options.values())  # the API wants strings
        assert int(options["timeout"]) <= 2 and int(options["attempts"]) == 1
        (container,) = template["spec"]["containers"]
        assert container["image"] == "img:t"
        assert container["imagePullPolicy"] == "IfNotPresent"  # a locally built image
        mounts = {m["name"]: m["mountPath"] for m in container["volumeMounts"]}
        volumes = {v["name"]: v["hostPath"]["path"] for v in template["spec"]["volumes"]}
        assert mounts["shared"] == "/shared" and volumes["shared"] == "/host/shared"
        assert volumes["src-distrainer"] == "/repo/distrainer"
        assert mounts["src-distrainer"] == "/app/distrainer"
        env = {e["name"]: e["value"] for e in container["env"]}
        assert env["RAY_TRAIN_V2_ENABLED"] == "1"
        assert env["S3_ENDPOINT"] == "http://minio:9000"


def test_minio_service_matches_the_harness_config_endpoint():
    docs = {d["kind"]: d for d in rendered("minio.yaml")}
    assert set(docs) == {"PersistentVolumeClaim", "Deployment", "Service"}
    (port,) = [p for p in docs["Service"]["spec"]["ports"] if p["name"] == "s3"]
    cfg = load_config(str(ROOT / "examples" / "hello_blocks" / "harness-minio.yaml"))
    assert cfg.storage.endpoint == f"http://{docs['Service']['metadata']['name']}:{port['port']}"
    (container,) = docs["Deployment"]["spec"]["template"]["spec"]["containers"]
    env = {e["name"]: e["value"] for e in container["env"]}
    assert env["MINIO_ROOT_USER"] == "distrainer"  # the default when no .env overrides it
    (head,) = raycluster()["spec"]["headGroupSpec"]["template"]["spec"]["containers"]
    head_env = {e["name"]: e["value"] for e in head["env"]}
    assert (env["MINIO_ROOT_USER"], env["MINIO_ROOT_PASSWORD"]) == (
        head_env["S3_ACCESS_KEY"],
        head_env["S3_SECRET_KEY"],
    )


def test_rayjob_targets_the_raycluster():
    (job,) = rendered("rayjob.yaml", IMAGE="img:t", CONFIG="examples/hello_blocks/harness.yaml")
    assert job["spec"]["clusterSelector"] == {"ray.io/cluster": raycluster()["metadata"]["name"]}
    assert job["spec"]["entrypoint"].endswith("--config examples/hello_blocks/harness.yaml")


def test_worker_pods_get_reverse_dns_records_from_a_headless_service():
    """torch.distributed reverse-resolves every peer; CoreDNS answers PTR queries locally only
    for pods behind a headless Service and forwards the rest outside the cluster."""
    docs = {
        d["kind"]: d
        for d in rendered("raycluster.yaml", SHARED="/s", ROOT="/r", IMAGE="i", REPLICAS="2")
    }
    svc = docs["Service"]["spec"]
    assert svc["clusterIP"] == "None" and svc["publishNotReadyAddresses"] is True  # headless
    cluster = docs["RayCluster"]
    assert svc["selector"] == {
        "ray.io/cluster": cluster["metadata"]["name"],
        "ray.io/node-type": "worker",
    }


def test_compose_and_kuberay_describe_the_same_cluster():
    """The two drivers must not drift apart: the S3 defaults, the image tag and the shape of a
    Ray node (CPUs, object store) are declared twice, once per driver."""
    compose = (ROOT / "deploy" / "docker-compose.yml").read_text()
    defaults: dict[str, str] = {}
    for key, value in re.findall(r"\$\{(S3_[A-Z_]+):-([^}]*)\}", compose):
        assert defaults.setdefault(key, value) == value, f"{key} has two defaults in compose"
    assert {k: defaults[k] for k in S3_DEFAULTS} == S3_DEFAULTS
    (image,) = set(re.findall(r"\$\{DISTRAINER_IMAGE:-([^}]*)\}", compose))
    assert f"DISTRAINER_IMAGE:-{image}" in DRIVER.read_text()
    head_sh = (ROOT / "deploy" / "ray-head.sh").read_text()
    worker_sh = (ROOT / "deploy" / "ray-worker.sh").read_text()
    cluster = raycluster()
    head = cluster["spec"]["headGroupSpec"]["rayStartParams"]
    (group,) = cluster["spec"]["workerGroupSpecs"]
    assert head["num-cpus"] == re.search(r"HEAD_CPUS:-(\d+)", head_sh).group(1)
    assert group["rayStartParams"]["num-cpus"] == re.search(r"--num-cpus=(\d+)", worker_sh).group(1)
    store = re.search(r"OBJECT_STORE_BYTES:-(\d+)", head_sh).group(1)
    assert head["object-store-memory"] == store
    assert group["rayStartParams"]["object-store-memory"] == store


# ---- uncloud (deploy/uncloud/compose.yml, deploy/drivers/uncloud.sh) ----

UNCLOUD = ROOT / "deploy" / "uncloud" / "compose.yml"
UNCLOUD_DRIVER = ROOT / "deploy" / "drivers" / "uncloud.sh"
COMPOSE = ROOT / "deploy" / "docker-compose.yml"
COMMON_VERBS = [  # the contract of deploy/driver.sh that every driver implements
    "build", "up", "down", "nuke", "wipe-shared", "scale", "exec-head", "kill-worker",
    "kill-head", "stop-worker", "cp-from-head", "shared", "endpoint", "mkbucket", "ps", "logs",
]  # fmt: skip
MACHINES_SH = ROOT / "deploy" / "uncloud" / "machines.sh"
AWS_SH = ROOT / "deploy" / "uncloud" / "aws.sh"
EXAMPLES = ROOT / "examples" / "hello_blocks"
INTERPOLATION = re.compile(
    r"\$\{([A-Z0-9_]+)(?::([-?])([^}]*))?\}"
)  # ${VAR}, ${VAR:-dflt}, ${VAR:?msg}


def compose_defaults(path: Path) -> dict:
    """The compose file with every ``${VAR:-default}`` replaced by its default (no .env; a
    required ``${VAR:?message}`` becomes empty)."""
    text = INTERPOLATION.sub(lambda m: m.group(3) if m.group(2) == "-" else "", path.read_text())
    return yaml.safe_load(text)


def test_uncloud_compose_describes_the_same_nodes_as_compose():
    """Image, node scripts, Ray environment, shared memory, MinIO image and credentials must not
    drift between the compose harness and the uncloud one (kuberay is pinned to compose above)."""
    uc, dc = compose_defaults(UNCLOUD)["services"], compose_defaults(COMPOSE)["services"]
    for name in ("head", "worker"):
        assert uc[name]["image"] == dc[name]["image"]
        assert uc[name]["command"] == dc[name]["command"]
        assert uc[name]["shm_size"] == dc[name]["shm_size"]
        assert uc[name]["environment"] == dc[name]["environment"]
    assert uc["head"]["healthcheck"] == dc["head"]["healthcheck"]
    assert uc["minio"]["image"] == dc["minio"]["image"]
    assert uc["minio"]["command"] == dc["minio"]["command"]
    assert uc["minio"]["environment"] == dc["minio"]["environment"]
    assert uc["minio"]["healthcheck"] == dc["minio"]["healthcheck"]
    assert uc["head"]["environment"]["S3_ENDPOINT"] == S3_DEFAULTS["S3_ENDPOINT"]


def test_uncloud_compose_carries_everything_in_the_image_and_pulls_nothing():
    """Nothing spans machines: no build, no bind mounts (the image carries the code), the locally
    pushed image is used as is, and MinIO's data is a named volume on its machine."""
    services = compose_defaults(UNCLOUD)["services"]
    for name in ("head", "worker"):
        assert "build" not in services[name] and "volumes" not in services[name]
        assert services[name]["pull_policy"] == "never"
    (volume,) = services["minio"]["volumes"]
    assert not volume.startswith(("/", ".")) and volume.endswith(":/data")


def test_uncloud_compose_pins_head_and_minio_together_and_workers_elsewhere():
    raw = yaml.safe_load(UNCLOUD.read_text())["services"]
    assert raw["head"]["x-machines"] == raw["minio"]["x-machines"]
    (head_var,) = INTERPOLATION.findall(raw["head"]["x-machines"][0])
    (worker_var,) = INTERPOLATION.findall(raw["worker"]["x-machines"])  # a comma-separated string
    assert head_var[0] == "DISTRAINER_UNCLOUD_HEAD_MACHINE"
    assert worker_var[0] == "DISTRAINER_UNCLOUD_WORKER_MACHINES"  # never the head machine's
    replicas = str(raw["worker"]["deploy"]["replicas"])
    assert INTERPOLATION.fullmatch(replicas) and "DISTRAINER_WORKERS" in replicas  # `up N`
    assert "head" in raw["worker"]["depends_on"]


def test_uncloud_compose_publishes_ports_only_inside_the_host_prefix():
    """Published ports bind to the machine's address on the machine network only (OrbStack
    forwards machine ports to the LAN otherwise); dashboard and MinIO sit on the head machine."""
    raw = yaml.safe_load(UNCLOUD.read_text())["services"]
    ports = {name: raw[name].get("x-ports", []) for name in raw}
    assert ports["worker"] == []
    for entry in ports["head"] + ports["minio"]:  # the prefix has no default: the driver sets it
        assert entry.startswith("${DISTRAINER_UNCLOUD_HOST_PREFIX:?") and entry.endswith("@host")
    assert any(":8265:8265/" in p for p in ports["head"])
    assert any(":9000:9000/" in p for p in ports["minio"])


def case_labels(driver: Path) -> set[str]:
    """The verbs of the driver's dispatch ``case "$verb" in`` block."""
    dispatch = driver.read_text().split('case "$verb" in', 1)[1]
    return set(re.findall(r"^  ([a-z-]+)\)", dispatch, re.MULTILINE))


def test_every_driver_implements_every_common_verb():
    header = (ROOT / "deploy" / "driver.sh").read_text()
    for verb in COMMON_VERBS:  # documented in the header, alone or as `ps | logs`
        assert re.search(rf"^#   (?:[a-z-]+ \| )*{re.escape(verb)}\b", header, re.MULTILINE), verb
    for driver in (ROOT / "deploy" / "drivers").glob("*.sh"):
        missing = set(COMMON_VERBS) - case_labels(driver)
        assert not missing, f"{driver.name} lacks {sorted(missing)}"


def test_uncloud_driver_names_its_context_and_shares_nothing():
    text = UNCLOUD_DRIVER.read_text()
    assert 'export UNCLOUD_CONTEXT="$ctx"' in text  # never whatever context uc points at
    assert "UNCLOUD_AUTO_CONFIRM=true" in text  # non-interactive deploys
    assert "DISTRAINER_IMAGE:-distrainer:local" in text  # the same default as compose
    assert re.search(r"^  shared\)\n\s+;;", text, re.MULTILINE)  # prints nothing


def test_uncloud_driver_exports_every_variable_the_compose_file_interpolates():
    """What compose.yml reads from the environment the driver sets (the S3 settings come from
    .env with the same defaults as docker-compose.yml)."""
    names = {m.group(1) for m in INTERPOLATION.finditer(UNCLOUD.read_text())}
    assert names >= {"DISTRAINER_IMAGE", "DISTRAINER_WORKERS", "DISTRAINER_UNCLOUD_HOST_PREFIX"}
    driver = UNCLOUD_DRIVER.read_text()
    for name in sorted(names - set(S3_DEFAULTS)):
        assert re.search(rf"(^|\s|export ){name}=", driver, re.MULTILINE), name


def test_uncloud_driver_and_bootstrap_agree_on_their_defaults():
    driver, machines = UNCLOUD_DRIVER.read_text(), MACHINES_SH.read_text()
    for setting in (
        "DISTRAINER_UNCLOUD_CONTEXT:-distrainer",
        "DISTRAINER_UNCLOUD_MACHINES:-uc1 uc2 uc3",
    ):
        assert setting in driver and setting in machines, setting


def test_uncloud_bootstraps_share_the_verbs_the_driver_dispatches():
    """machines.sh (OrbStack) and aws.sh (EC2) answer the same verbs, and the driver's machines-*
    verbs forward exactly those; aws.sh adds the bucket verbs."""
    base = {"up", "status", "stop", "start", "destroy"}
    assert case_labels(MACHINES_SH) == base
    assert case_labels(AWS_SH) == base | {"bucket", "bucket-rm", "env"}
    driver = UNCLOUD_DRIVER.read_text()
    (label,) = [
        lb for lb in re.findall(r"^  ([a-z|-]+)\)", driver, re.MULTILINE) if "machines-" in lb
    ]
    assert set(label.split("|")) == {f"machines-{v}" for v in base}
    assert "DISTRAINER_UNCLOUD_PROVIDER:-orbstack" in driver
    for script in (MACHINES_SH, AWS_SH):  # the shared helpers come from one place
        assert "common.sh" in script.read_text()
    header = (ROOT / "deploy" / "driver.sh").read_text()
    assert "machines-stop | machines-start" in header


def test_uncloud_driver_endpoint_names_the_store_and_reads_the_bootstrap_env():
    """`endpoint` prints minio= for MinIO in the cluster and s3= for a store outside it (what the
    runner's store_endpoint parses); the bootstrap's env file and ssh options are honoured."""
    driver = UNCLOUD_DRIVER.read_text()
    assert "minio=http://$ip:9000" in driver and "s3=$S3_ENDPOINT" in driver
    assert (
        "http://minio:9000)" in driver
    )  # the compose default means MinIO, anything else is outside
    assert "DISTRAINER_ENV_FILE" in driver and "DISTRAINER_UNCLOUD_SSH_OPTS" in driver
    runner = (ROOT / "integration_tests" / "cluster" / "run_scenarios.py").read_text()
    assert 'for kind in ("minio", "s3")' in runner and "DISTRAINER_ENV_FILE" in runner
    env_example = (ROOT / ".env.example").read_text()
    for name in (
        "DISTRAINER_ENV_FILE",
        "DISTRAINER_UNCLOUD_SSH_OPTS",
        "DISTRAINER_UNCLOUD_PROVIDER",
    ):
        assert name in env_example, name


def test_aws_bootstrap_writes_what_the_driver_and_the_runner_read():
    text = AWS_SH.read_text()
    for line in (
        "DISTRAINER_UNCLOUD_PROVIDER=aws",
        "DISTRAINER_UNCLOUD_CONTEXT=$ctx",
        "DISTRAINER_UNCLOUD_SSH=%s",
        'DISTRAINER_UNCLOUD_SSH_OPTS=\\"-F $state/ssh_config\\"',
        "DISTRAINER_UNCLOUD_HOST_PREFIX=$cidr",
        "DISTRAINER_UNCLOUD_HEAD_ADDRESS=$head_pub",
        "S3_ENDPOINT=https://s3.$region.amazonaws.com",
        "S3_ACCESS_KEY=$ak",
    ):
        assert line in text, line
    assert "--public-ip none" in text and '--wg-endpoint "$priv:$wg_port"' in text  # private peers
    assert (
        'allow "$1" udp "$wg_port" "$wg_port" "$1"' in text
    )  # WireGuard from the group itself only
    assert "HttpTokens=required" in text
    assert (
        "DISTRAINER_UNCLOUD_CONTEXT:-distrainer-aws" in text
    )  # never the OrbStack context by accident


def strip_storage(cfg: dict) -> dict:
    """A harness config without what names its bucket: storage.endpoint/region, and the bucket in
    storage_path / store_root (the prefix after the bucket stays)."""
    out = yaml.safe_load(yaml.safe_dump(cfg))
    for key in ("storage_path", "store_root"):
        out[key] = out[key].split("/", 1)[1]
    out["storage"] = {k: v for k, v in out["storage"].items() if k not in ("endpoint", "region")}
    return out


def test_s3_harness_configs_are_their_minio_twins_on_another_bucket():
    """harness-s3.yaml and harness-stream-s3.yaml differ from the MinIO configs only in the
    bucket, the endpoint and the region, and the two S3 configs agree on all three."""
    pairs = [
        ("harness-minio.yaml", "harness-s3.yaml"),
        ("harness-stream-minio.yaml", "harness-stream-s3.yaml"),
    ]
    buckets, endpoints = set(), set()
    for minio, s3 in pairs:
        m, x = (yaml.safe_load((EXAMPLES / f).read_text()) for f in (minio, s3))
        assert strip_storage(m) == strip_storage(x), (minio, s3)
        assert m["storage"]["endpoint"] == "http://minio:9000" and x["storage"][
            "endpoint"
        ].startswith("https://")
        assert x["storage"]["region"] not in (None, "auto")  # a real region for SigV4
        buckets.add(x["store_root"].split("/")[0])
        buckets.add(x["storage_path"].split("/")[0])
        endpoints.add((x["storage"]["endpoint"], x["storage"]["region"]))
    assert len(buckets) == 1 and len(endpoints) == 1
    aws = AWS_SH.read_text()  # the bootstrap's bucket default is read from harness-s3.yaml
    assert "harness-s3.yaml" in aws and "store_root:" in aws


def run_uncloud_driver(tmp_path, verb: str, env: dict, dotenv: str = "", env_file: str = "") -> str:
    """The real uncloud driver in a scratch tree (its .env and .harness/aws/env), for verbs that
    need no cluster: `endpoint` with DISTRAINER_UNCLOUD_HEAD_ADDRESS set calls neither uc nor
    orb."""
    import os
    import shutil
    import subprocess

    root = tmp_path / "repo"
    (root / "deploy" / "drivers").mkdir(parents=True, exist_ok=True)
    (root / "deploy" / "uncloud").mkdir(exist_ok=True)
    (root / ".harness" / "aws").mkdir(parents=True, exist_ok=True)
    shutil.copy(UNCLOUD_DRIVER, root / "deploy" / "drivers" / "uncloud.sh")
    shutil.copy(MACHINES_SH, root / "deploy" / "uncloud" / "machines.sh")
    for path, text in ((root / ".env", dotenv), (root / ".harness" / "aws" / "env", env_file)):
        if text:
            path.write_text(text)
        else:
            path.unlink(missing_ok=True)
    base = {"PATH": os.environ["PATH"], "HOME": os.environ.get("HOME", "/tmp")}
    base.update(
        {
            "DISTRAINER_UNCLOUD_HEAD_ADDRESS": "1.2.3.4",
            "DISTRAINER_UNCLOUD_HOST_PREFIX": "10.0.0.0/8",
        }
    )
    proc = subprocess.run(
        ["bash", str(root / "deploy" / "drivers" / "uncloud.sh"), verb],
        env={**base, **env}, capture_output=True, text=True,
    )  # fmt: skip
    assert proc.returncode == 0, proc.stderr
    return proc.stdout


def test_uncloud_driver_endpoint_prints_minio_or_the_external_store(tmp_path):
    for value in ("", "http://minio:9000", "http://minio:9000/", "http://minio.internal:9000"):
        env = {"S3_ENDPOINT": value} if value else {}
        out = run_uncloud_driver(tmp_path, "endpoint", env)
        assert (
            out
            == "dashboard=http://1.2.3.4:8265\nminio=http://1.2.3.4:9000 console=http://1.2.3.4:9001\n"
        ), value
    out = run_uncloud_driver(
        tmp_path, "endpoint", {"S3_ENDPOINT": "https://s3.us-east-1.amazonaws.com"}
    )
    assert out == "dashboard=http://1.2.3.4:8265\ns3=https://s3.us-east-1.amazonaws.com\n"


def test_uncloud_driver_env_precedence_is_caller_then_env_file_then_dotenv(tmp_path):
    """.env names the bootstrap's env file, which wins over .env; what the caller's environment
    sets wins over both (so an OrbStack command stays OrbStack while .env points at AWS)."""
    dotenv = "S3_ENDPOINT=http://minio:9000\nDISTRAINER_ENV_FILE=.harness/aws/env\n"
    env_file = (
        "S3_ENDPOINT=https://s3.us-east-1.amazonaws.com\nDISTRAINER_UNCLOUD_HEAD_ADDRESS=5.6.7.8\n"
    )
    out = run_uncloud_driver(tmp_path, "endpoint", {}, dotenv, env_file)
    assert out.splitlines() == [
        "dashboard=http://1.2.3.4:8265",
        "s3=https://s3.us-east-1.amazonaws.com",
    ]
    out = run_uncloud_driver(
        tmp_path, "endpoint", {"S3_ENDPOINT": "http://minio:9000"}, dotenv, env_file
    )
    assert "minio=http://1.2.3.4:9000" in out  # the caller said MinIO
    import subprocess

    proc = subprocess.run(  # a pointer to nothing is an error, not a fallback
        ["bash", str(tmp_path / "repo" / "deploy" / "drivers" / "uncloud.sh"), "endpoint"],
        env={
            "PATH": "/usr/bin:/bin",
            "DISTRAINER_ENV_FILE": "nowhere",
            "DISTRAINER_UNCLOUD_HOST_PREFIX": "10.0.0.0/8",
        },
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 2 and "DISTRAINER_ENV_FILE=nowhere" in proc.stderr
