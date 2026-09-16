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
    assert case_labels(AWS_SH) == base | {"bucket", "bucket-rm", "ecr", "ecr-rm", "env"}
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
    for other in (ROOT / "deploy" / "drivers").glob("*.sh"):  # the caller wins over .env everywhere
        text = other.read_text()
        assert 'set -a; . "$root/.env"; set +a; eval "$caller_env"' in text, other.name
        if other.name != "uncloud.sh":  # the env file belongs to an uncloud bed
            assert "DISTRAINER_ENV_FILE" not in text.split("caller_env=")[1], other.name
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
    aws_file = "DISTRAINER_UNCLOUD_PROVIDER=aws\n" + env_file
    out = run_uncloud_driver(tmp_path, "endpoint", {}, dotenv, aws_file)
    assert "s3=https://s3.us-east-1.amazonaws.com" in out
    out = run_uncloud_driver(  # another bed pinned by the caller: the AWS file is not read
        tmp_path, "endpoint", {"DISTRAINER_UNCLOUD_PROVIDER": "orbstack"}, dotenv, aws_file
    )
    assert "minio=http://1.2.3.4:9000" in out
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


def test_aws_bootstrap_admits_ssh_only_and_deletes_only_what_it_tagged():
    """The gpa findings of the cloud stage: no dashboard rule (an ssh tunnel instead), stale rules
    revoked on up and start, the allowed CIDR a narrow IPv4 prefix, the bucket tagged at creation
    or adopted explicitly and bucket-rm refusing anything else, credential files at mode 600."""
    text = AWS_SH.read_text()
    rules = [ln for ln in text.splitlines() if ln.strip().startswith('allow "$1"')]
    assert rules == [
        '  allow "$1" tcp 22 22 "$2" "from this Mac"',
        '  allow "$1" udp "$wg_port" "$wg_port" "$1" "WireGuard between the members"',
    ]
    assert "$2 != c || $3 != 22" in text  # another address or another port: revoked
    assert text.count('ensure_rules "$sg"') == 2  # up and start
    assert "2[4-9]|3[0-2])" in text and "*[!0-9.]*" in text  # /24 at the widest, dotted quads only
    assert text.count("put-bucket-tagging") == 2  # the adopt branch and the create branch
    ecr_rm = text.split("  ecr-rm)")[1].split("  env)")[0]  # the repository: the same rule
    assert 'owner="$(repo_owner "$arn")"' in ecr_rm and "refusing to delete it" in ecr_rm
    assert "need aws" in ecr_rm
    ecr = text.split("  ecr)")[1].split("  ecr-rm)")[0]
    assert (
        "tag-resource" in ecr and "Key=distrainer:cluster,Value=$ctx" in ecr
    )  # tagged when made or adopted
    exists_branch = text.split('echo "bucket $bucket exists')[0].rsplit(
        "if awsc s3api head-bucket", 1
    )[1]
    assert "DISTRAINER_AWS_ADOPT_BUCKET" in text and 'owner="$(bucket_owner' in exists_branch
    assert 'owner="$(bucket_owner "$bucket")"' in text.split("bucket-rm)")[1]  # the guard
    assert text.count("chmod 600") >= 3
    assert "S3_SECRET_KEY=\\).*/\\1<in the file>" in text  # `env` masks the secret


def test_aws_bootstrap_defaults_to_amd64_and_records_the_registry():
    text = AWS_SH.read_text()
    assert "DISTRAINER_AWS_ARCH:-amd64" in text and "DISTRAINER_AWS_HEAD_TYPE:-t3.large" in text
    assert "DISTRAINER_AWS_WORKER_TYPE:-t3.medium" in text
    ecr = text.split("  ecr)")[1].split("  ecr-rm)")[0]
    assert (
        "create-repository" in ecr
        and "put-lifecycle-policy" in ecr
        and 'echo "$uri" > "$state/ecr"' in ecr
    )
    assert 'DISTRAINER_IMAGE=$(cat "$state/ecr"):latest' in text  # what the driver reads
    assert "DISTRAINER_PLATFORMS=" in text
    assert "DISTRAINER_PLATFORMS" in (ROOT / ".env.example").read_text()


def stub_bin(tmp_path, log):
    """docker, aws, uc and ssh as stubs that log their arguments (ssh also what it read)."""
    b = tmp_path / "bin"
    b.mkdir(exist_ok=True)
    for name, body in {
        "docker": (
            'echo "docker $*" >> "$LOG"\n'
            'if [ "$1 $2" = "buildx inspect" ]; then exit 1; fi\nexit 0\n'
        ),
        "aws": (
            'echo "aws $*" >> "$LOG"\n'
            'case "$*" in *get-login-password*) echo TOKEN ;; esac\nexit 0\n'
        ),
        "uc": 'echo "uc $*" >> "$LOG"\nexit 0\n',
        "ssh": 'in=""; [ -t 0 ] || in="$(cat)"\necho "ssh $* <<< $in" >> "$LOG"\nexit 0\n',
    }.items():
        (b / name).write_text(f"#!/bin/bash\nLOG={log}\n{body}")
        (b / name).chmod(0o755)
    return b


def run_build(tmp_path, image: str, extra_env: dict | None = None) -> list[str]:
    import subprocess

    log = tmp_path / "calls.log"
    log.write_text("")
    b = stub_bin(tmp_path, log)
    root = tmp_path / "repo"
    (root / "deploy" / "drivers").mkdir(parents=True, exist_ok=True)
    (root / "deploy" / "uncloud").mkdir(exist_ok=True)
    (root / "deploy" / "Dockerfile").write_text("FROM scratch\n")
    import shutil

    shutil.copy(UNCLOUD_DRIVER, root / "deploy" / "drivers" / "uncloud.sh")
    env = {
        "PATH": f"{b}:/usr/bin:/bin",
        "HOME": str(tmp_path),
        "DISTRAINER_IMAGE": image,
        "DISTRAINER_UNCLOUD_HOST_PREFIX": "10.0.0.0/8",
        "DISTRAINER_UNCLOUD_MACHINES": "m1 m2",
        "DISTRAINER_UNCLOUD_SSH": "ubuntu@%s",
        **(extra_env or {}),
    }
    proc = subprocess.run(
        ["bash", str(root / "deploy" / "drivers" / "uncloud.sh"), "build"],
        env=env, capture_output=True, text=True,
    )  # fmt: skip
    assert proc.returncode == 0, proc.stderr
    return log.read_text().splitlines()


def test_uncloud_build_pushes_a_local_image_to_the_machines(tmp_path):
    calls = run_build(tmp_path, "distrainer:local")
    assert calls[0].startswith("docker build -t distrainer:local -f ")
    assert calls[1] == "uc machine ls"  # need_cluster
    assert calls[2] == "uc image push distrainer:local"
    assert len(calls) == 3


def test_uncloud_build_pushes_a_registry_image_once_and_every_machine_pulls(tmp_path):
    """An ECR image: the builder is created when missing, the Mac logs in with a token from the
    CLI, one multi-platform buildx push, then every machine logs in with the same token over the
    ssh route and pulls."""
    image = "123456789012.dkr.ecr.us-east-1.amazonaws.com/distrainer:latest"
    calls = run_build(tmp_path, image, {"DISTRAINER_PLATFORMS": "linux/amd64"})
    registry = image.split("/")[0]
    assert calls[0] == "docker buildx inspect distrainer"
    assert (
        calls[1] == "docker buildx create --name distrainer --driver docker-container --bootstrap"
    )
    assert calls[2] == "aws --region us-east-1 ecr get-login-password"
    assert calls[3] == f"docker login --username AWS --password-stdin {registry}"
    assert calls[4].startswith(
        f"docker buildx build --builder distrainer --platform linux/amd64 -t {image} --push -f "
    )
    assert calls[5] == "uc machine ls"
    pulls = sorted(c for c in calls[6:] if c.startswith("ssh "))
    assert len(pulls) == 4, calls
    for m in ("m1", "m2"):
        assert any(
            f"ubuntu@{m} sudo -n docker login --username AWS --password-stdin {registry} <<< TOKEN"
            in c
            for c in pulls
        ), m
        assert any(f"ubuntu@{m} sudo -n docker pull {image} <<< " in c for c in pulls), m


def test_uncloud_build_tells_a_registry_image_by_dockers_rule(tmp_path):
    """A slash alone does not make a registry image: the first component needs a dot, a colon
    or `localhost`. A non-ECR registry gets no token and no logins, only the push and the pulls."""
    calls = run_build(tmp_path, "myorg/distrainer:local")
    assert calls[0].startswith("docker build -t myorg/distrainer:local") and calls[-1].startswith(
        "uc image push"
    )
    calls = run_build(tmp_path, "ghcr.io/rahuldave/distrainer:latest")
    assert not any("get-login-password" in c or "docker login" in c for c in calls)
    assert any(
        c.startswith("docker buildx build --builder distrainer --platform linux/amd64,linux/arm64")
        for c in calls
    )
    assert (
        sum(1 for c in calls if "sudo -n docker pull ghcr.io/rahuldave/distrainer:latest" in c) == 2
    )
    calls = run_build(
        tmp_path, "localhost:5000/distrainer:dev", {"DISTRAINER_PLATFORMS": "linux/arm64"}
    )
    assert any("--platform linux/arm64 -t localhost:5000/distrainer:dev --push" in c for c in calls)
    calls = run_build(
        tmp_path, "distrainer:local", {"DISTRAINER_PLATFORMS": "linux/amd64"}
    )  # recipe C
    assert calls[0].startswith("docker build --platform linux/amd64 -t distrainer:local")


# ---- the GPU image: deploy/Dockerfile.gpu, deploy/runpod-entry.sh, workflows/gpu-image.yml ----

GPU_DOCKERFILE = ROOT / "deploy" / "Dockerfile.gpu"
CPU_DOCKERFILE = ROOT / "deploy" / "Dockerfile"
RUNPOD_ENTRY = ROOT / "deploy" / "runpod-entry.sh"
GPU_WORKFLOW = ROOT / ".github" / "workflows" / "gpu-image.yml"
GPU_IMAGE = "ghcr.io/rahuldave/distrainer-gpu"


def locked_version(package: str) -> str:
    """The version uv.lock pins for ``package`` (the CPU wheel's, without its local tag)."""
    text = (ROOT / "uv.lock").read_text()
    versions = {
        m.group(1).split("+")[0]
        for m in re.finditer(rf'^name = "{package}"\nversion = "([^"]+)"', text, re.MULTILINE)
    }
    assert len(versions) == 1, (package, versions)
    return versions.pop()


def dockerfile_args(path: Path) -> dict[str, str]:
    return dict(re.findall(r"^ARG ([A-Z_]+)=(\S+)", path.read_text(), re.MULTILINE))


def dockerfile_env(path: Path) -> dict[str, str]:
    """Every ``KEY=value`` of the ``ENV`` instructions (continuation lines included)."""
    text = path.read_text().replace("\\\n", " ")
    pairs: dict[str, str] = {}
    for line in re.findall(r"^ENV (.+)$", text, re.MULTILINE):
        pairs.update(re.findall(r"([A-Z_]+)=(\S+)", line))
    return pairs


def test_gpu_dockerfile_pins_the_locked_torch_versions_as_cuda_wheels():
    """The image swaps the CPU wheels for the CUDA ones of the same versions: the ARGs must
    follow uv.lock, and the install must ask for the local ``+cuXXX`` versions explicitly (a bare
    ``==2.14.0`` is already satisfied by the CPU wheel and would install nothing)."""
    args = dockerfile_args(GPU_DOCKERFILE)
    assert args["TORCH_VERSION"] == locked_version("torch")
    assert args["TORCHVISION_VERSION"] == locked_version("torchvision")
    text = GPU_DOCKERFILE.read_text()
    assert re.search(r"^ARG CUDA_TAG=cu\d+$", text, re.MULTILINE)
    assert 'download.pytorch.org/whl/${CUDA_TAG}"' in text
    assert '"torch==${TORCH_VERSION}+${CUDA_TAG}"' in text
    assert '"torchvision==${TORCHVISION_VERSION}+${CUDA_TAG}"' in text
    # the CUDA layer comes after the locked sync and before the project copy, and the project is
    # installed without a second sync (which would restore the CPU wheels)
    assert "--no-install-package torch --no-install-package torchvision" in text
    sync, cuda, copy, project = (
        text.index("uv sync --frozen --no-dev --no-install-project"),
        text.index("uv pip install --python /app/.venv \\\n    --index"),
        text.index("COPY . ."),
        text.index("uv pip install --python /app/.venv --no-deps -e ."),
    )
    assert sync < cuda < copy < project
    assert len(re.findall(r"^RUN .*uv sync ", text, re.MULTILINE)) == 1  # the comment aside


def test_gpu_dockerfile_matches_the_cpu_image_where_they_share():
    cpu, gpu = CPU_DOCKERFILE.read_text(), GPU_DOCKERFILE.read_text()
    (base_cpu,) = re.findall(r"^FROM (\S+)$", cpu, re.MULTILINE)
    (base_gpu,) = re.findall(r"^FROM (\S+)$", gpu, re.MULTILINE)
    assert base_cpu == base_gpu  # the same Python; the CUDA runtime rides in the wheels
    (uv_cpu,) = re.findall(r"^COPY --from=(ghcr.io/astral-sh/uv:\S+)", cpu, re.MULTILINE)
    (uv_gpu,) = re.findall(r"^COPY --from=(ghcr.io/astral-sh/uv:\S+)", gpu, re.MULTILINE)
    assert uv_cpu == uv_gpu
    env_cpu, env_gpu = dockerfile_env(CPU_DOCKERFILE), dockerfile_env(GPU_DOCKERFILE)
    assert env_cpu.items() <= env_gpu.items(), env_cpu.items() - env_gpu.items()
    assert "openssh-server" in gpu and "ssh-keygen -A" in gpu and "procps" in gpu
    assert 'ENTRYPOINT ["deploy/runpod-entry.sh"]' in gpu and 'CMD ["head"]' in gpu
    assert "chmod +x deploy/ray-head.sh deploy/ray-worker.sh deploy/runpod-entry.sh" in gpu
    assert "/etc/profile.d/distrainer-venv.sh" in gpu


def test_runpod_entrypoint_starts_sshd_from_the_injected_key_and_runs_the_role():
    text = RUNPOD_ENTRY.read_text()
    assert text.startswith("#!/usr/bin/env bash")
    assert 'if [ -n "${PUBLIC_KEY:-}" ]' in text
    assert ">> /root/.ssh/authorized_keys" in text and "/usr/sbin/sshd" in text
    # the key stays out of the exported environment, and so do the shell's own variables (a
    # login shell starts in /root: an exported PWD=/app would lie to every script trusting it)
    assert "grep -Ev '^(PUBLIC_KEY|PATH|PWD|OLDPWD|HOME|SHLVL|HOSTNAME|_|TERM)='" in text
    assert "> /etc/rp_environment" in text and "/etc/profile.d/distrainer-env.sh" in text
    assert re.search(r"^  head\) exec deploy/ray-head.sh ;;$", text, re.MULTILINE)
    assert (
        'NCCL_SOCKET_IFNAME="${NCCL_SOCKET_IFNAME:-$iface}"' in text
    )  # collectives on the 10.x interface
    assert re.search(r"^  worker\) exec deploy/ray-worker.sh ;;$", text, re.MULTILINE)
    assert re.search(r'^  \*\) exec "\$@" ;;$', text, re.MULTILINE)
    lint = (ROOT / "Justfile").read_text()
    assert "deploy/runpod-entry.sh; do bash -n" in lint  # `just lint` parses it
    ci = (ROOT / ".github" / "workflows" / "ci.yml").read_text()
    assert "deploy/runpod-entry.sh; do bash -n" in ci


def test_gpu_image_workflow_builds_the_gpu_dockerfile_for_amd64_into_ghcr():
    wf = yaml.safe_load(GPU_WORKFLOW.read_text())
    assert wf["permissions"] == {"contents": "read", "packages": "write"}
    on = wf[True] if True in wf else wf["on"]  # YAML parses a bare `on` as True
    assert on["push"]["branches"] == ["main", "gest/**"]
    assert "workflow_dispatch" in on
    for path in ("deploy/Dockerfile.gpu", "deploy/runpod-entry.sh", "uv.lock", "distrainer/**"):
        assert path in on["push"]["paths"], path
    (job,) = wf["jobs"].values()
    steps = {s.get("uses", "").split("@")[0]: s for s in job["steps"]}
    assert steps["docker/login-action"]["with"]["registry"] == "ghcr.io"
    assert steps["docker/login-action"]["with"]["password"] == "${{ secrets.GITHUB_TOKEN }}"
    assert steps["docker/metadata-action"]["with"]["images"] == GPU_IMAGE
    tags = steps["docker/metadata-action"]["with"]["tags"]
    assert "type=ref,event=branch" in tags and "{{is_default_branch}}" in tags
    build = steps["docker/build-push-action"]["with"]
    assert build["file"] == "deploy/Dockerfile.gpu" and build["platforms"] == "linux/amd64"
    assert build["push"] is True and build["context"] == "."
    (smoke,) = [s for s in job["steps"] if s.get("name") == "Smoke the head role"]
    assert "docker exec head ray status" in smoke["run"] and '"$IMAGE" head' in smoke["run"]
    assert "cd /app &&" in smoke["run"]  # a login shell starts in /root
    assert GPU_IMAGE in GPU_DOCKERFILE.read_text()  # the header names where it is pushed
