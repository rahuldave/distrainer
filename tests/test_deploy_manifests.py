"""The KubeRay manifests under deploy/k8s keep the harness contract of spec sections 9 and 11:
the head pod offers no `trainer` resource, every worker pod offers one, the placeholders the
driver renders are exactly the ones the manifests use, and MinIO is reachable where the harness
configs expect it. Static checks only; nothing here talks to a cluster."""

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
