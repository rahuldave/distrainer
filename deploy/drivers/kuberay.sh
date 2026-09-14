#!/usr/bin/env bash
# KubeRay driver: pods of a RayCluster are the Ray nodes (OrbStack's built-in Kubernetes on the
# Mac, or any cluster with the KubeRay operator; spec section 11). See deploy/driver.sh for the
# verbs. Compared with the compose driver:
#   - deploy/k8s/raycluster.yaml declares the head pod (no `trainer` resource) and one worker group
#     (`trainer: 1` per pod); `up N` renders it with N replicas, `scale N` patches replicas.
#   - deploy/k8s/minio.yaml (the `minio` argument of `up`, or DISTRAINER_MINIO=1) adds MinIO;
#     its PVC survives `down` and goes with `nuke`.
#   - shared storage is the same host directory the compose driver bind-mounts, reached from the
#     pods as a hostPath volume (OrbStack's Kubernetes runs in the VM that sees the Mac filesystem);
#     the source tree is hostPath-mounted over the image copy as well.
#   - node death is `kubectl delete pod --force`; the operator starts a replacement pod at once, so
#     a killed worker always comes back and DISTRAINER_RESTART_DELAY does not apply. `stop-worker`
#     and a `scale` down delete gracefully: KubeRay's preStop runs `ray stop` (a preemption notice
#     to the trainer) and the pod is killed after terminationGracePeriodSeconds (10 s, as
#     `docker stop`); the operator replaces a stopped worker as well.
#   - `operator` installs the KubeRay operator once (kustomize through kubectl; no Helm needed);
#     `submit [CONFIG]` applies deploy/k8s/rayjob.yaml.
# Worker index I (kill-worker, stop-worker) counts worker pods in creation order, 1-based.
set -euo pipefail
root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
if [ -f "$root/.env" ]; then set -a; . "$root/.env"; set +a; fi
ns="${DISTRAINER_K8S_NAMESPACE:-distrainer}"
cluster="distrainer"
image="${DISTRAINER_IMAGE:-distrainer:local}"
kuberay_version="${KUBERAY_VERSION:-v1.7.0}"
k=(kubectl --namespace "$ns")
shared="${DISTRAINER_SHARED:-$root/.harness/shared}"
case "$shared" in
  /*) ;;
  *) shared="$root/$shared" ;;
esac
if [ "$shared" = "/" ] || [ -z "$shared" ]; then
  echo "refusing to operate on shared storage path '$shared'" >&2; exit 2
fi
marker="$shared/.distrainer-shared"   # nuke / wipe-shared only delete a directory `up` created

render() {   # render <manifest> [replicas] [config]: substitute the host-specific placeholders;
  # the S3 settings follow .env with the defaults of docker-compose.yml
  sed -e "s|__SHARED__|$shared|g" -e "s|__ROOT__|$root|g" -e "s|__IMAGE__|$image|g" \
      -e "s|__REPLICAS__|${2:-2}|g" -e "s|__CONFIG__|${3:-}|g" \
      -e "s|__S3_ENDPOINT__|${S3_ENDPOINT:-http://minio:9000}|g" -e "s|__S3_REGION__|${S3_REGION:-auto}|g" \
      -e "s|__S3_ACCESS_KEY__|${S3_ACCESS_KEY:-distrainer}|g" \
      -e "s|__S3_SECRET_KEY__|${S3_SECRET_KEY:-distrainer123}|g" "$root/deploy/k8s/$1"
}
head_pod() {   # the newest head pod (after kill-head the operator creates a new one), "" if none;
  # a failing kubectl fails the caller (pipefail) instead of looking like "no pod yet"
  "${k[@]}" get pods -l "ray.io/cluster=$cluster,ray.io/node-type=head" \
    --sort-by=.metadata.creationTimestamp -o jsonpath='{range .items[*]}{.metadata.name}{"\n"}{end}' | tail -n 1
}
need_head() {   # pod="$(need_head)": the head pod name, or a clear failure
  local pod
  pod="$(head_pod)"
  if [ -z "$pod" ]; then echo "no head pod in namespace $ns (run 'up' first)" >&2; return 1; fi
  echo "$pod"
}
worker_pods() {   # worker pods in creation order, one per line
  "${k[@]}" get pods -l "ray.io/cluster=$cluster,ray.io/node-type=worker" \
    --sort-by=.metadata.creationTimestamp -o jsonpath='{range .items[*]}{.metadata.name}{"\n"}{end}'
}
worker_pod() {   # worker_pod I -> pod name (1-based index)
  local pod
  pod="$(worker_pods | sed -n "${1}p")"
  if [ -z "$pod" ]; then echo "no worker pod with index $1" >&2; worker_pods >&2; exit 1; fi
  echo "$pod"
}
pods_present() {   # 0 = pods exist, 1 = none; a failing kubectl aborts (never mistaken for "none")
  local out
  out="$("${k[@]}" get pods -o name)" || { echo "kubectl get pods failed in namespace $ns" >&2; exit 1; }
  [ -n "$out" ]
}
wait_head() {   # wait_head SECONDS: until the head pod is Ready
  local deadline=$((SECONDS + $1)) pod
  while :; do
    pod="$(head_pod)"
    if [ -n "$pod" ] && "${k[@]}" wait --for=condition=Ready "pod/$pod" --timeout=5s >/dev/null 2>&1; then
      return 0
    fi
    if [ "$SECONDS" -ge "$deadline" ]; then
      echo "head pod not ready after $1s" >&2; "${k[@]}" get pods -o wide >&2; exit 1
    fi
    sleep 2
  done
}
wait_gone() {   # wait_gone SECONDS: until no pod is left in the namespace
  local deadline=$((SECONDS + $1))
  while pods_present; do
    if [ "$SECONDS" -ge "$deadline" ]; then echo "pods still present after $1s" >&2; "${k[@]}" get pods >&2; exit 1; fi
    sleep 2
  done
}
exec_head() {
  local pod
  pod="$(need_head)"
  "${k[@]}" exec "$pod" -c ray-head -- "$@"
}
need_operator() {
  if ! kubectl get crd rayclusters.ray.io >/dev/null 2>&1; then
    echo "KubeRay CRDs not found: run 'deploy/driver.sh operator' (just kuberay-operator) once" >&2
    exit 2
  fi
}

verb="${1:-}"; shift || true
case "$verb" in
  build)
    docker build -t "$image" -f "$root/deploy/Dockerfile" "$root" ;;
  operator)
    # the kustomize base installs CRDs, RBAC and the operator Deployment into `default`;
    # server-side apply copes with the CRD size and makes the verb idempotent
    kubectl apply --server-side --force-conflicts \
      -k "github.com/ray-project/kuberay/ray-operator/config/default?ref=$kuberay_version&timeout=180s"
    kubectl --namespace default rollout status deployment/kuberay-operator --timeout=300s ;;
  up)
    n="${1:-2}"; shift || true
    minio="${DISTRAINER_MINIO:-0}"
    if [ "${1:-}" = "minio" ]; then minio=1; fi
    need_operator
    if ! docker image inspect "$image" >/dev/null 2>&1; then
      docker build -t "$image" -f "$root/deploy/Dockerfile" "$root"
    fi
    mkdir -p "$shared" && touch "$marker"
    kubectl create namespace "$ns" --dry-run=client -o yaml | kubectl apply -f - >/dev/null
    if [ "$minio" = "1" ]; then render minio.yaml | "${k[@]}" apply -f -; fi
    render raycluster.yaml "$n" | "${k[@]}" apply -f -
    if [ "$minio" = "1" ]; then "${k[@]}" rollout status deployment/minio --timeout=180s; fi
    wait_head 300
    "${k[@]}" get pods -o wide ;;
  down)
    # the RayCluster and MinIO go; the MinIO PVC stays (S9 restores from it); use nuke to drop it
    "${k[@]}" delete raycluster "$cluster" --ignore-not-found --wait=true
    "${k[@]}" delete deployment minio --ignore-not-found --wait=true
    "${k[@]}" delete service minio distrainer-workers --ignore-not-found
    "${k[@]}" delete rayjob --all --ignore-not-found >/dev/null 2>&1 || true
    wait_gone 180 ;;
  nuke)
    "$0" down
    "${k[@]}" delete pvc minio-data --ignore-not-found
    if [ -d "$shared" ] && [ ! -f "$marker" ]; then
      echo "nuke: $shared was not created by this driver (no marker); not deleting it" >&2; exit 2
    fi
    rm -rf "$shared" ;;
  wipe-shared)
    if pods_present; then
      echo "wipe-shared: pods are running in namespace $ns; run 'down' first" >&2; exit 2
    fi
    if [ -d "$shared" ] && [ ! -f "$marker" ]; then
      echo "wipe-shared: $shared was not created by this driver (no marker); not deleting it" >&2; exit 2
    fi
    rm -rf "$shared" && mkdir -p "$shared" && touch "$marker" ;;
  scale)
    n="${1:?worker count}"
    # a JSON patch: a merge patch would replace the whole workerGroupSpecs list
    "${k[@]}" patch raycluster "$cluster" --type json \
      -p "[{\"op\": \"replace\", \"path\": \"/spec/workerGroupSpecs/0/replicas\", \"value\": $n}]" ;;
  exec-head)
    exec_head "$@" ;;
  kill-worker)
    # immediate deletion = node death; the operator starts a replacement pod as a new Ray node
    i="${1:?worker index (1-based)}"
    pod="$(worker_pod "$i")"
    "${k[@]}" delete pod "$pod" --grace-period=0 --force
    echo "worker $i ($pod) killed; KubeRay is starting a replacement pod as a new Ray node" ;;
  kill-head)
    # the head pod (Ray head, Train controller and driver) dies; the operator recreates it (S10)
    pod="$(need_head)"
    "${k[@]}" delete pod "$pod" --grace-period=0 --force ;;
  stop-worker)
    i="${1:?worker index (1-based)}"
    "${k[@]}" delete pod "$(worker_pod "$i")" ;;   # SIGTERM, then a replacement pod
  cp-from-head)
    pod="$(need_head)"
    "${k[@]}" cp -c ray-head "$pod:${1:?src}" "${2:?dst}" ;;
  submit)
    # a RayJob for hello_blocks on the running cluster; follow it with `logs job/distrainer-hello`
    cfg="${1:-examples/hello_blocks/harness.yaml}"
    "${k[@]}" delete rayjob distrainer-hello --ignore-not-found >/dev/null
    render rayjob.yaml 2 "$cfg" | "${k[@]}" apply -f - ;;
  shared)
    echo "$shared" ;;
  endpoint)
    # the head service is headless, so the dashboard is at the head pod's IP; pod and service IPs
    # are routable from the Mac under OrbStack (elsewhere: kubectl port-forward svc/distrainer-head-svc 8265)
    hip="$("${k[@]}" get raycluster "$cluster" -o jsonpath='{.status.head.podIP}' 2>/dev/null || true)"
    mip="$("${k[@]}" get svc minio -o jsonpath='{.spec.clusterIP}' 2>/dev/null || true)"
    echo "dashboard=http://${hip:-<no head pod>}:8265"
    echo "minio=http://${mip:-<no minio service>}:9000 console=http://${mip:-<no minio service>}:9001" ;;
  mkbucket)
    # MinIO may still be starting; retry for a while
    bucket="${1:-distrainer}"
    for attempt in $(seq 1 15); do
      if exec_head python -c "
import os, pyarrow.fs as pafs
from urllib.parse import urlparse
u = urlparse(os.environ['S3_ENDPOINT'])
fs = pafs.S3FileSystem(access_key=os.environ['S3_ACCESS_KEY'], secret_key=os.environ['S3_SECRET_KEY'],
                       endpoint_override=u.netloc, scheme=u.scheme, region=os.environ.get('S3_REGION', 'auto'),
                       allow_bucket_creation=True)
fs.create_dir('$bucket')
print('bucket ready:', '$bucket')" 2>/dev/null; then break; fi
      if [ "$attempt" = "15" ]; then echo "mkbucket: MinIO not reachable" >&2; exit 1; fi
      sleep 2
    done ;;
  ps)
    "${k[@]}" get raycluster,pods,svc,pvc -o wide 2>/dev/null || "${k[@]}" get pods -o wide ;;
  logs)
    if [ "$#" -eq 0 ]; then pod="$(need_head)"; set -- "$pod"; fi
    "${k[@]}" logs --tail=100 "$@" ;;
  *)
    echo "unknown verb '$verb'; see deploy/driver.sh" >&2; exit 2 ;;
esac
