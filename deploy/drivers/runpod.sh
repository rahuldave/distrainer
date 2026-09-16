#!/usr/bin/env bash
# RunPod driver (DISTRAINER_DRIVER=runpod): every Ray node is a GPU pod on RunPod's secure cloud,
# joined by global networking (<pod id>.runpod.internal, TCP), created from the GPU image
# (deploy/Dockerfile.gpu, ghcr.io) through the REST v2 API (https://api.runpod.io/v2) with curl and
# jq. See deploy/driver.sh for the verbs, docs/tutorials/runpod.md for the operations guide.
#   - nothing spans pods: the store is the S3 bucket the S3_* variables name (passed to every pod),
#     `shared` prints nothing and the runner reads the bucket
#   - the driver acts only on pods it made: named <cluster>-head / <cluster>-worker-N and carrying
#     DISTRAINER_CLUSTER=<cluster> in their environment (the account holds other people's pods)
#   - v2 takes one GPU type per create, so DISTRAINER_RUNPOD_GPU_TYPES is tried in order; a type
#     whose price is above DISTRAINER_RUNPOD_MAX_GPU_HOURLY is skipped (the spend guard); every
#     create prints the pod's hourly cost and `ps` the cluster's total
#   - a worker dials RAY_HEAD_ADDRESS=<head pod id>.runpod.internal:6379; a new head has a new id,
#     so `up` after `kill-head` recreates the workers as well
#   - exec-head is ssh to the head's published 22/tcp (a login shell: `bash -lc "cd /app && ..."`)
#   - pods are terminated, never stopped (a stopped pod's disk bills by the month); `stop-worker`
#     is the exception, modelling a preemption notice
set -euo pipefail
root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
caller_env="$(export -p)"
if [ -f "$root/.env" ]; then set -a; . "$root/.env"; set +a; eval "$caller_env"; fi

api="${DISTRAINER_RUNPOD_API:-https://api.runpod.io/v2}"
cluster="${DISTRAINER_RUNPOD_CLUSTER:-distrainer}"
image="${DISTRAINER_IMAGE:-ghcr.io/rahuldave/distrainer-gpu:latest}"
gpu_types="${DISTRAINER_RUNPOD_GPU_TYPES:-NVIDIA RTX 2000 Ada Generation,NVIDIA RTX A4000,NVIDIA RTX A4500,NVIDIA RTX A5000,NVIDIA RTX 4000 Ada Generation,NVIDIA L4,NVIDIA A40}"
data_centers="${DISTRAINER_RUNPOD_DATA_CENTERS:-}"
cloud="${DISTRAINER_RUNPOD_CLOUD:-SECURE}"
max_hourly="${DISTRAINER_RUNPOD_MAX_GPU_HOURLY:-0.60}"
disk_gb="${DISTRAINER_RUNPOD_DISK_GB:-20}"
state="$root/.harness/runpod"
mkdir -p "$state"

need_key() {
  if [ -z "${RUNPOD_KEY:-}" ]; then echo "runpod: RUNPOD_KEY is not set (.env)" >&2; exit 2; fi
}

# api METHOD PATH [JSON]: the response body on stdout; a non-2xx status is an error with the body
api() {
  local method="$1" path="$2" body="${3:-}" out code
  need_key
  local args=(-sS -X "$method" -H "Authorization: Bearer $RUNPOD_KEY" -H "Content-Type: application/json" -w '\n%{http_code}')
  if [ -n "$body" ]; then args+=(-d "$body"); fi
  out="$(curl "${args[@]}" "$api$path")" || { echo "runpod: curl $method $path failed" >&2; return 1; }
  code="${out##*$'\n'}"; out="${out%$'\n'*}"
  case "$code" in
    2*) printf '%s\n' "$out" ;;
    *) echo "runpod: $method $path -> HTTP $code: $(printf '%s' "$out" | head -c 400)" >&2; return 1 ;;
  esac
}

# the cluster's pods as JSON lines {id, name, status, ...}: the name prefix and the env marker;
# a listing failure is an error (never "no pods": `down` would then leave everything billing)
cluster_pods() {
  local out
  out="$(api GET /pods)" || return 1
  printf '%s' "$out" | jq -c --arg c "$cluster" \
    '.pods[] | select((.name | startswith($c + "-")) and (.env.DISTRAINER_CLUSTER == $c) and (.status != "TERMINATED"))'
}
live_pods() {  # the ones that are, or are becoming, Ray nodes; EXITED and ERROR pods are dead nodes
  local out
  out="$(cluster_pods)" || return 1
  printf '%s\n' "$out" | jq -c 'select(.status == "RUNNING" or .status == "STARTING" or .status == "PROVISIONING")'
}
pod_named() { local out; out="$(live_pods)" || return 1; printf '%s\n' "$out" | jq -c --arg n "$1" 'select(.name == $n)' | head -n 1; }
head_pod() { pod_named "$cluster-head"; }
worker_pods() {  # sorted by index
  local out
  out="$(live_pods)" || return 1
  printf '%s\n' "$out" | jq -c --arg c "$cluster" 'select(.name | startswith($c + "-worker-"))' \
    | jq -sc --arg c "$cluster" 'sort_by(.name | ltrimstr($c + "-worker-") | tonumber) | .[]'
}

catalog_price() {  # CATALOG_JSON TYPE: the per-GPU hourly price of a type sold on the chosen cloud, else empty
  local field
  case "$cloud" in COMMUNITY) field=community ;; *) field=secure ;; esac
  printf '%s' "$1" | jq -r --arg t "$2" --arg f "$field" \
    '.gpus[] | select(.id == $t and (.[$f] == true)) | .price[$f] // empty'
}
gn_data_centers() {  # comma-separated: the configured ones, else every data center with global networking
  if [ -n "$data_centers" ]; then printf '%s\n' "$data_centers"; return; fi
  api GET /catalog/datacenters | jq -r '[.dataCenters[] | select(.globalNetwork == true) | .id] | join(",")'
}

# the public key the pods get: DISTRAINER_RUNPOD_SSH_KEY's .pub sibling when set (RunPod then skips
# the account's registered keys, which this Mac need not hold); else the account's keys (startSsh)
ssh_pubkey() {
  local key="${DISTRAINER_RUNPOD_SSH_KEY:-}"
  [ -n "$key" ] || return 0
  key="${key/#\~/$HOME}"
  if [ ! -f "$key.pub" ]; then echo "runpod: DISTRAINER_RUNPOD_SSH_KEY=$key has no $key.pub next to it" >&2; return 1; fi
  cat "$key.pub"
}

# pod_env ROLE [HEAD_ID]: the environment of a pod as a JSON object
pod_env() {
  local role="$1" head_id="${2:-}" pub
  pub="$(ssh_pubkey)" || return 1
  jq -nc --arg c "$cluster" --arg role "$role" --arg head "$head_id" --arg pub "$pub" \
    --arg ep "${S3_ENDPOINT:-}" --arg ak "${S3_ACCESS_KEY:-}" --arg sk "${S3_SECRET_KEY:-}" --arg rg "${S3_REGION:-auto}" \
    '{DISTRAINER_CLUSTER: $c, DISTRAINER_ROLE: $role,
      S3_ENDPOINT: $ep, S3_ACCESS_KEY: $ak, S3_SECRET_KEY: $sk, S3_REGION: $rg,
      RAY_health_check_period_ms: "1000", RAY_health_check_timeout_ms: "1000", RAY_health_check_failure_threshold: "3"}
     + (if $role == "worker" then {RAY_HEAD_ADDRESS: ($head + ".runpod.internal:6379")} else {} end)
     + (if $pub != "" then {PUBLIC_KEY: $pub} else {} end)'
}

# create_pod NAME ROLE DATACENTERS [HEAD_ID]: tries the GPU types in order under the spend guard;
# prints the created pod's JSON
create_pod() {
  local name="$1" role="$2" dcs="$3" head_id="${4:-}" t price body out catalog env err
  catalog="$(api GET /catalog/gpus)" || return 1   # once per pod, before anything bills
  env="$(pod_env "$role" "$head_id")" || return 1
  err="$(mktemp "${TMPDIR:-/tmp}/runpod-create.XXXXXX")"
  local IFS=','
  for t in $gpu_types; do
    unset IFS
    t="${t# }"
    price="$(catalog_price "$catalog" "$t")"
    if [ -z "$price" ]; then
      echo "runpod: $t has no $cloud price in the catalog (unknown type, or not on this cloud); skipped" >&2
      continue
    fi
    if awk -v p="$price" -v m="$max_hourly" 'BEGIN { exit !(p > m) }'; then
      echo "runpod: $t is $price USD/h on the $cloud cloud, above DISTRAINER_RUNPOD_MAX_GPU_HOURLY=$max_hourly; skipped" >&2
      continue
    fi
    body="$(jq -nc --arg name "$name" --arg image "$image" --arg role "$role" --arg cloud "$cloud" \
      --arg t "$t" --arg dcs "$dcs" --argjson disk "$disk_gb" --argjson env "$env" \
      '{name: $name, image: $image, args: $role, ports: ["22/tcp"], env: $env, disk: $disk, cloud: $cloud,
        dataCenterIds: ($dcs | split(",") | map(select(length > 0))), globalNetworking: true, startSsh: true,
        gpu: {id: $t, count: 1, minRamPerGpu: 8, minVcpuCountPerGpu: 2}}')"
    if out="$(api POST /pods "$body" 2>"$err")" && [ "$(printf '%s' "$out" | jq -r '.id // empty')" != "" ]; then
      echo "created $name: $(printf '%s' "$out" | jq -r '"\(.id) on \(.gpu.id // "?") in \(.dataCenterId // "?") at \(.cost // "?") USD/h"')" >&2
      printf '%s\n' "$out"
      rm -f "$err"
      return 0
    fi
    echo "runpod: no $t pod: $(head -c 200 "$err")" >&2
  done
  rm -f "$err"
  echo "runpod: no GPU type of DISTRAINER_RUNPOD_GPU_TYPES could be rented for $name" >&2
  return 1
}

# wait_running ID [need_ssh]: until the pod is RUNNING with its global-networking address (and, for
# the head, its published ssh port); DISTRAINER_RUNPOD_START_TIMEOUT seconds at most
wait_running() {
  local id="$1" need_ssh="${2:-0}" budget="${DISTRAINER_RUNPOD_START_TIMEOUT:-1500}" waited=0 pod st port
  while :; do
    pod="$(api GET "/pods/$id")" || return 1
    st="$(printf '%s' "$pod" | jq -r '.status')"
    port="$(printf '%s' "$pod" | jq -r '.ssh.direct.port // empty')"
    if [ "$st" = "RUNNING" ] && [ "$(printf '%s' "$pod" | jq -r '.globalNetworking.ip // empty')" != "" ] \
       && { [ "$need_ssh" = "0" ] || [ -n "$port" ]; }; then
      printf '%s\n' "$pod"; return 0
    fi
    case "$st" in ERROR|TERMINATED|EXITED) echo "runpod: pod $id is $st" >&2; return 1 ;; esac
    if [ "$waited" -ge "$budget" ]; then echo "runpod: pod $id not running after ${budget}s ($st)" >&2; return 1; fi
    sleep 10; waited=$((waited + 10))
  done
}

terminate() { api POST "/pods/$1/action" '{"action":"terminate"}' >/dev/null; }

# terminate_all "JSON LINES": every pod given, going on after a failure; non-zero if any failed
terminate_all() {
  local failed=0 p
  while read -r p; do
    [ -n "$p" ] || continue
    echo "terminating $(printf '%s' "$p" | jq -r '.name') ($(printf '%s' "$p" | jq -r '.id'), $(printf '%s' "$p" | jq -r '.status'))"
    terminate "$(printf '%s' "$p" | jq -r '.id')" || { echo "runpod: $(printf '%s' "$p" | jq -r '.id') still running; terminate it by hand" >&2; failed=1; }
  done <<< "$1"
  return $failed
}

ssh_target() {  # host port of the head's direct ssh, cached by `up` (the cache goes with the head)
  local pod host port
  if [ -f "$state/$cluster-head.ssh" ]; then cat "$state/$cluster-head.ssh"; return; fi
  pod="$(head_pod)"
  if [ -z "$pod" ]; then echo "runpod: no head pod for cluster '$cluster' (run 'up')" >&2; return 1; fi
  host="$(printf '%s' "$pod" | jq -r '.ssh.direct.host // empty')"
  port="$(printf '%s' "$pod" | jq -r '.ssh.direct.port // empty')"
  if [ -z "$host" ] || [ -z "$port" ]; then echo "runpod: the head has no direct ssh port yet" >&2; return 1; fi
  printf '%s %s\n' "$host" "$port" | tee "$state/$cluster-head.ssh"
}
SSH_ARGS=()
ssh_args() {  # fills SSH_ARGS for the head (no tty; the account's registered key); bash 3.2 has no mapfile
  local host port target
  target="$(ssh_target)" || return 1
  read -r host port <<< "$target"
  SSH_ARGS=(-T -p "$port" -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -o LogLevel=ERROR)
  if [ -n "${DISTRAINER_RUNPOD_SSH_KEY:-}" ]; then SSH_ARGS+=(-i "${DISTRAINER_RUNPOD_SSH_KEY/#\~/$HOME}"); fi
  # shellcheck disable=SC2206
  if [ -n "${DISTRAINER_RUNPOD_SSH_OPTS:-}" ]; then SSH_ARGS+=(${DISTRAINER_RUNPOD_SSH_OPTS}); fi
  SSH_ARGS+=("root@$host")
}
exec_head() {
  ssh_args || exit 1
  # two shell layers: the pod's login shell unwraps one %q quoting, then bash -lc runs the argv
  local q remote; q="$(printf '%q ' "$@")"; remote="cd /app && $q"
  ssh "${SSH_ARGS[@]}" "bash -lc $(printf '%q' "$remote")"
}

up_workers() {  # a worker i for every i in 1..N that has none dialling this head: in the head's data
  local n="$1" head_id="$2" dc="$3" i w   # center, or in any global-networking one when that is sold out
  i=1
  while [ "$i" -le "$n" ]; do
    w="$(pod_named "$cluster-worker-$i")" || exit 1
    if [ -z "$w" ] || [ "$(printf '%s' "$w" | jq -r '.env.RAY_HEAD_ADDRESS // empty')" != "$head_id.runpod.internal:6379" ]; then
      if ! create_pod "$cluster-worker-$i" worker "$dc" "$head_id" >/dev/null; then
        echo "runpod: nothing left in $dc for worker $i; trying every global-networking data center (a slower link to the head)" >&2
        create_pod "$cluster-worker-$i" worker "$(gn_data_centers)" "$head_id" >/dev/null || exit 1
      fi
    fi
    i=$((i + 1))
  done
}

verb="${1:-}"; shift || true
case "$verb" in
  build)
    # the image is built by .github/workflows/gpu-image.yml on GitHub's runners (about 3 GB of CUDA wheels)
    echo "runpod: nothing to build here; the pods pull $image (built and pushed by the gpu-image workflow)"
    if command -v docker >/dev/null 2>&1; then
      docker manifest inspect "$image" >/dev/null 2>&1 && echo "the tag exists and is readable without credentials" \
        || echo "warning: $image is not readable anonymously (not pushed yet, or the package is private)" >&2
    fi ;;
  up)
    n="${1:-2}"; shift || true
    if [ "${1:-}" = "minio" ]; then echo "runpod: no MinIO under runpod; the store is the bucket S3_ENDPOINT names" >&2; exit 2; fi
    if [ -z "${S3_ENDPOINT:-}" ] || [ -z "${S3_ACCESS_KEY:-}" ] || [ -z "${S3_SECRET_KEY:-}" ]; then
      echo "runpod: S3_ENDPOINT, S3_ACCESS_KEY and S3_SECRET_KEY must name the bucket (.env)" >&2; exit 2
    fi
    # dead nodes first: a stopped or errored pod of the cluster is replaced, never revived
    all="$(cluster_pods)" || exit 1
    dead="$(printf '%s\n' "$all" | jq -c 'select(.status == "EXITED" or .status == "ERROR")')"
    if [ -n "$dead" ]; then terminate_all "$dead" || exit 1; fi
    head="$(head_pod)" || exit 1
    if [ -z "$head" ]; then
      head="$(create_pod "$cluster-head" head "$(gn_data_centers)")" || exit 1
    else
      echo "head exists: $(printf '%s' "$head" | jq -r '"\(.id) (\(.status))"')"
    fi
    head_id="$(printf '%s' "$head" | jq -r '.id')"
    dc="$(printf '%s' "$head" | jq -r '.dataCenterId // empty')"
    # workers made for another head (a recreated one has a new id and name) cannot rejoin: replace them
    workers="$(worker_pods)" || exit 1
    stale="$(printf '%s\n' "$workers" | jq -c --arg h "$head_id.runpod.internal:6379" 'select((.env.RAY_HEAD_ADDRESS // "") != $h)')"
    if [ -n "$stale" ]; then echo "workers made for another head:"; terminate_all "$stale" || exit 1; fi
    # the workers are created before the head is up so the image pulls run side by side (a
    # worker retries until the head answers); the head's id and data center are known at creation
    up_workers "$n" "$head_id" "$dc"
    head="$(wait_running "$head_id" 1)" || exit 1
    printf '%s %s\n' "$(printf '%s' "$head" | jq -r '.ssh.direct.host')" "$(printf '%s' "$head" | jq -r '.ssh.direct.port')" > "$state/$cluster-head.ssh"
    workers="$(worker_pods)" || exit 1
    while read -r w; do [ -n "$w" ] && { wait_running "$(printf '%s' "$w" | jq -r '.id')" >/dev/null || exit 1; }; done <<< "$workers"
    "$0" ps ;;
  nuke)
    exec "$0" down ;;   # no volumes, no shared mount: nuke is down
  down)
    # terminate everything of the cluster; a listing failure or a failed terminate is an error
    all="$(cluster_pods)" || exit 1
    rm -f "$state/$cluster-head.ssh"
    if [ -z "$all" ]; then echo "no pods of cluster '$cluster'"; exit 0; fi
    terminate_all "$all" || exit 1 ;;
  wipe-shared)
    echo "runpod: nothing is shared between pods; the store is the bucket (delete its prefixes yourself)" ;;
  scale)
    # the index set becomes 1..N: workers above N go, missing ones are made (gaps included)
    n="${1:?worker count}"
    head="$(head_pod)" || exit 1; [ -n "$head" ] || { echo "runpod: no head pod; run 'up' first" >&2; exit 1; }
    head_id="$(printf '%s' "$head" | jq -r '.id')"; dc="$(printf '%s' "$head" | jq -r '.dataCenterId')"
    workers="$(worker_pods)" || exit 1
    extra="$(printf '%s\n' "$workers" | jq -c --arg c "$cluster" --argjson n "$n" 'select((.name | ltrimstr($c + "-worker-") | tonumber) > $n)')"
    if [ -n "$extra" ]; then terminate_all "$extra" || exit 1; fi
    up_workers "$n" "$head_id" "$dc" ;;
  exec-head)
    exec_head "$@" ;;
  kill-worker)
    # terminate = node death; a new pod with the same name after DISTRAINER_RESTART_DELAY s (0 = stays dead)
    i="${1:?worker index (1-based)}"
    w="$(pod_named "$cluster-worker-$i")" || exit 1; [ -n "$w" ] || { echo "runpod: no worker $i" >&2; exit 1; }
    head="$(head_pod)" || exit 1; [ -n "$head" ] || { echo "runpod: no head pod; the worker would dial nothing" >&2; exit 1; }
    head_id="$(printf '%s' "$head" | jq -r '.id')"; dc="$(printf '%s' "$head" | jq -r '.dataCenterId')"
    terminate "$(printf '%s' "$w" | jq -r '.id')"
    delay="${DISTRAINER_RESTART_DELAY:-5}"
    if [ "$delay" != "0" ]; then
      (sleep "$delay" && create_pod "$cluster-worker-$i" worker "$dc" "$head_id") >"$state/$cluster-worker-$i.restart.log" 2>&1 &
      disown
      echo "worker $i killed; a new pod of that name is created in ${delay}s (a new Ray node)"
    else
      echo "worker $i killed"
    fi ;;
  kill-head)
    head="$(head_pod)" || exit 1; [ -n "$head" ] || { echo "runpod: no head pod" >&2; exit 1; }
    terminate "$(printf '%s' "$head" | jq -r '.id')"
    rm -f "$state/$cluster-head.ssh"
    echo "head killed; 'up' makes a new head and new workers (the workers dialled the old head's name)" ;;
  stop-worker)
    i="${1:?worker index (1-based)}"
    w="$(pod_named "$cluster-worker-$i")" || exit 1; [ -n "$w" ] || { echo "runpod: no worker $i" >&2; exit 1; }
    api POST "/pods/$(printf '%s' "$w" | jq -r '.id')/action" '{"action":"stop"}' >/dev/null
    echo "worker $i stopped (a stopped pod keeps its disk; the next 'up' or 'down' terminates it)" ;;
  cp-from-head)
    read -r host port <<< "$(ssh_target)"
    opts=(-P "$port" -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -o LogLevel=ERROR)
    if [ -n "${DISTRAINER_RUNPOD_SSH_KEY:-}" ]; then opts+=(-i "${DISTRAINER_RUNPOD_SSH_KEY/#\~/$HOME}"); fi
    scp -r "${opts[@]}" "root@$host:${1:?src}" "${2:?dst}" ;;
  shared)
    ;;
  endpoint)
    if [ -f "$state/$cluster-head.ssh" ]; then
      read -r host port < "$state/$cluster-head.ssh"
      echo "dashboard=http://127.0.0.1:8265 (tunnel: ssh -p $port -L 8265:127.0.0.1:8265 root@$host)"
    else
      echo "dashboard=(no head; run 'up')"
    fi
    if [ -z "${S3_ENDPOINT:-}" ]; then echo "runpod: S3_ENDPOINT is not set; the store must be a bucket" >&2; exit 2; fi
    echo "s3=$S3_ENDPOINT" ;;
  mkbucket)
    echo "runpod: no MinIO under runpod; the bucket S3_ENDPOINT names exists outside the cluster" ;;
  ps)
    all="$(cluster_pods)" || exit 1
    printf '%s\n' "$all" | jq -r '"\(.name)\t\(.id)\t\(.status)\t\(.gpu.id // "-")\t\(.dataCenterId // "-")\t\(.cost // 0) USD/h\t\(.globalNetworking.internalDns // "-")"' \
      | sort | awk 'BEGIN { print "NAME\tID\tSTATUS\tGPU\tDC\tCOST\tINTERNAL" } { print }'
    printf '%s\n' "$all" | jq -sr 'map(.cost // 0) | add // 0 | "total: \(.) USD/h"' ;;
  logs)
    # the API streams the log as server-sent events: read for a few seconds, print the lines
    name="${1:-$cluster-head}"
    case "$name" in "$cluster"-*) ;; *) name="$cluster-$name" ;; esac
    p="$(pod_named "$name")" || exit 1; [ -n "$p" ] || { echo "runpod: no pod $name" >&2; exit 1; }
    need_key
    curl -N -sS --max-time "${DISTRAINER_RUNPOD_LOG_SECONDS:-5}" -H "Authorization: Bearer $RUNPOD_KEY" \
      "$api/pods/$(printf '%s' "$p" | jq -r '.id')/logs?tail=${2:-100}" 2>/dev/null \
      | sed -n 's/^data: //p' | jq -r 'if type == "object" then (.line // .message // tostring) else tostring end' || true ;;
  cost)
    # what the cluster costs per hour now, and the account's pod billing as the API reports it
    "$0" ps | tail -n 1
    api GET /billing/pods | jq -c '.' | { head -c 2000; cat >/dev/null; }; echo ;;
  catalog)
    # the configured GPU types: price on the chosen cloud and availability in global-networking data centers
    dcs="$(gn_data_centers)"
    api GET "/catalog/gpus?include=AVAILABILITY&product=POD" | jq -r --arg types "$gpu_types" --arg dcs "$dcs" --arg cloud "$cloud" '
      ($types | split(",") | map(ltrimstr(" "))) as $ts | ($dcs | split(",")) as $gn
      | .gpus[] | select(.id as $i | $ts | index($i))
      | "\(.id)\t\(if $cloud == "COMMUNITY" then .price.community else .price.secure end) USD/h\t" +
        ([(.dataCenters // [])[] | select((.id as $d | $gn | index($d)) and .availability != "NONE") | "\(.id)=\(.availability)"] | join(" "))' ;;
  *)
    echo "unknown verb '$verb'; see deploy/driver.sh" >&2; exit 2 ;;
esac
