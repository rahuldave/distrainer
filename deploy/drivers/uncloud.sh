#!/usr/bin/env bash
# uncloud driver: containers on a set of Docker hosts joined by uncloud's WireGuard mesh are the Ray
# nodes (docs/running-modes.md C; OrbStack Linux machines on the Mac through deploy/uncloud/
# machines.sh, or real machines). See deploy/driver.sh for the verbs. Compared with compose:
#   - deploy/uncloud/compose.yml is deployed with `uc deploy` (head and MinIO pinned to the head
#     machine, DISTRAINER_UNCLOUD_HEAD_MACHINE, the first of DISTRAINER_UNCLOUD_MACHINES; workers
#     spread over the other machines, DISTRAINER_UNCLOUD_WORKER_MACHINES); `up N` sets the replica
#     count, `scale N` uses `uc scale`, which stops a removed worker gracefully (a preemption
#     notice, as `docker stop`).
#   - nothing spans machines: storage is S3 only (MinIO on the head machine, or any endpoint),
#     `shared` prints nothing, `wipe-shared` has nothing to wipe, `cp-from-head` streams a tar
#     through `uc exec`. The MinIO volume lives on the head machine; `down` keeps it, `nuke` drops it.
#   - the image is pushed to every machine by `build` (`uc image push`); nothing is pulled from a
#     registry. No bind mounts: a code edit needs `build` again.
#   - node death is `docker kill` over ssh on the machine that runs the container (uncloud has no
#     per-container kill); the container is started again after DISTRAINER_RESTART_DELAY seconds
#     as with compose. The ssh route to a machine is DISTRAINER_UNCLOUD_SSH, a printf template
#     of the ssh destination with the machine name (default `%s@orb`, OrbStack's route, the one
#     `uc` uses), plus DISTRAINER_UNCLOUD_SSH_OPTS, extra ssh options (`-F <config>` for the
#     AWS bed, whose bootstrap writes one with a Host per machine); otherwise a port or a key go
#     in ~/.ssh/config. Machines need passwordless sudo for docker (uncloud's own requirement).
#   - `endpoint` prints the dashboard at the head machine's address (its WireGuard endpoint
#     address, where the ports are published; DISTRAINER_UNCLOUD_HEAD_ADDRESS overrides it, the
#     public address on AWS) and then the store: `minio=` at that address when MinIO on the head
#     machine is the store (S3_ENDPOINT unset or http://minio:9000), else `s3=<S3_ENDPOINT>`, a
#     store outside the cluster that the containers reach directly and the runner reads as is
#     (MinIO is not deployed then; the bucket exists before `up`).
#   - every uc call names its context (DISTRAINER_UNCLOUD_CONTEXT, default distrainer): down and
#     nuke delete things. A failing uc fails the verb, and never reads as "no containers".
#   - the machines come from a bootstrap script the `machines-*` verbs dispatch to by
#     DISTRAINER_UNCLOUD_PROVIDER: orbstack (deploy/uncloud/machines.sh, the default) or aws
#     (deploy/uncloud/aws.sh). A bootstrap may write a second env file with the context, the
#     machines, the ssh route and the store's settings; .env names it as DISTRAINER_ENV_FILE and
#     it is read after .env (so its S3_* lines win over the MinIO defaults there). What the
#     caller's environment sets wins over both files (docker compose's own precedence), so
#     `DISTRAINER_UNCLOUD_PROVIDER=orbstack deploy/driver.sh machines-up` means OrbStack even
#     while .env points at the AWS bed.
# Worker index I (kill-worker, stop-worker) counts worker containers sorted by machine then
# container id, 1-based: stable across a kill and restart of the same container, not across a
# scale or a redeploy (a new container's id sorts anywhere), so scale first, then look up.
set -euo pipefail
root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
# .env, then the env file it (or the caller) names as DISTRAINER_ENV_FILE (a bootstrap's,
# deploy/uncloud/aws.sh), and what the caller's environment sets wins over both files at every
# step (docker compose's own precedence): the exported variables are restored after each file.
# An env file that declares another bed than the caller pinned (DISTRAINER_UNCLOUD_PROVIDER)
# is skipped, so `just uncloud-machines` means the OrbStack bed while .env points at AWS.
caller_env="$(export -p)"
if [ -f "$root/.env" ]; then set -a; . "$root/.env"; set +a; eval "$caller_env"; fi
if [ -n "${DISTRAINER_ENV_FILE:-}" ]; then
  env_file="$DISTRAINER_ENV_FILE"
  case "$env_file" in /*) ;; *) env_file="$root/$env_file" ;; esac
  if [ ! -f "$env_file" ]; then echo "DISTRAINER_ENV_FILE=$DISTRAINER_ENV_FILE does not exist" >&2; exit 2; fi
  declared="$(sed -n 's/^\(export \)\{0,1\}DISTRAINER_UNCLOUD_PROVIDER=//p' "$env_file" | tail -1 \
    | sed 's/[[:space:]]*#.*$//; s/[[:space:]]*$//; s/^["'"'"']//; s/["'"'"']$//')"   # as the runner's reader: export, quotes, comments
  if [ -n "${DISTRAINER_UNCLOUD_PROVIDER:-}" ] && [ -n "$declared" ] && [ "$DISTRAINER_UNCLOUD_PROVIDER" != "$declared" ]; then
    echo "note: $DISTRAINER_ENV_FILE belongs to the $declared bed; not read for provider $DISTRAINER_UNCLOUD_PROVIDER" >&2
  else
    set -a; . "$env_file"; set +a; eval "$caller_env"
  fi
fi
ctx="${DISTRAINER_UNCLOUD_CONTEXT:-distrainer}"
export UNCLOUD_CONTEXT="$ctx" UNCLOUD_AUTO_CONFIRM=true
image="${DISTRAINER_IMAGE:-distrainer:local}"
export DISTRAINER_IMAGE="$image"
compose="$root/deploy/uncloud/compose.yml"
provider="${DISTRAINER_UNCLOUD_PROVIDER:-orbstack}"   # what the machines-* verbs drive
case "$provider" in
  orbstack) machines_sh="$root/deploy/uncloud/machines.sh" ;;
  *) machines_sh="$root/deploy/uncloud/$provider.sh" ;;
esac
read -r -a machines <<< "${DISTRAINER_UNCLOUD_MACHINES:-uc1 uc2 uc3}"
head_machine="${DISTRAINER_UNCLOUD_HEAD_MACHINE:-${machines[0]}}"
export DISTRAINER_UNCLOUD_HEAD_MACHINE="$head_machine"
if [ -z "${DISTRAINER_UNCLOUD_WORKER_MACHINES:-}" ]; then   # every machine but the head machine, comma-separated
  DISTRAINER_UNCLOUD_WORKER_MACHINES="$(for m in "${machines[@]}"; do [ "$m" != "$head_machine" ] && printf '%s,' "$m"; done | sed 's/,$//')"
  if [ -z "$DISTRAINER_UNCLOUD_WORKER_MACHINES" ]; then
    echo "no worker machine: DISTRAINER_UNCLOUD_MACHINES needs a machine besides the head machine $head_machine" >&2; exit 2
  fi
fi
export DISTRAINER_UNCLOUD_WORKER_MACHINES
ssh_template="${DISTRAINER_UNCLOUD_SSH:-%s@orb}"
case "$ssh_template" in
  *%s*) ;;
  *) echo "DISTRAINER_UNCLOUD_SSH must contain %s for the machine name (got '$ssh_template')" >&2; exit 2 ;;
esac
ssh_opts="${DISTRAINER_UNCLOUD_SSH_OPTS:-}"   # word-split on purpose: `-F file`, `-p 2222`, ...
if [ -z "${DISTRAINER_UNCLOUD_HOST_PREFIX:-}" ]; then
  # published ports (dashboard, MinIO) bind only to the head machine's addresses inside this
  # prefix: OrbStack's machine network here (OrbStack forwards machine ports to the LAN
  # otherwise); elsewhere it must be set (0.0.0.0/0 = every address, MinIO's credentials are
  # the compose defaults)
  DISTRAINER_UNCLOUD_HOST_PREFIX="$(orb config get network.subnet4 2>/dev/null || true)"
  if [ -z "$DISTRAINER_UNCLOUD_HOST_PREFIX" ]; then
    echo "set DISTRAINER_UNCLOUD_HOST_PREFIX (the CIDR the head machine publishes ports on; no OrbStack here)" >&2; exit 2
  fi
fi
export DISTRAINER_UNCLOUD_HOST_PREFIX

machine_ssh() {   # machine_ssh MACHINE CMD...: run a docker command on the machine (sudo)
  local m="$1"; shift
  # shellcheck disable=SC2059,SC2086  # the template and the split are the point
  ssh $ssh_opts -o BatchMode=yes -o LogLevel=ERROR "$(printf "$ssh_template" "$m")" sudo -n "$@"
}
# a failing uc must fail the verb, never read as "no containers": the listings below abort on error,
# explicitly with `|| exit 1` at every capture (an `exit` inside a command substitution ends only
# that subshell), and a table whose header changed is a failure too, not an empty listing
uc_table() {   # uc_table HEADER[|HEADER] ARGS...: stdout of a uc listing whose first column is one of HEADER
  local headers="$1" out h; shift
  out="$(uc "$@" 2>/dev/null)" || { echo "uc $* failed (context $ctx); is the cluster up?" >&2; uc "$@" >/dev/null; exit 1; }
  for h in ${headers//|/ }; do
    case "$out" in "$h"*) printf '%s\n' "$out"; return 0 ;; esac
  done
  echo "uc $*: unexpected output (no $headers header):" >&2; printf '%s\n' "$out" >&2; exit 1
}
containers() {   # containers SERVICE -> "id machine status" per line, sorted by machine then id;
  # CREATED ("About a minute ago") and STATUS ("Up 3 minutes (healthy)") contain spaces
  uc_table SERVICE ps | awk -v s="$1" '$1 == s && $2 ~ /^[0-9a-f]{12}$/ {
    st = "?"; for (i = 3; i <= NF; i++) if ($i == "ago") {st = $(i + 1); break}
    print $2, $NF, st}' | sort -k2,2 -k1,1
}
container() {   # container SERVICE [I] -> "id machine" of the I-th container (default the only one)
  local all line
  all="$(containers "$1")" || exit 1
  line="$(sed -n "${2:-1}p" <<< "$all")"
  if [ -z "$line" ]; then echo "no container ${2:-1} of service $1 (run 'up' first):" >&2; echo "$all" >&2; exit 1; fi
  echo "${line% *}"
}
services_present() {   # "ref name" of our services in the cluster, ref being what `uc rm` takes:
  # uncloud can end up with two services of one name (seen after a run whose `down` failed
  # mid-way); `uc ls` then adds an ID column in front of NAME and every command that takes a
  # name refuses to choose, so the ref is the id when that column exists, else the name
  local all
  all="$(uc_table 'ID|NAME' ls)" || exit 1
  case "$all" in
    ID*)   awk 'NR > 1 && ($2 == "head" || $2 == "worker" || $2 == "minio") {print $1, $2}' <<< "$all" ;;
    *)     awk 'NR > 1 && ($1 == "head" || $1 == "worker" || $1 == "minio") {print $1, $1}' <<< "$all" ;;
  esac
}
remove_duplicates() {   # a name registered twice breaks deploy and scale: remove every copy first
  local present names dup id name
  present="$(services_present)" || exit 1
  names="$(awk '{print $2}' <<< "$present" | sort | uniq -d)"
  [ -n "$names" ] || return 0
  for dup in $names; do
    echo "up: service '$dup' is registered more than once; removing every copy before deploying" >&2
    while read -r id name; do
      if [ "$name" = "$dup" ]; then uc rm "$id"; fi   # not `&&`: the loop's status is its last test
    done <<< "$present"
  done
}
head_address() {   # the head machine's address other hosts and the Mac reach (its WireGuard endpoint)
  local all
  all="$(uc_table NAME machine ls)" || exit 1
  awk -v m="$head_machine" '$1 == m {for (i = 1; i <= NF; i++) if ($i ~ /:[0-9]+$/) {sub(/:[0-9]+$/, "", $i); print $i; exit}}' <<< "$all"
}
exec_head() { uc exec -T head -- "$@"; }
retry_once() {   # retry_once CMD...: uncloud's membership can show a machine as DOWN for a moment
  # (seen right as a container was being started on it); a second attempt of a declarative
  # deploy or scale reconciles
  if ! "$@"; then
    echo "$1 $2 failed once; retrying in 10 s" >&2
    sleep 10
    "$@"
  fi
}
need_cluster() {
  if ! uc machine ls >/dev/null 2>&1; then
    echo "uc context '$ctx' unreachable: run 'deploy/driver.sh machines-up' (just uncloud-machines, or just aws-machines) once" >&2; exit 2
  fi
}

verb="${1:-}"; shift || true
case "$verb" in
  build)
    docker build -t "$image" -f "$root/deploy/Dockerfile" "$root"
    need_cluster
    uc image push "$image" ;;
  machines-up|machines-status|machines-stop|machines-start|machines-destroy)
    [ -x "$machines_sh" ] || { echo "no bootstrap for DISTRAINER_UNCLOUD_PROVIDER=$provider ($machines_sh)" >&2; exit 2; }
    "$machines_sh" "${verb#machines-}" ;;
  up)
    n="${1:-2}"; shift || true
    minio="${DISTRAINER_MINIO:-0}"
    if [ "${1:-}" = "minio" ]; then minio=1; fi
    need_cluster
    services=(head worker)
    if [ "$minio" = "1" ]; then services=(minio head worker); fi
    export DISTRAINER_WORKERS="$n"
    remove_duplicates
    retry_once uc deploy -f "$compose" -y "${services[@]}"
    # uc deploy recreates a container it finds stopped (observed with a killed head and a stopped
    # worker); should one stay stopped, start it: `up` must always end with every node running
    for s in head worker; do
      list="$(containers "$s")" || exit 1
      while read -r id m st; do
        [ -n "$id" ] || continue
        case "$st" in Up) ;; *) machine_ssh "$m" docker start "$id" >/dev/null && echo "started $s $id on $m" ;; esac
      done <<< "$list"
    done
    uc ps ;;
  down)
    # the services go (by id, every copy), the MinIO volume on the head machine stays (S9 restores
    # from it); nuke drops it
    present="$(services_present)" || exit 1
    for s in worker head minio; do
      while read -r id name; do
        [ "$name" = "$s" ] || continue
        if ! uc rm "$id"; then   # a failed rm may still have removed the service: re-check first
          sleep 10
          if services_present | grep -q "^$id "; then uc rm "$id"; fi
        fi
      done <<< "$present"
    done ;;
  nuke)
    "$0" down
    vols="$(uc_table NAME volume ls)" || exit 1
    if awk 'NR > 1 {print $1}' <<< "$vols" | grep -qx minio-data; then uc volume rm -y minio-data; fi ;;
  wipe-shared)
    echo "uncloud: no shared mount (storage is S3 only); nothing to wipe" ;;
  scale)
    n="${1:?worker count}"
    retry_once uc scale worker "$n" -y ;;
  exec-head)
    exec_head "$@" ;;
  kill-worker)
    # SIGKILL on the machine = node death; started again after DISTRAINER_RESTART_DELAY s (0 = stays dead)
    i="${1:?worker index (1-based)}"
    target="$(container worker "$i")" || exit 1
    read -r id m <<< "$target"
    machine_ssh "$m" docker kill "$id" >/dev/null
    delay="${DISTRAINER_RESTART_DELAY:-5}"
    if [ "$delay" != "0" ]; then
      (sleep "$delay" && machine_ssh "$m" docker start "$id") >/dev/null 2>&1 &
      disown
      echo "worker $i ($id on $m) killed; restarting in ${delay}s as a new Ray node"
    else
      echo "worker $i ($id on $m) killed"
    fi ;;
  kill-head)
    # the head (Ray head, Train controller and driver) dies; `up` starts it again (S10)
    target="$(container head)" || exit 1
    read -r id m <<< "$target"
    machine_ssh "$m" docker kill "$id" >/dev/null
    echo "head ($id on $m) killed" ;;
  stop-worker)
    i="${1:?worker index (1-based)}"
    target="$(container worker "$i")" || exit 1
    read -r id m <<< "$target"
    machine_ssh "$m" docker stop "$id" >/dev/null
    echo "worker $i ($id on $m) stopped" ;;
  cp-from-head)
    # no volume to read on the Mac: stream a tar out of the head container. As `docker cp`, DST
    # names the copy, or the directory to copy into when it exists; an existing copy is not replaced
    src="${1:?src}"; dst="${2:?dst}"
    name="$(basename "$src")"
    case "$dst" in /|/.|/..) echo "cp-from-head: refusing to copy into '$dst'" >&2; exit 2 ;; esac
    if [ -d "$dst" ]; then target="${dst%/}/$name"; else target="$dst"; fi
    if [ -e "$target" ]; then echo "cp-from-head: $target exists; remove it first" >&2; exit 1; fi
    tmp="$(mktemp -d)"
    trap 'rm -rf "$tmp"' EXIT
    exec_head tar cf - -C "$(dirname "$src")" "$name" | tar xf - -C "$tmp"
    mkdir -p "$(dirname "$target")"
    mv "$tmp/$name" "$target" ;;
  shared)
    ;;   # nothing spans machines: the runner reads the bucket instead
  endpoint)
    ip="${DISTRAINER_UNCLOUD_HEAD_ADDRESS:-}"
    if [ -z "$ip" ]; then ip="$(head_address)" || exit 1; fi
    if [ -z "$ip" ]; then echo "endpoint: no address for head machine '$head_machine' in uc machine ls" >&2; exit 1; fi
    echo "dashboard=http://$ip:8265"
    case "${S3_ENDPOINT:-http://minio:9000}" in
      http://minio:*|http://minio.internal:*|http://minio/*|http://minio)   # MinIO on the head machine is the store
        echo "minio=http://$ip:9000 console=http://$ip:9001" ;;
      *) echo "s3=$S3_ENDPOINT" ;;   # a store outside the cluster: the containers and the runner reach it directly
    esac ;;
  mkbucket)
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
    uc machine ls; uc ps ;;
  logs)
    if [ "$#" -eq 0 ]; then set -- head; fi
    uc logs -n 100 "$@" ;;
  *)
    echo "unknown verb '$verb'; see deploy/driver.sh" >&2; exit 2 ;;
esac
