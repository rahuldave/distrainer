#!/usr/bin/env bash
# docker compose driver (OrbStack on the Mac, or any Docker host). See deploy/driver.sh for verbs.
set -euo pipefail
root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
# compose interpolates variables from $root/.env; read the same file so host paths agree
# .env, then the env file it (or the caller) names as DISTRAINER_ENV_FILE (a bootstrap's,
# deploy/uncloud/aws.sh), and what the caller's environment sets wins over both files at every
# step (docker compose's own precedence): the exported variables are restored after each file
caller_env="$(export -p)"
if [ -f "$root/.env" ]; then set -a; . "$root/.env"; set +a; eval "$caller_env"; fi
if [ -n "${DISTRAINER_ENV_FILE:-}" ]; then
  env_file="$DISTRAINER_ENV_FILE"
  case "$env_file" in /*) ;; *) env_file="$root/$env_file" ;; esac
  if [ ! -f "$env_file" ]; then echo "DISTRAINER_ENV_FILE=$DISTRAINER_ENV_FILE does not exist" >&2; exit 2; fi
  set -a; . "$env_file"; set +a; eval "$caller_env"
fi
compose=(docker compose --project-directory "$root" -f "$root/deploy/docker-compose.yml")
profiles=()
if [ "${DISTRAINER_MINIO:-0}" = "1" ]; then profiles=(--profile minio); fi
shared="${DISTRAINER_SHARED:-$root/.harness/shared}"
case "$shared" in
  /*) ;;
  *) shared="$root/$shared" ;;   # compose resolves relative mounts against the project directory
esac
if [ "$shared" = "/" ] || [ -z "$shared" ]; then
  echo "refusing to operate on shared storage path '$shared'" >&2; exit 2
fi
marker="$shared/.distrainer-shared"   # nuke / wipe-shared only delete a directory `up` created

running_containers() { "${compose[@]}" --profile minio ps -q 2>/dev/null; }

verb="${1:-}"; shift || true
case "$verb" in
  build)
    "${compose[@]}" build head ;;
  up)
    n="${1:-2}"; shift || true
    if [ "${1:-}" = "minio" ]; then profiles=(--profile minio); fi
    if ! docker image inspect "${DISTRAINER_IMAGE:-distrainer:local}" >/dev/null 2>&1; then "${compose[@]}" build head; fi
    mkdir -p "$shared" && touch "$marker"
    "${compose[@]}" ${profiles[@]+"${profiles[@]}"} up -d --scale "worker=$n" --remove-orphans
    "${compose[@]}" ${profiles[@]+"${profiles[@]}"} ps ;;
  down)
    # containers go, the MinIO volume stays (S9 restores from it); use nuke to drop everything
    "${compose[@]}" --profile minio down --remove-orphans ;;
  nuke)
    "${compose[@]}" --profile minio down -v --remove-orphans
    if [ -d "$shared" ] && [ ! -f "$marker" ]; then
      echo "nuke: $shared was not created by this driver (no marker); not deleting it" >&2; exit 2
    fi
    rm -rf "$shared" ;;
  wipe-shared)
    # only meaningful with the containers down: a live bind mount would keep writing into the
    # deleted directory
    if [ -n "$(running_containers)" ]; then
      echo "wipe-shared: containers are running; run 'down' first" >&2; exit 2
    fi
    if [ -d "$shared" ] && [ ! -f "$marker" ]; then
      echo "wipe-shared: $shared was not created by this driver (no marker); not deleting it" >&2; exit 2
    fi
    rm -rf "$shared" && mkdir -p "$shared" && touch "$marker" ;;
  scale)
    n="${1:?worker count}"
    "${compose[@]}" ${profiles[@]+"${profiles[@]}"} up -d --no-recreate --scale "worker=$n" ;;
  exec-head)
    "${compose[@]}" exec -T head "$@" ;;
  kill-worker)
    # SIGKILL = node death. Docker treats kill like a manual stop (restart policies do not fire),
    # so the replacement node is started here after DISTRAINER_RESTART_DELAY seconds (0 = none).
    i="${1:?worker index (1-based)}"
    docker kill "distrainer-worker-$i"
    delay="${DISTRAINER_RESTART_DELAY:-5}"
    if [ "$delay" != "0" ]; then
      # detached from the caller's pipes so a capturing caller does not wait for the delay
      (sleep "$delay" && docker start "distrainer-worker-$i") >/dev/null 2>&1 &
      disown
      echo "worker $i killed; restarting in ${delay}s as a new Ray node"
    fi ;;
  kill-head)
    # SIGKILL the head: Ray head, Train controller and the driver die together (S10)
    docker kill distrainer-head-1 ;;
  stop-worker)
    i="${1:?worker index (1-based)}"
    docker stop "distrainer-worker-$i" ;;
  cp-from-head)
    "${compose[@]}" cp "head:${1:?src}" "${2:?dst}" ;;
  shared)
    echo "$shared" ;;
  endpoint)
    echo "dashboard=http://localhost:8265"
    echo "minio=http://localhost:9000 console=http://localhost:9001" ;;
  mkbucket)
    # MinIO may still be starting (nothing depends_on it); retry for a while
    bucket="${1:-distrainer}"
    for attempt in $(seq 1 15); do
      if "${compose[@]}" exec -T head python -c "
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
    "${compose[@]}" --profile minio ps ;;
  logs)
    "${compose[@]}" --profile minio logs --tail=100 "$@" ;;
  *)
    echo "unknown verb '$verb'; see deploy/driver.sh" >&2; exit 2 ;;
esac
