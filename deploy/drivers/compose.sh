#!/usr/bin/env bash
# docker compose driver (OrbStack on the Mac, or any Docker host). See deploy/driver.sh for verbs.
set -euo pipefail
root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
compose=(docker compose --project-directory "$root" -f "$root/deploy/docker-compose.yml")
profiles=()
if [ "${DISTRAINER_MINIO:-0}" = "1" ]; then profiles=(--profile minio); fi
shared="${DISTRAINER_SHARED:-$root/.harness/shared}"

verb="${1:-}"; shift || true
case "$verb" in
  build)
    "${compose[@]}" build head ;;
  up)
    n="${1:-2}"; shift || true
    if [ "${1:-}" = "minio" ]; then profiles=(--profile minio); fi
    mkdir -p "$shared"
    "${compose[@]}" ${profiles[@]+"${profiles[@]}"} up -d --scale "worker=$n" --remove-orphans
    "${compose[@]}" ${profiles[@]+"${profiles[@]}"} ps ;;
  down)
    # containers go, the MinIO volume stays (S9 restores from it); use nuke to drop everything
    "${compose[@]}" --profile minio down --remove-orphans ;;
  nuke)
    "${compose[@]}" --profile minio down -v --remove-orphans
    rm -rf "$shared" ;;
  wipe-shared)
    rm -rf "$shared" && mkdir -p "$shared" ;;
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
      (sleep "$delay" && docker start "distrainer-worker-$i" >/dev/null) &
      disown
      echo "worker $i killed; restarting in ${delay}s as a new Ray node"
    fi ;;
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
    bucket="${1:-distrainer}"
    "${compose[@]}" exec -T head python -c "
import os, pyarrow.fs as pafs
from urllib.parse import urlparse
u = urlparse(os.environ['S3_ENDPOINT'])
fs = pafs.S3FileSystem(access_key=os.environ['S3_ACCESS_KEY'], secret_key=os.environ['S3_SECRET_KEY'],
                       endpoint_override=u.netloc, scheme=u.scheme, region=os.environ.get('S3_REGION', 'auto'),
                       allow_bucket_creation=True)
fs.create_dir('$bucket')
print('bucket ready:', '$bucket')" ;;
  ps)
    "${compose[@]}" --profile minio ps ;;
  logs)
    "${compose[@]}" --profile minio logs --tail=100 "$@" ;;
  *)
    echo "unknown verb '$verb'; see deploy/driver.sh" >&2; exit 2 ;;
esac
