#!/usr/bin/env bash
# Entrypoint of the GPU node image on a RunPod pod (deploy/Dockerfile.gpu). RunPod injects the
# account's ssh public keys as PUBLIC_KEY when the pod is created with startSsh; its own images
# start sshd from it, so this image does the same, then exports the container's environment for
# ssh sessions (the driver's exec-head runs `bash -lc "cd /app && ..."`: a login shell reads
# /etc/profile.d and starts in /root, not in WORKDIR) the way RunPod's start script does
# (/etc/rp_environment), and finally runs the node's role:
#   runpod-entry.sh head      -> deploy/ray-head.sh   (the default CMD)
#   runpod-entry.sh worker    -> deploy/ray-worker.sh (RAY_HEAD_ADDRESS names the head pod)
#   runpod-entry.sh CMD...    -> anything else, for debugging
set -euo pipefail
if [ -n "${PUBLIC_KEY:-}" ]; then
  mkdir -p /root/.ssh && chmod 700 /root/.ssh
  printf '%s\n' "$PUBLIC_KEY" >> /root/.ssh/authorized_keys && chmod 600 /root/.ssh/authorized_keys
  mkdir -p /run/sshd
  /usr/sbin/sshd   # host keys were made at build time (ssh-keygen -A); daemonizes
fi
# the address Ray advertises: the pod's global-networking address (its own <id>.runpod.internal,
# or the 10.x interface), never the container's default one, which other pods cannot reach. The
# interface is attached moments after the container starts (seen 2026-09-16: nothing at start,
# the address there a little later), so look for up to two minutes before giving up
if [ -z "${RAY_NODE_IP:-}" ]; then
  ip=""; tries=0; max=1
  if [ -n "${RUNPOD_POD_ID:-}" ]; then max=60; fi   # only a RunPod pod has an address to wait for
  while [ -z "$ip" ] && [ "$tries" -lt "$max" ]; do
    if [ -n "${RUNPOD_POD_ID:-}" ]; then ip="$(getent hosts "$RUNPOD_POD_ID.runpod.internal" 2>/dev/null | awk '{print $1; exit}')" || ip=""; fi
    if [ -z "$ip" ]; then ip="$(hostname -I 2>/dev/null | tr ' ' '\n' | grep -m1 '^10\.')" || ip=""; fi
    if [ -z "$ip" ]; then tries=$((tries + 1)); sleep 2; fi
  done
  if [ -n "$ip" ]; then
    export RAY_NODE_IP="$ip"; echo "runpod-entry: RAY_NODE_IP=$ip after $tries retries" >&2
    # the collectives (NCCL for DDP on GPUs, Gloo otherwise) must use that interface too, or
    # their setup hangs on the container's default one; the driver sets these, this is the fallback
    iface="$(awk '$2 == "0000000A" { print $1; exit }' /proc/net/route 2>/dev/null)" || iface=""
    if [ -n "$iface" ]; then
      export NCCL_SOCKET_IFNAME="${NCCL_SOCKET_IFNAME:-$iface}" GLOO_SOCKET_IFNAME="${GLOO_SOCKET_IFNAME:-$iface}"
      export NCCL_IB_DISABLE="${NCCL_IB_DISABLE:-1}"
      echo "runpod-entry: collectives on $iface" >&2
    fi
  else
    echo "runpod-entry: no global-networking address found; Ray will advertise the container's own address (fine outside RunPod, unreachable for other pods on it)" >&2
  fi
fi
# every variable of the container (the S3 settings, RAY_*, DISTRAINER_*) for login shells; the
# key stays out, like RunPod's script keeps it, and so do the shell's own (a login shell has its
# own PWD, HOME and SHLVL; PATH comes from /etc/profile.d/distrainer-venv.sh); every value is
# quoted with %q so a credential with $, ` or \ survives the login shell untouched
: > /etc/rp_environment
printenv | grep -E '^[A-Za-z_][A-Za-z0-9_]*=' \
  | grep -Ev '^(PUBLIC_KEY|PATH|PWD|OLDPWD|HOME|SHLVL|HOSTNAME|_|TERM)=' \
  | while IFS= read -r line; do printf 'export %s=%q\n' "${line%%=*}" "${line#*=}"; done \
  >> /etc/rp_environment || true   # pipefail: an empty selection is not an error
cp /etc/rp_environment /etc/profile.d/distrainer-env.sh
role="${1:-head}"
case "$role" in
  head) exec deploy/ray-head.sh ;;
  worker) exec deploy/ray-worker.sh ;;
  *) exec "$@" ;;
esac
