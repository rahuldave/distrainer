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
# every variable of the container (the S3 settings, RAY_*, DISTRAINER_*) for login shells; the
# key stays out, like RunPod's script keeps it, and so do the shell's own (a login shell has its
# own PWD, HOME and SHLVL; PATH comes from /etc/profile.d/distrainer-venv.sh)
printenv | grep -E '^[A-Za-z_][A-Za-z0-9_]*=' \
  | grep -Ev '^(PUBLIC_KEY|PATH|PWD|OLDPWD|HOME|SHLVL|HOSTNAME|_|TERM)=' \
  | awk -F = '{ val = $0; sub(/^[^=]*=/, "", val); gsub(/"/, "\\\"", val); print "export " $1 "=\"" val "\"" }' \
  > /etc/rp_environment || true   # pipefail: an empty selection is not an error
cp /etc/rp_environment /etc/profile.d/distrainer-env.sh
role="${1:-head}"
case "$role" in
  head) exec deploy/ray-head.sh ;;
  worker) exec deploy/ray-worker.sh ;;
  *) exec "$@" ;;
esac
