#!/usr/bin/env bash
# Ray head: hosts GCS, the dashboard, the Train controller and the driver; advertises no
# `trainer` resource so it never runs a training worker. RAY_NODE_IP, when set, is the address
# the node advertises (RunPod's global-networking address; the default route's address otherwise).
set -euo pipefail
exec ray start --head --port=6379 --dashboard-host=0.0.0.0 --num-cpus="${HEAD_CPUS:-2}" \
  ${RAY_NODE_IP:+--node-ip-address="$RAY_NODE_IP"} \
  --object-store-memory="${OBJECT_STORE_BYTES:-200000000}" --disable-usage-stats --block
