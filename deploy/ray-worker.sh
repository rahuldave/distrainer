#!/usr/bin/env bash
# Ray worker node = one training worker: 1 CPU and one `trainer` resource. Retries until the head
# answers, so workers may start before the head is ready or rejoin after a head restart. While
# the drain marker exists the node stays out of Ray but the container lives on: a driver whose
# nodes are expensive to replace (RunPod pods) removes and adds nodes that way.
set -uo pipefail
DRAIN_MARKER="${DISTRAINER_DRAIN_MARKER:-/tmp/distrainer-drained}"
while :; do
  while [ -f "$DRAIN_MARKER" ]; do sleep 3; done
  ray start --address="${RAY_HEAD_ADDRESS:-head:6379}" --num-cpus=1 \
    ${RAY_NODE_IP:+--node-ip-address="$RAY_NODE_IP"} \
    --resources='{"trainer": 1}' --object-store-memory="${OBJECT_STORE_BYTES:-200000000}" \
    --disable-usage-stats --block && exit 0
  if [ -f "$DRAIN_MARKER" ]; then echo "worker: drained; the node stays out of Ray until the marker goes" >&2; continue; fi
  echo "worker: head not reachable yet, retrying in 2s" >&2
  sleep 2
done
