#!/usr/bin/env bash
# Ray worker node = one training worker: 1 CPU and one `trainer` resource. Retries until the head
# answers, so workers may start before the head is ready or rejoin after a head restart.
set -uo pipefail
until ray start --address="${RAY_HEAD_ADDRESS:-head:6379}" --num-cpus=1 \
  ${RAY_NODE_IP:+--node-ip-address="$RAY_NODE_IP"} \
  --resources='{"trainer": 1}' --object-store-memory="${OBJECT_STORE_BYTES:-200000000}" \
  --disable-usage-stats --block; do
  echo "worker: head not reachable yet, retrying in 2s" >&2
  sleep 2
done
