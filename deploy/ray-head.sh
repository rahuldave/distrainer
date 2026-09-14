#!/usr/bin/env bash
# Ray head: hosts GCS, the dashboard, the Train controller and the driver; advertises no
# `trainer` resource so it never runs a training worker.
set -euo pipefail
exec ray start --head --port=6379 --dashboard-host=0.0.0.0 --num-cpus="${HEAD_CPUS:-2}" \
  --object-store-memory="${OBJECT_STORE_BYTES:-200000000}" --disable-usage-stats --block
