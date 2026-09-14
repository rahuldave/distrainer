#!/usr/bin/env bash
# Cluster-driver verb interface (spec section 9). Dispatches to deploy/drivers/<DISTRAINER_DRIVER>.sh
# (default: compose). The verbs are the whole contract between the Justfile / scenario runner and
# a way of running containers:
#   build                  build the node image
#   up N [minio]           head + N workers (+ MinIO with the minio profile)
#   down                   stop the containers (the MinIO volume survives for cold-restore tests)
#   nuke                   down plus every volume and the shared mount
#   wipe-shared            empty the shared mount (simulates losing the shared volume; containers must be down)
#   scale N                resize the worker set without recreating running containers
#   exec-head CMD...       run a command inside the head container (the driver runs here)
#   kill-worker I          SIGKILL worker I (node death); the container is started again after
#                          DISTRAINER_RESTART_DELAY seconds (default 5, 0 = stays dead)
#   kill-head              SIGKILL the head (Ray head, Train controller and driver die; S10)
#   stop-worker I          SIGTERM worker I (graceful drain, like a preemption notice)
#   cp-from-head SRC DST   copy a file or directory out of the head container
#   shared                 print the host path of the shared storage
#   endpoint               print dashboard / S3 URLs
#   mkbucket [NAME]        create the S3 bucket (MinIO profile)
#   ps | logs [SERVICE]    inspect
set -euo pipefail
here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
driver="${DISTRAINER_DRIVER:-compose}"
exec "$here/drivers/$driver.sh" "$@"
