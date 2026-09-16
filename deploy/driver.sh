#!/usr/bin/env bash
# Cluster-driver verb interface (spec section 9). Dispatches to deploy/drivers/<DISTRAINER_DRIVER>.sh
# (default: compose; kuberay for OrbStack's Kubernetes; uncloud for a WireGuard mesh of Docker
# hosts; runpod for GPU pods on RunPod over global networking). The verbs are the whole contract
# between the Justfile / scenario runner and a way of running containers:
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
#   cp-from-head SRC DST   copy a file or directory out of the head container (DST names the
#                          copy, or the directory to copy into when it already exists)
#   shared                 print the host path of the shared storage (nothing when no volume spans
#                          the nodes: the runner then reads the S3 store instead)
#   endpoint               print the dashboard URL and the store: minio=<URL> (MinIO in the cluster) or
#                          s3=<URL> (a store outside it, from S3_ENDPOINT; uncloud only)
#   mkbucket [NAME]        create the S3 bucket (MinIO profile)
#   ps | logs [SERVICE]    inspect (SERVICE is a compose service, or a pod under kuberay)
#   operator               (kuberay only) install the KubeRay operator once
#   submit [CONFIG]        (kuberay only) run hello_blocks as a RayJob on the running cluster
#   render MANIFEST        (kuberay only) print a rendered deploy/k8s manifest
#   machines-up | machines-status | machines-stop | machines-start | machines-destroy
#                          (uncloud only) the machines that form the uncloud cluster: OrbStack
#                          machines (deploy/uncloud/machines.sh) or EC2 instances
#                          (deploy/uncloud/aws.sh), by DISTRAINER_UNCLOUD_PROVIDER
#   cost | catalog         (runpod only) the cluster's hourly cost and the account's pod billing;
#                          the configured GPU types' prices and availability
# Every driver reads .env, with the caller's environment winning over it; the uncloud driver
# also reads the env file .env names as DISTRAINER_ENV_FILE (a bed's bootstrap wrote it).
set -euo pipefail
here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
driver="${DISTRAINER_DRIVER:-compose}"
exec "$here/drivers/$driver.sh" "$@"
