#!/usr/bin/env bash
# uncloud driver: STUB. The verbs of deploy/driver.sh map onto the uncloud CLI roughly as below;
# fill in once a cluster of Docker hosts exists (OrbStack Linux machines joined with
# `uc machine init/add` are the planned local test bed, see docs/running-modes.md).
#   build          -> uc image build / push (Unregistry), or a registry the machines can pull from
#   up N [minio]   -> uc deploy -f deploy/docker-compose.yml (x-machines placement), worker scale N
#   down           -> uc rm distrainer
#   scale N        -> uc scale distrainer-worker N
#   exec-head CMD  -> uc exec distrainer-head -- CMD
#   kill-worker I  -> uc rm --force <container> (node death) ; stop-worker -> uc stop
#   cp-from-head   -> uc cp (or read the S3 store directly: no shared volume spans machines)
#   endpoint       -> the service DNS names (head.internal, minio.internal)
set -euo pipefail
echo "uncloud driver is not implemented yet (verb: ${1:-}); set DISTRAINER_DRIVER=compose" >&2
exit 2
