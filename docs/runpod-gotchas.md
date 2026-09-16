# RunPod gotchas

A running list, in the spirit of `docs/uncloud-gotchas.md`, of what bit while putting distrainer
on RunPod pods (M8, tutorial 6). Dates are when it was seen.

- **The image pull into a pod takes anything from 4 to 27 minutes** (2026-09-16): the 7.5 GB
  GPU image (3.5 GB compressed, one 6.6 GB CUDA layer) took 27 minutes from GHCR into a host in
  EU-RO-1 and 4 minutes into the next hosts (CA-MTL-1, and EU-RO-1 again), while the pod
  already billed. The pod's log shows one layer "Downloading" the whole time with no progress
  figure; the API reports `status: RUNNING` from the moment the pod is rented, but `runtime`
  stays null and `ssh.direct` absent until the container is up, which is what the driver waits
  for (a first version waited for `RUNNING` and returned with a worker still pulling). The driver's per-pod
  budget is 1500 s (`DISTRAINER_RUNPOD_START_TIMEOUT`) and workers are created right after the
  head so the pulls overlap; a pod outliving a timed-out `up` keeps pulling and the next `up`
  reuses it. A smaller image, or one of RunPod's cached base images, is the real remedy.
- **Cheap cards run out per data center** (2026-09-16): the head took the last RTX 2000 Ada in
  EU-RO-1 and every affordable type then answered "no longer any instances available" for the
  worker. The catalog's availability (`deploy/driver.sh catalog`) is a snapshot; a create is
  the only reservation. The driver falls back to any global-networking data center for a
  worker, at the price of a slower link to the head (CA-MTL-1 to EU-RO-1 on the second
  attempt); the blocks and checkpoints go through the bucket anyway, and the loss-side
  `all_gather` is kilobytes per step.
- **The global-networking interface arrives after the container starts** (2026-09-16): a
  one-shot lookup at start found no `10.x` address and no answer for the pod's own
  `<id>.runpod.internal`, so Ray advertised the container's `172.18.0.2`, which no other pod
  can reach (GCS at `<id>.runpod.internal:6379` answers, then hands out the wrong address for
  everything else). The entrypoint now looks for up to two minutes, inside a RunPod pod only;
  the head's log must show `runpod-entry: RAY_NODE_IP=10.…` and Ray's "Local node IP" must be
  that address.
- **DDP hangs across pods unless NCCL and Gloo are told the interface** (2026-09-16): the first
  two-node run sat for ten minutes with both Train workers alive and no step taken. NCCL (the
  DDP backend on GPUs) and Gloo bind the default route's interface, the container's `eth0`
  (172.18.x), which the other pod cannot reach; Ray itself was fine because it advertised the
  10.x address. The driver puts `NCCL_SOCKET_IFNAME=podnet1`, `GLOO_SOCKET_IFNAME=podnet1` and
  `NCCL_IB_DISABLE=1` into every pod's environment (`DISTRAINER_RUNPOD_NET_IFACE`), and the
  entrypoint derives the same from the route table as a fallback. An existing pod gets them
  through `PATCH /v2/pods/{id}` with the merged `env` while stopped, then `start`: no pull.
- **RunPod injects its API key into every pod** as `RUNPOD_API_KEY`, next to `RUNPOD_POD_ID`,
  `RUNPOD_DC_ID`, `RUNPOD_PUBLIC_IP`, `RUNPOD_TCP_PORT_22`, `RUNPOD_GPU_NAME`, `RUNPOD_CPU_COUNT`
  and `RUNPOD_MEM_GB`. The login-shell export carries it, as RunPod's own start script does.
  Never print a pod's environment into a log or a transcript; if it happens, rotate the key.
- **A login shell starts in `/root`**, not in the image's `WORKDIR`, and its environment is
  what the entrypoint exported to `/etc/profile.d`: `exec-head` runs `bash -lc "cd /app && …"`
  and the values are `%q`-quoted so a credential with `$` or a backtick survives.
- **The account's registered ssh keys** are what `startSsh` injects, and a shared account may
  hold none of this machine's. The driver injects `DISTRAINER_RUNPOD_SSH_KEY`'s `.pub` as
  `PUBLIC_KEY` instead, which makes RunPod skip the account keys for that pod.
- **The bucket's credentials are not in `.env`** on this Mac: they live in `.harness/aws/env`,
  which only the uncloud driver reads. Export the `S3_*` lines into the shell before a RunPod
  session (`export $(grep '^S3_' .harness/aws/env | xargs)`); putting them in `.env` would
  point the compose harness at the bucket too.
- **REST v1 is retired on 2026-11-15**; the driver is on v2 (`api.runpod.io/v2`). v2 takes one
  GPU type per create, has no name filter on the pod list (the driver filters client-side by
  name prefix and the `DISTRAINER_CLUSTER` marker), streams a pod's log as server-sent events,
  and reports a terminated pod as `TERMINATED` for a while: the driver never counts those.
- **A stopped pod restarts without a pull**, on the same host with the same id and internal
  name; the driver's `kill-worker` and `kill-head` are therefore stop and start, and only
  `down`, `scale` down and an errored pod terminate. A stopped pod's disk bills by the month.
- **The CIFAR-10 download is slow from anywhere** (the Toronto server served 72 KB/s on
  2026-09-16, 40 minutes for 170 MB). The tarball is staged in the bucket under `datasets/`;
  fetch it into the pod's `data_root` through a presigned URL before `make_blocks` (torchvision
  finds the tarball and skips the download). Writing the 192 blocks from a pod in Romania to a
  bucket in us-east-1 took 7.5 minutes (the transatlantic puts); once written, every run
  reuses them.
