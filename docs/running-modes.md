# Ways to run distrainer

The training code never changes between these modes. What changes is where the Ray nodes are,
where the driver script runs, which storage both can reach, and how you break things on purpose.
Every mode uses the same YAML shape (spec section 7); the fields that differ are called out below.

| | A. Laptop, single node | B. OrbStack containers | C. uncloud machines | D. KubeRay |
|---|---|---|---|---|
| Ray nodes | one: this Python process starts a local cluster | one container per node on a compose network | one container per node across a WireGuard mesh | one pod per node in a `RayCluster` |
| Training workers | Ray actors on the laptop | one worker container each (`trainer: 1` resource) | same | one worker pod each |
| Driver (`train.py`) | the same process (`ray_address: null`) | inside the `head` container (`ray_address: auto`) | inside the head container via `uc exec` | inside the head pod via `kubectl exec`, or a `RayJob` |
| Storage for blocks, log, audit, checkpoints | a local directory | a shared volume mounted at `/shared`, or MinIO | S3-compatible only (no volume spans machines): MinIO on the head machine, or S3/R2 | a `hostPath` volume at `/shared` (or a PVC), or MinIO in the cluster |
| Break things | not applicable | `docker kill` (node death), `docker stop` (preemption), `scale` | `docker kill` over ssh on the machine (node death), `docker stop` (preemption), `uc scale` | `kubectl delete pod --force` (node death; the operator replaces the pod), a `replicas` patch |
| Scenarios (spec sections 6.4 and 10) | S1, S5, S7 | S2, S3, S4, S6, S8, S9, S10, S11 (+ S1, S5, S7) | S2, S3, S4, S8, S9, S10, S11s3 on the bucket (S6 and S11 need the mount and skip) | the same set as B (S2, S3, S4 are the acceptance) |
| Milestone | M2 (done) | M3 (done) | M6 (done) | M5 (done) |
| Driver script | none, plain `uv run` | `deploy/drivers/compose.sh` | `deploy/drivers/uncloud.sh` (`DISTRAINER_DRIVER=uncloud`) | `deploy/drivers/kuberay.sh` (`DISTRAINER_DRIVER=kuberay`) |

All four modes run. The verbs of `deploy/driver.sh`
(`build`, `up N [minio]`, `down`, `nuke`, `scale N`, `exec-head`, `kill-worker I`, `stop-worker I`,
`cp-from-head`, `wipe-shared`, `shared`, `endpoint`, `mkbucket`) are the whole contract between the
Justfile or the scenario runner and a mode; nothing under `distrainer/`, `tests/`, or
`integration_tests/` knows which driver is active.

## A. Laptop, single node (what `just smoke` does)

`ray.init()` inside the driver starts a one-node cluster; `ScalingConfig(num_workers=2)` creates
two worker actors on the same machine, DDP runs over gloo between them, and `ray.train.report`
goes to a controller actor in the same process tree. Everything is on the local disk.

```yaml
ray_address: null            # local cluster in this process
storage: {kind: local}
storage_path: runs/hello     # checkpoints + Ray Train run state
store_root: blocks/hello     # blocks, log, audit
scaling: {num_workers: 2, resources_per_worker: {CPU: 1}}
```

```bash
just smoke                   # examples/hello_blocks, S1 + report-count check
just contrastive             # examples/toy_contrastive
just local-scenarios         # S1, S5, S7 (each a fresh hello_blocks run)
uv run distrainer log-ls blocks/hello -v
uv run distrainer inspect runs/hello/hello/checkpoint_g000003_p000012_n02_a00
uv run distrainer resume <checkpoint dir> --config examples/hello_blocks/local.yaml \
    --entry examples.hello_blocks.train:entry --run-name hello_resume
```

Use it for: the library itself, the training loop, checkpoint cadence, determinism, and the
resume arithmetic. It cannot lose a node, so it says nothing about fault tolerance or elasticity.

Two things are specific to this mode. Local relative paths are made absolute when the config is
loaded, because Ray Train workers do not share the driver's working directory. And Ray's `uv run`
integration is switched off by `distrainer.trainer.init_ray` so workers use the same interpreter
instead of re-launching through `uv` from an uploaded copy of the repository.

## B. OrbStack containers as Ray nodes (the local multi-node harness, M3)

OrbStack runs Linux containers in a lightweight VM on the Mac; Docker Compose gives them a private
network and DNS, so a `head` container and `N` `worker` containers behave like `N + 1` machines.
Each worker container runs `deploy/ray-worker.sh`:
`ray start --address=head:6379 --num-cpus=1 --resources='{"trainer":1}'`, retried until the head
answers, so it is exactly one training worker; the head (`deploy/ray-head.sh`) advertises no
`trainer` resource, so it hosts the Ray head, the Train controller, the driver, and the dashboard
(`http://localhost:8265`) but never trains. The image is built once from `uv.lock`
(`just build`, 1.5 GB, arm64); `distrainer/`, `examples/`, and `integration_tests/` are mounted
over the image copy, so code edits on the Mac are live in every node without a rebuild.
`deploy/ray-head.sh` and `deploy/ray-worker.sh` run *inside* the image and do need `just build`;
`deploy/driver.sh` and `deploy/drivers/*.sh` run on the Mac. Ports are published on
`127.0.0.1` only: OrbStack forwards published ports to the LAN by default, and the Ray dashboard
(8265) accepts job submissions without authentication.

```yaml
ray_address: auto            # the driver runs inside the head container
storage: {kind: local}       # the shared volume ...
storage_path: /shared/runs
store_root: /shared/blocks
log: {W: 24}                 # a multiple of every allowed world size: 2 and 3
scaling: {num_workers: [2, 3], resources_per_worker: {CPU: 1, trainer: 1}}
failure: {max_failures: 3}
```

or, for the object-store scenarios (S9 cold restore, S10 head loss), MinIO in the same compose
project:

```yaml
storage:
  kind: s3
  endpoint: http://minio:9000   # S3_ACCESS_KEY / S3_SECRET_KEY from the environment
  region: auto
storage_path: distrainer/runs   # bucket/prefix
store_root: distrainer/blocks
```

Two things S9 and S10 taught: `just down` keeps the MinIO volume (that is what a cold restore
restores from; `just nuke` removes it), and a run name that already exists on the bucket is
*restored* by Ray Train rather than started afresh, so the scenario runner deletes the old run
state and audit trail on the bucket before each MinIO scenario. MinIO's image now lives at
`quay.io/minio/minio`.

```bash
just build           # once, or after uv.lock changes
just up 2            # head + 2 workers; just up-minio 2 adds MinIO
just blocks          # build the corpus inside the head container
just train CFG       # driver inside the head container
just kill-worker 2   # SIGKILL a worker container: Ray sees the node die; the container is
                     # started again 5 s later as a new node (docker kill never triggers a
                     # restart policy, so the driver does it)
just scale 3         # elastic resize: the controller restarts the group at the new size
just integration S2  # scenario runner drives the verbs and checks the audit trail
just down            # containers go, the MinIO volume stays; just nuke removes everything
```

Shared storage is a bind mount, `.harness/shared` on the Mac at `/shared` in every container, so
audit trails and checkpoints can be read on the host while a run is going. The toy workload is
slowed with `train.step_sleep_s` in the harness configs so that a run lasts long enough to be
killed or resized mid-way (at full speed, 96 blocks take 0.35 s).

The Mac has 16 GB and the OrbStack VM 8 GB, so scenarios are written for 2 to 3 worker containers.
`docker kill` is node death; `docker stop` is a graceful drain that looks like a preemption notice.
Images are arm64-native; containers are reachable from the Mac as `service.project.orb.local`.

Use it for: everything that involves a node disappearing or the world size changing, the S3 path
of the log commit protocol, and head loss. It is the exit gate for v0.1.

## C. uncloud machines (M6)

uncloud is the same shape at machine scale: a set of Docker hosts joined by a WireGuard mesh with
cluster DNS, driven by a Compose-compatible file (`uc deploy`, `uc scale`, `uc exec`).
`deploy/drivers/uncloud.sh` implements the driver verbs on the `uc` CLI, so every harness target
and every bucket scenario runs with `DISTRAINER_DRIVER=uncloud`; `deploy/uncloud/compose.yml` is
the cluster (the same node image, scripts and Ray environment as mode B's compose file, pinned
against it by `tests/test_deploy_manifests.py`), and `deploy/uncloud/machines.sh` builds the
machines on the Mac. The one hard difference from every other mode: no volume spans machines, so
storage is S3-compatible only (MinIO on the head machine, or S3/R2), the driver's `shared` verb
prints nothing, and the scenario runner reads the audit trail and the checkpoints from the bucket
through the driver's `endpoint` instead of a shared directory. Tutorial 4
(`docs/tutorials/uncloud.md`) walks through it; `docs/uncloud-gotchas.md` lists what bites.

```bash
just uncloud-machines               # once: three OrbStack machines joined into the uncloud context `distrainer`
export DISTRAINER_DRIVER=uncloud
just build                          # docker build, then uc image push to every machine (no registry)
just up-minio 2 && just mkbucket && just blocks examples/hello_blocks/harness-minio.yaml \
    && just train examples/hello_blocks/harness-minio.yaml && just down
just integration S2                 # S3, S4, S8, S9, S10, S11s3 likewise; S6 and S11 skip (no shared mount)
```

What is where:

- The machines: OrbStack Linux machines `uc1` (the head machine, 5 GB, 2 CPUs), `uc2` and `uc3`
  (2 GB each), each with Docker and the uncloud daemon installed by `uc machine init` /
  `uc machine add` over OrbStack's `ucN@orb` ssh route (uncloud runs the system `ssh`, so the
  ProxyCommand in `~/.orbstack/ssh/config` applies and no sshd is needed inside). Each machine's
  address on OrbStack's machine network (`192.168.138.0/23`) is its WireGuard endpoint, given
  explicitly; the mesh (`uncloud` interface, MTU 1420) carries a Docker network `uncloud` with
  one `/24` per machine out of `10.210.0.0/16`, which the bootstrap checks against OrbStack's
  subnet. The memory limits are cgroup caps: the head machine runs the Ray head, the Train
  controller, the driver script and MinIO and thrashed under 3 and 4 GB.
- Placement: `head` and `minio` are pinned to the head machine with `x-machines`, `worker`
  replicas to the other machines (a comma-separated `x-machines` the driver exports as
  `DISTRAINER_UNCLOUD_WORKER_MACHINES`); a worker placed next to the head starved it. `up N`
  deploys with `deploy.replicas: N`, `scale N` is `uc scale worker N`, which stops a removed
  worker with a 10 s grace, the preemption notice `docker stop` gives in mode B.
- Names: every container gets `search internal.` from uncloud, so `head:6379` in
  `ray-worker.sh` and `http://minio:9000` in the harness configs resolve cluster-wide unchanged.
- Break things: `kill-worker I` and `kill-head` are `sudo docker kill` over ssh on the machine
  that runs the container (uncloud has no per-container kill; the machine's ssh destination is
  the `DISTRAINER_UNCLOUD_SSH` template, `%s@orb` by default); the container is started again
  after `DISTRAINER_RESTART_DELAY` seconds as in mode B. `stop-worker` is `docker stop`. `up`
  after `kill-head` finds the head stopped and `uc deploy` recreates it; the workers reconnect.
- The image is pushed, not pulled: `build` is `docker build` plus `uc image push` (the 1.36 GB
  dependency layer takes about three minutes to reach three machines, a code change seconds);
  no source tree is mounted, so a code edit needs `build` again.
- `endpoint` prints the dashboard and MinIO at the head machine's address, published only inside
  `DISTRAINER_UNCLOUD_HOST_PREFIX` (OrbStack's machine network; OrbStack forwards machine ports to
  the LAN otherwise). `cp-from-head` streams a tar through `uc exec` (there is no `uc cp`).
- Every `uc` call names its context (`DISTRAINER_UNCLOUD_CONTEXT`, default `distrainer`), a
  failing or reshaped `uc` listing fails the verb, `down` removes every copy of a service by id
  (uncloud can register a name twice), and `up` and `scale` retry once when membership shows a
  machine as `Down` for a moment.
- Timing: an `up` takes 60 to 90 s (containers start one at a time, monitored and
  health-checked), and a scenario 1.5 to 3 times its mode B wall time; the per-step pacing is
  the same, the difference is deploy time and object-store round trips over the mesh.
- On real machines (M7, tutorial 5, `docs/tutorials/aws.md`): `deploy/uncloud/aws.sh` is the EC2 twin of
  `machines.sh` (`DISTRAINER_UNCLOUD_PROVIDER=aws` routes the `machines-*` verbs to it; `stop`
  and `start` park and resume either bed): three Graviton instances, one security group (ssh
  from the Mac's address, WireGuard between the members), the cluster joined over the private
  addresses, and an S3 bucket with an IAM user scoped to it instead of MinIO. It writes
  `.harness/aws/env` (context, machines, the generated ssh config as
  `DISTRAINER_UNCLOUD_SSH_OPTS`, the head's public address, `S3_*`), which the driver and the
  runner read after `.env` through `DISTRAINER_ENV_FILE`. With `S3_ENDPOINT` naming a store
  outside the cluster, `endpoint` prints `s3=<URL>` instead of `minio=...`: the runner deploys
  no MinIO, makes no bucket, and runs `harness-s3.yaml` / `harness-stream-s3.yaml`.

## D. KubeRay on OrbStack Kubernetes (M5)

OrbStack has a built-in Kubernetes (k3s-based). With the KubeRay operator installed, a
`RayCluster` resource declares the head pod and a worker group, and the operator keeps the pods
matching it. `deploy/drivers/kuberay.sh` implements the driver verbs on top of that, so every
harness target and every cluster scenario runs unchanged with `DISTRAINER_DRIVER=kuberay`:

```bash
orb config set k8s.enable true      # once; restart OrbStack (orb stop && orb start) to apply it
just kuberay-operator               # once: KubeRay v1.7.0 through kubectl's kustomize (no Helm)
just build                          # the same image as mode B; OrbStack's Kubernetes sees Docker's images
export DISTRAINER_DRIVER=kuberay
just up 2 && just blocks && just train && just down
just integration S2                 # S3, S4, S6, S8 likewise; S9, S10, S11s3 bring MinIO up in the cluster
```

What is where:

- `deploy/k8s/raycluster.yaml`: one `RayCluster` named `distrainer` (namespace `distrainer`): the
  head pod (`num-cpus: 2`, no `trainer` resource) and the `workers` group (`num-cpus: 1`,
  `resources: {"trainer": 1}` in `rayStartParams`). The driver renders the host paths, the image
  and the replica count into it (`up N`) and JSON-patches `replicas` (`scale N`);
  `minReplicas` / `maxReplicas` (1 to 8) bound what `scale` may ask for, while the trainer's
  elastic bounds stay `scaling.num_workers` in the training config.
- Storage: `/shared` in every pod is a `hostPath` volume of `.harness/shared`, the same directory
  the compose driver bind-mounts, because OrbStack's Kubernetes runs in the VM that already sees
  the Mac filesystem. The scenario runner reads audit trails and checkpoints exactly as in mode
  B, and the source tree is mounted over the image copy the same way (edits are live).
- MinIO: `deploy/k8s/minio.yaml` (a Deployment, a Service `minio:9000`, a PVC that `down` keeps
  and `nuke` deletes), applied by `up N minio` or `DISTRAINER_MINIO=1`; the harness configs'
  `endpoint: http://minio:9000` resolves through cluster DNS. The `S3_*` settings (endpoint,
  region, credentials) come from `.env` with the same defaults as `docker-compose.yml`.
- Break things: `kill-worker I` force-deletes the I-th worker pod (creation order, 1-based); the
  operator starts a replacement pod at once, which joins as a new Ray node, so a killed worker
  always comes back (mode B's `DISTRAINER_RESTART_DELAY=0` has no equivalent). `kill-head`
  force-deletes the head pod; the operator recreates it and the workers restart into the new
  head. `stop-worker I` and a `scale` down delete gracefully: KubeRay's `preStop` hook runs
  `ray stop`, Ray reports that to the trainer as a preemption notice ("received preemption
  signal" in the driver log), and the pod is killed when `terminationGracePeriodSeconds` (10 s in
  the manifest, the same notice `docker stop` gives in mode B) runs out. Kubernetes' default of
  30 s let a scaled-down worker train to the end of a short run without ever dying, which is why
  the manifest sets it.
- The driver process: `exec-head` is `kubectl exec` into the head pod, so `train.py` runs on the
  head node as in mode B. `deploy/k8s/rayjob.yaml` is the Kubernetes-native alternative:
  `deploy/driver.sh submit [CONFIG]` creates a `RayJob` whose submitter Job runs `ray job submit`
  against the head's dashboard, and the entrypoint runs on the head node.
- The image needs `ray[default]`: KubeRay's readiness and liveness probes are HTTP checks against
  the dashboard agent (port 52365) and `ray job submit` talks to the dashboard, neither of which
  exists in Ray's minimal install (the dashboard logs "http server disabled" and the pods never
  become Ready). `pyproject.toml` pins `ray[data,default,train]` for that reason.
- Reverse DNS: `torch.distributed` looks up the peer's name on every process-group connection.
  CoreDNS answers PTR queries locally only for pods behind a headless Service and forwards the
  rest to the resolver outside the cluster; a slow upstream cost 15 s per lookup and stretched
  the trainer's start-up to about 40 s, which broke the pacing of S11. `raycluster.yaml` therefore
  adds a headless `distrainer-workers` Service over the worker pods (the head has one from
  KubeRay) and sets resolver `timeout: 2`, `attempts: 1` on the pods. Docker's embedded DNS
  answers those lookups itself in mode B, which is why compose never showed it.
- Every `kubectl` call of the driver names its context (`DISTRAINER_K8S_CONTEXT`, default
  `orbstack`) and namespace (`DISTRAINER_K8S_NAMESPACE`, default `distrainer`), so `down` and
  `nuke` never act on whatever context `kubectl` happens to point at; `render MANIFEST` prints
  what `up` would apply.
- `endpoint` prints the dashboard at the head pod's IP (the head Service is headless) and
  MinIO's ClusterIP; OrbStack routes pod and service IPs from the Mac, elsewhere use
  `kubectl port-forward svc/distrainer-head-svc 8265`.

## What is the same everywhere

- The block log and the ledger. A checkpoint from any mode can be inspected (`distrainer inspect`)
  and resumed in any other mode that can read the same storage; the ledger carries the world size
  it was written with, so the resume re-deals over whatever world exists now.
- The audit trail under `<store_root>/audit/<run_name>/`. The scenario checks read only this and
  the checkpoint metadata, never the cluster.
- `just verify` (lint, typecheck, compileall, unit and regression tests, smoke, diff check) needs
  only mode A; the harness scenarios (`just integration`, M3) need mode B, C or D and are not part of
  `verify`.
