# Ways to run distrainer

The training code never changes between these modes. What changes is where the Ray nodes are,
where the driver script runs, which storage both can reach, and how you break things on purpose.
Every mode uses the same YAML shape (spec section 7); the fields that differ are called out below.

| | A. Laptop, single node | B. OrbStack containers | C. uncloud machines | D. KubeRay |
|---|---|---|---|---|
| Ray nodes | one: this Python process starts a local cluster | one container per node on a compose network | one container per node across a WireGuard mesh | one pod per node in a `RayCluster` |
| Training workers | Ray actors on the laptop | one worker container each (`trainer: 1` resource) | same | one worker pod each |
| Driver (`train.py`) | the same process (`ray_address: null`) | inside the `head` container (`ray_address: auto`) | inside the head container via `uc exec` | a `RayJob` |
| Storage for blocks, log, audit, checkpoints | a local directory | a shared volume mounted at `/shared`, or MinIO | S3-compatible only (no volume spans machines) | a PVC (`hostPath` / `local-path`) or S3 |
| Break things | not applicable | `docker kill` (node death), `docker stop` (preemption), `scale` | `uc rm` / `uc scale` | `kubectl delete pod`, `kubectl scale` |
| Scenarios (spec sections 6.4 and 10) | S1, S5, S7 | S2, S3, S4, S6, S8, S9, S10, S11 (+ S1, S5, S7) | the same set, once machines exist | S1 to S4 |
| Milestone | M2 (done) | M3 | after M3, when machines are available | M5 |
| Driver script | none, plain `uv run` | `deploy/drivers/compose.sh` | `deploy/drivers/uncloud.sh` (stub) | `deploy/drivers/kuberay.sh` |

Modes B to D are the M3 and M5 plan (spec section 9); as of M2 only mode A runs, and `deploy/` is
empty. The design rule already holds: the verbs of `deploy/driver.sh` (`up N`, `down`, `exec-head`,
`kill-worker I`, `scale N`, `cp-from-head`, `endpoint`) will be the whole contract between the
Justfile or the scenario runner and a mode; nothing under `distrainer/`, `tests/`, or
`integration_tests/` may know which driver is active.

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

## B. OrbStack containers as Ray nodes (the local multi-node harness, M3, planned)

OrbStack runs Linux containers in a lightweight VM on the Mac; Docker Compose gives them a private
network and DNS, so a `head` container and `N` `worker` containers behave like `N + 1` machines.
The plan: each worker container starts
`ray start --address=head:6379 --num-cpus=1 --resources='{"trainer":1}'` and is therefore exactly
one training worker; the head advertises no `trainer` resource, so it hosts the Ray head, the Train
controller, the driver, and the dashboard but never trains. (The `trainer` key is already legal in
`resources_per_worker`; nothing starts containers yet.)

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

```bash
# M3 targets; the Justfile entries exist, the driver and compose files do not yet
just up 2            # head + 2 workers (+ MinIO with the minio profile)
just blocks          # build the corpus inside the head container
just train CFG       # driver inside the head container
just kill-worker 2   # SIGKILL a worker container: Ray sees the node die
just scale 3         # elastic resize: the controller restarts the group at the new size
just integration S2  # scenario runner drives the verbs and checks the audit trail
just down
```

The Mac has 16 GB and the OrbStack VM 8 GB, so scenarios are written for 2 to 3 worker containers.
`docker kill` is node death; `docker stop` is a graceful drain that looks like a preemption notice.
Images are arm64-native; containers are reachable from the Mac as `service.project.orb.local`.

Use it for: everything that involves a node disappearing or the world size changing, the S3 path
of the log commit protocol, and head loss. It is the exit gate for v0.1.

## C. uncloud machines (later)

uncloud is the same shape at machine scale: a set of Docker hosts joined by a WireGuard mesh with
cluster DNS, driven by a Compose-compatible file (`uc deploy`, `uc scale`, `uc exec`). The compose
file from mode B is reused with uncloud's `x-` extensions for placement; the driver verbs are
implemented with `uc` commands in `deploy/drivers/uncloud.sh`. The one hard difference: no volume
spans machines, so storage must be S3-compatible (MinIO on one machine, or R2/S3), which is why
`storage.kind: s3` exists from M1 on. The driver ships as a documented stub until machines exist.

It can very likely be tested on the Mac without real machines: OrbStack also runs lightweight
Linux *machines* (`orb create ubuntu uc1`), each with its own systemd, SSH address
(`uc1@orb`), and the ability to run Docker. Two or three of those, each with Docker and the
uncloud daemon installed, joined with `uc machine init` / `uc machine add` over SSH, form a real
uncloud cluster with the WireGuard mesh between them, and MinIO on one of them provides the S3
store. That is the plan for filling in `deploy/drivers/uncloud.sh`; containers alone are not
enough because uncloud wants a Docker host per node.

## D. KubeRay on OrbStack Kubernetes (M5)

OrbStack has a built-in Kubernetes (k3s-based, `orb config set k8s.enable true`), so this mode
is also testable on the Mac. With the KubeRay operator, a `RayCluster` resource declares a
head pod and a worker group (`minReplicas` / `maxReplicas` mirror `num_workers: [min, max]`), and a
`RayJob` runs `train.py`. Node death is `kubectl delete pod`, elasticity is editing `replicas`,
storage is a PVC mounted at `/shared` in every pod or S3. The purpose of this mode is to prove that
nothing in v0.1 assumed compose networking; the scenario checker is unchanged.

## What is the same everywhere

- The block log and the ledger. A checkpoint from any mode can be inspected (`distrainer inspect`)
  and resumed in any other mode that can read the same storage; the ledger carries the world size
  it was written with, so the resume re-deals over whatever world exists now.
- The audit trail under `<store_root>/audit/<run_name>/`. The scenario checks read only this and
  the checkpoint metadata, never the cluster.
- `just verify` (lint, typecheck, compileall, unit and regression tests, smoke, diff check) needs
  only mode A; the harness scenarios (`just integration`, M3) need mode B and are not part of
  `verify`.
