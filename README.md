# distrainer

Block-native distributed training on [Ray Train](https://docs.ray.io/en/latest/train/train.html):
configurable checkpointing anywhere between one batch and one epoch, row-exact resumption after
failures or preemption, elastic world size, and full control over what goes into a batch.

The core idea: make a **block** (a pre-built, named batch stored as a Parquet file) the unit of
composition, shuffling, dispatch, compute, and progress accounting. Progress becomes a single
number (the ledger cursor) that survives restarts and changes in the number of workers.

## Read first

The docs are also a site: <https://rahuldave.com/distrainer/> (`docs/`, built by MkDocs); `internal_docs/` holds
the handoffs, the agent workflow and the cheat sheets, which stay out of it.

- [`docs/introduction.md`](docs/introduction.md) — a from-zero introduction to distributed
  training, Ray Train, Ray Data, what Anyscale adds, and how distrainer's block abstraction works.
- [`docs/tutorials/batch.md`](docs/tutorials/batch.md) — tutorial 1: build a block log, train on
  it, read checkpoints and the ledger, resume with a different number of workers, passes.
- [`docs/tutorials/streaming.md`](docs/tutorials/streaming.md) — tutorial 2: train on a log that
  is still being written (an external producer, the re-mining segment hook), retention `gc`.
- [`docs/tutorials/kuberay.md`](docs/tutorials/kuberay.md) — tutorial 3: the same run on
  Kubernetes pods (KubeRay on OrbStack): setup, kill a worker, scale, a RayJob, MinIO, what to
  check when pods misbehave.
- [`docs/tutorials/uncloud.md`](docs/tutorials/uncloud.md) — tutorial 4: the same run on a
  cluster of machines (uncloud on OrbStack machines): the mesh, an object store instead of a shared
  mount, kill a worker, scale, lose the head, and what changes on cloud VMs with S3 or R2.
- [`docs/tutorials/aws.md`](docs/tutorials/aws.md) — tutorial 5: the same cluster on EC2 instances,
  an arm64 bed and an x86 bed, with an S3 bucket as the store and a private image repository: the
  account prerequisites, the scripts and their verbs, building and shipping the image for each
  architecture, the scenarios, day-to-day operation, the bill.
- [`docs/tutorials/runpod.md`](docs/tutorials/runpod.md) — tutorial 6: the image contrastive example (SimCLR on
  CIFAR-10 blocks, the machine learning explained), its GPU image and the Actions pipeline that builds it, RunPod pods (M8, in progress).
- [`docs/tutorials/parallel.md`](docs/tutorials/parallel.md) — tutorial 7: the parallel kinds (DDP, local SGD,
  DiLoCo, FSDP with sharded checkpoints) chosen by the `parallel:` section, run on the CPU.
- [`docs/collectives.md`](docs/collectives.md) — the primitives every distributed training is built from
  (barrier, broadcast, all-reduce, all-gather, reduce-scatter, send and receive), and where the loop uses them.
- [`docs/parallelism.md`](docs/parallelism.md) — the kinds of parallel training explained with those primitives
  (DDP, FSDP, tensor and pipeline, local SGD and DiLoCo), the head and rank 0, and where distrainer fits.
- [`internal_docs/handoff-m10.md`](internal_docs/handoff-m10.md) — the handoff after M9: what is there, what the
  parallel kinds taught, and what M10 runs on GPUs (`internal_docs/handoff-m9.md` is the M8 → M9 one).
- [`docs/runpod-gotchas.md`](docs/runpod-gotchas.md) — what bit on RunPod pods (pull times, capacity, the
  global-networking address, the injected API key).
- [`internal_docs/cheatsheets/orbstack.md`](internal_docs/cheatsheets/orbstack.md) and
  [`internal_docs/cheatsheets/uncloud.md`](internal_docs/cheatsheets/uncloud.md) — app-independent cheat sheets for the
  two tools the harness runs on.
- [`docs/uncloud-gotchas.md`](docs/uncloud-gotchas.md) — the running list of what uncloud does
  that the driver works around: membership, names, placement, memory caps, timing.
- [`docs/retention.md`](docs/retention.md) — the `gc` reference: what is deleted when, why the
  window is anchored on the last checkpoint, `num_to_keep` and what stays resumable, S3, non-goals.
- [`docs/cli.md`](docs/cli.md) — every command and flag: the `distrainer` CLI, the example
  scripts, the scenario checker, the `just` targets.
- [`docs/running-modes.md`](docs/running-modes.md) — the four ways to run the same code: laptop
  single-node Ray, OrbStack containers as Ray nodes, uncloud machines, KubeRay; where the driver
  runs, which storage works where, how failures are injected, which scenarios each validates.
- [`internal_docs/handoff-m8.md`](internal_docs/handoff-m8.md) — where development stands after M7 (the AWS bed),
  what it taught, and the pointers for what comes next, for whoever picks it up next
  (`internal_docs/handoff-m7.md`, `internal_docs/handoff-m6.md` and `internal_docs/handoff-m5.md` are the earlier ones).
- [`docs/examples-and-scenarios.md`](docs/examples-and-scenarios.md) — the two example workloads,
  their configs and knobs, and every verification scenario: how it is driven, what it asserts, its
  status; what each driver verb does under KubeRay and under uncloud.
- [`docs/distrainer-spec.md`](docs/distrainer-spec.md) — the v0.1 specification: interfaces,
  training loop, checkpoint/storage layout, the container harness and its KubeRay and uncloud variants, verification scenarios,
  milestones, and the development workflow.
- [`docs/distrainer-design.md`](docs/distrainer-design.md) — the design sketch that preceded the spec.
- [`internal_docs/gest_codex_workflow.md`](internal_docs/gest_codex_workflow.md) and
  [`internal_docs/tag_dependency_workflow.md`](internal_docs/tag_dependency_workflow.md) — the agent workflow behind
  `AGENTS.md` and `CLAUDE.md`: Gest tasks and iterations, tag classification, dependency impact.
- [`docs/ray-sub-epoch-training-report.md`](docs/ray-sub-epoch-training-report.md) — research on
  Ray / Anyscale sub-epoch training and shard handling that motivated the design.

## Status

M1 (core library), M2 (`DistTrainer` on Ray Train v2, the `hello_blocks` and `toy_contrastive`
examples, the `distrainer` CLI, single-node scenarios S1, S5, S7) and M3 (the OrbStack container
harness behind a driver interface, scenarios S2, S3, S4 on a shared mount and S9, S10 on MinIO)
are merged, and so is M4 (PR #12): segment hooks configured in YAML, the re-mining hook that
streams the `toy_contrastive` log (S6), an external streaming producer (S11, and S11s3 on
MinIO), the time-budget checkpoint policy (S8) and retention `gc`, and M5 (PR #13): the same
cluster scenarios with pods as Ray nodes on OrbStack's Kubernetes through a second driver,
`deploy/drivers/kuberay.sh` (`DISTRAINER_DRIVER=kuberay`), and M6 (PR #15): the bucket scenarios
on an uncloud cluster of OrbStack machines through a third driver, `deploy/drivers/uncloud.sh`
(`DISTRAINER_DRIVER=uncloud`), with the scenario runner reading the audit trail from the object
store when nothing is shared. M7 (PR #17) ran the same cluster on three EC2 instances with an S3 bucket as
the store (`deploy/uncloud/aws.sh`, tutorial 5); M8 (PR #21) put a GPU contrastive example on RunPod
pods through a fourth driver (`deploy/drivers/runpod.sh`, tutorial 6); `internal_docs/handoff-m9.md` says what it
taught and what comes next.
Milestones are in the spec (section 12);
development follows the `agent_gest_git_skills` workflow (section 14) and is tracked in GitHub
issues #1 to #7, #14 and #16.

## Quick start

```bash
just setup                 # uv sync (Python 3.13, CPU torch, Ray 2.58)
just smoke                 # hello_blocks: 48 linear-regression blocks, 2 local workers, S1 check
just contrastive           # toy_contrastive: 240 mined blocks, InfoNCE encoder, 2 passes
just local-scenarios       # S1 happy path, S5 checkpoint cadence, S7 determinism
just contrastive examples/toy_contrastive/local-remine.yaml   # the log streamed by the re-mining hook
just images                # image_contrastive: SimCLR on a CIFAR-10 subset (CPU size), a kNN probe at the end
just build && just up 2 && just integration S2 && just down   # multi-container harness (OrbStack)
just kuberay-operator && DISTRAINER_DRIVER=kuberay just integration S2   # the same on OrbStack's Kubernetes (KubeRay)
just uncloud-machines && DISTRAINER_DRIVER=uncloud just build && DISTRAINER_DRIVER=uncloud just integration S2   # on three OrbStack machines (uncloud)
uv run distrainer log-ls blocks/hello -v
uv run distrainer inspect runs/hello/hello/checkpoint_g000003_p000012_n02_a00
```

Your training code provides two functions, ``build_model(info) -> (model, optimizer)`` and
``train_step(model, optimizer, table, info) -> metrics``; `DistTrainer(train_step, build_model,
config).fit()` does the rest (see `examples/hello_blocks/train.py`).

## Development

Requires `uv`, `just`, and Python 3.13 (`.python-version`; 3.11+ supported). For the multi-node harness: Docker via OrbStack; for the KubeRay variant, OrbStack's Kubernetes (`orb config set k8s.enable true`) and `kubectl`; for the uncloud variant, the `uc` CLI (`brew install psviderski/tap/uncloud`).

```bash
just setup      # uv sync
just verify     # lint, typecheck, static, unit tests, smoke, diff-check
just up 2       # head + 2 worker containers (just up-minio 2 adds MinIO); DISTRAINER_DRIVER=kuberay for pods, uncloud for machines
```
