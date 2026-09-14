# distrainer

Block-native distributed training on [Ray Train](https://docs.ray.io/en/latest/train/train.html):
configurable checkpointing anywhere between one batch and one epoch, row-exact resumption after
failures or preemption, elastic world size, and full control over what goes into a batch.

The core idea: make a **block** (a pre-built, named batch stored as a Parquet file) the unit of
composition, shuffling, dispatch, compute, and progress accounting. Progress becomes a single
number (the ledger cursor) that survives restarts and changes in the number of workers.

## Read first

- [`docs/introduction.md`](docs/introduction.md) — a from-zero introduction to distributed
  training, Ray Train, Ray Data, what Anyscale adds, and how distrainer's block abstraction works.
- [`docs/tutorials/batch.md`](docs/tutorials/batch.md) — tutorial 1: build a block log, train on
  it, read checkpoints and the ledger, resume with a different number of workers, passes.
- [`docs/tutorials/streaming.md`](docs/tutorials/streaming.md) — tutorial 2: train on a log that
  is still being written (an external producer, the re-mining segment hook), retention `gc`.
- [`docs/retention.md`](docs/retention.md) — the `gc` reference: what is deleted when, why the
  window is anchored on the last checkpoint, `num_to_keep` and what stays resumable, S3, non-goals.
- [`docs/cli.md`](docs/cli.md) — every command and flag: the `distrainer` CLI, the example
  scripts, the scenario checker, the `just` targets.
- [`docs/running-modes.md`](docs/running-modes.md) — the four ways to run the same code: laptop
  single-node Ray, OrbStack containers as Ray nodes, uncloud machines, KubeRay; where the driver
  runs, which storage works where, how failures are injected, which scenarios each validates.
- [`docs/handoff-m6.md`](docs/handoff-m6.md) — where development stands after M5 (KubeRay),
  what it taught, and the plan for uncloud (mode C), for whoever picks it up next
  (`docs/handoff-m5.md` is the same for M4 -> M5).
- [`docs/examples-and-scenarios.md`](docs/examples-and-scenarios.md) — the two example workloads,
  their configs and knobs, and every verification scenario: how it is driven, what it asserts, its
  status.
- [`docs/distrainer-spec.md`](docs/distrainer-spec.md) — the v0.1 specification: interfaces,
  training loop, checkpoint/storage layout, the docker compose harness, verification scenarios,
  milestones, and the development workflow.
- [`docs/distrainer-design.md`](docs/distrainer-design.md) — the design sketch that preceded the spec.
- [`docs/ray-sub-epoch-training-report.md`](docs/ray-sub-epoch-training-report.md) — research on
  Ray / Anyscale sub-epoch training and shard handling that motivated the design.

## Status

M1 (core library), M2 (`DistTrainer` on Ray Train v2, the `hello_blocks` and `toy_contrastive`
examples, the `distrainer` CLI, single-node scenarios S1, S5, S7) and M3 (the OrbStack container
harness behind a driver interface, scenarios S2, S3, S4 on a shared mount and S9, S10 on MinIO)
are merged, and so is M4 (PR #12): segment hooks configured in YAML, the re-mining hook that
streams the `toy_contrastive` log (S6), an external streaming producer (S11, and S11s3 on
MinIO), the time-budget checkpoint policy (S8) and retention `gc`. M5 (KubeRay) runs the same
cluster scenarios with pods as Ray nodes on OrbStack's Kubernetes through a second driver,
`deploy/drivers/kuberay.sh` (`DISTRAINER_DRIVER=kuberay`).
Milestones are in the spec (section 12);
development follows the `agent_gest_git_skills` workflow (section 14) and is tracked in GitHub
issues #1 to #7.

## Quick start

```bash
just setup                 # uv sync (Python 3.13, CPU torch, Ray 2.58)
just smoke                 # hello_blocks: 48 linear-regression blocks, 2 local workers, S1 check
just contrastive           # toy_contrastive: 240 mined blocks, InfoNCE encoder, 2 passes
just local-scenarios       # S1 happy path, S5 checkpoint cadence, S7 determinism
just contrastive examples/toy_contrastive/local-remine.yaml   # the log streamed by the re-mining hook
just build && just up 2 && just integration S2 && just down   # multi-container harness (OrbStack)
just kuberay-operator && DISTRAINER_DRIVER=kuberay just integration S2   # the same on OrbStack's Kubernetes (KubeRay)
uv run distrainer log-ls blocks/hello -v
uv run distrainer inspect runs/hello/hello/checkpoint_g000003_p000012_n02_a00
```

Your training code provides two functions, ``build_model(info) -> (model, optimizer)`` and
``train_step(model, optimizer, table, info) -> metrics``; `DistTrainer(train_step, build_model,
config).fit()` does the rest (see `examples/hello_blocks/train.py`).

## Development

Requires `uv`, `just`, and Python 3.13 (`.python-version`; 3.11+ supported). For the multi-node harness: Docker via OrbStack; for the KubeRay variant, OrbStack's Kubernetes (`orb config set k8s.enable true`) and `kubectl`.

```bash
just setup      # uv sync
just verify     # lint, typecheck, static, unit tests, smoke, diff-check
just up 2       # head + 2 worker containers (just up-minio 2 adds MinIO); DISTRAINER_DRIVER=kuberay for pods
```
