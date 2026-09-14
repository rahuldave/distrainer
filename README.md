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
- [`docs/running-modes.md`](docs/running-modes.md) — the four ways to run the same code: laptop
  single-node Ray, OrbStack containers as Ray nodes, uncloud machines, KubeRay; where the driver
  runs, which storage works where, how failures are injected, which scenarios each validates.
- [`docs/distrainer-spec.md`](docs/distrainer-spec.md) — the v0.1 specification: interfaces,
  training loop, checkpoint/storage layout, the docker compose harness, verification scenarios,
  milestones, and the development workflow.
- [`docs/distrainer-design.md`](docs/distrainer-design.md) — the design sketch that preceded the spec.
- [`docs/ray-sub-epoch-training-report.md`](docs/ray-sub-epoch-training-report.md) — research on
  Ray / Anyscale sub-epoch training and shard handling that motivated the design.

## Status

M1 (core library) is merged. M2 adds `DistTrainer` on Ray Train v2, the `hello_blocks` example
(`just smoke`), the `toy_contrastive` example (`just contrastive`), the `distrainer` CLI, and the
single-node scenarios S1, S5, S7 (`just local-scenarios`). Multi-container runs (M3) and segment
hooks / streaming (M4) come next. Milestones are in the spec (section 12); development follows
the `agent_gest_git_skills` workflow (section 14) and is tracked in GitHub issues #1 to #7.

## Quick start

```bash
just setup                 # uv sync (Python 3.13, CPU torch, Ray 2.58)
just smoke                 # hello_blocks: 48 linear-regression blocks, 2 local workers, S1 check
just contrastive           # toy_contrastive: 240 mined blocks, InfoNCE encoder, 2 passes
just local-scenarios       # S1 happy path, S5 checkpoint cadence, S7 determinism
# multi-container harness (M3, not built yet): just up 2 / just integration S2
uv run distrainer log-ls blocks/hello -v
uv run distrainer inspect runs/hello/hello/checkpoint_g000003_p000012_n02_a00
```

Your training code provides two functions, ``build_model(info) -> (model, optimizer)`` and
``train_step(model, optimizer, table, info) -> metrics``; `DistTrainer(train_step, build_model,
config).fit()` does the rest (see `examples/hello_blocks/train.py`).

## Development

Requires `uv`, `just`, and Python 3.13 (`.python-version`; 3.11+ supported). For the multi-node harness: Docker via OrbStack.

```bash
just setup      # uv sync
just verify     # lint, typecheck, static, unit tests, smoke, diff-check
just up 2       # head + 2 worker containers + MinIO
```
