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
- [`docs/distrainer-spec.md`](docs/distrainer-spec.md) — the v0.1 specification: interfaces,
  training loop, checkpoint/storage layout, the docker compose harness, verification scenarios,
  milestones, and the development workflow.
- [`docs/distrainer-design.md`](docs/distrainer-design.md) — the design sketch that preceded the spec.
- [`docs/ray-sub-epoch-training-report.md`](docs/ray-sub-epoch-training-report.md) — research on
  Ray / Anyscale sub-epoch training and shard handling that motivated the design.

## Status

Pre-M0 skeleton. Milestones are in the spec (section 12); development follows the
`agent_gest_git_skills` workflow (section 14).

## Development

Requires `uv`, `just`, and Python 3.13 (`.python-version`; 3.11+ supported). For the multi-node harness: Docker via OrbStack.

```bash
just setup      # uv sync
just verify     # lint, typecheck, static, unit tests, smoke, diff-check
just up 2       # head + 2 worker containers + MinIO
```
