# distrainer — specification v0.1

*September 9, 2026. Supersedes the sketch in `research/distrainer-design.md` where they differ. Target: Ray 2.58 (Train v2 default-on), PyTorch CPU build for local testing, Python 3.11.*

## 1. Scope

distrainer is an open-source library on top of Ray Train v2 that provides:

- **Block-native data ingest.** The unit of composition, shuffle, dispatch, compute, and progress accounting is a *block*: a pre-built set of rows with a stable id (e.g. one contrastive batch with its hard negatives).
- **Configurable checkpoint cadence** from one block to one epoch, plus chunk and epoch boundaries.
- **Row-exact resumption** after worker failure, node loss, or preemption, at block granularity, without per-rank iterator state.
- **Elastic world size** using `ScalingConfig(num_workers=(min, max))`, with progress that survives resizes.
- **Chunk hooks** for sub-epoch work such as hard-negative re-mining.

v0.1 implements the **per-rank lane loader** ("backend B"). The Ray Data streaming backend ("backend A") is out of scope for v0.1 but the interfaces leave room for it.

Non-goals for v0.1: model-parallel checkpoint merging (FSDP/DeepSpeed shards) beyond what `ray.train.report` already supports; job-driver (head node) fault tolerance; GPU-specific paths (everything must run on CPU for local tests, GPU is a config flag).

## 2. Concepts and invariants

**Block** — `BlockRef(block_id: str, locator: str, num_rows: int, meta: dict)`. The `locator` names a Parquet file (v0.1) in the `BlockStore`. Block contents are opaque to distrainer; the user's `train_step` interprets columns.

**Chunk** — an ordered list of blocks for `(epoch, chunk_idx)`, produced by the `Planner`. A chunk is the sub-epoch unit: re-mining, re-shuffling, and (optionally) checkpointing happen at chunk boundaries. Chunk size is `plan.blocks_per_chunk` or "whole epoch".

**Plan** — `Plan(epoch, chunk_idx, block_ids: List[str], seed)`. Deterministic given `(store index, seed, epoch, chunk_idx)`.

**Assignment rule** — plan position `i` is consumed by rank `i mod n` at global step `i // n`. The last `len(plan) mod n` positions are dropped (documented `drop_last` semantics) so every rank takes the same number of steps.

**Global step** — one block per rank, all ranks. Steps are aligned because `ray.train.report` is a barrier.

**Ledger** — `Ledger(epoch, chunk_idx, cursor)` where `cursor` = number of completed global steps in the current chunk. Saved in every checkpoint as `ledger.json`. Invariant: at any checkpoint, all ranks have consumed exactly plan positions `[0, cursor * n)`.

**Resume rule** — rebuild `Plan(epoch, chunk_idx)`, take `block_ids[cursor * n_old:]`, re-deal over current `n_new` with the assignment rule. `n_old` is stored in the ledger. Blocks between the last checkpoint and the failure are replayed (bounded by checkpoint cadence).

**Audit trail** — every consumed block is appended as `{"attempt": run_attempt_id, "rank": r, "world_size": n, "epoch": e, "chunk": c, "step": k, "block_id": id, "ts": ...}` to `<storage>/audit/<run_name>/<attempt>-<rank>.jsonl`. This is what the verification harness reads.

## 3. Package layout

```
distrainer/
  __init__.py
  block.py        # BlockRef, block read/write helpers (pyarrow)
  store.py        # BlockStore: index + parquet files on shared storage
  planner.py      # Planner protocol + SeededPermutationPlanner
  ledger.py       # Ledger dataclass, (de)serialization, resume arithmetic
  policy.py       # CheckpointPolicy protocol + EveryKSteps, ChunkEnd, EpochEnd, TimeBudget, Any([...])
  loader.py       # LaneLoader: per-rank prefetching iterator over a lane of BlockRefs
  hooks.py        # ChunkHook protocol (on_chunk_end)
  trainer.py      # DistTrainer: wraps TorchTrainer, owns train_func
  audit.py        # audit-log writer/reader
  config.py       # DistrainerConfig (dataclasses), YAML loading
examples/
  toy_contrastive/
    make_blocks.py     # synthetic clustered data -> blocks with hard negatives (uses Ray Data)
    model.py           # tiny MLP encoder + InfoNCE with optional all_gather
    train.py           # DistTrainer entrypoint
    remine.py          # ChunkHook: re-embed, mine, write new chunk blocks
tests/                 # focused unit tests: planner determinism, ledger arithmetic, policy, loader
regression_tests/      # bug / API regression tests (added as bugs are found)
integration_tests/
  cluster/             # scenario runner that drives docker compose and checks audit logs (S1–S10)
deploy/
  Dockerfile
  docker-compose.yml
  ray-head.sh, ray-worker.sh
  k8s/                 # phase 2: KubeRay manifests
docs/                  # this spec, design, research report, gest_codex_workflow.md
AGENTS.md              # from agent_gest_git_skills AGENTS.template.md, project section filled in
CLAUDE.md              # adapter pointing at AGENTS.md / .agents/skills
.agents/skills/        # vendored g* skills (installed, not hand-edited)
Justfile               # python-uv command contract + harness targets (section 14)
pyproject.toml         # uv-managed; ruff, ty, pytest
```

## 4. Interfaces

```python
# block.py
@dataclass(frozen=True)
class BlockRef:
    block_id: str
    locator: str          # path relative to store root
    num_rows: int
    meta: dict = field(default_factory=dict)

def read_block(store_root: str, ref: BlockRef) -> pyarrow.Table: ...
def write_block(store_root: str, block_id: str, table: pyarrow.Table, meta: dict | None = None) -> BlockRef: ...

# store.py
class BlockStore:
    def __init__(self, root: str): ...                       # shared storage, e.g. /shared/blocks
    def put(self, table: pyarrow.Table, block_id: str, meta=None) -> BlockRef
    def get(self, ref: BlockRef) -> pyarrow.Table
    def list(self, namespace: str) -> list[BlockRef]         # namespace = e.g. "epoch0/chunk3" or "base"
    def write_index(self, namespace: str, refs: list[BlockRef]) -> None   # atomic (write tmp + rename)
    def read_index(self, namespace: str) -> list[BlockRef]

# planner.py
class Planner(Protocol):
    def plan(self, epoch: int, chunk_idx: int) -> Plan | None   # None = no more chunks in this epoch
    def num_chunks(self, epoch: int) -> int | None

@dataclass
class Plan:
    epoch: int
    chunk_idx: int
    block_ids: list[str]
    seed: int
    def lane(self, rank: int, world_size: int, start_pos: int = 0) -> list[str]:
        usable = (len(self.block_ids) - start_pos) // world_size * world_size
        return self.block_ids[start_pos:start_pos + usable][rank::world_size]

class SeededPermutationPlanner:
    """Blocks in namespace f"epoch{e}/chunk{c}" if present, else the base namespace,
    permuted with random.Random(hash((seed, e, c)))."""
    def __init__(self, store: BlockStore, seed: int, blocks_per_chunk: int | None): ...

# ledger.py
@dataclass
class Ledger:
    epoch: int = 0
    chunk_idx: int = 0
    cursor: int = 0          # completed global steps in this chunk
    world_size: int = 0      # n at the time of the checkpoint
    def resume_position(self) -> int: return self.cursor * self.world_size
    def save(self, dir: str) -> None; @classmethod def load(cls, dir: str) -> "Ledger"

# policy.py
class CheckpointPolicy(Protocol):
    def should_checkpoint(self, ctx: StepContext) -> bool
    # StepContext: global_step, steps_in_chunk, chunk_end: bool, epoch_end: bool, elapsed_s, rank

EveryKSteps(k)             # index-based, no communication
ChunkEnd(); EpochEnd()
TimeBudget(seconds, poll_every=M)   # rank 0 decides; broadcast_from_rank_zero every M steps
Any(policies)              # OR-combination

# loader.py
class LaneLoader:
    """Prefetching iterator over a lane of BlockRefs; yields (position, BlockRef, pyarrow.Table)."""
    def __init__(self, store: BlockStore, refs: list[BlockRef], prefetch: int = 2, threads: int = 2): ...
    def __iter__(self): ...
    def close(self): ...

# hooks.py
class ChunkHook(Protocol):
    def on_chunk_end(self, model, ledger: Ledger, ctx: TrainContextLite) -> None
    # may write new blocks + index for (epoch, chunk_idx + 1); runs on rank 0, others wait at a barrier

# trainer.py
class DistTrainer:
    def __init__(self, train_step, build_model, planner, store, policy,
                 config: DistrainerConfig, hooks: list[ChunkHook] = (),
                 scaling_config: ScalingConfig, run_config: RunConfig): ...
    def fit(self) -> ray.train.Result
# train_step(model, optimizer, table: pyarrow.Table, ctx) -> dict[str, float]   (user-provided; micro-batching inside)
# build_model(ctx) -> (model, optimizer)                                            (user-provided)
```

## 5. Training loop (inside `train_func`, every rank)

```
ctx      = ray.train.get_context(); rank, n = ctx.get_world_rank(), ctx.get_world_size()
model, opt = build_model(ctx); wrap with ray.train.torch.prepare_model
ledger   = Ledger()
ckpt     = ray.train.get_checkpoint()
if ckpt: load model/opt state; ledger = Ledger.load(ckpt_dir)
start_pos = ledger.resume_position()          # uses ledger.world_size (n_old)
for epoch in range(ledger.epoch, cfg.epochs):
    chunk_idx = ledger.chunk_idx if epoch == ledger.epoch else 0
    while (plan := planner.plan(epoch, chunk_idx)) is not None:
        lane   = plan.lane(rank, n, start_pos); start_pos = 0
        steps  = len(lane)                                   # identical on every rank by construction
        loader = LaneLoader(store, resolve(lane))
        for j, (pos, ref, table) in enumerate(loader):
            metrics = train_step(model, opt, table, ctx)
            audit.append(rank, n, epoch, chunk_idx, ledger.cursor, ref.block_id)
            ledger.cursor += 1; ledger.world_size = n
            chunk_end = (j == steps - 1); epoch_end = chunk_end and planner.plan(epoch, chunk_idx+1) is None
            if policy.should_checkpoint(StepContext(...)):
                report(metrics, checkpoint=save(model, opt, ledger) if rank == 0 else None,
                       checkpoint_upload_mode=ASYNC)
            else:
                report(metrics)                              # keeps report counts aligned
        loader.close()
        if rank == 0: for h in hooks: h.on_chunk_end(model, ledger, ctx)
        ray.train.collective.barrier()
        chunk_idx += 1; ledger.chunk_idx = chunk_idx; ledger.cursor = 0
    ledger.epoch = epoch + 1; ledger.chunk_idx = 0
```

Notes: `report` is called on every step so all ranks call it the same number of times regardless of policy; the cost of a metrics-only `report` is one small RPC. The checkpoint directory is a non-temporary dir (ASYNC upload requirement). Rank 0 saves the ledger *after* incrementing the cursor, so the ledger describes "steps completed including this one". Elastic resize or failure: Train restarts `train_func`; the ledger's `world_size` is the old `n`, `resume_position()` uses it, and the plan tail is re-dealt over the new `n`.

## 6. Checkpointing: where it happens, layout, storage, reconstitution

### 6.1 Who does what (Ray Train v2 mechanics)

- **The worker that calls `ray.train.report(checkpoint=...)` uploads.** `Checkpoint.from_directory(local_dir)` is a reference to a local directory; inside `report` the worker's train context copies that directory to `<storage_path>/<run_name>/<checkpoint_dir_name>/` with `pyarrow.fs.copy_files` (64 MB chunks; single-threaded for S3 uploads due to an Arrow issue). `SYNC` blocks the worker until the copy finishes; `ASYNC` copies in a per-worker thread (each `report` waits for the previous upload to finish); `NO_UPLOAD` skips the copy and trusts the `Checkpoint` you pass (use with `checkpoint_upload_fn` for custom uploaders). For DDP, rank 0 passes a checkpoint and other ranks pass `checkpoint=None`; for sharded states every rank passes its own directory and the files are merged into one checkpoint directory in storage.
- **The Train controller (on the head node) keeps the book.** Its `CheckpointManager` records every reported checkpoint with its metrics, enforces `CheckpointConfig(num_to_keep, checkpoint_score_attribute, checkpoint_score_order)` by deleting checkpoints from storage, and writes a JSON snapshot of its own state into the run directory so a restarted controller (`controller_failure_limit`) can continue. On worker-group restart it hands the latest checkpoint to workers via `ray.train.get_checkpoint()`.
- **Workers read.** `Checkpoint.to_directory(path)` downloads; `with checkpoint.as_directory() as d:` downloads to a temp dir (or yields the path directly when storage is local). `Checkpoint.get_metadata()` / `set_metadata(dict)` read/write a small JSON sidecar in the checkpoint directory without touching the payload.

Head container role: it hosts the Ray head, the Train controller, the driver (`train.py`), report generation over the audit logs, and the dashboard. It does not run a training worker (`ScalingConfig.resources_per_worker={"CPU": 1, "trainer": 1}` with workers started with `--resources='{"trainer": 1}'` and the head without).

### 6.2 Layout

```
<storage_path>/<run_name>/
  checkpoint_manager_snapshot.json         # Train's own bookkeeping (controller restarts)
  checkpoint_e{epoch}_c{chunk}_s{cursor}/  # checkpoint_dir_name set by distrainer
    model.pt                               # state_dict (rank 0) or model_rank{r}.pt shards
    optimizer.pt
    ledger.json                            # {"epoch","chunk_idx","cursor","world_size","plan_seed","run_attempt"}
    .metadata.json                         # Checkpoint.set_metadata: ledger + distrainer version (cheap to read)
```

distrainer's `CheckpointIO.save(model, opt, ledger) -> Checkpoint` writes to a **non-temporary** local dir (`/tmp/distrainer/<run>/ckpt_<cursor>`, required for ASYNC), calls `set_metadata({"ledger": ledger.asdict(), "distrainer": __version__})`, and deletes local dirs older than the last two after upload (`delete_local_checkpoint_after_upload=True`). `CheckpointIO.load(checkpoint) -> (state, Ledger)` uses `as_directory()`.

### 6.3 Storage backends: local, S3, and S3-compatible

`RunConfig(storage_path, storage_filesystem)` accepts a URI that pyarrow resolves (`/shared/runs`, `s3://bucket/prefix`, `gs://…`) or an explicit `pyarrow.fs.FileSystem` plus a prefix-stripped path. Ray requires `fsspec` for the local case (installed with `ray[train]`).

For **S3-compatible** stores (MinIO in the harness, Cloudflare R2, Ceph RGW, etc.) pass an explicit filesystem so the endpoint is unambiguous:

```python
import pyarrow.fs as pafs
fs = pafs.S3FileSystem(
    access_key=os.environ["S3_ACCESS_KEY"], secret_key=os.environ["S3_SECRET_KEY"],
    endpoint_override=os.environ["S3_ENDPOINT"],   # e.g. "http://minio:9000" or "https://<acct>.r2.cloudflarestorage.com"
    scheme="http" if os.environ["S3_ENDPOINT"].startswith("http://") else "https",
    region=os.environ.get("S3_REGION", "auto"),
)
run_config = RunConfig(name=run_name, storage_path="distrainer/runs", storage_filesystem=fs, ...)
```

distrainer exposes this as `config.storage: {kind: local|s3, path, endpoint, region, access_key_env, secret_key_env}` and builds the filesystem once in `storage.py`; the same `(fs, root)` pair is used by `BlockStore` (blocks are Parquet, read/written with `pyarrow.parquet` on the same fs), by `CheckpointIO`, and by the audit writer. Every worker and the head must have the same credentials (compose `environment:` or a `.env`). Note that with S3 storage the run dir is no longer on a shared volume, so `Result.from_path` and the report generator take the same `fs`.

Worker-local scratch (`/tmp/distrainer`) is the only non-shared location; nothing else in the design assumes a shared POSIX volume, so the compose `shared:` volume becomes optional once MinIO is enabled (kept for the audit logs by default because it is convenient to `tail`).

### 6.4 Reconstitution

Three entry points, all built on `ray.train.Checkpoint(path, filesystem)`:

1. **Automatic** — worker failure, preemption, or elastic resize: Train restarts the worker group and `ray.train.get_checkpoint()` returns the latest reported checkpoint; `train_func` loads the ledger and resumes at `cursor * world_size_old` (section 5).
2. **Explicit, from a run** — `Result.from_path("<storage_path>/<run_name>", storage_filesystem=fs)` restores the `Result` (latest and best checkpoints, metrics); `DistTrainer(..., resume_from_checkpoint=result.checkpoint)` starts a *new* run from it. This is the path for driver/head loss and for "continue training tomorrow".
3. **Explicit, from a URI** — `Checkpoint("s3://bucket/distrainer/runs/toy/checkpoint_e0_c3_s16", filesystem=fs)` (or a local path) → `resume_from_checkpoint=`. Works across runs, clusters, and world sizes because the ledger carries `world_size`.

CLI: `distrainer inspect <uri>` prints the ledger from `.metadata.json` without downloading weights; `distrainer resume <uri> --config cfg.yaml [--seed-override]` starts a run from it; `distrainer export <uri> <local_dir>` = `to_directory`. Reconstitution of the *plan* needs only `(store index, seed, epoch, chunk_idx)`, all present in the ledger plus the block store, so no per-rank state is ever required.

Verification scenario **S9 — cold restore**: run S1 to completion against MinIO, `down -v` the cluster (destroying the shared volume), `up`, then `distrainer resume s3://…/checkpoint_e0_c3_s16` for one more chunk and assert the audit positions continue from `cursor * world_size`. **S10 — head loss**: `docker kill head` mid-run, `up` again, `Result.from_path` + resume; same assertion.

## 7. Configuration

```yaml
run_name: toy
storage_path: /shared/runs
store_root: /shared/blocks
seed: 1234
epochs: 2
blocks_per_chunk: 16        # null = whole epoch
checkpoint:
  policy: any               # any | every_k | chunk_end | epoch_end | time
  every_k: 4
  time_budget_s: null
  num_to_keep: 3
loader:
  prefetch: 2
  threads: 2
scaling:
  num_workers: [2, 4]       # elastic (min, max); an int disables elasticity
  resources_per_worker: {CPU: 1}
  use_gpu: false
  elastic_resize_monitor_interval_s: 15
failure:
  max_failures: 3
hooks:
  remine: {every_chunk: true}
```

## 8. Toy workload (examples/toy_contrastive)

Synthetic data: `N=8192` items, `d=32` features drawn from `C=64` Gaussian clusters; positive = another item of the same cluster, hard negatives = items from the `k` nearest *other* clusters (by centroid distance). `make_blocks.py` uses Ray Data (`groupby("batch_id").map_groups`) to write blocks of `B=32` anchors, each row carrying `anchor, positive, neg_0..neg_{k-1}` and `item_id`. Model: 2-layer MLP encoder; loss: InfoNCE over in-block negatives, optional `all_gather` across ranks (config flag) to exercise the loss-side collective. `remine.py` re-embeds the items with the current model at each chunk end, recomputes nearest clusters in embedding space, and writes `epoch{e}/chunk{c+1}` blocks. CPU-only; one epoch of 256 blocks should train in well under a minute on an M1 with 4 worker containers.

## 9. Local multi-node harness (OrbStack, docker compose)

Containers are Ray nodes: one `head` and `N` `worker` services on a user-defined bridge network, sharing a named volume mounted at `/shared` (block store, checkpoints, audit logs). Ray Train needs shared storage across nodes; the volume provides it.

`deploy/Dockerfile` (arm64-native under OrbStack):

```dockerfile
FROM python:3.11-slim
RUN pip install --no-cache-dir "ray[data,train]==2.58.0" pyarrow pandas \
    && pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu
COPY . /app
RUN pip install -e /app
WORKDIR /app
```

`deploy/docker-compose.yml` (essentials):

```yaml
services:
  head:
    build: {context: .., dockerfile: deploy/Dockerfile}
    command: ray start --head --port=6379 --dashboard-host=0.0.0.0 --num-cpus=2 --block
    ports: ["8265:8265"]
    shm_size: "1g"
    volumes: ["shared:/shared"]
    environment: [RAY_TRAIN_V2_ENABLED=1]
  worker:
    build: {context: .., dockerfile: deploy/Dockerfile}
    command: ray start --address=head:6379 --num-cpus=1 --block
    depends_on: [head]
    shm_size: "1g"
    volumes: ["shared:/shared"]
    environment: [RAY_TRAIN_V2_ENABLED=1]
    deploy: {replicas: 2}
  minio:
    image: minio/minio
    command: server /data --console-address ":9001"
    environment: [MINIO_ROOT_USER=distrainer, MINIO_ROOT_PASSWORD=distrainer123]
    ports: ["9000:9000", "9001:9001"]
    volumes: ["minio:/data"]
volumes:
  shared: {}
  minio: {}
```

Workers and head get `S3_ENDPOINT=http://minio:9000`, `S3_ACCESS_KEY`, `S3_SECRET_KEY` via `.env`; `just mkbucket` creates the `distrainer` bucket with `mc`. Runs default to `storage.kind: s3` in the harness so that S9/S10 exercise the S3 path; `storage.kind: local` (the `shared:` volume) remains supported.

Each worker advertises 1 CPU so `resources_per_worker: {CPU: 1}` maps one Train worker per container — i.e. one container = one "node". The head advertises 2 CPUs for the Train controller and driver but is excluded from training by `ScalingConfig` resource shaping or by giving it a custom resource and `label_selector` if needed.

`justfile` targets:

```
up N=2:        docker compose -f deploy/docker-compose.yml up -d --build --scale worker={{N}}
down:          docker compose -f deploy/docker-compose.yml down -v
blocks:        docker compose exec head python examples/toy_contrastive/make_blocks.py --out /shared/blocks
train CFG:     docker compose exec head python examples/toy_contrastive/train.py --config {{CFG}}
kill-worker I: docker kill $(docker compose ps -q worker | sed -n '{{I}}p')      # node failure
scale N:       docker compose up -d --no-recreate --scale worker={{N}}           # elastic up/down
audit RUN:     uv run python integration_tests/cluster/check_audit.py /shared/audit/{{RUN}}         # via docker cp or volume mount
```

OrbStack notes: images build arm64-native (no emulation); give the OrbStack VM at least 6 GB memory in its settings for 4 workers; the Ray dashboard is at `http://localhost:8265`; `docker kill` (SIGKILL) is the right primitive for "node died" (Ray sees the raylet disappear), while `docker stop` gives a graceful drain that looks more like a preemption notice. The shared volume can be inspected from the host with `docker compose cp head:/shared/audit ./audit` or by bind-mounting a host directory instead of a named volume (slower on macOS but convenient).

## 10. Verification scenarios

Each scenario runs the toy workload with a distinct `run_name` and then asserts on the audit logs plus the ledger of the final checkpoint. `integration_tests/cluster/check_audit.py` implements the checks.

| # | Scenario | Drive | Assertions |
|---|---|---|---|
| S1 | Happy path | `up 2; blocks; train` | Per (epoch, chunk): the multiset of consumed `block_id`s equals the plan prefix of length `steps*n`; per step, ranks consumed the plan positions `k*n + r`; every rank has the same `report` count (metrics count in `Result`). |
| S2 | Worker kill mid-chunk | `train` in background; after ~N steps `kill-worker 2` (with `max_failures>=1`) | Training finishes. Attempt 2's first consumed positions equal `ledger.cursor * n` of the last checkpoint; the union of blocks over attempts equals the plan; replayed blocks are exactly those with position ≥ last checkpoint cursor·n in attempt 1. Replay count ≤ `every_k * n`. |
| S3 | Elastic scale up | `up 2`, `train` with `num_workers=[2,4]`; after a few steps `scale 4` | Within `elastic_resize_monitor_interval_s`, a new attempt starts with `world_size=4`; the plan tail is re-dealt (positions `k*4 + r`); no block is lost; total consumed set equals plan. |
| S4 | Elastic scale down | Start with 4 workers, `kill-worker` one, `min_workers=2` | Attempt continues with 3 (no full stall); assertions as S3 with `n=3`. |
| S5 | Checkpoint cadence | Run with `every_k=1`, `every_k=8`, `chunk_end` | Number of checkpoints in the run dir matches expectation; `ledger.cursor` of each checkpoint is a multiple of `k` (or equals chunk length). |
| S6 | Chunk hook / re-mining | Enable `remine` | `epoch0/chunk{c+1}` index exists before chunk `c+1` starts on any rank; block contents differ from `base`; audit shows the new block ids consumed. |
| S7 | Determinism | Two runs with the same seed, no failures | Identical audit sequences per rank. |
| S8 | Time-budget policy | `time_budget_s=5` | All ranks report the same number of checkpoints (consensus via broadcast). |

Exit criteria for v0.1: S1–S7 and S9–S10 (section 6.4) green on a 2–4 container cluster under OrbStack, against MinIO.

## 11. Phase 2: k3s / KubeRay on OrbStack

OrbStack ships a built-in Kubernetes; enable it, then `helm install kuberay-operator kuberay/kuberay-operator`. Define a `RayCluster` with a head pod and a worker group (`minReplicas`/`maxReplicas` matching `num_workers=(min,max)`) and a `RayJob` that runs `train.py`. Node failure = `kubectl delete pod <worker>`; elasticity = `kubectl scale`/edit `replicas` (the KubeRay autoscaler can also react to Train's elastic requests). Shared storage = a `hostPath` or `local-path` PVC mounted at `/shared` in all pods. The same `check_audit.py` applies. This phase mostly validates that nothing in v0.1 assumes docker compose networking.

## 12. Milestones (each is one Gest development iteration, see section 14)

0. **M0 — repository bootstrap**: install `agent_gest_git_skills`, run `gest_git_installer` and `gsu` (python-uv profile), fill `AGENTS.md`, create the `Justfile` command contract, register this spec as the Gest spec artifact, `gpl` the plan below.
1. **M1 — core library + unit tests**: block/store/planner/ledger/policy/loader; `Plan.lane` and resume arithmetic property-tested (e.g. Hypothesis) including world-size changes. Test strategy: test-first.
2. **M2 — DistTrainer + toy workload single-node** (`ray.init()` local, `num_workers=2`): S1, S5, S7. Test strategy: test-after with `just smoke` as the gate.
3. **M3 — compose harness**: Dockerfile, compose, Justfile targets, `check_audit.py`; S2, S3, S4, S9, S10. Test strategy: characterization-first (record the audit logs of a green run, then assert).
4. **M4 — chunk hooks**: `remine.py`, S6; time-budget policy, S8.
5. **M5 — KubeRay variant** (phase 2).

## 13. Open questions (decide at M1/M2)

- Whether `report` on every step is acceptable overhead at very small blocks, or whether to batch metrics and call `report` only at policy points *and* guarantee equal counts by making the policy purely index-based (dropping `TimeBudget`).
- Audit log location under heavy step rates: per-rank JSONL on the shared volume is fine for tests; production would want it optional.

## 14. Development workflow requirement: `agent_gest_git_skills`

All development of distrainer is done through Rahul's `rahuldave/agent_gest_git_skills` bundle (GitHub, or the local clone under `~/Projects/agent_gest_git_skills`). The repo is Gest-tracked; substantial work goes through the `g*` skills rather than ad-hoc edits, and an agent that skips Gest for a coding/debugging/docs/verification request must say why.

### 14.1 Bootstrap (M0)

Prerequisites on the Mac: `git`, `gest`, `just`, `uv` (required); `gh`, `but` (GitButler), `ast-grep`, `direnv`, `cx`, `rsync` (optional, unlock GitHub/stacked-PR/dependency/incremental-build flows).

From inside `~/Projects/distrainer`:

1. Install the skills — either `npx skills add rahuldave/agent_gest_git_skills -a codex --skill '*' -y` (add the Claude adapter as well so `/gtw` works from Claude Code), or from the local clone `~/Projects/agent_gest_git_skills/scripts/install.sh ~/Projects/distrainer`. The source-checkout installer also copies `.claude/`/`.codex/` hooks and `AGENTS.template.md → AGENTS.md`.
2. If installed via `npx`, ask the agent to run `gest_git_installer` (hooks/settings + AGENTS guidance; requires approval).
3. Run `gsu` with the **python-uv** language profile: `pyproject.toml` managed by `uv`, `.gitignore` from `base + python-uv` templates, `.envrc`/`.env.example` (S3 credentials for the harness live here, never committed), and the `Justfile` command contract below. Copy `docs/gest_codex_workflow.md` from the skills repo into `docs/` (the installer does not populate `docs/`).
4. Fill the `AGENTS.md` project section: project name `distrainer`; main source directory `distrainer/`; primary docs/specs `docs/distrainer-spec.md`, `docs/distrainer-design.md`, `docs/ray-sub-epoch-training-report.md`; verification commands = the `Justfile` targets; GitHub policy = PRs reviewed with `gpa`, merge only on explicit approval.
5. Register this spec as the Gest spec artifact (`gsp` — update rather than redraft), create the depth-1 outline parent with `gis`, and decompose milestones M1–M5 into iterations and leaf tasks with `gpl`. Seed the tag vocabulary: `data`, `checkpoint`, `elastic`, `policy`, `harness`, `hooks`, `docs`, `k8s`.

### 14.2 Command contract (`Justfile`, python-uv profile)

Native recipe dependencies, not recursive `just` calls; harness targets take positional arguments.

```just
export UV_CACHE_DIR := ".local/uv-cache"

setup:            uv sync
fmt path=".":     uv run ruff format {{path}}
lint path=".":    uv run ruff check {{path}}
typecheck:        uv run ty check
static:           uv run python -m compileall distrainer examples
test target="tests":  uv run python -m pytest {{target}}
regression:       uv run python -m pytest regression_tests
smoke:            uv run python examples/toy_contrastive/train.py --config examples/toy_contrastive/local.yaml   # single-node ray.init(), 2 workers, 1 chunk
diff-check:       git diff --check
verify: lint typecheck static test regression smoke diff-check

# harness (section 9); positional args
up N="2":         docker compose -f deploy/docker-compose.yml up -d --build --scale worker={{N}}
down:             docker compose -f deploy/docker-compose.yml down -v
mkbucket:         docker compose -f deploy/docker-compose.yml exec minio mc mb -p local/distrainer
blocks:           docker compose -f deploy/docker-compose.yml exec head python examples/toy_contrastive/make_blocks.py
train CFG:        docker compose -f deploy/docker-compose.yml exec head python examples/toy_contrastive/train.py --config {{CFG}}
kill-worker I:    docker kill $(docker compose -f deploy/docker-compose.yml ps -q worker | sed -n '{{I}}p')
scale N:          docker compose -f deploy/docker-compose.yml up -d --no-recreate --scale worker={{N}}
integration S="all":  uv run python integration_tests/cluster/run_scenarios.py --scenario {{S}}
docs:             uv run python tools/check_docs.py

# agent context targets from templates/just/agent-contract.just (agent-contract, agent-test-plan, agent-review-plan)
```

`smoke` is the fast gate run by `verify`; the compose scenarios run via `just integration` and are not part of `verify` because they need OrbStack up. `cx` is optional and, if adopted, wraps only file-producing stages such as `make_blocks.py` (blocks are durable outputs of an explicit input), never tests or lint.

### 14.3 Workflow rules for this project

- Route substantial work through `gtw`; it decides session vs. development shape, spec need, tags, branch model, test strategy, and review depth. Milestones M1–M5 are development iterations; small fixes are session tasks.
- One leaf task at a time via `gim`; `gfm` (ruff/ty/compileall/diff-check), `gte` (pytest, smoke, integration scenarios), `gdo` (this spec and `docs/`), and `grv` review before completing any leaf that touches callable code. Missing focused tests for changed callables are review findings.
- For code-facing changes, classify against the tag vocabulary and use `ast-grep` on changed contracts (e.g. `Plan.lane`, `Ledger.resume_position`, `CheckpointIO`) to expand tasks to their dependers; record `Tag classification:` and `Dependency impact:` in completion notes.
- Branches: `gest/<task-id>-summary` for development work, `session/<task-id>-summary` for session work; ordinary git for simple PRs. GitButler (`but`) only for stacked dependent PRs (e.g. M1 store → planner → ledger as a stack); physical git worktrees for independent parallel slices (e.g. M3 harness vs. M4 hooks). Never parallel write agents in one GitButler workspace.
- Commit at verified durable checkpoints with `gcm` (each milestone slice, every harness/config/persistence change, every publishable doc change); push with an upstream; open/update the PR and run `gpa`; report findings and ask before merging. No Gest IDs in commit messages.
- `gpr` decision is mandatory for every depth-1 parent and iteration: promote to a GitHub issue in `rahuldave/distrainer` (store `github.issue`/`github.url`) or record why not.
- At every durable checkpoint regenerate the Gest graphs with `tools/gest_mermaid_graph.py` (overall + latest iteration) and report graph paths, commit hash, push status, review status, and the issue decision.
- Agentic Just targets, `AGENT_TASK v1` / `AGENT_RESULT v1` / `AGENT_TASK_DRAFT v1` packets follow the bundle's protocol rules; `gor` runs phased iterations and decides per phase between sequential work and parallel worktrees/subagents.

Acceptance for M0: `just verify` green on an empty skeleton, `AGENTS.md` filled, Gest DB (`.gest/gest.db`) containing the spec artifact, outline parent, and M1 leaf tasks, and the first `gpr` decision recorded.
