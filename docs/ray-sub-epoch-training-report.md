# Distributed training between batch and epoch granularity in Ray and Anyscale (Enterprise Ray)

*Research report for the `distrainer` project — September 9, 2026. Verified against Ray 2.58.0 source (Train v2 is on by default) and current Ray / Anyscale docs.*

## 1. The short version

Ray Train does not have a native notion of "sub-epoch" work units. Its unit is the training loop you write inside a per-worker `train_func`; the framework gives you three primitives to shape granularity: a per-worker **data shard** (`ray.train.get_dataset_shard` → a `DataIterator`), a **synchronization + checkpoint point** (`ray.train.report`, which is a barrier across all workers), and **restart-from-checkpoint** fault tolerance (`FailureConfig` / `ScalingConfig`). Anything between "one batch" and "one epoch" is something you build by choosing how often you call `report` and how you slice the data.

The main thing that separates open-source Ray from Anyscale's runtime for this problem is **what happens to the data position on restart**. In OSS Ray, a restart re-executes the dataset from the top of the epoch; the iterator has no saved position, so a checkpoint taken at step 3,000 of a 10,000-step epoch does not know which rows were already seen. Anyscale's runtime adds **mid-epoch resumption** (beta): `DataIterator.state_dict()` and `get_dataset_shard(name, state_dict=...)`, backed by a row-ID-based data checkpoint. That is exactly the "between batch and epoch" checkpoint semantics you want, and it is the single most relevant enterprise-only feature.

## 2. How shards actually work in open-source Ray Train

### 2.1 The default: `streaming_split`, dynamic and lockstep

By default the Trainer splits every dataset you pass in `datasets={...}` across the `num_workers` training workers using `Dataset.streaming_split(world_size, equal=True, locality_hints=worker_node_ids)`. This is done by the built-in `DataConfig.configure()`; you can pass `DataConfig(datasets_to_split=["train"])` to leave e.g. the validation set unsplit (every worker gets the full thing), and `enable_shard_locality=False` to drop the locality hints.

The important facts about a streaming split shard, from the `SplitCoordinator` / `OutputSplitter` implementation:

- **A shard is not a fixed subset of rows.** A single `SplitCoordinator` actor (pinned to the node that created the split, i.e. the Train controller) runs the dataset's streaming executor and hands out *blocks* to whichever split index asks for the next one. An `OutputSplitter` operator tags each block with an `output_split_idx`, balancing rows across the `n` outputs and preferring the node named in `locality_hints`. So "worker 3's shard" means "the blocks worker 3 happened to pull this epoch", not "rows 30–40% of the dataset".
- **`equal=True` guarantees equal row counts, by truncation.** The splitter keeps an internal buffer large enough to equalize the tail; leftover rows are dropped. With `equal=False` nothing is dropped but shards can be uneven. Train uses `equal=True` so DDP workers take the same number of steps.
- **Order is not guaranteed** unless `DataContext.get_current().execution_options.preserve_order = True`. With `preserve_order`, the splitter disables the locality hints and the buffering so that block → split assignment is deterministic — that is the only way to get a reproducible shard assignment across runs from the same seed.
- **Every re-iteration is an epoch, and epochs are barriers.** Calling `iter_batches()` / `iter_torch_batches()` on the shard again calls `start_epoch` on the coordinator; the coordinator waits until *all n splits* have arrived, then force-shuts the previous executor and **re-executes the whole dataset pipeline** from scratch. Any `get` with a stale epoch id raises `Invalid iterator: the dataset has moved on to another epoch`. "If one iterator falls behind, other iterators may be stalled" — the splits are consumed in lockstep with bounded prefetch.
- **The pipeline re-runs each epoch**, so the docs tell you to `materialize()` the expensive, deterministic preprocessing and put per-epoch work (`random_shuffle(seed=)`, `randomize_block_order(seed=)`) after the materialize.

### 2.2 The per-worker iterator

`DataIterator.iter_batches(prefetch_batches=1, batch_size=256, batch_format, drop_last=False, local_shuffle_buffer_size=None, local_shuffle_seed=None)` and `iter_torch_batches(..., dtypes, device="auto", collate_fn, pin_memory)` are the consumption API. `prefetch_batches` runs a background thread pool to fetch/format the next N batches; `local_shuffle_buffer_size` gives a per-worker in-memory shuffle without a global shuffle.

Two implementation details matter for sub-epoch loops. First, breaking out of a `for batch in it.iter_batches()` loop shuts the streaming executor down cleanly via `GeneratorExit`; but if you hold `it = iter(shard.iter_batches())` yourself and stop early, the executor keeps producing until the reference is dropped (call `it.close()`). Second, the docs recommend `ds.limit(n)` on the dataset rather than an external batch cap (e.g. Lightning's `limit_train_batches`) when you want iteration to end naturally after n rows.

### 2.3 Static shards if you want them

If dynamic block dispatch is a problem (e.g. you need "worker k always sees partition k" so that a mid-epoch position is meaningful), subclass `DataConfig` and override `configure(datasets, world_size, worker_handles, worker_node_ids)` to return `List[Dict[str, DataIterator]]`. Inside it you can materialize and use `Dataset.split(n, equal=True, locality_hints=)`, `split_at_indices([...])`, or `split_proportionately([...])`, then return `.iterator()` on each piece. That gives fixed, addressable shards at the cost of materializing (object-store memory) and losing streaming.

## 3. Checkpoint / sync cadence in open-source Ray Train

`ray.train.report(metrics, checkpoint=None, checkpoint_dir_name=None, checkpoint_upload_mode=SYNC, delete_local_checkpoint_after_upload=None, checkpoint_upload_fn=None, validation=False)`:

- It is a **barrier**: "All workers must call `ray.train.report` the same number of times … This method acts as a barrier across all workers." So whatever cadence you pick (every K batches, every K rows, every epoch) has to be identical on every rank — which is easy when `equal=True` guarantees equal batch counts, and something to watch if you use `equal=False` or `drop_last=False`.
- Only rank 0's metrics are attached to the checkpoint; for DDP only one rank should upload a checkpoint, for FSDP/DeepSpeed each rank can upload its own piece and they are merged into one directory.
- `CheckpointUploadMode.ASYNC` uploads in a background thread (each `report` waits for the previous upload to finish before starting the next); `NO_UPLOAD` lets you upload yourself. Use ASYNC if you checkpoint frequently. Note `CheckpointConfig.checkpoint_frequency` / `checkpoint_at_end` are deprecated in v2 — cadence is entirely in your loop.
- `CheckpointConfig(num_to_keep, checkpoint_score_attribute, checkpoint_score_order)` controls retention; if you score checkpoints, the metric must be reported with every checkpoint.
- `ray.train.collective.barrier()` and `broadcast_from_rank_zero()` exist for extra synchronization points that don't involve a checkpoint.

There is no "checkpoint every N steps" knob — you write `if step % K == 0: report(...)`. That is the whole sub-epoch story in OSS as far as cadence goes.

## 4. What happens on failure or resize (open-source)

`FailureConfig(max_failures=0, controller_failure_limit=-1, max_preemption_failures=-1)`: worker errors retry up to `max_failures` times, node preemptions are counted separately (`max_preemption_failures`, default unlimited), and the controller itself can be retried. Ray Train v2 distinguishes worker-process, worker-node and job-driver fault tolerance.

Recovery is a **full restart of the worker group**: the controller shuts down all workers, starts a new group (possibly with a different size), calls your `train_func` again, and `ray.train.get_checkpoint()` returns the latest reported checkpoint. Crucially for `distrainer`:

- The `DatasetManager` creates a fresh set of `DataIterator`s for the new worker group. The old `SplitCoordinator` and its epoch state are gone. Iterating the shard starts at epoch 0 of a re-executed pipeline. **OSS has no iterator `state_dict`** — `ray.train.get_dataset_shard(dataset_name)` takes only the name.
- If you checkpointed mid-epoch, your only OSS options are: replay the partial epoch (accept that some rows are seen twice), skip ahead yourself (count batches consumed per rank, then `itertools.islice` past them — which only reproduces the same rows if `preserve_order=True` plus seeded shuffles, and still streams and discards the skipped blocks), or design your epochs to be small enough that replay is cheap (i.e. make the "sub-epoch" the real epoch: iterate over `ds.limit(n)` / a pre-split chunk).
- The controller has a torchft "replica groups" path that replaces only failing replica groups without a full restart (`_replace_bad_workers`), but it is only used when the world size doesn't change and replica groups are managed — an emerging feature, not the default.

**Elastic training is in OSS now.** `ScalingConfig(num_workers=(min_workers, max_workers), elastic_resize_monitor_interval_s=60.0)` enables an `ElasticScalingPolicy`: the run starts once `min_workers` are available, and while healthy the controller considers resizing every `elastic_resize_monitor_interval_s`. A resize is executed exactly like a failure recovery — full restart from the latest checkpoint with the new world size — so the data is re-split N-ways with the new N. Any progress counter that is "steps on this rank" becomes meaningless across a resize; count rows/samples globally instead, and keep the effective batch size or LR schedule world-size-aware. (Anyscale's docs still list elastic training as a runtime feature, but the 2.58 OSS `ScalingConfig` accepts the tuple and ships the policy.)

## 5. What Anyscale (Enterprise Ray / Anyscale Runtime, formerly RayTurbo) adds

### 5.1 Mid-epoch training resumption (beta) — the one that matters here

"Resume training from the exact point where it stopped without repeating or skipping data … each row exactly once per epoch, even when failures, preemptions, or manual stops interrupt training."

APIs:

```python
ray.train.get_dataset_shard(dataset_name: str, state_dict: Optional[Dict] = None) -> DataIterator
DataIterator.state_dict() -> Dict          # must be called on every rank at the same time

DatasetCheckpointConfig(
    id_column: str,
    generate_id_column: bool = False,      # auto-generate IDs (Parquet)
    checkpoint_path: Optional[str] = None,
    override_filesystem: Optional[pyarrow.fs.FileSystem] = None,
    delete_checkpoints_after_epoch: bool = True,
)
```

Pattern: at each mid-epoch checkpoint, save `ds_shard.state_dict()` next to the model weights inside the `ray.train.Checkpoint`; on restart, load it and pass `get_dataset_shard("train", state_dict=data_state_dict)`, which "skips forward to the saved position".

Requirements and limits: every row needs a unique ID (`id_column`, or `generate_id_column=True`); the pipeline may contain only map-style operators (`map`, `map_batches`, `filter`) and the ID must survive all of them (no shuffles/joins/aggregations); shared storage for the data-checkpoint metadata; fault tolerance must be on (`FailureConfig(max_failures > 0)`); all ranks call `state_dict()` together. It is an Anyscale Runtime feature — the OSS 2.58 `get_dataset_shard` signature has no `state_dict` and OSS `DataIterator` has no `state_dict()`.

How it works underneath: it is Ray Data's **job-level checkpointing** (row-ID tracking of completed rows, filtered out on re-execution) wired into the Train iterator. The OSS tree does contain `ray.data.checkpoint.CheckpointConfig(id_column, checkpoint_path, delete_checkpoint_on_success, ...)` (beta) and `DataContext.checkpoint_config`, but in OSS that is documented for read → map → write batch pipelines, not for training ingestion; Anyscale's version (`ray.anyscale.data.checkpoint.CheckpointConfig`) adds `generated_id_column` and the Train hook. So if you want row-exact resumption in OSS, you would be re-implementing this: tag rows with IDs, log consumed IDs per rank at each `report`, and `filter` them out on resume.

### 5.2 Other Anyscale-only or Anyscale-first training features

- **Spot / preemption recovery** marketed as "minimal interruption from spot instance preemption and node failure" (OSS has `max_preemption_failures`; Anyscale layers cluster-level preemption handling and node replacement on top).
- **Ray Train dashboard**: per-worker logs and metrics, progress visualization, error attribution, fault-tolerance events, integrated CPU/GPU profiling; logs persisted 30 days.
- **Job scheduling, job queues, priority scheduling, alerting, fractional heterogeneous resource allocation** — platform features from the comparison matrix, not training-loop semantics.
- **Ray Data runtime**: actor-pool autoscaler that starts a job before the full cluster is up, proactive detection of hanging operators / high memory, vectorized ops and faster sources/sinks. These affect ingest throughput, not shard semantics.

## 6. Option matrix for `distrainer` (work units between batch and epoch)

| Approach | How | Shard behaviour on restart / resize | Available in |
|---|---|---|---|
| **Step-cadence checkpoints** (`report` every K batches) | `for i, batch in enumerate(shard.iter_torch_batches()): … if i % K == 0: report(...)`; `ASYNC` upload | Model state restored; data position lost → replay from start of epoch or manual `islice` skip (exact only with `preserve_order` + seeds) | OSS |
| **Sub-epoch datasets** (make the chunk the epoch) | Materialize, `split_at_indices` / `limit` into M chunks per epoch; iterate chunk j as its own "epoch" and checkpoint `(epoch, j)` | Resume at chunk j: replay is bounded to one chunk; each chunk is re-streaming_split across current workers so resize is free | OSS |
| **Static per-worker partitions** | Custom `DataConfig.configure` returning `.iterator()` of fixed splits | Positions are meaningful per worker, but a resize changes the partitioning — you must re-partition and re-map progress; materialization cost | OSS |
| **Row-exact mid-epoch resume** | `DatasetCheckpointConfig` + `shard.state_dict()` + `get_dataset_shard(name, state_dict=)` | Exactly-once per epoch; data pipeline limited to map-style ops with a persistent ID column | Anyscale Runtime (beta) |
| **Elastic world size** | `ScalingConfig(num_workers=(min,max))` | Full restart from checkpoint with new N; streaming split re-divides automatically; combine with any of the above | OSS 2.58 (also Anyscale) |

Design notes that fall out of the mechanics above: keep your progress counter in global samples/rows rather than per-rank steps so it survives resizes; make every rank hit `report` the same number of times (rely on `equal=True`, or use `drop_last=True`); put deterministic preprocessing before `materialize()` and per-epoch shuffles after it; and if row-exact resumption without Anyscale is a requirement, the "sub-epoch datasets" row is the cheapest OSS approximation — it converts the missing iterator state into a chunk index, which the Train checkpoint can carry.

## Sources

- [ray.train.get_dataset_shard — Ray docs](https://docs.ray.io/en/latest/train/api/doc/ray.train.get_dataset_shard.html)
- [ray.data.Dataset.streaming_split — Ray docs](https://docs.ray.io/en/latest/data/api/doc/ray.data.Dataset.streaming_split.html)
- [Data Loading and Preprocessing — Ray Train user guide](https://docs.ray.io/en/latest/train/user-guides/data-loading-preprocessing.html)
- [Saving and Loading Checkpoints — Ray Train user guide](https://docs.ray.io/en/latest/train/user-guides/checkpoints.html)
- [Handling Failures and Node Preemption — Ray Train user guide](https://docs.ray.io/en/latest/train/user-guides/fault-tolerance.html)
- [ray.train.ScalingConfig — Ray docs](https://docs.ray.io/en/latest/train/api/doc/ray.train.ScalingConfig.html)
- [ray.train.DataConfig — Ray docs](https://docs.ray.io/en/latest/train/api/doc/ray.train.DataConfig.html)
- [Iterating over Data — Ray Data user guide](https://docs.ray.io/en/latest/data/iterating-over-data.html)
- [Ray Train V2: Unified Distributed Training on Ray — Anyscale blog](https://www.anyscale.com/blog/ray-train-v2-unified-distributed-training-on-ray)
- [Mid-epoch training resumption — Anyscale docs](https://docs.anyscale.com/runtime/mid-epoch-resumption)
- [Ray Train in the Anyscale Runtime — Anyscale docs](https://docs.anyscale.com/runtime/train)
- [Ray Data in the Anyscale Runtime — Anyscale docs](https://docs.anyscale.com/runtime/data)
- [RayTurbo Train — Anyscale docs](https://docs.anyscale.com/rayturbo/rayturbo-train)
- [Ray Train with Anyscale — product page](https://www.anyscale.com/product/library/ray-train)
- [Ray Train & Ray Data Dashboards on Anyscale — Anyscale blog](https://www.anyscale.com/blog/ray-train-data-dashboard)
- [Ray Data checkpoint — GitHub issue #49438](https://github.com/ray-project/ray/issues/49438)
- [torchft — GitHub](https://github.com/meta-pytorch/torchft)
- Ray 2.58.0 source: `ray/train/_internal/data_config.py`, `ray/data/_internal/iterator/stream_split_iterator.py`, `ray/data/_internal/execution/operators/output_splitter.py`, `ray/train/v2/_internal/execution/controller/controller.py`, `ray/train/v2/_internal/execution/scaling_policy/elastic.py`, `ray/data/checkpoint/interfaces.py`

## 7. Addendum: controlling batch composition (e.g. hard negatives for contrastive training)

Ray Data has no sampler; `iter_batches(batch_size=B)` slices B contiguous rows off the stream of blocks the shard receives, `collate_fn` only sees one batch, and `local_shuffle_buffer_size` mixes rows across blocks. Batch membership is therefore controlled by controlling block membership and block order:

- `iter_batches(batch_size=None)` / `iter_torch_batches(batch_size=None)` yields whole blocks as batches. Build blocks that *are* designed batches: assign a `batch_id` per row in a mining pass, then `groupby("batch_id").map_groups(fn)` (one block per group), or pack batches in `map_batches(pack_fn, batch_size=B*k)`. Keep blocks under `DataContext.target_max_block_size`.
- Shuffle at batch level with `randomize_block_order(seed=)` after `materialize()`; do not use `local_shuffle_buffer_size`.
- `streaming_split(equal=True)` dispatches whole bundles to the worker with the fewest rows so far and only slices blocks to equalize the tail. Uniform batch sizes with a batch count divisible by `world_size` means no slicing; otherwise trim upstream (`limit`) or split at batch boundaries in a custom `DataConfig.configure`.
- With `preserve_order=True` (locality hints and splitter buffer disabled) and uniform blocks, dispatch is round-robin, so step k on rank r receives block `k*n + r` — enough to design coherent "super-batches" across ranks for all_gather-style global negatives. Verify with a test before relying on it.

Communication: data moves at block granularity and a block goes to exactly one worker (locality-preferred), so batch composition does not add data movement. The comm that does depend on the contrastive strategy is the loss-side `all_gather` of embeddings for cross-device in-batch negatives (`world_size * B * d` per step, small relative to gradient all-reduce). Local hard negatives per block cost nothing extra; the two can be combined.

Because batches carry a `batch_id`, resumption state becomes "set of completed batch ids" (or a chunk index) instead of a per-rank stream position — the same idea as Anyscale's row-ID mid-epoch resumption, reproducible in OSS with a `filter` on resume.

Hard-negative re-mining every K steps is a natural sub-epoch unit. Plumbing options: (a) chunk-as-epoch, with a `map_batches` stage that reads the latest mined negatives at execution time (the pipeline re-executes each epoch, so it refreshes automatically); (b) `TorchTrainer(datasets={"train": callable})` plus `resume_from_checkpoint=` in a driver loop of mine → fit(chunk) → checkpoint (pays worker-group startup per chunk); (c) rank 0 builds the chunk dataset inside `train_func`, calls `streaming_split(world_size, equal=True, locality_hints=...)`, and distributes the (picklable) iterators via `ray.train.collective.broadcast_from_rank_zero`.

Alternative: run a plain PyTorch `DataLoader` per worker (`ray.train.torch.prepare_data_loader`) with a custom `BatchSampler` over a materialized index; sharding becomes "rank r takes every n-th designed batch" and resume state is the batch index. Loses streaming and locality-aware block movement; a hybrid (Ray Data for embedding/mining passes, torch samplers for the loop) is reasonable.
