# Examples and verification scenarios

Two example workloads exercise distrainer, and eleven scenarios (spec section 10) verify its
guarantees by reading audit trails and checkpoint metadata, never the cluster. This page says what
each example computes, which configs drive it, how every scenario is run and checked, and where
its artifacts end up. `docs/running-modes.md` covers the environments the scenarios run in.

## The examples

### hello_blocks (`examples/hello_blocks/`, the `just smoke` gate)

The smallest complete use of the API. `make_blocks.py` draws `n_blocks` blocks of `rows_per_block`
rows with `features` Gaussian inputs and `y = x . w + noise`, writes each block as a Parquet file,
and runs `BatchWriter` for `log.passes` passes with `log.W` blocks per segment. `train.py` trains
`torch.nn.Linear(features, 1)` with SGD and MSE; `build_model` seeds torch from `seed`, and
`train_step` reads the feature columns straight out of the Arrow table. It is deliberately tiny so
that a run measures the framework, not the model.

| config | where it runs | shape |
|---|---|---|
| `local.yaml` | laptop, local Ray, 2 workers | 48 blocks of 32 rows, `W=12` (4 segments), `every_k=2`, about 30 s |
| `harness.yaml` | container cluster, 2 to 3 workers, shared mount | 240 blocks, `W=24` (10 segments), `step_sleep_s: 0.25` so a run lasts about 30 s |
| `harness-minio.yaml` | container cluster, everything on MinIO | same shape, `storage.kind: s3` |

Knobs in the `train:` section: `n_blocks`, `rows_per_block`, `features`, `lr`, and `step_sleep_s`
(a sleep per step so failure injection lands mid-run; 0 on the laptop). Any config value can be
overridden on the command line with `--set key.path=value` (values are YAML), which is how the
scenario runners reuse one file:

```bash
uv run python examples/hello_blocks/train.py --config examples/hello_blocks/local.yaml \
    --set run_name=s5 --set checkpoint.policy=every_k --set checkpoint.every_k=4 \
    --set checkpoint.num_to_keep=null
```

`train.py` wipes the previous run of the same name (local storage only), builds the blocks if the
store is empty, trains, prints the final ledger, and applies the S1 checks plus the report-count
check unless `--no-check` is given. `examples.hello_blocks.train:entry` is the `--entry` for
`distrainer resume`.

### toy_contrastive (`examples/toy_contrastive/`, `just contrastive`)

The spec's section 8 workload. `make_blocks.py` draws `n_items` points in `features` dimensions
from `clusters` Gaussian clusters, picks for every item a positive (another item of its cluster)
and `hard_negatives` negatives (items from the nearest other clusters by centroid distance),
assigns items to blocks of `anchors_per_block` at random, and builds the block files with Ray Data
(`groupby("batch_id").map_groups`, one Parquet file per group). `model.py` is a two-layer MLP
encoder with L2-normalised outputs and an InfoNCE loss over the positive, the in-block hard
negatives, and the other anchors' positives; with `train.all_gather: true` the positives of every
rank are gathered so each anchor also sees the other ranks' candidates (the loss-side collective).

`local.yaml`: 7680 items, 64 clusters, 32 anchors per block with 4 hard negatives (240 blocks),
`W=24`, 2 passes, 2 workers, `every_k=4`: 480 steps, 60 checkpoints, about a minute, and the loss
falls from 0.87 to 0.54. The mining step re-runs with the current model in M4 (`remine.py`), which
turns this example into the streaming producer of S6.

### The CLI on top of them

```bash
uv run distrainer log-ls blocks/hello -v                     # meta, segments, _END
uv run distrainer inspect runs/hello/hello/<checkpoint dir>  # ledger from the metadata only
uv run distrainer resume <checkpoint dir or s3://...> --config examples/hello_blocks/local.yaml \
    --entry examples.hello_blocks.train:entry --run-name hello_resume
uv run distrainer export <checkpoint dir> /tmp/ckpt
uv run distrainer gc blocks/hello --keep-from 2
```

## What the checks read

Every rank appends one JSON line per consumed block to
`<store_root>/audit/<run_name>/<attempt>-<rank>.jsonl` (`.<part>.jsonl` parts on object stores):
`attempt`, `rank`, `world_size`, `segment`, `step`, `position`, `block_id`, `ts`. Ordering within a
rank is file order, never `ts`. Checkpoint directories are named
`checkpoint_g{segment}_p{positions}_n{world_size}_a{attempt}` and carry the ledger in
`.metadata.json`, so a checker can tell where a checkpoint stands without opening the weights.

The checks live in `integration_tests/cluster/check_audit.py` and are pure functions returning a
list of problems (empty means pass):

| check | what it asserts |
|---|---|
| `check_dealing` | every record sits at `segment*W + step*world_size + rank` |
| `check_s1` | one attempt; each segment's positions are exactly its `W` positions, once; dealing; equal per-rank counts |
| `check_report_count` / `expected_reports` | rank 0's final `reports` metric equals the number of policy points (plus one final report when the last step is not a checkpoint) |
| `check_s5` / `expected_checkpoints` | every checkpoint's cursor is a multiple of `k` or the segment end; directory name matches its ledger; count matches the cadence |
| `check_s7` | two trails have identical per-rank `(segment, step, position, block_id)` sequences |
| `check_recovery` | several attempts: dealing per attempt with its own world size; each new attempt starts on a step boundary of its world size at or before the position after the previous attempt's last one; replay of at most `2*every_k*n_old + n_new` positions; the union covers positions 0 to the end of the last segment with no gap |
| `check_resume` | a run resumed from a ledger covers exactly `start .. end of its last segment`, once, dealt by the rule |

The `2*every_k*n_old` term is the price of ASYNC checkpoint upload: a checkpoint reported just
before a failure can still be in flight, so the controller may resume from the one before it.

## The scenarios

| # | scenario | mode | driven by | injects | asserts | status |
|---|---|---|---|---|---|---|
| S1 | happy path | A (also B) | `just smoke`; `just local-scenarios S1`; `just train` in the harness | nothing | `check_s1`, report count | green |
| S2 | worker kill mid-segment | B | `just integration S2` | `kill-worker 2` after 12 blocks; the container is started again 5 s later | `check_recovery` with world sizes `[2, 2]` | green |
| S3 | elastic scale up | B | `just integration S3` | `scale 3` after 12 blocks | `check_recovery` with `[2, 3]` | green |
| S4 | elastic scale down | B | `just integration S4` | start with 3, `scale 2` after 12 blocks | `check_recovery` with `[3, 2]` | green |
| S5 | checkpoint cadence | A | `just local-scenarios S5` | three runs: `every_k=1`, `every_k=4`, `segment_end`, all with `num_to_keep: null` | `check_s5` with the expected count per cadence | green |
| S6 | segment hook / re-mining | B | M4 | the `remine` hook appends the next segment at each segment end | next segment committed before any rank consumes it; new block ids in the trail | planned (M4) |
| S7 | determinism | A | `just local-scenarios S7` | two stores built from the same seed, two runs | `check_s7` | green |
| S8 | time-budget policy | B | M4 | `time_budget_s` policy | every rank reports the same number of checkpoints | planned (M4) |
| S9 | cold restore | B + MinIO | `just integration S9` | full run on MinIO with all checkpoints kept, `down`, `wipe-shared`, `up`, `distrainer resume` from a mid-run checkpoint URI into a new run | `check_resume` from the checkpoint's position | green |
| S10 | head loss | B + MinIO | `just integration S10` | `docker kill` of the head 25 s into the run, `up` (workers rejoin), resume from the newest registered checkpoint | `check_resume` | green |
| S11 | streaming producer | B | M4 | a separate writer appends segments with pauses while ranks wait | ranks wait rather than fail; positions contiguous; `_END` ends the run; `gc` respects retention | planned (M4) |

Modes: A is the laptop (local Ray), B the OrbStack container cluster; see `docs/running-modes.md`.

### Running them

```bash
just smoke                      # S1 on the laptop, under a minute
just local-scenarios            # S1, S5, S7 on the laptop, about 3 minutes
just build && just up 2         # container cluster (once; the image is rebuilt only if uv.lock changes)
just integration S2             # one scenario; `all` runs S2, S3, S4, S9, S10 (about 12 minutes)
just down                       # containers stop, the MinIO volume stays; `just nuke` removes it
```

Each cluster scenario brings the cluster to the size it needs, waits until Ray reports that many
`trainer` resources, builds the blocks inside the head container if the store is empty, deletes the
run state and audit trail of an earlier run with the same name (on the shared mount, or on the
bucket for S9/S10), starts `train.py` inside the head, waits for the audit trail to show a few
blocks, injects the failure, waits for the run to finish, and applies the check. The driver's
output for the run is saved under `.harness/logs/<scenario>.log`; the audit trail of a
shared-mount scenario is under `.harness/shared/blocks/audit/<scenario>/`, and the checkpoints
under `.harness/shared/runs/<scenario>/`.

### Reading a failure

The runner prints one line per problem and then `S<n>: FAIL`. Three kinds of problem have come up
while building the harness and are worth recognising:

- `expected at least 2 attempts, found [0]`: the failure was injected after the run had finished.
  Lengthen the run (`n_blocks`, `step_sleep_s`) rather than shortening the wait.
- `world sizes per attempt [2, 3, 2], expected [3, 2]`: training started before every worker
  container had joined; the runner now waits for the `trainer` resources first.
- `resume FAIL: no audit records` on S9/S10: the resumed run had nothing left to do, usually
  because it resumed from the end of a completed run, or because a run with the same name already
  existed on the bucket and Ray Train restored it instead of starting afresh.
