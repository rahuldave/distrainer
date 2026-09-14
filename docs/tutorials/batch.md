# Tutorial 1: batch training on a block log

This walks through `examples/hello_blocks` end to end on a laptop: build a block log from a
finished corpus, train on it with `DistTrainer`, read the checkpoints and the ledger, resume a run
from a checkpoint, and change the number of workers on the way. Everything here runs with
`just smoke` in under a minute; the streaming version of the same ideas is
[Tutorial 2](streaming.md).

Prerequisites: `just setup` (uv, Python 3.13, CPU torch, Ray 2.58). Paths in this tutorial are
relative to the repository root; `runs/` and `blocks/` are gitignored output directories.

## 1. The unit of everything: a block

A **block** is one batch, pre-built and stored as one Parquet file. distrainer never looks at
the columns; your training step does. `examples/hello_blocks/make_blocks.py` draws
`rows_per_block` rows of `y = x . w + noise` per block and writes each with `write_block`:

```python
from distrainer.block import write_block

ref = write_block(fs, root, "b00017", table)  # -> BlockRef(block_id, locator, num_rows, meta)
```

`fs` and `root` come from the config (`cfg.store_fs()`): a `pyarrow` filesystem and a root
path, the same pair for a local directory and for an S3 bucket. The file lands at
`<root>/blocks/b00017.parquet` and every row carries a `block_id` column.

## 2. The log: segments of `W` blocks

Blocks are trained in the order a **log** gives them. The log is a directory of small JSON
files, one per **segment** of exactly `W` blocks, plus an `_END` marker once the producer is
finished:

```
blocks/hello/log/_meta.json      # W and the seed
blocks/hello/log/00000000.json   # segment 0: W block refs in their final (shuffled) order
blocks/hello/log/00000001.json
...
blocks/hello/log/_END
```

Segment `s` covers global **positions** `[s*W, (s+1)*W)`; a position is one block for one rank
at one step. `W` is fixed per log and must be a multiple of every number of workers the run may
use, so that every segment splits into whole steps.

In batch mode the whole corpus exists up front and `BatchWriter` cuts it into segments, one
pass per epoch:

```python
from distrainer.log import BlockLog
from distrainer.writer import BatchWriter

log = BlockLog.create(fs, root, W=cfg.log.W, seed=cfg.seed)
BatchWriter(
    log, refs, passes=cfg.log.passes, tail="error"
).run()  # appends every segment, then _END
```

Each pass permutes the *whole* block list with `Random(hash((seed, pass)))` before cutting, so
segment membership changes from pass to pass; `BlockLog.append` then shuffles within each
segment. `tail` says what happens when the corpus is not a multiple of `W`: `error`, `drop`, or
`wrap` (fill the last segment from the start of the same pass).

Build the log and look at it:

```bash
uv run python examples/hello_blocks/make_blocks.py --config examples/hello_blocks/local.yaml
uv run distrainer log-ls blocks/hello -v
```

```
log: /.../blocks/hello/log
W=12 seed=7 schema_version=1 by distrainer
segments: 4 (0..3)
ended: True
  00000000 pass=0 positions 0..11: b00019, b00009, b00037, b00024, ...
  00000001 pass=0 positions 12..23: b00031, b00047, b00016, b00021, ...
  ...
```

`local.yaml` asks for 48 blocks, `W=12`, one pass: four segments.

## 3. Two functions and a config

Your training code is two functions (`examples/hello_blocks/train.py`):

```python
def build_model(info: TrainInfo) -> tuple[torch.nn.Module, torch.optim.Optimizer]:
    torch.manual_seed(info.config.seed)
    model = torch.nn.Linear(int(info.train["features"]), 1)
    return model, torch.optim.SGD(model.parameters(), lr=float(info.train["lr"]))


def train_step(model, optimizer, table: pa.Table, info: TrainInfo) -> dict[str, float]:
    x, y = ...  # straight out of the Arrow table
    optimizer.zero_grad()
    loss = torch.nn.functional.mse_loss(model(x), y)
    loss.backward()  # DDP all-reduces the gradients here
    optimizer.step()
    return {"loss": float(loss.item())}
```

`TrainInfo` tells a step where it is: `rank`, `world_size`, `position`, `segment`,
`step_in_segment`, `pass_idx`, `attempt`, the `device`, and the whole config (`info.train` is
the free-form `train:` section). Then:

```python
cfg = load_config("examples/hello_blocks/local.yaml")
init_ray(cfg)  # local cluster, or ray_address: auto inside one
result = DistTrainer(train_step, build_model, cfg).fit()
```

The config (spec section 7) in the parts that matter for batch training:

```yaml
run_name: hello
storage_path: runs/hello        # checkpoints and Ray Train run state
store_root: blocks/hello        # blocks, log, audit trail
log:
  W: 12
  passes: 1
checkpoint:
  policy: any                   # every_k | segment_end | pass_end | time | any | never
  every_k: 2
  num_to_keep: 3
scaling:
  num_workers: 2                # or [min, max] for an elastic run
```

## 4. Run it

```bash
just smoke        # = uv run python examples/hello_blocks/train.py --config examples/hello_blocks/local.yaml
```

Every rank runs the same loop: open the log, deal the positions of each segment to the ranks
(position `p` goes to rank `p mod n` at step `p // n`), call your `train_step` once per block,
and at every step decide whether a checkpoint is due. The tail of the output:

```
final metrics: {'loss': 0.049, 'position': 46, 'segment': 3, 'cursor': 6, 'world_size': 2, 'attempt': 0, 'reports': 12, ...}
final ledger: Ledger(segment=3, cursor=6, world_size=2, pass_idx=0, run_attempt=0)
audit: attempt 0: world_size [2], ranks [0, 1], 48 blocks, segments 0..3, positions 0..47
reports per rank: 12 (expected 12)
S1 PASS
```

Two things to read there. The **ledger** is the whole progress state: segment 3, 6 steps done at
world size 2, so positions `[36, 36 + 6*2) = [36, 48)` of segment 3 are complete, which is the
end of the log. And `reports per rank: 12`: with `every_k: 2` and six steps per segment the
`any` policy checkpoints at steps 2, 4 and 6 of every segment, three per segment, twelve in the
run, and every rank called `ray.train.report` exactly that many times (`report` is a barrier
in Ray Train, so the counts must agree; the loop reports only where a checkpoint is taken and
averages the metrics in between, see spec section 5).

`S1 PASS` is the happy-path check on the **audit trail**: every rank appends one JSON line per
consumed block to `blocks/hello/audit/hello/<attempt>-<rank>.jsonl`, and `check_s1` verifies
that every segment's positions were consumed exactly once, dealt by the rule above, with equal
counts per rank. The verification scenarios in `docs/examples-and-scenarios.md` are all built
on this trail.

## 5. Checkpoints and the ledger

`num_to_keep: 3` leaves the last three checkpoints:

```bash
ls runs/hello/hello/
```

```
checkpoint_g000003_p000004_n02_a00
checkpoint_g000003_p000008_n02_a00
checkpoint_g000003_p000012_n02_a00
checkpoint_manager_snapshot.json
...
```

The name says where a checkpoint stands: segment `g`, `p` positions of it done, at world size
`n`, in attempt `a`. Inside are `model.pt`, `optimizer.pt` and `ledger.json`; the ledger is also
in the checkpoint's metadata, so it can be read without downloading the weights:

```bash
uv run distrainer inspect runs/hello/hello/checkpoint_g000003_p000008_n02_a00
```

```
checkpoint: runs/hello/hello/checkpoint_g000003_p000008_n02_a00
ledger: {'segment': 3, 'cursor': 4, 'world_size': 2, 'pass_idx': 0, 'run_attempt': 0}
done positions in segment 3: 8
written by distrainer 0.0.1
```

A checkpoint is legal at *every* step boundary because the ledger is enough to reconstruct the
data position: `cursor * world_size` positions of `segment` are done. That is what makes
sub-epoch checkpointing exact rather than approximate.

Policies (`checkpoint.policy`): `every_k` (every `k` steps within a segment), `segment_end`,
`pass_end`, `time` (a time budget, decided by rank 0 and broadcast so all ranks agree), `any`
(the union of the configured ones), `never`. `just local-scenarios S5` runs three cadences and
checks the checkpoint directories against them.

## 6. Resume, also with a different number of workers

`distrainer resume` starts a *new* run from a checkpoint (a reused run name would make Ray
Train restore that run's own state instead):

```bash
uv run distrainer resume runs/hello/hello/checkpoint_g000003_p000004_n02_a00 \
    --config examples/hello_blocks/local.yaml \
    --entry examples.hello_blocks.train:entry --run-name hello_resume
```

`--entry` names a function returning `(train_step, build_model)` for the config. The resumed
run reads segment 3, skips the 4 positions the ledger says are done, and deals the rest over the
current world size. Try it with a different `scaling.num_workers` in a copy of the config
(`W=12` allows 1, 2, 3, 4, 6 and 12): the ledger carries the *old* world size, the resume rounds
the done positions down to a multiple of the *new* one and replays the remainder, at most
`n_new - 1` blocks. The audit trail of `hello_resume` shows exactly which positions were trained
again.

The same mechanism handles a worker dying or an elastic resize mid-run: Ray Train restarts the
worker group from its last registered checkpoint, every restart gets a new `attempt` in the
audit trail, and the union of the attempts covers the log with a bounded replay. That needs
more than one machine to demonstrate; `docs/running-modes.md` and `just integration S2` (worker
kill), `S3`/`S4` (scale up and down) do it with containers.

## 7. Passes

Set `log.passes: 2` and rebuild the store (`rm -rf blocks/hello`): the log now has eight
segments, `pass=0` for the first four and `pass=1` for the next, with different segment
membership. Positions keep counting up across passes (segment 4 starts at position 48), so a
checkpoint in the second pass is still just a segment and a cursor. `checkpoint.policy:
pass_end` checkpoints at the last step of each pass; `TrainInfo.pass_idx` tells your step which
pass it is in.

## 8. Where to go next

- `examples/toy_contrastive` (`just contrastive`): the same API with a real workload, blocks
  built with Ray Data, InfoNCE with hard negatives mined into the blocks, two passes.
- [Tutorial 2](streaming.md): logs that are still being written, an external producer, the
  re-mining hook, and retention.
- `docs/examples-and-scenarios.md`: every verification scenario and what it asserts.
- `docs/cli.md`: every command and flag used above.
- `docs/distrainer-spec.md`: the contract.
