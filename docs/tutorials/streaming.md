# Tutorial 2: streaming, hooks and retention

[Tutorial 1](batch.md) trained on a log that was complete before training started. This one
trains on a log that is **still being written**, in the two ways distrainer supports: an external
producer process, and a *segment hook* that lets the trainer itself write the next segment from
the current model. Both reuse the batch loop unchanged; retention (`gc`) keeps a long stream
from growing forever.

## 1. What changes when the log is open

Nothing in the training loop, which is the point. A rank that reaches the end of segment `s`
asks the log for segment `s+1`; if the file is not there yet and there is no `_END`, it polls
(`log.wait_poll_s`) until it appears. The run ends when the ranks reach the end of the last
segment *and* `_END` exists. Two properties make this safe:

- **Segments are complete before they are trained on.** A writer commits a segment only after
  all of its `W` block files exist, with one atomic write of the segment file, and readers only
  trust committed segment files. So the segment being trained on is immutable, the ledger
  (`segment`, `cursor`, `world_size`) still names an exact data position, and resume works as in
  batch mode.
- **One writer per log.** A producer buffers `W` blocks, then commits; it must stay at least one
  segment ahead of the trainer for training to continue, and nothing stops it running further
  ahead (which is what `gc` is for). `W` is the streaming latency: the trainer cannot start a
  segment until `W` blocks have arrived, which is why streaming logs usually use a smaller `W`
  than batch logs.

## 2. An external producer

`examples/streaming_producer/produce.py` is a writer process: it creates the log, pushes blocks
one by one through `StreamingWriter`, sleeps after every committed segment, and ends the log
after `--segments` segments:

```python
from distrainer.writer import StreamingWriter

log = BlockLog.create(fs, root, W=W, seed=seed)
writer = StreamingWriter(log, shuffle_buffer_segments=k)
for block in arriving_blocks():  # each is a BlockRef from write_block
    segment = writer.push(block)  # commits a segment every W blocks
    if segment is not None:
        ...  # a segment is now visible to the trainer
writer.close()  # flushes what is left, writes _END
```

Run it on a laptop in two terminals. The producer first (it creates the log immediately, so the
trainer finds it and does not build a batch log of its own), then the trainer:

```bash
# terminal 1: 8 segments of W=12 blocks, 6 s of "arrival time" between segments (about 50 s)
uv run python examples/streaming_producer/produce.py \
    --config examples/hello_blocks/local-stream.yaml --segments 8 --sleep-s 6

# terminal 2
uv run python examples/hello_blocks/train.py --config examples/hello_blocks/local-stream.yaml
```

The producer prints `segment 0 committed at ...` and then one line every six seconds. Starting
a local Ray cluster and its two workers takes about fifteen seconds, so the trainer finds two or
three segments already committed, trains each in well under a second, catches up, and from then
on waits at every segment boundary. The audit trail shows the waiting: once caught up, the first
record of each segment is about six seconds after the first record of the previous one. Ranks
never fail on a missing segment, they wait; and `S1 PASS` at the end says the positions are
contiguous and each consumed once, exactly as in batch mode. (Make the producer outlast Ray's
startup, or you will see a plain batch run: with `--segments 6 --sleep-s 3` the whole log exists
before the first step.)

Two things about that pattern. The writer can run as far ahead as it likes: "one segment ahead"
is what the trainer needs for continuity, not a cap, and nothing throttles a producer on the
trainer's progress (a producer that wanted to stay one segment ahead would have to watch the
audit trail or the latest checkpoint before committing). And the latency you care about is the
steady state after the trainer has caught up: one segment's worth of blocks in the writer's
buffer plus the poll interval. Retention (next section) bounds the log behind the checkpoint,
not ahead of it.

`produce.py` takes the trainer's YAML for `store_root`, `seed`, `log.W`,
`log.shuffle_buffer_segments` and the block shape, so both sides agree; `--W`, `--seed`,
`--shuffle-buffer` and `--store` override it. The store must be empty: a stream cannot be
resumed by a second producer (`rm -rf blocks/hello_stream` between runs).

### The shuffle buffer

With `shuffle_buffer_segments: 1` a segment is the last `W` arrivals in arrival order (shuffled
within the segment by `BlockLog.append`). With `k > 1` the writer holds up to `k*W` blocks and
commits each segment as a random sample of `W` of them, so blocks mix beyond their arrival
window at the cost of `k` segments of latency before the first commit. `close()` flushes the
full segments still buffered, so `--segments N` always yields exactly `N` segments.

### Pacing and backpressure: the cases

There is exactly one flow-control mechanism in distrainer: a rank that needs segment `s+1`
polls for it. Everything else follows from who is faster.

| case | what happens | latency and cost |
|---|---|---|
| producer slower than the trainer (S11 after catching up, the tutorial run) | every rank finishes segment `s` and waits at the boundary until `s+1` is committed; all ranks wait equally, so no rank drifts | a block is trained at most `W` blocks plus one poll interval after it arrived; the workers idle while waiting |
| producer faster than the trainer | the log runs ahead; nothing throttles the producer and nothing is dropped | zero waiting, but the log grows without bound ahead of the trainer; `gc` only trims *behind* the last checkpoint, so a producer that will not be consumed for hours should pace itself |
| startup backlog (S11's first segments) | the producer started before Ray brought the workers up; the trainer drains the backlog at its own pace, then falls into the first case | temporary; the audit trail shows short gaps, then gaps equal to the producer's period |
| producer jitter, on average faster | the loader prefetches across segment boundaries, so a late segment is hidden as long as the next one arrives before the current one is consumed | none, up to `loader.prefetch` blocks |
| the hook as producer (S6) | rank 0 mines the next segment after its last step; the other ranks wait at the barrier for as long as mining takes | the mining time is paid once per segment by every rank; keep it small relative to `W/n` steps |
| batch mode | the whole corpus is committed before training starts: the extreme "producer ahead" case | no waiting ever; the log is as large as the corpus, `gc` behind the checkpoint keeps it from staying that large |
| restart with segments waiting | the new attempt resumes from its checkpoint and consumes the segments already committed in order; a hook-based producer replays the pending segment end if the old attempt died before appending | the resume replay of at most one checkpoint interval, as in batch mode |

Two non-goals worth stating. There is no producer-side backpressure: a writer never asks how far
the trainer has got, and a fast producer that fills the disk is the producer's problem
(it can read the audit trail or `distrainer inspect` the latest checkpoint if it wants to pace
itself). And there is no partial-segment consumption: a rank waits for a whole committed
segment rather than starting on the blocks that exist, because a committed segment is what makes
the ledger exact.

## 3. Retention: `gc`

A stream that runs for days cannot keep every segment. With

```yaml
log:
  gc: true
  retention_segments: 2
  gc_blocks: true
```

rank 0 runs `BlockLog.gc` at every segment end and drops every segment more than
`retention_segments` behind the segment of the last checkpoint it saved, together with the block
files no kept segment references. The window is anchored on the last *checkpoint*, not on the
current segment, because that is what a restart resumes from; and `retention_segments` must be
at least 1 because a checkpoint reported just before a failure may still be uploading, in which
case Ray Train restarts from the one before it. After the run above:

```bash
uv run distrainer log-ls blocks/hello_stream
```

```
segments: 3 (5..7)
ended: True
```

Segments 0 to 4 and their blocks are gone (`ls blocks/hello_stream/blocks | wc -l` says 36, three
segments' worth). Resuming from a checkpoint that points into the
deleted window fails with a clear error rather than silently training nothing. Set
`gc_blocks: false` if a producer will reference old blocks again in segments it has not written
yet; `distrainer gc <store> --keep-from N` does the same by hand. `docs/retention.md` is the
reference: the exact rule, `num_to_keep` and what stays resumable, cost on S3, non-goals.

## 4. The trainer as its own producer: segment hooks

Sometimes the *model* should decide what comes next: hard-negative mining in contrastive
training re-embeds the corpus with the current encoder and picks the negatives that are
hardest now. A **segment hook** runs on rank 0 right after the last step of a segment, before
the barrier that releases the other ranks, and may append the next segment:

```python
class MyHook:
    def __init__(self, config, **args): ...  # args come from the YAML

    def on_segment_end(self, model, ledger, log, ctx) -> None:
        nxt = ledger.segment + 1
        if log.ended() or log.has_segment(nxt):  # restarts replay segment ends: be idempotent
            return
        refs = [write_block(log.fs, log.root, f"s{nxt:06d}b{i:05d}", table_i) for i in range(log.W)]
        # (the shipped hook also puts ledger.run_attempt in the id, so a stale rank 0 of an
        # earlier attempt can never overwrite blocks the new attempt committed)
        log.append(refs, pass_idx=..., meta={"writer": "mine"})
        # or log.end() when the stream is finished
```

The hook gets the unwrapped `torch.nn.Module` (run it in `eval()` under `torch.no_grad()`, and
put it back), a snapshot of the ledger (`ledger.segment` is the segment that just ended), a
`BlockLog` of rank 0's own (the writer instance; the loader's reader keeps polling separately),
and the `TrainInfo` of the last step. It is configured, not passed in code, so the same
`train.py` runs with or without it:

```yaml
hooks:
  remine:
    entry: examples.toy_contrastive.remine:RemineHook    # pkg.module:factory
    segments: 12                                          # everything else is passed to the factory
    initial_segments: 1
```

`factory(config, **args)` is called on rank 0 inside the worker, so the hook can hold whatever
state it likes (the re-mining hook rebuilds the toy corpus from the seed); the driver only checks
that the entry imports. Two rules: only rank 0 runs hooks, and segment ends are delivered *at
least once*: after a failure or a resize the new attempt resumes from a checkpoint and re-runs
the segment ends the dead attempt already served, which is why the guard above matters
(`BlockLog.append` would otherwise allocate a fresh sequence number and drift the log).

### The re-mining example

```bash
uv run python examples/toy_contrastive/train.py --config examples/toy_contrastive/local-remine.yaml
uv run distrainer log-ls blocks/toy_remine -v
```

`make_blocks.py` sees `hooks.remine` and writes only segment 0 (negatives by centroid distance
in input space), leaving the log open. At the end of segment 0 rank 0 embeds all 7680 items with
the encoder as it is now, recomputes each cluster's nearest other clusters in *embedding* space,
mines a positive and four hard negatives for the 768 anchors of segment 1, writes them as
`s000001a00b00000..s000001a00b00023` (segment, attempt, block) and commits the segment; the other
ranks, waiting at the barrier,
then find segment 1 in the log. After `segments: 12` segments the hook writes `_END`:

```
segments: 12 (0..11)
ended: True
  00000000 pass=0 positions 0..23: s000000a00b00009, s000000a00b00008, ...
  00000001 pass=0 positions 24..47: s000001a00b00016, s000001a00b00004, ...
  ...
  00000010 pass=1 positions 240..263: s000010a00b00014, ...
  00000011 pass=1 positions 264..287: s000011a00b00022, ...
```

The anchors of segment `s` are a fixed slice of a per-pass permutation of the corpus, so
`pass_idx` advances when the slices wrap around (7680 items / 768 per segment = 10 segments per
pass), and `checkpoint.policy: pass_end` keeps working. Segment metadata records
`space: embedding` and `mined_after_segment`, and the audit trail shows the new block ids: the
scenario `S6` check asserts that every hook-made segment was committed before its first block
was consumed and that none of the base corpus was reused.

## 5. Failures and resume in streaming mode

Nothing new. The ledger in every checkpoint names a segment and a cursor; a restarted or resized
worker group re-deals from there over the new world size, and the writer, external or hook,
keeps going. With an external producer the trainer may find several segments already waiting
after a restart and consumes them in order; with a hook the restarted rank 0 skips the segment
ends whose successors exist. The only new failure mode is retention: resuming from a checkpoint
older than the gc window is refused.

## 6. Where the checks are

- `check_s6` (hook: commit precedes consumption, new block ids) and `check_s11` (producer:
  ranks waited, positions contiguous, `_END` ended the run, retention respected), both in
  `integration_tests/cluster/check_audit.py` and run on the container cluster by
  `just integration S6` and `just integration S11`, and on KubeRay pods with
  `DISTRAINER_DRIVER=kuberay` ([Tutorial 3](kuberay.md)); S11s3 puts the log on MinIO.
- `tests/test_remine.py` and `tests/test_streaming_producer.py` run the same paths on a fake
  Ray Train in the unit suite.
