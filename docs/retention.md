# Retention and garbage collection

A block log is append-only: segments and their block files accumulate until something deletes
them. In batch mode that is a nuisance (the corpus stays on disk for every pass); in streaming
mode it is a leak. `gc` is the one deletion mechanism, and this page is its reference: what it
deletes, when, why it is anchored where it is, how it interacts with checkpoints and resume,
and what it deliberately does not do. Tutorial 2 shows it in use; `docs/cli.md` has the flags.

## What gc deletes

`BlockLog.gc(keep_from_seq, delete_blocks=True)` deletes, in this order, for every committed
segment with a sequence number below `keep_from_seq`: the block files that no *kept* segment
references (only if `delete_blocks`), then the segment file. It returns the deleted sequence
numbers, it is idempotent, and re-running it after a crash finishes the job. Everything from
`keep_from_seq` on is untouched, so after gc the log simply starts later:

```
$ uv run distrainer log-ls blocks/hello_stream
segments: 3 (7..9)
ended: True
```

A block referenced by any kept segment survives even if a deleted segment also referenced it.
Batch logs commit every pass up front, so a block that pass 2 will train on is referenced by a
kept segment and is never deleted while pass 1's segments go. Set `gc_blocks: false` only when a
producer will reference old blocks again in segments it has *not committed yet*; then gc drops
segment files only.

## When the trainer runs it

```yaml
log:
  gc: true                 # default false
  retention_segments: 2    # default 4; must be at least 1 when gc is on
  gc_blocks: true          # default true
```

With `log.gc: true`, rank 0 runs gc at every segment end, after the segment hooks and before the
barrier that releases the other ranks, with

```
keep_from = segment of the last checkpoint rank 0 saved  -  retention_segments
```

and nothing is deleted while `keep_from` is 0 or less. "The last checkpoint rank 0 saved" is the
newest checkpoint of the current attempt, or, right after a restart, the checkpoint the attempt
resumed from (segment 0 for a fresh run). Worked example with `W=24`, two workers, the `any`
policy (a checkpoint at every segment end) and `retention_segments: 2`: at the end of segment 9
the last checkpoint is in segment 9, `keep_from = 7`, segments 0 to 6 and their blocks are
deleted, the log keeps 7, 8, 9. That is what scenario S11 asserts.

Two design points:

- **Anchored on the checkpoint, not on the current segment**, because a restart resumes from a
  checkpoint, and the resumed attempt re-reads that checkpoint's segment and everything after
  it. A window measured from the current segment would delete data a restart still needs
  whenever checkpoints are rarer than segments.
- **`retention_segments` must be at least 1**, because checkpoints are uploaded asynchronously:
  a checkpoint reported just before a failure may not be registered by the controller yet, which
  then restarts from the one before it. One segment of margin covers that.

## Checkpoints, `num_to_keep`, and what stays resumable

gc never touches checkpoints; Ray Train's `checkpoint.num_to_keep` does. The two interact: a
checkpoint is resumable only if the segment its resume starts in still exists. After a failure
Ray restores the newest registered checkpoint, which retention 1 protects. A manual
`distrainer resume` from an older checkpoint may point into the deleted window; the trainer then
refuses with

```
cannot resume at segment 4: the log starts at 7 (garbage-collected); resume from a newer
checkpoint or raise log.retention_segments
```

rather than training nothing. The same error covers a hole below the last committed segment.
Rule of thumb if you want every kept checkpoint to remain resumable: `retention_segments` must
be at least the number of segments the kept checkpoints span, which is
`floor((num_to_keep - 1) / checkpoints per segment)`, and at least 1. With a checkpoint at every
segment end and `num_to_keep: 3`, that is 2.

## Manual gc

```bash
uv run distrainer gc blocks/hello --keep-from 2                 # segments 0 and 1 and their blocks
uv run distrainer gc s3://distrainer/blocks_stream --keep-from 7 --keep-blocks   # segment files only
```

Use it on a batch log after a run, on a store the trainer has no `gc: true` for, or to shrink a
bucket by hand. Pick `--keep-from` from `distrainer inspect` of the oldest checkpoint you still
want to resume from.

## Cost and storage backends

Each gc call does one directory listing of `log/`, reads the kept segment files (a few KB each,
cached), and issues one delete per block file and per segment file. That is a few dozen
requests per segment end on S3 at `W=24`; the listing walks only committed segments, so a
stream at sequence 100000 does not stat every number below the window. Deletes on an object
store are per object and not atomic as a group; if gc dies between deleting a segment's blocks
and its segment file, the next call deletes the segment file, and no reader was going to read a
segment below the window anyway. Scenario S11s3 exercises gc against MinIO, under the compose
driver and under KubeRay with MinIO running in the cluster as a Deployment.

## What gc does not do

- It does not bound the log *ahead* of the trainer. A producer faster than the trainer can run
  arbitrarily far ahead; there is no producer-side backpressure (see the pacing table in
  Tutorial 2). Retention only trims behind the checkpoint.
- It does not sweep orphans. A block file that no segment ever referenced (a writer that
  crashed between writing blocks and committing the segment) is ignored by readers and left on
  disk; gc deletes only the blocks of the segments it drops. An orphan sweep is a listed
  follow-up.
- It does not delete checkpoints, audit trails, or Ray Train run state.
- It runs on rank 0 only, and only when `log.gc` is set; a batch run with the default config
  deletes nothing.

## Where it is verified

`tests/test_log.py` (the `gc` rules on blocks shared between segments), `tests/test_train_loop.py`
(the window behind the checkpoint, `gc_blocks: false`, a policy that never checkpoints, the
refused resume), `tests/test_streaming_producer.py` (an end-to-end stream with gc),
`check_retention` in `integration_tests/cluster/check_audit.py`, and scenarios S11 and S11s3 on
the container cluster and on KubeRay (`docs/examples-and-scenarios.md`).
