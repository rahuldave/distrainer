# Handoff: M4 (segment hooks, streaming) and M5 (KubeRay)

Written 2026-09-14 at the end of the M3 session for the thread that picks up M4. Read
`CLAUDE.md` and `AGENTS.md` first (workflow rules), then this file, then the spec sections named
below. Everything here was true when written; verify the Gest ids and branch state with
`gest task show` and `git status` before relying on them.

## 1. Where things stand

- M0 bootstrap, M1 core library, M2 trainer + examples + single-node scenarios: merged to `main`
  (PRs #8, #9, #10; issues #2, #3, #4 closed).
- M3 container harness: merged to `main` (PR #11, issue #5 closed). Scenarios S2, S3, S4 (shared
  mount) and S9, S10 (MinIO) are green on OrbStack. Branch M4 from `main`.
- Gest: root task `qvuukpsm` (tracking issue #1). M4 parent `vnupztox` (issue #6), iteration
  `kvxwtzwv`, leaves: `uxqnrqym` hooks.py wiring, `yvmqsynm` remine.py + S6, `xxwrtxok` streaming
  producer + S11 + gc, `wmkxvwrk` TimeBudget + S8, `pzvrvmzx` review/docs/PR. M5 parent
  `lntryyqu` (issue #7), iteration `ukvruwwu`, leaves `ykkvnqtp` (KubeRay manifests + driver) and
  `ywuxrspm` (S1 to S4 under KubeRay). `gest iteration graph kvxwtzwv` shows the M4 tree.
- Branch for M4: `gest/vnupztox-m4-hooks` from `main`; commits need the
  `AGENT_GEST_ALLOW_RAW_GIT_WRITES=1` prefix on raw git writes (see CLAUDE.md); one PR for the
  milestone, `gpa` review by an Opus subagent, ask before merging.

## 2. Environment checklist

```bash
just setup                 # uv sync, Python 3.13 (floor 3.11)
just verify                # lint, typecheck, static, unit tests, regression, smoke (laptop only)
just build                 # container image, once (rebuild only when uv.lock changes)
just up 2 && just blocks && just train && just down   # harness sanity check
just integration S2        # one cluster scenario; `all` takes ~12 minutes
```

Memory matters on this 16 GB Mac: the OrbStack VM has 8 GB, the head container uses ~1.4 GB and
each worker ~0.6 GB. Run one cluster scenario at a time, and do not launch Opus subagents that
start Ray or containers while a scenario runs; Claude Code's memory monitor kills background
commands when the host runs low. Reviews and unit tests by subagents are fine in parallel.

## 3. What M4 has to deliver (spec sections 2.2 writer/streaming, 4 hooks.py, 5, 8, 10 S6/S8/S11)

### 3.1 Hooks wired into the trainer (`uxqnrqym`)

`distrainer/hooks.py` has the `SegmentHook` protocol and `train_loop` already calls
`hook.on_segment_end(model, ledger, log, step_info)` on rank 0 at every segment end, before the
`barrier()`, for the `hooks` sequence passed to `DistTrainer`. Missing:

- Hooks from config. `hooks:` in the YAML is a free dict today. Define it as a list of
  `{entry: "pkg.module:Class", args: {...}}` (or keep the mapping form the spec sketches,
  `remine: {every_segment: true}`, and map names to entries); build them in `DistTrainer` with
  `importlib` the same way `cli.py resume --entry` does, and pass them through `loop_config`
  (cloudpickle by reference: the driver must run from the repo root, which it does in the head
  container).
- Give the hook its own `BlockLog` instance (`BlockLog.open(fs, root)` inside the hook, not the
  trainer's reader instance). `BlockLog` is lock-protected now, but a writer and the loader's
  producer thread sharing one instance is a review follow-up worth closing. Pass the hook the
  ledger (immutable snapshot), the unwrapped model (`distrainer.trainer.unwrap`), and `TrainInfo`.
- `gc` at segment end on rank 0 when `log.retention_segments` is set:
  `log.gc(keep_from_seq=ledger.segment - retention, delete_blocks=<streaming?>)`. Blocks may only
  be deleted when no later segment will reference them, i.e. streaming mode where every segment
  brings new blocks; in multi-pass batch mode pass `delete_blocks=False`. Expose the choice in
  config (`log.gc_blocks: bool`) rather than guessing.
- `train_loop` must tolerate the hook appending a segment *after* the loader's producer thread has
  already asked `wait_segment(seq+1)`: it polls, so this works, but S6 asserts the commit precedes
  the first consumption (compare the segment file's `meta.created_at` with the audit `ts` of the
  first record of that segment).

### 3.2 Re-mining hook and S6 (`yvmqsynm`)

`examples/toy_contrastive/remine.py`: a `SegmentHook` that (rank 0, at every segment end)
embeds the whole corpus with the current encoder (`unwrap(model)`, `torch.no_grad`), recomputes
each cluster's nearest other clusters *in embedding space* (reuse `hard_negative_clusters` on
embedded centroids), mines positives and negatives with `mine_rows`, writes the next `W` blocks
(`write_block` with fresh block ids such as `s{seq+1}b{i:05d}`), and appends them as segment
`seq+1` with `pass_idx` advancing when the corpus wraps. It writes `_END` after
`hooks.remine.segments` segments. For this mode `make_blocks.py` must build only the first
segment(s) and *not* end the log (add a `streaming: true` switch that stops after
`hooks.remine.initial_segments` and skips `log.end()`); the corpus arrays are reproducible from
`seed`, so the hook can rebuild them with `make_corpus` instead of shipping them in the checkpoint.
Keep the cost low: 7680 items through a 32x64x16 MLP is milliseconds; mining is
`mine_rows` (0.3 s at full scale).

S6 assertions (`check_s6`, new in `check_audit.py`): for every segment `seq+1 >= 1`, its
`meta.created_at` is earlier than the audit `ts` of the first record with that segment; the block
ids consumed in segments produced by the hook differ from those of the base corpus; the run ends
at `_END` with contiguous positions. Run it on the cluster (`just integration S6`) with a
`harness-remine.yaml` for toy_contrastive (2 workers, `W=24`, `step_sleep_s` is not needed since
the hook itself takes time).

### 3.3 Streaming producer and S11 (`xxwrtxok`)

`examples/streaming_producer/produce.py`: a separate process using `StreamingWriter` (M1,
tested) that creates the log (`BlockLog.create`), then pushes hello_blocks-style blocks with a
configurable sleep between segments (`--segments N --sleep-s S --W 24`) and calls `close()`
(which writes `_END`). Run it inside the head container concurrently with `train.py` (the runner
starts both with `exec-head`; the trainer's `wait_segment` polls every `log.wait_poll_s`). S11
assertions: the audit `ts` gap between the last record of segment `k` and the first of `k+1` is
at least the producer's sleep minus the poll interval (ranks waited rather than failed);
positions contiguous and every segment consumed once (`check_s1` semantics per attempt); the run
ends when `_END` appears; after the run, `log.first_seq()` is at least
`last checkpoint segment - retention_segments` when `gc` is on (segments behind the retention
window are gone, blocks of deleted segments too, blocks of kept segments intact). The
`StreamingWriter` shuffle buffer (`shuffle_buffer_segments`) is already implemented; give the
producer a flag for it and cover both values.

### 3.4 TimeBudget and S8 (`wmkxvwrk`)

`TimeBudget` exists in `policy.py` (rank 0 decides, `broadcast_from_rank_zero` every
`poll_every` steps) and `build_policy` accepts `policy: time`. S8 only needs a config
(`checkpoint.policy: time`, `time_budget_s: 5`, `time_poll_every: 2`) run on the cluster and
the assertion that every rank called `report` the same number of times: Ray Train v2 enforces
this (`_sync_checkpoint_dir_name_across_ranks` is a broadcast inside every `report`), so the run
finishing at all is the proof; additionally compare `Result.metrics["reports"]` with the number
of `checkpoint_*` directories (with `num_to_keep: null`). Watch the interaction with the
report-only-at-checkpoints cadence: a time policy that never fires means one final report only.

### 3.5 Docs and review (`pzvrvmzx`)

Update `docs/examples-and-scenarios.md` (S6, S8, S11 rows and the `streaming_producer`
example), `docs/running-modes.md` if the harness gains verbs, spec section 12 M4 status. Run
`grv` (adversarial, Opus subagent) on the M4 diff before the PR and `gpa` on the PR.

## 4. Behaviours learned in M2/M3 that will bite again

- **`ray.train.report` semantics.** Metrics-only reports block on a one-slot queue drained at
  the controller's poll interval (`RAY_TRAIN_HEALTH_CHECK_INTERVAL_S`, set to 0.5 by
  `init_ray`); reports with an ASYNC checkpoint return immediately and are queued by a background
  thread in order. The loop reports only at checkpoint points; a report per step caps throughput
  at about one step per poll.
- **In-flight checkpoints.** A checkpoint reported just before a failure may not be registered
  by the controller; the resumed attempt then starts one checkpoint interval earlier. The
  `check_recovery` bound is `2*every_k*n_old + n_new` for that reason.
- **A worker killed after its training function returned** but before the controller drained
  its last results made Ray restart the group without the newest checkpoint (seen once with a
  0.35 s run). Keep harness runs long enough (`step_sleep_s`) that failures land mid-run.
- **`docker kill` never triggers a restart policy** (Docker treats it as a manual stop); the
  driver's `kill-worker` starts the container again after `DISTRAINER_RESTART_DELAY` seconds.
- **A run name that already exists on the storage is restored, not restarted**, by Ray Train v2
  (`checkpoint_manager_snapshot.json`). `train.py` wipes local run dirs; on S3 the scenario
  runner deletes `runs/<name>` and `audit/<name>` first (`s3_rm`). Use fresh run names.
- **Resuming a completed run from its newest checkpoint does nothing** (the ledger is at the end
  of the log). S9 keeps all checkpoints (`num_to_keep: null`) and resumes from a mid-run one.
- **The elastic policy checks every `elastic_resize_monitor_interval_s`** (5 s in the harness)
  and a new container needs ~10 s to join; the runner waits for the requested number of
  `trainer` resources before starting a run (`wait_for_trainers`).
- **Edits to `deploy/drivers/*.sh` while a scenario is polling the driver** produce transient
  "command not found" noise (bash reads a half-written file). Edit between runs.
- **MinIO's Docker Hub image is gone**; the compose file uses `quay.io/minio/minio` and a
  `/dev/tcp` healthcheck (the image ships neither `mc` nor `curl`).
- **Ray's `uv run` runtime-env hook** would re-launch workers through `uv` from an uploaded
  copy of the repo; `init_ray` sets `RAY_ENABLE_UV_RUN_RUNTIME_ENV=0`. Local config paths are
  absolutised at load time because workers do not share the driver's cwd.
- **Spec inconsistencies already fixed**: `W: 16` with `num_workers: [2, 4]` cannot serve 3
  workers (now `W: 24`); partial segments are not supported (writer tail policy instead);
  checkpoint directories are named `checkpoint_g{seg}_p{positions}_n{ws}_a{attempt}`.

## 5. Review follow-ups still open (small, pick up when touching the area)

- `CheckpointIO.cleanup()` is never called; only Ray-emptied scratch dirs are removed at loop
  exit (`cleanup_uploaded`). In-flight ASYNC uploads across a worker restart can leak a dir.
- `log/<seq>.rows.parquet` (spec 3.1 optional reverse index) is neither written nor needed;
  either implement or drop the line from the spec.
- `tests/conftest.py::MemoryFS` clears fsspec's process-global memory store; give it an
  isolated instance.
- `BlockLog.has_segment`/`gc` touch the segment cache outside the lock (harmless under the GIL).
- `BatchWriter.run()` on an empty corpus writes `_END` on an empty log silently.
- `TrainInfo.step_in_segment` (steps done before this one) and `StepContext.step_in_segment`
  (including this one) share a name with two meanings.
- The upstream skills bundle: two hooks emit `hookSpecificOutput` without `hookEventName`, the
  installer clones an unpinned branch, and `raw-git-write-guard.sh` also blocks read-only
  `git branch --show-current`. Fix in `rahuldave/agent_gest_git_skills`, not here.

## 6. M5 (KubeRay) pointers

Enable OrbStack's Kubernetes (`orb config set k8s.enable true`), install the KubeRay operator
with Helm, write `deploy/k8s/raycluster.yaml` (head pod without the `trainer` resource, a worker
group with `minReplicas`/`maxReplicas` = `num_workers`, each worker pod with `trainer: 1` via
`rayStartParams.resources`) and `deploy/k8s/rayjob.yaml` running `train.py`; implement
`deploy/drivers/kuberay.sh` with the same verbs (`up N` = apply + scale, `kill-worker I` =
`kubectl delete pod`, `scale N` = patch `replicas`, `exec-head` = `kubectl exec` into the head
pod, `cp-from-head` = `kubectl cp`). Shared storage: a `hostPath`/`local-path` PVC mounted at
`/shared`, or MinIO as a Deployment with `storage.kind: s3`. Then `DISTRAINER_DRIVER=kuberay
just integration S2` and S3/S4; the checker is unchanged. uncloud (mode C) is a stub until
machines exist; the plan is OrbStack Linux machines as Docker hosts (`docs/running-modes.md`).
