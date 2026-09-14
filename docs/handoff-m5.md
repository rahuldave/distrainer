# Handoff: M5 (KubeRay) after M4 (segment hooks, streaming)

Written 2026-09-14 at the end of the M4 session for the thread that picks up M5. Read
`CLAUDE.md` and `AGENTS.md` first (workflow rules), then this file, then the spec sections named
below. Everything here was true when written; verify the Gest ids and branch state with
`gest task show` and `git status` before relying on them.

## 1. Where things stand

- M0 bootstrap, M1 core library, M2 trainer + examples + single-node scenarios, M3 container
  harness: merged to `main` (PRs #8 to #11; issues #2 to #5 closed).
- M4 segment hooks and streaming: branch `gest/vnupztox-m4-hooks`, PR #12 (issue #6). Scenarios
  S6 (re-mining hook), S8 (time-budget policy) and S11 (streaming producer with gc) are green on
  OrbStack next to S2, S3, S4, S9, S10; S1, S5, S7 on the laptop. Tutorials
  (`docs/tutorials/`), a CLI reference (`docs/cli.md`) and the scenario docs are part of the PR.
- Gest: root task `qvuukpsm` (tracking issue #1). M4 parent `vnupztox`, iteration `kvxwtzwv`.
  M5 parent `lntryyqu` (issue #7), iteration `ukvruwwu`, leaves `ykkvnqtp` (KubeRay manifests +
  driver) and `ywuxrspm` (S1 to S4 under KubeRay). `gest iteration graph ukvruwwu` shows the
  tree. Branch M5 from `main` once PR #12 is merged: `gest/lntryyqu-m5-kuberay`.
- Commits need the `AGENT_GEST_ALLOW_RAW_GIT_WRITES=1` prefix on raw git writes (see
  `CLAUDE.md`); one PR per milestone, `gpa` review by an Opus subagent, ask before merging.

## 2. Environment checklist

```bash
just setup                 # uv sync, Python 3.13 (floor 3.11)
just verify                # lint, typecheck, static, unit tests, regression, smoke (laptop only)
just build                 # container image, once (rebuild only when uv.lock changes)
just up 2 && just blocks && just train && just down   # harness sanity check
just integration S11       # one cluster scenario; `all` takes ~20 minutes
```

Memory matters on this 16 GB Mac: the OrbStack VM has 8 GB, the head container uses ~1.4 GB and
each worker ~0.6 GB. Run one cluster scenario at a time, and do not launch subagents that start
Ray or containers while a scenario runs; reviews and unit tests by subagents are fine in
parallel. `ruff format` also formats the fenced Python blocks in `docs/*.md`, so `just lint`
can fail on a tutorial.

## 3. What M4 delivered (spec sections 2.2, 4, 5, 7, 8, 10)

- **Hooks from config**: `hooks: {name: {entry: "pkg.module:factory", ...args}}`;
  `distrainer.hooks.build_hooks` instantiates `factory(cfg, **args)` on rank 0 inside the worker,
  `DistTrainer` resolves every entry on the driver first. Rank 0 owns a writer `BlockLog`
  (`hook_log`) separate from the loader's reader; hooks get the unwrapped model, a ledger
  snapshot, that log and the `TrainInfo` of the last step, before the barrier.
- **At-least-once segment ends**: a restart from a checkpoint taken at a segment end replays the
  pending hook call before the loop (`SegmentEnd` fires before the hooks; an attempt dying in
  between would otherwise leave every rank waiting forever). Hooks that append must guard with
  `log.ended()` / `log.has_segment(seq + 1)`.
- **Retention gc**: `log.gc` (default false), `log.gc_blocks` (default true). At every segment
  end rank 0 drops segments more than `retention_segments` behind the segment of the last
  checkpoint it saved; retention must be at least 1 because the controller may restart from the
  checkpoint before the last one saved (ASYNC upload in flight). A resume into the gc'd window,
  or a hole below the last committed segment, raises instead of ending as an empty run.
- **Re-mining hook** (`examples/toy_contrastive/remine.py`, S6): mines the next segment in the
  current encoder's embedding space; `make_blocks.py` streams only `initial_segments` when the
  hook is configured; block ids `s<seq>a<attempt>b<i>`; `n_items` must be a multiple of
  `W * anchors_per_block`.
- **Streaming producer** (`examples/streaming_producer/produce.py`, S11): `StreamingWriter`
  with sleeps and `_END`; `check_s11` parses commit times from the producer's output;
  `check_retention` pins the kept window and the block directory.
- **Time-budget policy** (S8): no library change; the scenario proves the broadcast consensus
  (the run finishing) and counts checkpoints against rank 0's `reports`.

## 4. Behaviours learned in M4 that will bite again

- **`SegmentEnd` checkpoints precede the hooks.** Anything that must happen "at the segment end"
  and is not idempotent has to cope with a restart landing exactly at the boundary; the trainer
  replays hooks, but a hook's own side effects (files, counters) must be guarded.
- **A streamed store belongs to one run.** An ended log from an earlier run makes the hook a
  no-op and S1 still passes; `toy_contrastive/train.py` wipes the store in streaming mode, the
  scenario runner wipes `/shared/blocks_toy` and `/shared/blocks_stream`.
- **Start the producer before the trainer**, and make it outlast Ray's startup. `train.py`
  builds a batch log when the store is empty; local Ray plus two workers takes ~15 s, so a
  producer that finishes sooner shows no waiting at all (the tutorial uses 8 segments 6 s apart).
- **The scenario runner's `finish()` dropped its log entry before reading the output**, so every
  scenario that reads the driver's stdout got `""`; S8 and S11 were the first to notice. Unit
  tests for the runner's bookkeeping live in `tests/test_run_scenarios.py`.
- **`check_s6` must check that the hook actually ran** (`meta.writer`, `mined_after_segment`,
  commit after rank 0's last record of the previous segment), otherwise a never-re-mined log
  passes.
- **Timestamps** (`created_at`, audit `ts`) are `time.time()` on the containers' host; the checks
  carry a small clock tolerance for the multi-node case.
- Everything from the M3 list still applies: `ray.train.report` semantics, in-flight
  checkpoints, `docker kill` never triggering a restart policy, run names being restored not
  restarted, resuming a completed run does nothing, the elastic monitor interval, editing
  `deploy/drivers/*.sh` mid-run, MinIO's image location, Ray's `uv run` hook.

## 5. Review follow-ups still open (small, pick up when touching the area)

- `CheckpointIO.cleanup()` is never called; only Ray-emptied scratch dirs are removed at loop
  exit. In-flight ASYNC uploads across a worker restart can leak a dir.
- `log/<seq>.rows.parquet` (spec 3.1 optional reverse index) is neither written nor needed;
  implement or drop the line from the spec.
- `tests/conftest.py::MemoryFS` clears fsspec's process-global memory store.
- `BlockLog.has_segment`/`gc` touch the segment cache outside the lock (harmless under the GIL).
- `BatchWriter.run()` on an empty corpus writes `_END` on an empty log silently.
- `TrainInfo.step_in_segment` (steps done before this one) and `StepContext.step_in_segment`
  (including this one) share a name with two meanings.
- Hooks passed programmatically to `DistTrainer(hooks=...)` are cloudpickled to every rank;
  only config hooks are built on rank 0.
- `produce.py` sleeps after commits, so with `shuffle_buffer > 1` the segments flushed at
  `close()` arrive back to back (documented; S11 runs with 1).
- Producer-side backpressure is a documented non-goal (a fast producer may run arbitrarily
  ahead; `gc` bounds the log behind the checkpoint only); the cases are tabulated in
  `docs/tutorials/streaming.md`.
- Streaming to S3 has not been run: batch logs on MinIO are exercised by S9/S10, but no producer
  or hook has appended segments to a bucket while ranks polled it, and `gc` has not deleted on
  S3. An S11 variant on MinIO (`harness-stream-minio.yaml`, the runner reading the trail through
  the head as S9 does) would close that.
- The upstream skills bundle issues listed in the M3 handoff (fix in
  `rahuldave/agent_gest_git_skills`, not here).

## 6. M5 (KubeRay) pointers

Enable OrbStack's Kubernetes (`orb config set k8s.enable true`), install the KubeRay operator
with Helm, write `deploy/k8s/raycluster.yaml` (head pod without the `trainer` resource, a worker
group with `minReplicas`/`maxReplicas` = `num_workers`, each worker pod with `trainer: 1` via
`rayStartParams.resources`) and `deploy/k8s/rayjob.yaml` running `train.py`; implement
`deploy/drivers/kuberay.sh` with the same verbs (`up N` = apply + scale, `kill-worker I` =
`kubectl delete pod`, `scale N` = patch `replicas`, `exec-head` = `kubectl exec` into the head
pod, `cp-from-head` = `kubectl cp`). Shared storage: a `hostPath`/`local-path` PVC mounted at
`/shared`, or MinIO as a Deployment with `storage.kind: s3`. Then `DISTRAINER_DRIVER=kuberay
just integration S2` and S3/S4; the checker is unchanged, and S6/S11 should run unchanged too
(their stores are under `/shared`). uncloud (mode C) is a stub until machines exist; the plan is
OrbStack Linux machines as Docker hosts (`docs/running-modes.md`).
