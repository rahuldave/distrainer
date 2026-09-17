# Handoff: after M8 (RunPod pods): what is there, what it taught, and what comes before DiLoCo and FSDP

Written 2026-09-17 at the end of the M8 session for the thread that picks up the next step.
Read `CLAUDE.md` and `AGENTS.md` first (workflow rules), then this file, then tutorial 6
(`docs/tutorials/runpod.md`), `docs/runpod-gotchas.md`, and the two explainers written with
this handoff (`docs/collectives.md`, `docs/parallelism.md`), which frame the next phase.
Verify the Gest ids and the branch state with `gest task show` and `git status` before
relying on them.

## 1. Where things stand

- M0 to M8 are merged to `main`: M8, the GPU contrastive example on RunPod pods (PR #21,
  issue #20, squash `d41719c`; Gest parent `zvymnspr`, iteration `toonvruz`). The parent and
  the iteration stay open for the docs leaf `nnlkrrul` (this handoff, the explainers, the
  running-modes and verb-map polish); everything else of M8 is done.
- **RunPod state at the end of the session**: every pod terminated (`down` without the stop
  mode; the API lists nothing with the `DISTRAINER_CLUSTER` marker). What remains: the bucket
  `distrainer-rahuldave` (us-east-1: the hello and image block logs, the probe split, the
  CIFAR tarball under `datasets/`, the runs and audit trails, a few hundred MB) and the ECR
  repository from M7b. The GHCR package `ghcr.io/rahuldave/distrainer-gpu` is public, with
  `latest` built from the merge and the branch tag `gest-zvymnspr-runpod` from the session.
  Spend for the day about 6 USD of the 30 USD ceiling.
- **Rotate the RunPod API key** (`RUNPOD_KEY` in `.env`): RunPod injects the account key into
  every pod's environment and it was printed once in a session transcript.

## 2. Environment checklist

```bash
just setup && just verify                      # laptop gate (226 unit tests, smoke)
just images                                    # the image example at CPU size (CIFAR-10 cached under .harness/datasets)
export $(grep '^S3_' .harness/aws/env | xargs)  # the bucket: the RunPod driver reads .env only
export DISTRAINER_DRIVER=runpod DISTRAINER_IMAGE=ghcr.io/rahuldave/distrainer-gpu:latest
export DISTRAINER_RUNPOD_DOWN=stop DISTRAINER_RUNPOD_DATA_CENTERS=EU-RO-1   # a session of scenarios
export DISTRAINER_WAIT_TRAINERS_S=900 DISTRAINER_LAG_INTERVALS=12           # the runner's pod-sized budgets
deploy/driver.sh catalog                       # what is rentable now, and where
deploy/driver.sh up 3                          # head + 3 workers; pulls of 1 to 35 min per pod
deploy/driver.sh ps                            # the pods and the hourly total
nohup uv run python integration_tests/cluster/run_scenarios.py --scenario S2 --keep-up > s2.out 2>&1 &   # detached: see section 4
deploy/driver.sh down                          # with DISTRAINER_RUNPOD_DOWN unset: terminate everything
```

`.env` holds `RUNPOD_KEY` and `DISTRAINER_RUNPOD_SSH_KEY=~/.ssh/id_rsa`; `.env.example`
documents every `DISTRAINER_RUNPOD_*` setting. OrbStack is stopped on the Mac (`orb start`
brings the compose, KubeRay and uncloud beds back); nothing else changed there.

## 3. What M8 delivered

- `examples/image_contrastive`: SimCLR on CIFAR-10 blocks (every image one Parquet row with
  its PNG bytes), a CIFAR-stem ResNet with width and depth knobs, batched per-sample
  augmentations on the device, NT-Xent through the toy's `info_nce` with `all_gather`, a
  weighted kNN probe after `fit()`; `just images`; 12 tests; laptop probe 0.29 on the tiny
  encoder, 0.448 on an A5000 at world size 1.
- `deploy/Dockerfile.gpu` and `.github/workflows/gpu-image.yml`: the CUDA 12.6 wheels of the
  locked torch over the CPU image's layout, sshd from RunPod's injected key, the
  global-networking address and the collectives' interface found at start, the drain marker
  in `ray-worker.sh`; built on the runner into GHCR with a native head-role smoke.
- `deploy/drivers/runpod.sh`: the driver verbs on the REST v2 API (v1 dies 2026-11-15): pods
  by a name prefix and an env marker, an ordered GPU list under a spend guard, an explicit
  data-center list, node death as stop and start, the session stop mode (`down` stops, a
  resize drains or undrains a worker's Ray node over ssh), a probed ssh port for a restarted
  head, `cost` and `catalog`; 15 stub-API tests. The runner gained `DISTRAINER_WAIT_TRAINERS_S`
  and `DISTRAINER_LAG_INTERVALS` (defaults unchanged).
- The runs, 2026-09-16, all on the bucket:

  | run | pods | result |
  |---|---|---|
  | image example, world size 1 | the head's A5000 | 95 s for 384 positions, probe 0.448 |
  | image example, world size 2 | CA-MTL-1 + EU-RO-1 | 15.5 min (2.5 s per step over the transatlantic link), probe 0.385 |
  | image example, a worker lost mid-run | the head alone after | resumed with no replay, probe 0.411 |
  | S2 / S3 / S4 / S9 / S10 (hello_blocks) | head + 2 or 3 workers in EU-RO-1 | pass: 218 / 194 / 297 / 1624 / 255 s |

- Docs: tutorial 6, `docs/runpod-gotchas.md`, running-modes mode E, the CLI reference, the
  spec's sections 8, 9 and 11, `.env.example`, and with this handoff the two explainers.

## 4. Behaviours learned in M8 that will bite again

- **The image pull is the clock**: 1 to 35 minutes per pod by host, the same image, the same
  data center. Rent everything at once, keep the pods for the session (the stop mode), and
  never plan a scenario around a new pod joining mid-run.
- **A stopped pod's card is rented away within minutes**; its restart is then refused. Hence
  the drain marker for resizes. A cold restore (S9) still stops and starts every pod and can
  end in replacements; it passed once in 27 minutes.
- **The API is eventually consistent about a restarted pod**: `status: RUNNING` from the
  moment a pod is rented, the old container's `runtime` for a while, the old direct ssh port
  for more than ten minutes. The driver waits for a newer `startedAt` and probes the port;
  `exec-head` refreshes its cache once on a refused connection.
- **The global-networking interface arrives after the container starts** (the entrypoint
  retries for two minutes, inside a RunPod pod only) and **NCCL and Gloo must be pinned to
  it** (`podnet1`), or DDP hangs across pods with both workers alive and no step taken.
- **Global networking needs secure-cloud NVIDIA GPU pods**, the head too; the cheap types show
  `LOW` and vanish before the create; two Montreal pods in a row could not reach the head at
  all while Romanian ones could. Pin the data center.
- **The Train controller lags further on a distant bucket**: a resumed attempt replayed 49
  positions from Romania to us-east-1 where the AWS bed's allowance gave 27.
- **The Mac's memory kills background tasks** (Xcode, a simulator, other assistants; not
  OrbStack in the end): run anything longer than ten minutes detached with `nohup … &` and
  poll its file; macOS has no `setsid`.
- **A login shell on a pod starts in `/root`**, its environment is what the entrypoint
  exported, and RunPod puts `RUNPOD_API_KEY` in it: never print a pod's environment.
- Everything in the M7 list still applies (`internal_docs/handoff-m8.md` section 4).

## 5. Review follow-ups still open

- The 7.5 GB image: a smaller one, or a RunPod-cached base image, would cut the pulls that
  dominated the day; the CUDA layer is 6.6 GB of NVIDIA libraries the wheels require.
- `kill-worker` still stops the pod (node death that releases the card); the drain marker
  would model node death without the refusal risk, at the cost of not being a real death.
- The runner's `lag_intervals` and trainer wait are environment settings; a driver could
  declare its own defaults instead.
- The image example's `all_gather` path has no unit test (the fake-Ray fixture starts no
  process group); an autocast dtype knob is undecided (the L4 and A5000 took bf16).
- `harness-s3.yaml` and the image config name a personal bucket; `wipe-shared` is a no-op
  under RunPod; `cp-from-head` is used by no scenario.
- The `RUNPOD_API_KEY` in every pod is RunPod's doing; a pod-scoped key would be better if
  RunPod offers one.
- The M4 to M7 lists in `internal_docs/handoff-m8.md` section 5 are unchanged.

## 6. The next phase: before DiLoCo, FSDP and the rest

`docs/parallelism.md` section 3 is the map: DDP is built, FSDP is a wrapping and a checkpoint
change, local SGD and DiLoCo fit the segment end, tensor and pipeline parallelism do not fit
and should not be attempted here. What has to exist before either of the two good fits is
implemented:

### 6.1 A real process group in the tests

The fake-Ray fixture (`tests/test_train_loop.py`) drives `train_loop` rank by rank in one
process with the collectives stubbed out. Every one of the additions below is *about*
collectives, so the first slice is a test harness that runs `n` ranks as real processes with
a Gloo group on localhost (`torch.multiprocessing.spawn`, `init_process_group("gloo")`, a
free port), fed by the same block stores the fake fixture builds. It should be able to run
the existing loop under real DDP on the CPU and assert the same audit facts. Until it exists,
`all_gather`, `no_sync` and a sync-at-segment-end cannot be tested off a cluster.

### 6.2 Sharded checkpoints in `CheckpointIO`

Today `CheckpointIO.save` writes `model.pt`, `optimizer.pt` and `ledger.json` from rank 0 and
`load` restores on every rank. FSDP needs a second shape: every rank writes its shard
(`torch.distributed.checkpoint` under the same directory, the ledger unchanged), the checkpoint
name carries the world size already, and `load` must reshard when the world size differs
(DCP does this on load for FSDP state). Ray Train's `Checkpoint` and `num_to_keep` count
directories, which is fine. The `distrainer resume --entry` CLI must accept both shapes and
`inspect` must show which one it holds. Design the ledger to be the same four integers in
both.

### 6.3 A `parallel:` section of the config, and `build_model` wrapping

`prepare_model` wraps in DDP unconditionally. A `parallel: {kind: ddp | fsdp | local_sgd,
...}` section chooses the wrap (`fully_shard` with its policy and mixed precision for FSDP;
DDP with `no_sync` for local SGD; plain DDP by default), validated in `config.py`, threaded
through `TrainInfo` so `train_step` can ask `info.parallel`. The contrastive `all_gather`
works under every kind (it gathers activations, not weights).

### 6.4 A sync point on every rank at the segment end

Hooks run on rank 0 only (writers). Local SGD and DiLoCo need a collective on *every* rank
at the segment end: average the parameters (all-reduce of the weights, or of the change since
the last sync fed to an outer optimizer whose state joins the checkpoint). The loop already
has the barrier there; the addition is a per-rank hook point (`on_segment_sync(model,
optimizer, info)`) that runs before rank 0's writer hooks, plus the rule that a checkpoint at
a segment end is taken *after* the sync. The dealing, the ledger and the audit need nothing.
`H` is then `W / n` steps, which is a nice consequence of the design: the segment is the sync
unit.

### 6.5 The bed for it

DiLoCo's experiment is exactly the transatlantic RunPod cluster (2.5 s per step under DDP
versus 0.15 s alone: the gap local SGD closes). FSDP's experiment wants several GPUs on one
fast link: a single RunPod pod with 2 to 4 GPUs (`gpu.count`), which needs a driver mode
where one pod hosts `k` ranks (Ray starts with `k` GPUs, `resources_per_worker` unchanged);
the driver's `up` takes a per-pod GPU count and the head's `trainer` rule stays. Both need
the image pull problem addressed first (section 5) or the patience for it.

### 6.6 Order of work for the session

1. `gpl` an M9 iteration from the Gest task below; the leaves in this order: the process-group
   test harness (6.1), the `parallel:` section with DDP as the only kind and the fixture
   proving it (6.3), the sync-at-segment-end hook with local SGD on the CPU harness (6.4),
   DiLoCo on the transatlantic pods (6.5), then sharded checkpoints and FSDP (6.2) on a
   multi-GPU pod.
2. Keep the runs at the end and the pods for the session; budget as before.

Sources: `docs/parallelism.md`, `docs/collectives.md`, the DiLoCo paper (Douillard et al.,
2023), PyTorch's FSDP and `torch.distributed.checkpoint` documentation, Ray Train v2's
`prepare_model`.
