# Handoff: after M9 (the parallel kinds on the CPU): what is there, what it taught, and M10

Written 2026-09-17 at the end of the M9 session for the thread that picks up M10. Read
`CLAUDE.md` and `AGENTS.md` first (workflow rules), then this file, then tutorial 7
(`docs/tutorials/parallel.md`) and `docs/parallelism.md`. Verify the Gest ids and the branch
state with `gest task show` and `git status` before relying on them.

## 1. Where things stand

- M0 to M9 are merged to `main`. M9, the parallel kinds proven on the CPU: PR #24, issue #23,
  squash `94aea7e` (Gest parent `nsqzpzxv`, iteration `zrtkpqzm`); the docs split and the site
  followed in the next PR. The docs are a GitHub Pages site built by MkDocs (Material) from `docs/`
  by `.github/workflows/docs.yml` on every push to `main` that touches them (`mkdocs.yml` holds
  the nav; `docs/index.md` is the home; Mermaid renders through superfences; GitHub's own
  Jekyll build was tried first and fails on the Mermaid `{{"..."}}` node syntax, which Liquid
  parses); `uvx --with mkdocs-material mkdocs build` builds it locally. `internal_docs/` holds
  the handoffs, the workflow docs and the cheat sheets and is not on the site.
- **M9 was re-scoped mid-session** (Rahul: no pod runs while Fable's weekly quota was at 99%
  and the session billed the API). The GPU runs and the driver's per-pod GPU count are M10;
  their Gest leaves `rslyvvkq` (the runs) and `sxytmvrk` (the GPU count) still sit under the M9
  parent and must be moved under an M10 parent by `gpl` (M8's shape: a depth-1 parent, an
  iteration with phases, one GitHub issue).
- **Nothing ran on a GPU in M9.** RunPod: every pod terminated since M8; the bucket
  `distrainer-rahuldave` and the GHCR image `ghcr.io/rahuldave/distrainer-gpu:latest` are as
  M8 left them. **Rotate the RunPod API key** before renting anything (the M8 handoff's warning
  still stands).

## 2. Environment checklist

```bash
just setup && just verify                       # laptop gate (246 unit tests, smoke)
just test tests/test_procgroup.py               # 10 tests on real ranks over Gloo, about 90 s
PROCGROUP_DEBUG=1 PROCGROUP_TIMEOUT_S=60 just test tests/test_procgroup.py -k diloco
uv run python examples/hello_blocks/train.py --config examples/hello_blocks/local.yaml --set parallel.kind=diloco
uv run python examples/image_contrastive/train.py --config examples/image_contrastive/local-synthetic.yaml --set parallel.kind=fsdp
```

The RunPod lines of `internal_docs/handoff-m9.md` section 2 are unchanged for M10.

## 3. What M9 delivered

- `tests/procgroup.py`: `run_world(cfg, n, step, build_model, out_dir, checkpoint=None,
  loop_extra=None)` spawns `n` ranks (torch.multiprocessing, spawn) that join a Gloo group on
  a free localhost port, patch the `ray.train` names `train_loop` looks up at call time onto the
  group (`prepare_model` the real DDP wrap when the kind asks for it, `barrier` and
  `broadcast_from_rank_zero` as `torch.distributed` calls, `report` recorded per rank,
  checkpoints returned as directories, `merge_checkpoint` doing what Ray does with the ranks'
  directories). A stuck rank dumps every thread's stack after the deadline.
- `parallel:` section (`ParallelConfig` in `config.py`): `kind` ddp | none | local_sgd |
  diloco | fsdp, DiLoCo's `outer_lr` / `outer_momentum` / `outer_nesterov`, FSDP's
  `reshard_after_forward` / `param_dtype` / `reduce_dtype`; `TrainInfo.parallel`.
- `distrainer/parallel.py`: the `SegmentSync` protocol, `LocalSGD`, `DiLoCo` (anchor + outer
  Nesterov SGD, state in `parallel.pt`), `build_sync`, `wrap_fsdp` (FSDP2 `fully_shard` per
  direct child then the root; the optimizer re-pointed at the sharded parameters), `is_sharded`.
- The loop: the sync on every rank right after the last step of a segment, before the
  policy's checkpoint and rank 0's hooks; under fsdp every rank reports its checkpoint shard.
- `CheckpointIO`: two shapes (full from rank 0; sharded, one DCP file per rank merged by Ray
  under one `checkpoint_dir_name`), shape detection and resharding on load, driver-side loads
  without a group, `shape()` from the listing, `distrainer inspect` printing it.
- Docs: spec sections 4, 5, 6.2, 7, 12; `docs/cli.md`; `docs/parallelism.md` brought to what
  is built; tutorial 7; the site; this file.

Numbers from the CPU (hello_blocks, 2 workers, 4 segments, the toy MSE): final loss 0.049 under
ddp, 0.052 local_sgd, 0.338 diloco (a tiny problem with few syncs; expected, not a defect);
the image example on the synthetic set: fsdp equal to ddp in loss and probe (kNN 1.000).

## 4. Behaviours learned in M9 that will bite again

- **`torch.multiprocessing.ProcessContext.join(timeout)` returns as soon as one rank has
  finished**, not when all have; it must be looped until it returns True. Read as a hang for
  an hour of the session.
- **A mid-segment checkpoint under local_sgd or diloco holds rank 0's drifted replica.** A
  resume from it restarts every rank from that replica (positions exact, the others' drift
  lost). A segment-end checkpoint resumes exactly. `checkpoint.policy: segment_end` is the
  setting for those kinds; the spec, the module and tutorial 7 say so.
- **`fully_shard` swaps the module's parameters for DTensor-backed ones under the same
  names**, so an optimizer built before the wrap no longer references the model's parameters
  (`get_optimizer_state_dict` fails with a KeyError). `wrap_fsdp` re-points the param groups by
  name; an optimizer that already has state is refused.
- **Ray Train v2 merges the checkpoint directories of every reporting rank** under one
  `checkpoint_dir_name` (its `report` docstring says so); the sharded shape relies on it. The
  process-group harness imitates it with `merge_checkpoint`.
- **FSDP2 and `torch.distributed.checkpoint` work on the CPU over Gloo** (torch 2.14), and a
  DCP directory loads into a plain module with no process group (a UserWarning about the
  single-process assumption, silenced in `CheckpointIO`). No GPU is needed to test fsdp.
- **The image encoder returns a view from the FSDP2-wrapped module**: torch warns that an
  in-place op on it would skip the all-gather. The results equal ddp; the fix is a `.clone()`
  (or a fresh tensor) at the end of `Encoder.forward` in `examples/image_contrastive/model.py`.
- **The toy in the tests diverges numerically** (lr 0.1 on `(w x)^2` with `x` up to 16: weights
  of 1e4 by segment 3); the tests assert determinism and structure, never convergence.
- **`distrainer inspect` on `checkpoint_manager_snapshot.json`** (which `ls checkpoint_*`
  matches) raises a raw `NotADirectoryError` from `ledger.py`; a clear message is wanted.
- Everything in the M8 list still applies (`internal_docs/handoff-m9.md` section 4).

## 5. Review follow-ups still open

- The M8 list (`internal_docs/handoff-m9.md` section 5) is unchanged: the 7.5 GB image,
  `kill-worker` stopping the pod, the runner's lag settings as environment, the image
  example's `all_gather` path with no unit test (now possible: run `info_nce` on two ranks of
  the process-group harness), the personal bucket names, `wipe-shared` a no-op, the account
  key in every pod.
- From the post-merge `gpa` of PRs #24 to #26 (the review ran after the merges; the rest of
  its findings were fixed in the review-fixes PR): `parallel.average_` issues one blocking
  all-reduce per tensor, about 160 transatlantic round trips per sync for a ResNet, which eats
  the win DiLoCo exists for -> coalesce into one flat buffer per sync
  (`torch.distributed.all_reduce_coalesced` or `_flatten_dense_tensors`) before the pod runs;
  the fsdp checkpoint assertions run through the harness's `merge_checkpoint`, so Ray's real
  merge of the ranks' directories and the ASYNC per-rank upload have no automated coverage ->
  one `integration_tests/single_node` scenario under `parallel.kind: fsdp` asserting
  `CheckpointIO.shape` on the stored checkpoint; `free_port()` in the harness can collide
  under concurrent runs.
- From M9: the encoder view (section 4); the `inspect` message (fixed in the review-fixes PR); the fsdp shard-unit policy is
  fixed at "every direct child" with no knob; a per-rank checkpoint shape (the DCP machinery)
  would make mid-segment resumes exact under local_sgd/diloco; `examples/*/harness*.yaml`
  carry no `parallel:` section yet (the default applies).

## 5b. The docs audit against the API (run 2026-09-17, fixes deferred to M10)

An Opus reviewer compared every API mention in the docs and the examples with the code at
`86a0a16` (after PR #27). The site now has `docs/api.md`, generated from the docstrings by
`mkdocstrings` at each build (static analysis, no install; `distrainer/**` triggers the
docs workflow), so the signatures there are the code's; the hand-kept surfaces below drifted.
Line numbers are from that commit; re-check them before editing.

**The spec (`docs/distrainer-spec.md`), section 4, the interface block.** Missing or wrong:
`BlockLog.open`, `exists`, the `W` property, `first_seq`, `committed_seqs`, `segments`,
`segment_path`, `next_pass_differs`; `wait_segment`'s `stop` argument (:216); `resume_start`'s
`W=` and the planner's `steps_per_segment` / `replayed_positions` (:241); `Ledger.run_attempt`
and its `asdict` / `to_json` / `from_dict` / `from_json`, `save` returns the path (:254-256);
`StepContext`'s `segment`, `pass_idx`, `world_size` (:261), the optional `on_checkpoint`,
`Never`, `build_policy`, `notify_checkpoint` (:260-266); `hook_specs` (:287); the `parallel.py`
header naming fsdp (:289), `DiLoCo`'s defaults (:295), `is_sharded` / `average_` /
`group_size` (:299); `CheckpointIO.__init__`, `read_ledger`, `load` raising for a directory of
neither shape (:301-303); `BatchWriter.__init__(tail, verify_blocks)`, `run` returning the
segments, `plan`, and `StreamingWriter` altogether (:310-311); **`DistTrainer.__init__` is
wrong** (no `log` argument, `config` third, `policy` optional, `hooks`,
`resume_from_checkpoint`, `scaling_config`, `run_config`; :315-317) and its
`scaling_config` / `run_config` / `loop_config` / `trainer` / `run_dir`; **`TrainInfo` is
referenced but never defined** (:280, :291, :319-320: `ctx` is a `TrainInfo`, with `train` and
`parallel` properties), nor `unwrap`, `parallel_strategy`, `checkpoint_dir_name`,
`train_loop`, `init_ray`.

**Section 5, the loop pseudo-code**, predates M2's report batching and M9: `build_model(info)`
with a `TrainInfo`, `BlockLog.open` and `steps_per_segment` (:329-330), `CheckpointIO.load(ckpt,
model, opt, sync)` (:333), `resume_start(ledger, n, W=W)` (:334), `step_info` in `train_step`
and the sync (:345, :349), `segment_end = cursor == steps` (:348), the `io.save(..., sync=,
sharded=, rank=)` call with `checkpoint_dir_name` and `delete_local_checkpoint_after_upload`
(:352), the `elif report_every_step` branch and the final flush instead of `report(metrics)`
every step (:354-355, contradicting the note at :364), the `keep_from > 0` guard (:359), the
`finally: loader.close(); audit.close(); io.cleanup_uploaded()` (:361).

**Section 6.2** (:390): the scratch path is `<tempdir>/distrainer/<run_name>/<uuid8>/
<checkpoint_dir_name>_<uuid6>`, not `/tmp/distrainer/<run>/ckpt_<cursor>`; nothing "deletes
local dirs older than the last two" (Ray's `delete_local_checkpoint_after_upload` plus
`cleanup_uploaded`); `load` returns a `Ledger`, not `(state, Ledger)`; add `shape` and
`read_ledger`. **Section 6.4 / the CLI** (:421): `inspect` prints the shape; `gc` has
`--keep-blocks`; every command but `resume` turns a bad path or a checkpoint of neither shape
into one stderr line and status 1.

**Other docs.** `docs/tutorials/batch.md`: the `inspect` sample lacks `shape: full` (:189);
the checkpoint contents sentence (:179) and the `backward` comment (:98) should name the
default kind; `info.parallel` in the `TrainInfo` list (:103-105). `docs/tutorials/streaming.md`
(:148): the hook runs after the segment-end sync and the checkpoint. `docs/cli.md`: the
directory-contents sentence reads unconditional (:35); the one-line-error rule (:78); the
`--set parallel.kind=...` row in the scripts table (:96). `docs/collectives.md` (:149-152): the
DDP row is kind-specific, and two rows are missing (fsdp's all-gather / reduce-scatter; the
segment-end all-reduce under local_sgd / diloco, before the report row). `docs/parallelism.md`:
`no_sync` is not what the kinds use (:189), "implements today" (:215), the fsdp shard note
on rank 0's duties (:72-73). `docs/examples-and-scenarios.md` (:145): shape from the listing.
`docs/introduction.md`: "focuses on data parallelism" (:122), FSDP as future (:341),
`prepare_model(model)` (:145). `README.md` (:84-87): the status paragraph still ends at M8
and names `handoff-m9.md`.

**Examples.** `examples/hello_blocks/train.py`: the `backward` comment (:52) and a docstring
line that tutorial 7 drives it with `--set parallel.kind=...`; `examples/image_contrastive/
train.py` (:9-10): the probe's load takes either shape; `distrainer/__init__.py`: the
docstring should name the surface (the torch-free core re-exported; `distrainer.trainer` and
`distrainer.parallel` imported directly because they pull in Ray and torch). The other
example modules are accurate.

**Docstrings missing on public callables** (they show a bare signature on `docs/api.md`;
about half of the surface). User-facing: `trainer.py` (`MetricAggregator.__init__` / `add`,
`CheckpointIO.__init__` / `cleanup`, `DistTrainer.__init__` / `scaling_config` /
`run_config` / `loop_config` / `trainer` / `fit`); `config.py` (every section dataclass but
`ParallelConfig`, `as_policy_dict`, `min_workers` / `max_workers` / `elastic`,
`DistrainerConfig` and `from_dict` / `from_yaml` / `allowed_world_sizes` / `validate`,
`load_config`); `parallel.py` (the protocol's four methods, `LocalSGD`'s four, `DiLoCo`'s
`__init__` / `on_segment_sync` / `state_dict` / `load_state_dict`); `hooks.py`
(`SegmentHook.on_segment_end`); `policy.py` (`CheckpointPolicy` and every policy class and
method); `log.py` (`segment_filename`, `LogMeta`, `Segment` and their `to_json` / `from_json`
/ `positions`, `BlockLog.__init__` / `open` / `exists` / `meta` / `W` / `segment_path` /
`has_segment` / `read_segment` / `ended`); `block.py` (`BlockRef`, `to_dict` / `from_dict`,
`validate_block_id`, `block_locator`, `validate_locator`, `read_block`); `ledger.py` (`Ledger`
and its six methods); `loader.py` (`LaneLoader`, whose docstring the spec quotes at :270 does
not exist in the code); `writer.py` (`BatchWriter.__init__` / `run`,
`StreamingWriter.__init__` / `buffered` / `close`); `planner.py` (`steps_per_segment`).
Internal: `audit.py`, `storage.py`, `cli.py`'s command functions.

The order for M10: docstrings first (they fix the API page for free), then section 4 of the
spec rewritten against `docs/api.md` (keep it a contract, not a copy: the shapes and the
rules, pointing at the API page for signatures), then sections 5 and 6, then the other docs
and the examples, then `just verify` and `mkdocs build --strict`.

## 6. M10

1. `gpl` an M10 parent and iteration from this file; move `rslyvvkq` and `sxytmvrk` under it,
   and add a docs leaf for section 5b (the docstrings and the spec's section 4 first).
2. The driver's per-pod GPU count (`sxytmvrk`): `create_pod`'s `gpu.count` from
   `DISTRAINER_RUNPOD_GPU_COUNT`, a `DISTRAINER_TRAINERS` value in the worker pod's env that
   `deploy/ray-worker.sh` turns into `--num-cpus=k --resources='{"trainer": k}'` (the head keeps
   advertising no trainer), `ps` showing the count; stub-API tests in
   `tests/test_runpod_driver.py` mirroring `test_up`'s create-payload assertions.
3. The runs (`rslyvvkq`), detached with `nohup`, one at a time, pods kept for the session:
   the image example under `diloco` on CA-MTL-1 + EU-RO-1 against M8's 2.5 s per step and
   probe 0.385; S2, S3, S4 under `local_sgd` in EU-RO-1; the image example under `fsdp` on one
   two-GPU pod. Budget as M8 (30 USD ceiling) unless Rahul says otherwise; key rotation first.
4. The follow-ups of section 5 that the runs touch: `kill-worker` through the drain marker,
   the runner's lag defaults per driver, the `all_gather` test, the encoder view.
5. Record every number in `internal_docs/handoff-m11.md`; docs to `docs/` (the site) only when
   user-facing.
