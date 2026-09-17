# Tutorial 7: the parallel kinds (DDP, local SGD, DiLoCo, FSDP) on the same block log

The loop of tutorials 1 to 6 wraps the model in DDP and all-reduces the gradients on every
step. Since M9 that is one of five settings of the `parallel:` section of the config; the log,
the dealer, the ledger, the audit trail and the scenarios are the same under all of them.
`docs/parallelism.md` explains the kinds with the collectives they use; this page is how to run
them.

## 1. Choosing a kind

```yaml
parallel:
  kind: ddp            # ddp | none | local_sgd | diloco | fsdp
  outer_lr: 0.7        # diloco: the outer Nesterov SGD over the anchor
  outer_momentum: 0.9
  outer_nesterov: true
  reshard_after_forward: true   # fsdp
  param_dtype: null    # fsdp mixed precision: fp32 | bf16 | fp16 | null
  reduce_dtype: null
```

| kind | the wrap | per step | at the segment end | the checkpoint |
|---|---|---|---|---|
| `ddp` (default) | `DistributedDataParallel` | the gradients all-reduced inside `backward` | nothing new | full, from rank 0 |
| `none` | none | nothing: the ranks train independently | nothing | full, from rank 0 (its replica) |
| `local_sgd` | none | nothing | every rank averages the parameters (and the float buffers) | full, taken after the sync |
| `diloco` | none | nothing | the averaged change since the last sync feeds an outer SGD with Nesterov momentum over the *anchor*; the model takes the anchor | full, plus `parallel.pt` (the anchor and the outer optimizer) |
| `fsdp` | FSDP2 `fully_shard` per top-level child | all-gather to use, reduce-scatter after | nothing new | sharded: one `torch.distributed.checkpoint` file per rank, re-cut on load |

Every example takes the kind from the command line, so nothing else changes:

```bash
uv run python examples/hello_blocks/train.py --config examples/hello_blocks/local.yaml --set parallel.kind=local_sgd
uv run python examples/hello_blocks/train.py --config examples/hello_blocks/local.yaml --set parallel.kind=diloco --set checkpoint.policy=segment_end
uv run python examples/image_contrastive/train.py --config examples/image_contrastive/local-synthetic.yaml --set parallel.kind=fsdp
```

All three end with `S1 PASS`: the audit trail is checked the same way whatever the kind.

## 2. Local SGD and DiLoCo: the segment is the sync unit

Under `local_sgd` and `diloco` no collective runs during a segment. The sync happens on every
rank right after the last step of a segment, *before* the policy's checkpoint (so a segment-end
checkpoint holds the synced weights) and before rank 0's writer hooks. `H`, the number of local
steps between two syncs, is therefore `W / n`: choose `log.W` for the sync interval you want. A
resize at a segment end hands the new ranks the synced weights like any other resume.

Two things follow from "rank 0 writes the checkpoint":

- A checkpoint taken *inside* a segment holds rank 0's replica, which has drifted from the
  others'. A resume from it restarts every rank from that replica: the positions stay exact
  (no block is replayed or skipped), the other replicas' drift is lost. `checkpoint.policy:
  segment_end` makes every checkpoint a synced one, which is the exact choice under these kinds.
- DiLoCo's state (the anchor and the outer momentum) rides in the checkpoint as `parallel.pt`
  and is restored on every rank; a checkpoint written under another kind resets the anchor to
  the loaded weights.

DiLoCo's defaults are the paper's (outer lr 0.7, Nesterov momentum 0.9, Douillard et al.
2023). With `outer_lr: 1` and `outer_momentum: 0` it is local SGD.

## 3. FSDP: shards in memory, one checkpoint file per rank

`fsdp` wraps every direct child of the model that has parameters, then the root, with FSDP2's
`fully_shard` over a mesh of the ranks. The parameters become DTensors holding one shard each;
the optimizer `build_model` returned is re-pointed at them (it must have no state yet, which is
the case for a freshly built one). Mixed precision and the reshard policy come from the section.

Checkpoints change shape: every rank saves its shard with `torch.distributed.checkpoint` into
a directory of the same name and reports it, and Ray Train merges the ranks' directories under
one `checkpoint_dir_name`; rank 0 alone adds the ledger and the metadata.

```
$ uv run distrainer inspect runs/hello/hello/checkpoint_g000003_p000012_n02_a00
ledger: {'segment': 3, 'cursor': 6, 'world_size': 2, 'pass_idx': 0, 'run_attempt': 0}
done positions in segment 3: 12
shape: sharded (2 shards)
```

A resume at another world size re-cuts the shards on load; a driver with no process group (the
image example's kNN probe, `distrainer resume`) loads the same directory into a plain module;
a full checkpoint loads into a sharded model (the optimizer starts afresh, with a warning). The
image example under `fsdp` on the CPU ends with the same loss and probe accuracy as under `ddp`.

## 4. The tests behind this page

`tests/procgroup.py` runs `n` ranks as real processes over a Gloo group on localhost with the
same block stores the unit tests build; `tests/test_procgroup.py` asserts, under every kind,
what the single-process fixture cannot see: the gradient averaging, the segment-end average
landing in the checkpoint, DiLoCo's state round-tripping, FSDP's shards and their re-cut.

```bash
just test tests/test_procgroup.py          # about a minute and a half
PROCGROUP_DEBUG=1 just test tests/test_procgroup.py -k local_sgd   # each rank's progress on stderr
```

## 5. What comes next

The kinds were built and proven on the CPU. The runs they are for are M10: DiLoCo across the
transatlantic RunPod pods (2.5 s per step under DDP against 0.15 s alone in M8: the gap local
SGD closes) and FSDP on one pod with several GPUs, through a per-pod GPU count in the driver.
