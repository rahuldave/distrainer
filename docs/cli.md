# Command-line reference

Everything you type at a shell to use distrainer: the `distrainer` command, the example scripts,
the scenario checker, and the `just` targets that wrap them. Tutorials 1 and 2
(`docs/tutorials/`) show these in context; `docs/examples-and-scenarios.md` says what each
example computes.

## `distrainer` (spec section 6.4)

Installed by `uv sync` as a console script; run it as `uv run distrainer <command>` (or
`python -m distrainer.cli`). Every `<uri>` or `<store>` argument is a local path, a `file://`
URI, or `s3://bucket/prefix`. For S3-compatible stores (MinIO, R2) set `S3_ENDPOINT` (and
optionally `S3_REGION`, default `auto`) plus the credentials `S3_ACCESS_KEY` and
`S3_SECRET_KEY` in the environment; the harness containers have all four set.

### `distrainer inspect <checkpoint uri>`

Print the ledger of a checkpoint from its metadata, without downloading the weights.

```
$ uv run distrainer inspect runs/hello/hello/checkpoint_g000003_p000008_n02_a00
checkpoint: runs/hello/hello/checkpoint_g000003_p000008_n02_a00
ledger: {'segment': 3, 'cursor': 4, 'world_size': 2, 'pass_idx': 0, 'run_attempt': 0}
done positions in segment 3: 8
written by distrainer 0.0.1
```

Checkpoint directories are named `checkpoint_g<segment>_p<positions done>_n<world size>_a<attempt>`
and hold `model.pt`, `optimizer.pt` and `ledger.json`.

### `distrainer export <checkpoint uri> <dir>`

Copy a checkpoint (weights, optimizer state, ledger) to a local directory, for example out of a
bucket.

### `distrainer resume <checkpoint uri> --config <yaml> --entry <pkg.module:function> [--run-name NAME] [--seed N]`

Start a **new** run from a checkpoint. `--entry` names a function `f(cfg) -> (train_step,
build_model)` (the examples ship one: `examples.hello_blocks.train:entry`,
`examples.toy_contrastive.train:entry`). The run name defaults to `<config run_name>_resume`
because a name that already exists on the storage would be *restored* by Ray Train instead of
started afresh. Everything else, including the number of workers and any `hooks:`, comes from
the config: the resumed run reads the checkpoint's segment, skips the positions the ledger says
are done (rounded down to a step boundary of the *current* world size), and continues to the end
of the log. Resuming into a window that `gc` has deleted is refused with a clear error.

```bash
uv run distrainer resume runs/hello/hello/checkpoint_g000001_p000012_n02_a00 \
    --config examples/hello_blocks/local.yaml \
    --entry examples.hello_blocks.train:entry --run-name hello_resume
uv run distrainer resume s3://distrainer/runs/s9/checkpoint_g000004_p000024_n02_a00 \
    --config examples/hello_blocks/harness-minio.yaml \
    --entry examples.hello_blocks.train:entry --run-name s9_resume       # inside the head container
```

### `distrainer log-ls <store> [-v]`

List a block log: `W`, seed, who created it, the committed segment range, whether `_END` is
present; `-v` adds one line per segment with its pass, positions and first block ids.

```
$ uv run distrainer log-ls blocks/hello_stream -v
log: /.../blocks/hello_stream/log
W=12 seed=7 schema_version=1 by streaming_producer
segments: 3 (5..7)
ended: True
  00000005 pass=0 positions 60..71: p000063, p000064, p000065, p000066, ...
  ...
```

Exit status 1 when there is no log under `<store>`.

### `distrainer gc <store> --keep-from N [--keep-blocks]`

Delete the segment files below sequence number `N` and, unless `--keep-blocks` is given, the
block files that no remaining segment references. This is the manual form of what the trainer
does at every segment end when `log.gc: true` (`N` = segment of the last checkpoint minus
`log.retention_segments`). Prints the deleted sequence numbers. `docs/retention.md` has the
rules and the interaction with checkpoints.

## Example scripts

All of them take `--config <yaml>` (spec section 7). Paths in a local config are made absolute
when it is loaded, so the scripts can be run from anywhere; inside the harness containers or pods run
them from `/app`.

| script | flags | what it does |
|---|---|---|
| `examples/hello_blocks/train.py` | `--config`, `--set key.path=value` (repeatable, YAML values), `--keep`, `--no-check` | wipes the previous run of the same name (local storage only, unless `--keep`), builds a batch log if the store is empty, trains, prints the final metrics and ledger, and runs the S1 audit check plus the report-count check unless `--no-check` |
| `examples/hello_blocks/make_blocks.py` | `--config` | builds the hello_blocks corpus and batch log (no Ray needed) |
| `examples/toy_contrastive/train.py` | same flags as hello_blocks | same flow with the InfoNCE encoder; with `hooks.remine` configured a fresh run also wipes the store, because a streamed log belongs to one run |
| `examples/toy_contrastive/make_blocks.py` | `--config` | builds the mined corpus with Ray Data and the batch log; with `hooks.remine` configured writes only the first `initial_segments` segments and leaves the log open |
| `examples/streaming_producer/produce.py` | `--config`, `--store`, `--W`, `--seed`, `--segments N` (default 8), `--sleep-s S` (default 5), `--shuffle-buffer k`, `--rows`, `--features` | creates a log and streams `N` segments of hello_blocks-style blocks into it, sleeping `S` seconds after every segment committed while pushing (the `k-1` segments flushed at the end commit back to back), then writes `_END`; prints `segment <n> committed at <time>` per commit. `--config` supplies `store_root`, `seed`, `log.W`, `log.shuffle_buffer_segments`, `train.rows_per_block` and `train.features`; the flags override it, `--store` also its store; without a config the defaults are `W=24`, `seed=7`, `k=1`, 32 rows, 8 features and `--store` is required. Refuses a store that already has a log |

`--set` is how the scenario runners reuse one config file:

```bash
uv run python examples/hello_blocks/train.py --config examples/hello_blocks/local.yaml \
    --set run_name=s5 --set checkpoint.policy=every_k --set checkpoint.every_k=4 \
    --set checkpoint.num_to_keep=null
```

## Scenario checker

`integration_tests/cluster/check_audit.py` is a library of pure checks (see
`docs/examples-and-scenarios.md`) with a small CLI for the ones a resumed or remote run needs:

```
uv run python integration_tests/cluster/check_audit.py --store-root <store> --run-name <name> --W <W> \
    --scenario S1
    --scenario S5 --run-uri <run dir> [--every-k K] [--expected-checkpoints N]
    --scenario S7 --other-run-name <name>
    --scenario resume (--start-position P | --ledger-segment G --ledger-positions P) [--expected-segments N]
```

It prints the audit summary, one `... FAIL: <problem>` line per problem and `<scenario> PASS` or
`FAIL`; exit status 1 on failure. `<store>` may be an `s3://` URI (S9 and S10 run it inside the
head container against MinIO).

## `just` targets (the command contract)

| target | runs |
|---|---|
| `just setup` | `uv sync --all-groups` |
| `just fmt [path]`, `just lint [path]` | `ruff format`; `ruff check`, `ruff format --check`, `bash -n` over `deploy/` |
| `just typecheck`, `just static` | `ty check`; `compileall` |
| `just test [target]`, `just regression` | `pytest tests` (or a path); `pytest regression_tests` |
| `just smoke` | hello_blocks on a local Ray cluster with the S1 check (the gate) |
| `just contrastive [CFG]` | toy_contrastive; `examples/toy_contrastive/local-remine.yaml` streams the log through the re-mining hook |
| `just local-scenarios [S]` | S1, S5, S7 on a local Ray cluster |
| `just verify` | lint, typecheck, static, test, regression, smoke, `git diff --check` |
| `just build` | the harness image (again when `uv.lock` or `deploy/Dockerfile` changes) |
| `just up [N]`, `just up-minio [N]`, `just down`, `just nuke` | head + N worker containers (+ MinIO); stop; stop and remove volumes and the shared mount |
| `just mkbucket` | create the `distrainer` bucket on MinIO |
| `just blocks [CFG]`, `just train [CFG]` | `make_blocks.py` / `train.py` of hello_blocks inside the head container |
| `just kill-worker I`, `just scale N` | kill worker `I` (compose: restarted after `DISTRAINER_RESTART_DELAY` s; uncloud: `docker kill` over ssh on its machine, restarted the same way; kuberay: replaced by the operator at once); resize the worker set |
| `just integration [S]` | cluster scenarios S2, S3, S4, S6, S8, S9, S10, S11, S11s3 (or `all`) |
| `just kuberay-operator` | install the KubeRay operator into the current Kubernetes context (once; `DISTRAINER_DRIVER=kuberay` for the targets above) |
| `just uncloud-machines` | create the OrbStack machines and the uncloud cluster (once; `DISTRAINER_DRIVER=uncloud` for the targets above; `DISTRAINER_DRIVER=uncloud deploy/driver.sh machines-destroy` removes them) |
| `just aws-bucket`, `just aws-machines` | the AWS bed (tutorial 5): the S3 bucket with an IAM user scoped to it, then three EC2 instances joined as uncloud context `distrainer-aws`; `deploy/driver.sh machines-stop|start|status|destroy` with `DISTRAINER_DRIVER=uncloud` afterwards, `deploy/uncloud/aws.sh bucket-rm` for the bucket |
| `just docs` | list the docs |

`DISTRAINER_DRIVER` selects the harness driver (`compose`, the default; `kuberay` for pods on
OrbStack's Kubernetes; `uncloud` for machines joined by uncloud's mesh), `DISTRAINER_MINIO=1` adds
MinIO, `DISTRAINER_SHARED` moves the shared mount (compose, kuberay), `DISTRAINER_IMAGE` names the
image tag (every driver), `DISTRAINER_K8S_CONTEXT` and `DISTRAINER_K8S_NAMESPACE` pin where the
KubeRay driver acts (`orbstack`, `distrainer`), and the uncloud driver reads
`DISTRAINER_UNCLOUD_CONTEXT` (`distrainer`), `DISTRAINER_UNCLOUD_MACHINES` (`uc1 uc2 uc3`, the first
is the head machine), `DISTRAINER_UNCLOUD_SSH` (`%s@orb`, the ssh destination template for a
machine), `DISTRAINER_UNCLOUD_HOST_PREFIX` (the CIDR the head machine publishes ports on) and
`DISTRAINER_UNCLOUD_HEAD_ADDRESS` (an override for `endpoint`), plus `DISTRAINER_UNCLOUD_SSH_OPTS`
(extra ssh options), `DISTRAINER_UNCLOUD_PROVIDER` (`orbstack` or `aws`: whose bootstrap the
`machines-*` verbs drive) and `DISTRAINER_ENV_FILE` (a bootstrap's env file, read after `.env` by
the uncloud driver and the scenario runner; what the shell exports wins over both files). The driver
verbs behind these targets are documented at the top of `deploy/driver.sh`.
