# Tutorial 6: an image contrastive example, its GPU image, and RunPod pods

The M8 tutorial, written slice by slice as the milestone lands. Sections 1 to 4 cover what is
done: the example and the machine learning in it, running it on a laptop, the GPU image and the
pipeline that builds it. Sections 5 and 6 (the RunPod driver, the run on pods) are written with
their slices; until then `docs/handoff-m8.md` section 6 holds the plan, the API facts and the
cost estimate. The mechanics of every example and scenario are in
`docs/examples-and-scenarios.md`; the flags in `docs/cli.md`.

## 1. Why this example exists

`examples/toy_contrastive` (spec section 8) is a two-layer MLP on 32 synthetic features: a GPU
changes nothing for it. `examples/image_contrastive` keeps the same harness shape (blocks of
rows in Parquet, segments of `W` blocks, one `ray.train.report` per step, the checkpoint policy,
resume and the elastic resize, the loss-side `all_gather`) and changes what a row is: an image.
A ResNet on images is where a step takes seconds on a CPU and well under a tenth of a second on a
modest GPU, which is what the RunPod stage is meant to show. On the laptop the same code runs a
tiny encoder on a subset, so the unit tests and the smoke stay quick.

## 2. The machine learning

**Self-supervised contrastive learning (SimCLR).** No label is used in training. Every image is
turned into two random *views* by augmentation; the encoder must map the two views of one image
close together and the views of every other image far away. A network that manages this has
learned features that separate objects without ever being told what they are, which is what the
probe at the end measures.

**The augmentations** (`model.augment`), applied per image, each with its own random draw:

- *random resized crop*, the one that matters most: a window with 20 to 100 percent of the area
  and an aspect ratio between 3:4 and 4:3, resampled back to 32 by 32. Two crops of the same
  image share the object but not the pixels, so the encoder cannot match views by pixels;
- *horizontal flip* with probability one half;
- *colour jitter* with probability 0.8: brightness, contrast and saturation each scaled by a
  factor in 0.6 to 1.4 (hue is skipped, it is expensive and matters least);
- *grayscale* with probability 0.2, which stops the encoder from matching views by colour alone.

They are implemented as batched tensor operations on the worker's device (one `grid_sample`
for the crop and flip, a few broadcasts for the colours), not as per-image PIL calls, so two
views of a 256-image block cost milliseconds on a GPU. The random parameters come from a CPU
generator seeded from the run's `seed` and the block's position, so a position replayed after a
restart sees the same views.

**The encoder** (`model.Encoder`) is ResNet-18's topology with the CIFAR stem: a 3 by 3 first
convolution and no max-pool, because a 32 by 32 image cannot afford ImageNet's 7 by 7 stride-2
stem plus a pool (it would be 8 by 8 before the first block). `width` and `layers` are knobs:
`64, [2, 2, 2, 2]` is ResNet-18 (11.5 M parameters, 512-wide features); the laptop configs use
`16, [1, 1, 1, 1]` (0.3 M parameters). A two-layer *projection head* follows. The loss is computed
on the L2-normalised head output and the probe reads the backbone features before the head:
SimCLR's finding is that the head absorbs the invariances the loss demands (to colour, to crop)
that the downstream task may still need.

**The loss** (`model.nt_xent`) is NT-Xent, the normalised-temperature cross-entropy: for each
view, a softmax over cosine similarities divided by `temperature` (0.1) where the target is the
image's other view and every other image in the batch is a negative, averaged over both
directions. It is the toy example's `info_nce` with an empty hard-negative tensor, so the code
is shared. With `train.all_gather: true` the positives of every rank are gathered before the
softmax (the `all_gather` inside `info_nce`), so each anchor also sees the other ranks' images
as negatives: more negatives per step, and the loss-side collective the spec's section 8 calls
for. At chance the loss is the log of the number of candidates; it falls as the encoder learns.

**Mixed precision.** On a CUDA device the forward runs under bf16 autocast (`train.amp: true`)
and the loss in float32. Both views go through the encoder in one forward pass, so BatchNorm
sees both, as in the reference implementation.

**The probe** (`probe.py`) is the run's sanity metric. A weighted k-nearest-neighbour classifier
on the *backbone* features: the first `probe_train` rows of the log, labels included, are the
memory bank; the first `probe_test` held-out images are the queries; each query's `probe_k`
nearest bank features by cosine similarity vote for their class with weight `exp(sim / 0.1)`.
It needs no training of its own, which is why it is the standard monitor for contrastive
methods. Chance is 0.10 on CIFAR-10. What to expect: an untrained encoder lands a little above
chance (random features still cluster colours), the tiny encoder after 48 positions of the
laptop config reaches about 0.29, SimCLR on the full ResNet-18 with a long schedule reaches the
high eighties; the point of a run here is a clearly rising number in minutes, not a benchmark.

**How the data is stored.** Every image is one row of a Parquet block: `item_id`, `label`, and
`image`, a binary column holding the PNG bytes inline (not a path, not a flattened pixel
array). A block of 256 rows is one self-contained file of about 750 KB, the unit the workers
fetch. The label rides along unused by training; the probe reads it. A pointer per row would
mean 256 more fetches per step; raw pixels would be a wash for CIFAR (a 32 by 32 PNG is 2.9 KB
against 3 KB raw) but lose for bigger images. The held-out split goes to
`<store_root>/probe/test.parquet`, outside the log so retention gc never touches it.

## 3. On the laptop

```bash
just images                                                    # a 3072-image CIFAR-10 subset
just images examples/image_contrastive/local-synthetic.yaml    # no download
uv run python examples/image_contrastive/train.py --config examples/image_contrastive/local.yaml --rebuild
```

`local.yaml`: the tiny encoder on 3072 CIFAR-10 training images in 24 blocks of 128, `W=12`,
2 passes, 2 workers, a checkpoint every 4 steps, the probe on 1000 held-out images against 1024
training images. The first run downloads CIFAR-10 (170 MB) through torchvision into
`.harness/datasets`; the Toronto server is slow (40 minutes at 72 KB/s on 2026-09-16), every
later run finds the cache. Then blocks and the probe split take 4 s to write, training, the
probe and the audit check 14 s: 71 s in all, `probe: knn_acc=0.290 (chance 0.10)`, `S1 PASS`.

`local-synthetic.yaml`: class-coloured striped noise from `data.synthetic_images` instead of a
download (the unit tests use it too), 12 blocks of 128 seen in 2 passes so the run crosses a
pass boundary, about 50 s, the probe at 1.0 because the classes are colours.

`--rebuild` wipes a local store first: a changed `n_items`, `rows_per_block`, `n_test` or `seed`
would otherwise reuse the existing log silently, since `make_blocks` is idempotent per store.
`--no-probe` skips the probe. `make_blocks` refuses a log that exists with no segments, which
an interrupted build leaves behind, and tells you to rebuild.

`harness-s3.yaml` is the GPU shape for section 6: 49 152 of the 50 000 training images in 192
blocks of 256 (8 segments of `W=24` per pass), 2 passes, ResNet-18, `use_gpu: true` with
`resources_per_worker: {GPU: 1, trainer: 1}`, bf16 autocast, a checkpoint every 8 steps (each
is a one-second upload), everything on the bucket of `examples/hello_blocks/harness-s3.yaml`.

## 4. The GPU image

### 4.1 What it is

`deploy/Dockerfile.gpu`, built for `linux/amd64` only, pushed to
`ghcr.io/rahuldave/distrainer-gpu` (a public package: RunPod pulls it with no credentials). It
is `deploy/Dockerfile` (the CPU node image) with two differences, and a test
(`tests/test_deploy_manifests.py`) keeps the rest identical (the base image, the uv version, the
`ENV` block):

1. **torch and torchvision are the CUDA 12.6 wheels of the versions `uv.lock` pins.** The
   locked `uv sync` runs with `--no-install-package torch --no-install-package torchvision`
   (their dependencies are still installed), then a layer of its own runs
   `uv pip install --index https://download.pytorch.org/whl/cu126 torch==2.14.0+cu126
   torchvision==0.29.0+cu126`. The `+cu126` local versions are spelled out because uv would
   hold a bare `==2.14.0` satisfied by the CPU wheel and install nothing. The project itself is
   installed last with `uv pip install --no-deps -e .`, because a second `uv sync` would put
   the CPU wheels back. The version `ARG`s at the top of the Dockerfile are pinned to `uv.lock`
   by the test, so a lock bump that changes torch fails the suite until the ARGs follow.
   No CUDA base image: the CUDA runtime libraries ride in the wheels (the `nvidia-*` packages)
   and the driver comes from the host through RunPod's NVIDIA runtime; the base stays
   `python:3.13-slim`.
2. **sshd**, started by the entrypoint `deploy/runpod-entry.sh` from the `PUBLIC_KEY` variable
   RunPod injects when a pod is created with `startSsh` (RunPod's own images do exactly this;
   an image that does not start sshd itself has no ssh access on RunPod). Host keys are made at
   build time (`ssh-keygen -A`). The entrypoint then writes the container's environment (the S3
   settings, `RAY_*`, `DISTRAINER_*`) to `/etc/rp_environment` and `/etc/profile.d/`, minus the
   key and minus the shell's own variables (`PATH`, `PWD`, `HOME`, ...), so a login shell
   (`bash -lc`) sees them; a login shell starts in `/root`, not in `/app`, so the driver's
   commands `cd /app` first. Finally the role: `head` runs `deploy/ray-head.sh`, `worker`
   runs `deploy/ray-worker.sh` (with `RAY_HEAD_ADDRESS` naming the head pod), anything else
   is exec'd, for debugging.

Why not a uv `gpu` extra (the first plan): an extra-conditional torch source collides with the
base CPU source in the Linux fork of the resolution unless torch leaves the base dependencies
for two conflicting extras (`cpu` and `gpu`), and uv has no default extras, so every plain
`uv run` on the laptop and in CI would then drop torch. Installing the CUDA wheels over the
lock keeps one lock and one set of commands.

### 4.2 How it is built

`.github/workflows/gpu-image.yml`, on GitHub's runners rather than a laptop (the image is
7.5 GB uncompressed, of which the CUDA layer is 6.6 GB; a home uplink pushes about a gigabyte
in ten minutes, the runner does it in a few):

- **Triggers**: a push to `main` or to a `gest/**` branch that touches the image's inputs
  (`deploy/Dockerfile.gpu`, the entrypoint and the two Ray scripts, `pyproject.toml`,
  `uv.lock`, `distrainer/`, `examples/`, `integration_tests/`, the workflow itself; not
  the docs, not `README.md`), and `workflow_dispatch` by hand from the Actions tab.
- **Tags**: `latest` on `main`, the branch name on a `gest/` branch (`gest-zvymnspr-runpod`
  while M8 is unmerged: that is the tag the driver's `DISTRAINER_IMAGE` names until the
  merge), and `sha-<short>` always.
- **Credentials**: the workflow's own `GITHUB_TOKEN` with `permissions: packages: write`; no
  secret to configure. The package came out public on the first push (the repository is
  public); check with `docker manifest inspect ghcr.io/rahuldave/distrainer-gpu:<tag>` from a
  machine that is not logged in.
- **Cache**: the GitHub Actions layer cache (`type=gha`), which holds the CUDA layer between
  runs unless evicted (the cache is capped at 10 GB per repository).
- **The smoke** after the push, on the runner (native amd64, no GPU): `docker run ... head`,
  wait 45 s, the container must still be up, `ray status` must answer, the torch versions must
  print, and `bash -lc "cd /app && python -c 'import distrainer'"` must work through the login
  shell path the driver uses.

The first run on 2026-09-16 (cold cache) took 15 min 47 s end to end and passed every step.

### 4.3 Building it locally, and what a Mac can and cannot check

```bash
docker build --platform linux/amd64 -f deploy/Dockerfile.gpu -t distrainer-gpu:local .
docker run --rm --platform linux/amd64 distrainer-gpu:local \
    python -c "import torch, torchvision; print(torch.__version__, torchvision.__version__, torch.version.cuda, torch.cuda.is_available())"
```

On the M1 with OrbStack the build took 7 min 09 s cold (3 GB of CUDA wheels) and 107 s with the
builder's cache warm, and the image runs under Rosetta well enough to import everything
(`2.14.0+cu126 0.29.0+cu126 12.6 False`, the `False` being no GPU) and to exercise the ssh path:

```bash
ssh-keygen -t ed25519 -N '' -f /tmp/smoke_key
docker run -d --platform linux/amd64 --name gpu-smoke -e PUBLIC_KEY="$(cat /tmp/smoke_key.pub)" \
    -e S3_ENDPOINT=http://smoke.example:9000 -p 2222:22 distrainer-gpu:local sleep infinity
ssh -i /tmp/smoke_key -p 2222 root@localhost 'bash -lc "pwd; echo $S3_ENDPOINT; which python; cd /app && ls deploy"'
docker rm -f gpu-smoke
```

What a Mac cannot check: the head role. Under Rosetta the raylet declares its dashboard agent
dead two milliseconds after starting it and exits (the agent's own log shows it starting fine;
the native arm64 CPU image stays up under the same test), so the verdict on `ray start --head`
comes from the workflow's smoke on a native runner, and the verdict on the GPU from the first
pod. Remove the local image afterwards (`docker rmi distrainer-gpu:local`); it is 7.5 GB.

### 4.4 The build's numbers

| item | value |
|---|---|
| base | `python:3.13-slim`, uv 0.9.8, the same as the CPU image |
| torch, torchvision | 2.14.0+cu126, 0.29.0+cu126 (the lock's versions) |
| image, uncompressed | 7.55 GB: CUDA layer 6.59 GB, the locked sync 747 MB, apt (procps, openssh-server) 29 MB, the code 1.2 MB |
| local build, M1 + OrbStack | 7 min 09 s cold, 107 s warm |
| Actions run (cold cache) | 15 min 47 s: build, push, the head-role smoke |
| pull on a RunPod pod | to be measured in section 6 (the compressed layers are about half the size) |

## 5. The RunPod driver

`deploy/drivers/runpod.sh`, selected with `DISTRAINER_DRIVER=runpod`, implements the verbs of
`deploy/driver.sh` on RunPod's REST v2 API (`https://api.runpod.io/v2`; v1 is retired on
2026-11-15) with `curl` and `jq`, so `just up 2`, `just integration S2`, `just kill-worker 1`
and the rest work unchanged. What a cluster is here:

- **Every Ray node is a GPU pod** on the secure cloud with global networking, which is what
  lets pods talk over TCP as `<pod id>.runpod.internal`. A CPU pod cannot join, so the head
  is a GPU pod too; it advertises no `trainer` resource and runs no training worker.
- **The store is the bucket**: the `S3_*` variables of `.env` are passed to every pod; the
  runner reads the audit trail and the checkpoints from the bucket through `endpoint`'s
  `s3=` line, `shared` prints nothing, `mkbucket` and `wipe-shared` have nothing to do.
- **The image** is `DISTRAINER_IMAGE` (section 4; the branch tag until M8 merges); `build`
  builds nothing and only checks that the tag is readable without credentials.
- **The driver acts only on pods it made**: named `<cluster>-head` and
  `<cluster>-worker-N` (`DISTRAINER_RUNPOD_CLUSTER`, default `distrainer`) *and* carrying
  `DISTRAINER_CLUSTER=<cluster>` in their environment; the account holds other people's pods
  and a name alone is not proof. `ps` lists the cluster with each pod's hourly cost and the
  total.

**Creating a pod.** v2 takes one GPU type per create, so `DISTRAINER_RUNPOD_GPU_TYPES` (an
ordered, comma-separated list) is tried type by type until one is rented; a type without a
price on the chosen cloud, or priced above `DISTRAINER_RUNPOD_MAX_GPU_HOURLY` (default 0.60
USD per GPU-hour), is skipped: the spend guard. The head is placed in any data center with
global networking (`GET /v2/catalog/datacenters`), or in `DISTRAINER_RUNPOD_DATA_CENTERS` if
set; the workers go to the head's data center, or to any global-networking one when that is
sold out (seen on the first run: the head took the last cheap card in EU-RO-1). The workers
are created right after the head, so the image pulls run side by side. A pod gets the image, `args` naming its role
(`head` or `worker`, what the entrypoint reads), `22/tcp` exposed and `startSsh` (RunPod
injects the account's registered ssh keys as `PUBLIC_KEY`; section 4.1), a
`DISTRAINER_RUNPOD_DISK_GB` container disk and no persistent volume, and for a worker
`RAY_HEAD_ADDRESS=<head pod id>.runpod.internal:6379`. Every create prints the pod's id, type,
data center and hourly cost; `up` waits until each pod is `RUNNING` with its global-networking
address and its ssh port (`DISTRAINER_RUNPOD_START_TIMEOUT`, default 1500 s per pod: the image
is 7.5 GB and the first pull in EU-RO-1 took more than ten minutes).

**Inside the pod** the entrypoint discovers the global-networking address (its own
`<id>.runpod.internal`, or the 10.x interface) and exports it as `RAY_NODE_IP`, which
`ray-head.sh` and `ray-worker.sh` pass as `--node-ip-address`: Ray must advertise that
address, not the container's default one, or the other pods cannot reach the ports GCS hands
out. `catalog` prints the configured types' prices and their availability in the
global-networking data centers, the thing to check before `up`.

**Breaking things.** Node death is RunPod's *stop*: the container is killed and the pod keeps
its host, its disk and its id, so a restart pulls no image (the pull is the slow part, section
6) and the head's address stays valid for the workers. `kill-worker I` stops worker I and,
after `DISTRAINER_RESTART_DELAY` seconds (default 5, 0 = stays dead), starts it again in the
background as a new Ray node; if the host refuses the start, a new pod of the same name is
created instead. `kill-head` stops the head; the next `up` starts every stopped pod of the
cluster again (an errored one is replaced). `scale N` creates the missing workers or
terminates the highest-numbered ones. `stop-worker I` is the same stop without the restart (a
preemption notice). `down` and `nuke` terminate every pod of the cluster; a stopped pod's disk
bills by the month (cents for a session), which is why `down` ends every session.

**Reaching the head.** `exec-head CMD...` is ssh to the head's published `22/tcp` port
(cached under `.harness/runpod/` by `up`), as `bash -lc "cd /app && CMD"`: a login shell
reads the exported environment and starts in `/root`, hence the `cd`. `cp-from-head` is scp.
`DISTRAINER_RUNPOD_SSH_KEY` names the key: its `.pub` is injected into every pod the driver
creates as `PUBLIC_KEY` (RunPod then skips the account's registered keys, so a shared account
needs no change and the pods admit only this machine), and it is the `-i` of both commands;
left empty, the account's registered keys are injected and this machine must hold one of them.
`DISTRAINER_RUNPOD_SSH_OPTS` adds options. The dashboard is not
exposed; `endpoint` prints the tunnel to open (`ssh -L 8265:127.0.0.1:8265`). `logs [NAME]`
fetches a pod's log from the API; `cost` prints the cluster's hourly total and the account's
pod billing.

`docs/runpod-gotchas.md` is the running list of what bit on the pods.

The driver is tested against a stub of the API (`tests/test_runpod_driver.py`: a fake `curl`
answering from canned JSON, so no account is touched): the create bodies, the guard and the
fall-through on a capacity error, `down` sparing other people's pods, `scale`, `kill-worker`
and `kill-head`, `up` after a head death, the ssh command line, `ps`.

## 6. The run on pods

The session of 2026-09-16, in the order it happened; the numbers are that day's.

### 6.1 The session's shell

```bash
export $(grep '^S3_' .harness/aws/env | xargs)        # the bucket's endpoint, region and keys
export DISTRAINER_DRIVER=runpod
export DISTRAINER_IMAGE=ghcr.io/rahuldave/distrainer-gpu:gest-zvymnspr-runpod   # the branch tag until the merge
deploy/driver.sh catalog                               # what is rentable now, and where
deploy/driver.sh up 1                                  # the head and one worker
deploy/driver.sh ps                                    # the pods and the hourly total
```

`.env` holds `RUNPOD_KEY` and `DISTRAINER_RUNPOD_SSH_KEY=~/.ssh/id_rsa`; the `S3_*` lines are
exported from the AWS bed's env file because the RunPod driver reads `.env` only (the gotchas
say why they are not in `.env`).

### 6.2 The data, once

The CIFAR-10 tarball was staged in the bucket from the laptop's cache (`aws s3 cp` to
`s3://distrainer-rahuldave/datasets/`), then fetched into the head over a presigned URL
(`aws s3 presign`, `urllib.request.urlretrieve` on the pod, seconds) so torchvision finds it
under `data_root` and skips the download. `make_blocks.py` on the head then wrote the 192
blocks, the 16 segments and the held-out split to the bucket in 7 min 28 s (the transatlantic
puts from EU-RO-1 to us-east-1; the PNG encoding is seconds). Every run since reuses them.

```bash
url=$(aws s3 presign s3://distrainer-rahuldave/datasets/cifar-10-python.tar.gz --expires-in 14400)
deploy/driver.sh exec-head python -c "import urllib.request, os, sys; os.makedirs('/tmp/datasets', exist_ok=True); urllib.request.urlretrieve(sys.argv[1], '/tmp/datasets/cifar-10-python.tar.gz')" "$url"
deploy/driver.sh exec-head python examples/image_contrastive/make_blocks.py --config examples/image_contrastive/harness-s3.yaml
```

### 6.3 World size 1, on the head's own GPU

The head advertises no `trainer` (the harness rule), so a single-pod run lends it the training
slot with two overrides; nothing else changes:

```bash
deploy/driver.sh exec-head python examples/image_contrastive/train.py \
    --config examples/image_contrastive/harness-s3.yaml --set run_name=gpu_w1 \
    --set scaling.num_workers=1 --set 'scaling.resources_per_worker={GPU: 1}'
```

On an RTX A5000 (CA-MTL-1): **95 s end to end** for 384 positions (two passes over the 192
blocks of 256 images), 48 checkpoints to the bucket (`every_k: 8`, each about 135 MB of model
and optimizer state, uploaded asynchronously), about 0.15 s per step, the loss from 5.5 (the
log of the number of candidates) to 1.93, and the probe **`knn_acc=0.448` against 0.10
chance** on 2000 held-out images against 5000 training images, computed on the GPU; the
audit check `S1 PASS`. On the laptop's tiny encoder the same probe reaches 0.29 after 48
positions; the full ResNet-18 after 384 positions on a GPU is where the example starts to
look like SimCLR.

### 6.4 What the pods cost and how long they took

| item | value |
|---|---|
| head, first attempt | RTX 2000 Ada, EU-RO-1, 0.24 USD/h; container up after a 27-minute pull; terminated (it advertised the wrong address, section 4 of the gotchas) |
| head, second attempt | RTX A5000, CA-MTL-1, 0.27 USD/h; container up 4 minutes after the create |
| worker, first | RTX 2000 Ada, EU-RO-1, 0.24 USD/h (the fallback: CA-MTL-1 was sold out); its host was still pulling after 31 minutes; terminated |
| worker, second | RTX 2000 Ada, EU-RO-1, 0.24 USD/h |
| the pair | 0.51 USD/h |
| spend to the world-size-1 run | about 0.40 USD |

### 6.5 Two nodes, and breaking things

Written when the two-node run and the kills have happened.
