# Handoff: after M7 and M7b (the AWS beds, ECR): what is there, what it taught, and M8 on RunPod

Written 2026-09-16 at the end of the M7/M7b session for the thread that picks up the next step.
Read `CLAUDE.md` and `AGENTS.md` first (workflow rules), then this file, then
`docs/tutorials/aws.md` (both beds, the registry, which image for which machine),
`docs/uncloud-gotchas.md` (the AWS section at the end) and the two cheat sheets under
`docs/cheatsheets/`. Verify the Gest ids and branch state with `gest task show` and `git status`
before relying on them.

## 1. Where things stand

- M0 to M7b are merged to `main`: M7, the cloud stage (PR #17, issue #16, squash `f988a4e`; Gest
  parent `rpmyqnzs`, iteration `mtoypxln`), and M7b, the image in a private registry and an x86
  bed (PR #19, issue #18, squash `ed576d0`; Gest parent `norpvuup`, iteration `mkooymum`).
  Nothing is open on GitHub.
- The same driver, compose file and scenario runner as M6 run on EC2 instances with an S3
  bucket as the store: `deploy/uncloud/aws.sh` builds the bed, `.harness/aws/env` tells the
  driver about it, `endpoint` prints `s3=` and the runner deploys no MinIO. The node image is
  built for `linux/amd64` and `linux/arm64` and lives in a private ECR repository; the machines
  pull it. Section 3 has the numbers.
- **The AWS state at the end of the session**: the x86 bed is **destroyed** at Rahul's request
  (the instances terminated, the security group and the key pair deleted, the uc context
  `distrainer-aws` forgotten; the `machines-destroy` verb ran twice because the first run was
  interrupted, and finished cleanly the second time). What remains, cents a month: the bucket
  `distrainer-rahuldave` (us-east-1, private, tagged; the block log, the streaming store and the
  scenario runs, about 2 MB) and the ECR repository `distrainer` (tagged; the multi-architecture
  `latest` plus earlier untagged manifests, a lifecycle policy keeps twelve). `.harness/aws/env`
  keeps the store and image settings, so `just aws-machines` and `just build` bring a bed back
  in about five minutes (tutorial 5 section 10 has the lifecycle table); `deploy/uncloud/aws.sh
  bucket-rm` and `ecr-rm` remove the rest. Spend for the whole session was under two dollars of
  the 15 USD budget.
  budget.

## 2. Environment checklist

```bash
just setup && just verify                    # laptop-only gate (196 unit tests, smoke)
brew install awscli psviderski/tap/uncloud   # the native CLI (the pkg one is x86_64 under Rosetta), uc 0.20
aws iam get-user                             # the default profile's IAM user: EC2, S3, ECR, and the inline distrainer-harness-iam policy
cat .env                                     # DISTRAINER_ENV_FILE=.harness/aws/env
deploy/uncloud/aws.sh status                 # the instances, the admitted address; uc machine ls once running
export DISTRAINER_DRIVER=uncloud
deploy/driver.sh machines-start              # if parked: new public addresses; the ssh config, the uc context and the ssh rule follow
just build                                   # after a code change: buildx pushes both halves to ECR, the machines pull (section 5: a repeat push is slow today)
just up 2 && just blocks examples/hello_blocks/harness-s3.yaml && just train examples/hello_blocks/harness-s3.yaml --set run_name=<new>
just integration S2                          # S3, S4, S8, S9, S10, S11s3 likewise; one at a time
deploy/driver.sh machines-stop               # park; machines-destroy removes everything but the bucket and the repository
```

The OrbStack bed is unchanged: `orb start uc1 uc2 uc3`, then `just uncloud-machines` or
`DISTRAINER_UNCLOUD_PROVIDER=orbstack deploy/driver.sh machines-status` (a pinned provider makes
the driver skip an env file that belongs to another bed, so the OrbStack defaults apply while
`.env` points at AWS; the compose and KubeRay drivers never read that file). A `kuberay-operator`
pod in OrbStack's k3s was crash-looping (45 restarts) since M5 and loads the VM:
`DISTRAINER_DRIVER=kuberay just down` and `kubectl -n <ns> delete deployment kuberay-operator`,
or disable k8s, before the next OrbStack session. Memory on the Mac stays tight with any cluster
up: one scenario at a time.

## 3. What M7 and M7b delivered

- `deploy/uncloud/aws.sh` (`up | status | stop | start | destroy | bucket | bucket-rm | ecr |
  ecr-rm | env`): key pair, security group (ssh from this Mac's address only, WireGuard between
  the members; the dashboard through an ssh tunnel), three tagged instances in a zone that
  offers the types (recorded, so a later `up` keeps the zone), the uncloud context over the
  private addresses, park and resume with the new public addresses rewritten into the generated
  ssh config, the uc context and the ssh rule, the bucket with an IAM user scoped to it, the
  tagged ECR repository with a lifecycle policy; `bucket-rm` and `ecr-rm` delete only what
  carries the tag (`DISTRAINER_AWS_ADOPT=1` adopts an existing one). `deploy/uncloud/common.sh`
  is shared with `machines.sh` (which gained `stop`/`start`). The bed defaults to x86
  (`t3.large`, `t3.medium`); Graviton (`t4g`) is a setting.
- Drivers: every driver lets the caller's environment win over `.env`; the uncloud driver also
  reads the env file `.env` names as `DISTRAINER_ENV_FILE` (skipped when the caller pins another
  bed), takes `DISTRAINER_UNCLOUD_SSH_OPTS`, dispatches `machines-*` by
  `DISTRAINER_UNCLOUD_PROVIDER`, prints `s3=<S3_ENDPOINT>` from `endpoint` for a store outside
  the cluster, and its `build` takes a registry image: `docker buildx build --platform
  linux/amd64,linux/arm64 --push` on a docker-container builder, then every machine logs in over
  the ssh route with the Mac's twelve-hour ECR token and pulls; a local image keeps `docker
  build` plus `uc image push`. `run_scenarios.py`: the bucket store from `endpoint` (`minio=` or
  `s3=`, KubeRay's placeholder before the deploy), S9/S10/S11s3 parametrised by the store, the
  Mac-side client signing with the bucket's region, a `lag_intervals` allowance on the replay
  bound for a store outside the cluster, `harness-s3.yaml` and `harness-stream-s3.yaml`.
- Results on 2026-09-16 (the bucket in us-east-1), every scenario green on both beds:

  | scenario | arm64 bed (`t4g`, image pushed) | x86 bed (`t3`, image from ECR) | OrbStack (MinIO) |
  |---|---|---|---|
  | S2 | 100 s | 107 s | 149 s |
  | S3 | 103 s | 118 s | 146 s |
  | S4 | 179 s | 193 s | 210 s |
  | S8 | 87 s | 98 s | 125 s |
  | S9 | 197 s | 212 s | 254 s |
  | S10 | 163 s | 161 s | 205 s |
  | S11s3 | 220 s | 222 s | 161 s |

  `up` 78 s (arm64) and 42 s (x86); the 240 blocks 71 s to write; the plain run 80 s on arm64
  with 0.67 s per step (0.25 on MinIO: each checkpoint is a one-second upload). The build of
  both halves takes about four minutes on a warm builder; the push of both halves about ten
  minutes from a home uplink; the three in-region pulls seconds.
- Docs: tutorial 5 (`docs/tutorials/aws.md`) as the operations guide for both beds, tutorial 4
  section 9 pointing at it, the AWS section of `docs/uncloud-gotchas.md`, running-modes C, the
  verb map and config tables, docs/cli.md, `.env.example`, the spec's section 11 and the
  milestone list, the cheat sheets under `docs/cheatsheets/`.

## 4. Behaviours learned in M7 and M7b that will bite again

- **macOS bash 3.2**: a failing command run through the `command` builtin ignores `set -e`'s
  suppression in `if` conditions and `||` lists; a failing command substitution in an
  assignment ends the script too. Both ended a bootstrap run silently before they were
  understood. Plain calls, and `out="$(...)" || out=""`.
- **An old account has no default VPC** and one of its zones (`us-east-1a`) has no Graviton:
  `create-default-vpc` and a subnet chosen by `describe-instance-type-offerings`.
- **The bootstrap's recorded settings outlive a changed default** (`.harness/aws/env` is read
  first by the bootstrap): a rename means destroy, delete the env file, up.
- **Real S3 signs by region** (`S3_REGION=auto` gets a 400 from `HeadObject` with no body), and
  **the Train controller lags the workers on S3**: its checkpoint bookkeeping is S3 round trips,
  so a resize restores one or two reports back (24 and 18 positions replayed against bounds of
  11 and 14). **Checkpoint latency sets the step pace** there (0.67 s per step with `every_k:
  2`), which inverts the S11 pacing premise.
- **uncloud has no registry login**: the machines pull with a token the Mac hands them over
  ssh; **a multi-platform push needs a docker-container builder**; **ECR's tag fields are
  capitalised** (`Key`, `Value`), unlike EC2's; **a run name that exists in the bucket is
  restored, not rerun**.
- **Credentials**: every new AWS service needed a policy added in the console (EC2, the scoped
  IAM policy, ECR); do the whole list of tutorial 5 section 2 before a session, not during it.
  The auto-mode classifier refuses to inspect other profiles or list policies; ask Rahul.
- Everything in the M6 list still applies (`docs/handoff-m7.md` section 4).

## 5. Review follow-ups still open

- **A repeat push to ECR is not cheap yet**: pushing the unchanged image again took eight
  minutes (`pushing layers 310 s`) where a registry should skip blobs it holds. Either the
  docker-container builder's cache let the dependency layer go (its own garbage collection) or
  its exporter recompressed it into new blobs. Check with `docker buildx du`, a build with
  `--cache-to type=registry` / `--cache-from`, or `--provenance=false` and a fixed compression;
  until then a code change costs a full push.
- The containers hold a long-lived IAM access key for S3 (no instance profile) and each machine
  keeps the ECR token for twelve hours in root's Docker config; an instance profile with ECR read
  and the bucket policy would retire both, and needs IAM role permissions the CLI user lacks.
- `lag_intervals` is an allowance over an unmodelled controller lag; the exact check would
  compare the resumed position with the ledger of the newest checkpoint registered at the
  restart, which `num_to_keep` deletes before the run ends.
- `wait_for_trainers` gives 180 s after a full teardown; S9 under compose timed out there once on
  a loaded Mac.
- `harness-s3.yaml` and `harness-stream-s3.yaml` name a personal bucket; anyone else edits and
  rebuilds. The lifecycle policy counts manifests (twelve), not builds.
- `stop-worker` and `cp-from-head` are used by no scenario under any driver.
- The M4 to M6 lists in `docs/handoff-m7.md` section 5 are unchanged.

## 6. M8: a GPU contrastive example on RunPod

Rahul's direction (2026-09-16): the next session's work is a contrastive training example that
needs a GPU, run on RunPod. The research below is from RunPod's documentation and price pages as
of this date; verify the moving parts (prices, data centers, API fields) before building on them.

### 6.1 What RunPod offers, and what fits

- **Pods** are single containers on a GPU host, created from an image (a template holds the
  image, the start command, ports, environment, disks) through the console, `runpodctl pod
  create --name ... --gpu-id "NVIDIA GeForce RTX 4090" --image ... --container-disk-in-gb 20
  --volume-in-gb 50`, the REST API (`POST https://rest.runpod.io/v1/pods` with `Authorization:
  Bearer <key>`; fields `name`, `imageName`, `gpuTypeIds`, `gpuCount`, `containerDiskInGb`,
  `volumeInGb`, `templateId`; `POST /v1/pods/{id}/stop`, `/start`, `DELETE /v1/pods/{id}`), or
  the `runpod` Python package. Private images use registry credentials stored in the account
  (a username and password pair) and referenced by the template: fine for GHCR with a personal
  access token or a private Docker Hub repository, not for ECR, whose password is a twelve-hour
  token. **Decision for Rahul: GHCR (free for a public repository's packages, a PAT with
  `write:packages`) or a private Docker Hub repository.**
- **Global networking** gives pods private TCP/IP connectivity with each other as
  `POD_ID.runpod.internal` (no ports to open, enabled at pod creation only, in about seventeen
  data centers, 100 Mbps between pods). That is the Ray cluster's network: the head binds to its
  internal address and the workers dial `<head pod>.runpod.internal:6379`; the loss-side
  `all_gather` of the contrastive example is a few kilobytes per step, well within 100 Mbps,
  and blocks and checkpoints go to the store, not between pods. There is no UDP promise, and
  none is needed: uncloud does not apply here (its WireGuard mesh would need it, and RunPod
  pods are not Docker hosts anyway).
- **Instant Clusters** are multi-node H100 (and similar) clusters with InfiniBand, up to 8
  nodes, with RunPod injecting `MASTER_ADDR`, `MASTER_PORT`, `NUM_NODES`, `NUM_TRAINERS`,
  `NODE_RANK` on every node; RunPod's own Ray guide runs `ray start --head
  --node-ip-address=$(hostname -I | awk '{print $1}') --port=6379 --num-gpus=$NUM_TRAINERS` on
  pod-0 and `ray start --address=$MASTER_ADDR:6379 ...` on the others, with `/dev/shm` raised
  to 8 GB. Too expensive and too big for a first experiment; global networking with cheap pods
  is the fit. The `ray start` recipe transfers as it is: `--node-ip-address` set to the pod's
  internal address, never `0.0.0.0`.
- **Storage**: the block log, the audit trail and the checkpoints need an S3-compatible store.
  Two options: keep the AWS bucket (works unchanged; S3 charges 0.09 USD per GB for data read
  from outside AWS, small at these sizes, and RunPod charges no egress), or a RunPod **network
  volume** with its **S3-compatible API** (`https://s3api-<datacenter>.runpod.io/`, keys made
  in the console's settings, `s3://<volume id>/prefix`; put, get, delete, list, multipart
  supported; no bucket creation, no presigned URLs; the volume also mounts at `/workspace` in
  every pod of that data center, a shared mount, which would even let S6 and S11 run). The
  library's `storage.kind: s3` with `endpoint` and `region` should work against it; verify
  path-style addressing and the `ListObjects` limits (directories over 10 000 files). **Decision
  for Rahul: the AWS bucket first (nothing to build), the network volume as the experiment's
  second half.**
- **Access**: ssh to a pod through `ssh.runpod.io` with the account's injected public key, or
  a directly exposed TCP port; `exec-head` becomes an ssh command, `cp-from-head` an scp. Pods
  have a public proxy for HTTP ports (the Ray dashboard could be exposed that way, behind
  RunPod's proxy).
- **Prices** (on-demand, per GPU-hour, 2026): RTX 4090 about 0.34 USD on the community cloud
  and 0.69 on the secure cloud; A100 80 GB about 1.4 to 1.6; H100 about 2.9 to 3.3; network
  volumes 0.07 USD per GB-month; container disk free while stopped (it is erased), a pod's
  volume disk 0.10 per GB-month (0.20 while the pod is stopped); no ingress or egress fees.
  Three RTX 4090 pods (a head and two workers) are about one dollar an hour on the community
  cloud; the head could be a CPU-only pod if a data center offers those with global networking.

### 6.2 The example: `examples/image_contrastive`

`examples/toy_contrastive` (spec section 8) is a two-layer MLP on 32 synthetic features: a
GPU changes nothing for it. The GPU-needing example keeps the same shape of the harness (blocks
of rows in Parquet, positives and hard negatives per row, InfoNCE with the loss-side
`all_gather`, the re-mining hook for S6) and changes what a row is:

- **Data**: CIFAR-10 (60 000 images of 32 by 32, 170 MB, from torchvision) or STL-10's
  unlabelled split (100 000 of 96 by 96, 2.6 GB) as PNG bytes in Parquet blocks of 256 rows
  (`image`, `label`, `item_id`, `block_id`); `make_blocks.py` writes them to the store as
  `hello_blocks/make_blocks.py` does. CIFAR-10 first: the download is quick and 235 blocks of
  256 make a log of `W=24` with `passes: 2` an eleven-segment run.
- **Model**: a ResNet-18 encoder (torchvision, no pretrained weights) with a two-layer
  projection head, SimCLR's two random augmentations per image as anchor and positive (the
  augmentation runs in `train_step` on the device), NT-Xent as the existing `info_nce` with
  `all_gather: true` and the in-block others as negatives; then the re-mining hook (`remine.py`
  of the toy example) mining hard negatives with the current encoder per segment, which is
  where a GPU makes the hook cheap. `use_gpu: true`, `resources_per_worker: {GPU: 1, trainer:
  1}`, a batch of 256 per step, mixed precision. On an RTX 4090 a step is well under 0.1 s
  against seconds on a CPU: that is the "needs a GPU" property, and the 1 s checkpoint upload
  becomes the pacing item again (`every_k` larger, or a time policy).
- **A sanity metric**: a k-nearest-neighbour or linear-probe accuracy on the CIFAR-10 test
  split at the end of the run, computed by rank 0 and printed as the final metric, so a run can
  be judged (SimCLR on ResNet-18 reaches 80 to 90 percent with a long schedule; a few segments
  will show a clear rise above 10 percent chance).
- **The image**: a second Dockerfile, `deploy/Dockerfile.gpu`: an x86 CUDA base image
  (`nvidia/cuda:12.x-runtime-ubuntu22.04` or RunPod's PyTorch image), the same uv-managed
  environment with a `gpu` extra that pins torch and torchvision from the CUDA wheel index
  (`https://download.pytorch.org/whl/cu126`), `ray[data,default,train]` as today. Six to eight
  gigabytes: build and push it in **GitHub Actions to GHCR** (free for a public repository),
  not from the Mac's uplink (the M7b measurement: ten minutes per gigabyte). The uncloud
  driver's `build` already handles a non-ECR registry (no token) if the Mac is logged in.
- **The driver**: `deploy/drivers/runpod.sh` behind the same verbs, on the REST API (`curl`
  and `jq`, or `runpodctl`): `up N` creates the head pod (the image, `deploy/ray-head.sh` as
  the start command, global networking, one data center, the S3 settings as environment) and
  then N worker pods with `RAY_HEAD_ADDRESS=<head pod id>.runpod.internal:6379` (the head's id
  is known once it exists; `ray-worker.sh` already retries until the head answers); `scale N`
  creates or terminates workers; `kill-worker I` terminates a worker pod and creates a new one
  (node death, then a new node, as `DISTRAINER_RESTART_DELAY` does); `kill-head` terminates the
  head; `exec-head` is ssh; `shared` prints nothing; `endpoint` prints the store; `down`
  terminates everything of the cluster's name. Pods take about a minute to start, so `up` and
  a resize cost more than on uncloud; the runner's `wait_for_trainers` budget may need
  raising. `docs/running-modes.md` gets a mode E; the verb map a RunPod column.
- **Acceptance**: the example runs on one RTX 4090 pod (world size 1) with the probe accuracy
  rising, then on three pods (a head and two workers) with S2, S3, S4, S9 and S10, and the
  re-mining variant (S6 needs the shared mount: the network volume at `/workspace`, or a
  bucket variant of S6 with `harness-remine.yaml` on S3, a follow-up from M6). Budget: a few
  dollars.

### 6.3 What this session already settled (2026-09-16, before the context was cleared)

- **Decisions made with Rahul**: the image goes to **GHCR as a public package** (RunPod pulls
  it with no stored credentials; GitHub Actions pushes it with its automatic token); the store
  is the **AWS bucket** (`harness-s3.yaml`, unchanged; a RunPod network volume through its S3
  API is a later experiment); the GPUs are the **cheapest that allow a reasonable test**: the
  API's prices on the community cloud were RTX A5000 0.16 USD per hour (24 GB), RTX A4000 0.17
  (16 GB), RTX 3090 0.22, RTX 4090 0.34, A40 0.35 (secure only), L4 0.44; so `gpuTypeIds`
  `["NVIDIA RTX A5000", "NVIDIA RTX A4000", "NVIDIA GeForce RTX 3090", "NVIDIA GeForce RTX
  4090"]` with `gpuTypePriority: availability` and `cloudType: COMMUNITY`, about half a dollar
  an hour for three pods. The head pod can be a CPU pod (`computeType: CPU`) if its data center
  also offers global networking; check at run time.
- **Budget and storage (Rahul, 2026-09-16, the next session)**: the whole RunPod example and
  integration test stays under about **20 USD**. The blocks, the log, the audit trail and the
  checkpoints live on the S3 bucket (`examples/image_contrastive/harness-s3.yaml` puts
  `store_root` and `storage_path` there); pod-local disk (the container disk, no persistent
  volume) holds only the CIFAR download on the head and Ray's checkpoint staging on the workers;
  a pod volume is allowed if something needs one, but nothing initial or final lives on a pod.
  Estimate from the API prices that day: global networking needs **secure-cloud NVIDIA GPU
  pods** (the head too, no CPU pod), where RTX 2000 Ada is 0.24 USD per hour, RTX A4000 and
  A4500 0.25, RTX A5000 0.27, RTX 4000 Ada 0.28, L4 and A40 0.49, RTX 4090 0.74; three pods
  0.72 per hour on the cheap types, about 1.50 on L4 or A40 class; the single-pod bring-up
  0.25 to 0.50, the cluster scenarios 0.75 to 1.50 (up to 3.00 on L4 or A40), the re-mining
  follow-up 0.40 to 0.75: about 1.5 to 3 USD in all, 5 at the outside. Terminate pods, never
  stop them (a stopped pod's volume costs 0.20 USD per GB-month). The driver carries a spend
  guard (`DISTRAINER_RUNPOD_MAX_GPU_HOURLY`, default 0.60) and prints the cluster's hourly cost.
- **The API is v2 now**: `https://api.runpod.io/v2` (REST v1 at `rest.runpod.io/v1` is retired
  on 2026-11-15; the fields in the next bullet are v1's). v2: `POST /v2/pods` with `name`,
  `image`, `args` (the start command, a string), `ports` (`["22/tcp"]`), `env`, `disk`
  (container GB), `cloud` (`SECURE` | `COMMUNITY`), `dataCenterIds`, `globalNetworking`
  (true; NVIDIA GPU pods in a global-networking data center only, secure cloud), `gpu: {id,
  count, minRamPerGpu, minVcpuCountPerGpu, allowedCudaVersions | minCudaVersion}` (**one**
  type per create: the driver tries `DISTRAINER_RUNPOD_GPU_TYPES` in order), `startSsh` (injects
  `PUBLIC_KEY` with the account's registered keys; **our image must start sshd from it**,
  RunPod's official images do), `mounts.persistent {size, path}` (omit: no volume). The pod
  object: `status` (`PROVISIONING STARTING RUNNING EXITED ERROR TERMINATED`),
  `globalNetworking {enabled, ip, internalDns}` (`<id>.runpod.internal`), `ssh {proxy, direct}`
  (`direct` needs `22/tcp` in `ports`: `{host, port, username, command}`), `runtime {uptime,
  ports}`, `cost` (USD per hour), `dataCenterId`, `cudaVersion`. `GET /v2/pods` returns
  `{pods: [...]}` with no name filter (filter client-side by the name prefix and the
  `DISTRAINER_CLUSTER` env marker); `GET /v2/pods/{id}`; `POST /v2/pods/{id}/action` with
  `{"action": "start" | "stop" | "restart" | "terminate"}`; `DELETE /v2/pods/{id}`; `GET
  /v2/pods/{id}/logs`; `GET /v2/catalog/gpus?include=AVAILABILITY&product=POD` (per type:
  `price {secure, community}`, `dataCenters [{id, availability}]`, `cudaVersions`); `GET
  /v2/catalog/datacenters` (`globalNetwork: true` for CA-MTL-1 CA-MTL-3 EU-CZ-1 EU-FR-1 EU-NL-1
  EU-RO-1 EU-SE-1 EUR-IS-2 EUR-IS-4 OC-AU-1 US-CA-2 US-GA-2 US-IL-1 US-KS-2 US-NC-1 US-TX-3
  US-TX-4 US-WA-1); `GET /v2/billing/pods` for the spend; `GET /v2/account/ssh-keys` (four
  registered). The spec: `https://api.runpod.io/v2/openapi.json`. Availability of the cheap
  types in those data centers was LOW or NONE at the time; RTX 2000 Ada in EU-RO-1 was the
  cheapest open door.
- **The account**: `RUNPOD_KEY` is in `.env` (that name, not `RUNPOD_API_KEY`); `runpodctl`
  1.9.0 and `jq` 1.7 are installed. The account holds five stopped pods that belong to a
  teammate: **the driver must act only on pods it created**, found by a name prefix
  (`distrainer-<context>-head`, `-worker-N`) and an environment marker
  (`DISTRAINER_CLUSTER=<context>`), never by "all pods".
- **The API, verified**: `POST https://rest.runpod.io/v1/pods` with `Authorization: Bearer
  $RUNPOD_KEY`; fields `name`, `imageName`, `gpuTypeIds`, `gpuTypePriority`, `gpuCount`,
  `cloudType` (`COMMUNITY` | `SECURE`), `dataCenterIds`, `dataCenterPriority`,
  `globalNetworking` (true; creation time only), `ports` (`["22/tcp"]` for direct ssh),
  `env`, `dockerEntrypoint`, `dockerStartCmd` (`["deploy/ray-head.sh"]` or
  `["deploy/ray-worker.sh"]`), `containerDiskInGb`, `volumeInGb`, `volumeMountPath`,
  `networkVolumeId`, `containerRegistryAuthId`, `templateId`, `supportPublicIp`,
  `minRAMPerGPU`, `minVCPUPerGPU`, `interruptible`, `computeType`, `vcpuCount`; the response
  has `id`, `publicIp`, `portMappings` (`{"22": 10341}`), `machine`, `desiredStatus`
  (`RUNNING` | `EXITED` | `TERMINATED`); `POST /v1/pods/{id}/stop`, `/start`, `DELETE
  /v1/pods/{id}`, `GET /v1/pods` lists (a teammate's pods included). The internal hostname is
  `<pod id>.runpod.internal`. GPU types and prices come from the GraphQL endpoint
  (`https://api.runpod.io/graphql`, query `gpuTypes { id displayName memoryInGb
  communityPrice securePrice lowestPrice(input:{gpuCount:1}) { uninterruptablePrice } }`).
  Not yet verified: how `GET /v1/pods/{id}` reports readiness and the internal address, the
  `/dev/shm` size in a pod (Ray's object store; the harness uses 200 MB, `shm_size: 1g` in
  compose), whether `dockerStartCmd` runs with the image's `PATH` (the Dockerfile sets it),
  and which data centers offer both global networking and the chosen GPUs.
- **Gest and git**: parent `zvymnspr` (issue #20), iteration `toonvruz`, leaves `xkmulnzr`
  (the example at CPU size, claimed, nothing written yet), `nysusvou` (the GPU image and the
  Actions workflow), `yrkynmlp` (the RunPod driver), `pmtxwnrx` (the run), `nnlkrrul` (docs
  and the M9 handoff). Branch `gest/zvymnspr-runpod` exists from `main` with no commits.
- **Implementation notes from reading the code** (so the next session does not re-read it):
  - An example is `make_blocks.py` plus `train.py` plus configs. Blocks are written with
    `distrainer.block.write_block(fs, root, block_id, table, meta)` and the log with
    `BlockLog.create(fs, root, W=..., seed=...)` then `BatchWriter(log, refs, passes=...,
    tail="error").run()`; `examples/toy_contrastive/make_blocks.py` builds the rows with Ray
    Data (`groupby("batch_id").map_groups`) and is the model to copy; `make_blocks` returns
    the existing refs when the log exists (idempotent per store).
  - `train.py` defines `build_model(info) -> (model, optimizer)`, `train_step(model,
    optimizer, table, info) -> metrics dict` (DDP all-reduces in `backward`), `entry(cfg)`
    returning both (for `distrainer resume --entry`), and a `main` with `--config`, `--keep`,
    `--no-check`, `--set KEY=VALUE`, `reset_run`, `ensure_blocks`, `init_ray`,
    `DistTrainer(train_step, build_model, cfg).fit()`, then `read_audit` and the `check_s1`
    assertions. `info.device` is Ray Train's device for the worker (the GPU with
    `scaling.use_gpu: true`); move the batch there in the step. `info.train` is the free-form
    `train:` section.
  - `examples/toy_contrastive/model.py` has `info_nce(anchor, positive, negatives,
    temperature, all_gather)`; NT-Xent is that with zero hard negatives (`negatives` of shape
    `[B, 0, e]`) and `all_gather: true`, so the new example imports it. The re-mining hook is
    `examples/toy_contrastive/remine.py` (`hooks: {remine: ...}` in the config, the log
    streamed); its GPU variant is a follow-up leaf, not the first one.
  - Dependencies: `pyproject.toml` pins torch and torchvision from the CPU index through
    `[tool.uv.sources]`. **A `gpu` extra does not work** (tried 2026-09-16): an
    extra-conditional source collides with the base CPU source in the Linux fork unless torch
    leaves the base dependencies for two conflicting extras, and uv has no default extras, so
    every plain `uv run` would then drop torch. `deploy/Dockerfile.gpu` instead installs the
    CUDA 12.6 wheels of the locked versions over the CPU ones in a layer of its own (`uv pip
    install --index .../whl/cu126 torch==2.14.0+cu126 torchvision==0.29.0+cu126`, the `+cu126`
    local versions spelled out, or uv sees `==2.14.0` as satisfied), on the same
    `python:3.13-slim` base as the CPU image (the CUDA runtime rides in the wheels, the driver
    comes from RunPod's NVIDIA runtime; no CUDA base image), and installs the project with `uv
    pip install --no-deps -e .` because a second `uv sync` would restore the CPU wheels.
    `tests/test_deploy_manifests.py` pins the Dockerfile's version ARGs to `uv.lock`. CI (`.github/workflows/ci.yml`) syncs with `--all-groups`; the GPU
    image workflow is a second file, triggered on pushes to `main` that touch `deploy/`,
    `pyproject.toml`, `uv.lock`, `distrainer/` or `examples/`, and by hand, with
    `permissions: packages: write` and `docker/login-action` against `ghcr.io` using
    `GITHUB_TOKEN`, `docker/build-push-action` with `platforms: linux/amd64`.
  - The driver `deploy/drivers/runpod.sh`: `build` is a no-op that prints where the image
    comes from (`DISTRAINER_IMAGE=ghcr.io/rahuldave/distrainer-gpu:latest`); `up N` creates
    the head with the S3 settings and `RAY_TRAIN_V2_ENABLED` etc. in `env`, waits until
    `desiredStatus` is `RUNNING` and ssh answers, then the workers with
    `RAY_HEAD_ADDRESS=<head id>.runpod.internal:6379`; `kill-worker I` deletes and recreates
    worker I (after `DISTRAINER_RESTART_DELAY`), `kill-head` deletes the head, `scale N`
    creates or deletes workers, `down` deletes every pod of the cluster, `exec-head` is
    `ssh -p <mapped port> root@<publicIp>` (the account's ssh public key must be set in
    RunPod's settings; RunPod injects it) or `ssh <id>-<hash>@ssh.runpod.io`, `cp-from-head`
    an scp, `shared` nothing, `endpoint` `s3=...`, `ps` and `logs` from the API. The runner's
    `wait_for_trainers` (180 s) and `up` will need a bigger budget for pod start times. The
    existing `tests/test_deploy_manifests.py::test_every_driver_implements_every_common_verb`
    will demand every verb.

### 6.4 Order of work for the session

1. Rahul: a RunPod account and API key (`RUNPOD_API_KEY` in `.env`), the registry choice
   (GHCR needs a PAT in the repository's secrets for the Actions workflow), and whether the
   store is the AWS bucket or a network volume.
2. `gpl` an M8 iteration from the Gest task below; the leaves: the example and its GPU image
   (with `just` targets and the Actions workflow), the RunPod driver, the run, the docs.
3. The example on the laptop at CPU size first (a tiny ResNet, a few blocks) so `just smoke`
   and the unit tests cover it; then the image; then one pod; then the cluster.

Sources: RunPod docs on [global networking](https://docs.runpod.io/pods/networking), [managing
pods](https://docs.runpod.io/pods/manage-pods), [the Ray instant-cluster
guide](https://docs.runpod.io/instant-clusters/ray-vllm), [the S3-compatible
API](https://docs.runpod.io/storage/s3-api), [the REST API announcement](https://www.runpod.io/blog/runpod-rest-api-gpu-management),
and price surveys ([Flexprice](https://flexprice.io/blog/runprod-pricing-guide-with-gpu-costs),
[Northflank](https://northflank.com/blog/runpod-gpu-pricing)).
