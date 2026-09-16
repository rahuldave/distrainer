# Handoff: after M7 (the AWS bed): what is there, what it taught, what is next

Written 2026-09-16 at the end of the M7 session for the thread that picks up the next step. Read
`CLAUDE.md` and `AGENTS.md` first (workflow rules), then this file, then `docs/tutorials/aws.md`,
`docs/uncloud-gotchas.md` (the AWS section at the end) and the two cheat sheets under
`docs/cheatsheets/`. Verify the Gest ids and branch state with `gest task show` and `git status`
before relying on them.

## 1. Where things stand

- M0 to M7 are merged to `main` (PRs #8 to #17). M7, the cloud stage, was branch
  `gest/rpmyqnzs-cloud-aws`, PR #17 (issue #16; Gest parent `rpmyqnzs` under the v0.1 root
  `qvuukpsm`, iteration `mtoypxln`, leaves `vvkwopls` bootstrap and driver, `xkpkkqqm` runner,
  `wuykxnsn` the cloud run, `ktkzopov` docs, `ryxskkpu` cheat sheets). Merged to `main` as the squash commit `f988a4e` (PR #17,
  issue #16 closed) on 2026-09-16; the branch is deleted.
- The same driver, compose file and scenario runner as M6 run on three EC2 instances with an S3
  bucket as the store: `deploy/uncloud/aws.sh` builds the bed, `.harness/aws/env` tells the
  driver about it, `endpoint` prints `s3=` and the runner deploys no MinIO. Section 3 has the
  numbers.
- The AWS bed was left **parked** (`aws1`, `aws2`, `aws3` stopped: disks only, about 0.13 USD per
  day; `deploy/driver.sh machines-start` brings it back with new public addresses, `machines-destroy`
  removes it, `deploy/uncloud/aws.sh bucket-rm` the bucket `distrainer-rahuldave` and its IAM user).
  The bucket holds the block log (`blocks/`), the streaming store (`blocks_stream/`) and the
  scenario runs (`runs/`): 897 objects, 1.9 MB, a fraction of a cent per month; leaving it saves
  the 71 s block build next time, and every scenario cleans its own run state. Spend for the
  session was about 0.3 USD; parked, the bed costs about 4 USD per month for the disks.

## 2. Environment checklist

```bash
just setup && just verify                    # laptop-only gate
brew install awscli psviderski/tap/uncloud   # the native CLI (the pkg one is x86_64 under Rosetta), uc 0.20
aws iam get-user                             # the default profile's IAM user: EC2, S3, and the inline distrainer-harness-iam policy
cat .env                                     # DISTRAINER_ENV_FILE=.harness/aws/env
deploy/uncloud/aws.sh status                 # the instances, the admitted address, uc machine ls
export DISTRAINER_DRIVER=uncloud
deploy/driver.sh machines-start              # if parked: new public addresses; the ssh config and the uc context follow
just build                                   # only after a code change (docker build + uc image push over ssh)
just up 2 && just blocks examples/hello_blocks/harness-s3.yaml && just train examples/hello_blocks/harness-s3.yaml
just integration S2                          # S3, S4, S8, S9, S10, S11s3 likewise; one at a time
deploy/driver.sh machines-stop               # park (disks only, 0.13 USD per day); machines-destroy removes everything but the bucket
```

The OrbStack bed is unchanged: `orb start uc1 uc2 uc3`, then `just uncloud-machines` or
`DISTRAINER_UNCLOUD_PROVIDER=orbstack deploy/driver.sh machines-status` (a pinned provider
makes the driver skip an env file that belongs to another bed, so the OrbStack defaults apply
while `.env` points at AWS; the compose and KubeRay drivers never read that file). A `kuberay-operator` pod in OrbStack's k3s was
crash-looping (45 restarts) since M5 and loads the VM: `DISTRAINER_DRIVER=kuberay just down` and
`kubectl -n <ns> delete deployment kuberay-operator`, or disable k8s, before the next OrbStack
session.

## 3. What M7 delivered

- `deploy/uncloud/aws.sh` (`up | status | stop | start | destroy | bucket | bucket-rm | env`):
  key pair, security group (ssh and the dashboard from this Mac's address only, WireGuard
  between the members), three tagged Graviton instances in a zone that offers the types,
  the uncloud context over the private addresses, park and resume with the new public
  addresses rewritten into the generated ssh config and the uc context's connections, the
  bucket with an IAM user scoped to it. `deploy/uncloud/common.sh` is shared with `machines.sh`
  (which gained `stop`/`start`).
- Drivers: `DISTRAINER_ENV_FILE` read after `.env` by the uncloud driver (and the runner under it),
  the caller's environment winning at every step; `DISTRAINER_UNCLOUD_SSH_OPTS`; `machines-*` by
  `DISTRAINER_UNCLOUD_PROVIDER`; `endpoint` printing `s3=<S3_ENDPOINT>` for a store outside the
  cluster. `run_scenarios.py`: the bucket store from `endpoint`, S9/S10/S11s3 parametrised by
  the store, `harness-s3.yaml` and `harness-stream-s3.yaml`.
- Results on 2026-09-16 (`t4g.large` head, two `t4g.medium` workers, us-east-1b, S3 in
  us-east-1), against the OrbStack numbers of M6: S2 100 s (149), S3 103 s (146), S4 179 s
  (210), S8 87 s (125), S9 197 s (254), S10 163 s (205), S11s3 220 s (161); `up` 78 s, the
  blocks 71 s, the plain run 80 s with 0.67 s per step (0.25 on MinIO: each checkpoint is a
  one-second upload to S3). All PASS after two runner fixes found on the first pass: the
  Mac-side client signs with the bucket's region, and S3/S4's replay bound takes an allowance
  (`lag_intervals`) on a store outside the cluster, where the Train controller registers
  checkpoints behind the workers (24 and 18 positions replayed against 11 and 14).
- Docs: tutorial 5 (`docs/tutorials/aws.md`), tutorial 4 section 9 pointing at it, the AWS
  section of `docs/uncloud-gotchas.md`, running-modes C, the verb map and config tables, the
  spec's section 11, the cheat sheets.

## 4. Behaviours learned in M7 that will bite again

- **macOS bash 3.2 and `command`**: a failing command run through the `command` builtin ignores
  `set -e`'s suppression in `if` conditions and `||` lists and exits the script silently. Two
  bootstrap runs died that way with no message before it was understood. Plain calls are fine.
- **An old account has no default VPC** and one of its zones (`us-east-1a`) has no Graviton
  capacity: `create-default-vpc` and a subnet chosen by `describe-instance-type-offerings`.
- **The bootstrap's recorded settings outlive a changed default** (`DISTRAINER_UNCLOUD_MACHINES`
  from `.harness/aws/env`): a rename means destroy, delete the env file, up.
- **The pkg AWS CLI is x86_64** (nine seconds per call under Rosetta, `sts get-caller-identity`
  never returning); the Homebrew build is native.
- **Credentials**: the default profile's IAM user needed `AmazonEC2FullAccess` and a scoped IAM
  policy added in the console; root sign-in wanted an MFA device that was not at hand. Do this
  before a session that needs the bed, not during it.
- **The Train controller lags the workers on S3**: its checkpoint bookkeeping is S3 round trips,
  so a resize restores one or two reports back; and **checkpoint latency sets the step pace**
  (0.67 s per step with `every_k: 2`), which inverts the S11 pacing premise. Checkpoint less
  often on an object store when pace matters; the harness configs keep `every_k: 2` so the
  beds compare.
- **Real S3 signs by region**: `S3_REGION=auto` gets a 400 from `HeadObject` with no body.
- Everything in the M6 list still applies (`docs/handoff-m7.md` section 4).

## 5. Review follow-ups still open

- The driver's `build` verb has no `--platform`; an amd64 bed (`DISTRAINER_AWS_ARCH=amd64`,
  `t3` types) needs `docker build --platform linux/amd64`, which OrbStack does under Rosetta.
- `wait_for_trainers` gives 180 s after a full teardown; S9 under compose timed out there once
  on a loaded Mac.
- `harness-s3.yaml` names a personal bucket; anyone else edits it and rebuilds.
- `aws.sh` does not manage the default VPC, the admin IAM user or a second region's bed.
- The containers hold a long-lived IAM access key in their environment (no instance profile,
  no rotation); an instance profile on the head and the workers would remove it from the image
  and the env file.
- `lag_intervals` is an allowance over an unmodelled controller lag; the exact check would
  compare the resumed position with the ledger of the newest checkpoint the controller had
  registered at the restart, which `num_to_keep` deletes before the run ends.
- The M4 to M6 lists in `docs/handoff-m7.md` section 5 are unchanged.

## 6. What comes next: pointers

The harness now runs in four places (laptop, containers, OrbStack machines, EC2) against two
stores (MinIO, S3). Candidates for the next stage, none scheduled; a Gest iteration with `gpl`
when one starts:

- **Checkpointing on an object store**: a checkpoint policy that adapts `every_k` to the
  measured upload time, or an asynchronous upload, so the step pace on S3 approaches the
  local one; then the S11 premise holds again and S11s3 gets back under 161 s.
- **A second store and a second architecture**: R2 (the same `.env` lines, another endpoint)
  and an amd64 bed (`DISTRAINER_AWS_ARCH=amd64`, `t3` types, `--platform linux/amd64` on
  `build`, which OrbStack does under Rosetta).
- **Spot instances** for the workers (a real preemption notice instead of `docker stop`), and
  `stop-worker`, which no scenario uses yet.
- **A GPU instance type** behind the existing config flag, with a workload that needs it.
- **Rahul's stated direction (2026-09-16): the next experiments run on RunPod**, and the image
  should live in a private registry (Amazon ECR was asked about). What that changes: the image
  is arm64 today (built natively on the Mac for Graviton) with CPU torch from the CPU wheel
  index; RunPod's GPU hosts are x86_64 with NVIDIA GPUs, so a second image (`--platform
  linux/amd64`, a CUDA torch index or a CUDA base image, `uv.lock` grown an extra) is the first
  step, published as a multi-arch manifest so each host pulls its own. RunPod pods pull from a
  registry and expose TCP ports through a proxy, no UDP, so uncloud's WireGuard mesh does not
  fit there: a RunPod "instant cluster" (a private network between nodes) or one multi-GPU pod
  with Ray's own networking is the shape, which means a new driver (`deploy/drivers/runpod.sh`)
  behind the same verbs. ECR for the EC2 bed is the smaller change: `build` pushes once from
  the Mac, the machines pull in-region through an instance profile with ECR read (the same
  profile would give the containers their S3 access and retire the long-lived key).
