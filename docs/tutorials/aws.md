# Tutorial 5: the cluster on AWS, an arm64 bed and an x86 bed, S3, a private registry

[Tutorial 4](uncloud.md) ran `hello_blocks` on three OrbStack machines joined by uncloud. This
one runs the same driver, the same compose file, the same configs and the same scenario checks
on **EC2 instances**, with an **S3 bucket** as the store and, optionally, a **private image
repository** (ECR) the machines pull from. Nothing in the training code changes. Two beds are
described, because they are what you choose between: **Graviton (arm64)**, the cheapest, running
the image an Apple Silicon Mac builds natively; and **x86 (amd64)**, the architecture of most
clouds and of GPU hosts such as RunPod's, running an image built for it. Both live in the same
region and zone here (us-east-1b); the scripts are the same, only settings differ.

Sections 1 and 2 are the one-time account work. Sections 3 to 9 are the bed and the runs.
Section 10 is day-to-day operation, 11 the bill, 12 what went wrong the first times, 13 which
image to build for which machine.

## 1. The scripts, the verbs, the files

Everything is driven by three things you already know from Tutorial 4 plus one new script:

| what | where | verbs |
|---|---|---|
| the AWS bootstrap | `deploy/uncloud/aws.sh` | `up`, `status`, `stop`, `start`, `destroy`, `bucket`, `bucket-rm`, `ecr`, `ecr-rm`, `env` |
| the driver | `deploy/driver.sh` with `DISTRAINER_DRIVER=uncloud` | the cluster verbs of every bed (`build`, `up N`, `down`, `scale N`, `exec-head`, `kill-worker I`, `kill-head`, `endpoint`, `ps`, `logs`) and `machines-up`, `machines-status`, `machines-stop`, `machines-start`, `machines-destroy`, which call the bootstrap when `DISTRAINER_UNCLOUD_PROVIDER=aws` |
| the Just targets | `Justfile` | `just aws-bucket`, `just aws-ecr`, `just aws-machines` (the bootstrap), `just build`, `just up 2`, `just blocks CFG`, `just train CFG`, `just integration S`, `just down` (the driver) |

The bootstrap keeps its state under `.harness/aws/` (gitignored): `env` (everything the driver
needs, read through `DISTRAINER_ENV_FILE`), `ssh_config` (a `Host` per machine) and
`known_hosts`, `instances`, `vpc`, `ecr`, `s3-credentials` (mode 600), `bucket-policy.json`; and
the key pair's private key at `~/.ssh/distrainer-aws.pem`. Its settings are environment
variables (or lines in `.env`), all listed at the top of the script; the ones you will touch
are in section 5.

## 2. Once: the account, a user, the CLI, the network

You need an AWS account and an **IAM user** whose access key the CLI on this Mac uses (never the
root user's). Sign in to the console as root, open IAM, Users, pick or create the user (no
console access needed), and on its "Permissions" tab attach, via "Attach policies directly":

- `AmazonEC2FullAccess`: instances, key pairs, security groups, VPCs;
- `AmazonEC2ContainerRegistryFullAccess`: the image repository and the push (section 4);
- S3 on your bucket, or `AmazonS3FullAccess` if the user has none;
- and, as an inline policy (JSON tab, named `distrainer-harness-iam`), the right to create the
  *second* IAM user the script makes for the containers, `distrainer-harness`, and nothing else
  in IAM. Replace the account id:

```json
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Effect": "Allow",
      "Action": [
        "iam:GetUser", "iam:CreateUser", "iam:DeleteUser", "iam:TagUser",
        "iam:PutUserPolicy", "iam:DeleteUserPolicy", "iam:ListUserPolicies",
        "iam:CreateAccessKey", "iam:DeleteAccessKey", "iam:ListAccessKeys"
      ],
      "Resource": "arn:aws:iam::<account id>:user/distrainer-harness"
    }
  ]
}
```

Then "Security credentials", "Access keys", "Create access key" (use case "Command Line
Interface"); copy both values. On the Mac, install the CLI from Homebrew (the pkg installer ships
an x86_64 build that runs under Rosetta at nine seconds per call) and give it the key:

```bash
brew install awscli psviderski/tap/uncloud   # the CLI, and uc 0.20
aws configure                                 # the key id, the secret, region us-east-1, output json -> the default profile
aws iam get-user                              # answers with the user (`aws sts get-caller-identity` hung on this Mac; skip it)
aws ec2 describe-vpcs --query 'Vpcs[].[VpcId,CidrBlock,IsDefault]' --output text
```

Several profiles: `DISTRAINER_AWS_PROFILE` selects one. Also needed: `ssh`, `uv` (the
repository's), and Docker on the Mac (OrbStack) to build the image.

The script launches into the region's **default VPC** (the private network AWS gives a new
account: an address range such as `172.31.0.0/16`, a subnet per availability zone, an internet
gateway, public addresses on). An old account may have none; the last command above shows
`IsDefault` `True` for at most one VPC. If none: `aws ec2 create-default-vpc` (configuration
only, no cost). Not every zone offers every instance type; the script picks the first default
subnet whose zone lists both of yours (`aws ec2 describe-instance-type-offerings
--location-type availability-zone`), says which, and keeps it for later `up`s. Another region
is `DISTRAINER_AWS_REGION`.

## 3. The store: the bucket and its user

Bucket names are global to all of S3, so first name yours in the two harness configs (the
`storage_path` and `store_root` lines of `examples/hello_blocks/harness-s3.yaml` and
`harness-stream-s3.yaml`; the region and the endpoint too if not `us-east-1`), then:

```bash
just aws-bucket                # deploy/uncloud/aws.sh bucket
echo 'DISTRAINER_ENV_FILE=.harness/aws/env' >> .env   # the driver and the runner read the env file after .env
```

It creates the bucket with public access blocked and a tag `distrainer:cluster=distrainer-aws`
(what `bucket-rm` looks for; a bucket that exists already must carry it, or
`DISTRAINER_AWS_ADOPT_BUCKET=1` says it is yours to tag), an IAM user `distrainer-harness` whose
only policy is that bucket, and one access key for it, kept at `.harness/aws/s3-credentials`
(a fresh key takes about ten seconds to work). Then it writes `.harness/aws/env`: the
`S3_ENDPOINT`, `S3_REGION` and the key that the containers and the scenario runner use.

## 4. The image repository

The image can reach the machines two ways. Without a repository, `just build` builds the Mac's
own architecture and copies the image to every machine with `uc image push`: about ten minutes
for 1.4 GB over a home uplink, three times, and again after every `destroy`. With a private
repository on ECR, `build` pushes once and the machines pull in-region in seconds:

```bash
just aws-ecr                   # deploy/uncloud/aws.sh ecr: the repository `distrainer`, tagged, a lifecycle policy keeping the last five images
cat .harness/aws/env           # now also DISTRAINER_IMAGE=<account>.dkr.ecr.us-east-1.amazonaws.com/distrainer:latest, DISTRAINER_PLATFORMS=linux/amd64,linux/arm64
```

The machines need no AWS credentials for it: uncloud has no registry login of its own, so
`build` takes a twelve-hour token from your CLI (`aws ecr get-login-password`), logs the Mac's
Docker in, and after the push logs each machine in over the ssh route with the same token and
pulls (each machine's Docker keeps that token for its twelve hours, in root's Docker config).
The repository is tagged like the bucket, and `deploy/uncloud/aws.sh ecr-rm` deletes only a
tagged one, with its images (`DISTRAINER_AWS_ADOPT=1` adopts an existing untagged bucket or
repository that is yours). It costs 0.10 USD per GB-month, cents here. The arm64 bed works without a repository (section 7, recipe A);
the x86 bed wants one (recipe B), because a cross-built image would otherwise have to be
pushed three times from the Mac (recipe C).

## 5. Choosing a bed: arm64 or x86

| | arm64 (Graviton) | x86 (amd64), the default |
|---|---|---|
| instance types | `t4g.large` head (2 vCPUs, 8 GB), `t4g.medium` workers (4 GB) | `t3.large` head, `t3.medium` workers |
| settings | `DISTRAINER_AWS_ARCH=arm64`, `DISTRAINER_AWS_HEAD_TYPE=t4g.large`, `DISTRAINER_AWS_WORKER_TYPE=t4g.medium` | the script's defaults |
| the image | what an Apple Silicon Mac builds natively; pushed with `uc image push` or pulled from ECR | built for `linux/amd64` (under Rosetta on the Mac) and pulled from ECR, or cross-built and pushed |
| running cost | about 0.15 USD per hour for the three with their public addresses | about 0.18 USD per hour |
| why | the cheapest bed; the image is the local one | the architecture of most clouds and of GPU hosts (RunPod); one image for here and there |

Put the settings in `.env` before `just aws-machines` (for the bootstrap, `.env` wins over
the env file it wrote; the shell wins over both). One bed at a time: both use the machine names
`aws1`, `aws2`, `aws3` and the uncloud context `distrainer-aws`. Switching is `deploy/driver.sh machines-destroy`, the new
settings in `.env`, `just aws-machines` again (five minutes), and `just build` unless the image
is already in the repository for that architecture. Both beds ran in us-east-1b.

## 6. Bringing the bed up

```bash
just aws-machines              # deploy/uncloud/aws.sh up, about five minutes
export DISTRAINER_DRIVER=uncloud
deploy/driver.sh machines-status     # the instances as AWS and uncloud see them, and the address ssh is admitted from
ssh -F .harness/aws/ssh_config aws1  # a shell on the head machine
```

What `up` creates, all tagged `distrainer:cluster=distrainer-aws` so `status` and `destroy`
find exactly these:

- a key pair, `distrainer`, saved as `~/.ssh/distrainer-aws.pem`;
- a security group `distrainer-uncloud`: ssh (22) admitted from this Mac's public address only
  (asked of checkip.amazonaws.com; `DISTRAINER_AWS_ALLOW_CIDR` to choose, `/24` at the widest;
  `up` and `start` replace the rule when the Mac has moved networks), and UDP 51820 from the
  group itself, so WireGuard runs between the members and nobody else. Nothing else is open:
  the Ray dashboard accepts job submissions from anyone who reaches it and a Mac's public
  address is often a shared NAT, so the dashboard is reached through an ssh tunnel (section 8);
- three instances, `aws1` (the head machine: the Ray head, the Train controller and the driver
  need its 8 GB) and `aws2`, `aws3` (the workers), Ubuntu 24.04 of the chosen architecture, a
  16 GB gp3 disk each, IMDSv2 only, unlimited CPU credits so a busy hour never throttles the
  step pacing the scenarios measure. The names are not `head` and `worker`: those are service
  names, which uncloud's DNS resolves cluster-wide;
- the uncloud cluster, context `distrainer-aws`: `uc machine init` on `aws1` and `uc machine
  add` for the others, over the public addresses, with each instance's **private** address as its
  WireGuard endpoint (stable across a stop and a start, inside the group's rule), ingress off,
  the cluster network `10.210.0.0/16` checked against the VPC's;
- `.harness/aws/ssh_config` (a `Host` per machine with the key and its own known-hosts file,
  what the driver uses for `docker kill` and the pulls), and the machine lines of
  `.harness/aws/env`: the context, the machine names, `DISTRAINER_UNCLOUD_SSH=%s` with
  `DISTRAINER_UNCLOUD_SSH_OPTS="-F .harness/aws/ssh_config"`, the VPC's CIDR as the host prefix
  (the dashboard binds to the head's private address), and the head's public address for
  `endpoint`.

`uc` runs the system `ssh`, so the instances' host keys are accepted into `~/.ssh/known_hosts`
(a stale entry for a reused address is dropped first). Precedence, two rules for two readers:
for the **driver and the scenario runner**, what your shell exports wins, then the env file,
then `.env` (the env file records what the bootstrap settled, so it overrides your `.env`
defaults); for the **bootstrap itself**, the shell, then `.env`, then the env file (your
settings direct what it builds). Only the uncloud driver and the runner read the env file, and
a command that pins another bed (`just uncloud-machines` sets
`DISTRAINER_UNCLOUD_PROVIDER=orbstack`) skips it, so the OrbStack bed stays reachable while
`.env` points at AWS. One consequence: once `just aws-ecr` has run, the env file names the
registry image, so recipes A and C of section 7 need `DISTRAINER_IMAGE=distrainer:local`
exported in the shell (or `ecr-rm`).

## 7. Building and shipping the image

One Dockerfile, three recipes. `just build` runs the right one from `DISTRAINER_IMAGE` (a local
name, or a registry image: by Docker's rule, a first path component with a dot, a colon or
`localhost`; `myorg/distrainer:local` is still a local name) and `DISTRAINER_PLATFORMS`:

- **A. A local arm64 image for the Graviton bed** (`DISTRAINER_IMAGE=distrainer:local`, the
  default; no repository): `docker build` for the Mac's own architecture, then `uc image push`
  to every machine. About a minute to build (cached afterwards), ten minutes to push the
  1.36 GB dependency layer three times over a home uplink, seconds for a code change afterwards
  (only the small layer moves).
- **B. One image for both architectures, through the repository** (after `just aws-ecr`,
  `DISTRAINER_IMAGE` names it): `docker buildx build --platform linux/amd64,linux/arm64 --push`
  on a BuildKit builder the driver creates once (`docker-container` driver: the plain `docker`
  driver cannot push a multi-platform image), the amd64 half under Rosetta, then a login and a
  `docker pull` on every machine over the ssh route. About four minutes to build both halves
  the first time, then the push once from the Mac (ten minutes on 2026-09-16 for about one
  gigabyte of compressed layers, both halves), the three pulls in-region in seconds. A
  repeat push should move only changed layers, but the measurement says otherwise for now: an
  unchanged image pushed again took eight minutes (five of them uploading), so either the
  builder's cache had let the dependency layer go or its compression made new blobs; an open
  item in the handoff. Works for either bed: each machine pulls its own half of the manifest. The repository then holds one `latest` index with an
  amd64 manifest (about 530 MB compressed) and an arm64 one (about 490 MB).
- **C. A cross-built local image for an x86 bed without a repository**
  (`DISTRAINER_PLATFORMS=linux/amd64`, `DISTRAINER_IMAGE=distrainer:local`): `docker build
  --platform linux/amd64` under Rosetta, then `uc image push` three times. Slower to build and
  the push is still three copies; recipe B is the better x86 path.

The builder's cache is separate from Docker's (`docker buildx ls`; `docker buildx rm distrainer`
drops it). A GPU image (CUDA torch, an x86 CUDA base image) is a separate Dockerfile, not a
platform of this one; section 13 has the table.

## 8. A run

```bash
just build                                             # recipe A, B or C by the settings
just up 2                                              # head + two workers, no MinIO: the store is the bucket
just blocks examples/hello_blocks/harness-s3.yaml      # log and blocks on s3://<bucket>/blocks
just train  examples/hello_blocks/harness-s3.yaml      # hello_s3 on s3://<bucket>/runs; S1 PASS
deploy/driver.sh endpoint                              # dashboard=http://<public address>:8265 (behind the tunnel below), s3=https://s3.us-east-1.amazonaws.com
ssh -F .harness/aws/ssh_config -L 8265:172.31.x.y:8265 aws1   # the dashboard at http://localhost:8265 (the head's private address is in .harness/aws/instances)
```

With `S3_ENDPOINT` naming a store outside the cluster, `endpoint` prints `s3=` instead of
`minio=`: the runner deploys no MinIO, makes no bucket, and runs `harness-s3.yaml` and
`harness-stream-s3.yaml` (the MinIO configs with the bucket, the endpoint and the region
changed; the tests pin them together). The containers carry the config in the image, so a
change to the bucket name needs `just build`. Sections 4 to 6 of Tutorial 4 work as written:
`just kill-worker 2` is `docker kill` over the generated ssh config, `just scale 3` lands the
third worker on a worker instance, `kill-head` and `up 2` recreate the head.

## 9. The scenarios and the numbers

```bash
just integration S2                                    # S3, S4, S8, S9, S10, S11s3 likewise; S6 and S11 skip (no shared mount)
```

On 2026-09-16 every cluster scenario passed on both beds, the bucket in us-east-1; the
OrbStack column is the M6 bed (MinIO on the head machine) for comparison:

| scenario | arm64 bed (`t4g`, recipe A) | x86 bed (`t3`, recipe B) | OrbStack bed, MinIO |
|---|---|---|---|
| S2 worker kill mid-run | 100 s | 107 s | 149 s |
| S3 scale up 2 to 3 | 103 s | 118 s | 146 s |
| S4 scale down 3 to 2 | 179 s | 193 s | 210 s |
| S8 time-budget policy | 87 s | 98 s | 125 s |
| S9 cold restore | 197 s | 212 s | 254 s |
| S10 head loss | 163 s | 161 s | 205 s |
| S11s3 streaming producer | 220 s | 222 s | 161 s |

On the arm64 bed `up` took 78 s, the 240 blocks 71 s to write, the plain run 80 s; on the x86
bed `up` took 42 s, the block step two seconds (the log was already in the bucket from the
arm64 run) and the plain run 32 s, which is not a training time: the run name `hello_s3`
existed in the bucket, so Ray Train restored the finished run (Tutorial 4 section 3 explains;
give a new run a new name). The two beds are within ten percent of each other on every
scenario: the `t3` and `t4g` types pace alike, and the store, not the CPU, sets the step.
The scenarios are faster than on the Mac because `uc deploy` and the restarts are faster on
real machines, while the steps themselves are slower: 0.67 s per step against 0.25 s, because
every checkpoint (one per two steps) is an upload of about a second from the worker to S3.
That is why S11s3 is the one scenario slower than on the Mac: a segment takes about 13 s
against the producer's 6.7 s cadence, so the trainer never catches the producer and the pacing
premise of `docs/uncloud-gotchas.md` ("a streaming trainer starts late") inverts (the checks
still hold: commit precedes consumption, the ranks wait for the end marker). Checkpoint less
often on a real object store when pace matters. Two things had to change in the runner for
this table: the Mac-side client must sign with the bucket's region (MinIO never cared), and the
replay bound of S3 and S4 takes an allowance on a store outside the cluster, because the Train
controller registers checkpoints a couple of reports behind the workers there (its bookkeeping
is S3 round trips) and restores one 24 and 18 positions back in S3 and S4 instead of at most 11
and 14; the ledger keeps every position trained either way.

## 10. Day to day: create, down, park, unpark, destroy

The bed's lifecycle in five verbs, all with `DISTRAINER_DRIVER=uncloud` exported. Read the
table top to bottom as "how much is gone":

| you want | run | it takes | what bills afterwards | what it keeps | back with |
|---|---|---|---|---|---|
| a bed | `just aws-machines` (`aws.sh up`) | about four minutes; then `just build` a minute (the machines pull from the repository) | the instances, about 0.18 USD per hour | | |
| the cluster gone, the machines kept (between two runs) | `just down` | seconds | the instances still, 0.18 USD per hour | the machines, the image on them, the cluster membership | `just up 2`, about 40 s |
| the machines parked (end of the day) | `deploy/driver.sh machines-stop` | a minute | the disks only, about 0.13 USD per day | everything on the disks: the OS, Docker, uncloud, the image, the membership | `deploy/driver.sh machines-start`, about two minutes: new public addresses, the ssh config, the uc context and the ssh rule rewritten |
| the machines gone (end of the work) | `deploy/driver.sh machines-destroy` | a minute | the bucket and the repository, cents a month | the bucket with the block log and the runs, the repository with the image, the env file's store and image settings, the default VPC, the admin user | `just aws-machines` and `just build`, about five minutes |
| everything gone | `machines-destroy`, then `deploy/uncloud/aws.sh bucket-rm` and `ecr-rm` | a minute | nothing | the default VPC and the admin user of section 2 | sections 3 to 6 from the start |

`just down` removes the head and the worker containers, not the machines: the right verb between
two runs on the same day, the wrong one at the end of it. `machines-stop` parks a bed with
containers still deployed too (they are gone when it comes back; `just up 2` again), and is
what `orb stop uc1 uc2 uc3` is for the OrbStack bed. `machines-destroy` is idempotent: if it is
interrupted (the instances gone, the group or the key pair still there), run it again.

```bash
deploy/driver.sh machines-stop        # park: only the disks are billed; the public addresses are released
deploy/driver.sh machines-start       # resume: new addresses; ssh config, known hosts, the uc context and the ssh rule rewritten
deploy/driver.sh machines-status      # the instances, the admitted address, uc machine ls (skipped while parked)
just build                            # after a code edit: the image carries the code (recipe A pushes the small layer only, B pushes and pulls it)
just down                             # the services go; the machines stay
deploy/driver.sh logs head            # uc logs; `.harness/logs/<scenario>.log` on the Mac holds a scenario's driver output
```

Resuming a parked bed changes the public addresses; the bootstrap rewrites everything that
named them, the image and the containers on the machines are untouched. Switching between the
arm64 and the x86 bed is section 5. A machine that must be replaced (a partial `destroy`) is
relaunched by `up` in the same zone.

```bash
deploy/driver.sh machines-destroy     # aws.sh destroy: terminate, delete the group and the key pair, drop the context
deploy/uncloud/aws.sh bucket-rm       # empty and delete the bucket, delete the IAM user and its key
deploy/uncloud/aws.sh ecr-rm          # delete the repository and its images
```

`destroy` leaves the bucket, its user and the repository (the data and the image); the other
two remove those. None of them touches the default VPC or the admin user of section 2.

## 11. What it costs

On-demand in us-east-1 (2026): `t3.large` 0.083 USD per hour, `t3.medium` 0.042; `t4g.large`
0.067, `t4g.medium` 0.034; a public IPv4 address 0.005 per hour each; so about 0.18 USD per hour
for the x86 three while they run, 0.15 for the arm64 three. 16 GB gp3 disks 0.13 USD per day
for the three while stopped (about 4 USD a month for a parked bed). The bucket after a full set
of scenarios held 897 objects and 1.9 MB, a fraction of a cent a month; the repository holds a
few images at 0.10 USD per GB-month. A `t` vCPU busy beyond its baseline (30% on the large, 20%
on the medium) adds 0.04 to 0.05 USD per hour under the unlimited credit setting; the harness
workload sleeps most of the time. `up` prints the running rate. The whole M7 work, two beds and
two full sets of scenarios, stayed under two dollars.

## 12. When something is off

- **`RunInstances ... Unsupported ... in your requested Availability Zone`**: that zone lacks
  the type. The script picks a zone that offers both types; if none does, change the types or
  the region.
- **`no default VPC`**: section 2.
- **`not authorized to perform: ec2:...`, `iam:...` or `ecr:...`**: section 2, the policies.
- **`InvalidClientTokenId` or `SignatureDoesNotMatch` inside the containers** right after
  `aws-bucket`: the new key needs about ten seconds; run `up` and the blocks a little later.
- **ssh refused after moving to another network**: the security group admits the address `up`
  or `start` last saw; either one replaces the rule for the current address (the parked bed
  needs `start` anyway).
- **`pull failed` from `build`**: the ECR token is twelve hours old at most, `build` takes a
  fresh one; a machine that cannot reach ECR has no route out (the default VPC's internet
  gateway) or `ecr:GetAuthorizationToken` was denied to your user.
- **The script stops with no message** after an `aws` call that was allowed to fail: macOS bash
  3.2 and the `command` builtin, or a failing command substitution in an assignment
  (`docs/uncloud-gotchas.md`); the script avoids both, keep it so.
- **Machines `Suspect` in `uc machine ls`** for a minute after a join or a start: wait; the
  bootstrap waits up to five minutes for `Up`.
- **`S3 FAIL: attempt 1 replays 24 positions (> 11)`** or an `HTTP status 400 ... HeadObject`
  from the runner: the two S3 behaviours of section 9, both handled by the current runner.
- The rest of the cluster behaviour is Tutorial 4 section 10 and `docs/uncloud-gotchas.md`.

## 13. Which image for which machine

The Dockerfile is the same for every architecture; what differs is what you build and where it
runs:

| machines | build | how it gets there |
|---|---|---|
| OrbStack machines (arm64) | `just build` with `DISTRAINER_IMAGE=distrainer:local`: `docker build`, the Mac's own architecture | `uc image push` |
| Graviton EC2 (`DISTRAINER_AWS_ARCH=arm64`, `t4g`) | the same local build (recipe A), or the registry image (recipe B) | `uc image push`, or a pull of the arm64 half of the manifest |
| x86 EC2 (the default, `t3`) | the registry image (recipe B): `docker buildx build --platform linux/amd64,linux/arm64 --push` | every machine pulls its own architecture |
| x86 without a repository | recipe C: `DISTRAINER_PLATFORMS=linux/amd64 just build`, a cross-build under Rosetta | `uc image push`, three copies |
| GPU hosts (RunPod, x86 with NVIDIA) | not this image: CPU torch from the CPU wheel index; a CUDA base image and a CUDA torch wheel are a second Dockerfile, the next stage | a registry RunPod can log in to |

## 14. Elsewhere than AWS

Any S3-compatible service (R2, Backblaze, MinIO on a machine of yours) is the same `.env` lines
(`S3_ENDPOINT`, `S3_REGION`, `S3_ACCESS_KEY`, `S3_SECRET_KEY`) and a copy of `harness-s3.yaml`
with its endpoint and bucket; any registry the machines can log in to works the same way as ECR
once `DISTRAINER_IMAGE` names it (the token step is ECR-specific; elsewhere log the machines
in yourself); the machines come from wherever, joined with `uc machine init` and `uc machine
add` as `aws.sh` does, with `DISTRAINER_UNCLOUD_MACHINES`, `DISTRAINER_UNCLOUD_CONTEXT`, the ssh
route and `DISTRAINER_UNCLOUD_HOST_PREFIX` set by hand in `.env`. Azure Blob Storage is not
S3-compatible: MinIO on a VM there, or an S3 gateway.
