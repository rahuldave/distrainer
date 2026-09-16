# Tutorial 5: the cluster on AWS, with S3 as the store

[Tutorial 4](uncloud.md) ran `hello_blocks` on three OrbStack machines joined by uncloud. This
one runs the same driver, the same compose file, the same configs and the same scenario checks on
**EC2 instances**, with an **S3 bucket** as the store instead of MinIO. Nothing in the training
code changes; what changes is how the machines come to exist (`deploy/uncloud/aws.sh` instead of
`deploy/uncloud/machines.sh`), where the store is, and how the Mac reaches both. Budget: about
0.15 USD per hour while the instances run; the whole set of scenarios cost well under a dollar.

Sections 1 and 2 are the one-time account work (an IAM user with the right permissions, the CLI).
Sections 3 to 7 are the run. Section 8 is the bill. Section 9 is what went wrong the first time.

## 1. Once: the account, a user, the CLI

You need an AWS account and a way for the CLI on this Mac to act in it. Do not use the root
user's credentials for that; make (or reuse) an **IAM user** with an access key:

1. Sign in to the console as the root user (the account email). Open IAM, Users, and either pick
   an existing user or "Create user" (a name such as `distrainer-admin`, no console access).
2. On the user's "Permissions" tab, "Add permissions", "Attach policies directly":
   `AmazonEC2FullAccess` (instances, key pairs, security groups, VPCs) and, if the user has no
   S3 rights yet, `AmazonS3FullAccess` (or a policy on your bucket only).
3. "Add permissions" again, "Create inline policy", JSON tab: the block below, named
   `distrainer-harness-iam`. It lets the user create the *second* IAM user the script makes for
   the containers, `distrainer-harness`, and nothing else in IAM. (`IAMFullAccess` works too and
   is broad.) Replace the account id.

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

4. "Security credentials" tab, "Access keys", "Create access key", use case "Command Line
   Interface", and copy the key id and the secret (shown once).

On the Mac, install the CLI from Homebrew (the pkg installer ships an x86_64 build that runs
under Rosetta at about nine seconds per call) and give it the key:

```bash
brew install awscli
aws configure                  # the key id, the secret, region us-east-1, output json -> the default profile
aws iam get-user               # answers with the user; `aws sts get-caller-identity` hung on this Mac, skip it
aws ec2 describe-vpcs --query 'Vpcs[].[VpcId,CidrBlock,IsDefault]' --output text
```

If you keep several profiles, set `DISTRAINER_AWS_PROFILE` (in `.env` or the shell) instead of
making this one the default. Also needed: the `uc` CLI (`brew install psviderski/tap/uncloud`),
`ssh`, `uv` (the repository's), and Docker on the Mac (OrbStack) to build the image.

## 2. The account's network: a default VPC, a zone with Graviton

The script launches into the region's **default VPC**, the private network AWS gives a new
account (an address range such as `172.31.0.0/16`, a subnet in every availability zone, an
internet gateway, public addresses on). An old account may have none; the last command above
shows `IsDefault` `True` for at most one VPC. If none:

```bash
aws ec2 create-default-vpc    # network configuration only, no cost
```

The instance types are **Graviton** (`t4g`: AWS's own arm64 processors, the cheapest family, and
the image built on an arm64 Mac runs on them as it is). Not every zone offers them; the script
picks the first default subnet whose zone lists both types in
`aws ec2 describe-instance-type-offerings --location-type availability-zone`, and says which.
Another region is `DISTRAINER_AWS_REGION` in `.env`.

## 3. The bucket and its user

Bucket names are global to all of S3, so first name yours in the two harness configs (the
`storage_path` and `store_root` lines of `examples/hello_blocks/harness-s3.yaml` and
`harness-stream-s3.yaml`; the region and the endpoint too if not `us-east-1`), then:

```bash
just aws-bucket               # deploy/uncloud/aws.sh bucket
```

It creates the bucket with public access blocked, an IAM user `distrainer-harness` whose only
policy is that bucket (list, get, put, delete, and the multipart calls), and one access key for
it, kept at `.harness/aws/s3-credentials` (mode 600; a fresh key takes about ten seconds to
work). Then it writes `.harness/aws/env`, the file every driver and the scenario runner read
after `.env` when `.env` says so:

```bash
echo 'DISTRAINER_ENV_FILE=.harness/aws/env' >> .env
cat .harness/aws/env          # S3_ENDPOINT, S3_REGION, the key; the machine lines appear after `up`
```

## 4. The machines

```bash
just aws-machines             # deploy/uncloud/aws.sh up, about five minutes
```

What it creates, all tagged `distrainer:cluster=distrainer-aws` so `status` and `destroy` find
exactly these:

- a key pair `distrainer`, saved as `~/.ssh/distrainer-aws.pem`;
- a security group `distrainer-uncloud`: ssh (22) admitted from this Mac's public address only
  (asked of checkip.amazonaws.com; `DISTRAINER_AWS_ALLOW_CIDR` to choose, `/24` at the widest;
  `up` and `start` replace the rule when the Mac has moved networks), and UDP 51820 from the
  group itself, so WireGuard runs between the members and nobody else. Nothing else is open:
  the Ray dashboard accepts job submissions from anyone who reaches it and a Mac's public
  address is often a shared NAT, so the dashboard is reached through an ssh tunnel (section 5);
- three instances, `aws1` (the head machine, `t4g.large`, 2 vCPUs and 8 GB: the Ray head, the
  Train controller and the driver need it) and `aws2`, `aws3` (`t4g.medium`, 4 GB), Ubuntu
  24.04 arm64, a 16 GB gp3 disk each, IMDSv2 only, unlimited CPU credits so a busy hour never
  throttles the step pacing the scenarios measure. The names are not `head` and `worker`: those
  are service names, which uncloud's DNS resolves cluster-wide;
- the uncloud cluster, context `distrainer-aws`: `uc machine init` on `aws1` and `uc machine
  add` for the others, over the public addresses, with each instance's **private** address as its
  WireGuard endpoint (stable across a stop and a start, inside the group's rule), ingress off,
  the cluster network `10.210.0.0/16` checked against the VPC's;
- `.harness/aws/ssh_config`, a `Host` per machine with the key and its own known-hosts file,
  which the driver uses for `docker kill` on a machine and you can use too; and the machine lines
  of `.harness/aws/env`: the context, the machine names, `DISTRAINER_UNCLOUD_SSH=%s` with
  `DISTRAINER_UNCLOUD_SSH_OPTS="-F .harness/aws/ssh_config"`, the VPC's CIDR as the host prefix
  (the dashboard binds to the head's private address, which AWS maps to its public one), and
  the head's public address for `endpoint`.

```bash
export DISTRAINER_DRIVER=uncloud
deploy/driver.sh machines-status          # the instances as AWS and uncloud see them, and the admitted address
ssh -F .harness/aws/ssh_config aws1       # a shell on the head machine
```

`uc` runs the system `ssh`, so the instances' host keys are accepted into `~/.ssh/known_hosts`
(a stale entry for a reused address is dropped first). Precedence: what your shell exports wins
over both files, and the env file wins over `.env`; only the uncloud driver and the scenario
runner read the env file (it belongs to an uncloud bed; the compose and KubeRay drivers never
see it), and a command that pins another bed (`just uncloud-machines` sets
`DISTRAINER_UNCLOUD_PROVIDER=orbstack`) skips it, so the OrbStack bed stays reachable while
`.env` points at AWS.

## 5. The image, the cluster, a run

```bash
just build                                             # docker build on the Mac, then uc image push to the three instances over ssh
just up 2                                              # head + two workers, no MinIO: the store is the bucket
just blocks examples/hello_blocks/harness-s3.yaml      # log and blocks on s3://<bucket>/blocks
just train  examples/hello_blocks/harness-s3.yaml      # hello_s3 on s3://<bucket>/runs; S1 PASS
deploy/driver.sh endpoint                              # dashboard=http://<public address>:8265, s3=https://s3.us-east-1.amazonaws.com
ssh -F .harness/aws/ssh_config -L 8265:172.31.x.y:8265 aws1   # the dashboard at http://localhost:8265 (the head's private address is in .harness/aws/instances)
```

The dashboard line of `endpoint` names where the port is published, on the head's private
address; the security group does not admit it from outside, hence the tunnel. The push carries
the 1.36 GB dependency layer once per instance, from the Mac's uplink; a code
change afterwards moves only the small layer on top. With `S3_ENDPOINT` naming a store outside
the cluster, `endpoint` prints `s3=` instead of `minio=`: the runner deploys no MinIO, makes no
bucket, and runs `harness-s3.yaml` and `harness-stream-s3.yaml`. Sections 4 to 6 of Tutorial 4
work as written: `just kill-worker 2` is `docker kill` over the generated ssh config, `just
scale 3` lands the third worker on a worker instance, `kill-head` and `up 2` recreate the head.

## 6. The scenarios

```bash
just integration S2                                    # S3, S4, S8, S9, S10, S11s3 likewise; S6 and S11 skip (no shared mount)
```

On 2026-09-16, three `t4g` instances in us-east-1b with the bucket in us-east-1, every cluster
scenario passed; the OrbStack column is the M6 bed (MinIO on the head machine) for comparison:

| scenario | AWS bed, S3 | OrbStack bed, MinIO |
|---|---|---|
| S2 worker kill mid-run | 100 s | 149 s |
| S3 scale up 2 to 3 | 103 s | 146 s |
| S4 scale down 3 to 2 | 179 s | 210 s |
| S8 time-budget policy | 87 s | 125 s |
| S9 cold restore | 197 s | 254 s |
| S10 head loss | 163 s | 205 s |
| S11s3 streaming producer | 220 s | 161 s |

`up` took 78 s, the 240 blocks 71 s to write, the plain run 80 s. The scenarios are faster
than on the Mac because `uc deploy` and the restarts are faster on real machines (the OrbStack
VM was sharing 8 GB with everything else), while the steps themselves are slower: 0.67 s per
step against 0.25 s, because every checkpoint (one per two steps) is an upload of about a
second from the worker to S3. That is why S11s3 is the one scenario slower than on the Mac:
a segment takes about 13 s against the producer's 6.7 s cadence, so the trainer never catches
the producer and the pacing premise of `docs/uncloud-gotchas.md` ("a streaming trainer starts late") inverts (the checks still hold:
commit precedes consumption, the ranks wait for the end marker). Checkpoint less often on a
real object store when pace matters. Two things had to change in the runner for this table:
the Mac-side client must sign with the bucket's region (MinIO never cared), and the replay
bound of S3 and S4 takes an allowance on a store outside the cluster, because the Train
controller registers checkpoints a couple of reports behind the workers there (its bookkeeping
is S3 round trips) and restores one 24 and 18 positions back in S3 and S4 instead of at most 11 and 14; the
ledger keeps every position trained either way.


## 7. Park, resume, remove

```bash
deploy/driver.sh machines-stop            # aws.sh stop: only the disks are billed; the public addresses are released
deploy/driver.sh machines-start           # aws.sh start: new addresses; ssh config, known hosts and the uc context's connections rewritten
deploy/driver.sh machines-destroy         # aws.sh destroy: terminate, delete the group and the key pair, drop the context
deploy/uncloud/aws.sh bucket-rm           # empty and delete the bucket, delete the IAM user and its key
```

`destroy` leaves the bucket and its user (the data); `bucket-rm` removes those. Neither touches
the default VPC or the admin user of section 1. A parked bed costs its disks only (about 4 USD
a month for three 16 GB volumes); the bucket after a full set of scenarios held 897 objects and
1.9 MB (the block log, the streaming store, the runs), a fraction of a cent a month, and keeping
it saves the block build next time.

## 8. What it costs

On-demand in us-east-1 (2026): `t4g.large` 0.067 USD per hour, `t4g.medium` 0.034, a public
IPv4 address 0.005 per hour each, so about 0.15 USD per hour for the three while they run;
16 GB gp3 disks 0.13 USD per day for the three while stopped; the bucket a few cents a month for
this much data; data into AWS free, the audit trails the Mac reads out negligible. A `t4g`
vCPU busy beyond its baseline (30% on the large, 20% on the medium) adds 0.04 USD per hour under
the unlimited credit setting; the harness workload sleeps most of the time. `up` prints the
running rate.

## 9. When something is off

- **`RunInstances ... Unsupported ... in your requested Availability Zone`**: that zone has no
  Graviton. The script picks a zone that offers the types; if none does, change
  `DISTRAINER_AWS_HEAD_TYPE` / `DISTRAINER_AWS_WORKER_TYPE` or the region.
- **`no default VPC`**: section 2.
- **`not authorized to perform: ec2:...` or `iam:...`**: section 1, the two policies.
- **`InvalidClientTokenId` or `SignatureDoesNotMatch` inside the containers** right after
  `aws-bucket`: the new key needs about ten seconds; run `up` and the blocks a little later.
- **ssh refused after moving to another network**: the security group admits the address `up`
  or `start` last saw; either one replaces the rule for the current address (the parked bed
  needs `start` anyway).
- **The script stops with no message** after an `aws` call that was allowed to fail: macOS bash
  3.2 and the `command` builtin (`docs/uncloud-gotchas.md`); the script avoids it, keep it so.
- **Machines `Suspect` in `uc machine ls`** for a minute after a join or a start: wait; the
  bootstrap waits up to five minutes for `Up`.
- **`S3 FAIL: attempt 1 replays 24 positions (> 11)`** or an `HTTP status 400 ... HeadObject`
  from the runner: the two S3 behaviours of section 6, both handled by the current runner
  (the replay allowance, the region in the Mac-side environment).
- The rest of the cluster behaviour is Tutorial 4 section 10 and `docs/uncloud-gotchas.md`.

## 10. Elsewhere than AWS

Any S3-compatible service (R2, Backblaze, MinIO on a machine of yours) is the same `.env` lines
(`S3_ENDPOINT`, `S3_REGION`, `S3_ACCESS_KEY`, `S3_SECRET_KEY`) and a copy of `harness-s3.yaml`
with its endpoint and bucket; the machines come from wherever, joined with `uc machine init`
and `uc machine add` as `aws.sh` does, with `DISTRAINER_UNCLOUD_MACHINES`,
`DISTRAINER_UNCLOUD_CONTEXT`, the ssh route and `DISTRAINER_UNCLOUD_HOST_PREFIX` set by hand
in `.env`. Azure Blob Storage is not S3-compatible: MinIO on a VM there, or an S3 gateway. An
amd64 cluster needs the image built for it (`docker buildx build --platform linux/amd64`).
