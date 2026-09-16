# Tutorial 4: the same run on a cluster of machines (uncloud)

[Tutorial 3](kuberay.md) ran `hello_blocks` on Kubernetes pods. This one runs the same code, the
same configs and the same checks on **machines**: Docker hosts joined into one cluster by
[uncloud](https://uncloud.run), a WireGuard mesh with cluster DNS driven by a Compose file
(`uc deploy`, `uc scale`, `uc exec`). Nothing in the training code changes. What changes is the
driver behind `just`: `DISTRAINER_DRIVER=uncloud` swaps `deploy/drivers/compose.sh` for
`deploy/drivers/uncloud.sh` (mode C in `docs/running-modes.md`). And one thing is different
from every mode so far: **no volume spans machines**, so the block log, the audit trail and the
checkpoints live on an object store (MinIO on the head machine here, S3 or R2 elsewhere), and
the scenario runner reads them from the bucket instead of a shared directory.

The machines here are three small OrbStack Linux machines on the Mac; section 9 says what
changes for real machines in a cloud. Prerequisites: `just setup`, OrbStack, the `uc` CLI
(`brew install psviderski/tap/uncloud`, 0.20), and memory: the OrbStack VM has 8 GB, so bring the
compose containers down (`just down`) and the KubeRay pods down (`DISTRAINER_DRIVER=kuberay just
down`) first. About forty minutes, most of it waiting for deploys.

## 1. Once: the machines, the cluster, the image

```bash
just uncloud-machines      # deploy/uncloud/machines.sh up: three machines, one uncloud cluster
```

The script creates `uc1` (the head machine, 5 GB, 2 CPUs), `uc2` and `uc3` (2 GB each) with
`orb create`, initialises the cluster on `uc1` with `uc machine init uc1@orb` and adds the other
two with `uc machine add`. `uc` talks to a machine over the system `ssh`, so OrbStack's `ucN@orb`
route works as it is; each machine gets its address on OrbStack's machine network as its
WireGuard endpoint, ingress stays off, and the script checks that uncloud's cluster network
(`10.210.0.0/16`) does not overlap OrbStack's, then waits until `uc machine ls` shows every
machine `Up`:

```
NAME   STATE   ADDRESS         PUBLIC IP   WIREGUARD ENDPOINTS     OS                   ...   DOCKER   VERSION
uc1    Up      10.210.0.1/24   -           192.168.139.221:51820   Ubuntu 24.04.5 LTS   ...   29.8.0   0.20.0
uc2    Up      10.210.1.1/24   -           192.168.139.90:51820    Ubuntu 24.04.5 LTS   ...   29.8.0   0.20.0
uc3    Up      10.210.2.1/24   -           192.168.139.194:51820   Ubuntu 24.04.5 LTS   ...   29.8.0   0.20.0
```

Two networks appear on every machine: the WireGuard interface `uncloud` (the `ADDRESS` column
is the machine's address on it) and a Docker network `uncloud` with one `/24` per machine, on
which the containers live. The cluster is saved as context `distrainer` in
`~/.config/uncloud/config.yaml`; the driver names it on every call. The sizes, names, distro
and network are environment settings listed at the top of `deploy/uncloud/machines.sh`; the
memory numbers matter (section 10).

```bash
export DISTRAINER_DRIVER=uncloud   # for the rest of this tutorial
just build                         # docker build, then `uc image push distrainer:local` to every machine
```

Nothing is pulled from a registry: `build` pushes the locally built image over the same ssh
route, about three minutes for the 1.36 GB dependency layer to three machines, seconds
afterwards for a code change (only the changed layer moves). Under this driver a code edit
needs `just build` again, because the image carries the code: the compose and KubeRay drivers
mount the source tree from the Mac, and no such mount exists across machines.

## 2. Bring a cluster up

```bash
just up-minio 2
```

`up` runs `uc deploy` with `deploy/uncloud/compose.yml`. Read it next to the output: the `head`
and `minio` services are pinned to the head machine (`x-machines`), the `worker` service
(`deploy.replicas: 2`) to the other machines, and every container carries the same Ray
environment, `shm_size` and health checks as `deploy/docker-compose.yml`
(`tests/test_deploy_manifests.py` pins the two files against each other). Storage is MinIO
only: there is no `/shared`, so `just up 2` gives you a cluster the harness configs cannot write
to; always bring MinIO along here.

```
SERVICE   CONTAINER ID   IMAGE                                              CREATED          STATUS                    IP ADDRESS   MACHINE
head      d786360f1206   distrainer:local                                   40 seconds ago   Up 39 seconds (healthy)   10.210.0.2   uc1
minio     fde0dc4c18a9   quay.io/minio/minio:RELEASE.2025-09-07T16-13-09Z   45 seconds ago   Up 44 seconds (healthy)   10.210.0.3   uc1
worker    d49f82839967   distrainer:local                                   32 seconds ago   Up 30 seconds             10.210.1.2   uc2
worker    c5278fa84c43   distrainer:local                                   38 seconds ago   Up 37 seconds             10.210.2.2   uc3
```

Containers start one at a time, each monitored for five seconds and health-checked, so an `up`
takes a minute to a minute and a half. The workers find the head as `head:6379` and MinIO as
`minio:9000`, names that resolve cluster-wide because uncloud gives every container
`search internal.` in its resolver configuration.

```bash
deploy/driver.sh ps        # uc machine ls, then uc ps: every container with its machine
deploy/driver.sh endpoint  # dashboard=http://192.168.139.221:8265, minio=http://192.168.139.221:9000 console=...:9001
```

The dashboard and MinIO are published on the head machine's address only (its OrbStack
address, inside the prefix the driver takes from `orb config get network.subnet4`), so the Mac
reaches them and the LAN does not. Log in to the MinIO console with `distrainer` /
`distrainer123` (the defaults of `docker-compose.yml`; `.env` overrides them).

## 3. Blocks and a run

```bash
just mkbucket                                              # the distrainer bucket, created from the head
just blocks examples/hello_blocks/harness-minio.yaml       # log and blocks on s3://distrainer/blocks
just train  examples/hello_blocks/harness-minio.yaml       # run hello_s3 on s3://distrainer/runs
```

The output is the one you know from Tutorial 1 at the harness size (`W=24`, ten segments, two
workers), a little slower than under compose: every block is a GET from MinIO over the mesh and
every checkpoint a PUT, but the steps themselves keep their pace. The script's own S1 check
prints `S1 PASS`. Look at what was written from the head (its environment names the endpoint
and the credentials):

```bash
deploy/driver.sh exec-head distrainer log-ls s3://distrainer/blocks -v
deploy/driver.sh exec-head distrainer inspect s3://distrainer/runs/hello_s3/checkpoint_g000009_p000024_n02_a00
```

or browse the bucket in the MinIO console. As in Tutorial 3, a run name that already exists on
the bucket is *restored* by Ray Train, not started again, and `train.py` leaves a bucket alone:
give another run a new name (`--set run_name=...` through `deploy/driver.sh exec-head`, as in
the next section; the `just train` recipe takes the config only).

## 4. Kill a worker while it trains

Terminal 1 (every checkpoint kept, so section 6 can pick a mid-run one):

```bash
deploy/driver.sh exec-head python examples/hello_blocks/train.py \
    --config examples/hello_blocks/harness-minio.yaml --set run_name=kill --set checkpoint.num_to_keep=null
```

Terminal 2, about ten seconds later (a new shell: export the driver again):

```bash
export DISTRAINER_DRIVER=uncloud
just kill-worker 2
# worker 2 (c5278fa84c43 on uc3) killed; restarting in 5s as a new Ray node
deploy/driver.sh ps         # Exited (137) ... then Up again
```

uncloud has no per-container kill, so `kill-worker` is `sudo docker kill` over ssh on the
machine that runs the container (index 2 counts worker containers sorted by machine and then by
id), a node death to Ray. Docker never restarts a killed container on its own, so the driver
starts it again five seconds later (`DISTRAINER_RESTART_DELAY`, `0` leaves it dead), and it
rejoins as a new Ray node. In terminal 1, Ray Train restarts the worker group from the last
checkpoint on the bucket; `just integration S2` does this unattended and checks the audit trail,
which it now reads from the bucket through the driver's `endpoint`.

## 5. Scale up, scale down

With a run going in terminal 1, in terminal 2 (`DISTRAINER_DRIVER=uncloud` exported there too):

```bash
just scale 3       # uc scale worker 3: a third worker on uc2 or uc3
just scale 2       # uc scale worker 2: the newest worker is stopped
```

`uc scale` adds a container on one of the worker machines (never the head machine); when its
`trainer` resource appears, Ray Train's elastic monitor restarts the group at world size 3. The
scale-down is graceful: uncloud stops the newest worker with a 10 s grace period, which Ray
reports to the trainer as a preemption notice before the node dies, exactly what `docker stop`
gives under compose and what KubeRay's `terminationGracePeriodSeconds: 10` gives under
Kubernetes. `just integration S3` and `S4` assert the 2 -> 3 and 3 -> 2 transitions from the
trail.

## 6. Lose the head

```bash
deploy/driver.sh kill-head        # head (d786360f1206 on uc1) killed
just up-minio 2                   # uc deploy recreates the stopped head; the workers reconnect
```

The Ray head, the Train controller and the driver script die together, as in scenario S10; the
run in terminal 1 is over. `up` finds the head container stopped and `uc deploy` recreates it
(a new container id, the same address), the worker scripts reconnect to `head:6379`, and the
run continues from one of its checkpoints on the bucket in a new run name. Pick a mid-run
checkpoint of `kill` in the MinIO console under `distrainer/runs/kill/` (the console address is
in `deploy/driver.sh endpoint`; section 4 kept every checkpoint), then:

```bash
deploy/driver.sh exec-head distrainer resume s3://distrainer/runs/kill/checkpoint_g000004_p000024_n02_a00 \
    --config examples/hello_blocks/harness-minio.yaml \
    --entry examples.hello_blocks.train:entry --run-name kill_resume
```

`just integration S10` runs this sequence unattended and checks the resumed trail; `S9` does the
cold restore instead (`down`, then `up-minio`, then a resume from a mid-run checkpoint; the
`wipe-shared` step of the compose version is a no-op here because there is nothing shared to
wipe).

## 7. Tear down

```bash
just down                            # the services go; the MinIO volume on uc1 stays
just nuke                            # also the MinIO volume
orb stop uc1 uc2 uc3                 # park the machines (orb start brings the cluster back)
deploy/driver.sh machines-destroy    # uc machine rm, orb delete, the uc context: gone for good
```

`machines-destroy` deletes only machines the bootstrap created (it keeps a list under
`.harness/uncloud/`); a machine that existed before is removed from the cluster and left in
place.

## 8. The scenarios

```bash
DISTRAINER_DRIVER=uncloud just integration S2      # S3, S4, S8, S9, S10, S11s3 likewise
```

The runner's driver verbs are the same; what differs is where it reads. `deploy/driver.sh
shared` prints nothing under this driver, and the runner then treats the bucket of
`harness-minio.yaml` as the store: it reaches it from the Mac through the `endpoint` verb, runs
S2, S3, S4 and S8 with the MinIO config, and cleans the run state and the audit trail on the
bucket before a run (`check_audit.py` does not change). S6 and S11 store under `/shared` by
config and are skipped with a message. On 2026-09-14 every cluster scenario passed under this
driver: S2 149 s, S3 146 s, S4 210 s, S8 125 s, S9 254 s, S10 205 s, S11s3 161 s.

## 9. Real machines: AWS, and S3 instead of MinIO

Everything above is OrbStack-specific only in `deploy/uncloud/machines.sh` (how the machines
come to exist and how ssh reaches them). [Tutorial 5](aws.md) runs the same driver, compose
file and scenarios on three EC2 instances with an S3 bucket as the store: `deploy/uncloud/aws.sh`
is the twin of `machines.sh` (the same verbs plus `bucket`, `bucket-rm`, `ecr` and `ecr-rm`, driven through the
`machines-*` verbs under `DISTRAINER_UNCLOUD_PROVIDER=aws`), it writes one env file with the
context, the machines, the ssh route and the S3 settings that the driver and the runner read
through `DISTRAINER_ENV_FILE`, and with `S3_ENDPOINT` naming a store outside the cluster the
driver's `endpoint` prints `s3=` instead of `minio=`, so the runner deploys no MinIO and runs
`harness-s3.yaml`. In short:

```bash
just aws-bucket && echo 'DISTRAINER_ENV_FILE=.harness/aws/env' >> .env
just aws-machines
export DISTRAINER_DRIVER=uncloud
just build && just up 2 && just blocks examples/hello_blocks/harness-s3.yaml && just train examples/hello_blocks/harness-s3.yaml
just integration S2
deploy/driver.sh machines-stop        # or machines-destroy
```

Tutorial 5 has the account prerequisites (an IAM user with three policies, the CLI, a default
VPC, a zone with the instance types), the arm64 and the x86 bed, the private image repository
and the three ways to build and ship the image, what the bootstrap builds and why, the numbers,
the bill, and the things that went wrong the first time. Any other S3-compatible store and any other set of
machines work the same way; its last section says how.

## 10. When something is off

The full list is `docs/uncloud-gotchas.md`. The four that cost the most time:

- **Everything is slow, `uc` drops its ssh session, Ray logs GCS keepalive errors, a worker
  exits "mistakenly marked as dead by the GCS".** The head machine is short of memory and
  thrashing. Check `orb -m uc1 cat /sys/fs/cgroup/memory.events` (millions of `max` hits) and
  raise the cap: `orb config set machine.uc1.memory_mib 5120`. The Ray head, MinIO and the driver
  script need about 5 GB together; the bootstrap's defaults assume nothing else runs there.
- **`uc scale` or `uc rm` refuse: "multiple services found with name 'worker'".** uncloud
  registered a service twice. `just down` removes every copy by id; the next `up` deploys clean.
- **`up` fails with "start container: not found" on one machine, and `uc machine ls` shows it
  `Down` or `Suspect` for a moment.** Membership flapped; the driver retries once. If it stays
  down, its uncloud daemon is the thing to look at (`orb -m uc2 systemctl status uncloud`).
- **A worker index points at the wrong container.** The index sorts worker containers by
  machine and then by id, which a scale or a redeploy reshuffles. Read `deploy/driver.sh ps`
  after scaling.

## 11. Where to go next

- `docs/running-modes.md` (mode C): the same material as a reference, next to the laptop,
  compose and KubeRay modes.
- `docs/examples-and-scenarios.md`, "Under uncloud": what every driver verb does on each side.
- `docs/uncloud-gotchas.md`: the running list.
- `docs/handoff-m7.md`: what building this taught, and what comes next (the cloud stage).
