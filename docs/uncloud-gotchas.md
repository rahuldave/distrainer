# uncloud gotchas

What building and running the uncloud driver (mode C in `docs/running-modes.md`) taught about
[uncloud](https://uncloud.run) 0.20 on 2026-09-14, on three OrbStack machines. Most of it applies
to real machines as well. The driver (`deploy/drivers/uncloud.sh`), the compose file
(`deploy/uncloud/compose.yml`) and the bootstrap script (`deploy/uncloud/machines.sh`) already
work around every item below; this page exists so nobody rediscovers them. Add to it.

## Cluster and machines

- **`uc machine init|add USER@HOST` runs the system `ssh`**, so anything `ssh` can reach works,
  including an OrbStack machine as `ucN@orb` through the ProxyCommand in `~/.orbstack/ssh/config`
  (no sshd inside the machine). The remote user needs passwordless sudo; the installer adds
  Docker from get.docker.com (unpinned: 29.8 today) and `uncloudd` as a systemd unit.
- **Give every machine its WireGuard endpoint explicitly** (`--wg-endpoint IP:51820`) and disable
  ingress (`--public-ip none`) on a test bed; auto-detection is for machines with a public
  address. `--no-caddy --no-dns` keeps the reverse proxy and the public DNS registration out.
- **Membership takes a while to settle.** After a join or a leave, machines (not only the new
  one) show `Suspect` or `Down` in `uc machine ls` for minutes, and the first ping to a new
  WireGuard peer is lost while the handshake completes. Wait for `Up` before deploying
  (`machines.sh up` waits up to five minutes).
- **A machine can flap to `DOWN` for under a minute** with nothing in its journal, most often
  when it is short of memory. A `uc deploy` in that window fails with `start container: not
  found` on that machine; a second deploy reconciles (the driver retries once).
- **`uc machine rm NAME -y` resets the machine** (uninstalls uncloud and its containers);
  `--no-reset` keeps them. **There is no `uc ctx rm`**: a context is removed by editing
  `~/.config/uncloud/config.yaml` (`machines.sh destroy` does it atomically with a backup).
- Every machine runs an `uncloud-corrosion` container, the cluster store. Leave it alone.
- `wg` is not installed on the machines; `apt install wireguard-tools` gives `wg show uncloud`
  (peers, endpoints, last handshake, bytes) when a link is in doubt.

## Networking

- **Two networks per machine**: the WireGuard interface `uncloud` (MTU 1420) and a Docker bridge
  network also named `uncloud`, one `/24` per machine out of the cluster CIDR (`--network`,
  default `10.210.0.0/16`); containers get MTU 1420. On OrbStack the underlay is its machine
  network (`orb config get network.subnet4`, `192.168.138.0/23`); the bootstrap checks the two
  do not overlap.
- **Every container gets `search internal.`**, so a bare service name (`head`, `minio`) resolves
  cluster-wide to all replicas of that service, the same as `<service>.internal`. This is why
  `ray-worker.sh` (`head:6379`) and the harness configs (`http://minio:9000`) run unchanged.
  Reverse lookups of overlay addresses return at once (torch.distributed does one per peer at
  process-group setup; under KubeRay a slow upstream once cost 15 s each).
- The uncloud DNS logs Ray's `metadata.google.internal` probes as "Failed to resolve service
  name" at DEBUG level. Harmless.
- **`ports:` is unsupported** except host mode; publish with `x-ports` and `@host`. A CIDR host
  prefix (`192.168.138.0/23:9000:9000/tcp@host`) binds only to the machine's addresses in that
  prefix; without one, OrbStack forwards machine ports to the LAN (MinIO with the default
  credentials would be reachable from it).

## Compose and deploy

- **Service names are cluster-global, and uncloud can register the same name twice** (seen after
  a `down` that failed mid-way). Then `uc scale|rm|exec|inspect NAME` all refuse ("multiple
  services found ... use the service ID"), and `uc ls` grows an `ID` column it does not print
  otherwise. Remove by id, one `uc rm ID` at a time; the driver's `down` removes every copy by
  id and `up` heals a duplicated name before deploying.
- **`x-machines` takes a list or a comma-separated string** (`"uc2,uc3"`, interpolation
  friendly); a space-separated string is one machine name that matches nothing and the deploy
  fails with "no machines available that satisfy all constraints".
- **Replicas spread by count, not by load.** With no constraint, one worker lands next to the
  Ray head, MinIO and the driver on the head machine and starves it (world size 3 ran five times
  slower). Pin the head and MinIO to the head machine and the workers to the others.
- `depends_on` is deploy ordering only (the worker script retries until the head answers);
  `restart:` is unsupported and every service is `unless-stopped` (harmless: `docker kill` never
  triggers a restart policy); `profiles` are untested here, the driver deploys named services
  instead; `build:` works but a locally built image pushed with `uc image push` and
  `pull_policy: never` is simpler (the 1.36 GB layer reaches three machines in about 3 minutes).
- **`uc deploy` recreates a container it finds stopped**, so `docker kill` followed by `up`
  yields a new container id. The driver's worker index (by machine, then id) is stable across a
  kill and restart, not across a scale or a redeploy: scale first, then look up.
- **Confirmation flags are inconsistent**: `uc deploy` and `uc scale` take `-y` or
  `UNCLOUD_AUTO_CONFIRM=true`; `uc rm` never prompts; `uc volume rm` prompts and ignores the
  variable (`-y` needed, or `nuke` hangs on a prompt nobody sees).
- **`uc exec` picks a random replica** unless `--container ID`; pass `-T` without a TTY. Its
  connection chatter ("Connecting to ssh://...") goes to stderr; stdout stays clean, so pipes and
  tar streams through it work. **There is no `uc cp`** (the driver streams a tar through `exec`).
- **No per-container kill, stop or start.** Node death is `sudo docker kill ID` over ssh on the
  machine that runs the container; `uc ps` shows the machine, and its CREATED and STATUS columns
  contain spaces (the driver parses from the `ago` token).
- `uc rm` of several services in one call printed nothing and removed nothing once; one service
  per call always worked.

## Resources and timing

- **OrbStack machine memory limits are cgroup caps, not reservations, and they count page
  cache.** The head machine (Ray head, autoscaler, dashboard, Train controller, the driver
  script, MinIO) needs about 5 GB; a worker machine about 1 GB per Ray worker, two fit in 2 GB.
  Raise a cap live with `orb config set machine.NAME.memory_mib N` (it takes effect at once).
- **The thrash signature** of a cap that is too small: millions of `max` hits in
  `/sys/fs/cgroup/memory.events`, thousands of `workingset_refault_file` per 10 s in
  `memory.stat`, VM load above 100 (`/proc/loadavg` is VM-wide), `uc` losing its ssh session
  ("error reading server preface: read |0: file already closed"), Ray GCS keepalive errors in the
  driver log, a raylet exiting "mistakenly marked as dead by the GCS" (the harness declares a
  node dead after 3 s of missed heartbeats), the Mac unable to reach MinIO on the head machine,
  steps stalling for tens of seconds while the median step stays nominal.
- **`up` takes 60 to 90 s**: `uc deploy` starts containers one at a time, monitors each for 5 s
  and waits for its health check. A scale-down stops the newest worker in about 12 s (10 s
  grace, a preemption notice to Ray Train as `docker stop` gives). Whole scenarios take 1.5 to
  3 times their compose wall time (S2 149 s against 90 s); per-step pacing at world size 2 is
  nominal, the difference is deploy time and object-store round trips over the mesh.
- **A streaming trainer starts late.** Ray Train takes about 20 s to start a worker group across
  machines, and over the mesh a segment costs the ranks 4 to 6 s against the producer's 6.7 s
  cadence, so the bucket variant of S11 runs 14 segments for the trainer to catch up and wait
  (10 under a shared mount).

## The AWS bed (real machines, S3)

What `deploy/uncloud/aws.sh` and the cloud run of 2026-09-16 taught, on top of the above.

- **macOS bash 3.2 and `command`.** A failing command run through the `command` builtin ignores
  `set -e`'s suppression inside `if` conditions and `||` lists: `f() { command false; }; if ! f;
  then ...` exits the script silently with the command's status, while `f() { false; }` behaves.
  The bootstrap's `awsc` wrapper therefore calls `aws` directly. Every script here runs under
  `/usr/bin/env bash`, which is 3.2 on a Mac without a Homebrew bash, so keep `command` out of
  anything that may fail on purpose.
- **The AWS CLI installed by the pkg is x86_64.** Under Rosetta every call takes about 9 s and
  `aws sts get-caller-identity` never returned on this Mac; `brew install awscli` gives a
  native arm64 build that answers in a second (put `/opt/homebrew/bin` first in `PATH`).
- **Not every zone offers Graviton.** In one account `us-east-1a` had no `t4g` capacity at all
  (`RunInstances ... Unsupported ... choosing us-east-1b, ...`); the bootstrap now picks the first
  default subnet whose zone lists both instance types in
  `describe-instance-type-offerings --location-type availability-zone`. An old account may also
  have no default VPC: `aws ec2 create-default-vpc` makes one (no cost).
- **The recorded settings win over a changed default.** `aws.sh` reads its own
  `.harness/aws/env` first, so `DISTRAINER_UNCLOUD_MACHINES` recorded by an earlier run (the
  names `destroy` must find) outlives a new default in the script; a rename means `destroy`,
  delete the env file, `up`. Set the variable in `.env` or on the command line to override.
- **Real S3 signs by region.** A client with the endpoint and the credentials but the wrong
  region (`auto`, what MinIO accepted for a year) gets `HTTP 400 ... No response body` from
  `HeadObject`. The runner's Mac-side environment now carries `S3_REGION` with the endpoint.
- **On S3 the Train controller registers checkpoints behind the workers.** Every registration
  is S3 round trips in the controller's loop (the checkpoint-manager snapshot, the `num_to_keep`
  deletions), so at a resize it restores a checkpoint one or two reports older than the newest
  upload, and the teardown lets the workers run a few more steps: S3 replayed 24 positions
  against the local bound of 11, S4 18 against 14. The ledger keeps every position trained;
  the check's replay bound takes a `lag_intervals` allowance for a store outside the cluster.
- **Checkpoint latency sets the step pace on S3.** With `every_k: 2` each checkpoint is an
  upload of about a second from a `t4g.medium` to S3: 0.67 s per step against 0.25 s on
  MinIO, and a segment of the streaming scenario takes about 13 s against the producer's
  6.7 s cadence, so the trainer never catches the producer (S11s3 still passes its checks:
  commit precedes consumption, the ranks wait for the end marker). Checkpoint less often
  (`every_k: 6` or a time policy) on a real object store; the harness configs keep `2` so the
  same run is comparable across beds.
