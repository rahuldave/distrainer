# Handoff: after M6 (uncloud): the cloud stage and the open follow-ups

Written 2026-09-14 at the end of the M6 session for the thread that picks up the next step. Read
`CLAUDE.md` and `AGENTS.md` first (workflow rules), then this file, then `docs/running-modes.md` C,
`docs/tutorials/uncloud.md` and `docs/uncloud-gotchas.md`. Verify the Gest ids and branch state with
`gest task show` and `git status` before relying on them.

## 1. Where things stand

- M0 to M6 are merged to `main` (PRs #8 to #15). M6, the uncloud variant (mode C), is the squash
  commit `ffdc9ad` (PR #15, issue #14 closed; Gest parent `wzmmpops`, iteration `yqpnypqm`, leaves
  `kptymlzx` bootstrap and spike, `vrsqosrp` compose file, driver and contract tests, `tzloklqq`
  the runner without a shared mount and the scenario runs, `utxlkkmx` docs, all done); its branch
  is deleted.
- Under `DISTRAINER_DRIVER=uncloud` the bucket scenarios run against three OrbStack machines
  with `run_scenarios.py` reading the bucket and `check_audit.py` unchanged (section 3 has the
  numbers). S6 and S11 need a shared mount and skip.
- Nothing is scheduled for the cloud stage yet (real VMs, S3 or R2): section 6 is the plan. It is
  Gest task `rpmyqnzs` under the v0.1 root (`qvuukpsm`), not in an iteration; create the
  iteration with `gpl` when it starts, as `docs/handoff-m6.md` did for M6.

## 2. Environment checklist

```bash
just setup && just verify                    # laptop-only gate
brew install psviderski/tap/uncloud          # uc 0.20.0 (the driver was built against it)
just uncloud-machines                        # deploy/uncloud/machines.sh up: uc1 (5G), uc2, uc3 (2G) joined as context `distrainer`
export DISTRAINER_DRIVER=uncloud
just build                                   # docker build + uc image push (about 3 minutes the first time)
just up-minio 2 && just mkbucket && just blocks examples/hello_blocks/harness-minio.yaml \
    && just train examples/hello_blocks/harness-minio.yaml && just down
just integration S2                          # S3, S4, S8, S9, S10, S11s3 likewise; one at a time
```

On this Mac the machines `uc1`, `uc2`, `uc3` already exist (created by hand during the session,
before the script did it, so they are not in `.harness/uncloud/machines` and
`machines-destroy` removes them from the cluster but leaves them in place; `orb delete -f ucN`
deletes them). Their caps were raised in place with `orb config set machine.NAME.memory_mib`
(5120, 2048, 2048); `machines.sh up` warns when an existing cap is below its default. Memory:
the OrbStack VM has 8 GB and the machines share it with Docker and the still-enabled Kubernetes
(k3s and the KubeRay operator): compose containers and KubeRay pods down before `up`, one
cluster scenario at a time, no Ray-starting subagents while one runs. A stale
`dagger-engine-v0.15.3` container autostarts with OrbStack and was stopped; stop it again if a
scenario is short of RAM. `orb stop uc1 uc2 uc3` parks the machines; `orb start` brings the
cluster back.

## 3. What M6 delivered (spec sections 9 and 11, `docs/running-modes.md` C)

- `deploy/uncloud/machines.sh` (`up | status | destroy`): the machines with `orb create` and
  memory and CPU caps, the cluster with `uc machine init uc1@orb` and `uc machine add`
  (uncloud runs the system `ssh`, so OrbStack's route works as it is), each machine's address on
  OrbStack's machine network as its explicit WireGuard endpoint, ingress off, the uncloud
  subnet checked against OrbStack's, a wait until every machine is `Up`; idempotent; `destroy`
  deletes only the machines it created and drops the uc context atomically when none is left.
- `deploy/uncloud/compose.yml`: the same node image, scripts, Ray environment, `shm_size`,
  health checks and MinIO as `docker-compose.yml`, pinned against it by
  `tests/test_deploy_manifests.py`; `head` and `minio` on the head machine, `worker` replicas on
  the other machines (a comma-separated `x-machines`), `deploy.replicas` from `up N`, ports
  published only inside a host prefix, no build, no bind mounts, `pull_policy: never`.
- `deploy/drivers/uncloud.sh`: every common verb on `uc`; `shared` prints nothing; node death is
  `docker kill` over ssh on the machine; `cp-from-head` streams a tar; `down` removes every
  copy of a service by id and `up` heals a duplicated name; one retry for deploy and scale;
  a failing or reshaped listing fails the verb; driver-only `machines-up|status|destroy`.
  `just uncloud-machines`; the Justfile's `bash -n` now checks every deploy script (it had only
  ever checked the first).
- `integration_tests/cluster/run_scenarios.py`: the `Store` abstraction (the shared mount, or
  the bucket reached from the Mac through the driver's `endpoint`); S2, S3, S4, S8 on the
  bucket when nothing is shared; `SkipScenario` for S6 and S11; the bucket streaming scenario
  runs 14 producer segments. `check_audit.py` unchanged. Unit tests in
  `tests/test_run_scenarios.py`.
- Results under `DISTRAINER_DRIVER=uncloud` on 2026-09-14 after the resize and the placement
  rule: S2 149 s, S3 146 s, S4 210 s, S8 125 s, S9 254 s, S10 205 s, S11s3 161 s, all PASS; S6
  and S11 SKIP. Compose regression: S2 90 s, S11s3 114 s PASS. Per-step pacing at world size 2
  is nominal; the extra wall time is `uc deploy` (60 to 90 s per `up`) and object-store round
  trips over the mesh.
- Docs: tutorial 4, `docs/uncloud-gotchas.md`, running-modes C, spec sections 9 and 11, the
  "Under uncloud" verb map, README, cli, `.env.example`.

## 4. Behaviours learned in M6 that will bite again

The long form is `docs/uncloud-gotchas.md`. The ones that shaped the code:

- **Memory caps on the head machine decide everything.** With the Ray head, the Train
  controller, the driver script, MinIO and one worker on a 3 GB (then 4 GB) machine, the cgroup
  thrashed: millions of cap hits, VM load above 200, `uc` losing its ssh session, Ray GCS
  keepalive errors, a raylet exiting "mistakenly marked as dead by the GCS" (the harness declares
  a node dead after 3 s), the Mac unable to reach MinIO. Steps stall for tens of seconds while
  the median step stays nominal, so read `memory.events` and `memory.stat` refaults before
  blaming code. 5 GB for the head machine and workers pinned elsewhere fixed it.
- **A worker next to the head starves it**: world size 3 ran five times slower until the
  workers were pinned to the other machines.
- **uncloud can register a service name twice** (after a `down` that failed mid-way); then
  every name-taking `uc` command refuses and `uc ls` grows an ID column. Remove by id.
- **Membership flaps**: a machine shows `Down` for under a minute right as a container starts
  there and the deploy fails with `start container: not found`; one retry reconciles.
- **The S11 pacing premise depends on start-up latency**: Ray Train takes 20 s to start across
  machines and a segment costs 4 to 6 s over the mesh against a 6.7 s producer cadence, so 10
  segments never let the trainer catch up (the bucket variant runs 14).
- `uc` details: `search internal.` makes bare names resolve; `x-machines` takes a comma string,
  not a space one; `uc volume rm` ignores `UNCLOUD_AUTO_CONFIRM`; `uc exec` chatter is on stderr;
  no `uc cp`, no per-container kill, no `uc ctx rm`.
- Everything in the M3 to M5 lists still applies (`docs/handoff-m6.md` section 4).

## 5. Review follow-ups still open

The M4 and M5 lists in `docs/handoff-m6.md` section 5 are unchanged. New in M6:

- The worker index sorts by machine then container id: stable across a kill and restart, not
  across a scale or a redeploy (documented; no scenario kills after scaling).
- `deploy/uncloud/machines.sh`: the create, add and destroy paths were exercised on a fourth
  machine (`uc4`, then removed); the three machines of this Mac were made by hand with the same
  commands. After that join and leave the other machines showed `Suspect` for minutes.
- No scenario uses `cp-from-head` or `stop-worker` under any driver.
- S6 and S11 stay shared-mount scenarios by config; a bucket variant of S6 would need
  `harness-remine.yaml` on S3.
- `uc rm` with several services in one call printed nothing and removed nothing once; the
  driver removes one per call.
- The harness health-check threshold (3 s) is aggressive for a mesh under load; it was kept
  because S2 relies on fast dead-node detection.
- The M6 commits' subjects are within 72 characters; PR #15's review packet is in the PR.

## 6. The cloud stage (real machines, S3 or R2): pointers

Only `machines.sh` is OrbStack-specific. What a cloud run needs, in order (Gest task under the
M6 parent, tagged `uncloud`, unscheduled):

1. **Machines**: two or three VMs (Ubuntu 22.04+ or Debian 11+; arm64 works, amd64 needs the
   image built for it), ssh as root or a passwordless-sudo user, UDP 51820 open between them
   (and only them), 5 GB or more for the head VM. On AWS the CLI on this Mac is authenticated:
   t4g instances in one security group, an ssh key in `~/.ssh/config` as `Host head`,
   `worker-a`, `worker-b` so `DISTRAINER_UNCLOUD_SSH='%s'` works. `uc machine init ubuntu@<ip>
   -c distrainer-aws -n head --no-caddy --no-dns`, `uc machine add ... -n worker-a`, `-n worker-b`.
2. **Driver settings** in `.env`: `DISTRAINER_UNCLOUD_CONTEXT`, `DISTRAINER_UNCLOUD_MACHINES="head
   worker-a worker-b"`, `DISTRAINER_UNCLOUD_SSH`, `DISTRAINER_UNCLOUD_HOST_PREFIX` (the VPC
   subnet). `just build` pushes the image over ssh (1.5 GB the first time).
3. **Storage**: MinIO on the head VM works unchanged. For S3: a bucket, an IAM user with
   `S3_ACCESS_KEY` / `S3_SECRET_KEY` in `.env`, `S3_ENDPOINT=https://s3.<region>.amazonaws.com`,
   `S3_REGION`, and a copy of `harness-minio.yaml` with the same endpoint and region and the
   bucket in `storage_path` / `store_root`; the runner's `Store` takes the endpoint from the
   driver's `endpoint` verb, so either point `DISTRAINER_UNCLOUD_HEAD_ADDRESS` at nothing and
   teach `endpoint` to print the configured S3 endpoint when MinIO is not deployed (small
   driver change), or run S9/S10-style checks from the head. R2 is the same with its account
   endpoint. Azure Blob is not S3-compatible: MinIO on a VM, or an S3 gateway.
4. **Acceptance**: S2, S3, S4, S9, S10 on S3 under `DISTRAINER_DRIVER=uncloud` with the cloud
   context; note wall times against section 3 and the S11 pacing premise.
5. **Docs**: tutorial 4 section 9 becomes a walkthrough; `docs/uncloud-gotchas.md` gets the cloud
   entries; the spec's section 11 paragraph records the result.
