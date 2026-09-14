# Handoff: after M5 (KubeRay): uncloud (mode C) and the open follow-ups

Written 2026-09-14 at the end of the M5 session for the thread that picks up the next step. Read
`CLAUDE.md` and `AGENTS.md` first (workflow rules), then this file, then `docs/running-modes.md`
and spec sections 9 and 11. Verify the Gest ids and branch state with `gest task show` and
`git status` before relying on them.

## 1. Where things stand

- M0 to M4 are merged to `main` (PRs #8 to #12). M5, the KubeRay variant, is the branch
  `gest/lntryyqu-m5-kuberay` (Gest parent `lntryyqu`, issue #7, iteration `ukvruwwu`, leaves
  `ykkvnqtp` manifests + driver and `ywuxrspm` scenarios); see the PR for its merge state.
- Under `DISTRAINER_DRIVER=kuberay` the cluster scenarios run against pods on OrbStack's
  Kubernetes with `run_scenarios.py` and `check_audit.py` unchanged (section 3 lists the results).
- Nothing is scheduled for uncloud (mode C) yet: `deploy/drivers/uncloud.sh` is still the
  documented stub. Section 6 is the plan; create it as the next development iteration with `gpl`
  (M6 in the numbering of spec section 12, which today ends at M5).

## 2. Environment checklist

```bash
just setup                        # uv sync; ray[data,default,train] since M5 (the dashboard extra)
just verify                       # laptop-only gate
just build                        # the node image; rebuild when uv.lock or deploy/Dockerfile changes
orb config set k8s.enable true    # once; OrbStack must restart (orb stop && orb start) to apply it
just kuberay-operator             # once per Kubernetes cluster: KubeRay v1.7.0 through kubectl's kustomize
DISTRAINER_DRIVER=kuberay just up 2 && DISTRAINER_DRIVER=kuberay just blocks && DISTRAINER_DRIVER=kuberay just train
DISTRAINER_DRIVER=kuberay just integration S2
just up 2 && just integration S2 && just down    # the compose harness still works the same way
```

Memory on the 16 GB Mac: the OrbStack VM has 8 GB; k3s adds a few hundred MB to the head's
~1.4 GB and ~0.6 GB per worker. Run one cluster scenario at a time (either driver), and do not
launch subagents that start Ray or containers while one runs. Bring the compose containers down
before using the KubeRay driver and vice versa; both bind the same `.harness/shared`. A stale
`dagger-engine` container from an old project autostarts with OrbStack and holds memory; stop it
if a scenario is short of RAM.

## 3. What M5 delivered (spec section 11, `docs/running-modes.md` D)

- `deploy/k8s/raycluster.yaml`: the `RayCluster` (head without `trainer`, `workers` group with
  `trainer: 1` per pod, hostPath volumes for `/shared` and the source tree,
  `terminationGracePeriodSeconds: 10`); `minio.yaml` (Deployment, Service `minio:9000`, PVC);
  `rayjob.yaml` (hello_blocks as a `RayJob`, `submit [CONFIG]`).
- `deploy/drivers/kuberay.sh`: every verb of `deploy/driver.sh` on `kubectl`, plus `operator`
  and `submit`. `up N` renders the placeholders (`__SHARED__`, `__ROOT__`, `__IMAGE__`,
  `__REPLICAS__`, `__CONFIG__`, and the `__S3_*__` settings from `.env` with compose's defaults)
  and applies; `scale N` JSON-patches `replicas`; `kill-*` force-delete pods. A failing
  `kubectl` fails the verb (never "no pods" by mistake: `wipe-shared` and `nuke` delete the
  shared directory only after a successful empty listing).
- `just kuberay-operator`; `tests/test_deploy_manifests.py` pins the manifests' contract.
- Image: `ray[default]` (dashboard and agent), `procps` (the runner's abort `pkill`), a
  `/etc/profile.d` entry that keeps the virtualenv on `PATH` in login shells.
- Scenario results under the KubeRay driver on 2026-09-14: S1 (`just train`, the S1 check in
  `train.py`), S2, S3, S4, S6, S8, S9, S10, S11 and S11s3 all PASS with `run_scenarios.py` and
  `check_audit.py` unchanged. Typical wall times: S2 ~80 s, S4 ~140 s, S9 ~190 s, S11s3 ~85 s.
  S1 also ran as a `RayJob` through `submit`.

## 4. Behaviours learned in M5 that will bite again

- **KubeRay's probes need the dashboard agent.** Ray's minimal install (`ray[data,train]`)
  disables the dashboard's HTTP server ("http server disabled" in `dashboard.log`), the readiness
  probe on port 52365 never passes, and the pods stay `0/1 Running` forever while `ray status`
  looks healthy. `ray[default]` fixes it; `ray job submit` needs it as well.
- **Graceful deletion is a preemption notice, not a death.** KubeRay adds a `preStop` hook that
  runs `ray stop`; Ray drains the node and Train logs "received preemption signal" with a
  deadline, and the ranks keep training until the pod is killed at
  `terminationGracePeriodSeconds`. With the Kubernetes default (30 s) S4's short run finished at
  world size 3 and the check saw one attempt; 10 s matches `docker stop` and restores the 3 -> 2
  restart. `kill-worker` and `kill-head` use `--grace-period=0 --force` and are unaffected.
- **Reverse DNS is on the training start-up path.** `torch.distributed` reverse-resolves the
  peer on every process-group connection; CoreDNS serves PTR records only for pods behind a
  headless Service and forwards everything else (`fallthrough in-addr.arpa`) to the resolver
  outside the cluster. During one window the upstream timed out, every lookup cost 15 s (the
  c10d warning "hostname of the client socket cannot be retrieved" is that timeout, not a missing
  record), trainers took ~40 s to start, and S11's producer ran away from the trainer. The
  headless `distrainer-workers` Service plus `dnsConfig` (`timeout: 2`, `attempts: 1`) in
  `raycluster.yaml` keep those lookups local; Docker's embedded DNS does the same for compose.
  If a KubeRay trainer is slow to reach "Started training worker group", check that first.
- **The head service is headless** (`clusterIP: None`): the dashboard URL uses the head pod IP
  (`.status.head.podIP` of the RayCluster); OrbStack routes pod and service IPs from the Mac.
- **Enabling Kubernetes restarts OrbStack** (`orb config set k8s.enable true` prints "Restart
  OrbStack"); every container stops with it. OrbStack's Kubernetes runs on the Docker engine, so
  `docker stats` lists the pods and locally built images work with `imagePullPolicy: IfNotPresent`.
- **Patch `replicas` with a JSON patch.** A merge patch on `workerGroupSpecs` replaces the whole
  list. Worker pod names change on replacement, so the driver's worker index is creation order.
- **Debian's `/etc/profile` resets `PATH`** in login shells; KubeRay 1.7 ran the Ray command with
  `bash -c` here, but the `/etc/profile.d` guard in the image keeps `ray` reachable if a version
  or a tool uses `bash -lc`.
- **`kubectl apply --server-side -k <kustomize url>`** installs the operator without Helm and is
  idempotent; the base targets the `default` namespace.
- Everything in the M3 and M4 lists still applies (`docs/handoff-m5.md` section 4).

## 5. Review follow-ups still open

The M4 list in `docs/handoff-m5.md` section 5 is unchanged. New in M5:

- `stop-worker` under KubeRay ends in a replacement pod (the operator keeps `replicas`), where
  compose leaves the container stopped; no scenario uses the verb, but the semantics differ.
- `DISTRAINER_RESTART_DELAY=0` (a killed worker stays dead) has no KubeRay equivalent; a scenario
  that needs it would scale down instead.
- The RayJob path (`submit`) is a manual check, not a scenario.
- The compose driver's dashboard on `127.0.0.1:8265` only works now that `ray[default]` is in the
  image; the tutorials never mention it.

## 6. uncloud (mode C) pointers

uncloud is a set of Docker hosts joined by a WireGuard mesh, driven by a Compose-compatible file
(`uc deploy`, `uc scale`, `uc exec`); `docs/running-modes.md` C and the M4 handoff record the
plan: fake the machines with OrbStack Linux machines (`orb create ubuntu uc1`, SSH as
`uc1@orb`), each with Docker and the uncloud daemon, joined with `uc machine init` /
`uc machine add`, MinIO on one of them. The `uc` CLI is not installed on this Mac.

What it needs beyond a driver, in the order to do it:

1. **A cluster bootstrap script** under `deploy/uncloud/` (create the machines, install Docker and
   uncloud, init and join, push the image with `uc image push` or a registry the machines can
   pull from). `build` becomes build + push.
2. **`deploy/drivers/uncloud.sh`** with the common verbs: `up N [minio]` = `uc deploy` of
   `deploy/docker-compose.yml` with `x-` placement plus `uc scale worker N`; `kill-worker I` =
   `uc rm --force` (or `orb stop uc2` for a real machine loss); `exec-head` = `uc exec`;
   `cp-from-head` = `uc cp`; `shared` prints nothing because no volume spans machines.
3. **The runner without a shared mount.** S2, S3, S4 read the audit trail from
   `shared()/blocks/audit` on the Mac; under uncloud the store must be S3
   (`examples/hello_blocks/harness-minio.yaml`), so `wait_for_blocks`, `records_for` and `fresh`
   need the bucket path that S9 and S10 already use (`wait_for_blocks_s3`, `head_python`,
   `s3_rm`). Make the choice a driver capability (`shared` empty means S3 only) rather than a
   scenario flag, and keep `check_audit.py` unchanged.
4. **Docs**: `docs/running-modes.md` C, the spec's section 9 driver list, README.

Acceptance mirrors M5: S2, S3, S4 (on S3), S9, S10 under `DISTRAINER_DRIVER=uncloud`.
