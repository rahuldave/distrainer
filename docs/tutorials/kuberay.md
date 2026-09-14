# Tutorial 3: the same run on Kubernetes (KubeRay)

[Tutorial 1](batch.md) ran `hello_blocks` on a laptop and [Tutorial 2](streaming.md) streamed
into it. This one runs the same code, the same configs and the same checks with **Kubernetes
pods as Ray nodes**, on the Kubernetes that ships with OrbStack, through the KubeRay operator.
Nothing in the training code changes. What changes is the driver behind `just`:
`DISTRAINER_DRIVER=kuberay` swaps `deploy/drivers/compose.sh` (containers on a compose network,
mode B in `docs/running-modes.md`) for `deploy/drivers/kuberay.sh` (mode D). Along the way you
kill a worker, resize the cluster, submit a `RayJob`, and restore a run from MinIO, by hand.

Prerequisites: `just setup`, OrbStack with Docker working, `kubectl` (OrbStack writes a
kubeconfig context named `orbstack`). Memory: the OrbStack VM has 8 GB; stop the compose
containers (`just down` with the default driver) before starting pods. About twenty minutes.

## 1. Once: Kubernetes, the operator, the image

```bash
orb config set k8s.enable true     # OrbStack answers "Restart OrbStack": orb stop && orb start
kubectl get nodes                  # one node, "orbstack", Ready (kubectl config current-context -> orbstack)
just kuberay-operator              # KubeRay v1.7.0: CRDs, RBAC and the operator, in namespace default
kubectl -n default get pods        # kuberay-operator-... 1/1 Running
just build                         # the node image, the same one the compose harness uses
```

`just kuberay-operator` applies KubeRay's kustomize base with `kubectl apply --server-side`
(no Helm on the machine is needed) and waits for the operator's rollout. `just build` builds the
same `distrainer:local` tag under either driver (compose through `docker compose build head`,
kuberay through `docker build`): OrbStack's Kubernetes runs on the same Docker engine, so the
pods use the local image directly (`imagePullPolicy: IfNotPresent`), with no registry to push
to. The image carries `ray[default]`; KubeRay's readiness probes are HTTP checks against Ray's
dashboard agent, which Ray's minimal install does not start (section 9 below).

## 2. Bring a cluster up

```bash
export DISTRAINER_DRIVER=kuberay   # for the rest of this tutorial
just up 2
```

`up` renders `deploy/k8s/raycluster.yaml` (the shared directory, the source tree, the image,
the replica count, and the S3 endpoint and credentials from `.env` are substituted in) and
applies it; the operator creates a head pod and
two worker pods, and `up` returns once the head is Ready:

```
NAME                              READY   STATUS    RESTARTS   AGE   IP
distrainer-head-b8msl             1/1     Running   0          14s   192.168.194.11
distrainer-workers-worker-6swln   0/1     Running   0          14s   192.168.194.10
distrainer-workers-worker-zbg9b   0/1     Running   0          14s   192.168.194.12
```

Read the manifest next to this output. The head's `rayStartParams` give it two CPUs and no
`trainer` resource, so the Train controller and your driver script run there but never a
training worker; each pod of the `workers` group starts Ray with `--num-cpus=1
--resources='{"trainer": 1}'`, exactly like a worker container in the compose harness, and the
harness configs ask for `resources_per_worker: {CPU: 1, trainer: 1}`. Every pod mounts
`.harness/shared` of the repository at `/shared` as a `hostPath` volume (OrbStack's Kubernetes
runs in the VM that already sees the Mac's filesystem), so the block log, the audit trail and
the checkpoints land in the same directory as with compose, and the scenario runner reads them
from the Mac without knowing which driver is active. The source directories are mounted over the
image copy too: edit `examples/hello_blocks/train.py` on the Mac and the next run uses it.

```bash
deploy/driver.sh ps        # the RayCluster (desired/available workers, "ready"), pods, services, the PVC
deploy/driver.sh endpoint  # dashboard=http://<head pod IP>:8265, open it in a browser
```

OrbStack routes pod and service IPs from the Mac, so the dashboard URL just works. Elsewhere,
`kubectl -n distrainer port-forward svc/distrainer-head-svc 8265`.

## 3. Blocks and a run

```bash
just blocks    # make_blocks.py inside the head pod: 240 blocks and 10 segments under /shared/blocks
just train     # train.py inside the head pod
```

The output is the one you know from Tutorial 1, at the harness config's size (`W=24`, ten
segments, two workers, about half a minute): the ranks report sixty times each, the last line is
`final metrics: {... 'position': 238, 'segment': 9, ... 'world_size': 2, 'attempt': 0 ...}`,
and the script's own S1 check prints `S1 PASS`. From the Mac:

```bash
uv run distrainer log-ls .harness/shared/blocks -v     # the log the pods wrote
ls .harness/shared/runs/hello/                          # checkpoint_g000009_p000024_n02_a00 and the two before it
uv run distrainer inspect .harness/shared/runs/hello/checkpoint_g000009_p000024_n02_a00
```

One rule of Ray Train to keep in mind: a run whose name already exists on the storage is
*restored*, not started again, and a completed run has nothing left to do. `train.py` handles it
for you on local storage: it removes the previous run of the same name and its audit trail
before starting, unless you pass `--keep`. With the MinIO config of section 7 it does not
(the bucket is left alone), which is why the resumed run there gets a new name.

## 4. Kill a worker while it trains

Terminal 1:

```bash
just train
```

Terminal 2, about ten seconds later (the first checkpoint exists after two steps):

```bash
just kill-worker 2
# worker 2 (distrainer-workers-worker-zbg9b) killed; KubeRay is starting a replacement pod as a new Ray node
kubectl -n distrainer get pods -w     # the old pod is gone, a new one is Pending, then Running
```

`kill-worker` is `kubectl delete pod --grace-period=0 --force`: the pod object disappears at
once and the container is killed, which is a node death to Ray (the health-check variables in
the manifest make the head declare it in about 3 s). The operator sees one replica missing and
creates a new pod, which joins as a new Ray node (about 15 s later on this Mac). In terminal 1,
Ray Train restarts the worker group from the last checkpoint; the audit trail gets at least one
further attempt (Ray sometimes needs more than one restart while the dead node's heartbeat
times out), so the final metrics show `'attempt': 1` or higher. `just integration S2` does
exactly this and asserts, from the audit trail, that the union of the attempts covers every
position exactly once at world size 2 throughout.

Two differences from the compose harness are worth knowing. The replacement is immediate and
has a new name (index 2 in `kill-worker 2` counts worker pods in creation order, so it now names
the newest pod), and a killed worker always comes back: `DISTRAINER_RESTART_DELAY=0`, which under
compose leaves the container dead, has no equivalent, because the operator's job is to keep
`replicas` pods running. To take a node away for good, scale.

## 5. Scale up, scale down

`harness.yaml` allows `num_workers: [2, 3]`. With a run going in terminal 1, in terminal 2:

```bash
just scale 3       # raycluster.ray.io/distrainer patched
```

`scale` is a JSON patch of the worker group's `replicas` (a merge patch would replace the whole
list of groups). The operator adds a pod; when its `trainer` resource appears, Ray Train's
elastic monitor (`elastic_resize_monitor_interval_s: 5` in the config) restarts the group at
world size 3 from the last checkpoint, and the rest of the log is dealt over three ranks. Then:

```bash
just scale 2
```

Now the operator *removes* a pod, and Kubernetes removes pods politely: the pod gets a
termination notice, KubeRay's `preStop` hook runs `ray stop` in it, Ray tells the trainer
(terminal 1 shows every rank logging `received preemption signal ... preempted_ranks=[1]`), and
the container is killed only when the grace period runs out. The manifest sets
`terminationGracePeriodSeconds: 10`, the same grace `docker stop` gives a container, so about
ten seconds after the patch the node dies, Ray Train restarts at world size 2, and the audit
trail ends in a 3 -> 2 transition; `just integration S4` asserts exactly that from a cluster
that started with three workers. Kubernetes' default grace of 30 s is longer than what is left
of this short run: the drained worker would train to the end at world size 3 and nothing would
ever restart, which is how S4 failed the first time under this driver. (`stop-worker I` is the
same graceful path for one pod, followed by a replacement.)

## 6. The Kubernetes-native way: a RayJob

So far the driver script ran through `kubectl exec` in the head pod, as the scenario runner
does. Kubernetes' own way to run a job is the `RayJob` resource in `deploy/k8s/rayjob.yaml`:

```bash
deploy/driver.sh submit examples/hello_blocks/harness.yaml   # rayjob.ray.io/distrainer-hello created
kubectl -n distrainer get rayjob -w                           # RUNNING ... SUCCEEDED, about a minute
kubectl -n distrainer logs job/distrainer-hello               # the driver's output, ending in S1 PASS
```

The operator creates a submitter Job (a pod from the same image) that runs `ray job submit`
against the head's dashboard; the entrypoint, `python examples/hello_blocks/train.py --config
...`, runs on the head node and reads `/shared` like everything else. `clusterSelector` points
the job at the running cluster instead of creating one per job. This is what a scheduler or a CI
pipeline would use; the driver's `exec-head` exists so the same scenario runner works under
both drivers.

## 7. MinIO in the cluster: a cold restore

With compose, MinIO is a profile; here `deploy/k8s/minio.yaml` is a Deployment, a Service
(`minio:9000`, the endpoint the `harness-minio.yaml` configs use) and a PersistentVolumeClaim.
`just up-minio N` applies it next to the RayCluster:

```bash
just down && just up-minio 2
just mkbucket                                              # the distrainer bucket, created from the head pod
just blocks examples/hello_blocks/harness-minio.yaml       # log and blocks on s3://distrainer/blocks
deploy/driver.sh exec-head python examples/hello_blocks/train.py \
    --config examples/hello_blocks/harness-minio.yaml \
    --set checkpoint.num_to_keep=null                      # run hello_s3 on s3://distrainer/runs, every checkpoint kept
```

(`just train examples/hello_blocks/harness-minio.yaml` is the same run with the config's
`num_to_keep: 3`, which leaves only the last three checkpoints for the restore below; S9 keeps
them all for the same reason.)

Now lose everything except the bucket, as scenario S9 does:

```bash
just down                        # the RayCluster and the MinIO Deployment go; the PVC stays
deploy/driver.sh wipe-shared     # the shared directory too (refused while pods exist)
just up-minio 2                  # a new cluster, the same bucket
deploy/driver.sh endpoint        # minio=http://<ClusterIP>:9000 console=...:9001 (login distrainer / distrainer123)
```

Pick a mid-run checkpoint of `hello_s3` in the MinIO console (or
`kubectl -n distrainer port-forward svc/minio 9001` if the ClusterIP is not routed on your
machine) and resume it into a new run, inside the head pod, with the CLI from `docs/cli.md`:

```bash
deploy/driver.sh exec-head distrainer resume s3://distrainer/runs/hello_s3/checkpoint_g000004_p000024_n02_a00 \
    --config examples/hello_blocks/harness-minio.yaml \
    --entry examples.hello_blocks.train:entry --run-name hello_s3_resume
```

The resumed run starts at the checkpoint's segment and cursor and finishes the log; its audit
trail on the bucket starts exactly where the checkpoint says. `just integration S9` runs this
sequence unattended and checks it, and `S10` kills the head pod mid-run instead (the operator
recreates it, the workers restart into the new head, the run resumes from the newest checkpoint
on the bucket).

## 8. Tear down

```bash
just down                            # RayCluster, MinIO Deployment and Services; the MinIO PVC stays
just nuke                            # also the PVC and the shared directory
orb config set k8s.enable false      # Kubernetes off; the compose harness does not need it
```

## 9. When something is off

- **Pods stay `0/1 Running` forever while `ray status` looks fine.** The readiness probe on the
  dashboard agent (port 52365) fails because the dashboard runs in "minimal" mode
  (`dashboard.log` in the pod says "http server disabled"): the image lacks `ray[default]`.
  `pyproject.toml` pins `ray[data,default,train]`; if you changed it, relock and `just build`.
- **A trainer takes 40 s to reach "Started training worker group".** Look for two c10d warnings
  "The hostname of the client socket cannot be retrieved" about 15 s apart: reverse DNS for the
  worker pods is being forwarded outside the cluster and timing out. The headless
  `distrainer-workers` Service in the manifest gives CoreDNS local PTR records; check it exists
  (`kubectl -n distrainer get svc`). S11's pacing check is the scenario that notices this.
- **`just up` says the CRDs are missing.** Run `just kuberay-operator` once per Kubernetes
  cluster. **Nothing answers on the API at all**: `orb config set k8s.enable true` needs the
  OrbStack restart, and `kubectl config current-context` must be `orbstack`.
- **The removed pod trains on after `just scale 2`.** That is the grace period; see section 5.
- **Out of memory.** The VM has 8 GB: the head takes about 1.4 GB and each worker about 0.6 GB
  (measured in M3, `docs/handoff-m5.md`), plus Kubernetes itself. One cluster at a time, compose
  containers down, and stop stray containers.

## 10. Where to go next

- `docs/running-modes.md` (mode D): the same material as a reference, next to the laptop,
  compose and uncloud modes.
- `docs/examples-and-scenarios.md`, "Under KubeRay": what every driver verb does on each side.
- `docs/handoff-m6.md`: what building this taught, and the plan for uncloud (mode C).
