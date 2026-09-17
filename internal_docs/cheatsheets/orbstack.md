# OrbStack cheat sheet

OrbStack on Apple Silicon: one Linux VM that hosts a Docker engine, optional Kubernetes (k3s)
and any number of Linux "machines" (lightweight VMs sharing the kernel). Nothing here is
specific to distrainer; it is what four milestones of running clusters on it taught.

## The VM

```bash
orb status                          # Running | Stopped
orb start / orb stop / orb restart  # a BARE `orb stop` stops all of OrbStack: Docker, Kubernetes, every machine
orb config show                     # every setting
orb config set memory_mib 8192      # VM memory (the machines, Docker and k8s share it); cpu likewise
orb config get network.subnet4      # the machine network, 192.168.138.0/23 by default
orb config get rosetta              # true: x86_64 containers and machines run under Rosetta
orb config set k8s.enable true      # Kubernetes; needs an OrbStack restart to apply
```

- Docker starts lazily: the first `docker` command wakes the VM. Quitting OrbStack from the menu
  bar removes the socket (`~/.orbstack/run/docker.sock`) and every `docker` call fails with
  "no such file or directory" until it is started again. `docker context show` says `orbstack`.
- Memory is one pool: `/proc/loadavg` and `free` inside a machine are VM-wide. A tight VM shows
  up as stalls everywhere at once (containers, machines, ssh sessions dropping).
- Kubernetes keeps running whatever you deployed there (an operator, pods) across restarts and
  eats memory quietly: `kubectl get pods -A` (context `orbstack`) before blaming anything else.
  Stale containers with restart policies (a `dagger-engine`, an old registry) also autostart.

## Docker

```bash
docker context show                                   # orbstack
docker run --rm --platform linux/amd64 alpine uname -m   # x86_64: Rosetta, in the same VM
docker build --platform linux/amd64 -t image .        # an x86 image; buildx lists arm64, amd64, and more
docker kill ID      # SIGKILL: models a node dying; restart policies do not fire
docker stop ID      # SIGTERM then SIGKILL after the grace: models a preemption notice
```

- Locally built images are visible to OrbStack's Kubernetes without a registry
  (`imagePullPolicy: IfNotPresent`).
- Container and pod IPs are routable from the Mac. Compose containers also answer at
  `service.project.orb.local`. Published ports appear on `localhost` and are forwarded to the
  LAN as well unless bound to a specific address or prefix.
- Bind mounts of Mac paths are fine for source trees; databases and object stores prefer named
  volumes (speed, file locking).

## Linux machines

```bash
orb create --memory 5G --cpus 2 ubuntu:noble NAME     # arm64 by default; -a amd64 for an x86 machine (fixed at creation)
orb list                                              # name, state, distro, arch, size
orb start NAME / orb stop NAME                        # always name the machines (see the bare `orb stop` above)
orb delete -f NAME
orb -m NAME <command>                                 # run a command inside (as the default user; sudo works)
orb -m NAME -u root <command>
ssh NAME@orb                                          # the ssh route: ProxyCommand in ~/.orbstack/ssh/config, key ~/.orbstack/ssh/id_ed25519,
                                                      # no sshd inside; tools that shell out to ssh (uncloud) reach machines this way
orb config get machine.NAME.memory_mib                # per-machine caps; `set` takes effect at once
orb config set machine.NAME.memory_mib 5120
```

- Address: `orb -m NAME ip -4 -o addr show dev eth0` (on the machine network); also `NAME.orb.local`.
- Files: the Mac is mounted at `/mnt/mac` inside a machine; a machine's filesystem is under
  `~/OrbStack/NAME/` on the Mac.
- **Memory caps are cgroup limits, not reservations, and they count page cache.** The thrash
  signature: millions of `max` hits in `/sys/fs/cgroup/memory.events`, thousands of
  `workingset_refault_file` per 10 s in `/sys/fs/cgroup/memory.stat`, load above 100, ssh
  sessions into the machine dropping, everything on it stalling for tens of seconds while
  short samples look nominal. Raise the cap live; a Ray head with a dashboard, a controller and
  an object store next to it wants about 5 GB.
- Machines are cheap to park and resume (`orb stop NAME`, `orb start NAME`): the installed
  daemons, Docker images and volumes survive. Deleting and recreating means reinstalling.
- A machine's ports are forwarded to the LAN like a container's unless the service binds to
  the machine's own address or a prefix (`orb config get network.subnet4`).

## Kubernetes (k3s)

```bash
kubectl config use-context orbstack
kubectl get pods -A                                   # what is still running
kubectl -n NS delete deployment X                     # stop an operator you no longer use
kubectl apply --server-side -k <kustomize dir>        # installing an operator without Helm
kubectl port-forward svc/NAME 8265                    # when a service IP is not routable
```

- `hostPath` volumes reach Mac paths. Pod and service IPs are routable from the Mac.
- Reverse DNS for pod IPs is forwarded outside the cluster unless a headless Service covers
  the pods; a slow upstream once cost 15 s per lookup at process-group setup.
