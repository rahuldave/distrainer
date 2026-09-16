# uncloud cheat sheet (0.20)

[uncloud](https://uncloud.run): Docker hosts joined by a WireGuard mesh with cluster DNS,
driven by a Compose file, no control plane. `uc` on the Mac talks to any machine over ssh.
Nothing here is specific to distrainer; it is what building a driver on it taught.

## Install and the daemon

```bash
brew install psviderski/tap/uncloud && uc version
```

- On each machine `uc machine init|add` installs Docker (get.docker.com, unpinned) and
  `uncloudd` as the systemd unit `uncloud.service`, plus `/usr/local/bin/uncloud-uninstall`.
  The remote user needs passwordless sudo. Every machine also runs an `uncloud-corrosion`
  container, the cluster store; leave it alone.
- `uc` runs the **system `ssh`**, so `~/.ssh/config`, ProxyCommands and `~/.ssh/known_hosts`
  apply: an unknown host key fails non-interactively (accept it once with
  `ssh -o StrictHostKeyChecking=accept-new user@host true`).

## Machines and contexts

```bash
uc machine init USER@HOST -c CTX -n NAME --network 10.210.0.0/16 --no-caddy --no-dns \
    --public-ip none --wg-endpoint IP:51820 -i KEY -y      # the first machine creates the cluster and the context
uc machine add USER@HOST -n NAME --no-caddy --public-ip none --wg-endpoint IP:51820 -i KEY -y
uc machine ls                        # NAME STATE ADDRESS PUBLIC-IP WIREGUARD-ENDPOINTS OS ... DOCKER VERSION
uc machine rm NAME -y                # RESETS the machine (uninstalls uncloud and its containers); --no-reset keeps them
uc ctx ls / uc ctx use CTX           # there is NO `uc ctx rm`: edit ~/.config/uncloud/config.yaml
UNCLOUD_CONTEXT=CTX uc ...           # or --connect ssh://user@host to bypass the config file
```

- `~/.config/uncloud/config.yaml`: `current_context`, and `contexts.<name>.connections`, a list
  of `{ssh: "user@host", ssh_key_file: ..., machine_id: ...}`. A machine whose address changed
  (a cloud instance stopped and started) needs its `ssh` entry rewritten by hand.
- Give the WireGuard endpoint explicitly (a private address is fine when every peer reaches
  it) and disable ingress (`--public-ip none`) on a test bed; auto-detection is for machines
  with a public address. `--no-caddy --no-dns` skips the reverse proxy and the public DNS
  registration under `*.uncloud.run`.
- **Membership settles slowly**: machines show `Suspect` or `Down` for minutes after a join,
  a leave or a restart, and the first ping to a new peer is lost during the handshake. Wait
  for `Up` before deploying. A machine can also flap to `Down` for under a minute when short
  of memory; a deploy in that window fails with `start container: not found`, a retry heals.
- `wg` is not installed on the machines: `apt install wireguard-tools`, then `wg show uncloud`
  (peers, endpoints, last handshake, bytes).

## Deploy and operate

```bash
uc deploy -f compose.yml -y [SERVICE...]     # declarative: creates, recreates stopped, scales; one container at a time, each monitored 5 s
uc ls                                        # services (an ID column appears only when a name is duplicated)
uc ps                                        # containers: SERVICE ID IMAGE CREATED STATUS IP MACHINE (CREATED and STATUS contain spaces)
uc scale SERVICE N -y                        # a removed replica gets a 10 s graceful stop
uc rm SERVICE|ID                             # one per call; several in one call once removed nothing
uc exec -T SERVICE -- cmd                    # a random replica unless --container ID; chatter on stderr, stdout clean
uc logs -n 100 SERVICE
uc inspect SERVICE
uc image push IMAGE                          # from the local Docker to every machine over ssh; no registry needed
uc volume ls / uc volume rm -y NAME          # volume rm prompts and ignores UNCLOUD_AUTO_CONFIRM
UNCLOUD_AUTO_CONFIRM=true                    # deploy and scale honour it (or -y)
```

- **No `uc cp`** (stream a tar through `uc exec`), **no per-container kill, stop or start**
  (ssh to the machine `uc ps` names and use `docker kill|stop|start ID`).
- **Service names are cluster-global and can be registered twice** (seen after a removal that
  failed mid-way). Then every name-taking command refuses ("multiple services found ... use
  the service ID") and `uc ls` grows an ID column: remove by id, one at a time.
- `uc deploy` recreates a container it finds stopped (new id, same address).

## Compose file extensions and limits

```yaml
services:
  web:
    image: myimage:local
    pull_policy: never                    # use the pushed image, pull nothing
    x-machines: ["m1"]                    # placement: a list, or a comma-separated string "m2,m3" (NOT space-separated)
    x-ports:
      - "10.0.0.0/8:8080:80/tcp@host"     # host-mode publish, bound to the machine's addresses inside the prefix
    deploy:
      replicas: 2                         # spread by count over the allowed machines, not by load
    depends_on: [db]                      # deploy order only
```

- `ports:` (non-host) and `restart:` are unsupported; every service restarts `unless-stopped`
  (harmless: `docker kill` never triggers a restart policy). `profiles` untested; deploy named
  services instead. `build:` works but push-then-`pull_policy: never` is simpler.
- Replicas spread by count: pin heavy services to their machine with `x-machines`, or one
  replica lands next to them and starves them.

## Networking

- Per machine: the WireGuard interface `uncloud` (MTU 1420; the `ADDRESS` column of `uc machine
  ls` is the machine's address on it) and a Docker bridge network `uncloud`, one `/24` per
  machine out of the cluster CIDR; containers get MTU 1420.
- DNS: every container gets `search internal.`, so a bare service name resolves cluster-wide
  to all replicas of that service, the same as `<service>.internal`. Machine names are not
  service names: do not name a machine after a service. Reverse lookups of overlay addresses
  answer at once. Probes for names like `metadata.google.internal` are logged at DEBUG as
  "Failed to resolve service name"; harmless.
- The underlay must not overlap the cluster CIDR (default `10.210.0.0/16`); `--network`
  changes it at `init`.
- Between machines only UDP 51820 (WireGuard) is needed; everything else rides the tunnel.
