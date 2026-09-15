#!/usr/bin/env bash
# OrbStack test bed for the uncloud driver (docs/running-modes.md C): two or three small Linux
# machines, each a Docker host running the uncloud daemon, joined into one uncloud cluster whose
# WireGuard mesh rides on OrbStack's machine network. Run once per Mac (`just uncloud-machines`),
# tear down with `destroy`. Needs OrbStack, the `uc` CLI (brew install psviderski/tap/uncloud) and
# ssh; `uc` talks to the machines through OrbStack's `NAME@orb` SSH route (system ssh, so the
# ProxyCommand in ~/.orbstack/ssh/config is honoured) and installs Docker and uncloudd itself.
#   up        create the machines that do not exist yet, init the cluster on the first, add the rest
#   status    the machines as OrbStack and uncloud see them
#   stop      park the machines (`orb stop` of these machines only: a bare `orb stop` stops all of
#             OrbStack, Docker included); the cluster state survives
#   start     bring parked machines back and wait until uncloud shows every one Up
#   destroy   remove the machines from the cluster, delete them, drop the uc context
# The AWS twin is aws.sh (the same verbs on EC2 instances); common.sh holds what they share.
# Settings (environment; the defaults are cgroup caps, not reservations, and suit the 8 GB
# OrbStack VM of a 16 GB Mac: about 5 GB is in use with three workers training):
#   DISTRAINER_UNCLOUD_MACHINES        "uc1 uc2 uc3"; the first is the head machine (head + MinIO)
#   DISTRAINER_UNCLOUD_CONTEXT         distrainer (the uc context; every driver call names it)
#   DISTRAINER_UNCLOUD_HEAD_MEMORY     5G      DISTRAINER_UNCLOUD_WORKER_MEMORY  2G
#   DISTRAINER_UNCLOUD_CPUS            2       DISTRAINER_UNCLOUD_DISTRO         ubuntu:noble
#   DISTRAINER_UNCLOUD_NETWORK         10.210.0.0/16 (uncloud's machine and container subnet)
#   DISTRAINER_UNCLOUD_SSH_KEY         ~/.orbstack/ssh/id_ed25519 (the key `uc` logs in with)
# `destroy` deletes only machines this script created (recorded in .harness/uncloud/machines);
# a machine that existed before `up` adopted it is removed from the cluster but left in place.
# The network overlay, explicitly: each machine's address on OrbStack's machine network
# (`orb config get network.subnet4`, 192.168.138.0/23 by default) is passed as its WireGuard
# endpoint instead of trusting auto-detection, ingress is disabled (`--public-ip none`), and the
# uncloud subnet is checked against OrbStack's so the two networks cannot overlap.
set -euo pipefail
root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
machines_str="${DISTRAINER_UNCLOUD_MACHINES:-uc1 uc2 uc3}"
read -r -a machines <<< "$machines_str"
ctx="${DISTRAINER_UNCLOUD_CONTEXT:-distrainer}"
# The limits are cgroup caps, not reservations. The head machine runs the Ray head (GCS, dashboard,
# autoscaler, Train controller), the driver and MinIO, about 2 GB of process memory plus the page
# cache of the image: under a 3G cap it thrashed (millions of cap hits, VM load above 200, ssh and
# GCS keepalives failing, steps stalling for tens of seconds), and a worker placed next to it made
# world size 3 five times slower, which is why the compose file keeps workers off the head machine.
# A worker machine needs about 1G per Ray worker; two fit in 2G. Raise a limit later with
# `orb config set machine.NAME.memory_mib`.
head_mem="${DISTRAINER_UNCLOUD_HEAD_MEMORY:-5G}"
worker_mem="${DISTRAINER_UNCLOUD_WORKER_MEMORY:-2G}"
cpus="${DISTRAINER_UNCLOUD_CPUS:-2}"
distro="${DISTRAINER_UNCLOUD_DISTRO:-ubuntu:noble}"
network="${DISTRAINER_UNCLOUD_NETWORK:-10.210.0.0/16}"
ssh_key="${DISTRAINER_UNCLOUD_SSH_KEY:-$HOME/.orbstack/ssh/id_ed25519}"
wg_port=51820
owned="$root/.harness/uncloud/machines"   # one name per line: machines `up` created (destroy deletes only these)
export UNCLOUD_CONTEXT="$ctx" UNCLOUD_AUTO_CONFIRM=true
. "$(dirname "${BASH_SOURCE[0]}")/common.sh"   # need, ctx_exists, cluster_has, check_overlap, ctx_forget, wait_up

machine_ip() {   # the machine's address on OrbStack's machine network (eth0)
  orb -m "$1" ip -4 -o addr show dev eth0 | awk '{split($4, a, "/"); print a[1]; exit}'
}
orb_has() { orb list -q 2>/dev/null | grep -qx "$1"; }
owned_has() { [ -f "$owned" ] && grep -qx "$1" "$owned"; }
orb_overlap() {   # the uncloud subnet must not overlap OrbStack's machine network
  local orb_net
  orb_net="$(orb config get network.subnet4 2>/dev/null || true)"
  [ -z "$orb_net" ] && return 0
  check_overlap "$network" "$orb_net" "OrbStack's machine network"
}

verb="${1:-}"; shift || true
case "$verb" in
  up)
    need orb "install OrbStack"; need uc "brew install psviderski/tap/uncloud"; need python3 "needed for the subnet check"
    orb_overlap
    i=0
    for m in "${machines[@]}"; do
      mem="$worker_mem"; [ "$i" = 0 ] && mem="$head_mem"
      if orb_has "$m"; then
        echo "machine $m exists (not created here: destroy will leave it in place)"
        have="$(orb config get "machine.$m.memory_mib" 2>/dev/null || true)"
        want="$(python3 -c "import re,sys; v=sys.argv[1].upper(); n=float(re.sub('[A-Z]', '', v)); print(int(n * 1024) if v.endswith('G') else int(n))" "$mem")"
        if [ -n "$have" ] && [ "$have" -lt "$want" ]; then
          echo "  its memory cap is ${have} MiB, below the ${want} MiB default: orb config set machine.$m.memory_mib $want" >&2
        fi
      else
        echo "creating $m ($distro, $mem, $cpus cpus)"
        orb create --memory "$mem" --cpus "$cpus" "$distro" "$m"
        mkdir -p "$(dirname "$owned")" && echo "$m" >> "$owned"
      fi
      ip="$(machine_ip "$m" 2>/dev/null || true)"
      [ -n "$ip" ] || { echo "no eth0 address on $m (is it running? orb list)" >&2; exit 1; }
      if cluster_has "$m"; then
        echo "$m is in cluster context $ctx"
      elif ! ctx_exists; then
        echo "initialising cluster context $ctx on $m (WireGuard endpoint $ip:$wg_port)"
        uc machine init "$m@orb" -c "$ctx" -n "$m" --network "$network" --no-caddy --no-dns \
          --public-ip none --wg-endpoint "$ip:$wg_port" -i "$ssh_key" -y
      else
        echo "adding $m to cluster context $ctx (WireGuard endpoint $ip:$wg_port)"
        uc machine add "$m@orb" -n "$m" --no-caddy --public-ip none --wg-endpoint "$ip:$wg_port" -i "$ssh_key" -y
      fi
      i=$((i + 1))
    done
    wait_up 300
    uc machine ls ;;
  status)
    orb list 2>/dev/null || true
    if ctx_exists; then uc machine ls; else echo "no uc context '$ctx'"; fi ;;
  stop)
    orb stop "${machines[@]}"
    orb list ;;
  start)
    orb start "${machines[@]}"
    if ctx_exists; then wait_up 300; uc machine ls; else orb list; fi ;;
  destroy)
    if ctx_exists; then
      for m in "${machines[@]}"; do
        if cluster_has "$m"; then uc machine rm "$m" -y || echo "uc machine rm $m failed (continuing)" >&2; fi
      done
    fi
    for m in "${machines[@]}"; do
      if ! orb_has "$m"; then continue; fi
      if owned_has "$m"; then
        orb delete -f "$m" && { grep -vx "$m" "$owned" > "$owned.tmp" || true; } && mv "$owned.tmp" "$owned"
      else
        echo "$m was not created by this script (not in $owned); left in place" >&2
      fi
    done
    # the context goes only when no machine is left in it (a partial destroy keeps the cluster)
    if ctx_exists && [ -z "$(uc machine ls 2>/dev/null | awk 'NR > 1 {print $1}')" ]; then ctx_forget; fi ;;
  *)
    echo "usage: $0 up | status | stop | start | destroy" >&2; exit 2 ;;
esac
