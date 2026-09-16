#!/usr/bin/env bash
# Shared by the uncloud bootstrap scripts (machines.sh for OrbStack machines, aws.sh for EC2):
# the uc context bookkeeping and the wait for a settled membership. Sourced, not run; the caller
# sets `root` (the repository) and `ctx` (the uc context) and exports UNCLOUD_CONTEXT.
need() { command -v "$1" >/dev/null 2>&1 || { echo "$1 not found: $2" >&2; exit 2; }; }
ctx_exists() { uc ctx ls 2>/dev/null | awk '{print $1}' | grep -qx "$ctx"; }
cluster_has() {   # the machine is a member of the uc context (by name); an unreachable cluster fails
  local all
  ctx_exists || return 1
  all="$(uc machine ls 2>/dev/null)" || { echo "uc machine ls failed for context $ctx" >&2; exit 1; }
  awk 'NR > 1 {print $1}' <<< "$all" | grep -qx "$1"
}
check_overlap() {   # check_overlap NET OTHER LABEL: the uncloud subnet must not overlap the underlay
  python3 - "$1" "$2" "$3" <<'PY'
import ipaddress, sys
a, b = (ipaddress.ip_network(x, strict=False) for x in sys.argv[1:3])
if a.overlaps(b):
    sys.exit(f"uncloud network {a} overlaps {sys.argv[3]} {b}; set DISTRAINER_UNCLOUD_NETWORK")
print(f"uncloud network {a}, {sys.argv[3]} {b}: no overlap")
PY
}
uc_config_rewrite() {   # uc_config_rewrite forget|reconnect [OLD=NEW ...]: edit the uc CLI config for
  # this context (uc has no `ctx rm` and no way to change a connection): drop the context, or
  # replace connection ssh destinations (a machine whose address changed). The file is rewritten
  # atomically next to a .bak; the repo's python has yaml; a failure only leaves the entry
  local cfg="${UNCLOUD_CONFIG:-$HOME/.config/uncloud/config.yaml}"
  [ -f "$cfg" ] || return 0
  need uv "the uc config rewrite runs the repo's python"
  uv run --project "$root" python - "$cfg" "$ctx" "$@" <<'PY' || echo "uc config not updated: edit $cfg by hand" >&2
import os
import shutil
import sys

import yaml

path, ctx, op, *pairs = sys.argv[1:]
with open(path) as f:
    cfg = yaml.safe_load(f) or {}
contexts = cfg.get("contexts") or {}
changed = False
if op == "forget" and ctx in contexts:
    del contexts[ctx]
    if cfg.get("current_context") == ctx:
        cfg.pop("current_context", None)
        if contexts:
            cfg["current_context"] = next(iter(contexts))
    changed = True
    what = f"removed uc context {ctx}"
elif op == "reconnect" and ctx in contexts:
    mapping = dict(p.split("=", 1) for p in pairs)
    for conn in contexts[ctx].get("connections") or []:
        new = mapping.pop(str(conn.get("ssh")), None)
        if new and new != conn["ssh"]:
            conn["ssh"], changed = new, True
    what = f"updated the connections of uc context {ctx}"
    if mapping:  # a connection the context does not store as user@host: uc would keep dialling the old address
        print(f"warning: no connection of uc context {ctx} matched {sorted(mapping)}; "
              f"check the connections in {path}", file=sys.stderr)
if changed:
    text = yaml.safe_dump(cfg, sort_keys=False)  # serialise first: nothing is truncated on error
    shutil.copy2(path, path + ".bak")
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        f.write(text)
    os.replace(tmp, path)
    print(f"{what} in {path} (backup: {path}.bak)")
PY
}
ctx_forget() { uc_config_rewrite forget; }
wait_up() {   # wait_up SECONDS: until every cluster machine reports Up (membership shows Suspect for
  # minutes after a join or a leave; a deploy in that window can fail on a machine shown as Down)
  local deadline=$((SECONDS + $1)) states
  while :; do
    states="$(uc machine ls 2>/dev/null | awk 'NR > 1 {print $2}' | sort -u | tr '\n' ' ')"
    [ "$states" = "Up " ] && return 0
    if [ "$SECONDS" -ge "$deadline" ]; then echo "machines not all Up after $1s: $states" >&2; uc machine ls >&2; return 1; fi
    sleep 3
  done
}
