#!/usr/bin/env bash
# AWS test bed for the uncloud driver: the twin of machines.sh on EC2 (docs/running-modes.md C on
# real machines, tutorial 5: docs/tutorials/aws.md). Two or three Graviton instances (arm64, so the image built
# on an arm64 Mac is pushed as it is), one key pair, one security group (ssh and the published
# ports from this Mac's address only, WireGuard between the members only), joined into one uncloud
# cluster over their private addresses; an S3 bucket with an IAM user scoped to it as the store
# instead of MinIO. Needs the AWS CLI authenticated for EC2 (`bucket`: S3 and IAM too), the `uc`
# CLI, ssh and uv. State lives under .harness/aws/ (gitignored): ssh_config (a Host per machine),
# known_hosts, instances, vpc, env (what the driver reads through DISTRAINER_ENV_FILE),
# s3-credentials (mode 600). The env file is read first here, then .env, then the caller's
# environment: a setting changed in .env or on the command line wins over what was recorded.
#   up          create what is missing (key pair, security group, instances), start what is
#               stopped, write the ssh config, init the cluster on the head, add the workers
#   status      the instances as AWS and uncloud see them, and the address ssh is admitted from
#   stop        park the instances (only the disks are billed; the public addresses are released)
#   start       bring parked instances back: new public addresses, so the ssh config, the known
#               hosts and the uc context's connections are rewritten
#   destroy     terminate the instances, delete the security group and the key pair, drop the uc
#               context (the bucket and the IAM user stay: bucket-rm)
#   bucket      create the bucket (private) and the IAM user with a policy scoped to it, one access
#               key kept under .harness/aws/; write the env file
#   bucket-rm   empty and delete the bucket; delete the user's keys, policy and the user
#   env         print the env file
# Settings (environment or .env):
#   DISTRAINER_AWS_PROFILE       the CLI's default profile     DISTRAINER_AWS_REGION       us-east-1
#   DISTRAINER_UNCLOUD_MACHINES  "aws1 aws2 aws3" (the first is the head machine; not `head`, a service name)
#   DISTRAINER_UNCLOUD_CONTEXT   distrainer-aws                DISTRAINER_UNCLOUD_NETWORK  10.210.0.0/16
#   DISTRAINER_AWS_HEAD_TYPE     t4g.large (2 vCPUs, 8 GB)     DISTRAINER_AWS_WORKER_TYPE  t4g.medium (4 GB)
#   DISTRAINER_AWS_ARCH          arm64 (the image's)           DISTRAINER_AWS_DISK_GB      16
#   DISTRAINER_AWS_NAME          distrainer (key pair, security group, IAM user, Name tags)
#   DISTRAINER_AWS_KEY           ~/.ssh/distrainer-aws.pem     DISTRAINER_AWS_USER         ubuntu
#   DISTRAINER_AWS_ALLOW_CIDR    this Mac's public address/32 (asked of checkip.amazonaws.com; /24 at the widest)
#   DISTRAINER_AWS_BUCKET        the bucket of examples/hello_blocks/harness-s3.yaml (store_root)
# Cost (us-east-1 on-demand, 2026): about 0.15 USD per hour for the three instances with their
# public addresses while they run, 0.13 USD per day for their disks while stopped; the bucket
# is cents. The t4g types are burstable and launched with unlimited CPU credits so a busy hour
# never throttles the step pacing the scenarios measure; a fully busy vCPU beyond the baseline
# (30% on t4g.large, 20% on t4g.medium) adds 0.04 USD per hour. `stop` when done for the day,
# `destroy` when done for good.
# The network, explicitly: the instances sit in one default subnet of the default VPC (the first
# whose zone offers both instance types, recorded so a later `up` keeps the same zone: no
# cross-zone traffic), WireGuard peers over the private addresses
# (stable across stop and start; UDP 51820 admitted from the group itself only), ingress off
# (`--public-ip none`), the uncloud subnet checked against the VPC's. The Mac reaches the
# machines over their public addresses by ssh only (admitted from DISTRAINER_AWS_ALLOW_CIDR;
# the Ray dashboard and MinIO, if ever deployed here, are reached through an ssh tunnel);
# `uc` runs the system ssh, so those addresses are accepted into ~/.ssh/known_hosts (stale
# entries for a reused address are dropped first).
set -euo pipefail
root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
state="$root/.harness/aws"
caller_env="$(export -p)"
if [ -f "$state/env" ]; then set -a; . "$state/env"; set +a; fi   # what a previous run settled on (lowest precedence)
if [ -f "$root/.env" ]; then set -a; . "$root/.env"; set +a; fi
eval "$caller_env"
profile="${DISTRAINER_AWS_PROFILE:-}"
region="${DISTRAINER_AWS_REGION:-us-east-1}"
read -r -a machines <<< "${DISTRAINER_UNCLOUD_MACHINES:-aws1 aws2 aws3}"
ctx="${DISTRAINER_UNCLOUD_CONTEXT:-distrainer-aws}"
network="${DISTRAINER_UNCLOUD_NETWORK:-10.210.0.0/16}"
head_type="${DISTRAINER_AWS_HEAD_TYPE:-t4g.large}"
worker_type="${DISTRAINER_AWS_WORKER_TYPE:-t4g.medium}"
arch="${DISTRAINER_AWS_ARCH:-arm64}"
disk_gb="${DISTRAINER_AWS_DISK_GB:-16}"
name="${DISTRAINER_AWS_NAME:-distrainer}"
key="${DISTRAINER_AWS_KEY:-$HOME/.ssh/distrainer-aws.pem}"
user="${DISTRAINER_AWS_USER:-ubuntu}"
wg_port=51820
export UNCLOUD_CONTEXT="$ctx" UNCLOUD_AUTO_CONFIRM=true
. "$(dirname "${BASH_SOURCE[0]}")/common.sh"   # need, ctx_exists, cluster_has, check_overlap, uc_config_rewrite, ctx_forget, wait_up

die() { echo "$*" >&2; exit 1; }
awsc() {   # the CLI with the profile and region of this bed, text output, no pager. Not `command aws`:
  # under macOS bash 3.2 a failing command run through `command` ignores set -e's suppression in
  # `if` conditions and `||` lists and exits the script with no message (the plain call is fine)
  # shellcheck disable=SC2086
  aws ${profile:+--profile "$profile"} --region "$region" --output text --no-cli-pager "$@"
}
my_cidr() {
  local ip
  ip="$(curl -fsS --max-time 10 https://checkip.amazonaws.com | tr -d '[:space:]')" || return 1
  case "$ip" in
    *[!0-9.]*|"") return 1 ;;   # not a dotted quad (a captive portal's HTML, an empty answer)
  esac
  echo "$ip/32"
}
allowed_cidr() {   # DISTRAINER_AWS_ALLOW_CIDR, else this Mac's public address; never the whole internet
  local cidr="${DISTRAINER_AWS_ALLOW_CIDR:-}"
  if [ -z "$cidr" ]; then
    cidr="$(my_cidr)" || die "cannot learn this Mac's public address (checkip.amazonaws.com): set DISTRAINER_AWS_ALLOW_CIDR"
  fi
  case "$cidr" in
    */*) ;;
    *) die "DISTRAINER_AWS_ALLOW_CIDR=$cidr is not a CIDR (address/32)" ;;
  esac
  case "${cidr%/*}" in
    *[!0-9.]*|"") die "DISTRAINER_AWS_ALLOW_CIDR=$cidr is not an IPv4 address/prefix" ;;
  esac
  case "${cidr#*/}" in
    2[4-9]|3[0-2]) ;;
    *) die "DISTRAINER_AWS_ALLOW_CIDR=$cidr admits too much (ssh to the machines): /24 at the widest, your address/32 is the norm" ;;
  esac
  echo "$cidr"
}
bucket_owner() {   # bucket_owner BUCKET -> the distrainer:cluster tag, empty when none (NoSuchTagSet) or unreadable;
  # never a failure: a failing command substitution in an assignment would end the script under set -e
  local out
  out="$(awsc s3api get-bucket-tagging --bucket "$1" --query 'TagSet[?Key==`"distrainer:cluster"`].Value' 2>/dev/null)" || out=""
  printf '%s' "$out" | tr -d '[:space:]'
}
bucket_name() {   # DISTRAINER_AWS_BUCKET, else the bucket harness-s3.yaml stores in
  if [ -n "${DISTRAINER_AWS_BUCKET:-}" ]; then echo "$DISTRAINER_AWS_BUCKET"; return; fi
  awk '$1 == "store_root:" {split($2, a, "/"); print a[1]; exit}' "$root/examples/hello_blocks/harness-s3.yaml"
}
instances() {   # "machine id state private public" for this context's instances that are not terminated
  awsc ec2 describe-instances --filters "Name=tag:distrainer:cluster,Values=$ctx" \
    "Name=instance-state-name,Values=pending,running,stopping,stopped" \
    --query 'Reservations[].Instances[].[Tags[?Key==`"distrainer:machine"`]|[0].Value, InstanceId, State.Name, PrivateIpAddress, PublicIpAddress]' \
    | tr '\t' ' '
}
instance_ids() { awk '{print $2}' | tr '\n' ' ' | sed 's/ $//'; }   # of the instances() lines on stdin
vpc_info() {   # "vpc cidr subnet": the default VPC and its first default subnet in a zone that offers
  # both instance types (a zone may lack Graviton capacity: us-east-1a of one account had none)
  local vpc cidr subnet out subnets zones z id want
  out="$(awsc ec2 describe-vpcs --filters Name=is-default,Values=true --query 'Vpcs[0].[VpcId,CidrBlock]' | tr '\t' ' ')"
  read -r vpc cidr <<< "$out"
  [ -n "$vpc" ] && [ "$vpc" != "None" ] || die "no default VPC in $region (create one: aws ec2 create-default-vpc)"
  subnets="$(awsc ec2 describe-subnets --filters "Name=vpc-id,Values=$vpc" Name=default-for-az,Values=true \
    --query 'sort_by(Subnets, &AvailabilityZone)[].[AvailabilityZone, SubnetId]' | tr '\t' ' ')"
  [ -n "$subnets" ] || die "no default subnet in $vpc"
  zones="$(awsc ec2 describe-instance-type-offerings --location-type availability-zone \
    --filters "Name=instance-type,Values=$head_type,$worker_type" --query 'InstanceTypeOfferings[].[Location, InstanceType]' | tr '\t' ' ')"
  [ -n "$zones" ] || die "no zone of $region offers $head_type or $worker_type (or ec2:DescribeInstanceTypeOfferings is denied)"
  want=2; { [ "$head_type" = "$worker_type" ] || [ "${#machines[@]}" -le 1 ]; } && want=1
  subnet=""
  while read -r z id; do
    [ -n "$z" ] || continue
    if [ "$(awk -v z="$z" '$1 == z {print $2}' <<< "$zones" | sort -u | wc -l | tr -d ' ')" -ge "$want" ]; then
      subnet="$id"; echo "zone $z offers $head_type and $worker_type: subnet $id" >&2; break
    fi
  done <<< "$subnets"
  [ -n "$subnet" ] || die "no default subnet of $vpc is in a zone offering $head_type and $worker_type (aws ec2 describe-instance-type-offerings --location-type availability-zone)"
  echo "$vpc $cidr $subnet"
}
ensure_key_pair() {
  if awsc ec2 describe-key-pairs --key-names "$name" >/dev/null 2>&1; then
    [ -f "$key" ] || die "key pair $name exists in AWS but $key is missing: delete it (aws ec2 delete-key-pair --key-name $name) and run up again"
  else
    [ ! -e "$key" ] || die "$key exists but key pair $name is not in AWS: move the file away or set DISTRAINER_AWS_KEY"
    mkdir -p "$(dirname "$key")" "$state"
    (umask 077; awsc ec2 create-key-pair --key-name "$name" --key-type ed25519 --key-format pem --query KeyMaterial > "$key.tmp")
    mv "$key.tmp" "$key"; chmod 600 "$key"; touch "$state/key-created"
    echo "created key pair $name ($key)"
  fi
}
ensure_sg() {   # ensure_sg VPC -> the group id, created when missing
  local sg
  sg="$(awsc ec2 describe-security-groups --filters "Name=group-name,Values=$name-uncloud" "Name=vpc-id,Values=$1" --query 'SecurityGroups[0].GroupId')"
  if [ -z "$sg" ] || [ "$sg" = "None" ]; then
    sg="$(awsc ec2 create-security-group --group-name "$name-uncloud" --description "distrainer uncloud cluster $ctx" --vpc-id "$1" \
      --tag-specifications "ResourceType=security-group,Tags=[{Key=distrainer:cluster,Value=$ctx}]" --query GroupId)"
    echo "created security group $name-uncloud ($sg)" >&2
  fi
  echo "$sg"
}
allow() {   # allow SG PROTO FROM TO SOURCE DESCRIPTION: one ingress rule (a CIDR, or a group id); exists is fine
  local src out
  case "$5" in sg-*) src="UserIdGroupPairs=[{GroupId=$5,Description=$6}]" ;; *) src="IpRanges=[{CidrIp=$5,Description=$6}]" ;; esac
  if ! out="$(awsc ec2 authorize-security-group-ingress --group-id "$1" --ip-permissions "IpProtocol=$2,FromPort=$3,ToPort=$4,$src" 2>&1)"; then
    case "$out" in *InvalidPermission.Duplicate*) ;; *) echo "$out" >&2; return 1 ;; esac
  fi
}
mac_rules() {   # mac_rules SG: "rule-id cidr port" of the rules admitting this Mac
  awsc ec2 describe-security-group-rules --filters "Name=group-id,Values=$1" \
    --query 'SecurityGroupRules[?IsEgress==`false` && Description==`"from this Mac"`].[SecurityGroupRuleId, CidrIpv4, FromPort]' | tr '\t' ' '
}
ensure_rules() {   # ensure_rules SG ALLOW_CIDR: this Mac in first, then a Mac that moved networks (or a rule
  # for another port, from an earlier version) out; WireGuard between the members
  local stale
  allow "$1" tcp 22 22 "$2" "from this Mac"
  allow "$1" udp "$wg_port" "$wg_port" "$1" "WireGuard between the members"
  # nothing else: the Ray dashboard (8265) accepts job submissions from anyone who reaches it and
  # the Mac's address may be a shared NAT; reach it through the ssh route instead:
  #   ssh -F .harness/aws/ssh_config -L 8265:<head private address>:8265 aws1
  stale="$(mac_rules "$1" | awk -v c="$2" '$2 != c || $3 != 22 {print $1}' | tr '\n' ' ')"
  if [ -n "${stale// /}" ]; then
    # shellcheck disable=SC2086
    awsc ec2 revoke-security-group-ingress --group-id "$1" --security-group-rule-ids $stale >/dev/null && echo "revoked rules for an earlier address: $stale"
  fi
}
ami_info() {   # "ami root-device": Canonical's newest Ubuntu 24.04 image for the architecture
  local out
  out="$(awsc ec2 describe-images --owners 099720109477 \
    --filters "Name=name,Values=ubuntu/images/hvm-ssd*/ubuntu-noble-24.04-$arch-server-*" Name=state,Values=available \
    --query 'sort_by(Images, &CreationDate)[-1].[ImageId, RootDeviceName]' | tr '\t' ' ')"
  [ -n "$out" ] && [ "${out% *}" != "None" ] || die "no Ubuntu 24.04 $arch AMI found in $region"
  echo "$out"
}
launch() {   # launch MACHINE TYPE "AMI ROOT-DEVICE" SG SUBNET -> instance id
  local ami="${3% *}" dev="${3#* }"
  awsc ec2 run-instances --image-id "$ami" --instance-type "$2" --key-name "$name" --security-group-ids "$4" --subnet-id "$5" \
    --associate-public-ip-address --count 1 --credit-specification CpuCredits=unlimited \
    --block-device-mappings "[{\"DeviceName\":\"$dev\",\"Ebs\":{\"VolumeSize\":$disk_gb,\"VolumeType\":\"gp3\",\"DeleteOnTermination\":true}}]" \
    --metadata-options HttpTokens=required,HttpEndpoint=enabled \
    --tag-specifications "ResourceType=instance,Tags=[{Key=Name,Value=$name-$1},{Key=distrainer:cluster,Value=$ctx},{Key=distrainer:machine,Value=$1}]" \
      "ResourceType=volume,Tags=[{Key=Name,Value=$name-$1},{Key=distrainer:cluster,Value=$ctx}]" \
    --query 'Instances[0].InstanceId'
}
refresh() {   # .harness/aws/instances and ssh_config from the current addresses; known_hosts starts over
  local rows m line id st priv pub
  mkdir -p "$state"
  rows="$(instances)"
  : > "$state/instances.tmp"
  echo "# a Host per machine of uncloud context $ctx, for the driver (DISTRAINER_UNCLOUD_SSH=%s, -F this file) and for you: ssh -F $state/ssh_config ${machines[0]}" > "$state/ssh_config.tmp"
  for m in "${machines[@]}"; do
    line="$(awk -v m="$m" '$1 == m {print; exit}' <<< "$rows")"
    [ -n "$line" ] || continue
    read -r _ id st priv pub <<< "$line"
    echo "$m $id $st $priv $pub" >> "$state/instances.tmp"
    [ "$pub" != "None" ] || continue
    cat >> "$state/ssh_config.tmp" <<EOC
Host $m
  HostName $pub
  User $user
  IdentityFile $key
  IdentitiesOnly yes
  StrictHostKeyChecking accept-new
  UserKnownHostsFile $state/known_hosts
  ServerAliveInterval 15
  ConnectTimeout 10
  LogLevel ERROR
EOC
  done
  mv "$state/instances.tmp" "$state/instances"; mv "$state/ssh_config.tmp" "$state/ssh_config"
  : > "$state/known_hosts"
}
wait_ssh() {   # until every machine answers over ssh and cloud-init is done (apt is locked until then)
  local deadline=$((SECONDS + 300)) m
  for m in "${machines[@]}"; do
    grep -q "^Host $m\$" "$state/ssh_config" || die "$m has no address (is it running? $0 status)"
    until ssh -F "$state/ssh_config" -o BatchMode=yes "$m" true 2>/dev/null; do
      [ "$SECONDS" -lt "$deadline" ] || die "$m does not answer over ssh after 300 s (the security group, the key, or still booting)"
      sleep 5
    done
    ssh -F "$state/ssh_config" "$m" 'cloud-init status --wait >/dev/null 2>&1 || true'
    echo "$m: ssh ok"
  done
}
accept_host_keys() {   # uc runs the system ssh: ~/.ssh/known_hosts must hold the new addresses' keys
  local m pub
  while read -r m _ _ _ pub; do
    [ -n "$m" ] && [ "$pub" != "None" ] || continue
    ssh-keygen -R "$pub" >/dev/null 2>&1 || true   # an address AWS handed out before, to a machine with another key
    ssh -i "$key" -o IdentitiesOnly=yes -o StrictHostKeyChecking=accept-new -o BatchMode=yes -o LogLevel=ERROR "$user@$pub" true
  done < "$state/instances"
}
join_cluster() {   # the head initialises the context, the others join; a member is left alone
  local m id st priv pub
  while read -r m id st priv pub; do
    [ -n "$m" ] || continue
    if cluster_has "$m"; then
      echo "$m is in cluster context $ctx"
    elif ! ctx_exists; then
      echo "initialising cluster context $ctx on $m (WireGuard endpoint $priv:$wg_port)"
      uc machine init "$user@$pub" -c "$ctx" -n "$m" --network "$network" --no-caddy --no-dns \
        --public-ip none --wg-endpoint "$priv:$wg_port" -i "$key" -y
    else
      echo "adding $m to cluster context $ctx (WireGuard endpoint $priv:$wg_port)"
      uc machine add "$user@$pub" -n "$m" --no-caddy --public-ip none --wg-endpoint "$priv:$wg_port" -i "$key" -y
    fi
  done < "$state/instances"
}
reconnect() {   # reconnect OLD_INSTANCES_TEXT: the uc context's ssh connections follow the new public addresses
  local pairs=() m pub old
  while read -r m _ _ _ pub; do
    [ -n "$m" ] && [ "$pub" != "None" ] || continue
    old="$(awk -v m="$m" '$1 == m {print $5}' <<< "$1")"
    [ -n "$old" ] && [ "$old" != "None" ] && [ "$old" != "$pub" ] && pairs+=("$user@$old=$user@$pub")
  done < "$state/instances"
  [ "${#pairs[@]}" -gt 0 ] && ctx_exists && uc_config_rewrite reconnect "${pairs[@]}"
  return 0
}
write_env() {   # .harness/aws/env: what the driver and the runner need, from the state files
  local f="$state/env" head_pub="" cidr="" ak sk
  mkdir -p "$state"
  [ -f "$state/instances" ] && head_pub="$(awk -v m="${machines[0]}" '$1 == m {print $5}' "$state/instances")"
  [ -f "$state/vpc" ] && cidr="$(awk '{print $2}' "$state/vpc")"
  (umask 077; {
    echo "# written by deploy/uncloud/aws.sh on $(date -u +%Y-%m-%dT%H:%MZ); the uncloud driver and the scenario runner read it after .env"
    echo "# when .env says DISTRAINER_ENV_FILE=.harness/aws/env (up, start, bucket and destroy rewrite it)"
    echo "DISTRAINER_UNCLOUD_PROVIDER=aws"
    echo "DISTRAINER_UNCLOUD_CONTEXT=$ctx"
    echo "DISTRAINER_UNCLOUD_MACHINES=\"${machines[*]}\""
    echo "DISTRAINER_UNCLOUD_SSH=%s"
    echo "DISTRAINER_UNCLOUD_SSH_OPTS=\"-F $state/ssh_config\""
    [ -n "$cidr" ] && echo "DISTRAINER_UNCLOUD_HOST_PREFIX=$cidr"
    if [ -n "$head_pub" ] && [ "$head_pub" != "None" ]; then echo "DISTRAINER_UNCLOUD_HEAD_ADDRESS=$head_pub"; fi
    if [ -f "$state/s3-credentials" ]; then
      read -r ak sk < "$state/s3-credentials"
      echo "S3_ENDPOINT=https://s3.$region.amazonaws.com"
      echo "S3_REGION=$region"
      echo "S3_ACCESS_KEY=$ak"
      echo "S3_SECRET_KEY=$sk"
      echo "# bucket: $(bucket_name) (the one harness-s3.yaml names, or DISTRAINER_AWS_BUCKET)"
    fi
  } > "$f.tmp")
  mv "$f.tmp" "$f"; chmod 600 "$f"
  echo "wrote $f (DISTRAINER_ENV_FILE=.harness/aws/env in .env makes the driver read it)"
}
bring_up() {   # bring_up OLD_INSTANCES_TEXT: after the instances run: addresses, ssh, the cluster, the env file
  refresh
  wait_ssh
  accept_host_keys
  reconnect "$1"
  join_cluster
  write_env
  wait_up 300
  uc machine ls
}
cost_note() {
  echo "running: 1 x $head_type + $((${#machines[@]} - 1)) x $worker_type with public addresses, about 0.15 USD per hour" \
    "(us-east-1 on-demand, 2026); '$0 stop' parks them (disks only), '$0 destroy' removes them"
}

verb="${1:-}"; shift || true
case "$verb" in
  up)
    need aws "install the AWS CLI"; need uc "brew install psviderski/tap/uncloud"; need ssh "ssh"; need curl "curl"
    need python3 "the subnet check"; need uv "the uc config rewrite"
    # the zone is chosen once: a later `up` (a machine relaunched after a partial destroy) keeps the
    # recorded subnet while it is still a default subnet of that VPC, so the bed stays in one zone
    if [ -f "$state/vpc" ] && read -r vpc cidr subnet _ < "$state/vpc" && [ -n "$subnet" ] \
      && [ "$(awsc ec2 describe-subnets --subnet-ids "$subnet" --filters "Name=vpc-id,Values=$vpc" Name=default-for-az,Values=true --query 'length(Subnets)' 2>/dev/null)" = "1" ]; then
      echo "keeping subnet $subnet of $vpc (recorded)"
    else
      info="$(vpc_info)"
      read -r vpc cidr subnet <<< "$info"
    fi
    check_overlap "$network" "$cidr" "the VPC"
    allow_cidr="$(allowed_cidr)"
    ensure_key_pair
    sg="$(ensure_sg "$vpc")"
    ensure_rules "$sg" "$allow_cidr"
    mkdir -p "$state"; echo "$vpc $cidr $subnet $sg" > "$state/vpc"
    old="$(cat "$state/instances" 2>/dev/null || true)"
    ami=""; i=0; ids=""
    for m in "${machines[@]}"; do
      type="$worker_type"; [ "$i" = 0 ] && type="$head_type"
      line="$(instances | awk -v m="$m" '$1 == m {print; exit}')"
      if [ -z "$line" ]; then
        [ -n "$ami" ] || ami="$(ami_info)"
        id="$(launch "$m" "$type" "$ami" "$sg" "$subnet")"
        echo "launched $m: $id ($type, ${ami% *}, $disk_gb GB gp3)"
      else
        read -r _ id st _ _ <<< "$line"
        case "$st" in
          stopped) awsc ec2 start-instances --instance-ids "$id" >/dev/null; echo "starting $m ($id)" ;;
          stopping) die "$m ($id) is stopping; wait and run up again" ;;
          *) echo "$m is $st ($id)" ;;
        esac
      fi
      ids="$ids $id"; i=$((i + 1))
    done
    # shellcheck disable=SC2086
    awsc ec2 wait instance-running --instance-ids $ids
    bring_up "$old"
    cost_note ;;
  status)
    echo "instances of context $ctx in $region (machine id state private public):"
    instances | sed 's/^/  /'
    if [ -f "$state/vpc" ]; then
      read -r vpc _ _ sg < "$state/vpc"
      echo "security group $sg in $vpc admits this Mac from: $(mac_rules "$sg" | awk '{print $2}' | sort -u | tr '\n' ' ')"
    fi
    if ! ctx_exists; then echo "no uc context '$ctx'"
    elif instances | awk '$3 == "running"' | grep -q .; then uc machine ls
    else echo "no instance running: uc context '$ctx' is unreachable until 'start'"; fi ;;
  stop)
    ids="$(instances | awk '$3 == "running" || $3 == "pending"' | instance_ids)"
    [ -n "$ids" ] || { echo "nothing running in context $ctx"; exit 0; }
    # shellcheck disable=SC2086
    awsc ec2 stop-instances --instance-ids $ids >/dev/null
    echo "stopping $ids (their public addresses are released; 'start' assigns new ones)"
    # shellcheck disable=SC2086
    awsc ec2 wait instance-stopped --instance-ids $ids
    instances ;;
  start)
    old="$(cat "$state/instances" 2>/dev/null || true)"
    ids="$(instances | awk '$3 == "stopped"' | instance_ids)"
    if [ -n "$ids" ]; then
      # shellcheck disable=SC2086
      awsc ec2 start-instances --instance-ids $ids >/dev/null
      echo "starting $ids"
    fi
    all="$(instances | instance_ids)"
    [ -n "$all" ] || die "no instances in context $ctx: run up"
    # shellcheck disable=SC2086
    awsc ec2 wait instance-running --instance-ids $all
    if [ -f "$state/vpc" ]; then   # the Mac may have moved networks; without an answer the rule stays as it is
      read -r _ _ _ sg < "$state/vpc"
      if cidr="$(allowed_cidr 2>/dev/null)"; then ensure_rules "$sg" "$cidr"; else echo "warning: could not learn this Mac's address; the ssh rule is unchanged" >&2; fi
    fi
    bring_up "$old"
    cost_note ;;
  destroy)
    ids="$(instances | instance_ids)"
    if [ -n "$ids" ]; then
      # shellcheck disable=SC2086
      awsc ec2 terminate-instances --instance-ids $ids >/dev/null
      echo "terminating $ids"
      # shellcheck disable=SC2086
      awsc ec2 wait instance-terminated --instance-ids $ids
    fi
    if [ -f "$state/vpc" ]; then
      read -r vpc _ _ sg < "$state/vpc"
      for attempt in 1 2 3 4 5 6; do   # the instances' network interfaces take a moment to let go of the group
        if awsc ec2 delete-security-group --group-id "$sg" 2>/dev/null; then echo "deleted security group $sg"; break; fi
        [ "$attempt" = 6 ] && echo "security group $sg not deleted (still in use?): aws ec2 delete-security-group --group-id $sg" >&2
        sleep 10
      done
    fi
    if awsc ec2 describe-key-pairs --key-names "$name" >/dev/null 2>&1; then
      awsc ec2 delete-key-pair --key-name "$name" >/dev/null && echo "deleted key pair $name"
      if [ -f "$state/key-created" ]; then rm -f "$key" "$state/key-created"; echo "removed $key"; fi
    fi
    if ctx_exists; then ctx_forget; fi
    rm -f "$state/instances" "$state/ssh_config" "$state/known_hosts" "$state/vpc"
    write_env
    echo "the bucket and its IAM user stay ('$0 bucket-rm' removes them)" ;;
  bucket)
    need aws "install the AWS CLI"
    bucket="$(bucket_name)"; iam_user="$name-harness"
    if awsc s3api head-bucket --bucket "$bucket" 2>/dev/null; then
      # an existing bucket is used only when this script tagged it (or DISTRAINER_AWS_ADOPT_BUCKET=1
      # says it is yours to tag): the name comes from an editable config, and bucket-rm deletes by tag
      owner="$(bucket_owner "$bucket")"
      if [ "$owner" = "$ctx" ]; then echo "bucket $bucket exists (tagged distrainer:cluster=$ctx)"
      elif [ -z "$owner" ] && [ "${DISTRAINER_AWS_ADOPT_BUCKET:-0}" = "1" ]; then
        awsc s3api put-bucket-tagging --bucket "$bucket" --tagging "TagSet=[{Key=distrainer:cluster,Value=$ctx}]"
        echo "bucket $bucket exists; adopted (tagged distrainer:cluster=$ctx)"
      else
        die "bucket $bucket exists and is not tagged distrainer:cluster=$ctx (${owner:-no tag}): name another bucket in harness-s3.yaml, or DISTRAINER_AWS_ADOPT_BUCKET=1 if it is yours"
      fi
    else
      if [ "$region" = "us-east-1" ]; then awsc s3api create-bucket --bucket "$bucket" >/dev/null
      else awsc s3api create-bucket --bucket "$bucket" --create-bucket-configuration "LocationConstraint=$region" >/dev/null; fi
      awsc s3api put-bucket-tagging --bucket "$bucket" --tagging "TagSet=[{Key=distrainer:cluster,Value=$ctx}]"   # what bucket-rm looks for
      echo "created bucket $bucket in $region (tagged distrainer:cluster=$ctx)"
    fi
    awsc s3api put-public-access-block --bucket "$bucket" \
      --public-access-block-configuration BlockPublicAcls=true,IgnorePublicAcls=true,BlockPublicPolicy=true,RestrictPublicBuckets=true
    if ! awsc iam get-user --user-name "$iam_user" >/dev/null 2>&1; then
      awsc iam create-user --user-name "$iam_user" --tags "Key=distrainer:cluster,Value=$ctx" >/dev/null
      echo "created IAM user $iam_user"
    fi
    mkdir -p "$state"
    cat > "$state/bucket-policy.json" <<EOP
{"Version": "2012-10-17", "Statement": [
  {"Effect": "Allow", "Action": ["s3:ListBucket", "s3:GetBucketLocation", "s3:ListBucketMultipartUploads"], "Resource": "arn:aws:s3:::$bucket"},
  {"Effect": "Allow", "Action": ["s3:GetObject", "s3:PutObject", "s3:DeleteObject", "s3:AbortMultipartUpload", "s3:ListMultipartUploadParts"],
   "Resource": "arn:aws:s3:::$bucket/*"}]}
EOP
    awsc iam put-user-policy --user-name "$iam_user" --policy-name "$name-bucket-$bucket" --policy-document "file://$state/bucket-policy.json"
    echo "policy $name-bucket-$bucket on $iam_user: the bucket only"
    if [ ! -f "$state/s3-credentials" ]; then
      (umask 077; awsc iam create-access-key --user-name "$iam_user" --query 'AccessKey.[AccessKeyId,SecretAccessKey]' | tr '\t' ' ' > "$state/s3-credentials.tmp")
      mv "$state/s3-credentials.tmp" "$state/s3-credentials"; chmod 600 "$state/s3-credentials"
      echo "created an access key for $iam_user ($state/s3-credentials, mode 600; a new key takes about ten seconds to work)"
    fi
    write_env ;;
  bucket-rm)
    bucket="$(bucket_name)"; iam_user="$name-harness"
    if awsc s3api head-bucket --bucket "$bucket" 2>/dev/null; then
      # only a bucket this script made or adopted (tagged by `bucket`): the name comes from an editable config
      owner="$(bucket_owner "$bucket")"
      [ "$owner" = "$ctx" ] || die "bucket $bucket is not tagged distrainer:cluster=$ctx (${owner:-no tag}): refusing to delete it"
      awsc s3 rm "s3://$bucket" --recursive >/dev/null
      # an interrupted checkpoint put leaves a multipart upload behind, and a bucket with one is not empty
      awsc s3api list-multipart-uploads --bucket "$bucket" --query 'Uploads[].[Key, UploadId]' | tr '\t' ' ' \
        | while read -r k u; do [ -n "$k" ] && [ "$k" != "None" ] && awsc s3api abort-multipart-upload --bucket "$bucket" --key "$k" --upload-id "$u"; done
      awsc s3api delete-bucket --bucket "$bucket"
      echo "deleted bucket $bucket"
    fi
    if awsc iam get-user --user-name "$iam_user" >/dev/null 2>&1; then
      for k in $(awsc iam list-access-keys --user-name "$iam_user" --query 'AccessKeyMetadata[].AccessKeyId'); do
        awsc iam delete-access-key --user-name "$iam_user" --access-key-id "$k"
      done
      for pol in $(awsc iam list-user-policies --user-name "$iam_user" --query 'PolicyNames'); do
        awsc iam delete-user-policy --user-name "$iam_user" --policy-name "$pol"
      done
      awsc iam delete-user --user-name "$iam_user"
      echo "deleted IAM user $iam_user"
    fi
    rm -f "$state/s3-credentials" "$state/bucket-policy.json"
    write_env ;;
  env)
    [ -f "$state/env" ] || write_env >/dev/null
    sed 's/^\(S3_SECRET_KEY=\).*/\1<in the file>/' "$state/env" ;;
  *)
    echo "usage: $0 up | status | stop | start | destroy | bucket | bucket-rm | env" >&2; exit 2 ;;
esac
