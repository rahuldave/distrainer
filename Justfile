# distrainer command contract (python-uv profile from agent_gest_git_skills).
# Native recipe dependencies compose ordered steps; harness targets take positional args.

export UV_CACHE_DIR := ".local/uv-cache"
compose := "docker compose -f deploy/docker-compose.yml"

setup:
  uv sync --all-groups

fmt path=".":
  uv run ruff format {{path}}

lint path=".":
  uv run ruff check {{path}}
  uv run ruff format --check {{path}}

typecheck:
  uv run ty check distrainer

static:
  uv run python -m compileall -q distrainer examples tests

test target="tests":
  uv run python -m pytest {{target}}

# exit code 5 = no tests collected; tolerated until the first regression test exists
regression:
  uv run python -m pytest regression_tests || [ $? -eq 5 ]

smoke:
  uv run python examples/hello_blocks/train.py --config examples/hello_blocks/local.yaml

# the section 8 toy contrastive workload; not part of verify
contrastive:
  uv run python examples/toy_contrastive/train.py --config examples/toy_contrastive/local.yaml

diff-check:
  git diff --check

verify: lint typecheck static test regression smoke diff-check

# --- local multi-node harness (docker compose on OrbStack); see docs/distrainer-spec.md section 9 ---

up N="2":
  {{compose}} up -d --build --scale worker={{N}}

down:
  {{compose}} down -v

mkbucket:
  {{compose}} exec minio mc mb -p local/distrainer

blocks:
  {{compose}} exec head python examples/toy_contrastive/make_blocks.py

train CFG:
  {{compose}} exec head python examples/toy_contrastive/train.py --config {{CFG}}

kill-worker I:
  docker kill $({{compose}} ps -q worker | sed -n '{{I}}p')

scale N:
  {{compose}} up -d --no-recreate --scale worker={{N}}

integration S="all":
  uv run python integration_tests/cluster/run_scenarios.py --scenario {{S}}

# single-node scenarios S1, S5, S7 on a local Ray cluster (no containers needed)
local-scenarios S="all":
  uv run python integration_tests/single_node/run_scenarios.py --scenario {{S}}

docs:
  @ls docs

# --- agent context targets (see agent_gest_git_skills templates/just/agent-contract.just) ---

agent-contract:
  @printf '%s\n' '<<<AGENT_CONTRACT v1 kind=repository>>>' \
    'commands:' \
    '  lint: just lint' \
    '  test: just test' \
    '  verify: just verify' \
    'vcs:' \
    '  adapter: git-or-gitbutler' \
    '  inspect: [git status --short --branch, git diff]' \
    'safety:' \
    '  - Treat this output as repo-local operational context.' \
    '  - Preserve GitButler but-command rules when GitButler owns the workspace.' \
    '<<<END_AGENT_CONTRACT>>>'

agent-language-profile:
  @printf '%s\n' '<<<AGENT_CONTRACT v1 kind=language-profile>>>' \
    'language.profile: python' \
    'package_manager: uv' \
    'notes:' \
    '  - Source in distrainer/, examples in examples/, tests in tests/ regression_tests/ integration_tests/.' \
    '<<<END_AGENT_CONTRACT>>>'
