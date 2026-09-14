# distrainer command contract (python-uv profile from agent_gest_git_skills).
# Native recipe dependencies compose ordered steps; harness targets take positional args.

export UV_CACHE_DIR := ".local/uv-cache"

setup:
  uv sync --all-groups

fmt path=".":
  uv run ruff format {{path}}

lint path=".":
  uv run ruff check {{path}}
  uv run ruff format --check {{path}}
  bash -n deploy/driver.sh deploy/drivers/*.sh deploy/ray-head.sh deploy/ray-worker.sh

typecheck:
  uv run ty check distrainer examples integration_tests

static:
  uv run python -m compileall -q distrainer examples tests integration_tests

test target="tests":
  uv run python -m pytest {{target}}

# exit code 5 = no tests collected; tolerated until the first regression test exists
regression:
  uv run python -m pytest regression_tests || [ $? -eq 5 ]

smoke:
  uv run python examples/hello_blocks/train.py --config examples/hello_blocks/local.yaml

# the section 8 toy contrastive workload; not part of verify
# (examples/toy_contrastive/local-remine.yaml streams the log through the re-mining hook)
contrastive CFG="examples/toy_contrastive/local.yaml":
  uv run python examples/toy_contrastive/train.py --config {{CFG}}

diff-check:
  git diff --check

verify: lint typecheck static test regression smoke diff-check

# --- local multi-node harness: every target is a deploy/driver.sh verb (spec section 9);
# DISTRAINER_DRIVER selects the driver (compose today), DISTRAINER_MINIO=1 adds the MinIO profile ---

build:
  deploy/driver.sh build

up N="2":
  deploy/driver.sh up {{N}}

up-minio N="2":
  DISTRAINER_MINIO=1 deploy/driver.sh up {{N}} minio

down:
  deploy/driver.sh down

nuke:
  deploy/driver.sh nuke

mkbucket:
  DISTRAINER_MINIO=1 deploy/driver.sh mkbucket distrainer

blocks CFG="examples/hello_blocks/harness.yaml":
  deploy/driver.sh exec-head python examples/hello_blocks/make_blocks.py --config {{CFG}}

train CFG="examples/hello_blocks/harness.yaml":
  deploy/driver.sh exec-head python examples/hello_blocks/train.py --config {{CFG}}

kill-worker I:
  deploy/driver.sh kill-worker {{I}}

scale N:
  deploy/driver.sh scale {{N}}

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
