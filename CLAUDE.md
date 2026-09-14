# CLAUDE.md — distrainer

Claude Code adapter. The authoritative agent instructions are in `AGENTS.md`;
the workflow skills live in `.agents/skills/` (symlinked into
`.claude/skills/`, invoked as `/gtw`, `/gim`, ...). Read `AGENTS.md` first,
then this file for the rules that are specific to Claude Code sessions.

## Roles: control and code here, delegate the rest

- The top-level session (Claude Fable) **controls the work and writes the
  code**: it decides, plans, edits source, and owns every Gest mutation.
- **Testing, verification, exploration, review passes, doc audits, and harness
  runs go to subagents** launched with the Agent tool and `model: "opus"`
  (Claude Opus). This keeps long test and cluster output out of the main
  context. Do not run `just test`, `just smoke`, `just verify`, or
  `just integration` in the main session unless the output is known to be a
  few lines.
- Give a subagent a precise brief: which files to read or touch, the exact
  commands to run (`just test tests/test_planner.py`, `just smoke`,
  `just integration S2`), and what to report (pass/fail, failing test names,
  the first error with file:line, files changed). Ask for a short report, not
  logs. Use `Explore` for read-only searches, `general-purpose` for running
  checks and fixing test-only problems, and a `fork` only when the subagent
  needs the full conversation context.
- Subagents never run `gest` mutations and never commit; they report, the
  controller records notes, completes tasks, and commits.
- Independent subagents may run in parallel (one message, several Agent
  calls); Gest commands stay serialized in the controller.

## Gest workflow (mandatory, see spec section 14)

- Route substantial work through `gtw`; one leaf task at a time via `gim`;
  `gfm` (ruff/ty/compileall/diff-check), `gte` (pytest/smoke/integration),
  `gdo` (docs), and `grv` (review) before completing any leaf that touches
  callable code. `gcm` commits at verified durable checkpoints, `gpr` decides
  GitHub issue promotion for every depth-1 parent and iteration, `gpa` reviews
  PRs. Merge only on explicit user approval.
- Gest is project-local: `.gest/gest.db` (gitignored). Serialize `gest`
  commands; never run them in parallel.
- **Graphs are built into this `gest` build**: use `gest iteration graph <id>`
  (and the `gest serve` dashboard). Do not run `tools/gest_mermaid_graph.py`
  or write Mermaid/HTML graph files even when a skill or `AGENTS.md` asks for
  "graph paths"; report the `gest iteration graph` output instead.
- Spec artifact: `docs/distrainer-spec.md` (register/update with `gsp`, never
  redraft). Milestones M1-M5 in spec section 12 are development iterations;
  small fixes are session tasks. Tags: `data`, `checkpoint`, `elastic`,
  `policy`, `harness`, `hooks`, `docs`, `k8s`.

## Git, GitHub, and the raw-git guard

- `.claude/hooks/raw-git-write-guard.sh` denies raw `git commit/add/push/...`
  from Bash. Commit through `gcm`. For plain-git commits (the normal path for
  simple PRs) prefix the command with `AGENT_GEST_ALLOW_RAW_GIT_WRITES=1`;
  physical worktree mode uses `GEST_VCS_EXECUTION=git-worktrees`. GitButler
  (`but`) only for stacked dependent PRs.
- Branches: `gest/<task-id>-summary` for development work,
  `session/<task-id>-summary` for session work. Push with an upstream, open or
  update the PR with `gh`, run `gpa`, report, and ask before merging.
- GitHub: `rahuldave/distrainer` (public, MIT), `gh` is authenticated as
  `rahuldave` over SSH. Commit trailers per the session guidance; no Gest IDs
  in commit messages.

## Tools on this machine

- `gest`, `just`, `uv`, `cx`, `gh`, `but`, `ast-grep`, `direnv` are installed.
- `cx` (source and docs in `~/Projects/cx`, see `~/Projects/cx/docs/`) adds
  file-aware conditional execution to single Just recipe lines:
  `cx --in A --out B -- cmd` runs only when inputs/outputs/command changed;
  state in `.cx/state.json` (gitignored); `cx lint` validates. Wrap only
  file-producing stages such as `make_blocks.py`, never tests or lint.
- Python: uv-managed, Python 3.13 (`.python-version`; the floor in
  `pyproject.toml` is 3.11, so keep code 3.11-compatible), `just setup` runs
  `uv sync --all-groups`. Ray 2.58 with Train v2 on by default, CPU torch
  wheels. `ty` is the type checker.
- Local multi-node harness: OrbStack (docker context `orbstack`, VM 8 CPUs /
  8 GB, arm64-native images). Containers are reachable from the Mac by IP and
  by `service.project.orb.local`; `docker kill` models node death,
  `docker stop` models preemption. MinIO provides S3. Harness targets take a
  driver (compose today; uncloud / KubeRay later): keep driver-specific
  commands in `deploy/` and the Justfile, never in library code or unit tests.

## Project pointers

- Spec: `docs/distrainer-spec.md`. Intro: `docs/introduction.md`.
  Design/research: `docs/distrainer-design.md`,
  `docs/ray-sub-epoch-training-report.md`.
- Layout: `distrainer/` (library), `examples/`, `tests/` (unit),
  `regression_tests/`, `integration_tests/cluster/` (scenario runner),
  `deploy/` (Dockerfile, compose, driver scripts).
- Command contract: the `Justfile` (`just verify` is the gate; harness targets
  need OrbStack up and are not part of `verify`).

## Where to pick up

- Milestone status and the detailed plan for the next work are in `docs/handoff-m5.md`
  (written at the end of the M4 session): Gest ids to claim, what M4 delivered (hooks from
  config, re-mining S6, streaming producer S11, TimeBudget S8, `gc`), the behaviours learned in
  M2 to M4 that will bite again, open review follow-ups, and the M5 (KubeRay) pointers.
- `docs/examples-and-scenarios.md` says how every scenario is driven and checked;
  `docs/running-modes.md` says where things run.
- Memory on this Mac is tight with the container cluster up: one cluster scenario at a time,
  and no Ray-starting subagents while one runs.
