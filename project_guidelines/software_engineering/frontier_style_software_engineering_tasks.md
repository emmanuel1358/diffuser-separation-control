# Authoring Frontier-Style Long-Horizon Software Engineering RL Tasks

This guide is for engineers who want to create reinforcement-learning tasks in
which an agent must understand, modify, debug, migrate, or operate a real
software repository over a long tool-using episode.

The framework supports the useful task shapes associated with FrontierBench:

- focused repository debugging,
- legacy and runtime migrations,
- concurrency and reliability repair,
- database and distributed-system changes,
- multi-service cutovers,
- stateful workflows with trusted pre-grade snapshots,
- tool-using environments exposed through MCP/SSE,
- separate, network-isolated verifier containers,
- Taiga deployment as one outer image containing all child images, and
- Harbor export from the same authored task.

It does **not** require authors to package a task in FrontierBench or Harbor
format. Author one native ISO task and let the shared exporters project it to
Taiga and Harbor.

Read this guide together with:

- [`docs/SOFTWARE_ENGINEERING_FRAMEWORK.md`](../../docs/SOFTWARE_ENGINEERING_FRAMEWORK.md)
  for the exact schema and runtime architecture;
- [`docs/SOFTWARE_TRANSFORMATION_TASKS.md`](../../docs/SOFTWARE_TRANSFORMATION_TASKS.md)
  for the repository-transformation grading contract;
- [`docs/RUBRIC_EVALUATION.md`](../../docs/RUBRIC_EVALUATION.md) for
  `RubricTask`, `WorkspaceArtifact`, and fault semantics;
- [`docs/REWARD_HACKING.md`](../../docs/REWARD_HACKING.md) for the shared
  security model; and
- [`new_task_design_workflow.md`](new_task_design_workflow.md) for the
  idea-to-acceptance workflow for a brand-new task.

## 1. The Task You Are Building

An RL task is an executable specification of an engineering challenge:

1. The agent receives `instruction.md`, a seeded repository, and public tools.
2. The agent works for many turns using shell, editor, tests, services, and
   optional MCP tools.
3. The agent leaves a repository or other declared artifacts.
4. A trusted grader commits those artifacts, runs hidden behavioral checks, and
   returns a deterministic reward between `0.0` and `1.0`.
5. Infrastructure failures are discarded; candidate failures become a kept
   agent-fault reward, normally `0.0`.

The task is useful for RL only when the reward tracks the engineering capability
we want to train. Repository size, build time, or a long prompt do not create a
long-horizon task by themselves. The task should require the agent to maintain
state, form and revise a plan, inspect multiple components, use feedback, and
make a coherent final change.

### 1.1 A strong task has all of these properties

- **Real engineering work.** The task resembles a change a strong engineer
  could own: diagnose a race, migrate a protocol, repair recovery behavior,
  preserve a UI invariant, or coordinate a service cutover.
- **A complete public specification.** Hidden tests vary cases, schedules,
  scales, and seeds. They do not introduce hidden product requirements.
- **Behavioral grading.** The grader measures externally meaningful behavior,
  not only filenames, line counts, or exact implementation text.
- **A verified oracle.** `solution/solve.sh` proves the task is solvable and
  scores `1.0`.
- **A meaningful floor.** A no-op, prompt-copy, hardcoded public-case solution,
  or reward-forgery attempt scores at or near `0.0`.
- **Determinism.** Repeated grading of the same committed submission returns the
  same result.
- **A fair path to progress.** Public checks expose the interface and basic
  workflow without disclosing the held-out answer.
- **Bounded execution.** Every build, test, service probe, capture, and artifact
  traversal has a timeout or size bound.
- **A real trust boundary.** Candidate code never executes directly in the
  root grader process and cannot read hidden fixtures.

### 1.2 Weak task patterns to avoid

Do not create tasks whose difficulty comes mostly from:

- missing dependencies or undocumented commands;
- enormous downloads or repeated clean builds;
- hidden APIs, hidden data formats, or unstated acceptance criteria;
- one brittle golden string;
- static checks that can be satisfied without correct behavior;
- random or flaky tests;
- unrestricted internet access;
- an LLM judge;
- a grader that rewards partial output after crashing;
- an oracle that contains the expected answer but does not derive it from
  disclosed inputs; or
- a service graph that exists only to make packaging complicated.

### 1.3 Non-negotiable acceptance requirements

An authored problem is complete only when both requirements hold:

1. **Trusted CI is green.** Every required trusted build, validation,
   ground-truth, grader QA, security, export, and task-specific check must pass.
   There must be no unresolved required check hidden behind a local-only pass.
2. **The Boreal aggregate score is `<= 0.4`.** This is the problem-level
   aggregate reported by Boreal across the configured evaluation attempts, not
   the oracle score and not a hand-selected single rollout.

The oracle must still score `1.0`, and trivial/attack baselines must remain at
the floor. The Boreal threshold is an additional difficulty requirement: a task
with aggregate score above `0.4` is too easy for the target cohort and must be
revised.

Do not lower the aggregate with hidden requirements, flaky tests, shorter
timeouts, or packaging friction. Improve difficulty by deepening the fully
specified engineering work, broadening semantic hidden coverage, strengthening
interacting invariants, or closing a legitimate shortcut.

## 2. Choose the Smallest Correct Execution Shape

The framework has three practical software-task modes. Start with the smallest
mode that expresses the real environment.

### 2.1 Mode A: secure single-image repository task

Use this for most debugging, feature, migration, and performance tasks.

- One task image contains the seeded repository, public tools, private grader,
  and trusted reference dependencies.
- The agent writes the final repository to `/tmp/output/repo`.
- `WorkspaceArtifact("repo", ...)` commits and validates it.
- Candidate builds and hidden tests run through `context.run_candidate()` or
  `context.run_candidate_suite()`.
- No native capability sections are required.

Start from:

```bash
uv run lbx-rl-template create \
  --name labelbox/my-software-task \
  --template software-engineering \
  --out problems
```

Canonical examples:

- [`wal-recovery-ordering`](https://github.com/Alignerr-Code-Labeling/lbx-rl-tasks-iso-template/tree/main/examples/wal-recovery-ordering)
- [`xfoil-rust-port`](https://github.com/Alignerr-Code-Labeling/lbx-rl-tasks-iso-template/tree/main/examples/xfoil-rust-port)

### 2.2 Mode B: capability task with an implicit main service

Use this when one image is still sufficient but you need typed workspace
initialization, checkpoint restoration, service-qualified artifacts, or explicit
evaluation metadata.

Declaring `[workspace]`, `[[artifacts]]`, `[[captures]]`, `[[mcp_servers]]`, or
`[evaluation]` opts into capability export. If `[[services]]` is omitted, the
framework synthesizes `main` from `environment/Dockerfile` or
`environment/main/Dockerfile`.

Important consequences:

- Taiga export uses an outer capsule image.
- The implicit service inherits the authored resource and network policy.
- The software grader still uses the simple repository contract when no
  explicit service graph is declared.
- Use this mode intentionally; do not add capability sections as decorative
  metadata.

### 2.3 Mode C: explicit multi-service capsule

Use this only when the engineering problem genuinely requires live services or
trusted state collection:

- databases, message brokers, browsers, or customer simulators;
- one-shot initialization or migration jobs;
- health-gated startup;
- cross-service artifacts;
- ordered pre-verification captures;
- an MCP/SSE sidecar; or
- a separate verifier image with different dependencies.

An explicit graph declares exactly one `role = "main"` service and may declare
sidecar, init, and verifier services. It also declares a canonical `[result]`
when a verifier service is present.

Canonical examples:

- [`frontier-service-cutover`](https://github.com/Alignerr-Code-Labeling/lbx-rl-tasks-iso-template/tree/main/examples/frontier-service-cutover)
- [`frontier-mcp-workspace`](https://github.com/Alignerr-Code-Labeling/lbx-rl-tasks-iso-template/tree/main/examples/frontier-mcp-workspace)

Do not choose a service graph merely because the source application normally
uses Docker Compose. If the hidden behavior can be tested in one image without
changing the engineering challenge, use Mode A.

## 3. Canonical Examples and What to Copy

Examples are reviewed pattern libraries, not templates to clone wholesale.

### 3.1 WAL recovery ordering

[`wal-recovery-ordering`](https://github.com/Alignerr-Code-Labeling/lbx-rl-tasks-iso-template/tree/main/examples/wal-recovery-ordering)
demonstrates:

- a focused concurrency/reliability bug;
- a public RPC adapter and hidden worker;
- structural, behavioral, determinism, and performance criteria;
- candidate execution under the shared UID-dropped boundary;
- required gates that prevent cosmetic partial credit; and
- reward-forgery and worker-mutation attacks that must score `0.0`.

Copy its pattern when the final artifact is one repository and the hidden suite
can orchestrate behavior from the grader.

### 3.2 XFOIL Rust port

[`xfoil-rust-port`](https://github.com/Alignerr-Code-Labeling/lbx-rl-tasks-iso-template/tree/main/examples/xfoil-rust-port)
demonstrates:

- a long legacy-modernization task;
- public reference, candidate, and diff modes over a versioned JSONL protocol;
- held-out geometry, inviscid, viscous, and stateful polar suites;
- an offline vendored toolchain;
- a root-owned monitor that prevents subprocess delegation;
- a typed `SYS_PTRACE` verifier capability;
- a sealed candidate build produced under UID 1000; and
- upstream source and license provenance.

Copy its protocol and anti-delegation ideas only when your task really needs
public differential testing or must prevent reuse of a legacy executable.

### 3.3 Service cutover

[`frontier-service-cutover`](https://github.com/Alignerr-Code-Labeling/lbx-rl-tasks-iso-template/tree/main/examples/frontier-service-cutover)
demonstrates:

- `main`, sidecar, init, and verifier roles;
- `started`, `healthy`, and `completed` dependencies;
- named shared volumes;
- ordered, atomic capture hooks;
- service-qualified artifact collection;
- a verifier with no network; and
- one result file written after the agent graph is frozen.

Copy this pattern for migrations and cutovers where the final answer includes
runtime state that cannot be reconstructed from the repository alone.

### 3.4 MCP workspace

[`frontier-mcp-workspace`](https://github.com/Alignerr-Code-Labeling/lbx-rl-tasks-iso-template/tree/main/examples/frontier-mcp-workspace)
demonstrates:

- a shared workspace service;
- an agent-facing MCP/SSE sidecar;
- readiness gates;
- service-DNS routing through the audited proxy; and
- the same isolated-verifier and artifact-freeze pattern.

Copy this pattern when an agent needs task-local tools whose calls must be
auditable and constrained to declared services.

## 4. Repository Layout

A strong single-image task normally contains:

```text
problems/<task_id>/
├── task.toml
├── instruction.md
├── metadata.json
├── README.md                     # provenance, licenses, maintainer notes
├── environment/
│   ├── Dockerfile
│   ├── apt.txt
│   ├── requirements.txt
│   └── public adapters/tools
├── starter/                      # agent-visible seed
├── data/                         # other public fixtures and docs
├── scorer/
│   ├── compute_score.py          # TASK = RubricTask(...)
│   ├── evaluation.plan.json      # generated from TASK
│   ├── requirements.txt          # grader-only dependencies
│   └── data/                     # hidden fixtures/drivers
├── solution/
│   ├── solve.sh
│   └── trusted implementation files
├── baselines/
│   └── noop.sh
├── attacks/                      # deliberate reward-hacking regressions
└── tests/
    └── test.sh
```

A capability task may add service build contexts:

```text
environment/
├── main/
│   └── Dockerfile
├── database/
│   └── Dockerfile
├── verifier/
│   └── Dockerfile
└── shared public files
```

Keep ownership explicit:

- `starter/`, `data/`, and public tools are agent-visible.
- `scorer/data/` is hidden and root-only.
- `solution/` is trusted author evidence, not agent input.
- `.alignerr/` contains generated validation/build evidence; commit only the
  files the validator and task contract require.

## 5. Metadata and Taxonomy

Every software-engineering task uses:

```toml
[difficulty]
task_type = "software_engineering"
domain = "repo_debugging"
reward_type = "multi_deterministic_rubrics"
```

Choose the narrowest domain:

- `legacy_modernization`
- `behavioral_compatibility`
- `runtime_migration`
- `schema_evolution`
- `build_system_migration`
- `framework_migration`
- `compiler_toolchain_migration`
- `concurrency_reliability`
- `distributed_protocol_evolution`
- `repo_debugging`
- `feature_implementation`
- `performance_optimization`
- `frontend_ui`
- `data_database_systems`
- `security_hardening`

Do not use `continuous_scoring_function` for software-engineering repository
tasks. The shared software contract requires declarative deterministic rubrics.

### 5.1 Domain selection examples

| Engineering challenge | Domain |
| --- | --- |
| Port a mature numerical package to Rust | `legacy_modernization` |
| Preserve responses while replacing a runtime | `runtime_migration` |
| Change a wire or storage schema compatibly | `schema_evolution` |
| Move Make/CMake to another build system | `build_system_migration` |
| Repair WAL, locking, retries, or ordering | `concurrency_reliability` |
| Change consensus, replication, or messaging | `distributed_protocol_evolution` |
| Diagnose and fix a scoped repository defect | `repo_debugging` |
| Add a user-visible capability | `feature_implementation` |
| Meet hidden latency or memory budgets | `performance_optimization` |
| Repair rendering, accessibility, or interaction | `frontend_ui` |
| Migrate or repair database behavior | `data_database_systems` |
| Remove a vulnerability without regressions | `security_hardening` |

## 6. Design the Engineering Challenge Before Packaging It

Write a one-page internal design before editing `task.toml`.

### 6.1 Define the capability under test

State:

- what system behavior is wrong or missing;
- why solving it requires engineering rather than a one-line lookup;
- what public evidence an agent can gather;
- what final behavior proves success;
- which dimensions hidden grading varies;
- which shortcuts must fail; and
- what a strong but imperfect attempt should score.

If you cannot state these precisely, the grader will become a collection of
incidental assertions.

### 6.2 Define the public/hidden boundary

Public material should disclose:

- the user-visible or protocol contract;
- repository architecture needed to begin;
- build and test commands;
- representative happy and failure cases;
- performance units and broad expectations;
- installed tools; and
- any compatibility promises.

Hidden material may vary:

- values and seeds;
- data sizes;
- operation order;
- thread schedules;
- retries and injected faults;
- valid edge cases;
- service timing;
- repeated-run checks; and
- performance samples.

Hidden material must not change:

- the meaning of correctness;
- required inputs or output schema;
- supported operations;
- an unstated dependency;
- an unstated compatibility target; or
- an unstated security policy.

### 6.3 Calibrate the horizon

Long-horizon difficulty should come from interacting constraints:

- tracing behavior across modules;
- making a migration while preserving compatibility;
- understanding stateful or concurrent behavior;
- iterating against public feedback;
- coordinating code, tests, configuration, and documentation;
- keeping changes robust across hidden scales; or
- planning a sequence of service operations.

Avoid artificial horizon inflation such as duplicated code, intentionally slow
builds, excessive generated files, or vague prompts.

## 7. Write an Agent-Facing Prompt

`instruction.md` is a product specification, not a grader explanation.

Include:

- the current failure or requested feature;
- the required public behavior;
- the editable repository path;
- public commands the agent should run;
- output expectations;
- compatibility, security, and performance constraints;
- explicit non-goals; and
- enough context to identify the relevant system boundaries.

Name tools directly. For example:

```text
Work in /tmp/output/repo.
Run `pytest -q` for the public suite and `cargo test --workspace` for the Rust
components. Preserve the transform/v1 JSONL protocol described in README.md.
```

Do not tell the agent to:

- inspect `task.toml`, `metadata.json`, the Dockerfile, or the base image to
  discover hidden requirements;
- read `/mcp_server`, `scorer/`, or verifier files;
- infer a dependency from CI internals;
- call a remote service not declared as part of the task; or
- optimize against rubric weights.

Do not reveal hidden case values, exact thresholds that should remain held out,
answer-key paths, or attack fixtures.

## 8. Seed and Submission Contracts

### 8.1 Simple repository output

The starter contract is:

```toml
[[outputs]]
path = "/tmp/output/repo"
required = true
description = "Completed source repository."
```

The environment copies `starter/` into that path and gives UID 1000 ownership
to the agent. The grader declares:

```python
WorkspaceArtifact(
    "repo",
    reject_native_payloads=True,
)
```

The output path and artifact path must agree.

### 8.2 Typed workspace initialization

Capability tasks can declare:

```toml
[workspace]
seed = "starter"
root = "/workdir"
agent_cwd = "/workdir/repo"
init_policy = "copy"
git_baseline = true
clean_paths = [".cache", "__pycache__"]
checkpoint_restore = true
```

Use:

- `copy` for a fresh seed;
- `overlay` to merge the seed onto an existing root;
- `reuse` only when state intentionally persists;
- `empty` when no seed should exist;
- `git_baseline = true` for a generated baseline commit; or
- a 40-character lowercase commit ID when exact history is part of the task.

The root and agent CWD must be writable task paths. Do not put the workspace
under `/mcp_server` or overlap the verifier result root.

### 8.3 Checkpoint behavior

`checkpoint_restore = true` makes initialization idempotent when Taiga restores
the same task state. Pair it with:

```toml
[runner]
checkpoint_ttl = 86400
serialize_restore_test_interval = 3600
```

Use values appropriate to the expected episode. These runner settings are
Taiga/Boreal controls; Harbor has its own runtime controls.

## 9. Service Graph Authoring

### 9.1 Roles

- `main`: the agent-facing primary environment; exactly one when services are
  explicit and always non-root.
- `sidecar`: a long-running dependency such as a database, broker, mock customer,
  browser, or tool server.
- `init`: a one-shot setup/migration job; other services may wait for
  `condition = "completed"`.
- `verifier`: a trusted post-agent grader. At most one; normally root and
  network-isolated.

Each service uses exactly one source:

```toml
build = { context = "environment/main", dockerfile = "Dockerfile" }
```

or:

```toml
image = "registry.example/service@sha256:<64 lowercase hex characters>"
```

External images must be digest pinned. Tags such as `latest` are rejected.
Build targets must resolve to a stage declared in the Dockerfile; a misspelled
target fails before Docker can treat it as a public image name.

### 9.2 Dependencies and health

```toml
[[services]]
name = "database"
role = "sidecar"
image = "postgres@sha256:<digest>"

[services.healthcheck]
command = ["CMD-SHELL", "pg_isready -U app"]
interval_sec = 2
timeout_sec = 3
retries = 30
start_period_sec = 5

[[services]]
name = "main"
role = "main"
build = { context = "environment/main", platform = "linux/amd64" }
user = "agent"

[[services.depends_on]]
service = "database"
condition = "healthy"
```

Use:

- `started` only when process creation is sufficient;
- `healthy` when the dependency exposes a real readiness signal; and
- `completed` only for `init` services.

Do not hide fixed sleeps in startup commands. Encode readiness explicitly.

### 9.3 Network policy

Each phase can declare:

```toml
[services.resources]
network = "isolated" # none | isolated | internet
```

Use:

- `none` for the verifier and services that need no network;
- `isolated` for service-to-service traffic without internet; and
- `internet` only when the task genuinely requires it and policy permits it.

Software tasks should normally keep
`[environment].allow_internet = false`. The software validator rejects
agent-reachable services that re-enable internet.

`network_mode = "share"` joins another declared service's namespace. It cannot
cross the verifier boundary, cannot target itself, and cannot be combined with
network aliases.

### 9.4 Named volumes

Declare backend-managed volumes:

```toml
[[volumes]]
name = "database-data"

[[services.volumes]]
volume = "database-data"
target = "/var/lib/postgresql/data"
mode = "rw"
```

Host bind sources, Docker/containerd sockets, and undeclared volumes are
forbidden. Do not share a named volume between the verifier and agent-reachable
services. Use sealed artifacts for verifier handoff.

### 9.5 Linux capabilities

The only authorable Linux capability is `SYS_PTRACE`:

```toml
[verifier]
capabilities = ["SYS_PTRACE"]
```

or on a service:

```toml
capabilities = ["SYS_PTRACE"]
```

Use it only for a trusted monitor whose need is tested, as in XFOIL. Privileged
containers, host devices, `SYS_ADMIN`, and arbitrary capabilities are rejected.

## 10. Captures, Artifacts, and Results

### 10.1 Ordered capture hooks

Captures run after the agent phase and before artifacts are frozen:

```toml
[[captures]]
name = "database-snapshot"
service = "database"
command = ["sh", "-lc", "pg_dump app > /tmp/db.sql.tmp && mv /tmp/db.sql.tmp /tmp/db.sql"]
timeout_sec = 60
atomic_destination = "/tmp/db.sql"
accepted_exit_codes = [0]
failure_policy = "agent"
```

Capture order is declaration order. A good cutover sequence commonly is:

1. finalize agent-visible work;
2. ask customer/workload services to emit results;
3. dump database or broker state;
4. record repository diff/status;
5. pause the agent graph;
6. collect sealed artifacts; and
7. run the verifier.

Use `failure_policy = "agent"` when a correct solution was responsible for
making the capture succeed. Use `infrastructure` when failure indicates a broken
environment and the rollout must be discarded.

### 10.2 Artifact kinds

```toml
[[artifacts]]
name = "repository"
kind = "tree"
source = "/workdir/repo"
destination = "repo"
service = "main"
required = true
max_bytes = 268435456
max_files = 5000
max_depth = 64
exclude = [".git", "target", "__pycache__"]
```

Supported kinds:

- `file`: one regular file;
- `tree`: a recursive directory;
- `path_set`: a stable set of files and trees;
- `binary`: exact bytes and executable mode; and
- `service`: an artifact explicitly collected from a sidecar/init service.

Every recursive artifact should set realistic byte, file, and depth limits.
Artifacts reject symlink escapes, special files, protected runtime paths, and
reserved manifest destinations.

For a software `WorkspaceArtifact("repo")`, an explicit service graph must
collect the main service's repository to destination `repo`.

### 10.3 Result contract

A verifier service requires explicit reward fields:

```toml
[result]
output_root = "/tmp/output/verifier"
reward_file = "grade.json"
reward_key = "score"
subscores_key = "subscores"
reports = ["behavioral", "performance"]
trace_file = "trace.json"
agent_fault_reward = 0.0
infrastructure_fault = "discard"
```

The verifier writes:

```json
{
  "score": 0.84,
  "subscores": {
    "behavioral": 1.0,
    "determinism": 1.0,
    "performance": 0.36
  }
}
```

Do not let candidate output select the result path, reward key, criterion
weights, or infrastructure disposition.

### 10.4 Reports and gates

`[[reports]]` declares bounded verifier diagnostics in JSON, CTRF, JUnit, text,
or a custom format. `[[gates]]` records structural, behavioral, performance,
determinism, static, or custom gates. The authored verifier remains responsible
for running its checks and producing declared reports; these sections make the
contract explicit and validate references.

## 11. MCP and Tool-Using Environments

Use MCP when the task needs a narrow, auditable tool API rather than unrestricted
shell access to a service.

```toml
[[mcp_servers]]
name = "browser-tools"
transport = "sse"
url = "http://browser-mcp:8080/sse"
service = "browser-mcp"
depends_on = ["browser-mcp"]
access = "agent"

[mcp_servers.readiness]
kind = "http"
url = "http://browser-mcp:8080/health"
accepted_statuses = [200]
timeout_sec = 30
interval_sec = 1
```

The runtime:

- resolves only declared service hosts;
- validates scheme, port, and path;
- bounds tool request and response data;
- waits for declared readiness;
- exposes tools only to the requested phase; and
- records calls through the audited proxy.

Native schema supports stdio and SSE, but Taiga outer-capsule export supports
audited SSE services only. Do not work around that restriction by starting an
untracked process or exposing a host URL.

## 12. Build the Grader on Shared APIs

### 12.1 Mandatory declarative shape

Software tasks register exactly one task:

```python
from grading.evaluation import (
    RubricCriterion,
    RubricEvaluation,
    RubricTask,
    TrustedJson,
    WorkspaceArtifact,
)


def evaluate(context) -> RubricEvaluation:
    ...


TASK = RubricTask(
    artifact=WorkspaceArtifact("repo", reject_native_payloads=True),
    fixtures={"cases": TrustedJson("hidden_cases.json", require_object=False)},
    criteria=(
        RubricCriterion("behavior", weight=0.7, required=True),
        RubricCriterion("determinism", weight=0.15, required=True),
        RubricCriterion("performance", weight=0.15),
    ),
    evaluate=evaluate,
)
```

Do not add an author-owned `compute_score()` alongside `TASK`.

### 12.2 Configure `WorkspaceArtifact` deliberately

Use shared controls instead of ad hoc cleanup:

```python
WorkspaceArtifact(
    "repo",
    max_files=5_000,
    max_total_bytes=256 * 1024 * 1024,
    max_file_bytes=16 * 1024 * 1024,
    clean_paths=(".git", "target", "__pycache__"),
    forbidden_names=("build.rs",),
    forbidden_suffixes=(".o", ".so", ".dll", ".exe"),
    forbidden_text_patterns=("std::process::Command", 'extern "C"'),
    text_suffixes=(".rs", ".toml", ".py"),
    reject_native_payloads=True,
)
```

Choose restrictions that defend the intended task without banning legitimate
solutions. Every restriction needs a test.

`WorkspaceArtifact`:

- copies the submitted tree into a grader-owned immutable master;
- rejects symlinks and special files;
- enforces file, byte, and depth limits;
- removes declared disposable paths;
- checks native magic and forbidden content;
- commits a digest; and
- creates a fresh candidate-owned clone for every candidate operation.

The last point matters: writes from one `context.run_candidate()` call do not
persist into another. Combine dependent build-and-test steps in one command, use
`run_candidate_suite()` for independent repetitions, or follow the XFOIL sealed
external-target pattern for a trusted monitor.

### 12.3 Execute candidate code only through the context

For one bounded operation:

```python
result = context.run_candidate(
    ["python3", "-I", "-B", "-m", "pytest", "-q"],
    cwd=".",
    env={"PYTHONHASHSEED": "0"},
    timeout_s=180,
    max_output_bytes=2 * 1024 * 1024,
)
```

For repeated hidden cases, use `CandidateCommandSpec` and
`context.run_candidate_suite()`. It centralizes:

- per-attempt timeouts;
- total suite bounds;
- environment sanitization;
- output limits;
- deterministic attempt metadata; and
- the candidate-fault boundary.

Use:

- `context.candidate_operation()` for candidate-controlled parsing;
- `context.trusted_operation()` for trusted fixture/reference processing;
- `context.number()`, `context.ratio()`, and `context.mean()` for finite numeric
  contracts;
- `context.fixture()` for hidden fixtures;
- `context.reject_candidate()` for invalid candidate behavior; and
- `context.grader_failure()` only for a broken trusted grader.

Use `context.run_solver()` only for trusted, root-owned reference tools. Never
pass candidate paths, candidate commands, or candidate-controlled environment
values to it.

### 12.4 Shared loaders for non-workspace data

When a task legitimately grades JSON, text, NPZ, HDF5, or a regular file, use
the shared artifact descriptors and helpers from `grading.evaluation` and
`grading.helpers`. Do not recreate path traversal, symlink, size, dtype,
non-finite, or pickle protections in task code.

If the shared library lacks a loader or execution primitive, add a generic,
tested implementation to the shared grader first. Do not solve the gap with a
task-local unsafe parser.

## 13. Design a Reward That Trains the Right Behavior

### 13.1 Start from observable outcomes

Good criteria measure:

- protocol behavior;
- state transitions;
- concurrency safety;
- crash/restart recovery;
- schema compatibility;
- deterministic replay;
- performance under declared units;
- resource bounds;
- security invariants; or
- UI interaction and accessibility behavior.

Weak criteria measure:

- a file exists;
- a symbol or string appears;
- a test count increased;
- the diff resembles the oracle;
- the implementation uses one expected library; or
- the candidate reports that it passed.

Static and structural checks are useful as gates, but should not dominate the
reward unless structure is itself the product requirement.

### 13.2 Use required gates

Mark criteria required when failure invalidates the whole solution:

- the project must build;
- the output protocol must parse;
- no required operation may be missing;
- no data corruption or security invariant may fail; and
- deterministic replay must hold when determinism is required.

Then grant partial credit within valid behavior. This prevents a fast but broken
implementation from collecting performance points.

### 13.3 Keep weights framework-owned

Define weights once in `RubricCriterion`. Return raw criterion subscores from
`evaluate()`. Do not recompute weighted totals in task code and do not let the
candidate supply a score or weight.

### 13.4 Recommended criterion families

A robust software task often combines:

- **structural validity**: builds, expected package shape, no forbidden payload;
- **core behavior**: required public/hidden operations;
- **edge behavior**: boundary values and failure modes;
- **state/recovery**: restart, persistence, migration, rollback;
- **concurrency/distribution**: hidden schedules, retries, duplicate delivery;
- **determinism**: repeated identical requests;
- **performance**: bounded, warmed, unit-normalized samples; and
- **security**: no delegation, introspection, or hidden-fixture access.

Not every task needs every family.

## 14. Build Hidden Behavioral Suites

### 14.1 Hidden tests vary, they do not surprise

Build public and hidden cases from one semantic model. Hidden cases should
exercise different representatives of documented equivalence classes.

For a recovery task, vary:

- record counts;
- torn tails;
- duplicate sequence numbers;
- commit markers;
- interrupted checkpoints;
- repeated recovery;
- concurrent append/recovery schedules; and
- large-log performance.

For a migration task, vary:

- operations;
- input sizes;
- ordering;
- state carried between requests;
- error paths;
- tolerances grounded in the legacy behavior; and
- repeated-run consistency.

For a service cutover, vary:

- health delays;
- retry timing;
- source and destination state;
- partial migration;
- customer traffic;
- idempotent re-entry; and
- rollback or failure-atomicity.

### 14.2 Generate expected values before candidate execution

Trusted expected responses must be produced before candidate code runs, then
kept under root control. Never allow candidate code to overwrite the reference,
fixture, tolerance, or report used to score it.

### 14.3 Keep tolerances meaningful

Use absolute or relative tolerances tied to domain error, not arbitrary score
smoothing. Public differential tooling should use the same full-credit
tolerances as hidden grading so it does not mislead the agent.

### 14.4 Test repeated runs

Run the same committed artifact multiple times with fixed seeds. A task is not
ready if:

- scores vary;
- service captures change nondeterministically;
- performance dominates due cold-cache noise;
- a background process survives between attempts; or
- test ordering changes the result.

## 15. Fault Semantics

Fault classification is part of reward correctness.

### 15.1 Candidate fault

Use `AgentFault` or `context.reject_candidate()` for:

- malformed output;
- build or test failure;
- timeout caused by candidate behavior;
- forbidden payload;
- protocol violation;
- missing required artifact; or
- a detected reward-hacking attempt.

The rollout is kept with the configured agent-fault reward, normally `0.0`.

### 15.2 Grader or infrastructure fault

Let `GraderFault` or `InfrastructureFault` propagate when:

- a hidden fixture is invalid;
- a trusted reference crashes;
- a required service cannot start for environment reasons;
- the grader contract is internally inconsistent;
- a trusted capture cannot run for infrastructure reasons; or
- the runtime cannot establish its security boundary.

These rollouts are discarded. Never catch a broad exception and convert it to
candidate score `0.0`; that poisons RL data by teaching the model from broken
environments.

## 16. Reward-Hacking Threat Model

Assume the candidate can:

- read every public file and prompt;
- inspect its own process environment and filesystem;
- emit arbitrary stdout/stderr and malformed files;
- create symlinks, FIFOs, sockets, or oversized trees before artifact commit;
- modify any agent-owned file;
- leave background processes;
- exploit test ordering or timing;
- hardcode public examples;
- copy a visible legacy executable;
- invoke subprocesses, FFI, or network clients if not prohibited;
- attempt to forge `reward.json` or reports; and
- exploit broad exception handling.

The candidate cannot be trusted merely because the prompt says not to cheat.

### 16.1 Mandatory defenses

- Use `WorkspaceArtifact` and shared artifact descriptors.
- Keep hidden fixtures under root-only `scorer/data/`.
- Run candidate code only via bounded shared APIs.
- Generate expected values before candidate execution.
- Keep score aggregation and weights in the framework.
- Make the verifier result path root-controlled.
- Quiesce and scrub agent-owned state before grading.
- Bound output, recursion, file counts, and runtime.
- Disable internet unless explicitly required.
- Test no-op and at least one realistic shortcut or attack.
- Commit the generated evaluation plan.

### 16.2 Prohibited grader patterns

Do not:

- use `subprocess`, `os.system`, `os.popen`, `exec`, `eval`, or dynamic import on
  candidate-controlled paths;
- call `pickle.load`, unsafe `torch.load`, or another code-bearing deserializer
  on candidate artifacts;
- read `/tmp/output` with raw `open()`, `Path.read_text()`, pandas, NumPy, or
  h5py inside the root grader;
- trust candidate stdout as a score;
- read a candidate-written report after executing more candidate code in the
  same mutable directory;
- return a score from `except Exception`;
- use wall-clock time or unseeded randomness in the scoring path;
- compare the candidate repository directly to the oracle diff; or
- expose verifier volumes, networks, or dependencies to the agent.

### 16.3 Advanced delegation defenses

If the task forbids delegating to a legacy binary or another process:

- remove or protect the public reference before hidden execution;
- forbid copied native payloads and known invocation patterns;
- build the candidate under UID 1000;
- seal the derived executable under root ownership;
- run it through a trusted monitor with the minimum required capability; and
- include an encoded/obfuscated delegation attack fixture.

Do not invent this machinery in every task. Reuse or generalize the XFOIL
monitor pattern and add shared tests.

## 17. Docker and Dependency Design

### 17.1 Users and permissions

Use:

```toml
[agent]
user = "agent"

[verifier]
user = "root"
env = []
```

The main service must be unprivileged. Make only the workspace and output paths
agent-writable. Keep `/mcp_server/data` and `/mcp_server/grader` root-owned with
directory mode `0700` and file mode `0600`.

Add an image-build probe proving UID 1000 cannot traverse private roots.

### 17.2 Dependency channels

- `environment/apt.txt`: agent/runtime OS dependencies.
- `environment/requirements.txt`: agent/runtime Python dependencies.
- `scorer/requirements.txt`: grader-only dependencies.
- vendored or lockfile-pinned language dependencies for offline builds.

Do not duplicate one dependency across public and private channels. Do not rely
on internet installation during an episode.

### 17.3 Build performance

Put stable dependency layers before task source. Use lockfiles and vendoring.
Do not force the agent or grader to rebuild unrelated toolchains repeatedly.
Set realistic build and runtime timeouts in `ResourceSpec`.

### 17.4 Provenance and licensing

For imported or migrated code, include:

- source repository URL;
- pinned upstream commit;
- archive checksum when vendored;
- original license;
- changes made for the task;
- attribution/NOTICE files; and
- any license boundary between framework code and task material.

Do not copy a benchmark task without preserving attribution. Do not use ML
license metadata fields for software tasks; document software licenses in the
task README and relevant files.

## 18. Oracle, Baselines, and Attack Fixtures

### 18.1 Oracle

`solution/solve.sh` must:

- derive the answer from disclosed source and public inputs;
- create the same artifact shape required from the agent;
- avoid reading hidden scorer data unless the contract explicitly requires a
  trusted reference input;
- run deterministically; and
- score exactly `1.0` within `score_epsilon`.

An oracle that writes a hardcoded answer without reading public derivation
inputs is invalid even if hidden grading accepts it.

### 18.2 No-op baseline

Create `baselines/noop.sh` or an equivalent naive baseline. It should produce a
validly shaped but non-solution artifact and score at or below:

```toml
[ground_truth]
max_trivial_score = 0.0
```

Also grade:

- an empty repository;
- unchanged starter;
- prompt examples;
- a public-case hardcode; and
- a partial implementation.

### 18.3 Attack fixtures

Choose attacks relevant to the task:

- candidate-written reward or report;
- mutation of a public worker;
- hidden-file path probing;
- symlink or special-file planting;
- copied legacy binary;
- subprocess or FFI delegation;
- service introspection;
- direct database snapshot fabrication;
- timing-dependent pass;
- background process persistence; or
- oversized artifact denial of service.

Each attack belongs in a deterministic regression test and should score `0.0`
or produce an explicit `AgentFault`.

## 19. Validation Workflow

### 19.1 Fast local checks

From the repository root:

```bash
uv sync

uv run lbx-rl-template check \
  --problem-dir problems/<task_id>

bash problems/<task_id>/tests/test.sh
```

`check` prints the friendly pre-submit summary and reward-hack advisories.
`validate` emits the full machine-readable validation result:

```bash
uv run lbx-rl-template validate \
  --problem-dir problems/<task_id>
```

### 19.2 Reference and ground truth

During iteration:

```bash
uv run lbx-rl-harness reference \
  --problem-dir problems/<task_id>
```

Before submission:

```bash
uv run lbx-rl-harness run \
  --runtime ground-truth \
  --problem-dir problems/<task_id>
```

Regenerate `scorer/evaluation.plan.json` through the harness or:

```bash
uv run python scripts/write_evaluation_plan.py problems/<task_id>
```

Never edit the plan by hand.

For in-container tasks, the validator may require a current committed
`.alignerr/build_proof.json` with ground-truth score `1.0` and trivial baseline.
Refresh it on the target architecture whenever `task.toml`, `solution/`,
`scorer/`, public data, baselines, or environment inputs change. Do not commit
unrelated local run directories or caches.

### 19.3 Trusted CI

Trusted mothership CI:

- validates the task and reward-hack contract;
- imports the fork under budgets;
- builds images in a trusted lane;
- creates capsule attestations for capability tasks;
- runs ground truth, baseline, and adversarial checks;
- publishes an author-facing grade report; and
- routes accepted tasks to Taiga.

Treat a green unit test as necessary but not sufficient. Review the trusted CI
report, criterion breakdown, baseline scores, runtime, and any discarded
rollouts.

### 19.4 Boreal difficulty calibration

After trusted CI is green, use the Boreal problem report to inspect all
configured attempts:

- confirm attempts reached and exercised the task rather than failing
  infrastructure;
- inspect per-criterion scores and trajectories;
- distinguish a common legitimate solution from one leaked shortcut;
- verify the aggregate is stable enough to interpret; and
- require the final **Boreal aggregate score to be `<= 0.4`**.

If the aggregate is above `0.4`, the task is not accepted. Improve the task and
rerun the complete trusted CI/Boreal cycle. Do not average in discarded
infrastructure failures or intentionally create agent-hostile noise to lower
the score.

## 20. Taiga and Harbor Export

### 20.1 Harbor

For a simple task:

```bash
uv run lbx-rl-template export-harbor \
  --problem-dir problems/<task_id> \
  --out /tmp/<task_id>-harbor
```

For a capability task, CI supplies a digest-pinned image when required:

```bash
uv run lbx-rl-template export-harbor \
  --problem-dir problems/<task_id> \
  --out /tmp/<task_id>-harbor \
  --image registry/task@sha256:<digest>
```

The exporter projects resources, Compose services, verifier environment,
captures, artifacts, and MCP settings into Harbor schema without changing the
authored source.

### 20.2 Taiga outer capsule

Capability tasks ship one outer image containing:

- all child service images;
- a digest manifest;
- Compose configuration;
- the task and grader;
- the nested service runtime; and
- the MCP bridge.

Child image build/pull is a trusted operation:

```bash
uv run lbx-rl-template export-capsule \
  --problem-dir problems/<task_id> \
  --out /tmp/<task_id>-capsule \
  --trusted-build
```

Authors should not use `--trusted-build` in an untrusted fork lane. Trusted CI
does it after validation.

After building and pushing the outer image:

```bash
uv run lbx-rl-template export-taiga \
  --problem-dir problems/<task_id> \
  --out /tmp/problems-metadata.json \
  --image registry/outer@sha256:<digest> \
  --outer-capsule
```

### 20.3 Current limitations

- Taiga nested-Docker capsules use CPU Firecracker tiers.
- Child GPU/TPU service resources are rejected.
- Taiga capsule MCP is audited SSE only.
- Runtime child images are prebuilt and loaded with `pull_policy: never`.
- External child images must be digest pinned.
- Host bind volumes, devices, privileged mode, and orchestration sockets are
  forbidden.
- `[runner]` checkpoint and episode controls are Taiga-specific; Harbor uses
  its own controls.

## 21. Troubleshooting

### Validation says the scorer never runs candidate code

Ensure the live `evaluate()` path calls `context.run_candidate()` or
`context.run_candidate_suite()`. A helper that is never invoked does not count.

### A build in one candidate call disappears in the next

This is expected. Every call receives a fresh disposable clone. Combine
dependent commands or use a sealed external target pattern.

### The verifier can see no service data

Do not share agent volumes with the verifier. Add an ordered capture and a
bounded artifact, then read the sealed artifact snapshot.

### A service waits forever

Replace sleeps with a real healthcheck/readiness probe. Check dependency
conditions and ensure every timeout is bounded.

### Harbor works but Taiga export fails

Check for:

- stdio MCP;
- child GPU/TPU resources;
- an unpinned image;
- a non-CPU outer resource tier;
- missing `--outer-capsule`; or
- a capsule image that is not digest pinned.

### Ground-truth proof is stale

Regenerate the evaluation plan with Python 3.13, rebuild on the target
architecture, rerun the ground-truth harness, and verify the no-op result. A
proof is tied to grading inputs and image identity.

### A no-op receives partial credit

Identify which criterion rewards shape instead of behavior. Make correctness a
required gate, recalibrate weights, and add unchanged-starter and reward-forgery
regressions.

## 22. Suggested Task Families

| Family | Strong hidden dimensions |
| --- | --- |
| Recovery and durability | torn state, replay, idempotence, scale, concurrency |
| Runtime migration | operation coverage, errors, state, compatibility |
| Build migration | clean/offline builds, workspace members, generated assets |
| Schema evolution | old/new readers, rollback, mixed versions, data migration |
| Distributed protocol | retries, duplication, partitions, ordering, restart |
| Database cutover | live writes, snapshots, consistency, rollback |
| Performance repair | varied workloads, warmup, correctness gate, resource use |
| Frontend repair | interactions, accessibility, layout, state transitions |
| Security hardening | exploit regression plus normal behavior |
| Tool/API integration | readiness, constrained calls, invalid input, audit trail |

## 23. Final Author Checklist

### Challenge and fairness

- [ ] The task tests a named engineering capability, not packaging friction.
- [ ] The public specification contains every required behavior.
- [ ] Public tests teach the interface without revealing the held-out answer.
- [ ] Hidden tests vary documented semantics rather than add requirements.
- [ ] The task needs multi-step reasoning and iteration.
- [ ] A human engineer can solve it with the disclosed tools and time budget.

### Repository and environment

- [ ] `task_type`, domain, and reward type are correct.
- [ ] The starter contains no oracle or hidden fixture.
- [ ] Dependencies are offline-capable and pinned.
- [ ] Agent and verifier users are separated.
- [ ] Private grader paths are root-only.
- [ ] Internet is disabled unless justified.
- [ ] Build and runtime timeouts are realistic.
- [ ] Upstream source and licenses are documented.

### Grader

- [ ] `TASK = RubricTask(...)` is the only grading entry.
- [ ] `WorkspaceArtifact` bounds and restrictions are task-appropriate.
- [ ] Candidate code runs only through shared bounded APIs.
- [ ] Expected values are generated before candidate execution.
- [ ] Raw candidate data never reaches trusted parsers unsafely.
- [ ] Criteria are behavioral, deterministic, and correctly weighted.
- [ ] Required gates block invalid solutions.
- [ ] Faults are classified as candidate vs infrastructure correctly.
- [ ] `evaluation.plan.json` matches `TASK`.

### Services and artifacts

- [ ] The service graph is necessary.
- [ ] Exactly one explicit main service exists when services are declared.
- [ ] Images are built from local validated stages or digest pinned.
- [ ] Readiness and dependency conditions are real.
- [ ] Volumes are named and never cross the verifier boundary.
- [ ] Captures are ordered, atomic, bounded, and correctly classified.
- [ ] Artifacts are service-qualified, sealed, and size-bounded.
- [ ] The verifier has no network and receives only sealed handoff data.
- [ ] `[result]` is explicit and root-controlled.

### Reward-hacking resistance

- [ ] Empty, unchanged-starter, and no-op attempts score at the floor.
- [ ] Public-case hardcoding does not pass hidden cases.
- [ ] Reward/report forgery does not affect scoring.
- [ ] Relevant symlink, subprocess, introspection, or delegation attacks fail.
- [ ] Repeated runs return the same score.
- [ ] Candidate crashes do not become grader failures, and grader failures do
  not become kept candidate zeros.

### Verification

- [ ] `lbx-rl-template check` passes.
- [ ] Host task tests pass.
- [ ] Ground truth scores `1.0`.
- [ ] Trivial baseline is at or below `max_trivial_score`.
- [ ] Target-platform/container checks pass when required.
- [ ] Harbor export validates.
- [ ] Capability tasks produce a trusted Taiga capsule and metadata.
- [ ] Every required trusted CI check is green.
- [ ] Bugbot has no unresolved findings.
- [ ] The Boreal problem aggregate score is `<= 0.4`.

When in doubt, add a reusable primitive or validator to the shared framework
instead of embedding another task-specific security or grading shim.
