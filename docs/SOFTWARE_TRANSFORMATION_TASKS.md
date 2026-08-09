# Software Transformation Tasks

Software transformation tasks ask an agent to change a visible repository while
preserving or extending behavior across languages, runtimes, frameworks,
schemas, build systems, protocols, or deployment models. They use:

```toml
[difficulty]
task_type = "software_engineering"
domain = "legacy_modernization"
reward_type = "multi_deterministic_rubrics"
```

Use the declarative `RubricTask` stack described in
[`RUBRIC_EVALUATION.md`](RUBRIC_EVALUATION.md). Do not add an author-owned
`compute_score()` or a second calibration curve.

For the complete author workflow, task-quality criteria, reward-hacking threat
model, service-graph patterns, and pre-submit checklist, read
[`project_guidelines/software_engineering/frontier_style_software_engineering_tasks.md`](../project_guidelines/software_engineering/frontier_style_software_engineering_tasks.md).
For schema/export/runtime internals, read
[`SOFTWARE_ENGINEERING_FRAMEWORK.md`](SOFTWARE_ENGINEERING_FRAMEWORK.md).
For the idea-to-acceptance workflow for a new task, read
[`project_guidelines/software_engineering/new_task_design_workflow.md`](../project_guidelines/software_engineering/new_task_design_workflow.md).

Author acceptance requires every applicable trusted CI check to be green and
the problem's final Boreal aggregate score to be `<= 0.4`. The oracle must still
score `1.0`, and trivial/attack baselines must remain at the floor.

Canonical examples:

- [`wal-recovery-ordering`](https://github.com/Alignerr-Code-Labeling/lbx-rl-tasks-iso-template/tree/main/examples/wal-recovery-ordering)
  for focused concurrency/reliability debugging;
- [`xfoil-rust-port`](https://github.com/Alignerr-Code-Labeling/lbx-rl-tasks-iso-template/tree/main/examples/xfoil-rust-port)
  for long-horizon legacy modernization;
- [`frontier-service-cutover`](https://github.com/Alignerr-Code-Labeling/lbx-rl-tasks-iso-template/tree/main/examples/frontier-service-cutover)
  for services, captures, artifacts, and an isolated verifier; and
- [`frontier-mcp-workspace`](https://github.com/Alignerr-Code-Labeling/lbx-rl-tasks-iso-template/tree/main/examples/frontier-mcp-workspace)
  for an audited MCP/SSE sidecar.

## Supported domains

Choose the narrowest applicable domain from the authoritative enum in
`alignerr_plugin.task_metadata`:

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

## Submission contract

Give the agent the complete editable source, relevant public documentation, and
public behavioral cases. The final submission is a repository beneath
`/tmp/output`, declared in `task.toml`:

```toml
[[outputs]]
path = "/tmp/output/repo"
required = true
description = "Completed source repository."
```

Public tooling should make the intended workflow easy to reproduce, for
example:

```text
transform-run reference public-cases.jsonl
transform-run candidate public-cases.jsonl
transform-run diff public-cases.jsonl
```

Hidden grading may change cases, parameters, schedules, fault injection, and
seeds, but not the stated meaning of correctness.

Prefer a versioned structured boundary when the application permits it:

```json
{"protocol":"transform/v1","case_id":"case-1","operation":"analyze","input":{}}
{"protocol":"transform/v1","case_id":"case-1","status":"ok","observations":{},"events":[]}
```

Trusted adapters may translate that envelope to a CLI, HTTP service, batch
files, a database, a thread schedule, or a fault-injection driver. Candidate
stdout is protocol data only; never relay it to the grader's score output.

## Secure workspace descriptor

Declare the submitted repository with `WorkspaceArtifact`:

```python
from grading.evaluation import (
    RubricCriterion,
    RubricTask,
    TrustedJson,
    WorkspaceArtifact,
)


TASK = RubricTask(
    artifact=WorkspaceArtifact(
        "repo",
        clean_paths=("target",),
        forbidden_names=("build.rs",),
        forbidden_suffixes=(".o", ".so", ".dll", ".exe"),
        forbidden_text_patterns=("std::process::Command", 'extern "C"'),
        text_suffixes=(".rs", ".toml"),
    ),
    fixtures={"cases": TrustedJson("hidden-cases.json", require_object=False)},
    criteria=(RubricCriterion("core_behavior", weight=1.0),),
    evaluate=evaluate,
)
```

The framework:

- removes only the declared top-level cache paths;
- performs cleanup before committing the replay digest;
- rejects undeclared siblings outside the repository by default;
- traverses directories and reads files through pinned, no-follow descriptors;
- rejects symlinks, FIFOs, sockets, devices, and other special entries;
- bounds file count, individual bytes, total bytes, entries, and depth;
- rejects common native executable, object/archive, and WebAssembly payload
  magic unless `reject_native_payloads=False` is explicitly reviewed;
- applies task-declared forbidden names, suffixes, and exact source patterns;
- copies validated bytes into an immutable grader-owned master and commits it;
- returns a typed `SubmittedWorkspace` whose `path`/`snapshot_path` is the
  committed master and whose `original_path` is informational only.

Keep `allow_extra_workspace_entries=False` unless the scorer has a reviewed need
for additional top-level outputs. `clean_paths` is for reproducible build caches,
not for hiding undeclared submission material.

## Building and running candidate code

Run all agent-controlled build tools and executables through
`context.run_candidate(...)`:

```python
def evaluate(context):
    cases = context.fixture("cases")

    reference = context.run_solver(
        ["reference-driver", "/mcp_server/data/hidden-cases.json"],
        timeout_s=120,
    )
    if not reference.ok:
        context.grader_failure("trusted reference failed")

    candidate = context.run_candidate(
        ["cargo", "run", "--release", "--offline"],
        stdin_bytes=encode_cases(cases),
        env={"PATH": "/usr/bin:/bin", "HOME": "/tmp"},
        timeout_s=1200,
    )
    if candidate.returncode:
        context.reject_candidate("candidate build or execution failed")

    return evaluate_capabilities(
        context,
        cases,
        reference.output,
        candidate.stdout,
    )
```

The shared candidate runner uses the non-root identity configured by
`RUBRIC_AGENT_USER` or the paired `RUBRIC_AGENT_UID`/`RUBRIC_AGENT_GID`
variables. It fails closed unless the verifier can establish that separation,
copies the committed master into a fresh disposable worktree for every call,
chowns only that clone, enters it through a pinned no-follow directory
descriptor, and securely removes it afterward. It uses a sanitized environment
unless one is explicitly supplied, bounds stdout, discards stderr at the
kernel, and kills the full process group on completion, timeout, or output
flood. Candidate limit failures become kept `AgentFault` zeros. Filesystem
changes do not carry between calls, so build and execution must happen in one
command when execution depends on build outputs. Do not call `subprocess`
directly from scorer code.

Generate all trusted reference results before candidate execution. Keep private
fixtures, reference source, and answer keys under root-only paths such as
`/mcp_server/data`; the dropped candidate process must not be able to read them.

## Reward design

Compilation and startup are gates, not substitutes for behavior. Use independent
hidden suites for the properties that matter, such as:

- nominal and malformed-input behavior;
- exact file, byte, schema, or protocol compatibility;
- stateful histories and migration/restart behavior;
- numerical and resource boundaries;
- concurrency schedules, retries, partitions, and injected faults;
- accessibility and interaction behavior for frontend work;
- query plans, transactional semantics, and rollback for data systems;
- metamorphic invariants that remain valid across hidden cases.

The checked-in reference must score exactly `1.0`. No-op, hard-coded, copied
binary, and delegation baselines must score zero or remain below the task's
reviewed trivial-score ceiling. Encode architectural or resource requirements as
enforceable artifact/process checks rather than subjective rubric claims.

Harness reference and ground-truth runs generate
`scorer/evaluation.plan.json`. Commit the generated plan and never edit it by
hand; trusted CI reseals it from `TASK`.

## Licensing

`software_engineering` tasks may omit the dataset-license metadata required for
`ml` tasks, but the task README must document upstream source licensing and any
redistribution or derivative-work obligations. A language or framework migration
does not remove the original license.
