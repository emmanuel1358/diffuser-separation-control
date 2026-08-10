---
name: software-engineering-tasks
description: Authors brand-new long-horizon software_engineering repository tasks with WorkspaceArtifact, RubricTask, behavioral hidden suites, secure candidate execution, and Boreal calibration. Use for repo debugging, modernization, concurrency, features, performance, frontend, database, or security tasks.
---

# Software Engineering Tasks

Read first:

- `project_guidelines/software_engineering/new_task_design_workflow.md`
- `project_guidelines/software_engineering/frontier_style_software_engineering_tasks.md`
- `docs/SOFTWARE_TRANSFORMATION_TASKS.md`
- `docs/SOFTWARE_ENGINEERING_FRAMEWORK.md`

## Acceptance bar

A software problem is complete only when:

- every required trusted CI check is green;
- oracle score is `1.0`;
- no-op/unchanged-starter/attack baselines stay at the floor; and
- Boreal aggregate score is `<= 0.4` across valid configured attempts.

Do not lower Boreal through hidden requirements, flaky tests, unfair timeouts,
or packaging friction.

## Start a new task

```bash
uv run lbx-rl-template create \
  --name labelbox/<task_id> \
  --template software-engineering \
  --out problems
```

```toml
[difficulty]
task_type = "software_engineering"
domain = "<narrowest supported software domain>"
reward_type = "multi_deterministic_rubrics"

[environment]
allow_internet = false

[runner]
attempts = 1
turn_limit = 1430
max_ctx = 1000000
context_mode = "autocompact"
```

Default context mode is Autocompact. Memory (`enable_memory`) requires Labelbox
consult first. Mode table: `docs/AUTHORING.md` / `docs/SOFTWARE_ENGINEERING_FRAMEWORK.md`.

Use the simple single-image `/tmp/output/repo` shape unless live services are
essential. If services/captures/MCP are needed, also read
`service-capsule-tasks`.

## Strong task design

- Define one valuable engineering capability and preserved invariants.
- Fully specify every hidden behavior publicly.
- Make the horizon come from diagnosis, cross-component reasoning, state,
  concurrency, compatibility, iteration, or tradeoffs.
- Provide usable public build/tests/protocol tooling.
- Vary hidden semantics, schedules, scale, errors, and seeds.
- Avoid golden patches, hidden requirements, and static-only rewards.

Canonical patterns:

- `examples/wal-recovery-ordering/`
- `examples/xfoil-rust-port/`

## Grader

```python
TASK = RubricTask(
    artifact=WorkspaceArtifact(
        "repo",
        reject_native_payloads=True,
        ...
    ),
    fixtures={...},
    criteria=(...),
    evaluate=evaluate,
)
```

- No author `compute_score()`.
- Candidate builds/tests use `context.run_candidate` or
  `context.run_candidate_suite`.
- Each candidate call gets a fresh disposable clone; dependent build/run steps
  belong in one call unless using a tested sealed-output pattern.
- Trusted references use `run_solver` and run before candidate code.
- Use required behavior/security gates before optional performance credit.
- Commit generated `scorer/evaluation.plan.json`; never hand-edit.

Read `deterministic-grading`, `rubric-design`, and
`reward-hacking-security`.

## Evidence

Add deterministic tests for:

- hidden behavioral families;
- repeated-run determinism;
- performance when relevant;
- candidate crash/timeout/malformed output;
- unchanged starter/no-op;
- public-case hardcode;
- reward/report forgery; and
- task-specific shortcut or delegation.

## Verify

```bash
uv run lbx-rl-harness reference --problem-dir problems/<task_id>
uv run lbx-rl-template check --problem-dir problems/<task_id>
uv run lbx-rl-harness run --runtime ground-truth --problem-dir problems/<task_id>
```

After trusted CI is green, inspect Boreal attempts/criteria/trajectories and
iterate until the valid problem aggregate is `<= 0.4`.

## References

- `examples/README.md`
- `docs/SOFTWARE_ENGINEERING_FRAMEWORK.md`
- `docs/SOFTWARE_TRANSFORMATION_TASKS.md`
- `docs/REWARD_HACKING.md`
