# New Software Engineering Task Design and Delivery Workflow

This guide is for authors creating **brand-new** long-horizon
software-engineering RL tasks with the ISO template and shared framework.

Use it as the idea-to-acceptance workflow. The companion
[author handbook](frontier_style_software_engineering_tasks.md) contains the
full implementation and security patterns, while the
[technical reference](../../docs/SOFTWARE_ENGINEERING_FRAMEWORK.md) documents
schema and runtime internals.

## 1. Definition of Done

A task is accepted only when all of the following are true:

1. The challenge tests a clear, valuable engineering capability.
2. The public specification is complete and fair.
3. The oracle scores `1.0`.
4. No-op, unchanged-starter, and relevant attack baselines score at the floor.
5. Repeated grading is deterministic.
6. All required **trusted CI checks are green**.
7. The problem's **Boreal aggregate score is `<= 0.4`**.

The Boreal aggregate is the problem-level aggregate across configured
evaluation attempts. It is not the oracle score, not one cherry-picked rollout,
and not a number authors should lower through flaky infrastructure. A score
above `0.4` means the task is too easy for the target cohort and must be
strengthened without introducing hidden requirements.

## 2. Start With a Capability, Not a Repository

Write one sentence in this format:

> A successful agent can `<engineering capability>` while preserving
> `<invariants>` under `<meaningful hidden variation>`.

Examples:

- A successful agent can repair write-ahead-log recovery while preserving
  committed ordering under torn tails, replay, scale, and hidden schedules.
- A successful agent can implement a versioned protocol feature while
  preserving backward compatibility across stateful request sequences.
- A successful agent can execute a live database cutover while preserving
  customer-visible consistency under concurrent writes and restart.
- A successful agent can harden an input parser while preserving valid behavior
  and blocking a family of exploit payloads.

Reject ideas that reduce to:

- one known function implementation;
- copying a public patch;
- installing a missing package;
- satisfying a single static assertion;
- guessing hidden constants;
- running a provided migration command;
- waiting for an external service; or
- producing prose judged by an LLM.

## 3. Write the Task Design Brief

Before creating files, write an internal design brief with these sections.

### 3.1 Capability

- What engineering skill does the task measure?
- Why is it useful for training?
- Why should it require multi-step investigation and iteration?
- Which existing example is the nearest framework pattern?

### 3.2 Starting state

- What repository/system does the agent receive?
- What behavior is currently broken or missing?
- Which components are relevant?
- What public tests already exist?
- Which tools are installed?

### 3.3 Final state

- What artifact must the agent leave?
- What externally observable behavior proves success?
- Which invariants must remain true?
- What is explicitly out of scope?

### 3.4 Public contract

- Prompt requirements
- Public protocol/schema
- Public commands
- Representative cases
- Performance units
- Compatibility/security expectations
- Allowed dependencies and network

### 3.5 Hidden semantic matrix

For each hidden family, record:

- public requirement it validates;
- varied dimensions;
- expected behavior;
- candidate-fault conditions;
- trusted/infrastructure-fault conditions;
- timeout/output bounds; and
- criterion/gate mapping.

### 3.6 Shortcut analysis

List the cheapest plausible incorrect solutions:

- unchanged starter;
- hardcoded public examples;
- copied reference binary;
- subprocess delegation;
- candidate-written reward/report;
- hidden-path probing;
- timing exploit;
- service-state fabrication;
- partial implementation that earns unrelated points; and
- oversized/malformed artifact.

Every high-risk shortcut needs a framework defense or regression fixture.

## 4. Qualify the Idea

The task should pass this review before implementation.

### 4.1 Fairness

- Every hidden behavior follows from the public contract.
- Required tools and commands are documented.
- The agent can reproduce the intended workflow locally.
- The environment is offline-capable.
- The task does not require privileged knowledge.

### 4.2 Horizon

The task requires several of:

- architecture discovery;
- cross-file reasoning;
- failure diagnosis;
- state-machine reasoning;
- concurrency reasoning;
- compatibility work;
- test-driven iteration;
- performance analysis;
- service coordination; and
- security/robustness tradeoffs.

### 4.3 Grading feasibility

- Correct behavior is observable.
- Expected values can be generated deterministically.
- The hidden suite fits resource/time budgets.
- A trusted oracle can be implemented.
- No LLM judge is needed.
- Candidate and infrastructure failures can be separated.

### 4.4 Novelty

Compare with the task catalog and canonical examples. The new problem should add
a distinct capability, environment, or interaction pattern rather than a
renamed copy with different constants.

## 5. Choose the Execution Shape

### 5.1 Secure single image

Choose for most repository debugging, feature, compatibility, build,
performance, frontend, and security tasks.

Final output:

```toml
[[outputs]]
path = "/tmp/output/repo"
required = true
```

Grader:

```python
WorkspaceArtifact("repo", reject_native_payloads=True)
```

### 5.2 Implicit-main capability

Choose when one image is enough but the task needs typed workspace/checkpoint or
artifact lifecycle. Remember that capability export means a Taiga outer
capsule.

### 5.3 Explicit services

Choose only when live topology is part of the engineering problem. Document why
each service is necessary and what would be lost by collapsing it.

## 6. Select Taxonomy and Identity

Choose a stable task ID and narrow domain:

```toml
[task]
name = "labelbox/<task-id>"
description = "<one-sentence behavior-focused summary>"

[difficulty]
task_type = "software_engineering"
domain = "<supported domain>"
reward_type = "multi_deterministic_rubrics"
```

Use the authoritative domain list from
[`SOFTWARE_TRANSFORMATION_TASKS.md`](../../docs/SOFTWARE_TRANSFORMATION_TASKS.md).

## 7. Scaffold

```bash
uv sync

uv run lbx-rl-template create \
  --name labelbox/<task-id> \
  --template software-engineering \
  --out problems
```

Do not delete starter hardening while prototyping. Keep:

- UID separation;
- root-only scorer data;
- `WorkspaceArtifact`;
- shared candidate execution;
- no-op baseline;
- generated evaluation plan; and
- host task checks.

## 8. Build the Public Environment First

Before hidden grading:

1. Create the starter repository.
2. Ensure it builds/runs enough to expose the intended defect or missing work.
3. Add public documentation.
4. Add public smoke tests.
5. Add public adapters/protocol tools.
6. Verify all commands offline.
7. Measure clean/warm build time and storage.
8. Run the environment as UID 1000.

The public environment should be useful even without the grader.

## 9. Write `instruction.md`

The prompt should explain:

- current state;
- requested final behavior;
- repository path;
- public commands;
- protocols;
- compatibility and security invariants;
- performance expectations in named units;
- final output; and
- non-goals.

Ask an engineer unfamiliar with the grader to read it. If they need hidden
knowledge to predict success, revise it.

## 10. Build the Hidden Suite

### 10.1 Use semantic families

Do not create a bag of examples. Define families such as:

- basic behavior;
- edge/error behavior;
- stateful sequence;
- restart/recovery;
- compatibility;
- concurrency/distribution;
- determinism;
- performance; and
- security.

### 10.2 Tie every case to a public requirement

Maintain an internal mapping from hidden family to prompt/public documentation.
Remove cases with no public justification.

### 10.3 Use trusted fixtures and drivers

Put hidden inputs under `scorer/data/`. Use `TrustedJson` or a root-owned driver.
Keep fixtures immutable and bound their size.

### 10.4 Generate expected values before candidate execution

Run trusted references first and seal results. Candidate code must not influence
expected outputs, tolerances, weights, or case selection.

## 11. Implement the Declarative Grader

Use `TASK = RubricTask(...)`.

Recommended implementation order:

1. Configure `WorkspaceArtifact`.
2. Declare trusted fixtures.
3. Declare criteria and required gates.
4. Implement trusted reference generation.
5. Implement one bounded candidate operation.
6. Parse candidate output under `candidate_operation`.
7. Compute raw criterion scores with shared numeric helpers.
8. Add repeated/determinism attempts.
9. Add fault mapping.
10. Generate the evaluation plan.

Do not add task-local process, file-loading, aggregation, or fault-handling code
when the shared grader already provides it.

## 12. Calibrate Reward Structure

### 12.1 Required gates

Use required criteria for:

- build/protocol validity;
- mandatory operation coverage;
- data integrity;
- security invariants;
- deterministic behavior when required; and
- service/cutover completion.

### 12.2 Partial credit

Partial credit should reflect meaningful engineering progress after required
validity is met.

### 12.3 Baselines

Run:

- empty output;
- unchanged starter;
- no-op shaped output;
- public-case hardcode;
- obvious partial solution; and
- relevant attacks.

The floor must be stable before running agents.

## 13. Add Reward-Hacking Regressions

At minimum test:

- reward/report forgery;
- symlink/special-file artifact;
- hidden path access;
- candidate crash;
- timeout;
- oversized output;
- repeated execution; and
- task-specific shortcut.

For service tasks also test:

- agent/verifier volume crossing;
- verifier network access;
- service introspection;
- capture fabrication; and
- dependency/readiness misuse.

## 14. Add Services Only When Needed

For each service:

1. Assign a role.
2. Pin image/build stage and platform.
3. Set user.
4. Set resource/network policy.
5. Add healthcheck.
6. Add explicit dependencies.
7. Add only named volumes.
8. Decide what state is captured.
9. Decide what artifact is sealed.
10. Prove verifier isolation.

For MCP:

1. Define a narrow schema.
2. Add SSE sidecar.
3. Add readiness.
4. Restrict access.
5. Test URL/argument abuse.

## 15. Implement the Oracle

The oracle must derive the solution from public task inputs, produce the exact
candidate artifact shape, and score `1.0`.

Run:

```bash
uv run lbx-rl-harness reference \
  --problem-dir problems/<task-id>
```

Inspect every criterion, not only headline score.

## 16. Run Local Preflight

```bash
uv run lbx-rl-template check \
  --problem-dir problems/<task-id>

uv run lbx-rl-template validate \
  --problem-dir problems/<task-id>

bash problems/<task-id>/tests/test.sh

uv run lbx-rl-harness run \
  --runtime ground-truth \
  --problem-dir problems/<task-id>
```

Resolve every blocking stage. Do not suppress a finding to make preflight green.

## 17. Verify Export Behavior

Simple/Harbor:

```bash
uv run lbx-rl-template export-harbor \
  --problem-dir problems/<task-id> \
  --out /tmp/<task-id>-harbor
```

Capability tasks are built in a trusted lane:

```bash
uv run lbx-rl-template export-capsule \
  --problem-dir problems/<task-id> \
  --out /tmp/<task-id>-capsule \
  --trusted-build
```

Verify parity for workspace, users, resources, services, captures, artifacts,
MCP, result, and faults.

## 18. Open the Task PR and Wait for Trusted CI

Local checks do not replace trusted CI.

The author must wait until **all required trusted CI checks are green**,
including the checks applicable to the task:

- task validation;
- trusted image/capsule build;
- ground truth;
- no-op/trivial baseline;
- grader QA;
- reward-hacking/security checks;
- Harbor/Taiga export;
- capsule attestation/conformance; and
- task-specific tests.

If a required check is missing, pending indefinitely, neutral because it did not
run, or bypassed, the task is not complete.

## 19. Evaluate in Boreal

After trusted CI is green, run the configured Boreal evaluation and inspect:

- attempt completion;
- discarded vs kept rollouts;
- aggregate score;
- per-criterion distribution;
- common solver strategies;
- trajectories and tool use;
- timeouts/build failures;
- evidence of shortcutting; and
- evidence of unfair hidden requirements.

### 19.1 Required difficulty threshold

The final problem-level **Boreal aggregate score must be `<= 0.4`**.

If it is above `0.4`:

1. Identify which criteria are commonly solved.
2. Determine whether the task lacks depth or has a shortcut.
3. Strengthen semantic variation or interacting invariants.
4. Keep all requirements public.
5. Update oracle, baselines, attacks, and tests.
6. Rerun local preflight.
7. Rerun all trusted CI.
8. Rerun Boreal.

Do not lower the score by:

- hiding requirements;
- removing useful public feedback;
- reducing fair timeouts;
- adding nondeterminism;
- causing infrastructure failures;
- increasing build friction;
- using arbitrary thresholds; or
- scoring style instead of behavior.

### 19.2 Interpreting `<= 0.4`

Confirm the aggregate represents valid attempts. A low aggregate caused by
discarded infrastructure, a broken grader, impossible instructions, or a
systematic environment failure is not acceptable.

The target is: fair, solvable, deterministic, trusted-CI-green, and difficult
enough that the evaluated cohort achieves no more than `0.4` aggregate.

## 20. Iterate From Evidence

Use Boreal trajectories to improve:

- prompt clarity;
- public tests;
- hidden semantic coverage;
- criterion calibration;
- runtime budget;
- service readiness;
- shared framework utilities; and
- attack defenses.

Every task revision invalidates prior acceptance. Rerun trusted CI and Boreal
after a material change to prompt, environment, grader, oracle, hidden fixtures,
or reward weights.

## 21. Final Acceptance Checklist

### Product

- [ ] Capability statement is clear.
- [ ] Task is novel and useful.
- [ ] Public contract is complete.
- [ ] Hidden suite maps to public requirements.
- [ ] Long horizon comes from engineering work.

### Environment

- [ ] Offline and reproducible.
- [ ] Dependencies pinned.
- [ ] UID separation correct.
- [ ] Private paths root-only.
- [ ] Resources/timeouts measured.
- [ ] Service graph is necessary and healthy.

### Grading

- [ ] Declarative `RubricTask`.
- [ ] Secure `WorkspaceArtifact`.
- [ ] Shared candidate execution APIs only.
- [ ] Required correctness gates.
- [ ] Deterministic repeated grading.
- [ ] Correct fault classification.
- [ ] Current evaluation plan.

### Evidence

- [ ] Oracle `1.0`.
- [ ] No-op/unchanged starter at floor.
- [ ] Relevant attacks at floor.
- [ ] Current build/ground-truth proof when required.
- [ ] Harbor/Taiga parity verified.

### Required acceptance bar

- [ ] **All required trusted CI checks are green.**
- [ ] **Boreal aggregate score is `<= 0.4`.**

Do not mark the problem complete until both acceptance-bar boxes are checked.
