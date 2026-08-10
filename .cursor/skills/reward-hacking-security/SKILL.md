---
name: reward-hacking-security
description: Enforces Alignerr grader security and reward-hacking defenses. Use when handling AgentFault, submission loaders, symlinks, candidate code, baselines, attack fixtures, policy isolation, or validate/lint-reward-hacks findings.
---

# Reward-Hacking and Grader Security

Assume candidates can create arbitrary files, stdout, symlinks, special files,
background processes, malformed payloads, copied binaries, and forged reports.

## Mandatory boundary

- Load candidate artifacts through shared descriptors/loaders.
- Run candidate code only in UID-dropped shared workers.
- Generate trusted expected values before candidate execution.
- Keep hidden fixtures and grader code root-only.
- Bound files, bytes, depth, output, attempts, and time.
- Keep score paths, weights, and fault policy framework-owned.
- Disable internet unless the task contract requires it.

## Never do this in a root grader

- raw `open`, `Path.read_text`, pandas/NumPy/HDF5 parsing on agent paths;
- pickle or code-bearing deserialization;
- dynamic import/exec/eval of submitted code;
- direct `subprocess` with candidate paths/commands;
- parsing candidate stdout as reward;
- broad `except Exception: return 0.0`; or
- reading mutable candidate data after running more candidate code.

Use `WorkspaceArtifact`, other evaluation descriptors, sanctioned
`grading.helpers`, `context.run_candidate`, `context.policy`, or
`load_submitted_policy`.

## Fault classification

- Candidate malformed input, failure, timeout, policy violation ->
  `AgentFault` / `context.reject_candidate` (kept).
- Broken fixture/reference/grader/runtime -> propagate grader or infrastructure
  fault (discarded).

Never convert environment failure into candidate score zero.

## Required regressions

Every task needs empty/no-op/unchanged-starter checks. Add attacks appropriate
to the task:

- reward/report forgery;
- hidden path or symlink access;
- worker mutation;
- subprocess/FFI/reference delegation;
- copied native payload;
- service introspection/capture fabrication;
- background process persistence;
- oversized/deep artifact; and
- timing/nondeterminism.

## Validate

```bash
uv run lbx-rl-template lint-reward-hacks \
  --problem-dir problems/<task_id>
uv run lbx-rl-template validate \
  --problem-dir problems/<task_id>
```

Fix blocking `agent_fault`, `grader_sandbox`, determinism, private-layout, and
software-contract findings at the root cause.

## References

- `docs/REWARD_HACKING.md`
- `docs/POLICY_ISOLATION.md`
- `.cursor/rules/grader-contract.mdc`
- `.cursor/rules/reward-hacking.mdc`
