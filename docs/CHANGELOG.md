# Task Author Changelog

Task-author-visible changes to the shared grader, runtime, validators, and
authoring contracts are recorded here. Entries use ISO 8601 timestamps with a
UTC offset and are prepended in reverse chronological order (latest first).

## 2026-08-12T16:28:59-07:00 — Harden shared grader contracts after Taiga QA

### Action required

- Rebuild task images to receive runtime security revision `2026-08-12.1`.
  Setup now seals `/task/task.toml` and any accidentally shipped `/solution`
  tree from the agent, and output-directory restoration no longer relies on
  gVisor honoring `O_NOFOLLOW`.
- `PolicyEvaluationTask` now serializes
  `policy-evaluation-task.v4`, uses protocol `paired-policy-challenge.v2`, and
  derives full-width nonce-bound scenario seeds. Refresh
  `scorer/evaluation.plan.json`, rerun the reference, and recalibrate migrated
  policy tasks.
- Newly generated continuous locks use schema `3.2`. Schemas `3.1` and `3.0`
  remain readable. Opt into
  `GeneratedCalibration(quality_floor_mode="effective_no_info")` and regenerate
  the lock when grade-time credit should begin above the measured
  no-information ceiling.
- Env/hybrid tasks must add `scorer/data/env_config.json` with explicit
  `allowed_env_kwargs` and `require_public_methods_allowlist=true`, and their
  env class must declare `_env_public_methods`. Exposed methods may not accept
  unrestricted `**kwargs`.

### Policy and runtime hardening

- `helpers.run_policy(...)` and `RubricContext.policy(...)` now accept
  `factory_name`. Compatibility auto-detection supports module-level `act`,
  then `load_policy()`, then `Policy()`, so prompt-compliant factory-only
  submissions no longer fail grading.
- Missing required methods are typed policy agent faults while preserving
  `hasattr` for optional hooks. `run_seeds(...)` forwards per-call, first-call,
  and cumulative worker deadlines.
- Unsupported policy return values fail serialization immediately instead of
  cold-importing large frameworks until the call deadline expires.
- Pre-grade quiescence tracks `(pid,starttime)`, waits for already-killed
  processes to exit, and distinguishes a true respawner from a process still
  in kernel teardown.

### Hidden environments and continuous scoring

- The env server caps global and per-connection live instances. Strict env
  configs fail closed when `_env_public_methods` is absent.
- Sized `PrivateTableChallenge` now defaults to `stable_subset`; use explicit
  `selection_policy="artifact_digest"` only while replaying a legacy `3.0`
  lock. Full-bank selection remains the default when `sample_size` is omitted.
- `CsvRows(..., join_key="id")` aligns candidate rows to trusted truth through
  the collision-safe shared join. Positional behavior remains available when
  `join_key` is omitted.
- Queryable-model prompts must disclose load, prediction, row-count, and reply
  limits. The ML starter includes those limits and the stable-subset/effective-
  floor defaults.

## 2026-08-11T13:22:00-07:00 — Apply tmux hardening to Claude Code

### Changed

- The Claude Code harness runtime now routes its host-side tmux MCP tool through
  the shared hardened launcher. Detached commands run as uid/gid `1000:1000`
  with the inherited per-process address-space cap and BLAS/OpenMP thread
  defaults, matching the DeepAgents/local tmux path.
- No task image, calibration lock, or evaluation-plan refresh is required for
  this harness-only fix. Update or re-sync the harness checkout before the next
  local Claude Code run.

## 2026-08-11T12:10:00-07:00 — Close policy free-veto paths

### Action required

- Rebuild task images to receive submitted-policy reply validation, cumulative
  compute budgets, sealed-trace fallback, and local tmux isolation. Updated
  grades report shared runtime security revision `2026-08-11.1`.
- `PolicyEvaluationTask` now serializes schema
  `policy-evaluation-task.v3` and declares `total_timeout_s` (default: `3600`).
  Refresh `scorer/evaluation.plan.json` and any plan-bound calibration/evidence
  before republishing a policy task. Set both `call_timeout_s` and
  `total_timeout_s` from measured reference behavior, disclose them in the
  prompt, and leave at least 20% of `grading_timeout_seconds` for trusted
  controls, trace writing, and result publication.
- Sealed policy graders must let `AgentFault` propagate. Remove patterns such as
  `except AgentFault: return 0.0`; the runtime produces the kept zero and
  required replay trace.
- Re-run reference, validation, and ground-truth checks after migration:

  ```bash
  uv run lbx-rl-harness reference --problem-dir problems/<task_id>
  uv run lbx-rl-template check --problem-dir problems/<task_id>
  uv run lbx-rl-harness run --runtime ground-truth --problem-dir problems/<task_id>
  ```

### Policy and trace hardening

- `PolicyWorker` and `load_submitted_policy(...)` accept an optional
  `total_timeout_s`. The budget charges cumulative time waiting for
  submitted-worker RPCs; exceeding it raises `PolicyTimeoutError`/`AgentFault`
  before the outer grader timeout can discard the run.
- Malformed MessagePack replies now become submitted-policy faults instead of
  escaping the trusted decoder. Non-finite scalar and array reply values are
  rejected at the worker boundary before task code can launder NaN into a
  finite high score.
- Trace-required evaluation now writes a nonce-bound failure trace for the
  runner's explicitly kept `unclassified_grader_crash` zero while preserving
  `critical_operator_alert`. Infrastructure failures and ordinary successful
  zero scores still require their normal trace and are not relabeled.
- Validation rejects sealed policy graders that catch `AgentFault` and return a
  score, requires both policy budgets in `instruction.md`, and rejects policy
  compute budgets above 80% of the effective grading timeout.

### Runtime resources

- The local harness tmux tool now runs as uid/gid `1000:1000`, inherits the same
  per-process address-space cap as rubric-launched agent tools, and defaults
  common BLAS/OpenMP thread pools to one thread.
- `RLIMIT_AS` remains a per-process limit. It does not cap the aggregate memory
  of a multiprocessing/joblib tree; production runtimes still need an
  aggregate agent cgroup or separate workload container to guarantee that a
  parallel sweep cannot kill the control plane.

## 2026-08-10T13:28:00-07:00 — Shared grader and sandbox hardening

### Action required

- Rebuild task images to receive the hidden-environment, memory-isolation, and
  grading-audit fixes.
- Regenerate continuous-task calibration locks when adopting the new contracts.
  New locks use schema `3.1` and the top-level task schema is
  `continuous-task.v3`. Default-only legacy `3.0` locks and their paired
  evidence remain readable through the v2 compatibility identity.
- Re-run reference, validation, and ground-truth checks after migration:

  ```bash
  uv run lbx-rl-harness reference --problem-dir problems/<task_id>
  uv run lbx-rl-template check --problem-dir problems/<task_id>
  uv run lbx-rl-harness run --runtime ground-truth --problem-dir problems/<task_id>
  ```

### Continuous evaluation

- `PrivateTableChallenge` now evaluates the full private bank when
  `sample_size` is omitted. Existing explicitly sized challenges retain their
  artifact-bound selection for lock compatibility; set
  `selection_policy="stable_subset"` to use the same hidden rows across
  artifacts and evaluation nonces.
- `EvaluationContext.replay(...)` reconstructs nonce-bound evidence only after
  verifying the committed artifact digest. The fresh nonce still controls
  permutation evidence, but no longer changes private-table row selection.
- `PythonPredictor` now declares `predict_timeout_s`,
  `first_call_timeout_s`, `max_rows`, and `max_reply_bytes`. These budgets are
  included in the task specification and must leave headroom for the reference.
- Set `prediction_scope="row_independent"` when each prediction must depend
  only on its own row. The grader compares full-batch output with shuffled
  partitions evaluated by fresh workers.
- `CsvRows` and `PythonPredictor` support per-column `value_domains` and opt-in
  `OneHot(...)` and `Simplex(...)` grouped constraints.
- CSV extra-column handling is now explicit:
  `extra_columns="reject" | "drop" | "preserve"`. Use `"drop"` for safe
  tolerance. The legacy `allow_extra_columns` argument remains a compatibility
  alias.
- Use `grading.helpers.join_submission_to_truth_or_fault(...)` for keyed
  prediction/truth joins. It projects declared fields before merging and
  attributes malformed candidate keys to `AgentFault`.
- Generated calibration locks now record both
  `qualification_naive_score` and `runtime_naive_quality_score`. A gap above
  `max_unacknowledged_naive_score_gap` requires
  `GeneratedCalibration(naive_semantic_gap_acknowledgement=...)` with a
  reviewed-exception rationale.

### Policy tasks

- `PolicyEvaluationTask` now declares `required_control_families` (default:
  `no_op`, `constant`, and `open_loop`).
- Pass `control_families` to `PolicyEvaluationTask.grade(...)` to enforce that
  every trusted control is classified exactly once. Legacy calls without the
  mapping remain supported but are recorded as unclassified.
- Submitted-policy failures after trusted worker setup, including bounded
  memory failures and call timeouts, are typed agent faults and produce a kept
  zero rather than voiding the rollout.

### Runtime and hidden environments

- The privileged MCP and supervised environment server now start from
  root-owned `/`. Environment-server restarts use
  `python -P -m env_server`, remove inherited Python import paths, use a
  root-owned executable `PATH`, and cannot import or execute agent-planted
  `/workdir` content as root.
- The environment server caps live connection handlers, rejects excess
  sockets, and backs off on `EMFILE`, `ENFILE`, `ENOBUFS`, and `ENOMEM`
  instead of crashing. Local/reference runs use the same hardened launch path.
- Exhausting the environment-server restart budget leaves the MCP alive so the
  episode can still reach grading.
- Agent shell/editor descendants and submitted policy/executable workers
  inherit a hard address-space limit. The default is 75% of the detected
  cgroup limit, capped at 56 GiB. Trusted deployments may override it with
  `RUBRIC_AGENT_MEMORY_LIMIT_BYTES` and
  `RUBRIC_POLICY_MEMORY_LIMIT_BYTES`.
- Grade metadata now records terminal artifact commitment, whether grading was
  attempted, an explicit grading state, and the shared runtime security
  revision.

### Validation and release gates

- Strict releases (`LBX_STRICT_RELEASE_GATES=1`) require
  `baselines/portfolio.json` with at least three named baseline workspaces below
  `baselines/` (the root itself is not a workspace) and explicit required
  families.
- Shared release APIs now cover baseline strength
  (`evaluate_baseline_portfolio`), representative-panel saturation
  (`evaluate_score_panel`), and unchanged-artifact regrade stability
  (`evaluate_regrade_stability`).

### Documentation

- `README.md` now links to this changelog. Shared framework changes that affect
  task authors must update this file in the same change set.
