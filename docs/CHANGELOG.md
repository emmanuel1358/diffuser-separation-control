# Task Author Changelog

Task-author-visible changes to the shared grader, runtime, validators, and
authoring contracts are recorded here. Entries use ISO 8601 timestamps with a
UTC offset and are prepended in reverse chronological order (latest first).

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
