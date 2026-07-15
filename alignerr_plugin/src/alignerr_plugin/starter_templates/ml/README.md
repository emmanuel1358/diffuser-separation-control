# ML task template (ML_Envs mode)

This scaffolds an **ML_Envs-mode** task: the minimal authoring contract for ML
tasks in this template. You edit a tiny `metadata.json`, the prompt, the grader,
and the data - there is **no `task.toml`, no per-task Dockerfile, and no
`tests/test.sh`**. Every operational field (task type, reward type, timeouts,
runner knobs, the `/tmp/output` convention) is pinned centrally; the image builds
from the shared `base/task.mlenvs.Dockerfile`. See
[`docs/MLENVS_TASKS.md`](../../../../../docs/MLENVS_TASKS.md) for the full
contract and [`examples/mle-tabular-classification/`](../../../../../examples/mle-tabular-classification/)
for a complete working task.

## Layout

```text
<task_id>/
  metadata.json          # ml_task_type, required_resources, domain, license(+source), description
  prompt.md              # agent-facing prompt (no anchors, no internals)
  test_file.py           # no-arg compute_score() reading /tmp/output + /mcp_server/data
  data/public/           # agent-visible  -> /data/
  data/private/          # root-only truth -> /mcp_server/data/
  reference_solution/    # solution.py (the expert reference; anchors 0.5)
  baselines/naive/       # >=1 naive solution scoring clearly below the reference
  data-generation/       # provenance for the data
```

## After scaffolding, fill in

- **`metadata.json`**: set `ml_task_type` (`dataset` | `env` | `hybrid` |
  `sim_policy`), the `required_resources` enum, `domain`, and the real `license`
  + upstream `license_source`. Add `docker-base`, `dependencies`, `apt_extras`,
  or `hf_resources` only if needed (see the docs).
- **`prompt.md`**: the exact task prompt. Name installed tools/libraries
  directly. Keep it agent-facing: do not reveal the scoring anchors or grader
  internals, do not tell the agent to read the base image / `metadata.json` /
  declared dependencies to find packages, and do not add your own accelerator or
  tmux lines (the export appends those runtime notices automatically).
- **`test_file.py`**: your no-arg `compute_score()`. Read submissions only
  through `grading.helpers.*` loaders; calibrate with FLOOR/REF/PERFECT +
  `PiecewiseLinearCurve`. Raise `AgentFault` only for agent-controlled failures.
- **`data/public/`** and **`data/private/`**: the agent-visible files and the
  root-only held-out truth.
- **`reference_solution/solution.py`**: the expert reference (must score
  `0.5 +/- 0.05`); **`baselines/naive/`**: a trivial solution scoring clearly
  below it (the learnability gate).

## Iterate

```bash
# Run the reference solve/grade loop.
uv run lbx-rl-tasks-harness reference --problem-dir problems/<task_id>

# Prove ground truth before opening a PR.
uv run lbx-rl-tasks-harness run --runtime ground-truth --problem-dir problems/<task_id>
```

On a low-RAM dev host, `run` also accepts `--flavor slim` (or the default
`--flavor auto`, which falls back to the slim base if the heavy build OOMs).
