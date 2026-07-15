# MLE Tabular Classification (ML_Envs-mode example)

The canonical **ML_Envs-mode** task: a continuous-scoring dataset task authored with the minimal ML_Envs contract. The contributor edits only a tiny `metadata.json`, the prompt, the grader, and the data -- no `task.toml`, no per-task Dockerfile, no `tests/test.sh`. Everything operational is pinned centrally and the image builds from the shared `base/task.mlenvs.Dockerfile`.

## Layout (the whole authored surface)

```text
examples/mle-tabular-classification/
|-- metadata.json          # 8-key minimal config (see below)
|-- prompt.md              # agent-facing prompt (== instruction.md in native mode)
|-- test_file.py           # no-arg compute_score(); reads /tmp/output + /mcp_server/data
|-- data/
|   |-- public/            # agent-visible -> /data/  (train/test parquet, column map)
|   `-- private/           # root-only held-out truth -> /mcp_server/data/
|-- reference_solution/    # solution.py + captured submission.csv + results.txt (0.5 anchor)
|-- baselines/             # naive / linear / gbt baselines (must score below the reference)
`-- data-generation/       # provenance for the synthetic data
```

`metadata.json` is the only config the author touches:

```json
{
  "ml_task_type": "dataset",
  "required_resources": "12vcpu+100gib+h100/2",
  "domain": "scientific_discovery_computational_science",
  "license": "CC0-1.0",
  "license_source": "https://creativecommons.org/publicdomain/zero/1.0/",
  "description": "Synthetic tabular regression plus classification task using continuous ML_Envs-style scoring"
}
```

Everything else -- `task_type = "ml"`, `reward_type = "continuous_scoring_function"`, `allow_internet = false`, timeouts, the runner knobs, the `/tmp/output` convention, and the `mlenvs-gpu` base flavor -- is derived/pinned by `alignerr_plugin.mlenvs`. Optional keys: `docker-base` (`default` / `cuda-graphics` / `tpu`), `dependencies` (extra pip), `apt_extras`, `description`.

## The grader: `test_file.py`

`compute_score()` takes **no arguments** and reads the baked runtime paths directly:

- the agent's submission under `/tmp/output/`
- the held-out truth under `/mcp_server/data/` (root-only; from `data/private/`)

It returns a score in `[0, 1]` (here a score dict whose `score` is authoritative) and raises `grading.faults.AgentFault` for agent-controlled failures (kept 0.0) while letting author/infra failures propagate (discarded). It reads the submission with the sanctioned loader (`grading.helpers.load_submission_or_fault`) -- never by hand.

Because the submission is a static CSV, the grader executes no agent Python; `sim_policy` / `env` tasks that grade a submitted `policy.py` use `grading.policy_eval` / the env server instead.

## Scoring

Per-target metrics and anchors (`FLOOR` = worst plausible, `REF` -> 0.5, `PERFECT` = optimum):

| Target | Metric | Direction | Floor | Reference | Perfect | Weight |
| --- | --- | --- | --- | --- | --- | --- |
| `t1` | SRE (`RMSE / std(true)`) | lower is better | `0.2661` | `0.0330` | `0.0` | `0.35` |
| `t2` | SRE (`RMSE / std(true)`) | lower is better | `0.7770` | `0.2107` | `0.0` | `0.35` |
| `label` | binary F1 | higher is better | `0.9231` | `0.9939` | `1.0` | `0.30` |

Per-target progress is weight-combined, then mapped through the **sanctioned `PiecewiseLinearCurve`** (`grading.calibration`) so the reference lands at `0.5` and sub-reference progress is scored proportionally. `ExponentialCurve` is deprecated and must not be used.

| Submission | Expected score |
| --- | --- |
| `reference_solution/submission.csv` (the reference) | about `0.4999` |
| `baselines/linear/submission.csv` | about `0.0` |

## Pattern to copy for a new ML_Envs task

1. Create `problems/<task_id>/` with `metadata.json` + `prompt.md` + `test_file.py` + `data/{public,private}/` + `reference_solution/` + `baselines/`.
2. Keep `metadata.json` minimal; set `ml_task_type`, `required_resources`, `domain`, `license`, `license_source`.
3. Write `test_file.py` with a **no-arg** `compute_score()` that reads `/tmp/output` + `/mcp_server/data`, loads the submission via a sanctioned loader, calibrates with `FLOOR / REF / PERFECT` + `PiecewiseLinearCurve`, and `raise AgentFault` for agent faults.
4. The reference must score `0.5 +/- 0.05`; committed baselines must score clearly below it (the learnability gate).

See `docs/MLENVS_TASKS.md` for the full guide.
