# Converting a removed-layout ML task

There used to be a second, ML-only authoring contract ("ML_Envs mode"): a task
dir with `metadata.json` + `prompt.md` + `test_file.py` and **no** `task.toml`,
built from a shared `base/task.mlenvs.Dockerfile` onto separate `mlenvs-*` base
images. It has been removed. Every task now uses the one native contract on the
flagship bases, described in [`ML_TASKS.md`](ML_TASKS.md).

Loading a task dir in the old layout fails with `LegacyTaskLayoutError`. Convert
it as follows.

## File mapping

| Removed layout | Native layout |
| --- | --- |
| `metadata.json` (config keys) | `task.toml` — see below |
| `metadata.json` (envelope) | `metadata.json` with `benchmark` + `problem_data` |
| `prompt.md` | `instruction.md` (verbatim) |
| `test_file.py` | `scorer/compute_score.py` (verbatim) |
| `data/public/` | `data/` |
| `data/private/` | `scorer/data/` |
| `reference_solution/` | `solution/` |
| `baselines/` | `baselines/` (unchanged) |
| `data-generation/` | `data_generation/` |
| (generated Dockerfile) | `environment/Dockerfile` — author it, see below |

Paths baked into the image are unchanged: public data still lands at `/data`,
held-out truth at root-only `/mcp_server/data`, the grader at
`/mcp_server/grader`, and the calibration lock at `/mcp_server/calibration`. Only
the **source** paths move, so update any `data/public/...` or `data/private/...`
references in `model.manifest.json`, `train.py`, `solution.py`, and the data
generators.

## `metadata.json` keys -> `task.toml`

| Removed key | Native equivalent |
| --- | --- |
| `ml_task_type` | dropped; set `[environment].hidden_env` for the old `env` / `hybrid` values |
| `required_resources` | `[environment].required_resources` |
| `domain` | `[difficulty].domain` |
| `license` / `license_source` | `[difficulty].license` / `.license_source` |
| `description` | `[task].description` |
| `docker-base` | `[environment].base_flavor` (usually omit; the resource tier resolves it) |
| `dependencies` | `environment/requirements.txt` |
| `apt_extras` | `environment/apt.txt` |
| `env_dependencies` | `scorer/env-requirements.txt` |
| `grading_dependencies` | `scorer/requirements.txt` |
| `hf_resources` | `[[preloaded_files]]` entries with `hf_repo` |

`[difficulty].task_type = "ml"` and
`[difficulty].reward_type = "continuous_scoring_function"` were implicit before;
declare them explicitly now. Timeouts stay pinned centrally for `ml`, so do not
author them.

An `hf_resources` entry maps field-for-field: a bare `"org/name"` string becomes
`hf_repo = "org/name"`, and an object's `repo_id` / `revision` / `repo_type` /
`allow_patterns` / `ignore_patterns` become `hf_repo` / `hf_revision` /
`repo_type` / `allow_patterns` / `ignore_patterns`. Leave `mount_path` unset; it
is derived. The cache root moved from `/tmp/.cache/huggingface` to
`/tmp/hf-cache`, but `HF_HOME` is set for you, so `from_pretrained("org/name")`
is unchanged.

## `environment/Dockerfile`

The old shared Dockerfile is now per-task, which is what makes the hardening
auditable. Copy the one from
[`examples/mle-tabular-classification/environment/Dockerfile`](../examples/mle-tabular-classification/environment/Dockerfile)
and adjust the base flavor. It must:

1. Run `/opt/lbx-runtime/install-task-deps.sh` over a staged copy of
   `environment/` + `scorer/`, which installs each dependency channel into its
   isolation boundary.
2. Copy `data/` to `/data` read-only.
3. Copy `scorer/data/` to `/mcp_server/data/` and `scorer/` to
   `/mcp_server/grader/`, then `rm -rf /mcp_server/grader/data`.
4. Copy `calibration.lock.json` to `/mcp_server/calibration/` **and** write the
   `.author-source` marker beside it. Without that marker the rubric server
   cannot tell the author's baked lock from the trusted-CI promoted mount, and
   will grade with the author's.
5. Chmod the whole private tree `0700`/`0600` and `/mcp_server` itself `0700`.

## Grader

`compute_score` is unchanged: both the no-arg form and
`(workspace, trajectory, private)` are supported. Continuous graders must be on
the v3 evaluation API — replace production `TASK.score(metrics)` with
`TASK.grade(...)`, regenerate the lock at schema v3, and prefer a queryable
`ContinuousTask.model()` challenge over a fixed prediction file. See
[`CONTINUOUS_EVALUATION.md`](CONTINUOUS_EVALUATION.md).

If the task used `ExponentialCurve`, switch to `PiecewiseLinearCurve` (the only
sanctioned curve). Anchors and weights are unchanged; the reference still lands
at 0.5.

## Recalibrate

The base image changed, so the reference and baseline anchors must be
re-measured:

```bash
uv run lbx-rl-harness run --runtime ground-truth --problem-dir problems/<task_id>
```

Then re-run validation:

```bash
uv run lbx-rl-template check --problem-dir problems/<task_id>
```

The umbrella checklist for all task types is [`TASK_MIGRATION.md`](TASK_MIGRATION.md).
