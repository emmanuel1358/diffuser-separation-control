# Grading Tasks

This is the holistic author guide for grading on the lbx-rl-tasks pipeline:
how to write `compute_score.py`, when to pick which return shape, what
flows back to Boreal, what flows to Harbor, and how an external customer
consumes the exported task for RL training.

## TL;DR

- Every task ships **one Python file**: `scorer/compute_score.py`.
- That function returns one of three shapes — pick whichever fits naturally.
- The same grader powers both the Boreal rubric MCP server and Harbor's
  `tests/test.sh`. Per-criterion subscores show up in both UIs automatically.
- A shared library at [`grader/`](../grader/) gives you
  `RubricBuilder`, deterministic helpers, and `PolicyWorker` — baked into the
  base image so authors can import from `grading`.
- Rubrics must be deterministic. Do not use LLMs as judges in
  `compute_score.py`; reward functions must be reproducible from files,
  simulator state, fixed seeds, and hidden fixtures.

## The contract

```python
# scorer/compute_score.py
from pathlib import Path

def compute_score(
    workspace: Path,                   # /tmp/output (agent's workspace)
    trajectory,                        # agent transcript (str), or [] if none
    private: Path,                     # /mcp_server/data (your hidden test set)
) -> float | dict | "Grade":
    ...
```

`trajectory` carries the full agent transcript as a string (empty `[]` when
none is available). Use `helpers.transcript_contains(trajectory, needle)` for
transcript-based anti-cheat checks — e.g. withholding credit when the
transcript shows the agent reading grader-private paths. Treat this as
defense-in-depth: string scans are circumventable, so the privilege-dropped
sandbox remains the real boundary.

The function is invoked once per attempt:

- by the Boreal rubric MCP server's `grade_problem` tool (after the agent finishes),
- by `/runtime/run_grader.py` inside the Harbor container (when `harbor run` calls `tests/test.sh`).

Both call sites use the same shared library to normalize whatever
`compute_score()` returns into a canonical `Grade.to_dict()`.

## Executable Policy Submissions

If the submitted artifact is executable Python, such as `/tmp/output/policy.py`
or a `*_processor.py`, **never** import or `exec` it in the grader process. The
grader runs as **root** so it can read the `0600` hidden fixtures, so
`spec_from_file_location`/`exec_module`/`runpy.run_path`/`exec`/`eval` on the
model's file gives the submission root-level code execution — it can monkeypatch
your scoring, steal the answer key off the call stack, or `import` your private
reference module and score `1.0` with no real work. The validator's
`grader_sandbox` stage hard-fails any `scorer/*.py` that uses those primitives.

Use `PolicyWorker` (or the `helpers.run_policy` / `helpers.run_model_module`
wrappers) so submitted code runs in a non-root subprocess and communicates over
a JSON protocol — read the result from the return value, never the policy's
stdout:

```python
from grading import PolicyWorker

def compute_score(workspace, trajectory, private):
    with PolicyWorker(workspace / "policy.py", timeout_s=0.1) as policy:
        action = policy.act({"qpos": [0.0], "qvel": [0.0]})
```

For a module that exposes a single callable (the common numerical/structural
"processor" pattern), the one-shot convenience is:

```python
from grading import helpers

def compute_score(workspace, trajectory, private):
    result = helpers.run_model_module(
        workspace / "fixed_fiber_section_processor.py",
        "evaluate_section",
        str(public_case_path),
    )
    # score `result` against the hidden reference; never trust module stdout.
```

Static artifacts such as `model.xml`, `submission.csv`, or JSON/text answers do
not need `PolicyWorker` because the submitted code is not running during
grading. See [`POLICY_ISOLATION.md`](POLICY_ISOLATION.md) for the full guide,
including named policy methods, callable-worker usage, per-case workers, NumPy
observations, and `init_model_xml(...)` for policies that load the submitted
model XML.

## Agent Faults vs Grader Failures

The grader runs as root so it can read `/mcp_server/data`. The runtime now
performs ML_Envs-style hardening before grading: it kills lingering agent
processes when running as root, removes symlinks under the output directory
that point outside it, removes FIFOs/sockets/devices that could hang grading,
uses a private grader cache, and runs Harbor scoring in a child process rather
than importing task code in the verifier parent.

Use `AgentFault` for problems caused by the submitted artifact itself. These
score a clean `0.0` and are kept as training signal:

```python
from grading import AgentFault, helpers

def compute_score(workspace, trajectory, private):
    sub = helpers.load_submission_or_fault(
        workspace / "submission.csv",
        required_columns=["id", "prediction"],
        numeric_columns=["prediction"],
        unique_key_column="id",
    )
    # score `sub` against hidden truth...
```

Do not catch grader bugs, missing hidden fixtures, simulator crashes, OOMs, or
unsupported return shapes as `AgentFault`. Those are marked
`env_internal_failure` so the platform can discard the rollout rather than
train on a misleading zero. Non-finite scores (`NaN`, `inf`) also fail closed;
return a finite number in `[0, 1]` or a supported `Grade`/score dict.

The `agent_fault` validator stage statically enforces this discipline: it flags
a broad `except` that returns a score (over-keep), pickle deserialization of an
agent artifact (RCE-as-root), and an unguarded read of an agent-writable path
(over-discard). A generic `except Exception: return 0.0` is rejected.

### Submission loaders (per-type attack closures)

Each submission type has a sanctioned loader that runs the read/execution under
the unprivileged `agent` account and raises `AgentFault` for malformed input.
Use them instead of reading the artifact by hand:

| Submission | Loader | Closes |
| --- | --- | --- |
| CSV / dataframe | `helpers.load_submission_or_fault` | symlink/FIFO, oversize, schema |
| NumPy `.npz` / `.npy` | `helpers.load_submission_npz_or_fault` | symlink-to-truth (`O_NOFOLLOW`), FIFO-hang (`O_NONBLOCK`), oversize, pickle-RCE (`allow_pickle=False`) |
| Python policy / model module | `helpers.run_policy` / `helpers.run_model_module` | RCE-as-root (sandboxed subprocess; score from return value, never stdout) |
| Executable | `helpers.run_submitted_executable` | uid-drop + caps stdout (kills the child past the cap) + discards stderr (memory) + AgentFault on timeout/over-cap; `streaming=True` also strips `RUBRIC_SCORE=` lines (capture mode returns raw stdout for you to parse — never relay it to the score parser) |
| HDF5 | `helpers.load_submission_h5_or_fault` | external links / virtual datasets that map `/mcp_server` truth into a "dataset" |
| K-fold CV of a model module | `score_kfold_cv` | cross-fold label leakage via memory (fresh worker), IPC (`CLONE_NEWIPC`), and disk (per-fold pristine-snapshot rebuild) |
| Any format read by hand | `helpers.require_regular_file(path)` before the read | symlink/FIFO/dir/device + oversize (`lstat` + `S_ISREG`; a bare `except OSError` does not stop a symlink) |

## Pick your return shape

| Shape | When to use | Where the headline comes from |
| --- | --- | --- |
| `float` | ML_Envs migration. The headline is a single anchor-mapped numeric metric (RMSE/F1/return). No per-criterion decomposition. | The float you returned, clamped to `[0, 1]`. |
| `dict {score, subscores, weights, metadata}` | You have an existing scorer that returns its own headline plus per-target diagnostics (e.g. `{score: exp_curve(x_agg), subscores: {rmse: x1, f1: x2}}`). Per-criterion rows show up in Boreal UI; the headline stays your custom number. | `dict["score"]` verbatim. **NOT** recomputed from `subscores * weights`. |
| `RubricBuilder.grade().to_dict()` | The natural shape when you want weighted deterministic criteria and/or penalties — and you don't already have a custom anchor mapping. | Weighted aggregate over criteria, minus penalties, optionally collapsed to binary. |

The runtime detects the return shape from the value type. Separately,
`[difficulty].reward_type` tells validation what the ground-truth/reference
score should be.

### Why `dict["score"]` is authoritative

ML_Envs-style continuous scorers compute their headline as a
piecewise-linear-anchored aggregate of per-target progress.
The headline is NOT `mean(subscores)` — it's a calibrated curve through
floor / reference / perfect anchors. If we silently recomputed the
headline from `weighted_avg(subscores)` the math would drift.

So the contract is: when you ship an explicit `score`, that's the score
the user sees. Subscores are diagnostic.

## Continuous reward functions from ML_Envs

Use a continuous reward function when task quality is naturally measured by
numeric performance rather than a checklist: prediction error, F1, rollout
return, success rate, regret, trajectory tracking error, final objective value,
or a simulator-derived performance metric. This is the ML_Envs scoring pattern
adapted to `scorer/compute_score.py`.

In this style, the grader is still deterministic. It computes raw metrics from
the submitted artifacts, hidden fixtures, and fixed-seed simulations, then maps
those metrics into a stable reward in `[0, 1]`. Do not use an LLM judge or any
non-reproducible service inside the reward function.

### Calibration anchors

For each metric, define three anchors before writing the final curve:

- `floor`: the **worst plausible** raw metric value, NOT a baseline's measured
  value. Example: for SRE a constant-mean predictor gives SRE `1.0`, so
  `floor = 1.0`. It maps to progress `0.0`. Setting `floor` to a baseline's
  measured metric pins that baseline to 0 and compresses the 0->reference range,
  inflating agent scores. Baselines under `baselines/` are *evidence* that naive
  models sit above the floor, not the floor itself.
- `reference`: the reference solution value. This should map the final reward
  to about `0.5`, leaving headroom for better agents.
- `perfect`: the theoretical or practical optimum. Examples: RMSE/SRE `0`,
  F1 `1`, success rate `1`, regret `0`, tracking error `0`.

Normalize each raw metric to progress `x_i in [0, 1]` using only the `floor`
and `perfect` anchors. The reference value is not used in the per-metric
normalization; it is used to calibrate the final aggregate curve.

For a higher-is-better metric:

```python
def progress_higher(value: float, floor: float, perfect: float) -> float:
    return max(0.0, min(1.0, (value - floor) / (perfect - floor)))
```

For a lower-is-better metric:

```python
def progress_lower(value: float, floor: float, perfect: float) -> float:
    return max(0.0, min(1.0, (floor - value) / (floor - perfect)))
```

### Aggregate and curve

Weight the per-metric progress values with weights that sum to `1.0`:

```python
x_agg = 0.4 * x_success + 0.35 * x_error + 0.25 * x_regret
```

Then map `x_agg` to the final reward with a calibrated curve through three
points:

- `(0, 0)`: the naive baseline earns reward `0`.
- `(x_ref, 0.5)`: the reference solution earns about reward `0.5`.
- `(1, 1)`: perfect performance earns reward `1`.

Do not hand-roll this math: import the shared, tested toolkit from
`grading.calibration` so every continuous task uses the identical curve.

```python
from grading import calibration

x_success = calibration.progress_higher_better(success_rate, floor=0.2, perfect=1.0)
x_error = calibration.progress_lower_better(tracking_error, floor=2.5, perfect=0.0)
x_agg = 0.6 * x_success + 0.4 * x_error

# Solve once from the reference aggregate (x_ref -> 0.5), reuse per submission.
# PiecewiseLinearCurve is the only sanctioned curve; ExponentialCurve is deprecated.
curve = calibration.PiecewiseLinearCurve.from_reference(X_REF)
score = curve.score(x_agg)
```

`FLOOR` is the worst plausible raw metric (e.g. for SRE, the constant-mean
predictor gives SRE 1.0) — not a baseline's measured value. `REF` is the
reference's measured metric (maps to 0.5); `PERFECT` is the optimum.

A score dict is usually the best return shape because it preserves the exact
calibrated headline while exposing diagnostic metric progress:

```python
def compute_score(workspace, trajectory, private) -> dict:
    success_rate = ...
    tracking_error = ...
    regret = ...

    x_success = progress_higher(success_rate, floor=0.2, perfect=1.0)
    x_error = progress_lower(tracking_error, floor=2.5, perfect=0.0)
    x_regret = progress_lower(regret, floor=0.4, perfect=0.0)
    x_agg = 0.4 * x_success + 0.35 * x_error + 0.25 * x_regret

    return {
        "score": exp_curve(x_agg),  # calibrated through x_ref -> 0.5
        "subscores": {
            "success_progress": x_success,
            "tracking_progress": x_error,
            "regret_progress": x_regret,
        },
        "weights": {
            "success_progress": 0.4,
            "tracking_progress": 0.35,
            "regret_progress": 0.25,
        },
        "metadata": {
            "success_rate": success_rate,
            "tracking_error": tracking_error,
            "regret": regret,
            "x_ref": X_REF,
        },
    }
```

If you only need the headline reward and no diagnostics, returning a bare
`float` is also valid.

### Difficulty targets

Use calibration to measure agent difficulty:

- For `reward_type = "continuous_scoring_function"`, the checked-in
  reference solution in `solution/` should score `0.5 ± 0.05`.
- For `reward_type = "multi_deterministic_rubrics"`, the oracle in
  `solution/` should score `1.0` within `[ground_truth].score_epsilon`.
- Naive baselines should score about `0.0`. Ship them under `baselines/<name>/`
  (a constant/mean predictor anchors `FLOOR`). The validator grades each
  committed baseline through the real scorer and **fails the task if any baseline
  reaches the reference target** (the learnability gate: a naive guess must not
  match the expert).
- **No submission must score 0.** For `continuous_scoring_function` tasks this
  is enforced (not just recommended): an empty submission graded through the
  real scorer may not exceed `[ground_truth].zero_anchor_epsilon` (default
  `0.01`). The validator's no-op probe checks host-probed tasks; for
  in-container / ML_Envs tasks the ground-truth harness grades an empty
  workspace in the task image and records the score as
  `ground_truth_result.trivial_baseline_score` in the build proof, which the
  validator then requires and re-checks in CI.
- A task is appropriately hard when agents consistently score below `0.30`.
- Scores in `0.30-0.50` usually mean the task is marginally too easy; tighten
  anchors, strengthen the baseline, improve the task, or add distribution
  shift.
- Scores above `0.50` usually mean the task needs redesign: more complex data,
  harder environment dynamics, stricter held-out cases, or better metrics.
- A score of exactly `0.0` often indicates a runtime or artifact-loading
  failure; inspect logs before treating it as a meaningful model result.

Re-measure and update anchors any time you change the data generator, hidden
fixtures, environment dynamics, baseline, ground-truth solution, or metric
definition. Commit the baseline and ground-truth artifacts or scripts so
reviewers can see what anchors the reward. See
[`GROUND_TRUTH.md`](GROUND_TRUTH.md) for the required oracle and reviewer video
workflow.

## The three shapes — minimal examples

### 1. Bare float (ML_Envs migration target)

```python
# scorer/compute_score.py
from pathlib import Path

def compute_score(workspace: Path, trajectory, private: Path) -> float:
    f1 = compute_my_f1(workspace / "submission.csv", private / "test_target.csv")
    return anchor_map(f1, floor=0.0, perfect=1.0)
```

This shape is intended for future ML_Envs migrations. Headline only — no
per-criterion breakdown in either UI.

### 2. Score dict (preserve a custom anchor map AND surface diagnostics)

```python
def compute_score(workspace, trajectory, private) -> dict:
    rmse = ...
    f1 = ...
    x_agg = 0.5 * progress(rmse) + 0.5 * f1
    return {
        "score": exp_curve(x_agg),                      # YOUR headline math
        "subscores": {"rmse_progress": progress(rmse), "f1": f1},
        "weights":   {"rmse_progress": 0.5, "f1": 0.5},
        "metadata":  {"raw_rmse": rmse, "x_ref": _X_REF},
    }
```

`structured_subscores` rows show up in Boreal UI; Harbor's
`reward.json` gets one flat key per subscore for trainers that want
per-target signals.

### 3. RubricBuilder (typed deterministic criteria + penalties)

```python
from pathlib import Path
from grading import RubricBuilder, helpers

def compute_score(workspace: Path, trajectory, private: Path):
    rb = RubricBuilder(workspace=workspace, trajectory=trajectory, private=private)

    @rb.criterion(id="answer_present", weight=1.0,
                  description="answer.txt exists and is non-empty")
    def _():
        return helpers.file_exists(workspace / "answer.txt", non_empty=True)

    @rb.criterion(id="csv_rows_match", weight=2.0,
                  description="Submission has exactly 200 rows")
    def _():
        return len((workspace / "submission.csv").read_text().splitlines()) == 201

    @rb.penalty(id="wrote_outside", value=-0.5,
                description="Agent wrote files outside /tmp/output")
    def _():
        return any((workspace / p).exists() for p in ["/etc/passwd", "/root/x"])

    return rb.grade().to_dict()
```

Live examples:
- [`examples/mujoco-pendulum`](../../lbx-rl-tasks-template/examples/mujoco-pendulum/) — 10 deterministic criteria.

## RubricBuilder API reference

### `RubricBuilder(workspace, trajectory, private, *, binary_scoring=False)`

- `workspace` / `trajectory` / `private` — passed straight through from
  `compute_score()`.
- `binary_scoring=True` collapses the headline to `1.0` iff every
  criterion's subscore is `>= 0.5` AND no penalty triggered. Useful for
  pass/fail style tasks where partial credit doesn't make sense.

### `@rb.criterion(id=..., weight=..., description="")`

Decorate a 0-arg predicate. Return `bool` (1.0 / 0.0) or `float` in
`[0, 1]`. Predicates may be sync or async — async is useful for
deterministic-but-slow probes (e.g. `await asyncio.gather(...)` over many
files). Predicates are run in parallel via `asyncio.gather`. Exceptions
are caught and turn into a 0-score with `error_type:
"predicate_exception"` in the criterion log.

### No LLM-judged criteria

Do not call `rb.llm_criterion(...)` or invoke any model provider from
`compute_score.py`. Boreal uses LLMs to solve tasks, but the verifier must
be deterministic. If a requirement sounds subjective, rewrite it as a
code-checkable property or remove it from the rubric.

### `@rb.penalty(id=..., value=...)`

Decorate a predicate that returns truthy iff the penalty triggered.
`value` must be `<= 0`; its absolute value is subtracted from the
weighted total before clamping. Penalties are mutually independent.

### `rb.grade() -> Grade`

Resolves all criteria + penalties (concurrently via asyncio) and returns
a `Grade`. Authors call `.to_dict()` on the result before returning it
from `compute_score()`.

### `rb.metadata: dict[str, Any]`

Free-form caller metadata that lands in `Grade.metadata`. Use it for
debug values (raw metrics, computed thresholds, anything you want to see
in `reward-details.json`).

## Deterministic helpers

`from grading import helpers` exposes:

- `file_exists(path, *, non_empty=False) -> bool`
- `file_contains(path, needle, *, case_sensitive=True) -> bool`
- `regex_search(text, pattern, *, flags=0) -> bool`
- `exact_match(actual, expected, *, value_type="string") -> float` — `string`/`number`/`boolean`/`list`/`set`
- `abs_error(actual, expected, *, tolerance) -> float` — anchored, `1.0` at zero error
- `kendall_tau(actual, expected) -> float` — ranking similarity in `[0, 1]`
- `jaccard(actual, expected) -> float` — set similarity in `[0, 1]`
- `json_path(data, path) -> Any` — JSONPath subset (`$.a[0]`, `$.x[?k=v].y`)
- `transcript_contains(transcript, needle, *, case_sensitive=False) -> bool`
- `load_json(path) -> Any | None`
- `require_regular_file(path, *, max_bytes=...) -> Path` — raise `AgentFault` if the agent path is a symlink / FIFO / dir / device or oversized; call it before a hand-rolled read (the submission loaders in the table above do this internally)

These are convenience — if you need something not on this list, just
write the Python directly inside the criterion.

## Where the data flows

```mermaid
flowchart LR
  CS[compute_score.py] -->|float / dict / Grade| N[normalize_compute_score_return]
  N --> G[canonical Grade]
  G -->|Grade.to_dict| TUI[Boreal UI<br/>structured_subscores + rubric_breakdown]
  G -->|reward.json + reward.txt + reward-details.json| HUI[Harbor /logs/verifier/]
  HUI --> RL[Customer RL loop:<br/>reward = json.load reward.json score]
```

### Boreal submission payload

The exporter ALWAYS emits:

```json
"rubric": [],
"grading_strategy": [{"type": "mcp", "weight": 1.0}]
```

The grader is image-baked. Per-criterion detail flows back via the in-
container MCP `Grade.metadata.structured_subscores` — Boreal UI reads
this directly, so we don't need to ship the rubric structure on the
submission side.

What DOES come from `task.toml`:

- `[runner]` block → job-level fields (`api_model_name`,
  `n_attempts_per_problem`, `max_ctx`, `turn_limit`, `priority`,
  `iteration_order`, `serialize_restore_test_interval`,
  `checkpoint_ttl`).
- `[runner.timeouts]` → `setup_timeout_seconds`, `grading_timeout_seconds`,
  `tool_timeout_seconds`, `max_timeout_seconds` (see the timeout mapping below).
- `[runner.required_tools]` → `required_tools` (default
  `["bash", "str_replace_editor", "tmux"]`).
- `[difficulty]` + `[task].description` → per-problem `metadata` block
  (no effect on grading; surfaced in Boreal UI for filtering).

When committed ground-truth proof exists, the exporter also emits a redacted
`extra_fields.ground_truth_evidence` object from `.alignerr/build_proof.json`.
This includes oracle score and reviewer artifact paths/hashes; render-required
tasks also include pass status. It does not include hidden fixtures, solution
files, raw trajectories, or scorer metadata.

### Runner timeouts (Taiga/Boreal only)

`[runner.timeouts]` is the single knob group for how long a Taiga/Boreal run
gets. Each field maps to exactly one place in the exported payload — this is the
"which knob drives which platform field" mapping:

| `task.toml` field                 | Taiga/Boreal payload field  | Where it lands                                                        |
| --------------------------------- | --------------------------- | -------------------------------------------------------------------- |
| `[runner.timeouts].setup_sec`     | `setup_timeout_seconds`     | per-problem entry                                                     |
| `[runner.timeouts].grading_sec`   | `grading_timeout_seconds`   | per-problem entry **and** the in-image rubric subprocess (`extra_fields`) |
| `[runner.timeouts].tool_sec`      | `tool_timeout_seconds`      | per-problem entry                                                     |
| `[runner.timeouts].max_episode_sec` | `max_timeout_seconds`     | job-level (`null` = unlimited)                                        |

Notes:

- **`task_type = "ml"` is force-pinned to hour-scale and ignores author
  timeouts.** The Taiga exporter pins all four fields for `ml` tasks to
  `setup 7200 / grading 10800 / tool 21600 / max_episode 21600` (2h/3h/6h/6h)
  regardless of what `task.toml` carries — minutes-scale caps silently fail real
  ML runs, and a human building a reference solution needs the same headroom.
  `grading_sec` in particular is pinned to Taiga's maximum (`10800`). These live
  as `schemas.ML_*_TIMEOUT_SEC` (a single source of truth also reused by
  ML_Envs' `PINNED_TIMEOUTS`).
- **All other task types use the RunnerTimeouts defaults and stay
  author-overridable.** Non-ml tasks default to `setup 600 / grading 600 /
  tool 120 / max_episode 3600` and can override `[runner.timeouts]` freely.
- **Harbor ignores these entirely.** Timeouts only affect Taiga/Boreal; Harbor
  exposes its own resource controls via `[environment]`.

### Harbor task layout

```
my-task/
├── task.toml                 # [runner] + [environment].docker_image
├── instruction.md
├── environment/
│   └── Dockerfile            # FROM .../lbx-tasks-base@sha256:<digest>
├── solution/solve.sh         # required oracle solver
└── tests/
    └── test.sh               # 1-line shim into /runtime/run_grader.py
```

`tests/test.sh` is generated by the exporter — authors don't write it.
By default it's:

```bash
#!/bin/bash
set -euo pipefail
exec /runtime/run_grader.py \
    --workspace /app \
    --grader-dir /mcp_server/grader \
    --private-dir /mcp_server/data \
    --output-dir /logs/verifier
```

`/runtime/run_grader.py` and the entire `grader/` package live
inside `lbx-tasks-base`, so per-task `tests/` carries no Python code.

If a customer wants a Harbor task that runs against any image (not just
descended from `lbx-tasks-base`), pass `--standalone` to the exporter —
it bundles `run_grader.py` + the `grading/` package under
`tests/_runtime/` and the emitted `tests/test.sh` falls back to the
local copy.

### Customer RL training flow

```python
import json, subprocess
from pathlib import Path

for episode in range(N):
    subprocess.run(
        ["harbor", "run", "-p", "my-task", "-a", "my-agent",
         "-m", "policy-checkpoint-42"],
        check=True,
    )
    reward = json.loads(Path("logs/verifier/reward.json").read_text())["score"]
    rl_buffer.append((trajectory, reward))

    # Per-criterion subscores for credit assignment / reward shaping
    details = json.loads(Path("logs/verifier/reward-details.json").read_text())
    for sub in details["structured_subscores"]:
        log_to_wandb(sub["name"], sub["score"], sub["weight"])
```

`reward.json` shape:

```json
{
  "score": 0.85,
  "compiled": 1.0,
  "single_hinge": 1.0,
  "stable_rollout": 0.5,
  ...
}
```

(One flat key per criterion id, plus `score` for the headline. Bare-float
graders emit only `{"score": <float>}`.)

`reward-details.json` shape: full `Grade.to_dict()` —
`structured_subscores`, `metadata.rubric_breakdown` (per-criterion
deterministic reasoning and error fields when criteria failed),
`metadata.weighted_total`, `metadata.reported_final_score`, etc.

## Configuration cheatsheet (`task.toml`)

```toml
schema_version = "1.1"

[task]
name = "labelbox/my-task"
description = "What the task asks the agent to do (1 sentence)."

[environment]
required_resources = "4vcpu+16gib"

[agent]
timeout_sec = 1800

[verifier]
timeout_sec = 600
env = []                      # verifier should not need model-provider keys

[[outputs]]
path = "/tmp/output/answer.txt"
required = true
description = "Final answer file the agent writes."

[runner]
attempts = 3                  # n_attempts_per_problem
turn_limit = 1000             # null/0 = unlimited
max_ctx = 1_000_000
context_mode = "none"         # none / autocompact / memory
api_model_name = "claude-fable-5"
priority = "high"
iteration_order = "problems_first"
required_tools = ["bash", "str_replace_editor", "tmux"]

# Non-ml default timeouts (author-overridable). IGNORED for task_type = "ml":
# ml is force-pinned to hour-scale (setup 7200 / grading 10800 / tool 21600 /
# max_episode 21600) by the exporter regardless of what you put here.
[runner.timeouts]
setup_sec = 600
grading_sec = 600
tool_sec = 120
max_episode_sec = 3600

[difficulty]
task_type = "ml"              # ml | mujoco | cfd | structures
domain = "scientific_discovery_computational_science"
reward_type = "continuous_scoring_function"

[[hint]]
text = "Try logistic regression first; the data generator is roughly linear."
enabled = true
spoiler_level = 0.3
```

`[[outputs]]` is the generic final-artifact contract for every task type. During
Taiga export, the declarations are copied to the problem entry as `outputs` and
mirrored under `extra_fields.task_metadata.outputs`; graders should read the same
paths from `workspace` (`/tmp/output` in-container) that the prompt asks the
agent to write.

## Choosing between the three shapes — flowchart

```mermaid
flowchart TD
  Q1{Headline is a single<br/>anchored metric?}
  Q1 -- yes --> Q1a{Want diagnostic<br/>per-target rows<br/>in Boreal UI?}
  Q1 -- no --> Q2{Have weighted deterministic<br/>criteria and/or penalties?}
  Q1a -- no --> A[Bare float.<br/>return float]
  Q1a -- yes --> B[Score dict.<br/>return dict score subscores weights]
  Q2 -- yes --> C[RubricBuilder.<br/>return rb.grade.to_dict]
  Q2 -- no --> A
```

## Common patterns

### "I want some criteria to be subjective"

Do not use an LLM judge. Convert subjective requirements into measurable
properties: parse files, inspect simulator state, run deterministic rollouts,
compare against hidden fixtures, or score against fixed numeric thresholds.

### "I want a penalty if the agent did something forbidden"

`@rb.penalty(value=-0.5)` with a predicate that returns truthy iff the
forbidden behavior happened. The penalty's absolute value is subtracted
from the weighted total. Use multiple penalties for different misbehaviors.

### "I want to migrate an ML_Envs task verbatim"

Keep the existing anchor / progress math and set
`[difficulty].reward_type = "continuous_scoring_function"` so validation expects
the reference score near `0.5`. Map the aggregate through
`PiecewiseLinearCurve` (the sanctioned curve); only fall back to the deprecated
`ExponentialCurve` if you are reproducing a legacy task's exact numerics.

Return a bare `float` when only the headline score matters. Return a
score dict when you want to preserve the same headline score and expose
diagnostic continuous target progress values:

```python
def compute_score(workspace, trajectory, private) -> dict:
    # Same ML_Envs metric math as before.
    final = calibration.PiecewiseLinearCurve.from_reference(x_ref).score(weighted_progress)
    return {
        "score": final,
        "subscores": {
            "rmse_progress": rmse_progress,
            "f1_progress": f1_progress,
        },
        "weights": {"rmse_progress": 0.5, "f1_progress": 0.5},
        "metadata": {"raw_rmse": rmse, "raw_f1": f1},
    }
```

This is not a rubric. The returned subscores are diagnostic continuous
signals; `score` remains authoritative and is not recomputed from them.
See [`examples/mle-tabular-classification`](../examples/mle-tabular-classification/)
for a complete adapted ML_Envs task with parquet data and hidden targets.

### "My grader runs slow because each criterion makes an HTTP call"

Use async predicates and `RubricBuilder` runs them concurrently via
`asyncio.gather`:

```python
@rb.criterion(id="service_state_a", weight=1.0)
async def _():
    async with httpx.AsyncClient() as client:
        resp = await client.get("http://svc:8080/state/a")
        return resp.json()["ok"]
```

## Things you don't need to do

- **Don't write a `rubric.toml`.** Authoring is code-first; the rubric
  IS the Python.
- **Don't ship a `tests/` directory.** The Harbor exporter generates
  `tests/test.sh` for you.
- **Don't shell out to a separate "grade" binary.** Just import what
  you need and call functions directly.
- **Don't worry about subscore weighting math.** `RubricBuilder`
  normalizes weights to sum to 1.0 for you.
- **Don't try to cap the score yourself when using `RubricBuilder`.**
  `Grade.weighted_total()` clamps to `[0, 1]`.

## Validation

Template PR validation in mothership runs presence-driven checks:

| Stage | What it checks |
| --- | --- |
| `schema` | required files present, `task.toml` parses, `metadata.json` valid |
| `grader_import` | `compute_score` is callable with the expected signature |
| `outputs` | `task.toml` declares at least one `[[outputs]]` |
| `compute_score_return` | sample-runs the grader against an empty workspace + reference solution; records `return_shape` (bare_float / score_dict / rubric_grade), `sample_score`, and deterministic rubric details |
| `local_build_proof` | `docker buildx build` against the per-task Dockerfile |
| `conditional` | `docker compose config`, custom env smoke checks |

The `sample_score` MUST land in `[0, 1]` — this catches the most common
ML_Envs migration bug (forgotten clip on a custom anchor mapping).

## See also

- [`README.md`](../README.md) — repo overview and contributor loop
- [`docs/AUTHORING.md`](AUTHORING.md) — task-creation walkthrough
- [`grader/src/grading/`](../grader/src/grading/) — shared library source you can read locally
- The mothership repo's `docs/MUJOCO_TASKS.md` and `docs/ML_ENVS_MIGRATION.md` (visible to FDEs only) cover task-family-specific patterns.
