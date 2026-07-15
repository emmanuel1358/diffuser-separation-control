# lbx RL Tasks Local Harness

The local harness lets you verify ground-truth proof locally and, optionally,
run an agent against a task on your laptop before spending CI/model budget. It
is meant to answer: "Does the oracle proof pass?" and, for optional agent runs,
"Would this task actually work in a Boreal-like agent run, and what did the agent
do turn by turn?"

## What It Runs

The harness accepts all three task shapes used by this repo:

- `problem-dir`: authoring directories under `examples/<task>` or `problems/<task>`.
- `taiga`: exported Boreal-compatible `problems-metadata.json` or job payloads.
- `harbor`: exported Harbor task directories.

With `--runtime claude-code` (the default) or `--runtime deepagents`, the harness
simulates the Boreal loop locally:

1. Builds the task image from `environment/Dockerfile`.
2. Starts that image with Docker using Boreal's `linux/amd64` platform.
3. Connects to the image-baked `/mcp_server` rubric MCP server.
4. Calls `setup_problem`.
5. Gives the model the in-container `bash`, file editor, and `tmux` tools.
6. Calls `grade_problem`.
7. Prints a readable turn-by-turn terminal trajectory plus final grader results.

## Quick Start

```bash
uv sync
```

For the default Claude Code runtime, use your Claude Code subscription: install
the `claude` CLI and run `claude /login`; the harness uses that local OAuth
session. If you do not have Claude Code subscription/login access, use
`--runtime deepagents` instead and create a local env file with provider API keys.
This file is gitignored. The repo includes `.env.example` with supported
providers and model settings. Template PR validation does not require your
provider keys.

```bash
cp .env.example .env.local
# Then edit .env.local and fill in keys only for provider-key runtimes.
```

Run the smallest example:

```bash
uv run lbx-rl-harness run --problem-dir examples/mujoco-pendulum
```

The local harness builds Docker images locally. If the required repo-local base
image is not already present, the harness first builds it from this repo's
`base/`, `taiga_runtime/`, and `grader/` directories, then builds the task image
from that local base. CPU tasks use
`lbx-tasks-base:runtime-ml-core-py313-local`; tasks with an H100
`[environment].required_resources` enum use
`lbx-tasks-base-gpu:runtime-ml-core-py313-local`; graphics and TPU enums select
their matching local bases. This does not require Google Artifact Registry
access; the first run can be slow because the base image installs Python and ML
dependencies.

For GPU tasks, local execution of GPU code also requires a Docker host with
NVIDIA GPU support. Mothership validation remains the authoritative GPU runtime
check.

You should see:

- numbered model-thinking blocks
- numbered tool-call blocks
- numbered tool-result blocks
- a final grader table with the headline score and criterion details

Run artifacts are written under `.harness-runs/` by default:

```text
.harness-runs/<task>-<format>-<timestamp>/
├── manifest.json
├── transcript.txt
├── trajectory.json
├── workspace/                 # copied final /tmp/output files
└── verifier/
    ├── reward.json
    ├── reward-details.json
    └── reward.txt
```

## API Keys

Boreal supplies model credentials in production. Local runs cannot use Boreal's
credentials, so you provide your own API key through environment variables.
Keys are read from the process environment only; the harness does not write
them to manifests, transcripts, task files, or trajectory files.

Supported provider env vars:

```bash
# Claude / Anthropic models
ANTHROPIC_API_KEY=sk-ant-...

# OpenAI models
OPENAI_API_KEY=sk-...

# Google Gemini models
GOOGLE_API_KEY=...
```

Recommended local setup:

```bash
cp .env.example .env.local
```

`.env.local` is ignored by git.

The harness automatically loads `.env.local` from the current working directory
when you run a command. You do not need to run `source .env.local` yourself.

You can set model defaults in `.env.local` too:

```bash
# Global model override. Use this to switch providers without passing --model.
LBX_RL_HARNESS_MODEL=claude-fable-5
# LBX_RL_HARNESS_MODEL=openai:gpt-5.1
# LBX_RL_HARNESS_MODEL=google:gemini-3-pro

# Provider-specific defaults. Used when the selected/task model belongs to
# that provider and LBX_RL_HARNESS_MODEL is unset.
ANTHROPIC_MODEL=claude-fable-5
# OPENAI_MODEL=openai:gpt-5.1
# GOOGLE_MODEL=google:gemini-3-pro
```

## Models

Model precedence:

1. CLI `--model`.
2. `.env.local` / env `LBX_RL_HARNESS_MODEL`.
3. Provider-specific env var matching the task model, such as `ANTHROPIC_MODEL`.
4. `task.toml [runner].api_model_name`, usually `claude-fable-5`.
5. Built-in fallback `claude-fable-5`.

You can override the model:

```bash
uv run lbx-rl-harness run \
  --problem-dir examples/mujoco-pendulum \
  --model claude-fable-5
```

Supported model naming:

```text
Claude / Anthropic:
  claude-fable-5
  claude-opus-4-7
  claude-sonnet-4-5
  anthropic:claude-fable-5
  env var: ANTHROPIC_API_KEY
  model env: ANTHROPIC_MODEL

OpenAI:
  openai:gpt-5.1
  gpt-5.1
  env var: OPENAI_API_KEY
  model env: OPENAI_MODEL

Google Gemini:
  google:gemini-3-pro
  gemini-3-pro
  env var: GOOGLE_API_KEY
  model env: GOOGLE_MODEL
```

If the provider is ambiguous, prefix it with `anthropic:`, `openai:`, or
`google:`.

## Common Commands

### Reference iteration (author loop)

Use `reference` while developing `solution/solve.sh`, especially for ML training
loops. It runs the same solve + grade path as ground truth but does **not**
write `ground_truth_result` or `harness_result` to `build_proof.json`.
Artifacts persist under `.alignerr/reference_cache/output` (by default) so you
can re-grade without retraining:

**Note:** container `reference` runs may still call `docker build` and update
`image_digest` / `task_dir_sha256` in an existing `build_proof.json`. They do
not add oracle or agent scores. Commit oracle proof only after
`--runtime ground-truth`.

```bash
# Solve + grade (uses cache after first run)
uv run lbx-rl-harness reference --problem-dir problems/<task_id>

# Grade cached outputs only
uv run lbx-rl-harness reference --problem-dir problems/<task_id> --skip-solve

# Solve/train only
uv run lbx-rl-harness reference --problem-dir problems/<task_id> --no-grade

# Wipe cache and rerun from scratch
uv run lbx-rl-harness reference --problem-dir problems/<task_id> --clean-cache
```

Each run writes `.alignerr/reference_run/latest/manifest.json` with score,
execution mode (`host` or `container`), and timing.

Optional `[reference]` block in `task.toml` (defaults shown):

```toml
[reference]
execution = "auto"   # auto | host | container
cache_dir = ".alignerr/reference_cache/output"
entrypoint = "solve.sh"
proof_mode = "auto"  # auto | artifact | execute
```

With `execution = "auto"`, the harness picks **container** when
`[ground_truth].in_container = true`, `[environment].required_resources` is an
accelerator enum, or `[difficulty].task_type = "ml"`; otherwise **host**.
Hidden-env tasks
(`[environment].hidden_env = "env"|"hybrid"`) start the env server inside the
container before running the reference entrypoint.

### Ground-truth proof (PR gate)

For PR submission, use `--runtime ground-truth`; this is the required
deterministic proof checked by template validation. By default,
`run --problem-dir ...` uses `--runtime claude-code`, which runs the local Claude
Code agent attempt plus rubric-quality review for optional local feedback.
`run-taiga` and `run-harbor` also default to `--runtime claude-code`.

Run only the deterministic checked-in solution/reference path:

```bash
uv run lbx-rl-harness run \
  --problem-dir examples/mujoco-pendulum \
  --runtime ground-truth
```

Run only a local Claude Code attempt:

```bash
uv run lbx-rl-harness run --problem-dir examples/mujoco-pendulum --runtime claude-code
```

Run the DeepAgents/provider-key path explicitly:

```bash
uv run lbx-rl-harness run --problem-dir examples/mujoco-pendulum --runtime deepagents
```

Run from an already-exported Boreal metadata file:

```bash
uv run lbx-rl-harness run-taiga \
  --metadata /tmp/problems-metadata.json \
  --source-problem-dir examples/mujoco-pendulum
```

Run from an already-exported Harbor task directory:

```bash
uv run lbx-rl-harness run-harbor \
  --task-dir /tmp/harbor-mujoco \
  --source-problem-dir examples/mujoco-pendulum
```

### Accelerator Tasks In Local Runs

Taiga submissions keep the existing prompt-visible availability note, such as
`A GPU may be available.` or `A TPU may be available.` Harbor exports carry that
availability as structured runtime notice metadata instead. For local harness
runs, the agent prompt is localized with explicit fallback guidance: probe
accelerator availability once, and if GPU/TPU access is unavailable or broken,
switch to a CPU-compatible path quickly. The required `/tmp/output` artifacts
are still mandatory.

Trusted CI routes GPU-configured tasks to the org `gpu` runner, but TPU tasks
use the default CPU runner. Taiga submission is separate and keeps the configured
accelerator resources.

## Troubleshooting

If Docker cannot resolve the base image on Apple Silicon, make sure you are on
the latest harness code. The harness forces `linux/amd64` to match Boreal's base
images.

If the model key is missing, the error will name the required env var, such as
`ANTHROPIC_API_KEY`, `OPENAI_API_KEY`, or `GOOGLE_API_KEY`.

If the final score is low, inspect the terminal trajectory first, then open:

```bash
cat .harness-runs/<run>/verifier/reward-details.json
cat .harness-runs/<run>/trajectory.json
```
