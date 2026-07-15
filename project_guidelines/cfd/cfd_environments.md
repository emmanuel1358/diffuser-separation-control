# Alignerr CFD RL Task Guide for CFD Engineers

This guide is for aerodynamicists and CFD engineers who want to create Alignerr
tasks around OpenFOAM, AVL, aerodynamic design, flow-control design, or
simulation-driven fluid-dynamics judgment.

You do not need to be an RL researcher to start. Think of an Alignerr RL task as
a small engineering challenge with three parts:

- a clear problem statement for the model,
- a deterministic checker that scores the model's final answer,
- a trusted reference solution that proves the task is solvable.

For CFD tasks, the model might choose a wing planform, submit a flow-control
configuration, tune boundary-condition parameters, repair a divergent case, or
optimize a geometry under constraints. The grader turns the result into a reward
score from `0.0` to `1.0`.

## 1. What This Project Is

The repository is an authoring template for Alignerr RL tasks. It gives you the
layout, local harness, grading library, Docker environment, and examples needed
to build tasks under `problems/`.

At a high level, the workflow is:

1. Write the task prompt in `instruction.md`.
2. Put any public files the model may inspect in `data/`.
3. Put hidden grading fixtures in `scorer/data/`.
4. Write a deterministic grader in `scorer/compute_score.py`.
5. Write an oracle solution in `solution/solve.sh`.
6. Run the local harness to prove the oracle scores `1.0`.
7. Run a model attempt, then use the PR Boreal report to confirm the task is
   challenging enough.

The model sees the prompt and public files. The grader sees the model's output
and private scorer data. This separation is important: it lets you use hidden
operating points, hidden target values, fixed solver settings, or precomputed
reference anchors without leaking the answer to the model.

## 2. RL in Plain English

Reinforcement learning is a way to improve a model using feedback from attempts.
For these tasks, the feedback is the grader score.

One episode looks like this:

1. The model reads the task instructions.
2. The model uses tools, writes files, runs scripts, or reasons through the
   engineering problem.
3. The model saves its final answer under `/tmp/output`.
4. The grader evaluates that answer and returns a reward from `0.0` to `1.0`.
5. Training or evaluation systems use that reward to compare attempts and improve
   future model behavior.

For a CFD task, the reward might be based on drag reduction, lift-to-drag ratio,
pressure recovery, vortex-shedding suppression, mesh validity, convergence, or
robustness over hidden flow conditions. The key idea is simple: if you can define
what a good fluid-dynamics answer looks like in measurable terms, you can turn
that into a training signal.

## 3. Get Your Task Repo and Dependencies

You need `git`, Docker, and `uv`. Docker is needed because tasks are checked in
containers that match the evaluation environment.

Before cloning anything, connect your GitHub account to the Labelbox project. New
task pickup is disabled until your GitHub account is linked.

Then pick up a task from the project:

1. Go to the project in Labelbox.
2. Open the **Tasks** tab.
3. Choose a task template.
4. Click the task's **Content URL**.

The Content URL opens the GitHub repository for that task. Before editing, review
the repo context:

- `README.md`
- `project_guidelines/cfd/`
- `examples/openfoam-hydrofoil-flap/`

Clone the task repository from GitHub. The repository name is task-specific and
will look like this:

```bash
git clone git@github.com:Alignerr-Code-Labeling/lbx-rl-tasks-***.git
cd lbx-rl-tasks-***
uv sync
uv run lbx-rl-harness --help
```

`uv sync` installs the local grading package, the harness, and the Alignerr
template utilities. Run authoring commands from the repository root with
`uv run ...`.

For local model runs, create `.env.local` from the example:

```bash
cp .env.example .env.local
```

Then open `.env.local` and put in your own Anthropic key:

```bash
ANTHROPIC_API_KEY=sk-ant-your-own-key-here
ANTHROPIC_MODEL=claude-fable-5
LBX_RL_HARNESS_MODEL=claude-fable-5
```

Do not commit `.env.local`. It is for your local machine only. The harness loads
it automatically when you run commands from the repo root.

## 4. The RL Template Anatomy

Each task lives in one directory:

```text
problems/<task_id>/
|-- task.toml
|-- metadata.json
|-- instruction.md
|-- environment/
|   `-- Dockerfile
|-- data/
|-- scorer/
|   |-- compute_score.py
|   |-- <solver>_case.py
|   `-- data/
|-- solution/
|   |-- solve.sh
|   |-- render.sh
|   `-- render_config.py
|-- baselines/
|   `-- naive.sh
`-- README.md
```

The main files are:

- `instruction.md`: the task prompt the model sees.
- `task.toml`: resources, required outputs, timeouts, difficulty metadata, and
  ground-truth rendering settings.
- `metadata.json`: task identity for the benchmark.
- `environment/Dockerfile`: installs task dependencies and copies public/private
  files into the container.
- `data/`: public files available to the model at `/data`.
- `scorer/compute_score.py`: deterministic grading code.
- `scorer/<solver>_case.py`: optional helper that builds, runs, and parses a
  deterministic solver case.
- `scorer/data/`: private files available only to the grader.
- `solution/solve.sh`: the oracle answer. It must produce a score of `1.0`.
- `solution/render.sh`: optional reviewer artifact generation.
- `baselines/naive.sh`: optional weak baseline to help calibrate difficulty.

Final model outputs should always go under `/tmp/output`. Do not ask the model
to put final answers in `/workspace` or another ad hoc location.

## 5. Choose the Right CFD Fidelity

Pick the cheapest solver that captures the physics your task tests.

| Tier | Solver | Where it lives | Use for |
| --- | --- | --- | --- |
| Low-fidelity potential or vortex-lattice | AVL | Per-task build from source | Wing planforms, trim, span loading, stability derivatives |
| Low-fidelity airfoil and VLM | AeroSandbox plus NeuralFoil | Base image Python packages | Airfoil polars, `L/D`, quick parametric sweeps |
| Mid/high-fidelity CFD | OpenFOAM v2412 | Base image solver environment | Separation, pressure recovery, wakes, vortex shedding, heat transfer |

OpenFOAM is available inside the task image through:

```bash
bash -lc 'source /etc/solver-envs.d/openfoam.sh && blockMesh -help'
```

After sourcing that script, commands such as `blockMesh`, `checkMesh`,
`simpleFoam`, and `pimpleFoam` are available. AVL is not in the base image
because it is a Fortran/X11 binary. If a task needs AVL, build it in
`environment/Dockerfile` and pin the source version.

### Keep solver requirements out of instruction.md

Every CFD task must keep `instruction.md` solver-agnostic. The prompt should
describe the engineering objective, deliverable, schema, units, and constraints
without mentioning solvers, naming OpenFOAM, mandating commands, asking for
solver evidence, or hinting at solver/tool implementation details. It should
still include physics-grounded scoring details in solver-agnostic terms.

OpenFOAM tasks must include solver-specific runnable material in the core
solution implementation (`solution/solve.sh` or solution files it calls).
Public `/data` may include ordinary task inputs and optional debug assets, but it
is not the blocking solver-validation surface. The oracle implementation should
include material such as:

```bash
bash -lc 'source /etc/solver-envs.d/openfoam.sh && blockMesh -case /tmp/case && checkMesh -case /tmp/case'
```

If a task uses another CFD engine, provide the equivalent runnable material in
the submitted oracle path outside `instruction.md`. The solver requirement is
enforced through the oracle source, physics, hidden oracle/scorer, ground-truth
proof, and image smoke tests; do not require the agent transcript or a
`solver_evidence.json` artifact to prove solver use.

## 6. Task Instructions Must Be Complete

A good CFD prompt should feel like a well-written engineering assignment. The
model should not have to guess what you meant.

Include:

- the exact deliverable and path, such as `/tmp/output/control.json`,
  `/tmp/output/wing.json`, `/tmp/output/airfoil.dat`, or `/tmp/output/case/`;
- the required file format and schema;
- units, coordinate systems, reference quantities, naming conventions, and valid
  parameter ranges;
- engineering constraints, such as blockage, drag caps, angle limits, geometric
  clearance, Reynolds-number assumptions, or convergence requirements;
- what public files are available in `/data`;
- a short scoring summary in physics-grounded, solver-agnostic terms, without
  revealing hidden case values or private anchors;
- any forbidden behavior, described without mentioning solvers or hidden
  solver/tool internals.

Avoid:

- ambiguous wording like "good", "stable", or "efficient" unless you define how
  it will be measured;
- hidden assumptions that are not in the prompt or public files;
- asking for one specific solution path when many engineering approaches could
  be valid;
- grading criteria that are not mentioned in the prompt.

The prompt and grader should agree. If the grader checks drag, the prompt must
define the drag-related objective. If the grader checks a geometry bound, the
prompt must state the bound.

## 7. Worked Example: OpenFOAM Hydrofoil Flap

The example task is:

```text
examples/openfoam-hydrofoil-flap
```

It asks the model to design a simplified two-dimensional marine hydrofoil aft
flap. The agent-facing prompt describes the physical objective, design bounds,
and scoring quantities without mentioning solvers, solver commands, or a
surrogate/probe. The agent chooses `flap_deflection_deg`, `hinge_gap_m`,
`flap_chord_fraction`, and `blend_radius_m`.

The model submits one file:

```text
/tmp/output/hydrofoil_flap.json
```

The hidden grader owns the private operating cases, fixed solver settings, mesh
generation, and scoring anchors. It recomputes the physical quantities from the
public problem definition and the submitted design; the agent does not submit
tool-provenance artifacts.

### Problem Definition

The model-facing prompt is in:

```text
examples/openfoam-hydrofoil-flap/instruction.md
```

It defines:

- the physical setting: a marine hydrofoil aft flap in a simplified water-tunnel
  design problem;
- the required output path for the flap design;
- public files available under `/data`;
- valid design bounds, units, and geometry feasibility constraints;
- the balanced objective: lift/control authority without excessive drag,
  separation, wake degradation, or poor hidden-case robustness.

The prompt includes enough information for the model to produce a valid design
without seeing hidden grader fixtures, and it states scoring in physical terms
without telling the agent which tool the grader uses.

### Public Data

Public engineering context lives in:

```text
examples/openfoam-hydrofoil-flap/data/
```

The key files are:

- `hydrofoil_flap_template.json`: required JSON schema starter.
- `public_operating_envelope.json`: bounds, packaging context, and public
  operating envelope.
- `public_baseline_summary.json`: baseline hydrofoil trends and tradeoffs.
- `public_calibration_samples.json`: broad public calibration windows.
- `public_transfer_guidance.json`: public off-design guidance for hidden-case
  transfer.

Public files should help the model reason about the engineering tradeoff. They
should not contain the hidden oracle answer, private operating points, hidden
reference values, full scoring thresholds, solver product names, solver command
checklists, or solver-provenance requirements.

### Task Configuration

The task settings are in:

```text
examples/openfoam-hydrofoil-flap/task.toml
```

Important entries include:

- `[task]`: task name and one-sentence description.
- `[environment]`: CPU, memory, storage, and internet settings.
- `[difficulty]`: `task_type = "cfd"`, `domain = "aerodynamics"`, and
  `reward_type = "multi_deterministic_rubrics"`.
- `[[outputs]]`: declares the required `/tmp/output` design file.
- `[ground_truth]`: sets `in_container = true` because OpenFOAM is available in
  the task image, not necessarily on the host.

Every CFD problem must include routable task metadata:

```toml
[difficulty]
task_type = "cfd"
domain = "<supported-cfd-domain>"
reward_type = "multi_deterministic_rubrics"
```

The `domain` value is exported as a task tag, so choose the most specific
supported CFD tag instead of inventing a new one. Supported CFD domain tags are:
`aerodynamics`, `airfoil_design`, `wing_planform_design`, `flow_control`,
`vortex_suppression`, `wake_dynamics`, `boundary_layer_modeling`,
`turbulence_modeling`, `mesh_generation`, `cfd_case_repair`,
`pressure_distribution_matching`, `vlm_trim_analysis`,
`xfoil_polar_optimization`, `rans_simulation`, `high_order_cfd`, and
`gpu_lattice_boltzmann`.

For OpenFOAM tasks, use `in_container = true` whenever the scorer, oracle, or
renderer needs OpenFOAM:

```toml
[ground_truth]
score_epsilon = 0.001
in_container = true
```

`in_container = true` is important because the host may not have OpenFOAM. The
local harness therefore runs the oracle solve and grader inside the built task
image and records the build proof from that environment.

### Grader and Reward

The deterministic scorer is:

```text
examples/openfoam-hydrofoil-flap/scorer/compute_score.py
```

It reads the submitted design, validates schemas and bounds, runs hidden
solver-backed cases, and returns a score dictionary.

The reward combines several kinds of signal:

- file validity and schema checks for the required output;
- design-bound and derived-geometry feasibility checks;
- public operating-envelope and transfer-guidance alignment;
- mesh and solver health checks;
- nominal lift, drag, separation, and wake-quality behavior;
- hidden-case robustness and spread across operating conditions.

Invalid JSON, missing outputs, invalid geometry, failed meshes, divergent solves,
NaNs, or non-converged cases should receive `0.0` for the affected criteria, and
often `0.0` for the whole task if the result is unusable.

The private fixed cases and reference anchors are stored in:

```text
examples/openfoam-hydrofoil-flap/scorer/data/
```

Those files are for the grader, not the model. Do not regenerate reference
anchors at grading time. Precompute them and commit them under `scorer/data/`.

### Oracle Solution

The reference solution is:

```text
examples/openfoam-hydrofoil-flap/solution/solve.sh
```

It searches through the public-guided design path, writes a known-good
`hydrofoil_flap.json`, and runs the public validation runner to generate
`public_openfoam_metrics.json` and `openfoam_run_manifest.json`. The oracle must
score `1.0`; otherwise the task is not ready.

Run it through the harness:

```bash
uv run lbx-rl-harness run   --runtime ground-truth   --problem-dir examples/openfoam-hydrofoil-flap
```

A passing run writes:

```text
examples/openfoam-hydrofoil-flap/.alignerr/build_proof.json
```

This example intentionally does not declare a reviewer MP4; solver-backed
validation is carried by the submitted oracle source, hidden grader, and build
proof. If your CFD task needs a visual review artifact, keep the same
`in_container = true` pattern and declare `render_outputs` explicitly.

## 8. CFD Determinism Rules

Determinism is load-bearing for CFD tasks. Pin every numerical choice that can
change the score.

- Use a pinned solver version. OpenFOAM v2412 is in the base image. AVL must be
  pinned in the task image if you build it yourself.
- Run grading in serial. Do not use `decomposePar` or MPI for scoring because
  parallel reductions can drift.
- Use fixed iteration control. For steady solves, use a fixed iteration budget or
  fixed convergence policy. For unsteady solves, use fixed `deltaT`, fixed
  `endTime`, and `adjustTimeStep no`.
- Use deterministic meshes. Prefer a parametric `blockMesh` or a pre-built mesh.
  Do not run `snappyHexMesh` at grade time.
- Keep schemes, solver settings, boundary conditions, initial conditions, and
  turbulence or transition models fixed.
- If an instability must be seeded, use a deterministic perturbation, not random
  noise.
- Grade on tolerances and physically meaningful summaries, not bitwise field
  equality.
- Treat divergence, NaNs, Courant blow-up, negative cell volume, and unusable
  non-convergence as failures.
- Keep hidden operating points, reference anchors, and exact scoring thresholds
  under `scorer/data/`.

Measure runtime early. The grader runs inside `verifier.timeout_sec`, and CFD can
consume the budget quickly. Keep meshes small, flow conditions moderate, solver
windows short, and hidden sweeps limited to the cases needed to test robustness.

## 9. Designing a CFD Rubric

Prefer a score dictionary built from deterministic criteria. A strong CFD task
usually has at least ten criteria, with at least five criteria that are hard
engineering checks rather than formatting checks.

Use criteria from several layers:

- Submission validity: required file exists, JSON or data format parses, schema
  matches, units and field names are correct.
- Feasibility shell: parameters are within physical bounds, geometry is not
  degenerate, blockage is limited, sub-grid features are rejected, and the design
  cannot simply wall off the flow.
- Mesh and case health: mesh generation succeeds, `checkMesh` passes selected
  thresholds, and boundary patches are valid.
- Solver health: solve completes without divergence or NaNs, residual behavior is
  acceptable, flux balance closes, and fixed timestep limits are respected.
- Physics objective: lift, drag, moment, `L/D`, pressure recovery, reattachment
  length, `RMS(Cl)`, Strouhal number, transition location, or heat transfer moves
  toward the intended target.
- Robustness: score the worst case or average case across hidden operating
  points, such as Reynolds number, angle of attack, inflow velocity, back
  pressure, CG location, or mesh perturbations.
- Anti-gaming: pair the main objective with cost caps, drag caps, blockage caps,
  mass limits, or geometry limits so the model cannot win by brute force.

No LLM judges should be needed. CFD rewards should come from the solver, mesh
checks, parsed field summaries, and numeric thresholds.

## 10. Example Task Families

Good CFD tasks are narrow enough to grade deterministically but open enough that
the model must make engineering decisions.

AVL or vortex-lattice tasks:

- Design a wing planform with twist, taper, and dihedral to hit a target `CL`
  while maximizing `L/D` across a hidden CG or speed sweep.
- Match a target spanwise lift distribution within tolerance.
- Achieve target stability derivatives such as `Cm_alpha` or `Cn_beta`.

OpenFOAM tasks:

- Generate or tune a backward-facing-step case that converges and reproduces a
  hidden reattachment length.
- Shape a two-dimensional diffuser for pressure recovery without separation over
  hidden inlet velocities.
- Suppress von Karman shedding behind a cylinder using passive flow-control
  geometry.
- Repair a provided case that diverges by fixing boundary conditions, schemes, or
  relaxation settings, then match a reference force value.

## 11. Validate That the Task Challenges the Model

Ground truth passing is necessary, but it is not enough. A task is useful for RL
only if the target model cannot trivially get a perfect score.

First, test locally with your own key in `.env.local`:

```bash
cp .env.example .env.local
```

Edit `.env.local`:

```bash
ANTHROPIC_API_KEY=sk-ant-your-own-key-here
ANTHROPIC_MODEL=claude-fable-5
LBX_RL_HARNESS_MODEL=claude-fable-5
```

Then run a local model attempt:

```bash
uv run lbx-rl-harness run \
  --runtime claude-code \
  --problem-dir examples/openfoam-hydrofoil-flap
```

The harness writes results under `.harness-runs/`. Inspect:

```text
.harness-runs/<run>/transcript.txt
.harness-runs/<run>/workspace/
.harness-runs/<run>/verifier/reward.json
.harness-runs/<run>/verifier/reward-details.json
```

Look for three things:

1. Did the model understand the prompt and produce the required file?
2. Did the grader run cleanly and return a meaningful score?
3. Did the model avoid a perfect `1.0` score?

### PR Boreal Complexity Target

After you open the task PR, the automated LBx validation comment will include
**Boreal Results**. Boreal runs a strong hidden model against the task and reports
the average score across attempts. Treat this as the authoritative difficulty
signal, not the local single-agent run.

**Timing:** the first `trusted-ci/grade` result usually appears after the local
mothership gates finish (roughly 10-15 minutes for typical solver tasks). The
Boreal rollout score and per-attempt table are posted later by the Taiga poller,
after the 5 production attempts complete, usually another 30-60 minutes. The
Taiga agentic QA findings (`env_linter` and reward-hacking detection) may take
longer because they are triggered after rollouts and fetched by the 10-minute
poller; expect about 1.5-3 hours from PR dispatch to a complete **Taiga QA
Findings** section. Do not treat the absence of that section as a clean QA pass;
it means the Taiga checks have not reported yet.

The target is:

```text
Average Boreal score < 0.400
```

If Boreal reports an average score of `0.400` or higher, the task is too easy for
the target SOTA hidden model even if local validation passed. Use the Boreal
per-attempt scores and per-criterion table to see what the model is solving, then
tighten the engineering challenge fairly.

If the model scores `1.0`, the task may be too easy, too constrained to one
obvious answer, or accidentally leaking the solution. Do not hide essential
instructions to make it harder. Instead, improve the engineering challenge:

- add more hidden fixed cases;
- vary flow conditions within stated public assumptions;
- make the design space larger while keeping the schema clear;
- add meaningful constraints, such as drag, blockage, convergence, or robustness;
- improve the baseline and oracle anchors;
- remove accidental hints that reveal the oracle geometry or settings;
- reduce large all-or-nothing score cliffs that let a partial solution earn too
  much credit;
- add independent hidden checks so matching one public flow condition or geometry
  pattern is not enough to cross the `0.400` average Boreal target.

The goal is not to trick the model. The goal is to create a fair engineering
problem where a strong model can make progress, but the current target model
stays below the Boreal acceptance threshold.

## 12. AVL Per-Task Install Recipe

AVL is a Fortran and X11 binary, not a base-image dependency. Build it in the
task Dockerfile only for tasks that need it:

```dockerfile
ARG BASE_IMAGE=lbx-tasks-base
ARG BASE_TAG=runtime-ml-core-py313-local
ARG PROBLEM_DIR=problems/<task-id>
FROM ${BASE_IMAGE}:${BASE_TAG}
USER root
RUN apt-get update && apt-get install -y --no-install-recommends \
      gfortran make libx11-dev curl && rm -rf /var/lib/apt/lists/* \
 && curl -fsSL https://web.mit.edu/drela/Public/web/avl/avl3.36.tgz | tar xz -C /opt \
 && cd /opt/Avl \
 && make -C plotlib FC=gfortran CC=gcc \
 && make -C bin FC=gfortran FFLAGS="-O -fallow-argument-mismatch" avl \
 && install -m 0755 bin/avl /usr/local/bin/avl
# ... then the standard grader COPY/install block ...
```

Call `avl` from the scorer with `subprocess`, feed it a deterministic command
script, and parse printed forces or stability derivatives. Treat failed or
non-converged AVL runs as failed grading cases.

## 13. Final Author Checklist

Before opening or updating a task PR, check:

- `instruction.md` is complete, non-ambiguous, and aligned with the grader.
- The required output path is under `/tmp/output`.
- Public files in `data/` contain everything the model needs, but not hidden
  answers.
- Private fixtures, hidden conditions, and reference anchors live under
  `scorer/data/`.
- `scorer/compute_score.py` is deterministic and returns a score in `[0, 1]`.
- Solver version, serial execution, mesh generation, timestep or iteration
  budget, schemes, boundary conditions, and seeds are pinned.
- The task uses deterministic `blockMesh` or a pre-built mesh; no nondeterministic
  meshing runs at grade time.
- Degenerate geometry, divergence, NaNs, invalid meshes, and non-convergence are
  handled as failures.
- `solution/solve.sh` produces the required files and scores `1.0`.
- Rendering works if `[ground_truth].render_outputs` is declared.
- Local ground-truth validation passes:

```bash
uv run lbx-rl-harness run \
  --runtime ground-truth \
  --problem-dir problems/<task_id>
```

- Optional local static validation passes:

```bash
uv run lbx-rl-template validate --problem-dir problems/<task_id>
```

- A local model run does not score `100%`:

```bash
uv run lbx-rl-harness run \
  --runtime claude-code \
  --problem-dir problems/<task_id>
```

- After the PR is opened, the LBx validation **Boreal Results** average is below
  `0.400`. If it is `0.400` or higher, revise the task and rerun validation.
- Wait for the **Taiga QA Findings** section in the LBx validation comment before
  treating the task as review-ready. Findings often appear 1.5-3 hours after the
  PR dispatch; if the section is still missing after about 3 hours, check the
  mothership `poll-taiga-runs.yml` workflow or re-run the poller.
- `.alignerr/build_proof.json` is committed after the final task edits.
- `.alignerr/ground_truth/` artifacts are committed when the task declares them.
- `.env.local`, `.harness-runs/`, API keys, and other secrets are not committed.

If those checks pass and Boreal stays below `0.400`, the task is in good shape
for review.

## Env Pre-flight QA (Blocking in CI)

Every task PR runs a blocking env pre-flight QA in trusted CI — a local
mirror of Taiga's env_linter — before any image is built or submitted.
Error findings fail `trusted-ci/grade`, skip the Taiga submission, and show
up in the PR comment under **Boreal Results — Env QA Pre-flight** with
suggested fixes. The same defects, if shipped, flip the problem to
**Failed QA** on Taiga after you tag "Ready for Customer".

What it checks, and what that means for CFD task design:

- **Oracle derivation (blocking).** `solution/solve.sh` must derive its
  answer from the public inputs (`/data` geometry, operating points, public
  baselines) by running the actual AVL/OpenFOAM computation — never heredoc a
  literal JSON answer. A constants-only oracle encodes the hidden reference
  and hides an unsolvable task behind a passing oracle gate.
- **Hidden-vs-public divergence (blocking).** If the grader compares against
  hidden reference values (target coefficients, reference fields, hidden
  operating conditions) within tight tolerances, those values must be
  estimable from the disclosed data. The check pairs numeric keys in
  `scorer/data/*.json` against `data/*.json` and blocks when the divergence
  dwarfs the tightest grader tolerance. If your hidden case intentionally
  differs (e.g. off-design evaluation), disclose the data that lets the agent
  bridge the gap — measured samples, transfer curves, or the divergence
  magnitude itself — in `data/` AND the task prompt.
- **Disclosure consistency (LLM review).** The prompt's framing must match
  the hidden magnitudes, and no validity/guardrail subscore may be earnable
  by a junk constants-only submission (tie floors to evidence of a real run:
  residual histories, mesh metadata, solver logs — consistent with the
  Anti-Gaming section above).

Findings flow: blocking pre-flight findings appear immediately in the PR
comment; advisory (warning/info) pre-flight findings are echoed under **Boreal
Results** when the Boreal run reports. Taiga's own agentic QA findings are
separate: after rollouts complete, mothership triggers/syncs Taiga's
`env_linter` and reward-hacking checks and renders them under **Taiga QA
Findings**. Budget 1.5-3 hours for that section on a fresh PR run.

Run the pre-flight check locally before pushing (from the mothership checkout):

```bash
python3 .github/scripts/env_preflight_qa.py \
  --problem-dir problems/<your-task> --skip-llm
```
