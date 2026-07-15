# Alignerr OpenSees RL Task Guide for Structural Engineers

This guide is for structural engineers who want to create Alignerr tasks around
OpenSees, structural analysis, retrofit design, or simulation-driven engineering
judgment.

You do not need to be an RL researcher to start. Think of an Alignerr RL task as
a small engineering challenge with three parts:

- a clear problem statement for the model,
- a deterministic checker that scores the model's final answer,
- a trusted reference solution that proves the task is solvable.

For structural engineering tasks, the model might choose a retrofit layout,
calibrate a simplified model, write an analysis script, interpret analysis
results, or optimize a design under constraints. The grader turns the result
into a reward score from `0.0` to `1.0`.

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
analysis cases, hidden target values, or fixed random seeds without leaking the
answer to the model.

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

For a structural task, the reward might be based on drift reduction, analysis
convergence, strength, cost, code checks, or agreement with hidden fixed cases.
The key idea is simple: if you can define what a good engineering answer looks
like in measurable terms, you can turn that into a training signal.

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
- `project_guidelines/strctural_engineering/`
- `examples/opensees-base-isolation/`

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
|   `-- data/
|-- solution/
|   |-- solve.sh
|   `-- render.sh
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
- `scorer/data/`: private files available only to the grader.
- `solution/solve.sh`: the oracle answer. It must produce a score of `1.0`.
- `solution/render.sh`: optional reviewer artifact generation.
- `baselines/naive.sh`: optional weak baseline to help calibrate difficulty.

Final model outputs should always go under `/tmp/output`. Do not ask the model
to put final answers in `/workspace` or another ad hoc location.

## 5. Task Instructions Must Be Complete

A good task prompt should feel like a well-written engineering assignment. The
model should not have to guess what you meant.

Include:

- the exact deliverable and path, such as `/tmp/output/retrofit_design.json`;
- the required file format and schema;
- units, coordinate systems, naming conventions, and valid ranges;
- engineering constraints, such as budget, strength limits, or convergence
  requirements;
- what public files are available in `/data`;
- a short scoring summary in physics-grounded, solver-agnostic terms, without
  revealing hidden case values or private answers;
- any forbidden behavior, such as writing the final output somewhere else.

Avoid:

- ambiguous wording like "reasonable", "good", or "efficient" unless you define
  how it will be measured;
- hidden assumptions that are not in the prompt or public files;
- asking for one specific solution path when many engineering approaches could
  be valid;
- grading criteria that are not mentioned in the prompt.

The prompt and grader should agree. If the grader checks cost, the prompt must
state the cost formula. If the grader checks story numbering, the prompt must
define story numbering.

## 6. Worked Example: OpenSees Base Isolation

The example task is:

```text
examples/opensees-base-isolation
```

It asks the model to design a lead-rubber base-isolation system for a
three-story building. The agent chooses `Qd_kip`, `Kd_kip_per_in`, and `Dy_in`,
then submits the design JSON.

The model submits one file:

```text
/tmp/output/isolation_design.json
```

The hidden grader owns the private ground-motion records and solver-backed final
scoring run. The submitted ground-truth oracle path must include solver-specific
runnable material. Public debug assets may remain under `/data`, but
`instruction.md` must not reference the solver, require solver commands, or ask
for solver evidence.

### Keep solver requirements out of instruction.md

Every structures task must keep `instruction.md` solver-agnostic. The prompt
should describe the structural problem, design variables, deliverable schema,
performance targets, and acceptance limits without naming OpenSees/OpenSeesPy,
mentioning solvers, mandating commands, asking for solver evidence, or hinting
at solver/tool implementation details. It should still include physics-grounded
scoring details in solver-agnostic terms.

Put the solver-specific runnable material in the core solution implementation
(`solution/solve.sh` or solution files it calls), and keep that tooling outside
the prompt. The solver requirement is enforced through the oracle source,
physics, hidden oracle/scorer, ground-truth proof, and image smoke tests; do not
require the agent transcript or a solver-evidence artifact to prove solver use.

### Problem Definition

The model-facing prompt is in:

```text
examples/opensees-base-isolation/instruction.md
```

It defines:

- the building and isolation-system design variables;
- the required output path: `/tmp/output/isolation_design.json`;
- public files available under `/data`;
- the disclosed ground-motion band, performance targets, moat capacity, and
  scoring curves;
- the tradeoff between soft isolation, displacement demand, drift, acceleration,
  and base shear.

The prompt includes enough information for the model to produce a valid design
without seeing hidden grader fixtures, and it states scoring in objective
structural-response terms rather than solver/tool terms.

### Public Data

Public engineering context lives in:

```text
examples/opensees-base-isolation/data/
```

The key files are:

- `building_description.md`: structural model, isolation-system context, and
  design rules.
- `design_schema.json`: the exact JSON schema for the required answer.
- `public_analysis_envelope.json`: disclosed analysis band, targets, scoring
  curves, and hidden-case sampling policy.
- `building_description.md`: structural context, units, design tradeoffs, and
  scoring summary.
- `design_schema.json`: the exact JSON schema for the required answer.
- `public_analysis_envelope.json`: disclosed response targets, credit curves,
  ground-motion band, and hidden-case sampling policy.
- `graded_case_sampling_policy.json`: deterministic stress-test envelope and
  suite-composition rules.
- `public_model_summary.py` and `isolation_starter.json`: lightweight constants
  and starter design data.

Public files should help the model understand the task. They should not contain
the hidden oracle answer, private evaluation records, solver product names,
solver command checklists, or solver-provenance requirements.

### Task Configuration

The task settings are in:

```text
examples/opensees-base-isolation/task.toml
```

Important entries include:

- `[task]`: task name and one-sentence description.
- `[environment]`: CPU, memory, storage, and internet settings.
- `[difficulty]`: `task_type = "structures"`, `domain = "structural_mechanics"`,
  and `reward_type = "multi_deterministic_rubrics"`.
- `[[outputs]]`: declares the required `/tmp/output/isolation_design.json` file.
- `[ground_truth]`: sets `in_container = true`, `score_epsilon`, and
  `max_trivial_score`.

Every structural engineering problem must include routable task metadata:

```toml
[difficulty]
task_type = "structures"
domain = "<supported-structures-domain>"
reward_type = "multi_deterministic_rubrics"
```

The `domain` value is exported as a task tag, so choose the most specific
supported structures tag instead of inventing a new one. Supported structures
domain tags are: `seismic_retrofit`, `structural_mechanics`,
`topology_optimization`, `truss_design`, `frame_analysis`, `beam_sizing`,
`buckling_analysis`, `modal_analysis`, `stress_displacement_analysis`,
`load_path_optimization`, `finite_element_model_repair`, `continuum_mechanics`,
`density_field_optimization`, `support_member_repair`, and
`mass_compliance_optimization`.

`in_container = true` is important for OpenSees tasks because the scorer and
oracle depend on packages installed in the task image (`openseespy`, `numpy`, and
other solver dependencies). The local harness therefore runs the oracle solve and
grader inside the built task image and commits the build proof from that
environment.

### Grader and Reward

The deterministic scorer is:

```text
examples/opensees-base-isolation/scorer/compute_score.py
```

It reads `isolation_design.json`, validates the design schema, runs deterministic
solver-backed response-history analyses on the hidden records, and returns a
score dictionary.

The reward is based on worst-case hidden-record performance for:

- isolator displacement, with a separate moat-capacity pounding gate;
- floor acceleration;
- interstory drift ratio;
- base-shear coefficient;
- convergence and residual displacement.

Invalid JSON, invalid design variables, non-convergence, missing outputs, or
out-of-range designs receive `0.0` through `AgentFault` or metric gates.

The private fixed cases are stored in:

```text
examples/opensees-base-isolation/scorer/data/hidden_cases.json
```

That file is for the grader, not the model.

### Oracle Solution

The reference solution is:

```text
examples/opensees-base-isolation/solution/solve.sh
```

It searches in-band public-style motions, writes a known-good
`isolation_design.json`, and must score `1.0` under the hidden grader. This is
the basic solvability proof for the problem.

Run it through the harness:

```bash
uv run lbx-rl-harness run   --runtime ground-truth   --problem-dir examples/opensees-base-isolation
```

A passing run writes:

```text
examples/opensees-base-isolation/.alignerr/build_proof.json
```

This example intentionally has `rendered = false` and does not declare reviewer
video output. If your structures task benefits from a visual artifact, declare
`render_outputs` explicitly and keep render generation deterministic.

## 7. Validate That the Task Challenges the Model

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
  --problem-dir examples/opensees-base-isolation
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
- vary load cases or geometry within stated public assumptions;
- make the design space larger while keeping the schema clear;
- add meaningful constraints, such as cost, convergence, drift, or robustness;
- improve the baseline and oracle anchors;
- remove accidental hints that reveal the oracle layout;
- reduce large all-or-nothing score cliffs that let a partial solution earn too
  much credit;
- add independent hidden checks so matching one public pattern is not enough to
  cross the `0.400` average Boreal target.

The goal is not to trick the model. The goal is to create a fair engineering
problem where a strong model can make progress, but the current target model
stays below the Boreal acceptance threshold.

## 8. Final Author Checklist

Before opening or updating a task PR, check:

- `instruction.md` is complete, non-ambiguous, and aligned with the grader.
- The required output path is under `/tmp/output`.
- Public files in `data/` contain everything the model needs, but not hidden
  answers.
- Private fixtures live under `scorer/data/`.
- `scorer/compute_score.py` is deterministic and returns a score in `[0, 1]`.
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

## 9. Grader Security and Calibration (Mandatory)

These rules encode the failure modes the Env Linter blocks at "Ready for
Customer". The local validator (`uv run lbx-rl-template validate`) and CI now
enforce most of them; a task that trips a blocking check is flipped to Failed
QA. Read this section before writing `compute_score.py` or the Dockerfile.

### 9.1 Never import, exec, or eval the model's code in the grader

The grading process runs as **root** so it can read `scorer/data/` (mounted at
`/mcp_server/data`, mode `0600`). If `compute_score.py` loads the model's
submitted Python with `importlib.util.spec_from_file_location` +
`spec.loader.exec_module`, `runpy.run_path`, `imp.load_source`, or `exec`/`eval`,
that submitted code runs **as root inside the grader**: it can monkeypatch your
scoring functions, walk the call stack to steal the hidden answer, or `import`
your private reference module — scoring `1.0` with zero real work.

If the task asks the model to submit a Python *processor* (e.g.
`fixed_fiber_section_processor.py`), call it through the sandbox instead:

```python
from grading import helpers

result = helpers.run_model_module(
    workspace / "fixed_fiber_section_processor.py",
    "evaluate_section",
    str(public_case_path),
)
# result crossed a JSON boundary from a NON-root subprocess; read the scored
# values from the return value — never trust the module's stdout.
```

`run_model_module` (and the lower-level `helpers.run_policy`) run the submission
in a non-root subprocess that cannot read the `0600` fixtures. The validator's
`grader_sandbox` stage fails any `scorer/*.py` that uses the dynamic-exec
primitives above.

### 9.2 Lock down private fixtures in the Dockerfile

Copy `scorer/` and `scorer/data/` to the private roots root-owned and unreadable
to the agent. The canonical block (used by the starter templates):

```dockerfile
COPY --chown=root:root ${PROBLEM_DIR}/scorer/data/ /mcp_server/data/
COPY --chown=root:root ${PROBLEM_DIR}/scorer/ /mcp_server/grader/
RUN rm -rf /mcp_server/grader/data \
    && chown -R root:root /mcp_server/data /mcp_server/grader \
    && find /mcp_server/data /mcp_server/grader -type d -exec chmod 0700 {} + \
    && find /mcp_server/data /mcp_server/grader -type f -exec chmod 0600 {} +
```

A plain `COPY scorer/data/ /mcp_server/data/` leaves the files world-readable
(`0644`): the agent can `cat /mcp_server/data/hidden_cases.json` and copy the
answer. The validator's `private_data_layout` stage now **requires** this
hardening (or a `--chmod=0600` COPY), and an in-image probe confirms uid 1000
cannot read the private roots.

### 9.3 The oracle must reach 1.0 through the real numeric path

`solution/solve.sh` must score `1.0` by actually running the analysis the grader
scores — not via an exact-string-match shortcut. Do **not** write
`if submission == oracle_design: return 1.0`. Such a fast-path hides the fact
that the oracle, run through the real formula, may not score `1.0` (an inverted
band, the wrong reference modes, etc.), and it makes `1.0` unreachable by any
honest engineering solution. If your oracle does not score ~`1.0` through the
numeric path, the **calibration is wrong** — fix the tolerances, bands, or
reference values, not the comparison.

### 9.4 Doing nothing (or copying the example) must score ~0

The validator grades an empty/no-op submission and, when extractable, the
prompt's own example, and fails the task if either scores above
`[ground_truth].max_trivial_score` (default `0.5`). Guardrail subscores (cost,
convergence, "didn't make it worse") must be **gated on non-trivial progress**
so a `{"devices": []}` submission cannot collect them. If a no-op out-scores
real attempts, your dominant criterion is inverted — recalibrate it.

### 9.5 Prefer numeric criteria; never reward sentiment over correctness

Text/keyword criteria (`"swapped alpha beta" in audit_text`) are satisfiable by
keyword stuffing. Tie credit to the numeric result: award "defect identified"
only when the corrected value the model reports is near-correct. Never score a
conclusion on positive sentiment ("viable/safe/proceed") or penalize an honest
negative verdict ("unsafe/unacceptable") — an honest negative engineering
judgment must be able to earn full credit. The advisory
`uv run lbx-rl-template lint-reward-hacks` flags these smells.

### 9.6 Disclose everything the grader checks

Public data, tolerances, required output keys, sign/unit conventions, and
rounding/precision requirements must all appear in `instruction.md` or `data/`.
An undocumented exact tolerance (e.g. comparing to 10 significant digits) or an
undocumented required boolean key silently caps honest runs. If the grader
checks it, the prompt must state it; if the public dataset is meant to guide
calibration, it must be consistent with the hidden reference.

## 10. Env Pre-flight QA — What CI Checks Before Taiga (Mandatory)

Every task PR runs a **blocking env pre-flight QA** in trusted CI (a local
mirror of Taiga's env_linter) *before* anything is built or submitted to
Taiga. Error findings fail the `trusted-ci/grade` check and skip the Taiga
submission entirely; the findings — with suggested fixes — appear in the PR
comment under **Boreal Results — Env QA Pre-flight**. The same class of
issues, if they slip through, will be caught by Taiga's env_linter after you
tag "Ready for Customer" and will flip the problem to **Failed QA** — so fix
them here, where the loop is minutes instead of hours.

### 10.1 Your oracle must derive its answer from the public inputs

`solution/solve.sh` that just heredocs a literal JSON answer is the canonical
unsolvable-task signature: the constants encode knowledge of the hidden spec,
so "oracle scores 1.0" proves nothing about whether an agent can solve the
task from what it actually sees. **Blocking.** Write the oracle as the real
computation: read `/data/*.json`, build the model (OpenSeesPy or otherwise),
solve, and write the result. If you cannot write such an oracle, the task is
unsolvable as specified — disclose the missing measurement data or weaken the
hidden divergence (see 10.2).

### 10.2 Hidden grading data must be reachable from the disclosed data

If the grader scores against hidden values (as-built frequencies, true
stiffnesses, reference responses) within tight tolerances, an agent must be
able to *estimate those values* from the disclosed files. The pre-flight QA
pairs numeric keys in `scorer/data/*.json` against `data/*.json` (and
`true_*` vs `nominal_*` inside the hidden spec) and **blocks** when the
divergence dwarfs the grader's tightest tolerance. Real failure example: a
Rayleigh-damping task whose hidden as-built frequencies were 20–36% off
nominal while the anchors tolerated 1.5–4% error — every honest attempt
flatlined at the 0.15 validity floor, and Taiga's env_linter failed the task.
Fix by either (a) adding measurement data under `data/` **and referencing it
in the task prompt**, or (b) regenerating the hidden spec as a genuinely mild
perturbation consistent with your tolerances.

### 10.3 The framing must match the hidden magnitudes

An LLM disclosure review compares the prompt narrative against the actual
hidden data. "Construction tolerances and in-service degradation" describing
a 67% stiffness loss is a misleading-framing error. Say what the agent should
expect ("frequencies may differ from nominal by up to X%") or supply data
that reveals it.

### 10.4 No subscore floor earnable by junk

Validity/guardrail subscores must verify *structure*, not a single scalar a
constants-only submission can hit (e.g. accepting any pair with
`sqrt(alpha*beta) ~ target`). Constrain the implied physics (anchor
frequencies in a plausible band), or gate the floor on evidence of real work.

### 10.5 Where you see the results

- PR comment, section **Boreal Results — Env QA Pre-flight**: every finding
  with severity, details, and a suggested fix.
- Blocking (error) findings: the `env_preflight` stage fails, Taiga
  submission is skipped.
- Advisory (warning/info) pre-flight findings: the task proceeds to Taiga, and
  the findings are echoed under **Boreal Results** when the Boreal run reports.
- Taiga's own agentic QA findings (`env_linter`, reward-hacking detection):
  these are triggered/synced after Boreal rollouts complete and render under
  **Taiga QA Findings**. Budget 1.5-3 hours from PR dispatch on a fresh run;
  absence of the section means pending, not clean.

Run it locally before pushing (from the mothership checkout):

```bash
python3 .github/scripts/env_preflight_qa.py \
  --problem-dir problems/<your-task> --skip-llm
```

(drop `--skip-llm` with `ANTHROPIC_API_KEY` set to get the full review).
