# MuJoCo Task Design Guidelines for LLM RL Training

A practical guide for robotics experts designing MuJoCo tasks for the Alignerr RL tasks pipeline.

## Why We Are Doing This

LLMs have made large gains in math and code through **RLVR**: Reinforcement Learning with Verifiable Rewards. The recipe is simple:

1. Pose a problem.
2. Let the model attempt it.
3. Run a programmatic checker.
4. Reward only what can be verified.

This is the same broad paradigm behind RL-trained reasoning systems. It works because the reward signal is deterministic, scalable, and unambiguous.

We are applying this to MuJoCo. The bet is that if we train LLMs on enough verifiable simulation tasks, including morphology design, MJCF debugging, controller authoring, reward shaping, and spec matching, they will become materially better at robotics work that researchers actually do.

Your role is to design those tasks.

Recent work points in this direction. LLM systems have been used to generate simulation environments, synthesize rewards, and co-design morphology and control. The gap we are filling is **RL training over verifiable robotics tasks at scale**.

## How Tasks Fit Our Repository

All tasks in this repo use one universal layout. There is no task-type enum.

Start from the MuJoCo starter:

```bash
mkdir -p problems
cp -R alignerr_plugin/src/alignerr_plugin/starter_templates/mujoco problems/<task-name>
```

The resulting task should look like:

```text
problems/<task-id>/
├── metadata.json
├── task.toml
├── instruction.md
├── environment/
│   └── Dockerfile
├── scorer/
│   ├── compute_score.py
│   └── data/                    # hidden MJCF assets, seeds, expected metrics
├── solution/
│   └── solve.sh                 # creates a reference /tmp/output artifact
├── baselines/
│   └── naive.sh                 # creates a low-scoring baseline
├── data/                        # public examples, stubs, starter MJCF
└── README.md
```

The LLM only sees public task materials and writes outputs under `/tmp/output`. The grader runs inside the task Docker image and reads hidden fixtures from `/mcp_server/data`.

**Important:** final agent artifacts must be written under `/tmp/output`, not `/workspace`. Use paths such as `/tmp/output/model.xml` or `/tmp/output/policy.py` in prompts, `task.toml`, reference solutions, and graders.

Before submission, you must run:

```bash
uv run lbx-rl-harness run --runtime ground-truth --problem-dir problems/<task-id>
```

The ground-truth runtime checks the reference output without calling an LLM and
records the reviewer render artifacts required by trusted CI. Open or update your
fork PR when ready; trusted CI uploads artifacts, runs the agent harness, rubric
QA, and Auto QA, then posts `trusted-ci/grade`.

### GPU Requirement for Policy and Model Training Tasks

Policy-training and model-training tasks must use GPU acceleration. If a task
asks the agent to train, fine-tune, tune, adapt, or optimize a learned
policy/model, it must declare GPU resources in `task.toml` and be designed so
accelerator use is genuinely necessary.

Do not make a CPU-style task look GPU-based by only selecting an H100 resource
enum. A GPU task should involve meaningful training complexity: batched rollouts,
large scenario variation, GPU-backed neural policy training, checkpoint export,
hyperparameter tuning, large datasets, or another workload where accelerator use
is justified. GPUs may also be requested for simulation or evaluation when the
task genuinely needs CUDA. Set the request in `task.toml`:

```toml
[environment]
required_resources = "12vcpu+100gib+h100/2"
storage_mb = 50000
```

When `required_resources` is an H100 enum, the local harness builds the
repo-local GPU base image (`lbx-tasks-base-gpu:runtime-ml-core-py313-local`),
and mothership exports the task with the production GPU base image. Keep the
task Dockerfile generic with `ARG BASE_IMAGE`, `ARG BASE_TAG`, and
`FROM ${BASE_IMAGE}:${BASE_TAG}` so the pipeline can choose the base
automatically. Local GPU execution still requires Docker/NVIDIA GPU support on
the authoring machine.

The grader should still be deterministic. GPU may be required for training or
policy/model execution, but scoring must use fixed seeds, fixed scenarios,
bounded rollouts, and reproducible metrics.

## Mental Model: What Is a MuJoCo Task?

A task has three pieces:

1. **Instruction**: the prompt in `instruction.md`. This is what the model sees.
2. **Solution format**: typically MJCF XML, Python that builds an `MjModel`, a `policy.py`, or a combination.
3. **Scorer**: `scorer/compute_score.py`, a deterministic function that returns a score in `[0, 1]`, a score dict, or `RubricBuilder.grade().to_dict()`.

The grader is the whole game. A wrong grader teaches the wrong lesson. A gameable grader gets gamed.

## Grader Contract

Every task uses the same grader contract:

```python
from pathlib import Path
from typing import Any


def compute_score(
    workspace: Path,
    trajectory: list[dict[str, Any]] | None,
    private: Path,
) -> float | dict[str, Any]:
    ...
```

For MuJoCo tasks:

- `workspace` is where the model output lives, for example `/tmp/output/model.xml` or `/tmp/output/policy.py`.
- `private` points at `/mcp_server/data`, where hidden assets, seeds, and reference metrics live.
- `trajectory` may be unused unless the task involves agent tool traces.

Prefer `RubricBuilder` for MuJoCo rubrics because it gives one row per
deterministic criterion:

```python
from grading import RubricBuilder

def compute_score(workspace, trajectory, private):
    rb = RubricBuilder(workspace=workspace, trajectory=trajectory, private=private)

    @rb.criterion(id="compiled", weight=0.1, description="MJCF compiles")
    def _():
        return load_model(workspace / "model.xml") is not None

    return rb.grade().to_dict()
```

Before writing rubric criteria, read the shared
[rubric guidance](../docs/RUBRIC_GUIDANCE.md). MuJoCo rubrics should capture
the task-specific properties a robotics expert would evaluate, not generic
robotics preferences or one arbitrary solution path.

Do not use an LLM judge in the scorer. MuJoCo rewards must be produced by
MuJoCo compilation, model inspection, deterministic rollouts, fixed
perturbations, and numeric thresholds.

### Executable Policies

When a MuJoCo task asks for `/tmp/output/policy.py` or another executable Python
controller, do not import the submitted module directly into `compute_score.py`
while hidden cases or perturbation schedules are live. Use `PolicyWorker`:

```python
from grading import PolicyWorker

with PolicyWorker(workspace / "policy.py", timeout_s=0.1) as policy:
    action = policy.act(public_observation)
```

The grader keeps hidden state in the parent process and sends only public
observations to the policy subprocess. This prevents submitted policy code from
using Python frame inspection to read hidden grader locals. Static-output tasks
such as the checked-in `examples/mujoco-pendulum` model-XML task do not need
`PolicyWorker`.

`PolicyWorker` also supports common MuJoCo migration patterns:

- call it directly as `policy(obs)` when existing scorer code expects a callable,
- use `policy.call("method_name", ...)` for custom APIs,
- use `policy.init_model_xml((workspace / "model.xml").read_text())` when a
  submitted policy needs to load `/tmp/output/model.xml`, and
- create a fresh worker per hidden case when policy state should reset between
  cases.

See [`docs/POLICY_ISOLATION.md`](../docs/POLICY_ISOLATION.md) before authoring
or reviewing executable policy tasks.

## Three Properties of a Good Task

### Verifiable

Every criterion must be checkable by code. Avoid criteria like "is elegant" or "looks plausible" unless they are backed by deterministic checks.

### Deterministic

Same submission, same score, every time.

Pin:

- MuJoCo version,
- integrator,
- timestep,
- solver settings,
- initial `qpos` / `qvel`,
- control inputs,
- RNG seeds,
- perturbation list.

If a grader needs randomness, the seed belongs inside the grader and must be fixed.

### Multi-Criterion

Pure pass/fail wastes signal. Aim for **10+ deterministic criteria per task** across structure, statics, rollouts, and robustness. A dense rubric:

- gives smoother reward,
- makes failures diagnosable,
- reduces reward hacking,
- supports curriculum design.

## Anatomy of a 10+ Criterion Rubric

Mine criteria from four strata. Most strong tasks naturally support several from each.

### Structural Criteria

Read directly from the compiled `mujoco.MjModel`:

- `nbody`, `njnt`, `nu`, `nsensor`,
- named bodies, joints, sites, geoms, actuators, sensors,
- joint types and axes,
- parent-child topology,
- mass and inertia bounds,
- joint limits,
- actuator configuration,
- sensor presence,
- AABB bounds.

Suggested weight: `0.02-0.05` each.

### Static Criteria

Evaluate after `mujoco.mj_forward(model, data)` at a defined pose:

- center of mass position,
- foot contact,
- support polygon,
- self-collision,
- IK feasibility for target points,
- Jacobian conditioning,
- default pose stability.

Suggested weight: `0.05-0.10` each.

### Deterministic Rollout Criteria

Step with pinned initial state and controls:

- terminal state,
- oscillation period,
- damping ratio,
- settling time,
- energy conservation,
- peak values,
- contact event timing,
- forward displacement,
- fall/tumble checks.

Suggested weight: `0.10-0.20` each.

### Robustness Criteria

Run the same model under a fixed list of perturbations:

- slope angles,
- payload masses,
- initial pose offsets,
- friction multipliers,
- actuator strength multipliers,
- torso mass multipliers.

Each perturbation can be its own sub-criterion. Suggested weight: `0.05-0.10` each.

## Starter Task Examples

These get progressively harder. Use them as curriculum anchors.

### Example 1: Damped Pendulum With Target Dynamics

Prompt:

> Build a single-pendulum MJCF where total mass is `1.0 kg ± 2%`, distance from joint axis to body COM is `0.5 m ± 1%`, small-angle period is `1.42 s ± 1%`, damping ratio is `0.05 ± 5%`, and joint position/velocity sensors are present.

Why it works:

- passive dynamics only,
- no controller,
- simple physical intuition,
- failures are easy to diagnose.

Candidate rubric:

1. MJCF compiles with no errors.
2. Exactly one hinge joint with horizontal axis.
3. Exactly one moving body attached to world by that hinge.
4. No extra DOFs (`model.nv == 1`).
5. Total mass within tolerance.
6. Joint-to-COM distance within tolerance.
7. Joint position sensor present.
8. Joint velocity sensor present.
9. Period of first five oscillations within tolerance.
10. Log-decrement damping ratio within tolerance.
11. Settles to vertical down by 30 simulated seconds.
12. Energy conserved when damping is zeroed programmatically.
13. No NaNs in `qpos` / `qvel`.
14. No upward overshoot beyond initial amplitude tolerance.
15. Geom AABB fits in expected bounds.

### Example 2: Quadruped Specification

Prompt:

> Build a quadruped MJCF with one torso free joint and four identical legs, each with hip and knee hinge joints. Total mass between 5 and 15 kg. Include torso IMU and joint position sensors. The robot must stand stably under gravity from default pose.

Why it works:

- tests kinematic tree reasoning,
- rewards sensor/actuator correctness,
- static stability is verifiable without a learned controller.

Candidate rubric:

1. MJCF compiles.
2. Exactly one torso body with free joint.
3. Four leg subtrees rooted at unique torso sites.
4. Each leg has exactly two hinge joints.
5. Hip axes align as specified.
6. Knee axes parallel hip axes within tolerance.
7. Joint limits on all leg joints.
8. Total mass in range.
9. Named foot site/geoms exist.
10. Torso IMU site has gyro and accelerometer.
11. Position sensors attached to all leg joints.
12. Default foot sites are at ground height.
13. COM projection inside support polygon.
14. Passive 5-second sim remains upright.
15. Passive 5-second sim has limited drift.

### Example 3: Fixed-Controller Hopper / Morphology Design

Prompt:

> Design a robot morphology that moves forward as far as possible in 5 seconds when actuators are driven by fixed sinusoidal controls. You may choose joint phases and morphology. Total mass under 20 kg and the model must fit in a 2 m cube.

Why it works:

- separates morphology design from policy learning,
- offers a rich design space,
- preserves deterministic grading.

Candidate rubric:

1. MJCF compiles.
2. Root free joint exists.
3. At least three actuated joints.
4. Positive body masses and positive-definite inertias.
5. Joint limits on actuated joints.
6. Contact friction in bounded range.
7. Total mass and AABB within limits.
8. No-control settling is stable.
9. Fixed sinusoidal rollout moves forward.
10. COM height stays above threshold.
11. Torso does not tumble.
12. No NaNs.
13. Energy does not explode.
14. Robust under friction multipliers.
15. Robust under torso mass multipliers.

## Common Pitfalls

### Sparse Rewards

`compiled == true` is not enough. Layer reward:

```text
0.05 parsed
0.10 compiled
0.25 structural
0.25 static
0.25 rollout
0.10 robustness
```

### Degenerate Trivial Solutions

Examples:

- "stand stably" becomes a flat puck,
- "move forward" becomes a projectile,
- "stable gait" becomes a sliding block.

Add feasibility shells:

- aspect ratio bounds,
- height bounds,
- mass bounds,
- contact requirements,
- no projectile trajectories,
- minimum number of meaningful joints/bodies.

### Numerical Instability as Reward Hacking

Watch for:

- penetrating contacts,
- exploding energy,
- NaNs,
- huge `qpos` / `qvel`,
- unrealistic contact normal forces,
- solver blowups that accidentally satisfy a metric.

Treat anomalies as failures.

### Single-Condition Rollouts

Models can overfit one trajectory. Use a fixed set of perturbations and score each one separately.

### Non-Determinism Creep

Common causes:

- missing `mj_resetData`,
- unpinned RNG,
- system-time seeds,
- random IK initial guesses,
- default MuJoCo integrator changes,
- uncontrolled floating dependencies.

Always restate full initial state at the top of the grader.

### Vacuous Structural Matches

"Has four hinge joints" can be satisfied by four useless disconnected hinges. Pair structural checks with static or dynamic checks that require function.

## Practical Tips

- Use subprocess timeouts around compilation and rollout.
- Use `model.jnt_qposadr[id]` and `model.jnt_dofadr[id]` for indexing.
- Keep rollouts short; 2-5 simulated seconds is often enough.
- Provide a working starter MJCF when possible.
- Prefer edit-based tasks over pure blank-file synthesis for early curriculum.
- Test graders against good, empty, partial, and adversarial solutions.
- If using MJX for batched training, keep the grader itself deterministic and small.

## Review and Acceptance Expectations

MuJoCo tasks are expected to improve over review. There are two paid milestones:
a paid PR attempt when the submission is complete enough to validate and
produce useful training signal, and a paid accepted task when the final version
meets the stricter difficulty and SME-review bar.

### $$ Paid PR Attempt Eligibility

A PR attempt is eligible for payment when it meets all of these before review:

- The task passes trusted CI (`trusted-ci/grade`), including committed
  ground-truth proof and reviewer render artifacts, plus agent harness, rubric
  QA, Auto QA, and downstream Taiga/Boreal feedback on the fork PR.
- The task does **not** score perfectly under validation; a model score below
  `1.0` shows that the task still contains meaningful headroom.
- If the task uses `RubricBuilder`, it includes at least **5 deterministic
  criteria**. Strong MuJoCo tasks should still aim for the 10+ criterion
  structure described above.
- If the task uses a continuous reward, follow the
  [continuous reward functions from ML_Envs](../docs/GRADING.md#continuous-reward-functions-from-ml_envs)
  pattern. The scoring function should be calibrated so current model attempts
  score at or below the human/reference baseline, not near the perfect score.
  See the grading guide's
  [difficulty targets](../docs/GRADING.md#difficulty-targets) for the expected
  baseline, reference, and perfect-score anchors.

### $$ Paid Accepted Task Criteria

The accepted task milestone is eligible for payment when the final task meets
all of these:

- Current model attempts consistently score below `0.40`. Expect to iterate on
  environment complexity, hidden fixtures, perturbations, and scoring anchors
  from the initial PR to reach this difficulty.
- A robotics SME reviews the task for scientific correctness: the physics,
  morphology, controls, rollout setup, and scoring checks should be meaningful
  for the robotics concept being tested.
- The task genuinely uses MuJoCo and appropriate robotics libraries/tools, not
  only generic file parsing or superficial XML checks.
- The grader remains deterministic, reproducible, and resistant to trivial or
  physically invalid solutions.

## Required Submission Contents

For this repo, a MuJoCo task should include:

```text
problems/<task-id>/
├── instruction.md
├── task.toml
├── environment/Dockerfile
├── scorer/compute_score.py
├── scorer/data/
│   ├── assets/              # hidden MJCF/assets if needed
│   ├── seeds.json           # fixed seeds / perturbation list
│   └── expected.json        # target metrics / normalization constants
├── solution/solve.sh        # writes reference output into /tmp/output
├── solution/render.sh       # calls the shared MuJoCo renderer
├── solution/render_config.py # task-specific rollout/camera hooks
├── baselines/naive.sh       # writes weak output into /tmp/output
├── data/                    # public prompt assets / starter MJCF
└── README.md
```

A task is not ready until:

- reference solution scores high enough,
- MuJoCo ground-truth rendering uses `uv run python -m lbx_rl_tasks_harness.render_mujoco`,
- policy artifacts expose `act(obs)` or `class Policy` with `act(obs)`,
- naive baseline scores low,
- at least one partial solution fails exactly the intended criteria,
- an adversarial trivial solution is caught,
- opening or updating the fork PR triggers trusted CI, which completes full QA
  and posts `trusted-ci/grade` before reviewer merge.

## Pre-Submission Checklist

- [ ] 10+ deterministic rubric criteria.
- [ ] Every criterion is code-checkable.
- [ ] Integrator, timestep, initial state, controls, and seeds pinned.
- [ ] At least three strata represented: structural, static, rollout, robustness.
- [ ] Reward shaping present; not pure 0/1.
- [ ] Feasibility shell prevents trivial solutions.
- [ ] Numerical sanity guards included.
- [ ] Grader tested against good / empty / partial / trivial submissions.
- [ ] Docker image builds locally.
- [ ] Trusted CI (`trusted-ci/grade`) passes on the fork PR.
- [ ] Full QA (agent harness, rubric QA, Auto QA) has completed on the latest commit.

## Further Reading

- Standard rubric guidance for task-specific, measurable criteria:
  [`docs/RUBRIC_GUIDANCE.md`](../docs/RUBRIC_GUIDANCE.md).
- MuJoCo documentation, especially model structure and `MjModel` / `MjData`.
- MJX documentation for batched simulator throughput.
- MuJoCo Playground for RL environment patterns.
- Language-to-rewards and simulation-generation papers for inspiration.

Questions or task proposals should be discussed before investing in a full grader.
