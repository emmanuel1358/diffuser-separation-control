---
name: numerical-solver-tasks
description: Author non-MuJoCo numerical-solver RL tasks (task_type cfd or structures) using the open-source solver stack baked into the base image. Use when creating tasks that need OpenFOAM, SU2, AeroSandbox/AVL, scikit-fem/PyNite/CalculiX/OpenSeesPy, or when setting [difficulty].domain.
---

# Numerical-Solver Tasks

Non-MuJoCo engineering tasks reuse the same scoring contract as other tasks
(`compute_score(workspace, trajectory, private)`, artifacts to `/tmp/output`,
enum-backed metadata, and reward-type-specific oracle targets). A numerical
solver is the deterministic referee.

References:
- `docs/NUMERICAL_SOLVERS.md` — full engine inventory + how to invoke each.
- `project_guidelines/cfd/cfd_environments.md` — CFD task guide (AVL + OpenFOAM).
- `project_guidelines/cfd/prometheus_cfd_environments.md` — non-eval Prometheus CFD guide.
- `project_guidelines/cfd/prometheus_eval_cfd_environments.md` — eval Prometheus CFD guide.
- `project_guidelines/strctural_engineering/PROMETHEUS_STRUCTURAL_ENGINEER_OPENSEES_AUTHORING.md` — non-eval Prometheus OpenSees guide.
- `project_guidelines/strctural_engineering/PROMETHEUS_EVAL_STRUCTURAL_ENGINEER_OPENSEES_AUTHORING.md` — eval Prometheus OpenSees guide.
- `examples/openfoam-hydrofoil-flap` — canonical worked example.

## task.toml

- Set `[difficulty].task_type` to `cfd` or `structures`.
- Set `[difficulty].domain` to the scoped enum (`aerodynamics` for CFD,
  `structural_mechanics` for the base-isolation structures example).
- Set `[difficulty].reward_type = "multi_deterministic_rubrics"` for the
  standard deterministic solver-rubric pattern.

## Engines in the base image

Import directly in `scorer/compute_score.py`:

| Domain | Library (import) |
| --- | --- |
| CFD / aero | `aerosandbox` |
| circuits | `PySpice` (ngspice) |
| structures | `skfem`, `Pynite`, `anastruct`, `openseespy` |
| electromagnetics | `skrf` |
| reacting flow / thermo | `cantera`, `CoolProp` |
| EDA | `klayout.db` |

Heavy engines run via subprocess (they live only in the base image):
- OpenFOAM: `source /etc/solver-envs.d/openfoam.sh` then `blockMesh`/`pimpleFoam`/etc.
- CalculiX `ccx`, `SU2_CFD` — on PATH.
- Meep: `/opt/solver-envs/meep/bin/python`.
- OpenROAD: `openroad` — opt into `base_flavor = "gpu-openroad"` with a non-graphics H100 `required_resources` enum.

Per-task installs (build in `environment/Dockerfile`): AVL, XFOIL, PyNEC, DREAMPlace.

## Determinism (load-bearing)

- Pin the solver version, run **serial**, fixed iteration counts or fixed
  `deltaT` + `endTime` (no adaptive timestep), fixed schemes/seeds.
- Mesh from a **parametric `blockMesh`** or a pre-built mesh — never
  `snappyHexMesh` at grade time. Prefer deterministic parametric geometry and fixed meshing assumptions; avoid
  adaptive remeshing or stochastic mesh generation at grade time.
- Grade on tolerances; treat divergence / NaN / non-convergence as `0` (+ penalty).
- Precompute reference values into `scorer/data/` (e.g. a no-control baseline);
  never regenerate them nondeterministically at grade time.

## Runtime budget

The grader runs the solver per hidden case sequentially. Keep
`N_cases * per_case_timeout` under `[verifier].timeout_sec`, use coarse meshes
and low/moderate Reynolds, and a small hidden sweep (2-3 conditions).

## In-container ground truth + solver evidence

A grader that calls a base-only engine cannot run on the host. Set
`[ground_truth].in_container = true` so the harness builds the task image and
runs solve + grade inside it. Keep `instruction.md` solver-agnostic, and put
solver-specific runnable material in the core solution implementation
(`solution/solve.sh` or solution files it calls). Do not require agent
transcript checks or solver-evidence artifacts; if you also declare a reviewer
video, use the generalized render contract (`rendering.mp4`, exactly `1280x720`
h264).

```toml
[difficulty]
task_type = "cfd"
domain = "aerodynamics"
reward_type = "multi_deterministic_rubrics"

[ground_truth]
in_container = true
# Add render_command/render_outputs only when the task declares a reviewer video.
```

Iterate: `uv run lbx-rl-harness reference --problem-dir problems/<task_id>`

Validate: `uv run lbx-rl-harness run --runtime ground-truth --problem-dir problems/<task_id>`
(oracle must match the declared `reward_type` target and commit the build proof; commit reviewer artifacts only when declared).
