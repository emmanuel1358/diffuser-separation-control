"""Deterministic MuJoCo grader for a simple pendulum model.

Authored as a declarative `RubricTask` so per-criterion subscores flow into
Boreal UI (via Grade.metadata.structured_subscores) and Harbor's reward.json
(one flat key per criterion). Equal weights and no penalties keep the
headline score identical to a flat dict; Boreal UI shows per-criterion rows
("compiled", "single_hinge", ...) instead of a single rolled-up number.
"""

from __future__ import annotations

import math
import mujoco
import numpy as np
from grading import helpers
from grading.evaluation import (
    RubricCriterion,
    RubricTask,
    TextArtifact,
)

# Per-criterion target tolerances, kept up top so reviewers can tune them
# without scrolling through `compute_score`. Both the criterion bodies
# below close over these values.
MASS_TARGET = 1.0
MASS_TOL = 0.05
COM_TARGET = 0.5
COM_TOL = 0.05
ROLLOUT_DURATION_SEC = 5.0


def _load_model(source: str) -> mujoco.MjModel:
    return mujoco.MjModel.from_xml_string(source)


def _sensor_type_present(model: mujoco.MjModel, sensor_type: int) -> bool:
    return any(int(model.sensor_type[i]) == sensor_type for i in range(model.nsensor))


def _rollout_is_stable(model: mujoco.MjModel) -> tuple[bool, bool]:
    """Run a 5-second swing from qpos=pi/2 and report (stable, no_nan).

    Stable = no |qpos| > pi excursion. no_nan = qpos/qvel stay finite.
    """
    data = mujoco.MjData(model)
    mujoco.mj_resetData(model, data)
    if model.nq:
        data.qpos[0] = math.pi / 2
    mujoco.mj_forward(model, data)
    stable = True
    no_nan = True
    steps = int(ROLLOUT_DURATION_SEC / max(model.opt.timestep, 1e-4))
    for _ in range(steps):
        mujoco.mj_step(model, data)
        if not (np.isfinite(data.qpos).all() and np.isfinite(data.qvel).all()):
            no_nan = False
            stable = False
            break
        if model.nq and abs(float(data.qpos[0])) > math.pi:
            stable = False
    return stable, no_nan


def evaluate(context):
    model = context.candidate_operation(
        "MJCF compilation",
        _load_model,
        context.candidate,
    )
    hinge_count = sum(
        int(model.jnt_type[i]) == mujoco.mjtJoint.mjJNT_HINGE for i in range(model.njnt)
    )
    has_jointpos = _sensor_type_present(model, mujoco.mjtSensor.mjSENS_JOINTPOS)
    has_jointvel = _sensor_type_present(model, mujoco.mjtSensor.mjSENS_JOINTVEL)
    moving_mass = float(model.body_mass[1:].sum()) if model.nbody > 1 else 0.0
    com = np.asarray(model.body_ipos[1]) if model.nbody > 1 else np.zeros(3)
    com_length = float(np.linalg.norm(com))
    rollout_stable, rollout_no_nan = context.candidate_operation(
        "MuJoCo rollout",
        _rollout_is_stable,
        model,
    )

    mass_score = (
        1.0
        if abs(moving_mass - MASS_TARGET) <= MASS_TOL
        else helpers.abs_error(moving_mass, MASS_TARGET, tolerance=MASS_TARGET)
    )
    com_score = (
        1.0
        if abs(com_length - COM_TARGET) <= COM_TOL
        else helpers.abs_error(com_length, COM_TARGET, tolerance=COM_TARGET)
    )
    return {
        "compiled": 1.0,
        "single_hinge": hinge_count == 1,
        "single_dof": model.nv == 1,
        "moving_body_count": model.nbody == 2,
        "mass_target": mass_score,
        "com_length_target": com_score,
        "jointpos_sensor": has_jointpos,
        "jointvel_sensor": has_jointvel,
        "stable_rollout": rollout_stable,
        "no_nan": rollout_no_nan,
    }


TASK = RubricTask(
    artifact=TextArtifact("model.xml", max_bytes=4 * 1024 * 1024),
    criteria=(
        RubricCriterion(
            "compiled",
            description="MJCF parses and MuJoCo compiles it without error",
            required=True,
        ),
        RubricCriterion("single_hinge", description="Exactly one hinge joint"),
        RubricCriterion(
            "single_dof",
            description="Exactly one degree of freedom (nv == 1)",
        ),
        RubricCriterion(
            "moving_body_count",
            description="Exactly one moving body (nbody == 2 incl. world)",
        ),
        RubricCriterion(
            "mass_target",
            description=f"Total moving mass within {MASS_TOL} of {MASS_TARGET} kg",
        ),
        RubricCriterion(
            "com_length_target",
            description=f"Center-of-mass {COM_TARGET} m from hinge axis (+/-{COM_TOL})",
        ),
        RubricCriterion(
            "jointpos_sensor",
            description="MJCF declares a jointpos sensor",
        ),
        RubricCriterion(
            "jointvel_sensor",
            description="MJCF declares a jointvel sensor",
        ),
        RubricCriterion(
            "stable_rollout",
            description="5s rollout from qpos=pi/2 stays in [-pi, pi]",
        ),
        RubricCriterion(
            "no_nan",
            description="Rollout produces finite qpos / qvel throughout",
        ),
    ),
    evaluate=evaluate,
)
