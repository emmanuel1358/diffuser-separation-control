"""Deterministic MuJoCo grader for a simple pendulum model.

Authored with the in-image `RubricBuilder` so per-criterion subscores
flow into Boreal UI (via Grade.metadata.structured_subscores) and
Harbor's reward.json (one flat key per criterion).

Replaces the older flat-dict shape — the headline score is identical
because RubricBuilder normalizes equal weights and there are no
penalties triggering, but Boreal UI now shows per-criterion rows
("compiled", "single_hinge", ...) instead of a single rolled-up number.
"""

from __future__ import annotations

import math
import tempfile
from pathlib import Path
from typing import Any

import mujoco
import numpy as np
from grading import (  # noqa: F401  -- helpers exposed for authors
    AgentFault,
    RubricBuilder,
    helpers,
)

# Per-criterion target tolerances, kept up top so reviewers can tune them
# without scrolling through `compute_score`. Both the criterion bodies
# below close over these values.
MASS_TARGET = 1.0
MASS_TOL = 0.05
COM_TARGET = 0.5
COM_TOL = 0.05
ROLLOUT_DURATION_SEC = 5.0


def _load_model(xml_path: Path) -> mujoco.MjModel:
    """Compile the MJCF, bouncing through a tmpfile so MuJoCo treats it as
    a real path (avoids silent caching of in-memory strings)."""
    try:
        source = xml_path.read_text()
    except OSError as exc:
        # A directory/FIFO planted at model.xml is an agent fault, not infra:
        # signal it as a typed AgentFault (kept 0.0) rather than letting the bare
        # OSError escape compute_score and discard the score the agent earned.
        raise AgentFault(f"could not read model.xml: {exc}") from exc
    with tempfile.NamedTemporaryFile("w", suffix=".xml", delete=False) as handle:
        handle.write(source)
        tmp_path = handle.name
    return mujoco.MjModel.from_xml_path(tmp_path)


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


def compute_score(
    workspace: Path, trajectory: list[dict[str, Any]] | None, private: Path
) -> dict[str, Any]:
    """Score a submitted MJCF using ten equally-weighted criteria.

    The grader compiles the model once up front and then evaluates
    structural and dynamic properties as independent criteria. A failed
    compile short-circuits the structural criteria to 0 (they all need a
    valid model) while still recording the compile error in metadata.
    """
    _ = trajectory, private

    rb = RubricBuilder(
        workspace=workspace,
        trajectory=trajectory,
        private=private,
    )

    xml_path = workspace / "model.xml"
    model: mujoco.MjModel | None = None
    compile_error: str | None = None
    rollout_stable: bool | None = None
    rollout_no_nan: bool | None = None
    moving_mass: float = 0.0
    com_length: float = 0.0
    hinge_count: int = 0
    has_jointpos: bool = False
    has_jointvel: bool = False

    # Compile once. Each criterion below references the resulting model
    # via closure, returning 0 when compile failed (so a single bad MJCF
    # cleanly fails every structural check instead of crashing).
    if xml_path.exists():
        try:
            model = _load_model(xml_path)
        except AgentFault:
            # A planted dir/FIFO at model.xml is a kept agent fault: let it
            # propagate so the runtime records a real 0.0, not a compile error.
            raise
        except Exception as exc:  # noqa: BLE001
            compile_error = str(exc)

    if model is not None:
        hinge_count = sum(
            int(model.jnt_type[i]) == mujoco.mjtJoint.mjJNT_HINGE
            for i in range(model.njnt)
        )
        has_jointpos = _sensor_type_present(model, mujoco.mjtSensor.mjSENS_JOINTPOS)
        has_jointvel = _sensor_type_present(model, mujoco.mjtSensor.mjSENS_JOINTVEL)
        moving_mass = float(model.body_mass[1:].sum()) if model.nbody > 1 else 0.0
        if model.nbody > 1:
            com = np.asarray(model.body_ipos[1])
        else:
            com = np.zeros(3)
        com_length = float(np.linalg.norm(com))
        rollout_stable, rollout_no_nan = _rollout_is_stable(model)

    # ── Criteria ─────────────────────────────────────────────────
    # Equal-weighted (0.1 each after normalization) so the headline
    # score matches the legacy implementation byte-for-byte on the
    # reference solution.

    @rb.criterion(
        id="compiled",
        weight=1.0,
        description="MJCF parses and MuJoCo compiles it without error",
    )
    def _():
        return model is not None

    @rb.criterion(id="single_hinge", weight=1.0, description="Exactly one hinge joint")
    def _():
        return model is not None and hinge_count == 1

    @rb.criterion(
        id="single_dof",
        weight=1.0,
        description="Exactly one degree of freedom (nv == 1)",
    )
    def _():
        return model is not None and model.nv == 1

    @rb.criterion(
        id="moving_body_count",
        weight=1.0,
        description="Exactly one moving body (nbody == 2 incl. world)",
    )
    def _():
        return model is not None and model.nbody == 2

    @rb.criterion(
        id="mass_target",
        weight=1.0,
        description=f"Total moving mass within {MASS_TOL} of {MASS_TARGET} kg",
    )
    def _():
        if model is None:
            return 0.0
        if abs(moving_mass - MASS_TARGET) <= MASS_TOL:
            return 1.0
        return helpers.abs_error(moving_mass, MASS_TARGET, tolerance=MASS_TARGET)

    @rb.criterion(
        id="com_length_target",
        weight=1.0,
        description=f"Center-of-mass {COM_TARGET} m from hinge axis (±{COM_TOL})",
    )
    def _():
        if model is None:
            return 0.0
        if abs(com_length - COM_TARGET) <= COM_TOL:
            return 1.0
        return helpers.abs_error(com_length, COM_TARGET, tolerance=COM_TARGET)

    @rb.criterion(
        id="jointpos_sensor", weight=1.0, description="MJCF declares a jointpos sensor"
    )
    def _():
        return has_jointpos

    @rb.criterion(
        id="jointvel_sensor", weight=1.0, description="MJCF declares a jointvel sensor"
    )
    def _():
        return has_jointvel

    @rb.criterion(
        id="stable_rollout",
        weight=1.0,
        description="5s rollout from qpos=pi/2 stays in [-pi, pi]",
    )
    def _():
        return bool(rollout_stable)

    @rb.criterion(
        id="no_nan",
        weight=1.0,
        description="Rollout produces finite qpos / qvel throughout",
    )
    def _():
        return bool(rollout_no_nan)

    if compile_error is not None:
        rb.metadata["compile_error"] = compile_error

    return rb.grade().to_dict()
