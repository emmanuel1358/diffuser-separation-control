from __future__ import annotations

import math

import mujoco


def initialize(model: mujoco.MjModel, data: mujoco.MjData) -> None:
    mujoco.mj_resetData(model, data)
    if model.nq:
        data.qpos[0] = math.pi / 2
    mujoco.mj_forward(model, data)
