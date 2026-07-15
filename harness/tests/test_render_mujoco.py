from __future__ import annotations

from pathlib import Path

import mujoco
import numpy as np
import pytest

from lbx_rl_tasks_harness.render_mujoco import (
    apply_action,
    build_observation,
    load_policy,
)


def _model() -> mujoco.MjModel:
    return mujoco.MjModel.from_xml_string("""
<mujoco>
  <worldbody>
    <body name="body">
      <joint name="slide" type="slide" axis="1 0 0"/>
      <geom type="sphere" size="0.05" mass="1"/>
    </body>
  </worldbody>
  <actuator>
    <motor joint="slide" ctrlrange="-2 2" ctrllimited="true"/>
  </actuator>
</mujoco>
""")


def _two_actuator_model() -> mujoco.MjModel:
    return mujoco.MjModel.from_xml_string("""
<mujoco>
  <worldbody>
    <body name="body">
      <joint name="x" type="slide" axis="1 0 0"/>
      <joint name="y" type="slide" axis="0 1 0"/>
      <geom type="sphere" size="0.05" mass="1"/>
    </body>
  </worldbody>
  <actuator>
    <motor joint="x" ctrlrange="-2 2" ctrllimited="true"/>
    <motor joint="y" ctrlrange="-3 3" ctrllimited="true"/>
  </actuator>
</mujoco>
""")


def test_build_observation_contains_standard_fields() -> None:
    model = _model()
    data = mujoco.MjData(model)

    obs = build_observation(model, data, step=3)

    assert obs["step"] == 3
    assert obs["time"] == 0.0
    assert isinstance(obs["qpos"], np.ndarray)
    assert isinstance(obs["qvel"], np.ndarray)
    assert obs["nu"] == 1


def test_apply_action_clips_to_ctrlrange() -> None:
    model = _model()
    data = mujoco.MjData(model)

    apply_action(model, data, [5.0])

    assert data.ctrl[0] == 2.0


def test_apply_action_broadcasts_scalar_to_all_actuators() -> None:
    model = _two_actuator_model()
    data = mujoco.MjData(model)

    apply_action(model, data, 5.0)

    assert data.ctrl.tolist() == [2.0, 3.0]


def test_load_policy_accepts_function_act(tmp_path: Path) -> None:
    policy_path = tmp_path / "policy.py"
    policy_path.write_text("def act(obs):\n    return 1.0\n")

    policy = load_policy(policy_path)

    assert policy.act({}) == 1.0


def test_load_policy_accepts_policy_class(tmp_path: Path) -> None:
    policy_path = tmp_path / "policy.py"
    policy_path.write_text(
        "class Policy:\n"
        "    def __init__(self):\n"
        "        self.did_reset = False\n"
        "    def reset(self, seed=None, metadata=None):\n"
        "        self.did_reset = True\n"
        "    def act(self, obs):\n"
        "        return [0.0]\n"
    )

    policy = load_policy(policy_path)

    assert policy.act({}) == [0.0]


def test_load_policy_rejects_missing_act(tmp_path: Path) -> None:
    policy_path = tmp_path / "policy.py"
    policy_path.write_text("VALUE = 1\n")

    with pytest.raises(TypeError):
        load_policy(policy_path)
