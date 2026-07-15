"""Tests for `helpers.world_integrity` — the shared MJCF physics validator.

These encode the recurring "rigged world" reward-hacks from QA: disabled or
tilted gravity, gravcomp free-floating, equality-weld slaving, and disabled
contacts. A legitimate model passes; each tampered model is flagged.
"""

from __future__ import annotations

import pytest

mujoco = pytest.importorskip("mujoco")

from grading import helpers

_CLEAN = """
<mujoco>
  <worldbody>
    <geom name="floor" type="plane" size="2 2 0.1"/>
    <body name="b" pos="0 0 1">
      <joint name="j" type="hinge" axis="0 1 0"/>
      <geom name="link" type="capsule" size="0.05 0.2"/>
    </body>
  </worldbody>
</mujoco>
"""


def _model(xml: str):
    return mujoco.MjModel.from_xml_string(xml)


def test_clean_world_passes() -> None:
    ok, violations = helpers.world_integrity(_model(_CLEAN))
    assert ok, violations
    assert violations == []


def test_tilted_gravity_is_flagged() -> None:
    xml = _CLEAN.replace("<worldbody>", '<option gravity="9.81 0 0"/><worldbody>')
    ok, violations = helpers.world_integrity(_model(xml))
    assert not ok
    assert any("gravity" in v for v in violations)


def test_disabled_gravity_is_flagged() -> None:
    xml = _CLEAN.replace("<worldbody>", '<option gravity="0 0 0"/><worldbody>')
    ok, violations = helpers.world_integrity(_model(xml))
    assert not ok
    assert any("gravity" in v for v in violations)


def test_gravcomp_is_flagged() -> None:
    xml = _CLEAN.replace('<body name="b"', '<body name="b" gravcomp="1"')
    ok, violations = helpers.world_integrity(_model(xml))
    assert not ok
    assert any("gravcomp" in v for v in violations)


def test_equality_weld_is_flagged() -> None:
    xml = _CLEAN.replace(
        "</worldbody>",
        '</worldbody><equality><weld body1="b"/></equality>',
    )
    ok, violations = helpers.world_integrity(_model(xml))
    assert not ok
    assert any("equality" in v for v in violations)


def test_disabled_contacts_is_flagged() -> None:
    xml = _CLEAN.replace(
        "<worldbody>",
        '<option><flag contact="disable"/></option><worldbody>',
    )
    ok, violations = helpers.world_integrity(_model(xml))
    assert not ok
    assert any("contact" in v for v in violations)


def test_gravity_check_can_be_skipped() -> None:
    xml = _CLEAN.replace("<worldbody>", '<option gravity="0 0 0"/><worldbody>')
    ok, violations = helpers.world_integrity(_model(xml), expect_gravity=None)
    assert ok, violations


def test_custom_gravity_target_passes() -> None:
    # A task on the Moon: expected gravity differs, model matches it.
    xml = _CLEAN.replace("<worldbody>", '<option gravity="0 0 -1.62"/><worldbody>')
    ok, violations = helpers.world_integrity(
        _model(xml), expect_gravity=(0.0, 0.0, -1.62)
    )
    assert ok, violations
