"""Deterministic migrations from retired task layouts."""

from alignerr_plugin.migrations.mujoco import (
    MujocoMigrationResult,
    migrate_legacy_mujoco_task,
)

__all__ = ["MujocoMigrationResult", "migrate_legacy_mujoco_task"]
