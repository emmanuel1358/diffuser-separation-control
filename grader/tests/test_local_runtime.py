from __future__ import annotations

import subprocess
from pathlib import Path

from alignerr_plugin import local_runtime
from alignerr_plugin.exporters.taiga import derive_taiga_resources
from alignerr_plugin.local_runtime import (
    LOCAL_BASE_TAG,
    LOCAL_CPU_BASE_IMAGE,
    ensure_local_base_image,
    local_base_image_for_problem,
)

ROOT = Path(__file__).resolve().parents[2]
MUJOCO = ROOT / "examples" / "mujoco-pendulum"


def test_local_base_ref_is_repo_local() -> None:
    base = local_base_image_for_problem(MUJOCO)

    assert base.image == LOCAL_CPU_BASE_IMAGE
    assert base.tag == LOCAL_BASE_TAG
    assert base.ref == f"{LOCAL_CPU_BASE_IMAGE}:{LOCAL_BASE_TAG}"
    assert "docker.pkg.dev" not in base.ref


def test_taiga_resources_still_use_production_base() -> None:
    resources = derive_taiga_resources(MUJOCO)

    assert resources["base_image"].endswith("/lbx-tasks-base")
    assert "docker.pkg.dev" in resources["base_image"]


def test_ensure_local_base_builds_missing_image(monkeypatch) -> None:
    calls: list[list[str]] = []

    monkeypatch.setattr(local_runtime.shutil, "which", lambda name: "/usr/bin/docker")

    def fake_run(args, **kwargs):  # noqa: ANN001, ANN202 - mirrors subprocess.run
        calls.append(args)
        if args[:3] == ["docker", "image", "inspect"]:
            return subprocess.CompletedProcess(args, 1, stdout="", stderr="")
        return subprocess.CompletedProcess(args, 0, stdout="", stderr="")

    monkeypatch.setattr(local_runtime.subprocess, "run", fake_run)

    # The base build streams output via Popen; fake it as immediately succeeding.
    class FakePopen:
        def __init__(self, args, **kwargs):  # noqa: ANN001, ANN204
            calls.append(args)
            self.stdout = iter(())  # no build output

        def wait(self, timeout=None):  # noqa: ANN001, ANN201
            return 0

        def kill(self):  # noqa: ANN201
            pass

    monkeypatch.setattr(local_runtime.subprocess, "Popen", FakePopen)

    base = ensure_local_base_image(ROOT, MUJOCO)

    assert base.ref == f"{LOCAL_CPU_BASE_IMAGE}:{LOCAL_BASE_TAG}"
    assert calls[0] == ["docker", "image", "inspect", base.ref]
    assert calls[1][:7] == [
        "docker",
        "build",
        "--progress",
        "plain",
        "--platform",
        "linux/amd64",
        "--file",
    ]
    assert str(ROOT / "base/cpu/Dockerfile") in calls[1]
    assert base.ref in calls[1]
    assert all("docker.pkg.dev" not in part for call in calls for part in call)
