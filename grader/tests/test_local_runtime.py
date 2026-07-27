from __future__ import annotations

import subprocess
from pathlib import Path

import pytest
from _fixture_guard import requires_examples

from alignerr_plugin import local_runtime
from alignerr_plugin.base_image import base_drift_hash
from alignerr_plugin.exporters.taiga import derive_taiga_resources
from alignerr_plugin.local_runtime import (
    ALLOW_STALE_BASE_ENV,
    LOCAL_BASE_DRIFT_LABEL,
    LOCAL_BASE_TAG,
    LOCAL_CPU_BASE_IMAGE,
    ensure_local_base_image,
    local_base_image_for_problem,
)

ROOT = Path(__file__).resolve().parents[2]
MUJOCO = ROOT / "examples" / "mujoco-pendulum"


def _fake_docker(monkeypatch, calls: list[list[str]], *, cached_label: str | None):
    """Stub docker so ``image inspect`` reports ``cached_label`` (None = uncached)."""
    monkeypatch.setattr(local_runtime.shutil, "which", lambda name: "/usr/bin/docker")
    monkeypatch.delenv(ALLOW_STALE_BASE_ENV, raising=False)

    def fake_run(args, **kwargs):  # noqa: ANN001, ANN202 - mirrors subprocess.run
        calls.append(args)
        if args[:3] == ["docker", "image", "inspect"]:
            if cached_label is None:
                return subprocess.CompletedProcess(args, 1, stdout="", stderr="")
            rendered = cached_label or "<no value>"
            return subprocess.CompletedProcess(args, 0, stdout=rendered + "\n")
        return subprocess.CompletedProcess(args, 0, stdout="", stderr="")

    monkeypatch.setattr(local_runtime.subprocess, "run", fake_run)

    class FakePopen:
        """The base build streams via Popen; fake it as immediately succeeding."""

        def __init__(self, args, **kwargs):  # noqa: ANN001, ANN204
            calls.append(args)
            self.stdout = iter(())  # no build output

        def wait(self, timeout=None):  # noqa: ANN001, ANN201
            return 0

        def kill(self):  # noqa: ANN201
            pass

    monkeypatch.setattr(local_runtime.subprocess, "Popen", FakePopen)


def _build_calls(calls: list[list[str]]) -> list[list[str]]:
    return [call for call in calls if call[:2] == ["docker", "build"]]


@requires_examples("mujoco-pendulum")
def test_local_base_ref_is_repo_local() -> None:
    base = local_base_image_for_problem(MUJOCO)

    assert base.image == LOCAL_CPU_BASE_IMAGE
    assert base.tag == LOCAL_BASE_TAG
    assert base.ref == f"{LOCAL_CPU_BASE_IMAGE}:{LOCAL_BASE_TAG}"
    assert "docker.pkg.dev" not in base.ref


@requires_examples("mujoco-pendulum")
def test_taiga_resources_still_use_production_base() -> None:
    resources = derive_taiga_resources(MUJOCO)

    assert resources["base_image"].endswith("/lbx-tasks-base")
    assert "docker.pkg.dev" in resources["base_image"]


@requires_examples("mujoco-pendulum")
def test_ensure_local_base_builds_missing_image(monkeypatch) -> None:
    calls: list[list[str]] = []
    _fake_docker(monkeypatch, calls, cached_label=None)

    base = ensure_local_base_image(ROOT, MUJOCO)

    assert base.ref == f"{LOCAL_CPU_BASE_IMAGE}:{LOCAL_BASE_TAG}"
    assert calls[0][:3] == ["docker", "image", "inspect"]
    assert base.ref in calls[0]
    build = _build_calls(calls)[0]
    assert build[:7] == [
        "docker",
        "build",
        "--progress",
        "plain",
        "--platform",
        "linux/amd64",
        "--file",
    ]
    assert str(ROOT / "base/cpu/Dockerfile") in build
    assert base.ref in build
    assert all("docker.pkg.dev" not in part for call in calls for part in call)


@requires_examples("mujoco-pendulum")
def test_ensure_local_base_labels_the_build_with_the_drift_hash(monkeypatch) -> None:
    calls: list[list[str]] = []
    _fake_docker(monkeypatch, calls, cached_label=None)

    ensure_local_base_image(ROOT, MUJOCO)

    build = _build_calls(calls)[0]
    assert f"{LOCAL_BASE_DRIFT_LABEL}={base_drift_hash(ROOT)}" in build


@requires_examples("mujoco-pendulum")
def test_fresh_cached_base_is_reused_without_rebuilding(monkeypatch, capsys) -> None:
    calls: list[list[str]] = []
    _fake_docker(monkeypatch, calls, cached_label=base_drift_hash(ROOT))

    base = ensure_local_base_image(ROOT, MUJOCO)

    assert base.ref == f"{LOCAL_CPU_BASE_IMAGE}:{LOCAL_BASE_TAG}"
    assert _build_calls(calls) == []
    assert "stale" not in capsys.readouterr().out


@requires_examples("mujoco-pendulum")
@pytest.mark.parametrize(
    ("cached_label", "expected_reason"),
    [
        ("0" * 12, "different base/ revision"),
        ("", "predates base-image drift labelling"),
    ],
)
def test_stale_cached_base_is_rebuilt_with_an_explanation(
    monkeypatch, capsys, cached_label: str, expected_reason: str
) -> None:
    calls: list[list[str]] = []
    _fake_docker(monkeypatch, calls, cached_label=cached_label)

    ensure_local_base_image(ROOT, MUJOCO)

    assert len(_build_calls(calls)) == 1
    out = capsys.readouterr().out
    assert "is stale" in out
    assert expected_reason in out
    # The author is told the cost before the rebuild, and how to decline it.
    assert "16.5 GB" in out
    assert ALLOW_STALE_BASE_ENV in out


@requires_examples("mujoco-pendulum")
def test_stale_cached_base_can_be_reused_with_an_override(monkeypatch, capsys) -> None:
    calls: list[list[str]] = []
    _fake_docker(monkeypatch, calls, cached_label="0" * 12)
    monkeypatch.setenv(ALLOW_STALE_BASE_ENV, "1")

    ensure_local_base_image(ROOT, MUJOCO)

    assert _build_calls(calls) == []
    out = capsys.readouterr().out
    assert "WARNING: reusing stale local base image" in out
    # Names the exact failure this check exists to stop being mysterious.
    assert "install-task-deps.sh" in out


def _fake_docker_per_image(monkeypatch, calls: list[list[str]], labels: dict) -> None:
    """Stub docker with a different cached drift label per image ref.

    ``labels`` maps an image ref to its label; a ref that is absent stands for
    an image that is not cached at all.
    """
    monkeypatch.setattr(local_runtime.shutil, "which", lambda name: "/usr/bin/docker")

    def fake_run(args, **kwargs):
        calls.append(args)
        if args[:3] == ["docker", "image", "inspect"]:
            ref = args[-1]
            if ref not in labels:
                return subprocess.CompletedProcess(args, 1, stdout="", stderr="")
            return subprocess.CompletedProcess(
                args, 0, stdout=(labels[ref] or "<no value>") + "\n"
            )
        return subprocess.CompletedProcess(args, 0, stdout="", stderr="")

    monkeypatch.setattr(local_runtime.subprocess, "run", fake_run)

    class FakePopen:
        def __init__(self, args, **kwargs):
            calls.append(args)
            self.stdout = iter(())

        def wait(self, timeout=None):
            return 0

        def kill(self):
            pass

    monkeypatch.setattr(local_runtime.subprocess, "Popen", FakePopen)


def _drift_label_of(build_call: list[str]) -> str:
    prefix = f"{LOCAL_BASE_DRIFT_LABEL}="
    return next(part[len(prefix) :] for part in build_call if part.startswith(prefix))


def test_an_overlay_built_on_a_stale_parent_inherits_the_stale_hash(
    monkeypatch,
) -> None:
    """One override must not permanently disarm the drift check for an overlay.

    Building the overlay against stale parent layers and then stamping it with
    the *current* hash would launder the staleness: every later run would see a
    current label and reuse an image that was never built from current `base/`.
    """
    overlay = local_runtime.LocalBaseImage(
        image=local_runtime.LOCAL_GPU_OPENROAD_BASE_IMAGE,
        tag=LOCAL_BASE_TAG,
        dockerfile=Path("base/gpu-openroad/Dockerfile"),
    )
    parent_ref = f"{local_runtime.LOCAL_GPU_BASE_IMAGE}:{LOCAL_BASE_TAG}"
    calls: list[list[str]] = []
    # Overlay absent, parent present but stale, override set.
    _fake_docker_per_image(monkeypatch, calls, {parent_ref: "0" * 12})
    monkeypatch.setenv(ALLOW_STALE_BASE_ENV, "1")

    monkeypatch.setattr(
        local_runtime, "local_base_image_for_problem", lambda problem_dir: overlay
    )

    drift = base_drift_hash(ROOT)
    ensure_local_base_image(ROOT, MUJOCO)

    # The stale parent was reused, and only the overlay was built...
    builds = _build_calls(calls)
    assert len(builds) == 1
    assert overlay.ref in builds[0]
    # ...carrying the parent's stale hash, so the next run still rebuilds it.
    assert _drift_label_of(builds[0]) == "0" * 12 != drift


def test_an_overlay_on_a_fresh_parent_is_labelled_current(monkeypatch) -> None:
    overlay = local_runtime.LocalBaseImage(
        image=local_runtime.LOCAL_GPU_OPENROAD_BASE_IMAGE,
        tag=LOCAL_BASE_TAG,
        dockerfile=Path("base/gpu-openroad/Dockerfile"),
    )
    parent_ref = f"{local_runtime.LOCAL_GPU_BASE_IMAGE}:{LOCAL_BASE_TAG}"
    drift = base_drift_hash(ROOT)
    calls: list[list[str]] = []
    _fake_docker_per_image(monkeypatch, calls, {parent_ref: drift})
    monkeypatch.delenv(ALLOW_STALE_BASE_ENV, raising=False)

    assert local_runtime._ensure_parent_base_image(ROOT, overlay, drift) == drift


def test_a_rebuilt_parent_chain_labels_every_image_with_the_current_hash(
    monkeypatch,
) -> None:
    """Nothing cached anywhere: the whole chain is current by construction."""
    overlay = local_runtime.LocalBaseImage(
        image=local_runtime.LOCAL_GPU_OPENROAD_BASE_IMAGE,
        tag=LOCAL_BASE_TAG,
        dockerfile=Path("base/gpu-openroad/Dockerfile"),
    )
    drift = base_drift_hash(ROOT)
    calls: list[list[str]] = []
    _fake_docker_per_image(monkeypatch, calls, {})
    monkeypatch.delenv(ALLOW_STALE_BASE_ENV, raising=False)

    inherited = local_runtime._ensure_parent_base_image(ROOT, overlay, drift)

    assert inherited == drift
    assert [_drift_label_of(call) for call in _build_calls(calls)] == [drift]
