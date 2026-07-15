from __future__ import annotations

from pathlib import Path

from alignerr_plugin.local_runtime import LocalBaseImage

from lbx_rl_tasks_harness import docker
from lbx_rl_tasks_harness.models import HarnessProblem, OutputSpec


def test_build_task_image_writes_build_proof(monkeypatch, tmp_path: Path) -> None:
    repo_root = tmp_path / "repo"
    problem_dir = repo_root / "problems" / "demo-task"
    (repo_root / "grader").mkdir(parents=True)
    (repo_root / "pyproject.toml").write_text("[project]\nname = 'demo'\n")
    (problem_dir / "environment").mkdir(parents=True)
    (problem_dir / "environment" / "Dockerfile").write_text("FROM scratch\n")

    base = LocalBaseImage(
        image="lbx-tasks-base",
        tag="runtime-ml-core-py313-local",
        dockerfile=Path("base/cpu/Dockerfile"),
    )
    monkeypatch.setattr(docker, "ensure_local_base_image", lambda *a, **k: base)
    monkeypatch.setattr(docker.time, "time", lambda: 1234.0)
    monotonic_values = iter([10.0, 12.5])
    monkeypatch.setattr(docker.time, "monotonic", lambda: next(monotonic_values))

    docker_calls: list[list[str]] = []

    def fake_docker(args: list[str], *, timeout: int = 1200):  # noqa: ANN202
        docker_calls.append(args)
        iidfile = Path(args[args.index("--iidfile") + 1])
        iidfile.write_text("sha256:demo")

    monkeypatch.setattr(docker, "_docker", fake_docker)

    proofs: list[dict[str, object]] = []

    def fake_write_build_proof(problem_dir_arg: Path, **kwargs: object) -> Path:
        proofs.append({"problem_dir": problem_dir_arg, **kwargs})
        return problem_dir_arg / ".alignerr" / "build_proof.json"

    monkeypatch.setattr(docker, "write_build_proof", fake_write_build_proof)

    problem = HarnessProblem(
        id="demo-task",
        source_format="problem-dir",
        prompt="demo",
        outputs=[OutputSpec(path="/tmp/output/result.txt")],
        source_problem_dir=problem_dir,
    )

    image_tag = docker.build_task_image(problem).image_tag

    assert image_tag == "lbx-rl-harness-demo-task:1234"
    assert docker_calls == [
        [
            "build",
            "--platform",
            "linux/amd64",
            "--file",
            str(problem_dir / "environment" / "Dockerfile"),
            "--build-arg",
            "BASE_IMAGE=lbx-tasks-base",
            "--build-arg",
            "BASE_TAG=runtime-ml-core-py313-local",
            "--build-arg",
            "PROBLEM_DIR=problems/demo-task",
            "--tag",
            "lbx-rl-harness-demo-task:1234",
            "--iidfile",
            str(problem_dir / ".alignerr" / "image.iid"),
            ".",
        ]
    ]
    assert proofs == [
        {
            "problem_dir": problem_dir,
            "image_digest": "sha256:demo",
            "base_image_ref": "lbx-tasks-base:runtime-ml-core-py313-local",
            "platform": "linux/amd64",
            "alignerr_cli_version": "0.1.0",
            "duration_seconds": 2.5,
        }
    ]


def test_build_task_image_skips_build_proof_when_disabled(
    monkeypatch, tmp_path: Path
) -> None:
    repo_root = tmp_path / "repo"
    problem_dir = repo_root / "problems" / "demo-task"
    (repo_root / "grader").mkdir(parents=True)
    (repo_root / "pyproject.toml").write_text("[project]\nname = 'demo'\n")
    (problem_dir / "environment").mkdir(parents=True)
    (problem_dir / "environment" / "Dockerfile").write_text("FROM scratch\n")

    base = LocalBaseImage(
        image="lbx-tasks-base",
        tag="runtime-ml-core-py313-local",
        dockerfile=Path("base/cpu/Dockerfile"),
    )
    monkeypatch.setattr(docker, "ensure_local_base_image", lambda *a, **k: base)
    monkeypatch.setattr(docker.time, "time", lambda: 1234.0)
    monotonic_values = iter([10.0, 12.5])
    monkeypatch.setattr(docker.time, "monotonic", lambda: next(monotonic_values))

    def fake_docker(args: list[str], *, timeout: int = 1200):  # noqa: ANN202
        iidfile = Path(args[args.index("--iidfile") + 1])
        iidfile.write_text("sha256:demo")

    monkeypatch.setattr(docker, "_docker", fake_docker)

    proofs: list[dict[str, object]] = []
    monkeypatch.setattr(
        docker,
        "write_build_proof",
        lambda problem_dir_arg, **kwargs: proofs.append(kwargs) or problem_dir_arg,
    )

    problem = HarnessProblem(
        id="demo-task",
        source_format="problem-dir",
        prompt="demo",
        outputs=[OutputSpec(path="/tmp/output/result.txt")],
        source_problem_dir=problem_dir,
    )

    docker.build_task_image(problem, write_proof=False)

    assert proofs == []


def test_start_task_container_isolates_network(monkeypatch) -> None:
    import subprocess

    captured: dict[str, list[str]] = {}

    def fake_docker(args: list[str], *, timeout: int = 1200):  # noqa: ANN202
        captured["args"] = args
        return subprocess.CompletedProcess(args, 0, stdout="cid123\n", stderr="")

    monkeypatch.setattr(docker, "_docker", fake_docker)
    monkeypatch.setattr(docker, "patch_rubric_stdout_noise", lambda cid: None)

    started = docker.start_task_container("img:tag")

    assert started.container_id == "cid123"
    args = captured["args"]
    # The agent's code execution must run with no internet (Taiga parity); the
    # model still runs on the host, driving the container over docker exec.
    assert "--network" in args
    assert args[args.index("--network") + 1] == "none"
