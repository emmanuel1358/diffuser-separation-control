from __future__ import annotations

import json
import shutil
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path

from alignerr_plugin import mlenvs
from alignerr_plugin.local_runtime import ensure_local_base_image
from alignerr_plugin.proof import write_build_proof

from lbx_rl_tasks_harness.models import HarnessProblem

TAIGA_PLATFORM = "linux/amd64"

# Side-channel written by build_task_image, read by the manifest writer.
FLAVOR_SIDECAR = Path(".alignerr") / "last_build_flavor.json"


@dataclass(frozen=True)
class StartedContainer:
    image_tag: str
    container_id: str


@dataclass(frozen=True)
class TaskBuild:
    image_tag: str
    image_flavor: str  # "heavy" | "slim" -- what actually built
    flavor_requested: str  # "auto" | "heavy" | "slim"
    flavor_fallback: bool  # heavy OOMed and we fell back to slim


def _docker(
    args: list[str], *, timeout: int = 1200
) -> subprocess.CompletedProcess[str]:
    if shutil.which("docker") is None:
        raise RuntimeError("docker is required for agent harness runtimes")
    return subprocess.run(
        ["docker", *args],
        text=True,
        capture_output=True,
        timeout=timeout,
        check=True,
    )


def _repo_root(problem_dir: Path) -> Path:
    for parent in [problem_dir, *problem_dir.parents]:
        if (parent / "pyproject.toml").exists() and (parent / "grader").exists():
            return parent
    raise RuntimeError(f"could not locate repo root for {problem_dir}")


def resolve_task_build(problem_dir: Path, repo_root: Path) -> tuple[Path, list[str]]:
    """Return (dockerfile, extra_build_args) for a task.

    metadata-mode tasks build from the shared ``base/task.mlenvs.Dockerfile`` with
    ``dependencies`` / ``apt_extras`` / ``env_dependencies`` forwarded as build
    args; native tasks use their per-task ``environment/Dockerfile``.

    ``env_dependencies`` install into the root-only ``/mcp_server/env_deps`` so a
    hidden-env simulator stays server-only; ``grading_dependencies`` install into
    the root-only ``/mcp_server/grading_deps`` so a grader-only scoring/reference
    library stays invisible to the agent; ``dependencies`` install system-wide.
    """
    if mlenvs.is_mlenvs_task(problem_dir):
        dockerfile = repo_root / "base" / "task.mlenvs.Dockerfile"
        meta = mlenvs.load_mlenvs_metadata(problem_dir)
        # Bake hidden_env into task.toml so the runtime env-server gate activates
        # for env/hybrid tasks (metadata mode ships no task.toml on disk).
        hidden_env = mlenvs.hidden_env_for_task_type(meta["ml_task_type"])
        extra = [
            "--build-arg",
            f"APT_EXTRAS={' '.join(meta.get('apt_extras', []) or [])}",
            "--build-arg",
            f"DEPENDENCIES={' '.join(meta.get('dependencies', []) or [])}",
            "--build-arg",
            f"ENV_DEPENDENCIES={' '.join(meta.get('env_dependencies', []) or [])}",
            "--build-arg",
            f"GRADING_DEPENDENCIES={' '.join(meta.get('grading_dependencies', []) or [])}",
            "--build-arg",
            f"HIDDEN_ENV={hidden_env}",
        ]
        return dockerfile, extra
    return problem_dir / "environment" / "Dockerfile", []


def build_task_image(
    problem: HarnessProblem, *, flavor: str = "auto", write_proof: bool = True
) -> TaskBuild:
    if problem.source_problem_dir is None:
        raise RuntimeError(
            "agent harness runtimes require --source-problem-dir for exported formats"
        )
    problem_dir = problem.source_problem_dir.resolve()
    repo_root = _repo_root(problem_dir)
    dockerfile, extra_build_args = resolve_task_build(problem_dir, repo_root)
    if not dockerfile.exists():
        raise FileNotFoundError(f"task Dockerfile not found: {dockerfile}")

    base = ensure_local_base_image(repo_root, problem_dir, flavor=flavor)
    rel_problem_dir = problem_dir.relative_to(repo_root)
    safe_id = problem.id.replace("/", "-").replace("_", "-")
    image_tag = f"lbx-rl-harness-{safe_id}:{int(time.time())}"
    iidfile = problem_dir / ".alignerr" / "image.iid"
    iidfile.parent.mkdir(parents=True, exist_ok=True)
    start = time.monotonic()
    _docker(
        [
            "build",
            "--platform",
            TAIGA_PLATFORM,
            "--file",
            str(dockerfile),
            "--build-arg",
            f"BASE_IMAGE={base.image}",
            "--build-arg",
            f"BASE_TAG={base.tag}",
            "--build-arg",
            f"PROBLEM_DIR={rel_problem_dir}",
            *extra_build_args,
            "--tag",
            image_tag,
            "--iidfile",
            str(iidfile),
            ".",
        ],
        timeout=3600,
    )
    image_digest = iidfile.read_text().strip() if iidfile.exists() else image_tag
    if write_proof:
        write_build_proof(
            problem_dir,
            image_digest=image_digest,
            base_image_ref=base.ref,
            platform=TAIGA_PLATFORM,
            alignerr_cli_version="0.1.0",
            duration_seconds=time.monotonic() - start,
        )
    build = TaskBuild(
        image_tag=image_tag,
        image_flavor=base.flavor,
        flavor_requested=flavor,
        flavor_fallback=base.flavor_fallback,
    )
    _write_flavor_sidecar(problem_dir, build)
    return build


def _write_flavor_sidecar(problem_dir: Path, build: TaskBuild) -> None:
    path = problem_dir / FLAVOR_SIDECAR
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "image_flavor": build.image_flavor,
                "flavor_requested": build.flavor_requested,
                "flavor_fallback": build.flavor_fallback,
            }
        )
        + "\n"
    )


def read_flavor_sidecar(problem_dir: Path) -> dict | None:
    """Read the flavor recorded by the last build_task_image (or None)."""
    path = problem_dir / FLAVOR_SIDECAR
    try:
        return json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return None


def start_task_container(image_tag: str) -> StartedContainer:
    # ``--network none`` isolates the agent's code execution from the internet,
    # mirroring the Taiga sandbox. The model itself runs on the HOST (the
    # claude-code / deepagents loop drives the container over ``docker exec`` via
    # the rubric MCP bridge + tmux), so cutting the container's network does NOT
    # break model calls -- it only stops the agent's own bash from pip
    # installing / curling freely, which would make local runs unrepresentative
    # of real Taiga scores.
    args = [
        "run",
        "--platform",
        TAIGA_PLATFORM,
        "--network",
        "none",
        "-d",
        "--rm",
        "-e",
        "ANTHROPIC_API_KEY",
        "-e",
        "IS_SANDBOX=yes",
        image_tag,
        "sleep",
        "infinity",
    ]
    proc = _docker(args, timeout=120)
    cid = proc.stdout.strip()
    if not cid:
        raise RuntimeError(f"docker run returned no container id for {image_tag}")
    patch_rubric_stdout_noise(cid)
    return StartedContainer(image_tag=image_tag, container_id=cid)


def patch_rubric_stdout_noise(cid: str) -> None:
    """Keep rubric grading prints off MCP stdout.

    The current image-baked rubric mirrors scorer subprocess stdout to its own
    stdout. That corrupts MCP stdio because lines like `RUBRIC_RESULT_JSON=...`
    are not JSON-RPC messages. Local harness runs patch the server to mirror
    those scorer lines to stderr instead.
    """
    script = r"""
from pathlib import Path

path = Path("/mcp_server/src/rubric/server.py")
if not path.exists():
    raise SystemExit(0)
text = path.read_text()
text = text.replace("sys.stdout.write(proc.stdout)", "sys.stderr.write(proc.stdout)")
path.write_text(text)
"""
    subprocess.run(
        ["docker", "exec", cid, "python", "-c", script],
        text=True,
        capture_output=True,
        check=False,
    )


def stop_task_container(cid: str) -> None:
    if shutil.which("docker") is None:
        return
    subprocess.run(
        ["docker", "rm", "-f", cid], text=True, capture_output=True, check=False
    )


def copy_output_from_container(cid: str, destination: Path) -> None:
    destination.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        ["docker", "cp", f"{cid}:/tmp/output/.", str(destination)],
        text=True,
        capture_output=True,
        check=False,
    )
