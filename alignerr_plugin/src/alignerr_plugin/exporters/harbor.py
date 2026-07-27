"""Export a task directory in self-contained Harbor task format.

Writes an ``environment/Dockerfile`` plus the grader/runtime sources needed to
build the task image from the exported directory itself, with no dependency on
the Boreal Artifact Registry. The generated ``tests/test.sh`` invokes
``/runtime/run_grader.py`` inside that image.
"""

from __future__ import annotations

import shutil
import tomllib
from pathlib import Path

import tomli_w

from alignerr_plugin.runtime_notices import runtime_notices_for_resources
from alignerr_plugin.solver_hints import task_type_solver_hint
from alignerr_plugin.utils import load_task_toml

# Repo-root-relative paths to runtime sources we bundle into the Harbor export.
# This file is at: <repo>/alignerr_plugin/src/alignerr_plugin/exporters/harbor.py
# So parents[4] is the repo root.
_REPO_ROOT = Path(__file__).resolve().parents[4]
_GRADER_DIR = _REPO_ROOT / "grader"
_GRADING_SRC = _GRADER_DIR / "src" / "grading"
_RUN_GRADER_SRC = _GRADER_DIR / "src" / "grader_runner" / "run_grader.py"
_RUBRIC_DIR = _REPO_ROOT / "taiga_runtime" / "rubric"
_BASE_DIR = _REPO_ROOT / "base"
_NUMERICAL_SOLVER_TASK_TYPES = frozenset({"cfd", "structures"})
DEFAULT_HARBOR_AGENT_USER = "agent"
DEFAULT_HARBOR_VERIFIER_USER = "root"

_SELF_CONTAINED_DOCKERFILE = """\
FROM python:3.13-slim

ENV UV_SYSTEM_PYTHON=1
ENV PATH="/opt/lbx-runtime/.venv/bin:/usr/local/bin:${PATH}"
ENV TORCH_INDEX_URL=https://download.pytorch.org/whl/cpu
ENV BASE_EXTRA_REQUIREMENTS=/tmp/base/requirements-cpu.txt
# This image builds FROM python:3.13-slim rather than a native base, so it has
# to repeat what every base/*/Dockerfile sets. [[preloaded_files]] derives its
# HF mount paths from this root (schemas.HF_HOME), so without it an exported ML
# task's offline from_pretrained/load_dataset looks in the wrong cache.
ENV HF_HOME=/tmp/hf-cache

COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

WORKDIR /mcp_server

COPY taiga_runtime/rubric/ /mcp_server/
COPY grader/ /runtime/grading/
COPY base/requirements-runtime.txt base/requirements-common.txt base/requirements-cpu.txt /tmp/base/
COPY --chmod=0755 base/install-common.sh /tmp/base/install-common.sh
RUN /tmp/base/install-common.sh
@@TASK_EXTRAS@@
COPY --chown=root:root scorer/data/ /mcp_server/data/
COPY --chown=root:root scorer/ /mcp_server/grader/
COPY data/ /workspace/data/
RUN rm -rf /data \
    && ln -s /workspace/data /data \
    && chown -R root:root /workspace/data \
    && find /workspace/data -type d -exec chmod 0755 {} + \
    && find /workspace/data -type f -exec chmod 0644 {} +
COPY task.toml instruction.md /task/
RUN rm -rf /mcp_server/grader/data \
    && chown -R root:root /mcp_server \
    && find /mcp_server/data /mcp_server/grader -type d -exec chmod 0700 {} + \
    && find /mcp_server/data /mcp_server/grader -type f -exec chmod 0600 {} + \
    && chmod 0700 /mcp_server

WORKDIR /workdir

CMD ["/bin/bash"]
"""

_SOLVER_SELF_CONTAINED_DOCKERFILE = """\
FROM python:3.13-slim

ENV UV_SYSTEM_PYTHON=1
ENV PATH="/opt/lbx-runtime/.venv/bin:/usr/local/bin:${PATH}"
ENV TORCH_INDEX_URL=https://download.pytorch.org/whl/cpu
ENV BASE_EXTRA_REQUIREMENTS=/tmp/base/requirements-cpu.txt
ENV HF_HOME=/tmp/hf-cache

COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

WORKDIR /mcp_server

COPY taiga_runtime/rubric/ /mcp_server/
COPY grader/ /runtime/grading/
COPY base/requirements-runtime.txt base/requirements-common.txt base/requirements-cpu.txt base/requirements-solvers.txt /tmp/base/
COPY --chmod=0755 base/install-common.sh base/install-solvers-heavy.sh /tmp/base/
RUN /tmp/base/install-common.sh
@@TASK_EXTRAS@@
COPY --chown=root:root scorer/data/ /mcp_server/data/
COPY --chown=root:root scorer/ /mcp_server/grader/
COPY data/ /workspace/data/
RUN rm -rf /data \
    && ln -s /workspace/data /data \
    && chown -R root:root /workspace/data \
    && find /workspace/data -type d -exec chmod 0755 {} + \
    && find /workspace/data -type f -exec chmod 0644 {} +
COPY task.toml instruction.md /task/
RUN rm -rf /mcp_server/grader/data \
    && chown -R root:root /mcp_server \
    && find /mcp_server/data /mcp_server/grader -type d -exec chmod 0700 {} + \
    && find /mcp_server/data /mcp_server/grader -type f -exec chmod 0600 {} + \
    && chmod 0700 /mcp_server

WORKDIR /workdir

CMD ["/bin/bash"]
"""


_TEST_SH_DEFAULT = """\
#!/bin/bash
set -euo pipefail

# Invoke the grader runner built into the self-contained Harbor image.
exec /runtime/run_grader.py \\
    --workspace /tmp/output \\
    --grader-dir /mcp_server/grader \\
    --private-dir /mcp_server/data \\
    --output-dir /logs/verifier
"""

_TEST_SH_STANDALONE = """\
#!/bin/bash
set -euo pipefail

# Prefer the image-baked runner, fall back to the bundled copy when the
# image is not built FROM lbx-tasks-base. The bundled copy lives at
# /tests/_runtime/ and is byte-identical to what the base image ships.
if [ -x /runtime/run_grader.py ]; then
    exec /runtime/run_grader.py \\
        --workspace /tmp/output \\
        --grader-dir /mcp_server/grader \\
        --private-dir /mcp_server/data \\
        --output-dir /logs/verifier
fi

export PYTHONPATH="/tests/_runtime${PYTHONPATH:+:${PYTHONPATH}}"
exec python /tests/_runtime/run_grader.py \\
    --workspace /tmp/output \\
    --grader-dir /mcp_server/grader \\
    --private-dir /mcp_server/data \\
    --output-dir /logs/verifier
"""


def _grader_uses_llm(problem_dir: Path) -> bool:
    """Best-effort static check: does the task scorer call into the LLM judge?"""
    grader_path = problem_dir / "scorer" / "compute_score.py"
    if not grader_path.exists():
        return False
    try:
        text = grader_path.read_text()
    except OSError:
        return False
    return "llm_criterion" in text or "LLMJudge" in text


def _ensure_harbor_user_separation(task_toml_path: Path) -> None:
    """Pin Harbor agent/verifier users for Prometheus v1 security."""
    data = tomllib.loads(task_toml_path.read_text())
    agent = data.setdefault("agent", {})
    verifier = data.setdefault("verifier", {})
    changed = False
    if agent.get("user") != DEFAULT_HARBOR_AGENT_USER:
        agent["user"] = DEFAULT_HARBOR_AGENT_USER
        changed = True
    if verifier.get("user") != DEFAULT_HARBOR_VERIFIER_USER:
        verifier["user"] = DEFAULT_HARBOR_VERIFIER_USER
        changed = True
    if changed:
        task_toml_path.write_text(tomli_w.dumps(data))


def _ensure_verifier_env_for_llm(task_toml_path: Path) -> None:
    """Add ``ANTHROPIC_API_KEY`` to ``[verifier.env]`` so ``harbor run``
    requests it from the user. Caller decides whether to invoke this."""
    data = tomllib.loads(task_toml_path.read_text())
    verifier = data.setdefault("verifier", {})
    env = verifier.setdefault("env", [])
    if isinstance(env, list):
        if "ANTHROPIC_API_KEY" not in env:
            env.append("ANTHROPIC_API_KEY")
    elif isinstance(env, dict):
        env.setdefault("ANTHROPIC_API_KEY", "${ANTHROPIC_API_KEY}")
    task_toml_path.write_text(tomli_w.dumps(data))


def _stamp_image_ref(task_toml_path: Path, image_ref: str | None) -> None:
    if not image_ref:
        return
    data = tomllib.loads(task_toml_path.read_text())
    environment = data.setdefault("environment", {})
    environment["docker_image"] = image_ref
    task_toml_path.write_text(tomli_w.dumps(data))


def _notice_resource_for_problem(problem_dir: Path) -> str:
    """Return the Taiga resource tier for runtime notice generation."""
    task_toml = load_task_toml(problem_dir)
    return task_toml.environment.required_resources


def _stamp_runtime_notices(task_toml_path: Path, problem_dir: Path) -> None:
    resource = _notice_resource_for_problem(problem_dir)
    data = tomllib.loads(task_toml_path.read_text())
    metadata = data.setdefault("metadata", {})
    notices = runtime_notices_for_resources(resource, metadata.get("runtime_notices"))
    if notices:
        metadata["runtime_notices"] = notices
    else:
        metadata.pop("runtime_notices", None)
    task_toml_path.write_text(tomli_w.dumps(data))


def _copy_standalone_runtime(tests_dir: Path) -> None:
    """Bundle a copy of the shared `grader/` package and `run_grader.py` under tests/_runtime/.

    Used by ``--standalone`` so the Harbor task runs even when the image is
    not built from `lbx-tasks-base`.
    """
    runtime_dir = tests_dir / "_runtime"
    runtime_dir.mkdir(parents=True, exist_ok=True)

    # Copy the grading package so `from grading import ...` resolves.
    # Exclude __pycache__ so the export is reproducible across machines.
    grading_dst = runtime_dir / "grading"
    if grading_dst.exists():
        shutil.rmtree(grading_dst)
    shutil.copytree(
        _GRADING_SRC,
        grading_dst,
        ignore=shutil.ignore_patterns("__pycache__", "*.pyc"),
    )

    shutil.copy2(_RUN_GRADER_SRC, runtime_dir / "run_grader.py")
    (runtime_dir / "run_grader.py").chmod(0o755)


def _copytree_clean(source: Path, destination: Path) -> None:
    if destination.exists():
        shutil.rmtree(destination)
    shutil.copytree(
        source,
        destination,
        ignore=shutil.ignore_patterns("__pycache__", "*.pyc", ".pytest_cache"),
    )


_TASK_DEPS_BLOCK = """
# Task-declared dependency channels. install-task-deps.sh routes each channel to
# its isolation boundary: agent-visible packages into the runtime venv,
# grader-only and hidden-env-only packages into root-only /mcp_server trees.
COPY --chmod=0755 base/install-task-deps.sh /tmp/base/install-task-deps.sh
COPY source_environment/ /tmp/task-deps/environment/
COPY --chown=root:root scorer/ /tmp/task-deps/scorer/
RUN /tmp/base/install-task-deps.sh /tmp/task-deps && rm -rf /tmp/task-deps
"""

# The baked lock is the author's, not the trusted-CI promoted one. The
# .author-source marker is what lets the rubric server refuse to grade with it
# when the export declares requires_trusted_mount.
_CALIBRATION_BLOCK = """
COPY --chown=root:root calibration.lock.json \\
    /mcp_server/calibration/calibration.lock.json
RUN printf 'author-image-fallback\\n' > /mcp_server/calibration/.author-source \\
    && chown -R root:root /mcp_server/calibration \\
    && chmod 0700 /mcp_server/calibration \\
    && chmod 0600 /mcp_server/calibration/calibration.lock.json \\
    /mcp_server/calibration/.author-source
"""

_DEPENDENCY_CHANNEL_FILES = (
    Path("environment") / "apt.txt",
    Path("environment") / "requirements.txt",
    Path("scorer") / "requirements.txt",
    Path("scorer") / "env-requirements.txt",
)


def _task_extras_block(problem_dir: Path) -> str:
    """Render the Dockerfile steps that depend on what the task actually ships."""
    blocks = []
    if any((problem_dir / rel).is_file() for rel in _DEPENDENCY_CHANNEL_FILES):
        blocks.append(_TASK_DEPS_BLOCK)
    if (problem_dir / "calibration.lock.json").is_file():
        blocks.append(_CALIBRATION_BLOCK)
    return "".join(blocks)


def _native_self_contained_dockerfile(problem_dir: Path) -> str:
    """Return the self-contained Harbor Dockerfile for a native ISO task."""
    task_toml = load_task_toml(problem_dir)
    task_type = (task_toml.difficulty.task_type or "").strip().lower()
    template = (
        _SOLVER_SELF_CONTAINED_DOCKERFILE
        if task_type in _NUMERICAL_SOLVER_TASK_TYPES
        else _SELF_CONTAINED_DOCKERFILE
    )
    return template.replace("@@TASK_EXTRAS@@\n", _task_extras_block(problem_dir))


def _prometheus_solver_instruction_hint(problem_dir: Path) -> str:
    """Return the solver hint that Prometheus sees inline in the instruction."""
    task_toml = load_task_toml(problem_dir)
    if task_toml.delivery.platform != "prometheus":
        return ""
    return task_type_solver_hint(task_toml.difficulty.task_type or "").strip()


def _append_instruction_hint(instruction: str, hint: str) -> str:
    if not hint:
        return instruction
    normalized = instruction.rstrip("\n")
    if hint in normalized:
        return normalized + "\n"
    return normalized + "\n\n" + hint + "\n"


def _append_prometheus_solver_hint(problem_dir: Path, output_dir: Path) -> None:
    hint = _prometheus_solver_instruction_hint(problem_dir)
    if not hint:
        return
    for path in (
        output_dir / "instruction.md",
        output_dir / "environment" / "instruction.md",
    ):
        if path.exists():
            path.write_text(_append_instruction_hint(path.read_text(), hint))


def _write_self_contained_environment(problem_dir: Path, output_dir: Path) -> None:
    environment_dir = output_dir / "environment"
    environment_dir.mkdir(parents=True, exist_ok=True)

    (environment_dir / "Dockerfile").write_text(
        _native_self_contained_dockerfile(problem_dir)
    )
    for name in ("task.toml", "instruction.md", "calibration.lock.json"):
        source = problem_dir / name
        if source.exists():
            shutil.copy2(source, environment_dir / name)
    for name in ("data", "scorer"):
        source = problem_dir / name
        destination = environment_dir / name
        if source.exists():
            _copytree_clean(source, destination)
        else:
            destination.mkdir(parents=True, exist_ok=True)

    _copytree_clean(_GRADER_DIR, environment_dir / "grader")
    _copytree_clean(_RUBRIC_DIR, environment_dir / "taiga_runtime" / "rubric")
    _copytree_clean(_BASE_DIR, environment_dir / "base")

    # Always materialized: the task-deps block COPYs source_environment/ even for
    # a task whose only declared channel lives under scorer/.
    original_environment = problem_dir / "environment"
    if original_environment.exists():
        _copytree_clean(original_environment, environment_dir / "source_environment")
    else:
        (environment_dir / "source_environment").mkdir(parents=True, exist_ok=True)


def export_harbor(
    problem_dir: Path,
    output_dir: Path,
    *,
    image_ref: str | None = None,
    standalone: bool = True,
    include_runtime_notices: bool = True,
) -> Path:
    """Write a Harbor-format copy of the task into ``output_dir``.

    Files copied or generated:

      * ``task.toml``            (``ANTHROPIC_API_KEY`` added to ``[verifier].env`` if
                                 the grader uses LLM judging; optional runtime
                                 notice metadata stamped)
      * ``instruction.md``
      * ``environment/``         (self-contained Dockerfile + runtime/grader/scorer files)
      * ``solution/``            (optional Oracle solver)
      * ``tests/test.sh``        (shim that calls /runtime/run_grader.py)
    """
    _ = image_ref, standalone
    if output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True)

    top_level = ["task.toml", "instruction.md", "solution"]
    for name in top_level:
        source = problem_dir / name
        destination = output_dir / name
        if source.is_dir():
            _copytree_clean(source, destination)
        elif source.exists():
            shutil.copy2(source, destination)

    _write_self_contained_environment(problem_dir, output_dir)
    if (problem_dir / "task.toml").exists():
        _append_prometheus_solver_hint(problem_dir, output_dir)

    tests_dir = output_dir / "tests"
    tests_dir.mkdir(parents=True, exist_ok=True)
    test_sh = tests_dir / "test.sh"
    test_sh.write_text(_TEST_SH_DEFAULT)
    test_sh.chmod(0o755)

    task_toml_path = output_dir / "task.toml"
    if task_toml_path.exists():
        if load_task_toml(problem_dir).delivery.platform == "prometheus":
            _ensure_harbor_user_separation(task_toml_path)
        if _grader_uses_llm(problem_dir):
            _ensure_verifier_env_for_llm(task_toml_path)
        if include_runtime_notices:
            _stamp_runtime_notices(task_toml_path, problem_dir)
        shutil.copy2(task_toml_path, output_dir / "environment" / "task.toml")

    return output_dir
