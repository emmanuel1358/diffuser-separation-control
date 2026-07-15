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

COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

WORKDIR /mcp_server

COPY taiga_runtime/rubric/ /mcp_server/
COPY grader/ /runtime/grading/
COPY base/requirements-runtime.txt base/requirements-common.txt base/requirements-cpu.txt /tmp/base/
COPY --chmod=0755 base/install-common.sh /tmp/base/install-common.sh
RUN /tmp/base/install-common.sh

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
    && chown -R root:root /mcp_server/data /mcp_server/grader \
    && find /mcp_server/data /mcp_server/grader -type d -exec chmod 0700 {} + \
    && find /mcp_server/data /mcp_server/grader -type f -exec chmod 0600 {} +

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
    && chown -R root:root /mcp_server/data /mcp_server/grader \
    && find /mcp_server/data /mcp_server/grader -type d -exec chmod 0700 {} + \
    && find /mcp_server/data /mcp_server/grader -type f -exec chmod 0600 {} +

WORKDIR /workdir

CMD ["/bin/bash"]
"""


# Self-contained Dockerfile TEMPLATE for ML_Envs-mode tasks. GPU-capable +
# CPU-tolerant (cu121 torch wheels bundle the CUDA runtime, so `import torch`
# works CPU-only and uses the GPU when Harbor grants NVIDIA runtime access).
# ``_mlenvs_self_contained_dockerfile`` fills the @@APT@@ / @@DEPS@@ /
# @@ENV_DEPS@@ / @@GRADING_DEPS@@ / @@HIDDEN_ENV@@ placeholders from metadata.json.
_MLENVS_SELF_CONTAINED_TEMPLATE = """\
FROM python:3.12-slim

ENV UV_SYSTEM_PYTHON=1
ENV PATH="/usr/local/nvidia/bin:/opt/lbx-runtime/.venv/bin:/usr/local/bin:${PATH}"
ENV LD_LIBRARY_PATH="/usr/local/nvidia/lib64:/usr/local/nvidia/lib"
ENV NVIDIA_VISIBLE_DEVICES=all
ENV NVIDIA_DRIVER_CAPABILITIES=compute,utility
ENV PYTHON_VERSION=3.12
ENV SKIP_COMMON_REQUIREMENTS=1
ENV BASE_EXTRA_REQUIREMENTS=/tmp/base/requirements-mlenvs-common.txt
ENV TORCH_INDEX_URL=https://download.pytorch.org/whl/cu121
ENV TORCH_PACKAGES="torch==2.4.1+cu121 torchvision==0.19.1+cu121 torchaudio==2.4.1+cu121"
ENV TORCH_INDEX_STRATEGY=unsafe-best-match
ENV HF_HOME=/tmp/.cache/huggingface
ENV HF_HUB_OFFLINE=1
ENV TRANSFORMERS_OFFLINE=1
ENV HF_DATASETS_OFFLINE=1

COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

WORKDIR /mcp_server

@@APT@@COPY taiga_runtime/rubric/ /mcp_server/
COPY grader/ /runtime/grading/
COPY base/requirements-runtime.txt base/requirements-mlenvs-common.txt /tmp/base/
COPY --chmod=0755 base/install-common.sh /tmp/base/install-common.sh
# NOTE: install-common installs requirements-mlenvs-common.txt, which pins the
# plain `lightgbm` PyPI wheel (CPU). The registry mlenvs-gpu base instead builds
# lightgbm from source with USE_GPU=ON, so a self-contained Harbor image gets
# CPU LightGBM. A task that needs GPU LightGBM should deploy via the registry
# base image (or add a from-source USE_GPU=ON build step here).
RUN /tmp/base/install-common.sh

@@DEPS@@@@ENV_DEPS@@@@GRADING_DEPS@@COPY --chown=root:root data/private/ /mcp_server/data/
COPY --chown=root:root test_file.py /mcp_server/grader/compute_score.py
COPY data/public/ /workspace/data/
RUN rm -rf /data \
    && ln -s /workspace/data /data \
    && chown -R root:root /workspace/data \
    && find /workspace/data -type d -exec chmod 0755 {} + \
    && find /workspace/data -type f -exec chmod 0644 {} +
COPY prompt.md /task/prompt.md
# Bake the env-server activation field so a hidden-env (env/hybrid) task's
# supervisor can start; agent-visible, holds no held-out truth.
RUN printf '[environment]\\nhidden_env = "@@HIDDEN_ENV@@"\\n' > /task/task.toml
# Re-lock the held-out truth + grader and re-seal /mcp_server after the COPYs
# above (install-common sealed it once; the COPYs re-added files).
RUN chown -R root:root /mcp_server/data /mcp_server/grader \\
    && find /mcp_server/data /mcp_server/grader -type d -exec chmod 0700 {} + \\
    && find /mcp_server/data /mcp_server/grader -type f -exec chmod 0600 {} + \\
    && chmod 0700 /mcp_server \\
    && mkdir -p /workdir /tmp/output \\
    && chmod 0777 /workdir /tmp/output

WORKDIR /workdir

CMD ["/bin/bash"]
"""


def _mlenvs_self_contained_dockerfile(problem_dir: Path) -> str:
    """Render the ML_Envs Harbor Dockerfile from the task's metadata.json, filling
    the apt_extras / dependencies / env_dependencies / grading_dependencies /
    hidden_env placeholders."""
    from alignerr_plugin import mlenvs

    meta = mlenvs.load_mlenvs_metadata(problem_dir)
    apt = " ".join(meta.get("apt_extras", []) or [])
    deps = " ".join(meta.get("dependencies", []) or [])
    env_deps = " ".join(meta.get("env_dependencies", []) or [])
    grading_deps = " ".join(meta.get("grading_dependencies", []) or [])
    hidden_env = mlenvs.hidden_env_for_task_type(meta["ml_task_type"])

    apt_block = (
        "RUN apt-get update \\\n"
        f"    && apt-get install -y --no-install-recommends {apt} \\\n"
        "    && rm -rf /var/lib/apt/lists/*\n\n"
        if apt
        else ""
    )
    deps_block = (
        "# Task dependencies (agent-visible) into the runtime venv.\n"
        f"RUN uv pip install --python /opt/lbx-runtime/.venv/bin/python --no-cache {deps}\n\n"
        if deps
        else ""
    )
    env_deps_block = (
        "# Env-server-only deps (env/hybrid): root-only /mcp_server/env_deps.\n"
        "RUN mkdir -p /mcp_server/env_deps \\\n"
        "    && uv pip install --python /opt/lbx-runtime/.venv/bin/python "
        f"--target /mcp_server/env_deps --no-cache {env_deps} \\\n"
        "    && chown -R root:root /mcp_server/env_deps \\\n"
        "    && find /mcp_server/env_deps -type d -exec chmod 0700 {} + \\\n"
        "    && find /mcp_server/env_deps -type f -exec chmod 0600 {} +\n\n"
        if env_deps
        else ""
    )
    grading_deps_block = (
        "# Grader-only deps (any task type): root-only /mcp_server/grading_deps.\n"
        "RUN mkdir -p /mcp_server/grading_deps \\\n"
        "    && uv pip install --python /opt/lbx-runtime/.venv/bin/python "
        f"--target /mcp_server/grading_deps --no-cache {grading_deps} \\\n"
        "    && chown -R root:root /mcp_server/grading_deps \\\n"
        "    && find /mcp_server/grading_deps -type d -exec chmod 0700 {} + \\\n"
        "    && find /mcp_server/grading_deps -type f -exec chmod 0600 {} +\n\n"
        if grading_deps
        else ""
    )
    return (
        _MLENVS_SELF_CONTAINED_TEMPLATE.replace("@@APT@@", apt_block)
        .replace("@@DEPS@@", deps_block)
        .replace("@@ENV_DEPS@@", env_deps_block)
        .replace("@@GRADING_DEPS@@", grading_deps_block)
        .replace("@@HIDDEN_ENV@@", hidden_env)
    )


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
    from alignerr_plugin import mlenvs

    grader_path = (
        problem_dir / "test_file.py"
        if mlenvs.is_mlenvs_task(problem_dir)
        else problem_dir / "scorer" / "compute_score.py"
    )
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


def _native_self_contained_dockerfile(problem_dir: Path) -> str:
    """Return the self-contained Harbor Dockerfile for a native ISO task."""
    task_toml = load_task_toml(problem_dir)
    task_type = (task_toml.difficulty.task_type or "").strip().lower()
    if task_type in _NUMERICAL_SOLVER_TASK_TYPES:
        return _SOLVER_SELF_CONTAINED_DOCKERFILE
    return _SELF_CONTAINED_DOCKERFILE


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
    from alignerr_plugin import mlenvs

    environment_dir = output_dir / "environment"
    environment_dir.mkdir(parents=True, exist_ok=True)

    if mlenvs.is_mlenvs_task(problem_dir):
        # ML_Envs layout: prompt.md + test_file.py + data/{public,private}; no
        # scorer/ or task.toml. Bake the image rendered from the task's metadata.
        (environment_dir / "Dockerfile").write_text(
            _mlenvs_self_contained_dockerfile(problem_dir)
        )
        for name in ("prompt.md", "test_file.py", "metadata.json"):
            source = problem_dir / name
            if source.exists():
                shutil.copy2(source, environment_dir / name)
        source_data = problem_dir / "data"
        destination_data = environment_dir / "data"
        if source_data.exists():
            _copytree_clean(source_data, destination_data)
        else:
            destination_data.mkdir(parents=True, exist_ok=True)
        for sub in ("public", "private"):
            (destination_data / sub).mkdir(parents=True, exist_ok=True)
    else:
        (environment_dir / "Dockerfile").write_text(
            _native_self_contained_dockerfile(problem_dir)
        )
        for name in ("task.toml", "instruction.md"):
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

    original_environment = problem_dir / "environment"
    if original_environment.exists():
        _copytree_clean(original_environment, environment_dir / "source_environment")


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
    from alignerr_plugin import mlenvs

    _ = image_ref, standalone
    if output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True)

    if mlenvs.is_mlenvs_task(problem_dir):
        top_level = ["prompt.md", "test_file.py", "metadata.json", "reference_solution"]
    else:
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
