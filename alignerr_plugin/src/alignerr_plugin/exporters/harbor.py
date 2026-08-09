"""Export a task directory in self-contained Harbor task format.

Writes an ``environment/Dockerfile`` plus the grader/runtime sources needed to
build the task image from the exported directory itself, with no dependency on
the Boreal Artifact Registry. The generated ``tests/test.sh`` invokes
``/runtime/run_grader.py`` inside that image.
"""

from __future__ import annotations

import copy
import re
import shlex
import shutil
import tomllib
from pathlib import Path

import tomli_w
import yaml

from alignerr_plugin.capabilities import (
    CapabilityConfig,
    ServiceSpec,
    implicit_agent_service,
    is_capability_task,
    is_digest_pinned_image,
    project_harbor_task_data,
    remap_depends_on,
    resolve_capabilities,
)
from alignerr_plugin.materialization import (
    copy_directory_contents_safe,
    materialize_task_inputs,
    materialize_workspace_seed,
    require_local_dockerfile_target,
    resolve_task_path,
    resolve_workspace,
    staged_output_directory,
    workspace_dockerfile_overlay,
)
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
_ENVIRONMENT_VARIABLE_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")
_PINNED_UV_IMAGE = (
    "ghcr.io/astral-sh/uv:0.8.17@sha256:"
    "e4644cb5bd56fdc2c5ea3ee0525d9d21eed1603bccd6a21f887a938be7e85be1"
)

_SELF_CONTAINED_DOCKERFILE = """\
FROM python:3.13-slim

ENV PYTHON_VERSION=3.13
ENV UV_SYSTEM_PYTHON=1
ENV PATH="/opt/lbx-runtime/.venv/bin:/usr/local/bin:${PATH}"
ENV TORCH_INDEX_URL=https://download.pytorch.org/whl/cpu
ENV BASE_EXTRA_REQUIREMENTS=/tmp/base/requirements-cpu.txt
# This image builds FROM python:3.13-slim rather than a native base, so it has
# to repeat what every base/*/Dockerfile sets. [[preloaded_files]] derives its
# HF mount paths from this root (schemas.HF_HOME), so without it an exported ML
# task's offline from_pretrained/load_dataset looks in the wrong cache.
ENV HF_HOME=/tmp/hf-cache

COPY --from=@@UV_IMAGE@@ /uv /usr/local/bin/uv

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

ENV PYTHON_VERSION=3.13
ENV UV_SYSTEM_PYTHON=1
ENV PATH="/opt/lbx-runtime/.venv/bin:/usr/local/bin:${PATH}"
ENV TORCH_INDEX_URL=https://download.pytorch.org/whl/cpu
ENV BASE_EXTRA_REQUIREMENTS=/tmp/base/requirements-cpu.txt
ENV HF_HOME=/tmp/hf-cache

COPY --from=@@UV_IMAGE@@ /uv /usr/local/bin/uv

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


def _harbor_env_dict(value: object, *, label: str) -> dict[str, str]:
    """Normalize ISO env-name lists to Harbor's value-bearing env tables."""
    if value is None:
        return {}
    if isinstance(value, dict):
        normalized: dict[str, str] = {}
        for raw_name, raw_value in value.items():
            if not isinstance(raw_name, str) or not _ENVIRONMENT_VARIABLE_RE.fullmatch(
                raw_name
            ):
                raise ValueError(f"{label} has invalid environment variable name")
            if not isinstance(raw_value, str):
                raise TypeError(f"{label}.{raw_name} must be a string")
            normalized[raw_name] = raw_value
        return normalized
    if isinstance(value, list):
        normalized = {}
        for raw_name in value:
            if not isinstance(raw_name, str) or not _ENVIRONMENT_VARIABLE_RE.fullmatch(
                raw_name
            ):
                raise ValueError(
                    f"{label} list entries must be environment variable names"
                )
            normalized[raw_name] = f"${{{raw_name}}}"
        return normalized
    raise ValueError(f"{label} must be a table or a list of variable names")


def _normalize_harbor_env_shapes(data: dict[str, object]) -> None:
    """Mutate a projected task so every Harbor env field has its canonical shape."""
    environment = data.get("environment")
    if environment is not None and not isinstance(environment, dict):
        raise ValueError("[environment] must be a table")
    if isinstance(environment, dict) and "env" in environment:
        environment["env"] = _harbor_env_dict(
            environment["env"], label="[environment].env"
        )

    agent = data.get("agent")
    if agent is not None and not isinstance(agent, dict):
        raise ValueError("[agent] must be a table")
    if isinstance(agent, dict) and "env" in agent:
        if environment is None:
            environment = {}
            data["environment"] = environment
        assert isinstance(environment, dict)
        agent_env = _harbor_env_dict(agent.pop("env"), label="[agent].env")
        environment_env = _harbor_env_dict(
            environment.get("env"),
            label="[environment].env",
        )
        for name, value in agent_env.items():
            existing = environment_env.get(name)
            if existing is not None and existing != value:
                raise ValueError(
                    f"[agent].env.{name} conflicts with [environment].env.{name}"
                )
            environment_env[name] = value
        environment["env"] = environment_env

    verifier = data.get("verifier")
    if verifier is not None and not isinstance(verifier, dict):
        raise ValueError("[verifier] must be a table")
    if isinstance(verifier, dict):
        verifier["env"] = _harbor_env_dict(
            verifier.get("env"),
            label="[verifier].env",
        )
        verifier_environment = verifier.get("environment")
        if verifier_environment is not None:
            if not isinstance(verifier_environment, dict):
                raise ValueError("[verifier.environment] must be a table")
            if "env" in verifier_environment:
                verifier_environment["env"] = _harbor_env_dict(
                    verifier_environment["env"],
                    label="[verifier.environment].env",
                )

    solution = data.get("solution")
    if solution is not None:
        if not isinstance(solution, dict):
            raise ValueError("[solution] must be a table")
        solution["env"] = _harbor_env_dict(
            solution.get("env"),
            label="[solution].env",
        )


def _normalize_harbor_task_file(task_toml_path: Path) -> None:
    data = tomllib.loads(task_toml_path.read_text())
    _normalize_harbor_env_shapes(data)
    task_toml_path.write_text(tomli_w.dumps(data))


def _ensure_verifier_env_for_llm(task_toml_path: Path) -> None:
    """Add ``ANTHROPIC_API_KEY`` to ``[verifier.env]`` so ``harbor run``
    requests it from the user. Caller decides whether to invoke this."""
    data = tomllib.loads(task_toml_path.read_text())
    _normalize_harbor_env_shapes(data)
    verifier = data.setdefault("verifier", {})
    assert isinstance(verifier, dict)
    env = verifier.setdefault("env", {})
    assert isinstance(env, dict)
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


def _is_software_task(task_data: dict[str, object]) -> bool:
    difficulty = task_data.get("difficulty")
    task_type = (
        str(difficulty.get("task_type") or "").strip().lower().replace("-", "_")
        if isinstance(difficulty, dict)
        else ""
    )
    return task_type == "software_engineering"


def _software_seed_relative_path(task_data: dict[str, object]) -> str:
    workspace = task_data.get("workspace")
    if isinstance(workspace, dict) and workspace.get("seed"):
        return str(workspace["seed"])
    return "starter"


def _software_output_repo_path(task_data: dict[str, object]) -> str:
    metadata = task_data.get("metadata")
    transformation = (
        metadata.get("transformation") if isinstance(metadata, dict) else None
    )
    if isinstance(transformation, dict):
        declared = transformation.get("workspace_root")
        if isinstance(declared, str) and declared.startswith("/"):
            return declared
    outputs = task_data.get("outputs")
    if isinstance(outputs, list):
        for output in outputs:
            if not isinstance(output, dict) or output.get("required") is False:
                continue
            path = output.get("path")
            if isinstance(path, str) and path.startswith("/tmp/output/"):
                return path
    return "/tmp/output/repo"


def _software_workspace_block(
    problem_dir: Path,
    task_data: dict[str, object],
) -> str:
    if not _is_software_task(task_data):
        return ""
    seed_relative = _software_seed_relative_path(task_data)
    seed = resolve_task_path(
        problem_dir,
        seed_relative,
        label="software workspace seed",
    )
    if not seed.is_dir():
        raise ValueError(
            "software workspace seed must be an existing task-relative directory: "
            f"{seed_relative!r}"
        )
    output_repo = _software_output_repo_path(task_data)
    quoted_repo = shlex.quote(output_repo)
    return f"""\

# Seed the non-root software workspace while keeping grader inputs root-only.
RUN id -u agent >/dev/null 2>&1 || (groupadd -g 1000 agent && \
    useradd -u 1000 -g agent -m -s /bin/bash agent)
RUN mkdir -p {quoted_repo}
COPY --chown=agent:agent workspace_seed/ {output_repo.rstrip("/")}/
RUN git -C {quoted_repo} init -q \
    && git -C {quoted_repo} add -A \
    && git -C {quoted_repo} -c user.name=alignerr \
       -c user.email=alignerr@local commit -q --allow-empty -m seed \
    && chown -R 1000:1000 {quoted_repo}
WORKDIR {output_repo}
USER agent
"""


def _native_self_contained_dockerfile(problem_dir: Path) -> str:
    """Return the self-contained Harbor Dockerfile for a native ISO task."""
    task_toml = load_task_toml(problem_dir)
    task_data = tomllib.loads((problem_dir / "task.toml").read_text())
    task_type = (task_toml.difficulty.task_type or "").strip().lower()
    template = (
        _SOLVER_SELF_CONTAINED_DOCKERFILE
        if task_type in _NUMERICAL_SOLVER_TASK_TYPES
        else _SELF_CONTAINED_DOCKERFILE
    )
    rendered = template.replace("@@UV_IMAGE@@", _PINNED_UV_IMAGE).replace(
        "@@TASK_EXTRAS@@\n", _task_extras_block(problem_dir)
    )
    return rendered.rstrip() + "\n" + _software_workspace_block(problem_dir, task_data)


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
    task_data = tomllib.loads((problem_dir / "task.toml").read_text())
    if _is_software_task(task_data):
        seed_relative = _software_seed_relative_path(task_data)
        seed = resolve_task_path(
            problem_dir,
            seed_relative,
            label="software workspace seed",
        )
        if not seed.is_dir():
            raise ValueError(
                "software workspace seed must be an existing task-relative "
                f"directory: {seed_relative!r}"
            )
        copy_directory_contents_safe(
            seed,
            environment_dir / "workspace_seed",
            task_root=problem_dir,
        )

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


_CAPABILITY_VERIFIER_OVERLAY = """\

# Alignerr capability verifier overlay. Harbor builds this Dockerfile from the
# tests/ context and materializes declared artifacts at their original paths.
USER root
COPY _runtime/ /tests/_runtime/
COPY scorer/data/ /mcp_server/data/
COPY scorer/ /mcp_server/grader/
COPY task/ /task/
COPY --chmod=0755 test.sh /tests/test.sh
RUN rm -rf /mcp_server/grader/data \
    && chown -R root:root /mcp_server /tests \
    && find /mcp_server -type d -exec chmod 0700 {} + \
    && find /mcp_server -type f -exec chmod 0600 {} +
ENTRYPOINT []
CMD ["sleep", "infinity"]
"""


def _capability_config(
    problem_dir: Path, task_data: dict[str, object]
) -> CapabilityConfig:
    capabilities = resolve_capabilities(task_data)
    if capabilities.agent_service is not None:
        return capabilities
    if (problem_dir / "environment" / "Dockerfile").is_file():
        default_context = "environment"
    elif (problem_dir / "environment" / "main" / "Dockerfile").is_file():
        default_context = "environment/main"
    else:
        raise ValueError(
            "capability task requires one role='agent' service or "
            "environment[/main]/Dockerfile"
        )
    agent = task_data.get("agent")
    agent_resources = agent.get("resources") if isinstance(agent, dict) else None
    platform = (
        str(agent_resources.get("platform"))
        if isinstance(agent_resources, dict) and agent_resources.get("platform")
        else None
    )
    default_agent = implicit_agent_service(
        context=default_context,
        resources=capabilities.agent_resources,
        network=(
            str(agent_resources.get("network"))
            if isinstance(agent_resources, dict) and agent_resources.get("network")
            else None
        ),
        platform=platform,
    )
    return CapabilityConfig(
        services=(default_agent, *capabilities.services),
        artifacts=capabilities.artifacts,
        captures=capabilities.captures,
        mcp_servers=capabilities.mcp_servers,
        verifier_mcp_servers=capabilities.verifier_mcp_servers,
        volumes=capabilities.volumes,
        agent_resources=capabilities.agent_resources,
        verifier_resources=capabilities.verifier_resources,
    )


def _service_build_paths(problem_dir: Path, service: ServiceSpec) -> tuple[Path, Path]:
    assert service.build is not None
    context = resolve_task_path(
        problem_dir,
        service.build.context,
        label=f"service {service.name!r} build context",
    )
    dockerfile = resolve_task_path(
        context,
        service.build.dockerfile,
        label=f"service {service.name!r} Dockerfile",
    )
    if not context.is_dir():
        raise ValueError(
            f"service {service.name!r} build context does not exist: {context}"
        )
    if not dockerfile.is_file():
        raise ValueError(
            f"service {service.name!r} Dockerfile does not exist: {dockerfile}"
        )
    return context, dockerfile


def _write_capability_agent_environment(
    problem_dir: Path,
    output_dir: Path,
    service: ServiceSpec,
    task_data: dict[str, object],
) -> None:
    environment_dir = output_dir / "environment"
    environment_dir.mkdir(parents=True, exist_ok=True)
    workspace = resolve_workspace(task_data)
    if service.build is not None:
        context, dockerfile = _service_build_paths(problem_dir, service)
        try:
            output_dir.resolve().relative_to(context)
        except ValueError:
            pass
        else:
            raise ValueError(
                "Harbor output directory cannot live inside the agent build context"
            )
        copy_directory_contents_safe(context, environment_dir, task_root=problem_dir)
        base_dockerfile = dockerfile.read_text().rstrip() + "\n"
        if service.build.target:
            target = require_local_dockerfile_target(
                base_dockerfile,
                service.build.target,
                label=f"agent service {service.name!r} build",
            )
            base_dockerfile += f"\nFROM {target}\n"
    elif service.image:
        base_dockerfile = f"FROM {service.image}\n"
    else:
        raise ValueError(
            f"agent service {service.name!r} has neither build nor pinned image"
        )
    materialize_workspace_seed(
        problem_dir,
        environment_dir / ".alignerr-workspace-seed",
        workspace,
    )
    (environment_dir / "Dockerfile").write_text(
        base_dockerfile
        + workspace_dockerfile_overlay(workspace, user=DEFAULT_HARBOR_AGENT_USER)
    )


def _compose_build(
    problem_dir: Path,
    environment_dir: Path,
    service: ServiceSpec,
) -> dict[str, object]:
    assert service.build is not None
    context, _ = _service_build_paths(problem_dir, service)
    destination = environment_dir / "services" / service.name
    copy_directory_contents_safe(context, destination, task_root=problem_dir)
    build: dict[str, object] = {
        "context": f"./services/{service.name}",
    }
    if service.build.dockerfile != "Dockerfile":
        build["dockerfile"] = service.build.dockerfile
    if service.build.args:
        build["args"] = dict(service.build.args)
    if service.build.target:
        build["target"] = service.build.target
    if service.build.pull:
        build["pull"] = True
    if service.build.no_cache:
        build["no_cache"] = True
    return build


def _without_verifier_dependencies(
    depends_on: object,
    capabilities: CapabilityConfig,
) -> object:
    mapped = remap_depends_on(depends_on, capabilities.services)
    verifier_names = {
        service.harbor_name
        for service in capabilities.services
        if service.role == "verifier"
    }
    if isinstance(mapped, dict):
        return {
            name: value for name, value in mapped.items() if name not in verifier_names
        }
    return [name for name in mapped if name not in verifier_names]


def capability_compose_data(
    problem_dir: Path,
    output_dir: Path,
    capabilities: CapabilityConfig,
) -> dict[str, object]:
    """Build Harbor's native compose override for an exported capability task."""
    environment_dir = output_dir / "environment"
    services: dict[str, object] = {}
    for service in sorted(
        capabilities.services,
        key=lambda item: (item.harbor_name != "main", item.harbor_name),
    ):
        if service.role == "verifier":
            continue
        definition = copy.deepcopy(service.compose)
        if service.build is not None and service.build.platform:
            definition["platform"] = service.build.platform
        if "depends_on" in definition:
            definition["depends_on"] = _without_verifier_dependencies(
                definition["depends_on"], capabilities
            )
            if not definition["depends_on"]:
                definition.pop("depends_on")
        network_mode = definition.get("network_mode")
        if isinstance(network_mode, str) and network_mode.startswith("service:"):
            target = network_mode.split(":", 1)[1]
            mapped = next(
                (
                    item.harbor_name
                    for item in capabilities.services
                    if item.name == target
                ),
                target,
            )
            definition["network_mode"] = f"service:{mapped}"
        if service.role != "agent":
            if service.build is not None:
                definition["build"] = _compose_build(
                    problem_dir, environment_dir, service
                )
            elif service.image:
                definition["image"] = service.image
            else:
                raise ValueError(
                    f"sidecar service {service.name!r} has neither build nor "
                    "pinned image"
                )
        elif service.build is not None:
            agent_build: dict[str, object] = {
                "context": ".",
                "dockerfile": "Dockerfile",
            }
            if service.build.args:
                agent_build["args"] = dict(service.build.args)
            if service.build.pull:
                agent_build["pull"] = True
            if service.build.no_cache:
                agent_build["no_cache"] = True
            definition["build"] = agent_build
        elif service.image:
            definition["build"] = {
                "context": ".",
                "dockerfile": "Dockerfile",
            }
        services[service.harbor_name] = definition
    services.setdefault("main", {})
    compose: dict[str, object] = {"services": services}
    volumes = {
        str(volume["name"]): {}
        for volume in capabilities.volumes
        if volume.get("scope") != "verifier"
    }
    if volumes:
        compose["volumes"] = volumes
    if any(
        "alignerr-isolated" in service.get("networks", {})
        for service in services.values()
    ):
        compose["networks"] = {"alignerr-isolated": {"internal": True}}
    return compose


def _write_default_capability_verifier_context(
    problem_dir: Path, tests_dir: Path, task_data: dict[str, object]
) -> str:
    _copytree_clean(_GRADER_DIR, tests_dir / "grader")
    _copytree_clean(_RUBRIC_DIR, tests_dir / "taiga_runtime" / "rubric")
    _copytree_clean(_BASE_DIR, tests_dir / "base")
    original_environment = problem_dir / "environment"
    if original_environment.is_dir():
        _copytree_clean(original_environment, tests_dir / "source_environment")
    else:
        (tests_dir / "source_environment").mkdir(parents=True, exist_ok=True)

    difficulty = task_data.get("difficulty")
    task_type = (
        str(difficulty.get("task_type") or "").strip().lower()
        if isinstance(difficulty, dict)
        else ""
    )
    template = (
        _SOLVER_SELF_CONTAINED_DOCKERFILE
        if task_type in _NUMERICAL_SOLVER_TASK_TYPES
        else _SELF_CONTAINED_DOCKERFILE
    )
    dockerfile = template.replace("@@UV_IMAGE@@", _PINNED_UV_IMAGE).replace(
        "@@TASK_EXTRAS@@\n", _task_extras_block(problem_dir)
    )
    return dockerfile.replace(
        "\nWORKDIR /workdir\n",
        "\nCOPY --chmod=0755 test.sh /tests/test.sh\n\nWORKDIR /workdir\n",
    )


def _artifact_parent_directories(
    capabilities: CapabilityConfig,
    task_data: dict[str, object],
) -> list[str]:
    parents: set[str] = set()
    for artifact in capabilities.artifacts:
        for key in ("source", "destination"):
            path = str(artifact.get(key) or "")
            if path.startswith("/"):
                parents.add(str(Path(path).parent))
    raw_captures = task_data.get("captures")
    capture_rows = raw_captures if isinstance(raw_captures, list) else []
    for capture in capture_rows:
        if not isinstance(capture, dict):
            continue
        destination = capture.get("atomic_destination") or capture.get("destination")
        if isinstance(destination, str) and destination.startswith("/"):
            parents.add(str(Path(destination).parent))
    result = task_data.get("result")
    if isinstance(result, dict):
        output_root = str(result.get("output_root") or "/tmp/output")
        parents.add(output_root)
        for key in ("reward_file", "trace_file"):
            relative = result.get(key)
            if isinstance(relative, str) and relative:
                parents.add(str(Path(output_root, relative).parent))
    for report in task_data.get("reports", []):
        if isinstance(report, dict):
            path = report.get("path")
            if isinstance(path, str) and path.startswith("/"):
                parents.add(str(Path(path).parent))
    return sorted(parents)


def _write_capability_verifier(
    problem_dir: Path,
    output_dir: Path,
    capabilities: CapabilityConfig,
    task_data: dict[str, object],
) -> None:
    tests_dir = output_dir / "tests"
    tests_dir.mkdir(parents=True, exist_ok=True)
    verifier = capabilities.verifier_service
    workspace = resolve_workspace(task_data)

    if verifier and verifier.build is not None:
        context, dockerfile = _service_build_paths(problem_dir, verifier)
        copy_directory_contents_safe(context, tests_dir, task_root=problem_dir)
        base_dockerfile = dockerfile.read_text().rstrip() + "\n"
        if verifier.build.target:
            target = require_local_dockerfile_target(
                base_dockerfile,
                verifier.build.target,
                label=f"verifier service {verifier.name!r} build",
            )
            base_dockerfile += f"\nFROM {target}\n"
        _copy_standalone_runtime(tests_dir)
    elif verifier and verifier.image:
        base_dockerfile = f"FROM {verifier.image}\n"
        _copy_standalone_runtime(tests_dir)
    else:
        base_dockerfile = _write_default_capability_verifier_context(
            problem_dir, tests_dir, task_data
        )
    materialize_workspace_seed(
        problem_dir,
        tests_dir / ".alignerr-workspace-seed",
        workspace,
    )

    scorer = problem_dir / "scorer"
    if scorer.is_dir():
        _copytree_clean(scorer, tests_dir / "scorer")
    else:
        (tests_dir / "scorer").mkdir(parents=True, exist_ok=True)
    (tests_dir / "scorer" / "data").mkdir(parents=True, exist_ok=True)
    data = problem_dir / "data"
    if data.is_dir():
        _copytree_clean(data, tests_dir / "data")
    else:
        (tests_dir / "data").mkdir(parents=True, exist_ok=True)
    task_dir = tests_dir / "task"
    materialize_task_inputs(problem_dir, task_dir)
    if verifier is None:
        for name in ("task.toml", "instruction.md", "calibration.lock.json"):
            source = output_dir / name
            if source.is_file():
                shutil.copy2(source, tests_dir / name)

    test_sh = tests_dir / "test.sh"
    test_sh.write_text(_TEST_SH_STANDALONE)
    test_sh.chmod(0o755)
    if verifier is not None:
        base_dockerfile += _CAPABILITY_VERIFIER_OVERLAY
    parent_directories = _artifact_parent_directories(capabilities, task_data)
    if parent_directories:
        base_dockerfile += (
            "RUN mkdir -p -- "
            + " ".join(shlex.quote(path) for path in parent_directories)
            + "\n"
        )
    base_dockerfile += workspace_dockerfile_overlay(
        workspace,
        user=DEFAULT_HARBOR_VERIFIER_USER,
        verifier=True,
    )
    (tests_dir / "Dockerfile").write_text(base_dockerfile)


def _export_capability_harbor(
    problem_dir: Path,
    output_dir: Path,
    *,
    task_data: dict[str, object],
    image_ref: str | None,
    include_runtime_notices: bool,
) -> Path:
    capabilities = _capability_config(problem_dir, task_data)
    output_dir.mkdir(parents=True, exist_ok=True)

    projected = project_harbor_task_data(task_data, capabilities)
    projected.setdefault("agent", {})["user"] = DEFAULT_HARBOR_AGENT_USER
    projected.setdefault("verifier", {})["user"] = DEFAULT_HARBOR_VERIFIER_USER
    _normalize_harbor_env_shapes(projected)
    if include_runtime_notices:
        environment = task_data.get("environment")
        required_resources = (
            str(environment.get("required_resources") or "")
            if isinstance(environment, dict)
            else ""
        )
        metadata = projected.setdefault("metadata", {})
        if isinstance(metadata, dict):
            notices = runtime_notices_for_resources(
                required_resources,
                metadata.get("runtime_notices"),
            )
            if notices:
                metadata["runtime_notices"] = notices
            else:
                metadata.pop("runtime_notices", None)
    (output_dir / "task.toml").write_text(tomli_w.dumps(projected))
    instruction = problem_dir / "instruction.md"
    if instruction.is_file():
        shutil.copy2(instruction, output_dir / "instruction.md")
    calibration = problem_dir / "calibration.lock.json"
    if calibration.is_file():
        shutil.copy2(calibration, output_dir / "calibration.lock.json")
    task_inputs = output_dir / "_task_inputs"
    materialize_task_inputs(problem_dir, task_inputs)
    solution = problem_dir / "solution"
    if solution.is_dir():
        copy_directory_contents_safe(
            solution,
            output_dir / "solution",
            task_root=problem_dir,
        )

    agent = capabilities.agent_service
    assert agent is not None
    _write_capability_agent_environment(problem_dir, output_dir, agent, task_data)
    compose = capability_compose_data(problem_dir, output_dir, capabilities)
    (output_dir / "environment" / "docker-compose.yaml").write_text(
        yaml.safe_dump(compose, sort_keys=True, default_flow_style=False)
    )
    _write_capability_verifier(
        problem_dir,
        output_dir,
        capabilities,
        task_data,
    )

    task_toml_path = output_dir / "task.toml"
    _stamp_image_ref(task_toml_path, image_ref)
    if _grader_uses_llm(problem_dir):
        _ensure_verifier_env_for_llm(task_toml_path)
    for verifier_task_toml in (
        output_dir / "tests" / "task.toml",
        output_dir / "tests" / "task" / "task.toml",
    ):
        verifier_task_toml.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(task_toml_path, verifier_task_toml)
    return output_dir


def _export_harbor_into(
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
    task = load_task_toml(problem_dir)
    task_data = task.model_dump(mode="json", exclude_none=True)
    if is_capability_task(task):
        return _export_capability_harbor(
            problem_dir,
            output_dir,
            task_data=task_data,
            image_ref=image_ref,
            include_runtime_notices=include_runtime_notices,
        )

    _ = standalone
    output_dir.mkdir(parents=True, exist_ok=True)

    software_task = _is_software_task(task_data)
    top_level = ["task.toml", "instruction.md", "solution"]
    for name in top_level:
        source = problem_dir / name
        destination = output_dir / name
        if source.is_dir():
            if software_task:
                copy_directory_contents_safe(
                    source,
                    destination,
                    task_root=problem_dir,
                )
            else:
                _copytree_clean(source, destination)
        elif source.exists():
            shutil.copy2(source, destination)
    if software_task:
        seed_relative = _software_seed_relative_path(task_data)
        seed = resolve_task_path(
            problem_dir,
            seed_relative,
            label="software workspace seed",
        )
        seed_export = (
            output_dir / "starter"
            if seed_relative == "starter"
            else output_dir / "workspace_seed"
        )
        copy_directory_contents_safe(
            seed,
            seed_export,
            task_root=problem_dir,
        )

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
        _normalize_harbor_task_file(task_toml_path)
        if (
            software_task
            or load_task_toml(problem_dir).delivery.platform == "prometheus"
        ):
            _ensure_harbor_user_separation(task_toml_path)
        if _grader_uses_llm(problem_dir):
            _ensure_verifier_env_for_llm(task_toml_path)
        if include_runtime_notices:
            _stamp_runtime_notices(task_toml_path, problem_dir)
        shutil.copy2(task_toml_path, output_dir / "environment" / "task.toml")

    return output_dir


def export_harbor(
    problem_dir: Path,
    output_dir: Path,
    *,
    image_ref: str | None = None,
    standalone: bool = True,
    include_runtime_notices: bool = True,
    force: bool = False,
) -> Path:
    """Atomically write a Harbor-format task, requiring force for replacement."""
    problem_dir = problem_dir.resolve()
    task = load_task_toml(problem_dir)
    if is_capability_task(task) and image_ref and not is_digest_pinned_image(image_ref):
        raise ValueError(
            "--image for capability tasks must use an immutable "
            "<repository>@sha256:<64 hex> reference"
        )
    with staged_output_directory(
        output_dir,
        force=force,
        source_tree=problem_dir,
    ) as stage:
        _export_harbor_into(
            problem_dir,
            stage,
            image_ref=image_ref,
            standalone=standalone,
            include_runtime_notices=include_runtime_notices,
        )
    return output_dir
