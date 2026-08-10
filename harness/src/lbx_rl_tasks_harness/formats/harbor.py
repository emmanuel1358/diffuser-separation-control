from __future__ import annotations

import copy
import tomllib
from pathlib import Path
from typing import Any

from alignerr_plugin.schemas import TaskToml

from lbx_rl_tasks_harness.formats.problem_dir import (
    load_problem_dir,
    task_toml_metadata,
)
from lbx_rl_tasks_harness.models import (
    GroundTruthSpec,
    HarnessProblem,
    OutputSpec,
    ReferenceSpec,
)

_HARBOR_RESOURCE_FIELDS = (
    "build_timeout_sec",
    "cpus",
    "memory_mb",
    "storage_mb",
    "gpus",
    "gpu_types",
)
_HARBOR_RESOURCE_MARKERS = (
    "build_timeout_sec",
    "cpus",
    "memory_mb",
    "gpus",
    "gpu_types",
    "network_mode",
    "mcp_servers",
    "tpu",
)
_HARBOR_ONLY_ENVIRONMENT_FIELDS = (
    "allowed_hosts",
    "docker_image",
    "env",
    "healthcheck",
    "mcp_servers",
    "network_mode",
    "os",
    "skills_dir",
    "tpu",
    "workdir",
)
_HARBOR_NETWORK_MODES = {
    "bridge": "isolated",
    "isolated": "isolated",
    "internet": "internet",
    "no-network": "none",
    "none": "none",
    "public": "internet",
}


def _merge_native_resources(
    section: dict[str, Any], harbor_resources: dict[str, Any]
) -> None:
    resources = {
        name: harbor_resources[name]
        for name in _HARBOR_RESOURCE_FIELDS
        if name in harbor_resources
    }
    network_mode = harbor_resources.get("network_mode")
    if isinstance(network_mode, str) and network_mode in _HARBOR_NETWORK_MODES:
        resources["network"] = _HARBOR_NETWORK_MODES[network_mode]
    if not resources:
        return

    native_resources = section.get("resources")
    if native_resources is None:
        section["resources"] = resources
    elif isinstance(native_resources, dict):
        for name, value in resources.items():
            native_resources.setdefault(name, value)


def _is_harbor_artifact_list(value: object) -> bool:
    return (
        isinstance(value, list)
        and bool(value)
        and all(
            isinstance(item, dict)
            and "source" in item
            and "destination" in item
            and "name" not in item
            and "kind" not in item
            for item in value
        )
    )


def _adapt_harbor_task_toml(raw_task_toml: dict[str, Any]) -> TaskToml:
    """Translate known Harbor projections without relaxing the native schema."""
    native_data = copy.deepcopy(raw_task_toml)

    agent = native_data.get("agent")
    if isinstance(agent, dict):
        _merge_native_resources(agent, agent)
        agent.pop("allowed_hosts", None)
        agent.pop("network_mode", None)

    environment = native_data.get("environment")
    if isinstance(environment, dict):
        has_phase_resources = any(
            name in environment for name in _HARBOR_RESOURCE_MARKERS
        )
        if has_phase_resources:
            if agent is None:
                agent = {}
                native_data["agent"] = agent
            if isinstance(agent, dict):
                _merge_native_resources(agent, environment)
        for name in _HARBOR_ONLY_ENVIRONMENT_FIELDS:
            environment.pop(name, None)
        for name in _HARBOR_RESOURCE_FIELDS:
            if name != "storage_mb":
                environment.pop(name, None)

    verifier = native_data.get("verifier")
    if isinstance(verifier, dict):
        _merge_native_resources(verifier, verifier)
        verifier_env = verifier.get("env")
        if isinstance(verifier_env, dict):
            verifier["env"] = list(verifier_env)
        verifier_environment = verifier.pop("environment", None)
        if isinstance(verifier_environment, dict):
            _merge_native_resources(verifier, verifier_environment)
        verifier.pop("environment_mode", None)
        verifier.pop("collect", None)
        verifier.pop("allowed_hosts", None)
        verifier.pop("network_mode", None)

    native_data.pop("solution", None)
    if _is_harbor_artifact_list(native_data.get("artifacts")):
        native_data.pop("artifacts")

    return TaskToml.model_validate(native_data)


def _load_harbor_task_toml(task_dir: Path) -> tuple[TaskToml, dict[str, Any]]:
    with (task_dir / "task.toml").open("rb") as handle:
        raw_task_toml = tomllib.load(handle)
    return _adapt_harbor_task_toml(raw_task_toml), raw_task_toml


def load_harbor_dir(
    task_dir: Path, source_problem_dir: Path | None = None
) -> HarnessProblem:
    task_dir = task_dir.resolve()
    task_toml, raw_task_toml = _load_harbor_task_toml(task_dir)
    prompt = (task_dir / "instruction.md").read_text()
    source_problem = (
        load_problem_dir(source_problem_dir) if source_problem_dir else None
    )
    exported_grader = task_dir / "environment" / "scorer"
    grader_dir = (
        source_problem.grader_dir
        if source_problem
        else exported_grader if exported_grader.is_dir() else None
    )
    private_dir = (
        source_problem.private_dir
        if source_problem
        else grader_dir / "data" if grader_dir else None
    )
    ground_truth = (
        source_problem.ground_truth
        if source_problem
        else GroundTruthSpec(
            render_command=task_toml.ground_truth.render_command,
            render_outputs=[
                OutputSpec(
                    path=out.path,
                    required=out.required,
                    description=out.description,
                )
                for out in task_toml.ground_truth.render_outputs
            ],
            score_epsilon=task_toml.ground_truth.score_epsilon,
            continuous_score_epsilon=task_toml.ground_truth.continuous_score_epsilon,
            in_container=task_toml.ground_truth.in_container,
            zero_anchor_epsilon=task_toml.ground_truth.zero_anchor_epsilon,
        )
    )
    reference = (
        source_problem.reference
        if source_problem
        else ReferenceSpec(
            execution=task_toml.reference.execution,
            cache_dir=task_toml.reference.cache_dir,
            entrypoint=task_toml.reference.entrypoint,
            proof_mode=task_toml.reference.proof_mode,
        )
    )
    raw_environment = raw_task_toml.get("environment")
    image = (
        raw_environment.get("docker_image")
        if isinstance(raw_environment, dict)
        and isinstance(raw_environment.get("docker_image"), str)
        else None
    )

    return HarnessProblem(
        id=source_problem.id if source_problem else task_toml.task.name.split("/")[-1],
        source_format="harbor",
        prompt=prompt,
        outputs=[
            OutputSpec(
                path=out.path, required=out.required, description=out.description
            )
            for out in task_toml.outputs
        ],
        task_dir=task_dir,
        source_problem_dir=(
            source_problem.source_problem_dir if source_problem else None
        ),
        grader_dir=grader_dir,
        private_dir=private_dir,
        image=image,
        required_resources=(
            source_problem.required_resources
            if source_problem
            else task_toml.environment.required_resources
        ),
        required_tools=list(task_toml.runner.required_tools),
        ground_truth=ground_truth,
        reference=reference,
        metadata={
            **task_toml_metadata(task_toml),
            "harbor_task_dir": str(task_dir),
            "harbor_task_toml": raw_task_toml,
            "tests": str(task_dir / "tests" / "test.sh"),
            "runtime_notices": task_toml.metadata.get("runtime_notices", []),
        },
        taiga_problem=source_problem.taiga_problem if source_problem else None,
    )
