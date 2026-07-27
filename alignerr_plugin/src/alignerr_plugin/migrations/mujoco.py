"""Convert legacy MuJoCo task paths and metadata to the native ISO layout."""

from __future__ import annotations

import json
import re
import shutil
import tomllib
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import tomli_w

from alignerr_plugin.proof import verify_build_proof
from alignerr_plugin.schemas import TaskToml
from alignerr_plugin.task_metadata import DOMAINS_BY_TASK_TYPE, REWARD_TYPES, normalize_enum_value
from alignerr_plugin.utils import load_task_toml

Layout = Literal["native", "harbor"]

_CPU_TIERS = (
    (2, 6, "2vcpu+6gib"),
    (4, 16, "4vcpu+16gib"),
    (6, 32, "6vcpu+32gib"),
    (8, 64, "8vcpu+64gib"),
    (16, 64, "16vcpu+64gib"),
    (16, 128, "16vcpu+128gib"),
)
_H100_TIERS = (
    (3, 25, "3vcpu+25gib+h100/8"),
    (6, 50, "6vcpu+50gib+h100/4"),
    (12, 100, "12vcpu+100gib+h100/2"),
    (24, 200, "24vcpu+200gib+h100/1"),
)
_SECURE_PRIVATE_PERMISSIONS = """RUN rm -rf /mcp_server/grader/data \\
    && chown -R root:root /mcp_server/data /mcp_server/grader \\
    && find /mcp_server/data /mcp_server/grader -type d -exec chmod 0700 {} + \\
    && find /mcp_server/data /mcp_server/grader -type f -exec chmod 0600 {} + \\
    && chmod 0700 /mcp_server"""
_LEGACY_RUNTIME_VENV = "/mcp_server/.venv"
_CURRENT_RUNTIME_VENV = "/opt/lbx-runtime/.venv"
_WORLD_MCP_CHMOD = re.compile(
    r"\bchmod\s+0?755\s+[^;&\n]*?(?<!\S)/mcp_server(?=\s|\\|$)"
)
_CHMOD_755_COMMAND = re.compile(
    r"\bchmod\s+0?755\s+(?P<args>.*?)(?=(?:\s+&&|\s*;|\s*$))",
    re.MULTILINE,
)


@dataclass(frozen=True)
class MujocoMigrationResult:
    source: Path
    destination: Path
    source_layout: Layout
    changes: tuple[str, ...]
    blockers: tuple[str, ...]

    @property
    def ready_for_validation(self) -> bool:
        return not self.blockers


def _detect_layout(source: Path) -> Layout:
    if (source / "scorer" / "compute_score.py").is_file():
        return "native"
    if (source / "environment" / "scorer" / "compute_score.py").is_file():
        return "harbor"
    raise ValueError(
        f"{source} is neither a native nor Harbor MuJoCo task: expected "
        "scorer/compute_score.py or environment/scorer/compute_score.py"
    )


def _copy_clean(source: Path, destination: Path) -> None:
    shutil.copytree(
        source,
        destination,
        ignore=shutil.ignore_patterns(".alignerr", ".taiga_submit.json", "__pycache__", "*.pyc"),
    )


def _materialize_native(source: Path, destination: Path, layout: Layout) -> None:
    if destination == source:
        if layout != "native":
            raise ValueError(
                "in-place migration is only supported for native tasks; Harbor exports must use a separate destination"
            )
        return
    if destination.exists():
        raise FileExistsError(f"refusing to overwrite existing migration destination: {destination}")
    if layout == "native":
        _copy_clean(source, destination)
        return

    destination.mkdir(parents=True)
    environment = source / "environment"
    for name in ("task.toml", "instruction.md", "README.md", "solution", "tests"):
        item = source / name
        if item.is_dir():
            _copy_clean(item, destination / name)
        elif item.is_file():
            shutil.copy2(item, destination / name)
    for name in ("data", "scorer"):
        item = environment / name
        if item.is_dir():
            _copy_clean(item, destination / name)
    calibration_lock = environment / "calibration.lock.json"
    if calibration_lock.is_file():
        shutil.copy2(calibration_lock, destination / calibration_lock.name)
    authored_environment = environment / "source_environment"
    if not authored_environment.is_dir():
        raise ValueError(
            f"{source} is a Harbor export without environment/source_environment; "
            "the generated Harbor Dockerfile cannot be used as a native task Dockerfile"
        )
    _copy_clean(authored_environment, destination / "environment")


def _minimum_tier(tiers: tuple[tuple[int, int, str], ...], cpus: int, memory_gib: int) -> str:
    for tier_cpus, tier_memory, name in tiers:
        if tier_cpus >= cpus and tier_memory >= memory_gib:
            return name
    largest_cpus, largest_memory, _ = tiers[-1]
    raise ValueError(
        f"legacy MuJoCo task requests {cpus} CPUs and {memory_gib} GiB RAM, "
        f"above the largest compatible Taiga tier ({largest_cpus} CPUs, "
        f"{largest_memory} GiB); a reviewed manual mapping is required"
    )


def legacy_resources_to_taiga(environment: dict[str, Any], *, dockerfile_text: str = "") -> str:
    """Map retired free-form resources without under-provisioning CPU or RAM."""
    current = str(environment.get("required_resources") or "").strip()
    if current:
        return current
    cpus = max(1, int(environment.get("cpus", 2)))
    memory_mb = max(1, int(environment.get("memory_mb", 6144)))
    memory_gib = (memory_mb + 1023) // 1024
    gpu_count = max(0, int(environment.get("gpus", 0)))
    if "gpus" not in environment and re.search(r"BASE_IMAGE\s*=.*(?:gpu|cuda)", dockerfile_text, re.IGNORECASE):
        gpu_count = 1
    if gpu_count:
        # The retired schema requested whole GPUs, so preserve that contract.
        if gpu_count == 1:
            full_cpus, full_memory, full_tier = _H100_TIERS[-1]
            if cpus <= full_cpus and memory_gib <= full_memory:
                return full_tier
            raise ValueError(
                f"legacy MuJoCo task requests one GPU, {cpus} CPUs, and "
                f"{memory_gib} GiB RAM, above the full-H100 Taiga tier "
                f"({full_cpus} CPUs, {full_memory} GiB); a reviewed manual "
                "mapping is required"
            )
        raise ValueError(
            f"legacy MuJoCo task requests {gpu_count} GPUs; current Taiga tiers expose "
            "at most one H100 and require a reviewed manual mapping"
        )
    return _minimum_tier(_CPU_TIERS, cpus, memory_gib)


def infer_mujoco_domain(config: dict[str, Any], task_id: str) -> str:
    outputs = config.get("outputs") or []
    output_paths = " ".join(str(item.get("path", "")) for item in outputs)
    description = str((config.get("task") or {}).get("description", ""))
    text = f"{task_id} {description} {output_paths}".lower()
    rules = (
        ("model.xml", "model_environment_construction"),
        ("biped", "balance_recovery"),
        ("upright", "balance_recovery"),
        ("recovery", "balance_recovery"),
        ("slider", "contact_rich_manipulation"),
        ("pushing", "contact_rich_manipulation"),
        ("glider", "swimming_aquatic_control"),
        ("underwater", "swimming_aquatic_control"),
        ("snake", "navigation_mobility"),
        ("maze", "navigation_mobility"),
        ("trailer", "controller_planner_authoring"),
        ("hopper", "locomotion"),
        ("wheg", "wheeled_mobile_control"),
        ("train", "policy_training_improvement"),
    )
    for marker, domain in rules:
        if marker in text:
            return domain
    return "controller_planner_authoring"


def _normalize_task_toml(task_path: Path, *, domain: str | None, reward_type: str | None) -> list[str]:
    with task_path.open("rb") as handle:
        config = tomllib.load(handle)
    if str((config.get("difficulty") or {}).get("task_type", "mujoco")).lower() != "mujoco":
        raise ValueError(f"{task_path} does not declare a MuJoCo task")

    changes: list[str] = []
    environment = dict(config.get("environment") or {})
    dockerfile = task_path.parent / "environment" / "Dockerfile"
    dockerfile_text = dockerfile.read_text() if dockerfile.is_file() else ""
    required_resources = legacy_resources_to_taiga(environment, dockerfile_text=dockerfile_text)
    normalized_environment = {
        "required_resources": required_resources,
        "storage_mb": int(environment.get("storage_mb", 50000)),
        "allow_internet": bool(environment.get("allow_internet", True)),
    }
    for key in ("base_flavor", "hidden_env"):
        if key in environment:
            normalized_environment[key] = environment[key]
    if environment != normalized_environment:
        changes.append(f"mapped legacy resources to environment.required_resources={required_resources!r}")
    config["environment"] = normalized_environment

    task_name = str((config.get("task") or {}).get("name") or task_path.parent.name)
    task_id = task_name.rsplit("/", 1)[-1]
    normalized_difficulty = dict(config.get("difficulty") or {})
    existing_domain = normalize_enum_value(normalized_difficulty.get("domain"))
    selected_domain = domain or (
        existing_domain if existing_domain in DOMAINS_BY_TASK_TYPE["mujoco"] else infer_mujoco_domain(config, task_id)
    )
    existing_reward_type = normalize_enum_value(normalized_difficulty.get("reward_type"))
    selected_reward_type = reward_type or (
        existing_reward_type if existing_reward_type in REWARD_TYPES else "multi_deterministic_rubrics"
    )
    normalized_difficulty.update(
        {
            "task_type": "mujoco",
            "domain": selected_domain,
            "reward_type": selected_reward_type,
        }
    )
    if config.get("difficulty") != normalized_difficulty:
        changes.append(f"set MuJoCo domain={selected_domain!r} and reward_type={selected_reward_type!r}")
    config["difficulty"] = normalized_difficulty

    for retired in ("policy", "scorer"):
        if retired in config:
            config.pop(retired)
            changes.append(f"removed retired [{retired}] task.toml section")
    config["schema_version"] = "1.1"
    TaskToml.model_validate(config)
    task_path.write_text(tomli_w.dumps(config))
    return changes


def _secure_dockerfile(path: Path) -> bool:
    text = path.read_text()

    def secure_chmod(match: re.Match[str]) -> str:
        raw_args = match.group("args")
        continued = raw_args.rstrip().endswith("\\")
        args = raw_args.rstrip().removesuffix("\\").split()
        if "/mcp_server" not in args:
            return match.group(0)
        public_args = [arg for arg in args if arg != "/mcp_server"]
        commands = []
        if public_args:
            commands.append(f"chmod 0755 {' '.join(public_args)}")
        commands.append("chmod 0700 /mcp_server")
        suffix = " \\" if continued else ""
        return " && ".join(commands) + suffix

    normalized = _CHMOD_755_COMMAND.sub(secure_chmod, text)
    if _SECURE_PRIVATE_PERMISSIONS not in normalized:
        normalized = normalized.rstrip() + "\n\n" + _SECURE_PRIVATE_PERMISSIONS + "\n"
    if normalized == text:
        return False
    path.write_text(normalized)
    return True


def _text_files_containing(path: Path, needle: str) -> list[Path]:
    encoded = needle.encode()
    matches: list[Path] = []
    for candidate in sorted(path.rglob("*")):
        if (
            candidate.is_file()
            and not candidate.is_symlink()
            and _file_contains(candidate, encoded)
        ):
            try:
                candidate.read_text()
            except UnicodeDecodeError:
                continue
            matches.append(candidate)
    return matches


def _file_contains(path: Path, needle: bytes) -> bool:
    overlap = b""
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            content = overlap + chunk
            if needle in content:
                return True
            overlap = content[-(len(needle) - 1) :] if len(needle) > 1 else b""
    return False


def _normalize_runtime_paths(destination: Path) -> list[Path]:
    """Point every task-owned text probe at the current base-image venv."""
    changed: list[Path] = []
    for path in _text_files_containing(destination, _LEGACY_RUNTIME_VENV):
        text = path.read_text()
        path.write_text(text.replace(_LEGACY_RUNTIME_VENV, _CURRENT_RUNTIME_VENV))
        changed.append(path.relative_to(destination))
    return changed


def _ensure_required_paths(destination: Path) -> list[str]:
    changes: list[str] = []
    for relative in (Path("data"), Path("scorer/data")):
        directory = destination / relative
        if not directory.exists():
            directory.mkdir(parents=True)
            (directory / ".gitkeep").touch()
            changes.append(f"created required {relative.as_posix()}/ directory")
    return changes


def _ensure_metadata(destination: Path) -> bool:
    path = destination / "metadata.json"
    if path.is_file():
        return False
    with (destination / "task.toml").open("rb") as handle:
        config = tomllib.load(handle)
    task = config["task"]
    task_id = str(task["name"]).rsplit("/", 1)[-1]
    path.write_text(
        json.dumps(
            {
                "benchmark": "taiga_task",
                "problem_data": {
                    "instance_id": task_id,
                    "description": str(task.get("description") or ""),
                },
            },
            indent=2,
        )
        + "\n"
    )
    return True


def _normalize_harbor_test_paths(destination: Path) -> bool:
    test_sh = destination / "tests" / "test.sh"
    if not test_sh.is_file():
        return False
    text = test_sh.read_text()
    normalized = text.replace("--workspace /app", "--workspace /tmp/output")
    if normalized == text:
        return False
    test_sh.write_text(normalized)
    return True


def _sealed_evaluation_plan_issue(path: Path) -> str | None:
    if not path.is_file():
        return "is missing"
    try:
        payload = json.loads(path.read_text())
        if not isinstance(payload, dict):
            raise ValueError("plan must be a JSON object")
        from grading.evaluation.plan import validate_serialized_plan

        validate_serialized_plan(payload)
    except (OSError, json.JSONDecodeError, ValueError) as exc:
        return f"is invalid: {exc}"
    return None


def _calibration_lock_issue(path: Path) -> str | None:
    if not path.is_file():
        return "is missing"
    try:
        from grading.evaluation.lock import load_calibration_lock

        load_calibration_lock(path)
    except (OSError, RuntimeError, ValueError) as exc:
        return f"is invalid: {exc}"
    return None


def _remaining_blockers(destination: Path, reward_type: str) -> list[str]:
    blockers: list[str] = []
    required = (
        "task.toml",
        "metadata.json",
        "instruction.md",
        "environment/Dockerfile",
        "scorer/compute_score.py",
        "solution/solve.sh",
    )
    for relative in required:
        if not (destination / relative).is_file():
            blockers.append(f"missing required native path: {relative}")
    scorer_path = destination / "scorer" / "compute_score.py"
    if scorer_path.is_file():
        scorer = scorer_path.read_text()
    else:
        scorer = ""
    evaluation_plan = destination / "scorer" / "evaluation.plan.json"
    calibration_lock = destination / "calibration.lock.json"
    if reward_type == "multi_deterministic_rubrics":
        if not re.search(r"\bTASK\s*=\s*RubricTask\s*\(", scorer):
            blockers.append(
                "scorer/compute_score.py still needs semantic conversion to "
                "TASK = RubricTask(...)"
            )
        if issue := _sealed_evaluation_plan_issue(evaluation_plan):
            blockers.append(
                "multi_deterministic_rubrics requires a valid generated "
                f"scorer/evaluation.plan.json; file {issue}"
            )
    elif reward_type == "continuous_scoring_function":
        if re.search(r"\bTASK\s*=\s*PolicyEvaluationTask\s*\(", scorer):
            if issue := _sealed_evaluation_plan_issue(evaluation_plan):
                blockers.append(
                    "PolicyEvaluationTask requires a valid generated "
                    f"scorer/evaluation.plan.json; file {issue}"
                )
        elif re.search(r"\bTASK\s*=\s*ContinuousTask\s*\(", scorer):
            if issue := _calibration_lock_issue(calibration_lock):
                blockers.append(
                    "ContinuousTask requires a valid generated "
                    f"calibration.lock.json; file {issue}"
                )
        else:
            blockers.append(
                "continuous_scoring_function requires semantic conversion to "
                "TASK = ContinuousTask(...) with calibration.lock.json or "
                "TASK = PolicyEvaluationTask(...) with scorer/evaluation.plan.json"
            )
    legacy_runtime_paths = _text_files_containing(destination, _LEGACY_RUNTIME_VENV)
    for path in legacy_runtime_paths:
        blockers.append(
            f"{path.relative_to(destination).as_posix()} still targets retired "
            f"runtime venv {_LEGACY_RUNTIME_VENV} instead of {_CURRENT_RUNTIME_VENV}"
        )
    dockerfile = destination / "environment" / "Dockerfile"
    if dockerfile.is_file():
        dockerfile_text = dockerfile.read_text()
        if _WORLD_MCP_CHMOD.search(dockerfile_text):
            blockers.append("environment/Dockerfile still grants traversal on /mcp_server")
        if "chmod 0700 /mcp_server" not in dockerfile_text:
            blockers.append("environment/Dockerfile does not make /mcp_server root-only")
    proof_ok, proof_errors, _ = verify_build_proof(destination)
    if not proof_ok:
        blockers.extend(f"local build proof: {error}" for error in proof_errors)
    return blockers


def migrate_legacy_mujoco_task(
    source: Path,
    destination: Path,
    *,
    domain: str | None = None,
    reward_type: str | None = None,
) -> MujocoMigrationResult:
    """Materialize a canonical task without pretending to rewrite grader semantics.

    Passing the same native path as source and destination performs an in-place
    migration. Harbor exports always require a separate destination because their
    nested generated environment must be disentangled from authored files.
    """
    source = source.resolve()
    destination = destination.resolve()
    layout = _detect_layout(source)
    _materialize_native(source, destination, layout)

    changes = _normalize_task_toml(destination / "task.toml", domain=domain, reward_type=reward_type)
    changes.extend(_ensure_required_paths(destination))
    if _ensure_metadata(destination):
        changes.append("generated native metadata.json from task.toml")
    if layout == "harbor" and _normalize_harbor_test_paths(destination):
        changes.append("changed Harbor grader workspace from /app to /tmp/output")
    normalized_paths = _normalize_runtime_paths(destination)
    if normalized_paths:
        rendered_paths = ", ".join(path.as_posix() for path in normalized_paths)
        changes.append(
            f"changed runtime venv from {_LEGACY_RUNTIME_VENV} to "
            f"{_CURRENT_RUNTIME_VENV} in {rendered_paths}"
        )
    dockerfile = destination / "environment" / "Dockerfile"
    if dockerfile.is_file():
        if _secure_dockerfile(dockerfile):
            changes.append("made /mcp_server and private grader mounts root-only")

    return MujocoMigrationResult(
        source=source,
        destination=destination,
        source_layout=layout,
        changes=tuple(changes),
        blockers=tuple(
            _remaining_blockers(
                destination,
                load_task_toml(destination).difficulty.reward_type,
            )
        ),
    )
