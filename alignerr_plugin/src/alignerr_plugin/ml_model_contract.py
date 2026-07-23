"""Hard contract for continuous ML committed models + training provenance.

Trusted CI / ground-truth / validate must never train. Authors commit:

* a training entrypoint (provenance / reproducibility only)
* trained model artifact(s)
* ``model.manifest.json`` with digests
* an inference-only ``solution.py`` / ``solve.sh``

This module is the shared source of truth for both ``lbx-rl-template validate``
and harness ground-truth / calibration.
"""

from __future__ import annotations

import ast
import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

MODEL_MANIFEST_FILENAME = "model.manifest.json"
FORBIDDEN_GENERATED_SCORE_FILES = frozenset({"results.txt"})
# Allowed as an additive Tier-B static artifact alongside the model contract;
# never a substitute for train.py / model weights / manifest.
TIER_B_STATIC_ARTIFACTS = frozenset({"submission.csv"})

# Attribute calls that indicate fitting/backprop. Deliberately exclude
# ``.train()`` — PyTorch uses Module.train()/Module.eval() to toggle mode
# during legitimate inference packaging.
_TRAINING_ATTR_CALLS = frozenset(
    {
        "fit",
        "partial_fit",
        "fit_transform",
        "backward",
        "zero_grad",
    }
)
# Bare-name calls (``train(...)`` as a local helper), not attribute access.
_TRAINING_NAME_CALLS = frozenset(
    {
        "fit",
        "partial_fit",
        "fit_transform",
        "train",
        "backward",
    }
)
_WRITE_OPEN_MODES = frozenset(
    {
        "w",
        "wb",
        "wt",
        "w+",
        "wb+",
        "wt+",
        "a",
        "ab",
        "at",
        "a+",
        "ab+",
        "at+",
        "x",
        "xb",
        "xt",
    }
)
_MODEL_WRITE_METHODS = frozenset(
    {
        "write_text",
        "write_bytes",
        "dump",
        "save",
        "savez",
        "savez_compressed",
        "to_csv",
        "to_parquet",
        "to_pickle",
        "to_json",
    }
)
_ARTIFACT_LOAD_ALLOWLIST = frozenset(
    {
        "copy",
        "copy2",
        "copyfile",
        "read_text",
        "read_bytes",
        "load",
        "loads",
        "open",
        "Path",
        "is_file",
        "exists",
        "torch.load",
        "joblib.load",
        "pickle.load",
        "json.load",
        "json.loads",
        "np.load",
        "numpy.load",
        "pd.read_pickle",
        "pandas.read_pickle",
    }
)
# Match invoked training scripts only (train.py / train.sh / training.py), not
# data files like train.csv that share the "train" stem. Require an explicit
# runner or ./ prefix so comments / echo text mentioning train.py do not match.
_SHELL_TRAIN_RE = re.compile(r"""(?ix)
    (?:^|[\s;&|`(])
    (?:
        (?:python(?:3(?:\.\d+)?)?|bash|sh|uv\s+run|pipenv\s+run|poetry\s+run)
        \s+
        (?:["'][^"']*["']\s+)*
        (?P<entry>(?:\./)?(?:[\w./-]+/)?(?P<name>train(?:ing)?\.(?:py|sh|bash)))
      |
        \./(?P<entry2>(?:[\w./-]+/)?(?P<name2>train(?:ing)?\.(?:py|sh|bash)))
    )
    (?:\s|["']|$)
    """)
_SHELL_MODEL_REF_RE = re.compile(r"""(?ix)
    (?:model\.manifest\.json|
       \bmodel\.(?:json|pt|pth|pkl|joblib|bin|safetensors|onnx|h5|hdf5|ckpt|weights)\b|
       \bweights?\b|
       \bcheckpoint\b|
       \bshutil\.copy|
       \bcp\s+)
    """)


@dataclass(frozen=True)
class StrategyContract:
    """Validated continuous-ML strategy surface."""

    role: str
    strategy_dir: Path
    manifest: dict[str, Any]
    training_entrypoint: str
    inference_entrypoint: str
    artifact_paths: tuple[str, ...]


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _strategy_relative(path_raw: str, *, field: str, role: str) -> Path:
    relative = Path(path_raw)
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError(f"{role} model manifest {field} must be strategy-relative")
    return relative


def validate_committed_model_manifest(
    strategy_dir: Path, *, role: str
) -> dict[str, Any]:
    """Validate a committed trained-model manifest without loading the model."""
    manifest_path = strategy_dir / MODEL_MANIFEST_FILENAME
    if not manifest_path.is_file():
        raise ValueError(
            f"{role} strategy is missing {MODEL_MANIFEST_FILENAME}: {strategy_dir}"
        )
    try:
        manifest = json.loads(manifest_path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"invalid {role} model manifest: {exc}") from exc
    if not isinstance(manifest, dict) or manifest.get("schema_version") != "1.0":
        raise ValueError(f"{role} model manifest must use schema_version '1.0'")
    if manifest.get("role") != role:
        raise ValueError(
            f"{role} model manifest role must be {role!r}, got {manifest.get('role')!r}"
        )
    entrypoints: dict[str, str] = {}
    for field in ("training_entrypoint", "inference_entrypoint"):
        raw = manifest.get(field)
        if not isinstance(raw, str) or not raw:
            raise ValueError(f"{role} model manifest is missing {field}")
        relative = _strategy_relative(raw, field=field, role=role)
        if not (strategy_dir / relative).is_file():
            raise ValueError(f"{role} model manifest {field} does not exist: {raw}")
        entrypoints[field] = raw
    if entrypoints["training_entrypoint"] == entrypoints["inference_entrypoint"]:
        raise ValueError(
            f"{role} training_entrypoint and inference_entrypoint must be distinct; "
            "Trusted CI / ground-truth must never train"
        )
    if not isinstance(manifest.get("seed"), int):
        raise ValueError(f"{role} model manifest seed must be an integer")
    training_data = manifest.get("public_training_data")
    if not isinstance(training_data, dict):
        raise ValueError(f"{role} model manifest is missing public_training_data")
    training_path_raw = training_data.get("path")
    training_sha = training_data.get("sha256")
    if not isinstance(training_path_raw, str) or not isinstance(training_sha, str):
        raise ValueError(
            f"{role} model manifest public_training_data needs path and sha256"
        )
    task_root = next(
        (
            parent
            for parent in (strategy_dir, *strategy_dir.parents)
            if (parent / "metadata.json").is_file() or (parent / "task.toml").is_file()
        ),
        None,
    )
    if task_root is None:
        raise ValueError(f"could not resolve task root for {role} model manifest")
    training_path = (strategy_dir / training_path_raw).resolve()
    try:
        training_path.relative_to(task_root.resolve())
    except ValueError as exc:
        raise ValueError(
            f"{role} model training data escapes the task root: {training_path_raw}"
        ) from exc
    if not training_path.is_file():
        raise ValueError(f"{role} model training data is missing: {training_path_raw}")
    actual_training_sha = file_sha256(training_path)
    if actual_training_sha != training_sha:
        raise ValueError(
            f"{role} model is stale for public training data {training_path_raw}: "
            f"expected {training_sha}, got {actual_training_sha}"
        )
    artifacts = manifest.get("artifacts")
    if not isinstance(artifacts, list) or not artifacts:
        raise ValueError(f"{role} model manifest must declare trained artifacts")
    for entry in artifacts:
        if not isinstance(entry, dict):
            raise ValueError(f"{role} model manifest artifact entries must be objects")
        raw_path = entry.get("path")
        expected = entry.get("sha256")
        if not isinstance(raw_path, str) or not isinstance(expected, str):
            raise ValueError(f"{role} model manifest artifact needs path and sha256")
        relative = _strategy_relative(raw_path, field="artifact path", role=role)
        artifact = strategy_dir / relative
        if not artifact.is_file():
            raise ValueError(f"{role} trained model artifact is missing: {raw_path}")
        actual = file_sha256(artifact)
        if actual != expected:
            raise ValueError(
                f"{role} model artifact digest mismatch for {raw_path}: "
                f"expected {expected}, got {actual}"
            )
    return manifest


def _call_name(node: ast.AST) -> str | None:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        parts: list[str] = []
        cur: ast.AST | None = node
        while isinstance(cur, ast.Attribute):
            parts.append(cur.attr)
            cur = cur.value
        if isinstance(cur, ast.Name):
            parts.append(cur.id)
            return ".".join(reversed(parts))
        return parts[0] if parts else None
    return None


def _string_literals(node: ast.AST) -> list[str]:
    return [
        child.value
        for child in ast.walk(node)
        if isinstance(child, ast.Constant) and isinstance(child.value, str)
    ]


def _path_mentions(names: Iterable[str], *, text: str, tree: ast.AST) -> bool:
    lowered = text.lower()
    for name in names:
        if name.lower() in lowered:
            return True
        for literal in _string_literals(tree):
            if name in literal or Path(name).name in literal:
                return True
    return False


def _literal_names_training_entrypoint(
    literal: str, *, training_rel: str, training_name: str
) -> bool:
    """True when ``literal`` names the training script, not e.g. train.csv."""
    if not literal:
        return False
    candidate = Path(literal)
    if candidate.name == training_name:
        return True
    if literal == training_rel or literal.endswith("/" + training_rel):
        return True
    # Exact path equality only — never match by stem (train.csv vs train.py).
    return False


def _call_opens_for_write(node: ast.Call, *, attr: str) -> bool:
    """Detect ``open(..., \"w\")`` / ``Path(...).open(\"wb\")`` rewrite modes."""
    if attr != "open" and _call_name(node.func) != "open":
        return False
    mode_values: list[str] = []
    for arg in node.args[1:]:
        if isinstance(arg, ast.Constant) and isinstance(arg.value, str):
            mode_values.append(arg.value)
    # Path(...).open("w") passes mode as the first positional arg.
    if isinstance(node.func, ast.Attribute) and node.args:
        first = node.args[0]
        if isinstance(first, ast.Constant) and isinstance(first.value, str):
            mode_values.append(first.value)
    for keyword in node.keywords:
        if keyword.arg == "mode" and isinstance(keyword.value, ast.Constant):
            value = keyword.value.value
            if isinstance(value, str):
                mode_values.append(value)
    return any(
        mode in _WRITE_OPEN_MODES or mode.split("+", 1)[0] in _WRITE_OPEN_MODES
        for mode in mode_values
    )


def _call_path_literals(node: ast.Call) -> list[str]:
    """Path-like string literals on a call, including Path(...).open receivers."""
    literals = _string_literals(node)
    if isinstance(node.func, ast.Attribute):
        literals.extend(_string_literals(node.func.value))
    return literals


def _python_inference_only_issues(
    source: str,
    *,
    role: str,
    inference_rel: str,
    training_rel: str,
    artifact_paths: tuple[str, ...],
) -> list[str]:
    issues: list[str] = []
    try:
        tree = ast.parse(source)
    except SyntaxError as exc:
        return [f"{role} inference entrypoint {inference_rel} has syntax error: {exc}"]

    training_stem = Path(training_rel).stem
    training_name = Path(training_rel).name
    for node in ast.walk(tree):
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            modules: list[str] = []
            if isinstance(node, ast.Import):
                modules = [alias.name for alias in node.names]
            elif node.module:
                modules = [node.module]
            for module in modules:
                leaf = module.rsplit(".", 1)[-1]
                if leaf in {training_stem, training_name} or module.endswith(
                    f".{training_stem}"
                ):
                    issues.append(
                        f"{role} inference entrypoint {inference_rel} must not import "
                        f"training entrypoint {training_rel}"
                    )
        if not isinstance(node, ast.Call):
            continue
        name = _call_name(node.func)
        if name is None:
            continue
        attr = name.rsplit(".", 1)[-1]
        # Bare ``train(...)`` is training; ``module.train()`` / ``.train(False)``
        # is PyTorch mode toggling and must not fail closed.
        training_call = attr in _TRAINING_ATTR_CALLS or (
            name in _TRAINING_NAME_CALLS and "." not in name
        )
        if attr == "step" and "optim" in name.lower():
            training_call = True
        if training_call:
            issues.append(
                f"{role} inference entrypoint {inference_rel} looks like a training "
                f"script (call `{name}`); Trusted CI / ground-truth must only load "
                "committed weights and run inference"
            )
        literals = _string_literals(node)
        if any(
            _literal_names_training_entrypoint(
                lit, training_rel=training_rel, training_name=training_name
            )
            for lit in literals
        ) and attr in {"system", "run", "Popen", "check_call", "check_output", "call"}:
            issues.append(
                f"{role} inference entrypoint {inference_rel} must not shell out to "
                f"training entrypoint {training_rel}"
            )
        if attr in _MODEL_WRITE_METHODS or _call_opens_for_write(node, attr=attr):
            path_literals = _call_path_literals(node)
            for lit in path_literals:
                for artifact in artifact_paths:
                    if artifact in lit or Path(artifact).name == Path(lit).name:
                        # Writing into /tmp/output is fine; rewriting committed
                        # strategy artifacts is training/re-export and forbidden.
                        if "/tmp/output" in lit or "LBT_OUTPUT" in lit:
                            continue
                        issues.append(
                            f"{role} inference entrypoint {inference_rel} must not "
                            f"overwrite committed model artifact {artifact}; train "
                            f"via {training_rel} and commit weights"
                        )

    required_names = (MODEL_MANIFEST_FILENAME, *artifact_paths)
    if not _path_mentions(required_names, text=source, tree=tree):
        issues.append(
            f"{role} inference entrypoint {inference_rel} must load "
            f"{MODEL_MANIFEST_FILENAME} or a declared model artifact "
            f"({', '.join(artifact_paths)}); inference-only packaging required"
        )
    else:
        # Prefer an explicit load/copy pattern when artifacts are referenced.
        has_loader = False
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            name = _call_name(node.func) or ""
            attr = name.rsplit(".", 1)[-1]
            if name in _ARTIFACT_LOAD_ALLOWLIST or attr in _ARTIFACT_LOAD_ALLOWLIST:
                if _path_mentions(required_names, text=ast.dump(node), tree=node):
                    has_loader = True
                    break
                # Heuristic: copy/load call near artifact mentions in source.
                if any(
                    artifact in source or Path(artifact).name in source
                    for artifact in required_names
                ):
                    has_loader = True
                    break
        if not has_loader and not any(
            token in source
            for token in (
                "shutil.copy",
                "shutil.copy2",
                "read_text",
                "read_bytes",
                "torch.load",
                "joblib.load",
                "pickle.load",
                "json.loads",
                "json.load",
            )
        ):
            issues.append(
                f"{role} inference entrypoint {inference_rel} must copy/load committed "
                "model bytes (e.g. shutil.copy2 / torch.load / json.loads); "
                "unrecognized inference packaging fails closed"
            )
    return issues


def _shell_inference_only_issues(
    source: str,
    *,
    role: str,
    inference_rel: str,
    training_rel: str,
    artifact_paths: tuple[str, ...],
) -> list[str]:
    issues: list[str] = []
    training_name = Path(training_rel).name
    # Strip shell comments so prose about train.py cannot false-positive.
    code_only = "\n".join(line.split("#", 1)[0] for line in source.splitlines())
    for match in _SHELL_TRAIN_RE.finditer(code_only):
        name = match.group("name") or match.group("name2") or ""
        if Path(name).name == training_name:
            issues.append(
                f"{role} inference entrypoint {inference_rel} must not invoke "
                f"training entrypoint {training_rel}"
            )
            break
    required = (MODEL_MANIFEST_FILENAME, *artifact_paths)
    if not (
        _SHELL_MODEL_REF_RE.search(source)
        or any(name in source for name in required)
        or any(Path(name).name in source for name in artifact_paths)
    ):
        issues.append(
            f"{role} inference entrypoint {inference_rel} must reference "
            f"{MODEL_MANIFEST_FILENAME} or a declared model artifact "
            f"({', '.join(artifact_paths)})"
        )
    return issues


def inference_only_issues(
    strategy_dir: Path,
    *,
    role: str,
    manifest: dict[str, Any],
) -> list[str]:
    """Static checks that the inference entrypoint does not train."""
    training_rel = str(manifest["training_entrypoint"])
    inference_rel = str(manifest["inference_entrypoint"])
    artifacts = tuple(
        str(entry["path"])
        for entry in manifest.get("artifacts") or []
        if isinstance(entry, dict) and isinstance(entry.get("path"), str)
    )
    inference_path = strategy_dir / inference_rel
    try:
        source = inference_path.read_text(encoding="utf-8")
    except OSError as exc:
        return [f"{role} inference entrypoint {inference_rel} is unreadable: {exc}"]

    if inference_path.suffix == ".py":
        return _python_inference_only_issues(
            source,
            role=role,
            inference_rel=inference_rel,
            training_rel=training_rel,
            artifact_paths=artifacts,
        )
    if inference_path.suffix in {".sh", ".bash"}:
        return _shell_inference_only_issues(
            source,
            role=role,
            inference_rel=inference_rel,
            training_rel=training_rel,
            artifact_paths=artifacts,
        )
    # Unknown extension: require that it is not the training entrypoint and that
    # a sibling Python/shell inference helper is not expected — fail closed.
    return [
        f"{role} inference entrypoint {inference_rel} must be a .py or .sh script "
        "that loads committed model artifacts"
    ]


def generated_score_artifact_issues(strategy_dir: Path, *, role: str) -> list[str]:
    issues: list[str] = []
    if not strategy_dir.is_dir():
        return [f"{role} strategy directory is missing: {strategy_dir}"]
    for path in sorted(strategy_dir.rglob("*")):
        if path.is_file() and path.name in FORBIDDEN_GENERATED_SCORE_FILES:
            issues.append(
                f"{role} strategy must not commit generated score artifact "
                f"{path.relative_to(strategy_dir)}"
            )
    return issues


def validate_ml_strategy_contract(strategy_dir: Path, *, role: str) -> StrategyContract:
    """Full fail-closed contract for one reference/baseline strategy."""
    issues = generated_score_artifact_issues(strategy_dir, role=role)
    if issues:
        raise ValueError("; ".join(issues))
    manifest = validate_committed_model_manifest(strategy_dir, role=role)
    inference_issues = inference_only_issues(strategy_dir, role=role, manifest=manifest)
    if inference_issues:
        raise ValueError("; ".join(inference_issues))
    artifacts = tuple(
        str(entry["path"])
        for entry in manifest["artifacts"]
        if isinstance(entry, dict) and isinstance(entry.get("path"), str)
    )
    return StrategyContract(
        role=role,
        strategy_dir=strategy_dir,
        manifest=manifest,
        training_entrypoint=str(manifest["training_entrypoint"]),
        inference_entrypoint=str(manifest["inference_entrypoint"]),
        artifact_paths=artifacts,
    )


def resolve_reference_strategy_dir(problem_dir: Path) -> Path:
    if (problem_dir / "reference_solution").is_dir():
        return problem_dir / "reference_solution"
    return problem_dir / "solution"


def resolve_declared_ml_strategies(
    problem_dir: Path, *, naive_rel: str | None = None
) -> list[tuple[str, Path]]:
    """Return (role, path) for reference + every calibration-declared baseline.

    Today calibration binds exactly one naive path via ``ContinuousTask.naive``
    (default ``baselines/naive``). Optional advisory baselines under
    ``baselines/<other>/`` are out of scope unless declared on TASK.
    """
    reference = resolve_reference_strategy_dir(problem_dir)
    strategies = [("reference", reference)]
    naive = (naive_rel or "baselines/naive").strip().strip("/")
    strategies.append(("naive", problem_dir / naive))
    return strategies


def validate_problem_ml_model_contracts(
    problem_dir: Path, *, naive_rel: str | None = None
) -> list[StrategyContract]:
    """Validate every declared continuous-ML strategy for a problem."""
    contracts: list[StrategyContract] = []
    for role, strategy_dir in resolve_declared_ml_strategies(
        problem_dir, naive_rel=naive_rel
    ):
        contracts.append(validate_ml_strategy_contract(strategy_dir, role=role))
    return contracts


__all__ = [
    "FORBIDDEN_GENERATED_SCORE_FILES",
    "MODEL_MANIFEST_FILENAME",
    "StrategyContract",
    "TIER_B_STATIC_ARTIFACTS",
    "file_sha256",
    "generated_score_artifact_issues",
    "inference_only_issues",
    "resolve_declared_ml_strategies",
    "resolve_reference_strategy_dir",
    "validate_committed_model_manifest",
    "validate_ml_strategy_contract",
    "validate_problem_ml_model_contracts",
]
