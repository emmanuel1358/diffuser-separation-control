from __future__ import annotations

import ast
import json
import os
from pathlib import Path
from typing import Any


def _json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise TypeError(f"{path} must contain an object")
    return value


def _policy_version(path: Path) -> str | None:
    tree = ast.parse(path.read_text(), filename=str(path))
    for node in tree.body:
        if not isinstance(node, (ast.Assign, ast.AnnAssign)):
            continue
        targets = node.targets if isinstance(node, ast.Assign) else [node.target]
        if not any(
            isinstance(target, ast.Name) and target.id == "POLICY_VERSION"
            for target in targets
        ):
            continue
        value = node.value
        if isinstance(value, ast.Constant) and isinstance(value.value, str):
            return value.value
    return None


def evaluate(root: Path) -> dict[str, Any]:
    workspace = _json(root / "evidence/workspace.json")
    source_ok = _policy_version(root / "repo/processor.py") == "deterministic-v1"
    evidence_ok = (
        workspace.get("schema_version") == "synthetic-claims-workspace.v1"
        and workspace.get("title") == "Synthetic Claims Inbox"
        and isinstance(workspace.get("cases"), list)
        and len(workspace["cases"]) == 2
    )
    score = float(source_ok and evidence_ok)
    return {
        "score": score,
        "subscores": {
            "source_contract": float(source_ok),
            "sealed_workspace_evidence": float(evidence_ok),
        },
        "schema_version": "frontier-mcp-verifier.v1",
    }


def main() -> None:
    root = Path(os.environ.get("ARTIFACT_ROOT", "/lbx/service-artifacts"))
    output_root = Path(os.environ.get("OUTPUT_ROOT", "/tmp/output"))
    output_root.mkdir(parents=True, exist_ok=True)
    destination = output_root / "grade.json"
    temporary = output_root / ".grade.json.tmp"
    result = evaluate(root)
    temporary.write_text(
        json.dumps(result, sort_keys=True, separators=(",", ":")) + "\n"
    )
    os.replace(temporary, destination)


if __name__ == "__main__":
    main()
