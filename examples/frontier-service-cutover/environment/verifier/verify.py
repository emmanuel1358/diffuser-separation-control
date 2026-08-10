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


def _target_field(path: Path) -> str | None:
    tree = ast.parse(path.read_text(), filename=str(path))
    for node in tree.body:
        if not isinstance(node, (ast.Assign, ast.AnnAssign)):
            continue
        targets = node.targets if isinstance(node, ast.Assign) else [node.target]
        if not any(
            isinstance(target, ast.Name) and target.id == "TARGET_FIELD"
            for target in targets
        ):
            continue
        value = node.value
        if isinstance(value, ast.Constant) and isinstance(value.value, str):
            return value.value
    return None


def evaluate(root: Path) -> dict[str, Any]:
    customer = _json(root / "evidence/customer-results.json")
    state = _json(root / "evidence/state.json")
    marker = _json(root / "evidence/init.json")
    patch = root / "evidence/workspace.patch"
    source_ok = _target_field(root / "repo/app.py") == "current"
    evidence_ok = (
        customer.get("schema_version") == "cutover-customer-results.v1"
        and customer.get("finalized") is True
        and customer.get("total") == 6
        and customer.get("passed") == customer.get("total")
        and state.get("schema_version") == "cutover-state.v1"
        and isinstance(state.get("records"), list)
        and len(state["records"]) == 6
        and marker
        == {
            "schema_version": "cutover-init.v1",
            "seed_count": 2,
        }
        and patch.is_file()
    )
    score = float(source_ok and evidence_ok)
    return {
        "score": score,
        "subscores": {
            "source_contract": float(source_ok),
            "sealed_service_evidence": float(evidence_ok),
        },
        "schema_version": "frontier-service-verifier.v1",
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
