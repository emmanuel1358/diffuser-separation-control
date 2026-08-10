from __future__ import annotations

import argparse
import re
from pathlib import Path


FORBIDDEN_LINE_TEXT = {
    "pull_request_target": "privileged fork-PR execution belongs in mothership",
    "/dispatches": "cross-repository dispatch belongs in mothership",
    "repository_dispatch": "repository dispatch belongs in mothership",
}

# Match the complete Actions expression so whitespace, bracket notation, helper
# calls, and line breaks cannot bypass the repository-secret boundary.
SECRET_EXPRESSION = re.compile(
    r"\$\{\{(?:(?!\}\}).)*\bsecrets\b(?:(?!\}\}).)*\}\}",
    re.DOTALL,
)
SECRET_MAPPING = re.compile(r"(?m)^[ \t]*secrets[ \t]*:")


def _line_number(text: str, offset: int) -> int:
    return text.count("\n", 0, offset) + 1


def audit_workflows(workflow_dir: Path) -> list[str]:
    violations: list[str] = []
    workflow_paths = sorted(workflow_dir.glob("*.yml")) + sorted(
        workflow_dir.glob("*.yaml")
    )
    for workflow_path in workflow_paths:
        workflow_text = workflow_path.read_text()
        for line_number, line in enumerate(workflow_text.splitlines(), start=1):
            for forbidden, reason in FORBIDDEN_LINE_TEXT.items():
                if forbidden in line:
                    violations.append(
                        f"{workflow_path}:{line_number}: {reason}: {forbidden}"
                    )
        for pattern in (SECRET_EXPRESSION, SECRET_MAPPING):
            for match in pattern.finditer(workflow_text):
                violations.append(
                    f"{workflow_path}:{_line_number(workflow_text, match.start())}: "
                    "template workflows must not consume repository secrets"
                )
    return violations


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Enforce the ISO template workflow trust boundary."
    )
    parser.add_argument(
        "workflow_dir",
        nargs="?",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "workflows",
    )
    args = parser.parse_args()

    violations = audit_workflows(args.workflow_dir)
    if violations:
        print("Template workflow trust-boundary violations:")
        print("\n".join(f"- {violation}" for violation in violations))
        return 1
    print("Template workflows are credential-free and do not dispatch CI.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
