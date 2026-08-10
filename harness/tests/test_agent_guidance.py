from __future__ import annotations

import re
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]
CURSOR_SKILLS = ROOT / ".cursor" / "skills"
CLAUDE_SKILLS = ROOT / ".claude" / "skills"
CURSOR_RULES = ROOT / ".cursor" / "rules"

EXPECTED_SKILLS = {
    "alignerr-task-authoring",
    "deterministic-grading",
    "ground-truth-oracle",
    "hidden-env-tasks",
    "ml-tasks",
    "mujoco-tasks",
    "numerical-solver-tasks",
    "prometheus-delivery",
    "reward-hacking-security",
    "rubric-design",
    "service-capsule-tasks",
    "software-engineering-tasks",
    "task-migration",
}

ESSENTIAL_RULES = {
    "continuous-scoring-tasks.mdc",
    "deterministic-rubrics.mdc",
    "github-secrets.mdc",
    "grader-contract.mdc",
    "ground-truth-oracle.mdc",
    "hidden-env.mdc",
    "local-harness.mdc",
    "mujoco-tasks.mdc",
    "numerical-solvers.mdc",
    "prometheus-delivery.mdc",
    "prompt-fairness.mdc",
    "reward-hacking.mdc",
    "software-engineering-tasks.mdc",
    "task-authoring.mdc",
    "task-delivery-gates.mdc",
    "task-dependencies.mdc",
    "template-owned-shared-code.mdc",
}


def _frontmatter(path: Path) -> tuple[dict[str, object], str]:
    text = path.read_text()
    assert text.startswith("---\n"), f"{path} has no YAML frontmatter"
    _opening, raw, body = text.split("---", 2)
    parsed = yaml.safe_load(raw)
    assert isinstance(parsed, dict), f"{path} frontmatter is not an object"
    return parsed, body


def _skill_paths(root: Path) -> dict[str, Path]:
    return {
        path.parent.name: path
        for path in sorted(root.glob("*/SKILL.md"))
        if path.is_file()
    }


def test_cursor_and_claude_skill_suites_are_complete_and_identical() -> None:
    cursor = _skill_paths(CURSOR_SKILLS)
    claude = _skill_paths(CLAUDE_SKILLS)

    assert set(cursor) == EXPECTED_SKILLS
    assert set(claude) == EXPECTED_SKILLS
    for name in sorted(EXPECTED_SKILLS):
        assert cursor[name].read_bytes() == claude[name].read_bytes(), name


def test_skill_metadata_is_discoverable_and_bounded() -> None:
    for directory, path in _skill_paths(CURSOR_SKILLS).items():
        metadata, body = _frontmatter(path)
        assert metadata["name"] == directory
        assert re.fullmatch(r"[a-z0-9-]{1,64}", directory)
        description = metadata.get("description")
        assert isinstance(description, str) and description.strip()
        assert len(description) <= 1024
        assert "Use " in description
        assert body.strip()
        assert len(path.read_text().splitlines()) < 500


def test_cursor_rule_suite_has_valid_scopes() -> None:
    paths = {path.name: path for path in sorted(CURSOR_RULES.glob("*.mdc"))}

    assert set(paths) == ESSENTIAL_RULES
    for name, path in paths.items():
        metadata, body = _frontmatter(path)
        description = metadata.get("description")
        assert isinstance(description, str) and description.strip(), name
        always_apply = metadata.get("alwaysApply")
        assert isinstance(always_apply, bool), name
        if not always_apply:
            globs = metadata.get("globs")
            assert isinstance(globs, str) and globs.strip(), name
        assert body.strip(), name
        assert len(path.read_text().splitlines()) < 500, name


def test_delivery_and_software_rules_encode_acceptance_bar() -> None:
    delivery = (CURSOR_RULES / "task-delivery-gates.mdc").read_text()
    software = (CURSOR_RULES / "software-engineering-tasks.mdc").read_text()

    for text in (delivery, software):
        assert "trusted CI" in text
        assert "Boreal aggregate" in text
        assert "<= 0.4" in text
