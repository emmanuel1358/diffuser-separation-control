from __future__ import annotations

import json
import shutil
from pathlib import Path
from types import SimpleNamespace

import pytest
from _fixture_guard import requires_examples
from alignerr_plugin.schemas import StageResult
from alignerr_plugin.utils import load_task_toml
from alignerr_plugin.validators.task.creator import TaskCreator
from alignerr_plugin.validators.task.validator import (
    TaskValidator,
    _container_user_identity,
    _service_network_policy,
    _software_service_network_issues,
    _software_workspace_seed,
)

ROOT = Path(__file__).resolve().parents[2]
SOFTWARE_STARTER = (
    ROOT
    / "alignerr_plugin"
    / "src"
    / "alignerr_plugin"
    / "starter_templates"
    / "software-engineering"
)
WAL_EXAMPLE = ROOT / "examples" / "wal-recovery-ordering"
SERVICE_EXAMPLES = (
    ROOT / "examples" / "frontier-mcp-workspace",
    ROOT / "examples" / "frontier-service-cutover",
)


@pytest.fixture
def software_problem(tmp_path: Path) -> Path:
    return TaskCreator().create_structure(
        tmp_path,
        {
            "name": "labelbox/software-contract-test",
            "template": "software-engineering",
        },
    )


def _replace_once(path: Path, old: str, new: str) -> None:
    text = path.read_text(encoding="utf-8")
    assert old in text
    path.write_text(text.replace(old, new, 1), encoding="utf-8")


def _issues(problem: Path) -> str:
    return "\n".join(TaskValidator()._software_contract(problem).issues)


def test_software_contract_no_ops_for_other_task_types(tmp_path: Path) -> None:
    problem = tmp_path / "mujoco-task"
    problem.mkdir()
    (problem / "task.toml").write_text(
        """
[task]
name = "labelbox/mujoco-task"

[environment]
required_resources = "4vcpu+16gib"

[difficulty]
task_type = "mujoco"
domain = "model_environment_construction"
reward_type = "multi_deterministic_rubrics"
""",
        encoding="utf-8",
    )

    stage = TaskValidator()._software_contract(problem)

    assert stage.passed
    assert stage.issues == []


def test_docker_user_group_uses_effective_identity() -> None:
    assert _container_user_identity("0:0") == "0"
    assert _container_user_identity("root:root") == "root"
    assert _container_user_identity("agent:agent") == "agent"


def test_verifier_network_policy_distinguishes_isolated_from_none() -> None:
    assert (
        _service_network_policy(
            SimpleNamespace(raw={"resources": {"network": "isolated"}})
        )
        == "isolated"
    )
    assert (
        _service_network_policy(
            SimpleNamespace(raw={"resources": {"network": "none"}})
        )
        == "none"
    )


def test_software_services_cannot_restore_internet_access() -> None:
    capabilities = SimpleNamespace(
        services=(
            SimpleNamespace(
                name="main",
                role="agent",
                raw={"resources": {"network": "internet"}},
            ),
            SimpleNamespace(
                name="database",
                role="sidecar",
                raw={"resources": {"network": "isolated"}},
            ),
            SimpleNamespace(
                name="verifier",
                role="verifier",
                raw={"resources": {"network": "none"}},
            ),
        )
    )

    assert _software_service_network_issues(capabilities) == [
        "software service 'main' must not enable internet access"
    ]


def test_software_contract_accepts_starter() -> None:
    stage = TaskValidator()._software_contract(SOFTWARE_STARTER)

    assert stage.passed, "\n".join(stage.issues)


def test_empty_workspace_does_not_require_starter_seed(
    software_problem: Path,
) -> None:
    with (software_problem / "task.toml").open("a", encoding="utf-8") as handle:
        handle.write(
            '\n[workspace]\nroot = "/workdir"\n'
            'agent_cwd = "/workdir"\ninit_policy = "empty"\n'
        )

    seed, issues = _software_workspace_seed(
        software_problem,
        load_task_toml(software_problem),
    )

    assert seed is None
    assert issues == []


def test_workspace_section_does_not_force_service_capsule(
    software_problem: Path,
) -> None:
    with (software_problem / "task.toml").open("a", encoding="utf-8") as handle:
        handle.write(
            '\n[workspace]\nseed = "starter"\nroot = "/workdir"\n'
            'agent_cwd = "/workdir"\ninit_policy = "copy"\n'
            "git_baseline = false\n"
        )

    task_toml = load_task_toml(software_problem)
    stage = TaskValidator()._software_contract(software_problem)

    assert not task_toml.services
    assert stage.passed, "\n".join(stage.issues)


def test_invalid_services_do_not_fall_back_to_simple_contract(
    software_problem: Path,
) -> None:
    digest = "a" * 64
    with (software_problem / "task.toml").open("a", encoding="utf-8") as handle:
        handle.write(
            '\n[result]\noutput_root = "/tmp/output/verifier"\n'
            'reward_file = "grade.json"\nreward_key = "score"\n'
            "\n[[services]]\n"
            'name = "main"\nrole = "main"\n'
            f'image = "example.invalid/main@sha256:{digest}"\n'
            'user = "agent"\nnetwork_mode = "share"\n'
            'network_share_target = "verifier"\n'
            "\n[[services]]\n"
            'name = "verifier"\nrole = "verifier"\n'
            f'image = "example.invalid/verifier@sha256:{digest}"\n'
            'user = "root"\n'
        )

    stage = TaskValidator()._software_contract(software_problem)

    assert any(
        "invalid software capability contract" in issue for issue in stage.issues
    )
    assert not any("WorkspaceArtifact root" in issue for issue in stage.issues)
    assert not any("root-only scorer/data layout" in issue for issue in stage.issues)


@requires_examples("wal-recovery-ordering")
def test_software_contract_accepts_wal_example() -> None:
    stage = TaskValidator()._software_contract(WAL_EXAMPLE)

    assert stage.passed, "\n".join(stage.issues)


@pytest.mark.parametrize(
    "example",
    [
        pytest.param(
            SERVICE_EXAMPLES[0],
            id="mcp-workspace",
            marks=requires_examples("frontier-mcp-workspace"),
        ),
        pytest.param(
            SERVICE_EXAMPLES[1],
            id="service-cutover",
            marks=requires_examples("frontier-service-cutover"),
        ),
    ],
)
def test_software_contract_accepts_capability_examples(
    example: Path,
) -> None:
    stage = TaskValidator()._software_contract(example)

    assert stage.passed, "\n".join(stage.issues)


@pytest.mark.parametrize(
    "example",
    [
        pytest.param(
            SERVICE_EXAMPLES[0],
            id="mcp-workspace",
            marks=requires_examples("frontier-mcp-workspace"),
        ),
        pytest.param(
            SERVICE_EXAMPLES[1],
            id="service-cutover",
            marks=requires_examples("frontier-service-cutover"),
        ),
    ],
)
def test_full_task_validator_runs_capability_software_stage(
    example: Path,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    passed = StageResult(passed=True, issues=[], duration_ms=0)
    monkeypatch.setattr(TaskValidator, "_prompt_quality", lambda *_args: passed)
    monkeypatch.setattr(TaskValidator, "_local_build_proof", lambda *_args: passed)
    monkeypatch.setattr(
        TaskValidator,
        "_compute_score_return",
        lambda *_args: (passed, {}),
    )

    result = TaskValidator().validate(example, tmp_path / "results", ROOT)

    assert result.status == "valid", {
        name: stage.issues for name, stage in result.stages.items() if not stage.passed
    }
    assert "software_contract" in result.stages
    assert result.stages["software_contract"].passed, "\n".join(
        result.stages["software_contract"].issues
    )


@requires_examples("frontier-service-cutover")
def test_software_capability_rejects_private_material_in_agent_image(
    tmp_path: Path,
) -> None:
    problem = tmp_path / "service-cutover"
    shutil.copytree(SERVICE_EXAMPLES[1], problem)
    dockerfile = problem / "environment" / "main" / "Dockerfile"
    dockerfile.write_text(
        dockerfile.read_text(encoding="utf-8")
        + "\nCOPY ../../../scorer/ /leaked-scorer/\n",
        encoding="utf-8",
    )

    issues = _issues(problem)

    assert "agent-facing service images must not copy scorer" in issues


@requires_examples("frontier-service-cutover")
def test_software_capability_requires_canonical_grade_output(tmp_path: Path) -> None:
    problem = tmp_path / "service-cutover"
    shutil.copytree(SERVICE_EXAMPLES[1], problem)
    _replace_once(
        problem / "task.toml",
        'path = "/tmp/output/grade.json"',
        'path = "/tmp/output/other.json"',
    )

    issues = _issues(problem)

    assert "canonical verifier result '/tmp/output/grade.json'" in issues


@pytest.mark.parametrize(
    ("old", "new", "expected"),
    [
        (
            'reward_type = "multi_deterministic_rubrics"',
            'reward_type = "continuous_scoring_function"',
            "multi_deterministic_rubrics",
        ),
        ("allow_internet = false", "allow_internet = true", "allow_internet"),
        (
            'path = "/tmp/output/repo"',
            'path = "/tmp/output/elsewhere"',
            "WorkspaceArtifact root",
        ),
        ("required = true", "required = false", "required [[outputs]]"),
    ],
)
def test_software_contract_rejects_invalid_task_policy(
    software_problem: Path,
    old: str,
    new: str,
    expected: str,
) -> None:
    _replace_once(software_problem / "task.toml", old, new)

    issues = _issues(software_problem)

    assert expected in issues


def test_software_contract_requires_workspace_artifact(
    software_problem: Path,
) -> None:
    (software_problem / "scorer" / "compute_score.py").write_text(
        """
from grading.evaluation import RubricCriterion, RubricTask, TextArtifact


def evaluate(context):
    context.run_candidate(["true"])
    return {"behavior": 1.0}


TASK = RubricTask(
    artifact=TextArtifact("result.txt"),
    criteria=(RubricCriterion("behavior"),),
    evaluate=evaluate,
)
""",
        encoding="utf-8",
    )

    issues = _issues(software_problem)

    assert "artifact=WorkspaceArtifact" in issues


def test_software_contract_requires_bounded_candidate_runner(
    software_problem: Path,
) -> None:
    (software_problem / "scorer" / "compute_score.py").write_text(
        """
from grading.evaluation import RubricCriterion, RubricTask, WorkspaceArtifact


def evaluate(context):
    return {"behavior": 1.0}


TASK = RubricTask(
    artifact=WorkspaceArtifact("repo"),
    criteria=(RubricCriterion("behavior"),),
    evaluate=evaluate,
)
""",
        encoding="utf-8",
    )

    issues = _issues(software_problem)

    assert "context.run_candidate()" in issues


def test_software_contract_rejects_candidate_module_import(
    software_problem: Path,
) -> None:
    scorer = software_problem / "scorer" / "compute_score.py"
    scorer.write_text(
        "import normalizer\n" + scorer.read_text(encoding="utf-8"),
        encoding="utf-8",
    )

    issues = _issues(software_problem)

    assert "must not import candidate workspace module 'normalizer'" in issues


def test_software_contract_rejects_direct_candidate_subprocess(
    software_problem: Path,
) -> None:
    scorer = software_problem / "scorer" / "compute_score.py"
    text = scorer.read_text(encoding="utf-8")
    text = text.replace(
        "import json\n",
        "import json\nfrom subprocess import run as launch\n",
        1,
    )
    text = text.replace("context.run_candidate_suite(", "launch(", 1)
    scorer.write_text(text, encoding="utf-8")

    issues = _issues(software_problem)

    assert "not direct process APIs" in issues


def test_software_contract_rejects_nonpositive_candidate_bound(
    software_problem: Path,
) -> None:
    _replace_once(
        software_problem / "scorer" / "compute_score.py",
        "timeout_s=60.0",
        "timeout_s=0.0",
    )

    issues = _issues(software_problem)

    assert "timeout_s must be positive" in issues


def test_software_contract_rejects_public_private_layout(
    software_problem: Path,
) -> None:
    dockerfile = software_problem / "environment" / "Dockerfile"
    _replace_once(
        dockerfile,
        "find /mcp_server/data /mcp_server/grader -type d -exec chmod 0700",
        "find /mcp_server/data /mcp_server/grader -type d -exec chmod 0755",
    )
    _replace_once(
        dockerfile,
        "find /mcp_server/data /mcp_server/grader -type f -exec chmod 0600",
        "find /mcp_server/data /mcp_server/grader -type f -exec chmod 0644",
    )

    issues = _issues(software_problem)

    assert "group/world" in issues


@pytest.mark.parametrize(
    ("hazard", "expected"),
    [
        ("symlink", "contains a symlink"),
        ("native", "compiled/native payload"),
        ("cache", "generated cache/build directory"),
        ("secret", "secret/credential file"),
    ],
)
def test_software_contract_rejects_workspace_seed_hazards(
    software_problem: Path,
    hazard: str,
    expected: str,
) -> None:
    starter = software_problem / "starter"
    if hazard == "symlink":
        (starter / "linked.py").symlink_to("normalizer.py")
    elif hazard == "native":
        (starter / "payload.bin").write_bytes(b"\x7fELF\x02\x01")
    elif hazard == "cache":
        cache = starter / "__pycache__"
        cache.mkdir()
        (cache / "normalizer.pyc").write_bytes(b"cache")
    else:
        (starter / ".env").write_text("TOKEN=not-for-redistribution\n")

    issues = _issues(software_problem)

    assert expected in issues


@pytest.mark.parametrize(
    ("relative", "expected"),
    [
        ("solution/solve.sh", "solution/solve.sh oracle"),
        ("baselines/noop.sh", "no-op or naive baseline"),
    ],
)
def test_software_contract_requires_oracle_and_baseline(
    software_problem: Path,
    relative: str,
    expected: str,
) -> None:
    (software_problem / relative).unlink()

    issues = _issues(software_problem)

    assert expected in issues


@pytest.mark.parametrize(
    ("old", "new", "expected"),
    [
        ("timeout_sec = 7200", "timeout_sec = 0", "positive timeout"),
        (
            "max_episode_sec = 7200",
            "max_episode_sec = 60",
            "max_episode_sec must cover",
        ),
    ],
)
def test_software_contract_rejects_invalid_long_horizon_timeouts(
    software_problem: Path,
    old: str,
    new: str,
    expected: str,
) -> None:
    _replace_once(software_problem / "task.toml", old, new)

    issues = _issues(software_problem)

    assert expected in issues


def test_software_contract_requires_plan_matching_task(
    software_problem: Path,
) -> None:
    plan_path = software_problem / "scorer" / "evaluation.plan.json"
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    plan["decision_ids"].append("tampered")
    plan_path.write_text(json.dumps(plan, indent=2) + "\n", encoding="utf-8")

    issues = _issues(software_problem)

    assert "evaluation.plan.json is stale" in issues


def test_software_contract_requires_private_hidden_fixtures(
    software_problem: Path,
) -> None:
    shutil.rmtree(software_problem / "scorer" / "data")

    issues = _issues(software_problem)

    assert "hidden fixtures under scorer/data" in issues
