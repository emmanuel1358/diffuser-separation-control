from __future__ import annotations

import json
import os
import stat
import subprocess
from pathlib import Path

import pytest
from grading import AgentFault, GraderFault
from grading.evaluation import (
    JsonArtifact,
    NumericField,
    RegularFileArtifact,
    RubricCriterion,
    RubricEvaluation,
    RubricTask,
    TrustedJson,
    WorkspaceArtifact,
)
from grading.evaluation.plan import validate_serialized_plan
from grading.numeric import NumericContractError, safe_mean, safe_ratio


def _task(*, evaluate=None, required: bool = False, fixtures=None) -> RubricTask:
    return RubricTask(
        artifact=JsonArtifact(
            "design.json",
            required_keys=("value",),
            numeric_fields=(NumericField("value"),),
        ),
        criteria=(
            RubricCriterion(
                id="quality",
                weight=1.0,
                description="Candidate quality",
                required=required,
            ),
        ),
        fixtures=fixtures or {},
        evaluate=evaluate or (lambda context: {"quality": context.candidate["value"]}),
    )


def _workspace(tmp_path: Path, payload: bytes = b'{"value": 0.75}') -> Path:
    workspace = tmp_path / "output"
    workspace.mkdir()
    workspace.joinpath("design.json").write_bytes(payload)
    return workspace


def test_rubric_task_grades_typed_candidate(tmp_path: Path) -> None:
    grade = _task().grade(workspace=_workspace(tmp_path), private=tmp_path)

    assert grade.score() == pytest.approx(0.75)
    assert grade.subscores == {"quality": 0.75}
    assert grade.metadata["return_shape"] == "declarative_rubric"
    assert grade.metadata["evaluation"]["protocol"] == "declarative-rubric.v1"


@pytest.mark.parametrize(
    "payload",
    [
        b"\xff\xfe{}",
        b"{",
        b"[]",
        ('{"value": %s}' % (10**400)).encode(),
        ('{"value": %s}' % ("[" * 2000 + "]" * 2000)).encode(),
    ],
)
def test_bad_json_content_is_agent_fault(tmp_path: Path, payload: bytes) -> None:
    with pytest.raises(AgentFault):
        _task().grade(workspace=_workspace(tmp_path, payload), private=tmp_path)


def test_symlink_artifact_is_agent_fault(tmp_path: Path) -> None:
    workspace = tmp_path / "output"
    workspace.mkdir()
    truth = tmp_path / "truth.json"
    truth.write_text('{"value": 1.0}')
    os.symlink(truth, workspace / "design.json")

    with pytest.raises(AgentFault):
        _task().grade(workspace=workspace, private=tmp_path)


def test_symlink_workspace_parent_is_agent_fault(tmp_path: Path) -> None:
    private = tmp_path / "private"
    private.mkdir()
    (private / "design.json").write_text('{"value": 1.0}')
    workspace = tmp_path / "output"
    os.symlink(private, workspace)

    with pytest.raises(AgentFault):
        _task().grade(workspace=workspace, private=tmp_path)


def test_regular_file_artifact_returns_immutable_snapshot(tmp_path: Path) -> None:
    workspace = tmp_path / "output"
    workspace.mkdir()
    original = workspace / "policy.py"
    original.write_text("VALUE = 1\n")

    submitted = RegularFileArtifact("policy.py").load(workspace)
    original.write_text("VALUE = 2\n")

    assert submitted.original_path == original
    assert submitted.path != original
    assert submitted.path.read_text() == "VALUE = 1\n"


def test_unclassified_evaluator_exception_is_kept_zero(tmp_path: Path) -> None:
    def evaluate(_context):
        raise ZeroDivisionError("candidate-dependent denominator")

    grade = _task(evaluate=evaluate).grade(
        workspace=_workspace(tmp_path),
        private=tmp_path,
    )

    assert grade.score() == 0.0
    assert grade.env_internal_failure is False
    assert grade.metadata["critical_operator_alert"] is True
    assert "traceback" not in grade.metadata
    assert (
        "ZeroDivisionError"
        in grade.criterion_logs["unclassified_grader_crash"]["error_message"]
    )


def test_declared_zero_denominator_policy_becomes_agent_fault(tmp_path: Path) -> None:
    def evaluate(context):
        return {
            "quality": context.ratio(
                context.candidate["value"],
                0,
                label="candidate efficiency",
                zero="agent_fault",
            )
        }

    with pytest.raises(AgentFault, match="denominator is zero"):
        _task(evaluate=evaluate).grade(
            workspace=_workspace(tmp_path),
            private=tmp_path,
        )


def test_missing_trusted_fixture_is_grader_fault(tmp_path: Path) -> None:
    task = _task(fixtures={"truth": TrustedJson("truth.json")})

    with pytest.raises(GraderFault):
        task.grade(workspace=_workspace(tmp_path), private=tmp_path / "private")


def test_required_criterion_zeroes_headline(tmp_path: Path) -> None:
    task = _task(
        required=True,
        evaluate=lambda _context: RubricEvaluation(
            subscores={"quality": 0.4},
            metadata={"domain": "fixture"},
        ),
    )

    grade = task.grade(workspace=_workspace(tmp_path), private=tmp_path)

    assert grade.subscores == {"quality": 0.4}
    assert grade.score() == 0.0
    assert grade.metadata["domain"] == "fixture"


def test_rubric_receipt_redacts_replay_and_writes_private_trace(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    trace = tmp_path / "evaluation-details.json"
    monkeypatch.setenv("LBX_EVALUATION_TRACE_PATH", str(trace))

    grade = _task().grade(workspace=_workspace(tmp_path), private=tmp_path)
    receipt = grade.metadata["evaluation"]
    private = json.loads(trace.read_text())

    assert "artifact_digest" not in receipt
    assert "nonce" not in receipt
    assert private["schema_version"] == "rubric-evaluation-trace.v1"
    assert len(private["replay"]["artifact_digest"]) == 64
    assert private["targets"]["quality"]["score"] == pytest.approx(0.75)


def test_evaluator_must_return_exact_criterion_set(tmp_path: Path) -> None:
    task = _task(evaluate=lambda _context: {"wrong": 1.0})

    with pytest.raises(GraderFault, match="criterion mismatch"):
        task.grade(workspace=_workspace(tmp_path), private=tmp_path)


def test_shared_numeric_helpers_require_explicit_empty_zero_policies() -> None:
    assert safe_ratio(1, 0, zero="zero") == 0.0
    assert safe_mean([], empty="zero") == 0.0
    with pytest.raises(NumericContractError):
        safe_ratio(1, 0)
    with pytest.raises(NumericContractError):
        safe_mean([])


def test_rubric_spec_is_stable_and_json_serializable() -> None:
    first = _task()
    second = _task()

    assert first.spec_sha256 == second.spec_sha256
    assert json.loads(json.dumps(first.spec_dict()))["protocol"] == (
        "declarative-rubric.v1"
    )

    plan = first.evaluation_plan
    payload = {**plan.to_dict(), "plan_sha256": plan.sha256}
    assert payload["schema_version"] == "evaluation-plan.v2"
    assert validate_serialized_plan(payload) == plan.sha256


def test_workspace_artifact_cleans_caches_and_validates_source(
    tmp_path: Path,
) -> None:
    workspace = tmp_path / "output"
    repo = workspace / "repo"
    (repo / "src").mkdir(parents=True)
    source = b"fn main() {}\n"
    (repo / "src" / "main.rs").write_bytes(source)
    (repo / "target").mkdir()
    (repo / "target" / "cached-binary").write_bytes(b"\x7fELF")

    artifact = WorkspaceArtifact(
        "repo",
        clean_paths=("target",),
        forbidden_text_patterns=("std::process::Command",),
        text_suffixes=(".rs",),
    )
    loaded = artifact.load(workspace)

    assert loaded.path != repo
    assert loaded.snapshot_path == loaded.path
    assert loaded.original_path == repo
    assert (loaded.path / "src" / "main.rs").read_bytes() == source
    assert loaded.file_count == 1
    assert loaded.total_bytes == len(source)
    assert not (repo / "target").exists()
    assert not (loaded.path / "target").exists()


@pytest.mark.parametrize(
    ("filename", "payload", "match"),
    [
        ("payload.bin", b"\x7fELF\x00\x00", "native payload"),
        (
            "main.rs",
            b'fn main() { std::process::Command::new("legacy"); }',
            "forbidden pattern",
        ),
        (
            "main.rs",
            b'fn main() { std :: process :: Command :: new("legacy"); }',
            "forbidden pattern",
        ),
    ],
)
def test_workspace_artifact_rejects_delegation_payloads(
    tmp_path: Path,
    filename: str,
    payload: bytes,
    match: str,
) -> None:
    workspace = tmp_path / "output"
    repo = workspace / "repo"
    repo.mkdir(parents=True)
    (repo / filename).write_bytes(payload)
    artifact = WorkspaceArtifact(
        "repo",
        forbidden_text_patterns=("std::process::Command",),
        text_suffixes=(".rs",),
    )

    with pytest.raises(AgentFault, match=match):
        artifact.load(workspace)


@pytest.mark.parametrize(
    ("filename", "artifact", "match"),
    [
        (
            "build.rs",
            WorkspaceArtifact("repo", forbidden_names=("build.rs",)),
            "forbidden file",
        ),
        (
            "payload.o",
            WorkspaceArtifact("repo", forbidden_suffixes=(".o",)),
            "forbidden suffix",
        ),
    ],
)
def test_workspace_artifact_rejects_forbidden_names_and_suffixes(
    tmp_path: Path,
    filename: str,
    artifact: WorkspaceArtifact,
    match: str,
) -> None:
    workspace = tmp_path / "output"
    repo = workspace / "repo"
    repo.mkdir(parents=True)
    (repo / filename).write_text("candidate\n")

    with pytest.raises(AgentFault, match=match):
        artifact.load(workspace)


@pytest.mark.parametrize("target_kind", ["file", "directory"])
def test_workspace_artifact_rejects_nested_symlinks(
    tmp_path: Path,
    target_kind: str,
) -> None:
    workspace = tmp_path / "output"
    repo = workspace / "repo"
    repo.mkdir(parents=True)
    (repo / "main.rs").write_text("fn main() {}\n")
    target = tmp_path / "private"
    if target_kind == "directory":
        target.mkdir()
    else:
        target.write_text("private\n")
    os.symlink(target, repo / "linked")

    with pytest.raises(AgentFault, match="non-regular"):
        WorkspaceArtifact("repo").load(workspace)


def test_workspace_artifact_rejects_special_files(tmp_path: Path) -> None:
    if not hasattr(os, "mkfifo"):
        pytest.skip("FIFOs are unavailable on this platform")
    workspace = tmp_path / "output"
    repo = workspace / "repo"
    repo.mkdir(parents=True)
    (repo / "main.rs").write_text("fn main() {}\n")
    os.mkfifo(repo / "candidate.pipe")

    with pytest.raises(AgentFault, match="non-regular"):
        WorkspaceArtifact("repo").load(workspace)


def test_workspace_artifact_rejects_sibling_payload(tmp_path: Path) -> None:
    workspace = tmp_path / "output"
    repo = workspace / "repo"
    repo.mkdir(parents=True)
    (repo / "main.rs").write_text("fn main() {}\n")
    (workspace / "stashed-oracle").write_bytes(b"\x7fELF")

    with pytest.raises(AgentFault, match="undeclared entries"):
        WorkspaceArtifact("repo").load(workspace)


def test_workspace_cache_cleanup_precedes_replay_commit(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    workspace = tmp_path / "output"
    repo = workspace / "repo"
    repo.mkdir(parents=True)
    (repo / "main.rs").write_text("fn main() {}\n")
    private = tmp_path / "private"
    private.mkdir()
    cache = repo / "target"
    os.symlink(private, cache)
    digest_calls: list[Path] = []

    def committed_digest(path: Path) -> str:
        digest_calls.append(path)
        assert not cache.exists()
        assert not cache.is_symlink()
        assert path != workspace
        assert (path / "repo" / "main.rs").read_text() == "fn main() {}\n"
        return "0" * 64

    monkeypatch.setattr(
        "grading.evaluation.rubric.workspace_artifact_digest",
        committed_digest,
    )
    task = RubricTask(
        artifact=WorkspaceArtifact("repo", clean_paths=("target",)),
        criteria=(RubricCriterion("quality"),),
        evaluate=lambda _context: {"quality": 1.0},
    )

    grade = task.grade(workspace=workspace, private=private)

    assert grade.score() == 1.0
    assert len(digest_calls) == 1


def test_workspace_rubric_routes_candidate_through_uid_dropped_boundary(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    workspace = tmp_path / "output"
    repo = workspace / "repo"
    repo.mkdir(parents=True)
    (repo / "main.rs").write_text("fn main() {}\n")
    seen: dict[str, object] = {}

    def fake_run(cmd, **kwargs):
        seen["cmd"] = cmd
        cwd_fd = kwargs.pop("cwd_fd")
        seen["cwd_identity"] = (
            os.fstat(cwd_fd).st_dev,
            os.fstat(cwd_fd).st_ino,
        )
        seen.update(kwargs)
        return subprocess.CompletedProcess(cmd, 0, stdout=b"ok\n", stderr=b"")

    monkeypatch.setattr("grading.helpers.run_submitted_executable", fake_run)

    def evaluate(context):
        result = context.run_candidate(["candidate"], stdin_bytes=b"request\n")
        return {"quality": result.stdout == b"ok\n"}

    task = RubricTask(
        artifact=WorkspaceArtifact("repo"),
        criteria=(RubricCriterion("quality"),),
        evaluate=evaluate,
    )
    grade = task.grade(workspace=workspace, private=tmp_path)

    assert grade.score() == 1.0
    assert seen["cwd_identity"] != (repo.stat().st_dev, repo.stat().st_ino)
    assert seen["stdin_bytes"] == b"request\n"
    assert seen["timeout_s"] == 120.0


def test_workspace_candidate_stays_on_snapshot_after_original_replacement(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    workspace = tmp_path / "output"
    repo = workspace / "repo"
    repo.mkdir(parents=True)
    (repo / "marker.txt").write_text("committed\n")
    seen_inodes: list[tuple[int, int]] = []

    def fake_run(cmd, **kwargs):
        cwd_fd = kwargs["cwd_fd"]
        info = os.fstat(cwd_fd)
        seen_inodes.append((info.st_dev, info.st_ino))
        marker_fd = os.open("marker.txt", os.O_RDONLY, dir_fd=cwd_fd)
        try:
            output = os.read(marker_fd, 1024)
        finally:
            os.close(marker_fd)
        return subprocess.CompletedProcess(cmd, 0, stdout=output, stderr=b"")

    monkeypatch.setattr("grading.helpers.run_submitted_executable", fake_run)

    def evaluate(context):
        first = context.run_candidate(["candidate"])
        original = context.candidate.original_path
        original.rename(workspace / "replaced-original")
        original.mkdir()
        (original / "marker.txt").write_text("attacker replacement\n")
        second = context.run_candidate(["candidate"])
        return {
            "quality": first.stdout == second.stdout == b"committed\n",
        }

    task = RubricTask(
        artifact=WorkspaceArtifact("repo"),
        criteria=(RubricCriterion("quality"),),
        evaluate=evaluate,
    )

    assert task.grade(workspace=workspace, private=tmp_path).score() == 1.0
    assert len(seen_inodes) == 2


def test_workspace_candidate_cwd_rejects_symlink_and_escape(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    workspace = tmp_path / "output"
    repo = workspace / "repo"
    repo.mkdir(parents=True)
    (repo / "main.rs").write_text("fn main() {}\n")
    outside = tmp_path / "outside"
    outside.mkdir()
    calls = 0

    def fake_run(cmd, **kwargs):
        nonlocal calls
        calls += 1
        return subprocess.CompletedProcess(cmd, 0, stdout=b"", stderr=b"")

    monkeypatch.setattr("grading.helpers.run_submitted_executable", fake_run)

    def evaluate(context):
        master_mode = stat.S_IMODE(context.candidate.path.stat().st_mode)
        os.chmod(context.candidate.path, 0o700)
        os.symlink(outside, context.candidate.path / "escape")
        try:
            with pytest.raises(GraderFault, match="could not be cloned safely"):
                context.run_candidate(["candidate"], cwd="escape")
        finally:
            (context.candidate.path / "escape").unlink()
            os.chmod(context.candidate.path, master_mode)
        with pytest.raises(GraderFault, match="stay within"):
            context.run_candidate(["candidate"], cwd="../outside")
        return {"quality": 1.0}

    task = RubricTask(
        artifact=WorkspaceArtifact("repo"),
        criteria=(RubricCriterion("quality"),),
        evaluate=evaluate,
    )

    assert task.grade(workspace=workspace, private=tmp_path).score() == 1.0
    assert calls == 0


@pytest.mark.skipif(
    os.geteuid() != 0,
    reason="actual uid drop only occurs when the grader runs as root",
)
def test_workspace_rubric_run_candidate_executes_as_uid_1000(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("RUBRIC_AGENT_UID", "1000")
    monkeypatch.setenv("RUBRIC_AGENT_GID", "1000")
    workspace = tmp_path / "output"
    repo = workspace / "repo"
    repo.mkdir(parents=True)
    (repo / "main.rs").write_text("fn main() {}\n")

    def evaluate(context):
        result = context.run_candidate(["/usr/bin/id", "-u"])
        return {"quality": result.stdout.strip() == b"1000"}

    task = RubricTask(
        artifact=WorkspaceArtifact("repo"),
        criteria=(RubricCriterion("quality"),),
        evaluate=evaluate,
    )

    assert task.grade(workspace=workspace, private=tmp_path).score() == 1.0


def _write_rubric_problem(problem_dir: Path) -> None:
    scorer = problem_dir / "scorer"
    scorer.mkdir(parents=True)
    scorer.joinpath("compute_score.py").write_text(
        "from grading.evaluation import JsonArtifact, NumericField, "
        "RubricCriterion, RubricTask\n"
        "\n"
        "def evaluate(context):\n"
        "    return {'quality': float(context.candidate['value'])}\n"
        "\n"
        "TASK = RubricTask(\n"
        "    artifact=JsonArtifact(\n"
        "        'design.json',\n"
        "        required_keys=('value',),\n"
        "        numeric_fields=(NumericField('value'),),\n"
        "    ),\n"
        "    criteria=(RubricCriterion('quality', weight=1.0),),\n"
        "    evaluate=evaluate,\n"
        ")\n",
        encoding="utf-8",
    )


def test_refresh_and_check_evaluation_plan(tmp_path: Path) -> None:
    from grading.evaluation.plan import check_evaluation_plan, refresh_evaluation_plan

    problem = tmp_path / "rubric-task"
    _write_rubric_problem(problem)

    first = refresh_evaluation_plan(problem)
    assert first.status == "written"
    assert first.path.is_file()

    second = refresh_evaluation_plan(problem)
    assert second.status == "unchanged"
    assert check_evaluation_plan(problem).status == "unchanged"

    first.path.write_text("{}\n", encoding="utf-8")
    assert check_evaluation_plan(problem).status == "stale"
    assert refresh_evaluation_plan(problem).status == "written"

    first.path.unlink()
    blocked = check_evaluation_plan(problem)
    assert blocked.status == "missing"
    assert "evaluation.plan.json" in blocked.message
    assert not first.path.exists()
