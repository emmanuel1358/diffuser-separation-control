from __future__ import annotations

import hashlib
import json
import os
import stat
import subprocess
from dataclasses import FrozenInstanceError
from pathlib import Path
from typing import Any

import pytest
from grading import AgentFault
from grading.evaluation import (
    CandidateAttempt,
    CandidateCommandSpec,
    RubricCriterion,
    RubricTask,
    WorkspaceArtifact,
    parse_candidate_json,
    parse_candidate_jsonl,
    run_candidate_suite,
)
from grading.evaluation import artifacts as artifacts_module
from grading.evaluation import candidate_suite as candidate_suite_module
from grading.evaluation.context import workspace_artifact_digest


class _FakeContext:
    def __init__(self, outcomes: list[Any]) -> None:
        self.outcomes = iter(outcomes)
        self.calls: list[dict[str, Any]] = []
        self.operation_labels: list[str] = []

    def run_candidate(self, cmd: list[str], **kwargs: Any) -> Any:
        self.calls.append({"cmd": cmd, **kwargs})
        outcome = next(self.outcomes)
        if isinstance(outcome, BaseException):
            raise outcome
        return outcome

    def candidate_operation(
        self,
        label: str,
        operation,
        *args: Any,
        **kwargs: Any,
    ) -> Any:
        self.operation_labels.append(label)
        try:
            return operation(*args, **kwargs)
        except Exception as exc:
            raise AgentFault(f"{label} failed: {type(exc).__name__}: {exc}") from exc


def _completed(stdout: bytes = b"ok\n", returncode: int = 0):
    return subprocess.CompletedProcess(
        ["candidate"],
        returncode,
        stdout=stdout,
        stderr=b"",
    )


def test_candidate_command_spec_is_immutable_bounded_and_stable() -> None:
    spec = CandidateCommandSpec(
        argv=["./candidate", "--mode", "jsonl"],
        cwd="src/.",
        env={"MODE": "test", "LANG": "C"},
        stdin_bytes=b"request\n",
        repeats=2,
        allowed_return_codes={0, 3},
        deterministic_stdout=True,
        max_attempt_elapsed_s=1.0,
        max_total_elapsed_s=3.0,
    )

    assert spec.argv == ("./candidate", "--mode", "jsonl")
    assert spec.cwd == "src"
    assert spec.env == (("LANG", "C"), ("MODE", "test"))
    assert spec.allowed_return_codes == (0, 3)
    with pytest.raises(FrozenInstanceError):
        spec.cwd = "elsewhere"  # type: ignore[misc]

    first = json.dumps(spec.spec_dict(), sort_keys=True)
    second = json.dumps(spec.spec_dict(), sort_keys=True)
    assert first == second
    assert spec.spec_dict()["stdin_bytes"] == {
        "size": 8,
        "sha256": hashlib.sha256(b"request\n").hexdigest(),
    }


@pytest.mark.parametrize(
    "kwargs",
    [
        {"argv": ()},
        {"argv": ("candidate",), "cwd": "../private"},
        {"argv": ("candidate",), "env": (("BAD=KEY", "value"),)},
        {"argv": ("candidate",), "timeout_s": float("nan")},
        {"argv": ("candidate",), "max_output_bytes": 0},
        {"argv": ("candidate",), "repeats": 33},
        {"argv": ("candidate",), "allowed_return_codes": ()},
        {"argv": ("candidate",), "deterministic_stdout": "yes"},
    ],
)
def test_candidate_command_spec_rejects_unbounded_or_unsafe_values(
    kwargs: dict[str, Any],
) -> None:
    with pytest.raises((TypeError, ValueError)):
        CandidateCommandSpec(**kwargs)


def test_candidate_command_spec_bounds_stdin(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(candidate_suite_module, "_MAX_STDIN_BYTES", 4)
    with pytest.raises(ValueError, match="stdin exceeds"):
        CandidateCommandSpec(argv=("candidate",), stdin_bytes=b"12345")


def test_candidate_suite_uses_secure_committed_workspace_boundary(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    workspace = tmp_path / "output"
    source = workspace / "repo" / "src"
    source.mkdir(parents=True)
    (source / "main.py").write_text("print('candidate')\n")
    original_identity = (source.stat().st_dev, source.stat().st_ino)
    seen: list[dict[str, Any]] = []

    def fake_run(cmd, **kwargs):
        cwd_fd = kwargs.pop("cwd_fd")
        info = os.fstat(cwd_fd)
        seen.append(
            {
                "cmd": cmd,
                "cwd_identity": (info.st_dev, info.st_ino),
                **kwargs,
            }
        )
        return _completed(b'{"ok":true}\n')

    monkeypatch.setattr("grading.helpers.run_submitted_executable", fake_run)

    spec = CandidateCommandSpec(
        argv=("./candidate",),
        cwd="src",
        env=(("MODE", "test"),),
        stdin_bytes=b"input\n",
        repeats=2,
        deterministic_stdout=True,
    )

    def evaluate(context):
        result = context.run_candidate_suite(spec)
        return {
            "quality": (
                len(result.attempts) == 2
                and result.metadata.deterministic_stdout
                and result.attempts[0].stdout == b'{"ok":true}\n'
            )
        }

    task = RubricTask(
        artifact=WorkspaceArtifact("repo"),
        criteria=(RubricCriterion("quality"),),
        evaluate=evaluate,
    )

    assert task.grade(workspace=workspace, private=tmp_path).score() == 1.0
    assert len(seen) == 2
    assert seen[0]["cwd_identity"] != original_identity
    assert seen[0]["env"] == {"MODE": "test"}
    assert seen[0]["stdin_bytes"] == b"input\n"


def test_candidate_suite_attempts_clone_clean_immutable_master(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    workspace = tmp_path / "output"
    repo = workspace / "repo"
    repo.mkdir(parents=True)
    (repo / "marker.txt").write_text("baseline\n")
    clone_modes: list[int] = []
    removed_clones: list[Path] = []
    attempt = 0
    remove_workspace_tree = artifacts_module._remove_workspace_tree

    def tracked_remove(path: Path, *, ignore_errors: bool = False) -> None:
        remove_workspace_tree(path, ignore_errors=ignore_errors)
        if path.name.startswith("lbx-workspace-run-"):
            removed_clones.append(path)

    monkeypatch.setattr(
        artifacts_module,
        "_remove_workspace_tree",
        tracked_remove,
    )

    def fake_run(cmd, **kwargs):
        nonlocal attempt
        cwd_fd = kwargs["cwd_fd"]
        clone_modes.append(stat.S_IMODE(os.fstat(cwd_fd).st_mode))
        marker_fd = os.open("marker.txt", os.O_RDONLY, dir_fd=cwd_fd)
        try:
            baseline = os.read(marker_fd, 1024)
        finally:
            os.close(marker_fd)
        if attempt == 0:
            os.unlink("marker.txt", dir_fd=cwd_fd)
            replacement_fd = os.open(
                "marker.txt",
                os.O_WRONLY | os.O_CREAT | os.O_EXCL,
                0o600,
                dir_fd=cwd_fd,
            )
            try:
                os.write(replacement_fd, b"mutated\n")
            finally:
                os.close(replacement_fd)
            output = b"first replaced clone\n"
        else:
            output = baseline
        attempt += 1
        return _completed(output)

    monkeypatch.setattr("grading.helpers.run_submitted_executable", fake_run)
    spec = CandidateCommandSpec(argv=("candidate",), repeats=2)
    master_evidence: dict[str, Any] = {}

    def evaluate(context):
        master = context.candidate.path
        marker = master / "marker.txt"
        before_digest = workspace_artifact_digest(master)
        before_modes = (
            stat.S_IMODE(master.stat().st_mode),
            stat.S_IMODE(marker.stat().st_mode),
        )
        result = context.run_candidate_suite(spec)
        after_modes = (
            stat.S_IMODE(master.stat().st_mode),
            stat.S_IMODE(marker.stat().st_mode),
        )
        master_evidence.update(
            {
                "owner": master.stat().st_uid,
                "before_modes": before_modes,
                "after_modes": after_modes,
                "digest_unchanged": workspace_artifact_digest(master) == before_digest,
                "contents": marker.read_bytes(),
            }
        )
        return {
            "quality": (
                result.attempts[0].stdout == b"first replaced clone\n"
                and result.attempts[1].stdout == b"baseline\n"
            )
        }

    task = RubricTask(
        artifact=WorkspaceArtifact("repo"),
        criteria=(RubricCriterion("quality"),),
        evaluate=evaluate,
    )

    assert task.grade(workspace=workspace, private=tmp_path).score() == 1.0
    assert all(mode & stat.S_IWUSR for mode in clone_modes)
    assert len(removed_clones) == 2
    assert not any(path.exists() for path in removed_clones)
    assert master_evidence == {
        "owner": os.geteuid(),
        "before_modes": (0o500, 0o400),
        "after_modes": (0o500, 0o400),
        "digest_unchanged": True,
        "contents": b"baseline\n",
    }


def test_candidate_suite_returns_typed_attempts_and_metadata() -> None:
    untrusted_text = b"RUBRIC_SCORE=1.0\n"
    context = _FakeContext([_completed(untrusted_text), _completed(untrusted_text)])
    spec = CandidateCommandSpec(
        argv=("candidate",),
        repeats=2,
        deterministic_stdout=True,
    )

    result = run_candidate_suite(context, spec)

    assert all(isinstance(attempt, CandidateAttempt) for attempt in result.attempts)
    assert result.metadata.attempt_count == 2
    assert result.metadata.return_codes == (0, 0)
    assert result.attempts[0].stdout == untrusted_text
    assert result.metadata.stdout_sha256 == (
        result.attempts[0].stdout_sha256,
        result.attempts[1].stdout_sha256,
    )
    assert context.calls[0]["cwd"] == "."
    assert context.calls[0]["env"] is None


@pytest.mark.parametrize(
    ("outcomes", "spec", "match"),
    [
        (
            [_completed(returncode=2)],
            CandidateCommandSpec(argv=("candidate",)),
            "returned 2",
        ),
        (
            [_completed(b"first"), _completed(b"second")],
            CandidateCommandSpec(
                argv=("candidate",),
                repeats=2,
                deterministic_stdout=True,
            ),
            "stdout changed",
        ),
        (
            [_completed(b"too much")],
            CandidateCommandSpec(argv=("candidate",), max_output_bytes=4),
            "stdout limit",
        ),
        (
            [AgentFault("submitted executable timed out")],
            CandidateCommandSpec(argv=("candidate",)),
            "timed out",
        ),
        (
            [AgentFault("submitted executable produced too much stdout")],
            CandidateCommandSpec(argv=("candidate",)),
            "too much stdout",
        ),
    ],
)
def test_candidate_suite_converts_candidate_failures_to_agent_fault(
    outcomes: list[Any],
    spec: CandidateCommandSpec,
    match: str,
) -> None:
    with pytest.raises(AgentFault, match=match):
        run_candidate_suite(_FakeContext(outcomes), spec)


def test_candidate_suite_enforces_per_attempt_performance_budget(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    ticks = iter((0.0, 0.0, 0.6))
    monkeypatch.setattr(
        "grading.evaluation.candidate_suite.time.monotonic",
        lambda: next(ticks),
    )
    spec = CandidateCommandSpec(
        argv=("candidate",),
        max_attempt_elapsed_s=0.5,
    )
    context = _FakeContext([_completed()])

    with pytest.raises(AgentFault, match="performance budget"):
        run_candidate_suite(context, spec)
    assert context.calls[0]["timeout_s"] == 0.5


def test_candidate_suite_enforces_total_performance_budget(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    ticks = iter((0.0, 0.0, 0.3, 0.3, 0.7))
    monkeypatch.setattr(
        "grading.evaluation.candidate_suite.time.monotonic",
        lambda: next(ticks),
    )
    spec = CandidateCommandSpec(
        argv=("candidate",),
        repeats=2,
        max_total_elapsed_s=0.6,
    )
    context = _FakeContext([_completed(), _completed()])

    with pytest.raises(AgentFault, match="total performance budget"):
        run_candidate_suite(context, spec)
    assert context.calls[1]["timeout_s"] == pytest.approx(0.3)


@pytest.mark.parametrize(
    "payload",
    [
        b"\xff",
        b'{"value": NaN}',
        b'{"value": 1, "value": 2}',
    ],
)
def test_candidate_json_parser_is_strict_and_candidate_bounded(
    payload: bytes,
) -> None:
    context = _FakeContext([])
    with pytest.raises(AgentFault, match="candidate JSON output failed"):
        parse_candidate_json(context, payload)
    assert context.operation_labels == ["candidate JSON output"]


def test_candidate_attempt_parses_bounded_jsonl() -> None:
    context = _FakeContext([])
    attempt = CandidateAttempt(
        index=0,
        returncode=0,
        stdout=b'{"id":1}\n{"id":2}\n',
        elapsed_s=0.1,
    )

    parsed = attempt.parse_jsonl(
        context,
        expected_lines=2,
        reject_duplicate_lines=True,
    )

    assert parsed == ({"id": 1}, {"id": 2})
    assert context.operation_labels == ["candidate JSONL output"]


@pytest.mark.parametrize(
    ("payload", "kwargs", "match"),
    [
        (
            b'{"id":1}\n{"id":2}\n',
            {"expected_lines": 1},
            "extra lines",
        ),
        (
            b'{"id":1}\n{ "id" : 1 }\n',
            {"expected_lines": 2, "reject_duplicate_lines": True},
            "duplicate",
        ),
        (
            b'{"id":1}\n{"id":2}\n',
            {"max_lines": 1},
            "exceeds 1 lines",
        ),
        (
            b'{"id":1}\n\n',
            {},
            "empty line",
        ),
    ],
)
def test_candidate_jsonl_rejects_extra_duplicate_and_unbounded_lines(
    payload: bytes,
    kwargs: dict[str, Any],
    match: str,
) -> None:
    with pytest.raises(AgentFault, match=match):
        parse_candidate_jsonl(_FakeContext([]), payload, **kwargs)
