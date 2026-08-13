"""Tests for the rubric MCP runtime (`taiga_runtime/rubric`).

These encode the contract behind three QA "harness"-owned findings:

* the grader's headline score (caps / gates / calibration / binary collapse)
  must reach the platform reward, not be stranded in metadata while the reward
  is recomputed as a raw weighted subscore sum;
* grading-side secrets must not leak into agent-facing subprocess environments;
* the agent transcript must be reachable by the grader so trajectory-based
  anti-cheat checks are no longer dead code.
"""

from __future__ import annotations

import asyncio
import glob
import hashlib
import json
import os
import tempfile
import textwrap
import time
from pathlib import Path
from types import SimpleNamespace

import pytest
from grading import runtime_hardening
from rubric import server


def _reward(grade: server.Grade) -> float:
    """The quantity the platform records: sum(subscore * weight)."""
    return sum(grade.subscores[k] * grade.weights.get(k, 0.0) for k in grade.subscores)


def test_setup_private_roots_seal_authoring_metadata_and_solution() -> None:
    assert "/task/task.toml" in server._SETUP_PRIVATE_ROOTS
    assert "/solution" in server._SETUP_PRIVATE_ROOTS
    assert "/task" not in server._SETUP_PRIVATE_ROOTS
    assert "/task/task.toml" not in server._SETUP_READONLY_PRIVATE_ROOTS
    assert "/solution" not in server._SETUP_READONLY_PRIVATE_ROOTS


def test_promoted_calibration_digest_is_verified(monkeypatch, tmp_path) -> None:
    lock = tmp_path / "calibration.lock.json"
    plan_sha = "b" * 64
    lock.write_text(
        json.dumps(
            {
                "schema_version": "3.0",
                "evaluation_plan_sha256": plan_sha,
            }
        )
        + "\n"
    )
    digest = hashlib.sha256(lock.read_bytes()).hexdigest()
    monkeypatch.setattr(server, "Path", lambda _raw: lock)

    evidence = {
        "calibration": {
            "lock_sha256": digest,
            "evaluation_plan_sha256": plan_sha,
        }
    }
    server._verify_calibration(evidence)

    with pytest.raises(RuntimeError, match="digest mismatch"):
        server._verify_calibration(
            {
                "calibration": {
                    "lock_sha256": "0" * 64,
                    "evaluation_plan_sha256": plan_sha,
                }
            }
        )

    (tmp_path / ".author-source").write_text("fallback\n")
    with pytest.raises(RuntimeError, match="trusted-CI promoted mount"):
        server._verify_calibration(
            {
                "calibration": {
                    "lock_sha256": digest,
                    "evaluation_plan_sha256": plan_sha,
                    "requires_trusted_mount": True,
                }
            }
        )
    server._verify_calibration(
        {
            "calibration": {
                "lock_sha256": digest,
                "evaluation_plan_sha256": plan_sha,
                "requires_trusted_mount": False,
            }
        }
    )


def test_continuous_evaluation_rejects_missing_evidence_and_stale_attestation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("LBX_EVALUATION_PLAN_ATTESTED", "1")
    fields = {"continuous_evaluation": {"required": True}}

    with pytest.raises(RuntimeError, match="requires trusted calibration"):
        server._verify_continuous_evaluation(fields)

    assert "LBX_EVALUATION_PLAN_ATTESTED" not in os.environ


def test_continuous_evaluation_attests_only_current_verified_call(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(server, "_verify_calibration", lambda _fields: True)
    monkeypatch.setattr(server, "_verify_evaluation_plan", lambda _fields: False)

    assert server._verify_continuous_evaluation(
        {
            "continuous_evaluation": {
                "required": True,
                "attestation_required": True,
            }
        }
    )
    assert os.environ["LBX_EVALUATION_PLAN_ATTESTED"] == "1"


def test_rubric_evaluation_uses_same_plan_attestation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(server, "_verify_calibration", lambda _fields: False)
    monkeypatch.setattr(server, "_verify_evaluation_plan", lambda _fields: True)

    assert server._verify_continuous_evaluation(
        {
            "rubric_evaluation": {
                "required": True,
                "attestation_required": True,
            }
        }
    )
    assert os.environ["LBX_EVALUATION_PLAN_ATTESTED"] == "1"


def test_local_verified_evaluation_remains_unattested(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(server, "_verify_calibration", lambda _fields: True)
    monkeypatch.setattr(server, "_verify_evaluation_plan", lambda _fields: False)

    assert server._verify_continuous_evaluation(
        {"continuous_evaluation": {"required": True}}
    )
    assert "LBX_EVALUATION_PLAN_ATTESTED" not in os.environ


# ── Cluster A: headline preservation ─────────────────────────────


def test_capped_headline_reaches_reward() -> None:
    """A headline below the weighted subscore sum (a cap/gate) is the reward."""
    payload = {
        "score": 0.28,  # grader's capped headline
        "structured_subscores": [
            {"criterion_id": "a", "description": "crit a", "score": 0.9, "weight": 0.5},
            {"criterion_id": "b", "description": "crit b", "score": 0.8, "weight": 0.5},
        ],
        "weights": {"a": 0.5, "b": 0.5},  # weighted sum would be 0.85
        "metadata": {},
    }
    grade = server._grade_from_payload(payload)
    assert _reward(grade) == pytest.approx(0.28)
    # The per-criterion view survives for the Boreal UI / credit assignment.
    assert grade.metadata["structured_subscores"]
    assert grade.metadata["headline_score"] == pytest.approx(0.28)


def test_plain_weighted_rubric_left_unchanged() -> None:
    """When headline == weighted sum, the per-criterion rows pass through."""
    payload = {
        "score": 0.85,  # == 0.9*0.5 + 0.8*0.5
        "structured_subscores": [
            {"criterion_id": "a", "description": "crit a", "score": 0.9, "weight": 0.5},
            {"criterion_id": "b", "description": "crit b", "score": 0.8, "weight": 0.5},
        ],
        "weights": {"a": 0.5, "b": 0.5},
        "metadata": {},
    }
    grade = server._grade_from_payload(payload)
    assert "score" not in grade.subscores
    assert "__headline_score__" not in grade.subscores
    assert _reward(grade) == pytest.approx(0.85)


def test_scalar_payload_left_unchanged() -> None:
    """The bare-float path already scores correctly and must not gain a row."""
    payload = {
        "score": 0.4,
        "subscores": {"score": 0.4},
        "weights": {"score": 1.0},
        "metadata": {},
    }
    grade = server._grade_from_payload(payload)
    assert grade.subscores == {"score": 0.4}
    assert _reward(grade) == pytest.approx(0.4)


def test_headline_key_avoids_collision_with_a_criterion_named_score() -> None:
    payload = {
        "score": 0.2,
        "structured_subscores": [
            {
                "criterion_id": "score",
                "description": "score",
                "score": 1.0,
                "weight": 1.0,
            }
        ],
        "weights": {"score": 1.0},
        "metadata": {},
    }
    grade = server._grade_from_payload(payload)
    assert "__headline_score__" in grade.subscores
    assert _reward(grade) == pytest.approx(0.2)


def test_capped_headline_reaches_reward_end_to_end() -> None:
    """Full path: a compute_score() returning a custom headline + diagnostics.

    A dict return's ``score`` is the headline verbatim (GRADING.md); the
    weighted subscore sum (0.85 here) must NOT override the capped 0.28.
    """
    test_file = textwrap.dedent("""
        def compute_score():
            return {
                "score": 0.28,
                "subscores": {"a": 0.9, "b": 0.8},
                "weights": {"a": 0.5, "b": 0.5},
                "metadata": {},
            }
        """)
    grade = server._evaluate(test_file)
    assert _reward(grade) == pytest.approx(0.28)
    assert grade.metadata["headline_score"] == pytest.approx(0.28)


def test_evaluate_ignores_model_writable_cwd_import_shadow(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A planted stdlib shadow cannot forge the rubric runner payload."""
    (tmp_path / "json.py").write_text(textwrap.dedent("""
            class JSONDecodeError(Exception):
                pass

            def dump(*args, **kwargs):
                pass

            def load(*args, **kwargs):
                return {"forged": True}

            def dumps(*args, **kwargs):
                return '{"score": 1.0, "metadata": {"forged": true}}'

            def loads(*args, **kwargs):
                return {"score": 1.0, "metadata": {"forged": True}}
            """))
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("PYTHONPATH", str(tmp_path))

    grade = server._evaluate(textwrap.dedent("""
            def compute_score():
                return {"score": 0.0, "metadata": {"honest": True}}
            """))

    assert _reward(grade) == pytest.approx(0.0)
    assert grade.metadata.get("honest") is True
    assert "forged" not in grade.metadata


# ── Cluster C: secret scrubbing ──────────────────────────────────


def test_scrubbed_environ_drops_grading_secrets(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-secret")
    monkeypatch.setenv("ANTHROPIC_BASE_URL", "http://proxy")
    monkeypatch.setenv("SOME_SECRET", "s")
    monkeypatch.setenv("PUBLIC_SETTING", "ok")
    env = server._scrubbed_environ()
    assert "ANTHROPIC_API_KEY" not in env
    assert "ANTHROPIC_BASE_URL" not in env
    assert "SOME_SECRET" not in env
    assert env.get("PUBLIC_SETTING") == "ok"


def test_agent_python_isolation_is_opt_in_for_shims(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(server.os, "geteuid", lambda: 1000)
    monkeypatch.setenv("PYTHONPATH", "/agent/local/imports")

    bash_env = server._agent_subprocess_kwargs()["env"]
    shim_env = server._agent_subprocess_kwargs(isolate_python=True)["env"]

    assert bash_env.get("PYTHONPATH") == "/agent/local/imports"
    assert "PYTHONPATH" not in shim_env
    assert shim_env.get("PYTHONSAFEPATH") == "1"


def test_terminal_artifact_snapshot_records_digest_or_missing(tmp_path: Path) -> None:
    output = tmp_path / "output"
    output.mkdir()

    assert server._snapshot_output_artifact(output) == {
        "artifact_snapshot_state": "artifact_missing"
    }

    (output / "policy.py").write_text("def act(obs): return 0\n")
    snapshot = server._snapshot_output_artifact(output)

    assert snapshot["artifact_snapshot_state"] == "artifact_committed"
    assert len(snapshot["artifact_digest"]) == 64


def _clear_agent_identity_env(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in (
        "RUBRIC_AGENT_USER",
        "RUBRIC_AGENT_UID",
        "RUBRIC_AGENT_GID",
        "RUBRIC_AGENT_HOME",
        "RUBRIC_AGENT_MEMORY_LIMIT_BYTES",
    ):
        monkeypatch.delenv(name, raising=False)


def test_agent_subprocess_kwargs_fails_closed_when_root_user_missing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _clear_agent_identity_env(monkeypatch)
    monkeypatch.setattr(server.os, "geteuid", lambda: 0)
    monkeypatch.setattr(
        server.pwd,
        "getpwnam",
        lambda name: (_ for _ in ()).throw(KeyError(name)),
    )

    with pytest.raises(RuntimeError, match="cannot drop privileges"):
        server._agent_subprocess_kwargs()


def test_agent_tools_do_not_spawn_when_privilege_drop_is_unavailable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _clear_agent_identity_env(monkeypatch)
    monkeypatch.setattr(server.os, "geteuid", lambda: 0)
    monkeypatch.setattr(
        server.pwd,
        "getpwnam",
        lambda name: (_ for _ in ()).throw(KeyError(name)),
    )

    def fail_run(*_args, **_kwargs):
        raise AssertionError("agent subprocess should not start")

    monkeypatch.setattr(server.subprocess, "run", fail_run)

    bash_result = asyncio.run(server.bash("id"))
    editor_result = asyncio.run(
        server.str_replace_editor(command="view", path="notes.txt")
    )

    assert "cannot drop privileges" in (bash_result.error or "")
    assert "cannot drop privileges" in (editor_result.error or "")


def test_agent_subprocess_kwargs_uses_configured_numeric_identity(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _clear_agent_identity_env(monkeypatch)
    monkeypatch.setattr(server.os, "geteuid", lambda: 0)
    monkeypatch.setenv("RUBRIC_AGENT_USER", "worker")
    monkeypatch.setenv("RUBRIC_AGENT_UID", "1234")
    monkeypatch.setenv("RUBRIC_AGENT_GID", "1235")
    monkeypatch.setenv("RUBRIC_AGENT_HOME", "/tmp/worker-home")
    monkeypatch.setattr(
        server.pwd,
        "getpwuid",
        lambda uid: (_ for _ in ()).throw(KeyError(uid)),
    )

    kwargs = server._agent_subprocess_kwargs(isolate_python=True)

    assert kwargs["user"] == 1234
    assert kwargs["group"] == 1235
    assert kwargs["extra_groups"] == []
    assert callable(kwargs["preexec_fn"])
    assert kwargs["env"]["HOME"] == "/tmp/worker-home"
    assert kwargs["env"]["USER"] == "worker"
    assert kwargs["env"]["LOGNAME"] == "worker"
    assert kwargs["env"]["PYTHONSAFEPATH"] == "1"


def test_agent_subprocess_kwargs_uses_configured_user_name(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _clear_agent_identity_env(monkeypatch)
    monkeypatch.setattr(server.os, "geteuid", lambda: 0)
    monkeypatch.setenv("RUBRIC_AGENT_USER", "worker")

    def fake_getpwnam(name: str):
        assert name == "worker"
        return SimpleNamespace(
            pw_uid=1234,
            pw_gid=1235,
            pw_dir="/home/worker",
            pw_name="worker",
        )

    monkeypatch.setattr(server.pwd, "getpwnam", fake_getpwnam)

    kwargs = server._agent_subprocess_kwargs()

    assert kwargs["user"] == 1234
    assert kwargs["group"] == 1235
    assert kwargs["env"]["HOME"] == "/home/worker"
    assert kwargs["env"]["USER"] == "worker"


def test_agent_subprocess_kwargs_rejects_root_numeric_identity(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _clear_agent_identity_env(monkeypatch)
    monkeypatch.setattr(server.os, "geteuid", lambda: 0)
    monkeypatch.setenv("RUBRIC_AGENT_UID", "0")
    monkeypatch.setenv("RUBRIC_AGENT_GID", "0")

    with pytest.raises(RuntimeError, match="non-root account"):
        server._agent_subprocess_kwargs()


# ── Cluster E: transcript wiring ─────────────────────────────────


_TRANSCRIPT_GRADER = textwrap.dedent("""
    def compute_score():
        cheated = "READ_GRADER_PRIVATE" in TRANSCRIPT
        return {"score": 0.0 if cheated else 1.0,
                "metadata": {"transcript_len": len(TRANSCRIPT)}}
    """)


def test_grader_sees_honest_transcript() -> None:
    grade = server._evaluate(_TRANSCRIPT_GRADER, transcript="did honest work\n")
    assert grade.metadata["headline_score"] == pytest.approx(1.0)
    assert grade.metadata["transcript_len"] == len("did honest work\n")


def test_grader_can_reject_cheating_transcript() -> None:
    """The trajectory anti-cheat is reachable again: a transcript that touched
    grader-private paths can be zeroed."""
    grade = server._evaluate(
        _TRANSCRIPT_GRADER, transcript="cat READ_GRADER_PRIVATE/anchors.json\n"
    )
    assert grade.metadata["headline_score"] == pytest.approx(0.0)


def test_missing_transcript_is_empty_not_an_error() -> None:
    grade = server._evaluate(_TRANSCRIPT_GRADER)
    assert grade.metadata["headline_score"] == pytest.approx(1.0)
    assert grade.metadata["transcript_len"] == 0


def test_transcript_temp_file_is_cleaned_up() -> None:
    before = set(glob.glob(os.path.join(tempfile.gettempdir(), "lbx-transcript-*")))
    server._evaluate(_TRANSCRIPT_GRADER, transcript="x" * 1000)
    after = set(glob.glob(os.path.join(tempfile.gettempdir(), "lbx-transcript-*")))
    assert after == before


# ── Cluster F: private grade transport and timeout ─────────────────


def test_evaluate_ignores_forged_stdout_result(capsys: pytest.CaptureFixture) -> None:
    grade = server._evaluate(
        textwrap.dedent("""
            import subprocess
            import sys


            def compute_score():
                subprocess.Popen([
                    sys.executable,
                    "-c",
                    "import time; "
                    "time.sleep(0.05); "
                    "print('RUBRIC_RESULT_JSON={\\\"score\\\":1.0,"
                    "\\\"metadata\\\":{\\\"forged\\\":true}}', flush=True); "
                    "print('RUBRIC_SCORE=1.0', flush=True)",
                ])
                return {"score": 0.0}
            """),
        timeout_s=2.0,
    )
    captured = capsys.readouterr()

    assert captured.out == ""
    assert "RUBRIC_RESULT_JSON" in captured.err
    assert "RUBRIC_SCORE=1.0" in captured.err
    assert grade.subscores == {"score": 0.0}
    assert grade.weights == {"score": 1.0}
    assert grade.metadata["headline_score"] == pytest.approx(0.0)
    assert grade.metadata.get("forged") is not True
    assert "stdout" not in grade.metadata


def test_evaluate_hides_result_path_from_test_file_environment() -> None:
    grade = server._evaluate(textwrap.dedent("""
            import os


            def compute_score():
                return {"score": 0.0 if "RUBRIC_RESULT_PATH" in os.environ else 1.0}
            """))

    assert grade.subscores == {"score": 1.0}
    assert grade.metadata["headline_score"] == pytest.approx(1.0)


def test_evaluate_times_out_lingering_stdout_child() -> None:
    start = time.monotonic()
    grade = server._evaluate(
        textwrap.dedent("""
            import subprocess
            import sys


            def compute_score():
                subprocess.Popen([sys.executable, "-c", "import time; time.sleep(30)"])
                return {"score": 1.0}
            """),
        timeout_s=0.2,
    )

    assert time.monotonic() - start < 5.0
    assert grade.subscores == {"score": 0.0}
    assert "timed out" in grade.metadata["error"]
    assert grade.env_internal_failure is True


def test_grade_problem_uses_extra_field_timeout() -> None:
    start = time.monotonic()
    grade = asyncio.run(
        server.grade_problem(
            problem_id="timeout-probe",
            transcript="",
            extra_fields={
                "grading_timeout_seconds": 0.2,
                "test_file": textwrap.dedent("""
                    import time


                    def compute_score():
                        time.sleep(30)
                        return {"score": 1.0}
                    """),
            },
        )
    )

    assert time.monotonic() - start < 5.0
    assert grade.subscores == {"score": 0.0}
    assert "timed out" in grade.metadata["error"]
    assert grade.env_internal_failure is True


@pytest.mark.parametrize("extra_fields", [None, {}, {"test_file": ""}])
def test_grade_problem_without_test_file_is_an_infra_failure(extra_fields) -> None:
    """A missing test_file is a caller fault, so it must not read as a real 0.0.

    Without this, any regrade or monitoring lane that omits test_file records a
    clean zero that is indistinguishable from a failed agent attempt.
    """
    grade = asyncio.run(
        server.grade_problem(
            problem_id="missing-test-file-probe",
            transcript="",
            extra_fields=extra_fields,
        )
    )

    assert grade.subscores == {"score": 0.0}
    assert grade.env_internal_failure is True
    assert grade.env_internal_failure_logs
    assert "test_file" in grade.metadata["error"]
    assert grade.metadata["grading_state"] == "grade_skipped"
    assert grade.metadata["grade_attempted"] is False


def test_agent_fault_scores_zero_without_env_internal_failure() -> None:
    grade = server._evaluate(
        textwrap.dedent("""
            from grading import AgentFault


            def compute_score():
                raise AgentFault("missing submission")
            """),
        timeout_s=2.0,
    )

    assert grade.subscores == {"score": 0.0}
    assert grade.env_internal_failure is False
    assert grade.metadata["agent_fault"] == "missing submission"
    assert grade.metadata["grading_state"] == "graded"


def test_pre_grade_quiesce_failure_is_infrastructure_failure(monkeypatch) -> None:
    def fail_quiesce(_output_dir):
        raise server.ProcessQuiesceError("respawning agent processes")

    monkeypatch.setattr(server, "pre_grade_cleanup", fail_quiesce)
    grade = server._evaluate("def compute_score():\n    return 1.0\n")

    assert _reward(grade) == 0.0
    assert grade.env_internal_failure is True
    assert "quiesce failed" in grade.metadata["error"]
    assert grade.metadata["grading_state"] == "grader_infra"


def test_respawning_agent_quiesce_failure_is_kept_zero(monkeypatch) -> None:
    def fail_quiesce(_output_dir):
        raise server.AgentProcessQuiesceError("respawning agent processes")

    monkeypatch.setattr(server, "pre_grade_cleanup", fail_quiesce)
    grade = server._evaluate("def compute_score():\n    return 1.0\n")

    assert _reward(grade) == 0.0
    assert grade.env_internal_failure is False
    assert grade.metadata["agent_fault"] == "respawning agent processes"


def _fake_statvfs(*, free_bytes: int, total_inodes: int, free_inodes: int):
    frsize = 4096
    blocks = free_bytes // frsize
    return SimpleNamespace(
        f_bfree=blocks,
        f_frsize=frsize,
        f_files=total_inodes,
        f_ffree=free_inodes,
    )


def test_staging_enospc_is_authoritative_zero(monkeypatch) -> None:
    monkeypatch.setattr(
        server, "_pre_grade_resource_cleanup", lambda: (None, {}, False)
    )
    monkeypatch.setattr(
        server,
        "sample_resource_exhaustion",
        lambda *a, **k: runtime_hardening.ResourceExhaustion(
            disk=False, shmem=False, meminfo_kib={}
        ),
    )
    monkeypatch.setattr(server, "_stage_result_file", lambda: (None, True))

    grade = server._evaluate("def compute_score():\n    return 1.0\n")

    assert _reward(grade) == 0.0
    assert grade.env_internal_failure is False
    assert grade.metadata["agent_fault"] == "disk_exhausted"


def test_tmpfs_flood_is_authoritative_zero(monkeypatch) -> None:
    monkeypatch.setattr(
        server,
        "_pre_grade_resource_cleanup",
        lambda: (1000, {"pre_grade_agent_tmpfs_flood": True}, True),
    )

    grade = server._evaluate("def compute_score():\n    return 1.0\n")

    assert _reward(grade) == 0.0
    assert grade.env_internal_failure is False
    assert grade.metadata["agent_fault"] == "agent_tmpfs_flood"


def test_grader_timeout_on_a_full_disk_is_charged_to_the_agent(monkeypatch) -> None:
    """A disk-full hang is the agent's zero, not a voided episode.

    Boreal's timeout path used to consult shared memory only, so an agent that
    filled the filesystem and hung the grader got the attempt discarded and
    retried. Harbor already charged it; both paths now share one verdict.
    """
    monkeypatch.setattr(
        server, "_pre_grade_resource_cleanup", lambda: (None, {}, False)
    )
    monkeypatch.setattr(
        server,
        "sample_resource_exhaustion",
        lambda *a, **k: runtime_hardening.ResourceExhaustion(
            disk=True, shmem=False, meminfo_kib={}
        ),
    )
    monkeypatch.setattr(server, "_evaluate_timeout", lambda _t: 0.05)

    grade = server._evaluate(
        "import time\n\ndef compute_score():\n    time.sleep(30)\n    return 1.0\n"
    )

    assert _reward(grade) == 0.0
    assert grade.env_internal_failure is False
    assert grade.metadata["agent_fault"] == "disk_exhausted"


def test_recorded_meminfo_describes_the_failure_not_the_recovery() -> None:
    """A disputed zero has to be filed with the reading that justified it.

    Re-reading /proc/meminfo when the grade is built would describe the box
    after cleanup released the agent's /dev/shm entries, i.e. a healthy machine
    next to an accusation of exhausting it.
    """
    at_failure = {"MemTotal": 16_000_000, "MemAvailable": 100_000, "Shmem": 15_000_000}

    grade = server._agent_resource_grade(
        "shared_memory_exhausted",
        "agent exhausted shared memory during grading",
        {},
        meminfo=at_failure,
    )

    assert grade.metadata["meminfo_kib_at_failure"] == at_failure


def test_oom_score_adjustments_are_best_effort(tmp_path: Path) -> None:
    self_target = tmp_path / "self-oom-score"
    self_target.write_text("0")
    server._protect_grader_from_oom(str(self_target))
    assert self_target.read_text() == "-1000"

    proc_dir = tmp_path / "4242"
    proc_dir.mkdir()
    child_target = proc_dir / "oom_score_adj"
    child_target.write_text("0")
    server._bias_agent_child_toward_oom(4242, proc_root=str(tmp_path))
    assert child_target.read_text() == str(server._AGENT_OOM_SCORE_ADJ)

    server._protect_grader_from_oom(str(tmp_path / "missing" / "score"))
    server._bias_agent_child_toward_oom(999999, proc_root=str(tmp_path))


def test_boreal_runner_invokes_declarative_task_directly() -> None:
    grade = server._evaluate(
        textwrap.dedent("""
            from grading.evaluation import RubricCriterion, RubricTask

            class Artifact:
                path = "none"
                def load(self, workspace):
                    return {"ok": True}
                def spec_dict(self):
                    return {"type": "test-artifact.v1"}

            def evaluate(context):
                return {"ok": context.candidate["ok"]}

            TASK = RubricTask(
                artifact=Artifact(),
                criteria=(RubricCriterion("ok"),),
                evaluate=evaluate,
            )
            """),
        timeout_s=2.0,
    )

    assert grade.subscores == {"ok": 1.0}
    assert grade.env_internal_failure is None
    assert grade.metadata["evaluation"]["protocol"] == "declarative-rubric.v1"


def test_untyped_candidate_evaluation_crash_is_kept_zero_with_alert() -> None:
    grade = server._evaluate(
        textwrap.dedent("""
            def compute_score():
                raise OverflowError("int too large to convert to float")
            """),
        timeout_s=2.0,
    )

    assert grade.subscores == {"score": 0.0}
    assert grade.env_internal_failure is False
    assert grade.metadata["critical_operator_alert"] is True
    assert "OverflowError" in grade.metadata["error"]


def test_untyped_noninfra_process_exit_is_kept_zero_with_alert(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        server, "_RUNNER", "import sys\nsys.stdin.read()\nsys.exit(1)\n"
    )

    grade = server._evaluate("def compute_score(): return 1.0", timeout_s=2.0)

    assert grade.subscores == {"score": 0.0}
    assert grade.env_internal_failure is False
    assert grade.metadata["critical_operator_alert"] is True
    assert grade.metadata["failure_classification"] == "unclassified_non_infra_exit"


def test_signal_process_exit_remains_env_internal_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        server,
        "_RUNNER",
        "import os, signal, sys\nsys.stdin.read()\nos.kill(os.getpid(), signal.SIGKILL)\n",
    )

    grade = server._evaluate("def compute_score(): return 1.0", timeout_s=2.0)

    assert grade.subscores == {"score": 0.0}
    assert grade.env_internal_failure is True


def test_attested_agent_fault_keeps_zero_and_records_replay_identity(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    output = tmp_path / "output"
    output.mkdir()
    stale_trace = output / ".lbx-evaluation"
    stale_trace.mkdir()
    stale_trace.joinpath("evaluation-details.json").write_text('{"stale": true}\n')
    monkeypatch.setattr(server, "OUTPUT_DIR", output)
    monkeypatch.setenv("LBX_EVALUATION_PLAN_ATTESTED", "1")

    grade = server._evaluate(
        textwrap.dedent("""
            from grading import AgentFault


            def compute_score():
                raise AgentFault("missing submission")
            """),
        timeout_s=2.0,
        trace_required=True,
    )

    trace = json.loads(
        (output / ".lbx-evaluation" / "evaluation-details.json").read_text()
    )
    assert grade.subscores == {"score": 0.0}
    assert grade.env_internal_failure is False
    assert trace["protocol"] == "agent-fault.v1"
    assert len(trace["replay"]["nonce"]) == 32


def test_attested_unclassified_crash_keeps_zero_and_records_failure_trace(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    output = tmp_path / "output"
    output.mkdir()
    monkeypatch.setattr(server, "OUTPUT_DIR", output)
    monkeypatch.setenv("LBX_EVALUATION_PLAN_ATTESTED", "1")

    grade = server._evaluate(
        "def compute_score():\n    raise ValueError('candidate-triggered decode')\n",
        timeout_s=2.0,
        trace_required=True,
    )

    trace = json.loads(
        (output / ".lbx-evaluation" / "evaluation-details.json").read_text()
    )
    assert grade.subscores == {"score": 0.0}
    assert grade.env_internal_failure is False
    assert grade.metadata["return_shape"] == "unclassified_grader_crash"
    assert grade.metadata["critical_operator_alert"] is True
    assert trace["protocol"] == "unclassified-grader-crash.v1"
    assert len(trace["replay"]["nonce"]) == 32


def test_non_finite_score_is_env_internal_failure() -> None:
    grade = server._evaluate(
        textwrap.dedent("""
            import math


            def compute_score():
                return math.nan
            """),
        timeout_s=2.0,
    )

    assert grade.subscores == {"score": 0.0}
    assert grade.env_internal_failure is True
    assert "non-finite" in grade.metadata["error"]


def test_evaluate_fails_closed_when_result_file_is_missing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        server,
        "_RUNNER",
        "import sys\nsys.stdin.read()\nsys.exit(0)\n",
    )

    grade = server._evaluate("def compute_score():\n    return 1.0\n", timeout_s=1.0)

    assert grade.subscores == {"score": 0.0}
    assert grade.weights == {"score": 1.0}
    assert grade.metadata["error"] == "missing rubric result"


def test_result_temp_file_is_cleaned_up() -> None:
    before = set(glob.glob(os.path.join(tempfile.gettempdir(), "lbx-rubric-result-*")))
    grade = server._evaluate(textwrap.dedent("""
            def compute_score():
                return {"score": 1.0}
            """))
    after = set(glob.glob(os.path.join(tempfile.gettempdir(), "lbx-rubric-result-*")))

    assert grade.subscores == {"score": 1.0}
    assert after == before


def test_attested_evaluate_persists_private_trace_in_output(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    output = tmp_path / "output"
    output.mkdir()
    stale_trace = output / ".lbx-evaluation"
    stale_trace.mkdir()
    stale_trace.joinpath("evaluation-details.json").write_text('{"stale": true}\n')
    monkeypatch.setattr(server, "OUTPUT_DIR", output)
    monkeypatch.setenv("LBX_EVALUATION_PLAN_ATTESTED", "1")
    monkeypatch.setenv("TEST_STALE_TRACE_PATH", str(stale_trace))

    grade = server._evaluate(
        textwrap.dedent("""
            import json
            import os


            def compute_score():
                assert not os.path.exists(os.environ["TEST_STALE_TRACE_PATH"])
                path = os.environ["LBX_EVALUATION_TRACE_PATH"]
                payload = {
                    "nonce": os.environ["LBX_EVALUATION_NONCE"],
                    "private": True,
                }
                flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
                fd = os.open(path, flags, 0o600)
                with os.fdopen(fd, "w") as handle:
                    json.dump(payload, handle)
                return {"score": 0.75}
            """),
        trace_required=True,
    )

    trace_path = output / ".lbx-evaluation" / "evaluation-details.json"
    trace = json.loads(trace_path.read_text())
    assert grade.subscores == {"score": 0.75}
    assert trace["private"] is True
    assert len(trace["nonce"]) == 32
    assert trace_path.stat().st_mode & 0o777 == 0o600
    assert trace["nonce"] not in json.dumps(grade.metadata)


def test_attested_evaluate_fails_when_required_trace_is_missing(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    output = tmp_path / "output"
    output.mkdir()
    monkeypatch.setattr(server, "OUTPUT_DIR", output)
    monkeypatch.setenv("LBX_EVALUATION_PLAN_ATTESTED", "1")

    grade = server._evaluate(
        "def compute_score():\n    return 1.0\n",
        trace_required=True,
    )

    assert grade.subscores == {"score": 0.0}
    assert grade.env_internal_failure is True
    assert "private replay trace" in grade.metadata["error"]


def test_runner_prepends_grading_deps_before_grader_import() -> None:
    """The rubric runner must prepend /mcp_server/grading_deps to sys.path so a
    task's grading_dependencies import on the platform (Boreal) grade, mirroring
    grader_runner.worker. Regression guard: without this, such tasks pass local CI
    and fail every platform grade as an env_internal_failure. See the grading-deps
    bug ticket."""
    runner = server._RUNNER

    # Resolves the same source-of-truth path as the worker, with a literal fallback
    # for the isolated runner where env_server may be unimportable.
    assert "from env_server.config import GRADING_DEPS_DIR" in runner
    assert '"/mcp_server/grading_deps"' in runner
    # Idempotent + no-op-when-absent guard, matching worker._prepend_grading_deps.
    assert "os.path.isdir(_grading_deps)" in runner
    assert "_grading_deps not in sys.path" in runner
    assert "sys.path.insert(0, _grading_deps)" in runner

    # Must run before the grader source is exec'd (so a top-level grading_dependency
    # import resolves) -- i.e. the prepend precedes the exec(source, ...) call.
    assert runner.index("sys.path.insert(0, _grading_deps)") < runner.index(
        "exec(source, namespace)"
    )
