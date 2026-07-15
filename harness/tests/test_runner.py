from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest
from alignerr_plugin import ground_truth as plugin_ground_truth
from alignerr_plugin.proof import write_build_proof
from alignerr_plugin.schemas import StageResult
from alignerr_plugin.utils import grading_inputs_sha256

from lbx_rl_tasks_harness import ground_truth as ground_truth_module
from lbx_rl_tasks_harness.cli import _expand_runtime
from lbx_rl_tasks_harness.formats.problem_dir import load_problem_dir
from lbx_rl_tasks_harness.models import (
    GroundTruthSpec,
    HarnessProblem,
    OutputSpec,
    ReferenceSpec,
)
from lbx_rl_tasks_harness import runner
from lbx_rl_tasks_harness.runner import run_auto_qa_only, run_harness
from lbx_rl_tasks_harness.runtimes import reference as reference_module
from lbx_rl_tasks_harness.runtimes.reference import (
    ReferenceRunOptions,
    ReferenceRunResult,
)

ROOT = Path(__file__).resolve().parents[2]
MUJOCO = ROOT / "examples" / "mujoco-pendulum"


def test_cli_runtime_expansion() -> None:
    assert _expand_runtime("ground-truth") == ["solution"]
    assert _expand_runtime("rubric-quality") == ["noop"]
    assert _expand_runtime("claude-code") == ["claude-code"]
    assert _expand_runtime("deepagents") == ["deepagents"]


def test_solution_runtime_scores_mujoco_example(monkeypatch, tmp_path: Path) -> None:
    pytest.importorskip("mujoco")
    pytest.importorskip("numpy")
    if shutil.which("ffmpeg") is None:
        pytest.skip("ffmpeg is required for ground-truth review video")
    monkeypatch.setattr(
        runner,
        "_run_rubric_quality_check",
        lambda *_args, **_kwargs: {"status": "skipped", "checks": []},
    )
    problem_dir = tmp_path / "mujoco-pendulum"
    shutil.copytree(MUJOCO, problem_dir)

    class FakeTaskValidator:
        def _local_build_proof(self, _problem_dir):
            return StageResult(passed=True, issues=[], duration_ms=0)

    monkeypatch.setattr(runner, "TaskValidator", FakeTaskValidator)
    problem = load_problem_dir(problem_dir)

    result = run_harness(problem, runtime="solution", run_dir_base=tmp_path)

    reward = json.loads(result.reward_path.read_text())
    assert result.score > 0.9
    assert reward["score"] == result.score
    assert (result.workspace / "model.xml").exists()
    assert result.review_artifacts
    assert result.review_artifacts[0]["path"] == ".alignerr/ground_truth/rendering.mp4"
    assert result.review_artifacts[0]["logical_path"] == "/tmp/output/rendering.mp4"
    assert result.review_artifacts[0]["bytes"] > 0
    assert len(result.review_artifacts[0]["sha256"]) == 64
    assert result.review_artifacts[0]["width"] == 1280
    assert result.review_artifacts[0]["height"] == 720
    assert (problem_dir / ".alignerr" / "ground_truth" / "rendering.mp4").exists()


def test_solution_runtime_requires_perfect_ground_truth(
    monkeypatch, tmp_path: Path
) -> None:
    problem_dir = tmp_path / "problem"
    (problem_dir / "solution").mkdir(parents=True)
    (problem_dir / "solution" / "solve.sh").write_text(
        "mkdir -p /tmp/output\nprintf ok > /tmp/output/result.txt\n"
    )

    def fake_grade_workspace(_problem, _workspace, output_dir, _transcript):
        output_dir.mkdir(parents=True)
        (output_dir / "reward.json").write_text('{"score": 0.99}\n')
        (output_dir / "reward-details.json").write_text(
            '{"score": 0.99, "structured_subscores": [{"name": "criterion", "score": 0.99, "max_score": 1.0}]}\n'
        )
        return 0.99

    monkeypatch.setattr(reference_module, "grade_workspace", fake_grade_workspace)
    problem = HarnessProblem(
        id="demo",
        source_format="problem-dir",
        prompt="demo",
        outputs=[OutputSpec(path="/tmp/output/result.txt")],
        source_problem_dir=problem_dir,
        ground_truth=GroundTruthSpec(
            render_command='printf video > "$LBT_OUTPUT_DIR/rendering.mp4"',
            render_outputs=[OutputSpec(path="/tmp/output/rendering.mp4")],
        ),
        metadata={
            "difficulty": {
                "task_type": "mujoco",
                "domain": "model_environment_construction",
                "reward_type": "multi_deterministic_rubrics",
            }
        },
    )

    with pytest.raises(RuntimeError, match="must score 1.0"):
        run_harness(problem, runtime="solution", run_dir_base=tmp_path / "runs")


def _mock_run_reference(score: float):
    def fake_run_reference(
        problem,
        workspace,
        verifier_dir,
        transcript,
        *,
        options=None,
        run_dir=None,
    ):
        verifier_dir.mkdir(parents=True, exist_ok=True)
        (verifier_dir / "reward.json").write_text(json.dumps({"score": score}) + "\n")
        (verifier_dir / "reward-details.json").write_text(
            json.dumps({"score": score}) + "\n"
        )
        return ReferenceRunResult(
            score=score,
            workspace=workspace,
            verifier_dir=verifier_dir,
            grade_payload={"score": score},
            manifest_path=None,
            execution="host",
            solution_returncode=0,
            grader_returncode=0,
        )

    return fake_run_reference


def test_solution_runtime_accepts_continuous_reference_score(
    monkeypatch, tmp_path: Path
) -> None:
    problem_dir = tmp_path / "problem"
    (problem_dir / "solution").mkdir(parents=True)
    (problem_dir / "solution" / "solve.sh").write_text(
        "mkdir -p /tmp/output\nprintf ok > /tmp/output/result.txt\n"
    )

    monkeypatch.setattr(runner, "run_reference", _mock_run_reference(0.51))
    # ml resolves to container execution, so the zero-anchor no-op probe grades
    # in the task image; stub it for the unit test.
    monkeypatch.setattr(
        runner, "grade_workspace_in_container", lambda *_args, **_kwargs: 0.0
    )
    problem = HarnessProblem(
        id="demo",
        source_format="problem-dir",
        prompt="demo",
        outputs=[OutputSpec(path="/tmp/output/result.txt")],
        source_problem_dir=problem_dir,
        metadata={
            "difficulty": {
                "task_type": "ml",
                "domain": "scientific_discovery_computational_science",
                "reward_type": "continuous_scoring_function",
            }
        },
    )

    result = run_harness(problem, runtime="solution", run_dir_base=tmp_path / "runs")

    assert result.score == 0.51


def test_require_zero_anchored_noop_helper() -> None:
    # Continuous tasks are held to the strict anchor; rubric tasks are not.
    ground_truth_module.require_zero_anchored_noop(
        0.0, reward_type="continuous_scoring_function", zero_anchor_epsilon=0.01
    )
    ground_truth_module.require_zero_anchored_noop(
        0.9, reward_type="multi_deterministic_rubrics", zero_anchor_epsilon=0.01
    )
    ground_truth_module.require_zero_anchored_noop(
        0.2, reward_type="continuous_scoring_function", zero_anchor_epsilon=0.3
    )
    with pytest.raises(RuntimeError, match="anchor an empty submission to 0"):
        ground_truth_module.require_zero_anchored_noop(
            0.2, reward_type="continuous_scoring_function", zero_anchor_epsilon=0.01
        )


def test_ground_truth_run_fails_when_continuous_noop_not_zero(
    monkeypatch, tmp_path: Path
) -> None:
    problem_dir = tmp_path / "problem"
    (problem_dir / "solution").mkdir(parents=True)
    (problem_dir / "solution" / "solve.sh").write_text(
        "mkdir -p /tmp/output\nprintf ok > /tmp/output/result.txt\n"
    )

    monkeypatch.setattr(runner, "run_reference", _mock_run_reference(0.51))
    monkeypatch.setattr(
        runner, "grade_workspace_in_container", lambda *_args, **_kwargs: 0.2
    )
    problem = HarnessProblem(
        id="demo",
        source_format="problem-dir",
        prompt="demo",
        outputs=[OutputSpec(path="/tmp/output/result.txt")],
        source_problem_dir=problem_dir,
        metadata={
            "difficulty": {
                "task_type": "ml",
                "domain": "scientific_discovery_computational_science",
                "reward_type": "continuous_scoring_function",
            }
        },
    )

    with pytest.raises(RuntimeError, match="anchor an empty submission to 0"):
        run_harness(problem, runtime="solution", run_dir_base=tmp_path / "runs")


def test_ground_truth_run_records_trivial_baseline_in_proof(
    monkeypatch, tmp_path: Path
) -> None:
    problem_dir = tmp_path / "problem"
    (problem_dir / "solution").mkdir(parents=True)
    (problem_dir / "solution" / "solve.sh").write_text(
        "mkdir -p /tmp/output\nprintf ok > /tmp/output/result.txt\n"
    )
    write_build_proof(
        problem_dir,
        image_digest="sha256:image",
        base_image_ref="lbx-tasks-base:runtime",
        platform="linux/amd64",
        alignerr_cli_version="0.1.0",
        duration_seconds=1.0,
    )

    graded_workspaces: list[Path] = []

    def fake_grade_noop(_problem, workspace, _verifier_dir, _transcript):
        graded_workspaces.append(workspace)
        return 0.0

    monkeypatch.setattr(runner, "run_reference", _mock_run_reference(0.51))
    monkeypatch.setattr(runner, "grade_workspace_in_container", fake_grade_noop)
    problem = HarnessProblem(
        id="demo",
        source_format="problem-dir",
        prompt="demo",
        outputs=[OutputSpec(path="/tmp/output/result.txt")],
        source_problem_dir=problem_dir,
        metadata={
            "difficulty": {
                "task_type": "ml",
                "domain": "scientific_discovery_computational_science",
                "reward_type": "continuous_scoring_function",
            }
        },
    )

    run_harness(problem, runtime="solution", run_dir_base=tmp_path / "runs")

    # The probe graded an empty workspace ...
    assert len(graded_workspaces) == 1
    assert graded_workspaces[0].is_dir()
    assert not any(graded_workspaces[0].iterdir())
    # ... and the score landed in the committed proof for the task validator.
    proof = json.loads((problem_dir / ".alignerr" / "build_proof.json").read_text())
    assert proof["ground_truth_result"]["trivial_baseline_score"] == 0.0


def test_container_ground_truth_refreshes_build_proof(
    monkeypatch, tmp_path: Path
) -> None:
    problem_dir = tmp_path / "problem"
    (problem_dir / "solution").mkdir(parents=True)
    (problem_dir / "solution" / "solve.sh").write_text(
        "mkdir -p /tmp/output\nprintf ok > /tmp/output/result.txt\n"
    )
    (problem_dir / "task.toml").write_text("[metadata]\nid = 'demo'\n")

    build_proof_calls: list[Path] = []

    class FakeTaskValidator:
        def _local_build_proof(self, refreshed_problem_dir: Path):
            build_proof_calls.append(refreshed_problem_dir)
            return StageResult(passed=True, issues=[], duration_ms=0)

    monkeypatch.setattr(runner, "TaskValidator", FakeTaskValidator)
    monkeypatch.setattr(runner, "run_reference", _mock_run_reference(0.51))
    monkeypatch.setattr(
        runner, "grade_workspace_in_container", lambda *_args, **_kwargs: 0.0
    )
    problem = HarnessProblem(
        id="demo",
        source_format="problem-dir",
        prompt="demo",
        outputs=[OutputSpec(path="/tmp/output/result.txt")],
        source_problem_dir=problem_dir,
        metadata={
            "difficulty": {
                "task_type": "ml",
                "domain": "scientific_discovery_computational_science",
                "reward_type": "continuous_scoring_function",
            }
        },
    )

    run_harness(problem, runtime="solution", run_dir_base=tmp_path / "runs")

    assert build_proof_calls == [problem_dir]


def test_ground_truth_reference_run_skips_author_manifest(
    monkeypatch, tmp_path: Path
) -> None:
    problem_dir = tmp_path / "problem"
    (problem_dir / "solution").mkdir(parents=True)
    (problem_dir / "solution" / "solve.sh").write_text(
        "mkdir -p /tmp/output\nprintf ok > /tmp/output/result.txt\n"
    )

    captured: list[ReferenceRunOptions] = []

    def capture_run_reference(*_args, **kwargs):
        options = kwargs["options"]
        captured.append(options)
        return _mock_run_reference(0.51)(*_args, **kwargs)

    monkeypatch.setattr(runner, "run_reference", capture_run_reference)
    monkeypatch.setattr(
        runner, "grade_workspace_in_container", lambda *_args, **_kwargs: 0.0
    )
    problem = HarnessProblem(
        id="demo",
        source_format="problem-dir",
        prompt="demo",
        outputs=[OutputSpec(path="/tmp/output/result.txt")],
        source_problem_dir=problem_dir,
        metadata={
            "difficulty": {
                "task_type": "ml",
                "domain": "scientific_discovery_computational_science",
                "reward_type": "continuous_scoring_function",
            }
        },
    )

    run_harness(problem, runtime="solution", run_dir_base=tmp_path / "runs")

    assert len(captured) == 1
    assert captured[0].mode == "prove"
    assert captured[0].write_manifest is False


def test_solution_runtime_requires_review_video(monkeypatch, tmp_path: Path) -> None:
    problem_dir = tmp_path / "problem"
    (problem_dir / "solution").mkdir(parents=True)
    (problem_dir / "solution" / "solve.sh").write_text(
        "mkdir -p /tmp/output\nprintf ok > /tmp/output/result.txt\n"
    )

    def fake_grade_workspace(_problem, _workspace, output_dir, _transcript):
        output_dir.mkdir(parents=True)
        (output_dir / "reward.json").write_text('{"score": 1.0}\n')
        (output_dir / "reward-details.json").write_text('{"score": 1.0}\n')
        return 1.0

    monkeypatch.setattr(reference_module, "grade_workspace", fake_grade_workspace)
    problem = HarnessProblem(
        id="demo",
        source_format="problem-dir",
        prompt="demo",
        outputs=[OutputSpec(path="/tmp/output/result.txt")],
        source_problem_dir=problem_dir,
        ground_truth=GroundTruthSpec(
            render_command="true",
            render_outputs=[OutputSpec(path="/tmp/output/rendering.mp4")],
        ),
        metadata={
            "difficulty": {
                "task_type": "mujoco",
                "domain": "model_environment_construction",
                "reward_type": "multi_deterministic_rubrics",
            }
        },
    )

    with pytest.raises(FileNotFoundError, match="required video"):
        run_harness(problem, runtime="solution", run_dir_base=tmp_path / "runs")


def test_validate_video_file_rejects_non_h264_codec(
    monkeypatch, tmp_path: Path
) -> None:
    video = tmp_path / "rendering.mp4"
    video.write_bytes(b"video")
    monkeypatch.setattr(
        plugin_ground_truth, "video_dimensions", lambda _path: (1280, 720)
    )
    monkeypatch.setattr(plugin_ground_truth, "video_codec", lambda _path: "mpeg4")

    with pytest.raises(RuntimeError, match="must use h264 video codec"):
        plugin_ground_truth.validate_video_file(video, "/tmp/output/rendering.mp4")


def test_solution_runtime_requires_rendering_mp4_name(tmp_path: Path) -> None:
    problem = HarnessProblem(
        id="demo",
        source_format="problem-dir",
        prompt="demo",
        outputs=[OutputSpec(path="/tmp/output/result.txt")],
        source_problem_dir=tmp_path / "problem",
        ground_truth=GroundTruthSpec(
            render_command='printf video > "$LBT_OUTPUT_DIR/preview.mp4"',
            render_outputs=[OutputSpec(path="/tmp/output/preview.mp4")],
        ),
        metadata={
            "difficulty": {
                "task_type": "mujoco",
                "domain": "model_environment_construction",
                "reward_type": "multi_deterministic_rubrics",
            }
        },
    )

    issues = ground_truth_module.validate_render_contract(problem)

    assert (
        "ground truth render output must be named /tmp/output/rendering.mp4" in issues
    )


def test_solution_render_uses_agent_timeout(monkeypatch, tmp_path: Path) -> None:
    problem_dir = tmp_path / "problem"
    (problem_dir / "solution").mkdir(parents=True)
    (problem_dir / "solution" / "solve.sh").write_text(
        "mkdir -p /tmp/output\nprintf ok > /tmp/output/result.txt\n"
    )
    captured: dict[str, object] = {}

    def fake_grade_workspace(_problem, _workspace, output_dir, _transcript):
        output_dir.mkdir(parents=True)
        (output_dir / "reward.json").write_text('{"score": 1.0}\n')
        (output_dir / "reward-details.json").write_text('{"score": 1.0}\n')
        return 1.0

    def fake_run_ground_truth_render(problem, **_kwargs):
        captured["timeout"] = problem.metadata.get("agent", {}).get("timeout_sec")
        return []

    monkeypatch.setattr(reference_module, "grade_workspace", fake_grade_workspace)
    monkeypatch.setattr(runner, "run_ground_truth_render", fake_run_ground_truth_render)
    problem = HarnessProblem(
        id="demo",
        source_format="problem-dir",
        prompt="demo",
        outputs=[OutputSpec(path="/tmp/output/result.txt")],
        source_problem_dir=problem_dir,
        ground_truth=GroundTruthSpec(
            render_command="true",
            render_outputs=[OutputSpec(path="/tmp/output/rendering.mp4")],
        ),
        metadata={
            "agent": {"timeout_sec": 7},
            "difficulty": {
                "task_type": "mujoco",
                "domain": "model_environment_construction",
                "reward_type": "multi_deterministic_rubrics",
            },
        },
    )

    run_harness(problem, runtime="solution", run_dir_base=tmp_path / "runs")

    assert captured["timeout"] == 7


def test_run_ground_truth_render_passes_agent_timeout_to_subprocess(
    monkeypatch, tmp_path: Path
) -> None:
    problem_dir = tmp_path / "problem"
    workspace = tmp_path / "workspace"
    run_dir = tmp_path / "run"
    transcript = run_dir / "transcript.txt"
    problem_dir.mkdir()
    workspace.mkdir()
    run_dir.mkdir()
    captured: dict[str, object] = {}

    class Completed:
        returncode = 0
        stdout = ""
        stderr = ""

    def fake_run(*_args, **kwargs):
        captured["timeout"] = kwargs["timeout"]
        output = Path(kwargs["env"]["LBT_OUTPUT_DIR"]) / "rendering.mp4"
        output.write_bytes(b"video")
        return Completed()

    monkeypatch.setattr(ground_truth_module.subprocess, "run", fake_run)
    monkeypatch.setattr(
        ground_truth_module,
        "validate_video_file",
        lambda _path, _logical_path: (1280, 720),
    )
    monkeypatch.setattr(
        ground_truth_module,
        "artifact_metadata",
        lambda _problem_dir, _submission_path, *, logical_path: {
            "path": ".alignerr/ground_truth/rendering.mp4",
            "logical_path": logical_path,
            "sha256": "a" * 64,
            "bytes": 5,
            "width": 1280,
            "height": 720,
        },
    )
    problem = HarnessProblem(
        id="demo",
        source_format="problem-dir",
        prompt="demo",
        outputs=[],
        source_problem_dir=problem_dir,
        ground_truth=GroundTruthSpec(
            render_command="true",
            render_outputs=[OutputSpec(path="/tmp/output/rendering.mp4")],
        ),
        metadata={"agent": {"timeout_sec": 7}, "difficulty": {"task_type": "mujoco"}},
    )

    ground_truth_module.run_ground_truth_render(
        problem, workspace=workspace, run_dir=run_dir, transcript_path=transcript
    )

    assert captured["timeout"] == 7


def test_render_failure_includes_stdout_and_stderr() -> None:
    message = ground_truth_module.render_failure_details(
        "render stdout line\n",
        "render stderr line\n",
    )

    assert "stdout:\nrender stdout line" in message
    assert "stderr:\nrender stderr line" in message


def test_solution_runtime_can_skip_render_and_preserve_existing_artifacts(
    monkeypatch, tmp_path: Path
) -> None:
    problem_dir = tmp_path / "problems" / "demo"
    artifact = problem_dir / ".alignerr" / "ground_truth" / "rendering.mp4"
    artifact.parent.mkdir(parents=True)
    artifact.write_bytes(b"video")
    (problem_dir / "solution").mkdir(parents=True)
    (problem_dir / "solution" / "solve.sh").write_text(
        "mkdir -p /tmp/output\nprintf ok > /tmp/output/result.txt\n"
    )
    (problem_dir / "task.toml").write_text("[task]\nname = 'demo'\n")
    existing_artifacts = [
        {
            "path": ".alignerr/ground_truth/rendering.mp4",
            "logical_path": "/tmp/output/rendering.mp4",
            "sha256": "a" * 64,
            "bytes": 5,
            "width": 1280,
            "height": 720,
        }
    ]
    proof_path = write_build_proof(
        problem_dir,
        image_digest="sha256:old",
        base_image_ref="lbx-tasks-base:runtime",
        platform="linux/amd64",
        alignerr_cli_version="0.1.0",
        duration_seconds=1.0,
    )
    proof = json.loads(proof_path.read_text())
    proof["ground_truth_result"] = {
        "score": 1.0,
        "runtime": "solution",
        "review_artifacts": existing_artifacts,
    }
    proof_path.write_text(json.dumps(proof, indent=2) + "\n")

    def fake_grade_workspace(_problem, _workspace, output_dir, _transcript):
        output_dir.mkdir(parents=True)
        (output_dir / "reward.json").write_text('{"score": 1.0}\n')
        (output_dir / "reward-details.json").write_text('{"score": 1.0}\n')
        return 1.0

    def fail_render(*_args, **_kwargs):
        raise AssertionError("render should be skipped in CI score-only mode")

    class FakeTaskValidator:
        def _local_build_proof(self, refreshed_problem_dir):
            write_build_proof(
                refreshed_problem_dir,
                image_digest="sha256:refreshed",
                base_image_ref="lbx-tasks-base:runtime",
                platform="linux/amd64",
                alignerr_cli_version="0.1.0",
                duration_seconds=1.0,
            )
            return StageResult(passed=True, issues=[], duration_ms=0)

    monkeypatch.setenv("LBX_RL_SKIP_GROUND_TRUTH_RENDER", "1")
    monkeypatch.setattr(reference_module, "grade_workspace", fake_grade_workspace)
    monkeypatch.setattr(runner, "run_ground_truth_render", fail_render)
    monkeypatch.setattr(runner, "TaskValidator", FakeTaskValidator)
    problem = HarnessProblem(
        id="demo",
        source_format="problem-dir",
        prompt="demo",
        outputs=[OutputSpec(path="/tmp/output/result.txt")],
        source_problem_dir=problem_dir,
        ground_truth=GroundTruthSpec(
            render_command="false",
            render_outputs=[OutputSpec(path="/tmp/output/rendering.mp4")],
        ),
        metadata={
            "difficulty": {
                "task_type": "mujoco",
                "domain": "model_environment_construction",
                "reward_type": "multi_deterministic_rubrics",
            }
        },
    )

    result = run_harness(problem, runtime="solution", run_dir_base=tmp_path / "runs")

    assert result.score == 1.0
    assert result.review_artifacts == existing_artifacts
    proof = json.loads(proof_path.read_text())
    assert proof["ground_truth_result"]["score"] == 1.0
    assert proof["ground_truth_result"]["review_artifacts"] == existing_artifacts


def test_run_harness_updates_existing_build_proof_with_grade(
    monkeypatch, tmp_path: Path
) -> None:
    problem_dir = tmp_path / "problems" / "demo"
    problem_dir.mkdir(parents=True)
    (problem_dir / "task.toml").write_text("[task]\nname = 'demo'\n")
    write_build_proof(
        problem_dir,
        image_digest="sha256:image",
        base_image_ref="lbx-tasks-base:runtime",
        platform="linux/amd64",
        alignerr_cli_version="0.1.0",
        duration_seconds=1.0,
    )

    def fake_grade_workspace(_problem, _workspace, output_dir, _transcript):
        output_dir.mkdir(parents=True)
        (output_dir / "reward.json").write_text('{"score": 0.25}\n')
        (output_dir / "reward-details.json").write_text(
            '{"score": 0.25, "subscores": {"criterion": 0.0}, '
            '"weights": {"criterion": 1.0}}\n'
        )
        (output_dir / "reward.txt").write_text("0.250000\n")
        return 0.25

    monkeypatch.setattr(runner, "grade_workspace", fake_grade_workspace)
    monkeypatch.setattr(
        runner,
        "_run_rubric_quality_check",
        lambda *_args, **_kwargs: {
            "status": "completed",
            "checks": [{"check": "Measurability", "status": "pass"}],
        },
    )
    problem = HarnessProblem(
        id="demo",
        source_format="problem-dir",
        prompt="demo",
        outputs=[OutputSpec(path="/tmp/output/result.txt")],
        source_problem_dir=problem_dir,
    )

    result = run_harness(problem, runtime="noop", run_dir_base=tmp_path)

    assert result.score == 0.25
    proof = (problem_dir / ".alignerr" / "build_proof.json").read_text()
    assert '"harness_result"' in proof
    assert '"score": 0.25' in proof
    assert '"subscores"' in proof
    assert '"rubric_quality"' in proof


def test_run_harness_noop_grades_in_container_for_container_tasks(
    monkeypatch, tmp_path: Path
) -> None:
    problem_dir = tmp_path / "problems" / "demo"
    problem_dir.mkdir(parents=True)
    (problem_dir / "task.toml").write_text("[task]\nname = 'demo'\n")
    write_build_proof(
        problem_dir,
        image_digest="sha256:image",
        base_image_ref="lbx-tasks-base:runtime",
        platform="linux/amd64",
        alignerr_cli_version="0.1.0",
        duration_seconds=1.0,
    )

    def fail_host_grade(*_args, **_kwargs):
        raise AssertionError("host grade_workspace must not run for container tasks")

    def fake_container_grade(_problem, _workspace, output_dir, _transcript):
        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / "reward.json").write_text('{"score": 0.0}\n')
        (output_dir / "reward-details.json").write_text('{"score": 0.0}\n')
        return 0.0

    monkeypatch.setattr(runner, "resolve_reference_execution", lambda *_a, **_k: "container")
    monkeypatch.setattr(runner, "grade_workspace", fail_host_grade)
    monkeypatch.setattr(runner, "grade_workspace_in_container", fake_container_grade)
    monkeypatch.setattr(
        runner,
        "_run_rubric_quality_check",
        lambda *_args, **_kwargs: {"status": "completed", "checks": []},
    )
    problem = HarnessProblem(
        id="demo",
        source_format="problem-dir",
        prompt="demo",
        outputs=[OutputSpec(path="/tmp/output/result.txt")],
        source_problem_dir=problem_dir,
    )

    result = run_harness(problem, runtime="noop", run_dir_base=tmp_path)

    assert result.score == 0.0


def test_claude_code_runtime_updates_existing_build_proof(
    monkeypatch, tmp_path: Path
) -> None:
    problem_dir = tmp_path / "problems" / "demo"
    problem_dir.mkdir(parents=True)
    write_build_proof(
        problem_dir,
        image_digest="sha256:image",
        base_image_ref="lbx-tasks-base:runtime",
        platform="linux/amd64",
        alignerr_cli_version="0.1.0",
        duration_seconds=1.0,
    )

    async def fake_run_claude_code(_problem, workspace, transcript, **_kwargs):
        workspace.mkdir(parents=True, exist_ok=True)
        transcript.write_text("transcript")
        return {
            "score": 0.5,
            "subscores": {"criterion": 0.5},
            "weights": {"criterion": 1.0},
        }

    monkeypatch.setattr(runner, "run_claude_code", fake_run_claude_code)
    captured_rubric_model: list[str | None] = []

    def fake_rubric_quality(*_args, **kwargs):
        captured_rubric_model.append(kwargs["model"])
        return {"status": "completed", "checks": []}

    monkeypatch.setattr(runner, "_run_rubric_quality_check", fake_rubric_quality)
    problem = HarnessProblem(
        id="demo",
        source_format="problem-dir",
        prompt="demo",
        outputs=[OutputSpec(path="/tmp/output/result.txt")],
        source_problem_dir=problem_dir,
    )

    result = run_harness(
        problem,
        runtime="claude-code",
        run_dir_base=tmp_path,
        model="claude-sonnet-4-5",
    )

    assert result.score == 0.5
    proof = json.loads((problem_dir / ".alignerr" / "build_proof.json").read_text())
    assert proof["harness_result"]["runtime"] == "claude-code"
    assert proof["harness_result"]["score"] == 0.5
    assert captured_rubric_model == [None]
    manifest = json.loads((result.run_dir / "manifest.json").read_text())
    assert manifest["model"] == "claude-sonnet-4-5"


def test_ml_solution_runtime_does_not_overwrite_harness_result(
    monkeypatch, tmp_path: Path
) -> None:
    problem_dir = tmp_path / "problems" / "demo"
    (problem_dir / "solution").mkdir(parents=True)
    (problem_dir / "solution" / "solve.sh").write_text(
        "mkdir -p /tmp/output\nprintf ok > /tmp/output/result.txt\n"
    )
    write_build_proof(
        problem_dir,
        image_digest="sha256:image",
        base_image_ref="lbx-tasks-base:runtime",
        platform="linux/amd64",
        alignerr_cli_version="0.1.0",
        duration_seconds=1.0,
    )

    def fake_grade_workspace(_problem, _workspace, output_dir, _transcript):
        output_dir.mkdir(parents=True)
        (output_dir / "reward.json").write_text('{"score": 1.0}\n')
        (output_dir / "reward-details.json").write_text('{"score": 1.0}\n')
        return 1.0

    monkeypatch.setattr(runner, "run_reference", _mock_run_reference(1.0))
    monkeypatch.setattr(runner, "grade_workspace", fake_grade_workspace)
    # ml resolves to container execution, so the rubric-quality (noop) grade now
    # runs in the task image; stub the in-container grader for the unit test.
    monkeypatch.setattr(runner, "grade_workspace_in_container", fake_grade_workspace)
    monkeypatch.setattr(
        runner,
        "_run_rubric_quality_check",
        lambda *_args, **_kwargs: {"status": "completed", "checks": []},
    )
    problem = HarnessProblem(
        id="demo",
        source_format="problem-dir",
        prompt="demo",
        outputs=[OutputSpec(path="/tmp/output/result.txt")],
        source_problem_dir=problem_dir,
        metadata={
            "difficulty": {
                "task_type": "ml",
                "domain": "scientific_discovery_computational_science",
                "reward_type": "multi_deterministic_rubrics",
            }
        },
    )

    run_harness(problem, runtime="noop", run_dir_base=tmp_path / "runs-noop")
    before = json.loads((problem_dir / ".alignerr" / "build_proof.json").read_text())

    run_harness(problem, runtime="solution", run_dir_base=tmp_path / "runs-solution")
    after = json.loads((problem_dir / ".alignerr" / "build_proof.json").read_text())

    assert after["harness_result"] == before["harness_result"]
    assert "rubric_quality" in after["harness_result"]
    assert after["ground_truth_result"]["score"] == 1.0
    assert after["ground_truth_result"]["runtime"] == "solution"


def test_solution_runtime_refreshes_stale_build_proof(
    monkeypatch, tmp_path: Path
) -> None:
    problem_dir = tmp_path / "problems" / "demo"
    (problem_dir / "solution").mkdir(parents=True)
    (problem_dir / "solution" / "solve.sh").write_text(
        "mkdir -p /tmp/output\nprintf ok > /tmp/output/result.txt\n"
    )
    (problem_dir / "task.toml").write_text("[task]\nname = 'demo'\n")
    write_build_proof(
        problem_dir,
        image_digest="sha256:old",
        base_image_ref="lbx-tasks-base:runtime",
        platform="linux/amd64",
        alignerr_cli_version="0.1.0",
        duration_seconds=1.0,
    )
    (problem_dir / "instruction.md").write_text("changed after proof\n")

    def fake_grade_workspace(_problem, _workspace, output_dir, _transcript):
        output_dir.mkdir(parents=True)
        (output_dir / "reward.json").write_text('{"score": 1.0}\n')
        (output_dir / "reward-details.json").write_text('{"score": 1.0}\n')
        return 1.0

    class FakeTaskValidator:
        def _local_build_proof(self, refreshed_problem_dir):
            write_build_proof(
                refreshed_problem_dir,
                image_digest="sha256:refreshed",
                base_image_ref="lbx-tasks-base:runtime",
                platform="linux/amd64",
                alignerr_cli_version="0.1.0",
                duration_seconds=1.0,
            )
            return StageResult(passed=True, issues=[], duration_ms=0)

    monkeypatch.setattr(reference_module, "grade_workspace", fake_grade_workspace)
    monkeypatch.setattr(runner, "TaskValidator", FakeTaskValidator)
    problem = HarnessProblem(
        id="demo",
        source_format="problem-dir",
        prompt="demo",
        outputs=[OutputSpec(path="/tmp/output/result.txt")],
        source_problem_dir=problem_dir,
        metadata={
            "difficulty": {
                "task_type": "structures",
                "domain": "structural_mechanics",
                "reward_type": "multi_deterministic_rubrics",
            }
        },
    )

    run_harness(problem, runtime="solution", run_dir_base=tmp_path / "runs-solution")

    proof = json.loads((problem_dir / ".alignerr" / "build_proof.json").read_text())
    assert proof["task_dir_sha256"] == grading_inputs_sha256(problem_dir)
    assert proof["image_digest"] == "sha256:refreshed"
    assert proof["ground_truth_result"]["score"] == 1.0


def test_run_auto_qa_only_uses_existing_proof_without_scoring(monkeypatch, tmp_path):
    problem_dir = tmp_path / "problems" / "demo"
    problem_dir.mkdir(parents=True)
    proof_path = problem_dir / ".alignerr" / "build_proof.json"
    proof_path.parent.mkdir()
    proof_path.write_text(
        json.dumps(
            {
                "schema_version": "1.0",
                "task_dir_sha256": "hash",
                "harness_result": {
                    "runtime": "deepagents",
                    "score": 0.25,
                    "subscores": {"criterion": 0.0},
                    "weights": {"criterion": 1.0},
                    "rubric_quality": {"status": "completed", "checks": []},
                },
            }
        )
        + "\n"
    )

    def fail(*_args, **_kwargs):
        raise AssertionError("Auto QA-only path must not run scoring or rubric checks")

    async def fake_auto_qa(*_args, **kwargs):
        assert kwargs["grade_payload"]["score"] == 0.25
        assert kwargs["rubric_quality"]["status"] == "completed"
        return {
            "status": "completed",
            "model": "openai:gpt-5.5",
            "reasoning_effort": "xhigh",
            "overall_assessment": "pass",
            "confidence": "high",
            "checks": [],
        }

    monkeypatch.setattr(reference_module, "grade_workspace", fail)
    monkeypatch.setattr(runner, "run_reference", fail)
    monkeypatch.setattr(runner, "run_claude_code", fail)
    monkeypatch.setattr(runner, "run_deepagents", fail)
    monkeypatch.setattr(runner, "_run_rubric_quality_check", fail)
    monkeypatch.setattr(runner, "run_auto_qa_check", fake_auto_qa)
    monkeypatch.setattr(runner, "print_auto_qa_review", lambda *_args, **_kwargs: None)

    problem = HarnessProblem(
        id="demo",
        source_format="problem-dir",
        prompt="demo",
        outputs=[OutputSpec(path="/tmp/output/result.txt")],
        source_problem_dir=problem_dir,
    )
    output_proof = tmp_path / "autoqa" / "build_proof.json"
    output_json = tmp_path / "autoqa" / "auto_qa.json"

    auto_qa = run_auto_qa_only(
        problem,
        output_proof_path=output_proof,
        output_json=output_json,
        model="openai:gpt-5.5",
        reasoning_effort="xhigh",
        require=True,
    )

    assert auto_qa["status"] == "completed"
    original = json.loads(proof_path.read_text())
    updated = json.loads(output_proof.read_text())
    assert "auto_qa" not in original["harness_result"]
    assert updated["harness_result"]["score"] == 0.25
    assert updated["harness_result"]["rubric_quality"]["status"] == "completed"
    assert updated["harness_result"]["auto_qa"]["overall_assessment"] == "pass"
    assert json.loads(output_json.read_text())["overall_assessment"] == "pass"
