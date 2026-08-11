from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pytest
from lbx_rl_tasks_harness.models import (
    GroundTruthSpec,
    HarnessProblem,
    OutputSpec,
    ReferenceSpec,
)
from lbx_rl_tasks_harness.reference_config import (
    reference_cache_path,
    reference_isolation_requirements,
    resolve_reference_execution,
    resolve_reference_proof_mode,
    solution_script_rel,
)
from lbx_rl_tasks_harness.runner import run_reference_harness
from lbx_rl_tasks_harness.runtimes import reference as reference_module
from lbx_rl_tasks_harness.runtimes.reference import (
    ReferenceRunOptions,
    _cache_has_required_outputs,
    _container_measure_script,
    _container_script_prefix,
    _container_script_stop_env_server,
    _container_solve_script,
    _docker_grade_command,
    _docker_solve_command,
    _effective_skip_solve,
    _hidden_env_mode,
    _run_streaming,
    collect_host_info,
)


def _problem(**overrides) -> HarnessProblem:
    defaults = {
        "id": "demo",
        "source_format": "problem-dir",
        "prompt": "demo",
        "outputs": [OutputSpec(path="/tmp/output/result.txt")],
        "source_problem_dir": Path("/tmp/demo"),
    }
    defaults.update(overrides)
    return HarnessProblem(**defaults)


def test_resolve_reference_execution_explicit_and_auto() -> None:
    host = _problem(reference=ReferenceSpec(execution="host"))
    assert resolve_reference_execution(host) == "host"

    container = _problem(reference=ReferenceSpec(execution="container"))
    assert resolve_reference_execution(container) == "container"

    in_container = _problem(ground_truth=GroundTruthSpec(in_container=True))
    assert resolve_reference_execution(in_container) == "container"

    gpu = _problem(
        metadata={"environment": {"required_resources": "12vcpu+100gib+h100/2"}}
    )
    assert resolve_reference_execution(gpu) == "container"

    ml = _problem(metadata={"difficulty": {"task_type": "ml"}})
    assert resolve_reference_execution(ml) == "container"

    mujoco = _problem(metadata={"difficulty": {"task_type": "mujoco"}})
    assert resolve_reference_execution(mujoco) == "host"


def test_prove_mode_honors_in_container_over_reference_host() -> None:
    problem = _problem(
        reference=ReferenceSpec(execution="host"),
        ground_truth=GroundTruthSpec(in_container=True),
    )
    # HAR-3: an explicit host request on an isolation-dependent task is upgraded
    # to the faithful container path on the iterate path too (not just prove).
    assert resolve_reference_execution(problem) == "container"
    assert resolve_reference_execution(problem, mode="prove") == "container"


def test_explicit_host_is_upgraded_when_task_needs_isolation() -> None:
    # hidden-env task: host cannot bootstrap the env_server, so an explicit host
    # request must be upgraded to the container path.
    hidden = _problem(
        reference=ReferenceSpec(execution="host"),
        metadata={"environment": {"hidden_env": "env"}},
    )
    assert reference_isolation_requirements(hidden)
    assert resolve_reference_execution(hidden) == "container"

    # in_container grader: same.
    in_container = _problem(
        reference=ReferenceSpec(execution="host"),
        ground_truth=GroundTruthSpec(in_container=True),
    )
    assert resolve_reference_execution(in_container) == "container"


def test_explicit_host_allowed_without_isolation_needs() -> None:
    # A plain task with no held-out truth / hidden-env / in_container keeps host.
    plain = _problem(reference=ReferenceSpec(execution="host"))
    assert reference_isolation_requirements(plain) == []
    assert resolve_reference_execution(plain) == "host"


def test_private_truth_task_host_upgrades_but_prove_unaffected(tmp_path: Path) -> None:
    # A CPU non-ML task that commits held-out truth under scorer/data: the
    # iterate path upgrades an explicit host request to container, while prove
    # still re-resolves via the auto path (unchanged) -- proving prove ignores
    # the upgrade as well as the host override.
    problem_dir = tmp_path / "problem"
    (problem_dir / "scorer" / "data").mkdir(parents=True)
    (problem_dir / "scorer" / "data" / "labels.json").write_text("[1,2,3]\n")
    problem = HarnessProblem(
        id="demo",
        source_format="problem-dir",
        prompt="demo",
        outputs=[OutputSpec(path="/tmp/output/result.txt")],
        source_problem_dir=problem_dir,
        grader_dir=problem_dir / "scorer",
        private_dir=problem_dir / "scorer" / "data",
        reference=ReferenceSpec(execution="host"),
    )
    assert reference_isolation_requirements(problem)
    assert resolve_reference_execution(problem) == "container"
    # prove re-resolves via auto (CPU non-ML -> host); the upgrade never applies.
    assert resolve_reference_execution(problem, mode="prove") == "host"


def test_resolve_reference_proof_mode() -> None:
    assert (
        resolve_reference_proof_mode(
            _problem(reference=ReferenceSpec(proof_mode="execute"))
        )
        == "execute"
    )
    assert (
        resolve_reference_proof_mode(
            _problem(reference=ReferenceSpec(proof_mode="artifact"))
        )
        == "artifact"
    )
    assert (
        resolve_reference_proof_mode(
            _problem(metadata={"difficulty": {"task_type": "ml"}})
        )
        == "execute"
    )
    assert (
        resolve_reference_proof_mode(
            _problem(metadata={"difficulty": {"task_type": "mujoco"}})
        )
        == "artifact"
    )


def test_reference_cache_and_entrypoint() -> None:
    problem = _problem(
        reference=ReferenceSpec(cache_dir="custom/cache", entrypoint="train.sh")
    )
    assert reference_cache_path(problem) == "custom/cache"
    assert solution_script_rel(problem, solution_dir="solution") == "solution/train.sh"


def test_native_ml_prefers_manifest_inference_entrypoint(tmp_path: Path) -> None:
    import hashlib
    import json

    problem_dir = tmp_path / "problem"
    strategy = problem_dir / "solution"
    strategy.mkdir(parents=True)
    (problem_dir / "task.toml").write_text(
        '[difficulty]\ntask_type = "ml"\nreward_type = "continuous_scoring_function"\n'
    )
    (problem_dir / "metadata.json").write_text("{}\n")
    train_csv = problem_dir / "train.csv"
    train_csv.write_text("x,y\n1,1\n")
    (strategy / "train.py").write_text("print('train')\n")
    (strategy / "solve.sh").write_text("#!/bin/bash\ncp model.json /tmp/output/\n")
    model = strategy / "model.json"
    model.write_text("{}\n")
    (strategy / "model.manifest.json").write_text(
        json.dumps(
            {
                "schema_version": "1.0",
                "role": "reference",
                "training_entrypoint": "train.py",
                "inference_entrypoint": "solve.sh",
                "seed": 1,
                "public_training_data": {
                    "path": "../train.csv",
                    "sha256": hashlib.sha256(train_csv.read_bytes()).hexdigest(),
                },
                "artifacts": [
                    {
                        "path": "model.json",
                        "sha256": hashlib.sha256(model.read_bytes()).hexdigest(),
                    }
                ],
            }
        )
        + "\n"
    )
    problem = _problem(
        source_problem_dir=problem_dir,
        reference=ReferenceSpec(entrypoint="train.sh"),
        metadata={"difficulty": {"task_type": "ml"}},
    )
    assert solution_script_rel(problem, solution_dir="solution") == "solution/solve.sh"


def test_continuous_ml_reference_does_not_fall_back_on_invalid_manifest(
    tmp_path: Path,
) -> None:
    problem_dir = tmp_path / "problem"
    strategy = problem_dir / "solution"
    strategy.mkdir(parents=True)
    (problem_dir / "task.toml").write_text(
        '[difficulty]\ntask_type = "ml"\n'
        'reward_type = "continuous_scoring_function"\n'
    )
    (problem_dir / "metadata.json").write_text("{}\n")
    (strategy / "solve.sh").write_text("#!/bin/bash\nexit 0\n")
    (strategy / "strategy.manifest.json").write_text("{not-json\n")
    problem = _problem(
        source_problem_dir=problem_dir,
        reference=ReferenceSpec(entrypoint="solve.sh"),
        metadata={
            "difficulty": {
                "task_type": "ml",
                "reward_type": "continuous_scoring_function",
            }
        },
    )

    with pytest.raises(ValueError, match="invalid reference strategy manifest"):
        solution_script_rel(problem, solution_dir="solution")


def test_custom_naive_path_uses_naive_manifest_role(tmp_path: Path) -> None:
    import hashlib

    problem_dir = tmp_path / "problem"
    strategy = problem_dir / "heuristic"
    strategy.mkdir(parents=True)
    (problem_dir / "task.toml").write_text(
        '[difficulty]\ntask_type = "ml"\n'
        'reward_type = "continuous_scoring_function"\n'
    )
    (problem_dir / "metadata.json").write_text("{}\n")
    policy = strategy / "policy.py"
    policy.write_text("def load_policy(): return object()\n")
    (strategy / "solve.sh").write_text(
        "#!/bin/bash\ncp policy.py /tmp/output/policy.py\n"
    )
    (strategy / "strategy.manifest.json").write_text(
        json.dumps(
            {
                "schema_version": "1.0",
                "role": "naive",
                "kind": "committed_artifact",
                "inference_entrypoint": "solve.sh",
                "artifacts": [
                    {
                        "path": "policy.py",
                        "sha256": hashlib.sha256(policy.read_bytes()).hexdigest(),
                    }
                ],
            }
        )
        + "\n"
    )
    problem = _problem(
        source_problem_dir=problem_dir,
        reference=ReferenceSpec(entrypoint="solve.sh"),
        metadata={
            "difficulty": {
                "task_type": "ml",
                "reward_type": "continuous_scoring_function",
            }
        },
    )

    assert (
        solution_script_rel(problem, solution_dir="heuristic") == "heuristic/solve.sh"
    )


def test_container_script_bootstraps_hidden_env() -> None:
    problem = _problem(metadata={"environment": {"hidden_env": "env"}})
    prefix = _container_script_prefix(problem, skip_solve=False)
    assert any("python -P -m env_server &" in line for line in prefix)
    assert "cd /" in prefix
    assert any("env -u PYTHONPATH -u PYTHONHOME" in line for line in prefix)
    assert any("PATH=/opt/lbx-runtime/.venv/bin:" in line for line in prefix)
    assert _container_script_stop_env_server(problem, skip_solve=False)
    assert _hidden_env_mode(problem) == "env"


def test_raw_measurement_bootstraps_hidden_env() -> None:
    problem = _problem(metadata={"environment": {"hidden_env": "hybrid"}})
    script = _container_measure_script(problem)

    assert "python -P -m env_server &" in script
    assert "grader_runner.raw_worker" in script
    assert 'kill "$ENV_PID"' in script


def test_container_script_skips_env_server_on_skip_solve() -> None:
    problem = _problem(metadata={"environment": {"hidden_env": "env"}})
    prefix = _container_script_prefix(problem, skip_solve=True)
    assert "env_server" not in " ".join(prefix)
    assert _container_script_stop_env_server(problem, skip_solve=True) == []


def test_run_reference_harness_writes_manifest(monkeypatch, tmp_path: Path) -> None:
    problem_dir = tmp_path / "problem"
    (problem_dir / "solution").mkdir(parents=True)
    (problem_dir / "solution" / "solve.sh").write_text(
        "mkdir -p /tmp/output\nprintf ok > /tmp/output/result.txt\n"
    )

    def fake_grade(_problem, _workspace, output_dir, _transcript):
        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / "reward.json").write_text('{"score": 0.75}\n')
        (output_dir / "reward-details.json").write_text('{"score": 0.75}\n')
        return 0.75

    monkeypatch.setattr(reference_module, "grade_workspace", fake_grade)
    problem = HarnessProblem(
        id="demo",
        source_format="problem-dir",
        prompt="demo",
        outputs=[OutputSpec(path="/tmp/output/result.txt")],
        source_problem_dir=problem_dir,
        reference=ReferenceSpec(execution="host"),
    )

    result = run_reference_harness(problem, run_dir_base=tmp_path / "runs")
    manifest = problem_dir / ".alignerr" / "reference_run" / "latest" / "manifest.json"

    assert result.score == 0.75
    assert manifest.exists()
    payload = json.loads(manifest.read_text())
    assert payload["mode"] == "iterate"
    assert payload["execution"] == "host"
    assert payload["score"] == 0.75
    assert (
        problem_dir / ".alignerr" / "reference_cache" / "output" / "result.txt"
    ).exists()


def test_skip_solve_grades_cache(monkeypatch, tmp_path: Path) -> None:
    problem_dir = tmp_path / "problem"
    cache = problem_dir / ".alignerr" / "reference_cache" / "output"
    cache.mkdir(parents=True)
    (cache / "result.txt").write_text("cached\n")
    (problem_dir / "solution").mkdir(parents=True)
    (problem_dir / "solution" / "solve.sh").write_text("exit 1\n")

    def fake_grade(_problem, _workspace, output_dir, _transcript):
        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / "reward.json").write_text('{"score": 0.42}\n')
        return 0.42

    monkeypatch.setattr(reference_module, "grade_workspace", fake_grade)
    problem = HarnessProblem(
        id="demo",
        source_format="problem-dir",
        prompt="demo",
        outputs=[OutputSpec(path="/tmp/output/result.txt")],
        source_problem_dir=problem_dir,
        reference=ReferenceSpec(execution="host"),
    )

    result = run_reference_harness(
        problem, run_dir_base=tmp_path / "runs", skip_solve=True
    )

    assert result.score == 0.42
    assert (result.workspace / "result.txt").read_text() == "cached\n"


def test_artifact_mode_skips_solve_when_cache_warm(monkeypatch, tmp_path: Path) -> None:
    problem_dir = tmp_path / "problem"
    cache = problem_dir / ".alignerr" / "reference_cache" / "output"
    cache.mkdir(parents=True)
    (cache / "result.txt").write_text("cached\n")
    (problem_dir / "solution").mkdir(parents=True)
    (problem_dir / "solution" / "solve.sh").write_text("exit 1\n")

    def fake_grade(_problem, _workspace, output_dir, _transcript):
        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / "reward.json").write_text('{"score": 0.42}\n')
        return 0.42

    monkeypatch.setattr(reference_module, "grade_workspace", fake_grade)
    problem = HarnessProblem(
        id="demo",
        source_format="problem-dir",
        prompt="demo",
        outputs=[OutputSpec(path="/tmp/output/result.txt")],
        source_problem_dir=problem_dir,
        reference=ReferenceSpec(execution="host", proof_mode="artifact"),
    )

    result = run_reference_harness(problem, run_dir_base=tmp_path / "runs")

    assert result.score == 0.42
    assert (result.workspace / "result.txt").read_text() == "cached\n"


def test_effective_skip_solve_helpers(tmp_path: Path) -> None:
    cache = tmp_path / "cache"
    cache.mkdir()
    problem = _problem(
        outputs=[OutputSpec(path="/tmp/output/result.txt")],
        reference=ReferenceSpec(proof_mode="artifact"),
    )
    options = ReferenceRunOptions()

    assert not _cache_has_required_outputs(problem, cache)
    assert not _effective_skip_solve(problem, options, cache)

    (cache / "result.txt").write_text("ok\n")
    assert _cache_has_required_outputs(problem, cache)
    assert _effective_skip_solve(problem, options, cache)

    execute = _problem(reference=ReferenceSpec(proof_mode="execute"))
    assert not _effective_skip_solve(execute, options, cache)

    prove = ReferenceRunOptions(mode="prove")
    assert not _effective_skip_solve(problem, prove, cache)

    optional_only = _problem(
        outputs=[OutputSpec(path="/tmp/output/result.txt", required=False)],
        reference=ReferenceSpec(proof_mode="artifact"),
    )
    assert not _cache_has_required_outputs(optional_only, cache)
    assert not _effective_skip_solve(optional_only, options, cache)


def test_no_grade_leaves_score_unset(monkeypatch, tmp_path: Path) -> None:
    problem_dir = tmp_path / "problem"
    (problem_dir / "solution").mkdir(parents=True)
    (problem_dir / "solution" / "solve.sh").write_text(
        "mkdir -p /tmp/output\nprintf ok > /tmp/output/result.txt\n"
    )

    monkeypatch.setattr(reference_module, "grade_workspace", lambda *_args: 0.99)
    problem = HarnessProblem(
        id="demo",
        source_format="problem-dir",
        prompt="demo",
        outputs=[OutputSpec(path="/tmp/output/result.txt")],
        source_problem_dir=problem_dir,
        reference=ReferenceSpec(execution="host"),
    )

    result = run_reference_harness(
        problem, run_dir_base=tmp_path / "runs", no_grade=True
    )

    assert result.score is None
    manifest = json.loads(
        (
            problem_dir / ".alignerr" / "reference_run" / "latest" / "manifest.json"
        ).read_text()
    )
    assert manifest["no_grade"] is True
    assert manifest["score"] is None


def test_container_solve_command_drops_privilege_and_isolates_network() -> None:
    cmd = _docker_solve_command("img:tag", Path("/src"), Path("/cache"), "echo hi")
    # The container starts as root so hidden-env setup can read root-only fixtures;
    # the solve script itself drops to the agent user.
    assert "--user" not in cmd
    assert "--network" in cmd
    assert cmd[cmd.index("--network") + 1] == "none"
    # The host scorer is NOT mounted in the solve phase, so the answer key
    # (/mcp_server/data, baked root-only) is unreachable as uid 1000.
    assert not any("/mcp_server/grader" in part for part in cmd)
    assert any(part == "/cache:/tmp/output" for part in cmd)


def test_container_solve_script_runs_env_server_as_root_then_solve_as_agent() -> None:
    problem = _problem(metadata={"environment": {"hidden_env": "env"}})

    script = _container_solve_script(problem, sol_rel="solution/solve.sh", render=False)

    assert "python -P -m env_server &" in script
    # uid-1000 account is resolved from the image rather than hardcoded, so the
    # drop works whatever the base flavor names that account.
    assert 'su -s /bin/bash "$(getent passwd 1000 | cut -d: -f1)" -c' in script
    assert "cd /host_task && bash solution/solve.sh" in script


def test_container_solve_command_can_disable_network_isolation() -> None:
    cmd = _docker_solve_command(
        "img:tag", Path("/src"), Path("/cache"), "echo hi", network_isolated=False
    )
    assert "--network" not in cmd


def test_container_grade_command_runs_root_with_host_scorer_mount(tmp_path) -> None:
    scorer_dir = tmp_path / "scorer"
    scorer_dir.mkdir()
    cmd = _docker_grade_command(
        "img:tag",
        Path("/cache"),
        scorer_dir,
        Path("/out"),
        "echo grade",
        verifier_capabilities=("SYS_PTRACE",),
    )
    # Grade runs as root (no --user drop) so it can read the root-only truth...
    assert "--user" not in cmd
    # ...and with no network (like the solve phase and Taiga), so a grader or
    # submitted policy cannot download at score time (e.g. from_pretrained),
    # which would make the reference calibration unrealistic.
    assert "--network" in cmd
    assert cmd[cmd.index("--network") + 1] == "none"
    assert cmd[cmd.index("--cap-add") + 1] == "SYS_PTRACE"
    # A native task ships a host scorer/; mount it read-only so editing
    # compute_score.py changes the next score with no image rebuild.
    assert any(part == f"{scorer_dir}:/mcp_server/grader:ro" for part in cmd)
    assert any(part == "/cache:/tmp/output" for part in cmd)


def test_container_grade_command_skips_grader_mount_when_host_scorer_missing(
    tmp_path,
) -> None:
    # Mounting a non-existent host path would shadow the image's baked
    # /mcp_server/grader with an empty dir, so the mount must be omitted and the
    # baked grader used instead.
    missing_scorer = tmp_path / "scorer"  # never created
    cmd = _docker_grade_command(
        "img:tag", Path("/cache"), missing_scorer, Path("/out"), "echo grade"
    )
    assert not any("/mcp_server/grader" in part for part in cmd)
    assert any(part == "/cache:/tmp/output" for part in cmd)


def test_container_grade_command_mounts_generated_calibration_lock(tmp_path) -> None:
    lock = tmp_path / "calibration.lock.json"
    lock.write_text("{}\n")
    cmd = _docker_grade_command(
        "img:tag",
        Path("/cache"),
        tmp_path / "missing-scorer",
        Path("/out"),
        "echo grade",
        calibration_lock=lock,
    )
    assert any(
        part == f"{lock}:/mcp_server/calibration/calibration.lock.json:ro"
        for part in cmd
    )


def test_run_streaming_tees_and_captures(capsys) -> None:
    rc, captured = _run_streaming(
        ["bash", "-c", "echo out; echo err 1>&2"],
        timeout=30,
        label="unit",
    )
    assert rc == 0
    # Combined stream is both captured (for the transcript) and emitted live.
    assert "out" in captured
    assert "err" in captured
    assert "[unit]" in captured
    streamed = capsys.readouterr().out
    assert "out" in streamed
    assert "err" in streamed


def test_grade_failure_detail_reads_the_shape_to_dict_actually_writes(
    tmp_path: Path,
) -> None:
    """Grade.to_dict() has no top-level criterion_logs: the failure only reaches
    metadata.grading_errors, metadata.rubric_breakdown and structured_subscores,
    and reading just the first shape left CI printing an empty stderr tail."""
    verifier = tmp_path / "verifier"
    verifier.mkdir()
    (verifier / "reward-details.json").write_text(
        json.dumps(
            {
                "score": 0.0,
                "structured_subscores": [
                    {"name": "run_grader", "reasoning": "[grader_launch_failed] EROFS"}
                ],
                "metadata": {
                    "grading_errors": [
                        {
                            "criterion": "run_grader",
                            "error_type": "grader_launch_failed",
                            "error_message": "could not reset root ownership",
                        }
                    ],
                    "traceback": "Traceback (most recent call last):\n  OSError",
                },
            }
        )
    )

    detail = reference_module._grade_failure_detail(verifier)

    assert "grader_launch_failed: could not reset root ownership" in detail
    assert "grader traceback:" in detail
    assert "OSError" in detail


def test_grade_failure_detail_is_empty_for_a_successful_grade(tmp_path: Path) -> None:
    """Only errored criteria carry the "[error_type] ..." reasoning prefix; a
    passing criterion's reasoning is ordinary judge prose and is not an error."""
    verifier = tmp_path / "verifier"
    verifier.mkdir()
    (verifier / "reward-details.json").write_text(
        json.dumps(
            {
                "score": 1.0,
                "structured_subscores": [
                    {"name": "accuracy", "reasoning": ""},
                    {
                        "name": "style",
                        "reasoning": "The solution matches the expected output.",
                    },
                ],
                "metadata": {"rubric_breakdown": [{"criterion": "accuracy"}]},
            }
        )
    )

    assert reference_module._grade_failure_detail(verifier) == ""


def test_run_streaming_propagates_nonzero_returncode() -> None:
    rc, _ = _run_streaming(["bash", "-c", "exit 3"], timeout=30, label="unit")
    assert rc == 3


def test_collect_host_info_reports_hardware() -> None:
    info = collect_host_info()
    assert "platform" in info
    assert "cpu_count" in info
    assert "mem_total_bytes" in info
    assert isinstance(info["gpus"], list)


def _fake_nvidia_smi(monkeypatch, stdout: str) -> None:
    """Stub only the nvidia-smi call. ``platform.platform()`` also shells out,
    so anything else has to reach the real subprocess.run."""
    real_run = subprocess.run

    def fake_run(cmd, **kwargs):
        if not (cmd and cmd[0] == "nvidia-smi"):
            return real_run(cmd, **kwargs)
        assert "--query-gpu=name,memory.total,driver_version,compute_cap" in cmd
        return subprocess.CompletedProcess(cmd, 0, stdout=stdout, stderr="")

    monkeypatch.setattr(
        reference_module.shutil, "which", lambda name: f"/usr/bin/{name}"
    )
    monkeypatch.setattr(reference_module.subprocess, "run", fake_run)


def test_collect_host_info_records_gpu_driver_version(monkeypatch) -> None:
    """The driver version decides which CUDA major a base image may target,
    so a reviewer has to be able to read it off a reference run."""
    _fake_nvidia_smi(monkeypatch, "NVIDIA H100 80GB HBM3, 81559, 550.54.15, 9.0\n")

    assert collect_host_info()["gpus"] == [
        {
            "name": "NVIDIA H100 80GB HBM3",
            "memory_total_mib": 81559,
            "driver_version": "550.54.15",
            "compute_cap": "9.0",
        }
    ]


def test_collect_host_info_tolerates_short_nvidia_smi_output(monkeypatch) -> None:
    _fake_nvidia_smi(monkeypatch, "Some GPU, 4096\n")

    assert collect_host_info()["gpus"] == [
        {
            "name": "Some GPU",
            "memory_total_mib": 4096,
            "driver_version": None,
            "compute_cap": None,
        }
    ]


def test_manifest_records_host_hardware(monkeypatch, tmp_path: Path) -> None:
    problem_dir = tmp_path / "problem"
    (problem_dir / "solution").mkdir(parents=True)
    (problem_dir / "solution" / "solve.sh").write_text(
        "mkdir -p /tmp/output\nprintf ok > /tmp/output/result.txt\n"
    )

    monkeypatch.setattr(reference_module, "grade_workspace", lambda *_a: 1.0)
    problem = HarnessProblem(
        id="demo",
        source_format="problem-dir",
        prompt="demo",
        outputs=[OutputSpec(path="/tmp/output/result.txt")],
        source_problem_dir=problem_dir,
        reference=ReferenceSpec(execution="host"),
    )

    run_reference_harness(problem, run_dir_base=tmp_path / "runs")
    manifest = json.loads(
        (
            problem_dir / ".alignerr" / "reference_run" / "latest" / "manifest.json"
        ).read_text()
    )
    assert "host" in manifest
    assert "platform" in manifest["host"]
    assert "gpus" in manifest["host"]


def test_host_run_warns_and_marks_manifest_no_isolation(
    monkeypatch, capsys, tmp_path: Path
) -> None:
    """A non-prove host run prints the no-isolation warning and the manifest
    records execution=host with an explicit no-isolation / not-Taiga-faithful /
    not-proof-authoritative marker (HAR-3)."""
    problem_dir = tmp_path / "problem"
    (problem_dir / "solution").mkdir(parents=True)
    (problem_dir / "solution" / "solve.sh").write_text(
        "mkdir -p /tmp/output\nprintf ok > /tmp/output/result.txt\n"
    )
    monkeypatch.setattr(reference_module, "grade_workspace", lambda *_a: 1.0)
    problem = HarnessProblem(
        id="demo",
        source_format="problem-dir",
        prompt="demo",
        outputs=[OutputSpec(path="/tmp/output/result.txt")],
        source_problem_dir=problem_dir,
        reference=ReferenceSpec(execution="host"),
    )

    run_reference_harness(problem, run_dir_base=tmp_path / "runs")

    out = capsys.readouterr().out
    assert "HOST path" in out
    assert "NOT Taiga-faithful" in out

    manifest = json.loads(
        (
            problem_dir / ".alignerr" / "reference_run" / "latest" / "manifest.json"
        ).read_text()
    )
    assert manifest["execution"] == "host"
    assert manifest["isolation"] == "none"
    assert manifest["taiga_faithful"] is False
    assert manifest["proof_authoritative"] is False
    assert manifest["requested_execution"] == "host"
