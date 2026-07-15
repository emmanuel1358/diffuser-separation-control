from __future__ import annotations

import json
from pathlib import Path

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
    _cache_has_required_outputs,
    _container_script_prefix,
    _container_script_stop_env_server,
    _container_solve_script,
    _docker_grade_command,
    _docker_solve_command,
    _effective_skip_solve,
    _hidden_env_mode,
    _run_streaming,
    collect_host_info,
    ReferenceRunOptions,
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

    gpu = _problem(metadata={"environment": {"required_resources": "12vcpu+100gib+h100/2"}})
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
    assert resolve_reference_proof_mode(_problem(reference=ReferenceSpec(proof_mode="execute"))) == "execute"
    assert resolve_reference_proof_mode(_problem(reference=ReferenceSpec(proof_mode="artifact"))) == "artifact"
    assert resolve_reference_proof_mode(_problem(metadata={"difficulty": {"task_type": "ml"}})) == "execute"
    assert resolve_reference_proof_mode(_problem(metadata={"difficulty": {"task_type": "mujoco"}})) == "artifact"


def test_reference_cache_and_entrypoint() -> None:
    problem = _problem(reference=ReferenceSpec(cache_dir="custom/cache", entrypoint="train.sh"))
    assert reference_cache_path(problem) == "custom/cache"
    assert solution_script_rel(problem, solution_dir="solution") == "solution/train.sh"


def test_container_script_bootstraps_hidden_env() -> None:
    problem = _problem(metadata={"environment": {"hidden_env": "env"}})
    prefix = _container_script_prefix(problem, skip_solve=False)
    assert "python -m env_server &" in prefix
    assert _container_script_stop_env_server(problem, skip_solve=False)
    assert _hidden_env_mode(problem) == "env"


def test_container_script_skips_env_server_on_skip_solve() -> None:
    problem = _problem(metadata={"environment": {"hidden_env": "env"}})
    prefix = _container_script_prefix(problem, skip_solve=True)
    assert "python -m env_server" not in " ".join(prefix)
    assert _container_script_stop_env_server(problem, skip_solve=True) == []


def test_run_reference_harness_writes_manifest(monkeypatch, tmp_path: Path) -> None:
    problem_dir = tmp_path / "problem"
    (problem_dir / "solution").mkdir(parents=True)
    (problem_dir / "solution" / "solve.sh").write_text("mkdir -p /tmp/output\nprintf ok > /tmp/output/result.txt\n")

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
    assert (problem_dir / ".alignerr" / "reference_cache" / "output" / "result.txt").exists()


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

    result = run_reference_harness(problem, run_dir_base=tmp_path / "runs", skip_solve=True)

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
    (problem_dir / "solution" / "solve.sh").write_text("mkdir -p /tmp/output\nprintf ok > /tmp/output/result.txt\n")

    monkeypatch.setattr(reference_module, "grade_workspace", lambda *_args: 0.99)
    problem = HarnessProblem(
        id="demo",
        source_format="problem-dir",
        prompt="demo",
        outputs=[OutputSpec(path="/tmp/output/result.txt")],
        source_problem_dir=problem_dir,
        reference=ReferenceSpec(execution="host"),
    )

    result = run_reference_harness(problem, run_dir_base=tmp_path / "runs", no_grade=True)

    assert result.score is None
    manifest = json.loads((problem_dir / ".alignerr" / "reference_run" / "latest" / "manifest.json").read_text())
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

    assert "python -m env_server &" in script
    # uid-1000 account is resolved from the image (agent on native, model on
    # mlenvs) rather than hardcoded, so the drop works on both base flavors.
    assert 'su -s /bin/bash "$(getent passwd 1000 | cut -d: -f1)" -c' in script
    assert "cd /host_task && bash solution/solve.sh" in script


def test_container_solve_command_can_disable_network_isolation() -> None:
    cmd = _docker_solve_command("img:tag", Path("/src"), Path("/cache"), "echo hi", network_isolated=False)
    assert "--network" not in cmd


def test_container_grade_command_runs_root_with_host_scorer_mount(tmp_path) -> None:
    scorer_dir = tmp_path / "scorer"
    scorer_dir.mkdir()
    cmd = _docker_grade_command("img:tag", Path("/cache"), scorer_dir, Path("/out"), "echo grade")
    # Grade runs as root (no --user drop) so it can read the root-only truth...
    assert "--user" not in cmd
    # ...and with no network (like the solve phase and Taiga), so a grader or
    # submitted policy cannot download at score time (e.g. from_pretrained),
    # which would make the reference calibration unrealistic.
    assert "--network" in cmd
    assert cmd[cmd.index("--network") + 1] == "none"
    # A native task ships a host scorer/; mount it read-only so editing
    # compute_score.py changes the next score with no image rebuild.
    assert any(part == f"{scorer_dir}:/mcp_server/grader:ro" for part in cmd)
    assert any(part == "/cache:/tmp/output" for part in cmd)


def test_container_grade_command_skips_grader_mount_for_mlenvs(tmp_path) -> None:
    # ML_Envs tasks have no host scorer/; the grader is baked into the image
    # (test_file.py -> /mcp_server/grader/compute_score.py). Mounting a
    # non-existent host path would shadow the baked grader with an empty dir, so
    # the grader mount must be omitted and the baked grader used.
    missing_scorer = tmp_path / "scorer"  # never created
    cmd = _docker_grade_command("img:tag", Path("/cache"), missing_scorer, Path("/out"), "echo grade")
    assert not any("/mcp_server/grader" in part for part in cmd)
    assert any(part == "/cache:/tmp/output" for part in cmd)


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


def test_run_streaming_propagates_nonzero_returncode() -> None:
    rc, _ = _run_streaming(["bash", "-c", "exit 3"], timeout=30, label="unit")
    assert rc == 3


def test_collect_host_info_reports_hardware() -> None:
    info = collect_host_info()
    assert "platform" in info
    assert "cpu_count" in info
    assert "mem_total_bytes" in info
    assert isinstance(info["gpus"], list)


def test_manifest_records_host_hardware(monkeypatch, tmp_path: Path) -> None:
    problem_dir = tmp_path / "problem"
    (problem_dir / "solution").mkdir(parents=True)
    (problem_dir / "solution" / "solve.sh").write_text("mkdir -p /tmp/output\nprintf ok > /tmp/output/result.txt\n")

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
    manifest = json.loads((problem_dir / ".alignerr" / "reference_run" / "latest" / "manifest.json").read_text())
    assert "host" in manifest
    assert "platform" in manifest["host"]
    assert "gpus" in manifest["host"]


def test_host_run_warns_and_marks_manifest_no_isolation(monkeypatch, capsys, tmp_path: Path) -> None:
    """A non-prove host run prints the no-isolation warning and the manifest
    records execution=host with an explicit no-isolation / not-Taiga-faithful /
    not-proof-authoritative marker (HAR-3)."""
    problem_dir = tmp_path / "problem"
    (problem_dir / "solution").mkdir(parents=True)
    (problem_dir / "solution" / "solve.sh").write_text("mkdir -p /tmp/output\nprintf ok > /tmp/output/result.txt\n")
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

    manifest = json.loads((problem_dir / ".alignerr" / "reference_run" / "latest" / "manifest.json").read_text())
    assert manifest["execution"] == "host"
    assert manifest["isolation"] == "none"
    assert manifest["taiga_faithful"] is False
    assert manifest["proof_authoritative"] is False
    assert manifest["requested_execution"] == "host"
