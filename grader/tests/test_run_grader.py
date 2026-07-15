"""End-to-end tests for `grader/run_grader.py`.

This is the regression suite that protects the customer Harbor flow.
Each test invokes `run_grader.main(...)` against a real example scorer and
asserts both `reward.json` (Harbor headline) and `reward-details.json`
(full Grade) shapes.
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest

from grader_runner import run_grader, worker
from grader_runner.run_grader import main


def _run_solve(
    solve_path: Path, workspace: Path, public_data: Path | None = None
) -> int:
    """Execute `solve.sh` after rewriting container paths for host execution."""
    src = solve_path.read_text()
    src = src.replace("/tmp/output", str(workspace))
    if public_data is not None:
        src = src.replace("/data/", str(public_data) + "/")
    completed = subprocess.run(
        ["bash", "-c", src],
        cwd=workspace,
        text=True,
        capture_output=True,
    )
    return completed.returncode


def _invoke_runner(
    workspace: Path,
    grader_dir: Path,
    output_dir: Path,
    *,
    private_dir: Path | None = None,
    timeout: float | None = None,
) -> int:
    args = [
        "--workspace",
        str(workspace),
        "--grader-dir",
        str(grader_dir),
        "--output-dir",
        str(output_dir),
    ]
    if private_dir is not None:
        args.extend(["--private-dir", str(private_dir)])
    if timeout is not None:
        args.extend(["--timeout", str(timeout)])
    return main(args)


# ── mujoco-pendulum (RubricBuilder + per-criterion subscores) ──


def test_mujoco_pendulum_reference_scores_one(
    template_examples: Path, workspace: Path, output_dir: Path
) -> None:
    pytest.importorskip("mujoco")
    pytest.importorskip("numpy")

    ex = template_examples / "mujoco-pendulum"
    rc = _run_solve(ex / "solution" / "solve.sh", workspace)
    assert rc == 0

    exit_code = _invoke_runner(
        workspace, ex / "scorer", output_dir, private_dir=ex / "scorer" / "data"
    )
    assert exit_code == 0

    reward = json.loads((output_dir / "reward.json").read_text())
    assert reward["score"] == 1.0
    # Per-criterion flat keys use display descriptions for Boreal/Harbor UI.
    assert "MJCF parses and MuJoCo compiles it without error" in reward
    assert "5s rollout from qpos=pi/2 stays in [-pi, pi]" in reward

    details = json.loads((output_dir / "reward-details.json").read_text())
    assert details["score"] == 1.0
    assert len(details["structured_subscores"]) == 10
    compiled = next(s for s in details["structured_subscores"] if s["id"] == "compiled")
    assert compiled["name"] == "MJCF parses and MuJoCo compiles it without error"
    assert "com_length" not in details["metadata"]
    assert "moving_mass" not in details["metadata"]
    assert "pass_threshold" not in details["metadata"]
    # Round-tripped from RubricBuilder -> Grade.to_dict -> normalize -> Grade.to_dict
    assert details["metadata"]["return_shape"] == "rubric_grade"


def test_mujoco_pendulum_naive_baseline_partial_score(
    template_examples: Path, workspace: Path, output_dir: Path
) -> None:
    pytest.importorskip("mujoco")
    pytest.importorskip("numpy")

    ex = template_examples / "mujoco-pendulum"
    rc = _run_solve(ex / "baselines" / "naive.sh", workspace)
    assert rc == 0

    _invoke_runner(
        workspace, ex / "scorer", output_dir, private_dir=ex / "scorer" / "data"
    )
    reward = json.loads((output_dir / "reward.json").read_text())
    assert 0.0 < reward["score"] < 1.0


def test_reward_txt_is_single_float(
    template_examples: Path, workspace: Path, output_dir: Path
) -> None:
    pytest.importorskip("mujoco")
    pytest.importorskip("numpy")

    ex = template_examples / "mujoco-pendulum"
    _run_solve(ex / "solution" / "solve.sh", workspace)
    _invoke_runner(
        workspace, ex / "scorer", output_dir, private_dir=ex / "scorer" / "data"
    )
    txt = (output_dir / "reward.txt").read_text().strip()
    assert float(txt) == 1.0


def test_runner_accepts_rubric_builder_return(
    workspace: Path, output_dir: Path, tmp_path: Path
) -> None:
    grader_dir = tmp_path / "rubric_builder"
    grader_dir.mkdir()
    (grader_dir / "compute_score.py").write_text(
        "from grading import RubricBuilder\n\n"
        "def compute_score(workspace, trajectory, private):\n"
        "    rb = RubricBuilder(workspace=workspace, trajectory=trajectory, private=private)\n"
        "    @rb.criterion(id='ok', weight=1.0, description='Criterion passes')\n"
        "    def _():\n"
        "        return True\n"
        "    return rb\n"
    )

    exit_code = _invoke_runner(workspace, grader_dir, output_dir)

    assert exit_code == 0
    reward = json.loads((output_dir / "reward.json").read_text())
    assert reward["score"] == 1.0
    details = json.loads((output_dir / "reward-details.json").read_text())
    assert details["score"] == 1.0
    assert details["structured_subscores"][0]["id"] == "ok"


# ── Failure modes ──


def test_runner_handles_missing_grader_gracefully(
    workspace: Path, output_dir: Path, tmp_path: Path
) -> None:
    grader_dir = tmp_path / "no_grader"
    grader_dir.mkdir()
    exit_code = _invoke_runner(workspace, grader_dir, output_dir)
    assert exit_code == 1

    reward = json.loads((output_dir / "reward.json").read_text())
    assert reward["score"] == 0.0
    details = json.loads((output_dir / "reward-details.json").read_text())
    errs = details["metadata"]["grading_errors"]
    assert errs[0]["error_type"] == "grader_import_failed"


def test_runner_handles_grader_runtime_error(
    workspace: Path, output_dir: Path, tmp_path: Path
) -> None:
    grader_dir = tmp_path / "broken"
    grader_dir.mkdir()
    (grader_dir / "compute_score.py").write_text(
        "def compute_score(workspace, trajectory, private):\n"
        "    raise RuntimeError('grader is broken')\n"
    )
    exit_code = _invoke_runner(workspace, grader_dir, output_dir)
    assert exit_code == 1

    details = json.loads((output_dir / "reward-details.json").read_text())
    assert (
        details["metadata"]["grading_errors"][0]["error_type"] == "grader_runtime_error"
    )


def test_runner_handles_unsupported_return_type(
    workspace: Path, output_dir: Path, tmp_path: Path
) -> None:
    grader_dir = tmp_path / "weird"
    grader_dir.mkdir()
    (grader_dir / "compute_score.py").write_text(
        "def compute_score(workspace, trajectory, private):\n"
        "    return 'not a number'\n"
    )
    exit_code = _invoke_runner(workspace, grader_dir, output_dir)
    assert exit_code == 1

    details = json.loads((output_dir / "reward-details.json").read_text())
    assert (
        details["metadata"]["grading_errors"][0]["error_type"]
        == "grader_return_unsupported"
    )
    assert details["env_internal_failure"] is True


def test_runner_keeps_agent_fault_as_score_zero(
    workspace: Path, output_dir: Path, tmp_path: Path
) -> None:
    grader_dir = tmp_path / "agent_fault"
    grader_dir.mkdir()
    (grader_dir / "compute_score.py").write_text(
        "from grading import AgentFault\n\n"
        "def compute_score(workspace, trajectory, private):\n"
        "    raise AgentFault('missing submission')\n"
    )

    exit_code = _invoke_runner(workspace, grader_dir, output_dir)

    assert exit_code == 0
    details = json.loads((output_dir / "reward-details.json").read_text())
    assert details["score"] == 0.0
    assert details["env_internal_failure"] is False
    assert details["metadata"]["agent_fault"] == "missing submission"


def test_runner_marks_non_finite_return_as_env_internal_failure(
    workspace: Path, output_dir: Path, tmp_path: Path
) -> None:
    grader_dir = tmp_path / "nan_score"
    grader_dir.mkdir()
    (grader_dir / "compute_score.py").write_text(
        "import math\n\n"
        "def compute_score(workspace, trajectory, private):\n"
        "    return math.nan\n"
    )

    exit_code = _invoke_runner(workspace, grader_dir, output_dir)

    assert exit_code == 1
    details = json.loads((output_dir / "reward-details.json").read_text())
    assert details["score"] == 0.0
    assert details["env_internal_failure"] is True
    assert (
        details["metadata"]["grading_errors"][0]["error_type"]
        == "grader_return_unsupported"
    )


def test_worker_pid_namespace_skips_non_posix_without_geteuid(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(worker.os, "name", "nt")
    monkeypatch.setattr(
        worker.os,
        "geteuid",
        lambda: (_ for _ in ()).throw(AssertionError("geteuid should not run")),
    )

    worker._enter_pid_namespace()


def test_runner_locks_down_grader_and_private_dirs_before_worker(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    workspace = tmp_path / "workspace"
    grader_dir = tmp_path / "grader"
    private_dir = tmp_path / "private"
    output_dir = tmp_path / "verifier"
    for path in (workspace, grader_dir, private_dir):
        path.mkdir()
    seen: dict[str, object] = {}

    def fake_lock(paths):
        seen["locked_paths"] = tuple(Path(path) for path in paths)

    def fake_cleanup(path):
        seen["cleanup_path"] = Path(path)
        return {"killed_processes": 0, "removed_symlinks": 0, "removed_nonregular": 0}

    def fake_run(cmd, **_kwargs):
        result_path = Path(cmd[cmd.index("--result-path") + 1])
        result_path.write_text(
            json.dumps(
                {
                    "score": 1.0,
                    "subscores": {"score": 1.0},
                    "weights": {"score": 1.0},
                    "structured_subscores": [],
                    "metadata": {},
                }
            )
        )
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    monkeypatch.setattr(run_grader, "lock_down_grader_private", fake_lock)
    monkeypatch.setattr(run_grader.os, "geteuid", lambda: 0)
    monkeypatch.setattr(run_grader, "pre_grade_cleanup", fake_cleanup)
    monkeypatch.setattr(run_grader, "prepare_grader_cache", lambda: {})
    monkeypatch.setattr(run_grader.subprocess, "run", fake_run)

    exit_code = run_grader._run_worker(
        workspace=workspace,
        grader_dir=grader_dir,
        private=private_dir,
        output_dir=output_dir,
        transcript=None,
        timeout_s=10.0,
    )

    assert exit_code == 0
    assert seen["locked_paths"] == (grader_dir, private_dir)
    assert seen["cleanup_path"] == workspace


def test_runner_writes_failure_when_worker_launch_raises(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    workspace = tmp_path / "workspace"
    grader_dir = tmp_path / "grader"
    private_dir = tmp_path / "private"
    output_dir = tmp_path / "verifier"
    for path in (workspace, grader_dir, private_dir):
        path.mkdir()

    monkeypatch.setattr(run_grader, "lock_down_grader_private", lambda _paths: None)
    monkeypatch.setattr(
        run_grader,
        "pre_grade_cleanup",
        lambda _path: {
            "killed_processes": 0,
            "removed_symlinks": 0,
            "removed_nonregular": 0,
        },
    )
    monkeypatch.setattr(run_grader, "prepare_grader_cache", lambda: {})

    def fail_run(*_args, **_kwargs):
        raise FileNotFoundError("python missing")

    monkeypatch.setattr(run_grader.subprocess, "run", fail_run)

    exit_code = run_grader._run_worker(
        workspace=workspace,
        grader_dir=grader_dir,
        private=private_dir,
        output_dir=output_dir,
        transcript=None,
        timeout_s=10.0,
    )

    assert exit_code == 1
    reward = json.loads((output_dir / "reward.json").read_text())
    details = json.loads((output_dir / "reward-details.json").read_text())
    assert reward["score"] == 0.0
    assert details["env_internal_failure"] is True
    assert (
        details["metadata"]["grading_errors"][0]["error_type"] == "grader_launch_failed"
    )


def test_runner_ignores_success_payload_on_signal_exit(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    workspace = tmp_path / "workspace"
    grader_dir = tmp_path / "grader"
    private_dir = tmp_path / "private"
    output_dir = tmp_path / "verifier"
    for path in (workspace, grader_dir, private_dir):
        path.mkdir()

    monkeypatch.setattr(run_grader, "lock_down_grader_private", lambda _paths: None)
    monkeypatch.setattr(
        run_grader,
        "pre_grade_cleanup",
        lambda _path: {
            "killed_processes": 0,
            "removed_symlinks": 0,
            "removed_nonregular": 0,
        },
    )
    monkeypatch.setattr(run_grader, "prepare_grader_cache", lambda: {})

    def killed_after_result(cmd, **_kwargs):
        result_path = Path(cmd[cmd.index("--result-path") + 1])
        result_path.write_text(
            json.dumps(
                {
                    "score": 1.0,
                    "subscores": {"score": 1.0},
                    "weights": {"score": 1.0},
                    "structured_subscores": [],
                    "metadata": {},
                }
            )
        )
        return SimpleNamespace(returncode=-9, stdout="", stderr="")

    monkeypatch.setattr(run_grader.subprocess, "run", killed_after_result)

    exit_code = run_grader._run_worker(
        workspace=workspace,
        grader_dir=grader_dir,
        private=private_dir,
        output_dir=output_dir,
        transcript=None,
        timeout_s=10.0,
    )

    assert exit_code == -9
    reward = json.loads((output_dir / "reward.json").read_text())
    details = json.loads((output_dir / "reward-details.json").read_text())
    assert reward["score"] == 0.0
    assert details["score"] == 0.0
    assert details["env_internal_failure"] is True
    assert (
        details["metadata"]["grading_errors"][0]["error_type"] == "grader_infra_failure"
    )


# --- grading_dependencies: grader-only, root-only pip deps -----------------


def test_prepend_grading_deps_makes_module_importable(tmp_path: Path, monkeypatch) -> None:
    """(a) A grader-only dep living ONLY in /mcp_server/grading_deps is importable
    by the (root) grader worker after _prepend_grading_deps -- so a dataset grader
    can use a scoring/reference library the agent never sees."""
    import sys

    from env_server import config as env_config

    deps = tmp_path / "grading_deps"
    deps.mkdir()
    (deps / "grading_only_lib.py").write_text("SECRET_METRIC = 42\n")
    monkeypatch.setattr(env_config, "GRADING_DEPS_DIR", deps)

    saved = list(sys.path)
    try:
        with pytest.raises(ModuleNotFoundError):  # not reachable before the prepend
            __import__("grading_only_lib")
        worker._prepend_grading_deps()
        assert sys.path[0] == str(deps)
        import grading_only_lib  # noqa: F401

        assert grading_only_lib.SECRET_METRIC == 42
        worker._prepend_grading_deps()  # idempotent: no duplicate
        assert sys.path.count(str(deps)) == 1
    finally:
        sys.path[:] = saved
        sys.modules.pop("grading_only_lib", None)


def test_prepend_grading_deps_noop_when_absent(tmp_path: Path, monkeypatch) -> None:
    import sys

    from env_server import config as env_config

    monkeypatch.setattr(env_config, "GRADING_DEPS_DIR", tmp_path / "nope")
    before = list(sys.path)
    worker._prepend_grading_deps()
    assert sys.path == before


def test_submission_worker_env_does_not_inherit_grading_deps(tmp_path: Path, monkeypatch) -> None:
    """(b) grading_deps reaches the grader via an in-process sys.path.insert, which
    a fresh-exec'd uid-1000 submission worker never inherits: its env scrubs
    PYTHONPATH and sets PYTHONSAFEPATH, so the worker cannot import grading_deps.
    (The 0700 /mcp_server filesystem block is the ultimate guard but can't be
    exercised as a non-root test.)"""
    import sys

    from env_server import config as env_config
    from grading import policy_runner

    deps = tmp_path / "grading_deps"
    deps.mkdir()
    monkeypatch.setattr(env_config, "GRADING_DEPS_DIR", deps)
    saved = list(sys.path)
    try:
        worker._prepend_grading_deps()
        assert str(deps) in sys.path  # on the grader process's path...
        env = policy_runner._isolated_python_environ()  # ...but not the worker's env
        assert "PYTHONPATH" not in env
        assert env["PYTHONSAFEPATH"] == "1"
        assert not any(str(deps) in value for value in env.values())
    finally:
        sys.path[:] = saved
