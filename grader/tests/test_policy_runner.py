from __future__ import annotations

import errno
import os
import textwrap
from pathlib import Path
from types import SimpleNamespace

import pytest

from grading import policy_runner
from grading.policy_runner import PolicyTimeoutError, PolicyWorker, PolicyWorkerError


def test_policy_worker_accepts_module_act(tmp_path: Path) -> None:
    policy_path = tmp_path / "policy.py"
    policy_path.write_text(
        "def act(obs):\n" "    return [obs['x'] + 1, obs['items'][1]]\n"
    )

    with PolicyWorker(policy_path) as policy:
        assert policy.act({"x": 2, "items": [4, 5]}) == [3, 5]
        assert policy({"x": 3, "items": [6, 7]}) == [4, 7]


def test_policy_worker_bootstrap_does_not_need_env_server_on_sys_path(
    tmp_path: Path, monkeypatch
) -> None:
    def fake_isolated_python_environ() -> dict[str, str]:
        env = policy_runner._scrubbed_environ()
        env.pop("PYTHONPATH", None)
        env["PYTHONSAFEPATH"] = "1"
        return env

    monkeypatch.setattr(
        policy_runner, "_isolated_python_environ", fake_isolated_python_environ
    )

    policy_path = tmp_path / "policy.py"
    policy_path.write_text(
        "import os\n"
        "import sys\n"
        "def act(obs):\n"
        "    return ('env_server' not in sys.modules and "
        "'RUBRIC_TRUSTED_RUNTIME_PATH' not in os.environ)\n"
    )

    with PolicyWorker(policy_path) as policy:
        assert policy.act({}) is True


def test_policy_worker_accepts_policy_class(tmp_path: Path) -> None:
    policy_path = tmp_path / "policy.py"
    policy_path.write_text(
        "class Policy:\n"
        "    def act(self, obs):\n"
        "        return {'u': obs['x'] * 2}\n"
    )

    with PolicyWorker(policy_path) as policy:
        assert policy.act({"x": 3}) == {"u": 6}


def test_policy_worker_can_call_named_methods(tmp_path: Path) -> None:
    policy_path = tmp_path / "policy.py"
    policy_path.write_text(
        "def identify_preset(name, ctrl, obs, scale=1):\n"
        "    return {'name': name, 'n': len(ctrl), 'first': obs[0][0], 'scale': scale}\n"
    )

    with PolicyWorker(policy_path) as policy:
        assert policy.call(
            "identify_preset",
            "model.xml",
            [[1.0, 2.0]],
            [[[3.0]]],
            scale=2,
        ) == {"name": "model.xml", "n": 1, "first": [3.0], "scale": 2}


def test_policy_worker_inherits_grader_cwd_by_default(
    tmp_path: Path, monkeypatch
) -> None:
    policy_dir = tmp_path / "workspace"
    policy_dir.mkdir()
    policy_path = policy_dir / "policy.py"
    policy_path.write_text(
        "from pathlib import Path\n" "def act(obs):\n" "    return Path.cwd().name\n"
    )
    grader_cwd = tmp_path / "grader-cwd"
    grader_cwd.mkdir()
    monkeypatch.chdir(grader_cwd)

    with PolicyWorker(policy_path) as policy:
        assert policy.act({}) == "grader-cwd"


def test_policy_worker_can_provide_output_model_xml(tmp_path: Path) -> None:
    policy_path = tmp_path / "policy.py"
    policy_path.write_text(
        "import os\n"
        "from pathlib import Path\n"
        "def get_action(obs, model=None):\n"
        "    path = '/tmp/output/model.xml'\n"
        "    if not os.path.exists(path):\n"
        "        path = 'model.xml'\n"
        "    return Path(path).read_text()\n"
    )

    with PolicyWorker(policy_path) as policy:
        policy.init_model_xml("<mujoco/>")
        assert policy.call("get_action", [], model=None) == "<mujoco/>"


def test_policy_worker_times_out(tmp_path: Path) -> None:
    policy_path = tmp_path / "policy.py"
    policy_path.write_text(
        "import time\n" "def act(obs):\n" "    time.sleep(10)\n" "    return 0\n"
    )

    # Pin the first-call budget too, otherwise the generous startup floor would
    # mask the per-step timeout being exercised here.
    with pytest.raises(TimeoutError) as excinfo:
        with PolicyWorker(
            policy_path, timeout_s=0.05, first_call_timeout_s=0.05
        ) as policy:
            policy.act({})
    # A per-call hang must be a PolicyTimeoutError so it is ALSO catchable by a
    # migrated grader's `except RuntimeError: return 0.0` (kept 0.0) and the
    # handshake's `except PolicyWorkerError` -- a bare builtin TimeoutError would
    # escape `except RuntimeError` and get the rollout DISCARDED (a free veto).
    assert isinstance(excinfo.value, PolicyTimeoutError)
    assert isinstance(excinfo.value, RuntimeError)
    assert isinstance(excinfo.value, PolicyWorkerError)


def test_policy_worker_first_call_gets_generous_budget(tmp_path: Path) -> None:
    """A tight per-step timeout must not kill a slow first (startup) call."""
    policy_path = tmp_path / "policy.py"
    policy_path.write_text(
        "import time\n"
        "time.sleep(0.3)\n"  # simulate slow import / MuJoCo init at module load
        "def act(obs):\n"
        "    return obs['x']\n"
    )

    # timeout_s well below the import delay; first_call floor must absorb it.
    with PolicyWorker(policy_path, timeout_s=0.05, first_call_timeout_s=5.0) as policy:
        assert policy.act({"x": 7}) == 7


def test_policy_worker_can_restart_after_kill(tmp_path: Path) -> None:
    policy_path = tmp_path / "policy.py"
    policy_path.write_text("def act(obs):\n    return obs['x']\n")

    policy = PolicyWorker(policy_path)
    try:
        assert policy.act({"x": 1}) == 1
        policy.kill()
        assert policy.act({"x": 2}) == 2
    finally:
        policy.close()


def test_policy_worker_surfaces_policy_error(tmp_path: Path) -> None:
    policy_path = tmp_path / "policy.py"
    policy_path.write_text("def act(obs):\n" "    raise RuntimeError('boom')\n")

    with pytest.raises(PolicyWorkerError, match="boom"):
        with PolicyWorker(policy_path) as policy:
            policy.act({})


def test_policy_worker_tolerates_policy_prints(tmp_path: Path) -> None:
    policy_path = tmp_path / "policy.py"
    policy_path.write_text(
        "print('import noise')\n"
        "def act(obs):\n"
        "    print('act noise')\n"
        "    return [1, 2]\n"
    )

    with PolicyWorker(policy_path) as policy:
        assert policy.act({}) == [1, 2]
        assert "import noise" in policy.stderr()
        assert "act noise" in policy.stderr()


def test_policy_worker_cannot_inspect_grader_locals(tmp_path: Path) -> None:
    policy_path = tmp_path / "policy.py"
    policy_path.write_text(textwrap.dedent("""
            import inspect

            def act(obs):
                for frame in inspect.stack():
                    if "hidden_case" in frame.frame.f_locals:
                        return frame.frame.f_locals["hidden_case"]
                return "not_visible"
            """))

    hidden_case = "secret schedule"
    with PolicyWorker(policy_path) as policy:
        assert policy.act({"public": True}) == "not_visible"
    assert hidden_case == "secret schedule"


@pytest.mark.skipif(
    os.geteuid() != 0, reason="privilege drop only fires when parent is root"
)
def test_policy_worker_drops_privileges_when_root(tmp_path: Path) -> None:
    import pwd

    try:
        agent = pwd.getpwnam("agent")
    except KeyError:
        pytest.skip("no 'agent' account to drop to")

    policy_path = tmp_path / "policy.py"
    policy_path.write_text(
        "import os\n" "def act(obs):\n" "    return [os.geteuid(), os.getegid()]\n"
    )
    # tmp_path is created as root with restrictive perms; loosen so the
    # dropped child can actually read the policy file it was handed.
    os.chmod(tmp_path, 0o755)
    os.chmod(policy_path, 0o644)

    with PolicyWorker(policy_path) as policy:
        assert policy.act({}) == [agent.pw_uid, agent.pw_gid]


def _clear_agent_identity_env(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in (
        "RUBRIC_AGENT_USER",
        "RUBRIC_AGENT_UID",
        "RUBRIC_AGENT_GID",
        "RUBRIC_AGENT_HOME",
    ):
        monkeypatch.delenv(name, raising=False)


def test_agent_drop_kwargs_fails_closed_when_root_user_missing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _clear_agent_identity_env(monkeypatch)
    monkeypatch.setattr(policy_runner.os, "geteuid", lambda: 0)
    monkeypatch.setattr(
        policy_runner.pwd,
        "getpwnam",
        lambda name: (_ for _ in ()).throw(KeyError(name)),
    )

    with pytest.raises(PolicyWorkerError, match="cannot drop privileges"):
        policy_runner._agent_drop_kwargs()


def test_policy_worker_does_not_start_when_privilege_drop_is_unavailable(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _clear_agent_identity_env(monkeypatch)
    policy_path = tmp_path / "policy.py"
    policy_path.write_text("def act(obs):\n    return 1\n")
    monkeypatch.setattr(policy_runner.os, "geteuid", lambda: 0)
    monkeypatch.setattr(
        policy_runner.pwd,
        "getpwnam",
        lambda name: (_ for _ in ()).throw(KeyError(name)),
    )

    def fail_popen(*_args, **_kwargs):
        raise AssertionError("policy worker should not start")

    monkeypatch.setattr(policy_runner.subprocess, "Popen", fail_popen)

    with pytest.raises(PolicyWorkerError, match="cannot drop privileges"):
        PolicyWorker(policy_path).start()


def test_agent_drop_kwargs_uses_configured_numeric_identity(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _clear_agent_identity_env(monkeypatch)
    monkeypatch.setattr(policy_runner.os, "geteuid", lambda: 0)
    monkeypatch.setenv("RUBRIC_AGENT_USER", "worker")
    monkeypatch.setenv("RUBRIC_AGENT_UID", "1234")
    monkeypatch.setenv("RUBRIC_AGENT_GID", "1235")
    monkeypatch.setenv("RUBRIC_AGENT_HOME", "/tmp/worker-home")
    monkeypatch.setattr(
        policy_runner.pwd,
        "getpwuid",
        lambda uid: (_ for _ in ()).throw(KeyError(uid)),
    )

    kwargs = policy_runner._agent_drop_kwargs()

    assert kwargs["user"] == 1234
    assert kwargs["group"] == 1235
    assert kwargs["extra_groups"] == []
    assert kwargs["env"]["HOME"] == "/tmp/worker-home"
    assert kwargs["env"]["USER"] == "worker"
    assert kwargs["env"]["LOGNAME"] == "worker"
    assert kwargs["env"]["PYTHONSAFEPATH"] == "1"


def test_agent_drop_kwargs_uses_configured_user_name(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _clear_agent_identity_env(monkeypatch)
    monkeypatch.setattr(policy_runner.os, "geteuid", lambda: 0)
    monkeypatch.setenv("RUBRIC_AGENT_USER", "worker")

    def fake_getpwnam(name: str):
        assert name == "worker"
        return SimpleNamespace(
            pw_uid=1234,
            pw_gid=1235,
            pw_dir="/home/worker",
            pw_name="worker",
        )

    monkeypatch.setattr(policy_runner.pwd, "getpwnam", fake_getpwnam)

    kwargs = policy_runner._agent_drop_kwargs()

    assert kwargs["user"] == 1234
    assert kwargs["group"] == 1235
    assert kwargs["env"]["HOME"] == "/home/worker"
    assert kwargs["env"]["USER"] == "worker"


def test_agent_drop_kwargs_rejects_root_numeric_identity(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _clear_agent_identity_env(monkeypatch)
    monkeypatch.setattr(policy_runner.os, "geteuid", lambda: 0)
    monkeypatch.setenv("RUBRIC_AGENT_UID", "0")
    monkeypatch.setenv("RUBRIC_AGENT_GID", "0")

    with pytest.raises(PolicyWorkerError, match="non-root account"):
        policy_runner._agent_drop_kwargs()


def test_policy_worker_round_trips_numpy_arrays(tmp_path: Path) -> None:
    import numpy as np

    policy_path = tmp_path / "policy.py"
    policy_path.write_text(textwrap.dedent("""
            import numpy as np

            def act(obs):
                assert isinstance(obs["ctrl"], np.ndarray), type(obs["ctrl"])
                assert obs["ctrl"].dtype == np.float64
                return {"action": obs["ctrl"] * 2, "shape": list(obs["ctrl"].shape)}
            """))

    ctrl = np.array([[1.0, 2.0], [3.0, 4.0]])
    with PolicyWorker(policy_path) as policy:
        result = policy.act({"ctrl": ctrl})

    assert isinstance(result["action"], np.ndarray)
    assert result["action"].dtype == np.float64
    np.testing.assert_array_equal(result["action"], ctrl * 2)
    assert result["shape"] == [2, 2]


# --- M28: default private IPC namespace for submitted-policy workers ----------


def _configure_root_agent_identity(monkeypatch: pytest.MonkeyPatch) -> None:
    _clear_agent_identity_env(monkeypatch)
    monkeypatch.setattr(policy_runner.os, "geteuid", lambda: 0)
    monkeypatch.setattr(
        policy_runner.pwd,
        "getpwnam",
        lambda name: SimpleNamespace(
            pw_uid=1234, pw_gid=1235, pw_dir="/home/worker", pw_name="worker"
        ),
    )
    monkeypatch.setenv("RUBRIC_AGENT_USER", "worker")


def test_unshare_ipc_defaults_to_true() -> None:
    """run_policy / load_submitted_policy / PolicyWorker all default the private
    IPC namespace ON (M28), matching ML_Envs -- only the C-level fast-drop path
    is opt-out now."""
    import inspect

    from grading import helpers

    assert (
        inspect.signature(PolicyWorker.__init__).parameters["unshare_ipc"].default
        is True
    )
    assert (
        inspect.signature(policy_runner.load_submitted_policy)
        .parameters["unshare_ipc"]
        .default
        is True
    )
    assert (
        inspect.signature(helpers.run_policy).parameters["unshare_ipc"].default
        is True
    )


def test_agent_drop_kwargs_unshare_ipc_selects_preexec_path(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _configure_root_agent_identity(monkeypatch)

    kwargs = policy_runner._agent_drop_kwargs(unshare_ipc=True)

    assert callable(kwargs.get("preexec_fn"))
    assert "user" not in kwargs and "group" not in kwargs


def test_agent_drop_kwargs_without_unshare_keeps_c_level_drop(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _configure_root_agent_identity(monkeypatch)

    kwargs = policy_runner._agent_drop_kwargs(unshare_ipc=False)

    assert "preexec_fn" not in kwargs
    assert kwargs["user"] == 1234
    assert kwargs["group"] == 1235
    assert kwargs["extra_groups"] == []


def test_agent_drop_kwargs_non_root_never_unshares(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _clear_agent_identity_env(monkeypatch)
    monkeypatch.setattr(policy_runner.os, "geteuid", lambda: 1000)

    kwargs = policy_runner._agent_drop_kwargs(unshare_ipc=True, ipc_status_fd=99)

    # Non-root: early return, no preexec / no unshare -- unisolated like before.
    assert "preexec_fn" not in kwargs
    assert "user" not in kwargs


def _run_preexec_capturing_status(
    monkeypatch: pytest.MonkeyPatch, *, unshare_ret: int, errno_val: int
) -> int | None:
    """Run the preexec closure with setuid/setgid stubbed out (we are not root
    in CI) and a fake libc, returning the status byte the hook wrote, or None."""
    monkeypatch.setattr(policy_runner.os, "setgroups", lambda groups: None)
    monkeypatch.setattr(policy_runner.os, "setgid", lambda gid: None)
    monkeypatch.setattr(policy_runner.os, "setuid", lambda uid: None)
    fake_libc = SimpleNamespace(unshare=lambda arg: unshare_ret)
    monkeypatch.setattr(policy_runner, "_LIBC", fake_libc)
    monkeypatch.setattr(policy_runner, "_CLONE_NEWIPC_ARG", 0x08000000)
    monkeypatch.setattr(policy_runner.ctypes, "set_errno", lambda v: None)
    monkeypatch.setattr(policy_runner.ctypes, "get_errno", lambda: errno_val)

    r, w = os.pipe()
    try:
        policy_runner._agent_preexec(1234, 1235, ipc_status_fd=w)()
        os.close(w)
        w = -1
        data = os.read(r, 1)
    finally:
        os.close(r)
        if w != -1:
            os.close(w)
    return data[0] if data else None


def test_agent_preexec_reports_unshare_success(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    status = _run_preexec_capturing_status(
        monkeypatch, unshare_ret=0, errno_val=0
    )
    assert status == 0


def test_agent_preexec_reports_unshare_failure_errno(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    status = _run_preexec_capturing_status(
        monkeypatch, unshare_ret=-1, errno_val=errno.EPERM
    )
    assert status == errno.EPERM


def test_warn_if_ipc_unisolated_warns_on_failure(
    caplog: pytest.LogCaptureFixture,
) -> None:
    r, w = os.pipe()
    os.write(w, bytes([errno.EPERM]))
    os.close(w)
    with caplog.at_level("WARNING", logger=policy_runner.__name__):
        PolicyWorker._warn_if_ipc_unisolated(r)
    assert any("NOT isolated" in rec.message for rec in caplog.records)


def test_warn_if_ipc_unisolated_silent_on_success(
    caplog: pytest.LogCaptureFixture,
) -> None:
    r, w = os.pipe()
    os.write(w, bytes([0]))
    os.close(w)
    with caplog.at_level("WARNING", logger=policy_runner.__name__):
        PolicyWorker._warn_if_ipc_unisolated(r)
    assert caplog.records == []


def test_warn_if_ipc_unisolated_warns_when_status_missing(
    caplog: pytest.LogCaptureFixture,
) -> None:
    r, w = os.pipe()
    os.close(w)  # no byte written: child died before reporting
    with caplog.at_level("WARNING", logger=policy_runner.__name__):
        PolicyWorker._warn_if_ipc_unisolated(r)
    assert any("could not determine" in rec.message for rec in caplog.records)
