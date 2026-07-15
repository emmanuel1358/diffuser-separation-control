from __future__ import annotations

import os
from pathlib import Path

import pytest

from grading import load_env_module, load_submitted_policy
from grading.faults import AgentFault


def _write(path: Path, body: str) -> Path:
    path.write_text(body)
    return path


# ---- load_submitted_policy (PolicyWorker parity) ----


def test_single_policy_handle(tmp_path: Path) -> None:
    p = _write(
        tmp_path / "policy.py",
        "def load_policy():\n"
        "    class P:\n"
        "        def choose(self):\n"
        "            return 3\n"
        "    return P()\n",
    )
    handle = load_submitted_policy(p, timeout_s=10)
    try:
        assert handle.choose() == 3
    finally:
        handle.close()


def test_dict_factory_returns_handles(tmp_path: Path) -> None:
    p = _write(
        tmp_path / "policy.py",
        "def load_policy():\n"
        "    class P:\n"
        "        def __init__(self, v):\n"
        "            self.v = v\n"
        "        def val(self):\n"
        "            return self.v\n"
        "    return {'a': P(1), 'b': P(2)}\n",
    )
    handles = load_submitted_policy(p, timeout_s=10)
    try:
        assert set(handles) == {"a", "b"}
        assert handles["a"].val() == 1
        assert handles["b"].val() == 2
    finally:
        next(iter(handles.values())).close()


def test_list_factory_returns_handles(tmp_path: Path) -> None:
    p = _write(
        tmp_path / "policy.py",
        "def load_policy():\n"
        "    class P:\n"
        "        def __init__(self, v):\n"
        "            self.v = v\n"
        "        def val(self):\n"
        "            return self.v\n"
        "    return [P(10), P(20)]\n",
    )
    handles = load_submitted_policy(p, timeout_s=10)
    try:
        assert [h.val() for h in handles] == [10, 20]
    finally:
        handles[0].close()


def test_source_mode_wraps_agent_artifact(tmp_path: Path) -> None:
    (tmp_path / "weights.txt").write_text("7")
    # Grader-supplied LITERAL wrapper that loads the agent's data artifact.
    wrapper = (
        "def load_policy():\n"
        "    class P:\n"
        "        def choose(self):\n"
        f"            return int(open({str(tmp_path / 'weights.txt')!r}).read())\n"
        "    return P()\n"
    )
    handle = load_submitted_policy(
        tmp_path / "ignored.py", source=wrapper, timeout_s=10
    )
    try:
        assert handle.choose() == 7
    finally:
        handle.close()


def test_missing_policy_raises_agent_fault(tmp_path: Path) -> None:
    with pytest.raises(AgentFault, match="missing submitted policy"):
        load_submitted_policy(tmp_path / "nope.py")


def test_directory_policy_raises_agent_fault(tmp_path: Path) -> None:
    d = tmp_path / "policy.py"
    d.mkdir()
    with pytest.raises(AgentFault, match="not a regular file"):
        load_submitted_policy(d)


def test_load_raise_is_agent_fault(tmp_path: Path) -> None:
    p = _write(tmp_path / "policy.py", "raise RuntimeError('boom at import')\n")
    with pytest.raises(AgentFault, match="boom at import"):
        load_submitted_policy(p, timeout_s=10)


def test_worker_recovers_after_failed_handshake(tmp_path: Path) -> None:
    """A failed init handshake must clear _proc so the same worker can restart."""
    from grading.policy_runner import PolicyWorker

    p = _write(tmp_path / "policy.py", "raise RuntimeError('boom at import')\n")
    worker = PolicyWorker(p, factory_name="load_policy", timeout_s=10)
    with pytest.raises(AgentFault):
        worker.start()
    assert worker._proc is None  # dead child torn down, not left dangling

    # Fix the policy and restart the SAME worker object -- it must respawn.
    _write(
        tmp_path / "policy.py",
        "def load_policy():\n"
        "    class P:\n"
        "        def choose(self):\n"
        "            return 5\n"
        "    return P()\n",
    )
    try:
        assert worker.call("choose") == 5
    finally:
        worker.close()


def test_fifo_policy_raises_agent_fault(tmp_path: Path) -> None:
    fifo = tmp_path / "policy.py"
    os.mkfifo(fifo)
    with pytest.raises(AgentFault, match="not a regular file"):
        load_submitted_policy(fifo)


# ---- M1: import-time self-kill must be a kept 0.0 (AgentFault), not a discard ----
#
# The two-phase handshake sends a pre-agent ack before the submitted module is
# exec'd. A hard self-kill at import sends no result frame, but because the ack
# already arrived the parent blames the agent (AgentFault / kept 0.0) instead of
# misclassifying it as env_internal_failure (discarded rollout). Otherwise an
# agent could "void" a losing rollout by crashing itself at import time.


def test_import_time_os_exit_is_agent_fault(tmp_path: Path) -> None:
    p = _write(tmp_path / "policy.py", "import os\nos._exit(0)\n")
    with pytest.raises(AgentFault):
        load_submitted_policy(p, timeout_s=10)


def test_import_time_sigkill_is_agent_fault(tmp_path: Path) -> None:
    p = _write(
        tmp_path / "policy.py",
        "import os, signal\nos.kill(os.getpid(), signal.SIGKILL)\n",
    )
    with pytest.raises(AgentFault):
        load_submitted_policy(p, timeout_s=10)


def test_import_time_segfault_is_agent_fault(tmp_path: Path) -> None:
    p = _write(
        tmp_path / "policy.py",
        "import ctypes\nctypes.string_at(0)\n",  # deref NULL -> SIGSEGV at import
    )
    with pytest.raises(AgentFault):
        load_submitted_policy(p, timeout_s=10)


# ---- load_env_module (trusted loader) ----


def test_load_env_module_loads_factory(tmp_path: Path) -> None:
    env = _write(tmp_path / "env.py", "def make_env(**k):\n    return {'ok': True}\n")
    mod = load_env_module(env, trusted_roots=(tmp_path,))
    assert mod.make_env() == {"ok": True}


def test_load_env_module_rejects_outside_trusted_roots(tmp_path: Path) -> None:
    env = _write(tmp_path / "env.py", "def make_env(**k):\n    return 1\n")
    with pytest.raises(ValueError, match="trusted-roots"):
        load_env_module(env, trusted_roots=(tmp_path / "other",))


def test_load_env_module_factory_check(tmp_path: Path) -> None:
    env = _write(tmp_path / "env.py", "X = 1\n")
    with pytest.raises(ValueError, match="make_env"):
        load_env_module(env, trusted_roots=(tmp_path,))
