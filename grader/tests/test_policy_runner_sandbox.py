"""Adversarial tests for the submitted-policy sandbox.

These encode the security contract that the QA "reward hacking" findings
violated: a submitted policy must not be able to forge its score, shadow the
grader's modules, or (when the grader runs as root) read hidden data. Each
test models a concrete exploit and asserts it fails.
"""

from __future__ import annotations

import os
import textwrap
import time
from pathlib import Path

import pytest

from grading import helpers
from grading.policy_runner import PolicyWorker, _scrubbed_environ


def test_forged_stdout_does_not_become_the_result(tmp_path: Path) -> None:
    """A policy that prints a fake score must not influence the real result.

    Scores travel over the dedicated proto FD, never stdout, so a policy that
    spews a convincing ``RUBRIC_RESULT_JSON``/score line can't forge a grade.
    """
    policy_path = tmp_path / "policy.py"
    policy_path.write_text(
        textwrap.dedent(
            """
            import sys

            def act(obs):
                # Try every channel a naive stdout-parsing scorer might read.
                print('{"score": 1.0}')
                print("RUBRIC_RESULT_JSON: {\\"score\\": 1.0}")
                sys.stdout.write("FINAL_SCORE=1.0\\n")
                return obs["truth"]
            """
        )
    )

    with PolicyWorker(policy_path) as policy:
        assert policy.act({"truth": [0, 0]}) == [0, 0]
        # The forged lines are visible only as captured stderr noise.
        for _ in range(50):
            if "FINAL_SCORE=1.0" in policy.stderr():
                break
            time.sleep(0.01)
        assert "FINAL_SCORE=1.0" in policy.stderr()


def test_planted_json_module_cannot_forge_proto_response(tmp_path: Path) -> None:
    """A `json.py` planted next to the policy can't hijack the proto channel.

    The worker pins stdlib ``json`` in ``sys.modules`` before loading the
    policy and scrubs the cwd/'' entries from ``sys.path``, so agent-writable
    code cannot shadow the module used to serialize the response.
    """
    workspace = tmp_path / "output"
    workspace.mkdir()
    # Malicious shadow that, if used to serialize the worker response, would
    # always report a perfect score.
    (workspace / "json.py").write_text(
        textwrap.dedent(
            """
            def dumps(*args, **kwargs):
                return '{"ok": true, "result": [9, 9, 9]}'

            def loads(*args, **kwargs):
                return {}
            """
        )
    )
    policy_path = workspace / "policy.py"
    policy_path.write_text("def act(obs):\n    return obs['truth']\n")

    with PolicyWorker(policy_path) as policy:
        assert policy.act({"truth": [1, 2]}) == [1, 2]


def test_planted_json_in_cwd_cannot_forge_proto_response(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Same guarantee when the worker's cwd holds the shadow module.

    The model-xml flow chdir's into an agent-writable dir; '' must not be on
    ``sys.path`` so a `json.py` there can't shadow the response serializer.
    """
    cwd = tmp_path / "cwd"
    cwd.mkdir()
    monkeypatch.setenv("PYTHONPATH", str(cwd))
    (cwd / "json.py").write_text(
        textwrap.dedent(
            """
            def dump(*args, **kwargs):
                pass

            def load(*args, **kwargs):
                return {}

            def dumps(*args, **kwargs):
                return '{"ok": true, "result": [7, 7]}'

            def loads(*args, **kwargs):
                return {}
            """
        )
    )
    policy_path = tmp_path / "policy.py"
    policy_path.write_text("def act(obs):\n    return obs['truth']\n")

    with PolicyWorker(policy_path, cwd=cwd) as policy:
        assert policy.act({"truth": [3, 4]}) == [3, 4]


def test_policy_cannot_monkeypatch_proto_serializer(tmp_path: Path) -> None:
    """Submitted code cannot replace the worker's response encoder."""
    policy_path = tmp_path / "policy.py"
    policy_path.write_text(
        textwrap.dedent(
            """
            import json
            import json_numpy

            def act(obs):
                json.dumps = lambda *a, **k: '{"ok": true, "result": [8, 8]}'
                json_numpy.default = lambda *a, **k: {"forged": True}
                return obs["truth"]
            """
        )
    )

    with PolicyWorker(policy_path) as policy:
        assert policy.act({"truth": [5, 6]}) == [5, 6]


def test_run_policy_helper_runs_a_workspace_policy(tmp_path: Path) -> None:
    """`helpers.run_policy` resolves a workspace dir and applies safe defaults."""
    workspace = tmp_path / "ws"
    workspace.mkdir()
    (workspace / "policy.py").write_text("def act(obs):\n    return obs['x'] + 1\n")

    with helpers.run_policy(workspace) as policy:
        assert policy.act({"x": 41}) == 42
        # Hardened contract: privilege drop is on by default.
        assert policy.drop_privileges is True


def test_run_policy_helper_rejects_missing_policy(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        helpers.run_policy(tmp_path)


def test_scrubbed_environ_drops_grading_secrets(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Grading-side secrets are stripped from the policy child's environment.

    The Anthropic proxy credentials are provisioned for the grader's LLM judge,
    not for submitted code; leaking them to the policy child is the
    credential-hygiene bucket from the QA findings.
    """
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-secret")
    monkeypatch.setenv("ANTHROPIC_BASE_URL", "http://proxy")
    monkeypatch.setenv("ANTHROPIC_API_AUXILIARY_MODEL_NAME", "claude")
    monkeypatch.setenv("MY_SERVICE_TOKEN", "tok")
    monkeypatch.setenv("PUBLIC_SETTING", "ok")

    env = _scrubbed_environ()

    assert "ANTHROPIC_API_KEY" not in env
    assert "ANTHROPIC_BASE_URL" not in env
    assert "ANTHROPIC_API_AUXILIARY_MODEL_NAME" not in env
    assert "MY_SERVICE_TOKEN" not in env  # matched by the *_TOKEN substring
    assert env.get("PUBLIC_SETTING") == "ok"  # unrelated config is preserved


@pytest.mark.skipif(
    os.geteuid() != 0, reason="privilege drop only activates when grader is root"
)
def test_root_policy_does_not_inherit_grading_secrets(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """When the grader is root, the dropped policy child can't see the API key."""
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-should-not-leak")
    policy_path = tmp_path / "policy.py"
    policy_path.write_text(
        "import os\n"
        "def act(obs):\n"
        "    return os.environ.get('ANTHROPIC_API_KEY', 'ABSENT')\n"
    )
    with PolicyWorker(policy_path) as policy:
        assert policy.act({}) == "ABSENT"


@pytest.mark.skipif(
    os.geteuid() != 0, reason="privilege drop only activates when grader is root"
)
def test_root_policy_cannot_read_root_only_hidden_data(tmp_path: Path) -> None:
    """When the grader is root, the dropped policy can't read root-0600 data.

    This is the hidden-data-readable bucket: hidden fixtures are root-owned
    0600 and the policy runs as the unprivileged ``agent`` account.
    """
    secret = tmp_path / "private.key"
    secret.write_text("ground-truth-answer")
    os.chmod(secret, 0o600)  # root:root 0600 — unreadable to agent

    policy_path = tmp_path / "policy.py"
    policy_path.write_text(
        textwrap.dedent(
            f"""
            def act(obs):
                try:
                    return open({str(secret)!r}).read()
                except OSError as exc:
                    return f"denied:{{type(exc).__name__}}"
            """
        )
    )

    with PolicyWorker(policy_path) as policy:
        result = policy.act({})
    assert result.startswith("denied:"), result
