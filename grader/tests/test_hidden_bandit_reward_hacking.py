from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pytest

from grader_runner.run_grader import main


def _run_script(script: Path, workspace: Path) -> None:
    source = script.read_text().replace("/tmp/output", str(workspace))
    completed = subprocess.run(
        ["bash", "-c", source],
        cwd=workspace,
        text=True,
        capture_output=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr


def _grade(example: Path, workspace: Path, output: Path) -> dict:
    rc = main(
        [
            "--workspace",
            str(workspace),
            "--grader-dir",
            str(example / "scorer"),
            "--private-dir",
            str(example / "scorer" / "data"),
            "--output-dir",
            str(output),
            "--timeout",
            "120",
        ]
    )
    assert rc == 0
    return json.loads((output / "reward-details.json").read_text())


def test_hidden_bandit_reference_passes_fresh_policy_challenges(
    template_examples: Path, tmp_path
) -> None:
    example = template_examples / "hidden-env-bandit"
    workspace = tmp_path / "reference"
    output = tmp_path / "reference-verifier"
    workspace.mkdir()
    _run_script(example / "solution" / "solve.sh", workspace)

    details = _grade(example, workspace, output)

    assert details["score"] == pytest.approx(0.5, abs=0.03)
    receipt = details["metadata"]["evaluation"]
    assert receipt["protocol"] == "paired-policy-challenge.v1"
    assert all(decision["accepted"] for decision in receipt["decisions"].values())


def test_hidden_bandit_constant_baseline_scores_zero(
    template_examples: Path, tmp_path
) -> None:
    example = template_examples / "hidden-env-bandit"
    workspace = tmp_path / "constant"
    output = tmp_path / "constant-verifier"
    workspace.mkdir()
    _run_script(example / "baselines" / "naive.sh", workspace)

    details = _grade(example, workspace, output)

    assert details["score"] == pytest.approx(0.0)
    assert all(
        not decision["accepted"]
        for decision in details["metadata"]["evaluation"]["decisions"].values()
    )


def test_hidden_bandit_reference_is_stable_across_private_nonces(
    template_examples: Path, tmp_path, monkeypatch
) -> None:
    example = template_examples / "hidden-env-bandit"
    workspace = tmp_path / "stable-reference"
    workspace.mkdir()
    _run_script(example / "solution" / "solve.sh", workspace)
    monkeypatch.setenv("LBX_EVALUATION_PLAN_ATTESTED", "1")

    scores = []
    for index in range(8):
        monkeypatch.setenv("LBX_EVALUATION_NONCE", f"attempt-{index}")
        scores.append(
            _grade(
                example,
                workspace,
                tmp_path / f"stable-verifier-{index}",
            )["score"]
        )

    assert min(scores) >= 0.45
    assert max(scores) <= 0.55


def test_hidden_bandit_open_loop_perturbations_do_not_bypass(
    template_examples: Path, tmp_path
) -> None:
    example = template_examples / "hidden-env-bandit"
    workspace = tmp_path / "open-loop"
    output = tmp_path / "open-loop-verifier"
    workspace.mkdir()
    (workspace / "policy.py").write_text(
        "def load_policy():\n"
        "    class Policy:\n"
        "        def reset(self, n_arms, budget):\n"
        "            self.n_arms, self.i = int(n_arms), 0\n"
        "        def choose(self):\n"
        "            self.i += 1\n"
        "            return self.i % self.n_arms\n"
        "        def observe(self, arm, reward):\n"
        "            pass\n"
        "        def recommend(self):\n"
        "            return self.i % self.n_arms\n"
        "    return Policy()\n",
        encoding="utf-8",
    )

    details = _grade(example, workspace, output)

    assert details["score"] == pytest.approx(0.0)


def test_hidden_bandit_policy_crash_is_kept_zero(
    template_examples: Path, tmp_path
) -> None:
    example = template_examples / "hidden-env-bandit"
    workspace = tmp_path / "crash"
    output = tmp_path / "crash-verifier"
    workspace.mkdir()
    (workspace / "policy.py").write_text(
        "def load_policy():\n"
        "    class Policy:\n"
        "        def reset(self, n_arms, budget): pass\n"
        "        def choose(self): raise RuntimeError('boom')\n"
        "        def observe(self, arm, reward): pass\n"
        "        def recommend(self): return 0\n"
        "    return Policy()\n",
        encoding="utf-8",
    )

    details = _grade(example, workspace, output)

    assert details["score"] == pytest.approx(0.0)
    assert details["env_internal_failure"] is False
