"""The task validator accepts a valid metadata-mode task and still catches
reward-hacking / prompt problems. Build/exec-coupled stages skip-with-warning
here, so the validator runs without Docker."""

from __future__ import annotations

import json
from pathlib import Path

from alignerr_plugin.validators.task.validator import TaskValidator

_PROMPT = (
    "# Tabular task\n\n"
    "Train a model on the dataset provided under `/data/`. The training split "
    "lives in `/data/train.csv`. Fit a model, then write your predictions for "
    "the held-out rows to `/tmp/output/submission.csv` so the grader can score "
    "them. Do not modify files outside the output directory.\n"
)

_GRADER = (
    "from pathlib import Path\n"
    "from grading.helpers import load_submission_or_fault  # noqa: F401\n"
    "from grading.faults import AgentFault  # noqa: F401\n\n"
    "SUBMISSION = Path('/tmp/output')\n"
    "PRIVATE = Path('/mcp_server/data')\n\n\n"
    "def compute_score():\n"
    "    return 0.5\n"
)


def _write_mlenvs_task(root: Path, *, grader: str = _GRADER, prompt: str = _PROMPT) -> Path:
    task_dir = root / "demo-task_taiga"
    (task_dir / "data" / "public").mkdir(parents=True)
    (task_dir / "data" / "private").mkdir(parents=True)
    (task_dir / "reference_solution").mkdir(parents=True)
    (task_dir / "baselines" / "naive").mkdir(parents=True)
    (task_dir / "metadata.json").write_text(
        json.dumps(
            {
                "ml_task_type": "dataset",
                "required_resources": "12vcpu+100gib+h100/2",
                "domain": "scientific_discovery_computational_science",
                "license": "CC0-1.0",
                "license_source": "https://creativecommons.org/publicdomain/zero/1.0/",
            },
            indent=2,
        )
    )
    (task_dir / "prompt.md").write_text(prompt)
    (task_dir / "test_file.py").write_text(grader)
    (task_dir / "data" / "public" / "train.csv").write_text("x,y\n1,2\n")
    (task_dir / "data" / "private" / "truth.csv").write_text("y\n2\n")
    (task_dir / "reference_solution" / "solution.py").write_text("print('ref')\n")
    (task_dir / "baselines" / "naive" / "solve.sh").write_text(
        "#!/bin/bash\necho naive\n"
    )
    return task_dir


def _validate(task_dir: Path, tmp_path: Path):
    results = tmp_path / "results"
    results.mkdir(exist_ok=True)
    return TaskValidator().validate(task_dir, results, task_dir.parent)


def test_valid_mlenvs_task_passes(tmp_path: Path):
    task_dir = _write_mlenvs_task(tmp_path)
    result = _validate(task_dir, tmp_path)
    failed = {name: s.issues for name, s in result.stages.items() if not s.passed}
    assert result.status == "valid", failed


def test_static_stages_target_mlenvs_files(tmp_path: Path):
    task_dir = _write_mlenvs_task(tmp_path)
    result = _validate(task_dir, tmp_path)
    # Schema accepts the metadata triplet (no task.toml / scorer/ required).
    assert result.stages["schema"].passed
    # grader_import accepts the no-arg signature in test_file.py.
    assert result.stages["grader_import"].passed


def test_three_arg_grader_in_test_file_is_rejected(tmp_path: Path):
    bad = (
        "def compute_score(workspace, trajectory, private):\n"
        "    return 0.5\n"
    )
    task_dir = _write_mlenvs_task(tmp_path, grader=bad)
    result = _validate(task_dir, tmp_path)
    assert not result.stages["grader_import"].passed
    assert any("no arguments" in i for i in result.stages["grader_import"].issues)


def _write_mlenvs_proof(task_dir: Path, ground_truth_result: dict) -> None:
    proof_dir = task_dir / ".alignerr"
    proof_dir.mkdir()
    (proof_dir / "build_proof.json").write_text(
        json.dumps({"schema_version": "1.0", "ground_truth_result": ground_truth_result})
    )


def test_mlenvs_without_ground_truth_result_skips_zero_anchor(tmp_path: Path):
    # Authoring iteration before the first ground-truth run: no proof (or a
    # proof without ground_truth_result) is not gated here -- the ground-truth
    # harness enforces and records the anchor at run time.
    task_dir = _write_mlenvs_task(tmp_path)
    stage, _meta = TaskValidator()._compute_score_return(task_dir)
    assert stage.passed, stage.issues

    _write_mlenvs_proof(task_dir, {"score": 0.5})
    proofless = _write_mlenvs_task(tmp_path / "no-result")
    (proofless / ".alignerr").mkdir()
    (proofless / ".alignerr" / "build_proof.json").write_text('{"schema_version": "1.0"}')
    stage, _meta = TaskValidator()._compute_score_return(proofless)
    assert stage.passed, stage.issues


def test_mlenvs_proof_requires_trivial_baseline_score(tmp_path: Path):
    # A proof that carries a ground_truth_result but no recorded no-op score is
    # stale w.r.t. the zero-anchor contract and must fail (ML_Envs tasks are
    # always continuous_scoring_function).
    task_dir = _write_mlenvs_task(tmp_path)
    _write_mlenvs_proof(task_dir, {"score": 0.5})
    stage, _meta = TaskValidator()._compute_score_return(task_dir)
    assert not stage.passed
    assert any("trivial_baseline_score is missing" in i for i in stage.issues)


def test_mlenvs_proof_accepts_zero_anchored_noop(tmp_path: Path):
    task_dir = _write_mlenvs_task(tmp_path)
    _write_mlenvs_proof(task_dir, {"score": 0.5, "trivial_baseline_score": 0.0})
    stage, meta = TaskValidator()._compute_score_return(task_dir)
    assert stage.passed, stage.issues
    assert meta["noop_score"] == 0.0


def test_mlenvs_proof_rejects_unanchored_noop(tmp_path: Path):
    task_dir = _write_mlenvs_task(tmp_path)
    _write_mlenvs_proof(task_dir, {"score": 0.5, "trivial_baseline_score": 0.2})
    stage, _meta = TaskValidator()._compute_score_return(task_dir)
    assert not stage.passed
    assert any("anchor an empty submission to 0" in i for i in stage.issues)


def test_llm_judge_in_test_file_is_rejected(tmp_path: Path):
    bad = (
        "from grading import LLMJudge\n\n\n"
        "def compute_score():\n"
        "    return 0.5\n"
    )
    task_dir = _write_mlenvs_task(tmp_path, grader=bad)
    result = _validate(task_dir, tmp_path)
    assert not result.stages["compute_score_return"].passed


def test_prompt_internal_env_reference_is_rejected(tmp_path: Path):
    bad_prompt = _PROMPT + "\nInspect metadata.json to discover the installed packages.\n"
    task_dir = _write_mlenvs_task(tmp_path, prompt=bad_prompt)
    result = _validate(task_dir, tmp_path)
    assert not result.stages["prompt_runtime_references"].passed


_PICKLE_GRADER = (
    "import torch\n"
    "from grading.helpers import load_submission_or_fault\n"
    "from grading.faults import AgentFault\n\n"
    "def compute_score():\n"
    "    truth = torch.load('{path}')\n"
    "    try:\n"
    "        sub = load_submission_or_fault('/tmp/output/predictions.csv')\n"
    "    except AgentFault:\n"
    "        return 0.0\n"
    "    return 0.5\n"
)


def test_pickle_load_of_trusted_truth_is_not_flagged(tmp_path: Path):
    # torch.load of the held-out truth under /mcp_server/data is trusted, not an
    # RCE vector; the agent_fault stage must not flag it (mirrors ML_Envs).
    grader = _PICKLE_GRADER.format(path="/mcp_server/data/truth.pt")
    task_dir = _write_mlenvs_task(tmp_path, grader=grader)
    result = _validate(task_dir, tmp_path)
    assert result.stages["agent_fault"].passed, result.stages["agent_fault"].issues


def test_pickle_load_of_agent_artifact_is_flagged(tmp_path: Path):
    # torch.load of an agent-controlled /tmp/output artifact IS an RCE vector.
    grader = _PICKLE_GRADER.format(path="/tmp/output/model.pt")
    task_dir = _write_mlenvs_task(tmp_path, grader=grader)
    result = _validate(task_dir, tmp_path)
    stage = result.stages["agent_fault"]
    assert not stage.passed
    assert any("deserializes pickle" in i for i in stage.issues)


def test_dataset_shipping_env_module_is_rejected(tmp_path: Path):
    # A dataset (no-socket) task must not ship a hidden env module under
    # data/private/; that paradigm is reserved for env/hybrid.
    task_dir = _write_mlenvs_task(tmp_path)
    (task_dir / "data" / "private" / "env.py").write_text(
        "def make_env(**kw):\n    return object()\n"
    )
    result = _validate(task_dir, tmp_path)
    stage = result.stages["mlenvs_structure"]
    assert not stage.passed
    assert any("env.py" in i for i in stage.issues)


def test_sim_policy_envs_package_is_allowed(tmp_path: Path):
    # A sim_policy task may package its held-out eval sim under an ``envs/``
    # directory (this fixture ships envs/__init__.py). With no env_config.json
    # declaring it, mlenvs_structure does not treat that package as the socket
    # hidden-env module (only env.py / env_config.json mark it), so the stage must
    # not reject it.
    task_dir = _write_mlenvs_task(tmp_path)
    meta = json.loads((task_dir / "metadata.json").read_text())
    meta["ml_task_type"] = "sim_policy"
    (task_dir / "metadata.json").write_text(json.dumps(meta, indent=2))
    (task_dir / "data" / "private" / "envs").mkdir()
    (task_dir / "data" / "private" / "envs" / "__init__.py").write_text(
        "def load_eval_sim():\n    return object()\n"
    )
    result = _validate(task_dir, tmp_path)
    stage = result.stages["mlenvs_structure"]
    assert stage.passed, stage.issues


def test_dataset_shipping_env_config_is_rejected(tmp_path: Path):
    # env_config.json declares the socket hidden-env server; a no-socket dataset
    # task must not ship it, so mlenvs_structure rejects it.
    task_dir = _write_mlenvs_task(tmp_path)
    (task_dir / "data" / "private" / "env_config.json").write_text(
        '{"module": "env.py", "factory": "make_env"}\n'
    )
    result = _validate(task_dir, tmp_path)
    stage = result.stages["mlenvs_structure"]
    assert not stage.passed
    assert any("env_config.json" in i for i in stage.issues)


def test_env_task_without_public_methods_allowlist_warns(tmp_path: Path):
    # An env task whose env declares no _env_public_methods allow-list gets a
    # reward-hacking WARNING (not a failure) on the hidden_env stage.
    task_dir = _write_mlenvs_task(tmp_path)
    meta = json.loads((task_dir / "metadata.json").read_text())
    meta["ml_task_type"] = "env"
    (task_dir / "metadata.json").write_text(json.dumps(meta, indent=2))
    (task_dir / "test_file.py").write_text(
        "from grading.env_loading import load_env_module\n\n"
        "def compute_score():\n"
        "    env = load_env_module('/mcp_server/data/env.py').make_env()\n"
        "    return 0.5\n"
    )
    (task_dir / "data" / "private" / "env.py").write_text(
        "class Env:\n"
        "    def reset(self):\n        return 0\n"
        "    def step(self, a):\n        return 0, 0, False, {}\n"
        "def make_env(**kw):\n    return Env()\n"
    )
    (task_dir / "data" / "public" / "env_client.py").write_text("# agent client\n")
    result = _validate(task_dir, tmp_path)
    stage = result.stages["hidden_env"]
    assert stage.passed, stage.issues  # advisory, not a failure
    assert any("_env_public_methods" in w for w in stage.warnings)


def test_reference_reading_ondisk_private_path_is_flagged(tmp_path: Path):
    # A reference reading the answer key by its on-disk ML_Envs path
    # (data/private/) must be flagged, not only the /mcp_server/data form.
    task_dir = _write_mlenvs_task(tmp_path)
    (task_dir / "reference_solution" / "solution.py").write_text(
        "import pandas as pd\n"
        "df = pd.read_csv('data/private/truth.csv')\n"
        "print('cheated')\n"
    )
    result = _validate(task_dir, tmp_path)
    stage = result.stages["solution_answer_key_leak"]
    assert not stage.passed
    assert any("data/private/" in i for i in stage.issues)


def test_reference_mentioning_private_path_in_docstring_is_allowed(tmp_path: Path):
    # Naming data/private/ in a docstring describes the layout without reading the
    # answer key; the scan targets reads, so a docstring mention must not flag.
    task_dir = _write_mlenvs_task(tmp_path)
    (task_dir / "reference_solution" / "solution.py").write_text(
        '"""Reference: trains only on /data; never reads data/private/."""\n'
        "import pandas as pd\n"
        "df = pd.read_csv('/data/train.csv')\n"
        "df.head().to_csv('/tmp/output/result.txt', index=False)\n"
    )
    result = _validate(task_dir, tmp_path)
    stage = result.stages["solution_answer_key_leak"]
    assert stage.passed, stage.issues


def test_reference_mentioning_private_path_in_shell_comment_is_allowed(tmp_path: Path):
    # A shell comment naming data/private/ is prose, not a read; it must not flag.
    task_dir = _write_mlenvs_task(tmp_path)
    (task_dir / "reference_solution" / "solve.sh").write_text(
        "#!/usr/bin/env bash\n"
        "set -euo pipefail\n"
        "# NOTE: the held-out truth under data/private/ is never touched here.\n"
        "python reference_solution/solution.py\n"
    )
    result = _validate(task_dir, tmp_path)
    stage = result.stages["solution_answer_key_leak"]
    assert stage.passed, stage.issues
