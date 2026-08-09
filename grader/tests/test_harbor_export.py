"""Tests for the Harbor exporter."""

from __future__ import annotations

import json
import re
import shutil
import tomllib
from pathlib import Path

from alignerr_plugin.exporters.harbor import (
    _SELF_CONTAINED_DOCKERFILE,
    _SOLVER_SELF_CONTAINED_DOCKERFILE,
    export_harbor,
)
from alignerr_plugin.schemas import HF_HOME

# A native-contract (task.toml) ml task written inline, for the runtime-notice
# tests that need native behavior. Kept as a helper, not a fixture directory.
_NATIVE_ML_TASK_TOML = """\
schema_version = "1.1"

[task]
name = "labelbox/native-ml-task"
description = "Native-contract ml task fixture for the Harbor export tests."

[environment]
required_resources = "12vcpu+100gib+h100/2"
storage_mb = 50000
allow_internet = true

[agent]
timeout_sec = 21600

[verifier]
timeout_sec = 5400
env = []

[ground_truth]
continuous_score_epsilon = 0.05

[[outputs]]
path = "/tmp/output/submission.csv"
required = true
description = "CSV predictions for the held-out test rows with columns t1, t2, label"

[runner]
attempts = 3
turn_limit = 1500
max_ctx = 1000000
context_mode = "none"
api_model_name = "claude-opus-4-7"
required_tools = ["bash", "str_replace_editor", "tmux"]

[runner.timeouts]
setup_sec = 7200
grading_sec = 5400
tool_sec = 21600
max_episode_sec = 21600

[difficulty]
task_type = "ml"
domain = "scientific_discovery_computational_science"
reward_type = "continuous_scoring_function"
license = "CC0-1.0"
license_source = "https://creativecommons.org/publicdomain/zero/1.0/"
"""

_NATIVE_ML_TASK_INSTRUCTION = """\
# Synthetic Tabular Regression + Classification

Train on the provided training rows, then produce predictions for the held-out
test rows.

The public data files live under `/data/`. Write predictions for every test row
to `/tmp/output/submission.csv` with a header row and columns `t1`, `t2`,
`label`.
"""


def _write_native_ml_task(problem_dir: Path) -> Path:
    problem_dir.mkdir(parents=True, exist_ok=True)
    (problem_dir / "task.toml").write_text(_NATIVE_ML_TASK_TOML)
    (problem_dir / "metadata.json").write_text(
        json.dumps(
            {
                "benchmark": "taiga_task",
                "problem_data": {
                    "instance_id": "native-ml-task",
                    "description": "Native-contract ml task fixture for the Harbor export tests",
                },
            }
        )
    )
    (problem_dir / "instruction.md").write_text(_NATIVE_ML_TASK_INSTRUCTION)
    return problem_dir


def _write_native_env_task(problem_dir: Path) -> Path:
    """A native hidden-env task declaring every dependency channel, so the
    Harbor render exercises the agent-visible, grader-only and env-only blocks."""
    problem_dir.mkdir(parents=True, exist_ok=True)
    (problem_dir / "task.toml").write_text("""\
schema_version = "1.1"

[task]
name = "labelbox/native-env-task"

[environment]
required_resources = "12vcpu+100gib+h100/2"
hidden_env = "env"

[difficulty]
task_type = "ml"
domain = "scientific_discovery_computational_science"
reward_type = "continuous_scoring_function"
license = "MIT"
license_source = "https://opensource.org/license/mit"

[[outputs]]
path = "/tmp/output/policy.py"
required = true
""")
    (problem_dir / "metadata.json").write_text(
        json.dumps(
            {
                "benchmark": "taiga_task",
                "problem_data": {"instance_id": "native-env-task"},
            }
        )
    )
    (problem_dir / "instruction.md").write_text(
        "# Task\nWrite /tmp/output/policy.py.\n"
    )
    (problem_dir / "environment").mkdir(parents=True)
    (problem_dir / "environment" / "requirements.txt").write_text("gymnasium==0.29.1\n")
    (problem_dir / "environment" / "apt.txt").write_text("libglfw3\n")
    (problem_dir / "scorer" / "data").mkdir(parents=True)
    (problem_dir / "scorer" / "compute_score.py").write_text(
        "from pathlib import Path\n"
        "SUB = Path('/tmp/output')\n"
        "PRIV = Path('/mcp_server/data')\n\n\n"
        "def compute_score(workspace, trajectory, private):\n    return 0.5\n"
    )
    (problem_dir / "scorer" / "requirements.txt").write_text("scikit-learn==1.5.0\n")
    (problem_dir / "scorer" / "env-requirements.txt").write_text("myosuite==2.9.0\n")
    (problem_dir / "calibration.lock.json").write_text("{}\n")
    (problem_dir / "data").mkdir(parents=True)
    (problem_dir / "solution").mkdir(parents=True)
    (problem_dir / "solution" / "solve.sh").write_text("echo ref\n")
    return problem_dir


def test_export_native_env_task(tmp_path: Path) -> None:
    problem_dir = _write_native_env_task(tmp_path / "native-env-task")
    out = tmp_path / "harbor"
    export_harbor(problem_dir, out)

    assert (out / "task.toml").exists()
    assert (out / "instruction.md").exists()
    assert (out / "environment" / "scorer" / "compute_score.py").exists()
    assert (out / "environment" / "calibration.lock.json").exists()
    # The dependency channels the Dockerfile COPYs must ship in the build context.
    assert (out / "environment" / "base" / "install-task-deps.sh").exists()
    assert (out / "environment" / "source_environment" / "requirements.txt").exists()

    dockerfile = (out / "environment" / "Dockerfile").read_text()
    # Hardening parity: the answer key, grader and calibration lock stay root-only.
    assert "chmod 0700 /mcp_server" in dockerfile
    assert "COPY --chown=root:root scorer/data/ /mcp_server/data/" in dockerfile
    assert "COPY --chown=root:root scorer/ /mcp_server/grader/" in dockerfile
    assert (
        "COPY --chown=root:root calibration.lock.json \\\n"
        "    /mcp_server/calibration/calibration.lock.json" in dockerfile
    )
    # The baked lock must be marked as the author fallback so the rubric server
    # refuses it when the export requires the trusted-CI mount.
    assert "/mcp_server/calibration/.author-source" in dockerfile
    # Every declared channel installs through the one hardened installer; the
    # private package names never appear in the agent-visible Dockerfile.
    assert "/tmp/base/install-task-deps.sh /tmp/task-deps" in dockerfile
    assert "myosuite" not in dockerfile
    assert "scikit-learn" not in dockerfile
    assert "@@TASK_EXTRAS@@" not in dockerfile
    # This image is FROM python:3.13-slim, not a native base, so it has to repeat
    # HF_HOME itself -- [[preloaded_files]] derives its mount paths from it, and
    # without it offline from_pretrained/load_dataset looks in the wrong cache.
    assert f"ENV HF_HOME={HF_HOME}" in dockerfile
    workdirs = re.findall(r"(?m)^WORKDIR\s+(\S+)\s*$", dockerfile)
    assert workdirs[-1] == "/workdir"


def test_every_self_contained_template_sets_the_shared_hf_home() -> None:
    """Both Harbor templates must agree with the bases on the HF cache root."""
    for template in (_SELF_CONTAINED_DOCKERFILE, _SOLVER_SELF_CONTAINED_DOCKERFILE):
        assert f"ENV HF_HOME={HF_HOME}" in template


def test_self_contained_templates_do_not_inherit_python_patch_version() -> None:
    """The official Python image exports an exact version uv may not publish yet."""
    for template in (_SELF_CONTAINED_DOCKERFILE, _SOLVER_SELF_CONTAINED_DOCKERFILE):
        assert "ENV PYTHON_VERSION=3.13" in template


def test_export_omits_task_deps_block_when_no_channels_declared(tmp_path: Path) -> None:
    problem_dir = _write_native_env_task(tmp_path / "no-channels")
    for rel in (
        "environment/requirements.txt",
        "environment/apt.txt",
        "scorer/requirements.txt",
        "scorer/env-requirements.txt",
        "calibration.lock.json",
    ):
        (problem_dir / rel).unlink()
    out = tmp_path / "harbor"
    export_harbor(problem_dir, out)

    dockerfile = (out / "environment" / "Dockerfile").read_text()
    assert "install-task-deps.sh" not in dockerfile
    assert "/mcp_server/calibration" not in dockerfile
    assert "@@TASK_EXTRAS@@" not in dockerfile


def test_export_default_mode(template_examples: Path, tmp_path: Path) -> None:
    out = tmp_path / "harbor"
    export_harbor(
        template_examples / "mujoco-pendulum",
        out,
        image_ref="gcr.io/example/p@sha256:abc",
    )

    assert (out / "task.toml").exists()
    assert (out / "instruction.md").exists()
    assert (out / "environment" / "Dockerfile").exists()
    assert (out / "environment" / "scorer" / "compute_score.py").exists()
    assert (out / "environment" / "data").exists()
    assert (out / "environment" / "grader" / "pyproject.toml").exists()
    assert (out / "environment" / "base" / "install-common.sh").exists()
    assert (
        out / "environment" / "taiga_runtime" / "rubric" / "pyproject.toml"
    ).exists()
    assert (out / "solution" / "solve.sh").exists()

    test_sh = (out / "tests" / "test.sh").read_text()
    dockerfile = (out / "environment" / "Dockerfile").read_text()
    assert "COPY data/ /workspace/data/" in dockerfile
    assert "base/requirements-runtime.txt" in dockerfile
    assert "requirements-solvers.txt" not in dockerfile
    assert "install-solvers-heavy.sh" not in dockerfile
    assert "rm -rf /data" in dockerfile
    assert "ln -s /workspace/data /data" in dockerfile
    assert "find /workspace/data -type d -exec chmod 0755" in dockerfile
    assert "find /workspace/data -type f -exec chmod 0644" in dockerfile
    assert "COPY --chown=root:root scorer/data/ /mcp_server/data/" in dockerfile
    assert "COPY --chown=root:root scorer/ /mcp_server/grader/" in dockerfile
    assert "rm -rf /mcp_server/grader/data" in dockerfile
    assert (
        "find /mcp_server/data /mcp_server/grader -type d -exec chmod 0700"
        in dockerfile
    )
    assert (
        "find /mcp_server/data /mcp_server/grader -type f -exec chmod 0600"
        in dockerfile
    )
    workdirs = re.findall(r"(?m)^WORKDIR\s+(\S+)\s*$", dockerfile)
    assert workdirs[-1] == "/workdir"
    assert "/runtime/run_grader.py" in test_sh
    assert "--workspace /tmp/output" in test_sh
    assert "--grader-dir /mcp_server/grader" in test_sh
    assert "--private-dir /mcp_server/data" in test_sh
    assert "--output-dir /logs/verifier" in test_sh

    assert sorted(p.name for p in (out / "tests").iterdir()) == ["test.sh"]

    assert not (out / "scorer").exists()


def test_export_numerical_solver_task_uses_solver_harbor_image(tmp_path: Path) -> None:
    repo_root = Path(__file__).resolve().parents[2]
    problem_dir = tmp_path / "prometheus-cfd"
    shutil.copytree(
        repo_root
        / "alignerr_plugin"
        / "src"
        / "alignerr_plugin"
        / "starter_templates"
        / "prometheus-cfd",
        problem_dir,
    )
    out = tmp_path / "harbor"

    export_harbor(problem_dir, out)

    dockerfile = (out / "environment" / "Dockerfile").read_text()
    task_toml = tomllib.loads((out / "task.toml").read_text())
    assert task_toml["difficulty"]["task_type"] == "cfd"
    assert task_toml["delivery"]["platform"] == "prometheus"
    assert task_toml["agent"]["user"] == "agent"
    assert task_toml["verifier"]["user"] == "root"
    assert "requirements-solvers.txt" in dockerfile
    assert "install-solvers-heavy.sh" in dockerfile
    assert "installing numerical-solver stack" not in dockerfile
    assert (out / "environment" / "base" / "requirements-solvers.txt").exists()
    assert (out / "environment" / "base" / "install-solvers-heavy.sh").exists()
    assert (out / "solution").exists()


def test_export_prometheus_numerical_solver_appends_solver_hint(
    tmp_path: Path,
) -> None:
    repo_root = Path(__file__).resolve().parents[2]
    templates = (
        repo_root / "alignerr_plugin" / "src" / "alignerr_plugin" / "starter_templates"
    )

    for template_name, heading, is_eval in (
        ("prometheus-cfd", "## OpenFOAM Availability", False),
        ("prometheus-structures", "## OpenSees Availability", False),
        ("prometheus-eval-cfd", "## OpenFOAM Availability", True),
        ("prometheus-eval-structures", "## OpenSees Availability", True),
    ):
        problem_dir = tmp_path / template_name
        shutil.copytree(templates / template_name, problem_dir)
        original_instruction = (problem_dir / "instruction.md").read_text()
        out = tmp_path / f"{template_name}-harbor"

        export_harbor(problem_dir, out)

        exported_instruction = (out / "instruction.md").read_text()
        image_instruction = (out / "environment" / "instruction.md").read_text()
        task_toml = tomllib.loads((out / "task.toml").read_text())
        assert exported_instruction.startswith(original_instruction.rstrip("\n"))
        assert image_instruction == exported_instruction
        assert heading in exported_instruction
        assert exported_instruction.count(heading) == 1
        assert task_toml["delivery"].get("eval", False) is is_eval


def test_export_harbor_calls_prometheus_solver_hint_once() -> None:
    source = Path(export_harbor.__code__.co_filename).read_text()

    assert source.count("_append_prometheus_solver_hint(problem_dir, output_dir)") == 1


def test_export_does_not_stamp_remote_image_ref(
    template_examples: Path, tmp_path: Path
) -> None:
    out = tmp_path / "harbor"
    export_harbor(
        template_examples / "mujoco-pendulum",
        out,
        image_ref="gcr.io/example/p@sha256:abc123",
    )
    text = (out / "task.toml").read_text()
    assert "docker_image" not in text


def test_export_gpu_task_writes_runtime_notice_metadata(
    template_examples: Path, tmp_path: Path
) -> None:
    problem_dir = _write_native_ml_task(tmp_path / "mle-gpu")
    original_instruction = (problem_dir / "instruction.md").read_text()
    out = tmp_path / "harbor"

    export_harbor(problem_dir, out)

    assert (out / "instruction.md").read_text() == original_instruction
    root_toml = tomllib.loads((out / "task.toml").read_text())
    image_toml = tomllib.loads((out / "environment" / "task.toml").read_text())
    expected = [
        {
            "kind": "accelerator_availability",
            "accelerator": "gpu",
            "text": "A GPU may be available.",
            "default_enabled": True,
        }
    ]
    assert root_toml["metadata"]["runtime_notices"] == expected
    assert image_toml["metadata"]["runtime_notices"] == expected


def test_export_tpu_task_writes_runtime_notice_metadata(
    template_examples: Path, tmp_path: Path
) -> None:
    problem_dir = tmp_path / "mle-tpu"
    _write_native_ml_task(problem_dir)
    task_toml = problem_dir / "task.toml"
    text = task_toml.read_text()
    text = text.replace(
        'required_resources = "12vcpu+100gib+h100/2"',
        'required_resources = "13vcpu+32gib+tpuv5e1x1"',
    )
    task_toml.write_text(text)
    out = tmp_path / "harbor"

    export_harbor(problem_dir, out)

    root_toml = tomllib.loads((out / "task.toml").read_text())
    assert root_toml["metadata"]["runtime_notices"] == [
        {
            "kind": "accelerator_availability",
            "accelerator": "tpu",
            "text": "A TPU may be available.",
            "default_enabled": True,
        }
    ]


def test_export_preserves_custom_runtime_notices(
    template_examples: Path, tmp_path: Path
) -> None:
    problem_dir = tmp_path / "mle-custom-notice"
    _write_native_ml_task(problem_dir)
    task_toml = problem_dir / "task.toml"
    task_toml.write_text(task_toml.read_text() + """

[metadata]
runtime_notices = [
  { kind = "custom_hint", text = "Use the provided cached dataset." },
]
""")
    out = tmp_path / "harbor"

    export_harbor(problem_dir, out)

    root_toml = tomllib.loads((out / "task.toml").read_text())
    image_toml = tomllib.loads((out / "environment" / "task.toml").read_text())
    expected = [
        {"kind": "custom_hint", "text": "Use the provided cached dataset."},
        {
            "kind": "accelerator_availability",
            "accelerator": "gpu",
            "text": "A GPU may be available.",
            "default_enabled": True,
        },
    ]
    assert root_toml["metadata"]["runtime_notices"] == expected
    assert image_toml["metadata"]["runtime_notices"] == expected


def test_export_removes_stale_accelerator_runtime_notice_for_cpu_task(
    template_examples: Path, tmp_path: Path
) -> None:
    problem_dir = tmp_path / "cpu-stale-notice"
    shutil.copytree(template_examples / "mujoco-pendulum", problem_dir)
    task_toml = problem_dir / "task.toml"
    task_toml.write_text(task_toml.read_text() + """

[metadata]
runtime_notices = [
  { kind = "custom_hint", text = "Keep the simulator deterministic." },
  { kind = "accelerator_availability", accelerator = "gpu", text = "A GPU may be available.", default_enabled = true },
]
""")
    out = tmp_path / "harbor"

    export_harbor(problem_dir, out)

    root_toml = tomllib.loads((out / "task.toml").read_text())
    image_toml = tomllib.loads((out / "environment" / "task.toml").read_text())
    expected = [{"kind": "custom_hint", "text": "Keep the simulator deterministic."}]
    assert root_toml["metadata"]["runtime_notices"] == expected
    assert image_toml["metadata"]["runtime_notices"] == expected


def test_export_can_disable_runtime_notice_metadata(
    template_examples: Path, tmp_path: Path
) -> None:
    out = tmp_path / "harbor"

    export_harbor(
        _write_native_ml_task(tmp_path / "native-disable"),
        out,
        include_runtime_notices=False,
    )

    root_toml = tomllib.loads((out / "task.toml").read_text())
    assert "runtime_notices" not in root_toml.get("metadata", {})
