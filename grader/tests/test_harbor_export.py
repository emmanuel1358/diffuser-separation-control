"""Tests for the Harbor exporter."""

from __future__ import annotations

import json
import re
import shutil
import tomllib
from pathlib import Path

from alignerr_plugin.exporters.harbor import export_harbor

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


def _write_mlenvs_task(problem_dir: Path) -> Path:
    """A minimal env task (metadata.json + prompt.md + test_file.py + data +
    reference_solution) with deps / apt_extras / env_dependencies so the Harbor
    render exercises every block."""
    problem_dir.mkdir(parents=True, exist_ok=True)
    (problem_dir / "metadata.json").write_text(
        json.dumps(
            {
                "ml_task_type": "env",
                "required_resources": "12vcpu+100gib+h100/2",
                "domain": "scientific_discovery_computational_science",
                "license": "MIT",
                "license_source": "https://opensource.org/license/mit",
                "dependencies": ["gymnasium==0.29.1"],
                "apt_extras": ["libglfw3"],
                "env_dependencies": ["myosuite==2.9.0"],
                "grading_dependencies": ["scikit-learn==1.5.0"],
            }
        )
    )
    (problem_dir / "prompt.md").write_text("# Task\nWrite /tmp/output/policy.py.\n")
    (problem_dir / "test_file.py").write_text(
        "from pathlib import Path\n"
        "SUB = Path('/tmp/output')\n"
        "PRIV = Path('/mcp_server/data')\n\n\n"
        "def compute_score():\n    return 0.5\n"
    )
    (problem_dir / "data" / "public").mkdir(parents=True)
    (problem_dir / "data" / "private").mkdir(parents=True)
    (problem_dir / "reference_solution").mkdir(parents=True)
    (problem_dir / "reference_solution" / "solution.py").write_text("print('ref')\n")
    return problem_dir


def test_export_mlenvs_task(tmp_path: Path) -> None:
    problem_dir = _write_mlenvs_task(tmp_path / "mlenvs-env-task")
    out = tmp_path / "harbor"
    export_harbor(problem_dir, out)

    # prompt.md + test_file.py exported; no task.toml/scorer.
    assert (out / "prompt.md").exists()
    assert (out / "test_file.py").exists()
    assert (out / "environment" / "test_file.py").exists()
    assert (out / "environment" / "data" / "public").exists()
    assert (out / "environment" / "data" / "private").exists()
    # The requirements file the Dockerfile COPYs must ship in the build context.
    assert (
        out / "environment" / "base" / "requirements-mlenvs-common.txt"
    ).exists()

    dockerfile = (out / "environment" / "Dockerfile").read_text()
    assert "base/requirements-mlenvs-common.txt" in dockerfile
    # GPU-capable (cu121), not CPU-only.
    assert "cu121" in dockerfile
    # Hardening parity.
    assert "chmod 0700 /mcp_server" in dockerfile
    assert "COPY --chown=root:root data/private/ /mcp_server/data/" in dockerfile
    assert "/mcp_server/grader/compute_score.py" in dockerfile
    # env activation baked so the env server can start.
    assert 'hidden_env = "env"' in dockerfile
    # Declared deps / apt extras / env-only deps are wired.
    assert "gymnasium==0.29.1" in dockerfile
    assert "libglfw3" in dockerfile
    assert "/mcp_server/env_deps" in dockerfile
    assert "myosuite==2.9.0" in dockerfile
    # Grader-only deps render a root-only /mcp_server/grading_deps block, and the
    # placeholder is fully substituted (no @@GRADING_DEPS@@ left in the output).
    assert "/mcp_server/grading_deps" in dockerfile
    assert "scikit-learn==1.5.0" in dockerfile
    assert "@@GRADING_DEPS@@" not in dockerfile
    workdirs = re.findall(r"(?m)^WORKDIR\s+(\S+)\s*$", dockerfile)
    assert workdirs[-1] == "/workdir"


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
    templates = repo_root / "alignerr_plugin" / "src" / "alignerr_plugin" / "starter_templates"

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
