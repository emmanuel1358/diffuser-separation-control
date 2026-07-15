from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pytest

from alignerr_plugin.ground_truth import sha256_file
from alignerr_plugin.local_runtime import LocalBaseImage
from alignerr_plugin.validators.task import validator as validator_module
from alignerr_plugin.validators.task.validator import (
    TaskValidator,
    _PRIVATE_LAYOUT_PROBE,
    reward_hack_lint,
)

_VALID_PROMPT = (
    "Train a classifier on the tabular dataset provided under /data/. "
    "The training and validation splits live in /data/ as CSV files, and you "
    "should read them from there. Fit a model, evaluate it, and write your "
    "predictions for the held-out rows to /tmp/output/result.txt so the grader "
    "can score them. Do not modify any files outside the output directory.\n"
)


_HARDENED_DOCKERFILE = """\
FROM lbx-tasks-base:runtime-ml-core-py313-local
RUN mkdir -p /mcp_server/data /workdir /tmp/output && chown -R 1000:1000 /workdir /tmp/output
COPY ${PROBLEM_DIR}/data/ /data/
COPY --chown=root:root ${PROBLEM_DIR}/scorer/data/ /mcp_server/data/
COPY --chown=root:root ${PROBLEM_DIR}/scorer/ /mcp_server/grader/
RUN rm -rf /mcp_server/grader/data \
    && chown -R root:root /mcp_server/data /mcp_server/grader \
    && find /mcp_server/data /mcp_server/grader -type d -exec chmod 0700 {} + \
    && find /mcp_server/data /mcp_server/grader -type f -exec chmod 0600 {} +
"""


def test_solver_validation_helpers_are_defined_once() -> None:
    source = Path(validator_module.__file__).read_text()

    assert source.count("_SOLVER_LEAK_PATTERNS:") == 1
    assert source.count("_SOLVER_ORACLE_MATERIAL_BY_TASK =") == 1
    assert source.count("def _strip_full_line_shell_comments(") == 1
    assert source.count("def _oracle_solver_material_issues(") == 1


def _default_domain(task_type: str) -> str:
    return {
        "ml": "scientific_discovery_computational_science",
        "mujoco": "model_environment_construction",
        "cfd": "aerodynamics",
        "structures": "seismic_retrofit",
    }[task_type]


def _write_problem(
    problem_dir: Path,
    *,
    task_type: str,
    domain: str | None = None,
    reward_type: str = "multi_deterministic_rubrics",
) -> None:
    (problem_dir / "scorer").mkdir(parents=True)
    (problem_dir / "solution").mkdir()
    (problem_dir / "metadata.json").write_text(
        json.dumps(
            {
                "benchmark": "taiga_task",
                "problem_data": {"instance_id": problem_dir.name},
            }
        )
    )
    ground_truth = ""
    if task_type == "mujoco":
        ground_truth = """
[ground_truth]
render_command = "bash solution/render.sh"
render_outputs = [
  { path = "/tmp/output/rendering.mp4", required = true },
]
"""
        (problem_dir / "solution" / "render.sh").write_text("exit 1\n")
    (problem_dir / "task.toml").write_text(f"""
[task]
name = "labelbox/{problem_dir.name}"

[environment]
required_resources = "4vcpu+16gib"

[difficulty]
task_type = "{task_type}"
domain = "{domain or _default_domain(task_type)}"
reward_type = "{reward_type}"
license = "MIT"
license_source = "https://github.com/owner/dataset/blob/main/LICENSE"

{ground_truth}
[[outputs]]
path = "/tmp/output/result.txt"
required = true
""")
    (problem_dir / "instruction.md").write_text(_VALID_PROMPT)
    (problem_dir / "solution" / "solve.sh").write_text("exit 7\n")
    (problem_dir / "scorer" / "compute_score.py").write_text("""
def compute_score(workspace, trajectory, private):
    return 0.25
""")


def _write_private_layout_problem(problem_dir: Path, dockerfile_text: str = _HARDENED_DOCKERFILE) -> None:
    _write_problem(problem_dir, task_type="ml")
    (problem_dir / "environment").mkdir()
    (problem_dir / "data").mkdir()
    (problem_dir / "scorer" / "data").mkdir()
    (problem_dir / "environment" / "Dockerfile").write_text(dockerfile_text)


def test_solution_answer_key_leak_flags_shell_reading_private_data(
    tmp_path: Path,
) -> None:
    problem_dir = tmp_path / "leak-shell"
    _write_problem(problem_dir, task_type="ml")
    (problem_dir / "solution" / "solve.sh").write_text(
        "#!/usr/bin/env bash\n" "cat /mcp_server/data/labels.json > /tmp/output/result.txt\n"
    )

    stage = TaskValidator()._solution_answer_key_leak(problem_dir)

    assert not stage.passed
    assert any(
        issue.startswith("solution/solve.sh:2:") and "must not read the private answer key" in issue
        for issue in stage.issues
    )


def test_solution_answer_key_leak_flags_python_literal_disk_path(
    tmp_path: Path,
) -> None:
    problem_dir = tmp_path / "leak-python"
    _write_problem(problem_dir, task_type="ml")
    (problem_dir / "solution" / "foo.py").write_text(
        "import json\n" "labels = json.load(open('scorer/data/labels.json'))\n" "print(labels)\n"
    )

    stage = TaskValidator()._solution_answer_key_leak(problem_dir)

    assert not stage.passed
    assert any(issue.startswith("solution/foo.py:2:") and "scorer/data" in issue for issue in stage.issues)


def test_solution_answer_key_leak_passes_clean_reference(tmp_path: Path) -> None:
    problem_dir = tmp_path / "clean-reference"
    _write_problem(problem_dir, task_type="ml")
    (problem_dir / "solution" / "solve.sh").write_text(
        "#!/usr/bin/env bash\n" "set -euo pipefail\n" "python solution/train.py\n"
    )
    (problem_dir / "solution" / "train.py").write_text(
        "import pandas as pd\n"
        "df = pd.read_csv('/data/train.csv')\n"
        "df.head().to_csv('/tmp/output/result.txt', index=False)\n"
    )

    stage = TaskValidator()._solution_answer_key_leak(problem_dir)

    assert stage.passed
    assert stage.issues == []


def test_solution_answer_key_leak_no_ops_without_solution(tmp_path: Path) -> None:
    problem_dir = tmp_path / "no-solution"
    _write_problem(problem_dir, task_type="ml")
    (problem_dir / "solution" / "solve.sh").unlink()
    (problem_dir / "solution").rmdir()

    stage = TaskValidator()._solution_answer_key_leak(problem_dir)

    assert stage.passed
    assert stage.issues == []


def test_validator_fails_failing_ml_solution(tmp_path: Path) -> None:
    problem_dir = tmp_path / "ml-task"
    _write_problem(problem_dir, task_type="ml")

    stage, meta = TaskValidator()._compute_score_return(problem_dir)

    assert not stage.passed
    assert meta["reference_solution_exit"] == 7
    assert any("ground truth solution exited" in issue for issue in stage.issues)


def test_validator_fails_failing_mujoco_solution(tmp_path: Path) -> None:
    problem_dir = tmp_path / "mujoco-task"
    _write_problem(problem_dir, task_type="mujoco")

    stage, meta = TaskValidator()._compute_score_return(problem_dir)

    assert not stage.passed
    assert meta["reference_solution_exit"] == 7
    assert any("ground truth solution exited" in issue for issue in stage.issues)


def test_validator_requires_rendering_mp4_name(tmp_path: Path) -> None:
    problem_dir = tmp_path / "mujoco-task"
    _write_problem(problem_dir, task_type="mujoco")
    task_toml = problem_dir / "task.toml"
    task_toml.write_text(task_toml.read_text().replace("/tmp/output/rendering.mp4", "/tmp/output/preview.mp4"))

    stage = TaskValidator()._ground_truth(problem_dir)

    assert not stage.passed
    assert any("must be named /tmp/output/rendering.mp4" in issue for issue in stage.issues)


def test_validator_does_not_require_render_for_plain_ml(tmp_path: Path) -> None:
    problem_dir = tmp_path / "ml-no-render"
    _write_problem(
        problem_dir,
        task_type="ml",
        reward_type="continuous_scoring_function",
    )

    stage = TaskValidator()._ground_truth(problem_dir)

    assert stage.passed
    assert stage.issues == []


def test_validator_rejects_solver_leak_in_cfd_instruction(tmp_path: Path) -> None:
    problem_dir = tmp_path / "cfd-solver-leak"
    _write_problem(problem_dir, task_type="cfd")
    (problem_dir / "instruction.md").write_text(
        "Design a two-dimensional hydrofoil flap and write the final JSON to "
        "/tmp/output/result.txt. The task should be solved by evaluating lift, "
        "drag, separation behavior, pressure recovery, and robustness across "
        "several deterministic operating points. Use clear engineering judgment "
        "and include only the requested design variables in the output. Run "
        "OpenFOAM before submitting the answer.\n"
    )

    stage = TaskValidator()._prompt_quality(problem_dir)

    assert not stage.passed
    assert any("instruction must be solver-agnostic" in issue for issue in stage.issues)


def test_validator_allows_solver_agnostic_structures_instruction(tmp_path: Path) -> None:
    problem_dir = tmp_path / "structures-solver-agnostic"
    _write_problem(problem_dir, task_type="structures")
    (problem_dir / "instruction.md").write_text(
        "Design a seismic retrofit package for the frame described in the task "
        "files and write the final JSON to /tmp/output/result.txt. The score "
        "rewards reduced peak interstory drift, controlled roof displacement, "
        "balanced demand near the setback transition, reasonable base-shear "
        "growth, budget compliance, and a valid analysis summary. Use the "
        "public frame description to choose robust device locations and sizes, "
        "then submit only the required JSON artifact.\n"
    )

    stage = TaskValidator()._prompt_quality(problem_dir)

    assert stage.passed
    assert stage.issues == []


def test_validator_requires_cfd_oracle_solver_material(tmp_path: Path) -> None:
    problem_dir = tmp_path / "cfd-no-oracle-solver"
    _write_problem(problem_dir, task_type="cfd")
    (problem_dir / "solution" / "solve.sh").write_text(
        "#!/usr/bin/env bash\n"
        "set -euo pipefail\n"
        "mkdir -p /tmp/output\n"
        "python3 - <<'PY'\n"
        "from pathlib import Path\n"
        "Path('/tmp/output/result.txt').write_text('ok')\n"
        "PY\n"
    )

    stage = TaskValidator()._ground_truth(problem_dir)

    assert not stage.passed
    assert any("runnable OpenFOAM material" in issue for issue in stage.issues)


def test_validator_accepts_cfd_oracle_solver_material_in_referenced_file(
    tmp_path: Path,
) -> None:
    problem_dir = tmp_path / "cfd-oracle-solver"
    _write_problem(problem_dir, task_type="cfd")
    (problem_dir / "solution" / "solve.sh").write_text(
        "#!/usr/bin/env bash\nset -euo pipefail\nsolution/oracle_openfoam.sh\n"
    )
    (problem_dir / "solution" / "oracle_openfoam.sh").write_text(
        "#!/usr/bin/env bash\n"
        "set -euo pipefail\n"
        "blockMesh -case /tmp/openfoam_case\n"
        "checkMesh -case /tmp/openfoam_case\n"
        "printf '{}' > /tmp/output/result.txt\n"
    )

    stage = TaskValidator()._ground_truth(problem_dir)

    assert stage.passed
    assert stage.issues == []


def test_validator_accepts_solver_material_in_solve_sh_core_implementation(
    tmp_path: Path,
) -> None:
    problem_dir = tmp_path / "cfd-solve-sh-core"
    _write_problem(problem_dir, task_type="cfd")
    (problem_dir / "solution" / "solve.sh").write_text(
        "#!/usr/bin/env bash\n"
        "set -euo pipefail\n"
        "source /etc/solver-envs.d/openfoam.sh\n"
        "blockMesh -case /tmp/oracle_openfoam_case\n"
        "checkMesh -case /tmp/oracle_openfoam_case\n"
    )

    stage = TaskValidator()._ground_truth(problem_dir)

    assert stage.passed
    assert stage.issues == []


def test_validator_accepts_solver_material_in_transitive_solution_helper(
    tmp_path: Path,
) -> None:
    problem_dir = tmp_path / "structures-transitive-helper"
    _write_problem(problem_dir, task_type="structures")
    (problem_dir / "solution" / "solve.sh").write_text(
        "#!/usr/bin/env bash\nset -euo pipefail\npython3 solution/driver.py\n"
    )
    (problem_dir / "solution" / "driver.py").write_text("import oracle_core\n")
    (problem_dir / "solution" / "oracle_core.py").write_text(
        "import openseespy.opensees as ops\n"
        "ops.model('basic', '-ndm', 2, '-ndf', 3)\n"
        "ops.node(1, 0.0, 0.0)\n"
        "ops.element('elasticBeamColumn', 1, 1, 1, 1.0, 1.0, 1.0, 1)\n"
        "ops.analyze(1)\n"
    )

    stage = TaskValidator()._ground_truth(problem_dir)

    assert stage.passed
    assert stage.issues == []


def test_validator_accepts_solver_material_in_imported_solution_helper(
    tmp_path: Path,
) -> None:
    problem_dir = tmp_path / "structures-imported-helper"
    _write_problem(problem_dir, task_type="structures")
    (problem_dir / "solution" / "solve.sh").write_text(
        "#!/usr/bin/env bash\nset -euo pipefail\npython3 solution/driver.py\n"
    )
    (problem_dir / "solution" / "driver.py").write_text(
        "import oracle_core\noracle_core.run()\n"
    )
    (problem_dir / "solution" / "oracle_core.py").write_text(
        "import openseespy.opensees as ops\n"
        "def run():\n"
        "    ops.model('basic', '-ndm', 2, '-ndf', 3)\n"
        "    ops.node(1, 0.0, 0.0)\n"
        "    ops.element('elasticBeamColumn', 1, 1, 1, 1.0, 1.0, 1.0, 1)\n"
        "    ops.analyze(1)\n"
    )

    stage = TaskValidator()._ground_truth(problem_dir)

    assert stage.passed
    assert stage.issues == []


def test_validator_accepts_solver_material_in_heredoc_imported_solution_helper(
    tmp_path: Path,
) -> None:
    problem_dir = tmp_path / "structures-heredoc-imported-helper"
    _write_problem(problem_dir, task_type="structures")
    (problem_dir / "solution" / "solve.sh").write_text(
        "#!/usr/bin/env bash\n"
        "set -euo pipefail\n"
        "PYTHONPATH=solution python3 - <<'PY'\n"
        "import oracle_core\n"
        "oracle_core.run()\n"
        "PY\n"
    )
    (problem_dir / "solution" / "oracle_core.py").write_text(
        "import openseespy.opensees as ops\n"
        "def run():\n"
        "    ops.model('basic', '-ndm', 2, '-ndf', 3)\n"
        "    ops.node(1, 0.0, 0.0)\n"
        "    ops.analyze(1)\n"
    )

    stage = TaskValidator()._ground_truth(problem_dir)

    assert stage.passed
    assert stage.issues == []


def test_validator_accepts_solver_material_in_qualified_imported_solution_helper(
    tmp_path: Path,
) -> None:
    problem_dir = tmp_path / "structures-qualified-imported-helper"
    _write_problem(problem_dir, task_type="structures")
    (problem_dir / "solution" / "pkg").mkdir()
    (problem_dir / "solution" / "pkg" / "__init__.py").write_text("")
    (problem_dir / "solution" / "solve.sh").write_text(
        "#!/usr/bin/env bash\n"
        "set -euo pipefail\n"
        "PYTHONPATH=solution python3 - <<'PY'\n"
        "from pkg.oracle_core import run\n"
        "run()\n"
        "PY\n"
    )
    (problem_dir / "solution" / "pkg" / "oracle_core.py").write_text(
        "import openseespy.opensees as ops\n"
        "def run():\n"
        "    ops.model('basic', '-ndm', 2, '-ndf', 3)\n"
        "    ops.node(1, 0.0, 0.0)\n"
        "    ops.analyze(1)\n"
    )

    stage = TaskValidator()._ground_truth(problem_dir)

    assert stage.passed
    assert stage.issues == []


def test_validator_accepts_solver_material_in_flat_package_imported_solution_helper(
    tmp_path: Path,
) -> None:
    problem_dir = tmp_path / "structures-flat-package-imported-helper"
    _write_problem(problem_dir, task_type="structures")
    (problem_dir / "solution" / "pkg").mkdir()
    (problem_dir / "solution" / "pkg" / "__init__.py").write_text("")
    (problem_dir / "solution" / "solve.sh").write_text(
        "#!/usr/bin/env bash\n"
        "set -euo pipefail\n"
        "PYTHONPATH=solution python3 solution/driver.py\n"
    )
    (problem_dir / "solution" / "driver.py").write_text(
        "from pkg import oracle_core\noracle_core.run()\n"
    )
    (problem_dir / "solution" / "pkg" / "oracle_core.py").write_text(
        "import openseespy.opensees as ops\n"
        "def run():\n"
        "    ops.model('basic', '-ndm', 2, '-ndf', 3)\n"
        "    ops.analyze(1)\n"
    )

    stage = TaskValidator()._ground_truth(problem_dir)

    assert stage.passed
    assert stage.issues == []


def test_validator_shell_comment_stripping_preserves_heredoc_comments() -> None:
    stripped = validator_module._strip_full_line_shell_comments(
        "# shell-only comment\n"
        "python3 - <<'PY'\n"
        "# heredoc content is not a shell comment\n"
        "print('ok')\n"
        "PY\n"
        "# another shell-only comment\n"
    )

    assert "# shell-only comment" not in stripped
    assert "# another shell-only comment" not in stripped
    assert "# heredoc content is not a shell comment" in stripped


def test_validator_accepts_solver_material_in_public_model_imported_by_solution(
    tmp_path: Path,
) -> None:
    problem_dir = tmp_path / "structures-imported-public-model"
    _write_problem(problem_dir, task_type="structures")
    (problem_dir / "data").mkdir(exist_ok=True)
    (problem_dir / "solution" / "solve.sh").write_text(
        "#!/usr/bin/env bash\nset -euo pipefail\nPYTHONPATH=/data python3 solution/oracle.py\n"
    )
    (problem_dir / "solution" / "oracle.py").write_text(
        "import public_isolation_model as model\nmodel.run_case()\n"
    )
    (problem_dir / "data" / "public_isolation_model.py").write_text(
        "import openseespy.opensees as ops\n"
        "def run_case():\n"
        "    ops.model('basic', '-ndm', 2, '-ndf', 3)\n"
        "    ops.node(1, 0.0, 0.0)\n"
        "    ops.analyze(1)\n"
    )

    stage = TaskValidator()._ground_truth(problem_dir)

    assert stage.passed
    assert stage.issues == []


def test_validator_ignores_unreferenced_solution_solver_files(
    tmp_path: Path,
) -> None:
    problem_dir = tmp_path / "structures-unused-helper"
    _write_problem(problem_dir, task_type="structures")
    (problem_dir / "solution" / "solve.sh").write_text(
        "#!/usr/bin/env bash\nset -euo pipefail\nprintf '{}' > /tmp/output/result.txt\n"
    )
    (problem_dir / "solution" / "unused_oracle.py").write_text(
        "import openseespy.opensees as ops\n"
        "ops.model('basic', '-ndm', 2, '-ndf', 3)\n"
        "ops.analyze(1)\n"
    )

    stage = TaskValidator()._ground_truth(problem_dir)

    assert not stage.passed
    assert any("runnable OpenSeesPy material" in issue for issue in stage.issues)


def test_validator_ignores_substring_file_references(tmp_path: Path) -> None:
    problem_dir = tmp_path / "structures-substring-reference"
    _write_problem(problem_dir, task_type="structures")
    (problem_dir / "solution" / "solve.sh").write_text(
        "#!/usr/bin/env bash\n"
        "set -euo pipefail\n"
        "printf 'not a reference: oracle_core.py' > /tmp/output/result.txt\n"
    )
    (problem_dir / "solution" / "oracle_core.py").write_text(
        "import openseespy.opensees as ops\n"
        "ops.model('basic', '-ndm', 2, '-ndf', 3)\n"
        "ops.analyze(1)\n"
    )

    stage = TaskValidator()._ground_truth(problem_dir)

    assert not stage.passed
    assert any("runnable OpenSeesPy material" in issue for issue in stage.issues)


def test_validator_rejects_openfoam_command_string_literals(
    tmp_path: Path,
) -> None:
    problem_dir = tmp_path / "cfd-command-string-literal"
    _write_problem(problem_dir, task_type="cfd")
    (problem_dir / "solution" / "solve.sh").write_text(
        "#!/usr/bin/env bash\nset -euo pipefail\npython3 solution/oracle.py\n"
    )
    (problem_dir / "solution" / "oracle.py").write_text(
        "from pathlib import Path\n"
        "COMMAND = 'blockMesh && simpleFoam'\n"
        "Path('/tmp/output/result.txt').write_text('{}')\n"
    )

    stage = TaskValidator()._ground_truth(problem_dir)

    assert not stage.passed
    assert any("runnable OpenFOAM material" in issue for issue in stage.issues)


def test_validator_ignores_ambiguous_duplicate_basename(
    tmp_path: Path,
) -> None:
    problem_dir = tmp_path / "structures-duplicate-basename"
    _write_problem(problem_dir, task_type="structures")
    (problem_dir / "solution" / "a").mkdir()
    (problem_dir / "solution" / "b").mkdir()
    (problem_dir / "solution" / "solve.sh").write_text(
        "#!/usr/bin/env bash\nset -euo pipefail\npython3 solution/a/driver.py\n"
    )
    (problem_dir / "solution" / "a" / "driver.py").write_text(
        "from pathlib import Path\nPath('/tmp/output/result.txt').write_text('{}')\n"
    )
    (problem_dir / "solution" / "b" / "driver.py").write_text(
        "import openseespy.opensees as ops\n"
        "ops.model('basic', '-ndm', 2, '-ndf', 3)\n"
        "ops.analyze(1)\n"
    )

    stage = TaskValidator()._ground_truth(problem_dir)

    assert not stage.passed
    assert any("runnable OpenSeesPy material" in issue for issue in stage.issues)


def test_validator_rejects_openfoam_mock_command_substrings(
    tmp_path: Path,
) -> None:
    problem_dir = tmp_path / "cfd-mock-command-substring"
    _write_problem(problem_dir, task_type="cfd")
    (problem_dir / "solution" / "solve.sh").write_text(
        "#!/usr/bin/env bash\nset -euo pipefail\npython3 solution/oracle.py\n"
    )
    (problem_dir / "solution" / "oracle.py").write_text(
        "class _MockCheckMeshResult:\n"
        "    pass\n"
        "MESSAGE = 'OpenFOAM-compatible mock, no executable solver command'\n"
    )

    stage = TaskValidator()._ground_truth(problem_dir)

    assert not stage.passed
    assert any("runnable OpenFOAM material" in issue for issue in stage.issues)


def test_validator_rejects_openfoam_env_setup_without_solver_action(
    tmp_path: Path,
) -> None:
    problem_dir = tmp_path / "cfd-source-only"
    _write_problem(problem_dir, task_type="cfd")
    (problem_dir / "solution" / "solve.sh").write_text(
        "#!/usr/bin/env bash\n"
        "set -euo pipefail\n"
        "source /etc/solver-envs.d/openfoam.sh\n"
        "printf '{}' > /tmp/output/result.txt\n"
    )

    stage = TaskValidator()._ground_truth(problem_dir)

    assert not stage.passed
    assert any("runnable OpenFOAM material" in issue for issue in stage.issues)


def test_validator_accepts_openfoam_chained_shell_commands(tmp_path: Path) -> None:
    problem_dir = tmp_path / "cfd-chained-shell-commands"
    _write_problem(problem_dir, task_type="cfd")
    (problem_dir / "solution" / "solve.sh").write_text(
        "#!/usr/bin/env bash\n"
        "set -euo pipefail\n"
        "bash -lc 'source /etc/solver-envs.d/openfoam.sh && blockMesh -case /tmp/case && checkMesh -case /tmp/case'\n"
        "printf '{}' > /tmp/output/result.txt\n"
    )

    stage = TaskValidator()._ground_truth(problem_dir)

    assert stage.passed
    assert stage.issues == []


def test_validator_accepts_openfoam_indented_shell_commands(tmp_path: Path) -> None:
    problem_dir = tmp_path / "cfd-indented-shell-commands"
    _write_problem(problem_dir, task_type="cfd")
    (problem_dir / "solution" / "solve.sh").write_text(
        "#!/usr/bin/env bash\n"
        "set -euo pipefail\n"
        "if command -v blockMesh >/dev/null 2>&1; then\n"
        "  blockMesh -help >/dev/null 2>&1 || true\n"
        "fi\n"
        "printf '{}' > /tmp/output/result.txt\n"
    )

    stage = TaskValidator()._ground_truth(problem_dir)

    assert stage.passed
    assert stage.issues == []


def test_validator_accepts_openfoam_python_subprocess_calls(tmp_path: Path) -> None:
    problem_dir = tmp_path / "cfd-python-subprocess"
    _write_problem(problem_dir, task_type="cfd")
    (problem_dir / "solution" / "solve.sh").write_text(
        "#!/usr/bin/env bash\nset -euo pipefail\npython3 solution/oracle.py\n"
    )
    (problem_dir / "solution" / "oracle.py").write_text(
        "import subprocess\n"
        "completed = subprocess.run(['blockMesh', '-help'], check=False)\n"
    )

    stage = TaskValidator()._ground_truth(problem_dir)

    assert stage.passed
    assert stage.issues == []


def test_validator_rejects_opensees_import_without_api_action(
    tmp_path: Path,
) -> None:
    problem_dir = tmp_path / "structures-import-only"
    _write_problem(problem_dir, task_type="structures")
    (problem_dir / "solution" / "solve.sh").write_text(
        "#!/usr/bin/env bash\nset -euo pipefail\npython3 solution/oracle.py\n"
    )
    (problem_dir / "solution" / "oracle.py").write_text(
        "import openseespy.opensees as ops\n"
        "from pathlib import Path\n"
        "Path('/tmp/output/result.txt').write_text('{}')\n"
    )

    stage = TaskValidator()._ground_truth(problem_dir)

    assert not stage.passed
    assert any("runnable OpenSeesPy material" in issue for issue in stage.issues)


def test_validator_rejects_opensees_prose_api_mentions(
    tmp_path: Path,
) -> None:
    problem_dir = tmp_path / "structures-prose-api-mentions"
    _write_problem(problem_dir, task_type="structures")
    (problem_dir / "solution" / "solve.sh").write_text(
        "#!/usr/bin/env bash\nset -euo pipefail\npython3 solution/oracle.py\n"
    )
    (problem_dir / "solution" / "oracle.py").write_text(
        "import openseespy.opensees as ops\n"
        "MESSAGE = 'ops.model(...) and ops.analyze(...) are documented here only'\n"
    )

    stage = TaskValidator()._ground_truth(problem_dir)

    assert not stage.passed
    assert any("runnable OpenSeesPy material" in issue for issue in stage.issues)


def test_validator_ignores_scorer_only_solver_use(tmp_path: Path) -> None:
    problem_dir = tmp_path / "structures-scorer-only"
    _write_problem(problem_dir, task_type="structures")
    (problem_dir / "solution" / "solve.sh").write_text(
        "#!/usr/bin/env bash\n"
        "set -euo pipefail\n"
        "mkdir -p /tmp/output\n"
        "printf '{}' > /tmp/output/result.txt\n"
    )
    (problem_dir / "scorer" / "compute_score.py").write_text(
        "import openseespy.opensees as ops\n"
        "def compute_score(workspace, trajectory, private):\n"
        "    ops.model('basic', '-ndm', 2, '-ndf', 3)\n"
        "    return 1.0\n"
    )

    stage = TaskValidator()._ground_truth(problem_dir)

    assert not stage.passed
    assert any("scorer solver use" in issue for issue in stage.issues)


def test_validator_accepts_structures_oracle_solver_material_in_referenced_file(
    tmp_path: Path,
) -> None:
    problem_dir = tmp_path / "structures-oracle-solver"
    _write_problem(problem_dir, task_type="structures")
    (problem_dir / "solution" / "solve.sh").write_text(
        "#!/usr/bin/env bash\nset -euo pipefail\npython3 solution/oracle_search.py\n"
    )
    (problem_dir / "solution" / "oracle_search.py").write_text(
        "import openseespy.opensees as ops\n"
        "ops.model('basic', '-ndm', 2, '-ndf', 3)\n"
        "ops.node(1, 0.0, 0.0)\n"
        "ops.element('elasticBeamColumn', 1, 1, 1, 1.0, 1.0, 1.0, 1)\n"
        "ops.analyze(1)\n"
    )

    stage = TaskValidator()._ground_truth(problem_dir)

    assert stage.passed
    assert stage.issues == []


def test_validator_rejects_invalid_metadata_enums(tmp_path: Path) -> None:
    problem_dir = tmp_path / "bad-metadata"
    _write_problem(
        problem_dir,
        task_type="ml",
        domain="model_environment_construction",
        reward_type="continuous_scoring_function",
    )

    stage = TaskValidator()._schema(problem_dir)

    assert not stage.passed
    assert any("[difficulty].domain" in issue for issue in stage.issues)


def test_validator_rejects_missing_reward_type(tmp_path: Path) -> None:
    problem_dir = tmp_path / "missing-reward"
    _write_problem(problem_dir, task_type="ml")
    task_toml = problem_dir / "task.toml"
    task_toml.write_text(
        "\n".join(line for line in task_toml.read_text().splitlines() if not line.startswith("reward_type")) + "\n"
    )

    stage = TaskValidator()._schema(problem_dir)

    assert not stage.passed
    assert any("[difficulty].reward_type" in issue for issue in stage.issues)


def test_validator_uses_committed_render_artifact_proof(tmp_path: Path) -> None:
    problem_dir = tmp_path / "mujoco-task"
    _write_problem(problem_dir, task_type="mujoco")
    (problem_dir / "solution" / "solve.sh").write_text("mkdir -p /tmp/output\nprintf ok > /tmp/output/result.txt\n")
    (problem_dir / "scorer" / "compute_score.py").write_text("""
def compute_score(workspace, trajectory, private):
    return 1.0 if (workspace / "result.txt").exists() else 0.0
""")
    artifact = problem_dir / ".alignerr" / "ground_truth" / "rendering.mp4"
    artifact.parent.mkdir(parents=True)
    artifact.write_bytes(b"video")
    (problem_dir / ".alignerr" / "build_proof.json").write_text(
        json.dumps(
            {
                "ground_truth_result": {
                    "review_artifacts": [
                        {
                            "path": ".alignerr/ground_truth/rendering.mp4",
                            "logical_path": "/tmp/output/rendering.mp4",
                            "sha256": sha256_file(artifact),
                            "bytes": artifact.stat().st_size,
                            "width": 1280,
                            "height": 720,
                        }
                    ]
                }
            }
        )
    )

    stage, meta = TaskValidator()._compute_score_return(problem_dir)

    assert stage.passed
    assert meta["ground_truth_passed"]
    assert meta["review_artifacts"] == ["/tmp/output/rendering.mp4"]


def test_private_data_layout_accepts_hardened_dockerfile(tmp_path: Path) -> None:
    problem_dir = tmp_path / "private-layout"
    _write_private_layout_problem(problem_dir)

    stage = TaskValidator()._private_data_layout(problem_dir)

    assert stage.passed
    assert stage.issues == []


def test_private_data_layout_rejects_private_copy_to_public_data(
    tmp_path: Path,
) -> None:
    problem_dir = tmp_path / "private-layout"
    _write_private_layout_problem(
        problem_dir,
        """\
FROM lbx-tasks-base:runtime-ml-core-py313-local
COPY ${PROBLEM_DIR}/scorer/data/ /data/
""",
    )

    stage = TaskValidator()._private_data_layout(problem_dir)

    assert not stage.passed
    assert any("public/model-writable path" in issue for issue in stage.issues)


def test_private_data_layout_rejects_json_form_private_copy_to_public_data(
    tmp_path: Path,
) -> None:
    problem_dir = tmp_path / "private-layout"
    _write_private_layout_problem(
        problem_dir,
        """\
FROM lbx-tasks-base:runtime-ml-core-py313-local
COPY ["${PROBLEM_DIR}/scorer/data/", "/data/"]
""",
    )

    stage = TaskValidator()._private_data_layout(problem_dir)

    assert not stage.passed
    assert any("public/model-writable path" in issue for issue in stage.issues)


def test_private_data_layout_rejects_scorer_copy_to_public_data(
    tmp_path: Path,
) -> None:
    problem_dir = tmp_path / "private-layout"
    _write_private_layout_problem(
        problem_dir,
        """\
FROM lbx-tasks-base:runtime-ml-core-py313-local
COPY ${PROBLEM_DIR}/scorer/ /workspace/scorer/
""",
    )

    stage = TaskValidator()._private_data_layout(problem_dir)

    assert not stage.passed
    assert any("public/model-writable path" in issue for issue in stage.issues)


def test_private_data_layout_accepts_public_path_with_scorer_substring(
    tmp_path: Path,
) -> None:
    problem_dir = tmp_path / "private-layout"
    _write_private_layout_problem(
        problem_dir,
        """\
FROM lbx-tasks-base:runtime-ml-core-py313-local
COPY ${PROBLEM_DIR}/nonscorer-assets/ /data/nonscorer-assets/
COPY --link ${PROBLEM_DIR}/score_reports/ /workspace/score_reports/
""",
    )

    stage = TaskValidator()._private_data_layout(problem_dir)

    assert stage.passed
    assert stage.issues == []


def test_private_data_layout_rejects_copy_agent_chown_private_roots(
    tmp_path: Path,
) -> None:
    problem_dir = tmp_path / "private-layout"
    _write_private_layout_problem(
        problem_dir,
        """\
FROM lbx-tasks-base:runtime-ml-core-py313-local
COPY --chown=1000:1000 ${PROBLEM_DIR}/scorer/data/ /mcp_server/data/
""",
    )

    stage = TaskValidator()._private_data_layout(problem_dir)

    assert not stage.passed
    assert any("COPY --chown" in issue for issue in stage.issues)


def test_private_data_layout_rejects_copy_public_chmod_private_roots(
    tmp_path: Path,
) -> None:
    problem_dir = tmp_path / "private-layout"
    _write_private_layout_problem(
        problem_dir,
        """\
FROM lbx-tasks-base:runtime-ml-core-py313-local
COPY --chmod=0755 ${PROBLEM_DIR}/scorer/data/ /mcp_server/data/
""",
    )

    stage = TaskValidator()._private_data_layout(problem_dir)

    assert not stage.passed
    assert any("COPY --chmod" in issue for issue in stage.issues)


def test_private_data_layout_rejects_json_form_copy_flags_private_roots(
    tmp_path: Path,
) -> None:
    problem_dir = tmp_path / "private-layout"
    _write_private_layout_problem(
        problem_dir,
        """\
FROM lbx-tasks-base:runtime-ml-core-py313-local
COPY --chown=agent:agent --chmod=0755 ["${PROBLEM_DIR}/scorer/data/", "/mcp_server/data/"]
""",
    )

    stage = TaskValidator()._private_data_layout(problem_dir)

    assert not stage.passed
    assert any("COPY --chown" in issue for issue in stage.issues)
    assert any("COPY --chmod" in issue for issue in stage.issues)


def test_private_data_layout_accepts_root_user_with_nonroot_group(
    tmp_path: Path,
) -> None:
    problem_dir = tmp_path / "private-layout"
    _write_private_layout_problem(
        problem_dir,
        """\
FROM lbx-tasks-base:runtime-ml-core-py313-local
COPY --chown=root:1000 --chmod=0600 ${PROBLEM_DIR}/scorer/data/ /mcp_server/data/
RUN chown -R root:1000 /mcp_server/data
""",
    )

    stage = TaskValidator()._private_data_layout(problem_dir)

    assert stage.passed
    assert stage.issues == []


def test_private_data_layout_rejects_nonroot_owner_private_roots(
    tmp_path: Path,
) -> None:
    problem_dir = tmp_path / "private-layout"
    _write_private_layout_problem(
        problem_dir,
        """\
FROM lbx-tasks-base:runtime-ml-core-py313-local
RUN chown -R 10000:1000 /mcp_server/grader
""",
    )

    stage = TaskValidator()._private_data_layout(problem_dir)

    assert not stage.passed
    assert any("remain root-owned" in issue for issue in stage.issues)


def test_private_data_layout_rejects_agent_owned_private_roots(
    tmp_path: Path,
) -> None:
    problem_dir = tmp_path / "private-layout"
    _write_private_layout_problem(
        problem_dir,
        """\
FROM lbx-tasks-base:runtime-ml-core-py313-local
RUN chown -R 1000:1000 /mcp_server/data /mcp_server/grader
""",
    )

    stage = TaskValidator()._private_data_layout(problem_dir)

    assert not stage.passed
    assert any("remain root-owned" in issue for issue in stage.issues)


def test_private_data_layout_rejects_group_world_readable_private_roots(
    tmp_path: Path,
) -> None:
    problem_dir = tmp_path / "private-layout"
    _write_private_layout_problem(
        problem_dir,
        """\
FROM lbx-tasks-base:runtime-ml-core-py313-local
RUN chmod -R 0755 /mcp_server/data
""",
    )

    stage = TaskValidator()._private_data_layout(problem_dir)

    assert not stage.passed
    assert any("grants group/world access" in issue for issue in stage.issues)


def test_private_data_layout_rejects_group_world_traversable_mcp_parent(
    tmp_path: Path,
) -> None:
    problem_dir = tmp_path / "private-layout"
    _write_private_layout_problem(
        problem_dir,
        """\
FROM lbx-tasks-base:runtime-ml-core-py313-local
RUN chmod 0755 /mcp_server
""",
    )

    stage = TaskValidator()._private_data_layout(problem_dir)

    assert not stage.passed
    assert any("group/world traversal" in issue for issue in stage.issues)


def test_private_data_layout_rejects_symbolic_group_world_private_chmod(
    tmp_path: Path,
) -> None:
    problem_dir = tmp_path / "private-layout"
    _write_private_layout_problem(
        problem_dir,
        """\
FROM lbx-tasks-base:runtime-ml-core-py313-local
RUN chmod -R a+rX /mcp_server/data
""",
    )

    stage = TaskValidator()._private_data_layout(problem_dir)

    assert not stage.passed
    assert any("grants group/world access" in issue for issue in stage.issues)


def test_private_data_layout_accepts_symbolic_private_chmod_removal(
    tmp_path: Path,
) -> None:
    problem_dir = tmp_path / "private-layout"
    _write_private_layout_problem(
        problem_dir,
        """\
FROM lbx-tasks-base:runtime-ml-core-py313-local
RUN chmod -R go-rwx /mcp_server/data
""",
    )

    stage = TaskValidator()._private_data_layout(problem_dir)

    assert stage.passed
    assert stage.issues == []


def test_private_data_layout_ignores_nonprivate_path_substrings(
    tmp_path: Path,
) -> None:
    problem_dir = tmp_path / "private-layout"
    _write_private_layout_problem(
        problem_dir,
        """\
FROM lbx-tasks-base:runtime-ml-core-py313-local
RUN chmod -R 0755 /workdir/not_mcp_server/data_backup
RUN chown -R 1000:1000 /workdir/not_mcp_server/grader_backup
""",
    )

    stage = TaskValidator()._private_data_layout(problem_dir)

    assert stage.passed
    assert stage.issues == []


def test_private_data_layout_rejects_sensitive_public_private_duplicate(
    tmp_path: Path,
) -> None:
    problem_dir = tmp_path / "private-layout"
    _write_private_layout_problem(problem_dir)
    fixture = b'{"scenario": "held-out"}\n'
    (problem_dir / "data" / "visible.json").write_bytes(fixture)
    (problem_dir / "scorer" / "data" / "hidden_scenarios.json").write_bytes(fixture)

    stage = TaskValidator()._private_data_layout(problem_dir)

    assert not stage.passed
    assert any("duplicates public data byte-for-byte" in issue for issue in stage.issues)


def test_private_data_layout_allows_unsensitive_substring_duplicate(
    tmp_path: Path,
) -> None:
    problem_dir = tmp_path / "private-layout"
    _write_private_layout_problem(problem_dir)
    fixture = b'{"scenario": "shared-public-fixture"}\n'
    (problem_dir / "data" / "visible.json").write_bytes(fixture)
    (problem_dir / "scorer" / "data" / "unexpected.json").write_bytes(fixture)

    stage = TaskValidator()._private_data_layout(problem_dir)

    assert stage.passed
    assert stage.issues == []


def test_private_layout_probe_avoids_public_filename_false_positives() -> None:
    assert "hidden-looking fixture in public root" not in _PRIVATE_LAYOUT_PROBE
    assert "has_sensitive_name" not in _PRIVATE_LAYOUT_PROBE


def test_private_layout_probe_skips_symlink_mode_bits() -> None:
    assert "stat.S_ISLNK(st.st_mode)" in _PRIVATE_LAYOUT_PROBE


def _write_build_proof_problem(tmp_path: Path) -> Path:
    problem_dir = tmp_path / "repo" / "problems" / "private-layout"
    _write_private_layout_problem(problem_dir)
    return problem_dir


def test_local_build_proof_runs_private_layout_and_agent_python_image_probes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    problem_dir = _write_build_proof_problem(tmp_path)
    run_calls: list[list[str]] = []
    proof_calls: list[dict[str, object]] = []

    def fake_run(args: list[str], **_kwargs: object) -> subprocess.CompletedProcess[str]:
        run_calls.append(args)
        if args[:3] == ["docker", "buildx", "build"]:
            Path(args[args.index("--iidfile") + 1]).write_text("sha256:local")
            return subprocess.CompletedProcess(args, 0, "", "")
        return subprocess.CompletedProcess(args, 0, "", "")

    monkeypatch.setattr(
        validator_module,
        "verify_build_proof",
        lambda _problem_dir: (False, ["missing build proof"], None),
    )
    monkeypatch.setattr(
        validator_module,
        "ensure_local_base_image",
        lambda _repo_root, _problem_dir: LocalBaseImage("lbx-tasks-base", "local", Path("Dockerfile")),
    )
    monkeypatch.setattr(
        validator_module,
        "write_build_proof",
        lambda _problem_dir, **kwargs: proof_calls.append(kwargs),
    )
    monkeypatch.setattr(validator_module.subprocess, "run", fake_run)

    stage = TaskValidator()._local_build_proof(problem_dir)

    assert stage.passed
    docker_runs = [call for call in run_calls if call[:3] == ["docker", "run", "--rm"]]
    assert [call[call.index("--user") + 1] for call in docker_runs] == [
        "root",
        "1000:1000",
    ]
    assert proof_calls[0]["image_digest"] == "sha256:local"


def test_local_build_proof_reports_private_layout_image_probe_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    problem_dir = _write_build_proof_problem(tmp_path)
    proof_calls: list[dict[str, object]] = []

    def fake_run(args: list[str], **_kwargs: object) -> subprocess.CompletedProcess[str]:
        if args[:3] == ["docker", "buildx", "build"]:
            Path(args[args.index("--iidfile") + 1]).write_text("sha256:local")
            return subprocess.CompletedProcess(args, 0, "", "")
        return subprocess.CompletedProcess(args, 1, "", "agent can read private root")

    monkeypatch.setattr(
        validator_module,
        "verify_build_proof",
        lambda _problem_dir: (False, ["missing build proof"], None),
    )
    monkeypatch.setattr(
        validator_module,
        "ensure_local_base_image",
        lambda _repo_root, _problem_dir: LocalBaseImage("lbx-tasks-base", "local", Path("Dockerfile")),
    )
    monkeypatch.setattr(
        validator_module,
        "write_build_proof",
        lambda _problem_dir, **kwargs: proof_calls.append(kwargs),
    )
    monkeypatch.setattr(validator_module.subprocess, "run", fake_run)

    stage = TaskValidator()._local_build_proof(problem_dir)

    assert not stage.passed
    assert any("private data layout image probe failed" in issue for issue in stage.issues)
    assert proof_calls == []


def test_local_build_proof_reports_agent_python_image_probe_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    problem_dir = _write_build_proof_problem(tmp_path)
    proof_calls: list[dict[str, object]] = []

    def fake_run(args: list[str], **_kwargs: object) -> subprocess.CompletedProcess[str]:
        if args[:3] == ["docker", "buildx", "build"]:
            Path(args[args.index("--iidfile") + 1]).write_text("sha256:local")
            return subprocess.CompletedProcess(args, 0, "", "")
        if args[:3] == ["docker", "run", "--rm"] and args[args.index("--user") + 1] == "root":
            return subprocess.CompletedProcess(args, 0, "", "")
        return subprocess.CompletedProcess(args, 126, "", "python: Permission denied")

    monkeypatch.setattr(
        validator_module,
        "verify_build_proof",
        lambda _problem_dir: (False, ["missing build proof"], None),
    )
    monkeypatch.setattr(
        validator_module,
        "ensure_local_base_image",
        lambda _repo_root, _problem_dir: LocalBaseImage("lbx-tasks-base", "local", Path("Dockerfile")),
    )
    monkeypatch.setattr(
        validator_module,
        "write_build_proof",
        lambda _problem_dir, **kwargs: proof_calls.append(kwargs),
    )
    monkeypatch.setattr(validator_module.subprocess, "run", fake_run)

    stage = TaskValidator()._local_build_proof(problem_dir)

    assert not stage.passed
    assert any("agent python image probe failed" in issue for issue in stage.issues)
    assert proof_calls == []


def test_grader_sandbox_flags_dynamic_exec_of_model_code(tmp_path: Path) -> None:
    problem_dir = tmp_path / "sandbox-task"
    _write_problem(problem_dir, task_type="ml")
    (problem_dir / "scorer" / "compute_score.py").write_text("""
import importlib.util


def compute_score(workspace, trajectory, private):
    spec = importlib.util.spec_from_file_location("p", workspace / "proc.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.run()
""")

    stage = TaskValidator()._grader_sandbox(problem_dir)

    assert not stage.passed
    assert any("spec_from_file_location" in issue for issue in stage.issues)
    assert any("exec_module" in issue for issue in stage.issues)
    assert any("run_policy" in issue for issue in stage.issues)


def test_grader_sandbox_passes_for_run_policy(tmp_path: Path) -> None:
    problem_dir = tmp_path / "sandbox-ok"
    _write_problem(problem_dir, task_type="ml")
    (problem_dir / "scorer" / "compute_score.py").write_text("""
from grading import helpers


def compute_score(workspace, trajectory, private):
    with helpers.run_policy(workspace) as policy:
        action = policy.act({})
    return 1.0 if action else 0.0
""")

    stage = TaskValidator()._grader_sandbox(problem_dir)

    assert stage.passed
    assert stage.issues == []


def test_agent_fault_flags_broad_except_returning_score() -> None:
    src = (
        "def compute_score(workspace, trajectory, private):\n"
        "    try:\n"
        "        return _run(workspace)\n"
        "    except Exception:\n"
        "        return 0.0\n"
        "\n"
        "def _run(workspace):\n"
        "    return 1.0\n"
    )
    issues = validator_module._agent_fault_issues("scorer/compute_score.py", src)
    assert any("broad `except`" in issue for issue in issues)


def test_agent_fault_flags_failure_helper_in_broad_except() -> None:
    src = (
        "def compute_score(workspace, trajectory, private):\n"
        "    try:\n"
        "        return _run(workspace)\n"
        "    except Exception as exc:\n"
        "        return _failure(str(exc))\n"
        "\n"
        "def _failure(msg):\n"
        "    return {'score': 0.0}\n"
        "\n"
        "def _run(workspace):\n"
        "    return 1.0\n"
    )
    issues = validator_module._agent_fault_issues("scorer/compute_score.py", src)
    assert any("broad `except`" in issue for issue in issues)


def test_agent_fault_accepts_broad_except_that_raises_agentfault() -> None:
    src = (
        "from grading import AgentFault\n"
        "\n"
        "def compute_score(workspace, trajectory, private):\n"
        "    try:\n"
        "        return _run(workspace)\n"
        "    except Exception as exc:\n"
        "        raise AgentFault(str(exc)) from exc\n"
        "\n"
        "def _run(workspace):\n"
        "    return 1.0\n"
    )
    assert validator_module._agent_fault_issues("scorer/compute_score.py", src) == []


def test_agent_fault_spares_sentinel_returning_helper() -> None:
    src = (
        "import numpy as np\n"
        "\n"
        "def compute_score(workspace, trajectory, private):\n"
        "    return float(_coerce(workspace)[1])\n"
        "\n"
        "def _coerce(raw):\n"
        "    try:\n"
        "        action = np.asarray(raw, dtype=float)\n"
        "    except Exception:\n"
        "        return np.zeros(3), False\n"
        "    return action, True\n"
    )
    assert validator_module._agent_fault_issues("scorer/compute_score.py", src) == []


def test_agent_fault_flags_pickle_load_of_agent_artifact() -> None:
    src = (
        "import pickle\n"
        "\n"
        "def compute_score(workspace, trajectory, private):\n"
        "    fh = open(workspace / 'model.pkl', 'rb')\n"
        "    model = pickle.load(fh)\n"
        "    return float(model.predict())\n"
    )
    issues = validator_module._agent_fault_issues("scorer/compute_score.py", src)
    assert any("deserializes pickle" in issue for issue in issues)


def test_agent_fault_flags_from_import_joblib_load_of_agent_artifact() -> None:
    # `from joblib import load; load(<agent path>)` -- the pickle-ban lint must
    # resolve the from-import, not only the module-qualified `joblib.load` form.
    src = (
        "from joblib import load\n"
        "\n"
        "def compute_score(workspace, trajectory, private):\n"
        "    model = load('/tmp/output/model.joblib')\n"
        "    return float(model.predict())\n"
    )
    issues = validator_module._agent_fault_issues("scorer/compute_score.py", src)
    assert any("deserializes pickle" in issue for issue in issues)


def test_agent_fault_flags_aliased_torch_load_of_agent_artifact() -> None:
    # `from torch import load as tload; tload(<agent path>)` -- the `as` alias
    # must be resolved to the underlying deserializer.
    src = (
        "from torch import load as tload\n"
        "\n"
        "def compute_score(workspace, trajectory, private):\n"
        "    model = tload('/tmp/output/model.pt')\n"
        "    return float(model.predict())\n"
    )
    issues = validator_module._agent_fault_issues("scorer/compute_score.py", src)
    assert any("deserializes pickle" in issue for issue in issues)


def test_agent_fault_flags_aliased_pickle_module_load() -> None:
    # `import pickle as p; p.load(fh)` where fh is an agent-writable handle.
    src = (
        "import pickle as p\n"
        "\n"
        "def compute_score(workspace, trajectory, private):\n"
        "    fh = open('/tmp/output/model.pkl', 'rb')\n"
        "    model = p.load(fh)\n"
        "    return float(model.predict())\n"
    )
    issues = validator_module._agent_fault_issues("scorer/compute_score.py", src)
    assert any("deserializes pickle" in issue for issue in issues)


def test_agent_fault_flags_from_import_pickle_loads_of_agent_bytes() -> None:
    # `from pickle import loads; loads(raw)` where raw came from an agent path.
    src = (
        "from pickle import loads\n"
        "\n"
        "def compute_score(workspace, trajectory, private):\n"
        "    raw = open('/tmp/output/blob.pkl', 'rb').read()\n"
        "    model = loads(raw)\n"
        "    return float(model.predict())\n"
    )
    issues = validator_module._agent_fault_issues("scorer/compute_score.py", src)
    assert any("deserializes pickle" in issue for issue in issues)


def test_agent_fault_allows_from_import_load_of_trusted_truth() -> None:
    # The alias resolution must NOT create a false positive: a from-imported
    # deserializer reading trusted held-out truth under /mcp_server is legitimate.
    src = (
        "from joblib import load\n"
        "\n"
        "def compute_score(workspace, trajectory, private):\n"
        "    truth = load('/mcp_server/data/truth.joblib')\n"
        "    return float(truth is not None)\n"
    )
    issues = validator_module._agent_fault_issues("scorer/compute_score.py", src)
    assert not any("deserializes pickle" in issue for issue in issues)


def test_agent_fault_flags_pickle_load_from_with_handle_over_param_path() -> None:
    # Fail-closed: `with open(<param> + '/m.pkl') as f: pickle.load(f)` reads an
    # agent artifact through a with-handle over a NON-trusted parameter path. A
    # gate keyed only on a literal /tmp/output token or a bare assigned name would
    # miss the handle; the deserialize must still be flagged.
    src = (
        "import pickle\n"
        "\n"
        "def compute_score(workspace, submission_dir, private):\n"
        "    with open(submission_dir + '/model.pkl', 'rb') as f:\n"
        "        model = pickle.load(f)\n"
        "    return float(model.predict())\n"
    )
    issues = validator_module._agent_fault_issues("scorer/compute_score.py", src)
    assert any("deserializes pickle" in issue for issue in issues)


def test_agent_fault_flags_pickle_load_of_compound_param_path() -> None:
    # Fail-closed: a deserialize of `submission_dir + '/m.joblib'` (a compound
    # path rooted at a non-trusted parameter) is not provably trusted and must be
    # flagged, even though no bare /tmp/output token appears in the argument.
    src = (
        "import joblib\n"
        "\n"
        "def compute_score(workspace, submission_dir, private):\n"
        "    model = joblib.load(submission_dir + '/model.joblib')\n"
        "    return float(model.predict())\n"
    )
    issues = validator_module._agent_fault_issues("scorer/compute_score.py", src)
    assert any("deserializes pickle" in issue for issue in issues)


def test_agent_fault_flags_pickle_load_of_bare_untrusted_param() -> None:
    # Fail-closed: a deserialize of a bare parameter that is not provably trusted
    # (not `private`, not a trusted literal) must be flagged.
    src = (
        "import joblib\n"
        "\n"
        "def compute_score(workspace, submission_dir, private):\n"
        "    model = joblib.load(submission_dir)\n"
        "    return float(model.predict())\n"
    )
    issues = validator_module._agent_fault_issues("scorer/compute_score.py", src)
    assert any("deserializes pickle" in issue for issue in issues)


def test_agent_fault_flags_pickle_load_of_relative_path() -> None:
    # Fail-closed: a relative path does not resolve to a trusted root, so it is
    # not provably trusted and a deserialize of it must be flagged.
    src = (
        "import joblib\n"
        "\n"
        "def compute_score(workspace, trajectory, private):\n"
        "    model = joblib.load('model.pkl')\n"
        "    return float(model.predict())\n"
    )
    issues = validator_module._agent_fault_issues("scorer/compute_score.py", src)
    assert any("deserializes pickle" in issue for issue in issues)


def test_agent_fault_allows_pickle_load_from_private_param_join() -> None:
    # No false positive: deserializing held-out truth reached via the `private`
    # parameter (os.path.join / Path division) is legitimate and must NOT flag.
    src = (
        "import os\n"
        "import joblib\n"
        "\n"
        "def compute_score(workspace, trajectory, private):\n"
        "    truth = joblib.load(os.path.join(private, 'reference.joblib'))\n"
        "    return float(truth is not None)\n"
    )
    issues = validator_module._agent_fault_issues("scorer/compute_score.py", src)
    assert not any("deserializes pickle" in issue for issue in issues)


def test_agent_fault_allows_pickle_load_from_with_handle_over_trusted_path() -> None:
    # No false positive: a with-handle over a trusted /mcp_server/data literal
    # inherits the open's trustedness, so `pickle.load(f)` is legitimate.
    src = (
        "import pickle\n"
        "\n"
        "def compute_score(workspace, trajectory, private):\n"
        "    with open('/mcp_server/data/reference.pkl', 'rb') as f:\n"
        "        truth = pickle.load(f)\n"
        "    return float(truth is not None)\n"
    )
    issues = validator_module._agent_fault_issues("scorer/compute_score.py", src)
    assert not any("deserializes pickle" in issue for issue in issues)


def test_agent_fault_allows_pickle_load_from_public_data_fstring() -> None:
    # No false positive: the read-only public data mount (/data, baked root 0444)
    # is agent-immutable, so an f-string rooted there is a trusted load.
    src = (
        "import torch\n"
        "\n"
        "def compute_score(workspace, trajectory, private):\n"
        "    name = 'reference_model.pt'\n"
        "    model = torch.load(f'/data/{name}')\n"
        "    return float(model is not None)\n"
    )
    issues = validator_module._agent_fault_issues("scorer/compute_score.py", src)
    assert not any("deserializes pickle" in issue for issue in issues)


def test_agent_fault_flags_unguarded_agent_read() -> None:
    src = (
        "import pandas as pd\n"
        "\n"
        "def compute_score(workspace, trajectory, private):\n"
        "    df = pd.read_csv(workspace / 'submission.csv')\n"
        "    return float(len(df) > 0)\n"
    )
    issues = validator_module._agent_fault_issues("scorer/compute_score.py", src)
    assert any("no guard" in issue for issue in issues)


def test_agent_fault_flags_guarded_data_read_without_regular_file_guard() -> None:
    # try/except doesn't stop a symlink-to-truth read; a raw agent-path reader
    # needs a sanctioned loader or an os.lstat + stat.S_ISREG guard.
    src = (
        "import pandas as pd\n"
        "from grading import AgentFault\n"
        "\n"
        "def compute_score(workspace, trajectory, private):\n"
        "    try:\n"
        "        df = pd.read_csv(workspace / 'submission.csv')\n"
        "    except OSError as exc:\n"
        "        raise AgentFault(str(exc)) from exc\n"
        "    return float(len(df) > 0)\n"
    )
    issues = validator_module._agent_fault_issues("scorer/compute_score.py", src)
    assert any("non-regular" in issue or "SYMLINK" in issue for issue in issues)


def test_agent_fault_accepts_sanctioned_loader() -> None:
    # The sanctioned loader lstat's + rejects symlinks/non-regular files
    # internally, so it is the accepted way to read an agent submission.
    src = (
        "from grading.helpers import load_submission_or_fault\n"
        "\n"
        "def compute_score(workspace, trajectory, private):\n"
        "    df = load_submission_or_fault(workspace / 'submission.csv')\n"
        "    return float(len(df) > 0)\n"
    )
    assert validator_module._agent_fault_issues("scorer/compute_score.py", src) == []


def test_agent_fault_flags_lstat_guarded_read_as_racy() -> None:
    # An os.lstat + stat.S_ISREG check is check-then-use on the PATH: a surviving
    # uid-1000 process races it (swaps a regular file for a symlink between the
    # check and the read). It must NOT clear the symlink finding -- only an
    # O_NOFOLLOW open (or a sanctioned loader) is race-free.
    src = (
        "import os\n"
        "import stat\n"
        "import pandas as pd\n"
        "from grading import AgentFault\n"
        "\n"
        "def compute_score(workspace, trajectory, private):\n"
        "    p = workspace / 'submission.csv'\n"
        "    if not stat.S_ISREG(os.lstat(p).st_mode):\n"
        "        raise AgentFault('not a regular file')\n"
        "    try:\n"
        "        df = pd.read_csv(p)\n"
        "    except OSError as exc:\n"
        "        raise AgentFault(str(exc)) from exc\n"
        "    return float(len(df) > 0)\n"
    )
    issues = validator_module._agent_fault_issues("scorer/compute_score.py", src)
    assert any("non-regular" in issue or "SYMLINK" in issue for issue in issues)


def test_agent_fault_accepts_nofollow_handroll() -> None:
    # os.open(..., O_NOFOLLOW) + a read of THAT descriptor is the race-free manual
    # alternative to a sanctioned loader: a symlink leaf fails atomically at open.
    src = (
        "import os\n"
        "import numpy as np\n"
        "from grading import AgentFault\n"
        "\n"
        "def compute_score(workspace, trajectory, private):\n"
        "    p = workspace / 'submission.npy'\n"
        "    fd = os.open(p, os.O_RDONLY | os.O_NOFOLLOW)\n"
        "    fh = os.fdopen(fd, 'rb')\n"
        "    try:\n"
        "        arr = np.load(fh)\n"
        "    except OSError as exc:\n"
        "        raise AgentFault(str(exc)) from exc\n"
        "    finally:\n"
        "        fh.close()\n"
        "    return float(arr.size > 0)\n"
    )
    assert validator_module._agent_fault_issues("scorer/compute_score.py", src) == []


def test_agent_fault_comment_token_does_not_clear_symlink_finding() -> None:
    # Guard detection is AST-scoped, not whole-source substring: a comment (or
    # docstring / string literal) mentioning O_NOFOLLOW must NOT clear the finding.
    src = (
        "import pandas as pd\n"
        "from grading import AgentFault\n"
        "\n"
        "def compute_score(workspace, trajectory, private):\n"
        "    try:\n"
        "        df = pd.read_csv(workspace / 'submission.csv')  # O_NOFOLLOW, trust me\n"
        "    except OSError as exc:\n"
        "        raise AgentFault(str(exc)) from exc\n"
        "    return float(len(df) > 0)\n"
    )
    issues = validator_module._agent_fault_issues("scorer/compute_score.py", src)
    assert any("non-regular" in issue or "SYMLINK" in issue for issue in issues)


def test_agent_fault_ignores_private_path_reads() -> None:
    src = (
        "import pandas as pd\n"
        "\n"
        "def compute_score(workspace, trajectory, private):\n"
        "    truth = pd.read_csv(private / 'truth.csv')\n"
        "    return float(len(truth) > 0)\n"
    )
    assert validator_module._agent_fault_issues("scorer/compute_score.py", src) == []


def test_agent_fault_flags_unguarded_agent_read_via_alias() -> None:
    # The path is built into a variable, then read -- the lint must trace it.
    src = (
        "import pandas as pd\n"
        "\n"
        "def compute_score(workspace, trajectory, private):\n"
        "    path = workspace / 'submission.csv'\n"
        "    df = pd.read_csv(path)\n"
        "    return float(len(df) > 0)\n"
    )
    issues = validator_module._agent_fault_issues("scorer/compute_score.py", src)
    assert any("no guard" in issue for issue in issues)


def test_agent_fault_ignores_private_path_read_via_alias() -> None:
    src = (
        "import pandas as pd\n"
        "\n"
        "def compute_score(workspace, trajectory, private):\n"
        "    p = private / 'truth.csv'\n"
        "    truth = pd.read_csv(p)\n"
        "    return float(len(truth) > 0)\n"
    )
    assert validator_module._agent_fault_issues("scorer/compute_score.py", src) == []


def test_agent_fault_flags_unguarded_path_read_text() -> None:
    # The path is the *receiver* of read_text(), not an argument; the lint must
    # still flag it (a planted dir/FIFO raises IsADirectoryError/OSError).
    src = (
        "def compute_score(workspace, trajectory, private):\n"
        "    return 0.5 if (workspace / 'answer.txt').read_text().strip() else 0.0\n"
    )
    issues = validator_module._agent_fault_issues("scorer/compute_score.py", src)
    assert any("no guard" in issue and "read_text" in issue for issue in issues)


def test_agent_fault_flags_unguarded_path_read_bytes_via_alias() -> None:
    src = (
        "def compute_score(workspace, trajectory, private):\n"
        "    path = workspace / 'submission.bin'\n"
        "    return float(len(path.read_bytes()) > 0)\n"
    )
    issues = validator_module._agent_fault_issues("scorer/compute_score.py", src)
    assert any("no guard" in issue and "read_bytes" in issue for issue in issues)


def test_agent_fault_accepts_guarded_path_read_text_with_agentfault() -> None:
    src = (
        "from grading import AgentFault\n"
        "\n"
        "def compute_score(workspace, trajectory, private):\n"
        "    answer = workspace / 'answer.txt'\n"
        "    try:\n"
        "        text = answer.read_text()\n"
        "    except OSError as exc:\n"
        "        raise AgentFault(str(exc)) from exc\n"
        "    return 0.5 if text.strip() else 0.0\n"
    )
    assert validator_module._agent_fault_issues("scorer/compute_score.py", src) == []


def test_agent_fault_ignores_private_path_read_text() -> None:
    src = (
        "import json\n"
        "\n"
        "def compute_score(workspace, trajectory, private):\n"
        "    cfg = json.loads((private / 'truth.json').read_text())\n"
        "    return float(len(cfg) > 0)\n"
    )
    assert validator_module._agent_fault_issues("scorer/compute_score.py", src) == []


def test_agent_fault_stage_passes_for_clean_scorer(tmp_path: Path) -> None:
    problem_dir = tmp_path / "clean-fault"
    _write_problem(problem_dir, task_type="ml")
    (problem_dir / "scorer" / "compute_score.py").write_text(
        "def compute_score(workspace, trajectory, private):\n    return 0.25\n"
    )
    stage = TaskValidator()._agent_fault(problem_dir)
    assert stage.passed
    assert stage.issues == []


def test_agent_fault_stage_fails_for_broad_swallow(tmp_path: Path) -> None:
    problem_dir = tmp_path / "swallow-fault"
    _write_problem(problem_dir, task_type="ml")
    (problem_dir / "scorer" / "compute_score.py").write_text(
        "def compute_score(workspace, trajectory, private):\n"
        "    try:\n"
        "        return _run(workspace)\n"
        "    except Exception:\n"
        "        return 0.0\n"
        "\n"
        "def _run(workspace):\n"
        "    return 1.0\n"
    )
    stage = TaskValidator()._agent_fault(problem_dir)
    assert not stage.passed
    assert any("broad `except`" in issue for issue in stage.issues)


def test_shipped_scorers_pass_agent_fault_lint() -> None:
    # Dogfood: every example/problem/starter scorer must obey the AgentFault
    # keep-vs-discard discipline, so the template never ships a scorer that
    # would fail its own blocking validator.
    repo_root = Path(__file__).resolve().parents[2]
    roots = [
        repo_root / "examples",
        repo_root / "problems",
        repo_root / "alignerr_plugin" / "src" / "alignerr_plugin" / "starter_templates",
    ]
    scorers = [
        path
        for root in roots
        if root.is_dir()
        for path in sorted(root.rglob("scorer/**/*.py"))
        if "__pycache__" not in path.parts
    ]
    assert scorers, "expected to find shipped scorer files to lint"
    offenders: dict[str, list[str]] = {}
    for path in scorers:
        issues = validator_module._agent_fault_issues(path.relative_to(repo_root).as_posix(), path.read_text())
        if issues:
            offenders[path.relative_to(repo_root).as_posix()] = issues
    assert not offenders, f"shipped scorers fail the agent_fault lint: {offenders}"


def test_private_data_layout_requires_hardening_for_plain_private_copy(
    tmp_path: Path,
) -> None:
    problem_dir = tmp_path / "private-layout"
    _write_private_layout_problem(
        problem_dir,
        """\
FROM lbx-tasks-base:runtime-ml-core-py313-local
COPY ${PROBLEM_DIR}/scorer/ /mcp_server/grader/
COPY ${PROBLEM_DIR}/scorer/data/ /mcp_server/data/
""",
    )

    stage = TaskValidator()._private_data_layout(problem_dir)

    assert not stage.passed
    assert any("left group/world-readable" in issue for issue in stage.issues)


def test_private_data_layout_accepts_copy_chmod_0600(tmp_path: Path) -> None:
    problem_dir = tmp_path / "private-layout"
    _write_private_layout_problem(
        problem_dir,
        """\
FROM lbx-tasks-base:runtime-ml-core-py313-local
COPY --chown=root:root --chmod=0600 ${PROBLEM_DIR}/scorer/ /mcp_server/grader/
COPY --chown=root:root --chmod=0600 ${PROBLEM_DIR}/scorer/data/ /mcp_server/data/
""",
    )

    stage = TaskValidator()._private_data_layout(problem_dir)

    assert stage.passed
    assert stage.issues == []


def _write_trivial_scoring_problem(
    problem_dir: Path, compute_src: str, *, instruction: str = "Write the result.\n"
) -> None:
    (problem_dir / "scorer").mkdir(parents=True)
    (problem_dir / "solution").mkdir()
    (problem_dir / "metadata.json").write_text(
        json.dumps(
            {
                "benchmark": "taiga_task",
                "problem_data": {"instance_id": problem_dir.name},
            }
        )
    )
    (problem_dir / "task.toml").write_text("""
[task]
name = "labelbox/trivial"

[environment]
required_resources = "4vcpu+16gib"

[difficulty]
task_type = "ml"
domain = "scientific_discovery_computational_science"
reward_type = "multi_deterministic_rubrics"
license = "MIT"
license_source = "https://github.com/owner/dataset/blob/main/LICENSE"

[[outputs]]
path = "/tmp/output/result.json"
required = true
""")
    (problem_dir / "instruction.md").write_text(instruction)
    (problem_dir / "solution" / "solve.sh").write_text(
        "mkdir -p /tmp/output\nprintf '{\"x\": 1}' > /tmp/output/result.json\n"
    )
    (problem_dir / "scorer" / "compute_score.py").write_text(compute_src)


def test_no_op_submission_must_not_outscore(tmp_path: Path) -> None:
    problem_dir = tmp_path / "noop-bad"
    _write_trivial_scoring_problem(
        problem_dir,
        """
def compute_score(workspace, trajectory, private):
    return 1.0
""",
    )

    stage, meta = TaskValidator()._compute_score_return(problem_dir)

    assert not stage.passed
    assert meta["noop_score"] == 1.0
    assert any("no-op" in issue and "max_trivial_score" in issue for issue in stage.issues)


def test_no_op_gate_passes_when_empty_scores_zero(tmp_path: Path) -> None:
    problem_dir = tmp_path / "noop-ok"
    _write_trivial_scoring_problem(
        problem_dir,
        """
import json


def compute_score(workspace, trajectory, private):
    result = workspace / "result.json"
    if not result.exists():
        return 0.0
    return 1.0 if json.loads(result.read_text()).get("x") == 1 else 0.0
""",
    )

    stage, meta = TaskValidator()._compute_score_return(problem_dir)

    assert stage.passed, stage.issues
    assert meta["noop_score"] == 0.0


def test_continuous_no_op_must_anchor_to_zero(tmp_path: Path) -> None:
    # 0.2 is under the max_trivial_score ceiling (0.5) but violates the strict
    # zero anchor that continuous scoring functions are held to.
    problem_dir = tmp_path / "zero-anchor-bad"
    _write_trivial_scoring_problem(
        problem_dir,
        """
def compute_score(workspace, trajectory, private):
    result = workspace / "result.json"
    if not result.exists():
        return 0.2
    return 0.5
""",
    )
    task_toml = problem_dir / "task.toml"
    task_toml.write_text(
        task_toml.read_text().replace(
            'reward_type = "multi_deterministic_rubrics"',
            'reward_type = "continuous_scoring_function"',
        )
    )

    stage, meta = TaskValidator()._compute_score_return(problem_dir)

    assert not stage.passed
    assert meta["noop_score"] == 0.2
    assert any("anchor an empty submission to 0" in issue for issue in stage.issues)


def test_continuous_no_op_passes_when_anchored_to_zero(tmp_path: Path) -> None:
    problem_dir = tmp_path / "zero-anchor-ok"
    _write_trivial_scoring_problem(
        problem_dir,
        """
def compute_score(workspace, trajectory, private):
    result = workspace / "result.json"
    if not result.exists():
        return 0.0
    return 0.5
""",
    )
    task_toml = problem_dir / "task.toml"
    task_toml.write_text(
        task_toml.read_text().replace(
            'reward_type = "multi_deterministic_rubrics"',
            'reward_type = "continuous_scoring_function"',
        )
    )

    stage, meta = TaskValidator()._compute_score_return(problem_dir)

    assert stage.passed, stage.issues
    assert meta["noop_score"] == 0.0


def test_continuous_no_op_agentfault_proves_zero_anchor(tmp_path: Path) -> None:
    # AgentFault on an empty workspace is the documented anchor path (a kept
    # 0.0 in production); the no-op probe must count it as a measured 0.0, not
    # skip the zero-anchor gate.
    problem_dir = tmp_path / "zero-anchor-fault"
    _write_trivial_scoring_problem(
        problem_dir,
        """
from grading.faults import AgentFault


def compute_score(workspace, trajectory, private):
    result = workspace / "result.json"
    if not result.exists():
        raise AgentFault("no submission")
    return 0.5
""",
    )
    task_toml = problem_dir / "task.toml"
    task_toml.write_text(
        task_toml.read_text().replace(
            'reward_type = "multi_deterministic_rubrics"',
            'reward_type = "continuous_scoring_function"',
        )
    )

    stage, meta = TaskValidator()._compute_score_return(problem_dir)

    assert stage.passed, stage.issues
    assert meta["noop_score"] == 0.0


def test_continuous_no_op_grader_crash_fails_zero_anchor(tmp_path: Path) -> None:
    # A NON-AgentFault crash on an empty workspace leaves the anchor unproven;
    # continuous tasks must fail instead of silently skipping the gate.
    problem_dir = tmp_path / "zero-anchor-crash"
    _write_trivial_scoring_problem(
        problem_dir,
        """
import json


def compute_score(workspace, trajectory, private):
    result = workspace / "result.json"
    if not result.exists():
        raise KeyError("boom")
    return 0.5
""",
    )
    task_toml = problem_dir / "task.toml"
    task_toml.write_text(
        task_toml.read_text().replace(
            'reward_type = "multi_deterministic_rubrics"',
            'reward_type = "continuous_scoring_function"',
        )
    )

    stage, meta = TaskValidator()._compute_score_return(problem_dir)

    assert not stage.passed
    assert meta["noop_score"] is None
    assert any("anchor cannot be proven" in issue for issue in stage.issues)


def test_rubric_no_op_below_ceiling_is_not_zero_anchored(tmp_path: Path) -> None:
    # Rubric tasks keep the max_trivial_score ceiling only; a no-op scoring 0.2
    # must NOT trip the continuous zero anchor.
    problem_dir = tmp_path / "rubric-noop"
    _write_trivial_scoring_problem(
        problem_dir,
        """
import json


def compute_score(workspace, trajectory, private):
    result = workspace / "result.json"
    if not result.exists():
        return 0.2
    return 1.0 if json.loads(result.read_text()).get("x") == 1 else 0.2
""",
    )

    stage, meta = TaskValidator()._compute_score_return(problem_dir)

    assert stage.passed, stage.issues
    assert meta["noop_score"] == 0.2
    assert not any("anchor an empty submission" in issue for issue in stage.issues)


def test_continuous_ground_truth_accepts_reference_score_near_half(
    tmp_path: Path,
) -> None:
    problem_dir = tmp_path / "continuous-ok"
    _write_trivial_scoring_problem(
        problem_dir,
        """
def compute_score(workspace, trajectory, private):
    result = workspace / "result.json"
    if not result.exists():
        return 0.0
    return {"score": 0.51, "subscores": {"reference": 0.51}, "weights": {"reference": 1.0}}
""",
    )
    task_toml = problem_dir / "task.toml"
    task_toml.write_text(
        task_toml.read_text().replace(
            'reward_type = "multi_deterministic_rubrics"',
            'reward_type = "continuous_scoring_function"',
        )
    )

    stage, meta = TaskValidator()._compute_score_return(problem_dir)

    assert stage.passed, stage.issues
    assert meta["ground_truth_passed"] is True


def test_continuous_ground_truth_rejects_perfect_score(tmp_path: Path) -> None:
    problem_dir = tmp_path / "continuous-bad"
    _write_trivial_scoring_problem(
        problem_dir,
        """
def compute_score(workspace, trajectory, private):
    result = workspace / "result.json"
    if not result.exists():
        return 0.0
    return 1.0
""",
    )
    task_toml = problem_dir / "task.toml"
    task_toml.write_text(
        task_toml.read_text().replace(
            'reward_type = "multi_deterministic_rubrics"',
            'reward_type = "continuous_scoring_function"',
        )
    )

    stage, _meta = TaskValidator()._compute_score_return(problem_dir)

    assert not stage.passed
    assert any("continuous_scoring_function" in issue and "0.5" in issue for issue in stage.issues)


_CONTINUOUS_SCORER = """
import json


def compute_score(workspace, trajectory, private):
    result = workspace / "result.json"
    if not result.exists():
        return 0.0
    return float(json.loads(result.read_text()).get("x", 0)) * 0.5
"""


def _make_continuous_task(problem_dir: Path) -> None:
    _write_trivial_scoring_problem(problem_dir, _CONTINUOUS_SCORER)
    task_toml = problem_dir / "task.toml"
    task_toml.write_text(
        task_toml.read_text().replace(
            'reward_type = "multi_deterministic_rubrics"',
            'reward_type = "continuous_scoring_function"',
        )
    )


def test_calibration_passes_for_baseline_below_reference(tmp_path: Path) -> None:
    # Reference solve.sh writes x=1 -> 0.5; baseline writes x=0 -> 0.0.
    problem_dir = tmp_path / "cal-ok"
    _make_continuous_task(problem_dir)
    base = problem_dir / "baselines" / "zero"
    base.mkdir(parents=True)
    (base / "result.json").write_text('{"x": 0}')

    stage, _meta = TaskValidator()._compute_score_return(problem_dir)

    assert stage.passed, stage.issues


def _write_in_container_proof_task(
    problem_dir: Path,
    *,
    reward_type: str,
    ground_truth_result: dict | None,
    extra_ground_truth_toml: str = "",
) -> None:
    _write_trivial_scoring_problem(problem_dir, _CONTINUOUS_SCORER)
    task_toml = problem_dir / "task.toml"
    task_toml.write_text(
        task_toml.read_text().replace(
            'reward_type = "multi_deterministic_rubrics"',
            f'reward_type = "{reward_type}"',
        )
        + f"\n[ground_truth]\nin_container = true\n{extra_ground_truth_toml}"
    )
    if ground_truth_result is not None:
        proof_dir = problem_dir / ".alignerr"
        proof_dir.mkdir()
        (proof_dir / "build_proof.json").write_text(
            json.dumps({"ground_truth_result": ground_truth_result})
        )


def _committed_ground_truth_issues(problem_dir: Path) -> list[str]:
    from alignerr_plugin.utils import load_task_toml

    return validator_module._validate_committed_ground_truth_result(
        problem_dir, task_toml=load_task_toml(problem_dir), meta={}
    )


def test_in_container_continuous_proof_requires_trivial_baseline(
    tmp_path: Path,
) -> None:
    problem_dir = tmp_path / "in-container-missing"
    _write_in_container_proof_task(
        problem_dir,
        reward_type="continuous_scoring_function",
        ground_truth_result={"score": 0.5},
    )

    issues = _committed_ground_truth_issues(problem_dir)

    assert any("trivial_baseline_score is missing" in issue for issue in issues)


def test_in_container_continuous_proof_accepts_zero_anchored_noop(
    tmp_path: Path,
) -> None:
    problem_dir = tmp_path / "in-container-ok"
    _write_in_container_proof_task(
        problem_dir,
        reward_type="continuous_scoring_function",
        ground_truth_result={"score": 0.5, "trivial_baseline_score": 0.0},
    )

    issues = _committed_ground_truth_issues(problem_dir)

    assert issues == []


def test_in_container_continuous_proof_rejects_unanchored_noop(
    tmp_path: Path,
) -> None:
    problem_dir = tmp_path / "in-container-bad"
    _write_in_container_proof_task(
        problem_dir,
        reward_type="continuous_scoring_function",
        ground_truth_result={"score": 0.5, "trivial_baseline_score": 0.2},
    )

    issues = _committed_ground_truth_issues(problem_dir)

    assert any("anchor an empty submission to 0" in issue for issue in issues)


def test_in_container_continuous_proof_honors_zero_anchor_epsilon(
    tmp_path: Path,
) -> None:
    problem_dir = tmp_path / "in-container-epsilon"
    _write_in_container_proof_task(
        problem_dir,
        reward_type="continuous_scoring_function",
        ground_truth_result={"score": 0.5, "trivial_baseline_score": 0.2},
        extra_ground_truth_toml="zero_anchor_epsilon = 0.3\n",
    )

    issues = _committed_ground_truth_issues(problem_dir)

    assert issues == []


def test_in_container_rubric_proof_keeps_ceiling_only(tmp_path: Path) -> None:
    # Rubric tasks are not zero-anchored and do not require the recorded
    # trivial_baseline_score; the max_trivial_score ceiling still applies when
    # a score IS recorded.
    problem_dir = tmp_path / "in-container-rubric"
    _write_in_container_proof_task(
        problem_dir,
        reward_type="multi_deterministic_rubrics",
        ground_truth_result={"score": 1.0, "trivial_baseline_score": 0.2},
    )

    issues = _committed_ground_truth_issues(problem_dir)
    assert issues == []

    problem_dir_missing = tmp_path / "in-container-rubric-missing"
    _write_in_container_proof_task(
        problem_dir_missing,
        reward_type="multi_deterministic_rubrics",
        ground_truth_result={"score": 1.0},
    )

    issues_missing = _committed_ground_truth_issues(problem_dir_missing)
    assert issues_missing == []


def test_calibration_fails_for_baseline_matching_reference(tmp_path: Path) -> None:
    # A baseline that also writes x=1 scores 0.5 == reference -> not learnable.
    problem_dir = tmp_path / "cal-bad"
    _make_continuous_task(problem_dir)
    base = problem_dir / "baselines" / "cheat"
    base.mkdir(parents=True)
    (base / "result.json").write_text('{"x": 1}')

    stage, _meta = TaskValidator()._compute_score_return(problem_dir)

    assert not stage.passed
    assert any("baseline" in issue and "learnable" in issue for issue in stage.issues)


_DETERMINISTIC_SCORER = """
import json


def compute_score(workspace, trajectory, private):
    result = workspace / "result.json"
    if not result.exists():
        return 0.0
    return 1.0 if json.loads(result.read_text()).get("x", 0) >= 1 else 0.0
"""


def test_grade_workspace_counts_agentfault_as_zero() -> None:
    from grading.faults import AgentFault

    def scorer(workspace, trajectory, private):  # noqa: ANN001, ARG001
        raise AgentFault("malformed submission")

    class _Dummy:
        pass

    params = ["workspace", "trajectory", "private"]
    # Default: an AgentFault submission is skipped (None) ...
    assert (
        validator_module._grade_workspace_score(Path("."), Path("."), scorer, params, lambda value: value, _Dummy)
        is None
    )
    # ... but the learnability gate counts it as a real 0.0 anchor.
    assert (
        validator_module._grade_workspace_score(
            Path("."),
            Path("."),
            scorer,
            params,
            lambda value: value,
            _Dummy,
            agent_fault_as_zero=True,
        )
        == 0.0
    )

    # Subclasses of AgentFault also count (matched across the MRO).
    class _SubFault(AgentFault):
        pass

    def scorer_sub(workspace, trajectory, private):  # noqa: ANN001, ARG001
        raise _SubFault("subclass fault")

    assert (
        validator_module._grade_workspace_score(
            Path("."),
            Path("."),
            scorer_sub,
            params,
            lambda value: value,
            _Dummy,
            agent_fault_as_zero=True,
        )
        == 0.0
    )


def test_calibration_skips_baseline_check_for_deterministic_tasks(tmp_path: Path) -> None:
    # Deterministic rubric task with a high-scoring baseline is not gated on
    # learnability (the baseline triad is a continuous-scoring concept).
    problem_dir = tmp_path / "cal-det"
    _write_trivial_scoring_problem(problem_dir, _DETERMINISTIC_SCORER)
    base = problem_dir / "baselines" / "cheat"
    base.mkdir(parents=True)
    (base / "result.json").write_text('{"x": 2}')

    stage, _meta = TaskValidator()._compute_score_return(problem_dir)

    assert stage.passed, stage.issues
    assert not any("learnable" in issue for issue in stage.issues)


def test_prompt_example_must_not_outscore(tmp_path: Path) -> None:
    problem_dir = tmp_path / "example-bad"
    _write_trivial_scoring_problem(
        problem_dir,
        """
import json


def compute_score(workspace, trajectory, private):
    result = workspace / "result.json"
    if not result.exists():
        return 0.0
    return 1.0 if json.loads(result.read_text()).get("x") == 1 else 0.0
""",
        instruction='Submit this example:\n\n```json\n{"x": 1}\n```\n',
    )

    stage, meta = TaskValidator()._compute_score_return(problem_dir)

    assert not stage.passed
    assert meta["example_score"] == 1.0
    assert any("example" in issue for issue in stage.issues)


def test_reward_hack_lint_flags_exact_match_and_sentiment(tmp_path: Path) -> None:
    problem_dir = tmp_path / "rh-task"
    _write_problem(problem_dir, task_type="ml")
    (problem_dir / "scorer" / "compute_score.py").write_text("""
def compute_score(workspace, trajectory, private):
    oracle = {"a": 1}
    submission = {"a": 1}
    if submission == oracle:
        return 1.0
    conclusion = "result is viable and safe to proceed"
    if "unsafe" in conclusion or "unacceptable" in conclusion:
        return 0.0
    return 0.5
""")

    warnings = reward_hack_lint(problem_dir)

    assert any("exact-match shortcut" in w for w in warnings)
    assert any("sentiment" in w for w in warnings)


def test_sanctioned_curve_stage_rejects_exponential_curve(tmp_path: Path) -> None:
    # The deprecated exponential curve is now a BLOCKING validation failure (was
    # only an advisory reward-hack warning the CLI never gated on).
    problem_dir = tmp_path / "exp-curve-task"
    _write_problem(problem_dir, task_type="ml")
    (problem_dir / "scorer" / "compute_score.py").write_text(
        "from grading.calibration import ExponentialCurve\n"
        "def compute_score(workspace, trajectory, private):\n"
        "    curve = ExponentialCurve.from_reference(0.66)\n"
        "    return curve.score(0.5)\n"
    )
    stage = TaskValidator()._sanctioned_curve(problem_dir)
    assert not stage.passed
    assert any("deprecated exponential" in i for i in stage.issues)


def test_sanctioned_curve_stage_allows_piecewise_and_ignores_comment(tmp_path: Path) -> None:
    # AST-based (not substring): a comment mentioning ExponentialCurve must NOT
    # trip the gate, and the sanctioned PiecewiseLinearCurve passes.
    problem_dir = tmp_path / "pwl-task"
    _write_problem(problem_dir, task_type="ml")
    (problem_dir / "scorer" / "compute_score.py").write_text(
        "# NOTE: do not use ExponentialCurve; it is deprecated.\n"
        "from grading.calibration import PiecewiseLinearCurve\n"
        "def compute_score(workspace, trajectory, private):\n"
        "    curve = PiecewiseLinearCurve.from_reference(0.66)\n"
        "    return curve.score(0.5)\n"
    )
    stage = TaskValidator()._sanctioned_curve(problem_dir)
    assert stage.passed, stage.issues


def test_sanctioned_curve_ignores_unrelated_local_symbol(tmp_path: Path) -> None:
    # Detection is scoped to grading.calibration: a grader's own local symbol that
    # merely shares the name `exponential_score` (here a private helper) must NOT
    # be mistaken for the deprecated calibration curve.
    problem_dir = tmp_path / "own-exp-score"
    _write_problem(problem_dir, task_type="ml")
    (problem_dir / "scorer" / "compute_score.py").write_text(
        "import math\n"
        "def exponential_score(x):\n"
        "    return 1.0 - math.exp(-x)\n"
        "def compute_score(workspace, trajectory, private):\n"
        "    return exponential_score(0.5)\n"
    )
    stage = TaskValidator()._sanctioned_curve(problem_dir)
    assert stage.passed, stage.issues


def test_sanctioned_curve_flags_module_qualified_reference(tmp_path: Path) -> None:
    # A calibration-module-qualified reference to the deprecated curve
    # (`import grading.calibration as gc; gc.exponential_score(...)`) is still
    # flagged, even though the deprecated name never appears as a bare identifier.
    problem_dir = tmp_path / "qualified-exp"
    _write_problem(problem_dir, task_type="ml")
    (problem_dir / "scorer" / "compute_score.py").write_text(
        "import grading.calibration as gc\n"
        "def compute_score(workspace, trajectory, private):\n"
        "    a, b = gc.solve_exponential_constants(0.66)\n"
        "    return gc.exponential_score(0.5, a, b)\n"
    )
    stage = TaskValidator()._sanctioned_curve(problem_dir)
    assert not stage.passed
    assert any("deprecated exponential" in i for i in stage.issues)


# The ml starter ships no per-task Dockerfile (hardened by the shared base, tested
# below); only the native starters have one here.
@pytest.mark.parametrize("template_name", ["mujoco", "cfd", "structures"])
def test_starter_dockerfiles_harden_private_roots(template_name: str) -> None:
    repo_root = Path(__file__).resolve().parents[2]
    dockerfile = (
        repo_root
        / "alignerr_plugin"
        / "src"
        / "alignerr_plugin"
        / "starter_templates"
        / template_name
        / "environment"
        / "Dockerfile"
    )
    text = dockerfile.read_text()

    assert "COPY --chown=root:root ${PROBLEM_DIR}/scorer/data/ /mcp_server/data/" in text
    assert "COPY --chown=root:root ${PROBLEM_DIR}/scorer/ /mcp_server/grader/" in text
    assert "rm -rf /mcp_server/grader/data" in text
    assert "find /mcp_server/data /mcp_server/grader -type d -exec chmod 0700" in text
    assert "find /mcp_server/data /mcp_server/grader -type f -exec chmod 0600" in text


def test_mlenvs_base_dockerfile_hardens_private_roots() -> None:
    # The shared base/task.mlenvs.Dockerfile stages data/private/ into the
    # root-only /mcp_server/data and the grader into /mcp_server/grader.
    repo_root = Path(__file__).resolve().parents[2]
    text = (repo_root / "base" / "task.mlenvs.Dockerfile").read_text()
    assert "/mcp_server/data" in text
    assert "0700" in text or "chmod 700" in text


def test_mlenvs_base_dockerfile_installs_env_deps_root_only() -> None:
    # env_dependencies (hidden-env server-only pip deps) install into the
    # root-only /mcp_server/env_deps via --target, NOT system-wide, so the agent
    # can't import them and bypass the RPC constraints.
    repo_root = Path(__file__).resolve().parents[2]
    text = (repo_root / "base" / "task.mlenvs.Dockerfile").read_text()
    assert "ARG ENV_DEPENDENCIES" in text
    assert "--target /mcp_server/env_deps" in text
    # never installed system-wide (that would re-expose it to the agent)
    assert "--system --break-system-packages --no-cache $ENV_DEPENDENCIES" not in text
    # and it lives under the 0700 /mcp_server barrier (final chmod covers it)
    assert "chmod 0700 /mcp_server" in text


# ── prompt runtime-reference guard ─────────────────────────────────────────


def test_prompt_runtime_references_flags_base_image_dependency_guidance() -> None:
    text = "Use the packages in the base image, including numpy and torch, to solve " "the task."
    issues = validator_module._prompt_internal_env_issues(text)
    assert any("base image" in issue for issue in issues)


def test_prompt_runtime_references_flags_task_metadata_dependency_guidance() -> None:
    text = "Read metadata.json to discover this task's declared dependencies."
    issues = validator_module._prompt_internal_env_issues(text)
    assert any("metadata.json" in issue or "declared dependencies" in issue for issue in issues)


def test_prompt_runtime_references_allows_direct_library_guidance() -> None:
    text = "Use installed Python packages such as NumPy, pandas, and scikit-learn."
    assert validator_module._prompt_internal_env_issues(text) == []


def test_prompt_runtime_references_allows_base_image_plus_direct_library_guidance() -> None:
    text = "The Docker base image is already prepared for the task. Use NumPy and " "torch, which are installed."
    assert validator_module._prompt_internal_env_issues(text) == []


def test_prompt_runtime_references_allows_dataset_metadata_file() -> None:
    text = "The dataset manifest is available at /data/metadata.json."
    assert validator_module._prompt_internal_env_issues(text) == []


def test_prompt_runtime_references_allows_nested_dataset_metadata_file() -> None:
    text = "The installed libraries are listed in the dataset manifest at " "/data/dataset/metadata.json."
    assert validator_module._prompt_internal_env_issues(text) == []


# ── scorer determinism ─────────────────────────────────────────────────────


def test_scorer_determinism_flags_wallclock() -> None:
    src = (
        "import time\n"
        "\n"
        "def compute_score(workspace, trajectory, private):\n"
        "    return float(time.time() % 2)\n"
    )
    issues = validator_module._scorer_determinism_issues("scorer/compute_score.py", src)
    assert any("wall-clock" in issue for issue in issues)


def test_scorer_determinism_flags_datetime_now() -> None:
    src = (
        "from datetime import datetime\n"
        "\n"
        "def compute_score(workspace, trajectory, private):\n"
        "    return float(datetime.now().second) / 60.0\n"
    )
    issues = validator_module._scorer_determinism_issues("scorer/compute_score.py", src)
    assert any("wall-clock" in issue for issue in issues)


def test_scorer_determinism_flags_unseeded_rng() -> None:
    src = (
        "import numpy as np\n"
        "\n"
        "def compute_score(workspace, trajectory, private):\n"
        "    return float(np.random.rand())\n"
    )
    issues = validator_module._scorer_determinism_issues("scorer/compute_score.py", src)
    assert any("no RNG" in issue for issue in issues)


def test_scorer_determinism_allows_seeded_rng() -> None:
    src = (
        "import numpy as np\n"
        "\n"
        "def compute_score(workspace, trajectory, private):\n"
        "    rng = np.random.default_rng(0)\n"
        "    return float(rng.random())\n"
    )
    assert validator_module._scorer_determinism_issues("scorer/compute_score.py", src) == []


def test_scorer_determinism_allows_clean_scorer() -> None:
    src = (
        "import numpy as np\n"
        "\n"
        "def compute_score(workspace, trajectory, private):\n"
        "    return float(np.mean([1.0, 2.0]))\n"
    )
    assert validator_module._scorer_determinism_issues("scorer/compute_score.py", src) == []


def test_scorer_determinism_ignores_dead_helper() -> None:
    # Non-determinism in a function the runner never reaches from compute_score
    # is dead during grading, so it is not flagged.
    src = (
        "import time\n"
        "\n"
        "def _debug_probe():\n"
        "    return time.time()\n"
        "\n"
        "def compute_score(workspace, trajectory, private):\n"
        "    return 1.0\n"
    )
    assert validator_module._scorer_determinism_issues("scorer/compute_score.py", src) == []


def test_scorer_determinism_seed_text_in_dead_code_does_not_suppress_live_rng() -> None:
    src = (
        "import numpy as np\n"
        "\n"
        "def _unused_seeded_helper():\n"
        "    # np.random.default_rng(0) in a comment is not a live seed\n"
        "    return 'RandomState(0) in a string is not a live seed either'\n"
        "\n"
        "def compute_score(workspace, trajectory, private):\n"
        "    return float(np.random.rand())\n"
    )
    issues = validator_module._scorer_determinism_issues("scorer/compute_score.py", src)
    assert any("no RNG" in issue for issue in issues)


def test_scorer_determinism_live_seed_suppresses_live_rng() -> None:
    src = (
        "import numpy as np\n"
        "\n"
        "def _score_with_seed():\n"
        "    np.random.seed(0)\n"
        "    return float(np.random.rand())\n"
        "\n"
        "def compute_score(workspace, trajectory, private):\n"
        "    return _score_with_seed()\n"
    )
    assert validator_module._scorer_determinism_issues("scorer/compute_score.py", src) == []


def test_scorer_determinism_seeded_local_generator_does_not_seed_global_numpy() -> None:
    src = (
        "import numpy as np\n"
        "\n"
        "def compute_score(workspace, trajectory, private):\n"
        "    rng = np.random.default_rng(0)\n"
        "    return float(np.random.rand()) + float(rng.random())\n"
    )
    issues = validator_module._scorer_determinism_issues("scorer/compute_score.py", src)
    assert any("no RNG" in issue for issue in issues)


def test_scorer_determinism_seeded_local_random_does_not_seed_global_random() -> None:
    src = (
        "import random\n"
        "\n"
        "def compute_score(workspace, trajectory, private):\n"
        "    rng = random.Random(0)\n"
        "    return random.random() + rng.random()\n"
    )
    issues = validator_module._scorer_determinism_issues("scorer/compute_score.py", src)
    assert any("no RNG" in issue for issue in issues)


def test_scorer_determinism_numpy_seed_does_not_seed_python_random() -> None:
    src = (
        "import random\n"
        "import numpy as np\n"
        "\n"
        "def compute_score(workspace, trajectory, private):\n"
        "    np.random.seed(0)\n"
        "    return random.random()\n"
    )
    issues = validator_module._scorer_determinism_issues("scorer/compute_score.py", src)
    assert any("no RNG" in issue for issue in issues)


def test_scorer_determinism_python_seed_does_not_seed_global_numpy() -> None:
    src = (
        "import random\n"
        "import numpy as np\n"
        "\n"
        "def compute_score(workspace, trajectory, private):\n"
        "    random.seed(0)\n"
        "    return float(np.random.rand())\n"
    )
    issues = validator_module._scorer_determinism_issues("scorer/compute_score.py", src)
    assert any("no RNG" in issue for issue in issues)


def test_scorer_determinism_seed_in_other_reachable_helper_does_not_suppress_rng() -> None:
    src = (
        "import numpy as np\n"
        "\n"
        "def _seed_only():\n"
        "    np.random.seed(0)\n"
        "\n"
        "def _score_without_seed():\n"
        "    return float(np.random.rand())\n"
        "\n"
        "def compute_score(workspace, trajectory, private):\n"
        "    if len(trajectory) > 0:\n"
        "        _seed_only()\n"
        "    return _score_without_seed()\n"
    )
    issues = validator_module._scorer_determinism_issues("scorer/compute_score.py", src)
    assert any("no RNG" in issue for issue in issues)


def test_scorer_determinism_conditional_seed_does_not_suppress_global_rng() -> None:
    src = (
        "import numpy as np\n"
        "\n"
        "def compute_score(workspace, trajectory, private):\n"
        "    if len(trajectory) > 0:\n"
        "        np.random.seed(0)\n"
        "    return float(np.random.rand())\n"
    )
    issues = validator_module._scorer_determinism_issues("scorer/compute_score.py", src)
    assert any("no RNG" in issue for issue in issues)


def test_scorer_determinism_same_scope_unconditional_global_seed_allowed() -> None:
    src = (
        "import random\n"
        "import numpy as np\n"
        "\n"
        "def compute_score(workspace, trajectory, private):\n"
        "    np.random.seed(0)\n"
        "    random.seed(0)\n"
        "    return float(np.random.rand()) + random.random()\n"
    )
    assert validator_module._scorer_determinism_issues("scorer/compute_score.py", src) == []


def test_scorer_determinism_variable_global_seed_does_not_suppress_rng() -> None:
    for seed_call, rng_call in (
        ("np.random.seed(SEED)", "np.random.rand()"),
        ("random.seed(SEED)", "random.random()"),
    ):
        src = (
            "import random\n"
            "import numpy as np\n"
            "\n"
            "SEED = None\n"
            "\n"
            "def compute_score(workspace, trajectory, private):\n"
            f"    {seed_call}\n"
            f"    return float({rng_call})\n"
        )
        issues = validator_module._scorer_determinism_issues("scorer/compute_score.py", src)
        assert any("no RNG" in issue for issue in issues), seed_call


def test_scorer_determinism_module_level_seed_suppresses_later_global_rng() -> None:
    src = (
        "import random\n"
        "import numpy as np\n"
        "\n"
        "np.random.seed(0)\n"
        "random.seed(0)\n"
        "\n"
        "def compute_score(workspace, trajectory, private):\n"
        "    return float(np.random.rand()) + random.random()\n"
    )
    assert validator_module._scorer_determinism_issues("scorer/compute_score.py", src) == []


def test_scorer_determinism_module_level_seed_after_function_still_applies() -> None:
    src = (
        "import random\n"
        "import numpy as np\n"
        "\n"
        "def compute_score(workspace, trajectory, private):\n"
        "    return float(np.random.rand()) + random.random()\n"
        "\n"
        "np.random.seed(0)\n"
        "random.seed(0)\n"
    )
    assert validator_module._scorer_determinism_issues("scorer/compute_score.py", src) == []


def test_scorer_determinism_module_level_rng_before_seed_still_flags() -> None:
    src = (
        "import random\n"
        "import numpy as np\n"
        "\n"
        "MODULE_SAMPLE = float(np.random.rand()) + random.random()\n"
        "\n"
        "np.random.seed(0)\n"
        "random.seed(0)\n"
        "\n"
        "def compute_score(workspace, trajectory, private):\n"
        "    return 1.0\n"
    )
    issues = validator_module._scorer_determinism_issues("scorer/compute_score.py", src)
    assert any("no RNG" in issue for issue in issues)


def test_scorer_determinism_module_level_init_helper_is_live() -> None:
    src = (
        "import time\n"
        "import numpy as np\n"
        "\n"
        "def _build_cache():\n"
        "    return time.time(), np.random.rand()\n"
        "\n"
        "CACHE = _build_cache()\n"
        "\n"
        "def compute_score(workspace, trajectory, private):\n"
        "    return 1.0\n"
    )
    issues = validator_module._scorer_determinism_issues("scorer/compute_score.py", src)
    assert any("wall-clock" in issue for issue in issues)
    assert any("no RNG" in issue for issue in issues)


def test_scorer_determinism_import_init_helper_seed_applies_globally() -> None:
    src = (
        "import random\n"
        "import numpy as np\n"
        "\n"
        "def _bootstrap_seed():\n"
        "    np.random.seed(0)\n"
        "    random.seed(0)\n"
        "\n"
        "_bootstrap_seed()\n"
        "\n"
        "def compute_score(workspace, trajectory, private):\n"
        "    return float(np.random.rand()) + random.random()\n"
    )
    assert validator_module._scorer_determinism_issues("scorer/compute_score.py", src) == []


def test_scorer_determinism_import_rng_before_seed_helper_still_flags() -> None:
    src = (
        "import random\n"
        "import numpy as np\n"
        "\n"
        "def _build_cache():\n"
        "    return float(np.random.rand()) + random.random()\n"
        "\n"
        "def _bootstrap_seed():\n"
        "    np.random.seed(0)\n"
        "    random.seed(0)\n"
        "\n"
        "CACHE = _build_cache()\n"
        "_bootstrap_seed()\n"
        "\n"
        "def compute_score(workspace, trajectory, private):\n"
        "    return 1.0\n"
    )
    issues = validator_module._scorer_determinism_issues("scorer/compute_score.py", src)
    assert any("no RNG" in issue for issue in issues)


def test_scorer_determinism_cross_function_seed_before_rng_allowed() -> None:
    src = (
        "import random\n"
        "import numpy as np\n"
        "\n"
        "def _seed():\n"
        "    np.random.seed(0)\n"
        "    random.seed(0)\n"
        "\n"
        "def _score():\n"
        "    return float(np.random.rand()) + random.random()\n"
        "\n"
        "def compute_score(workspace, trajectory, private):\n"
        "    _seed()\n"
        "    return _score()\n"
    )
    assert validator_module._scorer_determinism_issues("scorer/compute_score.py", src) == []


def test_scorer_determinism_multihop_seed_before_rng_allowed() -> None:
    src = (
        "import numpy as np\n"
        "\n"
        "def _seed():\n"
        "    np.random.seed(0)\n"
        "\n"
        "def _mid():\n"
        "    _seed()\n"
        "\n"
        "def compute_score(workspace, trajectory, private):\n"
        "    _mid()\n"
        "    return float(np.random.rand())\n"
    )
    assert validator_module._scorer_determinism_issues("scorer/compute_score.py", src) == []


def test_scorer_determinism_cross_function_rng_before_seed_still_flags() -> None:
    src = (
        "import random\n"
        "import numpy as np\n"
        "\n"
        "def _seed():\n"
        "    np.random.seed(0)\n"
        "    random.seed(0)\n"
        "\n"
        "def _score():\n"
        "    return float(np.random.rand()) + random.random()\n"
        "\n"
        "def compute_score(workspace, trajectory, private):\n"
        "    value = _score()\n"
        "    _seed()\n"
        "    return value\n"
    )
    issues = validator_module._scorer_determinism_issues("scorer/compute_score.py", src)
    assert any("no RNG" in issue for issue in issues)


def test_scorer_determinism_conditional_import_helper_is_not_live() -> None:
    src = (
        "import time\n"
        "\n"
        "def _debug_probe():\n"
        "    return time.time()\n"
        "\n"
        "if __name__ == '__main__':\n"
        "    _debug_probe()\n"
        "\n"
        "def compute_score(workspace, trajectory, private):\n"
        "    return 1.0\n"
    )
    assert validator_module._scorer_determinism_issues("scorer/compute_score.py", src) == []


def test_scorer_determinism_torch_manual_seed_does_not_seed_numpy_or_random() -> None:
    src = (
        "import random\n"
        "import numpy as np\n"
        "import torch\n"
        "\n"
        "def compute_score(workspace, trajectory, private):\n"
        "    torch.manual_seed(0)\n"
        "    return float(np.random.rand()) + random.random()\n"
    )
    issues = validator_module._scorer_determinism_issues("scorer/compute_score.py", src)
    assert any("no RNG" in issue for issue in issues)


def test_scorer_determinism_unqualified_rng_imports_are_checked() -> None:
    src = (
        "from numpy.random import default_rng, rand, seed\n"
        "from random import random as py_random\n"
        "from time import time\n"
        "\n"
        "def compute_score(workspace, trajectory, private):\n"
        "    rng = default_rng(None)\n"
        "    return float(rand()) + py_random() + time() + float(rng.random())\n"
    )
    issues = validator_module._scorer_determinism_issues("scorer/compute_score.py", src)
    assert any("no RNG" in issue for issue in issues)
    assert any("wall-clock" in issue for issue in issues)


def test_scorer_determinism_unqualified_seeded_imports_allowed() -> None:
    src = (
        "from numpy.random import rand, seed\n"
        "from random import random as py_random, seed as py_seed\n"
        "\n"
        "def compute_score(workspace, trajectory, private):\n"
        "    seed(0)\n"
        "    py_seed(0)\n"
        "    return float(rand()) + py_random()\n"
    )
    assert validator_module._scorer_determinism_issues("scorer/compute_score.py", src) == []


def test_scorer_determinism_flags_unseeded_rng_constructors() -> None:
    for call in ("np.random.RandomState()", "np.random.default_rng()", "random.Random()"):
        src = (
            "import random\n"
            "import numpy as np\n"
            "\n"
            "def compute_score(workspace, trajectory, private):\n"
            f"    rng = {call}\n"
            "    return 1.0\n"
        )
        issues = validator_module._scorer_determinism_issues("scorer/compute_score.py", src)
        assert any("no RNG" in issue for issue in issues), call


def test_scorer_determinism_allows_seeded_rng_constructors() -> None:
    for call in ("np.random.RandomState(0)", "np.random.default_rng(0)", "random.Random(0)"):
        src = (
            "import random\n"
            "import numpy as np\n"
            "\n"
            "def compute_score(workspace, trajectory, private):\n"
            f"    rng = {call}\n"
            "    return 1.0\n"
        )
        assert validator_module._scorer_determinism_issues("scorer/compute_score.py", src) == [], call


def test_scorer_determinism_flags_none_seed_rng_constructors() -> None:
    for call in (
        "np.random.RandomState(None)",
        "np.random.default_rng(None)",
        "np.random.default_rng(seed=None)",
        "random.Random(None)",
        "random.Random(x=None)",
    ):
        src = (
            "import random\n"
            "import numpy as np\n"
            "\n"
            "def compute_score(workspace, trajectory, private):\n"
            f"    rng = {call}\n"
            "    return 1.0\n"
        )
        issues = validator_module._scorer_determinism_issues("scorer/compute_score.py", src)
        assert any("no RNG" in issue for issue in issues), call


def test_scorer_determinism_flags_nonliteral_rng_constructor_seed() -> None:
    src = (
        "import numpy as np\n"
        "\n"
        "SEED = None\n"
        "\n"
        "def compute_score(workspace, trajectory, private):\n"
        "    rng = np.random.default_rng(SEED)\n"
        "    return 1.0\n"
    )
    issues = validator_module._scorer_determinism_issues("scorer/compute_score.py", src)
    assert any("no RNG" in issue for issue in issues)


def test_scorer_determinism_allows_keyword_literal_rng_constructor_seed() -> None:
    for call in ("np.random.default_rng(seed=0)", "random.Random(x=0)"):
        src = (
            "import random\n"
            "import numpy as np\n"
            "\n"
            "def compute_score(workspace, trajectory, private):\n"
            f"    rng = {call}\n"
            "    return 1.0\n"
        )
        assert validator_module._scorer_determinism_issues("scorer/compute_score.py", src) == [], call


def test_scorer_determinism_helper_module_ignores_dead_functions() -> None:
    src = "import time\n" "\n" "CONSTANT = 1.0\n" "\n" "def _unused_debug_probe():\n" "    return time.time()\n"
    assert validator_module._scorer_determinism_issues("scorer/utils.py", src) == []


def test_scorer_determinism_helper_module_flags_module_level_sources() -> None:
    src = "import time\n" "\n" "STARTED_AT = time.time()\n" "\n" "def helper():\n" "    return 1.0\n"
    issues = validator_module._scorer_determinism_issues("scorer/utils.py", src)
    assert any("wall-clock" in issue for issue in issues)


def test_prompt_quality_issues_flags_short_prompt() -> None:
    issues = validator_module._prompt_quality_issues(
        "instruction.md", "Use /data/ and write to /tmp/output/result.txt."
    )
    assert any("at least 200 characters" in issue for issue in issues)


def test_prompt_quality_issues_flags_missing_data_token() -> None:
    text = "x" * 250 + " write to /tmp/output/result.txt"
    issues = validator_module._prompt_quality_issues("instruction.md", text)
    assert any("/data/" in issue for issue in issues)
    assert not any("/tmp/output" in issue for issue in issues)


def test_prompt_quality_issues_flags_missing_output_token() -> None:
    text = "x" * 250 + " read from /data/train.csv"
    issues = validator_module._prompt_quality_issues("instruction.md", text)
    assert any("/tmp/output/" in issue for issue in issues)
    assert not any("data available in /data/" in issue for issue in issues)


def test_prompt_quality_issues_flags_emoji() -> None:
    text = _VALID_PROMPT + "\N{GRINNING FACE}"
    issues = validator_module._prompt_quality_issues("instruction.md", text)
    assert any("emoji" in issue for issue in issues)


def test_prompt_quality_issues_flags_ai_artifact_phrase() -> None:
    text = _VALID_PROMPT + " Certainly! As an AI, here is the plan."
    issues = validator_module._prompt_quality_issues("instruction.md", text)
    assert any("AI-artifact phrase" in issue for issue in issues)


def test_prompt_quality_issues_passes_clean_prompt() -> None:
    assert validator_module._prompt_quality_issues("instruction.md", _VALID_PROMPT) == []


def test_non_ascii_issues_locates_em_dash() -> None:
    text = "line one\nuse the dataset \u2014 then score it\nline three\n"
    issues = validator_module._non_ascii_issues("instruction.md", text)
    assert len(issues) == 1
    assert issues[0].startswith("instruction.md:2:")
    assert "U+2014" in issues[0]


def test_non_ascii_issues_locates_smart_quote() -> None:
    text = "the \u201csmart quote\u201d breaks json\n"
    issues = validator_module._non_ascii_issues("instruction.md", text)
    assert len(issues) == 1
    assert issues[0].startswith("instruction.md:1:")
    assert "U+201C" in issues[0]


def test_non_ascii_issues_passes_for_ascii() -> None:
    assert validator_module._non_ascii_issues("instruction.md", _VALID_PROMPT) == []


def test_prompt_quality_stage_passes_for_clean_prompt(tmp_path: Path) -> None:
    problem_dir = tmp_path / "clean-prompt"
    _write_problem(problem_dir, task_type="ml")

    stage = TaskValidator()._prompt_quality(problem_dir)

    assert stage.passed
    assert stage.issues == []


def test_prompt_quality_stage_fails_for_short_prompt(tmp_path: Path) -> None:
    problem_dir = tmp_path / "short-prompt"
    _write_problem(problem_dir, task_type="ml")
    (problem_dir / "instruction.md").write_text("Write to /tmp/output from /data/.\n")

    stage = TaskValidator()._prompt_quality(problem_dir)

    assert not stage.passed
    assert any("at least 200 characters" in issue for issue in stage.issues)


def test_prompt_quality_stage_fails_for_missing_paths(tmp_path: Path) -> None:
    problem_dir = tmp_path / "missing-paths"
    _write_problem(problem_dir, task_type="ml")
    (problem_dir / "data").mkdir()
    (problem_dir / "instruction.md").write_text("y" * 250 + "\n")

    stage = TaskValidator()._prompt_quality(problem_dir)

    assert not stage.passed
    assert any("/data/" in issue for issue in stage.issues)
    assert any("/tmp/output/" in issue for issue in stage.issues)


def test_prompt_quality_stage_skips_data_token_without_data_mount(
    tmp_path: Path,
) -> None:
    problem_dir = tmp_path / "sim-no-data"
    _write_problem(problem_dir, task_type="mujoco")
    (problem_dir / "instruction.md").write_text(
        "Control the simulated robot. The grader rolls out your policy in hidden "
        "MuJoCo episodes and scores the mean return over the rollouts. Write your "
        "policy implementation to /tmp/output/policy.py exposing act(obs).\n"
    )

    stage = TaskValidator()._prompt_quality(problem_dir)

    assert stage.passed
    assert stage.issues == []


def test_prompt_quality_stage_fails_for_emoji(tmp_path: Path) -> None:
    problem_dir = tmp_path / "emoji-prompt"
    _write_problem(problem_dir, task_type="ml")
    (problem_dir / "instruction.md").write_text(_VALID_PROMPT + "\N{ROCKET}\n")

    stage = TaskValidator()._prompt_quality(problem_dir)

    assert not stage.passed
    assert any("emoji" in issue for issue in stage.issues)


def test_prompt_quality_stage_fails_for_ai_artifact(tmp_path: Path) -> None:
    problem_dir = tmp_path / "ai-artifact-prompt"
    _write_problem(problem_dir, task_type="ml")
    (problem_dir / "instruction.md").write_text(_VALID_PROMPT + " Certainly! This is the dataset.\n")

    stage = TaskValidator()._prompt_quality(problem_dir)

    assert not stage.passed
    assert any("AI-artifact phrase" in issue for issue in stage.issues)


def test_prompt_quality_stage_fails_for_non_ascii_in_prompt(tmp_path: Path) -> None:
    problem_dir = tmp_path / "em-dash-prompt"
    _write_problem(problem_dir, task_type="ml")
    (problem_dir / "instruction.md").write_text(_VALID_PROMPT + "Then \u2014 finally \u2014 submit.\n")

    stage = TaskValidator()._prompt_quality(problem_dir)

    assert not stage.passed
    assert any("non-ASCII" in issue and "U+2014" in issue for issue in stage.issues)


def test_prompt_quality_stage_fails_for_non_ascii_in_scorer(tmp_path: Path) -> None:
    problem_dir = tmp_path / "non-ascii-scorer"
    _write_problem(problem_dir, task_type="ml")
    (problem_dir / "scorer" / "compute_score.py").write_text(
        "def compute_score(workspace, trajectory, private):\n" "    # caf\u00e9 score\n" "    return 1.0\n"
    )

    stage = TaskValidator()._prompt_quality(problem_dir)

    assert not stage.passed
    assert any(issue.startswith("scorer/compute_score.py:") and "non-ASCII" in issue for issue in stage.issues)


def test_prompt_quality_stage_fails_when_instruction_missing(tmp_path: Path) -> None:
    problem_dir = tmp_path / "no-instruction"
    _write_problem(problem_dir, task_type="ml")
    (problem_dir / "instruction.md").unlink()

    stage = TaskValidator()._prompt_quality(problem_dir)

    assert not stage.passed
    assert any("instruction.md is required" in issue for issue in stage.issues)


def test_prompt_quality_stage_scans_readme(tmp_path: Path) -> None:
    problem_dir = tmp_path / "readme-non-ascii"
    _write_problem(problem_dir, task_type="ml")
    (problem_dir / "README.md").write_text("Overview \u2014 see below.\n")

    stage = TaskValidator()._prompt_quality(problem_dir)

    assert not stage.passed
    assert any(issue.startswith("README.md:") and "non-ASCII" in issue for issue in stage.issues)


def test_baseline_trio_warns_when_fewer_than_three(tmp_path: Path) -> None:
    problem_dir = tmp_path / "ml-few-baselines"
    _write_problem(
        problem_dir, task_type="ml", reward_type="continuous_scoring_function"
    )
    (problem_dir / "baselines" / "linear").mkdir(parents=True)

    warnings = validator_module.baseline_trio_warnings(problem_dir)

    assert len(warnings) == 1
    assert "baseline trio advisory" in warnings[0]
    # Advisory only -- the conditional stage must NOT hard-fail on this.
    stage = TaskValidator()._conditional(problem_dir)
    assert stage.passed
    assert stage.warnings == warnings


def test_baseline_trio_clean_with_full_trio(tmp_path: Path) -> None:
    problem_dir = tmp_path / "ml-full-trio"
    _write_problem(
        problem_dir, task_type="ml", reward_type="continuous_scoring_function"
    )
    for name in ("naive", "linear", "gbt"):
        (problem_dir / "baselines" / name).mkdir(parents=True)

    assert validator_module.baseline_trio_warnings(problem_dir) == []


def test_baseline_trio_counts_flat_scripts(tmp_path: Path) -> None:
    problem_dir = tmp_path / "ml-flat-baselines"
    _write_problem(
        problem_dir, task_type="ml", reward_type="continuous_scoring_function"
    )
    baselines = problem_dir / "baselines"
    baselines.mkdir()
    for name in ("naive", "linear", "gbt"):
        (baselines / f"{name}.sh").write_text("#!/usr/bin/env bash\n")

    assert validator_module.baseline_trio_warnings(problem_dir) == []


def test_baseline_trio_advisory_scoped_to_ml_continuous(tmp_path: Path) -> None:
    # ml rubric task (not continuous) with zero baselines: no advisory.
    rubric = tmp_path / "ml-rubric"
    _write_problem(rubric, task_type="ml", reward_type="multi_deterministic_rubrics")
    assert validator_module.baseline_trio_warnings(rubric) == []

    # non-ml task with zero baselines: no advisory.
    mujoco = tmp_path / "mujoco-task"
    _write_problem(mujoco, task_type="mujoco")
    assert validator_module.baseline_trio_warnings(mujoco) == []
