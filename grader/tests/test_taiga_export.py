"""Tests for the Boreal submission payload builder.

The negative tests pin our "no worldsim CU/split-strategy fields ever
leak into the payload" invariant. If a future port accidentally
re-introduces `interaction_mode`, `cua_*`, `agentic_grader`, or top-level
`rubric` items, these fail loudly.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest

# conftest already adds REPO_ROOT to sys.path so `from alignerr_plugin import ...` resolves.
from alignerr_plugin.ground_truth import sha256_file
from alignerr_plugin.preloaded import PRELOADED_MANIFEST_PATH, manifest_entry
from alignerr_plugin.exporters.taiga import (
    QA_CPU_RESOURCE_ENV,
    _accelerator_notice,
    _derive_resources_from_toml,
    _max_required_resources,
    _task_type_hint,
    _taiga_container_runtime,
    build_batch_job_payload,
    build_job_payload,
    derive_taiga_resources,
    export_taiga,
    qa_cpu_flavor_for,
)
from alignerr_plugin.schemas import (
    Difficulty,
    EnvironmentSection,
    RunnerConfig,
    TaskToml,
)
from grading.evaluation.plan import canonical_plan_sha256

REPO_ROOT = Path(__file__).resolve().parents[2]

# A native-contract (task.toml) ml task written inline, used only by the few
# native-mode tests (base_flavor override, native resource validation). Kept as
# a helper, not a fixture directory.
_NATIVE_ML_TASK_TOML = """\
schema_version = "1.1"

[task]
name = "labelbox/native-ml-task"
description = "Native-contract ml task fixture for the Taiga export tests."

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


def _write_native_ml_task(
    problem_dir: Path,
    *,
    required_resources: str = "12vcpu+100gib+h100/2",
    instruction: str = _NATIVE_ML_TASK_INSTRUCTION,
) -> Path:
    problem_dir.mkdir(parents=True, exist_ok=True)
    task_toml = _NATIVE_ML_TASK_TOML.replace(
        'required_resources = "12vcpu+100gib+h100/2"',
        f'required_resources = "{required_resources}"',
    )
    (problem_dir / "task.toml").write_text(task_toml)
    (problem_dir / "metadata.json").write_text(
        json.dumps(
            {
                "benchmark": "taiga_task",
                "problem_data": {
                    "instance_id": "native-ml-task",
                    "description": "Native-contract ml task fixture for the Taiga export tests",
                },
            }
        )
    )
    (problem_dir / "instruction.md").write_text(instruction)
    (problem_dir / "scorer").mkdir(exist_ok=True)
    (problem_dir / "scorer" / "compute_score.py").write_text(
        "def compute_score(workspace, trajectory, private):\n    return 0.0\n"
    )
    return problem_dir


# ── Helpers ───────────────────────────────────────────────────────


_FORBIDDEN_FRAGMENTS = (
    "interaction_mode",
    "cua_",
    "agentic_grader",
    "task_rubric",
    "task_penalties",
    "penalty_avoidance",
)


def _assert_no_worldsim_leak(payload: dict, label: str) -> None:
    flat = json.dumps(payload)
    for forbidden in _FORBIDDEN_FRAGMENTS:
        assert (
            forbidden not in flat
        ), f"{label} payload leaked worldsim-only field: {forbidden!r}"


# ── Tests ─────────────────────────────────────────────────────────


def _resource_task_toml(
    required_resources: str,
    *,
    base_flavor: str = "auto",
) -> TaskToml:
    return TaskToml.model_construct(
        environment=EnvironmentSection(
            required_resources=required_resources,
            base_flavor=base_flavor,
        ),
        runner=RunnerConfig(),
        difficulty=Difficulty.model_construct(),
    )


def test_derive_resources_cpu_default(template_examples: Path) -> None:
    res = derive_taiga_resources(template_examples / "mujoco-pendulum")
    assert res["base_flavor"] == "cpu"
    assert res["base_image"].endswith("/lbx-tasks-base")
    assert res["base_image_ref"] == f"{res['base_image']}:{res['base_tag']}"
    assert res["api_model_name"] == "claude-fable-5"
    assert res["required_resources"] == "4vcpu+16gib"


def test_derive_resources_uses_required_resources_enum_verbatim() -> None:
    assert (
        _derive_resources_from_toml(_resource_task_toml("4vcpu+16gib"))[
            "required_resources"
        ]
        == "4vcpu+16gib"
    )
    assert (
        _derive_resources_from_toml(_resource_task_toml("12vcpu+100gib+h100/2"))[
            "base_flavor"
        ]
        == "gpu"
    )
    assert (
        _derive_resources_from_toml(
            _resource_task_toml("12vcpu+100gib+h100/2+graphics")
        )["base_flavor"]
        == "cuda-graphics"
    )
    assert (
        _derive_resources_from_toml(_resource_task_toml("13vcpu+32gib+tpuv5e1x1"))[
            "base_flavor"
        ]
        == "tpu"
    )


def test_derive_resources_openroad_gpu_flavor(
    template_examples: Path, tmp_path: Path
) -> None:
    problem_dir = tmp_path / "mle-openroad"
    _write_native_ml_task(problem_dir)
    task_toml = problem_dir / "task.toml"
    text = task_toml.read_text().replace(
        'required_resources = "12vcpu+100gib+h100/2"\n',
        'required_resources = "12vcpu+100gib+h100/2"\nbase_flavor = "gpu-openroad"\n',
    )
    task_toml.write_text(text)

    res = derive_taiga_resources(problem_dir)
    assert res["base_flavor"] == "gpu-openroad"
    assert res["base_image"].endswith("/lbx-tasks-base-gpu-openroad")
    assert res["base_image_ref"] == f"{res['base_image']}:{res['base_tag']}"
    assert res["required_resources"] == "12vcpu+100gib+h100/2"


def test_container_runtime_helper_cpu_vs_accelerator() -> None:
    assert _taiga_container_runtime("4vcpu+16gib") == "firecracker"
    assert _taiga_container_runtime("12vcpu+100gib+h100/2") == "gvisor"
    assert _taiga_container_runtime("12vcpu+100gib+h100/2+graphics") == "gvisor"
    assert _taiga_container_runtime("13vcpu+32gib+tpuv5e1x1") == "gvisor"


def test_batch_max_required_resources_uses_capacity_not_openapi_order() -> None:
    assert (
        _max_required_resources(
            [
                "16vcpu+64gib+tpuv5e2x2",
                "50vcpu+128gib+tpuv5e2x2",
            ]
        )
        == "50vcpu+128gib+tpuv5e2x2"
    )


def test_export_cpu_task_runs_under_firecracker(template_examples: Path) -> None:
    p = build_job_payload(
        template_examples / "mujoco-pendulum",
        image_ref="gcr.io/example/img@sha256:abc",
    )
    problem_set = p["problems_metadata"]["problem_set"]
    problem = problem_set["problems"][0]
    assert problem_set["container_runtime"] == "firecracker"
    assert problem["container_runtime"] == "firecracker"
    assert problem["metadata"]["base_image"]["flavor"] == "cpu"
    assert problem["metadata"]["base_image"]["ref"].endswith(
        f":{problem['metadata']['base_image']['tag']}"
    )


def test_export_gpu_task_runs_under_gvisor(tmp_path: Path) -> None:
    # Accelerator tasks must export gVisor, not the firecracker default (which
    # would fail on Taiga's GPU nodes).
    problem_dir = _write_native_ml_task(tmp_path / "mle-gvisor")
    p = build_job_payload(problem_dir, image_ref="gcr.io/example/img@sha256:abc")
    problem_set = p["problems_metadata"]["problem_set"]
    problem = problem_set["problems"][0]
    assert problem_set["container_runtime"] == "gvisor"
    assert problem["container_runtime"] == "gvisor"


def test_export_cpu_task_injects_tmux_notice_only(template_examples: Path) -> None:
    p = build_job_payload(
        template_examples / "mujoco-pendulum",
        image_ref="gcr.io/example/img@sha256:abc",
    )
    problem = p["problems_metadata"]["problem_set"]["problems"][0]
    prompt = problem["task_prompt"]
    assert "dedicated tmux tool" in prompt
    assert "not tmux inside the bash tool" in prompt
    assert "A GPU may be available." not in prompt
    assert "A TPU may be available." not in prompt


def test_export_gpu_task_injects_gpu_and_tmux_notices(tmp_path: Path) -> None:
    problem_dir = _write_native_ml_task(tmp_path / "mle-no-runtime-guidance")
    p = build_job_payload(problem_dir, image_ref="gcr.io/example/img@sha256:abc")
    problem = p["problems_metadata"]["problem_set"]["problems"][0]
    prompt = problem["task_prompt"]
    assert "A GPU may be available." in prompt
    assert "dedicated tmux tool" in prompt
    assert "A TPU may be available." not in prompt


def test_export_tpu_task_injects_tpu_notice(tmp_path: Path) -> None:
    problem_dir = _write_native_ml_task(
        tmp_path / "mle-tpu", required_resources="13vcpu+32gib+tpuv5e1x1"
    )
    p = build_job_payload(problem_dir, image_ref="gcr.io/example/img@sha256:abc")
    problem_set = p["problems_metadata"]["problem_set"]
    assert problem_set["required_resources"] == "13vcpu+32gib+tpuv5e1x1"
    problem = p["problems_metadata"]["problem_set"]["problems"][0]
    assert problem["required_resources"] == "13vcpu+32gib+tpuv5e1x1"
    assert problem["container_runtime"] == "gvisor"
    prompt = problem["task_prompt"]
    assert "A TPU may be available." in prompt
    assert "A GPU may be available." not in prompt


def test_export_tpu_task_rejects_unsupported_resource_request(
    template_examples: Path, tmp_path: Path
) -> None:
    problem_dir = tmp_path / "mle-tpu-too-large"
    _write_native_ml_task(problem_dir)
    task_toml = problem_dir / "task.toml"
    text = task_toml.read_text()
    text = text.replace(
        'required_resources = "12vcpu+100gib+h100/2"',
        'required_resources = "13vcpu+32gib+tpu-v5e/8"',
    )
    task_toml.write_text(text)

    with pytest.raises(ValueError, match="required_resources must be one of"):
        derive_taiga_resources(problem_dir)


def test_export_accelerator_notice_appends_despite_authored_guidance(
    tmp_path: Path,
) -> None:
    problem_dir = _write_native_ml_task(
        tmp_path / "mle-with-guidance",
        instruction="Use the GPU and keep long runs in tmux. Write /tmp/output/submission.csv.\n",
    )
    p = build_job_payload(problem_dir, image_ref="gcr.io/example/img@sha256:abc")
    problem = p["problems_metadata"]["problem_set"]["problems"][0]
    prompt = problem["task_prompt"]
    assert prompt.count("A GPU may be available.") == 1
    assert prompt.count("dedicated tmux tool") == 0


def test_export_tpu_notice_not_suppressed_by_gpu_wording(tmp_path: Path) -> None:
    problem_dir = _write_native_ml_task(
        tmp_path / "mle-tpu-with-gpu-word",
        required_resources="13vcpu+32gib+tpuv5e1x1",
        instruction="The starter mentions GPU in passing. Write /tmp/output/submission.csv.\n",
    )
    p = build_job_payload(problem_dir, image_ref="gcr.io/example/img@sha256:abc")
    problem = p["problems_metadata"]["problem_set"]["problems"][0]
    prompt = problem["task_prompt"]
    assert "A TPU may be available." in prompt


def test_export_gpu_notice_not_suppressed_by_tpu_wording(tmp_path: Path) -> None:
    problem_dir = _write_native_ml_task(
        tmp_path / "mle-gpu-with-tpu-word",
        instruction="This is not a TPU task. Write /tmp/output/submission.csv.\n",
    )
    p = build_job_payload(problem_dir, image_ref="gcr.io/example/img@sha256:abc")
    problem = p["problems_metadata"]["problem_set"]["problems"][0]
    prompt = problem["task_prompt"]
    assert "A GPU may be available." in prompt


def test_ml_starter_boilerplate_does_not_suppress_prompt_notices(
    tmp_path: Path,
) -> None:
    src = (
        REPO_ROOT
        / "alignerr_plugin"
        / "src"
        / "alignerr_plugin"
        / "starter_templates"
        / "ml"
    )
    problem_dir = tmp_path / "ml-starter"
    shutil.copytree(src, problem_dir)
    p = build_job_payload(problem_dir, image_ref="gcr.io/example/img@sha256:abc")
    problem = p["problems_metadata"]["problem_set"]["problems"][0]
    prompt = problem["task_prompt"]
    # The injected GPU + tmux notices must survive the scaffold's boilerplate prompt.
    assert "A GPU may be available." in prompt
    assert "dedicated tmux tool" in prompt
    assert "A TPU may be available." not in prompt


def test_export_task_toml_hints_as_native_taiga_hints_for_non_target_type(
    template_examples: Path, tmp_path: Path
) -> None:
    problem_dir = tmp_path / "mujoco-with-hint"
    shutil.copytree(template_examples / "mujoco-pendulum", problem_dir)
    task_toml = problem_dir / "task.toml"
    task_toml.write_text(
        task_toml.read_text()
        + '\n[[hint]]\ntext = "Start with a proportional controller."\n'
        + "enabled = false\nspoiler_level = 0.25\n"
    )

    p = build_job_payload(problem_dir, image_ref="gcr.io/example/img@sha256:abc")
    problem = p["problems_metadata"]["problem_set"]["problems"][0]
    assert problem["hints"] == [
        {
            "message": "Start with a proportional controller.",
            "enabled": False,
            "spoiler_level": 0.25,
        }
    ]
    assert "hints" not in problem["extra_fields"]


def test_export_preloaded_files_without_prompt_mount_notice(
    template_examples: Path, tmp_path: Path
) -> None:
    problem_dir = tmp_path / "mujoco-with-preloaded-files"
    shutil.copytree(template_examples / "mujoco-pendulum", problem_dir)
    manifest = [
        manifest_entry("task/m1-public.squashfs", "/data"),
        manifest_entry("task/m2-private.squashfs", "/mcp_server/data"),
    ]
    manifest_path = problem_dir / PRELOADED_MANIFEST_PATH
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps({"preloaded_files": manifest}) + "\n")

    p = build_job_payload(problem_dir, image_ref="gcr.io/example/img@sha256:abc")
    problem = p["problems_metadata"]["problem_set"]["problems"][0]

    assert problem["preloaded_files"] == manifest
    prompt = problem["task_prompt"]
    assert "mounted read-only" not in prompt
    assert "do not attempt to download or modify" not in prompt
    assert "preloaded_files" not in prompt
    assert "/mcp_server/data" not in prompt


def test_export_cfd_solver_availability_as_enabled_taiga_hint(
    template_examples: Path,
) -> None:
    p = build_job_payload(
        template_examples / "openfoam-hydrofoil-flap",
        image_ref="gcr.io/example/img@sha256:abc",
    )
    problem = p["problems_metadata"]["problem_set"]["problems"][0]

    assert not problem["task_prompt"].startswith("## OpenFOAM Availability")
    assert problem["hints"][0] == {
        "message": _task_type_hint("cfd").strip(),
        "enabled": True,
    }
    assert "hints" not in problem["extra_fields"]


def test_export_structures_solver_availability_as_enabled_taiga_hint(
    template_examples: Path,
) -> None:
    p = build_job_payload(
        template_examples / "opensees-base-isolation",
        image_ref="gcr.io/example/img@sha256:abc",
    )
    problem = p["problems_metadata"]["problem_set"]["problems"][0]

    assert not problem["task_prompt"].startswith("## OpenSees Availability")
    assert problem["hints"][0] == {
        "message": _task_type_hint("structures").strip(),
        "enabled": True,
    }
    assert "hints" not in problem["extra_fields"]
    assert problem["extra_fields"]["evaluation_plan"]["plan_sha256"]
    assert problem["extra_fields"]["rubric_evaluation"] == {
        "required": True,
        "security_tier": "sealed_rescore",
        "attestation_required": True,
        "trace_required": True,
    }


@pytest.mark.parametrize(
    ("starter_name", "task_type", "heading"),
    [
        ("prometheus-cfd", "cfd", "## OpenFOAM Availability"),
        ("prometheus-structures", "structures", "## OpenSees Availability"),
        ("prometheus-eval-cfd", "cfd", "## OpenFOAM Availability"),
        ("prometheus-eval-structures", "structures", "## OpenSees Availability"),
    ],
)
def test_prometheus_numerical_solver_taiga_export_uses_native_hint(
    tmp_path: Path,
    starter_name: str,
    task_type: str,
    heading: str,
) -> None:
    src = (
        REPO_ROOT
        / "alignerr_plugin"
        / "src"
        / "alignerr_plugin"
        / "starter_templates"
        / starter_name
    )
    problem_dir = tmp_path / starter_name
    shutil.copytree(src, problem_dir)

    payload = build_job_payload(
        problem_dir,
        image_ref="gcr.io/example/img@sha256:abc",
    )
    problem = payload["problems_metadata"]["problem_set"]["problems"][0]

    assert heading not in problem["task_prompt"]
    assert problem["hints"][0] == {
        "message": _task_type_hint(task_type).strip(),
        "enabled": True,
    }


@pytest.mark.parametrize(
    "starter_name",
    [
        "ml",
        "mujoco",
        "cfd",
        "structures",
    ],
)
def test_export_task_toml_outputs_for_all_task_types(
    tmp_path: Path, starter_name: str
) -> None:
    src = (
        REPO_ROOT
        / "alignerr_plugin"
        / "src"
        / "alignerr_plugin"
        / "starter_templates"
        / starter_name
    )
    problem_dir = tmp_path / f"{starter_name}-starter"
    shutil.copytree(src, problem_dir)
    p = build_job_payload(
        problem_dir,
        image_ref="gcr.io/example/img@sha256:abc",
    )
    problem = p["problems_metadata"]["problem_set"]["problems"][0]

    assert problem["outputs"]
    assert problem["outputs"] == problem["extra_fields"]["task_metadata"]["outputs"]
    for output in problem["outputs"]:
        assert set(output) == {"path", "required", "description"}
        assert output["path"].startswith("/tmp/output/")
        assert isinstance(output["required"], bool)
        assert isinstance(output["description"], str)


def test_build_job_payload_default_shape(template_examples: Path) -> None:
    p = build_job_payload(
        template_examples / "mujoco-pendulum",
        image_ref="gcr.io/example/img@sha256:abc",
    )

    _assert_no_worldsim_leak(p, "single-problem")

    problem_set = p["problems_metadata"]["problem_set"]
    assert problem_set["name"] == "lbx_rl_tasks_mujoco"
    problem = p["problems_metadata"]["problem_set"]["problems"][0]
    # The grader is image-baked. Per-criterion detail flows back via the
    # in-image MCP `Grade.metadata.structured_subscores`. So the Boreal
    # payload should always carry an empty rubric and a single mcp strategy.
    assert problem["rubric"] == []
    assert problem["grading_strategy"] == [{"type": "mcp", "weight": 1.0}]
    shim = problem["extra_fields"]["test_file"]
    assert "/runtime/grading/src" in shim
    assert "/mcp_server/grader/compute_score.py" in shim
    assert "RubricTask" in shim
    assert "_task.grade(" in shim
    assert 'sys.path.insert(0, "/mcp_server")' not in shim
    # The shim forwards the agent transcript (injected as a TRANSCRIPT global)
    # instead of the old hardcoded empty trajectory, so transcript-based
    # anti-cheat checks are reachable.
    assert 'trajectory=globals().get("TRANSCRIPT") or []' in shim
    assert "trajectory=[]" not in shim
    # Startup exec's the prebuilt venv binary directly (no per-start uv resolve
    # that blew the 120s MCP init timeout under load).
    assert problem["startup_command"] == "/opt/lbx-runtime/.venv/bin/rubric mcp"
    assert problem["enable_anthropic_api"] is False

    # Per-problem classification metadata for Boreal UI.
    md = problem["metadata"]
    assert "difficulty" not in md
    assert "task_type" in md
    assert "interaction_mode" not in md  # CU/CUA never appears

    # Defaults for runner config
    assert p["api_model_name"] == "claude-fable-5"
    assert p["n_attempts_per_problem"] == 3  # task.toml [runner] default
    assert p["max_ctx"] == 1_000_000
    assert p["priority"] == "high"
    assert p["iteration_order"] == "problems_first"


def test_build_job_payload_preserves_explicit_anthropic_api_opt_in(
    template_examples: Path,
) -> None:
    p = build_job_payload(
        template_examples / "opensees-base-isolation",
        image_ref="gcr.io/example/img@sha256:abc",
    )
    problem = p["problems_metadata"]["problem_set"]["problems"][0]
    assert problem["enable_anthropic_api"] is True


def test_build_job_payload_includes_redacted_ground_truth_evidence(
    template_examples: Path,
) -> None:
    problem_dir = template_examples / "mujoco-pendulum"
    p = build_job_payload(
        problem_dir,
        image_ref="gcr.io/example/img@sha256:abc",
    )

    problem = p["problems_metadata"]["problem_set"]["problems"][0]
    evidence = problem["extra_fields"]["ground_truth_evidence"]
    proof_path = problem_dir / ".alignerr" / "build_proof.json"

    assert set(evidence) == {
        "schema_version",
        "source",
        "build_proof_sha256",
        "task_type",
        "domain",
        "reward_type",
        "expected_score",
        "score_epsilon",
        "passed",
        "score",
        "runtime",
        "graded_at",
        "review_artifacts",
    }
    assert evidence["schema_version"] == 1
    assert evidence["source"] == ".alignerr/build_proof.json"
    assert evidence["build_proof_sha256"] == sha256_file(proof_path)
    assert evidence["task_type"] == "mujoco"
    assert evidence["domain"] == "model_environment_construction"
    assert evidence["reward_type"] == "multi_deterministic_rubrics"
    assert evidence["expected_score"] == 1.0
    assert evidence["passed"] is True
    assert evidence["score"] == 1.0
    assert evidence["runtime"] == "solution"

    artifact = evidence["review_artifacts"][0]
    assert set(artifact) == {
        "path",
        "logical_path",
        "sha256",
        "bytes",
        "width",
        "height",
    }
    assert artifact["path"] == ".alignerr/ground_truth/rendering.mp4"
    assert artifact["logical_path"] == "/tmp/output/rendering.mp4"
    assert artifact["width"] == 1280
    assert artifact["height"] == 720

    flat_evidence = json.dumps(evidence)
    assert "run_dir" not in flat_evidence
    assert "reward_path" not in flat_evidence
    assert "details_path" not in flat_evidence
    assert "metadata" not in flat_evidence
    assert "structured_subscores" not in flat_evidence


def test_build_job_payload_omits_ground_truth_evidence_without_proof(
    tmp_path: Path,
) -> None:
    problem_dir = _write_native_ml_task(tmp_path / "mle-no-proof")
    p = build_job_payload(problem_dir, image_ref="gcr.io/example/img@sha256:abc")

    problem = p["problems_metadata"]["problem_set"]["problems"][0]
    assert "ground_truth_evidence" not in problem["extra_fields"]
    assert problem["extra_fields"]["continuous_evaluation"]["required"] is True
    assert "calibration" not in problem["extra_fields"]


def test_trusted_export_rejects_continuous_task_without_evidence(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    problem_dir = _write_native_ml_task(tmp_path / "mle-missing-evidence")
    monkeypatch.setenv("LBX_REQUIRE_TRUSTED_CONTINUOUS_EVALUATION", "1")

    with pytest.raises(ValueError, match="requires trusted calibration"):
        build_job_payload(
            problem_dir,
            image_ref="gcr.io/example/img@sha256:abc",
        )


def test_build_job_payload_includes_calibration_identity(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    problem_dir = _write_native_ml_task(tmp_path / "mle-calibration")
    trusted_dir = tmp_path / "trusted-calibration"
    trusted_dir.mkdir()
    lock = {
        "schema_version": "3.0",
        "policy": "continuous-pwl-v3",
        "task_spec_sha256": "a" * 64,
        "evaluation_plan_sha256": "b" * 64,
        "evaluation_plan": {"security_tier": "sealed_rescore"},
        "inputs": {},
        "qualification": {},
    }
    lock_path = trusted_dir / "calibration.lock.json"
    lock_path.write_text(json.dumps(lock, sort_keys=True) + "\n")
    trusted_dir.joinpath("calibration.evidence.json").write_text(
        json.dumps(
            {
                "schema_version": "continuous-calibration-evidence.v1",
                "cache_key": "c" * 64,
                "lock_sha256": sha256_file(lock_path),
                "task_spec_sha256": "a" * 64,
                "evaluation_plan_sha256": "b" * 64,
                "security_tier": "sealed_rescore",
                "inputs": {},
                "qualification": {},
            }
        )
        + "\n"
    )
    monkeypatch.setenv("LBX_TRUSTED_CALIBRATION_DIR", str(trusted_dir))

    payload = build_job_payload(problem_dir, image_ref="gcr.io/example/img@sha256:abc")
    evidence = payload["problems_metadata"]["problem_set"]["problems"][0][
        "extra_fields"
    ]["calibration"]
    continuous = payload["problems_metadata"]["problem_set"]["problems"][0][
        "extra_fields"
    ]["continuous_evaluation"]

    assert evidence == {
        "schema_version": 1,
        "lock_sha256": sha256_file(lock_path),
        "task_spec_sha256": "a" * 64,
        "evaluation_plan_sha256": "b" * 64,
        "security_tier": "sealed_rescore",
        "policy": "continuous-pwl-v3",
        "requires_trusted_mount": True,
    }
    assert continuous == {
        "required": True,
        "security_tier": "sealed_rescore",
        "attestation_required": True,
        "trace_required": True,
    }

    local_payload = build_job_payload(problem_dir, image_ref="LOCAL_IMAGE")
    assert (
        local_payload["problems_metadata"]["problem_set"]["problems"][0][
            "extra_fields"
        ]["calibration"]["requires_trusted_mount"]
        is False
    )
    local_continuous = local_payload["problems_metadata"]["problem_set"]["problems"][0][
        "extra_fields"
    ]["continuous_evaluation"]
    assert local_continuous["attestation_required"] is False
    assert local_continuous["trace_required"] is False


def test_build_job_payload_ignores_orphan_lock_for_non_continuous_task(
    tmp_path: Path,
) -> None:
    problem_dir = _write_native_ml_task(tmp_path / "non-continuous-orphan-lock")
    task_toml = problem_dir / "task.toml"
    task_toml.write_text(
        task_toml.read_text().replace(
            'reward_type = "continuous_scoring_function"',
            'reward_type = "multi_deterministic_rubrics"',
        )
    )
    problem_dir.joinpath("calibration.lock.json").write_text("{}\n")

    payload = build_job_payload(
        problem_dir,
        image_ref="LOCAL_IMAGE",
    )
    extra_fields = payload["problems_metadata"]["problem_set"]["problems"][0][
        "extra_fields"
    ]

    assert "calibration" not in extra_fields


def test_build_job_payload_rejects_stale_lock_for_continuous_task(
    tmp_path: Path,
) -> None:
    problem_dir = _write_native_ml_task(tmp_path / "continuous-stale-lock")
    problem_dir.joinpath("calibration.lock.json").write_text("{}\n")

    with pytest.raises(
        ValueError, match="trusted continuous calibration evidence is missing or stale"
    ):
        build_job_payload(
            problem_dir,
            image_ref="gcr.io/example/img@sha256:abc",
        )


def test_build_job_payload_includes_native_evaluation_plan(tmp_path: Path) -> None:
    problem_dir = _write_native_ml_task(tmp_path / "native-evaluation")
    scorer = problem_dir / "scorer"
    plan = {
        "schema_version": "continuous-evaluation-plan.v1",
        "security_tier": "sealed_challenge",
    }
    plan["plan_sha256"] = canonical_plan_sha256(plan)
    plan_path = scorer / "evaluation.plan.json"
    plan_path.write_text(json.dumps(plan, sort_keys=True) + "\n")

    payload = build_job_payload(problem_dir, image_ref="gcr.io/example/img@sha256:abc")
    evidence = payload["problems_metadata"]["problem_set"]["problems"][0][
        "extra_fields"
    ]["evaluation_plan"]

    assert evidence == {
        "schema_version": 1,
        "path": "evaluation.plan.json",
        "sha256": sha256_file(plan_path),
        "plan_sha256": plan["plan_sha256"],
        "security_tier": "sealed_challenge",
        "requires_trusted_mount": True,
    }


def test_build_job_payload_uses_reward_type_for_continuous_ground_truth(
    tmp_path: Path,
) -> None:
    problem_dir = _write_native_ml_task(tmp_path / "mle-gt")
    proof_dir = problem_dir / ".alignerr"
    proof_dir.mkdir(exist_ok=True)
    (proof_dir / "build_proof.json").write_text(
        json.dumps(
            {
                "ground_truth_result": {
                    "score": 0.0,
                    "runtime": "solution",
                    "graded_at": "2026-05-31T00:00:00+00:00",
                }
            }
        )
    )

    p = build_job_payload(
        problem_dir,
        image_ref="gcr.io/example/img@sha256:abc",
    )

    problem = p["problems_metadata"]["problem_set"]["problems"][0]
    evidence = problem["extra_fields"]["ground_truth_evidence"]
    assert evidence["score"] == 0.0
    assert evidence["runtime"] == "solution"
    assert evidence["task_type"] == "ml"
    assert evidence["domain"] == "scientific_discovery_computational_science"
    assert evidence["reward_type"] == "continuous_scoring_function"
    assert evidence["expected_score"] == 0.5
    assert evidence["score_epsilon"] == 0.05
    assert evidence["passed"] is False


def test_build_job_payload_caller_overrides_runner_defaults(
    template_examples: Path,
) -> None:
    p = build_job_payload(
        template_examples / "mujoco-pendulum",
        image_ref="gcr.io/example/img@sha256:abc",
        n_attempts=5,
        turn_limit=2000,
        max_ctx=500_000,
        model="claude-sonnet-4-5",
    )
    assert p["n_attempts_per_problem"] == 5
    assert p["turn_limit"] == 2000
    assert p["max_ctx"] == 500_000
    assert p["api_model_name"] == "claude-sonnet-4-5"


def test_build_job_payload_ships_runner_timeouts(template_examples: Path) -> None:
    """A non-ml task that omits [runner.timeouts] keeps the low RunnerTimeouts
    defaults (the ml hour-scale pin must NOT leak into other task types)."""
    # mujoco-pendulum declares no [runner.timeouts] and is not an ml task, so
    # this exercises the RunnerTimeouts schema defaults flowing through unchanged.
    p = build_job_payload(
        template_examples / "mujoco-pendulum",
        image_ref="gcr.io/example/img@sha256:abc",
    )
    problem = p["problems_metadata"]["problem_set"]["problems"][0]
    assert problem["setup_timeout_seconds"] == 600
    assert problem["grading_timeout_seconds"] == 600
    assert problem["tool_timeout_seconds"] == 120
    # max_episode_sec is job-level, not per-problem.
    assert p["max_timeout_seconds"] == 3600


def test_build_job_payload_non_ml_timeouts_stay_author_overridable(
    template_examples: Path, tmp_path: Path
) -> None:
    """Non-ml tasks may still override the (now higher) defaults downward."""
    problem_dir = tmp_path / "mujoco-custom-timeouts"
    shutil.copytree(template_examples / "mujoco-pendulum", problem_dir)
    task_toml = problem_dir / "task.toml"
    task_toml.write_text(
        task_toml.read_text()
        + "\n[runner.timeouts]\n"
        + "setup_sec = 900\ngrading_sec = 1200\ntool_sec = 300\nmax_episode_sec = 4000\n"
    )
    p = build_job_payload(problem_dir, image_ref="gcr.io/example/img@sha256:abc")
    problem = p["problems_metadata"]["problem_set"]["problems"][0]
    assert problem["setup_timeout_seconds"] == 900
    assert problem["grading_timeout_seconds"] == 1200
    assert problem["tool_timeout_seconds"] == 300
    assert p["max_timeout_seconds"] == 4000


def test_ml_task_timeouts_force_pinned_to_hour_scale(tmp_path: Path) -> None:
    """ml tasks ignore author timeouts entirely: all four fields are force-pinned
    to the hour-scale platform defaults regardless of what task.toml carries."""
    problem_dir = tmp_path / "native-ml-low-timeouts"
    _write_native_ml_task(problem_dir)
    task_toml = problem_dir / "task.toml"
    # Author tries to set minutes-scale timeouts; the ml pin must override them.
    task_toml.write_text(
        task_toml.read_text().replace(
            "[runner.timeouts]\n"
            "setup_sec = 7200\n"
            "grading_sec = 5400\n"
            "tool_sec = 21600\n"
            "max_episode_sec = 21600\n",
            "[runner.timeouts]\n"
            "setup_sec = 600\n"
            "grading_sec = 600\n"
            "tool_sec = 120\n"
            "max_episode_sec = 3600\n",
        )
    )
    p = build_job_payload(problem_dir, image_ref="gcr.io/example/img@sha256:abc")
    problem = p["problems_metadata"]["problem_set"]["problems"][0]
    assert problem["setup_timeout_seconds"] == 7200
    assert problem["grading_timeout_seconds"] == 10800
    assert problem["extra_fields"]["grading_timeout_seconds"] == 10800
    assert problem["tool_timeout_seconds"] == 21600
    assert p["max_timeout_seconds"] == 21600


def test_batch_payload_rejects_missing_image_ref(template_examples: Path) -> None:
    with pytest.raises(ValueError, match="image_refs missing entry"):
        build_batch_job_payload(
            [template_examples / "mujoco-pendulum"],
            image_refs={},  # no entries
        )


def test_batch_payload_applies_container_runtime_override(
    template_examples: Path,
) -> None:
    p = build_batch_job_payload(
        [template_examples / "mujoco-pendulum"],
        image_refs={"mujoco-pendulum": "gcr.io/example/img@sha256:abc"},
        overrides={
            "required_resources": "24vcpu+200gib+h100/1",
            "container_runtime": "custom-runtime",
        },
    )
    ps = p["problems_metadata"]["problem_set"]
    assert ps["container_runtime"] == "custom-runtime"
    assert ps["problems"][0]["container_runtime"] == "custom-runtime"


def test_export_taiga_writes_legacy_problems_metadata(
    template_examples: Path, tmp_path: Path
) -> None:
    out = tmp_path / "problems-metadata.json"
    sidecar = export_taiga(
        template_examples / "mujoco-pendulum",
        out,
        image_ref="gcr.io/example/img@sha256:abc",
    )
    assert out.exists()
    data = json.loads(out.read_text())
    pset = data["problem_set"]
    assert pset["owner"] == "labelbox"
    assert pset["name"] == "lbx_rl_tasks_mujoco"
    assert pset["problems"][0]["id"] == "mujoco-pendulum"
    assert "difficulty" not in pset["problems"][0]["metadata"]
    assert pset["problems"][0]["extra_fields"]["ground_truth_evidence"]["score"] == 1.0

    # Sidecar carries the resource derivation that the workflow reads
    assert sidecar["task_id"] == "mujoco-pendulum"
    assert sidecar["base_flavor"] == "cpu"
    assert sidecar["base_image"].endswith("/lbx-tasks-base")
    assert sidecar["base_image_ref"] == f"{sidecar['base_image']}:{sidecar['base_tag']}"


# --- CPU QA lane: LBX_TAIGA_QA_CPU_RESOURCE override ------------------------

QA_CPU_TIER = "8vcpu+64gib"


def test_qa_cpu_flavor_for_maps_every_flavor_to_the_native_cpu_base():
    # Every task now builds on a flagship native base, so the CPU QA lane has a
    # single fallback: there is no accelerator-specific CPU variant to pick.
    for flavor in (
        "gpu",
        "tpu",
        "cuda-graphics",
        "cpu",
        "gpu-openroad",
        "gpu-blackwell",
    ):
        assert qa_cpu_flavor_for(flavor) == "cpu"


@pytest.mark.parametrize(
    "tier,declared,flavor",
    [
        ("12vcpu+100gib+h100/2", "auto", "gpu"),
        ("13vcpu+32gib+tpuv5e1x1", "auto", "tpu"),
        ("4vcpu+16gib", "auto", "cpu"),
        ("12vcpu+100gib+h100/2+graphics", "cuda-graphics", "cuda-graphics"),
    ],
)
def test_no_override_leaves_the_declared_tier_untouched(
    monkeypatch, tier, declared, flavor
):
    monkeypatch.delenv(QA_CPU_RESOURCE_ENV, raising=False)
    resources = _derive_resources_from_toml(
        _resource_task_toml(tier, base_flavor=declared)
    )
    assert resources["required_resources"] == tier
    assert resources["base_flavor"] == flavor


def test_override_forces_cpu_tier_and_native_cpu_base(monkeypatch):
    monkeypatch.setenv(QA_CPU_RESOURCE_ENV, QA_CPU_TIER)
    resources = _derive_resources_from_toml(
        _resource_task_toml("13vcpu+32gib+tpuv5e1x1")
    )
    assert resources["required_resources"] == QA_CPU_TIER
    assert resources["base_flavor"] == "cpu"
    # The image/tag follow the swapped flavor, not the declared TPU one.
    assert resources["base_image"].endswith("/lbx-tasks-base")
    assert (
        resources["base_image_ref"]
        == f"{resources['base_image']}:{resources['base_tag']}"
    )


def test_override_swaps_an_explicit_accelerator_flavor_to_native_cpu(monkeypatch):
    monkeypatch.setenv(QA_CPU_RESOURCE_ENV, QA_CPU_TIER)
    # An explicitly declared accelerator flavor must not survive the CPU QA lane:
    # the JAX-only tpu base has no CPU tier to run on.
    resources = _derive_resources_from_toml(
        _resource_task_toml("13vcpu+32gib+tpuv5e1x1", base_flavor="tpu")
    )
    assert resources["required_resources"] == QA_CPU_TIER
    assert resources["base_flavor"] == "cpu"
    assert resources["base_image"].endswith("/lbx-tasks-base")


def test_override_rejects_an_accelerator_tier(monkeypatch):
    # The QA lane is the one lane that submits without the deploy-to-taiga reviewer
    # gate, so a misconfigured tier must refuse to export rather than quietly
    # auto-submitting scarce compute.
    for bad in ("12vcpu+100gib+h100/2", "13vcpu+32gib+tpuv5e1x1"):
        monkeypatch.setenv(QA_CPU_RESOURCE_ENV, bad)
        with pytest.raises(ValueError, match="must be a CPU required_resources tier"):
            _derive_resources_from_toml(_resource_task_toml("12vcpu+100gib+h100/2"))


def test_override_rejects_a_tier_outside_the_enum(monkeypatch):
    monkeypatch.setenv(QA_CPU_RESOURCE_ENV, "999vcpu+1gib")
    with pytest.raises(ValueError):
        _derive_resources_from_toml(_resource_task_toml("4vcpu+16gib"))


def test_blank_override_is_ignored(monkeypatch):
    monkeypatch.setenv(QA_CPU_RESOURCE_ENV, "   ")
    resources = _derive_resources_from_toml(_resource_task_toml("12vcpu+100gib+h100/2"))
    assert resources["required_resources"] == "12vcpu+100gib+h100/2"
    assert resources["base_flavor"] == "gpu"


def test_override_still_validates_a_bogus_declared_flavor(monkeypatch):
    # The declared flavor is resolved against the ORIGINAL tier first, so an
    # author's mistake is not masked by the CPU swap.
    monkeypatch.setenv(QA_CPU_RESOURCE_ENV, QA_CPU_TIER)
    with pytest.raises(ValueError, match="unknown base_flavor"):
        _derive_resources_from_toml(
            _resource_task_toml("12vcpu+100gib+h100/2", base_flavor="not-a-flavor")
        )


def test_override_drops_the_agent_facing_accelerator_notice(monkeypatch):
    # The prompt notice is derived from required_resources, so forcing CPU at this
    # one point also stops promising the agent a GPU/TPU it will not have.
    monkeypatch.setenv(QA_CPU_RESOURCE_ENV, QA_CPU_TIER)
    resources = _derive_resources_from_toml(_resource_task_toml("12vcpu+100gib+h100/2"))
    assert _accelerator_notice(resources["required_resources"]) == ""
    assert _accelerator_notice("12vcpu+100gib+h100/2") != ""


def test_override_switches_container_runtime_to_firecracker(monkeypatch):
    monkeypatch.setenv(QA_CPU_RESOURCE_ENV, QA_CPU_TIER)
    resources = _derive_resources_from_toml(_resource_task_toml("12vcpu+100gib+h100/2"))
    assert _taiga_container_runtime(resources["required_resources"]) == "firecracker"
