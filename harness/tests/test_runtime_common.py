from __future__ import annotations

from lbx_rl_tasks_harness.models import HarnessProblem
from lbx_rl_tasks_harness.runtimes.common import (
    accelerator_kind_for_problem,
    localize_accelerator_prompt,
)


def test_localize_gpu_problem_dir_prompt_adds_cpu_fallback_guidance() -> None:
    problem = HarnessProblem(
        id="gpu-demo",
        source_format="problem-dir",
        prompt="demo",
        outputs=[],
        metadata={"environment": {"required_resources": "12vcpu+100gib+h100/2"}},
    )

    prompt = localize_accelerator_prompt(problem, "Write /tmp/output/submission.csv.")

    assert accelerator_kind_for_problem(problem) == "gpu"
    assert "may or may not expose working accelerator hardware" in prompt
    assert "do not debug drivers" in prompt
    assert "CPU-compatible fallback" in prompt


def test_localize_tpu_structured_notice_prompt_adds_cpu_fallback_guidance() -> None:
    problem = HarnessProblem(
        id="tpu-demo",
        source_format="taiga",
        prompt="demo",
        outputs=[],
        taiga_problem={
            "extra_fields": {
                "runtime_notices": [
                    {
                        "kind": "accelerator_availability",
                        "accelerator": "tpu",
                        "text": "A TPU may be available.",
                    }
                ]
            }
        },
    )

    prompt = localize_accelerator_prompt(problem, "Write /tmp/output/policy.py.")

    assert accelerator_kind_for_problem(problem) == "tpu"
    assert "this task may request a TPU" in prompt
    assert "CPU-compatible fallback" in prompt


def test_localize_tpu_resource_prompt_adds_cpu_fallback_guidance() -> None:
    problem = HarnessProblem(
        id="tpu-demo",
        source_format="taiga",
        prompt="demo",
        outputs=[],
        required_resources="13vcpu+32gib+tpuv5e1x1",
    )

    prompt = localize_accelerator_prompt(problem, "Write /tmp/output/policy.py.")

    assert accelerator_kind_for_problem(problem) == "tpu"
    assert "this task may request a TPU" in prompt
    assert "CPU-compatible fallback" in prompt


def test_localize_cpu_prompt_is_unchanged() -> None:
    problem = HarnessProblem(
        id="cpu-demo",
        source_format="problem-dir",
        prompt="demo",
        outputs=[],
        metadata={"environment": {"required_resources": "4vcpu+16gib"}},
    )
    prompt = "Write /tmp/output/result.json."

    assert localize_accelerator_prompt(problem, prompt) == prompt


def test_localize_strips_legacy_prompt_appended_accelerator_notice() -> None:
    problem = HarnessProblem(
        id="gpu-demo",
        source_format="taiga",
        prompt="demo",
        outputs=[],
        required_resources="12vcpu+100gib+h100/2",
    )

    prompt = localize_accelerator_prompt(
        problem,
        "Write /tmp/output/submission.csv.\n\nA GPU may be available.",
    )

    assert "A GPU may be available." not in prompt
    assert "this task may request a GPU" in prompt
