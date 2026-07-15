"""A minimal metadata.json is detected and expanded into the TaskToml / Taiga
payload, at the loader + exporter level (no Docker)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from alignerr_plugin import mlenvs
from alignerr_plugin.exporters.taiga import build_job_payload
from alignerr_plugin.utils import (
    grading_inputs_sha256,
    load_metadata,
    load_task_toml,
    read_prompt,
    task_id,
)

PROMPT = "# Demo task\n\nWrite predictions to `/tmp/output/submission.csv`.\n"
TEST_FILE = (
    "from grading.helpers import load_submission_or_fault\n\n\n"
    "def compute_score():\n"
    "    return 0.0\n"
)


def _write_mlenvs_task(root: Path, *, metadata: dict, name: str = "demo-task_taiga") -> Path:
    task_dir = root / name
    (task_dir / "data" / "public").mkdir(parents=True)
    (task_dir / "data" / "private").mkdir(parents=True)
    (task_dir / "reference_solution").mkdir(parents=True)
    (task_dir / "metadata.json").write_text(json.dumps(metadata, indent=2))
    (task_dir / "prompt.md").write_text(PROMPT)
    (task_dir / "test_file.py").write_text(TEST_FILE)
    (task_dir / "data" / "public" / "train.csv").write_text("x,y\n1,2\n")
    (task_dir / "data" / "private" / "truth.csv").write_text("y\n2\n")
    (task_dir / "reference_solution" / "solution.py").write_text("print('ref')\n")
    return task_dir


def _valid_metadata(**overrides) -> dict:
    base = {
        "ml_task_type": "dataset",
        "required_resources": "12vcpu+100gib+h100/2",
        "domain": "scientific_discovery_computational_science",
        "license": "CC0-1.0",
        "license_source": "https://creativecommons.org/publicdomain/zero/1.0/",
    }
    base.update(overrides)
    return base


# --- detection -------------------------------------------------------------


def test_is_mlenvs_task_true_for_task_type_key(tmp_path: Path):
    task_dir = _write_mlenvs_task(tmp_path, metadata=_valid_metadata())
    assert mlenvs.is_mlenvs_task(task_dir) is True


def test_is_mlenvs_task_false_for_native_envelope(tmp_path: Path):
    task_dir = tmp_path / "native"
    task_dir.mkdir()
    (task_dir / "metadata.json").write_text(
        json.dumps({"benchmark": "taiga_task", "problem_data": {"instance_id": "x"}})
    )
    assert mlenvs.is_mlenvs_task(task_dir) is False


def test_is_mlenvs_task_false_when_missing_or_malformed(tmp_path: Path):
    assert mlenvs.is_mlenvs_task(tmp_path / "nope") is False
    bad = tmp_path / "bad"
    bad.mkdir()
    (bad / "metadata.json").write_text("{not json")
    assert mlenvs.is_mlenvs_task(bad) is False


# --- synthesis: pinned + mapped fields ------------------------------------


def test_synthesized_task_toml_pins_and_maps(tmp_path: Path):
    task_dir = _write_mlenvs_task(tmp_path, metadata=_valid_metadata(**{"docker-base": "default"}))
    toml = load_task_toml(task_dir)

    # Mapped per-task fields.
    assert toml.environment.required_resources == "12vcpu+100gib+h100/2"
    assert toml.environment.base_flavor == "mlenvs-gpu"
    assert toml.environment.hidden_env == ""
    assert toml.difficulty.domain == "scientific_discovery_computational_science"
    assert toml.difficulty.license == "CC0-1.0"

    # Pinned constants.
    assert toml.difficulty.task_type == "ml"
    assert toml.difficulty.reward_type == "continuous_scoring_function"
    assert toml.environment.allow_internet is False
    assert toml.runner.attempts == 3
    assert toml.runner.turn_limit == 1500
    assert toml.runner.timeouts.grading_sec == 10800
    assert toml.runner.required_tools == ["bash", "str_replace_editor", "tmux"]
    assert toml.ground_truth.continuous_score_epsilon == 0.05
    assert toml.task.name == "labelbox/demo-task"
    assert toml.metadata.get("ml_task_type") == "dataset"
    # No per-task outputs: the /tmp/output convention is pinned.
    assert toml.outputs == []


def test_env_hybrid_map_to_hidden_env(tmp_path: Path):
    for task_type in ("env", "hybrid"):
        task_dir = _write_mlenvs_task(
            tmp_path / task_type, metadata=_valid_metadata(ml_task_type=task_type)
        )
        toml = load_task_toml(task_dir)
        assert toml.environment.hidden_env == task_type


def test_docker_base_tpu_maps_to_mlenvs_tpu_flavor(tmp_path: Path):
    task_dir = _write_mlenvs_task(
        tmp_path,
        metadata=_valid_metadata(
            **{"docker-base": "tpu", "required_resources": "13vcpu+32gib+tpuv5e1x1"}
        ),
    )
    toml = load_task_toml(task_dir)
    assert toml.environment.base_flavor == "mlenvs-tpu"


def test_graphics_tier_maps_to_mlenvs_cuda_graphics(tmp_path: Path):
    task_dir = _write_mlenvs_task(
        tmp_path,
        metadata=_valid_metadata(required_resources="12vcpu+100gib+h100/2+graphics"),
    )
    toml = load_task_toml(task_dir)
    assert toml.environment.base_flavor == "mlenvs-cuda-graphics"


def test_tpu_tier_auto_routes_to_mlenvs_tpu_without_docker_base(tmp_path: Path):
    # A TPU tier routes to mlenvs-tpu even with the default docker-base (mirrors
    # the native auto resolver); no need to also set docker-base="tpu".
    task_dir = _write_mlenvs_task(
        tmp_path, metadata=_valid_metadata(required_resources="13vcpu+32gib+tpuv5e1x1")
    )
    toml = load_task_toml(task_dir)
    assert toml.environment.base_flavor == "mlenvs-tpu"


def test_synthesized_envelope_and_task_id(tmp_path: Path):
    task_dir = _write_mlenvs_task(tmp_path, metadata=_valid_metadata())
    meta = load_metadata(task_dir)
    assert meta.benchmark == "taiga_task"
    assert meta.problem_data["instance_id"] == "demo-task"
    assert task_id(task_dir) == "demo-task"
    assert read_prompt(task_dir) == PROMPT


# --- full Taiga payload ---------------------------------------------------


def test_build_job_payload_for_mlenvs_task(tmp_path: Path):
    task_dir = _write_mlenvs_task(tmp_path, metadata=_valid_metadata())
    payload = build_job_payload(task_dir, image_ref="example.com/img:tag")
    problem = payload["problems_metadata"]["problem_set"]["problems"][0]

    assert problem["id"] == "demo-task"
    assert problem["output_directory"] == "/tmp/output"
    assert problem["required_resources"] == "12vcpu+100gib+h100/2"
    assert problem["grading_timeout_seconds"] == 10800  # ml pinned to Taiga max
    assert problem["task_prompt"].startswith("# Demo task")
    assert problem["metadata"]["task_type"] == "ml"
    assert problem["metadata"]["reward_type"] == "continuous_scoring_function"
    assert problem["metadata"]["domain"] == "scientific_discovery_computational_science"
    assert problem["extra_fields"]["task_metadata"]["ml_task_type"] == "dataset"


# --- hf_resources: read-only Hugging Face mounts --------------------------


def test_hf_resources_synthesized_as_preloaded_mounts(tmp_path: Path):
    # String and object entries both become read-only HF-hub-cache mounts.
    task_dir = _write_mlenvs_task(
        tmp_path,
        metadata=_valid_metadata(
            hf_resources=[
                "meta-llama/Llama-3.1-8B",
                {"repo_id": "org/ds", "repo_type": "dataset", "revision": "v2"},
            ]
        ),
    )
    toml = load_task_toml(task_dir)
    mounts = {p.hf_repo: p for p in toml.preloaded_files}
    assert set(mounts) == {"meta-llama/Llama-3.1-8B", "org/ds"}
    model = mounts["meta-llama/Llama-3.1-8B"]
    assert model.mount_path == "/tmp/.cache/huggingface/hub/models--meta-llama--Llama-3.1-8B"
    assert model.read_only is True and model.source == ""
    dataset = mounts["org/ds"]
    assert dataset.mount_path == "/tmp/.cache/huggingface/hub/datasets--org--ds"
    assert dataset.hf_revision == "v2"


def test_no_hf_resources_yields_no_preloaded_files(tmp_path: Path):
    task_dir = _write_mlenvs_task(tmp_path, metadata=_valid_metadata())
    assert load_task_toml(task_dir).preloaded_files == []


def test_hf_resources_mount_path_matches_packer():
    # Synthesis and the deploy-time packer must agree on the mount path.
    import importlib.util

    from alignerr_plugin.mlenvs import HF_HUB_CACHE, hf_resources_to_preloaded

    packer_path = (
        Path(__file__).resolve().parents[2] / "scripts" / "pack_hf_resource.py"
    )
    spec = importlib.util.spec_from_file_location("pack_hf_resource", packer_path)
    packer = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(packer)

    for repo_id, repo_type in [("org/name", "dataset"), ("meta-llama/L", "model")]:
        [preloaded] = hf_resources_to_preloaded(
            [{"repo_id": repo_id, "repo_type": repo_type}]
        )
        folder = packer.repo_folder_name(repo_id, repo_type)
        assert preloaded["mount_path"] == f"{HF_HUB_CACHE}/{folder}"


# --- env_dependencies: server-only hidden-env deps -------------------------


def test_env_dependencies_accepted_for_env_task(tmp_path: Path):
    task_dir = _write_mlenvs_task(
        tmp_path,
        metadata=_valid_metadata(ml_task_type="env", env_dependencies=["myosuite==2.9.0"]),
    )
    # env_dependencies is a build-only selector, not surfaced in the TaskToml.
    toml = load_task_toml(task_dir)
    assert toml.environment.hidden_env == "env"


def test_resolve_task_build_forwards_env_dependencies(tmp_path: Path):
    from lbx_rl_tasks_harness.docker import resolve_task_build

    task_dir = _write_mlenvs_task(
        tmp_path,
        metadata=_valid_metadata(
            ml_task_type="env", env_dependencies=["myosuite==2.9.0", "foo"]
        ),
    )
    _dockerfile, extra = resolve_task_build(task_dir, tmp_path)
    assert "--build-arg" in extra
    assert "ENV_DEPENDENCIES=myosuite==2.9.0 foo" in extra


def test_resolve_task_build_forwards_grading_dependencies(tmp_path: Path):
    from lbx_rl_tasks_harness.docker import resolve_task_build

    # grading_dependencies works for any task type -- here a plain dataset task.
    task_dir = _write_mlenvs_task(
        tmp_path,
        metadata=_valid_metadata(
            ml_task_type="dataset", grading_dependencies=["scikit-learn==1.5.0", "foo"]
        ),
    )
    _dockerfile, extra = resolve_task_build(task_dir, tmp_path)
    assert "GRADING_DEPENDENCIES=scikit-learn==1.5.0 foo" in extra


def test_grading_dependencies_accepted_for_dataset_task(tmp_path: Path):
    # Unlike env_dependencies (env/hybrid only), grading_dependencies is allowed
    # for EVERY task type -- a dataset grader has no other hidden-dep channel.
    task_dir = _write_mlenvs_task(
        tmp_path,
        metadata=_valid_metadata(
            ml_task_type="dataset", grading_dependencies=["scikit-learn==1.5.0"]
        ),
    )
    load_task_toml(task_dir)  # must not raise (build-only selector, not in TaskToml)


# --- validation errors ----------------------------------------------------


@pytest.mark.parametrize(
    "metadata, needle",
    [
        (_valid_metadata(**{"bogus": 1}), "unknown keys"),
        ({"required_resources": "12vcpu+100gib+h100/2"}, "missing required keys"),
        (_valid_metadata(ml_task_type="bandit"), "ml_task_type must be one of"),
        (_valid_metadata(domain="not_a_domain"), "domain"),
        (_valid_metadata(license="GPL-3.0"), "license"),
        (
            _valid_metadata(**{"docker-base": "tpu"}),  # h100 rr, tpu base
            "docker-base='tpu' requires a TPU",
        ),
        (_valid_metadata(hf_resources=["justname"]), "must look like 'org/name'"),
        (
            _valid_metadata(hf_resources=[{"repo_id": "org/name", "repo_type": "space"}]),
            "repo_type must be one of",
        ),
        (
            _valid_metadata(hf_resources=[{"revision": "main"}]),
            "repo_id must be a string",
        ),
        (
            _valid_metadata(
                hf_resources=["org/name"],
                **{"docker-base": "tpu", "required_resources": "13vcpu+32gib+tpuv5e1x1"},
            ),
            "hf_resources is not supported with a TPU base",
        ),
        (
            _valid_metadata(ml_task_type="dataset", env_dependencies=["myosuite"]),
            "env_dependencies is only used by env/hybrid",
        ),
        (
            _valid_metadata(
                ml_task_type="env",
                dependencies=["myosuite[all]"],
                env_dependencies=["myosuite==1.0"],
            ),
            "appear in both",
        ),
        (
            # Same package as a bare VCS URL in one list and a plain name in the
            # other evades the name-based overlap check; the un-analyzable-spec
            # guard catches it (L3a isolation evasion).
            _valid_metadata(
                ml_task_type="env",
                dependencies=["git+https://github.com/foo/myosuite.git"],
                env_dependencies=["myosuite==1.0"],
            ),
            "VCS URL or local path",
        ),
        (
            # Malformed HF repo_id with a slash but an empty component: the old
            # "/"-in-string check passed it; the format regex rejects it (L3b).
            _valid_metadata(hf_resources=["org//name"]),
            "must look like 'org/name'",
        ),
        (
            # A grader-only dep in agent-visible 'dependencies' too defeats the
            # grading_dependencies isolation.
            _valid_metadata(
                ml_task_type="dataset",
                dependencies=["scikit-learn"],
                grading_dependencies=["scikit-learn==1.5"],
            ),
            "appear in both",
        ),
        (
            # A bare VCS URL in grading_dependencies is un-analyzable for overlap.
            _valid_metadata(
                ml_task_type="dataset",
                grading_dependencies=["git+https://github.com/foo/secretlib.git"],
            ),
            "VCS URL or local path",
        ),
        (
            _valid_metadata(ml_task_type="env", env_dependencies="myosuite"),
            "env_dependencies must be a list of non-empty strings",
        ),
    ],
)
def test_invalid_metadata_raises(tmp_path: Path, metadata: dict, needle: str):
    task_dir = _write_mlenvs_task(tmp_path, metadata=metadata)
    with pytest.raises(mlenvs.MlEnvsMetadataError) as exc:
        load_task_toml(task_dir)
    assert needle in str(exc.value)


# --- build-proof scoping --------------------------------------------------


def test_grading_inputs_hash_scopes_to_mlenvs_inputs(tmp_path: Path):
    task_dir = _write_mlenvs_task(tmp_path, metadata=_valid_metadata())
    h0 = grading_inputs_sha256(task_dir)

    # Prose edit must NOT stale the proof.
    (task_dir / "prompt.md").write_text(PROMPT + "\nExtra prose.\n")
    assert grading_inputs_sha256(task_dir) == h0

    # test_file.py edit MUST stale it.
    (task_dir / "test_file.py").write_text(TEST_FILE + "# changed\n")
    h1 = grading_inputs_sha256(task_dir)
    assert h1 != h0

    # private data edit MUST stale it.
    (task_dir / "data" / "private" / "truth.csv").write_text("y\n3\n")
    assert grading_inputs_sha256(task_dir) != h1


def test_reference_solution_dir_selectable_for_mlenvs(tmp_path: Path):
    """ML_Envs reference runs <solution_dir>/solution.py; --solution-dir selects
    the directory (default reference_solution; e.g. a baseline), mirroring
    ML_Envs run_reference --solution-dir. Native tasks are unchanged."""
    import types

    from lbx_rl_tasks_harness.reference_config import solution_script_rel

    ml = _write_mlenvs_task(tmp_path, metadata=_valid_metadata())
    ml_prob = types.SimpleNamespace(
        source_problem_dir=ml,
        reference=types.SimpleNamespace(entrypoint="solve.sh"),
    )
    # native default "solution" (and empty) mean "unset" -> reference_solution
    assert solution_script_rel(ml_prob, solution_dir="solution") == "reference_solution/solution.py"
    assert solution_script_rel(ml_prob, solution_dir="") == "reference_solution/solution.py"
    # an explicit dir runs THAT directory's solution.py
    assert solution_script_rel(ml_prob, solution_dir="baselines/naive") == "baselines/naive/solution.py"
    assert solution_script_rel(ml_prob, solution_dir="reference_solution") == "reference_solution/solution.py"

    native = tmp_path / "native"
    native.mkdir()
    (native / "task.toml").write_text('[task]\nname = "x"\n')
    nat_prob = types.SimpleNamespace(
        source_problem_dir=native,
        reference=types.SimpleNamespace(entrypoint="solve.sh"),
    )
    assert solution_script_rel(nat_prob, solution_dir="solution") == "solution/solve.sh"
    assert solution_script_rel(nat_prob, solution_dir="solution2") == "solution2/solve.sh"
