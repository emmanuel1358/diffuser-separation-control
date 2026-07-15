"""Shared schemas for the universal task plugin."""

from pathlib import PurePosixPath
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator
from alignerr_plugin.delivery import normalize_delivery_platform
from alignerr_plugin.taiga_resources import (
    TaigaRequiredResources,
    validate_required_resources,
)

from alignerr_plugin.task_metadata import (
    metadata_validation_issues,
    normalize_enum_value,
)

# Accepted values for [environment].hidden_env (the env_server activation gate).
# "" disables it; "env"/"hybrid" turn the hidden-environment RPC server on.
HIDDEN_ENV_MODES: tuple[str, ...] = ("", "env", "hybrid")


class OutputSpec(BaseModel):
    """Expected agent output path."""

    path: str
    required: bool = True
    description: str = ""

    @field_validator("path")
    @classmethod
    def path_must_be_output_scoped(cls, value: str) -> str:
        """Keep authored outputs in Boreal's writable output directory."""
        path = PurePosixPath(value)
        if not path.is_absolute() or path.parts[:3] != ("/", "tmp", "output"):
            raise ValueError("output paths must be absolute and under /tmp/output")
        return value


class TaskSection(BaseModel):
    """Human-facing task metadata."""

    name: str
    description: str = ""
    authors: list[dict[str, str]] = Field(default_factory=list)
    keywords: list[str] = Field(default_factory=list)


class EnvironmentSection(BaseModel):
    """Runtime requirements shared by Harbor and Boreal export."""

    model_config = ConfigDict(extra="forbid")

    required_resources: TaigaRequiredResources
    storage_mb: int = 50000
    allow_internet: bool = True
    # Base image flavor: "auto" (infer from required_resources), or an explicit
    # compatible flavor. See alignerr_plugin.base_image.
    base_flavor: str = "auto"
    # Opt-in hidden-environment RPC server (env_server) for simulation-style
    # tasks where the agent interacts with a black-box env over /tmp/env.sock:
    #   ""       -> off (default; classic static task)
    #   "env"    -> env server on; agent interacts ONLY through the socket
    #   "hybrid" -> env server on; task also ships static data/ files
    # Requires scorer/data/env.py (make_env) and a public data/env_client.py.
    # See docs/HIDDEN_ENV.md.
    hidden_env: str = ""

    @field_validator("required_resources", mode="before")
    @classmethod
    def required_resources_must_be_taiga_enum(cls, value: object) -> str:
        """Force authors to pick a Taiga-supported resource tier verbatim."""
        return validate_required_resources(value)

    @model_validator(mode="after")
    def validate_environment(self) -> "EnvironmentSection":
        """Validate base flavor compatibility and hidden-env mode."""
        from alignerr_plugin.base_image import resolve_base_flavor_for_resource

        self.required_resources = validate_required_resources(self.required_resources)
        self.base_flavor = (
            (self.base_flavor or "auto").strip().lower().replace("_", "-")
        )
        resolve_base_flavor_for_resource(self.base_flavor, self.required_resources)
        self.hidden_env = (self.hidden_env or "").strip().lower()
        if self.hidden_env not in HIDDEN_ENV_MODES:
            raise ValueError(
                f"[environment].hidden_env must be one of {list(HIDDEN_ENV_MODES)} "
                f"(got {self.hidden_env!r}); use 'env' or 'hybrid' to enable the "
                "hidden-environment RPC server, or '' to disable it."
            )
        return self


class AgentSection(BaseModel):
    """Agent runtime configuration."""

    timeout_sec: int | None = 1800
    # Harbor/Prometheus: non-root uid the agent harness runs as (typically ``agent``).
    user: str | None = None


class VerifierSection(BaseModel):
    """Verifier runtime configuration.

    ``env`` is the list of environment variables the verifier (the
    in-image grader) needs at runtime — Harbor's `harbor run` requests
    approval for each. For LLM-judged criteria this should include
    ``ANTHROPIC_API_KEY``.
    """

    timeout_sec: int = 600
    env: list[str] = Field(default_factory=list)
    sandbox: bool = False  # opt-in: grade in a subprocess (e.g. for nvproxy cleanup)
    # Harbor/Prometheus: root verifier runs grading with access to /mcp_server/data.
    user: str | None = None


class GroundTruthSection(BaseModel):
    """Oracle solution and reviewer artifact requirements.

    ``solution/solve.sh`` is the executable ground-truth submission. The render
    command runs after that solution has produced its normal outputs and must
    create the declared reviewer video artifacts.
    """

    render_command: str = ""
    render_outputs: list[OutputSpec] = Field(default_factory=list)
    score_epsilon: float = 1e-9
    continuous_score_epsilon: float = 0.05
    # Opt-in: run the oracle solve + grade + render INSIDE the built task image
    # instead of on the host. Required for tasks whose grader/renderer depend on
    # engines that live only in the base image (OpenFOAM, SU2, Meep, OpenROAD).
    in_container: bool = False
    # Highest score a trivial / no-op / prompt-example submission is allowed to
    # earn. The validator grades an empty submission (and, when extractable, the
    # prompt's example) through the real scorer and fails the task if either
    # lands above this ceiling -- the signal that "do nothing" or "copy the
    # example" out-scores genuine work (a dominant reward-hacking failure mode).
    max_trivial_score: float = 0.5
    # Highest score an empty/no-op submission may earn on a
    # continuous_scoring_function task. Continuous graders must anchor
    # "no attempt" to 0; this tolerance only absorbs float/curve noise.
    zero_anchor_epsilon: float = 0.01


class ReferenceSection(BaseModel):
    """Local reference-solution runner configuration (author iteration loop).

    Mirrors ML_Envs ``run_reference`` semantics: in-container execution,
    persistent cache, and optional train/grade separation.
    """

    execution: Literal["auto", "host", "container"] = "auto"
    cache_dir: str = ".alignerr/reference_cache/output"
    entrypoint: str = "solve.sh"
    proof_mode: Literal["auto", "artifact", "execute"] = "auto"


# Hour-scale per-stage runner timeouts (seconds) that **ML tasks** are pinned to.
# Single source of truth for the ML "fair-chance" timeouts: the Taiga exporter
# force-pins ``task_type == "ml"`` to these, and ML_Envs mode reuses them via
# ``mlenvs.PINNED_TIMEOUTS``, so the numbers can never drift. Minutes-scale caps
# make Boreal/Taiga finish before an agent (or a human building a reference
# solution) can plausibly solve an ML task. Only ML tasks are pinned; other task
# types keep the (author-overridable) RunnerTimeouts defaults below.
ML_SETUP_TIMEOUT_SEC = 7200  # 2h — environment/setup phase
ML_GRADING_TIMEOUT_SEC = 10800  # 3h — Taiga's maximum grading timeout
ML_TOOL_TIMEOUT_SEC = 21600  # 6h — per tool-call budget
ML_MAX_EPISODE_SEC = 21600  # 6h — job-level episode wall-clock


class RunnerTimeouts(BaseModel):
    """Per-stage Boreal/Taiga runner timeouts (seconds).

    These are **Taiga/Boreal-only** — Harbor ignores them (it exposes its own
    resource controls via ``[environment]``). Each field drives a distinct field
    in the exported Taiga/Boreal payload (see ``exporters/taiga.py``):

    ==================  ==========================  ==================================
    RunnerTimeouts      Taiga payload field         Scope
    ==================  ==========================  ==================================
    ``setup_sec``       ``setup_timeout_seconds``   per-problem entry
    ``grading_sec``     ``grading_timeout_seconds`` per-problem entry **and** the
                                                    in-image rubric subprocess
                                                    (``extra_fields``)
    ``tool_sec``        ``tool_timeout_seconds``    per-problem entry
    ``max_episode_sec`` ``max_timeout_seconds``     job-level (``None`` = unlimited)
    ==================  ==========================  ==================================

    The defaults below apply to non-ML task types and stay author-overridable.
    ``task_type == "ml"`` ignores these entirely: the Taiga exporter force-pins
    ml tasks to the hour-scale ``ML_*_TIMEOUT_SEC`` values above regardless of
    author input, because minutes-scale caps silently fail real ML runs.
    """

    setup_sec: int = 600
    grading_sec: int = 600
    tool_sec: int = 120
    max_episode_sec: int | None = 3600  # None = unlimited


class RunnerConfig(BaseModel):
    """Boreal job + episode-level runtime configuration.

    These knobs become per-problem and job-level fields in the Boreal payload
    built by the exporter. They have no effect on Harbor (which exposes its
    own resource controls via `[environment]`).
    """

    model_config = ConfigDict(extra="allow")

    attempts: int = 3
    turn_limit: int | None = 1000  # None = unlimited
    max_ctx: int = 1_000_000
    context_mode: Literal["none", "autocompact", "memory"] = "none"
    priority: Literal["low", "high"] = "high"
    iteration_order: Literal["problems_first", "attempts_first"] = "problems_first"
    checkpoint_ttl: str | None = None  # e.g. "30d"
    serialize_restore_test_interval: int | None = None
    api_model_name: str = "claude-fable-5"
    required_tools: list[str] = Field(
        default_factory=lambda: ["bash", "str_replace_editor", "tmux"]
    )
    # In-container Anthropic API access is a reward-hack surface; opt in only
    # for tasks that explicitly require agent-side model calls.
    enable_anthropic_api: bool = False
    container_runtime: Literal["firecracker", "docker"] = "firecracker"

    timeouts: RunnerTimeouts = Field(default_factory=RunnerTimeouts)


class Difficulty(BaseModel):
    """Task classification metadata.

    These fields drive validation, routing, dashboards, and Taiga metadata.
    None of these affect grading directly.
    """

    # Required enum: ml / mujoco / cfd / structures.
    task_type: str = ""
    # Required enum scoped by task_type, e.g. model_environment_construction
    # for mujoco or scientific_discovery_computational_science for ml.
    domain: str = ""
    # Required enum: continuous_scoring_function / multi_deterministic_rubrics.
    reward_type: str = ""
    # Permissive dataset license (SPDX id). Required when task_type == "ml";
    # optional otherwise. See alignerr_plugin.task_metadata.LICENSES for the
    # allowed set. Preserved verbatim (not normalized) for downstream display.
    license: str = ""
    # Provenance pointer the licensing reviewer verifies. Required for ml tasks:
    # the upstream http(s) URL where the license was confirmed, or -- when
    # license == "self_generated" -- a short reason the task generates its own data.
    license_source: str = ""
    is_impossible: bool = False  # task is intentionally unachievable (red-teaming)

    @model_validator(mode="after")
    def validate_metadata_enums(self) -> "Difficulty":
        self.task_type = normalize_enum_value(self.task_type)
        self.domain = normalize_enum_value(self.domain)
        self.reward_type = normalize_enum_value(self.reward_type)
        issues = metadata_validation_issues(
            task_type=self.task_type,
            domain=self.domain,
            reward_type=self.reward_type,
            license_id=self.license,
            license_source=self.license_source,
        )
        if issues:
            raise ValueError("; ".join(issues))
        return self


class PreloadedFile(BaseModel):
    """A read-only artifact mounted into the container at deploy time instead of
    baked into the image.

    Declare exactly one source:

    * ``source`` -- a directory tree under the task dir, packed into a
      content-addressed squashfs and uploaded once (shared/deduped across tasks);
    * ``hf_repo`` -- a Hugging Face repo (optionally pinned to ``hf_revision``),
      fetched and packed so ``from_pretrained`` resolves it offline.

    The deploy pipeline (``scripts/sync_mount.sh``) packs/uploads and stamps the
    concrete ``remote_path`` into ``.alignerr/preloaded_files.json``; the
    exporter emits that manifest. This is how large datasets and model weights
    avoid image bloat (one shared object, not one copy per task image).
    """

    source: str = ""
    hf_repo: str = ""
    hf_revision: str = ""
    mount_path: str
    read_only: bool = True

    @model_validator(mode="after")
    def _validate(self) -> "PreloadedFile":
        if bool(self.source) == bool(self.hf_repo):
            raise ValueError(
                "[[preloaded_files]] requires exactly one of `source` or `hf_repo`"
            )
        if not PurePosixPath(self.mount_path).is_absolute():
            raise ValueError("[[preloaded_files]].mount_path must be an absolute path")
        return self


class Hint(BaseModel):
    """A single Taiga hint shipped with the task."""

    text: str
    enabled: bool = True
    spoiler_level: float = Field(default=0.5, ge=0.0, le=1.0)


class DeliverySection(BaseModel):
    """Post-CI delivery destination.

    Mothership reads this to select delivery. Omit the section to keep the
    historical default of Taiga delivery. Prometheus CFD/structures retain
    ``platform = "prometheus"`` while mothership also derives a parallel Taiga
    mirror for those numerical-solver task types.
    """

    platform: Literal["taiga", "prometheus"] = "taiga"
    eval: bool = False

    @field_validator("platform", mode="before")
    @classmethod
    def platform_must_be_canonical(cls, value: object) -> str:
        return normalize_delivery_platform(value)


class TaskToml(BaseModel):
    """Parsed task.toml.

    Sections beyond `[task]` mostly have sensible defaults. `[difficulty]` is
    required in practice because task_type, domain, and reward_type are enum
    metadata used by validation and dashboards.
    """

    schema_version: str = "1.1"
    task: TaskSection
    environment: EnvironmentSection = Field(default_factory=EnvironmentSection)
    agent: AgentSection = Field(default_factory=AgentSection)
    verifier: VerifierSection = Field(default_factory=VerifierSection)
    ground_truth: GroundTruthSection = Field(default_factory=GroundTruthSection)
    reference: ReferenceSection = Field(default_factory=ReferenceSection)
    outputs: list[OutputSpec] = Field(default_factory=list)
    runner: RunnerConfig = Field(default_factory=RunnerConfig)
    difficulty: Difficulty = Field(default_factory=Difficulty)
    hint: list[Hint] = Field(default_factory=list)
    preloaded_files: list[PreloadedFile] = Field(default_factory=list)
    delivery: DeliverySection = Field(default_factory=DeliverySection)
    metadata: dict[str, Any] = Field(default_factory=dict)


class ProblemMetadata(BaseModel):
    """Alignerr metadata.json."""

    benchmark: str = "taiga_task"
    problem_data: dict[str, Any]


class StageResult(BaseModel):
    """Result for one validation stage."""

    passed: bool
    issues: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    duration_ms: int = 0


class ValidationResult(BaseModel):
    """Full validation result emitted by the task validator."""

    problem_id: str
    benchmark: str
    status: Literal["valid", "invalid"]
    stages: dict[str, StageResult]
    metadata: dict[str, Any] = Field(default_factory=dict)
