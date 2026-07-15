from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

SourceFormat = Literal["problem-dir", "taiga", "harbor"]
RuntimeName = Literal[
    "claude-code",
    "deepagents",
    "ground-truth",
    "noop",
    "rubric-quality",
    "solution",
]


@dataclass(frozen=True)
class OutputSpec:
    path: str
    required: bool = True
    description: str = ""

    @property
    def relative_path(self) -> Path:
        prefix = "/tmp/output/"
        if self.path == "/tmp/output":
            return Path(".")
        if not self.path.startswith(prefix):
            raise ValueError(f"output path must be under /tmp/output: {self.path}")
        return Path(self.path.removeprefix(prefix))


@dataclass(frozen=True)
class ReferenceSpec:
    execution: str = "auto"
    cache_dir: str = ".alignerr/reference_cache/output"
    entrypoint: str = "solve.sh"
    proof_mode: str = "auto"


@dataclass(frozen=True)
class GroundTruthSpec:
    render_command: str = ""
    render_outputs: list[OutputSpec] = field(default_factory=list)
    score_epsilon: float = 1e-9
    continuous_score_epsilon: float = 0.05
    in_container: bool = False
    zero_anchor_epsilon: float = 0.01


@dataclass(frozen=True)
class HarnessProblem:
    id: str
    source_format: SourceFormat
    prompt: str
    outputs: list[OutputSpec]
    task_dir: Path | None = None
    source_problem_dir: Path | None = None
    grader_dir: Path | None = None
    private_dir: Path | None = None
    image: str | None = None
    required_tools: list[str] = field(default_factory=list)
    required_resources: str | None = None
    ground_truth: GroundTruthSpec = field(default_factory=GroundTruthSpec)
    reference: ReferenceSpec = field(default_factory=ReferenceSpec)
    metadata: dict[str, Any] = field(default_factory=dict)
    taiga_problem: dict[str, Any] | None = None

    def require_grader(self) -> tuple[Path, Path]:
        if self.grader_dir is None:
            raise ValueError(
                f"{self.source_format} input does not include a scorer. "
                "Pass --source-problem-dir so the harness can grade locally."
            )
        private = self.private_dir or self.grader_dir / "data"
        return self.grader_dir, private


@dataclass(frozen=True)
class HarnessResult:
    problem_id: str
    source_format: SourceFormat
    runtime: RuntimeName
    run_dir: Path
    workspace: Path
    reward_path: Path
    details_path: Path
    score: float | None
    review_artifacts: list[dict[str, Any]] = field(default_factory=list)
