"""Shared grading library for Alignerr RL tasks.

Authors `import grading` (or any submodule) inside their `compute_score.py`
to define deterministic rubrics or use helper predicates. The same
library is used by:

  * the Boreal rubric MCP server (`runtime/rubric/server.py`) when it
    invokes `compute_score()` after an episode finishes,
  * the Harbor `run_grader.py` script that emits `/logs/verifier/reward.json`
    for external customers running the exported task with `harbor run` or
    plain `docker run`.

Public surface:

  - `Grade`                          dataclass + `to_dict()` (Boreal/Harbor schema)
  - `AgentFault`                     mark agent-controlled malformed outputs as score 0
  - `RubricBuilder`                  decorator API for weighted deterministic criteria + penalties
  - `LLMJudge`                       legacy/internal judge helper; do not use in submitted task rubrics
  - `helpers`                        deterministic predicates (file_exists, regex_search, ...)
  - `PolicyWorker`                   out-of-process runner for submitted policy.py files
  - `normalize_compute_score_return` accepts float / dict / Grade, returns a Grade

The three accepted return shapes from `compute_score()` are:

  1. `float` in `[0, 1]`                    -> ML_Envs-style continuous scoring
  2. `dict {score, subscores, weights, metadata}` -> headline `score` is authoritative
  3. `RubricBuilder.grade().to_dict()`      -> rich rubric with per-criterion detail

See `docs/GRADING.md` in this repo for the full author guide.
"""

from grading.env_loading import load_env_module
from grading.faults import AgentFault
from grading.grade import PASS_THRESHOLD, Grade
from grading.judge import LLMJudge, LLMJudgeError
from grading.kfold import score_kfold_cv
from grading.normalize import normalize_compute_score_return
from grading.policy_runner import (
    PolicyHandle,
    PolicyTimeoutError,
    PolicyWorker,
    PolicyWorkerError,
    load_submitted_policy,
)
from grading.rubric_builder import RubricBuilder

# Helpers and the calibration toolkit are exported as submodules rather than
# flattened at top level. Keeps `from grading import RubricBuilder, helpers`
# clean and avoids name collisions with whatever the author has in scope.
from grading import calibration  # noqa: F401  re-exported
from grading import helpers  # noqa: F401  re-exported
from grading import policy_eval  # noqa: F401  re-exported

__all__ = [
    "Grade",
    "AgentFault",
    "LLMJudge",
    "LLMJudgeError",
    "PASS_THRESHOLD",
    "PolicyHandle",
    "PolicyTimeoutError",
    "PolicyWorker",
    "PolicyWorkerError",
    "RubricBuilder",
    "calibration",
    "helpers",
    "load_env_module",
    "load_submitted_policy",
    "normalize_compute_score_return",
    "policy_eval",
    "score_kfold_cv",
]
