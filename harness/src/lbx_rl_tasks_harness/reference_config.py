from __future__ import annotations

from alignerr_plugin.runtime_notices import accelerator_from_required_resources

from lbx_rl_tasks_harness.models import HarnessProblem, ReferenceSpec

EXECUTION_MODES = frozenset({"auto", "host", "container"})
PROOF_MODES = frozenset({"auto", "artifact", "execute"})
ReferenceExecutionMode = str  # iterate | prove | fast


def _resolve_reference_execution_auto(problem: HarnessProblem) -> str:
    if problem.ground_truth.in_container:
        return "container"

    environment = problem.metadata.get("environment") or {}
    if accelerator_from_required_resources(
        str(environment.get("required_resources") or "")
    ):
        return "container"

    difficulty = problem.metadata.get("difficulty") or {}
    task_type = str(difficulty.get("task_type") or "").strip().lower()
    if task_type == "ml":
        return "container"

    return "host"


def _has_private_truth(problem: HarnessProblem) -> bool:
    """True iff the task commits held-out truth (an answer key).

    The container path bakes this root-only at ``/mcp_server/data`` and runs the
    solution as uid 1000 so it is unreachable; the host path has no such barrier,
    so a task with private truth genuinely depends on the container isolation
    contract.
    """
    if problem.private_dir is not None:
        candidate = problem.private_dir
    elif problem.grader_dir is not None:
        candidate = problem.grader_dir / "data"
    elif problem.source_problem_dir is not None:
        candidate = problem.source_problem_dir / "scorer" / "data"
    else:
        return False
    try:
        return candidate.is_dir() and any(candidate.iterdir())
    except OSError:
        return False


def reference_isolation_requirements(problem: HarnessProblem) -> list[str]:
    """Reasons (if any) the task depends on the container isolation contract.

    The host reference path applies none of the uid-1000 drop, ``--network
    none``, or root-only-truth guarantees, so a task that relies on any of them
    must not run on host. Returns a list of human-readable reasons; empty means
    host execution is safe for this task.
    """
    reasons: list[str] = []
    if problem.ground_truth.in_container:
        reasons.append("[ground_truth].in_container = true")
    environment = problem.metadata.get("environment") or {}
    hidden = str(environment.get("hidden_env") or "").strip().lower()
    if hidden in {"env", "hybrid"}:
        reasons.append(f'[environment].hidden_env = "{hidden}" (needs the env_server sandbox)')
    if _has_private_truth(problem):
        reasons.append("held-out truth committed under scorer/data (root-only at /mcp_server/data on Taiga)")
    return reasons


def resolve_reference_execution(
    problem: HarnessProblem, *, mode: ReferenceExecutionMode = "iterate"
) -> str:
    """Return ``host`` or ``container`` for a reference run.

    ``prove`` always re-resolves via the auto path so ground-truth/build-proof
    ignores an author's ``[reference].execution = "host"`` override. For
    non-prove runs an explicit ``host`` request is honored ONLY when the task
    does not depend on the container isolation contract; otherwise it is
    upgraded to ``container`` (see :func:`reference_isolation_requirements`) so a
    host run can never silently drop the uid-1000 / no-network / root-only-truth
    guarantees a reviewer assumes.
    """
    if mode == "prove":
        # Ground-truth must honor [ground_truth].in_container even when authors
        # set [reference].execution = "host" for local iteration.
        return _resolve_reference_execution_auto(problem)

    spec = problem.reference
    if spec.execution == "container":
        return "container"
    if spec.execution == "host":
        # Refuse to silently disable isolation: a task that relies on it is
        # upgraded to the faithful container path instead of running on host.
        if reference_isolation_requirements(problem):
            return "container"
        return "host"

    return _resolve_reference_execution_auto(problem)


def resolve_reference_proof_mode(problem: HarnessProblem) -> str:
    """Return ``artifact`` or ``execute`` for proof expectations."""
    spec = problem.reference
    if spec.proof_mode in {"artifact", "execute"}:
        return spec.proof_mode

    difficulty = problem.metadata.get("difficulty") or {}
    task_type = str(difficulty.get("task_type") or "").strip().lower()
    if task_type == "ml":
        return "execute"

    environment = problem.metadata.get("environment") or {}
    if accelerator_from_required_resources(
        str(environment.get("required_resources") or "")
    ):
        return "execute"

    return "artifact"


def reference_cache_path(problem: HarnessProblem) -> str:
    return problem.reference.cache_dir.strip() or ".alignerr/reference_cache/output"


def _is_mlenvs_problem(problem: HarnessProblem) -> bool:
    from alignerr_plugin import mlenvs

    src = problem.source_problem_dir
    return src is not None and mlenvs.is_mlenvs_task(src)


def solution_script_rel(problem: HarnessProblem, *, solution_dir: str) -> str:
    base = solution_dir.strip().strip("/")
    if _is_mlenvs_problem(problem):
        # ML_Envs runs <solution_dir>/solution.py with python -- authors never write
        # a solve.sh. --solution-dir selects the directory (default
        # reference_solution; e.g. --solution-dir baselines/naive to run that
        # baseline's solution.py), mirroring ML_Envs run_reference --solution-dir.
        # The native default "solution" means "unset" here -> reference_solution.
        if not base or base == "solution":
            base = "reference_solution"
        return f"{base}/solution.py"
    entrypoint = problem.reference.entrypoint.strip() or "solve.sh"
    return f"{base or 'solution'}/{entrypoint}"


def reference_interpreter(sol_rel: str) -> str:
    """Interpreter for a reference entrypoint: ``python`` for a ``.py`` file
    (ML_Envs solution.py), ``bash`` for a ``.sh`` script (native solve.sh)."""
    return "python" if sol_rel.endswith(".py") else "bash"
