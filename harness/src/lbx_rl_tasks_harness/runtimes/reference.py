from __future__ import annotations

import json
import os
import platform
import shlex
import shutil
import subprocess
import sys
import threading
import time
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal

from lbx_rl_tasks_harness.env import harness_subprocess_env
from lbx_rl_tasks_harness.grading import grade_workspace
from lbx_rl_tasks_harness.models import HarnessProblem
from lbx_rl_tasks_harness.reference_config import (
    reference_cache_path,
    reference_isolation_requirements,
    resolve_reference_execution,
    resolve_reference_proof_mode,
    reference_interpreter,
    solution_script_rel,
)
from lbx_rl_tasks_harness.runtimes.solution import (
    TAIGA_PLATFORM,
    _copy_outputs,
    _isolated_tmp_output,
    run_solution,
)

ReferenceMode = Literal["iterate", "prove", "fast"]

_CONTAINER_VENV_PATH = "export PATH=/opt/lbx-runtime/.venv/bin:$PATH"

# Taiga runs the agent/solution as an unprivileged uid-1000 user with the
# held-out truth (``/mcp_server/data``) baked root-only, and with no network at
# runtime. The reference runner mirrors BOTH constraints so a reference that
# secretly reads the answer key or downloads at runtime fails LOCALLY instead of
# silently passing here and breaking on Taiga. This is the dynamic complement to
# the static M7 answer-key leak scan.
#
# The uid-1000 account is named ``agent`` on the native bases (created by
# install-common.sh) but ``model`` on the mlenvs bases (which never run
# install-common.sh). Resolve the name FROM THE IMAGE at run time by uid 1000
# rather than hardcoding, so the same command works on both -- a hardcoded
# ``agent`` yields ``su: user agent does not exist`` on every mlenvs task.
_DROP_USER_BY_UID = '"$(getent passwd 1000 | cut -d: -f1)"'


def _emit_reference_notice(lines: list[str]) -> None:
    """Print a banner to the run output (stdout) AND stderr.

    Surfaced on both so it shows in the captured run log and is hard to miss in
    an interactive terminal.
    """
    banner = "\n".join(lines)
    for stream in (sys.stdout, sys.stderr):
        stream.write(banner + "\n")
        stream.flush()


def _warn_host_isolation_bypass(problem: HarnessProblem) -> None:
    """Loudly flag that a host reference run drops every Taiga guarantee.

    The host path runs solve + grade as the invoking user, with full host
    network, and with the grader in-process (no root-only /mcp_server/data), so
    its result is NOT Taiga-faithful and must never be treated as
    proof-authoritative.
    """
    _emit_reference_notice(
        [
            "[reference][WARNING] running on the HOST path - isolation is NOT applied:",
            "  - no uid-1000 privilege drop (solve.sh runs as the invoking user)",
            "  - no network isolation (full host network, not --network none)",
            "  - no held-out-truth isolation (grader runs in-process, not root-only " "/mcp_server/data)",
            "  This result is NOT Taiga-faithful and is NOT proof-authoritative; use "
            "the container path (or `--runtime ground-truth`) before trusting it.",
        ]
    )


def _notify_host_upgraded_to_container(problem: HarnessProblem) -> None:
    """Explain why an explicit execution="host" request became a container run."""
    reasons = reference_isolation_requirements(problem)
    _emit_reference_notice(
        [
            '[reference] [reference].execution = "host" was requested but this task '
            "depends on the container isolation contract, so it was upgraded to the "
            "faithful container path:",
            *[f"  - {reason}" for reason in reasons],
            '  Set [reference].execution = "container" (or remove it) to silence this.',
        ]
    )


def _run_streaming(
    cmd: list[str],
    *,
    cwd: Path | None = None,
    env: dict[str, str] | None = None,
    timeout: int,
    label: str,
) -> tuple[int, str]:
    """Run ``cmd`` streaming combined stdout/stderr live while capturing it.

    ML_Envs streams the solve + grader output to the operator during long runs
    (``docker_exec_streaming``); ``subprocess.run(capture_output=True)`` instead
    leaves the operator staring at a blank terminal for a multi-hour training
    run. This tees each line to ``sys.stdout`` as it arrives and also returns the
    full captured text for the transcript.
    """
    header = f"[{label}]"
    sys.stdout.write(header + "\n")
    sys.stdout.flush()
    captured: list[str] = [header + "\n"]
    proc = subprocess.Popen(
        cmd,
        cwd=str(cwd) if cwd is not None else None,
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        bufsize=1,
    )

    def _pump() -> None:
        assert proc.stdout is not None
        for line in proc.stdout:
            sys.stdout.write(line)
            sys.stdout.flush()
            captured.append(line)

    pump = threading.Thread(target=_pump, daemon=True)
    pump.start()
    try:
        proc.wait(timeout=timeout)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait()
        pump.join(timeout=5)
        if proc.stdout is not None:
            proc.stdout.close()
        raise
    pump.join(timeout=5)
    if proc.stdout is not None:
        proc.stdout.close()
    return proc.returncode, "".join(captured)


def collect_host_info() -> dict:
    """Record the hardware a reference score was measured on.

    Mirrors ML_Envs ``collect_host_info`` so a reviewer can tell whether a score
    was produced on comparable hardware. GPU info is best-effort via
    ``nvidia-smi``; absence is recorded as an empty list rather than failing.
    """
    info: dict = {
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "python_version": platform.python_version(),
        "cpu_count": os.cpu_count(),
    }
    try:
        page_size = os.sysconf("SC_PAGE_SIZE")
        phys_pages = os.sysconf("SC_PHYS_PAGES")
        info["mem_total_bytes"] = page_size * phys_pages
    except (ValueError, OSError, AttributeError):
        info["mem_total_bytes"] = None

    gpus: list[dict] = []
    if shutil.which("nvidia-smi") is not None:
        try:
            proc = subprocess.run(
                [
                    "nvidia-smi",
                    "--query-gpu=name,memory.total",
                    "--format=csv,noheader,nounits",
                ],
                text=True,
                capture_output=True,
                timeout=15,
                check=False,
            )
            if proc.returncode == 0:
                for line in proc.stdout.strip().splitlines():
                    name, _, mem = line.partition(",")
                    name = name.strip()
                    mem = mem.strip()
                    if name:
                        gpus.append(
                            {
                                "name": name,
                                "memory_total_mib": int(mem) if mem.isdigit() else None,
                            }
                        )
        except (OSError, subprocess.SubprocessError):
            pass
    info["gpus"] = gpus
    return info


def _hidden_env_mode(problem: HarnessProblem) -> str:
    environment = problem.metadata.get("environment") or {}
    return str(environment.get("hidden_env") or "").strip().lower()


def _container_script_prefix(problem: HarnessProblem, *, skip_solve: bool) -> list[str]:
    lines = [
        "set -e",
        _CONTAINER_VENV_PATH,
        "export LBT_OUTPUT_DIR=/tmp/output",
        "mkdir -p /tmp/output /tmp/verifier",
    ]
    if not skip_solve and _hidden_env_mode(problem) in {"env", "hybrid"}:
        lines.extend(
            [
                "python -m env_server &",
                "ENV_PID=$!",
                "for _i in $(seq 1 120); do " "[ -S /tmp/env.sock ] && break; sleep 0.25; done",
                "test -S /tmp/env.sock",
            ]
        )
    return lines


def _container_script_stop_env_server(problem: HarnessProblem, *, skip_solve: bool) -> list[str]:
    if skip_solve or _hidden_env_mode(problem) not in {"env", "hybrid"}:
        return []
    return [
        'if [ -n "${ENV_PID:-}" ]; then '
        'kill "$ENV_PID" 2>/dev/null || true; '
        'wait "$ENV_PID" 2>/dev/null || true; fi',
        "rm -f /tmp/env.sock",
    ]


def _as_model_user(command: str) -> str:
    """Run a shell command as the Taiga agent user inside a root-started container.

    Resolves the uid-1000 account name from the image at run time (``agent`` on
    native bases, ``model`` on mlenvs bases) so the drop works on both flavors.
    """
    model_command = "export PATH=/opt/lbx-runtime/.venv/bin:$PATH; " "export LBT_OUTPUT_DIR=/tmp/output; " f"{command}"
    return f"su -s /bin/bash {_DROP_USER_BY_UID} -c {shlex.quote(model_command)}"


def _cache_has_required_outputs(problem: HarnessProblem, cache_dir: Path) -> bool:
    if not problem.outputs:
        return False
    has_required = False
    for spec in problem.outputs:
        if not spec.required:
            continue
        has_required = True
        if not (cache_dir / spec.relative_path).exists():
            return False
    return has_required


def _effective_skip_solve(problem: HarnessProblem, options: ReferenceRunOptions, cache_dir: Path) -> bool:
    if options.skip_solve:
        return True
    if options.mode == "prove":
        return False
    if resolve_reference_proof_mode(problem) == "artifact":
        return _cache_has_required_outputs(problem, cache_dir)
    return False


@dataclass(frozen=True)
class ReferenceRunOptions:
    mode: ReferenceMode = "iterate"
    skip_solve: bool = False
    no_grade: bool = False
    clean_cache: bool = False
    solution_dir: str = "solution"
    render: bool = False
    write_manifest: bool = True


@dataclass(frozen=True)
class ReferenceRunResult:
    score: float | None
    workspace: Path
    verifier_dir: Path
    grade_payload: dict
    manifest_path: Path | None
    execution: str
    solution_returncode: int | None
    grader_returncode: int | None


def _agent_timeout(problem: HarnessProblem) -> int:
    return int(problem.metadata.get("agent", {}).get("timeout_sec") or 3600)


def _verifier_timeout(problem: HarnessProblem) -> int:
    return int(problem.metadata.get("verifier", {}).get("timeout_sec") or 600)


def _cache_dir(problem: HarnessProblem, *, clean: bool) -> Path:
    if problem.source_problem_dir is None:
        raise ValueError("reference run requires a source problem directory")
    rel = reference_cache_path(problem)
    path = (problem.source_problem_dir / rel).resolve()
    if clean and path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def _manifest_dir(problem: HarnessProblem) -> Path:
    assert problem.source_problem_dir is not None
    return problem.source_problem_dir / ".alignerr" / "reference_run" / "latest"


def _write_manifest(
    problem: HarnessProblem,
    *,
    options: ReferenceRunOptions,
    execution: str,
    score: float | None,
    solution_returncode: int | None,
    grader_returncode: int | None,
    run_dir: Path | None,
    elapsed_s: float,
) -> Path | None:
    if not options.write_manifest:
        return None
    assert problem.source_problem_dir is not None
    manifest_dir = _manifest_dir(problem)
    manifest_dir.mkdir(parents=True, exist_ok=True)
    # HAR-3: record the isolation posture explicitly so a reviewer (or any
    # downstream consumer) can tell a host run apart from a Taiga-faithful
    # container run. A host run drops the uid-1000 / no-network / root-only-truth
    # guarantees and is never proof-authoritative.
    host_execution = execution == "host"
    manifest = {
        "problem_id": problem.id,
        "mode": options.mode,
        "execution": execution,
        "requested_execution": problem.reference.execution,
        "isolation": "none" if host_execution else "container",
        "taiga_faithful": not host_execution,
        "proof_authoritative": not host_execution,
        "solution_dir": options.solution_dir,
        "skip_solve": options.skip_solve,
        "no_grade": options.no_grade,
        "clean_cache": options.clean_cache,
        "render": options.render,
        "started_at": datetime.now(UTC).isoformat(),
        "elapsed_s": elapsed_s,
        "score": score,
        "solution_returncode": solution_returncode,
        "grader_returncode": grader_returncode,
        "run_dir": str(run_dir) if run_dir is not None else None,
        "cache_dir": reference_cache_path(problem),
        "proof_mode": resolve_reference_proof_mode(problem),
        "host": collect_host_info(),
    }
    path = manifest_dir / "manifest.json"
    path.write_text(json.dumps(manifest, indent=2) + "\n")
    score_path = manifest_dir / "score.json"
    score_path.write_text(json.dumps({"score": score}, indent=2) + "\n")
    return path


def _sync_output_tree(source: Path, dest: Path) -> None:
    if not source.exists():
        return
    dest.mkdir(parents=True, exist_ok=True)
    for item in source.iterdir():
        target = dest / item.name
        if item.is_dir():
            if target.exists():
                shutil.rmtree(target)
            shutil.copytree(item, target)
        else:
            shutil.copy2(item, target)


def _run_host_reference(
    problem: HarnessProblem,
    workspace: Path,
    verifier_dir: Path,
    transcript_path: Path,
    *,
    options: ReferenceRunOptions,
    cache_dir: Path,
) -> ReferenceRunResult:
    if problem.source_problem_dir is None:
        raise ValueError("reference run requires a source problem directory")

    skip_solve = _effective_skip_solve(problem, options, cache_dir)
    sol_rel = solution_script_rel(problem, solution_dir=options.solution_dir)
    solve_path = problem.source_problem_dir / sol_rel
    if not skip_solve and not solve_path.exists():
        raise FileNotFoundError(f"reference entrypoint not found: {solve_path}")

    solution_rc: int | None = None
    sol_output = ""

    if not skip_solve:
        env = harness_subprocess_env({"LBT_OUTPUT_DIR": "/tmp/output"})
        with _isolated_tmp_output() as tmp_output:
            solution_rc, sol_output = _run_streaming(
                [reference_interpreter(sol_rel), str(solve_path)],
                cwd=problem.source_problem_dir,
                env=env,
                timeout=_agent_timeout(problem),
                label="reference host solve",
            )
            if solution_rc != 0:
                transcript_path.write_text(sol_output)
                raise RuntimeError(f"reference solution exited with status {solution_rc}")
            _sync_output_tree(tmp_output, cache_dir)
            _copy_outputs(tmp_output, workspace, problem)
    else:
        _copy_outputs(cache_dir, workspace, problem)

    score: float | None = None
    grader_rc: int | None = None
    grade_payload: dict = {}

    if not options.no_grade:
        score = grade_workspace(problem, workspace, verifier_dir, transcript_path)
        grader_rc = 0
        details_path = verifier_dir / "reward-details.json"
        reward_path = verifier_dir / "reward.json"
        if details_path.exists():
            grade_payload = json.loads(details_path.read_text())
        elif reward_path.exists():
            grade_payload = json.loads(reward_path.read_text())
        else:
            grade_payload = {"score": score}
    else:
        transcript_path.write_text("\n".join([sol_output, "[reference] grading skipped (--no-grade)"]))

    return ReferenceRunResult(
        score=score,
        workspace=workspace,
        verifier_dir=verifier_dir,
        grade_payload=grade_payload,
        manifest_path=None,
        execution="host",
        solution_returncode=solution_rc,
        grader_returncode=grader_rc,
    )


def _container_solve_script(problem: HarnessProblem, *, sol_rel: str, render: bool) -> str:
    """Build the SOLVE-phase script: bring up env, run the reference, render."""
    lines = _container_script_prefix(problem, skip_solve=False)
    lines.append(
        _as_model_user(
            f"cd /host_task && {reference_interpreter(sol_rel)} {shlex.quote(sol_rel)}"
        )
    )
    lines.extend(_container_script_stop_env_server(problem, skip_solve=False))
    render_cmd = problem.ground_truth.render_command.strip()
    if render and render_cmd:
        lines.append(_as_model_user(render_cmd))
    return "\n".join(lines)


def _container_grade_script() -> str:
    """Build the GRADE-phase script: run the grader as root, export rewards."""
    return "\n".join(
        [
            "set -e",
            _CONTAINER_VENV_PATH,
            "mkdir -p /tmp/verifier",
            "rc=0",
            # _CONTAINER_VENV_PATH prepends the native venv to PATH, so `python`
            # resolves to the venv interpreter on native bases and to the system
            # interpreter (/usr/local) on mlenvs bases, which have no
            # /opt/lbx-runtime/.venv. Mirrors the solve phase + reference_interpreter.
            "python /runtime/run_grader.py "
            "--workspace /tmp/output --grader-dir /mcp_server/grader "
            "--private-dir /mcp_server/data --output-dir /tmp/verifier "
            "--transcript /tmp/verifier/transcript.txt || rc=$?",
            # Copy diagnostics out even when the grader fails: run_grader writes
            # reward-details.json (error_type + traceback) before returning
            # non-zero, but `set -e` would otherwise skip this copy and discard
            # it, leaving only the useless stderr tail on a grade failure.
            "cp -a /tmp/verifier/. /host_out/verifier/ 2>/dev/null || true",
            "exit $rc",
        ]
    )


def _grade_failure_detail(verifier_dir: Path) -> str:
    """Extract the grader's own error report from reward-details.json.

    run_grader/worker write ``error_type`` + ``error_message`` (and a full
    ``metadata.traceback``) into reward-details.json even when the grade fails,
    so surface it instead of leaving only the container stderr tail (which for
    mlenvs bases is just setup warnings).
    """
    details_path = verifier_dir / "reward-details.json"
    if not details_path.exists():
        return ""
    try:
        details = json.loads(details_path.read_text())
    except (OSError, ValueError):
        return ""
    parts: list[str] = []
    for log in (details.get("criterion_logs") or {}).values():
        if not isinstance(log, dict):
            continue
        etype = log.get("error_type")
        emsg = log.get("error_message") or log.get("reasoning")
        if etype or emsg:
            parts.append(f"{etype or 'error'}: {emsg or ''}".strip())
    tb = (details.get("metadata") or {}).get("traceback")
    detail = ""
    if parts:
        detail += "\ngrader error: " + "; ".join(parts)
    if isinstance(tb, str) and tb.strip():
        detail += "\ngrader traceback:\n" + tb.strip()[-2000:]
    return detail


def _docker_solve_command(
    image_tag: str,
    src: Path,
    cache_dir: Path,
    script: str,
    *,
    network_isolated: bool = True,
) -> list[str]:
    """SOLVE phase: start as root for hidden-env setup, then drop solve to uid 1000.

    The container still runs with no network and no host scorer mount. The shell
    script starts env_server as root when needed, then invokes solve/render via
    ``su`` as the unprivileged agent user so the answer key remains unreachable.
    """
    args = [
        "docker",
        "run",
        "--rm",
        "--platform",
        TAIGA_PLATFORM,
    ]
    if network_isolated:
        args += ["--network", "none"]
    args += [
        "-v",
        f"{src}:/host_task:ro",
        "-v",
        f"{cache_dir}:/tmp/output",
        "-e",
        "LBT_OUTPUT_DIR=/tmp/output",
        image_tag,
        "bash",
        "-lc",
        script,
    ]
    return args


def _docker_grade_command(
    image_tag: str,
    cache_dir: Path,
    scorer_dir: Path,
    container_out: Path,
    script: str,
) -> list[str]:
    """GRADE phase: run as root (so the grader can read the root-only held-out
    truth) and bind-mount the HOST ``scorer/`` over ``/mcp_server/grader`` so an
    edit to ``compute_score.py`` changes the next score with no image rebuild.

    Held-out truth still comes from the baked, root-only ``/mcp_server/data`` via
    ``--private-dir``; the host mount only supplies grader CODE. It is safe to
    expose here because this phase runs as root AFTER the unprivileged solve.

    ML_Envs tasks have no host ``scorer/``: the grader is baked into the image
    (``test_file.py`` -> ``/mcp_server/grader/compute_score.py``). Bind-mounting a
    non-existent host path would make Docker create an EMPTY dir and mount it over
    ``/mcp_server/grader``, shadowing the baked grader and failing with
    "no grader source (compute_score.py) found". Only mount when the host
    ``scorer/`` actually exists; otherwise use the baked grader.

    The grade runs with ``--network none`` (like the solve phase and Taiga). A
    grader or submitted policy that reaches the internet at score time -- e.g.
    ``from_pretrained("org/model")`` -- would otherwise succeed locally but fail
    on the network-isolated Taiga sandbox, giving an unrealistic reference
    calibration; isolating the grade keeps local scores representative.
    """
    cmd = [
        "docker",
        "run",
        "--rm",
        "--platform",
        TAIGA_PLATFORM,
        "--network",
        "none",
        "-v",
        f"{cache_dir}:/tmp/output",
    ]
    if scorer_dir.is_dir():
        cmd += ["-v", f"{scorer_dir}:/mcp_server/grader:ro"]
    cmd += [
        "-v",
        f"{container_out}:/host_out",
        "-e",
        "LBT_OUTPUT_DIR=/tmp/output",
        image_tag,
        "bash",
        "-lc",
        script,
    ]
    return cmd


def _run_container_reference(
    problem: HarnessProblem,
    workspace: Path,
    verifier_dir: Path,
    transcript_path: Path,
    *,
    options: ReferenceRunOptions,
    cache_dir: Path,
) -> ReferenceRunResult:
    from lbx_rl_tasks_harness.docker import build_task_image

    if problem.source_problem_dir is None:
        raise ValueError("reference run requires a source problem directory")

    src = problem.source_problem_dir.resolve()
    scorer_dir = src / "scorer"
    skip_solve = _effective_skip_solve(problem, options, cache_dir)
    sol_rel = solution_script_rel(problem, solution_dir=options.solution_dir)
    solve_path = src / sol_rel
    if not skip_solve and not solve_path.exists():
        raise FileNotFoundError(f"reference entrypoint not found: {solve_path}")

    image_tag = build_task_image(problem, write_proof=options.mode == "prove").image_tag
    container_out = workspace.parent / "container_out"
    (container_out / "verifier").mkdir(parents=True, exist_ok=True)
    verifier_dir.mkdir(parents=True, exist_ok=True)

    transcript_chunks: list[str] = []
    solution_rc: int | None = None

    if not skip_solve:
        # The cache dir is bind-mounted at /tmp/output and written by uid 1000;
        # make it group/world-writable so the unprivileged user can write into a
        # host-owned directory on Linux (Docker Desktop ignores this on macOS).
        os.chmod(cache_dir, 0o777)
        solve_script = _container_solve_script(problem, sol_rel=sol_rel, render=options.render)
        solve_cmd = _docker_solve_command(image_tag, src, cache_dir, solve_script)
        solve_timeout = _agent_timeout(problem) + 300
        solution_rc, solve_out = _run_streaming(
            solve_cmd,
            timeout=solve_timeout,
            label="reference container solve (root env_server, uid 1000, no network)",
        )
        transcript_chunks.append(solve_out)
        if solution_rc != 0:
            transcript_path.write_text("\n".join(transcript_chunks))
            raise RuntimeError(
                f"reference container solve failed (status {solution_rc}). " f"stderr tail:\n{solve_out[-1500:]}"
            )

    # Outputs live in the cache (bind-mounted /tmp/output); surface them to the
    # workspace for both the solve and skip-solve paths.
    _copy_outputs(cache_dir, workspace, problem)

    score: float | None = None
    grader_rc: int | None = None
    grade_payload: dict = {}
    if not options.no_grade:
        grade_cmd = _docker_grade_command(image_tag, cache_dir, scorer_dir, container_out, _container_grade_script())
        grade_timeout = _verifier_timeout(problem) + 300
        grader_rc, grade_out = _run_streaming(
            grade_cmd,
            timeout=grade_timeout,
            label="reference container grade (root, no network)",
        )
        transcript_chunks.append(grade_out)
        if grader_rc != 0:
            transcript_path.write_text("\n".join(transcript_chunks))
            raise RuntimeError(
                f"reference container grade failed (status {grader_rc}). "
                f"stderr tail:\n{grade_out[-1500:]}"
                + _grade_failure_detail(container_out / "verifier")
            )

        for name in ("reward.json", "reward-details.json", "reward.txt"):
            s = container_out / "verifier" / name
            if s.exists():
                shutil.copy2(s, verifier_dir / name)
        reward_path = verifier_dir / "reward.json"
        if reward_path.exists():
            reward = json.loads(reward_path.read_text())
            score = float(reward["score"])
        details_path = verifier_dir / "reward-details.json"
        if details_path.exists():
            grade_payload = json.loads(details_path.read_text())
        elif score is not None:
            grade_payload = {"score": score}

    transcript_path.write_text("\n".join(transcript_chunks))

    return ReferenceRunResult(
        score=score,
        workspace=workspace,
        verifier_dir=verifier_dir,
        grade_payload=grade_payload,
        manifest_path=None,
        execution="container",
        solution_returncode=solution_rc,
        grader_returncode=grader_rc,
    )


def grade_workspace_in_container(
    problem: HarnessProblem,
    workspace: Path,
    verifier_dir: Path,
    transcript_path: Path,
) -> float:
    """Grade an already-populated workspace INSIDE the task image.

    Mirrors the reference container GRADE phase (root grader, host ``scorer/``
    over ``/mcp_server/grader``, root-only truth via baked ``/mcp_server/data``)
    so the scorer runs with the task's own image dependencies (e.g. ``pandas``,
    ``pyarrow``, ``scikit-learn``) rather than the host harness env, which only
    carries the centrally pinned grader runtime. Used by rubric-quality for
    tasks that resolve to container execution (``ml`` / accelerator /
    ``in_container``); host-resolved tasks keep using
    :func:`lbx_rl_tasks_harness.grading.grade_workspace`.
    """
    from lbx_rl_tasks_harness.docker import build_task_image

    if problem.source_problem_dir is None:
        raise ValueError("container grading requires a source problem directory")

    src = problem.source_problem_dir.resolve()
    scorer_dir = src / "scorer"
    image_tag = build_task_image(problem, write_proof=False).image_tag
    container_out = workspace.parent / "container_out"
    (container_out / "verifier").mkdir(parents=True, exist_ok=True)
    verifier_dir.mkdir(parents=True, exist_ok=True)

    grade_cmd = _docker_grade_command(image_tag, workspace, scorer_dir, container_out, _container_grade_script())
    grader_rc, grade_out = _run_streaming(
        grade_cmd,
        timeout=_verifier_timeout(problem) + 300,
        label="rubric-quality container grade (root, no network)",
    )
    transcript_path.write_text(grade_out)
    if grader_rc != 0:
        raise RuntimeError(
            f"container grade failed (status {grader_rc}). "
            f"stderr tail:\n{grade_out[-1500:]}"
            + _grade_failure_detail(container_out / "verifier")
        )

    for name in ("reward.json", "reward-details.json", "reward.txt"):
        s = container_out / "verifier" / name
        if s.exists():
            shutil.copy2(s, verifier_dir / name)
    reward_path = verifier_dir / "reward.json"
    if not reward_path.exists():
        raise RuntimeError("container grade did not produce reward.json")
    return float(json.loads(reward_path.read_text())["score"])


def run_reference(
    problem: HarnessProblem,
    workspace: Path,
    verifier_dir: Path,
    transcript_path: Path,
    *,
    options: ReferenceRunOptions | None = None,
    run_dir: Path | None = None,
) -> ReferenceRunResult:
    """Run a reference solution with optional cache and train/grade separation."""
    opts = options or ReferenceRunOptions()
    started = time.monotonic()
    execution = resolve_reference_execution(problem, mode=opts.mode)

    # Make the lost guarantees loud, not silent (HAR-3). An explicit host
    # request that got upgraded to the container path is explained; a run that
    # actually resolves to host gets the no-isolation warning.
    if opts.mode != "prove":
        if problem.reference.execution == "host" and execution == "container":
            _notify_host_upgraded_to_container(problem)
        elif execution == "host":
            _warn_host_isolation_bypass(problem)

    if opts.mode == "fast" or execution == "host":
        cache_dir = _cache_dir(problem, clean=opts.clean_cache)
        if execution == "host" and opts.mode == "fast":
            run_solution(problem, workspace, transcript_path)
            score = grade_workspace(problem, workspace, verifier_dir, transcript_path)
            details_path = verifier_dir / "reward-details.json"
            grade_payload = json.loads(details_path.read_text()) if details_path.exists() else {"score": score}
            result = ReferenceRunResult(
                score=score,
                workspace=workspace,
                verifier_dir=verifier_dir,
                grade_payload=grade_payload,
                manifest_path=None,
                execution="host",
                solution_returncode=0,
                grader_returncode=0,
            )
        else:
            result = _run_host_reference(
                problem,
                workspace,
                verifier_dir,
                transcript_path,
                options=opts,
                cache_dir=cache_dir,
            )
    else:
        cache_dir = _cache_dir(problem, clean=opts.clean_cache)
        result = _run_container_reference(
            problem,
            workspace,
            verifier_dir,
            transcript_path,
            options=opts,
            cache_dir=cache_dir,
        )

    manifest_path = _write_manifest(
        problem,
        options=opts,
        execution=result.execution,
        score=result.score,
        solution_returncode=result.solution_returncode,
        grader_returncode=result.grader_returncode,
        run_dir=run_dir,
        elapsed_s=time.monotonic() - started,
    )
    return ReferenceRunResult(
        score=result.score,
        workspace=result.workspace,
        verifier_dir=result.verifier_dir,
        grade_payload=result.grade_payload,
        manifest_path=manifest_path,
        execution=result.execution,
        solution_returncode=result.solution_returncode,
        grader_returncode=result.grader_returncode,
    )
