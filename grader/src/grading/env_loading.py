"""Load TRUSTED task-author Python modules from ``/mcp_server/data/`` via importlib.

For loading agent-submitted code (e.g. ``/tmp/output/policy.py``), use
:func:`grading.helpers.load_submitted_policy` instead -- that path runs in a
sandboxed worker that drops privileges to the unprivileged ``agent`` account and
isolates stdout/stderr. Calling raw importlib on agent code (or this helper)
gives the agent in-process execution as root with live stdout, which lets them
read ``/mcp_server/data/`` and forge a grade.

This helper is for the OTHER case: loading code that ships with the task at
build time, lives under ``/mcp_server/data/`` (root-only, agent can't read it),
and is part of the trusted grader -- typically the hidden environment module
(``env.py`` defining ``make_env``) used by ``[environment].hidden_env`` tasks
and by the ``env_server`` package. No sandboxing is needed; this just
deduplicates the ``importlib.util.spec_from_file_location`` incantation.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType

# Filesystem roots a ``load_env_module`` caller may load from. The helper is for
# trusted task-author code, so the allowlist matches the canonical task image's
# root-only data path. ``/tmp/output`` is deliberately excluded: that is the
# agent's writable scratch, and loading code from there bypasses the no-sandbox
# premise (use ``load_submitted_policy`` for agent code instead).
_DEFAULT_TRUSTED_ROOTS: tuple[Path, ...] = (Path("/mcp_server"),)


def _prepend_env_deps_to_syspath() -> None:
    """Put the root-only env-only site-packages (``/mcp_server/env_deps``) on
    ``sys.path`` so a hidden ``env.py`` can import server-only deps declared via
    the task's ``env_dependencies`` metadata.

    This runs inside :func:`load_env_module` -- the single choke point every
    env-module load funnels through: the ``env_server`` subprocess
    (``EnvServer._load_make_env``), the supervisor pre-flight import, AND the
    grade-time in-process load in a task's ``compute_score`` -- so the deps
    resolve in every root context that loads the env, not only the socket server.
    ``env_server.server.serve`` also prepends explicitly, which is now redundant
    but harmless (idempotent).

    Idempotent, and a no-op whenever the directory is absent -- which is every
    non-env task, and any unprivileged context: the uid-1000 agent cannot
    traverse the 0700 ``/mcp_server`` to reach ``env_deps``, so ``is_dir()`` is
    False and nothing is added to the agent's path (this augments only the root
    grader/server sys.path, never the sandboxed submission worker's)."""
    # Lazy import: grading and env_server import each other, so keep the edge
    # out of module load. ENV_DEPS_DIR is the single source of truth for the path.
    from env_server.config import ENV_DEPS_DIR

    deps = str(ENV_DEPS_DIR)
    if deps not in sys.path and ENV_DEPS_DIR.is_dir():
        sys.path.insert(0, deps)


def load_env_module(
    path: str | Path,
    *,
    factory_name: str | None = "make_env",
    module_name: str | None = None,
    trusted_roots: tuple[Path, ...] | None = None,
) -> ModuleType:
    """Import a trusted Python module from a filesystem path and return it.

    Args:
        path: Path to a single ``.py`` file (e.g. ``/mcp_server/data/env.py``)
            or a package ``__init__.py`` (e.g.
            ``/mcp_server/data/envs/__init__.py``). The package case sets
            ``submodule_search_locations`` to the parent dir so one level of
            relative imports (``from .grid import GridEnv``) resolves.
        factory_name: Name of a callable the module must define. Defaults to
            ``"make_env"`` (the env-task convention). Pass ``None`` to skip the
            check (helpers / metric modules).
        module_name: ``sys.modules`` key. When ``None``, derived from the path
            stem plus a hash of the parent dir so concurrent loads of two
            ``env.py`` files under different task dirs do not alias.
        trusted_roots: Allowed parent directories for ``path``; paths outside
            them raise ``ValueError`` (prevents loading agent code from
            ``/tmp/output/``). Defaults to ``/mcp_server``.

    Returns:
        The loaded module; pull attributes directly (``module.make_env(seed=0)``).

    Raises:
        FileNotFoundError: ``path`` does not exist.
        ValueError: ``path`` is outside ``trusted_roots``, or ``factory_name``
            was given and the module does not define a callable of that name.
        RuntimeError: importlib could not build a spec for the path.
    """
    # Env-only deps (env_dependencies) must be importable BEFORE env.py's
    # top-level imports execute in spec.loader.exec_module below. Doing it here
    # covers every caller (subprocess, supervisor pre-flight, grade-time load).
    _prepend_env_deps_to_syspath()

    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"env module not found at {path}")

    # resolve() collapses ../symlinks so /mcp_server/../tmp/output/policy.py
    # cannot masquerade as living under /mcp_server.
    resolved = path.resolve()
    roots = trusted_roots if trusted_roots is not None else _DEFAULT_TRUSTED_ROOTS
    if not any(resolved.is_relative_to(root.resolve()) for root in roots):
        root_list = ", ".join(str(r) for r in roots)
        raise ValueError(
            f"load_env_module refuses to load {resolved} -- path is outside the "
            f"trusted-roots allowlist [{root_list}]. This helper is for "
            "task-author code under /mcp_server/data/, not agent code under "
            "/tmp/output/. For agent code use "
            "grading.helpers.load_submitted_policy (sandboxed worker). To extend "
            "the allowlist for a legitimate non-/mcp_server load, pass "
            "trusted_roots=(...)."
        )

    if module_name is None:
        import hashlib

        parent_hash = hashlib.sha1(
            str(resolved.parent).encode("utf-8")
        ).hexdigest()[:8]
        module_name = f"task_env__{resolved.stem}__{parent_hash}"

    submodule_locations = [str(path.parent)] if path.name == "__init__.py" else None
    spec = importlib.util.spec_from_file_location(
        module_name, path, submodule_search_locations=submodule_locations
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not create import spec for {path}")

    mod = importlib.util.module_from_spec(spec)
    # Register before exec so the module's own internal references (decorators
    # inspecting __module__, package-relative imports) resolve. Pop on failure
    # so a broken module does not leave a half-initialized stub behind.
    sys.modules[spec.name] = mod
    try:
        spec.loader.exec_module(mod)
    except BaseException:
        sys.modules.pop(spec.name, None)
        raise

    if factory_name is not None:
        factory = getattr(mod, factory_name, None)
        if not callable(factory):
            kind = type(factory).__name__ if factory is not None else "missing"
            raise ValueError(
                f"{path} must define a callable named {factory_name!r}; got {kind}"
            )

    return mod
