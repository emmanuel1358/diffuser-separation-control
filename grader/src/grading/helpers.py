"""Deterministic predicate helpers returning bool / float in [0, 1] for use inside RubricBuilder criteria."""

from __future__ import annotations

import contextlib
import fcntl
import io
import json
import os
import re
import shutil
import stat
import subprocess
import sys
import tempfile
import threading
import zipfile
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import numpy as np

from grading.faults import AgentFault

_DEFAULT_MAX_SUBMISSION_BYTES = 64 * 1024 * 1024
# stdout cap for a captured agent executable: a flood past this is killed and
# raised as an AgentFault (kept 0.0), so the child cannot OOM the grader.
_DEFAULT_MAX_EXECUTABLE_OUTPUT_BYTES = 16 * 1024 * 1024
# UNCOMPRESSED cap for an agent .npz: the file-size cap bounds only the
# compressed bytes, so a small archive can expand to many GiB on member access.
# Generous enough for honest submissions, tight enough to bound a bomb well under
# grader OOM.
_DEFAULT_MAX_NPZ_UNCOMPRESSED_BYTES = 2 * 1024 * 1024 * 1024  # 2 GiB
_DEFAULT_MAX_HDF5_BYTES = 2 * 1024 * 1024 * 1024
# Execution timeout for the privilege-dropped HDF5/h5ad reader worker. Bounds a
# crafted file that tries to hang or OOM the grader during the libhdf5 parse;
# the read itself is one RPC call, so this is the per-parse budget.
_H5_READ_TIMEOUT_S = 120.0


def file_exists(path: str | Path, *, non_empty: bool = False) -> bool:
    """True iff ``path`` exists. Pass ``non_empty=True`` to also require content."""
    p = Path(path)
    if not p.exists() or not p.is_file():
        return False
    if non_empty:
        try:
            return bool(p.read_bytes().strip())
        except OSError:
            return False
    return True


def file_contains(
    path: str | Path,
    needle: str,
    *,
    case_sensitive: bool = True,
    encoding: str = "utf-8",
) -> bool:
    """True iff the file at ``path`` contains ``needle``."""
    p = Path(path)
    if not p.exists() or not p.is_file():
        return False
    try:
        text = p.read_text(encoding=encoding, errors="replace")
    except OSError:
        return False
    if case_sensitive:
        return needle in text
    return needle.lower() in text.lower()


def regex_search(
    text: str,
    pattern: str,
    *,
    flags: int = 0,
) -> bool:
    """True iff ``pattern`` matches anywhere in ``text``. Returns False on bad regex."""
    try:
        return re.search(pattern, text, flags) is not None
    except re.error:
        return False


def exact_match(
    actual: Any,
    expected: Any,
    *,
    value_type: str = "string",
) -> float:
    """Return 1.0 iff ``actual`` matches ``expected`` under the given type.

    ``value_type``: ``"string"`` (default), ``"number"``, ``"boolean"``,
    ``"list"`` (ordered), ``"set"`` (unordered).
    """
    if value_type == "number":
        try:
            return 1.0 if float(actual) == float(expected) else 0.0
        except (TypeError, ValueError):
            return 0.0
    if value_type == "boolean":
        return (
            1.0 if str(actual).strip().lower() == str(expected).strip().lower() else 0.0
        )
    if value_type in ("list", "set"):
        actual_items = (
            list(actual)
            if isinstance(actual, (list, tuple))
            else [x.strip() for x in str(actual).split(",") if x.strip()]
        )
        expected_items = (
            list(expected)
            if isinstance(expected, (list, tuple))
            else [x.strip() for x in str(expected).split(",") if x.strip()]
        )
        if value_type == "set":
            return 1.0 if set(actual_items) == set(expected_items) else 0.0
        return 1.0 if actual_items == expected_items else 0.0
    return 1.0 if str(actual).strip() == str(expected).strip() else 0.0


def abs_error(
    actual: float,
    expected: float,
    *,
    tolerance: float,
) -> float:
    """Tolerance-anchored score: ``max(0, 1 - |actual-expected|/tolerance)``.

    Returns ``1.0`` when ``actual == expected``. Returns ``0.0`` when the
    error meets or exceeds ``tolerance``. Useful for "regression target"
    style criteria.
    """
    if tolerance == 0:
        return 1.0 if actual == expected else 0.0
    return max(0.0, 1.0 - abs(float(actual) - float(expected)) / abs(float(tolerance)))


def kendall_tau(
    actual: list[Any],
    expected: list[Any],
) -> float:
    """Kendall's tau ranking similarity, mapped to ``[0, 1]``.

    ``1.0`` = identical ordering, ``0.5`` = random, ``0.0`` = perfectly reversed.
    """
    if not expected or not actual:
        return 0.0
    n = min(len(expected), len(actual))
    concordant = 0
    discordant = 0
    for i in range(n):
        for j in range(i + 1, n):
            if i >= len(actual) or j >= len(actual):
                continue
            try:
                ei = expected.index(actual[i])
                ej = expected.index(actual[j])
            except ValueError:
                discordant += 1
                continue
            sign = (i - j) * (ei - ej)
            if sign > 0:
                concordant += 1
            elif sign < 0:
                discordant += 1
    total = concordant + discordant
    if total == 0:
        return 1.0
    tau = (concordant - discordant) / total
    return max(0.0, (tau + 1) / 2)


def jaccard(actual: Iterable[Any], expected: Iterable[Any]) -> float:
    """Jaccard set similarity: ``|A & E| / |A | E|``. Empty/empty = 1.0."""
    a = set(actual)
    e = set(expected)
    if not a and not e:
        return 1.0
    union = a | e
    if not union:
        return 0.0
    return len(a & e) / len(union)


def json_path(data: Any, path: str) -> Any:
    """JSONPath-like extraction. Returns ``None`` on miss.

    Supported syntax:
      * ``$.key.0.nested`` (dot-numeric for list indices)
      * ``$.items[0].name``
      * ``$['keys.with.dots']["display name"]``
      * ``$.items[?symbol=TRX].amount`` (filter by field equality)
    """
    if not path:
        return data
    parts = _parse_json_path(path)
    if parts is None:
        return None
    current: Any = data
    for part in parts:
        if isinstance(part, tuple):
            _, field, expected = part
            if not isinstance(current, (list, tuple)):
                return None
            match = None
            for item in current:
                value = (
                    item.get(field)
                    if isinstance(item, dict)
                    else getattr(item, field, None)
                )
                if value is not None and str(value) == expected:
                    match = item
                    break
            current = match
            if current is None:
                return None
            continue

        if not part:
            continue
        try:
            idx = int(part)
            if isinstance(current, (list, tuple)):
                try:
                    current = current[idx]
                except IndexError:
                    return None
                continue
        except ValueError:
            pass
        if isinstance(current, dict):
            current = current.get(part)
        else:
            current = getattr(current, part, None)
        if current is None:
            return None
    return current


def transcript_contains(
    transcript: str | None,
    needle: str,
    *,
    case_sensitive: bool = False,
) -> bool:
    """True iff ``needle`` appears in the transcript text."""
    if not transcript:
        return False
    if case_sensitive:
        return needle in transcript
    return needle.lower() in transcript.lower()


def load_json(path: str | Path) -> Any:
    """Load a JSON file, returning ``None`` on missing/invalid."""
    p = Path(path)
    if not p.exists() or not p.is_file():
        return None
    try:
        return json.loads(p.read_text())
    except (OSError, json.JSONDecodeError):
        return None


def load_submission_or_fault(
    path: os.PathLike | str,
    *,
    required_columns: Iterable[str] | None = None,
    n_rows: int | None = None,
    numeric_columns: Iterable[str] | None = None,
    unique_key_column: str | None = None,
    max_bytes: int = _DEFAULT_MAX_SUBMISSION_BYTES,
    read_csv_kwargs: dict[str, Any] | None = None,
    allow_extra_columns: bool = False,
):
    """Read an agent CSV submission, raising ``AgentFault`` for bad outputs.

    This closes common free-veto paths: symlinks to hidden data, FIFOs that
    hang grading, oversized files that OOM the verifier, malformed CSV, missing
    columns, non-finite values, duplicate merge keys, and unexpected columns.
    """
    import pandas as pd

    p = Path(path)
    try:
        st = os.lstat(p)
    except FileNotFoundError as exc:
        raise AgentFault(f"submission not found at {p}") from exc
    except OSError as exc:
        raise AgentFault(f"submission at {p} could not be stat'd: {exc}") from exc

    if not stat.S_ISREG(st.st_mode):
        raise AgentFault(
            f"submission at {p} is not a regular file "
            f"(mode={stat.filemode(st.st_mode)}); expected a plain CSV file"
        )
    if st.st_size == 0:
        raise AgentFault(f"submission at {p} is empty")
    if st.st_size > max_bytes:
        raise AgentFault(
            f"submission at {p} is {st.st_size} bytes, over the "
            f"{max_bytes}-byte limit"
        )

    try:
        df = pd.read_csv(p, **(read_csv_kwargs or {}))
    except Exception as exc:
        raise AgentFault(
            f"submission at {p} is not a readable CSV: {type(exc).__name__}: {exc}"
        ) from exc

    required = list(required_columns) if required_columns is not None else None
    numeric = list(numeric_columns) if numeric_columns is not None else None

    if required is not None:
        missing = [column for column in required if column not in df.columns]
        if missing:
            raise AgentFault(f"submission missing required columns: {missing}")

    if n_rows is not None and len(df) != n_rows:
        raise AgentFault(f"submission has {len(df)} rows, expected {n_rows}")

    if numeric is not None:
        absent = [column for column in numeric if column not in df.columns]
        if absent:
            raise AgentFault(f"submission missing numeric columns: {absent}")
        try:
            values = df[numeric].to_numpy(dtype=float)
        except (TypeError, ValueError) as exc:
            raise AgentFault(f"non-numeric values in columns {numeric}: {exc}") from exc
        if not np.isfinite(values).all():
            raise AgentFault(f"non-finite (NaN/Inf) values in columns {numeric}")

    if unique_key_column is not None:
        if unique_key_column not in df.columns:
            raise AgentFault(f"submission missing key column {unique_key_column!r}")
        duplicate_count = int(df[unique_key_column].duplicated().sum())
        if duplicate_count:
            raise AgentFault(
                f"submission has {duplicate_count} duplicate value(s) in key "
                f"column {unique_key_column!r}; expected one row per key"
            )

    if not allow_extra_columns and (required is not None or numeric is not None):
        expected: set[str] = set()
        if required is not None:
            expected.update(required)
        if numeric is not None:
            expected.update(numeric)
        if unique_key_column is not None:
            expected.add(unique_key_column)
        extra = [column for column in df.columns if column not in expected]
        if extra:
            raise AgentFault(
                f"submission has unexpected column(s) {extra}; expected "
                f"exactly {sorted(expected)}. Pass allow_extra_columns=True "
                "if extra columns are intended."
            )

    return df


def _reject_npz_decompression_bomb(
    raw: bytes, path: Path, max_uncompressed_bytes: int
) -> None:
    """Bound the UNCOMPRESSED size of an agent ``.npz`` before its members load.

    A ``.npz`` is a ZIP, so the file-size cap bounds only the COMPRESSED bytes: a
    ~5 MiB decompression-bomb archive can expand to many GiB when the caller
    materializes each member -> grader OOM (``MemoryError``, a non-AgentFault ->
    DISCARDED free veto). The ZIP central directory records each member's
    uncompressed size WITHOUT decompressing, so sum those and raise AgentFault
    past the cap. A plain ``.npy`` is not a ZIP (``BadZipFile``) and needs no
    check -- its on-disk size already bounds the array it holds.
    """
    try:
        with zipfile.ZipFile(io.BytesIO(raw)) as zf:
            total = sum(max(0, zi.file_size) for zi in zf.infolist())
    except zipfile.BadZipFile:
        return  # a .npy (or non-zip) -- the compressed file-size cap bounds it
    if total > max_uncompressed_bytes:
        raise AgentFault(
            f"submission at {path} decompresses to {total} bytes, over the "
            f"{max_uncompressed_bytes}-byte uncompressed limit (decompression bomb?)"
        )


def load_submission_npz_or_fault(
    path: os.PathLike | str,
    *,
    allow_pickle: bool = False,
    max_bytes: int = _DEFAULT_MAX_SUBMISSION_BYTES,
    max_uncompressed_bytes: int = _DEFAULT_MAX_NPZ_UNCOMPRESSED_BYTES,
):
    """Read an agent ``.npz`` / ``.npy`` submission as the (root) grader, raising
    ``AgentFault`` for bad outputs. Numpy analog of :func:`load_submission_or_fault`.

    Opens with ``O_NOFOLLOW`` so a symlink to the held-out truth cannot be
    followed (a raw ``np.load`` would read it back as "predictions"), rejects
    FIFO/dir/device/oversized files, and defaults ``allow_pickle=False`` so a
    crafted array cannot execute ``__reduce__`` in the root grader. Returns an
    ``ndarray`` for a ``.npy`` submission, or a plain ``dict`` ``{name: ndarray}``
    for a ``.npz`` -- whose members are materialized eagerly here (inside the
    AgentFault wrapper), so an object-array member is a kept-0.0 fault rather than
    a bare ``ValueError`` at ``data[name]`` access time (which the runtime would
    discard as env_internal_failure -- a void-veto).
    """
    p = Path(path)
    # O_NOFOLLOW: a symlink fails to open, never followed into /mcp_server.
    # O_NONBLOCK: a writerless FIFO opens immediately and is rejected below
    # instead of blocking the grader forever (a hang is itself a free veto).
    flags = os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK | getattr(os, "O_CLOEXEC", 0)
    try:
        fd = os.open(p, flags)
    except FileNotFoundError as exc:
        raise AgentFault(f"submission not found at {p}") from exc
    except OSError as exc:
        # ELOOP (symlink) / ENXIO / etc. -> agent-controlled, kept 0.0.
        raise AgentFault(
            f"submission at {p} could not be opened as a regular file "
            f"(symlink/FIFO/permission?): {exc}"
        ) from exc
    try:
        st = os.fstat(fd)
        if not stat.S_ISREG(st.st_mode):
            raise AgentFault(
                f"submission at {p} is not a regular file "
                f"(mode={stat.filemode(st.st_mode)}); expected a plain .npz/.npy"
            )
        if st.st_size == 0:
            raise AgentFault(f"submission at {p} is empty")
        if st.st_size > max_bytes:
            raise AgentFault(
                f"submission at {p} is {st.st_size} bytes, over the "
                f"{max_bytes}-byte limit"
            )
        # Confirmed a regular file: drop O_NONBLOCK so the read behaves normally.
        fcntl.fcntl(fd, fcntl.F_SETFL, fcntl.fcntl(fd, fcntl.F_GETFL) & ~os.O_NONBLOCK)
        with os.fdopen(fd, "rb", closefd=True) as fh:
            fd = -1  # fdopen owns the descriptor
            raw = fh.read()
    finally:
        if fd >= 0:
            os.close(fd)

    # Bound the UNCOMPRESSED expansion of a .npz before materializing members, so
    # a decompression bomb under the compressed cap cannot OOM the grader.
    _reject_npz_decompression_bomb(raw, p, max_uncompressed_bytes)

    try:
        obj = np.load(io.BytesIO(raw), allow_pickle=allow_pickle)
    except AgentFault:
        raise
    except Exception as exc:
        raise AgentFault(
            f"submission at {p} is not a readable .npz/.npy: {type(exc).__name__}: {exc}"
        ) from exc

    # For a .npz the return above is a LAZY NpzFile: allow_pickle is enforced per
    # member at access time, so an object-array member loads clean here and raises
    # a bare ValueError later at the grader's ``data[name]`` -- OUTSIDE this
    # AgentFault wrapper, so the runtime records env_internal_failure and DISCARDS
    # the rollout (a void-veto) instead of the earned 0.0. Materialize every member
    # HERE so a pickle-requiring (or truncated) member is an AgentFault, and return
    # a plain dict with no residual descriptor.
    if hasattr(obj, "files"):  # NpzFile
        try:
            materialized = {name: obj[name] for name in obj.files}
        except AgentFault:
            raise
        except Exception as exc:
            raise AgentFault(
                f"submission at {p} has an unreadable or pickle-requiring member: "
                f"{type(exc).__name__}: {exc}"
            ) from exc
        finally:
            obj.close()
        return materialized
    return obj


def require_regular_file(
    path: os.PathLike | str,
    *,
    max_bytes: int = _DEFAULT_MAX_SUBMISSION_BYTES,
) -> Path:
    """Verify an agent-submission path is a plain regular file within a size
    bound (raising ``AgentFault`` otherwise) and return it, for hand-rolled reads
    when no format-specific loader fits. Uses ``os.lstat`` so a symlink to the
    held-out truth is rejected rather than read as root.

    NOTE: this is an ``lstat``-then-read guard, race-free ONLY while no uid>=1000
    process runs concurrently with the subsequent read -- which holds for the
    ordinary "read the submission once, after the agent is SIGKILLed pre-grade"
    grader. If your grader runs agent code MID-grade (``run_submitted_executable``
    / ``run_policy``) and a double-forked child could survive it, that child can
    swap a regular file for a symlink between this check and your read; open the
    fd yourself with ``os.open(path, os.O_RDONLY | os.O_NOFOLLOW)`` and read THAT
    descriptor instead (``load_submission_npz_or_fault`` does this internally).
    """
    p = Path(path)
    try:
        st = os.lstat(p)
    except FileNotFoundError as exc:
        raise AgentFault(f"submission not found at {p}") from exc
    except OSError as exc:
        raise AgentFault(f"submission at {p} could not be stat'd: {exc}") from exc
    if not stat.S_ISREG(st.st_mode):
        raise AgentFault(
            f"submission at {p} is not a regular file "
            f"(mode={stat.filemode(st.st_mode)}); a symlink / FIFO / directory / "
            f"device is not accepted"
        )
    if st.st_size > max_bytes:
        raise AgentFault(
            f"submission at {p} is {st.st_size} bytes, over the {max_bytes}-byte limit"
        )
    return p


_POLICY_FILENAMES = ("policy.py", "submission.py", "agent.py")


def _resolve_policy_path(policy: str | Path) -> Path:
    """Resolve ``policy`` (a file or a dir/workspace) to a concrete policy file.

    Accepts the submitted policy file directly, or a workspace/output directory
    in which a conventional policy file (``policy.py`` etc.) is located.
    """
    p = Path(policy)
    if p.is_file():
        return p
    if p.is_dir():
        for name in _POLICY_FILENAMES:
            candidate = p / name
            if candidate.is_file():
                return candidate
    raise FileNotFoundError(f"no submitted policy found at {policy}")


def run_policy(
    policy: str | Path,
    *,
    timeout_s: float = 5.0,
    first_call_timeout_s: float | None = None,
    cwd: str | Path | None = None,
    unshare_ipc: bool = True,
):
    """Return a hardened worker for a submitted policy — the single approved way to run agent code.

    Wraps :class:`grading.policy_runner.PolicyWorker`: the policy runs in a
    non-root subprocess when the grader is root, and its return value travels
    over a dedicated fd so the score is NEVER parsed from stdout. Use as a
    context manager and read the score from the return value, not stdout.
    ``unshare_ipc`` (default True) gives each worker a private IPC namespace.
    """
    from grading.policy_runner import PolicyWorker

    return PolicyWorker(
        _resolve_policy_path(policy),
        timeout_s=timeout_s,
        first_call_timeout_s=first_call_timeout_s,
        cwd=Path(cwd) if cwd is not None else None,
        unshare_ipc=unshare_ipc,
    )


def run_model_module(
    policy: str | Path,
    method: str,
    *args: Any,
    timeout_s: float = 60.0,
    first_call_timeout_s: float | None = None,
    cwd: str | Path | None = None,
    **kwargs: Any,
) -> Any:
    """Call one function of a submitted module in the sandbox, and return it.

    Convenience over :func:`run_policy` for the common structural/numerical
    pattern where the model submits a module exposing a single callable — e.g.
    ``evaluate_section(case_path)`` or ``compute_rayleigh_damping(omegas, ...)``
    — and the grader calls it once and scores the returned values. This is the
    **only approved way** for a grader to execute that submitted function:
    never ``import``/``exec_module`` the model's file in the grading process
    (which runs as root). The call runs in a non-root subprocess; the result
    travels over a dedicated proto file descriptor, never the policy's stdout.

    Inputs and the return value cross a JSON boundary, so pass JSON-serializable
    arguments (e.g. a path string, lists of numbers) and return JSON-serializable
    results (dicts/lists/scalars). Example::

        from grading import helpers

        result = helpers.run_model_module(
            "/tmp/output/fixed_fiber_section_processor.py",
            "evaluate_section",
            str(public_case_path),
        )
        score = compare_to_reference(result)   # never trust the module's stdout
    """
    with run_policy(
        policy,
        timeout_s=timeout_s,
        first_call_timeout_s=first_call_timeout_s,
        cwd=cwd,
    ) as worker:
        return worker.call(method, *args, **kwargs)


def load_submitted_policy(*args: Any, **kwargs: Any) -> Any:
    """Load a submitted policy in a sandboxed worker, returning proxy handle(s).

    Thin re-export of :func:`grading.policy_runner.load_submitted_policy`; see
    that docstring for the full contract. Raises ``AgentFault`` for agent-caused
    load failures.
    """
    from grading.policy_runner import load_submitted_policy as _impl

    return _impl(*args, **kwargs)


# -- Submitted model artifacts (sandboxed pickle/joblib proxy) --------------
# load_submitted_model deserializes /tmp/output/model.pkl and forwards every
# predict() call inside the privilege-dropped worker, so agent code can neither
# read /mcp_server nor spoof the grader's stdout.

DEFAULT_MODEL_PATH = Path("/tmp/output/model.pkl")

# OOM-bomb backstop: reject an over-sized artifact before the worker reads it.
_MAX_MODEL_BYTES = 2 * 1024 * 1024 * 1024  # 2 GiB

# Match the ML_Envs oracle's model-loader timeouts (init 300s / rpc 300s). The 2
# GiB artifact cap admits honest large models whose deserialize exceeds the 30s
# first-call floor, or whose held-out-set predict exceeds the 5s per-call default
# of load_submitted_policy -- either would otherwise be killed and scored a kept
# 0.0. load_submitted_model forwards this for BOTH the (deserialize) handshake and
# every forwarded predict; it does not widen the env-rollout defaults.
_MODEL_LOAD_TIMEOUT_S = 300.0

# Fully LITERAL wrapper sources: `source` must never interpolate a runtime value.
# The artifact path reaches each wrapper via ``__file__`` (source mode sets it to
# the passed path). At factory-call time the wrapper imports the stdlib
# deserializer FIRST while /tmp/output is off sys.path (so an agent pickle.py
# cannot shadow it), THEN inserts the artifact dir + agent user-site for
# sibling-class resolution. The user-site is added with a plain sys.path.insert
# (never site.addsitedir / HOME) so no agent .pth executes as root, and only
# inside the dropped worker -- which holds no truth and returns over msgpack.
_AGENT_USER_SITE_SNIPPET = '''
    import pwd, glob
    try:
        _home = os.environ.get("TAIGA_AGENT_HOME") or pwd.getpwuid(os.getuid()).pw_dir
    except KeyError:
        # Worker uid has no /etc/passwd entry (non-canonical / derived image).
        # Degrade to skipping the user-site block rather than failing the load.
        # Set TAIGA_AGENT_HOME to recover the user-site resolution on such an image.
        _home = ""
    if _home:
        for _site in sorted(glob.glob(os.path.join(
                _home, ".local", "lib", "python*", "site-packages"))):
            if os.path.isdir(_site) and _site not in sys.path:
                sys.path.insert(0, _site)
'''

_PICKLE_WRAPPER = '''
def load_policy():
    import os, sys, pickle
    model_path = __file__
''' + _AGENT_USER_SITE_SNIPPET + '''
    data_dir = os.path.dirname(model_path) or "."
    if data_dir not in sys.path:
        sys.path.insert(0, data_dir)
    with open(model_path, "rb") as fh:
        return pickle.load(fh)
'''

_JOBLIB_WRAPPER = '''
def load_policy():
    import os, sys, joblib
    model_path = __file__
''' + _AGENT_USER_SITE_SNIPPET + '''
    data_dir = os.path.dirname(model_path) or "."
    if data_dir not in sys.path:
        sys.path.insert(0, data_dir)
    return joblib.load(model_path)
'''

_WRAPPERS = {"pickle": _PICKLE_WRAPPER, "joblib": _JOBLIB_WRAPPER}


def load_submitted_model(
    path: str | Path = DEFAULT_MODEL_PATH,
    *,
    deserializer: str = "pickle",
    max_bytes: int = _MAX_MODEL_BYTES,
) -> Any:
    """Load an agent-submitted model artifact in a sandboxed worker, returning a
    proxy whose ``.predict(...)`` etc. forward to the worker (call ``.close()``
    when done). The deserialize and every forwarded call run with privileges
    dropped to uid/gid 1000, so neither can read ``/mcp_server`` or spoof stdout.

    ``deserializer`` is ``"pickle"`` (also reads cloudpickle) or ``"joblib"``.
    ``predict`` inputs/outputs must be msgpack-encodable. Raises ``ValueError``
    for an unknown deserializer, ``AgentFault`` (kept 0.0) for a missing /
    non-regular / over-sized artifact or a deserialize that raises, and
    ``RuntimeError`` for an infra load failure (dead/malformed worker) or a load /
    predict that exceeds the 300s timeout (surfaced as ``PolicyTimeoutError``, a
    ``RuntimeError`` subclass, so ``except RuntimeError: return 0.0`` keeps it).
    """
    if deserializer not in _WRAPPERS:
        raise ValueError(
            "deserializer must be one of %s; got %r"
            % (sorted(_WRAPPERS), deserializer)
        )
    path = Path(path)
    if not path.exists():
        raise AgentFault("Missing submitted model at %s" % path)
    # Source mode skips load_submitted_policy's lstat/size guard, so do it here:
    # a non-regular or over-sized artifact is an agent fault (lstat won't follow
    # symlinks).
    st = os.lstat(path)
    if not stat.S_ISREG(st.st_mode):
        raise AgentFault(
            "submitted model at %s is not a regular file (mode=%s); "
            "expected a plain file" % (path, stat.filemode(st.st_mode))
        )
    if st.st_size > max_bytes:
        raise AgentFault(
            "submitted model at %s is %d bytes, over the %d-byte limit"
            % (path, st.st_size, max_bytes)
        )

    # source mode: `path` sets the worker's module.__file__ (read by the wrapper)
    # and is NOT auto-added to sys.path; sys_path_dirs=[] keeps /tmp/output off
    # the path until the wrapper inserts it, after the deserializer import.
    return load_submitted_policy(
        source=_WRAPPERS[deserializer],
        factory_name="load_policy",
        path=path,
        sys_path_dirs=[],
        timeout_s=_MODEL_LOAD_TIMEOUT_S,
        first_call_timeout_s=_MODEL_LOAD_TIMEOUT_S,
    )


def world_integrity(
    model: Any,
    *,
    expect_gravity: tuple[float, float, float] | None = (0.0, 0.0, -9.81),
    gravity_tol: float = 0.10,
    forbid_gravcomp: bool = True,
    forbid_equality: bool = True,
    require_contacts: bool = True,
) -> tuple[bool, list[str]]:
    """Validate a submitted MJCF preserves the intended physics. ``(ok, violations)``.

    Agents frequently "win" a MuJoCo task by quietly rigging the world rather
    than controlling it. This catches the recurring MJCF reward-hacks so a
    scorer can zero (or gate) a submission whose physics was tampered with:

    * **gravity** disabled or tilted away from ``expect_gravity`` (pass
      ``expect_gravity=None`` to skip, eg. space tasks);
    * **gravcomp** > 0 on any body (per-body gravity compensation = free
      floating);
    * **<equality>** constraints (weld/connect/joint) that can slave a
      "passive" body to an actuated one;
    * **contacts** globally disabled, or every geom opted out of collision.

    Each check is opt-out via its keyword so legitimately exotic tasks can relax
    it. ``model`` is a ``mujoco.MjModel``. Returns ``(True, [])`` when clean.

    Example::

        ok, why = helpers.world_integrity(model, expect_gravity=(0, 0, -9.81))
        if not ok:
            rb.metadata["world_violations"] = why
            return 0.0  # rigged world ⇒ no credit
    """
    import mujoco  # lazy: only graders that call this need the dep
    import numpy as np

    violations: list[str] = []

    if expect_gravity is not None:
        gravity = np.asarray(model.opt.gravity, dtype=float)
        target = np.asarray(expect_gravity, dtype=float)
        if float(np.linalg.norm(gravity - target)) > gravity_tol:
            violations.append(
                f"gravity {gravity.tolist()} deviates from expected "
                f"{target.tolist()} (tol {gravity_tol})"
            )

    if forbid_gravcomp:
        gravcomp = getattr(model, "body_gravcomp", None)
        if gravcomp is not None and float(np.max(np.abs(np.asarray(gravcomp)))) > 1e-9:
            violations.append("body gravcomp is non-zero (gravity compensation)")

    if forbid_equality and int(getattr(model, "neq", 0)) > 0:
        active = model.eq_active0 if hasattr(model, "eq_active0") else None
        n_active = (
            int(np.count_nonzero(np.asarray(active)))
            if active is not None
            else int(model.neq)
        )
        if n_active > 0:
            violations.append(
                f"{n_active} <equality> constraint(s) present (can slave passive bodies)"
            )

    if require_contacts:
        disabled = int(model.opt.disableflags) & int(
            mujoco.mjtDisableBit.mjDSBL_CONTACT
        )
        if disabled:
            violations.append("contacts globally disabled via opt.disableflags")
        elif model.ngeom > 0:
            contype = np.asarray(model.geom_contype)
            conaffinity = np.asarray(model.geom_conaffinity)
            if not contype.any() and not conaffinity.any():
                violations.append(
                    "every geom opts out of collision (contype/conaffinity all 0)"
                )

    return (not violations), violations


# ── Submission loaders: executable + HDF5 (per-type attack closures) ──────


# uid/gid 1000 is the base image's `model` user; /mcp_server is 0700 root-only,
# so after the drop this UID cannot read /mcp_server/data.
SUBPROCESS_UID = 1000
SUBPROCESS_GID = 1000


def _drop_privileges_to_model() -> None:
    """preexec_fn (after fork, before exec): demote the child to uid/gid 1000.

    setgroups([]) must come first (needs CAP_SETGID) to drop supplementary
    groups. Only runs when the grader is root; otherwise there is nothing to
    demote and setgroups/setgid would EPERM.
    """
    if os.geteuid() != 0:
        return
    os.setgroups([])
    os.setgid(SUBPROCESS_GID)
    os.setuid(SUBPROCESS_UID)


# Minimal env for an agent subprocess: only what bash/interpreters/locale need,
# deliberately excluding any grading-server secrets. The default env for
# run_submitted_executable absent an explicit env / env_passthrough.
_DEFAULT_ENV_ALLOWLIST = (
    "PATH",
    "HOME",
    "LANG",
    "LC_ALL",
    "LC_CTYPE",
    "TERM",
    "TMPDIR",
)


def _build_sanitized_env() -> dict[str, str]:
    """Minimal env dict copying only ``_DEFAULT_ENV_ALLOWLIST`` from the current process."""
    return {key: os.environ[key] for key in _DEFAULT_ENV_ALLOWLIST if key in os.environ}


def _kill_quietly(proc: subprocess.Popen) -> None:
    with contextlib.suppress(OSError):
        proc.kill()


def run_submitted_executable(
    cmd: list[str],
    *,
    stdin_bytes: bytes | None = None,
    cwd: str | Path | None = None,
    env: dict[str, str] | None = None,
    env_passthrough: bool = False,
    timeout_s: float | None = None,
    max_output_bytes: int = _DEFAULT_MAX_EXECUTABLE_OUTPUT_BYTES,
    streaming: bool = False,
) -> subprocess.CompletedProcess:
    """Spawn an agent-supplied executable with privileges dropped to uid/gid 1000
    and stdout isolated from the RUBRIC_SCORE= parser (the OS-process companion to
    ``load_submitted_policy``).

    Two output modes:

    * capture (default): stdout is streamed through a HARD-CAPPED reader
      (``max_output_bytes``) and returned as bytes; stderr is discarded at the
      kernel (``/dev/null``). Neither reaches the grading server's stdout, so
      RUBRIC_SCORE= spoofing is impossible UNLESS the caller relays ``proc.stdout``
      itself. Use when the solver's stdout IS the answer. RUBRIC_SCORE= is NOT
      scrubbed here (that would corrupt the very stdout the caller parses), so the
      caller must never relay raw ``proc.stdout`` to the score parser.
    * streaming (``streaming=True``): lines are pumped to ``sys.stderr`` (unscanned
      for the score) with any ``RUBRIC_SCORE=`` line dropped. Assumes
      newline-terminated output; a child writing >~64 KiB without a newline can
      deadlock the line pump.

    ``env`` (verbatim) overrides ``env_passthrough`` overrides the sanitized
    allowlist (the safe default). Capture mode returns bytes stdout + ``b""``
    stderr; streaming returns ``b""`` for both. Raises ``ValueError`` for
    stdin_bytes/timeout_s in streaming mode. In capture mode an agent-caused
    timeout OR a stdout flood past ``max_output_bytes`` raises ``AgentFault``
    (kept 0.0) -- NOT the builtin ``TimeoutExpired`` / ``MemoryError``, which
    would escape as env_internal_failure and DISCARD the rollout (a free veto).
    """
    if streaming:
        if stdin_bytes is not None:
            raise ValueError(
                "stdin_bytes is not supported in streaming mode (mixes binary "
                "stdin with text stdout streaming). Use capture mode if you need "
                "to pipe bytes to the agent's stdin."
            )
        if timeout_s is not None:
            raise ValueError(
                "timeout_s is not supported in streaming mode (the line-pump "
                "blocks on the child's stdout, defeating subprocess timeouts). "
                "Wrap the command with GNU timeout(1) or rely on Taiga's "
                "grading_timeout_seconds for the wall-clock cap."
            )

    cwd_str = str(cwd) if cwd is not None else None

    if env is not None:
        child_env: dict[str, str] | None = env
    elif env_passthrough:
        child_env = None
    else:
        child_env = _build_sanitized_env()

    if not streaming:
        # Bounded capture: stream stdout through a capped reader that kills the
        # child the moment its output crosses max_output_bytes (a true memory
        # bound, not a post-capture check), discard stderr at the kernel, and feed
        # stdin from a writer thread so a large stdin can't deadlock against a
        # child that fills the stdout pipe. An agent-caused timeout / over-cap is
        # converted to AgentFault (kept 0.0) instead of escaping as
        # TimeoutExpired / MemoryError -> a DISCARDED (free-veto) rollout.
        proc = subprocess.Popen(
            cmd,
            stdin=subprocess.PIPE if stdin_bytes is not None else subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            cwd=cwd_str,
            env=child_env,
            preexec_fn=_drop_privileges_to_model,
        )
        captured = bytearray()
        cap_limit = max_output_bytes + 1
        over_cap = threading.Event()

        def _drain_stdout() -> None:
            assert proc.stdout is not None
            try:
                while True:
                    chunk = proc.stdout.read1(65536)
                    if not chunk:
                        break
                    if len(captured) < cap_limit:
                        captured.extend(chunk)
                    if len(captured) > max_output_bytes:
                        over_cap.set()
                        _kill_quietly(proc)
                        break
            except (OSError, ValueError):
                pass

        def _feed_stdin() -> None:
            if stdin_bytes is None or proc.stdin is None:
                return
            try:
                proc.stdin.write(stdin_bytes)
            except (OSError, ValueError):
                pass
            finally:
                with contextlib.suppress(OSError):
                    proc.stdin.close()

        reader = threading.Thread(target=_drain_stdout, daemon=True)
        reader.start()
        writer = threading.Thread(target=_feed_stdin, daemon=True)
        writer.start()
        try:
            proc.wait(timeout=timeout_s)
        except subprocess.TimeoutExpired as exc:
            _kill_quietly(proc)
            raise AgentFault(
                f"submitted executable timed out after {timeout_s:.1f}s"
            ) from exc
        finally:
            reader.join(timeout=5.0)
            writer.join(timeout=5.0)
            with contextlib.suppress(OSError):
                if proc.stdout is not None:
                    proc.stdout.close()
        if over_cap.is_set():
            raise AgentFault(
                f"submitted executable produced more than {max_output_bytes} "
                f"bytes of stdout"
            )
        return subprocess.CompletedProcess(
            args=cmd, returncode=proc.returncode, stdout=bytes(captured), stderr=b"",
        )

    # Streaming: pump lines to sys.stderr (never the score-parsed sys.stdout),
    # dropping RUBRIC_SCORE= lines; utf-8/replace so a bad byte can't break the pump.
    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        cwd=cwd_str,
        env=child_env,
        encoding="utf-8",
        errors="replace",
        bufsize=1,
        preexec_fn=_drop_privileges_to_model,
    )
    assert proc.stdout is not None
    for line in proc.stdout:
        if "RUBRIC_SCORE=" in line:
            continue
        sys.stderr.write(line)
        sys.stderr.flush()
    proc.wait()
    return subprocess.CompletedProcess(
        args=cmd, returncode=proc.returncode, stdout=b"", stderr=b"",
    )


# LITERAL source wrapper executed INSIDE the privilege-dropped policy worker
# (grading.policy_runner.load_submitted_policy, source mode). The libhdf5 parse
# of the agent-controlled file therefore runs as the unprivileged agent uid --
# never in the (root) grader process -- so a crafted ``.h5`` that trips a
# libhdf5 memory-safety bug cannot execute with root + held-out-truth read
# access. Requested dataset names + the output dir arrive as msgpack CALL
# ARGUMENTS to ``read`` (never interpolated into this source, per the
# policy-loader security note). Each dataset is written to ``out_dir`` as a
# non-object ``.npy`` (the parent loads it with ``allow_pickle=False``, so no
# pickle ever executes on the cross-process transfer); only the small file
# manifest returns over the msgpack wire, so a multi-hundred-MiB read is not
# bounded by the policy protocol's 1 GiB per-frame limit. Every logical guard
# runs here, before any dataset is materialized.
_H5_READER_WRAPPER = '''
def load_reader():
    import os
    import h5py
    import numpy as np

    src_path = __file__

    class _Reader:
        def read(self, names, out_dir):
            try:
                written = []
                with h5py.File(src_path, "r", locking=False) as f:
                    if names is None:
                        names = list(f.keys())
                    for idx, name in enumerate(names):
                        if "/" in name.strip("/"):
                            return {"ok": False, "reason": "dataset %r must be top-level (no nested groups)" % (name,)}
                        link = f.get(name, getlink=True)
                        if link is None:
                            return {"ok": False, "reason": "dataset %r not found" % (name,)}
                        if not isinstance(link, h5py.HardLink):
                            return {"ok": False, "reason": "dataset %r must be a direct hard link, got %s (external/soft links are rejected)" % (name, type(link).__name__)}
                        obj = f[name]
                        if not isinstance(obj, h5py.Dataset):
                            return {"ok": False, "reason": "%r is not a dataset" % (name,)}
                        if getattr(obj, "is_virtual", False):
                            return {"ok": False, "reason": "dataset %r is a virtual dataset (rejected)" % (name,)}
                        if obj.file.filename != f.filename:
                            return {"ok": False, "reason": "dataset %r resolves outside the submission file" % (name,)}
                        if obj.dtype.hasobject:
                            return {"ok": False, "reason": "dataset %r has object/vlen dtype; use the .h5ad loader" % (name,)}
                        fname = "%d.npy" % idx
                        # allow_pickle=False refuses any object array (already
                        # rejected above); the parent also loads with
                        # allow_pickle=False, so the transfer never executes a
                        # pickle.
                        np.save(os.path.join(out_dir, fname), np.asarray(obj[()]), allow_pickle=False)
                        written.append({"name": name, "file": fname})
                return {"ok": True, "written": written}
            except (OSError, ValueError, KeyError, TypeError) as exc:
                return {"ok": False, "reason": "%s: %s" % (type(exc).__name__, exc)}

    return _Reader()
'''


def _np_load_regular_nofollow(path: str) -> np.ndarray:
    """``np.load`` a worker-written ``.npy``, rejecting a symlink or FIFO at the leaf.

    The HDF5 reader runs in a uid-1000 worker and writes each dataset into the
    world-writable ``out_dir``; the ROOT parent then loads them back. A worker
    that achieves a libhdf5 memory-safety code-exec could plant
    ``os.symlink('/mcp_server/data/y_true.npy', out_dir + '/0.npy')`` and have
    root dereference it, loading the held-out truth as the prediction for a
    perfect score. ``O_NOFOLLOW`` makes the open fail atomically on a symlink leaf
    (``basename()`` already pins the path to ``out_dir``'s top level), so the
    planted link is rejected instead of followed. ``O_NONBLOCK`` makes a planted
    writerless FIFO open immediately and be rejected by the ``S_ISREG`` check
    below instead of blocking the root loader forever (this readback runs in the
    root parent with no timeout, so a hang would itself be a free veto). Mirrors
    this file's own ``load_submission_npz_or_fault`` fd-open pattern.
    (``allow_pickle=False`` is orthogonal; it blocks pickle-RCE on the array,
    not symlink traversal.)
    """
    flags = os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK | getattr(os, "O_CLOEXEC", 0)
    try:
        fd = os.open(path, flags)
    except OSError as exc:
        raise AgentFault(
            f"HDF5 readback file {path} could not be opened as a regular file "
            f"(symlink/FIFO?): {exc}"
        ) from exc
    try:
        st = os.fstat(fd)
        if not stat.S_ISREG(st.st_mode):
            raise AgentFault(
                f"HDF5 readback file {path} is not a regular file "
                f"(mode={stat.filemode(st.st_mode)})"
            )
        # Confirmed a regular file: drop O_NONBLOCK so the read behaves normally.
        fcntl.fcntl(fd, fcntl.F_SETFL, fcntl.fcntl(fd, fcntl.F_GETFL) & ~os.O_NONBLOCK)
        with os.fdopen(fd, "rb", closefd=True) as fh:
            fd = -1  # fdopen owns the descriptor
            return np.load(fh, allow_pickle=False)
    finally:
        if fd >= 0:
            os.close(fd)


def load_submission_h5_or_fault(
    path: str | Path,
    *,
    datasets: Iterable[str] | None = None,
    max_bytes: int = _DEFAULT_MAX_HDF5_BYTES,
) -> dict[str, np.ndarray]:
    """Safely read top-level datasets from a submitted HDF5 file.

    The libhdf5 parse runs in a privilege-dropped worker (never ``h5py.File`` in
    this root grader), so a crafted ``.h5`` that trips a memory-safety bug cannot
    execute with truth-read access. Rejects external/soft links, virtual
    datasets, object/vlen dtypes, and datasets resolving to another file (any of
    which could pull ``/mcp_server`` truth in); only top-level datasets are read.
    Each dataset crosses as an ``allow_pickle=False`` ``.npy`` and only the
    manifest crosses the wire, so a read up to ``max_bytes`` isn't 1 GiB-capped.

    Agent-controlled failures raise ``AgentFault`` (kept 0.0); an infra fault
    (missing drop identity, un-spawnable worker, h5py absent) propagates. Returns
    ``{name: ndarray}`` (every top-level dataset when ``datasets`` is None).
    """
    from grading.policy_runner import PolicyWorkerError, load_submitted_policy

    # Imported (not used to parse) only to fail closed with an infra error when
    # the image lacks it; the real parse happens in the dropped worker below.
    try:
        import h5py  # noqa: F401
    except Exception as exc:  # pragma: no cover - the task image ships h5py
        raise RuntimeError(
            "h5py is required to read HDF5 submissions but is not installed"
        ) from exc

    p = Path(path)
    try:
        st = os.lstat(p)
    except OSError as exc:
        raise AgentFault(f"submitted HDF5 at {p} is missing: {exc}") from exc
    if not stat.S_ISREG(st.st_mode):
        raise AgentFault(f"submitted HDF5 at {p} is not a regular file")
    if st.st_size > max_bytes:
        raise AgentFault(
            f"submitted HDF5 at {p} is {st.st_size} bytes, over the "
            f"{max_bytes}-byte limit"
        )

    names = list(datasets) if datasets is not None else None

    # Must be writable by the dropped (uid-1000) worker when the grader is root.
    out_dir = tempfile.mkdtemp(prefix="h5-submission-read-")
    try:
        if os.geteuid() == 0:
            os.chmod(out_dir, 0o777)

        # A missing drop identity / un-spawnable worker raises PolicyWorkerError
        # (infra, discarded); the read RPC is wrapped separately so a parse
        # crash/timeout is attributed to the agent.
        reader = load_submitted_policy(
            source=_H5_READER_WRAPPER,
            factory_name="load_reader",
            path=p,
            sys_path_dirs=[],
            timeout_s=_H5_READ_TIMEOUT_S,
        )
        try:
            result = reader.read(names, out_dir)
        except TimeoutError as exc:
            raise AgentFault(
                f"submitted HDF5 at {p} timed out after {_H5_READ_TIMEOUT_S:.1f}s"
            ) from exc
        except PolicyWorkerError as exc:
            # Worker died parsing the agent file: attribute to the agent (kept 0.0).
            raise AgentFault(
                f"submitted HDF5 at {p} crashed the reader: {exc}"
            ) from exc
        finally:
            reader.close()

        if not isinstance(result, dict) or not result.get("ok"):
            reason = result.get("reason") if isinstance(result, dict) else result
            raise AgentFault(f"submitted HDF5 at {p} could not be read: {reason}")

        out: dict[str, np.ndarray] = {}
        for entry in result.get("written") or []:
            # basename() pins the file to out_dir's top level, and the loader
            # opens it O_NOFOLLOW + fstat-S_ISREG so a worker-planted symlink to
            # /mcp_server truth is rejected, not followed (allow_pickle=False
            # inside also blocks a smuggled pickled object array).
            npy = os.path.join(out_dir, os.path.basename(str(entry["file"])))
            out[str(entry["name"])] = _np_load_regular_nofollow(npy)
        return out
    finally:
        shutil.rmtree(out_dir, ignore_errors=True)


# LITERAL source wrapper executed INSIDE the privilege-dropped policy worker
# (source mode). Reading whole-object / categorical / sparse AnnData needs the
# anndata library's structure handling rather than a dense-dataset read; doing
# it in the dropped worker keeps the parse off the root grader process. The
# in-worker ``read_h5ad`` + ``write_h5ad`` round-trip yields a self-contained
# copy (no external links / virtual datasets), and the uid-1000 worker cannot
# reach ``/mcp_server``, so a truth-pointing submission fails the read instead
# of baking truth into the clean copy.
_H5AD_SANITIZE_WRAPPER = '''
def load_reader():
    src_path = __file__

    class _Sanitizer:
        def sanitize(self, out_path):
            try:
                import anndata
            except ImportError as exc:
                return {"ok": False, "infra": True, "reason": "anndata is not installed: %s" % (exc,)}
            try:
                anndata.read_h5ad(src_path).write_h5ad(out_path)
                return {"ok": True}
            except (OSError, ValueError, KeyError, TypeError) as exc:
                return {"ok": False, "reason": "%s: %s" % (type(exc).__name__, exc)}

    return _Sanitizer()
'''


def load_submission_h5ad_or_fault(
    path: str | Path,
    *,
    out_path: str | Path,
    max_bytes: int = _DEFAULT_MAX_HDF5_BYTES,
) -> Path:
    """Round-trip a submitted ``.h5ad`` through anndata IN the privilege-dropped
    worker into a self-contained clean copy at ``out_path``.

    Use this (not :func:`load_submission_h5_or_fault`) for categorical / sparse /
    whole-object AnnData. Drops external links / virtual datasets, and the
    unprivileged worker cannot reach ``/mcp_server``, so a truth-pointing
    submission fails the read. Agent-controlled failures raise ``AgentFault``
    (kept 0.0); a missing ``anndata`` or dead worker propagates as infra.
    ``out_path`` must be writable by the (possibly dropped) worker. Returns it.
    """
    from grading.policy_runner import PolicyWorkerError, load_submitted_policy

    p = Path(path)
    out = Path(out_path)
    try:
        st = os.lstat(p)
    except OSError as exc:
        raise AgentFault(f"submitted .h5ad at {p} is missing: {exc}") from exc
    if not stat.S_ISREG(st.st_mode):
        raise AgentFault(f"submitted .h5ad at {p} is not a regular file")
    if st.st_size > max_bytes:
        raise AgentFault(
            f"submitted .h5ad at {p} is {st.st_size} bytes, over the "
            f"{max_bytes}-byte limit"
        )

    sanitizer = load_submitted_policy(
        source=_H5AD_SANITIZE_WRAPPER,
        factory_name="load_reader",
        path=p,
        sys_path_dirs=[],
        timeout_s=_H5_READ_TIMEOUT_S,
    )
    try:
        result = sanitizer.sanitize(str(out))
    except TimeoutError as exc:
        raise AgentFault(
            f"submitted .h5ad at {p} timed out after {_H5_READ_TIMEOUT_S:.1f}s"
        ) from exc
    except PolicyWorkerError as exc:
        raise AgentFault(
            f"submitted .h5ad at {p} crashed the sanitizer: {exc}"
        ) from exc
    finally:
        sanitizer.close()

    if not isinstance(result, dict) or not result.get("ok"):
        reason = result.get("reason") if isinstance(result, dict) else result
        if isinstance(result, dict) and result.get("infra"):
            raise RuntimeError(f"could not sanitize submitted .h5ad: {reason}")
        raise AgentFault(f"submitted .h5ad at {p} could not be sanitized: {reason}")
    return out


# ── Internal: JSON path parser ────────────────────────────────────


def _parse_filter_expr(expr: str) -> tuple[str, str, str] | None:
    op = "==" if "==" in expr else "="
    if op not in expr:
        return None
    field, value = expr.split(op, 1)
    field = field.strip()
    value = value.strip()
    if not field or not value:
        return None
    if len(value) >= 2 and value[0] == value[-1] and value[0] in ("'", '"'):
        value = value[1:-1]
    return ("filter_eq", field, value)


def _parse_json_path(path: str):
    path = path.strip()
    if not path:
        return []

    parts: list = []
    i = 1 if path.startswith("$") else 0
    n = len(path)

    while i < n:
        char = path[i]
        if char == ".":
            i += 1
            start = i
            while i < n and path[i] not in ".[":
                i += 1
            if i > start:
                parts.append(path[start:i])
            continue

        if char == "[":
            i += 1
            while i < n and path[i].isspace():
                i += 1
            if i >= n:
                return None

            if path[i] == "?":
                i += 1
                start = i
                while i < n and path[i] != "]":
                    i += 1
                if i >= n:
                    return None
                filter_part = _parse_filter_expr(path[start:i])
                if filter_part is None:
                    return None
                i += 1
                parts.append(filter_part)
                continue

            if path[i] in ("'", '"'):
                quote = path[i]
                i += 1
                buf: list[str] = []
                while i < n:
                    ch = path[i]
                    if ch == "\\":
                        i += 1
                        if i >= n:
                            return None
                        buf.append(path[i])
                        i += 1
                        continue
                    if ch == quote:
                        i += 1
                        break
                    buf.append(ch)
                    i += 1
                else:
                    return None
                while i < n and path[i].isspace():
                    i += 1
                if i >= n or path[i] != "]":
                    return None
                i += 1
                parts.append("".join(buf))
                continue

            start = i
            while i < n and path[i] != "]":
                i += 1
            if i >= n:
                return None
            part = path[start:i].strip()
            i += 1
            if part:
                parts.append(part)
            continue

        start = i
        while i < n and path[i] not in ".[":
            i += 1
        if i > start:
            parts.append(path[start:i])

    return parts


__all__ = [
    "abs_error",
    "exact_match",
    "file_contains",
    "file_exists",
    "jaccard",
    "json_path",
    "kendall_tau",
    "load_json",
    "load_submission_h5_or_fault",
    "load_submission_h5ad_or_fault",
    "load_submission_npz_or_fault",
    "load_submission_or_fault",
    "load_submitted_model",
    "load_submitted_policy",
    "regex_search",
    "require_regular_file",
    "run_model_module",
    "run_policy",
    "run_submitted_executable",
    "transcript_contains",
    "world_integrity",
]
