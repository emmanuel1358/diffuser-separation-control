"""Deterministic predicate helpers returning bool / float in [0, 1].

Us inside ``RubricTask.evaluate`` / continuous ``compute_score`` criteria.
Legacy ``RubricBuilder`` criteria also consume these helpers.
"""

from __future__ import annotations

import bz2
import contextlib
import gzip
import io
import json
import os
import re
import shutil
import signal
import stat
import subprocess
import sys
import tempfile
import threading
import zipfile
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import numpy as np

from grading.faults import AgentFault, GraderFault
from grading.secure_io import (
    open_regular_file,
    persistent_regular_file_snapshot,
    read_regular_bytes,
    regular_file_snapshot,
)

_DEFAULT_MAX_SUBMISSION_BYTES = 64 * 1024 * 1024
_DEFAULT_MAX_CSV_UNCOMPRESSED_BYTES = 256 * 1024 * 1024
# stdout cap for a captured agent executable: a flood past this is killed and
# raised as an AgentFault (kept 0.0), so the child cannot OOM the grader.
_DEFAULT_MAX_EXECUTABLE_OUTPUT_BYTES = 16 * 1024 * 1024
# UNCOMPRESSED cap for an agent .npz: the file-size cap bounds only the
# compressed bytes, so a small archive can expand to many GiB on member access.
# Generous enough for honest submissions, tight enough to bound a bomb well under
# grader OOM.
_DEFAULT_MAX_NPZ_UNCOMPRESSED_BYTES = 512 * 1024 * 1024
_DEFAULT_MAX_HDF5_BYTES = 2 * 1024 * 1024 * 1024
_DEFAULT_MAX_HDF5_DATASETS = 256
_DEFAULT_MAX_HDF5_DATASET_BYTES = 256 * 1024 * 1024
_DEFAULT_MAX_HDF5_LOGICAL_BYTES = 512 * 1024 * 1024
# Execution timeout for the privilege-dropped HDF5 reader worker. Bounds a
# crafted file that tries to hang or OOM the grader during the libhdf5 parse;
# the read itself is one RPC call, so this is the per-parse budget.
_H5_READ_TIMEOUT_S = 120.0


@dataclass(frozen=True)
class SolverResult:
    """Bounded trusted-solver outcome; process failures never escape untyped."""

    returncode: int
    output: str
    timed_out: bool = False
    output_exceeded: bool = False

    @property
    def ok(self) -> bool:
        return self.returncode == 0 and not self.timed_out and not self.output_exceeded


def file_exists(path: str | Path, *, non_empty: bool = False) -> bool:
    """True iff ``path`` is a symlink-free regular file."""

    try:
        if not non_empty:
            with open_regular_file(path, max_bytes=None, allow_empty=True):
                return True
        data = read_regular_bytes(
            path,
            max_bytes=_DEFAULT_MAX_SUBMISSION_BYTES,
            allow_empty=True,
        )
    except OSError:
        return False
    return bool(data.strip())


def file_contains(
    path: str | Path,
    needle: str,
    *,
    case_sensitive: bool = True,
    encoding: str = "utf-8",
) -> bool:
    """True iff one descriptor-pinned regular file contains ``needle``."""

    try:
        text = read_regular_bytes(
            path,
            max_bytes=_DEFAULT_MAX_SUBMISSION_BYTES,
            allow_empty=True,
        ).decode(encoding=encoding, errors="replace")
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
    """Load descriptor-pinned JSON, returning ``None`` on missing/invalid."""

    try:
        raw = read_regular_bytes(
            path,
            max_bytes=_DEFAULT_MAX_SUBMISSION_BYTES,
            allow_empty=True,
        )
        return json.loads(raw.decode("utf-8"))
    except (OSError, json.JSONDecodeError, UnicodeDecodeError):
        return None


@contextlib.contextmanager
def open_submission_file_or_fault(
    path: os.PathLike | str,
    *,
    max_bytes: int = _DEFAULT_MAX_SUBMISSION_BYTES,
    allow_empty: bool = False,
):
    """Yield immutable bytes from a component-safe submission-file snapshot."""

    try:
        raw = read_regular_bytes(
            path,
            max_bytes=max_bytes,
            allow_empty=allow_empty,
        )
    except OSError as exc:
        raise AgentFault(
            f"submission at {path} could not be read as a stable regular file: {exc}"
        ) from exc
    with io.BytesIO(raw) as handle:
        yield handle


def _read_capped_stream(handle: Any, *, max_bytes: int, label: str) -> bytes:
    raw = handle.read(max_bytes + 1)
    if len(raw) > max_bytes:
        raise AgentFault(f"{label} expands beyond the {max_bytes}-byte limit")
    return raw


def _csv_compression_method(
    path: os.PathLike | str,
    kwargs: dict[str, Any],
) -> str | None:
    compression = kwargs.get("compression", "infer")
    if isinstance(compression, dict):
        if set(compression) != {"method"}:
            raise ValueError(
                "compressed CSV options support only {'method': ...}; "
                "decompress the submission before grading for custom options"
            )
        compression = compression["method"]
    if compression in (None, False):
        return None
    if compression == "infer":
        suffix = Path(path).suffix.lower()
        if suffix in {".lzma", ".xz", ".zip", ".zst", ".zstd"}:
            raise ValueError(
                f"{suffix} CSV compression is unsupported; use gzip, bz2, "
                "or an uncompressed CSV"
            )
        return {
            ".bz2": "bz2",
            ".gz": "gzip",
        }.get(suffix)
    method = str(compression).lower()
    if method not in {"bz2", "gzip"}:
        raise ValueError(
            f"unsupported compressed CSV method {compression!r}; "
            "supported methods are gzip and bz2"
        )
    return method


def _read_bounded_compressed_csv(
    snapshot: Path,
    *,
    method: str,
    max_uncompressed_bytes: int,
) -> bytes:
    if method == "gzip":
        with gzip.open(snapshot, "rb") as handle:
            return _read_capped_stream(
                handle,
                max_bytes=max_uncompressed_bytes,
                label="gzip CSV submission",
            )
    if method == "bz2":
        with bz2.open(snapshot, "rb") as handle:
            return _read_capped_stream(
                handle,
                max_bytes=max_uncompressed_bytes,
                label="bz2 CSV submission",
            )
    raise AssertionError(f"unexpected CSV compression method: {method}")


def load_submission_or_fault(
    path: os.PathLike | str,
    *,
    required_columns: Iterable[str] | None = None,
    n_rows: int | None = None,
    numeric_columns: Iterable[str] | None = None,
    unique_key_column: str | None = None,
    max_bytes: int = _DEFAULT_MAX_SUBMISSION_BYTES,
    max_uncompressed_bytes: int = _DEFAULT_MAX_CSV_UNCOMPRESSED_BYTES,
    read_csv_kwargs: dict[str, Any] | None = None,
    extra_columns: Literal["reject", "drop", "preserve"] | None = None,
    allow_extra_columns: bool | None = None,
):
    """Read an agent CSV submission, raising ``AgentFault`` for bad outputs.

    This closes common free-veto paths: symlinks to hidden data, FIFOs that
    hang grading, oversized files that OOM the verifier, malformed CSV, missing
    columns, non-finite values, duplicate merge keys, and unexpected columns.

    ``extra_columns="drop"`` is the safe tolerant mode: undeclared
    agent-controlled fields are removed before the frame reaches trusted joins.
    ``allow_extra_columns`` remains as a compatibility alias for
    ``"preserve"``/``"reject"`` and cannot be combined with ``extra_columns``.
    """
    import pandas as pd

    if min(max_bytes, max_uncompressed_bytes) <= 0:
        raise ValueError("CSV byte limits must be positive")
    if allow_extra_columns is not None:
        if extra_columns is not None:
            raise ValueError(
                "pass either extra_columns or allow_extra_columns, not both"
            )
        extra_policy = "preserve" if allow_extra_columns else "reject"
    else:
        extra_policy = extra_columns or "reject"
    if extra_policy not in {"reject", "drop", "preserve"}:
        raise ValueError("extra_columns must be 'reject', 'drop', or 'preserve'")
    csv_kwargs = dict(read_csv_kwargs or {})
    try:
        with regular_file_snapshot(path, max_bytes=max_bytes) as snapshot:
            compression = _csv_compression_method(path, csv_kwargs)
            if compression is None:
                df = pd.read_csv(snapshot, **csv_kwargs)
            else:
                raw = _read_bounded_compressed_csv(
                    snapshot,
                    method=compression,
                    max_uncompressed_bytes=max_uncompressed_bytes,
                )
                csv_kwargs["compression"] = None
                csv_kwargs.pop("memory_map", None)
                df = pd.read_csv(io.BytesIO(raw), **csv_kwargs)
    except FileNotFoundError as exc:
        raise AgentFault(f"submission not found at {path}") from exc
    except OSError as exc:
        raise AgentFault(
            f"submission at {path} is not a regular file or changed while "
            f"being read as CSV: {exc}"
        ) from exc
    except Exception as exc:
        raise AgentFault(
            f"submission at {path} is not a readable CSV: "
            f"{type(exc).__name__}: {exc}"
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

    if required is not None or numeric is not None or unique_key_column is not None:
        expected_order: list[str] = []
        if required is not None:
            expected_order.extend(required)
        if numeric is not None:
            expected_order.extend(numeric)
        if unique_key_column is not None:
            expected_order.append(unique_key_column)
        expected_order = list(dict.fromkeys(expected_order))
        expected = set(expected_order)
        extra = [column for column in df.columns if column not in expected]
        if extra and extra_policy == "reject":
            raise AgentFault(
                f"submission has unexpected column(s) {extra}; expected "
                f"exactly {sorted(expected)}. Pass extra_columns='drop' to "
                "ignore undeclared fields safely."
            )
        if extra_policy == "drop":
            df = df[expected_order].copy()

    return df


def join_submission_to_truth_or_fault(
    submission: Any,
    truth: Any,
    *,
    key_column: str,
    prediction_columns: Iterable[str],
    truth_columns: Iterable[str],
    truth_suffix: str = "_truth",
):
    """Safely join declared prediction fields to trusted truth by one key.

    Both frames are projected before the merge, so an agent cannot introduce a
    column that collides with a trusted field and changes suffix resolution.
    Candidate key omissions, additions, and duplicates are ``AgentFault``;
    malformed trusted truth is a ``GraderFault``.
    """
    import pandas as pd

    predictions = list(dict.fromkeys(str(c) for c in prediction_columns))
    trusted = list(dict.fromkeys(str(c) for c in truth_columns))
    if not key_column or not predictions or not trusted:
        raise ValueError(
            "key_column, prediction_columns, and truth_columns must be non-empty"
        )
    if not truth_suffix:
        raise ValueError("truth_suffix must be non-empty")

    missing_submission = [
        column
        for column in (key_column, *predictions)
        if column not in submission.columns
    ]
    if missing_submission:
        raise AgentFault(f"submission missing join column(s): {missing_submission}")
    missing_truth = [
        column for column in (key_column, *trusted) if column not in truth.columns
    ]
    if missing_truth:
        raise GraderFault(f"trusted truth missing join column(s): {missing_truth}")

    candidate = submission[[key_column, *predictions]].copy()
    expected = truth[[key_column, *trusted]].copy()
    try:
        pd.util.hash_pandas_object(candidate[key_column], index=False)
    except (TypeError, ValueError) as exc:
        raise AgentFault(
            f"submission join key {key_column!r} is not safely hashable: {exc}"
        ) from exc
    try:
        pd.util.hash_pandas_object(expected[key_column], index=False)
    except (TypeError, ValueError) as exc:
        raise GraderFault(
            f"trusted truth join key {key_column!r} is not safely hashable: {exc}"
        ) from exc
    duplicate_candidates = int(candidate[key_column].duplicated().sum())
    if duplicate_candidates:
        raise AgentFault(
            f"submission has {duplicate_candidates} duplicate join key(s) in "
            f"{key_column!r}"
        )
    duplicate_truth = int(expected[key_column].duplicated().sum())
    if duplicate_truth:
        raise GraderFault(
            f"trusted truth has {duplicate_truth} duplicate join key(s) in "
            f"{key_column!r}"
        )

    try:
        merged = pd.merge(
            candidate,
            expected,
            on=key_column,
            how="outer",
            suffixes=("", truth_suffix),
            validate="one_to_one",
            indicator=True,
            sort=False,
        )
    except (TypeError, ValueError) as exc:
        raise AgentFault(f"submission prediction join failed: {exc}") from exc
    except Exception as exc:
        raise GraderFault(f"trusted prediction join failed: {exc}") from exc

    missing_candidate = int((merged["_merge"] == "right_only").sum())
    unexpected_candidate = int((merged["_merge"] == "left_only").sum())
    if missing_candidate or unexpected_candidate:
        raise AgentFault(
            "submission join keys do not match trusted truth: "
            f"{missing_candidate} missing, {unexpected_candidate} unexpected"
        )
    return merged.drop(columns="_merge")


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

    Rejects symlinks in every path component, FIFO/dir/device/oversized files,
    and defaults ``allow_pickle=False`` so a crafted array cannot execute
    ``__reduce__`` in the root grader. Returns an ``ndarray`` for a ``.npy``
    submission, or a plain ``dict`` ``{name: ndarray}`` for a ``.npz`` -- whose
    members are materialized eagerly here (inside the AgentFault wrapper), so an
    object-array member is a kept-0.0 fault rather than a bare ``ValueError`` at
    ``data[name]`` access time (which the runtime would discard as
    env_internal_failure -- a void-veto).
    """
    p = Path(path)
    if allow_pickle:
        raise ValueError(
            "allow_pickle=True is forbidden for agent submissions; "
            "use load_submitted_model() for sandboxed pickle artifacts"
        )
    try:
        raw = read_regular_bytes(p, max_bytes=max_bytes)
    except FileNotFoundError as exc:
        raise AgentFault(f"submission not found at {p}") from exc
    except OSError as exc:
        raise AgentFault(
            f"submission at {p} could not be read as a stable regular file: {exc}"
        ) from exc

    # Bound the UNCOMPRESSED expansion of a .npz before materializing members, so
    # a decompression bomb under the compressed cap cannot OOM the grader.
    _reject_npz_decompression_bomb(raw, p, max_uncompressed_bytes)

    try:
        obj = np.load(io.BytesIO(raw), allow_pickle=False)
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
    """Return an immutable snapshot of a bounded regular submission file.

    The snapshot is rooted in a grader-owned directory and retained until this
    grader process exits, so a later pathname-based parser cannot be redirected
    by replacing the original file or any parent directory.
    """

    try:
        return persistent_regular_file_snapshot(
            path,
            max_bytes=max_bytes,
            allow_empty=True,
        )
    except FileNotFoundError as exc:
        raise AgentFault(f"submission not found at {path}") from exc
    except OSError as exc:
        raise AgentFault(
            f"submission at {path} could not be snapshotted as a stable "
            f"regular file: {exc}"
        ) from exc


_POLICY_FILENAMES = ("policy.py", "submission.py", "agent.py")


def _resolve_policy_path(policy: str | Path) -> Path:
    """Resolve ``policy`` (a file or a dir/workspace) to a concrete policy file.

    Accepts the submitted policy file directly, or a workspace/output directory
    in which a conventional policy file (``policy.py`` etc.) is located.
    """
    p = Path(policy)
    for candidate in (p, *(p / name for name in _POLICY_FILENAMES)):
        try:
            with open_regular_file(
                candidate,
                max_bytes=_DEFAULT_MAX_SUBMISSION_BYTES,
                allow_empty=True,
            ):
                return candidate
        except OSError:
            continue
    raise FileNotFoundError(f"no submitted policy found at {policy}")


def run_policy(
    policy: str | Path,
    *,
    timeout_s: float = 5.0,
    first_call_timeout_s: float | None = None,
    cwd: str | Path | None = None,
    unshare_ipc: bool = True,
    submitted_snapshot: str | Path | None = None,
):
    """Return a hardened worker for a submitted policy — the single approved way to run agent code.

    Wraps :class:`grading.policy_runner.PolicyWorker`: the policy runs in a
    non-root subprocess when the grader is root, and its return value travels
    over a dedicated fd so the score is NEVER parsed from stdout. Use as a
    context manager and read the score from the return value, not stdout.
    ``unshare_ipc`` (default True) gives each worker a private IPC namespace.
    """
    from grading.policy_runner import PolicyWorker

    resolved_policy = (
        Path(policy) if submitted_snapshot is not None else _resolve_policy_path(policy)
    )
    return PolicyWorker(
        resolved_policy,
        timeout_s=timeout_s,
        first_call_timeout_s=first_call_timeout_s,
        cwd=Path(cwd) if cwd is not None else None,
        unshare_ipc=unshare_ipc,
        submitted_snapshot=submitted_snapshot,
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
_AGENT_USER_SITE_SNIPPET = """
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
"""

_PICKLE_WRAPPER = (
    """
def load_policy():
    import os, sys, pickle
    model_path = __file__
    origin_path = __submission_origin__
"""
    + _AGENT_USER_SITE_SNIPPET
    + """
    data_dir = os.path.dirname(origin_path) or "."
    if data_dir not in sys.path:
        sys.path.insert(0, data_dir)
    with open(model_path, "rb") as fh:
        return pickle.load(fh)
"""
)

_JOBLIB_WRAPPER = (
    """
def load_policy():
    import os, sys, joblib
    model_path = __file__
    origin_path = __submission_origin__
"""
    + _AGENT_USER_SITE_SNIPPET
    + """
    data_dir = os.path.dirname(origin_path) or "."
    if data_dir not in sys.path:
        sys.path.insert(0, data_dir)
    return joblib.load(model_path)
"""
)

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
            f"deserializer must be one of {sorted(_WRAPPERS)}; got {deserializer!r}"
        )
    original = Path(path)
    try:
        snapshot = regular_file_snapshot(
            original,
            max_bytes=max_bytes,
            allow_empty=True,
        )
        pinned = snapshot.__enter__()
    except FileNotFoundError as exc:
        raise AgentFault(f"Missing submitted model at {original}") from exc
    except OSError as exc:
        raise AgentFault(
            f"submitted model at {original} is not a stable regular file: {exc}"
        ) from exc

    try:
        # Source mode: the immutable snapshot is module.__file__ for the wrapper.
        # The submitted model is fully deserialized during the worker handshake,
        # before the snapshot is removed.
        return load_submitted_policy(
            source=_WRAPPERS[deserializer],
            factory_name="load_policy",
            path=pinned,
            source_origin_path=original,
            sys_path_dirs=[],
            timeout_s=_MODEL_LOAD_TIMEOUT_S,
            first_call_timeout_s=_MODEL_LOAD_TIMEOUT_S,
        )
    finally:
        snapshot.__exit__(None, None, None)


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


def resolve_submitted_process_identity() -> tuple[int, int, str, str]:
    """Resolve the configured non-root identity for submitted executables."""
    from grading.policy_runner import PolicyWorkerError, _agent_identity

    try:
        return _agent_identity()
    except PolicyWorkerError as exc:
        raise GraderFault(
            f"cannot establish submitted-process identity: {exc}"
        ) from exc


def _submitted_process_preexec(
    uid: int,
    gid: int,
    cwd_fd: int | None,
    ipc_status_fd: int | None,
):
    """Isolate IPC, enter a pinned cwd, and irreversibly drop privileges."""
    from grading.policy_runner import _agent_preexec
    from grading.runtime_hardening import child_memory_limit_bytes

    try:
        memory_limit_bytes = child_memory_limit_bytes(
            env_keys=(
                "RUBRIC_POLICY_MEMORY_LIMIT_BYTES",
                "RUBRIC_AGENT_MEMORY_LIMIT_BYTES",
            )
        )
    except ValueError as exc:
        raise GraderFault(f"invalid submitted-process memory limit: {exc}") from exc

    return _agent_preexec(
        uid,
        gid,
        ipc_status_fd=ipc_status_fd,
        cwd_fd=cwd_fd,
        memory_limit_bytes=memory_limit_bytes,
    )


def _submitted_process_spawn_config(
    identity: tuple[int, int, str, str],
    *,
    cwd_fd: int | None,
    ipc_status_fd: int | None = None,
) -> dict[str, Any]:
    """Build fail-closed Popen kwargs for the submitted-process boundary."""
    if os.geteuid() != 0:
        raise GraderFault(
            "cannot establish submitted-process identity separation: "
            "the verifier must run as root"
        )
    uid, gid, _home, _name = identity
    if uid <= 0 or gid <= 0:
        raise GraderFault(
            "submitted-process identity must use a configured non-root uid/gid"
        )
    pass_fds = tuple(
        descriptor for descriptor in (cwd_fd, ipc_status_fd) if descriptor is not None
    )
    return {
        "preexec_fn": _submitted_process_preexec(
            uid,
            gid,
            cwd_fd,
            ipc_status_fd,
        ),
        "pass_fds": pass_fds,
    }


def _spawn_submitted_process(
    cmd: list[str],
    *,
    identity: tuple[int, int, str, str],
    cwd_fd: int | None,
    **popen_kwargs: Any,
) -> subprocess.Popen:
    """Spawn one submitted process and report its best-effort IPC boundary."""
    try:
        ipc_status_r, ipc_status_w = os.pipe()
    except OSError as exc:
        raise GraderFault(
            f"cannot establish submitted-process IPC status channel: {exc}"
        ) from exc

    try:
        spawn_config = _submitted_process_spawn_config(
            identity,
            cwd_fd=cwd_fd,
            ipc_status_fd=ipc_status_w,
        )
        proc = subprocess.Popen(cmd, **popen_kwargs, **spawn_config)
    except BaseException:
        with contextlib.suppress(OSError):
            os.close(ipc_status_r)
        with contextlib.suppress(OSError):
            os.close(ipc_status_w)
        raise

    with contextlib.suppress(OSError):
        os.close(ipc_status_w)
    from grading.policy_runner import _warn_if_ipc_unisolated

    _warn_if_ipc_unisolated(
        ipc_status_r,
        boundary="submitted executable",
    )
    return proc


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


def _kill_process_group(proc: subprocess.Popen) -> None:
    try:
        os.killpg(proc.pid, signal.SIGKILL)
    except OSError:
        with contextlib.suppress(OSError):
            proc.kill()


def _kill_and_reap_process_group(proc: subprocess.Popen) -> None:
    _kill_process_group(proc)
    try:
        proc.wait(timeout=5.0)
    except subprocess.TimeoutExpired:
        with contextlib.suppress(OSError):
            proc.kill()
        proc.wait()


def run_trusted_solver(
    cmd: list[str],
    *,
    cwd: str | Path | None = None,
    env: dict[str, str] | None = None,
    timeout_s: float = 120.0,
    max_output_bytes: int = _DEFAULT_MAX_EXECUTABLE_OUTPUT_BYTES,
) -> SolverResult:
    """Run a trusted solver with a process-group timeout and hard output cap.

    A missing executable or process-launch error is a ``GraderFault``. Solver
    nonzero exits, timeouts, and output floods are returned as typed outcomes so
    the declarative rubric can assign criterion zero without an uncaught
    ``TimeoutExpired`` voiding the rollout.
    """
    if not cmd or not all(isinstance(part, str) and part for part in cmd):
        raise GraderFault("trusted solver command must be a non-empty string list")
    if timeout_s <= 0.0 or max_output_bytes < 1:
        raise GraderFault("trusted solver timeout/output limits must be positive")
    try:
        proc = subprocess.Popen(
            cmd,
            cwd=str(cwd) if cwd is not None else None,
            env=env,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
    except OSError as exc:
        raise GraderFault(
            f"could not launch trusted solver {cmd[0]!r}: {type(exc).__name__}: {exc}"
        ) from exc

    captured = bytearray()
    exceeded = threading.Event()

    def _drain() -> None:
        assert proc.stdout is not None
        try:
            while True:
                chunk = proc.stdout.read1(65536)
                if not chunk:
                    break
                remaining = max_output_bytes + 1 - len(captured)
                if remaining > 0:
                    captured.extend(chunk[:remaining])
                if len(captured) > max_output_bytes:
                    exceeded.set()
                    with contextlib.suppress(OSError):
                        os.killpg(proc.pid, signal.SIGKILL)
                    break
        except (OSError, ValueError):
            pass

    reader = threading.Thread(target=_drain, daemon=True)
    reader.start()
    timed_out = False
    try:
        proc.wait(timeout=timeout_s)
    except subprocess.TimeoutExpired:
        timed_out = True
        with contextlib.suppress(OSError):
            os.killpg(proc.pid, signal.SIGKILL)
        proc.wait()
    finally:
        reader.join(timeout=5.0)
        with contextlib.suppress(OSError):
            if proc.stdout is not None:
                proc.stdout.close()

    return SolverResult(
        returncode=int(proc.returncode if proc.returncode is not None else -1),
        output=bytes(captured[:max_output_bytes]).decode("utf-8", errors="replace"),
        timed_out=timed_out,
        output_exceeded=exceeded.is_set(),
    )


def run_submitted_executable(
    cmd: list[str],
    *,
    stdin_bytes: bytes | None = None,
    cwd: str | Path | None = None,
    cwd_fd: int | None = None,
    env: dict[str, str] | None = None,
    env_passthrough: bool = False,
    timeout_s: float | None = None,
    max_output_bytes: int = _DEFAULT_MAX_EXECUTABLE_OUTPUT_BYTES,
    streaming: bool = False,
) -> subprocess.CompletedProcess:
    """Spawn an agent executable as the configured non-root identity.

    ``cwd_fd`` pins execution to an already-open directory and is mutually
    exclusive with ``cwd``. The verifier must be root so supplementary groups
    can be cleared and the requested uid/gid separation can be established.
    Each process also attempts a fresh IPC namespace before dropping
    privileges, preventing a rollout from relaying state through persistent
    SysV shared memory, semaphores, or message queues. Runtimes without the
    required namespace capability warn and continue. Stdout remains isolated
    from the RUBRIC_SCORE= parser (the OS-process companion to
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

    if cwd is not None and cwd_fd is not None:
        raise ValueError("cwd and cwd_fd are mutually exclusive")
    if cwd_fd is not None:
        try:
            cwd_info = os.fstat(cwd_fd)
        except OSError as exc:
            raise GraderFault(f"submitted executable cwd descriptor is invalid: {exc}")
        if not stat.S_ISDIR(cwd_info.st_mode):
            raise GraderFault("submitted executable cwd descriptor is not a directory")
    cwd_str = str(cwd) if cwd is not None else None
    identity = resolve_submitted_process_identity()

    if env is not None:
        child_env: dict[str, str] | None = env
    elif env_passthrough:
        child_env = None
    else:
        child_env = _build_sanitized_env()
    if child_env is not None:
        child_env = dict(child_env)
        _uid, _gid, home, name = identity
        child_env.setdefault("HOME", home)
        child_env["USER"] = name
        child_env["LOGNAME"] = name

    if not streaming:
        # Bounded capture: stream stdout through a capped reader that kills the
        # child the moment its output crosses max_output_bytes (a true memory
        # bound, not a post-capture check), discard stderr at the kernel, and feed
        # stdin from a writer thread so a large stdin can't deadlock against a
        # child that fills the stdout pipe. An agent-caused timeout / over-cap is
        # converted to AgentFault (kept 0.0) instead of escaping as
        # TimeoutExpired / MemoryError -> a DISCARDED (free-veto) rollout.
        try:
            proc = _spawn_submitted_process(
                cmd,
                identity=identity,
                cwd_fd=cwd_fd,
                stdin=(
                    subprocess.PIPE if stdin_bytes is not None else subprocess.DEVNULL
                ),
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL,
                cwd=cwd_str,
                env=child_env,
                start_new_session=True,
            )
        except subprocess.SubprocessError as exc:
            raise GraderFault(
                f"could not establish submitted-process boundary: {exc}"
            ) from exc
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
                        _kill_process_group(proc)
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
            _kill_and_reap_process_group(proc)
            raise AgentFault(
                f"submitted executable timed out after {timeout_s:.1f}s"
            ) from exc
        else:
            # A successful leader may have daemonized descendants that still
            # hold the disposable workspace or stdout pipe. The candidate
            # boundary owns the whole session, so no process may outlive it.
            _kill_process_group(proc)
        finally:
            reader.join(timeout=5.0)
            writer.join(timeout=5.0)
            with contextlib.suppress(OSError):
                if proc.stdout is not None:
                    proc.stdout.close()
        if over_cap.is_set():
            _kill_and_reap_process_group(proc)
            raise AgentFault(
                f"submitted executable produced more than {max_output_bytes} "
                f"bytes of stdout"
            )
        return subprocess.CompletedProcess(
            args=cmd,
            returncode=proc.returncode,
            stdout=bytes(captured),
            stderr=b"",
        )

    # Streaming: pump lines to sys.stderr (never the score-parsed sys.stdout),
    # dropping RUBRIC_SCORE= lines; utf-8/replace so a bad byte can't break the pump.
    try:
        proc = _spawn_submitted_process(
            cmd,
            identity=identity,
            cwd_fd=cwd_fd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            cwd=cwd_str,
            env=child_env,
            encoding="utf-8",
            errors="replace",
            bufsize=1,
            start_new_session=True,
        )
    except subprocess.SubprocessError as exc:
        raise GraderFault(
            f"could not establish submitted-process boundary: {exc}"
        ) from exc
    assert proc.stdout is not None
    for line in proc.stdout:
        if "RUBRIC_SCORE=" in line:
            continue
        sys.stderr.write(line)
        sys.stderr.flush()
    proc.wait()
    return subprocess.CompletedProcess(
        args=cmd,
        returncode=proc.returncode,
        stdout=b"",
        stderr=b"",
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
_H5_READER_WRAPPER = """
def load_reader():
    import os
    import h5py
    import numpy as np

    src_path = __file__

    class _Reader:
        def read(self, names, out_dir, max_datasets, max_dataset_bytes, max_total_bytes):
            try:
                written = []
                with h5py.File(src_path, "r", locking=False) as f:
                    if names is None:
                        if len(f) > max_datasets:
                            return {"ok": False, "reason": "file declares %d top-level datasets, over limit %d" % (len(f), max_datasets)}
                        names = list(f.keys())
                    if len(names) > max_datasets:
                        return {"ok": False, "reason": "requested %d datasets, over limit %d" % (len(names), max_datasets)}
                    if len(names) != len(set(names)):
                        return {"ok": False, "reason": "requested dataset names must be unique"}
                    total_bytes = 0
                    for idx, name in enumerate(names):
                        if not isinstance(name, str):
                            return {"ok": False, "reason": "dataset names must be strings"}
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
                            return {"ok": False, "reason": "dataset %r has object/vlen dtype; only bounded primitive HDF5 datasets are supported" % (name,)}
                        logical_bytes = int(obj.size) * int(obj.dtype.itemsize)
                        if logical_bytes > max_dataset_bytes:
                            return {"ok": False, "reason": "dataset %r expands to %d bytes, over per-dataset limit %d" % (name, logical_bytes, max_dataset_bytes)}
                        total_bytes += logical_bytes
                        if total_bytes > max_total_bytes:
                            return {"ok": False, "reason": "requested datasets expand to %d bytes, over aggregate limit %d" % (total_bytes, max_total_bytes)}
                        fname = "%d.npy" % idx
                        # allow_pickle=False refuses any object array (already
                        # rejected above); the parent also loads with
                        # allow_pickle=False, so the transfer never executes a
                        # pickle.
                        np.save(os.path.join(out_dir, fname), np.asarray(obj[()]), allow_pickle=False)
                        written.append({"name": name, "file": fname, "bytes": logical_bytes})
                return {"ok": True, "written": written}
            except (OSError, ValueError, KeyError, TypeError) as exc:
                return {"ok": False, "reason": "%s: %s" % (type(exc).__name__, exc)}

    return _Reader()
"""


def _np_load_regular_nofollow(
    path: str,
    *,
    max_bytes: int = _DEFAULT_MAX_HDF5_DATASET_BYTES,
) -> np.ndarray:
    """Load worker-written ``.npy`` through a component-safe descriptor.

    The HDF5 reader runs in a uid-1000 worker and writes each dataset into the
    world-writable ``out_dir``; the ROOT parent then loads them back. A worker
    that achieves a libhdf5 memory-safety code-exec could plant
    ``os.symlink('/mcp_server/data/y_true.npy', out_dir + '/0.npy')`` and have
    root dereference it, loading the held-out truth as the prediction for a
    perfect score. The component-safe open rejects symlinks in the leaf and all
    parent directories. ``O_NONBLOCK`` makes a planted writerless FIFO open
    immediately and be rejected instead of blocking the root loader forever.
    (``allow_pickle=False`` is orthogonal; it blocks pickle-RCE on the array,
    not symlink traversal.)
    """
    try:
        with (
            regular_file_snapshot(
                path,
                max_bytes=max_bytes,
            ) as snapshot,
            open_regular_file(
                snapshot,
                max_bytes=max_bytes,
            ) as (handle, _info),
        ):
            return np.load(handle, allow_pickle=False)
    except Exception as exc:
        raise AgentFault(
            f"HDF5 readback file {path} could not be snapshotted and parsed "
            f"safely: {type(exc).__name__}: {exc}"
        ) from exc


def load_submission_h5_or_fault(
    path: str | Path,
    *,
    datasets: Iterable[str] | None = None,
    max_bytes: int = _DEFAULT_MAX_HDF5_BYTES,
    max_datasets: int = _DEFAULT_MAX_HDF5_DATASETS,
    max_dataset_bytes: int = _DEFAULT_MAX_HDF5_DATASET_BYTES,
    max_total_bytes: int = _DEFAULT_MAX_HDF5_LOGICAL_BYTES,
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

    if min(max_bytes, max_datasets, max_dataset_bytes, max_total_bytes) <= 0:
        raise ValueError("all HDF5 size and dataset limits must be positive")

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
        snapshot = regular_file_snapshot(p, max_bytes=max_bytes)
        pinned = snapshot.__enter__()
    except FileNotFoundError as exc:
        raise AgentFault(f"submitted HDF5 at {p} is missing: {exc}") from exc
    except OSError as exc:
        raise AgentFault(
            f"submitted HDF5 at {p} is not a stable regular file: {exc}"
        ) from exc

    names = list(datasets) if datasets is not None else None

    # Must be writable by the dropped (uid-1000) worker when the grader is root.
    out_dir: str | None = None
    try:
        out_dir = os.path.realpath(tempfile.mkdtemp(prefix="h5-submission-read-"))
        if os.geteuid() == 0:
            os.chmod(out_dir, 0o777)

        # A missing drop identity / un-spawnable worker raises PolicyWorkerError
        # (infra, discarded); the read RPC is wrapped separately so a parse
        # crash/timeout is attributed to the agent.
        reader = load_submitted_policy(
            source=_H5_READER_WRAPPER,
            factory_name="load_reader",
            path=pinned,
            sys_path_dirs=[],
            timeout_s=_H5_READ_TIMEOUT_S,
        )
        try:
            result = reader.read(
                names,
                out_dir,
                max_datasets,
                max_dataset_bytes,
                max_total_bytes,
            )
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
        written = result.get("written") or []
        if len(written) > max_datasets:
            raise AgentFault(
                f"submitted HDF5 produced {len(written)} datasets, over limit "
                f"{max_datasets}"
            )
        seen_names: set[str] = set()
        seen_files: set[str] = set()
        total_bytes = 0
        for entry in written:
            # basename() pins the file to out_dir's top level; the loader pins
            # every directory component and the regular leaf before np.load.
            # allow_pickle=False also blocks a smuggled object array.
            name = str(entry["name"])
            filename = os.path.basename(str(entry["file"]))
            if name in seen_names or filename in seen_files:
                raise AgentFault("submitted HDF5 produced duplicate dataset outputs")
            seen_names.add(name)
            seen_files.add(filename)
            npy = os.path.join(out_dir, filename)
            array = _np_load_regular_nofollow(
                npy,
                max_bytes=max_dataset_bytes + 1024 * 1024,
            )
            logical_bytes = int(array.nbytes)
            if logical_bytes > max_dataset_bytes:
                raise AgentFault(
                    f"submitted HDF5 dataset {name!r} expands to {logical_bytes} "
                    f"bytes, over per-dataset limit {max_dataset_bytes}"
                )
            total_bytes += logical_bytes
            if total_bytes > max_total_bytes:
                raise AgentFault(
                    f"submitted HDF5 datasets expand to {total_bytes} bytes, "
                    f"over aggregate limit {max_total_bytes}"
                )
            out[name] = array
        return out
    finally:
        if out_dir is not None:
            shutil.rmtree(out_dir, ignore_errors=True)
        snapshot.__exit__(None, None, None)


def load_submission_h5ad_or_fault(
    path: str | Path,
    *,
    out_path: str | Path,
    max_bytes: int = _DEFAULT_MAX_HDF5_BYTES,
) -> Path:
    """Reject whole-object H5AD handoff to the root grader.

    A dropped parser cannot safely "sanitize" a complex H5AD for a second,
    root-side anndata parse: a compromised first parser could craft a new
    exploit payload. Evaluate H5AD entirely inside a task-specific dropped
    worker, or submit bounded primitive HDF5 datasets through
    :func:`load_submission_h5_or_fault`.
    """

    del path, out_path, max_bytes
    raise RuntimeError(
        "whole-object H5AD loading is disabled: evaluate it entirely inside a "
        "privilege-dropped worker or use bounded primitive HDF5 datasets"
    )


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
    "SolverResult",
    "abs_error",
    "exact_match",
    "file_contains",
    "file_exists",
    "jaccard",
    "join_submission_to_truth_or_fault",
    "json_path",
    "kendall_tau",
    "load_json",
    "load_submission_h5_or_fault",
    "load_submission_h5ad_or_fault",
    "load_submission_npz_or_fault",
    "load_submission_or_fault",
    "load_submitted_model",
    "load_submitted_policy",
    "open_submission_file_or_fault",
    "regex_search",
    "require_regular_file",
    "run_model_module",
    "run_policy",
    "run_submitted_executable",
    "run_trusted_solver",
    "transcript_contains",
    "world_integrity",
]
