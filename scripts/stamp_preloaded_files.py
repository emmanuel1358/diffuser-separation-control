#!/usr/bin/env python3
"""Write ``.alignerr/preloaded_files.json`` from packed + uploaded mount entries.

Called by ``scripts/sync_mount.sh`` once per task with one ``--entry`` per
uploaded object. The exporter reads the stamped manifest and emits it on the
Boreal problem entry; the platform mounts each object read-only.

Each ``--entry`` is ``<local_path>::<remote_path>::<read_only>``.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "alignerr_plugin" / "src"))

from alignerr_plugin.preloaded import (  # noqa: E402
    PRELOADED_MANIFEST_PATH,
    manifest_entry,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--problem-dir", required=True)
    parser.add_argument(
        "--entry",
        action="append",
        default=[],
        help="<local_path>::<remote_path>::<read_only>",
    )
    args = parser.parse_args()

    entries = []
    for raw in args.entry:
        parts = raw.split("::")
        if len(parts) != 3:
            parser.error(f"bad --entry {raw!r}; expected local::remote::read_only")
        local_path, remote_path, read_only = parts
        entries.append(
            manifest_entry(
                remote_path,
                local_path,
                read_only=read_only.strip().lower() in ("1", "true", "yes"),
            )
        )

    out_path = Path(args.problem_dir) / PRELOADED_MANIFEST_PATH
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps({"preloaded_files": entries}, indent=2) + "\n")
    print(f"stamped {len(entries)} preloaded_files -> {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
