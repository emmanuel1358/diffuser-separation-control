#!/usr/bin/env bash
set -euo pipefail

TASK_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
REPO_ROOT="$(cd "$TASK_DIR/../.." && pwd)"

cd "$REPO_ROOT"
TEST_FILE="$TASK_DIR/tests/test_frontier_mcp_workspace.py"
uv run pytest -q "$TEST_FILE" -k "not host_oracle_noop_and_introspection_cheat"
uv run pytest -q "$TEST_FILE::test_host_oracle_noop_and_introspection_cheat"
