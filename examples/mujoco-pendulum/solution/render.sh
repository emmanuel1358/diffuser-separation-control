#!/usr/bin/env bash
set -euo pipefail

OUTPUT_DIR="${LBT_OUTPUT_DIR:-/tmp/output}"
uv run python -m lbx_rl_tasks_harness.render_mujoco \
  --model "${OUTPUT_DIR}/model.xml" \
  --output "${OUTPUT_DIR}/rendering.mp4" \
  --config solution/render_config.py
