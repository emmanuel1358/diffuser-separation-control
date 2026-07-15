#!/usr/bin/env bash
# Harness-compatible entrypoint for the intentionally too-soft baseline.
set -euo pipefail

exec bash "$(dirname "$0")/naive.sh"
