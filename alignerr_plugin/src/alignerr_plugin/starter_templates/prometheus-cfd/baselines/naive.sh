#!/usr/bin/env bash
set -euo pipefail

# A do-nothing baseline must score ~0. Empty parameters fail the placeholder
# rubric; replace with a representative naive submission for your task.
echo '{}' > /tmp/output/control.json
