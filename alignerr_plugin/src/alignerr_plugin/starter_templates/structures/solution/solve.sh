#!/usr/bin/env bash
set -euo pipefail

# Replace with a real oracle: write the reference design your hidden grader
# scores as ~1.0. The placeholder below writes a non-empty design.json so the
# starter's placeholder rubric returns full marks.
cat > /tmp/output/design.json <<'JSON'
{
  "example_param": 1.0
}
JSON
