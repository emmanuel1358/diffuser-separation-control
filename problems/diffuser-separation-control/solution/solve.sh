#!/usr/bin/env bash
set -euo pipefail

mkdir -p /tmp/output

cat > /tmp/output/diffuser_design.json <<'EOF'
{
  "half_angle_deg": 7.0,
  "length_ratio": 6.0,
  "inlet_extension_m": 0.5
}
EOF

echo "Wrote /tmp/output/diffuser_design.json"