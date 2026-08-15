#!/bin/bash
set -e
mkdir -p /tmp/output
cat > /tmp/output/diffuser_design.json << 'JSON'
{
  "half_angle_deg": 4.0,
  "length_ratio": 3.0,
  "inlet_extension_m": 0.0}
JSON
