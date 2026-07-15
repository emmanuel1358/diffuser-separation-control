#!/usr/bin/env bash
set -euo pipefail

cat > /tmp/output/model.xml <<'XML'
<mujoco model="bad">
  <worldbody>
    <body name="box">
      <geom type="box" size="0.1 0.1 0.1" mass="1"/>
    </body>
  </worldbody>
</mujoco>
XML
