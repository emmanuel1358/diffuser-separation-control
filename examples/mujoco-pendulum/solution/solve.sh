#!/usr/bin/env bash
set -euo pipefail

cat > /tmp/output/model.xml <<'XML'
<mujoco model="damped_pendulum">
  <option timestep="0.002" integrator="RK4" gravity="0 0 -9.81"/>
  <visual>
    <global offwidth="1280" offheight="720"/>
  </visual>
  <worldbody>
    <body name="pendulum" pos="0 0 0">
      <joint name="hinge" type="hinge" axis="0 1 0" damping="0.05"/>
      <geom name="rod" type="capsule" fromto="0 0 0 0 0 -1" size="0.02" mass="1.0"/>
      <site name="tip" pos="0 0 -1"/>
    </body>
  </worldbody>
  <sensor>
    <jointpos name="hinge_pos" joint="hinge"/>
    <jointvel name="hinge_vel" joint="hinge"/>
  </sensor>
</mujoco>
XML
