# Damped Pendulum MuJoCo Task

Create a MuJoCo MJCF model at:

```text
/tmp/output/model.xml
```

The model should be a single damped pendulum:

- exactly one hinge joint,
- exactly one moving body,
- total moving-body mass near `1.0 kg`,
- center of mass roughly `0.5 m` from the hinge axis,
- joint position and velocity sensors,
- damping that makes the pendulum settle.

This example is designed for end-to-end workflow testing and as a simple starter for richer MuJoCo robotics tasks.
