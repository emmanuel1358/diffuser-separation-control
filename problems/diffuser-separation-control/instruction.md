## Diffuser Separation Control

Design a two-dimensional diffuser intended to maximize pressure recovery while avoiding excessive flow separation and maintaining good outlet flow quality.

### Available public data

The directory `/data` contains:

* `baseline_diffuser.json`
* `public_operating_conditions.json`
* `public_design_guidance.json`

These files describe the baseline geometry, operating envelope, and recommended design ranges.

### Required output

Write a JSON file to:

`/tmp/output/diffuser_design.json`

Use exactly this schema:

```json
{
  "half_angle_deg": 6.5,
  "length_ratio": 5.2,
  "inlet_extension_m": 0.5
}
```

### Parameter definitions

* `half_angle_deg`: diffuser half-angle in degrees.
* `length_ratio`: diffuser length divided by inlet height.
* `inlet_extension_m`: inlet duct extension length in meters.

### Constraints

* `4.0 <= half_angle_deg <= 12.0`
* `3.0 <= length_ratio <= 8.0`
* `0.0 <= wall_curvature_factor <= 0.4`

### Design objective

A good design should:

* increase static pressure recovery relative to the baseline,
* avoid large separated regions,
* maintain a reasonably uniform outlet velocity profile,
* remain robust across the operating conditions described in the public data.

The score is determined from deterministic flow-analysis metrics computed from the submitted geometry, including pressure-recovery behavior, separation-related penalties, outlet-flow quality, and robustness across multiple operating conditions.

Only the JSON file in `/tmp/output` is graded.
