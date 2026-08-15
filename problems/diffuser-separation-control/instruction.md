## Diffuser Separation Control

Design a two-dimensional diffuser intended to maximize pressure recovery while avoiding excessive flow separation and maintaining good outlet flow quality.

### Data available in /data/

The following data files are available in `/data/`:

* `baseline_diffuser.json` -- baseline diffuser geometry (inlet height, outlet height)
* `public_operating_conditions.json` -- flow conditions (inlet velocity, kinematic viscosity)
* `public_design_guidance.json` -- recommended design approach and operating envelope

These files describe the baseline geometry, operating envelope, and recommended design ranges.

### Required output

Write a JSON file to:

`/tmp/output/diffuser_design.json`

Use exactly this schema:

```json
{
  "half_angle_deg": 11.0,
  "length_ratio": 3.0,
  "inlet_extension_m": 0.0
}
Note: The values above are for schema illustration only and are deliberately suboptimal. They are not tuned for score.
Parameter definitions
half_angle_deg: diffuser half-angle in degrees.
length_ratio: diffuser length divided by inlet height.
inlet_extension_m: inlet duct extension length in meters.
Constraints
4.0 <= half_angle_deg <= 12.0
3.0 <= length_ratio <= 8.0
0.0 <= inlet_extension_m <= 1.0
Design objective
A good design should:
increase static pressure recovery relative to the fixed baseline area ratio,
avoid large separated regions,
maintain a reasonably uniform outlet velocity profile,
favor moderate half-angles that balance pressure recovery against separation risk.
The score is determined from deterministic flow-analysis metrics computed from the submitted geometry:
Pressure recovery (higher is better): how much static pressure is recovered relative to the ideal value for the fixed baseline area ratio (the reference geometry is the public baseline; the area ratio is not re-derived from your submitted half-angle and length).
Separation penalty (lower is better): flow separation risk increases with half-angle; designs with larger angles face higher penalties.
Outlet uniformity (higher is better): improves with sufficient diffuser length relative to inlet height.
Robustness (higher is better): a geometric stability proxy that favors moderate half-angles.
Inlet extension bonus (higher is better): a sufficient inlet extension improves inlet flow quality and earns bonus credit.
Partial credit is awarded for designs that improve on some metrics even if they are not optimal on all. Only the JSON file in /tmp/output is graded.
