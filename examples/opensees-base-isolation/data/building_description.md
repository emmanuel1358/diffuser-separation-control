# Base-isolated building - public engineering description

## Structure

A three-story shear-building superstructure sits on a rigid base mat that is
supported by a **lead-rubber base-isolation system**. You design the isolation
system; the superstructure is fixed.

Units are **kip, inch, second**, with gravity `g = 386.089 in/s^2`.

Seismic weight lumped at each level (carried by the isolators):

| Level | Weight (kip) |
| --- | --- |
| Base mat | 1200 |
| Floor 1 | 1000 |
| Floor 2 | 1000 |
| Floor 3 | 900 |
| **Total seismic weight W** | **4100** |

Superstructure story shear stiffness and height:

| Story | Shear stiffness (kip/in) | Height (in) |
| --- | --- | --- |
| Base mat -> Floor 1 | 1000 | 156 |
| Floor 1 -> Floor 2 | 800 | 144 |
| Floor 2 -> Floor 3 | 620 | 144 |

The superstructure carries **2% Rayleigh damping** anchored to two fixed-base
modes. All isolation energy dissipation comes from the lead core.

## Isolation system model

The isolation layer is a single zero-length element with two **parallel**
uniaxial springs (a standard bilinear lead-rubber bearing backbone):

- **Lead core** - elastic-perfectly-plastic (`ElasticPP`): yield force `Qd_kip`,
  initial stiffness `Qd_kip / Dy_in`.
- **Rubber** - linear elastic (`Elastic`): stiffness `Kd_kip_per_in`.

You choose three design variables for the whole system:

| Variable | Range | Meaning |
| --- | --- | --- |
| `Qd_kip` | 80 - 650 | characteristic (lead) strength |
| `Kd_kip_per_in` | 10 - 90 | post-yield (rubber) stiffness |
| `Dy_in` | 0.30 - 1.50 | lead-core yield displacement |

The hidden grader owns the exact nonlinear construction used for final scoring.
The public files disclose the response metrics, targets, design ranges, and
ground-motion envelope needed to reason about the design.

## Loading

Each analysis is a nonlinear response history under a deterministic synthetic
ground motion described by `(pga_g, tp_sec, duration_sec)`:

- `pga_g` - peak ground acceleration (g),
- `tp_sec` - predominant period of the motion (s),
- `duration_sec` - record length (s).

The generator (`synthesize_ground_motion`) builds a Kanai-Tajimi accelerogram
whose spectral peak sits near `1/tp_sec`, scaled to `pga_g`. It is fully
deterministic: a given `(pga_g, tp_sec, duration_sec, phase_seed)` tuple always
produces the same record. The disclosed
ground-motion band and the public representative records are in
`/data/public_analysis_envelope.json`.

The grader scores the worst case over six private records that span the full
band up to its maximum-considered (MCE) corner: the highest intensities (pga up
to 0.36) with the longest predominant periods (tp up to 3.4 s), across multiple
phase realizations. Intensity, period, and phase content affect the governing
response, so the MCE corner governs. A design that only looks good on milder
records will under-control displacement at the MCE corner and be gated by the
moat.

## The engineering decision

This is a competing-constraints isolation-design problem:

- A **softer / lighter** system (low `Kd`, low `Qd`) lengthens the isolation
  period and minimizes superstructure interstory drift, floor acceleration, and
  base shear - but its **isolator displacement grows large** and can exceed the
  **seismic moat** (clearance to the retaining wall), causing pounding.
- A **stiffer / stronger** system (high `Kd`, high `Qd`) keeps isolator
  displacement small - but transmits **higher floor accelerations, drift, and
  base shear** into the superstructure.

There is an interior sweet spot that holds isolator displacement within the
moat while keeping the superstructure demands low. Because the score is taken at
the **worst** ground motion in the band, the binding case is the
**high-intensity, long-predominant-period** corner (it drives the largest
isolator displacement). A design tuned only to the milder design-basis records
will under-control displacement at that corner.

## Graded response metrics (worst case over the six private records)

| Metric | Target | Direction |
| --- | --- | --- |
| `peak_isolator_disp_in` | 43.6 | lower better (and < 45.0 moat) |
| `peak_interstory_drift_ratio` | 0.00865 | lower better |
| `peak_floor_acceleration_g` | 0.720 | lower better |
| `peak_base_shear_coeff` | 0.366 | lower better (isolator force / W) |
| `residual_isolator_disp_in` | 8.2 | lower better (and < 9.0 recentering limit) |

`peak_base_shear_coeff` is the peak isolator force divided by the seismic weight
`W = 4100 kip`.

## Scoring summary

For each metric, the grader takes the worst (maximum) value over the six private
ground motions, then:

```text
credit_k         = lower_ratio_curve( worst_value_k / target_k )
weighted         = 0.22*disp + 0.25*accel + 0.19*drift + 0.18*base_shear + 0.16*residual
moat_gate        = moat_gate_curve( worst_isolator_disp / 45.0 )
recentering_gate = recentering_gate_curve( worst_residual_isolator_disp / 9.0 )
score            = weighted ** 3.8 * moat_gate * recentering_gate
```

The exact piecewise-linear metric and gate curves, weights, targets, moat and
recentering capacities, and exponent are all listed in
`/data/public_analysis_envelope.json`. A worst-case value at or below a target
earns full credit for that metric; the moat gate drives the score toward zero
once either physical capacity is exceeded.

A design receives a hard zero if it is missing, out of range, has the wrong
keys, or fails to converge on any design-basis motion.
