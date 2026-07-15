# Lead-rubber base-isolation design

Design the lead-rubber **base-isolation system** for a three-story building so
that it performs well across a band of design-basis ground motions. The
superstructure is fixed; you choose the isolation system.

## Deliverables

Write exactly this file:

```text
/tmp/output/isolation_design.json
```

Do not write the final answer anywhere else.

- `isolation_design.json` must satisfy `/data/design_schema.json`.

## Public files in `/data`

- `building_description.md` - structure, isolator model, units, the engineering
  trade-off, and the scoring summary.
- `design_schema.json` - required JSON format for the design.
- `public_analysis_envelope.json` - disclosed ground-motion band, public
  representative records, performance targets, moat capacity, credit curves,
  weights, and the final exponent (everything the grader scores).
- `graded_case_sampling_policy.json` - discloses the deterministic stress-test
  envelope, phase-seed convention, and suite-composition rules used to select
  the private grading records.
- `public_model_summary.py` - lightweight screening helpers for effective period
  and damping.
- `isolation_starter.json` - a valid but intentionally poor (too-soft) starter.

The hidden grading uses the **same** model, isolator, ground-motion generator
and phase-seed convention,
metric definitions, targets, moat capacity, credit curves, weights, and
exponent. Only the six fixed private ground-motion tuples and the reference
design are withheld; every `(pga_g, tp_sec, duration_sec, phase_seed)` lies
inside the disclosed envelope and follows its published suite-composition
rules.

## Design variables

`isolation_design.json` contains one object:

```json
{
  "isolation_system": {
    "Qd_kip": 300.0,
    "Kd_kip_per_in": 16.0,
    "Dy_in": 0.6
  }
}
```

| Variable | Range | Meaning |
| --- | --- | --- |
| `Qd_kip` | 80 - 650 | isolation-system characteristic (lead) strength |
| `Kd_kip_per_in` | 10 - 90 | isolation-system post-yield (rubber) stiffness |
| `Dy_in` | 0.30 - 1.50 | lead-core yield displacement |

The object must contain exactly the key `isolation_system`, which must contain
exactly those three numeric keys. The numbers above are the **too-soft starter**
in `/data/isolation_starter.json`, shown only to illustrate the format: it looks
fine on the milder design-basis records but its isolator displacement exceeds
the moat at the high-intensity, long-period corner, so it scores near zero.
Replace it with your own design.

## Design guidance

Use the public files to reason about the disclosed ground-motion band, response
targets, moat capacity, recentering gate, and the tradeoff between isolation
softness and superstructure demands. A design that only performs well on mild
motions may not be robust across the full disclosed envelope.

Before stopping, confirm the files exist and parse:

```bash
python -m json.tool /tmp/output/isolation_design.json >/dev/null
```

## Scoring summary

The grader runs the disclosed structural model on six private ground motions
inside the band and scores the **worst** case. For each response metric it
takes the worst (largest) value over the six records and awards credit by how
that value compares to the disclosed target:

| Metric | Target | Weight |
| --- | --- | --- |
| `peak_isolator_disp_in` | 43.6 | 0.22 |
| `peak_floor_acceleration_g` | 0.720 | 0.25 |
| `peak_interstory_drift_ratio` | 0.00865 | 0.19 |
| `peak_base_shear_coeff` | 0.366 | 0.18 |
| `residual_isolator_disp_in` | 8.2 | 0.16 |

```text
credit_k         = lower_ratio_curve( worst_value_k / target_k )
weighted         = 0.22*disp + 0.25*accel + 0.19*drift + 0.18*base_shear + 0.16*residual
moat_gate        = moat_gate_curve( worst_isolator_disp / 45.0 )
recentering_gate = recentering_gate_curve( worst_residual_isolator_disp / 9.0 )
score            = weighted ** 3.8 * moat_gate * recentering_gate
```

The exact credit and gate curves, targets, the `45.0 in` moat capacity, the
`9.0 in` recentering capacity, weights, and `3.8` exponent are in
`/data/public_analysis_envelope.json`. A worst-case value at or below its target
earns full credit for that metric. The **moat gate** is a separate hard limit
state: once the worst isolator displacement passes the 45.0 in moat capacity,
the building would pound the surrounding wall and the score is driven to zero,
regardless of how good the other metrics are. The **recentering gate** likewise
penalizes residual displacement beyond 9.0 in because inspection and
reoccupation would require jacking the building back to center.

Because the metrics compete - softer isolation lowers acceleration, drift, and
base shear but raises peak and residual isolator displacement, while stiffer
isolation does the opposite - a high score requires meeting all five targets at
the worst-case ground motion simultaneously. An honest design that cannot satisfy every target
should still report its true response; partial credit is awarded continuously.

A submission scores `0` if the file is missing, the design is out of range or
has wrong keys, or the response fails to converge on any design-basis motion.

Submit only the required JSON file.
