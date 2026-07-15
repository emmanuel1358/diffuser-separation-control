# Marine hydrofoil aft-flap design

Design a two-dimensional aft flap for a simplified marine hydrofoil section. The goal is to improve useful control authority while keeping drag, separation risk, wake loss, and packaging constraints under control across a small water-tunnel style operating envelope.

Your final answer must leave this file under `/tmp/output`:

- `/tmp/output/hydrofoil_flap.json`

The design file must be one JSON object with these numeric fields:

```json
{
  "flap_deflection_deg": 4.5,
  "hinge_gap_m": 0.016,
  "flap_chord_fraction": 0.18,
  "blend_radius_m": 0.006
}
```

All distances are meters. Deflection is degrees. `flap_chord_fraction` is relative to the disclosed hydrofoil chord. The example above is only the schema template; it is intentionally conservative and should not be treated as a recommended design.

Use the public files under `/data`:

- `hydrofoil_flap_template.json` gives the required schema.
- `public_operating_envelope.json` gives design bounds, packaging constraints, public operating ranges, and the public validation-case definition using public-prefixed diagnostic fields.
- `public_baseline_summary.json` describes baseline trends and qualitative hydrofoil tradeoffs.
- `public_calibration_samples.json` gives non-ranked example designs and qualitative response trends.
- `public_transfer_guidance.json` gives public off-design operating-regime guidance plus broad acceptable bands. It does not provide private coefficient targets.

The design variables must stay within these bounds:

- `flap_deflection_deg`: 2.0 to 12.0
- `hinge_gap_m`: 0.004 to 0.020
- `flap_chord_fraction`: 0.16 to 0.34
- `blend_radius_m`: 0.004 to 0.030

The verifier rejects degenerate or unmeshable designs. Keep the trailing-edge offset inside the test-section packaging envelope, preserve minimum clearance, keep the hinge gap meshable, and keep the blend radius physically plausible for the selected flap chord.

A good workflow is to inspect the public files, compare candidate designs against the disclosed bounds and qualitative tradeoffs, write `/tmp/output/hydrofoil_flap.json`, and revise the geometry if it appears to trade weak authority for excessive drag risk, separation risk, wake loss, or packaging risk.

The public transfer guidance gives enough public operating-regime information to reason about robustness across speed, trim, and submergence changes without exposing private case rows or coefficient anchors.

Scoring is deterministic and gradual. Credit comes from a valid design file, numeric fields, public-bound compliance, public guidance alignment, off-design transfer alignment, derived geometry feasibility, private case health, lift authority, drag control, separation control, wake quality, and robustness spread across private operating cases. The grader recomputes the physical response quantities from the public problem definition and the submitted geometry and checks them against hidden operating cases and tolerances.

Do not try to read private scorer files, create alternate output paths, or submit constants unrelated to the hydrofoil flap geometry. The verifier reads the required file under `/tmp/output` and evaluates the submitted geometry with fixed private cases.
