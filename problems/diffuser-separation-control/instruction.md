Marine hydrofoil aft-flap design

Design a two-dimensional aft flap for a simplified marine hydrofoil section. The goal is to improve useful control authority while keeping drag, separation risk, wake loss, and packaging constraints under control across a small water-tunnel style operating envelope.

Required output

Your final answer must leave this file under:

/tmp/output/hydrofoil_flap.json

The JSON file must contain exactly these four numeric fields and no additional fields:

{
  "flap_deflection_deg": 4.5,
  "hinge_gap_m": 0.016,
  "flap_chord_fraction": 0.18,
  "blend_radius_m": 0.006
}

The values above are a schema illustration only. They are intentionally conservative and must not be treated as a recommended or optimal design.

All distances are meters. Deflection is degrees. flap_chord_fraction is relative to the disclosed hydrofoil chord.

Public data available in /data/

Use the following public files:

hydrofoil_flap_template.json — required output schema.

public_operating_envelope.json — design bounds, packaging constraints, public operating ranges, and the public validation-case definition using public-prefixed diagnostic fields.

public_baseline_summary.json — baseline trends and qualitative hydrofoil tradeoffs.

public_calibration_samples.json — non-ranked example designs and qualitative response trends.

public_transfer_guidance.json — public off-design operating-regime guidance plus broad acceptable bands.

These public files provide engineering guidance for selecting a robust design. They do not provide private coefficient targets, hidden operating cases, ranked public candidates, or an official optimal geometry.

Design bounds

Keep every submitted design variable within these bounds:

flap_deflection_deg: 2.0 to 12.0

hinge_gap_m: 0.004 to 0.020

flap_chord_fraction: 0.16 to 0.34

blend_radius_m: 0.004 to 0.030

The verifier rejects degenerate or unmeshable designs.

Keep the trailing-edge offset inside the test-section packaging envelope, preserve minimum clearance, keep the hinge gap meshable, and keep the blend radius physically plausible for the selected flap chord.

Design objective

A good design should:

provide useful lift/control authority;

avoid excessive drag;

control aft-flap separation;

maintain acceptable wake quality;

remain feasible within the disclosed packaging constraints;

retain useful performance across the disclosed operating envelope rather than being tuned only to one operating point.

The disclosed public guidance indicates that useful designs generally use moderate flap deflection, enough flap chord to provide authority, a controlled hinge gap, and a nonzero blend radius. Excessive deflection, excessive gap, excessive flap length, or poor packaging can increase drag, separation risk, and wake loss.

Suggested workflow

Inspect the public files under /data/ before selecting the final geometry.

Compare candidate designs against:

the disclosed design bounds;

the public design-guidance regions;

the qualitative trends in the public calibration samples;

the public off-design transfer guidance;

derived packaging and geometry feasibility.

A candidate should be revised if it appears to obtain authority mainly by accepting excessive drag, separation risk, wake loss, or packaging risk.

The public transfer guidance provides broad operating-regime information for reasoning about robustness across speed, trim, Reynolds number, and submergence changes. It does not expose private coefficient targets or private case rows.

Scoring

Scoring is deterministic and gradual.

Credit is based on:

valid output-file existence;

valid JSON parsing;

presence of the required numeric fields;

finite numeric values;

compliance with public design bounds;

alignment with public design guidance and calibration trends;

alignment with public off-design transfer guidance;

derived geometry and packaging feasibility;

mesh health;

solver health;

nominal lift authority;

nominal drag control;

nominal separation control;

nominal wake quality;

hidden-case lift authority;

hidden-case drag control;

hidden-case wake quality;

hidden-case stability;

robustness across private operating cases.

The grader recomputes physical response quantities from the public problem definition and the submitted geometry and evaluates the design against fixed private operating cases.

Partial credit is awarded for designs that perform well on some criteria even when they are not strong across every criterion.

OpenFOAM

OpenFOAM is available in the environment as an optional tool. It may be used to investigate the disclosed public validation case and to check geometry/flow behavior, but the final deliverable remains the required JSON file.

Restrictions

Do not:

read private scorer files;

attempt to access hidden coefficients or hidden operating cases;

use alternate output paths;

submit constants unrelated to the hydrofoil flap geometry;

add extra JSON fields beyond the four required fields.

The verifier reads and evaluates:

/tmp/output/hydrofoil_flap.json

using fixed private operating cases.