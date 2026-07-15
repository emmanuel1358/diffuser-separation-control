# Marine hydrofoil aft-flap design

# _rev: 2026-06-17.b7prc04t3_batch8

This task asks an agent to design a simplified two-dimensional marine hydrofoil aft flap. Batch8 hardens the task around reviewer feedback:

- Public off-design transfer guidance is disclosed so hidden robustness is estimable without exposing private coefficient targets.
- The reward is less dominated by exact private lift/drag anchors. Hidden cases still matter, but broad public-guided robustness now gives meaningful partial credit.
- The prompt is solver-agnostic: one design output, no solver-provenance artifacts, no command checklist.

## Required output files

```text
/tmp/output/hydrofoil_flap.json
```

## Public data files

```text
data/hydrofoil_flap_template.json
data/public_operating_envelope.json
data/public_baseline_summary.json
data/public_calibration_samples.json
data/public_transfer_guidance.json
```

The public files provide bounds, qualitative trends, calibration samples, and transfer guidance. They do not disclose private coefficient targets.

## Scoring posture

The scorer uses deterministic criteria for file validity, design bounds, public guidance alignment, public transfer alignment, derived geometry, mesh and solver health, private-case physics, and robustness spread.

Exact private lift and drag anchors remain the high-credit path, but they no longer dominate the reward as a near-delta spike. Public transfer bands give an honest partial-credit bridge for designs that remain physically plausible across disclosed off-design operating conditions.

## Oracle posture

The oracle does not use a digest shortcut. It searches through the same scorer path and writes the same required design file.

## Expected validation

```text
tests/test.sh: pass
ground-truth: 1.0
validate: valid
agent/Boreal: target average below 0.400
```

If new QA findings appear, prioritize issues that affect public-hidden consistency, scoring honesty, ground truth, or active runtime behavior.
