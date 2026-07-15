# Lead-rubber base-isolation design

A `structures` task. The agent designs the lead-rubber base-isolation system
(characteristic strength `Qd_kip`, post-yield stiffness `Kd_kip_per_in`, lead
yield displacement `Dy_in`) for a three-story building and submits one JSON
design file.

## Engineering decision

Competing performance constraints under nonlinear response-history analysis:

- soft/weak isolation minimizes superstructure drift, floor acceleration, and
  base shear, but its isolator displacement grows and can exceed the seismic
  moat (pounding);
- stiff/strong isolation controls displacement but transmits higher drift,
  acceleration, and base shear.

The score is the worst case over six private ground motions that span the
disclosed band up to its maximum-considered (MCE) corner. A design optimized
only to mild design-basis records under-controls displacement at the MCE corner
and is gated by the moat.

## How it is graded

`scorer/compute_score.py` validates the design (it never imports/execs the
submission), then independently runs the hidden structural model on the six
private records and scores four worst-case metrics (isolator displacement, floor
acceleration, interstory drift, base shear) against disclosed targets and credit
curves, with a moat-capacity pounding gate and a disclosed final exponent.

## Local results

- Ground truth (`solution/solve.sh`, oracle search): **1.000**
- No-op / empty submission: **0.000**
- Mild-record-optimized soft design: ~0.04
- Typical "textbook" stiff design: ~0.41

The agent/Boreal difficulty is confirmed in CI (a local agent run was not
possible here for billing reasons).

## Files

- `instruction.md`, `task.toml`, `metadata.json`
- `data/` - public schema, envelope, summary, starter, description, and support model
- `scorer/compute_score.py`, `scorer/data/hidden_cases.json`
- `solution/solve.sh`, `solution/oracle_search.py`
- `baselines/naive.sh`, `tests/test.sh`
- `environment/Dockerfile`

Verify locally:

```bash
uv run lbx-rl-template validate --problem-dir examples/opensees-base-isolation
uv run lbx-rl-harness run --runtime ground-truth --problem-dir examples/opensees-base-isolation
bash examples/opensees-base-isolation/tests/test.sh
```
