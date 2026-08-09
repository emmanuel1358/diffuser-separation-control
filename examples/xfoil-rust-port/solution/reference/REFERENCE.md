# Calibration Reference

This solution-only source subset is derived from:

- Repository: <https://github.com/flexcompute/flexfoil>
- Commit: `cf303d70a9f8278d1ad4c7c87a3448a1ad703068`
- Declared upstream license: MIT

Only `rustfoil-core`, `rustfoil-bl`, `rustfoil-inviscid`, and their test-support
path dependency are retained. The rollout image does not copy the `solution/`
directory, so the agent cannot read this calibration implementation.
