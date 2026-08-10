# Software Engineering Starter

This is a compact native `software_engineering / repo_debugging` starter. It
seeds an editable repository at `/tmp/output/repo` and grades the committed
workspace through `WorkspaceArtifact` and `run_candidate_suite`.

Before publishing a real task:

- replace the demonstration repository and prompt with a substantive debugging
  problem;
- replace both root-only files in `scorer/data/` with independent hidden cases
  and a trusted driver;
- keep candidate builds and executables behind `run_candidate_suite`;
- make `solution/solve.sh` score exactly `1.0`;
- keep `baselines/noop.sh` at or below the reviewed trivial-score ceiling; and
- regenerate `scorer/evaluation.plan.json` from `TASK` rather than editing it.

The bundled demonstration source is original scaffolding authored for this
repository and has no third-party source or dataset dependency. When adapting
an upstream repository, retain its license and notices and document the source,
license, redistribution terms, and modifications here.

Run the focused host checks with:

```bash
bash tests/test.sh
```

Run full reference and ground-truth verification with:

```bash
uv run lbx-rl-harness reference --problem-dir problems/<task_id>
uv run lbx-rl-harness run --runtime ground-truth --problem-dir problems/<task_id>
```
