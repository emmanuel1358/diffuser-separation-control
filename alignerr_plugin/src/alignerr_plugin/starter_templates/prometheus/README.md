# Prometheus Task Template

Use this scaffold for ML-style tasks that mothership should route to
**Prometheus** instead of Taiga. The layout matches the `ml` starter; the
difference is `[delivery].platform = "prometheus"` in `task.toml`.

After creating a task, update:

- `instruction.md` with the exact prompt the agent should follow.
- `task.toml` with the task name, resources, timeouts, `[difficulty].license`
  (required for `ml`), and `[[outputs]]`.
- `scorer/compute_score.py` with your grading logic.
- `data/` with public files the agent may read.
- `scorer/data/` with private grader fixtures.
- `solution/solve.sh` if you have a reference solution for validation.

Scaffold a new task from the repo root:

```bash
mkdir -p problems
cp -R alignerr_plugin/src/alignerr_plugin/starter_templates/prometheus problems/my-task
```

For a complete working Taiga ML reference, see
`examples/mle-tabular-classification/` and the `ml` starter template.
