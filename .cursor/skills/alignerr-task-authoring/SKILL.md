---
name: alignerr-task-authoring
description: Guides end-to-end Alignerr RL task creation in iso-template. Use when creating or editing problems, task.toml, prompts, environments, scorers, solutions, baselines, local harness runs, or task PRs.
---

# Alignerr Task Authoring

## Start here

Create new tasks under `problems/<task_id>/`; `examples/` is a reviewed,
template-owned pattern library.

```bash
uv run lbx-rl-template create \
  --name labelbox/<task_id> \
  --template <starter> \
  --out problems
```

Required foundations:

- final agent artifacts live under `/tmp/output` and are declared in
  `[[outputs]]`;
- public inputs live in `data/`; hidden fixtures live in `scorer/data/`;
- the agent runs as UID 1000 and cannot traverse `/mcp_server`;
- task type, domain, and reward type use enums from
  `alignerr_plugin.task_metadata`;
- the oracle and grader use the same scoring contract; and
- all runtime dependencies are declared through task dependency channels.

## Route to the specialized skill

| Task or concern | Skill |
| --- | --- |
| Continuous-scored `ml` | `ml-tasks` |
| MuJoCo/MJCF/policy/video | `mujoco-tasks` |
| `cfd` or `structures` solvers | `numerical-solver-tasks` |
| New repository software task | `software-engineering-tasks` |
| Services, captures, MCP, outer capsule | `service-capsule-tasks` |
| Black-box interaction over `/tmp/env.sock` | `hidden-env-tasks` |
| Prometheus delivery | `prometheus-delivery` |
| Scorer implementation and plans | `deterministic-grading` |
| Criterion quality | `rubric-design` |
| AgentFault/loaders/security | `reward-hacking-security` |
| Oracle, proof, reviewer artifacts | `ground-truth-oracle` |
| Existing legacy-task conversion | `task-migration` |

Read every skill whose trigger applies. Service-capsule software tasks need both
`software-engineering-tasks` and `service-capsule-tasks`.

## Authoring loop

1. Define the engineering/research capability and public acceptance contract.
2. Choose the smallest correct starter and runtime shape.
3. Build public environment and tests before hidden scoring.
4. Implement the oracle and no-op/naive baselines.
5. Implement grading only through shared framework APIs.
6. Add deterministic hidden semantic coverage and relevant attack regressions.
7. Iterate:

```bash
uv run lbx-rl-harness reference --problem-dir problems/<task_id>
uv run lbx-rl-template check --problem-dir problems/<task_id>
```

8. Before PR:

```bash
uv run lbx-rl-harness run \
  --runtime ground-truth \
  --problem-dir problems/<task_id>
uv run lbx-rl-template validate --problem-dir problems/<task_id>
```

9. Open/update the assigned fork PR and wait for authoritative trusted CI.

## Acceptance

- Every required trusted CI check must be green.
- The oracle must meet its reward target and trivial/attack baselines must stay
  at the floor.
- Newly authored software problems also require Boreal aggregate `<= 0.4` over
  valid configured attempts.
- Prometheus CFD/structures use the acceptance policy in
  `prometheus-delivery`; do not substitute the software threshold.

Do not make a task harder with hidden requirements, flaky tests, shorter fair
timeouts, or packaging friction.

## Runner / context mode

Default `[runner]` uses `max_ctx = 1_000_000`, `turn_limit = 1430`, and
`context_mode = "autocompact"` (`enable_autocompact=true`).

| Mode | API flags | Guidance |
| --- | --- | --- |
| `none` | neither | Stop when context fills. |
| `memory` | `enable_memory=true` | Memory tool + context resets. **Do not enable without consulting Labelbox first.** |
| `autocompact` | `enable_autocompact=true` | Silent summarize/compress near the limit (default). |

## References

- `README.md`
- `docs/AUTHORING.md`
- `docs/GRADING.md`
- `docs/GROUND_TRUTH.md`
- `project_guidelines/`
- `examples/README.md`
