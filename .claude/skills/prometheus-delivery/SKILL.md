---
name: prometheus-delivery
description: Configures Prometheus delivery for CFD and structures tasks. Use with prometheus-cfd, prometheus-structures, prometheus-eval starters, delivery.platform prometheus, or Prometheus PR acceptance reports.
---

# Prometheus Delivery

Prometheus is a delivery route, not a task type. Keep:

- `task_type = "cfd"` or `"structures"`;
- the correct scoped domain;
- the normal deterministic solver/rubric contract; and
- all trusted CI, oracle, security, and validation checks.

```toml
[delivery]
platform = "prometheus"
eval = false # or true for prometheus-eval-* starters
```

## Starter matrix

| Domain | Non-eval | Eval |
| --- | --- | --- |
| CFD | `prometheus-cfd` | `prometheus-eval-cfd` |
| Structures | `prometheus-structures` | `prometheus-eval-structures` |

Do not convert the task type to `ml` merely because Prometheus runs an agent.

## Acceptance

- All required trusted CI checks must be green.
- Prometheus CFD/structures require target average `<= 0.5`.
- Eval rows treat Boreal as coaching after trusted CI; follow the eval guide.
- Non-eval rows require the configured Boreal QA surface with no unresolved
  critical findings before review; follow the non-eval guide.
- Do not apply the software-engineering Boreal `<= 0.4` gate to these
  Prometheus CFD/structures rows.

## Authoring

Use `numerical-solver-tasks` for solver, determinism, rubric, and resource
design. Prometheus delivery changes the post-CI destination, not engineering
quality.

Keep:

- `[agent].user = "agent"`;
- `[verifier].user = "root"`;
- solver-agnostic prompt;
- deterministic oracle;
- target-specific resource enum;
- offline dependencies; and
- required reviewer artifacts when declared.

## References

- `project_guidelines/cfd/prometheus_cfd_environments.md`
- `project_guidelines/cfd/prometheus_eval_cfd_environments.md`
- `project_guidelines/strctural_engineering/PROMETHEUS_STRUCTURAL_ENGINEER_OPENSEES_AUTHORING.md`
- `project_guidelines/strctural_engineering/PROMETHEUS_EVAL_STRUCTURAL_ENGINEER_OPENSEES_AUTHORING.md`
