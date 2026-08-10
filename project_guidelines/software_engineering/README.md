# Software Engineering RL Task Authoring

Use these guides to build long-horizon software-engineering environments with
the native ISO task framework:

- [Authoring Frontier-style software-engineering tasks](frontier_style_software_engineering_tasks.md)
  is the end-to-end handbook for task design, repository layout, deterministic
  grading, service graphs, reward-hacking resistance, and validation.
- [New task design and delivery workflow](new_task_design_workflow.md) explains
  how to take a brand-new task idea from capability definition through public
  specification, hidden-suite design, calibration, trusted CI, and acceptance.
- [Software-engineering framework technical reference](../../docs/SOFTWARE_ENGINEERING_FRAMEWORK.md)
  documents the schema, exporters, capsule runtime, service lifecycle, artifact
  handoff, MCP bridge, and trust boundaries.

The canonical worked examples live in the
[`lbx-rl-tasks-iso-template`](https://github.com/Alignerr-Code-Labeling/lbx-rl-tasks-iso-template)
repository:

- [`wal-recovery-ordering`](https://github.com/Alignerr-Code-Labeling/lbx-rl-tasks-iso-template/tree/main/examples/wal-recovery-ordering):
  focused repository debugging, concurrency, determinism, and performance.
- [`xfoil-rust-port`](https://github.com/Alignerr-Code-Labeling/lbx-rl-tasks-iso-template/tree/main/examples/xfoil-rust-port):
  long-horizon legacy modernization with public differential tooling and
  subprocess-delegation defenses.
- [`frontier-service-cutover`](https://github.com/Alignerr-Code-Labeling/lbx-rl-tasks-iso-template/tree/main/examples/frontier-service-cutover):
  multi-service orchestration, health gates, ordered captures, and an isolated
  verifier.
- [`frontier-mcp-workspace`](https://github.com/Alignerr-Code-Labeling/lbx-rl-tasks-iso-template/tree/main/examples/frontier-mcp-workspace):
  shared service state and an audited MCP/SSE tool bridge.

Start new work in `problems/<task_id>/`. Treat examples as reviewed references:
copy the smallest relevant pattern, not an example directory wholesale.

Every authored problem has two non-negotiable acceptance requirements:

1. all required trusted CI checks are green; and
2. the problem's Boreal aggregate score is **less than or equal to `0.4`**.

A task is not complete merely because its oracle and unit tests pass. If the
Boreal aggregate exceeds `0.4`, revise the task so it measures a deeper but
still fully specified engineering capability.
