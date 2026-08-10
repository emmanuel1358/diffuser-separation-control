# Frontier MCP workspace conformance example

This compact, fully synthetic task models the capability shape measured for
FrontierBench's `medical-claims-processing` task without using its
instructions, solution, hidden tests, fixtures, documents, or domain data.

The mapping is:

- `main`: an unprivileged git-backed agent repository.
- `workspace`: a deterministic HTTP workspace that owns the synthetic visible
  records and writes a shared snapshot.
- `browser-mcp`: a task-local SSE MCP service with browser-shaped
  `browser_navigate`, `browser_snapshot`, and `browser_click` tools.
- `workspace-shared`: writable only by `workspace` and read-only in `main` and
  `browser-mcp`.
- `verifier`: a separate trusted service that receives the sealed repo and
  workspace snapshot read-only and writes canonical `grade.json`.

The MCP service implements the MCP initialize, initialized notification,
`tools/list`, and `tools/call` JSON-RPC flow over the standard legacy SSE
transport using only Python's standard library. It is deliberately a mock: it
does not launch a browser or reach the internet.

All task and hidden rubric records are deterministic and self-generated. Every
child is built locally from a digest-pinned Python base. The generated capsule
Compose uses digest-locked archive tags with `pull_policy: never`, so runtime
does not perform mutable image pulls.
Agent-facing services run as non-root users. Only the isolated trusted verifier
runs as root, because the runtime deliberately seals its read-only handoff as
root-owned evidence.

The root scorer keeps expected outputs in its private fixture. Each bounded
candidate process receives one input claim and returns one observation; the
expected answer is compared only in the trusted orchestrator. The conformance
suite includes a frame, `sys.modules`, and garbage-collector introspection cheat
that must score zero.

## Fast checks

From the repository root:

```bash
examples/frontier-mcp-workspace/tests/test.sh
uvx ruff check --isolated --target-version py313 examples/frontier-mcp-workspace
uvx black --check --target-version py313 examples/frontier-mcp-workspace
```

The tests validate the native schema/runtime parser, generated Compose graph,
an actual local SSE MCP list/call session, the independent verifier output, and
oracle/no-op/introspection-cheat RubricTask scores.

Generate a Harbor directory with:

```bash
uv run lbx-rl-template export-harbor \
  --problem-dir examples/frontier-mcp-workspace \
  --out /tmp/frontier-mcp-workspace-harbor
docker compose \
  -f /tmp/frontier-mcp-workspace-harbor/environment/docker-compose.yaml \
  config
```

Full image build/run requires Docker with Compose. Taiga additionally requires
trusted CI support for the audited task-local MCP proxy in the nested-Docker
outer capsule. Host conformance tests use a local subprocess and do not need a
Docker daemon.
