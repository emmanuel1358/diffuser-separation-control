# Frontier service cutover conformance example

This is a compact, fully synthetic example of the nested service layer. It
models the capability shape measured for FrontierBench's
`live-database-cutover` task without using its instructions, solution, hidden
tests, credentials, or data.

The mapping is:

- `main`: an unprivileged, git-backed agent repository and HTTP API.
- `initialize`: a one-shot seed job gated on a healthy state service.
- `state`: a persistent HTTP state service with a service-owned atomic dump.
- `customer`: a concurrent load/customer service. It starts before `main`,
  retries requests while the API comes up, and exposes idempotent `/finalize`
  and `/results` endpoints.
- `verifier`: a separate trusted service that receives only a read-only sealed
  artifact snapshot and writes its own canonical `grade.json`.
- `state-data`: state-service persistence.
- `cutover-shared`: writable by `customer` and read-only in `main`.

The declared capture order is intentional: finalize customer traffic over
HTTP, atomically capture customer results, atomically dump state, then capture
the git diff. The runtime freezes services and copies the bounded,
service-owned artifacts only after those hooks succeed.

All records and private rubric cases are deterministic and self-generated.
Every child is built locally from a digest-pinned Python base. The generated
capsule Compose replaces child builds with digest-locked local archive tags and
sets `pull_policy: never`; no child image is pulled at task runtime.
Agent-facing services run as non-root users. Only the isolated trusted verifier
runs as root, because the runtime deliberately seals its read-only handoff as
root-owned evidence.

The root scorer keeps expected outputs in its private fixture. Each bounded
candidate process receives one input record and returns one observation; the
expected answer is compared only in the trusted orchestrator. The conformance
suite includes a frame, `sys.modules`, and garbage-collector introspection cheat
that must score zero.

## Fast checks

From the repository root:

```bash
examples/frontier-service-cutover/tests/test.sh
uvx ruff check --isolated --target-version py313 examples/frontier-service-cutover
uvx black --check --target-version py313 examples/frontier-service-cutover
```

The tests validate the native schema/runtime parser, generated Harbor and
capsule Compose graphs, capture ordering, the separate verifier result, and
oracle/no-op/introspection-cheat RubricTask scores.

Generate a runnable Harbor directory with:

```bash
uv run lbx-rl-template export-harbor \
  --problem-dir examples/frontier-service-cutover \
  --out /tmp/frontier-service-cutover-harbor
docker compose \
  -f /tmp/frontier-service-cutover-harbor/environment/docker-compose.yaml \
  config
```

A full build/run needs Docker with Compose and, for Taiga, trusted CI capable of
building the nested-Docker outer capsule. The focused host tests remain useful
when a local Docker daemon is unavailable.
