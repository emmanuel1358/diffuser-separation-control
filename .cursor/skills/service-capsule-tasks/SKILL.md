---
name: service-capsule-tasks
description: Authors multi-service software tasks with workspace lifecycle, sidecars/init jobs, captures, artifacts, MCP/SSE, isolated verifiers, Harbor Compose, and Taiga outer capsules. Use when task.toml declares services or native capability sections.
---

# Service and Capsule Tasks

Use this skill with `software-engineering-tasks`.

Canonical examples:

- `examples/frontier-service-cutover/`
- `examples/frontier-mcp-workspace/`

Full reference: `docs/SOFTWARE_ENGINEERING_FRAMEWORK.md`.

## Choose this mode only when necessary

Use explicit services when behavior depends on live databases, brokers,
customer/workload simulators, browser/tool sidecars, init jobs, cross-service
state, or post-agent state capture.

Do not add services when a trusted driver in one image preserves the same
engineering challenge.

## Graph contract

- Exactly one explicit `role = "main"` service.
- At most one `role = "verifier"` service.
- Main is unprivileged.
- A verifier requires explicit `[result].reward_file` and `reward_key`.
- Each service uses one local build or digest-pinned image.
- Build targets must name declared local Dockerfile stages.
- Dependencies use `started`, `healthy`, or init-only `completed`.
- Sleeps do not replace readiness/healthchecks.

## Security

- Software agent services remain offline/isolated.
- Verifier uses no network.
- No verifier dependency, volume, or namespace share crosses to the agent graph.
- Named volumes only; no host binds, devices, sockets, or privileged mode.
- Only `SYS_PTRACE` is allowlisted and must have a tested trusted need.
- External images use full SHA-256 digests.

## Workspace and evidence

Use `[workspace]` for seed/root/CWD/checkpoint behavior. Remember that any
capability section opts into outer-capsule export even without explicit
services.

Finalization order:

1. run ordered captures;
2. pause agent-reachable services;
3. collect bounded artifacts;
4. freeze the root-owned manifest/snapshot;
5. stop agent graph;
6. run network-isolated verifier; and
7. read canonical result.

Captures require timeout, accepted exit codes, atomic destination when
applicable, and correct agent-vs-infrastructure failure policy.

Artifacts require service, kind, destination, required flag, and realistic
byte/file/depth limits. The repository artifact for
`WorkspaceArtifact("repo")` is collected from main to destination `repo`.

## MCP

- Prefer a narrow task-owned SSE service.
- Declare service, dependencies, readiness, and phase access.
- Taiga supports only the audited capsule service-DNS SSE proxy.
- Do not expose arbitrary URLs, host services, Docker, or an unbundled proxy.
- Test valid calls, invalid arguments, unavailable readiness, and destination
  abuse.

## Export

Harbor:

```bash
uv run lbx-rl-template export-harbor \
  --problem-dir problems/<task_id> \
  --out /tmp/harbor \
  --image registry/task@sha256:<digest>
```

Trusted Taiga packaging:

```bash
uv run lbx-rl-template export-capsule \
  --problem-dir problems/<task_id> \
  --out /tmp/capsule \
  --trusted-build

uv run lbx-rl-template export-taiga \
  --problem-dir problems/<task_id> \
  --out /tmp/problems-metadata.json \
  --image registry/outer@sha256:<digest> \
  --outer-capsule
```

Child builds/pulls are trusted-CI operations. Taiga nested capsules currently
use CPU Firecracker tiers, prebuilt child images, and audited SSE MCP only.

## Verify

- `lbx-rl-template check`
- capsule/Harbor export parity
- service health/readiness
- capture ordering/fault classification
- artifact limits and symlink rejection
- verifier network/volume isolation
- task-specific introspection/capture forgery attacks

## References

- `docs/SOFTWARE_ENGINEERING_FRAMEWORK.md`
- `docs/FRONTIERBENCH_CAPABILITIES.md`
- `harness/tests/fixtures/frontierbench_native_conformance.toml`
