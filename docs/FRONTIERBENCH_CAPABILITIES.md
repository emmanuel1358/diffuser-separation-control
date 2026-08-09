# Frontier-Bench Capability Conformance

This document is generated from capability-only production metadata. Regenerate it and the JSON matrix with:

```bash
uv run python scripts/generate_frontierbench_capabilities.py --source ../frontier-bench
```

Source revision: `c622d7a24f67e5a495209931a347dda9c98e1505`. Input digest: `sha256:305a100b553a166f23b31ceff6141dbf1cf0efeb306f8f53673879435b78f56c`. The corpus contains exactly 74 unique production slugs (`sha256:d90cdc3a869482daa0d6d04655da2da1b9a33b8817944addb5c6dfb1e76b44d5`).

The generator reads only `tasks/*/task.toml`, optional production Compose overrides, and `tasks/*/tests/test.sh`. It does not read task instructions, hidden tests or fixtures, solutions, or credential stores. Environment values and shell commands are not emitted.

## Corpus counts

- 195 artifact entries: 174 string, 21 table; observed path shapes: 16 ambiguous_path, 23 binary, 134 file, 22 tree.
- 12 Compose tasks with 41 declared services.
- 7 tasks use 12 pre-verification collect hooks.
- 9 tasks collect 16 artifacts from non-main services.
- 1 task declares 1 task-local MCP server.
- 4 tasks request an agent GPU; 2 also request a verifier GPU.
- 7 tasks declare an independent verifier resource block.
- 6 tasks have static browser-stack evidence.
- 59 verifier entrypoints invoke pytest; 61 emit CTRF directly.
- 67 entrypoints reference `reward.txt` directly and 6 reference `reward.json`; the remainder delegate reward publication to their invoked suite.
- 2 entrypoints contain an explicit repetition/determinism signal and 14 contain an explicit anti-cheat or stale-output signal.

Artifact `ambiguous_path` means the legacy declaration does not say whether a suffix-less path is a file, tree, or mode-preserving binary. The JSON records all valid native candidates instead of guessing.

## Time and resources

- Agent timeout: 1800 seconds to 28800 seconds (74 declarations).
- Verifier timeout: 60 seconds to 18000 seconds (74 declarations).
- Build timeout: 300 seconds to 18000 seconds (74 declarations).
- Agent CPU: 1 vCPU to 16 vCPU (74 declarations).
- Agent memory: 2048 MiB to 32768 MiB (74 declarations).
- Agent storage: 4096 MiB to 1024000 MiB (74 declarations).

The native mapping uses independent `ResourceSpec` declarations for agent and verifier phases. Harbor projects those to its main and separate-verifier environments. Taiga preserves the selected resource enum, validates known phase peaks for outer capsules, and rejects nested-Docker GPU/TPU child requests that Firecracker cannot satisfy.

## Production service graphs

- **ctr-optimization** (2 services: `api`, `main`) — `main` -> `api` (service_healthy).
- **cumulative-layout-shift** (2 services: `barber-shop-data-backend`, `main`) — `main` -> `barber-shop-data-backend` (service_healthy).
- **erp-procurement-planning** (3 services: `main`, `odoo`, `postgres`) — `main` -> `odoo` (service_started); `odoo` -> `postgres` (service_healthy); named volumes: `odoo-api-key`.
- **freight-dispatch-shift** (2 services: `event-feed`, `main`) — `main` -> `event-feed` (service_healthy).
- **heat-pump-warranty** (7 services: `asset-ledger`, `compliance-ledger`, `document-vault`, `main`, `returns-ledger`, `warranty-inbox`, `warranty-portal`) — `main` -> `asset-ledger` (service_healthy); `main` -> `compliance-ledger` (service_healthy); `main` -> `document-vault` (service_healthy); `main` -> `returns-ledger` (service_healthy); `main` -> `warranty-inbox` (service_healthy); `main` -> `warranty-portal` (service_healthy).
- **intrastat-meldung** (6 services: `compliance-hub`, `dms`, `idev`, `main`, `odoo`, `services`) — `compliance-hub` -> `idev` (service_healthy); `main` -> `compliance-hub` (service_healthy); `main` -> `dms` (service_healthy); `main` -> `idev` (service_healthy); `main` -> `odoo` (service_healthy); `main` -> `services` (service_healthy); named volumes: `intrastat-runtime`.
- **kv-live-surgery** (2 services: `loadgen`, `main`) — `loadgen` -> `main` (service_healthy).
- **legacy-utility-triage** (3 services: `legacy-app`, `legacy-workstation`, `main`) — `legacy-workstation` -> `legacy-app` (service_healthy); `main` -> `legacy-workstation` (service_healthy); named volumes: `audit-log`, `legacy-runtime`, `legacy-secret`.
- **live-database-cutover** (5 services: `customer`, `main`, `mysql-db`, `postgres-db`, `redis`) — `customer` -> `mysql-db` (service_healthy); `main` -> `customer` (service_healthy); `main` -> `mysql-db` (service_healthy); `main` -> `postgres-db` (service_healthy); `main` -> `redis` (service_healthy).
- **medical-claims-processing** (3 services: `main`, `playwright-mcp`, `workspace`) — `main` -> `playwright-mcp` (service_healthy); `main` -> `workspace` (service_healthy); named volumes: `medical-shared`.
- **nextjs-performance** (2 services: `main`, `warehouse-api`) — `main` -> `warehouse-api` (service_healthy).
- **payments-pipeline-fix** (4 services: `customer`, `kafka`, `main`, `seeder`) — `main` -> `customer` (service_healthy); `main` -> `kafka` (service_healthy); `main` -> `seeder` (service_completed_successfully); `seeder` -> `kafka` (service_healthy).

Every graph maps to `ServiceSpec`, `ServiceDependency`, `ServiceHealthcheck`, `PortSpec`, and backend-managed `NamedVolume` as observed. Local build contexts become trusted `ServiceBuild` inputs. External images must be promoted to digest-pinned references before native export; runtime Compose builds, host binds, privileged containers, host namespaces, and runtime sockets are rejected.

## Representative mappings

- **wal-recovery-ordering**: 1 tree artifact, patterns `determinism.explicit, security.anti_cheat`. Maps to a bounded `TreeArtifact`, structural/performance/determinism gates, canonical reward, and sealed non-root verifier execution.
- **live-database-cutover**: 5 services, 5 collect hooks, 4 service-qualified artifacts, and independent verifier resources. Maps to the full service graph, ordered atomic captures, sealed service artifacts, and separate verifier result handling.
- **medical-claims-processing**: 3 services, named volume `medical-shared`, one SSE MCP server, browser evidence, and a sidecar artifact. Maps to the browser sidecar/MCP readiness contract and shared named volume. Harbor can project it natively; Taiga supports the declared SSE endpoint only inside a trusted outer capsule through the audited service-DNS proxy. Arbitrary unbundled SSE export remains fail-closed.
- **jax-speedrun-gpu**: agent and verifier each request 1 H100 GPU with independent timeouts and storage. It maps to separate agent/verifier `ResourceSpec` envelopes and the accelerator backend.
- **XFOIL**: excluded. No XFOIL slug or matching production task exists in the 74-task Frontier-Bench source scope at the recorded revision.

## Native schema and backend mapping

| Capability | Native primitive | Harbor | Taiga | Runtime |
|---|---|---|---|---|
| Workspace | `workspace.lifecycle` | workspace seed overlay in the main/verifier image contexts | outer-capsule workspace identity and code_root | WorkspaceRuntimeSpec and TaskServiceRuntime._initialize_workspace |
| Artifact shapes | `artifact.file, artifact.tree, artifact.path_set, artifact.binary, artifact.service` | schema 1.4 artifacts projection | digest-locked capsule artifact summary | ServiceArtifact and TaskServiceRuntime._collect_artifact |
| Compose graph | `service.graph, service.build, service.dependency, service.healthcheck, service.named_volume` | native environment/docker-compose.yaml projection | digest-locked child images in an outer nested-Docker capsule | TaskServiceConfig and TaskServiceRuntime service ownership |
| Collect hooks | `capture.pre_verification` | verifier.collect projection | outer-capsule capture summary | CaptureHook ordering, timeout, fault policy, and atomic publish |
| MCP/browser | `mcp.sse, browser.sidecar` | environment.mcp_servers projection | audited service-DNS SSE proxy inside a trusted digest-locked outer capsule; unbundled export is rejected | service-DNS-only bounded SSE ToolRegistry proxy |
| Phase resources | `resource.agent, resource.verifier` | environment resource projection | required_resources capacity validation and timeout projection | main-service command and startup time bounds |
| Reward/test/report | `evaluation.engine, evaluation.gate, evaluation.report, result.canonical` | separate tests image and tests/test.sh | image-baked rubric test_file bridge | root-owned grader or separate verifier service |
| Security/determinism | `security.sealed_verifier, determinism.repeated_gate` | non-root agent plus separate root verifier | digest-locked outer capsule under Firecracker | root-owned snapshots, no socket inheritance, fail-closed results |
| Heavy toolchains | `toolchain.image_owned` | trusted image build; toolchain remains image-owned | trusted child image bundle or accelerator base image | no runtime package installation primitive |

The complete per-field and per-pattern mapping is in `docs/frontierbench_capabilities.json`. `unmapped_fields` and `unmapped_patterns` are empty; the generator fails if a new observed field or classified capability pattern lacks a mapping decision.

## Conformance fixtures

`harness/tests/fixtures/frontierbench_native_conformance.toml` declares every required native primitive. `harness/tests/test_frontierbench_capabilities.py::test_native_conformance_fixture_covers_schema_and_export` validates the schema and export projection; `taiga_runtime/rubric/tests/test_frontierbench_conformance.py::test_native_conformance_fixture_loads_runtime_primitives` validates the dependency-free production runtime parser. The matrix records the exact primitive coverage for all three layers.

Check deterministic regeneration without writing files:

```bash
uv run python scripts/generate_frontierbench_capabilities.py --source ../frontier-bench --check
```
