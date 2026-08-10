# XFOIL Rust Port

This example demonstrates the generic source-visible transformation contract on
a genuinely long-horizon legacy modernization task.

The task is self-contained: a deterministic archive of the pinned original
XFOIL tree and the Rust ceiling reference source are committed under `data/`
and `solution/`. Image creation does not download either repository. The XFOIL
archive is checksum-verified before extraction and compilation.

The rollout agent receives:

- the complete vendored XFOIL source at `/data/xfoil-source`;
- an XFOIL 6.97 executable built from that exact source during image creation;
- unrestricted public differential tooling at
  `/data/transform_runner.py`;
- a directly editable Rust repository at `/tmp/output/repo`.

The scorer generates private reference behavior with pinned XFOIL, hides the
legacy source and executable from the candidate UID, rebuilds the committed
workspace snapshot from source against a read-only vendored Cargo registry, and
drives the resulting binary through the canonical `transform/v1` JSONL
protocol. It is a declarative `RubricTask`: `WorkspaceArtifact` owns repository
validation and snapshotting, `context.run_candidate` owns the privilege-dropped
build boundary, and a root-owned ptrace monitor owns hidden execution. The
sealed `scorer/evaluation.plan.json` binds the four capability criteria.

The checked-in solution combines a pinned partial Rust implementation with a
solution-only ceiling-response corpus generated from the pinned XFOIL binary.
The corpus is not copied into the rollout image; it exists solely to prove that
all four hidden suites and comparator tolerances can reach a score of `1.0`.
Each suite returns deterministic partial behavioral credit; the shared rubric
stack owns weights, aggregation, faults, receipts, and private replay traces.

The graded v1 scope is deliberately limited to NACA geometry/paneling,
single-point inviscid analysis, single-point viscous analysis, and stateful
polar sweeps. Full geometry and inverse-design menus are not claimed or graded.

## Isolation evidence

Before candidate compilation or execution, the scorer:

- computes trusted expected behavior, then changes the public XFOIL executable
  and `/data/xfoil-source` to mode `0700`;
- removes the candidate's `.git/` metadata and `target/` cache so repository
  internals or a copied executable cannot survive into grading;
- rejects undeclared `/tmp/output` siblings, symlinks, native
  executable/archive magic, Fortran/native artifacts, `build.rs`,
  subprocess/FFI source patterns, and direct oracle paths;
- builds only inside the framework-committed workspace snapshot as uid 1000;
- runs the hidden binary as uid 1000 beneath a root-owned ptrace parent that
  permits its initial exec and thread clones, but stops fork, vfork, process
  clones, and every subsequent exec before child or replacement code can run;
- keeps the process-event ledger in verifier memory and returns candidate stdout
  through a separate trusted envelope, so candidate bytes cannot forge or
  truncate monitor evidence.

The Docker build also fails unless uid 1000 cannot read
`/mcp_server/data/hidden_requests.json` or the scorer, and unless solution or
`.alignerr` artifacts are absent from the rollout filesystem. Only the
reference dependency manifests and lock participate in an intermediate Cargo
vendoring stage; no FlexFoil source or ceiling-response corpus is copied into
the rollout image. The resulting vendored registry is root-owned and read-only.

## Licensing

XFOIL source is GPL-2.0-or-later. Plotlib is separately licensed under GNU
Library GPL v2 or later (`LGPL-2.0-or-later`). Their complete notices remain
inside the checksum-pinned archive at `xfoil-source/src/gpl.txt` and
`xfoil-source/plotlib/GPL-library`. The original source and executable remain
subject to those upstream terms:

<https://web.mit.edu/drela/Public/web/xfoil/>

The separately bundled Rust ceiling reference is MIT licensed, with its notice
at `solution/reference/LICENSE`. Surrounding Apache-2.0 task/framework material
does not relicense XFOIL or Plotlib. No commercial-delivery approval is claimed;
downstream distribution requires explicit licensing and corresponding-source
review rather than inferring permission from the example's presence.

The ptrace monitor and pinned Debian package set are validated only on
Linux/amd64. Trusted CI must build and run this task on x86-64; an arm64
emulation run is useful diagnostics but is not accepted as build proof.

## Verification

```bash
uv run bash examples/xfoil-rust-port/tests/test.sh

XFOIL_TEST_IMAGE=local/xfoil-rust-port:refresh \
  bash examples/xfoil-rust-port/tests/test_container.sh

uv run pytest grader/tests/test_rubric_task.py \
  harness/tests/test_task_metadata.py

uv run lbx-rl-template validate \
  --problem-dir examples/xfoil-rust-port

uv run lbx-rl-harness run \
  --runtime ground-truth \
  --problem-dir examples/xfoil-rust-port
```
