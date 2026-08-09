# Port the XFOIL 6.97 analysis core to Rust

You are maintaining a source-visible legacy modernization project. The complete
pinned XFOIL 6.97 Fortran source is available at `/data/xfoil-source`, and the
`xfoil` executable built from that exact tree is installed for unrestricted
differential testing.

Directly edit the Rust repository at:

```text
/tmp/output/repo
```

The repository initially contains a compiling protocol stub. Replace it with a
standalone Rust implementation of XFOIL's headless computational behavior.
Your implementation must build offline with:

```bash
cd /tmp/output/repo
cargo build --release --offline --bin transform-candidate
```

The resulting executable must read one JSON object per line from stdin and
write exactly one normalized `transform/v1` JSON response per line to stdout.
The public cases exercise:

- XFOIL-compatible NACA geometry generation and paneling,
- inviscid operating-point analysis,
- viscous operating-point analysis and transition behavior,
- stateful polar sweeps.

These four operations are the complete graded scope, weighted equally at 25%.
`GDES`, `MDES`, `QDES`, plotting, and other interactive commands are out of
scope and ungraded.

The following public differential tools are available:

```bash
python3 /data/transform_runner.py legacy /data/public_cases.json
python3 /data/transform_runner.py candidate /data/public_cases.json
python3 /data/transform_runner.py diff /data/public_cases.json
```

You may also run the original executable directly and inspect all original
source files. Hidden grading uses additional geometries and operating
conditions through the same protocol.

The final implementation must be native Rust source. It may not invoke XFOIL,
compile or link Fortran, use FFI/native object files, bundle an existing
executable, or execute another process to answer requests. Build caches under
`target/` are ignored; the final repository itself is the graded artifact.

Leave `/tmp/output` containing only the `repo/` directory (no sibling notes,
logs, or other files). Do not ship a `.cargo/` directory or cargo config that
overrides the compiler. Rust `include!` macros are forbidden; use ordinary
in-repository modules instead. Runtime grading also rejects process/FFI
patterns in source (for example `std::process::Command`, `extern "C"`,
`dlopen`).
