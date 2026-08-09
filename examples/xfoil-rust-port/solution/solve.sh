#!/usr/bin/env bash
set -euo pipefail

OUT_DIR="${LBT_OUTPUT_DIR:-/tmp/output}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$OUT_DIR/repo"

mkdir -p "$OUT_DIR" "$REPO_DIR"
shopt -s dotglob nullglob
rm -rf "${REPO_DIR:?}/"*
shopt -u dotglob nullglob
cp -a "$SCRIPT_DIR/reference/." "$REPO_DIR/"
rm -rf "$REPO_DIR/target" "$REPO_DIR/.git"
find "$REPO_DIR" -type f \( \
    -name '*.f' -o -name '*.f77' -o -name '*.f90' -o -name '*.for' \
  \) -delete
python3 - "$REPO_DIR" <<'PY'
from pathlib import Path
import shutil
import sys

repo = Path(sys.argv[1])
for crate in (repo / "crates").iterdir():
    for name in ("tests", "examples"):
        shutil.rmtree(crate / name, ignore_errors=True)
testkit_src = repo / "crates" / "rustfoil-testkit" / "src"
shutil.rmtree(testkit_src, ignore_errors=True)
testkit_src.mkdir(parents=True)
(testkit_src / "lib.rs").write_text("// Validation-only empty test support crate.\n")
PY

mkdir -p "$REPO_DIR/crates/transform-candidate/src"
cat > "$REPO_DIR/Cargo.toml" <<'EOF'
[workspace]
resolver = "2"
members = [
    "crates/rustfoil-core",
    "crates/rustfoil-bl",
    "crates/rustfoil-inviscid",
    "crates/rustfoil-testkit",
    "crates/transform-candidate",
]

[workspace.package]
version = "0.1.0"
edition = "2021"
license = "MIT"
authors = ["Flexcompute, Inc.", "Harry Smith"]

[workspace.dependencies]
nalgebra = "=0.33.2"
thiserror = "=2.0.18"
EOF

cat > "$REPO_DIR/crates/transform-candidate/Cargo.toml" <<'EOF'
[package]
name = "transform-candidate"
version.workspace = true
edition.workspace = true
license.workspace = true
authors.workspace = true
publish = false

[[bin]]
name = "transform-candidate"
path = "src/main.rs"

[dependencies]
rustfoil-core = { path = "../rustfoil-core" }
rustfoil-inviscid = { path = "../rustfoil-inviscid" }
serde_json = "=1.0.149"
EOF

cp "$SCRIPT_DIR/transform_candidate_main.rs" \
    "$REPO_DIR/crates/transform-candidate/src/main.rs"
cp "$SCRIPT_DIR/ceiling_responses.json" \
    "$REPO_DIR/crates/transform-candidate/ceiling_responses.json"

chown -R 1000:1000 "$REPO_DIR"
echo "Prepared full-suite ceiling reference at $REPO_DIR"
