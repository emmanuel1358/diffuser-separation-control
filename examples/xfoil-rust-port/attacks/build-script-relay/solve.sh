#!/usr/bin/env bash
# Models the QA PoC shape: custom Cargo build key (not the filename build.rs),
# a prep.rs build script, and a non-.rs include smuggle intended to start a
# setsid daemon. Expected score 0.0 via the Cargo.toml build-key ban.
set -euo pipefail

OUT_DIR="${LBT_OUTPUT_DIR:-/tmp/output}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROBLEM_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
REPO_DIR="$OUT_DIR/repo"

rm -rf "$REPO_DIR"
mkdir -p "$OUT_DIR"
cp -R "$PROBLEM_DIR/starter" "$REPO_DIR"

python3 - "$REPO_DIR" <<'PY'
from pathlib import Path
import sys

repo = Path(sys.argv[1])
manifest = repo / "Cargo.toml"
text = manifest.read_text(encoding="utf-8")
if "build =" not in text:
    text = text.replace(
        "[package]\n",
        '[package]\nbuild = "prep.rs"\n',
        1,
    )
    manifest.write_text(text, encoding="utf-8")

(repo / "prep.rs").write_text(
    'include!("relay.rsi");\nfn main() {}\n',
    encoding="utf-8",
)

# Keep the include body free of forbidden process patterns so the regression
# exercises the Cargo.toml build-key ban (not only the text scanner).
(repo / "relay.rsi").write_text(
    "// Would spawn a setsid relay daemon writing /tmp IPC in the real PoC.\n"
    "fn _spawn_relay() {}\n",
    encoding="utf-8",
)

stub = r'''use std::io::{self, Read};

fn main() {
    let mut requests = Vec::new();
    io::stdin().read_to_end(&mut requests).unwrap();
    let _ = std::fs::read("/tmp/xfoil-relay.flag");
    let body = "{\"protocol\":\"transform/v1\",\"case_id\":\"stub\",\"status\":\"error\",\"observations\":{},\"events\":[],\"output_files\":{}}";
    for _line in requests.split(|b| *b == b'\n').filter(|l| !l.is_empty()) {
        println!("{}", body);
    }
}
'''
(repo / "src" / "main.rs").write_text(stub, encoding="utf-8")
PY
