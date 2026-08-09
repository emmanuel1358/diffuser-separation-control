#!/usr/bin/env bash
set -euo pipefail

PROBLEM_DIR="${PROBLEM_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
REPO_ROOT="${REPO_ROOT:-$(cd "$PROBLEM_DIR/../.." && pwd)}"
FRAMEWORK_ROOT="${FRAMEWORK_ROOT:-$REPO_ROOT}"
export PROBLEM_DIR REPO_ROOT FRAMEWORK_ROOT
export PYTHONDONTWRITEBYTECODE=1

if command -v python3 >/dev/null 2>&1; then
  PYTHON_BIN="${PYTHON_BIN:-python3}"
elif command -v python >/dev/null 2>&1; then
  PYTHON_BIN="${PYTHON_BIN:-python}"
else
  echo "No Python interpreter available" >&2
  exit 1
fi

for path in \
  README.md \
  instruction.md \
  metadata.json \
  task.toml \
  environment/Dockerfile \
  environment/docker-compose.yaml \
  environment/apt.txt \
  environment/install-task-deps.sh \
  environment/requirements.txt \
  data/XFOIL_PROVENANCE.md \
  data/public_cases.json \
  data/transform_runner.py \
  data/xfoil-source.tar.gz \
  data/xfoil-source.tar.gz.sha256 \
  scorer/candidate_monitor.py \
  scorer/compute_score.py \
  scorer/data/hidden_requests.json \
  scorer/evaluation.plan.json \
  scorer/requirements.txt \
  attacks/encoded-oracle/solve.sh \
  attacks/build-script-relay/solve.sh \
  tests/test_container.sh \
  tests/test_grading.py \
  solution/ceiling_responses.json \
  solution/solve.sh \
  starter/Cargo.toml \
  starter/src/main.rs; do
  test -s "$PROBLEM_DIR/$path" || {
    echo "Missing or empty file: $path" >&2
    exit 1
  }
done

bash -n "$PROBLEM_DIR/solution/solve.sh"
bash -n "$PROBLEM_DIR/baselines/naive.sh"
bash -n "$PROBLEM_DIR/attacks/encoded-oracle/solve.sh"
bash -n "$PROBLEM_DIR/attacks/build-script-relay/solve.sh"
bash -n "$PROBLEM_DIR/environment/install-task-deps.sh"
bash -n "$PROBLEM_DIR/tests/test_container.sh"

PYTHONPATH="$FRAMEWORK_ROOT/grader/src${PYTHONPATH:+:$PYTHONPATH}" \
  "$PYTHON_BIN" - <<'PY'
from __future__ import annotations

import contextlib
import copy
import gzip
import hashlib
import importlib.util
import io
import json
import os
import shutil
import stat
import sys
import tarfile
import tempfile
import tomllib
from pathlib import Path, PurePosixPath
from typing import Any

from grading import AgentFault

problem = Path(os.environ["PROBLEM_DIR"])
repo_root = Path(os.environ["REPO_ROOT"])


def load_scorer():
    path = problem / "scorer" / "compute_score.py"
    spec = importlib.util.spec_from_file_location("xfoil_task_scorer", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def expect_agent_fault(operation) -> None:
    try:
        operation()
    except AgentFault:
        return
    raise AssertionError("expected an AgentFault")


archive_path = problem / "data" / "xfoil-source.tar.gz"
checksum_path = problem / "data" / "xfoil-source.tar.gz.sha256"
checksum_parts = checksum_path.read_text().strip().split()
assert checksum_parts == [
    "fabf00130639aadf5365eea07446df1a267ff907b5f6f0b9ef6d14aa039cde2c",
    "xfoil-source.tar.gz",
]
archive_bytes = archive_path.read_bytes()
assert hashlib.sha256(archive_bytes).hexdigest() == checksum_parts[0]
assert archive_bytes[:2] == b"\x1f\x8b"
assert int.from_bytes(archive_bytes[4:8], "little") == 0

with tarfile.open(fileobj=io.BytesIO(archive_bytes), mode="r:gz") as source_archive:
    members = source_archive.getmembers()
    names = [member.name for member in members]
    assert names == sorted(names)
    assert names[0] == "xfoil-source"
    assert len([member for member in members if member.isfile()]) == 252
    assert "xfoil-source/CMakeLists.txt" in names
    assert "xfoil-source/src/xfoil.f" in names
    assert "xfoil-source/src/gpl.txt" in names
    assert "xfoil-source/plotlib/GPL-library" in names
    for member in members:
        relative = PurePosixPath(member.name)
        assert not relative.is_absolute()
        assert relative.parts and relative.parts[0] == "xfoil-source"
        assert ".." not in relative.parts
        assert member.isdir() or member.isfile()
        assert member.uid == 0 and member.gid == 0
        assert member.uname == "root" and member.gname == "root"
        assert member.mtime == 0
        assert member.mode == (0o755 if member.isdir() or member.mode & 0o111 else 0o644)

    with tempfile.TemporaryDirectory(prefix="xfoil-archive-test-") as temporary:
        source_archive.extractall(temporary, filter="data")
        source = Path(temporary) / "xfoil-source"
        rebuilt = io.BytesIO()
        with gzip.GzipFile(
            filename="",
            mode="wb",
            fileobj=rebuilt,
            compresslevel=9,
            mtime=0,
        ) as compressed:
            with tarfile.open(
                fileobj=compressed,
                mode="w",
                format=tarfile.GNU_FORMAT,
            ) as destination:
                paths = [
                    source,
                    *sorted(
                        source.rglob("*"),
                        key=lambda path: path.relative_to(source.parent).as_posix(),
                    ),
                ]
                for path in paths:
                    relative = path.relative_to(source.parent).as_posix()
                    info = tarfile.TarInfo(relative)
                    info.uid = 0
                    info.gid = 0
                    info.uname = "root"
                    info.gname = "root"
                    info.mtime = 0
                    if path.is_dir():
                        info.type = tarfile.DIRTYPE
                        info.mode = 0o755
                        destination.addfile(info)
                    else:
                        mode = stat.S_IMODE(path.stat().st_mode)
                        info.type = tarfile.REGTYPE
                        info.mode = 0o755 if mode & 0o111 else 0o644
                        info.size = path.stat().st_size
                        with path.open("rb") as handle:
                            destination.addfile(info, handle)
        assert rebuilt.getvalue() == archive_bytes

provenance = (problem / "data" / "XFOIL_PROVENANCE.md").read_text()
assert "d11a1544b53623c01bb3b0cceb5862311be0e1f8" in provenance
assert checksum_parts[0] in provenance
assert "xfoil-source/src/gpl.txt" in provenance
assert "GPL-2.0-or-later" in provenance
assert "LGPL-2.0-or-later" in provenance
assert "solution/reference" not in provenance
assert "maintainer-only" in provenance
assert "Apache-2.0" in provenance and "commercial-delivery approval" in provenance

task = tomllib.loads((problem / "task.toml").read_text())
assert task["environment"]["allow_internet"] is False
assert task["agent"]["user"] == "agent"
assert task["verifier"]["user"] == "root"
assert task["verifier"]["capabilities"] == ["SYS_PTRACE"]
assert task["verifier"]["timeout_sec"] >= 3600
assert task["ground_truth"]["max_trivial_score"] == 0.0
assert task["metadata"]["transformation"]["submission_mode"] == "workspace"
assert task["outputs"][0]["path"] == "/tmp/output/repo"
assert task.get("metadata", {}).get("taiga", {}).get("image_contract") is None
assert task["runner"]["context_mode"] == "autocompact"
assert task["runner"]["max_ctx"] == 1000000
assert task["runner"]["turn_limit"] == 1430

metadata = json.loads((problem / "metadata.json").read_text())
assert "6.97" in metadata["problem_data"]["description"]
assert "6.99" not in metadata["problem_data"]["description"]
provenance_metadata = metadata["provenance"]
assert provenance_metadata["source_archive_sha256"] == checksum_parts[0]
assert provenance_metadata["commercial_delivery_approved"] is False
component_licenses = {
    component["name"]: component["license"]
    for component in provenance_metadata["components"]
}
assert component_licenses == {
    "XFOIL 6.97": "GPL-2.0-or-later",
    "XFOIL Plotlib": "LGPL-2.0-or-later",
    "Rust ceiling reference": "MIT",
    "Task and framework material": "Apache-2.0",
}
ceiling = next(
    c for c in provenance_metadata["components"] if c["name"] == "Rust ceiling reference"
)
assert "solution/reference" not in ceiling["path"]
assert "maintainer-only" in ceiling["path"]

dockerfile = (problem / "environment" / "Dockerfile").read_text()
compose = (problem / "environment" / "docker-compose.yaml").read_text()
assert "cap_add:" in compose
assert "SYS_PTRACE" in compose
assert "sha256sum -c xfoil-source.tar.gz.sha256" in dockerfile
assert dockerfile.index("sha256sum -c") < dockerfile.index("tar -xzf")
assert "cargo vendor" in dockerfile and "--versioned-dirs" in dockerfile
assert "--locked" in dockerfile
assert "find /opt/cargo-home /opt/cargo-vendor -type f -exec chmod 0444" in dockerfile
assert "/tmp/install-task-deps.sh /tmp/task-deps" in dockerfile
assert "apt-get install" not in dockerfile
assert 'test "${TARGETARCH}" = "amd64"' in dockerfile
assert "strace --version" in dockerfile
assert "/mcp_server/grader/candidate_monitor.py" in dockerfile
assert 'root:root:600' in dockerfile
assert "COPY grader/src/grading/evaluation/" not in dockerfile
assert "${PROBLEM_DIR}/solution/ceiling_responses.json" not in dockerfile
assert "${PROBLEM_DIR}/solution/transform_candidate_main.rs" not in dockerfile
assert "${PROBLEM_DIR}/solution/reference/ /" not in dockerfile
assert "/mcp_server/data/hidden_requests.json" in dockerfile
assert "! su -s /bin/sh agent -c" in dockerfile
assert "test ! -e /solution" in dockerfile
assert "test ! -e /mcp_server/reference" in dockerfile

apt_packages = {
    line
    for line in (problem / "environment" / "apt.txt").read_text().splitlines()
    if line and not line.startswith("#")
}
assert apt_packages == {
    "cargo=1.85.0+dfsg3-1",
    "rustc=1.85.0+dfsg3-1",
    "strace=6.13+ds-1",
}
installer = (problem / "environment" / "install-task-deps.sh").read_text()
assert "environment/apt.txt" in installer
assert "environment/requirements.txt" in installer
assert "scorer/requirements.txt" in installer
assert "/mcp_server/grading_deps" in installer

scorer_text = (problem / "scorer" / "compute_score.py").read_text()
compile(scorer_text, str(problem / "scorer" / "compute_score.py"), "exec")
assert "WorkspaceArtifact" in scorer_text
assert scorer_text.count("context.run_candidate(") == 1
assert "_run_monitored_candidate(" in scorer_text
assert "context.run_solver(" in scorer_text
assert "_MONITOR_SHA256" in scorer_text
assert "candidate monitor digest mismatch" in scorer_text
assert "CARGO_TARGET_DIR" not in scorer_text
assert '"--target-dir"' in scorer_text
assert "tempfile.mkdtemp" in scorer_text
assert "os.O_NOFOLLOW" in scorer_text
assert "_MAX_CANDIDATE_BINARY_BYTES" in scorer_text
assert "destination.chmod(0o555)" in scorer_text
assert "sealed_directory.chmod(0o555)" in scorer_text
assert "subprocess" not in scorer_text
assert "kill_pre_grade_agent_processes" in scorer_text
assert "_quiesce_between_stages" in scorer_text
assert "_reject_disallowed_cargo_manifest" in scorer_text
assert 'Path("/data/transform_runner.py")' in scorer_text
assert 'Path("/data/xfoil-source.tar.gz")' in scorer_text
assert '".rsi"' in scorer_text
assert '".inc"' in scorer_text
assert '".json"' not in scorer_text
assert '".jsonl"' not in scorer_text
assert '".md"' not in scorer_text
assert '"gfortran"' not in scorer_text
assert '"/workdir/"' not in scorer_text
assert "_rust_source_uses_include" in scorer_text
assert "_reject_rust_include_macros" in scorer_text
assert "_scrub_agent_smuggle_roots" in scorer_text
assert "_agent_wipe_roots" in scorer_text
assert "DEFAULT_WIPE_ROOTS" in scorer_text
assert 'Path("/tmp/.cargo")' in scorer_text
assert '"HOME": str(build_directory / "home")' in scorer_text
assert '"HOME": "/tmp"' not in scorer_text
assert "RUSTC_WRAPPER" in scorer_text
assert "build.rustc-wrapper" in scorer_text
assert ".cargo" in scorer_text

monitor_path = problem / "scorer" / "candidate_monitor.py"
monitor_text = monitor_path.read_text()
compile(monitor_text, str(monitor_path), "exec")
assert "PTRACE_O_TRACEFORK" in monitor_text
assert "PTRACE_O_TRACEVFORK" in monitor_text
assert "PTRACE_O_TRACECLONE" in monitor_text
assert "PTRACE_O_TRACEEXEC" in monitor_text
assert "PTRACE_O_EXITKILL" in monitor_text
assert "candidate attempted a subsequent exec" in monitor_text
assert "candidate attempted a process clone" in monitor_text
assert 'ROSETTA_BOOTSTRAP = "/tmp/rstub"' in monitor_text
assert "emulator_exec_events == 0" in monitor_text
assert "os.setuid(uid)" in monitor_text
assert "os.getuid() != 1000" in monitor_text
assert "stdout_b64" in monitor_text

attack_text = (problem / "attacks" / "encoded-oracle" / "solve.sh").read_text()
assert "base64 /usr/local/bin/xfoil" in attack_text
assert "std :: process :: Command" in attack_text
assert "ChildCommand :: new" in attack_text
assert "std::process::Command" not in attack_text
assert "Command::new" not in attack_text

relay_attack = (problem / "attacks" / "build-script-relay" / "solve.sh").read_text()
assert 'build = "prep.rs"' in relay_attack
assert "relay.rsi" in relay_attack
assert "include!(" in relay_attack
assert "setsid" in relay_attack
container_test = (problem / "tests" / "test_container.sh").read_text()
assert "--platform linux/amd64" in container_test
assert "--cap-add SYS_PTRACE" in container_test

runner_path = problem / "data" / "transform_runner.py"
compile(runner_path.read_text(), str(runner_path), "exec")
runner_spec = importlib.util.spec_from_file_location("xfoil_transform_runner", runner_path)
assert runner_spec is not None and runner_spec.loader is not None
runner = importlib.util.module_from_spec(runner_spec)
runner_spec.loader.exec_module(runner)

public_cases_path = problem / "data" / "public_cases.json"
public_requests = json.loads(public_cases_path.read_text())
original_argv = sys.argv
original_candidate_responses = runner._candidate_responses
original_legacy_response = runner.legacy_response
try:
    runner._candidate_responses = lambda requests, _repo: [
        runner._base_response(request, "mock") for request in requests
    ]

    def forbidden_legacy(_request):
        raise AssertionError("candidate mode must not invoke the legacy oracle")

    runner.legacy_response = forbidden_legacy
    sys.argv = [
        str(runner_path),
        "candidate",
        str(public_cases_path),
        "--repo",
        "/missing",
    ]
    output = io.StringIO()
    with contextlib.redirect_stdout(output):
        runner.main()
    assert len(output.getvalue().splitlines()) == len(public_requests)
finally:
    sys.argv = original_argv
    runner._candidate_responses = original_candidate_responses
    runner.legacy_response = original_legacy_response

for manifest_path in (
    problem / "starter" / "Cargo.toml",
    problem / "solution" / "reference" / "Cargo.toml",
    problem / "solution" / "reference" / "crates" / "rustfoil-core" / "Cargo.toml",
    problem / "solution" / "reference" / "crates" / "rustfoil-bl" / "Cargo.toml",
    problem
    / "solution"
    / "reference"
    / "crates"
    / "rustfoil-inviscid"
    / "Cargo.toml",
    problem
    / "solution"
    / "reference"
    / "crates"
    / "rustfoil-testkit"
    / "Cargo.toml",
):
    manifest_text = manifest_path.read_text()
    assert 'serde_json = "1"' not in manifest_text
    assert 'nalgebra = "0.33"' not in manifest_text
    assert 'thiserror = "2.0"' not in manifest_text

locked_packages = {
    package["name"]: package["version"]
    for package in tomllib.loads(
        (problem / "solution" / "reference" / "Cargo.lock").read_text()
    )["package"]
}
assert locked_packages["nalgebra"] == "0.33.2"
assert locked_packages["serde"] == "1.0.228"
assert locked_packages["serde_json"] == "1.0.149"
assert locked_packages["thiserror"] == "2.0.18"
assert "rustfoil-cli" not in locked_packages

scorer = load_scorer()
assert hashlib.sha256(monitor_path.read_bytes()).hexdigest() == scorer._MONITOR_SHA256

for source in (
    'const DATA: &str = include!("/tmp/payload.rs");',
    'const DATA: &str = include! { r#"/tmp/payload.rs"# };',
    'const DATA: &str = include![concat!("/", "tmp/payload.rs")];',
    'const DATA: &str = include /* gap */ ! ("/tmp/payload.rs");',
    "const QUOTE: char = '\"'; include!(\"/tmp/payload.rs\");",
    'fn borrow<\'a>() { include!("/tmp/payload.rs"); }',
    'macro_rules! call { ($m:ident) => { $m!("/tmp/payload.rs") } } call!(include);',
):
    assert scorer._rust_source_uses_include(source), source

for source in (
    '// include!("/tmp/payload.rs")\nfn main() {}',
    '/* outer /* include!{"/tmp/payload.rs"} */ done */ fn main() {}',
    'const NOTE: &str = r##"include!("/tmp/payload.rs")"##;',
    'const NOTE: &[u8] = br#"include!("/tmp/payload.rs")"#;',
    'const NOTE: &str = "include![\\"/tmp/payload.rs\\"]";',
    'const DATA: &[u8] = include_bytes!("data/payload.bin");',
    'const DATA: &str = include_str!("data/payload.txt");',
):
    assert not scorer._rust_source_uses_include(source), source

with tempfile.TemporaryDirectory(prefix="xfoil-include-policy-") as temporary:
    source_root = Path(temporary)
    (source_root / "main.rs").write_text('include! { r#"/tmp/payload.rs"# }\n')
    expect_agent_fault(lambda: scorer._reject_rust_include_macros(source_root))

with tempfile.TemporaryDirectory(prefix="xfoil-smuggle-scrub-") as temporary:
    scratch = Path(temporary)
    fake_tmp = scratch / "tmp"
    fake_tmp.mkdir()
    fake_tmp_link = scratch / "tmp-link"
    fake_tmp_link.symlink_to(fake_tmp, target_is_directory=True)
    fake_output = fake_tmp / "output"
    fake_output.mkdir()
    (fake_output / "repo").mkdir()
    (fake_output / "repo" / "main.rs").write_text("fn main() {}\n")
    (fake_tmp / "payload.rs").write_text("relay")
    fake_workdir = scratch / "workdir"
    fake_workdir.mkdir()
    (fake_workdir / "payload").write_text("relay")
    fake_cargo = fake_tmp / ".cargo"
    fake_cargo.mkdir()
    (fake_cargo / "config.toml").write_text("[build]\nrustc = '/tmp/relay'\n")
    original_geteuid = scorer.os.geteuid
    try:
        scorer.os.geteuid = lambda: 0
        scorer._scrub_agent_smuggle_roots(
            roots=(fake_tmp_link, fake_workdir),
            protected=(fake_output,),
            cargo_config=fake_cargo,
            agent_uid=os.getuid(),
        )
    finally:
        scorer.os.geteuid = original_geteuid
    assert not fake_cargo.exists()
    assert not (fake_tmp / "payload.rs").exists()
    assert (fake_output / "repo" / "main.rs").is_file()
    assert not any(fake_workdir.iterdir())

requests = json.loads((problem / "scorer" / "data" / "hidden_requests.json").read_text())
expected = json.loads((problem / "solution" / "ceiling_responses.json").read_text())
assert {request["case_id"] for request in requests} == set(expected)


def shifted_first(value: Any, delta: float) -> Any:
    shifted = copy.deepcopy(value)
    if isinstance(shifted, list):
        assert shifted
        shifted[0] = shifted_first(shifted[0], delta)
        return shifted
    return float(shifted) + delta


for request in requests:
    operation = request["operation"]
    response = expected[request["case_id"]]
    if response["status"] != "ok":
        assert runner._responses_equal(request, copy.deepcopy(response), response)
        continue
    grader_tolerances = {
        rule.path.removeprefix("observations."): rule.atol
        for rule in scorer._rules_for(request, response)
        if rule.kind in {"numeric", "numeric_array"}
    }
    assert runner._DIFF_ATOLERANCES[operation] == grader_tolerances
    for field, tolerance in grader_tolerances.items():
        within = copy.deepcopy(response)
        within["observations"][field] = shifted_first(
            within["observations"][field],
            tolerance * 0.5,
        )
        assert runner._responses_equal(request, within, response)

        outside = copy.deepcopy(response)
        outside["observations"][field] = shifted_first(
            outside["observations"][field],
            tolerance * 1.5,
        )
        assert not runner._responses_equal(request, outside, response)

for request in requests:
    if request["operation"] != "naca_geometry":
        continue
    coordinates = runner._naca4_coordinates(
        int(request["designation"]),
        int(request["nside"]),
    )
    assert len(coordinates) == 2 * int(request["nside"]) - 1
    assert (
        expected[request["case_id"]]["observations"]["coordinates"] == coordinates
    )
actual = {
    request["case_id"]: {
        "protocol": scorer.PROTOCOL,
        "case_id": request["case_id"],
        "status": "unsupported",
        "observations": {},
        "events": [],
        "output_files": {},
    }
    for request in requests
}


class NumericContext:
    @staticmethod
    def number(value: Any, **_kwargs: Any) -> float:
        return float(value)

    @staticmethod
    def mean(values: list[float], **_kwargs: Any) -> float:
        return sum(values) / len(values)

    @staticmethod
    def ratio(numerator: float, denominator: float, **_kwargs: Any) -> float:
        return numerator / denominator


noop_scores = scorer._suite_scores(NumericContext(), requests, expected, actual)
assert noop_scores == {suite: 0.0 for suite in scorer.SUITE_WEIGHTS}

case_id = requests[0]["case_id"]
valid_response = json.dumps(actual[case_id]).encode()
try:
    scorer._parse_response_stream(
        valid_response + b"\nRUBRIC_SCORE=1\n",
        (case_id,),
    )
except (ValueError, json.JSONDecodeError):
    pass
else:
    raise AssertionError("worker/reward-forgery output was accepted")

artifact = scorer.TASK.artifact
assert artifact.clean_paths == (".git", "target")
assert artifact.max_file_bytes == 16 * 1024 * 1024


def workspace_from_starter(root: Path) -> Path:
    workspace = root / "output"
    shutil.copytree(problem / "starter", workspace / "repo")
    return workspace


with tempfile.TemporaryDirectory(prefix="xfoil-artifact-test-") as temporary:
    root = Path(temporary)
    noop_workspace = workspace_from_starter(root / "noop")
    loaded = artifact.load(noop_workspace)
    assert loaded.file_count >= 2

    oracle_workspace = workspace_from_starter(root / "oracle")
    oracle_source = oracle_workspace / "repo" / "src" / "main.rs"
    oracle_source.write_text(
        oracle_source.read_text() + "\n// std::process::Command oracle delegation\n"
    )
    expect_agent_fault(lambda: artifact.load(oracle_workspace))

    sibling_workspace = workspace_from_starter(root / "sibling")
    (sibling_workspace / "forged-score.txt").write_text("RUBRIC_SCORE=1\n")
    expect_agent_fault(lambda: artifact.load(sibling_workspace))

    symlink_workspace = workspace_from_starter(root / "symlink")
    symlink_source = symlink_workspace / "repo" / "src" / "main.rs"
    symlink_source.unlink()
    symlink_source.symlink_to(problem / "scorer" / "data" / "hidden_requests.json")
    expect_agent_fault(lambda: artifact.load(symlink_workspace))

assert not (problem / "data" / "hidden_requests.json").exists()
public_ids = {
    case["case_id"]
    for case in json.loads((problem / "data" / "public_cases.json").read_text())
}
hidden_ids = {request["case_id"] for request in requests}
assert public_ids.isdisjoint(hidden_ids)

attributes = (repo_root / ".gitattributes").read_text()
assert "examples/xfoil-rust-port/data/xfoil-source.tar.gz binary" in attributes
assert "examples/xfoil-rust-port/data/xfoil-source/**" not in attributes

print("XFOIL task archive, isolation, no-op, oracle, and worker checks passed")
PY

echo "ALL XFOIL HOST TESTS PASSED"
