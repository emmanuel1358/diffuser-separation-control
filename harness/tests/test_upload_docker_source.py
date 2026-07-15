from __future__ import annotations

import importlib.util
import tarfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]


def _load_upload_docker_source():
    script = REPO_ROOT / "scripts" / "upload_docker_source.py"
    spec = importlib.util.spec_from_file_location("upload_docker_source", script)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_build_tarball_includes_extra_files_at_archive_root(tmp_path: Path) -> None:
    module = _load_upload_docker_source()
    repo = tmp_path / "repo"
    problem = repo / "problems" / "toy"
    (problem / "environment").mkdir(parents=True)
    (problem / "environment" / "Dockerfile").write_text("FROM scratch\n")
    (problem / "task.toml").write_text("[task]\nname = 'toy'\n")
    (problem / "solution").mkdir()
    (problem / "solution" / "solve.sh").write_text("#!/usr/bin/env bash\n")
    (problem / ".alignerr").mkdir()
    (problem / ".alignerr" / "build_proof.json").write_text("{}\n")
    (problem / "baselines").mkdir()
    (problem / "baselines" / "baseline.py").write_text("pass\n")
    (problem / "data-generation").mkdir()
    (problem / "data-generation" / "generate.py").write_text("pass\n")
    (problem / "scorer" / "data").mkdir(parents=True)
    (problem / "scorer" / "data" / "truth.json").write_text("{}\n")
    (repo / "grader").mkdir()
    (repo / ".git").mkdir()
    generated = tmp_path / "problems-metadata.json"
    generated.write_text('{"problem_set": {"problems": []}}\n')
    notice = tmp_path / "NOTICE.md"
    notice.write_text("# NOTICE\n\n## Deploy-Time Mounts\n")
    tar_path = tmp_path / "source.tar.gz"

    module.build_tarball(
        str(repo),
        "problems/toy",
        str(tar_path),
        extra_files=[
            (str(generated), "problems-metadata.json"),
            (str(notice), "NOTICE.md"),
        ],
    )

    with tarfile.open(tar_path) as tar:
        names = set(tar.getnames())
    assert "Dockerfile" in names
    assert "problems/toy/task.toml" in names
    assert "problems/toy/solution/solve.sh" in names
    assert "problems/toy/.alignerr/build_proof.json" in names
    assert "problems/toy/baselines/baseline.py" in names
    assert "problems/toy/data-generation/generate.py" in names
    assert "problems/toy/scorer/data/truth.json" in names
    assert "grader" in names
    assert "problems-metadata.json" in names
    assert "NOTICE.md" in names
    assert ".git" not in names
